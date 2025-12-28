# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System v8.5
- v8.5 Upgrade:
  1. 보안: 비밀번호 복잡도(영문+숫자) 강제, PBKDF2 적용
  2. 편의: 세션 자동 연장 (Last Login 갱신 최적화)
  3. 관리: 관리자용 DAU(일간 활성 사용자) 통계 기반 마련
"""

import os
import json
import hashlib
import logging
import re
import time
import secrets
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

import requests
import streamlit as st

# ----------------- 로깅 설정 -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auth_user")

# 🔹 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USER_DB_PATH = os.path.join(DATA_DIR, "users_db.json")

CURRENT_USER_KEY = "ldy_current_user"
JUST_REGISTERED_KEY = "just_registered"

# [v9.0] 보안 질문 리스트 (비밀번호 찾기용)
SECURITY_QUESTIONS = [
    "선택하세요...",
    "가장 기억에 남는 여행지는?",
    "어릴 적 살던 동네 이름은?",
    "가장 좋아하는 보물 1호는?",
    "초등학교 담임 선생님 성함은?",
    "나의 좌우명은?",
    "부모님의 고향은 어디인가요?",
]

# ----------------- 설정값 로딩 -----------------
def _get_conf(key: str, default_val: str) -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default_val)

KEY_PRO   = _get_conf("LDY_KEY_PRO",   "220577")
KEY_PRIME = _get_conf("LDY_KEY_PRIME", "577220")
ADMIN_KEY = _get_conf("LDY_ADMIN_KEY", "2022322")

# 👑 Master Admin 설정 (DB 없이 즉시 로그인)
MASTER_ADMIN_ID = "admin"
MASTER_ADMIN_PW = "2022322"

# Gist 설정
GIST_ID_USERS = _get_conf("LDY_GIST_ID", "")
GIST_TOKEN    = _get_conf("LDY_GIST_TOKEN", "")
GIST_ID_SUBS = _get_conf("LDY_GIST_SUBS_ID", GIST_ID_USERS)
GIST_ID_INQ  = _get_conf("LDY_GIST_INQ_ID",  GIST_ID_USERS)


# ----------------- [Upgrade 1] 보안: Rate Limiting -----------------
@st.cache_resource
def get_login_attempts() -> Dict[str, List[Any]]:
    """
    앱이 재시작되기 전까지 유지되는 메모리 내 딕셔너리
    Key: email, Value: [실패횟수, 마지막실패시간_timestamp]
    """
    return {}

def check_rate_limit(email: str, limit: int = 5, lock_min: int = 15) -> Tuple[bool, str]:
    """로그인 시도 횟수 제한 확인"""
    attempts = get_login_attempts()
    if email not in attempts:
        return True, ""
    
    count, last_time = attempts[email]
    if count >= limit:
        # 잠금 시간이 지났는지 확인
        elapsed = time.time() - last_time
        if elapsed < (lock_min * 60):
            remain = int(lock_min - elapsed / 60)
            return False, f"⛔ 비밀번호 오류 횟수 초과 ({limit}회). {remain}분 후에 다시 시도하세요."
        else:
            # 잠금 시간 지났으면 초기화
            attempts[email] = [0, 0.0]
    
    return True, ""

def record_login_fail(email: str):
    """로그인 실패 기록"""
    attempts = get_login_attempts()
    if email not in attempts:
        attempts[email] = [1, time.time()]
    else:
        attempts[email][0] += 1
        attempts[email][1] = time.time()

def reset_login_fail(email: str):
    """로그인 성공 시 실패 기록 초기화"""
    attempts = get_login_attempts()
    if email in attempts:
        del attempts[email]


# ----------------- 유틸 함수 -----------------
def _now_utc_str() -> str:
    return datetime.now(timezone.utc).isoformat()

def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()

def _ensure_data_dir() -> None:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

def _normalize_user_db_structure(db_raw: dict) -> dict:
    if not isinstance(db_raw, dict):
        return {"users": {}}

    raw_users = db_raw.get("users")
    if raw_users is None or not isinstance(raw_users, dict):
        raw_users = db_raw

    normalized_users = {}
    for key, user in raw_users.items():
        if not isinstance(user, dict):
            continue

        email_norm = _normalize_email(key)
        if not email_norm:
            continue

        u = dict(user)
        u["login_id"] = _normalize_email(u.get("login_id", email_norm))
        u.setdefault("role", "free")

        now_utc = _now_utc_str()
        created = u.get("created_at")
        last = u.get("last_login")

        if not created and last:
            created = last
        elif not created:
            created = now_utc

        if not last:
            last = created

        u["created_at"] = created
        u["last_login"] = last

        if not u.get("nickname"):
            u["nickname"] = email_norm.split("@")[0]

        existing = normalized_users.get(email_norm)
        if existing:
            if u["last_login"] > existing.get("last_login", ""):
                normalized_users[email_norm] = u
        else:
            normalized_users[email_norm] = u

    return {"users": normalized_users}

def _load_user_db_local() -> dict:
    _ensure_data_dir()
    if not os.path.exists(USER_DB_PATH):
        return {"users": {}}
    try:
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        db = _normalize_user_db_structure(raw)
        return db
    except Exception:
        return {"users": {}}

def _save_user_db_local(db: dict) -> None:
    _ensure_data_dir()
    try:
        with open(USER_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ----------------- GitHub Gist 연동 유틸 -----------------
GIST_FILE_NAME = "users_db.json"

def _extract_json_from_text(content: str) -> str:
    if not content: return "{}"
    s = content.strip()
    m = re.search(r"[\{\[]", s)
    if not m: return "{}"
    start = m.start()
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    end = s.rfind(close_ch)
    if end == -1 or end <= start:
        return "{}" if open_ch == "{" else "[]"
    return s[start:end + 1]

def _load_user_db_from_gist() -> Optional[dict]:
    if not GIST_ID_USERS or not GIST_TOKEN:
        return None
    try:
        url = f"https://api.github.com/gists/{GIST_ID_USERS}"
        headers = {
            "Authorization": f"token {GIST_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        gist = resp.json()
        files = gist.get("files", {})
        file_obj = files.get(GIST_FILE_NAME)
        if not file_obj:
            return {"users": {}}
        content = (file_obj.get("content", "") or "").strip()
        if not content:
            return {"users": {}}
        json_text = _extract_json_from_text(content)
        raw = json.loads(json_text)
        db = _normalize_user_db_structure(raw)
        return db
    except Exception:
        return None

def _save_user_db_to_gist(db: dict) -> bool:
    if not GIST_ID_USERS or not GIST_TOKEN:
        return False
    try:
        url = f"https://api.github.com/gists/{GIST_ID_USERS}"
        headers = {
            "Authorization": f"token {GIST_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        payload = {
            "files": {
                GIST_FILE_NAME: {
                    "content": json.dumps(db, ensure_ascii=False, indent=2)
                }
            }
        }
        resp = requests.patch(url, headers=headers, data=json.dumps(payload), timeout=10)
        resp.raise_for_status()
        return True
    except Exception:
        return False

def load_json_from_gist_file(gist_id: str, file_name: str, default):
    if not gist_id or not GIST_TOKEN:
        return default
    try:
        url = f"https://api.github.com/gists/{gist_id}"
        headers = {
            "Authorization": f"token {GIST_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return default
        gist = resp.json()
        files = gist.get("files", {})
        file_obj = files.get(file_name)
        if not file_obj:
            return default
        content = (file_obj.get("content", "") or "").strip()
        if not content:
            return default
        json_text = _extract_json_from_text(content)
        return json.loads(json_text)
    except Exception:
        return default

def save_json_to_gist_file(gist_id: str, file_name: str, data: dict) -> bool:
    if not gist_id or not GIST_TOKEN:
        return False
    try:
        url = f"https://api.github.com/gists/{gist_id}"
        headers = {
            "Authorization": f"token {GIST_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        payload = {
            "files": {
                file_name: {
                    "content": json_str
                }
            }
        }
        resp = requests.patch(url, headers=headers, data=json.dumps(payload), timeout=10)
        resp.raise_for_status()
        return True
    except Exception:
        return False

# ----------------- 구독/문의 DB -----------------
SUBSCRIPTIONS_GIST_FILE = "subscriptions_db.json"
INQUIRIES_GIST_FILE     = "inquiries_db.json"
DEFAULT_SUBSCRIPTIONS_DB = {"subs": {}, "updated_at": None}
DEFAULT_INQUIRIES_DB     = {"inquiries": [], "updated_at": None}

def load_subscriptions_db() -> dict:
    data = load_json_from_gist_file(GIST_ID_SUBS, SUBSCRIPTIONS_GIST_FILE, default=DEFAULT_SUBSCRIPTIONS_DB)
    if not isinstance(data, dict): return dict(DEFAULT_SUBSCRIPTIONS_DB)
    if "subs" not in data: data["subs"] = {}
    return data

def save_subscriptions_db(db: dict) -> bool:
    if not isinstance(db, dict): db = dict(DEFAULT_SUBSCRIPTIONS_DB)
    db["updated_at"] = _now_utc_str()
    return save_json_to_gist_file(GIST_ID_SUBS, SUBSCRIPTIONS_GIST_FILE, db)

def load_inquiries_db() -> dict:
    data = load_json_from_gist_file(GIST_ID_INQ, INQUIRIES_GIST_FILE, default=DEFAULT_INQUIRIES_DB)
    if not isinstance(data, dict): return dict(DEFAULT_INQUIRIES_DB)
    if "inquiries" not in data: data["inquiries"] = []
    return data

def save_inquiries_db(db: dict) -> bool:
    if not isinstance(db, dict): db = dict(DEFAULT_INQUIRIES_DB)
    db["updated_at"] = _now_utc_str()
    return save_json_to_gist_file(GIST_ID_INQ, INQUIRIES_GIST_FILE, db)

def load_inquiry_items() -> list:
    return load_inquiries_db().get("inquiries", [])

def save_inquiry_items(items: list) -> bool:
    db = load_inquiries_db()
    db["inquiries"] = list(items) if isinstance(items, list) else []
    return save_inquiries_db(db)

# ----------------- [Upgrade 2] 통합 DB: Native Caching -----------------

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_user_db_cached() -> dict:
    """
    Gist/Local 데이터를 가져오고 60초간 Streamlit 캐시에 보관합니다.
    """
    # 1. Gist 시도
    db = _load_user_db_from_gist()
    if db:
        return db
    # 2. 실패 시 로컬 로드
    return _load_user_db_local()

def load_user_db() -> dict:
    """캐시된 DB 로드"""
    return _fetch_user_db_cached()

def save_user_db(db: dict) -> None:
    """DB 저장 후 캐시 무효화"""
    db = _normalize_user_db_structure(db)
    
    # 1. 로컬 & Gist 저장
    _save_user_db_local(db)
    if GIST_ID_USERS and GIST_TOKEN:
        _save_user_db_to_gist(db)
    
    # 2. [핵심] 캐시 초기화 (다음 load시 새로 받아옴)
    _fetch_user_db_cached.clear()


# ----------------- 보안 유틸 (v7.5 유지) -----------------
def _create_salt() -> str:
    """암호학적으로 안전한 32바이트 Salt 생성"""
    return secrets.token_hex(16)

def _hash_password(password: str, salt: str) -> str:
    """v7.5: PBKDF2-HMAC-SHA256 적용 (100,000 iterations)"""
    return hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()

# ✅ [v9.0] 보안 질문 답변 해싱 (소문자 변환/공백 제거 후 해싱)
def _hash_answer(answer: str, salt: str) -> str:
    norm_ans = answer.strip().lower()
    return _hash_password(norm_ans, salt)

# ✅ [v9.0] 비밀번호 찾기: 정보 검증
def verify_recovery_info(email: str, question_idx: int, answer: str) -> bool:
    email_norm = _normalize_email(email)
    db = load_user_db()
    users = db.get("users", {})
    user = users.get(email_norm)
    
    if not user: return False
    
    saved_q_idx = user.get("security_q_idx", 0)
    saved_a_hash = user.get("security_a_hash", "")
    salt = user.get("salt", "")
    
    # 1. 질문 유형 일치 여부 확인
    if saved_q_idx != question_idx:
        return False
        
    # 2. 답변 해시 일치 여부 확인
    input_hash = _hash_answer(answer, salt)
    return input_hash == saved_a_hash

# ✅ [v9.0] 비밀번호 찾기: 새 비밀번호 강제 설정
def reset_password_with_recovery(email: str, new_pw: str) -> Tuple[bool, str]:
    email_norm = _normalize_email(email)
    ok, msg = _validate_password(new_pw)
    if not ok: return False, msg
    
    db = load_user_db()
    users = db.get("users", {})
    if email_norm not in users: return False, "유저 정보를 찾을 수 없습니다."
    
    user = users[email_norm]
    # Salt 재발급 (보안 강화)
    new_salt = _create_salt() 
    user["salt"] = new_salt
    user["password_hash"] = _hash_password(new_pw, new_salt)
    
    # Salt가 바뀌었으므로 기존 보안 답변 해시는 무효화됨 -> 초기화
    user["security_a_hash"] = "" 
    user["security_q_idx"] = 0
    
    db["users"] = users
    save_user_db(db)
    return True, "비밀번호가 재설정되었습니다. (보안 질문이 초기화되었습니다. 로그인 후 다시 설정해주세요.)"

def _hash_password_legacy(password: str, salt: str) -> str:
    """v7.4 이하 구버전 해싱 (SHA256) - 마이그레이션용"""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email)) if email else False

def _validate_password(pw: str) -> Tuple[bool, str]:
    if len(pw) < 6: return False, "비밀번호는 6자 이상이어야 합니다."
    # ✅ [v8.5 Upgrade] 비밀번호 복잡도 체크 (영문 + 숫자)
    if not re.search(r"[a-zA-Z]", pw) or not re.search(r"\d", pw):
        return False, "비밀번호는 영문과 숫자를 모두 포함해야 합니다."
    return True, ""

# ----------------- 회원 관리 로직 -----------------

# ✅ (수정된 코드: 줄바꿈하여 주석과 코드를 분리했습니다)
def register_user(email: str, password: str, nickname: str, 
                  q_idx: int, q_answer: str,  # 👈 [v9.0] 추가된 인자
                  invite_code: str = ""):
    email_norm = _normalize_email(email)
    # 👇 [추가] 'admin' 아이디 등록 시도 차단
    if email_norm == MASTER_ADMIN_ID:
         return False, "해당 ID는 예약어로 사용할 수 없습니다.", None
    if not email_norm or not password:
        return False, "이메일 / 비밀번호를 입력해 주세요.", None
    if not _validate_email(email_norm):
        return False, "이메일 형식이 올바르지 않습니다.", None
    ok_pw, pw_msg = _validate_password(password)
    if not ok_pw:
        return False, pw_msg, None

    # [v9.0] 보안 답변 필수 체크
    if not q_answer or not q_answer.strip():
        return False, "비밀번호 찾기에 사용할 보안 답변을 입력해 주세요.", None
    
    db = load_user_db()
    users = db.get("users", {})
    if email_norm in users:
        return False, "이미 존재하는 이메일입니다.", None
    
    role = "free"
    invite_code = (invite_code or "").strip()
    if invite_code == ADMIN_KEY: role = "admin"
    elif invite_code == KEY_PRIME: role = "prime"
    elif invite_code == KEY_PRO: role = "pro"
    
    salt = _create_salt()
    pw_hash = _hash_password(password, salt)

    # [v9.0] 답변 해싱
    ans_hash = _hash_answer(q_answer, salt)

    now_utc = _now_utc_str()
    
    users[email_norm] = {
        "login_id": email_norm,
        "nickname": nickname or email_norm.split("@")[0],
        "role": role,
        "salt": salt,
        "password_hash": pw_hash,
        "created_at": now_utc,
        "last_login": now_utc,
        "is_banned": False,
        "security_q_idx": int(q_idx),
        "security_a_hash": ans_hash,
        "session_token": secrets.token_urlsafe(32), # ✅ 자동 로그인 및 세션 검증용 고유 토큰
    }
    
    db["users"] = users
    save_user_db(db)
    return True, f"회원가입 완료! 현재 권한: {role}", users[email_norm]

def authenticate_user(email: str, password: str):
    email_norm = _normalize_email(email)

    # 👑 Master Admin 특수 로그인 처리 (DB 조회 건너뜀)
    if email_norm == MASTER_ADMIN_ID:
        if password == MASTER_ADMIN_PW:
            return {
                "login_id": MASTER_ADMIN_ID,
                "nickname": "System Admin",
                "role": "admin",
                "created_at": _now_utc_str(),
                "last_login": _now_utc_str()
            }, "👑 관리자 로그인 성공"
        else:
            return None, "관리자 비밀번호가 일치하지 않습니다."
    
    # [보안] Rate Limit 체크
    is_allowed, limit_msg = check_rate_limit(email_norm)
    if not is_allowed:
        return None, limit_msg

    db = load_user_db()
    users = db.get("users", {})
    user = users.get(email_norm)
    
    if not user:
        record_login_fail(email_norm)
        return None, "이메일 또는 비밀번호가 일치하지 않습니다."

    # ✅ [v9.0] 차단 여부 확인 (비밀번호 확인 전에 수행)
    if user.get("is_banned", False):
        return None, "🚫 관리자에 의해 이용이 정지된 계정입니다."
    
    salt = user.get("salt", "")
    pw_hash = user.get("password_hash", "")
    
    # 비밀번호 검증 및 마이그레이션 플래그
    password_migrated = False

    # 1. 신규 방식(PBKDF2) 검증
    if _hash_password(password, salt) == pw_hash:
        pass 
    # 2. 구버전(SHA256) 검증 -> 신규 해시로 업데이트 (메모리 상)
    elif _hash_password_legacy(password, salt) == pw_hash:
        user["password_hash"] = _hash_password(password, salt)
        password_migrated = True
    else:
        record_login_fail(email_norm)
        return None, "이메일 또는 비밀번호가 일치하지 않습니다."

    # 로그인 성공
    reset_login_fail(email_norm)

    # ✅ v10.0 세션 토큰 새로 발급 (기존 세션 무효화 및 신규 세션 생성)
    user["session_token"] = secrets.token_urlsafe(32)
    
    # ✅ [v8.5 최적화] 로그인 시간 갱신 로직 (DB 부하 감소)
    now_str = _now_utc_str()
    last_login_str = user.get("last_login", "")
    should_save = True # 기본은 저장
    
    if last_login_str:
        try:
            # 마지막 로그인 시간 파싱 (ISO 8601)
            last_dt = datetime.fromisoformat(last_login_str.replace("Z", "+00:00"))
            # 현재 시간과 차이 계산
            diff_seconds = (datetime.now(timezone.utc) - last_dt).total_seconds()
            
            # 1시간(3600초) 이내 재접속이면 DB 저장 건너뜀 (메모리만 갱신)
            if diff_seconds < 3600 and not password_migrated:
                should_save = False
        except Exception:
            pass # 파싱 실패 시 안전하게 저장
            
    # 메모리 상의 유저 정보는 항상 최신화 (세션용)
    user["last_login"] = _now_utc_str()
    users[email_norm] = user
    
    # 실제 변경사항(비번 업데이트 or 오랜만의 로그인)이 있을 때만 DB 저장
    if should_save:
        db["users"] = users
        save_user_db(db)
    
    return user, "로그인 성공"

def get_user():
    return st.session_state.get(CURRENT_USER_KEY)

def list_users():
    db = load_user_db()
    users = db.get("users", {})
    return [users[k] for k in sorted(users.keys())]

def update_user_role(email: str, new_role: str, acting_admin_email: Optional[str] = None) -> bool:
    email_norm = _normalize_email(email)
    if acting_admin_email:
        acting_admin_email = _normalize_email(acting_admin_email)
    db = load_user_db()
    users = db.get("users", {})
    if email_norm not in users: return False
    
    if acting_admin_email == email_norm:
        current_role = users[email_norm].get("role", "free")
        if current_role == "admin" and new_role != "admin":
            return False
            
    users[email_norm]["role"] = new_role
    db["users"] = users
    save_user_db(db)
    return True

# ✅ [v9.0] 관리자 기능: 유저 차단 토글
def toggle_user_ban(email: str, acting_admin: str) -> Tuple[bool, str]:
    email_norm = _normalize_email(email)
    db = load_user_db()
    users = db.get("users", {})
    
    if email_norm not in users:
        return False, "유저를 찾을 수 없습니다."
    
    if email_norm == _normalize_email(acting_admin):
        return False, "관리자 자신을 차단할 수 없습니다."
        
    current_status = users[email_norm].get("is_banned", False)
    users[email_norm]["is_banned"] = not current_status # 상태 반전
    
    db["users"] = users
    save_user_db(db)
    
    action = "해제" if current_status else "차단(Ban)"
    return True, f"사용자 {email_norm} {action} 처리 완료."

# ----------------- [Upgrade 3] 마이페이지 기능 -----------------
def update_user_profile(email: str, new_nickname: str = None, new_password: str = None) -> Tuple[bool, str]:
    email_norm = _normalize_email(email)
    # 👇 [추가] 'admin' 계정은 프로필 수정 불가
    if email_norm == MASTER_ADMIN_ID:
        return False, "시스템 관리자(admin) 계정은 프로필을 변경할 수 없습니다."  
    db = load_user_db()
    users = db.get("users", {})
    
    if email_norm not in users:
        return False, "사용자 정보를 찾을 수 없습니다."
    
    user = users[email_norm]
    changed = False
    
    # 닉네임 변경
    if new_nickname and new_nickname != user.get("nickname"):
        user["nickname"] = new_nickname
        changed = True
        
    # 비밀번호 변경
    if new_password:
        ok_pw, msg = _validate_password(new_password)
        if not ok_pw: return False, msg
        
        # 새 Salt 생성 및 PBKDF2 해싱
        new_salt = _create_salt()
        new_hash = _hash_password(new_password, new_salt)
        user["salt"] = new_salt
        user["password_hash"] = new_hash
        changed = True
    
    if changed:
        user["last_login"] = _now_utc_str() # 정보 변경 시점 기록
        users[email_norm] = user
        db["users"] = users
        save_user_db(db)
        
        # 세션 정보도 최신화
        if st.session_state.get(CURRENT_USER_KEY):
             st.session_state[CURRENT_USER_KEY] = user
             
        return True, "정보가 성공적으로 수정되었습니다."
    
    return True, "변경할 내용이 없습니다."


# ----------------- UI: 로그인 / 회원가입 / 비밀번호 찾기 박스 -----------------
def render_auth_box(show_debug: bool = False):
    """
    사용자 인증 UI (로그인 / 회원가입 / 비밀번호 찾기)
    """
    cookie_user_email = None

    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None

    if JUST_REGISTERED_KEY not in st.session_state:
        st.session_state[JUST_REGISTERED_KEY] = False

    # [자동 로그인 복구 시도 - 비활성 상태]
    if st.session_state[CURRENT_USER_KEY] is None and cookie_user_email:
        db = load_user_db()
        users = db.get("users", {})
        saved_user = users.get(cookie_user_email)
        if saved_user:
            st.session_state[CURRENT_USER_KEY] = saved_user

    # 현재 세션 사용자 가져오기
    user = get_user()

    # -------------------- [v10.1 긴급 수정] 실시간 보안 & 세션 검증 --------------------
    if user:
        # 👑 Master Admin은 모든 검증을 건너뛰고 항상 통과 (복구용)
        if user.get('login_id') == MASTER_ADMIN_ID:
            pass 
        else:
            # 일반 유저만 DB 대조 및 차단 여부 확인
            db = load_user_db()
            current_db_user = db.get("users", {}).get(user['login_id'])
            
            # 1. 관리자가 유저를 차단(is_banned)했는지 즉시 확인
            # (DB에 유저 정보가 없거나, is_banned가 True면 로그아웃)
            if not current_db_user or current_db_user.get("is_banned", False):
                st.session_state[CURRENT_USER_KEY] = None
                st.error("🚨 보안 정책에 의해 이용이 정지되었습니다. 관리자에게 문의하세요.")
                time.sleep(2)
                st.rerun()

            # 2. 권한 변경(Role) 실시간 반영
            if current_db_user.get("role") != user.get("role"):
                user["role"] = current_db_user.get("role")
                st.session_state[CURRENT_USER_KEY] = user
                st.rerun()
                
            # 3. 세션 토큰 무결성 체크 (중복 로그인 방지용)
            # DB에 토큰이 없으면(구버전 유저) 통과, 토큰이 있는데 다르면 로그아웃
            db_token = current_db_user.get("session_token")
            user_token = user.get("session_token")
            
            if db_token and user_token and db_token != user_token:
                st.session_state[CURRENT_USER_KEY] = None
                st.warning("⚠️ 다른 기기에서 로그인하여 현재 세션이 종료되었습니다.")
                time.sleep(2)
                st.rerun()
    # ------------------------------------------------------------------------------

    # ---------------- [상태 1] 로그인 된 상태 ----------------
    if user:
        # 상단 정보 표시
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ **{user['nickname']}**님 환영합니다!")
        with col2:
            if st.button("로그아웃", key="btn_logout", type="secondary"):
                st.session_state[CURRENT_USER_KEY] = None
                st.session_state[JUST_REGISTERED_KEY] = False
                st.toast("로그아웃 되었습니다.", icon="👋")
                time.sleep(0.5) 
                st.rerun()

        # [Upgrade 3] 마이페이지 (정보 수정) - Admin은 수정 불가 예외 처리됨
        with st.expander(f"⚙️ 내 정보 수정 (권한: {user.get('role', 'free')})", expanded=False):
            st.info(f"가입일: {user.get('created_at', '')[:10]}")
            
            with st.form("profile_update_form"):
                new_nick = st.text_input("닉네임 변경", value=user['nickname'])
                new_pw = st.text_input("새 비밀번호 (변경 시에만 입력)", type="password", help="변경하지 않으려면 비워두세요.")
                
                btn_update = st.form_submit_button("정보 수정 적용")
                
                if btn_update:
                    ok, msg = update_user_profile(user['login_id'], new_nick, new_pw)
                    if ok: 
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
        
        return user

    # ---------------- [상태 2] 비로그인 상태 ----------------
    # (이하 기존 로그인/회원가입 탭 코드 그대로 유지)
    st.subheader("🔐 계정 로그인 / 회원가입")
    
    tab_login, tab_signup, tab_find = st.tabs(["로그인", "회원가입", "비밀번호 찾기"])

    # 1) 로그인 탭
    with tab_login:
        with st.form(key="login_form"):
            login_email = st.text_input("이메일")
            login_pw = st.text_input("비밀번호", type="password")
            remember_me = st.checkbox("로그인 상태 유지 (현재 점검중)", value=False, disabled=True)
            submit_login = st.form_submit_button("로그인", type="primary")
        
        if submit_login:
            user_obj, msg = authenticate_user(login_email, login_pw)
            if user_obj is None:
                st.error(msg)
            else:
                st.session_state[CURRENT_USER_KEY] = user_obj
                st.session_state[JUST_REGISTERED_KEY] = False
                st.toast(f"{user_obj['nickname']}님 환영합니다!", icon="🎉")
                time.sleep(0.5) 
                st.rerun()

    # 2) 회원가입 탭
    with tab_signup:
        with st.form(key="signup_form"):
            reg_email = st.text_input("이메일")
            reg_nick = st.text_input("닉네임 (선택)")
            reg_pw1 = st.text_input("비밀번호 (영문+숫자 6자 이상)", type="password")
            reg_pw2 = st.text_input("비밀번호 확인", type="password")
            
            st.markdown("---")
            st.caption("🔒 비밀번호 분실 시 찾기 위한 질문입니다.")
            q_idx = st.selectbox("보안 질문 선택", range(len(SECURITY_QUESTIONS)), format_func=lambda x: SECURITY_QUESTIONS[x])
            q_answer = st.text_input("보안 질문 답변")
            st.markdown("---")
            
            reg_code = st.text_input("초대/구독 코드 (선택)", type="password")
            
            st.caption("ℹ️ 가입 시 **상위 5개 추천 종목** 무료 열람 가능")
            submit_reg = st.form_submit_button("회원가입")

        if submit_reg:
            if reg_pw1 != reg_pw2:
                st.error("비밀번호가 서로 일치하지 않습니다.")
            elif q_idx == 0:
                st.error("보안 질문을 선택해 주세요.")
            elif not q_answer.strip():
                st.error("보안 질문 답변을 입력해 주세요.")
            else:
                ok, msg, new_user = register_user(reg_email, reg_pw1, reg_nick, q_idx, q_answer, reg_code)
                if ok:
                    st.session_state[CURRENT_USER_KEY] = new_user
                    st.session_state[JUST_REGISTERED_KEY] = True
                    st.success(msg)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)

    # 3) 비밀번호 찾기 탭
    with tab_find:
        st.markdown("##### 🔑 비밀번호 재설정")
        
        if "rec_step" not in st.session_state:
            st.session_state["rec_step"] = 1
            st.session_state["rec_email"] = ""
            st.session_state["rec_q_idx"] = 0

        if st.session_state["rec_step"] == 1:
            with st.form("find_pw_step1"):
                email_input = st.text_input("가입한 이메일 입력", placeholder="example@email.com")
                btn_check = st.form_submit_button("확인")
            
            if btn_check:
                db = load_user_db()
                users = db.get("users", {})
                u_norm = _normalize_email(email_input)
                
                if u_norm in users:
                    user_data = users[u_norm]
                    q_idx = user_data.get("security_q_idx", 0)
                    if q_idx > 0 and q_idx < len(SECURITY_QUESTIONS):
                        st.session_state["rec_email"] = u_norm
                        st.session_state["rec_q_idx"] = q_idx
                        st.session_state["rec_step"] = 2
                        st.rerun()
                    else:
                        st.error("이 계정은 보안 질문이 설정되지 않았습니다. 관리자에게 문의하세요.")
                else:
                    st.error("등록되지 않은 이메일입니다.")

        elif st.session_state["rec_step"] == 2:
            target_email = st.session_state["rec_email"]
            q_idx = st.session_state["rec_q_idx"]
            q_text = SECURITY_QUESTIONS[q_idx]

            st.info(f"대상 계정: **{target_email}**")
            st.warning(f"❓ 보안 질문: **{q_text}**")

            with st.form("find_pw_step2"):
                ans_input = st.text_input("답변 입력")
                new_pw_input = st.text_input("새로운 비밀번호 (영문+숫자 6자 이상)", type="password")
                btn_reset = st.form_submit_button("비밀번호 변경")

            if btn_reset:
                if verify_recovery_info(target_email, q_idx, ans_input):
                    ok, msg = reset_password_with_recovery(target_email, new_pw_input)
                    if ok:
                        st.success("✅ " + msg)
                        st.session_state["rec_step"] = 1
                        st.session_state["rec_email"] = ""
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("❌ 보안 질문 답변이 틀렸습니다.")

            if st.button("뒤로가기"):
                st.session_state["rec_step"] = 1
                st.session_state["rec_email"] = ""
                st.rerun()

    return None
