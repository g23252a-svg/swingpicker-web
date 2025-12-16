# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System v8.0
- v8.0 Upgrade:
  1. 보안: 로그인 시도 횟수 제한 (Rate Limiting) 적용 (Brute Force 방어)
  2. 성능: st.cache_data를 이용한 Native Caching (메모리 효율 및 속도 개선)
  3. 기능: 마이페이지 (닉네임/비밀번호 변경) 추가
  4. 기존: PBKDF2 보안, Gist/Local 이중화 유지
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

# 👇 [추가] 👑 Master Admin 설정 (DB 없이 즉시 로그인)
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

def _hash_password_legacy(password: str, salt: str) -> str:
    """v7.4 이하 구버전 해싱 (SHA256) - 마이그레이션용"""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
def _validate_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email)) if email else False

def _validate_password(pw: str) -> Tuple[bool, str]:
    if len(pw) < 6: return False, "비밀번호는 최소 6자 이상이어야 합니다."
    return True, ""

# ----------------- 회원 관리 로직 -----------------

def register_user(email: str, password: str, nickname: str, invite_code: str = ""):
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
    now_utc = _now_utc_str()
    
    users[email_norm] = {
        "login_id": email_norm,
        "nickname": nickname or email_norm.split("@")[0],
        "role": role,
        "salt": salt,
        "password_hash": pw_hash,
        "created_at": now_utc,
        "last_login": now_utc,
    }
    
    db["users"] = users
    save_user_db(db)
    return True, f"회원가입 완료! 현재 권한: {role}", users[email_norm]

def authenticate_user(email: str, password: str):
    email_norm = _normalize_email(email)

    # 👇 [추가] Master Admin 특수 로그인 처리 (DB 조회 건너뜀)
    if email_norm == MASTER_ADMIN_ID:
        if password == MASTER_ADMIN_PW:
            # 관리자 전용 가상 유저 객체 생성
            admin_user = {
                "login_id": MASTER_ADMIN_ID,
                "nickname": "System Admin",
                "role": "admin",
                "created_at": _now_utc_str(),
                "last_login": _now_utc_str()
            }
            return admin_user, "👑 관리자 로그인 성공"
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
    
    salt = user.get("salt", "")
    pw_hash = user.get("password_hash", "")
    
    # 1. 신규 방식(PBKDF2) 검증
    if _hash_password(password, salt) == pw_hash:
        pass # 성공
        
    # 2. 실패 시 구버전(SHA256) 검증 (마이그레이션)
    elif _hash_password_legacy(password, salt) == pw_hash:
        print(f"[System] Migrating password for {email_norm} to v7.5 security.")
        new_hash = _hash_password(password, salt)
        user["password_hash"] = new_hash
        # 저장은 아래에서 한 번에 처리
    else:
        record_login_fail(email_norm)
        return None, "이메일 또는 비밀번호가 일치하지 않습니다."

    # 로그인 성공
    reset_login_fail(email_norm)
    
    now_str = _now_utc_str()
    user["last_login"] = now_str
    users[email_norm] = user
    
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


# ----------------- UI: 로그인 / 회원가입 박스 -----------------
def render_auth_box(show_debug: bool = False):
    """
    브라우저 탭 전환 / 새로고침 시에도 로그인을 유지하기 위해 CookieManager 사용 (임시 비활성화)
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

        # [Upgrade 3] 마이페이지 (정보 수정)
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

    # ---------------- [상태 2] 비로그인 상태 (로그인 폼) ----------------
    st.subheader("🔐 계정 로그인 / 회원가입")
    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

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
            reg_pw1 = st.text_input("비밀번호", type="password")
            reg_pw2 = st.text_input("비밀번호 확인", type="password")
            reg_code = st.text_input("초대/구독 코드 (선택)", type="password")
            
            st.caption("ℹ️ 가입 시 **상위 5개 추천 종목** 무료 열람 가능")
            submit_reg = st.form_submit_button("회원가입")

        if submit_reg:
            if reg_pw1 != reg_pw2:
                st.error("비밀번호가 서로 일치하지 않습니다.")
            else:
                ok, msg, new_user = register_user(reg_email, reg_pw1, reg_nick, reg_code)
                if ok:
                    st.session_state[CURRENT_USER_KEY] = new_user
                    st.session_state[JUST_REGISTERED_KEY] = True
                    st.success(msg)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)

    return None
