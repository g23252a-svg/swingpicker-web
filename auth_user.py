# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System v12.0 (Security Hardening & Audit)
- v12.0:
  1. 감사 로그(Audit Log) 도입: 중요 보안 이벤트(로그인 실패, 권한 변경 등) 기록
  2. 세션 무결성 강화: 중복 로그인 시 즉시 차단
  3. 무차별 대입 방어: 로그인 실패 시 랜덤 지연 적용
"""

import os
import json
import hashlib
import logging
import re
import time
import secrets
import random
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

# 보안 질문 리스트
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

# 👑 Master Admin 설정
MASTER_ADMIN_ID = "admin"
MASTER_ADMIN_PW = "2022322"

# Gist 설정
GIST_ID_USERS = _get_conf("LDY_GIST_ID", "")
GIST_TOKEN    = _get_conf("LDY_GIST_TOKEN", "")
GIST_ID_SUBS = _get_conf("LDY_GIST_SUBS_ID", GIST_ID_USERS)
GIST_ID_INQ  = _get_conf("LDY_GIST_INQ_ID",  GIST_ID_USERS)


# ----------------- [Upgrade 1] 보안: Rate Limiting & Audit -----------------
@st.cache_resource
def get_login_attempts() -> Dict[str, List[Any]]:
    return {}

def check_rate_limit(email: str, limit: int = 5, lock_min: int = 15) -> Tuple[bool, str]:
    attempts = get_login_attempts()
    if email not in attempts:
        return True, ""
    
    count, last_time = attempts[email]
    if count >= limit:
        elapsed = time.time() - last_time
        if elapsed < (lock_min * 60):
            remain = int(lock_min - elapsed / 60)
            return False, f"⛔ 비밀번호 오류 횟수 초과 ({limit}회). {remain}분 후에 다시 시도하세요."
        else:
            attempts[email] = [0, 0.0]
    
    return True, ""

def record_login_fail(email: str):
    attempts = get_login_attempts()
    if email not in attempts:
        attempts[email] = [1, time.time()]
    else:
        attempts[email][0] += 1
        attempts[email][1] = time.time()
    
    # 🛡️ 무차별 대입 방어: 실패 시 랜덤 지연 (0.5 ~ 2.0초)
    time.sleep(random.uniform(0.5, 2.0))
    logger.warning(f"🔒 Login Failed: {email} (Count: {attempts[email][0]})")

def reset_login_fail(email: str):
    attempts = get_login_attempts()
    if email in attempts:
        del attempts[email]

def log_audit_event(event_type: str, actor: str, target: str, detail: str = ""):
    """중요 보안 이벤트를 로그에 남김 (추후 DB 저장 확장 가능)"""
    logger.info(f"🛡️ [AUDIT] [{event_type}] Actor:{actor} -> Target:{target} | {detail}")


# ----------------- 유틸 함수 -----------------
def _now_utc_str() -> str:
    return datetime.now(timezone.utc).isoformat()

def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()

def _ensure_data_dir() -> None:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

def _normalize_user_db_structure(db_raw: dict) -> dict:
    if not isinstance(db_raw, dict): return {"users": {}}
    raw_users = db_raw.get("users", db_raw)
    if not isinstance(raw_users, dict): raw_users = {}

    normalized_users = {}
    for key, user in raw_users.items():
        if not isinstance(user, dict): continue
        email_norm = _normalize_email(key)
        if not email_norm: continue

        u = dict(user)
        u["login_id"] = _normalize_email(u.get("login_id", email_norm))
        u.setdefault("role", "free")
        
        # 날짜 필드 보정
        now = _now_utc_str()
        if not u.get("created_at"): u["created_at"] = now
        if not u.get("last_login"): u["last_login"] = now
        if not u.get("nickname"): u["nickname"] = email_norm.split("@")[0]

        normalized_users[email_norm] = u

    return {"users": normalized_users}

def _load_user_db_local() -> dict:
    _ensure_data_dir()
    if not os.path.exists(USER_DB_PATH): return {"users": {}}
    try:
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            return _normalize_user_db_structure(json.load(f))
    except: return {"users": {}}

def _save_user_db_local(db: dict) -> None:
    _ensure_data_dir()
    try:
        with open(USER_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except: pass

# ----------------- GitHub Gist 연동 유틸 -----------------
GIST_FILE_NAME = "users_db.json"

def _extract_json_from_text(content: str) -> str:
    if not content: return "{}"
    s = content.strip()
    m = re.search(r"[\{\[]", s)
    if not m: return "{}"
    start = m.start()
    return s[start:]

def _load_user_db_from_gist() -> Optional[dict]:
    if not GIST_ID_USERS or not GIST_TOKEN: return None
    try:
        url = f"https://api.github.com/gists/{GIST_ID_USERS}"
        headers = {"Authorization": f"token {GIST_TOKEN}"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200: return None
        
        content = resp.json().get("files", {}).get(GIST_FILE_NAME, {}).get("content", "")
        return _normalize_user_db_structure(json.loads(_extract_json_from_text(content)))
    except: return None

def _save_user_db_to_gist(db: dict) -> bool:
    if not GIST_ID_USERS or not GIST_TOKEN: return False
    try:
        url = f"https://api.github.com/gists/{GIST_ID_USERS}"
        headers = {"Authorization": f"token {GIST_TOKEN}"}
        payload = {"files": {GIST_FILE_NAME: {"content": json.dumps(db, ensure_ascii=False, indent=2)}}}
        requests.patch(url, headers=headers, json=payload, timeout=10)
        return True
    except: return False

def load_json_from_gist_file(gist_id: str, file_name: str, default):
    if not gist_id or not GIST_TOKEN: return default
    try:
        url = f"https://api.github.com/gists/{gist_id}"
        headers = {"Authorization": f"token {GIST_TOKEN}"}
        resp = requests.get(url, headers=headers, timeout=10)
        content = resp.json().get("files", {}).get(file_name, {}).get("content", "")
        return json.loads(_extract_json_from_text(content)) if content else default
    except: return default

def save_json_to_gist_file(gist_id: str, file_name: str, data: dict) -> bool:
    if not gist_id or not GIST_TOKEN: return False
    try:
        url = f"https://api.github.com/gists/{gist_id}"
        headers = {"Authorization": f"token {GIST_TOKEN}"}
        payload = {"files": {file_name: {"content": json.dumps(data, ensure_ascii=False, indent=2)}}}
        requests.patch(url, headers=headers, json=payload, timeout=10)
        return True
    except: return False

# ----------------- 구독/문의 DB -----------------
SUBSCRIPTIONS_GIST_FILE = "subscriptions_db.json"
INQUIRIES_GIST_FILE     = "inquiries_db.json"
DEFAULT_SUBSCRIPTIONS_DB = {"subs": {}, "updated_at": None}
DEFAULT_INQUIRIES_DB     = {"inquiries": [], "updated_at": None}

def load_subscriptions_db() -> dict:
    return load_json_from_gist_file(GIST_ID_SUBS, SUBSCRIPTIONS_GIST_FILE, DEFAULT_SUBSCRIPTIONS_DB)

def save_subscriptions_db(db: dict) -> bool:
    db["updated_at"] = _now_utc_str()
    return save_json_to_gist_file(GIST_ID_SUBS, SUBSCRIPTIONS_GIST_FILE, db)

def load_inquiries_db() -> dict:
    return load_json_from_gist_file(GIST_ID_INQ, INQUIRIES_GIST_FILE, DEFAULT_INQUIRIES_DB)

def save_inquiries_db(db: dict) -> bool:
    db["updated_at"] = _now_utc_str()
    return save_json_to_gist_file(GIST_ID_INQ, INQUIRIES_GIST_FILE, db)

def load_inquiry_items() -> list:
    return load_inquiries_db().get("inquiries", [])

def save_inquiry_items(items: list) -> bool:
    db = load_inquiries_db()
    db["inquiries"] = items
    return save_inquiries_db(db)

# ----------------- [Upgrade 2] 통합 DB: Native Caching -----------------

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_user_db_cached() -> dict:
    db = _load_user_db_from_gist()
    return db if db else _load_user_db_local()

def load_user_db() -> dict:
    return _fetch_user_db_cached()

def save_user_db(db: dict) -> None:
    db = _normalize_user_db_structure(db)
    _save_user_db_local(db)
    if GIST_ID_USERS and GIST_TOKEN:
        _save_user_db_to_gist(db)
    _fetch_user_db_cached.clear()


# ----------------- 보안 유틸 (v7.5 유지) -----------------
def _create_salt() -> str:
    return secrets.token_hex(16)

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()

def _hash_answer(answer: str, salt: str) -> str:
    return _hash_password(answer.strip().lower(), salt)

def verify_recovery_info(email: str, question_idx: int, answer: str) -> bool:
    email_norm = _normalize_email(email)
    db = load_user_db()
    user = db.get("users", {}).get(email_norm)
    if not user: return False
    
    if user.get("security_q_idx", 0) != question_idx: return False
    return _hash_answer(answer, user.get("salt", "")) == user.get("security_a_hash", "")

def reset_password_with_recovery(email: str, new_pw: str) -> Tuple[bool, str]:
    email_norm = _normalize_email(email)
    ok, msg = _validate_password(new_pw)
    if not ok: return False, msg
    
    db = load_user_db()
    user = db.get("users", {}).get(email_norm)
    if not user: return False, "유저 정보를 찾을 수 없습니다."
    
    new_salt = _create_salt()
    user["salt"] = new_salt
    user["password_hash"] = _hash_password(new_pw, new_salt)
    user["security_a_hash"] = "" 
    user["security_q_idx"] = 0
    
    log_audit_event("PW_RESET", email_norm, email_norm, "Recovery used")
    save_user_db(db)
    return True, "비밀번호 재설정 완료."

def _hash_password_legacy(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _validate_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email)) if email else False

def _validate_password(pw: str) -> Tuple[bool, str]:
    if len(pw) < 6: return False, "비밀번호는 6자 이상이어야 합니다."
    if not re.search(r"[a-zA-Z]", pw) or not re.search(r"\d", pw):
        return False, "비밀번호는 영문과 숫자를 모두 포함해야 합니다."
    return True, ""

# ----------------- 회원 관리 로직 -----------------

def register_user(email: str, password: str, nickname: str, q_idx: int, q_answer: str, invite_code: str = ""):
    email_norm = _normalize_email(email)
    if email_norm == MASTER_ADMIN_ID: return False, "사용할 수 없는 ID입니다.", None
    if not _validate_email(email_norm): return False, "이메일 형식이 올바르지 않습니다.", None
    
    ok, msg = _validate_password(password)
    if not ok: return False, msg, None
    if not q_answer.strip(): return False, "보안 질문 답변을 입력해 주세요.", None
    
    db = load_user_db()
    users = db.get("users", {})
    if email_norm in users: return False, "이미 존재하는 이메일입니다.", None
    
    role = "free"
    if invite_code == ADMIN_KEY: role = "admin"
    elif invite_code == KEY_PRIME: role = "prime"
    elif invite_code == KEY_PRO: role = "pro"
    
    salt = _create_salt()
    now_utc = _now_utc_str()
    
    users[email_norm] = {
        "login_id": email_norm,
        "nickname": nickname or email_norm.split("@")[0],
        "role": role,
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "created_at": now_utc,
        "last_login": now_utc,
        "is_banned": False,
        "security_q_idx": int(q_idx),
        "security_a_hash": _hash_answer(q_answer, salt),
        "session_token": secrets.token_urlsafe(32),
    }
    
    log_audit_event("REGISTER", email_norm, email_norm, f"Role: {role}")
    save_user_db(db)
    return True, f"가입 완료! 권한: {role}", users[email_norm]

def authenticate_user(email: str, password: str):
    email_norm = _normalize_email(email)

    if email_norm == MASTER_ADMIN_ID:
        if password == MASTER_ADMIN_PW:
            return {"login_id": MASTER_ADMIN_ID, "nickname": "Admin", "role": "admin"}, "관리자 로그인"
        return None, "비밀번호 불일치"
    
    is_allowed, msg = check_rate_limit(email_norm)
    if not is_allowed: return None, msg

    db = load_user_db()
    user = db.get("users", {}).get(email_norm)
    
    if not user:
        record_login_fail(email_norm)
        return None, "계정 정보가 없습니다."

    if user.get("is_banned", False):
        return None, "🚫 정지된 계정입니다."
    
    salt = user.get("salt", "")
    pw_hash = user.get("password_hash", "")
    
    valid = False
    if _hash_password(password, salt) == pw_hash:
        valid = True
    elif _hash_password_legacy(password, salt) == pw_hash:
        user["password_hash"] = _hash_password(password, salt)
        valid = True

    if not valid:
        record_login_fail(email_norm)
        return None, "비밀번호가 일치하지 않습니다."

    reset_login_fail(email_norm)
    user["session_token"] = secrets.token_urlsafe(32)
    user["last_login"] = _now_utc_str()
    
    log_audit_event("LOGIN", email_norm, email_norm)
    save_user_db(db)
    return user, "로그인 성공"

def get_user():
    return st.session_state.get(CURRENT_USER_KEY)

def list_users():
    db = load_user_db()
    users = db.get("users", {})
    return [users[k] for k in sorted(users.keys())]

def update_user_role(email: str, new_role: str, acting_admin: str = "system") -> bool:
    email_norm = _normalize_email(email)
    db = load_user_db()
    users = db.get("users", {})
    if email_norm not in users: return False
    
    users[email_norm]["role"] = new_role
    log_audit_event("ROLE_CHANGE", acting_admin, email_norm, f"New Role: {new_role}")
    save_user_db(db)
    return True

def toggle_user_ban(email: str, acting_admin: str) -> Tuple[bool, str]:
    email_norm = _normalize_email(email)
    db = load_user_db()
    users = db.get("users", {})
    
    if email_norm not in users: return False, "유저 없음"
    if email_norm == _normalize_email(acting_admin): return False, "셀프 차단 불가"
    
    new_status = not users[email_norm].get("is_banned", False)
    users[email_norm]["is_banned"] = new_status
    
    action = "차단" if new_status else "해제"
    log_audit_event("BAN_TOGGLE", acting_admin, email_norm, f"Action: {action}")
    save_user_db(db)
    return True, f"{email_norm} {action} 완료."

def update_user_profile(email: str, new_nickname: str = None, new_password: str = None) -> Tuple[bool, str]:
    email_norm = _normalize_email(email)
    if email_norm == MASTER_ADMIN_ID: return False, "관리자 계정 수정 불가"
    
    db = load_user_db()
    users = db.get("users", {})
    if email_norm not in users: return False, "유저 없음"
    
    user = users[email_norm]
    changed = False
    
    if new_nickname and new_nickname != user.get("nickname"):
        user["nickname"] = new_nickname
        changed = True
        
    if new_password:
        ok, msg = _validate_password(new_password)
        if not ok: return False, msg
        salt = _create_salt()
        user["salt"] = salt
        user["password_hash"] = _hash_password(new_password, salt)
        changed = True
        
    if changed:
        user["last_login"] = _now_utc_str()
        log_audit_event("PROFILE_UPDATE", email_norm, email_norm)
        save_user_db(db)
        if st.session_state.get(CURRENT_USER_KEY):
             st.session_state[CURRENT_USER_KEY] = user
        return True, "정보 수정 완료"
    
    return True, "변경 사항 없음"


# ----------------- UI Component -----------------
def render_auth_box(show_debug: bool = False):
    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None
    if JUST_REGISTERED_KEY not in st.session_state:
        st.session_state[JUST_REGISTERED_KEY] = False

    user = get_user()

    # 🚨 실시간 세션 검증
    if user:
        if user.get('login_id') != MASTER_ADMIN_ID:
            db = load_user_db()
            db_user = db.get("users", {}).get(user['login_id'])
            
            if not db_user or db_user.get("is_banned", False):
                st.session_state[CURRENT_USER_KEY] = None
                st.error("🚨 계정이 정지되었습니다.")
                st.rerun()
            
            # 세션 토큰 불일치 (중복 로그인)
            if db_user.get("session_token") != user.get("session_token"):
                st.session_state[CURRENT_USER_KEY] = None
                st.warning("⚠️ 다른 기기 접속으로 로그아웃되었습니다.")
                st.rerun()

            # 권한 동기화
            if db_user.get("role") != user.get("role"):
                user["role"] = db_user.get("role")
                st.session_state[CURRENT_USER_KEY] = user
                st.rerun()

    # 로그인 상태 UI
    if user:
        c1, c2 = st.columns([3, 1])
        with c1: st.success(f"✅ **{user['nickname']}**님")
        with c2: 
            if st.button("Logout", type="secondary"):
                st.session_state[CURRENT_USER_KEY] = None
                st.rerun()
        
        with st.expander(f"⚙️ 설정 ({user.get('role','free')})"):
            with st.form("my_profile"):
                n_nick = st.text_input("닉네임 변경", value=user['nickname'])
                n_pw = st.text_input("새 비밀번호", type="password")
                if st.form_submit_button("수정"):
                    ok, msg = update_user_profile(user['login_id'], n_nick, n_pw)
                    if ok: st.success(msg); st.rerun()
                    else: st.error(msg)
        return user

    # 비로그인 상태 UI
    tab1, tab2, tab3 = st.tabs(["로그인", "회원가입", "비번찾기"])
    
    with tab1:
        with st.form("login"):
            id_ = st.text_input("이메일")
            pw_ = st.text_input("비밀번호", type="password")
            if st.form_submit_button("로그인", type="primary"):
                u, m = authenticate_user(id_, pw_)
                if u:
                    st.session_state[CURRENT_USER_KEY] = u
                    st.rerun()
                else: st.error(m)

    with tab2:
        with st.form("signup"):
            em = st.text_input("이메일")
            nk = st.text_input("닉네임")
            p1 = st.text_input("비밀번호 (6자+)", type="password")
            p2 = st.text_input("비밀번호 확인", type="password")
            q_idx = st.selectbox("보안질문", range(len(SECURITY_QUESTIONS)), format_func=lambda x: SECURITY_QUESTIONS[x])
            ans = st.text_input("답변")
            code = st.text_input("초대코드 (선택)", type="password")
            
            if st.form_submit_button("가입"):
                if p1 != p2: st.error("비밀번호 불일치")
                else:
                    ok, msg, u = register_user(em, p1, nk, q_idx, ans, code)
                    if ok:
                        st.session_state[CURRENT_USER_KEY] = u
                        st.session_state[JUST_REGISTERED_KEY] = True
                        st.success(msg)
                        st.rerun()
                    else: st.error(msg)

    with tab3:
        if "fs" not in st.session_state: st.session_state.fs = 1
        
        if st.session_state.fs == 1:
            with st.form("f1"):
                em = st.text_input("이메일")
                if st.form_submit_button("확인"):
                    db = load_user_db()
                    u = db.get("users", {}).get(_normalize_email(em))
                    if u and u.get("security_q_idx"):
                        st.session_state.f_em = _normalize_email(em)
                        st.session_state.f_q = u["security_q_idx"]
                        st.session_state.fs = 2
                        st.rerun()
                    else: st.error("정보를 찾을 수 없습니다.")
        
        elif st.session_state.fs == 2:
            q_txt = SECURITY_QUESTIONS[st.session_state.f_q]
            st.info(f"질문: {q_txt}")
            with st.form("f2"):
                ans = st.text_input("답변")
                npw = st.text_input("새 비밀번호", type="password")
                if st.form_submit_button("변경"):
                    if verify_recovery_info(st.session_state.f_em, st.session_state.f_q, ans):
                        reset_password_with_recovery(st.session_state.f_em, npw)
                        st.success("변경 완료. 로그인해주세요.")
                        st.session_state.fs = 1
                        st.rerun()
                    else: st.error("답변이 틀렸습니다.")
            
            if st.button("취소"):
                st.session_state.fs = 1
                st.rerun()

    return None
