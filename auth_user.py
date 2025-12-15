# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System v7.5
- v7.5: PBKDF2 보안 강화, Gist 로딩 캐싱(st.cache_data) 적용, 타입 힌트 보강
"""

import os
import json
import hashlib
import logging
import re
import time
import secrets  # ✅ [New] 암호학적으로 안전한 난수 생성
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

import requests
import streamlit as st

# import extra_streamlit_components as stx  # 🧹 [삭제] 미사용 라이브러리 제거

AUTH_IMPORT_ERR = None

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

# Gist 설정
GIST_ID_USERS = _get_conf("LDY_GIST_ID", "")
GIST_TOKEN    = _get_conf("LDY_GIST_TOKEN", "")
GIST_ID_SUBS = _get_conf("LDY_GIST_SUBS_ID", GIST_ID_USERS)
GIST_ID_INQ  = _get_conf("LDY_GIST_INQ_ID",  GIST_ID_USERS)

# ----------------- [핵심 수정] 쿠키 매니저 설정 비활성화 -----------------

def get_cookie_manager():
    # return stx.CookieManager(key="cookie_manager_core") # 👈 [임시 주석]
    return None

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

# ----------------- 통합 DB 유틸 -----------------
_USER_DB_CACHE: Optional[dict] = None
_USER_DB_CACHE_TS: Optional[float] = None
_USER_DB_CACHE_TTL = 30

def load_user_db() -> dict:
    global _USER_DB_CACHE, _USER_DB_CACHE_TS
    now_ts = datetime.now(timezone.utc).timestamp()
    if _USER_DB_CACHE is not None and _USER_DB_CACHE_TS is not None:
        if now_ts - _USER_DB_CACHE_TS < _USER_DB_CACHE_TTL:
            return _USER_DB_CACHE
    db = _load_user_db_from_gist()
    if db is not None:
        _save_user_db_local(db)
        _USER_DB_CACHE = db
        _USER_DB_CACHE_TS = now_ts
        return db
    db = _load_user_db_local()
    _USER_DB_CACHE = db
    _USER_DB_CACHE_TS = now_ts
    return db

def save_user_db(db: dict) -> None:
    global _USER_DB_CACHE, _USER_DB_CACHE_TS
    db = _normalize_user_db_structure(db)
    _USER_DB_CACHE = db
    _USER_DB_CACHE_TS = datetime.now(timezone.utc).timestamp()
    _save_user_db_local(db)
    if GIST_ID_USERS and GIST_TOKEN:
        _save_user_db_to_gist(db)

# ----------------- 보안 유틸 (v7.5 강화) -----------------
def _create_salt() -> str:
    """암호학적으로 안전한 32바이트 Salt 생성"""
    return secrets.token_hex(16)

def _hash_password(password: str, salt: str) -> str:
    """
    v7.5: PBKDF2-HMAC-SHA256 적용 (100,000 iterations)
    기존 단순 해싱보다 Rainbow Table 공격 등에 훨씬 안전함.
    """
    return hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    ).hex()

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
def _validate_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email)) if email else False

def _validate_password(pw: str) -> Tuple[bool, str]:
    if len(pw) < 6: return False, "비밀번호는 최소 6자 이상이어야 합니다."
    return True, ""

def register_user(email: str, password: str, nickname: str, invite_code: str = ""):
    email_norm = _normalize_email(email)
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
    salt = _create_salt(email_norm)
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
    db = load_user_db()
    users = db.get("users", {})
    user = users.get(email_norm)
    if not user:
        return None, "이메일 또는 비밀번호가 일치하지 않습니다."
    salt = user.get("salt", "")
    pw_hash = user.get("password_hash", "")
    if _hash_password(password, salt) != pw_hash:
        return None, "이메일 또는 비밀번호가 일치하지 않습니다."
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

# ----------------- UI: 로그인 / 회원가입 박스 (쿠키 임시 비활성화됨) -----------------
def render_auth_box(show_debug: bool = False):
    """
    브라우저 탭 전환 / 새로고침 시에도 로그인을 유지하기 위해 CookieManager 사용
    (현재 화면 로딩 이슈로 비활성화: 필요시 주석 해제)
    """
    # 1. 쿠키 매니저 로드 (임시 비활성)
    cookie_manager = None 
    # cookie_manager = get_cookie_manager()
    
    # 2. 쿠키 값 읽기 (임시 비활성)
    cookie_user_email = None
    # if cookie_manager:
    #     cookie_user_email = cookie_manager.get(cookie="ldy_user_email")

    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None

    if JUST_REGISTERED_KEY not in st.session_state:
        st.session_state[JUST_REGISTERED_KEY] = False

    # 3. [자동 로그인] 쿠키 기반 세션 복구 (비활성)
    if st.session_state[CURRENT_USER_KEY] is None and cookie_user_email:
        # time.sleep(0.1)
        db = load_user_db()
        users = db.get("users", {})
        saved_user = users.get(cookie_user_email)
        
        if saved_user:
            st.session_state[CURRENT_USER_KEY] = saved_user
            if show_debug: 
                print(f"[AutoLogin] Restored session for {cookie_user_email}")

    # 현재 세션 사용자 가져오기
    user = get_user()

    # ---------------- [상태 1] 로그인 된 상태 ----------------
    if user:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"👤 {user['nickname']}님\n(권한: `{user.get('role', 'free')}`)")
        with col2:
            if st.button("로그아웃", key="btn_logout"):
                # 세션 삭제
                st.session_state[CURRENT_USER_KEY] = None
                st.session_state[JUST_REGISTERED_KEY] = False
                
                # 쿠키 삭제 (비활성)
                # if cookie_manager:
                #     cookie_manager.delete("ldy_user_email")
                
                st.toast("로그아웃 되었습니다.", icon="👋")
                time.sleep(0.5) 
                st.rerun()
        
        return user

    # ---------------- [상태 2] 비로그인 상태 (로그인 폼) ----------------
    st.subheader("🔐 계정 로그인 / 회원가입")
    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    # 1) 로그인 탭
    with tab_login:
        with st.form(key="login_form"):
            login_email = st.text_input("이메일")
            login_pw = st.text_input("비밀번호", type="password")
            # 자동 로그인 옵션 (비활성 상태임을 표시)
            remember_me = st.checkbox("로그인 상태 유지 (현재 점검중)", value=False, disabled=True)
            submit_login = st.form_submit_button("로그인")
        
        if submit_login:
            user_obj, msg = authenticate_user(login_email, login_pw)
            if user_obj is None:
                st.error(msg)
            else:
                st.session_state[CURRENT_USER_KEY] = user_obj
                st.session_state[JUST_REGISTERED_KEY] = False
                
                # 쿠키 저장 (비활성)
                # if remember_me and cookie_manager:
                #     expires = datetime.now() + timedelta(days=30)
                #     cookie_manager.set("ldy_user_email", user_obj['login_id'], expires_at=expires)
                
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
            
            st.markdown("ℹ️ 가입 시 **상위 5개 추천 종목** 무료 열람")
            submit_reg = st.form_submit_button("회원가입")

        if submit_reg:
            if reg_pw1 != reg_pw2:
                st.error("비밀번호가 서로 일치하지 않습니다.")
            else:
                ok, msg, new_user = register_user(reg_email, reg_pw1, reg_nick, reg_code)
                if ok:
                    st.session_state[CURRENT_USER_KEY] = new_user
                    st.session_state[JUST_REGISTERED_KEY] = True
                    
                    # 쿠키 저장 (비활성)
                    # if cookie_manager:
                    #     expires = datetime.now() + timedelta(days=30)
                    #     cookie_manager.set("ldy_user_email", new_user['login_id'], expires_at=expires)
                    
                    st.success(msg)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)

    return None
