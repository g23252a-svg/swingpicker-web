# auth_user.py
# 간단한 파일 기반 회원 관리 + Streamlit UI

import os
import json
import hashlib
import secrets
import threading
from datetime import datetime

import streamlit as st

# ---------------------------
# 설정
# ---------------------------
USER_DB_PATH = os.getenv("LDY_USER_DB", "user_db.json")

# Pro / Prime / Admin 코드 (환경변수나 .streamlit/secrets.toml 로 관리 추천)
KEY_PRO = os.getenv("LDY_KEY_PRO", "PRO-2024")
KEY_PRIME = os.getenv("LDY_KEY_PRIME", "PRIME-2025")
ADMIN_KEY = os.getenv("LDY_ADMIN_KEY", "ADMIN-2022322")

_lock = threading.Lock()

if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

# ---------------------------
# 내부 유틸
# ---------------------------
def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _load_user_db():
    """user_db.json 로드 (없으면 기본 구조 반환)"""
    if not os.path.exists(USER_DB_PATH):
        return {"users": {}}
    try:
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"users": {}}

def _save_user_db(data):
    """user_db.json 저장"""
    os.makedirs(os.path.dirname(USER_DB_PATH) or ".", exist_ok=True)
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _hash_password(password: str, salt: str | None = None):
    """salt + SHA256 해시"""
    if salt is None:
        salt = secrets.token_hex(16)
    base = (salt + password).encode("utf-8")
    h = hashlib.sha256(base).hexdigest()
    return salt, h

# ---------------------------
# 외부에서 쓰는 함수
# ---------------------------
def get_user(login_id: str):
    db = _load_user_db()
    return db.get("users", {}).get(login_id)

def verify_login(login_id: str, password: str):
    """로그인 검증 → (True, user_dict) or (False, message)"""
    with _lock:
        db = _load_user_db()
        user = db.get("users", {}).get(login_id)
        if not user:
            return False, "존재하지 않는 계정입니다."
        salt = user.get("salt", "")
        _, h = _hash_password(password, salt)
        if h != user.get("password_hash"):
            return False, "비밀번호가 올바르지 않습니다."
        # last_login 업데이트
        user["last_login"] = _now_str()
        db["users"][login_id] = user
        _save_user_db(db)
        return True, user

def register_user(login_id: str, password: str, nickname: str, invite_code: str | None = None):
    """회원가입 → (True, msg) or (False, msg)"""
    login_id = login_id.strip().lower()
    if not login_id or "@" not in login_id:
        return False, "이메일 형식의 로그인 ID를 입력해 주세요."
    if len(password) < 6:
        return False, "비밀번호는 6자리 이상으로 설정해 주세요."
    if not nickname:
        return False, "닉네임을 입력해 주세요."

    with _lock:
        db = _load_user_db()
        users = db.setdefault("users", {})
        if login_id in users:
            return False, "이미 가입된 이메일입니다."

        # 기본은 free
        role = "free"
        invite_code = (invite_code or "").strip()
        if invite_code == KEY_PRO:
            role = "pro"
        elif invite_code == KEY_PRIME:
            role = "prime"
        elif invite_code == ADMIN_KEY:
            role = "admin"

        salt, pw_hash = _hash_password(password)
        users[login_id] = {
            "login_id": login_id,
            "nickname": nickname,
            "role": role,
            "salt": salt,
            "password_hash": pw_hash,
            "created_at": _now_str(),
            "last_login": "",
        }
        _save_user_db(db)
        return True, f"회원가입 완료! 현재 권한: {role}"

def list_users():
    """전체 유저 리스트 반환 (list[dict])"""
    db = _load_user_db()
    return list(db.get("users", {}).values())

def update_user_role(login_id: str, new_role: str):
    """관리자가 유저 role 변경"""
    with _lock:
        db = _load_user_db()
        users = db.setdefault("users", {})
        if login_id not in users:
            return False
        users[login_id]["role"] = new_role
        _save_user_db(db)
        return True

# ---------------------------
# Streamlit용 인증 박스
# ---------------------------
def render_auth_box():
    """사이드바에서 호출: 로그인/회원가입 UI + 세션 관리"""
    st.markdown("### 🔐 계정 로그인 / 회원가입")

    cur_user = st.session_state.get("current_user")

    # 이미 로그인된 경우 간단 표시 + 로그아웃 버튼
    if cur_user:
        st.success(f"{cur_user.get('nickname','')}님 환영합니다! ({cur_user.get('role','free')})")
        if st.button("로그아웃"):
            st.session_state["current_user"] = None
        return st.session_state.get("current_user")

    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    # --- 로그인 탭 ---
    with tab_login:
        login_id = st.text_input("이메일 (로그인 ID)", key="login_email")
        password = st.text_input("비밀번호", type="password", key="login_pw")
        if st.button("로그인", key="btn_login"):
            ok, res = verify_login(login_id, password)
            if ok:
                st.session_state["current_user"] = res
                st.success(f"{res.get('nickname','')}님 로그인 되었습니다.")
            else:
                st.error(res)

    # --- 회원가입 탭 ---
    with tab_signup:
        new_email = st.text_input("이메일", key="signup_email")
        nickname = st.text_input("닉네임", key="signup_nick")
        pw1 = st.text_input("비밀번호", type="password", key="signup_pw1")
        pw2 = st.text_input("비밀번호 확인", type="password", key="signup_pw2")
        invite = st.text_input("초대/구독 코드 (선택)", key="signup_invite", placeholder="Pro/Prime 코드가 있다면 입력")

        if st.button("회원가입", key="btn_signup"):
            if pw1 != pw2:
                st.error("비밀번호가 서로 일치하지 않습니다.")
            else:
                ok, msg = register_user(new_email, pw1, nickname, invite_code=invite)
                if ok:
                    st.success(msg)
                    st.info("이제 로그인 탭에서 로그인해 주세요.")
                else:
                    st.error(msg)

    return st.session_state.get("current_user")
