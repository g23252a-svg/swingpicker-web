# auth_user.py
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os
import json
import hashlib
from datetime import datetime, timezone
import streamlit as st

print("DEBUG GIST_ID =", GIST_ID)
print("DEBUG GIST_TOKEN set?", bool(GIST_TOKEN))

# 🔹 auth_user.py가 있는 폴더 기준으로 data 폴더 고정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USER_DB_PATH = os.path.join(DATA_DIR, "users_db.json")




# Secrets / Env 우선
def _get_conf(key, default_val):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except FileNotFoundError:
        pass
    return os.getenv(key, default_val)

KEY_PRO   = _get_conf("LDY_KEY_PRO",   "2024")
KEY_PRIME = _get_conf("LDY_KEY_PRIME", "2025")
ADMIN_KEY = _get_conf("LDY_ADMIN_KEY", "2022322")

CURRENT_USER_KEY = "ldy_current_user"


# ----------------- 공통 시간 유틸 (UTC 문자열) -----------------
def _now_utc_str() -> str:
    """
    DB에 저장용 공통 시간 포맷
    - 항상 UTC 기준, ISO8601 문자열로 저장
    - 예: 2025-12-04T10:22:11.123456+00:00
    """
    return datetime.now(timezone.utc).isoformat()


# ----------------- 기본 DB 유틸 -----------------
def _ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

def load_user_db():
    _ensure_data_dir()
    if not os.path.exists(USER_DB_PATH):
        return {"users": {}}
    try:
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
        if "users" not in db:
            db["users"] = {}
        return db
    except Exception:
        return {"users": {}}

def save_user_db(db):
    _ensure_data_dir()
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


# ----------------- 비밀번호 해시 -----------------
def _create_salt(email: str) -> str:
    base = f"{email}-{datetime.now().timestamp()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


# ----------------- 회원 가입 / 로그인 -----------------
def register_user(email: str, password: str, nickname: str, invite_code: str = ""):
    email = email.strip()
    if not email or not password:
        return False, "이메일 / 비밀번호를 입력해 주세요.", None

    db = load_user_db()
    users = db.get("users", {})

    if email in users:
        return False, "이미 존재하는 이메일입니다.", None

    role = "free"
    invite_code = (invite_code or "").strip()

    if invite_code == ADMIN_KEY:
        role = "admin"
    elif invite_code == KEY_PRIME:
        role = "prime"
    elif invite_code == KEY_PRO:
        role = "pro"

    salt = _create_salt(email)
    pw_hash = _hash_password(password, salt)
    now_utc = datetime.now(timezone.utc).isoformat()

    users[email] = {
        "login_id": email,
        "nickname": nickname or email.split("@")[0],
        "role": role,
        "salt": salt,
        "password_hash": pw_hash,
        "created_at": now_utc,
        "last_login": now_utc,   # 가입 시점 = 첫 로그인 시점
    }

    db["users"] = users
    save_user_db(db)
    return True, f"회원가입 완료! 현재 권한: {role}", users[email]


def authenticate_user(email: str, password: str):
    email = email.strip()
    db = load_user_db()
    users = db.get("users", {})
    user = users.get(email)

    if not user:
        return None, "존재하지 않는 계정입니다."

    salt = user.get("salt", "")
    pw_hash = user.get("password_hash", "")

    if _hash_password(password, salt) != pw_hash:
        return None, "비밀번호가 일치하지 않습니다."

    # 🔹 로그인 성공 시점에 last_login을 UTC 기준으로 갱신
    now_str = _now_utc_str()
    user["last_login"] = now_str
    users[email] = user
    db["users"] = users
    save_user_db(db)

    return user, "로그인 성공"


def get_user():
    return st.session_state.get(CURRENT_USER_KEY)


# ----------------- Admin용 헬퍼 (dashboard.py에서 사용) -----------------
def list_users():
    """전체 유저 목록 리스트 반환 (dashboard용)"""
    db = load_user_db()
    users = db.get("users", {})
    return list(users.values())


def update_user_role(email: str, new_role: str) -> bool:
    """관리자가 회원 권한 변경"""
    db = load_user_db()
    users = db.get("users", {})
    if email not in users:
        return False
    users[email]["role"] = new_role
    db["users"] = users
    save_user_db(db)
    return True


# ----------------- UI: 로그인 / 회원가입 박스 -----------------
def _render_admin_panel(current_user):
    """(선택) auth_user 내부에서도 간단 관리 패널 제공 가능"""
    db = load_user_db()
    users = db.get("users", {})
    st.markdown("---")
    st.subheader("🛠 회원 권한 관리 (Admin - auth_user)")

    if not users:
        st.info("등록된 계정이 없습니다.")
        return

    email_list = sorted(users.keys())
    target_email = st.selectbox("계정 선택", email_list)
    if not target_email:
        return

    target_user = users[target_email]
    current_role = target_user.get("role", "free")
    st.write(f"선택된 계정: `{target_email}` (현재 권한: `{current_role}`)")

    role_options = ["free", "pro", "prime", "admin"]
    idx = role_options.index(current_role) if current_role in role_options else 0
    new_role = st.radio(
        "새 권한",
        role_options,
        index=idx,
        horizontal=True,
        key=f"admin_role_{target_email}",
    )

    if st.button("권한 변경 적용", key=f"admin_apply_{target_email}"):
        ok = update_user_role(target_email, new_role)
        if ok:
            st.success(f"{target_email} 권한이 `{new_role}` 로 변경되었습니다.")
        else:
            st.error("변경 실패")


def render_auth_box():
    """사이드바 로그인/회원가입 UI"""
    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None

    st.subheader("🔐 계정 로그인 / 회원가입")
    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    # 로그인 탭
    with tab_login:
        login_email = st.text_input("이메일", key="login_email")
        login_pw = st.text_input("비밀번호", type="password", key="login_pw")
        if st.button("로그인", key="btn_login"):
            user, msg = authenticate_user(login_email, login_pw)
            if user is None:
                st.error(msg)
            else:
                st.session_state[CURRENT_USER_KEY] = user
                st.success(f"{user['nickname']}님 환영합니다! ({user['role']})")

    # 회원가입 탭
    with tab_signup:
        reg_email = st.text_input("이메일", key="reg_email")
        reg_nick = st.text_input("닉네임 (선택)", key="reg_nick")
        reg_pw1 = st.text_input("비밀번호", type="password", key="reg_pw1")
        reg_pw2 = st.text_input("비밀번호 확인", type="password", key="reg_pw2")
        reg_code = st.text_input(
            "초대/구독 코드 (선택 - Pro/Prime/Admin 용)",
            type="password",
            key="reg_code",
        )

        if st.button("회원가입", key="btn_register"):
            if reg_pw1 != reg_pw2:
                st.error("비밀번호가 서로 일치하지 않습니다.")
            else:
                ok, msg, new_user = register_user(
                    reg_email, reg_pw1, reg_nick, reg_code
                )
                if ok:
                    st.success(msg)
                    st.session_state[CURRENT_USER_KEY] = new_user
                else:
                    st.error(msg)

    user = get_user()
    if user:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"👤 {user['nickname']}님 로그인 중 (권한: `{user['role']}`)")
        with col2:
            if st.button("로그아웃", key="btn_logout"):
                st.session_state[CURRENT_USER_KEY] = None
                st.success("로그아웃 되었습니다.")
                user = None

    # 관리자면, 이 파일 안에서도 추가 패널 노출할 수 있음 (선택 사항)
    # if user and user.get("role") == "admin":
    #     _render_admin_panel(user)

    return user
