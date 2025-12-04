# auth_user.py
# -*- coding: utf-8 -*-

import os
import json
import hashlib
from datetime import datetime

import streamlit as st

# -----------------------
# 기본 설정
# -----------------------
DATA_DIR = "data"
USER_DB_PATH = os.path.join(DATA_DIR, "users_db.json")

# 구독 / 권한 코드 (환경변수 > 기본값)
KEY_PRO = os.getenv("LDY_KEY_PRO", "2024")
KEY_PRIME = os.getenv("LDY_KEY_PRIME", "2025")
ADMIN_KEY = os.getenv("LDY_ADMIN_KEY", "2022322")  # 👈 관리자 초대 코드

CURRENT_USER_KEY = "ldy_current_user"


# -----------------------
# 유틸 함수
# -----------------------
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
        # JSON 깨졌을 경우 초기화
        return {"users": {}}


def save_user_db(db):
    _ensure_data_dir()
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def _hash_password(password: str, salt: str) -> str:
    # 이메일별 salt + 비밀번호 해시
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _create_salt(email: str) -> str:
    # 이메일 기반 + random 을 섞어도 되지만, 여기선 간단히
    base = f"{email}-{datetime.now().timestamp()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def register_user(email: str, password: str, nickname: str, invite_code: str = ""):
    email = email.strip()
    if not email or not password:
        return False, "이메일 / 비밀번호를 입력해 주세요.", None

    db = load_user_db()
    users = db.get("users", {})

    if email in users:
        return False, "이미 존재하는 이메일입니다.", None

    # 권한 결정
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
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    users[email] = {
        "login_id": email,
        "nickname": nickname or email.split("@")[0],
        "role": role,
        "salt": salt,
        "password_hash": pw_hash,
        "created_at": now_str,
        "last_login": now_str,
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

    # 마지막 로그인 업데이트
    user["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    users[email] = user
    db["users"] = users
    save_user_db(db)

    return user, "로그인 성공"


def get_user():
    """현재 로그인한 유저 정보 반환"""
    return st.session_state.get(CURRENT_USER_KEY)


def _render_admin_panel(current_user):
    """관리자용: 다른 유저 권한 변경 UI"""
    db = load_user_db()
    users = db.get("users", {})

    st.markdown("---")
    st.subheader("🛠 회원 권한 관리 (Admin)")

    if not users:
        st.info("등록된 계정이 없습니다.")
        return

    email_list = sorted(users.keys())

    # 자기 자신도 보이게 둘지 말지는 취향, 여기선 포함
    target_email = st.selectbox("권한을 변경할 계정을 선택해주세요.", email_list)

    if not target_email:
        return

    target_user = users[target_email]
    current_role = target_user.get("role", "free")

    st.write(f"선택된 계정: `{target_email}` (현재 권한: `{current_role}`)")

    role_options = ["free", "pro", "prime", "admin"]
    try:
        idx = role_options.index(current_role)
    except ValueError:
        idx = 0

    new_role = st.radio(
        "새 권한 선택",
        role_options,
        index=idx,
        horizontal=True,
        key=f"role_radio_{target_email}",
    )

    if st.button("권한 변경 적용", key=f"apply_role_{target_email}"):
        users[target_email]["role"] = new_role
        db["users"] = users
        save_user_db(db)
        st.success(f"{target_email} 의 권한이 `{new_role}` 로 변경되었습니다.")


def render_auth_box():
    """사이드바에 로그인/회원가입 + 현재 유저 + (관리자용) 권한 관리까지 렌더링"""
    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None

    st.subheader("🔐 계정 로그인 / 회원가입")

    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    # ---------------- 로그인 탭 ----------------
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

    # ---------------- 회원가입 탭 ----------------
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

    # ---------------- 현재 로그인 상태 표시 ----------------
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

    # ---------------- 관리자일 경우: 권한 관리 UI ----------------
    if user and user.get("role") == "admin":
        _render_admin_panel(user)

    return user
