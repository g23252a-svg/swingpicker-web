# auth_user.py
# -*- coding: utf-8 -*-
"""
간단 회원관리 / 로그인 모듈 (LDY Pro Trader 전용)

- users.json 에 계정 저장
- 비밀번호는 salt + sha256 으로 해시 (간단 버전)
- Streamlit UI: render_auth_box(container=None) 로 렌더링
  * return 값: 로그인된 user dict or None
"""

import os
import json
import hashlib
import secrets
from datetime import datetime

import streamlit as st

# 유저 데이터 파일 경로 (환경변수로도 바꿀 수 있게)
USERS_FILE = os.getenv("LDY_USERS_FILE", "users.json")


# ---------------------------
# 내부 유틸
# ---------------------------
def _hash_password(password: str, salt: str) -> str:
    """salt + password 를 sha256 으로 해시"""
    base = (salt + password).encode("utf-8")
    return hashlib.sha256(base).hexdigest()


def _load_db() -> dict:
    """users.json 로드 (없으면 기본 구조 리턴)"""
    if not os.path.exists(USERS_FILE):
        return {"users": {}}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "users" not in data:
                data["users"] = {}
            return data
    except Exception:
        # 깨진 파일 보호용
        return {"users": {}}


def _save_db(data: dict) -> None:
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True) if os.path.dirname(USERS_FILE) else None
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _normalize_login_id(login_id: str) -> str:
    """로그인 ID는 소문자 이메일/아이디 기준으로 통일"""
    return login_id.strip().lower()


# ---------------------------
# 공개 API: 회원가입 / 로그인
# ---------------------------
def register_user(login_id: str, password: str, nickname: str = ""):
    """
    회원 가입 처리
    - login_id: 이메일 또는 아이디 (유일)
    - password: 비밀번호 평문 (입력값)
    - nickname: 닉네임 (선택)
    return: (ok: bool, msg: str)
    """
    login_id = _normalize_login_id(login_id)
    nickname = nickname.strip() or login_id

    if not login_id or not password:
        return False, "아이디와 비밀번호를 모두 입력해 주세요."

    db = _load_db()
    users = db.get("users", {})

    if login_id in users:
        return False, "이미 가입된 아이디입니다."

    # salt 생성 + 해시
    salt = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)

    user_obj = {
        "login_id": login_id,
        "nickname": nickname,
        "role": "free",  # 기본 등급은 free, 나중에 관리자 화면/직접 수정으로 pro/prime 부여
        "salt": salt,
        "password_hash": pw_hash,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": None,
        # 향후 확장용: 이메일, 전화번호, 메모 등
    }

    users[login_id] = user_obj
    db["users"] = users
    _save_db(db)

    return True, "회원가입이 완료되었습니다. 이제 로그인해 주세요."


def authenticate_user(login_id: str, password: str):
    """
    로그인 인증
    return: user dict (성공 시) / None (실패 시)
    """
    login_id = _normalize_login_id(login_id)
    db = _load_db()
    user = db.get("users", {}).get(login_id)

    if not user:
        return None

    salt = user.get("salt", "")
    pw_hash = user.get("password_hash", "")

    if _hash_password(password, salt) != pw_hash:
        return None

    # last_login 업데이트
    user["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db["users"][login_id] = user
    _save_db(db)

    return user


def get_user(role_required=None):
    """
    세션에서 현재 로그인 유저 가져오기
    role_required: "pro", "prime", "admin" 등 필요시 사용
    """
    user = st.session_state.get("user")
    if not user:
        return None
    if role_required is None:
        return user
    # 등급 체크 (대충: admin >= prime >= pro >= free 가정)
    role_order = ["free", "pro", "prime", "admin"]
    try:
        cur = role_order.index(user.get("role", "free"))
        need = role_order.index(role_required)
        if cur >= need:
            return user
        return None
    except ValueError:
        return None


# ---------------------------
# Streamlit UI: 로그인/회원가입 박스
# ---------------------------
def render_auth_box():
    """
    사이드바나 원하는 위치에서 호출:
        with st.sidebar:
            user = render_auth_box()
    return: 로그인된 user dict or None
    """
    if "user" not in st.session_state:
        st.session_state["user"] = None

    user = st.session_state["user"]

    # 이미 로그인 상태면 간단 정보 + 로그아웃 버튼
    if user:
        st.markdown(f"### 👋 {user.get('nickname', user.get('login_id'))} 님")
        st.caption(f"역할(role): **{user.get('role', 'free')}**")
        if st.button("로그아웃"):
            st.session_state["user"] = None
            st.success("로그아웃 되었습니다.")
            st.experimental_rerun()
        return user

    st.subheader("🔐 로그인 / 회원가입")

    mode = st.radio(
        "모드 선택",
        options=["로그인", "회원가입"],
        horizontal=True,
        key="auth_mode_radio",
    )

    if mode == "로그인":
        login_id = st.text_input("아이디 (이메일 등)", key="login_id_login")
        password = st.text_input("비밀번호", type="password", key="pw_login")
        if st.button("로그인", type="primary", use_container_width=True):
            user = authenticate_user(login_id, password)
            if user:
                st.session_state["user"] = user
                st.success("로그인 성공!")
                st.experimental_rerun()
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
    else:  # 회원가입
        login_id = st.text_input("아이디 (이메일 등)", key="login_id_sign")
        nickname = st.text_input("닉네임 (선택)", key="nickname_sign")
        pw1 = st.text_input("비밀번호", type="password", key="pw1_sign")
        pw2 = st.text_input("비밀번호 확인", type="password", key="pw2_sign")

        if st.button("회원가입", type="primary", use_container_width=True):
            if pw1 != pw2:
                st.error("비밀번호가 서로 다릅니다.")
            else:
                ok, msg = register_user(login_id, pw1, nickname)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    return st.session_state.get("user")
