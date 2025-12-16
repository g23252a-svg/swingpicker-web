# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System v7.5
- v7.5: PBKDF2 보안 강화, Gist 로딩 캐싱(st.cache_data) 적용, 타입 힌트 보강
LDY Pro Trader Auth System v8.0
- v8.0 Upgrade:
  1. 보안: 로그인 시도 횟수 제한 (Rate Limiting) 적용 (Brute Force 방어)
  2. 성능: st.cache_data를 이용한 Native Caching (메모리 효율 및 속도 개선)
  3. 기능: 마이페이지 (닉네임/비밀번호 변경) 추가
  4. 기존: PBKDF2 보안, Gist/Local 이중화 유지
"""

import os
@@ -10,17 +14,13 @@
import logging
import re
import time
import secrets  # ✅ [New] 암호학적으로 안전한 난수 생성
import secrets
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

import requests
import streamlit as st

# import extra_streamlit_components as stx  # 🧹 [삭제] 미사용 라이브러리 제거

AUTH_IMPORT_ERR = None

# ----------------- 로깅 설정 -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auth_user")
@@ -52,11 +52,50 @@ def _get_conf(key: str, default_val: str) -> str:
GIST_ID_SUBS = _get_conf("LDY_GIST_SUBS_ID", GIST_ID_USERS)
GIST_ID_INQ  = _get_conf("LDY_GIST_INQ_ID",  GIST_ID_USERS)

# ----------------- [핵심 수정] 쿠키 매니저 설정 비활성화 -----------------

def get_cookie_manager():
    # return stx.CookieManager(key="cookie_manager_core") # 👈 [임시 주석]
    return None
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
@@ -285,55 +324,54 @@ def save_inquiry_items(items: list) -> bool:
    db["inquiries"] = list(items) if isinstance(items, list) else []
    return save_inquiries_db(db)

# ----------------- 통합 DB 유틸 -----------------
_USER_DB_CACHE: Optional[dict] = None
_USER_DB_CACHE_TS: Optional[float] = None
_USER_DB_CACHE_TTL = 30
# ----------------- [Upgrade 2] 통합 DB: Native Caching -----------------

def load_user_db() -> dict:
    global _USER_DB_CACHE, _USER_DB_CACHE_TS
    now_ts = datetime.now(timezone.utc).timestamp()
    if _USER_DB_CACHE is not None and _USER_DB_CACHE_TS is not None:
        if now_ts - _USER_DB_CACHE_TS < _USER_DB_CACHE_TTL:
            return _USER_DB_CACHE
@st.cache_data(ttl=60, show_spinner=False)
def _fetch_user_db_cached() -> dict:
    """
    Gist/Local 데이터를 가져오고 60초간 Streamlit 캐시에 보관합니다.
    """
    # 1. Gist 시도
    db = _load_user_db_from_gist()
    if db is not None:
        _save_user_db_local(db)
        _USER_DB_CACHE = db
        _USER_DB_CACHE_TS = now_ts
    if db:
        return db
    db = _load_user_db_local()
    _USER_DB_CACHE = db
    _USER_DB_CACHE_TS = now_ts
    return db
    # 2. 실패 시 로컬 로드
    return _load_user_db_local()

def load_user_db() -> dict:
    """캐시된 DB 로드"""
    return _fetch_user_db_cached()

def save_user_db(db: dict) -> None:
    global _USER_DB_CACHE, _USER_DB_CACHE_TS
    """DB 저장 후 캐시 무효화"""
    db = _normalize_user_db_structure(db)
    _USER_DB_CACHE = db
    _USER_DB_CACHE_TS = datetime.now(timezone.utc).timestamp()
    
    # 1. 로컬 & Gist 저장
    _save_user_db_local(db)
    if GIST_ID_USERS and GIST_TOKEN:
        _save_user_db_to_gist(db)
    
    # 2. [핵심] 캐시 초기화 (다음 load시 새로 받아옴)
    _fetch_user_db_cached.clear()


# ----------------- 보안 유틸 (v7.5 강화) -----------------
# ----------------- 보안 유틸 (v7.5 유지) -----------------
def _create_salt() -> str:
    """암호학적으로 안전한 32바이트 Salt 생성"""
    return secrets.token_hex(16)

def _hash_password(password: str, salt: str) -> str:
    """
    v7.5: PBKDF2-HMAC-SHA256 적용 (100,000 iterations)
    기존 단순 해싱보다 Rainbow Table 공격 등에 훨씬 안전함.
    """
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
@@ -343,6 +381,8 @@ def _validate_password(pw: str) -> Tuple[bool, str]:
    if len(pw) < 6: return False, "비밀번호는 최소 6자 이상이어야 합니다."
    return True, ""

# ----------------- 회원 관리 로직 -----------------

def register_user(email: str, password: str, nickname: str, invite_code: str = ""):
    email_norm = _normalize_email(email)
    if not email_norm or not password:
@@ -352,18 +392,22 @@ def register_user(email: str, password: str, nickname: str, invite_code: str = "
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
    
    salt = _create_salt()
    pw_hash = _hash_password(password, salt)
    now_utc = _now_utc_str()
    
    users[email_norm] = {
        "login_id": email_norm,
        "nickname": nickname or email_norm.split("@")[0],
@@ -373,43 +417,47 @@ def register_user(email: str, password: str, nickname: str, invite_code: str = "
        "created_at": now_utc,
        "last_login": now_utc,
    }
    
    db["users"] = users
    save_user_db(db)
    return True, f"회원가입 완료! 현재 권한: {role}", users[email_norm]


# 👇 [추가] 구버전 비밀번호 확인용 함수
def _hash_password_legacy(password: str, salt: str) -> str:
    """v7.4 이하 구버전 해싱 (SHA256) - 마이그레이션용"""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def authenticate_user(email: str, password: str):
    email_norm = _normalize_email(email)
    
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

    # 1. 신규 방식(PBKDF2)으로 검증
    # 1. 신규 방식(PBKDF2) 검증
    if _hash_password(password, salt) == pw_hash:
        pass # 통과
        pass # 성공

    # 2. 실패 시, 구버전 방식(SHA256)으로 재검증 (마이그레이션)
    # 2. 실패 시 구버전(SHA256) 검증 (마이그레이션)
    elif _hash_password_legacy(password, salt) == pw_hash:
        print(f"[System] Migrating password for {email_norm} to v7.5 security.")
        # 구버전 암호가 맞으면 -> 신규 방식으로 암호화하여 DB 업데이트
        new_hash = _hash_password(password, salt)
        user["password_hash"] = new_hash
        # (저장은 아래에서 한 번에 처리)
        # 저장은 아래에서 한 번에 처리
    else:
        record_login_fail(email_norm)
        return None, "이메일 또는 비밀번호가 일치하지 않습니다."

    # 로그인 성공 처리
    # 로그인 성공
    reset_login_fail(email_norm)
    
    now_str = _now_utc_str()
    user["last_login"] = now_str
    users[email_norm] = user
@@ -434,69 +482,117 @@ def update_user_role(email: str, new_role: str, acting_admin_email: Optional[str
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
# ----------------- [Upgrade 3] 마이페이지 기능 -----------------
def update_user_profile(email: str, new_nickname: str = None, new_password: str = None) -> Tuple[bool, str]:
    email_norm = _normalize_email(email)
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
    브라우저 탭 전환 / 새로고침 시에도 로그인을 유지하기 위해 CookieManager 사용
    (현재 화면 로딩 이슈로 비활성화: 필요시 주석 해제)
    브라우저 탭 전환 / 새로고침 시에도 로그인을 유지하기 위해 CookieManager 사용 (임시 비활성화)
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
    # [자동 로그인 복구 시도 - 비활성 상태]
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
        # 상단 정보 표시
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"👤 {user['nickname']}님\n(권한: `{user.get('role', 'free')}`)")
            st.success(f"✅ **{user['nickname']}**님 환영합니다!")
        with col2:
            if st.button("로그아웃", key="btn_logout"):
                # 세션 삭제
            if st.button("로그아웃", key="btn_logout", type="secondary"):
                st.session_state[CURRENT_USER_KEY] = None
                st.session_state[JUST_REGISTERED_KEY] = False
                
                # 쿠키 삭제 (비활성)
                # if cookie_manager:
                #     cookie_manager.delete("ldy_user_email")
                
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

@@ -509,9 +605,8 @@ def render_auth_box(show_debug: bool = False):
        with st.form(key="login_form"):
            login_email = st.text_input("이메일")
            login_pw = st.text_input("비밀번호", type="password")
            # 자동 로그인 옵션 (비활성 상태임을 표시)
            remember_me = st.checkbox("로그인 상태 유지 (현재 점검중)", value=False, disabled=True)
            submit_login = st.form_submit_button("로그인")
            submit_login = st.form_submit_button("로그인", type="primary")

        if submit_login:
            user_obj, msg = authenticate_user(login_email, login_pw)
@@ -520,12 +615,6 @@ def render_auth_box(show_debug: bool = False):
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
@@ -539,7 +628,7 @@ def render_auth_box(show_debug: bool = False):
            reg_pw2 = st.text_input("비밀번호 확인", type="password")
            reg_code = st.text_input("초대/구독 코드 (선택)", type="password")

            st.markdown("ℹ️ 가입 시 **상위 5개 추천 종목** 무료 열람")
            st.caption("ℹ️ 가입 시 **상위 5개 추천 종목** 무료 열람 가능")
            submit_reg = st.form_submit_button("회원가입")

        if submit_reg:
@@ -550,12 +639,6 @@ def render_auth_box(show_debug: bool = False):
                if ok:
                    st.session_state[CURRENT_USER_KEY] = new_user
                    st.session_state[JUST_REGISTERED_KEY] = True
                    
                    # 쿠키 저장 (비활성)
                    # if cookie_manager:
                    #     expires = datetime.now() + timedelta(days=30)
                    #     cookie_manager.set("ldy_user_email", new_user['login_id'], expires_at=expires)
                    
                    st.success(msg)
                    time.sleep(0.5)
