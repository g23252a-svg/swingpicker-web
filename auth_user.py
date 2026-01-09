# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System (DuckDB Integrated + Dashboard Compatibility)
"""

import streamlit as st
import hashlib
import secrets
import time
import logging
from datetime import datetime, timezone
from db_utils import LDYDBManager

# 로깅 및 DB 초기화
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auth_user")

db = LDYDBManager()

CURRENT_USER_KEY = "ldy_current_user"
MASTER_ADMIN_ID = "admin"
MASTER_ADMIN_PW = "2022322"

SECURITY_QUESTIONS = [
    "선택하세요...", "가장 기억에 남는 여행지는?", "어릴 적 살던 동네 이름은?",
    "가장 좋아하는 보물 1호는?", "초등학교 담임 선생님 성함은?",
    "나의 좌우명은?", "부모님의 고향은 어디인가요?",
]

# ----------------- 보안 유틸 -----------------
def _create_salt() -> str:
    return secrets.token_hex(16)

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()

def _hash_answer(answer: str, salt: str) -> str:
    return _hash_password(answer.strip().lower(), salt)

def check_rate_limit(email: str, limit: int = 5):
    if "login_attempts" not in st.session_state: st.session_state.login_attempts = {}
    attempts = st.session_state.login_attempts.get(email, 0)
    if attempts >= limit: return False, "⛔ 시도 횟수 초과. 잠시 후 다시 시도하세요."
    return True, ""

# ----------------- 🚨 핵심: 대시보드(dashboard.py) 호환 함수들 -----------------
# 이 함수들이 없으면 dashboard.py에서 ImportError가 발생합니다.

def get_user():
    """현재 로그인된 사용자 세션 정보 반환"""
    return st.session_state.get(CURRENT_USER_KEY)

def list_users():
    """[관리자용] 모든 사용자 목록 조회"""
    return db.get_all_users()

def update_user_role(email, new_role, acting_admin="system"):
    """[관리자용] 사용자 권한 변경"""
    return db.update_user_role(email, new_role)

def toggle_user_ban(email, acting_admin="system"):
    """[관리자용] 사용자 차단/해제 토글"""
    return db.toggle_user_ban(email)

def load_inquiry_items():
    """문의 게시판 글 목록 로드"""
    return db.get_all_inquiries()

def save_inquiry_items(items):
    """문의 게시판 글 저장"""
    return db.save_inquiries(items)

def load_subscriptions_db():
    """[구독관리] 구독 정보 DB 로드 (dict 형태로 변환)"""
    users = db.get_all_users()
    subs_map = {}
    for u in users:
        email = u['id']
        expire = u.get('prime_expire_date')
        if expire:
            # datetime 객체를 YYYY-MM-DD 문자열로 변환
            exp_str = str(expire).split(" ")[0]
            join_str = str(u.get('join_date', '')).split(" ")[0]
            subs_map[email] = {
                "role": u.get('role', 'free'),
                "expire_at": exp_str,
                "paid_at": join_str
            }
    return {"subs": subs_map}

def save_subscriptions_db(db_dict):
    """[구독관리] 변경된 구독 정보 저장"""
    subs = db_dict.get("subs", {})
    for email, info in subs.items():
        role = info.get("role")
        expire = info.get("expire_at")
        if role and expire:
            db.update_user_subscription(email, role, expire)
    return True

# ----------------- UI Component (로그인/가입 화면) -----------------
def render_auth_box():
    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None

    user = st.session_state[CURRENT_USER_KEY]

    # 1. 로그인 상태
    if user:
        user_id = user if isinstance(user, str) else user.get('id')
        
        # 관리자 예외
        if user_id == MASTER_ADMIN_ID:
            st.sidebar.success("😎 관리자 모드")
        else:
            # DB 정보 조회
            user_info = db.get_user_by_id(user_id)
            if not user_info:
                st.session_state[CURRENT_USER_KEY] = None
                st.rerun()

            nickname = user_info.get('nickname', user_id)
            expire = user_info.get('prime_expire_date')
            role = user_info.get('role', 'free')
            
            is_prime = False
            remain_msg = "무료 회원"
            
            # 프라임 기간 체크
            if role in ['prime', 'pro', 'admin'] and expire:
                try:
                    if isinstance(expire, str):
                        expire = datetime.strptime(expire.split('.')[0], "%Y-%m-%d %H:%M:%S")
                    
                    remain = expire - datetime.now()
                    if remain.total_seconds() > 0:
                        is_prime = True
                        remain_msg = f"👑 {role.upper()} ({remain.days}일 남음)"
                    else:
                        remain_msg = "🌑 이용권 만료됨"
                except:
                    pass

            st.sidebar.markdown(f"### 👋 **{nickname}**님")
            if is_prime:
                st.sidebar.success(remain_msg)
            else:
                st.sidebar.info(remain_msg)
                if user_id != MASTER_ADMIN_ID:
                    st.sidebar.button("멤버십 구독하기")

        if st.sidebar.button("로그아웃", type="primary"):
            st.session_state[CURRENT_USER_KEY] = None
            st.rerun()
        return user_id

    # 2. 비로그인 상태
    st.markdown("### 🔐 LDY Pro Trader")
    tab1, tab2, tab3 = st.tabs(["로그인", "회원가입", "비밀번호 찾기"])

    # [로그인]
    with tab1:
        with st.form("login_form"):
            login_id = st.text_input("이메일")
            login_pw = st.text_input("비밀번호", type="password")
            if st.form_submit_button("로그인", type="primary"):
                if login_id == MASTER_ADMIN_ID and login_pw == MASTER_ADMIN_PW:
                    st.session_state[CURRENT_USER_KEY] = MASTER_ADMIN_ID
                    st.rerun()
                
                ok, msg = check_rate_limit(login_id)
                if not ok: st.error(msg)
                else:
                    u = db.get_user_by_id(login_id)
                    if u and _hash_password(login_pw, u['salt']) == u['password']:
                        db.update_login_timestamp(login_id)
                        st.session_state[CURRENT_USER_KEY] = login_id
                        st.success("로그인 성공")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("로그인 실패")

    # [회원가입]
    with tab2:
        st.info("🎁 가입 즉시 **7일간 프라임(유료) 기능** 무료!")
        with st.form("join_form"):
            nid = st.text_input("이메일")
            nnick = st.text_input("닉네임")
            npw = st.text_input("비밀번호 (6자 이상)", type="password")
            npw2 = st.text_input("비번 확인", type="password")
            q = st.selectbox("질문", range(len(SECURITY_QUESTIONS)), format_func=lambda x: SECURITY_QUESTIONS[x])
            a = st.text_input("답변")
            
            if st.form_submit_button("가입"):
                if npw != npw2: st.error("비번 불일치")
                elif len(npw) < 6: st.error("비번 6자 이상")
                else:
                    salt = _create_salt()
                    ph = _hash_password(npw, salt)
                    ah = _hash_answer(a, salt)
                    # DB 저장 호출
                    ok, msg = db.register_user(nid, ph, salt, nnick, q, ah)
                    if ok:
                        st.balloons()
                        st.success(msg)
                        st.session_state[CURRENT_USER_KEY] = nid
                        time.sleep(2)
                        st.rerun()
                    else: st.error(msg)

    # [비번찾기]
    with tab3:
        if "fs" not in st.session_state: st.session_state.fs = 1
        
        if st.session_state.fs == 1:
            with st.form("find_step1"):
                fid = st.text_input("아이디")
                if st.form_submit_button("확인"):
                    u = db.get_user_by_id(fid)
                    if u:
                        st.session_state.f_email = fid
                        st.session_state.f_salt = u['salt']
                        st.session_state.f_q_idx = u['security_q_idx']
                        st.session_state.f_a_hash = u['security_a_hash']
                        st.session_state.fs = 2
                        st.rerun()
                    else: st.error("계정 없음")
        elif st.session_state.fs == 2:
            q_text = SECURITY_QUESTIONS[st.session_state.f_q_idx] if st.session_state.f_q_idx < len(SECURITY_QUESTIONS) else "오류"
            st.info(f"질문: {q_text}")
            with st.form("find_step2"):
                fans = st.text_input("답변")
                fnpw = st.text_input("새 비번", type="password")
                if st.form_submit_button("변경"):
                    if _hash_answer(fans, st.session_state.f_salt) == st.session_state.f_a_hash:
                        nsalt = _create_salt()
                        nhash = _hash_password(fnpw, nsalt)
                        db.update_password(st.session_state.f_email, nhash, nsalt)
                        st.success("변경 완료")
                        st.session_state.fs = 1
                        st.rerun()
                    else: st.error("답변 틀림")
            if st.button("취소"):
                st.session_state.fs = 1
                st.rerun()
