# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System (Integrated with db_utils)
"""

import streamlit as st
import hashlib
import secrets
import time
import logging
from datetime import datetime
from db_utils import LDYDBManager  # ✅ 업데이트된 DB 매니저 연결

# 로깅 및 DB 초기화
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auth_user")

# 🔥 핵심: 여기서 db_utils를 불러와야 연동이 됩니다!
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
    """메모리 기반 레이트 리밋 (간소화)"""
    if "login_attempts" not in st.session_state: st.session_state.login_attempts = {}
    attempts = st.session_state.login_attempts.get(email, 0)
    if attempts >= limit: return False, "⛔ 시도 횟수 초과. 잠시 후 다시 시도하세요."
    return True, ""

# ----------------- UI 및 로직 -----------------
def render_auth_box():
    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None

    user = st.session_state[CURRENT_USER_KEY]

    # 1. 로그인 상태 (대시보드 사이드바)
    if user:
        user_id = user if isinstance(user, str) else user.get('id')
        
        # 관리자 예외 처리
        if user_id == MASTER_ADMIN_ID:
            st.sidebar.success("😎 관리자 모드")
        else:
            # 🔍 DB에서 최신 정보 조회 (프라임 여부 확인)
            user_info = db.get_user_by_id(user_id)
            
            if not user_info:
                st.session_state[CURRENT_USER_KEY] = None
                st.rerun()

            nickname = user_info.get('nickname', user_id)
            expire = user_info.get('prime_expire_date')
            
            is_prime = False
            remain_msg = "무료 회원"
            
            # 프라임 만료일 계산
            if expire:
                if isinstance(expire, str):
                    try:
                        expire = datetime.strptime(expire.split('.')[0], "%Y-%m-%d %H:%M:%S")
                    except:
                        pass # 파싱 실패 시 무시
                
                if isinstance(expire, datetime):
                    remain = expire - datetime.now()
                    if remain.total_seconds() > 0:
                        is_prime = True
                        remain_msg = f"👑 프라임 (남은 시간: {remain.days}일)"
                    else:
                        remain_msg = "🌑 체험 만료됨"

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
        return

    # 2. 비로그인 상태 (로그인창)
    st.markdown("### 🔐 LDY Pro Trader")
    tab1, tab2, tab3 = st.tabs(["로그인", "회원가입", "비밀번호 찾기"])

    # [로그인 탭]
    with tab1:
        with st.form("login_form"):
            login_id = st.text_input("이메일")
            login_pw = st.text_input("비밀번호", type="password")
            if st.form_submit_button("로그인", type="primary"):
                # 마스터 키
                if login_id == MASTER_ADMIN_ID and login_pw == MASTER_ADMIN_PW:
                    st.session_state[CURRENT_USER_KEY] = MASTER_ADMIN_ID
                    st.rerun()
                
                # 일반 유저 로그인
                ok, msg = check_rate_limit(login_id)
                if not ok: 
                    st.error(msg)
                else:
                    user_data = db.get_user_by_id(login_id)
                    if user_data:
                        # 비번 검증
                        salt = user_data.get('salt')
                        pw_hash = user_data.get('password')
                        if _hash_password(login_pw, salt) == pw_hash:
                            db.update_login_timestamp(login_id)
                            st.session_state[CURRENT_USER_KEY] = login_id
                            st.success("환영합니다!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("비밀번호가 일치하지 않습니다.")
                    else:
                        st.error("존재하지 않는 계정입니다.")

    # [회원가입 탭]
    with tab2:
        st.info("🎁 가입 즉시 **7일간 프라임 기능**을 무료로 드립니다!")
        with st.form("signup_form"):
            new_id = st.text_input("이메일")
            new_nick = st.text_input("닉네임")
            new_pw = st.text_input("비밀번호 (6자 이상)", type="password")
            new_pw2 = st.text_input("비밀번호 확인", type="password")
            
            q_idx = st.selectbox("보안질문", range(len(SECURITY_QUESTIONS)), format_func=lambda x: SECURITY_QUESTIONS[x])
            q_ans = st.text_input("답변")
            
            if st.form_submit_button("가입하기"):
                if new_pw != new_pw2:
                    st.error("비밀번호가 일치하지 않습니다.")
                elif len(new_pw) < 6:
                    st.error("비밀번호는 6자 이상이어야 합니다.")
                elif not q_ans:
                    st.error("보안 질문 답변을 입력해주세요.")
                else:
                    # 암호화
                    salt = _create_salt()
                    pw_hash = _hash_password(new_pw, salt)
                    a_hash = _hash_answer(q_ans, salt)
                    
                    # 🔥 [핵심] 여기서 db_utils의 register_user를 호출해야 
                    # 7일 무료 혜택이 적용됩니다!
                    ok, msg = db.register_user(new_id, pw_hash, salt, new_nick, q_idx, a_hash)
                    
                    if ok:
                        st.success(msg)
                        st.balloons()
                        time.sleep(2)
                        st.session_state[CURRENT_USER_KEY] = new_id
                        st.rerun()
                    else:
                        st.error(msg)

    # [비번 찾기 탭]
    with tab3:
        if "fs" not in st.session_state: st.session_state.fs = 1
        
        if st.session_state.fs == 1:
            with st.form("find_step1"):
                f_email = st.text_input("가입한 이메일")
                if st.form_submit_button("확인"):
                    u = db.get_user_by_id(f_email)
                    if u:
                        st.session_state.f_email = f_email
                        st.session_state.f_salt = u.get('salt')
                        st.session_state.f_q_idx = u.get('security_q_idx')
                        st.session_state.f_a_hash = u.get('security_a_hash')
                        st.session_state.fs = 2
                        st.rerun()
                    else:
                        st.error("계정을 찾을 수 없습니다.")
        
        elif st.session_state.fs == 2:
            q_text = SECURITY_QUESTIONS[st.session_state.f_q_idx] if st.session_state.f_q_idx < len(SECURITY_QUESTIONS) else "질문 오류"
            st.info(f"질문: {q_text}")
            with st.form("find_step2"):
                f_ans = st.text_input("답변")
                f_new_pw = st.text_input("새 비밀번호", type="password")
                if st.form_submit_button("변경하기"):
                    input_a_hash = _hash_answer(f_ans, st.session_state.f_salt)
                    if input_a_hash == st.session_state.f_a_hash:
                        new_salt = _create_salt()
                        new_pw_hash = _hash_password(f_new_pw, new_salt)
                        db.update_password(st.session_state.f_email, new_pw_hash, new_salt)
                        
                        st.success("비밀번호가 변경되었습니다.")
                        st.session_state.fs = 1
                        st.rerun()
                    else:
                        st.error("답변이 틀렸습니다.")
            if st.button("취소"):
                st.session_state.fs = 1
                st.rerun()
