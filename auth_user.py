# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System (순환 참조 완전 해결 버전)
"""
import streamlit as st
import hashlib
import secrets
import time
import logging
from datetime import datetime, timezone

# [중요] db_utils는 파일 맨 위에서 import 하지 않습니다.
# 이렇게 해야 dashboard.py가 이 파일을 불러올 때 막히지 않습니다.

# ----------------- 1. 필수 함수 즉시 정의 (Import 에러 방지용) -----------------

def get_user():
    """현재 로그인된 사용자 세션 정보 반환"""
    if "ldy_current_user" not in st.session_state:
        st.session_state["ldy_current_user"] = None
    return st.session_state["ldy_current_user"]

def _now_utc_str():
    """현재 UTC 시간을 문자열로 반환"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ----------------- 2. DB 지연 연결 (Lazy Loading) -----------------
# DB 연결이 필요할 때만 db_utils를 불러옵니다.

def get_db():
    try:
        from db_utils import LDYDBManager
        return LDYDBManager()
    except Exception as e:
        # DB 연결 실패 시 로그만 남기고 None 반환 (프로그램 중단 방지)
        logging.error(f"DB Load Error: {e}")
        return None

# ----------------- 3. 관리자 및 데이터 함수 -----------------

def list_users():
    """[관리자용] 모든 사용자 목록 조회"""
    db = get_db()
    if db: return db.get_all_users()
    return []

def update_user_role(email, new_role, acting_admin="system"):
    """[관리자용] 사용자 권한 변경"""
    db = get_db()
    if db: return db.update_user_role(email, new_role)
    return False

def toggle_user_ban(email, acting_admin="system"):
    """[관리자용] 사용자 차단/해제 토글"""
    db = get_db()
    if db: return db.toggle_user_ban(email)
    return False

def load_inquiry_items():
    """문의 게시판 글 목록 로드"""
    db = get_db()
    if db: return db.get_all_inquiries()
    return []

def save_inquiry_items(items):
    """문의 게시판 글 저장"""
    db = get_db()
    if db: return db.save_inquiries(items)
    return False

def load_subscriptions_db():
    """[구독관리] 구독 정보 로드"""
    db = get_db()
    if not db: return {"subs": {}}
    
    users = db.get_all_users()
    subs_map = {}
    for u in users:
        email = u.get('id', u.get('login_id'))
        expire = u.get('prime_expire_date')
        if email and expire:
            exp_str = str(expire).split(" ")[0]
            join_str = str(u.get('join_date', '')).split(" ")[0]
            subs_map[email] = {
                "role": u.get('role', 'free'),
                "expire_at": exp_str,
                "paid_at": join_str
            }
    return {"subs": subs_map}

def save_subscriptions_db(db_dict):
    """[구독관리] 구독 정보 저장"""
    db = get_db()
    if not db: return False
    
    subs = db_dict.get("subs", {})
    for email, info in subs.items():
        role = info.get("role")
        expire = info.get("expire_at")
        if role and expire:
            db.update_user_subscription(email, role, expire)
    return True

# ----------------- 4. 보안 및 UI 컴포넌트 -----------------

MASTER_ADMIN_ID = "admin"
MASTER_ADMIN_PW = "2022322"
CURRENT_USER_KEY = "ldy_current_user"

SECURITY_QUESTIONS = [
    "선택하세요...", "가장 기억에 남는 여행지는?", "어릴 적 살던 동네 이름은?",
    "가장 좋아하는 보물 1호는?", "초등학교 담임 선생님 성함은?",
    "나의 좌우명은?", "부모님의 고향은 어디인가요?",
]

def _create_salt():
    return secrets.token_hex(16)

def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()

def _hash_answer(answer, salt):
    return _hash_password(answer.strip().lower(), salt)

def check_rate_limit(email, limit=5):
    if "login_attempts" not in st.session_state: st.session_state.login_attempts = {}
    attempts = st.session_state.login_attempts.get(email, 0)
    if attempts >= limit: return False, "⛔ 시도 횟수 초과. 잠시 후 다시 시도하세요."
    return True, ""

def render_auth_box(show_debug=False):
    db = get_db()
    if not db:
        st.error("DB 연결 중입니다...")
        return None

    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None
    
    user = st.session_state[CURRENT_USER_KEY]

    # [로그인 상태]
    if user:
        user_id = user if isinstance(user, str) else user.get('id')
        
        # 관리자
        if user_id == MASTER_ADMIN_ID:
            st.sidebar.success("😎 관리자 모드")
            if st.sidebar.button("로그아웃", type="primary"):
                st.session_state[CURRENT_USER_KEY] = None
                st.rerun()
            return {"id": MASTER_ADMIN_ID, "login_id": MASTER_ADMIN_ID, "role": "admin", "nickname": "관리자"}

        # 일반 유저
        user_info = db.get_user_by_id(user_id)
        if not user_info:
            st.session_state[CURRENT_USER_KEY] = None
            st.rerun()

        nickname = user_info.get('nickname', user_id)
        expire = user_info.get('prime_expire_date')
        role = user_info.get('role', 'free')
        
        is_prime = False
        remain_msg = "무료 회원"

        if role in ['prime', 'pro', 'admin'] and expire:
            try:
                if isinstance(expire, str):
                    expire_dt = datetime.strptime(expire.split('.')[0], "%Y-%m-%d %H:%M:%S")
                else:
                    expire_dt = expire
                
                remain = expire_dt - datetime.now()
                if remain.total_seconds() > 0:
                    is_prime = True
                    remain_msg = f"👑 {role.upper()} ({remain.days}일 남음)"
                else:
                    remain_msg = "🌑 이용권 만료됨"
            except Exception: pass

        st.sidebar.markdown(f"### 👋 **{nickname}**님")
        if is_prime:
            st.sidebar.success(remain_msg)
        else:
            st.sidebar.info(remain_msg)
            st.sidebar.button("멤버십 구독하기")

        if st.sidebar.button("로그아웃", type="primary"):
            st.session_state[CURRENT_USER_KEY] = None
            st.rerun()
        
        user_info['login_id'] = user_info['id']
        return user_info

    # [비로그인 상태]
    st.markdown("### 🔐 LDY Pro Trader")
    tab1, tab2, tab3 = st.tabs(["로그인", "회원가입", "비밀번호 찾기"])

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
                    ok, msg = db.register_user(nid, ph, salt, nnick, q, ah)
                    if ok:
                        st.balloons()
                        st.success(msg)
                        st.session_state[CURRENT_USER_KEY] = nid
                        time.sleep(1)
                        st.rerun()
                    else: st.error(msg)
    
    with tab3:
        # 비번 찾기 로직 (생략 없이 간단 구현)
        fid = st.text_input("아이디 (비번 찾기)")
        if st.button("확인"):
            u = db.get_user_by_id(fid)
            if u: st.success("보안 질문: " + SECURITY_QUESTIONS[u['security_q_idx']])
            else: st.error("계정 없음")

    return None
