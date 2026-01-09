# auth_user.py (Dashboard Compatibility Patch)

import streamlit as st
import hashlib
import secrets
import time
import logging
from datetime import datetime, timezone
from db_utils import LDYDBManager

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

# --- Helper Functions ---
def _now_utc_str() -> str:
    return datetime.now(timezone.utc).isoformat()

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

# --- 🚨 Dashboard 호환용 함수들 (복원됨) ---

def get_user():
    return st.session_state.get(CURRENT_USER_KEY)

def list_users():
    """관리자 페이지용 유저 리스트 반환"""
    return db.get_all_users()

def update_user_role(email, new_role, acting_admin="system"):
    """관리자 페이지용 권한 변경"""
    return db.update_user_role(email, new_role)

def toggle_user_ban(email, acting_admin="system"):
    """관리자 페이지용 차단 토글"""
    return db.toggle_user_ban(email)

def load_inquiry_items():
    """문의 게시판 로드"""
    return db.get_all_inquiries()

def save_inquiry_items(items):
    """문의 게시판 저장"""
    return db.save_inquiries(items)

def load_subscriptions_db():
    """
    dashboard.py가 기대하는 형태: {'subs': {'email': {'role':..., 'expire_at':...}}}
    DB에서 읽어서 이 형태로 변환해줌 (Adapter Pattern)
    """
    users = db.get_all_users()
    subs_map = {}
    for u in users:
        email = u['id']
        expire = u.get('prime_expire_date')
        if expire:
            # datetime -> YYYY-MM-DD 문자열
            exp_str = str(expire).split(" ")[0]
            subs_map[email] = {
                "role": u.get('role', 'free'),
                "expire_at": exp_str,
                "paid_at": str(u.get('join_date', '')).split(" ")[0]
            }
    return {"subs": subs_map}

def save_subscriptions_db(db_dict):
    """
    dashboard.py가 수정한 subs 딕셔너리를 받아서 DB에 반영
    """
    subs = db_dict.get("subs", {})
    for email, info in subs.items():
        role = info.get("role")
        expire = info.get("expire_at")
        if role and expire:
            db.update_user_subscription(email, role, expire)
    return True

# --- Main UI ---
def render_auth_box(show_debug=False):
    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None
    user = st.session_state[CURRENT_USER_KEY]

    # 로그인 상태
    if user:
        user_id = user if isinstance(user, str) else user.get('id')
        if user_id == MASTER_ADMIN_ID:
            st.sidebar.success("😎 관리자 모드")
        else:
            u_info = db.get_user_by_id(user_id)
            if not u_info:
                st.session_state[CURRENT_USER_KEY] = None
                st.rerun()
            
            nick = u_info.get('nickname', user_id)
            expire = u_info.get('prime_expire_date')
            role = u_info.get('role', 'free')
            
            remain_msg = "무료 회원"
            is_prime = False
            
            if role in ['prime', 'pro', 'admin'] and expire:
                try:
                    # DB에서 문자열로 오거나 datetime으로 올 수 있음
                    if isinstance(expire, str):
                        expire = datetime.strptime(expire.split('.')[0], "%Y-%m-%d %H:%M:%S")
                    
                    remain = expire - datetime.now()
                    if remain.total_seconds() > 0:
                        is_prime = True
                        remain_msg = f"👑 {role.upper()} ({remain.days}일 남음)"
                    else:
                        remain_msg = "🌑 이용권 만료됨"
                except: pass

            st.sidebar.markdown(f"### 👋 **{nick}**님")
            if is_prime:
                st.sidebar.success(remain_msg)
            else:
                st.sidebar.info(remain_msg)
                st.sidebar.button("멤버십 구독하기")

        if st.sidebar.button("로그아웃", type="primary"):
            st.session_state[CURRENT_USER_KEY] = None
            st.rerun()
        return user_id

    # 비로그인 상태
    st.markdown("### 🔐 LDY Pro Trader")
    t1, t2, t3 = st.tabs(["로그인", "회원가입", "비밀번호 찾기"])

    with t1:
        with st.form("login_form"):
            lid = st.text_input("아이디")
            lpw = st.text_input("비밀번호", type="password")
            if st.form_submit_button("로그인", type="primary"):
                if lid == MASTER_ADMIN_ID and lpw == MASTER_ADMIN_PW:
                    st.session_state[CURRENT_USER_KEY] = MASTER_ADMIN_ID
                    st.rerun()
                
                u = db.get_user_by_id(lid)
                if u and _hash_password(lpw, u['salt']) == u['password']:
                    db.update_login_timestamp(lid)
                    st.session_state[CURRENT_USER_KEY] = lid
                    st.success("로그인 성공")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("로그인 실패")

    with t2:
        st.info("🎁 가입 즉시 **7일간 프라임 혜택** 무료!")
        with st.form("join_form"):
            nid = st.text_input("아이디")
            nnick = st.text_input("닉네임")
            npw = st.text_input("비밀번호", type="password")
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
                        time.sleep(2)
                        st.rerun()
                    else: st.error(msg)

    with t3: # 비번찾기 (간소화)
        with st.form("find_pw"):
            fid = st.text_input("아이디")
            fans = st.text_input("보안질문 답변")
            fnpw = st.text_input("새 비밀번호", type="password")
            if st.form_submit_button("변경"):
                u = db.get_user_by_id(fid)
                if u and _hash_answer(fans, u['salt']) == u['security_a_hash']:
                    nsalt = _create_salt()
                    nhash = _hash_password(fnpw, nsalt)
                    db.update_password(fid, nhash, nsalt)
                    st.success("변경 완료")
                else:
                    st.error("정보 불일치")
