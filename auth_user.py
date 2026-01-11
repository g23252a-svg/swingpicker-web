# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System (AttributeError Fix Version)
"""
import streamlit as st
import hashlib
import secrets
import time
import logging
from datetime import datetime, timezone

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auth_user")

# 상수 정의
CURRENT_USER_KEY = "ldy_current_user"
MASTER_ADMIN_ID = "admin"
MASTER_ADMIN_PW = st.secrets.get("auth", {}).get("master_admin_pw", "")

SECURITY_QUESTIONS = [
    "선택하세요...", "가장 기억에 남는 여행지는?", "어릴 적 살던 동네 이름은?",
    "가장 좋아하는 보물 1호는?", "초등학교 담임 선생님 성함은?",
    "나의 좌우명은?", "부모님의 고향은 어디인가요?",
]

# ----------------- 1. DB 지연 연결 (순환 참조 방지) -----------------
def get_db():
    try:
        from db_utils import LDYDBManager
        return LDYDBManager()
    except Exception as e:
        logger.error(f"DB Load Error: {e}")
        return None

# ----------------- 2. 핵심 함수 -----------------

def get_user():
    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None
    
    val = st.session_state[CURRENT_USER_KEY]
    if not val:
        return None
    
    if isinstance(val, str):
        if val == MASTER_ADMIN_ID:
            return {"id": MASTER_ADMIN_ID, "login_id": MASTER_ADMIN_ID, "role": "admin", "nickname": "관리자"}
        db = get_db()
        if db:
            user_info = db.get_user_by_id(val)
            if user_info:
                user_info['login_id'] = user_info['id']
                return user_info
    
    if isinstance(val, dict):
        return val
        
    return None

def _now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ----------------- 3. 데이터 관리 함수들 -----------------

def list_users():
    db = get_db()
    if not db:
        # 핵심: DB 실패를 "회원 없음"으로 숨기지 말고 화면에서 보이게
        st.error("❌ DB 연결 실패: 회원 목록을 불러올 수 없습니다.")
        return None
    return db.get_all_users()

def update_user_role(email, new_role, acting_admin="system"):
    db = get_db()
    return db.update_user_role(email, new_role) if db else False

def toggle_user_ban(email, acting_admin="system"):
    db = get_db()
    return db.toggle_user_ban(email) if db else False

def load_inquiry_items():
    db = get_db()
    return db.get_all_inquiries() if db else []

def save_inquiry_items(items):
    db = get_db()
    return db.save_inquiries(items) if db else False

def load_subscriptions_db():
    db = get_db()
    if not db: return {"subs": {}}
    users = db.get_all_users()
    subs_map = {}
    for u in users:
        email = u.get('id')
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
    db = get_db()
    if not db: return False
    subs = db_dict.get("subs", {})
    for email, info in subs.items():
        if info.get("role") and info.get("expire_at"):
            db.update_user_subscription(email, info["role"], info["expire_at"])
    return True

# [New] 전체 유저 이벤트 함수 (반드시 있어야 함)
def grant_all_users_trial(days=7):
    db = get_db()
    return db.grant_all_users_trial(days) if db else (False, "DB Error")

# ----------------- 4. 보안 및 UI -----------------

def _create_salt(): return secrets.token_hex(16)
def _hash_password(pw, salt): return hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 100000).hex()
def _hash_answer(ans, salt): return _hash_password(ans.strip().lower(), salt)

def check_rate_limit(email, limit=5, window_sec=300, lock_sec=600):
    """
    limit: window_sec 동안 허용 실패 횟수
    window_sec: 실패 카운트 유지 시간(예: 5분)
    lock_sec: limit 초과 시 잠금 시간(예: 10분)
    """
    if "login_rl" not in st.session_state:
        st.session_state.login_rl = {}

    now = time.time()
    rec = st.session_state.login_rl.get(email, {"fails": 0, "first_ts": now, "lock_until": 0})

    # 잠금 상태면 바로 차단
    if now < rec.get("lock_until", 0):
        remain = int(rec["lock_until"] - now)
        return False, f"⛔ 로그인 잠금({remain}s 남음)"

    # 윈도우 만료면 초기화
    if now - rec.get("first_ts", now) > window_sec:
        rec = {"fails": 0, "first_ts": now, "lock_until": 0}

    # 아직 시도 가능
    st.session_state.login_rl[email] = rec
    return True, ""

def record_login_failure(email, limit=5, window_sec=300, lock_sec=600):
    if "login_rl" not in st.session_state:
        st.session_state.login_rl = {}
    now = time.time()
    rec = st.session_state.login_rl.get(email, {"fails": 0, "first_ts": now, "lock_until": 0})

    # 윈도우 만료면 새로 시작
    if now - rec.get("first_ts", now) > window_sec:
        rec = {"fails": 0, "first_ts": now, "lock_until": 0}

    rec["fails"] = rec.get("fails", 0) + 1

    # 초과하면 잠금
    if rec["fails"] >= limit:
        rec["lock_until"] = now + lock_sec

    st.session_state.login_rl[email] = rec


def reset_login_failures(email):
    if "login_rl" in st.session_state and email in st.session_state.login_rl:
        st.session_state.login_rl.pop(email, None)

def render_auth_box(show_debug=False):
    db = get_db()
    if not db:
        st.error("시스템 연결 실패")
        return None

    user = get_user()

    if user:
        user_id = user['id']
        role = user.get('role', 'free')
        nickname = user.get('nickname', user_id)
        
        if role == 'admin':
            st.sidebar.success("😎 관리자 모드")
            db = get_db()
            if not db:
                st.sidebar.error("DB 연결 실패 (db_utils / secrets / 경로 확인)")
            else:
                try:
                    cnt = len(db.get_all_users() or [])
                    st.sidebar.info(f"DB 연결 OK · 회원수: {cnt}")
                except Exception as e:
                    st.sidebar.error(f"DB 조회 실패: {e}")
        else:
            st.sidebar.markdown(f"### 👋 **{nickname}**님")
            expire = user.get('prime_expire_date')
            if role in ['prime', 'pro'] and expire:
                 st.sidebar.info(f"👑 {role.upper()} 회원")
            else:
                 st.sidebar.text("무료 회원")
                 st.sidebar.button("멤버십 구독하기")

        if st.sidebar.button("로그아웃", type="primary"):
            st.session_state[CURRENT_USER_KEY] = None
            st.rerun()
        
        return user

    st.markdown("### 🔐 LDY Pro Trader")
    tab1, tab2, tab3 = st.tabs(["로그인", "회원가입", "비번찾기"])

    with tab1:
        with st.form("login"):
            lid = st.text_input("이메일")
            lpw = st.text_input("비밀번호", type="password")
            if st.form_submit_button("로그인", type="primary"):
                if lid == MASTER_ADMIN_ID and MASTER_ADMIN_PW and lpw == MASTER_ADMIN_PW:
                    reset_login_failures(lid)
                    st.session_state[CURRENT_USER_KEY] = MASTER_ADMIN_ID
                    st.rerun()
                ok, msg = check_rate_limit(lid, limit=5, window_sec=300, lock_sec=600)
                if not ok:
                    st.error(msg)
                else:
                    u = db.get_user_by_id(lid)
            
                    # 2) 유저 존재 여부
                    if not u:
                        record_login_failure(lid, limit=5, window_sec=300, lock_sec=600)
                        st.error("실패")
                    else:
                        # 3) BAN 체크 (DB 컬럼명이 다를 수 있어서 최대한 안전하게)
                        banned = (
                            u.get("is_banned") is True
                            or u.get("banned") is True
                            or str(u.get("is_banned", "")).upper() in ["Y", "TRUE", "1"]
                            or str(u.get("banned", "")).upper() in ["Y", "TRUE", "1"]
                        )
                        if banned:
                            st.error("⛔ 이용 제한 계정입니다. 관리자에게 문의하세요.")
                            # 밴은 비번 대입 시도 자체를 막는게 목적이라 실패 카운트는 선택
                            # 원하면 아래 한 줄 유지, 싫으면 삭제
                            record_login_failure(lid, limit=5, window_sec=300, lock_sec=600)
                        else:
                            # 4) 비밀번호 검증
                            if _hash_password(lpw, u["salt"]) == u["password"]:
                                reset_login_failures(lid)
                                st.session_state[CURRENT_USER_KEY] = lid
                                st.success("로그인 성공")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                record_login_failure(lid, limit=5, window_sec=300, lock_sec=600)
                                st.error("실패")

    with tab2:
        st.info("🎁 가입 시 7일 무료!")
        with st.form("join"):
            em = st.text_input("이메일")
            nk = st.text_input("닉네임")
            p1 = st.text_input("비밀번호 (6자+)", type="password")
            p2 = st.text_input("비밀번호 확인", type="password")
            q = st.selectbox("질문", range(len(SECURITY_QUESTIONS)), format_func=lambda x: SECURITY_QUESTIONS[x])
            ans = st.text_input("답변")
            if st.form_submit_button("가입"):
                if p1 != p2: st.error("불일치")
                elif len(p1) < 6: st.error("짧음")
                else:
                    salt = _create_salt()
                    ok, msg = db.register_user(em, _hash_password(p1, salt), salt, nk, q, _hash_answer(ans, salt))
                    if ok:
                        st.balloons()
                        st.success(msg)
                        st.session_state[CURRENT_USER_KEY] = em
                        time.sleep(1)
                        st.rerun()
                    else: st.error(msg)
    
    with tab3:
        fid = st.text_input("아이디 (비번 찾기)")
        if st.button("확인"):
            u = db.get_user_by_id(fid)
            if u: st.success(f"질문: {SECURITY_QUESTIONS[u['security_q_idx']]}")
            else: st.error("없음")
            
    return None
