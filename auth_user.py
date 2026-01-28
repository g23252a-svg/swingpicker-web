# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System (Debug & Robust Version)
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

# [핵심 수정 1] 비밀번호 로드 로직 강화 (문자열 변환 + 공백 제거)
raw_pw = st.secrets.get("MASTER_ADMIN_PW") or st.secrets.get("auth", {}).get("master_admin_pw", "")
MASTER_ADMIN_PW = str(raw_pw).strip() if raw_pw else ""

SECURITY_QUESTIONS = [
    "선택하세요...", "가장 기억에 남는 여행지는?", "어릴 적 살던 동네 이름은?",
    "가장 좋아하는 보물 1호는?", "초등학교 담임 선생님 성함은?",
    "나의 좌우명은?", "부모님의 고향은 어디인가요?",
]

# ### [수정] 허용할 이메일 도메인 리스트 (화이트리스트)
ALLOWED_DOMAINS = [
    "naver.com", "gmail.com", "daum.net", "hanmail.net", 
    "kakao.com", "nate.com", "icloud.com", "outlook.com", "hotmail.com",
    "yahoo.com", "taiyoinkproducts.co.kr" # 회사 메일 등 필요시 추가
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

def grant_all_users_trial(days=7):
    db = get_db()
    return db.grant_all_users_trial(days) if db else (False, "DB Error")

# ----------------- 4. 보안 및 UI -----------------

def _create_salt(): return secrets.token_hex(16)
def _hash_password(pw, salt): return hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 100000).hex()
def _hash_answer(ans, salt): return _hash_password(ans.strip().lower(), salt)

# ### [수정] 이메일 정규화 함수 추가 (Gmail 점/플러스 무시)
def normalize_email(email):
    email = email.strip().lower()
    if "@" not in email:
        return email
    
    local, domain = email.split("@", 1)
    
    # 1. Gmail의 경우 점(.) 제거 (google.mail = googlemail)
    if "gmail.com" in domain:
        local = local.replace(".", "")
    
    # 2. 플러스(+) 태그 제거 (myname+test@ -> myname@)
    if "+" in local:
        local = local.split("+")[0]
        
    return f"{local}@{domain}"

def check_rate_limit(email, limit=5, window_sec=300, lock_sec=600):
    if "login_rl" not in st.session_state:
        st.session_state.login_rl = {}

    now = time.time()
    rec = st.session_state.login_rl.get(email, {"fails": 0, "first_ts": now, "lock_until": 0})

    if now < rec.get("lock_until", 0):
        remain = int(rec["lock_until"] - now)
        return False, f"⛔ 로그인 잠금({remain}s 남음)"

    if now - rec.get("first_ts", now) > window_sec:
        rec = {"fails": 0, "first_ts": now, "lock_until": 0}

    st.session_state.login_rl[email] = rec
    return True, ""

def record_login_failure(email, limit=5, window_sec=300, lock_sec=600):
    if "login_rl" not in st.session_state:
        st.session_state.login_rl = {}
    now = time.time()
    rec = st.session_state.login_rl.get(email, {"fails": 0, "first_ts": now, "lock_until": 0})

    if now - rec.get("first_ts", now) > window_sec:
        rec = {"fails": 0, "first_ts": now, "lock_until": 0}

    rec["fails"] = rec.get("fails", 0) + 1

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
            if not db:
                st.sidebar.error("DB 연결 실패")
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
                input_pw_str = str(lpw).strip()
                
                # 1. 관리자 체크
                if lid == MASTER_ADMIN_ID:
                    if not MASTER_ADMIN_PW:
                        st.error("⚠️ 시스템 오류: 관리자 비밀번호가 설정되지 않았습니다")
                    elif input_pw_str == MASTER_ADMIN_PW:
                        reset_login_failures(lid)
                        st.session_state[CURRENT_USER_KEY] = MASTER_ADMIN_ID
                        st.rerun()
                    else:
                        st.error("비밀번호가 일치하지 않습니다.")
                
                # 2. 일반 유저 체크
                else:
                    # 로그인 시에도 이메일 정규화하여 체크 (가입할 때 정규화했으므로)
                    clean_lid = normalize_email(lid)
                    ok, msg = check_rate_limit(clean_lid, limit=5, window_sec=300, lock_sec=600)
                    
                    if not ok:
                        st.error(msg)
                    else:
                        # 정규화된 이메일로 조회
                        u = db.get_user_by_id(clean_lid)
                        if not u:
                            record_login_failure(clean_lid)
                            st.error("존재하지 않는 계정입니다.")
                        else:
                            banned = str(u.get("is_banned", "")).upper() in ["Y", "TRUE", "1", "TRUE"]
                            if banned:
                                st.error("⛔ 이용 제한 계정입니다.")
                            elif _hash_password(lpw, u["salt"]) == u["password"]:
                                reset_login_failures(clean_lid)
                                st.session_state[CURRENT_USER_KEY] = clean_lid
                                st.success("로그인 성공")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                record_login_failure(clean_lid)
                                st.error("비밀번호가 일치하지 않습니다.")

    with tab2:
        # ### [수정] 문구 변경 (7일 무료 삭제 -> 가입 환영)
        st.info("👋 가입을 환영합니다! (주요 메일 주소만 사용 가능)")
        with st.form("join"):
            em = st.text_input("이메일")
            nk = st.text_input("닉네임")
            p1 = st.text_input("비밀번호 (6자+)", type="password")
            p2 = st.text_input("비밀번호 확인", type="password")
            q = st.selectbox("질문", range(len(SECURITY_QUESTIONS)), format_func=lambda x: SECURITY_QUESTIONS[x])
            ans = st.text_input("답변")
            
            if st.form_submit_button("가입"):
                # 1. 도메인 체크
                domain = em.split("@")[-1].strip().lower()
                if domain not in ALLOWED_DOMAINS:
                    st.error(f"🚫 스팸 방지를 위해 주요 메일({', '.join(ALLOWED_DOMAINS)})로만 가입 가능합니다.")
                elif p1 != p2: 
                    st.error("비밀번호가 일치하지 않습니다.")
                elif len(p1) < 6: 
                    st.error("비밀번호는 6자 이상이어야 합니다.")
                else:
                    # 2. 이메일 정규화 (Gmail 점/플러스 제거)
                    clean_em = normalize_email(em)
                    
                    salt = _create_salt()
                    # 정규화된 이메일로 가입 요청
                    ok, msg = db.register_user(clean_em, _hash_password(p1, salt), salt, nk, q, _hash_answer(ans, salt))
                    
                    if ok:
                        st.balloons()
                        # 문구는 db_utils.py에서 반환하므로 거기서도 수정되었는지 확인 필요
                        st.success(msg) 
                        st.session_state[CURRENT_USER_KEY] = clean_em
                        time.sleep(1)
                        st.rerun()
                    else: st.error(msg)
    
    with tab3:
        fid = st.text_input("아이디 (비번 찾기)")
        if st.button("확인"):
            # 비번 찾기 시에도 정규화된 이메일로 검색
            clean_fid = normalize_email(fid)
            u = db.get_user_by_id(clean_fid)
            if u: st.success(f"질문: {SECURITY_QUESTIONS[u['security_q_idx']]}")
            else: st.error("없음")
            
    return None
