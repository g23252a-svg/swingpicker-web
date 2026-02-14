# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System (Debug & Robust Version)
"""
import streamlit as st
import hashlib
import secrets
import time
import re  # 👈 추가됨
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

# ----------------- 1. 보안 정책 강화 -----------------
def check_password_strength(pw: str) -> bool:
    """[v11.0] 8자 이상, 영문+숫자 혼합 강제"""
    if len(pw) < 8: return False
    if not re.search(r"[a-z]", pw.lower()): return False
    if not re.search(r"[0-9]", pw): return False
    return True


# ----------------- 2. 핵심 함수 -----------------

def get_user():
    """[채찍: Zero-Leak Caching] 민감 정보(PW, Salt)를 제거한 프로필만 세션 캐싱"""
    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None
    
    val = st.session_state[CURRENT_USER_KEY]
    if not val: return None
    
    # 세션에 이미 딕셔너리가 있다면 즉시 반환 (DB 부하 0)
    if isinstance(val, dict): return val
        
    # 최초 로드 시 정보 정제 후 캐싱
    db = get_db()
    if not db: return None
    
    raw_user = db.get_user_by_id(val) if val != MASTER_ADMIN_ID else \
               {"id": MASTER_ADMIN_ID, "role": "admin", "nickname": "관리자"}
    
    if raw_user:
        # 🚨 중요: 패스워드, 솔트, 보안답변은 세션에서 영구 삭제
        safe_profile = {
            "id": raw_user.get("id"),
            "login_id": raw_user.get("id"),
            "role": raw_user.get("role", "free"),
            "nickname": raw_user.get("nickname"),
            "prime_expire_date": raw_user.get("prime_expire_date"),
            "is_banned": raw_user.get("is_banned")
        }
        st.session_state[CURRENT_USER_KEY] = safe_profile
        return safe_profile
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

def render_auth_box():
    db = get_db()
    if not db:
        st.error("🚨 시스템 보안 엔진 연결 실패")
        return None

    user = get_user()

    # [1] 로그인 완료 상태 UI
    if user:
        with st.sidebar:
            st.markdown(f"### 👋 **{user.get('nickname')}**님")
            role = user.get('role', 'free')
            if role == 'admin': st.success("😎 마스터 관리자")
            else: st.info(f"👑 {role.upper()} 등급 이용 중")
            
            if st.button("로그아웃", type="primary", use_container_width=True):
                st.session_state[CURRENT_USER_KEY] = None
                st.rerun()
        return user

    # [2] 로그인/가입/복구 탭 UI
    st.markdown("### 🔐 LDY Pro Trader Ultimate Security")
    t1, t2, t3 = st.tabs(["로그인", "전략군 가입", "계정 복구"])

    # 탭 1: 로그인 (타이밍 공격 & 계정 노출 차단)
    with t1:
        with st.form("login_ultimate"):
            lid = st.text_input("이메일").strip()
            lpw = st.text_input("비밀번호", type="password")
            if st.form_submit_button("성문 개방", type="primary", use_container_width=True):
                start_t = time.time()
                clean_lid = normalize_email(lid)
                
                # 시도 제한 체크 (DB 연동 권장)
                ok, msg = check_rate_limit(clean_lid)
                if not ok: 
                    st.error(msg)
                else:
                    u = db.get_user_by_id(clean_lid)
                    # [전술] 존재하지 않는 계정이라도 '가짜 해싱'을 돌려 타이밍 공격을 방어함
                    dummy_salt = "static_dummy_salt"
                    provided_hash = _hash_password(lpw, u["salt"] if u else dummy_salt)
                    stored_hash = u["password"] if u else "dummy_match_fail_hash"
                    
                    if u and provided_hash == stored_hash:
                        if str(u.get("is_banned")).upper() in ["Y", "TRUE", "1"]:
                            st.error("🚫 접근 권한이 제한된 계정입니다.")
                        else:
                            reset_login_failures(clean_lid)
                            st.session_state[CURRENT_USER_KEY] = clean_lid
                            st.rerun()
                    else:
                        record_login_failure(clean_lid)
                        # 성공/실패 응답 시간을 0.5초로 통일
                        time.sleep(max(0, 0.5 - (time.time() - start_t)))
                        st.error("이메일 또는 비밀번호가 일치하지 않습니다.")

    # 탭 2: 회원가입 (정책 강화)
    with t2:
        st.info("👋 가입을 환영합니다! (주요 메일 주소만 사용 가능)")
        with st.form("join_ultimate"):
            em = st.text_input("이메일")
            nk = st.text_input("닉네임 (최대 8자)")
            p1 = st.text_input("비밀번호 (8자+, 영문/숫자 필수)", type="password")
            p2 = st.text_input("비밀번호 확인", type="password")
            q_idx = st.selectbox("보안 질문 (비번 분실 시 답변 필수)", range(len(SECURITY_QUESTIONS)), format_func=lambda x: SECURITY_QUESTIONS[x])
            ans = st.text_input("보안 질문 답변")
            
            if st.form_submit_button("전략군 가입 신청"):
                domain = em.split("@")[-1].lower() if "@" in em else ""
                if domain not in ALLOWED_DOMAINS:
                    st.error(f"🚫 허용된 도메인이 아닙니다. ({', '.join(ALLOWED_DOMAINS)})")
                elif not check_password_strength(p1):
                    st.error("⚠️ 비밀번호 정책 미달: 8자 이상, 영문과 숫자를 모두 포함해야 합니다.")
                elif p1 != p2: st.error("비밀번호가 일치하지 않습니다.")
                elif not ans.strip(): st.error("보안 질문 답변은 필수입니다.")
                else:
                    clean_em = normalize_email(em)
                    salt = _create_salt()
                    ok, msg = db.register_user(clean_em, _hash_password(p1, salt), salt, nk[:8], q_idx, _hash_answer(ans, salt))
                    if ok:
                        st.balloons()
                        st.success("🎉 가입 성공! 로그인 탭에서 접속하세요.")
                    else: st.error(msg)

    # 탭 3: 계정 복구 (정보 유출 0% 설계)
    with t3:
        st.caption("등록된 이메일과 보안 답변으로 비밀번호를 재설정합니다.")
        with st.form("recovery_ultimate"):
            fid = st.text_input("가입한 이메일").strip()
            ans_in = st.text_input("가입 시 설정한 보안 답변")
            new_pw = st.text_input("새 비밀번호 (8자+, 영문/숫자)")
            
            st.warning("⚠️ 정보가 일치하지 않으면 재설정되지 않으며, 시도 횟수가 기록됩니다.")
            
            if st.form_submit_button("본인 인증 및 비번 변경"):
                start_t = time.time()
                clean_fid = normalize_email(fid)
                u = db.get_user_by_id(clean_fid)
                
                success = False
                # [Salt Rotation 적용] 비번 변경 시 소금도 새로 발급하여 보안 등급 상향
                if u and _hash_answer(ans_in, u["salt"]) == u["security_ans"]:
                    if check_password_strength(new_pw):
                        new_salt = _create_salt()
                        new_hash = _hash_password(new_pw, new_salt)
                        if db.update_user_password(clean_fid, new_hash, new_salt): # 👈 DB 함수 수정 필요
                            success = True
                
                # 성공/실패와 무관하게 1초 지연 (계정 존재 여부 은폐)
                time.sleep(max(0, 1.0 - (time.time() - start_t)))
                
                if success:
                    st.success("✅ 인증 성공! 비밀번호가 변경되었습니다. 로그인 탭을 이용하세요.")
                else:
                    st.error("입력하신 정보가 올바르지 않거나 정책에 맞지 않습니다.")
                    if clean_fid: record_login_failure(clean_fid)
                    
    return None
