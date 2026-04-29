# -*- coding: utf-8 -*-
"""
LDY Pro Trader Auth System v3.0 (Security Review Patch)
───────────────────────────────────────────────────────
v3.0 개선사항:
  #1. 마스터 관리자 Brute-force 방어 — rate limit 최우선 적용
  #2. Salt Rotation 시 보안답변 동시 재해싱 — 복구 무한루프 방지
  #3. 세션 캐시 TTL(120s) + DB 경량 재검증 — 실시간 밴/강등 반영
  보너스. 회원가입 시 normalize_email → 도메인 검증 순서 정리
"""
import streamlit as st
import hashlib
import secrets
import time
import re  # 👈 추가됨
import os
import logging
from datetime import datetime, timezone

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auth_user")

# 상수 정의
CURRENT_USER_KEY = "ldy_current_user"
MASTER_ADMIN_ID = "admin"

# [핵심 수정 1] 비밀번호 로드 로직 강화 (secrets.toml 없는 환경에서도 안전)
try:
    _raw_pw = st.secrets.get("MASTER_ADMIN_PW") or st.secrets.get("auth", {}).get("master_admin_pw", "")
except Exception:
    _raw_pw = os.environ.get("MASTER_ADMIN_PW", "")
_raw_pw = str(_raw_pw).strip() if _raw_pw else ""

if not _raw_pw:
    logger.warning("⚠️ MASTER_ADMIN_PW 미설정 — 관리자 로그인 비활성화됨 (Streamlit Secrets 또는 환경변수에 설정 필요)")

# [v2.0] 관리자 비밀번호를 메모리에 해시로만 보관 (평문 제거)
_ADMIN_PW_HASH = hashlib.sha256(_raw_pw.encode()).hexdigest() if _raw_pw else ""
_ADMIN_PW_SET = bool(_raw_pw)  # 설정 여부 판별용 (평문 아님)

# ✅ [v14] #18: 평문 즉시 제거 — 전역에 평문 없음
del _raw_pw

def _verify_admin_password(input_pw: str) -> bool:
    """[v2.0] 관리자 비밀번호를 해시로 비교 (timing-safe)"""
    if not _ADMIN_PW_HASH:
        return False
    input_hash = hashlib.sha256(input_pw.encode()).hexdigest()
    return secrets.compare_digest(input_hash, _ADMIN_PW_HASH)

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
        from db_utils import get_db as _get_singleton_db
        return _get_singleton_db()
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
    """
    [v3.0 #3] Zero-Leak Caching + TTL 기반 권한 재검증
    
    민감 정보(PW, Salt)를 제거한 프로필만 세션 캐싱하되,
    일정 시간(SESSION_RECHECK_SEC)마다 DB에서 is_banned/role을 재확인.
    → 관리자가 유저를 밴/강등해도 세션 무한 캐시 악용 방지
    """
    SESSION_RECHECK_SEC = 120  # 2분마다 DB 재검증

    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None
    
    val = st.session_state[CURRENT_USER_KEY]
    if not val: return None
    
    # 세션에 딕셔너리가 있으면 TTL 체크
    if isinstance(val, dict):
        cached_at = val.get("_cached_at", 0)
        if time.time() - cached_at < SESSION_RECHECK_SEC:
            return val  # TTL 이내 → 캐시 그대로 반환
        
        # TTL 만료 → is_banned / role만 DB에서 경량 재검증
        login_id = val.get("login_id") or val.get("id")
        if login_id and login_id != MASTER_ADMIN_ID:
            db = get_db()
            if db:
                try:
                    fresh = db.get_user_by_id(login_id)
                    if not fresh:
                        # 계정 삭제됨 → 세션 무효화
                        st.session_state[CURRENT_USER_KEY] = None
                        return None
                    if str(fresh.get("is_banned", "")).upper() in ["Y", "TRUE", "1"]:
                        st.session_state[CURRENT_USER_KEY] = None
                        return None
                    # 권한 정보 갱신
                    val["role"] = fresh.get("role", val.get("role", "free"))
                    val["is_banned"] = fresh.get("is_banned")
                    val["prime_expire_date"] = fresh.get("prime_expire_date")
                except Exception:
                    pass  # DB 조회 실패 시 기존 캐시 유지 (가용성 우선)
        
        val["_cached_at"] = time.time()
        st.session_state[CURRENT_USER_KEY] = val
        return val
        
    # 최초 로드 시 (val이 문자열 = login_id) → DB 조회 후 정제 캐싱
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
            "is_banned": raw_user.get("is_banned"),
            "_cached_at": time.time(),
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
    """⚠️ [v22 Step Z] DEPRECATED — db.add_inquiry() 사용 권장.
    
    구버전 호환을 위해 함수는 유지하지만, 내부적으로 항상 False 반환.
    호출 코드는 add_inquiry()로 마이그레이션 필요.
    
    구버전 streamlit dashboard.py에서만 사용되며, NiceGUI main.py에서는 사용 X.
    """
    import logging as _lg
    _lg.getLogger(__name__).warning(
        "save_inquiry_items() DEPRECATED — db.add_inquiry() 사용 필요"
    )
    return False

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


# [v22.5] 신규 가입은 bcrypt로 — services.auth와 동일 정책
def _hash_password_bcrypt(pw):
    """신규 가입/비번변경용 bcrypt 해시. salt 컬럼은 ""로 저장."""
    import bcrypt as _bcrypt
    cost = int(os.environ.get("BCRYPT_COST", "12"))
    return _bcrypt.hashpw(pw.encode(), _bcrypt.gensalt(cost)).decode()

# ### [수정] 이메일 정규화 함수 (Gmail 전용 규칙 적용)
def normalize_email(email):
    email = email.strip().lower()
    if "@" not in email:
        return email
    
    local, domain = email.split("@", 1)
    
    is_gmail = domain in ("gmail.com", "googlemail.com")
    
    # 1. Gmail 전용: 점(.) 제거 (Gmail은 점을 무시함)
    if is_gmail:
        local = local.replace(".", "")
    
    # 2. Gmail 전용: 플러스(+) 태그 제거 (myname+test@ → myname@)
    #    다른 도메인은 + 가 유효한 주소 일부일 수 있으므로 건드리지 않음
    if is_gmail and "+" in local:
        local = local.split("+")[0]
        
    return f"{local}@{domain}"

def check_rate_limit(email, limit=5, lock_sec=600):
    """
    [v3.1] DB 우선 rate limiting — db_utils.get_login_failures() 기반
    
    DB에서 (login_fail_count, lock_until) 튜플을 조회하고,
    DB 불가 시에만 세션(메모리) fallback.
    
    Returns
    -------
    (bool, str) : (통과 여부, 에러 메시지)
    """
    now = datetime.now(timezone.utc)
    
    # 1순위: DB 조회
    db = get_db()
    if db:
        try:
            fail_count, lock_until = db.get_login_failures(email)
            
            if lock_until and now < lock_until:
                remain = int((lock_until - now).total_seconds())
                return False, f"⛔ 로그인 잠금({remain}s 남음)"
            
            # 잠금 시간이 지났으면 통과 (실패 카운트는 record에서 관리)
            return True, ""
        except Exception as e:
            logger.warning(f"DB rate limit 조회 실패, 세션 fallback: {e}")
    
    # 2순위: 세션(메모리) fallback — DB 불가 시에만 사용
    if "login_rl" not in st.session_state:
        st.session_state.login_rl = {}
    cached = st.session_state.login_rl.get(email, {})
    
    lock_until_ts = cached.get("lock_until", 0)
    now_ts = time.time()
    if lock_until_ts and now_ts < lock_until_ts:
        remain = int(lock_until_ts - now_ts)
        return False, f"⛔ 로그인 잠금({remain}s 남음)"
    
    return True, ""

def record_login_failure(email, limit=5, lock_sec=600):
    """
    [v3.1] DB 우선 실패 기록 — db_utils.record_login_failure() 호출
    
    DB 메서드가 내부적으로 fail_count 누적 + 5회 초과 시 10분 잠금 처리.
    DB 불가 시 세션(메모리) fallback으로 최소한의 보호 유지.
    """
    # 1순위: DB 기록
    db = get_db()
    if db:
        try:
            db.record_login_failure(email)
            return  # DB 기록 성공 → 끝
        except Exception as e:
            logger.warning(f"DB 실패 기록 오류, 세션 fallback: {e}")
    
    # 2순위: 세션 fallback
    now_ts = time.time()
    if "login_rl" not in st.session_state:
        st.session_state.login_rl = {}
    rec = st.session_state.login_rl.get(email, {"fails": 0, "lock_until": 0})
    
    rec["fails"] = rec.get("fails", 0) + 1
    if rec["fails"] >= limit:
        rec["lock_until"] = now_ts + lock_sec
        logger.warning(f"🔒 [세션] 계정 잠금: {email} ({limit}회 실패)")
    
    st.session_state.login_rl[email] = rec

def reset_login_failures(email):
    """
    [v3.1] DB + 세션 양쪽의 실패 기록 초기화
    """
    # DB 초기화
    db = get_db()
    if db:
        try:
            db.reset_login_failures(email)
        except Exception:
            pass
    
    # 세션 초기화
    if "login_rl" in st.session_state and email in st.session_state.login_rl:
        st.session_state.login_rl.pop(email, None)

def render_auth_box(show_debug: bool = False):
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
    st.markdown("### 🔐 시스템 로그인")
    t1, t2, t3 = st.tabs(["로그인", "전략군 가입", "계정 복구"])

    # 탭 1: 로그인 (마스터 키 로직 탑재)
    with t1:
        with st.form("login_ultimate"):
            lid = st.text_input("아이디 (또는 이메일)").strip()
            lpw = st.text_input("비밀번호", type="password")
            
            if st.form_submit_button("로그인", type="primary", use_container_width=True):
                start_t = time.time()
                
                # [v3.0 #1] Rate limit는 계정 종류 무관하게 최우선 실행
                # 마스터 관리자도 Brute-force 공격으로부터 동일하게 보호
                rl_key = MASTER_ADMIN_ID if lid == MASTER_ADMIN_ID else normalize_email(lid)
                ok, msg = check_rate_limit(rl_key)
                
                if not ok:
                    st.error(msg)
                
                # ── 마스터 관리자 인증 ──
                elif lid == MASTER_ADMIN_ID:
                    if _ADMIN_PW_SET and _verify_admin_password(lpw):
                        reset_login_failures(rl_key)
                        st.session_state[CURRENT_USER_KEY] = "admin"
                        st.toast("🛡️ 마스터 관리자 로그인 성공")
                        st.rerun()
                    else:
                        record_login_failure(rl_key)
                        time.sleep(max(0, 0.5 - (time.time() - start_t)))
                        st.error("아이디 또는 비밀번호가 일치하지 않습니다.")
                
                # ── 일반 유저 인증 ──
                else:
                    clean_lid = rl_key  # 이미 normalize_email 됨
                    u = db.get_user_by_id(clean_lid)
                    is_banned = u and str(u.get("is_banned")).upper() in ("Y", "TRUE", "1")
                    auth_ok = False

                    if u:
                        stored = u.get("password", "")
                        # [v22.5] bcrypt 또는 legacy pbkdf2
                        if stored.startswith(("$2a$", "$2b$", "$2y$")):
                            try:
                                import bcrypt as _bcrypt
                                auth_ok = _bcrypt.checkpw(lpw.encode(), stored.encode())
                            except (ValueError, TypeError):
                                auth_ok = False
                        else:
                            # legacy pbkdf2 검증
                            if u.get("salt") and _hash_password(lpw, u["salt"]) == stored:
                                auth_ok = True
                                # 차단 계정이 아닐 때만 bcrypt로 자동 업그레이드
                                if not is_banned:
                                    try:
                                        new_hash = _hash_password_bcrypt(lpw)
                                        db.update_user_password(clean_lid, new_hash, "")
                                        u["password"] = new_hash
                                        u["salt"] = ""
                                    except Exception as e:
                                        logger.warning(f"bcrypt 업그레이드 실패 (검증 성공): {e}")
                    else:
                        # 사용자 없음 — timing 균등화
                        _hash_password(lpw, "static_dummy_salt")

                    if u and auth_ok:
                        if is_banned:
                            st.error("🚫 접근 권한이 제한된 계정입니다.")
                        else:
                            reset_login_failures(clean_lid)
                            st.session_state[CURRENT_USER_KEY] = clean_lid
                            st.rerun()
                    else:
                        record_login_failure(clean_lid)
                        time.sleep(max(0, 0.5 - (time.time() - start_t)))
                        st.error("아이디 또는 비밀번호가 일치하지 않습니다.")

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
                # [v3.0 보너스] 정규화를 먼저 수행한 뒤 도메인 검증
                clean_em = normalize_email(em)
                domain = clean_em.split("@")[-1] if "@" in clean_em else ""
                if domain not in ALLOWED_DOMAINS:
                    st.error(f"🚫 허용된 도메인이 아닙니다. ({', '.join(ALLOWED_DOMAINS)})")
                elif not check_password_strength(p1):
                    st.error("⚠️ 비밀번호 정책 미달: 8자 이상, 영문과 숫자를 모두 포함해야 합니다.")
                elif p1 != p2: st.error("비밀번호가 일치하지 않습니다.")
                elif not ans.strip(): st.error("보안 질문 답변은 필수입니다.")
                else:
                    salt = _create_salt()  # 보안답변 hash용
                    # [v22.5] 비밀번호는 bcrypt, 보안답변은 pbkdf2 with salt
                    ok, msg = db.register_user(
                        clean_em,
                        _hash_password_bcrypt(p1),  # bcrypt
                        salt,  # DB salt 컬럼은 보안답변용
                        nk[:8], q_idx, _hash_answer(ans, salt)
                    )
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
                # [v3.0 #2] Salt Rotation: 비번 변경 시 보안답변도 새 salt로 재해싱
                # 이전 버전은 salt만 교체하고 security_ans를 갱신하지 않아
                # 다음 계정 복구 시 보안답변 검증이 영구 불가능했음
                if u and _hash_answer(ans_in, u["salt"]) == u["security_a_hash"]:
                    if check_password_strength(new_pw):
                        new_salt = _create_salt()
                        # [v22.5] 비밀번호는 bcrypt (self-contained), 보안답변은 새 salt로 재해싱
                        new_hash = _hash_password_bcrypt(new_pw)
                        new_ans_hash = _hash_answer(ans_in, new_salt)  # 보안답변도 새 salt로!
                        if db.update_user_password(
                            clean_fid, new_hash, new_salt,
                            new_security_ans=new_ans_hash,
                        ):
                            success = True
                
                # 성공/실패와 무관하게 1초 지연 (계정 존재 여부 은폐)
                time.sleep(max(0, 1.0 - (time.time() - start_t)))
                
                if success:
                    st.success("✅ 인증 성공! 비밀번호가 변경되었습니다. 로그인 탭을 이용하세요.")
                else:
                    st.error("입력하신 정보가 올바르지 않거나 정책에 맞지 않습니다.")
                    if clean_fid: record_login_failure(clean_fid)
                    
    return None # 👈 미로그인 상태일 때 명시적 None 반환
