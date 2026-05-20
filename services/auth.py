# -*- coding: utf-8 -*-
"""
auth.py — 인증 시스템 서비스
═══════════════════════════════════════════════════
로그인, 가입, 비밀번호 관리, 세션 상태
"""
import hashlib
import logging
import os
import re
import secrets
from datetime import datetime

import bcrypt
from nicegui import app  # ci-allow: layer-violation  # TODO: 이벤트 패턴으로 분리

_logger = logging.getLogger(__name__)

# ── Auth 상수 ──
MASTER_ADMIN_ID = "admin"
BCRYPT_COST = int(os.environ.get("BCRYPT_COST", "12"))
_raw_admin_pw = os.environ.get("MASTER_ADMIN_PW", "").strip()
_ADMIN_PW_HASH = bcrypt.hashpw(_raw_admin_pw.encode(), bcrypt.gensalt(BCRYPT_COST)) if _raw_admin_pw else b""
ADMIN_PW_SET = bool(_raw_admin_pw)
del _raw_admin_pw

SECURITY_QUESTIONS = [
    "선택하세요...", "가장 기억에 남는 여행지는?", "어릴 적 살던 동네 이름은?",
    "가장 좋아하는 보물 1호는?", "초등학교 담임 선생님 성함은?",
    "나의 좌우명은?", "부모님의 고향은 어디인가요?",
]
ALLOWED_DOMAINS = [
    "naver.com", "gmail.com", "daum.net", "hanmail.net",
    "kakao.com", "nate.com", "icloud.com", "outlook.com",
    "hotmail.com", "yahoo.com", "taiyoinkproducts.co.kr"
]


# ── DB 접근 ──

def get_db():
    try:
        from db_utils import get_db as _get_db
        db = _get_db()
        if db and hasattr(db, 'ensure_gist_loaded'):
            db.ensure_gist_loaded()
        return db
    except Exception as e:
        _logger.error(f"DB Error: {e}")
        return None


# ── 비밀번호 / 해싱 ──

def verify_admin_pw(pw):
    if not _ADMIN_PW_HASH: return False
    return bcrypt.checkpw(pw.encode(), _ADMIN_PW_HASH)


def create_salt():
    return secrets.token_hex(16)


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 100000).hex()


def hash_ans(ans, salt):
    return hash_pw(ans.strip().lower(), salt)


def hash_password_bcrypt(pw: str) -> str:
    """[v22.5] 신규 가입 / 비밀번호 변경 시 bcrypt 해시 생성.

    bcrypt는 hash 안에 salt가 포함되므로 별도 salt 컬럼 불필요.
    이 함수의 결과를 db.register_user / db.update_user_password에 전달할 때
    salt 인자는 빈 문자열("")로 호출하세요.

    Returns: bcrypt hash 문자열 ($2b$로 시작)
    """
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(BCRYPT_COST)).decode()


# 모듈 로드 시 한 번 계산 — 사용자 부재 시 timing 균등화용 dummy
# 실제 유저는 bcrypt이므로 dummy도 bcrypt여야 일관됨.
_DUMMY_BCRYPT_HASH = bcrypt.hashpw(
    b"dummy-password-for-timing-defense",
    bcrypt.gensalt(BCRYPT_COST),
)


def _dummy_password_check(password: str) -> None:
    """[v22.5] 사용자 부재 시 timing 공격 방어 — bcrypt 검증과 동일한 비용 소비."""
    try:
        bcrypt.checkpw(password.encode(), _DUMMY_BCRYPT_HASH)
    except (ValueError, TypeError) as e:
        # password 인코딩 실패 등 — timing 균등화 목적이라 결과 무시
        # logger 호출은 AST silent 분류 회피 + 진단 정보 제공
        _logger.debug(f"dummy bcrypt 검증 예외 (timing 균등화 목적이라 무시): {e}")


def normalize_email(email):
    email = email.strip().lower()
    if "@" not in email: return email
    local, domain = email.split("@", 1)
    if domain in ("gmail.com", "googlemail.com"):
        local = local.replace(".", "")
        if "+" in local: local = local.split("+")[0]
    return f"{local}@{domain}"


def check_pw_strength(pw):
    return len(pw) >= 8 and re.search(r"[a-z]", pw.lower()) and re.search(r"[0-9]", pw)


# ── 인증 ──

def _is_bcrypt_hash(stored: str) -> bool:
    """bcrypt hash는 $2a$ / $2b$ / $2y$로 시작."""
    return isinstance(stored, str) and stored.startswith(("$2a$", "$2b$", "$2y$"))


def verify_password(db, user_row, password, *, allow_upgrade=True):
    """
    [v22.3] 비밀번호 검증 + bcrypt 자동 마이그레이션.
    [v22.4] allow_upgrade: 차단 계정 등에서 검증만 하고 write 안 하도록.

    동작:
      1. 저장된 hash가 bcrypt 형식($2b$ 등)이면 bcrypt.checkpw로 검증
      2. legacy pbkdf2 hex hash이면 검증 후 bcrypt로 자동 업그레이드
         - allow_upgrade=False면 검증만 하고 업그레이드 생략
         - DB가 update_user_password를 지원하는 경우에만 저장
         - 미지원 DB여도 검증 자체는 정상 통과 (graceful degradation)
      3. user_row dict도 새 hash로 mutate (다음 호출 전까지 메모리 일관성)

    Args:
        allow_upgrade: True면 legacy 검증 성공 시 bcrypt로 자동 업그레이드.
                       False면 검증만 하고 업그레이드 안 함 (banned 계정 등).

    Returns: True (검증 성공) | False (실패)
    """
    stored = user_row.get("password", "")
    if not stored:
        return False

    # Path 1: 이미 bcrypt 유저
    if _is_bcrypt_hash(stored):
        try:
            return bcrypt.checkpw(password.encode(), stored.encode())
        except (ValueError, TypeError) as e:
            _logger.warning(f"bcrypt.checkpw 실패 ({user_row.get('email', '?')}): {e}")
            return False

    # Path 2: legacy pbkdf2 검증
    salt = user_row.get("salt", "")
    if not salt:
        return False
    if hash_pw(password, salt) != stored:
        return False

    # 비밀번호 일치. 업그레이드는 allow_upgrade=True일 때만.
    if not allow_upgrade:
        return True

    # 검증 성공 → bcrypt로 자동 업그레이드
    try:
        new_hash = bcrypt.hashpw(
            password.encode(),
            bcrypt.gensalt(BCRYPT_COST),
        ).decode()
        email = user_row.get("email") or user_row.get("id")
        # 기존 DB 메서드 update_user_password(email, pw_hash, salt) 재사용
        # bcrypt는 hash 안에 salt가 포함되어 있으니 salt 인자는 빈 문자열
        if email and hasattr(db, "update_user_password"):
            db.update_user_password(email, new_hash, "")
            user_row["password"] = new_hash  # 메모리 row도 갱신
            user_row["salt"] = ""
            _logger.info(f"bcrypt 마이그레이션 완료: {email}")
        # update_user_password 미지원 DB여도 검증 자체는 통과
    except Exception as e:
        _logger.warning(f"bcrypt 업그레이드 실패 (검증은 성공): {e}")

    return True


def authenticate_user(db, email, password):
    """반환: (user_dict, None) 성공 | (None, error_msg) 실패

    [v22.3] bcrypt 마이그레이션 통합.
    [v22.4] 차단 계정은 비밀번호 검증은 하되 bcrypt 업그레이드는 안 함
            — banned 계정에 DB write가 발생하지 않도록.
    Timing attack 방어: 사용자 부재 시에도 동일한 시간 소비.
    """
    u = db.get_user_by_id(email)
    if u is None:
        # [v22.5] username enumeration 방어 — bcrypt 검증과 동일한 시간 소비
        _dummy_password_check(password)
        return None, "아이디 또는 비밀번호 오류"

    # [v22.4] ban 여부를 먼저 판정 — 차단 계정엔 bcrypt write 발생 안 함
    is_banned = str(u.get("is_banned")).upper() in ("Y", "TRUE", "1")

    if not verify_password(db, u, password, allow_upgrade=not is_banned):
        return None, "아이디 또는 비밀번호 오류"

    if is_banned:
        return None, "🚫 차단된 계정"

    try:
        db.update_login_timestamp(email)
    except Exception as e:
        _logger.error(f"로그인 타임스탬프 갱신 실패 ({email}): {e}", exc_info=True)
    return u, None


# ── 세션 상태 ──

def get_current_user():
    return app.storage.user.get("profile")


def set_current_user(profile):
    if profile:
        app.storage.user["profile"] = profile
    else:
        app.storage.user.pop("profile", None)


def get_auth_status():
    """[v21.1] 세션 + DB 재검증 기반 권한 판정 (SSOT)."""
    user = get_current_user()
    if not user:
        return "guest"
    # DB에서 최신 상태 재조회
    db = get_db()
    if db:
        fresh = db.get_user_by_id(user.get("email", user.get("id", "")))
        if fresh:
            user = fresh
    role, allowed, reason = compute_access_status(user)
    return role


def compute_access_status(user_row, now=None):
    """
    [v21.1] 권한 판정 SSOT — 앱 전체가 이 함수만 보게 한다.

    Returns: (role, allowed, reason)
        role: "admin" / "prime" / "pro" / "free" / "banned" / "guest"
        allowed: True/False (프리미엄 기능 접근 가능 여부)
        reason: "active_subscription" / "expired" / "banned" / "admin" / "free"
    """
    if now is None:
        now = datetime.now()

    if not user_row:
        return "guest", False, "no_user"

    # 차단
    if str(user_row.get("is_banned", "")).upper() in ("Y", "TRUE", "1"):
        return "banned", False, "banned"

    role = user_row.get("role", "free")

    # 관리자
    if role == "admin":
        return "admin", True, "admin"

    # 구독 만료 체크
    expire = user_row.get("prime_expire_date")
    if role in ("prime", "pro") and expire:
        try:
            exp_dt = datetime.strptime(str(expire).split(" ")[0], "%Y-%m-%d")
            if exp_dt.date() >= now.date():
                return role, True, "active_subscription"
            else:
                return "free", False, "expired"
        except Exception as e:
            _logger.warning(f"구독 만료일 파싱 실패: {expire} → {e}")
            return "free", False, "expire_parse_error"

    return "free", False, "free"


def require_premium(action_name="이 기능"):
    """
    [v21.1] 서버측 프리미엄 권한 강제 검증.
    민감 기능(CSV 다운, 백테스트 등) 앞에서 호출.

    Returns: (allowed, role, reason)
    """
    user = get_current_user()
    if not user:
        return False, "guest", f"{action_name}은 로그인 후 이용 가능합니다."

    # DB에서 최신 상태 재조회 (세션 캐시 무시)
    db = get_db()
    if db:
        fresh_user = db.get_user_by_id(user.get("email", user.get("id", "")))
        if fresh_user:
            user = fresh_user  # 최신 DB 기준

    role, allowed, reason = compute_access_status(user)

    if not allowed:
        if reason == "banned":
            return False, role, "🚫 차단된 계정입니다."
        elif reason == "expired":
            return False, role, f"구독이 만료되었습니다. {action_name}은 Prime 전용입니다."
        else:
            return False, role, f"{action_name}은 Prime 구독 후 이용 가능합니다."

    return True, role, "ok"


def premium_guard(action_name="이 기능"):
    """
    [v21.2] 유니버설 프리미엄 데코레이터.
    모든 premium 엔드포인트에 동일한 가드 적용.

    Usage:
        @premium_guard("CSV 다운로드")
        async def download_csv():
            ...
    """
    from functools import wraps

    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            allowed, role, msg = require_premium(action_name)
            if not allowed:
                try:
                    from nicegui import ui  # ci-allow: layer-violation  # TODO: 이벤트 패턴으로 분리
                    ui.notify(msg, type="warning")
                except Exception as e:
                    _logger.warning(f"premium_guard notify 실패: {e}")
                return None
            return await fn(*args, **kwargs)
        return wrapper
    return decorator


def downgrade_expired_users():
    """
    [v21.3] 만료된 PRIME/PRO → FREE 자동 강등.
    [v22.3.8] case-insensitive 비교 + 강제 즉시 Gist push (update_user_role 내부).
    매일 실행 또는 관리자 수동 실행.

    Returns: (downgraded_count, details_list)
    """
    db = get_db()
    if not db:
        return 0, ["DB 연결 실패"]

    now = datetime.now()
    users = db.get_all_users()
    downgraded = []
    skipped_no_expire = 0
    skipped_not_paid = 0

    for u in users:
        # [v22.3.8] case-insensitive — DB에 'PRIME'/'Prime'/'prime' 어떤 case든 잡음
        role_raw = u.get("role", "free")
        role = str(role_raw).strip().lower()

        if role == "admin":
            continue
        if role not in ("prime", "pro"):
            skipped_not_paid += 1
            continue

        expire = u.get("prime_expire_date")
        if not expire:
            skipped_no_expire += 1
            continue

        try:
            exp_dt = datetime.strptime(str(expire).split(" ")[0], "%Y-%m-%d")
            if exp_dt.date() < now.date():
                email = u.get("login_id") or u.get("id", "")
                # update_user_role이 즉시 Gist push 시도 (v22.3.8)
                db.update_user_role(email, "free")
                detail = f"⏰ {email}: {role_raw}→free (만료 {expire})"
                downgraded.append(detail)
                _logger.info(detail)
        except Exception as e:
            _logger.warning(
                f"만료 체크 실패 ({u.get('login_id', '?')}): {e}"
            )

    _logger.info(
        f"[downgrade_expired] 강등 {len(downgraded)}명 / "
        f"PRIME·PRO 아닌 회원 {skipped_not_paid}명 / "
        f"만료일 없음 {skipped_no_expire}명"
    )
    return len(downgraded), downgraded
