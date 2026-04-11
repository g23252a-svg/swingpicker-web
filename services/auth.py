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
from nicegui import app

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

def authenticate_user(db, email, password):
    """반환: (user_dict, None) 성공 | (None, error_msg) 실패"""
    u = db.get_user_by_id(email)
    h = hash_pw(password, u["salt"] if u else "dummy")
    if not u or h != u.get("password"):
        return None, "아이디 또는 비밀번호 오류"
    if str(u.get("is_banned")).upper() in ("Y", "TRUE", "1"):
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
                    from nicegui import ui
                    ui.notify(msg, type="warning")
                except Exception:
                    pass
                return None
            return await fn(*args, **kwargs)
        return wrapper
    return decorator
