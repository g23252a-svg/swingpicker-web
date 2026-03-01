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
    user = get_current_user()
    if not user: return "guest"
    role = user.get("role", "free")
    if role == "admin": return "admin"
    expire = user.get("prime_expire_date")
    if expire:
        try:
            exp_dt = datetime.strptime(str(expire).split(" ")[0], "%Y-%m-%d")
            if exp_dt.date() >= datetime.now().date():
                return role
        except Exception as e:
            _logger.warning(f"구독 만료일 파싱 실패: {expire} → {e}")
    return "free" if role in ("pro", "prime") else role
