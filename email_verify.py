# -*- coding: utf-8 -*-
"""
email_verify.py — 이메일 인증코드 발송 (Resend API)
═══════════════════════════════════════════════════════
가입 시 6자리 인증코드를 발송하고 검증합니다.

환경변수:
    RESEND_API_KEY: Resend API 키 (https://resend.com)
"""
import logging
import os
import random
import threading
import time

import requests as _requests

_logger = logging.getLogger("email_verify")

# ── Resend API 설정 ──
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
RESEND_URL = "https://api.resend.com/emails"
FROM_EMAIL = "SwingPicker <noreply@ldyprotrader.com>"

# ── 인증코드 저장소 (메모리) ──
_codes: dict = {}
_lock = threading.Lock()

CODE_EXPIRE_SEC = 300  # 5분
MAX_ATTEMPTS = 5
MAX_SEND_PER_EMAIL = 3


def is_configured() -> bool:
    """Resend API가 설정되어 있는지 확인"""
    return bool(RESEND_API_KEY)


def generate_code() -> str:
    return str(random.randint(100000, 999999))


def send_verification_email(to_email: str) -> tuple:
    """
    인증코드를 이메일로 발송합니다. (Resend HTTP API)
    """
    if not is_configured():
        _logger.warning("Resend API 미설정 (RESEND_API_KEY)")
        return False, "이메일 인증 서비스가 설정되지 않았습니다."

    to_email = to_email.strip().lower()

    # 연속 발송 제한
    with _lock:
        existing = _codes.get(to_email)
        if existing and existing.get("send_count", 0) >= MAX_SEND_PER_EMAIL:
            if time.time() - existing.get("first_send", 0) < CODE_EXPIRE_SEC:
                return False, "⚠️ 잠시 후 다시 시도해주세요."

    code = generate_code()

    try:
        html = f"""
        <div style="font-family:sans-serif; max-width:480px; margin:0 auto;
                    padding:32px; background:#1a1a2e; border-radius:16px; color:white;">
            <h2 style="text-align:center; color:#818CF8;">💎 SwingPicker</h2>
            <p style="text-align:center; color:#9CA3AF;">이메일 인증코드</p>
            <div style="text-align:center; margin:24px 0;">
                <span style="font-size:36px; font-weight:bold; letter-spacing:8px;
                             background:#0f3460; padding:16px 32px; border-radius:12px;
                             color:#60A5FA;">{code}</span>
            </div>
            <p style="text-align:center; color:#6B7280; font-size:14px;">
                5분 이내에 입력해주세요.<br>
                본인이 요청하지 않았다면 이 메일을 무시하세요.
            </p>
        </div>
        """

        resp = _requests.post(
            RESEND_URL,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": FROM_EMAIL,
                "to": [to_email],
                "subject": f"[SwingPicker] 이메일 인증코드: {code}",
                "html": html,
            },
            timeout=10,
        )

        if resp.status_code not in (200, 201):
            err_msg = resp.json().get("message", resp.text[:100])
            _logger.error(f"Resend API 오류: {resp.status_code} {err_msg}")
            return False, f"이메일 발송 실패: {err_msg}"

        # 코드 저장
        with _lock:
            existing = _codes.get(to_email, {})
            send_count = existing.get("send_count", 0) + 1 if existing else 1
            first_send = existing.get("first_send", time.time()) if existing else time.time()
            _codes[to_email] = {
                "code": code,
                "expires": time.time() + CODE_EXPIRE_SEC,
                "attempts": 0,
                "send_count": send_count,
                "first_send": first_send,
            }

        _logger.info(f"✉️ 인증코드 발송: {to_email[:3]}***")
        return True, "인증코드가 발송되었습니다. 이메일을 확인해주세요."

    except _requests.Timeout:
        return False, "이메일 발송 시간 초과. 다시 시도해주세요."
    except Exception as e:
        _logger.error(f"이메일 발송 실패: {e}", exc_info=True)
        return False, f"이메일 발송 실패: {str(e)[:50]}"


def verify_code(email: str, code: str) -> tuple:
    """인증코드 검증"""
    email = email.strip().lower()

    with _lock:
        entry = _codes.get(email)

        if not entry:
            return False, "인증코드를 먼저 발송해주세요."

        if time.time() > entry["expires"]:
            del _codes[email]
            return False, "⏰ 인증코드가 만료되었습니다. 다시 발송해주세요."

        if entry["attempts"] >= MAX_ATTEMPTS:
            del _codes[email]
            return False, "🚫 시도 횟수 초과. 다시 발송해주세요."

        entry["attempts"] += 1

        if entry["code"] == code.strip():
            del _codes[email]
            return True, "✅ 인증 성공!"
        else:
            remaining = MAX_ATTEMPTS - entry["attempts"]
            return False, f"❌ 코드 불일치 (남은 시도: {remaining}회)"


def cleanup_expired():
    """만료된 코드 정리"""
    now = time.time()
    with _lock:
        expired = [k for k, v in _codes.items() if now > v["expires"]]
        for k in expired:
            del _codes[k]
