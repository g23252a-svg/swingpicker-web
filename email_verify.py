# -*- coding: utf-8 -*-
"""
email_verify.py — 이메일 인증코드 발송 (Gmail SMTP)
═══════════════════════════════════════════════════════
가입 시 6자리 인증코드를 발송하고 검증합니다.

환경변수:
    GMAIL_USER: 발송용 Gmail 주소 (예: myapp@gmail.com)
    GMAIL_APP_PW: Gmail 앱 비밀번호 (16자리)

Gmail 앱 비밀번호 발급 방법:
    1. Google 계정 → 보안 → 2단계 인증 활성화
    2. 앱 비밀번호 생성 → "메일" 선택 → 16자리 비밀번호 복사
"""
import logging
import os
import random
import smtplib
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

_logger = logging.getLogger("email_verify")

# ── Gmail SMTP 설정 ──
GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PW = os.environ.get("GMAIL_APP_PW", "")
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# ── 인증코드 저장소 (메모리) ──
# { "email": {"code": "123456", "expires": timestamp, "attempts": 0} }
_codes: dict = {}
_lock = threading.Lock()

CODE_EXPIRE_SEC = 300  # 5분
MAX_ATTEMPTS = 5       # 최대 검증 시도 횟수
MAX_SEND_PER_EMAIL = 3 # 같은 이메일 연속 발송 제한


def is_configured() -> bool:
    """Gmail SMTP가 설정되어 있는지 확인"""
    return bool(GMAIL_USER and GMAIL_APP_PW)


def generate_code() -> str:
    """6자리 인증코드 생성"""
    return str(random.randint(100000, 999999))


def send_verification_email(to_email: str) -> tuple[bool, str]:
    """
    인증코드를 이메일로 발송합니다.

    Returns:
        (True, "코드 발송 완료") or (False, "에러 메시지")
    """
    if not is_configured():
        _logger.warning("Gmail SMTP 미설정 (GMAIL_USER, GMAIL_APP_PW)")
        return False, "이메일 인증 서비스가 설정되지 않았습니다."

    to_email = to_email.strip().lower()

    # 연속 발송 제한 체크
    with _lock:
        existing = _codes.get(to_email)
        if existing and existing.get("send_count", 0) >= MAX_SEND_PER_EMAIL:
            if time.time() - existing.get("first_send", 0) < CODE_EXPIRE_SEC:
                return False, "⚠️ 잠시 후 다시 시도해주세요."

    code = generate_code()

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[SwingPicker] 이메일 인증코드: {code}"
        msg["From"] = f"SwingPicker <{GMAIL_USER}>"
        msg["To"] = to_email

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
        text = f"SwingPicker 인증코드: {code} (5분 이내 입력)"

        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PW)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())

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

    except smtplib.SMTPAuthenticationError:
        _logger.error("Gmail 인증 실패 — GMAIL_APP_PW 확인 필요")
        return False, "이메일 발송 실패 (서버 인증 오류)"
    except Exception as e:
        _logger.error(f"이메일 발송 실패: {e}", exc_info=True)
        return False, f"이메일 발송 실패: {str(e)[:50]}"


def verify_code(email: str, code: str) -> tuple[bool, str]:
    """
    인증코드를 검증합니다.

    Returns:
        (True, "인증 성공") or (False, "에러 메시지")
    """
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
    """만료된 코드 정리 (주기적 호출용)"""
    now = time.time()
    with _lock:
        expired = [k for k, v in _codes.items() if now > v["expires"]]
        for k in expired:
            del _codes[k]
