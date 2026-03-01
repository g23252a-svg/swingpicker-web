# -*- coding: utf-8 -*-
"""
email_verify.py — 이메일 인증코드 발송 (Gmail SMTP)
═══════════════════════════════════════════════════════
환경변수:
    GMAIL_USER: 발송용 Gmail 주소
    GMAIL_APP_PW: Gmail 앱 비밀번호 (16자리)
"""
import logging, os, random, smtplib, threading, time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

_logger = logging.getLogger("email_verify")

GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PW = os.environ.get("GMAIL_APP_PW", "")

_codes: dict = {}
_lock = threading.Lock()
CODE_EXPIRE_SEC = 300
MAX_ATTEMPTS = 5

def is_configured() -> bool:
    return bool(GMAIL_USER and GMAIL_APP_PW)

def generate_code() -> str:
    return str(random.randint(100000, 999999))

def send_verification_email(to_email: str) -> tuple:
    if not is_configured():
        return False, "이메일 인증 서비스가 설정되지 않았습니다."
    to_email = to_email.strip().lower()
    with _lock:
        existing = _codes.get(to_email)
        if existing and existing.get("send_count", 0) >= 3:
            if time.time() - existing.get("first_send", 0) < CODE_EXPIRE_SEC:
                return False, "⚠️ 잠시 후 다시 시도해주세요."
    code = generate_code()
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[LDY Pro Trader] 인증코드: {code}"
        msg["From"] = f"LDY Pro Trader <{GMAIL_USER}>"
        msg["To"] = to_email
        html = f"""
        <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:32px;background:#1a1a2e;border-radius:16px;color:white;">
            <h2 style="text-align:center;color:#818CF8;">💎 LDY Pro Trader</h2>
            <p style="text-align:center;color:#9CA3AF;">이메일 인증코드</p>
            <div style="text-align:center;margin:24px 0;">
                <span style="font-size:36px;font-weight:bold;letter-spacing:8px;background:#0f3460;padding:16px 32px;border-radius:12px;color:#60A5FA;">{code}</span>
            </div>
            <p style="text-align:center;color:#6B7280;font-size:14px;">5분 이내에 입력해주세요.</p>
        </div>"""
        msg.attach(MIMEText(f"LDY Pro Trader 인증코드: {code} (5분 이내 입력)", "plain"))
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PW)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())
        with _lock:
            existing = _codes.get(to_email, {})
            sc = existing.get("send_count", 0) + 1 if existing else 1
            fs = existing.get("first_send", time.time()) if existing else time.time()
            _codes[to_email] = {"code": code, "expires": time.time() + CODE_EXPIRE_SEC, "attempts": 0, "send_count": sc, "first_send": fs}
        _logger.info(f"✉️ 인증코드 발송: {to_email[:3]}***")
        return True, "✅ 인증코드가 발송되었습니다."
    except smtplib.SMTPAuthenticationError:
        return False, "이메일 발송 실패 (Gmail 인증 오류)"
    except Exception as e:
        _logger.error(f"이메일 발송 실패: {e}")
        return False, f"이메일 발송 실패"

def verify_code(email: str, code: str) -> tuple:
    email = email.strip().lower()
    with _lock:
        entry = _codes.get(email)
        if not entry:
            return False, "인증코드를 먼저 발송해주세요."
        if time.time() > entry["expires"]:
            del _codes[email]
            return False, "⏰ 인증코드가 만료되었습니다."
        if entry["attempts"] >= MAX_ATTEMPTS:
            del _codes[email]
            return False, "🚫 시도 횟수 초과. 다시 발송해주세요."
        entry["attempts"] += 1
        if entry["code"] == code.strip():
            del _codes[email]
            return True, "✅ 인증 성공!"
        remaining = MAX_ATTEMPTS - entry["attempts"]
        return False, f"❌ 코드 불일치 (남은 시도: {remaining}회)"
