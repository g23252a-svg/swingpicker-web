# -*- coding: utf-8 -*-
"""
payments.py — 결제 API 엔드포인트 (토스페이먼츠 + 빌링)
═══════════════════════════════════════════════════════════
Phase 2: 토스페이먼츠 결제 성공/실패 콜백 처리
Phase 3: 정기 구독 빌링 자동 연장

사용법:
    main.py에서 `register_payment_routes()` 호출
"""
import base64
import logging
import os
from datetime import datetime, timedelta, timezone

from nicegui import app, ui

_logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))

# ── 토스페이먼츠 설정 ──
TOSS_CLIENT_KEY = os.environ.get("TOSS_CLIENT_KEY", "")
TOSS_SECRET_KEY = os.environ.get("TOSS_SECRET_KEY", "")
TOSS_API_URL = "https://api.tosspayments.com/v1/payments/confirm"

# ── Telegram 알림 ──
TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_ID", "")

# ── 구독 기간 (일) ──
SUBSCRIPTION_DAYS = 30

# ── 가격 ──
try:
    from version_info import PRICE_PRO, PRICE_PRIME
except ImportError:
    PRICE_PRO = 19_900
    PRICE_PRIME = 19_900


def _get_db():
    try:
        from db_utils import get_db
        db = get_db()
        if db and hasattr(db, 'ensure_gist_loaded'):
            db.ensure_gist_loaded()
        return db
    except Exception:
        return None


def _send_telegram(text: str):
    if not TG_TOKEN or not TG_CHAT_ID:
        return False
    try:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        _logger.warning(f"TG 알림 실패: {e}")
        return False


def _parse_plan_from_order_id(order_id: str) -> str:
    """주문ID에서 플랜 추출: LDY-PRIME-20260302... → prime"""
    parts = order_id.upper().split("-")
    if len(parts) >= 2:
        plan = parts[1].lower()
        if plan in ("pro", "prime"):
            return plan
    return "pro"


def _confirm_toss_payment(payment_key: str, order_id: str, amount: int) -> dict:
    """
    토스페이먼츠 결제 승인 API 호출
    https://docs.tosspayments.com/reference#결제-승인

    Returns:
        dict with 'success' bool and payment details or error
    """
    if not TOSS_SECRET_KEY:
        return {"success": False, "error": "TOSS_SECRET_KEY 미설정"}

    try:
        import requests

        # Basic Auth: secret_key + ":"  → Base64
        auth_str = base64.b64encode(f"{TOSS_SECRET_KEY}:".encode()).decode()

        resp = requests.post(
            TOSS_API_URL,
            headers={
                "Authorization": f"Basic {auth_str}",
                "Content-Type": "application/json",
            },
            json={
                "paymentKey": payment_key,
                "orderId": order_id,
                "amount": amount,
            },
            timeout=15,
        )

        if resp.status_code == 200:
            data = resp.json()
            return {"success": True, "data": data}
        else:
            error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            return {
                "success": False,
                "error": error_data.get("message", f"HTTP {resp.status_code}"),
                "code": error_data.get("code", "UNKNOWN"),
            }

    except Exception as e:
        _logger.error(f"토스 결제 승인 실패: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _activate_subscription(email: str, plan: str):
    """구독 활성화: DB 등급 변경 + 만료일 설정"""
    db = _get_db()
    if not db:
        _logger.error("DB 연결 실패 — 구독 활성화 불가")
        return False

    expire_date = (datetime.now(KST) + timedelta(days=SUBSCRIPTION_DAYS)).strftime("%Y-%m-%d")
    db.update_user_subscription(email, plan, expire_date)
    _logger.info(f"✅ 구독 활성화: {email} → {plan} (만료: {expire_date})")
    return True


# ═══════════════════════════════════════════════════
#  NiceGUI 라우트 등록
# ═══════════════════════════════════════════════════
def register_payment_routes():
    """
    main.py에서 호출하여 결제 관련 API 엔드포인트를 등록합니다.

    등록 라우트:
        GET /api/payments/toss/success  — 결제 성공 콜백
        GET /api/payments/toss/fail     — 결제 실패 콜백
    """

    @ui.page('/api/payments/toss/success')
    async def toss_success():
        """
        토스페이먼츠 결제 성공 리다이렉트.
        쿼리: ?paymentKey=...&orderId=...&amount=...
        """
        from nicegui import app as _app
        from starlette.requests import Request

        request: Request = _app.native.main_window if hasattr(_app, 'native') else None

        # NiceGUI에서 쿼리파라미터 접근
        params = {}
        try:
            js_result = await ui.run_javascript(
                "new URLSearchParams(window.location.search).toString()"
            )
            if js_result:
                for pair in js_result.split("&"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        params[k] = v
        except Exception:
            pass

        payment_key = params.get("paymentKey", "")
        order_id = params.get("orderId", "")
        amount_str = params.get("amount", "0")

        try:
            amount = int(amount_str)
        except ValueError:
            amount = 0

        if not payment_key or not order_id or not amount:
            ui.label("❌ 결제 정보가 올바르지 않습니다.").classes("text-red-400 text-lg p-8")
            ui.button("홈으로", on_click=lambda: ui.navigate.to("/")).props("color=primary")
            return

        # 결제 승인
        result = _confirm_toss_payment(payment_key, order_id, amount)

        if result["success"]:
            # 플랜 & 이메일 추출
            plan = _parse_plan_from_order_id(order_id)
            payment_data = result.get("data", {})
            email = payment_data.get("customer", {}).get("email", "")

            # 이메일을 orderId에서도 시도
            if not email:
                parts = order_id.split("-")
                if len(parts) >= 4:
                    email = parts[3]  # LDY-PRIME-20260302HHMMSS-email

            if email:
                _activate_subscription(email, plan)

            # 관리자 알림
            _send_telegram(
                f"✅ <b>[결제 완료]</b>\n"
                f"📧 {email}\n"
                f"📦 {plan.upper()} ({amount:,}원)\n"
                f"🆔 {order_id}\n"
                f"💳 {payment_data.get('method', '-')}"
            )

            # 성공 페이지
            with ui.column().classes("w-full items-center p-12"):
                ui.label("✅").classes("text-6xl mb-4")
                ui.label("결제가 완료되었습니다!").classes("text-2xl font-bold text-white mb-2")
                ui.label(f"{plan.upper()} 등급이 활성화되었습니다.").classes("text-green-400 text-lg mb-4")
                ui.label(f"주문번호: {order_id}").classes("text-gray-400 text-sm")
                ui.button("🏠 홈으로 돌아가기", on_click=lambda: ui.navigate.to("/")).props("color=primary size=lg")

        else:
            _send_telegram(
                f"❌ <b>[결제 승인 실패]</b>\n"
                f"🆔 {order_id}\n"
                f"💰 {amount:,}원\n"
                f"❗ {result.get('error', 'unknown')}"
            )

            with ui.column().classes("w-full items-center p-12"):
                ui.label("❌").classes("text-6xl mb-4")
                ui.label("결제 승인에 실패했습니다.").classes("text-2xl font-bold text-white mb-2")
                ui.label(f"사유: {result.get('error', '알 수 없는 오류')}").classes("text-red-400 mb-4")
                with ui.row().classes("gap-4"):
                    ui.button("다시 시도", on_click=lambda: ui.navigate.to("/")).props("color=primary")
                    ui.button("문의하기", on_click=lambda: ui.navigate.to("/")).props("color=gray outlined")

    @ui.page('/api/payments/toss/fail')
    async def toss_fail():
        """토스페이먼츠 결제 실패/취소 리다이렉트"""
        params = {}
        try:
            js_result = await ui.run_javascript(
                "new URLSearchParams(window.location.search).toString()"
            )
            if js_result:
                for pair in js_result.split("&"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        params[k] = v
        except Exception:
            pass

        code = params.get("code", "UNKNOWN")
        message = params.get("message", "결제가 취소되었습니다.")
        order_id = params.get("orderId", "-")

        _logger.info(f"결제 실패/취소: {code} - {message} (order: {order_id})")

        with ui.column().classes("w-full items-center p-12"):
            ui.label("😥").classes("text-6xl mb-4")
            ui.label("결제가 완료되지 않았습니다").classes("text-xl font-bold text-white mb-2")
            ui.label(message).classes("text-gray-400 mb-4")
            ui.button("🏠 홈으로", on_click=lambda: ui.navigate.to("/")).props("color=primary")


# ═══════════════════════════════════════════════════
#  Phase 3: 정기 빌링 스케줄러 (별도 구현 필요)
# ═══════════════════════════════════════════════════
def check_and_renew_subscriptions():
    """
    [Phase 3] 매일 실행되는 구독 자동 갱신 체커.

    로직:
    1. DB에서 prime_expire_date가 내일인 유저 조회
    2. 저장된 빌링키(billing_key)로 자동 결제 요청
    3. 성공 시 expire_date += 30일
    4. 실패 시 알림 발송 + grace period

    호출 방법 (cron / APScheduler):
        from payments import check_and_renew_subscriptions
        scheduler.add_job(check_and_renew_subscriptions, 'cron', hour=9)
    """
    db = _get_db()
    if not db:
        _logger.error("빌링 체크 실패: DB 없음")
        return

    tomorrow = (datetime.now(KST) + timedelta(days=1)).strftime("%Y-%m-%d")

    # TODO Phase 3:
    # 1. DB 스키마에 billing_key 칼럼 추가
    #    ALTER TABLE users ADD COLUMN billing_key TEXT;
    #
    # 2. 만료 임박 유저 조회
    #    SELECT id, role, prime_expire_date, billing_key FROM users
    #    WHERE prime_expire_date = ? AND billing_key IS NOT NULL
    #
    # 3. 토스페이먼츠 빌링 API 호출
    #    POST https://api.tosspayments.com/v1/billing/{billingKey}
    #    { "customerKey": email, "amount": price, "orderId": ... }
    #
    # 4. 성공 → update_user_subscription(email, role, new_expire)
    #    실패 → 유저에게 알림 + 3일 유예 기간

    _logger.info(f"[Phase 3] 빌링 체크 실행 (만료일: {tomorrow}) — 미구현")
