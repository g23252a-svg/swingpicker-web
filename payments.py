# -*- coding: utf-8 -*-
"""
payments.py — 결제 API 엔드포인트 (토스페이먼츠 + 빌링)
═══════════════════════════════════════════════════════════
[v22 Step R] 보안 강화 + 안정성 개선
- Server-side 금액 검증 (위변조 방지)
- Query param 직접 받기 (FastAPI Request)
- 중복 결제 방지 (orderId 중복 체크)
- 결제 영수증 URL 표시
- Order ID 안전 파싱 (URL safe)

사용법:
    main.py에서 `register_payment_routes()` 호출
"""
import base64
import hashlib
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

# [Step R+] 라이브/테스트 모드 자동 감지
# live_* prefix → 운영 모드 (실제 결제 + 정산)
# test_* prefix → 테스트 모드 (실제 결제 X)
TOSS_IS_LIVE = TOSS_CLIENT_KEY.startswith("live_") and TOSS_SECRET_KEY.startswith("live_")
TOSS_IS_TEST = TOSS_CLIENT_KEY.startswith("test_") and TOSS_SECRET_KEY.startswith("test_")
TOSS_MODE = (
    "LIVE" if TOSS_IS_LIVE
    else "TEST" if TOSS_IS_TEST
    else "DISABLED"
)

# ── Telegram 알림 ──
TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_ID", "")

# ── 구독 기간 (일) ──
SUBSCRIPTION_DAYS = 30

# ── 가격 SSOT (위변조 방지용 server-side mapping) ──
# tab_pricing.py와 동일하게 유지!
PRICE_PRIME = 19_900
PRICE_PRO = 19_900  # 호환성

PLAN_PRICES = {
    "prime": PRICE_PRIME,
    "pro": PRICE_PRO,
}

# ── 처리된 주문 ID 캐시 (중복 결제 방지, 메모리 캐시) ──
# Production에서는 Redis 등으로 교체 권장
_processed_orders: set = set()


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


def _parse_order_id(order_id: str) -> dict:
    """[Step R+X] 주문ID 안전 파싱.
    
    포맷: LDY-{PLAN}-{TIMESTAMP}-{EMAIL_HASH12}
    예: LDY-PRIME-20260425143022-a3f9b21c4d5e
    
    [Step X] 이메일 해시는 12자리로 강화됨 (충돌 확률 0.001% → 0.00000003%).
    하위 호환: 8자리 해시도 정상 파싱 (기존 주문 호환).
    
    Returns:
        {"plan": str, "timestamp": str, "hash": str}
    """
    parts = order_id.upper().split("-")
    result = {"plan": "prime", "timestamp": "", "hash": ""}
    if len(parts) >= 2:
        plan = parts[1].lower()
        if plan in PLAN_PRICES:
            result["plan"] = plan
    if len(parts) >= 3:
        result["timestamp"] = parts[2]
    if len(parts) >= 4:
        # 8자리 또는 12자리 모두 정상 처리
        result["hash"] = parts[3].lower()
    return result


def _validate_payment_amount(plan: str, amount: int) -> bool:
    """[Step R] Server-side 금액 검증.
    
    클라이언트가 임의로 amount를 변조할 수 있으니
    plan별 정확한 가격과 일치하는지 확인.
    """
    expected = PLAN_PRICES.get(plan, 0)
    if expected <= 0:
        _logger.warning(f"알 수 없는 플랜: {plan}")
        return False
    if amount != expected:
        _logger.warning(
            f"⚠️ 금액 불일치 — plan={plan}, "
            f"expected={expected}, actual={amount}"
        )
        return False
    return True


def _confirm_toss_payment(payment_key: str, order_id: str, amount: int) -> dict:
    """
    토스페이먼츠 결제 승인 API 호출
    https://docs.tosspayments.com/reference#결제-승인
    
    [Step R+] 라이브 모드일 때 추가 로깅 + 자세한 알림

    Returns:
        dict with 'success' bool and payment details or error
    """
    if not TOSS_SECRET_KEY:
        return {"success": False, "error": "TOSS_SECRET_KEY 미설정"}

    # [Step R+] 라이브 결제 요청 시 명확히 로그 남김
    if TOSS_IS_LIVE:
        _logger.info(
            f"💳 [LIVE 결제 승인 요청] orderId={order_id}, amount={amount:,}원"
        )
    else:
        _logger.info(
            f"🧪 [TEST 결제 승인 요청] orderId={order_id}, amount={amount:,}원"
        )

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
            if TOSS_IS_LIVE:
                _logger.info(
                    f"✅ [LIVE 결제 승인 성공] orderId={order_id}, "
                    f"method={data.get('method')}, "
                    f"approvedAt={data.get('approvedAt')}"
                )
            return {"success": True, "data": data}
        else:
            error_data = (
                resp.json()
                if resp.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            _logger.warning(
                f"❌ [{TOSS_MODE} 결제 승인 실패] orderId={order_id}, "
                f"status={resp.status_code}, error={error_data.get('message', '')}"
            )
            return {
                "success": False,
                "error": error_data.get("message", f"HTTP {resp.status_code}"),
                "code": error_data.get("code", "UNKNOWN"),
            }

    except Exception as e:
        _logger.error(f"토스 결제 승인 실패: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _activate_subscription(email: str, plan: str) -> tuple:
    """[Step R+W] 구독 활성화: DB 등급 변경 + 만료일 연장.
    
    [Step W] 기존 Prime 사용자의 조기 갱신 시 기존 잔여 기간 보존:
    - 기존 만료일이 미래 → 기존 만료일 + 30일 (잔여 기간 보존)
    - 기존 만료일이 과거 또는 없음 → 오늘 + 30일
    
    Returns:
        (success: bool, expire_date: str)
    """
    db = _get_db()
    if not db:
        _logger.error("DB 연결 실패 — 구독 활성화 불가")
        return False, ""

    try:
        now_kst = datetime.now(KST)
        
        # [Step W] 기존 만료일 조회
        existing_expire = None
        if hasattr(db, "get_user_prime_expire"):
            existing_expire = db.get_user_prime_expire(email)
        
        # 만료일 계산
        if existing_expire and existing_expire > now_kst.replace(tzinfo=None):
            # 기존 만료일이 미래 → 기존 만료일 + 30일 (잔여 기간 보존)
            new_expire = existing_expire + timedelta(days=SUBSCRIPTION_DAYS)
            extension_msg = (
                f"기존 만료일({existing_expire.strftime('%Y-%m-%d')})에 "
                f"{SUBSCRIPTION_DAYS}일 추가 → {new_expire.strftime('%Y-%m-%d')}"
            )
        else:
            # 기존 만료 또는 신규 → 오늘 + 30일
            new_expire = now_kst.replace(tzinfo=None) + timedelta(days=SUBSCRIPTION_DAYS)
            extension_msg = (
                f"신규/만료된 구독 → 오늘부터 {SUBSCRIPTION_DAYS}일: "
                f"{new_expire.strftime('%Y-%m-%d')}"
            )
        
        expire_date = new_expire.strftime("%Y-%m-%d")
        db.update_user_subscription(email, plan, expire_date)
        _logger.info(f"✅ 구독 활성화: {email} → {plan} ({extension_msg})")
        return True, expire_date
    except Exception as e:
        _logger.error(f"구독 활성화 실패: {e}", exc_info=True)
        return False, ""


def _find_email_by_hash(email_hash: str) -> str:
    """[Step R+X] 해시로 사용자 이메일 역조회.
    
    Order ID에 이메일 평문이 들어가면 URL 인코딩 문제가 있어
    SHA-256 앞 12자리(Step X 강화) 해시로 저장. 결제 승인 시 
    모든 사용자 이메일에 같은 해시 적용해서 매칭되는 이메일 찾음.
    
    하위 호환: 8자리 해시 (Step R~W)도 매칭 처리.
    """
    if not email_hash:
        return ""
    db = _get_db()
    if not db:
        return ""
    try:
        hash_len = len(email_hash)
        # 모든 사용자 이메일에서 같은 해시 매칭 (8/12자리 둘 다 지원)
        users = db.get_all_users() if hasattr(db, "get_all_users") else []
        for u in users:
            email = u.get("id") or u.get("email") or ""
            if email:
                full = hashlib.sha256(email.lower().encode()).hexdigest()
                # 신규(12자리) 우선 매칭
                if hash_len == 12 and full[:12] == email_hash.lower():
                    return email
                # 하위 호환(8자리) 매칭
                elif hash_len == 8 and full[:8] == email_hash.lower():
                    return email
                # 기타 길이도 안전하게 매칭
                elif hash_len > 0 and full[:hash_len] == email_hash.lower():
                    return email
    except Exception as e:
        _logger.warning(f"이메일 역조회 실패: {e}")
    return ""


def email_to_hash(email: str) -> str:
    """[Step R+X] 이메일 → URL safe 해시 (Order ID 생성용).
    
    [Step X] 충돌 방지 강화: 8자리 → 12자리.
    공개 함수: tab_pricing.py에서 결제 위젯 호출 시 사용.
    """
    if not email:
        return ""
    return hashlib.sha256(email.lower().encode()).hexdigest()[:12]


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
    async def toss_success(
        paymentKey: str = "",
        orderId: str = "",
        amount: str = "0",
    ):
        """
        토스페이먼츠 결제 성공 리다이렉트.
        쿼리: ?paymentKey=...&orderId=...&amount=...
        
        [Step R+V] 결제 검증 체크리스트 (모두 구현됨):
        ✅ 1. paymentKey로 토스 서버 결제 승인 조회 — _confirm_toss_payment()
        ✅ 2. amount == PLAN_PRICES[plan] 검증 — _validate_payment_amount()
        ✅ 3. orderId 중복 처리 방지 — _processed_orders set
        ✅ 4. 결제 성공 DB 저장 — db.update_user_subscription()
        ✅ 5. user role 'prime' 변경 — _activate_subscription()
        ✅ 6. prime_expire_date 30일 연장 — SUBSCRIPTION_DAYS=30
        ✅ 추가: 실패/중복/위변조 시 Telegram 즉시 알림
        ✅ 추가: LIVE/TEST 모드 자동 감지 (live_*/test_* prefix)
        """
        try:
            amount_int = int(amount)
        except (ValueError, TypeError):
            amount_int = 0

        # 기본 검증
        if not paymentKey or not orderId or amount_int <= 0:
            with ui.column().classes("w-full items-center p-12"):
                ui.label("❌").classes("text-6xl mb-4")
                ui.label("결제 정보가 올바르지 않습니다").classes(
                    "text-2xl font-bold text-white mb-2"
                )
                ui.label("paymentKey, orderId, amount 누락").classes(
                    "text-red-400 mb-4 text-sm"
                )
                ui.button("🏠 홈으로",
                          on_click=lambda: ui.navigate.to("/")).props("color=primary")
            return

        # [Step R+W] 중복 결제 방지 — DB 우선, 메모리 set 보조
        # DB 중복 체크 (서버 재시작/멀티 인스턴스에서도 안정)
        db_for_dup = _get_db()
        is_duplicate_db = False
        if db_for_dup and hasattr(db_for_dup, "is_payment_processed"):
            is_duplicate_db = db_for_dup.is_payment_processed(orderId)
        
        if is_duplicate_db or orderId in _processed_orders:
            _logger.info(f"중복 결제 요청 무시: {orderId}")
            # DB에도 중복 시도 기록
            if db_for_dup and hasattr(db_for_dup, "record_payment"):
                # 이미 처리된 거니까 새로 기록 X (INSERT OR REPLACE라 status=success 유지)
                pass
            with ui.column().classes("w-full items-center p-12"):
                ui.label("ℹ️").classes("text-6xl mb-4")
                ui.label("이미 처리된 결제입니다").classes(
                    "text-xl font-bold text-white mb-2"
                )
                ui.label(f"주문번호: {orderId}").classes("text-gray-400 mb-4 text-sm")
                ui.button("🏠 홈으로",
                          on_click=lambda: ui.navigate.to("/")).props("color=primary")
            return

        # [Step R] Order ID 안전 파싱
        parsed = _parse_order_id(orderId)
        plan = parsed["plan"]
        email_hash = parsed["hash"]

        # [Step R] Server-side 금액 검증 — 위변조 방지
        if not _validate_payment_amount(plan, amount_int):
            _send_telegram(
                f"🚨 <b>[금액 위변조 시도?]</b>\n"
                f"🆔 {orderId}\n"
                f"📦 {plan}\n"
                f"💰 요청 금액: {amount_int:,}원\n"
                f"💵 정상 금액: {PLAN_PRICES.get(plan, 0):,}원"
            )
            # [Step W] DB에 위변조 시도 기록
            if db_for_dup and hasattr(db_for_dup, "record_payment"):
                db_for_dup.record_payment(
                    order_id=orderId,
                    payment_key=paymentKey,
                    email="",
                    plan=plan,
                    amount=amount_int,
                    status="amount_mismatch",
                    error_message=(
                        f"expected={PLAN_PRICES.get(plan, 0)}, "
                        f"actual={amount_int}"
                    ),
                )
            with ui.column().classes("w-full items-center p-12"):
                ui.label("⚠️").classes("text-6xl mb-4")
                ui.label("결제 금액이 올바르지 않습니다").classes(
                    "text-2xl font-bold text-white mb-2"
                )
                ui.label("관리자에게 문의해주세요").classes("text-red-400 mb-4")
                ui.button("🏠 홈으로",
                          on_click=lambda: ui.navigate.to("/")).props("color=primary")
            return

        # 결제 승인
        result = _confirm_toss_payment(paymentKey, orderId, amount_int)

        if result["success"]:
            payment_data = result.get("data", {})
            
            # [Step R] 이메일 우선순위: 토스 응답 → 해시 역조회
            email = payment_data.get("customer", {}).get("email", "")
            if not email and email_hash:
                email = _find_email_by_hash(email_hash)
            
            # 구독 활성화
            # [Step W] 구독 활성화 (기존 만료일 고려) + 정확한 만료일 반환
            activated = False
            actual_expire_date = ""
            if email:
                activated, actual_expire_date = _activate_subscription(email, plan)
                # 메모리 set에도 마킹 (보조 캐시)
                _processed_orders.add(orderId)
            
            # 영수증 URL
            receipt_url = payment_data.get("receipt", {}).get("url", "")
            method = payment_data.get("method", "-")
            approved_at = payment_data.get("approvedAt", "")[:19].replace("T", " ")
            
            # [Step W] 결제 기록 DB 저장 (성공)
            if db_for_dup and hasattr(db_for_dup, "record_payment"):
                db_for_dup.record_payment(
                    order_id=orderId,
                    payment_key=paymentKey,
                    email=email or "",
                    plan=plan,
                    amount=amount_int,
                    status="success",
                    method=method,
                    approved_at=approved_at,
                    receipt_url=receipt_url,
                )
            
            # [v22 Step AD+AE] 결제 성공 후 최종 동의 기록 — payment_success_consent
            # 결제창 열기 전 동의(payment_attempt)와 별개로,
            # 실제 결제 완료 시점의 동의 증빙 추가 보관
            # 실패 시 Telegram 관리자 알림 (결제는 성공했으므로 정상 처리 유지)
            if email and db_for_dup and hasattr(db_for_dup, "record_terms_agreement"):
                try:
                    consent_terms_ver = os.environ.get(
                        "TERMS_VERSION", "2026-04-25-v1"
                    )
                    consent_ok = db_for_dup.record_terms_agreement(
                        email=email,
                        terms_version=consent_terms_ver,
                        terms_type="refund",
                        context="payment_success",
                    )
                    if consent_ok:
                        _logger.info(
                            f"📜 결제 성공 시점 환불정책 동의 기록: {email}"
                        )
                    else:
                        # [v22 Step AE] 동의 기록 실패 → Telegram 관리자 알림
                        # (결제는 정상 처리, 운영자가 수동으로 확인)
                        _logger.warning(
                            f"⚠️ 결제 성공 동의 기록 실패: {email} / {orderId}"
                        )
                        _send_telegram(
                            f"⚠️ <b>[결제 성공 동의 기록 실패]</b>\n"
                            f"━━━━━━━━━━━━\n"
                            f"📧 {email}\n"
                            f"🆔 {orderId}\n"
                            f"💰 {amount_int:,}원 ({plan})\n"
                            f"📌 결제는 정상 완료됨 (sub 활성화 OK)\n"
                            f"⚠️ payment_success 동의 기록만 실패\n"
                            f"💡 terms_agreements 테이블에서 "
                            f"payment_attempt 기록은 있는지 확인 필요\n"
                            f"💡 분쟁 시 환불정책 동의는 "
                            f"payment_attempt context로 입증 가능"
                        )
                except Exception as ce:
                    _logger.warning(
                        f"결제 성공 동의 기록 예외 (결제는 정상): {ce}"
                    )
                    try:
                        _send_telegram(
                            f"⚠️ <b>[결제 성공 동의 기록 예외]</b>\n"
                            f"📧 {email}\n"
                            f"🆔 {orderId}\n"
                            f"❗ {str(ce)[:200]}"
                        )
                    except Exception:
                        pass
            
            # [Step W] 세션 권한 즉시 갱신 — "메뉴 새로고침" 불필요
            if activated and email:
                try:
                    user_profile = app.storage.user.get("profile", {}) or {}
                    current_email = user_profile.get("login_id") or user_profile.get("id", "")
                    # 결제한 이메일이 현재 로그인 사용자와 일치할 때만 갱신
                    if current_email and current_email.lower() == email.lower():
                        user_profile["role"] = plan
                        user_profile["prime_expire_date"] = actual_expire_date
                        app.storage.user["profile"] = user_profile
                        # auth 정보도 갱신
                        app.storage.user["auth"] = plan
                        _logger.info(f"✅ 세션 권한 즉시 갱신: {email} → {plan}")
                except Exception as session_err:
                    _logger.warning(f"세션 갱신 실패 (무시): {session_err}")

            # 관리자 알림
            mode_emoji = "💎" if TOSS_IS_LIVE else "🧪"
            mode_text = "LIVE 결제" if TOSS_IS_LIVE else "TEST 결제"
            _send_telegram(
                f"✅ <b>[{mode_emoji} {mode_text} 완료]</b>\n"
                f"━━━━━━━━━━━━\n"
                f"📧 {email or '(이메일 없음)'}\n"
                f"📦 {plan.upper()} ({amount_int:,}원)\n"
                f"💳 {method}\n"
                f"🆔 {orderId}\n"
                f"⏰ {approved_at}\n"
                f"📅 만료일: {actual_expire_date}\n"
                f"{'✅ 등급 활성화 + 세션 갱신 완료' if activated else '⚠️ 등급 활성화 실패 — 수동 처리 필요'}"
            )

            # 성공 페이지
            with ui.column().classes("w-full items-center p-8 max-w-2xl mx-auto"):
                ui.label("🎉").classes("text-7xl mb-4")
                ui.label("결제가 완료되었습니다!").classes(
                    "text-3xl font-bold text-white mb-3"
                )
                
                with ui.card().classes(
                    "w-full p-6 bg-gradient-to-br from-emerald-900/40 to-emerald-700/20 "
                    "border border-emerald-500/50 rounded-xl mb-4"
                ):
                    with ui.row().classes("w-full items-center justify-between mb-3"):
                        ui.label(f"👑 {plan.upper()}").classes(
                            "text-2xl font-bold text-amber-300"
                        )
                        ui.label(f"{amount_int:,}원").classes(
                            "text-2xl font-bold text-white"
                        )
                    ui.separator().classes("my-2")
                    
                    # [Step W] 실제 만료일 표시 (기존 잔여 기간 보존된 결과)
                    if actual_expire_date:
                        try:
                            expire_dt = datetime.strptime(actual_expire_date, "%Y-%m-%d")
                            expire_display = expire_dt.strftime("%Y년 %m월 %d일")
                        except Exception:
                            expire_display = actual_expire_date
                    else:
                        expire_display = (
                            datetime.now(KST) + timedelta(days=SUBSCRIPTION_DAYS)
                        ).strftime("%Y년 %m월 %d일")
                    
                    info_rows = [
                        ("📧 이메일", email or "(미확인)"),
                        ("💳 결제 수단", method),
                        ("⏰ 결제 시각", approved_at or "-"),
                        ("📅 이용 기한", f"~ {expire_display}"),
                        ("🆔 주문번호", orderId),
                    ]
                    for label, val in info_rows:
                        with ui.row().classes("w-full items-center gap-2 py-1"):
                            ui.label(label).classes("text-sm text-gray-400 w-32")
                            ui.label(str(val)).classes("text-sm text-white flex-1 break-all")
                
                if activated:
                    ui.label("✅ 프리미엄 기능이 활성화되었습니다!").classes(
                        "text-emerald-400 text-lg font-bold mb-2"
                    )
                    # [Step W] 세션 즉시 갱신됐음을 명시
                    ui.label(
                        "💡 지금 바로 모든 Prime 기능을 사용하실 수 있습니다 (재로그인 불필요)."
                    ).classes("text-gray-400 text-sm mb-4")
                else:
                    ui.label("⏳ 등급 활성화 처리 중입니다").classes(
                        "text-amber-400 text-lg mb-2"
                    )
                    ui.label(
                        "잠시 후 자동 활성화됩니다. 안 되면 운영자에게 문의해주세요."
                    ).classes("text-gray-400 text-sm mb-4")
                
                with ui.row().classes("gap-3 mt-2"):
                    ui.button(
                        "🏠 홈으로 돌아가기",
                        on_click=lambda: ui.navigate.to("/")
                    ).props("color=primary size=lg")
                    
                    if receipt_url:
                        ui.button(
                            "🧾 영수증 보기",
                            on_click=lambda url=receipt_url: ui.navigate.to(url, new_tab=True)
                        ).props("color=gray outlined size=lg")

        else:
            # [v22 Step X] 결제 실패도 DB에 기록 (감사 로그 완전성)
            error_msg = result.get("error", "unknown")
            error_code = result.get("code", "")
            if db_for_dup and hasattr(db_for_dup, "record_payment"):
                # email은 이 시점에 알 수 있으면 기록 (parsed에서 hash로 역조회)
                fail_email = ""
                try:
                    if email_hash:
                        fail_email = _find_email_by_hash(email_hash) or ""
                except Exception:
                    pass
                db_for_dup.record_payment(
                    order_id=orderId,
                    payment_key=paymentKey,
                    email=fail_email,
                    plan=plan,
                    amount=amount_int,
                    status="failed",
                    error_message=f"{error_code}: {error_msg}" if error_code else error_msg,
                )
            
            _send_telegram(
                f"❌ <b>[결제 승인 실패]</b>\n"
                f"🆔 {orderId}\n"
                f"💰 {amount_int:,}원\n"
                f"❗ {error_msg}\n"
                f"📛 코드: {error_code or '-'}"
            )

            with ui.column().classes("w-full items-center p-12 max-w-xl mx-auto"):
                ui.label("❌").classes("text-6xl mb-4")
                ui.label("결제 승인에 실패했습니다").classes(
                    "text-2xl font-bold text-white mb-3"
                )
                
                with ui.card().classes(
                    "w-full p-4 bg-red-900/20 border border-red-500/40 "
                    "rounded-xl mb-4"
                ):
                    ui.label("실패 사유").classes(
                        "text-xs text-red-300 font-bold mb-1"
                    )
                    ui.label(result.get("error", "알 수 없는 오류")).classes(
                        "text-sm text-red-100"
                    )
                    if result.get("code"):
                        ui.label(f"코드: {result['code']}").classes(
                            "text-xs text-gray-500 mt-1"
                        )
                
                ui.label(
                    "카드사에 문의하시거나, 다른 결제 수단으로 다시 시도해주세요."
                ).classes("text-gray-400 text-sm mb-4")
                
                with ui.row().classes("gap-3"):
                    ui.button("🔄 다시 시도",
                              on_click=lambda: ui.navigate.to("/?tab=t11")).props(
                        "color=primary"
                    )
                    ui.button("📮 문의하기",
                              on_click=lambda: ui.navigate.to("/?tab=t12")).props(
                        "color=gray outlined"
                    )

    @ui.page('/api/payments/toss/fail')
    async def toss_fail(
        code: str = "UNKNOWN",
        message: str = "결제가 취소되었습니다.",
        orderId: str = "-",
    ):
        """[Step R+AF+AG] 토스페이먼츠 결제 실패/취소 — query param 직접 받기"""
        _logger.info(f"결제 실패/취소: {code} - {message} (order: {orderId})")
        
        # [v22 Step AF+AG] 결제 취소/실패도 DB 기록 (감사 로그 완성도)
        # orderId가 "-"가 아닐 때만 기록 (유효 주문)
        if orderId != "-":
            try:
                _db = _get_db()
                
                # [v22 Step AG] 1. success 결제 보호 — 절대 cancelled로 덮지 않음
                # 외부 공격자가 success orderId로 fail URL 호출 시 방어
                already_success = False
                if _db and hasattr(_db, "is_payment_processed"):
                    already_success = _db.is_payment_processed(orderId)
                
                if already_success:
                    _logger.warning(
                        f"⚠️ 이미 success 처리된 주문에 대한 fail 콜백 — "
                        f"cancelled 기록 스킵: {orderId}"
                    )
                    # 보안 경고 Telegram (외부 공격 가능성)
                    _send_telegram(
                        f"🚨 <b>[fail 콜백 비정상 호출]</b>\n"
                        f"━━━━━━━━━━━━\n"
                        f"🆔 {orderId}\n"
                        f"⚠️ 이미 success 처리된 주문에 fail 콜백 도달\n"
                        f"📛 {code}: {message[:100]}\n"
                        f"💡 외부 공격 또는 토스 재전송 가능성 — 확인 필요"
                    )
                else:
                    # [v22 Step AG] 2. orderId 파싱 → plan/email 추정
                    parsed = _parse_order_id(orderId)
                    parsed_plan = parsed.get("plan", "prime")
                    parsed_hash = parsed.get("hash", "")
                    parsed_email = ""
                    if parsed_hash:
                        try:
                            parsed_email = _find_email_by_hash(parsed_hash) or ""
                        except Exception as fe:
                            _logger.debug(f"이메일 해시 매핑 실패: {fe}")
                    
                    if _db and hasattr(_db, "record_payment"):
                        _db.record_payment(
                            order_id=orderId,
                            payment_key="",
                            email=parsed_email,  # [Step AG] 해시→이메일 추정
                            plan=parsed_plan,    # [Step AG] orderId에서 추출
                            amount=0,
                            status="cancelled",
                            method="",
                            approved_at="",
                            receipt_url="",
                            error_message=f"{code}: {message}"[:300],
                        )
                        _logger.info(
                            f"💾 결제 취소 기록 저장: {orderId} / "
                            f"plan={parsed_plan} / email={parsed_email or '미식별'}"
                        )
                    
                    # Telegram 알림 (운영자용)
                    _send_telegram(
                        f"❌ <b>[결제 취소/실패]</b>\n"
                        f"━━━━━━━━━━━━\n"
                        f"🆔 {orderId}\n"
                        f"📧 {parsed_email or '미식별'}\n"
                        f"💎 {parsed_plan}\n"
                        f"📛 {code}\n"
                        f"💬 {message[:150]}"
                    )
            except Exception as e:
                _logger.warning(f"결제 취소 기록 실패 (무시): {e}")

        with ui.column().classes("w-full items-center p-12 max-w-xl mx-auto"):
            ui.label("😥").classes("text-6xl mb-4")
            ui.label("결제가 완료되지 않았습니다").classes(
                "text-xl font-bold text-white mb-3"
            )
            
            with ui.card().classes(
                "w-full p-4 bg-amber-900/20 border border-amber-500/40 "
                "rounded-xl mb-4"
            ):
                ui.label(message).classes("text-sm text-white mb-1")
                if code != "UNKNOWN":
                    ui.label(f"코드: {code}").classes("text-xs text-gray-400")
                if orderId != "-":
                    ui.label(f"주문번호: {orderId}").classes(
                        "text-xs text-gray-500 break-all"
                    )
            
            ui.label("결제는 진행되지 않았습니다. 다시 시도하시거나 무통장 입금을 이용해보세요.").classes(
                "text-gray-400 text-sm mb-4 text-center"
            )
            
            with ui.row().classes("gap-3"):
                ui.button("🔄 다시 시도",
                          on_click=lambda: ui.navigate.to("/?tab=t11")).props(
                    "color=primary"
                )
                ui.button("🏠 홈으로",
                          on_click=lambda: ui.navigate.to("/")).props(
                    "color=gray outlined"
                )


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
