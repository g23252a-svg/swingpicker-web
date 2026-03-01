# -*- coding: utf-8 -*-
"""
tab_pricing.py — 💎 멤버십 안내 & 결제 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════════════
Phase 1: 등급 비교 테이블 + 무통장 입금 안내 + 입금확인 요청 폼 (Telegram 웹훅)
Phase 2: 토스페이먼츠 결제 위젯 연동 (준비)
Phase 3: 정기 구독 빌링 자동화 (준비)
"""
import logging
import os
from datetime import datetime, timezone, timedelta

from nicegui import ui

_logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))

# ── 가격 설정 ──
PRICE_PRIME = 19_900

# ── 무통장 입금 계좌 (환경변수 or 하드코딩) ──
BANK_NAME = os.environ.get("BANK_NAME", "카카오뱅크")
BANK_ACCOUNT = os.environ.get("BANK_ACCOUNT", "3333-22-2658701")
BANK_HOLDER = os.environ.get("BANK_HOLDER", "이두영")

# ── Telegram 알림 ──
TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_ID", "")

# ── 토스페이먼츠 (Phase 2) ──
TOSS_CLIENT_KEY = os.environ.get("TOSS_CLIENT_KEY", "")
TOSS_SECRET_KEY = os.environ.get("TOSS_SECRET_KEY", "")
TOSS_ENABLED = bool(TOSS_CLIENT_KEY and TOSS_SECRET_KEY)


def _get_db():
    try:
        from db_utils import get_db
        db = get_db()
        if db and hasattr(db, 'ensure_gist_loaded'):
            db.ensure_gist_loaded()
        return db
    except Exception:
        return None


def _send_telegram_notification(text: str):
    """관리자에게 텔레그램 알림 발송"""
    if not TG_TOKEN or not TG_CHAT_ID:
        _logger.warning("텔레그램 미설정 — 입금 알림 발송 불가")
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TG_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        _logger.warning(f"텔레그램 알림 실패: {e}")
        return False


# ═══════════════════════════════════════════════════
#  렌더: 💎 멤버십 안내 탭
# ═══════════════════════════════════════════════════
def render_tab_pricing(auth, user):
    """
    Tab: 💎 멤버십

    Args:
        auth: "guest" | "free" | "prime" | "admin"
        user: 로그인 유저 정보 dict
    """
    if user is None:
        user = {}

    # ── 헤더 ──
    with ui.column().classes("w-full items-center mb-6"):
        ui.label("💎 멤버십 플랜").classes(
            "text-3xl font-bold text-transparent bg-clip-text "
            "bg-gradient-to-r from-blue-400 to-purple-400"
        ).style("font-family:Outfit,sans-serif")
        ui.label("AI 기반 퀀트 트레이딩의 모든 기능을 잠금 해제하세요").classes("text-gray-400 mt-1")

    # ── 현재 등급 표시 ──
    if auth != "guest":
        badge_map = {
            "free": ("🆓 Free", "gray", "무료 체험 중"),
            "prime": ("👑 Prime", "amber", "Prime 구독 중"),
            "admin": ("🛡️ Admin", "green", "관리자"),
        }
        emoji, color, desc = badge_map.get(auth, ("", "gray", ""))
        with ui.card().classes("w-full p-4 bg-[#1a1a2e] border border-gray-700 rounded-xl mb-4"):
            with ui.row().classes("items-center gap-3"):
                ui.badge(emoji).props(f"color={color}")
                ui.label(f"현재 등급: {desc}").classes("text-white text-sm")
                expire = user.get("prime_expire_date", "")
                if expire and auth == "prime":
                    ui.label(f"· 만료: {str(expire)[:10]}").classes("text-gray-400 text-xs")

    # ── 등급 비교 테이블 ──
    _render_comparison_table(auth)

    ui.separator().classes("my-6")

    # ── Phase 2: 토스페이먼츠 결제 (활성 시) ──
    if TOSS_ENABLED:
        _render_toss_payment(auth, user)
        ui.separator().classes("my-6")

    # ── Phase 1: 무통장 입금 안내 & 입금확인 폼 ──
    _render_bank_transfer(auth, user)

    # ── FAQ ──
    ui.separator().classes("my-6")
    _render_faq()


# ═══════════════════════════════════════════════════
#  등급 비교 테이블
# ═══════════════════════════════════════════════════
def _render_comparison_table(auth):
    """Free / Prime 기능 비교"""

    features = [
        ("📊 시장 현황 대시보드",     "✅", "✅"),
        ("🔭 종목 분석 (TOP 3만)",   "✅", "✅"),
        ("🔭 종목 분석 (전체 종목)",  "❌", "✅"),
        ("🔭 종목 상세 (AI 코멘트)", "❌", "✅"),
        ("💼 내 자산 AI 진단",        "❌", "✅"),
        ("📈 성과 리포트",            "❌", "✅"),
        ("📓 매매 일지",              "❌", "✅"),
        ("🧪 전략 샌드박스 (백테스트)", "❌", "✅"),
        ("🎯 켈리 비율 포지션 사이징",  "❌", "✅"),
        ("📬 텔레그램 시그널 알림",     "❌", "✅"),
        ("🆘 1:1 운영자 채팅 지원",    "❌", "✅"),
    ]

    with ui.row().classes("w-full gap-4 flex-wrap justify-center"):
        # Free
        _plan_card(
            title="Free",
            emoji="🆓",
            price="무료",
            period="",
            color_border="border-gray-600",
            color_gradient="from-gray-700 to-gray-800",
            features=[(f[0], f[1]) for f in features],
            is_current=(auth == "free"),
            button_text=None,
        )
        # Prime
        _plan_card(
            title="Prime",
            emoji="👑",
            price=f"{PRICE_PRIME:,}원",
            period="/월",
            color_border="border-amber-500",
            color_gradient="from-amber-900 to-yellow-800",
            features=[(f[0], f[2]) for f in features],
            is_current=(auth == "prime"),
            button_text="Prime 시작하기" if auth in ("guest", "free") else None,
            popular=True,
        )


def _plan_card(title, emoji, price, period, color_border, color_gradient,
               features, is_current=False, button_text=None, popular=False):
    """단일 플랜 카드 렌더"""
    with ui.card().classes(
        f"p-5 min-w-[280px] max-w-[340px] flex-1 rounded-2xl border-2 {color_border} "
        f"bg-gradient-to-b {color_gradient} relative"
    ):
        if popular:
            ui.badge("🔥 BEST").props("color=amber floating").classes("absolute top-3 right-3")

        if is_current:
            ui.badge("현재 등급").props("color=green floating").classes("absolute top-3 right-3")

        with ui.column().classes("items-center mb-4"):
            ui.label(f"{emoji} {title}").classes("text-xl font-bold text-white")
            with ui.row().classes("items-end gap-0"):
                ui.label(price).classes("text-3xl font-bold text-white")
                if period:
                    ui.label(period).classes("text-gray-400 text-sm mb-1")

        for feat_name, status in features:
            color = "text-white" if status == "✅" else "text-gray-600"
            with ui.row().classes("gap-2 items-center py-1"):
                ui.label(status).classes("text-sm")
                ui.label(feat_name).classes(f"{color} text-sm")

        if button_text:
            ui.button(
                f"✦ {button_text.upper()}",
                on_click=lambda: ui.navigate.to("#bank-transfer"),
            ).classes("w-full mt-4").props("color=primary rounded")


# ═══════════════════════════════════════════════════
#  Phase 1: 무통장 입금 안내
# ═══════════════════════════════════════════════════
def _render_bank_transfer(auth, user):
    """무통장 입금 안내 + 입금확인 요청 폼"""
    with ui.card().classes(
        "w-full p-6 bg-gradient-to-br from-[#1a1a2e] to-[#16213e] "
        "border border-blue-800 rounded-2xl"
    ).props("id=bank-transfer"):
        ui.label("🏦 무통장 입금 안내").classes("text-xl font-bold text-white mb-4")

        # ── 계좌 정보 ──
        with ui.card().classes("w-full p-4 bg-[#0d1b2a] rounded-xl mb-4"):
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("account_balance").classes("text-blue-400")
                ui.label("입금 계좌 정보").classes("text-blue-400 font-bold")

            for label, val in [("은행", BANK_NAME), ("계좌번호", BANK_ACCOUNT), ("예금주", BANK_HOLDER)]:
                with ui.row().classes("gap-2 items-center"):
                    ui.label(f"{label}:").classes("text-gray-400 text-sm w-20")
                    ui.label(val).classes("text-white font-mono text-sm")

        # ── 가격 안내 (Prime만) ──
        with ui.row().classes("w-full gap-2 mb-2"):
            ui.html(f"""
            <div style="background:#1e293b; border:2px solid #F59E0B; border-radius:8px; padding:10px 14px; flex:1;">
                <div style="color:#F59E0B; font-size:12px;">👑 Prime</div>
                <div style="color:white; font-size:18px; font-weight:bold;">{PRICE_PRIME:,}원<span style="color:#64748B; font-size:12px;">/월</span></div>
            </div>
            """)

        # ── 주의사항 ──
        with ui.row().classes("w-full gap-2 mb-4"):
            ui.html("""
            <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);
                        border-radius:8px; padding:10px 14px; width:100%;">
                <div style="color:#F59E0B; font-size:13px;">
                    ⚠️ 입금 시 <b>가입 이메일</b>을 입금자명에 포함해주세요.<br>
                    예) <code style="background:#1e293b; padding:2px 6px; border-radius:4px;">홍길동ldy</code>
                    → 확인이 빨라집니다!
                </div>
            </div>
            """)

        # ── 입금확인 요청 폼 ──
        ui.label("📋 입금 확인 요청").classes("text-white font-bold mt-2 mb-2")

        if auth == "guest":
            ui.label("⚠️ 로그인 후 이용 가능합니다.").classes("text-yellow-400")
            ui.button("🔐 로그인하기", on_click=lambda: ui.navigate.to("/login")).props("color=primary")
            return

        d_email = user.get("login_id", user.get("id", ""))
        d_nick = user.get("nickname", "")

        email_input = ui.input("가입 이메일", value=d_email).classes("w-full").props("readonly outlined dense")
        nick_input = ui.input("입금자명", value=d_nick, placeholder="입금 시 표시되는 이름").classes("w-full").props("outlined dense")
        plan_select = ui.select(
            {f"prime_{PRICE_PRIME}": f"👑 Prime ({PRICE_PRIME:,}원/월)"},
            label="신청 플랜",
            value=f"prime_{PRICE_PRIME}",
        ).classes("w-full").props("outlined dense")
        amount_input = ui.input("입금 금액 (원)", placeholder=f"{PRICE_PRIME:,}").classes("w-full").props("outlined dense type=number")
        note_input = ui.input("비고 (선택)", placeholder="입금 시각, 기타 메모").classes("w-full").props("outlined dense")

        result_label = ui.label("").classes("text-sm mt-2")

        async def submit_payment_request():
            email = email_input.value.strip()
            depositor = nick_input.value.strip()
            plan = plan_select.value
            amount = amount_input.value.strip()

            if not email or not depositor or not plan:
                ui.notify("이메일, 입금자명, 플랜을 모두 입력하세요.", type="warning")
                return
            if not amount:
                ui.notify("입금 금액을 입력하세요.", type="warning")
                return

            plan_label = "Prime"
            now_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

            # DB에 결제 요청 기록 (inquiries 테이블 활용)
            db = _get_db()
            if db:
                try:
                    db.save_inquiries(db.get_all_inquiries() + [{
                        "title": f"[💳 입금확인] {plan_label} - {depositor}",
                        "content": (
                            f"이메일: {email}\n"
                            f"입금자명: {depositor}\n"
                            f"플랜: {plan_label}\n"
                            f"금액: {amount}원\n"
                            f"비고: {note_input.value.strip()}\n"
                            f"요청시각: {now_kst}"
                        ),
                        "nickname": depositor,
                        "email": email,
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    }])
                except Exception as e:
                    _logger.warning(f"입금확인 DB 저장 실패: {e}")

            # Telegram 알림
            tg_msg = (
                f"💳 <b>[입금확인 요청]</b>\n"
                f"━━━━━━━━━━━━━\n"
                f"📧 이메일: {email}\n"
                f"👤 입금자명: {depositor}\n"
                f"📦 플랜: {plan_label}\n"
                f"💰 금액: {amount}원\n"
                f"📝 비고: {note_input.value.strip() or '-'}\n"
                f"🕐 요청: {now_kst}\n"
                f"━━━━━━━━━━━━━\n"
                f"👑 관리자 → Tab 8에서 등급 변경"
            )
            sent = _send_telegram_notification(tg_msg)

            if sent:
                result_label.set_text("✅ 입금확인 요청이 전송되었습니다! 확인 후 등급이 업그레이드됩니다.")
                result_label.classes(replace="text-green-400 text-sm mt-2")
            else:
                result_label.set_text("📨 요청이 접수되었습니다. 운영자 확인 후 등급이 변경됩니다.")
                result_label.classes(replace="text-blue-400 text-sm mt-2")

            ui.notify("📨 입금확인 요청 완료!", type="positive")

            # 입력 초기화
            amount_input.value = ""
            note_input.value = ""

        ui.button(
            "📨 입금 확인 요청 보내기",
            on_click=submit_payment_request,
        ).classes("w-full mt-2").props("color=primary rounded size=lg")


# ═══════════════════════════════════════════════════
#  Phase 2: 토스페이먼츠 결제 위젯
# ═══════════════════════════════════════════════════
def _render_toss_payment(auth, user):
    """토스페이먼츠 결제 위젯 (TOSS_CLIENT_KEY 설정 시 활성)"""

    with ui.card().classes(
        "w-full p-6 bg-gradient-to-br from-[#1a1a2e] to-[#0f3460] "
        "border border-indigo-600 rounded-2xl"
    ):
        ui.label("💳 간편 결제 (토스페이먼츠)").classes("text-xl font-bold text-white mb-4")

        if auth == "guest":
            ui.label("⚠️ 로그인 후 결제 가능합니다.").classes("text-yellow-400")
            return

        d_email = user.get("login_id", user.get("id", ""))

        plan_select = ui.select(
            {
                "prime": f"👑 Prime ({PRICE_PRIME:,}원/월)",
            },
            label="결제 플랜",
            value="prime",
        ).classes("w-full mb-4").props("outlined dense")

        async def open_toss_widget():
            plan = plan_select.value
            amount = PRICE_PRIME
            plan_name = "Prime"
            order_id = f"LDY-PRIME-{datetime.now().strftime('%Y%m%d%H%M%S')}-{d_email[:8]}"

            # 토스페이먼츠 결제 위젯 JS 삽입
            js_code = f"""
            (async () => {{
                if (!window.TossPayments) {{
                    const script = document.createElement('script');
                    script.src = 'https://js.tosspayments.com/v1/payment';
                    document.head.appendChild(script);
                    await new Promise(resolve => script.onload = resolve);
                }}
                const tossPayments = TossPayments('{TOSS_CLIENT_KEY}');
                tossPayments.requestPayment('카드', {{
                    amount: {amount},
                    orderId: '{order_id}',
                    orderName: 'SwingPicker {plan_name} 월간 구독',
                    customerName: '{user.get("nickname", "")}',
                    customerEmail: '{d_email}',
                    successUrl: window.location.origin + '/api/payments/toss/success',
                    failUrl: window.location.origin + '/api/payments/toss/fail',
                }});
            }})();
            """
            await ui.run_javascript(js_code)

        ui.button(
            "💳 카드로 결제하기",
            on_click=open_toss_widget,
        ).classes("w-full").props("color=indigo rounded size=lg")

        ui.label("카드/간편결제/계좌이체 모두 가능합니다").classes("text-gray-500 text-xs text-center mt-2")


# ═══════════════════════════════════════════════════
#  FAQ
# ═══════════════════════════════════════════════════
def _render_faq():
    """자주 묻는 질문"""
    with ui.column().classes("w-full"):
        ui.label("❓ 자주 묻는 질문").classes("text-lg font-bold text-white mb-3")

        faqs = [
            ("결제 후 등급은 언제 적용되나요?",
             "무통장 입금: 운영자 확인 후 수분~수시간 내 적용됩니다.\n"
             "카드 결제: 결제 완료 즉시 자동 적용됩니다."),
            ("구독 기간은 어떻게 되나요?",
             "결제일 기준 30일간 이용 가능합니다. 만료 전 알림을 보내드립니다."),
            ("환불은 가능한가요?",
             "결제 후 7일 이내, 유료 기능 미사용 시 전액 환불 가능합니다.\n"
             "📮 문의 탭에서 환불 요청을 남겨주세요."),
            ("Free와 Prime의 차이는 무엇인가요?",
             "Free: 시장 현황, TOP 3 종목 분석 등 기본 기능\n"
             "Prime: AI 자산진단, 전략 백테스트, 켈리 포지션 사이징,\n"
             "텔레그램 시그널 등 모든 프리미엄 기능을 이용할 수 있습니다."),
        ]

        for q, a in faqs:
            with ui.expansion(q).classes("w-full bg-[#1a1a2e] rounded-lg mb-1").props("dense"):
                ui.label(a).classes("text-gray-300 text-sm whitespace-pre-line p-2")
