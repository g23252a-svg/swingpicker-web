# -*- coding: utf-8 -*-
"""
tab_pricing.py — 💎 멤버십 안내 & 결제 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════════════
Phase 1: 등급 비교 테이블 + 무통장 입금 안내 + 입금확인 요청 폼 (Telegram 웹훅)
Phase 2: 토스페이먼츠 결제 위젯 연동 (준비)
Phase 3: 정기 구독 빌링 자동화 (준비)
"""
import hashlib
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
TOSS_IS_LIVE = TOSS_CLIENT_KEY.startswith("live_") and TOSS_SECRET_KEY.startswith("live_")
TOSS_IS_TEST = TOSS_CLIENT_KEY.startswith("test_") and TOSS_SECRET_KEY.startswith("test_")


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

    # ── [Step U] 사회적 증거 + 베타 배너 (등급 비교 위에 표시) ──
    try:
        from components.social_proof import render_social_proof_section
        render_social_proof_section(show_all=True)
    except ImportError:
        pass  # social_proof 모듈 없어도 정상 작동

    # ── 등급 비교 테이블 ──
    _render_comparison_table(auth)

    ui.separator().classes("my-6")

    # ═══════════════════════════════════════════════════
    # [Step R] 결제 흐름 — 토스 활성 시 토스 우선, 무통장은 접이식
    # ═══════════════════════════════════════════════════
    if TOSS_ENABLED:
        # 1순위: 토스페이먼츠 (즉시 활성화)
        _render_toss_payment(auth, user)
        
        ui.separator().classes("my-6")
        
        # 2순위: 무통장 입금 — expander로 접기 (혼란 방지)
        with ui.expansion(
            "💳 카드 결제가 어려우신가요? — 무통장 입금 안내",
            icon="account_balance",
        ).classes("w-full bg-[#1a1a2e] rounded-xl"):
            _render_bank_transfer(auth, user)
    else:
        # 토스 미설정: 무통장 입금만 표시 (기존 동작)
        if not TOSS_CLIENT_KEY and auth == "admin":
            # 관리자에게만 안내 (일반 사용자에게는 노출 X)
            with ui.card().classes(
                "w-full p-3 bg-amber-900/10 border border-amber-700/30 "
                "rounded-lg mb-4"
            ):
                ui.label(
                    "💡 [관리자 안내] 환경변수 TOSS_CLIENT_KEY/TOSS_SECRET_KEY를 "
                    "설정하면 토스페이먼츠 자동 결제가 활성화됩니다."
                ).classes("text-xs text-amber-300")
        
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

            # [Step Z] DB에 결제 요청 기록 — add_inquiry 사용 (save_inquiries DEPRECATED)
            db = _get_db()
            if db:
                try:
                    # 안정적인 inquiry_id 생성 (5초 윈도우 — 더블 클릭 방어)
                    now_5sec = int(datetime.now().timestamp() / 5) * 5
                    seed = (
                        f"{email}|입금확인|{plan_label}|{depositor}|"
                        f"{amount}|{now_5sec}"
                    )
                    inquiry_id = hashlib.sha256(seed.encode()).hexdigest()[:16]
                    
                    title = f"[💳 입금확인] {plan_label} - {depositor}"
                    content = (
                        f"이메일: {email}\n"
                        f"입금자명: {depositor}\n"
                        f"플랜: {plan_label}\n"
                        f"금액: {amount}원\n"
                        f"비고: {note_input.value.strip()}\n"
                        f"요청시각: {now_kst}"
                    )
                    created_at = datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    
                    if hasattr(db, 'add_inquiry'):
                        # Step Y+ 신규 함수 사용
                        db.add_inquiry(
                            inquiry_id=inquiry_id,
                            email=email,
                            nickname=depositor,
                            title=title,
                            content=content,
                            created_at=created_at,
                            category="payment",  # 결제 카테고리 (비공개)
                        )
                    else:
                        # Step Y 미적용 환경 (구 버전 호환)
                        _logger.warning(
                            "add_inquiry 함수 없음 — Step Y 미적용 환경. "
                            "Telegram 알림만으로 처리."
                        )
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
    """[Step R+V] 토스페이먼츠 결제 위젯 + UX 개선
    
    SDK 정책:
    - 토스페이먼츠 SDK v1 안정 버전 사용 (https://js.tosspayments.com/v1/payment)
    - requestPayment API (v2 위젯 SDK는 별도 마이그레이션 필요 시 검토)
    
    개선점:
    - Order ID에 이메일 평문 → SHA-256 해시 (URL safe)
    - 결제 직전 정보 확인 카드 표시
    - 모바일 친화적 디자인
    """

    with ui.card().classes(
        "w-full p-6 bg-gradient-to-br from-[#1a1a2e] to-[#0f3460] "
        "border border-indigo-600 rounded-2xl"
    ):
        with ui.row().classes("w-full items-center gap-2 mb-4"):
            ui.icon("credit_card", size="28px").classes("text-indigo-400")
            ui.label("💳 간편 결제 (토스페이먼츠)").classes(
                "text-xl font-bold text-white"
            )
            ui.badge("⚡ 즉시 활성화").props("color=green").classes("ml-auto")
        
        # [Step R+] 관리자에게만 LIVE/TEST 모드 표시
        if auth == "admin":
            if TOSS_IS_LIVE:
                ui.label(
                    "🔴 LIVE 모드 — 실제 결제가 처리됩니다 (정산 진행)"
                ).classes("text-xs text-red-400 font-bold mb-2")
            elif TOSS_IS_TEST:
                ui.label(
                    "🧪 TEST 모드 — 실제 결제 안 됨 (개발/테스트용)"
                ).classes("text-xs text-amber-400 font-bold mb-2")

        if auth == "guest":
            with ui.card().classes(
                "w-full p-4 bg-amber-900/20 border border-amber-600/40 rounded-lg"
            ):
                ui.label("⚠️ 로그인 후 결제 가능합니다").classes("text-amber-300 mb-2")
                ui.button("🔐 로그인하기",
                          on_click=lambda: ui.navigate.to("/login")).props("color=primary")
            return

        d_email = user.get("login_id", user.get("id", ""))
        d_nickname = user.get("nickname", "고객님")

        # [Step R] 결제 정보 확인 카드 (사용자 안심)
        with ui.card().classes(
            "w-full p-4 bg-[#0d1b2a] rounded-lg mb-4 border border-indigo-700/30"
        ):
            ui.label("📋 결제 정보 확인").classes(
                "text-xs text-indigo-300 font-bold mb-2"
            )
            for label, val in [
                ("📧 가입 이메일", d_email),
                ("👤 결제자명", d_nickname),
                ("📦 상품", f"SwingPicker Prime 1개월 구독"),
                ("💰 결제 금액", f"{PRICE_PRIME:,}원"),
                ("📅 이용 기한", "결제일로부터 30일"),
            ]:
                with ui.row().classes("w-full items-center gap-2 py-1"):
                    ui.label(label).classes("text-xs text-gray-400 w-28")
                    ui.label(val).classes("text-sm text-white flex-1")

        # 결제 수단 안내
        with ui.row().classes("w-full gap-2 mb-3 flex-wrap justify-center"):
            for icon, label in [
                ("💳", "카드"),
                ("📱", "간편결제"),
                ("🏦", "계좌이체"),
                ("💰", "가상계좌"),
            ]:
                with ui.card().classes(
                    "px-3 py-2 bg-[#0d1b2a] border border-gray-700 rounded-lg"
                ):
                    ui.label(f"{icon} {label}").classes("text-xs text-gray-300")

        # [v22 Step AC] 결제 동의 체크박스
        payment_consent = None
        consent_import_failed = False
        try:
            from components.terms_consent import PaymentConsent
            payment_consent = PaymentConsent()
            payment_consent.render()
        except ImportError:
            # [v22 Step AE] 동의 시스템 필수 — UI에 명시적 경고 표시
            consent_import_failed = True
            with ui.card().classes(
                "w-full p-3 bg-red-900/30 border border-red-500/60 "
                "rounded-lg my-3"
            ):
                ui.label("⚠️ 약관 동의 시스템 로드 실패").classes(
                    "text-sm font-bold text-red-300"
                )
                ui.label(
                    "결제 시스템이 일시적으로 비활성화되었습니다. "
                    "운영자에게 문의해주세요."
                ).classes("text-xs text-red-200")

        async def open_toss_widget():
            """[Step R+V+AC+AD+AE] 토스페이먼츠 결제 요청 (SDK v1 안정 버전 사용)"""
            
            # [v22 Step AE] 동의 시스템 import 실패 시 결제 차단
            if consent_import_failed:
                ui.notify(
                    "⚠️ 약관 동의 시스템을 불러올 수 없습니다. "
                    "운영자에게 문의해주세요.",
                    type="negative",
                    timeout=5000,
                )
                return
            
            # [v22 Step AC+AD] 약관 동의 검증 + 기록 실패 시 결제 중단
            if payment_consent is not None:
                if not payment_consent.is_valid():
                    ui.notify(
                        f"⚠️ {payment_consent.error_message}",
                        type="warning",
                    )
                    return
                
                # [v22 Step AD] 동의 기록 — 실패 시 결제창 열지 않음
                try:
                    from components.terms_consent import record_agreement
                    
                    # User-Agent 추출
                    ua = ""
                    try:
                        from nicegui import context as _ctx
                        req = getattr(_ctx, "client", None)
                        if req and hasattr(req, "request"):
                            ua = req.request.headers.get("user-agent", "")[:500]
                    except Exception:
                        pass
                    
                    consent_ok = record_agreement(
                        email=d_email,
                        terms_type="refund",
                        context="payment_attempt",
                        user_agent=ua,
                    )
                    if not consent_ok:
                        ui.notify(
                            "⚠️ 약관 동의 기록 저장에 실패했습니다. "
                            "다시 시도해주세요.",
                            type="warning",
                            timeout=5000,
                        )
                        return
                except ImportError:
                    # [v22 Step AE] 동의 시스템 필수 — ImportError 시 결제 차단
                    # 운영판에서 약관 동의 시스템은 법적 필수 기능
                    ui.notify(
                        "⚠️ 약관 동의 시스템을 불러올 수 없습니다. "
                        "운영자에게 문의해주세요.",
                        type="negative",
                        timeout=5000,
                    )
                    return
                except Exception as e:
                    _logger.error(f"동의 기록 시스템 오류: {e}", exc_info=True)
                    ui.notify(
                        "⚠️ 약관 동의 시스템 오류. 운영자에게 문의해주세요.",
                        type="negative",
                    )
                    return
            
            plan = "prime"
            amount = PRICE_PRIME
            plan_name = "Prime"
            
            # [Step R] Order ID — 이메일 해시 사용 (URL safe)
            try:
                from payments import email_to_hash
                email_hash = email_to_hash(d_email)
            except Exception:
                # fallback
                import hashlib as _hl
                email_hash = _hl.sha256(d_email.lower().encode()).hexdigest()[:8] if d_email else "00000000"
            
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            order_id = f"LDY-{plan.upper()}-{timestamp}-{email_hash}"

            # 토스페이먼츠 SDK v1 (안정 버전 유지, requestPayment API)
            # 결제 위젯 모달 띄움 → 사용자가 결제 수단 선택
            # 성공 시: successUrl로 GET 리다이렉트 (paymentKey, orderId, amount 포함)
            # 실패 시: failUrl로 GET 리다이렉트 (code, message 포함)
            js_code = f"""
            (async () => {{
                try {{
                    if (!window.TossPayments) {{
                        const script = document.createElement('script');
                        script.src = 'https://js.tosspayments.com/v1/payment';
                        document.head.appendChild(script);
                        await new Promise((resolve, reject) => {{
                            script.onload = resolve;
                            script.onerror = reject;
                        }});
                    }}
                    const tossPayments = TossPayments('{TOSS_CLIENT_KEY}');
                    await tossPayments.requestPayment('카드', {{
                        amount: {amount},
                        orderId: '{order_id}',
                        orderName: 'SwingPicker {plan_name} 1개월 구독',
                        customerName: {repr(d_nickname)},
                        customerEmail: {repr(d_email)},
                        successUrl: window.location.origin + '/api/payments/toss/success',
                        failUrl: window.location.origin + '/api/payments/toss/fail',
                    }});
                }} catch (err) {{
                    console.error('토스 결제 호출 실패:', err);
                    alert('결제 위젯을 불러오지 못했습니다. 잠시 후 다시 시도해주세요.\\n\\n오류: ' + (err.message || err));
                }}
            }})();
            """
            await ui.run_javascript(js_code)

        # 메인 결제 버튼
        ui.button(
            f"💳 {PRICE_PRIME:,}원 결제하기",
            on_click=open_toss_widget,
        ).classes("w-full mt-2").props("color=indigo rounded size=lg")

        # 보안/신뢰 신호
        with ui.row().classes("w-full justify-center gap-3 mt-3 flex-wrap"):
            for icon, label in [
                ("🔒", "SSL 보호"),
                ("🛡️", "토스페이먼츠 PG"),
                ("✅", "안심 결제"),
            ]:
                ui.label(f"{icon} {label}").classes("text-[10px] text-gray-500")
        
        ui.label(
            "💡 결제 완료 즉시 프리미엄 기능이 활성화됩니다 (재로그인 불필요)"
        ).classes("text-xs text-emerald-400/80 text-center mt-2 italic")


# ═══════════════════════════════════════════════════
#  FAQ
# ═══════════════════════════════════════════════════
def _render_faq():
    """[Step R] 자주 묻는 질문 — 결제 보안/실패 처리 추가"""
    with ui.column().classes("w-full"):
        ui.label("❓ 자주 묻는 질문").classes("text-lg font-bold text-white mb-3")

        faqs = [
            ("결제 후 등급은 언제 적용되나요?",
             "💳 카드/간편결제: 결제 완료 즉시 자동 활성화됩니다.\n"
             "    → 메뉴 새로고침 또는 페이지 이동 시 모든 기능 이용 가능\n"
             "🏦 무통장 입금: 운영자 확인 후 평균 2시간 내 (영업일 기준) 적용됩니다.\n"
             "    → 처리 완료 시 가입 이메일로 안내 발송"),
            ("구독 기간과 자동 갱신은 어떻게 되나요?",
             "결제일 기준 30일간 이용 가능합니다.\n"
             "현재는 정기 자동 결제 미지원 — 만료 7일 전 갱신 안내 발송됩니다.\n"
             "갱신을 원하시면 다시 결제해주세요."),
            ("환불은 가능한가요?",
             "💳 카드 결제: 결제 후 7일 이내, 유료 기능 미사용 시 전액 환불 가능\n"
             "🏦 무통장 입금: 결제 후 7일 이내, 동일 조건으로 환불\n"
             "📮 문의 탭에서 '환불 요청'으로 남겨주시면 영업일 기준 1~3일 내 처리됩니다."),
            ("결제 정보는 안전한가요?",
             "✅ 모든 결제는 국내 PG사 토스페이먼츠를 통해 처리됩니다.\n"
             "✅ 카드 정보는 토스페이먼츠 결제창에서 직접 입력되며, "
             "SwingPicker는 카드번호를 저장하지 않습니다.\n"
             "✅ HTTPS(SSL) 암호화 연결로 안전하게 처리됩니다.\n"
             "✅ 결제 영수증은 토스페이먼츠 공식 영수증으로 발행됩니다."),
            ("결제가 실패하면 어떻게 되나요?",
             "결제가 승인되지 않은 경우 청구되지 않습니다.\n"
             "다음 사항을 확인해주세요:\n"
             "  • 카드 한도 / 잔액\n"
             "  • 카드 유효기간\n"
             "  • 해외 결제 차단 여부 (일부 카드)\n"
             "지속 실패 시 다른 결제 수단(무통장 입금)을 이용하시거나 문의 탭에 남겨주세요."),
            ("Free와 Prime의 차이는 무엇인가요?",
             "🆓 Free:\n"
             "  • 시장 현황 대시보드\n"
             "  • TOP 3 종목 분석\n\n"
             "👑 Prime (모든 기능 잠금 해제):\n"
             "  • 전체 종목 분석 + AI 코멘트\n"
             "  • 내 자산 AI 진단 + DART 공시 리스크\n"
             "  • 성과 리포트 + 매매 일지\n"
             "  • 전략 백테스트 + 켈리 포지션 사이징\n"
             "  • 텔레그램 실시간 시그널\n"
             "  • 1:1 운영자 채팅 지원"),
        ]

        for q, a in faqs:
            with ui.expansion(q).classes("w-full bg-[#1a1a2e] rounded-lg mb-1").props("dense"):
                ui.label(a).classes("text-gray-300 text-sm whitespace-pre-line p-3")
        
        # [Step R] 사업자 정보 (한국 전자상거래법)
        ui.separator().classes("my-4")
        with ui.card().classes(
            "w-full p-4 bg-[#0d1b2a] border border-gray-700 rounded-lg"
        ):
            ui.label("📋 사업자 정보").classes("text-xs text-gray-400 font-bold mb-2")
            for label, val in [
                ("상호", os.environ.get("BUSINESS_NAME", "SwingPicker")),
                ("대표자", os.environ.get("BUSINESS_OWNER", BANK_HOLDER)),
                ("이메일", os.environ.get("BUSINESS_EMAIL", "support@swingpicker.com")),
                ("통신판매업 신고", os.environ.get("BUSINESS_LICENSE", "(신고 준비 중)")),
            ]:
                with ui.row().classes("w-full gap-2 py-1"):
                    ui.label(f"{label}").classes("text-xs text-gray-500 w-24")
                    ui.label(val).classes("text-xs text-gray-300 flex-1")
