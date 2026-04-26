# -*- coding: utf-8 -*-
"""
terms_consent.py — 📜 약관 동의 시스템
═══════════════════════════════════════════════════════════
[v22 Step AC] 가입/결제 시 약관 동의 + DB 기록 (전자상거래법 대비)

3가지 컴포넌트:
1. render_signup_consent() — 가입 폼 동의 체크박스
2. render_payment_consent() — 결제 폼 동의 체크박스
3. record_agreement() — DB 동의 기록 (Helper)

환경변수:
- TERMS_VERSION: 약관 버전 (예: 2026-04-25-v1)
                 변경 시 모든 사용자 재동의 필요
"""
import logging
import os
from typing import Tuple

from nicegui import ui

_logger = logging.getLogger(__name__)

# 현재 약관 버전 (환경변수, 변경 시 재동의 트리거)
TERMS_VERSION = os.environ.get("TERMS_VERSION", "2026-04-25-v1")


# ═══════════════════════════════════════════════════
#  Helper — 동의 기록
# ═══════════════════════════════════════════════════
def _extract_request_metadata() -> tuple:
    """[Step AD] 현재 요청에서 IP/User-Agent/Session ID 추출.
    
    NiceGUI 환경 변수에서 자동으로 가져옴.
    실패 시 빈 문자열 반환.
    
    Returns:
        (ip_address: str, user_agent: str, session_id: str)
    """
    ip = ""
    ua = ""
    sid = ""
    try:
        from nicegui import context as _ctx, app as _app
        # User-Agent
        client = getattr(_ctx, "client", None)
        if client and hasattr(client, "request"):
            req = client.request
            ua = req.headers.get("user-agent", "")[:500]
            # IP (X-Forwarded-For 우선, 없으면 client.host)
            xff = req.headers.get("x-forwarded-for", "")
            if xff:
                ip = xff.split(",")[0].strip()[:100]
            elif hasattr(req, "client") and req.client:
                ip = (req.client.host or "")[:100]
        # Session ID (NiceGUI app.storage.user 키 활용)
        try:
            sid_val = _app.storage.browser.get("id", "")
            if sid_val:
                sid = str(sid_val)[:100]
        except Exception:
            pass
    except Exception as e:
        _logger.debug(f"요청 메타데이터 추출 실패 (무시): {e}")
    return ip, ua, sid


def record_agreement(
    email: str,
    terms_type: str = "all",
    context: str = "signup",
    ip_address: str = "",
    user_agent: str = "",
    session_id: str = "",
) -> bool:
    """[Step AC+AD] 약관 동의 기록 — DB 저장.
    
    Args:
        email: 사용자 이메일
        terms_type: all / terms / privacy / refund / marketing
        context: signup / payment_attempt / payment_success / re_consent
        ip_address: 명시적 전달 X 시 자동 추출
        user_agent: 명시적 전달 X 시 자동 추출
        session_id: 명시적 전달 X 시 자동 추출
    
    Returns:
        True if recorded successfully
    """
    try:
        # [Step AD] 메타데이터 자동 추출 (호출자가 비워둔 경우)
        if not ip_address or not user_agent or not session_id:
            auto_ip, auto_ua, auto_sid = _extract_request_metadata()
            ip_address = ip_address or auto_ip
            user_agent = user_agent or auto_ua
            session_id = session_id or auto_sid
        
        from db_utils import get_db
        db = get_db()
        if not db or not hasattr(db, 'record_terms_agreement'):
            _logger.warning("record_terms_agreement 함수 없음 — DB 미적용 환경")
            return False
        
        # session_id는 user_agent 끝에 메타데이터로 부착 (DB 컬럼 호환)
        ua_with_sid = user_agent
        if session_id:
            ua_with_sid = f"{user_agent} | sid={session_id}"[:500]
        
        return db.record_terms_agreement(
            email=email,
            terms_version=TERMS_VERSION,
            terms_type=terms_type,
            context=context,
            ip_address=ip_address,
            user_agent=ua_with_sid,
        )
    except Exception as e:
        _logger.error(f"동의 기록 실패: {e}", exc_info=True)
        return False


def has_user_agreed(email: str) -> bool:
    """[Step AC] 사용자가 현재 약관 버전에 동의했는지 확인"""
    try:
        from db_utils import get_db
        db = get_db()
        if not db or not hasattr(db, 'has_agreed_to_version'):
            return False
        return db.has_agreed_to_version(email, TERMS_VERSION)
    except Exception:
        return False


# ═══════════════════════════════════════════════════
#  가입 폼 동의 체크박스
# ═══════════════════════════════════════════════════
class SignupConsent:
    """[Step AC] 가입 폼 동의 컴포넌트.
    
    사용법:
        consent = SignupConsent()
        consent.render()
        # ... 가입 처리 시
        if not consent.is_valid():
            ui.notify(consent.error_message, type='warning')
            return
        # 가입 성공 후
        record_agreement(email, terms_type='all', context='signup')
    """
    
    def __init__(self):
        self.cb_terms = None
        self.cb_privacy = None
        self.cb_age = None
        self.cb_marketing = None
        self.error_message = ""
    
    def render(self):
        """가입 폼에 통합되는 동의 영역"""
        with ui.card().classes(
            "w-full p-3 bg-[#1a1a2e] border border-cyan-500/30 "
            "rounded-lg mt-2"
        ):
            ui.label("📜 약관 동의").classes(
                "text-sm font-bold text-cyan-300 mb-2"
            )
            
            # 1. 만 14세 이상 (필수)
            with ui.row().classes("w-full items-center gap-2"):
                self.cb_age = ui.checkbox("").props("size=xs dense")
                ui.label("[필수] 본인은 만 14세 이상입니다").classes(
                    "text-xs text-white"
                )
            
            # 2. 이용약관 (필수) + 보기 버튼
            with ui.row().classes("w-full items-center gap-2"):
                self.cb_terms = ui.checkbox("").props("size=xs dense")
                ui.label("[필수] 이용약관에 동의합니다").classes(
                    "text-xs text-white"
                )
                ui.button(
                    "보기",
                    on_click=lambda: ui.navigate.to("/terms", new_tab=True),
                ).props("flat dense size=xs color=cyan").classes("ml-auto")
            
            # 3. 개인정보 처리방침 (필수) + 보기 버튼
            with ui.row().classes("w-full items-center gap-2"):
                self.cb_privacy = ui.checkbox("").props("size=xs dense")
                ui.label("[필수] 개인정보 처리방침에 동의합니다").classes(
                    "text-xs text-white"
                )
                ui.button(
                    "보기",
                    on_click=lambda: ui.navigate.to("/privacy", new_tab=True),
                ).props("flat dense size=xs color=cyan").classes("ml-auto")
            
            # 4. 마케팅 정보 수신 (선택)
            with ui.row().classes("w-full items-center gap-2"):
                self.cb_marketing = ui.checkbox("").props("size=xs dense")
                ui.label("[선택] 마케팅 정보 수신에 동의합니다").classes(
                    "text-xs text-gray-400"
                )
            
            # 안내
            ui.label(
                "💡 필수 항목에 동의해야 가입이 완료됩니다."
            ).classes("text-[10px] text-gray-500 mt-2")
    
    def is_valid(self) -> bool:
        """필수 항목 모두 체크됐는지 검증"""
        if not self.cb_age or not self.cb_age.value:
            self.error_message = "만 14세 이상 확인이 필요합니다."
            return False
        if not self.cb_terms or not self.cb_terms.value:
            self.error_message = "이용약관에 동의해주세요."
            return False
        if not self.cb_privacy or not self.cb_privacy.value:
            self.error_message = "개인정보 처리방침에 동의해주세요."
            return False
        return True
    
    @property
    def marketing_agreed(self) -> bool:
        return bool(self.cb_marketing and self.cb_marketing.value)


# ═══════════════════════════════════════════════════
#  결제 폼 동의 체크박스
# ═══════════════════════════════════════════════════
class PaymentConsent:
    """[Step AC] 결제 폼 동의 컴포넌트.
    
    사용법:
        consent = PaymentConsent()
        consent.render()
        # ... 결제 시
        if not consent.is_valid():
            ui.notify(consent.error_message, type='warning')
            return
        # 결제 성공 후
        record_agreement(email, terms_type='refund', context='payment')
    """
    
    def __init__(self):
        self.cb_refund = None
        self.cb_payment = None
        self.error_message = ""
    
    def render(self):
        """결제 폼에 통합되는 동의 영역"""
        with ui.card().classes(
            "w-full p-3 bg-amber-900/20 border border-amber-500/30 "
            "rounded-lg my-3"
        ):
            ui.label("📜 결제 전 약관 동의").classes(
                "text-sm font-bold text-amber-300 mb-2"
            )
            
            # 1. 환불정책 (필수)
            with ui.row().classes("w-full items-center gap-2"):
                self.cb_refund = ui.checkbox("").props("size=xs dense")
                ui.label(
                    "[필수] 환불정책을 확인했으며 동의합니다 "
                    "(7일 이내·미사용 시 전액 환불)"
                ).classes("text-xs text-white")
                ui.button(
                    "보기",
                    on_click=lambda: ui.navigate.to("/refund", new_tab=True),
                ).props("flat dense size=xs color=amber").classes("ml-auto")
            
            # 2. 정보통신망 이용 결제 (필수)
            with ui.row().classes("w-full items-center gap-2"):
                self.cb_payment = ui.checkbox("").props("size=xs dense")
                ui.label(
                    "[필수] 결제 진행 및 청약철회 기간 단축에 동의합니다"
                ).classes("text-xs text-white")
            
            ui.label(
                "💡 결제 진행 시 결제 정보는 토스페이먼츠로 전달됩니다."
            ).classes("text-[10px] text-gray-500 mt-2")
    
    def is_valid(self) -> bool:
        """필수 항목 모두 체크됐는지 검증"""
        if not self.cb_refund or not self.cb_refund.value:
            self.error_message = "환불정책에 동의해주세요."
            return False
        if not self.cb_payment or not self.cb_payment.value:
            self.error_message = "결제 진행에 동의해주세요."
            return False
        return True


# ═══════════════════════════════════════════════════
#  약관 변경 시 재동의 다이얼로그
# ═══════════════════════════════════════════════════
def show_re_consent_dialog(email: str, on_agreed=None):
    """[Step AC] 약관 변경 후 로그인한 사용자에게 재동의 요청.
    
    main.py에서 로그인 후 호출:
        if not has_user_agreed(email):
            show_re_consent_dialog(email)
    """
    if has_user_agreed(email):
        return  # 이미 동의함
    
    with ui.dialog().props("persistent") as dialog, ui.card().classes(
        "p-6 max-w-lg"
    ):
        ui.label("📜 약관이 변경되었습니다").classes(
            "text-xl font-bold text-amber-300 mb-3"
        )
        ui.label(
            f"현재 약관 버전: {TERMS_VERSION}\n"
            "이용 계속을 위해 변경된 약관에 동의해주세요."
        ).classes("text-sm text-white whitespace-pre-line mb-4")
        
        # 동의 체크박스 2개
        cb_terms = ui.checkbox("이용약관 변경 사항에 동의합니다").props(
            "size=sm"
        )
        cb_privacy = ui.checkbox("개인정보 처리방침 변경 사항에 동의합니다").props(
            "size=sm"
        )
        
        # 약관 보기 링크
        with ui.row().classes("w-full gap-2 mt-2"):
            ui.button(
                "📜 이용약관 보기",
                on_click=lambda: ui.navigate.to("/terms", new_tab=True),
            ).props("flat dense size=sm color=cyan")
            ui.button(
                "🔒 개인정보 보기",
                on_click=lambda: ui.navigate.to("/privacy", new_tab=True),
            ).props("flat dense size=sm color=cyan")
        
        result_label = ui.label("").classes("text-xs text-red-400 mt-2")
        
        async def do_agree():
            if not cb_terms.value or not cb_privacy.value:
                result_label.set_text("필수 약관에 모두 동의해주세요.")
                return
            
            ok = record_agreement(
                email=email,
                terms_type="all",
                context="re_consent",
            )
            if ok:
                ui.notify("동의 처리 완료", type="positive")
                dialog.close()
                if on_agreed:
                    on_agreed()
            else:
                result_label.set_text("처리 중 오류 발생. 다시 시도해주세요.")
        
        with ui.row().classes("w-full justify-end gap-2 mt-3"):
            ui.button(
                "❌ 거부 (로그아웃)",
                on_click=lambda: ui.navigate.to("/logout"),
            ).props("flat color=gray size=sm")
            ui.button(
                "✅ 동의하고 계속",
                on_click=do_agree,
            ).props("color=primary")
    
    dialog.open()
