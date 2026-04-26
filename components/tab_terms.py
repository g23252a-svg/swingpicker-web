# -*- coding: utf-8 -*-
"""
tab_terms.py — ⚖️ 이용약관 / 투자유의 / 정책 통합 탭
═══════════════════════════════════════════════════════════
[v22 Step AC] 전면 리팩토링 — legal_pages.py와 단일 진실(SSOT)

이전 (Step AB까지):
  - 단순 마크다운 1개 (60줄)
  - /terms /privacy /refund 별도 페이지 (legal_pages.py)
  - 두 곳 따로 관리

이후 (Step AC):
  - 4개 섹션 통합 (이용약관/개인정보/환불/투자유의)
  - 사이드 목차 (sticky)
  - 검색 박스
  - 인쇄 버튼
  - legal_pages.py 함수 직접 호출 (단일 소스)
"""
import logging
import os

from nicegui import ui

_logger = logging.getLogger(__name__)

TERMS_VERSION = os.environ.get("TERMS_VERSION", "2026-04-25-v1")


# ═══════════════════════════════════════════════════
#  탭 메인 렌더링
# ═══════════════════════════════════════════════════
def render_tab_terms():
    """[Step AC] 약관 통합 탭 — 4개 섹션 + 검색 + 목차 + 인쇄"""
    
    # ─── 헤더 ───
    with ui.row().classes("w-full items-center justify-between mb-3"):
        with ui.column().classes("gap-0"):
            ui.label("⚖️ 이용약관 · 정책").classes(
                "text-2xl font-bold text-white"
            )
            ui.label(
                f"버전: {TERMS_VERSION}  ·  시행일: 2026-04-25"
            ).classes("text-xs text-gray-400")
        
        # 우측 도구
        with ui.row().classes("gap-2"):
            ui.button(
                "🖨️ 인쇄",
                on_click=lambda: ui.run_javascript("window.print()"),
            ).props("flat dense color=cyan size=sm").tooltip(
                "현재 페이지를 인쇄/PDF 저장"
            )
            ui.button(
                "📜 변경 이력",
                on_click=lambda: ui.navigate.to("/terms/history", new_tab=True),
            ).props("flat dense color=cyan size=sm")
    
    # ─── 검색 박스 ───
    search_input = ui.input(
        placeholder="🔍 약관 내용 검색 (예: 환불, 청약철회, 만 14세)",
    ).classes("w-full mb-3").props("outlined dense clearable")
    
    # 본문 영역
    content_area = ui.column().classes("w-full")
    
    # 섹션 토글 상태
    section_state = {"active": "all", "search": ""}
    
    # 4개 섹션 정의 (단일 진실)
    sections = [
        {
            "id": "terms",
            "label": "📜 이용약관",
            "renderer": _render_terms_section,
        },
        {
            "id": "privacy",
            "label": "🔒 개인정보 처리방침",
            "renderer": _render_privacy_section,
        },
        {
            "id": "refund",
            "label": "💰 환불정책",
            "renderer": _render_refund_section,
        },
        {
            "id": "investment",
            "label": "📊 투자 유의사항",
            "renderer": _render_investment_section,
        },
    ]
    
    def refresh_content():
        content_area.clear()
        with content_area:
            # 섹션 필터
            if section_state["active"] == "all":
                visible_sections = sections
            else:
                visible_sections = [
                    s for s in sections if s["id"] == section_state["active"]
                ]
            
            search = (section_state["search"] or "").strip().lower()
            
            for sec in visible_sections:
                with ui.card().classes(
                    "w-full p-5 bg-[#1a1a2e] border border-gray-700/40 "
                    "rounded-lg mb-4"
                ):
                    ui.label(sec["label"]).classes(
                        "text-xl font-bold text-cyan-300 mb-3"
                    )
                    sec["renderer"](search)
    
    # ─── 탭 필터 ───
    with ui.row().classes("w-full gap-1 mb-3 flex-wrap"):
        toggle_options = {"all": "📌 전체"}
        toggle_options.update({s["id"]: s["label"] for s in sections})
        
        ui.toggle(
            toggle_options,
            value="all",
            on_change=lambda e: (
                section_state.update({"active": e.value}),
                refresh_content(),
            ),
        ).props("dense")
    
    def on_search_change(e):
        section_state["search"] = (e.value or "")
        refresh_content()
    
    search_input.on_value_change(on_search_change)
    
    # 첫 렌더
    refresh_content()
    
    # ─── 인쇄용 CSS ───
    ui.add_head_html("""
    <style>
    @media print {
        nav, .q-tabs, .q-toolbar, .ldy-no-print { display: none !important; }
        body { background: white !important; color: black !important; }
        .q-card { border: 1px solid #ccc !important; background: white !important; }
        h1, h2, h3, .text-white, .text-cyan-300 { color: black !important; }
        .text-gray-300, .text-gray-400 { color: #333 !important; }
    }
    </style>
    """)


# ═══════════════════════════════════════════════════
#  섹션 렌더러 — legal_pages.py 함수 직접 호출 (SSOT)
# ═══════════════════════════════════════════════════
def _highlight_search(text: str, search: str) -> tuple:
    """검색어 매칭 여부 + 표시 텍스트.
    
    Returns:
        (matched: bool, display_text: str)
    """
    if not search:
        return True, text
    if search.lower() in text.lower():
        return True, text
    return False, text


def _legal_section_inline(title: str, content: str, search: str = ""):
    """검색 매칭 시에만 표시되는 섹션."""
    # 검색어 있으면 매칭 검사
    if search:
        if (search.lower() not in title.lower()
                and search.lower() not in content.lower()):
            return  # 매칭 안 되면 표시 안 함
    
    with ui.column().classes("w-full mb-3"):
        ui.label(title).classes("text-base font-bold text-cyan-200 mb-1")
        ui.label(content).classes(
            "text-sm text-gray-300 whitespace-pre-line leading-relaxed"
        )


def _render_terms_section(search: str = ""):
    """이용약관 섹션 — legal_pages.py와 동일 내용"""
    try:
        # legal_pages.py에서 import한 환경변수 사용
        from components.legal_pages import (
            BUSINESS_NAME, BUSINESS_OWNER, PRICE_PRIME
        )
    except ImportError:
        BUSINESS_NAME = os.environ.get("BUSINESS_NAME", "SwingPicker")
        BUSINESS_OWNER = os.environ.get("BUSINESS_OWNER", "이두영")
        PRICE_PRIME = 19_900
    
    sections = [
        ("제1조 (목적)",
         f"이 약관은 {BUSINESS_NAME}(이하 '회사')이 제공하는 주식 분석 정보 제공 "
         f"서비스(이하 '서비스')의 이용 조건과 절차에 관한 사항을 규정함을 목적으로 합니다."),
        ("제2조 (정의)",
         "1. '서비스'란 회사가 제공하는 주식 시장 분석 정보, AI 기반 종목 분석, "
         "포트폴리오 진단, 백테스트 시뮬레이션 등 일체의 디지털 콘텐츠를 의미합니다.\n\n"
         "2. '회원'이란 본 약관에 동의하고 서비스에 가입한 자를 의미합니다.\n\n"
         "3. '유료 회원'이란 Prime 등 유료 플랜에 가입하여 결제를 완료한 회원을 의미합니다."),
        ("제3조 (서비스의 제공 및 변경)",
         "1. 회사는 다음과 같은 서비스를 제공합니다:\n"
         "   • 무료 회원: 시장 현황 대시보드, TOP 3 종목 분석\n"
         f"   • Prime 회원 (월 {PRICE_PRIME:,}원): 전체 종목 분석, 포트폴리오 진단, "
         "백테스트, AI 코멘트, DART 공시 분석 등 모든 기능\n\n"
         "2. 회사는 서비스 내용을 변경할 수 있으며, 중요한 변경 사항은 사전 공지합니다.\n\n"
         "3. 서비스는 24시간 제공함을 원칙으로 하나, 시스템 점검·보수 등을 위해 "
         "일시 중단될 수 있습니다."),
        ("제4조 (회원가입 및 이용계약)",
         "1. 회원가입은 이메일 인증을 통해 이루어집니다.\n\n"
         "2. 만 14세 미만 아동은 가입할 수 없습니다.\n\n"
         "3. 타인의 정보를 도용하거나 허위 정보로 가입한 경우 이용 계약은 무효입니다."),
        ("제5조 (요금 및 결제)",
         f"1. Prime 플랜의 이용 요금은 월 {PRICE_PRIME:,}원(부가세 포함)입니다.\n\n"
         "2. 결제는 토스페이먼츠를 통한 신용·체크카드, 간편결제, 계좌이체, "
         "가상계좌 또는 무통장 입금으로 가능합니다.\n\n"
         "3. 결제 완료 시점부터 30일간 서비스 이용이 가능합니다.\n\n"
         "4. 자동 갱신은 별도 신청 시에만 적용됩니다 (현재 미지원, 추후 지원 예정)."),
        ("제6조 (환불)",
         "환불 정책은 별도 [환불정책] 페이지를 따릅니다.\n"
         "요약: 결제 후 7일 이내 + 유료 기능 미사용 시 전액 환불."),
        ("제7조 (회원의 의무)",
         "1. 회원은 다음 행위를 해서는 안 됩니다:\n"
         "   • 타인의 정보 도용\n"
         "   • 회사 또는 제3자의 권리 침해\n"
         "   • 서비스 운영 방해 행위\n"
         "   • 자동화 도구·봇·스크래퍼를 통한 무단 데이터 수집\n"
         "   • 공공질서·미풍양속에 위배되는 정보 게시\n\n"
         "2. 회원은 본인의 계정 정보를 안전하게 관리할 의무가 있습니다."),
        ("제8조 (서비스의 면책)",
         "1. 본 서비스에서 제공하는 모든 분석 정보는 투자 참고 자료이며, "
         "특정 종목의 매수/매도를 권유하는 것이 아닙니다.\n\n"
         "2. 모든 투자 판단과 그에 따른 손익에 대한 책임은 회원 본인에게 있으며, "
         "회사는 어떠한 손실에 대해서도 책임지지 않습니다.\n\n"
         "3. 본 서비스는 「자본시장과 금융투자업에 관한 법률」상의 "
         "투자자문업·투자일임업에 해당하지 않습니다."),
        ("제9조 (지적재산권)",
         "1. 서비스의 모든 콘텐츠(텍스트, 이미지, 코드, 분석 알고리즘 등)에 대한 "
         "저작권은 회사에 있습니다.\n\n"
         "2. 회원은 서비스를 통해 제공받은 정보를 회사의 사전 동의 없이 "
         "복제·배포할 수 없습니다."),
        ("제10조 (계약의 해지)",
         "1. 회원은 언제든지 회원 탈퇴를 신청할 수 있습니다.\n\n"
         "2. 회사는 회원이 본 약관을 위반한 경우 사전 통지 후 이용 계약을 해지할 수 있습니다."),
        ("제11조 (분쟁의 해결)",
         "1. 본 약관과 관련된 분쟁은 대한민국 법률에 따릅니다.\n\n"
         "2. 분쟁 발생 시 회사 본점 소재지를 관할하는 법원을 전속관할로 합니다."),
    ]
    
    for title, content in sections:
        _legal_section_inline(title, content, search)


def _render_privacy_section(search: str = ""):
    """개인정보 처리방침 섹션"""
    try:
        from components.legal_pages import (
            BUSINESS_NAME, BUSINESS_OWNER, BUSINESS_EMAIL, BUSINESS_PHONE
        )
    except ImportError:
        BUSINESS_NAME = os.environ.get("BUSINESS_NAME", "SwingPicker")
        BUSINESS_OWNER = os.environ.get("BUSINESS_OWNER", "이두영")
        BUSINESS_EMAIL = os.environ.get("BUSINESS_EMAIL", "g23252a@gmail.com")
        BUSINESS_PHONE = os.environ.get("BUSINESS_PHONE", "")
    
    sections = [
        ("개요",
         f"{BUSINESS_NAME}(이하 '회사')은(는) 「개인정보 보호법」 제30조에 따라 "
         "정보주체의 개인정보를 보호하고 이와 관련한 고충을 신속하고 원활하게 처리할 수 있도록 "
         "하기 위하여 다음과 같이 개인정보처리방침을 수립·공개합니다."),
        ("제1조 (개인정보의 처리 목적)",
         "회사는 다음의 목적을 위하여 개인정보를 처리합니다:\n\n"
         "1. 회원 가입 및 관리\n"
         "   • 회원제 서비스 제공\n"
         "   • 본인 식별·인증\n"
         "   • 부정 이용 방지\n\n"
         "2. 서비스 제공\n"
         "   • Prime 멤버십 결제 처리\n"
         "   • 분석 정보 제공\n"
         "   • 사용자 맞춤 포트폴리오 진단\n\n"
         "3. 고충처리\n"
         "   • 문의·민원 처리\n"
         "   • 처리 결과 통보"),
        ("제2조 (수집하는 개인정보 항목)",
         "1. 회원가입 시:\n"
         "   • 필수: 이메일 주소, 비밀번호 (암호화 저장)\n"
         "   • 선택: 닉네임\n\n"
         "2. 결제 시:\n"
         "   • 결제 처리는 토스페이먼츠가 담당하며, 회사는 카드 정보를 저장하지 않습니다.\n"
         "   • 회사가 보관하는 정보: 결제 일시, 금액, 주문번호, 가입 이메일\n\n"
         "3. 서비스 이용 중:\n"
         "   • IP 주소, 쿠키, 접속 로그, 서비스 이용 기록\n"
         "   • 회원이 입력한 포트폴리오 데이터 (보유 종목, 평단가, 수량)"),
        ("제3조 (개인정보의 보유 및 이용 기간)",
         "1. 회원 탈퇴 시까지 보유합니다. 단, 다음의 정보는 관계 법령에 따라 일정 기간 보관:\n\n"
         "   ▶ 전자상거래법에 따른 보관:\n"
         "   • 계약 또는 청약철회 등에 관한 기록: 5년\n"
         "   • 대금결제 및 재화 등의 공급에 관한 기록: 5년\n"
         "   • 약관 동의 기록: 5년 (계약 증빙)\n"
         "   • 소비자 불만 또는 분쟁처리에 관한 기록: 3년\n\n"
         "   ▶ 통신비밀보호법에 따른 보관:\n"
         "   • 접속 로그(IP, User-Agent 등): 3개월\n\n"
         "2. 회사가 보관하는 데이터별 분류:\n"
         "   • users 테이블: 회원 탈퇴 시까지\n"
         "   • payments 테이블: 5년 (결제 기록)\n"
         "   • terms_agreements 테이블: 5년 (동의 기록)\n"
         "   • inquiries 테이블: 3년 (문의/분쟁처리)"),
        ("제4조 (개인정보의 제3자 제공)",
         "회사는 다음의 경우에만 개인정보를 제3자에게 제공합니다:\n\n"
         "1. 결제 처리:\n"
         "   • 제공받는 자: 토스페이먼츠 주식회사\n"
         "   • 제공 항목: 이메일, 결제 금액, 주문번호\n"
         "   • 이용 목적: 결제 처리 및 환불\n"
         "   • 보유 기간: 토스페이먼츠 정책에 따름\n\n"
         "2. 법령에 의거하여 수사기관 등의 요청이 있는 경우"),
        ("제5조 (개인정보의 처리 위탁)",
         "회사는 서비스 운영을 위해 다음 업무를 위탁합니다:\n\n"
         "1. 결제 처리: 토스페이먼츠 주식회사\n"
         "2. 인프라 호스팅: Railway (Railway Corp.)\n"
         "3. 데이터 저장: GitHub Inc. (Gist)\n"
         "4. 알림 발송: Telegram, Google (Gmail)\n\n"
         "위 수탁사들은 위탁받은 업무 범위 내에서만 개인정보를 처리하며, "
         "각 사의 개인정보처리방침을 따릅니다."),
        ("제6조 (정보주체의 권리)",
         "회원은 언제든지 다음의 권리를 행사할 수 있습니다:\n\n"
         "1. 개인정보 열람 요구\n"
         "2. 오류 등이 있을 경우 정정 요구\n"
         "3. 삭제 요구\n"
         "4. 처리 정지 요구\n\n"
         f"권리 행사: {BUSINESS_EMAIL} 으로 요청"),
        ("제7조 (개인정보의 안전성 확보 조치)",
         "1. 비밀번호: bcrypt 알고리즘으로 단방향 암호화 저장\n"
         "2. 통신 암호화: HTTPS(SSL) 적용\n"
         "3. 접근 통제: 관리자 계정 분리, 접근 로그 기록\n"
         "4. 개인정보 처리시스템 정기 점검"),
        ("제8조 (쿠키의 사용)",
         "1. 회사는 서비스 이용 편의 향상을 위해 쿠키를 사용합니다.\n\n"
         "2. 쿠키는 브라우저 설정에서 거부할 수 있으나, 일부 기능 이용에 제한이 있을 수 있습니다."),
        ("제9조 (개인정보 보호책임자)",
         f"성명: {BUSINESS_OWNER}\n직책: 대표\n이메일: {BUSINESS_EMAIL}\n"
         + (f"전화: {BUSINESS_PHONE}\n" if BUSINESS_PHONE else "")),
    ]
    
    for title, content in sections:
        _legal_section_inline(title, content, search)


def _render_refund_section(search: str = ""):
    """환불정책 섹션"""
    try:
        from components.legal_pages import BUSINESS_EMAIL, BUSINESS_KAKAO
    except ImportError:
        BUSINESS_EMAIL = os.environ.get("BUSINESS_EMAIL", "g23252a@gmail.com")
        BUSINESS_KAKAO = os.environ.get("BUSINESS_KAKAO", "")
    
    # 핵심 요약 카드 (검색 X일 때만)
    if not search:
        with ui.card().classes(
            "w-full p-3 bg-emerald-900/20 border border-emerald-500/40 "
            "rounded-lg mb-3"
        ):
            ui.label("✅ 환불 가능 조건 요약").classes(
                "text-base font-bold text-emerald-300 mb-1"
            )
            ui.label(
                "결제 후 7일 이내  ·  유료 기능 미사용  ·  전액 환불"
            ).classes("text-sm text-emerald-100")
    
    sections = [
        ("1. 환불 가능 조건",
         "다음 조건을 모두 충족하는 경우 전액 환불이 가능합니다:\n\n"
         "✅ 결제일로부터 7일(168시간) 이내\n"
         "✅ Prime 유료 기능을 단 한 번도 사용하지 않은 경우\n"
         "✅ 회원의 단순 변심 또는 서비스 불만족\n\n"
         "전자상거래법 제17조(청약철회)에 따른 권리 보장."),
        ("2. 환불 불가 조건",
         "다음의 경우 환불이 제한됩니다:\n\n"
         "❌ 결제일로부터 7일 경과\n"
         "❌ Prime 기능 사용 흔적 (최소 1회) 발생 시\n"
         "   • 종목 분석 (전체 종목) 조회\n"
         "   • 포트폴리오 AI 진단 실행\n"
         "   • 백테스트 시뮬레이션 실행\n"
         "   • DART 공시 분석 실행\n"
         "❌ 부정한 방법으로 가입한 경우\n"
         "❌ 약관 위반으로 회사가 계약 해지한 경우\n\n"
         "단, 서비스 장애 등 회사 귀책사유로 7일 이상 정상 이용이 불가능한 경우 "
         "사용 여부와 무관하게 환불."),
        ("3. 환불 절차",
         "1. 환불 요청 접수:\n"
         f"   • 이메일: {BUSINESS_EMAIL}\n"
         "   • 또는 SwingPicker 내 [📮 문의] 탭에서 신청\n\n"
         "2. 요청 시 필수 기재:\n"
         "   • 가입 이메일\n"
         "   • 결제일 및 결제 금액\n"
         "   • 환불 사유\n\n"
         "3. 처리 기간: 영업일 기준 1~3일 내 검토 및 처리\n\n"
         "4. 환불 방식:\n"
         "   • 카드 결제: 결제 취소 → 카드사 정책에 따라 영업일 3~7일 내 환급\n"
         "   • 계좌이체/가상계좌: 입금 계좌로 영업일 3일 내 송금\n"
         "   • 무통장 입금: 송금 시 사용한 계좌로 환급"),
        ("4. 일부 환불",
         "본 서비스는 월 단위 구독으로, 일부 환불(일할 계산)은 원칙적으로 불가합니다.\n\n"
         "단, 회사 귀책사유로 인한 서비스 중단 시:\n"
         "  • 미사용 일수 × 월 요금 / 30일 환산하여 환불"),
        ("5. 결제 수단별 환불 안내",
         "1. 신용·체크카드: 결제 취소 → 카드사 정책에 따라 환급 (3~7일)\n"
         "2. 간편결제 (카카오페이/네이버페이/토스페이): 해당 결제수단으로 환불 (1~3일)\n"
         "3. 계좌이체·가상계좌: 사용자 지정 계좌로 송금\n"
         "4. 무통장 입금: 입금 계좌로 환급 (수수료 약 500~1,000원 회원 부담)"),
        ("6. 분쟁 해결",
         "환불과 관련된 분쟁은 다음의 순서로 해결합니다:\n\n"
         "1. 회사와의 직접 협의 (영업일 7일)\n"
         "2. 한국소비자원(www.kca.go.kr) 분쟁조정\n"
         "3. 전자상거래분쟁조정위원회\n"
         "4. 회사 본점 소재지 관할 법원"),
    ]
    
    for title, content in sections:
        _legal_section_inline(title, content, search)


def _render_investment_section(search: str = ""):
    """투자 유의사항 섹션"""
    sections = [
        ("📌 서비스 성격",
         "SwingPicker는 퀀트 지표 기반의 데이터 분석 도구로, "
         "개별 종목의 매수·매도, 수익을 보장하는 리딩 서비스가 아닙니다.\n\n"
         "제공되는 모든 정보는 교육 및 참고용이며, "
         "투자 판단을 보조하는 연구·리서치 자료의 성격을 가집니다."),
        ("⚠️ 투자 책임에 대한 안내",
         "• 실제 매수·매도 등 최종 투자 의사결정은 전적으로 이용자 본인의 판단입니다.\n\n"
         "• 투자 결과로 발생하는 손익(수익, 손실, 기회비용 포함)은 "
         "모두 이용자 본인에게 귀속되며, 본 서비스 및 개발자는 이에 대해 "
         "법적 책임을 지지 않습니다.\n\n"
         "• 본 서비스는 미래 수익률, 특정 수익구간 달성, 손실 방지 등을 "
         "어떠한 형태로도 보증하지 않습니다."),
        ("📊 데이터 및 지표 한계",
         "• 사용되는 시장 데이터는 외부 데이터 제공처 및 증권사 API, "
         "공개 데이터 소스를 바탕으로 하며, 지연·오류·누락이 발생할 수 있습니다.\n\n"
         "• 지표 및 스코어는 과거 데이터를 기반으로 계산되며, "
         "향후 시장 상황과 괴리가 발생할 수 있습니다.\n\n"
         "• 알고리즘 로직은 지속적으로 개선/업데이트될 수 있으며, "
         "이 과정에서 종전 결과와 다른 스코어가 나올 수 있습니다."),
        ("💡 한 줄 요약",
         "👉 데이터와 퀀트는 도구일 뿐, 최종 책임은 언제나 본인에게 있다."),
        ("📞 문의",
         "본 서비스의 분석 정보는 「자본시장과 금융투자업에 관한 법률」상의 "
         "투자자문업·투자일임업에 해당하지 않으며, 관련 등록을 하지 않았습니다.\n\n"
         "투자 자문이 필요하신 경우 자격을 갖춘 투자자문업자에게 문의해주시기 바랍니다."),
    ]
    
    for title, content in sections:
        _legal_section_inline(title, content, search)


# ═══════════════════════════════════════════════════
#  /terms/history — 약관 변경 이력 페이지
# ═══════════════════════════════════════════════════
def render_terms_history_page():
    """[Step AC] 약관 변경 이력 페이지"""
    
    # 변경 이력 (수동 관리)
    HISTORY = [
        {
            "version": "2026-04-25-v1",
            "date": "2026-04-25",
            "type": "신규 제정",
            "changes": [
                "이용약관 신규 제정",
                "개인정보 처리방침 신규 제정",
                "환불정책 신규 제정",
                "약관 동의 기록 시스템 도입 (전자상거래법 대비)",
            ],
        },
    ]
    
    with ui.column().classes("w-full max-w-4xl mx-auto p-6"):
        ui.label("📜 약관 변경 이력").classes(
            "text-3xl font-bold text-white mb-2"
        )
        ui.label(
            "SwingPicker 이용약관·개인정보 처리방침·환불정책의 변경 이력입니다."
        ).classes("text-sm text-gray-400 mb-4")
        ui.separator().classes("my-3")
        
        for entry in HISTORY:
            with ui.card().classes(
                "w-full p-4 bg-[#1a1a2e] border border-cyan-500/30 "
                "rounded-lg mb-3"
            ):
                with ui.row().classes("w-full items-center gap-2 mb-2"):
                    ui.badge(entry["version"]).props("color=cyan")
                    ui.label(entry["date"]).classes(
                        "text-sm text-gray-400"
                    )
                    ui.badge(entry["type"]).props("color=blue").classes("ml-auto")
                
                ui.label("주요 변경 사항:").classes(
                    "text-sm font-bold text-white mb-1"
                )
                for ch in entry["changes"]:
                    ui.label(f"• {ch}").classes(
                        "text-sm text-gray-300 ml-2"
                    )
        
        with ui.row().classes("w-full justify-center mt-4"):
            ui.button(
                "🏠 홈으로",
                on_click=lambda: ui.navigate.to("/"),
            ).props("color=primary outline")
