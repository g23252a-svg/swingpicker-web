# -*- coding: utf-8 -*-
"""
legal_pages.py — 법적 페이지 (이용약관/개인정보처리방침/환불정책)
═══════════════════════════════════════════════════════════
[v22 Step S] 토스페이먼츠 심사 통과 + 한국 전자상거래법 준수

3개 페이지:
- /terms     : 이용약관
- /privacy   : 개인정보처리방침
- /refund    : 환불정책

main.py에서 register_legal_pages() 호출
"""
import os
from datetime import datetime

from nicegui import ui

# 사업자 정보 환경변수
BUSINESS_NAME = os.environ.get("BUSINESS_NAME", "SwingPicker")
BUSINESS_OWNER = os.environ.get("BUSINESS_OWNER", "이두영")
BUSINESS_EMAIL = os.environ.get("BUSINESS_EMAIL", "support@swingpicker.com")
BUSINESS_PHONE = os.environ.get("BUSINESS_PHONE", "")
BUSINESS_REG_NO = os.environ.get("BUSINESS_REG_NO", "")
BUSINESS_LICENSE = os.environ.get("BUSINESS_LICENSE", "")
BUSINESS_ADDRESS = os.environ.get("BUSINESS_ADDRESS", "")
# [Step T] 카카오톡 채널
BUSINESS_KAKAO = os.environ.get("BUSINESS_KAKAO", "")  # @swingpicker
BUSINESS_KAKAO_URL = os.environ.get("BUSINESS_KAKAO_URL", "")  # https://pf.kakao.com/_xxx
PRICE_PRIME = 19_900


def _legal_page_header(title: str, subtitle: str = ""):
    """공통 페이지 헤더"""
    with ui.column().classes("w-full max-w-4xl mx-auto p-6"):
        ui.label(title).classes(
            "text-3xl font-bold text-white mb-2"
        )
        if subtitle:
            ui.label(subtitle).classes("text-sm text-gray-400 mb-4")
        ui.separator().classes("my-3")


def _legal_section(title: str, content: str):
    """약관 섹션 — 제목 + 내용"""
    with ui.column().classes("w-full mb-4"):
        ui.label(title).classes("text-lg font-bold text-cyan-300 mb-2")
        ui.label(content).classes(
            "text-sm text-gray-300 whitespace-pre-line leading-relaxed"
        )


def _legal_footer():
    """공통 페이지 푸터 — 홈 버튼 + 사업자 정보"""
    ui.separator().classes("my-6")
    
    with ui.column().classes("w-full items-center text-center gap-1 mb-4"):
        ui.label(
            f"📌 {BUSINESS_NAME}  ·  대표: {BUSINESS_OWNER}"
        ).classes("text-xs text-gray-400")
        
        info_parts = []
        if BUSINESS_REG_NO:
            info_parts.append(f"사업자등록번호: {BUSINESS_REG_NO}")
        if BUSINESS_LICENSE:
            info_parts.append(f"통신판매업: {BUSINESS_LICENSE}")
        if info_parts:
            ui.label("  ·  ".join(info_parts)).classes(
                "text-[10px] text-gray-500"
            )
        
        contact_parts = []
        if BUSINESS_ADDRESS:
            contact_parts.append(BUSINESS_ADDRESS)
        if BUSINESS_PHONE:
            contact_parts.append(BUSINESS_PHONE)
        if BUSINESS_EMAIL:
            contact_parts.append(BUSINESS_EMAIL)
        if BUSINESS_KAKAO:
            contact_parts.append(f"카카오톡: {BUSINESS_KAKAO}")
        if contact_parts:
            ui.label("  ·  ".join(contact_parts)).classes(
                "text-[10px] text-gray-500"
            )
    
    with ui.row().classes("w-full justify-center mb-8"):
        ui.button(
            "🏠 홈으로",
            on_click=lambda: ui.navigate.to("/")
        ).props("color=primary outline")


# ═══════════════════════════════════════════════════
#  /terms — 이용약관
# ═══════════════════════════════════════════════════
def render_terms_page():
    """이용약관"""
    _legal_page_header(
        "📜 이용약관",
        f"시행일: 2026년 4월 25일  ·  {BUSINESS_NAME}"
    )
    
    with ui.column().classes("w-full max-w-4xl mx-auto px-6"):
        
        _legal_section("제1조 (목적)",
            f"이 약관은 {BUSINESS_NAME}(이하 '회사')이 제공하는 주식 분석 정보 제공 "
            f"서비스(이하 '서비스')의 이용 조건과 절차에 관한 사항을 규정함을 목적으로 합니다."
        )
        
        _legal_section("제2조 (정의)",
            "1. '서비스'란 회사가 제공하는 주식 시장 분석 정보, AI 기반 종목 분석, "
            "포트폴리오 진단, 백테스트 시뮬레이션 등 일체의 디지털 콘텐츠를 의미합니다.\n\n"
            "2. '회원'이란 본 약관에 동의하고 서비스에 가입한 자를 의미합니다.\n\n"
            "3. '유료 회원'이란 Prime 등 유료 플랜에 가입하여 결제를 완료한 회원을 의미합니다."
        )
        
        _legal_section("제3조 (서비스의 제공 및 변경)",
            "1. 회사는 다음과 같은 서비스를 제공합니다:\n"
            "   • 무료 회원: 시장 현황 대시보드, TOP 3 종목 분석\n"
            f"   • Prime 회원 (월 {PRICE_PRIME:,}원): 전체 종목 분석, 포트폴리오 진단, "
            "백테스트, AI 코멘트, DART 공시 분석 등 모든 기능\n\n"
            "2. 회사는 서비스 내용을 변경할 수 있으며, 중요한 변경 사항은 사전 공지합니다.\n\n"
            "3. 서비스는 24시간 제공함을 원칙으로 하나, 시스템 점검·보수 등을 위해 일시 중단될 수 있습니다."
        )
        
        _legal_section("제4조 (회원가입 및 이용계약)",
            "1. 회원가입은 이메일 인증을 통해 이루어집니다.\n\n"
            "2. 만 14세 미만 아동은 가입할 수 없습니다.\n\n"
            "3. 타인의 정보를 도용하거나 허위 정보로 가입한 경우 이용 계약은 무효입니다."
        )
        
        _legal_section("제5조 (요금 및 결제)",
            f"1. Prime 플랜의 이용 요금은 월 {PRICE_PRIME:,}원(부가세 포함)입니다.\n\n"
            "2. 결제는 토스페이먼츠를 통한 신용·체크카드, 간편결제, 계좌이체, "
            "가상계좌 또는 무통장 입금으로 가능합니다.\n\n"
            "3. 결제 완료 시점부터 30일간 서비스 이용이 가능합니다.\n\n"
            "4. 자동 갱신은 별도 신청 시에만 적용됩니다 (현재 미지원, 추후 지원 예정)."
        )
        
        _legal_section("제6조 (환불)",
            "환불 정책은 별도 [환불정책] 페이지를 따릅니다.\n"
            "요약: 결제 후 7일 이내 + 유료 기능 미사용 시 전액 환불."
        )
        
        _legal_section("제7조 (회원의 의무)",
            "1. 회원은 다음 행위를 해서는 안 됩니다:\n"
            "   • 타인의 정보 도용\n"
            "   • 회사 또는 제3자의 권리 침해\n"
            "   • 서비스 운영 방해 행위\n"
            "   • 자동화 도구·봇·스크래퍼를 통한 무단 데이터 수집\n"
            "   • 공공질서·미풍양속에 위배되는 정보 게시\n\n"
            "2. 회원은 본인의 계정 정보를 안전하게 관리할 의무가 있습니다."
        )
        
        _legal_section("제8조 (서비스의 면책)",
            "1. 본 서비스에서 제공하는 모든 분석 정보는 투자 참고 자료이며, "
            "특정 종목의 매수/매도를 권유하는 것이 아닙니다.\n\n"
            "2. 모든 투자 판단과 그에 따른 손익에 대한 책임은 회원 본인에게 있으며, "
            "회사는 어떠한 손실에 대해서도 책임지지 않습니다.\n\n"
            "3. 본 서비스는 「자본시장과 금융투자업에 관한 법률」상의 "
            "투자자문업·투자일임업에 해당하지 않습니다."
        )
        
        _legal_section("제9조 (지적재산권)",
            "1. 서비스의 모든 콘텐츠(텍스트, 이미지, 코드, 분석 알고리즘 등)에 대한 "
            "저작권은 회사에 있습니다.\n\n"
            "2. 회원은 서비스를 통해 제공받은 정보를 회사의 사전 동의 없이 복제·배포할 수 없습니다."
        )
        
        _legal_section("제10조 (계약의 해지)",
            "1. 회원은 언제든지 회원 탈퇴를 신청할 수 있습니다.\n\n"
            "2. 회사는 회원이 본 약관을 위반한 경우 사전 통지 후 이용 계약을 해지할 수 있습니다."
        )
        
        _legal_section("제11조 (분쟁의 해결)",
            "1. 본 약관과 관련된 분쟁은 대한민국 법률에 따릅니다.\n\n"
            "2. 분쟁 발생 시 회사 본점 소재지를 관할하는 법원을 전속관할로 합니다."
        )
        
        _legal_section("부칙",
            "이 약관은 2026년 4월 25일부터 시행됩니다."
        )
    
    _legal_footer()


# ═══════════════════════════════════════════════════
#  /privacy — 개인정보처리방침
# ═══════════════════════════════════════════════════
def render_privacy_page():
    """개인정보처리방침"""
    _legal_page_header(
        "🔒 개인정보처리방침",
        f"시행일: 2026년 4월 25일  ·  {BUSINESS_NAME}"
    )
    
    with ui.column().classes("w-full max-w-4xl mx-auto px-6"):
        
        ui.label(
            f"{BUSINESS_NAME}(이하 '회사')은(는) 「개인정보 보호법」 제30조에 따라 "
            "정보주체의 개인정보를 보호하고 이와 관련한 고충을 신속하고 원활하게 처리할 수 있도록 "
            "하기 위하여 다음과 같이 개인정보처리방침을 수립·공개합니다."
        ).classes("text-sm text-gray-300 mb-4 leading-relaxed")
        
        _legal_section("제1조 (개인정보의 처리 목적)",
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
            "   • 처리 결과 통보"
        )
        
        _legal_section("제2조 (수집하는 개인정보 항목)",
            "1. 회원가입 시:\n"
            "   • 필수: 이메일 주소, 비밀번호 (암호화 저장)\n"
            "   • 선택: 닉네임\n\n"
            "2. 결제 시:\n"
            "   • 결제 처리는 토스페이먼츠가 담당하며, 회사는 카드 정보를 저장하지 않습니다.\n"
            "   • 회사가 보관하는 정보: 결제 일시, 금액, 주문번호, 가입 이메일\n\n"
            "3. 서비스 이용 중:\n"
            "   • IP 주소, 쿠키, 접속 로그, 서비스 이용 기록\n"
            "   • 회원이 입력한 포트폴리오 데이터 (보유 종목, 평단가, 수량)"
        )
        
        _legal_section("제3조 (개인정보의 보유 및 이용 기간)",
            "1. 회원 탈퇴 시까지 보유합니다. 단, 다음의 정보는 관계 법령에 따라 일정 기간 보관합니다:\n\n"
            "   ▶ 전자상거래법에 따른 보관:\n"
            "   • 계약 또는 청약철회 등에 관한 기록: 5년 (전자상거래법 제6조)\n"
            "   • 대금결제 및 재화 등의 공급에 관한 기록: 5년 (전자상거래법 제6조)\n"
            "   • 약관 동의 기록: 5년 (전자상거래법 제6조 — 계약 증빙)\n"
            "   • 소비자 불만 또는 분쟁처리에 관한 기록: 3년 (전자상거래법 제6조)\n\n"
            "   ▶ 통신비밀보호법에 따른 보관:\n"
            "   • 접속 로그(IP, User-Agent 등): 3개월 (통신비밀보호법 제15조의2)\n\n"
            "2. 회사가 보관하는 데이터별 분류:\n"
            "   • users 테이블: 회원 탈퇴 시까지 (탈퇴 후 즉시 파기)\n"
            "   • payments 테이블: 5년 (결제 기록 — 전자상거래법)\n"
            "   • terms_agreements 테이블: 5년 (동의 기록 — 전자상거래법)\n"
            "   • inquiries 테이블: 3년 (문의/분쟁처리 — 전자상거래법)\n\n"
            "3. 보유 기간이 만료된 개인정보는 지체 없이 파기합니다."
        )
        
        _legal_section("제4조 (개인정보의 제3자 제공)",
            "회사는 다음의 경우에만 개인정보를 제3자에게 제공합니다:\n\n"
            "1. 결제 처리:\n"
            "   • 제공받는 자: 토스페이먼츠 주식회사\n"
            "   • 제공 항목: 이메일, 결제 금액, 주문번호\n"
            "   • 이용 목적: 결제 처리 및 환불\n"
            "   • 보유 기간: 토스페이먼츠 정책에 따름\n\n"
            "2. 법령에 의거하여 수사기관 등의 요청이 있는 경우"
        )
        
        _legal_section("제5조 (개인정보의 처리 위탁)",
            "회사는 서비스 운영을 위해 다음 업무를 위탁합니다:\n\n"
            "1. 결제 처리: 토스페이먼츠 주식회사\n"
            "2. 인프라 호스팅: Railway (Railway Corp.)\n"
            "3. 데이터 저장: GitHub Inc. (Gist)\n"
            "4. 알림 발송: Telegram, Google (Gmail)\n\n"
            "위 수탁사들은 위탁받은 업무 범위 내에서만 개인정보를 처리하며, "
            "각 사의 개인정보처리방침을 따릅니다."
        )
        
        _legal_section("제6조 (정보주체의 권리)",
            "회원은 언제든지 다음의 권리를 행사할 수 있습니다:\n\n"
            "1. 개인정보 열람 요구\n"
            "2. 오류 등이 있을 경우 정정 요구\n"
            "3. 삭제 요구\n"
            "4. 처리 정지 요구\n\n"
            f"권리 행사: {BUSINESS_EMAIL} 으로 요청"
        )
        
        _legal_section("제7조 (개인정보의 안전성 확보 조치)",
            "1. 비밀번호: bcrypt 알고리즘으로 단방향 암호화 저장\n"
            "2. 통신 암호화: HTTPS(SSL) 적용\n"
            "3. 접근 통제: 관리자 계정 분리, 접근 로그 기록\n"
            "4. 개인정보 처리시스템 정기 점검"
        )
        
        _legal_section("제8조 (쿠키의 사용)",
            "1. 회사는 서비스 이용 편의 향상을 위해 쿠키를 사용합니다.\n\n"
            "2. 쿠키는 브라우저 설정에서 거부할 수 있으나, 일부 기능 이용에 제한이 있을 수 있습니다."
        )
        
        _legal_section("제9조 (개인정보 보호책임자)",
            f"성명: {BUSINESS_OWNER}\n"
            f"직책: 대표\n"
            f"이메일: {BUSINESS_EMAIL}\n"
            + (f"전화: {BUSINESS_PHONE}\n" if BUSINESS_PHONE else "")
        )
        
        _legal_section("제10조 (방침 변경)",
            "이 개인정보처리방침은 시행일로부터 적용되며, 변경 시 시행 7일 전부터 공지합니다."
        )
        
        _legal_section("부칙",
            "이 방침은 2026년 4월 25일부터 시행됩니다."
        )
    
    _legal_footer()


# ═══════════════════════════════════════════════════
#  /refund — 환불정책
# ═══════════════════════════════════════════════════
def render_refund_page():
    """환불정책"""
    _legal_page_header(
        "💰 환불정책",
        f"시행일: 2026년 4월 25일  ·  {BUSINESS_NAME}"
    )
    
    with ui.column().classes("w-full max-w-4xl mx-auto px-6"):
        
        # 핵심 요약 카드
        with ui.card().classes(
            "w-full p-5 bg-gradient-to-br from-emerald-900/30 to-emerald-700/20 "
            "border border-emerald-500/40 rounded-xl mb-6"
        ):
            ui.label("✅ 환불 가능 조건 요약").classes(
                "text-lg font-bold text-emerald-300 mb-2"
            )
            ui.label(
                "결제 후 7일 이내  ·  유료 기능 미사용  ·  전액 환불"
            ).classes("text-base text-emerald-100")
        
        _legal_section("1. 환불 가능 조건",
            "다음 조건을 모두 충족하는 경우 전액 환불이 가능합니다:\n\n"
            "✅ 결제일로부터 7일(168시간) 이내\n"
            "✅ Prime 유료 기능을 단 한 번도 사용하지 않은 경우\n"
            "✅ 회원의 단순 변심 또는 서비스 불만족\n\n"
            "전자상거래법 제17조(청약철회)에 따른 권리 보장."
        )
        
        _legal_section("2. 환불 불가 조건",
            "다음의 경우 환불이 제한됩니다:\n\n"
            "❌ 결제일로부터 7일 경과\n"
            "❌ Prime 기능 사용 흔적 (최소 1회) 발생 시\n"
            "   • 종목 분석 (전체 종목) 조회\n"
            "   • 포트폴리오 AI 진단 실행\n"
            "   • 백테스트 시뮬레이션 실행\n"
            "   • DART 공시 분석 실행\n"
            "❌ 부정한 방법으로 가입한 경우\n"
            "❌ 약관 위반으로 회사가 계약 해지한 경우\n\n"
            "단, 서비스 장애 등 회사 귀책사유로 7일 이상 정상 이용이 불가능한 경우 사용 여부와 무관하게 환불."
        )
        
        _legal_section("3. 환불 절차",
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
            "   • 무통장 입금: 송금 시 사용한 계좌로 환급"
        )
        
        _legal_section("4. 일부 환불",
            "본 서비스는 월 단위 구독으로, 일부 환불(일할 계산)은 원칙적으로 불가합니다.\n\n"
            "단, 회사 귀책사유로 인한 서비스 중단 시:\n"
            "  • 미사용 일수 × 월 요금 / 30일 환산하여 환불"
        )
        
        _legal_section("5. 결제 수단별 환불 안내",
            "1. 신용·체크카드:\n"
            "   • 결제 취소 처리 → 카드사 정책에 따라 환급\n"
            "   • 통상 영업일 3~7일 소요\n\n"
            "2. 간편결제 (카카오페이/네이버페이/토스페이 등):\n"
            "   • 해당 결제수단으로 환불\n"
            "   • 영업일 1~3일 소요\n\n"
            "3. 계좌이체·가상계좌:\n"
            "   • 사용자 지정 계좌로 송금\n\n"
            "4. 무통장 입금:\n"
            "   • 입금 계좌로 환급\n"
            "   • 단, 송금 수수료(약 500~1,000원)는 회원 부담"
        )
        
        _legal_section("6. 부분 환불 면제 조항",
            "다음의 경우 부분 환불이 적용됩니다:\n\n"
            "1. 회사가 사전 공지 없이 24시간 이상 서비스를 중단한 경우:\n"
            "   • 중단 기간만큼 이용 기한 자동 연장 또는 환불\n\n"
            "2. 서비스 내용이 본질적으로 변경된 경우:\n"
            "   • 회원에게 사전 통지 후 미사용분 환불"
        )
        
        _legal_section("7. 분쟁 해결",
            "환불과 관련된 분쟁은 다음의 순서로 해결합니다:\n\n"
            "1. 회사와의 직접 협의 (영업일 7일)\n"
            "2. 한국소비자원(www.kca.go.kr) 분쟁조정\n"
            "3. 전자상거래분쟁조정위원회\n"
            "4. 회사 본점 소재지 관할 법원"
        )
        
        # 연락처 카드
        with ui.card().classes(
            "w-full p-5 bg-[#1a1a2e] border border-blue-500/40 rounded-xl mt-4"
        ):
            ui.label("📞 환불 문의").classes(
                "text-base font-bold text-blue-300 mb-2"
            )
            ui.label(f"이메일: {BUSINESS_EMAIL}").classes(
                "text-sm text-gray-300"
            )
            if BUSINESS_PHONE:
                ui.label(f"전화: {BUSINESS_PHONE}").classes(
                    "text-sm text-gray-300"
                )
            if BUSINESS_KAKAO:
                with ui.row().classes("w-full items-center gap-2 mt-1"):
                    ui.label(f"카카오톡: {BUSINESS_KAKAO}").classes(
                        "text-sm text-gray-300"
                    )
                    if BUSINESS_KAKAO_URL:
                        ui.button(
                            "💬 채널 바로가기",
                            on_click=lambda url=BUSINESS_KAKAO_URL: ui.navigate.to(
                                url, new_tab=True
                            ),
                        ).props("flat dense color=yellow size=sm").classes("text-xs")
            ui.label("응답 시간: 영업일 1~3일 이내").classes(
                "text-xs text-gray-500 mt-1"
            )
    
    _legal_footer()


# ═══════════════════════════════════════════════════
#  라우트 등록
# ═══════════════════════════════════════════════════
def register_legal_pages():
    """[v22 Step S+AC] 법적 페이지 라우트 등록.
    
    main.py에서 호출:
        from components.legal_pages import register_legal_pages
        register_legal_pages()
    """
    
    @ui.page('/terms')
    async def _terms_page():
        render_terms_page()
    
    @ui.page('/privacy')
    async def _privacy_page():
        render_privacy_page()
    
    @ui.page('/refund')
    async def _refund_page():
        render_refund_page()
    
    # [v22 Step AC] 약관 변경 이력 페이지
    @ui.page('/terms/history')
    async def _terms_history_page():
        try:
            from components.tab_terms import render_terms_history_page
            render_terms_history_page()
        except ImportError:
            ui.label("약관 변경 이력 페이지를 불러올 수 없습니다.").classes(
                "text-red-400 p-6"
            )
