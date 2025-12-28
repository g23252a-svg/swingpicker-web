# version_info.py
# -*- coding: utf-8 -*-

import os
import logging
import streamlit as st
from typing import List, Dict, Optional

logger = logging.getLogger("version_info")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _get_conf(key: str, default_val: str) -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except (FileNotFoundError, AttributeError):
        pass
    return os.getenv(key, default_val)


# --------------------------------------------------------------------
# 1) 버전 정보
#    - LDY_APP_VERSION 으로 오버라이드 가능
# --------------------------------------------------------------------
_RAW_APP_VERSION = _get_conf("LDY_APP_VERSION", "10.1.0")  # ✅ [Update] v10.1.0 (Weekly Trend & Security)
APP_VERSION = _RAW_APP_VERSION


def _shorten_version(ver: str) -> str:
    if not ver:
        return ""
    core = ver.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return core


VERSION_SHORT = _shorten_version(APP_VERSION)


# --------------------------------------------------------------------
# 2) PRIME 텔레그램 채널 URL
# --------------------------------------------------------------------
PRIME_TG_JOIN_URL = _get_conf(
    "LDY_PRIME_JOIN_URL",
    "https://t.me/+DovDEluWnEJhOTY1",  # 기본값
)


# --------------------------------------------------------------------
# 3) CHANGELOG
#    - 맨 앞 요소가 항상 최신 버전
# --------------------------------------------------------------------
CHANGELOG: List[Dict] = [
    {
        "version": "10.1.0",
        "date": "2025-12-28",
        "title": "Admin Control & Stability Patch",
        "items": [
            "👮 **Admin Power Tools:** 관리자 패널에서 회원의 **'이용권 만료일'**을 즉시 확인하고, 만료된 회원만 필터링하여 관리할 수 있습니다.",
            "🔐 **Session Integrity Fix:** 관리자(Admin) 계정의 로그인 안정성을 확보하고, 일반 유저의 중복 로그인을 방지하는 세션 토큰 로직을 강화했습니다.",
            "📉 **Auto Expiration Handling:** 이용 기간이 만료된 회원이 접속 시, 자동으로 등급이 조정(Prime→Free)되며 DB에 즉시 반영됩니다.",
        ],
    },
    {
        "version": "10.0.0",
        "date": "2025-12-28",
        "title": "Weekly Trend Traffic Light & Ironclad Security",
        "items": [
            "🚥 **Weekly Trend Traffic Light:** 일봉의 노이즈를 제거한 **'주봉 20선 대추세'** 신호등 UI를 도입했습니다. (초록: 상승장 / 빨강: 하락장)",
            "🛡️ **Instant Ban System:** 관리자가 차단(Ban) 버튼을 누르는 즉시, 해당 유저의 화면이 종료되고 강제 로그아웃되는 실시간 보안 체계를 구축했습니다.",
            "📈 **Weekly Overlay:** 차트에 은은한 **주봉 20일선 점선**을 추가하여, 현재 주가가 대추세 대비 어디에 위치하는지 직관적으로 파악할 수 있습니다.",
            "🔑 **Secure Session Token:** 기기별 고유 세션 토큰을 발급하여 세션 탈취 및 불법 공유를 원천 차단합니다.",
        ],
    },
    {
        "version": "9.0.0",
        "date": "2025-12-25",
        "title": "Deep Insight Edition: HMA, OBV & Smart Sector",
        "items": [
            "🚀 **Hull Moving Average (HMA):** 기존 이평선보다 반응이 빠르고 휩소가 적은 HMA 지표 탑재",
            "💰 **Smart Money Tracker (OBV):** 거래량 다이버전스를 통한 세력 매집 구간 식별",
            "🤖 **AI News Analysis:** Gemini LLM 기반 최신 호재/악재 뉴스 자동 요약",
            "📊 **Excel-Style Portfolio:** 엑셀처럼 수정 가능한 포트폴리오 에디터 도입",
            "🛡️ **Security & Recovery:** 보안 질문을 통한 비밀번호 찾기 기능 추가",
        ],
    },
    # ... (과거 로그 생략) ...
]


# --------------------------------------------------------------------
# 4) Changelog / 버전 유틸
# --------------------------------------------------------------------
def get_latest_log() -> Optional[Dict]:
    return CHANGELOG[0] if CHANGELOG else None


def find_changelog(version: str) -> Optional[Dict]:
    if not version:
        return None
    for log in CHANGELOG:
        if log.get("version") == version:
            return log
    return None


def get_version_label(include_build: bool = True) -> str:
    return APP_VERSION if include_build else VERSION_SHORT


# ✅ UI Helper: 사이드바나 메인화면에 업데이트 내역 표시
def show_recent_updates(limit: int = 1, expanded: bool = True):
    """최신 업데이트 내역을 Streamlit Expander로 렌더링"""
    
    # 스타일링을 위한 CSS (선택 사항)
    st.markdown("""
        <style>
        .update-badge {
            background-color: #FF4B4B;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    for i, log in enumerate(CHANGELOG[:limit]):
        ver = log['version']
        date = log['date']
        title = log['title']
        
        # 최신 버전은 아이콘 강조
        header_icon = "🚀" if i == 0 else "📜"
        label = f"{header_icon} v{ver} ({date}): {title}"
        
        with st.expander(label, expanded=(expanded and i == 0)):
            for item in log['items']:
                st.markdown(f"- {item}")
            
            if i == 0:
                st.caption("✨ 최신 기능이 적용되었습니다. 강력해진 보안과 트렌드 분석을 경험해보세요!")


# 버전 정합성 체크
_latest = get_latest_log()
if _latest:
    latest_ver = _latest.get("version")
    if latest_ver and latest_ver != APP_VERSION:
        # 안전하게 한 줄로 작성 (들여쓰기 오류 방지)
        logger.warning(f"version_info: APP_VERSION({APP_VERSION}) != CHANGELOG[0]({latest_ver})")
