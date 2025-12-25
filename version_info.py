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
_RAW_APP_VERSION = _get_conf("LDY_APP_VERSION", "9.0.0")  # ✅ [Update] v9.0.0 (Deep Insight Edition)
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
        "version": "9.0.0",
        "date": "2025-12-25", # ✅ 크리스마스 선물 업데이트! 🎅
        "title": "Deep Insight Edition: HMA, OBV & Smart Sector",
        "items": [
            "🚀 **Hull Moving Average (HMA):** 기존 이평선보다 반응이 빠르고 휩소가 적은 **HMA 지표**를 탑재하여 추세 전환을 신속하게 포착합니다.",
            "💰 **Smart Money Tracker (OBV):** 주가는 횡보/하락하는데 거래량이 증가하는 **'OBV 다이버전스'**를 감지하여 세력 매집 구간을 식별합니다.",
            "🤖 **AI News Analysis:** Gemini LLM을 활용해 상위 종목의 **최신 호재/악재 뉴스**를 자동으로 수집하고 분석하여 핵심만 요약 제공합니다.",
            "📊 **Excel-Style Portfolio:** 내 자산을 **엑셀처럼 편하게 수정**하고 자동 저장할 수 있는 최신 데이터 에디터를 도입했습니다.",
            "🛡️ **Security & Recovery:** 비밀번호 분실 시 **'보안 질문'**을 통한 재설정 기능 및 관리자용 악성 유저 차단(Ban) 시스템이 추가되었습니다.",
        ],
    },
    {
        "version": "8.5.0",
        "date": "2025-12-19",
        "title": "Institutional-Grade Analytics & Smart Money Management",
        "items": [
            "🟣 **VWAP Analytics Integration:** 기관 투자자의 핵심 지표인 **거래량 가중 평균 가격(VWAP)**을 도입하여 실시간 수급 지지 여부를 정밀 분석합니다.",
            "💰 **Smart Position Sizing:** 총 자산 대비 리스크(2%)를 기반으로 한 **적정 매수 수량 및 금액** 자동 계산 시스템을 구축했습니다.",
            "🕯️ **Price Action Detection:** 차트의 **망치형, 장악형** 등 주요 캔들 패턴을 자동으로 감지하여 기술적 신뢰도를 시각화합니다.",
            "📊 **Admin Performance Insights:** 관리자 전용 대시보드 내에 **DAU(일간 활성자) 및 WAU(주간 활성자)** 통계 집계 기능을 탑재했습니다.",
            "⚙️ **System Optimization:** 로그인 세션 갱신 로직 최적화 및 UI 컴포넌트 디자인 고도화를 진행했습니다.",
        ],
    },
    {
        "version": "8.0.0",
        "date": "2025-12-16",
        "title": "Ironclad Security & Native Performance",
        "items": [
            "🛡️ **Security Shield (Rate Limiting):** 무차별 대입 공격(Brute Force)을 방어하기 위한 **로그인 시도 횟수 제한** 기능 탑재 (5회 실패 시 15분 잠금).",
            "🚀 **Native Turbo Caching:** 기존 수동 캐싱을 Streamlit의 **`st.cache_data` 네이티브 엔진**으로 교체하여 메모리 효율성 극대화 및 로딩 속도 향상.",
            "👤 **User Self-Service (MyPage):** 사용자 스스로 **닉네임 및 비밀번호**를 변경할 수 있는 마이페이지 기능 추가.",
            "🔐 **PBKDF2 Standard:** v7.5에서 도입된 고강도 암호화(PBKDF2-HMAC-SHA256)가 모든 인증 로직에 표준으로 적용.",
            "🔧 **System Stabilization:** 레이스 컨디션 방지를 위한 데이터 로딩/저장 구조 최적화.",
        ],
    },
    {
        "version": "7.5.0",
        "date": "2025-12-15",
        "title": "Smart Swing Stop & V-Power Revolution",
        "items": [
            "🛡️ **Smart Swing Stop-Loss:** 기존 ATR 방식의 한계를 넘어, **최근 10일 전저점(Swing Low)**을 자동으로 인식하여 휩소(Whipsaw)에 의한 불필요한 손절 방지.",
            "💪 **V-Power Factor:** 단순 거래량이 아닌 **'상승일 vs 하락일 거래량 비율'**을 분석하여 세력 매집 강도를 포착하는 **V-Power** 팩터 도입.",
            "📊 **7-Factor Radar Chart:** 기술/세력(TEC) 축을 추가하여 V-Power와 기술적 완성도를 시각화.",
            "🧠 **AI Context Awareness:** 차트의 구조적 지지선 및 세력 매집 패턴을 인식하여 더욱 정교해진 AI 코멘트 제공.",
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
    for i, log in enumerate(CHANGELOG[:limit]):
        ver = log['version']
        date = log['date']
        title = log['title']
        
        # 최신 버전은 'New' 뱃지 효과
        header_prefix = "🔥" if i == 0 else "📜"
        label = f"{header_prefix} v{ver} ({date}): {title}"
        
        with st.expander(label, expanded=(expanded and i == 0)):
            for item in log['items']:
                st.markdown(f"- {item}")


# 버전 정합성 체크
_latest = get_latest_log()
if _latest:
    latest_ver = _latest.get("version")
    if latest_ver and latest_ver != APP_VERSION:
        logger.warning(
            "version_info: APP_VERSION(%s) != CHANGELOG[0](%s). Check consistency.",
            APP_VERSION, latest_ver
        )
            "version_info: APP_VERSION(%s) != CHANGELOG[0](%s). Check consistency.",
            APP_VERSION, latest_ver
        )
