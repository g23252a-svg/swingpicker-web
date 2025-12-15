# version_info.py
# -*- coding: utf-8 -*-

import os
import logging
import streamlit as st


logger = logging.getLogger("version_info")
if not logger.handlers:
    # 기본 로깅 설정
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _get_conf(key, default_val):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except FileNotFoundError:
        pass
    return os.getenv(key, default_val)


# --------------------------------------------------------------------
# 1) 버전 정보
#    - LDY_APP_VERSION 으로 오버라이드 가능
# --------------------------------------------------------------------
_RAW_APP_VERSION = _get_conf("LDY_APP_VERSION", "7.5.0") # ✅ [Update] v7.5.0
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
#    - 맨 앞 요소가 항상 최신 버전이라고 가정
# --------------------------------------------------------------------
CHANGELOG = [
    {
        "version": "7.5.0", # ✅ [New] v7.5
        "date": "2025-12-15",
        "title": "Smart Swing Stop & V-Power Revolution",
        "items": [
            "🛡️ **Smart Swing Stop-Loss:** 기존 ATR 방식의 한계를 넘어, **최근 10일 전저점(Swing Low)**을 자동으로 인식하여 휩소(Whipsaw)에 의한 불필요한 손절을 방지.",
            "💪 **V-Power Factor:** 단순 거래량이 아닌 **'상승일 vs 하락일 거래량 비율'**을 분석하여, 주가 횡보 중에도 세력의 매집 강도를 포착하는 **V-Power(매수체결강도)** 팩터 도입.",
            "📊 **7-Factor Radar Chart:** 기존 6각 레이더에 **'기술/세력(TEC)' 축을 추가**하여 V-Power와 기술적 완성도를 한눈에 시각화.",
            "🧠 **AI Context Awareness:** 차트의 **구조적 지지선 근접 여부**와 **세력 매집 패턴**을 인식하여 더욱 정교해진 AI 코멘트 제공.",
            "⚡ **Auth System V2:** Gist DB 로딩 속도를 획기적으로 개선하는 **캐싱 시스템(st.cache_data)** 적용 및 **PBKDF2-HMAC-SHA256** 암호화로 보안성 강화.",
            "🧹 **Code Clean-up:** 레거시 코드(로컬 파일 I/O 등)를 대거 정리하여 시스템 안정성 및 유지보수성 향상.",
        ],
    },
    {
        "version": "7.4.0",
        "date": "2025-12-14",
        "title": "Context-Aware AI Quant System",
        "items": [
            "🧠 **Dynamic Regime Weighting:** 시장 국면(🔥과열/🧊침체/🌤중립)을 자동 인식하여, 추세(MOM) vs 방어(RR/SL) 가중치를 실시간으로 최적화.",
            "🛡️ **Adaptive Stop-Loss:** 변동성 국면(Squeeze vs High Vol)에 따라 손절폭(ATR Multiplier 1.8~2.5)을 유동적으로 조절하여 휩소 방지 및 리스크 관리 강화.",
            "⏳ **Squeeze Duration Tracking:** TTM Squeeze의 에너지 응축 기간(일수)을 추적하여 '폭발 임박(Hot Zone)' 타이밍을 정밀 포착.",
            "⚡ **Hyper-Speed Data Engine:** 데이터 배치 수집 및 캐싱 시스템(Pickle) 도입으로 재실행 속도 비약적 향상.",
            "📊 **6-Factor Radar Chart:** 정규화된 6가지 팩터(모멘텀, 가성비, 수익여력, 안전성, 타점, 수급)를 기반으로 투명하고 직관적인 종목 분석 제공.",
            "📈 **Multi-Period Alpha:** 20일(단기), 60일(중기), 120일(장기) 벤치마크 대비 초과 수익률(Alpha)을 종합 분석하여 추세 판단력 고도화.",
        ],
    },
    # ... (과거 로그 생략) ...
]


# --------------------------------------------------------------------
# 4) Changelog / 버전 유틸
# --------------------------------------------------------------------
def get_latest_log():
    return CHANGELOG[0] if CHANGELOG else None


def find_changelog(version: str):
    if not version:
        return None
    for log in CHANGELOG:
        if log.get("version") == version:
            return log
    return None


def get_version_label(include_build: bool = True) -> str:
    return APP_VERSION if include_build else VERSION_SHORT


# 버전 정합성 체크
_latest = get_latest_log()
if _latest:
    latest_ver = _latest.get("version")
    if latest_ver and latest_ver != APP_VERSION:
        logger.warning(
            "version_info: APP_VERSION(%s) != CHANGELOG[0](%s). Check consistency.",
            APP_VERSION, latest_ver
        )
