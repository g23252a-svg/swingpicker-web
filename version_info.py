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
_RAW_APP_VERSION = _get_conf("LDY_APP_VERSION", "7.4.0") # ✅ [업데이트] 7.4.0
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
        "version": "7.4.0", # ✅ [New] v7.4 추가
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
    {
        "version": "6.9.3",
        "date": "2025-12-13",
        "title": "TTM Squeeze & Interactive Telegram",
        "items": [
            "🌪️ **TTM Squeeze Engine:** 존 카터의 TTM Squeeze 알고리즘 탑재 (Bollinger Bands가 Keltner Channel 안으로 수렴 시 '🔥 SQZ' 포착).",
            "📉 **Keltner Channel Chart:** 대시보드 차트에 켈트너 채널(1.5 ATR) 표시 옵션을 추가하여 스퀴즈 상태를 시각적으로 확인 가능.",
            "🚀 **Telegram On-Demand:** 대시보드에서 현재 필터링된 상위 5개 종목을 즉시 텔레그램으로 전송하는 버튼 추가.",
            "🏷️ **Strategy Tag Upgrade:** '🔥 SQZ (폭발대기)' 전략 태그 추가로 변동성 폭발 임박 종목 우선 순위 노출.",
        ],
    },
    # ... (기존 로그들은 그대로 유지) ...
    {
        "version": "6.9.2",
        "date": "2025-12-13",
        "title": "Gist DB Split (Users / Subscriptions / Inquiries)",
        "items": [
            "🧩 **DB 분리:** 회원 DB(users)와 별개로 **구독 DB(subscriptions)**, **문의 DB(inquiries)** 를 각각 Secret Gist로 분리.",
            "🔐 **Secrets 확장:** `LDY_GIST_SUBS_ID`, `LDY_GIST_INQ_ID` 지원(미설정 시 기존 `LDY_GIST_ID`로 자동 fallback).",
            "🕒 **updated_at 표준화:** subscriptions/inquiries 저장 시 `updated_at`을 UTC ISO8601으로 자동 갱신.",
            "🛡️ **호환성 유지:** 기존 dashboard 로직이 list 형태 문의를 쓰는 경우를 위한 helper(예: inquiry items)로 점진 전환 가능.",
        ],
    },
    # ... (이하 생략) ...
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
