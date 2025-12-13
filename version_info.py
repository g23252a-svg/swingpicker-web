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
    """
    공통 설정 헬퍼
    - 1순위: Streamlit secrets
    - 2순위: 환경변수(os.environ)
    - 3순위: 기본값
    """
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
_RAW_APP_VERSION = _get_conf("LDY_APP_VERSION", "6.9.2")
APP_VERSION = _RAW_APP_VERSION


def _shorten_version(ver: str) -> str:
    """
    "6.9.0" -> "6.9"
    "6.9.1-beta" -> "6.9"
    """
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
    {
        "version": "6.9.1",
        "date": "2025-12-12",
        "title": "Data Polish Patch",
        "items": [
            "💱 **KRX 호가단위 적용:** 추천매수가/손절가/목표가를 KRX 호가단위(틱) 기준으로 자동 라운딩하여 유효 호가로 정규화.",
            "💰 **가격 표시 품질 개선:** 종가·추천가 등 원화 금액을 천 단위 콤마 + '원'으로 통일 표시.",
            "🧼 **디버그 UI 제거:** 사용자 화면에서 데이터 디버그 패널을 제거하여 대시보드 집중도 개선.",
        ],
    },
    {
        "version": "6.9.0",
        "date": "2025-12-12",
        "title": "Deep Insight & Squeeze Hunter",
        "items": [
            "🌡️ **Market Breadth Gauge:** 시장 전체 종목 중 20일 이동평균선 상회 비율(%)을 시각화하여 지수 왜곡 없는 진짜 시장 온도를 측정.",
            "⚡ **Squeeze Hunter:** 볼린저 밴드폭(BandWidth)이 15% 미만으로 축소된 '변동성 폭발 임박' 종목을 원클릭 필터링.",
            "📊 **Advanced Charting v2:** 차트에 BandWidth 보조지표를 추가하여 변동성 축소/확대 구간을 직관적으로 확인.",
            "🔥 **Sector Heatmap Upgrade:** 단순 거래대금 크기가 아닌, 섹터별 평균 등락률(모멘텀)을 색상으로 입혀 주도 섹터 식별력 강화.",
            "🔐 **Auth Core Update:** PBKDF2-HMAC-SHA256 해시 적용 및 로그인/로그아웃 시 자동 새로고침(UX 개선).",
        ],
    },
    {
        "version": "6.8.0",
        "date": "2025-12-10",
        "title": "Reality Check & Deep Tech",
        "items": [
            "✅ **Reality Check System:** 과거 추천(기준일) 종목의 현재 수익률 성과를 메인 화면에서 자동 검증.",
            "📊 **Advanced Charting:** 볼린저 밴드 / RSI 보조지표를 On/Off 할 수 있는 전문가용 인터랙티브 차트 도입.",
            "🏥 **Portfolio Health Check:** 내 자산의 섹터 편중도와 현금 비중(%)을 분석하여 리스크 관리 조언 제공.",
            "🚀 **Sector Momentum:** 섹터별 최근 수익률/점수 Top 10 바 차트 추가.",
            "🔧 **System Stabilization:** Gist 기반 데이터 보존성 강화(회원DB/포트폴리오 등).",
        ],
    },
    {
        "version": "6.7.0",
        "date": "2025-12-08",
        "title": "Prime Top 100 + Role-based Daily Top",
        "items": [
            "등급별 Daily Top 노출 개수 분리 (Guest 3개, Free 5개, Pro 20개, Prime/Admin 100개).",
            "Prime/Admin 등급 전용 EBS·유동성 통과 종목 풀 확대 (최대 100개).",
            "필터링 구조 통일 (Daily Top, 차트, 분석 도구 간 데이터 동기화).",
        ],
    },
    {
        "version": "6.6.0",
        "date": "2025-12-07",
        "title": "Data Freshness + Market Snapshot",
        "items": [
            "GitHub 원격 CSV / 로컬 캐시 데이터 출처 태그(remote/local) 표시.",
            "추천 데이터 기준일 자동 추론 및 신선도 경고 시스템.",
            "FDR/pykrx 지수 기반 Market Snapshot 및 공포/탐욕 지수 통합.",
        ],
    },
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
