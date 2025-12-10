# version_info.py
# -*- coding: utf-8 -*-

import os
import logging
import streamlit as st


logger = logging.getLogger("version_info")
if not logger.handlers:
    # 기본 로깅 설정 (필요하면 최상위에서 다시 세팅해도 됨)
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
        # 로컬에서 .streamlit/secrets.toml 없을 때
        pass
    return os.getenv(key, default_val)


# --------------------------------------------------------------------
# 1) 버전 정보
#    - LDY_APP_VERSION 으로 오버라이드 가능
#    - 예: "6.5.0", "6.5.1-beta", "6.6.0+staging"
# --------------------------------------------------------------------
_RAW_APP_VERSION = _get_conf("LDY_APP_VERSION", "6.8.0")
APP_VERSION = _RAW_APP_VERSION  # 기존 코드 호환용 (대시보드에서 import 하는 값)


def _shorten_version(ver: str) -> str:
    """
    "6.5.0" -> "6.5"
    "6.5.1-beta" -> "6.5"
    "7" -> "7"
    """
    if not ver:
        return ""
    # 빌드/프리릴리즈 태그 제거 (예: 6.5.0-beta+001)
    core = ver.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return core


VERSION_SHORT = _shorten_version(APP_VERSION)  # UI에 간단히 표시하고 싶을 때 사용


# --------------------------------------------------------------------
# 2) PRIME 텔레그램 채널 URL
# --------------------------------------------------------------------
PRIME_TG_JOIN_URL = _get_conf(
    "LDY_PRIME_JOIN_URL",                   # 👉 키 이름 (환경변수/시크릿에서 찾을 이름)
    "https://t.me/+DovDEluWnEJhOTY1",       # 👉 기본값 (아무것도 없으면 이 URL 사용)
)


# --------------------------------------------------------------------
# 3) CHANGELOG
#    - 맨 앞 요소가 항상 최신 버전이라고 가정
# --------------------------------------------------------------------
CHANGELOG = [
    {
        "version": "6.8.0",
        "date": "2025-12-10",
        "title": "Reality Check & Deep Tech",
        "items": [
            "✅ **Reality Check System:** 과거 추천(기준일) 종목의 현재 수익률 성과를 메인 화면에서 자동 검증.",
            "📊 **Advanced Charting:** 볼린저 밴드 / RSI 보조지표를 On/Off 할 수 있는 전문가용 인터랙티브 차트 도입.",
            "🏥 **Portfolio Health Check:** 내 자산의 섹터 편중도와 현금 비중(%)을 분석하여 리스크 관리 조언 제공.",
            "🚀 **Sector Momentum:** 섹터별 최근 수익률/점수 Top 10 바 차트 추가.",
            "🔧 **System Stabilization:** Gist 파일 분리(회원DB/포트폴리오)로 데이터 보존성 강화 및 텔레그램-대시보드 정렬 로직 통일.",
        ],
    },
    {
        "version": "6.7.0",
        "date": "2025-12-08",
        "title": "Prime Top 100 + Role-based Daily Top",
        "items": [
            "게스트/Free/Pro/Prime/Admin 등급별 Daily Top 노출 개수를 분리 (Guest 3개, Free 5개, Pro 20개, Prime/Admin 100개).",
            "Prime/Admin 등급에서 EBS·유동성 통과 종목 풀을 넓혀, 더 많은 후보군 중에서 상위 100개를 열람할 수 있도록 개선.",
            "Daily Top List, 종목 선택 드롭다운, 개별 차트/레이더/리스크-리워드 분석이 동일한 필터 조건(LDY 점수, ROUTE, REGIME)에 따라 일관되게 동작하도록 구조 정리.",
            "필터가 과도하게 좁거나 데이터가 부족한 상황에서도 에러 없이 빈 상태 안내/경고만 출력되도록 방어 코드 보강.",
        ],
    },
    {
        "version": "6.6.0",
        "date": "2025-12-07",
        "title": "Data Freshness + Market Snapshot",
        "items": [
            "GitHub 원격 CSV / 로컬 캐시 데이터 출처 태그(remote/local) 표시.",
            "추천 데이터 기준일(기준일자/날짜/Date 컬럼 자동 인식) 추론 및 2일 이상 경과 시 신선도 경고 출력.",
            "FDR/pykrx 지수 기반 KOSPI/KOSDAQ Market Snapshot + 데이터 장애 시 로컬 스코어 기반으로 자동 Fallback.",
            "KS11 지수 기반 공포/탐욕 지수 + scored DF 기반 Fallback을 통합한 공포/탐욕 인덱스 구현.",
        ],
    },
    {
        "version": "6.5.0",
        "date": "2025-12-06",
        "title": "Collector v6.5 / 계정 시스템 안정화",
        "items": [
            "Collector v6.5: 60일 지수 수익률·상대강도(α) 반영",
            "KOSPI/KOSDAQ 지수 fallback 로직 개선 (직전 영업일 자동 탐색)",
            "회원 DB를 GitHub Gist + 로컬 캐시 구조로 안정화",
            "로그인/회원가입 시 이메일 소문자 통일 + 형식 검증 추가",
        ],
    },
    # 필요하면 과거 버전 계속 추가
]


# --------------------------------------------------------------------
# 4) Changelog / 버전 유틸
# --------------------------------------------------------------------
def get_latest_log():
    """
    최신(맨 위) CHANGELOG 항목 반환.
    CHANGELOG가 비어 있으면 None.
    """
    return CHANGELOG[0] if CHANGELOG else None


def find_changelog(version: str):
    """
    특정 버전에 해당하는 changelog 항목을 찾아 반환.
    없으면 None.
    """
    if not version:
        return None
    for log in CHANGELOG:
        if log.get("version") == version:
            return log
    return None


def get_version_label(include_build: bool = True) -> str:
    """
    UI에 표시할 버전 문자열 포맷터.
    - include_build=True  -> "6.5.0"
    - include_build=False -> "6.5"
    """
    return APP_VERSION if include_build else VERSION_SHORT


# 모듈 import 시점에 changelog와 버전이 일치하는지 한 번 점검
_latest = get_latest_log()
if _latest:
    latest_ver = _latest.get("version")
    if latest_ver and latest_ver != APP_VERSION:
        logger.warning(
            "version_info: APP_VERSION(%s)와 CHANGELOG[0].version(%s)이 일치하지 않습니다. "
            "버전 정합성을 확인해 주세요.",
            APP_VERSION,
            latest_ver,
        )
