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
_RAW_APP_VERSION = _get_conf("LDY_APP_VERSION", "6.5.0")
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
    {
        "version": "6.4.0",
        "date": "2025-11-30",
        "title": "점수 로직 / 라우팅 개선",
        "items": [
            "LDY_SCORE 스코어링 안정화 및 페널티 구조 조정",
            "ROUTE 태그: BRK / Watch / MR / PULL 기준 재정의",
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
