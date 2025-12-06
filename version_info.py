# version_info.py
# -*- coding: utf-8 -*-

import os
import streamlit as st

def _get_conf(key, default_val):
    """
    Streamlit secrets > 환경변수 > 기본값
    순서로 읽는 공통 헬퍼
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except FileNotFoundError:
        # 로컬에서 secrets.toml 없을 때
        pass
    return os.getenv(key, default_val)

# 🔹 PRIME 전용 텔레그램 채널 초대 링크
# 1순위: secrets["LDY_PRIME_JOIN_URL"]
# 2순위: 환경변수 LDY_PRIME_JOIN_URL
# 3순위: 기본값(아래 URL)
PRIME_TG_JOIN_URL = _get_conf(
    "LDY_PRIME_JOIN_URL",
    "https://t.me/+DovDEluWnEJhOTY1",
)

APP_VERSION = "6.5.0"

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
]
