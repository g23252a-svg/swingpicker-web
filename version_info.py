# version_info.py (v12.0)
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
# --------------------------------------------------------------------
_RAW_APP_VERSION = _get_conf("LDY_APP_VERSION", "12.0.0")  # ✅ [Major Update] v12.0.0 (AI + Kelly + DART)
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
        "version": "12.0.0",
        "date": "2026-01-07",
        "title": "Quantum Leap: AI Trust & Money Management",
        "items": [
            "💰 **Kelly Betting System:** '얼마를 걸어야 할까?'에 대한 수학적 해답. 승률과 손익비를 계산하여 **최적의 자금 관리 비중**을 제안합니다.",
            "👁️ **AI Trust Gauge:** 복잡한 수치 대신 직관적인 **속도계(Gauge) 차트**로 AI와 퀀트가 합의한 종합 신뢰도를 한눈에 보여줍니다.",
            "📜 **DART Deep Dive:** 단순 뉴스 키워드 매칭을 넘어, **공시 원문을 뜯어보고 호재/악재를 판별**하는 심층 분석 엔진을 탑재했습니다.",
            "🧪 **Strategy Lab:** (별도 실행) 내가 세운 가설(RSI<30 등)이 과거에 실제로 통했는지 검증하는 **백테스팅 시뮬레이터**가 추가되었습니다.",
        ],
    },
    {
        "version": "11.0.0",
        "date": "2025-12-29",
        "title": "The Singularity: Hybrid AI Trading System",
        "items": [
            "🧠 **Auto-ML Engine:** 과거 데이터를 스스로 학습하여 상승 패턴을 예측하는 **머신러닝(Random Forest) 엔진**을 탑재했습니다. (AI 점수 도입)",
            "🤝 **AI Consensus Matrix:** 기존 퀀트 로직과 AI 예측 모델이 **동시에 강력 추천하는 종목**을 한눈에 찾는 산점도 차트를 제공합니다.",
            "🏆 **Total Score System:** 퀀트 점수(70%)와 AI 점수(30%)를 결합한 **'종합 점수'**로 랭킹 정확도를 비약적으로 높였습니다.",
            "⚡ **Robust Data Pipeline:** KRX/FDR 외부 데이터 장애 시에도 **로컬 캐시를 우선 사용하여 중단 없는 서비스**를 제공합니다.",
        ],
    },
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
    # ... (과거 로그 생략 가능) ...
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
                st.caption("✨ 이제 AI와 퀀트가 함께 시장을 분석합니다. 압도적인 성능을 경험해보세요!")


# 버전 정합성 체크
_latest = get_latest_log()
if _latest:
    latest_ver = _latest.get("version")
    if latest_ver and latest_ver != APP_VERSION:
        logger.warning(f"version_info: APP_VERSION({APP_VERSION}) != CHANGELOG[0]({latest_ver})")
