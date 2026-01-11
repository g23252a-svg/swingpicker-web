# version_info.py (v12.0 Enhanced)
# -*- coding: utf-8 -*-

import os
import logging
import streamlit as st
from typing import List, Dict, Optional, Tuple

# 로깅 설정
logger = logging.getLogger("version_info")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --------------------------------------------------------------------
# 1) 설정 및 버전 상수
# --------------------------------------------------------------------
def _get_conf(key: str, default_val: str) -> str:
    # 1. Streamlit Secrets 확인
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except (FileNotFoundError, AttributeError):
        pass
    # 2. OS 환경 변수 확인
    return os.getenv(key, default_val)

# 현재 앱 버전 설정
_RAW_APP_VERSION = _get_conf("LDY_APP_VERSION", "12.0.0") 
APP_VERSION = _RAW_APP_VERSION

# 텔레그램 채널
PRIME_TG_JOIN_URL = _get_conf("LDY_PRIME_JOIN_URL", "https://t.me/+DovDEluWnEJhOTY1")

# --------------------------------------------------------------------
# 2) CHANGELOG 데이터 (최신순)
# --------------------------------------------------------------------
CHANGELOG: List[Dict] = [
    {
        "version": "12.0.0",
        "date": "2026-01-07",
        "type": "major",  # major, minor, patch, hotfix
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
        "type": "major",
        "title": "The Singularity: Hybrid AI Trading System",
        "items": [
            "🧠 **Auto-ML Engine:** 과거 데이터를 스스로 학습하여 상승 패턴을 예측하는 **머신러닝(Random Forest) 엔진** 탑재",
            "🤝 **AI Consensus Matrix:** 퀀트와 AI가 동시에 강력 추천하는 종목을 찾는 산점도 차트 제공",
            "🏆 **Total Score System:** 퀀트(70%) + AI(30%) 하이브리드 스코어링 도입",
            "⚡ **Robust Data Pipeline:** 외부 데이터 장애 시 로컬 캐시 자동 전환",
        ],
    },
    {
        "version": "10.1.0",
        "date": "2025-12-28",
        "type": "patch",
        "title": "Admin Control & Stability Patch",
        "items": [
            "👮 **Admin Power Tools:** 회원 이용권 만료일 조회 및 필터링 기능 추가",
            "🔐 **Session Integrity:** 관리자 로그인 보안 강화 및 세션 토큰 로직 개선",
            "📉 **Auto Expiration:** 만료 회원 자동 등급 조정 (Prime → Free)",
        ],
    },
]

# --------------------------------------------------------------------
# 3) 버전 유틸리티 함수
# --------------------------------------------------------------------

def get_latest_log() -> Optional[Dict]:
    return CHANGELOG[0] if CHANGELOG else None

def get_version_label() -> str:
    """v12.0.0 형태의 라벨 반환"""
    return f"v{APP_VERSION}"

def is_major_update(ver_str: str) -> bool:
    """버전 문자열을 분석하여 메이저 업데이트인지 확인 (x.0.0)"""
    try:
        parts = ver_str.split(".")
        return len(parts) >= 2 and parts[1] == "0" and parts[2] == "0"
    except:
        return False

# --------------------------------------------------------------------
# 4) UI 렌더링 함수 (핵심 개선)
# --------------------------------------------------------------------

def show_toast_notification():
    """
    세션당 1회, 최신 버전 업데이트 알림을 우측 하단 Toast 메시지로 띄움
    """
    if "has_seen_version_toast" not in st.session_state:
        latest = get_latest_log()
        if latest:
            msg = f"🚀 업데이트 완료! {get_version_label()}: {latest['title']}"
            st.toast(msg, icon="🎉")
        st.session_state["has_seen_version_toast"] = True

def render_sidebar_version_badge():
    """사이드바 상단에 예쁜 버전 배지를 표시"""
    latest = get_latest_log()
    ver = APP_VERSION
    date = latest['date'] if latest else ""
    
    st.sidebar.markdown(f"""
        <div style="
            display: flex; justify-content: space-between; align-items: center;
            background-color: #262730; padding: 10px; border-radius: 8px; margin-bottom: 10px;
            border: 1px solid #444;">
            <div>
                <span style="font-size: 0.8em; color: #aaa;">SYSTEM VERSION</span><br>
                <span style="font-size: 1.1em; font-weight: bold; color: #fff;">{ver}</span>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.7em; background: #FF4B4B; color: white; padding: 2px 6px; border-radius: 4px;">LATEST</span><br>
                <span style="font-size: 0.7em; color: #888;">{date}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_recent_updates(limit: int = 3, expanded: bool = True):
    """
    메인 화면이나 탭에 업데이트 내역을 표시하는 향상된 UI
    """
    st.markdown("### 🧩 시스템 업데이트 내역")
    
    # CSS 스타일 주입 (뱃지 등)
    st.markdown("""
    <style>
        .ver-badge-major { background-color: #FF4B4B; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
        .ver-badge-minor { background-color: #0083B8; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
        .ver-badge-patch { background-color: #2E7D32; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)

    for i, log in enumerate(CHANGELOG[:limit]):
        ver = log['version']
        date = log['date']
        title = log['title']
        u_type = log.get('type', 'patch') # major, minor, patch
        
        # 뱃지 클래스 결정
        badge_class = f"ver-badge-{u_type}" if u_type in ['major', 'minor'] else "ver-badge-patch"
        type_label = u_type.upper()

        # Expander 라벨 구성
        header_emoji = "🚀" if i == 0 else "📜"
        label = f"{header_emoji} [{ver}] {title}"
        
        # 최신 버전은 기본으로 열어둠
        is_expanded = (i == 0 and expanded)
        
        with st.expander(label, expanded=is_expanded):
            # 헤더 정보 (HTML로 예쁘게)
            st.markdown(f"""
                <div style='margin-bottom: 10px;'>
                    <span class='{badge_class}'>{type_label}</span>
                    <span style='color: #888; font-size: 0.9em; margin-left: 10px;'>📅 배포일: {date}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # 항목 리스트
            for item in log['items']:
                st.markdown(f"- {item}")
            
            # 최신 버전 하단 코멘트
            if i == 0 and u_type == 'major':
                st.info("💡 이번 업데이트는 시스템의 핵심 로직이 변경된 대규모 패치입니다. 투자 전략 수립 시 새로운 기능을 적극 활용해보세요!")

# --------------------------------------------------------------------
# 5) 무결성 체크 (실행 시 자동 수행)
# --------------------------------------------------------------------
_latest = get_latest_log()
if _latest:
    latest_ver = _latest.get("version")
    if latest_ver and latest_ver != APP_VERSION:
        logger.warning(f"⚠️ Version Mismatch: Code({APP_VERSION}) != Changelog({latest_ver})")
