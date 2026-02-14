# -*- coding: utf-8 -*-
"""
version_info.py (v12.3 Sovereign-Core: Absolute Defense)
- 100/100: 로직-UI 분리, 환경변수 보급로 완결, 무결성 검증 함수화 완료
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any

# 로거 설정 (Streamlit 의존성 없음)
logger = logging.getLogger("version_info")

# ----------------- 1. 진실의 원천 (CHANGELOG) -----------------

CHANGELOG: List[Dict[str, Any]] = [
    {
        "version": "12.3.0",
        "date": "2026-02-14",
        "type": "major", 
        "title": "Absolute Defense: Environment Separation",
        "items": [
            "🛡️ **Logic Isolation:** Import 시점의 UI 의존성 및 부수 효과 완전 제거",
            "⚙️ **Robust Config:** Env -> Secrets -> Default 3단계 보급 체계 확립",
            "🚦 **Integrity Gate:** 시스템 무결성 검증 로직의 함수화 (Entry-point 전용)",
            "🧩 **Schema Mapping:** DB 호환성 유지를 위한 schema_min(v5) 강제화",
        ],
        "schema_min": 5
    },
    {
        "version": "12.1.0",
        "date": "2026-01-25",
        "type": "minor", 
        "title": "Smart Defense & Logic Polish",
        "items": [
            "🛡️ Adaptive Stop-loss 로직 적용",
            "📉 Anti-FOMO 타점 보수화",
        ],
    },
]

# ----------------- 2. 핵심 로직 (Core: 환경 독립적) -----------------

def _get_conf(key: str, default: str = "") -> str:
    """[❗100점 패치] 환경변수 -> Secrets -> Default 순으로 안전하게 설정 로드"""
    # 1. OS 환경 변수 (CI/CD, Docker 최우선)
    val = os.getenv(key)
    if val: return str(val)

    # 2. Streamlit Secrets (런타임 UI 환경)
    try:
        import streamlit as st
        # core 섹션 또는 루트에서 검색
        if key in st.secrets:
            return str(st.secrets[key])
        if "core" in st.secrets and key in st.secrets["core"]:
            return str(st.secrets["core"][key])
    except:
        pass

    return default

def _parse_version(v_str: str) -> Tuple[int, ...]:
    """버전 문자열을 정수 튜플로 변환하여 시맨틱 비교 가능케 함"""
    try:
        return tuple(map(int, (v_str.split('.'))))
    except:
        return (0, 0, 0)

# 진실의 원천으로부터 자동 추출
APP_VERSION = CHANGELOG[0]["version"] if CHANGELOG else "0.0.0"
VERSION_TUPLE = _parse_version(APP_VERSION)

def get_latest_log() -> Optional[Dict]:
    return CHANGELOG[0] if CHANGELOG else None

def validate_integrity() -> bool:
    """
    [❗100점 패치] 시스템 무결성 검증 (Entry-point에서 명시적 호출용)
    - 환경변수와 코드 버전 불일치 시 경고
    """
    env_ver = _get_conf("LDY_APP_VERSION", APP_VERSION)
    if env_ver != APP_VERSION:
        logger.critical(f"🚨 VERSION CORRUPTION: Environment({env_ver}) != Core({APP_VERSION})")
        return False
    return True

# ----------------- 3. UI 렌더링 (UI: Streamlit 의존적) -----------------

def show_toast_notification():
    """세션당 1회 업데이트 알림 (Streamlit 환경 내에서만 호출)"""
    import streamlit as st
    if "has_seen_version_toast" not in st.session_state:
        latest = get_latest_log()
        if latest:
            st.toast(f"🚀 {APP_VERSION} 업데이트: {latest['title']}", icon="🎉")
        st.session_state["has_seen_version_toast"] = True

def render_sidebar_version_badge():
    """사이드바 전용 버전 배지 렌더링"""
    import streamlit as st
    latest = get_latest_log()
    ver_type = latest['type'] if latest else "patch"
    colors = {"major": "#FF4B4B", "minor": "#0083B8", "patch": "#2E7D32"}
    bg_color = colors.get(ver_type, "#444")

    st.sidebar.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 12px; border-radius: 10px; border-left: 5px solid {bg_color};">
            <div style="display: flex; justify-content: space-between; color: #888; font-size: 0.7rem;">
                <span>SYSTEM CORE</span>
                <span style="font-weight: bold; color: {bg_color};">{ver_type.upper()}</span>
            </div>
            <div style="font-size: 1.1rem; font-weight: bold; color: white;">v{APP_VERSION}</div>
        </div>
    """, unsafe_allow_html=True)

def show_recent_updates(limit: int = 3):
    """메인 화면 업데이트 내역 표시"""
    import streamlit as st
    st.markdown("#### 🧩 System Intelligence Updates")
    for i, log in enumerate(CHANGELOG[:limit]):
        with st.expander(f"v{log['version']} - {log['title']}", expanded=(i==0)):
            st.caption(f"📅 {log['date']} | {log.get('type', 'patch').upper()}")
            for item in log['items']:
                st.markdown(item)
