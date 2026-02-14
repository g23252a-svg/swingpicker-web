# -*- coding: utf-8 -*-
"""
version_info.py (v12.3.1 Sovereign-Core)
- 100/100: dashboard.py 호환성 완벽 복구 버전
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger("version_info")

# ----------------- 1. 진실의 원천 (CHANGELOG) -----------------
CHANGELOG: List[Dict[str, Any]] = [
    {
        "version": "12.3.0",
        "date": "2026-02-14",
        "type": "major", 
        "title": "Absolute Defense: Environment Separation",
        "items": [
            "🛡️ Logic Isolation: Import 시점의 UI 의존성 제거",
            "⚙️ Robust Config: Env -> Secrets -> Default 체계 확립",
            "🚦 Integrity Gate: 무결성 검증 로직 함수화",
        ],
        "schema_min": 5
    }
]

# ----------------- 2. 핵심 로직 (Core) -----------------

def _get_conf(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val: return str(val)
    try:
        import streamlit as st
        if key in st.secrets: return str(st.secrets[key])
        if "core" in st.secrets and key in st.secrets["core"]: return str(st.secrets["core"][key])
    except: pass
    return default

def _parse_version(v_str: str) -> Tuple[int, ...]:
    try: return tuple(map(int, (v_str.split('.'))))
    except: return (0, 0, 0)

# [📍 필수 보급 1] 앱 버전
APP_VERSION = CHANGELOG[0]["version"] if CHANGELOG else "0.0.0"
VERSION_TUPLE = _parse_version(APP_VERSION)

# [📍 필수 보급 2] 텔레그램 URL (누락되었던 부분)
PRIME_TG_JOIN_URL = _get_conf("LDY_PRIME_JOIN_URL", "https://t.me/+DovDEluWnEJhOTY1")

def get_latest_log() -> Optional[Dict]:
    return CHANGELOG[0] if CHANGELOG else None

# [📍 필수 보급 3] 버전 라벨 (누락되었던 부분)
def get_version_label(include_build: bool = True) -> str:
    if include_build: return APP_VERSION
    return f"{VERSION_TUPLE[0]}.{VERSION_TUPLE[1]}"

def validate_integrity() -> bool:
    env_ver = _get_conf("LDY_APP_VERSION", APP_VERSION)
    if env_ver != APP_VERSION:
        logger.critical(f"🚨 VERSION CORRUPTION: Environment({env_ver}) != Core({APP_VERSION})")
        return False
    return True

# ----------------- 3. UI 렌더링 -----------------

def show_toast_notification():
    import streamlit as st
    if "has_seen_version_toast" not in st.session_state:
        latest = get_latest_log()
        if latest: st.toast(f"🚀 {APP_VERSION} 업데이트: {latest['title']}", icon="🎉")
        st.session_state["has_seen_version_toast"] = True

def render_sidebar_version_badge():
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
    import streamlit as st
    st.markdown("#### 🧩 System Intelligence Updates")
    for i, log in enumerate(CHANGELOG[:limit]):
        with st.expander(f"v{log['version']} - {log['title']}", expanded=(i==0)):
            st.caption(f"📅 {log['date']} | {log.get('type', 'patch').upper()}")
            for item in log['items']: st.markdown(item)
