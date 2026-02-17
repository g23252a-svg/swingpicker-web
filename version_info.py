# -*- coding: utf-8 -*-
"""
version_info.py (v18.0 Premium-UI)
- CHANGELOG: 전체 업데이트 이력 관리
- UI: 프리미엄 사이드바 뱃지, 업데이트 타임라인
- 100/100: dashboard.py에서 요청하는 모든 물자(변수/함수) 완비
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger("version_info")

# ----------------- 1. 진실의 원천 (CHANGELOG) -----------------
CHANGELOG: List[Dict[str, Any]] = [
    {
        "version": "18.0.0",
        "date": "2026-02-17",
        "type": "major",
        "title": "Premium UI + ML v17.1 + Critical Bug Fixes",
        "items": [
            "🎨 **Premium Dark Theme:** Outfit 폰트, 글래스모피즘 카드, 그라디언트 메트릭, 탭 하이라이트 등 전면 UI 리뉴얼",
            "💎 **Hero Banner:** 기존 st.title → 그라디언트 히어로 배너 + AI 엔진 스펙 뱃지",
            "🃏 **Kanban 2.0:** 칸반 카드 글래스모피즘 + 점수별 색상 뱃지 + R:R 시각 바 + 호버 부유 효과",
            "🧠 **ML v17.1:** 피처 버전 메타데이터(trading_meta_v17.json) 저장 → 코드/모델 불일치 시 레거시 자동 폴백",
            "🔒 **Thread Safety:** load_model()에 threading.Lock 적용 → ThreadPoolExecutor 레이스 방지",
            "🐛 **매수가 중복 계산 Fix:** R:R 2.0→2.5 복원, 7%↑ 급등주 추격방지 정상화",
            "🚫 **OVERHEAT 필터:** RSI≥75 과열 종목이 정예군 Top120에서 자동 제외",
            "🤖 **ML 피처 확장:** 6→16개 (MFI, MACD_Hist, BB_Width, ATR, OBV_Slope 등)",
            "📊 **XGBoost 앙상블:** LSTM 60% + XGBoost 40% 가중평균, Focal Loss 적용",
            "🔑 **Secrets 매핑 테이블:** TOML 중첩 섹션 → 환경변수 확정 매핑 (telegram_token → TG_TOKEN)",
            "🗄️ **DB 컬럼 Fix:** users 테이블 14컬럼 vs INSERT 12값 불일치 → 컬럼명 명시로 해결",
            "🔍 **Gist 진단 패널:** 회원정보 미로드 시 10단계 진단 UI 자동 표시",
            "📈 **게이지 차트:** 투명 다크 배경, 4단계 구간 색상, Outfit 폰트 적용",
            "👑 **관리자 통계:** DAU/WAU를 3색 그라디언트 카드로 시각화",
            "🧪 **Unit Tests:** test_ml_engine.py 8개 테스트 케이스 (차원 불일치, 모델 없음, 정상 추론 등)",
        ],
        "schema_min": 5
    },
    {
        "version": "12.3.1",
        "date": "2026-02-14",
        "type": "major",
        "title": "Absolute Defense: Fix Import Issue",
        "items": [
            "🛡️ **Logic Isolation:** Import 시점의 UI 의존성 제거",
            "⚙️ **Robust Config:** 보급로(PRIME_TG_JOIN_URL) 변수 확립",
            "🚦 **Integrity Gate:** 무결성 검증 로직 함수화",
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

# [📍 핵심 보급품] 앱 버전
APP_VERSION = CHANGELOG[0]["version"] if CHANGELOG else "18.0.0"
VERSION_TUPLE = _parse_version(APP_VERSION)

# [📍 핵심 보급품] 텔레그램 조인 URL
PRIME_TG_JOIN_URL = _get_conf("LDY_PRIME_JOIN_URL", "https://t.me/+DovDEluWnEJhOTY1")

def get_latest_log() -> Optional[Dict]:
    return CHANGELOG[0] if CHANGELOG else None

# [📍 핵심 보급품] 버전 라벨 함수
def get_version_label(include_build: bool = True) -> str:
    if include_build:
        return APP_VERSION
    return f"{VERSION_TUPLE[0]}.{VERSION_TUPLE[1]}"

def validate_integrity() -> bool:
    env_ver = _get_conf("LDY_APP_VERSION", APP_VERSION)
    if env_ver != APP_VERSION:
        logger.critical(f"🚨 VERSION CORRUPTION: Environment({env_ver}) != Core({APP_VERSION})")
        return False
    return True

# ----------------- 3. 유틸리티 -----------------

def _hex_to_rgb(hex_color: str) -> str:
    """#3B82F6 → '59,130,246' (CSS rgba용)"""
    h = hex_color.lstrip('#')
    try:
        return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
    except:
        return "100,100,100"

# ----------------- 4. UI 렌더링 (Streamlit 기반) -----------------

def show_toast_notification():
    import streamlit as st
    if "has_seen_version_toast" not in st.session_state:
        latest = get_latest_log()
        if latest: st.toast(f"🚀 v{APP_VERSION} 업데이트!", icon="💎")
        st.session_state["has_seen_version_toast"] = True


def render_sidebar_version_badge():
    """[v18.0] 사이드바 버전 뱃지 — 프리미엄 디자인"""
    import streamlit as st
    latest = get_latest_log()
    ver_type = latest['type'] if latest else "patch"
    ver_date = latest.get('date', '') if latest else ''
    item_count = len(latest.get('items', [])) if latest else 0

    colors = {"major": "#3B82F6", "minor": "#06B6D4", "patch": "#10B981"}
    bg_color = colors.get(ver_type, "#64748B")
    type_label = {"major": "MAJOR", "minor": "MINOR", "patch": "PATCH"}.get(ver_type, "UPDATE")

    st.sidebar.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(30,30,30,0.95), rgba(20,20,30,0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 4px solid {bg_color};
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 8px;
            font-family: 'Outfit', -apple-system, sans-serif;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.65rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.08em;">
                    LDY Pro Trader
                </span>
                <span style="
                    font-size:0.6rem; font-weight:600; color:{bg_color};
                    background:rgba({_hex_to_rgb(bg_color)},0.12);
                    padding:2px 8px; border-radius:10px;
                    letter-spacing:0.05em;
                ">{type_label}</span>
            </div>
            <div style="font-size:1.3rem; font-weight:700; color:#F1F5F9; letter-spacing:-0.02em;">
                v{APP_VERSION}
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:0.68rem; color:rgba(255,255,255,0.35);">
                <span>{ver_date}</span>
                <span>{item_count} changes</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_recent_updates(limit: int = 5):
    """[v18.0] 업데이트 노트 — 타임라인 스타일"""
    import streamlit as st

    st.markdown("""
        <style>
        .update-timeline {
            position: relative;
            padding-left: 28px;
        }
        .update-timeline::before {
            content: '';
            position: absolute;
            left: 8px;
            top: 4px;
            bottom: 0;
            width: 2px;
            background: linear-gradient(180deg, #3B82F6, rgba(59,130,246,0.05));
            border-radius: 1px;
        }
        .update-entry {
            position: relative;
            margin-bottom: 28px;
            padding-left: 18px;
        }
        .update-entry::before {
            content: '';
            position: absolute;
            left: -24px;
            top: 6px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 2px solid #3B82F6;
            background: #0E1117;
        }
        .update-entry.latest::before {
            background: #3B82F6;
            box-shadow: 0 0 10px rgba(59,130,246,0.5);
        }
        .update-ver {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 1.05rem;
            font-weight: 700;
            color: #F1F5F9;
        }
        .update-title {
            font-size: 0.88rem;
            color: #94A3B8;
            margin-top: 2px;
        }
        .update-date {
            font-size: 0.7rem;
            color: #475569;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 2px;
        }
        .update-tag {
            display: inline-block;
            font-size: 0.58rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 10px;
            margin-left: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            vertical-align: middle;
        }
        .tag-major { background: rgba(59,130,246,0.15); color: #3B82F6; }
        .tag-minor { background: rgba(6,182,212,0.15); color: #06B6D4; }
        .tag-patch { background: rgba(16,185,129,0.15); color: #10B981; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧩 System Intelligence Updates")

    # 타임라인 렌더링
    html_parts = ['<div class="update-timeline">']

    for i, log in enumerate(CHANGELOG[:limit]):
        ver = log['version']
        date = log.get('date', '')
        title = log.get('title', '')
        ver_type = log.get('type', 'patch')
        tag_class = f"tag-{ver_type}"
        latest_class = "latest" if i == 0 else ""
        n_items = len(log.get('items', []))

        html_parts.append(f"""
        <div class="update-entry {latest_class}">
            <div>
                <span class="update-ver">v{ver}</span>
                <span class="update-tag {tag_class}">{ver_type}</span>
            </div>
            <div class="update-date">{date} · {n_items} changes</div>
            <div class="update-title">{title}</div>
        </div>
        """)

    html_parts.append('</div>')
    st.markdown(''.join(html_parts), unsafe_allow_html=True)

    # 상세 내용은 Expander
    for i, log in enumerate(CHANGELOG[:limit]):
        with st.expander(f"v{log['version']} — {log['title']}", expanded=(i == 0)):
            for item in log['items']:
                st.markdown(item)
