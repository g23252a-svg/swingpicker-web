# -*- coding: utf-8 -*-
"""
LDY Pro Trader Dashboard v7.5 (Trend Hunter & VBO)
- v7.5: 7-Factor Radar (Trend 추가), VBO 라인 차트 표시, ADX 시각화, VWAP 괴리율
- v7.0: 팩터 기반 레이더 차트, 스퀴즈 지속일(CNT) 표시, 켈트너 채널 차트
"""

# ---------------------------
# import
# ---------------------------
import os, io, math, json, requests, logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import re
from typing import Optional, Dict, Any, Tuple

# ---------------------------
# 버전 및 설정
# ---------------------------
APP_VERSION = "7.5"

# load_inquiry_items, save_inquiry_items, _now_utc_str (auth_user.py 연동)
try:
    from auth_user import (
        render_auth_box, get_user, list_users, update_user_role,
        load_inquiry_items, save_inquiry_items, _now_utc_str
    )
except ImportError:
    # 로컬 테스트용 더미 함수 (auth_user.py가 없을 경우)
    def render_auth_box(show_debug=False): return {"role": "admin", "login_id": "admin", "nickname": "Admin"}
    def get_user(): return {"role": "admin"}
    def list_users(): return []
    def update_user_role(e, r): return True
    def load_inquiry_items(): return []
    def save_inquiry_items(x): return True
    def _now_utc_str(): return datetime.now().isoformat()

from plotly.subplots import make_subplots
from version_info import (
    PRIME_TG_JOIN_URL,
    CHANGELOG,
    get_version_label,
    get_latest_log,
)

# ---------------------------
# 로깅 설정
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ldy")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("LDY_DATA_DIR", os.path.join(BASE_DIR, "data"))
RECOMMEND_LATEST_PATH = os.path.join(DATA_DIR, "recommend_latest.csv")
# 기타 경로 설정 생략 (필요시 추가)
os.makedirs(DATA_DIR, exist_ok=True)
REMOTE_RECOMMEND_URL = os.getenv("LDY_REMOTE_RECOMMEND_URL", "")

# ---------------------------
# 유틸 함수 (정규화, 시간 등)
# ---------------------------
def normalize_code(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"[^0-9]", "", s)
    return s.zfill(6) if s else ""

def postprocess_codes(df: pd.DataFrame) -> pd.DataFrame:
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].apply(normalize_code)
    return df

KST = timezone(timedelta(hours=9))

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def now_kst() -> datetime:
    return datetime.now(KST)

def to_kst_str(value, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if value is None or value == "" or value == "NaT":
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts): return ""
    try:
        if ts.year < 2000: return ""
    except: pass
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc).tz_convert(KST)
    else:
        ts = ts.tz_convert(KST)
    return ts.strftime(fmt)

def _mtime(path: str) -> int:
    try: return int(os.path.getmtime(path))
    except: return 0

def _normalize_github_raw(url: str) -> str:
    if not isinstance(url, str): return ""
    u = url.strip()
    if "github.com/" in u and "/blob/" in u:
        u = u.replace("https://github.com/", "https://raw.githubusercontent.com/")
        u = u.replace("/blob/", "/")
    return u

def _download_bytes(url: str, timeout: int = 30) -> bytes:
    u = _normalize_github_raw(url)
    r = requests.get(u, timeout=timeout, headers={"Cache-Control": "no-cache", "Pragma": "no-cache"})
    r.raise_for_status()
    return r.content

def _read_csv_bytes(b: bytes, enc: str = "utf-8-sig") -> pd.DataFrame:
    try: return pd.read_csv(io.BytesIO(b), encoding=enc)
    except UnicodeDecodeError: return pd.read_csv(io.BytesIO(b), encoding="utf-8")

def _read_csv_file(path: str, enc: str = "utf-8-sig") -> pd.DataFrame:
    try: return pd.read_csv(path, encoding=enc)
    except UnicodeDecodeError: return pd.read_csv(path, encoding="utf-8")

@st.cache_data(ttl=600)
def load_csv_url(url: str) -> pd.DataFrame:
    url = _normalize_github_raw(url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig")

@st.cache_data(ttl=600)
def load_csv_path(path: str, enc: str = "utf-8-sig") -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=enc)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")

@st.cache_data(ttl=600, show_spinner=False)
def _load_csv_cached(path: str, enc: str, remote_url: str, mtime_sig: int) -> pd.DataFrame:
    # Local First logic simplified for brevity
    if path and os.path.exists(path):
        return _read_csv_file(path, enc=enc)
    elif remote_url:
        return load_csv_url(remote_url)
    return pd.DataFrame()

def load_recommend_latest(local_path: str = None, remote_url: str = "") -> pd.DataFrame:
    p = local_path or RECOMMEND_LATEST_PATH
    sig = _mtime(p)
    return _load_csv_cached(path=p, enc="utf-8-sig", remote_url=remote_url, mtime_sig=sig)

def infer_data_timestamp(df_raw: pd.DataFrame):
    if df_raw is None or df_raw.empty: return None
    candidates = []
    now_utc_val = now_utc()
    cols = ["기준일자", "기준일", "날짜", "DATE", "Date", "date", "update_time", "updated_at"]
    for col in cols:
        if col in df_raw.columns:
            s = pd.to_datetime(df_raw[col], errors="coerce", utc=True)
            s = s[(s.notna()) & (s >= pd.Timestamp("2000-01-01", tz="UTC"))]
            if not s.empty: candidates.append(s.max())
    if candidates: return max(candidates)
    return None

# ---------------------------
# FDR & Chart Data
# ---------------------------
try:
    import FinanceDataReader as fdr
    FDR_OK = True
except Exception:
    fdr = None
    FDR_OK = False

try:
    from pykrx import stock
    PYKRX_OK = True
except Exception:
    stock = None
    PYKRX_OK = False

@st.cache_data(ttl=600)
def get_stock_chart_data(code):
    if not FDR_OK: return None
    try:
        code_str = str(code).zfill(6)
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        df = fdr.DataReader(code_str, start_date)
        if df is None or df.empty: return None
        
        # MA & BB
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BB_UPPER'] = df['MA20'] + 2.0 * std20
        df['BB_LOWER'] = df['MA20'] - 2.0 * std20

        # KC
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr20 = tr.rolling(window=20).mean()
        df['KC_UPPER'] = df['MA20'] + (1.5 * atr20)
        df['KC_LOWER'] = df['MA20'] - (1.5 * atr20)

        # RSI
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        df['RSI14_CHART'] = 100 - (100 / (1 + rs))

        # SuperTrend (Simple calc for visual)
        # (생략: 필요시 calculate_supertrend 함수 사용)
        
        return df.tail(120)
    except Exception:
        return None

# ---------------------------
# 시각화 함수 (v7.5 Updated)
# ---------------------------

def plot_radar_chart(row):
    """v7.5: 7-Factor Radar (Trend 추가)"""
    stats = {}
    # v7.5 팩터 확인
    if "NORM_MOM" in row.index:
        stats = {
            "모멘텀(MOM)": row.get("NORM_MOM", 0) * 100,
            "추세(TRD)": row.get("NORM_TRD", 0) * 100,   # [New]
            "가성비(RR)": row.get("NORM_RR", 0) * 100,
            "수익여력(T1)": row.get("NORM_T1", 0) * 100,
            "안전성(SL)": row.get("NORM_SL", 0) * 100,
            "타점(NEAR)": row.get("NORM_NEAR", 0) * 100,
            "수급(LIQ)": row.get("NORM_LIQ", 0) * 100,
        }
    else:
        # Fallback
        stats = {
            "모멘텀": min(100, (row.get("ret_5d_%", 0) + 5) * 10),
            "수급": row.get("MFI14", 50),
            "가성비": min(100, row.get("RR1", 1) * 50),
            "안전성": 100 - (row.get("이격도", 0) * 2),
            "종합점수": row.get("LDY_SCORE", 0),
        }

    values = [max(0, min(100, v)) for v in stats.values()]
    keys = list(stats.keys())
    values += values[:1]
    keys += keys[:1]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=keys, fill='toself',
        name=row.get('종목명', '종목'),
        line=dict(color='#00E5FF', width=3),
        fillcolor='rgba(0, 229, 255, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color='gray'), gridcolor='rgba(128,128,128,0.3)'),
            angularaxis=dict(tickfont=dict(size=12, weight='bold'), gridcolor='rgba(128,128,128,0.3)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False, height=280,
        margin=dict(l=40, r=40, t=30, b=30),
        title=dict(text="📊 7-Factor Analysis (v7.5)", x=0.5, y=0.95, font=dict(size=14))
    )
    return fig

def plot_interactive_chart(df, code, name, entry=None, stop=None, target1=None, target2=None, vbo=None, show_bb=True, show_kc=False, show_rsi=False):
    if df is None or df.empty: return go.Figure()

    rows = 3 if show_rsi else 2
    row_heights = [0.6, 0.2, 0.2] if show_rsi else [0.7, 0.3]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)

    # Colors
    C_UP, C_DOWN = '#FF3B30', '#007AFF'
    
    # Candle
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="주가", increasing={'line': {'color': C_UP}}, decreasing={'line': {'color': C_DOWN}}
    ), row=1, col=1)

    # MA & BB & KC
    if "MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="20일선", line=dict(color='#FFD700', width=1.5)), row=1, col=1)
    
    if show_bb and "BB_UPPER" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], line=dict(width=1, color='rgba(189,195,199,0.5)'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], line=dict(width=1, color='rgba(189,195,199,0.5)'), fill='tonexty', fillcolor='rgba(189,195,199,0.1)', name="BB"), row=1, col=1)

    if show_kc and "KC_UPPER" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["KC_UPPER"], line=dict(width=1.5, dash='dot', color='#E040FB'), name="KC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["KC_LOWER"], line=dict(width=1.5, dash='dot', color='#E040FB'), showlegend=False), row=1, col=1)

    # Volume
    colors = [C_UP if c >= o else C_DOWN for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color=colors, opacity=0.8), row=2, col=1)

    # RSI
    if show_rsi and "RSI14_CHART" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14_CHART"], name="RSI", line=dict(color='#AB47BC')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="blue", row=3, col=1)

    # Lines (Entry, Stop, Target)
    def _line(val, color, dash, txt):
        if val and val > 0:
            fig.add_hline(y=val, line_dash=dash, line_color=color, annotation_text=txt, annotation_font_color=color, row=1, col=1)

    _line(entry, '#FF9F0A', "dash", f"🚀진입: {int(entry):,}")
    _line(stop, '#00B0FF', "dot", f"🛡️손절: {int(stop):,}")
    _line(target1, '#30D158', "dot", f"💰목표: {int(target1):,}")

    # [v7.5 New] VBO Line
    if vbo and vbo > 0:
        fig.add_hline(y=vbo, line_dash="dashdot", line_color="#FFFF00", line_width=2, 
                      annotation_text=f"⚡VBO: {int(vbo):,}", annotation_position="top left", row=1, col=1)

    fig.update_layout(
        title=dict(text=f"{name} ({str(code).zfill(6)})", x=0, font=dict(size=16)),
        height=700 if show_rsi else 550, margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        dragmode="pan"
    )
    # Crosshair
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    return fig

# ---------------------------
# 유틸 (Make Preview)
# ---------------------------
def make_preview(df, n=5):
    if df is None or df.empty: return df
    if "LDY_RANK" in df.columns:
        return df.sort_values("LDY_RANK", ascending=True).head(n).copy()
    if "LDY_SCORE" in df.columns:
        return df.sort_values("LDY_SCORE", ascending=False).head(n).copy()
    return df.head(n).copy()

# ---------------------------
# Page Setup & Main
# ---------------------------
st.set_page_config(page_title=f"LDY Pro Trader v{APP_VERSION}", layout="wide", page_icon="💎")

# CSS
st.markdown("""
    <style>
    .block-container { padding-top: 1rem !important; padding-bottom: 3rem !important; }
    div[data-testid="stMetric"] { background-color: #1E1E1E; border: 1px solid #333; padding: 10px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# Data Load
RAW_SRC = os.getenv("LDY_RAW_URL", "")
LOCAL_RAW = "data/recommend_latest.csv"
DATA_TS, DATA_SRC = None, "unknown"

with st.spinner("🚀 데이터 로드 중..."):
    try:
        if RAW_SRC:
            df_latest = load_csv_url(RAW_SRC)
            DATA_SRC = "remote"
        else:
            df_latest = load_csv_path(LOCAL_RAW)
            DATA_SRC = "local"
        
        df_latest = postprocess_codes(df_latest)
        if "종목코드" in df_latest.columns:
            df_latest["종목코드"] = df_latest["종목코드"].astype(str).str.zfill(6)
        
        DATA_TS = infer_data_timestamp(df_latest)
        scored = df_latest.copy() # scored 변수 확보
        base = scored.head(100)   # base 변수 확보 (필요시 조정)
        top20 = scored.head(20)   # top20 변수 확보

    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

# Sidebar
with st.sidebar:
    user = render_auth_box(show_debug=False)
    auth_status = user.get("role", "guest") if user else "guest"
    
    # Force Beta Prime
    BETA_USERS = {"coolguyhaeng@naver.com", "kiljung87@nate.com", "coiil@naver.com", "quartzk123@gmail.com", "user5@example.com"}
    if user and user.get("login_id") in BETA_USERS: auth_status = "prime"

    st.caption(f"Status: **{auth_status.upper()}**")
    if st.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    st.markdown("### 💎 Premium")
    st.caption("Pro: 1.9만 / Prime: 3.9만 (월)")
    st.caption(f"Data Source: {DATA_SRC}")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 시장 (Market)", "🔭 종목 분석", "💼 내 자산", 
    "📮 문의 게시판", "⚖️ 약관", "🧩 업데이트"
])

# --- Tab 1: Market (기존 유지) ---
with tab1:
    st.success(f"📅 데이터 기준일: {to_kst_str(DATA_TS) if DATA_TS else '알수없음'}")
    # (여기 시장 지표 등 기존 코드 유지 가능, 생략)
    st.info("KOSPI / KOSDAQ / 섹터맵 기능 영역 (기존 코드 유지)")

# --- Tab 2: Analysis (v7.5 COMPLETE REPLACEMENT) ---
with tab2:
    # 1. 환영 메시지 (첫 방문 시)
    just_registered = st.session_state.get("just_registered", False)
    if just_registered:
        st.success("🎉 첫 가입 환영! Top 5 프리뷰")
        try:
            preview = make_preview(base, n=5)
            cols = ["종목명", "LDY_SCORE", "추천매수가"]
            cols = [c for c in cols if c in preview.columns]
            st.dataframe(preview[cols], use_container_width=True)
        except: pass
        st.session_state["just_registered"] = False
        st.divider()

    st.subheader("🎯 종목 발굴 & 상세 분석 (v7.5)")

    # 2. 필터 UI
    c_f1, c_f2, c_f3 = st.columns([1, 1, 1])
    with c_f1:
        min_score = st.slider("최소 LDY 점수", 0, 100, 70, step=5, key="v75_min_score")
    with c_f2:
        use_adx = st.checkbox("💪 강한 추세장 (ADX ≥ 25)", value=False, key="v75_use_adx")
        use_sqz = st.checkbox("🌪️ TTM Squeeze (폭발대기)", value=False, key="v75_use_sqz")
    with c_f3:
        search_txt = st.text_input("종목명/코드 검색", placeholder="예: 삼성전자", key="v75_search")

    # 3. 필터링 로직
    view_df = scored.copy()
    view_df = view_df[pd.to_numeric(view_df["LDY_SCORE"], errors='coerce') >= min_score]

    if use_adx and "ADX" in view_df.columns:
        view_df = view_df[pd.to_numeric(view_df["ADX"], errors='coerce') >= 25]
    if use_sqz and "TTM_SQUEEZE" in view_df.columns:
        view_df = view_df[view_df["TTM_SQUEEZE"] == 1]
    if search_txt:
        view_df = view_df[view_df["종목명"].str.contains(search_txt) | view_df["종목코드"].str.contains(search_txt)]

    # 권한별 제한
    if auth_status in ["pro", "prime", "admin"]:
        if auth_status == "pro": view_df = view_df.head(20)
        else: view_df = view_df.head(100)
    else:
        view_df = view_df.head(5 if user else 3)
        st.info("🔒 Guest/Free: 상위 일부 종목만 노출됩니다.")

    # 4. 리스트 출력
    st.markdown("##### 📋 Daily Opportunity List")
    if view_df.empty:
        st.warning("🔍 조건에 맞는 종목이 없습니다.")
    else:
        cols_candidate = ["종목명", "LDY_SCORE", "ROUTE", "종가", "추천매수가", "손절가", "ADX", "VBO_Price", "거래대금(억원)"]
        display_cols = [c for c in cols_candidate if c in view_df.columns]

        cfg = {
            "LDY_SCORE": st.column_config.ProgressColumn("점수", format="%.0f", min_value=0, max_value=100, width="small"),
            "ADX": st.column_config.ProgressColumn("추세(ADX)", help="25↑ 강세", format="%.0f", min_value=0, max_value=60, width="small"),
            "거래대금(억원)": st.column_config.BarChartColumn("대금(억)", width="small"),
            "종가": st.column_config.NumberColumn("현재가", format="%d원"),
            "추천매수가": st.column_config.NumberColumn("매수", format="%d원"),
            "손절가": st.column_config.NumberColumn("손절", format="%d원"),
            "VBO_Price": st.column_config.NumberColumn("⚡VBO", help="돌파기준", format="%d원"),
        }
        st.dataframe(view_df[display_cols], use_container_width=True, column_config=cfg, height=400, hide_index=True)

    st.divider()

    # 5. 상세 분석
    st.subheader("🔭 종목 상세 정밀 분석")
    if not view_df.empty:
        opts = view_df.apply(lambda r: f"{r['종목명']} ({r['종목코드']})", axis=1).tolist()
        sel = st.selectbox("분석할 종목 선택", opts, key="v75_sel_stock")

        if sel:
            sel_idx = opts.index(sel)
            row = view_df.iloc[sel_idx]
            code = str(row.get("종목코드", "")).zfill(6)
            name = row.get("종목명", "")

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("현재가", f"{int(row.get('종가',0)):,}원")
            score = row.get('LDY_SCORE', 0)
            m2.metric("LDY Score", f"{score}점", delta="Good" if score >= 80 else None)
            adx_val = row.get('ADX', 0)
            m3.metric("ADX (추세)", f"{adx_val}", delta="Strong" if adx_val >= 25 else "Weak")
            vbo_val = row.get('VBO_Price', 0)
            m4.metric("⚡VBO 돌파가", f"{int(vbo_val):,}원" if vbo_val > 0 else "-")

            st.divider()

            # Chart & Radar
            col_chart, col_radar = st.columns([2, 1])
            with col_chart:
                with st.expander("⚙️ 차트 옵션", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    show_bb = c1.checkbox("Bollinger", True, key=f"bb_{code}")
                    show_kc = c2.checkbox("Keltner", False, key=f"kc_{code}")
                    show_rsi = c3.checkbox("RSI", False, key=f"rsi_{code}")
                
                chart_df = get_stock_chart_data(code)
                if chart_df is not None:
                    fig = plot_interactive_chart(
                        chart_df, code, name,
                        entry=pd.to_numeric(row.get('추천매수가'), errors='coerce'),
                        stop=pd.to_numeric(row.get('손절가'), errors='coerce'),
                        target1=pd.to_numeric(row.get('추천매도가1'), errors='coerce'),
                        target2=pd.to_numeric(row.get('추천매도가2'), errors='coerce'),
                        vbo=pd.to_numeric(row.get('VBO_Price'), errors='coerce'),
                        show_bb=show_bb, show_kc=show_kc, show_rsi=show_rsi
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("차트 데이터 로드 실패")

            with col_radar:
                try:
                    st.plotly_chart(plot_radar_chart(row), use_container_width=True)
                except: st.caption("레이더 차트 오류")
                
                st.info(f"💡 **AI:** {row.get('AI_COMMENT', '-')}")
                with st.container(border=True):
                    c1, c2 = st.columns(2)
                    c1.write(f"**RSI:** {row.get('RSI14','-')}")
                    c1.write(f"**MFI:** {row.get('MFI14','-')}")
                    if "VWAP_Gap" in row: c2.write(f"**VWAP:** {row['VWAP_Gap']}%")
                    c2.write(f"**강도:** {row.get('거래강도','-')}")

    # CSV Download
    if auth_status in ["prime", "admin"]:
        st.divider()
        st.download_button("📥 전체 다운로드", scored.to_csv(index=False).encode('utf-8-sig'), "ldy_rank_v75.csv", "text/csv")
# ---------------------------
# 내 자산 (병렬 처리)
# ---------------------------
def fetch_current_price(code, name):
    """
    현재가 조회 함수 (FDR 우선 시도 -> 실패 시 pykrx 시도)
    """
    price = 0

    # 1차 시도: FinanceDataReader (속도가 빠름)
    if FDR_OK:
        try:
            # 최근 7일 데이터 조회 (휴장일 고려)
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            df = fdr.DataReader(str(code).zfill(6), start_date)

            if df is not None and not df.empty:
                price = int(df.iloc[-1]['Close'])
        except Exception:
            pass # FDR 실패 시 그냥 넘어감

    # 2차 시도: pykrx (FDR 실패 시 백업, collector.py와 동일 로직)
    # price가 0이고 pykrx 라이브러리가 로드되어 있다면 시도
    if price == 0 and PYKRX_OK:
        try:
            # pykrx는 YYYYMMDD 형식을 씀
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=7)

            # 오늘 날짜까지 조회
            df_k = stock.get_market_ohlcv_by_date(
                start_dt.strftime("%Y%m%d"), 
                end_dt.strftime("%Y%m%d"), 
                str(code).zfill(6)
            )

            if df_k is not None and not df_k.empty:
                # '종가' 컬럼이 있는지 확인 (pykrx 버전에 따라 컬럼명이 다를 수 있음)
                if '종가' in df_k.columns:
                    price = int(df_k.iloc[-1]['종가'])
                elif 'Close' in df_k.columns:
                    price = int(df_k.iloc[-1]['Close'])
        except Exception:
            pass

    return code, name, price
with tab3:
    # 1) 권한 체크
    if auth_status in ["guest", "free"]:
        st.info("🔒 내 자산 분석은 Pro 등급부터 가능합니다.")
    elif pf_input:
        try:
            # 2) 종목 코드 매핑 준비
            code_map = get_code_map() if 'get_code_map' in globals() else {}
            if not code_map and FDR_OK:
                try:
                    df_krx = fdr.StockListing('KRX')
                    code_map = dict(
                        zip(df_krx['Name'], df_krx['Code'].astype(str).str.zfill(6))
                    )
                except Exception:
                    pass

            # 3) 포트폴리오 파싱
            targets = []     # 종목 리스트
            cash_amt = 0.0   # 현금(예수금) 총액

            lines = pf_input.strip().split('\n')
            for line in lines:
                if ":" not in line:
                    continue

                parts = [p.strip() for p in line.split(':')]
                if len(parts) != 3:
                    continue

                name_input, avg_str, qty_str = parts

                # ✅ CASH / 현금 라인 분리 처리
                if name_input.strip().upper().startswith("CASH") or "현금" in name_input:
                    try:
                        cash_amt += float(avg_str.replace(',', '')) * int(qty_str.replace(',', ''))
                    except Exception:
                        pass
                    continue

                # ✅ 일반 종목 라인
                code = str(name_input).strip()
                if not code.isdigit():
                    code = code_map.get(code, code)  # 종목명 → 코드 매핑
                code = str(code).zfill(6)

                try:
                    avg = float(avg_str.replace(',', ''))
                    qty = int(qty_str.replace(',', ''))
                except Exception:
                    continue

                targets.append((code, name_input, avg, qty))

            if not targets and cash_amt <= 0:
                st.warning("포트폴리오 입력 형식이 올바른지 확인해 주세요. 예시: `NAVER:261000:10` 또는 `CASH:1000000:1`")
                st.stop()

            # 4) 현재가 조회 (병렬 처리)
            price_map = {}
            with st.spinner('⚡ 실시간 시세를 조회 중입니다...'):
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(fetch_current_price, t[0], t[1])
                        for t in targets
                    ]
                    for future in futures:
                        c, n, p = future.result()
                        price_map[c] = p

            # 5) 기본 지표 계산 (종목별/전체)
            
            total_buy = 0.0
            total_eval = 0.0

            rows_pf = []  # 섹터/파이차트용

            for idx, (code, name_input, avg, qty) in enumerate(targets):
                cur_price = price_map.get(code, 0)
                real_name = (
                    stock.get_market_ticker_name(code)
                    if (PYKRX_OK and cur_price > 0)
                    else name_input
                )

                buy_amt = avg * qty
                eval_amt = cur_price * qty
                total_buy += buy_amt
                total_eval += eval_amt

                rows_pf.append(
                    {
                        "code": code,
                        "name": real_name,
                        "avg": avg,
                        "qty": qty,
                        "eval": eval_amt,
                    }
                )

                # 수익률/평가손익 계산
                if cur_price > 0:
                    profit_rate = (cur_price - avg) / avg * 100
                    pnl = eval_amt - buy_amt
                    if profit_rate > 0:
                        signal = "🟢 수익"
                    elif profit_rate > -3:
                        signal = "🟡 보합"
                    else:
                        signal = "🔴 손실"
                else:
                    signal = "❓ 확인불가"
                    profit_rate = 0
                    pnl = 0

                # 🚨 [수정됨] 모바일 최적화: 카드형 UI (컨테이너 사용)
                with st.container(border=True):
                    c_main, c_pnl = st.columns([1.5, 1])
                    with c_main:
                        st.markdown(f"**{real_name}**")
                        st.caption(f"평단: {int(avg):,} / 수량: {qty}")
                        if cur_price > 0:
                            st.markdown(f"현재: **{cur_price:,}원**")
                        else:
                            st.markdown("시세 확인 불가")
                    
                    with c_pnl:
                        # 수익률 색상 강조
                        color = "green" if profit_rate > 0 else "red"
                        if profit_rate == 0: color = "gray"
                        
                        # 우측에 수익률 크게 표시
                        st.markdown(f":{color}[**{profit_rate:+.2f}%**]")
                        st.markdown(f":{color}[{int(pnl):,}원]")

            st.divider()

            # 6) 전체 포트폴리오 요약
            c1, c2, c3 = st.columns(3)
            tot_rate = (
                (total_eval - total_buy) / total_buy * 100
                if total_buy > 0
                else 0
            )

            c1.metric("총 매수", f"{int(total_buy):,}원")
            c2.metric("총 평가", f"{int(total_eval):,}원")
            c3.metric(
                "총 수익률",
                f"{tot_rate:+.2f}%",
                f"{int(total_eval - total_buy):,}원",
                delta_color="normal" if tot_rate >= 0 else "inverse",
            )

            # 7) 현금 비중 계산
            total_asset = total_eval + cash_amt
            if cash_amt > 0 and total_asset > 0:
                cash_ratio = cash_amt / total_asset * 100
                st.info(
                    f"💰 현재 현금(예수금) 비중은 **{cash_ratio:.1f}%** 입니다.\n"
                    "시장 변동성에 따라 보통 **10~30%** 사이에서 조절하는 전략이 많이 사용됩니다."
                )

            # 8) 종목별 평가금액 비중 파이차트
            try:
                if total_eval > 0:
                    df_pf = pd.DataFrame(rows_pf)
                    df_pf = df_pf[df_pf["eval"] > 0]

                    if not df_pf.empty:
                        fig_pie = go.Figure(
                            data=[
                                go.Pie(
                                    labels=df_pf["name"],
                                    values=df_pf["eval"],
                                    hole=0.4,
                                )
                            ]
                        )
                        fig_pie.update_layout(
                            title="📊 종목별 평가 금액 비중",
                            height=300,
                            margin=dict(l=10, r=10, t=40, b=10),
                            showlegend=True,
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
            except Exception:
                logger.exception("portfolio pie chart failed")

            # 9) 포트폴리오 Health Check (섹터 편중 + 현금 비중)
            try:
                st.subheader("🏥 포트폴리오 건강검진", anchor=False)

                df_pf = pd.DataFrame(rows_pf)
                df_pf = df_pf[df_pf["eval"] > 0]

                # 섹터 컬럼 선택
                sector_col = None
                if "업종_대분류" in scored.columns:
                    sector_col = "업종_대분류"
                elif "업종" in scored.columns:
                    sector_col = "업종"

                if sector_col:
                    # 종목코드 → 섹터 매핑
                    sector_map = (
                        scored
                        .dropna(subset=[sector_col, "종목코드"])
                        .drop_duplicates("종목코드")
                        .set_index("종목코드")[sector_col]
                        .to_dict()
                    )

                    df_pf["섹터"] = df_pf["code"].map(sector_map).fillna("기타")

                    sector_grp = (
                        df_pf.groupby("섹터")["eval"]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    total_eval_safe = sector_grp.sum()

                    if total_eval_safe > 0:
                        # 섹터 비중 (%)
                        sector_ratio = (sector_grp / total_eval_safe * 100).round(1)

                        fig_sector = go.Figure(
                            data=[
                                go.Bar(
                                    x=sector_ratio.values,
                                    y=sector_ratio.index,
                                    orientation="h",
                                    text=[f"{v:.1f}%" for v in sector_ratio.values],
                                    textposition="auto",
                                )
                            ]
                        )
                        fig_sector.update_layout(
                            title="섹터별 비중 (평가금액 기준)",
                            height=320,
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        st.plotly_chart(fig_sector, use_container_width=True)

                        # 편중 진단 코멘트
                        top_sector = sector_ratio.index[0]
                        top_ratio = float(sector_ratio.iloc[0])

                        comment = f"현재 가장 큰 비중은 **{top_sector} ({top_ratio:.1f}%)** 입니다.  \n"

                        if top_ratio >= 60:
                            comment += "➡ 단일 섹터 비중이 60%를 넘어 **위험도가 상당히 높은 편**입니다. 분산을 강하게 추천합니다."
                        elif top_ratio >= 40:
                            comment += "➡ 특정 섹터 비중이 40% 이상으로 **다소 편중된 상태**입니다. 다른 섹터 편입을 고민해 볼 만합니다."
                        else:
                            comment += "➡ 섹터 비중이 비교적 고르게 분산되어 있습니다."

                        # 현금 비중까지 함께 코멘트
                        total_asset = total_eval + cash_amt
                        if cash_amt > 0 and total_asset > 0:
                            cash_ratio = cash_amt / total_asset * 100
                            comment += f"\n\n또한 현금(예수금) 비중은 약 **{cash_ratio:.1f}%** 입니다."

                            if cash_ratio < 5:
                                comment += " 변동성이 큰 장세에서는 다소 낮은 편입니다. 방어력을 조금 보강하는 것도 고려해 보세요."
                            elif cash_ratio > 40:
                                comment += " 상당히 보수적인 비중으로, 기회 포착 속도는 느려질 수 있지만 하락 방어에는 유리한 편입니다."

                        st.info(comment)
                    else:
                        st.caption("※ 섹터별 평가금액이 거의 없어 건강검진을 생략했습니다.")
                else:
                    st.caption("※ 스코어 데이터에 섹터 정보(업종, 업종_대분류)가 없어 건강검진을 생략했습니다.")
            except Exception:
                logger.exception("portfolio health check failed")
                st.caption("※ 포트폴리오 건강검진 중 오류가 발생했습니다.")
        except Exception as e:
            st.error(f"포트폴리오 분석 중 오류 발생: {e}")

    else:
        st.info("👈 사이드바에 포트폴리오를 입력하고 '저장/분석' 버튼을 누르세요.")

with tab4:
    st.subheader("📮 문의 게시판")

    current_user = user if 'user' in globals() else None

    default_email = ""
    default_nick = ""
    if current_user:
        default_email = current_user.get("login_id", "")
        default_nick = current_user.get("nickname", "")

    st.markdown("#### ✏️ 문의 작성")

    with st.form("inquiry_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            nickname = st.text_input("닉네임", value=default_nick, placeholder="닉네임 또는 이름")
        with col_b:
            email = st.text_input("이메일 (선택)", value=default_email, placeholder="답변 받을 이메일 (선택)")

        title = st.text_input("제목", placeholder="문의 제목을 입력해 주세요.")
        content = st.text_area("내용", placeholder="사이트 사용 관련 문의를 자유롭게 남겨 주세요.", height=150)

        submitted = st.form_submit_button("💌 문의 등록")

    if submitted:
        if not title.strip() or not content.strip():
            st.error("제목과 내용을 모두 입력해 주세요.")
        else:
            # ✅ [수정됨] Gist에서 기존 목록을 불러옵니다.
            current_items = load_inquiry_items()
            
            # 새 문의 데이터 생성
            new_item = {
                "title": title.strip(),
                "content": content.strip(),
                "nickname": nickname.strip() or "익명",
                "email": email.strip(),
                "created_at": _now_utc_str(), # auth_user의 시간 함수 사용
            }
            
            # 리스트에 추가하고 Gist에 저장
            current_items.append(new_item)
            ok = save_inquiry_items(current_items)

            if ok:
                st.success("문의가 등록되었습니다. Gist에 저장 완료! 🙌")
                # 화면 갱신을 위해 rerun (Streamlit 버전에 따라 다름)
                try:
                    st.rerun()
                except:
                    pass
            else:
                st.error("저장 실패! (Gist 연동 오류 - 로그 확인 필요)")

    st.markdown("#### 📂 최근 문의 내역")

    # ✅ [수정됨] Gist에서 데이터를 불러와서 보여줍니다.
    inquiries = load_inquiry_items()

    if not inquiries:
        st.info("아직 등록된 문의가 없습니다.")
    else:
        # 최신순 정렬 (리스트 뒤집기)
        for item in reversed(inquiries[-50:]):
            box = st.container(border=True)
            with box:
                st.markdown(f"**제목:** {item.get('title', '-')}")
                
                # 날짜 포맷팅 (UTC -> KST 변환은 to_kst_str 함수가 있다면 사용, 없으면 그대로)
                date_str = item.get('created_at','-')
                if 'to_kst_str' in globals():
                    date_str = to_kst_str(date_str)

                meta = f"작성자: {item.get('nickname','익명')} · 작성일: {date_str}"
                if item.get("email"):
                    meta += f" · 이메일: {item.get('email')}"
                st.caption(meta)
                st.markdown(item.get("content", "").replace("\n", "  \n"))

with tab5:
    st.subheader("⚖️ 이용 약관 / 투자 유의사항")

    st.markdown("### 1. 서비스 성격")
    st.markdown(
        "- LDY Pro Trader는 **퀀트 지표 기반의 데이터 분석 도구**로, "
        "개별 종목의 매수·매도, 수익을 보장하는 리딩 서비스가 아닙니다.\n"
        "- 제공되는 모든 정보는 **교육 및 참고용**이며, "
        "투자 판단을 보조하는 **연구·리서치 자료**의 성격을 가집니다."
    )

    st.markdown("### 2. 투자 책임에 대한 안내")
    st.markdown(
        "- 실제 매수·매도 등 **최종 투자 의사결정**은 전적으로 이용자 본인의 판단입니다.\n"
        "- 투자 결과로 발생하는 **손익(수익, 손실, 기회비용 포함)**은 "
        "모두 이용자 본인에게 귀속되며, 본 서비스 및 개발자는 이에 대해 법적 책임을 지지 않습니다.\n"
        "- 본 서비스는 **미래 수익률, 특정 수익구간 달성, 손실 방지** 등을 어떠한 형태로도 보증하지 않습니다."
    )

    st.markdown("### 3. 데이터 및 지표 한계")
    st.markdown(
        "- 사용되는 시장 데이터는 외부 데이터 제공처 및 증권사 API, 공개 데이터 소스를 바탕으로 하며, "
        "지연·오류·누락이 발생할 수 있습니다.\n"
        "- 지표 및 스코어는 과거 데이터를 기반으로 계산되며, "
        "**향후 시장 상황과 괴리**가 발생할 수 있습니다.\n"
        "- 알고리즘 로직은 지속적으로 개선/업데이트될 수 있으며, "
        "이 과정에서 **종전 결과와 다른 스코어**가 나올 수 있습니다."
    )

    st.markdown("### 4. 이용권 및 계정 정책 (요약)")
    st.markdown(
        "- **Guest(비회원)** : 상위 3개 종목 맛보기.\n"
        "- **Free(회원)** : 상위 5개 종목 열람.\n"
        f"- **Pro 1개월 이용권 ({PRICE_PRO:,}원)** : 상위 20 종목, 내 자산 분석 기능 제공.\n"
        f"- **Prime 1개월 이용권 ({PRICE_PRIME:,}원)** : 전체 종목, CSV 다운로드, 텔레그램 알림 등 고급 기능 제공.\n"
        "- 자동 결제는 지원하지 않으며, 1개월 단위 선불 결제·연장 방식입니다.\n"
        "- 구체적인 결제/환불/이용 기간 정책은 별도 안내(카카오 채널, 약관 페이지 등)를 따릅니다."
    )

    st.markdown("### 5. 한 줄 요약")
    st.info("👉 **데이터와 퀀트는 도구일 뿐, 최종 책임은 언제나 본인에게 있다.**")

with tab6:
    st.subheader("🧩 LDY Pro Trader 업데이트 노트")

    if not CHANGELOG:
        st.info("아직 등록된 업데이트 기록이 없습니다.")
    else:
        latest = CHANGELOG[0]

        # 🔹 상단에 현재 버전 / 최근 업데이트 요약
        st.success(
            f"현재 버전: **v{APP_VERSION}**  \n"
            f"최근 업데이트: **{latest['date']} · {latest['title']}**"
        )

        st.markdown("---")

        # 🔹 버전별 상세 내역 (최신 버전은 기본 펼침)
        for idx, log in enumerate(CHANGELOG):
            header = f"v{log['version']} · {log['date']} — {log['title']}"
            is_latest = (idx == 0)

            with st.expander(
                f"⭐ {header}" if is_latest else header,
                expanded=is_latest,   # 최신 버전만 기본 펼침
            ):
                for item in log.get("items", []):
                    st.markdown(f"- {item}")
