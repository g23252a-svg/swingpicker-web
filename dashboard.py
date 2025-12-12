# -*- coding: utf-8 -*-
"""
LDY Pro Trader v6.9.0 (Deep Insight Update)
- 신규: Market Breadth Gauge — 시장 전체의 20일선 상회 비율(시장 온도) 시각화
- 신규: Squeeze Hunter — 볼린저 밴드폭(BandWidth) 축소 종목 필터링
- 신규: Chart Upgrade — BandWidth 보조지표 추가 (변동성 시각화)
- 개선: Sector Heatmap — 섹터 모멘텀(수익률) 기반 컬러링 적용
- 기반: v6.8.0 (Reality Check, Portfolio 등 기존 기능 모두 포함)
"""

import os, io, math, json, requests, logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import re

# Auth & Version Logic
# (auth_user.py, version_info.py 파일이 같은 폴더에 있다고 가정)
try:
    from auth_user import render_auth_box, update_user_role, list_users
    from version_info import PRIME_TG_JOIN_URL, APP_VERSION, CHANGELOG, get_latest_log
except ImportError:
    # 파일 없을 경우를 대비한 더미 함수/변수
    def render_auth_box(show_debug=False): return None
    def update_user_role(email, role): return False
    def list_users(): return []
    PRIME_TG_JOIN_URL = ""
    APP_VERSION = "6.9.0"
    CHANGELOG = []
    def get_latest_log(): return None

from plotly.subplots import make_subplots

# ---------------------------
# 로깅 설정
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ldy")

# ---------------------------
# 데이터 경로 및 설정
# ---------------------------
INQUIRY_DB_PATH = os.path.join("data", "inquiries_db.json")
SUBS_DB_PATH = os.path.join("data", "subscriptions_db.json")
KST = timezone(timedelta(hours=9))

def get_conf(key, default_val):
    try:
        if key in st.secrets: return st.secrets[key]
    except FileNotFoundError: pass
    return os.getenv(key, default_val)

RAW_SRC        = get_conf("LDY_RAW_URL",         "data/recommend_latest.csv")
LOCAL_RAW      = get_conf("LDY_LOCAL_RAW",       "data/recommend_latest.csv")
PORTFOLIO_FILE = get_conf("LDY_PORTFOLIO_FILE", "my_portfolio.json")

BANK_ACCOUNT = get_conf("LDY_BANK_ACCOUNT", "카카오뱅크 3333-22-2658701")
BANK_HOLDER  = get_conf("LDY_BANK_HOLDER",  "이OO")

# ---------------------------
# 라이브러리 체크
# ---------------------------
try:
    import FinanceDataReader as fdr
    FDR_OK = True
except Exception:
    FDR_OK = False

try:
    from pykrx import stock
    PYKRX_OK = True
except Exception:
    PYKRX_OK = False

# ---------------------------
# 페이지 설정
# ---------------------------
st.set_page_config(
    page_title=f"LDY Pro Trader v{APP_VERSION}",
    layout="wide",
    page_icon="💎",
)
st.title(f"🏆 LDY Pro Trader v{APP_VERSION} (Deep Insight)")
st.caption("AI Quant Analysis & Volatility Hunter — Scoring / Squeeze / Sector")

# ---------------------------
# 유틸리티 함수 (DB, Time, Format)
# ---------------------------
def now_kst() -> datetime:
    return datetime.now(KST)

def to_kst_str(value, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if value is None or str(value) == "" or str(value) == "NaT": return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts): return ""
    try:
        if ts.year < 2000: return ""
    except: pass
    if ts.tzinfo is None: ts = ts.tz_localize(KST)
    else: ts = ts.tz_convert(KST)
    return ts.strftime(fmt)

def nz_num(s):
    return pd.to_numeric(s, errors="coerce")

def load_inquiry_db():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(INQUIRY_DB_PATH): return {"inquiries": []}
    try:
        with open(INQUIRY_DB_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except: return {"inquiries": []}

def save_inquiry_db(db):
    os.makedirs("data", exist_ok=True)
    with open(INQUIRY_DB_PATH, "w", encoding="utf-8") as f: json.dump(db, f, indent=2, ensure_ascii=False)

def load_subs_db():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(SUBS_DB_PATH): return {"subs": {}}
    try:
        with open(SUBS_DB_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except: return {"subs": {}}

def save_subs_db(db):
    os.makedirs("data", exist_ok=True)
    with open(SUBS_DB_PATH, "w", encoding="utf-8") as f: json.dump(db, f, indent=2, ensure_ascii=False)

def set_subscription(email, role):
    if not email: return
    db = load_subs_db()
    subs = db.get("subs", {})
    today = now_kst().date()
    if role in ["pro", "prime"]:
        expire = today + timedelta(days=30)
        subs[email] = {"role": role, "paid_at": str(today), "expire_at": str(expire)}
    else:
        if email in subs:
            subs[email]["role"] = role
            subs[email]["expire_at"] = str(today)
    db["subs"] = subs
    save_subs_db(db)

def get_subscription(email):
    if not email: return None
    return load_subs_db().get("subs", {}).get(email)

def sync_user_role_with_subscription(user):
    if not user: return "free", None
    email = user.get("login_id", "")
    base_role = user.get("role", "free")
    
    # Beta User Hardcode (예시)
    if email in ["coolguyhaeng@naver.com", "quartzk123@gmail.com"]:
        return "prime", "∞"
    
    sub = get_subscription(email)
    if not sub: return base_role, None
    
    try:
        exp_date = datetime.strptime(sub.get("expire_at"), "%Y-%m-%d").date()
    except: return base_role, None
    
    if now_kst().date() > exp_date and base_role in ["pro", "prime"]:
        update_user_role(email, "free") # downgrade
        set_subscription(email, "free")
        return "free", str(exp_date)
        
    return sub.get("role", base_role), str(exp_date)

# 포트폴리오 관리 (로컬 파일)
def load_portfolio_file():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("data", "")
        except: pass
    return ""

def save_portfolio_file(text_data):
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump({"data": text_data}, f, ensure_ascii=False)
        return True
    except: return False

# ---------------------------
# [v6.9 핵심] 지표 및 차트 로직
# ---------------------------

def calculate_supertrend(df, period=10, multiplier=3):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high+low)/2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    
    for i in range(period, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else: final_upper.iloc[i] = final_upper.iloc[i-1]
        
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else: final_lower.iloc[i] = final_lower.iloc[i-1]
        
        if trend.iloc[i-1] == 1:
            trend.iloc[i] = -1 if close.iloc[i] < final_lower.iloc[i-1] else 1
        else:
            trend.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i-1] else -1
            
        supertrend.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]
        
    df['SuperTrend'] = supertrend
    df['Trend'] = trend
    return df

@st.cache_data(ttl=600)
def get_stock_chart_data(code):
    if not FDR_OK: return None
    try:
        code_str = str(code).zfill(6)
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        df = fdr.DataReader(code_str, start_date)
        if df is None or df.empty: return None
        
        # MA
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        
        # [v6.9] Bollinger & BandWidth
        std20 = df['Close'].rolling(20).std()
        df['BB_UPPER'] = df['MA20'] + 2 * std20
        df['BB_LOWER'] = df['MA20'] - 2 * std20
        df['BandWidth'] = ((df['BB_UPPER'] - df['BB_LOWER']) / df['MA20']) * 100
        
        # RSI
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        df['RSI14'] = 100 - (100 / (1 + rs))
        
        df = calculate_supertrend(df)
        return df.tail(100)
    except: return None

# ---------------------------
# [v6.9 핵심] 시각화 함수들
# ---------------------------

def plot_interactive_chart(df, code, name, entry, stop, t1, t2, show_bb=True, show_rsi=False, show_bw=False):
    """ v6.9: BandWidth 서브차트 지원 """
    rows = 2
    
    # RSI나 BandWidth가 켜지면 행 추가
    extra_charts = []
    if show_bw: extra_charts.append("BW")
    if show_rsi: extra_charts.append("RSI")
    
    rows += len(extra_charts)
    
    # 높이 배분
    if extra_charts:
        row_heights = [0.6, 0.15] + [0.25/len(extra_charts)]*len(extra_charts)
    else:
        row_heights = [0.7, 0.3]
    
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=row_heights if len(row_heights)==rows else None
    )

    # 1. Main Chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="주가", showlegend=False
    ), row=1, col=1)
    
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(color='lightgray', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(color='lightgray', width=1, dash='dot'), name='BB Lower'), row=1, col=1)
        
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
    
    # SuperTrend markers
    up = df[df['Trend'] == 1]
    down = df[df['Trend'] == -1]
    fig.add_trace(go.Scatter(x=up.index, y=up['SuperTrend'], mode='markers', marker=dict(color='green', size=2), name='상승추세'), row=1, col=1)
    fig.add_trace(go.Scatter(x=down.index, y=down['SuperTrend'], mode='markers', marker=dict(color='red', size=2), name='하락추세'), row=1, col=1)

    # Lines
    for val, c, lbl in [(entry, 'blue', '진입'), (stop, 'red', '손절'), (t1, 'green', '목표')]:
        if val > 0: fig.add_hline(y=val, line_dash="dash", line_color=c, row=1, col=1, annotation_text=lbl)

    # 2. Volume
    colors = ['red' if r['Close'] >= r['Open'] else 'blue' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, showlegend=False, name='거래량'), row=2, col=1)

    # 3. Sub Charts
    current_row = 3
    if show_bw and 'BandWidth' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BandWidth'], line=dict(color='teal', width=1.5), name='BandWidth(%)'), row=current_row, col=1)
        fig.add_hline(y=10, line_dash="dot", line_color="red", annotation_text="Squeeze(<10%)", row=current_row, col=1)
        current_row += 1
        
    if show_rsi and 'RSI14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], line=dict(color='violet', width=1.5), name='RSI'), row=current_row, col=1)
        fig.add_hline(y=30, line_color='blue', row=current_row, col=1)
        fig.add_hline(y=70, line_color='red', row=current_row, col=1)

    fig.update_layout(height=700 if extra_charts else 500, margin=dict(t=30, b=10, l=10, r=10), hovermode="x unified")
    return fig

def plot_market_breadth_gauge(ratio):
    """ [v6.9 New] 시장 온도계 (20일선 상회 비율) """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = ratio,
        title = {'text': "🌡️ 시장 온도 (20일선 상회%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 20], 'color': "#48cae4"}, # 침체
                {'range': [20, 80], 'color': "#f0f0f0"}, # 보통
                {'range': [80, 100], 'color': "#ff6b6b"}, # 과열
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': ratio
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_sector_treemap_v69(df):
    """ [v6.9] 섹터 모멘텀(수익률) 기반 컬러링 """
    if df.empty: return go.Figure()
    
    sector_col = "업종_대분류" if "업종_대분류" in df.columns else "업종"
    if sector_col not in df.columns: return go.Figure()
    
    # 섹터별 평균 수익률 계산
    sector_mom = df.groupby(sector_col)["ret_5d_%"].transform("mean")
    df["Sector_Mom"] = sector_mom

    fig = px.treemap(
        df, path=[sector_col, "종목명"], values="거래대금(억원)",
        color="Sector_Mom", color_continuous_scale="RdYlGn",
        range_color=[-10, 10], # -10% ~ +10% 색상 범위 고정
        title="<b>🔥 섹터 모멘텀 히트맵 (5일 수익률)</b>",
        custom_data=["ret_5d_%", "LDY_SCORE"]
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>수익률: %{customdata[0]:.2f}%<br>점수: %{customdata[1]:.1f}")
    fig.update_layout(height=350, margin=dict(t=40, l=5, r=5, b=5))
    return fig

# ---------------------------
# 데이터 처리
# ---------------------------
@st.cache_data(ttl=600)
def load_and_process_data(raw_url, local_path):
    df = None
    src = None

    raw_url = str(raw_url) if raw_url is not None else ""
    is_http = raw_url.startswith(("http://", "https://"))

    # 1) 원격 URL이면 먼저 원격 시도
    if is_http:
        try:
            r = requests.get(raw_url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            src = "remote"
        except Exception:
            df = None
            src = None

    # 2) 원격 실패 or 애초에 URL이 아니면 로컬 fallback
    if df is None:
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            src = "local"
        else:
            # ★ 너는 4개 변수로 받으니까, 반환도 4개로 맞춰야 함
            return None, None, None, "fail"

    # 2. Normalize
    if "거래대금(원)" in df.columns and "거래대금(억원)" not in df.columns:
        df["거래대금(억원)"] = (pd.to_numeric(df["거래대금(원)"], errors='coerce')/1e8).round(1)
        
    num_cols = ["LDY_SCORE", "ret_5d_%", "BandWidth", "ma20_above", "RR1", "EBS", "RSI14"]
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    # 3. Market Breadth Calculation (v6.9)
    if "ma20_above" in df.columns:
        breadth = df["ma20_above"].mean() * 100
    else:
        breadth = 50.0 

    # 4. Sort
    df = df.sort_values("LDY_SCORE", ascending=False)
    
    # 5. Timestamp infer
    ts = None
    if "기준일" in df.columns:
        ts = df["기준일"].max()
        
    return df, breadth, ts, src

# ---------------------------
# 메인 로직 실행
# ---------------------------
df_scored, market_breadth, data_ts, data_src = load_and_process_data(RAW_SRC, LOCAL_RAW)

if df_scored is None:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

# ---------------------------
# 사이드바 (필터 & Auth)
# ---------------------------
with st.sidebar:
    user = render_auth_box(show_debug=False)
    if user:
        auth_status, expire_str = sync_user_role_with_subscription(user)
    else:
        auth_status, expire_str = "guest", None
        
    st.divider()
    
    # [v6.9 New Filter] Squeeze Hunter
    st.subheader("⚡ 필터 설정")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        min_score = st.number_input("최소 점수", 0, 100, 70, step=5)
    with col_sb2:
        only_squeeze = st.checkbox("⚡ Squeeze", help="BandWidth < 15% 인 종목만 보기")
    
    selected_regimes = []
    if "REGIME" in df_scored.columns:
        all_regimes = sorted(df_scored["REGIME"].dropna().unique())
        selected_regimes = st.multiselect("추세(REGIME)", all_regimes, default=all_regimes)
        
    st.divider()

with st.sidebar:
    st.divider()
    with st.expander("🔧 데이터 디버그", expanded=True):
        st.write("data_src =", data_src)
        st.write("data_ts  =", data_ts)
        st.write("rows     =", len(df_scored))
        st.write("min_score =", min_score)
        st.write("only_squeeze =", only_squeeze)
        st.write("columns =", [c for c in df_scored.columns])

        if "LDY_SCORE" in df_scored.columns:
            st.write("LDY_SCORE desc")
            st.write(df_scored["LDY_SCORE"].describe())

        # ROUTE/REGIME가 진짜 존재하는지 즉시 확인
        show_cols = [c for c in ["종목명","종목코드","LDY_SCORE","ROUTE","REGIME","ENTRY_SCORE","거래대금(억원)"] if c in df_scored.columns]
        if show_cols:
            st.dataframe(df_scored[show_cols].head(20), use_container_width=True)

with st.sidebar:
    with st.expander("🔧 데이터 디버그(추가)", expanded=True):
        st.write("LDY_SCORE dtype =", df_scored["LDY_SCORE"].dtype)
        st.write("LDY_SCORE max   =", float(pd.to_numeric(df_scored["LDY_SCORE"], errors="coerce").max()))
        st.write("count >= 70     =", int((pd.to_numeric(df_scored["LDY_SCORE"], errors="coerce") >= 70).sum()))
        st.write("top10 scores    =", list(pd.to_numeric(df_scored["LDY_SCORE"], errors="coerce").sort_values(ascending=False).head(10)))

    
    # 내 자산 입력 (v6.8 복구)
    if auth_status in ["pro", "prime", "admin"]:
        st.subheader("💼 내 자산 입력")
        saved_pf = load_portfolio_file()
        pf_input = st.text_area(
            "종목명:평단가:수량 (엔터로 구분)",
            value=saved_pf,
            placeholder="삼성전자:70000:10\nCASH:5000000:1",
            height=100,
        )
        if st.button("💾 저장/분석"):
            save_portfolio_file(pf_input)
            st.success("저장되었습니다.")
    else:
        pf_input = ""
        
    st.divider()
    st.info(f"👤 등급: **{auth_status.upper()}**")
    if expire_str: st.caption(f"만료: {expire_str}")

# ---------------------------
# 필터링 로직
# ---------------------------

# [수정됨] Squeeze 체크가 켜져 있으면 '점수 컷'을 무시하고 밴드폭으로만 검색
if only_squeeze and "BandWidth" in df_scored.columns:
    filtered_df = df_scored[df_scored["BandWidth"] < 15].copy()
else:
    # 체크가 꺼져 있으면 기존대로 점수(LDY_SCORE) 기준 필터링
    filtered_df = df_scored[df_scored["LDY_SCORE"] >= min_score].copy()

if selected_regimes and "REGIME" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["REGIME"].isin(selected_regimes)]

# 권한별 제한
limit_map = {"guest": 3, "free": 5, "pro": 20, "prime": 200, "admin": 300}
view_df = filtered_df.head(limit_map.get(auth_status, 3))

# ---------------------------
# 메인 탭 UI
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 시장 대시보드", "🔭 종목 디테일", "💼 내 자산", 
    "📮 문의 게시판", "⚖️ 이용 약관", "🧩 업데이트 노트"
])

# --- TAB 1: 시장 대시보드 (v6.9) ---
with tab1:
    st.markdown(f"#### 📅 기준일: {data_ts} (Source: {data_src})")
    
    # 1. Market Gauges
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_market_breadth_gauge(market_breadth), use_container_width=True)
        st.caption("※ 시장 온도: 20일 이동평균선 위에 있는 종목 비율")
    with c2:
        avg_rsi = df_scored["RSI14"].mean() if "RSI14" in df_scored.columns else 50
        st.metric("시장 평균 RSI", f"{avg_rsi:.1f}")
        st.progress(min(100, max(0, int(avg_rsi))))
        st.caption("※ 전체 종목 평균 RSI (과열/침체 판단)")

    st.divider()
    
    # 2. Sector Heatmap
    st.subheader("🔥 섹터 모멘텀 히트맵")
    if "업종_대분류" in df_scored.columns:
        st.plotly_chart(plot_sector_treemap_v69(df_scored.head(100)), use_container_width=True)
    else:
        st.info("섹터 데이터가 없습니다.")

# --- TAB 2: 종목 디테일 ---
with tab2:
    st.subheader(f"🔍 필터 결과 ({len(view_df)}개)")
    
    if view_df.empty:
        st.warning("조건에 맞는 종목이 없습니다.")
    else:
        opts = [f"{r['종목명']} ({r['종목코드']}) - {r['LDY_SCORE']}점" for _, r in view_df.iterrows()]
        sel_opt = st.selectbox("종목 선택", opts)
        
        if sel_opt:
            idx = opts.index(sel_opt)
            row = view_df.iloc[idx]
            code = str(row['종목코드']).zfill(6)
            
            # v6.9 Chart Options
            c_opt1, c_opt2, c_opt3 = st.columns(3)
            with c_opt1: show_bb = st.toggle("볼린저 밴드", True)
            with c_opt2: show_bw = st.toggle("⚡ BandWidth (변동성)", True)
            with c_opt3: show_rsi = st.toggle("RSI 보조지표", False)
            
            chart_data = get_stock_chart_data(code)
            if chart_data is not None:
                fig = plot_interactive_chart(
                    chart_data, code, row['종목명'], 
                    row.get('추천매수가',0), row.get('손절가',0), row.get('추천매도가1',0), 0,
                    show_bb, show_rsi, show_bw
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Squeeze Alert
                curr_bw = chart_data['BandWidth'].iloc[-1]
                if curr_bw < 15:
                    st.error(f"⚡ **Squeeze Alert!** 현재 밴드폭 {curr_bw:.1f}%로 변동성 폭발 임박 구간입니다.")
                else:
                    st.info(f"현재 밴드폭: {curr_bw:.1f}%")
            else:
                st.error("차트 데이터를 불러올 수 없습니다.")
                
            st.dataframe(pd.DataFrame([row]).dropna(axis=1), use_container_width=True)

    st.divider()
    st.subheader("📋 리스트 뷰")
    
    disp_cols = ["종목명", "종목코드", "LDY_SCORE", "ROUTE", "BandWidth", "ret_5d_%", "추천매수가", "손절가", "REGIME"]
    valid_cols = [c for c in disp_cols if c in view_df.columns]
    
    st.dataframe(
        view_df[valid_cols],
        use_container_width=True,
        column_config={
            "LDY_SCORE": st.column_config.ProgressColumn("점수", min_value=0, max_value=100, format="%.1f"),
            "BandWidth": st.column_config.NumberColumn("밴드폭(%)", format="%.1f%%"),
            "ret_5d_%": st.column_config.NumberColumn("5일등락", format="%.1f%%")
        }
    )

# --- TAB 3: 내 자산 ---
with tab3:
    if auth_status in ["guest", "free"]:
        st.warning("🔒 내 자산 분석은 Pro 등급 이상부터 가능합니다.")
    elif not pf_input:
        st.info("👈 사이드바에 포트폴리오를 입력하고 저장해주세요.")
    else:
        lines = pf_input.strip().split('\n')
        total_eval = 0
        total_buy = 0
        pf_rows = []
        
        for line in lines:
            try:
                name, avg, qty = line.split(':')
                avg = float(avg); qty = int(qty)
                
                if "CASH" in name.upper():
                    cur_price = avg
                else:
                    # 데이터에서 종목명으로 현재가 찾기 (Fallback)
                    found = df_scored[df_scored['종목명'] == name]
                    if not found.empty:
                        cur_price = float(found.iloc[0]['종가'])
                    else:
                        cur_price = avg 
                
                eval_amt = cur_price * qty
                buy_amt = avg * qty
                total_eval += eval_amt
                total_buy += buy_amt
                
                pf_rows.append({
                    "종목명": name, "평가금액": int(eval_amt), 
                    "수익률": (cur_price-avg)/avg*100 if avg>0 else 0
                })
            except: pass
            
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매수금액", f"{int(total_buy):,}원")
        c2.metric("총 평가금액", f"{int(total_eval):,}원")
        rate = (total_eval - total_buy)/total_buy*100 if total_buy > 0 else 0
        c3.metric("수익률", f"{rate:+.2f}%", delta_color="normal" if rate>=0 else "inverse")
        
        if pf_rows:
            st.dataframe(pd.DataFrame(pf_rows), use_container_width=True)

# --- TAB 4: 문의 게시판 ---
with tab4:
    st.subheader("📮 문의 게시판")
    with st.form("inquiry_form"):
        nick = st.text_input("닉네임", value=user.get("nickname","") if user else "")
        title = st.text_input("제목")
        content = st.text_area("내용")
        if st.form_submit_button("등록"):
            db = load_inquiry_db()
            db["inquiries"].append({"nick": nick, "title": title, "content": content, "date": str(now_kst())})
            save_inquiry_db(db)
            st.success("등록되었습니다.")
            
    db = load_inquiry_db()
    for item in reversed(db.get("inquiries", [])[-10:]):
        with st.container(border=True):
            st.caption(f"{item.get('nick')} · {item.get('date')}")
            st.markdown(f"**{item.get('title')}**")
            st.text(item.get('content'))

# --- TAB 5: 이용 약관 ---
with tab5:
    st.markdown("### ⚠️ 투자 유의사항")
    st.warning("본 서비스는 투자 판단의 참고 자료일 뿐이며, 실제 투자의 책임은 전적으로 사용자에게 있습니다.")
    st.markdown("- 제공되는 데이터(FDR, KRX 등)는 지연되거나 오류가 있을 수 있습니다.")
    st.markdown("- 과거의 성과(백테스트, 스코어)가 미래의 수익을 보장하지 않습니다.")

# --- TAB 6: 업데이트 노트 ---
with tab6:
    st.subheader("🧩 업데이트 노트")
    if CHANGELOG:
        latest = CHANGELOG[0]
        st.success(f"**v{latest['version']} ({latest['date']})** : {latest['title']}")
        for log in CHANGELOG:
            with st.expander(f"v{log['version']} - {log['title']}"):
                for item in log['items']:
                    st.write(f"- {item}")
    else:
        st.info("버전 기록이 없습니다.")
