# -*- coding: utf-8 -*-
"""
LDY Pro Trader v4.6 — Global Rank (Market Radar + Instant Chart + Context)
- Market Radar: KOSPI/KOSDAQ 이동평균선 기반 시장 국면(Bull/Bear) 실시간 판단
- Instant Chart: Top 10 종목 클릭 시 최근 60일 캔들 차트 및 진입/청산 라인 시각화
- Scoring: 기존 v4.4의 복합 점수(LDY_SCORE) 알고리즘 계승
- Data Source: 배치 CSV(Github) + 실시간 보조 데이터(FDR)
"""

import os, io, math, json, requests, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -------- Optional deps --------
try:
    import FinanceDataReader as fdr
    FDR_OK = True
except ImportError:
    FDR_OK = False

try:
    from pykrx import stock
    PYKRX_OK = True
except Exception:
    PYKRX_OK = False

# -------- Page Config --------
st.set_page_config(page_title="LDY Pro Trader v4.6", layout="wide", page_icon="📈")
st.title("🏆 LDY Pro Trader v4.6")
st.caption("Global Rank Scoring + Market Radar + Instant Chart Visualization")

# -------- Constants --------
RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"

PASS_EBS = 4
MIN_TURN_KOSPI = 200.0
MIN_TURN_KOSDAQ = 100.0
MIN_TURN_DEFAULT = 100.0

# 가중치 (v4.4 동일)
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = 0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10
# 페널티 (v4.4 동일)
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = 6.0, 6.0, 4.0
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 4.0, 2.0

# -------- Helpers: IO --------
@st.cache_data(ttl=600)
def load_csv_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30); r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=600)
def load_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")

def log_src(df: pd.DataFrame, src: str):
    st.toast(f"Data Loaded: {src} ({len(df)} rows)", icon="✅")

# -------- Helpers: Market Radar (v4.6 NEW) --------
@st.cache_data(ttl=3600)
def get_market_status():
    """KOSPI, KOSDAQ 지수의 20일선 위/아래 여부 판단"""
    if not FDR_OK:
        return "Unknown", 0.0, "Unknown", 0.0
    
    def _check(ticker):
        try:
            df = fdr.DataReader(ticker, datetime.now().year - 1)
            if df.empty: return "Unknown", 0.0
            # 최근 60일치만
            df = df.tail(60)
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            curr = df['Close'].iloc[-1]
            diff = ((curr - ma20) / ma20) * 100
            status = "Bull" if diff > 0 else "Bear"
            return status, diff
        except:
            return "Error", 0.0

    kp_stat, kp_diff = _check('KS11') # KOSPI
    kq_stat, kq_diff = _check('KQ11') # KOSDAQ
    return kp_stat, kp_diff, kq_stat, kq_diff

# -------- Helpers: Charting (v4.6 NEW) --------
@st.cache_data(ttl=600) # 차트는 10분 캐시
def get_stock_chart_data(code):
    if not FDR_OK: return None
    try:
        # 넉넉히 4달치 가져와서 60일 표시
        start_date = datetime.now() - timedelta(days=120)
        df = fdr.DataReader(code, start_date)
        return df.tail(60)
    except:
        return None

def plot_interactive_chart(df, code, name, entry, stop, target1, target2):
    if df is None or df.empty:
        return go.Figure()

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name=name,
        increasing_line_color='#ef5350', # Red (KR style up)
        decreasing_line_color='#2979ff'  # Blue (KR style down)
    )])

    # 가로선 (Entry, Stop, Target)
    colors = {"Entry": "blue", "Stop": "red", "Target1": "green", "Target2": "green"}
    lines = [
        (entry, "Entry", "dash", "blue"),
        (stop, "Stop", "dot", "red"),
        (target1, "Tgt1", "dot", "green"),
        (target2, "Tgt2", "dot", "green")
    ]
    
    for val, label, dash, color in lines:
        if pd.notna(val) and val > 0:
            fig.add_hline(y=val, line_dash=dash, line_color=color, 
                          annotation_text=f"{label}: {val:,.0f}", annotation_position="top right")

    # 이동평균선 (20, 60)
    ma20 = df['Close'].rolling(20).mean()
    ma60 = df['Close'].rolling(60).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma20, line=dict(color='orange', width=1), name='MA20'))
    fig.add_trace(go.Scatter(x=df.index, y=ma60, line=dict(color='purple', width=1), name='MA60'))

    fig.update_layout(
        title=f"<b>{name}</b> ({code}) - Daily Chart",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -------- Utils (Same as v4.4) --------
def z6(x) -> str:
    s = str(x)
    return s.zfill(6) if s.isdigit() else s

def nz_num(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")

def ensure_turnover(df: pd.DataFrame) -> pd.DataFrame:
    if "거래대금(억원)" not in df.columns:
        base = None
        if "거래대금(원)" in df.columns:
            base = nz_num(df["거래대금(원)"])
        elif all(c in df.columns for c in ["거래량","종가"]):
            base = nz_num(df["거래량"]) * nz_num(df["종가"])
        if base is not None:
            df["거래대금(억원)"] = (base/1e8).round(2)
    return df

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cmap = {
        "Date":"날짜","date":"날짜",
        "Code":"종목코드","티커":"종목코드","ticker":"종목코드",
        "Name":"종목명","name":"종목명",
        "Open":"시가","High":"고가","Low":"저가","Close":"종가","Volume":"거래량",
        "거래대금":"거래대금(원)","시가총액":"시가총액(원)"
    }
    for k,v in cmap.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.replace(".0","", regex=False).map(z6)
    else:
        df["종목코드"] = None
    if "시장" not in df.columns: df["시장"] = "ALL"
    if "종목명" not in df.columns: df["종목명"] = None
    
    target_cols = ["시가","고가","저가","종가","거래량","거래대금(원)","시가총액(원)","거래대금(억원)","시가총액(억원)",
                   "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS",
                   "추천매수가","추천매도가1","추천매도가2","손절가","MFI14"]
    for c in target_cols:
        if c in df.columns: df[c] = nz_num(df[c])
    return ensure_turnover(df)

@st.cache_data(ttl=21600)
def load_name_map() -> pd.DataFrame | None:
    # 1. URL CSV
    try:
        m = load_csv_url(CODES_URL)
        if {"종목코드","종목명"}.issubset(m.columns):
            m["종목코드"] = m["종목코드"].astype(str).map(z6)
            return m[["종목코드","종목명"]].drop_duplicates("종목코드")
    except: pass
    # 2. Local
    if os.path.exists(LOCAL_MAP):
        try:
            m = load_csv_path(LOCAL_MAP)
            if {"종목코드","종목명"}.issubset(m.columns):
                m["종목코드"] = m["종목코드"].astype(str).map(z6)
                return m[["종목코드","종목명"]].drop_duplicates("종목코드")
        except: pass
    # 3. FDR
    if FDR_OK:
        try:
            lst = fdr.StockListing("KRX")
            m = lst.rename(columns={"Code":"종목코드","Name":"종목명"})[["종목코드","종목명"]]
            m["종목코드"] = m["종목코드"].astype(str).map(z6)
            return m.drop_duplicates("종목코드")
        except: pass
    return None

def apply_names(df: pd.DataFrame) -> pd.DataFrame:
    mp = load_name_map()
    if mp is not None:
        df["종목코드"] = df["종목코드"].astype(str).map(z6)
        if "종목명" not in df.columns: df["종목명"] = None
        df = df.merge(mp, on="종목코드", how="left", suffixes=("","_map"))
        df["종목명"] = df["종목명"].fillna(df["종목명_map"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_map")], errors="ignore")
    df["종목명"] = df["종목명"].fillna("(이름없음)")
    return df

# -------- Logic: Scoring & Routing (Same as v4.4) --------
def liquidity_gate(x_turn: pd.Series, market: pd.Series) -> pd.Series:
    min_map = {"KOSPI": MIN_TURN_KOSPI, "KOSDAQ": MIN_TURN_KOSDAQ}
    mins = market.map(min_map).fillna(MIN_TURN_DEFAULT)
    return nz_num(x_turn) >= mins

def cap_q(s: pd.Series, q=90, floor=1.0):
    s = nz_num(s)
    if s.notna().sum()==0: return floor
    c = np.nanpercentile(s, q)
    if not np.isfinite(c) or c<=0: c=floor
    return float(max(c, floor))

def pct_norm_pos(s: pd.Series, q=90, floor=1.0):
    s = nz_num(s).clip(lower=0)
    cap = cap_q(s, q=q, floor=floor)
    return np.clip(s / cap, 0, 1)

def inv_dist_norm(dist: pd.Series, cap):
    d = nz_num(dist)
    return np.clip(1 - (d / cap), 0, 1)

def route_tag(row) -> str:
    rsi = row.get("RSI14", np.nan); slope = row.get("MACD_slope", np.nan)
    kairi = row.get("乖離%", np.nan); r5 = row.get("ret_5d_%", np.nan)
    near = row.get("Now%", np.nan)
    
    if pd.notna(r5) and pd.notna(near) and pd.notna(slope):
        if (r5 >= 3) and (near <= 0.7) and (slope > 0) and (abs(kairi) <= 6): return "🔼 BRK (돌파)"
    if pd.notna(rsi) and pd.notna(near):
        if (45 <= rsi <= 60) and (near <= 1.0) and (abs(kairi) <= 5): return "↩️ PULL (눌림)"
    if pd.notna(slope) and slope > 0 and pd.notna(r5) and r5 > 0 and abs(kairi) <= 7: return "📈 TREND (추세)"
    if pd.notna(rsi) and (rsi >= 67 or rsi <= 40): return "🔁 MR (반전)"
    return "—"

def build_global_score(lat: pd.DataFrame) -> pd.DataFrame:
    x = lat.copy()
    # 필요한 컬럼 보정
    cols = ["종가","추천매수가","손절가","추천매도가1","거래대금(억원)","시가총액(억원)",
            "RSI14","MACD_slope","Vol_Z","乖離%","ret_5d_%","ret_10d_%","EBS","MACD_hist","MFI14"]
    for c in cols: 
        if c not in x.columns: x[c] = np.nan

    close, entry, stop, t1 = nz_num(x["종가"]), nz_num(x["추천매수가"]), nz_num(x["손절가"]), nz_num(x["추천매도가1"])
    turn, rsi, slope, volz = nz_num(x["거래대금(억원)"]), nz_num(x["RSI14"]), nz_num(x["MACD_slope"]), nz_num(x["Vol_Z"])
    kairi, r5, r10, ebs = nz_num(x["乖離%"]), nz_num(x["ret_5d_%"]), nz_num(x["ret_10d_%"]), nz_num(x["EBS"]).fillna(0)

    # 팩터 계산
    rr_den = (entry - stop)
    rr1 = ((t1 - entry) / rr_den.replace(0, np.nan)).mask(entry.isna() | stop.isna() | t1.isna())
    now_gap = ((close - entry).abs() / entry * 100)
    t1_room = ((t1 - close) / close * 100)
    sl_room = ((close - stop) / close * 100)

    # 정규화
    rr_norm   = pct_norm_pos(rr1, q=90, floor=1.0)
    t1_norm   = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1)
    sl_norm   = np.clip(sl_room / cap_q(sl_room, q=90, floor=3.0), 0, 1)
    near_norm = inv_dist_norm(now_gap, cap=cap_q(now_gap, q=75, floor=1.0))
    
    ers_bits = (ebs>=PASS_EBS).astype(int) + (slope>0).astype(int) + ((rsi>=45)&(rsi<=65)).astype(int)
    ers_norm = np.clip(ers_bits/3.0, 0, 1)
    slope_pos_norm = pct_norm_pos(slope, q=90, floor=1.0)
    rsi_center = (1 - np.minimum((rsi-55).abs()/10, 1)).clip(0,1).fillna(0)
    mom_norm = np.clip(0.5*ers_norm + 0.3*slope_pos_norm + 0.2*rsi_center, 0, 1)

    # 유동성
    if turn.notna().any():
        lo = np.nanpercentile(turn, 30) if np.isfinite(np.nanpercentile(turn.dropna(), 30)) else np.nanmin(turn)
        hi = np.nanpercentile(turn, 90) if np.isfinite(np.nanpercentile(turn.dropna(), 90)) else np.nanmax(turn)
        span = max(hi - lo, 1e-9)
        liq_norm = np.clip((turn - lo) / span, 0, 1)
    else:
        liq_norm = pd.Series(0.0, index=x.index)

    # 기술
    vol_sweet = (1 - np.minimum((volz - 1).abs()/3, 1)).clip(0,1).fillna(0)
    kairi_norm = (1 - np.minimum(kairi.abs()/cap_q(kairi.abs(), q=80, floor=3.0), 1)).clip(0,1).fillna(0)
    tec_norm = np.clip(0.6*vol_sweet + 0.4*kairi_norm, 0, 1)

    # 점수 합산
    base_score = (100*W_RR*rr_norm) + (100*W_T1*t1_norm) + (100*W_SL*sl_norm) + \
                 (100*W_NEAR*near_norm) + (100*W_MOM*mom_norm) + (100*W_LIQ*liq_norm) + (100*W_TEC*tec_norm)

    # 패널티
    pen = pd.Series(0.0, index=x.index)
    pen += P_OVERHEAT_5D * np.clip((r5 - 10)/10, 0, 1)
    pen += P_OVERHEAT_10D* np.clip((r10 - 20)/20, 0, 1)
    pen += P_RSI_OUT * ((rsi < 45) | (rsi > 65)).astype(float)
    pen += P_MACD_NEG * (slope < 0).astype(float)
    near_cap = cap_q(now_gap, q=75, floor=1.0)
    pen += P_NEAR_FAR * np.clip((now_gap - near_cap)/near_cap, 0, 1)
    if turn.notna().any():
        p20 = np.nanpercentile(turn.dropna(), 20) if turn.dropna().size else -np.inf
        pen += P_LIQ_LOW * (turn < p20).astype(float)
    pen += P_VOL_SPIKE * (volz > 3).astype(float)

    score = np.clip(base_score - pen, 0, 100)

    x["RR1"] = rr1
    x["Now%"] = now_gap; x["T1여유%"] = t1_room; x["SL여유%"] = sl_room
    x["ERS"] = ers_bits.astype(float)
    x["LDY_SCORE"] = score.round(1)
    
    x["ROUTE"] = (x.apply(route_tag, axis=1) if len(x) else "—")
    x["_GATE_OK"] = liquidity_gate(x["거래대금(억원)"], x["시장"]).fillna(False)

    x = x.sort_values("LDY_SCORE", ascending=False, na_position="last")
    x["LDY_RANK"] = range(1, len(x)+1)

    x["WHY"] = (
        "MOM+" + (100*W_MOM*mom_norm).round(0).astype(int).astype(str) + " "
        "LIQ+" + (100*W_LIQ*liq_norm).round(0).astype(int).astype(str) + " "
        "TEC+" + (100*W_TEC*tec_norm).round(0).astype(int).astype(str) + " "
        "PEN-" + pen.round(0).astype(int).astype(str)
    )
    return x

# -------- Main Execution --------
# 1. Load Data
try:
    df_raw = load_csv_url(RAW_URL); log_src(df_raw, "Remote CSV")
except:
    if os.path.exists(LOCAL_RAW):
        df_raw = load_csv_path(LOCAL_RAW); log_src(df_raw, "Local CSV")
    else:
        st.error("❌ 데이터가 없습니다."); st.stop()

df = normalize_cols(df_raw)
df = apply_names(df)
latest = df.sort_values(["종목코드","날짜"]).groupby("종목코드").tail(1) if "날짜" in df.columns else df.copy()

# 2. Scoring
scored = build_global_score(latest)
scored = scored.dropna(subset=["추천매수가","손절가","추천매도가1","종가"])

# 3. Filtering
base = scored[(nz_num(scored.get("EBS")) >= PASS_EBS) & (scored["_GATE_OK"])].copy()
if len(base) < 10: # Fallback
    fb = scored[nz_num(scored.get("EBS")) >= (PASS_EBS-1)].copy()
    mm = {"KOSPI": 150.0, "KOSDAQ": 80.0}
    fb["_min"] = fb["시장"].map(mm).fillna(MIN_TURN_DEFAULT)
    fb = fb[ nz_num(fb["거래대금(억원)"]) >= fb["_min"] ]
    base = pd.concat([base, fb]).drop_duplicates("종목코드").sort_values("LDY_SCORE", ascending=False)

top10 = base.head(10).copy()
top10["통과"] = np.where(nz_num(top10.get("EBS")) >= PASS_EBS, "🚀", "⚠️")
top10["P_hit"] = (top10["LDY_SCORE"] / 100.0 * 0.8).clip(0, 1) * 100 # Simplified P_hit

# -------- UI Rendering --------

# [관리자/유료회원 권한 관리 로직] ====================================
with st.sidebar:
    st.divider()
    st.header("🔐 로그인 (Login)")
    input_pw = st.text_input("비밀번호를 입력하세요", type="password", placeholder="Password")
    
    # 👇 1. 관리자 비밀번호 (다운로드 권한 O)
    ADMIN_KEY = "2022322"
    
    # 👇 2. 유료회원 비밀번호 (매달 변경 가능)
    MEMBER_KEY = "220577" 
    
    # 권한 상태 변수 초기화
    auth_status = "free" # 기본은 무료
    
    if input_pw == ADMIN_KEY:
        auth_status = "admin"
        st.success("✅ 관리자 모드 (다운로드 가능)")
        view_df = top10  # 전체 공개
        
    elif input_pw == MEMBER_KEY:
        auth_status = "member"
        st.success("🎉 환영합니다! (유료 회원)")
        view_df = top10  # 전체 공개
        
    else:
        auth_status = "free"
        if input_pw: # 비밀번호를 쳤는데 틀린 경우
            st.error("❌ 비밀번호가 일치하지 않습니다.")
        st.info("🔒 무료 버전 (Top 3 공개)")
        view_df = top10.head(3)  # 3개만 공개
# ================================================================

# [Section 1] Market Radar
kp_st, kp_df, kq_st, kq_df = get_market_status()
col_m1, col_m2 = st.columns(2)
with col_m1:
    clr = "off" if kp_st=="Bull" else "inverse"
    st.metric("KOSPI Trend (MA20)", f"{kp_st}", f"{kp_df:.2f}%", delta_color=clr)
with col_m2:
    clr = "off" if kq_st=="Bull" else "inverse"
    st.metric("KOSDAQ Trend (MA20)", f"{kq_st}", f"{kq_df:.2f}%", delta_color=clr)

if kp_st == "Bear" and kq_st == "Bear":
    st.warning("🚨 **WARNING:** 양대 시장 모두 약세(20일선 하회). 현금 비중을 높이거나 짧은 줄타기 매매만 권장합니다.")

st.divider()

# [Section 2] Instant Chart & Analysis
st.subheader("🔭 Instant Analysis (상세 분석)", anchor=False)

# view_df(3개 혹은 전체)만 선택 옵션에 나옴
options = view_df.apply(lambda r: f"{r['종목명']} ({r['종목코드']}) - {r['ROUTE']}", axis=1).tolist()
sel = st.selectbox("종목을 선택하여 상세 차트와 코멘트를 확인하세요:", options, index=0 if options else None)

if sel:
    code = sel.split("(")[1].split(")")[0]
    row = top10[top10["종목코드"]==code].iloc[0]
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Plot Chart
        chart_df = get_stock_chart_data(code)
        if chart_df is not None:
            fig = plot_interactive_chart(chart_df, code, row['종목명'], 
                                         row['추천매수가'], row['손절가'], row['추천매도가1'], row['추천매도가2'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("차트 데이터를 불러올 수 없습니다 (FDR 연결 실패).")
            
    with c2:
        st.markdown(f"### {row['종목명']}")
        st.caption(f"Score: {row['LDY_SCORE']} | Rank: {row['LDY_RANK']}")
        
        st.markdown("#### 🚦 Trading Guide")
        st.write(f"**Strategy:** `{row['ROUTE']}`")
        if "BRK" in row['ROUTE']:
            st.success("⚡ **돌파 전략:** 현재가 혹은 시가가 추천가를 강하게 뚫을 때 진입")
        elif "PULL" in row['ROUTE']:
            st.info("🧲 **눌림 전략:** 추천가 부근까지 조정받을 때 분할 진입")
        else:
            st.secondary_background_message = True
            st.write("🌊 **추세 추종:** 5일선 지지 확인 후 진입")
            
        st.markdown("---")
        col_a, col_b = st.columns(2)
        col_a.metric("진입가", f"{row['추천매수가']:,}")
        col_b.metric("손절가", f"{row['손절가']:,}", delta=f"{row['SL여유%']:.1f}%", delta_color="inverse")
        
        col_c, col_d = st.columns(2)
        col_c.metric("목표1", f"{row['추천매도가1']:,}", delta=f"{row['T1여유%']:.1f}%")
        col_d.metric("RR Ratio", f"{row['RR1']:.2f}")

# [Section 3] Table View
st.subheader("📋 Daily Top 10 List", anchor=False)

# 관리자가 아니면 블러 처리된 느낌을 주기 위해 안내 메시지 표시
if auth_status == "free":
    st.warning("🔒 **무료 버전은 상위 3개 종목만 공개됩니다.**")
    st.markdown("""
    > **나머지 7개 종목과 전체 데이터를 보고 싶으신가요?** > 🚀 **[LDY Pro Trader 정식 버전 구매하기 (링크)](https://your-sales-link.com)** > *서버비 0원, 평생 소장 가능한 소스코드 패키지를 제공합니다.*
    """)

disp_cols = [
    "LDY_RANK","통과","ROUTE","시장","종목명","종목코드","LDY_SCORE","P_hit",
    "종가","추천매수가","손절가","추천매도가1","RR1","Now%","MFI14", 
    "거래대금(억원)","ret_5d_%","WHY"
]

col_config = {
    "LDY_RANK": st.column_config.NumberColumn("RANK", format="%d"),
    "LDY_SCORE": st.column_config.ProgressColumn("Score", format="%.1f", min_value=0, max_value=100),
    "P_hit": st.column_config.NumberColumn("Prob(%)", format="%.1f"),
    
    # 👇 여기 가격 관련 컬럼들에 '%,d' 포맷을 적용했습니다.
    "종가": st.column_config.NumberColumn("Close", format="%,d"),
    "추천매수가": st.column_config.NumberColumn("Entry", format="%,d"),
    "손절가": st.column_config.NumberColumn("Stop", format="%,d"),
    "추천매도가1": st.column_config.NumberColumn("Target", format="%,d"),
    
    "Now%": st.column_config.NumberColumn("Gap%", format="%.2f"),
    "MFI14": st.column_config.NumberColumn("MFI(자금)", format="%.1f"),
    "거래대금(억원)": st.column_config.NumberColumn("Amt(억)", format="%,d"),
}

# view_df (필터링된 데이터) 사용
st.dataframe(
    view_df[disp_cols],
    hide_index=True,
    use_container_width=True,
    column_config=col_config,
    height=400 if auth_status in ["admin", "member"] else 200 # 무료일 땐 표 높이를 줄임
)

# [Section 4] Downloads (관리자 전용 기능)
if auth_status == "admin":
    full_export = scored.sort_values("LDY_SCORE", ascending=False).head(2000)
    csv_data = full_export.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 데이터 다운로드 (Admin Only)",
        data=csv_data,
        file_name="ldy_rank_v46.csv",
        mime="text/csv"
    )
elif auth_status == "member":
    st.button("📥 데이터 다운로드", disabled=True, help="데이터 다운로드는 관리자 전용 기능입니다.")
else:
    st.button("📥 전체 데이터 다운로드 (유료 전용)", disabled=True)
