# -*- coding: utf-8 -*-
"""
LDY Pro Trader v4.6 (Final Stable Version)
- Features: Market Radar, Instant Chart (MA20/60), Quant Scoring, Access Control
"""
import os, io, math, json, requests, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 라이브러리 로드 확인
try: import FinanceDataReader as fdr; FDR_OK = True
except: FDR_OK = False
try: from pykrx import stock; PYKRX_OK = True
except: PYKRX_OK = False

# 2. 페이지 설정
st.set_page_config(page_title="LDY Pro Trader v4.6", layout="wide", page_icon="📈")
st.title("🏆 LDY Pro Trader v4.6")
st.caption("Global Rank Scoring + Market Radar + Instant Chart Visualization")

# 3. 상수 설정
RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"

PASS_EBS = 4
MIN_TURN_KOSPI, MIN_TURN_KOSDAQ, MIN_TURN_DEFAULT = 200.0, 100.0, 100.0
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = 0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = 6.0, 6.0, 4.0
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 4.0, 2.0

# 4. 헬퍼 함수들
@st.cache_data(ttl=600)
def load_csv_url(url):
    r = requests.get(url, timeout=30); r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=600)
def load_csv_path(path): return pd.read_csv(path, encoding="utf-8")

def log_src(df, src): st.toast(f"Data Loaded: {src} ({len(df)} rows)", icon="✅")

def get_last_business_date(d=datetime.now()):
    d = d.date()
    if d.weekday() == 5: d -= timedelta(days=1) # 토요일 -> 금요일
    elif d.weekday() == 6: d -= timedelta(days=2) # 일요일 -> 금요일
    return d.strftime("%Y%m%d")

@st.cache_data(ttl=3600)
def get_market_status():
    if not FDR_OK: return "Unknown", 0.0, "Unknown", 0.0
    end_date_str = get_last_business_date()
    
    def _check(ticker):
        try:
            start_date = datetime.now().year - 2
            df = fdr.DataReader(ticker, str(start_date), end=end_date_str)
            if df.empty or len(df) < 60: return "Unknown", 0.0
            df = df.tail(60)
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            curr = df['Close'].iloc[-1]
            if np.isnan(ma20) or ma20 == 0: return "Unknown", 0.0
            return "Bull" if ((curr - ma20)/ma20) > 0 else "Bear", ((curr-ma20)/ma20)*100
        except: return "Error", 0.0
    
    kp_stat, kp_diff = _check('KS11')
    kq_stat, kq_diff = _check('KQ11')
    return kp_stat, kp_diff, kq_stat, kq_diff

@st.cache_data(ttl=600)
def get_stock_chart_data(code):
    if not FDR_OK: return None
    try:
        # 1년치 데이터 가져와서 MA60 계산 보장
        start_date = datetime.now() - timedelta(days=365)
        df = fdr.DataReader(code, start_date)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        return df.tail(60)
    except: return None

def plot_interactive_chart(df, code, name, entry, stop, target1, target2):
    if df is None or df.empty: return go.Figure()
    
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="주가", increasing_line_color='#ef5350', decreasing_line_color='#2979ff',
        hovertemplate="<b>날짜: %{x|%Y-%m-%d}</b><br>시가: %{open:,}원<br>고가: %{high:,}원<br>저가: %{low:,}원<br>종가: %{close:,}원<extra></extra>"
    )])
    
    lines = [(entry, "🔵진입", "dash", "blue"), (stop, "🔴손절", "dot", "red"), (target1, "🟢목표1", "dot", "green"), (target2, "🟢목표2", "dot", "green")]
    for val, label, dash, color in lines:
        if pd.notna(val) and val > 0:
            fig.add_hline(y=val, line_dash=dash, line_color=color, annotation_text=f"{label}: {val:,.0f}", annotation_position="top right", annotation_font=dict(size=12, color=color))
            
    if 'MA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1.5), name='20일선 (생명선)', hovertemplate="20일선: %{y:,.0f}원<extra></extra>"))
    if 'MA60' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='purple', width=1.5), name='60일선 (수급선)', hovertemplate="60일선: %{y:,.0f}원<extra></extra>"))
    
    fig.update_layout(title=dict(text=f"<b>{name}</b> ({code}) 일봉 차트", font=dict(size=20)), yaxis_title="주가 (원)", yaxis_tickformat=',', xaxis_tickformat='%Y-%m-%d', xaxis_rangeslider_visible=False, template="plotly_dark", height=500, margin=dict(l=20,r=20,t=50,b=20), legend=dict(orientation="h",y=1.02,x=1), hovermode="x unified")
    return fig

# 5. 데이터 처리 유틸
def z6(x): return str(x).zfill(6) if str(x).isdigit() else str(x)
def nz_num(s): return pd.to_numeric(s, errors="coerce")
def ensure_turnover(df):
    if "거래대금(억원)" not in df.columns:
        base = None
        if "거래대금(원)" in df.columns: base = nz_num(df["거래대금(원)"])
        elif "거래량" in df.columns and "종가" in df.columns: base = nz_num(df["거래량"]) * nz_num(df["종가"])
        if base is not None: df["거래대금(억원)"] = (base/1e8).round(2)
    return df

def normalize_cols(df): return ensure_turnover(df)
def apply_names(df): return df

def liquidity_gate(x_turn, market):
    min_map = {"KOSPI": MIN_TURN_KOSPI, "KOSDAQ": MIN_TURN_KOSDAQ}
    return nz_num(x_turn) >= market.map(min_map).fillna(MIN_TURN_DEFAULT)

def cap_q(s, q=90, floor=1.0):
    c = np.nanpercentile(nz_num(s), q)
    return float(max(c, floor)) if np.isfinite(c) else floor

def pct_norm_pos(s, q=90, floor=1.0):
    s = nz_num(s).clip(lower=0)
    return np.clip(s / cap_q(s, q, floor), 0, 1)

def inv_dist_norm(dist, cap): return np.clip(1 - (nz_num(dist)/cap), 0, 1)

def route_tag(row):
    rsi = row.get("RSI14", np.nan); slope = row.get("MACD_Slope", np.nan) 
    kairi = row.get("이격도", np.nan); r5 = row.get("ret_5d_%", np.nan) 
    near = row.get("Now%", np.nan)
    # 컬럼 이름 호환성 체크 (collector 버전에 따라 다를 수 있음)
    if pd.isna(slope): slope = row.get("MACD_slope", np.nan)
    if pd.isna(kairi): kairi = row.get("乖離%", np.nan)
    
    if pd.notna(r5) and pd.notna(near) and pd.notna(slope):
        if (r5 >= 3) and (near <= 0.7) and (slope > 0) and (abs(kairi) <= 6): return "🔼 BRK (돌파)"
    if pd.notna(rsi) and pd.notna(near):
        if (45 <= rsi <= 60) and (near <= 1.0) and (abs(kairi) <= 5): return "↩️ PULL (눌림)"
    return "—"

def build_global_score(lat):
    x = lat.copy()
    # 필수 컬럼 확보 (없으면 NaN)
    required = ["종가","추천매수가","손절가","추천매도가1","거래대금(억원)","RSI14","MACD_Slope","MACD_slope","Vol_Z","거래강도","이격도","乖離%","ret_5d_%","ret_10d_%","EBS","MACD_Hist","MACD_hist","MFI14"]
    for c in required:
        if c not in x.columns: x[c] = np.nan

    # 컬럼명 호환성 처리
    slope_col = "MACD_Slope" if "MACD_Slope" in x.columns and x["MACD_Slope"].notna().any() else "MACD_slope"
    kairi_col = "이격도" if "이격도" in x.columns and x["이격도"].notna().any() else "乖離%"
    vol_col = "거래강도" if "거래강도" in x.columns and x["거래강도"].notna().any() else "Vol_Z"

    close, entry, stop, t1 = nz_num(x["종가"]), nz_num(x["추천매수가"]), nz_num(x["손절가"]), nz_num(x["추천매도가1"])
    turn, rsi, slope, volz = nz_num(x["거래대금(억원)"]), nz_num(x["RSI14"]), nz_num(x[slope_col]), nz_num(x[vol_col])
    kairi, r5, ebs = nz_num(x[kairi_col]), nz_num(x["ret_5d_%"]), nz_num(x["EBS"]).fillna(0)

    rr_den = (entry - stop)
    rr1 = ((t1 - entry) / rr_den.replace(0, np.nan)).mask(entry.isna() | stop.isna() | t1.isna())
    now_gap = ((close - entry).abs() / entry * 100)
    t1_room = ((t1 - close) / close * 100)
    sl_room = ((close - stop) / close * 100)

    rr_norm   = pct_norm_pos(rr1, q=90, floor=1.0).fillna(0)
    t1_norm   = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1).fillna(0)
    sl_norm   = np.clip(sl_room / cap_q(sl_room, q=90, floor=3.0), 0, 1).fillna(0)
    near_norm = inv_dist_norm(now_gap, cap=cap_q(now_gap, q=75, floor=1.0)).fillna(0)
    
    ers_bits = (ebs>=PASS_EBS).astype(int) + (slope>0).astype(int) + ((rsi>=45)&(rsi<=65)).astype(int)
    ers_norm = np.clip(ers_bits/3.0, 0, 1).fillna(0)
    slope_pos_norm = pct_norm_pos(slope, q=90, floor=1.0).fillna(0)
    mom_norm = np.clip(0.5*ers_norm + 0.3*slope_pos_norm, 0, 1).fillna(0)

    if turn.notna().any():
        lo, hi = np.nanpercentile(turn, 30), np.nanpercentile(turn, 90)
        liq_norm = np.clip((turn - lo) / max(hi-lo, 1e-9), 0, 1).fillna(0)
    else: liq_norm = 0.0

    vol_sweet = (1 - np.minimum((volz - 1).abs()/3, 1)).clip(0,1).fillna(0)
    kairi_norm = (1 - np.minimum(kairi.abs()/cap_q(kairi.abs(), q=80, floor=3.0), 1)).clip(0,1).fillna(0)
    tec_norm = np.clip(0.6*vol_sweet + 0.4*kairi_norm, 0, 1).fillna(0)

    base_score = (100*W_RR*rr_norm) + (100*W_T1*t1_norm) + (100*W_SL*sl_norm) + \
                 (100*W_NEAR*near_norm) + (100*W_MOM*mom_norm) + (100*W_LIQ*liq_norm) + (100*W_TEC*tec_norm)
    
    pen = pd.Series(0.0, index=x.index)
    pen += P_OVERHEAT_5D * np.clip((r5 - 10)/10, 0, 1)
    pen += P_RSI_OUT * ((rsi < 45) | (rsi > 65)).astype(float)
    pen += P_MACD_NEG * (slope < 0).astype(float)
    
    score = np.clip(base_score - pen, 0, 100)

    x["RR1"] = rr1; x["Now%"] = now_gap
    x["LDY_SCORE"] = score.round(1)
    x["ROUTE"] = (x.apply(route_tag, axis=1) if len(x) else "—")
    x["_GATE_OK"] = liquidity_gate(x["거래대금(억원)"], x["시장"]).fillna(False)
    x = x.sort_values("LDY_SCORE", ascending=False, na_position="last")
    x["LDY_RANK"] = range(1, len(x)+1)
    
    # [Fix] IntCastingNaNError 방지 (fillna(0) 추가)
    x["WHY"] = ("MOM+" + (100*W_MOM*mom_norm).round(0).fillna(0).astype(int).astype(str) + " "
                "LIQ+" + (100*W_LIQ*liq_norm).round(0).fillna(0).astype(int).astype(str) + " "
                "TEC+" + (100*W_TEC*tec_norm).round(0).fillna(0).astype(int).astype(str) + " "
                "PEN-" + pen.round(0).fillna(0).astype(int).astype(str))
    return x

# 6. 메인 실행 및 UI
try: df_raw = load_csv_url(RAW_URL); log_src(df_raw, "Remote")
except: 
    if os.path.exists(LOCAL_RAW): df_raw = load_csv_path(LOCAL_RAW); log_src(df_raw, "Local")
    else: st.error("❌ 데이터 없음"); st.stop()

df = normalize_cols(df_raw)
latest = df.copy()

scored = build_global_score(latest)
base = scored[(scored["EBS"] >= PASS_EBS) & (scored["_GATE_OK"])].copy()
if len(base) < 10: base = scored.head(20) # Fallback
top10 = base.head(10).copy()
top10["P_hit"] = (top10["LDY_SCORE"] / 100.0 * 0.8).clip(0, 1) * 100

# [사이드바 - 로그인]
with st.sidebar:
    st.divider(); st.header("🔐 로그인 (Login)")
    input_pw = st.text_input("비밀번호", type="password")
    ADMIN_KEY, MEMBER_KEY = "2022322", "240521"
    auth_status = "free"
    if input_pw == ADMIN_KEY: auth_status = "admin"; st.success("✅ 관리자")
    elif input_pw == MEMBER_KEY: auth_status = "member"; st.success("🎉 유료회원")
    else: 
        auth_status = "free"
        if input_pw: st.error("❌ 불일치")
        st.info("🔒 무료 (Top 3)")

# [섹션 1 - Market Radar]
kp_st, kp_df, kq_st, kq_df = get_market_status()
c1, c2 = st.columns(2)
c1.metric("KOSPI (MA20)", f"{kp_st}", f"{kp_df:.2f}%", delta_color="off" if kp_st=="Bull" else "inverse")
c2.metric("KOSDAQ (MA20)", f"{kq_st}", f"{kq_df:.2f}%", delta_color="off" if kq_st=="Bull" else "inverse")
if kp_st == "Bear" and kq_st == "Bear": st.warning("🚨 약세장 경보")

st.divider()

# [섹션 2 - 상세 차트]
st.subheader("🔭 종목 상세 차트 (60일)", anchor=False)
view_df = top10 if auth_status != "free" else top10.head(3)
opts = view_df.apply(lambda r: f"{r['종목명']} ({r['종목코드']}) - {r['ROUTE']}", axis=1).tolist()
sel = st.selectbox("종목 선택", opts, index=0 if opts else None)

if sel:
    code = sel.split("(")[1].split(")")[0]
    row = top10[top10["종목코드"]==code].iloc[0]
    c1, c2 = st.columns([2, 1])
    with c1:
        chart_df = get_stock_chart_data(code)
        if chart_df is not None:
            st.plotly_chart(plot_interactive_chart(chart_df, code, row['종목명'], row['추천매수가'], row['손절가'], row['추천매도가1'], row['추천매도가2']), use_container_width=True)
        else: st.info("차트 데이터 없음")
    with c2:
        st.markdown(f"### {row['종목명']}"); st.caption(f"Score: {row['LDY_SCORE']}")
        st.write(f"**전략:** `{row['ROUTE']}`"); st.write(f"**근거:** {row['근거']}")
        c_a, c_b = st.columns(2); c_a.metric("진입", f"{row['추천매수가']:,}"); c_b.metric("손절", f"{row['손절가']:,}", delta="Stop")
        c_c, c_d = st.columns(2); c_c.metric("목표1", f"{row['추천매도가1']:,}"); c_d.metric("RR", f"{row['RR1']:.2f}")

# [섹션 3 - 리스트 테이블]
st.subheader("📋 Daily Top 10 List", anchor=False)
with st.expander("❓ 용어 설명 (클릭)", expanded=False):
    st.markdown("""
    - **종합점수:** AI 투자 매력도 (100점 만점)
    - **MFI:** 수급강도 (60↑ 좋음)
    - **RR:** 손익비 (1.2↑ 권장)
    - **상세분석:** MOM(힘), LIQ(거래량), TEC(기술), PEN(감점)
    """)

if auth_status == "free": st.warning("🔒 무료 버전: Top 3만 공개")

# 데이터 포맷팅 (문자열 변환으로 콤마 찍기)
safe_view = view_df.copy().reset_index(drop=True)
safe_view["LDY_RANK"] = safe_view.index + 1

price_cols = ["종가","추천매수가","손절가","추천매도가1","추천매도가2","거래대금(억원)"]
for c in price_cols: 
    if c in safe_view.columns: 
        safe_view[c] = pd.to_numeric(safe_view[c], errors='coerce').fillna(0).apply(lambda x: f"{int(x):,}")

# 숫자형 유지 (그래프용)
for c in ["MFI14", "LDY_SCORE", "P_hit"]:
    if c in safe_view.columns: safe_view[c] = pd.to_numeric(safe_view[c], errors='coerce').fillna(0.0)

cols = ["LDY_RANK","통과","ROUTE","시장","종목명","종목코드","LDY_SCORE","P_hit","종가","추천매수가","손절가","추천매도가1","추천매도가2","RR1","MFI14","거래대금(억원)","WHY"]
cfg = {
    "LDY_RANK": st.column_config.NumberColumn("순위"),
    "LDY_SCORE": st.column_config.ProgressColumn("점수", format="%.1f", min_value=0, max_value=100),
    "P_hit": st.column_config.NumberColumn("확률", format="%.1f"),
    "종가": st.column_config.TextColumn("현재가"),
    "추천매수가": st.column_config.TextColumn("진입가"),
    "손절가": st.column_config.TextColumn("손절가"),
    "추천매도가1": st.column_config.TextColumn("목표1"),
    "추천매도가2": st.column_config.TextColumn("목표2"),
    "MFI14": st.column_config.NumberColumn("MFI", format="%.1f"),
    "WHY": st.column_config.TextColumn("분석", width="medium")
}
st.dataframe(safe_view[cols], hide_index=True, use_container_width=True, column_config=cfg)

# [섹션 4 - 다운로드]
if auth_status == "admin":
    csv = scored.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 전체 다운로드 (Admin)", csv, "ldy_rank.csv", "text/csv", key="admin_dl")
elif auth_status == "member":
    st.button("📥 다운로드 제한 (Member)", disabled=True, key="member_dl")
else:
    st.button("📥 다운로드 제한 (Free)", disabled=True, key="free_dl")
