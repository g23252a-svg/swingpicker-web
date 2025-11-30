# -*- coding: utf-8 -*-
"""
LDY Pro Trader v5.6 (Final Stable Fix)
- Fix: IndexError on stock selection resolved (Using Index Match)
- Features: Smart Filters, Excel Portfolio, Visual Risk Bar, Open Beta
"""
import os, io, math, json, requests, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# 1. 라이브러리 로드
try: import FinanceDataReader as fdr; FDR_OK = True
except: FDR_OK = False
try: from pykrx import stock; PYKRX_OK = True
except: PYKRX_OK = False

# 2. 페이지 설정
st.set_page_config(page_title="LDY Pro Trader v5.6", layout="wide", page_icon="💎")
st.title("🏆 LDY Pro Trader v5.6 (Open Beta)")
st.caption("Free for Everyone: AI Quant Analysis & Portfolio Manager")

# 3. 상수 및 설정 (전역 변수)
RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
PORTFOLIO_FILE = "my_portfolio.json"

PASS_EBS = 4
MIN_TURN_KOSPI, MIN_TURN_KOSDAQ, MIN_TURN_DEFAULT = 200.0, 100.0, 100.0
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = 0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = 6.0, 6.0, 4.0
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 4.0, 2.0

# 4. 데이터 로딩 함수
@st.cache_data(ttl=600)
def load_csv_url(url):
    r = requests.get(url, timeout=30); r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=600)
def load_csv_path(path): return pd.read_csv(path, encoding="utf-8")

def log_src(df, src): st.toast(f"Data Loaded: {src} ({len(df)} rows)", icon="✅")

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return pd.DataFrame(json.load(f))
        except: pass
    return pd.DataFrame([{"종목명": "삼성전자", "평단가": 70000, "수량": 10}])

def save_portfolio(df):
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=4)
        return True
    except: return False

# 헬퍼 함수들
@st.cache_data(ttl=3600)
def get_code_map():
    if FDR_OK:
        try:
            df = fdr.StockListing('KRX')
            return dict(zip(df['Name'], df['Code'].astype(str).str.zfill(6)))
        except: pass
    return {}

def find_code_by_name(name_or_code, code_map):
    s = str(name_or_code).strip()
    if s.isdigit(): return s.zfill(6)
    return code_map.get(s, None)

def get_last_business_date(d=datetime.now()):
    d = d.date()
    if d.weekday() == 5: d -= timedelta(days=1)
    elif d.weekday() == 6: d -= timedelta(days=2)
    return d.strftime("%Y%m%d")

@st.cache_data(ttl=3600)
def get_market_status():
    kp_stat, kp_diff = "Unknown", 0.0
    kq_stat, kq_diff = "Unknown", 0.0
    if not FDR_OK: return kp_stat, kp_diff, kq_stat, kq_diff
    def _check(ticker):
        try:
            df = fdr.DataReader(ticker)
            if df is None or df.empty: return "Error", 0.0
            df = df.tail(60)
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            curr = df['Close'].iloc[-1]
            if pd.isna(ma20) or ma20 == 0: return "Unknown", 0.0
            return "Bull" if ((curr - ma20)/ma20) > 0 else "Bear", ((curr-ma20)/ma20)*100
        except: return "Error", 0.0
    kp_stat, kp_diff = _check('KS11')
    kq_stat, kq_diff = _check('KQ11')
    return kp_stat, kp_diff, kq_stat, kq_diff

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    if not FDR_OK: return 50, "Neutral"
    try:
        df = fdr.DataReader('KS11')
        if df.empty: return 50, "Neutral"
        delta = df['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0; down[down > 0] = 0
        rs = up.rolling(14).mean() / down.abs().rolling(14).mean()
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        ma20 = df['Close'].rolling(20).mean()
        disparity = (df['Close'] / ma20 * 100).iloc[-1]
        score = current_rsi
        if disparity > 105: score += 10
        elif disparity < 95: score -= 10
        score = max(0, min(100, score))
        if score >= 75: status = "Extreme Greed (매도)"
        elif score >= 60: status = "Greed (과열)"
        elif score <= 25: status = "Extreme Fear (매수)"
        elif score <= 40: status = "Fear (침체)"
        else: status = "Neutral (중립)"
        return score, status
    except: return 50, "Error"

def plot_fear_greed_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta", value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Market Fear & Greed", 'font': {'size': 20}},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "blue"}},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "rgba(0,0,0,0)"}, 
                 'steps': [{'range': [0, 25], 'color': '#4D96FF'}, {'range': [25, 45], 'color': '#87CEEB'},
                           {'range': [45, 55], 'color': '#D3D3D3'}, {'range': [55, 75], 'color': '#FFB347'},
                           {'range': [75, 100], 'color': '#FF6B6B'}],
                 'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': score}}
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def plot_sector_treemap(df):
    if '업종' not in df.columns: return None
    df_map = df.copy()
    df_map['업종'] = df_map['업종'].fillna('기타')
    df_map = df_map[df_map['업종'] != '기타']
    if df_map.empty: return None
    fig = px.treemap(
        df_map, path=['업종', '종목명'], values='거래대금(억원)', color='LDY_SCORE',
        color_continuous_scale='RdYlGn', title="<b>🔥 시장 주도 섹터 지도</b>"
    )
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=350)
    return fig

def calculate_supertrend(df, period=10, multiplier=3):
    high = df['High']; low = df['Low']; close = df['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)
    for i in range(period, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]: final_upper.iloc[i] = basic_upper.iloc[i]
        else: final_upper.iloc[i] = final_upper.iloc[i-1]
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]: final_lower.iloc[i] = basic_lower.iloc[i]
        else: final_lower.iloc[i] = final_lower.iloc[i-1]
        if trend.iloc[i-1] == 1:
            if close.iloc[i] < final_lower.iloc[i-1]: trend.iloc[i] = -1
            else: trend.iloc[i] = 1
        else:
            if close.iloc[i] > final_upper.iloc[i-1]: trend.iloc[i] = 1
            else: trend.iloc[i] = -1
        supertrend.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]
    df['SuperTrend'] = supertrend; df['Trend'] = trend
    return df

@st.cache_data(ttl=600)
def get_stock_chart_data(code):
    if not FDR_OK: return None
    try:
        code_str = str(code).zfill(6)
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        df = fdr.DataReader(code_str, start_date)
        if df is None or df.empty: return None
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df = calculate_supertrend(df)
        return df.tail(80)
    except: return None

def plot_radar_chart(row):
    stats = {
        "모멘텀": min(100, (row.get("ret_5d_%", 0) + 5) * 10),
        "수급(MFI)": row.get("MFI14", 50),
        "가성비(RR)": min(100, row.get("RR1", 1) * 50),
        "안전성": 100 - (row.get("이격도", 0) * 2),
        "종합점수": row.get("LDY_SCORE", 0)
    }
    values = [max(0, min(100, v)) for v in stats.values()]
    categories = list(stats.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=row['종목명'], line_color='#00CC96', fillcolor='rgba(0, 204, 150, 0.3)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], color="gray"), bgcolor='rgba(0,0,0,0)'), showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_interactive_chart(df, code, name, entry, stop, target1, target2):
    if df is None or df.empty: return go.Figure()
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="주가", increasing_line_color='#ef5350', decreasing_line_color='#2979ff'
    )])
    up_trend = df[df['Trend'] == 1]; down_trend = df[df['Trend'] == -1]
    fig.add_trace(go.Scatter(x=up_trend.index, y=up_trend['SuperTrend'], mode='markers', marker=dict(color='green', size=2), name='상승추세'))
    fig.add_trace(go.Scatter(x=down_trend.index, y=down_trend['SuperTrend'], mode='markers', marker=dict(color='red', size=2), name='하락추세'))
    lines = [(entry, "🔵진입", "blue"), (stop, "🔴손절", "red"), (target1, "🟢목표1", "green")]
    for val, label, color in lines:
        if pd.notna(val) and val > 0:
            fig.add_hline(y=val, line_dash="dash", line_color=color, annotation_text=label)
    if 'MA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange'), name='20일선'))
    if 'MA60' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='purple'), name='60일선'))
    fig.update_layout(title=f"{name} ({code})", xaxis_rangeslider_visible=False, height=500, margin=dict(l=20,r=20,t=40,b=20), hovermode="x unified")
    return fig

def plot_risk_reward_bar(buy, stop, target1, target2):
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["Price"], x=[buy - stop], orientation='h', name='Risk', marker=dict(color='red'), text="Risk"))
    fig.add_trace(go.Bar(y=["Price"], x=[target1 - buy], orientation='h', name='Reward 1', marker=dict(color='lightgreen'), text="Reward 1"))
    fig.add_trace(go.Bar(y=["Price"], x=[target2 - target1], orientation='h', name='Reward 2', marker=dict(color='green'), text="Reward 2"))
    fig.update_layout(barmode='stack', showlegend=False, height=60, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# 5. 유틸
def z6(x): return str(x).zfill(6) if str(x).isdigit() else str(x)
def nz_num(s): return pd.to_numeric(s, errors="coerce")
def ensure_turnover(df):
    if "거래대금(억원)" not in df.columns and "거래대금(원)" in df.columns:
        df["거래대금(억원)"] = (nz_num(df["거래대금(원)"])/1e8).round(2)
    return df
def normalize_cols(df): return ensure_turnover(df)
def apply_names(df): return df

# [중요] 유동성 게이트 (전역 상수 사용)
def liquidity_gate(x_turn, market):
    min_map = {"KOSPI": MIN_TURN_KOSPI, "KOSDAQ": MIN_TURN_KOSDAQ}
    return nz_num(x_turn) >= market.map(min_map).fillna(MIN_TURN_DEFAULT)

def build_global_score(lat):
    x = lat.copy()
    req = ["종가","추천매수가","손절가","추천매도가1","거래대금(억원)","RSI14","MACD_Slope","거래강도","이격도","ret_5d_%","ret_10d_%","EBS","MACD_Hist","MFI14"]
    for c in req:
        if c not in x.columns: x[c] = np.nan

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

    def cap_q(s, q=90, f=1.0): return float(max(np.nanpercentile(nz_num(s), q), f))
    def pct_norm(s, q=90, f=1.0): return np.clip(nz_num(s).clip(lower=0) / cap_q(s, q, f), 0, 1)
    def inv_dist_norm(dist, cap): return np.clip(1 - (nz_num(dist)/cap), 0, 1)

    rr_norm = pct_norm(rr1)
    t1_norm = np.clip(t1_room / cap_q(t1_room, 90, 5.0), 0, 1)
    sl_norm = np.clip(sl_room / cap_q(sl_room, 90, 3.0), 0, 1)
    near_norm = inv_dist_norm(now_gap, cap_q(now_gap, 75, 1.0))
    
    ers_bits = (ebs>=PASS_EBS).astype(int) + (slope>0).astype(int) + ((rsi>=45)&(rsi<=65)).astype(int)
    ers_norm = np.clip(ers_bits/3.0, 0, 1)
    slope_pos_norm = pct_norm(slope)
    mom_norm = np.clip(0.5*ers_norm + 0.3*slope_pos_norm, 0, 1)

    if turn.notna().any():
        lo, hi = np.nanpercentile(turn, 30), np.nanpercentile(turn, 90)
        liq_norm = np.clip((turn - lo) / max(hi-lo, 1e-9), 0, 1)
    else: liq_norm = 0.0

    vol_sweet = (1 - np.minimum((volz - 1).abs()/3, 1)).clip(0,1)
    kairi_norm = (1 - np.minimum(kairi.abs()/cap_q(kairi.abs(), 80, 3.0), 1)).clip(0,1)
    tec_norm = np.clip(0.6*vol_sweet + 0.4*kairi_norm, 0, 1)

    base_score = (100*W_RR*rr_norm) + (100*W_T1*t1_norm) + (100*W_SL*sl_norm) + \
                 (100*W_NEAR*near_norm) + (100*W_MOM*mom_norm) + (100*W_LIQ*liq_norm) + (100*W_TEC*tec_norm)
    
    pen = pd.Series(0.0, index=x.index)
    pen += P_OVERHEAT_5D * np.clip((r5 - 10)/10, 0, 1)
    pen += P_RSI_OUT * ((rsi < 45) | (rsi > 65)).astype(float)
    pen += P_MACD_NEG * (slope < 0).astype(float)
    
    score = np.clip(base_score - pen, 0, 100)
    x["RR1"] = rr1; x["Now%"] = now_gap
    x["LDY_SCORE"] = score.round(1)
    x["_GATE_OK"] = liquidity_gate(x["거래대금(억원)"], x["시장"]).fillna(False)
    x = x.sort_values("LDY_SCORE", ascending=False, na_position="last")
    x["LDY_RANK"] = range(1, len(x)+1)
    
    if "AI_COMMENT" in x.columns:
        x["WHY"] = x["AI_COMMENT"]
    else:
        x["WHY"] = ("MOM+" + (100*W_MOM*mom_norm).round(0).fillna(0).astype(int).astype(str) + " LIQ+" + (100*W_LIQ*liq_norm).round(0).fillna(0).astype(int).astype(str) + " TEC+" + (100*W_TEC*tec_norm).round(0).fillna(0).astype(int).astype(str) + " PEN-" + pen.round(0).fillna(0).astype(int).astype(str))
    return x

def route_tag(row):
    r5 = row.get("ret_5d_%", 0)
    slope = row.get("MACD_Slope", 0)
    if r5 >= 3 and slope > 0: return "🔼 BRK (돌파)"
    return "↩️ PULL (눌림)"

# 6. 메인 실행
try: df_raw = load_csv_url(RAW_URL); log_src(df_raw, "Remote")
except: 
    if os.path.exists(LOCAL_RAW): df_raw = load_csv_path(LOCAL_RAW); log_src(df_raw, "Local")
    else: st.error("❌ 데이터 없음"); st.stop()

df = normalize_cols(df_raw)
latest = df.copy()
scored = build_global_score(latest)
base = scored[(scored["EBS"] >= PASS_EBS) & (scored["_GATE_OK"])].copy()
if len(base) < 10: base = scored.head(20)
top10 = base.head(10).copy()
top10["P_hit"] = (top10["LDY_SCORE"] / 100.0 * 0.8).clip(0, 1) * 100

# [사이드바 - 엑셀형 포트폴리오]
with st.sidebar:
    st.header("💼 내 자산 (Excel Editor)")
    loaded_df = load_portfolio()
    edited_df = st.data_editor(loaded_df, num_rows="dynamic", key="pf_editor", use_container_width=True)
    if st.button("💾 포트폴리오 저장"):
        save_portfolio(edited_df); st.success("저장 완료!"); st.rerun()

# [메인 화면]
kp_stat, kp_diff, kq_stat, kq_diff = get_market_status()
st.plotly_chart(plot_fear_greed_gauge(get_fear_greed_index()[0]), use_container_width=True)
c1, c2 = st.columns(2)
c1.metric("KOSPI", f"{kp_stat}", f"{kp_diff:.2f}%")
c2.metric("KOSDAQ", f"{kq_stat}", f"{kq_diff:.2f}%")

st.divider()

# [포트폴리오 진단]
if not edited_df.empty:
    st.subheader("🧐 내 포트폴리오 진단", anchor=False)
    try:
        code_map = get_code_map()
        pf_list = []
        total_buy = 0; total_eval = 0
        
        for idx, row in edited_df.iterrows():
            name_input = str(row["종목명"])
            avg = float(row["평단가"]); qty = int(row["수량"])
            
            code = find_code_by_name(name_input, code_map)
            if not code: code = name_input 
            
            try:
                df_rt = fdr.DataReader(code)
                cur_price = int(df_rt.iloc[-1]['Close'])
                rate = (cur_price - avg) / avg * 100
                signal = "🟢" if rate > 0 else ("🔴" if rate < -3 else "🟡")
            except: cur_price = 0; rate = 0; signal = "❓"
            
            val = cur_price * qty
            pf_list.append({"종목": name_input, "상태": signal, "현재가": f"{cur_price:,}", "수익률": f"{rate:.2f}%", "평가액": f"{val:,}"})
            total_buy += avg * qty; total_eval += val
            
        c1, c2, c3 = st.columns(3)
        tot_rate = (total_eval - total_buy) / total_buy * 100 if total_buy > 0 else 0
        c1.metric("총 매수", f"{int(total_buy):,}원")
        c2.metric("총 평가", f"{int(total_eval):,}원")
        c3.metric("총 수익", f"{tot_rate:.2f}%", f"{int(total_eval-total_buy):,}원")
        st.dataframe(pd.DataFrame(pf_list), use_container_width=True, hide_index=True)
    except Exception as e: st.error(f"분석 실패: {e}")

st.divider()

# [스마트 필터]
st.subheader("🔭 종목 상세 분석 (Smart Filter)", anchor=False)
routes = ["전체"] + list(top10['ROUTE'].unique())
sectors = ["전체"] + list(top10['업종'].unique()) if '업종' in top10.columns else ["전체"]

c1, c2 = st.columns(2)
f_route = c1.selectbox("전략", routes)
f_sector = c2.selectbox("업종", sectors)

filtered = top10.copy()
if f_route != "전체": filtered = filtered[filtered['ROUTE'] == f_route]
if f_sector != "전체": filtered = filtered[filtered['업종'] == f_sector]

view_df = filtered if not filtered.empty else pd.DataFrame()

if not view_df.empty:
    opts = view_df.apply(lambda r: f"{r['종목명']} ({r['종목코드']})", axis=1).tolist()
    # [Fix] IndexError 완벽 해결 (인덱스 매칭)
    sel = st.selectbox("종목 선택", opts)
    
    if sel:
        sel_idx = opts.index(sel)
        row = view_df.iloc[sel_idx]
        code = row['종목코드']
        
        c1, c2 = st.columns([2, 1])
        with c1:
            chart_df = get_stock_chart_data(code)
            if chart_df is not None: st.plotly_chart(plot_interactive_chart(chart_df, code, row['종목명'], row['추천매수가'], row['손절가'], row['추천매도가1'], row['추천매도가2']), use_container_width=True)
        with c2:
            st.markdown(f"### {row['종목명']}")
            st.plotly_chart(plot_radar_chart(row), use_container_width=True)
            st.info(f"전략: {row['ROUTE']}")
            st.plotly_chart(plot_risk_reward_bar(row['추천매수가'], row['손절가'], row['추천매도가1'], row['추천매도가2']), use_container_width=True)

# [리스트]
st.subheader("📋 Daily Top 10 List", anchor=False)
cols = ["LDY_RANK","ROUTE","종목명","종목코드","LDY_SCORE","종가","추천매수가","손절가","추천매도가1"]
st.dataframe(view_df[cols], hide_index=True, use_container_width=True)

csv = latest.to_csv(index=False).encode('utf-8-sig')
st.download_button("📥 전체 다운로드", csv, "ldy_rank.csv", "text/csv")
