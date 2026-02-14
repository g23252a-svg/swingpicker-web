import streamlit as st
import pandas as pd
import numpy as np
import os, glob, pickle
import plotly.express as px
from datetime import datetime

# [📍혈자리 1] 틱 유틸리티 & 폴백
try:
    from utils.price_utils import round_to_tick, add_tick
except Exception:
    def round_to_tick(p, method="nearest"): return int(round(float(p)))
    def add_tick(p, ticks=1): return int(float(p) + ticks)

st.set_page_config(page_title="🧪 LDY 전략 실험실 v3.6", layout="wide")

# -------------------------- 데이터 로딩 (기존 유지) --------------------------
@st.cache_data
def load_all_data():
    rec_files = sorted(glob.glob(os.path.join("data", "recommend_*.csv")))
    dfs = []
    for f in rec_files:
        try:
            date_str = os.path.basename(f).split("_")[1].split(".")[0]
            df = pd.read_csv(f, dtype={'종목코드': str})
            df['추천일'] = pd.to_datetime(date_str[:8], format="%Y%m%d")
            dfs.append(df)
        except: continue
    
    cache_files = sorted(glob.glob(os.path.join("data", "ohlcv_cache_*.pkl")), reverse=True)
    price_map = {}
    if cache_files:
        with open(cache_files[0], 'rb') as f: price_map = pickle.load(f)
            
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(), price_map

# -------------------------- [📍혈자리 3] v3.6 Sovereign-Ultimate 엔진 --------------------------
def run_simulation(df, price_map, hold_days, target_pct, stop_pct,
                   entry_slip_ticks=1, exit_slip_ticks=1, 
                   commission_pct=0.015, sell_tax_pct=0.18,
                   ambiguity_mode='conservative', fill_mode='execution'):
    results = []
    hold_days = max(1, int(hold_days))
    stop_pct, target_pct = max(0.1, float(stop_pct)), max(0.1, float(target_pct))
    
    progress_bar = st.progress(0)
    total = len(df)
    if total == 0: return pd.DataFrame()

    entry_fee_rate = commission_pct / 100.0
    exit_fee_rate  = (commission_pct + sell_tax_pct) / 100.0
    colmap = {"Open": "시가", "High": "고가", "Low": "저가", "Close": "종가"}

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 100 == 0: progress_bar.progress(min(idx / total, 1.0))
        code, signal_date = row["종목코드"], row["추천일"]
        if code not in price_map: continue

        ohlcv = price_map[code].sort_index()
        for en, ko in colmap.items():
            if ko not in ohlcv.columns and en in ohlcv.columns: ohlcv[ko] = ohlcv[en]
        
        start_idx = ohlcv.index.searchsorted(signal_date, side='right')
        future_data = ohlcv.iloc[start_idx : start_idx + hold_days + 1]
        if future_data.empty: continue

        raw_entry_p = float(future_data.iloc[0]["시가"])
        actual_entry_p = add_tick(raw_entry_p, entry_slip_ticks)
        total_entry_cost = actual_entry_p * (1 + entry_fee_rate)

        # Clamping
        if fill_mode == 'execution':
            stop_p, target_p = round_to_tick(actual_entry_p * (1 - stop_pct/100), "up"), round_to_tick(actual_entry_p * (1 + target_pct/100), "down")
        else:
            stop_p, target_p = round_to_tick(actual_entry_p * (1 - stop_pct/100), "nearest"), round_to_tick(actual_entry_p * (1 + target_pct/100), "nearest")

        if stop_p >= actual_entry_p: stop_p = add_tick(actual_entry_p, -1)
        if target_p <= actual_entry_p: target_p = add_tick(actual_entry_p, 1)

        opens, highs, lows, closes, dates = future_data["시가"].values.astype(np.float64), future_data["고가"].values.astype(np.float64), future_data["저가"].values.astype(np.float64), future_data["종가"].values.astype(np.float64), future_data.index
        exit_price, exit_date, status = closes[-1], dates[-1], "HOLD"

        for i in range(len(future_data)):
            o, h, l, curr_dt = opens[i], highs[i], lows[i], dates[i]
            if i > 0:
                o_t = round_to_tick(o, "nearest")
                if o_t <= stop_p: exit_price, exit_date, status = o_t, curr_dt, "GAP_STOP"; break
                if o_t >= target_p: exit_price, exit_date, status = o_t, curr_dt, "GAP_WIN"; break
            
            hit_stop, hit_win = l <= stop_p, h >= target_p
            if hit_stop and hit_win:
                exit_date = curr_dt
                if ambiguity_mode == 'conservative': exit_price, status = stop_p, "STOP"; break
                elif ambiguity_mode == 'optimistic': exit_price, status = target_p, "WIN"; break
                else: exit_price, status = round_to_tick((stop_p*0.6 + target_p*0.4), "nearest"), "NEUTRAL_TOUCH"; break
            elif hit_stop: exit_price, exit_date, status = stop_p, curr_dt, "STOP"; break
            elif hit_win: exit_price, exit_date, status = target_p, curr_dt, "WIN"; break

        actual_exit_p = add_tick(exit_price, -exit_slip_ticks)
        ret = (actual_exit_p * (1 - exit_fee_rate) - total_entry_cost) / total_entry_cost * 100.0

        results.append({"진입일": dates[0], "청산일": exit_date, "종목명": row.get("종목명", code), "수익률": ret, "상태": status, "진입가": int(actual_entry_p), "청산가": int(actual_exit_p)})

    progress_bar.empty()
    return pd.DataFrame(results)

# -------------------------- UI 구성 --------------------------
st.title("🧪 LDY 전략 실험실 Sovereign v3.6")
all_recs, price_map = load_all_data()

# [📍혈자리 2] 사이드바 정밀 제어판
with st.sidebar:
    st.header("🛠️ 전략 & 필터")
    min_score = st.slider("최소 점수", 0, 100, 80)
    st.markdown("---")
    st.header("💰 매매 규칙")
    hold_days = st.number_input("보유 기간", 1, 60, 10)
    target_pct = st.number_input("익절 (%)", 1.0, 50.0, 10.0)
    stop_pct = st.number_input("손절 (%)", 1.0, 30.0, 5.0)
    st.markdown("---")
    st.header("🛡️ 실전 비용")
    slip_ticks = st.slider("슬리피지(틱)", 0, 5, 1)
    tax_rate = st.number_input("세율(%)", 0.0, 0.5, 0.18)
    ambiguity = st.selectbox("동시터치", ["conservative", "optimistic", "neutral"])

# 데이터 필터링
filtered_df = all_recs[all_recs['TOTAL_SCORE'] >= min_score].copy()

if st.button("🚀 시뮬레이션 시작", type="primary"):
    res_df = run_simulation(filtered_df, price_map, hold_days, target_pct, stop_pct, 
                            entry_slip_ticks=slip_ticks, exit_slip_ticks=slip_ticks, sell_tax_pct=tax_rate, ambiguity_mode=ambiguity)
    
    if not res_df.empty:
        # [📍혈자리 4] 복리 & MDD 리포팅
        res_df = res_df.sort_values('진입일')
        res_df['cum_ret'] = (1 + res_df['수익률']/100).cumprod()
        res_df['drawdown'] = (res_df['cum_ret'] / res_df['cum_ret'].cummax() - 1) * 100
        
        win_rate = (res_df['수익률'] > 0).mean() * 100
        mdd = res_df['drawdown'].min()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("승률", f"{win_rate:.1f}%")
        c2.metric("최종 수익률", f"{(res_df['cum_ret'].iloc[-1]-1)*100:.2f}%")
        c3.metric("최대 낙폭(MDD)", f"{mdd:.2f}%")
        c4.metric("거래 횟수", f"{len(res_df)}회")

        st.plotly_chart(px.line(res_df, x='진입일', y='cum_ret', title='📈 자산 성장 곡선 (복리)'))
        st.dataframe(res_df.style.format({'수익률': '{:.2f}%', 'drawdown': '{:.2f}%'}))
