import streamlit as st
import pandas as pd
import numpy as np
import os, glob, pickle
import plotly.express as px
from datetime import datetime

# [📍혈자리 1] 틱 유틸리티 & 폴백
try:
    from price_utils import round_to_tick, add_tick
except Exception:
    import math as _math
    
    def _krx_tick_size(price):
        """KRX 호가단위 (fallback)"""
        if price < 2000: return 1
        if price < 5000: return 5
        if price < 20000: return 10
        if price < 50000: return 50
        if price < 200000: return 100
        if price < 500000: return 500
        return 1000

    def round_to_tick(price, method="nearest"):
        """호가 단위에 맞춘 가격 반올림 (fallback)"""
        if price is None or (isinstance(price, float) and _math.isnan(price)):
            return None
        p = float(price)
        if method == "down":
            p = int(_math.floor(p))
        elif method == "up":
            p = int(_math.ceil(p))
        else:
            p = int(round(p))
        t = _krx_tick_size(p)
        remainder = p % t
        if remainder == 0:
            return p
        if method == "down":
            return p - remainder
        elif method == "up":
            return p + (t - remainder)
        return p + (t - remainder) if remainder >= (t / 2) else p - remainder

    def add_tick(price, ticks=1):
        """틱 단위 이동 (fallback)"""
        if price is None or (isinstance(price, float) and _math.isnan(price)):
            return 0
        curr = float(round_to_tick(price, "nearest"))
        direction = 1 if ticks > 0 else -1
        for _ in range(abs(ticks)):
            lookup_p = curr if direction > 0 else curr - 1
            t = _krx_tick_size(max(0, lookup_p))
            curr += t * direction
        return int(curr)

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

# 데이터 필터링 — TOTAL_SCORE / FINAL_SCORE / LDY_SCORE 순으로 탐색
_score_col = None
for _c in ["TOTAL_SCORE", "FINAL_SCORE", "LDY_SCORE", "RANK_SCORE"]:
    if _c in all_recs.columns:
        _score_col = _c
        break

if _score_col:
    all_recs[_score_col] = pd.to_numeric(all_recs[_score_col], errors='coerce').fillna(0)
    filtered_df = all_recs[all_recs[_score_col] >= min_score].copy()
else:
    st.warning("⚠️ 점수 컬럼(TOTAL_SCORE 등)이 없어 전체 데이터를 사용합니다.")
    filtered_df = all_recs.copy()

if st.button("🚀 시뮬레이션 시작", type="primary"):
    res_df = run_simulation(filtered_df, price_map, hold_days, target_pct, stop_pct, 
                            entry_slip_ticks=slip_ticks, exit_slip_ticks=slip_ticks, sell_tax_pct=tax_rate, ambiguity_mode=ambiguity)
    
    if not res_df.empty:
        # [📍혈자리 4] 복리 & 일별 MDD 리포팅
        res_df = res_df.sort_values('진입일')
        res_df['cum_ret'] = (1 + res_df['수익률']/100).cumprod()
        
        # ── 일별 MDD 계산 (트레이드 단위가 아닌 포트폴리오 일간 가치 기준) ──
        daily_equity = {}
        capital = 1.0  # 정규화된 시작 자본
        trades = res_df.to_dict('records')
        
        for t in trades:
            entry_dt = pd.Timestamp(t['진입일'])
            exit_dt = pd.Timestamp(t['청산일'])
            trade_ret = t['수익률'] / 100.0
            code = None
            
            # price_map에서 일간 종가 추적 (가능한 경우)
            matched_name = t.get('종목명', '')
            code_match = None
            for c, ohlcv in price_map.items():
                if ohlcv.index.min() <= entry_dt:
                    mask = (ohlcv.index >= entry_dt) & (ohlcv.index <= exit_dt)
                    if mask.any():
                        code_match = c
                        break
            
            if code_match and code_match in price_map:
                ohlcv = price_map[code_match]
                close_col = '종가' if '종가' in ohlcv.columns else 'Close'
                if close_col in ohlcv.columns:
                    mask = (ohlcv.index >= entry_dt) & (ohlcv.index <= exit_dt)
                    daily_prices = ohlcv.loc[mask, close_col]
                    if len(daily_prices) > 1:
                        entry_p = float(daily_prices.iloc[0])
                        if entry_p > 0:
                            for dt, px in daily_prices.items():
                                intra_ret = (float(px) - entry_p) / entry_p
                                day_key = pd.Timestamp(dt).normalize()
                                daily_equity[day_key] = daily_equity.get(day_key, capital) * (1 + intra_ret / max(len(daily_prices), 1))
            
            # 트레이드 완료 → 자본 갱신
            capital *= (1 + trade_ret)
            day_key = exit_dt.normalize()
            daily_equity[day_key] = capital
        
        # 일별 MDD 계산
        if daily_equity:
            eq_series = pd.Series(daily_equity).sort_index()
            eq_peak = eq_series.cummax()
            daily_dd = ((eq_series - eq_peak) / eq_peak * 100)
            mdd = daily_dd.min()
        else:
            # fallback: 트레이드 기준 MDD
            res_df['drawdown'] = (res_df['cum_ret'] / res_df['cum_ret'].cummax() - 1) * 100
            mdd = res_df['drawdown'].min()
        
        win_rate = (res_df['수익률'] > 0).mean() * 100
        avg_win = res_df.loc[res_df['수익률'] > 0, '수익률'].mean() if (res_df['수익률'] > 0).any() else 0
        avg_loss = res_df.loc[res_df['수익률'] <= 0, '수익률'].mean() if (res_df['수익률'] <= 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("승률", f"{win_rate:.1f}%")
        c2.metric("최종 수익률", f"{(res_df['cum_ret'].iloc[-1]-1)*100:.2f}%")
        c3.metric("최대 낙폭(MDD)", f"{mdd:.2f}%")
        c4.metric("손익비", f"{profit_factor:.2f}")
        c5.metric("거래 횟수", f"{len(res_df)}회")

        # 자산 성장 곡선
        st.plotly_chart(px.line(res_df, x='진입일', y='cum_ret', title='📈 자산 성장 곡선 (복리)'), use_container_width=True)
        
        # 일별 낙폭 차트
        if daily_equity:
            dd_df = pd.DataFrame({'날짜': daily_dd.index, '낙폭(%)': daily_dd.values})
            fig_dd = px.area(dd_df, x='날짜', y='낙폭(%)', title='📉 일별 Drawdown')
            fig_dd.update_traces(fillcolor='rgba(255,0,0,0.15)', line_color='red')
            st.plotly_chart(fig_dd, use_container_width=True)
        
        st.dataframe(res_df.style.format({'수익률': '{:.2f}%'}), use_container_width=True)
