# -*- coding: utf-8 -*-
"""
LDY Pro Trader Collector v5.4 (Ranking Sync)
- Update: Added LDY Scoring Logic to Collector for Telegram Sync
"""

import os
import time
import math
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from pykrx import stock
from tqdm import tqdm
import FinanceDataReader as fdr

# [Secret 로드]
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_ID = os.environ.get("TG_ID")

# ------------------------------- 설정 -------------------------------
KST = timezone(timedelta(hours=9))
LOOKBACK_DAYS = 250          
TOP_N = 600                 
MIN_TURNOVER_EOK = 50       
MIN_MCAP_EOK = 1000         
RSI_LOW, RSI_HIGH = 45, 65  
PASS_SCORE = 4              
SLEEP_SEC = 0.05            
OUT_DIR = "data"
UTF8 = "utf-8-sig"

# [LDY Score 가중치 설정] (대시보드와 동일하게 맞춤)
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = 0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = 6.0, 6.0, 4.0
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 4.0, 2.0

# ------------------------------- 유틸 -------------------------------
def log(msg: str): print(f"[{datetime.now(KST)}] {msg}")
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

# 지표 계산 함수들
def calc_rsi(close, period=14):
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(period).mean() / down.replace(0, np.nan).rolling(period).mean()
    return 100 - 100 / (1 + rs)

def calc_atr(high, low, close, period=14):
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_mfi(high, low, close, vol, period=14):
    tp = (high + low + close) / 3
    rmf = tp * vol
    pos = np.where(tp.diff() > 0, rmf, 0)
    neg = np.where(tp.diff() < 0, rmf, 0)
    return 100 - (100 / (1 + pd.Series(pos).rolling(period).sum() / pd.Series(neg).rolling(period).sum().replace(0, 1)))

def round_to_tick(price):
    if price < 2000: t = 1
    elif price < 5000: t = 5
    elif price < 20000: t = 10
    elif price < 50000: t = 50
    elif price < 200000: t = 100
    elif price < 500000: t = 500
    else: t = 1000
    return int(round(price / t) * t)

def _safe_sum(x): return pd.to_numeric(x, errors="coerce").fillna(0).sum()
def nz_num(s): return pd.to_numeric(s, errors="coerce")

# ------------------------------- 스코어링 로직 (이식됨) -------------------------------
def cap_q(s, q=90, floor=1.0):
    c = np.nanpercentile(nz_num(s), q)
    return float(max(c, floor)) if np.isfinite(c) else floor

def pct_norm_pos(s, q=90, floor=1.0):
    s = nz_num(s).clip(lower=0)
    return np.clip(s / cap_q(s, q, floor), 0, 1)

def inv_dist_norm(dist, cap): return np.clip(1 - (nz_num(dist)/cap), 0, 1)

def build_global_score(lat):
    x = lat.copy()
    # 컬럼 매핑
    close, entry, stop, t1 = nz_num(x["종가"]), nz_num(x["추천매수가"]), nz_num(x["손절가"]), nz_num(x["추천매도가1"])
    turn, rsi, slope, volz = nz_num(x["거래대금(억원)"]), nz_num(x["RSI14"]), nz_num(x["MACD_Slope"]), nz_num(x["거래강도"])
    kairi, r5, ebs = nz_num(x["이격도"]), nz_num(x["ret_5d_%"]), nz_num(x["EBS"]).fillna(0)

    rr_den = (entry - stop)
    rr1 = ((t1 - entry) / rr_den.replace(0, np.nan))
    now_gap = ((close - entry).abs() / entry * 100)
    t1_room = ((t1 - close) / close * 100)
    sl_room = ((close - stop) / close * 100)

    rr_norm = pct_norm_pos(rr1, q=90, floor=1.0).fillna(0)
    t1_norm = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1).fillna(0)
    sl_norm = np.clip(sl_room / cap_q(sl_room, q=90, floor=3.0), 0, 1).fillna(0)
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
    
    # 전략 태그 생성
    conditions = [
        (r5 >= 3) & (slope > 0),
        (rsi >= 40) & (rsi <= 60),
        (rsi <= 40)
    ]
    choices = ["🔼 BRK (돌파)", "↩️ PULL (눌림)", "🔁 MR (반전)"]
    x["ROUTE"] = np.select(conditions, choices, default="—")
    
    # WHY 문자열
    x["WHY"] = ("MOM+" + (100*W_MOM*mom_norm).round(0).fillna(0).astype(int).astype(str) + " " +
                "LIQ+" + (100*W_LIQ*liq_norm).round(0).fillna(0).astype(int).astype(str) + " " +
                "TEC+" + (100*W_TEC*tec_norm).round(0).fillna(0).astype(int).astype(str) + " " +
                "PEN-" + pen.round(0).fillna(0).astype(int).astype(str))
    return x

# ------------------------------- 데이터 수집 로직 -------------------------------
def main():
    log("🚀 LDY Collector v5.4 시작...")
    mcap_map, mcap_ymd = build_mcap_map()
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일: {trade_ymd}")

    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()
    kospi_set, kosdaq_set = get_market_sets(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)
    sector_map = get_sector_map()

    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS * 2 + 60)
    start_s, end_s = start_dt.strftime("%Y%m%d"), trade_ymd

    rows = []
    for t in tqdm(tickers, desc="Analyzing", unit="stock"):
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty: continue
            ohlcv = ohlcv.reset_index().rename(columns={"index": "날짜"})
            ohlcv["날짜"] = pd.to_datetime(ohlcv["날짜"])
            ohlcv = ohlcv.tail(LOOKBACK_DAYS)
            if len(ohlcv) < 120: continue

            c = ohlcv["종가"].astype(float)
            h = ohlcv["고가"].astype(float)
            l = ohlcv["저가"].astype(float)
            v = ohlcv["거래량"].astype(float)

            ma20 = c.rolling(20).mean(); ma60 = c.rolling(60).mean(); ma120 = c.rolling(120).mean()
            atr14 = calc_atr(h, l, c, 14); rsi14 = calc_rsi(c, 14); mfi14 = calc_mfi(h, l, c, v, 14)
            
            ema12 = ema(c, 12); ema26 = ema(c, 26)
            macd_line = ema12 - ema26; macd_sig = ema(macd_line, 9)
            macd_hist = macd_line - macd_sig; macd_slope= macd_hist.diff()

            vol_z = v / (v.rolling(20).mean()); disp = (c / ma20 - 1.0) * 100
            std20 = c.rolling(20).std()
            bb_w = ((ma20 + std20*2) - (ma20 - std20*2)) / ma20
            bb_w_avg = bb_w.rolling(20).mean()

            last = ohlcv.iloc[-1]
            cur_c = float(last["종가"])
            
            v_rsi = float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else 50.0
            v_mfi = float(mfi14.iloc[-1]) if not np.isnan(mfi14.iloc[-1]) else 50.0
            v_slp = float(macd_slope.iloc[-1]) if not np.isnan(macd_slope.iloc[-1]) else 0.0
            v_hist = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0.0
            v_disp = float(disp.iloc[-1]) if not np.isnan(disp.iloc[-1]) else 0.0
            v_volz = float(vol_z.iloc[-1]) if not np.isnan(vol_z.iloc[-1]) else 0.0
            v_m20 = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else 0.0
            v_m120 = float(ma120.iloc[-1]) if not np.isnan(ma120.iloc[-1]) else 0.0
            v_bw = float(bb_w.iloc[-1]) if not np.isnan(bb_w.iloc[-1]) else 0.0
            v_bw_avg = float(bb_w_avg.iloc[-1]) if not np.isnan(bb_w_avg.iloc[-1]) else 0.0
            
            ret5 = (c.pct_change(5).iloc[-1]*100) if len(c)>5 else 0.0
            ret10 = (c.pct_change(10).iloc[-1]*100) if len(c)>10 else 0.0

            mkt = "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타")
            name = name_map.get(str(t).zfill(6), "") or stock.get_market_ticker_name(t)
            sector = sector_map.get(str(t).zfill(6), "기타")
            tv_eok = float(top_df.loc[top_df["종목코드"]==t, "거래대금(원)"].values[0])/1e8
            mcap_eok = get_mcap_eok_from_map(mcap_map, t)

            if tv_eok < MIN_TURNOVER_EOK or (not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK): continue

            score = 0; reason = []
            if RSI_LOW <= v_rsi <= RSI_HIGH: score += 1; reason.append("RSI적정")
            if v_slp > 0: score += 1; reason.append("MACD상승")
            if -1.0 <= v_disp <= 5.0: score += 1; reason.append("20선근접")
            if v_volz > 1.2: score += 1; reason.append("거래량↑")
            if v_m20 > v_m60: score += 1; reason.append("정배열(단)")
            if cur_c > v_m120: score += 1; reason.append("장기추세(120↑)")
            else: score -= 1
            if v_mfi > 60: score += 1; reason.append("자금유입(MFI)")
            if v_bw < v_bw_avg * 0.8: score += 1; reason.append("에너지응축(Sqz)")
            if v_hist > 0: score += 1; reason.append("MACD>Sig")
            if ret5 < 12: score += 1; reason.append("과열X")
            
            # Entry/Target Logic
            try: atr = float(atr14.iloc[-1])
            except: atr = 0.0
            if np.isnan(atr) or atr <= 0: atr = cur_c * 0.03

            if v_m20 > 0 and cur_c > v_m20: buy = min(cur_c, v_m20 * 1.03)
            else: buy = cur_c

            raw_stop = buy - (2.0 * atr)
            max_loss_limit = buy * 0.93
            if raw_stop < max_loss_limit: raw_stop = max_loss_limit
            if raw_stop >= buy * 0.97: raw_stop = buy * 0.97
            stop = raw_stop

            risk = buy - stop
            if score >= 8: rr1, rr2 = 2.0, 4.0
            elif score >= 6: rr1, rr2 = 1.5, 3.0
            else: rr1, rr2 = 1.2, 2.5
            
            tgt1 = buy + (risk * rr1); tgt2 = buy + (risk * rr2)
            buy = round_to_tick(buy); stop = round_to_tick(stop); tgt1 = round_to_tick(tgt1); tgt2 = round_to_tick(tgt2)

            rows.append({
                "시장": mkt, "종목명": name, "종목코드": str(t).zfill(6), "업종": sector,
                "종가": int(cur_c), "거래대금(억원)": round(tv_eok, 2),
                "시가총액(억원)": None if np.isnan(mcap_eok) else round(mcap_eok, 1),
                "RSI14": round(v_rsi, 1), "MFI14": round(v_mfi, 1),
                "이격도": round(v_disp, 2), "MACD_Hist": round(v_hist, 4), "MACD_Slope": round(v_slp, 5),
                "거래강도": round(v_volz, 2), "ret_5d_%": round(ret5, 2), "ret_10d_%": round(ret10, 2),
                "EBS": int(score), "통과": "★" if score >= PASS_SCORE else "",
                "근거": ", ".join(reason),
                "추천매수가": buy, "손절가": stop, "추천매도가1": tgt1, "추천매도가2": tgt2
            })
        except Exception: pass
        time.sleep(SLEEP_SEC)

    if not rows: raise RuntimeError("수집 결과 없음")
    
    # [Fix] LDY Score 계산 후 정렬
    df_raw = pd.DataFrame(rows)
    df_scored = build_global_score(df_raw) # 점수 계산 적용
    
    # 이제 EBS가 아니라 LDY_SCORE 기준으로 정렬합니다.
    df_out = df_scored.sort_values(["LDY_SCORE", "거래대금(억원)"], ascending=[False, False])
    
    ensure_dir(OUT_DIR)
    path_day = os.path.join(OUT_DIR, f"recommend_{trade_ymd}.csv")
    path_lat = os.path.join(OUT_DIR, "recommend_latest.csv")
    df_out.to_csv(path_day, index=False, encoding=UTF8)
    df_out.to_csv(path_lat, index=False, encoding=UTF8)
    log(f"💾 저장완료: {path_lat} ({len(df_out)}건)")
    
    send_telegram_auto(df_out)

# ------------------------------- 헬퍼 함수 (아래) -------------------------------
def get_sector_map():
    try:
        df = fdr.StockListing('KRX')
        col = next((c for c in ['Sector', '업종', 'Industry'] if c in df.columns), None)
        if col:
            df['Code'] = df['Code'].astype(str).str.zfill(6)
            return dict(zip(df['Code'], df[col]))
    except: pass
    return {}
def get_market_sets(d):
    try: return set(stock.get_market_ticker_list(d, market='KOSPI')), set(stock.get_market_ticker_list(d, market='KOSDAQ'))
    except: return set(), set()
def get_name_map_cached(d):
    # 생략 (기존 로직 동일)
    return {} 
def resolve_trade_date():
    return (datetime.now(KST) - timedelta(days=0 if datetime.now(KST).hour >= 18 else 1)).strftime("%Y%m%d")
def build_mcap_map():
    return {}, datetime.now(KST).strftime("%Y%m%d") # 간단 처리 (실제 로직은 위에서 복사하세요)
# ------------------------------- 텔레그램 함수 -------------------------------
def send_telegram_auto(df):
    log("📨 텔레그램 발송 시작...")
    if not TG_TOKEN or not TG_ID: return
    try:
        top5 = df.head(5).reset_index(drop=True)
        msg = f"🔥 [LDY v5.4] 추천 Top 5 ({datetime.now(KST).strftime('%m/%d')})\n" + "-"*30 + "\n\n"
        for i, row in top5.iterrows():
            msg += f"{i+1}. {row['종목명']} ({row.get('ROUTE', '전략없음')})\n"
            msg += f"   🔵 매수: {row['추천매수가']:,}\n"
            msg += f"   🔴 손절: {row['손절가']:,}\n"
            msg += f"   🟢 목표1: {row['추천매도가1']:,}\n\n"
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id": TG_ID, "text": msg})
        log("🚀 전송 완료")
    except Exception as e: log(f"⚠️ 전송 실패: {e}")

if __name__ == "__main__":
    main()
