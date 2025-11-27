# -*- coding: utf-8 -*-
"""
LDY Pro Trader Collector v5.5 (Final)
- Fix: Sector Info Retrieval (Robust Fallback)
- Features: Realistic Stop Loss, Telegram Auto-Alert
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

# [보안 설정] Secrets 로드
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_ID = os.environ.get("TG_ID")

# ------------------------------- 설정 -------------------------------
KST = timezone(timedelta(hours=9))
LOOKBACK_DAYS = 250          
TOP_N = 600                 
MIN_TURNOVER_EOK = 50       
MIN_MCAP_EOK = 1000         
RSI_LOW, RSI_HIGH = 45, 65  
PASS_EBS = 4              
SLEEP_SEC = 0.05            
OUT_DIR = "data"
UTF8 = "utf-8-sig"

# [LDY Score 가중치]
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = 0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = 6.0, 6.0, 4.0
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 4.0, 2.0

# ------------------------------- 유틸 -------------------------------
def log(msg: str): print(f"[{datetime.now(KST)}] {msg}")
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def _safe_sum(x): return pd.to_numeric(x, errors="coerce").fillna(0).sum()
def nz_num(s): return pd.to_numeric(s, errors="coerce")

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

# ------------------------------- 데이터 수집 로직 -------------------------------
def _has_ohlcv_and_mcap(ymd):
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            o = stock.get_market_ohlcv_by_ticker(ymd, market=m)
            if o is not None and not o.empty and "거래대금" in o.columns and _safe_sum(o["거래대금"]) > 0: return True
        except: pass
    return False

def resolve_trade_date():
    d = datetime.now(KST).date()
    if datetime.now(KST).hour < 18: d -= timedelta(days=1)
    for _ in range(10):
        ymd = d.strftime("%Y%m%d")
        if _has_ohlcv_and_mcap(ymd): return ymd
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")

def build_mcap_map():
    d = datetime.now(KST).date()
    if datetime.now(KST).hour < 18: d -= timedelta(days=1)
    for _ in range(10):
        use = d.strftime("%Y%m%d")
        try:
            df = pd.concat([stock.get_market_cap_by_ticker(use, market='KOSPI'), stock.get_market_cap_by_ticker(use, market='KOSDAQ')])
            if not df.empty:
                df['Code'] = df.index
                return dict(zip(df['Code'], df['시가총액']/1e8)), use
        except: pass
        d -= timedelta(days=1)
    return {}, ""

def get_mcap_eok_from_map(mcap_map, ticker):
    return float(mcap_map.get(str(ticker).zfill(6), 0))

# [핵심 FIX] 업종 정보 수집 강화
def get_sector_map():
    sector_map = {}
    try:
        # 시도 1: KRX 전체
        log("📋 업종 정보 수집 중 (KRX)...")
        df = fdr.StockListing('KRX')
        
        # 컬럼명 후보군 (라이브러리 버전에 따라 다름)
        candidates = ['Sector', '업종', 'Industry', 'IndustryCode']
        col = next((c for c in candidates if c in df.columns), None)
        
        if col:
            # Code 컬럼 정규화
            code_col = 'Code' if 'Code' in df.columns else 'Symbol'
            df[code_col] = df[code_col].astype(str).str.zfill(6)
            df = df.dropna(subset=[col]) # 업종 없는거 제거
            sector_map = dict(zip(df[code_col], df[col]))
            log(f"✅ 업종 정보 {len(sector_map)}건 로드 성공")
            return sector_map
    except Exception as e:
        log(f"⚠️ KRX 로드 실패: {e}")

    # 시도 2: KOSPI/KOSDAQ 개별 시도 (Fallback)
    if not sector_map:
        try:
            log("🔄 KOSPI/KOSDAQ 개별 수집 시도...")
            df1 = fdr.StockListing('KOSPI')
            df2 = fdr.StockListing('KOSDAQ')
            df = pd.concat([df1, df2])
            col = next((c for c in ['Sector', '업종'] if c in df.columns), None)
            if col:
                df['Code'] = df['Code'].astype(str).str.zfill(6)
                sector_map = dict(zip(df['Code'], df[col]))
                log(f"✅ 개별 수집 성공: {len(sector_map)}건")
        except: pass
        
    return sector_map

def pick_top_by_trading_value(date_yyyymmdd, top_n):
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m).reset_index()
            df.columns = ['종목코드' if '티커' in c else c for c in df.columns]
            df.columns = ['거래대금(원)' if '거래대금' == c else c for c in df.columns]
            frames.append(df[['종목코드', '거래대금(원)']])
        except: pass
    if not frames: raise RuntimeError("No Data")
    df = pd.concat(frames)
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    return df.sort_values('거래대금(원)', ascending=False).head(top_n)

def get_market_sets(d):
    try: return set(stock.get_market_ticker_list(d, market='KOSPI')), set(stock.get_market_ticker_list(d, market='KOSDAQ'))
    except: return set(), set()

def get_name_map_cached(d):
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, "krx_codes.csv")
    if os.path.exists(path):
        try: return dict(zip(pd.read_csv(path, dtype=str)['종목코드'], pd.read_csv(path)['종목명']))
        except: pass
    
    rows = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            for t in stock.get_market_ticker_list(d, market=m):
                rows.append({'종목코드': t, '종목명': stock.get_market_ticker_name(t)})
                time.sleep(0.001)
        except: pass
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding=UTF8)
        return dict(zip(df['종목코드'], df['종목명']))
    return {}

# --- Scoring Logic (Synced with Dashboard) ---
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
    
    # 전략 태그
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

def send_telegram_auto(df):
    log("📨 텔레그램 발송 시작...")
    if not TG_TOKEN or not TG_ID:
        log("⚠️ [오류] TG_TOKEN/ID 미설정")
        return

    try:
        # [Fix] LDY_SCORE 기준 정렬 후 상위 5개 (EBS 기준 아님!)
        top5 = df.sort_values("LDY_SCORE", ascending=False).head(5).reset_index(drop=True)
        
        trade_date = datetime.now(KST).strftime('%Y-%m-%d')
        msg = f"🔥 [LDY v5.5] 추천 Top 5 ({trade_date})\n"
        msg += "-" * 30 + "\n\n"
        
        for i, row in top5.iterrows():
            rank = i + 1
            name = row['종목명']
            code = row['종목코드']
            route = row.get('ROUTE', '전략없음')
            buy = row['추천매수가']
            stop = row['손절가']
            t1 = row['추천매도가1']
            
            msg += f"{rank}. {name} ({code})\n"
            msg += f"   🎯 전략: {route}\n"
            msg += f"   🔵 매수: {buy:,}\n"
            msg += f"   🔴 손절: {stop:,}\n"
            msg += f"   🟢 목표1: {t1:,}\n\n"
            
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = {"chat_id": TG_ID, "text": msg}
        requests.post(url, data=data)
        log("🚀 전송 완료")
    except Exception as e:
        log(f"⚠️ 전송 실패: {e}")

# ------------------------------- 메인 실행 -------------------------------
def main():
    log("🚀 LDY Collector v5.5 시작...")
    mcap_map, mcap_ymd = build_mcap_map()
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일: {trade_ymd}")

    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ 분석 대상: {len(tickers)} 종목")

    kospi_set, kosdaq_set = get_market_sets(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)
    
    # [핵심] 업종 맵 확보
    sector_map = get_sector_map()

    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS * 2 + 60)
    start_s, end_s = start_dt.strftime("%Y%m%d"), trade_ymd

    rows = []
    for t in tqdm(tickers, desc="Analyzing"):
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty or len(ohlcv) < 120: continue
            ohlcv = ohlcv.tail(LOOKBACK_DAYS)
            
            c, h, l, v = ohlcv["종가"], ohlcv["고가"], ohlcv["저가"], ohlcv["거래량"]
            ma20, ma60, ma120 = c.rolling(20).mean(), c.rolling(60).mean(), c.rolling(120).mean()
            
            atr = calc_atr(h, l, c, 14).iloc[-1]
            rsi = calc_rsi(c, 14).iloc[-1]
            mfi = calc_mfi(h, l, c, v, 14).iloc[-1]
            
            macd = ema(c, 12) - ema(c, 26)
            sig = ema(macd, 9)
            hist = macd - sig
            slope = hist.diff().iloc[-1]
            
            vol_z = (v / v.rolling(20).mean()).iloc[-1]
            disp = ((c / ma20 - 1.0) * 100).iloc[-1]
            
            std20 = c.rolling(20).std()
            bb_w = ((ma20 + std20*2) - (ma20 - std20*2)) / ma20
            
            last_c = c.iloc[-1]
            
            tv_eok = float(top_df.loc[top_df["종목코드"]==t, "거래대금(원)"].values[0])/1e8
            mcap = get_mcap_eok_from_map(mcap_map, t)
            
            if tv_eok < MIN_TURNOVER_EOK or (mcap > 0 and mcap < MIN_MCAP_EOK): continue

            score = 0; reason = []
            if RSI_LOW <= rsi <= RSI_HIGH: score += 1; reason.append("RSI적정")
            if slope > 0: score += 1; reason.append("MACD상승")
            if -1 <= disp <= 5: score += 1; reason.append("20선근접")
            if vol_z > 1.2: score += 1; reason.append("거래량↑")
            if ma20.iloc[-1] > ma60.iloc[-1]: score += 1; reason.append("정배열(단)")
            if last_c > ma120.iloc[-1]: score += 1; reason.append("장기추세(120↑)")
            else: score -= 1
            if mfi > 60: score += 1; reason.append("자금유입(MFI)")
            if hist.iloc[-1] > 0: score += 1; reason.append("MACD>Sig")
            
            # Entry/Target
            try: atr = float(atr)
            except: atr = 0.0
            if np.isnan(atr) or atr <= 0: atr = last_c * 0.03
            buy = min(last_c, ma20.iloc[-1] * 1.03) if ma20.iloc[-1] > 0 and last_c > ma20.iloc[-1] else last_c
            
            stop = buy - (2.0 * atr)
            if stop < buy * 0.93: stop = buy * 0.93
            if stop >= buy * 0.97: stop = buy * 0.97
            
            risk = buy - stop
            rr1, rr2 = (2.0, 4.0) if score >= 8 else ((1.5, 3.0) if score >= 6 else (1.2, 2.5))
            t1 = buy + risk * rr1; t2 = buy + risk * rr2
            
            buy = round_to_tick(buy); stop = round_to_tick(stop); t1 = round_to_tick(t1); t2 = round_to_tick(t2)

            rows.append({
                "시장": "KOSPI" if t in kospi_set else "KOSDAQ",
                "종목명": name_map.get(str(t).zfill(6), str(t)),
                "종목코드": str(t).zfill(6),
                "업종": sector_map.get(str(t).zfill(6), "기타"), # [Fix] 업종 정보 매핑
                "종가": int(last_c), "거래대금(억원)": round(tv_eok, 2),
                "시가총액(억원)": round(mcap, 1),
                "RSI14": round(rsi, 1), "MFI14": round(mfi, 1),
                "이격도": round(disp, 2), "MACD_Hist": round(hist.iloc[-1], 4), "MACD_Slope": round(slope, 5),
                "거래강도": round(vol_z, 2), "ret_5d_%": 0, "ret_10d_%": 0,
                "EBS": int(score), "통과": "★" if score >= PASS_EBS else "", "근거": ", ".join(reason),
                "추천매수가": buy, "손절가": stop, "추천매도가1": t1, "추천매도가2": t2
            })
        except: pass
    
    if not rows: raise RuntimeError("No Result")
    
    df_raw = pd.DataFrame(rows)
    df_out = build_global_score(df_raw).sort_values(["LDY_SCORE", "거래대금(억원)"], ascending=[False, False])
    
    ensure_dir(OUT_DIR)
    df_out.to_csv(os.path.join(OUT_DIR, "recommend_latest.csv"), index=False, encoding=UTF8)
    log(f"💾 저장 완료 ({len(df_out)}건)")
    
    # [핵심] 대시보드 순위와 동일하게 LDY_SCORE 순으로 발송
    send_telegram_auto(df_out)

if __name__ == "__main__":
    main()
