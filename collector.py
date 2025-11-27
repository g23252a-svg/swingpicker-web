# -*- coding: utf-8 -*-
"""
LDY Pro Trader v4.6: Nightly Collector (Debugged)
- Update: Telegram Debugging Added
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

# [Secret 로드 확인용 로그]
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

# ------------------------------- 유틸 -------------------------------
def log(msg: str):
    print(f"[{datetime.now(KST)}] {msg}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def calc_rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - 100 / (1 + rs)

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_mfi(high, low, close, vol, period=14):
    tp = (high + low + close) / 3
    rmf = tp * vol
    diff = tp.diff()
    pos_flow = np.where(diff > 0, rmf, 0)
    neg_flow = np.where(diff < 0, rmf, 0)
    pos_sum = pd.Series(pos_flow).rolling(period).sum()
    neg_sum = pd.Series(neg_flow).rolling(period).sum()
    mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, 1)))
    return mfi

def round_to_tick(price: float) -> int:
    if price < 2000: tick = 1
    elif price < 5000: tick = 5
    elif price < 20000: tick = 10
    elif price < 50000: tick = 50
    elif price < 200000: tick = 100
    elif price < 500000: tick = 500
    else: tick = 1000
    return int(round(price / tick) * tick)

def _safe_sum(x) -> float:
    return pd.to_numeric(x, errors="coerce").fillna(0).sum()

# ------------------------------- 검증 및 데이터 -------------------------------
def _has_ohlcv_and_mcap(ymd: str) -> bool:
    o_valid = False
    for m in ("KOSPI", "KOSDAQ"):
        try:
            o = stock.get_market_ohlcv_by_ticker(ymd, market=m)
        except Exception: o = None
        if o is not None and not o.empty and "거래대금" in o.columns and _safe_sum(o["거래대금"]) > 0:
            o_valid = True; break
    
    m_valid = False
    for m in ("KOSPI", "KOSDAQ"):
        try:
            mc = stock.get_market_cap_by_ticker(ymd, market=m)
        except Exception: mc = None
        if mc is not None and not mc.empty and "시가총액" in mc.columns and _safe_sum(mc["시가총액"]) > 0:
            m_valid = True; break
            
    return o_valid and m_valid

def resolve_trade_date() -> str:
    now = datetime.now(KST)
    d = now.date()
    if now.hour < 18: d = d - timedelta(days=1)
    for _ in range(10):
        ymd = d.strftime("%Y%m%d")
        if _has_ohlcv_and_mcap(ymd): return ymd
        d = d - timedelta(days=1)
    return (now - timedelta(days=1)).strftime("%Y%m%d")

def build_mcap_map() -> tuple[dict, str]:
    today = datetime.now(KST).date()
    if datetime.now(KST).hour < 18: today = today - timedelta(days=1)
    d = today
    for _ in range(10):
        use = d.strftime("%Y%m%d")
        any_ok = False
        mcap: dict[str, float] = {}
        for m in ("KOSPI", "KOSDAQ"):
            try: df = stock.get_market_cap_by_ticker(use, market=m)
            except: df = None
            if df is None or df.empty or "시가총액" not in df.columns: continue
            raw = pd.to_numeric(df["시가총액"], errors="coerce")
            if raw.fillna(0).sum() <= 0: continue
            
            tickers = df.index.astype(str).str.zfill(6)
            vals = (raw / 1e8).astype(float)
            mcap.update({t: float(v) for t, v in zip(tickers, vals)})
            any_ok = True
            
        if any_ok and len(mcap) > 0:
            sample = mcap.get("005930", np.nan)
            if not (isinstance(sample, (int, float)) and sample > 0):
                d = d - timedelta(days=1); continue
            log(f"🧭 시총 기준일: {use} · 종목수 {len(mcap):,}")
            return mcap, use
        d = d - timedelta(days=1)
    return {}, today.strftime("%Y%m%d")

def get_mcap_eok_from_map(mcap_map: dict, ticker: str) -> float:
    v = mcap_map.get(str(ticker).zfill(6))
    return float(v) if (v is not None and v > 0) else np.nan

def get_sector_map():
    try:
        df = fdr.StockListing('KRX')
        col = next((c for c in ['Sector', '업종', 'Industry'] if c in df.columns), None)
        if not col:
            df1 = fdr.StockListing('KOSPI')
            df2 = fdr.StockListing('KOSDAQ')
            df = pd.concat([df1, df2])
            col = next((c for c in ['Sector', '업종', 'Industry'] if c in df.columns), None)
        
        if col:
            df['Code'] = df['Code'].astype(str).str.zfill(6)
            df = df.dropna(subset=[col])
            return dict(zip(df['Code'], df[col]))
        else:
            return {}
    except: return {}

def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    frames = []
    for m in ("KOSPI", "KOSDAQ"):
        try: df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m)
        except: df = None
        if df is not None and not df.empty:
            df = df.reset_index()
            if "티커" in df.columns: df.rename(columns={"티커": "종목코드"}, inplace=True)
            if "거래대금" in df.columns and "거래대금(원)" not in df.columns:
                df.rename(columns={"거래대금": "거래대금(원)"}, inplace=True)
            frames.append(df[["종목코드", "거래대금(원)"]])
    if not frames: raise RuntimeError("거래대금 데이터 없음")
    tv_df = pd.concat(frames, ignore_index=True)
    tv_df["종목코드"] = tv_df["종목코드"].astype(str).str.zfill(6)
    tv_df["거래대금(원)"] = pd.to_numeric(tv_df["거래대금(원)"], errors="coerce").fillna(0)
    tv_df = tv_df[tv_df["거래대금(원)"] > 0].sort_values("거래대금(원)", ascending=False).head(top_n)
    return tv_df.reset_index(drop=True)

def get_market_sets(date_yyyymmdd: str):
    def _get(mk):
        try: return set(stock.get_market_ticker_list(date_yyyymmdd, market=mk))
        except: return set()
    return _get("KOSPI"), _get("KOSDAQ")

def get_name_map_cached(date_yyyymmdd: str) -> dict:
    ensure_dir(OUT_DIR)
    map_path = os.path.join(OUT_DIR, "krx_codes.csv")
    mp = {}
    if os.path.exists(map_path):
        try:
            df = pd.read_csv(map_path, dtype={"종목코드":"string"}, encoding="utf-8")
            for _, r in df.iterrows(): mp[str(r["종목코드"]).zfill(6)] = r.get("종목명","")
        except: mp = {}
    if mp: return mp
    rows = []
    for m in ["KOSPI", "KOSDAQ", "KONEX"]:
        try: lst = stock.get_market_ticker_list(date_yyyymmdd, market=m)
        except: lst = []
        for t in lst:
            try: nm = stock.get_market_ticker_name(t)
            except: nm = ""
            rows.append({"종목코드": str(t).zfill(6), "종목명": nm, "시장": m})
            time.sleep(0.002)
    if rows:
        df = pd.DataFrame(rows).drop_duplicates("종목코드")
        df.to_csv(map_path, index=False, encoding=UTF8)
        mp = {str(r["종목코드"]).zfill(6): r["종목명"] for _, r in df.iterrows()}
    return mp

# [Modified] 텔레그램 자동 전송 함수 (ROUTE 컬럼 부재 시 예외 처리)
def send_telegram_auto(df):
    log("📨 텔레그램 발송 시작...")
    
    if not TG_TOKEN or not TG_ID:
        log("⚠️ [오류] TG_TOKEN 또는 TG_ID가 설정되지 않았습니다.")
        return

    try:
        top5 = df.head(5)
        trade_date = datetime.now(KST).strftime('%Y-%m-%d')
        msg = f"🔥 [LDY v5.3] 추천 Top 5 ({trade_date})\n"
        msg += "-" * 30 + "\n\n"
        
        for i, row in top5.iterrows():
            rank = i + 1
            name = row['종목명']
            code = row['종목코드']
            
            # [Fix] ROUTE 컬럼이 없으면 '근거' 컬럼의 첫 번째 항목 사용
            if 'ROUTE' in row:
                strategy = row['ROUTE']
            else:
                # 근거가 "RSI적정, MACD상승..." 형태라면 앞부분만 따옴
                strategy = row.get('근거', '전략없음').split(',')[0]

            buy = row['추천매수가']
            stop = row['손절가']
            t1 = row['추천매도가1']
            t2 = row['추천매도가2']
            
            msg += f"{rank}. {name} ({code})\n"
            msg += f"   🎯 전략: {strategy}\n"
            msg += f"   🔵 매수: {buy:,}\n"
            msg += f"   🔴 손절: {stop:,}\n"
            msg += f"   🟢 목표1: {t1:,}\n"
            msg += f"   🟢 목표2: {t2:,}\n\n"
            
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = {"chat_id": TG_ID, "text": msg}
        
        # 전송 시도
        res = requests.post(url, data=data)
        
        if res.status_code == 200:
            log("🚀 텔레그램 전송 성공!")
        else:
            log(f"⚠️ 텔레그램 전송 실패 (Code: {res.status_code}): {res.text}")
            
    except Exception as e:
        log(f"⚠️ 텔레그램 로직 에러: {e}")

# ------------------------------- 메인 로직 -------------------------------
def main():
    log("🚀 LDY Collector v5.3 시작...")

    # ... (기존 데이터 수집 로직은 동일하므로 생략하지 않고 그대로 둠) ...
    mcap_map, mcap_ymd = build_mcap_map()
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일: {trade_ymd}")

    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ 분석 대상: {len(tickers)} 종목")

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

            ma20  = c.rolling(20).mean()
            ma60  = c.rolling(60).mean()
            ma120 = c.rolling(120).mean()
            
            atr14 = calc_atr(h, l, c, 14)
            rsi14 = calc_rsi(c, 14)
            mfi14 = calc_mfi(h, l, c, v, 14)

            ema12 = ema(c, 12); ema26 = ema(c, 26)
            macd_line = ema12 - ema26
            macd_sig  = ema(macd_line, 9)
            macd_hist = macd_line - macd_sig
            macd_slope= macd_hist.diff()

            vol_z = v / (v.rolling(20).mean())
            disp  = (c / ma20 - 1.0) * 100

            std20 = c.rolling(20).std()
            bb_up = ma20 + (std20 * 2)
            bb_lo = ma20 - (std20 * 2)
            bb_w  = (bb_up - bb_lo) / ma20
            bb_w_avg = bb_w.rolling(20).mean()

            last = ohlcv.iloc[-1]
            cur_c = float(last["종가"])
            
            v_rsi  = float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else 50.0
            v_mfi  = float(mfi14.iloc[-1]) if not np.isnan(mfi14.iloc[-1]) else 50.0
            v_slp  = float(macd_slope.iloc[-1]) if not np.isnan(macd_slope.iloc[-1]) else 0.0
            v_hist = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0.0
            v_disp = float(disp.iloc[-1]) if not np.isnan(disp.iloc[-1]) else 0.0
            v_volz = float(vol_z.iloc[-1]) if not np.isnan(vol_z.iloc[-1]) else 0.0
            
            v_m20  = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else 0.0
            v_m60  = float(ma60.iloc[-1]) if not np.isnan(ma60.iloc[-1]) else 0.0
            v_m120 = float(ma120.iloc[-1]) if not np.isnan(ma120.iloc[-1]) else 0.0
            v_bw   = float(bb_w.iloc[-1]) if not np.isnan(bb_w.iloc[-1]) else 0.0
            v_bw_avg = float(bb_w_avg.iloc[-1]) if not np.isnan(bb_w_avg.iloc[-1]) else 0.0
            
            ret5  = (c.pct_change(5).iloc[-1]*100) if len(c)>5 else 0.0
            ret10 = (c.pct_change(10).iloc[-1]*100) if len(c)>10 else 0.0

            mkt = "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타")
            name = name_map.get(str(t).zfill(6), "") or stock.get_market_ticker_name(t)
            sector = sector_map.get(str(t).zfill(6), "기타")
            
            tv_eok = float(top_df.loc[top_df["종목코드"]==t, "거래대금(원)"].values[0])/1e8
            mcap_eok = get_mcap_eok_from_map(mcap_map, t)

            if tv_eok < MIN_TURNOVER_EOK or (not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK):
                continue

            score = 0
            reason = []

            if RSI_LOW <= v_rsi <= RSI_HIGH: score += 1; reason.append("RSI적정")
            if v_slp > 0: score += 1; reason.append("MACD상승")
            if -1.0 <= v_disp <= 5.0: score += 1; reason.append("20선근접")
            if v_volz > 1.2: score += 1; reason.append("거래량↑")
            if v_m20 > v_m60: score += 1; reason.append("정배열(단)")
            
            if cur_c > v_m120: 
                score += 1; reason.append("장기추세(120↑)") 
            else:
                score -= 1 
            
            if v_mfi > 60: 
                score += 1; reason.append("자금유입(MFI)")
            if v_bw < v_bw_avg * 0.8: 
                score += 1; reason.append("에너지응축(Sqz)")

            if v_hist > 0: score += 1; reason.append("MACD>Sig")
            if ret5 < 12: score += 1; reason.append("과열X")
            
            # Entry/Target
            try: atr = float(atr14.iloc[-1])
            except: atr = 0.0
            if np.isnan(atr) or atr <= 0: atr = cur_c * 0.03

            if v_m20 > 0 and cur_c > v_m20:
                buy = min(cur_c, v_m20 * 1.03) 
            else:
                buy = cur_c

            raw_stop = buy - (2.0 * atr)
            max_loss_limit = buy * 0.93
            if raw_stop < max_loss_limit: raw_stop = max_loss_limit
            if raw_stop >= buy * 0.97: raw_stop = buy * 0.97
            stop = raw_stop

            risk = buy - stop
            if score >= 80: rr1, rr2 = 2.0, 4.0
            elif score >= 70: rr1, rr2 = 1.5, 3.0
            else: rr1, rr2 = 1.2, 2.5
                
            tgt1 = buy + (risk * rr1)
            tgt2 = buy + (risk * rr2)
            
            buy = round_to_tick(buy); stop = round_to_tick(stop)
            tgt1 = round_to_tick(tgt1); tgt2 = round_to_tick(tgt2)

            rows.append({
                "시장": mkt, "종목명": name, "종목코드": str(t).zfill(6), "업종": sector,
                "종가": int(cur_c),
                "거래대금(억원)": round(tv_eok, 2),
                "시가총액(억원)": None if np.isnan(mcap_eok) else round(mcap_eok, 1),
                "RSI14": round(v_rsi, 1),
                "MFI14": round(v_mfi, 1),
                "이격도": round(v_disp, 2),       
                "MACD_Hist": round(v_hist, 4),
                "MACD_Slope": round(v_slp, 5),
                "거래강도": round(v_volz, 2),     
                "ret_5d_%": round(ret5, 2),
                "ret_10d_%": round(ret10, 2),
                "EBS": int(score),
                "통과": "★" if score >= PASS_SCORE else "",
                "근거": ", ".join(reason),
                "추천매수가": buy, "손절가": stop, "추천매도가1": tgt1, "추천매도가2": tgt2
            })

        except Exception as e:
            pass
        time.sleep(SLEEP_SEC)

    if not rows: raise RuntimeError("수집 결과 없음")

    df_out = pd.DataFrame(rows).sort_values(["EBS", "거래대금(억원)"], ascending=[False, False])
    
    ensure_dir(OUT_DIR)
    path_day = os.path.join(OUT_DIR, f"recommend_{trade_ymd}.csv")
    path_lat = os.path.join(OUT_DIR, "recommend_latest.csv")
    
    df_out.to_csv(path_day, index=False, encoding=UTF8)
    df_out.to_csv(path_lat, index=False, encoding=UTF8)
    log(f"💾 저장완료: {path_lat} ({len(df_out)}건)")
    
    # [텔레그램 발송 트리거]
    send_telegram_auto(df_out)

if __name__ == "__main__":
    main()
