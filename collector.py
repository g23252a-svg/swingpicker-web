# -*- coding: utf-8 -*-
"""
LDY Pro Trader: Nightly Collector (KRX) — v3.4
- 매일 장마감 후: 거래대금 상위 종목 선정 → 60거래일 OHLCV 수집 → EBS/추천가 산출
- 엔트리(추천매수가): MA20을 기준으로 모멘텀 방향(↑/↓)으로 0.2*ATR 치우침(= Now 0% 도배 완화)
- RR/손절여유/목표1여유/Now-Entry(%, 틱) 계산 내장 → 앱에서 바로 Top Picks 필터 사용
"""

import os
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pykrx import stock

# ------------------------------- 설정 -------------------------------
KST = timezone(timedelta(hours=9))
LOOKBACK_DAYS = 60
TOP_N = 600
MIN_TURNOVER_EOK = 50     # 거래대금(억원) 하한
MIN_MCAP_EOK = 1000       # 시총(억원) 하한

RSI_LOW, RSI_HIGH = 45, 65
PASS_SCORE = 4
SLEEP_SEC = 0.05
OUT_DIR = "data"

# 틱/바이어스
TICK = 10                 # 단순 10원 틱 (KRX 실제 호가단계와 다를 수 있음)
ATR_BIAS = 0.2            # MA20에서 모멘텀 방향으로 치우칠 정도 (0.0~0.5 권장)

UTF8 = "utf-8-sig"

# ------------------------------- 유틸 -------------------------------
def log(msg: str): print(f"[{datetime.now(KST)}] {msg}")
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def calc_rsi(close: pd.Series, period: int = 14):
    d = close.diff()
    up = d.clip(lower=0); dn = (-d).clip(lower=0)
    rs = up.rolling(period).mean() / dn.replace(0, np.nan).rolling(period).mean()
    return 100 - 100/(1+rs)

def calc_atr(high, low, close, period: int = 14):
    prev = close.shift(1)
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def round_to_tick(price: float) -> int:
    return int(round(price / TICK) * TICK)

# ------------------------------- 기준일 -------------------------------
def resolve_trade_date() -> str:
    now = datetime.now(KST)
    d = now.date()
    if now.hour < 18:
        d = d - timedelta(days=1)
    for _ in range(7):
        ymd = d.strftime("%Y%m%d")
        try:
            df = stock.get_market_ohlcv_by_ticker(ymd, market="KOSPI")
            if df is not None and not df.empty and "거래대금" in df.columns:
                return ymd
        except Exception:
            pass
        d = d - timedelta(days=1)
    return datetime.now(KST).strftime("%Y%m%d")

# ------------------------------- 거래대금 상위 -------------------------------
def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m)
            if df is None or df.empty: continue
            df = df.reset_index()
            if "티커" in df.columns:
                df = df.rename(columns={"티커": "종목코드"})
            if "거래대금(원)" not in df.columns and "거래대금" in df.columns:
                df = df.rename(columns={"거래대금": "거래대금(원)"})
            frames.append(df[["종목코드", "거래대금(원)"]])
        except Exception as e:
            log(f"⚠️ {m} TV 집계 실패: {e}")
    if not frames:
        raise RuntimeError("거래대금 상위 집계에 사용할 데이터가 없습니다.")
    tv_df = pd.concat(frames, ignore_index=True)
    tv_df["종목코드"] = tv_df["종목코드"].astype(str).str.zfill(6)
    tv_df["거래대금(원)"] = pd.to_numeric(tv_df["거래대금(원)"], errors="coerce").fillna(0)
    tv_df = tv_df.sort_values("거래대금(원)", ascending=False).head(top_n).reset_index(drop=True)
    return tv_df

def get_market_map(date_yyyymmdd: str):
    kospi = set(stock.get_market_ticker_list(date_yyyymmdd, market="KOSPI"))
    kosdaq = set(stock.get_market_ticker_list(date_yyyymmdd, market="KOSDAQ"))
    return kospi, kosdaq

def get_name_map_cached(date_yyyymmdd: str) -> dict:
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, "krx_codes.csv")
    mp = {}
    if os.path.exists(path):
        try:
            m = pd.read_csv(path, dtype={"종목코드":"string"})
            for _, r in m.iterrows():
                mp[str(r["종목코드"]).zfill(6)] = r.get("종목명", "")
        except Exception:
            mp = {}
    if not mp:
        rows = []
        for mk in ["KOSPI", "KOSDAQ", "KONEX"]:
            try:
                lst = stock.get_market_ticker_list(date_yyyymmdd, market=mk)
            except Exception:
                lst = []
            for t in lst:
                try: nm = stock.get_market_ticker_name(t)
                except Exception: nm = ""
                rows.append({"종목코드": str(t).zfill(6), "종목명": nm, "시장": mk})
                time.sleep(0.002)
        if rows:
            m = pd.DataFrame(rows).drop_duplicates("종목코드")
            m.to_csv(path, index=False, encoding=UTF8)
            mp = {str(r["종목코드"]).zfill(6): r["종목명"] for _, r in m.iterrows()}
    return mp

def get_mcap_eok(date_yyyymmdd: str, ticker: str) -> float:
    try:
        cap = stock.get_market_cap_by_date(date_yyyymmdd, date_yyyymmdd, ticker)
        return float(cap["시가총액"].iloc[0]) / 1e8
    except Exception:
        return np.nan

# ------------------------------- CP949 안전 치환 -------------------------------
def make_cp949_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.replace("乖離%", "괴리_%") for c in out.columns]
    if "통과" in out.columns:
        out["통과"] = out["통과"].replace({"🚀초입":"초입"})
    if "근거" in out.columns and out["근거"].dtype == object:
        out["근거"] = (out["근거"]
                       .str.replace("MACD↑","MACD상승",regex=False)
                       .str.replace("거래량↑","거래량증가",regex=False)
                       .str.replace("과열X","과열아님",regex=False))
    return out

# ------------------------------- 메인 -------------------------------
def main():
    log("전종목 수집 시작…")
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일 확정: {trade_ymd}")

    log("🔍 거래대금 상위 종목 선정 중…")
    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ TOP {len(tickers)} 종목 선정 완료")

    kospi_set, kosdaq_set = get_market_map(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)

    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS*2)
    start_s, end_s = start_dt.strftime("%Y%m%d"), trade_ymd

    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty: continue
            ohlcv = ohlcv.reset_index().rename(columns={"index":"날짜"})
            ohlcv["날짜"] = pd.to_datetime(ohlcv["날짜"])

            ohlcv = ohlcv.tail(LOOKBACK_DAYS)
            if len(ohlcv) < 20: continue

            close = ohlcv["종가"].astype(float)
            high  = ohlcv["고가"].astype(float)
            low   = ohlcv["저가"].astype(float)
            vol   = ohlcv["거래량"].astype(float)

            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            atr14 = calc_atr(high, low, close, 14)
            rsi14 = calc_rsi(close, 14)
            ema12 = ema(close,12); ema26 = ema(close,26)
            macd_line = ema12 - ema26
            macd_sig  = ema(macd_line, 9)
            macd_hist = macd_line - macd_sig
            macd_sl   = macd_hist.diff()
            vol_z = vol / vol.rolling(20).mean()
            disp  = (close/ma20 - 1.0) * 100

            last = ohlcv.iloc[-1]
            c = float(last["종가"])
            v_z   = float(vol_z.iloc[-1]) if not np.isnan(vol_z.iloc[-1]) else np.nan
            rsi_v = float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else np.nan
            m20   = float(ma20.iloc[-1])  if not np.isnan(ma20.iloc[-1])  else np.nan
            m60   = float(ma60.iloc[-1])  if not np.isnan(ma60.iloc[-1])  else np.nan
            atr   = float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else np.nan
            mh    = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else np.nan
            msl   = float(macd_sl.iloc[-1])   if not np.isnan(macd_sl.iloc[-1])   else np.nan
            dispv = float(disp.iloc[-1]) if not np.isnan(disp.iloc[-1]) else np.nan
            ret5  = (close.pct_change(5 ).iloc[-1]*100) if len(close)>=6  else np.nan
            ret10 = (close.pct_change(10).iloc[-1]*100) if len(close)>=11 else np.nan

            mkt = "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타")
            name = name_map.get(str(t).zfill(6), "") or stock.get_market_ticker_name(t)
            tv_eok = float(top_df.loc[top_df["종목코드"]==t,"거래대금(원)"].values[0]) / 1e8
            mcap_eok = get_mcap_eok(trade_ymd, t)

            # 컷(유동성/시총)
            if tv_eok < MIN_TURNOVER_EOK or (not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK):
                continue

            # EBS
            score, reason = 0, []
            if RSI_LOW <= rsi_v <= RSI_HIGH: score+=1; reason.append("RSI 45~65")
            if msl > 0:                      score+=1; reason.append("MACD상승")
            if not np.isnan(dispv) and -1.0 <= dispv <= 4.0: score+=1; reason.append("MA20 근처")
            if v_z > 1.2:                    score+=1; reason.append("거래량증가")
            if not np.isnan(m20) and not np.isnan(m60) and m20 > m60: score+=1; reason.append("상승구조")
            if mh > 0:                       score+=1; reason.append("MACD>sig")
            if not np.isnan(ret5) and ret5 < 10: score+=1; reason.append("과열아님")

            if np.isnan(atr) or np.isnan(m20): continue

            # --- 추천가(엔트리): 모멘텀 바이어스 + 밴드 클립 + 틱/동일가 보정 ---
            band_lo = m20 - 0.5*atr
            band_hi = m20 + 0.5*atr
            dir_sign = 1 if (msl > 0 or (not np.isnan(m60) and m20 > m60)) else -1
            buy_raw = m20 + dir_sign * ATR_BIAS * atr
            buy_raw = float(np.clip(buy_raw, band_lo, band_hi))

            buy  = round_to_tick(buy_raw)
            if int(round(c)) == buy:
                buy += (TICK if dir_sign > 0 else -TICK)

            stop = max(round_to_tick(band_lo - 0.2*atr), round_to_tick(m20 * 0.97))
            tgt1 = round_to_tick(buy_raw + 1.0*atr)
            tgt2 = round_to_tick(buy_raw + 1.8*atr)

            # 파생 지표 (퍼센트/틱, RR)
            now_entry_pct = (c - buy) / buy * 100.0 if buy > 0 else np.nan
            stop_buf_pct  = (buy - stop) / buy * 100.0 if buy > 0 else np.nan
            tgt1_buf_pct  = (tgt1 - buy) / buy * 100.0 if buy > 0 else np.nan
            rr_min = (tgt1_buf_pct / stop_buf_pct) if (stop_buf_pct and stop_buf_pct > 0) else np.nan
            now_ticks = (c - buy) / TICK

            rows.append({
                "시장": mkt,
                "종목명": name,
                "종목코드": t,
                "종가": int(c),
                "거래대금(억원)": round(tv_eok, 2),
                "시가총액(억원)": None if np.isnan(mcap_eok) else round(mcap_eok, 1),
                "RSI14": None if np.isnan(rsi_v) else round(rsi_v, 1),
                "乖離%": None if np.isnan(dispv) else round(dispv, 2),
                "MACD_hist": None if np.isnan(mh) else round(mh, 4),
                "MACD_slope": None if np.isnan(msl) else round(msl, 5),
                "Vol_Z": None if np.isnan(v_z) else round(v_z, 2),
                "ret_5d_%": None if np.isnan(ret5) else round(ret5, 2),
                "ret_10d_%": None if np.isnan(ret10) else round(ret10, 2),
                "EBS": int(score),
                "통과": "초입" if score >= PASS_SCORE else "",
                "근거": ", ".join(reason),
                "추천매수가": buy,
                "추천매도가1": tgt1,
                "추천매도가2": tgt2,
                "손절가": stop,
                # 파생
                "NOW_ENTRY_%": None if np.isnan(now_entry_pct) else round(now_entry_pct, 4),
                "NOW_TICKS": None if np.isnan(now_ticks) else int(round(now_ticks)),
                "STOP_BUF_%": None if np.isnan(stop_buf_pct) else round(stop_buf_pct, 4),
                "T1_BUF_%": None if np.isnan(tgt1_buf_pct) else round(tgt1_buf_pct, 4),
                "MIN_RR": None if np.isnan(rr_min) else round(rr_min, 6),
            })
        except Exception as e:
            log(f"⚠️ {t} 처리 실패: {e}")
        time.sleep(SLEEP_SEC)

    if not rows:
        raise RuntimeError("수집 결과가 비었습니다.")

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["EBS","거래대금(억원)"], ascending=[False, False]).reset_index(drop=True)

    ensure_dir(OUT_DIR)
    path_day_utf8    = os.path.join(OUT_DIR, f"recommend_{trade_ymd}.csv")
    path_latest_utf8 = os.path.join(OUT_DIR, "recommend_latest.csv")

    # UTF-8 본
    df_out.to_csv(path_day_utf8, index=False, encoding=UTF8)
    df_out.to_csv(path_latest_utf8, index=False, encoding=UTF8)

    # (선택) CP949 호환본도 남기고 싶으면 주석 해제
    # safe = make_cp949_safe(df_out)
    # safe.to_csv(os.path.join(OUT_DIR, f"recommend_{trade_ymd}_cp949.csv"), index=False, encoding="cp949")

    log(f"💾 저장 완료: {path_day_utf8} / {path_latest_utf8}")
    log("✅ 모든 작업 완료")

if __name__ == "__main__":
    main()
