# -*- coding: utf-8 -*-
"""
LDY Pro Trader: Nightly Collector (KRX)
- 유동성 상위 종목 선정 → 60거래일 OHLCV → 지표·EBS·추천가 계산
- 최종 CSV 저장 전에 종목명은 '사전 생성한 코드맵'으로 확정 매핑
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
MIN_TURNOVER_EOK = 50
MIN_MCAP_EOK = 1000
RSI_LOW, RSI_HIGH = 45, 65
PASS_SCORE = 4
SLEEP_SEC = 0.02
OUT_DIR = "data"
CODEMAP_PATH = os.path.join(OUT_DIR, "krx_codes.csv")

# ------------------------------- 유틸 -------------------------------
def log(msg: str):
    print(f"[{datetime.now(KST)}] {msg}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def calc_rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calc_atr(high, low, close, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def round_to_tick(price: float) -> int:
    return int(round(price / 10.0) * 10)

# ------------------------------- 코드맵 -------------------------------
def build_codemap(date_yyyymmdd: str) -> pd.DataFrame:
    """
    KOSPI/KOSDAQ/KONEX 전체 코드-이름-시장 맵을 생성해서 CSV로 저장.
    """
    ensure_dir(OUT_DIR)
    rows = []
    for m in ["KOSPI", "KOSDAQ", "KONEX"]:
        try:
            lst = stock.get_market_ticker_list(date_yyyymmdd, market=m)
        except Exception:
            lst = []
        for t in lst:
            try:
                nm = stock.get_market_ticker_name(t)
            except Exception:
                nm = None
            rows.append({"종목코드": str(t).zfill(6), "종목명": nm, "시장": m})
            time.sleep(0.005)
    df = pd.DataFrame(rows).drop_duplicates("종목코드")
    df.to_csv(CODEMAP_PATH, index=False, encoding="utf-8-sig")
    return df

def load_codemap() -> pd.DataFrame:
    if os.path.exists(CODEMAP_PATH):
        return pd.read_csv(CODEMAP_PATH, dtype={"종목코드":"string"})
    return pd.DataFrame(columns=["종목코드","종목명","시장"])

# ------------------------------- 데이터 추출 -------------------------------
def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    """
    투자주체별 매매금액(일자×일자)에서 '전체' 금액 기준 상위 추출
    반환: ['종목코드','거래대금(원)']
    """
    tv = stock.get_market_trading_value_by_date(date_yyyymmdd, date_yyyymmdd)
    tv = tv.reset_index().rename(columns={"티커": "종목코드"}, errors="ignore")
    if "종목코드" not in tv.columns:
        tv.insert(0, "종목코드", tv.index)

    if "전체" not in tv.columns:
        cand = [c for c in tv.columns if c not in ["종목코드"]]
        tv["전체"] = tv[cand].sum(axis=1)

    tv = tv[["종목코드", "전체"]].rename(columns={"전체": "거래대금(원)"})
    tv["종목코드"] = tv["종목코드"].astype(str).str.zfill(6)
    tv = tv.sort_values("거래대금(원)", ascending=False).head(top_n).reset_index(drop=True)
    return tv

def get_market_map(date_yyyymmdd: str):
    toset = lambda m: set([str(x).zfill(6) for x in stock.get_market_ticker_list(date_yyyymmdd, market=m)])
    return toset("KOSPI"), toset("KOSDAQ")

def get_mcap_eok(date_yyyymmdd: str, ticker: str) -> float:
    try:
        cap = stock.get_market_cap_by_date(date_yyyymmdd, date_yyyymmdd, ticker)
        return float(cap["시가총액"].iloc[0]) / 1e8
    except Exception:
        return np.nan

# ------------------------------- 메인 -------------------------------
def main():
    log("전종목 수집 시작…")
    end_dt = datetime.now(KST)
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
    end_s = end_dt.strftime("%Y%m%d")
    start_s = start_dt.strftime("%Y%m%d")

    # 1) 코드맵 준비(먼저 저장해 두고 이후에도 활용)
    log("🏷️ 코드맵 생성/로딩…")
    codemap = build_codemap(end_s)  # 항상 최신으로 갱신
    code2name = dict(zip(codemap["종목코드"], codemap["종목명"]))
    code2mkt  = dict(zip(codemap["종목코드"], codemap["시장"]))

    # 2) TOP 유동성 종목 선정
    log("🔍 거래대금 상위 종목 선정 중…")
    top_df = pick_top_by_trading_value(end_s, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ TOP {len(tickers)} 종목 선정 완료")

    kospi_set, kosdaq_set = get_market_map(end_s)

    rows = []
    for t in tickers:
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty:
                continue
            ohlcv = ohlcv.reset_index().rename(columns={"index": "날짜"})
            ohlcv["날짜"] = pd.to_datetime(ohlcv["날짜"])

            close = ohlcv["종가"].astype(float)
            high  = ohlcv["고가"].astype(float)
            low   = ohlcv["저가"].astype(float)
            vol   = ohlcv["거래량"].astype(float)

            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            atr14 = calc_atr(high, low, close, 14)
            rsi14 = calc_rsi(close, 14)

            ema12 = ema(close, 12); ema26 = ema(close, 26)
            macd_line = ema12 - ema26
            macd_signal = ema(macd_line, 9)
            macd_hist = macd_line - macd_signal
            macd_slope = macd_hist.diff()
            vol_z = vol / (vol.rolling(20).mean())
            disp = (close / ma20 - 1.0) * 100

            last = ohlcv.iloc[-1]
            c = float(last["종가"])
            v_z     = float(vol_z.iloc[-1])   if not np.isnan(vol_z.iloc[-1])   else np.nan
            rsi_v   = float(rsi14.iloc[-1])   if not np.isnan(rsi14.iloc[-1])   else np.nan
            macd_h  = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else np.nan
            macd_sl = float(macd_slope.iloc[-1]) if not np.isnan(macd_slope.iloc[-1]) else np.nan
            m20     = float(ma20.iloc[-1])    if not np.isnan(ma20.iloc[-1])    else np.nan
            m60     = float(ma60.iloc[-1])    if not np.isnan(ma60.iloc[-1])    else np.nan
            atr     = float(atr14.iloc[-1])   if not np.isnan(atr14.iloc[-1])   else np.nan
            disp_v  = float(disp.iloc[-1])    if not np.isnan(disp.iloc[-1])    else np.nan
            ret5    = (close.pct_change(5).iloc[-1]  * 100) if len(close) >= 6  else np.nan
            ret10   = (close.pct_change(10).iloc[-1] * 100) if len(close) >= 11 else np.nan

            # 시장/이름/시총/거래대금
            mkt = code2mkt.get(t, "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타"))
            name = code2name.get(t, "")
            tv_eok = float(top_df.loc[top_df["종목코드"] == t, "거래대금(원)"].values[0]) / 1e8
            mcap_eok = get_mcap_eok(end_s, t)

            # 필터
            if tv_eok < MIN_TURNOVER_EOK or (not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK):
                continue

            # EBS
            score, reason = 0, []
            if RSI_LOW <= rsi_v <= RSI_HIGH: score += 1; reason.append("RSI 45~65")
            if macd_sl > 0:                  score += 1; reason.append("MACD↑")
            if not np.isnan(disp_v) and -1.0 <= disp_v <= 4.0: score += 1; reason.append("MA20 근처")
            if v_z > 1.2:                    score += 1; reason.append("거래량↑")
            if not np.isnan(m20) and not np.isnan(m60) and m20 > m60: score += 1; reason.append("상승구조")
            if macd_h > 0:                   score += 1; reason.append("MACD>sig")
            if ret5 is not np.nan and ret5 < 10: score += 1; reason.append("과열X")

            if np.isnan(atr) or np.isnan(m20):
                continue

            buy  = min(c, m20 * 1.01)
            stop = buy - 1.5 * atr
            tgt1 = buy + (buy - stop) * 1.0
            tgt2 = buy + (buy - stop) * 2.0

            buy  = round_to_tick(buy)
            stop = max(round_to_tick(stop), round_to_tick(m20 * 0.97))
            tgt1 = round_to_tick(tgt1)
            tgt2 = round_to_tick(tgt2)

            rows.append({
                "시장": mkt,
                "종목명": name,            # 일단 코드맵 값
                "종목코드": t,
                "종가": int(c),
                "거래대금(억원)": round(tv_eok, 2),
                "시가총액(억원)": None if np.isnan(mcap_eok) else round(mcap_eok, 1),
                "RSI14": None if np.isnan(rsi_v) else round(rsi_v, 1),
                "乖離%": None if np.isnan(disp_v) else round(disp_v, 2),
                "MACD_hist": None if np.isnan(macd_h) else round(macd_h, 4),
                "MACD_slope": None if np.isnan(macd_sl) else round(macd_sl, 5),
                "Vol_Z": None if np.isnan(v_z) else round(v_z, 2),
                "ret_5d_%": None if np.isnan(ret5) else round(ret5, 2),
                "ret_10d_%": None if np.isnan(ret10) else round(ret10, 2),
                "EBS": int(score),
                "통과": "🚀초입" if score >= PASS_SCORE else "",
                "근거": ", ".join(reason),
                "추천매수가": buy,
                "추천매도가1": tgt1,
                "추천매도가2": tgt2,
                "손절가": stop,
            })
        except Exception as e:
            log(f"⚠️ {t} 처리 실패: {e}")
        time.sleep(SLEEP_SEC)

    if not rows:
        raise RuntimeError("수집 결과가 비었습니다.")

    # 3) 최종 매핑 보강(혹시 공란이 있으면 코드맵으로 다시 채움)
    df_out = pd.DataFrame(rows)
    df_out["종목코드"] = df_out["종목코드"].astype(str).str.zfill(6)
    if "종목명" in df_out.columns:
        mask_blank = df_out["종목명"].isna() | (df_out["종목명"].astype(str).str.strip() == "")
        if mask_blank.any():
            df_out = df_out.merge(codemap[["종목코드","종목명"]], on="종목코드", how="left", suffixes=("", "_map"))
            df_out["종목명"] = df_out["종목명"].where(~mask_blank, df_out["종목명_map"])
            df_out.drop(columns=["종목명_map"], inplace=True)

    df_out = df_out.sort_values(["EBS","거래대금(억원)"], ascending=[False, False]).reset_index(drop=True)

    ensure_dir(OUT_DIR)
    today = end_dt.strftime("%Y%m%d")
    path_day    = os.path.join(OUT_DIR, f"recommend_{today}.csv")
    path_latest = os.path.join(OUT_DIR, "recommend_latest.csv")
    df_out.to_csv(path_day,    index=False, encoding="utf-8-sig")
    df_out.to_csv(path_latest, index=False, encoding="utf-8-sig")
    log(f"💾 저장 완료: {path_day} (+ {path_latest})")

if __name__ == "__main__":
    main()
