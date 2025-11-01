# -*- coding: utf-8 -*-
"""
LDY Pro Trader: Nightly Collector (KRX)
- 매일 장마감 후: 유동성 상위(TV 상위) 종목 n개 선정
- 각 종목 60거래일 OHLCV 수집 후 당일 스냅샷 지표/점수(EBS) 계산
- 추천매수/목표/손절 가격 포함 CSV 저장
- TV 상위: get_market_ohlcv_by_ticker() 일원화 (pykrx 시그니처 차이 회피)
- 추천가: MA20±0.5ATR 밴드 중심(=밴드 미드), 종가와 독립 → Now-Entry 분포 정상화
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
LOOKBACK_DAYS = 60          # OHLCV 조회일수
TOP_N = 600                 # 거래대금 상위 샘플 크기(300~800 권장)
MIN_TURNOVER_EOK = 50       # 거래대금 하한(억원)
MIN_MCAP_EOK = 1000         # 시총 하한(억원)
RSI_LOW, RSI_HIGH = 45, 65  # RSI 범위
PASS_SCORE = 4              # 통과점수(EBS)
SLEEP_SEC = 0.05            # API call 간 딜레이(안정성)
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
    rs = up.rolling(period).mean() / (down.rolling(period).mean().replace(0, np.nan))
    return 100 - 100 / (1 + rs)

def calc_atr(high, low, close, period: int = 14):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def round_to_tick(price: float) -> int:
    # KRX 간단 틱(10원 단위)
    return int(round(price / 10.0) * 10)

# ------------------------------- 기준일 결정 -------------------------------
def resolve_trade_date() -> str:
    """
    장마감 집계 시차를 고려해서, 데이터가 비었으면 하루씩 뒤로 가며 최근 영업일을 찾는다.
    반환: 'YYYYMMDD'
    """
    now = datetime.now(KST)
    d = now.date()
    if now.hour < 18:
        d = d - timedelta(days=1)

    for _ in range(7):
        ymd = d.strftime("%Y%m%d")
        try:
            tmp = stock.get_market_ohlcv_by_ticker(ymd, market="KOSPI")
            if tmp is not None and not tmp.empty and ("거래대금" in tmp.columns or "거래대금(원)" in tmp.columns):
                return ymd
        except Exception:
            pass
        d = d - timedelta(days=1)
    return datetime.now(KST).strftime("%Y%m%d")

# ------------------------------- 상위 TV 선정 -------------------------------
def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m)
            if df is None or df.empty:
                continue
            df = df.reset_index()
            if "티커" in df.columns:
                df.rename(columns={"티커": "종목코드"}, inplace=True)
            if "거래대금(원)" not in df.columns and "거래대금" in df.columns:
                df.rename(columns={"거래대금": "거래대금(원)"}, inplace=True)
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
    map_path = os.path.join(OUT_DIR, "krx_codes.csv")
    mp = {}
    if os.path.exists(map_path):
        try:
            df = pd.read_csv(map_path, dtype={"종목코드": "string"})
            for _, r in df.iterrows():
                mp[str(r["종목코드"]).zfill(6)] = r.get("종목명", "")
        except Exception:
            mp = {}

    if not mp:
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
                    nm = ""
                rows.append({"종목코드": str(t).zfill(6), "종목명": nm, "시장": m})
                time.sleep(0.002)
        if rows:
            df = pd.DataFrame(rows).drop_duplicates("종목코드")
            df.to_csv(map_path, index=False, encoding=UTF8)
            mp = {str(r["종목코드"]).zfill(6): r["종목명"] for _, r in df.iterrows()}
    return mp

def get_mcap_eok(date_yyyymmdd: str, ticker: str) -> float:
    try:
        cap = stock.get_market_cap_by_date(date_yyyymmdd, date_yyyymmdd, ticker)
        return float(cap["시가총액"].iloc[0]) / 1e8
    except Exception:
        return np.nan

# ------------------------------- 메인 로직 -------------------------------
def main():
    log("전종목 수집 시작…")

    # 1) 기준일
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일 확정: {trade_ymd}")

    # 2) 상위 거래대금 종목
    log("🔍 거래대금 상위 종목 선정 중…")
    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ TOP {len(tickers)} 종목 선정 완료")

    # 3) 시장/이름 맵
    kospi_set, kosdaq_set = get_market_map(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)

    # 4) 각 종목 OHLCV + 지표/점수/추천가
    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS * 2)
    start_s = start_dt.strftime("%Y%m%d")
    end_s = trade_ymd

    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty:
                continue

            ohlcv = ohlcv.reset_index().rename(columns={"index": "날짜"})
            ohlcv["날짜"] = pd.to_datetime(ohlcv["날짜"])
            ohlcv = ohlcv.tail(LOOKBACK_DAYS)

            close = ohlcv["종가"].astype(float)
            high  = ohlcv["고가"].astype(float)
            low   = ohlcv["저가"].astype(float)
            vol   = ohlcv["거래량"].astype(float)

            if len(close) < 20:
                continue

            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            atr14 = calc_atr(high, low, close, 14)
            rsi14 = calc_rsi(close, 14)
            ema12 = ema(close, 12)
            ema26 = ema(close, 26)
            macd_line   = ema12 - ema26
            macd_signal = ema(macd_line, 9)
            macd_hist   = macd_line - macd_signal
            macd_slope  = macd_hist.diff()
            disp  = (close / ma20 - 1.0) * 100

            last = ohlcv.iloc[-1]
            c = float(last["종가"])
            v_z     = float((vol / vol.rolling(20).mean()).iloc[-1])
            rsi_v   = float(rsi14.iloc[-1])
            macd_h  = float(macd_hist.iloc[-1])
            macd_sl = float(macd_slope.iloc[-1])
            m20     = float(ma20.iloc[-1])
            m60     = float(ma60.iloc[-1]) if not np.isnan(ma60.iloc[-1]) else np.nan
            atr     = float(atr14.iloc[-1])
            disp_v  = float(disp.iloc[-1])
            ret5  = (close.pct_change(5 ).iloc[-1] * 100) if len(close) >= 6  else np.nan
            ret10 = (close.pct_change(10).iloc[-1] * 100) if len(close) >= 11 else np.nan

            mkt = "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타")
            name = name_map.get(str(t).zfill(6), "") or stock.get_market_ticker_name(t)
            tv_eok = float(top_df.loc[top_df["종목코드"] == t, "거래대금(원)"].values[0]) / 1e8
            mcap_eok = get_mcap_eok(trade_ymd, t)

            # 컷(개잡주 차단)
            if tv_eok < MIN_TURNOVER_EOK or (not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK):
                continue

            # EBS
            score = 0
            reason = []
            if RSI_LOW <= rsi_v <= RSI_HIGH:
                score += 1; reason.append("RSI 45~65")
            if macd_sl > 0:
                score += 1; reason.append("MACD상승")
            if not np.isnan(disp_v) and -1.0 <= disp_v <= 4.0:
                score += 1; reason.append("MA20 근처")
            if v_z > 1.2:
                score += 1; reason.append("거래량증가")
            if not np.isnan(m20) and not np.isnan(m60) and m20 > m60:
                score += 1; reason.append("상승구조")
            if macd_h > 0:
                score += 1; reason.append("MACD>sig")
            if not np.isnan(ret5) and ret5 < 10:
                score += 1; reason.append("과열아님")

            # 추천가: MA20±0.5ATR 밴드 중심
            band_lo  = m20 - 0.5 * atr
            band_hi  = m20 + 0.5 * atr
            band_mid = (band_lo + band_hi) / 2.0

            buy  = round_to_tick(np.clip(band_mid, band_lo, band_hi))
            stop = max(round_to_tick(band_lo - 0.2 * atr), round_to_tick(m20 * 0.97))
            tgt1 = round_to_tick(band_mid + 1.0 * atr)
            tgt2 = round_to_tick(band_mid + 1.8 * atr)

            rows.append({
                "시장": mkt,
                "종목명": name,
                "종목코드": str(t).zfill(6),
                "종가": int(round(c)),
                "거래대금(억원)": round(tv_eok, 2),
                "시가총액(억원)": None if np.isnan(mcap_eok) else round(mcap_eok, 1),
                "RSI14": round(rsi_v, 1) if not np.isnan(rsi_v) else None,
                "乖離%": round(disp_v, 2) if not np.isnan(disp_v) else None,
                "MACD_hist": round(macd_h, 4) if not np.isnan(macd_h) else None,
                "MACD_slope": round(macd_sl, 5) if not np.isnan(macd_sl) else None,
                "Vol_Z": round(v_z, 2) if not np.isnan(v_z) else None,
                "ret_5d_%": round(ret5, 2) if not np.isnan(ret5) else None,
                "ret_10d_%": round(ret10, 2) if not np.isnan(ret10) else None,
                "EBS": int(score),
                "통과": "초입" if score >= PASS_SCORE else "",
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

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["EBS", "거래대금(억원)"], ascending=[False, False]).reset_index(drop=True)

    ensure_dir(OUT_DIR)
    path_day_utf8    = os.path.join(OUT_DIR, f"recommend_{trade_ymd}.csv")
    path_latest_utf8 = os.path.join(OUT_DIR, "recommend_latest.csv")

    df_out.to_csv(path_day_utf8, index=False, encoding=UTF8)
    df_out.to_csv(path_latest_utf8, index=False, encoding=UTF8)
    log(f"💾 저장 완료: {path_day_utf8} (+ {path_latest_utf8})")

if __name__ == "__main__":
    main()
