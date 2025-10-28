# -*- coding: utf-8 -*-
"""
LDY Pro Trader — Collector (KRX Full Universe)
매일 장마감 후 실행 → 전종목 지표계산 → 급등초입 스코어링 → 추천리스트 CSV 저장

출력: daily_output/recommend_YYYYMMDD.csv
"""

import os
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from pykrx import stock

# ===== 공통 =====
KST = timezone(timedelta(hours=9))

def now_kst():
    return datetime.now(KST)

def ymd(d=None):
    d = d or now_kst()
    return d.strftime("%Y%m%d")

# ===== 지표 =====
def rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd_hist(close, fast=12, slow=26, sig=9):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig).mean()
    return macd - signal

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def zscore(x, window=20):
    return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-9)

# ===== 파라미터 (원하면 수정 가능) =====
LOOKBACK_DAYS = 120            # 지표 계산용 과거기간(거래일 기준 넉넉히)
TURNOVER_MIN_EOK = 50          # 거래대금 하한(억원)
MCAP_MIN_EOK = 1000            # (선택) 시총 하한(억원) — pykrx로 당일 시총 일괄 산출 어려워 임시 NaN 처리
RSI_RANGE = (40, 70)           # RSI 허용구간
MA20_DEV_RANGE = (0, 10)       # MA20乖離 % 범위(0~10%: 그라인딩 초기)
VOLZ_MIN = 1.5                 # 거래량 Z-score 하한 (최근 20일 대비 급증)
RET5_MAX = 8.0                 # 5일 수익률 상한(과열 방지)
RET10_MAX = 15.0               # 10일 수익률 상한(과열 방지)
TOP_N_TURNOVER = 1200          # 거래대금 상위 N 종목만 정밀 스캔(속도 최적화)
SCORE_PASS = 4                 # 통과 최소 점수
MAX_PICK = 60                  # 최종 추천 최대 개수

SAVE_DIR = "daily_output"

# ===== 데이터 수집 =====
def fetch_ohlcv_range(code: str, start_ymd: str, end_ymd: str, kospi_set: set):
    try:
        df = stock.get_market_ohlcv_by_date(start_ymd, end_ymd, code)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df["종목코드"] = code
        df["종목명"] = stock.get_market_ticker_name(code)
        df["시장"] = "KOSPI" if code in kospi_set else "KOSDAQ"

        # 거래대금(억원) 산출
        if "거래대금" in df.columns:
            df["거래대금(억원)"] = (df["거래대금"] / 1e8).round(2)
        else:
            df["거래대금(억원)"] = np.nan

        # (선택) 시총 칼럼 자리 — pykrx로 과거 일자 시총 일괄 호출은 API비용 커서 일단 NaN
        df["시가총액(억원)"] = np.nan

        # 표준 컬럼셋
        return df[["날짜","시장","종목명","종목코드","시가","고가","저가","종가","거래량","거래대금(억원)","시가총액(억원)"]]
    except Exception:
        return pd.DataFrame()

def load_universe_ohlcv(lookback_days=LOOKBACK_DAYS) -> pd.DataFrame:
    end = ymd()
    start = ymd(now_kst() - timedelta(days=int(lookback_days*1.7)))  # 휴일 고려 넉넉히
    kospi = stock.get_market_ticker_list(market="KOSPI")
    kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
    kospi_set = set(kospi)
    tickers = kospi + kosdaq

    # 먼저 당일 거래대금 상위 컷을 위해 today 기준 스냅샷
    today_df = stock.get_market_trading_value_by_ticker(end, market="ALL")  # 거래대금 상위 선별
    if today_df is None or today_df.empty:
        # 백업: 전종목 강행
        candidates = tickers
    else:
        # 거래대금 상위 티커 리스트
        tv = today_df.sort_values("거래대금", ascending=False).head(TOP_N_TURNOVER)
        candidates = [t for t in tv.index.tolist() if t in tickers]

    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(fetch_ohlcv_range, c, start, end, kospi_set) for c in candidates]
        for f in as_completed(futs):
            r = f.result()
            if not r.empty:
                results.append(r)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

# ===== 지표계산 & 스코어링 =====
def enrich_and_score(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for code, g in df.groupby("종목코드"):
        g = g.sort_values("날짜").copy()
        close = g["종가"]

        g["MA20"] = close.rolling(20).mean()
        g["乖離%"] = (g["종가"]/(g["MA20"]+1e-9) - 1) * 100.0
        g["RSI14"] = rsi(close, 14)
        g["MACD_hist"] = macd_hist(close)
        g["MACD_slope"] = g["MACD_hist"].diff()
        g["ATR14"] = atr(g["고가"], g["저가"], close, 14)
        g["Vol_Z"] = zscore(g["거래량"], 20)

        # 수익률
        g["ret_5d_%"] = (close / close.shift(5) - 1) * 100.0
        g["ret_10d_%"] = (close / close.shift(10) - 1) * 100.0

        out.append(g)

    df2 = pd.concat(out, ignore_index=True)

    # 스냅샷(마지막 일자)
    snap = df2.sort_values("날짜").groupby(["시장","종목코드","종목명"]).tail(1).copy()

    # 필터링 기본
    snap = snap[(snap["거래대금(억원)"] >= TURNOVER_MIN_EOK)]
    # 시총이 NaN이면 통과시키고 싶다면 다음 줄 주석 해제해서 NaN→큰 값으로 대체
    snap["시가총액(억원)"] = snap["시가총액(억원)"]  # .fillna(MCAP_MIN_EOK*2)

    # 점수화 (Early Breakout Score)
    snap["EBS"] = 0
    snap.loc[snap["MACD_hist"] > 0, "EBS"] += 1
    snap.loc[snap["MACD_slope"] > 0, "EBS"] += 1
    snap.loc[snap["RSI14"].between(*RSI_RANGE), "EBS"] += 1
    snap.loc[snap["乖離%"].between(*MA20_DEV_RANGE), "EBS"] += 1
    snap.loc[snap["Vol_Z"] >= VOLZ_MIN, "EBS"] += 1
    snap.loc[snap["ret_5d_%"] <= RET5_MAX, "EBS"] += 1
    snap.loc[snap["ret_10d_%"] <= RET10_MAX, "EBS"] += 1

    # 추천 매수/매도/손절 가격 (보수적 버전)
    # - 추천매수가: min(종가, MA20*0.99, 종가-0.5*ATR)
    # - 1차청산: 종가 + 1.0*ATR
    # - 2차청산: MA20*1.05 또는 종가 + 2.0*ATR 중 더 낮은 값
    # - 손절: 종가 - 1.2*ATR
    snap["추천매수가"] = np.minimum.reduce([
        snap["종가"],
        (snap["MA20"] * 0.99),
        (snap["종가"] - 0.5 * snap["ATR14"])
    ]).round(0)

    tp1 = (snap["종가"] + 1.0 * snap["ATR14"]).round(0)
    tp2_ma = (snap["MA20"] * 1.05).round(0)
    tp2_atr = (snap["종가"] + 2.0 * snap["ATR14"]).round(0)
    snap["추천매도가1"] = tp1
    snap["추천매도가2"] = np.minimum(tp2_ma, tp2_atr)

    snap["손절가"] = (snap["종가"] - 1.2 * snap["ATR14"]).round(0)

    # 정렬
    snap = snap.sort_values(["EBS","거래대금(억원)","Vol_Z"], ascending=[False,False,False])

    # 최종 컷
    picks = snap[snap["EBS"] >= SCORE_PASS].head(MAX_PICK).copy()

    # 보기 좋은 컬럼 정리
    cols = [
        "시장","종목명","종목코드","종가","거래대금(억원)","시가총액(억원)",
        "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%",
        "EBS","추천매수가","추천매도가1","추천매도가2","손절가"
    ]
    for c in ["RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"]:
        if c in picks.columns:
            picks[c] = picks[c].round(2)

    return picks[cols], snap[cols]

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    date_str = ymd()
    print(f"[{date_str}] 전종목 수집 시작…")

    df = load_universe_ohlcv(LOOKBACK_DAYS)
    if df.empty:
        raise RuntimeError("전종목 OHLCV 수집 실패(네트워크/휴일/차단 등). 로컬/서버 네트워크 확인 요망.")

    picks, universe_snap = enrich_and_score(df)

    # 저장
    out_file = os.path.join(SAVE_DIR, f"recommend_{date_str}.csv")
    picks.to_csv(out_file, index=False, encoding="utf-8-sig")

    # 참고용: 전체 스냅샷도 보관하고 싶으면 아래 활성화
    # universe_file = os.path.join(SAVE_DIR, f"universe_{date_str}.csv")
    # universe_snap.to_csv(universe_file, index=False, encoding="utf-8-sig")

    print(f"✅ {len(picks)}개 종목 저장 완료: {out_file}")

if __name__ == "__main__":
    main()
