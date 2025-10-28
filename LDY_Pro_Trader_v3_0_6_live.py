import streamlit as st
import pandas as pd
import numpy as np
import io, time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pykrx import stock

# =========================
# 기본 세팅
# =========================
st.set_page_config(page_title="LDY Pro Trader v3.0.6 (Live FullScan)", layout="wide")
st.title("📈 LDY Pro Trader v3.0.6 (Live FullScan)")
st.caption("KOSPI+KOSDAQ 전종목 급등 초입 자동 스캐너 | Made by LDY")

KST = timezone(timedelta(hours=9))
def effective_ymd(use_prev_close: bool) -> str:
    now = datetime.now(KST)
    roll = now.replace(hour=9, minute=5, second=0, microsecond=0)
    base = (now.date() - timedelta(days=1)) if (use_prev_close or now < roll) else now.date()
    return base.strftime("%Y%m%d")

# =========================
# 지표 계산 함수
# =========================
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

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ 스캔 조건 (전종목용)")
lookback = st.sidebar.number_input("조회일수", 20, 252, 60)
turnover_min = st.sidebar.number_input("거래대금 하한 (억원)", 0, 5000, 50)
mcap_min = st.sidebar.number_input("시총 하한 (억원)", 0, 1000000, 1000)
rsi_min = st.sidebar.number_input("RSI 하한", 0, 100, 45)
rsi_max = st.sidebar.number_input("RSI 상한", 0, 100, 65)
score_pass = st.sidebar.number_input("통과점수", 0, 7, 4)
use_prev_close = st.sidebar.checkbox("전일 기준(장마감)", True)
st.sidebar.divider()

# =========================
# 전종목 데이터 수집
# =========================
st.info("📊 KOSPI + KOSDAQ 전종목 불러오는 중 (약 1~2분 소요)...")

@st.cache_data(ttl=1800)
def load_full_ohlcv(lookback):
    end = datetime.now(KST).strftime("%Y%m%d")
    start = (datetime.now(KST) - timedelta(days=lookback * 1.5)).strftime("%Y%m%d")
    tickers = stock.get_market_ticker_list("KOSPI") + stock.get_market_ticker_list("KOSDAQ")
    results = []

    def fetch(code):
        try:
            df = stock.get_market_ohlcv_by_date(start, end, code)
            df["종목명"] = stock.get_market_ticker_name(code)
            df["종목코드"] = code
            df["시장"] = "KOSPI" if code in stock.get_market_ticker_list("KOSPI") else "KOSDAQ"
            return df.reset_index()
        except:
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch, code) for code in tickers]
        for f in as_completed(futures):
            r = f.result()
            if not r.empty:
                results.append(r)

    df_all = pd.concat(results, ignore_index=True)
    return df_all

df_raw = load_full_ohlcv(lookback)

st.success(f"✅ 총 {df_raw['종목코드'].nunique()}개 종목 데이터 수집 완료")

# =========================
# 지표 계산 및 필터링
# =========================
def enrich(df):
    out = []
    for code, g in df.groupby("종목코드"):
        g = g.sort_values("날짜")
        g["MA20"] = g["종가"].rolling(20).mean()
        g["乖離%"] = (g["종가"] / g["MA20"] - 1) * 100
        g["RSI14"] = rsi(g["종가"], 14)
        g["MACD_hist"] = macd_hist(g["종가"])
        g["MACD_slope"] = g["MACD_hist"].diff()
        g["ATR14"] = atr(g["고가"], g["저가"], g["종가"], 14)
        g["Vol_Z"] = zscore(g["거래량"], 20)
        out.append(g)
    return pd.concat(out, ignore_index=True)

df = enrich(df_raw)
snap = df.sort_values("날짜").groupby(["시장","종목코드","종목명"]).tail(1)

# =========================
# 점수화 및 조건 필터
# =========================
snap["EBS"] = 0
snap.loc[snap["MACD_hist"] > 0, "EBS"] += 1
snap.loc[snap["MACD_slope"] > 0, "EBS"] += 1
snap.loc[snap["RSI14"].between(rsi_min, rsi_max), "EBS"] += 1
snap.loc[snap["乖離%"].between(0, 10), "EBS"] += 1
snap.loc[snap["Vol_Z"] >= 1.5, "EBS"] += 1

picked = snap[
    (snap["거래대금"] / 1e8 >= turnover_min) &
    (snap["EBS"] >= score_pass)
].sort_values(["EBS","거래대금"], ascending=[False,False])

st.success(f"🔥 급등 초입 후보 {len(picked)}개 종목 발견!")

st.dataframe(picked[["시장","종목명","종목코드","종가","거래량","거래대금","乖離%","RSI14","MACD_hist","EBS"]], use_container_width=True)

csv = picked.to_csv(index=False, encoding="utf-8-sig")
st.download_button("📥 CSV 다운로드", data=csv, file_name=f"swingpicker_full_{effective_ymd(use_prev_close)}.csv", mime="text/csv")
