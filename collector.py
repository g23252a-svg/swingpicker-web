import os
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pykrx import stock
import subprocess

# ==============================================
# LDY Pro Collector v2.0 — Full Auto CSV Builder
# ==============================================
LOOKBACK_DAYS = 60
TOP_N = 300
OUTPUT_DIR = "data"

# ----------------------------------------------
# 1️⃣ 거래대금 상위 종목 추출
# ----------------------------------------------
def get_top_trading_value_universe(end: str, top_n: int = TOP_N) -> pd.DataFrame:
    """KOSPI+KOSDAQ 전종목 거래대금 기준 상위 top_n 추출"""
    print(f"[{datetime.now()}] 🔍 거래대금 상위 {top_n} 종목 선정 중...")

    kospi = stock.get_market_ohlcv_by_ticker(end, market="KOSPI")
    kosdaq = stock.get_market_ohlcv_by_ticker(end, market="KOSDAQ")

    kospi = kospi.reset_index().rename(columns={"티커": "종목코드"})
    kosdaq = kosdaq.reset_index().rename(columns={"티커": "종목코드"})
    kospi["시장"] = "KOSPI"
    kosdaq["시장"] = "KOSDAQ"

    df = pd.concat([kospi, kosdaq], ignore_index=True)

    if "거래대금" not in df.columns:
        df["거래대금"] = df["종가"] * df["거래량"]

    df = (
        df[["종목코드", "시장", "거래대금"]]
        .sort_values("거래대금", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    print(f"✅ {len(df)}개 종목 선택 완료")
    return df


# ----------------------------------------------
# 2️⃣ 개별 종목 OHLCV 로딩
# ----------------------------------------------
def fetch_ticker_data(ticker: str, start: str, end: str, market: str) -> pd.DataFrame:
    """개별 종목 과거 OHLCV"""
    try:
        df = stock.get_market_ohlcv_by_date(start, end, ticker)
        df = df.reset_index().rename(columns={"날짜": "날짜"})
        df["종목코드"] = ticker
        df["시장"] = market
        return df
    except Exception as e:
        print(f"❌ {ticker} 실패: {e}")
        return pd.DataFrame()


# ----------------------------------------------
# 3️⃣ 병렬 수집 및 통합
# ----------------------------------------------
def load_universe_ohlcv(lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """전종목 OHLCV 수집"""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=lookback_days*2)).strftime("%Y%m%d")

    top_df = get_top_trading_value_universe(end)
    tickers = top_df["종목코드"].tolist()
    markets = dict(zip(top_df["종목코드"], top_df["시장"]))

    results = []

    print(f"[{datetime.now()}] ⚙️ 병렬 데이터 수집 시작 ({len(tickers)}개 종목)...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_ticker_data, t, start, end, markets[t]) for t in tickers]
        for f in as_completed(futures):
            df = f.result()
            if not df.empty:
                results.append(df)

    if not results:
        print("⚠️ 데이터가 없습니다.")
        return pd.DataFrame()

    df_all = pd.concat(results, ignore_index=True)
    if "거래대금" in df_all.columns:
        df_all["거래대금(억원)"] = (df_all["거래대금"] / 1e8).round(2)
elif "거래대금(원)" in df_all.columns:
        df_all["거래대금(억원)"] = (df_all["거래대금(원)"] / 1e8).round(2)
else:
    print("⚠️ 거래대금 컬럼이 감지되지 않았습니다.")
    df_all["거래대금(억원)"] = np.nan
    print(f"✅ {len(df_all)}행 데이터 수집 완료")
    return df_all


# ----------------------------------------------
# 4️⃣ 결과 저장 및 git push
# ----------------------------------------------
def save_and_push(df: pd.DataFrame):
    """CSV 저장 및 main 브랜치로 push"""
    if df.empty:
        print("❌ 저장할 데이터 없음.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    fname = f"recommend_{datetime.now().strftime('%Y%m%d')}.csv"
    path = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"💾 저장 완료: {path}")

    # git push
    try:
        subprocess.run(["git", "config", "--global", "user.email", "github-actions@github.com"])
        subprocess.run(["git", "config", "--global", "user.name", "github-actions"])
        subprocess.run(["git", "add", path])
        subprocess.run(["git", "commit", "-m", f"Auto update: {fname}"])
        subprocess.run(["git", "push"])
        print("🚀 GitHub Push 완료.")
    except Exception as e:
        print(f"⚠️ Git push 실패: {e}")


# ----------------------------------------------
# 5️⃣ 메인 실행부
# ----------------------------------------------
def main():
    print(f"[{datetime.now()}] 전종목 수집 시작…")
    df = load_universe_ohlcv(LOOKBACK_DAYS)
    save_and_push(df)
    print("✅ 모든 프로세스 완료.")


if __name__ == "__main__":
    main()
