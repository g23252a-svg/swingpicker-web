import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pykrx import stock

KST = timezone(timedelta(hours=9))
LOOKBACK_DAYS = 60

def log(msg):
    now = datetime.now(KST)
    print(f"[{now}] {msg}")

def load_universe_ohlcv(lookback_days=60):
    """거래대금 상위 300종목 선정 후 OHLCV 병합"""
    end = datetime.now(KST).strftime("%Y%m%d")
    start = (datetime.now(KST) - timedelta(days=lookback_days)).strftime("%Y%m%d")

    log("🔍 거래대금 상위 300 종목 선정 중...")

    # ✅ 최신 pykrx 호환 코드
    kospi = stock.get_market_trading_value_by_date(end, end, "KOSPI")
    kosdaq = stock.get_market_trading_value_by_date(end, end, "KOSDAQ")

    df_all = pd.concat([kospi, kosdaq])

    # 열 이름이 다를 경우 자동 대응
    if "거래대금" in df_all.columns:
        df_all["거래대금(억원)"] = (df_all["거래대금"] / 1e8).round(2)
    elif "거래대금(원)" in df_all.columns:
        df_all["거래대금(억원)"] = (df_all["거래대금(원)"] / 1e8).round(2)
    else:
        raise KeyError("❌ 거래대금 컬럼을 찾을 수 없습니다.")

    df_all = df_all.sort_values("거래대금(억원)", ascending=False).head(300)
    tickers = df_all.index.to_list()
    log("✅ 300개 종목 선택 완료")

    dfs = []
    for t in tickers:
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start, end, t)
            ohlcv["종목코드"] = t
            dfs.append(ohlcv)
        except Exception as e:
            log(f"⚠️ {t} 수집 실패: {e}")
        time.sleep(0.1)

    df_all = pd.concat(dfs)
    log(f"📊 총 {len(df_all)}개 데이터 수집 완료")
    return df_all

def main():
    log("전종목 수집 시작…")
    df = load_universe_ohlcv(LOOKBACK_DAYS)
    today = datetime.now(KST).strftime("%Y%m%d")
    out_path = f"recommend_{today}.csv"
    df.to_csv(out_path, encoding="utf-8-sig")
    log(f"💾 {out_path} 저장 완료")

if __name__ == "__main__":
    main()
