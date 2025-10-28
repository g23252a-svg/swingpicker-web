import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pykrx import stock

# ---------------------------------------------------------------------
# ⚙️ 설정
# ---------------------------------------------------------------------
KST = timezone(timedelta(hours=9))
TODAY = datetime.now(KST).date()
DATA_DIR = "data"
LOOKBACK_DAYS = 30  # 최근 30일 기준
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 🧩 OHLCV 수집 함수
# ---------------------------------------------------------------------
def get_ohlcv(ticker: str, start: str, end: str):
    try:
        df = stock.get_market_ohlcv_by_date(start, end, ticker)
        df["티커"] = ticker
        return df
    except Exception as e:
        print(f"❌ {ticker} 수집 실패: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------------------
# 📊 거래대금 상위 300종목 추출
# ---------------------------------------------------------------------
def load_universe_ohlcv(lookback_days: int = 30):
    end = TODAY.strftime("%Y%m%d")
    start = (TODAY - timedelta(days=lookback_days)).strftime("%Y%m%d")

    print(f"[{datetime.now(KST)}] 전종목 수집 시작…")
    print(f"[{datetime.now(KST)}] 🔍 거래대금 상위 300 종목 선정 중...")

    # ✅ 최신 pykrx 버전용 함수 (get_market_trading_value_by_date)
    df_all = stock.get_market_trading_value_by_date(end, market="ALL")
    df_all = df_all.reset_index()

    # 거래대금 컬럼 정리
    if "거래대금" in df_all.columns:
        df_all["거래대금(억원)"] = (df_all["거래대금"] / 1e8).round(2)
    else:
        print("⚠️ 거래대금 컬럼이 감지되지 않아 0 처리")
        df_all["거래대금(억원)"] = 0

    df_ranked = (
        df_all.sort_values("거래대금(억원)", ascending=False)
        .head(300)
        .reset_index(drop=True)
    )

    tickers = df_ranked["티커"].tolist()
    print(f"✅ {len(tickers)}개 종목 선택 완료")

    # 병렬 OHLCV 수집
    ohlcv_list = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_ohlcv, t, start, end): t for t in tickers}
        for f in as_completed(futures):
            result = f.result()
            if not result.empty:
                ohlcv_list.append(result)

    df_merged = pd.concat(ohlcv_list)
    df_merged.reset_index(inplace=True)
    df_merged.rename(columns={"index": "날짜"}, inplace=True)

    print(f"✅ {len(df_merged)}행 데이터 수집 완료")
    return df_merged

# ---------------------------------------------------------------------
# 💹 매수 신호 로직
# ---------------------------------------------------------------------
def generate_recommendations(df: pd.DataFrame):
    result = []
    for ticker, grp in df.groupby("티커"):
        grp = grp.sort_values("날짜")
        if len(grp) < 10:
            continue

        ma5 = grp["종가"].rolling(5).mean().iloc[-1]
        ma20 = grp["종가"].rolling(20).mean().iloc[-1]
        last_close = grp["종가"].iloc[-1]

        # 매수 조건: 단기 상향 돌파 + 거래량 증가
        if ma5 > ma20 and grp["거래량"].iloc[-1] > grp["거래량"].iloc[-2]:
            result.append(
                {
                    "티커": ticker,
                    "종목명": stock.get_market_ticker_name(ticker),
                    "종가": int(last_close),
                    "추천매수가": round(last_close * 0.99, 1),
                    "추천사유": "5일선 상향돌파 + 거래량 증가",
                }
            )

    return pd.DataFrame(result)

# ---------------------------------------------------------------------
# 🧾 메인 실행부
# ---------------------------------------------------------------------
def main():
    df = load_universe_ohlcv(LOOKBACK_DAYS)
    if df.empty:
        print("❌ 데이터 수집 실패. 종료합니다.")
        return

    df_rec = generate_recommendations(df)

    save_path = os.path.join(DATA_DIR, f"recommend_{TODAY.strftime('%Y%m%d')}.csv")
    df_rec.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"💾 저장 완료: {save_path}")
    print(f"🚀 총 추천 종목 수: {len(df_rec)}개")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
