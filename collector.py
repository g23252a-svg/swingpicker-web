# collector.py  (pykrx 1.0.51 호환 확정본)
import os
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from pykrx import stock

KST = timezone(timedelta(hours=9))
LOOKBACK_DAYS = 60
TOP_N = 300

def log(msg: str):
    print(f"[{datetime.now(KST)}] {msg}", flush=True)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def pick_top_by_trading_value(end_yyyymmdd: str, top_n: int = TOP_N):
    """
    pykrx 1.0.51에는 '...by_ticker' 계열이 없음.
    -> 모든 티커 목록을 가져와서 '당일 OHLCV(종가*거래량)'로 거래대금 계산 후 TOP N 선정.
    """
    kospi = stock.get_market_ticker_list(end_yyyymmdd, market="KOSPI")
    kosdaq = stock.get_market_ticker_list(end_yyyymmdd, market="KOSDAQ")
    tickers = kospi + kosdaq

    rows = []
    for t in tickers:
        try:
            # 당일 한 날만 조회 (end~end)
            df = stock.get_market_ohlcv_by_date(end_yyyymmdd, end_yyyymmdd, t)
            if df is None or df.empty:
                continue
            close = df["종가"].iloc[-1]
            vol = df["거래량"].iloc[-1]
            tv = float(close) * float(vol)  # 원 단위 거래대금
            rows.append((t, tv))
        except Exception as e:
            log(f"⚠️ {t} 스킵: {e}")
        time.sleep(0.01)  # 과다요청 방지

    tv_df = pd.DataFrame(rows, columns=["종목코드", "거래대금(원)"])
    if tv_df.empty:
        raise RuntimeError("수집된 거래대금 데이터가 없습니다.")

    tv_df["거래대금(억원)"] = (tv_df["거래대금(원)"] / 1e8).round(2)
    tv_df = tv_df.sort_values("거래대금(원)", ascending=False).head(top_n).reset_index(drop=True)
    return tv_df

def load_universe_ohlcv(lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    end = datetime.now(KST)
    start = end - timedelta(days=lookback_days)
    end_s = end.strftime("%Y%m%d")
    start_s = start.strftime("%Y%m%d")

    log("🔍 거래대금 상위 종목 선정 중…")
    top_df = pick_top_by_trading_value(end_s, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ TOP {len(tickers)} 종목 선정 완료")

    frames = []
    for t in tickers:
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty:
                continue
            ohlcv = ohlcv.reset_index().rename(columns={"index": "날짜"})
            ohlcv["종목코드"] = t
            frames.append(ohlcv)
        except Exception as e:
            log(f"⚠️ {t} OHLCV 실패: {e}")
        time.sleep(0.02)

    if not frames:
        raise RuntimeError("OHLCV 수집 결과가 비었습니다.")

    df = pd.concat(frames, ignore_index=True)
    return df

def main():
    log("전종목 수집 시작…")
    df = load_universe_ohlcv(LOOKBACK_DAYS)

    # 저장
    ensure_dir("data")
    today = datetime.now(KST).strftime("%Y%m%d")
    csv_path = f"data/recommend_{today}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 최신 파일 포인터(웹에서 이걸 기본으로 읽게)
    df.to_csv("data/recommend_latest.csv", index=False, encoding="utf-8-sig")

    log(f"💾 저장 완료: {csv_path} (+ data/recommend_latest.csv)")

if __name__ == "__main__":
    main()
