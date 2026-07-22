# -*- coding: utf-8 -*-
"""
scripts/collect_us_daily.py

[v36] data/us_daily.csv 수집기 — 나스닥100(NDX)·S&P500 일봉.

배경(실측): 코스피는 나스닥과 강하게 연동(반도체 비중 ~30% + 외국인 수급 +
나스닥 선물 24h 거래). 야간 배치는 그날 한국 종가로 돌기 때문에 '전일 미국
마감'이 다음날 한국장에 미치는 오버나이트 효과가 알파 피처 후보다.
alpha_engine이 이 CSV를 날짜 조인(look-ahead 방지: 한국일 d에는 us_date<d만)
해 피처로 쓰며, 워크포워드 검증 게이트가 가치를 자동 판정한다.

출력: data/us_daily.csv — date(미국 달력일 YYYYMMDD), ndx_close, spx_close,
      rut_close([v37.3] 러셀2000 — 분석용, 알파 피처는 ndx/spx만 사용)

실행 (GitHub Actions — collect_kospi_daily.yml에서 KOSPI 수집 후):
    python scripts/collect_us_daily.py [--start YYYYMMDD] [--end YYYYMMDD]

실패 정책: FDR → yfinance → stooq CSV 3단 폴백. 전부 실패해도 exit 0
(기존 us_daily.csv 유지 — 배치를 깨지 않는다. KOSPI 수집과 독립).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

_logger = logging.getLogger("collect_us_daily")

# (심볼셋: FDR 티커, yfinance 티커, stooq 티커)
_SYMBOLS = {
    "ndx_close": ("NDX", "^NDX", "^ndx"),      # 나스닥 100
    "spx_close": ("US500", "^GSPC", "^spx"),   # S&P 500
    # [v37.3] 러셀 2000 — 미국 소형주. 지수별 오버나이트 상관 분석용
    # (추가 컬럼은 outer-merge라 기존 ndx/spx 피처 파이프라인에 영향 없음).
    "rut_close": ("RUT", "^RUT", "^rut"),
}


def _fetch_fdr(ticker: str, start: str, end: str):
    try:
        import FinanceDataReader as fdr
        import pandas as pd
    except ImportError:
        return None
    try:
        df = fdr.DataReader(ticker, start, end)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        return df[["date", "close"]]
    except Exception as e:
        _logger.warning(f"FDR({ticker}) 실패: {e}")
        return None


def _fetch_yfinance(ticker: str, start: str, end: str):
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        return None
    try:
        s = datetime.strptime(start, "%Y%m%d").strftime("%Y-%m-%d")
        e = datetime.strptime(end, "%Y%m%d").strftime("%Y-%m-%d")
        df = yf.download(ticker, start=s, end=e, progress=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df.columns = [
            str(c).lower() if not isinstance(c, tuple) else str(c[0]).lower()
            for c in df.columns
        ]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        return df[["date", "close"]]
    except Exception as e:
        _logger.warning(f"yfinance({ticker}) 실패: {e}")
        return None


def _fetch_stooq(ticker: str, start: str, end: str):
    """stooq.com 일봉 CSV (최후 폴백 — 무인증 공개 엔드포인트)."""
    try:
        import io
        import urllib.parse
        import urllib.request
        import pandas as pd
    except ImportError:
        return None
    try:
        url = (
            "https://stooq.com/q/d/l/?s="
            + urllib.parse.quote(ticker)
            + f"&i=d&d1={start}&d2={end}"
        )
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw))
        if df is None or df.empty or "Close" not in df.columns:
            return None
        df = df.rename(columns={"Date": "date", "Close": "close"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        return df[["date", "close"]]
    except Exception as e:
        _logger.warning(f"stooq({ticker}) 실패: {e}")
        return None


def _fetch_symbol(fdr_t: str, yf_t: str, stooq_t: str, start: str, end: str):
    for fetch, ticker, src in (
        (_fetch_fdr, fdr_t, "FDR"),
        (_fetch_yfinance, yf_t, "yfinance"),
        (_fetch_stooq, stooq_t, "stooq"),
    ):
        df = fetch(ticker, start, end)
        if df is not None and len(df) > 0:
            _logger.info(f"📥 {ticker} 수집 성공 ({src}) — {len(df)}일")
            return df
    return None


def merge_us_frames(frames: dict) -> "object":
    """심볼별 (date, close) 프레임들을 date 기준 outer 병합.

    Returns: date 오름차순 DataFrame (date, ndx_close, spx_close ...) 또는 None.
    수집된 심볼이 하나도 없으면 None.
    """
    import pandas as pd

    out = None
    for col, df in frames.items():
        if df is None or df.empty:
            continue
        part = df.rename(columns={"close": col})[["date", col]]
        out = part if out is None else out.merge(part, on="date", how="outer")
    if out is None:
        return None
    return out.sort_values("date").reset_index(drop=True)


def collect_us_daily(start: str, end: str, output_path: str) -> bool:
    frames = {}
    for col, (fdr_t, yf_t, stooq_t) in _SYMBOLS.items():
        frames[col] = _fetch_symbol(fdr_t, yf_t, stooq_t, start, end)

    merged = merge_us_frames(frames)
    if merged is None or merged.empty:
        _logger.error("미국 지수 수집 실패 — FDR/yfinance/stooq 모두 실패")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    _logger.info(f"💾 저장 완료: {output_path} ({len(merged)}일)")
    print(f"\n[US daily 수집 요약]")
    print(f"  기간: {merged['date'].iloc[0]} ~ {merged['date'].iloc[-1]}")
    print(f"  거래일 수: {len(merged)} · 컬럼: {list(merged.columns)}")
    return True


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=None, help="시작일 YYYYMMDD (기본: 5년 전)")
    p.add_argument("--end", default=None, help="종료일 YYYYMMDD (기본: 오늘)")
    p.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "us_daily.csv"),
        help="출력 경로",
    )
    args = p.parse_args()

    today = datetime.now()
    end = args.end or today.strftime("%Y%m%d")
    start = args.start or (today - timedelta(days=365 * 5)).strftime("%Y%m%d")

    ok = collect_us_daily(start, end, args.output)
    if not ok:
        # 배치를 깨지 않는다 — 기존 us_daily.csv가 있으면 그대로 사용됨.
        _logger.warning("US daily 수집 실패 — 기존 파일 유지, exit 0")
    sys.exit(0)


if __name__ == "__main__":
    main()
