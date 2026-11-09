"""
scripts/collect_kospi_daily.py
==============================
[v3.9.15e + 1] data/kospi_daily.csv 수집기.

목적:
- _calc_kospi_alpha()의 real 알파 경로 활성화
- 슬라이더 1~60일 전 구간에서 정확 알파 산출

출력: data/kospi_daily.csv
컬럼: date(YYYYMMDD), close, ret_1d_%, ret_3d_%, ret_5d_%, ret_10d_%, ret_20d_%, ret_60d_%

수익률 계산: df["close"].pct_change(n).shift(-n) * 100
  → t일 행에 t→t+n 수익률 기록
  → 백테스트 시 rec_date에서 바로 lookup 가능
  → lookahead bias는 백테스트 검증용이라 OK (라이브 추천엔 사용 안 함)

실행:
    python scripts/collect_kospi_daily.py [--start YYYYMMDD] [--end YYYYMMDD]

[v86] 야간 배치(auto_collect.yml)가 collector 직전에 --end <대상 세션>으로 부른다 —
  risk_off가 그날 종가로 판정하게(예전엔 배치 뒤에 갱신돼 33/33 배치가 하루 묵은 종가였다).

크론 권장 (매일 장마감 후):
    20 16 * * 1-5 cd /path/to/swingpicker && python scripts/collect_kospi_daily.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)

# 보유기간 정책 SSOT — services/benchmarks.py의 _BENCH_HOLD_MAP과 일치해야 함
_HOLD_PERIODS = [1, 3, 5, 10, 20, 60]


def collect_kospi_daily(start: str, end: str, output_path: str) -> bool:
    """KOSPI 일자별 수익률 수집.
    
    Args:
        start: 시작일 YYYYMMDD
        end:   종료일 YYYYMMDD
        output_path: 저장 경로
    
    Returns: 성공 여부
    """
    try:
        import pandas as pd
    except ImportError:
        _logger.error("pandas 미설치 — pip install pandas")
        return False

    # 1순위: pykrx (가장 정확)
    df = _fetch_pykrx(start, end)
    src = "pykrx"

    # 2순위: FinanceDataReader
    if df is None:
        df = _fetch_fdr(start, end)
        src = "FDR"

    # 3순위: yfinance
    if df is None:
        df = _fetch_yfinance(start, end)
        src = "yfinance"

    if df is None or df.empty:
        _logger.error("KOSPI 데이터 수집 실패 — pykrx / FDR / yfinance 모두 실패")
        return False

    _logger.info(f"📥 KOSPI 수집 성공 ({src}) — {len(df)}일 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")

    # [v86] end 이후 행은 버린다 — 배치(auto_collect)는 --end <대상 세션>으로 부른다.
    df = df[df["date"].astype(str) <= str(end)].reset_index(drop=True)
    if df.empty:
        _logger.error(f"KOSPI 수집 결과가 end={end} 이전에 없음")
        return False

    # [v86] 기존 파일보다 묵은 결과로 덮어쓰지 않는다. 9/18~10/13엔 수집이 조용히
    #   20260917에서 멈췄는데도 매일 '성공'으로 커밋됐다. 새 결과의 마지막 날짜가 기존보다
    #   이르면 기존 파일을 유지하고 실패로 알린다(워크플로가 신선도 검사로 잡는다).
    prev_last = _existing_last_date(output_path)
    new_last = str(df["date"].iloc[-1])
    if prev_last and new_last < prev_last:
        _logger.error(f"KOSPI 수집 결과가 기존보다 묵음 (새 {new_last} < 기존 {prev_last}) — 덮어쓰지 않음")
        return False

    # 보유기간별 수익률 계산 (t행에 t→t+n 수익률 기록)
    for n in _HOLD_PERIODS:
        df[f"ret_{n}d_%"] = df["close"].pct_change(n).shift(-n) * 100

    # CSV 저장 (utf-8-sig — Excel 호환)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    _logger.info(f"💾 저장 완료: {output_path}")

    # 요약 출력
    print(f"\n[KOSPI daily 수집 요약]")
    print(f"  소스: {src}")
    print(f"  기간: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(f"  거래일 수: {len(df)}")
    print(f"  컬럼: {list(df.columns)}")
    for n in _HOLD_PERIODS:
        col = f"ret_{n}d_%"
        valid = df[col].notna().sum()
        print(f"  {col}: {valid}일 유효 ({df[col].mean():+.2f}% 평균)")
    return True


def _existing_last_date(path: str):
    """[v86] 기존 kospi_daily.csv의 마지막 date(YYYYMMDD). 없거나 못 읽으면 None."""
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd
        d = pd.read_csv(path, usecols=["date"], dtype={"date": str})["date"].dropna()
        return str(d.str.replace("-", "").str[:8].max()) if len(d) else None
    except Exception as e:
        _logger.warning(f"기존 kospi_daily 읽기 실패: {e}")
        return None


def _fetch_pykrx(start: str, end: str):
    """pykrx로 KOSPI 종가 수집."""
    try:
        from pykrx import stock
        import pandas as pd
    except ImportError:
        _logger.info("pykrx 미설치 — 폴백 시도")
        return None
    try:
        df = stock.get_index_ohlcv(start, end, "1001")  # KOSPI
        if df is None or df.empty:
            return None
        df = df.reset_index().rename(columns={"날짜": "date", "종가": "close"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        return df[["date", "close"]]
    except Exception as e:
        _logger.warning(f"pykrx 실패: {e}")
        return None


def _fetch_fdr(start: str, end: str):
    """FinanceDataReader로 KOSPI 종가 수집 (폴백)."""
    try:
        import FinanceDataReader as fdr
        import pandas as pd
    except ImportError:
        return None
    try:
        df = fdr.DataReader("KS11", start, end)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        return df[["date", "close"]]
    except Exception as e:
        _logger.warning(f"FDR 실패: {e}")
        return None


def _fetch_yfinance(start: str, end: str):
    """yfinance로 KOSPI 종가 수집 (최후 폴백)."""
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        return None
    try:
        s = datetime.strptime(start, "%Y%m%d").strftime("%Y-%m-%d")
        # [v86] yfinance의 end는 미포함 — end 당일 종가를 받으려면 하루 뒤로 넘긴다.
        e = (datetime.strptime(end, "%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download("^KS11", start=s, end=e, progress=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df.columns = [str(c).lower() if not isinstance(c, tuple) else str(c[0]).lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        return df[["date", "close"]]
    except Exception as e:
        _logger.warning(f"yfinance 실패: {e}")
        return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--start", default=None, help="시작일 YYYYMMDD (기본: 5년 전)")
    p.add_argument("--end", default=None, help="종료일 YYYYMMDD (기본: 오늘)")
    p.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kospi_daily.csv"),
        help="출력 경로",
    )
    args = p.parse_args()

    today = datetime.now()
    end = args.end or today.strftime("%Y%m%d")
    # [v86] 기본 시작일은 end 기준 5년 전(같은 창 — --end로 과거를 다시 받아도 길이 동일)
    start = args.start or (datetime.strptime(end, "%Y%m%d") - timedelta(days=365 * 5)).strftime("%Y%m%d")

    ok = collect_kospi_daily(start, end, args.output)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
