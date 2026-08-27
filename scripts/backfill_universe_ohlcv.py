# -*- coding: utf-8 -*-
"""backfill_universe_ohlcv.py — 생존편향 없는 유니버스 시세 백필 [v71]

## 무엇을 하는가

`services.universe_history` 가 복원한 시점별 유니버스(2,841종목, 소멸 확인 46)의
일봉을 종목당 하나의 parquet으로 `data/universe_ohlcv/<code>.parquet` 에 채운다.
기존 `data/ohlcv_cache_*.parquet`(거래대금 상위 600 기준, 574종목, 소멸 흔적 0건)로는
편입 게이트를 검정할 수 없기 때문이다.

## 왜 야간 배치에 물리지 않는가

2,267종목을 매일 받으면 배치 시간이 폭증하고, 실패 시 추천 산출 자체가 위험해진다.
이건 **일회성 백필 + 가끔 증분**이지 매일 할 일이 아니다. 그래서 독립 스크립트다.
`pipeline_finalize` 에서 부르지 않는다.

## 재개 가능성

`_manifest.json` 에 종목별 상태(ok / empty / error / delisted_no_data)와 마지막
시도 시각·행수를 남긴다. 다시 돌리면 ok 인 종목은 건너뛴다. 실패만 다시 하려면
`--retry-failed`.

## 사용

    python scripts/backfill_universe_ohlcv.py --start 2025-02-01 --limit 200
    python scripts/backfill_universe_ohlcv.py --retry-failed
    python scripts/backfill_universe_ohlcv.py --status

네트워크가 이그레스 정책으로 막힌 환경에서는 즉시 실패한다(우회하지 않는다).
그 경우 `--status` 로 남은 작업량만 확인하고, 배치 환경에서 돌린다.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import universe_history as uh  # noqa: E402

logger = logging.getLogger("backfill")

STORE_DIR = "universe_ohlcv"
MANIFEST = "_manifest.json"
DEFAULT_START = "2025-02-01"

#: 기존 ohlcv_cache 와 같은 스키마로 맞춘다 — 두 소스를 한 코드로 읽기 위해서.
OUT_COLS = ["종목코드", "시가", "고가", "저가", "종가", "거래량", "등락률"]

ST_OK = "ok"
ST_EMPTY = "empty"
ST_ERROR = "error"

#: 등락률 단위. 출처가 선언한다 — 값 크기로 추측하지 않는다.
#: (FDR의 Change는 0.0123 형태의 비율, pykrx의 등락률은 1.23 형태의 %.
#:  값으로 판별하려 하면 "매일 +1.2%씩 오른 종목"을 비율로 오인해 123%로 만든다.)
UNIT_RATIO = "ratio"
UNIT_PERCENT = "percent"
UNIT_DERIVE = "derive"    # 종가로부터 다시 계산


def store_path(data_dir: str) -> str:
    return os.path.join(data_dir, STORE_DIR)


def manifest_path(data_dir: str) -> str:
    return os.path.join(store_path(data_dir), MANIFEST)


def load_manifest(data_dir: str) -> Dict[str, dict]:
    p = manifest_path(data_dir)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning("[v71] manifest 읽기 실패 %s: %s — 빈 것으로 시작", p, e)
        return {}


def save_manifest(data_dir: str, man: Dict[str, dict]) -> None:
    os.makedirs(store_path(data_dir), exist_ok=True)
    tmp = manifest_path(data_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, manifest_path(data_dir))


def normalize(df: pd.DataFrame, code: str, change_unit: str = UNIT_DERIVE) -> pd.DataFrame:
    """수집 결과를 ohlcv_cache 스키마로 정규화한다.

    change_unit 은 **호출자가 출처를 알고 넘긴다**. 기본값 UNIT_DERIVE 는
    출처의 등락률을 아예 쓰지 않고 종가에서 다시 계산한다 — 단위 혼동이
    구조적으로 불가능해지는 대신 첫 행의 등락률만 NaN 이 된다.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OUT_COLS)
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        for c in ("Date", "날짜", "date"):
            if c in d.columns:
                d = d.set_index(pd.to_datetime(d[c]))
                break
    ren = {"Open": "시가", "High": "고가", "Low": "저가", "Close": "종가",
           "Volume": "거래량", "Change": "등락률"}
    d = d.rename(columns={k: v for k, v in ren.items() if k in d.columns})
    for c in ("시가", "고가", "저가", "종가", "거래량", "등락률"):
        if c not in d.columns:
            d[c] = pd.NA
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if change_unit == UNIT_RATIO:
        d["등락률"] = d["등락률"] * 100.0
    elif change_unit == UNIT_DERIVE:
        d["등락률"] = pd.NA
    elif change_unit != UNIT_PERCENT:
        raise ValueError(f"알 수 없는 change_unit: {change_unit!r}")
    if d["등락률"].isna().all():
        d["등락률"] = pd.to_numeric(d["종가"], errors="coerce").pct_change() * 100.0
    d["종목코드"] = str(code).zfill(6)
    d = d[~d.index.isna()]
    d.index.name = "Date"
    d = d.dropna(subset=["종가"])
    d = d[d["종가"] > 0]
    return d[OUT_COLS].sort_index()


def fetch(code: str, start: str, end: Optional[str]) -> tuple:
    """(원본 DataFrame, 등락률 단위) 를 돌려준다.

    FDR 우선, 실패 시 pykrx. 둘 다 실패하면 예외를 올린다(조용히 삼키지 않는다).
    단위를 값이 아니라 **출처**로 결정하기 위해 함께 반환한다.
    """
    errs = []
    try:
        import FinanceDataReader as fdr
        return fdr.DataReader(str(code).zfill(6), start, end), UNIT_RATIO
    except Exception as e:
        errs.append(f"fdr:{type(e).__name__}:{e}")
    try:
        from pykrx import stock
        s = pd.Timestamp(start).strftime("%Y%m%d")
        e_ = pd.Timestamp(end).strftime("%Y%m%d") if end else datetime.now().strftime("%Y%m%d")
        return stock.get_market_ohlcv_by_date(s, e_, str(code).zfill(6)), UNIT_PERCENT
    except Exception as e:
        errs.append(f"pykrx:{type(e).__name__}:{e}")
    raise RuntimeError(" | ".join(errs))


def run(data_dir: str = "data", start: str = DEFAULT_START, end: Optional[str] = None,
        limit: Optional[int] = None, sleep: float = 0.2,
        retry_failed: bool = False, dry_run: bool = False,
        fetcher=fetch) -> dict:
    hist = uh.build(data_dir)
    if hist.empty:
        logger.error("[v71] 유니버스 이력이 비어 있다 — krx_codes 스냅샷을 확인하라")
        return dict(attempted=0, ok=0, empty=0, error=0)
    man = load_manifest(data_dir)
    targets: List[str] = []
    for code in uh.missing_codes(data_dir, hist):
        st = man.get(code, {}).get("status")
        if st == ST_OK:
            continue
        if st in (ST_EMPTY, ST_ERROR) and not retry_failed:
            continue
        targets.append(code)
    if limit is not None:
        targets = targets[:limit]
    logger.info("[v71] 백필 대상 %d종목 (start=%s end=%s)", len(targets), start, end or "today")
    if dry_run:
        return dict(attempted=len(targets), ok=0, empty=0, error=0, dry_run=True)

    os.makedirs(store_path(data_dir), exist_ok=True)
    names = dict(zip(hist["종목코드"], hist["종목명"]))
    n_ok = n_empty = n_err = 0
    for i, code in enumerate(targets, 1):
        rec = dict(code=code, name=names.get(code, ""), ts=datetime.now().isoformat(timespec="seconds"))
        try:
            got = fetcher(code, start, end)
            raw, unit = got if isinstance(got, tuple) else (got, UNIT_DERIVE)
            d = normalize(raw, code, change_unit=unit)
            if d.empty:
                rec.update(status=ST_EMPTY, rows=0)
                n_empty += 1
            else:
                d.to_parquet(os.path.join(store_path(data_dir), f"{code}.parquet"))
                rec.update(status=ST_OK, rows=int(len(d)),
                           first=str(d.index.min().date()), last=str(d.index.max().date()))
                n_ok += 1
        except Exception as e:
            rec.update(status=ST_ERROR, error=f"{type(e).__name__}: {e}"[:300])
            n_err += 1
        man[code] = rec
        if i % 25 == 0 or i == len(targets):
            save_manifest(data_dir, man)
            logger.info("[v71] %d/%d — ok %d · empty %d · error %d", i, len(targets), n_ok, n_empty, n_err)
        if sleep > 0:
            time.sleep(sleep)
    save_manifest(data_dir, man)
    return dict(attempted=len(targets), ok=n_ok, empty=n_empty, error=n_err)


def status(data_dir: str = "data") -> str:
    hist = uh.build(data_dir)
    cov = uh.coverage(data_dir, hist)
    man = load_manifest(data_dir)
    cnt: Dict[str, int] = {}
    for v in man.values():
        cnt[v.get("status", "?")] = cnt.get(v.get("status", "?"), 0) + 1
    parts = " · ".join(f"{k} {v}" for k, v in sorted(cnt.items())) or "기록 없음"
    return (f"{uh.summary_line(data_dir)}\n"
            f"[v71] 백필 상태: {parts}\n"
            f"[v71] 남은 작업 {cov['missing']:,}종목 "
            f"(소멸종목 {cov['delisted'] - cov['delisted_with_ohlcv']}종목 포함)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="생존편향 없는 유니버스 시세 백필")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.status:
        print(status(a.data_dir))
        return 0
    r = run(a.data_dir, start=a.start, end=a.end, limit=a.limit, sleep=a.sleep,
            retry_failed=a.retry_failed, dry_run=a.dry_run)
    print(json.dumps(r, ensure_ascii=False))
    print(status(a.data_dir))
    return 0 if r.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
