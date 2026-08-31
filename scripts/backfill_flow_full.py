# -*- coding: utf-8 -*-
"""과거 세션의 전종목 수급 백필 [v79] — 배치(GitHub Actions) 환경에서 수동 실행.

    python scripts/backfill_flow_full.py --start 20260226 --end 20260830

세션 달력은 data/ohlcv_cache 최신본의 날짜 집합을 쓴다(휴장일 호출 안 함).
이미 있는 날짜는 건너뛴다(중단 후 재실행 안전). 호출 간 짧은 대기로 예의를 지킨다.
"""
import argparse
import glob
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services import investor_flow_full as F  # noqa: E402


def sessions(data_dir: str):
    c = [f for f in sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_2*.parquet")))
         if "latest" not in f]
    if not c:
        return []
    d = pd.read_parquet(c[-1]).reset_index()
    return sorted(pd.to_datetime(d["Date"]).dt.strftime("%Y%m%d").unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()
    days = [s for s in sessions(a.data_dir) if a.start <= s <= a.end]
    have = set(F.have_days(a.data_dir))
    todo = [d for d in days if d not in have]
    print(f"세션 {len(days)}일 중 백필 대상 {len(todo)}일 (보유 {len(have)}일)")
    ok = fail = 0
    for i, ymd in enumerate(todo, 1):
        p = F.collect(a.data_dir, ymd)
        ok += bool(p); fail += (not p)
        print(f"[{i}/{len(todo)}] {ymd} {'OK' if p else 'FAIL'}")
        if not p and fail >= 3 and ok == 0:
            print("연속 실패 — 네트워크/차단 확인 후 재실행"); break
        time.sleep(a.sleep)
    print(f"완료 — 성공 {ok} · 실패 {fail} · 총 보유 {len(F.have_days(a.data_dir))}일")


if __name__ == "__main__":
    main()
