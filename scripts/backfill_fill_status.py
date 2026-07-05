# -*- coding: utf-8 -*-
"""
scripts/backfill_fill_status.py — v24.8 과거 per_trade_log 체결 검증 백필 (1회 실행)
═══════════════════════════════════════════════════════════════════════════════════
기존 per_trade_log.csv 전 행에 대해 OHLCV 실경로로 지정가 체결 여부를 검증한다.

  체결 규칙 (신규 시뮬과 동일): rec_date 이후 min(fill_window=3, horizon) 거래일 내
  해당 종목 저가 ≤ entry_price → filled (fill_day = 최초 도달일, 1-base)

출력:
  data/per_trade_log.csv           ← filled + unknown 행만 (fill_status/fill_day 컬럼 추가)
  data/per_trade_log_unfilled.csv  ← 유령체결(unfilled) 행 아카이브
  data/per_trade_log.csv.bak.{ts}  ← 원본 백업

사용:
  python scripts/backfill_fill_status.py --dry-run    # 통계만 출력, 파일 무변경
  python scripts/backfill_fill_status.py              # 실제 백필

주의: 일일 파이프라인(v24.8 collector)이 배포된 뒤 실행할 것 — 그래야 신규 행도
      같은 규칙으로 filled-only 기록되어 로그가 일관된다.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

FILL_WINDOW = 3


def load_ohlcv_lookup(data_dir: str):
    """최신 ohlcv_cache parquet(전기간) → {code: ndarray[[ymd, low], ...] date-sorted}."""
    files = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.parquet")))
    if not files:
        raise FileNotFoundError("ohlcv_cache_*.parquet 없음")
    path = files[-1]
    o = pd.read_parquet(path).reset_index()
    date_col = "Date" if "Date" in o.columns else o.columns[0]
    o["code"] = o["종목코드"].astype(str).str.zfill(6)
    o["d"] = pd.to_datetime(o[date_col]).dt.strftime("%Y%m%d").astype(int)
    o = o.sort_values(["code", "d"])
    lut = {c: g[["d", "저가"]].to_numpy(dtype=float) for c, g in o.groupby("code")}
    return lut, os.path.basename(path)


def fill_check_row(lut, code: str, rec_date: int, entry: float, horizon: int,
                   window: int = FILL_WINDOW):
    """(status, fill_day). status ∈ filled / unfilled / unknown."""
    if not np.isfinite(entry) or entry <= 0:
        return "unknown", 0
    try:
        h = int(horizon)
    except Exception:
        return "unknown", 0
    w = min(window, h) if h >= 1 else 0
    if w == 0:
        return "unknown", 0
    a = lut.get(str(code).zfill(6))
    if a is None or len(a) == 0:
        return "unknown", 0
    i = np.searchsorted(a[:, 0], rec_date, side="right")
    seg = a[i:i + w]
    if len(seg) == 0:
        return "unknown", 0
    hit = np.where(seg[:, 1] <= entry)[0]
    if len(hit):
        return "filled", int(hit[0]) + 1
    # 창이 완전하지 않고(미래 데이터 부족) 아직 미도달이면 판단 유보
    if len(seg) < w:
        return "unknown", 0
    return "unfilled", 0


def backfill(data_dir: str = "data", dry_run: bool = False) -> dict:
    log_path = os.path.join(data_dir, "per_trade_log.csv")
    df = pd.read_csv(log_path, low_memory=False)
    n0 = len(df)
    lut, src = load_ohlcv_lookup(data_dir)
    print(f"per_trade_log {n0:,}행 | OHLCV: {src} ({len(lut):,}종목)")

    if "fill_status" in df.columns:
        todo = df["fill_status"].isna() | (df["fill_status"] == "")
        print(f"fill_status 기존재 — 미기입 {int(todo.sum()):,}행만 검증")
    else:
        df["fill_status"] = ""
        df["fill_day"] = 0
        todo = pd.Series(True, index=df.index)

    stat = {"filled": 0, "unfilled": 0, "unknown": 0}
    codes = df["code"].astype(str).str.zfill(6).values
    recs = pd.to_numeric(df["rec_date"], errors="coerce").fillna(0).astype(int).values
    ents = pd.to_numeric(df["entry_price"], errors="coerce").values
    hors = pd.to_numeric(df["horizon"], errors="coerce").fillna(0).astype(int).values

    fs = np.array(df["fill_status"].astype(object).tolist(), dtype=object)
    fd = pd.to_numeric(df.get("fill_day", 0), errors="coerce").fillna(0).astype(int).to_numpy().copy()
    for idx in np.where(todo.values)[0]:
        s, d = fill_check_row(lut, codes[idx], recs[idx], ents[idx], hors[idx])
        fs[idx] = s
        fd[idx] = d
        stat[s] += 1
    df["fill_status"] = fs
    df["fill_day"] = fd

    print(f"검증 결과: filled {stat['filled']:,} / unfilled(유령) {stat['unfilled']:,} "
          f"/ unknown {stat['unknown']:,}")
    keep = df[df["fill_status"] != "unfilled"]
    drop = df[df["fill_status"] == "unfilled"]
    # 유령 제거 전후 로그 성과 비교 (h5 기준 참고 출력)
    for name, d5 in [("전(유령포함)", df), ("후(체결분만)", keep)]:
        h5 = d5[pd.to_numeric(d5["horizon"], errors="coerce") == 5]
        if len(h5):
            r = pd.to_numeric(h5["ret_pct"], errors="coerce")
            print(f"  h5 {name}: n={len(h5):,} 평균 {r.mean():+.2f}% 승률 {(r > 0).mean() * 100:.0f}%")

    if dry_run:
        print("[dry-run] 파일 무변경")
        return stat

    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = f"{log_path}.bak.{ts}"
    os.replace(log_path, bak)
    keep.to_csv(log_path, index=False, encoding="utf-8-sig")
    unf_path = os.path.join(data_dir, "per_trade_log_unfilled.csv")
    if len(drop):
        header = not os.path.exists(unf_path)
        drop.to_csv(unf_path, mode="a", header=header, index=False, encoding="utf-8-sig")
    print(f"저장: {log_path} ({len(keep):,}행) | 아카이브 +{len(drop):,}행 → {unf_path}")
    print(f"백업: {bak}")
    return stat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.exit(0 if backfill(args.data_dir, args.dry_run) is not None else 1)
