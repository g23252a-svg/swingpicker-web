# -*- coding: utf-8 -*-
"""
scripts/build_calibration_v4_table.py — v4.0 Phase 1 세그먼트 캘리브레이션 테이블 빌더

사용:
    python scripts/build_calibration_v4_table.py
    python scripts/build_calibration_v4_table.py --score-col score --segment-cols method horizon

출력:
    data/calibration_v4_table_latest.json
    data/calibration_v4_table_<YYYYMMDD>.json

근거 (2026-05-30 실측, 11,467 trades):
    - 글로벌 시간감쇠 승률 ≈ 0.65
    - 충분표본 세그먼트 중 7개가 0.55 돌파 (최고 0.80) → 기존 ELITE 단일축 ~0.51 캡 제거
    - ⚠️ ELITE_SCORE 방식은 비단조(80-90 band 0.43 < 70-80 band 0.61).
      DISPLAY_SCORE / FINAL_SCORE 가 더 잘 분리 → 기본 score-col 권장 = score(=DISPLAY/FINAL 계열)

⚠️ Phase 1 선결 과제 (TODO):
    현재 per-trade 로그(kelly_calibrator.save_per_trade_log)는 MACRO_REGIME_MODE / ACTION_TIER 를
    기록하지 않는다. 레짐·티어 축 세그먼트 캘리브레이션을 위해선 로거에 두 컬럼을 추가하고
    1~2개월 누적해야 한다. 그 전까지 세그먼트는 score × method(× horizon)로 한정한다.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_v4 import build_segmented_table  # noqa: E402

OUT_DIR = os.environ.get("SP_DATA_DIR", "data")


def _load_trades(out_dir: str):
    from kelly_calibrator import load_per_trade_log
    return load_per_trade_log(out_dir)


def _detect_cols(trades):
    """로그 스키마에서 score/win 컬럼을 안전하게 탐지한다."""
    cols = {c.lower(): c for c in trades.columns}
    score = cols.get("score") or cols.get("display_score") or cols.get("final_score") or cols.get("elite_score")
    win = cols.get("win") or cols.get("is_win") or cols.get("y") or cols.get("hit")
    return score, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=OUT_DIR)
    ap.add_argument("--score-col", default=None, help="미지정 시 자동 탐지(score 우선)")
    ap.add_argument("--win-col", default=None)
    ap.add_argument("--segment-cols", nargs="*", default=None,
                    help="미지정 시 로그에 존재하는 [ACTION_TIER, MACRO_REGIME_MODE, method] 중 사용")
    ap.add_argument("--asof", default=None, help="기준일 YYYYMMDD (미지정 시 최신 rec_date)")
    ap.add_argument("--win-basis", default="excess", choices=["absolute", "excess"],
                    help="excess(기본): 당일 시장 대비 초과승률 → 인플레 제거 / absolute: ret>0")
    ap.add_argument("--method-filter", default="DISPLAY_SCORE",
                    help="라이브 테이블 정렬축. 로그를 이 method 행으로 필터(DISPLAY_SCORE 권장). "
                         "'none'이면 필터 안 함(분석용 method-세그먼트 테이블)")
    ap.add_argument("--horizon", type=int, default=5, help="스윙 horizon 필터(기본 5). -1이면 미필터")
    ap.add_argument("--lookup-col", default=None,
                    help="추론 시 recommend row에서 버킷 읽을 컬럼. 미지정 시 method-filter와 동일")
    args = ap.parse_args()

    trades = _load_trades(args.data_dir)
    if trades is None or len(trades) == 0:
        print("❌ per-trade 로그가 비었습니다. 캘리브레이션 테이블을 만들 수 없습니다.")
        sys.exit(1)

    # ── 라이브 lookup 정렬: method/horizon 필터 ──
    method_filter = None if str(args.method_filter).lower() == "none" else args.method_filter
    lookup_col = args.lookup_col or method_filter  # method DISPLAY_SCORE ↔ recommend 컬럼 DISPLAY_SCORE
    if method_filter and "method" in trades.columns:
        _before = len(trades)
        trades = trades[trades["method"].astype(str) == method_filter].copy()
        print(f"   ↪ method='{method_filter}' 필터: {_before} → {len(trades)}행")
    if args.horizon >= 0 and "horizon" in trades.columns:
        trades = trades[pd.to_numeric(trades["horizon"], errors="coerce") == args.horizon].copy()
        print(f"   ↪ horizon={args.horizon} 필터 → {len(trades)}행")
    if len(trades) == 0:
        print("❌ 필터 후 표본 0. --method-filter/--horizon 조정 필요.")
        sys.exit(3)

    score_col, win_col = _detect_cols(trades)
    score_col = args.score_col or score_col
    win_col = args.win_col or win_col
    if not score_col or not win_col:
        print(f"❌ score/win 컬럼 탐지 실패. cols={list(trades.columns)}")
        sys.exit(2)

    # 세그먼트 축: 레짐/티어만(라이브 recommend에 있는 축). method/horizon은 이미 필터로 고정.
    live_axes = ["ACTION_TIER", "MACRO_REGIME_MODE"]
    seg_cols = args.segment_cols
    if seg_cols is None:
        seg_cols = [c for c in live_axes if c in trades.columns]  # 없으면 [] → score 버킷만

    asof = args.asof
    if asof is None and "rec_date" in trades.columns:
        try:
            asof = str(sorted(trades["rec_date"].astype(str))[-1])
        except Exception:
            asof = None

    table = build_segmented_table(
        trades, score_col=score_col, win_col=win_col,
        segment_cols=seg_cols, asof_ymd=asof,
        win_basis=args.win_basis, lookup_col=lookup_col,
    )
    table["meta"]["win_col"] = win_col
    table["meta"]["method_filter"] = method_filter
    table["meta"]["horizon_filter"] = args.horizon
    table["meta"]["built_at"] = datetime.now().isoformat(timespec="seconds")
    table["meta"]["WARN_no_regime_tier_in_log"] = not (
        "MACRO_REGIME_MODE" in trades.columns and "ACTION_TIER" in trades.columns
    )

    os.makedirs(args.data_dir, exist_ok=True)
    ymd = (asof or datetime.now().strftime("%Y%m%d")).replace("-", "")[:8]
    latest = os.path.join(args.data_dir, "calibration_v4_table_latest.json")
    dated = os.path.join(args.data_dir, f"calibration_v4_table_{ymd}.json")
    for path in (latest, dated):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=1)

    m = table["meta"]
    n_break = sum(1 for r in table["table"] if r["sufficient"] and r["p_win"] > 0.55)
    print(f"✅ 저장: {latest}")
    print(f"   win_basis={m.get('win_basis')} · benchmark={m.get('benchmark_source')} · prior={m.get('global_prior')}")
    print(f"   score_col={score_col} · lookup_col={m.get('lookup_col')} · seg_cols={m.get('segment_cols_used')}")
    print(f"   method_filter={m.get('method_filter')} · horizon={m.get('horizon_filter')} · 세그먼트 {m.get('n_segments')}개 · 0.55 돌파(충분) {n_break}개")
    if m.get("win_basis") == "excess":
        print("   ℹ️ excess 기준: prior가 ~0.5 근처면 정상(시장 베타 제거). 0.5 초과 세그먼트가 진짜 엣지.")
    if m["WARN_no_regime_tier_in_log"]:
        print("   ⚠️ 로그에 레짐/티어 없음 — 로거 보강 후 재빌드 시 다축 세그먼트 활성화")


if __name__ == "__main__":
    main()
