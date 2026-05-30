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
    args = ap.parse_args()

    trades = _load_trades(args.data_dir)
    if trades is None or len(trades) == 0:
        print("❌ per-trade 로그가 비었습니다. 캘리브레이션 테이블을 만들 수 없습니다.")
        sys.exit(1)

    score_col, win_col = _detect_cols(trades)
    score_col = args.score_col or score_col
    win_col = args.win_col or win_col
    if not score_col or not win_col:
        print(f"❌ score/win 컬럼 탐지 실패. cols={list(trades.columns)}")
        sys.exit(2)

    # 세그먼트 축: 레짐/티어가 로그에 있으면 우선, 없으면 method
    candidate_axes = ["ACTION_TIER", "MACRO_REGIME_MODE", "method", "horizon"]
    seg_cols = args.segment_cols
    if seg_cols is None:
        seg_cols = [c for c in candidate_axes if c in trades.columns][:2] or ["method"]

    asof = args.asof
    if asof is None and "rec_date" in trades.columns:
        try:
            asof = str(sorted(trades["rec_date"].astype(str))[-1])
        except Exception:
            asof = None

    table = build_segmented_table(
        trades, score_col=score_col, win_col=win_col,
        segment_cols=seg_cols, asof_ymd=asof,
    )
    table["meta"]["score_col"] = score_col
    table["meta"]["win_col"] = win_col
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
    print(f"   score_col={score_col} · seg_cols={m.get('segment_cols_used')} · prior={m.get('global_prior')}")
    print(f"   세그먼트 {m.get('n_segments')}개 · 0.55 돌파(충분) {n_break}개")
    if m["WARN_no_regime_tier_in_log"]:
        print("   ⚠️ 로그에 레짐/티어 없음 — 로거 보강 후 재빌드 시 다축 세그먼트 활성화")


if __name__ == "__main__":
    main()
