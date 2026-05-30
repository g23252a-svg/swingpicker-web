# -*- coding: utf-8 -*-
"""Validation Engine v3.9.4 — Carry-Stale Exit Validation (on top of v3.9.3).

Measurement-only. v3.9.3의 No-Buy / Shadow / Triage 검증을 그대로 재사용하고,
추천 시스템 v3.9.28이 만든 carry-stale 가드(CARRY_STALE_STAGE / CARRY_EXIT_SIGNAL)가
실제 forward return을 예측하는지 검증하는 레이어를 추가한다.

검증 질문:
  - CARRY_STALE_STAGE(FRESH/WATCH/STALE/DEAD)가 진행될수록 forward 성과가 나빠지는가?
  - CARRY_EXIT_SIGNAL=1(DEAD)이 실제로 부진한 포지션을 식별하는가, 아니면 너무 이른가?

산식/추천 무변경: CARRY_EXIT_SIGNAL 산정·stale 감점·TOP_PICK·BUY_NOW_ELIGIBLE·scoring_engine은
이 모듈에서 변경하지 않는다. 표시/측정 전용이다.

CLI:
  python scripts/validation_engine_v394.py --data-dir data --out-dir data
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ROOT를 path에 올려 스크립트/테스트 양쪽에서 v3.9.3 재사용 가능하게.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.validation_engine_v393 import (  # noqa: E402
    build_validation_engine_v393,
    _mean,
    _median,
    _rate,
    _truthy,
    _win_rate,
)

logger = logging.getLogger("validation_engine_v394")

CARRY_STAGES = ["FRESH", "WATCH", "STALE", "DEAD"]


def _carry_exit_grade(signal_n: int | None, signal_avg: float | None) -> str:
    """CARRY_EXIT_SIGNAL=1 그룹의 forward 성과로 가드 타당성을 판정.

    - 표본 부족(<5): NEED_MORE_N
    - 평균 forward ≤ -1.5%: EXIT_SIGNAL_VALIDATED (가드가 부진 포지션을 옳게 식별)
    - 평균 forward ≥ +2.0%: EXIT_TOO_AGGRESSIVE_WARNING (회복 포지션에 너무 이른 신호)
    - 그 외: INCONCLUSIVE
    """
    if signal_n is None or signal_n < 5:
        return "NEED_MORE_N"
    if signal_avg is not None and signal_avg <= -1.5:
        return "EXIT_SIGNAL_VALIDATED"
    if signal_avg is not None and signal_avg >= 2.0:
        return "EXIT_TOO_AGGRESSIVE_WARNING"
    return "INCONCLUSIVE"


def _group_stat(g: pd.DataFrame, kind: str, value: str, hint: str = "") -> dict:
    g_ret = g["realized_ret_pct"] if "realized_ret_pct" in g.columns else pd.Series(dtype=float)
    has_result = g_ret.notna()
    return {
        "GROUP_KIND": kind,
        "GROUP_VALUE": value,
        "N": int(len(g)),
        "RESULT_N": int(has_result.sum()),
        "AVG_RET_%": _mean(g_ret),
        "MEDIAN_RET_%": _median(g_ret),
        "WIN_RATE_%": _win_rate(g_ret),
        "STOP_HIT_RATE_%": _rate(g.loc[has_result, "stop_hit_bool"]) if "stop_hit_bool" in g.columns else None,
        "GRADE_OR_HINT": hint,
    }


def build_carry_stale_validation(row_df: pd.DataFrame) -> pd.DataFrame:
    """[v3.9.4] carry-stale 단계/청산신호별 forward 성과 검증표.

    measurement-only. carry 행(ROUTE=CARRY 또는 유효한 CARRY_STALE_STAGE)만 집계한다.
    carry-stale 컬럼이 전혀 없으면(추천 시스템 v3.9.28 이전 데이터) 빈 표를 반환한다.
    """
    if row_df is None or row_df.empty:
        return pd.DataFrame()

    has_stage = "CARRY_STALE_STAGE" in row_df.columns
    has_signal = "CARRY_EXIT_SIGNAL" in row_df.columns
    if not has_stage and not has_signal:
        return pd.DataFrame()

    work = row_df.copy()
    if has_stage:
        work["_stage"] = work["CARRY_STALE_STAGE"].astype(str).str.upper().str.strip()
        carry_mask = work["_stage"].isin(CARRY_STAGES)
    else:
        work["_stage"] = ""
        route = work.get("ROUTE", pd.Series("", index=work.index)).astype(str).str.upper()
        carry_mask = route.eq("CARRY")

    work = work[carry_mask].copy()
    if work.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    # 1) STAGE별 forward 성과 (FRESH → DEAD 진행에 따른 악화 여부)
    if has_stage:
        for stage in CARRY_STAGES:
            g = work[work["_stage"] == stage]
            if g.empty:
                continue
            rows.append(_group_stat(g, "STAGE", stage))

    # 2) EXIT_SIGNAL별 forward 성과 + 가드 판정 grade
    if has_signal:
        work["_sig"] = work["CARRY_EXIT_SIGNAL"].map(_truthy)
        sig1 = work[work["_sig"]]
        sig0 = work[~work["_sig"]]
        sig1_ret = sig1["realized_ret_pct"] if "realized_ret_pct" in sig1.columns else pd.Series(dtype=float)
        sig1_n = int(sig1_ret.notna().sum())
        sig1_avg = _mean(sig1_ret)
        grade = _carry_exit_grade(sig1_n, sig1_avg)
        rows.append(_group_stat(sig1, "EXIT_SIGNAL", "EXIT_SIGNAL=1", hint=grade))
        rows.append(_group_stat(sig0, "EXIT_SIGNAL", "EXIT_SIGNAL=0", hint=""))

    return pd.DataFrame(rows)


def _carry_stale_verdict(carry_df: pd.DataFrame) -> dict:
    """summary JSON용 carry-stale 핵심 판정 요약."""
    if carry_df is None or carry_df.empty:
        return {
            "status": "NO_CARRY_STALE_DATA",
            "note": "CARRY_STALE_STAGE/CARRY_EXIT_SIGNAL 없음 또는 carry 행 없음 "
                    "(추천 시스템 v3.9.28 이후 데이터 누적 필요).",
        }

    def _pick(kind: str, value: str) -> dict | None:
        sub = carry_df[(carry_df["GROUP_KIND"] == kind) & (carry_df["GROUP_VALUE"] == value)]
        if sub.empty:
            return None
        r = sub.iloc[0]
        return {
            "N": int(r["N"]),
            "RESULT_N": int(r["RESULT_N"]),
            "AVG_RET_%": r["AVG_RET_%"],
            "WIN_RATE_%": r["WIN_RATE_%"],
            "STOP_HIT_RATE_%": r["STOP_HIT_RATE_%"],
        }

    sig1_rows = carry_df[(carry_df["GROUP_KIND"] == "EXIT_SIGNAL") & (carry_df["GROUP_VALUE"] == "EXIT_SIGNAL=1")]
    grade = str(sig1_rows.iloc[0]["GRADE_OR_HINT"]) if not sig1_rows.empty else ""

    dead = _pick("STAGE", "DEAD")
    fresh = _pick("STAGE", "FRESH")
    out = {
        "status": "OK",
        "exit_signal_grade": grade,
        "exit_signal_1": _pick("EXIT_SIGNAL", "EXIT_SIGNAL=1"),
        "exit_signal_0": _pick("EXIT_SIGNAL", "EXIT_SIGNAL=0"),
        "stage_dead": dead,
        "stage_fresh": fresh,
    }
    if dead and fresh and dead["AVG_RET_%"] is not None and fresh["AVG_RET_%"] is not None:
        # 음수일수록 DEAD가 FRESH보다 forward가 나쁨 = 가드가 의미 있게 분별함
        out["dead_minus_fresh_avg_ret_%"] = round(dead["AVG_RET_%"] - fresh["AVG_RET_%"], 4)
    return out


def build_validation_engine_v394(
    data_dir: str | Path = "data", out_dir: str | Path = "data"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """v3.9.3 전체 리포트 + v3.9.4 carry-stale 검증.

    Returns:
        (row_level_df, no_buy_df, shadow_df, carry_stale_df, summary_payload)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # v3.9.3 리포트 재사용 (no_buy / shadow / triage / funnel + v393 JSON·CSV 산출)
    row_level, no_buy, shadow, base_summary = build_validation_engine_v393(data_dir, out_dir)

    # [v3.9.4] carry-stale 검증 추가
    carry = build_carry_stale_validation(row_level)
    if not carry.empty:
        carry.to_csv(out_path / "carry_stale_validation_latest.csv", index=False, encoding="utf-8-sig")

    verdict = _carry_stale_verdict(carry)
    summary = dict(base_summary)
    summary["version"] = "v3.9.4"
    summary["carry_stale_validation"] = verdict
    summary["notes"] = list(base_summary.get("notes", [])) + [
        "v3.9.4 adds carry-stale exit validation (CARRY_STALE_STAGE / CARRY_EXIT_SIGNAL).",
        "Measurement-only: does not change CARRY_EXIT_SIGNAL, stale penalty curve, or entry formulas.",
    ]
    with open(out_path / "validation_engine_v394_latest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return row_level, no_buy, shadow, carry, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation Engine v3.9.4 reports")
    parser.add_argument("--data-dir", default="data", help="Directory containing recommend/backtest CSV files")
    parser.add_argument("--out-dir", default="data", help="Directory to write validation reports")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    row_level, no_buy, shadow, carry, summary = build_validation_engine_v394(args.data_dir, args.out_dir)
    print("[v3.9.4] Validation Engine reports generated")
    print(f"- rows: {len(row_level)}")
    print(f"- no_buy_days: {len(no_buy)}")
    print(f"- shadow_flags: {len(shadow)}")
    print(f"- carry_stale_groups: {len(carry)}")
    cs = summary.get("carry_stale_validation", {})
    print(f"- carry_stale_status: {cs.get('status')} · exit_signal_grade: {cs.get('exit_signal_grade', '-')}")
    print(f"- asof: {summary.get('asof')}")


if __name__ == "__main__":
    main()
