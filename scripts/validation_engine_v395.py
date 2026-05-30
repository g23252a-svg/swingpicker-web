# -*- coding: utf-8 -*-
"""Validation Engine v3.9.5 — Shadow Relaxation Time-Series Tracker (on top of v3.9.4).

Measurement-only. v3.9.4(No-Buy / Shadow / Triage / Carry-Stale)를 그대로 재사용하고,
shadow 완화 플래그(SHADOW_MACRO/ENTRY/SCORE_RELAXED_ELIGIBLE 등)의 forward 성과를
**신호일(signal_date)별 시계열 + 최근 롤링 추세**로 추적하는 레이어를 추가한다.

진단 배경 (2026-05 기준):
  - 두 달간 공식 추천 0개의 진짜 병목은 매크로가 아니라 TOP_PICK 진입/점수 게이트
    (특히 TIMING_SCORE; 엠케이전자 TIMING 69.0으로 70 문턱에 1점 미달 등).
  - 따라서 "진입/점수를 완화했으면 그 종목들이 실제 수익이 났을지"를 측정해야 한다.
  - SHADOW_ENTRY/SCORE_RELAXED_ELIGIBLE이 그 완화 후보를 표시(shadow)하므로,
    이들의 forward 성과를 날짜별로 추적하면 완화 게이트 도입 여부를 데이터로 판단할 수 있다.

검증 질문:
  - 각 shadow 완화가 누적으로 수익이 나는가? (PROMOTION_CANDIDATE / REJECT)
  - 최근 윈도우에서 성과가 개선되는가? (TREND_IMPROVING) — 시장 회복 신호.

산식/추천 무변경: shadow 완화 산정·TOP_PICK·BUY_NOW_ELIGIBLE·scoring_engine은
이 모듈에서 변경하지 않는다. 측정 전용이다.

CLI:
  python scripts/validation_engine_v395.py --data-dir data --out-dir data --recent-days 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.validation_engine_v393 import _mean, _win_rate, _truthy  # noqa: E402
from scripts.validation_engine_v394 import build_validation_engine_v394  # noqa: E402

logger = logging.getLogger("validation_engine_v395")

# 추적할 shadow 완화 플래그 (존재하는 것만 사용)
SHADOW_RELAX_FLAGS = [
    "SHADOW_MACRO_RELAXED_ELIGIBLE",
    "SHADOW_ENTRY_RELAXED_ELIGIBLE",
    "SHADOW_SCORE_RELAXED_ELIGIBLE",
    "MACRO_RELAXED_SHADOW_PASS",
]


def _flag_label(flag: str) -> str:
    return {
        "SHADOW_MACRO_RELAXED_ELIGIBLE": "매크로 완화",
        "SHADOW_ENTRY_RELAXED_ELIGIBLE": "진입조건 완화",
        "SHADOW_SCORE_RELAXED_ELIGIBLE": "점수 완화",
        "MACRO_RELAXED_SHADOW_PASS": "매크로 완화(PASS)",
    }.get(flag, flag)


def build_shadow_timeseries(row_df: pd.DataFrame) -> pd.DataFrame:
    """[v3.9.5] shadow 완화 플래그 × signal_date별 forward 성과 시계열.

    measurement-only. 각 (flag, signal_date)에 대해 완화로 신규 후보가 된 종목 수(N),
    실현결과 수(RESULT_N), forward 평균/승률을 집계한다.
    shadow 컬럼이 전혀 없으면 빈 표.
    """
    if row_df is None or row_df.empty:
        return pd.DataFrame()

    present = [c for c in SHADOW_RELAX_FLAGS if c in row_df.columns]
    if not present or "signal_date" not in row_df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for flag in present:
        flagged = row_df[row_df[flag].map(_truthy)]
        if flagged.empty:
            continue
        for sig_date, g in flagged.groupby("signal_date", dropna=False):
            ret = g["realized_ret_pct"] if "realized_ret_pct" in g.columns else pd.Series(dtype=float)
            rows.append({
                "SHADOW_FLAG": flag,
                "FLAG_LABEL": _flag_label(flag),
                "SIGNAL_DATE": str(sig_date),
                "N": int(len(g)),
                "RESULT_N": int(ret.notna().sum()),
                "AVG_RET_%": _mean(ret),
                "WIN_RATE_%": _win_rate(ret),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["SHADOW_FLAG", "SIGNAL_DATE"]).reset_index(drop=True)


def _promotion_verdict(cum_result_n: int, cum_avg: float | None,
                       recent_avg: float | None, cum_win: float | None) -> str:
    """완화 플래그를 production 게이트로 올릴 만한지 누적+최근 추세로 판정.

    - 누적 결과 < 5: NEED_MORE_N (데이터 부족)
    - 누적 평균 ≥ +2.0% AND 누적 승률 ≥ 55%: PROMOTION_CANDIDATE (완화가 유효)
    - 누적 평균 ≤ -1.5%: REJECT_RELAXATION (완화하면 손실)
    - 최근 윈도우 평균이 누적보다 +1.0%p 이상 개선 & 최근≥0: TREND_IMPROVING (회복 조짐)
    - 그 외: KEEP_TRACKING
    """
    if cum_result_n < 5:
        return "NEED_MORE_N"
    if cum_avg is not None and cum_avg >= 2.0 and (cum_win is None or cum_win >= 55.0):
        return "PROMOTION_CANDIDATE"
    if cum_avg is not None and cum_avg <= -1.5:
        return "REJECT_RELAXATION"
    if (recent_avg is not None and cum_avg is not None
            and recent_avg >= 0.0 and (recent_avg - cum_avg) >= 1.0):
        return "TREND_IMPROVING"
    return "KEEP_TRACKING"


def build_shadow_relaxation_summary(row_df: pd.DataFrame, recent_days: int = 5) -> pd.DataFrame:
    """[v3.9.5] flag별 누적 + 최근 N거래일 윈도우 성과 + 승급 판정."""
    if row_df is None or row_df.empty:
        return pd.DataFrame()
    present = [c for c in SHADOW_RELAX_FLAGS if c in row_df.columns]
    if not present or "signal_date" not in row_df.columns:
        return pd.DataFrame()

    all_dates = sorted(row_df["signal_date"].dropna().astype(str).unique())
    recent_set = set(all_dates[-recent_days:]) if all_dates else set()

    rows: list[dict] = []
    for flag in present:
        flagged = row_df[row_df[flag].map(_truthy)].copy()
        if flagged.empty:
            continue  # 완화 후보 0건인 플래그는 추적 대상 아님 (legacy/미도입)
        cum_ret = flagged["realized_ret_pct"] if "realized_ret_pct" in flagged.columns else pd.Series(dtype=float)
        cum_result_n = int(cum_ret.notna().sum())
        cum_avg = _mean(cum_ret)
        cum_win = _win_rate(cum_ret)

        recent = flagged[flagged["signal_date"].astype(str).isin(recent_set)]
        recent_ret = recent["realized_ret_pct"] if "realized_ret_pct" in recent.columns else pd.Series(dtype=float)
        recent_avg = _mean(recent_ret)

        verdict = _promotion_verdict(cum_result_n, cum_avg, recent_avg, cum_win)
        rows.append({
            "SHADOW_FLAG": flag,
            "FLAG_LABEL": _flag_label(flag),
            "TOTAL_FLAGGED": int(len(flagged)),
            "CUM_RESULT_N": cum_result_n,
            "CUM_AVG_RET_%": cum_avg,
            "CUM_WIN_RATE_%": cum_win,
            "RECENT_DAYS": recent_days,
            "RECENT_FLAGGED": int(len(recent)),
            "RECENT_RESULT_N": int(recent_ret.notna().sum()),
            "RECENT_AVG_RET_%": recent_avg,
            "PROMOTION_VERDICT": verdict,
        })

    return pd.DataFrame(rows)


def _relaxation_headline(summary_df: pd.DataFrame) -> dict:
    """summary JSON용 핵심 판정 요약."""
    if summary_df is None or summary_df.empty:
        return {
            "status": "NO_SHADOW_DATA",
            "note": "SHADOW_*_RELAXED 컬럼 없음 또는 플래그 0건 "
                    "(추천 시스템에 shadow 완화 도입 이후 데이터 누적 필요).",
        }
    verdicts = summary_df.set_index("SHADOW_FLAG")["PROMOTION_VERDICT"].to_dict()
    promote = summary_df[summary_df["PROMOTION_VERDICT"] == "PROMOTION_CANDIDATE"]["SHADOW_FLAG"].tolist()
    improving = summary_df[summary_df["PROMOTION_VERDICT"] == "TREND_IMPROVING"]["SHADOW_FLAG"].tolist()
    reject = summary_df[summary_df["PROMOTION_VERDICT"] == "REJECT_RELAXATION"]["SHADOW_FLAG"].tolist()
    return {
        "status": "OK",
        "verdicts": {str(k): str(v) for k, v in verdicts.items()},
        "promotion_candidates": promote,
        "trend_improving": improving,
        "reject": reject,
    }


def build_validation_engine_v395(
    data_dir: str | Path = "data", out_dir: str | Path = "data", recent_days: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """v3.9.4 전체 리포트 + v3.9.5 shadow 완화 시계열/추세.

    Returns:
        (row_level_df, shadow_timeseries_df, shadow_summary_df, summary_payload)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # v3.9.4 재사용 (no_buy / shadow / triage / carry-stale + v393/v394 산출물)
    row_level, _no_buy, _shadow, _carry, base_summary = build_validation_engine_v394(data_dir, out_dir)

    # [v3.9.5] shadow 완화 시계열 + 추세 요약
    ts = build_shadow_timeseries(row_level)
    relax = build_shadow_relaxation_summary(row_level, recent_days=recent_days)

    if not ts.empty:
        ts.to_csv(out_path / "shadow_relaxation_timeseries_latest.csv", index=False, encoding="utf-8-sig")
    if not relax.empty:
        relax.to_csv(out_path / "shadow_relaxation_summary_latest.csv", index=False, encoding="utf-8-sig")

    headline = _relaxation_headline(relax)
    summary = dict(base_summary)
    summary["version"] = "v3.9.5"
    summary["shadow_relaxation"] = headline
    summary["recent_days_window"] = recent_days
    summary["notes"] = list(base_summary.get("notes", [])) + [
        "v3.9.5 adds shadow relaxation time-series tracking (SHADOW_*_RELAXED forward returns).",
        "Measurement-only: does not change relaxation flags, TOP_PICK, BUY_NOW_ELIGIBLE, or scoring.",
    ]
    with open(out_path / "validation_engine_v395_latest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return row_level, ts, relax, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation Engine v3.9.5 reports")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--recent-days", type=int, default=5, help="최근 추세 윈도우(거래일 수)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    row_level, ts, relax, summary = build_validation_engine_v395(
        args.data_dir, args.out_dir, recent_days=args.recent_days
    )
    print("[v3.9.5] Validation Engine reports generated")
    print(f"- rows: {len(row_level)}")
    print(f"- shadow_timeseries_rows: {len(ts)}")
    print(f"- shadow_flags_summarized: {len(relax)}")
    sr = summary.get("shadow_relaxation", {})
    print(f"- shadow_status: {sr.get('status')}")
    if sr.get("status") == "OK":
        for flag, v in sr.get("verdicts", {}).items():
            print(f"    · {flag}: {v}")
    print(f"- asof: {summary.get('asof')}")


if __name__ == "__main__":
    main()
