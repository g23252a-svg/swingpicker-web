# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation_engine_v393 import (  # noqa: E402
    build_validation_engine_v393,
    load_recommend_snapshots,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_v393_preserves_v3924_triage_and_grades_no_buy_day(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    out = tmp_path / "out"

    _write_csv(
        data / "recommend_20260527.csv",
        [
            {
                "종목코드": "321370",
                "종목명": "센서뷰",
                "TOP_PICK": 0,
                "BUY_NOW_ELIGIBLE": 0,
                "BUY_NOW_PASS": 1,
                "BUY_NOW_SCORE": 80,
                "FINAL_SCORE": 69.1,
                "ELITE_SCORE": 68.4,
                "ENTRY_GAP_PCT": 0.0,
                "VWAP_GAP": -0.17,
                "POC_GAP": 5.27,
                "RR_NOW_TP1": 1.45,
                "ROUTE": "ARMED",
                "CANDIDATE_TRIAGE_TYPE": "ENTRY_CLEAN_OBSERVE",
                "OFFICIAL_FUNNEL_STAGE": "ENTRY_READY_BUT_NOT_TOP_PICK",
                "SHADOW_MACRO_RELAXED_ELIGIBLE": 1,
            },
            {
                "종목코드": "089530",
                "종목명": "에이티세미콘",
                "TOP_PICK": 0,
                "BUY_NOW_ELIGIBLE": 0,
                "BUY_NOW_PASS": 1,
                "BUY_NOW_SCORE": 78,
                "FINAL_SCORE": 68.5,
                "ELITE_SCORE": 66.0,
                "ENTRY_GAP_PCT": 0.3,
                "VWAP_GAP": 1.2,
                "POC_GAP": 7.0,
                "RR_NOW_TP1": 1.35,
                "ROUTE": "ARMED",
                "CANDIDATE_TRIAGE_TYPE": "ENTRY_CLEAN_OBSERVE",
                "OFFICIAL_FUNNEL_STAGE": "ENTRY_READY_BUT_NOT_TOP_PICK",
                "SHADOW_MACRO_RELAXED_ELIGIBLE": 1,
            },
            {
                "종목코드": "195870",
                "종목명": "해성디에스",
                "TOP_PICK": 0,
                "BUY_NOW_ELIGIBLE": 0,
                "BUY_NOW_PASS": 0,
                "FINAL_SCORE": 90.3,
                "ELITE_SCORE": 82.0,
                "ENTRY_GAP_PCT": 7.0,
                "VWAP_GAP": 29.0,
                "POC_GAP": 70.0,
                "RR_NOW_TP1": 1.2,
                "ROUTE": "WAIT",
                "CANDIDATE_TRIAGE_TYPE": "HIGH_SCORE_OBSERVE",
                "OFFICIAL_FUNNEL_STAGE": "HIGH_SCORE_BUT_ENTRY_BLOCKED",
            },
            {
                "종목코드": "007110",
                "종목명": "프로이천",
                "TOP_PICK": 0,
                "BUY_NOW_ELIGIBLE": 0,
                "BUY_NOW_PASS": 0,
                "FINAL_SCORE": 85.3,
                "ELITE_SCORE": 77.0,
                "ENTRY_GAP_PCT": 16.4,
                "VWAP_GAP": 22.0,
                "POC_GAP": 55.0,
                "RR_NOW_TP1": 0.98,
                "ROUTE": "ATTACK",
                "CANDIDATE_TRIAGE_TYPE": "CHASE_RISK",
                "OFFICIAL_FUNNEL_STAGE": "ROUTE_ACTIVE_BUT_CHASE_RISK",
            },
        ],
    )
    _write_csv(
        data / "backtest_top3_trades_20260527.csv",
        [
            {"date": "2026-05-27", "code": "321370", "net_pct": 4.1, "tp1_hit": 1, "stop_hit": 0},
            {"date": "2026-05-27", "code": "089530", "net_pct": 2.9, "tp1_hit": 0, "stop_hit": 0},
            {"date": "2026-05-27", "code": "195870", "net_pct": -3.3, "tp1_hit": 0, "stop_hit": 1},
            {"date": "2026-05-27", "code": "007110", "net_pct": -4.8, "tp1_hit": 0, "stop_hit": 1},
        ],
    )

    row_level, no_buy, shadow, summary = build_validation_engine_v393(data, out)

    assert not row_level.empty
    assert set(row_level["CANDIDATE_TRIAGE_TYPE"]) >= {"ENTRY_CLEAN_OBSERVE", "HIGH_SCORE_OBSERVE", "CHASE_RISK"}
    assert no_buy.loc[0, "NO_BUY_DECISION_GRADE"] == "TOO_CONSERVATIVE_WARNING"
    assert summary["version"] == "v3.9.3"
    assert (out / "validation_engine_v393_latest.csv").exists()
    assert (out / "no_buy_decision_validation_latest.csv").exists()
    assert (out / "shadow_candidate_validation_latest.csv").exists()
    assert "SHADOW_MACRO_RELAXED_ELIGIBLE" in set(shadow["SHADOW_FLAG"])


def test_v393_fallback_classifies_legacy_recommend_without_v3924_columns(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()

    _write_csv(
        data / "recommend_20260528.csv",
        [
            {
                "종목코드": "321370",
                "종목명": "센서뷰",
                "TOP_PICK": 0,
                "BUY_NOW_ELIGIBLE": 0,
                "BUY_NOW_PASS": 1,
                "FINAL_SCORE": 69.1,
                "ELITE_SCORE": 68.4,
                "ENTRY_GAP_PCT": 0.0,
                "VWAP_GAP": 0.1,
                "POC_GAP": 5.0,
                "RR_NOW_TP1": 1.45,
                "ROUTE": "ARMED",
            },
            {
                "종목코드": "195870",
                "종목명": "해성디에스",
                "TOP_PICK": 0,
                "BUY_NOW_ELIGIBLE": 0,
                "BUY_NOW_PASS": 0,
                "FINAL_SCORE": 90.3,
                "ELITE_SCORE": 82.0,
                "ENTRY_GAP_PCT": 1.0,
                "VWAP_GAP": 10.0,
                "POC_GAP": 20.0,
                "RR_NOW_TP1": 1.3,
                "ROUTE": "WAIT",
            },
        ],
    )

    snapshots = load_recommend_snapshots(data)

    by_code = dict(zip(snapshots["code"], snapshots["CANDIDATE_TRIAGE_TYPE"]))
    assert by_code["321370"] == "ENTRY_CLEAN_OBSERVE"
    assert by_code["195870"] == "HIGH_SCORE_OBSERVE"
