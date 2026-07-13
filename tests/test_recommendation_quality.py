# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.recommendation_quality import (
    POLICY_VERSION,
    apply_recommendation_quality_guard,
    production_buy_mask,
    profit_recovery_candidate_mask,
)


def _row(**overrides):
    row = {
        # Legacy official flags are deliberately zero.  v26.1 must not depend
        # on the impossible historical TOP_PICK+BUY_NOW_ELIGIBLE intersection.
        "TOP_PICK": 0,
        "BUY_NOW_ELIGIBLE": 0,
        "BUY_NOW_GRADE": "BUY",
        "BUY_NOW_SCORE": 82,
        "FINAL_SCORE": 84,
        "STRUCT_SCORE": 88,
        "TIMING_SCORE": 72,
        "TOP_PICK_TYPE": "STABLE",
        "ENTRY_RISK_LEVEL": "GREEN",
        "ROUTE": "NEUTRAL",
        "MACRO_RISK": "NORMAL",
        "MARKET_BREADTH": 45,
        "MFI14": 74,
        "RES_RATIO_NEAR": 0.15,
        "VWAP_GAP": 10,
        "RR_NOW_TP1": 2.0,
        "DISPLAY_SCORE": 84,
        "추천매수가": 10_000,
        "거래대금(억원)": 100,
        "DATA_FRESHNESS_OK": True,
        "ABNORMAL_HISTORY_GUARD_FLAG": 0,
        "PROFIT_RECOVERY_BLOCK_FLAG": 0,
        "JULY_PROFIT_BLOCK_FLAG": 0,
        "NO_CHASE_FLAG": 0,
    }
    row.update(overrides)
    return row


def test_validated_candidate_is_promoted_without_legacy_flags():
    guarded = apply_recommendation_quality_guard(pd.DataFrame([_row()]))
    row = guarded.iloc[0]
    assert row["PROFIT_RECOVERY_CANDIDATE"] == 1
    assert row["PRODUCTION_BUY"] == 1
    assert row["BUY_NOW_ELIGIBLE"] == 1
    assert row["ACTION_DECISION"] == "BUY"
    assert row["QUALITY_POLICY_VERSION"] == POLICY_VERSION
    assert row["PRODUCTION_ENTRY_PRICE"] == 10_000
    assert row["PRODUCTION_STOP_PRICE"] == 9_400
    assert row["PRODUCTION_TARGET_PRICE"] == 11_000
    assert row["RECOMMENDED_WEIGHT_PCT"] == 3
    assert production_buy_mask(guarded).iloc[0]


def test_weak_breadth_keeps_real_setup_visible_as_watch_not_buy():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(MARKET_BREADTH=22.5)])
    )
    row = guarded.iloc[0]
    assert row["PROFIT_RECOVERY_CANDIDATE"] == 1
    assert row["PRODUCTION_BUY"] == 0
    assert row["ACTION_DECISION"] == "WATCH"
    assert "22.5% < 30%" in row["QUALITY_GUARD_REASON"]
    assert "매수 아님" in row["QUALITY_GUARD_REASON"]


def test_critical_macro_blocks_but_caution_only_reduces_size():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([
            _row(MACRO_RISK="CRITICAL"),
            _row(MACRO_RISK="CAUTION", DISPLAY_SCORE=90),
        ])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert guarded.iloc[0]["ACTION_DECISION"] == "WATCH"
    assert guarded.iloc[1]["PRODUCTION_BUY"] == 1
    assert guarded.iloc[1]["RECOMMENDED_WEIGHT_PCT"] == 2


def test_only_one_highest_ranked_setup_is_official():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([
            _row(DISPLAY_SCORE=82, FINAL_SCORE=80),
            _row(DISPLAY_SCORE=92, FINAL_SCORE=90),
        ])
    )
    assert guarded["PRODUCTION_BUY"].sum() == 1
    assert guarded.iloc[1]["PRODUCTION_BUY"] == 1
    assert guarded.iloc[0]["ACTION_DECISION"] == "WATCH"
    assert "당일 신규진입 1종목 제한" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_liquidity_risk_and_no_chase_are_hard_setup_failures():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([
            _row(**{"거래대금(억원)": 29}),
            _row(ENTRY_RISK_LEVEL="ORANGE"),
            _row(NO_CHASE_FLAG=1),
        ])
    )
    assert not profit_recovery_candidate_mask(guarded).any()
    assert guarded["PRODUCTION_BUY"].sum() == 0
    assert guarded["ACTION_DECISION"].eq("CASH").all()


def test_setup_boundaries_are_explicit_and_inclusive():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([
            _row(MFI14=60, RES_RATIO_NEAR=0.03, VWAP_GAP=-20, RR_NOW_TP1=1.2, DISPLAY_SCORE=55),
            _row(MFI14=88, VWAP_GAP=16, DISPLAY_SCORE=56),
            _row(MFI14=59.9),
        ])
    )
    assert guarded["PROFIT_RECOVERY_CANDIDATE"].tolist() == [1, 1, 0]


def test_guard_is_idempotent_and_preserves_legacy_audit_flag():
    first = apply_recommendation_quality_guard(
        pd.DataFrame([_row(BUY_NOW_ELIGIBLE=1, MARKET_BREADTH=22.5)])
    )
    second = apply_recommendation_quality_guard(first)
    assert second.iloc[0]["PRE_QUALITY_BUY_NOW_ELIGIBLE"] == 1
    assert second.iloc[0]["PRODUCTION_BUY"] == 0
    assert second.iloc[0]["ACTION_DECISION"] == "WATCH"
