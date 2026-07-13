# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.recommendation_quality import (
    apply_recommendation_quality_guard,
    production_buy_mask,
)


def _row(**overrides):
    row = {
        "TOP_PICK": 1,
        "BUY_NOW_ELIGIBLE": 1,
        "BUY_NOW_GRADE": "BUY",
        "BUY_NOW_SCORE": 82,
        "TOP_PICK_TYPE": "STABLE",
        "ENTRY_RISK_LEVEL": "GREEN",
        "ROUTE": "ATTACK",
        "MACRO_RISK": "NORMAL",
        "MFI14": 74,
        "RES_RATIO_NEAR": 0.15,
        "VWAP_GAP": 10,
        "RR_NOW_TP1": 2.0,
        "DISPLAY_SCORE": 84,
        "DATA_FRESHNESS_OK": True,
        "ABNORMAL_HISTORY_GUARD_FLAG": 0,
        "PROFIT_RECOVERY_BLOCK_FLAG": 0,
        "JULY_PROFIT_BLOCK_FLAG": 0,
    }
    row.update(overrides)
    return row


def test_strict_stable_candidate_can_be_production_buy():
    guarded = apply_recommendation_quality_guard(pd.DataFrame([_row()]))
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 1
    assert guarded.iloc[0]["ACTION_DECISION"] == "BUY"
    assert guarded.iloc[0]["RECOMMENDED_WEIGHT_PCT"] <= 5
    assert production_buy_mask(guarded).iloc[0]


def test_aggressive_candidate_is_blocked_after_observed_losses():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(TOP_PICK_TYPE="AGGRESSIVE")])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert guarded.iloc[0]["BUY_NOW_ELIGIBLE"] == 0
    assert "공격형 검증 실패" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_guard_never_promotes_non_official_candidate():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(TOP_PICK=0, BUY_NOW_ELIGIBLE=0)])
    )
    assert guarded.iloc[0]["QUALITY_GUARD_PASS"] == 1
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert guarded.iloc[0]["BUY_NOW_ELIGIBLE"] == 0


def test_risk_and_resistance_failures_force_cash_or_watch():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(ENTRY_RISK_LEVEL="ORANGE", RES_RATIO_NEAR=0.01)])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert guarded.iloc[0]["RECOMMENDED_WEIGHT_PCT"] == 0


def test_low_buy_now_score_is_never_official():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(BUY_NOW_SCORE=60)])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert "즉시매수점수 60" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_armed_route_is_watch_only_after_realised_loss_audit():
    guarded = apply_recommendation_quality_guard(pd.DataFrame([_row(ROUTE="ARMED")]))
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert "경로 ARMED" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_only_one_new_position_can_be_production_buy():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(DISPLAY_SCORE=82), _row(DISPLAY_SCORE=90)])
    )
    assert guarded["PRODUCTION_BUY"].sum() == 1
    assert guarded.iloc[1]["PRODUCTION_BUY"] == 1
    assert "당일 신규진입 1종목 제한" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_guard_is_idempotent_and_preserves_pre_quality_audit_flag():
    first = apply_recommendation_quality_guard(
        pd.DataFrame([_row(TOP_PICK_TYPE="AGGRESSIVE")])
    )
    second = apply_recommendation_quality_guard(first)
    assert second.iloc[0]["PRE_QUALITY_BUY_NOW_ELIGIBLE"] == 1
    assert second.iloc[0]["PRODUCTION_BUY"] == 0
