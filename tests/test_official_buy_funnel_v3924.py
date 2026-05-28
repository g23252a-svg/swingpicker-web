# -*- coding: utf-8 -*-
"""v3.9.24 Official Buy Funnel & Macro Regime Shadow tests."""

import pandas as pd

from pipeline_finalize import add_official_buy_funnel_columns


def test_official_buy_funnel_preserves_official_contract_columns():
    df = pd.DataFrame([
        {
            "종목명": "공식",
            "ROUTE": "ATTACK",
            "TOP_PICK": 1,
            "BUY_NOW_ELIGIBLE": 1,
            "BUY_NOW_PASS": 1,
            "BUY_NOW_GRADE": "BUY",
            "FINAL_SCORE": 82,
            "ELITE_SCORE": 80,
            "RR_NOW_TP1": 1.6,
            "ENTRY_GAP_PCT": 0.0,
            "VWAP_GAP": 1.0,
            "POC_GAP": 5.0,
            "PASS_EBS": 1,
        },
        {
            "종목명": "진입깨끗",
            "ROUTE": "ARMED",
            "TOP_PICK": 0,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_PASS": 1,
            "BUY_NOW_GRADE": "BUY",
            "FINAL_SCORE": 69,
            "ELITE_SCORE": 68,
            "RR_NOW_TP1": 1.45,
            "ENTRY_GAP_PCT": 0.0,
            "VWAP_GAP": -0.2,
            "POC_GAP": 5.0,
            "PASS_EBS": 1,
        },
    ])

    before = df[["TOP_PICK", "BUY_NOW_ELIGIBLE", "BUY_NOW_PASS", "BUY_NOW_GRADE"]].copy()
    out = add_official_buy_funnel_columns(df, macro_risk="NORMAL", market_breadth=52)

    pd.testing.assert_frame_equal(
        out[["TOP_PICK", "BUY_NOW_ELIGIBLE", "BUY_NOW_PASS", "BUY_NOW_GRADE"]],
        before,
    )
    assert out.loc[0, "STRICT_OFFICIAL_BUY_ELIGIBLE"] == 1
    assert out.loc[0, "CANDIDATE_TRIAGE_TYPE"] == "OFFICIAL_BUY"
    assert out.loc[1, "OFFICIAL_FUNNEL_STAGE"] == "ENTRY_READY_BUT_NOT_TOP_PICK"
    assert out.loc[1, "CANDIDATE_TRIAGE_TYPE"] == "ENTRY_CLEAN_OBSERVE"
    assert out.loc[1, "OFFICIAL_BLOCK_REASON_1"] == "TOP_PICK=0"


def test_official_buy_funnel_splits_high_score_and_chase_risk():
    df = pd.DataFrame([
        {
            "종목명": "고점수관찰",
            "ROUTE": "WAIT",
            "TOP_PICK": 0,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_PASS": 0,
            "FINAL_SCORE": 90,
            "ELITE_SCORE": 89,
            "RR_NOW_TP1": 1.2,
            "ENTRY_GAP_PCT": 0.0,
            "VWAP_GAP": 29,
            "POC_GAP": 70,
            "PASS_EBS": 1,
        },
        {
            "종목명": "추격위험",
            "ROUTE": "ATTACK",
            "TOP_PICK": 0,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_PASS": 0,
            "FINAL_SCORE": 85,
            "ELITE_SCORE": 84,
            "RR_NOW_TP1": 0.98,
            "ENTRY_GAP_PCT": 16.4,
            "VWAP_GAP": 10,
            "POC_GAP": 20,
            "PASS_EBS": 1,
        },
    ])

    out = add_official_buy_funnel_columns(df)

    assert out.loc[0, "CANDIDATE_TRIAGE_TYPE"] == "HIGH_SCORE_OBSERVE"
    assert out.loc[0, "OFFICIAL_FUNNEL_STAGE"] == "HIGH_SCORE_BUT_ENTRY_BLOCKED"
    assert out.loc[1, "CANDIDATE_TRIAGE_TYPE"] == "CHASE_RISK"
    assert out.loc[1, "OFFICIAL_FUNNEL_STAGE"] == "ROUTE_ACTIVE_BUT_CHASE_RISK"
    assert out.loc[1, "OFFICIAL_BLOCK_REASON_2"] in {"추천가 괴리 과다", "RR_NOW_TP1 부족"}


def test_macro_regime_shadow_distinguishes_high_fx_from_internal_weakness():
    base = pd.DataFrame([
        {
            "ROUTE": "ARMED",
            "TOP_PICK": 0,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_PASS": 1,
            "FINAL_SCORE": 70,
            "ELITE_SCORE": 70,
            "RR_NOW_TP1": 1.3,
            "ENTRY_GAP_PCT": 0.0,
            "VWAP_GAP": 2,
            "POC_GAP": 5,
            "PASS_EBS": 1,
        }
    ])

    ok = add_official_buy_funnel_columns(
        base,
        macro_risk="CRITICAL",
        market_breadth=52,
        macro_msg="환율 1515원 [05/25] (CRITICAL)",
    )
    weak = add_official_buy_funnel_columns(
        base,
        macro_risk="CRITICAL",
        market_breadth=17,
        macro_msg="환율 1515원 [05/25] (CRITICAL)",
    )

    assert ok.loc[0, "FX_HIGH_REGIME_FLAG"] == 1
    assert ok.loc[0, "MARKET_INTERNAL_WEAK_FLAG"] == 0
    assert ok.loc[0, "MACRO_REGIME_MODE"] == "FX_HIGH_REGIME"
    assert ok.loc[0, "SHADOW_MACRO_RELAXED_ELIGIBLE"] == 1

    assert weak.loc[0, "FX_HIGH_REGIME_FLAG"] == 1
    assert weak.loc[0, "MARKET_INTERNAL_WEAK_FLAG"] == 1
    assert weak.loc[0, "MACRO_REGIME_MODE"] == "FX_HIGH_AND_INTERNAL_WEAK"
    assert weak.loc[0, "SHADOW_MACRO_RELAXED_ELIGIBLE"] == 0
