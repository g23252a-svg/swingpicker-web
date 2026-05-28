# -*- coding: utf-8 -*-
import pandas as pd

from pipeline_finalize import add_abnormal_history_guard_columns, get_no_buy_breaker_rule_mask


def _base_row(**overrides):
    row = {
        "종목코드": "066430",
        "종목명": "아이로보틱스",
        "ROUTE": "ARMED",
        "FINAL_SCORE": 78.2,
        "STRUCT_SCORE": 96.2,
        "TIMING_SCORE": 28.7,
        "AI_SCORE": 90.7,
        "BUY_NOW_PASS": 1,
        "BUY_NOW_GRADE": "BUY",
        "BUY_NOW_SCORE": 90,
        "BUY_NOW_ELIGIBLE": 1,
        "TOP_PICK": 1,
        "PASS_EBS": 1,
        "거래대금(억원)": 89.29,
        "ENTRY_GAP_PCT": 0.0,
        "RR_NOW_TP1": 4.16,
        "ENTRY_RISK_LEVEL": "GREEN",
        "VWAP_GAP": -12.07,
        "POC_GAP": -14.87,
        "MFI14": 70.2,
        "ret_1d_%": -3.99,
        "ret_5d_%": -14.62,
        "ret_20d_%": 43.02,
        "ret_60d_%": 16.82,
        "ret_120d_%": 198.76,
        "CANDIDATE_TRIAGE_TYPE": "ENTRY_CLEAN_OBSERVE",
    }
    row.update(overrides)
    return row


def test_spike_reversal_candidate_is_production_blocked():
    df = pd.DataFrame([_base_row()])

    out = add_abnormal_history_guard_columns(df)
    row = out.iloc[0]

    assert row["ABNORMAL_HISTORY_GUARD_FLAG"] == 1
    assert row["ABNORMAL_HISTORY_GUARD_LEVEL"] == "BLOCK"
    assert row["ABNORMAL_HISTORY_GUARD_TYPE"] == "SPIKE_REVERSAL"
    assert row["SPIKE_REVERSAL_GUARD_FLAG"] == 1
    assert row["TOP_PICK"] == 0
    assert row["BUY_NOW_ELIGIBLE"] == 0
    assert row["BUY_NOW_PASS"] == 0
    assert row["BUY_NOW_GRADE"] == "AVOID"
    assert row["BUY_NOW_SCORE"] == 0
    assert row["CANDIDATE_TRIAGE_TYPE"] == "EXCLUDED_ABNORMAL_HISTORY"
    assert "초급등" in row["ABNORMAL_HISTORY_GUARD_REASON"]


def test_clean_candidate_is_not_blocked():
    df = pd.DataFrame([
        _base_row(
            종목명="정상후보",
            **{"ret_120d_%": 35.0, "ret_60d_%": 12.0, "ret_20d_%": 8.0, "ret_5d_%": -1.5, "ret_1d_%": -0.2},
        )
    ])

    out = add_abnormal_history_guard_columns(df)
    row = out.iloc[0]

    assert row["ABNORMAL_HISTORY_GUARD_FLAG"] == 0
    assert row["ABNORMAL_HISTORY_GUARD_LEVEL"] == "CLEAR"
    assert row["BUY_NOW_PASS"] == 1
    assert row["BUY_NOW_GRADE"] == "BUY"
    assert row["CANDIDATE_TRIAGE_TYPE"] == "ENTRY_CLEAN_OBSERVE"


def test_market_warning_column_blocks_candidate():
    df = pd.DataFrame([
        _base_row(
            종목명="시장경보종목",
            MARKET_WARNING="투자경고",
            **{"ret_120d_%": 10.0, "ret_20d_%": 2.0, "ret_5d_%": 0.1, "ret_1d_%": 0.0},
        )
    ])

    out = add_abnormal_history_guard_columns(df)
    row = out.iloc[0]

    assert row["ABNORMAL_HISTORY_GUARD_FLAG"] == 1
    assert row["MARKET_WARNING_GUARD_FLAG"] == 1
    assert row["ABNORMAL_HISTORY_GUARD_TYPE"] == "MARKET_WARNING"
    assert row["BUY_NOW_ELIGIBLE"] == 0
    assert row["CANDIDATE_TRIAGE_TYPE"] == "EXCLUDED_ABNORMAL_HISTORY"


def test_no_buy_breaker_cannot_revive_abnormal_candidate():
    df = pd.DataFrame([_base_row(TIMING_SCORE=65.0, FINAL_SCORE=78.0, RR_NOW_TP1=1.5)])
    guarded = add_abnormal_history_guard_columns(df)

    mask = get_no_buy_breaker_rule_mask(guarded, "RULE_D_ROUTE_ARMED_CLEAN_ENTRY")

    assert bool(mask.iloc[0]) is False
    assert guarded.iloc[0]["BUY_NOW_PASS"] == 0
