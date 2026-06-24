import pandas as pd

from pipeline_finalize import add_july_profit_defense_columns, finalize_sort


def _base_row(**overrides):
    row = {
        "종목명": "테스트",
        "TOP_PICK": 1,
        "BUY_NOW_ELIGIBLE": 1,
        "BUY_NOW_PASS": 1,
        "BUY_NOW_GRADE": "BUY",
        "BUY_NOW_REASON": "",
        "IS_NOW_ENTRY": 1,
        "ROUTE": "ARMED",
        "상태": "ARMED",
        "IS_ACTIVE": True,
        "IS_WATCH": False,
        "MARKET_BREADTH": 55.0,
        "ret_5d_%": 2.0,
        "VWAP_GAP": 10.0,
        "POC_GAP": 30.0,
        "Vol_Quality": 1.5,
        "STRUCT_SCORE": 85.0,
        "FINAL_SCORE": 80.0,
        "ELITE_SCORE": 70.0,
        "RR_NOW_TP1": 1.5,
        "BALANCE_SCORE": 50.0,
        "ENTRY_GAP_PCT": 0.0,
        "DISPLAY_SCORE": 80.0,
        "RSI14": 58.0,
        "SECTOR_RS": 5.0,
        "ENTRY_EDGE_LEVEL": "GREEN",
        "ENTRY_RISK_LEVEL": "GREEN",
        "ABNORMAL_HISTORY_GUARD_FLAG": 0,
        "SPIKE_REVERSAL_GUARD_FLAG": 0,
        "MARKET_WARNING_GUARD_FLAG": 0,
        "LONG_HISTORY_COLLAPSE_FLAG": 0,
    }
    row.update(overrides)
    return row


def test_july_profit_defense_blocks_overheated_official_buy():
    df = pd.DataFrame([
        _base_row(종목명="과열주", **{"ret_5d_%": 28.0, "VWAP_GAP": 42.0, "POC_GAP": 90.0}),
    ])

    out = add_july_profit_defense_columns(df, enforce=True)

    assert int(out.loc[0, "JULY_PROFIT_BLOCK_FLAG"]) == 1
    assert out.loc[0, "JULY_PROFIT_DEFENSE_LEVEL"] == "BLOCK"
    assert int(out.loc[0, "BUY_NOW_ELIGIBLE"]) == 0
    assert int(out.loc[0, "BUY_NOW_PASS"]) == 0
    assert int(out.loc[0, "TOP_PICK"]) == 0
    assert bool(out.loc[0, "NEW_ENTRY_BLOCKED"]) is True
    assert str(out.loc[0, "BUY_NOW_GRADE"]) == "AVOID"
    assert "7월 손실방어" in str(out.loc[0, "BUY_NOW_REASON"])


def test_july_profit_defense_keeps_clean_profile():
    df = pd.DataFrame([_base_row(종목명="방어통과")])

    out = add_july_profit_defense_columns(df, enforce=True)

    assert int(out.loc[0, "JULY_PROFIT_PROFILE_PASS"]) == 1
    assert out.loc[0, "JULY_PROFIT_DEFENSE_LEVEL"] == "PASS"
    assert int(out.loc[0, "BUY_NOW_ELIGIBLE"]) == 1
    assert int(out.loc[0, "TOP_PICK"]) == 1
    assert float(out.loc[0, "JULY_PROFIT_DEFENSE_SCORE"]) >= 80.0


def test_finalize_sort_prefers_july_defense_score_within_same_route():
    df = pd.DataFrame([
        _base_row(종목명="고위험고점", TOP_PICK=0, IS_NOW_ENTRY=1, JULY_PROFIT_DEFENSE_SCORE=30.0, ELITE_SCORE=95.0),
        _base_row(종목명="방어우선", TOP_PICK=0, IS_NOW_ENTRY=1, JULY_PROFIT_DEFENSE_SCORE=90.0, ELITE_SCORE=70.0),
    ])

    out = finalize_sort(df)

    assert out.loc[0, "종목명"] == "방어우선"
