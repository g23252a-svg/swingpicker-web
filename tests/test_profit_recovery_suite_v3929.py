import pandas as pd

from pipeline_finalize import add_profit_recovery_suite_columns, finalize_sort


def _row(**overrides):
    base = {
        "종목코드": "000001",
        "종목명": "테스트",
        "ROUTE": "ARMED",
        "상태": "ARMED",
        "TOP_PICK": 1,
        "BUY_NOW_ELIGIBLE": 1,
        "BUY_NOW_PASS": 1,
        "BUY_NOW_GRADE": "BUY",
        "IS_NOW_ENTRY": 1,
        "IS_ACTIVE": True,
        "IS_WATCH": False,
        "DISPLAY_SCORE": 82.0,
        "FINAL_SCORE": 82.0,
        "STRUCT_SCORE": 92.0,
        "TIMING_SCORE": 55.0,
        "ELITE_SCORE": 60.0,
        "BALANCE_SCORE": 70.0,
        "RR_NOW_TP1": 1.2,
        "ENTRY_GAP_PCT": 0.0,
        "ret_1d_%": 0.0,
        "ret_5d_%": -2.0,
        "VWAP_GAP": 8.0,
        "POC_GAP": 25.0,
        "Vol_Quality": 1.5,
        "MARKET_BREADTH": 62.0,
        "RSI14": 57.0,
        "SECTOR_RS": 3.0,
        "ENTRY_EDGE_LEVEL": "GREEN",
        "ENTRY_RISK_LEVEL": "GREEN",
        "JULY_PROFIT_DEFENSE_LEVEL": "PASS",
        "JULY_PROFIT_BLOCK_FLAG": 0,
        "ABNORMAL_HISTORY_GUARD_FLAG": 0,
        "SPIKE_REVERSAL_GUARD_FLAG": 0,
        "MARKET_WARNING_GUARD_FLAG": 0,
        "LONG_HISTORY_COLLAPSE_FLAG": 0,
        "DATA_INTEGRITY_OK": 1,
        "추천금액(만원)": 100.0,
        "켈리_수량": 10.0,
        "추천수량": 10.0,
    }
    base.update(overrides)
    return base


def test_profit_recovery_blocks_fomo_collision_and_removes_official_buy():
    df = pd.DataFrame([_row(종목명="추격위험")])
    df.loc[0, "ret_5d_%"] = 12.0
    df.loc[0, "VWAP_GAP"] = 22.0
    df.loc[0, "POC_GAP"] = 75.0

    out = add_profit_recovery_suite_columns(df, enforce=True)

    assert int(out.loc[0, "PROFIT_RECOVERY_BLOCK_FLAG"]) == 1
    assert out.loc[0, "PROFIT_RECOVERY_TIER"] == "BLOCK"
    assert out.loc[0, "PROFIT_RECOVERY_SETUP"] == "FOMO_COLLISION"
    assert int(out.loc[0, "TOP_PICK"]) == 0
    assert int(out.loc[0, "BUY_NOW_ELIGIBLE"]) == 0
    assert out.loc[0, "BUY_NOW_GRADE"] == "AVOID"
    assert out.loc[0, "ROUTE"] == "WAIT"
    assert float(out.loc[0, "PROFIT_RECOVERY_SIZE_MULT"]) == 0.0
    assert float(out.loc[0, "추천금액(만원)"]) == 0.0


def test_profit_recovery_scores_pullback_breadth_as_a_tier_without_promotion():
    df = pd.DataFrame([
        _row(TOP_PICK=0, BUY_NOW_ELIGIBLE=0, BUY_NOW_PASS=0, IS_NOW_ENTRY=0, ROUTE="WAIT", 상태="WAIT"),
    ])

    out = add_profit_recovery_suite_columns(df, enforce=True)

    assert int(out.loc[0, "PROFIT_RECOVERY_BLOCK_FLAG"]) == 0
    assert out.loc[0, "PROFIT_RECOVERY_TIER"] == "A"
    assert out.loc[0, "PROFIT_RECOVERY_SETUP"] in {"PULLBACK_BREADTH", "QUALITY_BREADTH"}
    assert float(out.loc[0, "PROFIT_RECOVERY_SCORE"]) >= 78.0
    # 회복 점수는 후보 설명/정렬용이며 공식 매수 승격은 하지 않는다.
    assert int(out.loc[0, "TOP_PICK"]) == 0
    assert int(out.loc[0, "BUY_NOW_ELIGIBLE"]) == 0
    assert float(out.loc[0, "PROFIT_RECOVERY_SIZE_MULT"]) <= 0.70


def test_profit_recovery_blocks_weak_market_knife_catch():
    df = pd.DataFrame([_row(종목명="약세낙폭", MARKET_BREADTH=32.0)])
    df.loc[0, "ret_5d_%"] = -14.0

    out = add_profit_recovery_suite_columns(df, enforce=True)

    assert int(out.loc[0, "PROFIT_RECOVERY_BLOCK_FLAG"]) == 1
    assert out.loc[0, "PROFIT_RECOVERY_SETUP"] == "WEAK_KNIFE"
    assert "약한 시장" in out.loc[0, "PROFIT_RECOVERY_REASON"]


def test_finalize_sort_uses_profit_recovery_score_after_july_score():
    df = pd.DataFrame([
        _row(종목명="낮은회복", TOP_PICK=0, BUY_NOW_ELIGIBLE=0, IS_NOW_ENTRY=1, ROUTE="WAIT", JULY_PROFIT_DEFENSE_SCORE=80.0, PROFIT_RECOVERY_SCORE=55.0, ELITE_SCORE=99.0),
        _row(종목명="높은회복", TOP_PICK=0, BUY_NOW_ELIGIBLE=0, IS_NOW_ENTRY=1, ROUTE="WAIT", JULY_PROFIT_DEFENSE_SCORE=80.0, PROFIT_RECOVERY_SCORE=90.0, ELITE_SCORE=40.0),
    ])

    out = finalize_sort(df)

    assert out.iloc[0]["종목명"] == "높은회복"
