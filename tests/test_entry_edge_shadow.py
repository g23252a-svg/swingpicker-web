# -*- coding: utf-8 -*-
"""v22.3.10 ENTRY_EDGE shadow production display tests.

이 테스트의 핵심 계약:
- B_red는 ENTRY_EDGE_SCORE 감점/표시만 만든다.
- BUY_NOW_ELIGIBLE / TOP_PICK / BUY_NOW_GRADE는 절대 변경하지 않는다.
"""

import pandas as pd

from pipeline_finalize import add_entry_edge_columns, add_entry_risk_columns


def test_b_red_adds_entry_edge_penalty_without_hard_block():
    df = pd.DataFrame([
        {
            "종목코드": "000001",
            "STRUCT_SCORE": 80,
            "VWAP_GAP": 9.2,
            "TOP_PICK": 1,
            "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 1,
        }
    ])

    out = add_entry_edge_columns(add_entry_risk_columns(df))

    assert out.loc[0, "ENTRY_RISK_RULE"] == "B_RED"
    assert out.loc[0, "ENTRY_EDGE_SCORE"] == 85.0
    assert out.loc[0, "ENTRY_EDGE_LEVEL"] == "CAUTION"
    assert out.loc[0, "ENTRY_EDGE_RULE"] == "B_RED_SHADOW"
    assert out.loc[0, "ENTRY_EDGE_SHADOW_FLAG"] == 1
    assert "공식 매수 차단 아님" in out.loc[0, "ENTRY_EDGE_REASON"]

    # 공식 신규매수 계약은 shadow 표시 때문에 변하면 안 된다.
    assert out.loc[0, "TOP_PICK"] == 1
    assert out.loc[0, "BUY_NOW_GRADE"] == "BUY"
    assert out.loc[0, "BUY_NOW_ELIGIBLE"] == 1


def test_green_row_keeps_entry_edge_neutral():
    df = pd.DataFrame([
        {
            "STRUCT_SCORE": 92,
            "VWAP_GAP": 3.5,
            "BUY_NOW_ELIGIBLE": 0,
        }
    ])

    out = add_entry_edge_columns(add_entry_risk_columns(df))

    assert out.loc[0, "ENTRY_EDGE_SCORE"] == 100.0
    assert out.loc[0, "ENTRY_EDGE_LEVEL"] == "GREEN"
    assert out.loc[0, "ENTRY_EDGE_RULE"] == ""
    assert out.loc[0, "ENTRY_EDGE_REASON"] == ""
    assert out.loc[0, "ENTRY_EDGE_SHADOW_FLAG"] == 0
    assert out.loc[0, "BUY_NOW_ELIGIBLE"] == 0


def test_entry_edge_recomputes_b_red_when_entry_risk_columns_missing():
    df = pd.DataFrame([
        {"STRUCT_SCORE": 85.0, "VWAP_GAP": 8.01, "BUY_NOW_ELIGIBLE": 1},
        {"STRUCT_SCORE": 85.1, "VWAP_GAP": 8.01, "BUY_NOW_ELIGIBLE": 1},
    ])

    out = add_entry_edge_columns(df)

    assert out.loc[0, "ENTRY_EDGE_LEVEL"] == "CAUTION"
    assert out.loc[0, "ENTRY_EDGE_SCORE"] == 85.0
    assert out.loc[1, "ENTRY_EDGE_LEVEL"] == "GREEN"
    assert out.loc[1, "ENTRY_EDGE_SCORE"] == 100.0
    assert out["BUY_NOW_ELIGIBLE"].tolist() == [1, 1]


def test_entry_edge_preserves_existing_buy_now_eligible_series():
    df = pd.DataFrame([
        {"STRUCT_SCORE": 80, "VWAP_GAP": 9, "BUY_NOW_ELIGIBLE": 1},
        {"STRUCT_SCORE": 80, "VWAP_GAP": 9, "BUY_NOW_ELIGIBLE": 0},
        {"STRUCT_SCORE": 92, "VWAP_GAP": 2, "BUY_NOW_ELIGIBLE": 1},
    ])
    before = df["BUY_NOW_ELIGIBLE"].copy()

    out = add_entry_edge_columns(add_entry_risk_columns(df))

    assert out["BUY_NOW_ELIGIBLE"].equals(before)
    assert out["ENTRY_EDGE_SHADOW_FLAG"].tolist() == [1, 1, 0]
