# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

import components.tab_stocks as ts


def test_today_action_summary_conditional_candidate(monkeypatch):
    cand = pd.DataFrame([
        {
            "종목코드": "028260",
            "종목명": "삼성물산",
            "VALIDATED_ACTION_TIER": "위험시장 조건부",
            "VALIDATED_RR": 1.50,
        }
    ])
    monkeypatch.setattr(ts, "_build_winrate_action_candidates", lambda df, max_n=3: {"candidates": cand})

    df = pd.DataFrame([
        {"TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0, "MACRO_RISK": "CRITICAL"}
    ])
    out = ts._build_today_action_summary(df)

    assert out["official_count"] == 0
    assert out["conditional_count"] == 1
    assert out["market_risk"] == "CRITICAL"
    assert out["first_name"] == "삼성물산"
    assert "신규매수 보류" in out["action"]


def test_today_action_summary_official_candidate(monkeypatch):
    monkeypatch.setattr(ts, "_build_winrate_action_candidates", lambda df, max_n=3: {"candidates": pd.DataFrame()})
    df = pd.DataFrame([
        {"TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1, "MACRO_RISK": "NORMAL"},
        {"TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0, "MACRO_RISK": "NORMAL"},
    ])
    out = ts._build_today_action_summary(df)

    assert out["official_count"] == 1
    assert out["tone"] == "official"
    assert "공식 신규매수" in out["headline"]


def test_table_score_display_marks_zero_as_excluded():
    row = pd.Series({"DISPLAY_SCORE": 0, "ROUTE": "WAIT", "IS_REAL_HOLDING": False})
    assert ts._table_score_display(row) == "검증제외"
