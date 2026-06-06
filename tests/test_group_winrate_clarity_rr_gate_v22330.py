# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

import components.tab_stocks as ts


def _summary():
    return pd.DataFrame([
        {
            "METHOD": "FINAL_SCORE",
            "TOPK": 3,
            "H(영업일)": 5,
            "TOTAL_N": 165,
            "WIN_RATE_%": 76.4,
            "AVG_RET_%": 16.36,
            "ALPHA_%": 12.17,
            "HIT_5%_%": 66.1,
        }
    ])


def _row(code, name, route="ARMED", rr=1.5, macro="CRITICAL"):
    return {
        "종목코드": code,
        "종목명": name,
        "ROUTE": route,
        "TOP_PICK": 0,
        "BUY_NOW_ELIGIBLE": 0,
        "FINAL_SCORE": 82,
        "DISPLAY_SCORE": 82,
        "ELITE_SCORE": 70,
        "TIMING_SCORE": 65,
        "STRUCT_SCORE": 70,
        "AI_SCORE": 70,
        "BALANCE_SCORE": 70,
        "AXIS_GAP": 25,
        "RR_NOW_TP1": rr,
        "GAP_PCT": 0,
        "VWAP_GAP": 5,
        "POC_GAP": 10,
        "ENTRY_RISK_LEVEL": "",
        "ENTRY_EDGE_LEVEL": "GREEN",
        "ROUTE_REASON": "",
        "MACRO_RISK": macro,
        "ret_10d_%": 2,
        "ret_20d_%": 5,
        "KELLY_수량": 1,
        "IS_REAL_HOLDING": 0,
    }


def test_critical_market_requires_active_route_and_rr_1_2(monkeypatch):
    monkeypatch.setattr(ts, "_actual_holding_mask", lambda work, holding_codes=None: pd.Series(False, index=work.index))
    df = pd.DataFrame([
        _row("111111", "RR부족", route="ARMED", rr=0.8, macro="CRITICAL"),
        _row("222222", "비활성", route="NEUTRAL", rr=1.8, macro="CRITICAL"),
        _row("333333", "통과", route="ARMED", rr=1.5, macro="CRITICAL"),
    ])

    r = ts._build_winrate_action_candidates(df, summary_df=_summary())
    assert list(r["candidates"]["종목명"]) == ["통과"]
    rejected = set(r["rejected"]["종목명"])
    assert {"RR부족", "비활성"} <= rejected


def test_veto_reason_mentions_critical_rr_and_route():
    reasons = ts._validated_action_veto_reasons(pd.Series({
        "DISPLAY_SCORE": 80,
        "TIMING_SCORE": 65,
        "AXIS_GAP": 20,
        "BALANCE_SCORE": 70,
        "RR_NOW_TP1": 0.8,
        "ROUTE": "NEUTRAL",
        "MACRO_RISK": "CRITICAL",
        "KELLY_수량": 1,
    }))
    assert "CRITICAL 시장에서 ROUTE 비활성" in reasons
    assert "CRITICAL 시장 RR<1.2" in reasons


def test_non_critical_market_can_keep_rr_1_0_plus(monkeypatch):
    monkeypatch.setattr(ts, "_actual_holding_mask", lambda work, holding_codes=None: pd.Series(False, index=work.index))
    df = pd.DataFrame([
        _row("111111", "비위험시장", route="WAIT", rr=1.0, macro="NORMAL"),
    ])

    r = ts._build_winrate_action_candidates(df, summary_df=_summary())
    assert list(r["candidates"]["종목명"]) == ["비위험시장"]
