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
        },
        {
            "METHOD": "ELITE_SCORE",
            "TOPK": 3,
            "H(영업일)": 5,
            "TOTAL_N": 108,
            "WIN_RATE_%": 56.5,
            "AVG_RET_%": 4.73,
            "ALPHA_%": -0.78,
            "HIT_5%_%": 40.7,
        },
    ])


def _row(code, name, final, rr=1.2, risk="", gap=0, route="WAIT", holding=0):
    return {
        "종목코드": code,
        "종목명": name,
        "ROUTE": route,
        "TOP_PICK": 0,
        "BUY_NOW_ELIGIBLE": 0,
        "FINAL_SCORE": final,
        "DISPLAY_SCORE": final,
        "ELITE_SCORE": final - 5,
        "TIMING_SCORE": 70,
        "STRUCT_SCORE": 70,
        "AI_SCORE": 65,
        "RR_NOW_TP1": rr,
        "GAP_PCT": gap,
        "VWAP_GAP": 5,
        "POC_GAP": 10,
        "ENTRY_RISK_LEVEL": risk,
        "ENTRY_EDGE_LEVEL": "GREEN",
        "IS_REAL_HOLDING": holding,
        "종가": 10000,
        "추천매수가": 10000,
        "손절가": 9000,
        "추천매도가1": 12000,
    }


def test_selects_validated_profile_from_rank_summary():
    p = ts._select_winrate_action_profile(_summary())
    assert p["ok"] is True
    assert p["method"] == "FINAL_SCORE"
    assert p["win_rate"] == 76.4


def test_winrate_action_candidates_exclude_red_and_holdings(monkeypatch):
    monkeypatch.setattr(ts, "_actual_holding_mask", lambda work, holding_codes=None: work["IS_REAL_HOLDING"].astype(bool))
    df = pd.DataFrame([
        _row("111111", "강한후보", 88, rr=1.4, risk="", gap=0, route="WAIT"),
        _row("222222", "레드제외", 95, rr=2.0, risk="RED", gap=0, route="ARMED"),
        _row("333333", "보유제외", 96, rr=2.0, risk="", gap=0, route="ARMED", holding=1),
    ])
    r = ts._build_winrate_action_candidates(df, summary_df=_summary())
    out = r["candidates"]
    assert len(out) == 1
    assert out.iloc[0]["종목명"] == "강한후보"
    assert out.iloc[0]["VALIDATED_ACTION_TIER"] == "조건부"


def test_winrate_action_candidates_active_route_gets_priority(monkeypatch):
    monkeypatch.setattr(ts, "_actual_holding_mask", lambda work, holding_codes=None: pd.Series(False, index=work.index))
    df = pd.DataFrame([
        _row("111111", "대기고득점", 90, rr=1.2, route="WAIT"),
        _row("222222", "진입대기", 84, rr=1.3, route="ARMED"),
    ])
    r = ts._build_winrate_action_candidates(df, summary_df=_summary())
    out = r["candidates"]
    assert len(out) >= 2
    assert "진입대기" in set(out["종목명"])
    assert out[out["종목명"] == "진입대기"].iloc[0]["VALIDATED_ACTION_TIER"] == "진입검토"
