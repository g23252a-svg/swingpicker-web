# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

import components.tab_stocks as ts


def _row(code: str, name: str, route: str = "CARRY") -> dict:
    return {
        "종목코드": code,
        "종목명": name,
        "ROUTE": route,
        "상태": "보유관리",
        "ACTION_LABEL": "보유관리",
        "TOP_PICK": 0,
        "BUY_NOW_ELIGIBLE": 0,
        "BUY_NOW_PASS": 0,
        "BUY_NOW_GRADE": "AVOID",
        "ELITE_SCORE": 70,
        "DISPLAY_SCORE": 70,
        "FINAL_SCORE": 70,
        "AI_SCORE": 60,
        "RR_NOW_TP1": 1.5,
        "GAP_PCT": 0,
        "VWAP_GAP": 0,
        "POC_GAP": 0,
    }


def test_stale_carry_display_is_rewritten_when_not_in_positions(monkeypatch):
    monkeypatch.setattr(ts, "_load_actual_holding_codes", lambda data_dir=None: {"111111"})
    df = pd.DataFrame([
        _row("111111", "진짜보유"),
        _row("222222", "가짜보유"),
    ])

    out = ts._apply_holding_ssot_display_guard(df)

    real = out[out["종목코드"] == "111111"].iloc[0]
    stale = out[out["종목코드"] == "222222"].iloc[0]

    assert int(real["IS_REAL_HOLDING"]) == 1
    assert real["ROUTE"] == "CARRY"
    assert int(stale["IS_REAL_HOLDING"]) == 0
    assert stale["ROUTE"] == "WAIT"
    assert stale["상태"] == "관망"


def test_candidate_triage_holding_manage_uses_positions_json_only(monkeypatch):
    monkeypatch.setattr(ts, "_load_actual_holding_codes", lambda data_dir=None: {"111111"})
    df = pd.DataFrame([
        _row("111111", "진짜보유"),
        _row("222222", "가짜보유"),
    ])

    guarded = ts._apply_holding_ssot_display_guard(df)
    triage = ts._build_candidate_triage(guarded)

    names = [x["name"] for x in triage["holding_manage"]]
    assert names == ["진짜보유"]
