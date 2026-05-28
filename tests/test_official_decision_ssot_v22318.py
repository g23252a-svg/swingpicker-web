# -*- coding: utf-8 -*-
"""v22.3.18 공식 판정 SSOT + 후보 유형 분리 회귀 가드."""

import sys
import types

import pandas as pd


class _DummyNiceGuiObject:
    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return self


if "nicegui" not in sys.modules:
    nicegui_stub = types.ModuleType("nicegui")
    nicegui_stub.ui = _DummyNiceGuiObject()
    nicegui_stub.run = _DummyNiceGuiObject()
    nicegui_stub.app = _DummyNiceGuiObject()
    sys.modules["nicegui"] = nicegui_stub

from components.tab_stocks import (  # noqa: E402
    _build_candidate_triage,
    _build_daily_official_decision,
    _official_decision_allows_entry,
    _resolve_member_summary_action,
)


def test_official_no_buy_forces_member_summary_to_hold_even_when_ev_positive():
    decision = {
        "status": "CASH_HOLD_NO_TOP_PICK",
        "official_count": 0,
        "top_pick_count": 0,
    }

    action = _resolve_member_summary_action(
        ev=0.05,
        tp1_rate=0.239,
        cap_ret=6.29,
        official_decision=decision,
    )

    assert action["status_txt"] == "매매 보류"
    assert "관찰 전용" in action["action_txt"]
    assert "진입" not in action["status_txt"]


def test_official_buy_available_allows_entry_wording_path():
    decision = {
        "status": "OFFICIAL_BUY_AVAILABLE",
        "official_count": 1,
        "top_pick_count": 1,
    }

    assert _official_decision_allows_entry(decision) is True

    action = _resolve_member_summary_action(
        ev=1.2,
        tp1_rate=0.40,
        cap_ret=3.0,
        official_decision=decision,
    )

    assert action["status_txt"] == "추천 신뢰 양호"


def test_candidate_triage_splits_entry_watch_and_high_score_watch():
    df = pd.DataFrame([
        {
            "종목코드": "321370",
            "종목명": "센서뷰",
            "ROUTE": "ARMED",
            "TOP_PICK": 0,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_PASS": 1,
            "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_SCORE": 80,
            "FINAL_SCORE": 69.1,
            "ELITE_SCORE": 68.4,
            "AI_SCORE": 85.9,
            "RR_NOW_TP1": 1.45,
            "GAP_PCT": 0.0,
            "VWAP_GAP": -0.17,
            "POC_GAP": 5.27,
        },
        {
            "종목코드": "195870",
            "종목명": "해성디에스",
            "ROUTE": "WAIT",
            "TOP_PICK": 0,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_PASS": 0,
            "BUY_NOW_GRADE": "AVOID",
            "BUY_NOW_SCORE": 0,
            "FINAL_SCORE": 90.3,
            "ELITE_SCORE": 89.5,
            "AI_SCORE": 86.9,
            "RR_NOW_TP1": 1.17,
            "GAP_PCT": 0.0,
            "VWAP_GAP": 29.0,
            "POC_GAP": 70.0,
        },
    ])

    triage = _build_candidate_triage(df)

    assert triage["official_buy"] == []
    assert triage["entry_watch"][0]["name"] == "센서뷰"
    assert triage["high_score_watch"][0]["name"] == "해성디에스"


def test_daily_decision_still_controls_official_buy_ssot():
    df = pd.DataFrame([
        {"종목코드": "321370", "종목명": "센서뷰", "TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0, "BUY_NOW_PASS": 1}
    ])

    decision = _build_daily_official_decision(df)

    assert decision["status"] == "CASH_HOLD_NO_TOP_PICK"
    assert _official_decision_allows_entry(decision) is False
