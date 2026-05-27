# -*- coding: utf-8 -*-
"""v22.3.13 오늘 신규진입 판정 카드 회귀 가드."""

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

from components.tab_stocks import _build_daily_official_decision


def test_daily_decision_detects_official_buy_available():
    df = pd.DataFrame([
        {
            "종목코드": "123456",
            "종목명": "공식후보",
            "TOP_PICK": 1,
            "BUY_NOW_ELIGIBLE": 1,
            "BUY_NOW_GRADE": "BUY",
            "FINAL_SCORE": 82.5,
            "ROUTE": "ATTACK",
        }
    ])

    d = _build_daily_official_decision(df)

    assert d["status"] == "OFFICIAL_BUY_AVAILABLE"
    assert d["official_count"] == 1
    assert d["top_pick"]["name"] == "공식후보"
    assert d["blockers"] == []


def test_daily_decision_explains_deferred_top_pick_blockers_and_conditions():
    df = pd.DataFrame([
        {
            "종목코드": "187870",
            "종목명": "디바이스",
            "LDY_RANK": 1,
            "TOP_PICK": 1,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_PASS": 0,
            "BUY_NOW_GRADE": "WATCH",
            "FINAL_SCORE": 88.3,
            "ELITE_SCORE": 77.7,
            "AXIS_GAP": 24.2,
            "RR_NOW_TP1": 1.52,
            "GAP_PCT": 1.5,
            "VWAP_GAP": 35.18,
            "POC_GAP": 106.95,
            "NO_CHASE_FLAG": 1,
            "PULLBACK_WAIT_FLAG": 1,
            "EBS": "8/8 (PASS)",
        }
    ])

    d = _build_daily_official_decision(df)
    blockers = " ".join(d["blockers"])
    conditions = " ".join(d["conversion_conditions"])

    assert d["status"] == "CASH_HOLD_TOP_PICK_DEFERRED"
    assert d["official_count"] == 0
    assert d["top_pick"]["name"] == "디바이스"
    assert "BUY_NOW_ELIGIBLE=0" in blockers
    assert "VWAP_GAP" in blockers
    assert "POC_GAP" in blockers
    assert "NO_CHASE_FLAG" in blockers
    assert "PULLBACK_WAIT_FLAG" in blockers
    assert "BUY_NOW_PASS=1" in conditions
    assert "BUY_NOW_ELIGIBLE=1" in conditions


def test_daily_decision_no_top_pick_is_cash_hold():
    df = pd.DataFrame([
        {"종목코드": "000001", "종목명": "일반종목", "TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0}
    ])

    d = _build_daily_official_decision(df)

    assert d["status"] == "CASH_HOLD_NO_TOP_PICK"
    assert d["official_count"] == 0
    assert d["top_pick_count"] == 0
    assert "현금 유지" in d["summary"]
