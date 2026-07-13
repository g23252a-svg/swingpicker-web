# -*- coding: utf-8 -*-
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fake_nicegui = types.ModuleType("nicegui")
fake_nicegui.ui = types.SimpleNamespace()
sys.modules.setdefault("nicegui", fake_nicegui)

from components.stock_decision_tab import (  # noqa: E402
    build_stock_tab_model,
    filter_stock_rows,
)


def _row(name: str, code: str, **overrides):
    row = {
        "종목명": name,
        "종목코드": code,
        "PRODUCTION_BUY": 0,
        "PROFIT_RECOVERY_CANDIDATE": 0,
        "ACTION_DECISION": "CASH",
        "QUALITY_GUARD_SCORE": 50,
        "DISPLAY_SCORE": 60,
        "추천매수가": 10_000,
        "MARKET_BREADTH": 22.5,
        "MACRO_RISK": "CRITICAL",
        "ML_STATUS": "NOT_VALIDATED",
        "QUALITY_POLICY_VERSION": "v26.1",
        "QUALITY_GUARD_REASON": "최종 기준 미충족",
    }
    row.update(overrides)
    return row


def test_model_separates_official_buy_watch_and_analysis():
    frame = pd.DataFrame([
        _row(
            "공식종목", "1234", PRODUCTION_BUY=1, ACTION_DECISION="BUY",
            QUALITY_GUARD_SCORE=91, PRODUCTION_ENTRY_PRICE=20_000,
            PRODUCTION_STOP_PRICE=18_800, PRODUCTION_TARGET_PRICE=22_000,
            PRODUCTION_RR=1.67, RECOMMENDED_WEIGHT_PCT=3,
        ),
        _row(
            "관찰종목", "005930", PROFIT_RECOVERY_CANDIDATE=1,
            ACTION_DECISION="WATCH", QUALITY_GUARD_SCORE=85,
            QUALITY_GUARD_REASON="시장 상승폭 22.5% < 30% · 매수 아님",
        ),
        _row("분석종목", "000660", QUALITY_GUARD_SCORE=70),
    ])

    model = build_stock_tab_model(frame)

    assert model["status"] == "BUY"
    assert model["counts"] == {"buy": 1, "watch": 1, "analysis": 3}
    assert model["buys"][0]["name"] == "공식종목"
    assert model["buys"][0]["code"] == "001234"
    assert model["watch"][0]["decision"] == "WATCH"


def test_watch_candidates_remain_visible_on_cash_day():
    frame = pd.DataFrame([
        _row(
            "첫후보", "111111", PROFIT_RECOVERY_CANDIDATE=1,
            ACTION_DECISION="WATCH", QUALITY_GUARD_SCORE=88,
        ),
        _row(
            "둘후보", "222222", PROFIT_RECOVERY_CANDIDATE=1,
            ACTION_DECISION="WATCH", QUALITY_GUARD_SCORE=82,
        ),
    ])

    model = build_stock_tab_model(frame)

    assert model["status"] == "CASH"
    assert model["counts"]["buy"] == 0
    assert [item["name"] for item in model["watch"]] == ["첫후보", "둘후보"]
    assert model["market"]["breadth"] == 22.5
    assert model["market"]["macro"] == "CRITICAL"


def test_official_card_uses_production_prices_not_legacy_prices():
    model = build_stock_tab_model(pd.DataFrame([
        _row(
            "가격검증", "333333", PRODUCTION_BUY=1, ACTION_DECISION="BUY",
            추천매수가=10_000, PRODUCTION_ENTRY_PRICE=9_900,
            PRODUCTION_STOP_PRICE=9_306, PRODUCTION_TARGET_PRICE=10_890,
            PRODUCTION_RR=1.67, RECOMMENDED_WEIGHT_PCT=2,
        )
    ]))

    stock = model["buys"][0]
    assert stock["entry"] == 9_900
    assert stock["stop"] == 9_306
    assert stock["target"] == 10_890
    assert stock["weight"] == 2


def test_filter_searches_name_and_padded_code_within_selected_lane():
    model = build_stock_tab_model(pd.DataFrame([
        _row("삼성전자", "5930", PROFIT_RECOVERY_CANDIDATE=1, ACTION_DECISION="WATCH"),
        _row("SK하이닉스", "000660"),
    ]))

    assert [row["name"] for row in filter_stock_rows(model, "관찰 후보", "005930")] == ["삼성전자"]
    assert [row["name"] for row in filter_stock_rows(model, "전체 분석", "하이닉스")] == ["SK하이닉스"]
