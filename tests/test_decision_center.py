# -*- coding: utf-8 -*-
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
fake_nicegui = types.ModuleType("nicegui")
fake_nicegui.ui = types.SimpleNamespace()
sys.modules.setdefault("nicegui", fake_nicegui)

from components.decision_center import build_decision_summary


def test_cash_is_explicit_when_no_production_buy():
    frame = pd.DataFrame([
        {
            "종목코드": "005930",
            "종목명": "삼성전자",
            "PRODUCTION_BUY": 0,
            "ACTION_DECISION": "WATCH",
            "QUALITY_GUARD_SCORE": 68,
            "QUALITY_GUARD_REASON": "즉시매수 기준 미달",
            "ML_STATUS": "UNVALIDATED_MODEL",
        }
    ])
    result = build_decision_summary(frame)
    assert result["status"] == "CASH"
    assert "신규매수하지 않습니다" in result["title"]
    assert result["production_count"] == 0


def test_only_production_buy_is_shown_as_buy():
    frame = pd.DataFrame([
        {
            "종목코드": "005930",
            "종목명": "삼성전자",
            "PRODUCTION_BUY": 1,
            "ACTION_DECISION": "BUY",
            "QUALITY_GUARD_SCORE": 88,
            "QUALITY_GUARD_REASON": "엄격 매수 기준 통과",
            "추천매수가": 70000,
            "손절가": 66500,
            "추천매도가1": 78000,
            "RR_NOW_TP1": 2.2,
            "RECOMMENDED_WEIGHT_PCT": 5,
            "ML_STATUS": "VALIDATED",
        },
        {
            "종목코드": "000660",
            "종목명": "SK하이닉스",
            "TOP_PICK": 1,
            "BUY_NOW_ELIGIBLE": 1,
            "PRODUCTION_BUY": 0,
            "ACTION_DECISION": "WATCH",
            "QUALITY_GUARD_SCORE": 70,
            "QUALITY_GUARD_REASON": "공격형 검증 실패",
        },
    ])
    result = build_decision_summary(frame)
    assert result["status"] == "BUY"
    assert result["production_count"] == 1
    assert [stock["code"] for stock in result["buys"]] == ["005930"]

