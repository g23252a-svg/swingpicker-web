# -*- coding: utf-8 -*-
import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
fake_nicegui = types.ModuleType("nicegui")
fake_nicegui.ui = types.SimpleNamespace()
sys.modules.setdefault("nicegui", fake_nicegui)

from components.decision_center import _humanize_reason, build_decision_summary


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
    assert result["action_label"] == "신규 주문을 넣지 말고 현금을 유지하세요"
    assert result["gates"][2]["value"] == "신규 주문 없음"


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
            "추천매도가2": 85000,
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
    assert result["buys"][0]["target"] == 78000
    assert result["buys"][0]["target2"] == 85000
    assert result["gates"][2]["status"] == "가능"


def test_market_readiness_and_watch_count_are_explained():
    frame = pd.DataFrame([
        {
            "종목코드": "138040",
            "종목명": "메리츠금융지주",
            "PRODUCTION_BUY": 0,
            "ACTION_DECISION": "WATCH",
            "QUALITY_GUARD_SCORE": 73,
            "QUALITY_GUARD_REASON": "TOP_PICK 아님 · 경로 WAIT · 상단여력 부족 · 시장폭 28% (내부 약세)",
            "MARKET_BREADTH": 27.6,
            "MACRO_RISK": "CAUTION",
            "MARKET_REGIME": "NEUTRAL",
        },
        {
            "종목코드": "045100",
            "종목명": "한양이엔지",
            "PRODUCTION_BUY": 0,
            "ACTION_DECISION": "WATCH",
            "QUALITY_GUARD_SCORE": 70,
            "MARKET_BREADTH": 27.6,
            "MACRO_RISK": "CAUTION",
            "MARKET_REGIME": "NEUTRAL",
        },
    ])

    result = build_decision_summary(frame)

    assert result["market_ready"] is False
    assert result["market_breadth"] == 27.6
    assert result["watch_count"] == 2
    assert result["gates"][0]["status"] == "대기"
    assert result["gates"][1]["value"] == "공식 0개 · 관찰 2개"
    assert result["blockers"] == [
        "상승 종목 비율 28%로 기준 35% 미만",
        "최종 선별 단계 미통과",
        "진입 타이밍 대기",
    ]


def test_engine_reasons_are_translated_for_non_experts():
    reason = _humanize_reason(
        "TOP_PICK 아님 · 경로 WAIT · VWAP 적정구간 아님 · 시장폭 28% (내부 약세)"
    )

    assert "TOP_PICK" not in reason
    assert "WAIT" not in reason
    assert "VWAP" not in reason
    assert "최종 선별 단계 미통과" in reason
    assert "진입 타이밍 대기" in reason
    assert "현재가가 적정 진입 구간 밖" in reason
    assert "상승 종목 비율 28%로 시장 내부가 약함" in reason


def test_cash_rows_are_not_mislabeled_as_watch_candidates():
    frame = pd.DataFrame([
        {
            "종목코드": "005930",
            "종목명": "삼성전자",
            "PRODUCTION_BUY": 0,
            "ACTION_DECISION": "CASH",
            "QUALITY_GUARD_SCORE": 90,
        }
    ])

    result = build_decision_summary(frame)

    assert result["watch"] == []
    assert result["watch_count"] == 0

