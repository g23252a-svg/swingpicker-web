# -*- coding: utf-8 -*-
"""v22.3.19 관찰 후보 비매수 사유 표시 회귀 가드.

표시 전용 패치이므로 `_build_candidate_triage` 분류 결과는 v22.3.18 그대로 두고,
각 후보 줄에 '왜 공식 신규매수가 아닌지' 사유가 올바르게 붙는지만 고정한다.
"""

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
    _candidate_watch_reason,
    _triage_line,
    _triage_line_with_reason,
    _build_candidate_triage,
)


def test_high_score_watch_reason_shows_vwap_and_poc_overheat():
    """해성디에스류: 점수는 높지만 VWAP/POC 과열 사유가 먼저 표시된다."""
    item = {
        "name": "해성디에스",
        "score": 90.3,
        "rr": 1.17,
        "vwap_gap": 29.0,
        "poc_gap": 70.0,
        "top_pick": False,
        "eligible": False,
    }
    reason = _candidate_watch_reason(item)
    assert reason == "VWAP 과열 +29% · POC 과열 +70%"


def test_entry_watch_reason_shows_official_gating_only():
    """센서뷰류: 가격 위치는 깨끗하므로 TOP_PICK/BUY_NOW 미충족만 사유로 표시된다."""
    item = {
        "name": "센서뷰",
        "score": 68.4,
        "rr": 1.45,
        "vwap_gap": -0.17,
        "poc_gap": 5.27,
        "top_pick": False,
        "eligible": False,
    }
    reason = _candidate_watch_reason(item)
    assert reason == "TOP_PICK 미선정 · BUY_NOW 미충족"


def test_reason_capped_at_two_by_default():
    """사유는 기본 최대 2개로 제한된다 (과열·부족·미선정·미충족 동시 발생 케이스)."""
    item = {
        "name": "복합사유",
        "score": 81.0,
        "rr": 1.05,
        "vwap_gap": 15.0,
        "poc_gap": 40.0,
        "top_pick": False,
        "eligible": False,
    }
    reason = _candidate_watch_reason(item)
    assert reason == "VWAP 과열 +15% · POC 과열 +40%"
    assert reason.count("·") == 1  # 정확히 2개 사유


def test_reason_priority_rr_before_gating():
    """과열이 없으면 RR 부족이 TOP_PICK 미선정보다 먼저 표시된다."""
    item = {
        "name": "RR부족",
        "score": 82.0,
        "rr": 1.10,
        "vwap_gap": 3.0,
        "poc_gap": 10.0,
        "top_pick": False,
        "eligible": False,
    }
    reason = _candidate_watch_reason(item)
    assert reason == "RR 부족 1.10 · TOP_PICK 미선정"


def test_reason_fallback_when_nothing_flags():
    """이론상 모든 조건이 양호하면 일반 사유로 fallback한다."""
    item = {
        "name": "엣지케이스",
        "score": 85.0,
        "rr": 2.0,
        "vwap_gap": 1.0,
        "poc_gap": 5.0,
        "top_pick": True,
        "eligible": True,
    }
    assert _candidate_watch_reason(item) == "공식 신규매수 조건 미충족"


def test_triage_line_with_reason_appends_reason_token():
    """렌더 문자열에 사유 토큰이 포함되고, 기존 _triage_line에는 없어야 한다."""
    items = [{
        "name": "해성디에스",
        "score": 90.3,
        "rr": 1.17,
        "gap": 0.0,
        "vwap_gap": 29.0,
        "poc_gap": 70.0,
        "top_pick": False,
        "eligible": False,
    }]
    enriched = _triage_line_with_reason(items)
    plain = _triage_line(items)

    assert "사유:" in enriched
    assert "VWAP 과열 +29%" in enriched
    assert "사유:" not in plain  # 기존 함수는 그대로 (하위호환)


def test_triage_line_with_reason_handles_empty():
    assert _triage_line_with_reason([]) == "해당 없음"
    assert _triage_line_with_reason([], empty_text="없음") == "없음"


def test_triage_reasons_match_v22318_classification():
    """v22.3.18 분류 결과(센서뷰=진입위치, 해성디에스=고점수)에 사유가 정상 결합되는지 통합 확인."""
    df = pd.DataFrame([
        {
            "종목코드": "321370", "종목명": "센서뷰", "ROUTE": "ARMED",
            "TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0, "BUY_NOW_PASS": 1,
            "BUY_NOW_GRADE": "BUY", "BUY_NOW_SCORE": 80,
            "FINAL_SCORE": 69.1, "ELITE_SCORE": 68.4, "AI_SCORE": 85.9,
            "RR_NOW_TP1": 1.45, "GAP_PCT": 0.0, "VWAP_GAP": -0.17, "POC_GAP": 5.27,
        },
        {
            "종목코드": "195870", "종목명": "해성디에스", "ROUTE": "WAIT",
            "TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0, "BUY_NOW_PASS": 0,
            "BUY_NOW_GRADE": "AVOID", "BUY_NOW_SCORE": 0,
            "FINAL_SCORE": 90.3, "ELITE_SCORE": 89.5, "AI_SCORE": 86.9,
            "RR_NOW_TP1": 1.17, "GAP_PCT": 0.0, "VWAP_GAP": 29.0, "POC_GAP": 70.0,
        },
    ])
    triage = _build_candidate_triage(df)

    entry_line = _triage_line_with_reason(triage["entry_watch"])
    high_line = _triage_line_with_reason(triage["high_score_watch"])

    assert "센서뷰" in entry_line and "TOP_PICK 미선정" in entry_line
    assert "해성디에스" in high_line and "VWAP 과열 +29%" in high_line
