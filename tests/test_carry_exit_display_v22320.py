# -*- coding: utf-8 -*-
"""v22.3.20 보유경과 청산 검토 신호 표시 회귀 가드.

대상: components.tab_portfolio_v2._classify_holding
- DEAD/CARRY_EXIT_SIGNAL=1 → '🔴 청산 검토' (점수 높아도 우선)
- DART 경고는 그보다 우선
- 표현 안전: '매도/자동/팔' 금지
- carry_stale_reason 이 결과 dict로 전달
"""
import sys
import types


class _Dummy:
    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return self


# nicegui / plotly 스텁 (테스트 환경 독립)
for _m in ("nicegui",):
    if _m not in sys.modules:
        _mod = types.ModuleType(_m)
        _mod.ui = _Dummy()
        _mod.app = _Dummy()
        _mod.run = _Dummy()
        sys.modules[_m] = _mod

if "plotly" not in sys.modules:
    _pl = types.ModuleType("plotly")
    _ple = types.ModuleType("plotly.express")
    _plg = types.ModuleType("plotly.graph_objects")
    sys.modules["plotly"] = _pl
    sys.modules["plotly.express"] = _ple
    sys.modules["plotly.graph_objects"] = _plg

from components.tab_portfolio_v2 import _classify_holding  # noqa: E402


def test_dead_exit_signal_classified_as_review():
    cls = _classify_holding(
        80, "CARRY",
        carry_exit_signal=1,
        carry_stale_stage="DEAD",
        carry_stale_reason="보유 10일차 · 손익 -9.0% · 강한 청산 검토",
    )
    assert cls["group"] == "caution"
    assert "청산 검토" in cls["action"]
    assert cls["carry_stale_reason"].startswith("보유 10일차")


def test_exit_signal_overrides_high_score_hold():
    """점수 80·CARRY면 원래 '보유 유지'지만, 청산 신호가 우선한다."""
    base = _classify_holding(80, "CARRY")
    assert base["action"] == "✅ 보유 유지"

    with_signal = _classify_holding(80, "CARRY", carry_exit_signal=1)
    assert with_signal["action"] == "🔴 청산 검토"
    assert with_signal["group"] == "caution"


def test_dart_warning_takes_priority_over_exit_signal():
    cls = _classify_holding(80, "CARRY", has_dart_warning=True, carry_exit_signal=1)
    assert cls["action"] == "🚨 공시 주의 + 검토"


def test_no_exit_signal_keeps_existing_behavior():
    """신호 0이면 기존 분류 그대로 (하위호환)."""
    assert _classify_holding(80, "CARRY")["action"] == "✅ 보유 유지"
    assert _classify_holding(30, "CARRY")["action"] == "🚨 교체 검토"
    assert _classify_holding(50, "WAIT")["action"] == "⚠️ 지켜보기"
    # 명시적으로 0 전달해도 동일
    assert _classify_holding(80, "CARRY", carry_exit_signal=0)["action"] == "✅ 보유 유지"


def test_exit_signal_accepts_various_truthy_types():
    """CSV에서 문자열/실수로 올 수 있는 신호값을 안전하게 해석한다."""
    assert _classify_holding(80, "CARRY", carry_exit_signal=1.0)["action"] == "🔴 청산 검토"
    assert _classify_holding(80, "CARRY", carry_exit_signal="1")["action"] == "🔴 청산 검토"
    assert _classify_holding(80, "CARRY", carry_exit_signal="0")["action"] == "✅ 보유 유지"
    assert _classify_holding(80, "CARRY", carry_exit_signal=None)["action"] == "✅ 보유 유지"


def test_carry_reason_present_in_all_results():
    """모든 분기에서 carry_stale_reason 키가 존재한다 (카드 렌더 안전)."""
    for kwargs in (
        {},
        {"carry_exit_signal": 1, "carry_stale_reason": "보유 12일차 · 손익 -7.0% · 청산 검토"},
        {"has_dart_warning": True, "carry_stale_reason": "x"},
    ):
        res = _classify_holding(80, "CARRY", **kwargs)
        assert "carry_stale_reason" in res


def test_no_unsafe_sell_wording_in_actions():
    """모든 액션 라벨에 '매도/자동/팔' 표현이 없어야 한다."""
    actions = [
        _classify_holding(80, "CARRY", carry_exit_signal=1)["action"],
        _classify_holding(30, "CARRY")["action"],
        _classify_holding(50, "WAIT")["action"],
        _classify_holding(80, "CARRY")["action"],
        _classify_holding(80, "CARRY", has_dart_warning=True)["action"],
    ]
    joined = " ".join(actions)
    for banned in ("매도", "자동", "팔"):
        assert banned not in joined, f"금지 표현 '{banned}' 발견: {joined}"
