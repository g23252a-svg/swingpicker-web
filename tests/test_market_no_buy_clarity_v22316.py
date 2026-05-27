# -*- coding: utf-8 -*-

import sys
import types

# Unit tests target pure helper functions; NiceGUI is not required in CI-lite contexts.
sys.modules.setdefault("nicegui", types.SimpleNamespace(ui=types.SimpleNamespace()))
sys.modules.setdefault(
    "shared_utils",
    types.SimpleNamespace(safe_float=lambda value, default=0: default if value is None else float(value)),
)

from components.tab_market import (
    _blocking_priority_text,
    _combo_section_title,
    _format_macro_delta,
    _is_market_no_buy_mode,
    _match_section_title,
)


def test_macro_delta_hides_nan_and_invalid_prev():
    text, css, value = _format_macro_delta(float("nan"), 1490)
    assert text == "전일 대비 —"
    assert css == "text-xs text-gray-500"
    assert value is None

    text, css, value = _format_macro_delta(1497.1, 0)
    assert text == "전일 대비 —"
    assert css == "text-xs text-gray-500"
    assert value is None


def test_macro_delta_formats_valid_value():
    text, css, value = _format_macro_delta(110, 100)
    assert text == "+10.00%"
    assert css == "text-xs text-green-400"
    assert value == 10


def test_market_no_buy_mode_detects_macro_and_route_block():
    assert _is_market_no_buy_mode("CRITICAL", "ATTACK") is True
    assert _is_market_no_buy_mode("WARNING", "ATTACK") is True
    assert _is_market_no_buy_mode("NORMAL", "WAIT") is True
    assert _is_market_no_buy_mode("NORMAL", "ATTACK") is False


def test_combo_and_match_titles_become_reference_only_in_no_buy_mode():
    assert "오늘 매수 신호 아님" in _combo_section_title(True)
    assert "데이터 기반 최적 조합" in _combo_section_title(False)

    title = _match_section_title(True, 1, "S≥80 T≥80 AI≥70 + ATTACK", 74.0)
    assert "관찰 매칭 종목" in title
    assert "공식 신규매수 아님" in title


def test_blocking_priority_shows_macro_before_score_shortfall():
    lines = _blocking_priority_text("CRITICAL", "ATTACK", "종합 점수 2.9점 부족 (75↑ 필요)")
    assert lines[0] == "1순위 차단: 매크로 위험 CRITICAL"
    assert lines[1] == "2순위 미달: 종합 점수 2.9점 부족 (75↑ 필요)"
