# -*- coding: utf-8 -*-
"""v22.3.8 UI safety regression tests.

화면 라벨이 BUY_NOW_GRADE/ELITE_LABEL만 보고 신규매수 가능처럼 보이지 않도록 보호한다.
공식 신규매수 기준은 TOP_PICK + BUY_NOW_ELIGIBLE이다.
"""
from pathlib import Path

from components.ui_terms import ELITE_LABEL_DISPLAY, ELITE_LABEL_DISPLAY_SHORT


def test_instant_internal_label_is_displayed_as_observation_candidate():
    assert ELITE_LABEL_DISPLAY["✅ 즉시진입"] == "🟡 관찰 후보"
    assert ELITE_LABEL_DISPLAY_SHORT["✅ 즉시진입"] == "🟡 관찰 후보"


def test_tab_stocks_has_no_unsafe_entry_possible_copy():
    src = Path("components/tab_stocks.py").read_text(encoding="utf-8")
    unsafe_phrases = [
        "오늘 즉시 매수 적합",
        "🟢 진입가능(엄격필터)",
        "🟢 진입가능 {instant_n}",
        "🟢 진입가능:",
        "🟢 진입가능 (",
        "🟢 진입가능 ×1.30",
        "🟢 진입가능 우선",
    ]
    for phrase in unsafe_phrases:
        assert phrase not in src
