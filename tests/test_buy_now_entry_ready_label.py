# -*- coding: utf-8 -*-
"""v22.3.11 BUY_NOW 표시 라벨 오해 방지 회귀 가드."""

from components.buy_now_badge import get_buy_now_display, format_buy_now_subtitle, format_buy_now_tooltip


def test_buy_but_not_eligible_subtitle_says_entry_condition_not_official_buy():
    row = {
        "TOP_PICK": 1,
        "BUY_NOW_GRADE": "BUY",
        "BUY_NOW_ELIGIBLE": 0,
        "BUY_NOW_SCORE": 75,
    }
    disp = get_buy_now_display(row)
    sub = format_buy_now_subtitle(disp)

    assert "진입조건" in sub
    assert "공식 신규매수" not in sub
    assert "🟢" not in sub


def test_official_buy_subtitle_can_say_official_new_buy():
    row = {
        "TOP_PICK": 1,
        "BUY_NOW_GRADE": "BUY",
        "BUY_NOW_ELIGIBLE": 1,
        "BUY_NOW_SCORE": 85,
    }
    disp = get_buy_now_display(row)
    sub = format_buy_now_subtitle(disp)

    assert "공식 신규매수" in sub
    assert "🟢" in sub


def test_buy_but_not_eligible_tooltip_explains_entry_condition_only():
    row = {
        "TOP_PICK": 1,
        "BUY_NOW_GRADE": "BUY",
        "BUY_NOW_ELIGIBLE": 0,
        "BUY_NOW_SCORE": 80,
    }
    disp = get_buy_now_display(row)
    tip = format_buy_now_tooltip(disp)

    assert "진입조건" in tip
    assert "공식 매수 대상 아님" in tip
