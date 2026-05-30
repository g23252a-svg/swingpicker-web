# -*- coding: utf-8 -*-
"""[v22.3.21] 초록 매수 CTA 게이트 테스트.
공식 신규매수 = TOP_PICK==1 AND BUY_NOW_ELIGIBLE==1. 그 외엔 초록 CTA 금지."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.buy_now_badge import is_official_new_buy


def test_requires_top_pick_and_eligible():
    assert is_official_new_buy({"TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1}) is True
    assert is_official_new_buy({"TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 1}) is False
    assert is_official_new_buy({"TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 0}) is False
    assert is_official_new_buy({"TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0}) is False


def test_handles_string_and_float_flags():
    assert is_official_new_buy({"TOP_PICK": "1", "BUY_NOW_ELIGIBLE": "1"}) is True
    assert is_official_new_buy({"TOP_PICK": 1.0, "BUY_NOW_ELIGIBLE": 1.0}) is True
    assert is_official_new_buy({"TOP_PICK": "0", "BUY_NOW_ELIGIBLE": "1"}) is False


def test_missing_columns_are_false():
    assert is_official_new_buy({}) is False
