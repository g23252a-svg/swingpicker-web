# -*- coding: utf-8 -*-
"""[v22.3.21] 엔진 상태 카드 톤 — 매수금지 구간 판정 테스트.

매크로 WARNING/CRITICAL 또는 ROUTE 차단이면 '신규매수 보류 구간'으로 표시되어야 한다.
표시 게이트인 _is_market_no_buy_mode 의 판정을 고정한다(엔진/추천 로직 무변경).
NiceGUI 미설치 sandbox에서는 import 단계에서 skip(운영 환경 전용).
"""
import sys, os
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from components.tab_market import _is_market_no_buy_mode
except Exception:  # nicegui 등 미설치
    pytest.skip("NiceGUI 미설치 — 운영 환경 전용", allow_module_level=True)


def test_critical_macro_is_no_buy():
    assert _is_market_no_buy_mode("CRITICAL", "ATTACK") is True
    assert _is_market_no_buy_mode("WARNING", "ARMED") is True


def test_normal_macro_is_not_no_buy():
    assert _is_market_no_buy_mode("NORMAL", "ATTACK") is False
    assert _is_market_no_buy_mode("NORMAL", "ARMED") is False


def test_caution_macro_alone_is_not_no_buy():
    assert _is_market_no_buy_mode("CAUTION", "ARMED") is False


def test_route_block_triggers_no_buy_even_if_macro_ok():
    assert _is_market_no_buy_mode("NORMAL", "BLOCKED") is True


def test_handles_none_and_case():
    assert _is_market_no_buy_mode(None, None) is False
    assert _is_market_no_buy_mode("critical", "attack") is True
