# -*- coding: utf-8 -*-
"""
test_route_contract_v22.py — Route enum 계약 + route_name 헬퍼 검증

[v22] scoring_engine.TOP_PICK_ROUTES = frozenset({"ATTACK", "ARMED"}) 는 문자열 기준.
다른 파일은 Route.ATTACK / Route.ARMED 사용. Route(str, Enum) 덕분에 == 가 작동하지만,
미래 enum 변경(예: IntEnum 전환) 위험 방어 계약 테스트.
"""
import pytest


def test_route_enum_str_compat():
    """Route enum이 str과 직접 비교 가능해야 — TOP_PICK_ROUTES의 문자열 비교 전제"""
    from collector_config import Route
    # isinstance(Route.ATTACK, str) 덕분에 == "ATTACK" True
    assert Route.ATTACK == "ATTACK", "Route.ATTACK이 'ATTACK' 문자열과 비교 불가 — v22 TOP_PICK_ROUTES 깨짐"
    assert Route.ARMED == "ARMED", "Route.ARMED가 'ARMED' 문자열과 비교 불가"
    assert Route.WAIT == "WAIT"
    assert Route.NEUTRAL == "NEUTRAL"


def test_route_in_frozenset():
    """Route.ATTACK이 {'ATTACK', 'ARMED'} set membership에서 True여야"""
    from collector_config import Route
    from scoring_engine import TOP_PICK_ROUTES
    assert Route.ATTACK in TOP_PICK_ROUTES, "Route.ATTACK이 TOP_PICK_ROUTES에 없음 (hash 불일치)"
    assert Route.ARMED in TOP_PICK_ROUTES
    assert Route.WAIT not in TOP_PICK_ROUTES, "WAIT이 positive gate에 포함됨 — 누출 가능성"
    assert Route.NEUTRAL not in TOP_PICK_ROUTES


def test_route_name_helper_enum():
    """route_name(Route.XXX) → 'XXX' 정확히 반환"""
    from collector_config import Route
    from shared_utils import route_name
    assert route_name(Route.ATTACK) == "ATTACK"
    assert route_name(Route.ARMED) == "ARMED"
    assert route_name(Route.WAIT) == "WAIT"


def test_route_name_helper_str():
    """route_name('ATTACK') → 'ATTACK' (문자열 passthrough)"""
    from shared_utils import route_name
    assert route_name("ATTACK") == "ATTACK"
    assert route_name("armed") == "ARMED", "소문자도 정규화되어야"
    assert route_name(" ATTACK ") == "ATTACK", "공백 제거"


def test_route_name_helper_edge():
    """route_name edge cases — None / 빈 문자열 / 숫자"""
    from shared_utils import route_name
    assert route_name(None) == ""
    assert route_name("") == ""
    assert route_name(0) == "0"   # 안전 변환


def test_pd_series_isin_route_enum():
    """pd.Series([Route.ATTACK, ...]).isin(['ATTACK']) 동작 — pipeline_finalize의 핵심 패턴"""
    import pandas as pd
    from collector_config import Route
    s = pd.Series([Route.ATTACK, Route.ARMED, Route.WAIT, "ATTACK", "WAIT"])
    mask = s.isin(["ATTACK", "ARMED"])
    # Route.ATTACK(0), Route.ARMED(1), "ATTACK"(3) → 3개 True
    assert mask.sum() == 3, f"expected 3 matches, got {mask.sum()}: {mask.tolist()}"


if __name__ == "__main__":
    # standalone 실행도 지원
    import sys
    try:
        test_route_enum_str_compat()
        test_route_in_frozenset()
        test_route_name_helper_enum()
        test_route_name_helper_str()
        test_route_name_helper_edge()
        test_pd_series_isin_route_enum()
        print("✅ 모든 Route 계약 테스트 통과")
    except AssertionError as e:
        print(f"❌ FAIL: {e}")
        sys.exit(1)
