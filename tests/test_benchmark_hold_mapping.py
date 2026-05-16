"""
tests/test_benchmark_hold_mapping.py
====================================
[v3.9.15e + 3] services/benchmarks.map_slider_to_bench_key 매핑 정합성.

검증 대상:
- 슬라이더 hold_days (1~60) → bench key (1·5·10·20·60) 매핑
- tab_backtest._RET_MAP의 임계값과 동일 (정합성 보장)
- 41~60일 구간이 가짜 20일로 떨어지지 않음 (거짓 알파 방지)
"""
import pytest


# ────────────────────────────────────────────────────────────────
# A. 매핑 테이블 12케이스 — 임계값 경계 포함
# ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "hold_days, expected_key",
    [
        # 1~3일 구간 → KOSPI 1일
        (1, 1),
        (2, 1),
        (3, 1),
        # 4~7일 구간 → KOSPI 5일
        (4, 5),
        (5, 5),
        (7, 5),
        # 8~15일 구간 → KOSPI 10일
        (8, 10),
        (10, 10),
        (15, 10),
        # 16~40일 구간 → KOSPI 20일
        (16, 20),
        (20, 20),
        (40, 20),
        # 41~60일 구간 → KOSPI 60일 (가짜 20일 매핑 금지)
        (41, 60),
        (50, 60),
        (60, 60),
    ],
)
def test_slider_to_bench_key_mapping(hold_days, expected_key):
    """슬라이더 값이 정확한 bench key로 매핑되어야 한다.

    실패 시 거짓 알파 위험:
    - 50일 보유 종목을 20일 KOSPI와 비교하면 systematic하게 왜곡됨
    - simple alpha 없음 < 거짓 alpha
    """
    from services.benchmarks import map_slider_to_bench_key
    assert map_slider_to_bench_key(hold_days) == expected_key, (
        f"hold_days={hold_days} → {map_slider_to_bench_key(hold_days)} "
        f"(기대값: {expected_key})"
    )


# ────────────────────────────────────────────────────────────────
# B. 41~60일 정직성 — bench_cache에 60 키 없을 때 None 반환
# ────────────────────────────────────────────────────────────────
def test_41_60_returns_none_when_cache_missing_60_key():
    """bench_cache에 60 키가 없으면 41~60일 보유는 None 반환해야 한다.

    절대 다른 키 (예: 20)로 폴백 매핑하면 안 됨.
    """
    from services.benchmarks import get_kospi_return

    # 60 키 없는 cache
    bench = {"KOSPI": {1: -0.5, 3: 1.0, 5: 2.5, 10: 5.0, 20: 8.0}}

    # 1~40일은 값 반환
    assert get_kospi_return(bench, 5) == 2.5
    assert get_kospi_return(bench, 20) == 8.0
    assert get_kospi_return(bench, 40) == 8.0  # 16~40일 → 20 키

    # 41~60일은 None (60 키 없으므로)
    assert get_kospi_return(bench, 41) is None, "41일 보유는 None이어야 함 (가짜 20일 매핑 금지)"
    assert get_kospi_return(bench, 50) is None
    assert get_kospi_return(bench, 60) is None


def test_60_key_present_returns_value():
    """bench_cache에 60 키가 있으면 41~60일 보유가 정상 값 반환."""
    from services.benchmarks import get_kospi_return

    bench = {"KOSPI": {1: -0.5, 3: 1.0, 5: 2.5, 10: 5.0, 20: 8.0, 60: 15.0}}

    assert get_kospi_return(bench, 41) == 15.0
    assert get_kospi_return(bench, 50) == 15.0
    assert get_kospi_return(bench, 60) == 15.0


# ────────────────────────────────────────────────────────────────
# C. 빈 cache / KOSPI 키 없음 → None
# ────────────────────────────────────────────────────────────────
def test_empty_bench_returns_none():
    """빈 cache는 모든 hold_days에 대해 None 반환."""
    from services.benchmarks import get_kospi_return
    assert get_kospi_return({}, 5) is None
    assert get_kospi_return({"KOSDAQ": {1: 1.0}}, 5) is None  # KOSPI 없음


# ────────────────────────────────────────────────────────────────
# D. 정합성 — _BENCH_HOLD_MAP과 tab_backtest._RET_MAP 임계값 일치
# ────────────────────────────────────────────────────────────────
def test_hold_map_thresholds_match_ret_map():
    """매핑 임계값이 _RET_MAP과 동일해야 함 (휴먼 에러 회귀 방지)."""
    from services.benchmarks import _BENCH_HOLD_MAP

    # _BENCH_HOLD_MAP: [(3, 1), (7, 5), (15, 10), (40, 20), (999, 60)]
    expected_thresholds = [3, 7, 15, 40, 999]
    actual_thresholds = [th for th, _ in _BENCH_HOLD_MAP]
    assert actual_thresholds == expected_thresholds, (
        f"_BENCH_HOLD_MAP 임계값이 tab_backtest._RET_MAP과 다름: "
        f"actual={actual_thresholds}, expected={expected_thresholds}"
    )
