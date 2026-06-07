# -*- coding: utf-8 -*-
"""v23.0 통합 GUARD 엔진 회귀 테스트.

    pytest tests/test_guard_system_v23.py -v

대상: guard_system.apply_guard_system (순수 함수) + 개별 GUARD 규칙.
- G1 유동성-손절 차단 / G2 RR 열화 / G3 보유경과 감점 / G4 저모멘텀 게이트
- G5 추세붕괴 경보 / G6 시장역행 / G7 윗꼬리 약세 / G8 사전경고
- ELITE_LABEL 게이트, GUARD_KELLY_MULT, TOP_PICK 재게이트, ON/OFF 모드
- 검증 케이스: 에스엔시스(5/7 진입, 4일차) · 신세계I&C(5/1 진입, 10일차)
"""
import numpy as np
import pandas as pd
import pytest

from guard_system import (
    apply_guard_system,
    guard_summary,
    GUARD_CONTRACT_COLS,
    GUARD_PASS_COLS,
)
from collector_config import DEFAULT_CONFIG, GuardConfig, CollectorConfig


def _row(**kw):
    base = {
        "종목명": "테스트", "종목코드": "000000",
        "ELITE_SCORE": 80.0, "TIMING_SCORE": 60.0, "AXIS_MEAN": 70.0,
        "거래대금(억원)": 500.0, "STOP_PCT": 8.0,
    }
    base.update(kw)
    return base


def _one(config=None, kospi_ret_1d=None, **kw):
    df = pd.DataFrame([_row(**kw)])
    return apply_guard_system(df, config=config, kospi_ret_1d=kospi_ret_1d).iloc[0]


# ── 계약/스키마 ──────────────────────────────────────────────────
def test_contract_columns_present():
    out = apply_guard_system(pd.DataFrame([_row(TOP_PICK=1)]))
    for c in GUARD_CONTRACT_COLS:
        assert c in out.columns, f"계약 컬럼 누락: {c}"


def test_empty_df_safe():
    empty = pd.DataFrame()
    assert apply_guard_system(empty) is empty  # 그대로 반환, 예외 없음


def test_missing_columns_robust():
    # 점수 컬럼 전부 없는 구버전 행 — 예외 없이 PASS 기본
    df = pd.DataFrame([{"종목명": "구버전", "종가": 1000}])
    out = apply_guard_system(df)
    assert bool(out.iloc[0]["GUARD_BLOCK"]) is False
    assert out.iloc[0]["GUARD_RR_MULT"] == 1.0


# ── G1 유동성-손절 차단 ──────────────────────────────────────────
def test_guard1_block_low_liquidity_tight_stop():
    r = _one(**{"거래대금(억원)": 80}, STOP_PCT=5)
    assert bool(r["GUARD_PASS_1"]) is False
    assert bool(r["GUARD_BLOCK"]) is True
    assert r["GUARDED_ELITE_SCORE"] == 0.0
    assert r["GUARD_KELLY_MULT"] == 0.0


def test_guard1_pass_if_either_ok():
    # 거래대금 충분 → 통과
    assert bool(_one(**{"거래대금(억원)": 500}, STOP_PCT=5)["GUARD_PASS_1"]) is True
    # 손절 여유 충분 → 통과
    assert bool(_one(**{"거래대금(억원)": 80}, STOP_PCT=10)["GUARD_PASS_1"]) is True


# ── G2 RR 열화 ───────────────────────────────────────────────────
def test_guard2_timing_zero():
    r = _one(TIMING_SCORE=0, AXIS_MEAN=70, ELITE_SCORE=80)
    assert r["GUARD_RR_MULT"] == pytest.approx(0.3)
    assert bool(r["GUARD_PASS_2"]) is False
    assert r["GUARDED_ELITE_SCORE"] == pytest.approx(24.0)  # 80 * 0.3


def test_guard2_axis_low():
    r = _one(TIMING_SCORE=50, AXIS_MEAN=35, ELITE_SCORE=80)
    assert r["GUARD_RR_MULT"] == pytest.approx(0.5)
    assert r["GUARDED_ELITE_SCORE"] == pytest.approx(40.0)


def test_guard2_worst_multiplier_wins():
    # TIMING=0(×0.3) AND AXIS<40(×0.5) → 더 작은 0.3 채택
    r = _one(TIMING_SCORE=0, AXIS_MEAN=30, ELITE_SCORE=100)
    assert r["GUARD_RR_MULT"] == pytest.approx(0.3)


# ── G3 보유경과 누적 감점 ────────────────────────────────────────
@pytest.mark.parametrize("age,expected", [
    (3, 0.0), (5, 15.0), (6, 15.0), (7, 25.0), (9, 25.0), (10, 45.0), (15, 45.0),
])
def test_guard3_carry_stale_curve(age, expected):
    r = _one(CARRY_AGE_DAYS=age)
    assert r["GUARD_PENALTY_3"] == pytest.approx(expected)


def test_guard3_no_column_zero():
    r = _one()  # CARRY_AGE_DAYS 없음
    assert r["GUARD_PENALTY_3"] == 0.0
    assert bool(r["GUARD_PASS_3"]) is True


# ── G4 저모멘텀 섹터 게이트 ──────────────────────────────────────
def test_guard4_holding_company_low_timing_blocked():
    r = _one(종목명="OO지주", 업종_대분류="금융", TIMING_SCORE=20)
    assert bool(r["GUARD_PASS_4"]) is False
    assert bool(r["GUARD_BLOCK"]) is True


def test_guard4_holding_company_high_timing_pass():
    # 저모멘텀 섹터라도 TIMING≥30이면 통과
    r = _one(종목명="OO지주", 업종_대분류="금융", TIMING_SCORE=45)
    assert bool(r["GUARD_PASS_4"]) is True


def test_guard4_normal_sector_not_gated():
    r = _one(종목명="반도체주", 업종_대분류="반도체", TIMING_SCORE=10)
    assert bool(r["GUARD_PASS_4"]) is True  # 저모멘텀 키워드 아님 → 게이트 미적용


# ── G5 추세선 붕괴 경보 ──────────────────────────────────────────
def test_guard5_three_breaks_alert():
    r = _one(SUPERTREND_DIR=-1, Above_MA20=0, IS_ABOVE_POC=0,
             HMA_Trend="▲", MACD_Slope_PCT=1)
    assert r["GUARD_TRENDLINE_BROKEN"] == 3
    assert bool(r["GUARD_FORCE_EXIT_ALERT"]) is True
    assert r["GUARD_PENALTY_5"] == pytest.approx(20.0)


def test_guard5_two_breaks_no_alert():
    r = _one(SUPERTREND_DIR=-1, Above_MA20=0, IS_ABOVE_POC=1,
             HMA_Trend="▲", MACD_Slope_PCT=1)
    assert r["GUARD_TRENDLINE_BROKEN"] == 2
    assert bool(r["GUARD_FORCE_EXIT_ALERT"]) is False
    assert r["GUARD_PENALTY_5"] == 0.0


def test_guard5_all_five_breaks():
    r = _one(SUPERTREND_DIR=-1, Above_MA20=0, IS_ABOVE_POC=0,
             HMA_Trend="▼", MACD_Slope_PCT=-2)
    assert r["GUARD_TRENDLINE_BROKEN"] == 5
    assert bool(r["GUARD_FORCE_EXIT_ALERT"]) is True


# ── G6 시장 역행 ─────────────────────────────────────────────────
def test_guard6_divergence_penalty():
    r = _one(kospi_ret_1d=2.5, **{"ret_1d_%": -6.0})
    assert bool(r["GUARD_PASS_6"]) is False
    assert r["GUARD_PENALTY_6"] == pytest.approx(25.0)


def test_guard6_no_kospi_skip():
    # KOSPI 미제공 & 컬럼 없음 → 발동 안 함
    r = _one(kospi_ret_1d=None, **{"ret_1d_%": -10.0})
    assert bool(r["GUARD_PASS_6"]) is True
    assert r["GUARD_PENALTY_6"] == 0.0


def test_guard6_market_down_no_penalty():
    # 장도 빠지면 역행 아님
    r = _one(kospi_ret_1d=-1.0, **{"ret_1d_%": -6.0})
    assert bool(r["GUARD_PASS_6"]) is True


# ── G7 윗꼬리 약세 ───────────────────────────────────────────────
def test_guard7_weak_upper_shadow_penalty():
    r = _one(Upper_Shadow_Ratio=0.7, 거래강도=0.5)
    assert bool(r["GUARD_PASS_7"]) is False
    assert r["GUARD_PENALTY_7"] == pytest.approx(15.0)


def test_guard7_strong_volume_no_penalty():
    # 윗꼬리 길어도 거래강도 충분하면 통과
    r = _one(Upper_Shadow_Ratio=0.7, 거래강도=1.5)
    assert bool(r["GUARD_PASS_7"]) is True


# ── G8 사전경고 ──────────────────────────────────────────────────
def test_guard8_prewarning_day4_flag_only():
    r = _one(CARRY_AGE_DAYS=4)
    assert bool(r["GUARD_PRE_WARNING"]) is True
    # 감점·차단 없음
    assert r["GUARD_PENALTY_TOTAL"] == 0.0
    assert bool(r["GUARD_BLOCK"]) is False
    # G8은 ELITE 판정(핵심 가드)에서 제외 — 사전경고만으로 ALL_PASS 유지
    assert bool(r["GUARD_ALL_PASS"]) is True


# ── ELITE_LABEL 게이트 / TOP_PICK 재게이트 ───────────────────────
def test_elite_label_clean_top_pick():
    r = _one(TOP_PICK=1, ELITE_SCORE=88, TIMING_SCORE=75, AXIS_MEAN=80,
             SUPERTREND_DIR=1, Above_MA20=1)
    assert r["ELITE_LABEL"] == "ELITE"
    assert int(r["TOP_PICK"]) == 1
    assert int(r["TOP_PICK_RAW"]) == 1


def test_elite_label_blocked_top_pick_demoted():
    # 보유 10일차(-45) → GUARDED 45 < 60 → ELITE 박탈
    r = _one(TOP_PICK=1, ELITE_SCORE=90, CARRY_AGE_DAYS=10)
    assert r["ELITE_LABEL"] == "GUARD_BLOCKED"
    assert int(r["TOP_PICK"]) == 0          # enforce 모드 재게이트
    assert int(r["TOP_PICK_RAW"]) == 1      # 원본 보존


def test_off_mode_does_not_touch_top_pick():
    off = CollectorConfig(guard=GuardConfig(guard_enforce_top_pick=False))
    r = _one(config=off, TOP_PICK=1, ELITE_SCORE=90, CARRY_AGE_DAYS=10)
    # shadow 컬럼은 부여하되 TOP_PICK 원본 유지, TOP_PICK_RAW 미생성
    assert int(r["TOP_PICK"]) == 1
    assert "TOP_PICK_RAW" not in r.index
    # ELITE_LABEL은 여전히 계산됨 (관찰용)
    assert r["ELITE_LABEL"] == "GUARD_BLOCKED"


# ── 검증 케이스 (memory 명시) ────────────────────────────────────
def test_case_snsys_day4_prewarning():
    """에스엔시스: 5/7 진입, 4일차 → G8 사전경고만, 감점/차단 없음."""
    r = _one(종목명="에스엔시스", CARRY_AGE_DAYS=4, CARRY_FROM_DATE="20260507",
             ELITE_SCORE=82, TIMING_SCORE=65, AXIS_MEAN=72, TOP_PICK=1)
    assert bool(r["GUARD_PRE_WARNING"]) is True
    assert r["GUARD_PENALTY_3"] == 0.0      # 4일차는 아직 G3 감점 전(5일차부터)
    assert "G8사전경고" in r["GUARD_REASON"]
    # 사전경고 단독으로는 ELITE 유지 (다른 가드 통과 시)
    assert r["ELITE_LABEL"] == "ELITE"


def test_case_shinsegae_ic_day10_heavy_penalty():
    """신세계I&C: 5/1 진입, 10일차 손실 방치 → G3 -45 + 추세붕괴 시 강제청산경보."""
    r = _one(종목명="신세계I&C", CARRY_AGE_DAYS=10, CARRY_FROM_DATE="20260501",
             CARRY_RET_PCT=-9.0, ELITE_SCORE=70, TIMING_SCORE=40, AXIS_MEAN=55,
             SUPERTREND_DIR=-1, Above_MA20=0, IS_ABOVE_POC=0,
             HMA_Trend="▼", MACD_Slope_PCT=-1, TOP_PICK=1)
    assert r["GUARD_PENALTY_3"] == pytest.approx(45.0)
    assert bool(r["GUARD_FORCE_EXIT_ALERT"]) is True       # 추세 5축 붕괴
    assert r["GUARD_PENALTY_TOTAL"] >= 45.0
    # 점수 대폭 하향 → ELITE 박탈
    assert r["ELITE_LABEL"] == "GUARD_BLOCKED"
    assert int(r["TOP_PICK"]) == 0


# ── 집계/요약 ────────────────────────────────────────────────────
def test_guard_summary_counts():
    df = pd.DataFrame([
        _row(종목명="정상", TOP_PICK=1, ELITE_SCORE=88, TIMING_SCORE=75,
             AXIS_MEAN=80, SUPERTREND_DIR=1, Above_MA20=1),
        _row(종목명="차단", **{"거래대금(억원)": 50}, STOP_PCT=4),
        _row(종목명="경보", SUPERTREND_DIR=-1, Above_MA20=0, IS_ABOVE_POC=0,
             HMA_Trend="▼", MACD_Slope_PCT=-1),
    ])
    out = apply_guard_system(df)
    s = guard_summary(out)
    assert s["n_rows"] == 3
    assert s["n_block"] == 1
    assert s["n_force_exit"] == 1
    assert s["n_elite"] == 1
