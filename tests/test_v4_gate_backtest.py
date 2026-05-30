# -*- coding: utf-8 -*-
"""v4 공식픽 게이트 백테스트 회귀 테스트.

pytest tests/test_v4_gate_backtest.py -v

대상: combo_gate_v4 (순수 함수)
리뷰 필수: ① baseline 재현율  ② MDD 일별 포트폴리오 기준
          ③ baseline_sel reindex 안전  + carry-through 격리 / 검증케이스 코드
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from combo_gate_v4 import (
    _simulate_official_pick, _evaluate_gate_combo, baseline_reproduction_rate,
    _v4_gate_combos, VALIDATION_CASES, TP1_BANDS, EXPLORATORY_BANDS,
)


def _rows(specs):
    """specs: list of dict(부분) → 표준 백테스트 행."""
    base = dict(code="000000", ret=1.0, win=1, trade_date="20260501",
                ELITE=80, BALANCE=80, TP1_PCT=10.0, RR=1.5, turnover=100.0,
                entry_gap=1.0, MATURE=True, EST_WR=0.40, TOP_PICK=0,
                TOP_PICK_TYPE="", DISPLAY_SCORE=80.0)
    return pd.DataFrame([{**base, **s} for s in specs])


def test_carry_through_aggressive_kept_regardless_of_knobs():
    """AGGRESSIVE 저장픽은 use_v4/밴드와 무관하게 항상 유지(carry)."""
    df = _rows([dict(TOP_PICK=1, TOP_PICK_TYPE="AGGRESSIVE", TP1_PCT=22.0, EST_WR=0.30)])
    for uv in (False, True):
        for band in TP1_BANDS:
            sel = _simulate_official_pick(df, uv, band, None)
            assert sel.iloc[0] == True, f"AGGRESSIVE carry 실패 uv={uv} band={band}"


def test_stable_rederived_by_tp1_band():
    """STABLE-type 픽은 TP1 밴드에 따라 재유도된다 (carry 아님)."""
    # TP1=16: baseline (7,15) 제외, (5,20) 포함
    df = _rows([dict(TOP_PICK=1, TOP_PICK_TYPE="STABLE", TP1_PCT=16.0, EST_WR=0.60)])
    assert _simulate_official_pick(df, False, (7, 15), None).iloc[0] == False
    assert _simulate_official_pick(df, False, (5, 20), None).iloc[0] == True


def test_baseline_reproduction_high_on_synthetic():
    """재현율: 저장 TOP_PICK vs sim(False,(7,15)) — carry-through로 ~100%."""
    df = _rows([
        dict(TOP_PICK=1, TOP_PICK_TYPE="AGGRESSIVE"),                       # carry
        dict(TOP_PICK=1, TOP_PICK_TYPE=""),                                 # legacy carry
        dict(TOP_PICK=1, TOP_PICK_TYPE="STABLE", TP1_PCT=10.0, EST_WR=0.60),# STABLE 재현
        dict(TOP_PICK=0, TP1_PCT=30.0),                                     # 비픽
    ])
    r = baseline_reproduction_rate(df, None)
    assert r["agreement_pct"] >= 95.0
    assert r["false_neg"] == 0


def test_mdd_uses_daily_portfolio_basis():
    """리뷰 ②: 같은 날 여러 종목 → MDD는 일별 평균수익 곡선 기준."""
    # day1: +10,+10 (평균 +10) / day2: -30,-30 (평균 -30) → 일별곡선 10 → -20, MDD=30
    df = _rows([
        dict(TOP_PICK=1, TOP_PICK_TYPE="AGGRESSIVE", trade_date="20260501", ret=10),
        dict(TOP_PICK=1, TOP_PICK_TYPE="AGGRESSIVE", trade_date="20260501", ret=10),
        dict(TOP_PICK=1, TOP_PICK_TYPE="AGGRESSIVE", trade_date="20260502", ret=-30),
        dict(TOP_PICK=1, TOP_PICK_TYPE="AGGRESSIVE", trade_date="20260502", ret=-30),
    ])
    base = df["TOP_PICK"] == 1
    stat = _evaluate_gate_combo(df, False, (7, 15), None, base)
    assert abs(stat["mdd"] - 30.0) < 1e-6, f"일별 MDD 기대 30, 실제 {stat['mdd']}"


def test_baseline_sel_reindex_safety():
    """리뷰 ③: baseline_sel index가 df와 달라도(부분집합) 크래시 없이 reindex."""
    df = _rows([dict(TOP_PICK=1, TOP_PICK_TYPE="AGGRESSIVE", ret=5) for _ in range(4)])
    sub = df.iloc[2:].copy()                         # OOS 부분집합 흉내
    full_baseline = df["TOP_PICK"] == 1              # 전체 index 기준 Series
    stat = _evaluate_gate_combo(sub, False, (7, 15), None, full_baseline)  # index mismatch
    assert stat["n"] == 2                            # 크래시 없이 평가


def test_delta_metrics_present():
    df = _rows([
        dict(TOP_PICK=1, TOP_PICK_TYPE="AGGRESSIVE", ret=5),
        dict(TOP_PICK=0, TOP_PICK_TYPE="STABLE", TP1_PCT=16.0, EST_WR=0.60, ret=-3),
    ])
    base = df["TOP_PICK"] == 1
    stat = _evaluate_gate_combo(df, False, (5, 20), None, base)  # STABLE 1개 추가
    for k in ["delta_pnl_vs_baseline", "delta_avg_ret_vs_baseline",
              "delta_daily_pnl_vs_baseline", "avoided_loss", "missed_gain",
              "n_added", "n_dropped", "win_rate_excess", "mdd"]:
        assert k in stat
    assert stat["n_added"] == 1                       # (5,20)에서 STABLE 추가


def test_validation_cases_use_codes_not_names():
    """리뷰 권장: 검증케이스는 종목코드 고정 (이름 변경/공백 이슈 회피)."""
    codes = {c["code"] for c in VALIDATION_CASES}
    assert "035510" in codes          # 신세계I&C (실 CSV 확인)
    assert "0008Z0" in codes          # 에스엔시스 (실 CSV 확인)
    for c in VALIDATION_CASES:
        assert c["code"] and c["date"]


def test_exploratory_band_flagged():
    """(3,30)은 exploratory 표시."""
    assert (3, 30) in EXPLORATORY_BANDS
    df = _rows([dict(TOP_PICK=1, TOP_PICK_TYPE="AGGRESSIVE")])
    stat = _evaluate_gate_combo(df, False, (3, 30), None, df["TOP_PICK"] == 1)
    assert stat["exploratory"] is True
    stat2 = _evaluate_gate_combo(df, False, (6, 18), None, df["TOP_PICK"] == 1)
    assert stat2["exploratory"] is False


def test_grid_includes_recommended_band():
    """리뷰 권장: (6,18) 격자 포함."""
    bands = {tuple(b) for _, b in _v4_gate_combos()}
    assert (6, 18) in bands
    assert (7, 15) in bands  # baseline


def test_real_data_smoke_if_present():
    """실 recommend_2026*.csv + price_snapshot 있으면 백테스트 smoke. 없으면 skip."""
    import glob
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not glob.glob(os.path.join(data, "recommend_2026*.csv")):
        import pytest
        pytest.skip("recommend CSV 없음 — 운영 환경 전용")
    from combo_gate_v4 import run_v4_gate_backtest
    res = run_v4_gate_backtest(data, horizon=5, save=False)
    assert res["baseline_reproduction"]["agreement_pct"] >= 95.0
    assert len(res["full_sample"]["combos"]) == len(_v4_gate_combos())
