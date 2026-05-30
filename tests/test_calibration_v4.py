# -*- coding: utf-8 -*-
"""v4.0 Phase 1 세그먼트 캘리브레이션 회귀 테스트.

pytest tests/test_calibration_v4.py -v

대상: calibration_v4 (순수 함수)
- 천장 제거: 표본 충분 + 고승률 셀은 0.55를 넘을 수 있어야 한다 (STABLE 부활)
- 경험적 베이즈: 표본 적은 셀은 글로벌 prior로 수축
- 상대 게이트: 절대 0.55 미사용, 당일 분위 + prior 마진
- SHADOW 안전: 본선 TOP_PICK / EST_WIN_RATE 컬럼 무변경
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from calibration_v4 import (
    build_segmented_table,
    score_segment,
    relative_stable_gate,
    add_v4_shadow_columns,
)


def _trades(n_high=80, n_low=4):
    """고승률 세그먼트(표본 충분) + 저표본 세그먼트 합성 체결 로그."""
    rng = np.random.default_rng(42)
    rows = []
    # 세그먼트 A: ELITE 80-90 / NOW_BUY / NORMAL — 표본 충분, 실제 승률 ~0.75
    for _ in range(n_high):
        rows.append(dict(ELITE_SCORE=85, ACTION_TIER="NOW_BUY", MACRO_REGIME_MODE="NORMAL",
                         is_win=int(rng.random() < 0.75), rec_date="20260520"))
    # 세그먼트 B: ELITE 80-90 / ACCUMULATION_READY / NORMAL — 표본 희박
    for _ in range(n_low):
        rows.append(dict(ELITE_SCORE=85, ACTION_TIER="ACCUMULATION_READY", MACRO_REGIME_MODE="NORMAL",
                         is_win=1, rec_date="20260520"))
    # 글로벌 prior 낮추는 잡음 표본
    for _ in range(120):
        rows.append(dict(ELITE_SCORE=55, ACTION_TIER="PASS", MACRO_REGIME_MODE="NORMAL",
                         is_win=int(rng.random() < 0.40), rec_date="20260520"))
    return pd.DataFrame(rows)


def test_table_builds_and_has_global_prior():
    t = build_segmented_table(_trades())
    assert t["meta"]["method"] == "SEGMENTED_EB"
    assert 0.3 < t["meta"]["global_prior"] < 0.7
    assert t["meta"]["n_segments"] >= 2


def test_ceiling_removed_high_segment_exceeds_055():
    """핵심: 표본 충분 + 고승률 셀은 0.55를 넘는다 (기존 단일축 천장 ~0.51 제거)."""
    t = build_segmented_table(_trades(n_high=120), asof_ymd="20260529")
    row = pd.Series(dict(ELITE_SCORE=85, ACTION_TIER="NOW_BUY", MACRO_REGIME_MODE="NORMAL"))
    p, n, key, suff = score_segment(row, t)
    assert suff is True
    assert p > 0.55, f"고승률 셀이 0.55를 못 넘음: {p}"


def test_eb_shrinks_sparse_segment_toward_prior():
    """표본 4개짜리 셀은 prior 쪽으로 수축되어 1.0이 되면 안 된다."""
    t = build_segmented_table(_trades(n_low=4))
    p0 = t["meta"]["global_prior"]
    row = pd.Series(dict(ELITE_SCORE=85, ACTION_TIER="ACCUMULATION_READY", MACRO_REGIME_MODE="NORMAL"))
    p, n, key, suff = score_segment(row, t)
    assert p < 1.0
    assert abs(p - p0) < abs(1.0 - p0), "수축이 prior 방향으로 작동하지 않음"
    assert suff is False


def test_missing_segment_falls_back_to_prior():
    """존재하지 않는 세그먼트 → prior 폴백, 예외 없음."""
    t = build_segmented_table(_trades())
    row = pd.Series(dict(ELITE_SCORE=85, ACTION_TIER="UNKNOWN_TIER", MACRO_REGIME_MODE="MARS"))
    p, n, key, suff = score_segment(row, t)
    assert n == 0.0 and suff is False
    assert abs(p - t["meta"]["global_prior"]) < 1e-6


def test_relative_gate_uses_quantile_not_absolute():
    p_win = pd.Series([0.40, 0.52, 0.58, 0.63, 0.70])
    n_eff = pd.Series([50, 50, 50, 50, 50])
    gate = relative_stable_gate(p_win, n_eff, prior=0.50)
    # 절대 0.55였다면 0.52는 탈락이지만, 분위 게이트는 상위만 통과
    assert gate.iloc[0] == False  # 0.40
    assert gate.iloc[-1] == True  # 0.70 상위
    assert gate.sum() >= 1


def test_low_sample_pool_disables_quantile_gate():
    """게이트 풀 표본<4면 분위 비활성, prior+margin만 적용."""
    p_win = pd.Series([0.60, 0.30])
    n_eff = pd.Series([50, 5])  # 두 번째는 min_n 미달
    gate = relative_stable_gate(p_win, n_eff, prior=0.50, min_n=20)
    assert gate.iloc[0] == True   # 0.60 >= 0.53
    assert gate.iloc[1] == False  # 표본 미달


def test_shadow_columns_do_not_touch_baseline():
    """SHADOW 안전: 기존 TOP_PICK / EST_WIN_RATE 값이 변하지 않는다."""
    t = build_segmented_table(_trades())
    df = pd.DataFrame([
        dict(종목명="가", ELITE_SCORE=85, TP1_PCT=10.0, BALANCE_SCORE=80,
             CALIBRATION_MODE="MATURE", ACTION_TIER="NOW_BUY", MACRO_REGIME_MODE="NORMAL",
             TOP_PICK=0, EST_WIN_RATE=0.417),
        dict(종목명="나", ELITE_SCORE=55, TP1_PCT=20.0, BALANCE_SCORE=40,
             CALIBRATION_MODE="MATURE", ACTION_TIER="PASS", MACRO_REGIME_MODE="NORMAL",
             TOP_PICK=0, EST_WIN_RATE=0.417),
    ])
    before_tp = df["TOP_PICK"].tolist()
    before_wr = df["EST_WIN_RATE"].tolist()
    out = add_v4_shadow_columns(df, t)
    assert out["TOP_PICK"].tolist() == before_tp
    assert out["EST_WIN_RATE"].tolist() == before_wr
    for c in ["EST_WIN_RATE_V4", "STABLE_GATE_V4_PASS", "TOP_PICK_V4_SHADOW"]:
        assert c in out.columns


def test_shadow_revives_stable_candidate():
    """천장 제거 효과: 구조는 STABLE이지만 단일축 WR로는 죽던 종목이 V4에서 부활."""
    t = build_segmented_table(_trades(n_high=120), asof_ymd="20260529")
    df = pd.DataFrame([dict(
        종목명="부활", ELITE_SCORE=85, TP1_PCT=10.0, BALANCE_SCORE=80,
        CALIBRATION_MODE="MATURE", ACTION_TIER="NOW_BUY", MACRO_REGIME_MODE="NORMAL",
        TOP_PICK=0, EST_WIN_RATE=0.495,
    )])
    out = add_v4_shadow_columns(df, t)
    assert out["TOP_PICK"].iloc[0] == 0            # 본선은 여전히 탈락
    assert out["TOP_PICK_V4_SHADOW"].iloc[0] == 1  # V4 shadow에서는 부활


# ── excess(벤치 대비) 기준 회귀 ──────────────────────────────────────────────
def _ret_trades():
    """ret_pct 기반 합성 로그: 상승장(모두 양수)이지만 분리력 있는 두 세그먼트."""
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(120):  # A: 시장 평균 위 (좋은 엣지)
        rows.append(dict(ELITE_SCORE=85, ACTION_TIER="NOW_BUY", MACRO_REGIME_MODE="NORMAL",
                         ret_pct=8 + rng.normal(0, 2), horizon=5, rec_date="20260520"))
    for _ in range(120):  # B: 같은 날 시장 평균 아래 (엣지 없음)
        rows.append(dict(ELITE_SCORE=85, ACTION_TIER="PASS", MACRO_REGIME_MODE="NORMAL",
                         ret_pct=2 + rng.normal(0, 2), horizon=5, rec_date="20260520"))
    df = pd.DataFrame(rows)
    df["win"] = (df["ret_pct"] > 0).astype(int)  # 절대 기준이면 거의 전부 win
    return df


def test_absolute_inflated_but_excess_realistic():
    """핵심: 상승장에서 absolute prior는 부풀고(>0.9), excess prior는 ~0.5로 정직."""
    tr = _ret_trades()
    abs_t = build_segmented_table(tr, score_col="ELITE_SCORE", win_col="win",
                                  segment_cols=["ACTION_TIER"], win_basis="absolute", asof_ymd="20260529")
    exc_t = build_segmented_table(tr, score_col="ELITE_SCORE", win_col="win", ret_col="ret_pct",
                                  segment_cols=["ACTION_TIER"], win_basis="excess", asof_ymd="20260529")
    assert abs_t["meta"]["global_prior"] > 0.9          # 절대 기준: 거의 다 win (인플레)
    assert 0.35 < exc_t["meta"]["global_prior"] < 0.65  # 초과 기준: 정직한 ~0.5
    assert exc_t["meta"]["benchmark_source"] == "day_relative_mean"


def test_excess_separates_good_from_market_following():
    """excess 기준: 시장 위 세그먼트 > 0.5, 시장 추종 세그먼트 < 0.5 로 분리."""
    tr = _ret_trades()
    exc_t = build_segmented_table(tr, score_col="ELITE_SCORE", win_col="win", ret_col="ret_pct",
                                  segment_cols=["ACTION_TIER"], win_basis="excess", asof_ymd="20260529")
    good = score_segment(pd.Series(dict(ELITE_SCORE=85, ACTION_TIER="NOW_BUY", MACRO_REGIME_MODE="NORMAL")), exc_t)[0]
    weak = score_segment(pd.Series(dict(ELITE_SCORE=85, ACTION_TIER="PASS", MACRO_REGIME_MODE="NORMAL")), exc_t)[0]
    assert good > weak
    assert good > 0.5 and weak < 0.5


def test_external_benchmark_hook():
    """benchmark_returns 콜백을 주면 지수 대비 초과로 계산되고 source가 바뀐다."""
    tr = _ret_trades()
    exc_t = build_segmented_table(tr, score_col="ELITE_SCORE", win_col="win", ret_col="ret_pct",
                                  segment_cols=["ACTION_TIER"], win_basis="excess",
                                  benchmark_returns=lambda d, h: 5.0, asof_ymd="20260529")
    assert exc_t["meta"]["benchmark_source"] == "external_benchmark"


# ── 리뷰 지적 수정 회귀 ──────────────────────────────────────────────────────
def test_score_segment_uses_table_score_col_not_always_elite():
    """치명 1: lookup은 테이블 score_col/lookup_col을 따라야 함 (무조건 ELITE 금지)."""
    trades = pd.DataFrame([
        dict(DISPLAY_SCORE=85, ELITE_SCORE=55, ACTION_TIER="NOW_BUY",
             MACRO_REGIME_MODE="NORMAL", is_win=1, rec_date="20260520")
        for _ in range(40)
    ])
    table = build_segmented_table(
        trades, score_col="DISPLAY_SCORE", win_col="is_win",
        segment_cols=["ACTION_TIER", "MACRO_REGIME_MODE"], asof_ymd="20260529",
    )
    assert table["meta"]["lookup_col"] == "DISPLAY_SCORE"
    row = pd.Series(dict(DISPLAY_SCORE=85, ELITE_SCORE=55,
                         ACTION_TIER="NOW_BUY", MACRO_REGIME_MODE="NORMAL"))
    p, n, key, suff = score_segment(row, table)
    assert key.startswith("80-90"), f"DISPLAY_SCORE(85) 버킷이어야 하는데: {key}"


def test_explicit_lookup_col_overrides_score_col():
    """라이브 패턴: 로그 score_col='score'지만 lookup_col='DISPLAY_SCORE'로 추론."""
    trades = pd.DataFrame([
        dict(score=85, ACTION_TIER="NOW_BUY", MACRO_REGIME_MODE="NORMAL",
             is_win=1, rec_date="20260520") for _ in range(40)
    ])
    table = build_segmented_table(
        trades, score_col="score", win_col="is_win", lookup_col="DISPLAY_SCORE",
        segment_cols=["ACTION_TIER", "MACRO_REGIME_MODE"], asof_ymd="20260529",
    )
    row = pd.Series(dict(DISPLAY_SCORE=85, ACTION_TIER="NOW_BUY", MACRO_REGIME_MODE="NORMAL"))
    p, n, key, suff = score_segment(row, table)
    assert key.startswith("80-90")


def test_build_segmented_table_accepts_none():
    """치명 2: trades=None이어도 터지지 않고 prior 폴백."""
    t = build_segmented_table(None)
    assert t["meta"]["global_prior"] == 0.50
    assert t["meta"]["is_sufficient"] is False
    assert t["table"] == []
    assert t["meta"]["n_raw_total"] == 0


def test_add_v4_shadow_real_recommend_schema():
    """실제 recommend 스키마(TP1_PCT/BALANCE_SCORE/CALIBRATION_MODE) 호환 + 컬럼 결측 안전."""
    t = build_segmented_table(_trades(n_high=120), asof_ymd="20260529")
    df = pd.DataFrame([
        dict(종목명="가", ELITE_SCORE=85, TP1_PCT=10.0, BALANCE_SCORE=80,
             CALIBRATION_MODE="MATURE", ACTION_TIER="NOW_BUY", MACRO_REGIME_MODE="NORMAL", TOP_PICK=0),
        dict(종목명="나", ELITE_SCORE=60, TP1_PCT=30.0, BALANCE_SCORE=40,
             CALIBRATION_MODE="LIGHT", ACTION_TIER="PASS", MACRO_REGIME_MODE="NORMAL", TOP_PICK=0),
    ])
    out = add_v4_shadow_columns(df, t)
    for c in ["EST_WIN_RATE_V4", "EST_WIN_RATE_V4_N", "STABLE_GATE_V4_PASS", "TOP_PICK_V4_SHADOW"]:
        assert c in out.columns
        assert out[c].notna().all(), f"{c}에 결측 발생"
    # shadow 폭증 방지: 추가 픽이 전체를 넘지 않음
    assert out["TOP_PICK_V4_SHADOW"].sum() <= len(out)


def test_csv_smoke_if_present():
    """실전 recommend_latest.csv가 있으면 shadow 적용 smoke (결측/폭증 점검). 없으면 skip."""
    import os
    csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "recommend_latest.csv")
    if not os.path.exists(csv):
        import pytest
        pytest.skip("recommend_latest.csv 없음 — 운영 환경에서만 실행")
    df = pd.read_csv(csv)
    t = build_segmented_table(_trades(n_high=120), asof_ymd="20260529")
    out = add_v4_shadow_columns(df, t)
    assert len(out) == len(df)
    assert out["EST_WIN_RATE_V4"].between(0, 1).all()
    assert out["TOP_PICK_V4_SHADOW"].sum() <= len(out)
