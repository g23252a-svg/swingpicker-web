# -*- coding: utf-8 -*-
"""test_momentum_lane.py — v23.1 Momentum Lane 회귀 테스트."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momentum_lane import (  # noqa: E402
    apply_momentum_lane,
    compute_market_risk_off,
    momentum_summary,
    MOMENTUM_LANE_COLS,
)


def _pool(n_overheat=6, n_attack=1, guard_pass=True, rr=0.7):
    """OVERHEAT n개 + ATTACK 1개 합성. 점수는 90,89,... 내림차순."""
    rows = []
    for i in range(n_overheat):
        rows.append(dict(종목명=f"과열{i}", ROUTE="OVERHEAT",
                         GUARD_ALL_PASS=guard_pass, GUARDED_ELITE_SCORE=90 - i,
                         ELITE_SCORE=90 - i, RR_NOW_TP1=rr))
    for j in range(n_attack):
        rows.append(dict(종목명=f"공격{j}", ROUTE="ATTACK",
                         GUARD_ALL_PASS=True, GUARDED_ELITE_SCORE=95,
                         ELITE_SCORE=95, RR_NOW_TP1=1.5))
    return pd.DataFrame(rows)


# ───────────────────────── 산출 컬럼 계약 ─────────────────────────
def test_contract_columns_present():
    out = apply_momentum_lane(_pool(), market_risk_off=False)
    for c in MOMENTUM_LANE_COLS:
        assert c in out.columns, f"누락 컬럼: {c}"


def test_empty_df_safe():
    out = apply_momentum_lane(pd.DataFrame(), market_risk_off=False)
    for c in MOMENTUM_LANE_COLS:
        assert c in out.columns
    assert len(out) == 0


def test_read_only_no_mutation():
    df = _pool()
    before = df.copy()
    _ = apply_momentum_lane(df, market_risk_off=False)
    pd.testing.assert_frame_equal(df, before)  # 원본 불변


# ───────────────────────── 레인 진입 조건 ─────────────────────────
def test_overheat_guard_pass_enters_lane():
    out = apply_momentum_lane(_pool(n_overheat=3), market_risk_off=False)
    in_lane = out[out["ROUTE"] == "OVERHEAT"]
    assert (in_lane["MOMENTUM_LANE_TIER"] != "").all()


def test_attack_route_excluded():
    out = apply_momentum_lane(_pool(), market_risk_off=False)
    atk = out[out["ROUTE"] == "ATTACK"]
    assert (atk["MOMENTUM_LANE"] == 0).all()
    assert (atk["MOMENTUM_LANE_TIER"] == "").all()


def test_guard_fail_excluded():
    out = apply_momentum_lane(_pool(guard_pass=False), market_risk_off=False)
    ov = out[out["ROUTE"] == "OVERHEAT"]
    assert (ov["MOMENTUM_LANE"] == 0).all()
    assert (ov["MOMENTUM_WATCH"] == 0).all()


def test_no_guard_column_disables_lane():
    df = _pool().drop(columns=["GUARD_ALL_PASS"])
    out = apply_momentum_lane(df, market_risk_off=False)
    assert int(out["MOMENTUM_LANE"].sum()) == 0


# ───────────────────────── Tier 랭크 ─────────────────────────
def test_top_n_are_tier_a():
    out = apply_momentum_lane(_pool(n_overheat=8), market_risk_off=False, config=None)
    # 기본 max_picks=5 → Tier A 5개
    assert int(out["MOMENTUM_LANE"].sum()) == 5
    assert int(out["MOMENTUM_WATCH"].sum()) == 3


def test_rank_follows_score():
    out = apply_momentum_lane(_pool(n_overheat=5), market_risk_off=False)
    a = out[out["MOMENTUM_LANE"] == 1].copy()
    # 점수 최고(과열0, 90점)가 랭크 1
    top = a.loc[a["MOMENTUM_LANE_SCORE"].idxmax()]
    assert int(top["MOMENTUM_LANE_RANK"]) == 1


# ───────────────────────── 모멘텀 역설 ─────────────────────────
def test_momentum_paradox_low_rr_still_tier_a():
    """RR이 낮아도(0.7) 점수 상위면 Tier A — RR로 거르지 않는다."""
    out = apply_momentum_lane(_pool(n_overheat=3, rr=0.5), market_risk_off=False)
    # RR 0.5인데도 점수 상위 3개 모두 Tier A
    assert int(out["MOMENTUM_LANE"].sum()) == 3
    assert (out[out["ROUTE"] == "OVERHEAT"]["MOMENTUM_LANE_TIER"] == "A").all()


# ───────────────────────── 시장국면 게이트 ─────────────────────────
def test_market_risk_off_disables_all():
    out = apply_momentum_lane(_pool(n_overheat=6), market_risk_off=True)
    assert int(out["MOMENTUM_LANE"].sum()) == 0
    assert int(out["MOMENTUM_WATCH"].sum()) == 0
    assert "위험회피" in out["MOMENTUM_LANE_REASON"].iloc[0]


def test_regime_uptrend_is_on():
    # 우상향 KOSPI → risk_off False
    dates = [f"2026{m:02d}{d:02d}" for m in [1, 2, 3] for d in range(1, 29)]
    closes = np.linspace(2000, 3000, len(dates))
    k = pd.DataFrame({"date": dates, "close": closes})
    ro, info = compute_market_risk_off(df_kospi=k)
    assert ro is False


def test_regime_downtrend_is_off():
    # close가 MA20 아래 + MA20 하락 + -3% 이상 이탈 → risk_off True
    dates = [f"2026{m:02d}{d:02d}" for m in [1, 2, 3] for d in range(1, 29)]
    up = np.linspace(3000, 3200, 60)
    down = np.linspace(3200, 2600, len(dates) - 60)  # 급락
    closes = np.concatenate([up, down])
    k = pd.DataFrame({"date": dates, "close": closes})
    ro, info = compute_market_risk_off(df_kospi=k)
    assert ro is True
    assert "하락전환" in info["reason"]


def test_regime_missing_data_defaults_on():
    ro, info = compute_market_risk_off(kospi_daily_path="/nonexistent/path.csv")
    assert ro is False  # 데이터 없으면 레인 ON(디폴트)


# ───────────────────────── 요약 ─────────────────────────
def test_summary_counts():
    out = apply_momentum_lane(_pool(n_overheat=8), market_risk_off=False)
    s = momentum_summary(out)
    assert s["tier_a"] == 5
    assert s["tier_b"] == 3
    assert s["top"] is not None


def test_max_picks_config_override():
    class Cfg:
        source_route = "OVERHEAT"
        require_guard = True
        max_picks = 2
    out = apply_momentum_lane(_pool(n_overheat=6), market_risk_off=False, config=Cfg())
    assert int(out["MOMENTUM_LANE"].sum()) == 2
    assert int(out["MOMENTUM_WATCH"].sum()) == 4

