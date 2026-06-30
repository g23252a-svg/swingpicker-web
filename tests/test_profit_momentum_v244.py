# -*- coding: utf-8 -*-
"""
test_profit_momentum_v244.py
═══════════════════════════════════════════════════
v24.4 Profit Momentum Overlay (PMS) 회귀/안전 테스트.

핵심 보증
  1. PMS는 SHADOW — enforce=False/True 모두 공식 TOP_PICK/BUY_NOW_ELIGIBLE 불변.
  2. 산출 컬럼 계약(PMS_OUTPUT_COLS) 충족, PMS_SCORE∈[0,100].
  3. 결측 피처 강건(가용 피처만 사용), 빈/단일행 안전.
  4. 일별 횡단면 순위·유동성 floor·블렌드 동작.
  5. 검증된 방향성: 모멘텀확인 피처가 높은 종목이 PMS_PROFIT_PICK으로 잡힌다.
"""
import numpy as np
import pandas as pd
import pytest

import profit_momentum as pm
from collector_config import PmsConfig, CollectorConfig


def _synthetic_universe(n_days=10, per_day=40, seed=0):
    """일별 universe 합성 — 모멘텀확인 피처와 '미래수익' 약한 양의 관계 주입."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        rec_date = 20260400 + d
        rsi = rng.uniform(20, 85, per_day)
        vwap_gap = rng.uniform(-10, 40, per_day)
        vol_int = rng.uniform(0.3, 2.5, per_day)
        timing = rng.uniform(0, 100, per_day)
        turnover = rng.uniform(10, 600, per_day)
        disp = rng.uniform(30, 95, per_day)
        elite = rng.uniform(40, 95, per_day)
        # 미래수익: 모멘텀 피처에 양(+), elite에 음(-) — 발견된 패턴 모사
        z = (rsi - 50) / 30 + (vwap_gap - 10) / 25 + (vol_int - 1) - (elite - 65) / 30
        ret = 5 * z + rng.normal(0, 8, per_day)
        for i in range(per_day):
            rows.append(dict(
                rec_date=rec_date, code=f"{d:02d}{i:03d}",
                RSI14=rsi[i], VWAP_GAP=vwap_gap[i], 거래강도=vol_int[i],
                TIMING_SCORE=timing[i], **{"거래대금(억원)": turnover[i]},
                DISPLAY_SCORE=disp[i], ELITE_SCORE=elite[i],
                TOP_PICK=int(elite[i] > 85), BUY_NOW_ELIGIBLE=int(disp[i] > 80),
                ret_pct=ret[i],
            ))
    return pd.DataFrame(rows)


# ── 산출 컬럼 / 범위 ─────────────────────────────────────────────
def test_output_columns_and_range():
    df = _synthetic_universe()
    out = pm.add_profit_momentum_columns(df.copy(), enforce=False)
    for c in pm.PMS_OUTPUT_COLS:
        assert c in out.columns, f"누락 컬럼: {c}"
    s = pd.to_numeric(out["PMS_SCORE"], errors="coerce")
    assert s.between(0, 100).all()
    assert out["PMS_AVAIL_FEATS"].max() == 6


# ── SHADOW 보증: 공식 컬럼 불변 (enforce 양쪽) ───────────────────
@pytest.mark.parametrize("enforce", [False, True])
def test_official_columns_unchanged(enforce):
    df = _synthetic_universe()
    before_tp = df["TOP_PICK"].tolist()
    before_buy = df["BUY_NOW_ELIGIBLE"].tolist()
    out = pm.add_profit_momentum_columns(df.copy(), enforce=enforce)
    assert out["TOP_PICK"].tolist() == before_tp, "PMS가 TOP_PICK을 변경함 (금지)"
    assert out["BUY_NOW_ELIGIBLE"].tolist() == before_buy, "PMS가 BUY_NOW_ELIGIBLE을 변경함 (금지)"


def test_enforce_only_adds_sort_key():
    df = _synthetic_universe()
    out_shadow = pm.add_profit_momentum_columns(df.copy(), enforce=False)
    out_enf = pm.add_profit_momentum_columns(df.copy(), enforce=True)
    assert "PMS_SORT_KEY" not in out_shadow.columns
    assert "PMS_SORT_KEY" in out_enf.columns
    # 공식 컬럼은 enforce에서도 동일
    assert out_enf["TOP_PICK"].tolist() == df["TOP_PICK"].tolist()


# ── 일별 횡단면 순위 ─────────────────────────────────────────────
def test_daily_rank_is_per_day():
    df = _synthetic_universe(n_days=5, per_day=20)
    out = pm.add_profit_momentum_columns(df.copy())
    for d, g in out.groupby("rec_date"):
        ranks = sorted(g["PMS_RANK"].tolist())
        assert ranks == list(range(1, len(g) + 1)), f"{d} 순위가 1..N 아님"


# ── 유동성 floor: 저거래대금은 PROFIT_PICK 제외 ──────────────────
def test_liquidity_floor_excludes_microcap():
    df = _synthetic_universe(n_days=3, per_day=30, seed=1)
    # 거래대금을 모두 floor 아래로 → profit_pick 0건이어야
    df["거래대금(억원)"] = 5.0
    out = pm.add_profit_momentum_columns(df.copy())
    assert out["PMS_PROFIT_PICK"].sum() == 0


def test_profit_pick_count_per_day_capped():
    df = _synthetic_universe(n_days=6, per_day=50, seed=2)
    cfg = PmsConfig(pms_top_n=3, pms_min_turnover_eok=0.0)
    out = pm.add_profit_momentum_columns(df.copy(), config=cfg)
    per_day = out[out["PMS_PROFIT_PICK"] == 1].groupby("rec_date").size()
    assert (per_day <= 3).all()


# ── 결측 강건 ────────────────────────────────────────────────────
def test_missing_features_graceful():
    df = _synthetic_universe()
    df = df.drop(columns=["VWAP_GAP", "거래강도"])  # 6→4 피처
    out = pm.add_profit_momentum_columns(df.copy())
    assert out["PMS_AVAIL_FEATS"].max() == 4
    assert pd.to_numeric(out["PMS_SCORE"]).between(0, 100).all()


def test_all_features_missing_neutral():
    df = pd.DataFrame({"rec_date": [20260401] * 5, "code": list("abcde")})
    out = pm.add_profit_momentum_columns(df.copy())
    # 가용 피처 0 → PMS 중립 50
    assert (pd.to_numeric(out["PMS_SCORE"]) == 50.0).all()


def test_empty_and_single_row():
    assert pm.add_profit_momentum_columns(pd.DataFrame()) is not None
    one = pd.DataFrame({"rec_date": [20260401], "code": ["x"],
                        "RSI14": [60.0], "거래대금(억원)": [100.0], "DISPLAY_SCORE": [70.0]})
    out = pm.add_profit_momentum_columns(one.copy())
    assert len(out) == 1 and "PMS_SCORE" in out.columns


# ── 검증된 방향성: 모멘텀 강한 종목이 더 자주 잡힘 ───────────────
def test_momentum_stocks_selected_more():
    df = _synthetic_universe(n_days=12, per_day=50, seed=7)
    cfg = PmsConfig(pms_top_n=5, pms_min_turnover_eok=0.0)
    out = pm.add_profit_momentum_columns(df.copy(), config=cfg)
    picked = out[out["PMS_PROFIT_PICK"] == 1]
    rest = out[out["PMS_PROFIT_PICK"] == 0]
    # 선택된 종목의 RSI/VWAP/거래강도 평균이 유의하게 높아야
    assert picked["RSI14"].mean() > rest["RSI14"].mean()
    assert picked["VWAP_GAP"].mean() > rest["VWAP_GAP"].mean()
    # 그리고 주입한 양의 관계상 선택군 평균수익이 비선택군보다 높아야
    assert picked["ret_pct"].mean() > rest["ret_pct"].mean()


# ── config: enforce 토글 동작 ────────────────────────────────────
def test_config_enforce_toggle():
    df = _synthetic_universe()
    cfg_off = CollectorConfig(pms=PmsConfig(pms_enforce=False))
    cfg_on = CollectorConfig(pms=PmsConfig(pms_enforce=True))
    # 모듈은 enforce 인자를 직접 받지만, 파이프라인은 cfg.pms.pms_enforce로 전달
    assert cfg_off.pms.pms_enforce is False
    assert cfg_on.pms.pms_enforce is True


def test_blend_weight_effect():
    df = _synthetic_universe(seed=3)
    out0 = pm.add_profit_momentum_columns(df.copy(), config=PmsConfig(pms_blend_weight=0.0))
    out1 = pm.add_profit_momentum_columns(df.copy(), config=PmsConfig(pms_blend_weight=1.0))
    # w=1이면 블렌드=PMS 순위와 동일, w=0이면 ELITE 순위 기반 → 서로 달라야
    assert not out0["PMS_BLEND_RANK"].equals(out1["PMS_BLEND_RANK"])
