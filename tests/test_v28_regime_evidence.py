# -*- coding: utf-8 -*-
"""[v28] 레짐 엔진 · 근거 문장 · 레짐 게이트 테스트."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_regime import (
    REGIME_DOWN,
    REGIME_NEUTRAL,
    REGIME_UP,
    REGIME_UNKNOWN,
    compute_market_regime,
    inject_regime_columns,
)
from services.reco_evidence import add_evidence_columns, build_evidence_row
from services.recommendation_quality import apply_recommendation_quality_guard


def _kospi(closes):
    n = len(closes)
    dates = pd.bdate_range(end="2026-07-13", periods=n).strftime("%Y%m%d")
    k = pd.DataFrame({"date": dates, "close": closes})
    k["ma5"] = k["close"].rolling(5).mean()
    k["ma20"] = k["close"].rolling(20).mean()
    return k


# ═══════════════════════════════════════════════════
# 1. 레짐 판정
# ═══════════════════════════════════════════════════

def test_regime_up_when_trend_and_breadth():
    k = _kospi(list(range(2400, 2400 + 25)))  # 꾸준한 상승 → 종가 > MA5 > MA20
    r = compute_market_regime("20260713", market_breadth=62.0, kospi_df=k)
    assert r["regime"] == REGIME_UP
    assert r["size_mult"] == 1.0
    assert r["allow_new_entry"] is True


def test_regime_down_on_breadth_collapse_even_above_ma():
    k = _kospi(list(range(2400, 2400 + 25)))  # 지수는 상승 추세여도
    r = compute_market_regime("20260713", market_breadth=22.5, kospi_df=k)
    assert r["regime"] == REGIME_DOWN  # 내부 붕괴 우선
    # [v32] 알파 전면 게이트 도입 — DOWN도 축소 사이즈(0.3)로 상위10% 진입 허용.
    # 진짜 위험한 하락(risk_off)은 NEW_ENTRY_BLOCKED가 별도 하드블록.
    assert r["size_mult"] == 0.3
    assert r["allow_new_entry"] is False


def test_regime_neutral_between():
    k = _kospi(list(range(2500, 2500 - 25, -1)))  # 하락 추세 지수
    r = compute_market_regime("20260713", market_breadth=45.0, kospi_df=k)
    assert r["regime"] == REGIME_NEUTRAL
    assert r["size_mult"] == 0.5


def test_regime_unknown_without_data():
    r = compute_market_regime("20260713", market_breadth=None, kospi_df=_kospi([2400] * 3))
    # MA20 산출 불가 + 시장폭 없음 → UNKNOWN (보수 운용)
    assert r["regime"] == REGIME_UNKNOWN
    assert r["size_mult"] == 0.5


def test_inject_regime_columns():
    df = pd.DataFrame({"종목코드": ["000001"]})
    out = inject_regime_columns(df, {"regime": "UP", "reason": "r", "size_mult": 1.0,
                                     "allow_new_entry": True})
    assert out.loc[0, "MARKET_REGIME"] == "UP"
    assert out.loc[0, "REGIME_ALLOW_ENTRY"] == 1


# ═══════════════════════════════════════════════════
# 2. 레짐 게이트 (PRODUCTION_BUY)
# ═══════════════════════════════════════════════════

def _quality_row(**overrides):
    row = {
        "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1, "BUY_NOW_GRADE": "BUY",
        "BUY_NOW_SCORE": 82, "TOP_PICK_TYPE": "STABLE",
        "ENTRY_RISK_LEVEL": "GREEN", "ROUTE": "ATTACK", "MACRO_RISK": "NORMAL",
        "MFI14": 74, "RES_RATIO_NEAR": 0.15, "VWAP_GAP": 10,
        "RR_NOW_TP1": 2.0, "DISPLAY_SCORE": 84, "DATA_FRESHNESS_OK": True,
        "ABNORMAL_HISTORY_GUARD_FLAG": 0, "PROFIT_RECOVERY_BLOCK_FLAG": 0,
        "JULY_PROFIT_BLOCK_FLAG": 0, "POC_GAP": 8.0, "MARKET_BREADTH": 55.0,
    }
    row.update(overrides)
    return row


def test_down_regime_blocks_production_buy():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_quality_row(MARKET_REGIME="DOWN", REGIME_SIZE_MULT=0.0)])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert "하락 레짐" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_up_regime_full_weight():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_quality_row(MARKET_REGIME="UP", REGIME_SIZE_MULT=1.0)])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 1
    assert guarded.iloc[0]["RECOMMENDED_WEIGHT_PCT"] in (3.0, 5.0)


def test_neutral_regime_halves_weight():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_quality_row(MARKET_REGIME="NEUTRAL", REGIME_SIZE_MULT=0.5)])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 1
    assert guarded.iloc[0]["RECOMMENDED_WEIGHT_PCT"] in (1.5, 2.5)


def test_missing_regime_column_passes():
    guarded = apply_recommendation_quality_guard(pd.DataFrame([_quality_row()]))
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 1


# ═══════════════════════════════════════════════════
# 3. 근거 문장 생성
# ═══════════════════════════════════════════════════

def test_evidence_prefers_realized_winrate():
    e, k, p, _b = build_evidence_row({
        "EST_WIN_RATE": 0.62, "EST_WIN_RATE_N": 120, "EST_WIN_RATE_MODE": "MATURE",
        "POC_GAP": 5.0, "RR_NOW_TP1": 2.1,
        "종가": 10000, "손절가": 9200, "추천매도가1": 11700,
        "MARKET_REGIME": "UP", "MARKET_BREADTH": 60.0,
    })
    assert "실측 승률 62%" in e
    assert "표본 120건" in e
    assert p == 62.0
    assert "가치영역 근접" in e
    assert "손익비 2.1" in e


def test_evidence_immature_winrate_is_flagged():
    e, _, p, _b = build_evidence_row({
        "EST_WIN_RATE": 0.7, "EST_WIN_RATE_N": 5, "EST_WIN_RATE_MODE": "FALLBACK",
    })
    assert "통계 미성숙" in e
    assert np.isnan(p)


def test_evidence_shows_extension_as_risk():
    _, k, _, _b = build_evidence_row({"POC_GAP": 45.0, "RSI14": 75})
    assert "확장" in k
    assert "RSI 75" in k


def test_evidence_discloses_ml_unused():
    e, _, _, _b = build_evidence_row({"AI_SCORE": 0, "ML_TRUSTED": 0})
    assert "AI 예측 미사용" in e


def test_evidence_flags_stale_price():
    _, k, _, _b = build_evidence_row({"DATA_FRESHNESS_OK": "False"})
    assert "stale" in k


def test_add_evidence_columns_bulk():
    df = pd.DataFrame([
        {"POC_GAP": 5.0, "RR_NOW_TP1": 1.5, "종가": 100, "손절가": 92, "추천매도가1": 115},
        {"POC_GAP": 50.0},
    ])
    out = add_evidence_columns(df)
    assert "RECO_EVIDENCE" in out.columns and "RECO_RISK" in out.columns
    assert "가치영역" in out.loc[0, "RECO_EVIDENCE"]
    assert "확장" in out.loc[1, "RECO_RISK"]
