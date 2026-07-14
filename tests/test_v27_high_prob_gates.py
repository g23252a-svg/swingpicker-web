# -*- coding: utf-8 -*-
"""[v27] 고확률 진입 게이트 테스트.

근거 데이터 (2026-02-26 ~ 2026-07-13 실현 트레이드 157건):
  - POC_GAP (0,20] 승률 40~48% / (20,∞) 승률 12~23%
  - POC_GAP≤20 & MARKET_BREADTH≥35 결합 시
    일별 최강 Top3: 평균 -2.2%/건·승률 33% → +2.3%/건·승률 50%
  - 환율 절대레벨 CRITICAL(≥1490)은 2026-03~07 90일 중 56일 발동
    → 구조적 고환율 레짐에서 추천 전면 차단 (top_pick 0)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest_validation import (
    _v27_entry_gate_ok,
    _v27_poc_sort_key,
    pick_top3_codes,
)
from scoring_engine import compute_elite_score
from services.recommendation_quality import apply_recommendation_quality_guard


# ═══════════════════════════════════════════════════
# 1. backtest_validation — 진입 게이트
# ═══════════════════════════════════════════════════

def test_entry_gate_blocks_extended_poc():
    assert not _v27_entry_gate_ok({"POC_GAP": "35.2", "MARKET_BREADTH": "60"})


def test_entry_gate_blocks_weak_breadth():
    assert not _v27_entry_gate_ok({"POC_GAP": "5.0", "MARKET_BREADTH": "22.5"})


def test_entry_gate_passes_good_setup():
    assert _v27_entry_gate_ok({"POC_GAP": "8.1", "MARKET_BREADTH": "55.0"})


def test_entry_gate_passes_legacy_rows_without_columns():
    # legacy CSV (컬럼 없음/빈값) → 통과 (backward compat)
    assert _v27_entry_gate_ok({})
    assert _v27_entry_gate_ok({"POC_GAP": "", "MARKET_BREADTH": ""})
    assert _v27_entry_gate_ok({"POC_GAP": "abc", "MARKET_BREADTH": None})


def test_poc_sort_key_missing_goes_last():
    assert _v27_poc_sort_key({"POC_GAP": "3.5"}) == 3.5
    assert _v27_poc_sort_key({}) == 99.0
    assert _v27_poc_sort_key({"POC_GAP": "n/a"}) == 99.0


def _strong_row(code: str, poc: float, breadth: float = 60.0, sector: str = "") -> dict:
    """🏆 최강 라벨 조건 (평균≥70·밸≥70·갭≤3%·RR≥0.8) 충족 행."""
    return {
        "종목코드": code, "종목명": f"T{code}", "업종": sector or f"S{code}",
        "STRUCT_SCORE": 80, "TIMING_SCORE": 75, "AI_SCORE": 70,
        "종가": 10000, "추천매수가": 10000, "손절가": 9200,
        "추천매도가1": 11500, "RR_NOW_TP1": 1.9,
        "POC_GAP": str(poc), "MARKET_BREADTH": str(breadth),
    }


def test_pick_top3_prefers_value_area_proximity():
    rows = [
        _strong_row("000001", poc=18.0),
        _strong_row("000002", poc=2.0),
        _strong_row("000003", poc=9.0),
    ]
    picked = pick_top3_codes(rows)
    assert [p["code"] for p in picked] == ["000002", "000003", "000001"]


def test_pick_top3_drops_extended_and_weak_market():
    rows = [
        _strong_row("000001", poc=45.0),               # 확장 → 차단
        _strong_row("000002", poc=5.0, breadth=20.0),  # 시장약세 → 차단
        _strong_row("000003", poc=5.0),
    ]
    picked = pick_top3_codes(rows)
    assert [p["code"] for p in picked] == ["000003"]


# ═══════════════════════════════════════════════════
# 2. scoring_engine — TOP_PICK POC hard gate
# ═══════════════════════════════════════════════════

def _elite_df(poc_gap):
    return pd.DataFrame([{
        "STRUCT_SCORE": 85, "TIMING_SCORE": 80, "AI_SCORE": 75,
        "종가": 10000, "손절가": 9300, "추천매도가1": 11800, "추천매수가": 10000,
        "ROUTE": "ATTACK", "PASS_EBS": 1, "거래대금(억원)": 500,
        "POC_GAP": poc_gap,
    }])


def test_elite_top_pick_blocked_when_extended():
    out, _ = compute_elite_score(_elite_df(poc_gap=50.0))
    assert int(out.iloc[0]["TOP_PICK"]) == 0
    assert int(out.iloc[0]["EXTENSION_BLOCK"]) == 1


def test_elite_top_pick_allowed_near_value_area():
    out, _ = compute_elite_score(_elite_df(poc_gap=5.0))
    assert int(out.iloc[0]["EXTENSION_BLOCK"]) == 0
    # 다른 게이트(AGGRESSIVE 기준)까지 충족하므로 TOP_PICK 유지
    assert int(out.iloc[0]["TOP_PICK"]) == 1


def test_elite_top_pick_legacy_without_poc_column():
    df = _elite_df(poc_gap=5.0).drop(columns=["POC_GAP"])
    out, _ = compute_elite_score(df)
    # POC 컬럼 없음 → 게이트 통과 (legacy 호환)
    assert int(out.iloc[0]["EXTENSION_BLOCK"]) == 0
    assert int(out.iloc[0]["TOP_PICK"]) == 1


# ═══════════════════════════════════════════════════
# 3. recommendation_quality — PRODUCTION_BUY 게이트
# ═══════════════════════════════════════════════════

def _quality_row(**overrides):
    row = {
        "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1, "BUY_NOW_GRADE": "BUY",
        "BUY_NOW_SCORE": 82, "TOP_PICK_TYPE": "STABLE",
        "ENTRY_RISK_LEVEL": "GREEN", "ROUTE": "ATTACK", "MACRO_RISK": "NORMAL",
        "MFI14": 74, "RES_RATIO_NEAR": 0.15, "VWAP_GAP": 10,
        "RR_NOW_TP1": 2.0, "DISPLAY_SCORE": 84, "DATA_FRESHNESS_OK": True,
        "ABNORMAL_HISTORY_GUARD_FLAG": 0, "PROFIT_RECOVERY_BLOCK_FLAG": 0,
        "JULY_PROFIT_BLOCK_FLAG": 0,
    }
    row.update(overrides)
    return row


def test_production_buy_blocked_on_poc_extension():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_quality_row(POC_GAP=42.0, MARKET_BREADTH=60.0)])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert "POC 확장" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_production_buy_blocked_on_weak_breadth():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_quality_row(POC_GAP=8.0, MARKET_BREADTH=22.5)])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert "시장폭" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_production_buy_passes_with_gates_satisfied():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_quality_row(POC_GAP=8.0, MARKET_BREADTH=55.0)])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 1


def test_production_buy_legacy_without_new_columns():
    # POC_GAP/MARKET_BREADTH 없음 → 게이트 통과 (기존 테스트와 동일 계약)
    guarded = apply_recommendation_quality_guard(pd.DataFrame([_quality_row()]))
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 1


# ═══════════════════════════════════════════════════
# 4. macro_filter — 환율 레짐 판정 (급등 진행형만 CRITICAL)
# ═══════════════════════════════════════════════════

def _fx_df(levels):
    idx = pd.bdate_range(end="2026-07-13", periods=len(levels))
    return pd.DataFrame({"Close": levels}, index=idx)


def test_fx_high_but_stable_is_caution_not_critical(monkeypatch):
    import macro_filter as mf
    # 1505원 고레벨이지만 5거래일 변화 ≈ 0 → 레짐 CAUTION
    fx = _fx_df([1503, 1504, 1506, 1505, 1504, 1505, 1506, 1505])
    monkeypatch.setattr(mf, "HAS_FDR", True)
    monkeypatch.setattr(mf, "HAS_YF", False)
    monkeypatch.setattr(
        mf, "_safe_fdr_fetch",
        lambda symbol, end_dt, lookback_bdays=50, min_rows=1:
            fx if symbol == "USD/KRW" else None,
    )
    risk, msg, _, _ = mf.check_macro_env("20260713")
    assert risk == "CAUTION"
    assert "레짐" in msg


def test_fx_high_and_rising_is_critical(monkeypatch):
    import macro_filter as mf
    # 1450 → 1510 급등 (5거래일 +3.4%) → CRITICAL 유지
    fx = _fx_df([1440, 1445, 1450, 1460, 1475, 1490, 1500, 1510])
    monkeypatch.setattr(mf, "HAS_FDR", True)
    monkeypatch.setattr(mf, "HAS_YF", False)
    monkeypatch.setattr(
        mf, "_safe_fdr_fetch",
        lambda symbol, end_dt, lookback_bdays=50, min_rows=1:
            fx if symbol == "USD/KRW" else None,
    )
    risk, msg, _, _ = mf.check_macro_env("20260713")
    assert risk == "CRITICAL"
