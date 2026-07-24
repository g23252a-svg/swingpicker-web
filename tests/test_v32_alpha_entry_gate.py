# -*- coding: utf-8 -*-
"""[v32] 알파 전면 진입 게이트 — ROUTE 거부권 대체 검증.

근거(3~7월 워크포워드 OOS 실측):
  · ROUTE ATTACK 알파 -2.9%p (t=-3.56, p=0.0004) — 5개월 전부 음수
  · ROUTE ARMED  알파  0.0%p (t=0.02, p=0.99) — 노이즈
  · 알파 모델    IC   +0.19 (t=6.8), Q5-Q1 +5.05%p, 십분위 승률 24%→41%
→ ROUTE를 진입 게이트에서 제거하고 검증된 알파를 primary 게이트로.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import alpha_engine as ae
from alpha_engine import (
    apply_alpha_entry_gate,
    recompute_route_with_alpha,
    ALPHA_GATE_THRESHOLD,
    ALPHA_GATE_DEFAULT_THRESHOLD,
)


def _row(**overrides):
    """리스크 가드를 통과하는 정상 종목 기본값."""
    row = {
        "종목코드": "000001",
        "종가": 10000.0,
        "손절가": 9500.0,
        "추천매도가1": 11500.0,   # RR=(11500-10000)/(10000-9500)=3.0
        "추천매수가": 10000.0,
        "거래대금(억원)": 100.0,
        "POC_GAP": 8.0,
        "PASS_EBS": 1,
        "ENTRY_RISK_GATE_OK": True,
        "ALPHA_VALIDATED": 1,
        "ALPHA_SCORE": 95.0,
        "MARKET_REGIME": "NEUTRAL",
        "ROUTE": "WAIT",          # ROUTE는 게이트가 아님
        "BUY_NOW_PASS": 1,
    }
    row.update(overrides)
    return row


def _df(rows):
    return pd.DataFrame([_row(**r) for r in rows])


# ── 문턱 상수 ──

def test_regime_thresholds_match_adaptive_choice():
    # [v40] NEUTRAL 80→85 상향 (OOS +1.35→+1.62%, t=2.08). UP/DOWN 유지.
    assert ALPHA_GATE_THRESHOLD["UP"] == 70.0
    assert ALPHA_GATE_THRESHOLD["NEUTRAL"] == 85.0
    assert ALPHA_GATE_THRESHOLD["DOWN"] == 90.0
    assert ALPHA_GATE_DEFAULT_THRESHOLD == 85.0


# ── 게이트 활성/비활성 ──

def test_gate_inactive_when_unvalidated_leaves_top_pick():
    df = _df([{"ALPHA_VALIDATED": 0, "TOP_PICK": 1}])
    out = apply_alpha_entry_gate(df)
    assert out.iloc[0]["ALPHA_GATE_ACTIVE"] == 0
    # 미검증이면 TOP_PICK을 건드리지 않는다(레거시 폴백).
    assert out.iloc[0]["TOP_PICK"] == 1


def test_gate_active_when_validated():
    out = apply_alpha_entry_gate(_df([{}]))
    assert out.iloc[0]["ALPHA_GATE_ACTIVE"] == 1


# ── 레짐 적응형 문턱 ──

def test_neutral_threshold_85_selects_top15():
    # [v40] NEUTRAL: 알파 85 미만은 탈락, 85 이상은 통과.
    df = _df([{"ALPHA_SCORE": 84.0}, {"ALPHA_SCORE": 86.0}])
    out = apply_alpha_entry_gate(df)
    assert out.iloc[0]["TOP_PICK"] == 0
    assert out.iloc[1]["TOP_PICK"] == 1
    assert out.iloc[1]["TOP_PICK_TYPE"] == "ALPHA"


def test_up_regime_is_more_permissive_than_down():
    # 같은 알파 75점: 상승(≥70)은 통과, 하락(≥90)은 탈락.
    up = apply_alpha_entry_gate(_df([{"ALPHA_SCORE": 75.0, "MARKET_REGIME": "UP"}]))
    down = apply_alpha_entry_gate(_df([{"ALPHA_SCORE": 75.0, "MARKET_REGIME": "DOWN"}]))
    assert up.iloc[0]["TOP_PICK"] == 1
    assert down.iloc[0]["TOP_PICK"] == 0


def test_unknown_regime_defaults_to_neutral():
    out = apply_alpha_entry_gate(_df([{"MARKET_REGIME": "UNKNOWN", "ALPHA_SCORE": 86.0}]))
    assert out.iloc[0]["ALPHA_ENTRY_THRESHOLD"] == 85.0
    assert out.iloc[0]["TOP_PICK"] == 1


# ── ROUTE 무관 ──

def test_route_wait_is_selected_when_alpha_high():
    # ROUTE=WAIT여도 알파가 문턱을 넘으면 진입 후보가 된다(ROUTE 거부권 제거).
    out = apply_alpha_entry_gate(_df([{"ROUTE": "WAIT", "ALPHA_SCORE": 95.0}]))
    assert out.iloc[0]["TOP_PICK"] == 1


def test_route_attack_is_rejected_when_alpha_low():
    # ROUTE=ATTACK여도 알파가 낮으면 탈락(역신호 차단).
    out = apply_alpha_entry_gate(_df([{"ROUTE": "ATTACK", "ALPHA_SCORE": 50.0}]))
    assert out.iloc[0]["TOP_PICK"] == 0


# ── 리스크 가드 존중 ──

def test_risk_gate_failure_blocks_even_high_alpha():
    # 알파 100점이어도 리스크 가드(손익비 등) 실패면 탈락.
    out = apply_alpha_entry_gate(_df([{"ALPHA_SCORE": 100.0, "ENTRY_RISK_GATE_OK": False}]))
    assert out.iloc[0]["TOP_PICK"] == 0


def test_risk_gate_reconstructed_when_column_missing():
    # ENTRY_RISK_GATE_OK 없으면 가격 컬럼에서 재구성 — RR<1 이면 탈락.
    r = _row(ALPHA_SCORE=99.0, 추천매도가1=10200.0)  # RR=0.4
    del r["ENTRY_RISK_GATE_OK"]
    out = apply_alpha_entry_gate(pd.DataFrame([r]))
    assert out.iloc[0]["TOP_PICK"] == 0


# ── 손실방어 차단 존중 ──

def test_new_entry_blocked_respected():
    # risk_off 등으로 NEW_ENTRY_BLOCKED면 알파 통과분도 진입 차단.
    out = apply_alpha_entry_gate(_df([{"ALPHA_SCORE": 99.0, "NEW_ENTRY_BLOCKED": True}]))
    assert out.iloc[0]["TOP_PICK"] == 0


def test_july_and_recovery_block_respected():
    out = apply_alpha_entry_gate(_df([
        {"ALPHA_SCORE": 99.0, "JULY_PROFIT_BLOCK_FLAG": 1},
        {"ALPHA_SCORE": 99.0, "PROFIT_RECOVERY_BLOCK_FLAG": 1},
    ]))
    assert out.iloc[0]["TOP_PICK"] == 0
    assert out.iloc[1]["TOP_PICK"] == 0


# ── BUY_NOW_ELIGIBLE 재계산 ──

def test_buy_now_eligible_recomputed_from_alpha_top_pick():
    out = apply_alpha_entry_gate(_df([
        {"ALPHA_SCORE": 95.0, "BUY_NOW_PASS": 1},   # 통과 → eligible 1
        {"ALPHA_SCORE": 95.0, "BUY_NOW_PASS": 0},   # BUY_NOW_PASS 0 → eligible 0
        {"ALPHA_SCORE": 40.0, "BUY_NOW_PASS": 1},   # 알파 미달 → eligible 0
    ]))
    assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == 1
    assert out.iloc[1]["BUY_NOW_ELIGIBLE"] == 0
    assert out.iloc[2]["BUY_NOW_ELIGIBLE"] == 0


# ── 빈 입력 안전성 ──

def test_empty_df_safe():
    out = apply_alpha_entry_gate(pd.DataFrame())
    assert out.empty


# ═══════════════════════════════════════════════════
#  ROUTE 자체 치료 (recompute_route_with_alpha)
# ═══════════════════════════════════════════════════

def _rr(**overrides):
    """ROUTE 재계산 입력 — 기술 구조 포함."""
    row = {
        "종목코드": "000001",
        "ALPHA_VALIDATED": 1,
        "ALPHA_SCORE": 95.0,
        "MARKET_REGIME": "NEUTRAL",   # 문턱 80
        "ROUTE": "WAIT",
        "ROUTE_REASON": "추세 관망",
        "Above_MA20": 1,
        "Low_Trend_PCT": 3.0,          # 저점 상승
        "ret_5d_%": 2.0,               # 비과열
        "MFI14": 55.0,
        "RSI14": 55.0,
        "TTM_SQUEEZE": 0,
    }
    row.update(overrides)
    return row


def _rdf(rows):
    return pd.DataFrame([_rr(**r) for r in rows])


def test_route_unchanged_when_unvalidated():
    df = _rdf([{"ALPHA_VALIDATED": 0, "ROUTE": "WAIT"}])
    out = recompute_route_with_alpha(df)
    assert out.iloc[0]["ROUTE"] == "WAIT"
    assert "ROUTE_ALPHA_HEALED" not in out.columns


def test_healed_attack_requires_alpha_structure_not_overheat():
    # 알파 상위 + MA20위 + 저점상승 + 비과열 → ATTACK
    out = recompute_route_with_alpha(_rdf([{"ALPHA_SCORE": 95.0}]))
    assert out.iloc[0]["ROUTE"] == "ATTACK"
    assert out.iloc[0]["ROUTE_ALPHA_HEALED"] == 1


def test_high_alpha_but_below_ma20_is_not_attack():
    # 알파 높아도 구조(MA20위·저점상승) 미확인이면 ATTACK 아님 → ARMED/WAIT.
    out = recompute_route_with_alpha(_rdf([{"ALPHA_SCORE": 95.0, "Above_MA20": 0}]))
    assert out.iloc[0]["ROUTE"] != "ATTACK"


def test_extended_stock_is_overheat_even_with_alpha():
    # 5일 급등(과열)은 알파 높아도 OVERHEAT (실측 역신호).
    out = recompute_route_with_alpha(_rdf([{"ALPHA_SCORE": 99.0, "ret_5d_%": 20.0}]))
    assert out.iloc[0]["ROUTE"] == "OVERHEAT"


def test_high_mfi_or_rsi_is_overheat():
    out = recompute_route_with_alpha(_rdf([
        {"ALPHA_SCORE": 99.0, "MFI14": 88.0},
        {"ALPHA_SCORE": 99.0, "RSI14": 80.0},
    ]))
    assert out.iloc[0]["ROUTE"] == "OVERHEAT"
    assert out.iloc[1]["ROUTE"] == "OVERHEAT"


def test_healed_armed_is_near_threshold_and_coiled():
    # 알파가 문턱(80)-15=65~80 사이 + 스퀴즈 → ARMED (ATTACK 아님).
    out = recompute_route_with_alpha(_rdf([
        {"ALPHA_SCORE": 70.0, "TTM_SQUEEZE": 1, "Above_MA20": 0, "Low_Trend_PCT": -1.0}
    ]))
    assert out.iloc[0]["ROUTE"] == "ARMED"


def test_low_alpha_is_wait():
    out = recompute_route_with_alpha(_rdf([{"ALPHA_SCORE": 30.0}]))
    assert out.iloc[0]["ROUTE"] == "WAIT"


def test_carry_and_exit_states_preserved():
    out = recompute_route_with_alpha(_rdf([
        {"ROUTE": "CARRY", "ALPHA_SCORE": 95.0},
        {"ROUTE": "EXIT_WARNING", "ALPHA_SCORE": 95.0},
    ]))
    assert out.iloc[0]["ROUTE"] == "CARRY"
    assert out.iloc[1]["ROUTE"] == "EXIT_WARNING"


def test_route_reason_rewritten_on_heal():
    out = recompute_route_with_alpha(_rdf([{"ALPHA_SCORE": 95.0}]))
    assert "알파" in out.iloc[0]["ROUTE_REASON"]
