# -*- coding: utf-8 -*-
"""[v34] 오늘 탭 UX — risk_off 잠금 시각화 + 알파 세계관 정합.

7/16 실사용 피드백: 시장폭 88%로 '시장 환경 통과' 표시인데 매수 0개 —
실제 차단자(risk_off)가 화면에 없어 '왜 안 사는지' 알 수 없었다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.decision_center import (
    _risk_off_state,
    _humanize_reason,
    _alpha_gate_on,
    build_decision_summary,
)


def _row(**overrides):
    row = {
        "종목코드": "000001", "종목명": "테스트",
        "TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0, "PRODUCTION_BUY": 0,
        "ACTION_DECISION": "WATCH", "QUALITY_GUARD_SCORE": 65.0,
        "QUALITY_GUARD_REASON": "알파 88점 (진입선 90점 미달)",
        "MARKET_BREADTH": 55.0, "MACRO_RISK": "NORMAL", "MARKET_REGIME": "NEUTRAL",
        "ALPHA_GATE_ACTIVE": 1, "ALPHA_SCORE": 88.0, "ALPHA_WIN_PROB": 0.41,
        "ALPHA_ENTRY_THRESHOLD": 90.0,
        "NEW_ENTRY_BLOCKED": False, "STOP_OVERRIDE_REASON": "",
        "추천매수가": 10000.0, "종가": 10000.0, "손절가": 9500.0,
        "추천매도가1": 11000.0, "RR_NOW_TP1": 2.0,
    }
    row.update(overrides)
    return row


# ── risk_off 잠금 감지 ──

def test_risk_off_detected_from_columns():
    df = pd.DataFrame([_row(NEW_ENTRY_BLOCKED=True,
                            STOP_OVERRIDE_REASON="risk_off: 추천손절 유지 + 신규진입 차단")] * 3)
    state = _risk_off_state(df)
    assert state["active"] is True
    assert state["line"]  # 해제 조건 문구 존재


def test_risk_off_not_detected_when_unblocked():
    df = pd.DataFrame([_row()] * 3)
    assert _risk_off_state(df)["active"] is False


def test_risk_off_requires_reason_and_blocked():
    # NEW_ENTRY_BLOCKED만 있고 사유가 risk_off가 아니면(7월방어 등) 잠금 아님
    df = pd.DataFrame([_row(NEW_ENTRY_BLOCKED=True,
                            STOP_OVERRIDE_REASON="JULY_PROFIT_DEFENSE: 과열")] * 3)
    assert _risk_off_state(df)["active"] is False


# ── 게이트/차단 사유 반영 ──

def test_gate1_blocked_when_risk_off():
    df = pd.DataFrame([_row(NEW_ENTRY_BLOCKED=True,
                            STOP_OVERRIDE_REASON="risk_off: 신규진입 차단",
                            MARKET_BREADTH=88.0)] * 5)
    s = build_decision_summary(df)
    g1 = s["gates"][0]
    # 시장폭 88%여도 '통과'로 오도하지 않는다
    assert g1["status"] == "차단"
    assert "폭락 방어" in g1["value"]
    assert s["market_ready"] is False
    assert any("risk_off" in b or "폭락 방어" in b for b in s["blockers"])
    assert s["risk_off"]["active"] is True


def test_gate1_passes_without_risk_off():
    df = pd.DataFrame([_row(MARKET_BREADTH=60.0)] * 5)
    s = build_decision_summary(df)
    assert s["gates"][0]["status"] == "통과"
    assert s["risk_off"]["active"] is False


def test_gate2_shows_alpha_threshold():
    df = pd.DataFrame([_row(ALPHA_ENTRY_THRESHOLD=90.0)] * 5)
    s = build_decision_summary(df)
    assert "알파 상위 10%" in s["gates"][1]["detail"]
    assert s["alpha_gate"] is True


# ── 알파 정렬·페이로드 ──

def test_watch_sorted_by_alpha_when_gate_active():
    df = pd.DataFrame([
        _row(종목명="알파낮음", ALPHA_SCORE=60.0, QUALITY_GUARD_SCORE=90.0),
        _row(종목명="알파높음", ALPHA_SCORE=99.0, QUALITY_GUARD_SCORE=50.0),
    ])
    s = build_decision_summary(df)
    assert s["watch"][0]["name"] == "알파높음"     # 품질점수보다 알파 우선
    assert s["watch"][0]["alpha"] == 99.0
    assert abs(s["watch"][0]["alpha_win_prob"] - 0.41) < 1e-9


# ── 사유 번역 (v32 용어) ──

def test_humanize_alpha_reason_passthrough():
    out = _humanize_reason("알파 88점 (진입선 90점 미달)")
    assert "알파 88점" in out


def test_humanize_loss_defense_translated():
    out = _humanize_reason("손실방어 차단")
    assert "폭락 방어" in out and "자동 해제" in out


def test_alpha_gate_on_detection():
    assert _alpha_gate_on(pd.DataFrame([_row()])) is True
    assert _alpha_gate_on(pd.DataFrame([_row(ALPHA_GATE_ACTIVE=0)])) is False
    assert _alpha_gate_on(pd.DataFrame()) is False


# ── quality guard 사유 정합 (v34) ──

def test_quality_reason_no_regime_block_text_under_alpha_gate():
    from services.recommendation_quality import apply_recommendation_quality_guard
    df = pd.DataFrame([_row(MARKET_REGIME="DOWN", MARKET_BREADTH=30.0,
                            ALPHA_ENTRY_OK=0)])
    out = apply_recommendation_quality_guard(df)
    reason = str(out.iloc[0]["QUALITY_GUARD_REASON"])
    # 알파 게이트에선 DOWN/시장폭이 하드블록 사유로 표기되지 않는다
    assert "하락 레짐 (신규진입 차단)" not in reason
    assert "내부 약세" not in reason


# ═══════════════════════════════════════════════════
# [v35] 하루 1종목 랭킹 — 알파×손익비 (역선택 수정)
# ═══════════════════════════════════════════════════
# 실측(46일): 품질점수 랭킹 -1.29%/일(승률 33%) vs 알파×RR +2.07%/일(승률 46%).

def _prod_row(**overrides):
    row = _row(TOP_PICK=1, BUY_NOW_ELIGIBLE=1, ACTION_DECISION="BUY",
               ALPHA_ENTRY_OK=1, RR_NOW_TP1=2.0, DATA_FRESHNESS_OK=True,
               ABNORMAL_HISTORY_GUARD_FLAG=0, PROFIT_RECOVERY_BLOCK_FLAG=0,
               JULY_PROFIT_BLOCK_FLAG=0, POC_GAP=8.0)
    row.update(overrides)
    return row


def test_v35_daily_pick_ranked_by_alpha_not_quality():
    from services.recommendation_quality import apply_recommendation_quality_guard
    df = pd.DataFrame([
        _prod_row(종목명="품질높음", QUALITY_GUARD_SCORE=95.0, DISPLAY_SCORE=90.0,
                  ALPHA_SCORE=82.0, RR_NOW_TP1=1.5),
        _prod_row(종목명="알파높음", QUALITY_GUARD_SCORE=70.0, DISPLAY_SCORE=40.0,
                  ALPHA_SCORE=97.0, RR_NOW_TP1=2.5),
    ])
    out = apply_recommendation_quality_guard(df)
    picked = out[out["PRODUCTION_BUY"] == 1]
    assert len(picked) == 1
    # 품질점수가 아니라 알파×RR이 높은 쪽이 공식 매수
    assert picked.iloc[0]["종목명"] == "알파높음"


def test_v35_legacy_rank_unchanged_without_alpha_gate():
    from services.recommendation_quality import apply_recommendation_quality_guard
    # 알파 미검증(레거시) — 기존 품질점수 랭킹 유지
    base = dict(TOP_PICK=1, BUY_NOW_ELIGIBLE=1, BUY_NOW_GRADE="BUY",
                BUY_NOW_SCORE=82, TOP_PICK_TYPE="STABLE",
                ENTRY_RISK_LEVEL="GREEN", ROUTE="ATTACK", MACRO_RISK="NORMAL",
                MFI14=74, RES_RATIO_NEAR=0.15, VWAP_GAP=10, RR_NOW_TP1=2.0,
                DATA_FRESHNESS_OK=True, ABNORMAL_HISTORY_GUARD_FLAG=0,
                PROFIT_RECOVERY_BLOCK_FLAG=0, JULY_PROFIT_BLOCK_FLAG=0,
                POC_GAP=8.0, MARKET_BREADTH=55.0, 종목코드="000001")
    df = pd.DataFrame([
        dict(base, 종목명="점수높음", DISPLAY_SCORE=92.0),
        dict(base, 종목명="점수낮음", DISPLAY_SCORE=75.0),
    ])
    out = apply_recommendation_quality_guard(df)
    picked = out[out["PRODUCTION_BUY"] == 1]
    assert len(picked) == 1
    assert picked.iloc[0]["종목명"] == "점수높음"
