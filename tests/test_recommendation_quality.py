# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.recommendation_quality import (
    apply_recommendation_quality_guard,
    production_buy_mask,
)


def _row(**overrides):
    row = {
        "TOP_PICK": 1,
        "BUY_NOW_ELIGIBLE": 1,
        "BUY_NOW_GRADE": "BUY",
        "BUY_NOW_SCORE": 82,
        "TOP_PICK_TYPE": "STABLE",
        "ENTRY_RISK_LEVEL": "GREEN",
        "ROUTE": "ATTACK",
        "MACRO_RISK": "NORMAL",
        "MFI14": 74,
        "RES_RATIO_NEAR": 0.15,
        "VWAP_GAP": 10,
        "RR_NOW_TP1": 2.0,
        "DISPLAY_SCORE": 84,
        "DATA_FRESHNESS_OK": True,
        "ABNORMAL_HISTORY_GUARD_FLAG": 0,
        "PROFIT_RECOVERY_BLOCK_FLAG": 0,
        "JULY_PROFIT_BLOCK_FLAG": 0,
    }
    row.update(overrides)
    return row


def test_strict_stable_candidate_can_be_production_buy():
    guarded = apply_recommendation_quality_guard(pd.DataFrame([_row()]))
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 1
    assert guarded.iloc[0]["ACTION_DECISION"] == "BUY"
    assert guarded.iloc[0]["RECOMMENDED_WEIGHT_PCT"] <= 5
    assert production_buy_mask(guarded).iloc[0]


def test_aggressive_candidate_is_blocked_after_observed_losses():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(TOP_PICK_TYPE="AGGRESSIVE")])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert guarded.iloc[0]["BUY_NOW_ELIGIBLE"] == 0
    assert "공격형 검증 실패" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_guard_never_promotes_non_official_candidate():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(TOP_PICK=0, BUY_NOW_ELIGIBLE=0)])
    )
    assert guarded.iloc[0]["QUALITY_GUARD_PASS"] == 1
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert guarded.iloc[0]["BUY_NOW_ELIGIBLE"] == 0


def test_risk_and_resistance_failures_force_cash_or_watch():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(ENTRY_RISK_LEVEL="ORANGE", RES_RATIO_NEAR=0.01)])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert guarded.iloc[0]["RECOMMENDED_WEIGHT_PCT"] == 0


def test_low_buy_now_score_is_never_official():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(BUY_NOW_SCORE=60)])
    )
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0
    assert "즉시매수점수 60" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_route_is_not_an_evidence_veto_under_alpha_gate_v32():
    """[v32] ROUTE는 진입 '근거'가 아니다 (ATTACK 알파 -2.9%p, p=0.0004).

    검증된 알파가 진입 SSOT이므로 ROUTE=WAIT라도 근거 판정
    (QUALITY_GUARD_PASS)은 통과하고, 탈락 사유에 모멘텀형 '경로 X'를 적지
    않는다. 이것이 v32가 실측으로 확립한 계약이며 그대로 유지된다.

    [v54] 단, '공식 매수'는 실행 가능해야 한다. 사이징 엔진
    (kelly_calibrator._KELLY_INACTIVE_ROUTES)이 WAIT에서 켈리를 0주로 강제하므로
    PRODUCTION_BUY는 0이 된다 — 근거 통과와 실행 가능성은 다른 판정이다.
    v32가 제거한 것은 ROUTE 근거 거부권이고, v54가 넣은 것은 실행 가능성
    정렬이다. 자세한 측정 기록은 test_v54_first_pick_contradictions.py 참조.
    """
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(ROUTE="WAIT", ALPHA_GATE_ACTIVE=1, ALPHA_ENTRY_OK=1,
                           ALPHA_SCORE=92, ALPHA_ENTRY_THRESHOLD=80,
                           MARKET_REGIME="NEUTRAL")])
    )
    assert guarded.iloc[0]["QUALITY_GUARD_PASS"] == 1
    assert "경로" not in guarded.iloc[0]["QUALITY_GUARD_REASON"]
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 0

    # 같은 행에서 상태만 실행 가능하게 바꾸면 공식 매수가 된다
    # → 탈락시킨 것은 알파도 근거도 아니라 실행 가능성뿐임을 고정한다.
    sizable = apply_recommendation_quality_guard(
        pd.DataFrame([_row(ROUTE="ATTACK", ALPHA_GATE_ACTIVE=1, ALPHA_ENTRY_OK=1,
                           ALPHA_SCORE=92, ALPHA_ENTRY_THRESHOLD=80,
                           MARKET_REGIME="NEUTRAL")])
    )
    assert sizable.iloc[0]["PRODUCTION_BUY"] == 1


def test_legacy_fallback_has_no_route_veto_v32():
    # 알파 미검증(레거시 폴백)에서도 ROUTE 거부권은 없다 —
    # 품질 게이트를 통과하면 ARMED도 ROUTE 사유로 막지 않는다.
    guarded = apply_recommendation_quality_guard(pd.DataFrame([_row(ROUTE="ARMED")]))
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 1
    assert "경로 ARMED" not in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_only_one_new_position_can_be_production_buy():
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(DISPLAY_SCORE=82), _row(DISPLAY_SCORE=90)])
    )
    assert guarded["PRODUCTION_BUY"].sum() == 1
    assert guarded.iloc[1]["PRODUCTION_BUY"] == 1
    assert "당일 신규진입 1종목 제한" in guarded.iloc[0]["QUALITY_GUARD_REASON"]


def test_guard_is_idempotent_and_preserves_pre_quality_audit_flag():
    first = apply_recommendation_quality_guard(
        pd.DataFrame([_row(TOP_PICK_TYPE="AGGRESSIVE")])
    )
    second = apply_recommendation_quality_guard(first)
    assert second.iloc[0]["PRE_QUALITY_BUY_NOW_ELIGIBLE"] == 1
    assert second.iloc[0]["PRODUCTION_BUY"] == 0
