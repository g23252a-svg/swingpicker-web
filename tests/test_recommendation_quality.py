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

    [v54] 단, '공식 매수'는 실행 가능해야 한다 — 사이징이 0주로 강제되는 상태를
    매수라 부르지 않는다. v32가 제거한 것은 ROUTE 근거 거부권이고, v54가 넣은
    것은 실행 가능성 정렬이다.

    [v57] 그런데 v54의 '실행 가능성'이 WAIT에서 걸린 이유는 위험 계산이 아니라
    켈리의 표시 관례였고, 그것이 v32가 없앤 ROUTE 진입 거부권을 뒷문으로
    복원했다(픽 가능일 29→15일 반감 · 워크포워드 76일). 측정 결과 '사이징 가능'은
    수익을 예측하지 못했고(Welch p=0.296 · 각 상위2 제거 후 차이 0.16%p) WAIT
    면제 쪽이 중위·승률·이상치제거·OOS에서 모두 같거나 나았다.
    → 알파 진입을 통과한 WAIT는 이제 실행 가능하다(PRODUCTION_BUY=1).
    v54 불변식('사이징 불가는 공식 매수가 아니다')은 CARRY 등에서 그대로 유효하다.
    측정 기록: tests/test_v57_route_exempt.py · kelly_calibrator의
    _ALPHA_EXEMPT_INACTIVE_ROUTES 주석.
    """
    guarded = apply_recommendation_quality_guard(
        pd.DataFrame([_row(ROUTE="WAIT", ALPHA_GATE_ACTIVE=1, ALPHA_ENTRY_OK=1,
                           ALPHA_SCORE=92, ALPHA_ENTRY_THRESHOLD=80,
                           MARKET_REGIME="NEUTRAL")])
    )
    assert guarded.iloc[0]["QUALITY_GUARD_PASS"] == 1
    assert "경로" not in guarded.iloc[0]["QUALITY_GUARD_REASON"]
    # [v57] 알파 진입 통과 → 관망 상태여도 공식 매수다
    assert guarded.iloc[0]["PRODUCTION_BUY"] == 1

    # 실행 가능성 요건 자체는 살아 있다 — 보유 중(CARRY)은 여전히 아니다.
    carry = apply_recommendation_quality_guard(
        pd.DataFrame([_row(ROUTE="CARRY", ALPHA_GATE_ACTIVE=1, ALPHA_ENTRY_OK=1,
                           ALPHA_SCORE=92, ALPHA_ENTRY_THRESHOLD=80,
                           MARKET_REGIME="NEUTRAL")])
    )
    assert carry.iloc[0]["QUALITY_GUARD_PASS"] == 1, "근거는 통과"
    assert carry.iloc[0]["PRODUCTION_BUY"] == 0, "실행 가능성에서 탈락"

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
