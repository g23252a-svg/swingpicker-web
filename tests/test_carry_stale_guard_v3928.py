# -*- coding: utf-8 -*-
"""v3.9.28 단계형 보유경과 청산 가드 회귀 테스트.

pytest tests/test_carry_stale_guard_v3928.py -v

대상: pipeline_calibrate.add_carry_stale_guard_columns (순수 함수)
- 단계: 0~3 FRESH · 4~6 WATCH · 7~9 STALE · 10+ DEAD
- 신호: DEAD에서만 CARRY_EXIT_SIGNAL=1 (자동매도 아님, 청산 검토)
- 표현 안전: '매도/자동/팔' 금지
"""
import numpy as np
import pandas as pd

from pipeline_calibrate import add_carry_stale_guard_columns


def _row(**kw):
    base = {"종목코드": "000000", "종목명": "테스트", "DISPLAY_SCORE": 80.0}
    base.update(kw)
    return base


def _one(**kw):
    return add_carry_stale_guard_columns(pd.DataFrame([_row(**kw)])).iloc[0]


def test_fresh_no_penalty_no_signal():
    r = _one(CARRY_AGE_DAYS=2, CARRY_RET_PCT=1.0)
    assert r["CARRY_STALE_STAGE"] == "FRESH"
    assert r["CARRY_EXIT_SIGNAL"] == 0
    assert r["STALE_PENALTY"] == 0
    assert bool(r["IS_STALE_CARRY"]) is False
    assert r["CARRY_STALE_REASON"] == ""


def test_watch_is_display_only():
    """WATCH(4~6일)는 사전경고만 — 감점/신호 없음."""
    r = _one(CARRY_AGE_DAYS=5, CARRY_RET_PCT=-2.0)
    assert r["CARRY_STALE_STAGE"] == "WATCH"
    assert r["CARRY_EXIT_SIGNAL"] == 0
    assert r["STALE_PENALTY"] == 0
    assert bool(r["IS_STALE_CARRY"]) is False
    assert "관찰" in r["CARRY_STALE_REASON"]
    # day6도 WATCH·감점0 (기존 v3.9.27의 day6 감점5 제거 확인)
    r6 = _one(CARRY_AGE_DAYS=6, CARRY_RET_PCT=0.5)
    assert r6["CARRY_STALE_STAGE"] == "WATCH"
    assert r6["STALE_PENALTY"] == 0


def test_stale_penalty_curve_and_loss_warn():
    # day8 이익: penalty 14, 신호 0, 손실 지속 없음
    r = _one(CARRY_AGE_DAYS=8, CARRY_RET_PCT=2.0)
    assert r["CARRY_STALE_STAGE"] == "STALE"
    assert r["CARRY_EXIT_SIGNAL"] == 0
    assert r["STALE_PENALTY"] == 14
    assert bool(r["IS_STALE_CARRY"]) is True
    assert "보유관리 주의" in r["CARRY_STALE_REASON"]
    assert "손실 지속" not in r["CARRY_STALE_REASON"]
    # day8 손실 -7%: penalty 14+5=19, 손실 지속 표기
    rl = _one(CARRY_AGE_DAYS=8, CARRY_RET_PCT=-7.0)
    assert rl["STALE_PENALTY"] == 19
    assert "손실 지속" in rl["CARRY_STALE_REASON"]


def test_stale_curve_values_7_8_9():
    assert _one(CARRY_AGE_DAYS=7, CARRY_RET_PCT=1.0)["STALE_PENALTY"] == 10
    assert _one(CARRY_AGE_DAYS=8, CARRY_RET_PCT=1.0)["STALE_PENALTY"] == 14
    assert _one(CARRY_AGE_DAYS=9, CARRY_RET_PCT=1.0)["STALE_PENALTY"] == 18


def test_dead_triggers_exit_signal_strong_review():
    """신세계I&C 케이스: day10·-9%·구조/타이밍 악화 → DEAD·신호1·강한 청산 검토."""
    r = _one(CARRY_AGE_DAYS=10, CARRY_RET_PCT=-9.0, STRUCT_SCORE=40.0, TIMING_SCORE=30.0)
    assert r["CARRY_STALE_STAGE"] == "DEAD"
    assert r["CARRY_EXIT_SIGNAL"] == 1
    assert r["STALE_PENALTY"] == 27  # 22 + 5(손실)
    assert "강한 청산 검토" in r["CARRY_STALE_REASON"]


def test_dead_not_strong_when_not_all_conditions():
    """DEAD지만 손실 작고 구조/타이밍 양호 → 신호1·'청산 검토'(강한 아님)."""
    r = _one(CARRY_AGE_DAYS=11, CARRY_RET_PCT=-2.0, STRUCT_SCORE=70.0, TIMING_SCORE=60.0)
    assert r["CARRY_STALE_STAGE"] == "DEAD"
    assert r["CARRY_EXIT_SIGNAL"] == 1
    assert r["STALE_PENALTY"] == 24  # 22 + (11-10)*2, 손실<-5 아님 → +5 없음
    assert "청산 검토" in r["CARRY_STALE_REASON"]
    assert "강한 청산 검토" not in r["CARRY_STALE_REASON"]


def test_dead_penalty_capped_at_35():
    r = _one(CARRY_AGE_DAYS=30, CARRY_RET_PCT=-9.0, STRUCT_SCORE=40.0, TIMING_SCORE=30.0)
    assert r["STALE_PENALTY"] == 35  # escalation + 손실 → cap


def test_display_score_is_reduced_by_penalty():
    r = _one(CARRY_AGE_DAYS=10, CARRY_RET_PCT=-9.0, STRUCT_SCORE=40.0, TIMING_SCORE=30.0,
             DISPLAY_SCORE=80.0)
    assert r["DISPLAY_SCORE"] == 80.0 - 27.0  # 53


def test_dead_caught_even_without_pnl():
    """손익(CARRY_RET_PCT) 컬럼이 없어도 age 기반 DEAD는 신호 발생 — 손익은 '—'."""
    r = add_carry_stale_guard_columns(
        pd.DataFrame([{"종목코드": "000000", "DISPLAY_SCORE": 80.0, "CARRY_AGE_DAYS": 10}])
    ).iloc[0]
    assert r["CARRY_STALE_STAGE"] == "DEAD"
    assert r["CARRY_EXIT_SIGNAL"] == 1
    assert r["STALE_PENALTY"] == 22  # 손실 미상 → +5 없음
    assert "청산 검토" in r["CARRY_STALE_REASON"]
    assert "—" in r["CARRY_STALE_REASON"]


def test_is_stale_carry_backward_compat_threshold():
    """기존 buy_now_badge 호환 — IS_STALE_CARRY는 7일+에서 True."""
    assert bool(_one(CARRY_AGE_DAYS=6)["IS_STALE_CARRY"]) is False
    assert bool(_one(CARRY_AGE_DAYS=7)["IS_STALE_CARRY"]) is True


def test_age_computed_from_carry_from_date():
    """CARRY_AGE_DAYS 없으면 CARRY_FROM_DATE로 경과일 계산."""
    df = pd.DataFrame([{"종목코드": "000000", "DISPLAY_SCORE": 70.0,
                        "CARRY_FROM_DATE": "20260501"}])
    r = add_carry_stale_guard_columns(df, today_ymd="20260515").iloc[0]
    assert r["CARRY_AGE_DAYS"] == 14
    assert r["CARRY_STALE_STAGE"] == "DEAD"


def test_no_unsafe_sell_wording_anywhere():
    """표현 안전: 어떤 단계의 사유에도 '매도/자동/팔' 표현이 없어야 한다."""
    df = pd.DataFrame([
        _row(CARRY_AGE_DAYS=2, CARRY_RET_PCT=1.0),
        _row(CARRY_AGE_DAYS=5, CARRY_RET_PCT=-2.0),
        _row(CARRY_AGE_DAYS=8, CARRY_RET_PCT=-7.0),
        _row(CARRY_AGE_DAYS=12, CARRY_RET_PCT=-9.0, STRUCT_SCORE=40.0, TIMING_SCORE=30.0),
    ])
    out = add_carry_stale_guard_columns(df, today_ymd="20260530")
    joined = " ".join(out["CARRY_STALE_REASON"].tolist())
    for banned in ("매도", "자동", "팔"):
        assert banned not in joined, f"금지 표현 '{banned}' 발견: {joined}"


def test_empty_df_safe():
    assert add_carry_stale_guard_columns(pd.DataFrame()).empty
