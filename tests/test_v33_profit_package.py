# -*- coding: utf-8 -*-
"""[v33] 수익률 향상 패키지 검증.

  1) 빠른 부분익절 (FAST_TP) — 실측: 알파 픽 2일차 +2%↑ → 이후 3일 -0.90% 되돌림
  2) Kelly 승률 입력 알파 교체 — 신호(v32)와 베팅 크기의 축 통일
  3) 백테스트 픽 레인 알파 정렬 — 실측 알파 정렬 +5.07%p vs POC 정렬
  4) 캘리브레이션 이중 창(30/90일) — 국면 전환 반응속도
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exit_plan import add_exit_plan_columns, FAST_TP_PCT, FAST_TP_DAY
from kelly_calibrator import apply_kelly_calibrated
from backtest_validation import _v33_alpha_sort_key


# ═══════════════════════════════════════════════════
# 1. 빠른 부분익절 (FAST_TP)
# ═══════════════════════════════════════════════════

def test_fast_tp_price_follows_the_adopted_trigger():
    """[v55.3 갱신] 트리거가 +2% → +3%로 바뀌었다 (값 하드코딩 대신 상수 참조).

    v33의 +2%·절반은 **전체 추천 풀**(n=526) 근거였다. v55.3은 같은 규칙을
    **공식 매수 픽**에만 적용해 격자탐색(트리거 2~6% × 익절비율 25/33/50% ×
    본전스톱 = 30조합, BH-FDR 보정)했고, '승률과 기대값 둘 다'를 목표로 하면
    +3%·1/4이 최적이었다:
        기준(보유)      승률 41.4% · 평균 +1.26% · MDD -33.5%
        +2%·절반(구값)  승률 65.5% · 평균 +0.55%(43%로 반감)
        +3%·1/4(채택)   승률 59.8% · 평균 +1.12%(89% 보존) · MDD -16.5%
                        p_BH=0.0001 · ΔIS +15.9 / ΔOOS +27.8
                        top-5(n=144) 재현: 승률 +16.7%p · 평균 88% 보존
    공식 픽은 우측 꼬리 분포라 익절 비율이 크면 유일한 수익원을 잘라낸다.
    이 테스트는 이제 특정 숫자가 아니라 **상수와 가격의 정합**을 고정한다 —
    파라미터는 근거가 갱신되면 바뀔 수 있고, 어긋나는 것만 막으면 된다.
    """
    out = add_exit_plan_columns(pd.DataFrame([dict(추천매수가=10000.0, ROUTE="WAIT")]))
    assert out.iloc[0]["FAST_TP_PRICE"] == round(10000.0 * (1 + FAST_TP_PCT / 100.0))
    assert FAST_TP_PCT == 3.0 and FAST_TP_DAY == 2


def test_fast_tp_in_note():
    """[v55.3] '절반 익절' → '25% 익절' (꼬리 보존이 변경의 요점)."""
    out = add_exit_plan_columns(pd.DataFrame([dict(추천매수가=10000.0, ROUTE="WAIT")]))
    note = out.iloc[0]["EXIT_PLAN_NOTE"]
    assert "25% 익절" in note and "D+2" in note
    assert "절반 익절" not in note


def test_fast_tp_nan_for_invalid_entry():
    out = add_exit_plan_columns(pd.DataFrame([dict(추천매수가=np.nan, ROUTE="WAIT")]))
    assert pd.isna(out.iloc[0]["FAST_TP_PRICE"])


# ── 치료된 ROUTE 위험등급 정합 (v32.1 후속) ──

def test_healed_attack_is_not_caution():
    # 치료된 ROUTE에서 ATTACK은 최고 신호(+3.70%p) — '과열 주의' 아님.
    out = add_exit_plan_columns(pd.DataFrame([
        dict(추천매수가=10000.0, ROUTE="ATTACK", ROUTE_ALPHA_HEALED=1)]))
    assert out.iloc[0]["EXIT_ROUTE_RISK"] == "OK"


def test_healed_overheat_still_caution():
    out = add_exit_plan_columns(pd.DataFrame([
        dict(추천매수가=10000.0, ROUTE="OVERHEAT", ROUTE_ALPHA_HEALED=1)]))
    assert out.iloc[0]["EXIT_ROUTE_RISK"] == "CAUTION"


def test_legacy_attack_keeps_caution():
    # 미치료(레거시) ROUTE의 ATTACK은 기존대로 CAUTION 유지.
    out = add_exit_plan_columns(pd.DataFrame([
        dict(추천매수가=10000.0, ROUTE="ATTACK")]))
    assert out.iloc[0]["EXIT_ROUTE_RISK"] == "CAUTION"


# ═══════════════════════════════════════════════════
# 2. Kelly 승률 입력 알파 교체
# ═══════════════════════════════════════════════════

def _kelly_row(**overrides):
    row = dict(ELITE_SCORE=80.0, 추천매수가=10000.0, 손절가=9500.0,
               추천매도가1=11500.0)
    row.update(overrides)
    return row


def test_kelly_uses_alpha_win_prob_when_validated(tmp_path):
    df = pd.DataFrame([
        _kelly_row(ALPHA_VALIDATED=1, ALPHA_WIN_PROB=0.41),
        _kelly_row(),  # 레거시 행
    ])
    out = apply_kelly_calibrated(df, out_dir=str(tmp_path))
    assert out.iloc[0]["KELLY_P_SOURCE"] == "ALPHA_WIN_PROB"
    assert out.iloc[1]["KELLY_P_SOURCE"].startswith("CAL_")


def test_kelly_alpha_row_bypasses_elite_threshold(tmp_path):
    # 알파 상위 픽은 ELITE가 낮은 게 정상(역상관 축) — ELITE 10이어도 배팅돼야.
    df = pd.DataFrame([
        _kelly_row(ELITE_SCORE=10.0, ALPHA_VALIDATED=1, ALPHA_WIN_PROB=0.41),
    ])
    out = apply_kelly_calibrated(df, out_dir=str(tmp_path))
    assert out.iloc[0]["KELLY_FRACTION"] > 0
    assert out.iloc[0]["켈리_금액(원)"] > 0


def test_kelly_legacy_row_keeps_elite_threshold(tmp_path):
    # 알파 없는 레거시 행은 기존 ELITE≥60 문턱 유지.
    df = pd.DataFrame([_kelly_row(ELITE_SCORE=10.0)])
    out = apply_kelly_calibrated(df, out_dir=str(tmp_path))
    assert out.iloc[0]["KELLY_FRACTION"] == 0


def test_kelly_invalid_alpha_prob_falls_back(tmp_path):
    # ALPHA_WIN_PROB 결측/0/1 경계는 알파 경로 제외 → 레거시 폴백.
    df = pd.DataFrame([
        _kelly_row(ALPHA_VALIDATED=1, ALPHA_WIN_PROB=np.nan),
        _kelly_row(ALPHA_VALIDATED=1, ALPHA_WIN_PROB=0.0),
    ])
    out = apply_kelly_calibrated(df, out_dir=str(tmp_path))
    assert (out["KELLY_P_SOURCE"].str.startswith("CAL_")).all()


# ═══════════════════════════════════════════════════
# 3. 백테스트 픽 레인 알파 정렬
# ═══════════════════════════════════════════════════

def test_alpha_sort_key_prefers_higher_alpha():
    hi = _v33_alpha_sort_key(dict(ALPHA_VALIDATED="1", ALPHA_SCORE="95"))
    lo = _v33_alpha_sort_key(dict(ALPHA_VALIDATED="1", ALPHA_SCORE="60"))
    assert hi < lo  # 오름차순 정렬에서 알파 높은 쪽이 먼저


def test_alpha_sort_key_legacy_neutral():
    # 미검증/결측 → 0.0 (전원 동률 → 기존 POC 순 유지)
    assert _v33_alpha_sort_key(dict()) == 0.0
    assert _v33_alpha_sort_key(dict(ALPHA_VALIDATED="0", ALPHA_SCORE="95")) == 0.0
    assert _v33_alpha_sort_key(dict(ALPHA_VALIDATED="1")) == 0.0


def test_alpha_sort_ordering_end_to_end():
    rows = [
        dict(alpha=_v33_alpha_sort_key(dict(ALPHA_VALIDATED="1", ALPHA_SCORE="60")), poc=1.0, score=90),
        dict(alpha=_v33_alpha_sort_key(dict(ALPHA_VALIDATED="1", ALPHA_SCORE="95")), poc=15.0, score=50),
    ]
    rows.sort(key=lambda r: (r["alpha"], r["poc"], -r["score"]))
    # 알파 95(poc 15)가 알파 60(poc 1)보다 먼저 — 알파가 POC에 우선
    assert rows[0]["poc"] == 15.0


# ═══════════════════════════════════════════════════
# 4. 캘리브레이션 이중 창
# ═══════════════════════════════════════════════════

def _write_ptl(tmp_path, rows):
    pd.DataFrame(rows).to_csv(tmp_path / "per_trade_log.csv", index=False)


def test_dual_window_blends_recent(tmp_path):
    from kelly_calibrator import build_calibration_table
    # 과거 60일: 전패(0) 40건 / 최근 20일: 전승(1) 20건 → 블렌드가 slow보다 높아야
    rows = []
    for i in range(40):
        rows.append(dict(rec_date="20260401", code=f"A{i}", method="ELITE_SCORE",
                         topk=1, horizon=5, score=85.0, win=0))
    for i in range(20):
        rows.append(dict(rec_date="20260707", code=f"B{i}", method="ELITE_SCORE",
                         topk=1, horizon=5, score=85.0, win=1))
    _write_ptl(tmp_path, rows)
    cal = build_calibration_table(str(tmp_path), asof_ymd="20260716")
    b = cal[(cal.method == "ELITE_SCORE") & (cal.horizon == 5)
            & (cal.score_lo <= 85) & (cal.score_hi > 85)].iloc[0]
    assert b["n_fast"] >= 8
    assert b["p_fast"] > b["p_slow"]          # 최근 창이 회복을 먼저 반영
    assert b["p_calibrated"] > b["p_slow"]    # 블렌드가 slow 단독보다 높음


def test_dual_window_falls_back_when_fast_sparse(tmp_path):
    from kelly_calibrator import build_calibration_table
    # 최근 30일 표본 3건(<8) → 블렌드 없이 slow 단독 (p_fast=None)
    rows = []
    for i in range(40):
        rows.append(dict(rec_date="20260401", code=f"A{i}", method="ELITE_SCORE",
                         topk=1, horizon=5, score=85.0, win=i % 2))
    for i in range(3):
        rows.append(dict(rec_date="20260707", code=f"B{i}", method="ELITE_SCORE",
                         topk=1, horizon=5, score=85.0, win=1))
    _write_ptl(tmp_path, rows)
    cal = build_calibration_table(str(tmp_path), asof_ymd="20260716")
    b = cal[(cal.method == "ELITE_SCORE") & (cal.horizon == 5)
            & (cal.score_lo <= 85) & (cal.score_hi > 85)].iloc[0]
    assert pd.isna(b["p_fast"]) or b["p_fast"] is None
    assert b["p_calibrated"] == b["p_slow"]


# ═══════════════════════════════════════════════════
# 5. [v33.1] 켈리 알파 사이징 배선 (finalize 재계산)
# ═══════════════════════════════════════════════════
# 7/19 첫 v33 배치 실측: collector 단계 켈리가 알파 주입 전에 돌아
# KELLY_P_SOURCE 전부 레거시 → 알파 게이트 이후 재계산으로 배선.

def test_resize_kelly_noop_when_gate_inactive(tmp_path):
    from kelly_calibrator import resize_kelly_with_alpha
    df = pd.DataFrame([_kelly_row(ALPHA_GATE_ACTIVE=0, 켈리_수량=7)])
    out = resize_kelly_with_alpha(df, str(tmp_path))
    assert out.iloc[0]["켈리_수량"] == 7          # 원본 유지
    assert "KELLY_ENGINE" not in out.columns or out.iloc[0].get("KELLY_ENGINE") != "v33_alpha_resize"


def test_resize_kelly_applies_alpha_p_for_active_route(tmp_path):
    from kelly_calibrator import resize_kelly_with_alpha
    df = pd.DataFrame([_kelly_row(
        ALPHA_GATE_ACTIVE=1, ALPHA_VALIDATED=1, ALPHA_WIN_PROB=0.41,
        ELITE_SCORE=10.0, ROUTE="ATTACK")])
    out = resize_kelly_with_alpha(df, str(tmp_path))
    assert out.iloc[0]["KELLY_P_SOURCE"] == "ALPHA_WIN_PROB"
    assert out.iloc[0]["KELLY_ENGINE"] == "v33_alpha_resize"
    assert out.iloc[0]["켈리_금액(원)"] > 0        # ATTACK은 사이징 유지


def test_resize_kelly_zeroes_inactive_route(tmp_path):
    from kelly_calibrator import resize_kelly_with_alpha
    df = pd.DataFrame([_kelly_row(
        ALPHA_GATE_ACTIVE=1, ALPHA_VALIDATED=1, ALPHA_WIN_PROB=0.41,
        ROUTE="WAIT")])
    out = resize_kelly_with_alpha(df, str(tmp_path))
    # WAIT은 계산은 되지만(진단 컬럼 보존) 베팅 0원
    assert out.iloc[0]["켈리_금액(원)"] == 0
    assert out.iloc[0]["KELLY_FRACTION"] == 0
