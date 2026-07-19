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

def test_fast_tp_price_is_entry_plus_2pct():
    out = add_exit_plan_columns(pd.DataFrame([dict(추천매수가=10000.0, ROUTE="WAIT")]))
    assert out.iloc[0]["FAST_TP_PRICE"] == 10200.0
    assert FAST_TP_PCT == 2.0 and FAST_TP_DAY == 2


def test_fast_tp_in_note():
    out = add_exit_plan_columns(pd.DataFrame([dict(추천매수가=10000.0, ROUTE="WAIT")]))
    note = out.iloc[0]["EXIT_PLAN_NOTE"]
    assert "절반 익절" in note and "D+2" in note


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
