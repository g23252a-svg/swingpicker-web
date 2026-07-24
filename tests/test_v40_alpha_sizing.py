# -*- coding: utf-8 -*-
"""[v40] 대규모 수익성 패치 — 알파 비례 사이징 + NEUTRAL 임계값 85.

근거(2/26~7/3 워크포워드 OOS, 게이트 통과 일별 포트폴리오, 페어드 t):
  · 동일가중 +1.35% → 알파 비례(상한 [0.5,2.0]) +1.70%  (+0.35%p, t=2.86, p=0.006)
  · 강건성: 전·후반 양수, 상위3일 제외 t=2.20, 누적 +114.8% vs +86.1%.
  · 임계값 알파≥80(+1.35%/59%) → ≥85(+1.62%/61%, t=2.08).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kelly_calibrator as kc
from alpha_engine import ALPHA_GATE_THRESHOLD, ALPHA_GATE_DEFAULT_THRESHOLD


def _gate_df(alphas, thr=85.0, qty=100):
    n = len(alphas)
    return pd.DataFrame({
        "ALPHA_GATE_ACTIVE": [1] * n,
        "ALPHA_SCORE": list(map(float, alphas)),
        "ALPHA_ENTRY_THRESHOLD": [thr] * n,
        "켈리_수량": [qty] * n,
        "켈리_금액(원)": [1_000_000] * n,
        "추천금액(만원)": [100] * n,
        "KELLY_FRACTION": [0.1] * n,
        "ROUTE": ["ATTACK"] * n,
    })


# ── 임계값 ──

def test_neutral_threshold_raised_to_85():
    assert ALPHA_GATE_THRESHOLD["NEUTRAL"] == 85.0
    assert ALPHA_GATE_DEFAULT_THRESHOLD == 85.0
    # UP/DOWN은 유지 (검증 범위 밖).
    assert ALPHA_GATE_THRESHOLD["UP"] == 70.0
    assert ALPHA_GATE_THRESHOLD["DOWN"] == 90.0


# ── 알파 비례 사이징 ──

def test_higher_alpha_gets_larger_size():
    out = kc.apply_alpha_proportional_sizing(_gate_df([85, 90, 95, 99]))
    m = out["ALPHA_SIZE_MULT"].tolist()
    # 단조 증가 — 알파 높을수록 큰 배수.
    assert m[0] < m[1] < m[2] < m[3]
    assert out.iloc[-1]["켈리_수량"] > out.iloc[0]["켈리_수량"]


def test_total_capital_preserved():
    df = _gate_df([85, 88, 92, 96, 99])
    out = kc.apply_alpha_proportional_sizing(df)
    # 총 배분 금액 보존 (재정규화).
    assert abs(out["켈리_금액(원)"].sum() - df["켈리_금액(원)"].sum()) < 1.0


def test_concentration_capped():
    # 극단 알파 격차에도 배수는 대략 [0.5,2.0] 범위(총자본 보존 재정규화로
    # 소폭 초과 가능 — 대부분 문턱 근접 + 1종목 극단 시 상단 ~2.3배).
    out = kc.apply_alpha_proportional_sizing(_gate_df([85, 85, 85, 100]))
    assert out["ALPHA_SIZE_MULT"].max() <= 2.4   # 하드캡 2.0 + 재정규화 여유
    assert out["ALPHA_SIZE_MULT"].min() >= 0.4


def test_noop_when_gate_inactive():
    df = _gate_df([90, 95])
    df["ALPHA_GATE_ACTIVE"] = 0
    out = kc.apply_alpha_proportional_sizing(df)
    assert (out["켈리_수량"] == 100).all()


def test_noop_single_bet_row():
    # 1종목이면 틸트 의미 없음 → 무변경.
    out = kc.apply_alpha_proportional_sizing(_gate_df([95]))
    assert out.iloc[0]["켈리_수량"] == 100
    assert out.iloc[0]["ALPHA_SIZE_MULT"] == 1.0


def test_engine_tag_set_when_applied():
    out = kc.apply_alpha_proportional_sizing(_gate_df([88, 95]))
    assert out.iloc[0]["KELLY_ENGINE"] == "v40_alpha_tilt"
