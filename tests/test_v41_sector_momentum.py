# -*- coding: utf-8 -*-
"""[v41] 섹터 모멘텀 × 알파 틸트 사이징.

근거(2/26~7/3 OOS, v40 기준선 대비 페어드 t):
  · v40 알파틸트 +1.94% → ×섹터모멘텀 +2.28% (+0.34%p, t=3.24, p=0.002)
  · 강건성: 전·후반 양수, 상위3일 제외 t=2.7, 파라미터 둔감(t≈3.2)
  · 대조군: 개별 ret5d 틸트 t=0.48 무효, 섹터모멘텀 단독 t=-0.19 무효
    → '강한 섹터의 고알파'라는 상호작용만이 엣지. 누적 +146% vs +124%.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kelly_calibrator as kc


def _df(sectors, ret5s, alphas):
    n = len(alphas)
    return pd.DataFrame({
        "ALPHA_GATE_ACTIVE": [1] * n,
        "ALPHA_SCORE": list(map(float, alphas)),
        "ALPHA_ENTRY_THRESHOLD": [85.0] * n,
        "켈리_수량": [100] * n,
        "켈리_금액(원)": [1_000_000] * n,
        "KELLY_FRACTION": [0.1] * n,
        "ROUTE": ["ATTACK"] * n,
        "업종_대분류": sectors,
        "ret_5d_%": ret5s,
    })


def test_strong_sector_high_alpha_gets_max_weight():
    out = kc.apply_alpha_proportional_sizing(_df(
        ["강", "약", "강", "약"], [8, -2, 8, -2], [95, 95, 88, 88]))
    top = out.loc[out["ALPHA_SIZE_MULT"].idxmax()]
    assert top["업종_대분류"] == "강" and top["ALPHA_SCORE"] == 95


def test_same_alpha_strong_sector_beats_weak():
    out = kc.apply_alpha_proportional_sizing(_df(
        ["강", "약", "강", "약"], [8, -2, 8, -2], [92, 92, 92, 92]))
    strong = out[out["업종_대분류"] == "강"]["ALPHA_SIZE_MULT"].mean()
    weak = out[out["업종_대분류"] == "약"]["ALPHA_SIZE_MULT"].mean()
    assert strong > weak
    assert out["SECTOR_MOM_FACTOR"].max() <= 1.5 + 1e-9
    assert out["SECTOR_MOM_FACTOR"].min() >= 0.5 - 1e-9


def test_capital_preserved_with_sector_factor():
    df = _df(["A", "B", "C", "A"], [5, 0, -5, 5], [90, 88, 93, 86])
    out = kc.apply_alpha_proportional_sizing(df)
    assert abs(out["켈리_금액(원)"].sum() - df["켈리_금액(원)"].sum()) < 1.0


def test_neutral_when_sector_columns_missing():
    # 섹터/수익률 컬럼 없으면 v40 알파틸트만 (중립 팩터).
    df = _df(["A", "B"], [1, 1], [90, 95]).drop(columns=["업종_대분류", "ret_5d_%"])
    out = kc.apply_alpha_proportional_sizing(df)
    assert "SECTOR_MOM_FACTOR" not in out.columns
    assert out.iloc[1]["ALPHA_SIZE_MULT"] > out.iloc[0]["ALPHA_SIZE_MULT"]  # 알파틸트는 유지
