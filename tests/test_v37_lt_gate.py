# -*- coding: utf-8 -*-
"""[v37] 알파 게이트 저점추세 결합 검증.

근거(2/26~7/3 워크포워드 OOS, 일별 페어드 t-검정):
  · 알파≥80 & Low_Trend_PCT≥0 : n=2,224  +1.97%  승률 48%
  · 알파≥80 & Low_Trend_PCT<0 : n=2,006  -0.12%  승률 45%   (p=0.015)
→ 게이트 풀 66→35개/일로 정제. 치료된 ATTACK 정의(LT≥0)와 일관.
결측은 통과 — 데이터 이슈가 전면 차단으로 번지지 않게 한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_engine import apply_alpha_entry_gate
from services.recommendation_quality import _reasons


def _row(**overrides):
    row = {
        "종목코드": "000001",
        "종가": 10000.0,
        "손절가": 9500.0,
        "추천매도가1": 11500.0,
        "추천매수가": 10000.0,
        "거래대금(억원)": 100.0,
        "POC_GAP": 8.0,
        "PASS_EBS": 1,
        "ENTRY_RISK_GATE_OK": True,
        "ALPHA_VALIDATED": 1,
        "ALPHA_SCORE": 95.0,
        "MARKET_REGIME": "NEUTRAL",
        "BUY_NOW_PASS": 1,
    }
    row.update(overrides)
    return row


def _df(rows):
    return pd.DataFrame([_row(**r) for r in rows])


def test_lt_negative_blocks_even_with_high_alpha():
    out = apply_alpha_entry_gate(_df([{"Low_Trend_PCT": -5.0}]))
    assert out.iloc[0]["ALPHA_LT_OK"] == 0
    assert out.iloc[0]["ALPHA_ENTRY_OK"] == 0
    assert out.iloc[0]["TOP_PICK"] == 0


def test_lt_positive_passes():
    out = apply_alpha_entry_gate(_df([{"Low_Trend_PCT": 3.0}]))
    assert out.iloc[0]["ALPHA_LT_OK"] == 1
    assert out.iloc[0]["TOP_PICK"] == 1


def test_lt_zero_passes_boundary():
    out = apply_alpha_entry_gate(_df([{"Low_Trend_PCT": 0.0}]))
    assert out.iloc[0]["TOP_PICK"] == 1


def test_lt_missing_passes_no_data_lockout():
    # 컬럼 자체가 없는 경우 (구버전 파이프라인)
    out = apply_alpha_entry_gate(_df([{}]))
    assert out.iloc[0]["ALPHA_LT_OK"] == 1
    assert out.iloc[0]["TOP_PICK"] == 1
    # 컬럼은 있는데 결측인 경우
    out2 = apply_alpha_entry_gate(_df([{"Low_Trend_PCT": np.nan}]))
    assert out2.iloc[0]["ALPHA_LT_OK"] == 1
    assert out2.iloc[0]["TOP_PICK"] == 1


def test_lt_gate_only_when_validated():
    # 미검증이면 게이트 자체가 비활성 — LT<0이어도 TOP_PICK 보존.
    out = apply_alpha_entry_gate(
        _df([{"ALPHA_VALIDATED": 0, "TOP_PICK": 1, "Low_Trend_PCT": -9.0}])
    )
    assert out.iloc[0]["ALPHA_GATE_ACTIVE"] == 0
    assert out.iloc[0]["TOP_PICK"] == 1


def test_reason_mentions_lt_when_alpha_passed_but_lt_failed():
    df = apply_alpha_entry_gate(_df([{"Low_Trend_PCT": -5.0}]))
    reasons = _reasons(df, pd.Series([False], index=df.index))
    assert "저점추세 하락" in reasons.iloc[0]


def test_reason_prefers_alpha_shortfall_over_lt():
    # 알파 미달이 1차 사유 — LT 사유는 알파 통과자에게만.
    df = apply_alpha_entry_gate(_df([{"ALPHA_SCORE": 50.0, "Low_Trend_PCT": -5.0}]))
    reasons = _reasons(df, pd.Series([False], index=df.index))
    assert "알파 50점" in reasons.iloc[0]
    assert "저점추세" not in reasons.iloc[0]
