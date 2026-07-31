# -*- coding: utf-8 -*-
"""[v46] 저점추세 게이트: 절대 문턱 → 당일 횡단면 분위.

누적 백데이터 정밀 분석 근거
  패널: recommend 스냅샷 105일 × OHLCV 캐시 합집합(1,709종목, 생존편향 제거)
  수익: 진입 t+1 시가 · -8% 장중 스톱 · t+5 종가 (프로덕션 실행과 동일)
  검정: 일별 평균의 페어드 t (횡단면 상관 보정)

  현행 LT≥0        엣지 +0.16%p  t=+1.30           ← 유의하지 않음
  신규 LT 분위>30%   엣지 +0.32%p  t=+5.29 (n=45,389)
                    IS t=+3.30 / OOS t=+4.29 (양 구간 재현)
                    FWD3 +4.27 · FWD10 +4.48 · FWD20 +1.95
                    (현행은 FWD20에서 -0.97로 역전)

두 규칙의 차집합이 결정적
  현행만 통과(신규가 탈락)  n=2,077  평균 -1.95%  승률 26.9%  엣지 -0.57  t=-2.19
  신규만 통과(현행이 탈락)  n=9,940  평균 -0.57%  승률 33.3%  엣지 +0.43  t=+2.53
→ 절대 문턱은 손실군을 들여보내면서 수익군을 버리고 있었다.

부수 효과: 유지율 51.5% → 70.1%. 후보가 덜 사라진다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alpha_engine as ae
from alpha_engine import apply_alpha_entry_gate
from services.recommendation_quality import _reasons


def _row(**ov):
    r = {
        "종목코드": "000001", "종가": 10000.0, "손절가": 9500.0,
        "추천매도가1": 11500.0, "추천매수가": 10000.0,
        "거래대금(억원)": 100.0, "POC_GAP": 8.0, "PASS_EBS": 1,
        "ENTRY_RISK_GATE_OK": True, "ALPHA_VALIDATED": 1,
        "ALPHA_SCORE": 95.0, "MARKET_REGIME": "NEUTRAL", "BUY_NOW_PASS": 1,
    }
    r.update(ov)
    return r


def _batch(lt_values):
    """실제 배치 크기(분위 경로 작동)의 DataFrame."""
    return pd.DataFrame([
        _row(종목코드=f"{i:06d}", Low_Trend_PCT=v) for i, v in enumerate(lt_values)
    ])


# ── ① 분위 경로가 작동하는가 ──

def test_percentile_rule_engages_on_real_batch():
    out = apply_alpha_entry_gate(_batch(list(range(-50, 50))))     # n=100
    assert out["ALPHA_LT_RULE"].iloc[0] == f"pctl>{ae._LT_PCTL_FLOOR:.2f}"
    assert out["LOW_TREND_PCTL"].notna().all()
    # 하위 30%만 탈락 — 유지율이 대략 70%
    keep = out["ALPHA_LT_OK"].mean()
    assert 0.65 <= keep <= 0.75, keep


def test_bottom_tercile_blocked_top_passes():
    out = apply_alpha_entry_gate(_batch(list(range(100))))          # LT 0..99
    lo = out[out["Low_Trend_PCT"] < 25]
    hi = out[out["Low_Trend_PCT"] > 75]
    assert (lo["ALPHA_LT_OK"] == 0).all(), "하위권이 통과됨"
    assert (hi["ALPHA_LT_OK"] == 1).all(), "상위권이 차단됨"
    assert (hi["TOP_PICK"] == 1).all()


def test_all_positive_batch_still_filters_bottom():
    """현행 규칙이 놓치던 핵심: LT 전부 양수여도 하위권은 걸러야 한다."""
    out = apply_alpha_entry_gate(_batch([float(v) for v in range(1, 101)]))
    assert (out["ALPHA_LT_OK"] == 0).any(), "전부 양수면 무조건 통과 = 구 규칙 잔존"
    assert int(out.loc[out["Low_Trend_PCT"] <= 20, "ALPHA_LT_OK"].max()) == 0


def test_all_negative_batch_still_admits_top():
    """반대편: LT 전부 음수인 폭락일에도 상대 상위군은 통과해야 한다.

    현행 LT≥0은 이런 날 후보를 전멸시켰다(중앙일 61.4%가 LT<0).
    """
    out = apply_alpha_entry_gate(_batch([float(-v) for v in range(1, 101)]))
    assert int(out["ALPHA_LT_OK"].sum()) > 0, "폭락일 전멸 — 구 규칙 잔존"
    # 가장 덜 나쁜 쪽(LT=-1)이 통과
    assert int(out.loc[out["Low_Trend_PCT"] == -1.0, "ALPHA_LT_OK"].iloc[0]) == 1


# ── ② 폴백/안전성 ──

def test_small_batch_falls_back_to_absolute_rule():
    out = apply_alpha_entry_gate(_batch([-5.0, 3.0]))               # n=2 < 30
    assert out["ALPHA_LT_RULE"].iloc[0] == "abs>=0"
    assert int(out.loc[out["Low_Trend_PCT"] == -5.0, "ALPHA_LT_OK"].iloc[0]) == 0
    assert int(out.loc[out["Low_Trend_PCT"] == 3.0, "ALPHA_LT_OK"].iloc[0]) == 1


def test_missing_lt_passes_no_data_lockout():
    vals = [float(v) for v in range(40)]
    df = _batch(vals)
    df.loc[0, "Low_Trend_PCT"] = np.nan          # 결측 1건
    out = apply_alpha_entry_gate(df)
    assert int(out.loc[0, "ALPHA_LT_OK"]) == 1, "결측이 차단으로 번짐"


def test_missing_column_entirely_passes():
    df = pd.DataFrame([_row(종목코드=f"{i:06d}") for i in range(40)])
    out = apply_alpha_entry_gate(df)             # Low_Trend_PCT 컬럼 자체가 없음
    assert (out["ALPHA_LT_OK"] == 1).all()


def test_gate_inactive_when_unvalidated():
    df = _batch([float(-v) for v in range(1, 41)])
    df["ALPHA_VALIDATED"] = 0
    df["TOP_PICK"] = 1
    out = apply_alpha_entry_gate(df)
    assert (out["ALPHA_GATE_ACTIVE"] == 0).all()
    assert (out["TOP_PICK"] == 1).all()          # 미검증 시 보존


# ── ③ 사유 문구 ──

def test_reason_reports_percentile():
    df = apply_alpha_entry_gate(_batch([float(v) for v in range(-50, 50)]))
    reasons = _reasons(df, pd.Series(False, index=df.index))
    blocked = df.index[df["ALPHA_LT_OK"] == 0]
    txt = reasons.loc[blocked[0]]
    assert "저점추세 당일 하위" in txt, txt


def test_reason_falls_back_without_percentile():
    df = apply_alpha_entry_gate(_batch([-5.0, 3.0]))     # 폴백 → 분위 결측
    reasons = _reasons(df, pd.Series(False, index=df.index))
    assert "저점추세 하락" in reasons.iloc[0]


# ── ④ 상수 고정 (근거 없는 변경 방지) ──

def test_constants_match_validated_values():
    assert ae._LT_PCTL_FLOOR == 0.30, "검증된 임계(하위30%)와 불일치"
    assert ae._LT_PCTL_MIN_SAMPLE == 30
