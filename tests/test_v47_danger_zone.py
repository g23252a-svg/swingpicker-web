# -*- coding: utf-8 -*-
"""[v47] 위험구간 — 검증된 숏 신호를 공매도 없이 실행 가능한 형태로.

배경
  v46까지는 롱온리만 봤다. 누적 데이터에서 절대수익이 있던 쪽은 숏이었다:
    하위10% 숏      +2.01%/5일  t=+3.82  겹침보정 자본 +35.3%
    하위10%+LT 숏   +2.88%/5일  t=+4.53  자본 +53.9%  양수일 74%
  그러나 개별종목 공매도는 개인 접근이 사실상 막혀 있어 그대로 못 쓴다.
  공매도 없이 가능한 형태 = '보유 중이면 매도 + 후보에서 배제'.

채택 규칙: 알파 당일 하위10% AND 저점추세 당일 하위30%
  유니버스 대비 엣지 -1.59%p (t=-4.43) · 승률 21.5% (유니버스 32.2%)
  IS -2.16 (t=-3.88) / OOS -1.03 (t=-2.31) — 양 구간 유의
  전 보유기간 성립: FWD3 -1.31(t=-3.78) · FWD10 -2.17(t=-7.04)
                    FWD20 -1.77(t=-3.57), 20일 승률 10.8%
  절대 5일 -2.88% vs 유니버스 -1.28% → 회피만으로 +1.6%p 방어
  규모 12.8종목/일 (라이브 292행 실측 7종목, 2.4%)

구성요소 독립 검증
  · 저점추세 하위30% 단독 IS -0.85(t=-3.45) / OOS -0.84(t=-3.56)
  · 알파 순위는 야간 워크포워드 게이트가 매일 재검증

선별력이 방향베팅이 아님을 확인: 롱숏 vs 시장수익 상관 -0.135,
시장 사분위 중 3개에서 양수(시장최악 +1.01% t=4.10 / 시장좋음 +0.73% t=1.89).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alpha_engine as ae
from alpha_engine import apply_alpha_entry_gate, flag_danger_zone


def _batch(n=100, validated=1):
    """알파·저점추세가 함께 나빠지는 구간을 포함한 배치."""
    rows = []
    for i in range(n):
        rows.append({
            "종목코드": f"{i:06d}", "종목명": f"종목{i}",
            "종가": 10000.0, "손절가": 9500.0, "추천매도가1": 11500.0,
            "추천매수가": 10000.0, "거래대금(억원)": 100.0, "POC_GAP": 8.0,
            "ENTRY_RISK_GATE_OK": True, "ALPHA_VALIDATED": validated,
            "MARKET_REGIME": "NEUTRAL", "BUY_NOW_PASS": 1,
            "ALPHA_SCORE": float(i),          # 0..n-1 (낮을수록 나쁨)
            "Low_Trend_PCT": float(i - n / 2),
        })
    return pd.DataFrame(rows)


# ── ① 규칙이 의도한 교집합만 잡는가 ──

def test_danger_zone_is_intersection_not_union():
    out = apply_alpha_entry_gate(_batch(100))
    dz = out["DANGER_ZONE"].astype(int)
    a = out["ALPHA_SCORE"].rank(pct=True)
    lt = out["Low_Trend_PCT"].rank(pct=True)
    expect = ((a <= ae._DANGER_ALPHA_PCTL) & (lt <= ae._DANGER_LT_PCTL))
    assert (dz == expect.astype(int)).all(), "교집합이 아님(합집합/오프바이원)"
    assert 0 < dz.sum() < 20, dz.sum()


def test_alpha_bad_but_lt_fine_is_not_flagged():
    """알파만 나쁘고 자리는 괜찮으면 위험구간이 아니다 — 과잉경고 방지."""
    df = _batch(100)
    df["Low_Trend_PCT"] = 50.0                    # 전원 동일 → 분위 균일
    out = apply_alpha_entry_gate(df)
    assert int(out["DANGER_ZONE"].sum()) == 0


def test_lt_bad_but_alpha_fine_is_not_flagged():
    df = _batch(100)
    df["ALPHA_SCORE"] = 95.0                      # 전원 고알파
    out = apply_alpha_entry_gate(df)
    assert int(out["DANGER_ZONE"].sum()) == 0


def test_worst_stock_is_flagged():
    out = apply_alpha_entry_gate(_batch(100))
    worst = out["ALPHA_SCORE"].idxmin()
    assert int(out.loc[worst, "DANGER_ZONE"]) == 1, "최악 종목이 누락됨"


# ── ② TOP_PICK과 모순 없어야 ──

def test_danger_never_overlaps_top_pick():
    out = apply_alpha_entry_gate(_batch(100))
    assert int(((out["TOP_PICK"] > 0) & (out["DANGER_ZONE"] == 1)).sum()) == 0


# ── ③ 컬럼 계약 — 어떤 경로에서도 존재해야 ──

def test_column_always_present_when_unvalidated():
    out = apply_alpha_entry_gate(_batch(100, validated=0))
    assert "DANGER_ZONE" in out.columns and "DANGER_ZONE_REASON" in out.columns
    assert int(out["DANGER_ZONE"].sum()) == 0, "미검증인데 매도경고 발생"


def test_column_always_present_on_small_batch():
    out = apply_alpha_entry_gate(_batch(5))
    assert "DANGER_ZONE" in out.columns
    assert int(out["DANGER_ZONE"].sum()) == 0, "표본 부족인데 경고 발생"


def test_empty_frame_safe():
    out = flag_danger_zone(pd.DataFrame())
    assert "DANGER_ZONE" in out.columns and len(out) == 0


def test_missing_lt_column_safe():
    df = _batch(100).drop(columns=["Low_Trend_PCT"])
    out = apply_alpha_entry_gate(df)
    assert int(out["DANGER_ZONE"].sum()) == 0, "자리 정보 없이 매도경고 발생"


def test_missing_alpha_column_safe():
    df = _batch(100).drop(columns=["ALPHA_SCORE"])
    out = flag_danger_zone(df)
    assert int(out["DANGER_ZONE"].sum()) == 0


# ── ④ 사유 문구 ──

def test_reason_present_and_quantified():
    out = apply_alpha_entry_gate(_batch(100))
    txt = out.loc[out["DANGER_ZONE"] == 1, "DANGER_ZONE_REASON"].iloc[0]
    assert "위험구간" in txt
    assert "21.5%" in txt, "실측 승률이 사유에 없음"
    assert "정리 검토" in txt
    # 비위험 종목은 사유가 비어 있어야 한다
    assert (out.loc[out["DANGER_ZONE"] == 0, "DANGER_ZONE_REASON"]
            .astype(str).str.strip().eq("").all())


# ── ⑤ 상수 고정 ──

def test_constants_match_validated_values():
    assert ae._DANGER_ALPHA_PCTL == 0.10
    assert ae._DANGER_LT_PCTL == 0.30
    assert ae._DANGER_MIN_SAMPLE == 30
