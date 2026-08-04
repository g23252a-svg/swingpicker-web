# -*- coding: utf-8 -*-
"""[v51] 리스크 게이트에서 PASS_EBS 거부권 제거.

ENTRY_RISK_GATE_OK은 7조건 AND였고 v46 퍼널에서 22개→8개로 가장 크게 자르는데
컴포넌트별 검증이 한 번도 없었다. 풀 내 실제로 binding 하는 것은
EBS(통과 69.5%)·RR(66.5%)·POC(82.4%)·진입갭(94.1%)뿐이고
stop/tp1/거래대금은 100% 통과라 판별력이 없다.

■ PASS_EBS 한계기여 (나머지 6조건 통과 풀 내, 통과 vs 탈락, 일별평균 페어드)
    차이 -0.21%p  t=-0.36  p=0.72   (IS -0.62 / OOS +0.04)
  4개 렌즈 적대적 검증 전부 이 '예측력 없음'을 반증하지 못했다 (0/4 반증):
    · 풀정의: 알파상위 5~30% × 저점추세하한 .20~.50 = 16셀에서 p<0.05 0/16,
      표본 4.6배 유니버스 전체 -0.23%p t=-1.42 (IS -0.24 / OOS -0.22 동시 음수)
    · 시간: 월 3/4 음수 · |t|>2 0개월 · 주별 양 8/15 · 에피소드 -0.09(t=-0.13)
      종목·날짜 클러스터 부트 CI [-0.96,+0.75] / [-1.48,+0.84]
    · 교란: 13변수 층화 13/13 p>0.26 · PSM 6종 · 일FE회귀 전부 |t|<1.1
      (기준값의 '탈락군 승률이 더 높다'는 오히려 반증됨 — 일별페어드 -0.3pp
       t=-0.07, 매칭 후 통과군 우위 20/20 → 역신호가 아니라 무용지물)
    · 보유기간: FWD3/5/10/20 × ALL/IS/OOS = 12셀 전부 p>0.36 (최대 |t|=0.92)

■ 제거 영향 (짝맞춘 69일 포트폴리오)
    통과 898→1,495행 (일평균 11.8→19.7, +66%)
    수익차 +0.05%p (t=+0.49, p=0.63) · NW5 t=+0.64 · 부트 CI[-0.08,+0.22]
    4지평·월별 4버킷 전부 |t|<0.9
    위험 동등~미세개선: MaxDD -77.7(기존 -82.0) · 스톱적중 32.8%(33.4%)
                        sd 4.47(4.52) · 최악5분위 -6.24(-6.33)

■ 구조적 근거
  EBS는 5개 이진체크 합(저점추세>0 · 거래품질 · MACD기울기 · RSI대역 ·
  TTM/BB확장)이고 그중 둘이 이미 검증된 저점추세 게이트(v46 t=+5.29)와 중복이다.

■ 유지 결정한 컴포넌트
  RR>=1.0 : 제거하면 OOS -0.26%p t=-3.10 (MaxDD -95.7) → 유지
  POC<=20 : 기준값 유의성이 4/4 반증됐으나 제거도 IS -0.37%p t=-2.79 → 유지
  묶어서 제거하면 중립성이 상쇄된다(-EBS-POC t=-1.63, 셋다 에피소드 OOS
  t=-2.58 p=0.014) → EBS 단독 제거만 유효.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scoring_engine as se


def _row(**ov):
    """게이트 6조건을 모두 통과하는 기본 행."""
    r = {
        "종목코드": "000001", "종목명": "테스트",
        "종가": 10000.0, "손절가": 9200.0,
        "추천매도가1": 11500.0, "추천매수가": 10000.0,
        "거래대금(억원)": 200.0, "POC_GAP": 5.0,
        "PASS_EBS": 1, "EBS": 5,
        "ROUTE": "ATTACK", "MARKET_REGIME": "NEUTRAL",
        # compute_elite_score가 요구하는 상류 3축
        "STRUCT_SCORE": 80.0, "TIMING_SCORE": 80.0, "AI_SCORE": 80.0,
        "ML_SCORE": 80.0,
    }
    r.update(ov)
    return r


def _gate(rows):
    df = pd.DataFrame([_row(**r) for r in rows])
    out, _ = se.compute_elite_score(df)
    return out, out["ENTRY_RISK_GATE_OK"].astype(bool)


# ── ① EBS 거부권이 제거됐는가 ──

def test_ebs_fail_no_longer_blocks_gate():
    """핵심: PASS_EBS=0이어도 나머지 조건이 좋으면 게이트를 통과해야 한다."""
    out, g = _gate([{"PASS_EBS": 0, "EBS": 1}])
    assert bool(g.iloc[0]), "PASS_EBS 거부권이 여전히 살아있음"


def test_ebs_pass_and_fail_treated_identically():
    """EBS만 다른 두 행의 게이트 판정이 같아야 한다."""
    out, g = _gate([{"종목코드": "000001", "PASS_EBS": 1, "EBS": 5},
                    {"종목코드": "000002", "PASS_EBS": 0, "EBS": 0}])
    assert bool(g.iloc[0]) and bool(g.iloc[1])


def test_ebs_column_still_produced():
    """게이트에서만 뺀다 — 컬럼은 표시·알파 피처로 계속 산출돼야 한다."""
    out, _ = _gate([{"PASS_EBS": 0}])
    assert "PASS_EBS" in out.columns and "EBS" in out.columns


def test_missing_ebs_column_is_safe():
    df = pd.DataFrame([_row()]).drop(columns=["PASS_EBS", "EBS"])
    out, _ = se.compute_elite_score(df)
    assert bool(out["ENTRY_RISK_GATE_OK"].iloc[0])


# ── ② 나머지 조건은 그대로 살아있는가 (과잉 완화 방지) ──

def test_rr_gate_retained():
    """RR<1.0은 계속 차단 — 제거 시 OOS -0.26%p(t=-3.10)로 악화."""
    # 목표가를 종가 바로 위로 → reward 작고 risk 크므로 RR<1
    out, g = _gate([{"추천매도가1": 10100.0, "손절가": 9000.0}])
    assert float(out["RR_NOW_TP1"].iloc[0]) < 1.0
    assert not bool(g.iloc[0]), "RR 게이트가 사라졌다"


def test_poc_gate_retained():
    """POC_GAP>20은 계속 차단 — 제거 시 IS -0.37%p(t=-2.79)."""
    out, g = _gate([{"POC_GAP": 35.0}])
    assert not bool(g.iloc[0]), "POC 게이트가 사라졌다"


def test_turnover_gate_retained():
    out, g = _gate([{"거래대금(억원)": 10.0}])
    assert not bool(g.iloc[0])


def test_stop_and_tp1_sanity_retained():
    out, g = _gate([{"손절가": 10500.0}])          # 손절가 > 종가
    assert not bool(g.iloc[0])
    out2, g2 = _gate([{"추천매도가1": 9000.0}])     # 목표가 < 종가
    assert not bool(g2.iloc[0])


def test_entry_gap_gate_retained():
    out, g = _gate([{"추천매수가": 9000.0}])        # 진입갭 11% > 5%
    assert float(out["ENTRY_GAP_PCT"].iloc[0]) > 5.0
    assert not bool(g.iloc[0])


# ── ③ 팬텀 수정이 들어가지 않았는가 ──

def test_no_rr_missing_bypass_shipped():
    """RR 결측 우회는 팬텀 버그 대응이었으므로 코드에 남아선 안 된다.

    2026-04-01~09의 '추천 0건'은 그 시점 CSV에 RR_NOW_TP1 컬럼이 없었던
    리서치 패널 아티팩트다(04-10 도입). 프로덕션은 RR을 매 배치 가격에서
    재계산하므로 결측이 될 수 없다 — 우회 경로는 게이트만 약화시킨다.
    """
    src = (ROOT / "scoring_engine.py").read_text(encoding="utf-8")
    assert "_RR_MISSING_BYPASS_RATE" not in src
    assert "RR_GATE_DEGRADED" not in src
    out, _ = _gate([{}])
    assert "RR_GATE_DEGRADED" not in out.columns


def test_rr_is_recomputed_not_read_from_input():
    """입력 RR이 쓰레기여도 가격에서 재계산된다(팬텀 버그의 근거)."""
    out, g = _gate([{"RR_NOW_TP1": np.nan}])
    assert float(out["RR_NOW_TP1"].iloc[0]) > 1.0
    assert bool(g.iloc[0])


# ── ④ 근거 기록 (수치 없는 제거 방지) ──

def test_removal_rationale_recorded_with_numbers():
    src = (ROOT / "scoring_engine.py").read_text(encoding="utf-8")
    assert "t=-0.36" in src, "한계기여 통계량 누락"
    assert "0/16" in src, "풀정의 렌즈 근거 누락"
    assert "t=+0.49" in src, "퍼널 중립 근거 누락"
    assert "탐지 불가" in src, "검정력 한계 미기록"
    assert "단독 제거만 유효" in src, "묶어 제거 금지 경고 누락"
