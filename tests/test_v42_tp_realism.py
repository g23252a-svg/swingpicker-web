# -*- coding: utf-8 -*-
"""[v42] TP1 현실화 — 3.5×ATR 상한 + 실측 도달률 확률.

배경(카카오 7/27, RR 7.12 사례): 반등으로 근거리 후보(BB·20일고점)가 원천
제외되고, '최소 RR 1.5' 필터가 ATR 2배(+9.9%)·60일 고점마저 잘라, 폭락 전
매물대(POC 58,400, +57.6%)만 남아 TP1 등극 → RR 7.12 착시.

실측(과거 배치 16,664건, 7일내 고가 도달):
  0~1.5×ATR 46% / 1.5~2.5 23% / 2.5~3.5 12% / 3.5~5 5% / 5+ 2%
  → 기존 TP1의 47%가 도달률 5% 이하 비현실 구간인데 확률 45~80으로 표시.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stop_logic as sl


def _downtrend_ohlcv(n=130, start=58000, end=36000, seed=7):
    np.random.seed(seed)
    close = pd.Series(np.linspace(start, end, n) + np.random.randn(n) * 500).clip(lower=30000)
    return pd.DataFrame({"종가": close, "고가": close * 1.018, "저가": close * 0.982})


def _kakao_case():
    # 하락추세 + 넓은 손절 + 먼 POC — RR 필터 역선택 재현 조건.
    return sl.compute_realistic_targets(
        _downtrend_ohlcv(), entry=37050.0, stop=34050.0,
        poc_p=58400.0, res_ratio=0.3, res_ratio_near=0.15)


def test_tp1_capped_at_realistic_atr_distance():
    r = _kakao_case()
    ohlcv = _downtrend_ohlcv()
    atr = sl._atr_from_ohlcv(ohlcv, 14)
    # TP1은 3.5×ATR 이내 (호가 반올림 여유 2%).
    assert (r["TP1"] - 37050.0) <= atr * sl._TP1_MAX_ATR_MULT * 1.02
    # 먼 POC(58,400)가 TP1이 되면 안 됨.
    assert r["TP1"] < 50000


def test_far_poc_demoted_with_honest_prob():
    r = _kakao_case()
    # POC는 TP2/TP3로 강등되고, 실측 도달률(5+ATR → 2%)로 정직 표기.
    far = [k for k in ("TP2", "TP3") if r.get(k) == 58400]
    assert far, "먼 POC가 상위 TP로 강등되지 않음"
    assert r[f"{far[0]}_PROB"] <= 5


def test_prob_reflects_empirical_reach():
    assert sl._empirical_reach_prob(1.0) == 46
    assert sl._empirical_reach_prob(2.2) == 23
    assert sl._empirical_reach_prob(3.4) == 12
    assert sl._empirical_reach_prob(4.9) == 5
    assert sl._empirical_reach_prob(11.6) == 2   # 카카오 POC 거리


def test_near_candidates_survive_rr_filter():
    # 넓은 손절(-8%)로 RR 1.5 문턱이 +12%가 돼도 근거리 후보가 남아야 함.
    r = _kakao_case()
    assert r["TP1_PCT"] < 20.0   # 근거리 현실 목표
    assert r["N_CANDIDATES"] >= 2


def test_synthetic_atr_target_when_all_far():
    # 후보가 전부 원거리인 극단 케이스 → ATR 2배 합성 TP1 폴백.
    flat = pd.DataFrame({"종가": [10000.0] * 60,
                         "고가": [10150.0] * 60, "저가": [9850.0] * 60})
    r = sl.compute_realistic_targets(flat, entry=10000.0, stop=9200.0,
                                     poc_p=16000.0, res_ratio_near=0.2)
    atr = sl._atr_from_ohlcv(flat, 14)
    assert (r["TP1"] - 10000.0) <= atr * sl._TP1_MAX_ATR_MULT * 1.02


def test_normal_uptrend_unchanged_behavior():
    # 정상 상승 케이스: 근거리 스윙 고점이 있으면 기존처럼 동작하고 상한 내.
    np.random.seed(3)
    close = pd.Series(np.linspace(9000, 10000, 130) + np.random.randn(130) * 80)
    ohlcv = pd.DataFrame({"종가": close, "고가": close * 1.02, "저가": close * 0.98})
    r = sl.compute_realistic_targets(ohlcv, entry=10000.0, stop=9500.0)
    atr = sl._atr_from_ohlcv(ohlcv, 14)
    assert r["TP1"] > 10000
    assert (r["TP1"] - 10000.0) <= atr * sl._TP1_MAX_ATR_MULT * 1.02
