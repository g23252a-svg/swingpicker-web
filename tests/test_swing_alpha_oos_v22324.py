# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from swing_alpha_oos_v22324 import select_current_swing_candidates


def _row(name, code, timing, final, struct, rr1, tp2, tp3, foreign, inst, poc=10, vwap=5, tp2_prob=50):
    return {
        "종목명": name,
        "종목코드": code,
        "TIMING_SCORE": timing,
        "FINAL_SCORE": final,
        "STRUCT_SCORE": struct,
        "RR_NOW_TP1": rr1,
        "추천매수가": 100.0,
        "종가": 100.0,
        "손절가": 90.0,
        "추천매도가2": tp2,
        "추천매도가3": tp3,
        "TP2_PROB": tp2_prob,
        "외인순매수": foreign,
        "기관순매수": inst,
        "메이저순매수": foreign + inst,
        "개인순매수": -(foreign + inst),
        "POC_GAP": poc,
        "VWAP_GAP": vwap,
        "ENTRY_RISK_LEVEL": "GREEN",
    }


def test_swing_alpha_selects_high_quality_swing_and_blocks_weak_high_rr():
    df = pd.DataFrame([
        _row("GoodSwing", "000001", 85, 70, 65, 1.3, 125, 140, 1000, 100),
        _row("WeakHighRR", "000002", 58, 45, 49, 3.0, 130, 150, 1000, -100),
        _row("NoSwingPotential", "000003", 80, 70, 65, 1.21, 112, 118, 1000, 100, tp2_prob=20),
    ])
    profile = {
        "timing_min": 55.0,
        "final_min": 55.0,
        "struct_min": 50.0,
        "rr_min": 1.2,
        "poc_max": 90.0,
        "vwap_max": None,
        "oos_pass": True,
    }

    out = select_current_swing_candidates(df, profile, topk=3)
    pick_names = set(out["picks"]["종목명"].tolist())
    near_names = set(out["near"]["종목명"].tolist())

    assert "GoodSwing" in pick_names
    assert "WeakHighRR" not in pick_names
    assert "WeakHighRR" not in near_names
    assert "NoSwingPotential" in near_names
