# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

import components.tab_stocks as ts


def _summary():
    return pd.DataFrame([
        {
            "METHOD": "FINAL_SCORE",
            "TOPK": 3,
            "H(영업일)": 5,
            "TOTAL_N": 165,
            "WIN_RATE_%": 76.4,
            "AVG_RET_%": 16.36,
            "ALPHA_%": 12.17,
            "HIT_5%_%": 66.1,
        }
    ])


def _row(code, name, **kw):
    base = {
        "종목코드": code,
        "종목명": name,
        "ROUTE": "WAIT",
        "TOP_PICK": 0,
        "BUY_NOW_ELIGIBLE": 0,
        "FINAL_SCORE": 82,
        "DISPLAY_SCORE": 82,
        "ELITE_SCORE": 70,
        "TIMING_SCORE": 65,
        "STRUCT_SCORE": 70,
        "AI_SCORE": 70,
        "BALANCE_SCORE": 70,
        "AXIS_GAP": 25,
        "RR_NOW_TP1": 1.5,
        "GAP_PCT": 0,
        "VWAP_GAP": 5,
        "POC_GAP": 10,
        "ENTRY_RISK_LEVEL": "",
        "ENTRY_EDGE_LEVEL": "GREEN",
        "ROUTE_REASON": "",
        "MACRO_RISK": "",
        "ret_10d_%": 2,
        "ret_20d_%": 5,
        "KELLY_수량": 1,
        "IS_REAL_HOLDING": 0,
    }
    base.update(kw)
    return base


def test_quality_veto_blocks_proicheon_like_false_positive(monkeypatch):
    monkeypatch.setattr(ts, "_actual_holding_mask", lambda work, holding_codes=None: pd.Series(False, index=work.index))
    df = pd.DataFrame([
        _row(
            "321260",
            "프로이천",
            FINAL_SCORE=80.9,
            DISPLAY_SCORE=0,
            TIMING_SCORE=35.9,
            STRUCT_SCORE=100,
            AI_SCORE=84.6,
            BALANCE_SCORE=19.9,
            AXIS_GAP=64.1,
            RR_NOW_TP1=3.23,
            ROUTE_REASON="캐리 재계산 실패: legacy snapshot",
            ret_10d_=-11.71,
            **{"ret_10d_%": -11.71, "ret_20d_%": 50.2, "KELLY_수량": 0},
        ),
        _row("111111", "정상후보", FINAL_SCORE=82, DISPLAY_SCORE=82, TIMING_SCORE=65),
    ])
    r = ts._build_winrate_action_candidates(df, summary_df=_summary())
    out = r["candidates"]

    assert list(out["종목명"]) == ["정상후보"]
    rejected_names = set(r["rejected"]["종목명"])
    assert "프로이천" in rejected_names
    veto = r["rejected"].set_index("종목명").loc["프로이천", "VALIDATED_VETO_REASON"]
    assert "DISPLAY_SCORE<=0" in veto
    assert "TIMING<50" in veto
    assert "AXIS_GAP>45" in veto
    assert "legacy/carry" in veto


def test_quality_veto_blocks_zero_kelly(monkeypatch):
    monkeypatch.setattr(ts, "_actual_holding_mask", lambda work, holding_codes=None: pd.Series(False, index=work.index))
    df = pd.DataFrame([
        _row("111111", "제로켈리", **{"KELLY_수량": 0}),
        _row("222222", "정상후보", **{"KELLY_수량": 3}),
    ])
    r = ts._build_winrate_action_candidates(df, summary_df=_summary())
    assert list(r["candidates"]["종목명"]) == ["정상후보"]


def test_veto_reason_helper():
    reasons = ts._validated_action_veto_reasons(pd.Series({
        "DISPLAY_SCORE": 0,
        "TIMING_SCORE": 35,
        "AXIS_GAP": 60,
        "BALANCE_SCORE": 10,
        "ROUTE_REASON": "캐리 재계산 실패: legacy snapshot",
        "ret_10d_%": -5,
        "ret_20d_%": 40,
        "KELLY_수량": 0,
    }))
    assert "DISPLAY_SCORE<=0" in reasons
    assert "KELLY 수량 0" in reasons
    assert "TIMING<50" in reasons
    assert "급등 후 식는 패턴" in reasons
