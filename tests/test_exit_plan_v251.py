# -*- coding: utf-8 -*-
"""test_exit_plan_v251.py — v25.1 Exit Discipline Layer 테스트."""
import numpy as np
import pandas as pd
import pytest

from exit_plan import add_exit_plan_columns, exit_summary, EXIT_OUTPUT_COLS
from collector_config import ExitPlanConfig, CollectorConfig


def _df(**kw):
    base = dict(
        종목명=["A", "B", "C", "D"],
        추천매수가=[10000.0, 20000.0, 50000.0, np.nan],
        손절가=[9000.0, 18000.0, 45000.0, 5000.0],
        ROUTE=["WAIT", "CARRY", "ATTACK", "NEUTRAL"],
        TOP_PICK=[1, 0, 0, 0],
        BUY_NOW_ELIGIBLE=[1, 0, 0, 0],
    )
    base.update(kw)
    return pd.DataFrame(base)


def test_columns_and_route_risk():
    out = add_exit_plan_columns(_df())
    for c in EXIT_OUTPUT_COLS:
        assert c in out.columns
    assert out.loc[0, "EXIT_ROUTE_RISK"] == "OK"          # WAIT
    assert out.loc[1, "EXIT_ROUTE_RISK"] == "HIGH_AVOID"  # CARRY
    assert out.loc[2, "EXIT_ROUTE_RISK"] == "CAUTION"     # ATTACK
    assert out.loc[3, "EXIT_ROUTE_RISK"] == "HIGH_AVOID"  # NEUTRAL


def test_stop_tp_math():
    out = add_exit_plan_columns(_df())
    # 10000 × 0.93 = 9300, × 1.10 = 11000
    assert out.loc[0, "EXIT_STOP_TIGHT"] == 9300
    assert out.loc[0, "EXIT_TP_QUICK"] == 11000
    assert out.loc[1, "EXIT_STOP_TIGHT"] == 18600  # 20000×0.93
    assert out.loc[1, "EXIT_TP_QUICK"] == 22000


def test_official_and_stop_unchanged():
    df = _df()
    tp, be, sg = df.TOP_PICK.tolist(), df.BUY_NOW_ELIGIBLE.tolist(), df.손절가.tolist()
    out = add_exit_plan_columns(df)
    assert out.TOP_PICK.tolist() == tp
    assert out.BUY_NOW_ELIGIBLE.tolist() == be
    assert out.손절가.tolist() == sg   # 기존 손절가 절대 불변


def test_invalid_entry_graceful():
    out = add_exit_plan_columns(_df())
    assert pd.isna(out.loc[3, "EXIT_STOP_TIGHT"])  # 추천매수가 NaN
    assert pd.isna(out.loc[3, "EXIT_TP_QUICK"])
    # 그래도 ROUTE 위험은 표기됨
    assert out.loc[3, "EXIT_ROUTE_RISK"] == "HIGH_AVOID"


def test_note_content():
    out = add_exit_plan_columns(_df())
    assert "손절 9,300" in out.loc[0, "EXIT_PLAN_NOTE"]
    assert "고위험" in out.loc[1, "EXIT_PLAN_NOTE"]
    assert "주의" in out.loc[2, "EXIT_PLAN_NOTE"]


def test_custom_config_and_disable():
    cfg = ExitPlanConfig(stop_tight_pct=-5.0, tp_quick_pct=8.0)
    out = add_exit_plan_columns(_df(), config=cfg)
    assert out.loc[0, "EXIT_STOP_TIGHT"] == 9500   # ×0.95
    assert out.loc[0, "EXIT_TP_QUICK"] == 10800    # ×1.08
    off = add_exit_plan_columns(_df(), config=ExitPlanConfig(exit_enabled=False))
    assert "EXIT_STOP_TIGHT" not in off.columns


def test_facade_and_validation():
    c = CollectorConfig(exit=ExitPlanConfig(stop_tight_pct=-6.0))
    out = add_exit_plan_columns(_df(), config=c)
    assert out.loc[0, "EXIT_STOP_TIGHT"] == 9400
    with pytest.raises(ValueError):
        ExitPlanConfig(stop_tight_pct=-20.0)
    with pytest.raises(ValueError):
        ExitPlanConfig(tp_quick_pct=100.0)


def test_summary_and_empty():
    s = exit_summary(add_exit_plan_columns(_df()))
    assert s["high_avoid"] == 2 and s["caution"] == 1 and s["ok"] == 1
    assert add_exit_plan_columns(pd.DataFrame()) is not None
    assert add_exit_plan_columns(None) is None
    assert exit_summary(pd.DataFrame()) == {}


def test_card_shows_exit_discipline():
    from components.action_cards import build_cards_html
    df = pd.DataFrame([{
        "종목명": "테스트", "종목코드": "123456", "종가": 10000,
        "추천매수가": 10000, "손절가": 9000, "추천매도가1": 12000,
        "RSI14": 60, "DISPLAY_SCORE": 70, "ROUTE": "WAIT", "거래대금(억원)": 200,
        "PRODUCTION_BUY": 1, "ACTION_DECISION": "BUY",
        "EXIT_STOP_TIGHT": 9300, "EXIT_TP_QUICK": 11000, "EXIT_ROUTE_RISK": "OK",
    }])
    h = build_cards_html(df)
    assert "9,300" in h and "규율" in h
    # 고위험 루트 경고
    df2 = df.copy(); df2.loc[0, "EXIT_ROUTE_RISK"] = "HIGH_AVOID"
    assert "고위험 루트" in build_cards_html(df2)
