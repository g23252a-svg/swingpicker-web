# -*- coding: utf-8 -*-

from datetime import date
import sys
import types

import pandas as pd

# Unit tests target pure helper functions; NiceGUI/chart deps are not required.
sys.modules.setdefault("nicegui", types.SimpleNamespace(ui=types.SimpleNamespace()))
sys.modules.setdefault(
    "shared_utils",
    types.SimpleNamespace(safe_float=lambda value, default=0: default if value is None else float(value)),
)

from components.tab_market import (  # noqa: E402
    _build_no_buy_gate_audit,
    _extract_fx_level,
    _extract_macro_msg_date,
    _fx_regime_diagnosis,
)


def test_extract_fx_level_and_macro_msg_date_from_korean_message():
    msg = "환율 1513원 [05/24] (CRITICAL)"
    assert _extract_fx_level(msg) == 1513
    assert _extract_macro_msg_date(msg, date(2026, 5, 27)) == date(2026, 5, 24)


def test_fx_regime_flags_stale_high_fx_with_ok_breadth_as_regime_check():
    diag = _fx_regime_diagnosis(
        macro_msg="환율 1513원 [05/24] (CRITICAL)",
        macro_risk="CRITICAL",
        breadth=51.7,
        max_route="ATTACK",
        reference_date=date(2026, 5, 27),
    )
    assert diag["fx_level"] == 1513
    assert diag["stale_days"] == 3
    assert diag["is_stale"] is True
    assert diag["is_high_fx"] is True
    assert diag["breadth_ok"] is True
    assert "전면 매수금지" in diag["verdict"]
    assert diag["tone"] == "amber"


def test_no_buy_gate_audit_counts_active_routes_and_gate_reasons():
    df = pd.DataFrame([
        {
            "종목명": "A",
            "ROUTE": "ATTACK",
            "ELITE_SCORE": 72,
            "RR_NOW_TP1": 1.0,
            "VWAP_GAP": 12,
            "POC_GAP": 10,
            "TOP_PICK": 0,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_PASS": 0,
            "EBS_STATUS": "FAIL",
        },
        {
            "종목명": "B",
            "ROUTE": "ARMED",
            "ELITE_SCORE": 83,
            "RR_NOW_TP1": 2.0,
            "VWAP_GAP": 3,
            "POC_GAP": 35,
            "TOP_PICK": 1,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_PASS": 1,
            "EBS_STATUS": "PASS",
        },
        {
            "종목명": "C",
            "ROUTE": "WAIT",
            "ELITE_SCORE": 90,
            "RR_NOW_TP1": 3.0,
            "TOP_PICK": 0,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_PASS": 0,
            "EBS_STATUS": "PASS",
        },
    ])
    audit = _build_no_buy_gate_audit(df, {"macro_risk": "CRITICAL", "max_allowed_route": "ATTACK"})
    counts = audit["counts"]

    assert counts["total"] == 3
    assert counts["armed_attack"] == 2
    assert counts["top_pick"] == 1
    assert counts["official_buy"] == 0
    assert counts["macro_blocked"] == 2
    assert counts["final_under_75"] == 1
    assert counts["buy_now_pass_0"] == 1
    assert counts["buy_now_eligible_0"] == 2
    assert counts["vwap_poc_overheat"] == 2
    assert counts["rr_under_1_2"] == 1
    assert counts["ebs_fail"] == 1
    assert audit["closest"][0]["name"] == "B"
