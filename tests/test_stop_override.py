# -*- coding: utf-8 -*-
"""test_stop_override.py — [v23.2] 손절 override 회귀 테스트."""
import pandas as pd
import pytest

from stop_override import apply_stop_override, stop_override_summary, STOP_OVERRIDE_COLS


def _df():
    return pd.DataFrame([
        dict(종목명="공식A", 추천매수가=10000.0, 손절가=9400.0, TOP_PICK=1, BUY_NOW_ELIGIBLE=1),
        dict(종목명="공식B", 추천매수가=20000.0, 손절가=18800.0, TOP_PICK=1, BUY_NOW_ELIGIBLE=0),
        dict(종목명="비공식", 추천매수가=5000.0, 손절가=4700.0, TOP_PICK=0, BUY_NOW_ELIGIBLE=0),
    ])


def test_columns_present():
    out = apply_stop_override(_df())
    for c in STOP_OVERRIDE_COLS:
        assert c in out.columns


def test_bull_official_override_price():
    out = apply_stop_override(_df(), market_risk_off=False).set_index("종목명")
    assert bool(out.loc["공식A", "STOP_OVERRIDE_ACTIVE"])
    # 진입가 10000 × (1-0.10) = 9000
    assert abs(float(out.loc["공식A", "STOP_OVERRIDE_PRICE"]) - 9000.0) <= 1
    assert abs(float(out.loc["공식B", "STOP_OVERRIDE_PRICE"]) - 18000.0) <= 1
    assert abs(float(out.loc["공식A", "STOP_OVERRIDE_PCT"]) - 0.10) < 1e-9


def test_non_official_excluded():
    out = apply_stop_override(_df()).set_index("종목명")
    assert not bool(out.loc["비공식", "STOP_OVERRIDE_ACTIVE"])
    assert float(out.loc["비공식", "STOP_OVERRIDE_PRICE"]) == 0.0


def test_original_stop_preserved():
    out = apply_stop_override(_df()).set_index("종목명")
    assert abs(float(out.loc["공식A", "손절가"]) - 9400.0) <= 1
    assert abs(float(out.loc["공식B", "손절가"]) - 18800.0) <= 1


def test_bear_disables_and_blocks():
    out = apply_stop_override(_df(), market_risk_off=True).set_index("종목명")
    assert int(out["STOP_OVERRIDE_ACTIVE"].sum()) == 0       # override OFF
    assert bool(out.loc["공식A", "NEW_ENTRY_BLOCKED"])        # 신규진입 차단
    assert bool(out.loc["비공식", "NEW_ENTRY_BLOCKED"])


def test_active_and_blocked_mutually_exclusive():
    out = apply_stop_override(_df(), market_risk_off=False)
    both = (out["STOP_OVERRIDE_ACTIVE"].astype(bool) & out["NEW_ENTRY_BLOCKED"].astype(bool)).any()
    assert not both


def test_empty_df():
    out = apply_stop_override(pd.DataFrame())
    assert len(out) == 0
    for c in STOP_OVERRIDE_COLS:
        assert c in out.columns


def test_summary_counts():
    s = stop_override_summary(apply_stop_override(_df()))
    assert s["active"] == 2 and s["blocked"] == 0
    s2 = stop_override_summary(apply_stop_override(_df(), market_risk_off=True))
    assert s2["active"] == 0 and s2["blocked"] == 3


class _DisabledCfg:
    class stop_override:
        enabled = False
        stop_pct = 0.10
        apply_to_official_only = True
        disable_on_risk_off = True
        block_new_entry_on_risk_off = True


def test_disabled_no_op():
    out = apply_stop_override(_df(), config=_DisabledCfg())
    assert int(out["STOP_OVERRIDE_ACTIVE"].sum()) == 0
    assert int(out["NEW_ENTRY_BLOCKED"].sum()) == 0


def test_ssot_wired_into_collector_config():
    from collector_config import DEFAULT_CONFIG
    assert hasattr(DEFAULT_CONFIG, "stop_override")
    assert abs(DEFAULT_CONFIG.stop_override.stop_pct - 0.10) < 1e-9
    assert DEFAULT_CONFIG.stop_override.apply_to_official_only is True
    assert DEFAULT_CONFIG.config_version == "2.4.0"
