# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import macro_filter


def _frame(values, end="2026-07-10"):
    index = pd.bdate_range(end=pd.Timestamp(end), periods=len(values))
    return pd.DataFrame({"Close": values}, index=index)


def _patch_sources(monkeypatch, fx_values):
    fx = _frame(fx_values)
    nq = _frame([20_000, 20_100])

    def fake_fetch(symbol, *_args, **_kwargs):
        return fx if symbol == "USD/KRW" else nq

    monkeypatch.setattr(macro_filter, "HAS_FDR", True)
    monkeypatch.setattr(macro_filter, "HAS_YF", False)
    monkeypatch.setattr(macro_filter, "_safe_fdr_fetch", fake_fetch)


def test_high_but_stable_fx_is_caution_not_permanent_critical(monkeypatch):
    _patch_sources(monkeypatch, [1500, 1502, 1501, 1504, 1503, 1506])
    risk, message, _ebs, _limit = macro_filter.check_macro_env("20260710")
    assert risk == "CAUTION"
    assert "고환율 CAUTION" in message
    assert "5일 +0.4%" in message


def test_high_fx_with_five_day_shock_is_critical(monkeypatch):
    _patch_sources(monkeypatch, [1460, 1465, 1470, 1480, 1490, 1506])
    risk, message, _ebs, _limit = macro_filter.check_macro_env("20260710")
    assert risk == "CRITICAL"
    assert "급등 CRITICAL" in message
    assert "5일 +3.2%" in message
