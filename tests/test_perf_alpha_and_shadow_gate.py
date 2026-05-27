# -*- coding: utf-8 -*-
"""v22.3.14 행별 KOSPI 알파 + Shadow Promotion Gate 회귀 가드."""
import importlib
import sys
import types

import pandas as pd


def _import_tab_perf_with_ui_stub():
    fake_nicegui = types.ModuleType("nicegui")
    fake_nicegui.ui = types.SimpleNamespace()
    sys.modules.setdefault("nicegui", fake_nicegui)
    sys.modules.pop("components.tab_perf", None)
    return importlib.import_module("components.tab_perf")


def test_resolve_alpha_prefers_row_exact_columns():
    tab_perf = _import_tab_perf_with_ui_stub()
    df = pd.DataFrame({
        "AVG_RET_%": [10.0, 0.0],
        "KOSPI_RET_%": [2.0, 4.0],
        "ALPHA_%": [8.0, -4.0],
        "TOTAL_N": [3, 1],
    })

    kospi, alpha, source = tab_perf._resolve_alpha_metrics(
        df, bench_data={"KOSPI": {5: 99.0}}, hold_days=5
    )

    assert round(kospi, 2) == 2.50
    assert round(alpha, 2) == 5.00
    assert source == "row_exact"


def test_resolve_alpha_falls_back_to_bench_average():
    tab_perf = _import_tab_perf_with_ui_stub()
    df = pd.DataFrame({"AVG_RET_%": [7.0, 3.0], "TOTAL_N": [1, 1]})

    kospi, alpha, source = tab_perf._resolve_alpha_metrics(
        df, bench_data={"KOSPI": {5: 4.0}}, hold_days=5
    )

    assert kospi == 4.0
    assert alpha == 1.0
    assert source == "bench_avg"


def test_shadow_promotion_gate_marks_sample_shortage():
    tab_perf = _import_tab_perf_with_ui_stub()
    rule = {
        "delta_ev": 1.2,
        "delta_non_win_avg_ret": 0.5,
        "changed_pick_rate": 0.20,
        "single_backtest_ok": True,
        "n": 8,
    }

    out = tab_perf._score_shadow_promotion_rule("B_red", rule, rwf_pass=True)

    assert out["verdict"] == "표본 부족 · 표시 유지"
    assert out["checks"]["sample"] is False


def test_shadow_promotion_gate_marks_score_candidate():
    tab_perf = _import_tab_perf_with_ui_stub()
    rule = {
        "delta_ev": 0.9,
        "delta_non_win_avg_ret": 0.2,
        "changed_pick_rate": 0.31,
        "single_backtest_ok": True,
        "n": 35,
    }

    out = tab_perf._score_shadow_promotion_rule("B_red", rule, rwf_pass=True)

    assert out["verdict"] == "감점/표시 승격 후보"
    assert out["checks"]["changed"] is True
    assert out["checks"]["rwf"] is True


def test_shadow_promotion_gate_marks_drop_candidate():
    tab_perf = _import_tab_perf_with_ui_stub()
    rule = {
        "delta_ev": -0.2,
        "changed_pick_rate": 0.10,
        "single_backtest_ok": False,
        "n": 40,
    }

    out = tab_perf._score_shadow_promotion_rule("C_orange", rule, rwf_pass=False)

    assert out["verdict"] == "폐기 후보"
    assert out["tone"] == "red"
