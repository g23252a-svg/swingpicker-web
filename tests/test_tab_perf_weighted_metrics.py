# -*- coding: utf-8 -*-
"""v22.3.11 성과탭 KPI 표본가중/낙폭 계산 회귀 가드."""
import sys
import types
import importlib

import pandas as pd


def _import_tab_perf_with_ui_stub():
    """NiceGUI 미설치 테스트 환경에서도 순수 계산 helper만 import."""
    fake_nicegui = types.ModuleType("nicegui")
    fake_nicegui.ui = types.SimpleNamespace()
    sys.modules.setdefault("nicegui", fake_nicegui)
    sys.modules.pop("components.tab_perf", None)
    return importlib.import_module("components.tab_perf")


def test_weighted_mean_uses_total_n_not_row_mean():
    tab_perf = _import_tab_perf_with_ui_stub()
    df = pd.DataFrame({"WIN_RATE_%": [100.0, 0.0], "TOTAL_N": [1, 9]})

    # 단순 평균이면 50.0이지만, 표본가중이면 10.0이어야 한다.
    assert tab_perf._weighted_mean(df, "WIN_RATE_%") == 10.0


def test_weighted_mean_falls_back_when_total_n_missing():
    tab_perf = _import_tab_perf_with_ui_stub()
    df = pd.DataFrame({"AVG_RET_%": [2.0, 6.0]})

    assert tab_perf._weighted_mean(df, "AVG_RET_%") == 4.0


def test_worst_drawdown_uses_max_abs_risk_not_mean():
    tab_perf = _import_tab_perf_with_ui_stub()
    df = pd.DataFrame({"WORST_MDD_%": [-3.0, -13.56, 2.0]})

    # 평균이 아니라 선택 기간 중 가장 큰 위험값을 음수로 표시한다.
    assert tab_perf._worst_drawdown_value(df, "WORST_MDD_%") == -13.56


def test_select_perf_default_slice_prefers_elite_top5_5d():
    tab_perf = _import_tab_perf_with_ui_stub()
    df = pd.DataFrame({
        "METHOD": ["FINAL_SCORE", "ELITE_SCORE", "ELITE_SCORE"],
        "TOPK": [5, 5, 10],
        "H(영업일)": [5, 5, 5],
        "WIN_RATE_%": [40.0, 60.0, 90.0],
    })

    out = tab_perf._select_perf_default_slice(df)
    assert len(out) == 1
    assert out.iloc[0]["METHOD"] == "ELITE_SCORE"
    assert int(out.iloc[0]["TOPK"]) == 5
    assert int(out.iloc[0]["H(영업일)"]) == 5
