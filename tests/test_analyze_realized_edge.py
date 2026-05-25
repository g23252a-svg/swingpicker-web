# -*- coding: utf-8 -*-
"""scripts/analyze_realized_edge.py 회귀 테스트."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_realized_edge.py"
    spec = importlib.util.spec_from_file_location("analyze_realized_edge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_code_handles_int_float_and_string():
    mod = _load_module()
    assert mod.normalize_code(3690) == "003690"
    assert mod.normalize_code(187870.0) == "187870"
    assert mod.normalize_code("A005930") == "005930"


def test_build_edge_dataset_merges_recommend_features(tmp_path):
    mod = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rec = pd.DataFrame([
        {"종목코드": "001111", "종목명": "강한종목", "RR_NOW_TP1": 2.1, "AXIS_GAP": 10, "ROUTE": "ATTACK"},
        {"종목코드": "002222", "종목명": "약한종목", "RR_NOW_TP1": 0.8, "AXIS_GAP": 45, "ROUTE": "WAIT"},
    ])
    rec.to_csv(data_dir / "recommend_20260501.csv", index=False)

    trades = pd.DataFrame([
        {"date": 20260501, "code": 1111, "name": "강한종목", "method": "ohlc", "fill_date": "2026-05-02", "outcome": "WIN", "net_pct": 6.0, "tp1_before_stop": True, "stop_hit": False},
        {"date": 20260501, "code": 2222, "name": "약한종목", "method": "ohlc", "fill_date": "2026-05-02", "outcome": "LOSS", "net_pct": -5.0, "tp1_before_stop": False, "stop_hit": True},
    ])
    trades.to_csv(data_dir / "backtest_top1_trades_20260503.csv", index=False)

    merged = mod.build_edge_dataset(data_dir)
    assert len(merged) == 2
    assert set(merged["code_norm"]) == {"001111", "002222"}
    assert "RR_NOW_TP1" in merged.columns
    assert merged["realized_net_pct"].tolist() == [6.0, -5.0]


def test_numeric_slice_detects_edge_and_risk():
    mod = _load_module()
    rows = []
    for i in range(20):
        rows.append({
            "RR_NOW_TP1": 2.0 + i * 0.01,
            "realized_net_pct": 4.0,
            "win": 1,
            "tp1_before_stop": True,
            "stop_hit": False,
            "not_filled": 0,
        })
    for i in range(20):
        rows.append({
            "RR_NOW_TP1": 0.6 + i * 0.01,
            "realized_net_pct": -4.0,
            "win": 0,
            "tp1_before_stop": False,
            "stop_hit": True,
            "not_filled": 0,
        })
    df = pd.DataFrame(rows)
    slices = mod.evaluate_numeric_slices(df, features=["RR_NOW_TP1"], min_n=8)
    assert not slices.empty
    assert "EDGE" in set(slices["verdict"])
    assert "RISK" in set(slices["verdict"])
    assert any((slices["feature"] == "RR_NOW_TP1") & (slices["direction"] == "HIGH"))


def test_write_outputs_creates_report_files(tmp_path):
    mod = _load_module()
    df = pd.DataFrame([
        {"realized_net_pct": 3.0, "win": 1, "tp1_before_stop": True, "stop_hit": False, "not_filled": 0},
        {"realized_net_pct": -2.0, "win": 0, "tp1_before_stop": False, "stop_hit": True, "not_filled": 0},
    ])
    slices = pd.DataFrame([
        {"kind": "numeric", "feature": "RR_NOW_TP1", "verdict": "EDGE", "return_alpha_pct": 1.2, "win_alpha_pp": 10.0, "rule": "RR_NOW_TP1 > 1.5"}
    ])
    report = mod.build_report(df, slices, min_n=1)
    mod.write_outputs(df, slices, report, tmp_path)
    assert (tmp_path / "realized_edge_dataset_latest.csv").exists()
    assert (tmp_path / "realized_edge_slices_latest.csv").exists()
    assert (tmp_path / "realized_edge_report_latest.json").exists()
