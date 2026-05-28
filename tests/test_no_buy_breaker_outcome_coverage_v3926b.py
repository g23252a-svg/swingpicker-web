
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.no_buy_breaker_backtest_v3926 import run_backtest


def _recommend_row(code="000001", name="테스트", close=100.0, **overrides):
    row = {
        "종목코드": code,
        "종목명": name,
        "종가": close,
        "추천매수가": 100.0,
        "손절가": 94.0,
        "추천매도가1": 110.0,
        "ROUTE": "ARMED",
        "TOP_PICK": 0,
        "BUY_NOW_ELIGIBLE": 0,
        "BUY_NOW_PASS": 1,
        "PASS_EBS": 1,
        "거래대금(억원)": 100,
        "ENTRY_GAP_PCT": 0.0,
        "RR_NOW_TP1": 1.3,
        "STRUCT_SCORE": 95,
        "TIMING_SCORE": 65,
        "AI_SCORE": 80,
        "FINAL_SCORE": 82,
        "ELITE_SCORE": 74,
        "ENTRY_RISK_LEVEL": "GREEN",
        "VWAP_GAP": 4.0,
        "POC_GAP": 5.0,
        "MFI14": 60,
        "ret_1d_%": 0.5,
        "ret_5d_%": 3.0,
    }
    row.update(overrides)
    return row


def test_direct_recommend_close_fills_missing_backtest_trade_result(tmp_path: Path):
    data = tmp_path / "data"
    out = tmp_path / "out"
    data.mkdir()

    dates = ["20260501", "20260502", "20260503", "20260504", "20260505", "20260506"]
    closes = [100, 101, 102, 103, 104, 108]
    for d, c in zip(dates, closes):
        pd.DataFrame([_recommend_row(close=c)]).to_csv(data / f"recommend_{d}.csv", index=False, encoding="utf-8-sig")

    payload = run_backtest(str(data), str(out))
    trades = pd.read_csv(out / "no_buy_breaker_trades_latest.csv")
    rules = pd.read_csv(out / "no_buy_breaker_rules_latest.csv")

    assert payload["realized_rows"] > 0
    assert "DIRECT_RECOMMEND_CLOSE_5D" in set(trades["TRADE_SOURCE"].dropna().astype(str))
    assert trades["REALIZED_RET_5D"].notna().any()
    assert "CANDIDATE_N" in rules.columns
    assert "REALIZED_N" in rules.columns
    assert "VALIDATION_COVERAGE" in rules.columns


def test_pending_outcome_not_counted_as_realized(tmp_path: Path):
    data = tmp_path / "data"
    out = tmp_path / "out"
    data.mkdir()

    for d in ["20260501", "20260502", "20260503"]:
        pd.DataFrame([_recommend_row(close=100)]).to_csv(data / f"recommend_{d}.csv", index=False, encoding="utf-8-sig")

    payload = run_backtest(str(data), str(out))
    trades = pd.read_csv(out / "no_buy_breaker_trades_latest.csv")
    assert payload["realized_rows"] == 0
    assert payload["pending_rows"] > 0
    assert trades["REALIZED_RET_5D"].isna().all()
    assert trades["OUTCOME_STATUS"].isin(["PENDING", "NO_OUTCOME"]).all()
