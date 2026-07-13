# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_profit_recovery_v261 import simulate_signal


def _history(*, both_on_first=False, never_fill=False):
    rows = []
    dates = pd.bdate_range("2026-04-13", periods=10)
    for i, date in enumerate(dates):
        low = 10_100 if never_fill else 9_950
        high = 10_200
        if both_on_first and i == 0:
            low, high = 9_300, 11_100
        rows.append(
            {
                "date": date,
                "시가": 10_000,
                "고가": high,
                "저가": low,
                "종가": 10_100,
            }
        )
    return pd.DataFrame(rows)


def _signal():
    return pd.Series(
        {
            "signal_date": "20260410",
            "code": "005930",
            "name": "테스트",
            "entry": 10_000,
            "rank_score": 80,
            "breadth": 50,
        }
    )


def test_same_bar_stop_and_target_is_conservatively_a_stop():
    result = simulate_signal(_signal(), _history(both_on_first=True))
    assert result["fill_status"] == "FILLED"
    assert result["exit_type"] == "STOP_SAME_BAR"
    assert result["net_return_pct"] == -6.22


def test_unfilled_limit_order_counts_as_zero_not_a_win():
    result = simulate_signal(_signal(), _history(never_fill=True))
    assert result["fill_status"] == "UNFILLED"
    assert result["net_return_pct"] == 0.0
