# -*- coding: utf-8 -*-
import json
from pathlib import Path

import pandas as pd

from scripts.official_buy_validation import build_official_buy_validation


def _write_csv(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_official_buy_validation_tracks_official_and_holdout(tmp_path):
    data = tmp_path / "data"
    data.mkdir()

    _write_csv(
        data / "recommend_20260501.csv",
        [
            {
                "종목코드": "000001",
                "종목명": "공식매수",
                "TOP_PICK": 1,
                "BUY_NOW_ELIGIBLE": 1,
                "BUY_NOW_GRADE": "BUY",
                "FINAL_SCORE": 82,
                "ELITE_SCORE": 77,
                "ROUTE": "ATTACK",
            }
        ],
    )
    _write_csv(
        data / "recommend_20260502.csv",
        [
            {
                "종목코드": "000002",
                "종목명": "보류탑픽",
                "TOP_PICK": 1,
                "BUY_NOW_ELIGIBLE": 0,
                "BUY_NOW_GRADE": "WATCH",
                "FINAL_SCORE": 88,
                "ELITE_SCORE": 80,
                "ROUTE": "ARMED",
            }
        ],
    )
    _write_csv(
        data / "backtest_top1_trades_20260510.csv",
        [
            {
                "date": "20260501",
                "code": "000001",
                "name": "공식매수",
                "outcome": "WIN",
                "net_pct": 7.0,
                "ret_pct": 7.4,
                "days_held": 5,
            },
            {
                "date": "20260502",
                "code": "000002",
                "name": "보류탑픽",
                "outcome": "LOSS",
                "net_pct": -3.0,
                "ret_pct": -2.6,
                "days_held": 4,
            },
        ],
    )

    df, payload = build_official_buy_validation(data, data)
    s = payload["summary"]

    assert len(df) == 2
    assert s["signal_days"] == 2
    assert s["official_buy_signals"] == 1
    assert s["official_buy_results"] == 1
    assert s["official_buy_win_rate"] == 100.0
    assert s["official_buy_avg_net_pct"] == 7.0
    assert s["no_official_buy_days"] == 1
    assert s["top_pick_holdout_results"] == 1
    assert s["holdout_top_pick_avg_net_pct"] == -3.0
    assert s["cash_vs_top_pick_avg_pct"] == 3.0
    assert s["cash_avoided_loss_days"] == 1
    assert s["cash_opportunity_cost_days"] == 0

    holdout = df[df["code"] == "000002"].iloc[0]
    assert holdout["cash_verdict"] == "CASH_AVOIDED_LOSS"
    assert holdout["cash_vs_top_pick_pct"] == 3.0

    assert (data / "official_buy_validation_latest.csv").exists()
    j = json.loads((data / "official_buy_validation_latest.json").read_text(encoding="utf-8"))
    assert j["summary"]["cash_vs_top_pick_avg_pct"] == 3.0


def test_official_buy_validation_handles_pending_results(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(
        data / "recommend_20260503.csv",
        [
            {
                "종목코드": "3",
                "종목명": "미확정",
                "TOP_PICK": 1,
                "BUY_NOW_ELIGIBLE": 0,
                "BUY_NOW_GRADE": "WATCH",
            }
        ],
    )

    df, payload = build_official_buy_validation(data, data)
    s = payload["summary"]

    assert len(df) == 1
    assert s["top_pick_holdout_signals"] == 1
    assert s["top_pick_holdout_results"] == 0
    assert s["cash_vs_top_pick_avg_pct"] is None
    assert df.iloc[0]["cash_verdict"] == "PENDING"
