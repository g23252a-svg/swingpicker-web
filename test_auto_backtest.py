# -*- coding: utf-8 -*-
"""test_auto_backtest.py — P3 #13 자동 백테스트 피드백 루프 (pytest 표준)
═══════════════════════════════════════════════════
실행: pytest test_auto_backtest.py -v
"""
import os
import json
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

from auto_backtest import (
    BacktestConfig, DEFAULT_BT_CONFIG,
    compute_realized_returns, build_winrate_table,
    kelly_from_table, auto_calibrate,
)


def _make_test_data(tmp):
    """테스트용 recommend + price_snapshot 생성 (10영업일)."""
    days = ["20260210", "20260211", "20260212", "20260213", "20260214",
            "20260217", "20260218", "20260219", "20260220", "20260221"]
    for ymd in days:
        rec = pd.DataFrame({
            "종목코드": ["005930", "000660", "035720"],
            "종목명": ["삼성전자", "SK하이닉스", "카카오"],
            "DISPLAY_SCORE": [85.0, 65.0, 45.0],
            "FINAL_SCORE": [85.0, 65.0, 45.0],
            "손절가": [68000, 140000, 45000],
        })
        rec.to_csv(os.path.join(tmp, f"recommend_{ymd}.csv"), index=False)

        base = {"005930": 70000, "000660": 150000, "035720": 50000}
        delta = (int(ymd[-2:]) - 10) * 100
        snap = pd.DataFrame([
            {"종목코드": "005930", "시가": base["005930"]+delta,
             "고가": base["005930"]+delta+500, "저가": base["005930"]+delta-300,
             "종가": base["005930"]+delta+200},
            {"종목코드": "000660", "시가": base["000660"]-delta,
             "고가": base["000660"], "저가": base["000660"]-delta-200,
             "종가": base["000660"]-delta+100},
            {"종목코드": "035720", "시가": base["035720"],
             "고가": base["035720"]+200, "저가": base["035720"]-400,
             "종가": base["035720"]-100},
        ])
        snap.to_csv(os.path.join(tmp, f"price_snapshot_{ymd}.csv"), index=False)


@pytest.fixture()
def bt_dir():
    tmp = tempfile.mkdtemp()
    _make_test_data(tmp)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture()
def cfg5():
    return BacktestConfig(horizon_bdays=5, min_n=1, min_effective_n=0.1, lookback_days=30)


# ═══════════════════════════════════════════════
#  1. Config 검증
# ═══════════════════════════════════════════════
class TestBacktestConfig:
    def test_defaults(self):
        cfg = DEFAULT_BT_CONFIG
        assert cfg.horizon_bdays == 5
        assert cfg.min_n == 30
        assert cfg.kelly_fraction_mult == 0.25
        assert cfg.kelly_cap == 0.10
        assert cfg.round_trip_cost_pct > 0
        assert 0.3 < cfg.round_trip_cost_pct < 1.0


# ═══════════════════════════════════════════════
#  2. 안전장치1: 미확정 추천 제외 (look-ahead)
# ═══════════════════════════════════════════════
class TestLookAheadPrevention:
    def test_no_late_recommendations(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if not returns.empty:
            assert returns["rec_date"].max() <= "20260214"
            assert len(returns[returns["rec_date"] > "20260214"]) == 0


# ═══════════════════════════════════════════════
#  3. 안전장치2: 진입=다음날 시가
# ═══════════════════════════════════════════════
class TestEntryPrice:
    def test_entry_is_next_day_open(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if returns.empty:
            pytest.skip("returns empty")
        r0210 = returns[returns["rec_date"] == "20260210"]
        entry = r0210[r0210["code"] == "005930"]["entry_price"]
        if not entry.empty:
            # 20260211 시가 = 70000 + (11-10)*100 = 70100
            assert abs(float(entry.iloc[0]) - 70100) < 1


# ═══════════════════════════════════════════════
#  4. 안전장치3: 비용 차감
# ═══════════════════════════════════════════════
class TestCostDeduction:
    def test_gross_minus_net_equals_cost(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if returns.empty or "ret_gross_pct" not in returns.columns:
            pytest.skip("no returns or missing columns")
        diff = returns["ret_gross_pct"] - returns["ret_net_pct"]
        assert abs(diff.mean() - cfg5.round_trip_cost_pct) < 0.01


# ═══════════════════════════════════════════════
#  5. 안전장치4: min_n + 라플라스 스무딩
# ═══════════════════════════════════════════════
class TestMinNSmoothing:
    def test_winrate_table(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if returns.empty:
            pytest.skip("returns empty")
        table = build_winrate_table(returns, cfg5)
        assert not table.empty

    def test_strict_min_n(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if returns.empty:
            pytest.skip("returns empty")
        cfg_strict = BacktestConfig(horizon_bdays=5, min_n=30, min_effective_n=10.0, lookback_days=30)
        table_strict = build_winrate_table(returns, cfg_strict)
        if not table_strict.empty:
            insufficient = table_strict[~table_strict["sufficient"]]
            assert len(insufficient) >= len(table_strict) - 1

    def test_laplace_smoothing(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if returns.empty:
            pytest.skip("returns empty")
        table = build_winrate_table(returns, cfg5)
        nonzero_bins = table[table["n_raw"] > 0]
        for _, row in nonzero_bins.iterrows():
            assert 0 < row["p_win"] < 1


# ═══════════════════════════════════════════════
#  6. 안전장치5: 켈리 제한
# ═══════════════════════════════════════════════
class TestKellyLimit:
    def test_kelly_within_cap(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if returns.empty:
            pytest.skip("returns empty")
        table = build_winrate_table(returns, cfg5)
        if table.empty:
            pytest.skip("table empty")
        k85 = kelly_from_table(85.0, table, config=cfg5)
        assert 0 <= k85 <= cfg5.kelly_cap

    def test_insufficient_sample_kelly_zero(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if returns.empty:
            pytest.skip("returns empty")
        cfg_strict = BacktestConfig(horizon_bdays=5, min_n=30, min_effective_n=10.0, lookback_days=30)
        table_strict = build_winrate_table(returns, cfg_strict)
        k = kelly_from_table(85.0, table_strict, config=cfg_strict)
        assert k == 0.0

    def test_out_of_range_score_kelly_zero(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if returns.empty:
            pytest.skip("returns empty")
        table = build_winrate_table(returns, cfg5)
        assert kelly_from_table(200.0, table, config=cfg5) == 0.0


# ═══════════════════════════════════════════════
#  7. auto_calibrate 통합
# ═══════════════════════════════════════════════
class TestAutoCalibrate:
    def test_summary_and_json(self, bt_dir, cfg5):
        summary = auto_calibrate(bt_dir, "20260221", cfg5)
        assert isinstance(summary, dict)
        assert summary.get("n_trades", 0) > 0

        json_path = os.path.join(bt_dir, "winrate_table_20260221.json")
        assert os.path.exists(json_path)
        assert os.path.exists(os.path.join(bt_dir, "winrate_table_latest.json"))

        data = json.load(open(json_path))
        assert isinstance(data, dict) and "table" in data
        assert data.get("meta", {}).get("version") == "v20.8"
        assert data.get("meta", {}).get("horizon_bdays") == cfg5.horizon_bdays


# ═══════════════════════════════════════════════
#  8. 기업행위 필터
# ═══════════════════════════════════════════════
class TestCorporateActionFilter:
    def test_extreme_return_excluded(self, bt_dir, cfg5):
        # 극단 가격 삽입 (400% 수익)
        # entry = 다음날 시가 (20260211), exit = +5bdays 종가
        # 코드의 bday 계산에 따라 exit이 20260218 또는 20260219일 수 있음 → 둘 다 커버
        extreme_entry = pd.DataFrame([{"종목코드": "999999", "시가": 10000,
                                       "고가": 10000, "저가": 10000, "종가": 10000}])
        extreme_exit = pd.DataFrame([{"종목코드": "999999", "시가": 50000,
                                      "고가": 50000, "저가": 50000, "종가": 50000}])

        for ymd_entry in ["20260211"]:
            snap = pd.read_csv(os.path.join(bt_dir, f"price_snapshot_{ymd_entry}.csv"),
                               dtype={"종목코드": str})
            pd.concat([snap, extreme_entry]).to_csv(
                os.path.join(bt_dir, f"price_snapshot_{ymd_entry}.csv"), index=False)

        for ymd_exit in ["20260218", "20260219"]:
            snap_path = os.path.join(bt_dir, f"price_snapshot_{ymd_exit}.csv")
            if os.path.exists(snap_path):
                snap = pd.read_csv(snap_path, dtype={"종목코드": str})
                pd.concat([snap, extreme_exit]).to_csv(snap_path, index=False)
            else:
                extreme_exit.to_csv(snap_path, index=False)

        rec = pd.read_csv(os.path.join(bt_dir, "recommend_20260210.csv"),
                          dtype={"종목코드": str})
        extreme_rec = pd.DataFrame([{"종목코드": "999999", "종목명": "극단종목",
                                     "DISPLAY_SCORE": 75.0, "FINAL_SCORE": 75.0, "손절가": 8000}])
        pd.concat([rec, extreme_rec]).to_csv(
            os.path.join(bt_dir, "recommend_20260210.csv"), index=False)

        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        assert len(returns[returns["code"] == "999999"]) == 0


# ═══════════════════════════════════════════════
#  9. 수학적 일관성
# ═══════════════════════════════════════════════
class TestMathConsistency:
    def test_winrate_matches(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if returns.empty:
            pytest.skip("returns empty")
        summary = auto_calibrate(bt_dir, "20260221", cfg5)
        manual_wr = returns["win"].mean()
        assert abs(summary["overall_winrate"] - manual_wr) < 0.001

    def test_net_equals_gross_minus_cost(self, bt_dir, cfg5):
        returns = compute_realized_returns(bt_dir, "20260221", cfg5)
        if returns.empty:
            pytest.skip("returns empty")
        cost = cfg5.round_trip_cost_pct
        check = (returns["ret_gross_pct"] - returns["ret_net_pct"] - cost).abs()
        assert check.max() < 0.001
