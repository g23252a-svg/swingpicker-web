# -*- coding: utf-8 -*-
"""test_p5_final.py — P5 #20 스코어링/지표 + #21 일일 성과 리포트 (pytest 표준)
═══════════════════════════════════════════════════
실행: pytest test_p5_final.py -v
"""
import os
import tempfile
import shutil
import textwrap

import numpy as np
import pandas as pd
import pytest

from scoring_engine import (
    calculate_structural_score, calculate_timing_score,
    calculate_ebs_independent, build_global_score, safe_float,
)


# ═══════════════════════════════════════════════
#  1. TIMING: None/NaN/누락 방어
# ═══════════════════════════════════════════════
class TestTimingDefense:
    def test_empty_row(self):
        ts = calculate_timing_score({})
        assert 0 <= ts <= 100

    def test_none_values(self):
        none_row = {k: None for k in [
            "RAW_TRIGGER_SCORE", "RES_RATIO", "RES_RATIO_NEAR", "POC_GAP",
            "IS_ABOVE_POC", "gap_pct", "SECTOR_RANK", "SUPERTREND_DIR", "TTM_SQUEEZE",
        ]}
        ts = calculate_timing_score(none_row)
        assert 0 <= ts <= 100

    def test_nan_values(self):
        nan_row = {k: float('nan') for k in [
            "RAW_TRIGGER_SCORE", "RES_RATIO", "RES_RATIO_NEAR", "POC_GAP",
            "IS_ABOVE_POC", "gap_pct", "SECTOR_RANK", "SUPERTREND_DIR", "TTM_SQUEEZE",
        ]}
        ts = calculate_timing_score(nan_row)
        assert 0 <= ts <= 100

    def test_extreme_values(self):
        extreme_row = {
            "RAW_TRIGGER_SCORE": 99999, "RES_RATIO": -100, "RES_RATIO_NEAR": 999,
            "POC_GAP": -50, "IS_ABOVE_POC": 1, "gap_pct": 100, "SECTOR_RANK": 0,
            "SUPERTREND_DIR": 1, "TTM_SQUEEZE": 1,
        }
        ts = calculate_timing_score(extreme_row)
        assert np.isfinite(ts)


# ═══════════════════════════════════════════════
#  2. STRUCT: 경계값 검증
# ═══════════════════════════════════════════════
class TestStructuralScore:
    def test_empty_row(self):
        ss = calculate_structural_score({})
        assert 0 <= ss <= 100

    def test_best_conditions(self):
        best_row = {
            "Low_Trend_PCT": 5.0, "MFI14": 80, "Vol_Quality": 2.5,
            "Range_Pos": 1.0, "이격도": 3.0, "Above_MA20": 1,
            "MTF_WEEKLY_TREND": 1, "MTF_MONTHLY_TREND": 1,
            "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_BONUS": 10,
        }
        ss = calculate_structural_score(best_row)
        assert ss > 0, f"best conditions should score > 0, got {ss}"
        assert ss <= 100

    def test_worst_conditions(self):
        worst_row = {
            "Low_Trend_PCT": -5.0, "MFI14": 0, "Vol_Quality": 0,
            "Range_Pos": 0, "이격도": -10, "Above_MA20": 0,
            "MTF_WEEKLY_TREND": -1, "MTF_MONTHLY_TREND": -1,
            "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_PENALTY": 15,
        }
        ss = calculate_structural_score(worst_row)
        assert ss == 0.0


# ═══════════════════════════════════════════════
#  3. SuperTrend: 짧은 데이터 방어
# ═══════════════════════════════════════════════
class TestSuperTrend:
    @pytest.fixture(autouse=True)
    def _extract_calc_supertrend(self):
        """collector.py에서 calc_supertrend 함수만 추출 (import 부작용 회피)."""
        collector_path = os.path.join(os.path.dirname(__file__), "collector.py")
        src = open(collector_path).read()
        exec_ns = {"np": np, "pd": pd, "Tuple": tuple}
        atr_start = src.index("def calc_atr(")
        atr_end = src.index("\ndef ", atr_start + 1)
        exec(src[atr_start:atr_end], exec_ns)
        st_start = src.index("def calc_supertrend(")
        st_end = src.index("\ndef ", st_start + 1)
        exec(src[st_start:st_end], exec_ns)
        self.calc_supertrend = exec_ns["calc_supertrend"]

    def test_short_data(self):
        st_line, st_dir = self.calc_supertrend(
            pd.Series([100, 101, 102]), pd.Series([98, 99, 100]),
            pd.Series([99, 100, 101]), period=10,
        )
        assert len(st_line) == 3

    def test_empty_series(self):
        e = pd.Series([], dtype=float)
        st_e, dir_e = self.calc_supertrend(e, e, e)
        assert len(st_e) == 0

    def test_normal_data(self):
        np.random.seed(42)
        n = 100
        c = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        h = c + abs(np.random.randn(n))
        l = c - abs(np.random.randn(n))
        st_norm, dir_norm = self.calc_supertrend(h, l, c, period=10)
        assert len(st_norm) == n
        assert set(dir_norm.dropna().unique()).issubset({-1, 1})


# ═══════════════════════════════════════════════
#  4. EBS: 경계값
# ═══════════════════════════════════════════════
class TestEBS:
    def test_empty_row(self):
        assert calculate_ebs_independent({}) <= 2

    def test_max_score(self):
        ebs_max = calculate_ebs_independent({
            "Low_Trend_PCT": 1, "Vol_Quality": 1.5,
            "MACD_Slope_PCT": 0.1, "RSI14": 55, "TTM_SQUEEZE": 1,
        })
        assert ebs_max == 10


# ═══════════════════════════════════════════════
#  5. safe_float 방어
# ═══════════════════════════════════════════════
class TestSafeFloat:
    def test_none(self):
        assert safe_float(None) == 0

    def test_string(self):
        assert safe_float("abc") == 0

    def test_nan(self):
        assert safe_float(float('nan')) == 0

    def test_int(self):
        assert safe_float(42) == 42.0

    def test_numeric_string(self):
        assert safe_float("3.14") == 3.14


# ═══════════════════════════════════════════════
#  6. build_global_score 불변식
# ═══════════════════════════════════════════════
class TestBuildGlobalScore:
    @pytest.fixture()
    def test_df(self):
        return pd.DataFrame([{
            "RSI14": 55, "MFI14": 50, "이격도": 2.0, "BB_BW": 0.1,
            "ret_5d_%": 3.0, "MACD_Slope_PCT": 0.01, "Range_Pos": 0.7,
            "Vol_Quality": 1.2, "Above_MA20": 1, "Low_Trend_PCT": 1.0,
            "RAW_TRIGGER_SCORE": 60, "TTM_SQUEEZE": 0, "SUPERTREND_DIR": 0,
            "RES_RATIO": 0, "RES_RATIO_NEAR": 0, "POC_GAP": 0, "IS_ABOVE_POC": 1,
            "gap_pct": 0, "SECTOR_RANK": 99,
        }])

    @pytest.mark.parametrize("col", [
        "STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE",
        "FINAL_SCORE", "PASS_EBS", "DISPLAY_SCORE",
    ])
    def test_required_columns(self, test_df, col):
        out = build_global_score(test_df, "NORMAL")
        assert col in out.columns

    def test_score_ranges(self, test_df):
        out = build_global_score(test_df, "NORMAL")
        assert 0 <= float(out["STRUCT_SCORE"].iloc[0]) <= 100
        assert 0 <= float(out["TIMING_SCORE"].iloc[0]) <= 200
        assert float(out["FINAL_SCORE"].iloc[0]) > 0


# ═══════════════════════════════════════════════
#  7. #21: 일일 성과 리포트
# ═══════════════════════════════════════════════
class TestDailyReport:
    @pytest.fixture()
    def report_dir(self):
        tmp = tempfile.mkdtemp()
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
            delta = (int(ymd[-2:]) - 10) * 100
            snap = pd.DataFrame([
                {"종목코드": "005930", "시가": 70000+delta, "고가": 70500+delta,
                 "저가": 69700+delta, "종가": 70200+delta},
                {"종목코드": "000660", "시가": 150000-delta, "고가": 150500,
                 "저가": 149500-delta, "종가": 150100-delta},
                {"종목코드": "035720", "시가": 50000, "고가": 50200,
                 "저가": 49600, "종가": 49900},
            ])
            snap.to_csv(os.path.join(tmp, f"price_snapshot_{ymd}.csv"), index=False)
        yield tmp
        shutil.rmtree(tmp, ignore_errors=True)

    def test_report_generation(self, report_dir):
        from daily_report import generate_daily_report, DailyReport
        report = generate_daily_report(report_dir, "20260221", lookback_days=30)
        assert isinstance(report, DailyReport)
        assert report.get("n_recommendations", -1) >= 0
        assert "overall_winrate" in report
        assert report.get("report_key", "").startswith("daily_perf:")

    def test_report_text_format(self, report_dir):
        from daily_report import generate_daily_report, _format_report_text
        report = generate_daily_report(report_dir, "20260221", lookback_days=30)
        text = _format_report_text(report)
        assert len(text) > 0
        assert "20260221" in text

    def test_idempotent_report_key(self, report_dir):
        from daily_report import generate_daily_report
        r1 = generate_daily_report(report_dir, "20260221", lookback_days=30)
        r2 = generate_daily_report(report_dir, "20260221", lookback_days=30)
        assert r1["report_key"] == r2["report_key"]

    def test_cost_deduction(self, report_dir):
        from daily_report import generate_daily_report
        report = generate_daily_report(report_dir, "20260221", lookback_days=30)
        if report.get("n_recommendations", 0) > 0:
            cost_diff = report["avg_ret_gross_pct"] - report["avg_ret_net_pct"]
            assert cost_diff > 0
