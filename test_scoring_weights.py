# -*- coding: utf-8 -*-
"""test_scoring_weights.py — 스코어링 가중치 동적화 테스트 (pytest 표준)
═══════════════════════════════════════════════════
실행: pytest test_scoring_weights.py -v
"""
import os

import numpy as np
import pandas as pd
import pytest

from scoring_engine import (
    _calc_ml_weight, build_global_score,
    calculate_structural_score, calculate_timing_score,
)


# ── 공용 fixture ──
@pytest.fixture()
def base_df():
    return pd.DataFrame({
        "RSI14": [55, 70, 40], "MFI14": [50, 60, 45],
        "이격도": [2.0, 5.0, -1.0], "BB_BW": [0.1, 0.2, 0.05],
        "ret_5d_%": [3.0, 8.0, -2.0], "ret_10d_%": [5.0, 12.0, 0.0],
        "ret_20d_%": [10.0, 15.0, -5.0], "ret_60d_%": [20.0, 30.0, 5.0],
        "MACD_Slope_PCT": [0.01, 0.03, -0.01], "Range_Pos": [0.7, 0.9, 0.3],
        "Vol_Quality": [1.2, 1.8, 0.8], "Above_MA20": [1, 1, 0],
        "거래강도": [2.0, 4.0, 0.5], "Low_Trend_PCT": [1.0, 2.0, -1.0],
        "IS_SWING_SUPPORT": [1, 0, 0], "SECTOR_RANK": [2, 5, 10],
        "ML_SCORE": [0.0, 0.0, 0.0],
    })


@pytest.fixture()
def base_row():
    return {
        "RSI14": 55, "MFI14": 50, "이격도": 2.0, "BB_BW": 0.1,
        "ret_5d_%": 3.0, "MACD_Slope_PCT": 0.01, "Range_Pos": 0.7,
        "Vol_Quality": 1.2, "Above_MA20": 1, "Low_Trend_PCT": 1.0,
        "RAW_TRIGGER_SCORE": 60, "TTM_SQUEEZE": 0, "SUPERTREND_DIR": 0,
        "RES_RATIO": 0, "RES_RATIO_NEAR": 0, "POC_GAP": 0, "IS_ABOVE_POC": 1,
        "gap_pct": 0, "SECTOR_RANK": 99, "SECTOR_RS": 0.0,
    }


# ═══════════════════════════════════════════════
#  1. ML 비활성 → AI 가중치 0
# ═══════════════════════════════════════════════
class TestMLInactive:
    def test_wa_zero(self):
        w_s, w_t, w_a = _calc_ml_weight(pd.Series([0.0]*100), "NORMAL")
        assert np.isclose(w_a, 0.0)
        assert np.isclose(w_s + w_t, 1.0)

    def test_normal_balanced(self):
        w_s, w_t, _ = _calc_ml_weight(pd.Series([0.0]*100), "NORMAL")
        assert abs(w_s - w_t) < 0.01


# ═══════════════════════════════════════════════
#  2. ML 완전 활성 → AI 가중치 0.20
# ═══════════════════════════════════════════════
class TestMLActive:
    def test_wa_020(self):
        w_s, w_t, w_a = _calc_ml_weight(pd.Series([50.0]*100), "NORMAL")
        assert np.isclose(w_a, 0.20)
        assert np.isclose(w_s + w_t + w_a, 1.0)


# ═══════════════════════════════════════════════
#  3. 선형 보간
# ═══════════════════════════════════════════════
class TestLinearInterpolation:
    def test_mid_value(self):
        w_s, w_t, w_a = _calc_ml_weight(pd.Series([15.0]*100), "NORMAL")
        assert 0 < w_a < 0.20
        assert np.isclose(w_s + w_t + w_a, 1.0)
        assert np.isclose(w_a, 0.10, atol=0.01)


# ═══════════════════════════════════════════════
#  4. 커버리지 게이트
# ═══════════════════════════════════════════════
class TestCoverageGate:
    def test_sparse_ml_wa_zero(self):
        ml_sparse = pd.Series([0.0]*90 + [80.0]*10)
        _, _, w_a = _calc_ml_weight(ml_sparse, "NORMAL")
        assert np.isclose(w_a, 0.0)


# ═══════════════════════════════════════════════
#  5. CRITICAL 매크로
# ═══════════════════════════════════════════════
class TestCriticalMacro:
    def test_struct_heavier(self):
        w_s, w_t, _ = _calc_ml_weight(pd.Series([50.0]*100), "CRITICAL")
        assert w_s > w_t

    def test_critical_ml_off(self):
        w_s, w_t, w_a = _calc_ml_weight(pd.Series([0.0]*100), "CRITICAL")
        assert w_s > w_t
        assert np.isclose(w_a, 0.0)


# ═══════════════════════════════════════════════
#  6. 경계 연속성
# ═══════════════════════════════════════════════
class TestBoundaryContinuity:
    def test_boundary_49_51(self):
        _, _, w_a49 = _calc_ml_weight(pd.Series([4.9]*100), "NORMAL")
        _, _, w_a51 = _calc_ml_weight(pd.Series([5.1]*100), "NORMAL")
        assert np.isclose(w_a49, 0.0)
        assert w_a51 < 0.02
        assert abs(w_a51 - w_a49) < 0.02


# ═══════════════════════════════════════════════
#  7. build_global_score 통합
# ═══════════════════════════════════════════════
class TestBuildGlobalScoreIntegration:
    def test_final_formula_ml_off(self, base_df):
        result = build_global_score(base_df, "NORMAL")
        assert "FINAL_SCORE" in result.columns
        assert result["FINAL_SCORE"].mean() > 0

        w_s, w_t, w_a = _calc_ml_weight(base_df["ML_SCORE"], "NORMAL")
        expected = ((result["STRUCT_SCORE"]*w_s + result["TIMING_SCORE"]*w_t
                     + result["AI_SCORE"]*w_a)).round(1)
        assert expected.equals(result["FINAL_SCORE"])
        assert np.isclose(w_a, 0.0)

    def test_ml_active_changes_final(self, base_df):
        result_off = build_global_score(base_df, "NORMAL")
        df2 = base_df.copy()
        df2["ML_SCORE"] = [80.0, 70.0, 60.0]
        result_on = build_global_score(df2, "NORMAL")
        assert not result_off["FINAL_SCORE"].equals(result_on["FINAL_SCORE"])

        w_s, w_t, w_a = _calc_ml_weight(df2["ML_SCORE"], "NORMAL")
        expected = ((result_on["STRUCT_SCORE"]*w_s + result_on["TIMING_SCORE"]*w_t
                     + result_on["AI_SCORE"]*w_a)).round(1)
        assert expected.equals(result_on["FINAL_SCORE"])


# ═══════════════════════════════════════════════
#  8. NaN 방어
# ═══════════════════════════════════════════════
class TestNaNDefense:
    def test_nan_series(self):
        w_s, w_t, w_a = _calc_ml_weight(pd.Series([np.nan, np.nan, 50.0]), "NORMAL")
        assert np.isclose(w_s + w_t + w_a, 1.0)


# ═══════════════════════════════════════════════
#  9. ML_SCORE clip
# ═══════════════════════════════════════════════
class TestMLScoreClip:
    def test_clip_values(self, base_df):
        df = base_df.copy()
        df["ML_SCORE"] = [-50.0, 150.0, 80.0]
        result = build_global_score(df, "NORMAL")
        assert result["AI_SCORE"].iloc[0] == 0.0
        assert result["AI_SCORE"].iloc[1] == 100.0
        assert result["AI_SCORE"].iloc[2] == 80.0


# ═══════════════════════════════════════════════
#  10. ML_SCORE 컬럼 없음
# ═══════════════════════════════════════════════
class TestNoMLScoreColumn:
    def test_no_ml_column(self, base_df):
        df = base_df.drop(columns=["ML_SCORE"])
        result = build_global_score(df, "NORMAL")
        assert "FINAL_SCORE" in result.columns
        assert (result["AI_SCORE"] == 0.0).all()


# ═══════════════════════════════════════════════
#  11. macro_risk = HIGH
# ═══════════════════════════════════════════════
class TestHighMacro:
    def test_high_struct_heavier(self):
        ml = pd.Series([50.0]*100)
        w_sh, w_th, _ = _calc_ml_weight(ml, "HIGH")
        assert w_sh > w_th
        w_sn, _, _ = _calc_ml_weight(ml, "NORMAL")
        assert w_sh > w_sn


# ═══════════════════════════════════════════════
#  12. 절사평균 왜곡 방지
# ═══════════════════════════════════════════════
class TestTrimmedMean:
    def test_outlier_removal(self):
        ml_skew = pd.Series([30.0]*95 + [100.0]*5)
        _, _, w_a = _calc_ml_weight(ml_skew, "NORMAL")
        assert np.isclose(w_a, 0.20, atol=0.01)

    def test_extreme_distribution_gate(self):
        ml_extreme = pd.Series([0.0]*90 + [80.0]*10)
        _, _, w_a = _calc_ml_weight(ml_extreme, "NORMAL")
        assert np.isclose(w_a, 0.0)


# ═══════════════════════════════════════════════
#  13. PASS_EBS 생성
# ═══════════════════════════════════════════════
class TestPassEBS:
    def test_pass_ebs_column(self, base_df):
        result = build_global_score(base_df, "NORMAL")
        assert "PASS_EBS" in result.columns
        assert set(result["PASS_EBS"].unique()).issubset({0, 1})


# ═══════════════════════════════════════════════
#  14. config 외부화
# ═══════════════════════════════════════════════
class TestConfigExternalization:
    def test_custom_ml_max_weight(self):
        from collector_config import CollectorConfig, MacroConfig
        custom = CollectorConfig(macro=MacroConfig(ml_max_weight=0.30, ml_low=0.0, ml_high=10.0))
        w_s, w_t, w_a = _calc_ml_weight(pd.Series([8.0]*100), "NORMAL", config=custom)
        assert np.isclose(w_a, 0.24, atol=0.02)
        assert np.isclose(w_s + w_t + w_a, 1.0)


# ═══════════════════════════════════════════════
#  15. macro_risk 방어
# ═══════════════════════════════════════════════
class TestMacroRiskFallback:
    def test_unknown_risk_falls_to_normal(self):
        ml = pd.Series([50.0]*100)
        w_u = _calc_ml_weight(ml, "UNKNOWN_RISK")
        w_n = _calc_ml_weight(ml, "NORMAL")
        assert np.isclose(w_u[0], w_n[0]) and np.isclose(w_u[1], w_n[1])


# ═══════════════════════════════════════════════
#  16. 소규모 n (trimmed mean)
# ═══════════════════════════════════════════════
class TestSmallN:
    def test_n5(self):
        ml = pd.Series([30.0, 30.0, 30.0, 100.0, 0.0])
        w_s, w_t, w_a = _calc_ml_weight(ml, "NORMAL")
        assert np.isclose(w_s + w_t + w_a, 1.0)
        assert w_a > 0

    def test_n1(self):
        w_s, w_t, w_a = _calc_ml_weight(pd.Series([50.0]), "NORMAL")
        assert np.isclose(w_s + w_t + w_a, 1.0)


# ═══════════════════════════════════════════════
#  17. 섹터 이중 보상 완전 잠금
# ═══════════════════════════════════════════════
class TestSectorDoubleDip:
    def test_struct_sector_invariant(self, base_row):
        top = {**base_row, "SECTOR_RANK": 1, "SECTOR_RS": 5.0}
        assert np.isclose(calculate_structural_score(base_row),
                          calculate_structural_score(top))

    def test_timing_sector_bonus(self, base_row):
        top = {**base_row, "SECTOR_RANK": 1, "SECTOR_RS": 5.0}
        delta = calculate_timing_score(top) - calculate_timing_score(base_row)
        assert np.isclose(delta, 8.0, atol=0.5)

    def test_final_delta_is_timing_only(self, base_row):
        top = {**base_row, "SECTOR_RANK": 1, "SECTOR_RS": 5.0}
        out_base = build_global_score(pd.DataFrame([base_row]), "NORMAL")
        out_top = build_global_score(pd.DataFrame([top]), "NORMAL")
        final_delta = float(out_top["FINAL_SCORE"].iloc[0]) - float(out_base["FINAL_SCORE"].iloc[0])
        timing_delta = calculate_timing_score(top) - calculate_timing_score(base_row)
        _, w_t, _ = _calc_ml_weight(out_base["ML_SCORE"], "NORMAL")
        assert np.isclose(final_delta, timing_delta * w_t, atol=0.5)

    def test_sector_rs_only_no_effect(self, base_row):
        diff_rs = {**base_row, "SECTOR_RANK": 99, "SECTOR_RS": 99.0}
        f_base = float(build_global_score(pd.DataFrame([base_row]), "NORMAL")["FINAL_SCORE"].iloc[0])
        f_diff = float(build_global_score(pd.DataFrame([diff_rs]), "NORMAL")["FINAL_SCORE"].iloc[0])
        assert np.isclose(f_base, f_diff)

    def test_w_sector_dead_code(self):
        src = open(os.path.join(os.path.dirname(__file__), "collector.py")).read()
        assert "W_SECTOR =" not in src
        assert "* W_SECTOR" not in src


# ═══════════════════════════════════════════════
#  18. Multi-Timeframe 경로 잠금
# ═══════════════════════════════════════════════
class TestMultiTimeframe:
    @pytest.fixture()
    def mtf_base(self, base_row):
        return {**base_row, "MTF_WEEKLY_TREND": 0, "MTF_MONTHLY_TREND": 0,
                "MTF_DATA_SUFFICIENT": 0}

    def test_no_data_no_effect(self, mtf_base):
        mtf_up = {**mtf_base, "MTF_WEEKLY_TREND": 1, "MTF_MONTHLY_TREND": 1,
                  "MTF_DATA_SUFFICIENT": 0}
        assert np.isclose(calculate_structural_score(mtf_base),
                          calculate_structural_score(mtf_up))

    def test_both_up_plus_10(self, mtf_base):
        mtf_up = {**mtf_base, "MTF_WEEKLY_TREND": 1, "MTF_MONTHLY_TREND": 1,
                  "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_BONUS": 10}
        delta = calculate_structural_score(mtf_up) - calculate_structural_score(mtf_base)
        assert np.isclose(delta, 10.0, atol=0.5)

    def test_both_down_penalty(self, mtf_base):
        mtf_dn = {**mtf_base, "MTF_WEEKLY_TREND": -1, "MTF_MONTHLY_TREND": -1,
                  "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_PENALTY": 15}
        delta = calculate_structural_score(mtf_dn) - calculate_structural_score(mtf_base)
        assert delta < -5.0

    def test_one_side_up_half_bonus(self, mtf_base):
        mtf_w = {**mtf_base, "MTF_WEEKLY_TREND": 1, "MTF_MONTHLY_TREND": 0,
                 "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_BONUS": 10}
        delta = calculate_structural_score(mtf_w) - calculate_structural_score(mtf_base)
        assert np.isclose(delta, 5.0, atol=0.5)

    def test_timing_unaffected_by_mtf(self, mtf_base):
        mtf_up = {**mtf_base, "MTF_WEEKLY_TREND": 1, "MTF_MONTHLY_TREND": 1,
                  "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_BONUS": 10}
        assert np.isclose(calculate_timing_score(mtf_base),
                          calculate_timing_score(mtf_up))

    def test_final_delta_struct_path_only(self, mtf_base):
        mtf_up = {**mtf_base, "MTF_WEEKLY_TREND": 1, "MTF_MONTHLY_TREND": 1,
                  "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_BONUS": 10}
        out_b = build_global_score(pd.DataFrame([mtf_base]), "NORMAL")
        out_u = build_global_score(pd.DataFrame([mtf_up]), "NORMAL")
        final_delta = float(out_u["FINAL_SCORE"].iloc[0]) - float(out_b["FINAL_SCORE"].iloc[0])
        struct_delta = calculate_structural_score(mtf_up) - calculate_structural_score(mtf_base)
        w_s, _, _ = _calc_ml_weight(out_b["ML_SCORE"], "NORMAL")
        assert np.isclose(final_delta, struct_delta * w_s, atol=0.5)
