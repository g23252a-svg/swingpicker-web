# -*- coding: utf-8 -*-
"""test_ssot_import.py — collector.py ↔ scoring_engine.py SSOT import smoke test (pytest 표준)
═══════════════════════════════════════════════════
실행: pytest test_ssot_import.py -v
"""
import os
import re

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════
#  1. scoring_engine import 성공
# ═══════════════════════════════════════════════
class TestScoringEngineImport:
    def test_all_functions_importable(self):
        from scoring_engine import (
            determine_state,
            determine_state_dynamic,
            calculate_ebs_independent,
            calculate_structural_score,
            calculate_timing_score,
            build_global_score,
            _calc_ml_weight,
        )
        assert determine_state is not None


# ═══════════════════════════════════════════════
#  2. 순환 import 방어
# ═══════════════════════════════════════════════
class TestCircularImport:
    @pytest.fixture(autouse=True)
    def _load_src(self):
        import scoring_engine
        self.src = open(scoring_engine.__file__).read()

    def test_no_collector_import_in_scoring_engine(self):
        assert "import collector " not in self.src
        assert "from collector " not in self.src
        assert "from collector_config" in self.src

    def test_weight_config_removed(self):
        assert "WEIGHT_CONFIG" not in self.src or "WEIGHT_CONFIG →" in self.src


# ═══════════════════════════════════════════════
#  3. CollectorConfig SSOT 접근
# ═══════════════════════════════════════════════
class TestCollectorConfigSSOT:
    def test_default_config_exists(self):
        from collector_config import DEFAULT_CONFIG, CollectorConfig
        assert isinstance(DEFAULT_CONFIG, CollectorConfig)

    def test_ml_low_field(self):
        from collector_config import DEFAULT_CONFIG
        assert hasattr(DEFAULT_CONFIG, 'ml_low')

    def test_macro_weights_field(self):
        from collector_config import DEFAULT_CONFIG
        assert hasattr(DEFAULT_CONFIG, 'macro_weights')

    def test_snapshot_method(self):
        from collector_config import DEFAULT_CONFIG
        assert hasattr(DEFAULT_CONFIG, 'snapshot')


# ═══════════════════════════════════════════════
#  4. 함수 호출 smoke
# ═══════════════════════════════════════════════
class TestFunctionSmoke:
    def test_calc_ml_weight_sum_one(self):
        from scoring_engine import _calc_ml_weight
        ml = pd.Series([50.0] * 20)
        w_s, w_t, w_a = _calc_ml_weight(ml, "NORMAL")
        assert np.isclose(w_s + w_t + w_a, 1.0)

    def test_determine_state_dynamic(self):
        from scoring_engine import determine_state_dynamic
        row = {
            "RSI14": 50, "ret_1d_%": 2, "ret_5d_%": 3,
            "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5,
            "Vol_Quality": 1.0, "TIMING_SCORE": 50,
            "거래강도": 2.0, "Low_Trend_PCT": 0, "Above_MA20": 1,
            "거래대금(원)": 5e9, "외인순매수": 0, "개인순매수": 0,
        }
        th = {"vol_q75": 1.2, "range_q75": 0.8}
        state = determine_state_dynamic(row, th)
        assert state in {"ATTACK", "ARMED", "WAIT", "NEUTRAL", "OVERHEAT", "EXIT_WARNING"}

    def test_build_global_score(self):
        from scoring_engine import build_global_score
        test_df = pd.DataFrame({
            "RSI14": [55], "MFI14": [50], "이격도": [2.0], "BB_BW": [0.1],
            "ret_5d_%": [3.0], "ret_10d_%": [5.0], "ret_20d_%": [10.0], "ret_60d_%": [20.0],
            "MACD_Slope_PCT": [0.01], "Range_Pos": [0.7], "Vol_Quality": [1.2],
            "Above_MA20": [1], "거래강도": [2.0], "Low_Trend_PCT": [1.0],
            "IS_SWING_SUPPORT": [1], "SECTOR_RANK": [2], "ML_SCORE": [0.0],
        })
        result = build_global_score(test_df, "NORMAL")
        assert "FINAL_SCORE" in result.columns
        assert "PASS_EBS" in result.columns


# ═══════════════════════════════════════════════
#  5. collector에 중복 정의 없음
# ═══════════════════════════════════════════════
class TestNoDuplicateDefinitions:
    @pytest.fixture(autouse=True)
    def _load_collector_src(self):
        self.collector_src = open(
            os.path.join(os.path.dirname(__file__), "collector.py")
        ).read()

    @pytest.mark.parametrize("fn_def", [
        "def calculate_ebs_independent",
        "def calculate_structural_score",
        "def calculate_timing_score",
        "def determine_state_dynamic",
    ])
    def test_no_function_in_collector(self, fn_def):
        assert fn_def not in self.collector_src

    def test_no_build_global_score_in_collector(self):
        bgs_defs = re.findall(r"^def build_global_score", self.collector_src, re.MULTILINE)
        assert len(bgs_defs) == 0
