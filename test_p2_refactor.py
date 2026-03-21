# -*- coding: utf-8 -*-
"""test_p2_refactor.py — P2 개선과제 #9~#12 통합 테스트 (pytest 표준)
═══════════════════════════════════════════════════
실행: pytest test_p2_refactor.py -v
"""
import os
import re
import tempfile
import shutil
import pickle

import numpy as np
import pandas as pd
import pytest

from collector_config import CollectorConfig, DEFAULT_CONFIG


# ═══════════════════════════════════════════════
#  1. Config 중앙화 (#11)
# ═══════════════════════════════════════════════
class TestConfigCentralization:
    def test_default_config_instance(self):
        assert isinstance(DEFAULT_CONFIG, CollectorConfig)

    def test_rsi_low(self):
        assert DEFAULT_CONFIG.rsi_low == 45.0

    def test_rsi_high(self):
        assert DEFAULT_CONFIG.rsi_high == 65.0

    def test_rsi_overheat(self):
        assert DEFAULT_CONFIG.rsi_overheat == 75.0

    def test_pass_ebs(self):
        assert DEFAULT_CONFIG.pass_ebs == 4

    def test_min_mcap_eok(self):
        assert DEFAULT_CONFIG.min_mcap_eok == 1000

    def test_out_dir_property(self):
        assert DEFAULT_CONFIG.out_dir.endswith("data")

    def test_rsi_range_tuple(self):
        assert DEFAULT_CONFIG.rsi_range == (45.0, 65.0)

    def test_custom_config_via_sub_configs(self):
        """현행 CollectorConfig는 중첩 설정 객체 사용."""
        from collector_config import MacroConfig
        custom = CollectorConfig(macro=MacroConfig(ml_max_weight=0.30))
        assert custom.macro.ml_max_weight == 0.30

    def test_weight_sum_approx_one(self):
        w_sum = (DEFAULT_CONFIG.w_rr + DEFAULT_CONFIG.w_t1 + DEFAULT_CONFIG.w_sl +
                 DEFAULT_CONFIG.w_near + DEFAULT_CONFIG.w_mom + DEFAULT_CONFIG.w_liq +
                 DEFAULT_CONFIG.w_tec)
        assert abs(w_sum - 1.0) < 0.01, f"sum={w_sum}"


# ═══════════════════════════════════════════════
#  2. DataSource 추상화 (#12)
# ═══════════════════════════════════════════════
class TestDataSourceAbstraction:
    def test_krx_data_source_creation(self):
        from data_source import KRXDataSource
        ds = KRXDataSource()
        assert ds is not None
        assert hasattr(ds, '_pykrx_ok')
        assert hasattr(ds, '_fdr_ok')

    def test_singleton(self):
        from data_source import get_data_source
        ds1 = get_data_source()
        ds2 = get_data_source()
        assert ds1 is ds2


# ═══════════════════════════════════════════════
#  3. Parquet 캐시 (#10)
# ═══════════════════════════════════════════════
class TestParquetCache:
    @pytest.fixture()
    def cache_dir(self):
        tmp = tempfile.mkdtemp()
        yield tmp
        shutil.rmtree(tmp, ignore_errors=True)

    def test_save_and_load(self, cache_dir):
        from data_source import OHLCVCache
        cache = OHLCVCache(cache_dir, fmt="parquet")

        data = {
            "005930": pd.DataFrame({"Close": [70000, 71000, 72000], "Volume": [100, 200, 300]}),
            "000660": pd.DataFrame({"Close": [150000, 151000], "Volume": [50, 60]}),
        }
        cache.save("20260222", data)
        # 현행: 단일 통합 파일로 저장 (디렉토리가 아님)
        assert cache.exists("20260222")

        loaded = cache.load("20260222")
        assert len(loaded) == 2
        assert loaded["005930"]["Close"].tolist() == [70000, 71000, 72000]
        assert len(loaded["000660"]) == 2

    def test_exists(self, cache_dir):
        from data_source import OHLCVCache
        cache = OHLCVCache(cache_dir, fmt="parquet")
        data = {"005930": pd.DataFrame({"Close": [1]})}
        cache.save("20260222", data)
        assert cache.exists("20260222")
        assert not cache.exists("99999999")

    def test_empty_date_returns_empty(self, cache_dir):
        from data_source import OHLCVCache
        cache = OHLCVCache(cache_dir, fmt="parquet")
        assert cache.load("20260101") == {}

    def test_pickle_fallback_allowed(self, cache_dir):
        from data_source import OHLCVCache
        pkl_path = os.path.join(cache_dir, "ohlcv_cache_20260220.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"999999": pd.DataFrame({"x": [1]})}, f)

        cache = OHLCVCache(cache_dir, fmt="parquet", allow_legacy_pickle=True)
        loaded = cache.load("20260220")
        assert len(loaded) == 1 and "999999" in loaded

    def test_pickle_blocked_by_default(self, cache_dir):
        from data_source import OHLCVCache
        pkl_path = os.path.join(cache_dir, "ohlcv_cache_20260220.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"999999": pd.DataFrame({"x": [1]})}, f)

        cache = OHLCVCache(cache_dir, fmt="parquet", allow_legacy_pickle=False)
        assert cache.load("20260220") == {}


# ═══════════════════════════════════════════════
#  4. 모듈 분할 (#9)
# ═══════════════════════════════════════════════
class TestModuleSplit:
    def test_macro_filter_import(self):
        from macro_filter import check_macro_env, compute_market_breadth, label_market_temp
        assert "과열" in label_market_temp(70)
        assert "침체" in label_market_temp(30)

    def test_market_breadth(self):
        from macro_filter import compute_market_breadth
        breadth = compute_market_breadth(pd.DataFrame({"ret_1d_%": [1, -1, 2, -0.5, 3]}))
        assert breadth["ALL"] == 60.0

    def test_news_engine_import(self):
        from news_engine import fetch_naver_news_headlines, analyze_sentiment_llm, get_naver_theme_tags
        assert fetch_naver_news_headlines is not None

    def test_telegram_import(self):
        from telegram_sender import send_telegram_auto
        assert send_telegram_auto is not None

    def test_validation_import(self):
        from validation import (list_snapshot_days, load_close_map, load_price_maps,
                                pick_recommend_file_per_day, run_reality_check)
        assert list_snapshot_days is not None


# ═══════════════════════════════════════════════
#  5. 순환 import 방어
# ═══════════════════════════════════════════════
class TestCircularImportDefense:
    @pytest.mark.parametrize("mod_name,forbidden", [
        ("collector_config", ["collector", "data_source", "macro_filter"]),
        ("data_source", ["collector", "macro_filter", "news_engine"]),
        ("macro_filter", ["collector", "news_engine", "telegram_sender"]),
    ])
    def test_no_forbidden_imports(self, mod_name, forbidden):
        mod_file = os.path.join(os.path.dirname(__file__), f"{mod_name}.py")
        src = open(mod_file).read()
        for f in forbidden:
            has_import = (f"import {f} " in src or f"import {f}\n" in src or
                         f"from {f} " in src or f"from {f}." in src)
            assert not has_import, f"found '{f}' import in {mod_name}"


# ═══════════════════════════════════════════════
#  6. Config 타입 안전성
# ═══════════════════════════════════════════════
class TestConfigTypeSafety:
    def test_has_many_properties(self):
        """CollectorConfig는 충분한 설정 속성을 갖고 있어야 함."""
        attrs = [a for a in dir(DEFAULT_CONFIG) if not a.startswith('_')]
        assert len(attrs) > 20, f"got {len(attrs)}"

    def test_rsi_overheat_is_numeric(self):
        assert isinstance(DEFAULT_CONFIG.rsi_overheat, (int, float))


# ═══════════════════════════════════════════════
#  7. 매직넘버 잔존 자동검출 (#11)
# ═══════════════════════════════════════════════
class TestMagicNumberDetection:
    @pytest.fixture(autouse=True)
    def _load_sources(self):
        base = os.path.dirname(__file__)
        self.collector_src = open(os.path.join(base, "collector.py")).read()
        self.se_src = open(os.path.join(base, "scoring_engine.py")).read()

    @pytest.mark.parametrize("pattern,desc", [
        (r'\bRSI_LOW\s*=\s*\d', "RSI_LOW 재정의"),
        (r'\bRSI_HIGH\s*=\s*\d', "RSI_HIGH 재정의"),
        (r'\bPASS_EBS\s*=\s*\d', "PASS_EBS 재정의"),
        (r'\bMIN_MCAP_EOK\s*=\s*\d', "MIN_MCAP_EOK 재정의"),
        (r'\bBB_PERIOD\s*=\s*\d', "BB_PERIOD 재정의"),
    ])
    def test_no_hardcoded_in_collector(self, pattern, desc):
        matches = re.findall(pattern, self.collector_src)
        real = [m for m in matches if "_CFG" not in m]
        assert len(real) == 0, f"{desc}: found {len(real)}"

    @pytest.mark.parametrize("pattern,desc", [
        (r'\bRSI_LOW\s*=\s*\d', "RSI_LOW in scoring_engine"),
        (r'\bMIN_MCAP\s*=\s*\d', "MIN_MCAP in scoring_engine"),
    ])
    def test_no_hardcoded_in_scoring_engine(self, pattern, desc):
        assert len(re.findall(pattern, self.se_src)) == 0, desc
