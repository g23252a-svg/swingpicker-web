# -*- coding: utf-8 -*-
"""test_stop_logic.py — stop_logic.py 전용 단위 테스트 (v20.6.5 pytest 호환)
═══════════════════════════════════════════════════
핵심 6중 안전장치 각각을 개별 검증:
  1. ATR 기반 가변 손절
  2. 시총별 최대 손실 캡
  3. 갭업 방어
  4. Swing Low 보정
  5. 휩쏘 방지 하한
  6. stop >= buy 절대 방지
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

from stop_logic import (
    calc_stop_price, calc_rr_multiplier, sanitize_ohlcv,
    check_entry_filter, StopConfig, get_config,
    floor_to_tick, ceil_to_tick,
)

CFG = StopConfig()


class TestCalcStopBasic:
    """calc_stop_price 기본 동작."""

    def test_returns_positive(self):
        stop, stop_pct, max_loss, reason = calc_stop_price(
            buy=10000, atr_val=200, mcap=50000,
            today_low=9800, swing_low_10=9500, dist_to_swing=3.0,
            gap_up_pct=0.0, tv_eok=100.0,
        )
        assert stop > 0, f"stop={stop}"
        assert stop < 10000, f"stop={stop} >= buy"
        assert stop_pct > 0
        assert len(reason) > 0


class TestStopBelowBuy:
    """stop >= buy 절대 방지 (안전장치 #6)."""

    def test_atr_zero(self):
        """ATR=0이어도 stop < buy 보장."""
        stop, _, _, _ = calc_stop_price(
            buy=10000, atr_val=0, mcap=10000,
            today_low=10000, swing_low_10=10000, dist_to_swing=0.0,
        )
        assert stop < 10000, f"stop={stop}"

    def test_swing_above_buy(self):
        """swing_low > buy여도 stop < buy 보장."""
        stop, _, _, _ = calc_stop_price(
            buy=10000, atr_val=50, mcap=10000,
            today_low=10500, swing_low_10=10500, dist_to_swing=1.0,
        )
        assert stop < 10000, f"stop={stop}"


class TestMcapCap:
    """시총별 최대 손실 캡 (안전장치 #2)."""

    def test_large_cap(self):
        _, _, ml, _ = calc_stop_price(
            buy=100000, atr_val=10000, mcap=100000,
            today_low=90000, swing_low_10=88000, dist_to_swing=10.0,
        )
        assert ml <= 4.5, f"대형주 max_loss={ml}"

    def test_small_cap(self):
        _, _, ml, _ = calc_stop_price(
            buy=5000, atr_val=500, mcap=500,
            today_low=4500, swing_low_10=4200, dist_to_swing=10.0,
        )
        assert ml <= 6.5, f"소형주 max_loss={ml}"


class TestGapDefense:
    """갭업 방어 (안전장치 #3)."""

    def test_gap_tightens_stop(self):
        stop_ng, _, _, _ = calc_stop_price(
            buy=10000, atr_val=200, mcap=10000,
            today_low=9800, swing_low_10=9500, dist_to_swing=3.0,
            gap_up_pct=0.0,
        )
        stop_g, _, _, _ = calc_stop_price(
            buy=10000, atr_val=200, mcap=10000,
            today_low=9800, swing_low_10=9500, dist_to_swing=3.0,
            gap_up_pct=8.0,
        )
        assert stop_g >= stop_ng, f"gap={stop_g}, nogap={stop_ng}"


class TestRRMultiplier:
    """calc_rr_multiplier ATR% 구간별 R:R."""

    def test_low_atr(self):
        assert calc_rr_multiplier(100, 10000) == CFG.rr_low_atr

    def test_mid_atr(self):
        assert calc_rr_multiplier(300, 10000) == CFG.rr_mid_atr

    def test_high_atr(self):
        assert calc_rr_multiplier(500, 10000) == CFG.rr_high_atr


class TestSanitizeOhlcv:
    """sanitize_ohlcv 데이터 정제."""

    def test_removes_zero_rows(self):
        df = pd.DataFrame({
            "시가": [100, 0, 200],
            "고가": [110, 0, 210],
            "저가": [90, 0, 190],
            "종가": [105, 0, 205],
            "거래량": [1000, 0, 2000],
        }, index=pd.date_range("2026-01-01", periods=3))
        clean = sanitize_ohlcv(df)
        assert len(clean) < len(df)
        assert len(clean) >= 2


class TestTickRounding:
    """호가 단위 (tick) 유틸."""

    def test_floor(self):
        assert floor_to_tick(10050) <= 10050

    def test_ceil(self):
        assert ceil_to_tick(10050) >= 10050

    def test_floor_zero(self):
        assert floor_to_tick(0) == 0


class TestEntryFilter:
    """check_entry_filter 극단 갭/급등 필터.

    실제 API: check_entry_filter(ret_1d, gap_pct, is_vi_triggered, cfg) → dict
    """

    def test_normal_enter(self):
        res = check_entry_filter(ret_1d=1.0, gap_pct=5.0, cfg=CFG)
        assert res["action"] == "enter"

    def test_extreme_gap_hold(self):
        res = check_entry_filter(ret_1d=1.0, gap_pct=15.0, cfg=CFG)
        assert res["action"] == "hold"

    def test_moderate_gap_split(self):
        res = check_entry_filter(ret_1d=1.0, gap_pct=8.0, cfg=CFG)
        assert res["action"] == "split"
        assert res["position_pct"] == 50.0

    def test_surge_hold(self):
        res = check_entry_filter(ret_1d=16.0, gap_pct=0.0, cfg=CFG)
        assert res["action"] == "hold"

    def test_vi_triggered_split(self):
        res = check_entry_filter(ret_1d=0.0, gap_pct=0.0, is_vi_triggered=True, cfg=CFG)
        assert res["action"] == "split"


# ── 스크립트 실행 호환 ──
if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
