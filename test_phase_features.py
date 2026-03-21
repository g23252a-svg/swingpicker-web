# -*- coding: utf-8 -*-
"""test_phase_features.py — Phase 1+2 기능별 단위 테스트 (pytest 표준)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Time Stop / Hard Block / 슬리피지 / 트레일링 / 전략팩토리 / 캘리브레이션
실행: pytest test_phase_features.py -v
"""
import numpy as np
import pandas as pd
import pytest

from collector_config import DEFAULT_CONFIG, CollectorConfig
from trade_plan import (
    TradePlan, ExecRule, exec_multi_bar, BarResult,
    build_trade_plan, estimate_slippage_bps,
)
from validation import apply_hard_blocks, block_summary, HardBlockRule, HARD_BLOCK_RULES
from strategies import StrategyFactory, select_strategies
from stop_logic import check_entry_defense, ENTRY_DEFENSE_RULES


CFG = DEFAULT_CONFIG


# ═══════════════════════════════════════════════
#  1. Config SSOT — WEIGHT_CONFIG 완전 제거 확인
# ═══════════════════════════════════════════════
class TestConfigSSOT:
    def test_ml_low_field(self):
        assert hasattr(DEFAULT_CONFIG, "ml_low")

    def test_macro_weights_field(self):
        assert hasattr(DEFAULT_CONFIG, "macro_weights")

    def test_to_weight_config_dict_removed(self):
        assert not hasattr(DEFAULT_CONFIG, "to_weight_config_dict")

    def test_snapshot_works(self):
        assert isinstance(DEFAULT_CONFIG.snapshot(), dict)

    def test_snapshot_json_works(self):
        assert isinstance(DEFAULT_CONFIG.snapshot_json(), str)

    def test_no_weight_config_var_in_scoring_engine(self):
        import scoring_engine
        src = open(scoring_engine.__file__).read()
        has_wc = "WEIGHT_CONFIG =" in src or "WEIGHT_CONFIG=" in src
        assert not has_wc


# ═══════════════════════════════════════════════
#  2. Time Stop 단위 테스트
# ═══════════════════════════════════════════════
class TestTimeStop:
    def test_inactive_timeout(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000, time_stop_days=0)
        r = exec_multi_bar(plan, [(10000, 10100, 9900, 10050)] * 20, max_hold_days=20)
        assert r.action == "timeout"

    def test_flat_movement_triggers_time_stop(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000,
                         time_stop_days=5, time_stop_min_move_pct=2.0,
                         time_stop_extend_if_profit=False)
        r = exec_multi_bar(plan, [(10000, 10100, 9900, 10050)] * 10, max_hold_days=10)
        assert r.action == "time_stop"
        assert "time_stop_5d" in r.reason

    def test_profit_extension(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000,
                         time_stop_days=3, time_stop_min_move_pct=2.0,
                         time_stop_extend_if_profit=True)
        bars = [(10000, 10200, 9950, 10150), (10150, 10350, 10100, 10300),
                (10300, 10500, 10250, 10400), (10400, 10600, 10350, 10500)]
        r = exec_multi_bar(plan, bars, max_hold_days=4)
        assert r.action != "time_stop"

    def test_sl_takes_priority(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000,
                         time_stop_days=7, time_stop_min_move_pct=2.0)
        bars = [(10000, 10100, 9900, 10000), (10000, 10050, 9100, 9150)]
        r = exec_multi_bar(plan, bars, max_hold_days=7)
        assert r.action == "stop_hit"

    def test_auto_injection(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000)
        assert plan.time_stop_days == CFG.time_stop_days


# ═══════════════════════════════════════════════
#  3. 동적 슬리피지
# ═══════════════════════════════════════════════
class TestSlippage:
    def test_high_liquidity(self):
        assert estimate_slippage_bps(100.0, CFG) == CFG.slippage_base_bps

    def test_monotonic_decrease(self):
        s_high = estimate_slippage_bps(100.0, CFG)
        s_mid = estimate_slippage_bps(10.0, CFG)
        s_low = estimate_slippage_bps(3.0, CFG)
        s_none = estimate_slippage_bps(None, CFG)
        assert s_none >= s_low >= s_mid >= s_high

    def test_none_equals_max(self):
        s_none = estimate_slippage_bps(None, CFG)
        assert s_none == CFG.slippage_base_bps * CFG.slippage_low_liq_mult

    def test_zero_equals_none(self):
        assert estimate_slippage_bps(0, CFG) == estimate_slippage_bps(None, CFG)


# ═══════════════════════════════════════════════
#  4. Hard Block
# ═══════════════════════════════════════════════
class TestHardBlock:
    def test_normal_passes(self):
        df = pd.DataFrame({"ticker": ["A"], "ret_5d_%": [5.0], "거래대금(억)": [50.0],
                           "gap_pct": [1.0], "RSI14": [55.0], "_data_length": [120]})
        p, b = apply_hard_blocks(df)
        assert len(p) == 1 and len(b) == 0

    def test_surge_blocked(self):
        df = pd.DataFrame({"ret_5d_%": [45.0], "거래대금(억)": [50.0],
                           "gap_pct": [1.0], "RSI14": [55.0], "_data_length": [120]})
        p, b = apply_hard_blocks(df)
        assert len(b) == 1
        assert "연속급등" in str(b["BLOCK_REASON"].iloc[0])

    def test_multi_violation(self):
        df = pd.DataFrame({"ret_5d_%": [45.0], "거래대금(억)": [2.0],
                           "gap_pct": [1.0], "RSI14": [90.0], "_data_length": [120]})
        _, b = apply_hard_blocks(df)
        assert len(b) == 1
        assert str(b["BLOCK_REASON"].iloc[0]).count("[") >= 3

    def test_empty_df(self):
        p, b = apply_hard_blocks(pd.DataFrame())
        assert len(p) == 0 and len(b) == 0

    def test_block_summary(self):
        df = pd.DataFrame({"ret_5d_%": [45.0], "거래대금(억)": [2.0],
                           "gap_pct": [1.0], "RSI14": [90.0], "_data_length": [120]})
        _, b = apply_hard_blocks(df)
        s = block_summary(b)
        assert s["total_blocked"] >= 1


# ═══════════════════════════════════════════════
#  5. 트레일링 스탑
# ═══════════════════════════════════════════════
class TestTrailingStop:
    RULE_ON = ExecRule(trailing_stop_enabled=True,
                       trailing_stop_trigger_pct=3.0, trailing_stop_distance_pct=2.0)

    def test_trailing_triggers(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000)
        bars = [(10000, 10200, 9900, 10100), (10100, 10400, 10050, 10350),
                (10350, 10500, 10300, 10400), (10400, 10420, 10200, 10250)]
        r = exec_multi_bar(plan, bars, rule=self.RULE_ON)
        assert r.action == "trailing_stop"
        assert r.return_pct > 0

    def test_trailing_off(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000)
        bars = [(10000, 10200, 9900, 10100), (10100, 10400, 10050, 10350),
                (10350, 10500, 10300, 10400), (10400, 10420, 10200, 10250)]
        r = exec_multi_bar(plan, bars, rule=ExecRule(trailing_stop_enabled=False))
        assert r.action != "trailing_stop"

    def test_tp_takes_priority(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000)
        bars = [(10000, 10400, 9900, 10300), (10300, 12100, 10200, 12050)]
        r = exec_multi_bar(plan, bars, rule=self.RULE_ON)
        assert r.action == "tp_hit"


# ═══════════════════════════════════════════════
#  6. 전략 팩토리
# ═══════════════════════════════════════════════
class TestStrategyFactory:
    def test_strategy_selection(self):
        assert set(select_strategies("NORMAL", 60)) == {"breakout", "pullback"}
        assert select_strategies("NORMAL", 45) == ["pullback"]
        assert select_strategies("NORMAL", 30) == ["mean_revert"]
        assert select_strategies("CRITICAL", 60) == []

    def test_factory_creation(self):
        for name in StrategyFactory.available():
            strat = StrategyFactory.create(name)
            assert hasattr(strat, "filter") and hasattr(strat, "score")

    def test_pipeline(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "종목코드": [f"{i:06d}" for i in range(30)],
            "TTM_SQUEEZE": np.random.choice([0, 1], 30),
            "BB_Expanding": np.random.choice([0, 1], 30),
            "Vol_Quality": np.random.uniform(0.5, 3.0, 30),
            "IS_ABOVE_POC": np.random.choice([0, 1], 30),
            "Above_MA20": np.random.choice([0, 1], 30),
            "MACD_Slope_PCT": np.random.uniform(-5, 5, 30),
            "Low_Trend_PCT": np.random.uniform(-3, 5, 30),
            "이격도": np.random.uniform(-5, 10, 30),
            "RSI14": np.random.uniform(20, 80, 30),
            "MTF_WEEKLY_TREND": np.random.choice([-1, 0, 1], 30),
            "ret_5d_%": np.random.uniform(-10, 15, 30),
            "STRUCT_SCORE": np.random.uniform(30, 90, 30),
            "TIMING_SCORE": np.random.uniform(20, 95, 30),
            "AI_SCORE": np.random.uniform(0, 80, 30),
            "FINAL_SCORE": np.random.uniform(40, 95, 30),
        })
        strat = StrategyFactory.create("breakout")
        filtered = strat.filter(df)
        scored = strat.score(filtered)
        picks = strat.rank_and_pick(scored, top_k=3)
        if not picks.empty:
            assert "STRATEGY" in picks.columns
            assert "STRATEGY_SCORE" in picks.columns


# ═══════════════════════════════════════════════
#  7. 방어 규칙
# ═══════════════════════════════════════════════
class TestEntryDefense:
    def test_rule_count(self):
        assert len(ENTRY_DEFENSE_RULES) == 8

    def test_normal_enter(self):
        r = check_entry_defense({"gap_pct": 1, "ret_1d_%": 2, "RSI14": 55, "거래대금(억)": 50})
        assert r["action"] == "enter"

    def test_gap_12_hold(self):
        assert check_entry_defense({"gap_pct": 13})["action"] == "hold"

    def test_vi_hold(self):
        assert check_entry_defense({"is_vi_triggered": True})["action"] == "hold"

    def test_low_liquidity_hold(self):
        assert check_entry_defense({"거래대금(억)": 3})["action"] == "hold"

    def test_rsi_85_split(self):
        r = check_entry_defense({"RSI14": 85})
        assert r["action"] == "split" and r["position_pct"] == 50


# ═══════════════════════════════════════════════
#  8. 경계값/회귀 테스트
# ═══════════════════════════════════════════════
class TestEdgeCases:
    # -- Time Stop 경계 --
    def test_ts_4_days_no_trigger(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000,
                         time_stop_days=5, time_stop_min_move_pct=2.0,
                         time_stop_extend_if_profit=False)
        r = exec_multi_bar(plan, [(10000, 10100, 9900, 10050)] * 4, max_hold_days=4)
        assert r.action == "timeout"

    def test_ts_exact_below_threshold(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000,
                         time_stop_days=3, time_stop_min_move_pct=2.0,
                         time_stop_extend_if_profit=False)
        # +1.99% < 2.0%
        bars = [(10000, 10200, 9900, 10050), (10050, 10200, 9950, 10100),
                (10100, 10300, 10050, 10199)]
        assert exec_multi_bar(plan, bars, max_hold_days=3).action == "time_stop"

    def test_ts_exact_at_threshold(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000,
                         time_stop_days=3, time_stop_min_move_pct=2.0,
                         time_stop_extend_if_profit=False)
        # +2.0% = threshold → 통과
        bars = [(10000, 10200, 9900, 10050), (10050, 10200, 9950, 10100),
                (10100, 10300, 10050, 10200)]
        assert exec_multi_bar(plan, bars, max_hold_days=3).action == "timeout"

    def test_ts_1_day(self):
        plan = TradePlan(entry=10000, stop=9200, tp1=12000,
                         time_stop_days=1, time_stop_min_move_pct=2.0,
                         time_stop_extend_if_profit=False)
        assert exec_multi_bar(plan, [(10000, 10100, 9900, 10050)],
                              max_hold_days=1).action == "time_stop"

    # -- 슬리피지 경계 --
    def test_slippage_at_threshold(self):
        assert estimate_slippage_bps(CFG.slippage_liq_threshold_eok, CFG) == CFG.slippage_base_bps

    def test_slippage_just_below_threshold(self):
        assert estimate_slippage_bps(CFG.slippage_liq_threshold_eok - 0.01, CFG) > CFG.slippage_base_bps

    def test_slippage_negative(self):
        assert estimate_slippage_bps(-10.0, CFG) == CFG.slippage_base_bps * CFG.slippage_low_liq_mult

    def test_slippage_huge(self):
        assert estimate_slippage_bps(100000.0, CFG) == CFG.slippage_base_bps

    # -- Hard Block 경계 --
    def test_rsi_85_passes(self):
        df = pd.DataFrame({"RSI14": [85.0], "거래대금(억)": [50], "_data_length": [120],
                           "ret_5d_%": [5], "gap_pct": [1]})
        p, _ = apply_hard_blocks(df)
        assert len(p) == 1  # gt, not gte

    def test_rsi_85_01_blocked(self):
        df = pd.DataFrame({"RSI14": [85.01], "거래대금(억)": [50], "_data_length": [120],
                           "ret_5d_%": [5], "gap_pct": [1]})
        _, b = apply_hard_blocks(df)
        assert len(b) == 1

    def test_tv_30_passes(self):
        df = pd.DataFrame({"거래대금(억)": [30.0], "RSI14": [50], "_data_length": [120],
                           "ret_5d_%": [5], "gap_pct": [1]})
        p, _ = apply_hard_blocks(df)
        assert len(p) == 1

    def test_tv_29_99_blocked(self):
        df = pd.DataFrame({"거래대금(억)": [29.99], "RSI14": [50], "_data_length": [120],
                           "ret_5d_%": [5], "gap_pct": [1]})
        _, b = apply_hard_blocks(df)
        assert len(b) == 1

    def test_all_nan_no_crash(self):
        df = pd.DataFrame({"RSI14": [np.nan], "거래대금(억)": [np.nan], "_data_length": [np.nan],
                           "ret_5d_%": [np.nan], "gap_pct": [np.nan]})
        p, b = apply_hard_blocks(df)
        assert len(p) + len(b) == 1

    # -- 트레일링 스탑 경계 --
    def test_trailing_exact_trigger(self):
        rule = ExecRule(trailing_stop_enabled=True,
                        trailing_stop_trigger_pct=3.0, trailing_stop_distance_pct=2.0)
        plan = TradePlan(entry=10000, stop=9200, tp1=12000)
        bars = [(10000, 10300, 9950, 10300),
                (10300, 10310, 10090, 10100)]
        assert exec_multi_bar(plan, bars, rule=rule).action == "trailing_stop"

    def test_trailing_below_trigger(self):
        rule = ExecRule(trailing_stop_enabled=True,
                        trailing_stop_trigger_pct=3.0, trailing_stop_distance_pct=2.0)
        plan = TradePlan(entry=10000, stop=9200, tp1=12000)
        bars = [(10000, 10290, 9950, 10290),
                (10290, 10300, 10050, 10050)]
        assert exec_multi_bar(plan, bars, rule=rule).action != "trailing_stop"
