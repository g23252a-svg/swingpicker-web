# -*- coding: utf-8 -*-
"""
test_trade_plan.py — SSOT 체결 엔진 유닛/회귀 테스트 (pytest 표준)
═══════════════════════════════════════════════════════════════════
[v20.8] sys.exit/글로벌 카운터 제거 → 순수 pytest 스타일

실행: pytest test_trade_plan.py -v
"""
import random
import numpy as np
import pytest

import stop_logic as SL
from trade_plan import (
    TradePlan, ExecRule, BarResult,
    build_trade_plan, exec_bar, exec_multi_bar,
    validate_row, check_lookahead,
    REQUIRED_PLAN_KEYS,
)


@pytest.fixture(autouse=True)
def _reset_config():
    """매 테스트 전후 config를 normal로 리셋."""
    SL.switch_config(SL.config_normal())
    yield
    SL.switch_config(SL.config_normal())


# ═══════════════════════════════════════════════════
#  1. build_trade_plan 정상
# ═══════════════════════════════════════════════════

class TestBuildTradePlan:
    def test_entry_positive(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        assert plan.entry > 0

    def test_stop_below_entry(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        assert plan.stop < plan.entry

    def test_tp1_above_entry(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        assert plan.tp1 > plan.entry

    def test_stop_pct_positive(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        assert plan.stop_pct > 0

    def test_exec_rule_id(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        assert plan.exec_rule_id == ExecRule().rule_id


# ═══════════════════════════════════════════════════
#  2. Contract 검증
# ═══════════════════════════════════════════════════

class TestContract:
    def test_required_keys(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        row = plan.to_row()
        assert REQUIRED_PLAN_KEYS.issubset(set(row.keys()))

    def test_entry_zero_raises(self):
        with pytest.raises(ValueError):
            validate_row({"ENTRY_PRICE": 0})

    def test_stop_above_entry_raises(self):
        with pytest.raises(ValueError):
            validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 11000,
                          "TP1": 12000, "POSITION_PCT": 100,
                          "ENTRY_ACTION": "enter", "PLAN_REASON": "test",
                          "EXEC_RULE_ID": "v1"})

    def test_hold_nonzero_position_raises(self):
        with pytest.raises(ValueError):
            validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": 11000,
                          "POSITION_PCT": 50.0, "ENTRY_ACTION": "hold",
                          "PLAN_REASON": "test", "EXEC_RULE_ID": "v1",
                          "STOP_PCT": 5.0})

    def test_split_full_position_raises(self):
        with pytest.raises(ValueError):
            validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": 11000,
                          "POSITION_PCT": 100.0, "ENTRY_ACTION": "split",
                          "PLAN_REASON": "test", "EXEC_RULE_ID": "v1",
                          "STOP_PCT": 5.0})

    def test_tp2_below_tp1_raises(self):
        with pytest.raises(ValueError):
            validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": 11000,
                          "TP2": 10500, "TP3": 12000,
                          "POSITION_PCT": 100.0, "ENTRY_ACTION": "enter",
                          "PLAN_REASON": "test", "EXEC_RULE_ID": "v1",
                          "STOP_PCT": 5.0})

    def test_stop_pct_over_30_raises(self):
        with pytest.raises(ValueError):
            validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": 11000,
                          "POSITION_PCT": 100.0, "ENTRY_ACTION": "enter",
                          "PLAN_REASON": "test", "EXEC_RULE_ID": "v1",
                          "STOP_PCT": 35.0})

    def test_tp1_zero_raises(self):
        with pytest.raises(ValueError):
            validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": 0,
                          "POSITION_PCT": 100, "ENTRY_ACTION": "enter",
                          "PLAN_REASON": "test", "EXEC_RULE_ID": "v1", "STOP_PCT": 5.0})

    def test_tp1_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": None,
                          "POSITION_PCT": 100, "ENTRY_ACTION": "enter",
                          "PLAN_REASON": "test", "EXEC_RULE_ID": "v1", "STOP_PCT": 5.0})

    def test_frozen_immutable(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        with pytest.raises((AttributeError, TypeError)):
            plan.entry = 99999  # type: ignore


# ═══════════════════════════════════════════════════
#  3. exec_bar 장중 체결
# ═══════════════════════════════════════════════════

class TestExecBar:
    PLAN = TradePlan(entry=10000, stop=9500, tp1=11500, exec_rule_id="v1")
    RULE = ExecRule()

    def test_sl_hit(self):
        r = exec_bar(self.PLAN, 10100, 10200, 9400, 9600, self.RULE)
        assert r.action == "stop_hit"
        assert r.fill_price <= 9500
        assert r.return_pct < 0

    def test_tp_hit(self):
        r = exec_bar(self.PLAN, 10100, 11600, 10000, 11400, self.RULE)
        assert r.action == "tp_hit"
        assert r.fill_price < 11500  # slippage
        assert r.return_pct > 0

    def test_gap_down_sl(self):
        r = exec_bar(self.PLAN, 9200, 9300, 9100, 9250, self.RULE)
        assert r.action == "stop_hit"
        assert r.fill_price == 9200
        assert "gap_down" in r.reason

    def test_gap_up_tp(self):
        r = exec_bar(self.PLAN, 11700, 11800, 11600, 11750, self.RULE)
        assert r.action == "tp_hit"
        assert r.fill_price == 11700
        assert "gap_up" in r.reason

    def test_same_bar_sl_priority(self):
        r = exec_bar(self.PLAN, 10100, 11600, 9400, 10000, self.RULE)
        assert r.action == "stop_hit"
        assert "same_bar" in r.reason

    def test_same_bar_tp_priority(self):
        tp_rule = ExecRule(tp_sl_same_bar_priority="TP", rule_id="v2_tp_first")
        r = exec_bar(self.PLAN, 10100, 11600, 9400, 10000, tp_rule)
        assert r.action == "tp_hit"

    def test_hold_no_trigger(self):
        r = exec_bar(self.PLAN, 10100, 10200, 9600, 10050, self.RULE)
        assert r.action == "hold"
        assert r.fill_price == 0.0

    def test_limit_down(self):
        r = exec_bar(self.PLAN, 9300, 9300, 9300, 9300, self.RULE)
        assert r.action == "stop_hit"
        assert r.fill_price == 9300

    def test_limit_down_boundary(self):
        plan_limit = TradePlan(entry=10000, stop=9300, tp1=11500, exec_rule_id="v1")
        r = exec_bar(plan_limit, 9300, 9300, 9300, 9300, self.RULE)
        assert r.action == "stop_hit"
        assert "limit_down" in r.reason

    def test_invalid_bar(self):
        r = exec_bar(self.PLAN, 0, 0, 0, 0, self.RULE)
        assert r.action == "none"

    def test_invalid_plan(self):
        plan_bad = TradePlan(entry=0, stop=0, tp1=0, exec_rule_id="v1")
        r = exec_bar(plan_bad, 10000, 10100, 9900, 10050, self.RULE)
        assert r.action == "none"

    def test_gap_fill_level(self):
        level_rule = ExecRule(gap_fill="LEVEL", rule_id="v1_level")
        r = exec_bar(self.PLAN, 11700, 11800, 11600, 11750, level_rule)
        assert r.action == "tp_hit"
        assert r.fill_price <= self.PLAN.tp1

    def test_fee_reduces_return(self):
        r_fee = exec_bar(self.PLAN, 10100, 11600, 10000, 11400, self.RULE)
        no_fee_rule = ExecRule(fee_bps=0.0)
        r_no_fee = exec_bar(self.PLAN, 10100, 11600, 10000, 11400, no_fee_rule)
        assert r_fee.return_pct < r_no_fee.return_pct


# ═══════════════════════════════════════════════════
#  4. exec_multi_bar
# ═══════════════════════════════════════════════════

class TestExecMultiBar:
    PLAN = TradePlan(entry=10000, stop=9500, tp1=11500, exec_rule_id="v1")
    RULE = ExecRule()

    def test_multi_bar_sl(self):
        bars = [
            (10100, 10200, 9600, 10050),
            (10050, 10100, 9550, 9800),
            (9800, 9900, 9400, 9500),
            (9500, 9600, 9400, 9550),
        ]
        r = exec_multi_bar(self.PLAN, bars, self.RULE)
        assert r.action == "stop_hit"

    def test_timeout(self):
        bars_hold = [(10100, 10200, 9600, 10050)] * 5
        r = exec_multi_bar(self.PLAN, bars_hold, self.RULE, max_hold_days=5)
        assert r.action == "timeout"

    def test_scaleout(self):
        rule_scale = ExecRule(use_scaleout=True, rule_id="v_scaleout")
        plan_s = TradePlan(entry=10000, stop=9500, tp1=10500, tp2=11000, tp3=11500,
                           exec_rule_id=rule_scale.rule_id)
        bars_s = [
            (10000, 10600, 9900, 10500),
            (10500, 11100, 10400, 11000),
            (11000, 11600, 10800, 11400),
        ]
        r = exec_multi_bar(plan_s, bars_s, rule_scale, max_hold_days=5)
        assert "scaleout:" in r.reason
        assert np.isfinite(r.return_pct)
        assert r.return_pct > 0
        assert r.action == "tp_hit"

    def test_no_scaleout(self):
        rule_no = ExecRule(use_scaleout=False)
        plan_s = TradePlan(entry=10000, stop=9500, tp1=10500, tp2=11000, tp3=11500,
                           exec_rule_id="v1")
        bars_s = [
            (10000, 10600, 9900, 10500),
            (10500, 11100, 10400, 11000),
            (11000, 11600, 10800, 11400),
        ]
        r = exec_multi_bar(plan_s, bars_s, rule_no, max_hold_days=5)
        assert r.action == "tp_hit"


# ═══════════════════════════════════════════════════
#  5. 특수 케이스
# ═══════════════════════════════════════════════════

class TestSpecialCases:
    def test_tick_rounding(self):
        plan = build_trade_plan(buy=10333, atr_val=300, last_c=10333, mcap=50000)
        assert plan.entry % SL.tick_size(plan.entry) == 0
        assert plan.stop % SL.tick_size(plan.stop) == 0

    def test_position_sizing(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000,
                                mcap=50000, account_risk_pct=0.5)
        assert 0 < plan.position_pct <= 100

    def test_extreme_gap_hold(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000,
                                mcap=50000, ret_1d=16.0, gap_pct=13.0)
        assert plan.entry_action == "hold"
        assert plan.position_pct == 0.0

    def test_regime_switch(self):
        SL.switch_config(SL.config_normal())
        plan_n = build_trade_plan(buy=10000, atr_val=500, last_c=10000, mcap=50000)
        SL.switch_config(SL.config_high_vol())
        plan_h = build_trade_plan(buy=10000, atr_val=500, last_c=10000, mcap=50000)
        assert plan_n.rr_mult != plan_h.rr_mult

    def test_regime_name_propagation(self):
        SL.switch_config(SL.config_high_vol())
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        assert plan.regime == "high_vol"

    def test_ef_reason_in_plan(self):
        SL.switch_config(SL.config_normal())
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        assert "EF:" in plan.plan_reason

    def test_exec_rule_id_tracking(self):
        rule_v2 = ExecRule(rule_id="v2_tp_first", tp_sl_same_bar_priority="TP")
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000,
                                mcap=50000, exec_rule=rule_v2)
        assert plan.exec_rule_id == "v2_tp_first"

    def test_rr_ssot(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        risk = plan.entry - plan.stop
        if risk > 0:
            implied_rr = (plan.tp1 - plan.entry) / risk
            assert abs(implied_rr - plan.rr_mult) < 0.5

    def test_tp_ordering(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
        assert plan.tp1 < plan.tp2
        assert plan.tp2 < plan.tp3

    def test_min_order_skip(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000,
                                account_value_krw=50000)
        assert plan.entry_action == "hold"
        assert plan.position_pct == 0.0
        assert "MIN_ORDER_SKIP" in plan.plan_reason

    def test_min_order_sufficient(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000,
                                account_value_krw=10000000)
        assert plan.entry_action == "enter"

    def test_pm_block_intraday_tag(self):
        SL.switch_config(SL.config_normal())
        _, _, _, reason = SL.calc_stop_price(buy=10000, atr_val=300, mcap=50000,
                                             today_low=9000, gap_up_pct=10.0)
        assert "PM_BLOCK_INTRADAY" in reason


# ═══════════════════════════════════════════════════
#  6. Lookahead
# ═══════════════════════════════════════════════════

class TestLookahead:
    def test_future_data_detected(self):
        assert len(check_lookahead("20260210", ["20260209", "20260210", "20260211"])) == 1

    def test_normal_data_passes(self):
        assert len(check_lookahead("20260210", ["20260208", "20260210"])) == 0

    def test_yyyymmdd_format(self):
        assert len(check_lookahead(20260210, [20260211])) == 1

    def test_dash_format(self):
        assert len(check_lookahead("2026-02-10", ["2026-02-11"])) == 1

    def test_mixed_format(self):
        assert len(check_lookahead("20260210", ["2026-02-11"])) == 1

    def test_past_passes(self):
        assert len(check_lookahead("2026-02-10", ["2026-02-09"])) == 0


# ═══════════════════════════════════════════════════
#  7. Property 테스트 (랜덤 불변조건)
# ═══════════════════════════════════════════════════

class TestProperty:
    def test_build_invariants_50_cases(self):
        """랜덤 50 케이스에서 불변조건 유지."""
        random.seed(42)
        failures = 0
        for _ in range(50):
            buy_r = random.uniform(1000, 500000)
            atr_r = buy_r * random.uniform(0.005, 0.10)
            mcap_r = random.uniform(100, 200000)
            try:
                p = build_trade_plan(buy=buy_r, atr_val=atr_r, last_c=buy_r, mcap=mcap_r)
                assert p.stop < p.entry
                assert p.tp1 > p.entry
                assert p.tp1 < p.tp2 < p.tp3
                assert 0 < p.stop_pct < 30
            except Exception:
                failures += 1
        assert failures == 0, f"{failures}/50 invariant violations"

    def test_tp_fill_level_invariant(self):
        """tp_hit → fill ≤ tp (LEVEL) 50 케이스."""
        random.seed(42)
        level_rule = ExecRule(gap_fill="LEVEL", rule_id="v1_level")
        violations = 0
        for _ in range(50):
            entry_r = random.uniform(5000, 100000)
            risk_r = entry_r * random.uniform(0.01, 0.08)
            stop_r = entry_r - risk_r
            tp_r = entry_r + risk_r * random.uniform(1.5, 4.0)
            p = TradePlan(entry=entry_r, stop=stop_r, tp1=tp_r, exec_rule_id="v1")
            hi = tp_r * random.uniform(1.0, 1.05)
            lo = stop_r * random.uniform(0.99, 1.05)
            r = exec_bar(p, entry_r, hi, lo, entry_r, level_rule)
            if r.action == "tp_hit" and r.fill_price > tp_r + 0.01:
                violations += 1
        assert violations == 0


# ═══════════════════════════════════════════════════
#  8. [v20.8] PolicyConfig E2E via trade_plan
# ═══════════════════════════════════════════════════

class TestPolicyE2E:
    def test_rsi_split_via_trade_plan(self):
        from collector_config import DEFAULT_CONFIG
        p = DEFAULT_CONFIG.policy
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000,
                                tv_eok=100, gap_pct=2, ret_1d=1,
                                rsi14=p.entry_rsi_split + 1)
        assert plan.entry_action == "split"

    def test_consecutive_limit_up_hold(self):
        plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000,
                                tv_eok=100, gap_pct=2, ret_1d=1,
                                consecutive_limit_up=2)
        assert plan.entry_action == "hold"
