# -*- coding: utf-8 -*-
"""test_run_health.py — RunHealth 신뢰도/행동 상한 테스트 (pytest 표준)
═══════════════════════════════════════════════════════
[v20.8] unittest → pytest 변환
실행: pytest test_run_health.py -v
"""
import numpy as np
import pandas as pd
import pytest

from run_health import RunHealth, check_run_health


# ═══════════════════════════════════════════════
#  1. DEGRADED/CRITICAL/OK 상태별 행동 상한
# ═══════════════════════════════════════════════
class TestMaxAllowedRoute:
    def test_ok_high_confidence_allows_attack(self):
        h = RunHealth()
        h.add_ok("MCAP")
        h.add_ok("BENCH")
        assert h.status == "OK"
        assert h.max_allowed_route == "ATTACK"

    def test_degraded_always_caps_at_armed(self):
        h = RunHealth()
        h.add_issue("FLOW_ZERO")
        assert h.status == "DEGRADED"
        assert h.confidence_score >= 70
        assert h.max_allowed_route == "ARMED"

    def test_degraded_many_issues_caps_at_wait(self):
        h = RunHealth()
        for code in ["MCAP_EMPTY", "BENCH_FAIL", "FLOW_ZERO", "NEWS_OFF"]:
            h.add_issue(code)
        assert h.status == "DEGRADED"
        assert h.confidence_score < 50
        assert h.max_allowed_route == "WAIT"

    def test_critical_always_wait(self):
        h = RunHealth()
        h.add_issue("FATAL_ERROR", severity="CRITICAL")
        assert h.status == "CRITICAL"
        assert h.max_allowed_route == "WAIT"

    def test_degraded_two_issues_armed(self):
        h = RunHealth()
        h.add_issue("FLOW_ZERO")
        h.add_issue("NEWS_OFF")
        assert h.status == "DEGRADED"
        assert len(h.reasons) == 2
        assert h.max_allowed_route == "ARMED"

    def test_ok_low_confidence_caps_at_wait(self):
        h = RunHealth()
        h.confidence_score = 30
        assert h.status == "OK"
        assert h.max_allowed_route == "WAIT"


# ═══════════════════════════════════════════════
#  2. 축 결손 시 신뢰도 감점
# ═══════════════════════════════════════════════
class TestConfidenceScore:
    def test_full_health_is_100(self):
        assert RunHealth().confidence_score == 100.0

    def test_mcap_empty_loses_20(self):
        h = RunHealth()
        h.add_issue("MCAP_EMPTY")
        assert h.confidence_score == 80.0

    def test_all_axes_down(self):
        h = RunHealth()
        for code in ["MCAP_EMPTY", "BENCH_FAIL", "FLOW_ZERO", "NEWS_OFF", "SECTOR_FAIL"]:
            h.add_issue(code)
        assert h.confidence_score <= 35

    def test_unknown_issue_loses_5(self):
        h = RunHealth()
        h.add_issue("UNKNOWN_THING")
        assert h.confidence_score == 95.0


# ═══════════════════════════════════════════════
#  3. check_run_health() 통합
# ═══════════════════════════════════════════════
class TestCheckRunHealth:
    @pytest.fixture()
    def base_df(self):
        return pd.DataFrame({
            "NEWS_SCORE": [5.0] * 3,
            "SECTOR_RANK": [3.0] * 3,
            "시가총액(억원)": [5000] * 3,
            "rel_60d_%": [1.0] * 3,
            "추천매도가1": [5000] * 3,
            "추천매도가2": [6000] * 3,
        })

    def test_all_ok(self, base_df):
        h = check_run_health(
            base_df,
            mcap_map={"005930": 100000},
            bench_map={"KOSPI": {60: 1.5}},
            inv_maps={"frg": {"005930": 100}, "inst": {"005930": 50}},
        )
        assert h.status == "OK"
        assert h.max_allowed_route == "ATTACK"

    def test_typical_degraded(self):
        """수급+뉴스 결손 → DEGRADED, ARMED."""
        df = pd.DataFrame({
            "NEWS_SCORE": [0.0] * 3,
            "SECTOR_RANK": [3.0] * 3,
            "시가총액(억원)": [5000] * 3,
            "rel_60d_%": [1.0] * 3,
            "추천매도가1": [5000] * 3,
            "추천매도가2": [6000] * 3,
        })
        h = check_run_health(
            df,
            mcap_map={"005930": 100000},
            bench_map={"KOSPI": {60: 1.5}},
            inv_maps=None,
        )
        assert h.status == "DEGRADED"
        assert h.max_allowed_route == "ARMED"
        assert "FLOW_ZERO" in h.reasons
        assert "NEWS_OFF" in h.reasons


# ═══════════════════════════════════════════════
#  4. FLOW_PARTIAL 분기
# ═══════════════════════════════════════════════
class TestFlowPartial:
    @pytest.fixture()
    def base_df(self):
        return pd.DataFrame({
            "NEWS_SCORE": [5.0] * 3,
            "SECTOR_RANK": [3.0] * 3,
            "시가총액(억원)": [5000] * 3,
            "rel_60d_%": [1.0] * 3,
            "추천매도가1": [5000] * 3,
            "추천매도가2": [6000] * 3,
        })

    def test_ant_only_is_partial(self, base_df):
        h = check_run_health(
            base_df,
            mcap_map={"005930": 100000},
            bench_map={"KOSPI": {60: 1.5}},
            inv_maps={"frg": {}, "inst": {}, "ant": {"005930": 500}},
        )
        assert "FLOW_PARTIAL" in h.reasons
        assert "FLOW_ZERO" not in h.reasons
        assert h.status == "DEGRADED"
        assert h.confidence_score == 92.0
        assert h.max_allowed_route == "ARMED"

    def test_all_empty_is_flow_zero(self, base_df):
        h = check_run_health(
            base_df,
            mcap_map={"005930": 100000},
            bench_map={"KOSPI": {60: 1.5}},
            inv_maps={"frg": {}, "inst": {}, "ant": {}},
        )
        assert "FLOW_ZERO" in h.reasons
        assert "FLOW_PARTIAL" not in h.reasons
        assert h.confidence_score == 85.0

    def test_major_present_is_ok(self, base_df):
        h = check_run_health(
            base_df,
            mcap_map={"005930": 100000},
            bench_map={"KOSPI": {60: 1.5}},
            inv_maps={"frg": {"005930": 100}, "inst": {}, "ant": {}},
        )
        assert "FLOW_ZERO" not in h.reasons
        assert "FLOW_PARTIAL" not in h.reasons
        assert h.status == "OK"

    def test_partial_less_penalty_than_zero(self):
        h_partial = RunHealth()
        h_partial.add_issue("FLOW_PARTIAL")
        h_zero = RunHealth()
        h_zero.add_issue("FLOW_ZERO")
        assert h_partial.confidence_score > h_zero.confidence_score
