# -*- coding: utf-8 -*-
"""
test_run_health.py — RunHealth 신뢰도/행동 상한 테스트
═══════════════════════════════════════════════════════
[v20.0.2] DEGRADED → 최대 ARMED 규칙 검증

실행:
  pytest test_run_health.py -v
  python test_run_health.py
"""

import sys, os, unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_health import RunHealth, check_run_health


class TestMaxAllowedRoute(unittest.TestCase):
    """DEGRADED/CRITICAL/OK 상태별 행동 상한 테스트"""

    def test_ok_high_confidence_allows_attack(self):
        """OK + 신뢰도 100 → ATTACK 허용"""
        h = RunHealth()
        h.add_ok("MCAP")
        h.add_ok("BENCH")
        self.assertEqual(h.status, "OK")
        self.assertEqual(h.max_allowed_route, "ATTACK")

    def test_degraded_always_caps_at_armed(self):
        """DEGRADED → 신뢰도 높아도 절대 ATTACK 안 됨"""
        h = RunHealth()
        h.add_issue("FLOW_ZERO")  # -15점 → 85/100
        self.assertEqual(h.status, "DEGRADED")
        self.assertGreaterEqual(h.confidence_score, 70)  # 신뢰도는 높지만
        self.assertEqual(h.max_allowed_route, "ARMED")   # ATTACK 금지!

    def test_degraded_many_issues_caps_at_wait(self):
        """DEGRADED + 결손 3개 이상 + 신뢰도 50 미만 → WAIT"""
        h = RunHealth()
        h.add_issue("MCAP_EMPTY")   # -20
        h.add_issue("BENCH_FAIL")   # -15
        h.add_issue("FLOW_ZERO")    # -15
        h.add_issue("NEWS_OFF")     # -10 → 총 60 감점 → 40점
        self.assertEqual(h.status, "DEGRADED")
        self.assertLess(h.confidence_score, 50)
        self.assertEqual(h.max_allowed_route, "WAIT")

    def test_critical_always_wait(self):
        """CRITICAL → 무조건 WAIT"""
        h = RunHealth()
        h.add_issue("FATAL_ERROR", severity="CRITICAL")
        self.assertEqual(h.status, "CRITICAL")
        self.assertEqual(h.max_allowed_route, "WAIT")

    def test_degraded_two_issues_armed(self):
        """DEGRADED + 결손 2개 → ARMED (3개 미만이므로 WAIT 아님)"""
        h = RunHealth()
        h.add_issue("FLOW_ZERO")    # -15
        h.add_issue("NEWS_OFF")     # -10
        self.assertEqual(h.status, "DEGRADED")
        self.assertEqual(len(h.reasons), 2)
        self.assertEqual(h.max_allowed_route, "ARMED")

    def test_ok_low_confidence_caps_at_wait(self):
        """OK이지만 신뢰도 <40 → WAIT"""
        h = RunHealth()
        h.confidence_score = 30  # 수동 설정
        self.assertEqual(h.status, "OK")
        self.assertEqual(h.max_allowed_route, "WAIT")


class TestConfidenceScore(unittest.TestCase):
    """축 결손 시 신뢰도 감점 테스트"""

    def test_full_health_is_100(self):
        """결손 없음 → 100점"""
        h = RunHealth()
        self.assertEqual(h.confidence_score, 100.0)

    def test_mcap_empty_loses_20(self):
        """시총 결손 → -20점"""
        h = RunHealth()
        h.add_issue("MCAP_EMPTY")
        self.assertEqual(h.confidence_score, 80.0)

    def test_all_axes_down(self):
        """전축 결손 → 대량 감점"""
        h = RunHealth()
        for code in ["MCAP_EMPTY", "BENCH_FAIL", "FLOW_ZERO", "NEWS_OFF", "SECTOR_FAIL"]:
            h.add_issue(code)
        self.assertLessEqual(h.confidence_score, 35)  # 100 - 20 - 15 - 15 - 10 - 10 = 30

    def test_unknown_issue_loses_5(self):
        """알 수 없는 이슈 → 기본 -5점"""
        h = RunHealth()
        h.add_issue("UNKNOWN_THING")
        self.assertEqual(h.confidence_score, 95.0)


class TestCheckRunHealth(unittest.TestCase):
    """check_run_health() 통합 테스트"""

    def test_all_ok(self):
        """전축 정상 → OK"""
        df = pd.DataFrame({
            "NEWS_SCORE": [5.0] * 3,
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
            inv_maps={"frg": {"005930": 100}, "inst": {"005930": 50}},
        )
        self.assertEqual(h.status, "OK")
        self.assertEqual(h.max_allowed_route, "ATTACK")

    def test_typical_degraded(self):
        """수급+뉴스 결손 (실전 자주 발생) → DEGRADED, ARMED"""
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
            inv_maps=None,  # 수급 없음
        )
        self.assertEqual(h.status, "DEGRADED")
        self.assertEqual(h.max_allowed_route, "ARMED")  # ATTACK 금지!
        self.assertIn("FLOW_ZERO", h.reasons)
        self.assertIn("NEWS_OFF", h.reasons)


class TestFlowPartial(unittest.TestCase):
    """FLOW_PARTIAL 분기 테스트 — 개인 수급만 있는 경우"""

    def _base_df(self):
        return pd.DataFrame({
            "NEWS_SCORE": [5.0] * 3,
            "SECTOR_RANK": [3.0] * 3,
            "시가총액(억원)": [5000] * 3,
            "rel_60d_%": [1.0] * 3,
            "추천매도가1": [5000] * 3,
            "추천매도가2": [6000] * 3,
        })

    def test_flow_partial_when_ant_only(self):
        """외인/기관 0, 개인(ant)만 있음 → FLOW_PARTIAL(-8), DEGRADED, ARMED"""
        h = check_run_health(
            self._base_df(),
            mcap_map={"005930": 100000},
            bench_map={"KOSPI": {60: 1.5}},
            inv_maps={"frg": {}, "inst": {}, "ant": {"005930": 500}},
        )
        self.assertIn("FLOW_PARTIAL", h.reasons)
        self.assertNotIn("FLOW_ZERO", h.reasons)
        self.assertEqual(h.status, "DEGRADED")
        # 감점: FLOW_PARTIAL=8 → confidence=92 → max_allowed=ARMED
        self.assertEqual(h.confidence_score, 92.0)
        self.assertEqual(h.max_allowed_route, "ARMED")

    def test_flow_zero_when_all_empty(self):
        """외인/기관/개인 모두 0 → FLOW_ZERO(-15) 유지"""
        h = check_run_health(
            self._base_df(),
            mcap_map={"005930": 100000},
            bench_map={"KOSPI": {60: 1.5}},
            inv_maps={"frg": {}, "inst": {}, "ant": {}},
        )
        self.assertIn("FLOW_ZERO", h.reasons)
        self.assertNotIn("FLOW_PARTIAL", h.reasons)
        self.assertEqual(h.confidence_score, 85.0)

    def test_flow_ok_when_major_present(self):
        """외인 있음 → FLOW OK, 감점 없음"""
        h = check_run_health(
            self._base_df(),
            mcap_map={"005930": 100000},
            bench_map={"KOSPI": {60: 1.5}},
            inv_maps={"frg": {"005930": 100}, "inst": {}, "ant": {}},
        )
        self.assertNotIn("FLOW_ZERO", h.reasons)
        self.assertNotIn("FLOW_PARTIAL", h.reasons)
        self.assertEqual(h.status, "OK")

    def test_flow_partial_confidence_is_higher_than_flow_zero(self):
        """FLOW_PARTIAL 감점(8) < FLOW_ZERO 감점(15) 확인"""
        h_partial = RunHealth()
        h_partial.add_issue("FLOW_PARTIAL")
        h_zero = RunHealth()
        h_zero.add_issue("FLOW_ZERO")
        self.assertGreater(h_partial.confidence_score, h_zero.confidence_score)


if __name__ == "__main__":
    unittest.main(verbosity=2)
