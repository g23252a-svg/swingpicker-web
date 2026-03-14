# -*- coding: utf-8 -*-
"""test_toxic_filter.py — TOXIC Filter 회귀 테스트 (v20.6.5 pytest 호환)
═══════════════════════════════════════════════════
determine_state_dynamic의 EXIT_WARNING 검출 정확성 검증.
pytest / 스크립트 양방향 실행 가능.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pytest
from scoring_engine import determine_state_dynamic

# ── 공용 thresholds ──
TH = {"vol_q75": 1.2, "range_q75": 0.8}

# ── 공용 베이스 row (거래대금 없는 상태) ──
_BASE = {
    "RSI14": 50, "ret_1d_%": 6.0, "ret_5d_%": 10.0,
    "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5, "Vol_Quality": 1.0,
    "TIMING_SCORE": 50, "거래강도": 2.0, "Low_Trend_PCT": 0,
    "Above_MA20": 1,
    "외인순매수": -500_000_000,
    "개인순매수": 500_000_000,
}


class TestTurnoverMissing:
    """거래대금 누락/이상 시 EXIT_WARNING 도배 방지."""

    def test_no_turnover_key(self):
        """거래대금(원) 키 자체가 없으면 EXIT_WARNING 아님."""
        state = determine_state_dynamic(_BASE, TH)
        assert state != "EXIT_WARNING", f"state={state}"

    def test_turnover_zero(self):
        """거래대금=0 → EXIT_WARNING 아님."""
        row = {**_BASE, "거래대금(원)": 0}
        state = determine_state_dynamic(row, TH)
        assert state != "EXIT_WARNING", f"state={state}"

    def test_turnover_one(self):
        """거래대금=1.0 (기존 default) → EXIT_WARNING 아님."""
        row = {**_BASE, "거래대금(원)": 1.0}
        state = determine_state_dynamic(row, TH)
        assert state != "EXIT_WARNING", f"state={state}"

    def test_turnover_nan(self):
        """거래대금=NaN → fail-safe, EXIT_WARNING 아님."""
        row = {**_BASE, "거래대금(원)": float("nan")}
        state = determine_state_dynamic(row, TH)
        assert state != "EXIT_WARNING", f"state={state}"

    def test_turnover_string(self):
        """거래대금=문자열 → 크래시 없이 EXIT_WARNING 아님."""
        row = {**_BASE, "거래대금(원)": "invalid"}
        state = determine_state_dynamic(row, TH)
        assert state != "EXIT_WARNING", f"state={state}"


class TestToxicDetection:
    """정상 거래대금 + TOXIC 조건 → EXIT_WARNING 정상 작동."""

    def test_flow_toxic(self):
        """외인 대량 매도 + 개인 대량 매수 → EXIT_WARNING."""
        row = {
            "RSI14": 50, "ret_1d_%": 6.0, "ret_5d_%": 10.0,
            "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5, "Vol_Quality": 1.0,
            "TIMING_SCORE": 50, "거래강도": 2.0, "Low_Trend_PCT": 0,
            "Above_MA20": 1,
            "거래대금(원)": 10_000_000_000,
            "외인순매수": -3_000_000_000,
            "개인순매수": 3_000_000_000,
        }
        state = determine_state_dynamic(row, TH)
        assert state == "EXIT_WARNING", f"state={state}"

    def test_volz_toxic(self):
        """vol_z≥10 + r1≥10 → EXIT_WARNING (거래대금 무관)."""
        row = {
            "RSI14": 50, "ret_1d_%": 12.0, "ret_5d_%": 5.0,
            "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5, "Vol_Quality": 1.0,
            "TIMING_SCORE": 50, "거래강도": 15.0, "Low_Trend_PCT": 0,
            "Above_MA20": 1,
        }
        state = determine_state_dynamic(row, TH)
        assert state == "EXIT_WARNING", f"state={state}"


class TestNormalRouting:
    """정상 종목 → 비-EXIT 상태 반환."""

    VALID_STATES = {"ATTACK", "ARMED", "WAIT", "NEUTRAL", "OVERHEAT"}

    def test_normal_stock(self):
        """정상 지표 → EXIT_WARNING 아닌 유효 상태."""
        row = {
            "RSI14": 55, "ret_1d_%": 2.0, "ret_5d_%": 5.0,
            "MACD_Slope_PCT": 0.02, "Range_Pos": 0.9, "Vol_Quality": 1.5,
            "TIMING_SCORE": 70, "거래강도": 3.0, "Low_Trend_PCT": 1.0,
            "Above_MA20": 1,
            "거래대금(원)": 5_000_000_000,
            "외인순매수": 100_000_000,
            "개인순매수": -50_000_000,
        }
        state = determine_state_dynamic(row, TH)
        assert state != "EXIT_WARNING", f"state={state}"
        assert state in self.VALID_STATES, f"state={state}"


class TestThresholdCustom:
    """turnover_min_valid 커스텀 thresholds."""

    @staticmethod
    def _toxic_row(turnover):
        return {
            "RSI14": 50, "ret_1d_%": 6.0, "ret_5d_%": 10.0,
            "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5, "Vol_Quality": 1.0,
            "TIMING_SCORE": 50, "거래강도": 2.0, "Low_Trend_PCT": 0,
            "Above_MA20": 1,
            "거래대금(원)": turnover,
            "외인순매수": -3_000_000_000,
            "개인순매수": 3_000_000_000,
        }

    def test_default_min_allows(self):
        """1억 + 기본min(5천만) → 유효 → EXIT_WARNING."""
        state = determine_state_dynamic(self._toxic_row(100_000_000), TH)
        assert state == "EXIT_WARNING", f"state={state}"

    def test_strict_min_blocks(self):
        """1억 + strict_min(2억) → 무효 → EXIT_WARNING 아님."""
        th_strict = {**TH, "turnover_min_valid": 200_000_000}
        state = determine_state_dynamic(self._toxic_row(100_000_000), th_strict)
        assert state != "EXIT_WARNING", f"state={state}"


class TestNamedColumns:
    """외인순매수금액 (원 단위 컬럼) 우선 사용 확인."""

    def test_amount_column_priority(self):
        """외인순매수금액 컬럼이 있으면 외인순매수보다 우선."""
        row = {
            "RSI14": 50, "ret_1d_%": 6.0, "ret_5d_%": 10.0,
            "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5, "Vol_Quality": 1.0,
            "TIMING_SCORE": 50, "거래강도": 2.0, "Low_Trend_PCT": 0,
            "Above_MA20": 1,
            "거래대금(원)": 10_000_000_000,
            "외인순매수금액": -3_000_000_000,
            "개인순매수금액": 3_000_000_000,
            "외인순매수": -100,
            "개인순매수": 100,
        }
        state = determine_state_dynamic(row, TH)
        assert state == "EXIT_WARNING", f"state={state}"


# ── 스크립트 실행 호환 ──
if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
