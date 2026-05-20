"""
tests/test_buy_now_v3922c_rules.py
===================================
[v3.9.22c] BUY_NOW 룰 확장 회귀 가드.

신규 룰:
- P1: ENTRY_RISK_LEVEL=RED → AVOID (HARD BLOCK)
- P2: ENTRY_RISK_LEVEL=ORANGE → -15점 (soft penalty)
- P3-1: ret_5d > 20 AND ret_1d < -5 → AVOID
- P3-2: 이격도 > 15 AND ret_1d < -3 → AVOID
- P3-3: 이격도 > 10 (~15) → -15점
- P3-4: ret_5d > 15 AND ret_1d < -3 AND not (P3-1) → -15점
- P3-5: MFI > 75 AND ret_5d > 15 AND ret_1d < 0 → -10점
- P3-6: EBS < 6 AND TOP_PICK != 1 → BUY_NOW_ELIGIBLE 강제 0

핵심 시나리오: 현대해상 20260520
- ELITE_SCORE 62.7, EBS 4/8, ret_1d -5.53, ret_5d +21.32, MFI 76.5, 이격도 14.77
- 기존 v3.9.22a 룰만으로는 BUY 등급 (점수 70대)
- v3.9.22c 신규 룰로는 AVOID (이격도 > 10 + reversal + ORANGE)
"""
import sys
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def scoring_module():
    for mod in list(sys.modules.keys()):
        if mod == "scoring_engine":
            del sys.modules[mod]
    return pytest.importorskip(
        "scoring_engine",
        reason="scoring_engine import 불가",
        exc_type=ImportError,
    )


def _make_input_df(rows):
    """compute_elite_score 입력 — 정상 종목 기본."""
    defaults = {
        "종목코드": "001",
        "종목명": "테스트",
        "STRUCT_SCORE": 85.0,
        "TIMING_SCORE": 80.0,
        "AI_SCORE": 75.0,
        "ML_SCORE": 75.0,
        "FINAL_SCORE": 80.0,
        "DISPLAY_SCORE": 80.0,
        "ROUTE": "ATTACK",
        "EBS": 8,
        "PASS_EBS": 1,
        "종가": 10000,
        "추천매수가": 10000,
        "손절가": 9500,
        "추천매도가1": 11500,
        "거래대금(억원)": 100,
        "EST_WIN_RATE": 0.60,
        "EST_WIN_RATE_MODE": "MATURE",
        "CALIBRATION_MODE": "MATURE",
        "ret_1d_%": 1.0,
        "ret_5d_%": 5.0,
        "VWAP_GAP": 5.0,
        "POC_GAP": 5.0,
        "RES_RATIO_NEAR": 0.10,
        "MFI14": 60.0,
        "Range_Pos": 0.60,
        # v3.9.22c 신규 입력
        "이격도": 5.0,
        "ENTRY_RISK_LEVEL": "GREEN",
    }
    data = []
    for r in rows:
        d = dict(defaults)
        d.update(r)
        data.append(d)
    return pd.DataFrame(data)


# ════════════════════════════════════════════════════════════════
# P1: ENTRY_RISK_LEVEL=RED → AVOID
# ════════════════════════════════════════════════════════════════
class TestEntryRiskRed:

    def test_red_forces_avoid(self, scoring_module):
        """RED → BUY_NOW_GRADE=AVOID, SCORE=0."""
        df = _make_input_df([{
            "ENTRY_RISK_LEVEL": "RED",
            # 다른 지표는 정상이라 v3.9.22a 룰만으로는 BUY
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["BUY_NOW_GRADE"] == "AVOID"
        assert out.iloc[0]["BUY_NOW_SCORE"] == 0.0
        assert "RED" in str(out.iloc[0]["BUY_NOW_REASON"])

    def test_red_eligible_zero(self, scoring_module):
        """RED는 ELIGIBLE도 강제 0."""
        df = _make_input_df([{"ENTRY_RISK_LEVEL": "RED"}])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == 0


# ════════════════════════════════════════════════════════════════
# P2: ENTRY_RISK_LEVEL=ORANGE → -15 soft
# ════════════════════════════════════════════════════════════════
class TestEntryRiskOrange:

    def test_orange_soft_penalty(self, scoring_module):
        """ORANGE → -15점 (BUY 유지 가능)."""
        df = _make_input_df([{"ENTRY_RISK_LEVEL": "ORANGE"}])
        out, _ = scoring_module.compute_elite_score(df)
        # 정상 종목 100 - 15 = 85점, BUY 유지
        assert out.iloc[0]["BUY_NOW_SCORE"] == 85.0
        assert out.iloc[0]["BUY_NOW_GRADE"] == "BUY"
        assert "ORANGE" in str(out.iloc[0]["BUY_NOW_REASON"])

    def test_orange_plus_other_risks_to_watch(self, scoring_module):
        """ORANGE + 다른 SOFT 위험 → WATCH/AVOID로 강등."""
        df = _make_input_df([{
            "ENTRY_RISK_LEVEL": "ORANGE",  # -15
            "ret_1d_%": -8.0,  # -20 (칼날잡기)
            "VWAP_GAP": 25.0,  # -10
        }])
        out, _ = scoring_module.compute_elite_score(df)
        # 100 - 15 - 20 - 10 = 55점 (WATCH)
        assert out.iloc[0]["BUY_NOW_SCORE"] == 55.0
        assert out.iloc[0]["BUY_NOW_GRADE"] == "WATCH"


# ════════════════════════════════════════════════════════════════
# P3-1: ret_5d > 20 AND ret_1d < -5 → AVOID
# ════════════════════════════════════════════════════════════════
class TestReversalRiskHardBlock:

    def test_short_term_surge_then_drop(self, scoring_module):
        """단기 +21% 급등 후 -5.5% 음봉 (현대해상형) → AVOID."""
        df = _make_input_df([{
            "ret_5d_%": 21.0,
            "ret_1d_%": -5.5,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["BUY_NOW_GRADE"] == "AVOID"
        assert out.iloc[0]["BUY_NOW_SCORE"] == 0.0
        assert "급등후 음봉" in str(out.iloc[0]["BUY_NOW_REASON"])

    def test_disparity_plus_negative_day(self, scoring_module):
        """이격도 > 15 + 전일 음봉 -3% → AVOID."""
        df = _make_input_df([{
            "이격도": 16.0,
            "ret_1d_%": -4.0,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["BUY_NOW_GRADE"] == "AVOID"
        assert "이격도" in str(out.iloc[0]["BUY_NOW_REASON"])


# ════════════════════════════════════════════════════════════════
# P3-3~5: 약한 reversal-risk → SOFT 감점
# ════════════════════════════════════════════════════════════════
class TestReversalRiskSoft:

    def test_disparity_10_to_15_penalty(self, scoring_module):
        """이격도 10~15 → -15점."""
        df = _make_input_df([{"이격도": 12.0}])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["BUY_NOW_SCORE"] == 85.0

    def test_mfi_surge_negative_penalty(self, scoring_module):
        """MFI > 75 + ret_5d > 15 + ret_1d < 0 → -10."""
        df = _make_input_df([{
            "MFI14": 76.0,
            "ret_5d_%": 16.0,
            "ret_1d_%": -1.0,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        # 100 - 10 = 90 (BUY 유지 — 약한 신호)
        assert out.iloc[0]["BUY_NOW_SCORE"] == 90.0

    def test_mid_surge_negative_penalty(self, scoring_module):
        """ret_5d > 15 + ret_1d < -3 (P3-1 미만) → -15."""
        # P3-1 (HARD BLOCK)은 ret_5d > 20 AND ret_1d < -5
        # P3-4 (SOFT)는 ret_5d > 15 AND ret_1d < -3 (HARD 미만)
        df = _make_input_df([{
            "ret_5d_%": 17.0,
            "ret_1d_%": -4.0,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        # 100 - 15 = 85
        assert out.iloc[0]["BUY_NOW_SCORE"] == 85.0
        assert out.iloc[0]["BUY_NOW_GRADE"] == "BUY"


# ════════════════════════════════════════════════════════════════
# P3-6: EBS < 6 AND TOP_PICK != 1 → ELIGIBLE 0
# ════════════════════════════════════════════════════════════════
class TestEbsTopPickBlock:

    def test_low_ebs_non_top_pick_eligible_zero(self, scoring_module):
        """EBS=4 AND TOP_PICK=0 → ELIGIBLE 강제 0."""
        # TOP_PICK 게이트 실패시키기 (ROUTE NEUTRAL)
        df = _make_input_df([{
            "EBS": 4,
            "ROUTE": "NEUTRAL",  # TOP_PICK=0 만들기
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["TOP_PICK"] == 0
        assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == 0

    def test_low_ebs_with_top_pick_unchanged(self, scoring_module):
        """EBS<6이어도 TOP_PICK=1이면 ELIGIBLE 영향 없음.

        (실제로는 TOP_PICK이 EBS≥5 게이트라 EBS<6이지만 TOP_PICK=1 케이스는
        EBS=5뿐. EBS=5 시뮬레이션)
        """
        df = _make_input_df([{"EBS": 5}])  # TOP_PICK은 정상 통과
        out, _ = scoring_module.compute_elite_score(df)
        if out.iloc[0]["TOP_PICK"] == 1:
            # TOP_PICK=1이면 ELIGIBLE은 BUY_NOW_PASS에 의존
            assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == int(
                out.iloc[0]["BUY_NOW_PASS"]
            )


# ════════════════════════════════════════════════════════════════
# 현대해상 시나리오 — 통합 검증
# ════════════════════════════════════════════════════════════════
class TestHyundaiMarineCase:
    """5/20 현대해상: 모든 신규 룰이 함께 작동하는 케이스."""

    def test_hyundai_marine_fully_blocked(self, scoring_module):
        """현대해상 케이스 — v3.9.22c로 AVOID + ELIGIBLE=0."""
        df = _make_input_df([{
            "종목명": "현대해상",
            "ELITE_SCORE": 62.7,
            "STRUCT_SCORE": 54.0,
            "EBS": 4,
            "ROUTE": "NEUTRAL",  # TOP_PICK=0 만들기
            "ret_1d_%": -5.53,
            "ret_5d_%": 21.32,
            "MFI14": 76.5,
            "이격도": 14.77,
            "ENTRY_RISK_LEVEL": "ORANGE",
            "VWAP_GAP": 9.4,
            "추천매도가1": 10940,  # RR ~1.10
        }])
        out, _ = scoring_module.compute_elite_score(df)

        # v3.9.22c HARD BLOCK 트리거: ret_5d>20 AND ret_1d<-5 (P3-1)
        assert out.iloc[0]["BUY_NOW_GRADE"] == "AVOID"
        assert out.iloc[0]["BUY_NOW_SCORE"] == 0.0
        # TOP_PICK=0 + EBS<6 → ELIGIBLE 강제 0
        assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == 0
        # 사유에 핵심 위험 명시
        reason = str(out.iloc[0]["BUY_NOW_REASON"])
        assert ("급등후 음봉" in reason or "이격도" in reason
                or "ORANGE" in reason)


# ════════════════════════════════════════════════════════════════
# 기존 GREEN 종목 영향 없음 확인 (회귀 방지)
# ════════════════════════════════════════════════════════════════
class TestGreenNotAffected:

    def test_normal_green_still_buy(self, scoring_module):
        """ENTRY_RISK_LEVEL=GREEN + 정상 지표 → BUY 유지."""
        df = _make_input_df([{
            "ENTRY_RISK_LEVEL": "GREEN",
            # 다른 모든 지표 정상
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["BUY_NOW_GRADE"] == "BUY"
        assert out.iloc[0]["BUY_NOW_SCORE"] >= 70

    def test_no_entry_risk_column_compat(self, scoring_module):
        """ENTRY_RISK_LEVEL 컬럼 없는 legacy CSV → 정상 동작."""
        df = _make_input_df([{}])
        df = df.drop(columns=["ENTRY_RISK_LEVEL"])
        out, _ = scoring_module.compute_elite_score(df)
        # 기본 정상 종목이므로 BUY
        assert out.iloc[0]["BUY_NOW_GRADE"] == "BUY"

    def test_no_disparity_column_compat(self, scoring_module):
        """이격도 컬럼 없는 legacy CSV → 정상 동작."""
        df = _make_input_df([{}])
        df = df.drop(columns=["이격도"])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["BUY_NOW_GRADE"] == "BUY"
