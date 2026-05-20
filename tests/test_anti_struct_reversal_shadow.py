"""
tests/test_anti_struct_reversal_shadow.py
==========================================
[v3.9.23a] Anti-STRUCT Reversal Shadow 회귀 가드.

★ 절대 지킬 룰 (평가 명시):
1. TOP_PICK 무수정
2. BUY_NOW_ELIGIBLE 무수정
3. ROUTE / LDY_RANK / ELITE_SCORE 무수정
4. 화면 추천 로직에 영향 없음 (shadow only)
5. 4월 편향/데이터마이닝 위험 → production 승격은 v3.9.23b 검증 통과 후
"""
import sys
import pytest
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
    """compute_elite_score 입력 — v3.9.22c 호환 + v3.9.23a 신규 입력."""
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
# A. SHADOW 컬럼 생성 자체
# ════════════════════════════════════════════════════════════════
class TestShadowColumnsExist:

    def test_columns_added(self, scoring_module):
        """4개 신규 컬럼 모두 출력 DataFrame에 존재."""
        df = _make_input_df([{}])
        out, _ = scoring_module.compute_elite_score(df)
        for col in [
            "ANTI_STRUCT_REVERSAL_FLAG",
            "ANTI_STRUCT_REVERSAL_TYPE",
            "ANTI_STRUCT_REVERSAL_SCORE",
            "ANTI_STRUCT_REVERSAL_REASON",
        ]:
            assert col in out.columns, f"{col} 컬럼 없음"


# ════════════════════════════════════════════════════════════════
# B. FLAG 조건 정확성
# ════════════════════════════════════════════════════════════════
class TestFlagCondition:

    def test_basic_flag_triggers(self, scoring_module):
        """TIMING≥70 AND STRUCT<50 AND 주의 조건 미해당 → FLAG=1."""
        df = _make_input_df([{
            "TIMING_SCORE": 75.0,
            "STRUCT_SCORE": 45.0,
            "이격도": 5.0,
            "MFI14": 60.0,
            "ret_1d_%": 1.0,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_FLAG"] == 1

    def test_timing_too_low(self, scoring_module):
        """TIMING<70 → FLAG=0."""
        df = _make_input_df([{"TIMING_SCORE": 65.0, "STRUCT_SCORE": 45.0}])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_FLAG"] == 0

    def test_struct_too_high(self, scoring_module):
        """STRUCT≥50 → FLAG=0."""
        df = _make_input_df([{"TIMING_SCORE": 80.0, "STRUCT_SCORE": 55.0}])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_FLAG"] == 0

    def test_disparity_exclude(self, scoring_module):
        """이격도>15 → FLAG=0 (주의 조건)."""
        df = _make_input_df([{
            "TIMING_SCORE": 90.0,
            "STRUCT_SCORE": 40.0,
            "이격도": 17.0,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_FLAG"] == 0

    def test_mfi_exclude(self, scoring_module):
        """MFI>82 → FLAG=0."""
        df = _make_input_df([{
            "TIMING_SCORE": 90.0,
            "STRUCT_SCORE": 40.0,
            "MFI14": 85.0,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_FLAG"] == 0

    def test_ret1d_jump_exclude(self, scoring_module):
        """ret_1d>10 → FLAG=0 (이미 점프)."""
        df = _make_input_df([{
            "TIMING_SCORE": 90.0,
            "STRUCT_SCORE": 40.0,
            "ret_1d_%": 12.0,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_FLAG"] == 0


# ════════════════════════════════════════════════════════════════
# C. TYPE 분류
# ════════════════════════════════════════════════════════════════
class TestTypeClassification:

    def test_basic_type(self, scoring_module):
        """T 70~84 AND S<50 → BASIC."""
        df = _make_input_df([{
            "TIMING_SCORE": 75.0,
            "STRUCT_SCORE": 40.0,
            "RR_NOW_TP1_HINT": None,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_TYPE"] == "BASIC"
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_SCORE"] == 60.0

    def test_strong_type(self, scoring_module):
        """T≥85 AND S<50 → STRONG."""
        df = _make_input_df([{
            "TIMING_SCORE": 87.0,
            "STRUCT_SCORE": 40.0,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_TYPE"] == "STRONG"
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_SCORE"] == 75.0

    def test_champion_type(self, scoring_module):
        """T≥90 AND S<60 AND RR<0.5 → CHAMPION."""
        df = _make_input_df([{
            "TIMING_SCORE": 95.0,
            "STRUCT_SCORE": 40.0,
            # 추천매도가1을 진입가에 거의 근접 → RR 매우 작음
            "추천매수가": 10000,
            "추천매도가1": 10050,  # +0.5%
            "손절가": 9000,  # -10%
        }])
        out, _ = scoring_module.compute_elite_score(df)
        # RR = (10050-10000)/(10000-9000) = 0.05 → CHAMPION
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_TYPE"] == "CHAMPION"
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_SCORE"] == 90.0


# ════════════════════════════════════════════════════════════════
# D. ★ PRODUCTION 영향 없음 (평가 절대 지킬 룰)
# ════════════════════════════════════════════════════════════════
class TestNoProductionImpact:

    def test_top_pick_unchanged(self, scoring_module):
        """ASR FLAG=1이어도 TOP_PICK 자동 부여 X."""
        df = _make_input_df([{
            "TIMING_SCORE": 95.0,
            "STRUCT_SCORE": 40.0,
            "EBS": 4,  # TOP_PICK 게이트 실패시키기
            "PASS_EBS": 0,
            "ROUTE": "NEUTRAL",
        }])
        out, _ = scoring_module.compute_elite_score(df)
        # ASR FLAG는 1
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_FLAG"] == 1
        # 하지만 TOP_PICK은 0 유지
        assert out.iloc[0]["TOP_PICK"] == 0

    def test_buy_now_eligible_unchanged(self, scoring_module):
        """ASR FLAG=1이어도 BUY_NOW_ELIGIBLE에 자동 부여 X."""
        df = _make_input_df([{
            "TIMING_SCORE": 95.0,
            "STRUCT_SCORE": 40.0,
            "ROUTE": "NEUTRAL",  # TOP_PICK 실패
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_FLAG"] == 1
        # ELIGIBLE은 TOP_PICK AND BUY_NOW_PASS만 따름
        assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == 0

    def test_buy_now_grade_independent(self, scoring_module):
        """ASR과 BUY_NOW_GRADE는 독립적."""
        # ASR 챔피언인데 BUY_NOW HARD BLOCK 걸리는 종목
        df = _make_input_df([{
            "TIMING_SCORE": 95.0,
            "STRUCT_SCORE": 40.0,
            "추천매수가": 10000,
            "추천매도가1": 10050,  # RR 매우 작음
            "손절가": 9000,
            # BUY_NOW HARD BLOCK 트리거
            "ENTRY_RISK_LEVEL": "RED",
        }])
        out, _ = scoring_module.compute_elite_score(df)
        # ASR는 자체 룰로 인정
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_TYPE"] in (
            "BASIC", "STRONG", "CHAMPION"
        )
        # BUY_NOW는 별개로 AVOID
        assert out.iloc[0]["BUY_NOW_GRADE"] == "AVOID"

    def test_legacy_csv_no_disparity(self, scoring_module):
        """이격도 컬럼 없는 legacy CSV도 정상 동작."""
        df = _make_input_df([{
            "TIMING_SCORE": 80.0,
            "STRUCT_SCORE": 40.0,
        }])
        df = df.drop(columns=["이격도"])
        out, _ = scoring_module.compute_elite_score(df)
        # 이격도 default 0 → 주의 조건 미해당 → FLAG=1
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_FLAG"] == 1


# ════════════════════════════════════════════════════════════════
# E. KX하이텍 케이스 — 평가 명시 부분 매치
# ════════════════════════════════════════════════════════════════
class TestKxHitechCase:
    """KX하이텍: T 97.8 (✅), S 92.8 (❌), RR 0.19 (✅), GAP 21.4 (✅).
    챔피언 패턴과 부분 매치. STRUCT가 너무 높아 ASR_FLAG=0 이어야 함."""

    def test_kx_hitech_not_asr(self, scoring_module):
        df = _make_input_df([{
            "종목명": "KX하이텍",
            "TIMING_SCORE": 97.8,
            "STRUCT_SCORE": 92.8,
            "이격도": 27.47,
            "MFI14": 81.2,
            "ret_1d_%": 4.18,
            "ret_5d_%": 5.23,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        # STRUCT 92.8 ≥ 50이고 이격도 27 > 15 둘 다 걸림
        # → ASR_FLAG = 0
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_FLAG"] == 0
        assert out.iloc[0]["ANTI_STRUCT_REVERSAL_TYPE"] == ""


# ════════════════════════════════════════════════════════════════
# F. REASON 텍스트
# ════════════════════════════════════════════════════════════════
class TestReason:

    def test_reason_when_flag_on(self, scoring_module):
        df = _make_input_df([{"TIMING_SCORE": 92.0, "STRUCT_SCORE": 40.0}])
        out, _ = scoring_module.compute_elite_score(df)
        reason = str(out.iloc[0]["ANTI_STRUCT_REVERSAL_REASON"])
        assert "T=" in reason
        assert "S=" in reason

    def test_reason_when_excluded_by_disparity(self, scoring_module):
        df = _make_input_df([{
            "TIMING_SCORE": 90.0,
            "STRUCT_SCORE": 40.0,
            "이격도": 18.0,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        reason = str(out.iloc[0]["ANTI_STRUCT_REVERSAL_REASON"])
        assert "이격도" in reason or "18" in reason
