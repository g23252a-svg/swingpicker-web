"""
tests/test_buy_now_integration.py
==================================
[v3.9.22a 미니패치 3] BUY_NOW production integration test.

목적:
- test_buy_now_shadow.py는 로직 재구현형 → production 코드 변경 시
  테스트는 통과하지만 실제 동작은 깨질 수 있음 (평가 명시).
- 이 파일은 실제 scoring_engine.compute_elite_score()를 호출하여
  BUY_NOW_* 컬럼이 실제로 생성되는지 검증.

검증 항목:
1. compute_elite_score 호출 시 BUY_NOW_SCORE/GRADE/PASS/REASON/ELIGIBLE 생성
2. TOP_PICK=1 AND BUY_NOW_PASS=1 → BUY_NOW_ELIGIBLE=1
3. TOP_PICK=1 AND BUY_NOW_PASS=0 → BUY_NOW_ELIGIBLE=0
4. TOP_PICK=0 AND BUY_NOW_PASS=1 → BUY_NOW_ELIGIBLE=0 (오해 방지 핵심)
5. critical 컬럼 2+ 결측 → WATCH 강등 + REASON="데이터 부족"
6. 실제 5/19 데이터 시뮬레이션
"""
import sys
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def scoring_module():
    """real scoring_engine import."""
    for mod in list(sys.modules.keys()):
        if mod == "scoring_engine":
            del sys.modules[mod]
    return pytest.importorskip(
        "scoring_engine",
        reason="scoring_engine import 불가",
        exc_type=ImportError,
    )


def _make_input_df(rows):
    """compute_elite_score(df)가 받는 입력 형식 — production 함수에 맞춤.

    compute_elite_score는 build_global_score 다음 단계라 다음 컬럼이 이미
    계산되어 있다고 가정:
    - STRUCT_SCORE / TIMING_SCORE / AI_SCORE / ML_SCORE / FINAL_SCORE
    - ROUTE / EBS / PASS_EBS / EST_WIN_RATE / EST_WIN_RATE_MODE
    - 종가/추천매수가/손절가/추천매도가1
    - 거래대금(억원)
    - BUY_NOW 입력: ret_1d_% / ret_5d_% / VWAP_GAP / POC_GAP /
      RES_RATIO_NEAR / MFI14 / Range_Pos
    """
    defaults = {
        "종목코드": "001",
        "종목명": "테스트종목",
        # 3축 점수 (compute_elite_score가 직접 읽음)
        "STRUCT_SCORE": 85.0,
        "TIMING_SCORE": 80.0,
        "AI_SCORE": 75.0,
        "ML_SCORE": 75.0,
        "FINAL_SCORE": 80.0,
        "DISPLAY_SCORE": 80.0,
        # ROUTE / EBS — TOP_PICK 게이트 통과 위해
        "ROUTE": "ATTACK",
        "EBS": 8,
        "PASS_EBS": 1,
        # 가격 — RR / ENTRY_GAP 자동 계산
        "종가": 10000,
        "추천매수가": 10000,
        "손절가": 9500,
        "추천매도가1": 11500,  # +15% TP1 → AGGRESSIVE 조건
        # 유동성
        "거래대금(억원)": 100,
        # 승률 — STABLE 분기용
        "EST_WIN_RATE": 0.60,
        "EST_WIN_RATE_MODE": "MATURE",
        "CALIBRATION_MODE": "MATURE",
        # BUY_NOW 입력 — 정상 종목
        "ret_1d_%": 1.0,
        "ret_5d_%": 5.0,
        "VWAP_GAP": 5.0,
        "POC_GAP": 5.0,
        "RES_RATIO_NEAR": 0.10,
        "MFI14": 60.0,
        "Range_Pos": 0.60,
    }
    data = []
    for row in rows:
        d = dict(defaults)
        d.update(row)
        data.append(d)
    return pd.DataFrame(data)


# ════════════════════════════════════════════════════════════════
# A. 컬럼 생성 확인
# ════════════════════════════════════════════════════════════════
class TestColumnGeneration:
    """compute_elite_score 호출 시 BUY_NOW_* 컬럼 생성."""

    def test_all_buy_now_columns_exist(self, scoring_module):
        """v3.9.22a 신규 컬럼 6개 모두 생성되는지."""
        df = _make_input_df([{}])  # 정상 1건
        out, _meta = scoring_module.compute_elite_score(df)

        expected_cols = [
            "BUY_NOW_SCORE", "BUY_NOW_GRADE", "BUY_NOW_PASS",
            "BUY_NOW_REASON", "BUY_NOW_ELIGIBLE",
            "NO_CHASE_FLAG", "PULLBACK_WAIT_FLAG",
        ]
        for col in expected_cols:
            assert col in out.columns, f"필수 컬럼 '{col}' 없음"

    def test_top_pick_preserved(self, scoring_module):
        """TOP_PICK 컬럼이 그대로 보존."""
        df = _make_input_df([{}])
        out, _meta = scoring_module.compute_elite_score(df)
        assert "TOP_PICK" in out.columns
        assert "TOP_PICK_TYPE" in out.columns


# ════════════════════════════════════════════════════════════════
# B. BUY_NOW_ELIGIBLE — 오해 방지 핵심
# ════════════════════════════════════════════════════════════════
class TestEligibleColumn:
    """평가 미니패치 1: TOP_PICK AND BUY_NOW_PASS."""

    def test_eligible_when_top_pick_and_pass(self, scoring_module):
        """TOP_PICK=1 AND BUY_NOW_PASS=1 → ELIGIBLE=1."""
        # 정상 종목 → TOP_PICK=1, BUY_NOW=BUY
        df = _make_input_df([{}])
        out, _ = scoring_module.compute_elite_score(df)

        # 사전조건: TOP_PICK=1, BUY_NOW_PASS=1이어야 의미있음
        if out.iloc[0]["TOP_PICK"] == 1 and out.iloc[0]["BUY_NOW_PASS"] == 1:
            assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == 1

    def test_not_eligible_when_top_pick_but_avoid(self, scoring_module):
        """TOP_PICK=1 AND BUY_NOW_GRADE=AVOID → ELIGIBLE=0."""
        # RR<1.10 → HARD BLOCK → AVOID. RR은 가격으로 자동 계산되므로 가격 조정
        # reward=(tp1-close), risk=(close-stop) → RR=(tp1-close)/(close-stop)
        # close=10000, stop=9500 → risk=500. tp1=10500 → reward=500 → RR=1.0
        # close=10000, stop=9500, tp1=10520 → RR=520/500=1.04 (<1.10)
        df = _make_input_df([{
            "추천매도가1": 10520,  # RR ~1.04
        }])
        out, _ = scoring_module.compute_elite_score(df)

        # TOP_PICK 통과 여부 확인
        if out.iloc[0]["TOP_PICK"] == 1:
            # BUY_NOW는 AVOID (RR<1.10 HARD BLOCK)
            assert out.iloc[0]["BUY_NOW_GRADE"] == "AVOID"
            assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == 0

    def test_not_eligible_when_not_top_pick(self, scoring_module):
        """TOP_PICK=0 AND BUY_NOW_PASS=1 → ELIGIBLE=0 (오해 방지 핵심).

        평가 명시: PRIME 회원이 CSV 보면 TOP_PICK=0인데 BUY_NOW_PASS=1인
        종목이 매수 후보로 오해될 수 있음.
        """
        # TOP_PICK 게이트 실패 — ROUTE를 NEUTRAL로 (ATTACK/ARMED 아닌)
        df = _make_input_df([{"ROUTE": "NEUTRAL"}])
        out, _ = scoring_module.compute_elite_score(df)

        # TOP_PICK=0이지만 BUY_NOW는 정상 종목이라 BUY 가능
        assert out.iloc[0]["TOP_PICK"] == 0
        # ELIGIBLE은 반드시 0 (오해 방지)
        assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == 0


# ════════════════════════════════════════════════════════════════
# C. 결측 데이터 보호 — 미니패치 2
# ════════════════════════════════════════════════════════════════
class TestMissingDataProtection:
    """평가 미니패치 2: critical 컬럼 2+ 결측 시 WATCH 강등."""

    def test_two_critical_missing_downgrades_to_watch(
        self, scoring_module
    ):
        """VWAP_GAP + POC_GAP 결측 → WATCH 강등."""
        df = _make_input_df([{}])
        # 결측 처리 (None은 read_csv가 NaN으로 만드는 케이스 시뮬레이션)
        df.loc[0, "VWAP_GAP"] = np.nan
        df.loc[0, "POC_GAP"] = np.nan

        out, _ = scoring_module.compute_elite_score(df)

        # 위험 신호 없는 정상이지만 critical 2개 결측 → WATCH 강등
        assert out.iloc[0]["BUY_NOW_GRADE"] == "WATCH"
        assert out.iloc[0]["BUY_NOW_PASS"] == 0
        assert "데이터 부족" in str(out.iloc[0]["BUY_NOW_REASON"])

    def test_one_critical_missing_still_buy(self, scoring_module):
        """1개만 결측이면 BUY 유지 (1개는 허용)."""
        df = _make_input_df([{}])
        df.loc[0, "POC_GAP"] = np.nan  # 1개만 결측

        out, _ = scoring_module.compute_elite_score(df)
        # 정상 종목 + 1개 결측은 BUY 유지
        assert out.iloc[0]["BUY_NOW_GRADE"] == "BUY"

    def test_three_critical_missing_downgrades(self, scoring_module):
        """3개 이상 결측 → WATCH."""
        df = _make_input_df([{}])
        df.loc[0, "VWAP_GAP"] = np.nan
        df.loc[0, "POC_GAP"] = np.nan
        df.loc[0, "MFI14"] = np.nan

        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["BUY_NOW_GRADE"] == "WATCH"
        assert "데이터 부족" in str(out.iloc[0]["BUY_NOW_REASON"])

    def test_avoid_not_upgraded_by_missing(self, scoring_module):
        """이미 AVOID인 종목은 결측 강등으로 WATCH가 되지 않음.

        WATCH 강등은 BUY/원래 점수 종목만 대상. AVOID는 더 보수적이므로 유지.
        """
        # RR<1.10 → HARD BLOCK → AVOID
        df = _make_input_df([{"추천매도가1": 10520}])  # RR ~1.04
        # critical 2+ 결측까지 추가
        df.loc[0, "VWAP_GAP"] = np.nan
        df.loc[0, "POC_GAP"] = np.nan

        out, _ = scoring_module.compute_elite_score(df)
        # AVOID 유지 (더 보수적인 등급이므로 WATCH로 격상되면 안 됨)
        assert out.iloc[0]["BUY_NOW_GRADE"] == "AVOID"


# ════════════════════════════════════════════════════════════════
# D. 실제 시나리오 — production 호출 검증
# ════════════════════════════════════════════════════════════════
class TestProductionScenarios:
    """실제 5/19 데이터로 production compute_elite_score 동작 검증."""

    def test_mirae_asset_venture_5_19(self, scoring_module):
        """5/19 미래에셋벤처투자: RR 1.08 → BUY_NOW=AVOID 0점."""
        # RR ~1.08을 가격으로 표현: close=10000, stop=9500, tp1=10540 → RR=540/500=1.08
        df = _make_input_df([{
            "종목명": "미래에셋벤처투자",
            "추천매도가1": 10540,
            "ret_1d_%": -9.79,
            "ret_5d_%": 3.23,
            "VWAP_GAP": 55.24,
            "POC_GAP": 113.94,
            "RES_RATIO_NEAR": 0.029,
            "MFI14": 59.6,
            "Range_Pos": 0.64,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        # RR<1.10 HARD BLOCK
        assert out.iloc[0]["BUY_NOW_SCORE"] == 0.0
        assert out.iloc[0]["BUY_NOW_GRADE"] == "AVOID"
        assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == 0  # 핵심
        # REASON에 RR 정보
        reason = str(out.iloc[0]["BUY_NOW_REASON"])
        # RR 또는 다른 hard block 사유 있어야 함
        assert reason and ("RR" in reason or "데이터" in reason or "진입괴리" in reason or "5일" in reason)

    def test_kx_hitech_5_18(self, scoring_module):
        """5/18 KX하이텍 정상 종목 → BUY."""
        # RR ~1.69 → close=10000, stop=9500, tp1=10845 → RR=845/500=1.69
        df = _make_input_df([{
            "종목명": "KX하이텍",
            "추천매도가1": 10845,
            "ret_1d_%": -1.58,
            "ret_5d_%": 12.18,
            "VWAP_GAP": 4.33,
            "POC_GAP": 15.68,
            "RES_RATIO_NEAR": 0.249,
            "MFI14": 75.8,
            "Range_Pos": 0.46,
        }])
        out, _ = scoring_module.compute_elite_score(df)
        assert out.iloc[0]["BUY_NOW_GRADE"] == "BUY"
        # TOP_PICK이면 ELIGIBLE=1
        if out.iloc[0]["TOP_PICK"] == 1:
            assert out.iloc[0]["BUY_NOW_ELIGIBLE"] == 1


# ════════════════════════════════════════════════════════════════
# E. 컬럼 타입 검증
# ════════════════════════════════════════════════════════════════
class TestColumnTypes:

    def test_eligible_is_int(self, scoring_module):
        """BUY_NOW_ELIGIBLE은 int (0/1)."""
        df = _make_input_df([{}])
        out, _ = scoring_module.compute_elite_score(df)
        v = out.iloc[0]["BUY_NOW_ELIGIBLE"]
        assert v in (0, 1) or isinstance(v, (int, np.integer))

    def test_grade_is_one_of_three(self, scoring_module):
        """BUY_NOW_GRADE는 'BUY' / 'WATCH' / 'AVOID' 중 하나."""
        df = _make_input_df([
            {},  # BUY
            {"추천매도가1": 10520},  # AVOID (RR ~1.04 HARD BLOCK)
            {"VWAP_GAP": np.nan, "POC_GAP": np.nan},  # WATCH (결측)
        ])
        out, _ = scoring_module.compute_elite_score(df)
        for grade in out["BUY_NOW_GRADE"]:
            assert grade in ("BUY", "WATCH", "AVOID")

    def test_score_in_range(self, scoring_module):
        """BUY_NOW_SCORE는 0~100."""
        df = _make_input_df([{}])
        out, _ = scoring_module.compute_elite_score(df)
        s = out.iloc[0]["BUY_NOW_SCORE"]
        assert 0.0 <= s <= 100.0
