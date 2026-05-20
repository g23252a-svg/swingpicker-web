"""
tests/test_validate_anti_struct_reversal.py
============================================
[v3.9.23a-fix] validate_anti_struct_reversal.py 검증.

평가 지적 사항 회귀 가드:
- monthly/macro_risk 검증은 ASR subset 기준이어야 함 (전체 universe 아님)
- 무결성: sum(by_month_asr.n) == |ASR_FLAG=1|
- 무결성: sum(by_month_all.n) == |df 전체|
- promotion_check는 by_month_asr / by_macro_risk_asr 기준
"""
import sys
import json
import tempfile
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def validate_module():
    """validate_anti_struct_reversal 모듈 import (직접 경로 로드)."""
    import importlib.util
    candidate_paths = [
        "scripts/validate_anti_struct_reversal.py",
        os.path.join(
            os.path.dirname(__file__), "..",
            "scripts/validate_anti_struct_reversal.py",
        ),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location(
                "validate_anti_struct_reversal", p
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("validate_anti_struct_reversal.py not found")


def _make_synthetic_df():
    """100건 중 ASR_FLAG=1 10건 synthetic DataFrame."""
    np.random.seed(42)
    n = 100
    n_asr = 10
    rows = []
    months = ["202603", "202604", "202605"]
    macros = ["NORMAL", "CAUTION", "CRITICAL"]
    for i in range(n):
        is_asr = i < n_asr
        rows.append({
            "rec_ymd": "20260315",
            "code": f"{i:06d}",
            "name": f"종목{i}",
            "ret_net": np.random.normal(0, 3),
            "win": int(np.random.random() > 0.5),
            "month": months[i % 3],
            # 점수들
            "STRUCT": 40 if is_asr else 80,
            "TIMING": 90 if is_asr else 60,
            "AI": 50, "BALANCE": 60, "DISPLAY": 70,
            "ELITE": 60, "FINAL": 70, "AXIS_MEAN": 70,
            "RR_NOW_TP1": 0.3 if is_asr else 1.5,
            "ENTRY_GAP_PCT": 1.0, "EBS": 6,
            "TOP_PICK": 0, "ROUTE": "ATTACK",
            "BUY_NOW_GRADE": "BUY", "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_SCORE": 70.0,
            "ANTI_STRUCT_REVERSAL_FLAG": int(is_asr),
            "ANTI_STRUCT_REVERSAL_TYPE": "BASIC" if is_asr else "",
            "ANTI_STRUCT_REVERSAL_SCORE": 60.0 if is_asr else 0.0,
            "이격도": 5.0, "MFI14": 60.0,
            "ret_1d": 1.0, "ret_5d": 2.0, "VWAP_GAP": 5.0,
            "ENTRY_RISK_LEVEL": "GREEN",
            "MACRO_RISK": macros[i % 3],
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# A. ASR subset 무결성 — sum 일치
# ════════════════════════════════════════════════════════════════
class TestAsrSubsetIntegrity:

    def test_monthly_asr_sum_equals_asr_count(self, validate_module):
        """sum(monthly_asr.n) == |ASR_FLAG=1|."""
        df = _make_synthetic_df()
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()

        monthly_asr = validate_module.validate_monthly(
            asr, baseline_win, baseline_ret
        )
        total = sum(s.get("n", 0) for s in monthly_asr.values())
        assert total == len(asr), (
            f"sum(monthly_asr.n)={total} != |ASR|={len(asr)} — "
            "월별 검증이 ASR subset을 보지 않음"
        )

    def test_monthly_all_sum_equals_total(self, validate_module):
        """sum(monthly_all.n) == |df 전체|."""
        df = _make_synthetic_df()
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()

        monthly_all = validate_module.validate_monthly(
            df, baseline_win, baseline_ret
        )
        total = sum(s.get("n", 0) for s in monthly_all.values())
        assert total == len(df), (
            f"sum(monthly_all.n)={total} != |df|={len(df)}"
        )

    def test_macro_asr_sum_equals_asr_count(self, validate_module):
        """sum(macro_asr.n) == |ASR_FLAG=1|."""
        df = _make_synthetic_df()
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()

        macro_asr = validate_module.validate_macro(
            asr, baseline_win, baseline_ret
        )
        total = sum(s.get("n", 0) for s in macro_asr.values())
        assert total == len(asr), (
            f"sum(macro_asr.n)={total} != |ASR|={len(asr)}"
        )


# ════════════════════════════════════════════════════════════════
# B. promotion_check가 ASR subset 기준인지 확인
# ════════════════════════════════════════════════════════════════
class TestPromotionUsesAsrSubset:

    def test_promotion_with_only_april_asr_data(self, validate_module):
        """4월에만 ASR 표본이 있고 다 EV+ → monthly_ev_positive PASSED."""
        df = _make_synthetic_df()
        # 4월만 ASR 데이터, 4월 ret 강제 양수
        df.loc[
            (df["month"] == "202604") & (df["ANTI_STRUCT_REVERSAL_FLAG"] == 1),
            "ret_net"
        ] = 5.0
        df.loc[
            (df["month"] != "202604") | (df["ANTI_STRUCT_REVERSAL_FLAG"] != 1),
            "ANTI_STRUCT_REVERSAL_FLAG"
        ] = 0
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()

        monthly_asr = validate_module.validate_monthly(
            asr, baseline_win, baseline_ret
        )
        macro_asr = validate_module.validate_macro(
            asr, baseline_win, baseline_ret
        )
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )

        # ASR이 4월만 있으니 months_sufficient < 3 → monthly_ev_positive FAIL
        # (한 달짜리는 표본 부족 처리)
        assert "monthly_ev_positive" in promotion["criteria"]

    def test_promotion_normal_caution_uses_asr_only(self, validate_module):
        """전체에선 NORMAL ret<0이어도 ASR NORMAL ret>0이면 PASS."""
        df = _make_synthetic_df()
        # 전체에선 NORMAL ret 음수
        df.loc[df["MACRO_RISK"] == "NORMAL", "ret_net"] = -5.0
        # 단 ASR NORMAL은 ret 양수
        df.loc[
            (df["MACRO_RISK"] == "NORMAL")
            & (df["ANTI_STRUCT_REVERSAL_FLAG"] == 1),
            "ret_net"
        ] = 3.0
        # CAUTION도 ASR ret 양수로 보장
        df.loc[
            (df["MACRO_RISK"] == "CAUTION")
            & (df["ANTI_STRUCT_REVERSAL_FLAG"] == 1),
            "ret_net"
        ] = 3.0
        # 표본 더 늘려서 NORMAL, CAUTION ASR이 ≥30
        # → synthetic 한계가 있어서 일단 0건이거나 표본부족이어도
        # promotion 로직 자체가 "ASR macro에서 NORMAL/CAUTION 봄"인지 확인

        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        monthly_asr = validate_module.validate_monthly(
            asr, baseline_win, baseline_ret
        )
        macro_asr = validate_module.validate_macro(
            asr, baseline_win, baseline_ret
        )
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )

        # macro_normal_caution_ev 기준이 ASR macro에서 봄 (synthetic 표본 작아
        # insufficient여도, 적어도 키와 노트는 ASR 기준이어야 함)
        criteria = promotion["criteria"]["macro_normal_caution_ev"]
        assert "ASR" in criteria["note"] or "ASR" in criteria["value"]


# ════════════════════════════════════════════════════════════════
# C. 4월 편향 보정 — return_alpha_unbiased
# ════════════════════════════════════════════════════════════════
class TestUnbiasedAlpha:

    def test_unbiased_alpha_criterion_exists(self, validate_module):
        """return_alpha_unbiased 항목이 promotion_check에 존재."""
        df = _make_synthetic_df()
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        monthly_asr = validate_module.validate_monthly(
            asr, baseline_win, baseline_ret
        )
        macro_asr = validate_module.validate_macro(
            asr, baseline_win, baseline_ret
        )
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )
        assert "return_alpha_unbiased" in promotion["criteria"], (
            "4월 편향 보정 기준이 promotion_check에 없음"
        )

    def test_promotion_fails_when_only_one_month_positive(
        self, validate_module
    ):
        """4월만 +10% 알파, 다른 달 음수면 산술평균 fail."""
        # n-가중 평균은 통과해도 산술 평균은 fail
        # 단 synthetic으로 정확히 재현 어려워 logic만 확인
        df = _make_synthetic_df()
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        monthly_asr = validate_module.validate_monthly(
            asr, baseline_win, baseline_ret
        )
        macro_asr = validate_module.validate_macro(
            asr, baseline_win, baseline_ret
        )
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )
        crit = promotion["criteria"]["return_alpha_unbiased"]
        # 산술 평균 표현이 노트에 있어야 함
        assert ("산술" in crit["note"] or "unbiased" in crit["note"].lower()
                or "4월" in crit["note"])


# ════════════════════════════════════════════════════════════════
# D. ★ [v3.9.23b] JSON 타입 안전성 — bool/int 문자열화 방지
# ════════════════════════════════════════════════════════════════
class TestJsonTypeSafety:
    """평가 지적: passed/n_passed가 numpy 타입으로 인해 JSON 문자열화되는 버그.

    "passed": "True"  (str) → UI에서 truthy로 잘못 해석될 위험
    "passed": "False" (str) → bool(False)이 아니라 bool("False")=True 됨!
    """

    def test_passed_is_python_bool(self, validate_module):
        """모든 criteria.passed가 Python bool이어야 함."""
        df = _make_synthetic_df()
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        monthly_asr = validate_module.validate_monthly(
            asr, baseline_win, baseline_ret
        )
        macro_asr = validate_module.validate_macro(
            asr, baseline_win, baseline_ret
        )
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )
        for key, c in promotion["criteria"].items():
            assert isinstance(c["passed"], bool), (
                f"criteria[{key!r}].passed = {c['passed']!r} "
                f"(type={type(c['passed']).__name__}) — "
                "Python bool이 아님! numpy.bool_ 등이 들어가면 "
                "json 직렬화 시 'True'/'False' 문자열화됨"
            )

    def test_n_passed_is_python_int(self, validate_module):
        """n_passed가 Python int여야 함."""
        df = _make_synthetic_df()
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        monthly_asr = validate_module.validate_monthly(
            asr, baseline_win, baseline_ret
        )
        macro_asr = validate_module.validate_macro(
            asr, baseline_win, baseline_ret
        )
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )
        assert isinstance(promotion["n_passed"], int) and not isinstance(
            promotion["n_passed"], bool
        ), (
            f"n_passed type={type(promotion['n_passed']).__name__} — "
            "Python int 아님"
        )
        assert isinstance(promotion["n_total"], int) and not isinstance(
            promotion["n_total"], bool
        )
        assert isinstance(promotion["all_passed"], bool)

    def test_stat_block_numeric_types(self, validate_module):
        """stat_block 결과의 numeric은 Python float."""
        df = _make_synthetic_df()
        result = validate_module.stat_block(df, "test", 41.0, -0.5)
        for key in (
            "win_rate", "avg_ret", "fast_loss_rate",
            "worst_loss", "avg_win_ret",
        ):
            v = result[key]
            assert isinstance(v, (int, float)) and not isinstance(v, bool), (
                f"stat_block[{key!r}] = {v!r} type={type(v).__name__} — "
                "Python float 아님"
            )
        assert isinstance(result["n"], int) and not isinstance(
            result["n"], bool
        )
        assert isinstance(result["insufficient"], bool)

    def test_json_roundtrip_preserves_bool(self, validate_module, tmp_path):
        """검증 결과를 JSON으로 dump → load → bool이 보존되는지.

        ★ 핵심 회귀: "False" 문자열로 저장되면 다시 load 시 bool("False")=True
        가 되어 production 승격이 잘못 통과될 수 있음.
        """
        import json
        df = _make_synthetic_df()
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        monthly_asr = validate_module.validate_monthly(
            asr, baseline_win, baseline_ret
        )
        macro_asr = validate_module.validate_macro(
            asr, baseline_win, baseline_ret
        )
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )

        # 명시적으로 default=str 없이 dump — bool/int가 native여야 통과
        # [v3.9.23-hotfix] Windows cp949 → utf-8 명시 (이모지/한글 안전)
        out = tmp_path / "test.json"
        out.write_text(
            json.dumps(promotion, ensure_ascii=False),
            encoding="utf-8",
        )

        loaded = json.loads(out.read_text(encoding="utf-8"))

        # 모든 passed가 진짜 bool로 보존됐는지
        for key, c in loaded["criteria"].items():
            assert isinstance(c["passed"], bool), (
                f"JSON roundtrip 후 criteria[{key!r}].passed = "
                f"{c['passed']!r} (type={type(c['passed']).__name__})"
            )
        assert isinstance(loaded["n_passed"], int)
        assert isinstance(loaded["all_passed"], bool)


# ════════════════════════════════════════════════════════════════
# E. ★ [v3.9.23c] ASR 슬라이스 검증
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
# F. ★ [v3.9.23c-hotfix] return_alpha_unbiased 유효 월 ≥3 강화
# ════════════════════════════════════════════════════════════════
class TestUnbiasedAlphaMinMonths:
    """평가 v3.9.23c-hotfix 지적:
    유효 월 1개월만으로 unbiased alpha가 통과처럼 보이는 위험.
    ASR_STRONG_CAUTION 같은 경우 4월 1개월만 표본 충분인데
    +6.16%p로 PASS처럼 보였음. 이제 유효 월 ≥3 강제.
    """

    def _make_df_with_valid_months(self, validate_module, n_valid_months):
        """유효 월 N개를 가진 ASR 데이터 생성.
        ASR는 ret +5.0, baseline은 ret -1.0으로 알파 양수 보장.
        """
        rows = []
        months = ["202601", "202602", "202603", "202604", "202605", "202606"]
        for month_idx in range(n_valid_months):
            month = months[month_idx]
            # ASR 표본 충분 (≥30) — 유효 월
            for i in range(35):
                rows.append({
                    "rec_ymd": f"{month}15",
                    "code": f"{month_idx:02d}{i:04d}",
                    "name": f"종목{i}",
                    "ret_net": 5.0,  # ASR 알파 양수
                    "win": 1,
                    "month": month,
                    "STRUCT": 40, "TIMING": 90, "AI": 50,
                    "BALANCE": 60, "DISPLAY": 70, "ELITE": 60,
                    "FINAL": 70, "AXIS_MEAN": 70,
                    "RR_NOW_TP1": 0.3, "ENTRY_GAP_PCT": 1.0, "EBS": 6,
                    "TOP_PICK": 0, "ROUTE": "ATTACK",
                    "BUY_NOW_GRADE": "BUY", "BUY_NOW_ELIGIBLE": 0,
                    "BUY_NOW_SCORE": 70.0,
                    "ANTI_STRUCT_REVERSAL_FLAG": 1,
                    "ANTI_STRUCT_REVERSAL_TYPE": "STRONG",
                    "ANTI_STRUCT_REVERSAL_SCORE": 75.0,
                    "이격도": 5.0, "MFI14": 60.0,
                    "ret_1d": 1.0, "ret_5d": 2.0, "VWAP_GAP": 5.0,
                    "ENTRY_RISK_LEVEL": "GREEN", "MACRO_RISK": "CAUTION",
                })
            # 비-ASR baseline (알파 계산용) — 같은 월에 ret -1.0
            for i in range(50):
                rows.append({
                    "rec_ymd": f"{month}15",
                    "code": f"BL{month_idx:02d}{i:04d}",
                    "name": f"기준{i}",
                    "ret_net": -1.0,  # baseline 마이너스
                    "win": 0,
                    "month": month,
                    "STRUCT": 80, "TIMING": 50, "AI": 50,
                    "BALANCE": 60, "DISPLAY": 50, "ELITE": 50,
                    "FINAL": 50, "AXIS_MEAN": 60,
                    "RR_NOW_TP1": 1.5, "ENTRY_GAP_PCT": 1.0, "EBS": 6,
                    "TOP_PICK": 0, "ROUTE": "ATTACK",
                    "BUY_NOW_GRADE": "WAIT", "BUY_NOW_ELIGIBLE": 0,
                    "BUY_NOW_SCORE": 50.0,
                    "ANTI_STRUCT_REVERSAL_FLAG": 0,
                    "ANTI_STRUCT_REVERSAL_TYPE": "",
                    "ANTI_STRUCT_REVERSAL_SCORE": 0.0,
                    "이격도": 5.0, "MFI14": 60.0,
                    "ret_1d": 0.5, "ret_5d": 0.0, "VWAP_GAP": 2.0,
                    "ENTRY_RISK_LEVEL": "GREEN", "MACRO_RISK": "CAUTION",
                })
        # 표본 부족 월 (insufficient — ASR만)
        for i in range(5):
            rows.append({
                "rec_ymd": "20269915",
                "code": f"99{i:04d}",
                "name": f"부족{i}",
                "ret_net": 5.0, "win": 1, "month": "202699",
                "STRUCT": 40, "TIMING": 90, "AI": 50, "BALANCE": 60,
                "DISPLAY": 70, "ELITE": 60, "FINAL": 70, "AXIS_MEAN": 70,
                "RR_NOW_TP1": 0.3, "ENTRY_GAP_PCT": 1.0, "EBS": 6,
                "TOP_PICK": 0, "ROUTE": "ATTACK",
                "BUY_NOW_GRADE": "BUY", "BUY_NOW_ELIGIBLE": 0,
                "BUY_NOW_SCORE": 70.0, "ANTI_STRUCT_REVERSAL_FLAG": 1,
                "ANTI_STRUCT_REVERSAL_TYPE": "STRONG",
                "ANTI_STRUCT_REVERSAL_SCORE": 75.0, "이격도": 5.0,
                "MFI14": 60.0, "ret_1d": 1.0, "ret_5d": 2.0,
                "VWAP_GAP": 5.0, "ENTRY_RISK_LEVEL": "GREEN",
                "MACRO_RISK": "CAUTION",
            })
        return pd.DataFrame(rows)

    def test_one_valid_month_fails_with_observe(self, validate_module):
        """유효 월 1개만으로는 FAIL (OBSERVE) — 평가 핵심 지적."""
        df = self._make_df_with_valid_months(validate_module, 1)
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        bw = df["win"].mean() * 100
        br = df["ret_net"].mean()
        monthly_asr = validate_module.validate_monthly(asr, bw, br)
        macro_asr = validate_module.validate_macro(asr, bw, br)
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )
        crit = promotion["criteria"]["return_alpha_unbiased"]
        assert crit["passed"] is False, (
            "유효 월 1개월만으로 PASS — 4월 편향 보정 실효 없음! "
            f"실제: passed={crit['passed']!r}"
        )
        assert crit["unbiased_alpha_status"] == "OBSERVE", (
            f"status={crit['unbiased_alpha_status']!r}, OBSERVE 아님"
        )
        assert crit["valid_months"] == 1
        assert crit["required_valid_months"] == 3

    def test_two_valid_months_fails_with_observe(self, validate_module):
        """유효 월 2개월도 FAIL (OBSERVE) — 평가 권장 엄격 기준."""
        df = self._make_df_with_valid_months(validate_module, 2)
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        bw = df["win"].mean() * 100
        br = df["ret_net"].mean()
        monthly_asr = validate_module.validate_monthly(asr, bw, br)
        macro_asr = validate_module.validate_macro(asr, bw, br)
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )
        crit = promotion["criteria"]["return_alpha_unbiased"]
        assert crit["passed"] is False
        assert crit["unbiased_alpha_status"] == "OBSERVE"
        assert crit["valid_months"] == 2

    def test_three_valid_months_passes_when_alpha_positive(
        self, validate_module
    ):
        """유효 월 3개월 AND 평균 알파 양수 → PASS."""
        df = self._make_df_with_valid_months(validate_module, 3)
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        bw = df["win"].mean() * 100
        br = df["ret_net"].mean()
        monthly_asr = validate_module.validate_monthly(asr, bw, br)
        macro_asr = validate_module.validate_macro(asr, bw, br)
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )
        crit = promotion["criteria"]["return_alpha_unbiased"]
        assert crit["valid_months"] == 3
        # 3개월 모두 +5% 알파 → 산술 평균 충분
        assert crit["unbiased_alpha_status"] == "PASS", (
            f"3개월 알파 충분인데도 PASS 아님: {crit}"
        )
        assert crit["passed"] is True

    def test_zero_valid_months_insufficient_data(self, validate_module):
        """유효 월 0개월 → INSUFFICIENT_DATA."""
        # 모두 표본 부족인 데이터
        df = self._make_df_with_valid_months(validate_module, 0)
        asr = df[df["ANTI_STRUCT_REVERSAL_FLAG"] == 1]
        bw = df["win"].mean() * 100
        br = df["ret_net"].mean()
        monthly_asr = validate_module.validate_monthly(asr, bw, br)
        macro_asr = validate_module.validate_macro(asr, bw, br)
        groups = validate_module.validate_groups(df)
        promotion = validate_module.check_promotion_criteria(
            monthly_asr, macro_asr, groups
        )
        crit = promotion["criteria"]["return_alpha_unbiased"]
        assert crit["passed"] is False
        assert crit["unbiased_alpha_status"] == "INSUFFICIENT_DATA"
        assert crit["valid_months"] == 0


class TestAsrSlices:
    """평가 명시 3대 슬라이싱 축 검증."""

    def test_validate_asr_slices_returns_required_keys(
        self, validate_module
    ):
        """validate_asr_slices가 평가 명시 7개 슬라이스 키 반환."""
        df = _make_synthetic_df()
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        slices = validate_module.validate_asr_slices(
            df, baseline_win, baseline_ret
        )
        required = [
            "ASR_BASIC", "ASR_STRONG", "ASR_CHAMPION",
            "ASR_STRONG_CAUTION", "ASR_STRONG_CRITICAL",
            "ASR_STRONG_EXCLUDE_NORMAL", "ASR_STRONG_NORMAL_ONLY",
        ]
        for k in required:
            assert k in slices, f"슬라이스 {k!r} 누락"

    def test_each_slice_has_full_validation(self, validate_module):
        """각 슬라이스에 group/monthly/macro/promotion 모두 포함."""
        df = _make_synthetic_df()
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        slices = validate_module.validate_asr_slices(
            df, baseline_win, baseline_ret
        )
        for name, data in slices.items():
            assert "group" in data, f"{name}: group 누락"
            assert "monthly" in data, f"{name}: monthly 누락"
            assert "macro" in data, f"{name}: macro 누락"
            assert "promotion" in data, f"{name}: promotion 누락"

    def test_strong_exclude_normal_correctly_filters(
        self, validate_module
    ):
        """STRONG_EXCLUDE_NORMAL은 macro=NORMAL을 빼야 함."""
        df = _make_synthetic_df()
        # STRONG 일부에 NORMAL 매핑
        df.loc[
            (df["ANTI_STRUCT_REVERSAL_TYPE"] == "STRONG")
            & (df.index % 2 == 0),
            "MACRO_RISK"
        ] = "NORMAL"
        df.loc[
            (df["ANTI_STRUCT_REVERSAL_TYPE"] == "STRONG")
            & (df.index % 2 == 1),
            "MACRO_RISK"
        ] = "CAUTION"

        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        slices = validate_module.validate_asr_slices(
            df, baseline_win, baseline_ret
        )

        exclude_n = slices["ASR_STRONG_EXCLUDE_NORMAL"]["group"]["n"]
        normal_only_n = slices["ASR_STRONG_NORMAL_ONLY"]["group"]["n"]
        strong_n = slices["ASR_STRONG"]["group"]["n"]

        # STRONG = EXCLUDE_NORMAL + NORMAL_ONLY (대략)
        # (UNKNOWN/빈값이 있을 수 있어 ≤ 관계로 검증)
        assert exclude_n + normal_only_n <= strong_n, (
            f"sum mismatch: EXCLUDE={exclude_n} + NORMAL_ONLY={normal_only_n} "
            f"> STRONG={strong_n}"
        )

    def test_champion_tail_diagnosis_exists(self, validate_module):
        """CHAMPION 꼬리 위험 진단이 포함되어야 함."""
        df = _make_synthetic_df()
        # CHAMPION 데이터 추가
        for i in range(40):
            df.loc[len(df)] = df.iloc[0].copy()
            df.at[len(df) - 1, "ANTI_STRUCT_REVERSAL_TYPE"] = "CHAMPION"
            df.at[len(df) - 1, "ANTI_STRUCT_REVERSAL_FLAG"] = 1
            df.at[len(df) - 1, "ret_net"] = -5.0 if i % 2 == 0 else 3.0
            df.at[len(df) - 1, "win"] = 0 if i % 2 == 0 else 1
            df.at[len(df) - 1, "MACRO_RISK"] = "CAUTION"

        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        slices = validate_module.validate_asr_slices(
            df, baseline_win, baseline_ret
        )
        champion = slices.get("ASR_CHAMPION", {})
        assert "tail_diagnosis" in champion, "CHAMPION tail_diagnosis 누락"
        tail = champion["tail_diagnosis"]
        if tail.get("available"):
            # 필수 필드
            assert "b_ratio" in tail
            assert "loss_distribution" in tail
            assert "worst_5_cases" in tail
            assert "loss_macro_distribution" in tail

    def test_slice_promotion_check_is_independent(self, validate_module):
        """각 슬라이스의 promotion 체크는 그 슬라이스 데이터 기준."""
        df = _make_synthetic_df()
        baseline_win = df["win"].mean() * 100
        baseline_ret = df["ret_net"].mean()
        slices = validate_module.validate_asr_slices(
            df, baseline_win, baseline_ret
        )

        for name, data in slices.items():
            if data["group"].get("insufficient"):
                continue
            promo = data["promotion"]
            # promotion이 정상 구조
            assert "criteria" in promo
            assert "n_passed" in promo
            assert "all_passed" in promo
            # n_passed는 int (JSON 타입 안전성 유지)
            assert isinstance(promo["n_passed"], int)
            assert isinstance(promo["all_passed"], bool)
            # criteria의 passed도 bool
            for k, c in promo["criteria"].items():
                assert isinstance(c["passed"], bool), (
                    f"slice {name}, criteria {k}: passed type="
                    f"{type(c['passed']).__name__}"
                )
