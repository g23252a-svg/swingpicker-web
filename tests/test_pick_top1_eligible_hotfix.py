"""
tests/test_pick_top1_eligible_hotfix.py
========================================
[v3.9.22b-hotfix] Top1 / Top3 선별 함수가 BUY_NOW_ELIGIBLE=1 종목만 잡는지 검증.

핵심 시나리오 (현대해상 20260520):
- ELITE_LABEL = "✅ 즉시진입" + ROUTE=ATTACK + RR=1.10 + 갭=0
- 하지만 TOP_PICK=0, BUY_NOW_ELIGIBLE=0
- 기존 코드: pick_top1이 fallback으로 이 종목을 Top1로 선택 (버그!)
- 핫픽스: BUY_NOW_ELIGIBLE=1 필터 추가 → Top1에서 제외

평가 명시:
- BUY_NOW_ELIGIBLE=1 종목만 Top1/Top3에 올린다
- legacy CSV (컬럼 없음)는 backward compat 유지
- 0건일 때 LDY_RANK/TOP_PICK/ROUTE fallback 절대 금지
"""
import sys
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def pick_module():
    """tab_stocks의 pick_top1/pick_top3 함수만 import.

    nicegui UI 의존성 우회 위해 핵심 함수만 따로 로드.
    """
    # nicegui 모킹 (run/ui/app/observables 등 모두)
    if "nicegui" not in sys.modules:
        nicegui_mock = type(sys)("nicegui")
        for attr in ("ui", "app", "run", "Tailwind", "observables", "events"):
            setattr(nicegui_mock, attr, type(sys)(f"nicegui.{attr}"))
            sys.modules[f"nicegui.{attr}"] = getattr(nicegui_mock, attr)
        sys.modules["nicegui"] = nicegui_mock

    for mod in list(sys.modules.keys()):
        if mod == "components.tab_stocks":
            del sys.modules[mod]
    try:
        import components.tab_stocks as ts
        return ts
    except ImportError as e:
        pytest.skip(f"components.tab_stocks import 실패: {e}")


def _make_df(rows):
    """추천 종목 DataFrame 생성.

    필수 컬럼: ELITE_LABEL / ELITE_RANK_SCORE / 종목코드 / 종목명 /
              RR_NOW_TP1 / GAP_PCT / ROUTE / BUY_NOW_ELIGIBLE / TOP_PICK
    """
    defaults = {
        "종목코드": "001450",
        "종목명": "테스트종목",
        "ELITE_LABEL": "🛡️ 콤보",
        "ELITE_RANK_SCORE": 80.0,
        "RR_NOW_TP1": 1.50,
        "GAP_PCT": 0.0,
        "ROUTE": "ATTACK",
        "TOP_PICK": 1,
        "BUY_NOW_ELIGIBLE": 1,
        "업종": "테스트업종",
    }
    data = []
    for i, row in enumerate(rows):
        d = dict(defaults)
        d["종목코드"] = f"{int(d['종목코드']) + i:06d}"
        d.update(row)
        data.append(d)
    return pd.DataFrame(data)


# ════════════════════════════════════════════════════════════════
# A. 콤보 1순위 — BUY_NOW_ELIGIBLE=1만
# ════════════════════════════════════════════════════════════════
class TestComboPath:
    """🛡️ 콤보 라벨 → BUY_NOW_ELIGIBLE=1만 Top1."""

    def test_combo_eligible_picked(self, pick_module):
        """콤보 + ELIGIBLE=1 → Top1."""
        df = _make_df([{
            "ELITE_LABEL": "🛡️ 콤보",
            "ELITE_RANK_SCORE": 90.0,
            "BUY_NOW_ELIGIBLE": 1,
        }])
        result = pick_module.pick_top1(df)
        assert len(result) == 1, "ELIGIBLE=1 콤보는 Top1로 잡혀야 함"

    def test_combo_not_eligible_excluded(self, pick_module):
        """콤보지만 ELIGIBLE=0 → Top1에서 제외."""
        df = _make_df([{
            "ELITE_LABEL": "🛡️ 콤보",
            "ELITE_RANK_SCORE": 90.0,
            "BUY_NOW_ELIGIBLE": 0,
        }])
        result = pick_module.pick_top1(df)
        assert result == [], (
            "ELIGIBLE=0이면 콤보여도 Top1 제외 (LDY_RANK fallback 금지)"
        )

    def test_combo_eligible_priority(self, pick_module):
        """ELIGIBLE=1 콤보가 ELIGIBLE=0 콤보보다 우선."""
        # ELIGIBLE=0이 RANK_SCORE 더 높아도 ELIGIBLE=1만 잡혀야 함
        df = _make_df([
            {
                "종목코드": "100001",
                "종목명": "차단대상",
                "ELITE_LABEL": "🛡️ 콤보",
                "ELITE_RANK_SCORE": 95.0,
                "BUY_NOW_ELIGIBLE": 0,
            },
            {
                "종목코드": "100002",
                "종목명": "정상후보",
                "ELITE_LABEL": "🛡️ 콤보",
                "ELITE_RANK_SCORE": 80.0,
                "BUY_NOW_ELIGIBLE": 1,
            },
        ])
        result = pick_module.pick_top1(df)
        assert result == ["100002"], (
            "RANK_SCORE 낮아도 ELIGIBLE=1이 선택돼야 함"
        )


# ════════════════════════════════════════════════════════════════
# B. 즉시진입 fallback — BUY_NOW_ELIGIBLE=1만 (현대해상 시나리오)
# ════════════════════════════════════════════════════════════════
class TestInstantFallback:
    """✅ 즉시진입 fallback에서도 ELIGIBLE 필터 작동."""

    def test_hyundai_marine_blocked(self, pick_module):
        """★ 현대해상 시나리오 (20260520):
        ELITE_LABEL='✅ 즉시진입', ROUTE='ATTACK', RR=1.10, 갭=0
        BUY_NOW_ELIGIBLE=0 (TOP_PICK=0이라서)
        → Top1에 못 올라가야 함
        """
        df = _make_df([{
            "종목코드": "001450",
            "종목명": "현대해상",
            "ELITE_LABEL": "✅ 즉시진입",
            "ELITE_RANK_SCORE": 62.7,
            "RR_NOW_TP1": 1.10,
            "GAP_PCT": 0.0,
            "ROUTE": "ATTACK",
            "TOP_PICK": 0,
            "BUY_NOW_ELIGIBLE": 0,
        }])
        result = pick_module.pick_top1(df)
        assert result == [], (
            "현대해상 (ELIGIBLE=0)은 Top1에서 제외돼야 함 — "
            "fallback 절대 금지"
        )

    def test_instant_eligible_picked(self, pick_module):
        """즉시진입 + ELIGIBLE=1 → fallback Top1."""
        df = _make_df([{
            "종목코드": "002000",
            "종목명": "정상즉시진입",
            "ELITE_LABEL": "✅ 즉시진입",
            "ELITE_RANK_SCORE": 75.0,
            "RR_NOW_TP1": 1.50,
            "GAP_PCT": 0.0,
            "ROUTE": "ATTACK",
            "TOP_PICK": 1,
            "BUY_NOW_ELIGIBLE": 1,
        }])
        result = pick_module.pick_top1(df)
        assert result == ["002000"]

    def test_instant_route_neutral_excluded(self, pick_module):
        """ROUTE=NEUTRAL이면 ELIGIBLE=1이어도 제외 (기존 필터 유지)."""
        df = _make_df([{
            "ELITE_LABEL": "✅ 즉시진입",
            "ELITE_RANK_SCORE": 75.0,
            "ROUTE": "NEUTRAL",
            "BUY_NOW_ELIGIBLE": 1,
        }])
        result = pick_module.pick_top1(df)
        assert result == []

    def test_instant_rr_low_excluded(self, pick_module):
        """RR<1.0이면 ELIGIBLE=1이어도 제외 (기존 필터 유지)."""
        df = _make_df([{
            "ELITE_LABEL": "✅ 즉시진입",
            "ELITE_RANK_SCORE": 75.0,
            "RR_NOW_TP1": 0.95,
            "BUY_NOW_ELIGIBLE": 1,
        }])
        result = pick_module.pick_top1(df)
        assert result == []


# ════════════════════════════════════════════════════════════════
# C. Legacy CSV 호환 — BUY_NOW_ELIGIBLE 컬럼 없음
# ════════════════════════════════════════════════════════════════
class TestLegacyCompat:
    """v3.9.22a 이전 CSV는 컬럼 없으니 backward compat."""

    def test_no_eligible_column_combo_picked(self, pick_module):
        """ELIGIBLE 컬럼 없는 legacy CSV → 콤보 정상 선택."""
        df = _make_df([{
            "ELITE_LABEL": "🛡️ 콤보",
            "ELITE_RANK_SCORE": 90.0,
        }])
        # ELIGIBLE 컬럼 제거
        df = df.drop(columns=["BUY_NOW_ELIGIBLE"])
        result = pick_module.pick_top1(df)
        assert len(result) == 1, (
            "legacy CSV는 ELIGIBLE 필터 없이 기존 동작 유지"
        )

    def test_no_eligible_column_instant_fallback(self, pick_module):
        """ELIGIBLE 컬럼 없는 legacy CSV → 즉시진입 fallback 정상."""
        df = _make_df([{
            "ELITE_LABEL": "✅ 즉시진입",
            "ELITE_RANK_SCORE": 75.0,
        }])
        df = df.drop(columns=["BUY_NOW_ELIGIBLE"])
        result = pick_module.pick_top1(df)
        assert len(result) == 1


# ════════════════════════════════════════════════════════════════
# D. pick_top3도 동일하게 적용
# ════════════════════════════════════════════════════════════════
class TestPickTop3Eligible:

    def test_top3_filters_non_eligible(self, pick_module):
        """Top3에서도 ELIGIBLE=0 종목 제외."""
        df = _make_df([
            {
                "종목코드": "100001",
                "ELITE_LABEL": "🛡️ 콤보",
                "ELITE_RANK_SCORE": 95.0,
                "BUY_NOW_ELIGIBLE": 1,
                "업종": "A업종",
            },
            {
                "종목코드": "100002",
                "ELITE_LABEL": "🛡️ 콤보",
                "ELITE_RANK_SCORE": 90.0,
                "BUY_NOW_ELIGIBLE": 0,  # ★ 차단
                "업종": "B업종",
            },
            {
                "종목코드": "100003",
                "ELITE_LABEL": "🛡️ 콤보",
                "ELITE_RANK_SCORE": 85.0,
                "BUY_NOW_ELIGIBLE": 1,
                "업종": "C업종",
            },
        ])
        result = pick_module.pick_top3(df)
        assert "100002" not in result, "ELIGIBLE=0 종목 제외"
        assert "100001" in result and "100003" in result

    def test_top3_all_blocked_returns_empty(self, pick_module):
        """모든 후보가 ELIGIBLE=0 → 빈 리스트."""
        df = _make_df([
            {
                "종목코드": "100001",
                "ELITE_LABEL": "🛡️ 콤보",
                "ELITE_RANK_SCORE": 95.0,
                "BUY_NOW_ELIGIBLE": 0,
            },
            {
                "종목코드": "100002",
                "ELITE_LABEL": "✅ 즉시진입",
                "ELITE_RANK_SCORE": 80.0,
                "BUY_NOW_ELIGIBLE": 0,
            },
        ])
        result = pick_module.pick_top3(df)
        assert result == [], "모두 ELIGIBLE=0이면 빈 리스트 (fallback 금지)"


# ════════════════════════════════════════════════════════════════
# E. ELIGIBLE 다양한 값 형식 — string/bool/int
# ════════════════════════════════════════════════════════════════
class TestEligibleRobustness:

    @pytest.mark.parametrize("val,expected_picked", [
        (1, True),
        (1.0, True),
        ("1", True),
        ("True", True),
        ("TRUE", True),
        ("Y", True),
        (0, False),
        ("0", False),
        ("False", False),
        ("", False),
        (None, False),
    ])
    def test_eligible_value_formats(self, pick_module, val, expected_picked):
        """다양한 ELIGIBLE 값 형식 모두 정확히 인식."""
        df = _make_df([{
            "ELITE_LABEL": "🛡️ 콤보",
            "ELITE_RANK_SCORE": 90.0,
            "BUY_NOW_ELIGIBLE": val,
        }])
        result = pick_module.pick_top1(df)
        if expected_picked:
            assert len(result) == 1, f"{val!r} → 선택돼야 함"
        else:
            assert result == [], f"{val!r} → 제외돼야 함"
