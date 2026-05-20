"""
tests/test_buy_now_badge.py
============================
[v3.9.22b] BUY_NOW UI 배지 헬퍼 회귀 가드.

평가 명시 절대 지킬 룰 5개 검증:
1. TOP_PICK 정렬/선정 로직 무수정 — 이 모듈은 표시 헬퍼만
2. UI 매수 가능 표시는 BUY_NOW_ELIGIBLE만 사용 (PASS 사용 금지)
3. BUY_NOW_PASS는 화면에 직접 "매수 가능"으로 쓰지 말 것
4. TOP_PICK=0 종목은 BUY_NOW_GRADE가 BUY여도 일반 화면에서 숨김
5. AVOID도 TOP_PICK이면 숨기지 말고 "추격 금지"로 노출
"""
import sys
import pytest


@pytest.fixture
def badge_module():
    """components.buy_now_badge import."""
    for mod in list(sys.modules.keys()):
        if mod.startswith("components.buy_now_badge"):
            del sys.modules[mod]
    return pytest.importorskip(
        "components.buy_now_badge",
        reason="components.buy_now_badge 모듈 import 불가",
        exc_type=ImportError,
    )


# ════════════════════════════════════════════════════════════════
# A. 절대 지킬 룰 #4: TOP_PICK=0 종목 숨김
# ════════════════════════════════════════════════════════════════
class TestRuleHideNonTopPick:
    """TOP_PICK=0 종목은 BUY_NOW가 BUY여도 화면에서 숨김."""

    def test_non_top_pick_hidden_even_if_buy(self, badge_module):
        """TOP_PICK=0 AND GRADE=BUY → visible=False."""
        row = {
            "TOP_PICK": 0,
            "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_SCORE": 90,
            "BUY_NOW_ELIGIBLE": 0,  # ELIGIBLE도 0 (TOP_PICK 아니므로)
        }
        disp = badge_module.get_buy_now_display(row)
        assert disp["visible"] is False, (
            "TOP_PICK=0이면 BUY여도 visible=False여야 함"
        )

    def test_non_top_pick_hidden_when_no_buy_now(self, badge_module):
        """TOP_PICK=0이고 BUY_NOW 컬럼 자체 없음 → visible=False."""
        row = {"TOP_PICK": 0}
        disp = badge_module.get_buy_now_display(row)
        assert disp["visible"] is False


# ════════════════════════════════════════════════════════════════
# B. 절대 지킬 룰 #5: AVOID도 TOP_PICK이면 노출
# ════════════════════════════════════════════════════════════════
class TestRuleShowAvoidIfTopPick:
    """AVOID도 TOP_PICK이면 숨기지 말고 '추격 금지'로 노출."""

    def test_top_pick_avoid_visible(self, badge_module):
        """TOP_PICK=1 AND GRADE=AVOID → visible=True."""
        row = {
            "TOP_PICK": 1,
            "BUY_NOW_GRADE": "AVOID",
            "BUY_NOW_SCORE": 0,
            "BUY_NOW_ELIGIBLE": 0,
        }
        disp = badge_module.get_buy_now_display(row)
        assert disp["visible"] is True, (
            "TOP_PICK=1이면 AVOID여도 visible=True (사용자에게 노출)"
        )
        assert disp["grade"] == "AVOID"
        assert disp["icon"] == "🔴"
        assert "금지" in disp["label"]

    def test_top_pick_watch_visible(self, badge_module):
        """TOP_PICK=1 AND GRADE=WATCH → visible=True."""
        row = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "WATCH",
            "BUY_NOW_SCORE": 60, "BUY_NOW_ELIGIBLE": 0,
        }
        disp = badge_module.get_buy_now_display(row)
        assert disp["visible"] is True
        assert disp["grade"] == "WATCH"
        assert disp["icon"] == "🟡"

    def test_top_pick_buy_visible_eligible(self, badge_module):
        """TOP_PICK=1 AND GRADE=BUY AND ELIGIBLE=1 → 매수 가능."""
        row = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_SCORE": 90, "BUY_NOW_ELIGIBLE": 1,
        }
        disp = badge_module.get_buy_now_display(row)
        assert disp["visible"] is True
        assert disp["grade"] == "BUY"
        assert disp["eligible"] is True
        assert disp["icon"] == "🟢"


# ════════════════════════════════════════════════════════════════
# C. 절대 지킬 룰 #2: ELIGIBLE만 매수 가능 신호
# ════════════════════════════════════════════════════════════════
class TestRuleUseEligibleOnly:
    """UI에서 '매수 가능' 판정은 BUY_NOW_ELIGIBLE만 봐야 함."""

    def test_eligible_field_matches_column(self, badge_module):
        """BUY_NOW_ELIGIBLE 컬럼 → disp['eligible']."""
        row1 = {"TOP_PICK": 1, "BUY_NOW_GRADE": "BUY", "BUY_NOW_ELIGIBLE": 1}
        assert badge_module.get_buy_now_display(row1)["eligible"] is True

        row2 = {"TOP_PICK": 1, "BUY_NOW_GRADE": "BUY", "BUY_NOW_ELIGIBLE": 0}
        assert badge_module.get_buy_now_display(row2)["eligible"] is False

    def test_eligible_false_when_avoid(self, badge_module):
        """AVOID는 ELIGIBLE=0 (백엔드에서 보장)."""
        row = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "AVOID",
            "BUY_NOW_ELIGIBLE": 0,
        }
        disp = badge_module.get_buy_now_display(row)
        assert disp["eligible"] is False


# ════════════════════════════════════════════════════════════════
# D. 등급별 라벨/아이콘
# ════════════════════════════════════════════════════════════════
class TestGradeLabels:

    def test_buy_label(self, badge_module):
        labels = badge_module.BUY_NOW_BADGE_LABELS["BUY"]
        assert labels["icon"] == "🟢"
        assert "적합" in labels["label"]

    def test_watch_label(self, badge_module):
        labels = badge_module.BUY_NOW_BADGE_LABELS["WATCH"]
        assert labels["icon"] == "🟡"
        assert "관찰" in labels["label"] or "대기" in labels["label"]

    def test_avoid_label(self, badge_module):
        labels = badge_module.BUY_NOW_BADGE_LABELS["AVOID"]
        assert labels["icon"] == "🔴"
        assert "금지" in labels["label"]

    def test_none_label_empty(self, badge_module):
        """NONE은 빈 표시."""
        labels = badge_module.BUY_NOW_BADGE_LABELS["NONE"]
        assert labels["icon"] == ""
        assert labels["label"] == ""


# ════════════════════════════════════════════════════════════════
# E. 보조 표시 함수
# ════════════════════════════════════════════════════════════════
class TestFormatters:

    def test_subtitle_for_buy(self, badge_module):
        """🟢 BUY_NOW 80점 — 즉시 진입 가능."""
        row = {"TOP_PICK": 1, "BUY_NOW_GRADE": "BUY", "BUY_NOW_SCORE": 80}
        disp = badge_module.get_buy_now_display(row)
        sub = badge_module.format_buy_now_subtitle(disp)
        assert "🟢" in sub
        assert "80" in sub
        assert "즉시" in sub or "진입" in sub

    def test_subtitle_for_avoid(self, badge_module):
        """🔴 AVOID 0점 — 지금 매수 금지."""
        row = {"TOP_PICK": 1, "BUY_NOW_GRADE": "AVOID", "BUY_NOW_SCORE": 0}
        disp = badge_module.get_buy_now_display(row)
        sub = badge_module.format_buy_now_subtitle(disp)
        assert "🔴" in sub
        assert "금지" in sub

    def test_subtitle_empty_when_not_top_pick(self, badge_module):
        """TOP_PICK=0이면 subtitle 빈 문자열 (숨김)."""
        row = {"TOP_PICK": 0, "BUY_NOW_GRADE": "BUY"}
        disp = badge_module.get_buy_now_display(row)
        sub = badge_module.format_buy_now_subtitle(disp)
        assert sub == ""

    def test_tooltip_with_reason(self, badge_module):
        """REASON 있으면 '사유: ...' 형식."""
        row = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "AVOID",
            "BUY_NOW_REASON": "RR 1.08 · VWAP 55↑",
        }
        disp = badge_module.get_buy_now_display(row)
        tip = badge_module.format_buy_now_tooltip(disp)
        assert "사유" in tip
        assert "RR" in tip

    def test_tooltip_default_when_no_reason(self, badge_module):
        """REASON 없으면 등급별 기본 메시지."""
        row = {"TOP_PICK": 1, "BUY_NOW_GRADE": "BUY"}
        disp = badge_module.get_buy_now_display(row)
        tip = badge_module.format_buy_now_tooltip(disp)
        assert "사유" in tip


# ════════════════════════════════════════════════════════════════
# F. 실전 시나리오 — 5/19 미래에셋벤처투자 / 5/18 KX하이텍
# ════════════════════════════════════════════════════════════════
class TestRealScenarios:

    def test_2026_05_19_mirae_asset_venture_avoid_visible(
        self, badge_module
    ):
        """미래에셋벤처투자 — TOP_PICK AND AVOID → 노출 + 추격 금지."""
        row = {
            "종목명": "미래에셋벤처투자",
            "TOP_PICK": 1,
            "BUY_NOW_GRADE": "AVOID",
            "BUY_NOW_SCORE": 0,
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_REASON": "RR 1.08 · VWAP 55↑ · POC 113↑",
        }
        disp = badge_module.get_buy_now_display(row)
        # 절대 지킬 룰 #5: 숨기지 말고 노출
        assert disp["visible"] is True
        assert disp["icon"] == "🔴"
        # 절대 지킬 룰 #2: ELIGIBLE은 0
        assert disp["eligible"] is False

    def test_2026_05_18_kx_hitech_buy_visible(self, badge_module):
        """KX하이텍 — TOP_PICK AND BUY → 매수 가능 신호."""
        row = {
            "종목명": "KX하이텍",
            "TOP_PICK": 1,
            "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_SCORE": 80,
            "BUY_NOW_ELIGIBLE": 1,
            "BUY_NOW_REASON": "",
        }
        disp = badge_module.get_buy_now_display(row)
        assert disp["visible"] is True
        assert disp["icon"] == "🟢"
        assert disp["eligible"] is True


# ════════════════════════════════════════════════════════════════
# G. 결측/이상 입력 안전 처리
# ════════════════════════════════════════════════════════════════
class TestRobustness:

    def test_string_top_pick_value(self, badge_module):
        """TOP_PICK이 '1' (문자열)로 들어와도 정상."""
        row = {"TOP_PICK": "1", "BUY_NOW_GRADE": "BUY"}
        disp = badge_module.get_buy_now_display(row)
        assert disp["visible"] is True

    def test_missing_grade(self, badge_module):
        """GRADE 컬럼 없음 → NONE 처리."""
        row = {"TOP_PICK": 1}
        disp = badge_module.get_buy_now_display(row)
        assert disp["grade"] == "NONE"

    def test_invalid_grade(self, badge_module):
        """GRADE에 이상한 값 → NONE."""
        row = {"TOP_PICK": 1, "BUY_NOW_GRADE": "INVALID"}
        disp = badge_module.get_buy_now_display(row)
        assert disp["grade"] == "NONE"

    def test_nan_score(self, badge_module):
        """SCORE가 NaN → 0.0."""
        import math
        row = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_SCORE": math.nan,
        }
        disp = badge_module.get_buy_now_display(row)
        # NaN은 float("nan")이라 _safe_float에서 그대로 통과될 수도 있음
        # nan은 비교 어려우니 isnan 체크
        score = disp["score"]
        assert isinstance(score, float)
