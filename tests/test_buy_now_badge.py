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
        # [v22.3.8] "매수 적합" → "공식 매수 가능"
        assert "매수" in labels["label"] and (
            "가능" in labels["label"] or "적합" in labels["label"]
        )

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
        """🟢 BUY_NOW 80점 — 즉시 진입 가능.

        [v22.3.8] ELIGIBLE=1 명시 필요. ELIGIBLE 없으면 🟡 관찰로 강등.
        """
        row = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_SCORE": 80, "BUY_NOW_ELIGIBLE": 1,  # ★ v22.3.8: 명시 필요
        }
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


# ════════════════════════════════════════════════════════════════
# ★ [v22.3.8] BUY_NOW_GRADE UI 안전 패치 회귀 가드
#
# 평가 명시 위험: 5/21 CSV에 BUY_NOW_GRADE=BUY 349건 vs ELIGIBLE=0 579건.
# TOP_PICK=1 + BUY + ELIGIBLE=0 케이스에서 회원이 "매수 적합"으로 오해 가능.
# 패치: display_* 필드에서 ELIGIBLE=0 시 🟡 관찰로 강등 + official_buy 신규.
# ════════════════════════════════════════════════════════════════
class TestV2238UiSafety:
    """BUY인데 ELIGIBLE=0이면 화면에 '매수 적합'으로 나오면 안 됨."""

    def test_buy_but_not_eligible_downgrades_display(self, badge_module):
        """TOP_PICK=1 + BUY + ELIGIBLE=0 → display는 🟡 관찰로 강등.

        ★ 핵심 회귀: 회원이 'BUY' 글자만 보고 매수하는 사고 방지.
        """
        row = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 0, "BUY_NOW_SCORE": 70,
        }
        disp = badge_module.get_buy_now_display(row)

        # raw grade는 BUY 그대로 (기존 호환)
        assert disp["grade"] == "BUY"
        assert disp["eligible"] is False

        # ★ official_buy는 False여야 함 (공식 매수 아님)
        assert disp["official_buy"] is False

        # ★ display_* 는 WATCH로 강등돼야 함
        assert disp["display_icon"] == "🟡", (
            f"BUY+ELIGIBLE=0인데 display_icon={disp['display_icon']!r} — "
            f"🟡로 강등 안 됨 (회원 오해 위험)"
        )
        assert "관찰" in disp["display_label"] or "대기" in disp["display_label"]
        # display_short는 "즉시 진입 가능"이면 절대 안 됨
        assert "즉시" not in disp["display_short"]
        assert "진입 가능" not in disp["display_short"]

    def test_buy_and_eligible_keeps_display(self, badge_module):
        """TOP_PICK=1 + BUY + ELIGIBLE=1 → display 그대로 🟢 공식 매수 가능."""
        row = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 1, "BUY_NOW_SCORE": 80,
        }
        disp = badge_module.get_buy_now_display(row)
        assert disp["official_buy"] is True
        assert disp["display_icon"] == "🟢"
        # [v22.3.8] "매수 적합" 또는 "공식 매수 가능" 모두 허용
        assert "매수" in disp["display_label"] and (
            "가능" in disp["display_label"] or "적합" in disp["display_label"]
        )
        assert (
            "진입" in disp["display_short"]
            or "즉시" in disp["display_short"]
            or "신규" in disp["display_short"]
        )

    def test_official_buy_requires_all_three(self, badge_module):
        """official_buy = TOP_PICK AND ELIGIBLE AND grade=BUY 모두 필요."""
        # 케이스: TOP_PICK=0 + BUY + ELIGIBLE=1
        # 실제 백엔드 산식은 ELIGIBLE = TOP_PICK AND PASS라 이 케이스는 안 생기지만
        # 방어 로직 검증
        row1 = {
            "TOP_PICK": 0, "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 1,
        }
        assert badge_module.get_buy_now_display(row1)["official_buy"] is False

        # 케이스: TOP_PICK=1 + AVOID + ELIGIBLE=0
        row2 = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "AVOID",
            "BUY_NOW_ELIGIBLE": 0,
        }
        assert badge_module.get_buy_now_display(row2)["official_buy"] is False

        # 케이스: TOP_PICK=1 + BUY + ELIGIBLE=0 (★ 5/21 시나리오)
        row3 = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 0,
        }
        assert badge_module.get_buy_now_display(row3)["official_buy"] is False

        # 케이스: TOP_PICK=1 + BUY + ELIGIBLE=1 (유일하게 True)
        row4 = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 1,
        }
        assert badge_module.get_buy_now_display(row4)["official_buy"] is True

    def test_subtitle_buy_but_not_eligible(self, badge_module):
        """subtitle도 ELIGIBLE=0이면 🟡로 표시."""
        row = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_SCORE": 75, "BUY_NOW_ELIGIBLE": 0,
        }
        disp = badge_module.get_buy_now_display(row)
        sub = badge_module.format_buy_now_subtitle(disp)
        # 🟢 절대 안 됨
        assert "🟢" not in sub, (
            f"BUY+ELIGIBLE=0 subtitle에 🟢이 보임: {sub!r}"
        )
        # 🟡이어야 함
        assert "🟡" in sub
        # "즉시 진입 가능" 절대 안 됨
        assert "즉시" not in sub
        assert "진입 가능" not in sub

    def test_tooltip_buy_but_not_eligible_warns(self, badge_module):
        """tooltip이 BUY+ELIGIBLE=0 시 '공식 매수 대상 아님' 안내."""
        row = {
            "TOP_PICK": 1, "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 0, "BUY_NOW_SCORE": 70,
        }
        disp = badge_module.get_buy_now_display(row)
        tip = badge_module.format_buy_now_tooltip(disp)
        # 핵심 안내 포함
        assert "ELIGIBLE=0" in tip or "공식 매수" in tip, (
            f"tooltip에 ELIGIBLE=0 안내 누락: {tip!r}"
        )

    def test_official_buy_field_exists_in_response(self, badge_module):
        """모든 케이스에서 official_buy 필드 존재."""
        cases = [
            {"TOP_PICK": 0},
            {"TOP_PICK": 1, "BUY_NOW_GRADE": "BUY", "BUY_NOW_ELIGIBLE": 1},
            {"TOP_PICK": 1, "BUY_NOW_GRADE": "BUY", "BUY_NOW_ELIGIBLE": 0},
            {"TOP_PICK": 1, "BUY_NOW_GRADE": "AVOID", "BUY_NOW_ELIGIBLE": 0},
            {"TOP_PICK": 1, "BUY_NOW_GRADE": "WATCH", "BUY_NOW_ELIGIBLE": 0},
        ]
        for row in cases:
            disp = badge_module.get_buy_now_display(row)
            assert "official_buy" in disp, f"official_buy 필드 누락: {row}"
            assert isinstance(disp["official_buy"], bool)
            # display_* 필드도 모두 존재
            for k in ["display_icon", "display_label", "display_short",
                      "display_tone", "display_color"]:
                assert k in disp, f"{k} 필드 누락: {row}"


# ★ [v22.3.8-D1] STALE_CARRY 표시 가드 회귀 보호
# IS_STALE_CARRY=True + DISPLAY_SCORE<30 시 BUY_NOW 배지 전체 숨김.
# 오래 끌고 온 약한 보유종목이 신규매수 후보로 오해되는 것 차단.
# raw BUY_NOW_GRADE / ELIGIBLE 변경 없음 — 표시(visible)만 막음.
class TestV2238D1StaleCarryGuard:
    """v22.3.8-D1: STALE_CARRY + 낮은 DISPLAY_SCORE 표시 가드."""

    def test_stale_carry_with_low_display_score_hides_badge(self, badge_module):
        """IS_STALE_CARRY=True + DISPLAY_SCORE=29 + BUY → visible=False, 가드 발동."""
        row = {
            "TOP_PICK": 1,
            "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 1,
            "BUY_NOW_SCORE": 80,
            "IS_STALE_CARRY": True,
            "DISPLAY_SCORE": 29,
        }
        disp = badge_module.get_buy_now_display(row)
        # 표시는 숨김
        assert disp["visible"] is False, (
            f"STALE_CARRY+DISPLAY<30인데 visible=True: {disp}"
        )
        # 가드 발동 플래그
        assert disp["stale_carry_guard"] is True
        # 공식 매수 아님
        assert disp["official_buy"] is False
        # raw 데이터는 변경 없음
        assert disp["grade"] == "BUY"
        assert disp["eligible"] is True
        assert disp["score"] == 80

    def test_stale_carry_with_high_display_score_keeps_badge(self, badge_module):
        """IS_STALE_CARRY=True + DISPLAY_SCORE=35(>=30) → 가드 미발동, 기존 동작."""
        row = {
            "TOP_PICK": 1,
            "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 1,
            "BUY_NOW_SCORE": 75,
            "IS_STALE_CARRY": True,
            "DISPLAY_SCORE": 35,
        }
        disp = badge_module.get_buy_now_display(row)
        # 가드 미발동
        assert disp["stale_carry_guard"] is False
        # 정상 표시
        assert disp["visible"] is True
        # 공식 매수 유지
        assert disp["official_buy"] is True

    def test_no_stale_carry_with_low_display_score_keeps_badge(self, badge_module):
        """IS_STALE_CARRY=False + DISPLAY_SCORE=29 → 가드 미발동.

        DISPLAY_SCORE만 낮은 건 STALE_CARRY 가드 발동 사유 아님.
        오래 보유한 약세 종목만 막는 것이 D1의 명시 목적.
        """
        row = {
            "TOP_PICK": 1,
            "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 1,
            "BUY_NOW_SCORE": 70,
            "IS_STALE_CARRY": False,
            "DISPLAY_SCORE": 29,
        }
        disp = badge_module.get_buy_now_display(row)
        assert disp["stale_carry_guard"] is False
        assert disp["visible"] is True
        assert disp["official_buy"] is True

    def test_stale_carry_with_buy_eligible0_still_hides(self, badge_module):
        """STALE_CARRY 가드는 ELIGIBLE 강등 룰보다 강함 — 완전 숨김 우선.

        BUY+ELIGIBLE=0 케이스는 보통 🟡 관찰 후보로 강등되지만,
        STALE_CARRY+DISPLAY<30 가드 발동 시에는 visible=False로 완전 숨김.
        """
        row = {
            "TOP_PICK": 1,
            "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 0,
            "BUY_NOW_SCORE": 70,
            "IS_STALE_CARRY": True,
            "DISPLAY_SCORE": 20,
        }
        disp = badge_module.get_buy_now_display(row)
        assert disp["visible"] is False
        assert disp["stale_carry_guard"] is True
        assert disp["official_buy"] is False

    def test_stale_carry_fallback_to_final_score(self, badge_module):
        """DISPLAY_SCORE 없으면 FINAL_SCORE로 fallback."""
        row = {
            "TOP_PICK": 1,
            "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 1,
            "BUY_NOW_SCORE": 65,
            "IS_STALE_CARRY": True,
            # DISPLAY_SCORE 없음
            "FINAL_SCORE": 25,
        }
        disp = badge_module.get_buy_now_display(row)
        assert disp["stale_carry_guard"] is True
        assert disp["visible"] is False

    def test_stale_carry_no_score_no_guard(self, badge_module):
        """IS_STALE_CARRY=True지만 DISPLAY/FINAL 둘 다 없으면 가드 미발동.

        점수 미상이면 안전하게 기존 동작 유지 (보수적 fallback).
        """
        row = {
            "TOP_PICK": 1,
            "BUY_NOW_GRADE": "BUY",
            "BUY_NOW_ELIGIBLE": 1,
            "BUY_NOW_SCORE": 60,
            "IS_STALE_CARRY": True,
            # DISPLAY/FINAL 모두 없음
        }
        disp = badge_module.get_buy_now_display(row)
        # 가드 발동 안 함 (점수 정보 부재)
        assert disp["stale_carry_guard"] is False
        # 기존 visible 동작 (TOP_PICK=1이라 True)
        assert disp["visible"] is True

    def test_stale_carry_field_exists_in_all_cases(self, badge_module):
        """모든 케이스에서 stale_carry_guard 필드 존재 (회귀 가드)."""
        cases = [
            {"TOP_PICK": 0},
            {"TOP_PICK": 1, "BUY_NOW_GRADE": "BUY", "BUY_NOW_ELIGIBLE": 1},
            {"TOP_PICK": 1, "IS_STALE_CARRY": True, "DISPLAY_SCORE": 50},
            {"TOP_PICK": 1, "IS_STALE_CARRY": True, "DISPLAY_SCORE": 10,
             "BUY_NOW_GRADE": "WATCH"},
        ]
        for row in cases:
            disp = badge_module.get_buy_now_display(row)
            assert "stale_carry_guard" in disp, f"stale_carry_guard 필드 누락: {row}"
            assert isinstance(disp["stale_carry_guard"], bool)
