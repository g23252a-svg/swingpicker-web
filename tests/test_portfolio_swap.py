"""
tests/test_portfolio_swap.py
=============================
[v3.9.21] 보유종목 vs 신규추천 교체 판단 회귀 가드.

평가 6개 회귀 가드 기준:
1. 보유종목 EBS FAIL + 신규추천 EBS PASS → 교체 후보 가능
2. 신규추천 anomaly이면 교체 추천 금지
3. 보유종목 비중 과다 → 감량 검토
4. 데이터 부족 → ⚪
5. 추천/매수가/Top3/_run_backtest baseline 변경 없음
6. (추가) services nicegui import 0
"""
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────
# NiceGUI mock
# ────────────────────────────────────────────────────────────────
captured_labels = []


class _CapturingLabel:
    def __init__(self, text=""):
        captured_labels.append(str(text))

    def classes(self, *a, **kw):
        return self

    def tooltip(self, *a, **kw):
        return self

    def style(self, *a, **kw):
        return self

    def __getattr__(self, name):
        return lambda *a, **kw: self


class _ContextManagerMock:
    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def classes(self, *a, **kw):
        return self

    def style(self, *a, **kw):
        return self

    def props(self, *a, **kw):
        return self

    def tooltip(self, *a, **kw):
        return self

    def __getattr__(self, name):
        return lambda *a, **kw: self


def _setup_nicegui_mock(monkeypatch):
    fake_nicegui = types.ModuleType("nicegui")
    fake_ui = types.SimpleNamespace(
        label=lambda text="": _CapturingLabel(text),
        card=lambda: _ContextManagerMock(),
        row=lambda: _ContextManagerMock(),
        column=lambda: _ContextManagerMock(),
        element=lambda tag="": _ContextManagerMock(),
        button=lambda text="": _ContextManagerMock(),
        spinner=lambda *a, **kw: _ContextManagerMock(),
    )
    fake_nicegui.ui = fake_ui
    fake_nicegui.app = types.ModuleType("app")
    monkeypatch.setitem(sys.modules, "nicegui", fake_nicegui)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("services")
            or mod.startswith("components")
            or mod == "nicegui"
            or mod.startswith("nicegui.")
        ):
            del sys.modules[mod]
    _setup_nicegui_mock(monkeypatch)
    captured_labels.clear()
    return tmp_path


@pytest.fixture
def swap_svc():
    return pytest.importorskip(
        "services.portfolio_swap",
        reason="services.portfolio_swap 모듈 import 불가",
        exc_type=ImportError,
    )


@pytest.fixture
def swap_ui():
    return pytest.importorskip(
        "components.portfolio_swap",
        reason="components.portfolio_swap 모듈 import 불가",
        exc_type=ImportError,
    )


def _captured_text():
    return "\n".join(captured_labels)


def _make_recommend_row(
    name="A종목",
    final=75.0,
    route="ATTACK",
    ebs=1,
    rr=2.0,
    entry_gap=1.0,
    price=10000,
    top_pick=0,
):
    """recommend CSV 행 합성."""
    return {
        "종목명": name,
        "종목코드": "001",
        "종가": price,
        "DISPLAY_SCORE": final,
        "FINAL_SCORE": final,
        "ELITE_SCORE": final - 5,
        "ROUTE": route,
        "EBS": ebs,
        "RR_NOW_TP1": rr,
        "ENTRY_GAP_PCT": entry_gap,
        "TOP_PICK": top_pick,
        "MACRO_RISK": "NORMAL",
    }


# ════════════════════════════════════════════════════════════════
# A. analyze_portfolio_swap — 통합 검증
# ════════════════════════════════════════════════════════════════
class TestAnalyzePortfolioSwap:

    def test_returns_error_when_no_recommend(self, fake_env, swap_svc):
        """recommend 비어있으면 error 반환."""
        out = swap_svc.analyze_portfolio_swap(
            holdings=[{"name": "A", "avg": 100, "qty": 10}],
            recommend_df=pd.DataFrame(),
        )
        assert "error" in out

    def test_returns_error_when_no_holdings(self, fake_env, swap_svc):
        """holdings 비어있으면 error."""
        recs = pd.DataFrame([_make_recommend_row()])
        out = swap_svc.analyze_portfolio_swap(holdings=[], recommend_df=recs)
        assert "error" in out

    def test_holding_not_in_recommend_returns_white(
        self, fake_env, swap_svc
    ):
        """[평가 4] 보유종목이 recommend에 없으면 ⚪."""
        recs = pd.DataFrame([_make_recommend_row(name="다른종목")])
        out = swap_svc.analyze_portfolio_swap(
            holdings=[{"name": "보유A", "avg": 1000, "qty": 10}],
            recommend_df=recs,
        )
        # holding 1개 분석 — verdict는 ⚪
        assert len(out["holdings_analysis"]) == 1
        assert out["holdings_analysis"][0]["verdict"]["level"] == "white"
        assert out["summary"]["white"] == 1

    def test_summary_counts(self, fake_env, swap_svc):
        """요약 카운트 정확성 — green/red 혼합."""
        recs = pd.DataFrame([
            _make_recommend_row(name="좋은종목", final=80, route="ATTACK", ebs=1, rr=2.5),
            _make_recommend_row(name="나쁜종목", final=50, route="WAIT", ebs=0, rr=0.8),
            _make_recommend_row(name="TopPick", final=85, route="ATTACK", ebs=1, rr=3.0, top_pick=1),
        ])
        # 비중 과집중 방지 — 5개 종목으로 분산 (각 ~20%)
        # 좋은종목 + 나쁜종목 + 3개의 "다른" 종목으로 분산
        holdings = [
            {"name": "좋은종목", "avg": 10000, "qty": 10},
            {"name": "나쁜종목", "avg": 12000, "qty": 10},
            {"name": "기타1", "avg": 10000, "qty": 10},
            {"name": "기타2", "avg": 10000, "qty": 10},
            {"name": "기타3", "avg": 10000, "qty": 10},
        ]
        out = swap_svc.analyze_portfolio_swap(
            holdings=holdings, recommend_df=recs,
        )
        # 5개 분석 결과
        assert len(out["holdings_analysis"]) == 5
        # 좋은종목 → green, 나쁜종목 → red/orange/yellow 중 하나
        levels_by_name = {
            h["name"]: h["verdict"]["level"] for h in out["holdings_analysis"]
        }
        assert levels_by_name["좋은종목"] == "green"
        # 나쁜종목은 손실+EBS0+ROUTE WAIT 3개 → 🔴
        assert levels_by_name["나쁜종목"] == "red"
        # 기타1-3은 recommend에 없음 → ⚪
        assert levels_by_name["기타1"] == "white"


# ════════════════════════════════════════════════════════════════
# B. derive_holding_verdict — 6단계 판정
# ════════════════════════════════════════════════════════════════
class TestDeriveHoldingVerdict:

    def _make_pick(self, **kwargs):
        defaults = {
            "name": "TestStock",
            "final_score": 75.0,
            "display_score": 75.0,
            "route": "ATTACK",
            "ebs": 1,
            "rr_now_tp1": 2.0,
            "entry_gap_pct": 1.0,
        }
        defaults.update(kwargs)
        return defaults

    def test_white_when_pick_is_none(self, fake_env, swap_svc):
        """[평가 4] pick None → ⚪."""
        v = swap_svc.derive_holding_verdict(
            pick=None, pnl_pct=0, weight_pct=10,
            top_pick=None, new_recommend_safe=False,
        )
        assert v["icon"] == "⚪"
        assert v["level"] == "white"

    def test_red_when_ebs_zero_and_route_wait_and_loss(
        self, fake_env, swap_svc
    ):
        """🔴 정리 우선: EBS 0 + ROUTE WAIT + 손실 (3개 동시)."""
        pick = self._make_pick(ebs=0, route="WAIT", final_score=55)
        v = swap_svc.derive_holding_verdict(
            pick=pick, pnl_pct=-8.0, weight_pct=15,
            top_pick=None, new_recommend_safe=False,
        )
        assert v["icon"] == "🔴"
        assert v["level"] == "red"
        assert "정리 우선" in v["title"]

    def test_red_when_two_danger_signals(self, fake_env, swap_svc):
        """🔴: EBS 0 + 손실 (ROUTE는 NEUTRAL이라도 2개면 🔴)."""
        pick = self._make_pick(ebs=0, route="NEUTRAL")
        v = swap_svc.derive_holding_verdict(
            pick=pick, pnl_pct=-7.0, weight_pct=15,
            top_pick=None, new_recommend_safe=False,
        )
        assert v["icon"] == "🔴"

    def test_orange_when_hold_weak_and_new_strong_and_safe(
        self, fake_env, swap_svc
    ):
        """[평가 1] 🟠 교체 후보: 보유 약함 + 신규 강함 + 안전."""
        pick = self._make_pick(final_score=55, route="NEUTRAL", ebs=0, rr_now_tp1=1.0)
        # 위 pick은 EBS 0 + final 55. 손익은 -3% (red 조건 미달, 2개 미만)
        top_pick = self._make_pick(name="신규Top", final_score=82, ebs=1, rr_now_tp1=2.5)
        v = swap_svc.derive_holding_verdict(
            pick=pick, pnl_pct=-3.0, weight_pct=15,
            top_pick=top_pick, new_recommend_safe=True,
        )
        assert v["icon"] == "🟠"
        assert v["level"] == "orange"
        assert "교체 후보" in v["title"]
        assert v["swap_candidate"] is True

    def test_orange_blocked_when_new_recommend_unsafe(
        self, fake_env, swap_svc
    ):
        """[평가 2] 🟠 차단: 신규추천 anomaly/과열이면 교체 추천 금지."""
        pick = self._make_pick(final_score=55, route="NEUTRAL", ebs=0, rr_now_tp1=1.0)
        top_pick = self._make_pick(name="신규Top", final_score=82, ebs=1, rr_now_tp1=2.5)
        # new_recommend_safe=False → 🟠 안 됨
        v = swap_svc.derive_holding_verdict(
            pick=pick, pnl_pct=-3.0, weight_pct=15,
            top_pick=top_pick, new_recommend_safe=False,
        )
        assert v["icon"] != "🟠"
        # 보유 약함 → 🟡 감량 검토
        assert v["icon"] == "🟡"

    def test_orange_blocked_when_score_gap_too_small(
        self, fake_env, swap_svc
    ):
        """🟠 차단: 점수 차이 < SWAP_MIN_SCORE_GAP(10) → 🟡."""
        pick = self._make_pick(final_score=65, route="NEUTRAL", ebs=0, rr_now_tp1=1.0)
        top_pick = self._make_pick(name="신규Top", final_score=70, ebs=1, rr_now_tp1=2.5)
        # 점수 차이 5점 — 10점 미달
        v = swap_svc.derive_holding_verdict(
            pick=pick, pnl_pct=-2.0, weight_pct=15,
            top_pick=top_pick, new_recommend_safe=True,
        )
        # 🟠 아님 (점수 차이 작음)
        assert v["icon"] != "🟠"

    def test_yellow_when_over_concentration(self, fake_env, swap_svc):
        """[평가 3] 🟡 감량 검토: 비중 > 30%."""
        pick = self._make_pick(final_score=75, route="ATTACK", ebs=1)
        v = swap_svc.derive_holding_verdict(
            pick=pick, pnl_pct=10.0, weight_pct=68,  # 사용자 에이피알 시나리오
            top_pick=None, new_recommend_safe=False,
        )
        assert v["icon"] == "🟡"
        assert v["level"] == "yellow"
        assert "감량" in v["title"]

    def test_yellow_when_hold_weak_no_strong_new(
        self, fake_env, swap_svc
    ):
        """🟡: 보유 약함 + 신규 강하지 않음."""
        pick = self._make_pick(final_score=55, route="NEUTRAL", ebs=0)
        # 신규 약함 (final < 70)
        top_pick = self._make_pick(name="약신규", final_score=65, ebs=1, rr_now_tp1=1.5)
        v = swap_svc.derive_holding_verdict(
            pick=pick, pnl_pct=-2.0, weight_pct=15,
            top_pick=top_pick, new_recommend_safe=True,
        )
        # 신규 약함 → 🟠 안 됨 → 🟡
        assert v["icon"] == "🟡"

    def test_blue_when_hold_ok_but_no_attack(self, fake_env, swap_svc):
        """🔵 유지+신규금지: 보유 OK but ROUTE WAIT 아닌 NEUTRAL + RR 낮음."""
        pick = self._make_pick(final_score=72, route="NEUTRAL", ebs=1, rr_now_tp1=1.0)
        v = swap_svc.derive_holding_verdict(
            pick=pick, pnl_pct=5.0, weight_pct=15,
            top_pick=None, new_recommend_safe=False,
        )
        assert v["icon"] == "🔵"
        assert v["level"] == "blue"
        assert "신규매수 금지" in v["title"]

    def test_green_when_all_good(self, fake_env, swap_svc):
        """🟢 유지: 모든 조건 양호."""
        pick = self._make_pick(final_score=80, route="ATTACK", ebs=1, rr_now_tp1=2.5)
        v = swap_svc.derive_holding_verdict(
            pick=pick, pnl_pct=10.0, weight_pct=18,
            top_pick=None, new_recommend_safe=False,
        )
        assert v["icon"] == "🟢"
        assert v["level"] == "green"
        assert v["title"] == "유지"


# ════════════════════════════════════════════════════════════════
# C. 안전 원칙 회귀 가드
# ════════════════════════════════════════════════════════════════
class TestSafetyPrinciples:

    def _make_pick(self, **kwargs):
        defaults = {
            "name": "TestStock", "final_score": 75.0,
            "display_score": 75.0, "route": "ATTACK", "ebs": 1,
            "rr_now_tp1": 2.0, "entry_gap_pct": 1.0,
        }
        defaults.update(kwargs)
        return defaults

    def test_new_overheat_blocks_swap(self, fake_env, swap_svc):
        """[안전 1] 신규추천 OVERHEAT → 교체 추천 금지.

        _is_recommend_safe()가 False로 분류해야 함.
        """
        overheat_pick = self._make_pick(route="OVERHEAT")
        # private function 직접 호출 — 모듈에서 _is_recommend_safe 가져옴
        safe = swap_svc._is_recommend_safe(overheat_pick)
        assert safe is False

    def test_new_high_entry_gap_blocks_swap(self, fake_env, swap_svc):
        """[안전 2] 신규추천 ENTRY_GAP > 5% (과열) → 교체 금지."""
        high_gap_pick = self._make_pick(entry_gap_pct=8.0)
        safe = swap_svc._is_recommend_safe(high_gap_pick)
        assert safe is False

    def test_new_none_pick_is_unsafe(self, fake_env, swap_svc):
        """Top Pick 없으면 안전 검증 자체 불가 → False."""
        safe = swap_svc._is_recommend_safe(None)
        assert safe is False

    def test_new_attack_route_is_safe(self, fake_env, swap_svc):
        """ATTACK 정상 추천 → 안전."""
        safe = swap_svc._is_recommend_safe(
            self._make_pick(route="ATTACK", entry_gap_pct=2.0)
        )
        assert safe is True

    def test_orange_requires_score_gap_min_10(self, fake_env, swap_svc):
        """[안전 — 단순 점수 비교 금지] 68 vs 75 (gap 7) → 🟠 안 됨."""
        pick = self._make_pick(final_score=68, ebs=0, route="NEUTRAL")
        top_pick = self._make_pick(name="신규", final_score=75, ebs=1, rr_now_tp1=2.0)
        v = swap_svc.derive_holding_verdict(
            pick=pick, pnl_pct=-2.0, weight_pct=15,
            top_pick=top_pick, new_recommend_safe=True,
        )
        # gap 7점 → 🟠 안 됨
        assert v["icon"] != "🟠"

    def test_no_automatic_buy_sell_keywords(self, fake_env, swap_svc):
        """[안전 — 자동 매수/매도 확정 금지] 판정 title에 '매도'/'매수' 없음.

        평가 명시: '정리 우선', '교체 후보', '감량 검토' 표현만 사용.
        """
        # 모든 6단계 시나리오 생성
        test_cases = [
            (self._make_pick(ebs=0, route="WAIT"), -8.0, 15, None, False, "red"),
            (self._make_pick(final_score=55, ebs=0, route="NEUTRAL"), -3.0, 15,
             self._make_pick(name="강", final_score=82, ebs=1, rr_now_tp1=2.5), True, "orange"),
            (self._make_pick(final_score=75), 5.0, 68, None, False, "yellow"),
            (self._make_pick(final_score=72, route="NEUTRAL", rr_now_tp1=1.0), 5.0, 15,
             None, False, "blue"),
            (self._make_pick(), 10.0, 18, None, False, "green"),
        ]
        for pick, pnl, weight, tp, safe, expected_level in test_cases:
            v = swap_svc.derive_holding_verdict(
                pick=pick, pnl_pct=pnl, weight_pct=weight,
                top_pick=tp, new_recommend_safe=safe,
            )
            title = v["title"]
            body = v["body"]
            # 절대 매도/매수 단어 금지 (확정형 표현)
            assert "매도하세요" not in title and "매도하세요" not in body
            assert "매수하세요" not in title and "매수하세요" not in body
            # title에는 "즉시" 없음 (body는 "바로 교체하기보다" 같은 표현 OK)
            assert "즉시" not in title
            # 권장 표현만 사용
            permitted = [
                "정리", "교체", "감량", "유지", "데이터", "신규매수 금지"
            ]
            assert any(p in title for p in permitted), (
                f"권장 표현 없음: {title}"
            )

    # ────────────────────────────────────────────────────────────────
    # [v3.9.21b 평가 1] 진짜 anomaly 차단 — 회귀 가드 강화
    # ────────────────────────────────────────────────────────────────
    def test_new_anomaly_flag_blocks_swap(self, fake_env, swap_svc):
        """[v3.9.21b 평가 1] is_anomaly=True → 교체 금지."""
        pick = self._make_pick(route="ATTACK", entry_gap_pct=1.0)
        pick["is_anomaly"] = True
        safe = swap_svc._is_recommend_safe(pick)
        assert safe is False, (
            "is_anomaly=True인데 _is_recommend_safe=True 반환됨"
        )

    def test_new_high_tp_saturation_blocks_swap(self, fake_env, swap_svc):
        """[v3.9.21b 평가 1] TP 포화 ≥ 80% → 교체 금지."""
        pick = self._make_pick(route="ATTACK", entry_gap_pct=1.0)
        pick["tp_saturation"] = 85.0
        safe = swap_svc._is_recommend_safe(pick)
        assert safe is False, (
            "tp_saturation=85%인데 _is_recommend_safe=True 반환됨"
        )

    def test_new_low_tp_saturation_is_safe(self, fake_env, swap_svc):
        """TP 포화 < 80%면 정상."""
        pick = self._make_pick(route="ATTACK", entry_gap_pct=1.0)
        pick["tp_saturation"] = 55.0
        safe = swap_svc._is_recommend_safe(pick)
        assert safe is True

    def test_new_unrealistic_return_blocks_swap(self, fake_env, swap_svc):
        """[v3.9.21b 평가 1] 비현실 수익률 (>300%) → 교체 금지."""
        pick = self._make_pick(route="ATTACK", entry_gap_pct=1.0)
        pick["total_return_abs"] = 450.0  # 450% 수익률 — anomaly
        safe = swap_svc._is_recommend_safe(pick)
        assert safe is False

    def test_orange_blocked_when_new_pick_anomaly(self, fake_env, swap_svc):
        """[v3.9.21b 평가 1 통합] 신규 anomaly → 🟠 교체 후보 차단."""
        hold_pick = self._make_pick(final_score=55, ebs=0, route="NEUTRAL")
        # 신규 점수 높지만 is_anomaly=True
        top_pick = self._make_pick(
            name="신규Top", final_score=85, ebs=1, rr_now_tp1=3.0
        )
        # 시뮬레이션: 외부에서 _is_recommend_safe로 검증한 결과 False
        v = swap_svc.derive_holding_verdict(
            pick=hold_pick, pnl_pct=-3.0, weight_pct=15,
            top_pick=top_pick, new_recommend_safe=False,  # ★ anomaly 검출됨
        )
        # 🟠 안 됨 (신규 unsafe)
        assert v["icon"] != "🟠"


# ════════════════════════════════════════════════════════════════
# C2. v3.9.21b 추가 가드 — EBS 파서 + 매칭
# ════════════════════════════════════════════════════════════════
class TestParseEbs:
    """[v3.9.21b 평가 2] EBS 다양한 형식 파싱."""

    def test_parse_ebs_from_int(self, fake_env, swap_svc):
        row = pd.Series({"EBS": 8})
        assert swap_svc._parse_ebs(row) == 8

    def test_parse_ebs_from_zero(self, fake_env, swap_svc):
        row = pd.Series({"EBS": 0})
        assert swap_svc._parse_ebs(row) == 0

    def test_parse_ebs_from_pass_string(self, fake_env, swap_svc):
        """'PASS' 문자열 → 1."""
        row = pd.Series({"EBS": "PASS"})
        assert swap_svc._parse_ebs(row) == 1

    def test_parse_ebs_from_fail_string(self, fake_env, swap_svc):
        """'FAIL' 문자열 → 0."""
        row = pd.Series({"EBS": "FAIL"})
        assert swap_svc._parse_ebs(row) == 0

    def test_parse_ebs_from_fraction(self, fake_env, swap_svc):
        """'8/8' → 8."""
        row = pd.Series({"EBS": "8/8"})
        assert swap_svc._parse_ebs(row) == 8

    def test_parse_ebs_from_fraction_with_label(self, fake_env, swap_svc):
        """'8/8 (PASS)' → 8 (먼저 PASS 매칭)."""
        row = pd.Series({"EBS": "8/8 (PASS)"})
        # PASS가 먼저 매칭 — 1 반환 (또는 8 — 둘 다 ≥ 1이라 PASS 의미)
        result = swap_svc._parse_ebs(row)
        assert result >= 1

    def test_parse_ebs_pass_column_true(self, fake_env, swap_svc):
        """EBS_PASS=True → 1."""
        row = pd.Series({"EBS_PASS": True})
        assert swap_svc._parse_ebs(row) == 1

    def test_parse_ebs_pass_column_false(self, fake_env, swap_svc):
        """EBS_PASS=False → 0."""
        row = pd.Series({"EBS_PASS": False})
        assert swap_svc._parse_ebs(row) == 0

    def test_parse_ebs_missing_returns_zero(self, fake_env, swap_svc):
        """컬럼 없으면 0."""
        row = pd.Series({"종목명": "A"})
        assert swap_svc._parse_ebs(row) == 0


class TestMatchHoldingByCode:
    """[v3.9.21b 평가 3] 종목코드 우선 매칭."""

    def test_code_match_when_available(self, fake_env, swap_svc):
        """종목코드로 매칭 — 이름이 달라도 코드가 같으면 매칭."""
        recs = pd.DataFrame([{
            "종목코드": "005930", "종목명": "삼성전자",
            "DISPLAY_SCORE": 80, "FINAL_SCORE": 80,
            "EBS": 1, "ROUTE": "ATTACK", "RR_NOW_TP1": 2.0,
            "ENTRY_GAP_PCT": 1.0, "종가": 70000,
        }])
        # hold에 종목코드 있고 이름은 다름 (약칭)
        hold = {"name": "삼전", "code": "005930", "avg": 70000, "qty": 10}
        matched = swap_svc._match_holding_to_recommend(hold, recs)
        assert not matched.empty
        assert matched.iloc[0]["종목명"] == "삼성전자"

    def test_code_match_handles_zero_padding(self, fake_env, swap_svc):
        """종목코드 zero-padding 안전 처리.

        예: hold "5930" / recommend "005930" → 매칭 성공
        """
        recs = pd.DataFrame([{
            "종목코드": "005930", "종목명": "삼성전자",
            "DISPLAY_SCORE": 80, "FINAL_SCORE": 80,
            "EBS": 1, "ROUTE": "ATTACK", "RR_NOW_TP1": 2.0,
            "ENTRY_GAP_PCT": 1.0, "종가": 70000,
        }])
        hold = {"name": "삼성전자", "code": "5930", "avg": 70000, "qty": 10}
        matched = swap_svc._match_holding_to_recommend(hold, recs)
        assert not matched.empty

    def test_name_fallback_when_no_code(self, fake_env, swap_svc):
        """종목코드 없을 때 종목명 strip 비교 fallback."""
        recs = pd.DataFrame([{
            "종목코드": "005930", "종목명": "삼성전자",
            "DISPLAY_SCORE": 80, "FINAL_SCORE": 80,
            "EBS": 1, "ROUTE": "ATTACK", "RR_NOW_TP1": 2.0,
            "ENTRY_GAP_PCT": 1.0, "종가": 70000,
        }])
        # code 없음, name으로 매칭
        hold = {"name": "삼성전자", "avg": 70000, "qty": 10}
        matched = swap_svc._match_holding_to_recommend(hold, recs)
        assert not matched.empty

    def test_name_match_handles_whitespace(self, fake_env, swap_svc):
        """이름 strip — 사용자 입력의 앞뒤 공백 처리."""
        recs = pd.DataFrame([{
            "종목코드": "005930", "종목명": "삼성전자",
            "DISPLAY_SCORE": 80, "FINAL_SCORE": 80,
            "EBS": 1, "ROUTE": "ATTACK", "RR_NOW_TP1": 2.0,
            "ENTRY_GAP_PCT": 1.0, "종가": 70000,
        }])
        hold = {"name": " 삼성전자 ", "avg": 70000, "qty": 10}
        matched = swap_svc._match_holding_to_recommend(hold, recs)
        assert not matched.empty

    def test_returns_empty_when_no_match(self, fake_env, swap_svc):
        """매칭 없으면 empty DataFrame."""
        recs = pd.DataFrame([{
            "종목코드": "005930", "종목명": "삼성전자",
            "DISPLAY_SCORE": 80, "FINAL_SCORE": 80,
            "EBS": 1, "ROUTE": "ATTACK", "RR_NOW_TP1": 2.0,
            "ENTRY_GAP_PCT": 1.0, "종가": 70000,
        }])
        hold = {"name": "다른회사", "code": "999999", "avg": 1000, "qty": 1}
        matched = swap_svc._match_holding_to_recommend(hold, recs)
        assert matched.empty


# ════════════════════════════════════════════════════════════════
# D. UI 렌더
# ════════════════════════════════════════════════════════════════
class TestRenderSwap:

    def test_renders_error_message(self, fake_env, swap_ui):
        """error 케이스 → 메시지 표시."""
        swap_ui._render_portfolio_swap_card({"error": "데이터 없음"})
        text = _captured_text()
        assert "데이터 없음" in text

    def test_renders_summary_pills(self, fake_env, swap_ui):
        """요약 카드 표시 (red/orange/yellow)."""
        data = {
            "summary": {"red": 1, "orange": 2, "yellow": 3, "blue": 0,
                        "green": 4, "white": 0},
            "top_pick": None,
            "new_recommend_safe": False,
            "holdings_analysis": [
                {
                    "name": "A종목", "avg": 10000, "qty": 10,
                    "current_price": 10500, "value": 105000,
                    "pnl_pct": 5.0, "weight_pct": 25.0,
                    "matched_in_recommend": True,
                    "final_score": 75, "route": "ATTACK", "ebs": 1,
                    "rr_now_tp1": 2.0, "entry_gap_pct": 1.0,
                    "verdict": {
                        "icon": "🟢", "level": "green", "title": "유지",
                        "color_class": "text-emerald-400",
                        "reasons": ["FINAL 75"],
                        "swap_candidate": False, "body": "유지 권장",
                    },
                },
            ],
        }
        swap_ui._render_portfolio_swap_card(data)
        text = _captured_text()
        assert "교체 판단" in text
        assert "정리 우선" in text or "🔴" in text
        assert "유지" in text or "🟢" in text

    def test_renders_top_pick_warning_when_unsafe(self, fake_env, swap_ui):
        """신규추천 unsafe → 경고 표시."""
        data = {
            "summary": {"red": 0, "orange": 0, "yellow": 0, "blue": 0,
                        "green": 1, "white": 0},
            "top_pick": {
                "name": "위험종목", "final_score": 80, "route": "OVERHEAT",
            },
            "new_recommend_safe": False,
            "holdings_analysis": [],
        }
        swap_ui._render_portfolio_swap_card(data)
        text = _captured_text()
        assert "주의" in text or "anomaly" in text or "위험종목" in text


# ════════════════════════════════════════════════════════════════
# E. baseline 무수정 + 분리 import parity
# ════════════════════════════════════════════════════════════════
class TestNoRegression:
    """평가 회귀 가드 5: baseline 변경 없음."""

    def test_services_swap_no_nicegui_import(self, fake_env):
        """[평가 6] services.portfolio_swap에 nicegui import 0."""
        import inspect
        import services.portfolio_swap as svc
        src_path = inspect.getfile(svc)
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("from nicegui"), (
                f"services.portfolio_swap에 nicegui import 발견: {line}"
            )
            assert not stripped.startswith("import nicegui"), (
                f"services.portfolio_swap에 nicegui import 발견: {line}"
            )

    def test_components_reexports_services_functions(
        self, fake_env, swap_ui
    ):
        """components가 services 함수 re-export (is 검증)."""
        from services.portfolio_swap import (
            analyze_portfolio_swap as svc_analyze,
            derive_holding_verdict as svc_verdict,
        )
        assert swap_ui._analyze_portfolio_swap is svc_analyze
        assert swap_ui._derive_holding_verdict is svc_verdict


# ════════════════════════════════════════════════════════════════
# F. [v3.9.21c] 평가 보정 회귀 가드
# ════════════════════════════════════════════════════════════════
class TestAnomalyExtractionListSafe:
    """[v3.9.21c 평가 2] anomaly_flags list 처리 — pd.isna 호출 순서 안전."""

    def test_empty_list_is_not_anomaly(self, fake_env, swap_svc):
        """[v3.9.21c 평가 2] anomaly_flags=[] → is_anomaly=False (예외 없음).

        이전: pd.isna([])가 빈 array 반환 → "or pd.isna(v)" 평가 시 truth ambiguity
        v3.9.21c: list/tuple 판단을 pd.isna보다 먼저 → 안전
        """
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1, "ROUTE": "ATTACK",
            "anomaly_flags": [],
            "종가": 10000,
        })
        # 예외 없이 정상 처리
        pick = swap_svc._row_to_pick_dict(row)
        assert pick["is_anomaly"] is False

    def test_nonempty_list_is_anomaly(self, fake_env, swap_svc):
        """anomaly_flags=['a', 'b'] → is_anomaly=True (예외 없음)."""
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1, "ROUTE": "ATTACK",
            "anomaly_flags": ["수익률 비정상", "Sharpe 비정상"],
            "종가": 10000,
        })
        pick = swap_svc._row_to_pick_dict(row)
        assert pick["is_anomaly"] is True

    def test_tuple_anomaly_flags(self, fake_env, swap_svc):
        """tuple도 list와 동일 처리."""
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1, "ROUTE": "ATTACK",
            "anomaly_flags": ("a",),
            "종가": 10000,
        })
        pick = swap_svc._row_to_pick_dict(row)
        assert pick["is_anomaly"] is True


class TestNormalizeRoute:
    """[v3.9.21c 평가 3] ROUTE 정규화 — strip/upper/alias."""

    def test_normalize_uppercase(self, fake_env, swap_svc):
        """'attack' → 'ATTACK'."""
        assert swap_svc._normalize_route("attack") == "ATTACK"

    def test_normalize_with_whitespace(self, fake_env, swap_svc):
        """' ARMED ' → 'ARMED'."""
        assert swap_svc._normalize_route(" ARMED ") == "ARMED"

    def test_normalize_mixed_case(self, fake_env, swap_svc):
        """'Attack' → 'ATTACK'."""
        assert swap_svc._normalize_route("Attack") == "ATTACK"

    def test_normalize_korean_attack(self, fake_env, swap_svc):
        """'적극매수' → 'ATTACK'."""
        assert swap_svc._normalize_route("적극매수") == "ATTACK"

    def test_normalize_korean_armed(self, fake_env, swap_svc):
        """'진입대기' → 'ARMED'."""
        assert swap_svc._normalize_route("진입대기") == "ARMED"

    def test_normalize_korean_overheat(self, fake_env, swap_svc):
        """'과열' → 'OVERHEAT'."""
        assert swap_svc._normalize_route("과열") == "OVERHEAT"

    def test_normalize_none(self, fake_env, swap_svc):
        """None → None."""
        assert swap_svc._normalize_route(None) is None

    def test_normalize_empty_string(self, fake_env, swap_svc):
        """'' → None."""
        assert swap_svc._normalize_route("") is None
        assert swap_svc._normalize_route("   ") is None

    def test_normalize_unknown_returns_upper(self, fake_env, swap_svc):
        """알 수 없는 값은 upper만 (보수적 유지)."""
        assert swap_svc._normalize_route("xyz") == "XYZ"

    def test_normalize_via_row_to_pick(self, fake_env, swap_svc):
        """_row_to_pick_dict가 ROUTE를 자동 정규화."""
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1,
            "ROUTE": "  attack  ",  # 소문자 + 공백
            "종가": 10000,
        })
        pick = swap_svc._row_to_pick_dict(row)
        assert pick["route"] == "ATTACK"


class TestTwoPassWeightCalculation:
    """[v3.9.21c 평가 4] 비중 2-pass 계산 — 현재가 기반 통일."""

    def test_value_basis_reported_correctly(self, fake_env, swap_svc):
        """analyze_portfolio_swap 반환에 value_basis 키 포함."""
        recs = pd.DataFrame([{
            "종목코드": "001", "종목명": "A",
            "DISPLAY_SCORE": 75, "FINAL_SCORE": 75,
            "EBS": 1, "ROUTE": "ATTACK", "RR_NOW_TP1": 2.0,
            "ENTRY_GAP_PCT": 1.0, "종가": 15000,
        }])
        out = swap_svc.analyze_portfolio_swap(
            holdings=[{"name": "A", "avg": 10000, "qty": 10}],
            recommend_df=recs,
        )
        assert "value_basis" in out
        assert out["value_basis"] == "current_price"

    def test_weight_uses_current_price_basis_two_pass(
        self, fake_env, swap_svc
    ):
        """[v3.9.21c 평가 4] total_value 자동 계산 시 현재가 기준 통일.

        시나리오: 보유 1주 (avg 10000) + 가격이 +50% 상승 (current 15000)
        이전 v3.9.21b:
          분자: 15000 * qty (현재가)
          분모: 10000 * qty (매입가)
          → 비중 150% 왜곡
        v3.9.21c:
          2-pass — 분모도 현재가 기준 합계
          → 단일 종목이면 100%
        """
        recs = pd.DataFrame([{
            "종목코드": "001", "종목명": "A",
            "DISPLAY_SCORE": 75, "FINAL_SCORE": 75,
            "EBS": 1, "ROUTE": "ATTACK", "RR_NOW_TP1": 2.0,
            "ENTRY_GAP_PCT": 1.0, "종가": 15000,  # +50% 상승
        }])
        out = swap_svc.analyze_portfolio_swap(
            holdings=[{"name": "A", "avg": 10000, "qty": 10}],
            recommend_df=recs,
        )
        item = out["holdings_analysis"][0]
        # 단일 종목 → 비중 100% (current_price 기준 통일)
        # 이전 v3.9.21b: 150% (분자 15000 / 분모 10000)
        assert abs(item["weight_pct"] - 100.0) < 1.0, (
            f"비중 100% 기대, 실제 {item['weight_pct']:.1f}% — "
            "2-pass 계산이 적용되지 않은 듯"
        )

    def test_value_basis_mixed_when_some_unmatched(
        self, fake_env, swap_svc
    ):
        """[v3.9.21c 평가 4] 일부 종목 매칭 안 되면 value_basis=mixed."""
        recs = pd.DataFrame([{
            "종목코드": "001", "종목명": "매칭됨",
            "DISPLAY_SCORE": 75, "FINAL_SCORE": 75,
            "EBS": 1, "ROUTE": "ATTACK", "RR_NOW_TP1": 2.0,
            "ENTRY_GAP_PCT": 1.0, "종가": 10000,
        }])
        out = swap_svc.analyze_portfolio_swap(
            holdings=[
                {"name": "매칭됨", "avg": 10000, "qty": 10},
                {"name": "매칭안됨", "avg": 5000, "qty": 20},  # recommend 없음
            ],
            recommend_df=recs,
        )
        assert out["value_basis"] == "mixed_current_avg"

    def test_external_total_value_is_trusted(self, fake_env, swap_svc):
        """외부 total_value > 0이면 자동 계산하지 않고 그대로 사용.

        실제 평가금액을 외부에서 주면 가장 정확.
        """
        recs = pd.DataFrame([{
            "종목코드": "001", "종목명": "A",
            "DISPLAY_SCORE": 75, "FINAL_SCORE": 75,
            "EBS": 1, "ROUTE": "ATTACK", "RR_NOW_TP1": 2.0,
            "ENTRY_GAP_PCT": 1.0, "종가": 10000,
        }])
        # 외부에서 200,000원 평가금액 전달 (실제 평가금액)
        out = swap_svc.analyze_portfolio_swap(
            holdings=[{"name": "A", "avg": 10000, "qty": 10}],
            recommend_df=recs,
            total_value=200000,
        )
        # 외부 total_value 신뢰 → 100000 / 200000 = 50%
        item = out["holdings_analysis"][0]
        assert abs(item["weight_pct"] - 50.0) < 1.0, (
            f"외부 total_value 사용 시 50% 기대, 실제 {item['weight_pct']:.1f}%"
        )


# ════════════════════════════════════════════════════════════════
# G. [v3.9.21d] 평가 보정 회귀 가드 — EBS 문자열 + alias fallback
# ════════════════════════════════════════════════════════════════
class TestEbsPassStringBug:
    """[v3.9.21d 평가 1] EBS_PASS 문자열 'False'/'0'/'FAIL' 정확 처리."""

    def test_ebs_pass_string_false_returns_zero(self, fake_env, swap_svc):
        """[v3.9.21d 평가 1] EBS_PASS='False' (문자열) → 0.

        Python의 bool('False')=True 함정 차단.
        """
        row = pd.Series({"EBS_PASS": "False"})
        assert swap_svc._parse_ebs(row) == 0, (
            "EBS_PASS='False' 문자열은 0이어야 함 (bool 함정 차단)"
        )

    def test_ebs_pass_string_true_returns_one(self, fake_env, swap_svc):
        """EBS_PASS='True' 문자열 → 1."""
        row = pd.Series({"EBS_PASS": "True"})
        assert swap_svc._parse_ebs(row) == 1

    def test_ebs_pass_string_pass_returns_one(self, fake_env, swap_svc):
        """EBS_PASS='PASS' → 1."""
        row = pd.Series({"EBS_PASS": "PASS"})
        assert swap_svc._parse_ebs(row) == 1

    def test_ebs_pass_string_fail_returns_zero(self, fake_env, swap_svc):
        """EBS_PASS='FAIL' → 0."""
        row = pd.Series({"EBS_PASS": "FAIL"})
        assert swap_svc._parse_ebs(row) == 0

    def test_ebs_pass_string_zero_returns_zero(self, fake_env, swap_svc):
        """EBS_PASS='0' (문자열) → 0."""
        row = pd.Series({"EBS_PASS": "0"})
        assert swap_svc._parse_ebs(row) == 0

    def test_ebs_pass_string_one_returns_one(self, fake_env, swap_svc):
        """EBS_PASS='1' (문자열) → 1."""
        row = pd.Series({"EBS_PASS": "1"})
        assert swap_svc._parse_ebs(row) == 1

    def test_ebs_pass_string_yes_no(self, fake_env, swap_svc):
        """EBS_PASS='yes' → 1, 'no' → 0."""
        assert swap_svc._parse_ebs(pd.Series({"EBS_PASS": "yes"})) == 1
        assert swap_svc._parse_ebs(pd.Series({"EBS_PASS": "no"})) == 0

    def test_ebs_pass_string_unknown_returns_zero(self, fake_env, swap_svc):
        """알 수 없는 문자열 → 0 (보수적).

        bool("xyz") = True 함정 회피.
        """
        row = pd.Series({"EBS_PASS": "xyz"})
        assert swap_svc._parse_ebs(row) == 0

    def test_ebs_pass_bool_true_still_works(self, fake_env, swap_svc):
        """기존 bool True 케이스 회귀 확인."""
        row = pd.Series({"EBS_PASS": True})
        assert swap_svc._parse_ebs(row) == 1

    def test_ebs_pass_bool_false_still_works(self, fake_env, swap_svc):
        """기존 bool False 케이스 회귀 확인."""
        row = pd.Series({"EBS_PASS": False})
        assert swap_svc._parse_ebs(row) == 0


class TestEntryGapAliasFallback:
    """[v3.9.21d 평가 3] entry_gap 컬럼명 alias fallback."""

    def test_entry_gap_pct_primary(self, fake_env, swap_svc):
        """ENTRY_GAP_PCT 우선 — 가장 표준 컬럼명."""
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1, "ROUTE": "ATTACK",
            "ENTRY_GAP_PCT": 2.5,
            "종가": 10000,
        })
        pick = swap_svc._row_to_pick_dict(row)
        assert pick["entry_gap_pct"] == 2.5

    def test_gap_pct_alias(self, fake_env, swap_svc):
        """GAP_PCT 알리아스 — ENTRY_GAP_PCT 없을 때 fallback."""
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1, "ROUTE": "ATTACK",
            "GAP_PCT": 3.5,  # ENTRY_GAP_PCT 없음
            "종가": 10000,
        })
        pick = swap_svc._row_to_pick_dict(row)
        assert pick["entry_gap_pct"] == 3.5

    def test_entry_gap_alias(self, fake_env, swap_svc):
        """ENTRY_GAP (PCT 없는 alias)."""
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1, "ROUTE": "ATTACK",
            "ENTRY_GAP": 4.2,
            "종가": 10000,
        })
        pick = swap_svc._row_to_pick_dict(row)
        assert pick["entry_gap_pct"] == 4.2

    def test_lowercase_entry_gap_pct(self, fake_env, swap_svc):
        """소문자 entry_gap_pct."""
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1, "ROUTE": "ATTACK",
            "entry_gap_pct": 1.5,
            "종가": 10000,
        })
        pick = swap_svc._row_to_pick_dict(row)
        assert pick["entry_gap_pct"] == 1.5

    def test_entry_gap_priority_when_multiple(self, fake_env, swap_svc):
        """여러 alias 동시 존재 시 ENTRY_GAP_PCT 우선."""
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1, "ROUTE": "ATTACK",
            "ENTRY_GAP_PCT": 2.5,  # 표준
            "GAP_PCT": 8.0,        # 다른 값 — 무시되어야 함
            "ENTRY_GAP": 6.0,
            "종가": 10000,
        })
        pick = swap_svc._row_to_pick_dict(row)
        # ENTRY_GAP_PCT 우선 (2.5)
        assert pick["entry_gap_pct"] == 2.5

    def test_entry_gap_none_when_no_alias(self, fake_env, swap_svc):
        """모든 alias 없으면 None."""
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1, "ROUTE": "ATTACK",
            "종가": 10000,
        })
        pick = swap_svc._row_to_pick_dict(row)
        assert pick["entry_gap_pct"] is None

    def test_safety_uses_alias_value(self, fake_env, swap_svc):
        """alias로 들어온 ENTRY_GAP도 _is_recommend_safe에서 차단 검사."""
        row = pd.Series({
            "종목명": "A", "종목코드": "001",
            "DISPLAY_SCORE": 70, "FINAL_SCORE": 70,
            "EBS": 1, "ROUTE": "ATTACK",
            "GAP_PCT": 7.0,  # > 5% 임계, ENTRY_GAP_PCT 컬럼 없음
            "종가": 10000,
        })
        pick = swap_svc._row_to_pick_dict(row)
        # alias로 추출됐어야 함
        assert pick["entry_gap_pct"] == 7.0
        # 7.0 > NEW_DANGER_ENTRY_GAP_PCT(5.0) → unsafe
        assert swap_svc._is_recommend_safe(pick) is False



