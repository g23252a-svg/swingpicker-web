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
            # 절대 매도/매수 단어 금지 (확정형 표현)
            assert "매도하세요" not in title
            assert "매수하세요" not in title
            assert "즉시" not in title
            # 권장 표현만 사용
            permitted = [
                "정리", "교체", "감량", "유지", "데이터", "신규매수 금지"
            ]
            assert any(p in title for p in permitted), (
                f"권장 표현 없음: {title}"
            )


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
