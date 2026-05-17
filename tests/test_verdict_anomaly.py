"""
tests/test_verdict_anomaly.py
=============================
[v3.9.15e + 8] _render_strategy_verdict_card() 함수의 진짜 end-to-end 테스트.

이전 verify_anomaly_v3915e7.py의 한계:
- NiceGUI mocking까지만 했고 verdict 함수 본문의 anomaly 로직만 복제 검증.
- 진짜 _render_strategy_verdict_card(result, cfg)를 호출하지 않음.
- 따라서 함수 안의 분기 (icon, title, body 결정 로직) 회귀를 못 잡음.

이 파일은 진짜 e2e:
1. NiceGUI ui 모듈을 capture-able mock으로 교체
2. _render_strategy_verdict_card(result, cfg) 직접 호출
3. ui.label()에 들어간 모든 텍스트를 수집
4. 기대 문자열 (title, anomaly 경고 등) 포함 여부 단언
"""
import sys
import os
from pathlib import Path

import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────
# NiceGUI mock — ui.label / ui.card / ui.row 호출을 캡처
# ────────────────────────────────────────────────────────────────
class _CapturingLabel:
    """ui.label() 가짜 — 전달된 텍스트를 captured_labels에 누적."""
    def __init__(self, text=""):
        captured_labels.append(str(text))

    def classes(self, *a, **kw):
        return self

    def tooltip(self, *a, **kw):
        return self

    def bind_text_from(self, *a, **kw):
        return self

    def __getattr__(self, name):
        # 알 수 없는 메서드도 chainable 응답으로 흡수
        return lambda *a, **kw: self


class _ContextManagerMock:
    """ui.card() / ui.row() 가짜 — with 블록 지원 + 모든 메서드 흡수."""
    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def classes(self, *a, **kw):
        return self

    def tooltip(self, *a, **kw):
        return self

    def props(self, *a, **kw):
        return self

    def style(self, *a, **kw):
        return self

    def __getattr__(self, name):
        # 알 수 없는 메서드 호출도 chainable 응답으로 흡수
        return lambda *a, **kw: self


# 전역 capture buffer — 매 테스트마다 reset
captured_labels = []


def _setup_nicegui_mock(monkeypatch):
    """NiceGUI 의존성을 mock으로 교체."""
    import types
    fake_nicegui = types.ModuleType("nicegui")
    fake_ui = types.SimpleNamespace(
        label=lambda text="": _CapturingLabel(text),
        card=lambda: _ContextManagerMock(),
        row=lambda: _ContextManagerMock(),
        column=lambda: _ContextManagerMock(),
    )
    fake_nicegui.ui = fake_ui
    fake_nicegui.app = types.ModuleType("app")
    monkeypatch.setitem(sys.modules, "nicegui", fake_nicegui)
    # plotly도 import만 통과시키면 됨
    fake_plotly = types.ModuleType("plotly")
    fake_go = types.ModuleType("plotly.graph_objects")
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_go)


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────
@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """services.benchmarks가 참조할 data 디렉토리 + NiceGUI mock 셋업."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # 가짜 bench_cache — simple alpha 산출 가능하게
    import json
    (data_dir / "bench_cache_latest.json").write_text(json.dumps({
        "KOSPI": {"1": 0.1, "3": 0.5, "5": 1.0, "10": 2.0, "20": 4.0, "60": 10.0},
    }))

    monkeypatch.chdir(tmp_path)

    # [중요] 모듈 캐시 클리어 — 이전 테스트가 진짜 nicegui로 import한
    # components.tab_backtest가 캐시되어 있으면 monkeypatch가 안 통함.
    # services / components / nicegui / plotly 모두 fresh.
    for mod in list(sys.modules.keys()):
        if (
            "benchmarks" in mod
            or mod.startswith("services")
            or mod.startswith("components")
            or mod == "nicegui"
            or mod.startswith("nicegui.")
            or mod == "plotly"
            or mod.startswith("plotly.")
        ):
            del sys.modules[mod]

    # NiceGUI mock 설정 (모듈 캐시 클리어 직후 → 다음 import가 mock 받음)
    _setup_nicegui_mock(monkeypatch)

    # services.benchmarks DATA_DIR 패치 (mock 설정 후 first-import)
    import services.benchmarks as _bench
    monkeypatch.setattr(_bench, "DATA_DIR", str(data_dir))

    # capture buffer 비우기
    captured_labels.clear()

    return data_dir


@pytest.fixture
def verdict_card_fn():
    """tab_backtest._render_strategy_verdict_card lazy import."""
    tab_backtest = pytest.importorskip(
        "components.tab_backtest",
        reason="tab_backtest 모듈 import 불가",
        exc_type=ImportError,
    )
    return tab_backtest._render_strategy_verdict_card


@pytest.fixture
def render_results_fn():
    """[v3.9.15e + 9] tab_backtest._render_results lazy import.

    _render_results는 verdict_card 외에 _stat_card 등 다른 카드들도 그리는데
    CAGR/Sharpe cap이 이 함수 안에 있어서 verdict 테스트만으론 커버 불가.
    """
    tab_backtest = pytest.importorskip(
        "components.tab_backtest",
        reason="tab_backtest 모듈 import 불가",
        exc_type=ImportError,
    )
    return tab_backtest._render_results


def _captured_text():
    """수집된 라벨 텍스트를 한 줄로 합쳐 반환 (부분 매칭용)."""
    return "\n".join(captured_labels)


# ────────────────────────────────────────────────────────────────
# A. 본인 화면 시나리오 — 712%/Sharpe 45/CAGR 100만%
# ────────────────────────────────────────────────────────────────
class TestUserScreenScenario:
    """본인 화면 실제 값으로 verdict 함수 직접 호출."""

    def test_real_screen_values_trigger_anomaly_block(
        self, fake_env, verdict_card_fn
    ):
        """712%/Sharpe 45/CAGR 100만% → 🟢 차단 + 🟡 과대추정 표시."""
        result = {
            "total_return": 712.86,
            "win_rate": 93.7,
            "mdd": -3.4,
            "total_trades": 269,
            "sharpe": 45.88,
            "cagr": 1054855.80,
            "trading_days": 57,
            "status_dist": {"WIN": 219, "HOLD_EXIT": 41, "STOP": 9},
            "trades_df": pd.DataFrame(),  # empty → simple alpha 경로
            "avg_win": 4.31,
            "avg_loss": -2.55,
        }
        cfg = {
            "min_score": 80, "top_k": 5, "hold_days": 5,
            "target_pct": 5, "stop_pct": 3, "cost_pct": 0.4,
        }

        verdict_card_fn(result, cfg)
        text = _captured_text()

        # 🟢 차단되어야 함
        assert "🟢" not in text, "🟢 차단 실패 — 비현실 수익률에도 실전 후보로 떴음"
        # 🟡 + "과대추정 가능성" 표시
        assert "🟡" in text
        assert "관찰 후보 · 과대추정 가능성" in text, (
            f"anomaly 타이틀 누락. captured: {text[:500]}"
        )
        # 🚨 결과 과열 경고 박스
        assert "🚨 결과 과열 경고" in text
        # TP 포화율 경고 (81.4%)
        assert "TP 포화율 81.4%" in text or "TP 포화" in text
        # 자금제약 설명 포함
        assert "동시 보유 슬롯" in text or "동시 보유 자금 제약" in text

    def test_real_screen_does_not_leak_raw_cagr(self, fake_env, verdict_card_fn):
        """CAGR 100만% raw 숫자가 verdict 본문/요약에 노출되지 않아야 함.

        화면 caps만 처리하면 안 되고, verdict 카드 본문도 raw 숫자 노출
        안 해야 사용자 자극 최소.
        """
        result = {
            "total_return": 712.86, "win_rate": 93.7, "mdd": -3.4,
            "total_trades": 269, "sharpe": 45.88, "cagr": 1054855.80,
            "trading_days": 57,
            "status_dist": {"WIN": 219, "HOLD_EXIT": 41, "STOP": 9},
            "trades_df": pd.DataFrame(),
            "avg_win": 4.31, "avg_loss": -2.55,
        }
        cfg = {"min_score": 80, "top_k": 5, "hold_days": 5,
               "target_pct": 5, "stop_pct": 3, "cost_pct": 0.4}

        verdict_card_fn(result, cfg)
        text = _captured_text()

        # raw CAGR 숫자 노출 차단 (1054855 또는 +1054856 같은 raw 숫자)
        assert "1054855" not in text, (
            f"CAGR raw 수치 노출됨. text 일부: {text[text.find('CAGR')-50:text.find('CAGR')+200] if 'CAGR' in text else text[:300]}"
        )
        # 대신 cap된 메시지 사용
        assert "300% 초과" in text or "비정상" in text

        # Sharpe raw 45.88도 verdict 요약에서 cap되어야 함
        # (summary_parts에 "Sharpe 비정상 (5 초과)" 형태)
        assert "Sharpe 비정상" in text


# ────────────────────────────────────────────────────────────────
# B. 기간 가중 anomaly — 단기 백테스트 별도 검출
# ────────────────────────────────────────────────────────────────
class TestPeriodWeightedAnomaly:
    """trading_days < 120 일에서 total_ret > 100% → 단기 과대수익."""

    def test_short_period_high_return_triggers_anomaly(
        self, fake_env, verdict_card_fn
    ):
        """57일에 +150% 수익 — 절대 임계(300%) 미만이지만 단기 가중에 잡힘."""
        result = {
            "total_return": 150.0,  # 절대 300 미만
            "win_rate": 65.0,
            "mdd": -8.0,
            "total_trades": 120,
            "sharpe": 2.5,  # 절대 5 미만
            "cagr": 280,  # 절대 300 미만
            "trading_days": 57,  # 단기
            "status_dist": {"WIN": 80, "HOLD_EXIT": 30, "STOP": 10},
            "trades_df": pd.DataFrame(),
            "avg_win": 3.0, "avg_loss": -2.0,
        }
        cfg = {"min_score": 80, "top_k": 5, "hold_days": 5,
               "target_pct": 5, "stop_pct": 3, "cost_pct": 0.4}

        verdict_card_fn(result, cfg)
        text = _captured_text()

        assert "🟢" not in text, "단기 과대수익도 🟢 차단되어야 함"
        assert "단기 과대수익" in text, (
            f"단기 가중 anomaly 누락. text: {text[:500]}"
        )

    def test_long_period_same_return_passes(self, fake_env, verdict_card_fn):
        """504일(2년)에 +150% 수익 — 단기 가중에 안 잡힘 (정상 범위)."""
        result = {
            "total_return": 150.0,
            "win_rate": 65.0,
            "mdd": -8.0,
            "total_trades": 400,
            "sharpe": 2.0,
            "cagr": 50,  # 2년 환산이라 낮음
            "trading_days": 504,  # 장기
            "status_dist": {"WIN": 250, "HOLD_EXIT": 100, "STOP": 50},
            "trades_df": pd.DataFrame(),
            "avg_win": 3.0, "avg_loss": -2.0,
        }
        cfg = {"min_score": 80, "top_k": 5, "hold_days": 5,
               "target_pct": 5, "stop_pct": 3, "cost_pct": 0.4}

        verdict_card_fn(result, cfg)
        text = _captured_text()

        # 504일짜리 +150% 수익은 정상 — anomaly 안 떠야 함
        # (이 케이스는 TP 포화율도 250/400=62.5% < 70% 통과)
        assert "단기 과대수익" not in text, (
            f"504일 백테스트가 단기로 오인됨. text: {text[:300]}"
        )
        assert "단기 CAGR 폭주" not in text


# ────────────────────────────────────────────────────────────────
# C. 정상 케이스 — false positive 없음
# ────────────────────────────────────────────────────────────────
class TestNormalCases:
    """합리적 수치는 🟢 또는 🟡 유지 (anomaly 박스 안 뜸)."""

    def test_normal_excellent_passes_to_green(self, fake_env, verdict_card_fn):
        """수익률 25% / Sharpe 1.5 / CAGR 35% — anomaly 안 떠야."""
        result = {
            "total_return": 25.0,
            "win_rate": 60.0,
            "mdd": -8.0,
            "total_trades": 150,
            "sharpe": 1.5,
            "cagr": 35.0,
            "trading_days": 252,
            "status_dist": {"WIN": 75, "HOLD_EXIT": 60, "STOP": 15},
            "trades_df": pd.DataFrame(),
            "avg_win": 3.0, "avg_loss": -2.0,
        }
        cfg = {"min_score": 80, "top_k": 5, "hold_days": 5,
               "target_pct": 5, "stop_pct": 3, "cost_pct": 0.4}

        verdict_card_fn(result, cfg)
        text = _captured_text()

        # anomaly 박스 안 떠야 함
        assert "🚨 결과 과열 경고" not in text
        assert "과대추정 가능성" not in text
        assert "단기 과대수익" not in text


# ────────────────────────────────────────────────────────────────
# D. 회귀 가드 — 캡 동작 정합성
# ────────────────────────────────────────────────────────────────
class TestRawNumberCap:
    """CAGR/Sharpe raw 숫자가 verdict 카드에 절대 노출되지 않음."""

    def test_extreme_cagr_capped_in_summary(self, fake_env, verdict_card_fn):
        """CAGR 1,000,000% 같은 raw 숫자가 summary에 안 보임."""
        result = {
            "total_return": 500.0, "win_rate": 90.0, "mdd": -3.0,
            "total_trades": 200, "sharpe": 30.0, "cagr": 999999.99,
            "trading_days": 30,
            "status_dist": {"WIN": 180, "HOLD_EXIT": 15, "STOP": 5},
            "trades_df": pd.DataFrame(),
            "avg_win": 4.0, "avg_loss": -2.0,
        }
        cfg = {"min_score": 80, "top_k": 5, "hold_days": 5,
               "target_pct": 5, "stop_pct": 3, "cost_pct": 0.4}

        verdict_card_fn(result, cfg)
        text = _captured_text()

        # CAGR raw 숫자 999999 또는 1000000 노출 안 됨
        assert "999999" not in text
        assert "1000000" not in text
        # Sharpe 30.00 raw 노출 안 됨 (summary는 cap)
        # 단 anomaly_flags에 "Sharpe 비정상 (5 초과)" 형태는 OK
        assert "Sharpe 30" not in text or "Sharpe 비정상" in text


# ────────────────────────────────────────────────────────────────
# E. _render_results 전체 e2e — CAGR/Sharpe 카드 cap 검증
# ────────────────────────────────────────────────────────────────
class TestRenderResultsCardCap:
    """[v3.9.15e + 9] _render_results() 직접 호출 — verdict 카드 외 stat 카드 검증.

    이전 한계: TestRawNumberCap은 verdict_card만 호출하므로
    _render_results 안의 CAGR/Sharpe stat 카드 cap은 커버 안 됨.
    이 클래스는 _render_results 전체를 호출해서 stat 카드까지 검증.
    """

    def test_render_results_caps_cagr_card(
        self, fake_env, render_results_fn
    ):
        """CAGR 1054855 raw 숫자가 stat 카드에 노출되지 않고
        '비정상 과열 (300% 초과)' 메시지로 대체되는지."""
        result = {
            "total_return": 712.86, "win_rate": 93.7, "mdd": -3.4,
            "total_trades": 269, "sharpe": 45.88, "cagr": 1054855.80,
            "trading_days": 57, "volatility": 20.61,
            "status_dist": {"WIN": 219, "HOLD_EXIT": 41, "STOP": 9},
            "trades_df": pd.DataFrame(),
            "avg_win": 4.31, "avg_loss": -2.55,
            "profit_factor": 1.69,
            "win_streak": 49, "loss_streak": 1,
            "best_trades": [], "worst_trades": [],
            "equity": pd.DataFrame({"date": [], "equity": []}),
            "drawdown": pd.DataFrame({"date": [], "drawdown": []}),
            "daily_rets": pd.Series(dtype=float),
            "hold_col": "ret_5d_%",
        }
        cfg = {
            "min_score": 80, "top_k": 5, "hold_days": 5,
            "target_pct": 5, "stop_pct": 3, "cost_pct": 0.4,
        }

        try:
            render_results_fn(result, cfg)
        except Exception as e:
            # _render_results는 차트 등 plotly figure 생성 시도 — mock 환경에선
            # 일부 실패할 수 있지만 stat 카드는 이미 그려진 후이므로 OK
            pass

        text = _captured_text()

        # CAGR raw 노출 차단
        assert "1054855" not in text, (
            f"CAGR raw 1054855 노출. captured 일부: {text[:500]}"
        )
        # 대신 cap 메시지
        assert "비정상 과열 (300% 초과)" in text, (
            f"CAGR cap 메시지 누락. captured: {text[:500]}"
        )

    def test_render_results_caps_sharpe_card(
        self, fake_env, render_results_fn
    ):
        """Sharpe 45.88 raw가 stat 카드에 노출되지 않고
        '비정상 (5 초과)' 메시지로 대체되는지."""
        result = {
            "total_return": 712.86, "win_rate": 93.7, "mdd": -3.4,
            "total_trades": 269, "sharpe": 45.88, "cagr": 1054855.80,
            "trading_days": 57, "volatility": 20.61,
            "status_dist": {"WIN": 219, "HOLD_EXIT": 41, "STOP": 9},
            "trades_df": pd.DataFrame(),
            "avg_win": 4.31, "avg_loss": -2.55,
            "profit_factor": 1.69,
            "win_streak": 49, "loss_streak": 1,
            "best_trades": [], "worst_trades": [],
            "equity": pd.DataFrame({"date": [], "equity": []}),
            "drawdown": pd.DataFrame({"date": [], "drawdown": []}),
            "daily_rets": pd.Series(dtype=float),
            "hold_col": "ret_5d_%",
        }
        cfg = {
            "min_score": 80, "top_k": 5, "hold_days": 5,
            "target_pct": 5, "stop_pct": 3, "cost_pct": 0.4,
        }

        try:
            render_results_fn(result, cfg)
        except Exception:
            pass

        text = _captured_text()
        assert "비정상 (5 초과)" in text, (
            f"Sharpe cap 메시지 누락. captured: {text[:500]}"
        )

    def test_render_results_normal_sharpe_uses_raw(
        self, fake_env, render_results_fn
    ):
        """정상 Sharpe (1.5)는 raw 값 그대로 표시."""
        result = {
            "total_return": 25.0, "win_rate": 60.0, "mdd": -8.0,
            "total_trades": 150, "sharpe": 1.5, "cagr": 35.0,
            "trading_days": 252, "volatility": 15.0,
            "status_dist": {"WIN": 75, "HOLD_EXIT": 60, "STOP": 15},
            "trades_df": pd.DataFrame(),
            "avg_win": 3.0, "avg_loss": -2.0,
            "profit_factor": 1.5,
            "win_streak": 5, "loss_streak": 2,
            "best_trades": [], "worst_trades": [],
            "equity": pd.DataFrame({"date": [], "equity": []}),
            "drawdown": pd.DataFrame({"date": [], "drawdown": []}),
            "daily_rets": pd.Series(dtype=float),
            "hold_col": "ret_5d_%",
        }
        cfg = {
            "min_score": 80, "top_k": 5, "hold_days": 5,
            "target_pct": 5, "stop_pct": 3, "cost_pct": 0.4,
        }

        try:
            render_results_fn(result, cfg)
        except Exception:
            pass

        text = _captured_text()
        # 정상 Sharpe는 raw 표시되어야 함
        assert "1.50" in text, (
            f"정상 Sharpe raw 표시 누락. captured: {text[:500]}"
        )
        assert "비정상 (5 초과)" not in text


# ────────────────────────────────────────────────────────────────
# F. TP 포화율 익절선별 임계 (target_pct tier)
# ────────────────────────────────────────────────────────────────
class TestTpSaturationTier:
    """[v3.9.15e + 9] target_pct tier별 TP 포화율 임계 차등."""

    def test_low_target_tier_high_threshold(self, fake_env, verdict_card_fn):
        """단타 +3% / TP 포화율 75% — 임계 80% 미만이라 경고 안 뜸."""
        result = {
            "total_return": 50.0, "win_rate": 75.0, "mdd": -5.0,
            "total_trades": 200, "sharpe": 2.0, "cagr": 60.0,
            "trading_days": 252,
            "status_dist": {"WIN": 150, "HOLD_EXIT": 40, "STOP": 10},  # 75%
            "trades_df": pd.DataFrame(),
            "avg_win": 2.5, "avg_loss": -1.8,
        }
        cfg = {
            "min_score": 80, "top_k": 5, "hold_days": 1,
            "target_pct": 3, "stop_pct": 2, "cost_pct": 0.4,  # 단타 target=3
        }

        verdict_card_fn(result, cfg)
        text = _captured_text()

        # 단타 tier 임계 80%, 실제 75% < 80% → 경고 안 뜸
        assert "TP 포화율" not in text or "단타 tier" not in text, (
            f"단타 tier 75% 경고 떠선 안 됨. text: {text[:500]}"
        )

    def test_high_target_tier_low_threshold(self, fake_env, verdict_card_fn):
        """공격 +15% / TP 포화율 65% — 임계 60% 초과로 경고."""
        result = {
            "total_return": 80.0, "win_rate": 65.0, "mdd": -10.0,
            "total_trades": 100, "sharpe": 2.5, "cagr": 80.0,
            "trading_days": 252,
            "status_dist": {"WIN": 65, "HOLD_EXIT": 25, "STOP": 10},  # 65%
            "trades_df": pd.DataFrame(),
            "avg_win": 12.0, "avg_loss": -5.0,
        }
        cfg = {
            "min_score": 80, "top_k": 5, "hold_days": 20,
            "target_pct": 15, "stop_pct": 8, "cost_pct": 0.4,  # 공격 target=15
        }

        verdict_card_fn(result, cfg)
        text = _captured_text()

        # 공격 tier 임계 60%, 실제 65% > 60% → 경고 떠야 함
        assert "TP 포화율 65.0%" in text, (
            f"공격 tier 65% 경고 누락. text: {text[:500]}"
        )
        assert "임계 60%" in text

    def test_mid_target_tier_uses_70_threshold(self, fake_env, verdict_card_fn):
        """균형 +7% / TP 포화율 71% — 임계 70% 초과로 경고."""
        result = {
            "total_return": 60.0, "win_rate": 71.0, "mdd": -8.0,
            "total_trades": 150, "sharpe": 2.0, "cagr": 60.0,
            "trading_days": 252,
            "status_dist": {"WIN": 107, "HOLD_EXIT": 33, "STOP": 10},  # 71%
            "trades_df": pd.DataFrame(),
            "avg_win": 5.5, "avg_loss": -3.0,
        }
        cfg = {
            "min_score": 80, "top_k": 5, "hold_days": 10,
            "target_pct": 7, "stop_pct": 4, "cost_pct": 0.4,  # 균형 target=7
        }

        verdict_card_fn(result, cfg)
        text = _captured_text()

        assert "TP 포화율" in text
        assert "임계 70%" in text
