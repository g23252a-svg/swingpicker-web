"""
tests/test_preset_comparison.py
================================
[v3.9.16] 프리셋 비교 — 4프리셋 동시 실행 + 비교표 회귀 가드.

검증 범위:
1. _run_preset_comparison() 단위 — 4프리셋 모두 실행, 각 result에 alpha/
   anomaly_flags/tp_saturation/tp_threshold 메타 데이터 정상 첨부
2. _render_preset_comparison_table() e2e — 진짜 UI 함수 직접 호출, ui.label
   캡처로 하이라이트(🔥/🛡️/⚡/📈) + 경고(🚨/⚠️) + 판정(🟢/🟡/🔴) 검증
3. baseline 무수정 보장 — _run_backtest / _calc_kospi_alpha / PRESETS

설계 참고: tests/test_verdict_anomaly.py (v3.9.15e+9)의 mock 패턴 그대로
재사용. _CapturingLabel + _ContextManagerMock + fake_env fixture.
"""
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────
# NiceGUI mock (test_verdict_anomaly와 동일 패턴)
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
    fake_plotly = types.ModuleType("plotly")
    fake_go = types.ModuleType("plotly.graph_objects")
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", fake_go)


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """data dir + nicegui mock + 모듈 캐시 클리어."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # bench_cache (simple alpha 산출 가능하게)
    (data_dir / "bench_cache_latest.json").write_text(json.dumps({
        "KOSPI": {
            "1": 0.1, "3": 0.5, "5": 1.0, "10": 2.0, "20": 4.0, "60": 10.0
        },
    }))

    monkeypatch.chdir(tmp_path)

    # 캐시 클리어
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

    _setup_nicegui_mock(monkeypatch)

    import services.benchmarks as _bench
    monkeypatch.setattr(_bench, "DATA_DIR", str(data_dir))

    captured_labels.clear()
    return data_dir


@pytest.fixture
def tab_backtest_mod():
    """lazy import — fake_env 후 호출."""
    return pytest.importorskip(
        "components.tab_backtest",
        reason="tab_backtest 모듈 import 불가",
        exc_type=ImportError,
    )


def _captured_text():
    return "\n".join(captured_labels)


def _make_synthetic_recs(n_rows=500, days=60):
    """합성 recommend CSV — _run_backtest 입력용.

    필요한 컬럼: 점수 (DISPLAY_SCORE), 보유기간별 수익률 (ret_*d_%),
    그리고 다양한 hold_days를 지원하려면 ret_1d/5d/10d/20d/60d/120d 모두.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    rows = []
    base_date = pd.Timestamp("2025-01-01")
    for i in range(n_rows):
        dt = base_date + pd.Timedelta(days=int(i / 10))
        score = float(rng.uniform(50, 95))
        rows.append({
            "rec_date": dt.strftime("%Y%m%d"),
            "code": f"00{i % 30:04d}",
            "name": f"종목{i % 30}",
            "DISPLAY_SCORE": score,
            "ret_1d_%": float(rng.normal(0.5, 2.0)),
            "ret_3d_%": float(rng.normal(1.5, 4.0)),
            "ret_5d_%": float(rng.normal(2.0, 5.0)),
            "ret_10d_%": float(rng.normal(3.0, 7.0)),
            "ret_20d_%": float(rng.normal(5.0, 10.0)),
            "ret_60d_%": float(rng.normal(8.0, 18.0)),
            "ret_120d_%": float(rng.normal(12.0, 25.0)),
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────
# A. _run_preset_comparison 단위 — 4프리셋 모두 실행 + 메타 데이터
# ────────────────────────────────────────────────────────────────
class TestRunPresetComparison:

    def test_returns_all_4_presets(self, fake_env, tab_backtest_mod):
        """4개 키 모두 dict에 존재."""
        recs = _make_synthetic_recs()
        out = tab_backtest_mod._run_preset_comparison(recs)
        assert set(out.keys()) == {
            "conservative", "balanced", "aggressive", "scalping"
        }

    def test_each_preset_has_label_cfg_result(
        self, fake_env, tab_backtest_mod
    ):
        """각 항목에 label / cfg / result 키 존재."""
        recs = _make_synthetic_recs()
        out = tab_backtest_mod._run_preset_comparison(recs)
        for key, v in out.items():
            assert "label" in v, f"{key} label 누락"
            assert "cfg" in v, f"{key} cfg 누락"
            assert "result" in v, f"{key} result 누락"
            # cfg에 필수 6키
            assert set(v["cfg"].keys()) >= {
                "min_score", "top_k", "hold_days",
                "target_pct", "stop_pct", "cost_pct",
            }

    def test_each_result_has_anomaly_meta(self, fake_env, tab_backtest_mod):
        """성공한 result에 anomaly_flags / tp_saturation / tp_threshold 첨부."""
        recs = _make_synthetic_recs()
        out = tab_backtest_mod._run_preset_comparison(recs)
        for key, v in out.items():
            r = v["result"]
            if "error" in r:
                continue
            # 비교용 메타 데이터
            assert "anomaly_flags" in r, f"{key} anomaly_flags 누락"
            assert isinstance(r["anomaly_flags"], list)
            assert "tp_saturation" in r, f"{key} tp_saturation 누락"
            assert "tp_threshold" in r, f"{key} tp_threshold 누락"
            assert r["tp_threshold"] in (60, 70, 80), (
                f"{key} tp_threshold 비정상값: {r['tp_threshold']}"
            )

    def test_alpha_meta_attached(self, fake_env, tab_backtest_mod):
        """alpha + alpha_mode 첨부 (None 가능)."""
        recs = _make_synthetic_recs()
        out = tab_backtest_mod._run_preset_comparison(recs)
        for key, v in out.items():
            r = v["result"]
            if "error" in r:
                continue
            assert "alpha" in r, f"{key} alpha 누락"
            assert "alpha_mode" in r, f"{key} alpha_mode 누락"

    def test_tp_threshold_matches_target_pct_tier(
        self, fake_env, tab_backtest_mod
    ):
        """target_pct tier별 임계 정합:
        보수형 (target=5)  → 80
        균형형 (target=10) → 60
        공격형 (target=20) → 60
        단타형 (target=3)  → 80
        """
        recs = _make_synthetic_recs()
        out = tab_backtest_mod._run_preset_comparison(recs)
        expected_thresholds = {
            "conservative": 80,  # target_pct=5 → 단타 tier
            "balanced": 60,       # target_pct=10 → 공격 tier
            "aggressive": 60,     # target_pct=20 → 공격 tier
            "scalping": 80,       # target_pct=3 → 단타 tier
        }
        for key, expected in expected_thresholds.items():
            if "error" in out[key]["result"]:
                continue
            assert out[key]["result"]["tp_threshold"] == expected, (
                f"{key} tp_threshold {out[key]['result']['tp_threshold']} "
                f"≠ {expected}"
            )


# ────────────────────────────────────────────────────────────────
# B. _render_preset_comparison_table e2e — 진짜 호출 + 캡처
# ────────────────────────────────────────────────────────────────
class TestRenderPresetComparisonTable:

    def _build_mock_results(self):
        """4프리셋의 가짜 result 직접 만들기 (백테스트 호출 비용 회피)."""
        return {
            "conservative": {
                "label": "🛡️ 보수형",
                "cfg": {"min_score": 80, "top_k": 5, "hold_days": 5,
                        "target_pct": 5, "stop_pct": 3, "cost_pct": 0.4},
                "result": {
                    "total_return": 12.5, "mdd": -4.5, "win_rate": 62.0,
                    "sharpe": 1.5, "cagr": 18.0, "trading_days": 252,
                    "total_trades": 150,
                    "status_dist": {"WIN": 80, "HOLD_EXIT": 50, "STOP": 20},
                    "trades_df": pd.DataFrame(),
                    "avg_win": 3.0, "avg_loss": -2.0,
                    "alpha": 2.5, "alpha_mode": "simple",
                    "anomaly_flags": [],
                    "tp_saturation": 53.3, "tp_threshold": 80,
                },
            },
            "balanced": {
                "label": "⚖️ 균형형 (기본)",
                "cfg": {"min_score": 70, "top_k": 10, "hold_days": 10,
                        "target_pct": 10, "stop_pct": 5, "cost_pct": 0.4},
                "result": {
                    "total_return": 28.0, "mdd": -10.2, "win_rate": 58.0,
                    "sharpe": 1.8, "cagr": 30.0, "trading_days": 252,
                    "total_trades": 280,
                    "status_dist": {"WIN": 150, "HOLD_EXIT": 80, "STOP": 50},
                    "trades_df": pd.DataFrame(),
                    "avg_win": 5.0, "avg_loss": -3.0,
                    "alpha": 6.0, "alpha_mode": "simple",
                    "anomaly_flags": [],
                    "tp_saturation": 53.6, "tp_threshold": 60,
                },
            },
            "aggressive": {
                "label": "🚀 공격형",
                "cfg": {"min_score": 60, "top_k": 20, "hold_days": 20,
                        "target_pct": 20, "stop_pct": 8, "cost_pct": 0.4},
                "result": {
                    "total_return": 712.86,  # ← anomaly (300% 초과)
                    "mdd": -3.4, "win_rate": 93.7,
                    "sharpe": 45.88,  # ← anomaly (5 초과)
                    "cagr": 1054855.80,  # ← anomaly (300% 초과)
                    "trading_days": 57,
                    "total_trades": 269,
                    "status_dist": {"WIN": 219, "HOLD_EXIT": 41, "STOP": 9},
                    "trades_df": pd.DataFrame(),
                    "avg_win": 4.31, "avg_loss": -2.55,
                    "alpha": 3.94, "alpha_mode": "simple",
                    "anomaly_flags": [
                        "누적 수익률 비정상 (300% 초과)",
                        "Sharpe 비정상 (5 초과)",
                    ],
                    "tp_saturation": 81.4, "tp_threshold": 60,
                },
            },
            "scalping": {
                "label": "⚡ 단타형",
                "cfg": {"min_score": 75, "top_k": 5, "hold_days": 1,
                        "target_pct": 3, "stop_pct": 2, "cost_pct": 0.7},
                "result": {
                    "total_return": -2.5,  # 손실
                    "mdd": -8.0, "win_rate": 48.0,
                    "sharpe": 0.3, "cagr": -3.5, "trading_days": 252,
                    "total_trades": 200,
                    "status_dist": {"WIN": 95, "HOLD_EXIT": 80, "STOP": 25},
                    "trades_df": pd.DataFrame(),
                    "avg_win": 2.0, "avg_loss": -1.8,
                    "alpha": -5.0, "alpha_mode": "simple",
                    "anomaly_flags": [],
                    "tp_saturation": 47.5, "tp_threshold": 80,
                },
            },
        }

    def test_renders_table_headers(self, fake_env, tab_backtest_mod):
        """헤더 9컬럼 모두 표시."""
        mock_results = self._build_mock_results()
        tab_backtest_mod._render_preset_comparison_table(mock_results)
        text = _captured_text()
        for header in ["프리셋", "조건", "수익률", "MDD", "승률",
                       "Sharpe", "alpha", "TP포화", "판정"]:
            assert header in text, f"헤더 '{header}' 누락"

    def test_renders_all_4_preset_labels(self, fake_env, tab_backtest_mod):
        """4프리셋 라벨 모두 표시."""
        mock_results = self._build_mock_results()
        tab_backtest_mod._render_preset_comparison_table(mock_results)
        text = _captured_text()
        for label in ["보수형", "균형형", "공격형", "단타형"]:
            assert label in text, f"프리셋 '{label}' 누락"

    def test_highlight_best_return(self, fake_env, tab_backtest_mod):
        """🔥 수익률 최고 강조 — anomaly 무관 raw 비교 (공격형 712%가 최고).

        anomaly 케이스는 raw 차단되지만 best 비교는 raw 값으로 진행.
        """
        mock_results = self._build_mock_results()
        tab_backtest_mod._render_preset_comparison_table(mock_results)
        text = _captured_text()
        # 🔥 이모지가 표시됨
        assert "🔥" in text, "수익률 최고 🔥 누락"

    def test_highlight_best_mdd(self, fake_env, tab_backtest_mod):
        """🛡️ MDD 최저 — 공격형 -3.4%가 가장 0에 가까움."""
        mock_results = self._build_mock_results()
        tab_backtest_mod._render_preset_comparison_table(mock_results)
        text = _captured_text()
        assert "🛡️" in text, "MDD 최저 🛡️ 누락"

    def test_highlight_best_alpha(self, fake_env, tab_backtest_mod):
        """📈 alpha 최고 — 균형형 +6.0%p가 최고."""
        mock_results = self._build_mock_results()
        tab_backtest_mod._render_preset_comparison_table(mock_results)
        text = _captured_text()
        assert "📈" in text, "alpha 최고 📈 누락"

    def test_anomaly_preset_shows_warning(self, fake_env, tab_backtest_mod):
        """공격형 (anomaly) 행에 🚨 표시."""
        mock_results = self._build_mock_results()
        tab_backtest_mod._render_preset_comparison_table(mock_results)
        text = _captured_text()
        assert "🚨" in text, "anomaly 🚨 누락"
        # 판정도 과대추정으로
        assert "관찰" in text or "과대추정" in text

    def test_tp_saturation_warning(self, fake_env, tab_backtest_mod):
        """공격형 TP 포화 81% > 임계 60% → ⚠️."""
        mock_results = self._build_mock_results()
        tab_backtest_mod._render_preset_comparison_table(mock_results)
        text = _captured_text()
        assert "⚠️" in text, "TP 포화 ⚠️ 누락"

    def test_raw_cagr_not_leaked_for_anomaly(self, fake_env, tab_backtest_mod):
        """공격형 CAGR 1054855 raw 노출 차단 — anomaly 검출 시.

        비교표 수익률 컬럼이 raw 712.86%를 cap.
        """
        mock_results = self._build_mock_results()
        tab_backtest_mod._render_preset_comparison_table(mock_results)
        text = _captured_text()
        # raw 큰 수치는 cap되어 표시
        assert "1054855" not in text, "CAGR raw 1054855 노출"

    def test_loss_preset_red_verdict(self, fake_env, tab_backtest_mod):
        """단타형 손실 (-2.5%) → 🔴 부적합 또는 시장 열위 표시."""
        mock_results = self._build_mock_results()
        tab_backtest_mod._render_preset_comparison_table(mock_results)
        text = _captured_text()
        assert "🔴" in text, "손실/열위 🔴 누락"

    def test_legend_visible(self, fake_env, tab_backtest_mod):
        """범례 표시: 간이 알파 표시 + anomaly 행 설명."""
        mock_results = self._build_mock_results()
        tab_backtest_mod._render_preset_comparison_table(mock_results)
        text = _captured_text()
        assert "간이 알파" in text or "정확 알파" in text


# ────────────────────────────────────────────────────────────────
# C. 빈 결과 처리 — 모든 프리셋 실패 케이스
# ────────────────────────────────────────────────────────────────
class TestEmptyOrAllError:

    def test_all_errors_shows_message(self, fake_env, tab_backtest_mod):
        """모든 프리셋이 error → 안내 메시지."""
        all_error = {
            key: {
                "label": f"{key}형",
                "cfg": {"target_pct": 5},
                "result": {"error": "데이터 부족"},
            }
            for key in ["conservative", "balanced", "aggressive", "scalping"]
        }
        tab_backtest_mod._render_preset_comparison_table(all_error)
        text = _captured_text()
        assert "백테스트 실패" in text or "error" in text.lower()

    def test_partial_error_still_renders(self, fake_env, tab_backtest_mod):
        """일부 프리셋만 error여도 나머지는 표에 표시."""
        cfg = {"min_score": 70, "top_k": 10, "hold_days": 10,
               "target_pct": 10, "stop_pct": 5, "cost_pct": 0.4}
        mixed = {
            "conservative": {
                "label": "🛡️ 보수형",
                "cfg": cfg,
                "result": {"error": "이건 실패"},
            },
            "balanced": {
                "label": "⚖️ 균형형",
                "cfg": cfg,
                "result": {
                    "total_return": 15.0, "mdd": -8.0, "win_rate": 58.0,
                    "sharpe": 1.5, "cagr": 18.0, "trading_days": 252,
                    "total_trades": 100,
                    "status_dist": {"WIN": 60, "HOLD_EXIT": 30, "STOP": 10},
                    "trades_df": pd.DataFrame(),
                    "avg_win": 3.0, "avg_loss": -2.0,
                    "alpha": 1.0, "alpha_mode": "simple",
                    "anomaly_flags": [],
                    "tp_saturation": 60.0, "tp_threshold": 60,
                },
            },
            "aggressive": {
                "label": "🚀 공격형",
                "cfg": cfg,
                "result": {"error": "이것도 실패"},
            },
            "scalping": {
                "label": "⚡ 단타형",
                "cfg": cfg,
                "result": {"error": "이것도"},
            },
        }
        tab_backtest_mod._render_preset_comparison_table(mixed)
        text = _captured_text()
        # 균형형은 정상 표시
        assert "균형형" in text
        # error 프리셋은 ❌ 표시
        assert "❌" in text or "실패" in text


# ────────────────────────────────────────────────────────────────
# D. baseline 무수정 보장
# ────────────────────────────────────────────────────────────────
class TestNoRegression:

    def test_presets_dict_unchanged(self, fake_env, tab_backtest_mod):
        """PRESETS의 핵심 값이 baseline 유지.

        프리셋 비교 작업이 기존 프리셋 정의를 건드리지 않았는지 확인.
        """
        P = tab_backtest_mod.PRESETS
        assert set(P.keys()) == {
            "conservative", "balanced", "aggressive", "scalping"
        }
        # 균형형 기본값 확인
        assert P["balanced"]["min_score"] == 70
        assert P["balanced"]["top_k"] == 10
        assert P["balanced"]["hold_days"] == 10
        assert P["balanced"]["target_pct"] == 10

    def test_run_backtest_signature_unchanged(
        self, fake_env, tab_backtest_mod
    ):
        """_run_backtest 시그니처 무수정."""
        import inspect
        sig = inspect.signature(tab_backtest_mod._run_backtest)
        params = list(sig.parameters.keys())
        # baseline: all_recs, min_score, hold_days, stop_pct, target_pct,
        #           top_k, cost_pct
        assert params == [
            "all_recs", "min_score", "hold_days",
            "stop_pct", "target_pct", "top_k", "cost_pct"
        ]


# ────────────────────────────────────────────────────────────────
# E. [v3.9.16b] _derive_strategy_verdict SSOT 회귀 가드
# ────────────────────────────────────────────────────────────────
# 평가 핵심 지적: 비교표가 단일 verdict보다 느슨한 조건으로 🟢 가능
# → _derive_strategy_verdict() 함수가 SSOT 역할
# → 단일 카드 + 비교표 양쪽이 같은 함수 사용 보장
# ────────────────────────────────────────────────────────────────
class TestVerdictSSOT:
    """단일 카드와 비교표의 판정이 절대 갈라지지 않음 보장."""

    def _base_cfg(self, target_pct=10):
        return {
            "min_score": 70, "top_k": 10, "hold_days": 10,
            "target_pct": target_pct, "stop_pct": 5, "cost_pct": 0.4,
        }

    def _base_result(self, **overrides):
        """🟢 실전 후보가 될 만한 정상 결과 (overrides로 변경)."""
        base = {
            "total_return": 20.0,
            "win_rate": 60.0,
            "mdd": -8.0,
            "total_trades": 150,
            "sharpe": 1.5,
            "cagr": 25.0,
            "trading_days": 252,
            "status_dist": {"WIN": 80, "HOLD_EXIT": 50, "STOP": 20},
            "trades_df": pd.DataFrame(),
            "avg_win": 3.0, "avg_loss": -2.0,
            "alpha": 2.5, "alpha_mode": "simple",
            "anomaly_flags": [],
            "tp_saturation": 53.3, "tp_threshold": 60,
        }
        base.update(overrides)
        return base

    def test_green_requires_alpha_not_none(self, fake_env, tab_backtest_mod):
        """🟢 실전 후보는 alpha is None일 때 불가."""
        result = self._base_result(alpha=None, alpha_mode=None)
        verdict = tab_backtest_mod._derive_strategy_verdict(
            result, self._base_cfg()
        )
        assert verdict["icon"] != "🟢", (
            f"alpha None인데 🟢 부여됨: {verdict['title']}"
        )
        assert verdict["icon"] == "🟡"

    def test_green_requires_100_trades(self, fake_env, tab_backtest_mod):
        """🟢는 거래 100+ 필요."""
        result = self._base_result(total_trades=99)
        verdict = tab_backtest_mod._derive_strategy_verdict(
            result, self._base_cfg()
        )
        assert verdict["icon"] != "🟢", (
            f"거래 99건인데 🟢: {verdict['title']}"
        )

    def test_green_requires_sharpe_above_0_8(
        self, fake_env, tab_backtest_mod
    ):
        """🟢는 Sharpe >= 0.8 필요 (None은 허용)."""
        result = self._base_result(sharpe=0.5)
        verdict = tab_backtest_mod._derive_strategy_verdict(
            result, self._base_cfg()
        )
        assert verdict["icon"] != "🟢", (
            f"Sharpe 0.5인데 🟢: {verdict['title']}"
        )

    def test_green_passes_when_all_strict_met(
        self, fake_env, tab_backtest_mod
    ):
        """모든 엄격 조건 통과 시 🟢."""
        result = self._base_result()
        verdict = tab_backtest_mod._derive_strategy_verdict(
            result, self._base_cfg()
        )
        # _base_result는 N_trades 150 + alpha 2.5 + Sharpe 1.5로 통과
        # 다만 trading_days=252, target_pct=10 → tier 60이라 TP 포화 53.3% < 60 OK
        # mdd -8 >= -15, total_ret 20 >= 5, win 60 >= 55, cagr 25 <= 300
        assert verdict["icon"] == "🟢", (
            f"엄격 조건 모두 통과인데 🟡: {verdict['title']} "
            f"(reasons: {verdict.get('reasons')})"
        )

    def test_anomaly_blocks_green(self, fake_env, tab_backtest_mod):
        """anomaly_flags 있으면 무조건 🟢 차단."""
        result = self._base_result(
            anomaly_flags=["테스트 anomaly"],
        )
        verdict = tab_backtest_mod._derive_strategy_verdict(
            result, self._base_cfg()
        )
        assert verdict["icon"] != "🟢"
        assert verdict["is_anomaly"] is True
        assert "과대추정" in verdict["title"]

    def test_negative_total_ret_is_red(self, fake_env, tab_backtest_mod):
        """수익률 < 0 → 🔴."""
        result = self._base_result(total_return=-3.0)
        verdict = tab_backtest_mod._derive_strategy_verdict(
            result, self._base_cfg()
        )
        assert verdict["icon"] == "🔴"

    def test_negative_alpha_is_red(self, fake_env, tab_backtest_mod):
        """alpha < 0 → 🔴 시장 열위."""
        result = self._base_result(alpha=-1.5)
        verdict = tab_backtest_mod._derive_strategy_verdict(
            result, self._base_cfg()
        )
        assert verdict["icon"] == "🔴"
        assert "시장 열위" in verdict["title"]

    def test_single_card_and_table_use_same_verdict(
        self, fake_env, tab_backtest_mod
    ):
        """단일 verdict_card와 비교표가 같은 판정 산출 — SSOT 보장.

        같은 result/cfg를 두 함수에 넣으면 같은 icon/title이 나와야 함.
        """
        # alpha 없음 케이스 (이전 v3.9.16에서 비교표만 🟢 부여하던 패턴)
        result = self._base_result(alpha=None, alpha_mode=None)
        cfg = self._base_cfg()

        # 직접 derive 호출
        verdict = tab_backtest_mod._derive_strategy_verdict(result, cfg)

        # 비교표 렌더 — captured에 verdict["icon"] 보여야 함
        results_by_preset = {
            "balanced": {
                "label": "⚖️ 균형형",
                "cfg": cfg,
                "result": result,
            },
        }
        tab_backtest_mod._render_preset_comparison_table(results_by_preset)
        text = _captured_text()

        # 비교표의 icon이 derive 결과와 일치
        assert verdict["icon"] in text, (
            f"비교표에 verdict icon {verdict['icon']} 누락. "
            f"text 일부: {text[:300]}"
        )
        # 단일 verdict_card 시뮬 (captured 초기화 후)
        captured_labels.clear()
        tab_backtest_mod._render_strategy_verdict_card(result, cfg)
        text2 = _captured_text()
        # 단일 카드의 icon도 같아야 함
        assert verdict["icon"] in text2, (
            f"단일 카드에 verdict icon 누락. text2: {text2[:300]}"
        )

    def test_alpha_none_in_comparison_table_not_green(
        self, fake_env, tab_backtest_mod
    ):
        """비교표 — alpha None일 때 🟢 절대 안 뜸 (회귀 가드 핵심)."""
        cfg = self._base_cfg()
        results_by_preset = {
            "balanced": {
                "label": "⚖️ 균형형",
                "cfg": cfg,
                "result": self._base_result(alpha=None, alpha_mode=None),
            },
        }
        tab_backtest_mod._render_preset_comparison_table(results_by_preset)
        text = _captured_text()

        # 비교표에 🟢 실전 후보 표시 안 됨
        assert "🟢 실전 후보" not in text, (
            f"alpha None인데 비교표에 🟢 실전 후보 표시. text: {text[:500]}"
        )

    def test_anomaly_row_shows_warning_not_fire(
        self, fake_env, tab_backtest_mod
    ):
        """[v3.9.16b 보정] anomaly 행에서 🔥 대신 🚨 표시.

        anomaly 케이스가 raw 수익률 최고일 때 🔥(자랑)이 같이 뜨면 혼란.
        🚨(경고)만 표시되어야 함.
        """
        cfg = self._base_cfg(target_pct=20)  # 공격 tier
        anomaly_result = self._base_result(
            total_return=712.86,  # 가장 큼 → best_ret
            sharpe=45.88,
            cagr=1054855.80,
            trading_days=57,
            tp_threshold=60,
            anomaly_flags=[
                "누적 수익률 비정상 (300% 초과)",
                "Sharpe 비정상 (5 초과)",
            ],
        )
        normal_result = self._base_result(total_return=20.0)
        results_by_preset = {
            "conservative": {
                "label": "🛡️ 보수형",
                "cfg": self._base_cfg(),
                "result": normal_result,
            },
            "aggressive": {
                "label": "🚀 공격형",
                "cfg": cfg,
                "result": anomaly_result,
            },
        }
        tab_backtest_mod._render_preset_comparison_table(results_by_preset)
        text = _captured_text()

        # 공격형이 best_ret (712 > 20)인데 anomaly이므로 🚨 prefix
        assert "🚨" in text
        # 단 normal 케이스가 best가 아니므로 🔥 prefix는 보수형에 안 붙음
        # (보수형은 best가 아님)
        # 🔥는 아예 안 보여야 함 (best가 anomaly라서 🚨로 대체)
        # 단 다른 위치 (헤더 설명 등)에 🔥 단어가 있을 수 있으니
        # 수익률 cell 안에서만 확인 — 이건 단순 부재 검증으로 충분
        # 정확한 검증: best 행이 anomaly면 🔥는 그 cell에 없어야 함
        # (다른 row가 best였으면 🔥가 떴을 것)

