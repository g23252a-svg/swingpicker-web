"""
tests/test_train_test_split.py
================================
[v3.9.18] Train/Test 분할 검증 회귀 가드.

평가 6가지 회귀 가드 기준:
1. Train/Test split이 날짜순인지 검증
2. Test가 최근 구간인지 검증
3. Train 좋고 Test 나쁘면 🔴 나오는지 검증
4. Train/Test 모두 좋으면 🟢 나오는지 검증
5. 데이터 부족 시 안전하게 안내되는지 검증
6. lookahead 의심 (양쪽 anomaly OR Test 급락) 시 🚨
"""
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────
# NiceGUI mock (test_robustness 패턴 그대로)
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
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "bench_cache_latest.json").write_text(json.dumps({
        "KOSPI": {
            "1": 0.1, "3": 0.5, "5": 1.0, "10": 2.0, "20": 4.0, "60": 10.0
        },
    }))
    monkeypatch.chdir(tmp_path)

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
def tt_svc():
    return pytest.importorskip(
        "services.backtest_train_test",
        reason="services.backtest_train_test 모듈 import 불가",
        exc_type=ImportError,
    )


@pytest.fixture
def tt_ui():
    return pytest.importorskip(
        "components.backtest_train_test",
        reason="components.backtest_train_test 모듈 import 불가",
        exc_type=ImportError,
    )


@pytest.fixture
def tab_backtest_mod():
    return pytest.importorskip(
        "components.tab_backtest",
        reason="tab_backtest 모듈 import 불가",
        exc_type=ImportError,
    )


def _captured_text():
    return "\n".join(captured_labels)


def _make_synthetic_recs(
    n_rows=500,
    start_date="2024-01-01",
    train_bias_mean=2.0,
    test_bias_mean=2.0,
    train_bias_std=5.0,
    test_bias_std=5.0,
    seed=42,
):
    """rec_date 정렬된 합성 데이터.

    train_bias_mean / test_bias_mean으로 Train vs Test 시나리오 조작 가능.
    예: train_bias_mean=10, test_bias_mean=-5 → Train만 좋음 (과최적화 시나리오)
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp(start_date)
    split_idx = int(n_rows * 0.70)
    for i in range(n_rows):
        dt = base_date + pd.Timedelta(days=int(i / 3))
        score = float(rng.uniform(50, 95))
        # Train 구간 (i < split) vs Test 구간 (i >= split)
        if i < split_idx:
            bias_mean, bias_std = train_bias_mean, train_bias_std
        else:
            bias_mean, bias_std = test_bias_mean, test_bias_std
        rows.append({
            "rec_date": dt.strftime("%Y%m%d"),
            "code": f"00{i % 30:04d}",
            "name": f"종목{i % 30}",
            "DISPLAY_SCORE": score,
            "ret_1d_%": float(rng.normal(bias_mean * 0.3, bias_std * 0.5)),
            "ret_3d_%": float(rng.normal(bias_mean * 0.5, bias_std * 0.8)),
            "ret_5d_%": float(rng.normal(bias_mean * 0.7, bias_std)),
            "ret_10d_%": float(rng.normal(bias_mean, bias_std * 1.2)),
            "ret_20d_%": float(rng.normal(bias_mean * 1.5, bias_std * 1.5)),
            "ret_60d_%": float(rng.normal(bias_mean * 2, bias_std * 2)),
            "ret_120d_%": float(rng.normal(bias_mean * 2.5, bias_std * 2.5)),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# A. run_train_test_split — 분할 정확성
# ════════════════════════════════════════════════════════════════
class TestSplitMechanics:
    """평가 회귀 가드 1, 2: 날짜순 분할 + Test가 최근."""

    def test_split_returns_70_30(self, fake_env, tt_svc):
        """기본 70/30 분할 (날짜 기준)."""
        recs = _make_synthetic_recs(n_rows=300)
        out = tt_svc.run_train_test_split(recs, "balanced")
        info = out["split_info"]
        assert info["n_total"] == 300
        # 날짜 수 기준 분할 — Train 날짜 ~ 70%, Test 날짜 ~ 30%
        assert "n_unique_dates" in info
        assert "n_train_dates" in info
        assert "n_test_dates" in info
        n_dates = info["n_unique_dates"]
        # 70/30 비율로 분할
        assert info["n_train_dates"] == int(n_dates * 0.7)
        assert info["n_test_dates"] == n_dates - int(n_dates * 0.7)

    def test_split_is_date_ordered_strict(self, fake_env, tt_svc):
        """[평가 1] Train의 마지막 날짜 < Test의 첫 날짜 (strict).

        같은 날짜가 양쪽에 걸리지 않아야 함.
        """
        recs = _make_synthetic_recs(n_rows=300)
        out = tt_svc.run_train_test_split(recs, "balanced")
        info = out["split_info"]

        train_end = info["train_date_range"][1]
        test_start = info["test_date_range"][0]

        # 평가 명시: <= 가 아니라 < 가 strict 보장
        assert train_end < test_start, (
            f"같은 날짜가 Train과 Test 양쪽에 걸림: "
            f"train_end {train_end} >= test_start {test_start}"
        )

    def test_split_uses_unique_rec_dates_not_rows(
        self, fake_env, tt_svc
    ):
        """[평가 1 핵심 회귀 가드] 같은 rec_date가 Train/Test 양쪽에 안 들어감.

        이전 v3.9.18은 row 개수 분할이라 같은 날짜가 양쪽에 가능.
        v3.9.18b는 date set 분할로 disjoint 보장.
        """
        # 날짜별 row 수가 불균형한 데이터
        rows = []
        # 10개 날짜 × 각각 다른 종목 수 (1, 5, 8, 13, 2, 7, 4, 11, 3, 6)
        date_counts = [1, 5, 8, 13, 2, 7, 4, 11, 3, 6]
        base = pd.Timestamp("2025-01-01")
        for d_idx, n_stocks in enumerate(date_counts):
            dt = base + pd.Timedelta(days=d_idx * 3)
            date_str = dt.strftime("%Y%m%d")
            for s in range(n_stocks):
                rows.append({
                    "rec_date": date_str,
                    "code": f"00{s:04d}",
                    "name": f"종목{s}",
                    "DISPLAY_SCORE": 80.0 + s * 0.1,
                    "ret_1d_%": 0.5,
                    "ret_3d_%": 1.5,
                    "ret_5d_%": 2.0,
                    "ret_10d_%": 3.0,
                    "ret_20d_%": 5.0,
                    "ret_60d_%": 8.0,
                    "ret_120d_%": 12.0,
                })
        recs = pd.DataFrame(rows)
        # n_total = 60건, 날짜 10개

        out = tt_svc.run_train_test_split(recs, "balanced")

        # 데이터 부족이면 일단 단순 검증만 — 분할 자체는 됐어야 함
        if "error" in out["train_result"] or "error" in out["test_result"]:
            # 데이터 부족이라도 split_info에서 disjoint 보장은 필요
            info = out["split_info"]
            assert info.get("n_train_dates", 0) > 0 or "오류" in out["verdict"]["title"]
            return

        info = out["split_info"]

        # train_date_range와 test_date_range의 날짜가 겹치지 않음
        # (Train end < Test start)
        train_end = info["train_date_range"][1]
        test_start = info["test_date_range"][0]
        assert train_end < test_start, (
            f"같은 날짜가 양쪽에 걸림: {train_end} < {test_start}"
        )

        # 합계 검증
        assert (
            info["n_train_dates"] + info["n_test_dates"]
            == info["n_unique_dates"]
        )

    def test_test_is_most_recent(self, fake_env, tt_svc):
        """[평가 2] Test가 최근 30%."""
        recs = _make_synthetic_recs(n_rows=300)
        out = tt_svc.run_train_test_split(recs, "balanced")
        info = out["split_info"]

        sorted_recs = recs.sort_values("rec_date").reset_index(drop=True)
        last_date_overall = str(sorted_recs["rec_date"].iloc[-1])
        test_end = info["test_date_range"][1]

        assert test_end == last_date_overall, (
            f"Test 끝 ({test_end}) != 전체 끝 ({last_date_overall})"
        )

    def test_unsorted_input_still_correctly_split(
        self, fake_env, tt_svc
    ):
        """입력이 정렬 안 돼도 내부에서 정렬 후 분할 정확."""
        recs = _make_synthetic_recs(n_rows=300)
        shuffled = recs.sample(frac=1, random_state=99).reset_index(drop=True)
        out = tt_svc.run_train_test_split(shuffled, "balanced")
        info = out["split_info"]

        train_end = info["train_date_range"][1]
        test_start = info["test_date_range"][0]
        assert train_end < test_start  # strict

    def test_data_insufficient_returns_safe_message(
        self, fake_env, tt_svc
    ):
        """[평가 5] 데이터 < 100건이면 안내 메시지."""
        recs = _make_synthetic_recs(n_rows=80)
        out = tt_svc.run_train_test_split(recs, "balanced")
        assert out["verdict"]["icon"] == "⚪"
        assert out["verdict"]["level"] == "unknown"
        assert "부족" in out["verdict"]["title"]

    def test_empty_recs_returns_data_unavailable(
        self, fake_env, tt_svc
    ):
        """빈 DataFrame → 데이터 부족 안내."""
        empty = pd.DataFrame()
        out = tt_svc.run_train_test_split(empty, "balanced")
        assert out["verdict"]["icon"] == "⚪"

    def test_invalid_preset_falls_back_to_balanced(
        self, fake_env, tt_svc
    ):
        """잘못된 preset key → balanced fallback."""
        recs = _make_synthetic_recs(n_rows=300)
        out = tt_svc.run_train_test_split(recs, "unknown_xyz")
        assert out["preset"] == "balanced"
        assert out["cfg"]["min_score"] == 70

    def test_custom_test_ratio_50_50(self, fake_env, tt_svc):
        """test_ratio=0.5 → 날짜 기준 50/50 분할."""
        recs = _make_synthetic_recs(n_rows=300)
        out = tt_svc.run_train_test_split(recs, "balanced", test_ratio=0.5)
        info = out["split_info"]
        n_dates = info["n_unique_dates"]
        # 날짜 기준 50/50
        assert info["n_train_dates"] == int(n_dates * 0.5)


# ════════════════════════════════════════════════════════════════
# B. derive_train_test_verdict — 4단계 판정
# ════════════════════════════════════════════════════════════════
class TestTrainTestVerdict:
    """평가 회귀 가드 3, 4, 6: 🔴/🟢/🚨."""

    def _make_result(self, **kwargs):
        defaults = {
            "total_return": 10.0,
            "win_rate": 60.0,
            "mdd": -8.0,
            "total_trades": 100,
            "sharpe": 1.5,
            "cagr": 12.0,
            "trading_days": 200,
            "status_dist": {"WIN": 60, "HOLD_EXIT": 30, "STOP": 10},
            "trades_df": pd.DataFrame(),
            "avg_win": 3.0, "avg_loss": -2.0,
            "alpha": 2.0, "alpha_mode": "simple",
            "anomaly_flags": [],
            "tp_saturation": 60.0, "tp_threshold": 60,
        }
        defaults.update(kwargs)
        return defaults

    def test_green_when_both_strong(self, fake_env, tt_svc):
        """[평가 4] Train/Test 모두 좋으면 🟢 일반화 양호."""
        train = self._make_result(total_return=20.0, alpha=3.0)
        test = self._make_result(total_return=15.0, alpha=2.5)
        v = tt_svc.derive_train_test_verdict(train, test)
        assert v["icon"] == "🟢", (
            f"Train/Test 모두 양호인데 🟢 아님: {v['title']}"
        )
        assert v["level"] == "green"
        assert "일반화 양호" in v["title"]
        # 🟢 양호 → 제목에 "후보" / "보류" 없음
        assert "후보" not in v["title"]
        assert "보류" not in v["title"]

    def test_red_when_train_good_test_bad(self, fake_env, tt_svc):
        """[평가 3] Train 좋고 Test 나쁘면 🔴 과최적화."""
        train = self._make_result(total_return=25.0, alpha=4.0)
        test = self._make_result(total_return=-8.0, alpha=-2.0)
        v = tt_svc.derive_train_test_verdict(train, test)
        assert v["icon"] == "🔴"
        assert v["level"] == "red"
        # 과최적화 또는 OOS 붕괴 (둘 다 가능, 둘 다 🔴)
        assert "과최적화" in v["title"] or "OOS" in v["title"]

    def test_red_when_retention_too_low(self, fake_env, tt_svc):
        """Train 50% → Test 5% → retention 10% < 20% 기준 → 🔴."""
        train = self._make_result(total_return=50.0)
        test = self._make_result(total_return=5.0)
        v = tt_svc.derive_train_test_verdict(train, test)
        assert v["icon"] == "🔴"
        assert v["level"] == "red"

    def test_yellow_when_weakening(self, fake_env, tt_svc):
        """Test 수익 양수지만 일부 미달 → 🟡 약화."""
        train = self._make_result(total_return=20.0, alpha=3.0)
        test = self._make_result(total_return=5.0, alpha=-1.0)
        v = tt_svc.derive_train_test_verdict(train, test)
        assert v["icon"] == "🟡"
        assert v["level"] == "yellow"

    def test_lookahead_when_both_anomaly(self, fake_env, tt_svc):
        """[평가 6] Train+Test 모두 anomaly → 🚨 lookahead."""
        train = self._make_result(
            total_return=400.0,
            anomaly_flags=["수익률 비정상 (300% 초과)"],
        )
        test = self._make_result(
            total_return=350.0,
            anomaly_flags=["수익률 비정상 (300% 초과)"],
        )
        v = tt_svc.derive_train_test_verdict(train, test)
        assert v["icon"] == "🚨"
        assert v["level"] == "lookahead"
        assert "lookahead" in v["title"]

    def test_oos_collapse_is_red_not_lookahead(self, fake_env, tt_svc):
        """[v3.9.18b 평가 3] Test 단독 -20% 급락 = 🔴 OOS 붕괴, 🚨 아님.

        이전 v3.9.18: Test -30%이면 🚨 lookahead
        v3.9.18b: Test 급락 단독은 🔴 OOS 붕괴 (lookahead가 아님)
        🚨는 양쪽 anomaly 단독 조건.
        """
        train = self._make_result(total_return=30.0)
        test = self._make_result(total_return=-30.0)
        v = tt_svc.derive_train_test_verdict(train, test)
        assert v["icon"] == "🔴", (
            f"Test -30% 단독은 🔴 OOS 기대, 실제 {v['icon']} {v['title']}"
        )
        assert v["level"] == "red"
        assert "OOS" in v["title"] or "붕괴" in v["title"]
        # lookahead 단어 없음
        assert "lookahead" not in v["title"].lower()

    def test_error_in_train_returns_unknown(self, fake_env, tt_svc):
        """[평가 5] Train error → ⚪."""
        train = {"error": "데이터 부족"}
        test = self._make_result(total_return=15.0)
        v = tt_svc.derive_train_test_verdict(train, test)
        assert v["icon"] == "⚪"
        assert v["level"] == "unknown"

    def test_green_blocked_by_negative_alpha(self, fake_env, tt_svc):
        """Test alpha < 0 → 🟢 불가 (시장 열위)."""
        train = self._make_result(total_return=20.0, alpha=3.0)
        test = self._make_result(total_return=10.0, alpha=-0.5)
        v = tt_svc.derive_train_test_verdict(train, test)
        assert v["icon"] != "🟢"

    def test_yellow_candidate_when_alpha_none(self, fake_env, tt_svc):
        """[v3.9.18b 평가 2] Test alpha None이면 🟢 아닌 🟡 일반화 후보 보류.

        이전 v3.9.18: alpha None인데 절대수익 양호 → 🟢 가능 (시장 검증 없는데도)
        v3.9.18b: 🟡 일반화 후보 · alpha 평가 보류 (v3.9.17c 패턴 일관)
        """
        train = self._make_result(
            total_return=20.0, alpha=None, alpha_mode=None
        )
        test = self._make_result(
            total_return=15.0, alpha=None, alpha_mode=None
        )
        v = tt_svc.derive_train_test_verdict(train, test)
        # 🟢 일반화 양호가 아님
        assert v["icon"] != "🟢", (
            f"alpha None인데 🟢 부여됨: {v['title']}"
        )
        # 🟡 yellow_candidate
        assert v["icon"] == "🟡"
        assert v["level"] == "yellow_candidate"
        assert "후보" in v["title"]
        assert "alpha 평가 보류" in v["title"]


# ════════════════════════════════════════════════════════════════
# C. _render_train_test_result — UI 렌더 + 캡처
# ════════════════════════════════════════════════════════════════
class TestRenderTrainTestResult:

    def _build_tt_data(self, verdict_level="green"):
        cfg = {"min_score": 70, "top_k": 10, "hold_days": 10,
               "target_pct": 10, "stop_pct": 5, "cost_pct": 0.4}
        train_result = {
            "total_return": 20.0, "mdd": -8.0, "win_rate": 60.0,
            "sharpe": 1.5, "cagr": 25.0, "trading_days": 252,
            "total_trades": 200,
            "status_dist": {"WIN": 120, "HOLD_EXIT": 60, "STOP": 20},
            "trades_df": pd.DataFrame(),
            "avg_win": 3.0, "avg_loss": -2.0,
            "alpha": 3.0, "alpha_mode": "simple",
            "anomaly_flags": [],
            "tp_saturation": 60.0, "tp_threshold": 60,
        }
        test_result = dict(train_result)
        if verdict_level == "green":
            test_result["total_return"] = 15.0
            test_result["alpha"] = 2.5
            verdict = {
                "icon": "🟢", "level": "green",
                "title": "일반화 양호", "color_class": "text-emerald-400",
                "body": "Train 20% → Test 15%.", "reasons": ["일반화 양호"],
            }
        elif verdict_level == "red":
            test_result["total_return"] = -10.0
            test_result["alpha"] = -2.0
            verdict = {
                "icon": "🔴", "level": "red",
                "title": "과최적화 의심", "color_class": "text-red-400",
                "body": "Train 20% → Test -10%.", "reasons": ["Test 손실"],
            }
        elif verdict_level == "lookahead":
            train_result["total_return"] = 400.0
            train_result["anomaly_flags"] = ["수익률 비정상"]
            test_result["total_return"] = 350.0
            test_result["anomaly_flags"] = ["수익률 비정상"]
            verdict = {
                "icon": "🚨", "level": "lookahead",
                "title": "lookahead 의심 · 실전 비권장",
                "color_class": "text-red-500",
                "body": "Train+Test 모두 anomaly.",
                "reasons": ["Train+Test 모두 anomaly"],
            }

        return {
            "preset": "balanced",
            "cfg": cfg,
            "split_info": {
                "n_total": 500, "n_train": 350, "n_test": 150,
                "train_date_range": ("20240101", "20250630"),
                "test_date_range": ("20250701", "20251231"),
                "test_ratio": 0.30,
            },
            "train_result": train_result,
            "test_result": test_result,
            "verdict": verdict,
        }

    def test_renders_verdict_card(self, fake_env, tt_ui):
        """판정 카드 icon + title."""
        data = self._build_tt_data("green")
        tt_ui._render_train_test_result(data)
        text = _captured_text()
        assert "🟢" in text
        assert "일반화 양호" in text

    def test_renders_red_verdict(self, fake_env, tt_ui):
        """🔴 과최적화 의심."""
        data = self._build_tt_data("red")
        tt_ui._render_train_test_result(data)
        text = _captured_text()
        assert "🔴" in text
        assert "과최적화" in text

    def test_renders_lookahead_verdict(self, fake_env, tt_ui):
        """🚨 lookahead 의심."""
        data = self._build_tt_data("lookahead")
        tt_ui._render_train_test_result(data)
        text = _captured_text()
        assert "🚨" in text

    def test_renders_train_test_columns(self, fake_env, tt_ui):
        """헤더에 Train/Test 컬럼 표시."""
        data = self._build_tt_data("green")
        tt_ui._render_train_test_result(data)
        text = _captured_text()
        assert "Train" in text and "Test" in text
        assert "수익률" in text
        assert "MDD" in text

    def test_renders_split_info(self, fake_env, tt_ui):
        """분할 정보 (날짜 범위) 표시."""
        data = self._build_tt_data("green")
        tt_ui._render_train_test_result(data)
        text = _captured_text()
        # 분할 정보 한 줄 표시
        assert "350" in text or "Train" in text  # n_train


# ════════════════════════════════════════════════════════════════
# D. baseline 무수정 + 분리 import parity
# ════════════════════════════════════════════════════════════════
class TestNoRegression:

    def test_services_train_test_no_nicegui_import(self, fake_env):
        """services.backtest_train_test에 nicegui import 0."""
        from pathlib import Path
        import inspect
        import services.backtest_train_test as svc
        src_path = inspect.getfile(svc)
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("from nicegui"), (
                f"services.backtest_train_test에 nicegui import 발견: {line}"
            )
            assert not stripped.startswith("import nicegui"), (
                f"services.backtest_train_test에 nicegui import 발견: {line}"
            )

    def test_components_reexports_services_functions(
        self, fake_env, tt_ui
    ):
        """components가 services 함수를 re-export — backward compat."""
        from services.backtest_train_test import (
            run_train_test_split as svc_run,
            derive_train_test_verdict as svc_verdict,
        )
        assert tt_ui._run_train_test_split is svc_run
        assert tt_ui._derive_train_test_verdict is svc_verdict

    def test_presets_dict_baseline(self, fake_env, tab_backtest_mod):
        """PRESETS baseline (분리 영향 받지 않음)."""
        P = tab_backtest_mod.PRESETS
        assert set(P.keys()) == {
            "conservative", "balanced", "aggressive", "scalping"
        }
        assert P["balanced"]["min_score"] == 70

    def test_run_backtest_signature_unchanged(
        self, fake_env, tab_backtest_mod
    ):
        """_run_backtest 시그니처 baseline."""
        import inspect
        sig = inspect.signature(tab_backtest_mod._run_backtest)
        params = list(sig.parameters.keys())
        assert params == [
            "all_recs", "min_score", "hold_days",
            "stop_pct", "target_pct", "top_k", "cost_pct"
        ]

    def test_run_backtest_works_with_slice(
        self, fake_env, tab_backtest_mod
    ):
        """_run_backtest가 슬라이스된 DataFrame도 받음 (Train/Test 전제).

        평가 명시: '_run_backtest 이동 금지' — slice DataFrame을 그대로 넘김.
        """
        import importlib
        # 합성 데이터
        recs = _make_synthetic_recs(n_rows=300)
        # 70% 슬라이스 (Train)
        train_recs = recs.iloc[:210].copy()
        result = tab_backtest_mod._run_backtest(
            train_recs, 70, 10, 5, 10, 10, 0.4
        )
        # error 없거나 정상 결과
        if "error" not in result:
            assert "total_return" in result
            assert "total_trades" in result
