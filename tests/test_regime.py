"""
tests/test_regime.py
=====================
[v3.9.19] 시장 국면별 성과 회귀 가드.

평가 6가지 회귀 가드 기준:
1. 국면 데이터 없는 경우 안전 fallback
2. 특정 국면 표본 부족 시 ⚪ 처리
3. NORMAL은 좋고 CRITICAL은 나쁘면 "국면 의존/하락장 취약" 판정
4. 국면별 날짜 매칭이 rec_date 기준으로 되는지 검증
5. 기존 _run_backtest / PRESETS / 추천/매수가/Top3 baseline 변경 없음
6. (추가) UI 렌더 출력 + SSOT parity
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


def _write_run_health(data_dir, date_str, macro_risk):
    """run_health_YYYYMMDD.json 합성."""
    path = data_dir / f"run_health_{date_str}.json"
    health = {
        "trade_ymd": date_str,
        "status": "OK",
        "macro_risk": macro_risk,
        "market_breadth": 50.0,
        "confidence_score": 100.0,
        "max_allowed_route": "ATTACK",
    }
    path.write_text(json.dumps(health))


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
def regime_svc():
    return pytest.importorskip(
        "services.backtest_regime",
        reason="services.backtest_regime 모듈 import 불가",
        exc_type=ImportError,
    )


@pytest.fixture
def regime_ui():
    return pytest.importorskip(
        "components.backtest_regime",
        reason="components.backtest_regime 모듈 import 불가",
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


def _make_synthetic_recs_with_dates(date_list, n_per_date=10, ret_bias=2.0):
    """각 날짜에 n_per_date건씩 합성."""
    import numpy as np
    rng = np.random.default_rng(42)
    rows = []
    for d_idx, date_str in enumerate(date_list):
        for s in range(n_per_date):
            rows.append({
                "rec_date": date_str,
                "code": f"00{s % 30:04d}",
                "name": f"종목{s % 30}",
                "DISPLAY_SCORE": float(rng.uniform(70, 95)),
                "ret_1d_%": float(rng.normal(ret_bias * 0.3, 1.5)),
                "ret_3d_%": float(rng.normal(ret_bias * 0.5, 3.0)),
                "ret_5d_%": float(rng.normal(ret_bias * 0.7, 4.0)),
                "ret_10d_%": float(rng.normal(ret_bias, 5.0)),
                "ret_20d_%": float(rng.normal(ret_bias * 1.5, 8.0)),
                "ret_60d_%": float(rng.normal(ret_bias * 2, 12.0)),
                "ret_120d_%": float(rng.normal(ret_bias * 2.5, 18.0)),
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# A. load_macro_regime_map — JSON 로드
# ════════════════════════════════════════════════════════════════
class TestLoadMacroRegimeMap:
    """평가 회귀 가드 1: 국면 데이터 없는 경우 안전 fallback."""

    def test_returns_empty_when_no_data_dir(self, fake_env, regime_svc):
        """data 디렉토리 없으면 빈 dict 반환 (예외 없음)."""
        out = regime_svc.load_macro_regime_map("nonexistent_dir")
        assert out == {}

    def test_returns_empty_when_no_run_health_files(
        self, fake_env, regime_svc
    ):
        """data 디렉토리는 있지만 run_health 파일 없으면 빈 dict."""
        # fake_env에 bench_cache는 있지만 run_health는 없음
        out = regime_svc.load_macro_regime_map(str(fake_env))
        assert out == {}

    def test_loads_regime_correctly(self, fake_env, regime_svc):
        """파일에서 trade_ymd + macro_risk 정확 로드."""
        _write_run_health(fake_env, "20260101", "NORMAL")
        _write_run_health(fake_env, "20260102", "CAUTION")
        _write_run_health(fake_env, "20260103", "CRITICAL")
        out = regime_svc.load_macro_regime_map(str(fake_env))
        assert out == {
            "20260101": "NORMAL",
            "20260102": "CAUTION",
            "20260103": "CRITICAL",
        }

    def test_ignores_invalid_macro_risk(self, fake_env, regime_svc):
        """잘못된 macro_risk 값은 무시."""
        (fake_env / "run_health_20260101.json").write_text(
            json.dumps({"trade_ymd": "20260101", "macro_risk": "INVALID_X"})
        )
        out = regime_svc.load_macro_regime_map(str(fake_env))
        assert out == {}

    def test_uses_filename_as_fallback_when_no_trade_ymd(
        self, fake_env, regime_svc
    ):
        """JSON에 trade_ymd 없어도 파일명에서 날짜 추출."""
        (fake_env / "run_health_20260101.json").write_text(
            json.dumps({"macro_risk": "NORMAL"})
        )
        out = regime_svc.load_macro_regime_map(str(fake_env))
        assert "20260101" in out
        assert out["20260101"] == "NORMAL"

    def test_skips_malformed_json(self, fake_env, regime_svc):
        """깨진 JSON은 silent skip (예외 없음)."""
        (fake_env / "run_health_20260101.json").write_text("not valid json{")
        _write_run_health(fake_env, "20260102", "NORMAL")
        out = regime_svc.load_macro_regime_map(str(fake_env))
        # 깨진 파일은 skip, 정상 파일만 로드
        assert out == {"20260102": "NORMAL"}


# ════════════════════════════════════════════════════════════════
# B. run_regime_split — 분할 + 날짜 매칭
# ════════════════════════════════════════════════════════════════
class TestRunRegimeSplit:
    """평가 회귀 가드 4: rec_date 기준 매칭."""

    def test_no_regime_data_returns_safe_fallback(
        self, fake_env, regime_svc
    ):
        """[평가 1] run_health JSON 없으면 ⚪ '국면 데이터 없음'."""
        recs = _make_synthetic_recs_with_dates(["20260101"], n_per_date=30)
        out = regime_svc.run_regime_split(recs, "balanced", str(fake_env))
        assert out["verdict"]["icon"] == "⚪"
        assert "국면 데이터 없음" in out["verdict"]["title"]

    def test_matches_recs_to_regime_by_rec_date(
        self, fake_env, regime_svc
    ):
        """[평가 4] rec_date 기준 정확 매칭."""
        # 3개 날짜 × 30건씩 = 90건
        _write_run_health(fake_env, "20260101", "NORMAL")
        _write_run_health(fake_env, "20260102", "CAUTION")
        _write_run_health(fake_env, "20260103", "CRITICAL")
        recs = _make_synthetic_recs_with_dates(
            ["20260101", "20260102", "20260103"], n_per_date=30
        )

        out = regime_svc.run_regime_split(recs, "balanced", str(fake_env))

        # 각 국면에 30건씩 들어갔는지
        assert out["regimes"]["NORMAL"]["n_recs"] == 30
        assert out["regimes"]["CAUTION"]["n_recs"] == 30
        assert out["regimes"]["CRITICAL"]["n_recs"] == 30

    def test_insufficient_sample_per_regime_returns_error(
        self, fake_env, regime_svc
    ):
        """[평가 2] 국면별 표본 < MIN_TRADES_PER_REGIME(20)이면 error."""
        # 모든 날짜 NORMAL인데 표본은 10건만 (20 미만)
        _write_run_health(fake_env, "20260101", "NORMAL")
        recs = _make_synthetic_recs_with_dates(["20260101"], n_per_date=10)

        out = regime_svc.run_regime_split(recs, "balanced", str(fake_env))

        # NORMAL은 표본 부족 error, CAUTION/CRITICAL은 0건
        assert "error" in out["regimes"]["NORMAL"]["result"]
        assert "표본 부족" in out["regimes"]["NORMAL"]["result"]["error"]

    def test_unmatched_dates_not_counted(self, fake_env, regime_svc):
        """rec_date에 매칭되는 macro_risk 없으면 그 추천은 제외."""
        _write_run_health(fake_env, "20260101", "NORMAL")
        # 20260102에 대한 run_health 없음
        recs = _make_synthetic_recs_with_dates(
            ["20260101", "20260102"], n_per_date=30
        )

        out = regime_svc.run_regime_split(recs, "balanced", str(fake_env))
        # NORMAL은 30건 (20260101만), 다른 국면 0건
        assert out["regimes"]["NORMAL"]["n_recs"] == 30
        # 20260102 추천은 어디에도 매칭 안 됨

    def test_invalid_preset_falls_back_to_balanced(
        self, fake_env, regime_svc
    ):
        """잘못된 preset → balanced fallback."""
        _write_run_health(fake_env, "20260101", "NORMAL")
        recs = _make_synthetic_recs_with_dates(["20260101"], n_per_date=30)
        out = regime_svc.run_regime_split(recs, "unknown", str(fake_env))
        assert out["preset"] == "balanced"

    def test_empty_recs_returns_data_unavailable(
        self, fake_env, regime_svc
    ):
        """빈 DataFrame → 추천 데이터 부족 안내."""
        _write_run_health(fake_env, "20260101", "NORMAL")
        out = regime_svc.run_regime_split(
            pd.DataFrame(), "balanced", str(fake_env)
        )
        assert out["verdict"]["icon"] == "⚪"
        assert "데이터 부족" in out["verdict"]["title"]

    # ──────────────────────────────────────────────────────────────
    # [v3.9.19b] 평가 2 회귀 가드 — rec_date 정규화
    # ──────────────────────────────────────────────────────────────
    def test_rec_date_normalized_from_dashed_format(
        self, fake_env, regime_svc
    ):
        """[v3.9.19b 평가 2] rec_date "2026-01-01" 형식도 "20260101"로 매칭.

        run_health JSON은 YYYYMMDD인데 recommend가 YYYY-MM-DD면
        이전 v3.9.19는 astype(str)만 해서 매칭 실패.
        v3.9.19b는 pd.to_datetime 통해 정규화.
        """
        _write_run_health(fake_env, "20260101", "NORMAL")
        # rec_date를 "2026-01-01" 형식으로 만듦
        recs = _make_synthetic_recs_with_dates(["2026-01-01"], n_per_date=30)
        out = regime_svc.run_regime_split(recs, "balanced", str(fake_env))

        # 정규화로 NORMAL과 매칭 → 30건
        assert out["regimes"]["NORMAL"]["n_recs"] == 30, (
            f"rec_date 정규화 실패: NORMAL={out['regimes']['NORMAL']['n_recs']}"
        )

    def test_rec_date_normalized_from_datetime(
        self, fake_env, regime_svc
    ):
        """rec_date가 Timestamp 객체일 경우에도 정규화."""
        _write_run_health(fake_env, "20260101", "NORMAL")
        recs = _make_synthetic_recs_with_dates(["20260101"], n_per_date=30)
        # rec_date 컬럼을 Timestamp로 변환
        recs["rec_date"] = pd.to_datetime(recs["rec_date"], format="%Y%m%d")

        out = regime_svc.run_regime_split(recs, "balanced", str(fake_env))
        assert out["regimes"]["NORMAL"]["n_recs"] == 30

    def test_rec_date_unparseable_falls_back_to_str(
        self, fake_env, regime_svc
    ):
        """파싱 불가 rec_date는 원본 문자열 fallback (에러 없이 처리).

        예: "INVALID_DATE" 같은 값이 들어와도 silent skip — 정상 날짜만 매칭.
        """
        _write_run_health(fake_env, "20260101", "NORMAL")
        # 일부 row의 rec_date를 정상, 일부를 깨진 값으로
        recs_normal = _make_synthetic_recs_with_dates(
            ["20260101"], n_per_date=30
        )
        recs_broken = _make_synthetic_recs_with_dates(
            ["20260102"], n_per_date=10
        )
        recs_broken["rec_date"] = "INVALID_DATE"
        recs = pd.concat([recs_normal, recs_broken], ignore_index=True)

        # 예외 없이 정상 처리
        out = regime_svc.run_regime_split(recs, "balanced", str(fake_env))
        # 정상 rec_date는 NORMAL에 매칭 (30건)
        assert out["regimes"]["NORMAL"]["n_recs"] == 30


# ════════════════════════════════════════════════════════════════
# C. derive_regime_verdict — 판정 로직
# ════════════════════════════════════════════════════════════════
class TestRegimeVerdict:
    """평가 회귀 가드 3: 🟢/🟡/🔴/⚪."""

    def _make_result(self, **kwargs):
        defaults = {
            "total_return": 10.0,
            "mdd": -8.0,
            "win_rate": 60.0,
            "sharpe": 1.5,
            "alpha": 2.0,
            "alpha_mode": "simple",
            "anomaly_flags": [],
            "stop_ratio": 10.0,
            "tp_saturation": 50.0,
            "tp_threshold": 60,
        }
        defaults.update(kwargs)
        return defaults

    def _make_regimes(self, normal=None, caution=None, critical=None):
        """국면별 result dict 묶음."""
        def _wrap(r):
            if r is None:
                return {"n_recs": 0, "result": {"error": "표본 부족"}}
            return {"n_recs": 30, "result": r}
        return {
            "NORMAL": _wrap(normal),
            "CAUTION": _wrap(caution),
            "CRITICAL": _wrap(critical),
        }

    def test_green_when_all_regimes_strong(self, fake_env, regime_svc):
        """3국면 모두 양호 → 🟢 전천후."""
        regimes = self._make_regimes(
            normal=self._make_result(total_return=15.0, alpha=3.0),
            caution=self._make_result(total_return=8.0, alpha=2.0),
            critical=self._make_result(total_return=5.0, alpha=1.0),
        )
        v = regime_svc.derive_regime_verdict(regimes)
        assert v["icon"] == "🟢"
        assert v["level"] == "green"
        assert "전천후" in v["title"]

    def test_red_when_critical_loses(self, fake_env, regime_svc):
        """[평가 3] CRITICAL에서 큰 손실 → 🔴 하락장 취약."""
        regimes = self._make_regimes(
            normal=self._make_result(total_return=15.0),
            caution=self._make_result(total_return=5.0),
            critical=self._make_result(total_return=-10.0, mdd=-25.0),
        )
        v = regime_svc.derive_regime_verdict(regimes)
        assert v["icon"] == "🔴"
        assert v["level"] == "red"
        assert "하락장 취약" in v["title"]

    def test_red_when_caution_loses(self, fake_env, regime_svc):
        """[평가 3] CAUTION에서 큰 손실 → 🔴 하락장 취약."""
        regimes = self._make_regimes(
            normal=self._make_result(total_return=15.0),
            caution=self._make_result(total_return=-8.0),
            critical=self._make_result(total_return=3.0),
        )
        v = regime_svc.derive_regime_verdict(regimes)
        assert v["icon"] == "🔴"

    def test_yellow_when_critical_weak_but_not_loss(
        self, fake_env, regime_svc
    ):
        """[평가 3] NORMAL 좋고 CRITICAL은 약하지만 손실 아님 → 🟡 국면 의존.

        시나리오: 절대수익은 모두 양수지만 CRITICAL에서 alpha 음수 (시장 열위) →
        🟢 조건 (alpha_ok) 실패 → 🟡
        """
        regimes = self._make_regimes(
            normal=self._make_result(total_return=20.0, alpha=3.0),
            caution=self._make_result(total_return=4.0, alpha=0.5),
            # CRITICAL: 수익 양수지만 alpha 음수 (시장 열위)
            critical=self._make_result(total_return=1.0, alpha=-1.5),
        )
        v = regime_svc.derive_regime_verdict(regimes)
        # CRITICAL > -5% 라 🔴 아님, but alpha 음수로 🟢 차단 → 🟡
        assert v["icon"] == "🟡"
        assert v["level"] == "yellow"
        assert "국면 의존" in v["title"]

    def test_unknown_when_two_regimes_missing(
        self, fake_env, regime_svc
    ):
        """[평가 2] 평가 가능 국면 1개만 → ⚪ 표본 부족."""
        regimes = self._make_regimes(
            normal=self._make_result(total_return=15.0),
            caution=None,
            critical=None,
        )
        v = regime_svc.derive_regime_verdict(regimes)
        assert v["icon"] == "⚪"
        assert v["level"] == "unknown"

    def test_unknown_when_all_regimes_missing(
        self, fake_env, regime_svc
    ):
        """모든 국면 error → ⚪."""
        regimes = self._make_regimes()  # 모두 None
        v = regime_svc.derive_regime_verdict(regimes)
        assert v["icon"] == "⚪"

    def test_red_priority_over_green(self, fake_env, regime_svc):
        """[평가 3 회귀 핵심] NORMAL이 양호해도 CRITICAL 손실 시 🔴.

        같은 시나리오를 평가가 명시: "NORMAL은 좋고 CRITICAL은 나쁘면
        '국면 의존/하락장 취약' 판정"
        """
        regimes = self._make_regimes(
            normal=self._make_result(total_return=25.0, alpha=4.0),
            caution=self._make_result(total_return=8.0),
            critical=self._make_result(total_return=-12.0, mdd=-18.0),
        )
        v = regime_svc.derive_regime_verdict(regimes)
        # NORMAL 25% + alpha 4% 양호해도 CRITICAL -12% → 🔴 강제
        assert v["icon"] == "🔴", (
            f"CRITICAL 큰 손실인데 🔴 아님: {v['icon']} {v['title']}"
        )
        assert "하락장 취약" in v["title"]

    # ──────────────────────────────────────────────────────────────
    # [v3.9.19b] 평가 1 회귀 가드 — alpha coverage 분기
    # ──────────────────────────────────────────────────────────────
    def test_yellow_candidate_when_all_alpha_none(
        self, fake_env, regime_svc
    ):
        """[v3.9.19b 평가 1] 3국면 모두 alpha None → 🟡 전천후 후보 (이전 🟢).

        시나리오: 절대수익/MDD/anomaly 모두 양호 but alpha 산출 0.
        - 이전 v3.9.19: alpha_ok = (len==0 or all >= 0) → True → 🟢
        - v3.9.19b: alpha_coverage 0 < 0.67 → 🟡 yellow_candidate
        """
        regimes = self._make_regimes(
            normal=self._make_result(total_return=15.0, alpha=None),
            caution=self._make_result(total_return=8.0, alpha=None),
            critical=self._make_result(total_return=5.0, alpha=None),
        )
        v = regime_svc.derive_regime_verdict(regimes)
        # 🟢 아님 — alpha coverage 부족
        assert v["icon"] != "🟢", (
            f"alpha 전부 None인데 🟢 부여됨: {v['title']}"
        )
        assert v["icon"] == "🟡"
        assert v["level"] == "yellow_candidate"
        assert "전천후 후보" in v["title"]
        assert "alpha 평가 보류" in v["title"]

    def test_yellow_candidate_when_only_one_alpha(
        self, fake_env, regime_svc
    ):
        """[v3.9.19b 평가 1] 3국면 중 1국면만 alpha → coverage 33% < 65% → 🟡.

        REGIME_GREEN_ALPHA_COVERAGE=0.65 — 3국면 중 최소 2국면 alpha 필요.
        """
        regimes = self._make_regimes(
            normal=self._make_result(total_return=15.0, alpha=3.0),
            caution=self._make_result(total_return=8.0, alpha=None),
            critical=self._make_result(total_return=5.0, alpha=None),
        )
        v = regime_svc.derive_regime_verdict(regimes)
        assert v["icon"] == "🟡"
        assert v["level"] == "yellow_candidate"

    def test_green_when_two_alphas_available(
        self, fake_env, regime_svc
    ):
        """[v3.9.19b 평가 1] 3국면 중 2국면 alpha → coverage 67% > 65% → 🟢.

        coverage 67% ≥ 65% 기준 → 🟢. 산출된 alpha 모두 ≥ 0 필수.
        """
        regimes = self._make_regimes(
            normal=self._make_result(total_return=15.0, alpha=3.0),
            caution=self._make_result(total_return=8.0, alpha=1.5),
            critical=self._make_result(total_return=5.0, alpha=None),
        )
        v = regime_svc.derive_regime_verdict(regimes)
        assert v["icon"] == "🟢"
        assert v["level"] == "green"
        assert "전천후" in v["title"]
        # 🟢는 제목에 "후보"/"보류" 없음
        assert "후보" not in v["title"]
        assert "보류" not in v["title"]

    def test_yellow_when_alpha_coverage_full_but_negative(
        self, fake_env, regime_svc
    ):
        """coverage 100% but alpha 일부 음수 → 🟡 (alpha_all_nonneg 실패).

        v3.9.17c의 alpha 음수 차단과 일관.
        """
        regimes = self._make_regimes(
            normal=self._make_result(total_return=15.0, alpha=3.0),
            caution=self._make_result(total_return=8.0, alpha=-0.5),
            critical=self._make_result(total_return=5.0, alpha=1.0),
        )
        v = regime_svc.derive_regime_verdict(regimes)
        # CAUTION alpha 음수 → basic_pass 실패 → 🟡
        assert v["icon"] == "🟡"
        # yellow_candidate가 아닌 일반 yellow (basic_pass 자체 실패)
        assert v["level"] == "yellow"


# ════════════════════════════════════════════════════════════════
# D. _render_regime_table — UI 렌더
# ════════════════════════════════════════════════════════════════
class TestRenderRegimeTable:

    def _build_regime_data(self, verdict_level="green"):
        from services.backtest_regime import REGIMES
        cfg = {"min_score": 70, "top_k": 10, "hold_days": 10,
               "target_pct": 10, "stop_pct": 5, "cost_pct": 0.4}
        regimes = {}
        for r in REGIMES:
            regimes[r] = {
                "n_recs": 50,
                "result": {
                    "total_return": 10.0, "mdd": -8.0, "win_rate": 60.0,
                    "sharpe": 1.5, "alpha": 2.0, "alpha_mode": "simple",
                    "anomaly_flags": [], "stop_ratio": 10.0,
                    "tp_saturation": 50.0, "tp_threshold": 60,
                    "status_dist": {"WIN": 30, "HOLD_EXIT": 15, "STOP": 5},
                    "trades_df": pd.DataFrame(),
                    "total_trades": 50,
                },
            }

        if verdict_level == "green":
            verdict = {
                "icon": "🟢", "level": "green",
                "title": "전천후 · 모든 국면 양호",
                "color_class": "text-emerald-400",
                "body": "3국면 모두 양호.", "reasons": ["전 국면 일관 양호"],
            }
        elif verdict_level == "red":
            regimes["CRITICAL"]["result"]["total_return"] = -10.0
            verdict = {
                "icon": "🔴", "level": "red",
                "title": "하락장 취약 · 실전 비권장",
                "color_class": "text-red-400",
                "body": "CRITICAL 큰 손실.", "reasons": ["CRITICAL -10%"],
            }

        return {
            "preset": "balanced", "cfg": cfg,
            "regime_map_info": {
                "n_dates_with_regime": 30,
                "regime_dist": {"NORMAL": 15, "CAUTION": 10, "CRITICAL": 5},
            },
            "regimes": regimes,
            "verdict": verdict,
        }

    def test_renders_verdict_card(self, fake_env, regime_ui):
        """판정 카드 icon + title."""
        data = self._build_regime_data("green")
        regime_ui._render_regime_table(data)
        text = _captured_text()
        assert "🟢" in text
        assert "전천후" in text

    def test_renders_red_verdict(self, fake_env, regime_ui):
        """🔴 하락장 취약."""
        data = self._build_regime_data("red")
        regime_ui._render_regime_table(data)
        text = _captured_text()
        assert "🔴" in text
        assert "하락장 취약" in text

    def test_renders_three_regime_rows(self, fake_env, regime_ui):
        """NORMAL/CAUTION/CRITICAL 3국면 모두 표시."""
        data = self._build_regime_data("green")
        regime_ui._render_regime_table(data)
        text = _captured_text()
        assert "NORMAL" in text
        assert "CAUTION" in text
        assert "CRITICAL" in text

    def test_renders_table_headers(self, fake_env, regime_ui):
        """표 헤더 — 거래수/수익률/MDD/STOP/이슈 등."""
        data = self._build_regime_data("green")
        regime_ui._render_regime_table(data)
        text = _captured_text()
        assert "수익률" in text
        assert "MDD" in text
        assert "STOP" in text or "STOP율" in text


# ════════════════════════════════════════════════════════════════
# E. baseline 무수정 + 분리 import parity
# ════════════════════════════════════════════════════════════════
class TestNoRegression:
    """평가 회귀 가드 5: 기존 _run_backtest / PRESETS baseline 변경 없음."""

    def test_services_regime_no_nicegui_import(self, fake_env):
        """services.backtest_regime에 nicegui import 0."""
        import inspect
        import services.backtest_regime as svc
        src_path = inspect.getfile(svc)
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("from nicegui"), (
                f"services.backtest_regime에 nicegui import 발견: {line}"
            )
            assert not stripped.startswith("import nicegui"), (
                f"services.backtest_regime에 nicegui import 발견: {line}"
            )

    def test_components_reexports_services_functions(
        self, fake_env, regime_ui
    ):
        """components가 services 함수 re-export (is 검증)."""
        from services.backtest_regime import (
            run_regime_split as svc_run,
            derive_regime_verdict as svc_verdict,
            load_macro_regime_map as svc_load,
        )
        assert regime_ui._run_regime_split is svc_run
        assert regime_ui._derive_regime_verdict is svc_verdict
        assert regime_ui._load_macro_regime_map is svc_load

    def test_presets_dict_baseline(self, fake_env, tab_backtest_mod):
        """PRESETS baseline."""
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

    def test_run_backtest_accepts_filtered_subset(
        self, fake_env, tab_backtest_mod
    ):
        """_run_backtest가 국면 필터된 추천 sub-DataFrame 받아도 정상.

        평가 명시: '_run_backtest 이동 금지' — 그대로 호출, baseline 무변경.
        """
        recs = _make_synthetic_recs_with_dates(
            ["20260101", "20260102"], n_per_date=30
        )
        # NORMAL 시뮬레이션 (한 날짜만 필터)
        subset = recs[recs["rec_date"] == "20260101"].copy()
        result = tab_backtest_mod._run_backtest(
            subset, 70, 10, 5, 10, 10, 0.4
        )
        # error 없거나 정상 결과 (sub set 처리 가능)
        if "error" not in result:
            assert "total_return" in result
