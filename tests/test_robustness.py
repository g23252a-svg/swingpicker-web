"""
tests/test_robustness.py
========================
[v3.9.17] 강건성 테스트 — 27조합 백테스트 회귀 가드.

검증 범위:
1. _run_robustness_test() — 27조합 정확히 생성, 각 조합 cfg 검증
2. _summarize_robustness() — 통계 집계 정확성
3. _derive_robustness_verdict() — 🟢/🟡/🔴 임계값 정합
4. _render_robustness_table() — 진짜 UI 함수 호출, 핵심 라벨 캡처
5. 파일 분리 후 import parity — tab_backtest re-export 동작 보장

설계 패턴: test_verdict_anomaly.py / test_preset_comparison.py와 동일
(_CapturingLabel + _ContextManagerMock + fake_env fixture).
"""
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────
# NiceGUI mock — test_preset_comparison 패턴 그대로
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

    # bench_cache
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
def robust_mod():
    return pytest.importorskip(
        "components.backtest_robustness",
        reason="backtest_robustness 모듈 import 불가",
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


def _make_synthetic_recs(n_rows=500):
    """합성 recommend CSV — _run_backtest 입력용."""
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


# ════════════════════════════════════════════════════════════════
# A. _run_robustness_test — 27조합 생성 + 메타데이터 부착
# ════════════════════════════════════════════════════════════════
class TestRunRobustnessTest:

    def test_returns_27_combos(self, fake_env, robust_mod):
        """3 × 3 × 3 = 27조합 정확히 생성."""
        recs = _make_synthetic_recs()
        out = robust_mod._run_robustness_test(recs, "balanced")
        assert "combos" in out
        assert len(out["combos"]) == 27, (
            f"27조합 기대, 실제 {len(out['combos'])}개"
        )

    def test_has_base_combo(self, fake_env, robust_mod):
        """Δ=0/0/0 base 조합 정확히 1개 존재."""
        recs = _make_synthetic_recs()
        out = robust_mod._run_robustness_test(recs, "balanced")
        base_combos = [
            c for c in out["combos"]
            if c["delta_min_score"] == 0
            and c["delta_top_k"] == 0
            and c["delta_hold_days"] == 0
        ]
        assert len(base_combos) == 1, (
            f"base 조합 1개 기대, 실제 {len(base_combos)}개"
        )

    def test_each_combo_has_cfg_and_result(self, fake_env, robust_mod):
        """각 조합에 cfg + result 키 존재."""
        recs = _make_synthetic_recs()
        out = robust_mod._run_robustness_test(recs, "balanced")
        for c in out["combos"]:
            assert "cfg" in c
            assert "result" in c
            assert "delta_min_score" in c
            assert "delta_top_k" in c
            assert "delta_hold_days" in c
            # cfg 6개 필수 키
            assert set(c["cfg"].keys()) >= {
                "min_score", "top_k", "hold_days",
                "target_pct", "stop_pct", "cost_pct",
            }

    def test_each_result_has_robustness_meta(
        self, fake_env, robust_mod
    ):
        """성공한 result에 alpha/anomaly_flags/tp_saturation 첨부."""
        recs = _make_synthetic_recs()
        out = robust_mod._run_robustness_test(recs, "balanced")
        valid = [c for c in out["combos"] if "error" not in c["result"]]
        assert valid, "유효 결과 0개 — 합성 데이터 부족"
        for c in valid:
            r = c["result"]
            assert "anomaly_flags" in r
            assert isinstance(r["anomaly_flags"], list)
            assert "tp_saturation" in r
            assert "tp_threshold" in r
            assert r["tp_threshold"] in (60, 70, 80)
            assert "alpha" in r
            assert "alpha_mode" in r

    def test_base_cfg_matches_preset(self, fake_env, robust_mod):
        """base_cfg가 균형형 프리셋 값 그대로."""
        recs = _make_synthetic_recs()
        out = robust_mod._run_robustness_test(recs, "balanced")
        assert out["base_preset"] == "balanced"
        assert out["base_cfg"]["min_score"] == 70  # PRESETS["balanced"]
        assert out["base_cfg"]["top_k"] == 10
        assert out["base_cfg"]["hold_days"] == 10

    def test_combos_param_deltas_cover_range(self, fake_env, robust_mod):
        """3개 파라미터 ±5 delta 모두 등장."""
        recs = _make_synthetic_recs()
        out = robust_mod._run_robustness_test(recs, "balanced")
        deltas_ms = set(c["delta_min_score"] for c in out["combos"])
        deltas_tk = set(c["delta_top_k"] for c in out["combos"])
        deltas_hd = set(c["delta_hold_days"] for c in out["combos"])
        assert deltas_ms == {-5, 0, +5}
        assert deltas_tk == {-5, 0, +5}
        assert deltas_hd == {-5, 0, +5}

    def test_invalid_base_preset_falls_back_to_balanced(
        self, fake_env, robust_mod
    ):
        """알 수 없는 preset key 입력 시 balanced로 폴백."""
        recs = _make_synthetic_recs()
        out = robust_mod._run_robustness_test(recs, "unknown_preset_xyz")
        # base_preset은 fallback해서 balanced가 됨
        assert out["base_preset"] == "balanced"


# ════════════════════════════════════════════════════════════════
# B. _summarize_robustness — 통계 집계 정확성
# ════════════════════════════════════════════════════════════════
class TestSummarize:

    def _mock_combos(self, results):
        """results list → combos 구조."""
        cfg_base = {"min_score": 70, "top_k": 10, "hold_days": 10,
                    "target_pct": 10, "stop_pct": 5, "cost_pct": 0.4}
        return [
            {
                "delta_min_score": 0,
                "delta_top_k": 0,
                "delta_hold_days": 0,
                "cfg": cfg_base,
                "result": r,
            }
            for r in results
        ]

    def test_all_positive_returns(self, fake_env, robust_mod):
        """모든 조합 수익 양수 → positive_ret_ratio=1.0."""
        combos = self._mock_combos([
            {"total_return": 10.0, "mdd": -5.0, "anomaly_flags": []},
            {"total_return": 20.0, "mdd": -8.0, "anomaly_flags": []},
            {"total_return": 5.0, "mdd": -3.0, "anomaly_flags": []},
        ])
        s = robust_mod._summarize_robustness(combos)
        assert s["n_total"] == 3
        assert s["n_success"] == 3
        assert s["n_positive_ret"] == 3
        assert s["positive_ret_ratio"] == 1.0
        assert s["worst_return"] == 5.0
        assert s["best_return"] == 20.0

    def test_anomaly_ratio_counted(self, fake_env, robust_mod):
        """anomaly_flags 있는 조합 정확히 카운트."""
        combos = self._mock_combos([
            {"total_return": 10.0, "mdd": -5.0, "anomaly_flags": []},
            {"total_return": 700.0, "mdd": -3.0,
             "anomaly_flags": ["수익률 비정상"]},
            {"total_return": 800.0, "mdd": -2.0,
             "anomaly_flags": ["수익률 비정상"]},
        ])
        s = robust_mod._summarize_robustness(combos)
        assert s["n_anomaly"] == 2
        # 0.666... 근방
        assert abs(s["anomaly_ratio"] - 2/3) < 0.01

    def test_mdd_within_15_counted(self, fake_env, robust_mod):
        """MDD ≥ -15% 비율 정확히."""
        combos = self._mock_combos([
            {"total_return": 10.0, "mdd": -5.0, "anomaly_flags": []},
            {"total_return": 10.0, "mdd": -14.0, "anomaly_flags": []},
            {"total_return": 10.0, "mdd": -20.0, "anomaly_flags": []},
        ])
        s = robust_mod._summarize_robustness(combos)
        assert s["n_mdd_within_15"] == 2  # -5, -14

    def test_error_combos_excluded_from_stats(self, fake_env, robust_mod):
        """error 조합은 통계에서 제외."""
        combos = self._mock_combos([
            {"total_return": 10.0, "mdd": -5.0, "anomaly_flags": []},
            {"error": "데이터 부족"},
            {"total_return": 20.0, "mdd": -8.0, "anomaly_flags": []},
        ])
        s = robust_mod._summarize_robustness(combos)
        assert s["n_total"] == 3
        assert s["n_success"] == 2
        assert s["n_positive_ret"] == 2

    def test_all_errors_returns_zeros(self, fake_env, robust_mod):
        """모든 조합 error여도 ZeroDivision 안 남."""
        combos = self._mock_combos([
            {"error": "이것도"},
            {"error": "저것도"},
        ])
        s = robust_mod._summarize_robustness(combos)
        assert s["n_success"] == 0
        assert s["positive_ret_ratio"] == 0.0
        assert s["anomaly_ratio"] == 0.0


# ════════════════════════════════════════════════════════════════
# C. _derive_robustness_verdict — 🟢/🟡/🔴 임계값
# ════════════════════════════════════════════════════════════════
class TestRobustnessVerdict:

    def _build_summary(self, **kwargs):
        """summary dict 생성 helper.

        [v3.9.17b] alpha_coverage_ratio / n_alpha_calculated 키 추가.
        """
        defaults = {
            "n_total": 27, "n_success": 27,
            "n_positive_ret": 20, "n_positive_alpha": 15,
            "n_alpha_calculated": 25,
            "n_anomaly": 3, "n_mdd_within_15": 18,
            "positive_ret_ratio": 20/27,
            "positive_alpha_ratio": 15/27,
            "alpha_coverage_ratio": 25/27,
            "anomaly_ratio": 3/27,
            "mdd_within_15_ratio": 18/27,
            "worst_return": -2.0, "avg_return": 8.5, "best_return": 25.0,
        }
        defaults.update(kwargs)
        return defaults

    def test_green_when_all_strict_met(self, fake_env, robust_mod):
        """positive_ret 85%, anomaly 10%, mdd_within 70%, alpha 60% → 🟢."""
        s = self._build_summary(
            n_positive_ret=23, positive_ret_ratio=23/27,
            n_anomaly=3, anomaly_ratio=3/27,
            n_mdd_within_15=19, mdd_within_15_ratio=19/27,
            n_positive_alpha=16, positive_alpha_ratio=16/27,
            n_alpha_calculated=25, alpha_coverage_ratio=25/27,
        )
        v = robust_mod._derive_robustness_verdict(s)
        assert v["icon"] == "🟢", f"expected 🟢, got {v['icon']} {v['title']}"
        assert v["level"] == "green"
        assert "강건" in v["title"]

    def test_red_when_low_positive_ratio(self, fake_env, robust_mod):
        """positive_ret 40% → 🔴 (50% 미만)."""
        s = self._build_summary(
            n_positive_ret=11, positive_ret_ratio=11/27,
            n_anomaly=2, anomaly_ratio=2/27,
        )
        v = robust_mod._derive_robustness_verdict(s)
        assert v["icon"] == "🔴"
        assert v["level"] == "red"

    def test_red_when_high_anomaly_ratio(self, fake_env, robust_mod):
        """anomaly 55% → 🔴 (수익률 양호해도 강제 🔴)."""
        s = self._build_summary(
            n_positive_ret=24, positive_ret_ratio=24/27,
            n_anomaly=15, anomaly_ratio=15/27,
        )
        v = robust_mod._derive_robustness_verdict(s)
        assert v["icon"] == "🔴"
        assert v["level"] == "red"

    def test_yellow_when_partial(self, fake_env, robust_mod):
        """positive_ret 60%, anomaly 25%, mdd_within 40% → 🟡."""
        s = self._build_summary(
            n_positive_ret=16, positive_ret_ratio=16/27,
            n_anomaly=7, anomaly_ratio=7/27,
            n_mdd_within_15=11, mdd_within_15_ratio=11/27,
        )
        v = robust_mod._derive_robustness_verdict(s)
        assert v["icon"] == "🟡"
        assert v["level"] == "yellow"

    def test_unknown_when_all_errors(self, fake_env, robust_mod):
        """n_success=0 → ⚪ 데이터 부족."""
        s = self._build_summary(
            n_success=0, n_positive_ret=0, positive_ret_ratio=0,
            n_anomaly=0, anomaly_ratio=0,
            n_positive_alpha=0, positive_alpha_ratio=0,
            n_alpha_calculated=0, alpha_coverage_ratio=0,
        )
        v = robust_mod._derive_robustness_verdict(s)
        assert v["icon"] == "⚪"
        assert v["level"] == "unknown"

    # ────────────────────────────────────────────────────────────
    # [v3.9.17b] 평가 지적 1+2 회귀 가드 — alpha 평가
    # ────────────────────────────────────────────────────────────
    def test_alpha_low_blocks_green(self, fake_env, robust_mod):
        """[평가 1] 기본 3기준 모두 통과해도 alpha 양수 비율 낮으면 🟡.

        시나리오: positive_ret 85%, anomaly 10%, mdd_within 70% (3기준 통과)
                  but alpha 양수 비율 30% (50% 미만) + coverage 90% (평가 가능)
        → 🟡 (시장 대비 열위 조합 많음)
        """
        s = self._build_summary(
            n_positive_ret=23, positive_ret_ratio=23/27,
            n_anomaly=3, anomaly_ratio=3/27,
            n_mdd_within_15=19, mdd_within_15_ratio=19/27,
            n_positive_alpha=8, positive_alpha_ratio=8/27,  # 30% — 50% 미만
            n_alpha_calculated=25, alpha_coverage_ratio=25/27,  # 92% — coverage OK
        )
        v = robust_mod._derive_robustness_verdict(s)
        assert v["icon"] != "🟢", (
            f"alpha 양수 30%인데 🟢 부여됨: {v['title']}"
        )
        assert v["icon"] == "🟡"
        # reasons에 alpha 사유 포함 확인
        reasons_text = " ".join(v.get("reasons", []))
        assert "alpha" in reasons_text.lower()

    def test_low_alpha_coverage_returns_yellow_candidate_not_green(
        self, fake_env, robust_mod
    ):
        """[v3.9.17c] 평가 절충안: coverage 부족 시 🟡 강건함 후보 (이전 🟢).

        시나리오: 기본 3기준 OK, alpha coverage 30% (70% 미만)
        - 이전 v3.9.17b: 🟢 강건함 + body에 "alpha 평가 보류" 명시
        - v3.9.17c (절충안): 🟡 강건함 후보 · alpha 평가 보류
          → 제목 자체에서 alpha 검증 부족 즉시 인지
        """
        s = self._build_summary(
            n_positive_ret=23, positive_ret_ratio=23/27,
            n_anomaly=3, anomaly_ratio=3/27,
            n_mdd_within_15=19, mdd_within_15_ratio=19/27,
            n_positive_alpha=3, positive_alpha_ratio=3/27,
            n_alpha_calculated=8, alpha_coverage_ratio=8/27,  # coverage 30%
        )
        v = robust_mod._derive_robustness_verdict(s)
        # 🟡 강건함 후보 — 🟢가 아님
        assert v["icon"] == "🟡", (
            f"coverage 부족 시 🟡 기대, 실제 {v['icon']} {v['title']}"
        )
        assert v["level"] == "yellow_candidate"
        assert "강건함 후보" in v["title"]
        assert "alpha 평가 보류" in v["title"]
        # body에 coverage 명시
        assert "coverage" in v["body"].lower()

    def test_alpha_coverage_denominator_is_n_success(
        self, fake_env, robust_mod
    ):
        """[평가 v3.9.17 지적 2] alpha_coverage_ratio = n_alpha_calculated / n_success.

        시나리오: 1개만 alpha 산출, 그 1개가 양수
        - 이전 v3.9.17: positive_alpha_ratio = 1/1 = 100% (잘못)
        - v3.9.17b: positive_alpha_ratio = 1/27 = 3.7%, coverage = 1/27 = 3.7% → 🟢
        - v3.9.17c: coverage 3.7% < 70% → 🟡 강건함 후보 (절충안)
        """
        s = self._build_summary(
            n_positive_ret=23, positive_ret_ratio=23/27,
            n_anomaly=3, anomaly_ratio=3/27,
            n_mdd_within_15=19, mdd_within_15_ratio=19/27,
            n_positive_alpha=1, positive_alpha_ratio=1/27,
            n_alpha_calculated=1, alpha_coverage_ratio=1/27,
        )
        v = robust_mod._derive_robustness_verdict(s)
        # [v3.9.17c] coverage 부족 → 🟡 강건함 후보 (v3.9.17b의 🟢에서 변경)
        assert v["icon"] == "🟡"
        assert v["level"] == "yellow_candidate"
        assert "강건함 후보" in v["title"]
        # body에 coverage 명시
        assert "coverage" in v["body"].lower() or "평가 보류" in v["body"]

    def test_alpha_coverage_full_with_positive_alpha_returns_green(
        self, fake_env, robust_mod
    ):
        """[v3.9.17c] coverage 충분 + alpha 양수 비율 통과 시 🟢 유지.

        평가 기준 2: "alpha coverage 충분 + alpha 양수 비율 통과
                    → 기존처럼 🟢 강건함 유지"
        """
        s = self._build_summary(
            n_positive_ret=23, positive_ret_ratio=23/27,
            n_anomaly=3, anomaly_ratio=3/27,
            n_mdd_within_15=19, mdd_within_15_ratio=19/27,
            n_positive_alpha=18, positive_alpha_ratio=18/27,  # 67%
            n_alpha_calculated=27, alpha_coverage_ratio=27/27,  # 100%
        )
        v = robust_mod._derive_robustness_verdict(s)
        assert v["icon"] == "🟢", (
            f"coverage 100% + alpha 양수 67%인데 🟢 아님: {v['title']}"
        )
        assert v["level"] == "green"
        assert v["title"] == "강건함"
        # 🟢 강건함은 제목에 "후보" / "평가 보류" 없어야 함
        assert "후보" not in v["title"]
        assert "평가 보류" not in v["title"]

    def test_high_coverage_with_majority_negative_alpha_yellow(
        self, fake_env, robust_mod
    ):
        """[평가 1 회귀 핵심] coverage 100% + alpha 대부분 음수.

        시나리오: 27조합 모두 alpha 산출 (coverage 100%),
                  그 중 5개만 양수 (positive_alpha 18%, 50% 미만)
                  하지만 절대수익 양수 90%
        → 🟡 (시장 대비 열위 조합 많음) — 이전 v3.9.17은 이걸 🟢로 잘못 판정
        """
        s = self._build_summary(
            n_positive_ret=24, positive_ret_ratio=24/27,
            n_anomaly=2, anomaly_ratio=2/27,
            n_mdd_within_15=20, mdd_within_15_ratio=20/27,
            n_positive_alpha=5, positive_alpha_ratio=5/27,  # 18% — 시장 열위
            n_alpha_calculated=27, alpha_coverage_ratio=27/27,  # 100%
        )
        v = robust_mod._derive_robustness_verdict(s)
        assert v["icon"] != "🟢"
        assert v["icon"] == "🟡"


# ════════════════════════════════════════════════════════════════
# D. _render_robustness_table — 진짜 UI 호출 + 캡처
# ════════════════════════════════════════════════════════════════
class TestRenderRobustnessTable:

    def _build_data(self, verdict_level="green"):
        """robustness_data 모킹 — 정상 케이스."""
        cfg = {"min_score": 70, "top_k": 10, "hold_days": 10,
               "target_pct": 10, "stop_pct": 5, "cost_pct": 0.4}

        combos = []
        for d_ms in [-5, 0, +5]:
            for d_tk in [-5, 0, +5]:
                for d_hd in [-5, 0, +5]:
                    combo_cfg = dict(cfg)
                    combo_cfg["min_score"] = max(50, cfg["min_score"] + d_ms)
                    combo_cfg["top_k"] = max(1, cfg["top_k"] + d_tk)
                    combo_cfg["hold_days"] = max(1, cfg["hold_days"] + d_hd)
                    combos.append({
                        "delta_min_score": d_ms,
                        "delta_top_k": d_tk,
                        "delta_hold_days": d_hd,
                        "cfg": combo_cfg,
                        "result": {
                            "total_return": 10.0 + d_ms,
                            "mdd": -8.0,
                            "win_rate": 58.0,
                            "alpha": 2.0,
                            "alpha_mode": "simple",
                            "anomaly_flags": [],
                            "tp_saturation": 55.0,
                            "tp_threshold": 60,
                        },
                    })

        if verdict_level == "green":
            verdict = {
                "icon": "🟢", "level": "green", "title": "강건함",
                "color_class": "text-emerald-400",
                "body": "27조합 중 양호.",
                "reasons": ["주변 조합 일관 양호"],
            }
        elif verdict_level == "red":
            verdict = {
                "icon": "🔴", "level": "red", "title": "과최적화 의심",
                "color_class": "text-red-400",
                "body": "수익 양수 비율 낮음.",
                "reasons": ["수익 양수 비율 낮음"],
            }
        else:
            verdict = {
                "icon": "🟡", "level": "yellow", "title": "조건부",
                "color_class": "text-yellow-400",
                "body": "일부 조합만 양호.",
                "reasons": ["수익 양수 비율 60%"],
            }

        return {
            "base_preset": "balanced",
            "base_cfg": cfg,
            "combos": combos,
            "summary": {
                "n_total": 27, "n_success": 27,
                "n_positive_ret": 20, "n_positive_alpha": 15,
                "n_anomaly": 3, "n_mdd_within_15": 18,
                "positive_ret_ratio": 20/27,
                "positive_alpha_ratio": 15/27,
                "anomaly_ratio": 3/27,
                "mdd_within_15_ratio": 18/27,
                "worst_return": -2.0, "avg_return": 8.5, "best_return": 25.0,
            },
            "verdict": verdict,
        }

    def test_renders_verdict_card(self, fake_env, robust_mod):
        """판정 카드 icon + title 표시."""
        data = self._build_data("green")
        robust_mod._render_robustness_table(data)
        text = _captured_text()
        assert "🟢" in text
        assert "강건함" in text

    def test_renders_red_verdict(self, fake_env, robust_mod):
        """🔴 과최적화 의심 표시."""
        data = self._build_data("red")
        robust_mod._render_robustness_table(data)
        text = _captured_text()
        assert "🔴" in text
        assert "과최적화" in text

    def test_renders_summary_pills(self, fake_env, robust_mod):
        """요약 통계 pill 표시: 성공/수익 양수/anomaly/MDD."""
        data = self._build_data("green")
        robust_mod._render_robustness_table(data)
        text = _captured_text()
        assert "수익 양수" in text
        assert "anomaly" in text
        assert "MDD" in text

    def test_renders_base_marker(self, fake_env, robust_mod):
        """⭐ 기준 표시 (Δ=0/0/0 조합에)."""
        data = self._build_data("green")
        robust_mod._render_robustness_table(data)
        text = _captured_text()
        assert "⭐" in text or "기준" in text


# ════════════════════════════════════════════════════════════════
# E. baseline 무수정 + 파일 분리 import parity
# ════════════════════════════════════════════════════════════════
class TestNoRegression:

    def test_tab_backtest_reexports_calc_kospi_alpha(
        self, fake_env, tab_backtest_mod
    ):
        """[v3.9.17] 파일 분리 후 backward compat re-export 보장."""
        assert hasattr(tab_backtest_mod, "_calc_kospi_alpha")

    def test_tab_backtest_reexports_derive_verdict(
        self, fake_env, tab_backtest_mod
    ):
        """[v3.9.17] _derive_strategy_verdict re-export."""
        assert hasattr(tab_backtest_mod, "_derive_strategy_verdict")

    def test_tab_backtest_reexports_preset_compare(
        self, fake_env, tab_backtest_mod
    ):
        """[v3.9.17] _run_preset_comparison / _render_preset_comparison_table re-export."""
        assert hasattr(tab_backtest_mod, "_run_preset_comparison")
        assert hasattr(tab_backtest_mod, "_render_preset_comparison_table")

    def test_run_backtest_signature_unchanged(
        self, fake_env, tab_backtest_mod
    ):
        """_run_backtest 시그니처 baseline (분리 영향 받지 않음)."""
        import inspect
        sig = inspect.signature(tab_backtest_mod._run_backtest)
        params = list(sig.parameters.keys())
        assert params == [
            "all_recs", "min_score", "hold_days",
            "stop_pct", "target_pct", "top_k", "cost_pct"
        ]

    def test_presets_dict_unchanged(self, fake_env, tab_backtest_mod):
        """PRESETS 4개 키 그대로 + balanced 기본값 baseline."""
        P = tab_backtest_mod.PRESETS
        assert set(P.keys()) == {
            "conservative", "balanced", "aggressive", "scalping"
        }
        assert P["balanced"]["min_score"] == 70
        assert P["balanced"]["hold_days"] == 10

    def test_derive_verdict_parity_after_split(
        self, fake_env, tab_backtest_mod
    ):
        """v3.9.16b SSOT — tab_backtest의 _derive_strategy_verdict가
        backtest_verdict 모듈 함수와 동일 객체.

        파일 분리 후에도 두 import 경로가 같은 함수를 가리켜야 함.
        """
        from components.backtest_verdict import _derive_strategy_verdict as v1
        v2 = tab_backtest_mod._derive_strategy_verdict
        assert v1 is v2, (
            "re-export가 같은 함수 객체를 가리키지 않음 — "
            "파일 분리 시 SSOT 깨짐"
        )

    # ──────────────────────────────────────────────────────────────
    # [v3.9.17b] 평가 지적 4 회귀 가드 — services / components 분리
    # ──────────────────────────────────────────────────────────────
    def test_services_robustness_no_nicegui_import(self, fake_env):
        """services.backtest_robustness 모듈은 nicegui import 0.

        UI 비의존 보장. 향후 CLI/배치 작업에서 import해서 사용 가능.
        """
        import importlib
        # 깨끗한 import — nicegui mock 안 한 상태로 services만 import
        # (fake_env가 nicegui mock 이미 함, 그래도 services는 nicegui 안 씀)
        # 모듈 source 읽어서 'from nicegui' 또는 'import nicegui'가 없는지 검증
        from pathlib import Path
        import inspect
        import services.backtest_robustness as svc_robust
        src_path = inspect.getfile(svc_robust)
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        # 주석 안의 nicegui 언급은 허용, 실제 import 문만 검증
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("from nicegui"), (
                f"services.backtest_robustness에 nicegui import 발견: {line}"
            )
            assert not stripped.startswith("import nicegui"), (
                f"services.backtest_robustness에 nicegui import 발견: {line}"
            )

    def test_robustness_logic_reexport_parity(self, fake_env, robust_mod):
        """components.backtest_robustness는 services에서 re-export.

        같은 함수 객체여야 함 (backward compat + SSOT).
        """
        from services.backtest_robustness import (
            run_robustness_test as svc_run,
            summarize_robustness as svc_sum,
            derive_robustness_verdict as svc_verdict,
        )
        # components의 _ prefix re-export가 services 함수와 동일
        assert robust_mod._run_robustness_test is svc_run
        assert robust_mod._summarize_robustness is svc_sum
        assert robust_mod._derive_robustness_verdict is svc_verdict


