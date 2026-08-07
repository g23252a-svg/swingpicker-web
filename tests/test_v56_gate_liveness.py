# -*- coding: utf-8 -*-
"""
test_v56_gate_liveness.py — "선언된 게이트가 실제로 발동하는가"를 상시 검사.

배경 (2026-08 mutation 감사)
  v55.4에서 feature 계약 게이트가 **6주간 죽어 있었던 것**이 우연히 발견됐다.
  그래서 선언된 게이트 16개에 각각 "그 게이트가 막는다고 주장하는 위반"을
  주입해 봤다. 결과: 15개는 살아있었고 **1개는 죽어 있었다** —
  check_alpha_schema_integrity가 components/tab_backtest.py의
  _calc_kospi_alpha를 봤지만 그 함수는 components/backtest_verdict.py로
  옮겨진 상태였고, 게이트는 "함수 없음 = 미사용 환경"이라며 조용히 통과했다.

이 파일이 하는 일
  같은 mutation을 **테스트로 상설화**한다. 게이트가 다시 죽으면 CI가 깨진다.
  실제 저장소 파일은 건드리지 않는다 — check_contract_gate.PROJECT_ROOT를
  임시 트리로 바꿔치기해 위반을 심는다(정적 게이트), 또는 대상 심볼을
  가상 트리에서 제거한다(메타 게이트).

보증
  1. 정적 게이트 6종: 깨끗한 트리 → 0건, 위반 트리 → ≥1건.
  2. 게이트 대상 소실(파일/심볼/함수 이동)은 통과가 아니라 실패다.
  3. 행동 계약 게이트(10~13)는 의존성이 있는 환경에서 '스킵 경로로
     빠지지 않고' 실제로 실행된다.
  4. monotonicity 게이트: FAIL 전파 · 스테일 리포트 차단 · 공허한 PASS 표시.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import check_contract_gate as gate  # noqa: E402
from scripts.check_monotonicity_gate import evaluate as mono_evaluate  # noqa: E402


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """게이트가 바라보는 트리를 임시 디렉터리로 바꾼다.

    [v56] 이 fixture 자체가 처음엔 세션을 오염시켰다: 게이트가
    `sys.path.insert(0, PROJECT_ROOT)`를 하므로 임시 트리가 sys.path에 남아,
    뒤에 도는 테스트의 `import ml_engine`이 여기 심은 **합성 ml_engine**을
    집어왔다(실행 순서에 따라 통과/실패가 갈림 — v55.4와 같은 유형).
    게이트 쪽은 _ensure_import_path로 고쳤고, 여기서도 이중으로 되돌린다.
    """
    monkeypatch.setattr(gate, "PROJECT_ROOT", str(tmp_path))
    gate._TREE_CACHE.clear()
    yield tmp_path
    gate._TREE_CACHE.clear()
    tmp = str(tmp_path)
    sys.path[:] = [p for p in sys.path if p != tmp]
    for name, mod in list(sys.modules.items()):
        src = getattr(mod, "__file__", None) or ""
        if src.startswith(tmp):
            del sys.modules[name]


def _w(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# 최소한의 "정상" 트리 — 각 게이트가 통과해야 하는 상태
def _clean_tree(root: Path):
    from feature_contract import FEATURE_CONTRACT
    cols = list(FEATURE_CONTRACT.columns)
    _w(root, "ml_engine.py", f"FEATURE_COLS = {cols!r}\n")
    for f in ("validation.py", "stop_logic.py", "trade_plan.py"):
        _w(root, f, "from collector_config import PolicyConfig\n")
    _w(root, "components/backtest_verdict.py",
       "def _calc_kospi_alpha(result, cfg):\n"
       "    return result.get('rec_date'), result.get('net_ret')\n")
    _w(root, "services/backtest_policy.py",
       "ANOMALY_TOTAL_RET_ABS = 300\nANOMALY_SHARPE_MAX = 5\nANOMALY_CAGR_MAX = 300\n"
       "ANOMALY_SHORT_DAYS_RET = 120\nANOMALY_SHORT_RET = 100\n"
       "ANOMALY_SHORT_DAYS_CAGR = 252\nTP_SAT_THRESH_LOW_TARGET = 80\n"
       "TP_SAT_THRESH_MID_TARGET = 70\nTP_SAT_THRESH_HIGH_TARGET = 60\n"
       "def tp_saturation_threshold():\n    return 0\n"
       "def detect_anomaly_flags():\n    return []\n")
    (root / "data").mkdir(exist_ok=True)
    _w(root, "data/bench_cache_latest.json",
       json.dumps({"KOSPI": {"1": 0, "3": 0, "5": 0, "10": 0, "20": 0, "60": 0}}))
    _w(root, "data/kospi_daily.csv",
       "date,close,ret_1d_%,ret_3d_%,ret_5d_%,ret_10d_%,ret_20d_%,ret_60d_%\n"
       "20260101,2500,0,0,0,0,0,0\n")
    return root


# ══════════════════════════════════════════════════════════════
#  1. 정적 게이트 — 깨끗하면 통과, 위반을 심으면 반드시 검출
# ══════════════════════════════════════════════════════════════
class TestStaticGatesFire:
    def test_clean_tree_passes_all_static_gates(self, fake_root):
        _clean_tree(fake_root)
        for fn in (gate.check_feature_contract, gate.check_stale_versions,
                   gate.check_policy_ssot, gate.check_duplicate_feature_defs,
                   gate.check_json_meta_versions, gate.check_alpha_schema_integrity,
                   gate.check_benchmark_cache_completeness,
                   gate.check_kospi_daily_csv_schema, gate.check_anomaly_thresholds):
            assert fn() == [], f"{fn.__name__}가 깨끗한 트리에서 오류를 냈다"

    def test_feature_contract_mismatch_detected(self, fake_root):
        _clean_tree(fake_root)
        _w(fake_root, "ml_engine.py", 'FEATURE_COLS = ["Log_Ret"]\n')
        assert gate.check_feature_contract(), "FEATURE_COLS 불일치를 놓쳤다"

    def test_stale_version_detected(self, fake_root):
        _clean_tree(fake_root)
        _w(fake_root, "somewhere.py", 'X = "v20.7"\n')
        assert gate.check_stale_versions(), "구버전 문자열을 놓쳤다"

    def test_policy_ssot_violation_detected(self, fake_root):
        _clean_tree(fake_root)
        _w(fake_root, "validation.py", "MIN_TURNOVER = 30\n")   # SSOT 미참조
        assert gate.check_policy_ssot(), "PolicyConfig 미참조를 놓쳤다"

    def test_duplicate_feature_cols_detected(self, fake_root):
        _clean_tree(fake_root)
        _w(fake_root, "rogue.py", 'FEATURE_COLS = [\n    "A",\n]\n')
        assert gate.check_duplicate_feature_defs(), "FEATURE_COLS 중복 정의를 놓쳤다"

    def test_anomaly_threshold_relaxation_detected(self, fake_root):
        _clean_tree(fake_root)
        p = fake_root / "services/backtest_policy.py"
        p.write_text(p.read_text(encoding="utf-8")
                     .replace("ANOMALY_SHARPE_MAX = 5", "ANOMALY_SHARPE_MAX = 100"),
                     encoding="utf-8")
        assert gate.check_anomaly_thresholds(), "임계값 완화를 놓쳤다"

    def test_bench_cache_missing_key_detected(self, fake_root):
        _clean_tree(fake_root)
        _w(fake_root, "data/bench_cache_latest.json",
           json.dumps({"KOSPI": {"1": 0, "3": 0, "5": 0, "10": 0, "20": 0}}))
        assert gate.check_benchmark_cache_completeness(), "60 키 누락을 놓쳤다"

    def test_kospi_csv_missing_column_detected(self, fake_root):
        _clean_tree(fake_root)
        _w(fake_root, "data/kospi_daily.csv",
           "date,close,ret_1d_%,ret_3d_%,ret_5d_%,ret_10d_%,ret_20d_%\n20260101,2500,0,0,0,0,0\n")
        assert gate.check_kospi_daily_csv_schema(), "필수 컬럼 누락을 놓쳤다"


# ══════════════════════════════════════════════════════════════
#  2. 실제로 죽어 있던 게이트 — 회귀 봉쇄
# ══════════════════════════════════════════════════════════════
class TestAlphaSchemaGateWasDead:
    def test_forbidden_literal_detected_wherever_function_lives(self, fake_root):
        """v56 실측: 함수가 다른 파일로 옮겨지자 게이트가 침묵했다.
        어느 파일에 있든 금지 리터럴은 잡혀야 한다."""
        _clean_tree(fake_root)
        _w(fake_root, "components/somewhere_else.py",
           'def _calc_kospi_alpha(result, cfg):\n    return result["net_pct"]\n')
        errs = gate.check_alpha_schema_integrity()
        assert errs, "다른 파일의 금지 리터럴을 놓쳤다 — 게이트가 다시 죽었다"
        assert "net_pct" in errs[0]

    def test_vanished_target_is_failure_not_silent_pass(self, fake_root):
        """대상 함수가 사라지면 통과가 아니라 실패여야 한다."""
        _clean_tree(fake_root)
        (fake_root / "components/backtest_verdict.py").unlink()
        gate._TREE_CACHE.clear()
        errs = gate.check_alpha_schema_integrity()
        assert errs, "대상이 사라졌는데 조용히 통과했다"
        assert "소실" in errs[0]

    def test_real_repo_gate_is_alive(self):
        """실제 저장소에서도 대상을 찾을 수 있어야 한다(트리 탐색 경로 확인)."""
        gate._TREE_CACHE.clear()
        assert gate.find_function_defs("_calc_kospi_alpha"), \
            "_calc_kospi_alpha 정의를 저장소에서 찾지 못했다"
        assert gate.check_alpha_schema_integrity() == []


# ══════════════════════════════════════════════════════════════
#  3. 메타 게이트 — 게이트 대상 소실 탐지
# ══════════════════════════════════════════════════════════════
class TestGateTargets:
    def test_real_repo_targets_all_resolve(self):
        gate._TREE_CACHE.clear()
        assert gate.check_gate_targets() == [], "게이트 대상 중 사라진 것이 있다"

    def test_missing_file_detected(self, fake_root):
        _clean_tree(fake_root)
        (fake_root / "validation.py").unlink()
        gate._TREE_CACHE.clear()
        errs = gate.check_gate_targets()
        assert any("validation.py" in e for e in errs)

    def test_renamed_symbol_detected(self, fake_root, monkeypatch):
        _clean_tree(fake_root)
        _w(fake_root, "momentum_lane.py",
           "MOMENTUM_LANE_COLS = []\n"
           "def apply_momentum_lane():\n    pass\n"
           "def compute_market_risk_off_RENAMED():\n    pass\n")
        monkeypatch.setattr(gate, "GATE_TARGETS", [
            ("11 Momentum 계약", "momentum_lane.py",
             ["MOMENTUM_LANE_COLS", "apply_momentum_lane", "compute_market_risk_off"]),
        ])
        gate._TREE_CACHE.clear()
        errs = gate.check_gate_targets()
        assert errs and "compute_market_risk_off" in errs[0]

    def test_gate_does_not_pollute_sys_path(self, fake_root):
        """게이트가 임시 트리를 sys.path에 영구히 남기면, 뒤에 도는 테스트가
        합성 모듈을 진짜로 착각한다 — 실제로 터졌던 순서 의존 실패."""
        _clean_tree(fake_root)
        gate.check_feature_contract()
        assert str(fake_root) not in sys.path, "임시 트리가 sys.path에 남았다"

    def test_real_ml_engine_still_importable_after_fake_root(self, fake_root):
        """fake_root 사용 후에도 `import ml_engine`은 진짜 모듈이어야 한다."""
        _clean_tree(fake_root)
        gate.check_feature_contract()
        gate.check_alpha_schema_integrity()
        import importlib
        mod = importlib.import_module("ml_engine")
        assert not str(getattr(mod, "__file__", "")).startswith(str(fake_root))
        assert hasattr(mod, "TORCH_AVAILABLE"), "합성 ml_engine을 집어왔다"

    def test_registry_covers_every_behavior_gate(self):
        """행동 계약 게이트(10~13)의 모듈이 레지스트리에 다 들어있는지."""
        files = {rel for _, rel, _ in gate.GATE_TARGETS if rel}
        for expected in ("guard_system.py", "momentum_lane.py",
                         "stop_override.py", "data_integrity.py"):
            assert expected in files, f"{expected}가 GATE_TARGETS에 없다"


# ══════════════════════════════════════════════════════════════
#  4. 행동 계약 게이트가 '의존성 스킵'으로 빠지지 않는지
# ══════════════════════════════════════════════════════════════
class TestBehaviorGatesActuallyRun:
    @pytest.mark.parametrize("fn_name", [
        "check_guard_v23_contract", "check_momentum_lane_contract",
        "check_stop_override_contract", "check_data_integrity_contract",
    ])
    def test_gate_runs_and_passes_with_deps_present(self, fn_name, capsys):
        """pandas가 있는 환경에서 '미설치 환경 — 건너뜀'이 찍히면 안 된다.
        (그 메시지가 나오면 게이트는 아무것도 검증하지 않은 것이다.)"""
        pytest.importorskip("pandas")
        gate._TREE_CACHE.clear()
        errs = getattr(gate, fn_name)()
        out = capsys.readouterr().out
        assert "건너뜀" not in out, f"{fn_name}가 스킵 경로로 빠졌다: {out.strip()}"
        assert errs == [], f"{fn_name} 위반: {errs}"


# ══════════════════════════════════════════════════════════════
#  5. monotonicity 게이트 (인라인 YAML → 스크립트로 이관된 부분)
# ══════════════════════════════════════════════════════════════
def _report(**kw):
    base = dict(asof_ymd="20260807", ci_hard=[], ci_soft=[],
                ci_hard_all_pass=True, ci_soft_any_warn=False)
    base.update(kw)
    return base


class TestMonotonicityGate:
    def test_hard_fail_propagates(self):
        ok, _ = mono_evaluate(_report(
            ci_hard=[{"gate": "g", "status": "FAIL", "subject_n": 3}],
            ci_hard_all_pass=False), event="push", expect_ymd="20260807")
        assert ok is False

    def test_stale_report_blocks_on_push(self):
        """실제 구멍: daily_briefing 실패 시 커밋된 과거 리포트가 통과했다."""
        ok, lines = mono_evaluate(_report(asof_ymd="20260701"),
                                  event="push", expect_ymd="20260807")
        assert ok is False
        assert any("스테일" in l for l in lines)

    def test_stale_report_warns_on_pr(self):
        ok, lines = mono_evaluate(_report(asof_ymd="20260701"),
                                  event="pull_request", expect_ymd="20260807")
        assert ok is True, "PR에서는 통과시키는 기존 정책 유지"
        assert any("스테일" in l for l in lines)

    def test_fresh_report_passes(self):
        ok, _ = mono_evaluate(_report(), event="push", expect_ymd="20260807")
        assert ok is True

    def test_vacuous_pass_is_labeled(self):
        """대상 0건 통과를 '검증된 통과'로 표시하면 안 된다."""
        ok, lines = mono_evaluate(_report(ci_hard=[
            {"gate": "a", "status": "PASS", "subject_n": 0},
            {"gate": "b", "status": "SKIP", "subject_n": 0},
        ]), event="push", expect_ymd="20260807")
        assert ok is True
        text = "\n".join(lines)
        assert "공허한 통과" in text
        assert "검증된 하드게이트: 0/2" in text
        assert "검사 대상 없음" in text

    def test_verified_pass_is_counted(self):
        _, lines = mono_evaluate(_report(ci_hard=[
            {"gate": "a", "status": "PASS", "subject_n": 5},
        ]), event="push", expect_ymd="20260807")
        text = "\n".join(lines)
        assert "검증된 하드게이트: 1/1" in text
        assert "대상 5건 검증" in text

    def test_real_report_has_subject_n(self):
        """daily_briefing이 subject_n을 실제로 기록하는지 (없으면 공허 판정 불가)."""
        p = ROOT / "data" / "monotonicity_report_latest.json"
        if not p.exists():
            pytest.skip("리포트 없음 — collector 미실행 환경")
        r = json.loads(p.read_text(encoding="utf-8"))
        hard = r.get("ci_hard", [])
        if not hard:
            pytest.skip("ci_hard 비어 있음")
        assert all("subject_n" in g for g in hard), \
            "하드 게이트에 subject_n이 없다 — 공허한 PASS를 구분할 수 없다"
