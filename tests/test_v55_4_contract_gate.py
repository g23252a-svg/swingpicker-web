# -*- coding: utf-8 -*-
"""
test_v55_4_contract_gate.py — v55.4 "검사하지 않던 검사" 봉쇄.

배경 (2026-08 실측)
  feature_contract ↔ ml_engine.FEATURE_COLS 동기화는 하드 게이트로 선언돼
  있었지만, 실제로는 **CI에서 한 번도 비교되지 않았다**:
    · check_contract_gate.check_feature_contract() — `from ml_engine import
      FEATURE_COLS`가 torch 부재로 ImportError → 경고만 찍고 통과
    · test_policy_consistency의 계약 테스트 2개 — 같은 이유로 항상 pytest.skip
  그 조용한 스킵 가드가 **가짜 모듈까지 통과**시켰다. tests/test_fill_aware_v248
  이 심는 스텁(FEATURE_COLS=[])이 sys.modules에 남으면, 뒤에 도는 계약 테스트는
  ImportError를 못 보므로 스킵하지 않고 빈 목록과 비교해 실패했다
  (tests/ + 루트 테스트를 한 프로세스로 돌릴 때만 재현).

보증
  1. FEATURE_COLS를 torch 없이 정적(AST)으로 읽고, 항상 실제로 대조한다.
  2. 불일치(개수/원소/순서)는 반드시 오류로 잡힌다 — 조용히 넘어가지 않는다.
  3. 정적 판독이 불가능해지면(동적 생성 등) 통과가 아니라 실패한다.
  4. sys.modules에 스텁이 심겨 있어도 정적 대조 결과는 오염되지 않는다.
  5. ml_engine은 torch 없이도 import된다(순수 numpy 축은 CI에서 실제로 검증).
     torch를 건드리는 경로는 조용히 열화되지 않고 RuntimeError로 실패한다.
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_contract import (  # noqa: E402
    FEATURE_CONTRACT,
    feature_contract_sync_errors,
    is_stub_ml_engine,
    read_ml_engine_feature_cols,
)


def _write_ml_engine(tmp_path, body: str) -> str:
    p = tmp_path / "ml_engine.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


# ── 1. 정적 판독은 torch 없이도 항상 동작 ──────────────────────────
class TestStaticRead:
    def test_reads_real_ml_engine_without_importing_it(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "ml_engine", raising=False)
        cols = read_ml_engine_feature_cols()
        assert len(cols) == 16
        assert cols[0] == "Log_Ret"
        # 판독에 import 부작용이 없어야 한다 — 그래서 torch·스텁과 무관해진다
        assert "ml_engine" not in sys.modules, "정적 판독이 모듈을 import했다"

    def test_real_repo_is_in_sync(self):
        assert feature_contract_sync_errors() == [], "계약과 ml_engine이 어긋났다"

    def test_static_read_matches_contract_exactly(self):
        assert read_ml_engine_feature_cols() == list(FEATURE_CONTRACT.columns)


# ── 2. 불일치는 반드시 잡힌다 ─────────────────────────────────────
class TestMismatchDetected:
    def test_missing_column_detected(self, tmp_path):
        cols = list(FEATURE_CONTRACT.columns)[:-1]
        path = _write_ml_engine(tmp_path, f"FEATURE_COLS = {cols!r}\n")
        errs = feature_contract_sync_errors(path)
        assert errs, "컬럼이 빠졌는데 통과했다"
        assert "Upper_Shadow_Ratio" in errs[0]

    def test_extra_column_detected(self, tmp_path):
        cols = list(FEATURE_CONTRACT.columns) + ["Bogus_Feature"]
        path = _write_ml_engine(tmp_path, f"FEATURE_COLS = {cols!r}\n")
        errs = feature_contract_sync_errors(path)
        assert errs and "Bogus_Feature" in errs[0]

    def test_order_only_difference_detected(self, tmp_path):
        """원소는 같고 순서만 달라도 모델 입력이 어긋난다 — 반드시 오류."""
        cols = list(FEATURE_CONTRACT.columns)
        cols[0], cols[1] = cols[1], cols[0]
        path = _write_ml_engine(tmp_path, f"FEATURE_COLS = {cols!r}\n")
        errs = feature_contract_sync_errors(path)
        assert errs, "순서 차이를 놓쳤다"
        assert "순서" in errs[0]


# ── 3. 판독 불가 = 실패 (조용한 통과 금지) ─────────────────────────
class TestFailsLoudly:
    def test_missing_file_is_error_not_pass(self, tmp_path):
        errs = feature_contract_sync_errors(str(tmp_path / "nope.py"))
        assert errs and "판독 실패" in errs[0]
        with pytest.raises(FileNotFoundError):
            read_ml_engine_feature_cols(str(tmp_path / "nope.py"))

    def test_dynamic_definition_is_error(self, tmp_path):
        """FEATURE_COLS가 동적으로 만들어지면 계약 검사 방식을 재설계해야 한다.
        조용히 통과시키면 다시 '검사하지 않는 검사'가 된다."""
        path = _write_ml_engine(tmp_path, "FEATURE_COLS = [f'f{i}' for i in range(3)]\n")
        with pytest.raises(ValueError):
            read_ml_engine_feature_cols(path)
        assert feature_contract_sync_errors(path)

    def test_empty_list_is_error(self, tmp_path):
        path = _write_ml_engine(tmp_path, "FEATURE_COLS = []\n")
        with pytest.raises(ValueError):
            read_ml_engine_feature_cols(path)

    def test_no_definition_is_error(self, tmp_path):
        path = _write_ml_engine(tmp_path, "OTHER = 1\n")
        with pytest.raises(ValueError):
            read_ml_engine_feature_cols(path)

    def test_non_string_elements_is_error(self, tmp_path):
        path = _write_ml_engine(tmp_path, "FEATURE_COLS = ['a', 3]\n")
        with pytest.raises(ValueError):
            read_ml_engine_feature_cols(path)


# ── 4. 스텁이 정적 대조를 오염시킬 수 없다 ─────────────────────────
class TestStubIsolation:
    def test_stub_detection(self):
        assert is_stub_ml_engine(None)

        m = types.ModuleType("ml_engine")
        m.FEATURE_COLS = list(FEATURE_CONTRACT.columns)
        m.__is_test_stub__ = True
        assert is_stub_ml_engine(m), "명시 표시된 스텁을 진짜로 봤다"

        m2 = types.ModuleType("ml_engine")
        m2.FEATURE_COLS = []
        assert is_stub_ml_engine(m2), "빈 FEATURE_COLS를 진짜로 봤다"

    def test_real_module_is_not_stub(self):
        import ml_engine  # torch 없이도 import 가능해야 한다 (아래 5번)
        assert not is_stub_ml_engine(ml_engine)

    def test_stub_in_sys_modules_does_not_poison_static_check(self, monkeypatch):
        """실제로 터졌던 회귀: 스텁이 sys.modules에 남아 있어도 정적 대조는 무영향."""
        stub = types.ModuleType("ml_engine")
        stub.FEATURE_COLS = []
        stub.__is_test_stub__ = True
        monkeypatch.setitem(sys.modules, "ml_engine", stub)
        assert feature_contract_sync_errors() == [], "스텁에 오염됐다"

    def test_fill_aware_stub_is_cleaned_up(self):
        """tests/test_fill_aware_v248의 스텁이 세션에 남지 않아야 한다."""
        import tests.test_fill_aware_v248 as fa
        fa.teardown_module(fa)
        left = sys.modules.get("ml_engine")
        assert left is None or not is_stub_ml_engine(left)


# ── 5. ml_engine은 torch 없이 import된다 ──────────────────────────
class TestTorchOptional:
    def test_module_imports_and_declares_torch_state(self):
        import ml_engine
        assert isinstance(ml_engine.TORCH_AVAILABLE, bool)
        assert len(ml_engine.FEATURE_COLS) == 16

    def test_pure_numpy_reliability_gate_is_exercised(self):
        """torch가 없어도 신뢰도 게이트는 CI에서 실제로 돌아야 한다."""
        import numpy as np

        import ml_engine
        y = np.array([0, 1] * 150, dtype=float)
        r = ml_engine.evaluate_model_reliability(y, np.full(300, 0.5))
        assert r["reliable"] is False

    def test_torch_paths_fail_loudly_when_missing(self):
        import ml_engine
        if ml_engine.TORCH_AVAILABLE:
            pytest.skip("torch 설치 환경 — 부재 시 동작은 CI(무 torch)에서 검증")
        with pytest.raises(RuntimeError):
            ml_engine.TradingAttnLSTM(16)
        assert ml_engine.train_model() is False

    def test_apply_ml_score_reports_torch_missing_honestly(self):
        import pandas as pd

        import ml_engine
        if ml_engine.TORCH_AVAILABLE:
            pytest.skip("torch 설치 환경")
        out = ml_engine.apply_ml_score(pd.DataFrame({"종목코드": ["000001"]}), {})
        # 'MODEL_NOT_FOUND'로 뭉개지 않고 원인을 그대로 남긴다
        assert out["ML_STATUS"].iloc[0] == "TORCH_MISSING"
        assert float(out["ML_SCORE"].iloc[0]) == 0.0


# ── 6. 하드 게이트 자체가 실제로 검사한다 ──────────────────────────
class TestGateScript:
    def test_check_feature_contract_actually_compares(self):
        import check_contract_gate as gate
        assert gate.check_feature_contract() == []

    def test_gate_uses_static_path(self):
        """게이트가 정적 헬퍼를 경유하는지 — ImportError 스킵으로 되돌아가면 실패."""
        import inspect

        import check_contract_gate as gate
        src = inspect.getsource(gate.check_feature_contract)
        assert "feature_contract_sync_errors" in src
