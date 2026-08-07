# -*- coding: utf-8 -*-
"""
feature_contract.py — ML Feature Schema Contract
═══════════════════════════════════════════════════
[v20.7] 실전/백테스트/학습이 동일한 feature를 쓰도록 강제.

사용법:
    from feature_contract import FEATURE_CONTRACT, validate_features

    # 추론 전 검증
    ok, errors = validate_features(df, context="inference")
    if not ok:
        logger.warning(f"Feature mismatch: {errors}")
        # ML 축 비활성 + 상태 기록
"""
import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureContract:
    """ML Feature Schema의 단일 원천.

    ml_engine.py, auto_backtest.py, collector.py 모두
    이 Contract를 참조해야 함.
    """
    # ── Feature 컬럼 (순서 중요 — 모델 입력 순서) ──
    columns: Tuple[str, ...] = (
        "Log_Ret", "Volume_Norm", "Low_Trend", "Vol_Quality", "Dist_MA20",
        "RSI", "MFI", "MACD_Hist_Norm", "BB_Width", "ATR_Pct",
        "OBV_Slope", "Range_Pos", "Vol_Ratio_5", "Ret_5d", "Ret_20d",
        "Upper_Shadow_Ratio",
    )

    # ── 모델 메타 ──
    seq_length: int = 40
    rolling_z_enabled: bool = True
    schema_version: str = "v20.8"

    @property
    def n_features(self) -> int:
        return len(self.columns)

    @property
    def schema_hash(self) -> str:
        """컬럼 순서 + 설정 해시."""
        payload = f"{','.join(self.columns)}|seq={self.seq_length}|rz={self.rolling_z_enabled}"
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    def validate(self, df_columns: List[str], context: str = "") -> Tuple[bool, List[str]]:
        """DataFrame 컬럼이 Contract와 일치하는지 검증.

        Returns:
            (is_valid, error_messages)
        """
        errors = []
        expected = set(self.columns)
        actual = set(df_columns)

        missing = expected - actual
        if missing:
            errors.append(f"[{context}] Missing columns: {sorted(missing)}")

        extra = actual - expected
        # extra는 경고만 (추가 컬럼은 허용)

        # 순서 검증 (순서가 다르면 모델 입력이 꼬임)
        actual_ordered = [c for c in df_columns if c in expected]
        expected_ordered = list(self.columns)
        if actual_ordered != expected_ordered[:len(actual_ordered)]:
            errors.append(f"[{context}] Column order mismatch")

        return (len(errors) == 0, errors)

    def to_dict(self) -> dict:
        return {
            "columns": list(self.columns),
            "n_features": self.n_features,
            "seq_length": self.seq_length,
            "rolling_z_enabled": self.rolling_z_enabled,
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_and_verify(cls, path: str) -> Tuple[bool, List[str]]:
        """저장된 schema와 현재 Contract 비교."""
        contract = cls()
        try:
            with open(path, 'r') as f:
                saved = json.load(f)
            errors = []
            if saved.get("schema_hash") != contract.schema_hash:
                errors.append(
                    f"Schema hash mismatch: saved={saved.get('schema_hash')}, "
                    f"current={contract.schema_hash}"
                )
            if saved.get("n_features") != contract.n_features:
                errors.append(
                    f"Feature count: saved={saved.get('n_features')}, "
                    f"current={contract.n_features}"
                )
            return (len(errors) == 0, errors)
        except Exception as e:
            return (False, [f"Schema file load failed: {e}"])


# ── 싱글턴 ──
FEATURE_CONTRACT = FeatureContract()


# ══════════════════════════════════════════════════════════════════
#  [v55.4] ml_engine.FEATURE_COLS 정적 판독 — torch 없이도 계약을 검사한다
# ══════════════════════════════════════════════════════════════════
#
# 이 계약을 검사하던 두 곳이 **정작 CI에서는 아무것도 검사하지 않고 있었다**:
#   · check_contract_gate.check_feature_contract() — `except ImportError`로
#     경고만 내고 통과 (CI에 torch가 없으므로 항상 이 경로)
#   · test_policy_consistency의 계약 테스트 2개 — `pytest.skip`으로 항상 스킵
# ml_engine.py는 최상단에서 `import torch`를 하므로, 계약 하나 읽으려고
# 수백 MB 의존성을 요구하는 구조였다. 그 결과 "동기화 하드 게이트"가
# 사실상 데드코드였다.
#
# 게다가 스킵 가드가 조용해서 **가짜 모듈까지 통과시켰다**:
#   tests/test_fill_aware_v248.py가 torch 부재 시 `FEATURE_COLS = []`인
#   스텁 ml_engine을 sys.modules에 심고 제거하지 않는다. 그 뒤에 도는 계약
#   테스트는 ImportError를 못 보므로 스킵하지 않고, 빈 목록과 비교해 실패한다.
#   (2026-08 실측: tests/ + 루트 테스트를 한 프로세스로 돌릴 때만 재현)
#
# 해법: 소스를 AST로 파싱해 리터럴만 읽는다. import 부작용이 없으므로
# torch 유무·스텁 유무와 무관하게 항상 같은 답을 준다.
_ML_ENGINE_FILENAME = "ml_engine.py"


def read_ml_engine_feature_cols(path: Optional[str] = None) -> List[str]:
    """ml_engine.py의 FEATURE_COLS를 정적으로 읽는다 (import 없음).

    Raises:
        FileNotFoundError: ml_engine.py를 찾을 수 없을 때.
        ValueError: FEATURE_COLS가 없거나 문자열 리스트 리터럴이 아닐 때.
                    (동적 생성으로 바뀌었다면 이 계약 검사를 다시 설계해야 하므로
                     조용히 넘기지 않고 실패시킨다.)
    """
    import ast
    import os

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            _ML_ENGINE_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{_ML_ENGINE_FILENAME} 없음: {path}")
    tree = ast.parse(open(path, "r", encoding="utf-8").read(), filename=path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id == "FEATURE_COLS"):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"FEATURE_COLS가 정적 리터럴이 아니다 ({e}) — "
                    f"계약 검사 방식을 다시 설계해야 한다") from e
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"FEATURE_COLS가 비었거나 리스트가 아니다: {value!r}")
            if not all(isinstance(v, str) for v in value):
                raise ValueError("FEATURE_COLS에 문자열 아닌 원소가 있다")
            return list(value)
    raise ValueError(f"{_ML_ENGINE_FILENAME}에 FEATURE_COLS 정의가 없다")


def is_stub_ml_engine(module) -> bool:
    """테스트 스텁 ml_engine인지 판별 — 진짜 모듈로 착각하면 안 된다."""
    if module is None:
        return True
    if getattr(module, "__is_test_stub__", False):
        return True
    src = getattr(module, "__file__", None)
    if not src or not str(src).endswith(_ML_ENGINE_FILENAME):
        return True
    cols = getattr(module, "FEATURE_COLS", None)
    return not cols          # 빈 목록/없음 → 스텁으로 취급


def feature_contract_sync_errors(path: Optional[str] = None) -> List[str]:
    """계약 ↔ ml_engine 동기화 검사 (정적). 오류 목록을 돌려준다."""
    try:
        ml_cols = read_ml_engine_feature_cols(path)
    except (FileNotFoundError, ValueError) as e:
        return [f"FEATURE_COLS 정적 판독 실패: {e}"]
    contract_cols = list(FEATURE_CONTRACT.columns)
    if contract_cols == ml_cols:
        return []
    only_c = [c for c in contract_cols if c not in ml_cols]
    only_m = [c for c in ml_cols if c not in contract_cols]
    detail = [f"FEATURE_COLS 불일치 (개수 contract {len(contract_cols)} / "
              f"ml_engine {len(ml_cols)})"]
    if only_c:
        detail.append(f"  contract에만: {only_c}")
    if only_m:
        detail.append(f"  ml_engine에만: {only_m}")
    if not only_c and not only_m:
        detail.append("  원소는 같고 순서가 다르다 — 모델 입력 순서가 어긋난다")
    return ["\n".join(detail)]


def validate_features(df, context: str = "inference") -> Tuple[bool, List[str]]:
    """편의 함수: DataFrame의 feature 컬럼 검증."""
    return FEATURE_CONTRACT.validate(list(df.columns), context=context)
