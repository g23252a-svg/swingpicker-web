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


def validate_features(df, context: str = "inference") -> Tuple[bool, List[str]]:
    """편의 함수: DataFrame의 feature 컬럼 검증."""
    return FEATURE_CONTRACT.validate(list(df.columns), context=context)
