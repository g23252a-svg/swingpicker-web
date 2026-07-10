# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd

from build_manifest import BUILD_MANIFEST
from run_health import RunHealth
import version_info

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_version_ssot():
    assert version_info.APP_VERSION == BUILD_MANIFEST.app_version == "25.1.1"
    assert version_info.RECOMMENDATION_ENGINE_VERSION == BUILD_MANIFEST.recommendation_engine_version
    assert version_info.VALIDATION_ENGINE_VERSION == BUILD_MANIFEST.validation_engine_version
    assert version_info.CHANGELOG[0]["version"] == BUILD_MANIFEST.app_version


def test_backtest_engine_uses_manifest_version():
    import backtest_validation
    assert backtest_validation.BACKTEST_ENGINE_VERSION == f"v{BUILD_MANIFEST.backtest_engine_version}"


def test_data_health_column_keeps_legacy_alias():
    health = RunHealth(confidence_score=83.0)
    out = health.inject_columns(pd.DataFrame({"종목코드": ["000001"]}))
    assert out.loc[0, "DATA_HEALTH_SCORE"] == 83.0
    assert out.loc[0, "CONFIDENCE_SCORE"] == 83.0


def test_auto_collect_blocks_stale_validation_and_code_autocommit():
    workflow = (ROOT / ".github/workflows/auto_collect.yml").read_text(encoding="utf-8")
    assert "Verify fresh outputs" in workflow
    assert "stale artifact blocked" in workflow
    assert "기존 JSON 유지" not in workflow
    assert "git add -A" not in workflow
    assert "-X ours" not in workflow
