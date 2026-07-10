#!/usr/bin/env python3
"""Fail-fast validation for build manifest and version labels."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_manifest import BUILD_MANIFEST  # noqa: E402
import version_info  # noqa: E402

assert version_info.APP_VERSION == BUILD_MANIFEST.app_version
assert version_info.RECOMMENDATION_ENGINE_VERSION == BUILD_MANIFEST.recommendation_engine_version
assert version_info.VALIDATION_ENGINE_VERSION == BUILD_MANIFEST.validation_engine_version
assert version_info.CHANGELOG[0]["version"] == BUILD_MANIFEST.app_version
print("build manifest OK:", BUILD_MANIFEST)
