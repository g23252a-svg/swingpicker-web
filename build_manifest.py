# -*- coding: utf-8 -*-
"""Fail-fast runtime build/version manifest loader."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

_MANIFEST_PATH = Path(__file__).with_name("build_manifest.json")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


@dataclass(frozen=True)
class BuildManifest:
    app_version: str
    recommendation_engine_version: str
    validation_engine_version: str
    backtest_engine_version: str
    schema_version: int
    release_channel: str
    git_sha: str


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"build manifest field {key!r} must be a non-empty string")
    return value.strip()


def load_build_manifest(path: Path = _MANIFEST_PATH) -> BuildManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load build manifest: {path}") from exc

    app_version = _required_str(payload, "app_version")
    recommendation_version = _required_str(payload, "recommendation_engine_version")
    validation_version = _required_str(payload, "validation_engine_version")
    backtest_version = _required_str(payload, "backtest_engine_version")
    for key, value in (
        ("app_version", app_version),
        ("recommendation_engine_version", recommendation_version),
        ("validation_engine_version", validation_version),
        ("backtest_engine_version", backtest_version),
    ):
        if not _SEMVER_RE.fullmatch(value):
            raise RuntimeError(f"build manifest field {key!r} is not semantic version: {value!r}")

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < 1:
        raise RuntimeError("build manifest field 'schema_version' must be a positive integer")

    return BuildManifest(
        app_version=app_version,
        recommendation_engine_version=recommendation_version,
        validation_engine_version=validation_version,
        backtest_engine_version=backtest_version,
        schema_version=schema_version,
        release_channel=_required_str(payload, "release_channel"),
        git_sha=os.getenv("GITHUB_SHA", os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown")),
    )


BUILD_MANIFEST = load_build_manifest()
APP_VERSION = BUILD_MANIFEST.app_version
RECOMMENDATION_ENGINE_VERSION = BUILD_MANIFEST.recommendation_engine_version
VALIDATION_ENGINE_VERSION = BUILD_MANIFEST.validation_engine_version
BACKTEST_ENGINE_VERSION = BUILD_MANIFEST.backtest_engine_version
SCHEMA_VERSION = BUILD_MANIFEST.schema_version
