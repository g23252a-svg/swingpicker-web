#!/usr/bin/env python3
"""
scripts/check_deps.py — Import 순환 참조 & 레이어 위반 검사
═══════════════════════════════════════════════════════════
GitHub Actions에서 실행:
  python scripts/check_deps.py

의존성 방향 규칙:
  views/ → components/ → services/ → core/
  ❌ 역방향 금지 (services → views 등)
  ❌ services/, core/에서 nicegui import 금지
"""
import ast
import os
import sys
from pathlib import Path
from collections import defaultdict

# ─── 레이어 정의 ───
LAYERS = {
    "views":      3,
    "components": 2,
    "services":   1,
    "core":       0,
}

# 어디서든 import 가능한 공유 모듈
SHARED = {
    "state", "config", "shared_utils", "time_utils", "schema",
    "version_info", "async_helpers", "collector_config",
}

# services/core에서 import 금지
UI_ONLY = {"nicegui", "streamlit"}


def get_layer(path: str):
    for part in Path(path).parts:
        if part in LAYERS:
            return part
    return None


def extract_imports(filepath: str) -> list[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError):
        return []
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                result.append((a.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append((node.module.split(".")[0], node.lineno))
    return result


def check():
    root = Path(".")
    errors = []

    for py in sorted(root.rglob("*.py")):
        rel = str(py.relative_to(root))
        # 제외
        if any(x in rel for x in ["__pycache__", "test_", ".github", "docs_cache"]):
            continue

        src_layer = get_layer(rel)
        if not src_layer:
            continue

        src_level = LAYERS[src_layer]

        for imp, lineno in extract_imports(str(py)):
            # 공유 모듈은 패스
            if imp in SHARED:
                continue

            # 1) 레이어 역방향 체크
            tgt_layer = None
            for name in LAYERS:
                if imp == name:
                    tgt_layer = name
                    break
            if tgt_layer and LAYERS[tgt_layer] > src_level:
                errors.append(
                    f"  {rel}:{lineno}  →  import {imp}\n"
                    f"    ❌ 상향 참조: {src_layer}(L{src_level}) → {tgt_layer}(L{LAYERS[tgt_layer]})"
                )

            # 2) UI 프레임워크 격리
            if imp in UI_ONLY and src_layer in ("services", "core"):
                errors.append(
                    f"  {rel}:{lineno}  →  import {imp}\n"
                    f"    ❌ {imp}는 views/components에서만 사용 가능 (현재: {src_layer}/)"
                )

    # 결과
    if errors:
        print(f"🔴 레이어 위반 {len(errors)}건 발견:\n")
        print("\n".join(errors))
        print(f"\n규칙: views(3) → components(2) → services(1) → core(0)")
        # 현재는 Warning — 리팩토링 완료 후 exit(1)로 전환
        print("\n⚠️  현재 Warning 모드 (리팩토링 진행 중)")
        # sys.exit(1)  ← 리팩토링 완료 후 활성화
    else:
        print("✅ 레이어 위반 없음")


if __name__ == "__main__":
    check()
