#!/usr/bin/env python3
"""
scripts/check_deps.py — Import 검사 (v2)
═══════════════════════════════════════════════════════════════════════
GitHub Actions에서 실행:
  python scripts/check_deps.py

검사 항목:
  1. 레이어 방향 위반 (services → views, core → components 등)
  2. UI 프레임워크 격리 (services/, core/는 nicegui/streamlit 직접 import 금지)
  3. 모듈 간 순환참조 (실제 cycle detection — DFS 기반)

═══════════════════════════════════════════════════════════════════════
[v22.2] 인라인 디렉티브 + 실제 Cycle Detection
─────────────────────────────────────────────────────────────────────
이전 버전(v22.1) 한계:
  ❌ ALLOWED_LEGACY_VIOLATIONS가 라인번호 기반 → 주석 한 줄 추가하면 깨짐
  ❌ 이름은 "순환참조 검사"인데 실제로는 레이어 방향만 검사
  ❌ backup_*, data, venv 디렉터리 제외 누락

v22.2 변경:
  ✅ 인라인 디렉티브: `# ci-allow: layer-violation` 주석으로 격리
     · 라인 시프트에 강함 (코드 자체에 붙어 있음)
     · 코드 리뷰 시 명시적으로 보임
  ✅ 실제 순환참조 검사 추가 (DFS 기반)
  ✅ EXCLUDE_DIRS 일원화

레이어 방향 규칙:
  views/(L3) → components/(L2) → services/(L1) → core/(L0)
  ❌ 역방향 금지
  ❌ services/, core/에서 nicegui/streamlit import 금지
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

# ─── 설정 ───
LAYERS = {
    "views":      3,
    "components": 2,
    "services":   1,
    "core":       0,
}

# 어디서든 import 가능한 공유 모듈 (레이어 검사 면제)
SHARED = {
    "state", "config", "shared_utils", "time_utils", "schema",
    "version_info", "async_helpers", "collector_config",
}

# services/core에서 import 금지
UI_ONLY = {"nicegui", "streamlit"}

# 디렉터리 제외 (공통 헬퍼와 동일 정책)
EXCLUDE_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "data", ".pytest_cache", ".mypy_cache", "node_modules",
    ".github", "docs_cache", "static",
}
EXCLUDE_DIR_PREFIXES = ("backup_", "legacy_")

# 인라인 디렉티브 — 위반 라인에 이 주석이 있으면 격리
DIRECTIVE_LAYER = "ci-allow: layer-violation"
DIRECTIVE_CYCLE = "ci-allow: import-cycle"

# 순환참조 baseline — 기존 cycle은 격리
CYCLES_BASELINE_FILE = ".import_cycles_baseline.json"


def should_skip(path: Path) -> bool:
    parts = path.parts
    for part in parts:
        if part in EXCLUDE_DIR_NAMES:
            return True
        if any(part.startswith(p) for p in EXCLUDE_DIR_PREFIXES):
            return True
    if path.name.startswith("test_") or path.name.startswith("conftest"):
        return True
    return False


def get_layer(rel: Path) -> str | None:
    for part in rel.parts:
        if part in LAYERS:
            return part
    return None


def extract_imports(filepath: Path) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    result: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                result.append((a.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append((node.module.split(".")[0], node.lineno))
    return result


def line_has_directive(filepath: Path, lineno: int, directive: str) -> bool:
    """해당 라인이 `# ci-allow: <directive>` 주석을 포함하는지."""
    try:
        with filepath.open(encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if i == lineno:
                    return directive in line
                if i > lineno:
                    break
    except OSError:
        return False
    return False


# ─── 1. 레이어/UI 격리 검사 ─────────────────────────────────

def check_layer_violations(root: Path) -> tuple[list[str], list[str]]:
    """Returns (allowed_with_directive, new_violations)."""
    allowed: list[str] = []
    new: list[str] = []

    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root)
        if should_skip(rel):
            continue

        src_layer = get_layer(rel)
        if not src_layer:
            continue
        src_level = LAYERS[src_layer]

        for imp, lineno in extract_imports(py):
            if imp in SHARED:
                continue

            violation = None

            # 1) 레이어 역방향
            if imp in LAYERS and LAYERS[imp] > src_level:
                violation = (
                    f"  {rel}:{lineno}  →  import {imp}\n"
                    f"    ❌ 상향 참조: {src_layer}(L{src_level}) → "
                    f"{imp}(L{LAYERS[imp]})"
                )

            # 2) UI 프레임워크 격리
            if imp in UI_ONLY and src_layer in ("services", "core"):
                violation = (
                    f"  {rel}:{lineno}  →  import {imp}\n"
                    f"    ❌ {imp}는 views/components에서만 사용 가능 "
                    f"(현재: {src_layer}/)"
                )

            if violation:
                if line_has_directive(py, lineno, DIRECTIVE_LAYER):
                    allowed.append(f"  · {rel}:{lineno}  import {imp}")
                else:
                    new.append(violation)

    return allowed, new


# ─── 2. 순환참조 검사 (DFS) ───────────────────────────────

def get_first_party_modules(root: Path) -> set[str]:
    mods: set[str] = set()
    for py in root.rglob("*.py"):
        rel = py.relative_to(root)
        if should_skip(rel):
            continue
        mods.add(py.stem)
    return mods


def build_import_graph(root: Path, first_party: set[str]) -> dict[str, set[str]]:
    """노드 = 모듈명(stem), edge = import 관계."""
    graph: dict[str, set[str]] = defaultdict(set)
    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root)
        if should_skip(rel):
            continue
        src = py.stem
        for imp, _ in extract_imports(py):
            if imp == src:
                continue  # self-import 무시
            if imp in first_party:
                graph[src].add(imp)
    return graph


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """DFS로 모든 단순 cycle 검출. 정규화된 형태(최소 노드부터 회전)로 dedupe."""
    cycles: set[tuple[str, ...]] = set()
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in graph}
    # graph에 없지만 edge로만 등장하는 노드도 추가
    for nbrs in list(graph.values()):
        for nb in nbrs:
            color.setdefault(nb, WHITE)

    stack: list[str] = []

    def dfs(u: str) -> None:
        color[u] = GRAY
        stack.append(u)
        for v in sorted(graph.get(u, set())):
            if color.get(v, WHITE) == GRAY:
                # back edge → cycle
                idx = stack.index(v)
                cycle = tuple(stack[idx:])
                # 정규화: 최소 노드부터 시작하도록 회전
                m = min(cycle)
                k = cycle.index(m)
                norm = cycle[k:] + cycle[:k]
                cycles.add(norm)
            elif color.get(v, WHITE) == WHITE:
                dfs(v)
        stack.pop()
        color[u] = BLACK

    for n in sorted(color):
        if color[n] == WHITE:
            dfs(n)

    return [list(c) for c in sorted(cycles)]


def cycle_signature(cycle: list[str]) -> str:
    """cycle의 정규화된 키 (최소 노드부터 정렬된 형태)."""
    return "→".join(cycle) + "→" + cycle[0]


def cycle_has_directive(root: Path, cycle: list[str]) -> bool:
    """cycle에 포함된 import 중 하나라도 `# ci-allow: import-cycle` 주석이 있으면 격리."""
    # cycle이 a→b→c→a라면, a→b, b→c, c→a 각 import 라인을 점검
    nodes = cycle + [cycle[0]]
    for i in range(len(cycle)):
        src_mod, dst_mod = nodes[i], nodes[i + 1]
        # src_mod라는 stem의 .py 파일 찾기
        for py in root.rglob(f"{src_mod}.py"):
            rel = py.relative_to(root)
            if should_skip(rel):
                continue
            # 그 파일에서 dst_mod를 import하는 라인 찾기
            for imp, lineno in extract_imports(py):
                if imp == dst_mod:
                    if line_has_directive(py, lineno, DIRECTIVE_CYCLE):
                        return True
    return False


def load_cycle_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("cycles", []))
    except (json.JSONDecodeError, OSError):
        return set()


def write_cycle_baseline(path: Path, signatures: set[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "doc": (
                    "Existing import cycles. Add new cycles here only with "
                    "explicit review. Generated automatically on first run."
                ),
                "cycles": sorted(signatures),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


# ─── Main ─────────────────────────────────────────────────

def main() -> int:
    regenerate = "--regenerate" in sys.argv
    root = Path(".")

    print("🔍 Import & Layer Direction Check (v2)")
    print("─" * 60)

    # 1) 레이어 위반
    allowed, new_violations = check_layer_violations(root)
    print(f"\n[1/2] 레이어 / UI 격리")
    print(f"  격리된 위반 (디렉티브):  {len(allowed)}건")
    print(f"  신규 위반:               {len(new_violations)}건")
    if allowed:
        print("\n  📦 디렉티브로 격리된 위반:")
        for line in allowed:
            print(line)
    if new_violations:
        print("\n  🔴 신규 레이어 위반:")
        print()
        print("\n".join(new_violations))
        print()
        print("  해결:")
        print("    1. 의존성 방향 반전 (UI 호출 → 콜백/이벤트 패턴)")
        print(f"    2. 정당한 사유라면 라인 끝에 ` # {DIRECTIVE_LAYER}` 추가")

    # 2) 순환참조
    print(f"\n[2/2] 모듈 순환참조 (Cycle Detection)")
    first_party = get_first_party_modules(root)
    graph = build_import_graph(root, first_party)
    cycles = find_cycles(graph)

    baseline_path = root / CYCLES_BASELINE_FILE

    # cycle별로 디렉티브 격리 여부 판단
    cycle_sigs_current: set[str] = set()
    cycle_sigs_with_directive: set[str] = set()
    for c in cycles:
        sig = cycle_signature(c)
        cycle_sigs_current.add(sig)
        if cycle_has_directive(root, c):
            cycle_sigs_with_directive.add(sig)

    if regenerate:
        write_cycle_baseline(baseline_path, cycle_sigs_current)
        print(f"  ✅ Baseline 재생성됨 → {CYCLES_BASELINE_FILE}")
        print(f"     {len(cycle_sigs_current)}건 등록")
        if new_violations:
            return 1
        return 0

    baseline_sigs = load_cycle_baseline(baseline_path)
    if not baseline_path.exists():
        # 첫 실행 — baseline 자동 생성
        write_cycle_baseline(baseline_path, cycle_sigs_current)
        print(f"  ⚠️  Baseline 없음 → 자동 생성 ({len(cycle_sigs_current)}건)")
        print(f"     이번 PR에서 같이 커밋하세요.")
        if new_violations:
            return 1
        return 0

    # cycle 분류
    new_cycles = cycle_sigs_current - baseline_sigs - cycle_sigs_with_directive
    removed_cycles = baseline_sigs - cycle_sigs_current

    print(f"  Baseline 등록 cycle:     {len(baseline_sigs)}건")
    print(f"  현재 발견 cycle:         {len(cycle_sigs_current)}건")
    print(f"  디렉티브 격리 cycle:     {len(cycle_sigs_with_directive)}건")
    print(f"  ✨ 신규 cycle:            {len(new_cycles)}건")
    print(f"  🗑️  제거된 cycle:          {len(removed_cycles)}건")

    if new_cycles:
        print()
        print("  🔴 신규 순환참조:")
        for sig in sorted(new_cycles):
            print(f"     · {sig}")
        print()
        print("  해결:")
        print("    1. 두 모듈 사이의 import 방향을 단방향으로 (의존성 분리)")
        print(f"    2. 정당한 사유라면 import 라인에 `# {DIRECTIVE_CYCLE}` 또는")
        print(f"       baseline 재생성: python scripts/check_deps.py --regenerate")

    # ─── 최종 판정 ───
    print()
    print("─" * 60)
    if new_violations or new_cycles:
        print(f"❌ FAIL — 레이어 위반 {len(new_violations)}, 신규 cycle {len(new_cycles)}")
        return 1

    print("✅ PASS — 신규 위반/cycle 없음")
    if removed_cycles:
        print(f"   💡 {len(removed_cycles)}건 cycle 제거됨 — baseline 갱신 권장")
    return 0


if __name__ == "__main__":
    sys.exit(main())
