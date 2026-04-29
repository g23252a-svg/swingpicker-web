#!/usr/bin/env python3
"""
scripts/check_silent_exceptions.py — Silent Exception Detector (v2, AST-based)
═══════════════════════════════════════════════════════════════════════════════
GitHub Actions에서 실행:
  python scripts/check_silent_exceptions.py

사용법:
  python scripts/check_silent_exceptions.py                  # 검사
  python scripts/check_silent_exceptions.py --regenerate     # baseline 재생성
  python scripts/check_silent_exceptions.py --show-removed   # 제거된 항목 표시

═══════════════════════════════════════════════════════════════════════════════
[v22.2] AST 기반 + Baseline JSON
─────────────────────────────────────────────────────────────────────────────
이전 버전(v22.1) 한계:
  ❌ regex 기반 → except Exception as e: pass 누락
  ❌ regex 기반 → bare except: 누락
  ❌ regex 기반 → except (TypeError, KeyError): 누락
  ❌ 총량 예산 → 안전한 곳 1건 제거 + 위험한 곳 1건 추가하면 통과
  ❌ backup_*, data, venv 제외 누락

v22.2 변경:
  ✅ AST 방문 → 모든 except 변형 정확히 분류
  ✅ 심각도 분류:
      SILENT      = body가 pass뿐
      QUASI       = body가 print(...)만 — 거의 silent
      OK          = logger/logging 호출 또는 raise 있음
  ✅ Baseline JSON → 위치+구조 해시 기반 매칭
  ✅ EXCLUDE_DIRS 일원화 (backup_*, data, .venv 등)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# ─── 설정 ───
BASELINE_FILE = ".silent_exceptions_baseline.json"

# 하위 디렉터리 이름이 이것 중 하나라도 있으면 스킵
EXCLUDE_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "data", ".pytest_cache", ".mypy_cache", "node_modules",
}
# prefix로 시작하는 디렉터리도 스킵
EXCLUDE_DIR_PREFIXES = ("backup_", "legacy_")

# 이 파일들은 검사 대상에서 제외
EXCLUDE_FILE_PATTERNS = ("fix4_silent_exceptions.py",)

# ImportError류는 합법적 선택적 의존 처리 — 스킵
LEGITIMATE_IMPORT_EXCEPTIONS = {
    "ImportError", "ModuleNotFoundError",
}


def should_skip(path: Path) -> bool:
    """Common skip predicate — used by all CI scripts."""
    parts = path.parts
    for part in parts:
        if part in EXCLUDE_DIR_NAMES:
            return True
        if any(part.startswith(p) for p in EXCLUDE_DIR_PREFIXES):
            return True
    name = path.name
    if any(p in name for p in EXCLUDE_FILE_PATTERNS):
        return True
    return False


# ─── AST 분석 ─────────────────────────────────────────

def _exc_type_name(node: ast.expr | None) -> str:
    """except 절의 type 부분을 사람이 읽을 수 있는 이름으로."""
    if node is None:
        return "bare"            # except:
    if isinstance(node, ast.Name):
        return node.id           # except Exception:
    if isinstance(node, ast.Attribute):
        # ex) except mod.Error:
        parts = []
        cur: Any = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Tuple):
        names = [_exc_type_name(elt) for elt in node.elts]
        return "(" + ",".join(names) + ")"
    # ast.unparse는 3.9+, fallback로 클래스명
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _is_logger_call(stmt: ast.stmt) -> bool:
    """statement가 logger.X(...) 또는 logging.X(...) 호출인지."""
    if not isinstance(stmt, ast.Expr):
        return False
    call = stmt.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if isinstance(func, ast.Attribute):
        # 가장 바깥 객체 이름 추출
        cur: Any = func.value
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Name):
            base = cur.id.lower()
            return any(kw in base for kw in (
                "logger", "logging", "log", "_log", "_logger"
            ))
    return False


def _is_print_only(stmt: ast.stmt) -> bool:
    """statement가 print(...) 호출인지."""
    if not isinstance(stmt, ast.Expr):
        return False
    call = stmt.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if isinstance(func, ast.Name) and func.id == "print":
        return True
    return False


def _classify_handler(handler: ast.ExceptHandler) -> tuple[str | None, str]:
    """
    Returns (severity, body_kind).
    severity: "SILENT" | "QUASI" | None (OK, no finding)
    body_kind: "pass" | "print_only" | "log_or_raise" | "code"
    """
    body = handler.body
    if not body:
        return ("SILENT", "pass")

    # 1) pass뿐
    if all(isinstance(s, ast.Pass) for s in body):
        return ("SILENT", "pass")

    # 2) raise 또는 logger 호출이 있으면 OK
    has_raise = any(isinstance(s, ast.Raise) for s in body)
    has_log = any(_is_logger_call(s) for s in body)
    if has_raise or has_log:
        return (None, "log_or_raise")

    # 3) print(...)뿐이면 QUASI silent
    only_print = all(
        _is_print_only(s) or isinstance(s, ast.Pass)
        for s in body
    )
    if only_print:
        return ("QUASI", "print_only")

    # 4) 그 외 코드(할당, 함수 호출, return, ...) → 정상 처리로 간주
    return (None, "code")


class SilentExceptVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str, source_lines: list[str]):
        self.filepath = filepath
        self.source_lines = source_lines
        self.findings: list[dict] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        exc_type = _exc_type_name(node.type)

        # 합법적 import 가드는 스킵
        if exc_type in LEGITIMATE_IMPORT_EXCEPTIONS:
            self.generic_visit(node)
            return
        # 튜플 안에 ImportError 단독이어도 스킵 (선택적 import 패턴)
        if exc_type == "(ImportError,ModuleNotFoundError)" or \
           exc_type == "(ModuleNotFoundError,ImportError)":
            self.generic_visit(node)
            return

        severity, body_kind = _classify_handler(node)
        if severity is None:
            self.generic_visit(node)
            return

        line = node.lineno
        code_line = self.source_lines[line - 1] if 0 < line <= len(self.source_lines) else ""

        # 안정적 ID — 라인 시프트에 강함
        ident_seed = f"{self.filepath}::{exc_type}::{body_kind}::{code_line.strip()}"
        ident = hashlib.sha1(ident_seed.encode("utf-8")).hexdigest()[:12]

        self.findings.append({
            "file": self.filepath,
            "line": line,
            "exc_type": exc_type,
            "severity": severity,
            "body_kind": body_kind,
            "id": ident,
            "code": code_line.strip()[:100],
        })
        self.generic_visit(node)


def scan_file(filepath: Path) -> list[dict]:
    try:
        source = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError, OSError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = SilentExceptVisitor(str(filepath), source.splitlines())
    visitor.visit(tree)
    return visitor.findings


def scan_all(root: Path) -> list[dict]:
    results: list[dict] = []
    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root)
        if should_skip(rel):
            continue
        results.extend(scan_file(py))
    # 결과는 file → 정규화된 상대경로로
    root_str = str(root)
    for r in results:
        if r["file"].startswith(root_str + "/"):
            r["file"] = r["file"][len(root_str) + 1:]
        elif r["file"].startswith("./"):
            r["file"] = r["file"][2:]
        r["file"] = r["file"].replace("\\", "/")
    # Fingerprint(중복 가능) — 라인번호 제외해서 라인 시프트에 강함
    for r in results:
        fp_seed = f"{r['file']}::{r['exc_type']}::{r['body_kind']}::{r['code']}"
        r["fingerprint"] = hashlib.sha1(fp_seed.encode("utf-8")).hexdigest()[:12]
    return results


def fingerprint_counts(findings: list[dict]) -> dict[str, int]:
    """fingerprint별 발생 횟수 Counter."""
    counts: dict[str, int] = {}
    for f in findings:
        counts[f["fingerprint"]] = counts.get(f["fingerprint"], 0) + 1
    return counts


# ─── Baseline I/O ─────────────────────────────────────

def load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️  Baseline 파싱 실패 ({e}). 신규처럼 처리.")
        return None


def write_baseline(path: Path, findings: list[dict]) -> None:
    counts = fingerprint_counts(findings)
    data = {
        "version": 2,
        "doc": (
            "DO NOT edit by hand unless you know what you're doing. "
            "Regenerate with: python scripts/check_silent_exceptions.py --regenerate. "
            "Matching is by (file, exc_type, body_kind, code) multiset count — "
            "robust to line shifts but distinguishes duplicates within same file."
        ),
        "count": len(findings),
        "by_severity": {
            "SILENT": sum(1 for f in findings if f["severity"] == "SILENT"),
            "QUASI": sum(1 for f in findings if f["severity"] == "QUASI"),
        },
        # fingerprint별 카운트 — multiset 비교용
        "fingerprint_counts": dict(sorted(counts.items())),
        # 발견 위치 목록 — 사람이 읽기/추적용 (매칭 로직은 사용 안 함)
        "findings": [
            {
                "fingerprint": f["fingerprint"],
                "file": f["file"],
                "line": f["line"],
                "exc_type": f["exc_type"],
                "severity": f["severity"],
                "body_kind": f["body_kind"],
            }
            for f in sorted(findings, key=lambda x: (x["file"], x["line"]))
        ],
    }
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ─── Main ─────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true",
                        help="Baseline 파일을 현재 상태로 재생성")
    parser.add_argument("--show-removed", action="store_true",
                        help="Baseline에 있지만 더 이상 없는 항목도 표시")
    args = parser.parse_args()

    root = Path(".")
    baseline_path = root / BASELINE_FILE
    findings = scan_all(root)

    silent_n = sum(1 for f in findings if f["severity"] == "SILENT")
    quasi_n = sum(1 for f in findings if f["severity"] == "QUASI")

    print("🔇 Silent Exception 검사 (AST + Baseline)")
    print("─" * 60)
    print(f"  현재 SILENT (pass뿐):       {silent_n:>4d}건")
    print(f"  현재 QUASI  (print(e)뿐):  {quasi_n:>4d}건")
    print(f"  현재 합계:                 {len(findings):>4d}건")
    print()

    # ─── Regenerate 모드 ───
    if args.regenerate:
        write_baseline(baseline_path, findings)
        print(f"✅ Baseline 재생성됨 → {BASELINE_FILE}")
        print(f"   findings: {len(findings)}건 (SILENT={silent_n}, QUASI={quasi_n})")
        print(f"   git add {BASELINE_FILE} 후 커밋하세요.")
        return 0

    # ─── 일반 모드 ───
    baseline = load_baseline(baseline_path)
    if baseline is None:
        # 첫 실행 — baseline이 없음. 자동 생성하고 WARN.
        write_baseline(baseline_path, findings)
        print(f"⚠️  Baseline이 없어서 자동 생성됨 → {BASELINE_FILE}")
        print(f"   이번 PR에서 같이 커밋하세요. 다음 PR부터 회귀 차단 작동.")
        return 0

    # ─── Multiset 비교 ───
    # baseline의 fingerprint별 카운트와 비교.
    # baseline에 없는 fingerprint: 100% 신규.
    # baseline에 있지만 현재가 더 많음: 차분만큼 신규.
    baseline_counts = baseline.get("fingerprint_counts")
    if baseline_counts is None:
        # v1 포맷 baseline (id 기반) — 자동 마이그레이션 후 다시 검사
        print("⚠️  Baseline이 구버전 포맷(v1)입니다. 자동 재생성 → multiset 포맷(v2)")
        write_baseline(baseline_path, findings)
        print(f"   {BASELINE_FILE}을 커밋하세요. 다음 실행부터 정확한 회귀 차단 작동.")
        return 0

    current_counts = fingerprint_counts(findings)

    new_findings: list[dict] = []
    seen_per_fp: dict[str, int] = {}
    # 라인 순으로 정렬해서, fingerprint별로 처음 baseline_count개는 known,
    # 그 이후의 발생은 신규.
    for f in sorted(findings, key=lambda x: (x["file"], x["line"])):
        fp = f["fingerprint"]
        seen_per_fp[fp] = seen_per_fp.get(fp, 0) + 1
        if seen_per_fp[fp] > baseline_counts.get(fp, 0):
            new_findings.append(f)

    # 제거된 항목 (baseline 카운트 > current 카운트)
    removed_fps = {
        fp: baseline_counts[fp] - current_counts.get(fp, 0)
        for fp in baseline_counts
        if baseline_counts[fp] > current_counts.get(fp, 0)
    }
    removed_count = sum(removed_fps.values())

    print(f"  Baseline 등록 카운트:      {sum(baseline_counts.values()):>4d}건 "
          f"({len(baseline_counts)} 패턴)")
    print(f"  ✨ 신규 (NEW):             {len(new_findings):>4d}건")
    print(f"  🗑️  제거 (REMOVED):         {removed_count:>4d}건")
    print()

    if new_findings:
        print("🔴 신규 Silent Exception (Baseline에 없음):")
        print()
        by_file: dict[str, list[dict]] = {}
        for f in new_findings:
            by_file.setdefault(f["file"], []).append(f)
        for fp, items in sorted(by_file.items()):
            print(f"  📄 {fp}")
            for it in items:
                tag = "🔴" if it["severity"] == "SILENT" else "🟡"
                print(f"     L{it['line']:4d} [{tag} {it['severity']}] "
                      f"except {it['exc_type']}: ({it['body_kind']})")
                print(f"           {it['code']}")
            print()

    if args.show_removed and removed_fps:
        print("🗑️  Baseline에서 제거된 패턴:")
        for fp, n in sorted(removed_fps.items(), key=lambda x: -x[1]):
            # baseline.findings에서 해당 fp의 첫 등장 위치 1개만 표시
            example = next(
                (b for b in baseline.get("findings", []) if b["fingerprint"] == fp),
                None,
            )
            if example:
                print(f"   · {n}회 — {example['file']} (예: line {example['line']}, "
                      f"except {example['exc_type']}: ({example['body_kind']}))")
        print()

    # ─── 판정 ───
    if new_findings:
        print(f"❌ 신규 silent exception {len(new_findings)}건 발견 → CI 실패")
        print(f"   해결 방법:")
        print(f"     1. logger.exception() 또는 logger.error() 추가")
        print(f"     2. 정당한 이유라면 baseline 재생성:")
        print(f"        python scripts/check_silent_exceptions.py --regenerate")
        print(f"        git add {BASELINE_FILE}")
        return 1

    if removed_count:
        print(f"✅ 통과 — {removed_count}건 제거됨 (baseline 갱신 권장)")
        print(f"   python scripts/check_silent_exceptions.py --regenerate")
        print(f"   git add {BASELINE_FILE}")
    else:
        print("✅ 통과 — 신규 silent exception 없음")

    return 0


if __name__ == "__main__":
    sys.exit(main())
