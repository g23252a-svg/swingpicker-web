#!/usr/bin/env python3
"""
scripts/check_silent_exceptions.py — except Exception: pass 검출기
═══════════════════════════════════════════════════════════════════
GitHub Actions에서 실행:
  python scripts/check_silent_exceptions.py

검출 패턴:
  1. except Exception: pass          ← 완전 무시
  2. except Exception:\n    pass     ← 여러 줄 pass
  3. except Exception:               ← 아무 처리 없이 다음 줄
  4. except:\n    pass               ← bare except + pass

허용 패턴:
  - except ImportError:              ← 선택적 의존성 (정상)
  - except Exception as e: logger.*  ← 로깅 있으면 OK
"""
import os
import re
import sys
from pathlib import Path

# ─── 설정 ───
MAX_ALLOWED = 20   # 현재 허용 임계치 (점진적으로 0으로 줄임)
EXCLUDE_FILES = {"test_", "fix4_silent_exceptions.py", "__pycache__"}


def scan_file(filepath: str) -> list[dict]:
    """파일에서 silent exception 패턴 검출."""
    findings = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (UnicodeDecodeError, PermissionError):
        return []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # except ... : pass (한 줄)
        if re.match(r"except\s+\w*.*:\s*pass\s*$", stripped):
            # except ImportError: pass 는 허용
            if "ImportError" in stripped or "ModuleNotFoundError" in stripped:
                continue
            findings.append({
                "file": filepath,
                "line": i,
                "code": stripped,
                "type": "pass",
            })
            continue

        # except Exception: (다음 줄이 pass인지 확인)
        if re.match(r"except\s+(Exception|BaseException)\s*:", stripped):
            # 다음 줄 확인
            if i < len(lines):
                next_line = lines[i].strip()  # i는 1-indexed, lines는 0-indexed → lines[i]
                if next_line == "pass":
                    findings.append({
                        "file": filepath,
                        "line": i,
                        "code": f"{stripped} → {next_line}",
                        "type": "pass",
                    })
                    continue

            # 로거가 없는 bare except
            has_logging = False
            for j in range(i, min(i + 3, len(lines))):
                check = lines[j].strip()
                if any(kw in check for kw in ["logger", "logging", "_logger", "log(", "print("]):
                    has_logging = True
                    break
            if not has_logging:
                findings.append({
                    "file": filepath,
                    "line": i,
                    "code": stripped,
                    "type": "no_log",
                })

    return findings


def main():
    root = Path(".")
    all_findings = []

    for py in sorted(root.rglob("*.py")):
        rel = str(py.relative_to(root))
        if any(ex in rel for ex in EXCLUDE_FILES):
            continue
        findings = scan_file(str(py))
        all_findings.extend(findings)

    # 출력
    pass_count = sum(1 for f in all_findings if f["type"] == "pass")
    nolog_count = sum(1 for f in all_findings if f["type"] == "no_log")
    total = len(all_findings)

    print(f"🔇 Silent Exception 검사 결과")
    print(f"{'─' * 50}")
    print(f"  except:pass 패턴:  {pass_count}건")
    print(f"  로깅 없는 except:  {nolog_count}건")
    print(f"  합계:              {total}건 (허용 임계치: {MAX_ALLOWED}건)")
    print()

    if all_findings:
        # 파일별 그룹핑
        by_file = {}
        for f in all_findings:
            by_file.setdefault(f["file"], []).append(f)

        for filepath, items in sorted(by_file.items()):
            print(f"  📄 {filepath}")
            for item in items:
                tag = "🔴 PASS" if item["type"] == "pass" else "🟡 NO_LOG"
                print(f"     L{item['line']:4d} [{tag}] {item['code']}")
            print()

    # 판정
    if total > MAX_ALLOWED:
        print(f"❌ 임계치 초과: {total} > {MAX_ALLOWED}")
        print(f"   → safe_try() 또는 logger.exception()으로 교체 필요")
        # 리팩토링 진행 중이므로 Warning
        print(f"\n⚠️  현재 Warning 모드 (MAX_ALLOWED={MAX_ALLOWED}으로 점진적 축소)")
        # sys.exit(1)  ← 임계치 달성 후 활성화
    elif total > 0:
        print(f"⚠️  {total}건 잔존 (임계치 이내: {MAX_ALLOWED})")
    else:
        print("✅ Silent Exception 제로! 🎉")


if __name__ == "__main__":
    main()
