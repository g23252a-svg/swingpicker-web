#!/usr/bin/env python3
"""
fix4_silent_exceptions.py — '침묵의 살인자' except 패턴 일괄 수정 스크립트
═══════════════════════════════════════════════════════════════════════════
실행 방법:
  python fix4_silent_exceptions.py

효과:
  - collector.py, main.py, scoring_engine.py 내
    `except Exception:` + `pass/return 0/return None` 패턴을
    `except Exception as e:` + `logger.warning(...)` 으로 자동 교체

  - bare `except:` → `except Exception as e:` + 로깅 추가

주의:
  - 실행 전 git commit 또는 백업 권장
  - 의도적으로 조용히 넘기는 패턴(예: 외국인 데이터 없는 종목)은 수동 확인 필요
"""

import re
import os


# ═══════════════════════════════════════════
#  패턴별 교체 규칙
# ═══════════════════════════════════════════

REPLACEMENTS = [
    # ── Pattern 1: except Exception:\n            pass ──
    # → except Exception as e:\n            logger.warning(f"[무시됨] {context}: {e}")
    {
        "pattern": r"except Exception:\s*\n(\s*)pass",
        "replacement": lambda m: f'except Exception as e:\n{m.group(1)}logger.warning(f"[무시됨] {{e}}", exc_info=True)',
        "desc": "except Exception: pass → 로깅 추가",
    },

    # ── Pattern 2: except:\n            pass ──
    {
        "pattern": r"except:\s*\n(\s*)pass",
        "replacement": lambda m: f'except Exception as e:\n{m.group(1)}logger.warning(f"[무시됨] {{e}}", exc_info=True)',
        "desc": "bare except: pass → 로깅 추가",
    },

    # ── Pattern 3: except Exception:\n            return 0.0 ──
    {
        "pattern": r"except Exception:\s*\n(\s*)return 0\.0",
        "replacement": lambda m: f'except Exception as e:\n{m.group(1)}logger.warning(f"[fallback→0.0] {{e}}", exc_info=True)\n{m.group(1)}return 0.0',
        "desc": "except → return 0.0 에 로깅 추가",
    },

    # ── Pattern 4: except Exception:\n            return 0 ──
    {
        "pattern": r"except Exception:\s*\n(\s*)return 0(?!\.)(?!\d)",
        "replacement": lambda m: f'except Exception as e:\n{m.group(1)}logger.warning(f"[fallback→0] {{e}}", exc_info=True)\n{m.group(1)}return 0',
        "desc": "except → return 0 에 로깅 추가",
    },

    # ── Pattern 5: except Exception:\n            return None ──
    {
        "pattern": r"except Exception:\s*\n(\s*)return None",
        "replacement": lambda m: f'except Exception as e:\n{m.group(1)}logger.warning(f"[fallback→None] {{e}}", exc_info=True)\n{m.group(1)}return None',
        "desc": "except → return None 에 로깅 추가",
    },

    # ── Pattern 6: except Exception:\n            return [] ──
    {
        "pattern": r"except Exception:\s*\n(\s*)return \[\]",
        "replacement": lambda m: f'except Exception as e:\n{m.group(1)}logger.warning(f"[fallback→[]] {{e}}", exc_info=True)\n{m.group(1)}return []',
        "desc": "except → return [] 에 로깅 추가",
    },

    # ── Pattern 7: except Exception:\n            return 0, 0 ──
    {
        "pattern": r"except Exception:\s*\n(\s*)return 0, 0",
        "replacement": lambda m: f'except Exception as e:\n{m.group(1)}logger.warning(f"[fallback→(0,0)] {{e}}", exc_info=True)\n{m.group(1)}return 0, 0',
        "desc": "except → return 0,0 에 로깅 추가",
    },
]

# ═══════════════════════════════════════════
#  Logger import 보장
# ═══════════════════════════════════════════

LOGGER_IMPORTS = {
    "collector.py": 'logger = logging.getLogger("collector")',
    "main.py": 'logger = logging.getLogger("main")',
    "scoring_engine.py": 'import logging\n_se_logger = logging.getLogger("scoring_engine")',
}


def ensure_logger(content: str, filename: str) -> str:
    """파일에 logger가 없으면 추가"""
    if "logging.getLogger" in content:
        return content  # 이미 있음

    # import 블록 끝에 추가
    import_line = LOGGER_IMPORTS.get(filename, 'import logging\nlogger = logging.getLogger(__name__)')
    # 첫 번째 빈 줄 이후에 삽입
    lines = content.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i + 1
    lines.insert(insert_idx, import_line)
    return "\n".join(lines)


def apply_fixes(filepath: str):
    """파일에 모든 교체 규칙 적용"""
    if not os.path.exists(filepath):
        print(f"  ⏭️  {filepath} 없음, 스킵")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    original = content

    # 1. Logger import 보장
    content = ensure_logger(content, filename)

    # 2. 패턴 교체
    total_fixes = 0
    for rule in REPLACEMENTS:
        matches = list(re.finditer(rule["pattern"], content))
        if matches:
            content = re.sub(rule["pattern"], rule["replacement"], content)
            total_fixes += len(matches)
            print(f"  ✅ {rule['desc']}: {len(matches)}건")

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  📝 {filepath} 수정 완료 (총 {total_fixes}건)")
    else:
        print(f"  ℹ️  {filepath} 변경 없음")


# ═══════════════════════════════════════════
#  수동 확인 필요 항목 리스트
# ═══════════════════════════════════════════

MANUAL_REVIEW = """
═══════════════════════════════════════════════════
  ⚠️  수동 확인 필요 항목
═══════════════════════════════════════════════════

다음은 자동 치환되지 않는 패턴입니다. 직접 확인해주세요:

1. collector.py:2031 — "외국인 데이터 없는 종목은 정상"
   → 의도적 무시이므로 logger.debug()로 변경 권장
   
2. collector.py:1628 — bare `except:` + pass (이름 매핑)
   → 어떤 에러인지 모르므로 반드시 로깅 추가

3. scoring_engine.py:305 — determine_state_dynamic의 최종 fallback
   → 전체 함수가 실패하는 케이스이므로 logger.error() 권장

4. main.py의 async 함수 내 except Exception:
   → UI 컴포넌트 에러는 사용자에게 ui.notify()로 알려주는 게 좋음

═══════════════════════════════════════════════════
  💡 추천 로깅 구조 (프로젝트 전체)
═══════════════════════════════════════════════════

# config.py 또는 main.py 상단에 추가:
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("swingpicker.log", encoding="utf-8"),
    ]
)

# 이렇게 하면 모든 logger.warning() 호출이
# 1) 콘솔에 출력되고
# 2) swingpicker.log 파일에도 기록됩니다.
# exc_info=True 옵션이 있으면 Traceback도 함께 기록됩니다.
"""


if __name__ == "__main__":
    print("🔧 Silent Exception 자동 수정 시작...\n")

    targets = ["collector.py", "main.py", "scoring_engine.py"]
    for t in targets:
        print(f"\n── {t} ──")
        apply_fixes(t)

    print(MANUAL_REVIEW)
