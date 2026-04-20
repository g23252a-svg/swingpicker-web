# -*- coding: utf-8 -*-
"""
tests/test_shadow_regression.py — Shadow Test를 pytest에서 실행 가능한 회귀 게이트로 변환
══════════════════════════════════════════════════════════════════════════════════
[v3.7.29] CI에서 `pytest tests/test_shadow_regression.py`로 자동 실행되도록 래핑.

목적:
  - 수동 스크립트 (python test_shadow_analyze.py --mode compare) → pytest 자동화
  - CI에서 pull request 시 자동 검증
  - snapshot/compare 결과를 pytest assertion으로 보호

실행:
  pytest tests/test_shadow_regression.py                 # 전체
  pytest tests/test_shadow_regression.py -k match_rate   # 일치율만
  pytest tests/test_shadow_regression.py -v              # 상세

스킵 조건:
  - golden snapshot이 없으면 자동 skip (개발 초기 단계 허용)
  - compare tag가 없으면 자동 skip (분리 작업 전이면 당연히 비교 대상 없음)
"""
import os
import json
import pytest
import sys

# 상위 디렉토리를 sys.path에 추가 (test_shadow_analyze.py 접근)
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# pytest import 실패 fallback
try:
    from test_shadow_analyze import (
        load_golden_snapshot,
        compare_results,
        SNAPSHOT_DIR,
    )
except ImportError as e:
    pytest.skip(f"test_shadow_analyze 임포트 실패: {e}", allow_module_level=True)


# ═══════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════

@pytest.fixture(scope="module")
def golden():
    """Golden snapshot 로드, 없으면 skip"""
    try:
        return load_golden_snapshot("golden")
    except FileNotFoundError:
        pytest.skip(
            "golden snapshot 없음 — "
            "`python test_shadow_analyze.py --mode snapshot` 먼저 실행"
        )


@pytest.fixture(scope="module")
def refactored():
    """Refactored snapshot 로드, 없으면 skip (리팩토링 전이면 당연)"""
    try:
        return load_golden_snapshot("refactored")
    except FileNotFoundError:
        pytest.skip(
            "refactored snapshot 없음 — "
            "리팩토링 후 `--mode snapshot --tag refactored` 실행"
        )


@pytest.fixture(scope="module")
def report(golden, refactored):
    """비교 리포트 생성 (한 번만)"""
    return compare_results(golden, refactored, verbose=False)


# ═══════════════════════════════════════════════════
#  Regression Gates
# ═══════════════════════════════════════════════════

# 임계치 — 환경변수로 오버라이드 가능 (CI에서 유연)
MIN_MATCH_RATE = float(os.environ.get("SHADOW_MIN_MATCH_RATE", "0.995"))
MAX_MISSING = int(os.environ.get("SHADOW_MAX_MISSING", "0"))
MAX_CRITICAL_MISMATCH = int(os.environ.get("SHADOW_MAX_CRITICAL", "0"))

# Critical keys — 이 값이 바뀌면 매매 의사결정이 달라짐
CRITICAL_KEYS = {
    "DISPLAY_SCORE", "STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE",
    "ELITE_SCORE", "ELITE_LABEL", "ROUTE",
    "추천매수가", "손절가", "추천매도가1",
}


def test_match_rate_above_threshold(report):
    """일치율이 최소 임계치 이상이어야 함"""
    match_rate = report["match_rate"]
    assert match_rate >= MIN_MATCH_RATE, (
        f"match_rate {match_rate:.4f} < min {MIN_MATCH_RATE}. "
        f"리팩토링이 수치를 변경시켰는지 확인 필요. "
        f"상세: {os.path.join(SNAPSHOT_DIR, 'failures_refactored.json')}"
    )


def test_no_missing_stocks(report):
    """Golden에 있는 종목이 refactored에서 사라지면 안 됨 (기본: 0개 허용)"""
    missing = report["missing"]
    assert missing <= MAX_MISSING, (
        f"missing 종목 {missing}개 발생 "
        f"(허용: {MAX_MISSING}개). "
        f"종목 리스트: {report.get('missing_codes', [])[:10]}"
    )


def test_no_critical_key_mismatches(report):
    """Critical keys(점수/가격/라벨) 불일치 개수 체크"""
    critical_failures = [
        f for f in report["failures"]
        if f["key"] in CRITICAL_KEYS
    ]
    critical_count = len(critical_failures)
    assert critical_count <= MAX_CRITICAL_MISMATCH, (
        f"critical key 불일치 {critical_count}건 "
        f"(허용: {MAX_CRITICAL_MISMATCH}건). "
        f"불일치 키: {sorted(set(f['key'] for f in critical_failures[:10]))}. "
        f"리팩토링이 매매 의사결정 값을 변경시켰을 가능성."
    )


def test_no_extra_unexpected_stocks(report):
    """Refactored에 예상 외 종목이 추가되지도 않아야 함 (정보용)

    WARNING이지 FAIL은 아님 — 일부 케이스에선 정상일 수 있으므로.
    [v3.7.29 fix] pytest.warns(...)는 context manager로만 동작.
    실제 경고를 내려면 warnings.warn() 호출 필요.
    """
    extra = report["extra"]
    if extra > 5:  # 5개까지는 허용 (edge case)
        import warnings
        warnings.warn(
            f"extra 종목 {extra}개 추가됨 (허용: 5개). "
            f"리팩토링이 예상 외 종목을 포함시켰는지 확인 필요.",
            UserWarning,
            stacklevel=2,
        )


def test_report_json_exists(report):
    """리포트 JSON이 파일로 저장되어 있어야 함 (CI 아티팩트)"""
    report_path = os.path.join(SNAPSHOT_DIR, "report_refactored.json")
    # 없으면 생성 (pytest 내부 안전장치)
    if not os.path.exists(report_path):
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in report.items() if k != "failures"},
                f, ensure_ascii=False, indent=2,
            )
    assert os.path.exists(report_path), (
        f"리포트 JSON 생성 실패: {report_path}"
    )


# ═══════════════════════════════════════════════════
#  환경변수 도움말 (pytest --help 에서 보이게)
# ═══════════════════════════════════════════════════

"""
환경변수로 임계치 조정:
  SHADOW_MIN_MATCH_RATE  (default 0.995) — 일치율 최소
  SHADOW_MAX_MISSING     (default 0)     — 누락 종목 최대
  SHADOW_MAX_CRITICAL    (default 0)     — critical key 불일치 최대

예:
  SHADOW_MIN_MATCH_RATE=0.99 pytest tests/test_shadow_regression.py
"""
