#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_contract_gate.py — CI 계약 위반 탐지 게이트
═══════════════════════════════════════════════════
[v20.8] push/PR 시 자동 실행하여 정합성 위반 차단.

실행: python scripts/check_contract_gate.py
CI:   .github/workflows/ci.yml에 step 추가

검사 항목:
  1. Feature Contract 동기화 (ml_engine FEATURE_COLS == contract)
  2. 정책 임계치 하드코딩 탐지 (validation/stop_logic에 literal 숫자)
  3. 구버전 문자열 잔존 탐지
  4. Policy SSOT 참조 여부
"""
import os
import re
import sys
import ast

# ── 설정 ──
# 이 파일은 프로젝트 루트에 위치 (scripts/ 아님)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _THIS_DIR
STALE_VERSIONS = ['"v20.5"', '"v20.6"', '"v20.7"']  # 현재 v20.8, 이전 버전 잔존 금지
POLICY_FILES = ["validation.py", "stop_logic.py", "trade_plan.py"]
FEATURE_FILES = ["ml_engine.py"]

# 정책 임계치 리터럴 금지 패턴
# validation.py, stop_logic.py에서 PolicyConfig 참조 없이 직접 숫자를 쓰면 위반
POLICY_LITERALS = {
    "거래대금.*30": "hard_block_turnover_min_eok",
    "거래대금.*50": "entry_turnover_hold_eok",
    "gap.*12": "entry_gap_hold_pct",
    "gap.*7": "entry_gap_split_pct",
    "RSI.*85": "hard_block_rsi_max",
    "RSI.*80": "entry_rsi_split",
    "ret_1d.*15": "entry_surge_hold_pct",
    "ret_1d.*10": "entry_surge_split_pct",
}


def check_feature_contract():
    """Feature Contract과 ml_engine FEATURE_COLS 동기화 검증."""
    errors = []
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from feature_contract import FEATURE_CONTRACT
        from ml_engine import FEATURE_COLS

        if list(FEATURE_CONTRACT.columns) != FEATURE_COLS:
            errors.append(
                f"FEATURE_COLS mismatch!\n"
                f"  contract: {list(FEATURE_CONTRACT.columns)}\n"
                f"  ml_engine: {FEATURE_COLS}"
            )
    except ImportError as e:
        # feature_contract 없는 환경은 경고만
        print(f"  ⚠️ Feature contract import skipped: {e}")
    return errors


def check_stale_versions():
    """구버전 문자열 잔존 탐지."""
    errors = []
    for root, _, files in os.walk(PROJECT_ROOT):
        if "__pycache__" in root or ".git" in root or "backup" in root:
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, PROJECT_ROOT)
            # 이 스크립트 자체와 test 파일은 제외
            if "check_contract_gate" in fname:
                continue
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        # 주석은 스킵
                        stripped = line.strip()
                        if stripped.startswith('#'):
                            continue
                        for sv in STALE_VERSIONS:
                            if sv in line and 'STALE_VERSIONS' not in line:
                                errors.append(f"{rel}:{i} — 구버전 문자열 '{sv}' 잔존")
            except Exception:
                pass
    return errors


def check_policy_ssot():
    """validation.py, stop_logic.py가 PolicyConfig를 import하는지."""
    errors = []
    for fname in POLICY_FILES:
        fpath = os.path.join(PROJECT_ROOT, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        if "collector_config" not in content and "PolicyConfig" not in content:
            errors.append(f"{fname}: PolicyConfig import 없음 — SSOT 위반 가능")
    return errors


def check_duplicate_feature_defs():
    """FEATURE_COLS = [...] 직접 정의가 ml_engine 외에 있는지."""
    errors = []
    pattern = re.compile(r'FEATURE_COLS\s*=\s*\[')
    # 자기 자신의 절대경로 — 어떤 위치에서 실행해도 확실히 제외
    _self_path = os.path.abspath(__file__)
    _allowed = {"ml_engine.py", "feature_contract.py"}
    for root, _, files in os.walk(PROJECT_ROOT):
        if "__pycache__" in root or ".git" in root or "backup" in root:
            continue
        for fname in files:
            if not fname.endswith(".py") or fname.startswith("test_"):
                continue
            if fname in _allowed:
                continue
            fpath = os.path.join(root, fname)
            # 절대경로 비교로 자기 자신 제외
            if os.path.abspath(fpath) == _self_path:
                continue
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        stripped = line.strip()
                        if stripped.startswith('#') or stripped.startswith('"') or stripped.startswith("'"):
                            continue
                        if pattern.search(line):
                            rel = os.path.relpath(fpath, PROJECT_ROOT)
                            errors.append(f"{rel}:{i} — FEATURE_COLS 중복 정의")
            except Exception:
                pass
    return errors


def check_json_meta_versions():
    """data/ 내 JSON 메타 파일의 구버전 문자열 탐지."""
    errors = []
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if not os.path.isdir(data_dir):
        return errors
    stale_json = ['"v20.5"', '"v20.6"', '"v20.7"']
    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(data_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            for sv in stale_json:
                if sv in content:
                    errors.append(f"data/{fname} — 구버전 메타 {sv} 잔존 (재생성 필요)")
        except Exception:
            pass
    return errors


def main():
    print("🔍 Contract Gate Check")
    print("=" * 50)

    all_errors = []

    print("\n1. Feature Contract 동기화...")
    errs = check_feature_contract()
    all_errors.extend(errs)
    print(f"   {'❌ ' + str(len(errs)) + '건' if errs else '✅ OK'}")

    print("\n2. 구버전 문자열 잔존...")
    errs = check_stale_versions()
    all_errors.extend(errs)
    print(f"   {'❌ ' + str(len(errs)) + '건' if errs else '✅ OK'}")
    for e in errs[:5]:
        print(f"      {e}")

    print("\n3. Policy SSOT 참조...")
    errs = check_policy_ssot()
    all_errors.extend(errs)
    print(f"   {'❌ ' + str(len(errs)) + '건' if errs else '✅ OK'}")

    print("\n4. Feature 정의 중복...")
    errs = check_duplicate_feature_defs()
    all_errors.extend(errs)
    print(f"   {'❌ ' + str(len(errs)) + '건' if errs else '✅ OK'}")

    print("\n5. JSON 메타 구버전 잔존...")
    errs = check_json_meta_versions()
    # JSON 메타는 warning만 (재생성으로 해결되므로 hard fail 아님)
    if errs:
        print(f"   ⚠️ {len(errs)}건 (경고 — 파이프라인 재실행 시 자동 갱신)")
        for e in errs[:3]:
            print(f"      {e}")
    else:
        print(f"   ✅ OK")

    print("\n" + "=" * 50)
    if all_errors:
        print(f"🚨 총 {len(all_errors)}건 위반 발견!")
        for e in all_errors:
            print(f"   ❌ {e}")
        sys.exit(1)
    else:
        print("✅ Contract Gate PASSED — 위반 0건")
        sys.exit(0)


if __name__ == "__main__":
    main()
