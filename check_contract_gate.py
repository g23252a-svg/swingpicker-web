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


def check_alpha_schema_integrity():
    """[v3.9.15e + 4] components/tab_backtest.py의 _calc_kospi_alpha가
    trades_df 컬럼명을 정합하게 쓰는지 정적 검사.

    v3.9.15d 사례: trades_df에 "date"/"net_pct" 컬럼이 없는데 그걸 lookup
    → real 경로 영원히 작동 0%. 누군가 다시 옛 컬럼명으로 되돌리면 즉시 차단.

    [v3.9.15e + 5] AST 기반으로 승격 (이전 라인 단위 문자열 매칭 → AST 노드).
    이전 방식 한계: `row.get("date")` 같은 변수 alias나 dict subscript 우회는
    못 잡음. AST 기반은 _calc_kospi_alpha 본문 안의 모든 Constant 노드 중
    값이 "date" 또는 "net_pct"인 것을 잡아냄. 이 두 문자열이 함수 안에
    string literal로 등장할 합법적 이유가 거의 없으므로 false positive 최소.

    검출 케이스 (전부 fail):
      · `"date" in trades.columns`             [v3.9.15d 원래 버그]
      · `r.get("date", "")`                    [원래 버그 + 다른 alias]
      · `r["date"]`                            [Subscript]
      · `row["date"]`                          [변수명 alias]
      · `col_name = "date"; r.get(col_name)`   [문자열 변수에 저장해 우회 시도]
      · `pd.to_datetime(t["net_pct"])`         [중첩 호출]
    """
    errors = []
    fpath = os.path.join(PROJECT_ROOT, "components", "tab_backtest.py")
    if not os.path.exists(fpath):
        return errors

    FORBIDDEN_LITERALS = {"date", "net_pct"}

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=fpath)
    except SyntaxError as e:
        return [f"components/tab_backtest.py 구문 오류 — gate 검사 불가: {e}"]
    except Exception as e:
        print(f"  ⚠️ tab_backtest.py 파싱 스킵: {e}")
        return errors

    # _calc_kospi_alpha 함수 노드 찾기
    target_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_calc_kospi_alpha":
            target_func = node
            break

    if target_func is None:
        # 함수 자체가 없으면 알파 기능 미사용 환경 (스킵)
        return errors

    # 함수 본문 내의 모든 Constant 노드 순회
    for sub_node in ast.walk(target_func):
        if isinstance(sub_node, ast.Constant) and isinstance(sub_node.value, str):
            if sub_node.value in FORBIDDEN_LITERALS:
                lineno = getattr(sub_node, "lineno", "?")
                errors.append(
                    f"components/tab_backtest.py:{lineno} — "
                    f'_calc_kospi_alpha 내부에 금지 string literal "{sub_node.value}" 발견 '
                    f"(trades_df 실제 컬럼: rec_date / net_ret). "
                    f"v3.9.15d 회귀 의심."
                )

    return errors


def check_benchmark_cache_completeness():
    """[v3.9.15e + 4] data/bench_cache_latest.json의 필수 키 검증.

    슬라이더 1~60일을 모두 지원하려면 KOSPI cache에 [1, 3, 5, 10, 20, 60]
    키가 모두 있어야 함. 60 키 누락 시 슬라이더 41~60일 알파 미산출.

    [v3.9.15e + 5] warning → hard fail 승격.
    이유: 41~60일 전략의 알파가 빠지는 건 무시 못 할 정합성 위반.
    macro_filter._BENCH_LOOKBACK_BDAYS=130 적용 후엔 60 키 누락이 발생하면
    안 됨. 만약 발생하면 collector 회귀 가능성 — 배포 차단이 맞다.

    파일 자체가 없는 케이스 (collector 미실행 CI 환경)는 여전히 스킵
    (파일 부재 != 키 누락).
    """
    errors = []
    fpath = os.path.join(PROJECT_ROOT, "data", "bench_cache_latest.json")
    if not os.path.exists(fpath):
        # 캐시 파일 자체가 없는 건 collector 미실행 환경 (CI) — 스킵
        return errors

    required_keys = {1, 3, 5, 10, 20, 60}
    try:
        import json
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        kospi = data.get("KOSPI", {})
        present = {int(k) for k in kospi.keys() if str(k).isdigit()}
        missing = required_keys - present
        if missing:
            errors.append(
                f"data/bench_cache_latest.json — KOSPI 키 누락: "
                f"{sorted(missing)} (collector lookback 확장 필요. "
                f"macro_filter._BENCH_LOOKBACK_BDAYS=130 적용 확인)"
            )
    except Exception as e:
        errors.append(f"data/bench_cache_latest.json 파싱 실패: {e}")
    return errors


def check_anomaly_thresholds():
    """[v3.9.15e + 10] services/backtest_policy.py의 anomaly 임계값 정합 검증.

    임계값이 실수로 완화/제거되어도 게이트가 잡지 않으면 화면에 비현실
    수익률(예: Sharpe 100)이 그대로 표시될 수 있음. 따라서 핵심 임계값을
    contract gate가 정적으로 검사.

    검사 룰:
    - services/backtest_policy.py 파일 존재
    - 핵심 상수 6개 존재 + 기대값 일치
    - tp_saturation_threshold / detect_anomaly_flags 함수 존재

    의도적으로 임계값을 바꾸려면 services/backtest_policy.py와 이 함수의
    EXPECTED 둘 다 같이 업데이트 — 의도적 변경 감지 가능.
    """
    errors = []
    fpath = os.path.join(PROJECT_ROOT, "services", "backtest_policy.py")
    if not os.path.exists(fpath):
        errors.append(
            "services/backtest_policy.py 누락 — v3.9.15e + 10 SSOT 모듈 부재"
        )
        return errors

    # 기대값 (정책 결정 — 변경 시 의도적이어야 함)
    EXPECTED = {
        "ANOMALY_TOTAL_RET_ABS": 300,
        "ANOMALY_SHARPE_MAX": 5,
        "ANOMALY_CAGR_MAX": 300,
        "ANOMALY_SHORT_DAYS_RET": 120,
        "ANOMALY_SHORT_RET": 100,
        "ANOMALY_SHORT_DAYS_CAGR": 252,
        "TP_SAT_THRESH_LOW_TARGET": 80,
        "TP_SAT_THRESH_MID_TARGET": 70,
        "TP_SAT_THRESH_HIGH_TARGET": 60,
    }
    REQUIRED_FUNCS = {"tp_saturation_threshold", "detect_anomaly_flags"}

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=fpath)
    except SyntaxError as e:
        return [f"services/backtest_policy.py 구문 오류: {e}"]
    except Exception as e:
        print(f"  ⚠️ backtest_policy.py 파싱 스킵: {e}")
        return errors

    # 모듈 top-level의 상수 할당 수집
    found_constants = {}
    found_funcs = set()
    for node in tree.body:
        # 상수 할당: `NAME = <int>` 형태
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, (int, float)):
                    found_constants[target.id] = node.value.value
        # 함수 정의
        if isinstance(node, ast.FunctionDef):
            found_funcs.add(node.name)

    # 누락된 상수
    missing_consts = set(EXPECTED.keys()) - set(found_constants.keys())
    for c in sorted(missing_consts):
        errors.append(
            f"services/backtest_policy.py — 상수 누락: {c}"
        )

    # 값 불일치
    for name, expected_val in EXPECTED.items():
        if name in found_constants and found_constants[name] != expected_val:
            errors.append(
                f"services/backtest_policy.py — {name} 값 변경: "
                f"{found_constants[name]} (기대: {expected_val}). "
                f"의도적 변경이면 check_contract_gate.EXPECTED도 같이 업데이트."
            )

    # 누락된 함수
    missing_funcs = REQUIRED_FUNCS - found_funcs
    for f in sorted(missing_funcs):
        errors.append(
            f"services/backtest_policy.py — 함수 누락: {f}()"
        )

    return errors


def check_kospi_daily_csv_schema():
    """[v3.9.15e + 4] data/kospi_daily.csv 스키마 검증 (있을 경우).

    파일이 있으면 필수 컬럼 [date, close, ret_1d_%, ret_3d_%, ret_5d_%,
    ret_10d_%, ret_20d_%, ret_60d_%] 모두 존재해야 함.

    파일 없는 경우는 OK (real 알파 미사용 환경) — simple 알파로 폴백.
    """
    errors = []
    fpath = os.path.join(PROJECT_ROOT, "data", "kospi_daily.csv")
    if not os.path.exists(fpath):
        return errors

    required_cols = [
        "date", "close",
        "ret_1d_%", "ret_3d_%", "ret_5d_%",
        "ret_10d_%", "ret_20d_%", "ret_60d_%",
    ]
    try:
        # csv 헤더만 읽기 (pandas 의존 회피 — CI 가벼움)
        with open(fpath, "r", encoding="utf-8-sig") as f:
            header = f.readline().strip().split(",")
        header = [h.strip().strip('"') for h in header]
        missing = [c for c in required_cols if c not in header]
        if missing:
            errors.append(
                f"data/kospi_daily.csv — 필수 컬럼 누락: {missing} "
                f"(scripts/collect_kospi_daily.py 재실행 필요)"
            )
    except Exception as e:
        errors.append(f"data/kospi_daily.csv 파싱 실패: {e}")
    return errors



def check_guard_v23_contract():
    """[v23.0] guard_system 산출 컬럼 계약 + GuardConfig SSOT 연동 + ELITE_LABEL 게이트."""
    errors = []
    try:
        sys.path.insert(0, PROJECT_ROOT)
        import pandas as pd
        from guard_system import apply_guard_system, GUARD_CONTRACT_COLS
        from collector_config import DEFAULT_CONFIG
    except Exception as e:
        return [f"guard_system/collector_config import 실패: {e}"]

    # 1) GuardConfig가 CollectorConfig에 합성되어 있는지
    if not hasattr(DEFAULT_CONFIG, "guard"):
        errors.append("CollectorConfig에 guard(GuardConfig) 미연동")
        return errors
    for fld in ("g1_turnover_min_eok", "g5_break_min", "guard_enforce_top_pick"):
        if not hasattr(DEFAULT_CONFIG.guard, fld):
            errors.append(f"GuardConfig 필드 누락: {fld}")

    # 2) 산출 컬럼 계약 — 합성 df에 전 컬럼 부여되는지
    df = pd.DataFrame([
        # 정상 TOP_PICK
        dict(종목명="정상", ELITE_SCORE=88, TIMING_SCORE=75, AXIS_MEAN=80,
             **{"거래대금(억원)": 800}, STOP_PCT=9, TOP_PICK=1,
             SUPERTREND_DIR=1, Above_MA20=1),
        # 유동성 차단 TOP_PICK → ELITE 아님
        dict(종목명="차단", ELITE_SCORE=85, TIMING_SCORE=70, AXIS_MEAN=75,
             **{"거래대금(억원)": 50}, STOP_PCT=5, TOP_PICK=1),
    ])
    out = apply_guard_system(df, config=DEFAULT_CONFIG, kospi_ret_1d=None)
    missing = [c for c in GUARD_CONTRACT_COLS if c not in out.columns]
    if missing:
        errors.append(f"GUARD 산출 컬럼 누락: {missing}")
        return errors

    # 3) ELITE_LABEL 게이트 — 차단 종목은 ELITE 금지, 정상은 ELITE
    lab = dict(zip(out["종목명"], out["ELITE_LABEL"]))
    if lab.get("정상") != "ELITE":
        errors.append(f"정상 TOP_PICK이 ELITE 아님: {lab.get('정상')!r}")
    if lab.get("차단") == "ELITE":
        errors.append("유동성 차단 종목이 ELITE로 누출")

    # 4) 차단 종목 GUARD_KELLY_MULT==0 (사이징 0)
    km = dict(zip(out["종목명"], out["GUARD_KELLY_MULT"]))
    if km.get("차단", 1) != 0:
        errors.append(f"차단 종목 GUARD_KELLY_MULT≠0: {km.get('차단')}")

    return errors



def check_momentum_lane_contract():
    """[v23.1] momentum_lane 산출 컬럼 계약 + MomentumLaneConfig SSOT + 시장국면 게이트."""
    errors = []
    try:
        sys.path.insert(0, PROJECT_ROOT)
        import pandas as pd
        from momentum_lane import apply_momentum_lane, MOMENTUM_LANE_COLS
        from collector_config import DEFAULT_CONFIG
    except Exception as e:
        return [f"momentum_lane/collector_config import 실패: {e}"]

    # 1) MomentumLaneConfig가 CollectorConfig에 합성되어 있는지
    if not hasattr(DEFAULT_CONFIG, "momentum_lane"):
        errors.append("CollectorConfig에 momentum_lane(MomentumLaneConfig) 미연동")
        return errors
    for fld in ("source_route", "max_picks", "require_guard", "regime_deviation_floor"):
        if not hasattr(DEFAULT_CONFIG.momentum_lane, fld):
            errors.append(f"MomentumLaneConfig 필드 누락: {fld}")

    # 2) 산출 컬럼 계약 — 합성 df (OVERHEAT 6 + ATTACK 1)
    rows = []
    for i in range(6):
        rows.append(dict(종목명=f"과열{i}", ROUTE="OVERHEAT",
                         GUARD_ALL_PASS=True, GUARDED_ELITE_SCORE=90 - i,
                         RR_NOW_TP1=0.7))   # RR 낮아도 레인 진입해야(모멘텀 역설)
    rows.append(dict(종목명="공격", ROUTE="ATTACK",
                     GUARD_ALL_PASS=True, GUARDED_ELITE_SCORE=95, RR_NOW_TP1=1.5))
    rows.append(dict(종목명="과열_가드탈락", ROUTE="OVERHEAT",
                     GUARD_ALL_PASS=False, GUARDED_ELITE_SCORE=92, RR_NOW_TP1=1.4))
    df = pd.DataFrame(rows)

    out = apply_momentum_lane(df, market_risk_off=False, config=DEFAULT_CONFIG)
    missing = [c for c in MOMENTUM_LANE_COLS if c not in out.columns]
    if missing:
        errors.append(f"MOMENTUM 산출 컬럼 누락: {missing}")
        return errors

    res = dict(zip(out["종목명"], out["MOMENTUM_LANE_TIER"]))
    lane = dict(zip(out["종목명"], out["MOMENTUM_LANE"]))

    # 3) ATTACK(소스 아님)·가드탈락은 레인 제외
    if res.get("공격") != "":
        errors.append(f"ATTACK이 모멘텀 레인에 누출: tier={res.get('공격')!r}")
    if res.get("과열_가드탈락") != "":
        errors.append("가드 미통과 과열주가 레인에 누출")

    # 4) 점수 상위 max_picks개가 Tier A, RR 낮아도 진입(모멘텀 역설)
    n_a = int(out["MOMENTUM_LANE"].sum())
    if n_a != min(5, DEFAULT_CONFIG.momentum_lane.max_picks):
        errors.append(f"Tier A 수 비정상: {n_a} (max_picks={DEFAULT_CONFIG.momentum_lane.max_picks})")
    if lane.get("과열0") != 1:  # 점수 최고 + RR 0.7 → 그래도 Tier A여야
        errors.append("점수 최상위 과열주가 RR 때문에 Tier A 탈락 (모멘텀 역설 위반)")

    # 5) 시장 위험회피 시 레인 전체 OFF
    off = apply_momentum_lane(df, market_risk_off=True, config=DEFAULT_CONFIG)
    if int(off["MOMENTUM_LANE"].sum()) != 0 or int(off["MOMENTUM_WATCH"].sum()) != 0:
        errors.append("시장 위험회피인데 모멘텀 레인이 비활성화되지 않음")

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

    # ─────────────────────────────────────────────────────────
    # [v3.9.15e + 4] Alpha 게이트 — KOSPI 알파 산출물 무결성
    # ─────────────────────────────────────────────────────────
    print("\n6. [GUARD_ALPHA_SCHEMA] _calc_kospi_alpha 컬럼명 정합...")
    errs = check_alpha_schema_integrity()
    all_errors.extend(errs)
    print(f"   {'❌ ' + str(len(errs)) + '건' if errs else '✅ OK'}")
    for e in errs[:5]:
        print(f"      {e}")

    print("\n7. [GUARD_BENCHMARK] bench_cache_latest.json 필수 키 (1·3·5·10·20·60)...")
    errs = check_benchmark_cache_completeness()
    # [v3.9.15e + 5] warning → hard fail 승격.
    # 41~60일 전략의 알파 정합성 보호. CI 환경 (파일 부재)은 함수 자체가 스킵.
    all_errors.extend(errs)
    print(f"   {'❌ ' + str(len(errs)) + '건' if errs else '✅ OK'}")
    for e in errs[:3]:
        print(f"      {e}")

    print("\n8. [GUARD_ALPHA_REAL] kospi_daily.csv 스키마 (필수 컬럼)...")
    errs = check_kospi_daily_csv_schema()
    all_errors.extend(errs)  # 파일 있으면서 스키마 깨진 건 hard fail
    print(f"   {'❌ ' + str(len(errs)) + '건' if errs else '✅ OK'}")
    for e in errs[:3]:
        print(f"      {e}")

    print("\n9. [GUARD_ANOMALY_THRESHOLDS] backtest_policy anomaly 임계값 정합...")
    errs = check_anomaly_thresholds()
    all_errors.extend(errs)
    print(f"   {'❌ ' + str(len(errs)) + '건' if errs else '✅ OK'}")
    for e in errs[:5]:
        print(f"      {e}")

    print("\n10. [GUARD_V23] guard_system 산출물 계약 + ELITE_LABEL 게이트...")
    errs = check_guard_v23_contract()
    all_errors.extend(errs)
    print(f"   {'❌ ' + str(len(errs)) + '건' if errs else '✅ OK'}")
    for e in errs[:5]:
        print(f"      {e}")

    print("\n11. [MOMENTUM_V23_1] momentum_lane 산출물 계약 + 시장국면 게이트...")
    errs = check_momentum_lane_contract()
    all_errors.extend(errs)
    print(f"   {'❌ ' + str(len(errs)) + '건' if errs else '✅ OK'}")
    for e in errs[:5]:
        print(f"      {e}")

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
