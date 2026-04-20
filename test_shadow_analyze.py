# -*- coding: utf-8 -*-
"""
test_shadow_analyze.py — analyze_ticker 분리를 위한 섀도우 테스트
═══════════════════════════════════════════════════════════════════
[v3.2] God Function 분리의 사전 작업.
[v3.7.27+] CONFIG_SNAPSHOT CSV → JSON 분리 후 테스트 흐름 호환
[v3.7.29] CONFIG migration 완료 — load_config_snapshot() 통해 참조

목적:
  1. 현재 analyze_ticker의 입출력을 스냅샷으로 기록 (golden snapshot)
  2. 향후 분리된 함수들의 조합 결과와 golden snapshot을 100% 비교
  3. 수치 차이가 있으면 정확히 어떤 키에서 발생했는지 리포트

Config 참조 정책 (v3.7.29):
  · CSV 행 데이터에는 CONFIG_VERSION 문자열만 있음
  · 전체 config snapshot이 필요하면 반드시 load_config_snapshot() 헬퍼 사용
  · 예: from pipeline_finalize import load_config_snapshot
        snapshot = load_config_snapshot(trade_ymd)

사용법:
  # Step 1: 현재 코드의 golden snapshot 생성
  python test_shadow_analyze.py --mode snapshot

  # Step 2: 함수 분리 후 비교 검증 (CI 친화 — exit code 0/1)
  python test_shadow_analyze.py --mode compare --min-match-rate 0.995

  # Step 3: (선택) 특정 종목만 디버그
  python test_shadow_analyze.py --mode debug --code 005930

CI/자동화 모드 (v3.7.29 신규):
  · --min-match-rate: 이 수치 미만이면 exit code 1 (CI 실패)
  · --report-json:   비교 결과를 머신 리더블 JSON으로 저장
  · --quiet:         최소 출력 (CI 로그 절약)
"""
import os
import json
import sys
import argparse
import time
import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

# [v3.7.29] logger 사용 — 운영 모드에서 레벨 제어 가능
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  1. 스냅샷 I/O
# ═══════════════════════════════════════════════════

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "_shadow_snapshots")


def _serialize_value(v):
    """JSON 직렬화를 위한 타입 변환"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def save_golden_snapshot(results: List[Dict[str, Any]], tag: str = "golden") -> str:
    """analyze_ticker 결과 리스트를 JSON 스냅샷으로 저장"""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    
    # 직렬화
    serialized = []
    for row in results:
        serialized.append({k: _serialize_value(v) for k, v in row.items()})
    
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=1)
    
    print(f"💾 스냅샷 저장: {path} ({len(serialized)}건)")
    return path


def load_golden_snapshot(tag: str = "golden") -> List[Dict[str, Any]]:
    """저장된 스냅샷 로드"""
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{tag}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"스냅샷 없음: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════
#  2. 비교 엔진
# ═══════════════════════════════════════════════════

# 수치 비교 시 허용 오차 (부동소수점 오차 + 타이밍 차이 허용)
NUMERIC_RTOL = 1e-6   # 상대 오차
NUMERIC_ATOL = 1e-4   # 절대 오차

# 비교에서 제외할 키 (실행 시점에 따라 달라지는 값)
# [v3.7.27+28] CONFIG_SNAPSHOT은 이제 CSV에 없음 (data/config_snapshot_YYYYMMDD.json으로 분리)
# 하위 호환을 위해 SKIP_KEYS에 유지 — 예전 CSV와 비교 시 자동 스킵됨
# 새 파이프라인 산출물 비교 시엔 두 CSV 모두 컬럼이 없으므로 무해
SKIP_KEYS = {
    "거래대금(원)",   # float 정밀도 이슈
    "CONFIG_SNAPSHOT",  # v3.7.27+에서 JSON 파일로 분리됨 (예전 CSV 호환용)
    "CONFIG_VERSION",   # 실행 시점 메타
}

# 완전 일치가 아닌 근사 비교가 필요한 수치 키
NUMERIC_KEYS_PATTERNS = [
    "ret_", "rel_", "RSI", "MFI", "BB_", "RR_", "STOP_", "Vol_",
    "Range_Pos", "MACD_", "Low_Trend", "거래강도", "V_POWER",
    "이격도", "VWAP_GAP", "POC_GAP", "RES_RATIO", "NEAR_THRES",
    "HMA20", "SUPERTREND_VAL", "SCORE", "점수",
]


def _is_numeric_key(key: str) -> bool:
    """수치 근사 비교가 필요한 키인지 판별"""
    return any(pat in key for pat in NUMERIC_KEYS_PATTERNS)


def _values_equal(key: str, v1, v2) -> Tuple[bool, str]:
    """
    두 값을 비교. 
    Returns: (일치 여부, 불일치 시 상세 메시지)
    """
    # 둘 다 None
    if v1 is None and v2 is None:
        return True, ""
    
    # 한쪽만 None
    if v1 is None or v2 is None:
        return False, f"None mismatch: golden={v1}, new={v2}"
    
    # 수치 비교
    if _is_numeric_key(key) or isinstance(v1, (int, float)) or isinstance(v2, (int, float)):
        try:
            f1, f2 = float(v1), float(v2)
            if math.isnan(f1) and math.isnan(f2):
                return True, ""
            if math.isclose(f1, f2, rel_tol=NUMERIC_RTOL, abs_tol=NUMERIC_ATOL):
                return True, ""
            return False, f"numeric diff: golden={f1}, new={f2}, delta={abs(f1-f2):.6f}"
        except (ValueError, TypeError):
            pass
    
    # 문자열/기타 비교
    if str(v1) != str(v2):
        return False, f"string diff: golden='{v1}', new='{v2}'"
    
    return True, ""


def compare_results(
    golden: List[Dict[str, Any]],
    current: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    golden snapshot과 현재 결과를 종목 단위로 비교.
    
    Returns: {
        "total": int,        # 비교 종목 수
        "passed": int,       # 완전 일치 종목 수
        "failed": int,       # 불일치 종목 수
        "missing": int,      # golden에만 있는 종목 수
        "extra": int,        # 현재에만 있는 종목 수
        "failures": [{code, key, msg}],  # 불일치 상세
        "match_rate": float, # 일치율 (0~1)
    }
    """
    # 종목코드 기준 인덱싱
    golden_map = {r["종목코드"]: r for r in golden if "종목코드" in r}
    current_map = {r["종목코드"]: r for r in current if "종목코드" in r}
    
    all_codes = set(golden_map.keys()) | set(current_map.keys())
    golden_only = set(golden_map.keys()) - set(current_map.keys())
    current_only = set(current_map.keys()) - set(golden_map.keys())
    common_codes = set(golden_map.keys()) & set(current_map.keys())
    
    failures = []
    passed = 0
    
    for code in sorted(common_codes):
        g_row = golden_map[code]
        c_row = current_map[code]
        
        all_keys = set(g_row.keys()) | set(c_row.keys())
        row_ok = True
        
        for key in sorted(all_keys):
            if key in SKIP_KEYS:
                continue
            
            g_val = g_row.get(key)
            c_val = c_row.get(key)
            
            ok, msg = _values_equal(key, g_val, c_val)
            if not ok:
                row_ok = False
                failures.append({
                    "code": code,
                    "key": key,
                    "golden": g_val,
                    "current": c_val,
                    "msg": msg,
                })
                if verbose:
                    print(f"  ❌ {code} | {key}: {msg}")
        
        if row_ok:
            passed += 1
        elif verbose:
            print(f"  ⚠️ {code}: 불일치 발견")
    
    total = len(common_codes)
    failed = total - passed
    match_rate = passed / total if total > 0 else 0.0
    
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "missing": len(golden_only),
        "extra": len(current_only),
        "missing_codes": sorted(golden_only),
        "extra_codes": sorted(current_only),
        "failures": failures,
        "match_rate": match_rate,
    }


def print_report(report: Dict[str, Any]) -> None:
    """비교 리포트 출력"""
    print("\n" + "=" * 60)
    print("📊 SHADOW TEST REPORT")
    print("=" * 60)
    print(f"  비교 종목 수:  {report['total']}")
    print(f"  ✅ 일치:       {report['passed']}")
    print(f"  ❌ 불일치:     {report['failed']}")
    print(f"  ➖ golden에만: {report['missing']} {report.get('missing_codes', [])[:5]}")
    print(f"  ➕ 신규:       {report['extra']} {report.get('extra_codes', [])[:5]}")
    print(f"  📈 일치율:     {report['match_rate']:.2%}")
    print("=" * 60)
    
    if report["failures"]:
        # 키별 불일치 빈도
        from collections import Counter
        key_freq = Counter(f["key"] for f in report["failures"])
        print("\n🔍 불일치 키 빈도 Top 10:")
        for key, cnt in key_freq.most_common(10):
            print(f"  {key}: {cnt}건")
        
        print(f"\n📋 전체 불일치: {len(report['failures'])}건")
        # 샘플 5건
        for f in report["failures"][:5]:
            print(f"  [{f['code']}] {f['key']}: {f['msg']}")
    
    # 결론
    if report["match_rate"] >= 1.0 and report["missing"] == 0:
        print("\n🎉 PERFECT MATCH — 분리 진행 안전!")
    elif report["match_rate"] >= 0.99:
        print("\n✅ NEAR MATCH — 미세 차이 확인 후 진행 가능")
    elif report["match_rate"] >= 0.95:
        print("\n⚠️ CAUTION — 불일치 항목 검토 필요")
    else:
        print("\n🚨 FAIL — 분리 보류, 불일치 원인 수정 필요")


# ═══════════════════════════════════════════════════
#  3. 실행 모드
# ═══════════════════════════════════════════════════

def run_snapshot_mode(tag: str = "golden"):
    """현재 analyze_ticker로 스냅샷 생성 (golden 또는 refactored).

    [v3.7.29] config_snapshot 연결:
      - 이전엔 CONFIG_SNAPSHOT 컬럼이 CSV에 있어 별도 로드 불필요했음
      - 지금은 JSON에서 읽어서 스냅샷 메타에 포함 → 재현성 보장
    """
    from collector import (
        analyze_ticker, collect_ohlcv_parallel, build_name_map,
        build_sector_map, get_market_cap_map, _get_top_filtered,
        _get_kospi_kosdaq_sets, _get_bench_returns, _get_investor_maps,
        find_latest_valid_date, _has_ohlcv_and_mcap, now_kst,
    )
    from collector_config import DEFAULT_CONFIG as _CFG
    # [v3.7.29] 새 migration 경로 사용 — 실제 참조 흐름 연결
    try:
        from pipeline_finalize import load_config_snapshot
    except ImportError:
        load_config_snapshot = lambda _: {}  # fallback
    
    OUT_DIR = _CFG.output_dir
    
    # 최신 거래일 탐색
    trade_ymd = find_latest_valid_date(_has_ohlcv_and_mcap)
    print(f"📅 거래일: {trade_ymd}")
    
    # 데이터 수집 (캐시 활용)
    print("📦 OHLCV 데이터 로딩...")
    top_df = _get_top_filtered(trade_ymd)
    tickers = top_df.index.tolist() if top_df is not None else []
    full_ohlcv_map = collect_ohlcv_parallel(trade_ymd, tickers[:50])  # 50개만 (테스트)
    
    # 메타데이터
    mcap_map = get_market_cap_map(trade_ymd)
    kospi_set, kosdaq_set = _get_kospi_kosdaq_sets(trade_ymd)
    name_map = build_name_map(trade_ymd, kospi_set | kosdaq_set)
    sector_map = build_sector_map()
    bench_map = _get_bench_returns(trade_ymd)
    inv_maps = _get_investor_maps(trade_ymd)
    
    # analyze_ticker 실행
    print(f"🔬 {len(full_ohlcv_map)}개 종목 분석 중...")
    results = []
    for t in tickers[:50]:
        code6 = str(t).zfill(6)
        df_t = full_ohlcv_map.get(code6)
        if df_t is None or df_t.empty:
            continue
        
        row = analyze_ticker(
            t, df_t, top_df, mcap_map,
            kospi_set, kosdaq_set, name_map, sector_map,
            bench_map, inv_maps,
        )
        if row is not None:
            results.append(row)
    
    print(f"✅ {len(results)}건 분석 완료")

    # [v3.7.29] config_snapshot을 메타로 포함 — 재현성 보장 + 실제 참조 흐름 연결
    # 이전엔 각 행의 CONFIG_SNAPSHOT 컬럼에 있던 걸 이제 snapshot 전체에 한 번만 포함
    try:
        cfg_snapshot = load_config_snapshot(trade_ymd)
        if cfg_snapshot:
            meta_file = os.path.join(SNAPSHOT_DIR, f"{tag}_config.json")
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({
                    "tag": tag,
                    "trade_ymd": trade_ymd,
                    "config_version": cfg_snapshot.get("config_version", "unknown"),
                    "n_records": len(results),
                    "config_snapshot": cfg_snapshot,
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"📋 config meta → {meta_file}")
    except Exception as e:
        logger.debug(f"config meta 저장 스킵: {e}")

    save_golden_snapshot(results, tag=tag)
    if tag == "golden":
        print("\n💡 다음 단계: analyze_ticker를 분리한 뒤 --mode snapshot --tag refactored 실행")
    else:
        print(f"\n💡 다음 단계: --mode compare --tag {tag} 로 비교 실행")


def run_compare_mode(
    new_tag: str = "refactored",
    min_match_rate: float = 0.995,
    critical_keys: Optional[List[str]] = None,
    report_json_path: Optional[str] = None,
    quiet: bool = False,
) -> int:
    """분리된 함수 결과와 golden snapshot 비교.

    [v3.7.29] CI Regression Gate 모드 추가.

    Args:
        new_tag: 비교 대상 태그 (e.g. 'refactored', 'v4_candidate')
        min_match_rate: 이 수치 미만이면 FAIL (exit code 1)
        critical_keys: 이 키 중 하나라도 불일치면 FAIL (중요 점수 보호)
        report_json_path: 리포트 JSON 출력 경로 (CI 파싱용)
        quiet: True면 요약만 출력 (CI 로그 절약)

    Returns:
        int: 0 = PASS, 1 = FAIL (exit code 규약)
    """
    # 기본 critical keys — 이 값들이 바뀌면 매매 의사결정 자체가 달라짐
    if critical_keys is None:
        critical_keys = [
            "DISPLAY_SCORE", "STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE",
            "ELITE_SCORE", "ELITE_LABEL", "ROUTE",
            "추천매수가", "손절가", "추천매도가1",
        ]

    try:
        golden = load_golden_snapshot("golden")
    except FileNotFoundError:
        print(f"❌ 'golden' 스냅샷이 없습니다. --mode snapshot 먼저 실행하세요.")
        return 1

    try:
        current = load_golden_snapshot(new_tag)
    except FileNotFoundError:
        print(f"❌ '{new_tag}' 스냅샷이 없습니다.")
        print("   분리된 함수로 스냅샷을 먼저 생성하세요:")
        print(f"   save_golden_snapshot(results, tag='{new_tag}')")
        return 1

    report = compare_results(golden, current, verbose=not quiet)

    if not quiet:
        print_report(report)

    # 리포트 JSON 저장 (기본 경로)
    default_report_path = os.path.join(SNAPSHOT_DIR, f"report_{new_tag}.json")
    out_path = report_json_path or default_report_path
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in report.items() if k != "failures"},
                      f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"report JSON 저장 실패: {e}")

    # 불일치 상세 저장 (기본 경로)
    if report["failures"]:
        fail_path = os.path.join(SNAPSHOT_DIR, f"failures_{new_tag}.json")
        try:
            with open(fail_path, "w", encoding="utf-8") as f:
                json.dump(report["failures"][:100], f, ensure_ascii=False, indent=2)
            if not quiet:
                print(f"\n📋 불일치 상세: {fail_path}")
        except Exception as e:
            logger.warning(f"failures JSON 저장 실패: {e}")

    # ═══════════════════════════════════════════════════
    #  [v3.7.29] Gate 판정 — pass/fail
    # ═══════════════════════════════════════════════════
    gate_failures = []

    # Gate 1: 일치율 임계치
    if report["match_rate"] < min_match_rate:
        gate_failures.append(
            f"match_rate {report['match_rate']:.4f} < min {min_match_rate}"
        )

    # Gate 2: critical keys 무결성
    critical_mismatches = [
        f for f in report["failures"]
        if f["key"] in critical_keys
    ]
    if critical_mismatches:
        gate_failures.append(
            f"critical keys 불일치 {len(critical_mismatches)}건: "
            f"{sorted(set(f['key'] for f in critical_mismatches[:5]))}"
        )

    # Gate 3: missing 종목 (golden에 있는데 current에 없음)
    if report["missing"] > 0:
        gate_failures.append(f"missing 종목 {report['missing']}개")

    # 결과 출력
    print("\n" + "=" * 60)
    print("🎯 REGRESSION GATE RESULT")
    print("=" * 60)
    print(f"  match_rate:     {report['match_rate']:.4f} (min: {min_match_rate})")
    print(f"  critical keys:  {len(critical_mismatches)}건 불일치")
    print(f"  missing:        {report['missing']}개")

    if gate_failures:
        print("\n🚨 GATE FAIL:")
        for gf in gate_failures:
            print(f"  · {gf}")
        print("\n→ 재작업 후 --mode compare 재실행 필요")
        return 1
    else:
        print("\n✅ GATE PASS — 리팩토링 안전")
        return 0


def run_debug_mode(code: str):
    """특정 종목의 analyze_ticker 결과를 상세 출력"""
    golden = load_golden_snapshot("golden")
    
    target = None
    for row in golden:
        if row.get("종목코드") == code.zfill(6):
            target = row
            break
    
    if target is None:
        print(f"❌ {code} 를 golden snapshot에서 찾을 수 없음")
        return
    
    print(f"\n📊 {target.get('종목명', '?')} ({code})")
    print("=" * 50)
    for k, v in sorted(target.items()):
        print(f"  {k:30s} = {v}")


# ═══════════════════════════════════════════════════
#  4. CLI
# ═══════════════════════════════════════════════════
# [v3.7.29] CI Regression Gate 모드 지원:
#   - exit code 0/1 로 pass/fail 신호
#   - --min-match-rate 임계치 설정 가능
#   - --report-json 으로 머신 리더블 출력
#   - --quiet 로 CI 로그 절약
# CI 예시:
#   python test_shadow_analyze.py --mode compare \
#       --tag v4_candidate --min-match-rate 0.995 \
#       --report-json /tmp/shadow_report.json --quiet
#   echo "Exit: $?"   # 0 = PASS, 1 = FAIL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="analyze_ticker 섀도우 테스트 (v3.7.29 CI Gate 지원)"
    )
    parser.add_argument(
        "--mode", choices=["snapshot", "compare", "debug"],
        required=True, help="실행 모드"
    )
    parser.add_argument("--code", default="", help="debug 모드: 종목코드 (6자리)")
    parser.add_argument(
        "--tag", default="refactored",
        help="compare 모드: 비교 대상 태그"
    )

    # [v3.7.29] CI 친화 플래그
    parser.add_argument(
        "--min-match-rate", type=float, default=0.995,
        help="compare 모드: 이 수치 미만이면 FAIL (default 0.995)"
    )
    parser.add_argument(
        "--critical-keys", default="",
        help="compare 모드: 쉼표 구분 critical key 목록. "
             "비어 있으면 기본 (점수/가격/라벨) 사용"
    )
    parser.add_argument(
        "--report-json", default="",
        help="compare 모드: 리포트 JSON 출력 경로 (CI 파싱용)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="최소 출력 (CI 로그 절약)"
    )

    args = parser.parse_args()

    # [v3.7.29] 로거 레벨 설정
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    exit_code = 0
    if args.mode == "snapshot":
        # [v3.7.29 fix] --tag refactored가 실제 refactored로 저장되도록
        # 이전: args.tag != "refactored" else "golden" (로직 꼬임)
        # 이후: args.tag 그대로 사용 (기본값은 argparse의 "refactored")
        # golden snapshot이 필요하면 명시적으로: --tag golden
        run_snapshot_mode(tag=args.tag)
    elif args.mode == "compare":
        critical_list = (
            [k.strip() for k in args.critical_keys.split(",") if k.strip()]
            if args.critical_keys
            else None  # 함수 기본값 사용
        )
        exit_code = run_compare_mode(
            new_tag=args.tag,
            min_match_rate=args.min_match_rate,
            critical_keys=critical_list,
            report_json_path=args.report_json or None,
            quiet=args.quiet,
        )
    elif args.mode == "debug":
        if not args.code:
            print("❌ --code 필수 (예: --code 005930)")
            exit_code = 1
        else:
            run_debug_mode(args.code)

    sys.exit(exit_code)
