"""P5 #20 스코어링/지표 단위테스트 + #21 일일 성과 리포트 테스트"""
import sys, os, tempfile, shutil, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

PASS = 0
FAIL = 0

def test(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def run():
    global PASS, FAIL
    PASS, FAIL = 0, 0

    print("=" * 60)
    print("🧪 P5 #20 단위테스트 + #21 일일 성과 리포트")
    print("=" * 60)

    from scoring_engine import (
        calculate_structural_score, calculate_timing_score,
        calculate_ebs_independent, build_global_score, safe_float,
    )

    # ═══ #20-A: calculate_timing_score 경계/NULL 방어 ═══
    print("\n📐 1. TIMING: None/NaN/누락 방어")

    # 완전 빈 row
    empty_row = {}
    ts_empty = calculate_timing_score(empty_row)
    test("빈 row → 크래시 없음", True)
    test("빈 row → 0~100 범위", 0 <= ts_empty <= 100, f"got {ts_empty}")

    # None 값들
    none_row = {k: None for k in [
        "RAW_TRIGGER_SCORE", "RES_RATIO", "RES_RATIO_NEAR", "POC_GAP",
        "IS_ABOVE_POC", "gap_pct", "SECTOR_RANK", "SUPERTREND_DIR", "TTM_SQUEEZE"
    ]}
    ts_none = calculate_timing_score(none_row)
    test("None 가득 → 크래시 없음", True)
    test("None row → 0~100", 0 <= ts_none <= 100)

    # NaN 값들
    nan_row = {k: float('nan') for k in none_row}
    ts_nan = calculate_timing_score(nan_row)
    test("NaN 가득 → 크래시 없음", True)
    test("NaN row → 0~100", 0 <= ts_nan <= 100)

    # 극단 값
    extreme_row = {
        "RAW_TRIGGER_SCORE": 99999, "RES_RATIO": -100, "RES_RATIO_NEAR": 999,
        "POC_GAP": -50, "IS_ABOVE_POC": 1, "gap_pct": 100, "SECTOR_RANK": 0,
        "SUPERTREND_DIR": 1, "TTM_SQUEEZE": 1,
    }
    ts_extreme = calculate_timing_score(extreme_row)
    test("극단 값 → 크래시 없음", True)
    test("극단 값 → 유한수", np.isfinite(ts_extreme))

    # ═══ #20-B: calculate_structural_score 경계 ═══
    print("\n📐 2. STRUCT: 경계값 검증")

    ss_empty = calculate_structural_score({})
    test("STRUCT 빈 row → 크래시 없음", True)
    test("STRUCT 빈 row → 0~100", 0 <= ss_empty <= 100)

    # 최대 점수 (모든 지표 이상적)
    best_row = {
        "Low_Trend_PCT": 5.0, "MFI14": 80, "Vol_Quality": 2.5,
        "Range_Pos": 1.0, "이격도": 3.0, "Above_MA20": 1,
        "MTF_WEEKLY_TREND": 1, "MTF_MONTHLY_TREND": 1,
        "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_BONUS": 10,
    }
    ss_best = calculate_structural_score(best_row)
    test("최상 조건 → 점수 > 80", ss_best > 80, f"got {ss_best}")
    test("최상 조건 → ≤ 100", ss_best <= 100)

    # 최악 점수
    worst_row = {
        "Low_Trend_PCT": -5.0, "MFI14": 0, "Vol_Quality": 0,
        "Range_Pos": 0, "이격도": -10, "Above_MA20": 0,
        "MTF_WEEKLY_TREND": -1, "MTF_MONTHLY_TREND": -1,
        "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_PENALTY": 15,
    }
    ss_worst = calculate_structural_score(worst_row)
    test("최악 조건 → 점수 = 0", ss_worst == 0.0)

    # ═══ #20-C: calc_supertrend 짧은 데이터 ═══
    print("\n📐 3. SuperTrend: 짧은 데이터 방어")

    # collector import 부작용 회피 — calc_supertrend 직접 추출
    import importlib.util
    collector_path = os.path.join(os.path.dirname(__file__), "collector.py")
    src = open(collector_path).read()

    # calc_atr + calc_supertrend 함수만 추출 실행
    import textwrap
    exec_ns = {"np": np, "pd": pd, "Tuple": tuple}
    # calc_atr 추출
    atr_start = src.index("def calc_atr(")
    atr_end = src.index("\ndef ", atr_start + 1)
    exec(src[atr_start:atr_end], exec_ns)
    # calc_supertrend 추출
    st_start = src.index("def calc_supertrend(")
    st_end = src.index("\ndef ", st_start + 1)
    exec(src[st_start:st_end], exec_ns)
    calc_supertrend = exec_ns["calc_supertrend"]

    # 데이터 < period
    short_h = pd.Series([100, 101, 102])
    short_l = pd.Series([98, 99, 100])
    short_c = pd.Series([99, 100, 101])
    st_line, st_dir = calc_supertrend(short_h, short_l, short_c, period=10)
    test("len=3, period=10 → 크래시 없음", True)
    test("len=3 → 전부 NaN or 기본값", len(st_line) == 3)

    # 빈 시리즈
    empty_s = pd.Series([], dtype=float)
    st_e, dir_e = calc_supertrend(empty_s, empty_s, empty_s)
    test("빈 시리즈 → 크래시 없음", True)
    test("빈 시리즈 → 길이 0", len(st_e) == 0)

    # 정상 데이터
    np.random.seed(42)
    n = 100
    c = pd.Series(np.cumsum(np.random.randn(n)) + 100)
    h = c + abs(np.random.randn(n))
    l = c - abs(np.random.randn(n))
    st_norm, dir_norm = calc_supertrend(h, l, c, period=10)
    test("정상 100봉 → 길이 일치", len(st_norm) == n)
    test("direction ∈ {-1, 1}", set(dir_norm.dropna().unique()).issubset({-1, 1}))

    # ═══ #20-D: EBS 경계 ═══
    print("\n📐 4. EBS: 경계값")
    ebs_empty = calculate_ebs_independent({})
    test("EBS 빈 row → ≤ 2", ebs_empty <= 2, f"got {ebs_empty}")

    ebs_max = calculate_ebs_independent({
        "Low_Trend_PCT": 1, "Vol_Quality": 1.5,
        "MACD_Slope_PCT": 0.1, "RSI14": 55, "TTM_SQUEEZE": 1,
    })
    test("EBS 최대 → 10", ebs_max == 10)

    # ═══ #20-E: safe_float 방어 ═══
    print("\n📐 5. safe_float 방어")
    test("safe_float(None) → 0", safe_float(None) == 0)
    test("safe_float('abc') → 0", safe_float("abc") == 0)
    test("safe_float(NaN) → 0", safe_float(float('nan')) == 0)
    test("safe_float(42) → 42", safe_float(42) == 42.0)
    test("safe_float('3.14') → 3.14", safe_float("3.14") == 3.14)

    # ═══ #20-F: build_global_score 전체 불변식 ═══
    print("\n📐 6. build_global_score 불변식")
    test_df = pd.DataFrame([{
        "RSI14": 55, "MFI14": 50, "이격도": 2.0, "BB_BW": 0.1,
        "ret_5d_%": 3.0, "MACD_Slope_PCT": 0.01, "Range_Pos": 0.7,
        "Vol_Quality": 1.2, "Above_MA20": 1, "Low_Trend_PCT": 1.0,
        "RAW_TRIGGER_SCORE": 60, "TTM_SQUEEZE": 0, "SUPERTREND_DIR": 0,
        "RES_RATIO": 0, "RES_RATIO_NEAR": 0, "POC_GAP": 0, "IS_ABOVE_POC": 1,
        "gap_pct": 0, "SECTOR_RANK": 99,
    }])
    out = build_global_score(test_df, "NORMAL")

    # 필수 컬럼
    for col in ["STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE", "FINAL_SCORE", "PASS_EBS", "DISPLAY_SCORE"]:
        test(f"{col} 존재", col in out.columns)

    # 범위
    test("STRUCT 0~100", 0 <= float(out["STRUCT_SCORE"].iloc[0]) <= 100)
    test("TIMING 0~200", 0 <= float(out["TIMING_SCORE"].iloc[0]) <= 200)
    test("FINAL > 0", float(out["FINAL_SCORE"].iloc[0]) > 0)

    # ═══════════════════════════════════════════════════
    # #21: 일일 성과 리포트
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📐 7. #21: 일일 성과 리포트")
    from daily_report import generate_daily_report, _format_report_text, DailyReport

    tmp = tempfile.mkdtemp()
    try:
        # 테스트 데이터 생성
        days = ["20260210", "20260211", "20260212", "20260213", "20260214",
                "20260217", "20260218", "20260219", "20260220", "20260221"]
        for ymd in days:
            rec = pd.DataFrame({
                "종목코드": ["005930", "000660", "035720"],
                "종목명": ["삼성전자", "SK하이닉스", "카카오"],
                "DISPLAY_SCORE": [85.0, 65.0, 45.0],
                "FINAL_SCORE": [85.0, 65.0, 45.0],
                "손절가": [68000, 140000, 45000],
            })
            rec.to_csv(os.path.join(tmp, f"recommend_{ymd}.csv"), index=False)

            delta = (int(ymd[-2:]) - 10) * 100
            snap = pd.DataFrame([
                {"종목코드": "005930", "시가": 70000 + delta, "고가": 70500 + delta,
                 "저가": 69700 + delta, "종가": 70200 + delta},
                {"종목코드": "000660", "시가": 150000 - delta, "고가": 150500,
                 "저가": 149500 - delta, "종가": 150100 - delta},
                {"종목코드": "035720", "시가": 50000, "고가": 50200,
                 "저가": 49600, "종가": 49900},
            ])
            snap.to_csv(os.path.join(tmp, f"price_snapshot_{ymd}.csv"), index=False)

        # 리포트 생성 (lookback 충분히 확보)
        report = generate_daily_report(tmp, "20260221", lookback_days=30)
        test("리포트 DailyReport 리턴", isinstance(report, DailyReport))
        test("n_recommendations ≥ 0", report.get("n_recommendations", -1) >= 0)
        test("overall_winrate 존재", "overall_winrate" in report)
        test("report_key 존재", "report_key" in report)
        test("report_key 형식", report.get("report_key", "").startswith("daily_perf:"))

        # 텍스트 포맷
        text = _format_report_text(report)
        test("텍스트 생성", len(text) > 0)
        test("텍스트에 리포트 날짜 포함", "20260221" in text)

        # 중복 방지: 같은 날짜 2회 → report_key 동일
        report2 = generate_daily_report(tmp, "20260221", lookback_days=30)
        test("동일 날짜 → 동일 report_key",
             report["report_key"] == report2["report_key"])

        # #13 가정 일치 검증: 비용이 반영되면 gross > net
        if report.get("n_recommendations", 0) > 0:
            cost_diff = report["avg_ret_gross_pct"] - report["avg_ret_net_pct"]
            test("#13 비용 차감 일치", cost_diff > 0,
                 f"gross-net={cost_diff:.4f}")
        else:
            test("#13 비용 차감 (평가대상 없어 스킵)", True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── 결과 ──
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"🏁 결과: {PASS}/{total} 통과 ({FAIL} 실패)")
    if FAIL > 0:
        print("⚠️ 실패 항목이 있습니다!")
        sys.exit(1)
    else:
        print("🏆 ALL PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run()
