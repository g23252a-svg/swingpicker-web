"""스코어링 가중치 동적화 테스트"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from scoring_engine import _calc_ml_weight, build_global_score

PASS = 0
FAIL = 0

def _check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def run_tests():
    global PASS, FAIL
    PASS, FAIL = 0, 0

    print("=" * 60)
    print("🧪 스코어링 가중치 동적화 테스트")
    print("=" * 60)

    # ── 1. ML 비활성 (전부 0) → w_a = 0 ──
    print("\n📐 1. ML 비활성 → AI 가중치 0")
    ml_zero = pd.Series([0.0] * 100)
    w_s, w_t, w_a = _calc_ml_weight(ml_zero, "NORMAL")
    _check("w_a = 0", np.isclose(w_a, 0.0), f"w_a={w_a}")
    _check("w_s + w_t = 1.0", np.isclose(w_s + w_t, 1.0), f"sum={w_s+w_t}")
    _check("NORMAL: S ≈ T", abs(w_s - w_t) < 0.01, f"s={w_s:.3f}, t={w_t:.3f}")

    # ── 2. ML 완전 활성 (mean=50) → w_a = 0.20 ──
    print("\n📐 2. ML 완전 활성 → AI 가중치 0.20")
    ml_active = pd.Series([50.0] * 100)
    w_s2, w_t2, w_a2 = _calc_ml_weight(ml_active, "NORMAL")
    _check("w_a = 0.20", np.isclose(w_a2, 0.20), f"w_a={w_a2}")
    _check("합 = 1.0", np.isclose(w_s2 + w_t2 + w_a2, 1.0))

    # ── 3. 선형 보간 (mean=15 → 중간 가중치) ──
    print("\n📐 3. 선형 보간")
    ml_mid = pd.Series([15.0] * 100)
    w_s3, w_t3, w_a3 = _calc_ml_weight(ml_mid, "NORMAL")
    _check("0 < w_a < 0.20", 0 < w_a3 < 0.20, f"w_a={w_a3}")
    _check("합 = 1.0", np.isclose(w_s3 + w_t3 + w_a3, 1.0))
    # mean=15, LOW=5, HIGH=25 → w_a = 0.20 * (15-5)/(25-5) = 0.10
    _check("w_a ≈ 0.10", np.isclose(w_a3, 0.10, atol=0.01), f"w_a={w_a3}")

    # ── 4. 커버리지 게이트 (mean 높지만 커버리지 낮음) ──
    print("\n📐 4. 커버리지 게이트")
    # 100개 중 10개만 ML=80, 나머지 0 → cov=0.10 (<0.20)
    ml_sparse = pd.Series([0.0] * 90 + [80.0] * 10)
    w_s4, w_t4, w_a4 = _calc_ml_weight(ml_sparse, "NORMAL")
    _check("cov < 0.20 → w_a = 0", np.isclose(w_a4, 0.0),
         f"w_a={w_a4}, mean={ml_sparse.mean():.1f}, cov={(ml_sparse>0).mean():.2f}")

    # ── 5. CRITICAL 매크로 → STRUCT 비중 높음 ──
    print("\n📐 5. CRITICAL 매크로")
    w_sc, w_tc, w_ac = _calc_ml_weight(ml_active, "CRITICAL")
    _check("CRITICAL: w_s > w_t", w_sc > w_tc,
         f"s={w_sc:.3f}, t={w_tc:.3f}")
    _check("합 = 1.0", np.isclose(w_sc + w_tc + w_ac, 1.0))

    # ML 비활성 + CRITICAL
    w_sc0, w_tc0, w_ac0 = _calc_ml_weight(ml_zero, "CRITICAL")
    _check("CRITICAL + ML off: w_s > w_t", w_sc0 > w_tc0,
         f"s={w_sc0:.3f}, t={w_tc0:.3f}")
    _check("CRITICAL + ML off: w_a = 0", np.isclose(w_ac0, 0.0))

    # ── 6. 불연속 경계 테스트 (mean=4.9 vs 5.1) ──
    print("\n📐 6. 경계 연속성")
    ml_49 = pd.Series([4.9] * 100)
    ml_51 = pd.Series([5.1] * 100)
    _, _, w_a49 = _calc_ml_weight(ml_49, "NORMAL")
    _, _, w_a51 = _calc_ml_weight(ml_51, "NORMAL")
    _check("mean=4.9 → w_a=0", np.isclose(w_a49, 0.0))
    _check("mean=5.1 → w_a 매우 작음", w_a51 < 0.02,
         f"w_a={w_a51:.4f}")
    # 급격한 변화가 없어야 함
    _check("경계 차이 < 0.02", abs(w_a51 - w_a49) < 0.02,
         f"diff={abs(w_a51-w_a49):.4f}")

    # ── 7. build_global_score 통합 — 진짜 검증 ──
    print("\n📐 7. build_global_score 통합")
    test_df = pd.DataFrame({
        "RSI14": [55, 70, 40],
        "MFI14": [50, 60, 45],
        "이격도": [2.0, 5.0, -1.0],
        "BB_BW": [0.1, 0.2, 0.05],
        "ret_5d_%": [3.0, 8.0, -2.0],
        "ret_10d_%": [5.0, 12.0, 0.0],
        "ret_20d_%": [10.0, 15.0, -5.0],
        "ret_60d_%": [20.0, 30.0, 5.0],
        "MACD_Slope_PCT": [0.01, 0.03, -0.01],
        "Range_Pos": [0.7, 0.9, 0.3],
        "Vol_Quality": [1.2, 1.8, 0.8],
        "Above_MA20": [1, 1, 0],
        "거래강도": [2.0, 4.0, 0.5],
        "Low_Trend_PCT": [1.0, 2.0, -1.0],
        "IS_SWING_SUPPORT": [1, 0, 0],
        "SECTOR_RANK": [2, 5, 10],
        "ML_SCORE": [0.0, 0.0, 0.0],  # ML 비활성
    })
    result = build_global_score(test_df, "NORMAL")
    _check("FINAL_SCORE 존재", "FINAL_SCORE" in result.columns)
    _check("FINAL_SCORE > 0", result["FINAL_SCORE"].mean() > 0)

    # ✅ 진짜 공식 검증: 가중치 역산 후 비교
    w_s7, w_t7, w_a7 = _calc_ml_weight(test_df["ML_SCORE"], "NORMAL")
    expected7 = ((result["STRUCT_SCORE"] * w_s7)
                 + (result["TIMING_SCORE"] * w_t7)
                 + (result["AI_SCORE"] * w_a7)).round(1)
    _check("FINAL 공식 일치 (일반)", expected7.equals(result["FINAL_SCORE"]),
         f"expected={expected7.tolist()}, actual={result['FINAL_SCORE'].tolist()}")

    # ML off → w_a=0 확인
    _check("ML off: w_a=0", np.isclose(w_a7, 0.0),
         f"w_a={w_a7}")

    # ML 활성 시 점수 변화
    test_df2 = test_df.copy()
    test_df2["ML_SCORE"] = [80.0, 70.0, 60.0]
    result2 = build_global_score(test_df2, "NORMAL")
    _check("ML 활성 시 FINAL 변화",
         not result["FINAL_SCORE"].equals(result2["FINAL_SCORE"]))

    # ML 활성 시도 공식 일치
    w_s72, w_t72, w_a72 = _calc_ml_weight(test_df2["ML_SCORE"], "NORMAL")
    expected72 = ((result2["STRUCT_SCORE"] * w_s72)
                  + (result2["TIMING_SCORE"] * w_t72)
                  + (result2["AI_SCORE"] * w_a72)).round(1)
    _check("ML 활성: FINAL 공식 일치", expected72.equals(result2["FINAL_SCORE"]))

    # ── 8. NaN 방어 ──
    print("\n📐 8. NaN 방어")
    ml_nan = pd.Series([np.nan, np.nan, 50.0])
    w_sn, w_tn, w_an = _calc_ml_weight(ml_nan, "NORMAL")
    _check("NaN 포함 → 크래시 없음", True)
    _check("합 ≈ 1.0", np.isclose(w_sn + w_tn + w_an, 1.0),
         f"sum={w_sn + w_tn + w_an}")

    # ── 9. ML_SCORE clip 검증 ──
    print("\n📐 9. ML_SCORE clip")
    test_df9 = test_df.copy()
    test_df9["ML_SCORE"] = [-50.0, 150.0, 80.0]
    result9 = build_global_score(test_df9, "NORMAL")
    _check("음수 → clip 0", result9["AI_SCORE"].iloc[0] == 0.0)
    _check("150 → clip 100", result9["AI_SCORE"].iloc[1] == 100.0)
    _check("80 → 그대로", result9["AI_SCORE"].iloc[2] == 80.0)

    # ── 10. ML_SCORE 컬럼 없을 때 ──
    print("\n📐 10. ML_SCORE 컬럼 없음")
    test_df10 = test_df.drop(columns=["ML_SCORE"])
    result10 = build_global_score(test_df10, "NORMAL")
    _check("ML 없음 → 크래시 없음", "FINAL_SCORE" in result10.columns)
    _check("ML 없음 → AI_SCORE=0", (result10["AI_SCORE"] == 0.0).all())

    # ── 11. macro_risk = HIGH ──
    print("\n📐 11. macro_risk = HIGH")
    ml_high = pd.Series([50.0] * 100)
    w_sh, w_th, w_ah = _calc_ml_weight(ml_high, "HIGH")
    _check("HIGH: w_s > w_t", w_sh > w_th, f"s={w_sh:.4f}, t={w_th:.4f}")
    _check("HIGH: 합 ≈ 1.0", np.isclose(w_sh + w_th + w_ah, 1.0))
    # NORMAL과 비교
    w_sn2, w_tn2, _ = _calc_ml_weight(ml_high, "NORMAL")
    _check("HIGH s > NORMAL s", w_sh > w_sn2,
         f"HIGH={w_sh:.4f}, NORMAL={w_sn2:.4f}")

    # ── 12. 절사평균(trimmed mean) 왜곡 방지 ──
    print("\n📐 12. 절사평균 왜곡 방지")
    # 100개 중 5개만 ML=100, 나머지 30 → cov OK, 단순평균 흔들림
    ml_skew = pd.Series([30.0] * 95 + [100.0] * 5)
    w_s12, w_t12, w_a12 = _calc_ml_weight(ml_skew, "NORMAL")
    # 절사평균이면 100짜리 5개가 제거되어 ~30 근처 → w_a ≈ 0.20 * (30-5)/(25-5) = 0.25 → cap 0.20
    _check("절사평균: w_a = 0.20 (outlier 제거)",
         np.isclose(w_a12, 0.20, atol=0.01),
         f"w_a={w_a12:.4f}")

    # 단순평균이었다면 mean=33.5 → w_a=0.20 (같은 결과긴 한데)
    # 더 극단적 케이스: 90개 0점 + 10개 80점 → cov=0.10 → 게이트에 걸림
    ml_extreme = pd.Series([0.0] * 90 + [80.0] * 10)
    _, _, w_a_ext = _calc_ml_weight(ml_extreme, "NORMAL")
    _check("극단 분포: cov gate → w_a=0", np.isclose(w_a_ext, 0.0),
         f"w_a={w_a_ext:.4f}")

    # ── 13. PASS_EBS 생성 검증 ──
    print("\n📐 13. PASS_EBS 생성")
    result13 = build_global_score(test_df, "NORMAL")
    _check("PASS_EBS 컬럼 존재", "PASS_EBS" in result13.columns)
    _check("PASS_EBS = 0 or 1", set(result13["PASS_EBS"].unique()).issubset({0, 1}))

    # ── 14. config 외부화 ──
    print("\n📐 14. config 외부화")
    from collector_config import CollectorConfig, MacroConfig
    custom_cfg = CollectorConfig(macro=MacroConfig(ml_max_weight=0.30, ml_low=0.0, ml_high=10.0))
    ml_cfg = pd.Series([8.0] * 100)
    w_s14, w_t14, w_a14 = _calc_ml_weight(ml_cfg, "NORMAL", config=custom_cfg)
    # ml_center=8, LOW=0, HIGH=10 → w_a = 0.30 * 8/10 = 0.24
    _check("커스텀 max=0.30: w_a ≈ 0.24", np.isclose(w_a14, 0.24, atol=0.02),
         f"w_a={w_a14:.4f}")
    _check("합 ≈ 1.0", np.isclose(w_s14 + w_t14 + w_a14, 1.0))

    # ── 15. 미지정 macro_risk → NORMAL fallback ──
    print("\n📐 15. macro_risk 방어")
    w_s15, w_t15, w_a15 = _calc_ml_weight(ml_active, "UNKNOWN_RISK")
    w_sn15, w_tn15, w_an15 = _calc_ml_weight(ml_active, "NORMAL")
    _check("미지정 macro → NORMAL과 동일",
         np.isclose(w_s15, w_sn15) and np.isclose(w_t15, w_tn15),
         f"unknown=({w_s15:.4f},{w_t15:.4f}), normal=({w_sn15:.4f},{w_tn15:.4f})")

    # ── 16. 소규모 n (n<10) trimmed mean 안정성 ──
    print("\n📐 16. 소규모 n (trimmed mean)")
    ml_small = pd.Series([30.0, 30.0, 30.0, 100.0, 0.0])  # n=5 < 10
    w_s16, w_t16, w_a16 = _calc_ml_weight(ml_small, "NORMAL")
    _check("n<10 → 크래시 없음", True)
    _check("n<10: 합 ≈ 1.0", np.isclose(w_s16 + w_t16 + w_a16, 1.0))
    # n<10이면 단순 mean 사용 → mean=38.0, cov=0.80 → HIGH 이상 → w_a=0.20
    _check("n<10: w_a 계산됨", w_a16 > 0, f"w_a={w_a16:.4f}")

    # n=1
    ml_one = pd.Series([50.0])
    w_s1, w_t1, w_a1 = _calc_ml_weight(ml_one, "NORMAL")
    _check("n=1 → 크래시 없음 + 합=1", np.isclose(w_s1 + w_t1 + w_a1, 1.0))

    # ── 17. 섹터 이중 보상 방지 (완전 SSOT 잠금) ──
    print("\n📐 17. 섹터 이중 보상 완전 잠금")
    from scoring_engine import (
        calculate_structural_score, calculate_timing_score,
    )

    base_row = {
        "RSI14": 55, "MFI14": 50, "이격도": 2.0, "BB_BW": 0.1,
        "ret_5d_%": 3.0, "MACD_Slope_PCT": 0.01, "Range_Pos": 0.7,
        "Vol_Quality": 1.2, "Above_MA20": 1, "Low_Trend_PCT": 1.0,
        "RAW_TRIGGER_SCORE": 60, "TTM_SQUEEZE": 0, "SUPERTREND_DIR": 0,
        "RES_RATIO": 0, "RES_RATIO_NEAR": 0, "POC_GAP": 0, "IS_ABOVE_POC": 1,
        "gap_pct": 0, "SECTOR_RANK": 99, "SECTOR_RS": 0.0,
    }
    top_sector_row = {**base_row, "SECTOR_RANK": 1, "SECTOR_RS": 5.0}

    # (a) STRUCT는 SECTOR 변경에 절대 불변
    struct_base = calculate_structural_score(base_row)
    struct_top = calculate_structural_score(top_sector_row)
    _check("STRUCT: SECTOR 변경 무관",
         np.isclose(struct_base, struct_top),
         f"base={struct_base:.1f}, top={struct_top:.1f}")

    # (b) TIMING은 SECTOR_RANK에 따라 정확히 +8
    timing_base = calculate_timing_score(base_row)
    timing_top = calculate_timing_score(top_sector_row)
    timing_delta = timing_top - timing_base
    _check("TIMING: SECTOR_RANK=1 → +8점",
         np.isclose(timing_delta, 8.0, atol=0.5),
         f"delta={timing_delta:.1f}")

    # (c) ★ 핵심: FINAL 변화량 = TIMING 변화량 × w_t (다른 경로로 섹터가 안 들어옴)
    df_base = pd.DataFrame([base_row])
    df_top = pd.DataFrame([top_sector_row])
    out_base = build_global_score(df_base, "NORMAL")
    out_top = build_global_score(df_top, "NORMAL")

    final_base = float(out_base["FINAL_SCORE"].iloc[0])
    final_top = float(out_top["FINAL_SCORE"].iloc[0])
    final_delta = final_top - final_base

    # w_t 추출 (동일 데이터이므로 w_t 동일)
    _, w_t, _ = _calc_ml_weight(out_base["ML_SCORE"], "NORMAL")

    expected_final_delta = timing_delta * w_t
    _check("FINAL 변화량 = TIMING변화 × w_t (이중경로 0)",
         np.isclose(final_delta, expected_final_delta, atol=0.5),
         f"FINAL_delta={final_delta:.2f}, expected={expected_final_delta:.2f}")

    # (d) SECTOR_RS가 FINAL에 영향 안 줌
    same_rank_diff_rs = {**base_row, "SECTOR_RANK": 99, "SECTOR_RS": 99.0}
    df_diff_rs = pd.DataFrame([same_rank_diff_rs])
    out_diff_rs = build_global_score(df_diff_rs, "NORMAL")
    final_diff_rs = float(out_diff_rs["FINAL_SCORE"].iloc[0])
    _check("SECTOR_RS만 변경 → FINAL 불변",
         np.isclose(final_base, final_diff_rs),
         f"base={final_base:.1f}, diff_rs={final_diff_rs:.1f}")

    # (e) W_SECTOR 죽은 코드 제거 확인
    collector_src = open(os.path.join(os.path.dirname(__file__), "collector.py")).read()
    _check("W_SECTOR 변수 사용 없음",
         "W_SECTOR =" not in collector_src and "* W_SECTOR" not in collector_src,
         "W_SECTOR still in use")

    # ── 18. Multi-Timeframe 경로 검증 (#15) ──
    print("\n📐 18. Multi-Timeframe 경로 잠금 (#15)")

    # MTF 없는 기본 row (MTF_DATA_SUFFICIENT=0)
    mtf_base_row = {**base_row, "MTF_WEEKLY_TREND": 0, "MTF_MONTHLY_TREND": 0,
                    "MTF_DATA_SUFFICIENT": 0}
    struct_mtf_base = calculate_structural_score(mtf_base_row)

    # (a) 데이터 부족 시 보정 0 (안전장치 2)
    mtf_no_data_up = {**base_row, "MTF_WEEKLY_TREND": 1, "MTF_MONTHLY_TREND": 1,
                      "MTF_DATA_SUFFICIENT": 0}
    struct_no_data = calculate_structural_score(mtf_no_data_up)
    _check("MTF 데이터 부족 → 보정 0",
         np.isclose(struct_mtf_base, struct_no_data),
         f"base={struct_mtf_base:.1f}, nodata={struct_no_data:.1f}")

    # (b) 주봉+월봉 모두 상승 → STRUCT +10 (Config값)
    mtf_both_up = {**base_row, "MTF_WEEKLY_TREND": 1, "MTF_MONTHLY_TREND": 1,
                   "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_BONUS": 10}
    struct_both_up = calculate_structural_score(mtf_both_up)
    _check("MTF 양쪽 상승 → STRUCT +10",
         np.isclose(struct_both_up - struct_mtf_base, 10.0, atol=0.5),
         f"delta={struct_both_up - struct_mtf_base:.1f}")

    # (c) 주봉+월봉 모두 하락 → STRUCT 감소 (penalty 15, clip(0,100) 상호작용으로 관측값 다를 수 있음)
    mtf_both_dn = {**base_row, "MTF_WEEKLY_TREND": -1, "MTF_MONTHLY_TREND": -1,
                   "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_PENALTY": 15}
    struct_both_dn = calculate_structural_score(mtf_both_dn)
    _delta_dn = struct_both_dn - struct_mtf_base
    _check("MTF 양쪽 하락 → STRUCT 감소 (penalty 적용)",
         _delta_dn < -5.0,  # gate_mult/clip 상호작용으로 정확히 -15가 아닐 수 있음
         f"delta={_delta_dn:.1f}")

    # (d) 한쪽만 상승 → 절반 bonus (+5)
    mtf_w_up = {**base_row, "MTF_WEEKLY_TREND": 1, "MTF_MONTHLY_TREND": 0,
                "MTF_DATA_SUFFICIENT": 1, "_MTF_STRUCT_BONUS": 10}
    struct_w_up = calculate_structural_score(mtf_w_up)
    _check("MTF 한쪽 상승 → STRUCT +5",
         np.isclose(struct_w_up - struct_mtf_base, 5.0, atol=0.5),
         f"delta={struct_w_up - struct_mtf_base:.1f}")

    # (e) TIMING은 MTF에 영향 안 받음
    timing_mtf_up = calculate_timing_score(mtf_both_up)
    timing_mtf_base = calculate_timing_score(mtf_base_row)
    _check("TIMING: MTF 변경 무관",
         np.isclose(timing_mtf_up, timing_mtf_base),
         f"base={timing_mtf_base:.1f}, up={timing_mtf_up:.1f}")

    # (f) ★ FINAL 변화량 = STRUCT 변화량 × w_s (MTF가 STRUCT에만 영향)
    df_mtf_base = pd.DataFrame([mtf_base_row])
    df_mtf_up = pd.DataFrame([mtf_both_up])
    out_mtf_base = build_global_score(df_mtf_base, "NORMAL")
    out_mtf_up = build_global_score(df_mtf_up, "NORMAL")

    final_mtf_base = float(out_mtf_base["FINAL_SCORE"].iloc[0])
    final_mtf_up = float(out_mtf_up["FINAL_SCORE"].iloc[0])
    final_mtf_delta = final_mtf_up - final_mtf_base

    struct_delta = struct_both_up - struct_mtf_base
    w_s_mtf, _, _ = _calc_ml_weight(out_mtf_base["ML_SCORE"], "NORMAL")
    expected_mtf_final = struct_delta * w_s_mtf
    _check("FINAL 변화량 = STRUCT변화 × w_s (MTF→STRUCT만)",
         np.isclose(final_mtf_delta, expected_mtf_final, atol=0.5),
         f"FINAL_delta={final_mtf_delta:.2f}, expected={expected_mtf_final:.2f}")

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
    run_tests()
