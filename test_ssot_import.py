"""collector.py ↔ scoring_engine.py SSOT import smoke test"""

import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0

def test(name, cond, detail=""):
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
    print("🧪 SSOT Import Smoke Test")
    print("=" * 60)

    # ── 1. scoring_engine import 성공 ──
    print("\n📐 1. scoring_engine import")
    try:
        from scoring_engine import (
            determine_state,
            determine_state_dynamic,
            calculate_ebs_independent,
            calculate_structural_score,
            calculate_timing_score,
            build_global_score,
            _calc_ml_weight,
        )
        test("모든 함수 import 성공", True)
    except ImportError as e:
        test("모든 함수 import 성공", False, str(e))
        print("\n" + "=" * 60)
        print(f"🏁 결과: {PASS}/{PASS+FAIL} 통과 ({FAIL} 실패)")
        sys.exit(1)

    # ── 2. 순환 import 없음 ──
    print("\n📐 2. 순환 import 방어")
    import scoring_engine
    src = open(scoring_engine.__file__).read()
    test("scoring_engine에 collector import 없음",
         "import collector " not in src and "from collector " not in src
         and "from collector_config" in src)  # collector_config는 OK (SSOT)
    test("WEIGHT_CONFIG 완전 제거",
         "WEIGHT_CONFIG" not in src or "WEIGHT_CONFIG →" in src)  # 주석 OK, 변수 NO

    # ── 3. CollectorConfig SSOT 접근 ──
    print("\n📐 3. CollectorConfig SSOT 접근")
    from collector_config import DEFAULT_CONFIG, CollectorConfig
    test("DEFAULT_CONFIG 존재", isinstance(DEFAULT_CONFIG, CollectorConfig))
    test("ml_low 필드", hasattr(DEFAULT_CONFIG, 'ml_low'))
    test("macro_weights 필드", hasattr(DEFAULT_CONFIG, 'macro_weights'))
    test("snapshot 메서드", hasattr(DEFAULT_CONFIG, 'snapshot'))

    # ── 4. 함수 호출 smoke ──
    print("\n📐 4. 함수 호출 smoke")
    import pandas as pd
    import numpy as np

    ml = pd.Series([50.0] * 20)
    w_s, w_t, w_a = _calc_ml_weight(ml, "NORMAL")
    test("_calc_ml_weight 합=1", np.isclose(w_s + w_t + w_a, 1.0))

    row = {"RSI14": 50, "ret_1d_%": 2, "ret_5d_%": 3,
           "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5,
           "Vol_Quality": 1.0, "TIMING_SCORE": 50,
           "거래강도": 2.0, "Low_Trend_PCT": 0, "Above_MA20": 1,
           "거래대금(원)": 5e9, "외인순매수": 0, "개인순매수": 0}
    th = {"vol_q75": 1.2, "range_q75": 0.8}
    state = determine_state_dynamic(row, th)
    test("determine_state_dynamic 실행",
         state in {"ATTACK", "ARMED", "WAIT", "NEUTRAL", "OVERHEAT", "EXIT_WARNING"})

    test_df = pd.DataFrame({
        "RSI14": [55], "MFI14": [50], "이격도": [2.0], "BB_BW": [0.1],
        "ret_5d_%": [3.0], "ret_10d_%": [5.0], "ret_20d_%": [10.0], "ret_60d_%": [20.0],
        "MACD_Slope_PCT": [0.01], "Range_Pos": [0.7], "Vol_Quality": [1.2],
        "Above_MA20": [1], "거래강도": [2.0], "Low_Trend_PCT": [1.0],
        "IS_SWING_SUPPORT": [1], "SECTOR_RANK": [2], "ML_SCORE": [0.0],
    })
    result = build_global_score(test_df, "NORMAL")
    test("build_global_score 실행", "FINAL_SCORE" in result.columns)
    test("PASS_EBS 생성", "PASS_EBS" in result.columns)

    # ── 5. collector에 중복 정의 없음 ──
    print("\n📐 5. collector 중복 정의 없음")
    collector_src = open(os.path.join(os.path.dirname(__file__), "collector.py")).read()
    for fn in ["def calculate_ebs_independent", "def calculate_structural_score",
               "def calculate_timing_score", "def determine_state_dynamic"]:
        test(f"collector에 '{fn}' 없음", fn not in collector_src)

    bgs_defs = re.findall(r"^def build_global_score", collector_src, re.MULTILINE)
    test("collector에 'def build_global_score' 없음", len(bgs_defs) == 0,
         f"found {len(bgs_defs)}")

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
