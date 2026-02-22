"""TOXIC Filter 회귀 테스트 — determine_state_dynamic fail-safe"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from scoring_engine import determine_state_dynamic

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

    th = {"vol_q75": 1.2, "range_q75": 0.8}

    print("=" * 60)
    print("🧪 TOXIC Filter 회귀 테스트")
    print("=" * 60)

    # ── 1. 거래대금(원) 누락 시 EXIT_WARNING 도배 방지 ──
    print("\n📐 1. 거래대금 누락 → EXIT_WARNING 방지")
    # 이전 버그: turnover=1.0(default) → frg_ratio 폭발 → EXIT_WARNING
    row_no_turnover = {
        "RSI14": 50, "ret_1d_%": 6.0, "ret_5d_%": 10.0,
        "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5, "Vol_Quality": 1.0,
        "TIMING_SCORE": 50, "거래강도": 2.0, "Low_Trend_PCT": 0,
        "Above_MA20": 1,
        # 거래대금(원) 없음!
        "외인순매수": -500_000_000,  # 5억 매도
        "개인순매수": 500_000_000,   # 5억 매수
    }
    state = determine_state_dynamic(row_no_turnover, th)
    test("거래대금 누락 → EXIT_WARNING 아님", state != "EXIT_WARNING",
         f"state={state}")

    # ── 2. 거래대금=0 → EXIT_WARNING 방지 ──
    print("\n📐 2. 거래대금=0")
    row_zero = {**row_no_turnover, "거래대금(원)": 0}
    state2 = determine_state_dynamic(row_zero, th)
    test("거래대금=0 → EXIT_WARNING 아님", state2 != "EXIT_WARNING",
         f"state={state2}")

    # ── 3. 거래대금=1.0(기존 default) → EXIT_WARNING 방지 ──
    print("\n📐 3. 거래대금=1.0 (기존 default)")
    row_one = {**row_no_turnover, "거래대금(원)": 1.0}
    state3 = determine_state_dynamic(row_one, th)
    test("거래대금=1.0 → EXIT_WARNING 아님", state3 != "EXIT_WARNING",
         f"state={state3}")

    # ── 4. 정상 거래대금 + TOXIC 조건 → EXIT_WARNING 정상 작동 ──
    print("\n📐 4. 정상 TOXIC 조건 → EXIT_WARNING")
    row_toxic = {
        "RSI14": 50, "ret_1d_%": 6.0, "ret_5d_%": 10.0,
        "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5, "Vol_Quality": 1.0,
        "TIMING_SCORE": 50, "거래강도": 2.0, "Low_Trend_PCT": 0,
        "Above_MA20": 1,
        "거래대금(원)": 10_000_000_000,  # 100억 (정상)
        "외인순매수": -3_000_000_000,    # 30억 매도 (30% 이상)
        "개인순매수": 3_000_000_000,     # 30억 매수
    }
    state4 = determine_state_dynamic(row_toxic, th)
    test("정상 TOXIC → EXIT_WARNING", state4 == "EXIT_WARNING",
         f"state={state4}")

    # ── 5. vol_z TOXIC (turnover 무관) ──
    print("\n📐 5. vol_z TOXIC (거래대금 무관)")
    row_volz = {
        "RSI14": 50, "ret_1d_%": 12.0, "ret_5d_%": 5.0,
        "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5, "Vol_Quality": 1.0,
        "TIMING_SCORE": 50, "거래강도": 15.0, "Low_Trend_PCT": 0,
        "Above_MA20": 1,
        # 거래대금 없어도 vol_z 기반 TOXIC은 작동해야 함
    }
    state5 = determine_state_dynamic(row_volz, th)
    test("vol_z≥10 + r1≥10 → EXIT_WARNING", state5 == "EXIT_WARNING",
         f"state={state5}")

    # ── 6. 정상 종목 → ATTACK/ARMED/WAIT/NEUTRAL 중 하나 ──
    print("\n📐 6. 정상 종목")
    row_normal = {
        "RSI14": 55, "ret_1d_%": 2.0, "ret_5d_%": 5.0,
        "MACD_Slope_PCT": 0.02, "Range_Pos": 0.9, "Vol_Quality": 1.5,
        "TIMING_SCORE": 70, "거래강도": 3.0, "Low_Trend_PCT": 1.0,
        "Above_MA20": 1,
        "거래대금(원)": 5_000_000_000,
        "외인순매수": 100_000_000,
        "개인순매수": -50_000_000,
    }
    VALID_STATES = {"ATTACK", "ARMED", "WAIT", "NEUTRAL", "OVERHEAT"}
    state6 = determine_state_dynamic(row_normal, th)
    test("정상 → EXIT_WARNING 아님", state6 != "EXIT_WARNING",
         f"state={state6}")
    test("정상 → 유효 상태", state6 in VALID_STATES,
         f"state={state6}")

    # ── 7. 거래대금=NaN → fail-safe ──
    print("\n📐 7. 거래대금=NaN")
    import math
    row_nan = {**row_no_turnover, "거래대금(원)": float("nan")}
    state7 = determine_state_dynamic(row_nan, th)
    test("NaN → EXIT_WARNING 아님", state7 != "EXIT_WARNING",
         f"state={state7}")

    # ── 8. 거래대금=문자열 → fail-safe ──
    print("\n📐 8. 거래대금=문자열")
    row_str = {**row_no_turnover, "거래대금(원)": "invalid"}
    state8 = determine_state_dynamic(row_str, th)
    test("문자열 → 크래시 없음 + EXIT_WARNING 아님", state8 != "EXIT_WARNING",
         f"state={state8}")

    # ── 9. turnover_min_valid thresholds 커스텀 ──
    print("\n📐 9. turnover_min_valid 커스텀")
    # 거래대금 1억 (기본 5천만 기준이면 유효, 1억 기준이면 유효)
    row_1e = {**row_toxic, "거래대금(원)": 100_000_000}  # 1억
    # 기본 thresholds (5천만) → 유효 → TOXIC 판정
    state9a = determine_state_dynamic(row_1e, th)
    test("1억 + 기본min(5천만) → EXIT_WARNING", state9a == "EXIT_WARNING",
         f"state={state9a}")
    # 커스텀 2억 기준 → 1억은 무효 → TOXIC 스킵
    th_strict = {**th, "turnover_min_valid": 200_000_000}
    state9b = determine_state_dynamic(row_1e, th_strict)
    test("1억 + strict_min(2억) → EXIT_WARNING 아님", state9b != "EXIT_WARNING",
         f"state={state9b}")

    # ── 10. 외인순매수금액 우선 사용 ──
    print("\n📐 10. 외인순매수금액 fallback")
    row_named = {
        "RSI14": 50, "ret_1d_%": 6.0, "ret_5d_%": 10.0,
        "MACD_Slope_PCT": 0.01, "Range_Pos": 0.5, "Vol_Quality": 1.0,
        "TIMING_SCORE": 50, "거래강도": 2.0, "Low_Trend_PCT": 0,
        "Above_MA20": 1,
        "거래대금(원)": 10_000_000_000,
        # 외인순매수금액 (원 단위) 우선
        "외인순매수금액": -3_000_000_000,
        "개인순매수금액": 3_000_000_000,
        # 외인순매수 (혹시 주수 기반) — fallback으로만 사용
        "외인순매수": -100,
        "개인순매수": 100,
    }
    state10 = determine_state_dynamic(row_named, th)
    test("금액 컬럼 우선 → EXIT_WARNING", state10 == "EXIT_WARNING",
         f"state={state10}")

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
