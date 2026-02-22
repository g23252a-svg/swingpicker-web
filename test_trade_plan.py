# -*- coding: utf-8 -*-
"""
test_trade_plan.py — SSOT 체결 엔진 유닛/회귀 테스트
═══════════════════════════════════════════════════════

실행: python test_trade_plan.py
     또는 pytest test_trade_plan.py -v

시나리오:
  - 정상 진입/손절/익절
  - 갭하락/갭상승
  - 동시터치 (SL 우선)
  - 하한가
  - NaN/0 입력 방어
  - 계약(Contract) 위반
  - 포지션 사이징
  - 레짐 전환
  - 룩어헤드 검증
"""

import sys
import numpy as np

# stop_logic을 먼저 import해야 trade_plan이 동작
import stop_logic as SL
from trade_plan import (
    TradePlan, ExecRule, BarResult,
    build_trade_plan, exec_bar, exec_multi_bar,
    validate_row, check_lookahead,
    REQUIRED_PLAN_KEYS,
)

PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def run_tests():
    global PASS, FAIL

    print("=" * 60)
    print("🧪 SSOT 체결 엔진 테스트")
    print("=" * 60)

    rule = ExecRule()

    # ── 1. 정상 TradePlan 생성 ──
    print("\n📐 1. build_trade_plan 정상")
    plan = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
    test("entry > 0", plan.entry > 0, f"entry={plan.entry}")
    test("stop < entry", plan.stop < plan.entry, f"stop={plan.stop}")
    test("tp1 > entry", plan.tp1 > plan.entry, f"tp1={plan.tp1}")
    test("stop_pct > 0", plan.stop_pct > 0, f"stop_pct={plan.stop_pct}")
    test("exec_rule_id 설정됨", plan.exec_rule_id == rule.rule_id)

    # ── 2. Contract 검증 ──
    print("\n📐 2. Contract 검증")
    row = plan.to_row()
    test("필수 컬럼 존재", REQUIRED_PLAN_KEYS.issubset(set(row.keys())))
    try:
        validate_row({"ENTRY_PRICE": 0})
        test("entry=0 에러", False, "should have raised")
    except ValueError:
        test("entry=0 에러 발생", True)
    try:
        validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 11000,
                       "TP1": 12000, "POSITION_PCT": 100,
                       "ENTRY_ACTION": "enter", "PLAN_REASON": "test",
                       "EXEC_RULE_ID": "v1"})
        test("stop>entry 에러", False, "should have raised")
    except ValueError:
        test("stop>entry 에러 발생", True)

    # ── 3. 장중 SL 터치 ──
    print("\n📐 3. 장중 SL 터치")
    plan3 = TradePlan(entry=10000, stop=9500, tp1=11500, exec_rule_id="v1")
    r = exec_bar(plan3, 10100, 10200, 9400, 9600, rule)
    test("SL 발동", r.action == "stop_hit", f"action={r.action}")
    test("체결가 ≤ SL", r.fill_price <= 9500, f"fill={r.fill_price}")
    test("수익률 < 0", r.return_pct < 0, f"ret={r.return_pct}")

    # ── 4. 장중 TP 터치 ──
    print("\n📐 4. 장중 TP 터치")
    r = exec_bar(plan3, 10100, 11600, 10000, 11400, rule)
    test("TP 발동", r.action == "tp_hit", f"action={r.action}")
    test("체결가 < TP (슬리피지)", r.fill_price < 11500, f"fill={r.fill_price}")
    test("수익률 > 0", r.return_pct > 0, f"ret={r.return_pct}")

    # ── 5. 갭하락 ──
    print("\n📐 5. 갭하락 (시가 < SL)")
    r = exec_bar(plan3, 9200, 9300, 9100, 9250, rule)
    test("갭하락 SL", r.action == "stop_hit")
    test("시가 체결", r.fill_price == 9200, f"fill={r.fill_price}")
    test("reason=gap_down", "gap_down" in r.reason)

    # ── 6. 갭상승 ──
    print("\n📐 6. 갭상승 (시가 > TP)")
    r = exec_bar(plan3, 11700, 11800, 11600, 11750, rule)
    test("갭상승 TP", r.action == "tp_hit")
    test("시가 체결", r.fill_price == 11700, f"fill={r.fill_price}")
    test("reason=gap_up", "gap_up" in r.reason)

    # ── 7. 동시터치 (SL 우선) ──
    print("\n📐 7. 동시터치 (High≥TP & Low≤SL)")
    r = exec_bar(plan3, 10100, 11600, 9400, 10000, rule)
    test("SL 우선", r.action == "stop_hit", f"action={r.action}")
    test("reason=same_bar_sl", "same_bar" in r.reason)

    # ── 8. 동시터치 (TP 우선 규칙) ──
    print("\n📐 8. 동시터치 (TP 우선 규칙)")
    tp_rule = ExecRule(tp_sl_same_bar_priority="TP", rule_id="v2_tp_first")
    r = exec_bar(plan3, 10100, 11600, 9400, 10000, tp_rule)
    test("TP 우선", r.action == "tp_hit", f"action={r.action}")

    # ── 9. 미체결 ──
    print("\n📐 9. 미체결 (SL/TP 미도달)")
    r = exec_bar(plan3, 10100, 10200, 9600, 10050, rule)
    test("hold", r.action == "hold")
    test("fill=0", r.fill_price == 0.0)

    # ── 10. 하한가 의심 ──
    print("\n📐 10. 하한가 (시가=저가=고가, 시가<SL)")
    r = exec_bar(plan3, 9300, 9300, 9300, 9300, rule)
    test("SL 발동", r.action == "stop_hit")
    test("시가 체결 (갭하락)", r.fill_price == 9300, f"fill={r.fill_price}")
    # 시가 9300 < SL 9500이므로 갭하락 규칙이 먼저 적용 (정상)

    # 하한가이면서 시가==SL인 경계 케이스
    plan_limit = TradePlan(entry=10000, stop=9300, tp1=11500, exec_rule_id="v1")
    r2 = exec_bar(plan_limit, 9300, 9300, 9300, 9300, rule)
    test("하한가 경계(시가==SL)", r2.action == "stop_hit")
    test("reason=limit_down", "limit_down" in r2.reason, f"reason={r2.reason}")

    # ── 11. NaN/0 입력 방어 ──
    print("\n📐 11. NaN/0 입력 방어")
    r = exec_bar(plan3, 0, 0, 0, 0, rule)
    test("invalid 처리", r.action == "none")
    plan_bad = TradePlan(entry=0, stop=0, tp1=0, exec_rule_id="v1")
    r = exec_bar(plan_bad, 10000, 10100, 9900, 10050, rule)
    test("entry=0 → none", r.action == "none")

    # ── 12. multi-bar 실행 ──
    print("\n📐 12. exec_multi_bar")
    bars = [
        (10100, 10200, 9600, 10050),  # day1: hold
        (10050, 10100, 9550, 9800),   # day2: hold
        (9800, 9900, 9400, 9500),     # day3: SL hit
        (9500, 9600, 9400, 9550),     # day4: 이미 청산
    ]
    r = exec_multi_bar(plan3, bars, rule)
    test("multi-bar SL", r.action == "stop_hit")

    # ── 13. 타임아웃 ──
    print("\n📐 13. 타임아웃 (max_hold_days)")
    bars_hold = [(10100, 10200, 9600, 10050)] * 5
    r = exec_multi_bar(plan3, bars_hold, rule, max_hold_days=5)
    test("timeout", r.action == "timeout", f"action={r.action}")

    # ── 14. tick 라운딩 ──
    print("\n📐 14. tick 라운딩 검증")
    plan14 = build_trade_plan(buy=10333, atr_val=300, last_c=10333, mcap=50000)
    test("entry tick", plan14.entry % SL.tick_size(plan14.entry) == 0,
         f"entry={plan14.entry}")
    test("stop tick", plan14.stop % SL.tick_size(plan14.stop) == 0,
         f"stop={plan14.stop}")

    # ── 15. 포지션 사이징 ──
    print("\n📐 15. 포지션 사이징 (account_risk)")
    plan_risk = build_trade_plan(buy=10000, atr_val=300, last_c=10000,
                                  mcap=50000, account_risk_pct=0.5)
    test("position_pct ≤ 100", plan_risk.position_pct <= 100)
    test("position_pct > 0", plan_risk.position_pct > 0)

    # ── 16. 극단 갭 진입 보류 ──
    print("\n📐 16. 극단 갭 진입 보류")
    plan_hold = build_trade_plan(buy=10000, atr_val=300, last_c=10000,
                                  mcap=50000, ret_1d=16.0, gap_pct=13.0)
    test("entry_action=hold", plan_hold.entry_action == "hold")
    test("position_pct=0", plan_hold.position_pct == 0.0)

    # ── 17. 레짐 전환 ──
    print("\n📐 17. 레짐 전환")
    SL.switch_config(SL.config_normal())
    plan_n = build_trade_plan(buy=10000, atr_val=500, last_c=10000, mcap=50000)
    SL.switch_config(SL.config_high_vol())
    plan_h = build_trade_plan(buy=10000, atr_val=500, last_c=10000, mcap=50000)
    test("RR 다름", plan_n.rr_mult != plan_h.rr_mult,
         f"normal={plan_n.rr_mult}, high={plan_h.rr_mult}")
    SL.switch_config(SL.config_normal())

    # ── 18. 룩어헤드 검증 ──
    print("\n📐 18. 룩어헤드 검증")
    violations = check_lookahead("20260210", ["20260209", "20260210", "20260211"])
    test("미래 데이터 감지", len(violations) == 1)
    test("정상 데이터 통과", len(check_lookahead("20260210", ["20260208", "20260210"])) == 0)

    # ── 19. frozen 불변성 ──
    print("\n📐 19. TradePlan 불변성 (frozen)")
    try:
        plan.entry = 99999  # type: ignore
        test("frozen 보장", False, "should raise")
    except (AttributeError, TypeError):
        test("frozen 보장", True)

    # ── 20. ExecRule ID 변경 추적 ──
    print("\n📐 20. ExecRule ID 추적")
    rule_v2 = ExecRule(rule_id="v2_tp_first", tp_sl_same_bar_priority="TP")
    plan_v2 = build_trade_plan(buy=10000, atr_val=300, last_c=10000,
                                mcap=50000, exec_rule=rule_v2)
    test("rule_id 반영", plan_v2.exec_rule_id == "v2_tp_first")

    # ── 21. gap_fill=LEVEL TP 체결가 (tp보다 좋아지면 버그) ──
    print("\n📐 21. gap_fill=LEVEL TP 체결가")
    level_rule = ExecRule(gap_fill="LEVEL", rule_id="v1_level")
    r = exec_bar(plan3, 11700, 11800, 11600, 11750, level_rule)
    test("tp_hit", r.action == "tp_hit")
    test("fill ≤ TP (LEVEL)", r.fill_price <= plan3.tp1, f"fill={r.fill_price}, tp={plan3.tp1}")

    # ── 22. fee 반영 ──
    print("\n📐 22. fee 반영 (왕복 수수료)")
    r_fee = exec_bar(plan3, 10100, 11600, 10000, 11400, rule)
    test("ret > 0 (TP)", r_fee.return_pct > 0, f"ret={r_fee.return_pct}")
    no_fee_rule = ExecRule(fee_bps=0.0)
    r_no_fee = exec_bar(plan3, 10100, 11600, 10000, 11400, no_fee_rule)
    test("fee 적용으로 ret 감소", r_fee.return_pct < r_no_fee.return_pct,
         f"with_fee={r_fee.return_pct:.3f}, no_fee={r_no_fee.return_pct:.3f}")

    # ── 23. regime_name 전파 ──
    print("\n📐 23. regime_name 전파")
    SL.switch_config(SL.config_high_vol())
    plan_hv = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
    test("regime=high_vol", plan_hv.regime == "high_vol", f"regime={plan_hv.regime}")
    SL.switch_config(SL.config_normal())
    plan_nm = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
    test("regime=normal", plan_nm.regime == "normal", f"regime={plan_nm.regime}")

    # ── 24. R:R SSOT (TP는 tick 후 risk 기준) ──
    print("\n📐 24. R:R SSOT (tick 후 risk)")
    plan24 = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
    risk_actual = plan24.entry - plan24.stop
    tp_implied_rr = (plan24.tp1 - plan24.entry) / risk_actual if risk_actual > 0 else 0
    test("TP1/R ≈ RR_MULT", abs(tp_implied_rr - plan24.rr_mult) < 0.5,
         f"implied={tp_implied_rr:.2f}, rr={plan24.rr_mult}")

    # ── 25. entry_filter reason in plan_reason ──
    print("\n📐 25. entry_filter reason in plan_reason")
    test("EF: in reason", "EF:" in plan_nm.plan_reason, f"reason={plan_nm.plan_reason}")

    # ── 26. Contract 불변조건: hold → pos=0 ──
    print("\n📐 26. Contract 불변조건")
    try:
        validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": 11000,
                       "POSITION_PCT": 50.0, "ENTRY_ACTION": "hold",
                       "PLAN_REASON": "test", "EXEC_RULE_ID": "v1",
                       "STOP_PCT": 5.0})
        test("hold+pos>0 에러", False, "should raise")
    except ValueError:
        test("hold+pos>0 에러 발생", True)

    # split → pos < 100
    try:
        validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": 11000,
                       "POSITION_PCT": 100.0, "ENTRY_ACTION": "split",
                       "PLAN_REASON": "test", "EXEC_RULE_ID": "v1",
                       "STOP_PCT": 5.0})
        test("split+pos=100 에러", False, "should raise")
    except ValueError:
        test("split+pos=100 에러 발생", True)

    # TP 정렬: TP2 <= TP1이면 에러
    try:
        validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": 11000,
                       "TP2": 10500, "TP3": 12000,
                       "POSITION_PCT": 100.0, "ENTRY_ACTION": "enter",
                       "PLAN_REASON": "test", "EXEC_RULE_ID": "v1",
                       "STOP_PCT": 5.0})
        test("TP2<TP1 에러", False, "should raise")
    except ValueError:
        test("TP2<TP1 에러 발생", True)

    # STOP_PCT > 30이면 에러
    try:
        validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": 11000,
                       "POSITION_PCT": 100.0, "ENTRY_ACTION": "enter",
                       "PLAN_REASON": "test", "EXEC_RULE_ID": "v1",
                       "STOP_PCT": 35.0})
        test("STOP_PCT>30 에러", False, "should raise")
    except ValueError:
        test("STOP_PCT>30 에러 발생", True)

    # ── 27. TP 정렬 정상 (build_trade_plan 결과) ──
    print("\n📐 27. TP 정렬 (TP1 < TP2 < TP3)")
    plan27 = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000)
    test("TP1 < TP2", plan27.tp1 < plan27.tp2, f"tp1={plan27.tp1}, tp2={plan27.tp2}")
    test("TP2 < TP3", plan27.tp2 < plan27.tp3, f"tp2={plan27.tp2}, tp3={plan27.tp3}")

    # ── 28. Property 테스트: 랜덤 입력에서도 불변조건 유지 ──
    print("\n📐 28. Property 테스트 (랜덤 50케이스)")
    import random
    random.seed(42)
    prop_fail = 0
    for i in range(50):
        buy_r = random.uniform(1000, 500000)
        atr_r = buy_r * random.uniform(0.005, 0.10)
        mcap_r = random.uniform(100, 200000)

        try:
            p = build_trade_plan(buy=buy_r, atr_val=atr_r, last_c=buy_r, mcap=mcap_r)
            # 불변: stop < entry
            assert p.stop < p.entry, f"stop({p.stop}) >= entry({p.entry})"
            # 불변: tp1 > entry
            assert p.tp1 > p.entry, f"tp1({p.tp1}) <= entry({p.entry})"
            # 불변: tp1 < tp2 < tp3
            assert p.tp1 < p.tp2 < p.tp3, f"tp 순서 위반: {p.tp1}, {p.tp2}, {p.tp3}"
            # 불변: stop_pct > 0, < 30
            assert 0 < p.stop_pct < 30, f"stop_pct={p.stop_pct}"
        except Exception as e:
            prop_fail += 1
            if prop_fail <= 3:
                print(f"    ⚠️ case {i}: buy={buy_r:.0f}, atr={atr_r:.0f} → {e}")

    test(f"Property 불변 ({50 - prop_fail}/50)", prop_fail == 0,
         f"{prop_fail}건 위반")

    # ── 29. exec_bar Property: tp_hit → fill ≤ tp (LEVEL) ──
    print("\n📐 29. Property: tp_hit → fill ≤ tp")
    level_rule = ExecRule(gap_fill="LEVEL", rule_id="v1_level")
    prop_fail2 = 0
    for _ in range(50):
        entry_r = random.uniform(5000, 100000)
        risk_r = entry_r * random.uniform(0.01, 0.08)
        stop_r = entry_r - risk_r
        tp_r = entry_r + risk_r * random.uniform(1.5, 4.0)
        p = TradePlan(entry=entry_r, stop=stop_r, tp1=tp_r, exec_rule_id="v1")
        hi = tp_r * random.uniform(1.0, 1.05)
        lo = stop_r * random.uniform(0.99, 1.05)  # might not trigger SL
        r = exec_bar(p, entry_r, hi, lo, entry_r, level_rule)
        if r.action == "tp_hit" and r.fill_price > tp_r + 0.01:
            prop_fail2 += 1
    test(f"tp_hit → fill ≤ tp ({50 - prop_fail2}/50)", prop_fail2 == 0)

    # ── 30. lookahead 날짜 정규화 ──
    print("\n📐 30. lookahead 날짜 정규화 (다양한 포맷)")
    test("YYYY-MM-DD 감지", len(check_lookahead("2026-02-10", ["2026-02-11"])) == 1)
    test("YYYYMMDD 감지", len(check_lookahead(20260210, [20260211])) == 1)
    test("혼합 포맷", len(check_lookahead("20260210", ["2026-02-11"])) == 1)
    test("정상 통과", len(check_lookahead("2026-02-10", ["2026-02-09"])) == 0)

    # ── 31. Contract: TP1=0 진짜 거절 ──
    print("\n📐 31. Contract: TP1=0 진짜 거절")
    try:
        validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": 0,
                       "POSITION_PCT": 100, "ENTRY_ACTION": "enter",
                       "PLAN_REASON": "test", "EXEC_RULE_ID": "v1", "STOP_PCT": 5.0})
        test("TP1=0 에러", False, "should raise")
    except ValueError:
        test("TP1=0 에러 발생", True)

    try:
        validate_row({"ENTRY_PRICE": 10000, "STOP_PRICE": 9500, "TP1": None,
                       "POSITION_PCT": 100, "ENTRY_ACTION": "enter",
                       "PLAN_REASON": "test", "EXEC_RULE_ID": "v1", "STOP_PCT": 5.0})
        test("TP1=None 에러", False, "should raise")
    except (ValueError, TypeError):
        test("TP1=None 에러 발생", True)

    # ── 32. min_order_amount 스킵 ──
    print("\n📐 32. min_order_amount 스킵")
    plan_min = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000,
                                 account_value_krw=50000)  # 5만원 계좌 → 주문금액 부족
    test("hold (min_order)", plan_min.entry_action == "hold",
         f"action={plan_min.entry_action}")
    test("position=0", plan_min.position_pct == 0.0)
    test("MIN_ORDER_SKIP in reason", "MIN_ORDER_SKIP" in plan_min.plan_reason,
         f"reason={plan_min.plan_reason}")

    # 충분한 계좌 → 정상 진입
    plan_ok = build_trade_plan(buy=10000, atr_val=300, last_c=10000, mcap=50000,
                                account_value_krw=10000000)  # 1000만원
    test("enter (sufficient)", plan_ok.entry_action == "enter")

    # ── 33. scaleout 동작 ──
    print("\n📐 33. scaleout exec_multi_bar")
    from trade_plan import exec_multi_bar as _emb
    rule_scale = ExecRule(use_scaleout=True, rule_id="v_scaleout")
    plan_s = TradePlan(entry=10000, stop=9500, tp1=10500, tp2=11000, tp3=11500,
                        exec_rule_id=rule_scale.rule_id)
    bars_s = [
        (10000, 10600, 9900, 10500),   # TP1 터치
        (10500, 11100, 10400, 11000),   # TP2 터치
        (11000, 11600, 10800, 11400),   # TP3 터치
    ]
    r_s = _emb(plan_s, bars_s, rule_scale, max_hold_days=5)
    test("scaleout reason", "scaleout:" in r_s.reason, f"reason={r_s.reason}")
    test("scaleout ret finite", np.isfinite(r_s.return_pct), f"ret={r_s.return_pct}")
    test("scaleout ret > 0", r_s.return_pct > 0, f"ret={r_s.return_pct}")
    test("scaleout tp_hit", r_s.action == "tp_hit")

    # scaleout OFF → 기존 동작
    rule_no_scale = ExecRule(use_scaleout=False)
    r_ns = _emb(plan_s, bars_s, rule_no_scale, max_hold_days=5)
    test("no_scaleout tp_hit", r_ns.action == "tp_hit")

    # ── 34. PM_BLOCK_INTRADAY 태그 ──
    print("\n📐 34. PM_BLOCK_INTRADAY 태그")
    SL.switch_config(SL.config_normal())
    s34, _, _, reason34 = SL.calc_stop_price(buy=10000, atr_val=300, mcap=50000,
                                              today_low=9000, gap_up_pct=10.0)
    test("PM_BLOCK tag", "PM_BLOCK_INTRADAY" in reason34, f"reason={reason34}")

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
