# -*- coding: utf-8 -*-
"""
test_phase_features.py — Phase 1+2 기능별 단위 테스트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Time Stop / Hard Block / 슬리피지 / 트레일링 / 전략팩토리 / 캘리브레이션
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

PASS = 0
FAIL = 0
def _check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" ({detail})" if detail else ""))

# ═══════════════════════════════════════════════
#  1. Config SSOT — WEIGHT_CONFIG 완전 제거 확인
# ═══════════════════════════════════════════════
print("📐 1. Config SSOT — WEIGHT_CONFIG 완전 제거")

from collector_config import DEFAULT_CONFIG, CollectorConfig
_check("ml_low 필드 존재", hasattr(DEFAULT_CONFIG, "ml_low"))
_check("macro_weights 필드 존재", hasattr(DEFAULT_CONFIG, "macro_weights"))
_check("to_weight_config_dict 삭제됨", not hasattr(DEFAULT_CONFIG, "to_weight_config_dict"))
_check("snapshot 동작", isinstance(DEFAULT_CONFIG.snapshot(), dict))
_check("snapshot_json 동작", isinstance(DEFAULT_CONFIG.snapshot_json(), str))

import scoring_engine
src = open(scoring_engine.__file__, encoding='utf-8').read()
# "WEIGHT_CONFIG =" 형태의 변수 선언이 없어야 함
has_wc_var = "WEIGHT_CONFIG =" in src or "WEIGHT_CONFIG=" in src
_check("scoring_engine에 WEIGHT_CONFIG 변수 없음", not has_wc_var)

# ═══════════════════════════════════════════════
#  2. Time Stop 단위 테스트
# ═══════════════════════════════════════════════
print("\n📐 2. Time Stop 단위 테스트")

from trade_plan import TradePlan, ExecRule, exec_multi_bar, BarResult

# 2a. 비활성 (time_stop_days=0)
plan0 = TradePlan(entry=10000, stop=9200, tp1=12000, time_stop_days=0)
bars_flat = [(10000, 10100, 9900, 10050)] * 20
r0 = exec_multi_bar(plan0, bars_flat, max_hold_days=20)
_check("비활성(0일): timeout", r0.action == "timeout")

# 2b. 활성 + 움직임 부족 → time_stop 발동
plan5 = TradePlan(entry=10000, stop=9200, tp1=12000,
                   time_stop_days=5, time_stop_min_move_pct=2.0,
                   time_stop_extend_if_profit=False)
bars_flat5 = [(10000, 10100, 9900, 10050)] * 10
r5 = exec_multi_bar(plan5, bars_flat5, max_hold_days=10)
_check("5일 움직임 부족: time_stop", r5.action == "time_stop")
_check("time_stop reason 포함", "time_stop_5d" in r5.reason)

# 2c. 수익 중 연장
plan_ext = TradePlan(entry=10000, stop=9200, tp1=12000,
                      time_stop_days=3, time_stop_min_move_pct=2.0,
                      time_stop_extend_if_profit=True)
bars_up = [(10000, 10200, 9950, 10150),  # +1.5%
           (10150, 10350, 10100, 10300),  # +3.0%
           (10300, 10500, 10250, 10400),  # +4.0%  ← day3, 수익 중 → 연장
           (10400, 10600, 10350, 10500)]  # +5.0%
r_ext = exec_multi_bar(plan_ext, bars_up, max_hold_days=4)
_check("수익중 연장: time_stop 아님", r_ext.action != "time_stop")

# 2d. SL이 time_stop보다 먼저 발동
plan_sl = TradePlan(entry=10000, stop=9200, tp1=12000,
                     time_stop_days=7, time_stop_min_move_pct=2.0)
bars_sl = [(10000, 10100, 9900, 10000),
           (10000, 10050, 9100, 9150)]  # SL 9200 터치
r_sl = exec_multi_bar(plan_sl, bars_sl, max_hold_days=7)
_check("SL 우선: stop_hit", r_sl.action == "stop_hit")

# 2e. build_trade_plan 내부 자동 주입 확인
from trade_plan import build_trade_plan
plan_auto = build_trade_plan(buy=10000, atr_val=300, last_c=10000)
_check("build_trade_plan time_stop 자동주입", plan_auto.time_stop_days == DEFAULT_CONFIG.time_stop_days,
     f"got {plan_auto.time_stop_days}, expected {DEFAULT_CONFIG.time_stop_days}")

# ═══════════════════════════════════════════════
#  3. 동적 슬리피지 단위 테스트
# ═══════════════════════════════════════════════
print("\n📐 3. 동적 슬리피지 단위 테스트")

from trade_plan import estimate_slippage_bps

cfg = DEFAULT_CONFIG
s_high = estimate_slippage_bps(100.0, cfg)  # 고유동
s_mid = estimate_slippage_bps(10.0, cfg)    # 중유동
s_low = estimate_slippage_bps(3.0, cfg)     # 저유동
s_none = estimate_slippage_bps(None, cfg)   # 데이터 없음
s_zero = estimate_slippage_bps(0, cfg)      # 0

_check("고유동(100억): base", s_high == cfg.slippage_base_bps, f"{s_high}")
_check("중유동(10억) > 고유동", s_mid > s_high, f"{s_mid} vs {s_high}")
_check("저유동(3억) > 중유동", s_low > s_mid, f"{s_low} vs {s_mid}")
_check("None = 최대", s_none == cfg.slippage_base_bps * cfg.slippage_low_liq_mult)
_check("0 = 최대", s_zero == s_none)
_check("슬리피지 단조감소 (유동↑ → 슬리피지↓)", s_none >= s_low >= s_mid >= s_high)

# ═══════════════════════════════════════════════
#  4. Hard Block 단위 테스트
# ═══════════════════════════════════════════════
print("\n📐 4. Hard Block 단위 테스트")

from validation import apply_hard_blocks, block_summary, HardBlockRule, HARD_BLOCK_RULES

# 4a. 정상 종목 통과
df_ok = pd.DataFrame({
    "ticker": ["A"],
    "ret_5d_%": [5.0], "거래대금(억)": [50.0],
    "gap_pct": [1.0], "RSI14": [55.0], "_data_length": [120],
})
passed, blocked = apply_hard_blocks(df_ok)
_check("정상 종목 통과", len(passed) == 1 and len(blocked) == 0)

# 4b. 연속급등 차단
df_surge = pd.DataFrame({"ret_5d_%": [45.0], "거래대금(억)": [50.0], "gap_pct": [1.0], "RSI14": [55.0], "_data_length": [120]})
p, b = apply_hard_blocks(df_surge)
_check("연속급등(45%) 차단", len(b) == 1)
_check("차단사유에 '연속급등' 포함", "연속급등" in str(b["BLOCK_REASON"].iloc[0]) if len(b) else False)

# 4c. 복합 위반
df_multi = pd.DataFrame({"ret_5d_%": [45.0], "거래대금(억)": [2.0], "gap_pct": [1.0], "RSI14": [90.0], "_data_length": [120]})
p, b = apply_hard_blocks(df_multi)
_check("복합 위반: 차단", len(b) == 1)
reasons = str(b["BLOCK_REASON"].iloc[0]) if len(b) else ""
_check("복합 사유 3개 이상", reasons.count("[") >= 3, reasons)

# 4d. 빈 DataFrame
p_empty, b_empty = apply_hard_blocks(pd.DataFrame())
_check("빈 DF 안전", len(p_empty) == 0 and len(b_empty) == 0)

# 4e. block_summary
summary = block_summary(b) if len(b) else {"total_blocked": 0}
_check("block_summary 동작", summary["total_blocked"] >= 1 if len(b) else True)

# ═══════════════════════════════════════════════
#  5. 트레일링 스탑 단위 테스트
# ═══════════════════════════════════════════════
print("\n📐 5. 트레일링 스탑 단위 테스트")

rule_trail = ExecRule(trailing_stop_enabled=True, trailing_stop_trigger_pct=3.0, trailing_stop_distance_pct=2.0)
plan_t = TradePlan(entry=10000, stop=9200, tp1=12000)

# 5a. 상승 후 하락 → trailing_stop 발동
bars_trail = [
    (10000, 10200, 9900, 10100),   # +1%
    (10100, 10400, 10050, 10350),  # +3.5% → 트레일 활성
    (10350, 10500, 10300, 10400),  # 고점 10500
    (10400, 10420, 10200, 10250),  # 고점 대비 -2.38% → 트리거
]
r_trail = exec_multi_bar(plan_t, bars_trail, rule=rule_trail)
_check("트레일링 발동", r_trail.action == "trailing_stop")
_check("트레일링 수익 양수", r_trail.return_pct > 0)

# 5b. 트레일링 OFF → 미발동
rule_off = ExecRule(trailing_stop_enabled=False)
r_off = exec_multi_bar(plan_t, bars_trail, rule=rule_off)
_check("트레일링 OFF: 미발동", r_off.action != "trailing_stop")

# 5c. 충분한 상승 → TP 먼저 발동
bars_tp = [
    (10000, 10400, 9900, 10300),   # +3%
    (10300, 12100, 10200, 12050),  # TP 12000 터치
]
r_tp_first = exec_multi_bar(plan_t, bars_tp, rule=rule_trail)
_check("TP 우선: tp_hit", r_tp_first.action == "tp_hit")

# ═══════════════════════════════════════════════
#  6. 전략 팩토리 단위 테스트
# ═══════════════════════════════════════════════
print("\n📐 6. 전략 팩토리 단위 테스트")

from strategies import StrategyFactory, select_strategies

# 6a. 전략 선택
_check("활황: breakout+pullback", set(select_strategies("NORMAL", 60)) == {"breakout", "pullback"})
_check("보통: pullback", select_strategies("NORMAL", 45) == ["pullback"])
_check("침체: mean_revert", select_strategies("NORMAL", 30) == ["mean_revert"])
_check("CRITICAL: 관망", select_strategies("CRITICAL", 60) == [])

# 6b. 팩토리 생성
for name in StrategyFactory.available():
    strat = StrategyFactory.create(name)
    _check(f"{name} 생성 OK", hasattr(strat, "filter") and hasattr(strat, "score"))

# 6c. filter → score → rank_and_pick 파이프라인
np.random.seed(42)
df_strat = pd.DataFrame({
    "종목코드": [f"{i:06d}" for i in range(30)],
    "TTM_SQUEEZE": np.random.choice([0, 1], 30),
    "BB_Expanding": np.random.choice([0, 1], 30),
    "Vol_Quality": np.random.uniform(0.5, 3.0, 30),
    "IS_ABOVE_POC": np.random.choice([0, 1], 30),
    "Above_MA20": np.random.choice([0, 1], 30),
    "MACD_Slope_PCT": np.random.uniform(-5, 5, 30),
    "Low_Trend_PCT": np.random.uniform(-3, 5, 30),
    "이격도": np.random.uniform(-5, 10, 30),
    "RSI14": np.random.uniform(20, 80, 30),
    "MTF_WEEKLY_TREND": np.random.choice([-1, 0, 1], 30),
    "ret_5d_%": np.random.uniform(-10, 15, 30),
    "STRUCT_SCORE": np.random.uniform(30, 90, 30),
    "TIMING_SCORE": np.random.uniform(20, 95, 30),
    "AI_SCORE": np.random.uniform(0, 80, 30),
    "FINAL_SCORE": np.random.uniform(40, 95, 30),
})

strat = StrategyFactory.create("breakout")
filtered = strat.filter(df_strat)
scored = strat.score(filtered)
picks = strat.rank_and_pick(scored, top_k=3)
_check("breakout 파이프라인: STRATEGY 컬럼", "STRATEGY" in picks.columns if not picks.empty else True)
_check("breakout 파이프라인: STRATEGY_SCORE 컬럼",
     "STRATEGY_SCORE" in picks.columns if not picks.empty else True)

# ═══════════════════════════════════════════════
#  7. 방어 규칙 단위 테스트
# ═══════════════════════════════════════════════
print("\n📐 7. 방어 규칙 단위 테스트")

from stop_logic import check_entry_defense, ENTRY_DEFENSE_RULES

_check("방어 규칙 8개", len(ENTRY_DEFENSE_RULES) == 8)

r_normal = check_entry_defense({"gap_pct": 1, "ret_1d_%": 2, "RSI14": 55, "거래대금(억)": 50})
_check("정상: enter", r_normal["action"] == "enter")

r_gap12 = check_entry_defense({"gap_pct": 13})
_check("갭12%: hold", r_gap12["action"] == "hold")

r_vi = check_entry_defense({"is_vi_triggered": True})
_check("VI: hold", r_vi["action"] == "hold")

r_low_liq = check_entry_defense({"거래대금(억)": 3})
_check("저유동(3억): hold", r_low_liq["action"] == "hold")  # [v20.8] PolicyConfig: 50억 미만 hold

r_rsi80 = check_entry_defense({"RSI14": 85})
_check("RSI 85: split 50%", r_rsi80["action"] == "split" and r_rsi80["position_pct"] == 50)

# ═══════════════════════════════════════════════
#  8. 경계값/회귀 테스트
# ═══════════════════════════════════════════════
print("\n📐 8. 경계값/회귀 테스트")

# -- Time Stop 경계 --
# 정확히 N-1일째 → 아직 안 걸림
plan_edge = TradePlan(entry=10000, stop=9200, tp1=12000,
                       time_stop_days=5, time_stop_min_move_pct=2.0,
                       time_stop_extend_if_profit=False)
bars_4 = [(10000, 10100, 9900, 10050)] * 4  # 4일만 (5일 기준이면 인덱스4=day5)
r_edge4 = exec_multi_bar(plan_edge, bars_4, max_hold_days=4)
_check("Time Stop 경계: 4일(day5 미도달) → timeout", r_edge4.action == "timeout")

# 정확히 N일째(인덱스 N-1) + 움직임 정확히 threshold
plan_exact = TradePlan(entry=10000, stop=9200, tp1=12000,
                        time_stop_days=3, time_stop_min_move_pct=2.0,
                        time_stop_extend_if_profit=False)
# day3 종가 10199 = +1.99% < 2.0% → time_stop
bars_exact = [(10000, 10200, 9900, 10050), (10050, 10200, 9950, 10100), (10100, 10300, 10050, 10199)]
r_exact = exec_multi_bar(plan_exact, bars_exact, max_hold_days=3)
_check("Time Stop 경계: +1.99% < 2.0% → time_stop", r_exact.action == "time_stop")

# day3 종가 10200 = +2.0% = threshold → 안 걸림 (미만이 조건)
bars_pass = [(10000, 10200, 9900, 10050), (10050, 10200, 9950, 10100), (10100, 10300, 10050, 10200)]
r_pass = exec_multi_bar(plan_exact, bars_pass, max_hold_days=3)
_check("Time Stop 경계: +2.0% = threshold → timeout(통과)", r_pass.action == "timeout")

# time_stop_days=1 (최소값)
plan_1d = TradePlan(entry=10000, stop=9200, tp1=12000,
                     time_stop_days=1, time_stop_min_move_pct=2.0,
                     time_stop_extend_if_profit=False)
bars_1d = [(10000, 10100, 9900, 10050)]  # day1 +0.5%
r_1d = exec_multi_bar(plan_1d, bars_1d, max_hold_days=1)
_check("Time Stop 1일: +0.5% → time_stop", r_1d.action == "time_stop")

# -- 슬리피지 경계 --
# threshold 정확히 일치
s_at_threshold = estimate_slippage_bps(cfg.slippage_liq_threshold_eok, cfg)
_check("슬리피지 threshold 정확히: base_bps", s_at_threshold == cfg.slippage_base_bps,
     f"{s_at_threshold} vs {cfg.slippage_base_bps}")

# threshold 바로 아래
s_just_below = estimate_slippage_bps(cfg.slippage_liq_threshold_eok - 0.01, cfg)
_check("슬리피지 threshold-0.01: > base_bps", s_just_below > cfg.slippage_base_bps)

# 음수 거래대금
s_neg = estimate_slippage_bps(-10.0, cfg)
_check("슬리피지 음수: 최대", s_neg == cfg.slippage_base_bps * cfg.slippage_low_liq_mult)

# 매우 큰 거래대금
s_huge = estimate_slippage_bps(100000.0, cfg)
_check("슬리피지 10만억: base_bps", s_huge == cfg.slippage_base_bps)

# -- Hard Block 경계 --
# RSI 정확히 85.0 → 차단 (gt 85)
df_rsi85 = pd.DataFrame({"RSI14": [85.0], "거래대금(억)": [50], "_data_length": [120], "ret_5d_%": [5], "gap_pct": [1]})
p85, b85 = apply_hard_blocks(df_rsi85)
_check("Hard Block RSI=85.0: 통과 (gt, not gte)", len(p85) == 1)

df_rsi85_1 = pd.DataFrame({"RSI14": [85.01], "거래대금(억)": [50], "_data_length": [120], "ret_5d_%": [5], "gap_pct": [1]})
p851, b851 = apply_hard_blocks(df_rsi85_1)
_check("Hard Block RSI=85.01: 차단", len(b851) == 1)

# [v20.8] PolicyConfig SSOT: 거래대금 임계치 = 30.0억
# 거래대금 정확히 30.0 → 통과 (lt 30.0이면 차단, 30.0은 통과)
df_tv30 = pd.DataFrame({"거래대금(억)": [30.0], "RSI14": [50], "_data_length": [120], "ret_5d_%": [5], "gap_pct": [1]})
p30, b30 = apply_hard_blocks(df_tv30)
_check("Hard Block 거래대금=30.0: 통과 (lt, not lte)", len(p30) == 1)

df_tv299 = pd.DataFrame({"거래대금(억)": [29.99], "RSI14": [50], "_data_length": [120], "ret_5d_%": [5], "gap_pct": [1]})
p299, b299 = apply_hard_blocks(df_tv299)
_check("Hard Block 거래대금=29.99: 차단", len(b299) == 1)

# NaN 컬럼 처리
df_nan = pd.DataFrame({"RSI14": [np.nan], "거래대금(억)": [np.nan], "_data_length": [np.nan], "ret_5d_%": [np.nan], "gap_pct": [np.nan]})
p_nan, b_nan = apply_hard_blocks(df_nan)
_check("Hard Block 전부 NaN: 처리 안 죽음", len(p_nan) + len(b_nan) == 1)

# -- 트레일링 스탑 경계 --
# 정확히 trigger_pct에서 활성화
rule_t3 = ExecRule(trailing_stop_enabled=True, trailing_stop_trigger_pct=3.0, trailing_stop_distance_pct=2.0)
plan_tb = TradePlan(entry=10000, stop=9200, tp1=12000)
bars_trigger_exact = [
    (10000, 10300, 9950, 10300),   # 고가 10300 = +3.0% 정확히 → 활성화
    (10300, 10310, 10090, 10100),  # 고가기준 10310, 종가 10100, drop = 2.04% → 트리거
]
r_te = exec_multi_bar(plan_tb, bars_trigger_exact, rule=rule_t3)
_check("트레일링 trigger 정확히 3%: 활성→발동", r_te.action == "trailing_stop")

# trigger 미달
bars_no_trigger = [
    (10000, 10290, 9950, 10290),   # 고가 10290 = +2.9% < 3% → 미활성
    (10290, 10300, 10050, 10050),  # drop이 있어도 미활성이면 안 걸림
]
r_nt = exec_multi_bar(plan_tb, bars_no_trigger, rule=rule_t3)
_check("트레일링 trigger 미달(2.9%): 미발동", r_nt.action != "trailing_stop")

# ═══════════════════════════════════════════════
#  결과
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"🏁 결과: {PASS}/{PASS+FAIL} 통과 ({FAIL} 실패)")
if FAIL == 0:
    print("🏆 ALL PASSED!")
else:
    print("⚠️ 실패 항목이 있습니다!")
print("=" * 60)


def test_phase_features_suite_passed():
    """pytest collection 중 sys.exit가 발생하지 않도록 결과만 검증한다."""
    assert FAIL == 0, f"phase feature checks failed: {FAIL}"


if __name__ == "__main__":
    sys.exit(FAIL)
