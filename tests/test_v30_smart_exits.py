# -*- coding: utf-8 -*-
"""[v30] 데이터 근거 기반 손절/목표가 테스트.

청산 그리드 검증 (게이트 통과 후보 851건, train 2~5월 / OOS 6~7월):
  - 손절 상한 8% (기존 15%): train 평균 -0.83→+1.03%/건, p5 -15.2→-8.2%
  - 약세장 손절 확대(×1.4/×1.2): 유해 → 제거 (손절 트레이드 51%가 +5% 선터치)
  - TP 3.5×ATR (기존 risk×RR): 목표를 종목의 실제 변동폭에 비례
  - 본전전환 +5%: 상위 8개 조합 전부 포함
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stop_logic as SL
from exit_plan import BE_TRIGGER_PCT, add_exit_plan_columns
from trade_plan import build_trade_plan


def test_stop_ceiling_is_8pct():
    # 초고변동 종목 (ATR 10%) — 기존이면 15%까지 벌어졌음
    stop, stop_pct, _, _ = SL.calc_stop_price(buy=10000, atr_val=1000, mcap=50000)
    assert stop_pct <= 8.5  # tick 라운딩 여유
    assert stop >= 10000 * 0.915


def test_no_crash_widening():
    # 시장폭 20 (패닉)과 60 (정상)에서 손절폭이 같아야 함 — 확대 로직 제거 확인
    cfg_panic = SL.StopConfig(market_breadth=20.0)
    cfg_norm = SL.StopConfig(market_breadth=60.0)
    s1, p1, _, _ = SL.calc_stop_price(buy=10000, atr_val=300, mcap=50000, cfg=cfg_panic)
    s2, p2, _, _ = SL.calc_stop_price(buy=10000, atr_val=300, mcap=50000, cfg=cfg_norm)
    assert abs(p1 - p2) < 0.01


def test_tp_ladder_atr_proportional():
    plan = build_trade_plan(buy=10000, atr_val=400, last_c=10000, mcap=50000)
    # TP1≈+3.5ATR, TP2≈+5ATR, TP3≈+6.5ATR (tick 라운딩 허용)
    assert abs(plan.tp1 - (plan.entry + 3.5 * 400)) <= plan.entry * 0.01
    assert abs(plan.tp2 - (plan.entry + 5.0 * 400)) <= plan.entry * 0.01
    assert abs(plan.tp3 - (plan.entry + 6.5 * 400)) <= plan.entry * 0.01
    assert plan.tp1 < plan.tp2 < plan.tp3


def test_be_trigger_column():
    df = pd.DataFrame({"추천매수가": [10000.0], "ROUTE": ["ATTACK"]})
    out = add_exit_plan_columns(df)
    assert "BE_TRIGGER_PRICE" in out.columns
    assert float(out.loc[0, "BE_TRIGGER_PRICE"]) == round(10000 * (1 + BE_TRIGGER_PCT / 100))
    assert "본전전환" in out.loc[0, "EXIT_PLAN_NOTE"]
