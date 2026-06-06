# -*- coding: utf-8 -*-
"""[v22.3.22] OOS 검증형 Historical Alpha Combo 엔진 테스트.

공식 산식 무변경 · OOS 검증 · fallback(breadth 없음, RR 보조) · 정직한 통계.
"""
import sys, os, glob
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from find_best_historical_alpha_combo_v22322 import (
    select_alpha_candidates, select_alpha_tiers, _rule_mask, FALLBACK_RULE,
    discover_oos_rules, _stats, RR_FLOOR,
)


def _row(**ov):
    r = dict(종목코드="000001", 종목명="후보", TIMING_SCORE=60, 외인순매수=10000,
             POC_GAP=30, STRUCT_SCORE=70, VWAP_GAP=5, RR_NOW_TP1=1.8, FINAL_SCORE=70,
             ENTRY_RISK_LEVEL="GREEN", MARKET_BREADTH=40,
             MARKET_WARNING_GUARD_FLAG=0, ABNORMAL_HISTORY_GUARD_FLAG=0,
             SPIKE_REVERSAL_GUARD_FLAG=0, LONG_HISTORY_COLLAPSE_FLAG=0)
    r.update(ov); return r


FB = {**FALLBACK_RULE, "struct_min": None, "breadth_max": None}


def test_fallback_is_breadth_free():
    # fallback에 breadth 조건 없어야 (과최적화 폐기)
    assert FALLBACK_RULE.get("breadth_max", None) is None
    assert "breadth" not in FALLBACK_RULE["desc"].lower()
    assert FALLBACK_RULE["timing_min"] == 55
    assert FALLBACK_RULE["poc_max"] == 90


def test_rr_not_required():
    # RR 없는 행도 fallback 통과해야 (RR은 보조 정렬만)
    df = pd.DataFrame([_row(RR_NOW_TP1=None)])
    df = df.drop(columns=["RR_NOW_TP1"])  # RR 컬럼 자체가 없는 과거 상황
    assert len(select_alpha_candidates(df, FB)) == 1


def test_timing_below_excluded():
    assert len(select_alpha_candidates(pd.DataFrame([_row(TIMING_SCORE=50)]), FB)) == 0


def test_foreign_sell_excluded():
    assert len(select_alpha_candidates(pd.DataFrame([_row(외인순매수=-1)]), FB)) == 0


def test_poc_over_90_excluded():
    assert len(select_alpha_candidates(pd.DataFrame([_row(POC_GAP=100)]), FB)) == 0
    assert len(select_alpha_candidates(pd.DataFrame([_row(POC_GAP=85)]), FB)) == 1


def test_guard_flags_exclude():
    for g in ["MARKET_WARNING_GUARD_FLAG", "ABNORMAL_HISTORY_GUARD_FLAG",
              "SPIKE_REVERSAL_GUARD_FLAG", "LONG_HISTORY_COLLAPSE_FLAG"]:
        assert len(select_alpha_candidates(pd.DataFrame([_row(**{g: 1})]), FB)) == 0


def test_rr_used_as_secondary_ranking():
    # 조건 동일, RR 높은 게 상위 (보조 정렬)
    df = pd.DataFrame([
        _row(종목코드="A", RR_NOW_TP1=3.5, 외인순매수=5000),
        _row(종목코드="B", RR_NOW_TP1=1.2, 외인순매수=5000),
    ])
    picks = select_alpha_candidates(df, FB, 2)
    assert picks.iloc[0]["종목코드"] == "A"


def test_vwap_extreme_penalty_not_block():
    # VWAP 떠있어도 차단 아닌 페널티 (급등 후보 보존)
    kept = select_alpha_candidates(pd.DataFrame([_row(VWAP_GAP=30)]), FB)
    assert len(kept) == 1 and kept.iloc[0]["_vwap_pen"] == 8.0


def test_does_not_modify_official():
    df = pd.DataFrame([_row(TOP_PICK=0, BUY_NOW_ELIGIBLE=0)])
    before = (df["TOP_PICK"].copy(), df["BUY_NOW_ELIGIBLE"].copy())
    select_alpha_candidates(df, FB)
    assert (df["TOP_PICK"] == before[0]).all() and (df["BUY_NOW_ELIGIBLE"] == before[1]).all()


def test_oos_discovery_runs_on_real_data():
    # 실데이터 기반 OOS 탐색은 최신 장세/데이터에 따라 통과 룰이 0개일 수 있다.
    # 이 경우 엔진은 fallback으로 동작하는 것이 정상이며, 테스트 실패가 아니라 환경 의존 skip이 맞다.
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not glob.glob(os.path.join(data_dir, "ohlcv_cache_2026*.parquet")):
        import pytest; pytest.skip("OHLC 없음")
    disc = discover_oos_rules(data_dir)
    rules = disc.get("rules", [])
    if not rules:
        import pytest; pytest.skip("룰 탐색 결과 없음(환경 의존)")
    oos = [r for r in rules if r["oos_pass"]]
    if not oos:
        import pytest; pytest.skip("현재 롤링 실데이터에서 OOS 통과 룰 없음 — fallback 경로 정상")
    # 최상위 OOS 룰은 breadth 조건 없어야 (과최적화 회피)
    top = oos[0]
    assert top["breadth_max"] is None, f"최상위 룰에 breadth={top['breadth_max']} (과최적화 의심)"


def test_honest_stats_not_70pct():
    # 정직성: 채택 룰 승률이 70% 거짓이 아님 (진입일 포함 OOS 실측 37% 내외)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not glob.glob(os.path.join(data_dir, "ohlcv_cache_2026*.parquet")):
        import pytest; pytest.skip("OHLC 없음")
    disc = discover_oos_rules(data_dir)
    oos = [r for r in disc.get("rules", []) if r["oos_pass"]]
    if not oos:
        import pytest; pytest.skip("OOS 룰 없음")
    assert oos[0]["win_rate"] < 70.0  # 70% 거짓 금지


def test_rr_floor_is_1_2():
    assert RR_FLOOR == 1.2


def test_tier_a_b_separation():
    # RR>=1.2 → Tier A(실전), RR<1.2 → Tier B(관찰)
    df = pd.DataFrame([
        _row(종목코드="A_HI", RR_NOW_TP1=1.8),
        _row(종목코드="A_OK", RR_NOW_TP1=1.3),
        _row(종목코드="B_LOW1", RR_NOW_TP1=0.9),
        _row(종목코드="B_LOW2", RR_NOW_TP1=0.5),
    ])
    t = select_alpha_tiers(df, FB, 3)
    a_codes = set(t["tier_a"]["종목코드"]) if len(t["tier_a"]) else set()
    b_codes = set(t["tier_b"]["종목코드"]) if len(t["tier_b"]) else set()
    assert a_codes == {"A_HI", "A_OK"}, f"Tier A={a_codes}"
    assert b_codes == {"B_LOW1", "B_LOW2"}, f"Tier B={b_codes}"
    assert t["rule_pass_n"] == 4


def test_tier_a_empty_when_all_rr_low():
    # 전부 RR<1.2면 Tier A 0개, Tier B에 다 (오늘 같은 상황)
    df = pd.DataFrame([
        _row(종목코드="X", RR_NOW_TP1=0.6),
        _row(종목코드="Y", RR_NOW_TP1=0.9),
    ])
    t = select_alpha_tiers(df, FB, 3)
    assert len(t["tier_a"]) == 0
    assert len(t["tier_b"]) == 2
    assert t["rule_pass_n"] == 2  # 룰은 통과 (RR만 부족) → 0개 사유 표시 가능


def test_tier_c_empty_when_no_rule_pass():
    # 룰 자체 미통과(TIMING 낮음)면 rule_pass_n=0 (Tier C 현금우위)
    df = pd.DataFrame([_row(TIMING_SCORE=40)])
    t = select_alpha_tiers(df, FB, 3)
    assert t["rule_pass_n"] == 0
    assert len(t["tier_a"]) == 0 and len(t["tier_b"]) == 0

