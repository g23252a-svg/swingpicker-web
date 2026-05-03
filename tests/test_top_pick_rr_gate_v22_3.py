# tests/test_top_pick_rr_gate_v22_3.py
"""
v22.3 회귀 테스트 — TOP_PICK RR≥1.0 하드게이트
==================================================

평가 피드백 96.5점 핵심 항목:
"STABLE 타입 대창(RR=0.87)이 TOP_PICK인 건 추천 신뢰도 깎는다"
"""
import pandas as pd
import numpy as np


def test_rr_below_1_blocks_top_pick():
    """RR=0.87 → TOP_PICK=0"""
    from scoring_engine import compute_elite_score

    # close=10000, stop=9500(-5%), tp1=10435 → reward 435 / risk 500 = RR 0.87
    df = pd.DataFrame({
        "종목명":           ["TEST"],
        "ROUTE":            ["ARMED"],
        "STRUCT_SCORE":     [85.0], "TIMING_SCORE": [76.0], "AI_SCORE": [70.0],
        "ELITE_SCORE":      [78.0], "BALANCE_SCORE": [75.0],
        "EST_WIN_RATE":     [0.60], "EST_WIN_RATE_MODE": ["MATURE"],
        "종가":              [10000.0], "추천매수가": [10000.0],
        "손절가":            [9500.0],  "추천매도가1": [10435.0],
        "ENTRY_GAP_PCT":    [0.0],     "거래대금(억원)": [200.0],
        "PASS_EBS":         [1],       "TP1_PCT": [4.35],
    })
    result, _meta = compute_elite_score(df.copy())
    rr = float(result["RR_NOW_TP1"].iloc[0])
    assert abs(rr - 0.87) < 0.05, f"RR 계산 검증: {rr}"
    assert result["TOP_PICK"].iloc[0] == 0, f"v22.3 차단 실패: RR={rr}"


def test_rr_just_below_1_blocked():
    """RR=0.99 (경계 직전) → TOP_PICK=0"""
    from scoring_engine import compute_elite_score

    df = pd.DataFrame({
        "종목명":           ["TEST"], "ROUTE": ["ARMED"],
        "STRUCT_SCORE":     [85.0], "TIMING_SCORE": [76.0], "AI_SCORE": [70.0],
        "ELITE_SCORE":      [78.0], "BALANCE_SCORE": [75.0],
        "EST_WIN_RATE":     [0.60], "EST_WIN_RATE_MODE": ["MATURE"],
        "종가":              [10000.0], "추천매수가": [10000.0],
        "손절가":            [9500.0],  "추천매도가1": [10495.0],  # RR 0.99
        "ENTRY_GAP_PCT":    [0.0], "거래대금(억원)": [200.0],
        "PASS_EBS":         [1], "TP1_PCT": [4.95],
    })
    result, _meta = compute_elite_score(df.copy())
    rr = float(result["RR_NOW_TP1"].iloc[0])
    assert rr < 1.0
    assert result["TOP_PICK"].iloc[0] == 0


def test_rr_at_1_passes_rr_gate():
    """RR=1.0 → RR 게이트는 통과 (다른 조건 따로 봄)"""
    from scoring_engine import compute_elite_score

    # RR 1.0, STOP 8%, TP1 +8% (STABLE의 7~15 범위 진입)
    df = pd.DataFrame({
        "종목명":           ["TEST"], "ROUTE": ["ARMED"],
        "STRUCT_SCORE":     [85.0], "TIMING_SCORE": [80.0], "AI_SCORE": [75.0],
        "ELITE_SCORE":      [78.0], "BALANCE_SCORE": [80.0],
        "EST_WIN_RATE":     [0.60], "EST_WIN_RATE_MODE": ["MATURE"],
        "종가":              [10000.0], "추천매수가": [10000.0],
        "손절가":            [9200.0],  "추천매도가1": [10800.0],  # RR 1.0
        "ENTRY_GAP_PCT":    [0.0], "거래대금(억원)": [200.0],
        "PASS_EBS":         [1], "TP1_PCT": [8.0],
    })
    result, meta = compute_elite_score(df.copy())
    rr = float(result["RR_NOW_TP1"].iloc[0])
    assert abs(rr - 1.0) < 0.02, f"RR 검증: {rr}"
    # hard_gate(RR≥1.0 포함)는 1건이어야 함 — RR 게이트는 통과
    funnel = meta.get("stable_funnel", {})
    assert funnel.get("hard_gate", 0) == 1, \
        f"RR=1.0인데 hard_gate에서 떨어짐: funnel={funnel}"


def test_real_world_daechang_case():
    """평가 발견 사례: 대창 RR=0.87 → 차단되어야 함"""
    from scoring_engine import compute_elite_score

    df = pd.DataFrame({
        "종목명":           ["대창"], "ROUTE": ["ARMED"],
        "STRUCT_SCORE":     [82.0], "TIMING_SCORE": [76.0], "AI_SCORE": [70.0],
        "ELITE_SCORE":      [79.4], "BALANCE_SCORE": [71.0],
        "EST_WIN_RATE":     [0.60], "EST_WIN_RATE_MODE": ["MATURE"],
        "종가":              [3000.0], "추천매수가": [3000.0],
        "손절가":            [2850.0], "추천매도가1": [3130.5],  # RR 0.87
        "ENTRY_GAP_PCT":    [0.0], "거래대금(억원)": [120.0],
        "PASS_EBS":         [1], "TP1_PCT": [4.35],
    })
    result, _meta = compute_elite_score(df.copy())
    rr = float(result["RR_NOW_TP1"].iloc[0])
    assert rr < 1.0
    assert result["TOP_PICK"].iloc[0] == 0, \
        f"대창 RR={rr} 차단 실패 — v22.3 패치 미적용"
    # TOP_PICK=0이면 TYPE은 빈 문자열 또는 "NONE" (legacy 호환)
    assert result["TOP_PICK_TYPE"].iloc[0] in ("", "NONE"), \
        f"차단된 종목 TYPE이 비어있지 않음: {result['TOP_PICK_TYPE'].iloc[0]}"
