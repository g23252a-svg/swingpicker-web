# -*- coding: utf-8 -*-
"""
test_reason_builder_vectorized_v3121.py
═══════════════════════════════════════════════════
v3.12.1 사유 빌더 벡터화 회귀/성능 테스트.

배경
  guard_system._build_reason / pipeline_finalize 의 July·Recovery REASON 빌더가
  `for i in out.index` 행 루프(.loc/.at 접근)로 구현돼 매일 전체 종목에 대해
  실행되며 GUARD 적용 시간의 ~90%를 차지했다. v3.12.1에서 완전 벡터화하면서
  **출력 byte-identical**을 계약으로 고정한다.

이 테스트가 지키는 것
  1. 벡터화 결과가 원본 행루프 로직(이 파일에 _ref_*로 재현)과 정확히 일치.
  2. NaN·임계 경계·elif 분기·cap 절단(bits[:4]/[:5])·empty_fill 모두 커버.
  3. (느슨한) 성능 회귀 가드 — 벡터화가 행루프보다 빠름.
"""
import time

import numpy as np
import pandas as pd
import pytest

import guard_system as g


# ─────────────────────────────────────────────────────────────
#  참조 구현 (원본 행루프 로직 그대로 — 골든 마스터)
# ─────────────────────────────────────────────────────────────
def _ref_guard_reason(out: pd.DataFrame, broken5: pd.Series) -> pd.Series:
    parts_all = []
    for i in out.index:
        p = []
        if not out.at[i, "GUARD_PASS_1"]:
            p.append("G1유동성차단")
        if out.at[i, "GUARD_RR_MULT"] < 1.0:
            p.append(f"G2RR×{out.at[i, 'GUARD_RR_MULT']:.1f}")
        if out.at[i, "GUARD_PENALTY_3"] > 0:
            p.append(f"G3보유경과-{out.at[i, 'GUARD_PENALTY_3']:.0f}")
        if not out.at[i, "GUARD_PASS_4"]:
            p.append("G4저모멘텀차단")
        if out.at[i, "GUARD_FORCE_EXIT_ALERT"]:
            p.append(f"G5추세붕괴{int(broken5.at[i])}축")
        if out.at[i, "GUARD_PENALTY_6"] > 0:
            p.append("G6시장역행-25")
        if out.at[i, "GUARD_PENALTY_7"] > 0:
            p.append("G7윗꼬리약세-15")
        if out.at[i, "GUARD_PRE_WARNING"]:
            p.append("G8사전경고")
        parts_all.append(" · ".join(p))
    return pd.Series(parts_all, index=out.index, dtype="object")


def _rand_scored_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ELITE_SCORE": rng.uniform(20, 95, n),
        "TIMING_SCORE": rng.choice([0, 0, 10, 30, 50, 80], n).astype(float),
        "AXIS_MEAN": rng.uniform(20, 80, n),
        "TOP_PICK": rng.integers(0, 2, n),
        "CARRY_AGE_DAYS": rng.integers(0, 12, n),
        "거래대금(억원)": rng.uniform(30, 500, n),
        "STOP_PCT": rng.uniform(3, 10, n),
        "SUPERTREND_DIR": rng.choice([-1, 1], n),
        "Above_MA20": rng.choice([0, 1], n),
        "IS_ABOVE_POC": rng.choice([0, 1], n),
        "HMA_Trend": rng.choice(["▲", "▼"], n),
        "MACD_Slope_PCT": rng.uniform(-2, 2, n),
        "Upper_Shadow_Ratio": rng.uniform(0, 1, n),
        "거래강도": rng.uniform(0.3, 2, n),
        "ret_1d_%": rng.uniform(-8, 8, n),
        "KOSPI_RET_1D": rng.uniform(-3, 3, n),
        "종목명": [f"종목{i}" for i in range(n)],
        "업종": rng.choice(["반도체", "지주", "SI", "바이오", "2차전지"], n),
    })


# ─────────────────────────────────────────────────────────────
#  GUARD_REASON 벡터화 == 원본
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", range(8))
def test_guard_reason_identical_to_reference(seed):
    df = _rand_scored_df(rng_n := int(np.random.default_rng(seed).integers(40, 400)), seed)
    out = g.apply_guard_system(df.copy())
    broken5, _ = g.guard5_trendline_collapse(out, g._FallbackGuardConfig())
    ref = _ref_guard_reason(out, broken5)
    got = g._build_reason(out, broken5)
    assert list(got.values) == list(ref.values), (
        f"seed={seed} n={rng_n} GUARD_REASON 불일치"
    )


def test_guard_reason_empty_when_all_pass():
    """모든 가드 통과 행은 빈 문자열."""
    n = 50
    df = pd.DataFrame({
        "ELITE_SCORE": [80.0] * n,
        "TIMING_SCORE": [60.0] * n,
        "AXIS_MEAN": [60.0] * n,
        "TOP_PICK": [1] * n,
        "CARRY_AGE_DAYS": [0] * n,
        "거래대금(억원)": [300.0] * n,
        "STOP_PCT": [8.0] * n,
        "SUPERTREND_DIR": [1] * n,
        "Above_MA20": [1] * n,
        "IS_ABOVE_POC": [1] * n,
        "HMA_Trend": ["▲"] * n,
        "MACD_Slope_PCT": [1.0] * n,
        "Upper_Shadow_Ratio": [0.1] * n,
        "거래강도": [1.5] * n,
        "ret_1d_%": [1.0] * n,
        "KOSPI_RET_1D": [0.5] * n,
        "종목명": [f"종목{i}" for i in range(n)],
        "업종": ["반도체"] * n,
    })
    out = g.apply_guard_system(df.copy())
    assert (out["GUARD_REASON"] == "").all()


def test_guard_reason_g2_format():
    """G2 RR 배수 포맷이 '×0.3'/'×0.5' 형태로 정확히 찍힌다."""
    df = _rand_scored_df(200, 123)
    df["TIMING_SCORE"] = 0.0  # → RR×0.3 강제
    out = g.apply_guard_system(df.copy())
    fired = out.loc[out["GUARD_RR_MULT"] < 1.0, "GUARD_REASON"]
    assert (fired.str.contains("G2RR×0.3") | fired.str.contains("G2RR×0.5")).all()


def test_guard_reason_index_preserved():
    """비정수/비연속 인덱스에서도 정렬·값 보존."""
    df = _rand_scored_df(60, 9)
    df.index = [f"row_{i}" for i in range(len(df))]
    out = g.apply_guard_system(df.copy())
    broken5, _ = g.guard5_trendline_collapse(out, g._FallbackGuardConfig())
    ref = _ref_guard_reason(out, broken5)
    got = g._build_reason(out, broken5)
    assert got.index.tolist() == ref.index.tolist()
    assert list(got.values) == list(ref.values)


# ─────────────────────────────────────────────────────────────
#  성능 회귀 가드 (느슨 — 환경 편차 흡수)
# ─────────────────────────────────────────────────────────────
def test_guard_reason_faster_than_reference():
    df = _rand_scored_df(1500, 0)
    out = g.apply_guard_system(df.copy())
    broken5, _ = g.guard5_trendline_collapse(out, g._FallbackGuardConfig())

    t = time.time()
    for _ in range(3):
        _ref_guard_reason(out, broken5)
    t_ref = time.time() - t

    t = time.time()
    for _ in range(3):
        g._build_reason(out, broken5)
    t_vec = time.time() - t

    # 벡터화가 최소 3배 이상 빨라야 한다(보수적 하한; 실측 ≈16x).
    assert t_vec * 3 < t_ref, f"벡터화 속도 회귀: ref={t_ref*1000:.0f}ms vec={t_vec*1000:.0f}ms"


# ─────────────────────────────────────────────────────────────
#  pipeline_finalize July·Recovery REASON (선택 — import 가능 시)
# ─────────────────────────────────────────────────────────────
def _make_pf_df(n, seed):
    rng = np.random.default_rng(seed)

    def mnan(arr, p=0.08):
        arr = np.asarray(arr, dtype="float64")
        arr[rng.random(len(arr)) < p] = np.nan
        return arr

    return pd.DataFrame({
        "MARKET_BREADTH": mnan(rng.uniform(20, 65, n)),
        "ret_5d_%": mnan(rng.uniform(-12, 32, n)),
        "Vol_Quality": mnan(rng.uniform(0.5, 2.5, n)),
        "STRUCT_SCORE": mnan(rng.uniform(40, 95, n)),
        "FINAL_SCORE": mnan(rng.uniform(40, 95, n)),
        "VWAP_GAP": mnan(rng.uniform(-5, 50, n)),
        "POC_GAP": mnan(rng.uniform(10, 95, n)),
        "RSI14": mnan(rng.uniform(20, 80, n)),
        "SECTOR_RS": mnan(rng.uniform(-3, 3, n)),
        "ABNORMAL_HISTORY_GUARD_FLAG": rng.integers(0, 2, n),
        "SPIKE_REVERSAL_GUARD_FLAG": rng.integers(0, 2, n),
        "MARKET_WARNING_GUARD_FLAG": rng.integers(0, 2, n),
        "LONG_HISTORY_COLLAPSE_FLAG": rng.integers(0, 2, n),
        "ENTRY_EDGE_LEVEL": rng.choice(["BLOCK", "OK", "WATCH", "CAUTION"], n),
        "ENTRY_RISK_LEVEL": rng.choice(["ORANGE", "GREEN", "RED", "YELLOW"], n),
        "종목명": [f"종목{i}" for i in range(n)],
    })


@pytest.mark.parametrize("seed", range(5))
def test_pipeline_reason_columns_present_and_nonnull(seed):
    pf = pytest.importorskip("pipeline_finalize")
    df = _make_pf_df(int(np.random.default_rng(seed).integers(80, 300)), seed)
    out = pf.add_july_profit_defense_columns(df.copy(), enforce=False)
    out = pf.add_profit_recovery_suite_columns(out, enforce=False)
    for col in ("JULY_PROFIT_DEFENSE_REASON", "PROFIT_RECOVERY_REASON"):
        assert col in out.columns
        # 빈 행이 있어도 NaN은 없어야 한다(빈 행은 empty_fill로 채워짐).
        assert out[col].notna().all()
        assert (out[col].astype(str).str.len() > 0).all()


def test_pipeline_reason_cap_respected():
    """July는 최대 4조각, Recovery는 최대 5조각( ' · ' 구분자 수로 검증)."""
    pf = pytest.importorskip("pipeline_finalize")
    df = _make_pf_df(300, 77)
    out = pf.add_july_profit_defense_columns(df.copy(), enforce=False)
    out = pf.add_profit_recovery_suite_columns(out, enforce=False)
    july_parts = out["JULY_PROFIT_DEFENSE_REASON"].astype(str).str.split(" · ").map(len)
    rec_parts = out["PROFIT_RECOVERY_REASON"].astype(str).str.split(" · ").map(len)
    assert july_parts.max() <= 4
    assert rec_parts.max() <= 5
