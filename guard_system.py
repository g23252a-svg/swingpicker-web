# -*- coding: utf-8 -*-
"""
guard_system.py — v23.0 통합 GUARD 엔진 (SSOT)
═══════════════════════════════════════════════════
8개 GUARD 규칙을 단일 모듈에서 벡터화 적용한다.

설계 원칙
  1. 순수 함수: 입력 df → GUARD_* 컬럼 부여한 df 반환 (in-place 아님).
  2. compute_elite_score 직후 · Kelly 직전에 apply_guard_system() 1회 호출.
  3. 임계값은 전부 collector_config.GuardConfig(SSOT)에서 주입 (하드코딩 금지).
  4. 컬럼 누락에 강건 — 모든 입력은 _num/_txt safe accessor 경유, 없으면 guard PASS.
  5. ON/OFF 스위치: GuardConfig.guard_enforce_top_pick=False면 shadow 컬럼만 부여하고
     TOP_PICK/ELITE_LABEL은 건드리지 않는다 (combo backtest OFF 모드).

GUARD 규칙 요약
  G1 유동성-손절 차단:  거래대금<100억 & STOP_PCT≤6      → BLOCK
  G2 RR 열화:           TIMING=0 → RR×0.3, AXIS_MEAN<40 → RR×0.5
  G3 CARRY STALE 감점:  5/7/10일차 누적 +15/+10/+20점 감점
  G4 저모멘텀 섹터 게이트: 지주/금융/SI 등 & TIMING<30      → BLOCK
  G5 추세선 붕괴 경보:  SUPERTREND/MA20/POC/HMA/MACD 3개+ → FORCE_EXIT_ALERT + 감점
  G6 시장 역행 감점:    KOSPI+2% & 종목-5%                → -25점
  G7 윗꼬리 약세 감점:  Upper_Shadow>0.5 & 거래강도<0.7    → -15점
  G8 CARRY 사전경고:    CARRY_AGE_DAYS=4                  → PRE_WARNING(감점/차단 없음)

산출 컬럼
  GUARD_PASS_1..8        : 각 규칙 통과 여부(bool)
  GUARD_PENALTY_1..8     : 각 규칙 점수 감점(G2는 0, 배수는 GUARD_RR_MULT)
  GUARD_PENALTY_TOTAL    : 점수 감점 합계(G3+G5+G6+G7)
  GUARD_RR_MULT          : G2 RR 배수(0.3/0.5/1.0)
  GUARD_BLOCK            : 하드 차단(G1 or G4)
  GUARD_FORCE_EXIT_ALERT : G5 강제 청산 경보
  GUARD_PRE_WARNING      : G8 사전 경고
  GUARD_ALL_PASS         : 핵심 가드(1~7) 전부 통과 & 미차단
  GUARDED_ELITE_SCORE    : clip(ELITE×RR_MULT − PENALTY_TOTAL, 0, 100), 차단 시 0
  GUARD_KELLY_MULT       : Kelly 분율 축소 배수(0~1) — kelly_calibrator가 소비
  GUARD_REASON           : 발동 가드 한 줄 요약
  ELITE_LABEL            : 가드 통과한 TOP_PICK만 'ELITE'
  TOP_PICK_RAW           : 가드 적용 전 원본 TOP_PICK (enforce 모드에서만)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("guard_system")

# 산출 컬럼 계약 — check_contract_gate.py가 이 목록을 검증한다.
GUARD_PASS_COLS = [f"GUARD_PASS_{i}" for i in range(1, 9)]
GUARD_PENALTY_COLS = [f"GUARD_PENALTY_{i}" for i in range(1, 9)]
GUARD_AGG_COLS = [
    "GUARD_PENALTY_TOTAL", "GUARD_RR_MULT", "GUARD_BLOCK",
    "GUARD_FORCE_EXIT_ALERT", "GUARD_PRE_WARNING", "GUARD_ALL_PASS",
    "GUARDED_ELITE_SCORE", "GUARD_KELLY_MULT", "GUARD_REASON", "ELITE_LABEL",
]
GUARD_CONTRACT_COLS = GUARD_PASS_COLS + GUARD_PENALTY_COLS + GUARD_AGG_COLS

# 핵심 가드(ELITE_LABEL/ALL_PASS 판정) — G8(사전경고)은 제외.
CRITICAL_GUARD_IDS = (1, 2, 3, 4, 5, 6, 7)


# ── safe accessors ──────────────────────────────────────────────
def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _txt(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].astype("object").fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def _flag_below(series: pd.Series, threshold: float) -> pd.Series:
    """NaN은 '판정 불가' → False(붕괴 아님)로 보수 처리."""
    return (series < threshold).fillna(False)


# ── 개별 GUARD 규칙 ──────────────────────────────────────────────
def guard1_liquidity_stop(df: pd.DataFrame, cfg) -> pd.Series:
    """G1: 거래대금<min & STOP_PCT≤max → 차단(True=위반)."""
    turnover = _num(df, "거래대금(억원)", default=np.inf)
    stop_pct = _num(df, "STOP_PCT", default=np.inf)
    violated = (turnover < cfg.g1_turnover_min_eok) & (stop_pct <= cfg.g1_stop_pct_max)
    return violated.fillna(False)


def guard2_rr_degrade(df: pd.DataFrame, cfg) -> pd.Series:
    """G2: TIMING=0 → ×rr0, AXIS_MEAN<min → ×axis. 둘 중 작은 배수 채택."""
    timing = _num(df, "TIMING_SCORE", default=100.0)
    axis = _num(df, "AXIS_MEAN", default=100.0)
    mult = pd.Series(1.0, index=df.index, dtype="float64")
    mult = mult.where(~(timing <= cfg.g2_timing_zero_eps).fillna(False),
                      cfg.g2_rr_mult_timing0)
    mult = np.minimum(
        mult,
        pd.Series(1.0, index=df.index).where(
            ~(axis < cfg.g2_axis_min).fillna(False), cfg.g2_rr_mult_axis_low
        ),
    )
    return pd.Series(mult, index=df.index).astype("float64")


def guard3_carry_stale(df: pd.DataFrame, cfg) -> pd.Series:
    """G3: 보유경과 5/7/10일차 누적 감점(점수). 컬럼 없으면 0."""
    age = _num(df, "CARRY_AGE_DAYS", default=0.0).fillna(0.0)
    pen = pd.Series(0.0, index=df.index, dtype="float64")
    pen = pen + np.where(age >= 5, cfg.g3_pen_day5, 0.0)
    pen = pen + np.where(age >= 7, cfg.g3_pen_day7, 0.0)
    pen = pen + np.where(age >= 10, cfg.g3_pen_day10, 0.0)
    return pd.Series(pen, index=df.index).astype("float64")


def guard4_low_momentum_sector(df: pd.DataFrame, cfg) -> pd.Series:
    """G4: 저모멘텀 섹터(키워드 매칭) & TIMING<gate → 차단(True=위반)."""
    timing = _num(df, "TIMING_SCORE", default=100.0)
    blob = (
        _txt(df, "업종_대분류") + "|"
        + _txt(df, "업종") + "|"
        + _txt(df, "업종_상세") + "|"
        + _txt(df, "종목명")
    )
    kw = [k for k in cfg.g4_low_mom_keywords if k]
    is_low = pd.Series(False, index=df.index)
    for k in kw:
        is_low = is_low | blob.str.contains(k, case=False, na=False, regex=False)
    violated = is_low & (timing < cfg.g4_timing_gate).fillna(True)
    return violated.fillna(False)


def guard5_trendline_collapse(df: pd.DataFrame, cfg):
    """G5: 추세선 붕괴 개수 산정 → (붕괴수, 경보bool). 5축 중 break_min 이상이면 경보."""
    broken = pd.Series(0, index=df.index, dtype="int64")

    st_dir = _num(df, "SUPERTREND_DIR", default=1.0)
    broken = broken + (st_dir < 0).fillna(False).astype(int)

    above_ma20 = _num(df, "Above_MA20", default=1.0)
    broken = broken + (above_ma20 <= 0).fillna(False).astype(int)

    # POC: IS_ABOVE_POC=0 또는 POC_GAP<0 → 이탈
    if "IS_ABOVE_POC" in df.columns:
        below_poc = (_num(df, "IS_ABOVE_POC", default=1.0) <= 0).fillna(False)
    else:
        below_poc = (_num(df, "POC_GAP", default=0.0) < 0).fillna(False)
    broken = broken + below_poc.astype(int)

    # HMA: HMA_Trend '▼' 또는 HMA_On=0
    hma_trend = _txt(df, "HMA_Trend", default="▲")
    hma_on = _num(df, "HMA_On", default=1.0)
    hma_broken = hma_trend.str.contains("▼", na=False) | (hma_on <= 0).fillna(False)
    broken = broken + hma_broken.astype(int)

    macd_slope = _num(df, "MACD_Slope_PCT", default=0.0)
    broken = broken + (macd_slope < 0).fillna(False).astype(int)

    alert = (broken >= cfg.g5_break_min)
    return broken, alert


def guard6_market_divergence(df: pd.DataFrame, cfg,
                             kospi_ret_1d: Optional[float]) -> pd.Series:
    """G6: 장 강세(+2%) & 종목 급락(-5%) → 위반(True). KOSPI값 없으면 전부 False."""
    if kospi_ret_1d is None:
        col = _num(df, "KOSPI_RET_1D", default=np.nan)
        if "MARKET_RET_1D" in df.columns:
            col = col.fillna(_num(df, "MARKET_RET_1D", default=np.nan))
        market_up = (col >= cfg.g6_kospi_up_pct).fillna(False)
    else:
        market_up = pd.Series(
            bool(kospi_ret_1d >= cfg.g6_kospi_up_pct), index=df.index
        )
    stock_ret = _num(df, "ret_1d_%", default=0.0)
    if "ret_1d_%" not in df.columns:
        stock_ret = _num(df, "ret_1d", default=0.0)
    stock_down = (stock_ret <= cfg.g6_stock_down_pct).fillna(False)
    return (market_up & stock_down).fillna(False)


def guard7_upper_shadow_weak(df: pd.DataFrame, cfg) -> pd.Series:
    """G7: 윗꼬리비율>0.5 & 거래강도<0.7 → 위반(True)."""
    shadow = _num(df, "Upper_Shadow_Ratio", default=np.nan)
    if "Upper_Shadow_Ratio" not in df.columns:
        shadow = _num(df, "V23_Upper_Shadow_Ratio", default=np.nan)
    vol_int = _num(df, "거래강도", default=np.inf)
    violated = (shadow > cfg.g7_shadow_max) & (vol_int < cfg.g7_vol_intensity_min)
    return violated.fillna(False)


def guard8_carry_prewarning(df: pd.DataFrame, cfg) -> pd.Series:
    """G8: CARRY_AGE_DAYS == prewarn_day → 사전경고(True). 감점/차단 없음."""
    age = _num(df, "CARRY_AGE_DAYS", default=-1.0).fillna(-1.0)
    return (age == cfg.g8_prewarn_day)


# ── 메인 엔진 ────────────────────────────────────────────────────
def apply_guard_system(df: pd.DataFrame,
                       config=None,
                       kospi_ret_1d: Optional[float] = None) -> pd.DataFrame:
    """8개 GUARD를 적용하고 GUARD_* / ELITE_LABEL / (옵션)재게이트된 TOP_PICK 부여.

    Parameters
    ----------
    df : 스코어링 완료 DataFrame (ELITE_SCORE/TIMING_SCORE/... 포함 권장)
    config : CollectorConfig 또는 GuardConfig. None이면 DEFAULT_CONFIG.guard 사용.
    kospi_ret_1d : 당일 KOSPI 등락률(%). None이면 컬럼/스킵 fallback.
    """
    if df is None or len(df) == 0:
        return df

    cfg = _resolve_guard_config(config)
    out = df.copy()
    idx = out.index

    # ── 개별 규칙 ──
    g1 = guard1_liquidity_stop(out, cfg)                 # 위반 bool
    rr_mult = guard2_rr_degrade(out, cfg)                # 배수
    pen3 = guard3_carry_stale(out, cfg)                  # 점수
    g4 = guard4_low_momentum_sector(out, cfg)            # 위반 bool
    broken5, alert5 = guard5_trendline_collapse(out, cfg)
    g6 = guard6_market_divergence(out, cfg, kospi_ret_1d)  # 위반 bool
    g7 = guard7_upper_shadow_weak(out, cfg)              # 위반 bool
    g8 = guard8_carry_prewarning(out, cfg)               # 경고 bool

    pen5 = pd.Series(np.where(alert5, cfg.g5_penalty, 0.0), index=idx).astype("float64")
    pen6 = pd.Series(np.where(g6, cfg.g6_penalty, 0.0), index=idx).astype("float64")
    pen7 = pd.Series(np.where(g7, cfg.g7_penalty, 0.0), index=idx).astype("float64")

    # ── PASS 컬럼 ──
    out["GUARD_PASS_1"] = (~g1).astype(bool)
    out["GUARD_PASS_2"] = (rr_mult >= 1.0)
    out["GUARD_PASS_3"] = (pen3 <= 0)
    out["GUARD_PASS_4"] = (~g4).astype(bool)
    out["GUARD_PASS_5"] = (~alert5).astype(bool)
    out["GUARD_PASS_6"] = (~g6).astype(bool)
    out["GUARD_PASS_7"] = (~g7).astype(bool)
    out["GUARD_PASS_8"] = (~g8).astype(bool)  # 경고 없으면 True

    # ── PENALTY 컬럼 (G1/G2/G4/G8은 점수감점 0 — 차단/배수/경고로 처리) ──
    out["GUARD_PENALTY_1"] = 0.0
    out["GUARD_PENALTY_2"] = 0.0
    out["GUARD_PENALTY_3"] = pen3.round(1)
    out["GUARD_PENALTY_4"] = 0.0
    out["GUARD_PENALTY_5"] = pen5.round(1)
    out["GUARD_PENALTY_6"] = pen6.round(1)
    out["GUARD_PENALTY_7"] = pen7.round(1)
    out["GUARD_PENALTY_8"] = 0.0

    # ── 집계 ──
    penalty_total = (pen3 + pen5 + pen6 + pen7).round(1)
    out["GUARD_PENALTY_TOTAL"] = penalty_total
    out["GUARD_RR_MULT"] = rr_mult.round(3)
    out["GUARD_BLOCK"] = (g1 | g4).astype(bool)
    out["GUARD_FORCE_EXIT_ALERT"] = alert5.astype(bool)
    out["GUARD_PRE_WARNING"] = g8.astype(bool)
    out["GUARD_TRENDLINE_BROKEN"] = broken5.astype("int64")

    crit_pass = pd.Series(True, index=idx)
    for i in CRITICAL_GUARD_IDS:
        crit_pass = crit_pass & out[f"GUARD_PASS_{i}"]
    out["GUARD_ALL_PASS"] = (crit_pass & ~out["GUARD_BLOCK"]).astype(bool)

    # ── GUARDED_ELITE_SCORE & Kelly 배수 ──
    elite = _num(out, "ELITE_SCORE", default=0.0).fillna(0.0)
    guarded = (elite * rr_mult - penalty_total).clip(lower=0, upper=100)
    guarded = guarded.where(~out["GUARD_BLOCK"], 0.0)
    out["GUARDED_ELITE_SCORE"] = guarded.round(1)

    kelly_mult = (guarded / elite.clip(lower=1)).clip(0, 1)
    kelly_mult = kelly_mult.where(~out["GUARD_BLOCK"], 0.0)
    out["GUARD_KELLY_MULT"] = kelly_mult.round(3)

    # ── 사유 문자열 ──
    out["GUARD_REASON"] = _build_reason(out, broken5)

    # ── ELITE_LABEL & TOP_PICK 재게이트 ──
    has_top = "TOP_PICK" in out.columns
    top_raw = (
        pd.to_numeric(out["TOP_PICK"], errors="coerce").fillna(0).astype(int)
        if has_top else pd.Series(0, index=idx)
    )

    label = pd.Series("", index=idx, dtype="object")
    is_elite = (top_raw == 1) & out["GUARD_ALL_PASS"] & (guarded >= cfg.guard_top_pick_min)
    is_blocked_pick = (top_raw == 1) & ~is_elite
    label = label.mask(is_elite, "ELITE")
    label = label.mask(is_blocked_pick, "GUARD_BLOCKED")
    out["ELITE_LABEL"] = label

    if cfg.guard_enforce_top_pick and has_top:
        out["TOP_PICK_RAW"] = top_raw
        out["TOP_PICK"] = is_elite.astype(int)
        if "TOP_PICK_TYPE" in out.columns:
            # object 캐스팅 후 할당 — all-NaN float 컬럼에 ""대입 시 dtype 충돌 방지
            out["TOP_PICK_TYPE"] = out["TOP_PICK_TYPE"].astype("object")
            out.loc[~is_elite, "TOP_PICK_TYPE"] = ""

    return out


def _build_reason(out: pd.DataFrame, broken5: pd.Series) -> pd.Series:
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


def guard_summary(df: pd.DataFrame) -> dict:
    """로깅용 요약 — 각 가드 발동 건수."""
    if df is None or "GUARD_BLOCK" not in df.columns:
        return {}
    s = {
        "n_rows": int(len(df)),
        "n_block": int(df["GUARD_BLOCK"].sum()),
        "n_force_exit": int(df["GUARD_FORCE_EXIT_ALERT"].sum()),
        "n_pre_warning": int(df["GUARD_PRE_WARNING"].sum()),
        "n_all_pass": int(df["GUARD_ALL_PASS"].sum()),
    }
    for i in range(1, 9):
        c = f"GUARD_PASS_{i}"
        if c in df.columns:
            s[f"g{i}_fail"] = int((~df[c]).sum())
    if "ELITE_LABEL" in df.columns:
        s["n_elite"] = int((df["ELITE_LABEL"] == "ELITE").sum())
        s["n_guard_blocked_pick"] = int((df["ELITE_LABEL"] == "GUARD_BLOCKED").sum())
    return s


def _resolve_guard_config(config):
    """CollectorConfig / GuardConfig / None → GuardConfig 인스턴스."""
    if config is None:
        try:
            from collector_config import DEFAULT_CONFIG
            return DEFAULT_CONFIG.guard
        except Exception:
            return _FallbackGuardConfig()
    # GuardConfig 자체
    if hasattr(config, "g1_turnover_min_eok"):
        return config
    # CollectorConfig facade
    if hasattr(config, "guard"):
        return config.guard
    return _FallbackGuardConfig()


class _FallbackGuardConfig:
    """collector_config 미연동 환경용 폴백(기본값) — 테스트/단독 실행."""
    g1_turnover_min_eok = 100.0
    g1_stop_pct_max = 6.0
    g2_timing_zero_eps = 0.0
    g2_rr_mult_timing0 = 0.3
    g2_axis_min = 40.0
    g2_rr_mult_axis_low = 0.5
    g3_pen_day5 = 15.0
    g3_pen_day7 = 10.0
    g3_pen_day10 = 20.0
    g4_timing_gate = 30.0
    g4_low_mom_keywords = ("지주", "홀딩스", "금융지주", "SI", "시스템통합", "전산")
    g5_break_min = 3
    g5_penalty = 20.0
    g6_kospi_up_pct = 2.0
    g6_stock_down_pct = -5.0
    g6_penalty = 25.0
    g7_shadow_max = 0.5
    g7_vol_intensity_min = 0.7
    g7_penalty = 15.0
    g8_prewarn_day = 4
    guard_top_pick_min = 60.0
    guard_enforce_top_pick = True
