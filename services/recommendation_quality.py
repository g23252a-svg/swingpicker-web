# -*- coding: utf-8 -*-
"""Final recommendation contract for the v26.1 profit-recovery policy.

The legacy rank, TOP_PICK and ML columns remain useful research inputs, but
none of them is an order instruction by itself.  ``PRODUCTION_BUY == 1`` is
the only official new-entry contract.

Policy v1 was selected with chronological, fill-aware OHLC validation:

* next three sessions must touch the recommended limit price;
* costs are deducted;
* same-bar stop/target ambiguity is resolved to the stop (conservative);
* train and later test periods both had positive mean return;
* no more than one new position is allowed per day.

The sample is still small, so this module deliberately caps position size and
never claims that the observed edge is guaranteed to persist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from price_utils import round_to_tick


POLICY_VERSION = "profit_recovery_v1"
POLICY_SOURCE = "fill_aware_chronological_oos"

MIN_MARKET_BREADTH = 30.0
MIN_TURNOVER_EOK = 30.0
STOP_LOSS_PCT = 6.0
TAKE_PROFIT_PCT = 10.0
MAX_NEW_POSITIONS = 1

DANGEROUS_MACRO = frozenset({"WARNING", "CRITICAL"})
BLOCKED_ENTRY_RISK = frozenset({"ORANGE", "RED", "BLACK"})


def _num(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _text(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[column].fillna(default).astype(str).str.strip().str.upper()


def _flag(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return (
        df[column]
        .fillna(0)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "1.0", "true", "t", "yes", "y"})
    )


def profit_recovery_rank_score(df: pd.DataFrame) -> pd.Series:
    """Return the exact transparent rank used by the validated policy."""
    score = (
        _num(df, "DISPLAY_SCORE", 0).clip(0, 100) * 0.30
        + _num(df, "FINAL_SCORE", 0).clip(0, 100) * 0.25
        + _num(df, "STRUCT_SCORE", 0).clip(0, 100) * 0.20
        + _num(df, "TIMING_SCORE", 0).clip(0, 100) * 0.10
        + _num(df, "BUY_NOW_SCORE", 0).clip(0, 100) * 0.10
        + _num(df, "MFI14", 50).clip(0, 100) * 0.05
    )
    return score.clip(0, 100).round(2)


def profit_recovery_candidate_mask(df: pd.DataFrame) -> pd.Series:
    """Technical/liquidity setup mask before the daily market-regime gate.

    This is intentionally independent of legacy TOP_PICK/route flags.  The
    previous production guard required flags that were never simultaneously
    true in the available history, which made the official list permanently
    empty even when a validated setup was present.
    """
    if df is None or len(df) == 0:
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)

    entry = _num(df, "추천매수가", 0)
    mfi = _num(df, "MFI14", 50)
    resistance = _num(df, "RES_RATIO_NEAR", 0)
    vwap_gap = _num(df, "VWAP_GAP", 99)
    rr = _num(df, "RR_NOW_TP1", 0)
    display = _num(df, "DISPLAY_SCORE", 0)
    turnover = _num(df, "거래대금(억원)", 0)
    risk = _text(df, "ENTRY_RISK_LEVEL", "UNKNOWN")

    return (
        (entry > 0)
        & (mfi >= 60)
        & (mfi <= 88)
        & (resistance >= 0.03)
        & (vwap_gap >= -20)
        & (vwap_gap <= 16)
        & (rr >= 1.20)
        & (display >= 55)
        & (turnover >= MIN_TURNOVER_EOK)
        & ~risk.isin(BLOCKED_ENTRY_RISK)
        & ~_flag(df, "ABNORMAL_HISTORY_GUARD_FLAG")
        & ~_flag(df, "PROFIT_RECOVERY_BLOCK_FLAG")
        & ~_flag(df, "JULY_PROFIT_BLOCK_FLAG")
        & ~_flag(df, "NO_CHASE_FLAG")
        & _flag(df, "DATA_FRESHNESS_OK", True)
    )


def _production_prices(entry: pd.Series, production: pd.Series) -> tuple[pd.Series, pd.Series]:
    stop = pd.Series(0, index=entry.index, dtype="int64")
    target = pd.Series(0, index=entry.index, dtype="int64")
    for idx in entry.index[production]:
        price = float(entry.loc[idx])
        stop.loc[idx] = int(round_to_tick(price * (1 - STOP_LOSS_PCT / 100), "down") or 0)
        target.loc[idx] = int(round_to_tick(price * (1 + TAKE_PROFIT_PCT / 100), "up") or 0)
    return stop, target


def _reasons(
    df: pd.DataFrame,
    setup: pd.Series,
    regime_pass: pd.Series,
    production: pd.Series,
) -> pd.Series:
    breadth = _num(df, "MARKET_BREADTH", 0)
    macro = _text(df, "MACRO_RISK", "NORMAL")
    mfi = _num(df, "MFI14", 50)
    resistance = _num(df, "RES_RATIO_NEAR", 0)
    vwap_gap = _num(df, "VWAP_GAP", 99)
    rr = _num(df, "RR_NOW_TP1", 0)
    display = _num(df, "DISPLAY_SCORE", 0)
    turnover = _num(df, "거래대금(억원)", 0)
    risk = _text(df, "ENTRY_RISK_LEVEL", "UNKNOWN")

    result: list[str] = []
    for pos in range(len(df)):
        if bool(production.iloc[pos]):
            result.append("검증 규칙 통과 · 지정가 체결 시 -6% 손절 / +10% 익절")
            continue

        row_reasons: list[str] = []
        if bool(setup.iloc[pos]) and not bool(regime_pass.iloc[pos]):
            if breadth.iloc[pos] < MIN_MARKET_BREADTH:
                row_reasons.append(
                    f"시장 상승폭 {breadth.iloc[pos]:.1f}% < {MIN_MARKET_BREADTH:.0f}%"
                )
            if macro.iloc[pos] in DANGEROUS_MACRO:
                row_reasons.append(f"시장위험 {macro.iloc[pos]}")
        elif bool(setup.iloc[pos]) and bool(regime_pass.iloc[pos]):
            row_reasons.append("당일 신규진입 1종목 제한")
        else:
            if mfi.iloc[pos] < 60 or mfi.iloc[pos] > 88:
                row_reasons.append(f"MFI {mfi.iloc[pos]:.0f} (기준 60~88)")
            if resistance.iloc[pos] < 0.03:
                row_reasons.append("상단 여력 3% 미만")
            if vwap_gap.iloc[pos] < -20 or vwap_gap.iloc[pos] > 16:
                row_reasons.append(f"VWAP 이격 {vwap_gap.iloc[pos]:+.1f}%")
            if rr.iloc[pos] < 1.20:
                row_reasons.append(f"손익비 {rr.iloc[pos]:.2f} < 1.20")
            if display.iloc[pos] < 55:
                row_reasons.append(f"기본점수 {display.iloc[pos]:.0f} < 55")
            if turnover.iloc[pos] < MIN_TURNOVER_EOK:
                row_reasons.append(f"거래대금 {turnover.iloc[pos]:.1f}억 < 30억")
            if risk.iloc[pos] in BLOCKED_ENTRY_RISK:
                row_reasons.append(f"진입위험 {risk.iloc[pos]}")
            if _flag(df.iloc[[pos]], "NO_CHASE_FLAG").iloc[0]:
                row_reasons.append("추격매수 금지")

        suffix = " · 매수 아님" if bool(setup.iloc[pos]) else ""
        result.append((" · ".join(row_reasons[:3]) or "검증 기준 미달") + suffix)
    return pd.Series(result, index=df.index, dtype="object")


def apply_recommendation_quality_guard(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the fill-aware v26.1 production contract."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    output_columns = (
        "PRE_QUALITY_BUY_NOW_ELIGIBLE",
        "PROFIT_RECOVERY_CANDIDATE",
        "PROFIT_RECOVERY_RANK_SCORE",
        "QUALITY_GUARD_SCORE",
        "QUALITY_GUARD_PASS",
        "QUALITY_GUARD_REASON",
        "PRODUCTION_BUY",
        "PRODUCTION_ENTRY_PRICE",
        "PRODUCTION_STOP_PRICE",
        "PRODUCTION_TARGET_PRICE",
        "PRODUCTION_RR",
        "PRODUCTION_SOURCE",
        "ACTION_DECISION",
        "RECOMMENDED_WEIGHT_PCT",
        "QUALITY_POLICY_VERSION",
        "QUALITY_POLICY_SOURCE",
    )
    if out.empty:
        for column in output_columns:
            out[column] = pd.Series(dtype="object")
        return out

    eligible_source = (
        "PRE_QUALITY_BUY_NOW_ELIGIBLE"
        if "PRE_QUALITY_BUY_NOW_ELIGIBLE" in out.columns
        else "BUY_NOW_ELIGIBLE"
    )
    eligible_before = _flag(out, eligible_source)
    out["PRE_QUALITY_BUY_NOW_ELIGIBLE"] = eligible_before.astype(int)

    setup = profit_recovery_candidate_mask(out)
    score = profit_recovery_rank_score(out)
    breadth = _num(out, "MARKET_BREADTH", 0)
    macro = _text(out, "MACRO_RISK", "NORMAL")
    regime_pass = (
        setup
        & (breadth >= MIN_MARKET_BREADTH)
        & ~macro.isin(DANGEROUS_MACRO)
    )

    # Deterministic tie-break: validated rank, display score, then stable row
    # order.  At most one new position can be opened on any signal date.
    display = _num(out, "DISPLAY_SCORE", 0)
    rank_key = score * 1_000 + display
    production = regime_pass & rank_key.where(regime_pass).rank(
        method="first", ascending=False
    ).le(MAX_NEW_POSITIONS)

    entry = _num(out, "추천매수가", 0)
    stop, target = _production_prices(entry, production)

    out["PROFIT_RECOVERY_CANDIDATE"] = setup.astype(int)
    out["PROFIT_RECOVERY_RANK_SCORE"] = score
    out["QUALITY_GUARD_SCORE"] = score
    out["QUALITY_GUARD_PASS"] = regime_pass.astype(int)
    out["PRODUCTION_BUY"] = production.astype(int)
    out["PRODUCTION_ENTRY_PRICE"] = np.where(production, entry, 0)
    out["PRODUCTION_STOP_PRICE"] = stop
    out["PRODUCTION_TARGET_PRICE"] = target
    out["PRODUCTION_RR"] = np.where(
        production & (entry > stop),
        (target - entry) / (entry - stop).replace(0, np.nan),
        0.0,
    )
    out["PRODUCTION_RR"] = pd.to_numeric(
        out["PRODUCTION_RR"], errors="coerce"
    ).fillna(0).round(2)
    out["QUALITY_GUARD_REASON"] = _reasons(out, setup, regime_pass, production)
    out["QUALITY_POLICY_VERSION"] = POLICY_VERSION
    out["QUALITY_POLICY_SOURCE"] = POLICY_SOURCE
    out["PRODUCTION_SOURCE"] = np.select(
        [production, setup],
        ["VALIDATED_PROFIT_RECOVERY", "WATCHLIST_ONLY"],
        default="NONE",
    )

    # Legacy consumers must see exactly the same official decision.
    out["BUY_NOW_ELIGIBLE"] = production.astype(int)

    watch = setup & ~production
    out["ACTION_DECISION"] = np.select(
        [production, watch], ["BUY", "WATCH"], default="CASH"
    )
    out["RECOMMENDED_WEIGHT_PCT"] = np.where(
        production,
        np.where(macro.eq("CAUTION"), 2.0, 3.0),
        0.0,
    )
    return out


def production_buy_mask(df: pd.DataFrame) -> pd.Series:
    """Read the official contract, with a conservative legacy fallback."""
    if df is None or len(df) == 0:
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)
    if "PRODUCTION_BUY" in df.columns:
        return _flag(df, "PRODUCTION_BUY")
    return _flag(df, "TOP_PICK") & _flag(df, "BUY_NOW_ELIGIBLE")
