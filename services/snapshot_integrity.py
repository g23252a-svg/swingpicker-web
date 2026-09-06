"""Validate recommendation snapshots without inventing dates or trading signals."""
from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import pandas as pd

DATE_COLUMNS = ("기준일", "trade_date", "DATA_DATE")


def normalize_ymd(value) -> str:
    """Accept explicit calendar dates, never epoch numbers or partial dates."""
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.strftime("%Y%m%d") if pd.notna(value) else ""
    text = str(value).strip()
    if re.fullmatch(r"\d{8}\.0", text):
        text = text[:-2]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        text = text.replace("-", "")
    if not re.fullmatch(r"\d{8}", text):
        return ""
    try:
        return datetime.strptime(text, "%Y%m%d").strftime("%Y%m%d")
    except ValueError:
        return ""


def snapshot_date(df: pd.DataFrame) -> str:
    for column in DATE_COLUMNS:
        if column in df:
            days = df[column].map(normalize_ymd)
            if days.ne("").all() and days.nunique() == 1:
                return days.iloc[0]
    return ""


def validate_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Reject truncated/mixed snapshots before they replace a working cache."""
    if df is None or df.empty:
        raise ValueError("추천 데이터가 비어 있습니다")
    missing = {"종목코드", "종목명", "종가"} - set(df.columns)
    if missing:
        raise ValueError("필수 컬럼 누락: " + ", ".join(sorted(missing)))
    out = df.copy()
    codes = out["종목코드"].fillna("").astype(str).str.strip()
    codes = codes.str.replace(r"\.0$", "", regex=True)
    if not codes.str.fullmatch(r"(?:[0-9]{1,6}|[0-9A-Z]{6})").all():
        raise ValueError("올바르지 않은 종목코드가 있습니다")
    out["종목코드"] = codes.str.zfill(6)
    if out["종목코드"].duplicated().any():
        raise ValueError("중복 종목코드가 있습니다")
    observed = []
    for column in DATE_COLUMNS:
        if column not in out:
            continue
        days = out[column].map(normalize_ymd)
        if days.eq("").any() or days.nunique() != 1:
            raise ValueError(f"{column}: 기준일이 없거나 서로 다릅니다")
        observed.append(days.iloc[0])
    if len(set(observed)) > 1:
        raise ValueError("기준일 컬럼끼리 날짜가 다릅니다")
    # Undated legacy files remain readable; the UI explicitly says unknown.
    return out


def input_integrity_mask(df: pd.DataFrame) -> pd.Series:
    """Present-but-invalid prices/ratios cannot become an executable buy.

    Missing legacy columns are tolerated. Strategy thresholds are unchanged.
    """
    valid = pd.Series(True, index=df.index, dtype=bool)
    for column in ("종가", "추천매수가", "손절가", "추천매도가1", "RR_NOW_TP1"):
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce")
            valid &= np.isfinite(values) & values.gt(0)
    if {"추천매수가", "손절가", "추천매도가1"}.issubset(df.columns):
        entry = pd.to_numeric(df["추천매수가"], errors="coerce")
        stop = pd.to_numeric(df["손절가"], errors="coerce")
        target = pd.to_numeric(df["추천매도가1"], errors="coerce")
        valid &= stop.lt(entry) & entry.lt(target)
    if "ALPHA_SCORE" in df and "ALPHA_GATE_ACTIVE" in df:
        active = df["ALPHA_GATE_ACTIVE"].astype(str).str.lower().isin(
            {"1", "1.0", "true"})
        alpha = pd.to_numeric(df["ALPHA_SCORE"], errors="coerce")
        valid &= ~active | (np.isfinite(alpha) & alpha.between(0, 100))
    return valid.fillna(False)


def freshness_summary(df: pd.DataFrame) -> dict:
    """Display actual snapshot/price dates, independently of wall-clock time."""
    batch = snapshot_date(df) if df is not None and not df.empty else ""
    prices = []
    if df is not None and "PRICE_ASOF" in df:
        prices = sorted(set(df["PRICE_ASOF"].map(normalize_ymd)) - {""})
    price = "확인 불가"
    if prices:
        price = prices[0] if len(prices) == 1 else f"{prices[0]} ~ {prices[-1]}"
    stale = False
    if df is not None and "SESSION_STALE" in df:
        stale = df["SESSION_STALE"].astype(str).str.lower().isin(
            {"1", "1.0", "true"}).any()
    stale = bool(stale or (batch and prices and prices[0] < batch))
    return {"batch_date": batch or "확인 불가", "price_date": price,
            "stale": stale, "unknown": not batch or not prices}
