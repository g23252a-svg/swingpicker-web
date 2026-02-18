# -*- coding: utf-8 -*-
"""
shared_utils.py — collector.py / dashboard.py 공용 유틸리티
─────────────────────────────────────────────────────────────
양쪽에서 복붙되어 있던 함수들을 단일 소스로 통합합니다.
"""
import math
import numpy as np
import pandas as pd
from typing import Any

# ───────────────────── 수치 안전 변환 ─────────────────────

def nz_num(s: Any) -> pd.Series:
    """문자열 혼합 Series → 숫자 변환 (실패 시 NaN)"""
    return pd.to_numeric(s, errors="coerce")


def safe_float(x, default: float = 0.0) -> float:
    """단일 값 → float 안전 변환 (NaN/None → default)"""
    try:
        if x is None:
            return default
        val = float(x)
        return default if (math.isnan(val) or math.isinf(val)) else val
    except Exception:
        return default


def _safe_sum(x: pd.Series) -> float:
    return pd.to_numeric(x, errors="coerce").fillna(0).sum()


def safe_quantile(s, q: float, fallback: float = 0.0) -> float:
    """Pandas Series 안전 분위수 (빈 데이터/에러 시 fallback)"""
    if s is None:
        return fallback
    try:
        if hasattr(s, 'empty') and s.empty:
            return fallback
        v = s.quantile(q)
        return fallback if pd.isna(v) else float(v)
    except Exception:
        return fallback


# ───────────────────── 이동평균 (공용) ─────────────────────

def ema(s: pd.Series, span: int) -> pd.Series:
    """지수 이동 평균"""
    return s.ewm(span=span, adjust=False).mean()


def wma(s: pd.Series, period: int) -> pd.Series:
    """가중 이동 평균 (HMA 계산 기반)"""
    weights = np.arange(1, period + 1)

    def _calc(x):
        return np.dot(x, weights) / weights.sum()

    return s.rolling(period).apply(_calc, raw=True)


def calc_hma(s: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average (HMA) — 빠른 반응 + 낮은 휩소"""
    if len(s) < period:
        return pd.Series(np.nan, index=s.index)

    half_length = int(period / 2)
    sqrt_length = int(math.sqrt(period))

    wma_half = wma(s, half_length)
    wma_full = wma(s, period)

    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt_length)


# ───────────────────── 정규화 / 클리핑 ─────────────────────

def cap_q(s: pd.Series, q: int = 90, floor: float = 1.0) -> float:
    c = np.nanpercentile(nz_num(s), q)
    return float(max(c, floor)) if np.isfinite(c) else floor


def pct_norm_pos(s: pd.Series, q: int = 90, floor: float = 1.0) -> pd.Series:
    s = nz_num(s).clip(lower=0)
    return np.clip(s / cap_q(s, q, floor), 0, 1)


def inv_dist_norm(dist: pd.Series, cap: float) -> pd.Series:
    return np.clip(1 - (nz_num(dist) / cap), 0, 1)
