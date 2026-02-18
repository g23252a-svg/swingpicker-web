# -*- coding: utf-8 -*-
"""
indicators.py — 기술적 지표 순수 함수 모음
──────────────────────────────────────────
collector.py 에서 추출한 지표 계산 함수들.
외부 상태(DB, 네트워크)에 의존하지 않는 순수 함수만 포함.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


# ───────────────────── RSI ─────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (0–100)
    - 완전 상승 = 100, 완전 하락 = 0, 변동 없음 = 50
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    both_zero = (roll_up == 0) & (roll_down == 0)
    rsi = rsi.where(~both_zero, 50)
    rsi = rsi.where(~((roll_down == 0) & (roll_up != 0)), 100)
    rsi = rsi.where(~((roll_up == 0) & (roll_down != 0)), 0)

    return rsi


# ───────────────────── ATR ─────────────────────

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    """Average True Range"""
    tr = pd.concat(
        [(high - low),
         (high - close.shift(1)).abs(),
         (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


# ───────────────────── SuperTrend ─────────────────────

def calc_supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 10, multiplier: float = 3.0
                    ) -> Tuple[pd.Series, pd.Series]:
    """
    SuperTrend 지표 → (supertrend_line, trend_direction)
    trend_direction: 1 = 상승, -1 = 하락
    """
    atr = calc_atr(high, low, close, period)

    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    st_out = [np.nan] * len(close)
    trend_out = [1] * len(close)

    vals_c = close.values
    vals_bu = basic_upper.values
    vals_bl = basic_lower.values

    start_idx = period
    if start_idx >= len(close):
        return (pd.Series(st_out, index=close.index),
                pd.Series(trend_out, index=close.index))

    final_upper = vals_bu[start_idx]
    final_lower = vals_bl[start_idx]

    st_out[start_idx] = final_lower
    trend_out[start_idx] = 1

    for i in range(start_idx + 1, len(close)):
        if (vals_bu[i] < final_upper) or (vals_c[i - 1] > final_upper):
            final_upper = vals_bu[i]
        if (vals_bl[i] > final_lower) or (vals_c[i - 1] < final_lower):
            final_lower = vals_bl[i]

        prev_trend = trend_out[i - 1]
        if prev_trend == 1:
            if vals_c[i] < final_lower:
                curr_trend = -1
                final_upper = vals_bu[i]
            else:
                curr_trend = 1
        else:
            if vals_c[i] > final_upper:
                curr_trend = 1
                final_lower = vals_bl[i]
            else:
                curr_trend = -1

        trend_out[i] = curr_trend
        st_out[i] = final_upper if curr_trend == -1 else final_lower

    return (pd.Series(st_out, index=close.index),
            pd.Series(trend_out, index=close.index))


# ───────────────────── MFI ─────────────────────

def calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
             vol: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index (0–100)"""
    tp = (high + low + close) / 3
    rmf = tp * vol
    pos = np.where(tp.diff() > 0, rmf, 0)
    neg = np.where(tp.diff() < 0, rmf, 0)
    pos_s = pd.Series(pos, index=close.index).rolling(period).sum()
    neg_s = pd.Series(neg, index=close.index).rolling(period).sum().replace(0, 1)
    return 100 - (100 / (1 + (pos_s / neg_s)))


# ───────────────────── VWAP ─────────────────────

def calc_vwap(df: pd.DataFrame) -> float:
    """
    거래량 가중 평균 가격
    DataFrame 에 '고가', '저가', '종가', '거래량' 컬럼 필요
    """
    if df.empty:
        return 0.0
    v = df['거래량']
    tp = (df['고가'] + df['저가'] + df['종가']) / 3
    vol_sum = v.sum()
    if vol_sum == 0:
        return 0.0
    return (tp * v).sum() / vol_sum


# ───────────────────── OBV ─────────────────────

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — 스마트 머니 추적"""
    change = np.sign(close.diff()).fillna(0)
    return (change * volume).cumsum()


# ───────────────────── Bollinger Bands ─────────────────────

def calc_bollinger(close: pd.Series, window: int = 20,
                   n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    볼린저 밴드 → (upper, middle, lower)
    """
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + n_std * std
    lower = middle - n_std * std
    return upper, middle, lower


# ───────────────────── 캔들 패턴 ─────────────────────

def check_candle_pattern(o: pd.Series, h: pd.Series,
                         l: pd.Series, c: pd.Series) -> List[str]:
    """
    최근 캔들 패턴 감지 (망치형, 상승 장악형)
    최소 2봉 필요
    """
    if len(c) < 2:
        return []

    patterns = []

    curr_o, curr_h, curr_l, curr_c = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    prev_o, prev_c = o.iloc[-2], c.iloc[-2]

    body = abs(curr_c - curr_o)
    upper_shadow = curr_h - max(curr_c, curr_o)
    lower_shadow = min(curr_c, curr_o) - curr_l

    # 망치형 (Hammer)
    if (lower_shadow >= body * 2) and (upper_shadow <= body * 0.5) and (body > 0):
        patterns.append("망치형")

    # 상승 장악형 (Bullish Engulfing)
    is_prev_red = prev_c < prev_o
    is_curr_green = curr_c > curr_o
    if is_prev_red and is_curr_green:
        if (curr_o <= prev_c) and (curr_c >= prev_o):
            patterns.append("장악형")

    return patterns


# ───────────────────── 업종 모멘텀 ─────────────────────

def add_sector_momentum(df: pd.DataFrame,
                        group_col: str = "업종_대분류"
                        ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    업종별 중앙값 대비 상대 강도를 계산하여 'SECTOR_MOM' 컬럼 추가.
    Returns: (df_with_column, sector_medians)
    """
    if group_col not in df.columns or "ret_5d_%" not in df.columns:
        df["SECTOR_MOM"] = 0.0
        return df, pd.Series(dtype=float)

    medians = df.groupby(group_col)["ret_5d_%"].median()
    df["SECTOR_MOM"] = df[group_col].map(medians).fillna(0)
    return df, medians
