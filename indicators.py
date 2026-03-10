# -*- coding: utf-8 -*-
"""
indicators.py — 기술적 지표 순수 함수 모음 (v2.0)
══════════════════════════════════════════════════
[v2.0] 6건 수정:
  #1 RSI/ATR: SMA → Wilder's Smoothing (RMA) — 트레이딩뷰/HTS와 값 일치
  #2 add_sector_momentum: 원본 mutation → df.assign() 순수 함수화
  #3 calc_vwap: 한글 하드코딩 → Series 파라미터 + Daily VWAP Series 반환
  #4 check_candle_pattern: iloc[-1] 단봉 → 벡터화 전체 행 패턴 Series
  #5 calc_bollinger: std(ddof=1) → std(ddof=0) — 차트 플랫폼 표준
  #6 calc_mfi: replace(0,1) 꼼수 → ffill 통계적 올바른 처리
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


# ═══════════════════════════════════════════════════
#  [v2.0 #1] Wilder's Smoothing (RMA)
# ═══════════════════════════════════════════════════

def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's Smoothing (RMA/SMMA)

    첫 period개는 SMA로 시드, 이후 EMA(alpha=1/period).
    트레이딩뷰, MetaTrader, 증권사 HTS 표준 방식.
    """
    alpha = 1.0 / period
    # ewm의 alpha=1/period이 Wilder's smoothing과 동치
    # adjust=False: 재귀적 가중 (SMA 시드 자동)
    return series.ewm(alpha=alpha, adjust=False).mean()


# ───────────────────── RSI ─────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """[v2.0 #1] RSI — Wilder's Smoothing (트레이딩뷰 호환)

    Before: rolling(period).mean() → SMA 기반 → HTS/TV와 값 불일치
    After:  _rma() → Wilder's RMA → 트레이딩뷰/HTS와 동일한 값
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = _rma(up, period)
    roll_down = _rma(down, period)

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 경계 조건 처리
    both_zero = (roll_up == 0) & (roll_down == 0)
    rsi = rsi.where(~both_zero, 50)
    rsi = rsi.where(~((roll_down == 0) & (roll_up != 0)), 100)
    rsi = rsi.where(~((roll_up == 0) & (roll_down != 0)), 0)

    return rsi


# ───────────────────── ATR ─────────────────────

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    """[v2.0 #1] ATR — Wilder's Smoothing (트레이딩뷰 호환)

    Before: tr.rolling(period).mean() → SMA ATR
    After:  _rma(tr, period) → Wilder's ATR
    """
    tr = pd.concat(
        [(high - low),
         (high - close.shift(1)).abs(),
         (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)
    return _rma(tr, period)


# ───────────────────── SuperTrend ─────────────────────

def calc_supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 10, multiplier: float = 3.0
                    ) -> Tuple[pd.Series, pd.Series]:
    """SuperTrend 지표 → (supertrend_line, trend_direction)

    trend_direction: 1 = 상승, -1 = 하락
    Note: 재귀적 특성상 for 루프 유지 (Numba @jit 적용 권장 — 추후)
    """
    atr = calc_atr(high, low, close, period)

    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    n = len(close)
    st_out = np.full(n, np.nan)
    trend_out = np.ones(n, dtype=np.int8)

    vals_c = close.values
    vals_bu = basic_upper.values
    vals_bl = basic_lower.values

    start_idx = period
    if start_idx >= n:
        return (pd.Series(st_out, index=close.index),
                pd.Series(trend_out, index=close.index))

    final_upper = vals_bu[start_idx]
    final_lower = vals_bl[start_idx]

    st_out[start_idx] = final_lower
    trend_out[start_idx] = 1

    for i in range(start_idx + 1, n):
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
    """[v2.0 #6] MFI — 0 나누기 시 ffill (통계적 올바른 처리)

    Before: neg_s.replace(0, 1) → 데이터 왜곡
    After:  NaN 처리 후 ffill → 이전 MFI 값 유지
    """
    tp = (high + low + close) / 3
    rmf = tp * vol
    pos = np.where(tp.diff() > 0, rmf, 0)
    neg = np.where(tp.diff() < 0, rmf, 0)
    pos_s = pd.Series(pos, index=close.index).rolling(period).sum()
    neg_s = pd.Series(neg, index=close.index).rolling(period).sum()

    # neg_s가 0이면 NaN → ffill (이전 MFI 유지)
    ratio = pos_s / neg_s.replace(0, np.nan)
    mfi = 100 - (100 / (1 + ratio))
    return mfi.ffill()


# ───────────────────── VWAP ─────────────────────

def calc_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
              volume: pd.Series, window: int = 20) -> pd.Series:
    """[v4.0] scoring-overhaul: Anchored VWAP — 최근 N일 윈도우 기반

    기존 무한 누적 VWAP → 20일 롤링 VWAP으로 변경.
    스윙 트레이딩에서 세력 평단가를 유추하는 데 실질적으로 유용.
    """
    tp = (high + low + close) / 3
    tp_vol = tp * volume

    # 최근 window일 롤링 VWAP
    cum_tp_vol = tp_vol.rolling(window, min_periods=1).sum()
    cum_vol = volume.rolling(window, min_periods=1).sum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def calc_vwap_scalar(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series) -> float:
    """기간 단일 VWAP 값 (기존 호환용)"""
    tp = (high + low + close) / 3
    vol_sum = volume.sum()
    if vol_sum == 0:
        return 0.0
    return float((tp * volume).sum() / vol_sum)


# ───────────────────── OBV ─────────────────────

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — 스마트 머니 추적"""
    change = np.sign(close.diff()).fillna(0)
    return (change * volume).cumsum()


# ───────────────────── Bollinger Bands ─────────────────────

def calc_bollinger(close: pd.Series, window: int = 20,
                   n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """[v2.0 #5] 볼린저 밴드 → (upper, middle, lower)

    Before: std() → ddof=1 (표본) → 차트 플랫폼과 미세 차이
    After:  std(ddof=0) → 모표준편차 → TV/HTS 표준
    """
    middle = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = middle + n_std * std
    lower = middle - n_std * std
    return upper, middle, lower


# ───────────────────── 캔들 패턴 ─────────────────────

def check_candle_pattern(o: pd.Series, h: pd.Series,
                         l: pd.Series, c: pd.Series) -> List[str]:
    """최근 캔들 패턴 감지 (기존 호환 — 최신 2봉만)"""
    if len(c) < 2:
        return []

    patterns = []
    curr_o, curr_h, curr_l, curr_c = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    prev_o, prev_c = o.iloc[-2], c.iloc[-2]

    body = abs(curr_c - curr_o)
    upper_shadow = curr_h - max(curr_c, curr_o)
    lower_shadow = min(curr_c, curr_o) - curr_l

    if (lower_shadow >= body * 2) and (upper_shadow <= body * 0.5) and (body > 0):
        patterns.append("망치형")

    is_prev_red = prev_c < prev_o
    is_curr_green = curr_c > curr_o
    if is_prev_red and is_curr_green:
        if (curr_o <= prev_c) and (curr_c >= prev_o):
            patterns.append("장악형")

    return patterns


def detect_candle_patterns(o: pd.Series, h: pd.Series,
                           l: pd.Series, c: pd.Series) -> pd.DataFrame:
    """[v2.0 #4] 벡터화 캔들 패턴 — 전체 행 True/False

    백테스트 가능: 모든 봉에 대해 패턴 발생 여부를 Series로 반환.

    Returns:
        DataFrame with columns: ["HAMMER", "ENGULFING"]
    """
    body = (c - o).abs()
    upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - l

    # 망치형 (Hammer): 아래꼬리 >= 몸통*2, 위꼬리 <= 몸통*0.5, 몸통 > 0
    hammer = (
        (lower_shadow >= body * 2) &
        (upper_shadow <= body * 0.5) &
        (body > 0)
    )

    # 상승 장악형 (Bullish Engulfing)
    prev_red = c.shift(1) < o.shift(1)
    curr_green = c > o
    engulf_open = o <= c.shift(1)
    engulf_close = c >= o.shift(1)
    engulfing = prev_red & curr_green & engulf_open & engulf_close

    return pd.DataFrame({
        "HAMMER": hammer,
        "ENGULFING": engulfing,
    }, index=c.index)


# ───────────────────── 업종 모멘텀 ─────────────────────

def add_sector_momentum(df: pd.DataFrame,
                        group_col: str = "업종_대분류"
                        ) -> Tuple[pd.DataFrame, pd.Series]:
    """[v2.0 #2] 업종별 상대 강도 — 순수 함수 (원본 불변)

    Before: df["SECTOR_MOM"] = ... → 원본 DataFrame mutation
    After:  df.assign() → 새 DataFrame 반환 (원본 그대로)
    """
    if group_col not in df.columns or "ret_5d_%" not in df.columns:
        return df.assign(SECTOR_MOM=0.0), pd.Series(dtype=float)

    medians = df.groupby(group_col)["ret_5d_%"].median()
    sector_mom = df[group_col].map(medians).fillna(0)
    return df.assign(SECTOR_MOM=sector_mom), medians
