# -*- coding: utf-8 -*-
"""
price_utils.py

KRX(유가/코스닥/코넥스) 호가가격단위(틱) 기반 가격 라운딩/표시 유틸.

참고: 호가가격단위(7단계)
 - 2,000원 미만: 1원
 - 2,000~5,000원 미만: 5원
 - 5,000~20,000원 미만: 10원
 - 20,000~50,000원 미만: 50원
 - 50,000~200,000원 미만: 100원
 - 200,000~500,000원 미만: 500원
 - 500,000원 이상: 1,000원

※ ETF/ETN/ELW 등은 상품별 호가단위가 다를 수 있어
  현재는 “일반 주식” 기준으로만 적용합니다.
"""

from __future__ import annotations  # ✅ 반드시 여기(인코딩/모듈독스트링 다음, 다른 import보다 위)

import math
from typing import Optional, Union, Any  # ✅ Any 추가
import numpy as np

Number = Union[int, float]


def is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return True


def krx_tick_size(price: float) -> int:
    if price < 2000:
        return 1
    if price < 5000:
        return 5
    if price < 20000:
        return 10
    if price < 50000:
        return 50
    if price < 200000:
        return 100
    if price < 500000:
        return 500
    return 1000


def _round_half_up(x: float) -> int:
    """0.5 올림(은행가 반올림 방지)."""
    return int(math.floor(x + 0.5))


def round_to_tick(price, method: str = "nearest"):
    try:
        if price is None:
            return None
        if isinstance(price, float) and (math.isnan(price) or np.isinf(price)):
            return None

        p = float(price)
        t = krx_tick_size(p)
        if t <= 0:
            return int(p)

        q = p / t

        if method == "down":
            return int(math.floor(q) * t)
        if method == "up":
            return int(math.ceil(q) * t)

        # nearest (0.5는 올림)
        return int(_round_half_up(q) * t)

    except Exception:
        return None


def add_tick(price: Number, ticks: int = 1) -> int:
    """
    현재 가격에서 n호가 위/아래 가격을 계산 (단순 근사치)
    ※ 경계선 부근에서는 호가 단위가 변하므로 완벽하지 않을 수 있으나,
       일반적인 목표가/손절가 계산에는 충분함.
    """
    if is_nan(price): return 0
    p = float(price)
    t = krx_tick_size(p)
    
    # 틱을 더한 값 계산
    target = p + (t * ticks)
    
    # 계산된 가격 기준으로 다시 라운딩하여 유효 가격으로 보정
    return round_to_tick(target, method="nearest") or int(target)

def calc_return(current: Number, base: Number) -> float:
    """수익률 안전 계산 (0으로 나누기 방지)"""
    if is_nan(current) or is_nan(base) or base == 0:
        return 0.0
    return ((float(current) - float(base)) / float(base)) * 100.0

def format_krw(x) -> str:
    try:
        if x is None:
            return "-"
        v = float(x)
        if math.isnan(v) or np.isinf(v):
            return "-"
        return f"{int(v):,}원"
    except Exception:
        return "-"


def format_volume(x: Optional[Number], unit: str = "억") -> str:
    """
    거래대금 등을 '억' 단위로 변환하여 표시
    입력값이 '원' 단위라고 가정 (예: 100000000 -> 1억)
    """
    if is_nan(x): return "-"
    try:
        val = float(x)
        if unit == "억":
            # 1억 미만은 그대로 표시
            if val < 100000000:
                return f"{int(val):,}"
            return f"{int(val / 100000000):,}억"
        return f"{int(val):,}"
    except: return "-"

def format_signed_krw(value: Optional[Number], suffix: str = "원") -> str:
    if is_nan(value):
        return "-"
    try:
        iv = int(_round_half_up(float(value)))
    except Exception:
        return str(value)
    sign = "+" if iv > 0 else ""
    return f"{sign}{iv:,}{suffix}"


def format_pct(value: Optional[Number], digits: int = 2) -> str:
    if is_nan(value):
        return "-"
    try:
        fv = float(value)
    except Exception:
        return str(value)
    sign = "+" if fv > 0 else ""
    return f"{sign}{fv:.{digits}f}%"
