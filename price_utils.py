# -*- coding: utf-8 -*-
"""
KRX 주식 가격 및 호가단위 정밀 연산 유틸리티
- v2.2: 무결점 라운딩 및 경계선 틱 연산 보정 완료
"""

import math
from typing import Optional, Union
import numpy as np

Number = Union[int, float]

# ----------------- 1. 호가 단위(Tick Size) 정의 -----------------

def krx_tick_size(price: float) -> int:
    """KRX 표준 호가단위 (2023년 개정 통합 기준)"""
    if price < 2000: return 1
    if price < 5000: return 5
    if price < 20000: return 10
    if price < 50000: return 50
    if price < 200000: return 100
    if price < 500000: return 500
    return 1000

def vec_krx_tick_size(prices: np.ndarray) -> np.ndarray:
    """Numpy 배열을 한 번에 처리하는 고속 벡터 연산 (Batch용)"""
    conditions = [
        (prices < 2000), (prices < 5000), (prices < 20000),
        (prices < 50000), (prices < 200000), (prices < 500000)
    ]
    choices = [1, 5, 10, 50, 100, 500]
    return np.select(conditions, choices, default=1000)

# ----------------- 2. 정밀 가격 연산 (Sovereign Logic) -----------------

def round_to_tick(price: Number, method: str = "nearest") -> Optional[int]:
    """[❗100점 패치] 메서드별 정수화 방식 분리형 무결점 라운딩"""
    if price is None or (isinstance(price, float) and (math.isnan(price) or math.isinf(price))):
        return None

    p_float = float(price)

    # 메서드별 정수화 분리 (내림/올림/반올림 정합성 확보)
    if method == "down":
        p = int(math.floor(p_float))
    elif method == "up":
        p = int(math.ceil(p_float))
    else:  # nearest
        p = int(round(p_float))

    t = krx_tick_size(p)
    remainder = p % t
    
    if remainder == 0: return p

    if method == "down":
        return p - remainder
    elif method == "up":
        return p + (t - remainder)
    else: # nearest
        return p + (t - remainder) if remainder >= (t / 2) else p - remainder

def add_tick(price: Number, ticks: int = 1) -> int:
    """[❗v2.3 패치] 경계선 돌파를 인지하는 정밀 틱 이동 연산"""
    if price is None or (isinstance(price, float) and math.isnan(price)): return 0
    
    # 시작가를 먼저 가장 가까운 틱 그리드에 정렬
    curr = float(round_to_tick(price, "nearest"))
    direction = 1 if ticks > 0 else -1
    
    for _ in range(abs(ticks)):
        # 하향 이동 시: 경계값(2000, 5000 등)에서 아래 틱존을 참조하기 위해
        # -1 로 탐색 (기존 -0.1 은 tick=1 구간에서 부정확할 수 있음)
        lookup_p = curr if direction > 0 else curr - 1
        t = krx_tick_size(max(1, lookup_p))
        curr += t * direction
        
    return int(curr)

def calc_return(current: Number, base: Number) -> float:
    """수익률 안전 계산 (ZeroDivision 방어)"""
    if not current or not base or base == 0: return 0.0
    return ((float(current) - float(base)) / float(base)) * 100.0

# ----------------- 3. 시각적 포맷팅 (UI 최적화) -----------------

def format_krw(x: Optional[Number]) -> str:
    """천단위 콤마가 포함된 금액 표시 (1,500원)"""
    if x is None or (isinstance(x, float) and math.isnan(x)): return "-"
    return f"{int(x):,}원"

def format_volume(x: Optional[Number], input_unit: str = "won") -> str:
    """거래대금/거래량 스케일 명시형 포맷터 (1.3억 등)"""
    if x is None or (isinstance(x, float) and math.isnan(x)): return "-"
    
    val_won = float(x) * 100_000_000 if input_unit == "100m" else float(x)
    
    if val_won >= 100_000_000:
        return f"{val_won / 100_000_000:.1f}억"
    elif val_won >= 10_000:
        return f"{int(val_won / 10_000):,}만"
    return f"{int(val_won):,}"

def format_pct(value: Optional[Number], digits: int = 2) -> str:
    """부호가 포함된 수익률 표시 (+2.50%)"""
    if value is None or (isinstance(value, float) and math.isnan(value)): return "-"
    fv = float(value)
    sign = "+" if fv > 0 else ""
    return f"{sign}{fv:.{digits}f}%"
