# -*- coding: utf-8 -*-
"""SwingStrategy ABC — 모든 전략의 기본 인터페이스"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import pandas as pd
import numpy as np


class SwingStrategy(ABC):
    """스윙 전략 추상 클래스.

    모든 전략은 filter → score → rank_and_pick 3단계를 구현.
    """

    name: str = "base"
    horizon_days: int = 7
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.5

    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """전략 조건 충족 종목만 필터"""
        ...

    @abstractmethod
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """전략 전용 점수 산출 → STRATEGY_SCORE 컬럼 추가"""
        ...

    def rank_and_pick(self, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """점수 상위 K개 선택 + 메타 추가"""
        if df.empty:
            return df
        score_col = "STRATEGY_SCORE" if "STRATEGY_SCORE" in df.columns else "FINAL_SCORE"
        result = df.nlargest(top_k, score_col).copy()
        result["STRATEGY"] = self.name
        result["STRATEGY_HORIZON"] = self.horizon_days
        return result

    def _safe_col(self, df: pd.DataFrame, col: str, default=0) -> pd.Series:
        """컬럼이 없으면 기본값 시리즈 반환"""
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index)
