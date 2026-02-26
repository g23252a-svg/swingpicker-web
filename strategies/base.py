# -*- coding: utf-8 -*-
"""
SwingStrategy ABC + 자동 등록 팩토리 + 템플릿 메서드
═══════════════════════════════════════════════════════
[v3.4] 3건 리팩터링:
  #1 StrategyConfig 삭제 → 각 전략 내부에 하이퍼파라미터 캡슐화
  #2 filter()/score() 보일러플레이트 → 템플릿 메서드(선언적 정의)
  #3 config 전체 전달 금지 → **overrides 딕셔너리로 최소 권한

설계:
  - 각 전략 클래스는 DEFAULTS (dict)로 자신의 하이퍼파라미터를 선언
  - 팩토리는 **overrides로 특정 값만 덮어씌움 (그리드 서치)
  - filter()는 _filter_rules() 선언적 리스트를 AND 결합
  - score()는 _score_weights() 선언적 딕셔너리로 가중합
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Callable
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════
#  자동 등록 팩토리
# ═══════════════════════════════════════════════════

class StrategyFactory:
    """전략 자동 등록 + 생성 + 선택 팩토리."""
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """클래스 데코레이터 — 전략 자동 등록."""
        def wrapper(subclass):
            if name in cls._registry:
                raise ValueError(f"전략 '{name}' 중복 등록")
            cls._registry[name] = subclass
            subclass.name = name
            return subclass
        return wrapper

    @classmethod
    def create(cls, name: str, **overrides) -> "SwingStrategy":
        """이름으로 전략 인스턴스 생성.
        
        Args:
            name: 등록된 전략 이름
            **overrides: 하이퍼파라미터 덮어쓰기 (그리드 서치용)
                         예: StrategyFactory.create("breakout", min_breadth=60)
        """
        klass = cls._registry.get(name)
        if klass is None:
            raise ValueError(f"Unknown strategy: '{name}'. Available: {list(cls._registry.keys())}")
        return klass(**overrides)

    @classmethod
    def available(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def select(
        cls,
        macro_risk: str = "NORMAL",
        breadth: float = 50.0,
        **overrides,
    ) -> List[Tuple[str, float]]:
        """가중치 기반 전략 선택 → [(이름, 비중)] 리스트.
        
        CollectorConfig를 받지 않음 — 최소 권한 원칙.
        overrides는 모든 전략에 일괄 전달됨 (해당 없는 키는 무시).
        """
        if macro_risk == "CRITICAL":
            return []

        candidates = []
        for name, klass in cls._registry.items():
            strat = klass(**overrides)
            if strat.is_suitable(macro_risk, breadth):
                weight = strat.confidence(macro_risk, breadth)
                candidates.append((name, round(weight, 3)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates


# ═══════════════════════════════════════════════════
#  SwingStrategy ABC — 템플릿 메서드 패턴
# ═══════════════════════════════════════════════════

class SwingStrategy(ABC):
    """스윙 전략 추상 클래스.
    
    서브클래스가 오버라이드해야 할 것:
      - DEFAULTS: dict — 하이퍼파라미터 기본값 (클래스 변수)
      - is_suitable(macro_risk, breadth) → bool
      - confidence(macro_risk, breadth) → float (0~1)
      - _filter_rules() → List[pd.Series(bool)] — AND 결합
      - _score_weights() → Dict[str, float] — {컬럼: 가중치}
      - _score_bonus(df) → pd.Series — 추가 보너스 (선택)
    """

    # 서브클래스가 오버라이드할 기본값
    name: str = "base"
    DEFAULTS: Dict[str, float] = {}

    # 공통 기본값
    horizon_days: int = 7
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.5

    def __init__(self, **overrides):
        """하이퍼파라미터 초기화: DEFAULTS + overrides.
        
        전략이 필요한 값만 자기 내부에 보유 (최소 권한).
        CollectorConfig를 받지 않음.
        """
        self.params = {**self.DEFAULTS}
        # overrides 중 이 전략의 DEFAULTS에 있는 키만 수용
        for k, v in overrides.items():
            if k in self.DEFAULTS:
                self.params[k] = v

    def _p(self, key: str) -> float:
        """하이퍼파라미터 접근 헬퍼."""
        return self.params[key]

    # ── 적합성 (서브클래스 오버라이드) ──

    @abstractmethod
    def is_suitable(self, macro_risk: str, breadth: float) -> bool:
        """이 시장에서 유효한가?"""
        ...

    @abstractmethod
    def confidence(self, macro_risk: str, breadth: float) -> float:
        """이 시장에서의 신뢰도 (0.0 ~ 1.0)."""
        ...

    # ── 템플릿 메서드: filter ──

    @abstractmethod
    def _filter_rules(self, df: pd.DataFrame) -> List[pd.Series]:
        """필터 조건 시리즈 리스트. AND로 결합됨.
        
        예: [rsi.between(40, 65), low_trend > 0, ...]
        """
        ...

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """공통 필터 파이프라인 — _filter_rules()를 AND 결합."""
        if df.empty:
            return df
        rules = self._filter_rules(df)
        if not rules:
            return df
        mask = rules[0]
        for rule in rules[1:]:
            mask = mask & rule
        return df[mask].copy()

    # ── 템플릿 메서드: score ──

    @abstractmethod
    def _score_weights(self) -> Dict[str, float]:
        """점수 가중치 딕셔너리: {컬럼명: 가중치}.
        
        예: {"TIMING_SCORE": 0.6, "STRUCT_SCORE": 0.3, "AI_SCORE": 0.1}
        """
        ...

    def _score_bonus(self, df: pd.DataFrame) -> pd.Series:
        """추가 보너스 시리즈 (선택 오버라이드). 기본: 0."""
        return pd.Series(0.0, index=df.index)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """공통 점수 파이프라인 — _score_weights() 가중합 + _score_bonus()."""
        if df.empty:
            return df
        df = df.copy()
        weights = self._score_weights()
        total = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            total += self._safe_col(df, col) * w
        total += self._score_bonus(df)
        df["STRATEGY_SCORE"] = total.round(1).clip(0, 100)
        return df

    # ── rank_and_pick ──

    def rank_and_pick(self, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        if df.empty:
            return df
        score_col = "STRATEGY_SCORE" if "STRATEGY_SCORE" in df.columns else "FINAL_SCORE"
        result = df.nlargest(top_k, score_col).copy()
        result["STRATEGY"] = self.name
        result["STRATEGY_HORIZON"] = self.horizon_days
        return result

    # ── 유틸 ──

    def _safe_col(self, df: pd.DataFrame, col: str, default=0) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index)
