# -*- coding: utf-8 -*-
"""
strategies/ — Swing Mode 전략 팩토리
═══════════════════════════════════════
[Phase 2-1] 단일 점수 대신, 시장 상황별로 다른 전략 적용.

Usage:
    from strategies import StrategyFactory, select_strategies
    active = select_strategies(macro_risk, breadth_all)
    for name in active:
        strat = StrategyFactory.create(name, config)
        filtered = strat.filter(df)
        scored = strat.score(filtered)
        picks = strat.rank_and_pick(scored, top_k=5)
"""

from strategies.base import SwingStrategy
from strategies.breakout import BreakoutStrategy
from strategies.pullback import PullbackStrategy
from strategies.mean_revert import MeanRevertStrategy


class StrategyFactory:
    _registry = {
        "breakout": BreakoutStrategy,
        "pullback": PullbackStrategy,
        "mean_revert": MeanRevertStrategy,
    }

    @classmethod
    def create(cls, name: str, config=None) -> SwingStrategy:
        klass = cls._registry.get(name)
        if klass is None:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._registry.keys())}")
        return klass(config=config)

    @classmethod
    def available(cls):
        return list(cls._registry.keys())


def select_strategies(
    macro_risk: str = "NORMAL",
    breadth_all: float = 50.0,
) -> list:
    """시장 상황에 따라 활성화할 전략 결정."""
    if macro_risk == "CRITICAL":
        return []

    active = []
    if breadth_all >= 55:
        active.append("breakout")
    if breadth_all >= 40:
        active.append("pullback")
    if breadth_all < 35 and macro_risk == "NORMAL":
        active.append("mean_revert")

    if not active and macro_risk != "CRITICAL":
        active.append("pullback")

    return active
