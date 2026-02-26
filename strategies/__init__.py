# -*- coding: utf-8 -*-
"""
strategies/ — Swing Mode 전략 팩토리
═══════════════════════════════════════
[v3.4] 리팩터링:
  - StrategyConfig 삭제 → 각 전략 내부 DEFAULTS
  - CollectorConfig 종속성 제거 → **overrides만 수용
  - 템플릿 메서드 → _filter_rules / _score_weights 선언적 정의

Usage:
    from strategies import StrategyFactory

    # 가중치 기반 선택 (config 불필요)
    candidates = StrategyFactory.select("NORMAL", 65.0)

    # 그리드 서치: 특정 전략 파라미터만 오버라이드
    strat = StrategyFactory.create("breakout", min_breadth=60, vol_quality_min=1.5)

    # 기존 호환
    active = select_strategies("NORMAL", 65.0)
"""

from strategies.base import SwingStrategy, StrategyFactory

# ── 전략 import (= 자동 등록 트리거) ──
from strategies.breakout import BreakoutStrategy       # noqa: F401
from strategies.pullback import PullbackStrategy       # noqa: F401
from strategies.mean_revert import MeanRevertStrategy   # noqa: F401


def select_strategies(
    macro_risk: str = "NORMAL",
    breadth_all: float = 50.0,
    **overrides,
) -> list:
    """기존 호환 래퍼 — 이름 리스트 반환."""
    candidates = StrategyFactory.select(macro_risk, breadth_all, **overrides)
    return [name for name, _weight in candidates]
