# -*- coding: utf-8 -*-
"""Strategy B: Pullback Swing — 눌림목 반등"""
import pandas as pd
from strategies.base import SwingStrategy, StrategyFactory


@StrategyFactory.register("pullback")
class PullbackStrategy(SwingStrategy):
    horizon_days = 7
    tp_atr_mult = 2.0
    sl_atr_mult = 1.5

    DEFAULTS = {
        "min_breadth": 40.0,
        "base_weight": 1.0,
        "sweet_spot_breadth": 55.0,
    }

    def is_suitable(self, macro_risk: str, breadth: float) -> bool:
        return macro_risk != "CRITICAL" and breadth >= self._p("min_breadth")

    def confidence(self, macro_risk: str, breadth: float) -> float:
        base_w = self._p("base_weight")
        distance = abs(breadth - self._p("sweet_spot_breadth"))
        scale = max(0.5, 1.0 - distance / 50)
        if macro_risk == "HIGH":
            scale *= 0.8
        return round(base_w * scale, 3)

    def _filter_rules(self, df: pd.DataFrame):
        return [
            self._safe_col(df, "Low_Trend_PCT") > 0,
            self._safe_col(df, "이격도").between(-3, 3),
            self._safe_col(df, "RSI14", 50).between(40, 65),
            self._safe_col(df, "MTF_WEEKLY_TREND") >= 0,
        ]

    def _score_weights(self):
        return {"STRUCT_SCORE": 0.50, "TIMING_SCORE": 0.35, "AI_SCORE": 0.15}

    def _score_bonus(self, df: pd.DataFrame):
        disp = self._safe_col(df, "이격도").abs()
        return (disp < 1.5).astype(int) * 5
