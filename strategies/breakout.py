# -*- coding: utf-8 -*-
"""Strategy A: Breakout Swing — 수축→확장 돌파"""
import pandas as pd
from strategies.base import SwingStrategy, StrategyFactory


@StrategyFactory.register("breakout")
class BreakoutStrategy(SwingStrategy):
    horizon_days = 5
    tp_atr_mult = 2.5
    sl_atr_mult = 1.2

    # 하이퍼파라미터 — 이 전략이 소유 (collector_config에 없음)
    DEFAULTS = {
        "min_breadth": 55.0,
        "base_weight": 1.0,
        "vol_quality_min": 1.3,
    }

    def is_suitable(self, macro_risk: str, breadth: float) -> bool:
        return macro_risk != "CRITICAL" and breadth >= self._p("min_breadth")

    def confidence(self, macro_risk: str, breadth: float) -> float:
        base_w = self._p("base_weight")
        threshold = self._p("min_breadth")
        scale = min(1.0, max(0.5, (breadth - threshold) / 30 + 0.7))
        if macro_risk == "HIGH":
            scale *= 0.7
        return round(base_w * scale, 3)

    def _filter_rules(self, df: pd.DataFrame):
        ttm = self._safe_col(df, "TTM_SQUEEZE")
        bb_exp = self._safe_col(df, "BB_Expanding")
        return [
            (ttm == 1) | (bb_exp == 1),
            self._safe_col(df, "Vol_Quality", 1.0) >= self._p("vol_quality_min"),
            self._safe_col(df, "IS_ABOVE_POC") == 1,
            self._safe_col(df, "Above_MA20") == 1,
            self._safe_col(df, "MACD_Slope_PCT") > 0,
        ]

    def _score_weights(self):
        return {"TIMING_SCORE": 0.60, "STRUCT_SCORE": 0.30, "AI_SCORE": 0.10}

    def _score_bonus(self, df: pd.DataFrame):
        ttm = self._safe_col(df, "TTM_SQUEEZE")
        vol_q = self._safe_col(df, "Vol_Quality", 1.0)
        return (ttm * 5) + ((vol_q >= 2.0).astype(int) * 3)
