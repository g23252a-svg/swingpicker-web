# -*- coding: utf-8 -*-
"""Strategy C: Mean Revert — 과매도 기술반등 (제한적)"""
import pandas as pd
from strategies.base import SwingStrategy, StrategyFactory


@StrategyFactory.register("mean_revert")
class MeanRevertStrategy(SwingStrategy):
    horizon_days = 5
    tp_atr_mult = 1.5
    sl_atr_mult = 2.0

    DEFAULTS = {
        "max_breadth": 35.0,
        "base_weight": 0.7,
        "rsi_threshold": 35.0,
        "min_drop_5d": -5.0,
    }

    def is_suitable(self, macro_risk: str, breadth: float) -> bool:
        return macro_risk == "NORMAL" and breadth < self._p("max_breadth")

    def confidence(self, macro_risk: str, breadth: float) -> float:
        base_w = self._p("base_weight")
        threshold = self._p("max_breadth")
        scale = min(1.0, (threshold - breadth) / 20 + 0.5)
        return round(base_w * scale, 3)

    def _filter_rules(self, df: pd.DataFrame):
        # [v4.0] scoring-overhaul: 떨어지는 칼날 방지
        # '싸졌다'만으로 진입 금지 → 반전 확인 신호 필수
        rsi = self._safe_col(df, "RSI14", 50)
        rsi_prev = self._safe_col(df, "RSI14_PREV", 50)

        return [
            # 기본 조건: 과매도 + 하락 충분
            rsi < self._p("rsi_threshold"),
            self._safe_col(df, "Vol_Quality", 1.0) >= 1.2,
            self._safe_col(df, "ret_5d_%") < self._p("min_drop_5d"),
            self._safe_col(df, "Above_MA20") == 0,
            # [v4.0] 반전 확인 게이트 (하나 이상 충족 필수)
            # ① RSI 반등 시작 (어제보다 오늘 RSI 높음)
            # ② MACD 기울기 양전환
            # ③ 양봉 (종가 > 시가)
            (rsi > rsi_prev) |                                          # RSI 반등
            (self._safe_col(df, "MACD_Slope_PCT", 0) > 0) |           # MACD 양전환
            (self._safe_col(df, "ret_1d_%", 0) > 0),                  # 당일 양봉
        ]

    def _score_weights(self):
        return {"STRUCT_SCORE": 0.40, "TIMING_SCORE": 0.40}

    def _score_bonus(self, df: pd.DataFrame):
        rsi = self._safe_col(df, "RSI14", 50)
        return ((self._p("rsi_threshold") - rsi.clip(20, self._p("rsi_threshold"))) * 2 * 0.20).round(1)
