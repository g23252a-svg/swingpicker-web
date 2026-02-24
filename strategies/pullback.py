# -*- coding: utf-8 -*-
"""
Strategy B: Pullback Swing — 눌림목 반등
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
진입 조건: 상승추세 + MA20 근처 눌림 + RSI 과열 아님
청산: ATR 2.0배 TP, 1.5배 SL, 7일 Time Stop
적합 시장: breadth >= 40 (보통 이상)
"""

import pandas as pd
import numpy as np
from strategies.base import SwingStrategy


class PullbackStrategy(SwingStrategy):
    name = "pullback"
    horizon_days = 7
    tp_atr_mult = 2.0
    sl_atr_mult = 1.5

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        low_trend = self._safe_col(df, "Low_Trend_PCT")
        disp = self._safe_col(df, "이격도")
        rsi = self._safe_col(df, "RSI14", 50)
        mtf_weekly = self._safe_col(df, "MTF_WEEKLY_TREND")

        mask = (
            (low_trend > 0) &                        # 저점 상승 (상승추세)
            (disp.between(-3, 3)) &                   # MA20 근처
            (rsi.between(40, 65)) &                   # 과열 아님
            (mtf_weekly >= 0)                          # 주봉 하락 아님
        )
        return df[mask].copy()

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        struct = self._safe_col(df, "STRUCT_SCORE")
        timing = self._safe_col(df, "TIMING_SCORE")
        ai = self._safe_col(df, "AI_SCORE")

        # Pullback: STRUCT 비중 높임 (체력 중시)
        df["STRATEGY_SCORE"] = (struct * 0.50 + timing * 0.35 + ai * 0.15).round(1)

        # MA20 수렴 보너스
        disp = self._safe_col(df, "이격도").abs()
        ma_bonus = ((disp < 1.5).astype(int) * 5)
        df["STRATEGY_SCORE"] = (df["STRATEGY_SCORE"] + ma_bonus).clip(0, 100)

        return df
