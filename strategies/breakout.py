# -*- coding: utf-8 -*-
"""
Strategy A: Breakout Swing — 수축→확장 돌파
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
진입 조건: TTM Squeeze + 거래량 확인 + POC 상단 + MA20 상방
청산: ATR 2.5배 TP, 1.2배 SL, 5일 Time Stop
적합 시장: breadth >= 55 (활황)
"""

import pandas as pd
import numpy as np
from strategies.base import SwingStrategy


class BreakoutStrategy(SwingStrategy):
    name = "breakout"
    horizon_days = 5
    tp_atr_mult = 2.5
    sl_atr_mult = 1.2

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        ttm = self._safe_col(df, "TTM_SQUEEZE")
        vol_q = self._safe_col(df, "Vol_Quality", 1.0)
        above_poc = self._safe_col(df, "IS_ABOVE_POC")
        above_ma20 = self._safe_col(df, "Above_MA20")
        macd_slope = self._safe_col(df, "MACD_Slope_PCT")
        bb_exp = self._safe_col(df, "BB_Expanding")

        mask = (
            ((ttm == 1) | (bb_exp == 1)) &  # 스퀴즈 또는 BB 확장
            (vol_q >= 1.3) &                  # 거래량 품질
            (above_poc == 1) &                # POC 상단
            (above_ma20 == 1) &               # MA20 위
            (macd_slope > 0)                  # MACD 상승
        )
        return df[mask].copy()

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        timing = self._safe_col(df, "TIMING_SCORE")
        struct = self._safe_col(df, "STRUCT_SCORE")
        ai = self._safe_col(df, "AI_SCORE")

        # Breakout: TIMING 비중 높임
        df["STRATEGY_SCORE"] = (timing * 0.60 + struct * 0.30 + ai * 0.10).round(1)

        # 추가 보너스
        ttm = self._safe_col(df, "TTM_SQUEEZE")
        vol_q = self._safe_col(df, "Vol_Quality", 1.0)
        bonus = (ttm * 5) + ((vol_q >= 2.0).astype(int) * 3)
        df["STRATEGY_SCORE"] = (df["STRATEGY_SCORE"] + bonus).clip(0, 100)

        return df
