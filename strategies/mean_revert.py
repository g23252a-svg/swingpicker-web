# -*- coding: utf-8 -*-
"""
Strategy C: Mean Revert — 과매도 기술반등 (제한적 사용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
진입 조건: RSI 과매도 + 거래량 확인 + 하락 과도
청산: ATR 1.5배 TP (보수적), 2.0배 SL, 5일 Time Stop
적합 시장: breadth < 35 & macro NORMAL (침체기)
"""

import pandas as pd
import numpy as np
from strategies.base import SwingStrategy


class MeanRevertStrategy(SwingStrategy):
    name = "mean_revert"
    horizon_days = 5
    tp_atr_mult = 1.5
    sl_atr_mult = 2.0

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        rsi = self._safe_col(df, "RSI14", 50)
        vol_q = self._safe_col(df, "Vol_Quality", 1.0)
        ret_5d = self._safe_col(df, "ret_5d_%")
        above_ma20 = self._safe_col(df, "Above_MA20")

        mask = (
            (rsi < 35) &              # 과매도
            (vol_q >= 1.2) &           # 바닥 매집 조짐
            (ret_5d < -5) &            # 충분한 하락
            (above_ma20 == 0)          # MA20 아래 (이탈 후 반등)
        )
        return df[mask].copy()

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        struct = self._safe_col(df, "STRUCT_SCORE")
        timing = self._safe_col(df, "TIMING_SCORE")
        rsi = self._safe_col(df, "RSI14", 50)

        # Mean Revert: RSI 역전 보너스
        rsi_bonus = ((35 - rsi.clip(20, 35)) * 2).round(1)  # RSI 20→30, 35→0
        df["STRATEGY_SCORE"] = (struct * 0.40 + timing * 0.40 + rsi_bonus * 0.20).round(1)
        df["STRATEGY_SCORE"] = df["STRATEGY_SCORE"].clip(0, 100)

        return df
