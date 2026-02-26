# data/services/market_service.py — 시장 분석 비즈니스 로직
"""
UI 의존성 0. pandas + 순수 계산만.

사용법:
    from data.services.market_service import MarketService
    svc = MarketService()
    summary = svc.get_market_summary(df)
"""
import pandas as pd
from typing import Dict, Any, Tuple


class MarketService:
    """시장 전체 상태 분석 — scored DataFrame 기반"""

    def get_fear_greed(self, df: pd.DataFrame) -> Tuple[float, str]:
        """공포/탐욕 지수 산출 → (점수 0~100, 라벨)"""
        if df.empty or "DISPLAY_SCORE" not in df.columns:
            return 50.0, "데이터 부족"

        avg = df["DISPLAY_SCORE"].mean()
        score = min(max(avg, 0), 100)

        if score >= 80:
            label = "극단적 탐욕"
        elif score >= 60:
            label = "탐욕"
        elif score >= 40:
            label = "중립"
        elif score >= 20:
            label = "공포"
        else:
            label = "극단적 공포"

        return round(score, 1), label

    def get_market_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """scored DataFrame → 시장 요약 지표 dict"""
        fg_score, fg_label = self.get_fear_greed(df)

        total = len(df)
        armed = 0
        attack = 0
        if "ROUTE" in df.columns:
            route_str = df["ROUTE"].astype(str)
            armed = int(route_str.str.contains("ARMED", na=False).sum())
            attack = int(route_str.str.contains("ATTACK", na=False).sum())

        avg_ret = 0.0
        if "ret_1d_%" in df.columns and not df.head(20).empty:
            avg_ret = round(df.head(20)["ret_1d_%"].mean(), 2)

        return {
            "fg_score": fg_score,
            "fg_label": fg_label,
            "total_count": total,
            "armed_count": armed,
            "attack_count": attack,
            "active_count": armed + attack,
            "avg_ret_top20": avg_ret,
        }

    def get_sector_breakdown(self, df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """상위 종목의 섹터별 분포 → 트리맵/바 차트용 DataFrame"""
        if df.empty or "업종" not in df.columns:
            return pd.DataFrame()

        sub = df.head(top_n)
        sector_df = (
            sub.groupby("업종")
            .agg(
                count=("업종", "size"),
                avg_score=("DISPLAY_SCORE", "mean"),
            )
            .sort_values("count", ascending=False)
            .reset_index()
        )
        return sector_df
