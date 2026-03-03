# -*- coding: utf-8 -*-
"""
viz_components.py — 📊 고급 시각화 컴포넌트 모듈
═══════════════════════════════════════════════════
Tab 7(성과), Tab 9(매매 일지), Tab 10(전략 샌드박스)에서 사용할
인터랙티브 Plotly 차트 라이브러리

[구성]
 1. 누적 수익률 곡선 (Equity Curve with Benchmark)
 2. 월별/요일별 수익금 히트맵
 3. 캘린더 형태 손익 분포도
 4. 점수 구간별 승률 바 차트
 5. 일별 거래 건수 + 수익률 이중축
 6. 섹터별 수익 기여도 트리맵

통합 방법:
  from viz_components import (
      plot_equity_curve, plot_monthly_heatmap, plot_calendar_pnl,
      plot_score_winrate, plot_daily_dual_axis, plot_sector_treemap
  )
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
#  공통 테마 & 유틸
# ═══════════════════════════════════════════════════

_DARK_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    font_family="Pretendard, sans-serif",
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)

_COLORS = {
    "profit": "#10B981",    # 녹색 (수익)
    "loss": "#EF4444",      # 적색 (손실)
    "neutral": "#6B7280",   # 회색 (0)
    "primary": "#3B82F6",   # 파란색
    "secondary": "#8B5CF6", # 보라색
    "accent": "#F59E0B",    # 황색
    "bg_card": "#1a1a2e",
}


def _apply_dark_theme(fig, height=380, title=""):
    """모든 차트에 공통 다크 테마 적용."""
    fig.update_layout(
        **_DARK_TEMPLATE,
        height=height,
        title=dict(text=title, font=dict(size=14, color="white")),
        margin=dict(t=50 if title else 30, b=30, l=50, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    return fig


# ═══════════════════════════════════════════════════
#  1. 누적 수익률 곡선 (Equity Curve)
# ═══════════════════════════════════════════════════

def plot_equity_curve(
    daily_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "📈 자산 성장 곡선 (복리 기준)",
    height: int = 400,
) -> go.Figure:
    """누적 수익률 곡선 + 벤치마크 비교 + MDD 영역.

    Args:
        daily_returns: 일별 수익률(%) Series (index=날짜)
        benchmark_returns: (선택) 벤치마크 수익률(%) Series
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
    )

    # 전략 누적 수익곡선
    equity = (1 + daily_returns / 100).cumprod()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", fill="tozeroy",
        line=dict(color=_COLORS["primary"], width=2.5),
        fillcolor="rgba(59,130,246,0.08)",
        name="전략 수익률",
    ), row=1, col=1)

    # 벤치마크
    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_eq = (1 + benchmark_returns / 100).cumprod()
        fig.add_trace(go.Scatter(
            x=bench_eq.index, y=bench_eq.values,
            mode="lines",
            line=dict(color=_COLORS["neutral"], width=1.5, dash="dash"),
            name="벤치마크 (KOSPI)",
        ), row=1, col=1)

    # 원금 기준선
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="원금", annotation_font_color="gray",
                  row=1, col=1)

    # MDD (Drawdown) 하단
    peak = equity.cummax()
    drawdown = ((equity - peak) / peak) * 100
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode="lines", fill="tozeroy",
        line=dict(color=_COLORS["loss"], width=1),
        fillcolor="rgba(239,68,68,0.12)",
        name=f"낙폭 (MDD: {drawdown.min():.1f}%)",
    ), row=2, col=1)

    fig.update_yaxes(title_text="자산 가치", row=1, col=1)
    fig.update_yaxes(title_text="낙폭(%)", row=2, col=1)

    return _apply_dark_theme(fig, height, title)


# ═══════════════════════════════════════════════════
#  2. 월별/요일별 수익금 히트맵
# ═══════════════════════════════════════════════════

def plot_monthly_heatmap(
    trades_df: pd.DataFrame,
    date_col: str = "created_at",
    profit_col: str = "profit_pct",
    mode: str = "monthly",  # "monthly" | "weekday"
    title: str = "",
    height: int = 350,
) -> go.Figure:
    """월별 또는 요일별 수익률 히트맵.

    Args:
        trades_df: 거래 기록 DataFrame
        date_col: 날짜 컬럼명
        profit_col: 수익률 컬럼명
        mode: "monthly" (연×월) 또는 "weekday" (요일×시간대)
    """
    df = trades_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, profit_col])

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))
        return _apply_dark_theme(fig, height, title or "히트맵")

    if mode == "monthly":
        df["년"] = df[date_col].dt.year
        df["월"] = df[date_col].dt.month
        pivot = df.pivot_table(values=profit_col, index="년", columns="월", aggfunc="sum")
        pivot = pivot.reindex(columns=range(1, 13), fill_value=0)

        x_labels = [f"{m}월" for m in range(1, 13)]
        y_labels = [str(y) for y in pivot.index]

        default_title = "📅 월별 누적 수익률 (%)"
    else:
        day_names = ["월", "화", "수", "목", "금"]
        df["요일"] = df[date_col].dt.dayofweek  # 0=Mon
        df = df[df["요일"] < 5]  # 주말 제외

        pivot = df.pivot_table(values=profit_col, index="요일", aggfunc=["sum", "count", "mean"])

        # 요일별 단순 합산
        day_sums = df.groupby("요일")[profit_col].agg(["sum", "count", "mean"])
        pivot = pd.DataFrame({
            "sum": day_sums["sum"],
            "count": day_sums["count"],
            "mean": day_sums["mean"],
        }).reindex(range(5), fill_value=0)

        x_labels = ["합계(%)"]
        y_labels = day_names
        default_title = "📅 요일별 수익률 분포"

    if mode == "monthly":
        z_data = pivot.values
        text_data = [[f"{v:+.1f}%" for v in row] for row in z_data]
    else:
        z_data = pivot[["sum"]].values
        text_data = [[f"{v:+.1f}%"] for v in pivot["sum"].values]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        text=text_data,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=[
            [0, _COLORS["loss"]],
            [0.5, "#1a1a2e"],
            [1, _COLORS["profit"]],
        ],
        zmid=0,
        showscale=True,
        colorbar=dict(title="수익률(%)", tickfont=dict(color="white")),
    ))

    return _apply_dark_theme(fig, height, title or default_title)


# ═══════════════════════════════════════════════════
#  3. 캘린더 형태 손익 분포도
# ═══════════════════════════════════════════════════

def plot_calendar_pnl(
    trades_df: pd.DataFrame,
    date_col: str = "created_at",
    profit_col: str = "profit_pct",
    year: Optional[int] = None,
    title: str = "📆 일별 손익 캘린더",
    height: int = 250,
) -> go.Figure:
    """GitHub 잔디 스타일 일별 손익 캘린더.

    각 셀 = 하루, 색상 = 수익/손실, 가로축 = 주차, 세로축 = 요일
    """
    df = trades_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if year:
        df = df[df[date_col].dt.year == year]
    elif not df.empty:
        year = df[date_col].dt.year.max()
        df = df[df[date_col].dt.year == year]

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))
        return _apply_dark_theme(fig, height, title)

    # 일별 합산
    daily = df.groupby(df[date_col].dt.date)[profit_col].sum().reset_index()
    daily.columns = ["date", "pnl"]
    daily["date"] = pd.to_datetime(daily["date"])

    # 연초~연말 범위 생성
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    all_dates = pd.date_range(start, end)

    cal_df = pd.DataFrame({"date": all_dates})
    cal_df = cal_df.merge(daily, on="date", how="left")
    cal_df["pnl"] = cal_df["pnl"].fillna(0)

    cal_df["week"] = cal_df["date"].dt.isocalendar().week.astype(int)
    cal_df["weekday"] = cal_df["date"].dt.dayofweek  # 0=Mon

    # 주말 제외
    cal_df = cal_df[cal_df["weekday"] < 5]

    # Pivot: 요일(y) × 주차(x) → pnl
    pivot = cal_df.pivot_table(values="pnl", index="weekday", columns="week", aggfunc="sum")
    pivot = pivot.fillna(0)

    day_labels = ["월", "화", "수", "목", "금"]

    # 날짜 호버 텍스트 생성
    hover_text = cal_df.pivot_table(
        values="date", index="weekday", columns="week",
        aggfunc=lambda x: x.iloc[0].strftime("%m/%d") if len(x) > 0 else ""
    ).fillna("")

    z_vals = pivot.values
    customdata = hover_text.reindex(index=pivot.index, columns=pivot.columns).fillna("").values

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=[str(w) for w in pivot.columns],
        y=day_labels[:len(pivot.index)],
        customdata=customdata,
        hovertemplate="%{customdata}<br>수익률: %{z:+.1f}%<extra></extra>",
        colorscale=[
            [0, _COLORS["loss"]],
            [0.5, "#1e1e3a"],
            [1, _COLORS["profit"]],
        ],
        zmid=0,
        showscale=True,
        colorbar=dict(title="%", len=0.8, tickfont=dict(color="white")),
        xgap=2, ygap=2,
    ))

    fig.update_xaxes(title_text="주차", showticklabels=False)
    fig.update_yaxes(title_text="")

    return _apply_dark_theme(fig, height, f"{title} ({year})")


# ═══════════════════════════════════════════════════
#  4. 점수 구간별 승률 바 차트
# ═══════════════════════════════════════════════════

def plot_score_winrate(
    trades_df: pd.DataFrame,
    score_col: str = "score",
    profit_col: str = "net_ret",
    bins: List[int] = None,
    title: str = "🎯 점수 구간별 승률 & 평균 수익",
    height: int = 380,
) -> go.Figure:
    """점수 구간별 승률 + 평균 수익률 이중축 차트.

    Args:
        trades_df: 거래 기록 (score, profit 컬럼 필요)
        bins: 구간 경계 (기본: 40~95, 5점 단위)
    """
    df = trades_df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce")
    df = df.dropna(subset=[score_col, profit_col])

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))
        return _apply_dark_theme(fig, height, title)

    if bins is None:
        bins = list(range(40, 100, 5))

    df["구간"] = pd.cut(df[score_col], bins=bins + [100],
                        labels=[f"{b}-{b+4}" for b in bins], right=False)
    df = df.dropna(subset=["구간"])

    stats = df.groupby("구간", observed=True).agg(
        승률=pd.NamedAgg(column=profit_col, aggfunc=lambda x: (x > 0).mean() * 100),
        평균수익=pd.NamedAgg(column=profit_col, aggfunc="mean"),
        거래수=pd.NamedAgg(column=profit_col, aggfunc="count"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 승률 바
    colors = [_COLORS["profit"] if v >= 50 else _COLORS["loss"] for v in stats["승률"]]
    fig.add_trace(go.Bar(
        x=stats["구간"], y=stats["승률"],
        name="승률(%)",
        marker_color=colors,
        opacity=0.8,
        text=[f"{v:.0f}%" for v in stats["승률"]],
        textposition="outside",
        textfont=dict(size=11),
    ), secondary_y=False)

    # 평균 수익 라인
    fig.add_trace(go.Scatter(
        x=stats["구간"], y=stats["평균수익"],
        name="평균 수익(%)",
        mode="lines+markers+text",
        line=dict(color=_COLORS["accent"], width=2.5),
        marker=dict(size=8),
        text=[f"{v:+.1f}" for v in stats["평균수익"]],
        textposition="top center",
        textfont=dict(size=10, color=_COLORS["accent"]),
    ), secondary_y=True)

    fig.update_yaxes(title_text="승률 (%)", secondary_y=False)
    fig.update_yaxes(title_text="평균 수익 (%)", secondary_y=True)
    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="50%", secondary_y=False)

    return _apply_dark_theme(fig, height, title)


# ═══════════════════════════════════════════════════
#  5. 일별 거래 건수 + 수익률 이중축
# ═══════════════════════════════════════════════════

def plot_daily_dual_axis(
    trades_df: pd.DataFrame,
    date_col: str = "rec_date",
    profit_col: str = "net_ret",
    title: str = "📊 일별 거래 건수 & 평균 수익률",
    height: int = 350,
) -> go.Figure:
    """일별 거래 건수(바) + 평균 수익률(라인) 이중축."""
    df = trades_df.copy()
    df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce")
    df = df.dropna(subset=[profit_col])

    if df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, height, title)

    daily = df.groupby(date_col).agg(
        거래수=pd.NamedAgg(column=profit_col, aggfunc="count"),
        평균수익=pd.NamedAgg(column=profit_col, aggfunc="mean"),
        누적합=pd.NamedAgg(column=profit_col, aggfunc="sum"),
    ).reset_index().sort_values(date_col)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 거래 건수 바
    fig.add_trace(go.Bar(
        x=daily[date_col], y=daily["거래수"],
        name="거래 건수",
        marker_color="rgba(59,130,246,0.4)",
        opacity=0.7,
    ), secondary_y=False)

    # 평균 수익률 라인
    colors = [_COLORS["profit"] if v >= 0 else _COLORS["loss"] for v in daily["평균수익"]]
    fig.add_trace(go.Scatter(
        x=daily[date_col], y=daily["평균수익"],
        name="평균 수익(%)",
        mode="lines+markers",
        line=dict(color=_COLORS["accent"], width=2),
        marker=dict(size=5, color=colors),
    ), secondary_y=True)

    fig.update_yaxes(title_text="거래 건수", secondary_y=False)
    fig.update_yaxes(title_text="평균 수익(%)", secondary_y=True)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", secondary_y=True)

    return _apply_dark_theme(fig, height, title)


# ═══════════════════════════════════════════════════
#  6. 섹터별 수익 기여도 트리맵
# ═══════════════════════════════════════════════════

def plot_sector_treemap(
    trades_df: pd.DataFrame,
    sector_col: str = "업종_대분류",
    profit_col: str = "net_ret",
    name_col: str = "name",
    title: str = "🏭 섹터별 수익 기여도",
    height: int = 400,
) -> go.Figure:
    """섹터별 수익 기여도 트리맵.

    면적 = 거래 건수, 색상 = 평균 수익률
    """
    df = trades_df.copy()
    if sector_col not in df.columns:
        # 섹터 정보 없으면 종목명 기반으로 단순화
        sector_col = name_col
        if name_col not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="섹터 정보 없음", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))
            return _apply_dark_theme(fig, height, title)

    df[profit_col] = pd.to_numeric(df[profit_col], errors="coerce")
    df = df.dropna(subset=[profit_col])

    if df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, height, title)

    sector_stats = df.groupby(sector_col).agg(
        건수=pd.NamedAgg(column=profit_col, aggfunc="count"),
        평균수익=pd.NamedAgg(column=profit_col, aggfunc="mean"),
        합계수익=pd.NamedAgg(column=profit_col, aggfunc="sum"),
    ).reset_index()

    sector_stats = sector_stats[sector_stats["건수"] >= 1]

    fig = go.Figure(go.Treemap(
        labels=sector_stats[sector_col],
        parents=[""] * len(sector_stats),
        values=sector_stats["건수"],
        marker=dict(
            colors=sector_stats["평균수익"],
            colorscale=[
                [0, _COLORS["loss"]],
                [0.5, "#1e1e3a"],
                [1, _COLORS["profit"]],
            ],
            cmid=0,
            showscale=True,
            colorbar=dict(title="평균수익(%)", tickfont=dict(color="white")),
        ),
        text=[f"{n}<br>거래: {c}건<br>수익: {s:+.1f}%"
              for n, c, s in zip(sector_stats[sector_col],
                                  sector_stats["건수"],
                                  sector_stats["합계수익"])],
        hovertemplate="%{text}<extra></extra>",
        textfont=dict(size=13, color="white"),
    ))

    return _apply_dark_theme(fig, height, title)


# ═══════════════════════════════════════════════════
#  7. 포트폴리오 리스크 게이지 (VaR / 집중도)
# ═══════════════════════════════════════════════════

def plot_risk_gauge(
    value: float,
    title: str = "리스크",
    max_val: float = 100,
    thresholds: List[float] = None,
    height: int = 200,
) -> go.Figure:
    """반원 게이지 차트 (리스크 지표용).

    Args:
        value: 현재 값
        thresholds: [낮음, 보통, 높음] 경계 (기본: [30, 60, 90])
    """
    if thresholds is None:
        thresholds = [30, 60, 90]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(size=14, color="white")),
        number=dict(suffix="%", font=dict(size=24, color="white")),
        gauge=dict(
            axis=dict(range=[0, max_val], tickfont=dict(color="gray")),
            bar=dict(color=_COLORS["primary"]),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, thresholds[0]], color="rgba(16,185,129,0.2)"),
                dict(range=[thresholds[0], thresholds[1]], color="rgba(245,158,11,0.2)"),
                dict(range=[thresholds[1], max_val], color="rgba(239,68,68,0.2)"),
            ],
            threshold=dict(
                line=dict(color=_COLORS["loss"], width=3),
                thickness=0.8,
                value=value,
            ),
        ),
    ))

    return _apply_dark_theme(fig, height, "")


# ═══════════════════════════════════════════════════
#  NiceGUI 통합 헬퍼
# ═══════════════════════════════════════════════════

def render_viz_section(fig: go.Figure, container=None):
    """NiceGUI에 Plotly 차트 렌더링.

    Usage:
        from nicegui import ui
        from viz_components import plot_monthly_heatmap, render_viz_section

        fig = plot_monthly_heatmap(trades_df)
        render_viz_section(fig)
    """
    from nicegui import ui

    if container:
        with container:
            ui.plotly(fig).classes("w-full")
    else:
        ui.plotly(fig).classes("w-full")


def trim_for_viz(
    df: pd.DataFrame,
    date_col: str = "created_at",
    max_years: int = 2,
) -> pd.DataFrame:
    """시각화 전 데이터 크기 최적화 — 최근 N년 치만 필터링.

    브라우저 렌더링 부하 방지용.
    수년 치 데이터가 쌓였을 때 차트 전달 전 호출:
        trimmed = trim_for_viz(journal_df, date_col="created_at", max_years=2)
        fig = plot_calendar_pnl(trimmed)

    Args:
        df: 원본 DataFrame
        date_col: 날짜 컬럼명
        max_years: 유지할 최근 연수 (기본 2년)

    Returns:
        최근 N년 치만 남은 DataFrame
    """
    if df.empty or date_col not in df.columns:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=max_years)
    filtered = df[df[date_col] >= cutoff]

    if len(filtered) < len(df):
        _logger.info(f"viz 최적화: {len(df)} → {len(filtered)}행 ({max_years}년 필터)")

    return filtered
