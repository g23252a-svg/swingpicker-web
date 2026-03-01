# -*- coding: utf-8 -*-
"""
chart_components.py — Plotly 차트 컴포넌트 모듈
──────────────────────────────────────────────────
dashboard.py 에서 추출한 시각화 함수 전체.

순수 Plotly (go.Figure 반환) 함수와
Streamlit 의존 함수를 분리 표기합니다.

사용법
------
    from chart_components import (
        plot_ai_gauge_chart, plot_radar_chart, plot_score_waterfall,
        plot_fear_greed_gauge, plot_sector_treemap, plot_sector_momentum_bar,
        plot_ai_consensus, plot_opportunity_map, plot_kelly_visual,
        add_volume_profile,
    )
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ═══════════════════════════════════════════════════
#  1. 게이지 (Gauge) 차트
# ═══════════════════════════════════════════════════

def plot_ai_gauge_chart(score: float) -> go.Figure:
    """AI 종합 신뢰도 게이지 (프리미엄)"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🏆 AI 종합 신뢰도", 'font': {'size': 16, 'family': 'Outfit'}},
        delta={'reference': 80,
               'increasing': {'color': "#10B981"},
               'decreasing': {'color': "#EF4444"}},
        number={'font': {'size': 42, 'family': 'Outfit', 'color': '#F1F5F9'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1,
                     'tickcolor': "rgba(255,255,255,0.2)",
                     'tickfont': {'color': 'rgba(255,255,255,0.5)', 'size': 10}},
            'bar': {'color': "#3B82F6", 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.03)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239,68,68,0.12)'},
                {'range': [40, 60], 'color': 'rgba(245,158,11,0.10)'},
                {'range': [60, 80], 'color': 'rgba(59,130,246,0.10)'},
                {'range': [80, 100], 'color': 'rgba(16,185,129,0.12)'},
            ],
            'threshold': {
                'line': {'color': "#F59E0B", 'width': 3},
                'thickness': 0.75,
                'value': score,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Outfit'},
    )
    return fig


def plot_fear_greed_gauge(score: float) -> go.Figure:
    """시장 공포/탐욕 지수 게이지"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "시장 공포/탐욕 지수", 'font': {'size': 20}},
        delta={'reference': 50,
               'increasing': {'color': "red"},
               'decreasing': {'color': "blue"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 25], 'color': '#4D96FF'},
                {'range': [25, 45], 'color': '#87CEEB'},
                {'range': [45, 55], 'color': '#D3D3D3'},
                {'range': [55, 75], 'color': '#FFB347'},
                {'range': [75, 100], 'color': '#FF6B6B'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score,
            },
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ═══════════════════════════════════════════════════
#  2. 켈리 비중 시각화
# ═══════════════════════════════════════════════════

def plot_kelly_visual(win_rate_est: float, reward_risk: float,
                      kelly_pct: float) -> go.Figure:
    """켈리 베팅 비중 수평 바 차트"""
    metrics = ['승률(Win Rate)', '손익비(Reward/Risk)', '켈리 권장 비중']
    values = [win_rate_est * 100, reward_risk * 10, kelly_pct * 100]
    colors = ['#FF7043', '#42A5F5', '#66BB6A']
    text_vals = [f"{win_rate_est * 100:.1f}%",
                 f"{reward_risk:.2f}배",
                 f"{kelly_pct * 100:.1f}%"]

    fig = go.Figure(go.Bar(
        x=values, y=metrics, orientation='h',
        marker_color=colors, text=text_vals, textposition='auto',
    ))
    fig.update_layout(
        title="💰 켈리 자금 관리 (승률 vs 손익비)",
        height=200, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ═══════════════════════════════════════════════════
#  3. 레이더 차트 (7-Factor)
# ═══════════════════════════════════════════════════

def plot_radar_chart(row) -> go.Figure:
    """종목별 7-Factor 레이더 차트"""
    raw_name = row.get('종목명')
    stock_name = str(raw_name) if pd.notna(raw_name) else "종목"

    def _safe(key, default=0):
        val = pd.to_numeric(row.get(key), errors='coerce')
        return float(val) if pd.notna(val) else default

    def _clamp(v, lo=0, hi=100):
        return max(lo, min(hi, v))

    # ── 7-Factor 실계산 (CSV 실제 컬럼 기반) ──
    close = _safe("종가")
    entry = _safe("추천매수가")
    stop = _safe("손절가")
    t1 = _safe("추천매도가1")

    # 1) 모멘텀: RSI14 (이미 0~100)
    momentum = _clamp(_safe("RSI14", 50))

    # 2) 가성비(RR): (T1-Entry)/(Entry-Stop), 4:1이면 100점
    risk = entry - stop if entry > stop else 1
    reward = t1 - entry if t1 > entry else 0
    rr_ratio = reward / risk if risk > 0 else 0
    rr_score = _clamp(rr_ratio / 4 * 100)

    # 3) 상승여력: T1까지 잔여 상승률, 20%이면 100점
    upside_pct = ((t1 / close) - 1) * 100 if close > 0 and t1 > 0 else 0
    upside_score = _clamp(upside_pct / 20 * 100)

    # 4) 안전마진: 종가→손절 거리, 10%이면 100점
    sl_dist_pct = ((close - stop) / close) * 100 if close > 0 and stop > 0 else 0
    safety_score = _clamp(sl_dist_pct / 10 * 100)

    # 5) 타이밍: TIMING_SCORE (이미 0~100)
    timing = _clamp(_safe("TIMING_SCORE", 50))

    # 6) 유동성: 거래대금(억원), 2000억이면 100점
    liquidity_raw = _safe("거래대금(억원)")
    liquidity = _clamp(liquidity_raw / 2000 * 100)

    # 7) 세력강도: V_POWER (-1~3 범위를 0~100으로)
    vp = _safe("V_POWER")
    tech_score = _clamp((vp + 1) / 4 * 100)  # -1→0, 3→100

    keys = ["모멘텀", "가성비(RR)", "상승여력", "안전마진",
            "타이밍", "유동성", "세력강도"]
    values = [momentum, rr_score, upside_score, safety_score,
              timing, liquidity, tech_score]

    # 폐곡선
    values += values[:1]
    keys += keys[:1]

    fig = go.Figure()

    # 80점 가이드라인
    fig.add_trace(go.Scatterpolar(
        r=[80] * len(keys), theta=keys,
        mode='lines',
        line=dict(color='rgba(0, 255, 0, 0.3)', width=1, dash='dot'),
        hoverinfo='skip', showlegend=False,
    ))

    # 메인 데이터
    fig.add_trace(go.Scatterpolar(
        r=values, theta=keys,
        fill='toself', name=stock_name,
        line=dict(color='#00E5FF', width=3),
        fillcolor='rgba(0, 229, 255, 0.3)',
        marker=dict(size=6, color='white'),
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9, color='gray'),
                            gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(tickfont=dict(size=12, weight='bold'),
                             gridcolor='rgba(255,255,255,0.1)'),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False, height=320,
        margin=dict(l=40, r=40, t=40, b=20),
        title=dict(text=f"💎 <b>{stock_name}</b> 7-Factor",
                   x=0.5, font=dict(size=16)),
    )
    return fig


# ═══════════════════════════════════════════════════
#  4. 워터폴 (점수 기여도)
# ═══════════════════════════════════════════════════

def plot_score_waterfall(row) -> go.Figure:
    """STRUCT + TIMING + AI → FINAL 기여도 워터폴"""
    if row is None:
        return go.Figure()

    def _get(k):
        return float(row.get(k, 0))

    struct = _get('STRUCT_SCORE')
    timing = _get('TIMING_SCORE')
    ai = _get('AI_SCORE')
    final = _get('FINAL_SCORE')

    val_s = struct * 0.4
    val_t = timing * 0.4
    val_a = ai * 0.2
    adj = final - (val_s + val_t + val_a)

    keys = ["구조(체력)", "타이밍(맥)", "AI(예측)", "보정", "최종점수"]
    values = [val_s, val_t, val_a, adj, final]
    measures = ["relative", "relative", "relative", "relative", "total"]

    fig = go.Figure(go.Waterfall(
        name="Score Breakdown", orientation="v",
        measure=measures, x=keys, y=values,
        text=[f"{v:.1f}" for v in values],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#FF5252"}},
        increasing={"marker": {"color": "#4CAF50"}},
        totals={"marker": {"color": "#2196F3"}},
    ))
    fig.update_layout(
        title="🧩 점수 구성 (Struct + Timing + AI)",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(title="점수", range=[0, 110], fixedrange=True),
    )
    return fig


# ═══════════════════════════════════════════════════
#  5. 섹터 트리맵
# ═══════════════════════════════════════════════════

def plot_sector_treemap(df_map: pd.DataFrame) -> go.Figure:
    """섹터별 거래대금 트리맵 (대분류 우선)"""
    if df_map is None or df_map.empty:
        return go.Figure()

    sector_key = "업종_대분류" if "업종_대분류" in df_map.columns else "업종"
    if sector_key not in df_map.columns:
        return go.Figure()

    fig = px.treemap(
        df_map,
        path=[sector_key, "종목명"],
        values="거래대금(억원)",
        color="LDY_SCORE",
        color_continuous_scale="RdYlGn",
        title="<b>🔥 시장 주도 섹터 지도</b>",
        custom_data=["LDY_SCORE", sector_key],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b>"
            "<br>섹터: %{customdata[1]}"
            "<br>점수: %{customdata[0]:.1f}"
            "<br>대금: %{value}억"
            "<extra></extra>"
        )
    )
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=350)
    return fig


# ═══════════════════════════════════════════════════
#  6. 섹터 모멘텀 바
# ═══════════════════════════════════════════════════

def plot_sector_momentum_bar(scored_df: pd.DataFrame) -> go.Figure:
    """섹터별 모멘텀 Top 10 수평 바 차트"""
    if scored_df is None or scored_df.empty:
        return go.Figure()

    sector_col = ("업종_대분류" if "업종_대분류" in scored_df.columns
                  else "업종" if "업종" in scored_df.columns else None)
    if sector_col is None:
        return go.Figure()

    metric = "ret_5d_%" if "ret_5d_%" in scored_df.columns else "LDY_SCORE"

    grp = (scored_df
           .dropna(subset=[sector_col, metric])
           .groupby(sector_col)[metric]
           .mean()
           .sort_values(ascending=False)
           .head(10))
    if grp.empty:
        return go.Figure()

    fmt = "{:.2f}%" if metric == "ret_5d_%" else "{:.2f}"
    fig = go.Figure(go.Bar(
        x=grp.values, y=grp.index, orientation="h",
        text=[fmt.format(v) for v in grp.values],
        textposition="auto",
    ))
    title_metric = "5일 평균 수익률" if metric == "ret_5d_%" else "LDY 평균 점수"
    fig.update_layout(
        title=f"🚀 섹터 모멘텀 Top 10 ({title_metric})",
        height=320, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ═══════════════════════════════════════════════════
#  7. AI vs 퀀트 합의 산점도
# ═══════════════════════════════════════════════════

def plot_ai_consensus(df: pd.DataFrame):
    """AI Score vs Rule Score 산점도 (우상단 = 양쪽 강추)"""
    if df is None or df.empty or "ML_SCORE" not in df.columns:
        return None

    plot_df = df[(df["RANK_SCORE"] > 0) & (df["ML_SCORE"] > 0)].copy()
    if plot_df.empty:
        return None

    fig = px.scatter(
        plot_df,
        x="RANK_SCORE", y="ML_SCORE",
        color="TOTAL_SCORE", size="거래대금(억원)",
        hover_name="종목명",
        hover_data=["종목코드", "업종", "ROUTE"],
        color_continuous_scale="RdYlGn",
        title="<b>🧠 AI(세로) vs 퀀트(가로) 합의 지점</b>",
        labels={"RANK_SCORE": "퀀트(Rule) 점수", "ML_SCORE": "AI(ML) 예측 점수"},
    )
    fig.add_hline(y=80, line_dash="dot", line_color="gray", annotation_text="AI 강력매수")
    fig.add_vline(x=80, line_dash="dot", line_color="gray", annotation_text="퀀트 강력매수")
    fig.add_shape(type="rect", x0=80, y0=80, x1=100, y1=100,
                  line=dict(color="red", width=2),
                  fillcolor="rgba(255, 0, 0, 0.1)")
    fig.update_layout(
        height=400, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[40, 105]), yaxis=dict(range=[40, 105]),
    )
    return fig


# ═══════════════════════════════════════════════════
#  8. Opportunity Map (버블)
# ═══════════════════════════════════════════════════

def plot_opportunity_map(df: pd.DataFrame):
    """TOTAL × TRIGGER 버블 차트 (우상단 = 주도주)"""
    if df is None or df.empty:
        return None
    try:
        plot_df = df.copy()
        for c in ["TOTAL_SCORE", "TRIGGER_SCORE", "거래대금(억원)"]:
            if c in plot_df.columns:
                plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce').fillna(0)

        plot_df['size_scaled'] = np.log1p(plot_df['거래대금(억원)']) * 2

        fig = px.scatter(
            plot_df,
            x="TOTAL_SCORE", y="TRIGGER_SCORE",
            size="size_scaled", color="FINAL_SCORE",
            color_continuous_scale="RdYlGn_r",
            hover_name="종목명",
            hover_data=["종목코드", "종가", "ROUTE"],
            text="종목명",
            title="<b>🚀 Opportunity Map (우상단 = 주도주)</b>",
        )
        fig.add_shape(type="rect", x0=80, y0=70, x1=100, y1=100,
                      fillcolor="rgba(0, 255, 0, 0.05)", line_width=0, layer="below")
        fig.add_annotation(x=90, y=85, text="🔥 주도주 영역", showarrow=False,
                           font=dict(size=20, color="rgba(0,255,0,0.3)"))
        fig.add_vline(x=70, line_dash="dot", line_color="gray", annotation_text="구조 우수")
        fig.add_hline(y=60, line_dash="dot", line_color="gray", annotation_text="타이밍 포착")

        fig.update_traces(
            textposition='top center',
            marker=dict(line=dict(width=1, color='DarkSlateGrey'), opacity=0.85),
        )
        fig.update_layout(
            height=550,
            xaxis=dict(title="⚙️ 구조 점수", range=[30, 105],
                       showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
            yaxis=dict(title="⚡ 타이밍 점수", range=[20, 105],
                       showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig
    except Exception:
        return None


# ═══════════════════════════════════════════════════
#  9. 매물대 (Volume Profile) 오버레이
# ═══════════════════════════════════════════════════

def add_volume_profile(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """차트 우측에 매물대 + POC 수평선 추가"""
    if df is None or df.empty:
        return fig

    typ_price = (df['High'] + df['Low'] + df['Close']) / 3
    price_min = df['Low'].min()
    price_max = df['High'].max()

    if price_min == price_max:
        return fig

    num_bins = 50 if len(df) > 100 else 30
    bins = np.linspace(price_min, price_max, num_bins)
    hist, bin_edges = np.histogram(typ_price, bins=bins, weights=df['Volume'])
    y_vals = (bin_edges[:-1] + bin_edges[1:]) / 2

    max_vol = hist.max() if len(hist) > 0 else 1
    max_vol_idx = np.argmax(hist)
    poc_price = y_vals[max_vol_idx]

    colors = ['rgba(128, 128, 128, 0.15)'] * len(hist)
    colors[max_vol_idx] = 'rgba(255, 165, 0, 0.4)'

    bar_trace = go.Bar(
        y=y_vals, x=hist, orientation='h', name='매물대',
        marker=dict(color=colors, line=dict(width=0)),
        xaxis='x2', hoverinfo='none', showlegend=False,
    )
    fig.add_trace(bar_trace, row=1, col=1)

    fig.add_hline(
        y=poc_price, line_dash="dot",
        line_color="rgba(255, 165, 0, 0.8)", line_width=1.5,
        annotation_text=f" POC ({int(poc_price):,})",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="orange"),
    )

    data_list = list(fig.data)
    if data_list:
        new_trace = data_list.pop()
        data_list.insert(0, new_trace)
        fig.data = tuple(data_list)

    fig.update_layout(
        xaxis2=dict(
            overlaying='x', side='top',
            showgrid=False, visible=False,
            range=[max_vol * 4, 0],
        )
    )
    return fig
