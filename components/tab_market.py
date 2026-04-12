# -*- coding: utf-8 -*-
"""
tab_market.py — 📊 시장 현황 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
공포/탐욕 지수, 매크로 스파크라인, 섹터 트리맵/모멘텀
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from nicegui import ui

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from async_helpers import run_sync
except ImportError:
    async def run_sync(fn, *a, **kw):
        return fn(*a, **kw)

FDR_OK = False
fdr = None
try:
    import FinanceDataReader as _fdr
    fdr = _fdr
    FDR_OK = True
except ImportError:
    pass

from chart_components import (
    plot_fear_greed_gauge, plot_sector_treemap, plot_sector_momentum_bar,
)
from shared_utils import safe_float

_logger = logging.getLogger(__name__)

# 매크로 캐시 (1시간 TTL)
_MACRO_CACHE: dict = {}
_MACRO_CACHE_TIME: dict = {}


# ── UI 유틸 ──

def _section_title(text):
    ui.label(text).classes("text-lg font-bold text-white mt-6 mb-2 border-b border-gray-700 pb-2")


def _metric_card(title, value, delta="", positive=True):
    with ui.card().classes("p-4 min-w-[140px] bg-[#1a1a2e] border border-gray-700 rounded-xl"):
        ui.label(title).classes("text-xs text-gray-400 uppercase tracking-wide")
        ui.label(str(value)).classes("text-xl font-bold text-white mt-1")
        if delta:
            color = "text-green-400" if positive else "text-red-400"
            ui.label(str(delta)).classes(f"text-sm {color} mt-0.5")


def _plotly_dark(fig, height=300):
    if fig:
        fig.update_layout(
            height=height, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
            margin=dict(t=30, b=10, l=10, r=10),
        )
    return fig


# ── 매크로 스파크라인 ──

def _render_macro_sparklines():
    if not FDR_OK:
        return

    MACRO_TICKERS = [
        ("USD/KRW",    "USD/KRW",   "#F59E0B"),
        ("NASDAQ",     "IXIC",      "#3B82F6"),
        ("KOSPI",      "KS11",      "#10B981"),
        ("US 10Y",     "US10YT=RR", "#E040FB"),
    ]

    with ui.card().classes("w-full p-3 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
        ui.label("🌍 글로벌 매크로").classes("text-xs text-gray-400 mb-2")
        with ui.row().classes("w-full gap-3 flex-wrap"):
            for label, ticker, color in MACRO_TICKERS:
                _spark_card(label, ticker, color)


def _spark_card(label: str, ticker: str, color: str):
    with ui.card().classes("flex-1 min-w-[140px] p-2 bg-[#1a1a2e] border border-gray-700/50 rounded-lg"):
        val_label = ui.label("—").classes("text-sm font-bold text-white")
        chg_label = ui.label("—").classes("text-xs")
        chart_slot = ui.column().classes("w-full")

        async def _load():
            try:
                now = time.time()
                if ticker in _MACRO_CACHE and (now - _MACRO_CACHE_TIME.get(ticker, 0)) < 3600:
                    d = _MACRO_CACHE[ticker]
                else:
                    start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
                    d = await run_sync(fdr.DataReader, ticker, start)
                    if d is not None and not d.empty:
                        d = d.tail(20)
                        _MACRO_CACHE[ticker] = d
                        _MACRO_CACHE_TIME[ticker] = now

                if d is None or d.empty:
                    val_label.set_text("N/A")
                    return
                last = float(d["Close"].iloc[-1])
                prev = float(d["Close"].iloc[-2]) if len(d) > 1 else last
                chg = (last - prev) / prev * 100 if prev else 0

                if ticker in ("USD/KRW",):
                    fmt = f"{last:,.1f}"
                elif "10Y" in ticker:
                    fmt = f"{last:.3f}%"
                else:
                    fmt = f"{last:,.2f}"

                val_label.set_text(f"{label}: {fmt}")
                chg_label.set_text(f"{chg:+.2f}%")
                chg_label.classes(replace="text-xs text-green-400" if chg >= 0 else "text-xs text-red-400")

                if PLOTLY_OK:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(d))),
                        y=d["Close"].tolist(),
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        fill="tozeroy",
                        fillcolor=f"{color}22",
                        showlegend=False,
                    ))
                    fig.update_layout(
                        height=50, margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                    )
                    chart_slot.clear()
                    with chart_slot:
                        ui.plotly(fig).classes("w-full")
            except Exception as e:
                _logger.warning(f"⚠️ 매크로 조회 실패 ({ticker}): {e}")
                val_label.set_text(f"{label}: —")

        async def _safe_load():
            await asyncio.sleep(0.5)
            try:
                if chart_slot.is_deleted:
                    return
            except AttributeError:
                pass
            await _load()

        asyncio.create_task(_safe_load())


# ── 공포/탐욕 지수 ──

def _get_fear_greed(df):
    if df.empty or "DISPLAY_SCORE" not in df.columns:
        return 50, "데이터 부족"
    avg = df["DISPLAY_SCORE"].mean()
    score = min(max(avg, 0), 100)
    if score >= 80: label = "극단적 탐욕"
    elif score >= 60: label = "탐욕"
    elif score >= 40: label = "중립"
    elif score >= 20: label = "공포"
    else: label = "극단적 공포"
    return score, label


# ── 메인 렌더 ──

def render_tab_market(df):
    """Tab 1: 시장 현황"""
    import os, json

    fg_score, fg_label = _get_fear_greed(df)
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # run_meta 로드
    meta = {}
    try:
        mp = os.path.join(DATA_DIR, "run_meta_latest.json")
        if os.path.exists(mp):
            with open(mp, 'r') as f:
                meta = json.load(f)
    except Exception:
        pass

    _render_macro_sparklines()

    # ── [v21.3] 엔진 상태 요약 ──
    macro_risk = meta.get("macro_risk", "—")
    breadth = meta.get("market_breadth", 0)
    confidence = meta.get("confidence_score", 0)
    max_route = meta.get("max_allowed_route", "—")
    macro_msg = meta.get("macro_msg", "")

    risk_color = {"NORMAL": "#10B981", "CAUTION": "#F59E0B", "WARNING": "#EF4444", "CRITICAL": "#DC2626"}.get(macro_risk, "#6B7280")
    risk_kr = {"NORMAL": "정상", "CAUTION": "주의", "WARNING": "경고", "CRITICAL": "위험"}.get(macro_risk, macro_risk)

    with ui.card().classes("w-full p-4 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
        ui.label("🛡️ 엔진 상태").classes("text-xs text-gray-400 mb-2")
        with ui.row().classes("w-full gap-4 flex-wrap"):
            with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                ui.label("매크로 리스크").classes("text-xs text-gray-400")
                ui.label(f"{'🟢' if macro_risk == 'NORMAL' else '🟡' if macro_risk == 'CAUTION' else '🔴'} {risk_kr}").classes("text-lg font-bold").style(f"color:{risk_color}")
                if macro_msg:
                    ui.label(macro_msg).classes("text-xs text-gray-500")

            with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                ui.label("시장 Breadth").classes("text-xs text-gray-400")
                bc = "#10B981" if breadth >= 60 else "#F59E0B" if breadth >= 40 else "#EF4444"
                ui.label(f"{breadth:.1f}%").classes("text-lg font-bold").style(f"color:{bc}")
                ui.label("상승 종목 비율").classes("text-xs text-gray-500")

            with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                ui.label("엔진 신뢰도").classes("text-xs text-gray-400")
                cc = "#10B981" if confidence >= 80 else "#F59E0B" if confidence >= 50 else "#EF4444"
                ui.label(f"{confidence:.0f}/100").classes("text-lg font-bold").style(f"color:{cc}")
                ui.label(f"최대허용: {max_route}").classes("text-xs text-gray-500")

            # ELITE/TOP_PICK 요약
            if "ELITE_SCORE" in df.columns:
                elite_avg = df["ELITE_SCORE"].mean()
                tp_count = int(df.get("TOP_PICK", pd.Series(0)).sum()) if "TOP_PICK" in df.columns else 0
                with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                    ui.label("오늘의 ELITE").classes("text-xs text-gray-400")
                    ui.label(f"🏆 {tp_count}종목").classes("text-lg font-bold text-yellow-400")
                    ui.label(f"평균 ELITE {elite_avg:.0f}").classes("text-xs text-gray-500")

    # ── [v21.3] 오늘의 Top 추천 — ELITE 기준 ──
    try:
        if "ELITE_SCORE" in df.columns:
            _top_df = df.copy()
            # TOP_PICK 우선, 없으면 ELITE 상위
            if "TOP_PICK" in _top_df.columns:
                _picks = _top_df[_top_df["TOP_PICK"].astype(int) == 1].nlargest(3, "ELITE_SCORE")
                if len(_picks) < 3:
                    _extra = _top_df[~_top_df.index.isin(_picks.index)].nlargest(3 - len(_picks), "ELITE_SCORE")
                    _picks = pd.concat([_picks, _extra])
            else:
                _picks = _top_df.nlargest(3, "ELITE_SCORE")

            if not _picks.empty:
                with ui.card().classes("w-full p-4 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
                    ui.label("🏆 오늘의 ELITE Top").classes("text-xs text-gray-400 mb-2")
                    with ui.row().classes("w-full gap-3 flex-wrap"):
                        for _, s in _picks.iterrows():
                            route = str(s.get("ROUTE", ""))
                            route_icon = {"ATTACK": "🚀", "ARMED": "🔫", "CARRY": "📌"}.get(route, "👀")
                            elite = safe_float(s.get("ELITE_SCORE", 0))
                            close = safe_float(s.get("종가", 0))
                            tp1 = safe_float(s.get("추천매도가1", 0))
                            rr = safe_float(s.get("RR_NOW_TP1", 0))
                            wr = safe_float(s.get("EST_WIN_RATE", 0))
                            bal = safe_float(s.get("BALANCE_SCORE", 0))
                            tp_flag = "🏆 " if int(s.get("TOP_PICK", 0)) == 1 else ""
                            tp1_pct = (tp1 / close - 1) * 100 if close > 0 else 0

                            with ui.card().classes("flex-1 min-w-[200px] p-3 bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                                with ui.row().classes("items-center gap-2"):
                                    ui.label(f"{route_icon} {tp_flag}{s.get('종목명', '')}").classes("text-white font-bold text-sm")
                                    ui.badge(f"E{elite:.0f}", color="#10B981" if elite >= 80 else "#3B82F6").classes("text-xs")
                                ui.label(f"S{safe_float(s.get('STRUCT_SCORE', 0)):.0f} T{safe_float(s.get('TIMING_SCORE', 0)):.0f} AI{safe_float(s.get('AI_SCORE', 0)):.0f} | 균형 {bal:.0f}").classes("text-xs text-gray-400 mt-1")
                                ui.label(f"{close:,.0f} → {tp1:,.0f} ({tp1_pct:+.1f}%) | RR {rr:.1f}:1 | 승률 {wr * 100:.0f}%").classes("text-xs text-cyan-400")
    except Exception as _te:
        _logger.warning(f"Top 추천 렌더 실패: {_te}")

    _section_title("📡 시장 현황")
    with ui.row().classes("w-full gap-4 flex-wrap"):
        fg_icon = "🟢" if fg_score >= 50 else "🔴"
        _metric_card("시장 심리", f"{fg_icon} {fg_label}", f"지수: {fg_score:.0f}/100", fg_score >= 50)

        if "ret_1d_%" in df.columns:
            avg_ret = df.head(20)["ret_1d_%"].mean()
            _metric_card("Top20 평균 수익률", f"{avg_ret:+.2f}%", "전일 대비", avg_ret >= 0)

        total = len(df)
        armed = len(df[df.get("ROUTE", pd.Series()).str.contains("ARMED|ATTACK", na=False)]) if "ROUTE" in df.columns else 0
        _metric_card("분석 종목", f"{total}개", f"ARMED/ATTACK: {armed}개")

    _section_title("🌡️ 공포/탐욕 & 주도 섹터")
    with ui.row().classes("w-full gap-4 flex-wrap items-start"):
        with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
            fig_g = plot_fear_greed_gauge(fg_score)
            if fig_g:
                ui.plotly(_plotly_dark(fig_g, 280)).classes("w-full")

        with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
            ui.label("🔥 오늘의 주도 섹터").classes("text-sm font-bold text-white mb-2")
            if "업종" in df.columns:
                fig_m = plot_sector_treemap(df.head(50))
                if fig_m:
                    ui.plotly(_plotly_dark(fig_m, 280)).classes("w-full")
                else:
                    ui.label("섹터 데이터 부족").classes("text-gray-500")

    _section_title("🚀 섹터 모멘텀 Top 10")
    fig_mom = plot_sector_momentum_bar(df)
    if fig_mom and len(fig_mom.data) > 0:
        ui.plotly(_plotly_dark(fig_mom, 350)).classes("w-full")
    else:
        ui.label("모멘텀 데이터 부족").classes("text-gray-500")
