# -*- coding: utf-8 -*-
"""
tab_stocks.py — Tab 2: 종목 분석 (테이블 + 칸반 + 상세)
═══════════════════════════════════════════════════
[v3.4] (2026-04-17)
  #1 함수명: render_tab2_stocks → render_tab_stocks (main.py 호출부와 정합)
  #2 시그니처: (df, auth) → (df, auth, store=None) (main.py의 3-인자 호출 대응)
[v3.3]
  #1 ui.timer 제거 → asyncio.create_task 직접 실행
  #2 dialog.on_close → 유령 태스크 취소 (생명주기 관리)
  #3 차트 데이터 로드: run.io_bound (GIL 블로킹 방지)
"""
import asyncio
import os
import logging
from typing import Optional

import pandas as pd
from nicegui import ui, run, app

_logger = logging.getLogger(__name__)

# ── 외부 모듈 (지연 임포트) ──
try:
    from chart_components import (
        plot_candle_chart as _plot_candle,
        plot_radar_chart, plot_score_waterfall,
    )
except ImportError:
    _plot_candle = None
    plot_radar_chart = None
    plot_score_waterfall = None

try:
    from data_source import get_data_source
    _ds = get_data_source()
except ImportError:
    _ds = None


# ═══════════════════════════════════════════════════
#  유틸
# ═══════════════════════════════════════════════════

def _nz(val, default=0):
    """None/NaN → default"""
    try:
        v = float(val)
        return v if pd.notna(v) else default
    except (ValueError, TypeError):
        return default


def _plotly_dark(fig, height=300):
    """Plotly 차트 다크 테마"""
    if fig is None:
        return fig
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def _get_chart_data(code: str):
    """캔들차트 데이터 (동기 — run.io_bound로 호출)"""
    if _ds is None:
        return None
    try:
        return _ds.get_ohlcv(code, period=120)
    except Exception as e:
        _logger.warning(f"차트 데이터 로드 실패 [{code}]: {e}")
        return None


def _metric_card(icon: str, value: str, sub: str = "", positive: bool = True):
    """메트릭 카드 컴포넌트"""
    color = "text-green-400" if positive else "text-red-400"
    with ui.card().classes("p-3 bg-[rgba(255,255,255,0.05)] rounded-xl min-w-[140px]"):
        ui.label(icon).classes("text-xs text-gray-400")
        ui.label(value).classes(f"text-lg font-bold {color}")
        if sub:
            ui.label(sub).classes("text-xs text-gray-500")


# ═══════════════════════════════════════════════════
#  메인 렌더
# ═══════════════════════════════════════════════════

def render_tab_stocks(df: pd.DataFrame, auth: str, store=None):
    """Tab 2: AI & Quant 추천 종목

    Args:
        df: 스코어링된 종목 DataFrame
        auth: 사용자 권한 ("admin" / "premium" / "free" / ...)
        store: services.data_store.store 인스턴스 (현재 미사용, 장래 확장용)
    """

    ui.label("🎯 AI & Quant 추천 종목").classes(
        "text-xl font-bold text-white mb-4"
    )

    # ── 뷰모드 + 필터 ──
    with ui.row().classes("w-full gap-4 items-center flex-wrap mb-4"):
        view_mode = ui.toggle(
            ["📋 테이블", "🃏 칸반"], value="📋 테이블"
        )
        route_filter = ui.select(
            ["전체", "ATTACK", "ARMED", "WAIT", "NEUTRAL"],
            value="전체", label="상태",
        ).classes("min-w-[120px]")
        sort_mode = ui.toggle(
            ["🔢 점수순", "🚦 상태순"], value="🔢 점수순"
        )

    table_area = ui.column().classes("w-full")
    detail_area = ui.column().classes("w-full mt-4")

    def _filtered():
        fdf = df.copy()
        if route_filter.value != "전체" and "ROUTE" in fdf.columns:
            fdf = fdf[fdf["ROUTE"].astype(str).str.contains(
                route_filter.value, na=False
            )]
        if sort_mode.value == "🔢 점수순" and "DISPLAY_SCORE" in fdf.columns:
            fdf = fdf.sort_values("DISPLAY_SCORE", ascending=False)
        # 접근 제한
        limits = {"guest": 3, "free": 5, "pro": 20}
        fdf = fdf.head(limits.get(auth, 50))
        return fdf

    def _build_view():
        table_area.clear()
        show = _filtered()
        with table_area:
            if view_mode.value == "🃏 칸반":
                _render_kanban(show, df)
            else:
                _render_table(show, df)

    def _render_table(show: pd.DataFrame, full_df: pd.DataFrame):
        columns = [
            {"name": "route", "label": "상태", "field": "route", "align": "center"},
            {"name": "name", "label": "종목명", "field": "name", "align": "left"},
            {"name": "score", "label": "점수", "field": "score",
             "align": "center", "sortable": True},
            {"name": "close", "label": "현재가", "field": "close", "align": "right"},
            {"name": "buy", "label": "매수", "field": "buy", "align": "right"},
            {"name": "stop", "label": "손절", "field": "stop", "align": "right"},
            {"name": "t1", "label": "T1목표", "field": "t1", "align": "right"},
            {"name": "sector", "label": "업종", "field": "sector", "align": "left"},
        ]
        rows = []
        for _, r in show.iterrows():
            rows.append({
                "code": str(r.get("종목코드", "")).zfill(6),
                "route": str(r.get("ROUTE", "—")),
                "name": str(r.get("종목명", "—")),
                "score": f'{_nz(r.get("DISPLAY_SCORE", 0)):.0f}',
                "close": f'{int(_nz(r.get("종가", 0))):,}',
                "buy": f'{int(_nz(r.get("추천매수가", 0))):,}',
                "stop": f'{int(_nz(r.get("손절가", 0))):,}',
                "t1": f'{int(_nz(r.get("추천매도가1", 0))):,}',
                "sector": str(r.get("업종", "—")),
            })
        tbl = ui.table(
            columns=columns, rows=rows, row_key="code",
            selection="single", pagination={"rowsPerPage": 15},
        ).classes("w-full").props("dense dark flat bordered")
        tbl.on("selection", lambda e: _on_stock_select(e, full_df))

    def _render_kanban(show: pd.DataFrame, full_df: pd.DataFrame):
        if show.empty:
            ui.label("표시할 종목 없음").classes("text-gray-400")
            return
        route_col = "ROUTE" if "ROUTE" in show.columns else None
        if route_col:
            df_atk = show[show[route_col].astype(str).str.contains(
                "ATTACK", case=False, na=False
            )]
            df_arm = show[show[route_col].astype(str).str.contains(
                "ARMED", case=False, na=False
            )]
            ex = df_atk.index.union(df_arm.index)
            df_watch = show[~show.index.isin(ex)]
        else:
            df_atk = df_arm = pd.DataFrame()
            df_watch = show

        with ui.row().classes("w-full gap-4 flex-wrap items-start"):
            _kanban_col("🚀 ATTACK", df_atk, "#EF4444")
            _kanban_col("🔫 ARMED", df_arm, "#F59E0B")
            _kanban_col("👀 WATCH", df_watch, "#3B82F6")

    def _kanban_col(title: str, sub_df: pd.DataFrame, color: str):
        with ui.column().classes("kanban-col min-w-[280px] flex-1"):
            ui.label(f"{title} ({len(sub_df)})").classes(
                "text-white font-bold mb-2"
            ).style(f"border-bottom: 2px solid {color}")
            if sub_df.empty:
                ui.label("비어 있음").classes("text-gray-500 text-sm")
                return
            for _, r in sub_df.iterrows():
                score = _nz(r.get("DISPLAY_SCORE", 0))
                sc = "#10B981" if score >= 80 else "#3B82F6" if score >= 60 else "#94A3B8"
                with ui.card().classes(
                    "p-3 mb-2 cursor-pointer bg-[rgba(255,255,255,0.05)] "
                    "border border-[rgba(255,255,255,0.1)] rounded-xl "
                    "hover:bg-[rgba(255,255,255,0.08)]"
                ):
                    with ui.row().classes("justify-between items-center"):
                        ui.label(str(r.get("종목명", ""))).classes(
                            "text-white font-bold text-sm"
                        )
                        ui.badge(f"{score:.0f}", color=sc).classes("text-xs")
                    buy = int(_nz(r.get("추천매수가", 0)))
                    stop = int(_nz(r.get("손절가", 0)))
                    t1 = int(_nz(r.get("추천매도가1", 0)))
                    if buy > 0:
                        ui.label(
                            f"🎯 {buy:,}  🛡️ {stop:,}  🟢 {t1:,}"
                        ).classes("text-xs text-gray-400 mt-1")

    # ── 종목 상세 분석 ──
    def _on_stock_select(event, full_df: pd.DataFrame):
        detail_area.clear()
        sel = event.args.get("rows", []) if hasattr(event, "args") else []
        if not sel:
            return
        code = sel[0].get("code", "")
        match = full_df[full_df["종목코드"].astype(str).str.zfill(6) == code]
        if match.empty:
            return
        row = match.iloc[0]
        _render_stock_detail(code, row)

    def _render_stock_detail(code: str, row):
        name = row.get("종목명", "")
        _close = _nz(row.get("종가", 0))
        _entry = _nz(row.get("추천매수가", 0))
        _stop = _nz(row.get("손절가", 0))
        _t1 = _nz(row.get("추천매도가1", 0))
        _t2 = _nz(row.get("추천매도가2", 0))

        with detail_area:
            ui.label(
                f"🔍 {name} ({code}) 상세 분석"
            ).classes("text-lg font-bold text-white mb-3")

            # 목표가 카드
            if _close > 0 and _entry > 0:
                risk = _entry - _stop if _stop > 0 else 1
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    _metric_card(
                        "🔴 손절가", f"{int(_stop):,}",
                        f"{(_stop/_close-1)*100:+.1f}%" if _close > 0 else "",
                        False,
                    )
                    _metric_card("🔵 매수가", f"{int(_entry):,}", "시스템 추천")
                    if _t1 > 0:
                        rr1 = (_t1 - _entry) / risk if risk > 0 else 0
                        _metric_card(
                            "🟢 T1 목표", f"{int(_t1):,}",
                            f"+{(_t1/_close-1)*100:.1f}% (RR {rr1:.1f}:1)",
                        )
                    if _t2 > 0 and _t2 != _t1:
                        rr2 = (_t2 - _entry) / risk if risk > 0 else 0
                        _metric_card(
                            "🟡 T2 목표", f"{int(_t2):,}",
                            f"+{(_t2/_close-1)*100:.1f}% (RR {rr2:.1f}:1)",
                        )

            # ── 캔들차트 (비동기 로드 + 태스크 생명주기 관리) ──
            with ui.card().classes("w-full p-2 bg-[#1a1a2e] mt-2"):
                loading_label = ui.label(
                    "🕯️ 캔들차트 로딩 중..."
                ).classes("text-gray-400")
                chart_holder = ui.column().classes("w-full")

                async def _load_chart():
                    """[v3.3 #3] run.io_bound로 GIL 블로킹 방지"""
                    try:
                        cdata = await run.io_bound(_get_chart_data, code)
                    except asyncio.CancelledError:
                        return  # 태스크 취소 시 조용히 종료

                    loading_label.set_visibility(False)
                    chart_holder.clear()
                    with chart_holder:
                        if cdata is not None and _plot_candle is not None:
                            fig = _plot_candle(
                                cdata, code, name,
                                _entry, _stop, _t1, _t2,
                            )
                            _plotly_dark(fig, 400)
                            ui.plotly(fig).classes("w-full")
                        elif cdata is not None:
                            ui.label(
                                "📉 차트 렌더러 미로드"
                            ).classes("text-yellow-400")
                        else:
                            ui.label(
                                "📉 차트 데이터 로드 실패"
                            ).classes("text-yellow-400")

                # [v3.3 #1] ui.timer 제거 → asyncio.create_task 직접 실행
                # [v3.3 #2] 태스크 변수 저장 → detail_area.clear() 시 자동 취소
                _chart_task = asyncio.create_task(_load_chart())

                # 상세 영역이 다시 clear()될 때 유령 태스크 방지
                def _cancel_on_clear():
                    if _chart_task and not _chart_task.done():
                        _chart_task.cancel()

                detail_area.on("clear", _cancel_on_clear)

            # ── 레이더 + 워터폴 ──
            with ui.row().classes("w-full gap-4 flex-wrap mt-4"):
                with ui.card().classes(
                    "flex-1 min-w-[280px] p-2 bg-[#1a1a2e]"
                ):
                    try:
                        if plot_radar_chart:
                            fig_r = plot_radar_chart(row)
                            if fig_r:
                                _plotly_dark(fig_r, 300)
                                ui.plotly(fig_r).classes("w-full")
                    except Exception:
                        ui.label("레이더 차트 오류").classes("text-gray-500")

                with ui.card().classes(
                    "flex-1 min-w-[280px] p-2 bg-[#1a1a2e]"
                ):
                    try:
                        if plot_score_waterfall:
                            fig_w = plot_score_waterfall(row)
                            if fig_w:
                                _plotly_dark(fig_w, 300)
                                ui.plotly(fig_w).classes("w-full")
                    except Exception:
                        ui.label("워터폴 차트 오류").classes("text-gray-500")

            # ROUTE 배지
            rv = str(row.get("ROUTE", "NEUTRAL"))
            rc = {
                "ATTACK": "#EF4444", "ARMED": "#F59E0B",
                "WAIT": "#3B82F6", "NEUTRAL": "#6B7280",
            }.get(rv, "#6B7280")
            ui.badge(rv, color=rc).classes("mt-2")

    # ── 이벤트 바인딩 ──
    for widget in [view_mode, route_filter, sort_mode]:
        widget.on("update:model-value", lambda _: _build_view())

    _build_view()
