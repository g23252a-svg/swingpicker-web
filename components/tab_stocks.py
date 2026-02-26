# components/tab_stocks.py — Tab 2: 종목 분석
"""
AgGrid 테이블 + 칸반 보드 + 비동기 상세 다이얼로그

UI만 담당. 데이터는 StockService / global_data에서 조회.
"""
import math
import os
from nicegui import ui, run

import pandas as pd
from shared_utils import nz_num, safe_float
from state import global_data
from data.services.stock_service import StockService
from chart_components import plot_radar_chart, plot_score_waterfall

_stock_svc = StockService()

# 캔들차트 — chart_components에서 임포트 (순환 참조 방지)
try:
    from chart_components import plot_candle_chart as _plot_candle
except ImportError:
    _plot_candle = None


# ═══════════════════════════════════════════
#  공통 헬퍼
# ═══════════════════════════════════════════

def _section_title(text: str):
    ui.label(text).classes(
        "text-lg font-bold text-white mt-6 mb-2 border-b border-gray-700 pb-2"
    )


def _metric_card(title: str, value: str, delta: str = "", positive: bool = True):
    with ui.card().classes("p-4 min-w-[140px] bg-[#1a1a2e] border border-gray-700 rounded-xl"):
        ui.label(title).classes("text-xs text-gray-400 uppercase tracking-wide")
        ui.label(str(value)).classes("text-xl font-bold text-white mt-1")
        if delta:
            color = "text-green-400" if positive else "text-red-400"
            ui.label(str(delta)).classes(f"text-sm {color} mt-0.5")


def _plotly_dark(fig, height: int = 300):
    if fig:
        fig.update_layout(
            height=height, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
            margin=dict(t=30, b=10, l=10, r=10),
        )


def _price_bar_html(stop, entry, close, t1, t2=0):
    """가격 위치 바 — 손절~매수~현재~T1~T2"""
    points = [("손절", stop, "#EF4444"), ("매수", entry, "#3B82F6"), ("현재", close, "#FFFFFF")]
    if t1 > 0:
        points.append(("T1", t1, "#10B981"))
    if t2 > 0 and t2 != t1:
        points.append(("T2", t2, "#EAB308"))
    points.sort(key=lambda x: x[1])
    p_min, p_max = points[0][1] * 0.98, points[-1][1] * 1.02
    rng = p_max - p_min
    if rng <= 0:
        return ""
    html = '<div style="position:relative;height:55px;background:linear-gradient(90deg,rgba(239,68,68,0.15) 0%,rgba(16,185,129,0.15) 100%);border-radius:10px;margin:8px 0 20px 0;">'
    for label, price, color in points:
        pct = max(3, min((price - p_min) / rng * 100, 97))
        is_cur = label == "현재"
        sz = "14px" if is_cur else "10px"
        bdr = "2px solid #FFF" if is_cur else "none"
        fw = "bold" if is_cur else "normal"
        html += (
            f'<div style="position:absolute;left:{pct}%;top:50%;transform:translate(-50%,-50%);'
            f'z-index:{"10" if is_cur else "5"};text-align:center;">'
            f'<div style="width:{sz};height:{sz};background:{color};border-radius:50%;border:{bdr};margin:0 auto;"></div>'
            f'<div style="font-size:10px;color:{color};white-space:nowrap;margin-top:3px;font-weight:{fw};">'
            f'{label}<br>{int(price):,}</div></div>'
        )
    html += "</div>"
    return html


# ═══════════════════════════════════════════
#  메인 렌더
# ═══════════════════════════════════════════

def render_tab2_stocks(auth: str):
    """Tab 2: 종목 분석 — 테이블/칸반 토글 + 행 클릭 상세"""
    df = global_data.scored

    _section_title("🎯 AI & Quant 추천 종목")

    # ── 컨트롤 바 ──
    with ui.row().classes("w-full gap-4 items-center flex-wrap mb-4"):
        view_mode = ui.toggle(["📋 테이블", "🃏 칸반"], value="📋 테이블")
        route_filter = ui.select(
            ["전체", "ATTACK", "ARMED", "WAIT", "NEUTRAL"],
            value="전체", label="상태",
        ).classes("min-w-[120px]")
        sort_mode = ui.toggle(["🔢 점수순", "🚦 상태순"], value="🔢 점수순")

        # CSV 다운로드 (Prime/Admin)
        if auth in ("prime", "admin"):
            async def _download_csv():
                csv_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "data", "recommend_latest.csv"
                )
                if os.path.exists(csv_path):
                    ui.download(csv_path, f"ldy_recommend_{global_data.data_ts}.csv")
                else:
                    ui.notify("CSV 파일 없음", type="warning")
            ui.button("📥 CSV", on_click=_download_csv).props("flat dense").classes("text-green-400 ml-auto")

    content_area = ui.column().classes("w-full")
    detail_area = ui.column().classes("w-full mt-4")

    def _get_filtered() -> pd.DataFrame:
        fdf = df.copy()
        if route_filter.value != "전체" and "ROUTE" in fdf.columns:
            fdf = fdf[fdf["ROUTE"].astype(str).str.contains(route_filter.value, na=False)]
        if sort_mode.value == "🔢 점수순" and "DISPLAY_SCORE" in fdf.columns:
            fdf = fdf.sort_values("DISPLAY_SCORE", ascending=False)
        # 접근 제한
        limits = {"guest": 3, "free": 5, "pro": 20}
        limit = limits.get(auth, 50)
        return fdf.head(limit)

    def _rebuild():
        content_area.clear()
        show = _get_filtered()
        with content_area:
            if view_mode.value == "🃏 칸반":
                _render_kanban(show, detail_area)
            else:
                _render_table(show, detail_area)

    # 이벤트 바인딩
    for widget in [view_mode, route_filter, sort_mode]:
        widget.on("update:model-value", lambda _: _rebuild())

    _rebuild()


# ═══════════════════════════════════════════
#  테이블 뷰 (AgGrid)
# ═══════════════════════════════════════════

def _render_table(show: pd.DataFrame, detail_area: ui.column):
    if show.empty:
        ui.label("표시할 종목 없음").classes("text-gray-400")
        return

    columns = [
        {"name": "route", "label": "상태", "field": "route", "align": "center"},
        {"name": "name", "label": "종목명", "field": "name", "align": "left"},
        {"name": "score", "label": "점수", "field": "score", "align": "center", "sortable": True},
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
            "score": f'{safe_float(r.get("DISPLAY_SCORE", 0)):.0f}',
            "close": f'{int(nz_num(r.get("종가", 0))):,}',
            "buy": f'{int(nz_num(r.get("추천매수가", 0))):,}',
            "stop": f'{int(nz_num(r.get("손절가", 0))):,}',
            "t1": f'{int(nz_num(r.get("추천매도가1", 0))):,}',
            "sector": str(r.get("업종", "—")),
        })

    tbl = ui.table(
        columns=columns, rows=rows, row_key="code",
        selection="single", pagination={"rowsPerPage": 15},
    ).classes("w-full").props("dense dark flat bordered")

    tbl.on("selection", lambda e: _on_row_select(e, show, detail_area))


def _on_row_select(event, show: pd.DataFrame, detail_area: ui.column):
    """행 선택 → 상세 다이얼로그 (async)"""
    sel = event.args.get("rows", [])
    if not sel:
        return
    code = sel[0].get("code", "")
    row = show[show["종목코드"].astype(str).str.zfill(6) == code]
    if row.empty:
        return
    _open_detail_dialog(row.iloc[0])


# ═══════════════════════════════════════════
#  칸반 뷰
# ═══════════════════════════════════════════

def _render_kanban(show: pd.DataFrame, detail_area: ui.column):
    if show.empty:
        ui.label("표시할 종목 없음").classes("text-gray-400")
        return

    route_col = show["ROUTE"].astype(str) if "ROUTE" in show.columns else pd.Series(dtype=str)
    df_atk = show[route_col.str.contains("ATTACK", case=False, na=False)]
    df_arm = show[route_col.str.contains("ARMED", case=False, na=False)]
    used = df_atk.index.union(df_arm.index)
    df_watch = show[~show.index.isin(used)]

    with ui.row().classes("w-full gap-4 flex-wrap items-start overflow-x-auto"):
        _kanban_col("🚀 ATTACK", df_atk, "#EF4444")
        _kanban_col("🔫 ARMED", df_arm, "#F59E0B")
        _kanban_col("👀 WATCH", df_watch, "#3B82F6")


def _kanban_col(title: str, sub_df: pd.DataFrame, color: str):
    with ui.column().classes("min-w-[280px] flex-1"):
        ui.label(f"{title} ({len(sub_df)})").classes("text-white font-bold mb-2").style(
            f"border-bottom: 2px solid {color}"
        )
        if sub_df.empty:
            ui.label("비어 있음").classes("text-gray-500 text-sm")
            return
        for _, r in sub_df.iterrows():
            _kanban_card(r)


def _kanban_card(r: pd.Series):
    score = safe_float(r.get("DISPLAY_SCORE", 0))
    buy = int(nz_num(r.get("추천매수가", 0)))
    stop = int(nz_num(r.get("손절가", 0)))
    t1 = int(nz_num(r.get("추천매도가1", 0)))
    sc = "#10B981" if score >= 80 else "#3B82F6" if score >= 60 else "#94A3B8"

    with ui.card().classes(
        "p-3 mb-2 cursor-pointer bg-[rgba(255,255,255,0.05)] "
        "border border-[rgba(255,255,255,0.1)] rounded-xl "
        "hover:bg-[rgba(255,255,255,0.08)] transition-all"
    ).on("click", lambda _, row=r: _open_detail_dialog(row)):
        with ui.row().classes("justify-between items-center"):
            ui.label(str(r.get("종목명", ""))).classes("text-white font-bold text-sm")
            ui.badge(f"{score:.0f}", color=sc).classes("text-xs")
        if buy > 0:
            ui.label(f"🎯 {buy:,}  🛡️ {stop:,}  🟢 {t1:,}").classes("text-xs text-gray-400 mt-1")
        rr = safe_float(r.get("RR1", 0))
        if rr > 0:
            rc = "#10B981" if rr >= 2 else "#F59E0B" if rr >= 1 else "#EF4444"
            ui.label(f"R:R {rr:.1f}").classes("text-xs mt-1").style(f"color:{rc}")


# ═══════════════════════════════════════════
#  상세 분석 다이얼로그 (Async — 차트 로딩 시 UI 안 멈춤)
# ═══════════════════════════════════════════

def _open_detail_dialog(row: pd.Series):
    """종목 상세 다이얼로그 — 차트는 비동기 로드"""
    code = str(row.get("종목코드", "")).zfill(6)
    name = str(row.get("종목명", ""))

    _close = float(nz_num(row.get("종가", 0)))
    _entry = float(nz_num(row.get("추천매수가", 0)))
    _stop = float(nz_num(row.get("손절가", 0)))
    _t1 = float(nz_num(row.get("추천매도가1", 0)))
    _t2 = float(nz_num(row.get("추천매도가2", 0)))
    _atr = float(nz_num(row.get("TARGET_ATR", 0)))

    with ui.dialog() as dialog, ui.card().classes(
        "w-full max-w-[960px] max-h-[85vh] overflow-y-auto bg-[#0d0d1a] p-6"
    ):
        # 헤더
        with ui.row().classes("w-full justify-between items-center"):
            ui.label(f"🔍 {name} ({code})").classes("text-xl font-bold text-white")
            rv = str(row.get("ROUTE", "NEUTRAL"))
            rc = {"ATTACK": "#EF4444", "ARMED": "#F59E0B", "WAIT": "#3B82F6"}.get(rv, "#6B7280")
            ui.badge(rv, color=rc).classes("text-sm")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense").classes("text-white")

        ui.separator().classes("my-2")

        # ── 목표가 카드 ──
        if _close > 0 and _entry > 0:
            risk = _entry - _stop if _stop > 0 else 1
            with ui.row().classes("w-full gap-3 flex-wrap"):
                _metric_card("🔴 손절가", f"{int(_stop):,}",
                             f"{(_stop / _close - 1) * 100:+.1f}%" if _close else "", False)
                _metric_card("🔵 매수가", f"{int(_entry):,}", "시스템 추천")
                if _t1 > 0:
                    rr1 = (_t1 - _entry) / risk if risk > 0 else 0
                    _metric_card("🟢 T1 목표", f"{int(_t1):,}",
                                 f"+{(_t1 / _close - 1) * 100:.1f}% (RR {rr1:.1f}:1)")
                if _t2 > 0 and _t2 != _t1:
                    rr2 = (_t2 - _entry) / risk if risk > 0 else 0
                    _metric_card("🟡 T2 목표", f"{int(_t2):,}",
                                 f"+{(_t2 / _close - 1) * 100:.1f}% (RR {rr2:.1f}:1)")

        # ── 가격 바 ──
        if _close > 0 and _stop > 0 and _t1 > 0:
            ui.html(_price_bar_html(_stop, _entry, _close, _t1, _t2))

        # ── 캔들차트 (비동기 로드) ──
        chart_holder = ui.column().classes("w-full")
        with chart_holder:
            ui.label("🕯️ 캔들차트 로딩 중...").classes("text-gray-400 py-4")

        async def _load_chart():
            # FDR 호출을 스레드에서 실행 → 이벤트 루프 블로킹 없음
            cdata = await run.io_bound(_stock_svc.get_chart_data, code)
            chart_holder.clear()
            with chart_holder:
                if cdata is not None and _plot_candle is not None:
                    fig = _plot_candle(cdata, code, name, _entry, _stop, _t1, _t2)
                    ui.plotly(fig).classes("w-full")
                elif cdata is not None:
                    ui.label("📉 차트 렌더러 미로드 (plot_candle_chart)").classes("text-yellow-400")
                else:
                    ui.label("📉 차트 데이터 로드 실패").classes("text-yellow-400")

        ui.timer(0.1, _load_chart, once=True)

        # ── 레이더 + 워터폴 ──
        with ui.row().classes("w-full gap-4 flex-wrap mt-4"):
            with ui.card().classes("flex-1 min-w-[280px] p-2 bg-[#1a1a2e]"):
                try:
                    fig_r = plot_radar_chart(row)
                    if fig_r:
                        _plotly_dark(fig_r, 300)
                        ui.plotly(fig_r).classes("w-full")
                except Exception:
                    ui.label("레이더 차트 오류").classes("text-gray-500")
            with ui.card().classes("flex-1 min-w-[280px] p-2 bg-[#1a1a2e]"):
                try:
                    fig_w = plot_score_waterfall(row)
                    if fig_w:
                        _plotly_dark(fig_w, 300)
                        ui.plotly(fig_w).classes("w-full")
                except Exception:
                    ui.label("워터폴 차트 오류").classes("text-gray-500")

    dialog.open()
