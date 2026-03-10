# -*- coding: utf-8 -*-
"""
tab_perf.py — 📈 시스템 성과 추세 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
rank_validation_summary_*.csv 파일 glob → 병합 → Plotly 이중축 차트
"""
import glob
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from nicegui import ui

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

_logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _now_kst():
    return datetime.now(KST)


def _load_history() -> pd.DataFrame:
    """rank_validation_summary_*.csv 파일 병합"""
    pattern = os.path.join(DATA_DIR, "rank_validation_summary_*.csv")
    all_files = sorted(glob.glob(pattern))

    dfs = []
    for f in all_files:
        try:
            base = os.path.basename(f)
            ds = base.replace("rank_validation_summary_", "").replace(".csv", "")
            d = pd.read_csv(f)
            if "latest" in ds:
                d['Date'] = pd.to_datetime(_now_kst().strftime("%Y-%m-%d"))
            else:
                d['Date'] = pd.to_datetime(ds, format="%Y%m%d")
            dfs.append(d)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).sort_values('Date')


def render_tab_perf():
    """Tab 7: 시스템 성과 추세"""
    ui.label("📈 시스템 성과 추세").classes("text-2xl font-bold mb-4 text-white")

    history = _load_history()

    if history.empty:
        ui.label("축적된 성과 데이터가 부족합니다.").classes("text-gray-400 p-8")
        return

    col_win, col_ret = 'WIN_RATE_%', 'AVG_RET_%'
    if col_win not in history.columns or col_ret not in history.columns:
        ui.label("필요 컬럼 없음").classes("text-gray-400")
        return

    # ── 필터 ──
    with ui.row().classes("w-full gap-4 flex-wrap mb-4"):
        methods = sorted(history['METHOD'].unique()) if 'METHOD' in history.columns else []
        def_m = 'RANK_SCORE' if 'RANK_SCORE' in methods else (methods[0] if methods else None)
        sel_m = ui.select(methods, value=def_m, label="Method").classes("min-w-[150px]") if methods else None

        topks = sorted(history['TOPK'].unique().tolist()) if 'TOPK' in history.columns else []
        def_k = 5 if 5 in topks else (topks[0] if topks else None)
        sel_k = ui.select([str(k) for k in topks], value=str(def_k), label="Top K").classes("min-w-[100px]") if topks else None

        holds = sorted(history['H(영업일)'].unique().tolist()) if 'H(영업일)' in history.columns else []
        def_h = 5 if 5 in holds else (holds[0] if holds else None)
        sel_h = ui.select([str(h) for h in holds], value=str(def_h), label="보유기간").classes("min-w-[100px]") if holds else None

    chart_area = ui.column().classes("w-full")

    def _build_chart():
        chart_area.clear()
        cdf = history.copy()
        if sel_m and sel_m.value: cdf = cdf[cdf['METHOD'] == sel_m.value]
        if sel_k and sel_k.value: cdf = cdf[cdf['TOPK'] == int(sel_k.value)]
        if sel_h and sel_h.value: cdf = cdf[cdf['H(영업일)'] == int(sel_h.value)]
        cdf = cdf.sort_values('Date').tail(30)

        # Timestamp → 문자열 (NiceGUI orjson 직렬화 호환)
        if 'Date' in cdf.columns:
            cdf['Date'] = cdf['Date'].apply(
                lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else str(x)
            )

        with chart_area:
            if cdf.empty:
                ui.label("조건 맞는 데이터 없음").classes("text-gray-400")
                return

            if PLOTLY_OK:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=cdf['Date'], y=cdf[col_win], name="승률(%)", marker_color='#FFA726', opacity=0.6), secondary_y=False)
                fig.add_trace(go.Scatter(x=cdf['Date'], y=cdf[col_ret], name="수익률(%)", mode='lines+markers', line=dict(color='#29B6F6', width=3)), secondary_y=True)
                fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', hovermode="x unified", legend=dict(orientation="h", y=1.1))
                fig.update_yaxes(title_text="승률(%)", range=[0, 100], secondary_y=False)
                fig.update_yaxes(title_text="수익률(%)", secondary_y=True)
                ui.plotly(fig).classes("w-full")

            # 요약 메트릭
            with ui.row().classes("w-full gap-4 mt-2"):
                with ui.card().classes("p-4 bg-[#1a1a2e] border border-amber-700/40 rounded-xl"):
                    ui.label("평균 승률").classes("text-sm text-amber-400")
                    ui.label(f"{cdf[col_win].mean():.1f}%").classes("text-2xl font-bold text-amber-300")
                with ui.card().classes("p-4 bg-[#1a1a2e] border border-blue-700/40 rounded-xl"):
                    ui.label("평균 수익률").classes("text-sm text-blue-400")
                    ui.label(f"{cdf[col_ret].mean():.2f}%").classes("text-2xl font-bold text-blue-300")

    for w in [sel_m, sel_k, sel_h]:
        if w: w.on("update:model-value", lambda _: _build_chart())
    _build_chart()

    # ── Research Workbench 확장 ──
    try:
        from research_tab import render_research_tab
        ui.separator().classes("my-6")
        render_research_tab(data_dir=DATA_DIR)
    except Exception as _rt_err:
        ui.label(f"⚠️ Research 탭 로드 실패: {_rt_err}").classes("text-gray-400")
