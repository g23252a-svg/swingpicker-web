# -*- coding: utf-8 -*-
"""
tab_perf.py — 📈 시스템 성과 추세 (NiceGUI)
═══════════════════════════════════════════════════
[v1.0] Streamlit → NiceGUI 이식

핵심:
  - rank_validation_summary_*.csv 파일 glob → 병합
  - I/O 병목 → run.io_bound 비동기 처리
  - Plotly 이중축 (승률 막대 + 수익률 선)
  - 필터: Method / Top K / 보유 기간
"""
import glob
import logging
import os
from typing import Optional

import pandas as pd
from nicegui import ui, run

_logger = logging.getLogger(__name__)

# CSV 디렉토리 (Railway 기준)
DATA_DIR = os.environ.get("LDY_DATA_DIR", "data")


# ═══════════════════════════════════════════════════
#  데이터 로딩 (I/O bound)
# ═══════════════════════════════════════════════════

def _load_performance_history() -> pd.DataFrame:
    """rank_validation_summary_*.csv 파일들을 병합 (동기 — run.io_bound용)"""
    pattern = os.path.join(DATA_DIR, "rank_validation_summary_*.csv")
    files = sorted(glob.glob(pattern))

    dfs = []
    for f in files:
        try:
            base = os.path.basename(f)
            if "latest" in base:
                continue

            date_str = base.replace("rank_validation_summary_", "").replace(".csv", "")
            df = pd.read_csv(f)

            try:
                df["Date"] = pd.to_datetime(date_str, format="%Y%m%d")
            except (ValueError, TypeError):
                df["Date"] = date_str

            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if "Date" in combined.columns:
        combined = combined.sort_values("Date")
    return combined


# ═══════════════════════════════════════════════════
#  메인 렌더
# ═══════════════════════════════════════════════════

def render_tab_perf():
    """Tab 7: 시스템 성과 추세"""
    ui.label("📈 시스템 성과 추세 (System Performance Lab)").classes("text-2xl font-bold mb-2")
    ui.label(
        "과거 추천 종목들의 검증 데이터를 기반으로 시스템의 승률 변화를 추적합니다."
    ).classes("text-sm text-gray-500 mb-4")

    # 결과 영역
    chart_container = ui.column().classes("w-full")
    load_btn = ui.button("📊 성과 데이터 로드", color="primary")

    async def on_load():
        load_btn.disable()
        load_btn.text = "⏳ 로딩 중..."

        try:
            history_df = await run.io_bound(_load_performance_history)

            chart_container.clear()
            with chart_container:
                if history_df.empty:
                    ui.label("📉 아직 축적된 성과 검증 데이터가 충분하지 않습니다.").classes(
                        "text-gray-500 mt-4"
                    )
                    return

                _render_perf_dashboard(history_df)

        except Exception as e:
            _logger.error(f"성과 데이터 로드 실패: {e}")
            ui.notify(f"로드 실패: {e}", type="negative")
        finally:
            load_btn.enable()
            load_btn.text = "📊 성과 데이터 로드"

    load_btn.on_click(on_load)


def _render_perf_dashboard(history_df: pd.DataFrame):
    """성과 대시보드 전체 렌더"""
    col_win = "WIN_RATE_%"
    col_ret = "AVG_RET_%"

    if col_win not in history_df.columns or col_ret not in history_df.columns:
        ui.label(f"데이터 컬럼을 찾을 수 없습니다. (필요: {col_win}, {col_ret})").classes(
            "text-red-500"
        )
        ui.label(f"현재 컬럼: {history_df.columns.tolist()}").classes("text-xs text-gray-400")
        return

    # ── 필터 UI ──
    ui.label("🔍 지표 상세 필터").classes("text-lg font-semibold mt-4 mb-2")

    methods = sorted(history_df["METHOD"].unique()) if "METHOD" in history_df.columns else []
    topks = sorted(history_df["TOPK"].unique()) if "TOPK" in history_df.columns else []
    holds = sorted(history_df["H(영업일)"].unique()) if "H(영업일)" in history_df.columns else []

    with ui.row().classes("gap-4 items-end flex-wrap"):
        sel_method = ui.select(
            "스코어링 기준",
            options=methods,
            value="RANK_SCORE" if "RANK_SCORE" in methods else (methods[0] if methods else None),
        ).classes("w-48")

        sel_k = ui.select(
            "Top K",
            options=topks,
            value=5 if 5 in topks else (topks[0] if topks else None),
        ).classes("w-32")

        sel_h = ui.select(
            "보유 기간 (일)",
            options=holds,
            value=5 if 5 in holds else (holds[0] if holds else None),
        ).classes("w-32")

    # ── 차트 + 통계 영역 (필터 변경 시 갱신) ──
    result_area = ui.column().classes("w-full mt-4")

    def refresh_chart():
        """필터 적용 후 차트 재렌더"""
        df = history_df.copy()
        if sel_method.value:
            df = df[df["METHOD"] == sel_method.value]
        if sel_k.value is not None:
            df = df[df["TOPK"] == sel_k.value]
        if sel_h.value is not None:
            df = df[df["H(영업일)"] == sel_h.value]

        result_area.clear()
        with result_area:
            if df.empty:
                ui.label("선택한 조건에 맞는 데이터가 없습니다.").classes("text-amber-600 mt-2")
                return

            df = df.sort_values("Date").tail(30)
            _render_chart(df, col_win, col_ret, sel_method.value, sel_k.value, sel_h.value)
            _render_stats(df, col_win, col_ret)
            _render_detail_table(df, col_win, col_ret)

    # 필터 변경 시 자동 갱신
    sel_method.on_value_change(lambda _: refresh_chart())
    sel_k.on_value_change(lambda _: refresh_chart())
    sel_h.on_value_change(lambda _: refresh_chart())

    # 초기 렌더
    refresh_chart()


# ═══════════════════════════════════════════════════
#  차트 / 통계 / 테이블
# ═══════════════════════════════════════════════════

def _render_chart(df: pd.DataFrame, col_win: str, col_ret: str,
                  method: str, topk, hold):
    """Plotly 이중축 차트 (승률 막대 + 수익률 선)"""
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        ui.label("Plotly 미설치 — 차트 생략").classes("text-gray-400")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 막대: 승률
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df[col_win],
            name="승률(%)",
            marker_color="#FFA726",
            opacity=0.6,
            text=df[col_win].apply(lambda x: f"{x:.1f}%"),
            textposition="auto",
        ),
        secondary_y=False,
    )

    # 선: 평균 수익률
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df[col_ret],
            name="평균수익률(%)",
            mode="lines+markers+text",
            line=dict(color="#29B6F6", width=3),
            text=df[col_ret].apply(lambda x: f"{x:.1f}%"),
            textposition="top center",
        ),
        secondary_y=True,
    )

    title_text = f"📊 {method} (Top {topk}, {hold}일 보유) 성과 추이"
    fig.update_layout(
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=16)),
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="승률 (%)", range=[0, 100], secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text="수익률 (%)", secondary_y=True, showgrid=False)

    ui.plotly(fig).classes("w-full")


def _render_stats(df: pd.DataFrame, col_win: str, col_ret: str):
    """요약 통계 메트릭"""
    avg_win = df[col_win].mean()
    avg_ret = df[col_ret].mean()

    with ui.row().classes("gap-4 mt-4 flex-wrap"):
        with ui.card().classes("p-4 bg-amber-50 min-w-[180px]"):
            ui.label("기간 평균 승률").classes("text-sm text-amber-700")
            ui.label(f"{avg_win:.1f}%").classes("text-2xl font-bold text-amber-800")

        with ui.card().classes("p-4 bg-blue-50 min-w-[180px]"):
            ui.label("기간 평균 수익률").classes("text-sm text-blue-700")
            ui.label(f"{avg_ret:.2f}%").classes("text-2xl font-bold text-blue-800")

        with ui.card().classes("p-4 bg-gray-50 min-w-[180px]"):
            ui.label("조회 건수").classes("text-sm text-gray-600")
            ui.label(f"최근 {len(df)}건").classes("text-2xl font-bold text-gray-700")


def _render_detail_table(df: pd.DataFrame, col_win: str, col_ret: str):
    """상세 데이터 테이블"""
    with ui.expansion("📄 상세 데이터 보기").classes("w-full mt-4"):
        disp_cols = ["Date", "METHOD", "TOPK", "H(영업일)", col_win, col_ret, "TOTAL_N"]
        disp_df = df[[c for c in disp_cols if c in df.columns]].copy()

        if "Date" in disp_df.columns:
            disp_df["Date"] = disp_df["Date"].apply(
                lambda x: x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp) else str(x)
            )

        disp_df = disp_df.sort_values("Date", ascending=False)

        columns = [
            {"name": c, "label": c, "field": c, "align": "right" if "%" in c else "left", "sortable": True}
            for c in disp_df.columns
        ]
        rows = disp_df.to_dict("records")

        ui.table(columns=columns, rows=rows, row_key="Date").classes("w-full").props("dense flat")
