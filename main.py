# -*- coding: utf-8 -*-
"""
LDY Pro Trader — NiceGUI Edition (Phase 0 PoC)
────────────────────────────────────────────────
Streamlit → NiceGUI 마이그레이션 검증용 최소 작동 버전

기존 모듈 100% 재사용:
  - scoring_engine.py, ml_engine.py, db_utils.py
  - chart_components.py, shared_utils.py, indicators.py
  - collector.py (GitHub Actions — 변경 없음)
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache

from nicegui import ui, app

# ─── 기존 모듈 재사용 (그대로 import) ───
from shared_utils import nz_num, safe_float
from chart_components import (
    plot_fear_greed_gauge, plot_sector_treemap,
    plot_sector_momentum_bar, plot_radar_chart,
    plot_score_waterfall, plot_ai_gauge_chart,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ldy-nicegui")

# ═══════════════════════════════════════════
#  1. 설정
# ═══════════════════════════════════════════
APP_VERSION = "1.0.0-nicegui"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RECOMMEND_PATH = os.path.join(DATA_DIR, "recommend_latest.csv")

# ═══════════════════════════════════════════
#  2. 데이터 로딩 (앱 시작 시 1번, 이후 캐시)
# ═══════════════════════════════════════════
class DataStore:
    """
    앱 전역 데이터 저장소.
    Streamlit은 매 인터랙션마다 다시 로드하지만,
    NiceGUI에서는 한 번 로드 후 메모리에 유지.
    refresh()로 수동/주기적 갱신 가능.
    """
    def __init__(self):
        self.scored: pd.DataFrame = pd.DataFrame()
        self.data_ts: str = ""
        self.loaded: bool = False

    def refresh(self):
        """CSV 다시 로드"""
        if not os.path.exists(RECOMMEND_PATH):
            logger.warning(f"CSV not found: {RECOMMEND_PATH}")
            return

        try:
            df = pd.read_csv(RECOMMEND_PATH, dtype={"종목코드": str})
            # 숫자 컬럼 변환
            num_cols = [
                "FINAL_SCORE", "DISPLAY_SCORE", "STRUCT_SCORE",
                "TIMING_SCORE", "AI_SCORE", "TOTAL_SCORE",
                "거래대금(억원)", "RSI14", "종가",
                "추천매수가", "손절가", "추천매도가1", "추천매도가2",
            ]
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            # DISPLAY_SCORE 통합
            if "DISPLAY_SCORE" in df.columns:
                primary = "DISPLAY_SCORE"
            elif "FINAL_SCORE" in df.columns:
                primary = "FINAL_SCORE"
            else:
                primary = "TOTAL_SCORE"

            if primary in df.columns:
                df["DISPLAY_SCORE"] = df[primary]

            # 날짜 추론
            ts_col = next((c for c in ["trade_date", "DATA_DATE"] if c in df.columns), None)
            if ts_col:
                self.data_ts = str(df[ts_col].iloc[0])
            else:
                self.data_ts = datetime.now().strftime("%Y-%m-%d")

            self.scored = df
            self.loaded = True
            logger.info(f"✅ 데이터 로드 완료: {len(df)}종목, 기준일 {self.data_ts}")

        except Exception as e:
            logger.exception(f"데이터 로드 실패: {e}")

# 전역 싱글톤
store = DataStore()


# ═══════════════════════════════════════════
#  3. 공통 UI 컴포넌트
# ═══════════════════════════════════════════
def metric_card(title: str, value: str, delta: str = "", positive: bool = True):
    """st.metric() 대체 카드 컴포넌트"""
    with ui.card().classes(
        "p-4 min-w-[140px] bg-[#1a1a2e] border border-gray-700 rounded-xl"
    ):
        ui.label(title).classes("text-xs text-gray-400 uppercase tracking-wide")
        ui.label(value).classes("text-xl font-bold text-white mt-1")
        if delta:
            color = "text-green-400" if positive else "text-red-400"
            ui.label(delta).classes(f"text-sm {color} mt-0.5")


def section_title(text: str):
    """섹션 구분 타이틀"""
    ui.label(text).classes(
        "text-lg font-bold text-white mt-6 mb-2 "
        "border-b border-gray-700 pb-2"
    )


# ═══════════════════════════════════════════
#  4. 페이지 구성
# ═══════════════════════════════════════════
@ui.page("/")
def index():
    # ─── 다크 테마 CSS ───
    ui.add_head_html("""
    <style>
        body { background: #0d0d1a !important; }
        .nicegui-content { max-width: 1400px; margin: 0 auto; }
        .q-tab-panel { padding: 8px 0 !important; }
        .q-card { background: #1a1a2e !important; }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    """)

    # ─── 데이터 로드 ───
    if not store.loaded:
        store.refresh()

    df = store.scored

    # ─── Hero Banner ───
    with ui.row().classes(
        "w-full items-center justify-between px-4 py-3 rounded-xl mb-4"
        " bg-gradient-to-r from-[#1a1a2e] via-[#16213e] to-[#0f3460]"
    ):
        with ui.column().classes("gap-0"):
            ui.label("LDY Pro Trader").classes(
                "text-2xl font-bold text-transparent bg-clip-text "
                "bg-gradient-to-r from-blue-400 to-purple-400"
            ).style("font-family: Outfit, sans-serif")
            ui.label(f"v{APP_VERSION} · NiceGUI Edition").classes(
                "text-xs text-gray-400"
            )
        # 새로고침 버튼
        ui.button("🔄", on_click=lambda: _refresh_data()).props(
            "flat round dense"
        ).classes("text-white")

    if df.empty:
        ui.label("⚠️ 데이터가 없습니다. data/recommend_latest.csv를 확인해주세요.").classes(
            "text-yellow-400 text-lg p-8"
        )
        return

    # ─── 탭 구조 ───
    with ui.tabs().classes("w-full text-white") as tabs:
        tab_market = ui.tab("📊 시장")
        tab_stocks = ui.tab("🔭 종목 분석")
        tab_portfolio = ui.tab("💼 내 자산")

    with ui.tab_panels(tabs, value=tab_market).classes("w-full"):
        # ── Tab 1: 시장 ──
        with ui.tab_panel(tab_market):
            _render_market_tab(df)

        # ── Tab 2: 종목 분석 ──
        with ui.tab_panel(tab_stocks):
            _render_stocks_tab(df)

        # ── Tab 3: 내 자산 (Phase 2) ──
        with ui.tab_panel(tab_portfolio):
            ui.label("🔒 Phase 2에서 구현 예정").classes("text-gray-400 p-8")

    # ─── 푸터 ───
    ui.label(
        f"📅 데이터 기준: {store.data_ts} · "
        "⚠️ 투자 판단은 본인 책임입니다."
    ).classes("text-xs text-gray-500 text-center mt-8 mb-4")


# ═══════════════════════════════════════════
#  5. Tab1: 시장 (Market)
# ═══════════════════════════════════════════
def _render_market_tab(df: pd.DataFrame):
    """시장 탭 렌더링 — dashboard.py Tab1 변환"""

    # ── 공포/탐욕 지수 계산 ──
    fg_score = 50.0
    if "DISPLAY_SCORE" in df.columns:
        avg_score = df["DISPLAY_SCORE"].mean()
        fg_score = min(max(avg_score, 0), 100)

    # ── 시장 상태 메트릭 ──
    section_title("📡 시장 현황")
    with ui.row().classes("w-full gap-4 flex-wrap"):
        # 공포/탐욕 요약
        fg_label = (
            "🟢 극단적 탐욕" if fg_score >= 80
            else "🟢 탐욕" if fg_score >= 60
            else "🟡 중립" if fg_score >= 40
            else "🔴 공포" if fg_score >= 20
            else "🔴 극단적 공포"
        )
        metric_card("시장 심리", fg_label, f"지수: {fg_score:.0f}/100", fg_score >= 50)

        # 상위 종목 평균 수익률
        if "ret_1d_%" in df.columns:
            avg_ret = df.head(20)["ret_1d_%"].mean()
            metric_card(
                "Top20 평균 수익률",
                f"{avg_ret:+.2f}%",
                "전일 대비",
                avg_ret >= 0,
            )

        # 종목 수
        total = len(df)
        armed = len(df[df.get("ROUTE", pd.Series()).str.contains("ARMED|ATTACK", na=False)])
        metric_card("분석 종목", f"{total}개", f"ARMED/ATTACK: {armed}개")

    # ── 공포/탐욕 게이지 + 섹터 트리맵 ──
    section_title("🌡️ 공포/탐욕 & 주도 섹터")
    with ui.row().classes("w-full gap-4 flex-wrap items-start"):
        # 게이지 (좌)
        with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
            fig_gauge = plot_fear_greed_gauge(fg_score)
            if fig_gauge:
                fig_gauge.update_layout(
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    margin=dict(t=30, b=10, l=10, r=10),
                )
                ui.plotly(fig_gauge).classes("w-full")

        # 섹터 트리맵 (우)
        with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
            ui.label("🔥 오늘의 주도 섹터").classes("text-sm font-bold text-white mb-2")
            if "업종" in df.columns:
                fig_map = plot_sector_treemap(df.head(50))
                if fig_map:
                    fig_map.update_layout(
                        height=280,
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=10, b=10, l=10, r=10),
                    )
                    ui.plotly(fig_map).classes("w-full")
                else:
                    ui.label("섹터 데이터 부족").classes("text-gray-500")
            else:
                ui.label("업종 정보 없음").classes("text-gray-500")

    # ── 섹터 모멘텀 ──
    section_title("🚀 섹터 모멘텀 Top 10")
    fig_mom = plot_sector_momentum_bar(df)
    if fig_mom and len(fig_mom.data) > 0:
        fig_mom.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        ui.plotly(fig_mom).classes("w-full")
    else:
        ui.label("모멘텀 데이터 부족").classes("text-gray-500")


# ═══════════════════════════════════════════
#  6. Tab2: 종목 분석 (Stocks)
# ═══════════════════════════════════════════
def _render_stocks_tab(df: pd.DataFrame):
    """종목 분석 탭 — dashboard.py Tab2 핵심 변환"""

    section_title("🎯 AI & Quant 추천 종목")

    if df.empty:
        ui.label("데이터 없음").classes("text-gray-400")
        return

    # ── 필터 옵션 ──
    with ui.row().classes("w-full gap-4 items-center flex-wrap mb-4"):
        # 상태 필터
        route_filter = ui.select(
            ["전체", "ATTACK", "ARMED", "WAIT", "NEUTRAL"],
            value="전체",
            label="상태 필터",
        ).classes("min-w-[120px]")

        # 정렬 기준
        sort_mode = ui.toggle(
            ["🚦 상태순", "🔢 점수순"], value="🔢 점수순"
        )

    # ── 종목 리스트 (테이블) ──
    table_container = ui.column().classes("w-full")
    # ── 종목 상세 분석 영역 ──
    detail_container = ui.column().classes("w-full mt-4")

    def _build_table():
        """필터/정렬 적용 후 테이블 갱신"""
        table_container.clear()
        filtered = df.copy()

        # 필터 적용
        route_val = route_filter.value
        if route_val != "전체" and "ROUTE" in filtered.columns:
            filtered = filtered[
                filtered["ROUTE"].astype(str).str.contains(route_val, na=False)
            ]

        # 정렬
        if sort_mode.value == "🔢 점수순" and "DISPLAY_SCORE" in filtered.columns:
            filtered = filtered.sort_values("DISPLAY_SCORE", ascending=False)

        # 상위 30개만
        show = filtered.head(30)

        # 테이블 데이터 준비
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

        with table_container:
            table = ui.table(
                columns=columns,
                rows=rows,
                row_key="code",
                selection="single",
                pagination={"rowsPerPage": 15},
            ).classes("w-full").props('dense dark flat bordered')

            # 행 선택 시 상세 분석
            table.on("selection", lambda e: _on_stock_select(e, df))

    def _on_stock_select(event, full_df):
        """종목 선택 시 상세 분석 표시 — 이 함수만 실행됨 (전체 재렌더링 없음!)"""
        detail_container.clear()

        selection = event.args.get("rows", [])
        if not selection:
            return

        code = selection[0].get("code", "")
        row = full_df[full_df["종목코드"].astype(str).str.zfill(6) == code]
        if row.empty:
            return
        row = row.iloc[0]

        with detail_container:
            section_title(f"🔍 {row.get('종목명', '')} ({code}) 상세 분석")

            # ── 목표가 카드 ──
            _close = nz_num(row.get("종가", 0))
            _entry = nz_num(row.get("추천매수가", 0))
            _stop = nz_num(row.get("손절가", 0))
            _t1 = nz_num(row.get("추천매도가1", 0))
            _t2 = nz_num(row.get("추천매도가2", 0))

            if _close > 0 and _entry > 0:
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    risk = _entry - _stop if _stop > 0 else 1
                    rr1 = (_t1 - _entry) / risk if risk > 0 and _t1 > 0 else 0

                    metric_card(
                        "🔴 손절가", f"{int(_stop):,}",
                        f"{(_stop / _close - 1) * 100:+.1f}%" if _close > 0 else "",
                        False,
                    )
                    metric_card("🔵 매수가", f"{int(_entry):,}", "시스템 추천")
                    metric_card(
                        "🟢 T1 목표", f"{int(_t1):,}",
                        f"+{(_t1 / _close - 1) * 100:.1f}%  (RR {rr1:.1f}:1)" if _close > 0 and _t1 > 0 else "",
                    )
                    if _t2 > 0 and _t2 != _t1:
                        rr2 = (_t2 - _entry) / risk if risk > 0 else 0
                        metric_card(
                            "🟡 T2 목표", f"{int(_t2):,}",
                            f"+{(_t2 / _close - 1) * 100:.1f}%  (RR {rr2:.1f}:1)" if _close > 0 else "",
                        )

            # ── 가격 바 시각화 ──
            if _close > 0 and _stop > 0 and _t1 > 0:
                _render_price_bar(_stop, _entry, _close, _t1, _t2)

            # ── 레이더 차트 + 워터폴 ──
            with ui.row().classes("w-full gap-4 flex-wrap mt-4"):
                with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
                    try:
                        fig_radar = plot_radar_chart(row)
                        if fig_radar:
                            fig_radar.update_layout(
                                height=300,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="white",
                            )
                            ui.plotly(fig_radar).classes("w-full")
                    except Exception:
                        ui.label("레이더 차트 생성 실패").classes("text-gray-500")

                with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
                    try:
                        fig_water = plot_score_waterfall(row)
                        if fig_water:
                            fig_water.update_layout(
                                height=300,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="white",
                            )
                            ui.plotly(fig_water).classes("w-full")
                    except Exception:
                        ui.label("워터폴 차트 생성 실패").classes("text-gray-500")

            # ── ROUTE 상태 배지 ──
            route_val = str(row.get("ROUTE", "NEUTRAL"))
            route_colors = {
                "ATTACK": "#EF4444", "ARMED": "#F59E0B",
                "WAIT": "#3B82F6", "NEUTRAL": "#6B7280", "OVERHEAT": "#A855F7",
            }
            color = route_colors.get(route_val, "#6B7280")
            ui.label(f"⚡ 현재 상태: {route_val}").classes(
                "text-lg font-bold mt-4 px-4 py-2 rounded-lg text-white"
            ).style(f"background: {color}")

    # 이벤트 연결
    route_filter.on("update:model-value", lambda _: _build_table())
    sort_mode.on("update:model-value", lambda _: _build_table())

    # 초기 렌더링
    _build_table()


# ═══════════════════════════════════════════
#  7. 가격 바 시각화 컴포넌트
# ═══════════════════════════════════════════
def _render_price_bar(stop, entry, close, t1, t2=0):
    """손절~목표 구간 시각적 가격 바"""
    points = [("손절", stop, "#EF4444"), ("매수", entry, "#3B82F6"), ("현재", close, "#FFFFFF")]
    if t1 > 0:
        points.append(("T1", t1, "#10B981"))
    if t2 > 0 and t2 != t1:
        points.append(("T2", t2, "#EAB308"))
    points.sort(key=lambda x: x[1])

    p_min = points[0][1] * 0.98
    p_max = points[-1][1] * 1.02
    rng = p_max - p_min
    if rng <= 0:
        return

    bar_html = (
        '<div style="position:relative; height:55px; '
        "background:linear-gradient(90deg, rgba(239,68,68,0.15) 0%, "
        "rgba(16,185,129,0.15) 100%); "
        'border-radius:10px; margin:8px 0 20px 0;">'
    )
    for label, price, color in points:
        pct = max(3, min((price - p_min) / rng * 100, 97))
        is_cur = label == "현재"
        sz = "14px" if is_cur else "10px"
        bdr = "2px solid #FFF" if is_cur else "none"
        fw = "bold" if is_cur else "normal"
        bar_html += (
            f'<div style="position:absolute; left:{pct}%; top:50%; '
            f'transform:translate(-50%,-50%); z-index:{"10" if is_cur else "5"}; text-align:center;">'
            f'<div style="width:{sz}; height:{sz}; background:{color}; '
            f'border-radius:50%; border:{bdr}; margin:0 auto;"></div>'
            f'<div style="font-size:10px; color:{color}; white-space:nowrap; '
            f'margin-top:3px; font-weight:{fw};">{label}<br>{int(price):,}</div>'
            f"</div>"
        )
    bar_html += "</div>"
    ui.html(bar_html)


# ═══════════════════════════════════════════
#  8. 유틸리티
# ═══════════════════════════════════════════
async def _refresh_data():
    """데이터 새로고침"""
    store.refresh()
    ui.notify("🔄 데이터 새로고침 완료!", type="positive")
    # 1초 후 페이지 리로드
    await ui.run_javascript("setTimeout(() => location.reload(), 500)")


# ═══════════════════════════════════════════
#  9. 앱 실행
# ═══════════════════════════════════════════
if __name__ in {"__main__", "__mp_main__"}:
    # 시작 시 데이터 로드
    store.refresh()

    ui.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        title=f"LDY Pro Trader v{APP_VERSION}",
        favicon="💎",
        dark=True,
        reload=False,  # 프로덕션에서는 False
        show=False,     # 서버 모드 (브라우저 자동 열기 비활성)
    )
