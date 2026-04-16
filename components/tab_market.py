# components/tab_market.py — Tab 1: 시장 현황
"""
NiceGUI UI 전용 — 데이터 로직은 MarketService에 위임.

사용법 (main.py):
    from components.tab_market import render_tab1_market
    with ui.tab_panel(t1):
        render_tab1_market()
"""
from nicegui import ui

from state import global_data
from data.services.market_service import MarketService
from chart_components import (
    plot_fear_greed_gauge,
    plot_sector_treemap,
    plot_sector_momentum_bar,
)

_market_svc = MarketService()


def render_tab1_market():
    """Tab 1: 시장 현황 — 공포/탐욕 + 섹터 분석"""
    df = global_data.scored
    summary = _market_svc.get_market_summary(df)

    # ── 헤더 메트릭 카드 ──
    _section_title("📡 시장 현황")
    with ui.row().classes("w-full gap-4 flex-wrap"):
        fg_icon = "🟢" if summary["fg_score"] >= 50 else "🔴"
        _metric_card(
            "시장 심리",
            f"{fg_icon} {summary['fg_label']}",
            f"지수: {summary['fg_score']:.0f}/100",
            positive=summary["fg_score"] >= 50,
        )

        if summary["avg_ret_top20"] != 0:
            _metric_card(
                "Top20 평균 수익률",
                f"{summary['avg_ret_top20']:+.2f}%",
                "전일 대비",
                positive=summary["avg_ret_top20"] >= 0,
            )

        _metric_card(
            "분석 종목",
            f"{summary['total_count']}개",
            f"ARMED/ATTACK: {summary['active_count']}개",
        )

        _metric_card(
            "데이터 기준일",
            global_data.data_ts,
        )

    # ── 공포/탐욕 게이지 + 섹터 트리맵 ──
    _section_title("🌡️ 공포/탐욕 & 주도 섹터")
    with ui.row().classes("w-full gap-4 flex-wrap items-start"):
        # 왼쪽: 게이지
        with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
            fig_g = plot_fear_greed_gauge(summary["fg_score"])
            if fig_g:
                _plotly_dark(fig_g, 280)
                ui.plotly(fig_g).classes("w-full")

        # 오른쪽: 섹터 트리맵
        with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
            ui.label("🔥 오늘의 주도 섹터").classes("text-sm font-bold text-white mb-2")
            if "업종" in df.columns:
                fig_m = plot_sector_treemap(df.head(50))
                if fig_m:
                    _plotly_dark(fig_m, 280)
                    ui.plotly(fig_m).classes("w-full")
                else:
                    ui.label("섹터 데이터 부족").classes("text-gray-500")

    # ── 섹터 모멘텀 바 차트 ──
    _section_title("🚀 섹터 모멘텀 Top 10")
    fig_mom = plot_sector_momentum_bar(df)
    if fig_mom and len(fig_mom.data) > 0:
        _plotly_dark(fig_mom, 350)
        ui.plotly(fig_mom).classes("w-full")
    else:
        ui.label("모멘텀 데이터 부족").classes("text-gray-500")


# ═══════════════════════════════════════════
#  공통 UI 헬퍼 (추후 components/common.py로 이동)
# ═══════════════════════════════════════════

def _section_title(text: str):
    ui.label(text).classes(
        "text-lg font-bold text-white mt-6 mb-2 border-b border-gray-700 pb-2"
    )


def _metric_card(title: str, value: str, delta: str = "", positive: bool = True):
    with ui.card().classes(
        "p-4 min-w-[140px] bg-[#1a1a2e] border border-gray-700 rounded-xl"
    ):
        ui.label(title).classes("text-xs text-gray-400 uppercase tracking-wide")
        ui.label(str(value)).classes("text-xl font-bold text-white mt-1")
        if delta:
            color = "text-green-400" if positive else "text-red-400"
            ui.label(str(delta)).classes(f"text-sm {color} mt-0.5")


def _plotly_dark(fig, height: int = 300):
    """Plotly 차트 다크 테마 적용 (in-place)"""
    if fig:
        fig.update_layout(
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(t=30, b=10, l=10, r=10),
        )
