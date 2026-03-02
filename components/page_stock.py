# -*- coding: utf-8 -*-
"""
SwingPicker — 종목 개별 페이지 (/stock/{code})
═══════════════════════════════════════════════
공유 가능한 종목 분석 URL: 토스/유튜브/카톡에 링크 가능
비로그인 유저도 기본 분석 열람 가능 → CTA로 가입 유도
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta

from nicegui import ui
from async_helpers import run_sync

from components.tab_stocks import (
    _route_kr, _route_desc, _route_color,
    _section_title, _metric_card, _price_bar_html, _plotly_dark,
    _get_stock_chart_data, _plot_candle_chart,
    ROUTE_KR, ROUTE_COLOR,
)
from shared_utils import nz_num, safe_float
from components.ui_utils import DARK_CSS

logger = logging.getLogger("ldy-nicegui")


def _og_meta(name: str, code: str, score: float, route: str, close: int):
    """Open Graph 메타태그 삽입 (소셜 미리보기)"""
    title = f"{name}({code}) AI 분석 | SwingPicker"
    desc = f"AI 점수 {score:.0f}점 · {_route_kr(route)} · 현재가 {close:,}원"
    ui.add_head_html(f'''
    <meta property="og:title" content="{title}"/>
    <meta property="og:description" content="{desc}"/>
    <meta property="og:type" content="article"/>
    <meta name="description" content="{desc}"/>
    <title>{title}</title>
    ''')


def _ret_color(val):
    """수익률 색상"""
    v = safe_float(val)
    if v > 0: return "#10B981"
    if v < 0: return "#EF4444"
    return "#6B7280"


async def render_stock_page(code: str, store):
    """종목 개별 페이지 렌더링"""
    ui.add_head_html(DARK_CSS)

    # 데이터 로드
    if not store.loaded:
        await run_sync(store.refresh)
    df = store.scored

    if df.empty:
        ui.label("⚠️ 데이터 로드 실패").classes("text-yellow-400 text-lg p-8")
        return

    # 종목 검색
    code = str(code).zfill(6)
    row = df[df["종목코드"].astype(str).str.zfill(6) == code]
    if row.empty:
        with ui.column().classes("w-full items-center justify-center p-12"):
            ui.label("❌ 종목을 찾을 수 없습니다").classes("text-2xl text-red-400")
            ui.label(f"종목코드: {code}").classes("text-gray-400 mt-2")
            ui.button("← 메인으로", on_click=lambda: ui.navigate.to("/")).props("outline").classes("mt-4")
        return

    row = row.iloc[0]
    name = str(row.get("종목명", code))
    score = safe_float(row.get("DISPLAY_SCORE", 0))
    route = str(row.get("ROUTE", "NEUTRAL"))
    close = int(nz_num(row.get("종가", 0)))
    entry = int(nz_num(row.get("추천매수가", 0)))
    stop = int(nz_num(row.get("손절가", 0)))
    t1 = int(nz_num(row.get("추천매도가1", 0)))
    t2 = int(nz_num(row.get("추천매도가2", 0)))
    sector = str(row.get("업종_대분류", ""))
    market = str(row.get("시장", ""))
    rsi = safe_float(row.get("RSI14", 0))
    est_wr = safe_float(row.get("EST_WIN_RATE", 0))
    v_power = safe_float(row.get("V_POWER", 0))

    # OG 메타태그
    _og_meta(name, code, score, route, close)

    # ═══════════════════════════════════════
    #  헤더 네비게이션
    # ═══════════════════════════════════════
    with ui.row().classes("w-full items-center justify-between px-4 py-3 mb-4 "
                          "bg-gradient-to-r from-[#1a1a2e] via-[#16213e] to-[#0f3460] rounded-xl"):
        with ui.row().classes("items-center gap-3 cursor-pointer").on("click", lambda: ui.navigate.to("/")):
            ui.label("💎 SwingPicker").classes(
                "text-xl font-bold text-transparent bg-clip-text "
                "bg-gradient-to-r from-blue-400 to-purple-400"
            ).style("font-family:Outfit,sans-serif")
        with ui.row().classes("gap-2"):
            ui.button("← 전체 종목", on_click=lambda: ui.navigate.to("/")).props("flat dense").classes("text-white text-sm")
            ui.button("🔐 로그인", on_click=lambda: ui.navigate.to("/login")).props("flat dense").classes("text-white text-sm")

    # ═══════════════════════════════════════
    #  종목 히어로 섹션
    # ═══════════════════════════════════════
    rc = _route_color(route)
    sc_color = "#10B981" if score >= 80 else "#3B82F6" if score >= 60 else "#F59E0B" if score >= 40 else "#94A3B8"

    with ui.card().classes("w-full p-6 rounded-2xl border").style(
        f"border-color:{rc}; background: linear-gradient(135deg, rgba(26,26,46,0.95), rgba(15,52,96,0.6));"
    ):
        # 종목명 + 뱃지
        with ui.row().classes("w-full items-start justify-between flex-wrap gap-3"):
            with ui.column().classes("gap-1"):
                ui.label(f"{name}").classes("text-3xl font-bold text-white").style("font-family:Outfit,sans-serif")
                with ui.row().classes("gap-2 items-center"):
                    ui.badge(code, color="#374151").classes("text-xs")
                    ui.badge(market, color="#1E3A5F").classes("text-xs")
                    if sector:
                        ui.badge(sector, color="#1E3A5F").classes("text-xs")
            with ui.column().classes("items-end gap-1"):
                ui.label(f"{close:,}원").classes("text-3xl font-bold text-white")
                # 수익률 표시
                for label, key in [("5일", "ret_5d_%"), ("20일", "ret_20d_%"), ("60일", "ret_60d_%")]:
                    val = safe_float(row.get(key, 0))
                    if val != 0:
                        ui.label(f"{label} {val:+.1f}%").classes("text-sm").style(f"color:{_ret_color(val)}")

        ui.separator().classes("my-4")

        # AI 점수 + 신호 + 승률
        with ui.row().classes("w-full gap-4 flex-wrap"):
            with ui.card().classes("flex-1 min-w-[160px] p-4 bg-[rgba(255,255,255,0.05)] rounded-xl text-center"):
                ui.label("AI 점수").classes("text-xs text-gray-400")
                ui.label(f"{score:.0f}").classes("text-4xl font-bold").style(f"color:{sc_color}")
                ui.label("/ 100").classes("text-xs text-gray-500")

            with ui.card().classes("flex-1 min-w-[160px] p-4 rounded-xl text-center").style(
                f"background:rgba({','.join(str(int(rc.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15)"
            ):
                ui.label("현재 신호").classes("text-xs text-gray-400")
                ui.label(_route_kr(route)).classes("text-2xl font-bold").style(f"color:{rc}")
                ui.label(_route_desc(route)).classes("text-xs text-gray-500 mt-1")

            with ui.card().classes("flex-1 min-w-[160px] p-4 bg-[rgba(255,255,255,0.05)] rounded-xl text-center"):
                ui.label("추정 승률").classes("text-xs text-gray-400")
                wr_pct = est_wr * 100 if est_wr <= 1 else est_wr
                wr_color = "#10B981" if wr_pct >= 60 else "#F59E0B" if wr_pct >= 45 else "#EF4444"
                ui.label(f"{wr_pct:.0f}%").classes("text-4xl font-bold").style(f"color:{wr_color}")
                ui.label("캘리브레이션 기반").classes("text-xs text-gray-500")

    # ═══════════════════════════════════════
    #  매매 시나리오 카드
    # ═══════════════════════════════════════
    if entry > 0 and stop > 0:
        risk = entry - stop
        rr1 = (t1 - entry) / risk if risk > 0 and t1 > 0 else 0
        stop_pct = (stop / entry - 1) * 100
        t1_pct = (t1 / entry - 1) * 100 if t1 > 0 else 0

        with ui.card().classes("w-full p-5 mt-4 bg-[#0d1b2a] border border-gray-700 rounded-2xl"):
            ui.label("📋 매매 시나리오").classes("text-lg font-bold text-white mb-3")

            with ui.row().classes("w-full gap-3 flex-wrap"):
                _metric_card("🔵 추천 매수가", f"{entry:,}", "AI 산출")
                _metric_card("🔴 손절가", f"{stop:,}", f"{stop_pct:+.1f}%", False)
                if t1 > 0:
                    _metric_card("🟢 목표가 1", f"{t1:,}", f"{t1_pct:+.1f}% (RR {rr1:.1f}:1)")
                if t2 > 0 and t2 != t1:
                    t2_pct = (t2 / entry - 1) * 100
                    rr2 = (t2 - entry) / risk if risk > 0 else 0
                    _metric_card("🟡 목표가 2", f"{t2:,}", f"{t2_pct:+.1f}% (RR {rr2:.1f}:1)")

            # 가격 바
            if close > 0:
                ui.html(_price_bar_html(stop, entry, close, t1, t2))

    # ═══════════════════════════════════════
    #  기술적 지표 요약
    # ═══════════════════════════════════════
    with ui.card().classes("w-full p-5 mt-4 bg-[#0d1b2a] border border-gray-700 rounded-2xl"):
        ui.label("📊 기술적 지표").classes("text-lg font-bold text-white mb-3")

        with ui.row().classes("w-full gap-3 flex-wrap"):
            # RSI
            rsi_color = "#EF4444" if rsi > 70 else "#10B981" if rsi < 30 else "#3B82F6"
            _metric_card("RSI (14)", f"{rsi:.1f}", "과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립")

            # V-Power
            vp_color = "#10B981" if v_power > 0.5 else "#EF4444" if v_power < -0.5 else "#6B7280"
            _metric_card("V-Power", f"{v_power:.2f}", "매수세 우위" if v_power > 0 else "매도세 우위")

            # 주봉 추세
            weekly_trend = str(row.get("주봉추세", ""))
            _metric_card("주봉 추세", weekly_trend or "—", str(row.get("주봉20선_상회", "")))

            # 이격도
            gap = safe_float(row.get("이격도", 0))
            _metric_card("이격도", f"{gap:+.1f}%", "괴리 주의" if abs(gap) > 5 else "정상 범위")

    # ═══════════════════════════════════════
    #  캔들차트
    # ═══════════════════════════════════════
    with ui.card().classes("w-full p-4 mt-4 bg-[#0d1b2a] border border-gray-700 rounded-2xl"):
        ui.label("🕯️ 일봉 차트 (120일)").classes("text-lg font-bold text-white mb-2")
        chart_holder = ui.column().classes("w-full")

        async def load_chart():
            chart_holder.clear()
            with chart_holder:
                cdata = await run_sync(_get_stock_chart_data, code)
                if cdata is not None:
                    fig = _plot_candle_chart(cdata, code, name, entry, stop, t1, t2)
                    ui.plotly(fig).classes("w-full")
                else:
                    ui.label("📉 차트 데이터 로드 실패").classes("text-yellow-400")

        async def _safe_chart():
            await asyncio.sleep(0.1)
            try:
                if chart_holder.is_deleted:
                    return
            except AttributeError:
                pass
            await load_chart()

        asyncio.create_task(_safe_chart())

    # ═══════════════════════════════════════
    #  CTA — 가입 유도
    # ═══════════════════════════════════════
    with ui.card().classes("w-full p-6 mt-6 rounded-2xl text-center "
                           "bg-gradient-to-r from-purple-900/40 to-blue-900/40 "
                           "border border-purple-500/30"):
        ui.label("🔒 전략 샌드박스 · 매매일지 · 켈리 계산기").classes("text-lg font-bold text-white")
        ui.label("무료 회원가입으로 전체 107종목 분석을 확인하세요").classes("text-gray-300 mt-2")
        with ui.row().classes("justify-center gap-3 mt-4"):
            ui.button("무료 회원가입", on_click=lambda: ui.navigate.to("/login")).props(
                "rounded unelevated"
            ).classes("bg-blue-600 text-white px-6")
            ui.button("💎 Prime 구독", on_click=lambda: ui.navigate.to("/")).props(
                "rounded outline"
            ).classes("text-purple-300 px-6")

    # ═══════════════════════════════════════
    #  푸터
    # ═══════════════════════════════════════
    with ui.column().classes("w-full items-center mt-8 mb-4 gap-1"):
        ui.label(f"📅 데이터 기준: {store.data_ts} · 분석일: {datetime.now().strftime('%Y-%m-%d')}").classes("text-xs text-gray-500")
        ui.label("⚠️ 본 자료는 투자 권유가 아닌 AI 분석 참고 자료입니다. 투자 판단은 본인 책임.").classes("text-xs text-gray-600")
        ui.label("© SwingPicker by LDY Pro Trader").classes("text-xs text-gray-600 mt-1")
