# -*- coding: utf-8 -*-
"""
SwingPicker — 오늘의 Top 3 페이지 (/briefing)
═══════════════════════════════════════════════
매일 자동 생성된 브리핑을 웹에서 표시
비로그인 유저도 열람 가능 → 가입 유도
"""

import os
import json
import logging
from datetime import datetime

from nicegui import ui
from components.ui_utils import DARK_CSS
from components.tab_stocks import (
    _route_kr, _route_color, _metric_card, _price_bar_html,
)
from shared_utils import nz_num, safe_float

logger = logging.getLogger("ldy-nicegui")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_briefing() -> dict:
    """최신 브리핑 JSON 로드"""
    path = os.path.join(DATA_DIR, "briefing_latest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"브리핑 JSON 로드 실패: {e}")
        return {}


async def render_briefing_page():
    """오늘의 Top 3 브리핑 페이지"""
    ui.add_head_html(DARK_CSS)
    ui.add_head_html('''
    <meta property="og:title" content="SwingPicker AI 오늘의 Top 3"/>
    <meta property="og:description" content="AI가 107종목을 분석해서 뽑은 오늘의 핵심 종목"/>
    <meta property="og:type" content="article"/>
    <title>오늘의 Top 3 | SwingPicker</title>
    ''')

    # 네비게이션
    with ui.row().classes("w-full items-center justify-between px-4 py-3 mb-4 "
                          "bg-gradient-to-r from-[#1a1a2e] via-[#16213e] to-[#0f3460] rounded-xl"):
        with ui.row().classes("items-center gap-3 cursor-pointer").on("click", lambda: ui.navigate.to("/")):
            ui.label("💎 SwingPicker").classes(
                "text-xl font-bold text-transparent bg-clip-text "
                "bg-gradient-to-r from-blue-400 to-purple-400"
            ).style("font-family:Outfit,sans-serif")
        with ui.row().classes("gap-2"):
            ui.button("← 메인", on_click=lambda: ui.navigate.to("/")).props("flat dense").classes("text-white text-sm")
            ui.button("🔐 로그인", on_click=lambda: ui.navigate.to("/login")).props("flat dense").classes("text-white text-sm")

    # 브리핑 데이터 로드
    data = _load_briefing()
    if not data or not data.get("stocks"):
        with ui.column().classes("w-full items-center p-12"):
            ui.label("📝 아직 오늘의 브리핑이 없습니다").classes("text-xl text-gray-400")
            ui.label("매일 장 마감 후 자동 생성됩니다").classes("text-sm text-gray-500 mt-2")
        return

    trade_date = data.get("trade_date", "")
    date_display = f"{trade_date[:4]}.{trade_date[4:6]}.{trade_date[6:]}" if len(trade_date) == 8 else trade_date
    stocks = data["stocks"]

    # 히어로
    with ui.card().classes("w-full p-6 rounded-2xl text-center "
                           "bg-gradient-to-r from-[#1a1a2e] via-[#16213e] to-[#0f3460]"):
        ui.label("🎯 AI 오늘의 Top 3").classes(
            "text-3xl font-bold text-transparent bg-clip-text "
            "bg-gradient-to-r from-yellow-400 to-orange-400"
        ).style("font-family:Outfit,sans-serif")
        ui.label(f"107종목 분석 → 핵심 3종목 선별 · {date_display}").classes("text-gray-300 mt-2")

    # 종목 카드
    for stock in stocks:
        code = stock["code"]
        name = stock["name"]
        route = stock["route"]
        score = stock["score"]
        entry = stock["entry"]
        stop = stock["stop"]
        t1 = stock["target1"]
        rr = stock["rr"]
        wr = stock["est_win_rate"]
        close = stock["close"]

        rc = _route_color(route)
        sc_color = "#10B981" if score >= 80 else "#3B82F6" if score >= 60 else "#F59E0B"
        wr_pct = wr * 100 if wr <= 1 else wr

        stop_pct = (stop / entry - 1) * 100 if entry > 0 and stop > 0 else 0
        t1_pct = (t1 / entry - 1) * 100 if entry > 0 and t1 > 0 else 0

        with ui.card().classes("w-full p-5 mt-4 rounded-2xl border cursor-pointer").style(
            f"border-color:{rc}; background: linear-gradient(135deg, rgba(26,26,46,0.95), rgba(15,52,96,0.4));"
        ).on("click", lambda c=code: ui.navigate.to(f"/stock/{c}")):

            # 상단: 순위 + 종목명 + 점수
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-3"):
                    ui.label(f"#{stock['rank']}").classes("text-2xl font-bold").style(f"color:{rc}")
                    with ui.column().classes("gap-0"):
                        ui.label(name).classes("text-xl font-bold text-white")
                        with ui.row().classes("gap-2"):
                            ui.badge(code, color="#374151").classes("text-xs")
                            ui.badge(stock.get("market", ""), color="#1E3A5F").classes("text-xs")
                            if stock.get("sector"):
                                ui.badge(stock["sector"], color="#1E3A5F").classes("text-xs")
                with ui.column().classes("items-end"):
                    ui.label(f"{score:.0f}점").classes("text-2xl font-bold").style(f"color:{sc_color}")
                    ui.label(_route_kr(route)).classes("text-sm").style(f"color:{rc}")

            ui.separator().classes("my-3")

            # 매매 시나리오
            with ui.row().classes("w-full gap-3 flex-wrap"):
                _metric_card("현재가", f"{close:,}", "")
                if entry > 0:
                    _metric_card("매수가", f"{entry:,}", "AI 산출")
                    _metric_card("손절가", f"{stop:,}", f"{stop_pct:+.1f}%", False)
                if t1 > 0:
                    _metric_card("목표가", f"{t1:,}", f"{t1_pct:+.1f}% (RR {rr:.1f}:1)")

            # 승률 + 상세 링크
            with ui.row().classes("w-full items-center justify-between mt-3"):
                wr_color = "#10B981" if wr_pct >= 60 else "#F59E0B" if wr_pct >= 45 else "#EF4444"
                ui.label(f"추정 승률 {wr_pct:.0f}%").classes("text-sm font-bold").style(f"color:{wr_color}")
                ui.label("📊 상세 분석 보기 →").classes("text-sm text-blue-400")

    # CTA
    with ui.card().classes("w-full p-6 mt-6 rounded-2xl text-center "
                           "bg-gradient-to-r from-purple-900/40 to-blue-900/40 "
                           "border border-purple-500/30"):
        ui.label("💎 전체 107종목 분석 + 전략 샌드박스").classes("text-lg font-bold text-white")
        ui.label("무료 가입으로 모든 종목의 AI 분석을 확인하세요").classes("text-gray-300 mt-2")
        with ui.row().classes("justify-center gap-3 mt-4"):
            ui.button("무료 회원가입", on_click=lambda: ui.navigate.to("/login")).props(
                "rounded unelevated").classes("bg-blue-600 text-white px-6")
            ui.button("💎 Prime 구독", on_click=lambda: ui.navigate.to("/")).props(
                "rounded outline").classes("text-purple-300 px-6")

    # 푸터
    with ui.column().classes("w-full items-center mt-8 mb-4 gap-1"):
        ui.label(f"📅 기준일: {date_display} · 생성: {data.get('generated_at', '')}").classes("text-xs text-gray-500")
        ui.label("⚠️ AI 분석 참고 자료이며 투자 권유가 아닙니다. 투자 판단은 본인 책임.").classes("text-xs text-gray-600")
