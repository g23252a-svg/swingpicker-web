# -*- coding: utf-8 -*-
"""
LDY Pro Trader — NiceGUI Full Edition
═══════════════════════════════════════
순수 라우터: 페이지 등록 + 탭 배선 + 앱 부트스트랩

Tab 1: 📊 시장 현황       → components/tab_market.py
Tab 2: 🔭 종목 분석       → components/tab_stocks.py
Tab 3: 💼 내 자산         → components/tab_portfolio.py
Tab 4: 📮 문의 게시판      → components/tab_inquiry.py
Tab 5: ⚖️ 이용 약관       → components/tab_terms.py
Tab 6: 🧩 업데이트 노트    → components/tab_updates.py
Tab 7: 📈 시스템 성과      → components/tab_perf.py
Tab 8: 👑 회원 관리        → components/tab_admin.py
Tab 9: 📓 매매 일지        → trade_journal_tab.py
"""

import os
import logging

from nicegui import ui, app
from async_helpers import run_sync, register_shutdown

# ─── 상태 & 인증 ───
from services.data_store import store
from services.auth import get_current_user, set_current_user, get_auth_status

# ─── UI ───
from components.ui_utils import DARK_CSS
from views.login_page import login_page  # noqa: F401 — @ui.page('/login') 등록

# ─── 탭 컴포넌트 ───
from components.tab_market import render_tab_market
from components.tab_stocks import render_tab_stocks
from components.tab_portfolio import render_tab_portfolio
from components.tab_inquiry import render_tab_inquiry
from components.tab_terms import render_tab_terms
from components.tab_updates import render_tab_updates
from components.tab_perf import render_tab_perf
from components.tab_admin import render_tab_admin

# ─── 매매일지 (선택) ───
try:
    from trade_journal_tab import render_trade_journal_tab
    JOURNAL_OK = True
except ImportError:
    JOURNAL_OK = False

# ─── 버전 정보 ───
try:
    from version_info import APP_VERSION
except Exception:
    APP_VERSION = "12.3.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ldy-nicegui")


# ═══════════════════════════════════════════
#  메인 페이지
# ═══════════════════════════════════════════
@ui.page('/')
async def index():
    ui.add_head_html(DARK_CSS)
    if not store.loaded:
        await run_sync(store.refresh)
    df = store.scored
    auth = get_auth_status()
    user = get_current_user()

    # ─── Hero Banner ───
    with ui.row().classes("w-full items-center justify-between px-4 py-3 rounded-xl mb-2 "
                          "bg-gradient-to-r from-[#1a1a2e] via-[#16213e] to-[#0f3460]"):
        with ui.column().classes("gap-0"):
            ui.label("💎 LDY Pro Trader").classes(
                "text-2xl font-bold text-transparent bg-clip-text "
                "bg-gradient-to-r from-blue-400 to-purple-400"
            ).style("font-family:Outfit,sans-serif")
            ui.label(f"v{APP_VERSION} · NiceGUI Edition").classes("text-xs text-gray-400")
        with ui.row().classes("gap-2 items-center"):
            if user:
                ui.label(f"👋 {user.get('nickname', '')}").classes("text-white text-sm")
                badge_color = "green" if auth in ("admin", "prime") else "blue" if auth == "pro" else "gray"
                ui.badge(auth.upper()).props(f"color={badge_color}")
                ui.button("로그아웃", on_click=lambda: (set_current_user(None), ui.navigate.to("/login"))
                          ).props("flat dense").classes("text-white text-xs")
            else:
                ui.button("🔐 로그인", on_click=lambda: ui.navigate.to("/login")
                          ).props("flat dense").classes("text-white")
            ui.button("🔄", on_click=_do_refresh).props("flat round dense").classes("text-white")

    if df.empty:
        ui.label("⚠️ 데이터 없음 — data/recommend_latest.csv 확인").classes("text-yellow-400 text-lg p-8")
        return

    # ─── 탭 구성 ───
    with ui.tabs().classes("w-full text-white") as tabs:
        t1 = ui.tab("📊 시장")
        t2 = ui.tab("🔭 종목 분석")
        t3 = ui.tab("💼 내 자산")
        t4 = ui.tab("📮 문의")
        t5 = ui.tab("⚖️ 약관")
        t6 = ui.tab("🧩 업데이트")
        t7 = ui.tab("📈 성과")
        t9 = ui.tab("📓 매매 일지")
        if auth == "admin":
            t8 = ui.tab("👑 관리")

    with ui.tab_panels(tabs, value=t1).classes("w-full"):
        with ui.tab_panel(t1): render_tab_market(df)
        with ui.tab_panel(t2): render_tab_stocks(df, auth, store)
        with ui.tab_panel(t3): render_tab_portfolio(df, auth)
        with ui.tab_panel(t4): render_tab_inquiry(auth, user)
        with ui.tab_panel(t5): render_tab_terms()
        with ui.tab_panel(t6): render_tab_updates()
        with ui.tab_panel(t7): render_tab_perf()
        with ui.tab_panel(t9):
            if JOURNAL_OK:
                render_trade_journal_tab(df_scored=df)
            else:
                ui.label("⚠️ trade_journal_tab 모듈 없음").classes("text-yellow-400")
        if auth == "admin":
            with ui.tab_panel(t8): render_tab_admin()

    ui.label(f"📅 데이터 기준: {store.data_ts} · ⚠️ 투자 판단은 본인 책임"
             ).classes("text-xs text-gray-500 text-center mt-8 mb-4")


async def _do_refresh():
    await run_sync(store.refresh)
    ui.notify("🔄 데이터 새로고침 완료!", type="positive")
    await ui.run_javascript("setTimeout(()=>location.reload(),500)")


# ═══════════════════════════════════════════
#  앱 실행
# ═══════════════════════════════════════════
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.add_static_files("/static", STATIC_DIR)

if __name__ in {"__main__", "__mp_main__"}:
    store.refresh()
    register_shutdown(app)
    ui.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        title=f"LDY Pro Trader v{APP_VERSION}",
        favicon="💎",
        dark=True,
        storage_secret=os.environ["STORAGE_SECRET"],
        reload=False,
        show=False,
    )
