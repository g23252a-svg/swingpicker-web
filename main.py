# -*- coding: utf-8 -*-
"""
SwingPicker — NiceGUI Full Edition
═══════════════════════════════════════
순수 라우터 + Lazy Loading: 탭 클릭 시에만 렌더링

Tab 1: 📊 시장 현황       → 즉시 로드 (기본 탭)
Tab 2~10: Lazy Loading    → 최초 클릭 시에만 렌더링
Tab 10: 🧪 전략 샌드박스   → components/tab_backtest.py (Prime 전용)
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
from components.tab_portfolio_v2 import render_tab_portfolio
from components.tab_inquiry import render_tab_inquiry
from components.tab_terms import render_tab_terms
from components.tab_updates import render_tab_updates
from components.tab_perf import render_tab_perf
from components.tab_admin import render_tab_admin
from components.tab_backtest import render_tab_backtest
from components.tab_pricing import render_tab_pricing
# [v22 Step AA] 사이트 전체 공지 배너
try:
    from components.site_banner import render_site_banner
except ImportError:
    def render_site_banner():
        pass  # 모듈 없어도 정상 작동
from components.page_stock import render_stock_page
from components.page_briefing import render_briefing_page

try:
    from payments import register_payment_routes
    PAYMENTS_OK = True
except ImportError:
    PAYMENTS_OK = False

# [v22 Step S+AC] 법적 페이지 라우트 (/terms /privacy /refund /terms/history)
try:
    from components.legal_pages import register_legal_pages
    LEGAL_OK = True
except ImportError:
    LEGAL_OK = False

try:
    from trade_journal_tab import render_trade_journal_tab
    JOURNAL_OK = True
except ImportError:
    JOURNAL_OK = False

try:
    from version_info import APP_VERSION
except Exception:
    APP_VERSION = "12.3.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ldy-nicegui")


# ═══════════════════════════════════════════
#  종목 개별 페이지 (/stock/{code}) — 공유 가능 URL
# ═══════════════════════════════════════════
@ui.page('/stock/{code}')
async def stock_page(code: str):
    await render_stock_page(code, store)


# ═══════════════════════════════════════════
#  오늘의 Top 3 브리핑 (/briefing) — 매일 자동 생성
# ═══════════════════════════════════════════
@ui.page('/briefing')
async def briefing_page():
    await render_briefing_page()


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

    # [v22 Step AA] 사이트 전체 공지 배너 (환경변수로 제어)
    render_site_banner()
    
    # [v22 Step AD] 약관 변경 시 재동의 강제 (관리자 제외)
    # TERMS_VERSION 환경변수 변경 시 모든 사용자에게 재동의 요청
    if user and auth not in ("guest", "admin"):
        try:
            from components.terms_consent import (
                has_user_agreed, show_re_consent_dialog
            )
            email = user.get("login_id") or user.get("id") or user.get("email", "")
            if email and not has_user_agreed(email):
                # persistent 다이얼로그 — 동의하지 않으면 사이트 사용 불가
                show_re_consent_dialog(email)
        except ImportError:
            pass  # terms_consent 모듈 없으면 무시 (하위 호환)
        except Exception as e:
            logger.warning(f"재동의 다이얼로그 호출 실패: {e}")

    # ─── Hero Banner ───
    with ui.row().classes("w-full items-center justify-between px-4 py-3 rounded-xl mb-2 "
                          "bg-gradient-to-r from-[#1a1a2e] via-[#16213e] to-[#0f3460]"):
        with ui.column().classes("gap-0"):
            ui.label("💎 SwingPicker").classes(
                "text-2xl font-bold text-transparent bg-clip-text "
                "bg-gradient-to-r from-blue-400 to-purple-400"
            ).style("font-family:Outfit,sans-serif")
            ui.label(f"v{APP_VERSION}").classes("text-xs text-gray-400")
        with ui.row().classes("gap-2 items-center"):
            if user:
                ui.label(f"👋 {user.get('nickname', '')}").classes("text-white text-sm")
                badge_color = "green" if auth in ("admin", "prime") else "gray"
                badge_label = {"admin": "관리자", "prime": "프라임", "free": "무료"}.get(auth, auth)
                ui.badge(badge_label).props(f"color={badge_color}")
                ui.button("로그아웃", on_click=lambda: (set_current_user(None), ui.navigate.to("/login"))
                          ).props("flat dense").classes("text-white text-xs")
            else:
                ui.button("🔐 로그인", on_click=lambda: ui.navigate.to("/login")
                          ).props("flat dense").classes("text-white")
            ui.button("🔄", on_click=_do_refresh).props("flat round dense").classes("text-white")

    if df.empty:
        ui.label("⚠️ 데이터 없음 — data/recommend_latest.csv 확인").classes("text-yellow-400 text-lg p-8")
        return

    # ─── 탭 정의 ───
    TAB_DEFS = [
        ("t1", "📊 시장"),
        ("t2", "🔭 종목 분석"),
        ("t3", "💼 내 자산"),
        ("t11", "💎 멤버십"),
        ("t4", "📮 문의"),
        ("t5", "⚖️ 약관"),
        ("t6", "🧩 업데이트"),
        ("t7", "📈 성과"),
        ("t10", "🧪 전략 샌드박스"),
        ("t9", "📓 매매 일지"),
    ]
    if auth == "admin":
        TAB_DEFS.append(("t8", "👑 관리"))

    with ui.tabs().classes("w-full text-white") as tabs:
        tab_refs = {}
        label_to_key = {}
        for key, label in TAB_DEFS:
            tab_refs[key] = ui.tab(label)
            label_to_key[label] = key

    # ─── 빈 컨테이너 패널 (Lazy Loading 핵심) ───
    containers = {}
    with ui.tab_panels(tabs, value=tab_refs["t1"]).classes("w-full"):
        for key in tab_refs:
            with ui.tab_panel(tab_refs[key]):
                containers[key] = ui.column().classes("w-full")

    # ─── 렌더 함수 매핑 ───
    def _render_journal():
        if JOURNAL_OK:
            render_trade_journal_tab(df_scored=df)
        else:
            ui.label("⚠️ trade_journal_tab 모듈 없음").classes("text-yellow-400")

    render_map = {
        "t1": lambda: render_tab_market(df),
        "t2": lambda: render_tab_stocks(df, auth, store),
        "t3": lambda: render_tab_portfolio(df, auth),
        "t11": lambda: render_tab_pricing(auth, user),
        "t4": lambda: render_tab_inquiry(auth, user),
        "t5": lambda: render_tab_terms(),
        "t6": lambda: render_tab_updates(),
        "t7": lambda: render_tab_perf(),
        "t10": lambda: render_tab_backtest(df, auth),
        "t9": _render_journal,
    }
    if auth == "admin":
        render_map["t8"] = lambda: render_tab_admin()

    # ─── Lazy 로더 ───
    loaded = set()

    def load_tab(key):
        if key in loaded or key not in render_map:
            return
        loaded.add(key)
        container = containers.get(key)
        if not container:
            return
        with container:
            try:
                render_map[key]()
            except Exception as e:
                logger.error(f"탭 렌더링 오류 [{key}]: {e}", exc_info=True)
                ui.label(f"❌ 로딩 실패: {e}").classes("text-red-400")

    # Tab 1 즉시 렌더 (기본 탭)
    load_tab("t1")

    # 나머지는 클릭 시 Lazy 렌더
    def on_tab_change(e):
        tab_val = e.value if isinstance(e.value, str) else getattr(e.value, "label", str(e.value))
        key = label_to_key.get(tab_val)
        if key:
            load_tab(key)

    tabs.on_value_change(on_tab_change)

    # ─── 푸터 ───
    ui.label(f"📅 데이터 기준: {store.data_ts} · ⚠️ 투자 판단은 본인 책임"
             ).classes("text-xs text-gray-500 text-center mt-8 mb-4")


async def _do_refresh():
    await run_sync(store.refresh)
    try:
        ui.notify("🔄 데이터 새로고침 완료!", type="positive")
        await ui.run_javascript("setTimeout(()=>location.reload(),500)")
    except RuntimeError:
        pass  # 페이지 이탈 시 슬롯 삭제됨 — 무시 가능


# ═══════════════════════════════════════════
#  앱 실행
# ═══════════════════════════════════════════
# ═══════════════════════════════════════════
#  Phase 2: 결제 API 라우트 등록
# ═══════════════════════════════════════════
if PAYMENTS_OK:
    register_payment_routes()
    logger.info("💳 결제 API 라우트 등록 완료")

# [v22 Step S+AC] 법적 페이지 라우트 등록 (/terms /privacy /refund /terms/history)
if LEGAL_OK:
    register_legal_pages()
    logger.info("📜 법적 페이지 라우트 등록 완료")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.add_static_files("/static", STATIC_DIR)

if __name__ in {"__main__", "__mp_main__"}:
    store.refresh()
    register_shutdown(app)
    ui.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        title=f"SwingPicker v{APP_VERSION}",
        favicon="💎",
        dark=True,
        storage_secret=os.environ["STORAGE_SECRET"],
        reload=False,
        show=False,
    )
