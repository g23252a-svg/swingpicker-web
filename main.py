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
    from version_info import APP_VERSION, get_version_layer_label
except Exception:
    APP_VERSION = "12.3.0"
    def get_version_layer_label():  # [v22.3.8 A2] fallback — import 실패 시 단일 표시
        return f"v{APP_VERSION}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ldy-nicegui")


# ═══════════════════════════════════════════
#  [임시] 종목 상세 v2 테스트 (/stock-v2-test/{code})
#  머지 직후 화면 검증용 — 와이투솔루션(011690) / 한온시스템(018880)
#  production 통합 후 제거 예정
#
#  보안:
#    - admin 로그인 OR (non-production 환경에서) ENABLE_STOCK_V2_TEST=1 일 때 접근 가능
#    - APP_ENV=production 에서는 환경변수 무시, admin 만 허용
#    - 그 외는 차단 메시지 표시
# ═══════════════════════════════════════════
@ui.page('/stock-v2-test/{code}')
async def stock_v2_test_page(code: str):
    """[임시] 종목 상세 v2 — Step 2A-2E 풀 대시보드 화면 검증용."""
    # ── 보안 가드 (무거운 import 전에 먼저 체크) ──
    is_prod = os.getenv("APP_ENV", "production").lower() == "production"
    is_env_enabled = os.getenv("ENABLE_STOCK_V2_TEST", "0") == "1"
    is_admin = False
    try:
        is_admin = get_auth_status() == "admin"
    except Exception as _e:
        # auth status 조회 실패 시 보안상 비-admin으로 처리 (디폴트 False)
        # 보안 관련 path라 logger.warning으로 추적
        import logging
        logging.getLogger(__name__).warning(f"[stock_v2_test_page] auth_status 조회 실패: {_e}")

    # production에서는 환경변수 무시, admin만 허용 (실수 토글 방지)
    if is_prod:
        is_env_enabled = False

    if not (is_env_enabled or is_admin):
        ui.add_head_html('<style>body{background:#0F1117;}</style>')
        with ui.column().classes("w-full items-center p-12"):
            ui.label("🔒").classes("text-6xl mb-3")
            ui.label(
                "이 페이지는 관리자 전용 테스트 페이지입니다"
            ).classes("text-base text-gray-300 text-center")
        return

    # ── 보안 통과 후에만 무거운 import 수행 ──
    import pandas as pd
    from components.stock_detail_v2 import render_stock_detail_v2_partial

    # NiceGUI Quasar 기본 컨테이너 폭 제약 해제 + 다크 배경 + viewport meta (모바일 인식)
    ui.add_head_html('''
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          body { background: #0F1117; margin: 0; padding: 0; }
          /* NiceGUI/Quasar 기본 max-width 해제 — 전체 폭 사용 */
          .q-page-container, .q-page, .nicegui-content { max-width: none !important; }
          .nicegui-content { padding: 16px !important; gap: 0 !important; }
          /* 페이지 최상위 column이 전체 폭 사용하도록 */
          .nicegui-content > * { width: 100% !important; max-width: none !important; }
          /* 모바일에선 padding 축소 */
          @media (max-width: 768px) {
            .nicegui-content { padding: 8px !important; }
          }
        </style>
    ''')

    # ── CSV 인코딩 fallback (utf-8 → utf-8-sig → cp949) ──
    df = None
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(
                "data/recommend_latest.csv",
                dtype={"종목코드": str},
                encoding=enc,
            )
            break
        except Exception as e:
            last_err = e
            continue
    if df is None:
        ui.label(f"CSV 로드 실패 (모든 인코딩): {last_err}").style("color: red")
        return

    sub = df[df["종목코드"] == code.zfill(6)]
    if sub.empty:
        ui.label(
            f"종목코드 {code} 가 recommend_latest.csv 에 없음"
        ).style("color: orange")
        return

    row = sub.iloc[0].to_dict()

    # 비교 종목 자동 매핑 (하단 4섹터의 compare 패널 검증용)
    code_norm = code.zfill(6)
    compare_map = {
        "011690": "한온시스템",      # 와이투솔루션 → 한온시스템
        "018880": "와이투솔루션",    # 한온시스템 → 와이투솔루션
    }
    compare_name = compare_map.get(code_norm, "")

    render_stock_detail_v2_partial(
        row,
        rank=int(row.get("LDY_RANK", 0) or 0),
        total=len(df),
        timestamp=str(row.get("기준일", "")),
        compare_name=compare_name,
    )


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


# [v22 Step AF] 명시적 /logout 라우트 — 404 방지
@ui.page('/logout')
async def logout_page():
    """로그아웃 후 /login으로 리다이렉트.
    
    재동의 거부 / 헤더 로그아웃 버튼 / terms_consent.show_re_consent_dialog
    에서 호출됨.
    """
    try:
        set_current_user(None)
    except Exception as e:
        logger.warning(f"set_current_user(None) 실패: {e}")
    ui.navigate.to("/login")


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
    
    # [v22 Step AD+AE] 약관 변경 시 재동의 강제 (관리자 제외)
    # TERMS_VERSION 환경변수 변경 시 모든 사용자에게 재동의 요청
    # [Step AE] 미완료 시 핵심 탭 렌더링 차단 (다이얼로그만이 아닌 접근 자체 제한)
    re_consent_required = False
    if user and auth not in ("guest", "admin"):
        try:
            from components.terms_consent import (
                has_user_agreed, show_re_consent_dialog
            )
            email = user.get("login_id") or user.get("id") or user.get("email", "")
            if email and not has_user_agreed(email):
                # [v22 Step AF] 동의 완료 시 자동 새로고침 — 차단 화면 즉시 해제
                show_re_consent_dialog(
                    email,
                    on_agreed=lambda: ui.run_javascript(
                        "window.location.reload()"
                    ),
                )
                re_consent_required = True
        except ImportError:
            pass  # terms_consent 모듈 없으면 무시 (하위 호환)
        except Exception as e:
            logger.warning(f"재동의 다이얼로그 호출 실패: {e}")
    
    # [v22 Step AE] 재동의 미완료 시 핵심 탭 차단 — 빈 화면 + 안내
    if re_consent_required:
        with ui.column().classes("w-full items-center p-12 max-w-2xl mx-auto"):
            ui.label("📜").classes("text-7xl mb-4")
            ui.label("약관이 변경되었습니다").classes(
                "text-2xl font-bold text-white mb-3"
            )
            ui.label(
                "이용 계속을 위해 변경된 약관에 동의해주세요.\n"
                "동의 다이얼로그가 화면에 표시됩니다."
            ).classes("text-sm text-gray-300 text-center mb-4 whitespace-pre-line")
            with ui.row().classes("gap-3 mt-3"):
                ui.button(
                    "🔄 새로고침",
                    on_click=lambda: ui.run_javascript("window.location.reload()"),
                ).props("color=primary outlined")
                ui.button(
                    "❌ 로그아웃",
                    on_click=lambda: ui.navigate.to("/logout"),
                ).props("flat color=gray")
        return  # 핵심 탭 렌더링 차단

    # ─── Hero Banner ───
    with ui.row().classes("w-full items-center justify-between px-4 py-3 rounded-xl mb-2 "
                          "bg-gradient-to-r from-[#1a1a2e] via-[#16213e] to-[#0f3460]"):
        with ui.column().classes("gap-0"):
            ui.label("💎 SwingPicker").classes(
                "text-2xl font-bold text-transparent bg-clip-text "
                "bg-gradient-to-r from-blue-400 to-purple-400"
            ).style("font-family:Outfit,sans-serif")
            # [v22.3.8 A2] UI / 추천 / 검증 레이어 버전 분리 표시
            ui.label(get_version_layer_label()).classes("text-xs text-gray-400")
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
    # [업데이트 알림] 회원이 아직 안 본 새 버전이 있으면 업데이트 탭에 🔴 표시
    #   - app.storage.user["last_seen_version"] 와 APP_VERSION 비교
    #   - 업데이트 탭 클릭 시 last_seen_version 갱신 → 🔴 사라짐
    try:
        _last_seen = app.storage.user.get("last_seen_version", "")
    except Exception:
        _last_seen = ""
    _has_new_update = bool(_last_seen != APP_VERSION)
    _update_label = "🧩 업데이트 🔴" if _has_new_update else "🧩 업데이트"

    TAB_DEFS = [
        ("t1", "📊 시장"),
        ("t2", "🔭 종목 분석"),
        ("t3", "💼 내 자산"),
        ("t11", "💎 멤버십"),
        ("t4", "📮 문의"),
        ("t5", "⚖️ 약관"),
        ("t6", _update_label),
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
        "t1": lambda: render_tab_market(df, auth),
        "t2": lambda: render_tab_stocks(df, auth, store),
        "t3": lambda: render_tab_portfolio(df, auth),
        "t11": lambda: render_tab_pricing(auth, user),
        "t4": lambda: render_tab_inquiry(auth, user),
        "t5": lambda: render_tab_terms(),
        "t6": lambda: render_tab_updates(),
        "t7": lambda: render_tab_perf(auth),
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
        # [업데이트 알림] 업데이트 탭 클릭 시 → 현재 버전을 '본 것'으로 저장 + 🔴 제거
        #   set_label()은 탭의 label(표시 텍스트)만 바꾸고 name(라우팅 키)은 유지하므로
        #   label_to_key 매핑이 깨지지 않는다.
        if key == "t6":
            try:
                app.storage.user["last_seen_version"] = APP_VERSION
            except Exception as exc:
                logger.warning(f"last_seen_version 저장 실패: {exc}")
            try:
                t6_tab = tab_refs.get("t6")
                if t6_tab is not None:
                    t6_tab.set_label("🧩 업데이트")
            except Exception as exc:
                logger.warning(f"업데이트 탭 라벨 갱신 실패: {exc}")

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
