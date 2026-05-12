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

# [v2 토글] Prime/admin 자동 v2 활성화용
try:
    from services.auth import get_auth_status
except Exception:
    def get_auth_status():
        return "guest"

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
    """종목 개별 페이지 렌더링.

    [v2 토글] 다음 조건 중 하나라도 만족 시 v2 풀 대시보드 렌더링:
      1) USE_STOCK_DETAIL_V2=1 환경변수 (모든 사용자)
      2) ?v2=1 쿼리스트링 (특정 URL만)
      3) Prime / admin 회원 (자동 v2)
    실패 시 즉시 v1 fallback (운영 안전성 보장).
    """
    # ═══════════════════════════════════════════
    #  [v2 토글] 환경변수 / 쿼리스트링 / Prime·admin 자동 활성화
    # ═══════════════════════════════════════════
    use_v2_env = os.getenv("USE_STOCK_DETAIL_V2", "0") == "1"

    # Prime / admin 자동 v2 (유료 회원 혜택)
    use_v2_member = False
    try:
        auth = get_auth_status()
        if auth in ("prime", "admin"):
            use_v2_member = True
    except Exception:
        pass

    # ?v2=1 쿼리스트링 (개발/베타 검증용)
    use_v2_query = False
    try:
        from nicegui import context
        req = context.client.request
        if req and req.query_params.get("v2") == "1":
            use_v2_query = True
    except Exception:
        pass

    # ?v2=0 명시적 비활성화 (Prime이라도 v1 보고 싶을 때 escape hatch)
    force_v1 = False
    try:
        from nicegui import context
        req = context.client.request
        if req and req.query_params.get("v2") == "0":
            force_v1 = True
    except Exception:
        pass

    use_v2 = (use_v2_env or use_v2_member or use_v2_query) and not force_v1

    if use_v2:
        try:
            await _render_stock_page_v2(code, store)
            return
        except Exception as e:
            # v2 실패 시 v1으로 graceful fallback (운영 안전성)
            logger.warning(
                f"stock_detail_v2 렌더링 실패, v1 fallback [{code}]: {e}",
                exc_info=True,
            )
            try:
                ui.label("⚠️ v2 렌더링 실패, 기본 화면으로 전환합니다").classes(
                    "text-yellow-400 text-xs p-2"
                )
            except Exception:
                pass

    # ═══════════════════════════════════════════
    #  v1 기본 렌더링 (기존 코드)
    # ═══════════════════════════════════════════
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

        # 1행: 모멘텀 지표
        ui.label("모멘텀").classes("text-xs text-gray-500 uppercase tracking-wide mb-2")
        with ui.row().classes("w-full gap-3 flex-wrap"):
            # RSI
            _metric_card("RSI (14)", f"{rsi:.1f}",
                         "과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립",
                         rsi < 70)

            # MFI
            mfi = safe_float(row.get("MFI14", 0))
            _metric_card("MFI (14)", f"{mfi:.1f}",
                         "과매수" if mfi > 80 else "과매도" if mfi < 20 else "중립",
                         mfi < 80)

            # MACD 기울기
            macd_slope = safe_float(row.get("MACD_Slope_PCT", 0))
            _metric_card("MACD 기울기", f"{macd_slope:+.3f}",
                         "상승 모멘텀" if macd_slope > 0 else "하락 모멘텀",
                         macd_slope > 0)

            # RSI 상승 여부
            rsi_rising = str(row.get("RSI_Rising", ""))
            _metric_card("RSI 방향", "상승 ↑" if rsi_rising == "1" else "하락 ↓",
                         "추세 강화" if rsi_rising == "1" else "추세 약화",
                         rsi_rising == "1")

        ui.separator().classes("my-3")

        # 2행: 추세 지표
        ui.label("추세").classes("text-xs text-gray-500 uppercase tracking-wide mb-2")
        with ui.row().classes("w-full gap-3 flex-wrap"):
            # 주봉 추세
            weekly_trend = str(row.get("주봉추세", ""))
            _metric_card("주봉 추세", weekly_trend or "—",
                         f"20선 {row.get('주봉20선_상회', '—')}")

            # HMA 추세
            hma_trend = str(row.get("HMA_Trend", ""))
            hma_on = str(row.get("HMA_On", ""))
            _metric_card("HMA 추세", hma_trend or "—",
                         f"HMA 위: {hma_on}" if hma_on else "")

            # SuperTrend
            st_dir = safe_float(row.get("SUPERTREND_DIR", 0))
            st_val = safe_float(row.get("SUPERTREND_VAL", 0))
            _metric_card("SuperTrend",
                         "매수 ↑" if st_dir > 0 else "매도 ↓",
                         f"기준: {int(st_val):,}" if st_val > 0 else "",
                         st_dir > 0)

            # MA20 위치
            above_ma20 = str(row.get("Above_MA20", ""))
            _metric_card("MA20 위치",
                         "상회 ✓" if above_ma20 == "1" else "하회 ✗",
                         "단기 추세 양호" if above_ma20 == "1" else "단기 추세 약화",
                         above_ma20 == "1")

        ui.separator().classes("my-3")

        # 3행: 변동성 & 거래량
        ui.label("변동성 · 거래량").classes("text-xs text-gray-500 uppercase tracking-wide mb-2")
        with ui.row().classes("w-full gap-3 flex-wrap"):
            # V-Power
            _metric_card("V-Power", f"{v_power:.2f}",
                         "매수세 우위" if v_power > 0 else "매도세 우위",
                         v_power > 0)

            # TTM 스퀴즈
            ttm = str(row.get("TTM_SQUEEZE", ""))
            ttm_cnt = safe_float(row.get("TTM_SQUEEZE_CNT", 0))
            _metric_card("TTM 스퀴즈",
                         f"{'압축 중' if ttm == '1' else '해제'} ({int(ttm_cnt)}봉)",
                         "에너지 축적 중" if ttm == "1" else "방향 결정",
                         ttm == "1")

            # 볼린저밴드 폭
            bb_bw = safe_float(row.get("BB_BW", 0))
            bb_exp = str(row.get("BB_Expanding", ""))
            _metric_card("BB 밴드폭", f"{bb_bw:.1f}%",
                         "확장 중" if bb_exp == "1" else "수축 중",
                         bb_exp == "1")

            # 이격도
            gap = safe_float(row.get("이격도", 0))
            _metric_card("이격도", f"{gap:+.1f}%",
                         "괴리 주의" if abs(gap) > 5 else "정상 범위",
                         abs(gap) <= 5)

            # 거래강도
            vol_str = safe_float(row.get("거래강도", 0))
            _metric_card("거래강도", f"{vol_str:.2f}",
                         "매수 우위" if vol_str > 1 else "매도 우위",
                         vol_str > 1)

            # 거래대금
            vol_amt = safe_float(row.get("거래대금(억원)", 0))
            _metric_card("거래대금", f"{vol_amt:,.0f}억",
                         "활발" if vol_amt > 500 else "보통" if vol_amt > 100 else "한산")

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
    #  AI 점수 구성
    # ═══════════════════════════════════════
    struct_score = safe_float(row.get("STRUCT_SCORE", 0))
    timing_score = safe_float(row.get("TIMING_SCORE", 0))
    ml_score = safe_float(row.get("ML_SCORE", 0))
    trigger_score = safe_float(row.get("TRIGGER_SCORE", 0))

    if any([struct_score, timing_score, ml_score]):
        with ui.card().classes("w-full p-5 mt-4 bg-[#0d1b2a] border border-gray-700 rounded-2xl"):
            ui.label("🧠 AI 점수 구성").classes("text-lg font-bold text-white mb-3")
            with ui.row().classes("w-full gap-3 flex-wrap"):
                _metric_card("구조 점수", f"{struct_score:.1f}", "차트 패턴 + 추세")
                _metric_card("타이밍 점수", f"{timing_score:.0f}", "진입 시점 적절성")
                _metric_card("ML 점수", f"{ml_score:.1f}", "LSTM + XGBoost")
                _metric_card("트리거 점수", f"{trigger_score:.1f}", "매수 조건 충족도")

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


# ═══════════════════════════════════════════════════════════
#  [v2] 종목 상세 v2 — Full Dashboard 렌더링
#  활성화: USE_STOCK_DETAIL_V2=1 환경변수 또는 ?v2=1 쿼리스트링
#  실패 시 자동으로 v1 fallback (render_stock_page 상단에서 처리)
# ═══════════════════════════════════════════════════════════
async def _render_stock_page_v2(code: str, store):
    """[v2] stock_detail_v2 풀 대시보드로 렌더링."""
    from components.stock_detail_v2 import render_stock_detail_v2_partial

    # NiceGUI Quasar 컨테이너 폭 제약 해제 + 다크 배경 + viewport (v2 레이아웃 요구사항)
    ui.add_head_html('''
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          body { background: #0F1117; margin: 0; padding: 0; }
          .q-page-container, .q-page, .nicegui-content { max-width: none !important; }
          .nicegui-content { padding: 16px !important; gap: 0 !important; }
          .nicegui-content > * { width: 100% !important; max-width: none !important; }
          @media (max-width: 768px) {
            .nicegui-content { padding: 8px !important; }
          }
        </style>
    ''')

    # 데이터 로드
    if not store.loaded:
        await run_sync(store.refresh)
    df = store.scored

    if df.empty:
        ui.label("⚠️ 데이터 로드 실패").classes("text-yellow-400 text-lg p-8")
        return

    # 종목 검색
    code_norm = str(code).zfill(6)
    row_df = df[df["종목코드"].astype(str).str.zfill(6) == code_norm]
    if row_df.empty:
        with ui.column().classes("w-full items-center justify-center p-12"):
            ui.label("❌ 종목을 찾을 수 없습니다").classes("text-2xl text-red-400")
            ui.label(f"종목코드: {code_norm}").classes("text-gray-400 mt-2")
            ui.button("← 메인으로", on_click=lambda: ui.navigate.to("/")).props(
                "outline"
            ).classes("mt-4")
        return

    row = row_df.iloc[0].to_dict()

    # OG 메타태그 (소셜 미리보기 — v1과 동일)
    name = str(row.get("종목명", code_norm))
    score = safe_float(row.get("DISPLAY_SCORE", 0))
    route = str(row.get("ROUTE", "NEUTRAL"))
    close = int(nz_num(row.get("종가", 0)))
    _og_meta(name, code_norm, score, route, close)

    # 상단 네비게이션 헤더 (v1 스타일 유지 — 사용자에게 일관된 네비)
    with ui.row().classes(
        "w-full items-center justify-between px-4 py-3 mb-2 "
        "bg-gradient-to-r from-[#1a1a2e] via-[#16213e] to-[#0f3460] rounded-xl"
    ):
        with ui.row().classes("items-center gap-3 cursor-pointer").on(
            "click", lambda: ui.navigate.to("/")
        ):
            ui.label("💎 SwingPicker").classes(
                "text-xl font-bold text-transparent bg-clip-text "
                "bg-gradient-to-r from-blue-400 to-purple-400"
            ).style("font-family:Outfit,sans-serif")
        with ui.row().classes("gap-2"):
            ui.button(
                "← 전체 종목", on_click=lambda: ui.navigate.to("/")
            ).props("flat dense").classes("text-white text-sm")
            ui.button(
                "🔐 로그인", on_click=lambda: ui.navigate.to("/login")
            ).props("flat dense").classes("text-white text-sm")

    # 랭크/총개수 계산
    try:
        rank = int(row.get("LDY_RANK", 0) or 0)
    except Exception:
        rank = 0
    total = len(df)

    # v2 풀 대시보드 렌더링
    render_stock_detail_v2_partial(
        row,
        rank=rank,
        total=total,
        timestamp=str(row.get("기준일", "")),
        compare_name="",  # 운영에서는 일반적으로 빈값 (테스트 라우트만 자동 매핑)
    )

    # 푸터 (v1 스타일 유지)
    with ui.column().classes("w-full items-center mt-8 mb-4 gap-1"):
        ui.label(
            f"📅 데이터 기준: {store.data_ts} · "
            f"분석일: {datetime.now().strftime('%Y-%m-%d')}"
        ).classes("text-xs text-gray-500")
        ui.label(
            "⚠️ 본 자료는 투자 권유가 아닌 AI 분석 참고 자료입니다. "
            "투자 판단은 본인 책임."
        ).classes("text-xs text-gray-600")
        ui.label("© SwingPicker by LDY Pro Trader").classes(
            "text-xs text-gray-600 mt-1"
        )
