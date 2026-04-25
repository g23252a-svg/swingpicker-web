# -*- coding: utf-8 -*-
"""
tab_market.py — 📊 시장 현황 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
공포/탐욕 지수, 매크로 스파크라인, 섹터 트리맵/모멘텀
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from nicegui import ui

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from async_helpers import run_sync
except ImportError:
    async def run_sync(fn, *a, **kw):
        return fn(*a, **kw)

FDR_OK = False
fdr = None
try:
    import FinanceDataReader as _fdr
    fdr = _fdr
    FDR_OK = True
except ImportError:
    pass

from chart_components import (
    plot_fear_greed_gauge, plot_sector_treemap, plot_sector_momentum_bar,
)
from shared_utils import safe_float

_logger = logging.getLogger(__name__)

# 매크로 캐시 (1시간 TTL)
_MACRO_CACHE: dict = {}
_MACRO_CACHE_TIME: dict = {}


# ── UI 유틸 ──

def _hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """#RRGGBB → rgba(r,g,b,alpha). Plotly가 8자리 hex를 거부하므로 변환 필요.

    기존 `#10B98122` 같은 8자리 hex (뒤 2자리가 알파) 도 허용하여 알파를 추출한다.
    파싱 실패 시 원문 그대로 반환 (Plotly가 named color로 처리하도록).
    """
    try:
        h = (hex_color or "").lstrip("#")
        if len(h) == 8:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            alpha = int(h[6:8], 16) / 255.0
        elif len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        elif len(h) == 3:
            r = int(h[0] * 2, 16); g = int(h[1] * 2, 16); b = int(h[2] * 2, 16)
        else:
            return hex_color
        return f"rgba({r},{g},{b},{alpha:.3f})"
    except Exception:
        return hex_color


def _section_title(text):
    ui.label(text).classes("text-lg font-bold text-white mt-6 mb-2 border-b border-gray-700 pb-2")


def _metric_card(title, value, delta="", positive=True):
    with ui.card().classes("p-4 min-w-[140px] bg-[#1a1a2e] border border-gray-700 rounded-xl"):
        ui.label(title).classes("text-xs text-gray-400 uppercase tracking-wide")
        ui.label(str(value)).classes("text-xl font-bold text-white mt-1")
        if delta:
            color = "text-green-400" if positive else "text-red-400"
            ui.label(str(delta)).classes(f"text-sm {color} mt-0.5")


def _plotly_dark(fig, height=300):
    if fig:
        fig.update_layout(
            height=height, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
            margin=dict(t=30, b=10, l=10, r=10),
        )
    return fig


# ── 매크로 스파크라인 ──

def _render_macro_sparklines():
    if not FDR_OK:
        return

    MACRO_TICKERS = [
        ("USD/KRW",    "USD/KRW",   "#F59E0B"),
        ("NASDAQ",     "IXIC",      "#3B82F6"),
        ("KOSPI",      "KS11",      "#10B981"),
        ("US 10Y",     "US10YT",    "#E040FB"),  # FDR 내부 매핑: US10YT → ^TNX
    ]

    with ui.card().classes("w-full p-3 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
        ui.label("🌍 글로벌 매크로").classes("text-xs text-gray-400 mb-2")
        with ui.row().classes("w-full gap-3 flex-wrap"):
            for label, ticker, color in MACRO_TICKERS:
                _spark_card(label, ticker, color)


def _spark_card(label: str, ticker: str, color: str):
    with ui.card().classes("flex-1 min-w-[140px] p-2 bg-[#1a1a2e] border border-gray-700/50 rounded-lg"):
        val_label = ui.label("—").classes("text-sm font-bold text-white")
        chg_label = ui.label("—").classes("text-xs")
        chart_slot = ui.column().classes("w-full")

        async def _load():
            try:
                now = time.time()
                if ticker in _MACRO_CACHE and (now - _MACRO_CACHE_TIME.get(ticker, 0)) < 3600:
                    d = _MACRO_CACHE[ticker]
                else:
                    start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
                    d = await run_sync(fdr.DataReader, ticker, start)
                    if d is not None and not d.empty:
                        d = d.tail(20)
                        _MACRO_CACHE[ticker] = d
                        _MACRO_CACHE_TIME[ticker] = now

                if d is None or d.empty:
                    val_label.set_text("N/A")
                    return
                last = float(d["Close"].iloc[-1])
                prev = float(d["Close"].iloc[-2]) if len(d) > 1 else last
                chg = (last - prev) / prev * 100 if prev else 0

                if ticker in ("USD/KRW",):
                    fmt = f"{last:,.1f}"
                elif "10Y" in ticker:
                    fmt = f"{last:.3f}%"
                else:
                    fmt = f"{last:,.2f}"

                val_label.set_text(f"{label}: {fmt}")
                chg_label.set_text(f"{chg:+.2f}%")
                chg_label.classes(replace="text-xs text-green-400" if chg >= 0 else "text-xs text-red-400")

                if PLOTLY_OK:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(d))),
                        y=d["Close"].tolist(),
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        fill="tozeroy",
                        fillcolor=_hex_to_rgba(color, alpha=0.13),
                        showlegend=False,
                    ))
                    fig.update_layout(
                        height=50, margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                    )
                    chart_slot.clear()
                    with chart_slot:
                        ui.plotly(fig).classes("w-full")
            except Exception as e:
                _logger.warning(f"⚠️ 매크로 조회 실패 ({ticker}): {e}")
                val_label.set_text(f"{label}: —")

        async def _safe_load():
            await asyncio.sleep(0.5)
            try:
                if chart_slot.is_deleted:
                    return
            except AttributeError:
                pass
            await _load()

        asyncio.create_task(_safe_load())


# ── 공포/탐욕 지수 ──

def _get_fear_greed(df):
    if df.empty or "DISPLAY_SCORE" not in df.columns:
        return 50, "데이터 부족"
    avg = df["DISPLAY_SCORE"].mean()
    score = min(max(avg, 0), 100)
    if score >= 80: label = "극단적 탐욕"
    elif score >= 60: label = "탐욕"
    elif score >= 40: label = "중립"
    elif score >= 20: label = "공포"
    else: label = "극단적 공포"
    return score, label


# ═══════════════════════════════════════════════════
# [v22 UI] 오늘의 결론 Hero 카드 — 첫 화면 1초 답변
# ═══════════════════════════════════════════════════
def _render_today_hero(df: pd.DataFrame):
    """첫 화면 최상단 Hero 카드.
    
    3가지 시나리오:
      A) TOP_PICK >= 1   → 🏆 추천 카드 (큼직)
      B) TOP_PICK = 0, ARMED/ATTACK 있음 → ⏸️ 관찰 모드 + 가까운 후보
      C) 활성 후보도 0  → 🔴 매수 신호 없음
    
    안전 설계:
      - try/except로 에러 시 카드만 안 띄우고 진행
      - 컬럼 누락 graceful fallback
    """
    try:
        if df is None or df.empty:
            return
        
        # TOP_PICK 종목 — 강건 파서 (1, 1.0, "True", "Y" 등)
        top_picks = pd.DataFrame()
        if 'TOP_PICK' in df.columns:
            tp_str = df['TOP_PICK'].astype(str).str.strip().str.upper()
            tp_mask = tp_str.isin(['1', '1.0', 'TRUE', 'Y', 'YES'])
            top_picks = df[tp_mask].copy()
        
        n_top = len(top_picks)
        
        # ─────────────────────────────────────────────
        # 시나리오 A: TOP_PICK >= 1 → 🏆 추천 카드
        # ─────────────────────────────────────────────
        if n_top >= 1:
            # AGGRESSIVE / STABLE 분류
            if 'TOP_PICK_TYPE' in top_picks.columns:
                tp_type_str = top_picks['TOP_PICK_TYPE'].astype(str).str.upper()
                n_agg = (tp_type_str == 'AGGRESSIVE').sum()
                n_stb = (tp_type_str == 'STABLE').sum()
            else:
                n_agg = 0
                n_stb = 0
            
            # 헤더 카드 — 결론 한 줄
            with ui.card().classes(
                "w-full p-5 mb-4 rounded-xl "
                "bg-gradient-to-r from-[#0a3d2a] via-[#0d5440] to-[#0a3d2a] "
                "border-2 border-emerald-500/50"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.column().classes("gap-1"):
                        ui.label("🟢 오늘 신규 진입 가능").classes(
                            "text-lg font-bold text-emerald-300"
                        )
                        type_summary = []
                        if n_agg > 0: type_summary.append(f"🔥 공격형 {n_agg}")
                        if n_stb > 0: type_summary.append(f"💎 안정형 {n_stb}")
                        if not type_summary: type_summary.append(f"⭐ 추천 {n_top}")
                        ui.label(f"TOP_PICK {n_top}개  ·  " + " / ".join(type_summary)).classes(
                            "text-sm text-emerald-100"
                        )
                    ui.label(f"{n_top}").classes(
                        "text-5xl font-black text-emerald-300"
                    )
            
            # TOP_PICK 카드들 (최대 3개)
            top_picks_sorted = top_picks.sort_values('ELITE_SCORE', ascending=False).head(3) \
                if 'ELITE_SCORE' in top_picks.columns else top_picks.head(3)
            
            with ui.row().classes("w-full gap-3 flex-wrap mb-4"):
                for rank, (_, row) in enumerate(top_picks_sorted.iterrows(), 1):
                    name = str(row.get('종목명', 'N/A'))
                    tp_type = str(row.get('TOP_PICK_TYPE', '')).upper()
                    
                    if tp_type == 'AGGRESSIVE':
                        emoji = '🔥'
                        type_label = '공격형'
                        accent = '#EF4444'   # red
                    elif tp_type == 'STABLE':
                        emoji = '💎'
                        type_label = '안정형'
                        accent = '#10B981'   # green
                    else:
                        emoji = '⭐'
                        type_label = '추천'
                        accent = '#F59E0B'   # amber
                    
                    elite = safe_float(row.get('ELITE_SCORE', 0))
                    rr = safe_float(row.get('RR_NOW_TP1', 0))
                    gap = safe_float(row.get('ENTRY_GAP_PCT', 0))
                    amt = safe_float(row.get('추천금액(만원)', 0))
                    ewr = safe_float(row.get('EST_WIN_RATE', 0))
                    
                    buy = safe_float(row.get('추천매수가', 0))
                    tp1 = safe_float(row.get('추천매도가1', 0))
                    stop = safe_float(row.get('손절가', 0))
                    tp1_pct = (tp1 / buy - 1) * 100 if buy > 0 else 0
                    stop_pct = (stop / buy - 1) * 100 if buy > 0 else 0
                    
                    with ui.card().classes(
                        f"flex-1 min-w-[280px] p-4 bg-[#1a1a2e] "
                        f"border-l-4 rounded-xl"
                    ).style(f"border-left-color: {accent}"):
                        # 종목명 + 타입 + 순위
                        with ui.row().classes("w-full items-center gap-2 mb-2"):
                            ui.label(f"{emoji} {rank}순위 · {name}").classes(
                                "text-base font-bold text-white"
                            )
                            ui.badge(f"E{elite:.0f}", color="#3B82F6").classes("text-xs")
                        
                        ui.label(f"{type_label}  ·  RR {rr:.1f}:1  ·  진입갭 {gap:+.1f}%").classes(
                            "text-xs text-gray-400 mb-2"
                        )
                        
                        # 가격 (매수 → 목표 / 손절)
                        if buy > 0 and tp1 > 0:
                            ui.label(f"매수 {int(buy):,} → 목표 {int(tp1):,}  ({tp1_pct:+.1f}%)").classes(
                                "text-sm text-cyan-300"
                            )
                        if stop > 0 and buy > 0:
                            ui.label(f"손절 {int(stop):,}원  ({stop_pct:+.1f}%)").classes(
                                "text-xs text-red-300"
                            )
                        
                        # 추천 비중 + 승률
                        with ui.row().classes("w-full gap-3 mt-2 items-center"):
                            if amt > 0:
                                ui.label(f"💰 {amt:.0f}만원").classes(
                                    "text-sm font-bold text-amber-300"
                                )
                            if ewr > 0:
                                ui.label(f"승률 {ewr*100:.0f}%").classes(
                                    "text-xs text-gray-400"
                                )
            return
        
        # ─────────────────────────────────────────────
        # 시나리오 B/C: TOP_PICK 0건
        # ─────────────────────────────────────────────
        active = pd.DataFrame()
        if 'ROUTE' in df.columns:
            route_upper = df['ROUTE'].astype(str).str.strip().str.upper()
            active = df[route_upper.isin(['ATTACK', 'ARMED'])].copy()
        
        if len(active) > 0 and 'ELITE_SCORE' in active.columns:
            # 시나리오 B: 관찰 모드
            top_cand = active.sort_values('ELITE_SCORE', ascending=False).iloc[0]
            cand_name = str(top_cand.get('종목명', 'N/A'))
            cand_score = safe_float(top_cand.get('ELITE_SCORE', 0))
            cand_route = str(top_cand.get('ROUTE', ''))
            cand_tp1 = safe_float(top_cand.get('TP1_PCT', 0))
            cand_buy = safe_float(top_cand.get('추천매수가', 0))
            cand_target = safe_float(top_cand.get('추천매도가1', 0))
            
            # 부족한 점수 진단
            struct = safe_float(top_cand.get('STRUCT_SCORE', 0))
            timing = safe_float(top_cand.get('TIMING_SCORE', 0))
            balance = safe_float(top_cand.get('BALANCE_SCORE', 0))
            
            shortfall_msg = ""
            if struct > 0 and struct < 80:
                shortfall_msg = f"STRUCT {80 - struct:.1f}점 부족 (80↑ 필요)"
            elif cand_score < 75:
                shortfall_msg = f"ELITE {75 - cand_score:.1f}점 부족 (75↑ 필요)"
            elif timing > 0 and timing < 70:
                shortfall_msg = f"TIMING {70 - timing:.1f}점 부족"
            else:
                shortfall_msg = "조건 일부 미달"
            
            # 헤더 카드 — 관찰 모드
            with ui.card().classes(
                "w-full p-5 mb-4 rounded-xl "
                "bg-gradient-to-r from-[#3d2a0a] via-[#544013] to-[#3d2a0a] "
                "border-2 border-amber-500/50"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.column().classes("gap-1"):
                        ui.label("⏸️ 오늘은 관찰 모드").classes(
                            "text-lg font-bold text-amber-300"
                        )
                        ui.label(f"정식 추천 0건  ·  활성 후보 {len(active)}종목").classes(
                            "text-sm text-amber-100"
                        )
                    ui.label("0").classes(
                        "text-5xl font-black text-amber-300/60"
                    )
            
            # 가까운 후보 카드
            with ui.card().classes(
                "w-full p-4 mb-4 rounded-xl "
                "bg-[#1a1a2e] border border-amber-700/40"
            ):
                ui.label(f"💡 가장 가까운 후보").classes("text-xs text-gray-400 mb-2")
                
                with ui.row().classes("w-full items-center gap-3 mb-2"):
                    ui.label(f"👀 {cand_name}").classes(
                        "text-lg font-bold text-white"
                    )
                    ui.badge(f"E{cand_score:.1f}", color="#F59E0B").classes("text-xs")
                    ui.badge(cand_route, color="#3B82F6").classes("text-xs")
                
                ui.label(f"└ {shortfall_msg}").classes(
                    "text-sm text-amber-300"
                )
                
                if cand_buy > 0 and cand_target > 0:
                    ui.label(
                        f"매수 {int(cand_buy):,} → 목표 {int(cand_target):,}  (+{cand_tp1:.1f}%)"
                    ).classes("text-sm text-cyan-400 mt-1")
                
                ui.label(
                    "시스템이 신중하게 골라서 오늘은 통과한 종목이 없어요. "
                    "무리한 진입은 자제하시고 다음 기회를 기다리세요."
                ).classes("text-xs text-gray-500 mt-2 italic")
        else:
            # 시나리오 C: 매수 신호 없음
            with ui.card().classes(
                "w-full p-5 mb-4 rounded-xl "
                "bg-gradient-to-r from-[#3d0a0a] via-[#541313] to-[#3d0a0a] "
                "border-2 border-red-500/50"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.column().classes("gap-1"):
                        ui.label("🔴 오늘은 매수 신호 없음").classes(
                            "text-lg font-bold text-red-300"
                        )
                        ui.label(
                            "ATTACK/ARMED 종목 0건 — 시장 약세. 다음 거래일 대기."
                        ).classes("text-sm text-red-100")
                    ui.icon("warning", size="48px").classes("text-red-400")
    
    except Exception as _e:
        # Hero 카드 실패해도 나머지 화면은 정상 표시
        _logger.warning(f"Hero 카드 렌더 실패 (silent fail): {_e}")


# ── 메인 렌더 ──

def render_tab_market(df):
    """Tab 1: 시장 현황"""
    import os, json

    # ═══════════════════════════════════════════════════
    # [v22 UI] 오늘의 결론 Hero 카드 — 가장 먼저 (1초 답변)
    # ═══════════════════════════════════════════════════
    _render_today_hero(df)

    fg_score, fg_label = _get_fear_greed(df)
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # run_meta 로드
    meta = {}
    try:
        mp = os.path.join(DATA_DIR, "run_meta_latest.json")
        if os.path.exists(mp):
            with open(mp, 'r') as f:
                meta = json.load(f)
    except Exception:
        pass

    # ═══════════════════════════════════════════════════
    # [v22 UI Step A] 12개 분석 섹션을 expansion으로 접기 — 첫 화면 깔끔
    # 펼치면: 매크로 스파크라인 / 엔진 상태 / ELITE Top / 시장 현황 /
    #         공포탐욕 / 섹터 / 모멘텀 / 지표 승률 / 조합 / 매칭 / ROUTE / 예측력
    # ═══════════════════════════════════════════════════
    with ui.expansion("📊 시장 상세 분석 보기 (매크로 · 엔진 · 섹터 · 지표 승률)",
                       icon="analytics").classes(
        "w-full mb-4 bg-[#0d0d1a] border border-gray-700/50 rounded-xl"
    ):
        _render_macro_sparklines()

        # ── [v21.3] 엔진 상태 요약 ──
        macro_risk = meta.get("macro_risk", "—")
        breadth = meta.get("market_breadth", 0)
        confidence = meta.get("confidence_score", 0)
        max_route = meta.get("max_allowed_route", "—")
        macro_msg = meta.get("macro_msg", "")

        risk_color = {"NORMAL": "#10B981", "CAUTION": "#F59E0B", "WARNING": "#EF4444", "CRITICAL": "#DC2626"}.get(macro_risk, "#6B7280")
        risk_kr = {"NORMAL": "정상", "CAUTION": "주의", "WARNING": "경고", "CRITICAL": "위험"}.get(macro_risk, macro_risk)

        with ui.card().classes("w-full p-4 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
            ui.label("🛡️ 엔진 상태").classes("text-xs text-gray-400 mb-2")
            with ui.row().classes("w-full gap-4 flex-wrap"):
                with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                    ui.label("매크로 리스크").classes("text-xs text-gray-400")
                    ui.label(f"{'🟢' if macro_risk == 'NORMAL' else '🟡' if macro_risk == 'CAUTION' else '🔴'} {risk_kr}").classes("text-lg font-bold").style(f"color:{risk_color}")
                    if macro_msg:
                        ui.label(macro_msg).classes("text-xs text-gray-500")

                with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                    ui.label("시장 Breadth").classes("text-xs text-gray-400")
                    bc = "#10B981" if breadth >= 60 else "#F59E0B" if breadth >= 40 else "#EF4444"
                    ui.label(f"{breadth:.1f}%").classes("text-lg font-bold").style(f"color:{bc}")
                    ui.label("상승 종목 비율").classes("text-xs text-gray-500")

                with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                    ui.label("엔진 신뢰도").classes("text-xs text-gray-400")
                    cc = "#10B981" if confidence >= 80 else "#F59E0B" if confidence >= 50 else "#EF4444"
                    ui.label(f"{confidence:.0f}/100").classes("text-lg font-bold").style(f"color:{cc}")
                    ui.label(f"최대허용: {max_route}").classes("text-xs text-gray-500")

                # ELITE/TOP_PICK 요약
                if "ELITE_SCORE" in df.columns:
                    elite_avg = df["ELITE_SCORE"].mean()
                    tp_count = int(df.get("TOP_PICK", pd.Series(0)).sum()) if "TOP_PICK" in df.columns else 0
                    with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                        ui.label("오늘의 ELITE").classes("text-xs text-gray-400")
                        ui.label(f"🏆 {tp_count}종목").classes("text-lg font-bold text-yellow-400")
                        ui.label(f"평균 ELITE {elite_avg:.0f}").classes("text-xs text-gray-500")

        # ── [v21.3] 오늘의 Top 추천 — ELITE 기준 ──
        try:
            if "ELITE_SCORE" in df.columns:
                _top_df = df.copy()
                # TOP_PICK 우선, 없으면 ELITE 상위
                if "TOP_PICK" in _top_df.columns:
                    _picks = _top_df[_top_df["TOP_PICK"].astype(int) == 1].nlargest(3, "ELITE_SCORE")
                    if len(_picks) < 3:
                        _extra = _top_df[~_top_df.index.isin(_picks.index)].nlargest(3 - len(_picks), "ELITE_SCORE")
                        _picks = pd.concat([_picks, _extra])
                else:
                    _picks = _top_df.nlargest(3, "ELITE_SCORE")

                if not _picks.empty:
                    with ui.card().classes("w-full p-4 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
                        ui.label("🏆 오늘의 ELITE Top").classes("text-xs text-gray-400 mb-2")
                        with ui.row().classes("w-full gap-3 flex-wrap"):
                            for _, s in _picks.iterrows():
                                route = str(s.get("ROUTE", ""))
                                route_icon = {"ATTACK": "🚀", "ARMED": "🔫", "CARRY": "📌"}.get(route, "👀")
                                elite = safe_float(s.get("ELITE_SCORE", 0))
                                close = safe_float(s.get("종가", 0))
                                tp1 = safe_float(s.get("추천매도가1", 0))
                                rr = safe_float(s.get("RR_NOW_TP1", 0))
                                wr = safe_float(s.get("EST_WIN_RATE", 0))
                                bal = safe_float(s.get("BALANCE_SCORE", 0))
                                tp_flag = "🏆 " if int(s.get("TOP_PICK", 0)) == 1 else ""
                                tp1_pct = (tp1 / close - 1) * 100 if close > 0 else 0

                                with ui.card().classes("flex-1 min-w-[200px] p-3 bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                                    with ui.row().classes("items-center gap-2"):
                                        ui.label(f"{route_icon} {tp_flag}{s.get('종목명', '')}").classes("text-white font-bold text-sm")
                                        ui.badge(f"E{elite:.0f}", color="#10B981" if elite >= 80 else "#3B82F6").classes("text-xs")
                                    ui.label(f"S{safe_float(s.get('STRUCT_SCORE', 0)):.0f} T{safe_float(s.get('TIMING_SCORE', 0)):.0f} AI{safe_float(s.get('AI_SCORE', 0)):.0f} | 균형 {bal:.0f}").classes("text-xs text-gray-400 mt-1")
                                    ui.label(f"{close:,.0f} → {tp1:,.0f} ({tp1_pct:+.1f}%) | RR {rr:.1f}:1 | 승률 {wr * 100:.0f}%").classes("text-xs text-cyan-400")
        except Exception as _te:
            _logger.warning(f"Top 추천 렌더 실패: {_te}")

        _section_title("📡 시장 현황")
        with ui.row().classes("w-full gap-4 flex-wrap"):
            fg_icon = "🟢" if fg_score >= 50 else "🔴"
            _metric_card("시장 심리", f"{fg_icon} {fg_label}", f"지수: {fg_score:.0f}/100", fg_score >= 50)

            if "ret_1d_%" in df.columns:
                avg_ret = df.head(20)["ret_1d_%"].mean()
                _metric_card("Top20 평균 수익률", f"{avg_ret:+.2f}%", "전일 대비", avg_ret >= 0)

            total = len(df)
            armed = len(df[df.get("ROUTE", pd.Series()).str.contains("ARMED|ATTACK", na=False)]) if "ROUTE" in df.columns else 0
            _metric_card("분석 종목", f"{total}개", f"ARMED/ATTACK: {armed}개")

        _section_title("🌡️ 공포/탐욕 & 주도 섹터")
        with ui.row().classes("w-full gap-4 flex-wrap items-start"):
            with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
                fig_g = plot_fear_greed_gauge(fg_score)
                if fig_g:
                    ui.plotly(_plotly_dark(fig_g, 280)).classes("w-full")

            with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
                ui.label("🔥 오늘의 주도 섹터").classes("text-sm font-bold text-white mb-2")
                if "업종" in df.columns:
                    fig_m = plot_sector_treemap(df.head(50))
                    if fig_m:
                        ui.plotly(_plotly_dark(fig_m, 280)).classes("w-full")
                    else:
                        ui.label("섹터 데이터 부족").classes("text-gray-500")

        _section_title("🚀 섹터 모멘텀 Top 10")
        fig_mom = plot_sector_momentum_bar(df)
        if fig_mom and len(fig_mom.data) > 0:
            ui.plotly(_plotly_dark(fig_mom, 350)).classes("w-full")
        else:
            ui.label("모멘텀 데이터 부족").classes("text-gray-500")

        # ── [v21.3] 지표별 승률 분석 ──
        if "EST_WIN_RATE" in df.columns and len(df) >= 20:
            _section_title("📊 지표별 승률 분석")

            # 최적 조합 표시
            try:
                opt_path = os.path.join(DATA_DIR, "optimal_filter_latest.json")
                if os.path.exists(opt_path):
                    with open(opt_path, 'r') as f:
                        opt = json.load(f)
                    best = opt.get("best", {})
                    meta = opt.get("meta", {})
                    if best:
                        with ui.card().classes("w-full p-4 bg-[#0a1628] border border-yellow-600/50 rounded-xl mb-4"):
                            ui.label("🎯 데이터 기반 최적 조합 (자동 탐색)").classes("text-sm font-bold text-yellow-400 mb-2")

                            # 승률 최적
                            with ui.row().classes("w-full gap-4 flex-wrap items-center"):
                                ui.label("🛡️ 안정형 (자주 이기는 조합):").classes("text-xs text-gray-400")
                                ui.label(
                                    f"S≥{best.get('S_min', 0)} T≥{best.get('T_min', 0)} AI≥{best.get('AI_min', 0)} + {'+'.join(best.get('routes', []))}"
                                ).classes("text-sm font-bold text-white")
                                ui.badge(f"승률 {best.get('win_rate', 0)}%", color="#10B981").classes("text-sm px-2 py-1")
                                ui.badge(f"기대수익 {best.get('ev', 0):+.1f}", color="#6B7280").classes("text-sm px-2 py-1")
                                ui.badge(f"{best.get('n', 0)}건", color="#6B7280").classes("text-sm px-2 py-1")

                            # EV 최적
                            best_ev = opt.get("best_ev", {})
                            if best_ev and best_ev != best:
                                with ui.row().classes("w-full gap-4 flex-wrap items-center mt-1"):
                                    ui.label("💰 수익형 (크게 버는 조합):").classes("text-xs text-gray-400")
                                    ui.label(
                                        f"S≥{best_ev.get('S_min', 0)} T≥{best_ev.get('T_min', 0)} AI≥{best_ev.get('AI_min', 0)} + {'+'.join(best_ev.get('routes', []))}"
                                    ).classes("text-sm font-bold text-white")
                                    ui.badge(f"기대수익 {best_ev.get('ev', 0):+.1f}", color="#F59E0B").classes("text-sm px-2 py-1")
                                    ui.badge(f"승률 {best_ev.get('win_rate', 0)}%", color="#6B7280").classes("text-sm px-2 py-1")
                                    ui.badge(f"수익 {best_ev.get('avg_ret', 0):+.1f}%", color="#3B82F6").classes("text-sm px-2 py-1")
                                    ui.badge(f"{best_ev.get('n', 0)}건", color="#6B7280").classes("text-sm px-2 py-1")

                            ui.label(
                                f"전체 승률 {meta.get('total_win_rate', 0)}% 대비 승률 +{best.get('win_rate', 0) - meta.get('total_win_rate', 0):.1f}%p 초과 | "
                                f"{meta.get('matched_days', 0)}일 × {meta.get('total_trades', 0):,}건 분석 | 보유 {meta.get('horizon', 3)}일"
                            ).classes("text-xs text-gray-500 mt-2")

                        # [v21.3] 통합 조합 성과 테이블
                        wr_combos = opt.get("top_combos", [])[:5]
                        ev_combos = opt.get("top_combos_ev", [])[:5]

                        seen = set()
                        merged = []
                        for c in wr_combos + ev_combos:
                            key = f"{c['S_min']}-{c['T_min']}-{c['AI_min']}-{'+'.join(c.get('routes',[]))}"
                            if key not in seen:
                                seen.add(key)
                                merged.append(c)
                        merged.sort(key=lambda x: -x.get("ev", 0))

                        if merged:
                            best_ev = opt.get("best_ev", {})
                            best_wr_key = f"{best.get('S_min')}-{best.get('T_min')}-{best.get('AI_min')}"
                            best_ev_key = f"{best_ev.get('S_min')}-{best_ev.get('T_min')}-{best_ev.get('AI_min')}"

                            ui.label("📋 조합별 성과 비교").classes("text-sm font-bold text-white mb-2")
                            combo_rows = []
                            for c in merged[:8]:
                                key = f"{c['S_min']}-{c['T_min']}-{c['AI_min']}"
                                tag = ""
                                if key == best_wr_key:
                                    tag = "🛡️"
                                if key == best_ev_key:
                                    tag = "💰" if not tag else "🛡️💰"

                                combo_rows.append({
                                    "tag": tag,
                                    "combo": f"S≥{c['S_min']} T≥{c['T_min']} AI≥{c['AI_min']}",
                                    "n": c["n"],
                                    "wr": f"{c['win_rate']:.0f}%",
                                    "avg_win": f"+{c.get('avg_win', 0):.1f}%",
                                    "avg_loss": f"-{c.get('avg_loss', 0):.1f}%",
                                    "ev": round(c.get("ev", 0), 1),
                                })

                            ui.table(
                                columns=[
                                    {"name": "tag", "label": "", "field": "tag", "align": "center"},
                                    {"name": "combo", "label": "조합 조건", "field": "combo", "align": "left"},
                                    {"name": "n", "label": "샘플", "field": "n", "align": "center", "sortable": True},
                                    {"name": "wr", "label": "승률", "field": "wr", "align": "center"},
                                    {"name": "avg_win", "label": "이길 때", "field": "avg_win", "align": "center"},
                                    {"name": "avg_loss", "label": "질 때", "field": "avg_loss", "align": "center"},
                                    {"name": "ev", "label": "기대수익", "field": "ev", "align": "center", "sortable": True},
                                ],
                                rows=combo_rows, row_key="combo",
                            ).classes("w-full mb-2").props("dense dark flat bordered")
                            ui.label("🛡️ = 가장 자주 이기는 조합 | 💰 = 1회당 기대수익 최대 조합").classes("text-xs text-gray-500")
                            ui.label("💡 기대수익 = 승률 × 이길 때 − (1−승률) × 질 때").classes("text-xs text-gray-500")

                        # [v21.3] 최적 조합 매칭 종목 리스트 — 상위 조합 순서대로 시도
                        all_combos = opt.get("top_combos", [])
                        ai_col = "AI_SCORE" if "AI_SCORE" in df.columns else "ML_SCORE"
                        matched = pd.DataFrame()
                        used_combo = None

                        for combo in all_combos:
                            s_min = combo.get("S_min", 0)
                            t_min = combo.get("T_min", 0)
                            ai_min = combo.get("AI_min", 0)
                            b_routes = combo.get("routes", [])

                            _matched = df[
                                (df.get("STRUCT_SCORE", pd.Series(0, index=df.index)) >= s_min)
                                & (df.get("TIMING_SCORE", pd.Series(0, index=df.index)) >= t_min)
                                & (df.get(ai_col, pd.Series(0, index=df.index)) >= ai_min)
                                & (df.get("ROUTE", pd.Series("", index=df.index)).isin(b_routes))
                            ]
                            if not _matched.empty:
                                matched = _matched
                                used_combo = combo
                                break

                        if not matched.empty and used_combo:
                            elite_col = "ELITE_SCORE" if "ELITE_SCORE" in matched.columns else "DISPLAY_SCORE"
                            matched = matched.sort_values(elite_col, ascending=False)

                            _uc = used_combo
                            _combo_label = f"S≥{_uc['S_min']} T≥{_uc['T_min']} AI≥{_uc['AI_min']} + {'+'.join(_uc['routes'])}"
                            ui.label(f"🎯 매칭 종목 ({len(matched)}개) — {_combo_label} (승률 {_uc['win_rate']}%)").classes("text-sm font-bold text-yellow-400 mb-2")

                            match_rows = []
                            for _, s in matched.iterrows():
                                _close = safe_float(s.get("종가", 0))
                                _tp1 = safe_float(s.get("추천매도가1", 0))
                                _tp1_pct = (_tp1 / _close - 1) * 100 if _close > 0 else 0
                                _rr = safe_float(s.get("RR_NOW_TP1", 0))
                                _wr = safe_float(s.get("EST_WIN_RATE", 0))
                                _elite = safe_float(s.get("ELITE_SCORE", 0))
                                match_rows.append({
                                    "route": str(s.get("ROUTE", "")),
                                    "name": str(s.get("종목명", "")),
                                    "elite": f"{_elite:.0f}",
                                    "s": f"{safe_float(s.get('STRUCT_SCORE', 0)):.0f}",
                                    "t": f"{safe_float(s.get('TIMING_SCORE', 0)):.0f}",
                                    "ai": f"{safe_float(s.get(ai_col, 0)):.0f}",
                                    "rr": f"{_rr:.1f}",
                                    "wr": f"{_wr * 100:.0f}%",
                                    "close": f"{_close:,.0f}",
                                    "tp1": f"{_tp1:,.0f} ({_tp1_pct:+.1f}%)",
                                })
                            ui.table(
                                columns=[
                                    {"name": "route", "label": "신호", "field": "route", "align": "center"},
                                    {"name": "name", "label": "종목명", "field": "name", "align": "left"},
                                    {"name": "elite", "label": "ELITE", "field": "elite", "align": "center"},
                                    {"name": "s", "label": "S", "field": "s", "align": "center"},
                                    {"name": "t", "label": "T", "field": "t", "align": "center"},
                                    {"name": "ai", "label": "AI", "field": "ai", "align": "center"},
                                    {"name": "rr", "label": "RR", "field": "rr", "align": "center"},
                                    {"name": "wr", "label": "승률", "field": "wr", "align": "center"},
                                    {"name": "close", "label": "현재가", "field": "close", "align": "right"},
                                    {"name": "tp1", "label": "목표가", "field": "tp1", "align": "right"},
                                ],
                                rows=match_rows, row_key="name",
                            ).classes("w-full").props("dense dark flat bordered")
                        else:
                            ui.label("⚠️ 오늘 상위 10개 조합 모두 매칭 종목 없음").classes("text-xs text-gray-500")
            except Exception:
                pass

            with ui.card().classes("w-full p-4 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
                # ROUTE별 통계
                ui.label("🚦 ROUTE별 승률").classes("text-sm font-bold text-white mb-2")
                route_rows = []
                for route in ["ATTACK", "ARMED", "WAIT", "NEUTRAL", "CARRY"]:
                    sub = df[df.get("ROUTE", pd.Series(dtype=str)) == route]
                    if sub.empty:
                        continue
                    wr = sub["EST_WIN_RATE"].mean() * 100
                    elite = sub["ELITE_SCORE"].mean() if "ELITE_SCORE" in sub.columns else 0
                    rr = sub["RR_NOW_TP1"].mean() if "RR_NOW_TP1" in sub.columns else 0
                    route_rows.append({
                        "route": route, "n": f"{len(sub)}종목",
                        "wr": f"{wr:.1f}%", "elite": f"{elite:.0f}", "rr": f"{rr:.2f}"
                    })

                if route_rows:
                    ui.table(
                        columns=[
                            {"name": "route", "label": "ROUTE", "field": "route", "align": "center"},
                            {"name": "n", "label": "종목수", "field": "n", "align": "center"},
                            {"name": "wr", "label": "평균 승률", "field": "wr", "align": "center"},
                            {"name": "elite", "label": "평균 ELITE", "field": "elite", "align": "center"},
                            {"name": "rr", "label": "평균 RR", "field": "rr", "align": "center"},
                        ],
                        rows=route_rows, row_key="route",
                    ).classes("w-full").props("dense dark flat bordered")

            with ui.card().classes("w-full p-4 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
                ui.label("📈 지표별 승률 예측력 (상위20% vs 하위20%)").classes("text-sm font-bold text-white mb-2")
                axes_check = [
                    ("DISPLAY_SCORE", "종합점수"), ("STRUCT_SCORE", "구조(S)"),
                    ("TIMING_SCORE", "타이밍(T)"), ("AI_SCORE", "AI"),
                    ("ELITE_SCORE", "ELITE"), ("BALANCE_SCORE", "밸런스"),
                    ("RR_NOW_TP1", "손익비(RR)"),
                ]
                ax_rows = []
                n20 = max(1, int(len(df) * 0.2))
                for col, name in axes_check:
                    if col not in df.columns:
                        continue
                    top20 = df.nlargest(n20, col)
                    bot20 = df.nsmallest(n20, col)
                    top_wr = top20["EST_WIN_RATE"].mean() * 100
                    bot_wr = bot20["EST_WIN_RATE"].mean() * 100
                    spread = top_wr - bot_wr
                    ax_rows.append({
                        "name": name, "top": f"{top_wr:.1f}%", "bot": f"{bot_wr:.1f}%",
                        "spread": f"{spread:+.1f}%p"
                    })

                if ax_rows:
                    ui.table(
                        columns=[
                            {"name": "name", "label": "지표", "field": "name", "align": "left"},
                            {"name": "top", "label": "상위20% 승률", "field": "top", "align": "center"},
                            {"name": "bot", "label": "하위20% 승률", "field": "bot", "align": "center"},
                            {"name": "spread", "label": "차이", "field": "spread", "align": "center", "sortable": True},
                        ],
                        rows=ax_rows, row_key="name",
                    ).classes("w-full").props("dense dark flat bordered")
                    ui.label("💡 차이가 클수록 해당 지표의 승률 예측력이 강함").classes("text-xs text-gray-500 mt-1")
