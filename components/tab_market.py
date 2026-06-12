# -*- coding: utf-8 -*-
"""
tab_market.py — 📊 시장 현황 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
공포/탐욕 지수, 매크로 스파크라인, 섹터 트리맵/모멘텀
"""
import asyncio
import logging
import math
import re
import time
from datetime import datetime, timedelta

import pandas as pd
from nicegui import ui

# ═══════════════════════════════════════════════════
# [v22 UI Step I] 공통 용어 사전 import
# 시장 탭/종목 탭 양쪽이 같은 함수 사용 → 용어 정합성
# 배포 중 import 경로 꼬여도 화면 죽지 않게 fallback 제공
# ═══════════════════════════════════════════════════
try:
    from components.ui_terms import (
        route_display,
        route_icon,
        pick_type_info,
        kelly_engine_label,
        gap_direction,
        is_truthy_flag,
        is_route_blocked,
        VERDICT_LABELS,
    )
except Exception as _ui_terms_err:
    logging.getLogger(__name__).warning(
        f"ui_terms import 실패, fallback 사용: {_ui_terms_err}"
    )
    def route_display(x):
        _map = {"ATTACK": "🚀 적극 매수", "ARMED": "🎯 매수 준비",
                "WAIT": "⏸️ 관망", "NEUTRAL": "👁️ 중립",
                "CARRY": "📌 보유 관리", "OVERHEAT": "🔥 과열 주의",
                "EXIT_WARNING": "⚠️ 이탈 주의", "BLOCKED": "⛔ 제외"}
        return _map.get(str(x or "").strip().upper(), str(x or ""))
    def route_icon(x):
        _icons = {"ATTACK": "🚀", "ARMED": "🎯", "WAIT": "⏸️",
                  "NEUTRAL": "👁️", "CARRY": "📌"}
        return _icons.get(str(x or "").strip().upper(), "👀")
    def pick_type_info(x):
        t = str(x or "").strip().upper()
        if t == "AGGRESSIVE": return ("🔥", "공격형", "#EF4444")
        if t == "STABLE": return ("💎", "안정형", "#10B981")
        return ("⭐", "추천", "#F59E0B")
    def kelly_engine_label(x):
        s = str(x or "").strip()
        if not s or s.lower() in ("nan", "none"):
            return ("", "", "")
        # [v3.9.10] 회원에게 "v22_calibrated" 같은 내부 이름은 의미 없음
        # → "권장 비중 계산: 정상/보수모드"로 단순화. 내부 코드는 tooltip으로
        # [v3.9.11] 3-tuple — 관리자 tooltip용 raw 엔진명 추가
        if "fallback" in s.lower():
            return ("⚠️ 권장 비중 계산: 보수모드", "text-xs text-red-300", s)
        return ("권장 비중 계산: 정상", "text-xs text-gray-500", s)
    def gap_direction(g):
        try: v = float(g)
        except (TypeError, ValueError): return ""
        if abs(v) < 0.05: return "현재가 일치"
        return "현재가 높음" if v > 0 else "현재가 낮음"
    def is_truthy_flag(v):
        if v is None: return False
        return str(v).strip().upper() in {"1", "1.0", "TRUE", "Y", "YES"}
    def is_route_blocked(r):
        s = str(r or "").strip().upper()
        if not s: return False
        return s not in {"ATTACK", "ARMED", "ALL", "FULL",
                          "TOP_PICK", "ATTACK_ONLY", "ALLOW_ATTACK"}
    VERDICT_LABELS = {
        "OK":           "🟢 오늘 매수 OK",
        "HALF":         "🟠 절반만 매수 권장",
        "BLOCK_ENGINE": "🟠 신규 매수 자제 (엔진 제한)",
        "BLOCK_MARKET": "🔴 오늘 매수 금지 (시장 위험)",
        "OBSERVE":      "⏸️ 오늘은 지켜보세요",
        "NO_SIGNAL":    "🔴 매수 신호 없음",
    }

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


def _safe_number_or_none(value):
    """UI 표시용 숫자 변환. NaN/inf/None은 화면에 노출하지 않는다."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num) or pd.isna(num):
        return None
    return num


def _format_macro_delta(last, prev):
    """글로벌 매크로 카드의 전일 대비 표기 안전화.

    FDR/캐시 데이터가 비어 있거나 NaN이면 `+nan%`를 표시하지 않고
    사용자에게 의미 있는 fallback 문구를 반환한다.
    """
    last_num = _safe_number_or_none(last)
    prev_num = _safe_number_or_none(prev)
    if last_num is None or prev_num is None or prev_num == 0:
        return "전일 대비 —", "text-xs text-gray-500", None
    chg = (last_num - prev_num) / prev_num * 100
    if not math.isfinite(chg) or pd.isna(chg):
        return "전일 대비 —", "text-xs text-gray-500", None
    css = "text-xs text-green-400" if chg >= 0 else "text-xs text-red-400"
    return f"{chg:+.2f}%", css, chg


def _is_market_no_buy_mode(macro_risk=None, max_route=None) -> bool:
    """시장/엔진 상태 때문에 신규매수를 표시상 금지해야 하는지 판정."""
    risk = str(macro_risk or "").strip().upper()
    return risk in {"WARNING", "CRITICAL"} or is_route_blocked(max_route)


def _combo_section_title(is_no_buy_mode: bool) -> str:
    if is_no_buy_mode:
        return "🎯 과거 유사패턴 참고용 조합 (오늘 매수 신호 아님)"
    return "🎯 데이터 기반 최적 조합 (자동 탐색)"


def _match_section_title(is_no_buy_mode: bool, n: int, combo_label: str, win_rate) -> str:
    prefix = "👀 관찰 매칭 종목 · 공식 신규매수 아님" if is_no_buy_mode else "🎯 매칭 종목"
    return f"{prefix} ({int(n)}개) — {combo_label} (승률 {win_rate}%)"


def _blocking_priority_text(macro_risk=None, max_route=None, shortfall_msg="") -> list[str]:
    """가장 가까운 후보 카드에 보여줄 차단 사유 우선순위."""
    lines: list[str] = []
    risk = str(macro_risk or "").strip().upper()
    if risk in {"WARNING", "CRITICAL"}:
        lines.append(f"1순위 차단: 매크로 위험 {risk}")
    elif is_route_blocked(max_route):
        lines.append(f"1순위 차단: 엔진 제한 {route_display(max_route)}")
    if shortfall_msg:
        prefix = "2순위 미달" if lines else "미달 사유"
        lines.append(f"{prefix}: {shortfall_msg}")
    return lines or ["조건 일부 미달"]


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
                last = _safe_number_or_none(d["Close"].iloc[-1])
                prev = _safe_number_or_none(d["Close"].iloc[-2]) if len(d) > 1 else last
                if last is None:
                    val_label.set_text(f"{label}: —")
                    chg_label.set_text("전일 대비 —")
                    chg_label.classes(replace="text-xs text-gray-500")
                    return
                delta_text, delta_css, _chg_value = _format_macro_delta(last, prev)

                if ticker in ("USD/KRW",):
                    fmt = f"{last:,.1f}"
                elif "10Y" in ticker:
                    fmt = f"{last:.3f}%"
                else:
                    fmt = f"{last:,.2f}"

                val_label.set_text(f"{label}: {fmt}")
                chg_label.set_text(delta_text)
                chg_label.classes(replace=delta_css)

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
def _sort_top_picks_for_hero(top_picks: pd.DataFrame) -> pd.DataFrame:
    """[v22 UI Step D2] TOP_PICK 다축 정렬 — Hero 1순위는 실전형으로
    
    우선순위:
      1) LDY_RANK 있으면 그것
      2) IS_NOW_ENTRY desc (지금 진입 가능 우선)
      3) RR_NOW_TP1 desc (손익비 좋은 거 우선)
      4) BALANCE_SCORE desc (3축 균형)
      5) ENTRY_GAP_PCT abs asc (현재가 ~ 진입가 가까운 거)
      6) ELITE_SCORE desc (점수)
    """
    x = top_picks.copy()
    if x.empty:
        return x
    
    # LDY_RANK 우선
    if "LDY_RANK" in x.columns:
        x["_rank"] = pd.to_numeric(x["LDY_RANK"], errors="coerce").fillna(9999)
        return x.sort_values("_rank").drop(columns=["_rank"])
    
    # 다축 정렬 (ui_terms.is_truthy_flag 사용)
    x["_is_now"] = x.get("IS_NOW_ENTRY", "0").apply(is_truthy_flag).astype(int)
    x["_rr"] = pd.to_numeric(x.get("RR_NOW_TP1", 0), errors="coerce").fillna(0)
    x["_bal"] = pd.to_numeric(x.get("BALANCE_SCORE", 0), errors="coerce").fillna(0)
    x["_gap"] = pd.to_numeric(x.get("ENTRY_GAP_PCT", 999), errors="coerce").abs().fillna(999)
    x["_elite"] = pd.to_numeric(x.get("ELITE_SCORE", 0), errors="coerce").fillna(0)
    
    sorted_x = x.sort_values(
        ["_is_now", "_rr", "_bal", "_gap", "_elite"],
        ascending=[False, False, False, True, False]
    )
    # 보조 컬럼 제거
    return sorted_x.drop(columns=["_is_now", "_rr", "_bal", "_gap", "_elite"])


def _render_today_hero(df: pd.DataFrame, meta: dict = None, auth: str = "free"):
    """첫 화면 최상단 Hero 카드.
    
    [v22 UI Step D] meta 인자 추가 — macro risk 기반 verdict
    
    Verdict 매트릭스:
      NORMAL  + TOP_PICK + IS_NOW_ENTRY → 🟢 오늘 매수 OK
      CAUTION + TOP_PICK              → 🟠 절반만 매수 권장 (50% 비중)
      WARNING/CRITICAL                → 🔴 오늘 매수 금지 (관찰만)
      NORMAL/CAUTION + TOP_PICK 0 + 관찰 후보 → ⏸️ 관찰 모드
      관찰 후보도 0                   → 🔴 매수 신호 없음
    
    안전 설계:
      - try/except로 에러 시 카드만 안 띄우고 진행
      - meta 누락이면 NORMAL로 가정 (기존 동작 유지)
      - 컬럼 누락 graceful fallback
    """
    try:
        if df is None or df.empty:
            return
        
        # [Step D] meta 안전 추출
        meta = meta or {}
        macro_risk = str(meta.get("macro_risk", "NORMAL")).upper()
        max_route = str(meta.get("max_allowed_route", "")).upper().strip()
        is_macro_dangerous = macro_risk in ("WARNING", "CRITICAL")
        is_macro_caution = macro_risk == "CAUTION"
        
        # [Step I] max_allowed_route 차단 감지 → ui_terms.is_route_blocked 사용
        # ALLOWED_MAX_ROUTES 세트는 ui_terms.py에 통일됨
        # 변수명은 함수와 충돌 방지 위해 'route_blocked'로 (함수: is_route_blocked)
        route_blocked = is_route_blocked(max_route)
        
        # TOP_PICK 종목 — ui_terms.is_truthy_flag 사용
        top_picks = pd.DataFrame()
        if 'TOP_PICK' in df.columns:
            tp_mask = df['TOP_PICK'].apply(is_truthy_flag)
            top_picks = df[tp_mask].copy()
        
        n_top = len(top_picks)
        
        # ─────────────────────────────────────────────
        # 시나리오 A: TOP_PICK >= 1
        # [Step D + E1] macro risk + max_route 기반 verdict 분기
        # 우선순위:
        #   1. is_macro_dangerous (CRITICAL/WARNING) — 가장 강한 경고 🔴
        #   2. route_blocked (엔진이 ROUTE 제한) — 🟠
        #   3. is_macro_caution (CAUTION) — 🟠 분할 진입
        #   4. NORMAL + 모든 통과 — 🟢
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
            
            # ─── [Step D + E1 + J] verdict 결정 (VERDICT_LABELS 키 사용) ───
            # VERDICT_LABELS는 "이모지 + 텍스트" 통합형이라 헤더에서 직접 사용
            # verdict_emoji는 카드 디자인용으로 별도 유지
            if is_macro_dangerous:
                # 🔴 매크로 위험 — TOP_PICK 있어도 신규 진입 금지
                verdict_emoji = "🔴"
                verdict_text = VERDICT_LABELS.get(
                    "BLOCK_MARKET", "🔴 오늘 매수 금지 (시장 위험)"
                ).replace("🔴 ", "")  # 헤더에서 emoji는 별도 표시
                verdict_subtitle = (
                    f"공식 추천 {n_top}개  ·  하지만 시장 위험 "
                    f"({macro_risk}) — 지켜보기만 권장"
                )
                gradient_from = "#3d0a0a"
                gradient_via = "#541313"
                border_color = "border-red-500/50"
                text_main = "text-red-300"
                text_sub = "text-red-100"
                count_color = "text-red-300/60"
            elif route_blocked:
                # [Step E1] 🟠 엔진이 ROUTE 제한 — 매크로는 정상이지만 신규 진입 X
                verdict_emoji = "🟠"
                verdict_text = VERDICT_LABELS.get(
                    "BLOCK_ENGINE", "🟠 신규 매수 자제 (엔진 제한)"
                ).replace("🟠 ", "")
                verdict_subtitle = (
                    f"공식 추천 {n_top}개  ·  엔진 상태={route_display(max_route)} "
                    f"— 지켜보기만 권장"
                )
                gradient_from = "#3d2a0a"
                gradient_via = "#544013"
                border_color = "border-orange-500/50"
                text_main = "text-orange-300"
                text_sub = "text-orange-100"
                count_color = "text-orange-300"
            elif is_macro_caution:
                # 🟠 시장 주의 — 보수적 분할 진입
                verdict_emoji = "🟠"
                verdict_text = VERDICT_LABELS.get(
                    "HALF", "🟠 절반만 매수 권장"
                ).replace("🟠 ", "")
                verdict_subtitle = (
                    f"공식 추천 {n_top}개  ·  시장 주의 "
                    f"({macro_risk}) — 비중 절반으로 축소"
                )
                gradient_from = "#3d2a0a"
                gradient_via = "#544013"
                border_color = "border-orange-500/50"
                text_main = "text-orange-300"
                text_sub = "text-orange-100"
                count_color = "text-orange-300"
            else:
                # 🟢 정상 — 신규 진입 가능
                verdict_emoji = "🟢"
                verdict_text = VERDICT_LABELS.get(
                    "OK", "🟢 오늘 매수 OK"
                ).replace("🟢 ", "")
                type_summary = []
                if n_agg > 0: type_summary.append(f"🔥 공격형 {n_agg}")
                if n_stb > 0: type_summary.append(f"💎 안정형 {n_stb}")
                if not type_summary: type_summary.append(f"⭐ Top Pick {n_top}")
                verdict_subtitle = f"공식 추천 {n_top}개  ·  " + " / ".join(type_summary)
                gradient_from = "#0a3d2a"
                gradient_via = "#0d5440"
                border_color = "border-emerald-500/50"
                text_main = "text-emerald-300"
                text_sub = "text-emerald-100"
                count_color = "text-emerald-300"
            
            # 헤더 카드 — verdict
            with ui.card().classes(
                f"w-full p-5 mb-4 rounded-xl "
                f"border-2 {border_color}"
            ).style(
                f"background: linear-gradient(to right, {gradient_from}, {gradient_via}, {gradient_from})"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.column().classes("gap-1"):
                        ui.label(f"{verdict_emoji} {verdict_text}").classes(
                            f"text-lg font-bold {text_main}"
                        )
                        ui.label(verdict_subtitle).classes(f"text-sm {text_sub}")
                    ui.label(f"{n_top}").classes(
                        f"text-5xl font-black {count_color}"
                    )
            
            # ─── [Step D2] TOP_PICK 카드들 — 다축 정렬 적용 ───
            top_picks_sorted = _sort_top_picks_for_hero(top_picks).head(3)
            
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
                    
                    # [v22 UI Step C + I] 3축 + 밸런스 + IS_NOW_ENTRY (ui_terms 사용)
                    struct = safe_float(row.get('STRUCT_SCORE', 0))
                    timing = safe_float(row.get('TIMING_SCORE', 0))
                    ai_sc = safe_float(row.get('AI_SCORE', row.get('ML_SCORE', 0)))
                    balance = safe_float(row.get('BALANCE_SCORE', 0))
                    is_now_entry = is_truthy_flag(row.get('IS_NOW_ENTRY', '0'))
                    # [v22.3.21] 공식 신규매수 게이트 — 초록 CTA는 TOP_PICK & ELIGIBLE일 때만
                    from components.buy_now_badge import is_official_new_buy
                    official_buy = is_official_new_buy(row)
                    
                    # [v22 UI Step E4 + F3] Kelly engine + error 요약
                    kelly_engine = str(row.get('KELLY_ENGINE', '')).strip()
                    kelly_error = str(row.get('KELLY_ERROR', '')).strip()
                    
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
                        
                        # [Step I] 진입갭 방향성 → ui_terms.gap_direction
                        gap_desc = gap_direction(gap)
                        ui.label(
                            f"{type_label}  ·  수익:손실 {rr:.1f}:1  ·  추천가 차이 {gap:+.1f}% ({gap_desc})"
                        ).classes("text-xs text-gray-400 mb-2")
                        
                        # [v22 UI Step C] 3축 + 밸런스 한 줄
                        ui.label(
                            f"구조 {struct:.0f} · 타이밍 {timing:.0f} · AI {ai_sc:.0f}  ·  3축 균형 {balance:.0f}"
                        ).classes("text-xs text-purple-300 mb-1")
                        
                        # [v22 UI Step C] IS_NOW_ENTRY 배지
                        # [v3.9.9] 시장 모드가 종목 카드 문구를 덮어씀
                        # — 종목 추천가 도달 + 시장 위험 → 표현 강도 조정
                        if is_now_entry:
                            if is_macro_dangerous:
                                # 🔴 WARNING/CRITICAL → 신규 매수 보류
                                ui.label("🚫 신규 매수 보류").classes(
                                    "text-xs text-red-400 font-bold mb-1"
                                )
                                ui.label(
                                    "시장 위험 구간 — 신규 진입보다 보유 리스크 관리 우선"
                                ).classes("text-[10px] text-red-300/80 mb-1")
                            elif is_macro_caution:
                                # 🟠 CAUTION → 조건부 소액 매수
                                ui.label("🟠 조건부 소액 매수").classes(
                                    "text-xs text-orange-400 font-bold mb-1"
                                )
                                ui.label(
                                    "시장 주의 구간 — 평소 비중의 50% 이하로 제한"
                                ).classes("text-[10px] text-orange-300/80 mb-1")
                            elif official_buy:
                                # 🟢 NORMAL + 공식(TOP_PICK & ELIGIBLE) → 매수 가능
                                ui.label("✅ 오늘 신규 매수 가능").classes(
                                    "text-xs text-emerald-400 font-bold mb-1"
                                )
                            else:
                                # [v22.3.21] TOP_PICK이나 BUY_NOW_ELIGIBLE=0 → 초록 CTA 금지(중립)
                                ui.label("⏳ 가격 도달 — 공식 매수 대상 아님").classes(
                                    "text-xs text-gray-400 mb-1"
                                )
                        else:
                            ui.label("⏳ 추천가 도달 대기").classes(
                                "text-xs text-amber-400 mb-1"
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
                                # [Step F1] 위험/차단/주의 3단계 비중 안내
                                if is_macro_dangerous or route_blocked:
                                    # ⛔ 위험장 또는 엔진 차단 — 신규 매수 0원
                                    ui.label(
                                        f"⛔ 신규매수 0원  ·  기준 {amt:.0f}만원은 관찰용"
                                    ).classes("text-sm font-bold text-red-300")
                                elif is_macro_caution:
                                    # 🟠 시장 주의 — 50% 비중 권장
                                    ui.label(
                                        f"💰 기준 {amt:.0f}만원  ·  주의 시 권장 {amt*0.5:.0f}만원"
                                    ).classes("text-sm font-bold text-amber-300")
                                else:
                                    # 🟢 정상 — 기준값 그대로
                                    ui.label(f"💰 {amt:.0f}만원").classes(
                                        "text-sm font-bold text-amber-300"
                                    )
                            if ewr > 0:
                                # [v3.9.10] 종목 카드 "승률 47%" vs 조합 분석 "승률 76%" 충돌
                                # → 명시적으로 "개별 모델 승률"로 표기 + 툴팁
                                ui.label(f"개별 모델 승률 {ewr*100:.0f}%").classes(
                                    "text-xs text-gray-400"
                                ).tooltip(
                                    "이 종목 현재 조건의 모델 예측 승률입니다. "
                                    "아래 '조합별 성과'의 승률은 과거 유사 패턴 평균이라 의미가 다릅니다."
                                )
                        
                        # [Step E4 + F3 + I] Kelly engine 표시 → ui_terms.kelly_engine_label
                        # [v3.9.11] 3-tuple: (text, css, raw). admin이면 raw 엔진명 tooltip
                        kelly_text, kelly_cls, kelly_raw = kelly_engine_label(kelly_engine)
                        if kelly_text:
                            _lbl_k = ui.label(kelly_text).classes(f"{kelly_cls} mt-1")
                            if auth == "admin" and kelly_raw:
                                _lbl_k.tooltip(f"engine: {kelly_raw}")
                            # [F3] fallback일 때 KELLY_ERROR 요약 (있을 때만, 80자)
                            if 'fallback' in kelly_engine.lower():
                                if kelly_error and kelly_error.lower() not in ("nan", "none", ""):
                                    _err_short = kelly_error[:80]
                                    if len(kelly_error) > 80:
                                        _err_short += "…"
                                    ui.label(_err_short).classes(
                                        "text-[10px] text-red-400/70"
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
            
            # [v22 UI Step D3] 후보 카드 강화 — 3축/밸런스/RR/진입갭/IS_NOW_ENTRY
            cand_struct = safe_float(top_cand.get('STRUCT_SCORE', 0))
            cand_timing = safe_float(top_cand.get('TIMING_SCORE', 0))
            cand_ai = safe_float(top_cand.get('AI_SCORE', top_cand.get('ML_SCORE', 0)))
            cand_balance = safe_float(top_cand.get('BALANCE_SCORE', 0))
            cand_rr = safe_float(top_cand.get('RR_NOW_TP1', 0))
            cand_gap = safe_float(top_cand.get('ENTRY_GAP_PCT', 0))
            cand_is_now = is_truthy_flag(top_cand.get('IS_NOW_ENTRY', '0'))
            # [v22.3.21] 공식 신규매수 게이트 — 초록 CTA는 TOP_PICK & ELIGIBLE일 때만
            from components.buy_now_badge import is_official_new_buy
            cand_official_buy = is_official_new_buy(top_cand)
            
            # 부족한 점수 진단 (변수 재사용 — 위에서 추출했으므로 그대로)
            shortfall_msg = ""
            if cand_struct > 0 and cand_struct < 80:
                shortfall_msg = f"구조 점수 {80 - cand_struct:.1f}점 부족 (80↑ 필요)"
            elif cand_score < 75:
                shortfall_msg = f"종합 점수 {75 - cand_score:.1f}점 부족 (75↑ 필요)"
            elif cand_timing > 0 and cand_timing < 70:
                shortfall_msg = f"타이밍 점수 {70 - cand_timing:.1f}점 부족 (70↑ 필요)"
            else:
                shortfall_msg = "조건 일부 미달"
            
            # 헤더 카드 — 관찰 모드
            # [Step D + E1] macro risk + route_blocked 통합 분기
            if is_macro_dangerous:
                _hdr_emoji = "🔴"
                _hdr_text = "오늘 매수 금지 (시장 위험)"
                _hdr_subtitle = (
                    f"관찰 후보 {len(active)}종목 있지만 "
                    f"매크로 위험 ({macro_risk}) — 관찰만"
                )
                _hdr_g_from = "#3d0a0a"; _hdr_g_via = "#541313"
                _hdr_border = "border-red-500/50"
                _hdr_text_main = "text-red-300"
                _hdr_text_sub = "text-red-100"
            elif route_blocked:
                # [Step E1] 엔진 ROUTE 제한 — 신규 진입 X
                _hdr_emoji = "🟠"
                _hdr_text = "신규 매수 자제 (엔진 제한)"
                _hdr_subtitle = (
                    f"관찰 후보 {len(active)}종목 있지만 "
                    f"엔진 상태={route_display(max_route)} — 관찰만"
                )
                _hdr_g_from = "#3d2a0a"; _hdr_g_via = "#544013"
                _hdr_border = "border-orange-500/50"
                _hdr_text_main = "text-orange-300"
                _hdr_text_sub = "text-orange-100"
            else:
                _hdr_emoji = "⏸️"
                _hdr_text = "오늘은 지켜보세요"
                _suffix = f" · 시장 주의({macro_risk})" if is_macro_caution else ""
                _hdr_subtitle = f"공식 추천 0개  ·  관찰 후보 {len(active)}종목{_suffix}"
                _hdr_g_from = "#3d2a0a"; _hdr_g_via = "#544013"
                _hdr_border = "border-amber-500/50"
                _hdr_text_main = "text-amber-300"
                _hdr_text_sub = "text-amber-100"
            
            with ui.card().classes(
                f"w-full p-5 mb-4 rounded-xl border-2 {_hdr_border}"
            ).style(
                f"background: linear-gradient(to right, {_hdr_g_from}, {_hdr_g_via}, {_hdr_g_from})"
            ):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.column().classes("gap-1"):
                        ui.label(f"{_hdr_emoji} {_hdr_text}").classes(
                            f"text-lg font-bold {_hdr_text_main}"
                        )
                        ui.label(_hdr_subtitle).classes(
                            f"text-sm {_hdr_text_sub}"
                        )
                    ui.label("0").classes(
                        f"text-5xl font-black {_hdr_text_main}/60"
                    )
            
            # 가까운 후보 카드 (강화됨)
            with ui.card().classes(
                "w-full p-4 mb-4 rounded-xl "
                "bg-[#1a1a2e] border border-amber-700/40"
            ):
                ui.label(f"💡 가장 가까운 종목 (왜 통과 못했나?)").classes("text-xs text-gray-400 mb-2")
                
                with ui.row().classes("w-full items-center gap-3 mb-2"):
                    ui.label(f"👀 {cand_name}").classes(
                        "text-lg font-bold text-white"
                    )
                    ui.badge(f"E{cand_score:.1f}", color="#F59E0B").classes("text-xs")
                    ui.badge(cand_route, color="#3B82F6").classes("text-xs")
                
                # [Step D3] 3축 + 밸런스 (한국어 풀어쓰기)
                ui.label(
                    f"구조 {cand_struct:.0f} · 타이밍 {cand_timing:.0f} · AI {cand_ai:.0f}  "
                    f"·  3축 균형 {cand_balance:.0f}"
                ).classes("text-xs text-purple-300 mb-1")
                
                # [Step I] RR + 진입갭 → ui_terms.gap_direction
                _cand_gap_desc = gap_direction(cand_gap)
                ui.label(
                    f"수익:손실 {cand_rr:.1f}:1  ·  추천가 차이 {cand_gap:+.1f}% ({_cand_gap_desc})"
                ).classes("text-xs text-gray-400 mb-1")
                
                # [Step D3] IS_NOW_ENTRY 배지 (관찰모드는 보통 ⏳)
                # [v3.9.9] 시장 모드 덮어쓰기 (관찰 후보도 동일하게)
                if cand_is_now:
                    if is_macro_dangerous:
                        ui.label("🚫 신규 매수 보류 (시장 위험)").classes(
                            "text-xs text-red-400 font-bold mb-1"
                        )
                    elif is_macro_caution:
                        ui.label("🟠 조건부 소액 매수 (시장 주의 — 비중 50% 이하)").classes(
                            "text-xs text-orange-400 font-bold mb-1"
                        )
                    elif cand_official_buy:
                        ui.label("✅ 오늘 신규 매수 가능").classes(
                            "text-xs text-emerald-400 mb-1"
                        )
                    else:
                        # [v22.3.21] 공식 매수 미통과(TOP_PICK&ELIGIBLE 아님) → 초록 CTA 금지(중립)
                        ui.label("⏳ 가격 도달 — 공식 매수 대상 아님").classes(
                            "text-xs text-gray-400 mb-1"
                        )
                else:
                    ui.label("⏳ 추천가 도달 대기").classes(
                        "text-xs text-amber-400 mb-1"
                    )
                
                for _line in _blocking_priority_text(macro_risk, max_route, shortfall_msg):
                    ui.label(f"└ {_line}").classes(
                        "text-sm text-amber-300 mt-1"
                    )
                
                # [v22.3.21 FOMO-safety] 매수보류 카드에서 목표가/매수가는 매수 CTA처럼
                # 띄우지 않고 회색 '참고용' 각주로 강등한다 (백엔드 안전신호 보호).
                if cand_buy > 0 and cand_target > 0:
                    ui.label(
                        f"참고용 — 조건 충족 시 목표 +{cand_tp1:.1f}% "
                        f"({int(cand_buy):,} → {int(cand_target):,}) · 오늘은 매수 대상이 아닙니다"
                    ).classes("text-xs text-gray-500 mt-1")
                
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

def render_tab_market(df, auth: str = "free"):
    """Tab 1: 시장 현황
    
    [v3.9.11] auth 추가 — 관리자에게 kelly engine 내부명 등 디버그 정보 노출
    """
    import os, json

    fg_score, fg_label = _get_fear_greed(df)
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # ═══════════════════════════════════════════════════
    # [v22 UI Step D] meta 먼저 로드 — Hero가 macro risk 알 수 있게
    # ═══════════════════════════════════════════════════
    meta = {}
    try:
        mp = os.path.join(DATA_DIR, "run_meta_latest.json")
        if os.path.exists(mp):
            with open(mp, 'r') as f:
                meta = json.load(f)
    except Exception:
        pass

    # ═══════════════════════════════════════════════════
    # [v22 UI] 오늘의 결론 Hero 카드 — 가장 먼저 (1초 답변)
    # [Step D] meta 인자 추가 → macro risk 기반 verdict
    # ═══════════════════════════════════════════════════
    _render_today_hero(df, meta, auth)

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
        is_macro_dangerous = str(macro_risk or "").strip().upper() in {"WARNING", "CRITICAL"}
        route_blocked = is_route_blocked(max_route)
        is_no_buy_mode = _is_market_no_buy_mode(macro_risk, max_route)

        with ui.card().classes("w-full p-4 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
            ui.label("🛡️ 엔진 상태").classes("text-xs text-gray-400 mb-2")
            with ui.row().classes("w-full gap-4 flex-wrap"):
                with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                    ui.label("매크로 리스크").classes("text-xs text-gray-400")
                    ui.label(f"{'🟢' if macro_risk == 'NORMAL' else '🟡' if macro_risk == 'CAUTION' else '🔴'} {risk_kr}").classes("text-lg font-bold").style(f"color:{risk_color}")
                    if macro_msg:
                        ui.label(macro_msg).classes("text-xs text-gray-500")
                    # [v22.3.21] 매수금지 구간이면 행동 지침 한 줄 (가격/CTA 톤과 일관)
                    if is_no_buy_mode:
                        ui.label("→ 오늘은 신규매수 보류 구간").classes("text-xs text-red-300/80 mt-1")

                with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                    ui.label("시장 Breadth").classes("text-xs text-gray-400")
                    bc = "#10B981" if breadth >= 60 else "#F59E0B" if breadth >= 40 else "#EF4444"
                    ui.label(f"{breadth:.1f}%").classes("text-lg font-bold").style(f"color:{bc}")
                    ui.label("상승 종목 비율").classes("text-xs text-gray-500")

                with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                    ui.label("엔진 신뢰도").classes("text-xs text-gray-400")
                    cc = "#10B981" if confidence >= 80 else "#F59E0B" if confidence >= 50 else "#EF4444"
                    ui.label(f"{confidence:.0f}/100").classes("text-lg font-bold").style(f"color:{cc}")
                    # [Step J] 최대 허용 ROUTE도 한국어로 표시
                    # [v22.3.21] 매수금지 구간이면 매수성 ROUTE 단어 대신 '신규매수 보류'로 표시
                    #   (시장이 막은 날 '최대허용: 적극 매수' 등이 매수 가능으로 오해되는 것 방지)
                    if is_no_buy_mode:
                        ui.label("최대허용: 신규매수 보류").classes("text-xs text-gray-500")
                    else:
                        _max_route_disp = route_display(max_route) if max_route else "-"
                        ui.label(f"최대허용: {_max_route_disp}").classes("text-xs text-gray-500")

                # ELITE/TOP_PICK 요약
                # [v3.9.10] 회원이 "공식 Top Pick 현황 / 평균 점수 32" 보고
                # "넥스트칩 E87인데 왜 평균이 32?" 혼란 → TOP_PICK 평균만 계산
                # [v3.9.10 hotfix] TOP_PICK 값이 "1"/1.0/True 등 다양해서
                # is_truthy_flag로 통일 (다른 코드와 일관성)
                if "ELITE_SCORE" in df.columns:
                    if "TOP_PICK" in df.columns:
                        _tp_mask = df["TOP_PICK"].apply(is_truthy_flag)
                        tp_count = int(_tp_mask.sum())
                    else:
                        _tp_mask = None
                        tp_count = 0
                    # TOP_PICK 종목들의 ELITE_SCORE 평균 (없으면 표시 생략)
                    if _tp_mask is not None and tp_count > 0:
                        elite_avg = df.loc[_tp_mask, "ELITE_SCORE"].mean()
                        avg_label = f"평균 ELITE {elite_avg:.0f}"
                    else:
                        # TOP_PICK 없으면 표시 안 함 (혼란 방지)
                        avg_label = ""
                    with ui.card().classes("p-3 min-w-[130px] bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                        ui.label("공식 Top Pick 현황").classes("text-xs text-gray-400")
                        ui.label(f"🏆 {tp_count}종목").classes("text-lg font-bold text-yellow-400")
                        if avg_label:
                            ui.label(avg_label).classes("text-xs text-gray-500")

                        # [v3.9.22b → v22.3.8 safety] BUY_NOW 분포 (3순위)
                        # TOP_PICK 중 ELIGIBLE/관찰/AVOID 카운트
                        # [v22.3.8] BUY 카운트는 ELIGIBLE=1 기준 (회원 오해 방지).
                        # BUY_NOW_GRADE=BUY인데 ELIGIBLE=0이면 "관찰 후보"로 분류.
                        if (_tp_mask is not None and tp_count > 0
                                and "BUY_NOW_GRADE" in df.columns):
                            _tp_df = df.loc[_tp_mask]
                            # ELIGIBLE 컬럼 안전 처리 (legacy CSV 호환)
                            if "BUY_NOW_ELIGIBLE" in _tp_df.columns:
                                _eligible = (
                                    _tp_df["BUY_NOW_ELIGIBLE"]
                                    .fillna(0).astype(int) == 1
                                )
                            else:
                                _eligible = _tp_df["BUY_NOW_GRADE"] == "BUY"
                            _is_buy = _tp_df["BUY_NOW_GRADE"] == "BUY"
                            _is_avoid = _tp_df["BUY_NOW_GRADE"] == "AVOID"
                            # ★ ELIGIBLE=1 AND BUY → 공식 매수
                            n_buy = int((_is_buy & _eligible).sum())
                            # BUY이지만 ELIGIBLE=0이면 관찰로 강등
                            n_watch = int(
                                (_tp_df["BUY_NOW_GRADE"] == "WATCH").sum()
                                + (_is_buy & ~_eligible).sum()
                            )
                            n_avoid = int(_is_avoid.sum())
                            with ui.row().classes("gap-1 mt-1"):
                                if n_buy > 0:
                                    ui.label(f"🟢{n_buy}").classes(
                                        "text-xs text-emerald-400"
                                    ).tooltip("공식 매수 가능 (ELIGIBLE)")
                                if n_watch > 0:
                                    ui.label(f"🟡{n_watch}").classes(
                                        "text-xs text-amber-400"
                                    ).tooltip("관찰 후보 (BUY이나 ELIGIBLE=0 포함)")
                                if n_avoid > 0:
                                    ui.label(f"🔴{n_avoid}").classes(
                                        "text-xs text-red-400"
                                    ).tooltip("추격 금지")

        # ── [v22 UI Step B + K] 점수 우수 후보 더 보기 — TOP_PICK 제외 ──
        # Hero 카드에 이미 TOP_PICK이 표시되므로, 여기서는 TOP_PICK 제외한 후보만
        try:
            if "ELITE_SCORE" in df.columns:
                _top_df = df.copy()
                
                # [v22 UI Step B] TOP_PICK 제외 (Hero 카드 중복 제거)
                if "TOP_PICK" in _top_df.columns:
                    _tp_mask = _top_df["TOP_PICK"].apply(is_truthy_flag)
                    _candidates = _top_df[~_tp_mask].copy()
                    # [v3.9.10] "점수 우수 후보 더 보기" → "관찰 후보 — 오늘 매매 제외"
                    # CAUTION 시장에서 회원이 후보를 추천으로 오해 방지
                    _label_text = "👀 관찰 후보 — 오늘 매매 제외"
                else:
                    _candidates = _top_df.copy()
                    _label_text = "👀 점수 우수 관찰 종목 Top"
                
                # 활성 ROUTE 우선 정렬 (관찰 가치 있는 종목)
                _picks = _candidates.nlargest(3, "ELITE_SCORE") if not _candidates.empty else _candidates

                if not _picks.empty:
                    with ui.card().classes("w-full p-4 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
                        ui.label(_label_text).classes("text-xs text-gray-400 mb-1")
                        # [v3.9.10] 회원 혼란 방지 — 왜 추천이 아닌지 한 줄 안내
                        # [v3.9.11] 종목별 개별 제외 사유는 각 카드에 별도 표시
                        ui.label(
                            "점수는 좋지만 현재 진입 조건/시장 상태상 오늘의 추천에서는 제외된 종목입니다. "
                            "각 종목 카드 하단에 제외 사유가 표시됩니다."
                        ).classes("text-[10px] text-gray-500 italic mb-2")
                        with ui.row().classes("w-full gap-3 flex-wrap"):
                            for _, s in _picks.iterrows():
                                route = str(s.get("ROUTE", ""))
                                # [Step I] route_icon → ui_terms.route_icon
                                _route_icon = route_icon(route)
                                elite = safe_float(s.get("ELITE_SCORE", 0))
                                close = safe_float(s.get("종가", 0))
                                tp1 = safe_float(s.get("추천매도가1", 0))
                                rr = safe_float(s.get("RR_NOW_TP1", 0))
                                wr = safe_float(s.get("EST_WIN_RATE", 0))
                                bal = safe_float(s.get("BALANCE_SCORE", 0))
                                gap_pct = safe_float(s.get("GAP_PCT", 0))
                                elite_lbl = str(s.get("ELITE_LABEL", "") or "")
                                # [v3.9.11 hotfix] 비교용 키는 strip().upper() — 데이터에
                                # 공백/소문자 들어와도 silent miss 방지
                                risk_lvl = str(s.get("ENTRY_RISK_LEVEL", "") or "").strip().upper()
                                route_key = str(s.get("ROUTE", "") or "").strip().upper()
                                tp_flag = ""   # [Step B] TOP_PICK 제외했으므로 항상 빈 문자열
                                tp1_pct = (tp1 / close - 1) * 100 if close > 0 else 0

                                # [v3.9.11] 개별 제외 사유 — 우선순위 순서로 첫 매칭 표시
                                _reason = ""
                                if is_macro_dangerous:
                                    _reason = "🚫 시장 위험 구간 — 오늘 매매 제외"
                                elif risk_lvl == "RED":
                                    _reason = "🔴 RED 위험 — 진입 위험 패턴"
                                elif risk_lvl == "ORANGE":
                                    _reason = "🟠 ORANGE — 과열 주의"
                                elif "추격" in elite_lbl:
                                    _reason = "⚠️ 추격 위험 — 추천가 위에서 매수 비추"
                                elif abs(gap_pct) > 5:
                                    _reason = f"📏 추천가 차이 큼 ({gap_pct:+.1f}%)"
                                elif route_key in ("WAIT", "NEUTRAL"):
                                    _reason = "👀 진입 시점 아님 — 관망 구간"
                                elif is_macro_caution:
                                    _reason = "⚠️ 시장 CAUTION — 보수 접근 구간"

                                with ui.card().classes("flex-1 min-w-[200px] p-3 bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                                    with ui.row().classes("items-center gap-2"):
                                        ui.label(f"{_route_icon} {tp_flag}{s.get('종목명', '')}").classes("text-white font-bold text-sm")
                                        ui.badge(f"E{elite:.0f}", color="#10B981" if elite >= 80 else "#3B82F6").classes("text-xs")
                                    ui.label(f"구조 {safe_float(s.get('STRUCT_SCORE', 0)):.0f} · 타이밍 {safe_float(s.get('TIMING_SCORE', 0)):.0f} · AI {safe_float(s.get('AI_SCORE', 0)):.0f} | 3축 균형 {bal:.0f}").classes("text-xs text-gray-400 mt-1")
                                    # [v22.3.21 FOMO-safety] 제외 종목의 목표가/매수가는
                                    # 회색 '참고용' 각주로 강등 (매수 신호 오해 방지).
                                    ui.label(f"참고용 — 조건 충족 시 목표 +{tp1_pct:.1f}% ({close:,.0f} → {tp1:,.0f}) · 수익:손실 {rr:.1f}:1 · 승률 {wr * 100:.0f}% · 오늘은 매수 대상 아님").classes("text-[10px] text-gray-500")
                                    # [v3.9.11] 종목별 개별 제외 사유 — 하단에 한 줄
                                    if _reason:
                                        ui.label(f"└ 제외 사유: {_reason}").classes(
                                            "text-[10px] text-amber-300/80 mt-1 italic"
                                        )
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
            # [v3.9.10] "조합 승률 76%" vs "개별 모델 승률 47%" 충돌 해소
            # 회원이 두 승률이 다른 의미임을 알 수 있게 한 줄 안내
            ui.label(
                "ℹ️ 아래 \"승률\"은 과거 유사 패턴(조합)의 평균이며, "
                "위 종목 카드의 \"개별 모델 승률\"과는 다른 의미입니다."
            ).classes("text-[11px] text-blue-300 italic mb-2")

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
                            ui.label(_combo_section_title(is_no_buy_mode)).classes("text-sm font-bold text-yellow-400 mb-2")
                            if is_no_buy_mode:
                                ui.label(
                                    "※ 현재 시장/엔진 상태상 이 조합은 과거 통계 참고용입니다. 오늘 공식 신규매수 신호가 아닙니다."
                                ).classes("text-[10px] text-red-300 italic mb-2")

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
                            ui.label(
                                _match_section_title(is_no_buy_mode, len(matched), _combo_label, _uc['win_rate'])
                            ).classes("text-sm font-bold text-yellow-400 mb-2")
                            if is_no_buy_mode:
                                ui.label(
                                    "※ 현재 시장/엔진 상태상 아래 매칭 종목은 관찰 전용이며 공식 신규매수가 아닙니다."
                                ).classes("text-[10px] text-red-300 italic mb-2")

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
                ui.label("🚦 상태별 승률").classes("text-sm font-bold text-white mb-2")
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
                            {"name": "elite", "label": "평균 종합 점수", "field": "elite", "align": "center"},
                            {"name": "rr", "label": "평균 수익:손실", "field": "rr", "align": "center"},
                        ],
                        rows=route_rows, row_key="route",
                    ).classes("w-full").props("dense dark flat bordered")

            with ui.card().classes("w-full p-4 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
                ui.label("📈 지표별 승률 예측력 (상위20% vs 하위20%)").classes("text-sm font-bold text-white mb-2")
                axes_check = [
                    ("DISPLAY_SCORE", "종합점수"), ("STRUCT_SCORE", "구조(S)"),
                    ("TIMING_SCORE", "타이밍(T)"), ("AI_SCORE", "AI"),
                    ("ELITE_SCORE", "ELITE"), ("BALANCE_SCORE", "밸런스"),
                    ("RR_NOW_TP1", "수익:손실"),
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


def _safe_float_or_default(value, default=0.0):
    num = _safe_number_or_none(value)
    return default if num is None else num


def _parse_reference_date(value=None):
    """메타/문자열/날짜 객체를 date로 안전 변환한다."""
    if value is None:
        return datetime.now().date()
    if hasattr(value, "date"):
        try:
            return value.date()
        except TypeError as e:
            # date가 호출 불가능한 속성인 객체 — 아래 문자열 파싱으로 폴백
            _logger.debug("[tab_market] reference date .date() 호출 실패 (문자열 파싱 폴백): %s", e)
    text = str(value or "").strip()
    for pat in (r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", r"(\d{4})(\d{2})(\d{2})"):
        m = re.search(pat, text)
        if m:
            y, mo, d = map(int, m.groups())
            try:
                return datetime(y, mo, d).date()
            except ValueError:
                return datetime.now().date()
    return datetime.now().date()


def _extract_macro_msg_date(macro_msg: str, reference_date=None):
    """`환율 1513원 [05/24]` 같은 macro_msg에서 기준일을 추출한다."""
    msg = str(macro_msg or "")
    ref = _parse_reference_date(reference_date)
    m = re.search(r"\[(\d{1,2})/(\d{1,2})\]", msg)
    if not m:
        return None
    month, day = map(int, m.groups())
    year = ref.year
    try:
        dt = datetime(year, month, day).date()
    except ValueError:
        return None
    if dt > ref + timedelta(days=31):
        try:
            dt = datetime(year - 1, month, day).date()
        except ValueError:
            return None
    return dt


def _extract_fx_level(macro_msg: str):
    """macro_msg에서 원/달러 환율 숫자를 추출한다."""
    msg = str(macro_msg or "")
    patterns = [
        r"환율\s*([0-9,]+(?:\.\d+)?)\s*원",
        r"USD\s*/?\s*KRW\s*[:=]?\s*([0-9,]+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, msg, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
    return None


def _fx_regime_diagnosis(macro_msg="", macro_risk="", breadth=0, max_route="", reference_date=None):
    """환율 CRITICAL이 새 충격인지 장기 고환율 레짐인지 표시용으로 분해한다."""
    ref_date = _parse_reference_date(reference_date)
    msg_date = _extract_macro_msg_date(macro_msg, ref_date)
    fx_level = _extract_fx_level(macro_msg)
    risk = str(macro_risk or "").strip().upper()
    breadth_num = _safe_float_or_default(breadth, 0)

    stale_days = None
    if msg_date is not None:
        stale_days = max((ref_date - msg_date).days, 0)

    is_stale = stale_days is not None and stale_days >= 2
    is_high_fx = fx_level is not None and fx_level >= 1500
    breadth_ok = breadth_num >= 45
    route_limited = is_route_blocked(max_route)

    lines = []
    if fx_level is not None:
        if is_high_fx:
            lines.append(f"환율 {fx_level:,.0f}원 — 고환율 레짐")
        else:
            lines.append(f"환율 {fx_level:,.0f}원 — 절대 레벨은 CRITICAL 단독 근거로 약함")
    elif macro_msg:
        lines.append(f"매크로 메시지: {macro_msg}")
    else:
        lines.append("매크로 상세 메시지 없음")

    if msg_date is not None:
        freshness = f"데이터 기준 {msg_date.strftime('%m/%d')}"
        if is_stale:
            freshness += f" — {stale_days}일 경과, 최신성 확인 필요"
        else:
            freshness += " — 최신성 양호"
        lines.append(freshness)
    else:
        lines.append("환율 기준일 파싱 불가 — 최신성 확인 필요")

    lines.append(f"Breadth {breadth_num:.1f}% — {'시장 내부는 중립 이상' if breadth_ok else '시장 내부도 약함'}")

    if risk in {"WARNING", "CRITICAL"} and is_high_fx and is_stale and breadth_ok:
        verdict = "고환율 장기/안정 레짐 가능 — 전면 매수금지 지속 여부 점검 필요"
        tone = "amber"
    elif risk in {"WARNING", "CRITICAL"} and not is_stale:
        verdict = "최근 매크로 충격 가능 — 보수 모드 유지"
        tone = "red"
    elif route_limited:
        verdict = "엔진 ROUTE 제한이 우선 — 매크로보다 엔진 상태 확인"
        tone = "orange"
    else:
        verdict = "매크로 hard block 단독 원인은 약함 — 종목/진입 게이트 확인"
        tone = "gray"

    return {
        "fx_level": fx_level,
        "macro_msg_date": msg_date,
        "stale_days": stale_days,
        "is_stale": is_stale,
        "is_high_fx": is_high_fx,
        "breadth_ok": breadth_ok,
        "verdict": verdict,
        "tone": tone,
        "lines": lines,
    }


def _truthy_series(df, col: str):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].apply(is_truthy_flag)


def _numeric_series(df, cols, default=0.0):
    for col in cols:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index)


def _route_active_mask(df):
    if "ROUTE" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["ROUTE"].astype(str).str.upper().str.contains("ARMED|ATTACK", na=False)


def _ebs_fail_mask(df):
    for col in ("EBS_STATUS", "EBS", "EBS_PASS"):
        if col in df.columns:
            s = df[col].astype(str).str.upper()
            return s.str.contains("FAIL|FALSE|미통과|0/", na=False) | s.isin({"0", "0.0", "N", "NO"})
    return pd.Series(False, index=df.index)


def _build_no_buy_gate_audit(df, meta=None) -> dict:
    """공식 신규매수 0개 원인을 gate별로 분해한다. 표시/진단 전용."""
    meta = meta or {}
    if df is None:
        df = pd.DataFrame()

    total = int(len(df))
    active_mask = _route_active_mask(df)
    top_mask = _truthy_series(df, "TOP_PICK")
    eligible_mask = _truthy_series(df, "BUY_NOW_ELIGIBLE")
    pass_mask = _truthy_series(df, "BUY_NOW_PASS")
    official_mask = top_mask & eligible_mask

    active = df[active_mask].copy()
    score = _numeric_series(active, ["ELITE_SCORE", "DISPLAY_SCORE", "FINAL_SCORE"], 0)
    rr = _numeric_series(active, ["RR_NOW_TP1", "RR_MULT"], 0)
    vwap = _numeric_series(active, ["VWAP_GAP", "VWAP_GAP_PCT"], 0)
    poc = _numeric_series(active, ["POC_GAP", "POC_GAP_PCT"], 0)

    macro_risk = str(meta.get("macro_risk", "")).strip().upper()
    max_route = meta.get("max_allowed_route", "")
    macro_block = macro_risk in {"WARNING", "CRITICAL"} or is_route_blocked(max_route)

    counts = {
        "total": total,
        "armed_attack": int(active_mask.sum()),
        "top_pick": int(top_mask.sum()),
        "official_buy": int(official_mask.sum()),
        "macro_blocked": int(active_mask.sum()) if macro_block else 0,
        "final_under_75": int((score < 75).sum()) if not active.empty else 0,
        "buy_now_pass_0": int((~pass_mask[active_mask]).sum()) if "BUY_NOW_PASS" in df.columns else 0,
        "buy_now_eligible_0": int((~eligible_mask[active_mask]).sum()) if "BUY_NOW_ELIGIBLE" in df.columns else 0,
        "vwap_poc_overheat": int(((vwap > 10) | (poc > 30)).sum()) if not active.empty else 0,
        "rr_under_1_2": int((rr < 1.2).sum()) if not active.empty else 0,
        "ebs_fail": int(_ebs_fail_mask(active).sum()) if not active.empty else 0,
    }

    closest = []
    if not active.empty:
        active["_score"] = score
        active["_rr"] = rr
        active["_vwap"] = vwap
        active["_poc"] = poc
        for _, row in active.sort_values(["_score", "_rr"], ascending=[False, False]).head(3).iterrows():
            closest.append({
                "name": str(row.get("종목명", row.get("name", ""))),
                "route": str(row.get("ROUTE", "")),
                "score": _safe_float_or_default(row.get("ELITE_SCORE", row.get("DISPLAY_SCORE", row.get("FINAL_SCORE", 0))), 0),
                "rr": _safe_float_or_default(row.get("RR_NOW_TP1", row.get("RR_MULT", 0)), 0),
                "vwap_gap": _safe_float_or_default(row.get("VWAP_GAP", row.get("VWAP_GAP_PCT", 0)), 0),
                "poc_gap": _safe_float_or_default(row.get("POC_GAP", row.get("POC_GAP_PCT", 0)), 0),
                "top_pick": is_truthy_flag(row.get("TOP_PICK", 0)),
                "eligible": is_truthy_flag(row.get("BUY_NOW_ELIGIBLE", 0)),
            })

    return {"counts": counts, "closest": closest, "macro_block": macro_block}

