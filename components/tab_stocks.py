# -*- coding: utf-8 -*-
"""
tab_stocks.py — 🎯 AI & Quant 추천 종목 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
종목 테이블/칸반, 상세 분석(캔들·레이더·워터폴), CSV 다운로드
"""
import asyncio
import logging
import math
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from nicegui import ui

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
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

try:
    from indicators import calculate_supertrend
except ImportError:
    def calculate_supertrend(df): return df

try:
    from kelly_widget import render_kelly_calculator
    KELLY_OK = True
except ImportError:
    KELLY_OK = False

try:
    from trade_journal_tab import save_trade
    JOURNAL_OK = True
except ImportError:
    JOURNAL_OK = False

from shared_utils import nz_num, safe_float, calc_hma as calc_hma_series
from chart_components import plot_radar_chart, plot_score_waterfall

_logger = logging.getLogger(__name__)
# ── 상태 한글화 매핑 ──
ROUTE_KR = {
    "ATTACK": "🚀 매수 돌입",
    "ARMED": "🔫 매수 대기",
    "WAIT": "👀 관망",
    "OVERHEAT": "🔥 과열",
    "NEUTRAL": "⚪ 중립",
    "CARRY": "📌 보유 관찰",
}
ROUTE_DESC = {
    "ATTACK": "매수 시그널 발생! 진입 조건 충족",
    "ARMED": "조건 근접, 돌파 시 매수 준비",
    "WAIT": "아직 조건 미충족, 추이 관망",
    "OVERHEAT": "과열 구간, 신규 진입 주의",
    "NEUTRAL": "뚜렷한 방향성 없음",
    "CARRY": "이전 추천 종목 — 손절/익절가 관찰 중",
}
ROUTE_COLOR = {
    "ATTACK": "#EF4444",
    "ARMED": "#F59E0B",
    "WAIT": "#3B82F6",
    "OVERHEAT": "#F97316",
    "NEUTRAL": "#6B7280",
    "CARRY": "#8B5CF6",
}

def _route_kr(route_en):
    """영문 ROUTE를 한글 뱃지 텍스트로 변환"""
    return ROUTE_KR.get(str(route_en).upper().strip(), str(route_en))

def _route_desc(route_en):
    """ROUTE 설명"""
    return ROUTE_DESC.get(str(route_en).upper().strip(), "")

def _route_color(route_en):
    """ROUTE 색상"""
    return ROUTE_COLOR.get(str(route_en).upper().strip(), "#6B7280")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ══════════════════════════════════════════════════════
#  공유 UI 유틸 (main.py에서 이식)
# ══════════════════════════════════════════════════════

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


def _price_bar_html(stop, entry, close, t1, t2=0):
    points = [("손절", stop, "#EF4444"), ("매수", entry, "#3B82F6"), ("현재", close, "#FFFFFF")]
    if t1 > 0: points.append(("목표1", t1, "#10B981"))
    if t2 > 0 and t2 != t1: points.append(("목표2", t2, "#EAB308"))
    points.sort(key=lambda x: x[1])
    p_min, p_max = points[0][1] * 0.98, points[-1][1] * 1.02
    rng = p_max - p_min
    if rng <= 0: return ""

    html = '<div style="position:relative;height:90px;background:linear-gradient(90deg,rgba(239,68,68,0.15) 0%,rgba(16,185,129,0.15) 100%);border-radius:10px;margin:8px 0 20px 0;">'
    # 중앙 라인
    html += '<div style="position:absolute;top:50%;left:2%;right:2%;height:2px;background:rgba(255,255,255,0.15);"></div>'
    for i, (label, price, color) in enumerate(points):
        pct = max(3, min((price - p_min) / rng * 100, 97))
        is_cur = label == "현재"
        sz = "14px" if is_cur else "10px"
        bdr = "2px solid #FFF" if is_cur else "none"
        fw = "bold" if is_cur else "normal"
        # 짝수 → 위, 홀수 → 아래 (겹침 방지)
        if i % 2 == 0:
            dot_top = "25%"
            label_style = "bottom:100%;margin-bottom:4px;"
        else:
            dot_top = "55%"
            label_style = "top:100%;margin-top:4px;"
        html += (
            f'<div style="position:absolute;left:{pct}%;top:{dot_top};transform:translate(-50%,-50%);z-index:{"10" if is_cur else "5"};text-align:center;">'
            f'<div style="width:{sz};height:{sz};background:{color};border-radius:50%;border:{bdr};margin:0 auto;position:relative;">'
            f'<div style="position:absolute;left:50%;transform:translateX(-50%);{label_style}font-size:10px;color:{color};white-space:nowrap;font-weight:{fw};line-height:1.2;">{label}<br>{int(price):,}</div>'
            f'</div></div>'
        )
    html += '</div>'
    return html


# ══════════════════════════════════════════════════════
#  캔들차트 데이터 & 렌더링
# ══════════════════════════════════════════════════════

def _get_stock_chart_data(code):
    """캔들차트 데이터 (FDR 기반)"""
    if not FDR_OK: return None
    try:
        code_str = str(code).zfill(6)
        start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        df = fdr.DataReader(code_str, start)
        if df is None or df.empty: return None

        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()

        std20 = df['Close'].rolling(20).std()
        df['BB_UPPER'] = df['MA20'] + 2.0 * std20
        df['BB_LOWER'] = df['MA20'] - 2.0 * std20

        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr20 = tr.rolling(20).mean()
        df['KC_UPPER'] = df['MA20'] + 1.5 * atr20
        df['KC_LOWER'] = df['MA20'] - 1.5 * atr20

        delta = df['Close'].diff()
        rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()
        df['RSI14_CHART'] = 100 - (100 / (1 + rs))

        try:
            df['HMA20'] = calc_hma_series(df['Close'], 20)
        except Exception:
            pass

        change = np.sign(df['Close'].diff()).fillna(0)
        df['OBV'] = (change * df['Volume']).cumsum()

        df = calculate_supertrend(df)

        try:
            logic_w = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df_w = df.resample('W').apply(logic_w)
            df_w['WMA20'] = df_w['Close'].rolling(20).mean()
            df['WEEKLY_MA20'] = df.index.map(
                lambda x: df_w.loc[df_w.index <= x, 'WMA20'].iloc[-1]
                if not df_w.loc[df_w.index <= x, 'WMA20'].empty else np.nan
            )
        except Exception:
            pass

        return df.tail(120)
    except Exception:
        _logger.exception("get_stock_chart_data failed")
        return None


def _add_vpvr_trace(fig, df, price_bins=25):
    """가격대별 거래량 히스토그램 (VPVR)"""
    try:
        if "Volume" not in df.columns or df.empty:
            return
        lo = float(df["Low"].min())
        hi = float(df["High"].max())
        if hi <= lo:
            return

        bin_size = (hi - lo) / price_bins
        centers, volumes = [], []
        for i in range(price_bins):
            b_lo = lo + bin_size * i
            b_hi = b_lo + bin_size
            mid = (b_lo + b_hi) / 2
            mask = (df["Close"] >= b_lo) & (df["Close"] < b_hi)
            vol = int(df.loc[mask, "Volume"].sum())
            centers.append(mid)
            volumes.append(vol)

        if not volumes or max(volumes) == 0:
            return

        arr = np.array(volumes, dtype=float)
        clip_ceil = float(np.percentile(arr, 95))
        arr_clipped = np.clip(arr, 0, clip_ceil)
        arr_log = np.log1p(arr_clipped)
        max_log = arr_log.max() if arr_log.max() > 0 else 1.0

        n = len(df)
        for mid, log_vol, raw_vol in zip(centers, arr_log, volumes):
            if log_vol == 0:
                continue
            bar_w = (log_vol / max_log) * n * 0.15
            is_hv = raw_vol >= float(np.percentile(arr, 70))
            color = "rgba(251,191,36,0.55)" if is_hv else "rgba(100,116,139,0.22)"
            x_start_idx = max(0, n - 1 - int(bar_w))
            x_start = df.index[x_start_idx]
            x_end = df.index[-1]
            fig.add_shape(
                type="rect", x0=x_start, x1=x_end,
                y0=mid - bin_size * 0.45, y1=mid + bin_size * 0.45,
                fillcolor=color, line=dict(width=0), layer="below", row=1, col=1,
            )
    except Exception as e:
        _logger.debug(f"VPVR 렌더 실패: {e}")


def _plot_candle_chart(df, code, name, entry=None, stop=None, target1=None, target2=None):
    """캔들차트 렌더링"""
    if df is None or df.empty:
        return go.Figure()

    df = df.copy()
    df.index = df.index.strftime('%Y-%m-%d')
    col_map = {"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    rows = 2
    has_rsi = "RSI14_CHART" in df.columns
    if has_rsi: rows += 1
    row_heights = [0.6] + [0.4 / (rows - 1)] * (rows - 1)

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights)

    C_UP, C_DOWN = '#EF4444', '#3B82F6'

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=C_UP, increasing_fillcolor=C_UP,
        decreasing_line_color=C_DOWN, decreasing_fillcolor=C_DOWN,
        showlegend=False
    ), row=1, col=1)

    if "MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color='#FFD700', width=1.5)), row=1, col=1)
    if "WEEKLY_MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["WEEKLY_MA20"], name="주봉20선", line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dot')), row=1, col=1)

    if "BB_UPPER" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], fill='tonexty', fillcolor='rgba(33,150,243,0.07)', line=dict(width=0), name="BB", hoverinfo='skip'), row=1, col=1)

    if "Trend" in df.columns and "SuperTrend" in df.columns:
        up = df[df["Trend"] == 1]["SuperTrend"]
        if not up.empty:
            fig.add_trace(go.Scatter(x=up.index, y=up, mode='lines', line=dict(color=C_UP, width=2), name='SuperTrend'), row=1, col=1)
        dn = df[df["Trend"] == -1]["SuperTrend"]
        if not dn.empty:
            fig.add_trace(go.Scatter(x=dn.index, y=dn, mode='lines', line=dict(color=C_DOWN, width=2), showlegend=False), row=1, col=1)

    def _add_hline(val, color, label, dash="dash"):
        if val is None: return
        try:
            v = float(str(val).replace(',', ''))
            if v > 0 and not math.isnan(v):
                fig.add_hline(y=v, line_dash=dash, line_color=color, line_width=1,
                              annotation_text=label, annotation_position="top right", annotation_font_color=color)
        except Exception:
            pass

    _add_hline(entry, "#2962FF", "매수가", "solid")
    _add_hline(stop, "#FF3B30", "손절가")
    _add_hline(target1, "#00E676", "목표1")
    _add_hline(target2, "#EAB308", "목표2", "dashdot")

    cur_row = 2
    if "Volume" in df.columns:
        colors = [C_UP if c >= o else C_DOWN for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, opacity=0.6, showlegend=False), row=cur_row, col=1)
        cur_row += 1

    if has_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14_CHART"], name="RSI", line=dict(color='#E040FB', width=1.5)), row=cur_row, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, layer="below", row=cur_row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="blue", opacity=0.1, layer="below", row=cur_row, col=1)

    try:
        _add_vpvr_trace(fig, df)
    except Exception:
        pass

    fig.update_layout(
        title=dict(text=f"<b>{name}</b> ({code})", x=0.02),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        height=550, margin=dict(l=10, r=50, t=50, b=10),
        hovermode="x unified", showlegend=False, font_color="white",
    )
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.1)', side='right')
    return fig


# ══════════════════════════════════════════════════════
#  메인 렌더 함수
# ══════════════════════════════════════════════════════

def render_tab_stocks(df, auth, store=None):
    """Tab 2: AI & Quant 추천 종목

    Args:
        df: 추천 종목 DataFrame
        auth: "guest" | "free" | "pro" | "prime" | "admin"
        store: DataStore 인스턴스 (CSV 다운로드용)
    """
    # [v21.2] 시간외 가격 오버레이 → LIVE RR + ELITE + TOP_PICK 전체 재계산
    try:
        _am_path = os.path.join(DATA_DIR, "aftermarket_prices_latest.csv")
        if os.path.exists(_am_path):
            _am = pd.read_csv(_am_path, dtype={"종목코드": str})
            if "종목코드" in _am.columns and "시간외종가" in _am.columns:
                _am["종목코드"] = _am["종목코드"].astype(str).str.zfill(6)
                _price_map = dict(zip(_am["종목코드"], pd.to_numeric(_am["시간외종가"], errors="coerce")))
                if _price_map:
                    df = df.copy()
                    df["LIVE_PRICE"] = df["종목코드"].map(_price_map)
                    _has_live = df["LIVE_PRICE"].notna()
                    if _has_live.any():
                        _live = df.loc[_has_live, "LIVE_PRICE"]
                        _stop = pd.to_numeric(df.loc[_has_live, "손절가"], errors="coerce").fillna(0)
                        _tp1 = pd.to_numeric(df.loc[_has_live, "추천매도가1"], errors="coerce").fillna(0)
                        _buy = pd.to_numeric(df.loc[_has_live, "추천매수가"], errors="coerce").fillna(0)
                        _risk = (_live - _stop).clip(lower=1)
                        _reward = (_tp1 - _live).clip(lower=0)
                        # RR 재계산
                        df.loc[_has_live, "RR_NOW_TP1"] = (_reward / _risk).round(2)
                        # ENTRY_GAP 재계산
                        df.loc[_has_live, "ENTRY_GAP_PCT"] = ((_live - _buy).abs() / _buy.clip(lower=1) * 100).round(1)
                        # ELITE_SCORE 재계산
                        _s = pd.to_numeric(df.loc[_has_live, "STRUCT_SCORE"], errors="coerce").fillna(0)
                        _t = pd.to_numeric(df.loc[_has_live, "TIMING_SCORE"], errors="coerce").fillna(0)
                        _m = pd.to_numeric(df.loc[_has_live, "AI_SCORE"], errors="coerce").fillna(0)
                        _ax = ((_s + _t + _m) / 3)
                        _gap = pd.concat([_s, _t, _m], axis=1).max(axis=1) - pd.concat([_s, _t, _m], axis=1).min(axis=1)
                        _bal = (100 - _gap * 1.25).clip(0, 100)
                        _rr_sc = (df.loc[_has_live, "RR_NOW_TP1"] / 3.0 * 100).clip(0, 100)
                        _ent_sc = (100 - df.loc[_has_live, "ENTRY_GAP_PCT"] * 20).clip(0, 100)
                        _rt_map = {"ATTACK":100,"ARMED":75,"WAIT":40,"NEUTRAL":30,"OVERHEAT":10,"CARRY":35}
                        _rt_sc = df.loc[_has_live, "ROUTE"].astype(str).map(_rt_map).fillna(30)
                        _elite_live = (_ax*0.40 + _bal*0.20 + _rr_sc*0.25 + _ent_sc*0.10 + _rt_sc*0.05).round(1)
                        _elite_live = _elite_live.where(_live > _stop, 0).where(_live < _tp1, 0)
                        df.loc[_has_live, "ELITE_SCORE"] = _elite_live
                        df.loc[_has_live, "BALANCE_SCORE"] = _bal.round(1)
                        # TOP_PICK 재판정
                        _tp_mask = (
                            (_elite_live >= 60)
                            & df.loc[_has_live, "ROUTE"].isin(["ATTACK", "ARMED"])
                            & (_live > _stop) & (_live < _tp1)
                            & (_bal >= 50)
                            & (df.loc[_has_live, "RR_NOW_TP1"] >= 0.5)
                            & (df.loc[_has_live, "ENTRY_GAP_PCT"] <= 5.0)
                        )
                        df.loc[_has_live, "TOP_PICK"] = _tp_mask.astype(int)
    except Exception as _live_err:
        import logging as _lg
        _lg.getLogger(__name__).warning(f"⚠️ LIVE price overlay 실패: {_live_err}")

    _section_title("🎯 AI 추천 종목")

    _tbl = None

    with ui.row().classes("w-full gap-4 items-center flex-wrap mb-4"):
        view_mode = ui.toggle(["📋 테이블", "🃏 칸반"], value="📋 테이블")
        route_filter = ui.select({"전체": "전체", "ATTACK": "🚀 매수 돌입", "ARMED": "🔫 매수 대기", "CARRY": "📌 보유 관찰", "WAIT": "👀 관망", "NEUTRAL": "⚪ 중립"}, value="전체", label="상태").classes("min-w-[140px]")
        sort_mode = ui.toggle(["🏆 ELITE순", "🔢 점수순", "🚦 상태순"], value="🏆 ELITE순")

        def _add_checked_to_journal():
            nonlocal _tbl
            tbl = _tbl
            if tbl is None:
                ui.notify("⚠️ 테이블 뷰에서 사용해주세요", type="warning"); return
            selected = tbl.selected
            if not selected:
                ui.notify("⚠️ 종목을 먼저 체크하세요", type="warning"); return
            if not JOURNAL_OK:
                ui.notify("⚠️ trade_journal_tab 모듈 없음", type="negative"); return

            added = 0
            for sel_row in selected:
                code = str(sel_row.get("code", "")).zfill(6)
                row = df[df["종목코드"].astype(str).str.zfill(6) == code]
                if row.empty: continue
                r = row.iloc[0]
                tid = save_trade({
                    "stock_name": str(r.get("종목명", sel_row.get("name", ""))),
                    "stock_code": code,
                    "route": str(r.get("ROUTE", sel_row.get("route", ""))),
                    "score": safe_float(r.get("DISPLAY_SCORE", 0)),
                    "recommend_price": nz_num(r.get("추천매수가", 0)),
                    "actual_price": nz_num(r.get("LIVE_PRICE", r.get("종가", r.get("추천매수가", 0)))),
                    "stop_price": nz_num(r.get("손절가", 0)),
                    "target_price": nz_num(r.get("추천매도가1", 0)),
                    "qty": 0,
                    "notes": f"[자동등록] 점수 {safe_float(r.get('DISPLAY_SCORE', 0)):.0f} | {r.get('ROUTE', '')}",
                })
                if tid > 0: added += 1

            if added > 0:
                ui.notify(f"✅ {added}건 매매일지 추가 완료!", type="positive")
                tbl.selected.clear(); tbl.update()
            else:
                ui.notify("❌ 추가 실패", type="negative")

        ui.button("📝 매매일지 추가", on_click=_add_checked_to_journal).props(
            "flat dense"
        ).classes("text-yellow-400 border border-yellow-700/50").tooltip(
            "체크한 종목들을 매매일지에 자동 등록합니다"
        )

        if auth in ("prime", "admin"):
            from services.auth import premium_guard

            @premium_guard("CSV 다운로드")
            async def download_csv():
                csv_path = os.path.join(DATA_DIR, "recommend_latest.csv")
                if os.path.exists(csv_path):
                    ts = store.data_ts if store else ""
                    ui.download(csv_path, f"ldy_recommend_{ts}.csv")
                else:
                    ui.notify("CSV 파일 없음", type="warning")
            ui.button("📥 CSV 다운로드", on_click=download_csv).props("flat dense").classes("text-green-400 ml-auto")

    table_area = ui.column().classes("w-full")
    detail_area = ui.column().classes("w-full mt-4")

    def _filtered():
        fdf = df.copy()
        if route_filter.value != "전체" and "ROUTE" in fdf.columns:
            fdf = fdf[fdf["ROUTE"].astype(str).str.contains(route_filter.value, na=False)]
        if sort_mode.value == "🏆 ELITE순" and "ELITE_SCORE" in fdf.columns:
            fdf = fdf.sort_values("ELITE_SCORE", ascending=False)
        elif sort_mode.value == "🔢 점수순" and "DISPLAY_SCORE" in fdf.columns:
            fdf = fdf.sort_values("DISPLAY_SCORE", ascending=False)
        elif sort_mode.value == "🚦 상태순" and "ACTION_PRIORITY" in fdf.columns:
            _sc = "ELITE_SCORE" if "ELITE_SCORE" in fdf.columns else "DISPLAY_SCORE"
            fdf = fdf.sort_values(["ACTION_PRIORITY", _sc], ascending=[True, False])
        if auth == "guest": fdf = fdf.head(3)
        elif auth == "free": fdf = fdf.head(5)
        else: fdf = fdf.head(50)
        return fdf

    def _build_view():
        nonlocal _tbl
        table_area.clear()
        _tbl = None
        show = _filtered()
        with table_area:
            if view_mode.value == "🃏 칸반":
                _render_kanban(show)
            else:
                _render_table(show)

    def _render_table(show):
        columns = [
            {"name": "route", "label": "신호", "field": "route", "align": "center"},
            {"name": "name", "label": "종목명", "field": "name", "align": "left"},
            {"name": "elite", "label": "ELITE", "field": "elite", "align": "center", "sortable": True},
            {"name": "score", "label": "종합", "field": "score", "align": "center", "sortable": True},
            {"name": "s", "label": "S", "field": "s", "align": "center"},
            {"name": "t", "label": "T", "field": "t", "align": "center"},
            {"name": "m", "label": "AI", "field": "m", "align": "center"},
            {"name": "bal", "label": "균형", "field": "bal", "align": "center", "sortable": True},
            {"name": "rr", "label": "RR", "field": "rr", "align": "center", "sortable": True},
            {"name": "close", "label": "현재가", "field": "close", "align": "right"},
            {"name": "t1", "label": "목표가", "field": "t1", "align": "right"},
            {"name": "stop", "label": "손절", "field": "stop", "align": "right"},
        ]
        rows = []
        for _, r in show.iterrows():
            rows.append({
                "code": str(r.get("종목코드", "")).zfill(6),
                "route": _route_kr(r.get("ROUTE", "—")),
                "name": str(r.get("종목명", "—")),
                "elite": f'{safe_float(r.get("ELITE_SCORE", 0)):.0f}',
                "score": f'{safe_float(r.get("DISPLAY_SCORE", 0)):.0f}',
                "s": f'{safe_float(r.get("STRUCT_SCORE", 0)):.0f}',
                "t": f'{safe_float(r.get("TIMING_SCORE", 0)):.0f}',
                "m": f'{safe_float(r.get("AI_SCORE", r.get("ML_SCORE", 0))):.0f}',
                "bal": f'{safe_float(r.get("BALANCE_SCORE", 0)):.0f}',
                "rr": f'{safe_float(r.get("RR_NOW_TP1", 0)):.1f}',
                "close": f'{int(nz_num(r.get("LIVE_PRICE", r.get("종가", 0)))):,}',
                "t1": f'{int(nz_num(r.get("추천매도가1", 0))):,}',
                "stop": f'{int(nz_num(r.get("손절가", 0))):,}',
            })
        tbl = ui.table(columns=columns, rows=rows, row_key="code", selection="multiple",
                       pagination={"rowsPerPage": 15}).classes("w-full").props("dense dark flat bordered")
        nonlocal _tbl
        _tbl = tbl

        def _on_selection_change(e):
            sel = tbl.selected
            if sel:
                last_code = sel[-1].get("code", "")
                if last_code:
                    _on_stock_select_by_code(last_code, df)

        tbl.on("selection", _on_selection_change)

    def _render_kanban(show):
        if show.empty:
            ui.label("표시할 종목 없음").classes("text-gray-400"); return
        df_atk = show[show['ROUTE'].astype(str).str.contains("ATTACK", case=False, na=False)] if "ROUTE" in show.columns else pd.DataFrame()
        df_arm = show[show['ROUTE'].astype(str).str.contains("ARMED", case=False, na=False)] if "ROUTE" in show.columns else pd.DataFrame()
        ex = df_atk.index.union(df_arm.index) if not df_atk.empty or not df_arm.empty else pd.Index([])
        df_watch = show[~show.index.isin(ex)]

        with ui.row().classes("w-full gap-4 flex-wrap items-start"):
            _kanban_col("🚀 매수 돌입", df_atk, "#EF4444", df)
            _kanban_col("🔫 매수 대기", df_arm, "#F59E0B", df)
            _kanban_col("👀 관망", df_watch, "#3B82F6", df)

    def _kanban_col(title, sub_df, color, full_df):
        with ui.column().classes("kanban-col min-w-[280px] flex-1"):
            ui.label(f"{title} ({len(sub_df)})").classes("text-white font-bold mb-2").style(f"border-bottom: 2px solid {color}")
            if sub_df.empty:
                ui.label("비어 있음").classes("text-gray-500 text-sm"); return
            for _, r in sub_df.iterrows():
                score = safe_float(r.get("DISPLAY_SCORE", 0))
                buy = int(nz_num(r.get("추천매수가", 0)))
                stop = int(nz_num(r.get("손절가", 0)))
                t1 = int(nz_num(r.get("추천매도가1", 0)))
                sc = "#10B981" if score >= 80 else "#3B82F6" if score >= 60 else "#94A3B8"
                code = str(r.get("종목코드", "")).zfill(6)

                with ui.card().classes("p-3 mb-2 cursor-pointer bg-[rgba(255,255,255,0.05)] border border-[rgba(255,255,255,0.1)] rounded-xl hover:bg-[rgba(255,255,255,0.08)]"):
                    with ui.row().classes("justify-between items-center"):
                        ui.label(f"{r.get('종목명', '')}").classes("text-white font-bold text-sm")
                        with ui.row().classes("gap-1"):
                            if int(r.get("TOP_PICK", 0)) == 1:
                                ui.badge("🏆", color="#FFD700").classes("text-xs")
                            elite = safe_float(r.get("ELITE_SCORE", score))
                            ec = "#10B981" if elite >= 80 else "#3B82F6" if elite >= 60 else "#94A3B8"
                            ui.badge(f"E{elite:.0f}", color=ec).classes("text-xs")
                    # 3축 미니
                    _ss = safe_float(r.get("STRUCT_SCORE", 0))
                    _ts = safe_float(r.get("TIMING_SCORE", 0))
                    _ms = safe_float(r.get("AI_SCORE", r.get("ML_SCORE", 0)))
                    bal = safe_float(r.get("BALANCE_SCORE", 0))
                    ui.label(f"S{_ss:.0f} T{_ts:.0f} AI{_ms:.0f} | 균형 {bal:.0f}").classes("text-xs text-gray-400 mt-1")
                    # ELITE_REASON 직접 노출
                    _er = str(r.get("ELITE_REASON", ""))
                    if _er:
                        ui.label(_er).classes("text-xs text-cyan-400 mt-0.5")
                    if buy > 0:
                        ui.label(f"매수 {buy:,} · 손절 {stop:,} · 목표 {t1:,}").classes("text-xs text-gray-400 mt-1")
                    rr = safe_float(r.get("RR_NOW_TP1", safe_float(r.get("RR1", 0))))
                    if rr > 0:
                        rc = "#10B981" if rr >= 2 else "#F59E0B" if rr >= 1 else "#EF4444"
                        ui.label(f"손익비 {rr:.1f}:1").classes("text-xs mt-1").style(f"color:{rc}")

    def _on_stock_select_by_code(code, full_df):
        detail_area.clear()
        if not code: return
        row = full_df[full_df["종목코드"].astype(str).str.zfill(6) == code]
        if row.empty: return
        row = row.iloc[0]

        with detail_area:
            with ui.row().classes("w-full items-center justify-between mt-6 mb-2"):
                ui.label(f"🔍 {row.get('종목명', '')} ({code}) 상세 분석").classes(
                    "text-lg font-bold text-white border-b border-gray-700 pb-2")
                with ui.row().classes("gap-2"):
                    _share_url = f"/stock/{code}"
                    ui.button("🔗 공유 링크", on_click=lambda u=_share_url: ui.run_javascript(
                        f'navigator.clipboard.writeText(window.location.origin + "{u}");'
                    )).props("flat dense").classes("text-blue-400 text-xs")
                    ui.button("↗ 새 탭", on_click=lambda u=_share_url: ui.navigate.to(u)
                              ).props("flat dense").classes("text-purple-400 text-xs")

            _close = nz_num(row.get("LIVE_PRICE", row.get("종가", 0)))
            _entry = nz_num(row.get("추천매수가", 0))
            _stop = nz_num(row.get("손절가", 0))
            _t1 = nz_num(row.get("추천매도가1", 0))
            _t2 = nz_num(row.get("추천매도가2", 0))
            _atr = nz_num(row.get("TARGET_ATR", 0))

            if _close > 0 and _entry > 0:
                # [v21.0] RR은 현재가 기준으로 통일 (메인표와 일치)
                risk_now = _close - _stop if _stop > 0 else 1
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    _metric_card("🔴 손절가", f"{int(_stop):,}", f"{(_stop/_close-1)*100:+.1f}%" if _close > 0 else "", False)
                    _metric_card("🔵 매수가", f"{int(_entry):,}", f"현재가 갭 {(_close/_entry-1)*100:+.1f}%")
                    if _t1 > 0:
                        rr1 = (_t1 - _close) / risk_now if risk_now > 0 else 0
                        _metric_card("🟢 목표가 1", f"{int(_t1):,}", f"+{(_t1/_close-1)*100:.1f}% (RR {rr1:.1f}:1)")
                    if _t2 > 0 and _t2 != _t1:
                        rr2 = (_t2 - _close) / risk_now if risk_now > 0 else 0
                        _metric_card("🟡 목표가 2", f"{int(_t2):,}", f"+{(_t2/_close-1)*100:.1f}% (RR {rr2:.1f}:1)")
                    if _atr > 0 and _atr != _t1:
                        _metric_card("⚪ ATR 목표가", f"{int(_atr):,}", f"+{(_atr/_close-1)*100:.1f}%")

            if _close > 0 and _stop > 0 and _t1 > 0:
                ui.html(_price_bar_html(_stop, _entry, _close, _t1, _t2))

            with ui.card().classes("w-full p-2 bg-[#1a1a2e] mt-2"):
                ui.label("🕯️ 캔들차트 로딩 중...").classes("text-gray-400")
                chart_holder = ui.column().classes("w-full")

                async def load_chart():
                    chart_holder.clear()
                    with chart_holder:
                        cdata = await run_sync(_get_stock_chart_data, code)
                        if cdata is not None:
                            fig = _plot_candle_chart(cdata, code, row.get("종목명", ""), _entry, _stop, _t1, _t2)
                            ui.plotly(fig).classes("w-full")
                        else:
                            ui.label("📉 차트 데이터 로드 실패").classes("text-yellow-400")

                async def _safe_load_chart():
                    await asyncio.sleep(0.1)
                    try:
                        if chart_holder.is_deleted: return
                    except AttributeError:
                        pass
                    await load_chart()

                asyncio.create_task(_safe_load_chart())

            with ui.row().classes("w-full gap-4 flex-wrap mt-4"):
                with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
                    try:
                        fig_r = plot_radar_chart(row)
                        if fig_r: ui.plotly(_plotly_dark(fig_r, 300)).classes("w-full")
                    except Exception:
                        ui.label("레이더 차트 오류").classes("text-gray-500")
                with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
                    try:
                        fig_w = plot_score_waterfall(row)
                        if fig_w: ui.plotly(_plotly_dark(fig_w, 300)).classes("w-full")
                    except Exception:
                        ui.label("워터폴 차트 오류").classes("text-gray-500")

            rv = str(row.get("ROUTE", "NEUTRAL"))
            rc = _route_color(rv)
            kr_label = _route_kr(rv)
            kr_desc = _route_desc(rv)
            with ui.card().classes("w-full p-4 mt-4 rounded-xl border").style(f"border-color:{rc}; background:rgba(0,0,0,0.3)"):
                with ui.row().classes("items-center gap-3"):
                    ui.label(f"⚡ 현재 신호: {kr_label}").classes("text-lg font-bold text-white").style(f"color:{rc}")
                ui.label(kr_desc).classes("text-gray-300 text-sm mt-1")

            # ── 추천 근거 요약 — [v21.0] ELITE 기반 ──
            score = safe_float(row.get("DISPLAY_SCORE", 0))
            rr1 = safe_float(row.get("RR_NOW_TP1", safe_float(row.get("RR1", 0))))
            elite = safe_float(row.get("ELITE_SCORE", 0))
            bal = safe_float(row.get("BALANCE_SCORE", 0))
            _ss = safe_float(row.get("STRUCT_SCORE", 0))
            _ts = safe_float(row.get("TIMING_SCORE", 0))
            _ms = safe_float(row.get("AI_SCORE", row.get("ML_SCORE", 0)))
            elite_reason = str(row.get("ELITE_REASON", ""))

            reasons = []
            if int(row.get("TOP_PICK", 0)) == 1:
                reasons.append("🏆 TOP PICK — 실전 매수 최우선 후보")
            if elite >= 80: reasons.append(f"🏆 ELITE {elite:.0f}점 (최상위)")
            elif elite >= 60: reasons.append(f"🏆 ELITE {elite:.0f}점 (양호)")
            if bal >= 80: reasons.append(f"⚖️ 3축 밸런스 우수 (S{_ss:.0f}/T{_ts:.0f}/AI{_ms:.0f})")
            elif bal >= 60: reasons.append(f"⚖️ 3축 밸런스 양호")
            if score >= 80: reasons.append("📊 종합점수 상위권")
            if rr1 >= 3: reasons.append(f"💰 손익비 매우 우수 ({rr1:.1f}:1)")
            elif rr1 >= 2: reasons.append(f"💰 손익비 양호 ({rr1:.1f}:1)")
            if rv == "ATTACK": reasons.append("🚀 매수 진입 시그널 활성")
            elif rv == "ARMED": reasons.append("🔫 매수 조건 근접 (돌파 대기)")
            elif rv == "CARRY": reasons.append("📌 이전 추천 종목 — 손절/익절가 관찰 중")
            # ELITE_REASON 직접 노출
            if elite_reason:
                reasons.append(f"📋 {elite_reason}")

            if reasons:
                with ui.card().classes("w-full p-4 mt-2 bg-[#1a1a2e] border border-gray-700 rounded-xl"):
                    ui.label("💡 추천 근거").classes("text-white font-bold mb-2")
                    for reason in reasons:
                        ui.label(reason).classes("text-gray-300 text-sm py-0.5")

            if KELLY_OK:
                kelly_holder = ui.card().classes("w-full p-4 bg-[#1a1a2e] border border-yellow-700/40 rounded-xl mt-4")
                render_kelly_calculator(row.to_dict(), kelly_holder)

    for w in [view_mode, route_filter, sort_mode]:
        w.on("update:model-value", lambda _: _build_view())
    _build_view()
