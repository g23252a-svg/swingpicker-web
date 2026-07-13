# -*- coding: utf-8 -*-
"""v26.1 stock decision tab.

Only the production contract is a buy. Watch candidates are explicitly
non-actionable, and the rest of the universe is analysis rather than advice.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
from nicegui import ui

from services.recommendation_quality import production_buy_mask


def _series(df: pd.DataFrame, key: str, default: Any = "") -> pd.Series:
    if key in df.columns:
        return df[key]
    return pd.Series(default, index=df.index)


def _number(row: pd.Series, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


def _text(row: pd.Series, *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if value is not None and str(value).strip() not in {"", "nan", "None"}:
            return str(value).strip()
    return default


def _code(value: Any) -> str:
    raw = str(value or "").strip().split(".")[0]
    return raw.zfill(6) if raw.isdigit() else raw


def _stock_payload(row: pd.Series, decision: str) -> dict[str, Any]:
    is_buy = decision == "BUY"
    entry = _number(
        row,
        "PRODUCTION_ENTRY_PRICE" if is_buy else "추천매수가",
        "추천매수가",
    )
    return {
        "decision": decision,
        "code": _code(_text(row, "종목코드", default="")),
        "name": _text(row, "종목명", default="이름 미확인"),
        "quality": _number(row, "QUALITY_GUARD_SCORE", "PROFIT_RECOVERY_RANK_SCORE"),
        "display_score": _number(row, "DISPLAY_SCORE", "FINAL_SCORE"),
        "entry": entry,
        "close": _number(row, "종가", "현재가"),
        "stop": _number(row, "PRODUCTION_STOP_PRICE") if is_buy else 0.0,
        "target": _number(row, "PRODUCTION_TARGET_PRICE") if is_buy else 0.0,
        "rr": _number(row, "PRODUCTION_RR", "RR_NOW_TP1") if is_buy else 0.0,
        "weight": _number(row, "RECOMMENDED_WEIGHT_PCT") if is_buy else 0.0,
        "reason": _text(
            row,
            "QUALITY_GUARD_REASON",
            "PROFIT_RECOVERY_REASON",
            "ROUTE_REASON",
            default="최종 운영 기준 미충족",
        ),
        "sector": _text(row, "업종", "섹터", default="-"),
        "route": _text(row, "ROUTE", default="-"),
    }


def build_stock_tab_model(df: pd.DataFrame | None) -> dict[str, Any]:
    """Build a deterministic, UI-independent stock-tab view model."""
    if df is None or df.empty:
        return {
            "status": "NO_DATA",
            "buys": [],
            "watch": [],
            "analysis": [],
            "counts": {"buy": 0, "watch": 0, "analysis": 0},
            "market": {"breadth": 0.0, "macro": "UNKNOWN", "ml": "UNKNOWN", "policy": "-"},
        }

    work = df.copy()
    work["_quality"] = pd.to_numeric(
        _series(work, "QUALITY_GUARD_SCORE", 0), errors="coerce"
    ).fillna(0)
    work["_display"] = pd.to_numeric(
        _series(work, "DISPLAY_SCORE", 0), errors="coerce"
    ).fillna(0)
    work["_official"] = production_buy_mask(work)
    action = _series(work, "ACTION_DECISION", "CASH").fillna("CASH").astype(str).str.upper()
    candidate = pd.to_numeric(
        _series(work, "PROFIT_RECOVERY_CANDIDATE", 0), errors="coerce"
    ).fillna(0).eq(1)
    work["_action"] = action.mask((action == "CASH") & candidate, "WATCH")
    work = work.sort_values(["_quality", "_display"], ascending=False, kind="stable")

    buys = [
        _stock_payload(row, "BUY")
        for _, row in work[work["_official"]].iterrows()
    ]
    watch = [
        _stock_payload(row, "WATCH")
        for _, row in work[(~work["_official"]) & work["_action"].eq("WATCH")].iterrows()
    ]
    analysis = []
    for _, row in work.iterrows():
        decision = "BUY" if bool(row["_official"]) else (
            "WATCH" if row["_action"] == "WATCH" else "ANALYSIS"
        )
        analysis.append(_stock_payload(row, decision))

    breadth_values = pd.to_numeric(
        _series(work, "MARKET_BREADTH", 0), errors="coerce"
    ).dropna()
    macro_values = _series(work, "MACRO_RISK", "UNKNOWN").dropna().astype(str)
    ml_values = _series(work, "ML_STATUS", "UNKNOWN").dropna().astype(str)
    policy_values = _series(work, "QUALITY_POLICY_VERSION", "-").dropna().astype(str)
    breadth = float(breadth_values.median()) if len(breadth_values) else 0.0
    macro = macro_values.mode().iloc[0] if len(macro_values) else "UNKNOWN"
    ml_status = ml_values.mode().iloc[0] if len(ml_values) else "UNKNOWN"
    policy = policy_values.iloc[0] if len(policy_values) else "-"

    return {
        "status": "BUY" if buys else "CASH",
        "buys": buys,
        "watch": watch,
        "analysis": analysis,
        "counts": {"buy": len(buys), "watch": len(watch), "analysis": len(analysis)},
        "market": {
            "breadth": breadth,
            "macro": macro,
            "ml": ml_status,
            "policy": policy,
        },
    }


def filter_stock_rows(
    model: dict[str, Any], group: str, query: str = ""
) -> list[dict[str, Any]]:
    """Return rows for the selected lane and optional name/code query."""
    key = {"공식 매수": "buys", "관찰 후보": "watch", "전체 분석": "analysis"}.get(
        group, "analysis"
    )
    rows = list(model.get(key, []))
    needle = str(query or "").strip().casefold()
    if needle:
        rows = [
            row for row in rows
            if needle in row["name"].casefold() or needle in row["code"].casefold()
        ]
    return rows


def _money(value: float) -> str:
    return f"{value:,.0f}원" if value > 0 else "-"


def _render_metric(label: str, value: str, tone: str = "text-slate-100") -> None:
    with ui.column().classes("sp26-stock-metric gap-1 p-3 rounded-xl"):
        ui.label(label).classes("text-[11px] text-slate-400")
        ui.label(value).classes(f"text-sm md:text-base font-bold {tone}")


def _render_buy_card(stock: dict[str, Any]) -> None:
    with ui.card().classes("sp26-stock-card sp26-stock-buy w-full p-4 md:p-5 rounded-2xl"):
        with ui.row().classes("w-full justify-between items-start gap-3 flex-wrap"):
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    ui.badge("공식 매수", color="#059669").classes("font-bold")
                    ui.label(stock["name"]).classes("text-xl font-bold text-white")
                    ui.label(stock["code"]).classes("text-xs text-slate-400")
                ui.label(stock["reason"]).classes("text-sm text-emerald-100/90 leading-relaxed")
            with ui.column().classes("items-end gap-0"):
                ui.label(f"품질 {stock['quality']:.0f}").classes("text-lg font-bold text-emerald-300")
                ui.label(f"최대 비중 {stock['weight']:.0f}%").classes("text-xs text-slate-300")
        with ui.grid(columns=4).classes("sp26-stock-price-grid w-full gap-2 mt-3"):
            _render_metric("지정 매수가", _money(stock["entry"]), "text-blue-300")
            _render_metric("손절가", _money(stock["stop"]), "text-rose-300")
            _render_metric("1차 목표가", _money(stock["target"]), "text-emerald-300")
            _render_metric("손익비", f"{stock['rr']:.2f}:1", "text-white")
        ui.button(
            "근거·차트 보기",
            icon="open_in_new",
            on_click=lambda code=stock["code"]: ui.navigate.to(f"/stock/{code}"),
        ).props("flat dense no-caps").classes("self-end text-blue-300 mt-2")


def _render_watch_card(stock: dict[str, Any], rank: int) -> None:
    with ui.card().classes("sp26-stock-card sp26-stock-watch w-full p-4 rounded-2xl"):
        with ui.row().classes("w-full justify-between items-start gap-3"):
            with ui.column().classes("gap-1 min-w-0"):
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    ui.badge("매수 아님", color="#B45309").classes("font-bold")
                    ui.label(f"{rank}. {stock['name']}").classes("text-base md:text-lg font-bold text-white")
                    ui.label(stock["code"]).classes("text-xs text-slate-500")
                ui.label(stock["reason"]).classes("text-sm text-slate-300 leading-relaxed")
            ui.label(f"품질 {stock['quality']:.0f}").classes("shrink-0 text-sm font-bold text-amber-300")
        with ui.row().classes("w-full justify-between items-center gap-3 mt-2 flex-wrap"):
            ui.label(f"관찰 기준가 {_money(stock['entry'])}").classes("text-xs text-slate-400")
            ui.button(
                "상세 보기",
                on_click=lambda code=stock["code"]: ui.navigate.to(f"/stock/{code}"),
            ).props("flat dense no-caps").classes("text-xs text-slate-300")


def _render_analysis_card(stock: dict[str, Any]) -> None:
    label = {"BUY": "공식 매수", "WATCH": "관찰", "ANALYSIS": "분석만"}[stock["decision"]]
    color = {"BUY": "#059669", "WATCH": "#B45309", "ANALYSIS": "#475569"}[stock["decision"]]
    with ui.card().classes("sp26-stock-card w-full p-3 rounded-xl"):
        with ui.row().classes("w-full items-center justify-between gap-2"):
            with ui.row().classes("items-center gap-2 min-w-0"):
                ui.badge(label, color=color)
                ui.label(stock["name"]).classes("font-bold text-slate-100 truncate")
                ui.label(stock["code"]).classes("text-xs text-slate-500")
            ui.label(f"품질 {stock['quality']:.0f}").classes("text-sm text-slate-300 shrink-0")


def render_stock_decision_tab(df: pd.DataFrame, auth: str = "free", store=None) -> None:
    """Render the redesigned stock tab. ``auth`` and ``store`` are API-stable."""
    del auth, store
    model = build_stock_tab_model(df)
    market = model["market"]
    # Always start from the official contract.  On a cash day the nearest
    # watch candidates are already visible above, so selecting WATCH here
    # would duplicate the same cards and make the page noisy again.
    default_group = "공식 매수"

    ui.add_head_html(
        """
        <style>
          .sp26-stock-shell{max-width:1180px;margin:0 auto;}
          .sp26-stock-hero{background:#0f172a!important;border:1px solid rgba(148,163,184,.22);box-shadow:none!important;}
          .sp26-stock-summary{background:rgba(15,23,42,.72)!important;border:1px solid rgba(148,163,184,.16);box-shadow:none!important;}
          .sp26-stock-card{background:#111827!important;border:1px solid rgba(148,163,184,.18);box-shadow:none!important;}
          .sp26-stock-buy{border-left:4px solid #10b981!important;}
          .sp26-stock-watch{border-left:4px solid #f59e0b!important;}
          .sp26-stock-metric{background:rgba(2,6,23,.42);border:1px solid rgba(148,163,184,.12);}
          .sp26-stock-controls .q-btn{min-height:38px;}
          @media(max-width:720px){
            .sp26-stock-price-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important;}
            .sp26-stock-summary-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important;}
            .sp26-stock-controls{align-items:stretch!important;}
            .sp26-stock-controls>*{width:100%;}
          }
        </style>
        """
    )

    with ui.column().classes("sp26-stock-shell w-full gap-4 pb-10"):
        with ui.card().classes("sp26-stock-hero w-full p-5 md:p-7 rounded-3xl"):
            with ui.row().classes("w-full justify-between items-start gap-4 flex-wrap"):
                with ui.column().classes("gap-2 max-w-3xl"):
                    ui.label("STOCK DECISION").classes("text-xs font-bold tracking-[0.18em] text-slate-500")
                    ui.label("종목 의사결정").classes("text-2xl md:text-4xl font-bold text-white")
                    if model["status"] == "BUY":
                        ui.label(f"지금 살 종목 {model['counts']['buy']}개").classes("text-lg md:text-2xl font-bold text-emerald-300")
                        ui.label("공식 매수 카드의 지정가·손절가·목표가·최대 비중만 실제 주문 기준입니다.").classes("text-sm text-slate-300")
                    elif model["status"] == "NO_DATA":
                        ui.label("추천 데이터가 없습니다").classes("text-lg md:text-2xl font-bold text-rose-300")
                        ui.label("데이터 수집 상태를 확인해 주세요.").classes("text-sm text-slate-300")
                    else:
                        ui.label("오늘 공식 매수 0개 · 현금 유지").classes("text-lg md:text-2xl font-bold text-amber-300")
                        ui.label("관찰 후보는 조건에 가까운 종목일 뿐 매수 추천이 아닙니다. 차단 사유가 해소되기 전에는 주문하지 않습니다.").classes("text-sm text-slate-300 leading-relaxed")
                ui.badge("BUY" if model["status"] == "BUY" else "CASH", color="#059669" if model["status"] == "BUY" else "#B45309").classes("text-sm font-bold px-3 py-1")

        with ui.grid(columns=4).classes("sp26-stock-summary-grid w-full gap-2"):
            for label, value, tone in [
                ("공식 매수", f"{model['counts']['buy']}개", "text-emerald-300"),
                ("관찰 후보", f"{model['counts']['watch']}개", "text-amber-300"),
                ("상승 종목 비율", f"{market['breadth']:.1f}%", "text-sky-300"),
                ("시장 위험", str(market["macro"]), "text-slate-200"),
            ]:
                with ui.card().classes("sp26-stock-summary p-3 md:p-4 rounded-xl"):
                    ui.label(label).classes("text-[11px] text-slate-500")
                    ui.label(value).classes(f"text-base md:text-lg font-bold {tone}")

        with ui.row().classes("w-full gap-2 flex-wrap"):
            ui.badge(f"운영 정책 {market['policy']}", color="#334155")
            ml_ok = str(market["ml"]).upper() == "VALIDATED"
            ui.badge("ML 검증 통과" if ml_ok else "ML 점수는 매수 판단에서 제외", color="#2563EB" if ml_ok else "#B45309")
            ui.badge("신규 진입 하루 최대 1종목", color="#334155")
            ui.badge("종목당 최대 3%", color="#334155")

        if not model["buys"] and model["watch"]:
            ui.label("매수 조건에 가장 가까운 관찰 후보").classes("text-lg font-bold text-white mt-1")
            ui.label("아래 종목은 매수 추천이 아닙니다. 각 카드에서 진입이 막힌 이유를 먼저 확인하세요.").classes("text-sm text-slate-400")
            for rank, stock in enumerate(model["watch"][:6], 1):
                _render_watch_card(stock, rank)

        ui.separator().classes("bg-slate-700/70 my-2")
        ui.label("종목 목록").classes("text-xl font-bold text-white")
        with ui.row().classes("sp26-stock-controls w-full items-center gap-3 flex-wrap"):
            mode = ui.toggle(["공식 매수", "관찰 후보", "전체 분석"], value=default_group).props("no-caps")
            search = ui.input("종목명 또는 코드 검색", placeholder="예: 삼성전자 / 005930").props("dense clearable outlined").classes("min-w-[260px] grow")
        results = ui.column().classes("w-full gap-2")

        def refresh() -> None:
            rows = filter_stock_rows(model, mode.value, search.value)
            results.clear()
            with results:
                ui.label(f"{len(rows)}개 표시").classes("text-xs text-slate-500")
                if not rows:
                    with ui.card().classes("sp26-stock-card w-full p-5 rounded-2xl"):
                        message = "공식 매수 종목이 없습니다. 현금을 유지하세요." if mode.value == "공식 매수" else "조건에 맞는 종목이 없습니다."
                        ui.label(message).classes("font-bold text-slate-300")
                elif mode.value == "공식 매수":
                    for stock in rows:
                        _render_buy_card(stock)
                elif mode.value == "관찰 후보":
                    for rank, stock in enumerate(rows[:20], 1):
                        _render_watch_card(stock, rank)
                else:
                    for stock in rows[:50]:
                        _render_analysis_card(stock)
                    if len(rows) > 50:
                        ui.label(f"성능을 위해 상위 50개만 표시했습니다. 검색으로 {len(rows) - 50}개 종목을 더 찾을 수 있습니다.").classes("text-xs text-slate-500")

        mode.on_value_change(lambda _: refresh())
        search.on_value_change(lambda _: refresh())
        refresh()

        ui.label("수익률을 보장하지 않습니다. 공식 매수 여부와 지정가·손절가·비중을 함께 지켜야 검증 결과와 실제 운용의 차이를 줄일 수 있습니다.").classes("text-xs text-slate-600 mt-2")
