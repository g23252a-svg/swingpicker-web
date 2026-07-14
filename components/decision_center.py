# -*- coding: utf-8 -*-
"""Single-decision home screen for SwingPicker.

The first screen answers one question: what, if anything, is safe enough to
buy today?  Research lanes and raw ranks remain available elsewhere, but are
never presented as equivalent to an official production decision here.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
from nicegui import ui

from services.recommendation_quality import production_buy_mask


def _number(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _stock_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "code": str(row.get("종목코드", "")).split(".")[0].zfill(6),
        "name": str(row.get("종목명", "-")),
        "score": _number(row, "QUALITY_GUARD_SCORE"),
        "display_score": _number(row, "DISPLAY_SCORE"),
        "entry": _number(row, "추천매수가"),
        "close": _number(row, "종가"),
        "stop": _number(row, "손절가"),
        "target": _number(row, "추천매도가1"),
        "target2": _number(row, "추천매도가2"),
        "rr": _number(row, "RR_NOW_TP1"),
        "weight": _number(row, "RECOMMENDED_WEIGHT_PCT"),
        "reason": str(row.get("QUALITY_GUARD_REASON", "")),
        "route": str(row.get("ROUTE", "")),
        "decision": str(row.get("ACTION_DECISION", "CASH")),
    }


def build_decision_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "status": "NO_DATA",
            "title": "추천 데이터를 확인할 수 없습니다",
            "subtitle": "새로고침 또는 수집 상태를 확인하세요.",
            "buys": [],
            "watch": [],
            "production_count": 0,
            "watch_count": 0,
            "ml_status": "NO_DATA",
        }

    work = df.copy()
    production = production_buy_mask(work)
    quality = pd.to_numeric(
        work.get("QUALITY_GUARD_SCORE", pd.Series(0, index=work.index)),
        errors="coerce",
    ).fillna(0)
    action = work.get(
        "ACTION_DECISION", pd.Series("CASH", index=work.index)
    ).fillna("CASH").astype(str).str.upper()
    work["_quality"] = quality

    buys_df = work[production].sort_values("_quality", ascending=False).head(3)
    watch_df = work[(~production) & action.eq("WATCH")].sort_values(
        "_quality", ascending=False
    ).head(3)
    if watch_df.empty:
        watch_df = work[~production].sort_values("_quality", ascending=False).head(3)

    buys = [_stock_payload(row) for _, row in buys_df.iterrows()]
    watch = [_stock_payload(row) for _, row in watch_df.iterrows()]
    ml_values = (
        work.get("ML_STATUS", pd.Series("UNKNOWN", index=work.index))
        .fillna("UNKNOWN")
        .astype(str)
    )
    ml_status = ml_values.iloc[0] if len(ml_values) else "UNKNOWN"

    if buys:
        title = f"오늘 신규매수 {len(buys)}개"
        subtitle = "아래 종목만 최종 품질게이트를 통과했습니다. 지정가와 최대 비중을 지키세요."
        status = "BUY"
    else:
        title = "오늘은 신규매수하지 않습니다"
        nearest = watch[0]["reason"] if watch else "통과 후보 없음"
        subtitle = f"현금 보유가 공식 결정입니다. 가장 가까운 후보도 ‘{nearest}’로 차단됐습니다."
        status = "CASH"

    return {
        "status": status,
        "title": title,
        "subtitle": subtitle,
        "buys": buys,
        "watch": watch,
        "production_count": int(production.sum()),
        "watch_count": int(action.eq("WATCH").sum()),
        "ml_status": ml_status,
    }


def _money(value: float) -> str:
    return f"{value:,.0f}원" if value > 0 else "—"


def _move_from_entry(entry: float, price: float) -> str:
    if entry <= 0 or price <= 0:
        return ""
    move = (price / entry - 1.0) * 100.0
    return f"{move:+.1f}% vs 진입"


def _render_buy_card(stock: dict[str, Any]) -> None:
    with ui.card().classes("sp-buy-card w-full p-5 rounded-2xl"):
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.badge("공식 매수", color="#10B981").classes("font-bold")
                    ui.label(stock["name"]).classes("text-xl font-bold text-white")
                    ui.label(stock["code"]).classes("text-xs text-slate-400")
                ui.label(stock["reason"]).classes("text-sm text-emerald-100")
            with ui.column().classes("items-end gap-0"):
                ui.label(f"품질 {stock['score']:.0f}").classes("text-xl font-bold text-emerald-300")
                ui.label(f"최대 {stock['weight']:.0f}% 비중").classes("text-xs text-slate-300")
                ui.label(f"손익비 {stock['rr']:.2f}:1").classes("text-xs text-slate-400")
        with ui.grid(columns=4).classes("sp-price-grid w-full gap-2 mt-3"):
            for label, value, detail, css in [
                ("지정 매수가", _money(stock["entry"]), "익일 지정가 기준", "text-blue-300"),
                ("손절가", _money(stock["stop"]), _move_from_entry(stock["entry"], stock["stop"]), "text-rose-300"),
                ("1차 익절", _money(stock["target"]), _move_from_entry(stock["entry"], stock["target"]), "text-emerald-300"),
                ("연장 목표", _money(stock["target2"]), _move_from_entry(stock["entry"], stock["target2"]), "text-violet-300"),
            ]:
                with ui.column().classes("sp-price-cell gap-0 rounded-xl p-3"):
                    ui.label(label).classes("text-[11px] text-slate-400")
                    ui.label(value).classes(f"text-base font-bold {css}")
                    ui.label(detail or "—").classes("text-[10px] text-slate-500")
        ui.button(
            "근거와 차트 보기",
            on_click=lambda code=stock["code"]: ui.navigate.to(f"/stock/{code}"),
        ).props("flat dense no-caps").classes("self-end text-blue-300 mt-2")


def _render_watch_card(stock: dict[str, Any], rank: int) -> None:
    with ui.card().classes("sp-watch-card w-full p-4 rounded-xl"):
        with ui.row().classes("w-full items-center justify-between gap-2"):
            with ui.row().classes("items-center gap-2"):
                ui.badge("매수 아님", color="#475569")
                ui.label(f"{rank}. {stock['name']}").classes("font-bold text-slate-100")
                ui.label(stock["code"]).classes("text-xs text-slate-500")
            ui.label(f"품질 {stock['score']:.0f}").classes("text-sm font-bold text-amber-300")
        ui.label(stock["reason"] or "최종 기준 미달").classes("text-sm text-slate-400 mt-1")


def render_decision_center(df: pd.DataFrame, auth: str = "free") -> None:
    del auth  # reserved for future entitlement-specific detail rows
    summary = build_decision_summary(df)
    ui.add_head_html(
        """
        <style>
          .sp-decision-shell { max-width: 1120px; margin: 0 auto; }
          .sp-decision-hero { background:linear-gradient(135deg,#0f172a 0%,#111c35 55%,#172033 100%)!important;
            border:1px solid rgba(148,163,184,.18); box-shadow:0 18px 50px rgba(0,0,0,.25); }
          .sp-buy-card { background:linear-gradient(135deg,rgba(6,78,59,.72),rgba(15,23,42,.95))!important;
            border:1px solid rgba(52,211,153,.4); }
          .sp-watch-card { background:rgba(15,23,42,.72)!important; border:1px solid rgba(100,116,139,.22); }
          .sp-price-cell { background:rgba(15,23,42,.55); border:1px solid rgba(148,163,184,.12); }
          @media(max-width:720px){ .sp-price-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important;} }
        </style>
        """
    )

    with ui.column().classes("sp-decision-shell w-full gap-4 pb-8"):
        cash = summary["status"] != "BUY"
        with ui.card().classes("sp-decision-hero w-full p-6 md:p-8 rounded-3xl"):
            with ui.row().classes("w-full items-start justify-between gap-4 flex-wrap"):
                with ui.column().classes("gap-2 max-w-3xl"):
                    ui.label("오늘의 최종 결정").classes("text-xs font-bold tracking-[0.18em] text-slate-400 uppercase")
                    with ui.row().classes("items-center gap-3"):
                        ui.label("⏸" if cash else "✓").classes(
                            "text-4xl " + ("text-amber-300" if cash else "text-emerald-300")
                        )
                        ui.label(summary["title"]).classes(
                            "text-2xl md:text-4xl font-bold "
                            + ("text-amber-200" if cash else "text-emerald-200")
                        )
                    ui.label(summary["subtitle"]).classes("text-sm md:text-base text-slate-300 leading-relaxed")
                with ui.column().classes("items-end gap-1"):
                    ui.badge(
                        "CASH" if cash else "BUY",
                        color="#F59E0B" if cash else "#10B981",
                    ).classes("text-sm font-bold px-3 py-1")
                    ui.label(f"관찰 {summary['watch_count']}개").classes("text-xs text-slate-500")

        with ui.row().classes("w-full gap-2 flex-wrap"):
            ui.badge(f"공식 매수 {summary['production_count']}개", color="#10B981" if not cash else "#475569")
            ml_ok = summary["ml_status"] == "VALIDATED"
            ui.badge(
                "ML 검증 통과" if ml_ok else "ML 추천 가중치 제외",
                color="#2563EB" if ml_ok else "#B45309",
            )
            ui.badge("하루 신규진입 최대 1종목", color="#334155")
            ui.badge("종목당 최대 5%", color="#334155")

        if summary["buys"]:
            ui.label("지금 살 수 있는 종목").classes("text-lg font-bold text-white mt-2")
            for stock in summary["buys"]:
                _render_buy_card(stock)
        else:
            with ui.card().classes("w-full p-4 rounded-2xl bg-amber-950/20 border border-amber-500/20"):
                ui.label("왜 현금 보유인가요?").classes("font-bold text-amber-200")
                ui.label(
                    "좋은 후보가 없을 때 종목 수를 채우지 않습니다. 높은 원점수·AI 점수·관찰 후보는 공식 매수와 다릅니다."
                ).classes("text-sm text-slate-300 mt-1")

        if summary["watch"]:
            with ui.expansion("가장 가까운 관찰 후보 (매수 추천 아님)", icon="visibility").classes(
                "w-full rounded-xl border border-slate-700/60"
            ):
                with ui.column().classes("w-full gap-2 p-2"):
                    for rank, stock in enumerate(summary["watch"], 1):
                        _render_watch_card(stock, rank)

        with ui.row().classes("w-full items-center justify-between gap-3 flex-wrap mt-2"):
            ui.label("전체 후보와 시장 지표는 ‘시장’ 및 ‘종목’ 탭에서 확인할 수 있습니다.").classes("text-xs text-slate-500")
            ui.label("성과 보장 아님 · 지정가/손절/비중 준수").classes("text-xs text-slate-600")

