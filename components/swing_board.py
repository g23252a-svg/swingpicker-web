"""Swing-entry shortlist. Forecast ranking never changes the execution contract."""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd
from nicegui import ui

from services.recommendation_quality import production_buy_mask
from services.snapshot_integrity import normalize_ymd, snapshot_date


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _code(value: Any) -> str:
    value = re.sub(r"\.0$", "", _text(value)).upper()
    return value.zfill(6) if re.fullmatch(r"(?:[0-9]{1,6}|[0-9A-Z]{6})", value) else ""


def _row_payload(row: pd.Series, official: bool) -> dict[str, Any]:
    status = _text(row.get("SWING_STATUS")).upper()
    trade_validated = _text(row.get("SWING_TRADE_VALIDATED")).lower() in {"1", "1.0", "true"}
    probability = _finite(row.get("SWING_PROB"))
    # Accuracy alone can coexist with negative expectancy. The shortlist only
    # exposes probabilities after both forecasting and net-return validation.
    if status != "VALIDATED" or not trade_validated or probability is None or not 0 <= probability <= 1:
        probability = None
    score = _finite(row.get("SWING_SCORE"))
    if score is not None and not 0 <= score <= 100:
        score = None
    return {
        "code": _code(row.get("종목코드")),
        "name": _text(row.get("종목명")) or _code(row.get("종목코드")),
        "official": bool(official),
        "close": _finite(row.get("종가")),
        "probability": probability,
        "score": score,
        "status": "VALIDATED" if probability is not None else "RESEARCH",
        "reason": _text(row.get("SWING_REASON")),
        "source_rank": _finite(row.get("SWING_RANK")),
        "trade_validated": trade_validated,
    }


def build_swing_board(df: pd.DataFrame, summary: dict | None = None,
                      limit: int = 10) -> dict[str, Any]:
    """Use validated predictions; missing/failed models preserve the existing shortlist.

    Only the newest dated model cohort is eligible. A stale VALIDATED row cannot
    fill a hole left by a newer unavailable/research model. Probabilities are
    never inferred from alpha, quality, rank, or a historical group hit rate.
    Failed-model rows are returned only for an explicitly collapsed research view.
    """
    summary = summary or {}
    work = df.copy().reset_index(drop=True) if df is not None else pd.DataFrame()
    official = production_buy_mask(work) if not work.empty else pd.Series(dtype=bool)
    out = {"mode": "FALLBACK", "asof": "", "rows": [], "total": 0,
           "research_rows": [], "research_total": 0, "model_report": False,
           "validation_reason": "", "oos_coverage": None, "oos_universe_return": None,
           "validation_days": None, "oos_hit_rate": None, "oos_net_return": None,
           "trade_validated": False}
    if not work.empty and "SWING_STATUS" in work:
        dates = work.get("SWING_ASOF", pd.Series("", index=work.index)).map(normalize_ymd)
        newest = dates.max()
        # A cached forecast belongs to exactly one snapshot, not the wall clock.
        # Both older and future forecasts are rejected, as are undated batches.
        batch_date = snapshot_date(work)
        if newest and newest == batch_date:
            cohort = work.loc[dates.eq(newest)].copy()
            first = cohort.iloc[0]
            out["asof"] = newest
            out["model_report"] = True
            out["validation_reason"] = _text(first.get("SWING_VALIDATION_REASON"))
            out["validation_days"] = _finite(first.get("SWING_VALIDATION_DAYS"))
            hit_rate = _finite(first.get("SWING_OOS_HIT_RATE"))
            out["oos_hit_rate"] = hit_rate if hit_rate is not None and 0 <= hit_rate <= 1 else None
            out["oos_net_return"] = _finite(first.get("SWING_OOS_NET_RETURN"))
            out["oos_universe_return"] = _finite(first.get("SWING_OOS_UNIVERSE_NET_RETURN"))
            coverage = _finite(first.get("SWING_OOS_COVERAGE"))
            out["oos_coverage"] = coverage if coverage is not None and 0 <= coverage <= 1 else None
            ranks = pd.to_numeric(cohort.get("SWING_RANK", pd.Series(float("nan"), index=cohort.index)), errors="coerce")
            statuses = cohort["SWING_STATUS"].fillna("").astype(str).str.upper()
            cohort = cohort.loc[statuses.isin(["VALIDATED", "RESEARCH"]) & ranks.ge(1) & ranks.lt(float("inf"))]
            cohort = cohort.assign(_rank=ranks.reindex(cohort.index),
                                   _code=cohort.get("종목코드", pd.Series("", index=cohort.index)).map(_code))
            cohort = cohort.sort_values(["_rank", "_code"], kind="stable")
            seen: set[str] = set()
            rows = []
            for index, row in cohort.iterrows():
                item = _row_payload(row, bool(official.loc[index]))
                if not item["code"] or item["code"] in seen:
                    continue
                if item["probability"] is None and item["score"] is None:
                    continue
                seen.add(item["code"])
                rows.append(item)
            if rows:
                validated = [item for item in rows if item["probability"] is not None]
                out["research_rows"] = [item for item in rows if item["probability"] is None]
                # Failed research must never replace the user's existing picks,
                # even when it has a high relative score or a prominent rank.
                if validated:
                    out.update(mode="MODEL", rows=validated, trade_validated=True)

    if not out["rows"]:
        # The summary has already applied the production and measured-depth
        # contracts. Preserve that selection instead of promoting raw ranks.
        official_codes = {_code(work.loc[i].get("종목코드")) for i in official.index[official]}
        seen = set()
        for stock in [*(summary.get("buys") or []), *(summary.get("watch") or [])]:
            code = _code(stock.get("code"))
            if not code or code in seen:
                continue
            seen.add(code)
            out["rows"].append({
                "code": code, "name": _text(stock.get("name")) or code,
                "official": code in official_codes, "close": _finite(stock.get("close")),
                "probability": None, "score": None, "status": "FALLBACK",
                "reason": _text(stock.get("reason")), "source_rank": None,
                "trade_validated": False,
            })
    out["total"] = len(out["rows"])
    out["rows"] = out["rows"][:max(1, min(int(limit), 10))]
    for rank, item in enumerate(out["rows"], 1):
        item["rank"] = rank
    out["research_total"] = len(out["research_rows"])
    out["research_rows"] = out["research_rows"][:max(1, min(int(limit), 10))]
    for rank, item in enumerate(out["research_rows"], 1):
        item["rank"] = rank
    return out


def _signal_text(stock: dict[str, Any]) -> tuple[str, str]:
    if stock["probability"] is not None:
        return f"{stock['probability'] * 100:.1f}%", "비용 후 스윙 수익 확률"
    if stock["score"] is not None:
        return f"{stock['score']:.1f}점", "검증 중 · 상대순위"
    return "기존 선별", "스윙 확률 미제공"


def _render_rows(rows: list[dict[str, Any]]) -> None:
    for stock in rows:
        with ui.element("div").classes("sp-nd-row w-full"):
            ui.label(f"{stock['rank']:02d}").classes("sp-nd-rank")
            with ui.column().classes("sp-nd-name gap-1 min-w-0"):
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    ui.link(stock["name"], f"/stock/{stock['code']}").classes("text-base font-bold text-white no-underline")
                    ui.label(stock["code"]).classes("text-xs text-slate-500")
                    ui.badge("매수 가능 · 공식" if stock["official"] else "관찰 · 매수 아님",
                             color="#047857" if stock["official"] else "#475569").classes("text-sm")
                if stock["reason"]:
                    ui.label(stock["reason"]).classes("text-sm text-slate-400 leading-relaxed break-words")
            signal, label = _signal_text(stock)
            with ui.column().classes("sp-nd-signal gap-0"):
                ui.label(signal).classes("text-xl font-bold text-sky-200 tabular-nums")
                ui.label(label).classes("text-sm text-slate-400")
            with ui.column().classes("sp-nd-close gap-0"):
                close = stock.get("close")
                ui.label(f"{close:,.0f}원" if close is not None and close > 0 else "—").classes("text-sm font-semibold text-slate-200 tabular-nums")
                ui.label("기준 종가").classes("text-xs text-slate-500")
            ui.button("상세", on_click=lambda code=stock["code"]: ui.navigate.to(f"/stock/{code}")).props("flat dense no-caps").classes("sp-nd-link text-sky-300")


def render_swing_board(df: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    board = build_swing_board(df, summary)
    ui.add_head_html("""<style>
      .sp-nd-board{background:#101a2b!important;border:1px solid #273a55;box-shadow:none!important;}
      .sp-nd-row{display:grid;grid-template-columns:30px minmax(0,1fr) 178px 100px 48px;
        gap:16px;align-items:center;padding:18px 0;border-top:1px solid rgba(148,163,184,.13);}
      .sp-nd-rank{font-size:15px;font-weight:800;color:#64748b;font-variant-numeric:tabular-nums;}
      @media(max-width:720px){
        .sp-nd-row{grid-template-columns:24px minmax(0,1fr) 68px;gap:10px;padding:16px 0;}
        .sp-nd-name{grid-column:2/4}.sp-nd-signal{grid-column:2;grid-row:auto;}
        .sp-nd-close{grid-column:3;grid-row:auto}.sp-nd-link{grid-column:3;justify-self:end;}
      }
    </style>""")
    with ui.card().classes("sp-nd-board w-full rounded-2xl p-4 md:p-6 gap-3"):
        with ui.row().classes("w-full items-center justify-between gap-3 flex-wrap"):
            with ui.column().classes("gap-1"):
                ui.label("다음 거래일 진입 후보").classes("text-xl md:text-2xl font-bold text-white")
                if board["asof"]:
                    asof = board["asof"]
                    ui.label(f"{asof[:4]}.{asof[4:6]}.{asof[6:]} 종가 기준").classes("text-xs text-slate-400")
            has_probability = any(r["probability"] is not None for r in board["rows"])
            ui.badge("스윙 예측 · 검증 통과" if has_probability else "기존 엔진 후보",
                     color="#1D4ED8" if has_probability else "#475569").classes("text-sm")
        description = "종가 분석 → 다음 거래일 진입 · 약 5거래일 보유. "
        if has_probability:
            description += "표시 확률은 비용 후 스윙 수익이 플러스일 추정치입니다."
        else:
            description += "기존 공식 매수·관찰 후보를 표시합니다. 검증된 수익 확률은 아직 제공하지 않습니다."
        ui.label(description).classes("text-sm text-slate-400 leading-relaxed")
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            count = summary.get("production_count", 0)
            ui.badge(f"공식 매수 {count}개", color="#047857" if count else "#92400E").classes("text-sm")
            ui.label("공식 매수의 지정가·손절가·수량은 아래 매수 계획에서 확인" if count else
                     "현재 공식 매수 없음 · 아래 순위는 관찰용입니다").classes("text-sm text-slate-300")
            actionable = (summary.get("holdings") or {}).get("actionable", 0)
            if actionable:
                ui.badge(f"보유 조치 {actionable}건 · 아래 관리 상세 확인", color="#92400E").classes("text-sm")
        if board["mode"] == "FALLBACK" and board["model_report"]:
            ui.label("신규 모델 검증 미달 · 기존 선별 사용").classes("text-sm text-amber-200")

        holder = ui.column().classes("w-full gap-0")

        def show_rows(count: int = 5) -> None:
            holder.clear()
            with holder:
                if board["rows"]:
                    _render_rows(board["rows"][:count])
                else:
                    ui.label("현재 선별 기준을 통과한 후보가 없습니다. 다음 종가 데이터 갱신 후 확인하세요.").classes("text-sm text-slate-300 py-5")

        if len(board["rows"]) > 5:
            with ui.row().classes("w-full items-center justify-end gap-3"):
                ui.label("표시 개수").classes("text-xs text-slate-500")
                ui.toggle({5: "상위 5개", 10: "상위 10개"}, value=5,
                          on_change=lambda e: show_rows(int(e.value))).props("dense no-caps")
        show_rows()
        if board["model_report"]:
            title = "연구 모델 순위 · 검증 결과" if board["research_rows"] or board["mode"] == "FALLBACK" else "스윙 예측 검증 결과"
            with ui.expansion(title, icon="analytics", value=False).classes("w-full text-xs text-slate-400").props("dense"):
                if board["research_rows"]:
                    ui.label("검증 미달 연구 모델입니다. 아래 상대순위는 기본 추천과 매수 판단에 반영하지 않습니다.").classes("text-sm text-amber-200")
                if board["validation_reason"]:
                    ui.label(board["validation_reason"]).classes("text-sm text-slate-300")
                ui.label("다음 거래일 시가 진입 · 최대 5거래일 · 손절 반영 · 왕복 비용 0.3% 가정").classes("text-xs")
                ui.label("손절 기준 -8% · 갭 하락 시 실제 손실은 더 커질 수 있습니다.").classes("text-xs")
                days = board["validation_days"]
                if days is not None and days >= 0:
                    ui.label(f"과거 검증 {int(days)}거래일").classes("text-xs")
                hit = board["oos_hit_rate"]
                if hit is not None:
                    ui.label(f"과거 검증 스윙 수익 적중률 {hit * 100:.1f}% · 개별 종목 확률과 다름").classes("text-xs")
                ret = board["oos_net_return"]
                if ret is not None:
                    ui.label(f"과거 다음 거래일 진입·약 5거래일 보유 평균 수익 {ret * 100:+.2f}% · 비용 반영").classes("text-xs")
                universe = board["oos_universe_return"]
                if universe is not None:
                    ui.label(f"같은 검증 구간 유니버스 평균 수익 {universe * 100:+.2f}%").classes("text-xs")
                coverage = board["oos_coverage"]
                if coverage is not None:
                    ui.label(f"검증 대상 결과 확인 비율 {coverage * 100:.1f}%").classes("text-xs")
                ui.label("수익 확률의 정확도와 평균 매매 수익의 우위는 별도입니다. 검증 중 점수는 확률로 해석하지 마세요.").classes("text-xs")
                if board["research_rows"]:
                    _render_rows(board["research_rows"])
    return board
