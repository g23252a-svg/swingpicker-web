"""
components/portfolio_swap.py
=============================
[v3.9.21] 보유종목 vs 신규추천 교체 판단 — UI 렌더 모듈.

로직(analyze_portfolio_swap, derive_holding_verdict)은 services로 분리.
이 모듈은 _render_portfolio_swap_card만 보유.

Backward compat re-export:
- _analyze_portfolio_swap, _derive_holding_verdict 등은 services에서 re-export.

호출처:
- components/tab_portfolio_v2.py (또는 별도 탭): 🔄 교체 판단 섹션
"""
from __future__ import annotations

import logging

import pandas as pd
from nicegui import ui

# [v3.9.21] services 로직 re-export (backward compat)
from services.portfolio_swap import (
    analyze_portfolio_swap as _analyze_portfolio_swap,
    derive_holding_verdict as _derive_holding_verdict,
    HOLD_WEAK_FINAL_SCORE,
    HOLD_OVER_CONCENTRATION_PCT,
    NEW_STRONG_FINAL_SCORE,
)

_logger = logging.getLogger(__name__)


# verdict level → 카드 스타일 (배경 + 아이콘 색)
_LEVEL_STYLES = {
    "green": {
        "bg": "background: rgba(16, 185, 129, 0.10);",
        "border": "border-emerald-500/30",
    },
    "blue": {
        "bg": "background: rgba(59, 130, 246, 0.10);",
        "border": "border-blue-500/30",
    },
    "yellow": {
        "bg": "background: rgba(234, 179, 8, 0.10);",
        "border": "border-yellow-500/30",
    },
    "orange": {
        "bg": "background: rgba(249, 115, 22, 0.15);",
        "border": "border-orange-500/40",
    },
    "red": {
        "bg": "background: rgba(239, 68, 68, 0.15);",
        "border": "border-red-500/40",
    },
    "white": {
        "bg": "background: rgba(120, 120, 120, 0.10);",
        "border": "border-gray-500/30",
    },
}

_SUMMARY_LABELS = {
    "red": ("🔴", "정리 우선"),
    "orange": ("🟠", "교체 후보"),
    "yellow": ("🟡", "감량 검토"),
    "blue": ("🔵", "유지·매수금지"),
    "green": ("🟢", "유지"),
    "white": ("⚪", "데이터 부족"),
}


def _render_portfolio_swap_card(swap_data: dict) -> None:
    """[v3.9.21] 보유종목 vs 신규추천 교체 판단 카드."""
    if "error" in swap_data:
        with ui.card().classes(
            "w-full p-3 bg-gray-900/30 border border-gray-600/40 rounded-lg"
        ):
            ui.label(f"⚠️ {swap_data['error']}").classes(
                "text-amber-400 text-sm"
            )
        return

    summary = swap_data.get("summary", {})
    top_pick = swap_data.get("top_pick")
    new_safe = swap_data.get("new_recommend_safe", False)
    analysis = swap_data.get("holdings_analysis", [])

    # ─── 1. 상단 요약 (그룹별 카운트) ───
    ui.label("🔄 보유종목 교체 판단").classes(
        "text-base font-bold text-cyan-300 mb-2"
    )

    with ui.row().classes("w-full gap-2 flex-wrap mb-3"):
        # 정리 우선 / 교체 후보 / 감량 검토 / 유지 순으로 표시
        for level in ("red", "orange", "yellow", "blue", "green", "white"):
            n = summary.get(level, 0)
            if n == 0:
                continue
            icon, label = _SUMMARY_LABELS[level]
            style = _LEVEL_STYLES[level]
            with ui.card().classes(
                f"p-2 min-w-[110px] {style['border']} border rounded"
            ).style(style["bg"]):
                ui.label(f"{icon} {label}").classes(
                    "text-[10px] text-gray-300"
                )
                ui.label(f"{n}개").classes(
                    "text-base font-bold text-gray-100"
                )

    # ─── 2. 신규추천 (Top Pick) 표시 ───
    if top_pick is not None:
        new_safe_icon = "✅" if new_safe else "⚠️"
        new_safe_color = (
            "text-emerald-400" if new_safe else "text-amber-400"
        )
        new_safe_text = (
            "신규추천 안전 (교체 가능)" if new_safe
            else "신규추천 주의 (anomaly/과열 — 교체 제한)"
        )
        with ui.card().classes(
            "w-full p-2 mb-3 bg-cyan-900/15 border border-cyan-500/30 rounded"
        ):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("🎯 오늘 Top Pick:").classes(
                    "text-xs font-bold text-cyan-300"
                )
                ui.label(f"{top_pick.get('name', '?')}").classes(
                    "text-sm font-bold text-gray-100"
                )
                final = top_pick.get("final_score")
                if final is not None:
                    ui.label(f"FINAL {final:.0f}").classes(
                        "text-xs text-gray-300"
                    )
                route = top_pick.get("route")
                if route:
                    ui.label(f"ROUTE {route}").classes(
                        "text-xs text-gray-300"
                    )
            ui.label(f"{new_safe_icon} {new_safe_text}").classes(
                f"text-[11px] {new_safe_color} mt-1"
            )

    # ─── 3. 보유종목별 상세 표 ───
    if not analysis:
        ui.label("보유종목 분석 결과 없음").classes(
            "text-xs text-gray-500"
        )
        return

    ui.label("📋 보유종목별 판정").classes(
        "text-sm font-bold text-cyan-300 mb-2"
    )

    # 정렬: 위험도 높은 순 (red > orange > yellow > blue > green > white)
    level_order = {"red": 0, "orange": 1, "yellow": 2,
                   "blue": 3, "green": 4, "white": 5}
    sorted_analysis = sorted(
        analysis,
        key=lambda x: level_order.get(x["verdict"]["level"], 99),
    )

    # 보유종목 카드들
    for item in sorted_analysis:
        _render_holding_row(item)

    # ─── 4. 안내 ───
    # [v3.9.21c 평가 4] value_basis 안내 — 비중 계산 근거
    value_basis = swap_data.get("value_basis", "current_price")
    if value_basis == "mixed_current_avg":
        ui.label(
            "⚠️ 일부 보유종목이 오늘 추천에 없어 비중이 매입가 기준으로 "
            "포함되었습니다 (혼합 기준). 정확한 비중을 보려면 외부에서 "
            "total_value를 평가금액으로 직접 전달하세요."
        ).classes("text-[10px] text-amber-400 italic mb-1")

    ui.label(
        "💡 이 판단은 SwingPicker 신호 + 보유 비중/손익 기반 권장입니다. "
        "최종 매수/매도는 사용자가 직접 판단하시기 바랍니다. "
        "표현: 정리 우선 / 교체 후보 / 감량 검토 등은 모두 '검토 권장' 수준입니다."
    ).classes("text-[10px] text-gray-500 italic mt-3")


def _render_holding_row(item: dict) -> None:
    """단일 보유종목 카드."""
    verdict = item["verdict"]
    level = verdict["level"]
    style = _LEVEL_STYLES.get(level, _LEVEL_STYLES["white"])

    with ui.card().classes(
        f"w-full p-3 mb-2 {style['border']} border rounded"
    ).style(style["bg"]):
        # 헤더 줄: 종목명 / 판정 / 손익률 / 비중
        with ui.row().classes("w-full items-center gap-3 mb-1"):
            ui.label(verdict["icon"]).classes("text-xl")
            ui.label(item["name"]).classes(
                "text-sm font-bold text-gray-100"
            )
            ui.label(verdict["title"]).classes(
                f"text-xs font-bold {verdict['color_class']}"
            )

            # 손익률
            pnl = item.get("pnl_pct", 0)
            pnl_color = (
                "text-emerald-400" if pnl >= 0 else "text-red-400"
            )
            ui.label(f"손익 {pnl:+.1f}%").classes(
                f"text-xs {pnl_color}"
            )

            # 비중
            weight = item.get("weight_pct", 0)
            weight_color = (
                "text-amber-400" if weight >= HOLD_OVER_CONCENTRATION_PCT
                else "text-gray-300"
            )
            ui.label(f"비중 {weight:.0f}%").classes(
                f"text-xs {weight_color}"
            )

            # 점수 + ROUTE + EBS
            if item.get("final_score") is not None:
                final = item["final_score"]
                final_color = (
                    "text-emerald-400" if final >= 70
                    else "text-yellow-400" if final >= HOLD_WEAK_FINAL_SCORE
                    else "text-red-400"
                )
                ui.label(f"FINAL {final:.0f}").classes(
                    f"text-xs {final_color}"
                )

            if item.get("route"):
                ui.label(f"ROUTE {item['route']}").classes(
                    "text-xs text-gray-300"
                )
            if item.get("ebs") is not None:
                ebs = int(item["ebs"])
                ebs_color = (
                    "text-emerald-400" if ebs >= 1 else "text-red-400"
                )
                ui.label(f"EBS {ebs}").classes(f"text-xs {ebs_color}")

        # body — 권장 사유
        ui.label(verdict["body"]).classes(
            "text-xs text-gray-300 leading-relaxed"
        )

        # reasons 칩
        if verdict.get("reasons"):
            with ui.row().classes("w-full gap-1 mt-1 flex-wrap"):
                for reason in verdict["reasons"][:4]:
                    ui.label(f"· {reason}").classes(
                        "text-[10px] text-gray-400"
                    )

        # 교체 후보 표시
        if verdict.get("swap_candidate"):
            ui.label("→ 교체 가능 (오늘 Top Pick 대안)").classes(
                "text-[10px] text-orange-300 font-bold mt-1"
            )
