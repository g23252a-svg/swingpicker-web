"""
components/backtest_robustness.py
=================================
[v3.9.17b] 강건성 테스트 — UI 렌더 모듈.

평가 v3.9.17 4건 보정:
4. UI 비의존 분리:
   - 로직 (run_robustness_test / summarize_robustness / derive_robustness_verdict)
     → services/backtest_robustness.py로 이동
   - 이 모듈은 _render_robustness_table만 보유 (nicegui 의존)
   - 향후 CLI/배치 작업에서도 services 모듈만 import해서 사용 가능

호출처:
- components/tab_backtest.py: 🧱 강건성 버튼 4개 (프리셋별)에서 호출

Backward compat re-export (이전 v3.9.17 import 경로 보존):
- _run_robustness_test, _summarize_robustness, _derive_robustness_verdict
  은 services에서 re-export
- 상수 ROBUST_* / PARAM_DELTAS도 re-export
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from nicegui import ui

# [v3.9.17b] 로직 함수는 services에서 re-export (backward compat)
from services.backtest_robustness import (
    run_robustness_test as _run_robustness_test,
    summarize_robustness as _summarize_robustness,
    derive_robustness_verdict as _derive_robustness_verdict,
    ROBUST_GREEN_POSITIVE_RATIO,
    ROBUST_GREEN_ANOMALY_MAX,
    ROBUST_GREEN_MDD_RATIO,
    ROBUST_GREEN_ALPHA_RATIO,
    ROBUST_GREEN_ALPHA_COVERAGE,
    ROBUST_YELLOW_POSITIVE_RATIO,
    ROBUST_RED_ANOMALY_RATIO,
    PARAM_DELTAS,
)

_logger = logging.getLogger(__name__)


def _render_robustness_table(robustness_data: dict) -> None:
    """[v3.9.17] 강건성 결과 — 요약 + 27조합 표 (UI 전용)."""
    base_preset = robustness_data["base_preset"]
    base_cfg = robustness_data["base_cfg"]
    summary = robustness_data["summary"]
    verdict = robustness_data["verdict"]
    combos = robustness_data["combos"]

    # ─── 1. 판정 카드 ───
    with ui.card().classes(
        "w-full p-3 mb-3 bg-gray-900/30 border border-gray-600/40 rounded-lg"
    ):
        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.label(verdict["icon"]).classes("text-2xl")
            ui.label(
                f"강건성 판정: {verdict['title']}"
            ).classes(f"text-lg font-bold {verdict['color_class']}")
        ui.label(
            f"기준 프리셋: {base_preset} "
            f"(min_score={base_cfg['min_score']}, top_k={base_cfg['top_k']}, "
            f"hold_days={base_cfg['hold_days']})"
        ).classes("text-xs text-gray-400 mb-1")
        ui.label(verdict["body"]).classes(
            "text-sm text-gray-200 leading-relaxed"
        )

    # ─── 2. 요약 통계 카드 ───
    ui.label("📊 27조합 통계 요약").classes(
        "text-sm font-bold text-cyan-300 mb-2"
    )
    with ui.row().classes("w-full gap-2 flex-wrap mb-3"):
        _summary_pill(
            "성공",
            f"{summary['n_success']}/{summary['n_total']}",
            summary['n_success'] >= 20,
        )
        _summary_pill(
            "수익 양수",
            f"{summary['n_positive_ret']}/{summary['n_success']} "
            f"({summary['positive_ret_ratio']*100:.0f}%)",
            summary['positive_ret_ratio'] >= 0.7,
        )
        # [v3.9.17b] alpha 표시 — coverage + positive 동시
        acr = summary.get("alpha_coverage_ratio", 0)
        par = summary.get("positive_alpha_ratio", 0)
        if acr >= ROBUST_GREEN_ALPHA_COVERAGE:
            _summary_pill(
                "alpha 양수",
                f"{summary['n_positive_alpha']}/{summary['n_success']} "
                f"({par*100:.0f}%)",
                par >= ROBUST_GREEN_ALPHA_RATIO,
            )
        else:
            # coverage 부족 — 회색 표시
            _summary_pill(
                "alpha 양수",
                f"coverage {acr*100:.0f}% 부족",
                False,
            )
        _summary_pill(
            "anomaly",
            f"{summary['n_anomaly']} ({summary['anomaly_ratio']*100:.0f}%)",
            summary['anomaly_ratio'] <= 0.2,
            inverse=True,
        )
        _summary_pill(
            "MDD ≥ -15%",
            f"{summary['n_mdd_within_15']} ({summary['mdd_within_15_ratio']*100:.0f}%)",
            summary['mdd_within_15_ratio'] >= 0.5,
        )

    with ui.row().classes("w-full gap-2 flex-wrap mb-3"):
        _summary_pill(
            "최악 수익", f"{summary['worst_return']:+.1f}%",
            summary['worst_return'] >= -10,
        )
        _summary_pill(
            "평균 수익", f"{summary['avg_return']:+.1f}%",
            summary['avg_return'] > 0,
        )
        _summary_pill(
            "최고 수익", f"{summary['best_return']:+.1f}%",
            summary['best_return'] > 0,
        )

    # ─── 3. 27조합 상세 표 ───
    ui.label("📋 조합별 결과").classes(
        "text-sm font-bold text-cyan-300 mb-2"
    )
    ui.label(
        f"기준값: min_score={base_cfg['min_score']}, top_k={base_cfg['top_k']}, "
        f"hold_days={base_cfg['hold_days']}. "
        "각 파라미터 ±5 (3×3×3=27조합)."
    ).classes("text-xs text-gray-400 mb-2")

    grid_cols = (
        "minmax(140px, 1.2fr) minmax(100px, 1fr) minmax(80px, 0.8fr) "
        "minmax(80px, 0.8fr) minmax(100px, 1fr) minmax(80px, 0.8fr) "
        "minmax(100px, 1fr)"
    )
    headers = [
        ("조합 (Δ)", "left"),
        ("수익률", "right"),
        ("MDD", "right"),
        ("승률", "right"),
        ("alpha", "right"),
        ("TP포화", "right"),
        ("이슈", "left"),
    ]

    with ui.element("div").classes(
        "w-full overflow-x-auto border border-gray-700 rounded-lg"
    ).style("background: rgba(20, 20, 35, 0.6);"):
        with ui.element("div").classes("w-full").style(
            f"display: grid; grid-template-columns: {grid_cols}; "
            f"background: rgba(0, 100, 150, 0.25); "
            f"border-bottom: 1px solid #4b5563;"
        ):
            for name, align in headers:
                ui.label(name).classes(
                    f"p-2 font-bold text-cyan-200 text-{align} text-xs"
                ).style("white-space: nowrap;")

        sorted_combos = sorted(
            combos,
            key=lambda c: abs(c["delta_min_score"])
            + abs(c["delta_top_k"])
            + abs(c["delta_hold_days"]),
        )
        for combo in sorted_combos:
            r = combo["result"]
            is_base = (
                combo["delta_min_score"] == 0
                and combo["delta_top_k"] == 0
                and combo["delta_hold_days"] == 0
            )

            row_bg = ""
            if "error" not in r:
                if r.get("anomaly_flags"):
                    row_bg = "background: rgba(120, 70, 0, 0.18);"
                elif is_base:
                    row_bg = "background: rgba(0, 80, 100, 0.18);"

            with ui.element("div").classes("w-full").style(
                f"display: grid; grid-template-columns: {grid_cols}; "
                f"border-bottom: 1px solid #374151; {row_bg}"
            ):
                delta_label = (
                    f"Δ {combo['delta_min_score']:+d}/"
                    f"{combo['delta_top_k']:+d}/"
                    f"{combo['delta_hold_days']:+d}"
                )
                with ui.column().classes("p-2 gap-0"):
                    base_mark = "⭐ 기준 " if is_base else ""
                    ui.label(f"{base_mark}{delta_label}").classes(
                        "text-xs text-gray-200 font-mono"
                    )
                    ui.label(
                        f"score={combo['cfg']['min_score']} / "
                        f"top={combo['cfg']['top_k']} / "
                        f"hold={combo['cfg']['hold_days']}"
                    ).classes("text-[10px] text-gray-500")

                if "error" in r:
                    with ui.column().classes("p-2 gap-0").style(
                        "grid-column: span 6;"
                    ):
                        ui.label(f"❌ {r['error']}").classes(
                            "text-xs text-red-400"
                        )
                    continue

                total_ret = float(r.get("total_return", 0) or 0)
                mdd = float(r.get("mdd", 0) or 0)
                win_rate = float(r.get("win_rate", 0) or 0)
                alpha = r.get("alpha")
                alpha_mode = r.get("alpha_mode")
                tp_sat = float(r.get("tp_saturation", 0) or 0)
                tp_th = int(r.get("tp_threshold", 70) or 70)
                anom = r.get("anomaly_flags", [])

                from services.backtest_policy import (
                    ANOMALY_TOTAL_RET_ABS as _RA,
                )
                with ui.column().classes("p-2 gap-0 items-end"):
                    if total_ret > _RA:
                        ui.label(f"{_RA}%+ 비정상").classes(
                            "text-amber-400 text-xs"
                        )
                    else:
                        color = (
                            "text-emerald-400" if total_ret >= 0
                            else "text-red-400"
                        )
                        ui.label(f"{total_ret:+.1f}%").classes(
                            f"{color} text-sm font-bold"
                        )

                with ui.column().classes("p-2 gap-0 items-end"):
                    color = (
                        "text-emerald-400" if mdd >= -10
                        else "text-amber-400" if mdd >= -20
                        else "text-red-400"
                    )
                    ui.label(f"{mdd:.1f}%").classes(f"{color} text-sm")

                with ui.column().classes("p-2 gap-0 items-end"):
                    ui.label(f"{win_rate:.0f}%").classes(
                        "text-gray-300 text-sm"
                    )

                with ui.column().classes("p-2 gap-0 items-end"):
                    if alpha is None:
                        ui.label("⚪ —").classes(
                            "text-gray-500 text-xs"
                        )
                    else:
                        color = (
                            "text-emerald-400" if alpha > 0
                            else "text-red-400"
                        )
                        mode_mark = "" if alpha_mode == "real" else "*"
                        ui.label(f"{alpha:+.1f}{mode_mark}").classes(
                            f"{color} text-sm"
                        )

                with ui.column().classes("p-2 gap-0 items-end"):
                    warn = tp_sat >= tp_th
                    color = "text-amber-400" if warn else "text-gray-300"
                    prefix = "⚠️" if warn else ""
                    ui.label(f"{prefix}{tp_sat:.0f}%").classes(
                        f"{color} text-xs"
                    )

                with ui.column().classes("p-2 gap-0"):
                    if anom:
                        ui.label(f"🚨 {anom[0]}").classes(
                            "text-[10px] text-amber-300"
                        )
                    elif total_ret < 0:
                        ui.label("🔴 손실").classes(
                            "text-[10px] text-red-400"
                        )
                    elif alpha is not None and alpha < 0:
                        ui.label("🔴 시장 열위").classes(
                            "text-[10px] text-red-400"
                        )
                    else:
                        ui.label("—").classes(
                            "text-[10px] text-gray-500"
                        )

    # ─── 4. 범례 ───
    ui.label(
        "💡 * 표시 = 간이 알파. ⭐ 기준 = 베이스 프리셋. "
        "anomaly 행은 amber 배경."
    ).classes("text-[10px] text-gray-500 mt-2 italic")
    ui.label(
        "📊 강건성 판정 기준: 🟢 강건함 (수익 양수 80%+ AND anomaly 20% 이하 "
        "AND MDD 이내 60%+ AND alpha 양수 50%+ 또는 coverage 부족) · 🟡 조건부 · "
        "🔴 과최적화 의심 (수익 양수 50% 미만 OR anomaly 50% 초과)"
    ).classes("text-[10px] text-gray-500 italic")


def _summary_pill(
    label: str, value: str, is_good: bool, inverse: bool = False
):
    """요약 통계용 작은 카드 (gray/emerald/amber)."""
    bg = (
        "bg-emerald-900/20 border-emerald-500/40"
        if is_good
        else "bg-amber-900/20 border-amber-500/40"
    )
    if inverse and is_good:
        pass
    with ui.card().classes(
        f"p-2 min-w-[120px] {bg} border rounded"
    ):
        ui.label(label).classes("text-[10px] text-gray-300")
        color = "text-emerald-300" if is_good else "text-amber-300"
        ui.label(value).classes(f"text-sm font-bold {color}")
