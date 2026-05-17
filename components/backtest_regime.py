"""
components/backtest_regime.py
==============================
[v3.9.19] 시장 국면별 성과 — UI 렌더 모듈.

로직(run_regime_split, derive_regime_verdict)은 services로 분리.
이 모듈은 _render_regime_table만 보유.

Backward compat re-export:
- _run_regime_split, _derive_regime_verdict, load_macro_regime_map는 services에서
  re-export. 외부 모듈/테스트가 components에서 import해도 동작.

호출처:
- components/tab_backtest.py: 🌡️ 시장 국면별 성과 버튼에서 호출
"""
from __future__ import annotations

import logging

import pandas as pd
from nicegui import ui

# [v3.9.19] services 로직 re-export (backward compat)
from services.backtest_regime import (
    run_regime_split as _run_regime_split,
    derive_regime_verdict as _derive_regime_verdict,
    load_macro_regime_map as _load_macro_regime_map,
    REGIMES,
    MIN_TRADES_PER_REGIME,
    DEFAULT_RUN_HEALTH_DIR,
)

_logger = logging.getLogger(__name__)


# 국면별 색상 배경 (NORMAL=emerald, CAUTION=amber, CRITICAL=red)
_REGIME_STYLES = {
    "NORMAL": {
        "label": "🟢 NORMAL (활황)",
        "bg": "background: rgba(16, 185, 129, 0.10);",
        "border_color": "border-emerald-500/30",
        "text_color": "text-emerald-300",
    },
    "CAUTION": {
        "label": "🟡 CAUTION (주의)",
        "bg": "background: rgba(245, 158, 11, 0.10);",
        "border_color": "border-amber-500/30",
        "text_color": "text-amber-300",
    },
    "CRITICAL": {
        "label": "🔴 CRITICAL (위험)",
        "bg": "background: rgba(239, 68, 68, 0.10);",
        "border_color": "border-red-500/30",
        "text_color": "text-red-300",
    },
}


def _render_regime_table(regime_data: dict) -> None:
    """[v3.9.19] 시장 국면별 성과 — 판정 카드 + 3국면 비교 표."""
    preset = regime_data["preset"]
    cfg = regime_data["cfg"]
    info = regime_data["regime_map_info"]
    regimes = regime_data["regimes"]
    verdict = regime_data["verdict"]

    # ─── 1. 판정 카드 ───
    with ui.card().classes(
        "w-full p-3 mb-3 bg-gray-900/30 border border-gray-600/40 rounded-lg"
    ):
        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.label(verdict["icon"]).classes("text-2xl")
            ui.label(
                f"국면별 판정: {verdict['title']}"
            ).classes(f"text-lg font-bold {verdict['color_class']}")

        # 국면 데이터 정보
        n_dates = info.get("n_dates_with_regime", 0)
        dist = info.get("regime_dist", {})
        if n_dates > 0:
            dist_str = " · ".join(
                f"{r} {dist.get(r, 0)}일" for r in REGIMES
            )
            ui.label(
                f"프리셋: {preset} · 국면 매칭 {n_dates}일 ({dist_str})"
            ).classes("text-xs text-gray-400 mb-1")
        else:
            ui.label(
                f"프리셋: {preset} · 국면 데이터 없음"
            ).classes("text-xs text-amber-400 mb-1")

        ui.label(verdict["body"]).classes(
            "text-sm text-gray-200 leading-relaxed"
        )

    # 국면 데이터 자체가 없으면 여기서 종료
    if n_dates == 0:
        return

    # ─── 2. 3국면 카드 (각 국면 요약) ───
    ui.label("📊 국면별 요약").classes(
        "text-sm font-bold text-cyan-300 mb-2"
    )
    with ui.row().classes("w-full gap-2 flex-wrap mb-3"):
        for regime in REGIMES:
            data = regimes.get(regime, {"n_recs": 0, "result": {}})
            style = _REGIME_STYLES[regime]
            with ui.card().classes(
                f"flex-1 min-w-[200px] p-2 {style['border_color']} border rounded"
            ).style(style["bg"]):
                ui.label(style["label"]).classes(
                    f"text-xs font-bold {style['text_color']} mb-1"
                )
                ui.label(f"표본: {data['n_recs']}건").classes(
                    "text-[10px] text-gray-400"
                )

                result = data.get("result", {})
                if "error" in result:
                    ui.label(f"❌ {result['error']}").classes(
                        "text-xs text-red-400"
                    )
                    continue

                ret = float(result.get("total_return", 0) or 0)
                mdd = float(result.get("mdd", 0) or 0)
                ret_color = (
                    "text-emerald-300" if ret >= 0 else "text-red-400"
                )
                ui.label(f"수익: {ret:+.1f}%").classes(
                    f"text-sm font-bold {ret_color}"
                )
                ui.label(f"MDD: {mdd:.1f}%").classes(
                    "text-xs text-gray-300"
                )

    # ─── 3. 상세 비교 표 ───
    ui.label("📋 국면별 상세 지표").classes(
        "text-sm font-bold text-cyan-300 mb-2"
    )

    grid_cols = (
        "minmax(120px, 1fr) minmax(60px, 0.6fr) minmax(80px, 0.8fr) "
        "minmax(80px, 0.8fr) minmax(70px, 0.7fr) minmax(70px, 0.7fr) "
        "minmax(80px, 0.8fr) minmax(70px, 0.7fr) minmax(70px, 0.7fr) "
        "minmax(80px, 0.8fr)"
    )
    headers = [
        ("국면", "left"),
        ("거래수", "right"),
        ("수익률", "right"),
        ("MDD", "right"),
        ("승률", "right"),
        ("Sharpe", "right"),
        ("alpha", "right"),
        ("STOP율", "right"),
        ("TP포화", "right"),
        ("이슈", "left"),
    ]

    with ui.element("div").classes(
        "w-full overflow-x-auto border border-gray-700 rounded-lg"
    ).style("background: rgba(20, 20, 35, 0.6);"):
        # 헤더
        with ui.element("div").classes("w-full").style(
            f"display: grid; grid-template-columns: {grid_cols}; "
            f"background: rgba(0, 100, 150, 0.25); "
            f"border-bottom: 1px solid #4b5563;"
        ):
            for name, align in headers:
                ui.label(name).classes(
                    f"p-2 font-bold text-cyan-200 text-{align} text-xs"
                ).style("white-space: nowrap;")

        # 3국면 행
        for regime in REGIMES:
            data = regimes.get(regime, {"n_recs": 0, "result": {}})
            style = _REGIME_STYLES[regime]
            result = data.get("result", {})

            with ui.element("div").classes("w-full").style(
                f"display: grid; grid-template-columns: {grid_cols}; "
                f"border-bottom: 1px solid #374151; {style['bg']}"
            ):
                # 1. 국면
                with ui.column().classes("p-2 gap-0"):
                    ui.label(style["label"]).classes(
                        f"text-xs {style['text_color']} font-bold"
                    )

                # 2. 거래수
                with ui.column().classes("p-2 gap-0 items-end"):
                    ui.label(f"{data['n_recs']}").classes(
                        "text-xs text-gray-300"
                    )

                if "error" in result:
                    # error → 나머지 8칸 합쳐서 메시지
                    with ui.column().classes("p-2 gap-0").style(
                        "grid-column: span 8;"
                    ):
                        ui.label(f"⚪ {result['error']}").classes(
                            "text-xs text-gray-500"
                        )
                    continue

                # 정상 결과 — 8칸 채움
                total_ret = float(result.get("total_return", 0) or 0)
                mdd = float(result.get("mdd", 0) or 0)
                win_rate = float(result.get("win_rate", 0) or 0)
                sharpe = float(result.get("sharpe", 0) or 0)
                alpha = result.get("alpha")
                alpha_mode = result.get("alpha_mode")
                stop_ratio = float(result.get("stop_ratio", 0) or 0)
                tp_sat = float(result.get("tp_saturation", 0) or 0)
                tp_th = int(result.get("tp_threshold", 70) or 70)
                anom = result.get("anomaly_flags", [])

                # 3. 수익률
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

                # 4. MDD
                with ui.column().classes("p-2 gap-0 items-end"):
                    mdd_color = (
                        "text-emerald-400" if mdd >= -10
                        else "text-amber-400" if mdd >= -20
                        else "text-red-400"
                    )
                    ui.label(f"{mdd:.1f}%").classes(
                        f"{mdd_color} text-sm"
                    )

                # 5. 승률
                with ui.column().classes("p-2 gap-0 items-end"):
                    ui.label(f"{win_rate:.0f}%").classes(
                        "text-gray-300 text-sm"
                    )

                # 6. Sharpe
                with ui.column().classes("p-2 gap-0 items-end"):
                    ui.label(f"{sharpe:.2f}").classes(
                        "text-gray-300 text-sm"
                    )

                # 7. alpha
                with ui.column().classes("p-2 gap-0 items-end"):
                    if alpha is None:
                        ui.label("⚪ —").classes(
                            "text-gray-500 text-xs"
                        )
                    else:
                        alpha_color = (
                            "text-emerald-400" if alpha > 0
                            else "text-red-400"
                        )
                        mode_mark = "" if alpha_mode == "real" else "*"
                        ui.label(f"{alpha:+.2f}{mode_mark}").classes(
                            f"{alpha_color} text-sm"
                        )

                # 8. STOP율
                with ui.column().classes("p-2 gap-0 items-end"):
                    stop_color = (
                        "text-emerald-400" if stop_ratio <= 15
                        else "text-amber-400" if stop_ratio <= 30
                        else "text-red-400"
                    )
                    ui.label(f"{stop_ratio:.0f}%").classes(
                        f"{stop_color} text-xs"
                    )

                # 9. TP포화
                with ui.column().classes("p-2 gap-0 items-end"):
                    warn = tp_sat >= tp_th
                    tp_color = (
                        "text-amber-400" if warn else "text-gray-300"
                    )
                    prefix = "⚠️" if warn else ""
                    ui.label(f"{prefix}{tp_sat:.0f}%").classes(
                        f"{tp_color} text-xs"
                    )

                # 10. 이슈
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
        "💡 국면 분류: NORMAL(활황) · CAUTION(주의) · CRITICAL(위험)는 "
        "run_health JSON의 macro_risk 기준. * 표시 = 간이 알파. "
        f"각 국면 최소 {MIN_TRADES_PER_REGIME}건 필요."
    ).classes("text-[10px] text-gray-500 mt-2 italic")
    ui.label(
        "📊 판정 기준: 🟢 전천후 (3국면 모두 양호) · 🟡 국면 의존 "
        "(일부 약함) · 🔴 하락장 취약 (CAUTION/CRITICAL 큰 손실) · "
        "⚪ 표본 부족"
    ).classes("text-[10px] text-gray-500 italic")
