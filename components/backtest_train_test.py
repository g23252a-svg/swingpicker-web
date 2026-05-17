"""
components/backtest_train_test.py
==================================
[v3.9.18] Train/Test 분할 검증 — UI 렌더 모듈.

로직(run_train_test_split, derive_train_test_verdict)은 services로 분리.
이 모듈은 _render_train_test_result만 보유.

Backward compat re-export:
- _run_train_test_split, _derive_train_test_verdict은 services에서 re-export
- 외부 모듈/테스트가 components에서 import해도 동작

호출처:
- components/tab_backtest.py: 🔬 Train/Test 분할 버튼에서 호출
"""
from __future__ import annotations

import logging

import pandas as pd
from nicegui import ui

# [v3.9.18] services 로직 re-export (backward compat)
from services.backtest_train_test import (
    run_train_test_split as _run_train_test_split,
    derive_train_test_verdict as _derive_train_test_verdict,
    DEFAULT_TEST_RATIO,
    MIN_RECS_PER_SPLIT,
)

_logger = logging.getLogger(__name__)


def _render_train_test_result(tt_data: dict) -> None:
    """[v3.9.18] Train/Test 분할 결과 — 판정 카드 + Train/Test 나란히 비교."""
    preset = tt_data["preset"]
    cfg = tt_data["cfg"]
    split_info = tt_data["split_info"]
    train_result = tt_data["train_result"]
    test_result = tt_data["test_result"]
    verdict = tt_data["verdict"]

    # ─── 1. 판정 카드 ───
    with ui.card().classes(
        "w-full p-3 mb-3 bg-gray-900/30 border border-gray-600/40 rounded-lg"
    ):
        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.label(verdict["icon"]).classes("text-2xl")
            ui.label(
                f"Train/Test 판정: {verdict['title']}"
            ).classes(f"text-lg font-bold {verdict['color_class']}")

        # 분할 정보 한 줄
        if "train_date_range" in split_info:
            train_range = split_info["train_date_range"]
            test_range = split_info["test_date_range"]
            # [v3.9.18b] 날짜 수 + row 수 둘 다 표시
            n_train_dates = split_info.get("n_train_dates", "?")
            n_test_dates = split_info.get("n_test_dates", "?")
            n_unique_dates = split_info.get("n_unique_dates", "?")
            ui.label(
                f"프리셋: {preset} · 총 {split_info['n_total']}건 / "
                f"{n_unique_dates}일 → "
                f"Train {split_info['n_train']}건 / {n_train_dates}일 "
                f"({train_range[0]}~{train_range[1]}) · "
                f"Test {split_info['n_test']}건 / {n_test_dates}일 "
                f"({test_range[0]}~{test_range[1]})"
            ).classes("text-xs text-gray-400 mb-1")
        else:
            ui.label(
                f"프리셋: {preset} · 데이터 분할 실패"
            ).classes("text-xs text-amber-400 mb-1")

        ui.label(verdict["body"]).classes(
            "text-sm text-gray-200 leading-relaxed"
        )

    # 데이터 부족이면 여기서 종료
    if "error" in train_result or "error" in test_result:
        return

    # ─── 2. Train vs Test 나란히 표시 ───
    ui.label("📊 Train vs Test 비교").classes(
        "text-sm font-bold text-cyan-300 mb-2"
    )

    grid_cols = "1fr 1fr 1fr"  # 지표 / Train / Test
    with ui.element("div").classes(
        "w-full overflow-x-auto border border-gray-700 rounded-lg"
    ).style("background: rgba(20, 20, 35, 0.6);"):
        # 헤더
        with ui.element("div").classes("w-full").style(
            f"display: grid; grid-template-columns: {grid_cols}; "
            f"background: rgba(0, 100, 150, 0.25); "
            f"border-bottom: 1px solid #4b5563;"
        ):
            for name in ["지표", "🟦 Train (과거 70%)", "🟧 Test (최근 30%)"]:
                ui.label(name).classes(
                    "p-2 font-bold text-cyan-200 text-center text-xs"
                ).style("white-space: nowrap;")

        # 각 지표별 행
        _render_metric_row(
            "수익률", train_result, test_result, "total_return",
            fmt="{:+.2f}%", is_higher_better=True,
        )
        _render_metric_row(
            "MDD", train_result, test_result, "mdd",
            fmt="{:.2f}%", is_higher_better=True,  # 덜 음수일수록 좋음
        )
        _render_metric_row(
            "승률", train_result, test_result, "win_rate",
            fmt="{:.1f}%", is_higher_better=True,
        )
        _render_metric_row(
            "Sharpe", train_result, test_result, "sharpe",
            fmt="{:.2f}", is_higher_better=True,
        )
        _render_metric_row(
            "alpha", train_result, test_result, "alpha",
            fmt="{:+.2f}%p", is_higher_better=True,
            extra_field="alpha_mode",
        )
        _render_metric_row(
            "거래 수", train_result, test_result, "total_trades",
            fmt="{:d}", is_higher_better=False,  # 표본 양호하면 중립
        )
        _render_anomaly_row(train_result, test_result, grid_cols)

    # ─── 3. 성능 유지율 카드 ───
    train_ret = float(train_result.get("total_return", 0) or 0)
    test_ret = float(test_result.get("total_return", 0) or 0)
    if train_ret > 0:
        retention = test_ret / train_ret
        if retention >= 0.80:
            color = "text-emerald-400"
            label = "성능 유지율 양호"
        elif retention >= 0.40:
            color = "text-yellow-400"
            label = "성능 유지율 보통"
        else:
            color = "text-red-400"
            label = "성능 유지율 낮음"
        with ui.card().classes(
            "w-full p-3 mt-3 bg-gray-900/30 border border-gray-600/40 rounded-lg"
        ):
            ui.label(f"📈 {label}").classes(
                f"text-sm font-bold {color}"
            )
            ui.label(
                f"Test/Train 비율: {retention*100:.0f}% "
                f"({test_ret:+.1f}% / {train_ret:+.1f}%) — "
                f"100%면 Test가 Train과 동일, 0%면 Test에서 완전 붕괴."
            ).classes("text-xs text-gray-300 mt-1")

    # ─── 4. 범례 ───
    ui.label(
        "💡 Train = 과거 70%, Test = 최근 30% (rec_date 기준). "
        "🟢 일반화 양호: Test도 양호 · 🟡 약화: Test 일부 미달 · "
        "🔴 과최적화: Train만 좋음 · 🚨 lookahead: 비현실 패턴."
    ).classes("text-[10px] text-gray-500 mt-2 italic")


def _render_metric_row(
    label: str,
    train: dict,
    test: dict,
    field: str,
    fmt: str,
    is_higher_better: bool = True,
    extra_field: str = None,
):
    """단일 지표 행 — Train vs Test."""
    grid_cols = "1fr 1fr 1fr"
    train_val = train.get(field)
    test_val = test.get(field)

    def _fmt_value(v):
        if v is None:
            return "—"
        try:
            if fmt.endswith("d}"):
                return fmt.format(int(v))
            return fmt.format(float(v))
        except (TypeError, ValueError):
            return str(v)

    train_str = _fmt_value(train_val)
    test_str = _fmt_value(test_val)

    # extra_field (예: alpha_mode "real"/"simple")
    if extra_field and test.get(extra_field) is not None:
        mode = test.get(extra_field)
        if mode == "simple":
            test_str += " *"

    # 색상 — 0 기준 단순 비교
    def _color(v, threshold=0):
        if v is None:
            return "text-gray-400"
        try:
            v = float(v)
            if is_higher_better:
                return "text-emerald-400" if v >= threshold else "text-red-400"
            return "text-gray-300"
        except (TypeError, ValueError):
            return "text-gray-400"

    with ui.element("div").classes("w-full").style(
        f"display: grid; grid-template-columns: {grid_cols}; "
        f"border-bottom: 1px solid #374151;"
    ):
        ui.label(label).classes(
            "p-2 text-xs text-gray-300 font-bold"
        )
        ui.label(train_str).classes(
            f"p-2 text-sm text-center {_color(train_val)} font-bold"
        )
        ui.label(test_str).classes(
            f"p-2 text-sm text-center {_color(test_val)} font-bold"
        )


def _render_anomaly_row(train: dict, test: dict, grid_cols: str):
    """anomaly 행 — flag 목록 표시."""
    train_anom = train.get("anomaly_flags", [])
    test_anom = test.get("anomaly_flags", [])

    def _fmt_anom(flags):
        if not flags:
            return ("없음", "text-emerald-400")
        return (f"🚨 {flags[0]}", "text-amber-400")

    train_str, train_color = _fmt_anom(train_anom)
    test_str, test_color = _fmt_anom(test_anom)

    with ui.element("div").classes("w-full").style(
        f"display: grid; grid-template-columns: {grid_cols}; "
        f"border-bottom: 1px solid #374151;"
    ):
        ui.label("anomaly").classes(
            "p-2 text-xs text-gray-300 font-bold"
        )
        ui.label(train_str).classes(
            f"p-2 text-xs text-center {train_color}"
        )
        ui.label(test_str).classes(
            f"p-2 text-xs text-center {test_color}"
        )
