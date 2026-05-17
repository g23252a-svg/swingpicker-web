"""
components/backtest_preset_compare.py
=====================================
[v3.9.17] 4프리셋 비교 — 동시 실행 + 비교표 + 하이라이트.

평가 v3.9.16 후속 분리. tab_backtest.py 파일 크기 절감 (2441줄 → ~1800줄).

이 모듈은 4개 프리셋(보수/균형/공격/단타)을 한 번에 실행하고 비교표로
렌더한다. 판정은 services 공유 SSOT인 backtest_verdict._derive_strategy_verdict
함수를 사용해서 단일 카드와 절대 갈라지지 않는다.

의존성:
- services.backtest_policy (anomaly 검출 + TP saturation tier)
- components.backtest_verdict (판정 SSOT)
- components.tab_backtest (PRESETS dict, _run_backtest, _calc_kospi_alpha)
  ← 모듈 top-level이 아닌 함수 안에서 lazy import (순환 방지)
"""
from __future__ import annotations

import logging

import pandas as pd
from nicegui import ui

_logger = logging.getLogger(__name__)


def _run_preset_comparison(all_recs: pd.DataFrame) -> dict:
    """[v3.9.16] 4개 프리셋 동시 실행 → 비교용 dict 반환.

    [v3.9.17] components.tab_backtest의 PRESETS / _run_backtest / _calc_kospi_alpha를
    lazy import로 사용 — 순환 import 방지.

    Args:
        all_recs: 누적 추천 CSV 데이터프레임

    Returns:
        {
          "conservative": {
              "label": "🛡️ 보수형",
              "cfg": {min_score, top_k, hold_days, target_pct, stop_pct, cost_pct},
              "result": {total_return, win_rate, mdd, ..., alpha, alpha_mode,
                         anomaly_flags, tp_saturation, tp_threshold} 또는 {"error":...}
          },
          ...
        }
    """
    # [v3.9.17] lazy import — tab_backtest ↔ backtest_preset_compare 순환 방지
    from components.tab_backtest import PRESETS, _run_backtest
    from components.backtest_verdict import _calc_kospi_alpha
    from services.backtest_policy import (
        detect_anomaly_flags,
        tp_saturation_threshold,
    )

    out = {}
    for key, preset in PRESETS.items():
        cfg = {
            "min_score": preset["min_score"],
            "top_k": preset["top_k"],
            "hold_days": preset["hold_days"],
            "target_pct": preset["target_pct"],
            "stop_pct": preset["stop_pct"],
            "cost_pct": preset["cost_pct"],
        }
        try:
            result = _run_backtest(
                all_recs,
                cfg["min_score"], cfg["hold_days"],
                cfg["stop_pct"], cfg["target_pct"],
                cfg["top_k"], cfg["cost_pct"],
            )
        except Exception as e:
            _logger.warning(f"[preset_comparison] {key} 백테스트 실패: {e}")
            result = {"error": f"백테스트 실행 실패: {e}"}

        if "error" not in result:
            # alpha (real or simple)
            try:
                alpha, alpha_mode = _calc_kospi_alpha(result, cfg)
            except Exception as e:
                _logger.debug(f"[preset_comparison] {key} alpha 계산 실패: {e}")
                alpha, alpha_mode = None, None
            result["alpha"] = alpha
            result["alpha_mode"] = alpha_mode

            # anomaly flags (services SSOT 사용)
            result["anomaly_flags"] = detect_anomaly_flags(
                total_ret=float(result.get("total_return", 0) or 0),
                sharpe_val=result.get("sharpe"),
                cagr_val=result.get("cagr"),
                trading_days=int(result.get("trading_days", 0) or 0),
            )

            # TP 포화율 + tier 임계
            sd = result.get("status_dist", {}) or {}
            n_win = int(sd.get("WIN", 0) or 0)
            n_stop = int(sd.get("STOP", 0) or 0)
            n_hold = int(sd.get("HOLD_EXIT", 0) or 0)
            n_total = n_win + n_stop + n_hold
            result["tp_saturation"] = (
                (n_win / n_total * 100) if n_total > 0 else 0.0
            )
            result["tp_threshold"] = tp_saturation_threshold(cfg["target_pct"])

        out[key] = {
            "label": preset["label"],
            "cfg": cfg,
            "result": result,
        }
    return out


def _render_preset_comparison_table(results_by_preset: dict) -> None:
    """[v3.9.16] 프리셋 비교 표 + 최고/최저 하이라이트.

    하이라이트 규칙 (각 컬럼별 베스트):
        🔥 수익률 최고
        🛡️ MDD 최저 (-3% > -10%)
        ⚡ Sharpe 최고
        📈 alpha 최고

    경고:
        🚨 anomaly 있음 → 행 전체 amber 배경
        ⚠️ TP 포화율 임계 초과
        ⚪ alpha 없음 (간이 알파 미산출) → 회색 표시
    """
    # 유효 결과만 추출
    valid = {
        k: v for k, v in results_by_preset.items() if "error" not in v["result"]
    }
    if not valid:
        ui.label("⚠️ 모든 프리셋이 백테스트 실패").classes(
            "text-red-400 p-3"
        )
        for k, v in results_by_preset.items():
            err = v["result"].get("error", "알 수 없는 오류")
            ui.label(f"  · {v['label']}: {err}").classes(
                "text-xs text-red-300 ml-3"
            )
        return

    # 하이라이트 베스트 결정
    def _max_by(key, prefer_lower=False, allow_none=False):
        """key 기준 최고 (또는 최저) 프리셋 키 반환."""
        scored = []
        for k, v in valid.items():
            val = v["result"].get(key)
            if val is None:
                if allow_none:
                    continue
                continue
            try:
                scored.append((k, float(val)))
            except (TypeError, ValueError):
                continue
        if not scored:
            return None
        if prefer_lower:
            return min(scored, key=lambda x: x[1])[0]
        return max(scored, key=lambda x: x[1])[0]

    best_ret = _max_by("total_return")
    # MDD는 음수 — 절대값 작은(0에 가까운) 게 좋음 → max 사용 (-3 > -10)
    best_mdd = _max_by("mdd")
    best_sharpe = _max_by("sharpe", allow_none=True)
    best_alpha = _max_by("alpha", allow_none=True)

    # ─── 비교표 헤더 ───
    ui.label("📊 프리셋 비교 (4종 동시 실행)").classes(
        "text-lg font-bold text-cyan-300 mb-2"
    )
    ui.label(
        "각 컬럼 베스트: 🔥수익률 · 🛡️MDD · ⚡Sharpe · 📈alpha. "
        "🚨 anomaly 있음 / ⚠️ TP 포화율 임계 초과."
    ).classes("text-xs text-gray-400 mb-3")

    # ─── 표 본체 (NiceGUI grid 기반 — column flex로 안전 렌더) ───
    headers = [
        ("프리셋", "left"),
        ("조건", "left"),
        ("수익률", "right"),
        ("MDD", "right"),
        ("승률", "right"),
        ("Sharpe", "right"),
        ("alpha", "right"),
        ("TP포화", "right"),
        ("판정", "left"),
    ]
    # grid-template-columns: 9 컬럼 자동 폭
    grid_cols = (
        "minmax(140px, 1.2fr) minmax(180px, 1.5fr) "
        "minmax(100px, 1fr) minmax(80px, 0.8fr) minmax(70px, 0.7fr) "
        "minmax(80px, 0.8fr) minmax(100px, 1fr) minmax(80px, 0.8fr) "
        "minmax(180px, 1.5fr)"
    )

    with ui.element("div").classes(
        "w-full overflow-x-auto border border-gray-700 rounded-lg"
    ).style(f"background: rgba(20, 20, 35, 0.6);"):
        # 헤더 행
        with ui.element("div").classes("w-full").style(
            f"display: grid; grid-template-columns: {grid_cols}; "
            f"background: rgba(0, 100, 150, 0.25); "
            f"border-bottom: 1px solid #4b5563;"
        ):
            for name, align in headers:
                ui.label(name).classes(
                    f"p-2 font-bold text-cyan-200 text-{align} text-xs"
                ).style("white-space: nowrap;")

        # 본체 — 프리셋별 행
        preset_order = ["conservative", "balanced", "aggressive", "scalping"]
        for key in preset_order:
            if key not in results_by_preset:
                continue
            v = results_by_preset[key]
            r = v["result"]

            if "error" in r:
                # 에러 행 — 단순 한 줄
                with ui.element("div").classes("w-full p-2").style(
                    "background: rgba(100, 30, 30, 0.25); "
                    "border-bottom: 1px solid #374151;"
                ):
                    ui.label(
                        f"❌ {v['label']}: {r['error']}"
                    ).classes("text-xs text-red-300")
                continue

            # anomaly 시 행 배경 amber
            row_bg = (
                "background: rgba(120, 70, 0, 0.18);"
                if r.get("anomaly_flags")
                else ""
            )

            total_ret = float(r.get("total_return", 0) or 0)
            mdd = float(r.get("mdd", 0) or 0)
            win_rate = float(r.get("win_rate", 0) or 0)
            sharpe = r.get("sharpe")
            alpha = r.get("alpha")
            alpha_mode = r.get("alpha_mode")
            tp_sat = float(r.get("tp_saturation", 0) or 0)
            tp_th = int(r.get("tp_threshold", 70) or 70)
            anom = r.get("anomaly_flags", [])
            cfg = v["cfg"]

            with ui.element("div").classes("w-full").style(
                f"display: grid; grid-template-columns: {grid_cols}; "
                f"border-bottom: 1px solid #374151; {row_bg}"
            ):
                # 1. 프리셋 이름
                with ui.column().classes("p-2 gap-0"):
                    ui.label(v["label"]).classes(
                        "font-bold text-gray-100 text-sm"
                    )
                    if anom:
                        ui.label("🚨 anomaly").classes(
                            "text-[10px] text-amber-300"
                        )

                # 2. 조건 요약
                with ui.column().classes("p-2 gap-0"):
                    ui.label(
                        f"{cfg['min_score']}점↑ / Top-{cfg['top_k']} / "
                        f"보유 {cfg['hold_days']}일"
                    ).classes("text-[11px] text-gray-300")
                    ui.label(
                        f"+{cfg['target_pct']:.0f}/{cfg['stop_pct']:.0f}% / "
                        f"비용 {cfg['cost_pct']:.1f}%"
                    ).classes("text-[10px] text-gray-400")

                # 3. 수익률 (anomaly 시 raw 노출 차단)
                from services.backtest_policy import (
                    ANOMALY_TOTAL_RET_ABS as _RA,
                    ANOMALY_SHARPE_MAX as _SM,
                )
                with ui.column().classes("p-2 gap-0 items-end"):
                    if total_ret > _RA:
                        disp = f"{_RA}%+ 비정상"
                        color = "text-amber-400"
                    else:
                        disp = f"{total_ret:+.1f}%"
                        color = (
                            "text-emerald-400" if total_ret >= 0
                            else "text-red-400"
                        )
                    # [v3.9.16b 보정] anomaly 행이면 🔥 대신 🚨 — 사용자 혼란 방지
                    # 최고 수익률 정보 자체는 정보로 가치 있어서 베스트에서 빼지 않음.
                    # 다만 🔥(자랑)과 🚨(경고)를 겹치지 않게 후자만 표시.
                    if key == best_ret and total_ret > 0:
                        prefix = "🚨 " if anom else "🔥 "
                    else:
                        prefix = ""
                    ui.label(f"{prefix}{disp}").classes(
                        f"{color} font-bold text-sm"
                    )

                # 4. MDD
                with ui.column().classes("p-2 gap-0 items-end"):
                    prefix = "🛡️ " if key == best_mdd else ""
                    color = (
                        "text-emerald-400" if mdd >= -10
                        else "text-amber-400" if mdd >= -20
                        else "text-red-400"
                    )
                    ui.label(f"{prefix}{mdd:.1f}%").classes(
                        f"{color} font-bold text-sm"
                    )

                # 5. 승률
                with ui.column().classes("p-2 gap-0 items-end"):
                    color = (
                        "text-emerald-400" if win_rate >= 55
                        else "text-gray-300" if win_rate >= 45
                        else "text-amber-400"
                    )
                    ui.label(f"{win_rate:.1f}%").classes(
                        f"{color} text-sm"
                    )

                # 6. Sharpe
                with ui.column().classes("p-2 gap-0 items-end"):
                    if sharpe is None:
                        ui.label("—").classes("text-gray-500 text-sm")
                    elif sharpe > _SM:
                        ui.label(f"{_SM}+ 비정상").classes(
                            "text-amber-400 text-xs"
                        )
                    else:
                        prefix = "⚡ " if key == best_sharpe else ""
                        color = (
                            "text-emerald-400" if sharpe >= 1.0
                            else "text-gray-300"
                        )
                        ui.label(f"{prefix}{sharpe:.2f}").classes(
                            f"{color} text-sm"
                        )

                # 7. alpha
                with ui.column().classes("p-2 gap-0 items-end"):
                    if alpha is None:
                        ui.label("⚪ —").classes(
                            "text-gray-500 text-xs"
                        )
                    else:
                        prefix = (
                            "📈 " if key == best_alpha and alpha > 0 else ""
                        )
                        color = (
                            "text-emerald-400" if alpha > 0
                            else "text-red-400"
                        )
                        mode_mark = "" if alpha_mode == "real" else "*"
                        ui.label(
                            f"{prefix}{alpha:+.2f}{mode_mark}"
                        ).classes(f"{color} font-bold text-sm")

                # 8. TP 포화
                with ui.column().classes("p-2 gap-0 items-end"):
                    warn = tp_sat >= tp_th
                    prefix = "⚠️ " if warn else ""
                    color = "text-amber-400" if warn else "text-gray-300"
                    ui.label(f"{prefix}{tp_sat:.0f}%").classes(
                        f"{color} text-sm"
                    )
                    ui.label(f"임계 {tp_th}").classes(
                        "text-[9px] text-gray-500"
                    )

                # 9. 판정 — [v3.9.16b] SSOT 함수 사용 (verdict_card와 동일)
                # [v3.9.17] backtest_verdict 모듈로 분리됨
                from components.backtest_verdict import (
                    _derive_strategy_verdict,
                )
                vd = _derive_strategy_verdict(r, cfg)
                with ui.column().classes("p-2 gap-0"):
                    ui.label(f"{vd['icon']} {vd['title']}").classes(
                        f"{vd['color_class']} text-xs font-bold"
                    )
                    # 짧은 사유 — anomaly나 misses 첫 2개
                    if vd.get("reasons"):
                        ui.label(" · ".join(vd["reasons"][:2])).classes(
                            "text-[10px] text-amber-200"
                            if vd["is_anomaly"]
                            else "text-[10px] text-gray-400"
                        )

    # ─── 표 아래 범례 ───
    ui.label(
        "💡 * 표시 = 간이 알파 (KOSPI 단일 시점). 표시 없음 = 정확 알파 "
        "(거래일별)."
    ).classes("text-[10px] text-gray-500 mt-2 italic")
    ui.label(
        "⚠️ anomaly 행은 amber 배경. 간이 백테스트 구조상 lookahead/자금 "
        "제약 미반영으로 과대추정 가능성."
    ).classes("text-[10px] text-gray-500 italic")


