# -*- coding: utf-8 -*-
"""
kelly_widget.py — Kelly Criterion 기반 권장 매수 비중 계산기 (v1.0)
═══════════════════════════════════════════════════════════════════
Tab 3 (내 자산) 또는 종목 상세 창에서 호출.

Half-Kelly 적용: 실전에서는 Full-Kelly의 50%만 투입하여 변동성을 낮춤.
"""

import math
import logging
from typing import Optional

logger = logging.getLogger("kelly_widget")


# ─────────────────────────────────────────────────────
# ✅ [Fix #4] 실전 매매 일지에서 최근 N번 승률 조회
# 백테스트 승률(system)과 실전 승률(real) 비교 후 페널티 적용
# ─────────────────────────────────────────────────────
def _get_real_win_rate(recent_n: int = 20) -> Optional[float]:
    """
    trade_journal_tab의 DB에서 최근 `recent_n`건 완료 거래 승률 반환.
    데이터 부족(< 5건) 시 None 반환 → caller가 시스템 승률 그대로 사용.
    """
    try:
        import sqlite3, os
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        db_path  = os.path.join(data_dir, "ldy_trader.db")
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cur  = conn.execute(
            """
            SELECT outcome FROM trade_journal
            WHERE outcome IN ('WIN','LOSS')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (recent_n,),
        )
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        if len(rows) < 5:          # 5건 미만이면 통계 불신뢰
            return None
        wr = rows.count("WIN") / len(rows)
        return round(wr, 4)
    except Exception as e:
        logger.debug(f"실전 승률 조회 실패: {e}")
        return None


def _apply_win_rate_discount(system_wr: float, real_wr: Optional[float]) -> tuple:
    """
    실전 승률 < 시스템 승률이면 페널티 할인율 적용.
    Returns: (adjusted_wr, discount_pct, warning_msg)

    할인율 공식:
      ratio = real / system
      adjusted = system × sqrt(ratio)   ← 너무 급격한 할인을 완화하는 √ 스무딩
    """
    if real_wr is None:
        return system_wr, 0.0, ""

    ratio = real_wr / system_wr if system_wr > 0 else 1.0

    if ratio >= 1.0:
        # 실전 승률이 더 높거나 같음 → 할인 없음 (보수적 유지)
        return system_wr, 0.0, f"✅ 실전 승률({real_wr*100:.1f}%) ≥ 시스템 승률 — 할인 없음"

    import math
    adjusted   = system_wr * math.sqrt(ratio)
    discount   = (1 - math.sqrt(ratio)) * 100
    warning    = (
        f"⚠️ 실전 승률({real_wr*100:.1f}%) < 시스템({system_wr*100:.1f}%) "
        f"→ √보정 적용, 사용 승률 {adjusted*100:.1f}% (할인 {discount:.1f}%)"
    )
    return round(adjusted, 4), round(discount, 1), warning


def kelly_fraction(win_rate: float, rr: float) -> float:
    """
    Kelly Criterion: f* = (b*p - q) / b
      p = win_rate, q = 1 - p
      b = reward/risk ratio (RR)
    Returns: Half-Kelly fraction (0.0 ~ 0.25 사이로 클리핑)
    """
    if win_rate <= 0 or win_rate >= 1 or rr <= 0:
        return 0.0
    p = win_rate
    q = 1.0 - p
    b = rr
    full_k = (b * p - q) / b
    if full_k <= 0:
        return 0.0
    half_k = full_k * 0.5
    return round(min(half_k, 0.25), 4)   # 최대 25%로 캡


def kelly_summary(
    total_capital: float,
    win_rate: float,
    rr: float,
    stock_name: str = "",
    use_real_wr: bool = True,          # ✅ [Fix #4] 실전 승률 피드백 ON/OFF
) -> dict:
    """
    켈리 계산 결과 요약 dict 반환.
    use_real_wr=True 시 매매 일지 최근 20건 실전 승률로 시스템 승률 보정.
    """
    # ── ✅ Fix #4: 실전 승률 조회 + 페널티 적용 ──
    real_wr      = _get_real_win_rate(20) if use_real_wr else None
    adj_wr, disc, wr_warning = _apply_win_rate_discount(win_rate, real_wr)
    effective_wr = adj_wr   # 실제 계산에 쓰이는 승률

    p, q, b = effective_wr, 1 - effective_wr, rr
    full_k = (b * p - q) / b if b > 0 else 0
    neg_ev = full_k <= 0

    if neg_ev:
        return {
            "fraction": 0.0,
            "fraction_pct": 0.0,
            "bet_amount": 0,
            "full_kelly_pct": round(full_k * 100, 1),
            "is_negative_ev": True,
            "win_rate_used": round(effective_wr * 100, 1),
            "real_win_rate": round(real_wr * 100, 1) if real_wr else None,
            "discount_pct": disc,
            "wr_warning": wr_warning,
            "label": f"⚠️ 기대수익 음수 (EV<0). {stock_name} 진입 비권장",
        }

    hk  = min(full_k * 0.5, 0.25)
    amt = int(total_capital * hk)

    if hk >= 0.20:
        emoji, desc = "🚀", "강한 우위 — Half-Kelly 풀배팅 구간"
    elif hk >= 0.10:
        emoji, desc = "💪", "양호한 우위 — 집중 투자 적합"
    elif hk >= 0.05:
        emoji, desc = "👌", "보통 우위 — 분할 매수 권장"
    else:
        emoji, desc = "🤏", "약한 우위 — 소규모 테스트 투자"

    label = f"{emoji} {stock_name}: 총 자산의 {hk*100:.1f}% ({amt:,}원) — {desc}"

    return {
        "fraction": round(hk, 4),
        "fraction_pct": round(hk * 100, 2),
        "bet_amount": amt,
        "full_kelly_pct": round(full_k * 100, 1),
        "is_negative_ev": False,
        "win_rate_used": round(effective_wr * 100, 1),
        "real_win_rate": round(real_wr * 100, 1) if real_wr else None,
        "discount_pct": disc,
        "wr_warning": wr_warning,
        "label": label,
    }


# ────────────────────────────────────────────────
# NiceGUI 위젯 렌더러
# ────────────────────────────────────────────────
def render_kelly_calculator(row_data: dict, container):
    """
    [v21.3] 종목 선택 시 승률·손익비 자동 반영.
    row_data keys: 종목명, EST_WIN_RATE, RR_NOW_TP1, DISPLAY_SCORE
    """
    try:
        from nicegui import ui
    except ImportError:
        return

    name     = row_data.get("종목명", "이 종목")

    # [v21.3] EST_WIN_RATE → RR_NOW_TP1 자동 매핑 (현재가 기준)
    win_rate = float(row_data.get("EST_WIN_RATE",
                     row_data.get("WIN_RATE",
                     row_data.get("win_rate", 0.55))))
    rr       = float(row_data.get("RR_NOW_TP1",
                     row_data.get("RR1",
                     row_data.get("rr1", 2.0))))

    if win_rate <= 0 or win_rate > 1:
        win_rate = 0.55
    if rr <= 0:
        rr = 2.0

    with container:
        ui.label("💰 Kelly Criterion 포지션 사이저").classes(
            "text-sm font-bold text-yellow-400 mb-2"
        )

        # [v21.3] 종목 정보 헤더
        _close = row_data.get("LIVE_PRICE", None)
        if _close is None or (isinstance(_close, float) and _close != _close):
            _close = row_data.get("종가", 0)
        _tp1 = row_data.get("추천매도가1", 0)
        _stop = row_data.get("손절가", 0)
        try:
            _close, _tp1, _stop = float(_close), float(_tp1), float(_stop)
        except (TypeError, ValueError):
            _close, _tp1, _stop = 0, 0, 0

        ui.label(
            f"📌 {name} | 승률 {win_rate*100:.1f}% | "
            f"현재가→TP1 RR {rr:.2f}:1"
        ).classes("text-xs text-cyan-400 mb-2")

        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            capital_input = ui.number(
                "총 투자 가능 금액 (원)",
                value=10_000_000,
                min=100_000,
                step=500_000,
                format="%.0f",
            ).classes("flex-1 min-w-[200px]")

            # 승률·손익비는 자동 세팅, 읽기전용 표시
            wr_display = ui.number(
                "시스템 승률 (%)",
                value=round(win_rate * 100, 1),
            ).classes("min-w-[130px]").props("readonly outlined")

            rr_display = ui.number(
                "현재가 RR (TP1)",
                value=round(rr, 2),
            ).classes("min-w-[110px]").props("readonly outlined")

        result_area = ui.column().classes("w-full mt-2")
        use_real = ui.checkbox("📓 매매 일지 실전 승률 반영", value=True).classes("text-xs text-gray-300")

        def _calc():
            result_area.clear()
            try:
                cap = float(str(capital_input.value).replace(",", ""))
                wr  = win_rate  # [v21.3] 종목 자동 세팅값 사용
                rr_ = rr
            except (TypeError, ValueError):
                return

            res = kelly_summary(cap, wr, rr_, name, use_real_wr=use_real.value)

            with result_area:
                # ✅ [Fix #4] 실전 승률 vs 시스템 승률 경고 배너
                if res.get("wr_warning"):
                    warn_color = "text-yellow-400 bg-yellow-900/20 border-yellow-700/50"
                    if res["discount_pct"] > 0:
                        warn_color = "text-orange-400 bg-orange-900/20 border-orange-700/50"
                    ui.label(res["wr_warning"]).classes(
                        f"text-xs p-2 rounded-lg border w-full mb-2 {warn_color}"
                    )
                if res.get("real_win_rate") is not None:
                    ui.label(
                        f"📊 사용 승률: {res['win_rate_used']}%  "
                        f"(시스템 입력: {round(wr*100,1)}%  /  실전 최근 20건: {res['real_win_rate']}%)"
                    ).classes("text-xs text-gray-400 mb-2")

                if res["is_negative_ev"]:
                    ui.label(res["label"]).classes(
                        "text-red-400 text-sm font-bold p-3 rounded-lg bg-red-900/30 w-full"
                    )
                    return

                # 결과 카드
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    _kcard("권장 투입 비중",    f'{res["fraction_pct"]:.1f}%',   "#F59E0B")
                    _kcard("권장 투자금액",     f'{res["bet_amount"]:,}원',       "#10B981")
                    _kcard("Full-Kelly 참고",  f'{res["full_kelly_pct"]:.1f}%',  "#6B7280")

                ui.label(res["label"]).classes(
                    "text-sm p-3 rounded-lg bg-yellow-900/20 border border-yellow-700/50 w-full mt-1"
                ).style("color:#FCD34D")

                # Kelly 비중 게이지
                pct = min(res["fraction_pct"], 25)
                bar_w = int(pct / 25 * 100)
                color = "#10B981" if pct < 10 else "#F59E0B" if pct < 20 else "#EF4444"
                ui.html(f"""
                <div style="margin-top:8px;">
                  <div style="font-size:11px;color:#9CA3AF;margin-bottom:4px;">
                    Half-Kelly 비중 (최대 25%)
                  </div>
                  <div style="background:rgba(255,255,255,0.1);border-radius:6px;height:14px;width:100%;">
                    <div style="background:{color};width:{bar_w}%;height:100%;border-radius:6px;
                                transition:width 0.4s;display:flex;align-items:center;padding-left:6px;">
                      <span style="font-size:10px;color:white;font-weight:bold;">{pct:.1f}%</span>
                    </div>
                  </div>
                </div>
                """)

        def _kcard(title, val, color):
            with ui.card().classes(
                "p-3 min-w-[130px] bg-[#0d0d1a] border border-gray-700 rounded-lg"
            ):
                ui.label(title).classes("text-xs text-gray-400")
                ui.label(val).classes("text-base font-bold").style(f"color:{color}")

        ui.button("📐 계산", on_click=_calc).props("dense flat").classes(
            "mt-2 text-yellow-400 border border-yellow-700"
        )
        capital_input.on("update:model-value", lambda _: _calc())  # [v21.3] 금액 변경 시 자동 재계산
        _calc()   # 초기 계산


# ── 포트폴리오 전체 Kelly 요약 (Tab 3 상단) ──
def render_portfolio_kelly_summary(pf_rows: list, total_capital: float, container):
    """
    pf_rows: [{종목명, 점수, WIN_RATE, RR1, ...}, ...]
    포트폴리오 전체에 대한 Kelly 분산 비중 테이블 렌더
    """
    try:
        from nicegui import ui
    except ImportError:
        return

    if not pf_rows or total_capital <= 0:
        return

    with container:
        ui.label("📐 Kelly 기반 권장 비중 배분").classes(
            "text-sm font-bold text-yellow-400 mb-3"
        )
        rows_out = []
        total_frac = 0.0
        for r in pf_rows:
            wr  = float(r.get("WIN_RATE", r.get("win_rate", 0.55)))
            rr_ = float(r.get("RR1",      r.get("rr1",      2.0)))
            if wr <= 0 or wr > 1: wr = 0.55
            if rr_ <= 0: rr_ = 2.0
            fk = kelly_fraction(wr, rr_)
            total_frac += fk
            rows_out.append({
                "name": r.get("종목명", "?"),
                "frac": fk,
            })

        # 전체 비중이 100% 초과 시 정규화 (비례 축소)
        if total_frac > 1.0:
            scale = 1.0 / total_frac
            for r2 in rows_out:
                r2["frac"] *= scale

        cols = [
            {"name": "name",  "label": "종목명", "field": "name",  "align": "left"},
            {"name": "pct",   "label": "Kelly %", "field": "pct",  "align": "center"},
            {"name": "amt",   "label": "권장금액 (원)", "field": "amt", "align": "right"},
        ]
        tbl_rows = []
        for r2 in rows_out:
            tbl_rows.append({
                "name": r2["name"],
                "pct":  f'{r2["frac"]*100:.1f}%',
                "amt":  f'{int(total_capital * r2["frac"]):,}',
            })

        ui.table(
            columns=cols, rows=tbl_rows, row_key="name"
        ).classes("w-full").props("dense dark flat bordered")
        ui.label(
            "* Half-Kelly 적용 | 총 비중 합이 100% 초과 시 비례 정규화"
        ).classes("text-xs text-gray-500 mt-1")
