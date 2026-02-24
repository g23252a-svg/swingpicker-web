# -*- coding: utf-8 -*-
"""
trade_plan.py — SSOT(단일 진실) 트레이딩 계획 엔진
═══════════════════════════════════════════════════════

100점 근접을 위한 7대 조건 구현:
  1. SSOT: ENTRY/SL/TP를 이 파일에서만 산출 (다른 곳 계산 금지)
  2. ExecRule: 체결 규칙을 문장+코드로 고정 (ID 부여)
  3. Contract: 추천 결과 필수 스키마 강제 (누락 시 에러)
  4. 룩어헤드 방지: pre_market 모드에서 장중 데이터 차단
  5. 포지션 사이징: 계좌 대비 1회 손실 한도 기반
  6. Observability: 모든 의사결정 근거를 reason에 남김
  7. 회귀테스트: 동일 입력 → 동일 출력 보장 (frozen dataclass)

사용법:
    from trade_plan import build_trade_plan, exec_bar, validate_row, ExecRule
    
    plan = build_trade_plan(features, exec_rule=ExecRule())
    row = plan.to_row()  # → recommend CSV에 저장
    
    # 백테스트에서:
    result = exec_bar(plan, bar_open, bar_high, bar_low, bar_close, rule=ExecRule())
"""

import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple

import stop_logic as SL


# ═══════════════════════════════════════════════════
#  1. 체결 규칙 (Execution Rule) — 버전 관리
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class ExecRule:
    """
    체결 규칙을 코드로 고정. 규칙이 바뀌면 rule_id가 바뀜.
    → 백테스트 결과를 rule_id별로 비교 가능
    """
    rule_id: str = "v1_sl_first_gap_at_open"

    # 동시터치: 같은 바에서 TP/SL 모두 터치 시 우선순위
    tp_sl_same_bar_priority: str = "SL"

    # 갭 체결: 시가가 SL/TP를 뛰어넘을 때
    gap_fill: str = "OPEN"

    # 슬리피지 (bps = 0.01%) — 체결비용의 SSOT
    sl_slippage_bps: float = 30.0   # SL 체결 시 (0.3%)
    tp_slippage_bps: float = 5.0    # TP 체결 시 (0.05%)

    # 수수료 (왕복, bps)
    fee_bps: float = 15.0           # 매수+매도 합산 0.15%

    # 스케일아웃: True이면 TP1/TP2/TP3에서 분할 청산
    # False이면 TP1에서만 전량 청산 (TP2/TP3는 참고용)
    use_scaleout: bool = False
    scaleout_pcts: tuple = (50.0, 30.0, 20.0)  # TP1/TP2/TP3 비중

    # 최소 주문금액 (원) — 이하이면 진입 스킵
    min_order_amount: float = 100000.0


# ═══════════════════════════════════════════════════
#  2. 트레이딩 계획 (Trade Plan) — SSOT
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class TradePlan:
    """
    SSOT: 모든 가격(entry/sl/tp)은 여기서만 생성.
    frozen=True → 생성 후 변경 불가 → 회귀테스트 안정성 보장
    """
    # 가격 (tick 라운딩 완료된 값)
    entry: float
    stop: float
    tp1: float
    tp2: Optional[float] = None
    tp3: Optional[float] = None

    # 진입 제어
    position_pct: float = 100.0     # 진입 비중 (0=보류, 50=분할, 100=전량)
    entry_action: str = "enter"     # "enter" | "split" | "hold"

    # 근거/메타
    plan_reason: str = ""           # NORMAL/GAP/SWING/EST_MCAP 등
    stop_pct: float = 0.0           # 실제 손절 퍼센트
    max_loss_pct: float = 0.0       # 시총 기반 캡
    rr_mult: float = 0.0           # R:R 배수
    regime: str = "normal"          # normal/high_vol/low_vol
    exec_rule_id: str = ""          # 적용된 체결 규칙 ID

    # [Phase 1-2] Time Stop
    time_stop_days: int = 0          # 0=비활성 (하위호환), 7=7영업일
    time_stop_min_move_pct: float = 2.0  # N일 내 이 수준 미달 시 청산
    time_stop_extend_if_profit: bool = True  # 수익 중이면 연장 허용

    def to_row(self) -> Dict[str, Any]:
        """recommend CSV용 dict 변환 + 계약 검증"""
        row = {
            "ENTRY_PRICE": self.entry,
            "STOP_PRICE": self.stop,
            "TP1": self.tp1,
            "TP2": self.tp2,
            "TP3": self.tp3,
            "POSITION_PCT": self.position_pct,
            "ENTRY_ACTION": self.entry_action,
            "PLAN_REASON": self.plan_reason,
            "STOP_PCT": round(self.stop_pct, 2),
            "MAX_LOSS_PCT": round(self.max_loss_pct, 1),
            "RR_MULT": round(self.rr_mult, 1),
            "REGIME": self.regime,
            "EXEC_RULE_ID": self.exec_rule_id,
            "TIME_STOP_DAYS": self.time_stop_days,
        }
        validate_row(row)
        return row


# ═══════════════════════════════════════════════════
#  3. 데이터 계약 (Contract) — 누락 시 에러
# ═══════════════════════════════════════════════════

REQUIRED_PLAN_KEYS = frozenset({
    "ENTRY_PRICE", "STOP_PRICE", "TP1",
    "POSITION_PCT", "ENTRY_ACTION", "PLAN_REASON",
    "EXEC_RULE_ID",
})


def validate_row(row: Dict[str, Any]) -> None:
    """추천 결과가 계약을 만족하는지 강제 검증 (불변조건 포함)."""
    missing = REQUIRED_PLAN_KEYS - set(row.keys())
    if missing:
        raise ValueError(f"[Contract] 필수 컬럼 누락: {missing}")

    # ✅ entry/stop/tp1을 '진짜 필수 숫자'로 강제 (None/0/NaN 불허)
    try:
        entry = float(row["ENTRY_PRICE"])
        stop = float(row["STOP_PRICE"])
        tp1 = float(row["TP1"])
    except (TypeError, ValueError, KeyError) as e:
        raise ValueError(f"[Contract] price cast fail: {e}")

    if not np.isfinite(entry) or entry <= 0:
        raise ValueError(f"[Contract] ENTRY_PRICE={entry} invalid")
    if not np.isfinite(stop) or stop <= 0:
        raise ValueError(f"[Contract] STOP_PRICE={stop} invalid")
    if not np.isfinite(tp1) or tp1 <= 0:
        raise ValueError(f"[Contract] TP1={tp1} invalid")

    if stop >= entry:
        raise ValueError(f"[Contract] STOP_PRICE={stop} ≥ ENTRY_PRICE={entry}")
    if tp1 <= entry:
        raise ValueError(f"[Contract] TP1={tp1} ≤ ENTRY_PRICE={entry}")

    act = str(row.get("ENTRY_ACTION", ""))
    if act not in ("enter", "split", "hold"):
        raise ValueError(f"[Contract] ENTRY_ACTION invalid: {act}")

    pos = float(row.get("POSITION_PCT", 0) or 0)
    if not (0.0 <= pos <= 100.0):
        raise ValueError(f"[Contract] POSITION_PCT out of range: {pos}")

    # ── 불변조건 (Invariants) ──
    if act == "hold" and pos > 0:
        raise ValueError(f"[Contract] hold인데 POSITION_PCT={pos} > 0")
    if act == "split" and pos >= 100.0:
        raise ValueError(f"[Contract] split인데 POSITION_PCT={pos} ≥ 100")

    # TP 정렬 (있을 경우만)
    tp2v = row.get("TP2", None)
    tp3v = row.get("TP3", None)
    tp2 = float(tp2v) if (tp2v is not None and str(tp2v).strip() != "" and float(tp2v) > 0) else None
    tp3 = float(tp3v) if (tp3v is not None and str(tp3v).strip() != "" and float(tp3v) > 0) else None
    if tp2 is not None and tp2 <= tp1:
        raise ValueError(f"[Contract] TP2={tp2} ≤ TP1={tp1}")
    if tp3 is not None and tp2 is not None and tp3 <= tp2:
        raise ValueError(f"[Contract] TP3={tp3} ≤ TP2={tp2}")

    stop_pct = float(row.get("STOP_PCT", 0) or 0)
    if stop_pct < 0 or stop_pct > 30:
        raise ValueError(f"[Contract] STOP_PCT={stop_pct} out of [0,30]")


# ═══════════════════════════════════════════════════
#  4. SSOT 빌더 — build_trade_plan()
# ═══════════════════════════════════════════════════

def build_trade_plan(
    buy: float,
    atr_val: float,
    last_c: float,
    mcap: Optional[float] = None,
    tv_eok: Optional[float] = None,
    today_low: Optional[float] = None,
    gap_up_pct: Optional[float] = None,
    swing_low_10: Optional[float] = None,
    dist_to_swing: Optional[float] = None,
    ret_1d: float = 0.0,
    gap_pct: float = 0.0,
    major_net: float = 0.0,
    major_ratio: float = 0.0,
    is_vi_triggered: bool = False,
    account_risk_pct: float = 1.0,
    account_value_krw: Optional[float] = None,
    exec_rule: Optional[ExecRule] = None,
) -> TradePlan:
    """
    ✅ SSOT: ENTRY/STOP/TP를 여기서만 만든다.
    collector의 analyze_ticker는 이 함수만 호출하면 됨.

    Args:
        buy: 기본 매수가 (종가 기반)
        atr_val: ATR 값 (원)
        last_c: 최종 종가 (원)
        mcap: 시가총액 (억원)
        tv_eok: 거래대금 (억원)
        today_low: 금일 저가
        gap_up_pct: 갭상승률 (%)
        swing_low_10: 10일 Swing Low
        dist_to_swing: Swing Low까지 거리 (%)
        ret_1d: 당일 등락률 (%)
        gap_pct: 갭 퍼센트 (%)
        major_net: 메이저 순매수 (원)
        major_ratio: 메이저 비중
        is_vi_triggered: VI 발동 여부
        account_risk_pct: 계좌 대비 1회 최대 손실 (%, 기본 1%)
        exec_rule: 체결 규칙 (None이면 기본값)

    Returns:
        TradePlan (frozen, immutable)
    """
    if exec_rule is None:
        exec_rule = ExecRule()

    cfg = SL.get_config()
    regime = getattr(cfg, "regime_name", "normal")

    # ── (1) 진입 필터 ──
    entry_filter = SL.check_entry_filter(
        ret_1d=ret_1d, gap_pct=gap_pct,
        is_vi_triggered=is_vi_triggered,
    )
    entry_action = entry_filter["action"]
    position_pct = entry_filter["position_pct"]

    # ── (2) 손절가 (SSOT — stop_logic.calc_stop_price) ──
    stop, stop_pct, max_loss_pct, stop_reason = SL.calc_stop_price(
        buy=buy,
        atr_val=atr_val,
        mcap=mcap,
        today_low=today_low,
        gap_up_pct=gap_up_pct if (gap_pct >= 7.0 or ret_1d >= 7.0) else None,
        swing_low_10=swing_low_10,
        dist_to_swing=dist_to_swing,
        tv_eok=tv_eok,
        use_tick=True,
    )

    # ── (3) 수급 보정 ──
    buy_adj, stop_adj = SL.adjust_by_flow(
        buy=buy, stop=stop, last_c=last_c,
        major_net=major_net, major_ratio=major_ratio,
    )

    # ── (4) tick 라운딩 (정책 고정) ──
    entry_final = float(SL.round_to_tick(buy_adj))
    stop_final = float(SL.floor_to_tick(stop_adj))

    # 최후 안전: stop >= entry 방지
    if stop_final >= entry_final and entry_final > 0:
        stop_final = float(SL.floor_to_tick(entry_final * 0.99))

    # stop_pct 재계산 (tick 라운딩 후)
    actual_stop_pct = (1.0 - stop_final / entry_final) * 100.0 if entry_final > 0 else stop_pct

    # ── (5) R:R + 목표가 (✅ tick 반영된 risk로 계산 — SSOT) ──
    rr_mult = SL.calc_rr_multiplier(atr_val, entry_final)
    risk_final = max(0.0, entry_final - stop_final)

    # risk가 0이 되는 극단 방어
    if risk_final <= 0:
        risk_final = entry_final * 0.01

    tp1_raw = entry_final + risk_final * rr_mult
    tp2_raw = entry_final + risk_final * (rr_mult + 0.5)
    tp3_raw = entry_final + risk_final * (rr_mult + 1.0)

    tp1_final = float(SL.ceil_to_tick(tp1_raw))
    tp2_final = float(SL.ceil_to_tick(tp2_raw))
    tp3_final = float(SL.ceil_to_tick(tp3_raw))

    # ── (6) 포지션 사이징 ──
    if risk_final > 0 and account_risk_pct > 0:
        risk_ratio = (actual_stop_pct / 100.0)
        if risk_ratio > 0:
            max_position = min(account_risk_pct / risk_ratio * 100.0, 100.0)
            position_pct = min(position_pct, max_position)

    # ── (6-B) 최소 주문금액 체크 ──
    if account_value_krw is not None and account_value_krw > 0:
        est_order = account_value_krw * (position_pct / 100.0)
        if est_order < exec_rule.min_order_amount:
            entry_action = "hold"
            position_pct = 0.0
            reason_parts_extra = f"MIN_ORDER_SKIP({est_order:.0f}<{exec_rule.min_order_amount:.0f})"
        else:
            reason_parts_extra = ""
    else:
        reason_parts_extra = ""

    # ── (7) 근거 조합 ──
    reason_parts = [stop_reason]
    ef_reason = entry_filter.get("reason", "")
    if ef_reason:
        reason_parts.append(f"EF:{ef_reason}")
    if entry_action != "enter":
        reason_parts.append(f"ENTRY:{entry_action}")
    if reason_parts_extra:
        reason_parts.append(reason_parts_extra)
    plan_reason = "+".join([p for p in reason_parts if p])

    return TradePlan(
        entry=entry_final,
        stop=stop_final,
        tp1=tp1_final,
        tp2=tp2_final,
        tp3=tp3_final,
        position_pct=round(position_pct, 1),
        entry_action=entry_action,
        plan_reason=plan_reason,
        stop_pct=actual_stop_pct,
        max_loss_pct=max_loss_pct,
        rr_mult=rr_mult,
        regime=regime,
        exec_rule_id=exec_rule.rule_id,
    )


# ═══════════════════════════════════════════════════
#  5. 체결 엔진 — exec_bar()
# ═══════════════════════════════════════════════════

@dataclass
class BarResult:
    """단일 바(일봉) 체결 결과"""
    action: str = "hold"        # "hold" | "stop_hit" | "tp_hit" | "time_stop" | "timeout" | "none"
    fill_price: float = 0.0     # 체결가 (0이면 미체결)
    return_pct: float = 0.0     # 수익률 (%)
    reason: str = ""


# ═══════════════════════════════════════════════════
#  [Phase 1-3] 동적 슬리피지 추정
# ═══════════════════════════════════════════════════

def estimate_slippage_bps(tv_eok, config=None):
    """
    거래대금(억) 기반 슬리피지 추정 (bps).

    Args:
        tv_eok: 거래대금 (억 원). None/0 → 최대 슬리피지
        config: CollectorConfig (None이면 DEFAULT_CONFIG)

    Returns:
        슬리피지 (bps). 10bps = 0.1%
    """
    from collector_config import DEFAULT_CONFIG as _DC
    cfg = config or _DC

    base = cfg.slippage_base_bps
    mult = cfg.slippage_low_liq_mult
    threshold = cfg.slippage_liq_threshold_eok

    if tv_eok is None or tv_eok <= 0:
        return base * mult

    if tv_eok < threshold:
        ratio = 1.0 + (mult - 1.0) * (1.0 - tv_eok / threshold)
        return base * ratio

    return base


def _apply_fee(ret_pct: float, rule: ExecRule) -> float:
    """fee_bps(왕복)를 수익률에서 차감: 15bps = 0.15%"""
    return float(ret_pct) - float(rule.fee_bps) / 100.0


def exec_bar(
    plan: TradePlan,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    rule: Optional[ExecRule] = None,
) -> BarResult:
    """
    ✅ SSOT 체결 엔진: collector/validation/backtest 모두 이 함수를 호출.

    Returns:
        BarResult
    """
    if rule is None:
        rule = ExecRule()

    # NaN/inf/None 방어
    vals = [plan.entry, plan.stop, plan.tp1, bar_open, bar_high, bar_low, bar_close]
    if any(v is None or not np.isfinite(float(v)) for v in vals):
        return BarResult(action="none", reason="non_finite_prices")

    entry = float(plan.entry)
    sl = float(plan.stop)
    tp = float(plan.tp1)
    bar_open = float(bar_open)
    bar_high = float(bar_high)
    bar_low = float(bar_low)
    bar_close = float(bar_close)

    # high/low 뒤집힘 방어
    if bar_high < bar_low:
        bar_high, bar_low = bar_low, bar_high

    if entry <= 0 or sl <= 0 or bar_open <= 0:
        return BarResult(action="none", reason="invalid_prices")

    sl_slip = sl * (rule.sl_slippage_bps / 10000.0)
    tp_slip = tp * (rule.tp_slippage_bps / 10000.0) if tp and tp > 0 else 0.0

    # ── (1) 갭하락: 시가 < SL ──
    if bar_open < sl:
        fill = bar_open if rule.gap_fill == "OPEN" else (sl - sl_slip)
        ret = _apply_fee((fill / entry - 1.0) * 100.0, rule)
        return BarResult(action="stop_hit", fill_price=fill, return_pct=ret,
                         reason="gap_down_sl")

    # ── (2) 갭상승: 시가 > TP ──
    if tp and tp > 0 and bar_open > tp:
        if rule.gap_fill == "OPEN":
            fill = bar_open
        else:
            # ✅ FIX: LEVEL이면 tp-슬립 (tp보다 좋아지면 안 됨)
            fill = tp - tp_slip
        ret = _apply_fee((fill / entry - 1.0) * 100.0, rule)
        return BarResult(action="tp_hit", fill_price=fill, return_pct=ret,
                         reason="gap_up_tp")

    # ── (3) 동시터치 (High≥TP & Low≤SL) ──
    sl_touched = (bar_low <= sl)
    tp_touched = (tp and tp > 0 and bar_high >= tp)

    if sl_touched and tp_touched:
        if rule.tp_sl_same_bar_priority == "SL":
            fill = sl - sl_slip
            ret = _apply_fee((fill / entry - 1.0) * 100.0, rule)
            return BarResult(action="stop_hit", fill_price=fill, return_pct=ret,
                             reason="same_bar_sl_priority")
        else:
            fill = tp - tp_slip
            ret = _apply_fee((fill / entry - 1.0) * 100.0, rule)
            return BarResult(action="tp_hit", fill_price=fill, return_pct=ret,
                             reason="same_bar_tp_priority")

    # ── (4) 장중 SL ──
    if sl_touched:
        if bar_open == bar_low == bar_high and bar_low <= sl:
            fill = bar_open
            reason = "limit_down_sl"
        else:
            fill = sl - sl_slip
            reason = "intraday_sl"
        ret = _apply_fee((fill / entry - 1.0) * 100.0, rule)
        return BarResult(action="stop_hit", fill_price=fill, return_pct=ret,
                         reason=reason)

    # ── (5) 장중 TP ──
    if tp_touched:
        fill = tp - tp_slip
        ret = _apply_fee((fill / entry - 1.0) * 100.0, rule)
        return BarResult(action="tp_hit", fill_price=fill, return_pct=ret,
                         reason="intraday_tp")

    # ── (6) 미체결 ──
    return BarResult(action="hold", reason="no_trigger")


# ═══════════════════════════════════════════════════
#  6. 백테스트 헬퍼 — multi-bar 실행
# ═══════════════════════════════════════════════════

def _fee_pct_for_exits(rule: ExecRule, n_exits: int) -> float:
    """scaleout 시 매도 횟수에 따른 근사 수수료 (왕복 기준)"""
    base = float(rule.fee_bps) / 100.0
    return base * (0.5 + 0.5 * max(1, n_exits))


def exec_multi_bar(
    plan: TradePlan,
    bars: List[Tuple[float, float, float, float]],
    rule: Optional[ExecRule] = None,
    max_hold_days: int = 20,
) -> BarResult:
    """
    여러 바에 걸쳐 TradePlan의 체결을 시뮬레이션.
    use_scaleout=True이면 TP1/TP2/TP3에서 분할 청산.
    """
    if rule is None:
        rule = ExecRule()

    # ── scaleout OFF: 기존 단순 로직 + [Phase 1-2] Time Stop ──
    if not rule.use_scaleout:
        for i, (o, h, l, c) in enumerate(bars[:max_hold_days]):
            if o <= 0 or not np.isfinite(o):
                continue
            result = exec_bar(plan, o, h, l, c, rule=rule)
            if result.action in ("stop_hit", "tp_hit"):
                return result

            # ── [Phase 1-2] Time Stop 체크 ──
            if plan.time_stop_days > 0 and i >= plan.time_stop_days - 1:
                move_pct = (c / plan.entry - 1.0) * 100.0 if plan.entry > 0 else 0.0
                # 수익 중이고 extend_if_profit 설정이면 → 연장
                if plan.time_stop_extend_if_profit and move_pct > 0:
                    pass  # 수익 중이면 time_stop 스킵
                elif move_pct < plan.time_stop_min_move_pct:
                    ret = _apply_fee(move_pct, rule)
                    return BarResult(
                        action="time_stop", fill_price=c,
                        return_pct=ret,
                        reason=f"time_stop_{plan.time_stop_days}d_move_{move_pct:.1f}%"
                    )

        if bars and len(bars) > 0:
            last_close = bars[min(max_hold_days - 1, len(bars) - 1)][3]
            if last_close > 0 and np.isfinite(last_close) and plan.entry > 0:
                ret = _apply_fee((last_close / plan.entry - 1.0) * 100.0, rule)
                return BarResult(action="timeout", fill_price=last_close,
                                 return_pct=ret, reason=f"max_hold_{max_hold_days}d")
        return BarResult(action="none", reason="no_data")

    # ── scaleout ON: 상태 기반 분할 청산 ──
    entry = float(plan.entry)
    sl = float(plan.stop)
    tp_levels = [plan.tp1, plan.tp2, plan.tp3]
    tp_levels = [float(x) for x in tp_levels if (x is not None and float(x) > 0)]
    tp_levels.sort()

    scale_pcts = list(rule.scaleout_pcts)[:len(tp_levels)]
    s = sum(scale_pcts) if scale_pcts else 100.0
    if s <= 0:
        scale_pcts = [100.0 / len(tp_levels)] * len(tp_levels)
        s = 100.0
    scale_pcts = [p * 100.0 / s for p in scale_pcts]

    # 각 TP 레벨이 이미 청산되었는지 추적
    tp_filled = [False] * len(tp_levels)
    remaining = 100.0
    exits = []
    n_exits = 0

    sl_slip = sl * (rule.sl_slippage_bps / 10000.0)

    def _tp_fill(tp_val, bar_o):
        tp_s = tp_val * (rule.tp_slippage_bps / 10000.0)
        if rule.gap_fill == "OPEN" and bar_o > tp_val:
            return bar_o
        return tp_val - tp_s

    for i, (o, h, l, c) in enumerate(bars[:max_hold_days]):
        if remaining <= 0:
            break
        if o <= 0 or not np.isfinite(o):
            continue
        o, h, l, c = float(o), float(h), float(l), float(c)
        if h < l:
            h, l = l, h

        # 갭하락: 전량 손절
        if o < sl:
            fill = o if rule.gap_fill == "OPEN" else (sl - sl_slip)
            exits.append((remaining, fill, "gap_down_sl"))
            remaining = 0.0
            n_exits += 1
            break

        sl_touched = (l <= sl)
        tp_touched_any = any(h >= tp for tp, filled in zip(tp_levels, tp_filled) if not filled)

        # 동시터치 + SL 우선
        if sl_touched and tp_touched_any and rule.tp_sl_same_bar_priority == "SL":
            fill = sl - sl_slip
            exits.append((remaining, fill, "same_bar_sl"))
            remaining = 0.0
            n_exits += 1
            break

        # TP 순차 청산
        for idx, (tp_val, pct) in enumerate(zip(tp_levels, scale_pcts)):
            if tp_filled[idx] or remaining <= 0:
                continue
            if h >= tp_val:
                sell_pct = min(remaining, pct)
                fill = _tp_fill(tp_val, o)
                exits.append((sell_pct, fill, f"tp{idx+1}@{tp_val:.0f}"))
                remaining -= sell_pct
                tp_filled[idx] = True
                n_exits += 1

        # SL 터치 (TP 이후 남은 물량)
        if remaining > 0 and sl_touched:
            if o == l == h:
                fill, tag = o, "limit_down_sl"
            else:
                fill, tag = sl - sl_slip, "intraday_sl"
            exits.append((remaining, fill, tag))
            remaining = 0.0
            n_exits += 1
            break

    # 타임아웃
    if remaining > 0 and bars:
        last_close = float(bars[min(max_hold_days - 1, len(bars) - 1)][3])
        if last_close > 0 and np.isfinite(last_close):
            exits.append((remaining, last_close, f"timeout_{max_hold_days}d"))
            n_exits += 1

    if not exits:
        return BarResult(action="none", reason="no_data")

    # 가중평균 수익률
    total_ret = 0.0
    vw_fill = 0.0
    for pct, fill, _ in exits:
        if pct <= 0:
            continue
        vw_fill += fill * (pct / 100.0)
        total_ret += ((fill / entry - 1.0) * 100.0) * (pct / 100.0)

    fee = _fee_pct_for_exits(rule, n_exits)
    net_ret = total_ret - fee

    has_tp = any("tp" in t for _, _, t in exits)
    reason = "scaleout:" + ",".join([f"{p:.0f}%@{f:.0f}({t})" for p, f, t in exits])
    return BarResult(
        action="tp_hit" if has_tp else "stop_hit",
        fill_price=vw_fill,
        return_pct=float(net_ret),
        reason=reason,
    )


# ═══════════════════════════════════════════════════
#  7. 룩어헤드 검증 유틸
# ═══════════════════════════════════════════════════

def _normalize_date(d) -> int:
    """날짜를 YYYYMMDD int로 정규화 (str/datetime/int 모두 지원)"""
    if isinstance(d, int):
        return d
    s = str(d).replace("-", "").replace("/", "").strip()[:8]
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def check_lookahead(
    plan_date,
    data_dates: List,
    feature_names: List[str] = None,
) -> List[str]:
    """
    추천일(plan_date) 이후의 데이터가 features에 포함되었는지 검사.
    날짜 포맷: str("YYYYMMDD"/"YYYY-MM-DD"), int(YYYYMMDD), datetime 모두 지원.
    Returns: 위반 항목 리스트 (비어있으면 정상)
    """
    plan_int = _normalize_date(plan_date)
    if plan_int == 0:
        return [f"invalid_plan_date: {plan_date}"]

    violations = []
    for d in data_dates:
        d_int = _normalize_date(d)
        if d_int > plan_int:
            violations.append(f"future_data: {d} > plan_date {plan_date}")
    return violations
