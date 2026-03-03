# -*- coding: utf-8 -*-
"""
trailing_stop.py — 🛡️ 트레일링 스탑 로직
═══════════════════════════════════════════════════
고점 대비 일정 비율 하락 시 수익 보존 강제 청산

[설계 원칙]
 1. 진입가 대비 일정 수익 구간 도달 후 트레일링 활성화
 2. 고점 대비 하락 비율에 따라 동적 청산
 3. ATR 기반 / 고정비율 / 스텝형 3가지 모드 지원
 4. 기존 stop_logic.py의 StopConfig와 호환

[통합 방법]
  stop_logic.py 하단에 import 또는, 독립 모듈로 사용:
    from trailing_stop import TrailingStopConfig, calc_trailing_stop, simulate_trailing_stop

[전략 샌드박스 연동]
  tab_backtest.py의 _run_backtest()에서 trailing_stop 파라미터 추가 시:
    from trailing_stop import simulate_trailing_stop
    result = simulate_trailing_stop(price_series, entry_price, config)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  설정
# ═══════════════════════════════════════════════════

@dataclass
class TrailingStopConfig:
    """트레일링 스탑 설정.

    Attributes:
        mode: "fixed" | "atr" | "step"
            - fixed: 고점 대비 고정 비율 하락 시 청산
            - atr: ATR 배수 기반 동적 트레일
            - step: 수익 구간별 차등 트레일 (계단식)
        activation_pct: 트레일링 활성화 최소 수익률(%)
            진입가 대비 이 수익률 이상이 되어야 트레일링 시작
        trail_pct: [fixed] 고점 대비 하락 비율(%) — 이 비율만큼 떨어지면 청산
        atr_mult: [atr] ATR 배수 — 고점에서 ATR×배수만큼 하락 시 청산
        step_levels: [step] 계단식 트레일 레벨
            예: [(5, 3), (10, 5), (20, 8)]
            → 수익 5%+ 시 고점 대비 3% 하락에 청산
            → 수익 10%+ 시 고점 대비 5% 하락에 청산
            → 수익 20%+ 시 고점 대비 8% 하락에 청산
        min_profit_lock_pct: 최소 확보 수익률(%)
            트레일링 활성화 후 이 수익률 아래로는 내려가지 않음
        use_tick_rounding: 호가 단위 반올림 적용 여부
    """
    mode: str = "fixed"
    activation_pct: float = 3.0
    trail_pct: float = 3.0
    atr_mult: float = 2.0
    step_levels: List[Tuple[float, float]] = field(
        default_factory=lambda: [(5, 3), (10, 5), (20, 8)]
    )
    min_profit_lock_pct: float = 1.0
    use_tick_rounding: bool = True


# ── 프리셋 ──

def config_conservative() -> TrailingStopConfig:
    """보수적 트레일링: 소폭 수익에서도 빠르게 보존."""
    return TrailingStopConfig(
        mode="fixed",
        activation_pct=2.0,
        trail_pct=2.0,
        min_profit_lock_pct=0.5,
    )


def config_aggressive() -> TrailingStopConfig:
    """공격적 트레일링: 큰 흐름을 잡기 위해 여유 제공."""
    return TrailingStopConfig(
        mode="fixed",
        activation_pct=5.0,
        trail_pct=5.0,
        min_profit_lock_pct=2.0,
    )


def config_atr_based() -> TrailingStopConfig:
    """ATR 기반: 변동성 적응형."""
    return TrailingStopConfig(
        mode="atr",
        activation_pct=3.0,
        atr_mult=2.0,
        min_profit_lock_pct=1.0,
    )


def config_step() -> TrailingStopConfig:
    """계단식: 수익 구간별 차등 보호."""
    return TrailingStopConfig(
        mode="step",
        activation_pct=3.0,
        step_levels=[(3, 2), (7, 4), (15, 7), (25, 10)],
        min_profit_lock_pct=1.0,
    )


# ═══════════════════════════════════════════════════
#  호가 유틸 (stop_logic.py의 것과 호환)
# ═══════════════════════════════════════════════════

def _tick_size(price: float) -> int:
    if price < 2000: return 1
    if price < 5000: return 5
    if price < 20000: return 10
    if price < 50000: return 50
    if price < 200000: return 100
    if price < 500000: return 500
    return 1000


def _floor_to_tick(price: float) -> int:
    if price <= 0: return 0
    p = int(math.floor(price))
    t = _tick_size(p)
    return p - (p % t)


# ═══════════════════════════════════════════════════
#  핵심 로직: 트레일링 스탑 가격 계산
# ═══════════════════════════════════════════════════

def calc_trailing_stop(
    entry_price: float,
    highest_price: float,
    current_price: float,
    cfg: TrailingStopConfig,
    atr_val: Optional[float] = None,
) -> Dict[str, Any]:
    """트레일링 스탑 상태 계산.

    Args:
        entry_price: 진입가
        highest_price: 진입 이후 최고가
        current_price: 현재가
        cfg: 트레일링 스탑 설정
        atr_val: ATR 값 (mode="atr" 시 필수)

    Returns:
        {
            "active": bool,         # 트레일링 활성화 여부
            "stop_price": float,    # 현재 트레일링 스탑 가격
            "triggered": bool,      # 청산 트리거 여부
            "profit_from_entry": float,  # 진입 대비 현재 수익률(%)
            "drawdown_from_peak": float, # 고점 대비 하락률(%)
            "trail_distance_pct": float, # 적용된 트레일 거리(%)
            "mode": str,
        }
    """
    result = {
        "active": False,
        "stop_price": 0.0,
        "triggered": False,
        "profit_from_entry": 0.0,
        "drawdown_from_peak": 0.0,
        "trail_distance_pct": 0.0,
        "mode": cfg.mode,
    }

    if entry_price <= 0 or highest_price <= 0 or current_price <= 0:
        return result

    # 진입 대비 최고 수익률
    peak_profit_pct = (highest_price - entry_price) / entry_price * 100
    # 현재 수익률
    current_profit_pct = (current_price - entry_price) / entry_price * 100
    # 고점 대비 하락률
    drawdown_pct = (highest_price - current_price) / highest_price * 100 if highest_price > 0 else 0

    result["profit_from_entry"] = round(current_profit_pct, 2)
    result["drawdown_from_peak"] = round(drawdown_pct, 2)

    # ── 활성화 체크 ──
    if peak_profit_pct < cfg.activation_pct:
        return result  # 아직 활성화 안 됨

    result["active"] = True

    # ── 트레일 거리 결정 ──
    trail_pct = cfg.trail_pct  # 기본값

    if cfg.mode == "fixed":
        trail_pct = cfg.trail_pct

    elif cfg.mode == "atr":
        if atr_val and atr_val > 0 and highest_price > 0:
            trail_pct = (atr_val * cfg.atr_mult) / highest_price * 100
            trail_pct = max(trail_pct, 1.0)  # 최소 1%
            trail_pct = min(trail_pct, 10.0)  # 최대 10%
        else:
            trail_pct = cfg.trail_pct  # ATR 없으면 기본값

    elif cfg.mode == "step":
        # 현재 최고 수익에 해당하는 레벨 찾기
        sorted_levels = sorted(cfg.step_levels, key=lambda x: x[0], reverse=True)
        trail_pct = cfg.trail_pct  # 기본
        for threshold, distance in sorted_levels:
            if peak_profit_pct >= threshold:
                trail_pct = distance
                break

    result["trail_distance_pct"] = round(trail_pct, 2)

    # ── 트레일링 스탑 가격 계산 ──
    stop_price = highest_price * (1 - trail_pct / 100)

    # 최소 확보 수익 적용
    min_lock_price = entry_price * (1 + cfg.min_profit_lock_pct / 100)
    stop_price = max(stop_price, min_lock_price)

    # 호가 단위 반올림
    if cfg.use_tick_rounding:
        stop_price = _floor_to_tick(stop_price)

    result["stop_price"] = stop_price

    # ── 트리거 체크 ──
    if current_price <= stop_price:
        result["triggered"] = True

    return result


# ═══════════════════════════════════════════════════
#  시뮬레이션: 가격 시리즈에 트레일링 적용
# ═══════════════════════════════════════════════════

def simulate_trailing_stop(
    price_series: pd.Series,
    entry_price: float,
    cfg: TrailingStopConfig,
    fixed_stop: Optional[float] = None,
    fixed_target: Optional[float] = None,
    atr_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """가격 시리즈에 트레일링 스탑을 적용하여 시뮬레이션.

    Args:
        price_series: 일별 종가 Series (index=날짜)
        entry_price: 진입가
        cfg: 트레일링 스탑 설정
        fixed_stop: (선택) 기존 고정 손절가 — 트레일링 전에 이 가격 도달 시 손절
        fixed_target: (선택) 고정 익절가 — 도달 시 즉시 청산
        atr_series: (선택) ATR 시리즈 (mode="atr" 시)

    Returns:
        {
            "exit_price": float,
            "exit_date": str,
            "exit_reason": str,  # "trailing_stop" | "fixed_stop" | "fixed_target" | "hold_exit"
            "return_pct": float,
            "hold_days": int,
            "peak_price": float,
            "peak_return_pct": float,
            "trail_history": [{"date": str, "price": float, "stop": float, "active": bool}],
        }
    """
    if price_series.empty or entry_price <= 0:
        return {"exit_price": entry_price, "exit_date": "", "exit_reason": "no_data",
                "return_pct": 0, "hold_days": 0, "peak_price": entry_price,
                "peak_return_pct": 0, "trail_history": []}

    highest = entry_price
    trail_history = []

    for i, (date, price) in enumerate(price_series.items()):
        price = float(price)
        if price <= 0:
            continue

        # 최고가 갱신
        if price > highest:
            highest = price

        # ATR 값 (있으면)
        atr_val = None
        if atr_series is not None and date in atr_series.index:
            atr_val = float(atr_series.loc[date])

        # 1. 고정 손절 체크
        if fixed_stop and price <= fixed_stop:
            return {
                "exit_price": fixed_stop,
                "exit_date": str(date),
                "exit_reason": "fixed_stop",
                "return_pct": round((fixed_stop - entry_price) / entry_price * 100, 2),
                "hold_days": i + 1,
                "peak_price": highest,
                "peak_return_pct": round((highest - entry_price) / entry_price * 100, 2),
                "trail_history": trail_history,
            }

        # 2. 고정 익절 체크
        if fixed_target and price >= fixed_target:
            return {
                "exit_price": fixed_target,
                "exit_date": str(date),
                "exit_reason": "fixed_target",
                "return_pct": round((fixed_target - entry_price) / entry_price * 100, 2),
                "hold_days": i + 1,
                "peak_price": highest,
                "peak_return_pct": round((highest - entry_price) / entry_price * 100, 2),
                "trail_history": trail_history,
            }

        # 3. 트레일링 스탑 체크
        ts = calc_trailing_stop(entry_price, highest, price, cfg, atr_val)

        trail_history.append({
            "date": str(date),
            "price": price,
            "stop": ts["stop_price"],
            "active": ts["active"],
            "highest": highest,
        })

        if ts["triggered"]:
            exit_price = ts["stop_price"]
            return {
                "exit_price": exit_price,
                "exit_date": str(date),
                "exit_reason": "trailing_stop",
                "return_pct": round((exit_price - entry_price) / entry_price * 100, 2),
                "hold_days": i + 1,
                "peak_price": highest,
                "peak_return_pct": round((highest - entry_price) / entry_price * 100, 2),
                "trail_history": trail_history,
            }

    # 보유기간 종료 — 마지막 가격에 청산
    final_price = float(price_series.iloc[-1])
    return {
        "exit_price": final_price,
        "exit_date": str(price_series.index[-1]),
        "exit_reason": "hold_exit",
        "return_pct": round((final_price - entry_price) / entry_price * 100, 2),
        "hold_days": len(price_series),
        "peak_price": highest,
        "peak_return_pct": round((highest - entry_price) / entry_price * 100, 2),
        "trail_history": trail_history,
    }


# ═══════════════════════════════════════════════════
#  백테스트 통합 함수
# ═══════════════════════════════════════════════════

def apply_trailing_to_backtest(
    raw_ret: float,
    entry_price: float = 10000,
    cfg: TrailingStopConfig = None,
    hold_days: int = 10,
    stop_pct: float = 5.0,
    target_pct: float = 10.0,
    peak_ret_override: Optional[float] = None,
) -> Tuple[float, str]:
    """기존 백테스트 로직에 트레일링 스탑을 오버레이하는 간이 함수.

    기존 tab_backtest.py의 _run_backtest() 내부에서 사용:
      기존:
        if raw_ret <= -stop_pct: applied_ret = -stop_pct; status = "STOP"
        elif raw_ret >= target_pct: applied_ret = target_pct; status = "WIN"
      
      변경:
        applied_ret, status = apply_trailing_to_backtest(
            raw_ret, cfg=trailing_cfg, stop_pct=stop_pct, target_pct=target_pct
        )

    Args:
        raw_ret: 원래 수익률(%)
        cfg: 트레일링 스탑 설정 (None이면 기존 로직)
        hold_days: 보유 기간
        stop_pct: 기존 고정 손절 비율(%)
        target_pct: 기존 고정 익절 비율(%)
        peak_ret_override: 실제 고점 수익률(%) — 고가 데이터가 있을 때 사용
            None이면 휴리스틱으로 추정

    Returns:
        (applied_ret, status)
    """
    if cfg is None:
        # 기존 로직 유지
        if raw_ret <= -stop_pct:
            return -stop_pct, "STOP"
        elif raw_ret >= target_pct:
            return target_pct, "WIN"
        return raw_ret, "HOLD_EXIT"

    # 트레일링 적용 (합성 가격 경로로 근사)
    # ────────────────────────────────────────
    # peak_ret_override가 있으면 실제 고가 데이터 사용
    # 없으면 일봉 종가 기반 휴리스틱 추정
    # ────────────────────────────────────────
    if peak_ret_override is not None:
        peak_ret = peak_ret_override
    elif raw_ret > 0:
        # 양수 수익: 고점은 최종수익 + 일중변동(1~3%) 추정
        intraday_buffer = min(3.0, max(1.0, raw_ret * 0.2))
        peak_ret = raw_ret + intraday_buffer
    else:
        # 음수 수익: 진입 초기 소폭 반등 후 하락 가정
        peak_ret = max(2.0, abs(raw_ret) * 0.3)

    # 고정 손절 먼저 체크
    if raw_ret <= -stop_pct:
        return -stop_pct, "STOP"

    # 트레일링 활성화 조건
    if peak_ret >= cfg.activation_pct:
        # 트레일 거리 결정
        if cfg.mode == "step":
            trail_dist = cfg.trail_pct
            for threshold, dist in sorted(cfg.step_levels, reverse=True):
                if peak_ret >= threshold:
                    trail_dist = dist
                    break
        else:
            trail_dist = cfg.trail_pct

        # 고점에서 trail_dist만큼 하락했는지 체크
        drawdown_from_peak = peak_ret - raw_ret
        if drawdown_from_peak >= trail_dist:
            # 트레일링 청산
            trail_exit_ret = max(peak_ret - trail_dist, cfg.min_profit_lock_pct)
            return round(trail_exit_ret, 2), "TRAIL_STOP"

    # 고정 익절 체크
    if raw_ret >= target_pct:
        return target_pct, "WIN"

    return raw_ret, "HOLD_EXIT"


# ═══════════════════════════════════════════════════
#  비교 분석: 기존 vs 트레일링
# ═══════════════════════════════════════════════════

def compare_strategies(
    returns: List[float],
    stop_pct: float = 5.0,
    target_pct: float = 10.0,
    trailing_cfg: TrailingStopConfig = None,
    cost_pct: float = 0.4,
) -> Dict[str, Dict[str, float]]:
    """기존 손절/익절 vs 트레일링 스탑 성과 비교.

    Returns:
        {
            "fixed": {"win_rate": ..., "avg_ret": ..., "total_ret": ..., ...},
            "trailing": {"win_rate": ..., "avg_ret": ..., "total_ret": ..., ...},
            "improvement": {"win_rate_diff": ..., "total_ret_diff": ..., ...},
        }
    """
    if trailing_cfg is None:
        trailing_cfg = TrailingStopConfig()

    fixed_results = []
    trail_results = []

    for raw_ret in returns:
        # 기존 로직
        f_ret, f_status = apply_trailing_to_backtest(
            raw_ret, stop_pct=stop_pct, target_pct=target_pct, cfg=None)
        fixed_results.append(f_ret - cost_pct)

        # 트레일링 로직
        t_ret, t_status = apply_trailing_to_backtest(
            raw_ret, stop_pct=stop_pct, target_pct=target_pct, cfg=trailing_cfg)
        trail_results.append(t_ret - cost_pct)

    def _stats(rets):
        arr = np.array(rets)
        wins = (arr > 0).sum()
        n = len(arr)
        eq = np.cumprod(1 + arr / 100)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        return {
            "win_rate": wins / n * 100 if n > 0 else 0,
            "avg_ret": arr.mean(),
            "total_ret": (eq[-1] - 1) * 100 if len(eq) > 0 else 0,
            "mdd": dd.min(),
            "profit_factor": abs(arr[arr > 0].mean() / arr[arr <= 0].mean()) if (arr <= 0).any() and (arr > 0).any() else 0,
            "trades": n,
        }

    fixed = _stats(fixed_results)
    trailing = _stats(trail_results)

    return {
        "fixed": fixed,
        "trailing": trailing,
        "improvement": {
            "win_rate_diff": trailing["win_rate"] - fixed["win_rate"],
            "total_ret_diff": trailing["total_ret"] - fixed["total_ret"],
            "mdd_diff": trailing["mdd"] - fixed["mdd"],
        },
    }
