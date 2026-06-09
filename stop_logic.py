# -*- coding: utf-8 -*-
"""
stop_logic.py v2.0 — 스마트 손절/익절/진입 필터 엔진
══════════════════════════════════════════════════════
collector.py의 analyze_ticker에서 사용하는 손절가, 진입 필터,
호가 단위 라운딩을 독립 모듈로 분리.

설계 원칙:
  1. ATR% 기반 가변 손절 — 변동성에 비례하되 퍼센트 캡으로 제한
  2. 시가총액별 최대 손실 차등 — 대형주 타이트, 소형주 여유
  3. 급등(갭업) 방어 — stop을 "올리는" 방향으로만 조정
  4. WORST_MDD -100% 방지 — 데이터 0값 컷 + 체결가 보수 처리
  5. 극단 갭(12%+)/VI 구간 진입 보류/분할 필터
  6. 호가 단위(tick) 유틸 통합 — 중복 제거
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any


from dataclasses import dataclass, field

from collector_config import DEFAULT_CONFIG as _CFG


# ═══════════════════════════════════════════════════
#  0. Config — 모든 파라미터를 한 곳에서 관리
# ═══════════════════════════════════════════════════

@dataclass
class StopConfig:
    """
    손절/익절/진입 필터 파라미터 — 레짐별 프리셋으로 전환 가능

    단위 규약 (입력 스키마):
      - buy, atr_val, today_low, swing_low_10: 가격 단위 (원)
      - mcap: 억원
      - tv_eok: 억원
      - gap_up_pct, dist_to_swing: %
      - slippage: %
    """
    # 레짐 이름 (trade_plan에서 참조)
    regime_name: str = "normal"

    # 의사결정 시점 모드
    mode: str = "pre_market"

    # 손절 관련
    atr_mult: float = 1.5
    max_stop_pct: float = 5.0
    hard_stop_floor_pct: float = 20.0   # [v24] 손절폭 하드캡: 이 이상으로 손절이 벌어지지 않게
    gap_atr_mult: float = 0.4
    gap_trigger_pct: float = 7.0

    # 시총별 최대 손실 캡 (억원 기준)
    mcap_large: float = 50000
    mcap_mid: float = 5000
    max_loss_large: float = 4.0
    max_loss_mid: float = 5.0
    max_loss_small: float = 6.0

    # 시총별 하한(휩쏘 방지)
    min_loss_large: float = 1.5
    min_loss_mid: float = 2.0
    min_loss_small: float = 3.0

    # R:R 배수 구간
    rr_low_atr: float = 2.0
    rr_mid_atr: float = 2.5
    rr_high_atr: float = 3.0

    # 수급 보정
    flow_buy_ratio: float = 0.10
    flow_sell_ratio: float = 0.15
    flow_buy_boost: float = 1.01
    flow_sell_tight: float = 1.005

    # 체결 슬리피지 — ⚠️ DEPRECATED: 체결비용은 ExecRule(bps)에서 관리.
    # 하위호환용으로 유지하되, 신규 코드에서는 ExecRule을 사용할 것.
    sl_slippage_pct: float = 0.3
    tp_slippage_pct: float = 0.05

    # 극단 진입 필터
    extreme_gap_pct: float = 12.0
    vi_trigger_pct: float = 10.0

    # 안전장치
    min_stop_pct_if_above_buy: float = 1.0

    # [v3.0] ATR-adaptive dynamic stop
    adaptive_stop: bool = True
    adaptive_atr_mult: float = 2.0       # ATR% * this = base stop
    adaptive_floor_pct: float = 3.0      # absolute minimum stop %
    adaptive_ceil_pct: float = 15.0      # absolute maximum stop %
    market_breadth: float = 50.0         # set by collector (0~100)
    use_tick_rounding: bool = True

    # [v4.0] 도달 확률 기반 목표가 파라미터
    tp_swing_lookback_short: int = 20    # 단기 고점 (20일)
    tp_swing_lookback_mid: int = 60      # 중기 고점 (60일)
    tp_swing_lookback_long: int = 120    # 장기 고점 (120일)
    tp_bb_period: int = 20               # BB 상단 기간
    tp_bb_std: float = 2.0               # BB 표준편차
    tp_atr_tp1_mult: float = 2.0         # ATR 기반 TP1 배수
    tp_atr_tp2_mult: float = 3.5         # ATR 기반 TP2 배수
    tp_fibo_levels: tuple = (1.0, 1.272, 1.618)  # 피보나치 확장 레벨
    tp_min_rr: float = 1.5              # 최소 RR — 이 미만이면 TP 상향
    tp_min_pct: float = 3.0             # 최소 목표 수익률 %


# ── 레짐별 프리셋 ──

def config_normal() -> StopConfig:
    """표준 시장 (VIX < 20, 코스피 변동성 보통)"""
    return StopConfig()

def config_high_vol() -> StopConfig:
    """고변동 레짐 (VIX > 25, 급등락 빈번)"""
    return StopConfig(
        regime_name="high_vol",
        max_stop_pct=6.0,
        gap_atr_mult=0.5,
        max_loss_large=5.0,
        max_loss_mid=6.0,
        max_loss_small=7.0,
        min_loss_large=2.0,
        min_loss_mid=2.5,
        min_loss_small=3.5,
        rr_low_atr=1.8,
        rr_mid_atr=2.2,
        rr_high_atr=2.5,
        sl_slippage_pct=0.5,
    )

def config_low_vol() -> StopConfig:
    """저변동 레짐 (VIX < 15, 횡보장)"""
    return StopConfig(
        regime_name="low_vol",
        max_stop_pct=4.0,
        gap_atr_mult=0.3,
        max_loss_large=3.5,
        max_loss_mid=4.5,
        max_loss_small=5.5,
        min_loss_large=1.0,
        min_loss_mid=1.5,
        min_loss_small=2.5,
        rr_low_atr=2.5,
        rr_mid_atr=3.0,
        rr_high_atr=3.5,
    )


# ── 활성 설정 (전역) ──
# collector에서 레짐 감지 후 switch_config()로 전환
_active_config = StopConfig()

def get_config() -> StopConfig:
    return _active_config

def switch_config(cfg: StopConfig) -> None:
    global _active_config
    _active_config = cfg

    # 하위 호환: 모듈 레벨 상수도 동시 갱신
    # Python의 from X import Y는 "값 복사"이므로,
    # 모듈 내부 상수를 참조하는 코드도 최신 cfg를 반영하게 함
    globals().update({
        "ATR_MULT": cfg.atr_mult,
        "MAX_STOP_PCT": cfg.max_stop_pct,
        "GAP_ATR_MULT": cfg.gap_atr_mult,
        "GAP_TRIGGER_PCT": cfg.gap_trigger_pct,
        "MCAP_LARGE": cfg.mcap_large,
        "MCAP_MID": cfg.mcap_mid,
        "MAX_LOSS_LARGE": cfg.max_loss_large,
        "MAX_LOSS_MID": cfg.max_loss_mid,
        "MAX_LOSS_SMALL": cfg.max_loss_small,
        "MIN_LOSS_LARGE": cfg.min_loss_large,
        "MIN_LOSS_MID": cfg.min_loss_mid,
        "MIN_LOSS_SMALL": cfg.min_loss_small,
        "RR_LOW_ATR": cfg.rr_low_atr,
        "RR_MID_ATR": cfg.rr_mid_atr,
        "RR_HIGH_ATR": cfg.rr_high_atr,
        "FLOW_BUY_RATIO": cfg.flow_buy_ratio,
        "FLOW_SELL_RATIO": cfg.flow_sell_ratio,
        "FLOW_BUY_BOOST": cfg.flow_buy_boost,
        "FLOW_SELL_TIGHT": cfg.flow_sell_tight,
        "DEFAULT_SLIPPAGE_PCT": cfg.sl_slippage_pct,
        "TP_SLIPPAGE_PCT": cfg.tp_slippage_pct,
        "EXTREME_GAP_PCT": cfg.extreme_gap_pct,
        "VI_TRIGGER_PCT": cfg.vi_trigger_pct,
        "MIN_STOP_PCT_IF_STOP_ABOVE_BUY": cfg.min_stop_pct_if_above_buy,
        "USE_TICK_ROUNDING": cfg.use_tick_rounding,
    })


# ── 하위 호환: 기존 상수명으로 접근 (모듈 레벨) ──
# 기존 코드에서 ATR_MULT 등을 직접 참조하는 부분이 있으므로 유지
ATR_MULT = _active_config.atr_mult
MAX_STOP_PCT = _active_config.max_stop_pct
GAP_ATR_MULT = _active_config.gap_atr_mult
GAP_TRIGGER_PCT = _active_config.gap_trigger_pct
MCAP_LARGE = _active_config.mcap_large
MCAP_MID = _active_config.mcap_mid
MAX_LOSS_LARGE = _active_config.max_loss_large
MAX_LOSS_MID = _active_config.max_loss_mid
MAX_LOSS_SMALL = _active_config.max_loss_small
MIN_LOSS_LARGE = _active_config.min_loss_large
MIN_LOSS_MID = _active_config.min_loss_mid
MIN_LOSS_SMALL = _active_config.min_loss_small
RR_LOW_ATR = _active_config.rr_low_atr
RR_MID_ATR = _active_config.rr_mid_atr
RR_HIGH_ATR = _active_config.rr_high_atr
FLOW_BUY_RATIO = _active_config.flow_buy_ratio
FLOW_SELL_RATIO = _active_config.flow_sell_ratio
FLOW_BUY_BOOST = _active_config.flow_buy_boost
FLOW_SELL_TIGHT = _active_config.flow_sell_tight
DEFAULT_SLIPPAGE_PCT = _active_config.sl_slippage_pct
TP_SLIPPAGE_PCT = _active_config.tp_slippage_pct
EXTREME_GAP_PCT = _active_config.extreme_gap_pct
VI_TRIGGER_PCT = _active_config.vi_trigger_pct
MIN_STOP_PCT_IF_STOP_ABOVE_BUY = _active_config.min_stop_pct_if_above_buy
USE_TICK_ROUNDING = _active_config.use_tick_rounding


# ═══════════════════════════════════════════════════
#  0-A. 입력 스키마 검증 — 단위/타입 강제
# ═══════════════════════════════════════════════════

def validate_stop_inputs(
    buy: float,
    atr_val: float,
    mcap: Optional[float] = None,
    tv_eok: Optional[float] = None,
) -> Dict[str, Any]:
    """
    calc_stop_price 입력값 검증 + 정규화

    단위 규약:
      buy, atr_val: 원 (양수)
      mcap: 억원 (양수 또는 None)
      tv_eok: 억원 (양수 또는 None)

    Returns:
        {"valid": bool, "warnings": List[str], "buy": float, "atr_val": float, ...}
    """
    warnings = []

    # buy 검증
    buy = float(buy) if buy is not None else 0.0
    if buy <= 0:
        return {"valid": False, "warnings": ["buy <= 0"], "buy": 0.0,
                "atr_val": 0.0, "mcap": None, "tv_eok": None}

    # atr 검증 (buy 대비 비율로 상식 체크)
    atr_val = float(atr_val) if atr_val is not None else 0.0
    if atr_val > 0:
        atr_pct = (atr_val / buy) * 100
        if atr_pct > 20:
            warnings.append(f"ATR%={atr_pct:.1f}% > 20% — 이상치 의심")
        if atr_pct < 0.1:
            warnings.append(f"ATR%={atr_pct:.2f}% < 0.1% — 데이터 오류 의심")

    # mcap 검증 (억원 단위 상식 체크)
    if mcap is not None:
        mcap = float(mcap)
        if mcap > 0 and mcap < 1:
            warnings.append(f"mcap={mcap:.2f}억 — 원 단위로 들어온 것 아닌지 확인")
        if mcap > 10000000:
            warnings.append(f"mcap={mcap:.0f}억 — 원 단위 혼입 의심(>1000조)")

    # tv_eok 검증
    if tv_eok is not None:
        tv_eok = float(tv_eok)
        if tv_eok < 0:
            tv_eok = 0.0
            warnings.append("tv_eok < 0 → 0으로 보정")

    return {
        "valid": True,
        "warnings": warnings,
        "buy": buy,
        "atr_val": atr_val,
        "mcap": mcap,
        "tv_eok": tv_eok,
    }


# ═══════════════════════════════════════════════════
#  0-B. 호가 단위(Tick) 유틸 — 통합 관리
# ═══════════════════════════════════════════════════

def tick_size(price: float) -> int:
    """한국 거래소 호가 단위"""
    if price < 2000: return 1
    if price < 5000: return 5
    if price < 20000: return 10
    if price < 50000: return 50
    if price < 200000: return 100
    if price < 500000: return 500
    return 1000

def round_to_tick(price: float) -> int:
    """가장 가까운 호가로 반올림"""
    if price is None or not np.isfinite(price) or price <= 0:
        return 0
    t = tick_size(float(price))
    return int(round(price / t) * t)

def floor_to_tick(price: float) -> int:
    """호가 단위로 내림 (손절가용)"""
    if price is None or not np.isfinite(price) or price <= 0:
        return 0
    t = tick_size(float(price))
    return int(math.floor(float(price) / t) * t)

def ceil_to_tick(price: float) -> int:
    """호가 단위로 올림 (목표가용)"""
    if price is None or not np.isfinite(price) or price <= 0:
        return 0
    t = tick_size(float(price))
    return int(math.ceil(float(price) / t) * t)

def floor_to_tick_by(price: float, tick: int) -> int:
    """지정 tick 단위로 내림"""
    if price is None or not np.isfinite(price) or tick <= 0:
        return 0
    return int(math.floor(float(price) / tick) * tick)

def ceil_to_tick_by(price: float, tick: int) -> int:
    """지정 tick 단위로 올림"""
    if price is None or not np.isfinite(price) or tick <= 0:
        return 0
    return int(math.ceil(float(price) / tick) * tick)

def round_to_tick_by(price: float, tick: int) -> int:
    """지정 tick 단위로 반올림"""
    if price is None or not np.isfinite(price) or tick <= 0:
        return 0
    return int(round(float(price) / tick) * tick)


# ═══════════════════════════════════════════════════
#  0-C. 극단 진입 필터 — 상한가/VI/갭 12%+ 방어
# ═══════════════════════════════════════════════════

def check_entry_filter(
    ret_1d: float,
    gap_pct: float,
    is_vi_triggered: bool = False,
    cfg: Optional['StopConfig'] = None,
) -> Dict[str, Any]:
    """
    극단 상황에서 진입을 보류/분할하는 필터 (cfg 기반 동적 참조)
    """
    c = cfg if cfg is not None else get_config()

    if gap_pct >= c.extreme_gap_pct or ret_1d >= 15.0:
        return {
            "action": "hold",
            "reason": f"극단갭({gap_pct:.1f}%)/급등({ret_1d:.1f}%) → 진입 보류",
            "position_pct": 0.0,
        }

    if is_vi_triggered or gap_pct >= c.gap_trigger_pct or ret_1d >= c.vi_trigger_pct:
        return {
            "action": "split",
            "reason": f"갭({gap_pct:.1f}%)/급등({ret_1d:.1f}%)/VI → 50% 분할",
            "position_pct": 50.0,
        }

    return {"action": "enter", "reason": "정상", "position_pct": 100.0}


# ═══════════════════════════════════════════════════
#  1. 핵심: 손절가 계산
# ═══════════════════════════════════════════════════

def calc_stop_price(
    buy: float,
    atr_val: float,
    mcap: Optional[float] = None,
    today_low: Optional[float] = None,
    gap_up_pct: Optional[float] = None,
    swing_low_10: Optional[float] = None,
    dist_to_swing: Optional[float] = None,
    tv_eok: Optional[float] = None,
    use_tick: bool = True,
    cfg: Optional[Any] = None,
) -> Tuple[float, float, float, str]:
    """
    ATR% 기반 가변 손절가 계산

    Args:
        cfg: StopConfig 인스턴스. None이면 전역 _active_config 사용.
             cfg.mode == "pre_market" → today_low/gap_up_pct 무시 (look-ahead 차단)
             cfg.mode == "intraday"   → 장중 데이터 사용 OK

    Returns:
        (stop_price, actual_stop_pct, max_loss_pct, stop_reason)
    """
    c = cfg if cfg is not None else get_config()
    stop_reason = "NORMAL"
    reason_tags = []  # 사유 태그 누적

    # ── (0) 입력 스키마 검증 ──
    v = validate_stop_inputs(buy, atr_val, mcap, tv_eok)
    if not v["valid"]:
        return 0.0, 5.0, 5.0, "INVALID"
    buy = v["buy"]
    atr_val = v["atr_val"]
    mcap = v["mcap"]
    tv_eok = v["tv_eok"]

    # (2) warnings 있으면 reason에 반영
    if v["warnings"]:
        reason_tags.append("WARN:" + "|".join(v["warnings"][:2]))  # 최대 2개

    # ── look-ahead 차단 ──
    if c.mode == "pre_market":
        if today_low is not None or gap_up_pct is not None:
            reason_tags.append("PM_BLOCK_INTRADAY")
        today_low = None
        gap_up_pct = None

    if atr_val <= 0:
        atr_pct = 3.0  # fallback
    else:
        atr_pct = (atr_val / buy) * 100.0

    if c.adaptive_stop:
        # [v3.0] ATR-adaptive: stop width scales with actual volatility
        base_stop_pct = atr_pct * c.adaptive_atr_mult
        # Market regime scaling: wider stops in weak markets
        if c.market_breadth < 25:
            base_stop_pct *= 1.4   # panic/crash: 40% wider
        elif c.market_breadth < 40:
            base_stop_pct *= 1.2   # weak: 20% wider
        # Clamp to floor/ceiling
        base_stop_pct = max(c.adaptive_floor_pct, min(base_stop_pct, c.adaptive_ceil_pct))
    else:
        base_stop_pct = min(atr_pct * c.atr_mult, c.max_stop_pct)

    # ── (1) 시총별 최대 손실 제한 ──
    effective_mcap = mcap
    mcap_estimated = False  # (3) 추정 여부 추적
    if effective_mcap is None or effective_mcap <= 0:
        if tv_eok is not None and tv_eok > 0:
            mcap_estimated = True  # tv_eok 기반 추정
            if tv_eok >= 1000:
                effective_mcap = 100000
            elif tv_eok >= 500:
                effective_mcap = 60000
            elif tv_eok >= 200:
                effective_mcap = 30000
            elif tv_eok >= 50:
                effective_mcap = 10000
            elif tv_eok >= 15:
                effective_mcap = 5000
            else:
                effective_mcap = 2000

    if effective_mcap is None or effective_mcap <= 0:
        max_loss_pct = c.max_loss_mid
    elif effective_mcap > c.mcap_large:
        max_loss_pct = c.max_loss_large
    elif effective_mcap > c.mcap_mid:
        max_loss_pct = c.max_loss_mid
    else:
        max_loss_pct = c.max_loss_small

    # ── (2) 최종 손절폭: ATR 기반 vs 시총 기반 중 타이트한 쪽 ──
    if c.adaptive_stop:
        # [v3.0] adaptive: ATR already determines width, skip static cap
        # but still use mcap-based max as a soft reference for R:R
        stop_pct = base_stop_pct
    else:
        stop_pct = min(base_stop_pct, max_loss_pct)
    stop = buy * (1.0 - stop_pct / 100.0)

    # ── (3) 급등(갭업) 방어 ──
    was_tightened = False

    if (gap_up_pct is not None and gap_up_pct >= c.gap_trigger_pct
            and today_low is not None and today_low > 0
            and atr_val > 0):
        gap_stop = float(today_low) - c.gap_atr_mult * atr_val

        cap_stop = buy * (1.0 - max_loss_pct / 100.0)
        gap_stop = max(gap_stop, cap_stop)

        if gap_stop >= buy:
            gap_stop = buy * (1.0 - c.min_stop_pct_if_above_buy / 100.0)

        if gap_stop > stop:
            stop = gap_stop
            was_tightened = True
            stop_reason = "GAP"

    # ── (4) 구조적 지지선(Swing Low) 보정 ──
    # ⑤ dist_to_swing: abs + finite 체크 강제
    if (swing_low_10 is not None and swing_low_10 > 0
            and dist_to_swing is not None
            and np.isfinite(float(dist_to_swing))):
        dist_abs = abs(float(dist_to_swing))
        if dist_abs < 8.0:
            swing_stop = float(swing_low_10) * 0.97

            if swing_stop >= buy:
                swing_stop = buy * (1.0 - c.min_stop_pct_if_above_buy / 100.0)

            if swing_stop > stop:
                stop = swing_stop
                was_tightened = True
                stop_reason = "GAP+SWING" if "GAP" in stop_reason else "SWING"

    # ── (5) 최종 안전장치: 하한(휩쏘 방지) ──
    if not was_tightened:
        if effective_mcap is not None and effective_mcap > c.mcap_large:
            min_loss_pct = c.min_loss_large
        elif effective_mcap is not None and effective_mcap > c.mcap_mid:
            min_loss_pct = c.min_loss_mid
        else:
            min_loss_pct = c.min_loss_small

        if atr_val > 0:
            atr_pct_check = (atr_val / buy) * 100.0
            if atr_pct_check < 1.5:
                min_loss_pct = min(min_loss_pct, 1.5)

        min_stop = buy * (1.0 - min_loss_pct / 100.0)
        if stop >= min_stop:
            stop = min_stop

    # ── (6) 공통: stop >= buy 절대 방지 ──
    if stop >= buy:
        stop = buy * (1.0 - c.min_stop_pct_if_above_buy / 100.0)

    # tick 적용
    if use_tick and c.use_tick_rounding:
        stop = float(floor_to_tick(stop))

    actual_stop_pct = (1.0 - stop / buy) * 100.0 if buy > 0 else stop_pct

    # ── stop_reason 최종 조합 ──
    # (3) tv_eok 기반 추정 시 EST_MCAP 태그
    if mcap_estimated:
        reason_tags.append("EST_MCAP")

    # reason_tags가 있으면 stop_reason에 합침
    if reason_tags:
        stop_reason = stop_reason + "+" + "+".join(reason_tags)

    return float(stop), float(actual_stop_pct), float(max_loss_pct), stop_reason


# ═══════════════════════════════════════════════════
#  2. R:R 배수 계산 (기존 v8.5 로직 개선)
# ═══════════════════════════════════════════════════

def calc_rr_multiplier(atr_val: float, buy: float, cfg: Optional['StopConfig'] = None) -> float:
    """ATR% 기반 가변 R:R 배수 (cfg 기반 동적 참조)"""
    c = cfg if cfg is not None else get_config()
    if buy <= 0:
        return c.rr_mid_atr
    atr_pct = (atr_val / buy) * 100.0
    if atr_pct < 2.0:
        return c.rr_low_atr
    elif atr_pct < 4.0:
        return c.rr_mid_atr
    else:
        return c.rr_high_atr


# ═══════════════════════════════════════════════════
#  3. 데이터 정합성 — WORST_MDD -100% 방지
# ═══════════════════════════════════════════════════

def sanitize_ohlcv(ohlcv: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    OHLCV 데이터에서 비정상 행 제거 (상폐, 0원, 이상치)

    이걸 빠뜨리면 손절가를 아무리 잘 잡아도 MDD -100%가 찍힘:
    - Close=0, Low=0인 행 → 수익률 계산 시 분모 0 → -100%
    - 거래정지 등으로 가격이 0인 행

    analyze_ticker 진입부에서 호출 권장:
        ohlcv = sanitize_ohlcv(ohlcv)

    Returns:
        정제된 DataFrame (제거 건수는 .attrs["sanitize_removed"]에 저장)
    """
    if ohlcv.empty:
        return ohlcv

    df = ohlcv.copy()
    n_before = len(df)

    # 한글/영문 컬럼 동시 지원 — 둘 다 있으면 둘 다 체크
    col_pairs = [
        ("종가", "Close"), ("저가", "Low"), ("시가", "Open"), ("고가", "High")
    ]

    def _positive_mask(col_name):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors='coerce').fillna(0) > 0
        return pd.Series(True, index=df.index)

    mask = pd.Series(True, index=df.index)
    for kor, eng in col_pairs:
        mask &= _positive_mask(kor) & _positive_mask(eng)

    df = df[mask]

    n_removed = n_before - len(df)
    df.attrs["sanitize_removed"] = n_removed

    if verbose and n_removed > 0:
        print(f"  ⚠️ [sanitize] {n_removed}행 제거됨 "
              f"({n_before}→{len(df)}, 제거율 {n_removed/max(n_before,1)*100:.1f}%)")

    return df


def conservative_exit_price(
    stop: float,
    today_open: float,
    today_low: float,
    today_high: float = 0.0,
    slippage_pct: float = 0.3,
) -> float:
    """
    손절 체결가 보수 처리 (백테스트 현실성 확보)

    기존 문제:
    - "Low가 stop 아래면 stop에서 체결" → 갭하락 시 비현실적으로 좋은 가격
    - MDD가 과소 추정되어 실전과 괴리

    보수 체결 규칙:
    1. 갭하락(시가 < stop): 시가에서 체결 (이미 stop을 뚫고 시작)
    2. 장중 하락(시가 >= stop, Low <= stop): stop - 슬리피지에서 체결
    3. 하한가 의심(시가=저가=고가, 변동없음): 시가에서 체결 (체결 어려움 반영)
    4. stop 미도달: 0.0 리턴 (체결 없음)

    Args:
        stop: 손절가
        today_open: 당일 시가
        today_low: 당일 저가
        today_high: 당일 고가 (하한가 감지용)
        slippage_pct: 슬리피지 비율 (기본 0.3%)

    Returns:
        체결가 (0.0이면 미체결)
    """
    if today_open <= 0 or stop <= 0:
        return 0.0

    # 하한가 의심: 시가=저가=고가 (변동 없음, 체결 어려움)
    if today_high > 0 and today_open == today_low == today_high:
        # 하한가면 실제로 매도 체결이 안 될 수 있음
        # → 보수적으로 시가(=하한가)에서 체결 가정
        if today_low <= stop:
            return today_open
        return 0.0

    if today_open < stop:
        # 갭하락: 시가가 이미 stop 아래 → 시가에서 체결
        return today_open
    elif today_low <= stop:
        # 장중 하락: stop 부근에서 체결 + 슬리피지 반영
        slip = stop * (slippage_pct / 100.0)
        return stop - slip
    else:
        # stop 미도달
        return 0.0


def conservative_tp_price(
    target: float,
    today_open: float,
    today_high: float,
    slippage_pct: float = 0.05,
) -> float:
    """
    보수 익절 체결가 (백테스트 현실성 확보)

    규칙:
    1. 갭상승(시가 > 목표가): 시가에서 체결 (리밋이면 더 좋은 가격)
    2. 장중 도달(고가 >= 목표가): 목표가 - 슬리피지에서 체결
    3. 미도달: 0.0 리턴

    Returns:
        체결가 (0.0이면 미체결)
    """
    if target <= 0 or today_open <= 0:
        return 0.0
    if today_open > target:
        return float(today_open)
    if today_high >= target:
        slip = float(target) * (slippage_pct / 100.0)
        return float(target) - slip
    return 0.0


# ═══════════════════════════════════════════════════
#  4. 수급 보정 (기존 v8.5 로직 이식)
# ═══════════════════════════════════════════════════

def adjust_by_flow(
    buy: float,
    stop: float,
    last_c: float,
    major_net: float,
    major_ratio: float,
    cfg: Optional['StopConfig'] = None,
) -> Tuple[float, float]:
    """메이저(외인+기관) 순매수 기반 매수가/손절가 보정 (cfg 기반 동적 참조)"""
    c = cfg if cfg is not None else get_config()

    if major_net > 0 and major_ratio >= c.flow_buy_ratio:
        buy = min(buy * c.flow_buy_boost, last_c)
    elif major_net < 0 and major_ratio >= c.flow_sell_ratio:
        stop = max(stop, stop * c.flow_sell_tight)

    return buy, stop


# ═══════════════════════════════════════════════════
#  4-A. 도달 확률 기반 목표가 엔진 (v4.0)
# ═══════════════════════════════════════════════════
#
# 설계 철학:
#   기존: stop_loss × RR = target (기계적, 도달 근거 없음)
#   개선: 실제 기술적 저항 레벨에서 목표가 후보를 산출하고,
#         도달 확률이 높은 순으로 TP1 > TP2 > TP3 배정
#
# 후보 산출 메서드 5가지:
#   ① Swing High (20/60/120일) — 가장 현실적, 과거에 실제로 도달한 가격
#   ② Bollinger Band 상단 — 단기 과매수 경계, 도달 확률 높음
#   ③ ATR 배수 — 변동성 비례, RR 최소 보장용 안전망
#   ④ Fibonacci Extension — 추세 연장 시 예상 도달점
#   ⑤ Volume Profile 저항 — 매물대 상단 (데이터 있을 때)
#
# 최종 선택 로직:
#   - TP1: 도달 확률 70%+ (보수적, "거의 확실히 닿는 곳")
#   - TP2: 도달 확률 45~70% (적극적, "추세가 이어지면 닿는 곳")
#   - TP3: 도달 확률 25~45% (공격적, "풀 스윙 시 가능한 곳")
#   - 모든 TP는 최소 RR 1.5배 이상 보장
# ═══════════════════════════════════════════════════

def _atr_from_ohlcv(ohlcv: pd.DataFrame, period: int = 14) -> float:
    """OHLCV에서 ATR 직접 계산"""
    if len(ohlcv) < period + 1:
        return 0.0
    h = ohlcv['고가']
    l = ohlcv['저가']
    c = ohlcv['종가'].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return float(tr.tail(period).mean())


def _tp_swing_high(ohlcv: pd.DataFrame, entry: float,
                   cfg: StopConfig) -> list:
    """
    ① Swing High — 과거 N일 최고가

    왜 좋은가: 시장이 실제로 도달했던 가격이므로 도달 확률이 높음.
    20일 고점은 "최근에 찍었으니 다시 갈 수 있다", 120일은 더 공격적.
    """
    results = []
    h = ohlcv['고가']

    for lookback, label, base_conf in [
        (cfg.tp_swing_lookback_short, "SWING_20D", 80),
        (cfg.tp_swing_lookback_mid, "SWING_60D", 60),
        (cfg.tp_swing_lookback_long, "SWING_120D", 45),
    ]:
        if len(h) < lookback:
            continue

        high = float(h.tail(lookback).max())

        # 현재가보다 3% 이상 위에 있어야 의미 있는 목표
        if high <= entry * 1.03:
            continue

        pct = (high / entry - 1) * 100
        results.append({
            "price": high,
            "pct": round(pct, 2),
            "method": label,
            "confidence": base_conf,
        })

    return results


def _tp_bb_upper(ohlcv: pd.DataFrame, entry: float,
                 cfg: StopConfig) -> list:
    """
    ② Bollinger Band 상단 — 단기 과매수 영역

    왜 좋은가: 통계적으로 종가의 95%가 BB 내에 있으므로,
    BB 상단 도달 시 되돌림 확률이 높음 → "여기서 팔면 고점 근처"
    """
    c = ohlcv['종가']
    if len(c) < cfg.tp_bb_period:
        return []

    ma = c.rolling(cfg.tp_bb_period).mean()
    std = c.rolling(cfg.tp_bb_period).std()
    bb_upper = float((ma + cfg.tp_bb_std * std).iloc[-1])

    if bb_upper <= entry * 1.02:
        return []

    pct = (bb_upper / entry - 1) * 100
    return [{
        "price": bb_upper,
        "pct": round(pct, 2),
        "method": "BB_UPPER",
        "confidence": 75,  # BB 내 회귀 확률이 높으므로 도달 확률도 높음
    }]


def _tp_atr_based(ohlcv: pd.DataFrame, entry: float,
                  cfg: StopConfig) -> list:
    """
    ③ ATR 배수 — 변동성 비례 목표

    왜 있는가: 다른 메서드가 후보를 못 만들 때 안전망.
    ATR 2배는 "정상 변동 범위 상단", 3.5배는 "강한 추세 시 가능".
    """
    atr = _atr_from_ohlcv(ohlcv, 14)
    if atr <= 0:
        return []

    results = []
    for mult, label, conf in [
        (cfg.tp_atr_tp1_mult, "ATR_2x", 70),
        (cfg.tp_atr_tp2_mult, "ATR_3.5x", 45),
    ]:
        tp = entry + atr * mult
        if tp <= entry * 1.02:
            continue
        pct = (tp / entry - 1) * 100
        results.append({
            "price": tp,
            "pct": round(pct, 2),
            "method": label,
            "confidence": conf,
        })
    return results


def _tp_fibonacci_ext(ohlcv: pd.DataFrame, entry: float,
                      cfg: StopConfig) -> list:
    """
    ④ Fibonacci Extension — 직전 스윙 기반 확장

    로직: 최근 60일 내 저점→고점 스윙을 찾고,
    고점에서 피보나치 확장(1.0, 1.272, 1.618)을 적용.
    "추세가 이어지면 어디까지 갈 수 있나"를 보여줌.
    """
    if len(ohlcv) < 20:
        return []

    tail = ohlcv.tail(60)
    swing_low = float(tail['저가'].min())
    swing_high = float(tail['고가'].max())
    swing_range = swing_high - swing_low

    if swing_range <= 0 or swing_low <= 0:
        return []

    # 저점이 고점보다 먼저 나와야 상승 스윙
    low_idx = tail['저가'].idxmin()
    high_idx = tail['고가'].idxmax()
    if low_idx >= high_idx:
        return []  # 하락 스윙이면 피보 확장 무의미

    results = []
    for level, conf in [(1.0, 55), (1.272, 40), (1.618, 25)]:
        tp = swing_high + swing_range * (level - 1.0)
        if tp <= entry * 1.03:
            continue
        pct = (tp / entry - 1) * 100
        results.append({
            "price": tp,
            "pct": round(pct, 2),
            "method": f"FIBO_{level}",
            "confidence": conf,
        })
    return results


def _tp_volume_resistance(entry: float,
                          poc_p: float = 0.0,
                          res_ratio: float = 0.0,
                          res_ratio_near: float = 0.0) -> list:
    """
    ⑤ Volume Profile 저항 — 매물대 상단

    ticker_analyzer에서 이미 계산된 POC/RES_RATIO를 활용.
    매물대 저항이 강하면 그 레벨이 자연스러운 목표가.
    """
    if poc_p <= 0 or poc_p <= entry * 1.02:
        return []

    # POC 위에 있고 저항이 강하면, POC 자체가 저항으로 작용할 수 있음
    # 하지만 이미 POC 위에 있으면 다음 저항을 찾아야 함
    # → 이 함수는 POC가 위에 있을 때만 의미
    pct = (poc_p / entry - 1) * 100
    conf = 65 if res_ratio_near > 0.10 else 50

    return [{
        "price": poc_p,
        "pct": round(pct, 2),
        "method": "VP_RESISTANCE",
        "confidence": conf,
    }]


def compute_realistic_targets(
    ohlcv: pd.DataFrame,
    entry: float,
    stop: float,
    cfg: Optional[StopConfig] = None,
    poc_p: float = 0.0,
    res_ratio: float = 0.0,
    res_ratio_near: float = 0.0,
    use_tick: bool = True,
) -> Dict[str, Any]:
    """
    도달 확률 기반 목표가 산출 (메인 함수)

    Args:
        ohlcv: OHLCV DataFrame (한글 컬럼: 종가/고가/저가)
        entry: 진입가 (원)
        stop: 손절가 (원)
        cfg: StopConfig (None이면 전역)
        poc_p: POC 가격 (Volume Profile, 0이면 미사용)
        res_ratio: 매물대 저항 비율
        res_ratio_near: 근접 매물대 비율
        use_tick: 호가 단위 반올림

    Returns:
        {
            "TP1": int,        # 1차 목표 (보수적, 도달 확률 높음)
            "TP1_PCT": float,  # 1차 목표 수익률 %
            "TP1_METHOD": str, # 산출 근거
            "TP1_PROB": int,   # 추정 도달 확률 %
            "TP2": int,        # 2차 목표 (적극적)
            "TP2_PCT": float,
            "TP2_METHOD": str,
            "TP2_PROB": int,
            "TP3": int,        # 3차 목표 (공격적)
            ... (없을 수 있음)
            "MIN_RR": float,   # TP1 기준 최소 RR
            "N_CANDIDATES": int,
            "CANDIDATES": list, # 전체 후보 리스트 (디버깅용)
        }
    """
    c = cfg if cfg is not None else get_config()

    if entry <= 0 or stop <= 0 or ohlcv is None or ohlcv.empty:
        return _fallback_targets(entry, stop, c, use_tick)

    stop_pct = (1 - stop / entry) * 100
    if stop_pct <= 0:
        stop_pct = 3.0  # 안전값

    # ── 모든 메서드에서 후보 수집 ──
    candidates = []
    candidates.extend(_tp_swing_high(ohlcv, entry, c))
    candidates.extend(_tp_bb_upper(ohlcv, entry, c))
    candidates.extend(_tp_atr_based(ohlcv, entry, c))
    candidates.extend(_tp_fibonacci_ext(ohlcv, entry, c))
    candidates.extend(_tp_volume_resistance(entry, poc_p, res_ratio, res_ratio_near))

    if not candidates:
        return _fallback_targets(entry, stop, c, use_tick)

    # ── 중복 제거: 가격 차이 1% 이내면 confidence 높은 쪽만 유지 ──
    candidates.sort(key=lambda x: x["price"])
    deduped = []
    for cand in candidates:
        merged = False
        for prev in deduped:
            if abs(cand["price"] - prev["price"]) / prev["price"] < 0.01:
                # 같은 레벨 → 더 높은 confidence 유지
                if cand["confidence"] > prev["confidence"]:
                    prev.update(cand)
                merged = True
                break
        if not merged:
            deduped.append(cand)

    # ── RR 필터: 최소 RR 미달 후보 제거 ──
    risk = entry - stop
    if risk > 0:
        deduped = [cd for cd in deduped
                   if (cd["price"] - entry) / risk >= c.tp_min_rr
                   or cd["confidence"] >= 75]  # 고확률은 RR 낮아도 유지

    if not deduped:
        return _fallback_targets(entry, stop, c, use_tick)

    # ── 최소 수익률 필터 ──
    deduped = [cd for cd in deduped if cd["pct"] >= c.tp_min_pct]
    if not deduped:
        return _fallback_targets(entry, stop, c, use_tick)

    # ── TP 배정: confidence 순 정렬 후 3단계 배정 ──
    deduped.sort(key=lambda x: (-x["confidence"], x["price"]))

    result = {"N_CANDIDATES": len(deduped), "CANDIDATES": deduped}

    # TP1: confidence 가장 높은 것 (보수적, 도달 확률 최고)
    tp1 = deduped[0]
    tp1_price = ceil_to_tick(tp1["price"]) if use_tick else round(tp1["price"], 0)
    tp1_rr = (tp1_price - entry) / risk if risk > 0 else 0
    result.update({
        "TP1": int(tp1_price),
        "TP1_PCT": round((tp1_price / entry - 1) * 100, 1),
        "TP1_METHOD": tp1["method"],
        "TP1_PROB": tp1["confidence"],
        "MIN_RR": round(tp1_rr, 1),
    })

    # TP2: TP1보다 높으면서 confidence 45+ 중 가장 좋은 것
    tp2_candidates = [cd for cd in deduped
                      if cd["price"] > tp1["price"] * 1.02
                      and cd["confidence"] >= 25]  # FIBO도 TP2 후보로 허용
    if tp2_candidates:
        tp2 = tp2_candidates[0]
        tp2_price = ceil_to_tick(tp2["price"]) if use_tick else round(tp2["price"], 0)
        result.update({
            "TP2": int(tp2_price),
            "TP2_PCT": round((tp2_price / entry - 1) * 100, 1),
            "TP2_METHOD": tp2["method"],
            "TP2_PROB": tp2["confidence"],
        })

        # TP3: TP2보다 더 높은 공격적 목표
        tp3_candidates = [cd for cd in deduped
                          if cd["price"] > tp2["price"] * 1.02
                          and cd["confidence"] >= 20]
        if tp3_candidates:
            tp3 = tp3_candidates[0]
            tp3_price = ceil_to_tick(tp3["price"]) if use_tick else round(tp3["price"], 0)
            result.update({
                "TP3": int(tp3_price),
                "TP3_PCT": round((tp3_price / entry - 1) * 100, 1),
                "TP3_METHOD": tp3["method"],
                "TP3_PROB": tp3["confidence"],
            })

    return result


def _fallback_targets(entry: float, stop: float,
                      cfg: StopConfig, use_tick: bool) -> Dict[str, Any]:
    """
    후보가 없을 때 기존 RR 배수 방식으로 폴백

    이 경우에도 최소 RR 1.5배는 보장.
    """
    risk = entry - stop
    if risk <= 0:
        risk = entry * 0.03  # 3% 기본

    rr = max(cfg.tp_min_rr, 2.0)
    tp1 = entry + risk * rr
    tp2 = entry + risk * (rr + 1.0)

    tp1_price = ceil_to_tick(tp1) if use_tick else round(tp1, 0)
    tp2_price = ceil_to_tick(tp2) if use_tick else round(tp2, 0)

    return {
        "TP1": int(tp1_price),
        "TP1_PCT": round((tp1_price / entry - 1) * 100, 1),
        "TP1_METHOD": "RR_FALLBACK",
        "TP1_PROB": 50,
        "TP2": int(tp2_price),
        "TP2_PCT": round((tp2_price / entry - 1) * 100, 1),
        "TP2_METHOD": "RR_FALLBACK",
        "TP2_PROB": 35,
        "MIN_RR": round(rr, 1),
        "N_CANDIDATES": 0,
        "CANDIDATES": [],
    }


def format_targets(targets: Dict[str, Any], entry: float, stop: float) -> str:
    """목표가 결과를 사람이 읽기 좋은 문자열로 포맷"""
    lines = ["🎯 도달 확률 기반 목표가"]
    lines.append(f"   진입: {int(entry):,}원 → 손절: {int(stop):,}원 "
                 f"({(1-stop/entry)*100:.1f}%)")
    lines.append("")

    for i in range(1, 4):
        key = f"TP{i}"
        if key in targets:
            prob = targets.get(f"{key}_PROB", "?")
            method = targets.get(f"{key}_METHOD", "?")
            pct = targets.get(f"{key}_PCT", 0)
            lines.append(
                f"   TP{i}: {targets[key]:,}원 (+{pct:.1f}%) "
                f"| 도달확률 {prob}% | {method}"
            )

    rr = targets.get("MIN_RR", 0)
    n = targets.get("N_CANDIDATES", 0)
    lines.append(f"\n   최소 RR: {rr:.1f}배 | 후보 {n}개 분석")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════
#  5. 성과지표 — 손절 설계 목표 KPI
# ═══════════════════════════════════════════════════

def calc_risk_kpi(returns: np.ndarray) -> Dict[str, Any]:
    """
    손절 리팩토링 효과를 측정하는 핵심 KPI 5종

    기존 문제: 단순 승률/평균수익률만 보면 손절 개선 효과가 안 보임.
    개선 후 봐야 할 지표:
      1. WORST_MDD: 가장 큰 단일 손실 (목표: -6% 이내)
      2. 95% 손실꼬리(VaR): 하위 5% 손실 평균 (목표: -4% 이내)
      3. 손익비(실현): 평균이익/평균손실 (목표: 1.5+)
      4. 연속손실 길이: 최대 연속 손실 횟수 (목표: 5 이하)
      5. 승률: 참고용

    Args:
        returns: 각 거래의 수익률 배열 (%, 예: [-3.2, 5.1, -1.5, ...])

    Returns:
        dict with keys: worst_mdd, var_95, profit_loss_ratio,
                        max_consecutive_loss, win_rate, n_trades
    """
    if returns is None or len(returns) == 0:
        return {
            "worst_mdd": 0.0, "var_95": 0.0, "profit_loss_ratio": 0.0,
            "max_consecutive_loss": 0, "win_rate": 0.0, "n_trades": 0,
        }

    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {
            "worst_mdd": 0.0, "var_95": 0.0, "profit_loss_ratio": 0.0,
            "max_consecutive_loss": 0, "win_rate": 0.0, "n_trades": 0,
        }

    # 1. WORST_MDD: 단일 최대 손실
    worst_mdd = float(r.min())

    # 2. 95% VaR (Conditional): 하위 5% 손실의 평균
    cutoff = int(max(1, len(r) * 0.05))
    sorted_r = np.sort(r)
    var_95 = float(sorted_r[:cutoff].mean())

    # 3. 손익비 (실현)
    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 1.0  # 0 방지
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # 4. 최대 연속 손실 길이
    max_consec = 0
    current = 0
    for ret in r:
        if ret < 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    # 5. 승률
    win_rate = float((r > 0).sum() / len(r)) * 100.0

    return {
        "worst_mdd": round(worst_mdd, 2),
        "var_95": round(var_95, 2),
        "profit_loss_ratio": round(profit_loss_ratio, 2),
        "max_consecutive_loss": max_consec,
        "win_rate": round(win_rate, 1),
        "n_trades": len(r),
    }


def format_risk_kpi(kpi: Dict[str, Any]) -> str:
    """KPI dict를 사람이 읽기 좋은 문자열로 포맷"""
    return (
        f"📊 리스크 KPI ({kpi['n_trades']}건)\n"
        f"   WORST_MDD: {kpi['worst_mdd']:+.2f}%\n"
        f"   95% VaR:   {kpi['var_95']:+.2f}%\n"
        f"   손익비:     {kpi['profit_loss_ratio']:.2f}\n"
        f"   연속손실:   {kpi['max_consecutive_loss']}회\n"
        f"   승률:       {kpi['win_rate']:.1f}%"
    )


# ═══════════════════════════════════════════════════
#  6. 워크포워드(Rolling) 기간별 KPI 검증
# ═══════════════════════════════════════════════════

def rolling_risk_kpi(
    dates: np.ndarray,
    returns: np.ndarray,
    window_days: int = 20,
) -> pd.DataFrame:
    """
    기간별 리스크 KPI를 롤링으로 계산하여 "시간에 따른 안정성"을 검증

    손절 설계가 잘 되었으면:
    - worst_mdd가 시간이 지나도 일정 범위(-4~-6%) 내에 머무름
    - stop_pct/max_loss_pct가 시장 상황에 따라 자연스럽게 변동
    - var_95가 갑자기 악화되는 구간이 없음

    Args:
        dates: 각 거래의 날짜 배열 (YYYYMMDD str 또는 datetime)
        returns: 각 거래의 수익률 배열 (%)
        window_days: 롤링 윈도우 크기 (거래 건수 기준)

    Returns:
        DataFrame with columns:
        period_start, period_end, n_trades, win_rate, avg_ret,
        worst_mdd, var_95, pl_ratio, max_consec_loss, stability_flag
    """
    if len(dates) != len(returns) or len(dates) == 0:
        return pd.DataFrame()

    dates = np.asarray(dates)
    returns = np.asarray(returns, dtype=np.float64)

    # NaN/inf 제거
    valid = np.isfinite(returns)
    dates = dates[valid]
    returns = returns[valid]

    if len(returns) < window_days:
        # 데이터 부족 시 전체를 하나의 윈도우로
        kpi = calc_risk_kpi(returns)
        kpi["period_start"] = str(dates[0]) if len(dates) > 0 else ""
        kpi["period_end"] = str(dates[-1]) if len(dates) > 0 else ""
        kpi["stability_flag"] = "🟢"
        return pd.DataFrame([kpi])

    rows = []
    for start in range(0, len(returns) - window_days + 1, max(1, window_days // 2)):
        end = min(start + window_days, len(returns))
        chunk = returns[start:end]
        kpi = calc_risk_kpi(chunk)
        kpi["period_start"] = str(dates[start])
        kpi["period_end"] = str(dates[end - 1])

        # 안정성 플래그
        if kpi["worst_mdd"] < -10.0 or kpi["profit_loss_ratio"] < 0.8:
            kpi["stability_flag"] = "🔴"  # 위험: 손절 설계 재점검 필요
        elif kpi["worst_mdd"] < -6.0 or kpi["profit_loss_ratio"] < 1.2:
            kpi["stability_flag"] = "🟡"  # 주의
        else:
            kpi["stability_flag"] = "🟢"  # 정상

        rows.append(kpi)

    result = pd.DataFrame(rows)
    # 컬럼 순서 정리
    col_order = ["period_start", "period_end", "n_trades", "win_rate",
                 "worst_mdd", "var_95", "profit_loss_ratio",
                 "max_consecutive_loss", "stability_flag"]
    for c in col_order:
        if c not in result.columns:
            result[c] = ""
    return result[col_order]


# ═══════════════════════════════════════════════════
#  [Phase 2-3] 표준화된 진입 방어 규칙
# ═══════════════════════════════════════════════════

def _build_entry_defense_rules(policy=None):
    """[v20.7] PolicyConfig SSOT에서 Entry Defense 규칙 생성."""
    p = policy or _CFG.policy
    return [
        {
            "name": "gap_hold",
            "condition": lambda row, _th=p.entry_gap_hold_pct: float(row.get("gap_pct", 0) or 0) > _th,
            "action": "hold",
            "position_pct": 0,
            "reason": f"갭 {p.entry_gap_hold_pct:.0f}%+ 진입 보류",
        },
        {
            "name": "gap_split",
            "condition": lambda row, _lo=p.entry_gap_split_pct, _hi=p.entry_gap_hold_pct: _lo < float(row.get("gap_pct", 0) or 0) <= _hi,
            "action": "split",
            "position_pct": 50,
            "reason": f"갭 {p.entry_gap_split_pct:.0f}%+ 분할 진입",
        },
        {
            "name": "vi_triggered",
            "condition": lambda row: bool(row.get("is_vi_triggered", False)),
            "action": "hold",
            "position_pct": 0,
            "reason": "VI 발동 진입 보류",
        },
        {
            "name": "consecutive_limit_up",
            "condition": lambda row, _th=p.entry_consecutive_limit_up: int(row.get("consecutive_limit_up", 0) or 0) >= _th,
            "action": "hold",
            "position_pct": 0,
            "reason": "연속 상한가 진입 보류",
        },
        {
            "name": "extreme_surge",
            "condition": lambda row, _th=p.entry_surge_hold_pct: float(row.get("ret_1d_%", 0) or 0) > _th,
            "action": "hold",
            "position_pct": 0,
            "reason": f"당일 {p.entry_surge_hold_pct:.0f}%+ 급등 진입 보류",
        },
        {
            "name": "moderate_surge",
            "condition": lambda row, _lo=p.entry_surge_split_pct, _hi=p.entry_surge_hold_pct: _lo < float(row.get("ret_1d_%", 0) or 0) <= _hi,
            "action": "split",
            "position_pct": 50,
            "reason": f"당일 {p.entry_surge_split_pct:.0f}%+ 급등 분할 진입",
        },
        {
            "name": "very_low_liquidity",
            "condition": lambda row, _th=p.entry_turnover_hold_eok: float(row.get("거래대금(억)", row.get("거래대금(억원)", 999)) or 999) < _th,
            "action": "hold",
            "position_pct": 0,
            "reason": f"거래대금 {p.entry_turnover_hold_eok:.0f}억 미만 진입 차단",
        },
        {
            "name": "rsi_overheat",
            "condition": lambda row, _th=p.entry_rsi_split: float(row.get("RSI14", 50) or 50) > _th,
            "action": "split",
            "position_pct": 50,
            "reason": f"RSI {p.entry_rsi_split:.0f}+ 과열 분할 진입",
        },
    ]

ENTRY_DEFENSE_RULES = _build_entry_defense_rules()


def check_entry_defense(row: dict) -> dict:
    """
    표준화된 진입 방어 규칙 적용.
    
    Args:
        row: 종목 데이터 (dict 또는 Series)
    
    Returns:
        {"action": "enter"|"split"|"hold", "position_pct": float, "reason": str, "rules_triggered": list}
    """
    triggered = []
    for rule in ENTRY_DEFENSE_RULES:
        try:
            if rule["condition"](row):
                triggered.append(rule)
        except Exception:
            continue

    if not triggered:
        return {"action": "enter", "position_pct": 100.0, "reason": "정상", "rules_triggered": []}

    # hold 규칙이 하나라도 있으면 hold
    holds = [r for r in triggered if r["action"] == "hold"]
    if holds:
        reasons = " + ".join([r["reason"] for r in holds])
        return {
            "action": "hold",
            "position_pct": 0.0,
            "reason": reasons,
            "rules_triggered": [r["name"] for r in triggered],
        }

    # split 규칙만 있으면 가장 작은 position_pct
    min_pct = min(r["position_pct"] for r in triggered)
    reasons = " + ".join([r["reason"] for r in triggered])
    return {
        "action": "split",
        "position_pct": min_pct,
        "reason": reasons,
        "rules_triggered": [r["name"] for r in triggered],
    }
