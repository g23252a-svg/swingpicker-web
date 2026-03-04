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

ENTRY_DEFENSE_RULES = [
    {
        "name": "gap_12pct",
        "condition": lambda row: float(row.get("gap_pct", 0) or 0) > 12,
        "action": "hold",
        "position_pct": 0,
        "reason": "갭 12%+ 진입 보류",
    },
    {
        "name": "gap_7pct",
        "condition": lambda row: 7 < float(row.get("gap_pct", 0) or 0) <= 12,
        "action": "split",
        "position_pct": 50,
        "reason": "갭 7%+ 분할 진입",
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
        "condition": lambda row: int(row.get("consecutive_limit_up", 0) or 0) >= 2,
        "action": "hold",
        "position_pct": 0,
        "reason": "연속 상한가 진입 보류",
    },
    {
        "name": "extreme_surge",
        "condition": lambda row: float(row.get("ret_1d_%", 0) or 0) > 15,
        "action": "hold",
        "position_pct": 0,
        "reason": "당일 15%+ 급등 진입 보류",
    },
    {
        "name": "moderate_surge",
        "condition": lambda row: 10 < float(row.get("ret_1d_%", 0) or 0) <= 15,
        "action": "split",
        "position_pct": 50,
        "reason": "당일 10%+ 급등 분할 진입",
    },
    # ── [Phase 2-3] 추가 방어 규칙 ──
    {
        "name": "very_low_liquidity",
        "condition": lambda row: float(row.get("거래대금(억)", row.get("거래대금(억원)", 999)) or 999) < 5,
        "action": "split",
        "position_pct": 30,
        "reason": "거래대금 5억 미만 소량 진입",
    },
    {
        "name": "rsi_overheat",
        "condition": lambda row: float(row.get("RSI14", 50) or 50) > 80,
        "action": "split",
        "position_pct": 50,
        "reason": "RSI 80+ 과열 분할 진입",
    },
]


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
