# -*- coding: utf-8 -*-
"""
scoring_engine.py — 종목 스코어링 + 상태 머신 엔진
──────────────────────────────────────────────────
collector.py 에서 추출한 스코어링 함수 전체.
순수 계산 로직만 포함 (DB/네트워크 의존 없음).
"""
import numpy as np
import pandas as pd
from shared_utils import nz_num, safe_float


# ═══════════════════════════════════════════════════
#  1. 개별 스코어 함수
# ═══════════════════════════════════════════════════

def calculate_ebs_independent(row) -> int:
    """[독립형 EBS] 5가지 펀더멘털 체크리스트 (0~10점)"""
    score = 0
    if row.get('Low_Trend_PCT', 0) > 0:
        score += 2
    if row.get('Vol_Quality', 0) >= 1.1:
        score += 2
    if row.get('MACD_Slope_PCT', 0) > 0:
        score += 2
    rsi = row.get('RSI14', 50)
    if 45 <= rsi <= 70:
        score += 2
    if row.get('TTM_SQUEEZE', 0) == 1 or row.get('BB_Expanding', 0) == 1:
        score += 2
    return score


def calculate_structural_score(row) -> float:
    """
    [STRUCT_SCORE] 종목의 기초 체력 (0~100)
    ✅ 섹터 보너스 미포함 (TIMING_SCORE에서만 반영 → 이중 보상 방지)
    ✅ [v15] Multi-Timeframe 보정: 완료된 주봉/월봉 기반 bonus/penalty
    """
    def _norm(val, max_val):
        return min(max(val / max_val, 0), 1)

    trend_score = _norm(row.get('Low_Trend_PCT', 0), 3.0) * 40
    mfi_score = _norm(row.get('MFI14', 50) - 30, 40) * 15
    vq_score = _norm(row.get('Vol_Quality', 0) - 0.8, 1.2) * 15
    range_score = _norm(row.get('Range_Pos', 0), 1.0) * 15

    disp = row.get('이격도', 0)
    if 0 <= disp <= 5:
        disp_score = 15
    elif disp < 0:
        disp_score = 5
    else:
        disp_score = max(15 - (disp - 5), 0)

    base = trend_score + mfi_score + vq_score + range_score + disp_score
    penalty = 0
    if row.get('Above_MA20', 0) == 0:
        penalty += 20

    # ✅ [v15] Multi-Timeframe 보정 (주봉/월봉)
    # MTF_WEEKLY_TREND: +1=상승, 0=중립, -1=하락 (collector에서 주입)
    # MTF_MONTHLY_TREND: 동일
    # Config: MTF_STRUCT_BONUS / MTF_STRUCT_PENALTY
    _sf = safe_float
    mtf_w = int(_sf(row.get('MTF_WEEKLY_TREND', 0)))
    mtf_m = int(_sf(row.get('MTF_MONTHLY_TREND', 0)))
    mtf_data_ok = int(_sf(row.get('MTF_DATA_SUFFICIENT', 0)))

    mtf_bonus_val = float(row.get('_MTF_STRUCT_BONUS', 10))
    mtf_penalty_val = float(row.get('_MTF_STRUCT_PENALTY', 15))

    mtf_adj = 0.0
    if mtf_data_ok:
        # 주봉+월봉 모두 상승 → bonus
        if mtf_w >= 1 and mtf_m >= 1:
            mtf_adj = mtf_bonus_val
        # 주봉+월봉 모두 하락 → penalty
        elif mtf_w <= -1 and mtf_m <= -1:
            mtf_adj = -mtf_penalty_val
        # 한쪽만 상승 → 소폭 bonus
        elif mtf_w >= 1 or mtf_m >= 1:
            mtf_adj = mtf_bonus_val * 0.5
        # 한쪽만 하락 → 소폭 penalty
        elif mtf_w <= -1 or mtf_m <= -1:
            mtf_adj = -mtf_penalty_val * 0.5
        # 양쪽 중립 → 0
    # mtf_data_ok=0 → 데이터 부족 → 보정 0 (안전장치 2)

    return max(0.0, min(100.0, base - penalty + mtf_adj))


def calculate_timing_score(row) -> float:
    """
    [v14] TIMING_SCORE 계산 — 매물대 + 기술적 + 섹터 보정
    ✅ 섹터 보너스는 여기서만 반영 (STRUCT에는 미포함 → 이중 보상 방지)
    """
    _sf = safe_float  # alias

    raw = _sf(row.get('RAW_TRIGGER_SCORE', row.get('TRIGGER_SCORE', 0)))
    std_trigger = min(raw / 90.0 * 100.0, 100.0)

    bonus = 0
    penalty = 0

    # 매물대(Volume Profile) 보정
    res_all = _sf(row.get('RES_RATIO', 0))
    res_near = _sf(row.get('RES_RATIO_NEAR', 0))
    poc_gap = _sf(row.get('POC_GAP', 0))
    is_above = int(_sf(row.get('IS_ABOVE_POC', 0)))

    if is_above == 1:
        bonus += max(0, 12 * (1 - min(res_all, 0.30) / 0.30))
        if res_near < 0.05:
            bonus += 3
        if poc_gap > 12:
            bonus = max(0, bonus - 4)
    else:
        penalty += min(15, 15 * (min(res_all, 0.45) / 0.45))
        if res_near > 0.20:
            penalty += 5

    # 기술적 보너스 / 패널티
    if int(_sf(row.get('TTM_SQUEEZE', 0))) == 1:
        bonus += 10
    if int(_sf(row.get('SUPERTREND_DIR', 0))) == 1:
        bonus += 5

    rsi = _sf(row.get('RSI14', 50))
    gap_pct = _sf(row.get('gap_pct', 0))

    if rsi > 75:
        penalty += 20
    if gap_pct > 5.0:
        penalty += 10

    # ✅ [v14] 섹터 모멘텀 보너스 (TIMING 한 곳에서만)
    sector_rank = _sf(row.get('SECTOR_RANK', 99))
    if sector_rank <= 3:
        bonus += 8
    elif sector_rank <= 6:
        bonus += 4

    return max(0.0, min(std_trigger + bonus - penalty, 100.0))


# ═══════════════════════════════════════════════════
#  2. 추세 분류 (REGIME)
# ═══════════════════════════════════════════════════

def detect_regime_row(row: pd.Series) -> str:
    """추세 단계 텍스트 분류 (rel_60d_%, MACD_Slope, RSI14 기반)"""
    def _fv(key: str, default: float = 0.0) -> float:
        try:
            val = row.get(key)
            if val is None or pd.isna(val):
                return default
            return float(val)
        except Exception:
            return default

    rel60 = _fv("rel_60d_%", 0.0)
    slope = _fv("MACD_Slope_PCT", 0.0)
    if slope == 0.0:
        slope = _fv("MACD_Slope", 0.0)
    rsi = _fv("RSI14", 50.0)

    if rel60 > 10 and slope > 0 and 50 <= rsi <= 70:
        return "① 강한 상승 추세"
    if rel60 > 5 and slope <= 0:
        return "② 상승 후 조정"
    if -5 <= rel60 <= 5:
        return "③ 박스 / 중립"
    if rel60 <= -5 and slope > 0:
        return "④ 바닥 반등 시도"
    return "⑤ 하락 / 약세"


# ═══════════════════════════════════════════════════
#  3. 곡선형 패널티
# ═══════════════════════════════════════════════════

def apply_curve_penalty(val, threshold, power=2.0, weight=1.0):
    """임계치 초과 시 제곱(power)으로 커지는 패널티"""
    if val <= threshold:
        return 0.0
    return ((val - threshold) ** power) * weight


# ═══════════════════════════════════════════════════
#  4. 상태 머신 (ROUTE)
# ═══════════════════════════════════════════════════

def determine_state(row, RouteState=None):
    """
    [정적 임계치] ATTACK/ARMED/WAIT/NEUTRAL/OVERHEAT 결정.
    ⚠️ 레거시 함수: 신규 코드는 determine_state_dynamic 사용 권장.
    RouteState: collector 의 schema.RouteState (또는 문자열 반환용 None)
    """
    # RouteState 가 없으면 문자열로 반환
    if RouteState is None:
        class _RS:
            OVERHEAT = "OVERHEAT"
            ATTACK = "ATTACK"
            ARMED = "ARMED"
            WAIT = "WAIT"
            NEUTRAL = "NEUTRAL"
        RouteState = _RS()

    try:
        rsi = float(row.get('RSI14', 50))
        r5 = float(row.get('ret_5d_%', 0))
        above_ma20 = int(row.get('Above_MA20', 0))
        slope = float(row.get('MACD_Slope_PCT', 0))
        # ✅ (B) TIMING_SCORE 우선, fallback으로 TRIGGER_SCORE (하위호환)
        t_score = float(row.get('TIMING_SCORE', row.get('TRIGGER_SCORE', 0)))
        is_squeeze = int(row.get('TTM_SQUEEZE', 0))
        vol_qual = float(row.get('Vol_Quality', 1.0))
        range_pos = float(row.get('Range_Pos', 0))

        if rsi >= 75 or r5 >= 20.0:
            return RouteState.OVERHEAT
        if (above_ma20 == 1 and slope > 0 and t_score >= 60
                and vol_qual >= 1.3 and range_pos >= 0.8):
            return RouteState.ATTACK
        if is_squeeze == 1 and above_ma20 == 1:
            return RouteState.ARMED
        if vol_qual >= 2.0:
            return RouteState.ARMED
        if float(row.get('Low_Trend_PCT', 0)) > 0:
            return RouteState.WAIT
        return RouteState.NEUTRAL
    except Exception:
        return RouteState.NEUTRAL


def determine_state_dynamic(row, thresholds: dict):
    """
    [동적 임계치] 수급 상대 비중 기반 Toxic Filter + 상태 결정
    """
    try:
        def _get(k, default=0.0):
            val = row.get(k, default)
            try:
                return float(val) if not pd.isna(val) else default
            except Exception:
                return default

        rsi = _get('RSI14', 50)
        r1 = _get('ret_1d_%', 0)
        r5 = _get('ret_5d_%', 0)
        slope = _get('MACD_Slope_PCT', 0)
        range_pos = _get('Range_Pos', 0)
        vol_qual = _get('Vol_Quality', 1.0)
        t_score = _get('TIMING_SCORE', 0)
        vol_z = _get('거래강도', 0)
        low_trend = _get('Low_Trend_PCT', 0)
        above_ma20 = int(_get('Above_MA20', 0))

        turnover = _get('거래대금(원)', 0.0)
        frg_net = _get('외인순매수금액', _get('외인순매수', 0))
        ind_net = _get('개인순매수금액', _get('개인순매수', 0))

        # ✅ [v14] fail-safe: turnover 유효성 검증
        # thresholds로 최소값 관리 (기본: 5000만원 = 소형주 최소 거래대금 수준)
        _turnover_min = thresholds.get('turnover_min_valid', 50_000_000)
        _turnover_valid = turnover >= _turnover_min
        if _turnover_valid:
            frg_ratio = (frg_net / turnover) * 100
            ant_ratio = (ind_net / turnover) * 100
        else:
            frg_ratio = 0.0
            ant_ratio = 0.0

        # Toxic Filter (turnover 유효할 때만 수급 기반 판정)
        if _turnover_valid and r1 > 5.0 and frg_ratio < -20.0 and ant_ratio > 20.0:
            return "EXIT_WARNING"
        if vol_z >= 10.0 and r1 >= 10.0:
            return "EXIT_WARNING"

        # (1) 과열
        if rsi >= 75 or r5 >= 25.0:
            return "OVERHEAT"

        # (2) ATTACK
        vol_cut = thresholds.get('vol_q75', 1.2)
        range_cut = thresholds.get('range_q75', 0.8)
        if (slope > 0 and range_pos >= range_cut and vol_qual >= vol_cut
                and t_score >= 60 and above_ma20 == 1):
            if low_trend < -3.0:
                return "WAIT"
            return "ATTACK"

        # (3) ARMED
        is_squeeze = int(row.get('TTM_SQUEEZE', 0))
        if (is_squeeze == 1 or vol_qual >= 2.0) and above_ma20 == 1:
            if low_trend >= -3.0:
                return "ARMED"

        # (4) WAIT
        if low_trend > 0 or r1 > 0:
            return "WAIT"

        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


# ═══════════════════════════════════════════════════
#  5. 글로벌 스코어 통합 (collector 용)
# ═══════════════════════════════════════════════════
#  [Phase 1-1] CollectorConfig SSOT 전환 완료
# ═══════════════════════════════════════════════════

from collector_config import DEFAULT_CONFIG, CollectorConfig


def _calc_ml_weight(ml_series: pd.Series, macro_risk: str,
                    config=None) -> tuple:
    """
    ML 활성도 기반 동적 가중치 (w_struct, w_timing, w_ai).

    config: CollectorConfig 인스턴스 (None → DEFAULT_CONFIG)
    """
    cfg = config if isinstance(config, CollectorConfig) else DEFAULT_CONFIG

    ml = ml_series.fillna(0)
    ml_cov = float((ml > 0).mean())

    # 절사평균(상하 trim_pct 제거)
    n = len(ml)
    if n >= 10:
        trim_k = max(1, int(n * cfg.trim_pct))
        ml_sorted = ml.sort_values().values
        ml_center = float(ml_sorted[trim_k:-trim_k].mean())
    else:
        ml_center = float(ml.mean())

    # AI 가중치: 선형 보간 + 커버리지 게이트
    if ml_center <= cfg.ml_low or ml_cov < cfg.ml_cov_gate:
        w_a = 0.0
    elif ml_center >= cfg.ml_high:
        w_a = cfg.ml_max_weight
    else:
        w_a = cfg.ml_max_weight * (ml_center - cfg.ml_low) / (cfg.ml_high - cfg.ml_low)

    # 매크로별 기본 S:T 비율
    base_s, base_t = cfg.macro_weights.get(macro_risk, cfg.macro_weights.get("NORMAL", (0.40, 0.40)))

    # S/T에 비례 재분배
    rem = 1.0 - w_a
    st_sum = base_s + base_t
    w_s = rem * (base_s / st_sum)
    w_t = rem * (base_t / st_sum)

    # 최종 정규화
    total = w_s + w_t + w_a
    if total > 0:
        w_s /= total
        w_t /= total
        w_a /= total

    return round(w_s, 6), round(w_t, 6), round(w_a, 6)


def build_global_score(df: pd.DataFrame, macro_risk: str,
                       config=None) -> pd.DataFrame:
    """
    STRUCT + TIMING + AI → FINAL_SCORE 산출.
    EBS는 필터/게이트 역할 → PASS_EBS 컬럼 생성 (점수 합산에는 미포함).
    ✅ [v14] ML 활성도 기반 동적 가중치
    """
    cfg = config if isinstance(config, CollectorConfig) else DEFAULT_CONFIG
    x = df.copy()

    x["EBS"] = x.apply(calculate_ebs_independent, axis=1)
    x["PASS_EBS"] = (x["EBS"] >= cfg.ebs_pass_threshold).astype(int)

    x["STRUCT_SCORE"] = x.apply(calculate_structural_score, axis=1).round(1)
    x["TIMING_SCORE"] = x.apply(calculate_timing_score, axis=1).round(1)

    if "ML_SCORE" not in x.columns:
        x["ML_SCORE"] = 0.0
    x["AI_SCORE"] = x["ML_SCORE"].clip(0, 100).round(1)

    w_s, w_t, w_a = _calc_ml_weight(x["ML_SCORE"], macro_risk, config=cfg)

    x["FINAL_SCORE"] = (
        (x["STRUCT_SCORE"] * w_s)
        + (x["TIMING_SCORE"] * w_t)
        + (x["AI_SCORE"] * w_a)
    ).round(1)

    x["DISPLAY_SCORE"] = x["FINAL_SCORE"]
    return x
