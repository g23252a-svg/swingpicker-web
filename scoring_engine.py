# -*- coding: utf-8 -*-
"""
scoring_engine.py — 종목 스코어링 + 상태 머신 엔진 (v6.0 Vectorized)
──────────────────────────────────────────────────────────────────────
[v6.0] apply(axis=1) 전면 제거 → 벡터 연산으로 리팩토링
  - calculate_ebs_independent  → _vec_ebs()
  - calculate_structural_score → _vec_structural_score()
  - calculate_timing_score     → _vec_timing_score()
  - 100종목 기준 약 30~50x 속도 향상
"""
import numpy as np
import pandas as pd
from shared_utils import nz_num, safe_float

from collector_config import DEFAULT_CONFIG, CollectorConfig


# ═══════════════════════════════════════════════════
#  1. 벡터화된 스코어 함수
# ═══════════════════════════════════════════════════

def _safe_col(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    """컬럼이 없으면 default로 채운 Series 반환"""
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index)


def _vec_ebs(df: pd.DataFrame) -> pd.Series:
    """
    [Vectorized EBS] 5가지 펀더멘털 체크리스트 (0~10점)
    기존: df.apply(calculate_ebs_independent, axis=1)
    """
    score = pd.Series(0, index=df.index, dtype='int64')

    score += (_safe_col(df, 'Low_Trend_PCT') > 0).astype(int) * 2
    score += (_safe_col(df, 'Vol_Quality') >= 1.1).astype(int) * 2
    score += (_safe_col(df, 'MACD_Slope_PCT') > 0).astype(int) * 2

    rsi = _safe_col(df, 'RSI14', 50)
    score += ((rsi >= 45) & (rsi <= 70)).astype(int) * 2

    ttm = _safe_col(df, 'TTM_SQUEEZE')
    bb_exp = _safe_col(df, 'BB_Expanding')
    score += ((ttm == 1) | (bb_exp == 1)).astype(int) * 2

    return score


def _vec_structural_score(df: pd.DataFrame) -> pd.Series:
    """
    [Vectorized STRUCT_SCORE] 종목의 기초 체력 (0~100)
    기존: df.apply(calculate_structural_score, axis=1)
    """
    # 정규화 헬퍼 (벡터 버전)
    def _norm(s, max_val):
        return (s / max_val).clip(0, 1)

    trend_score = _norm(_safe_col(df, 'Low_Trend_PCT'), 3.0) * 40
    mfi_score = _norm(_safe_col(df, 'MFI14', 50) - 30, 40) * 15
    vq_score = _norm(_safe_col(df, 'Vol_Quality') - 0.8, 1.2) * 15
    range_score = _norm(_safe_col(df, 'Range_Pos'), 1.0) * 15

    # 이격도 점수 (조건부 벡터)
    disp = _safe_col(df, '이격도')
    disp_score = pd.Series(0.0, index=df.index)
    disp_score = np.where((disp >= 0) & (disp <= 5), 15.0, disp_score)
    disp_score = np.where(disp < 0, 5.0, disp_score)
    disp_score = np.where(disp > 5, np.maximum(15 - (disp - 5), 0), disp_score)
    disp_score = pd.Series(disp_score, index=df.index)

    base = trend_score + mfi_score + vq_score + range_score + disp_score

    # 패널티
    penalty = (_safe_col(df, 'Above_MA20') == 0).astype(float) * 20

    # Multi-Timeframe 보정
    mtf_w = _safe_col(df, 'MTF_WEEKLY_TREND').astype(int)
    mtf_m = _safe_col(df, 'MTF_MONTHLY_TREND').astype(int)
    mtf_ok = _safe_col(df, 'MTF_DATA_SUFFICIENT').astype(int)

    bonus_val = _safe_col(df, '_MTF_STRUCT_BONUS', 10.0)
    penalty_val = _safe_col(df, '_MTF_STRUCT_PENALTY', 15.0)

    mtf_adj = pd.Series(0.0, index=df.index)
    # 주봉+월봉 모두 상승
    both_up = mtf_ok & (mtf_w >= 1) & (mtf_m >= 1)
    mtf_adj = np.where(both_up, bonus_val, mtf_adj)
    # 주봉+월봉 모두 하락
    both_dn = mtf_ok & (mtf_w <= -1) & (mtf_m <= -1) & ~both_up
    mtf_adj = np.where(both_dn, -penalty_val, mtf_adj)
    # 한쪽만 상승
    one_up = mtf_ok & ((mtf_w >= 1) | (mtf_m >= 1)) & ~both_up & ~both_dn
    mtf_adj = np.where(one_up, bonus_val * 0.5, mtf_adj)
    # 한쪽만 하락
    one_dn = mtf_ok & ((mtf_w <= -1) | (mtf_m <= -1)) & ~both_up & ~both_dn & ~one_up
    mtf_adj = np.where(one_dn, -penalty_val * 0.5, mtf_adj)

    mtf_adj = pd.Series(mtf_adj, index=df.index, dtype=float)

    return (base - penalty + mtf_adj).clip(0, 100).round(1)


def _vec_timing_score(df: pd.DataFrame) -> pd.Series:
    """
    [Vectorized TIMING_SCORE] 매물대 + 기술적 + 섹터 보정 (0~100)
    기존: df.apply(calculate_timing_score, axis=1)
    """
    raw = _safe_col(df, 'RAW_TRIGGER_SCORE')
    # fallback: RAW_TRIGGER_SCORE가 없으면 TRIGGER_SCORE
    mask_zero = raw == 0
    if mask_zero.any() and 'TRIGGER_SCORE' in df.columns:
        raw = raw.where(~mask_zero, _safe_col(df, 'TRIGGER_SCORE'))

    std_trigger = (raw / 90.0 * 100.0).clip(upper=100)

    bonus = pd.Series(0.0, index=df.index)
    penalty = pd.Series(0.0, index=df.index)

    # 매물대(Volume Profile) 보정
    res_all = _safe_col(df, 'RES_RATIO')
    res_near = _safe_col(df, 'RES_RATIO_NEAR')
    poc_gap = _safe_col(df, 'POC_GAP')
    is_above = _safe_col(df, 'IS_ABOVE_POC').astype(int)

    # is_above == 1
    above_bonus = np.maximum(0, 12 * (1 - res_all.clip(upper=0.30) / 0.30))
    above_bonus = np.where(res_near < 0.05, above_bonus + 3, above_bonus)
    above_bonus = np.where(poc_gap > 12, np.maximum(0, above_bonus - 4), above_bonus)
    bonus += np.where(is_above == 1, above_bonus, 0)

    # is_above != 1
    below_pen = np.minimum(15, 15 * (res_all.clip(upper=0.45) / 0.45))
    below_pen = np.where(res_near > 0.20, below_pen + 5, below_pen)
    penalty += np.where(is_above != 1, below_pen, 0)

    # 기술적 보너스 / 패널티
    bonus += (_safe_col(df, 'TTM_SQUEEZE').astype(int) == 1).astype(float) * 10
    bonus += (_safe_col(df, 'SUPERTREND_DIR').astype(int) == 1).astype(float) * 5

    rsi = _safe_col(df, 'RSI14', 50)
    gap_pct = _safe_col(df, 'gap_pct')

    penalty += (rsi > 75).astype(float) * 20
    penalty += (gap_pct > 5.0).astype(float) * 10

    # 섹터 모멘텀 보너스
    sector_rank = _safe_col(df, 'SECTOR_RANK', 99)
    bonus += (sector_rank <= 3).astype(float) * 8
    bonus += ((sector_rank > 3) & (sector_rank <= 6)).astype(float) * 4

    return (std_trigger + bonus - penalty).clip(0, 100).round(1)


# ═══════════════════════════════════════════════════
#  레거시 호환용 (단일 row dict → 점수)
#  외부에서 row 단위로 호출하는 코드가 있을 수 있으므로 유지
# ═══════════════════════════════════════════════════

def calculate_ebs_independent(row) -> int:
    """[레거시 호환] 단일 row dict → EBS 점수"""
    score = 0
    if row.get('Low_Trend_PCT', 0) > 0: score += 2
    if row.get('Vol_Quality', 0) >= 1.1: score += 2
    if row.get('MACD_Slope_PCT', 0) > 0: score += 2
    rsi = row.get('RSI14', 50)
    if 45 <= rsi <= 70: score += 2
    if row.get('TTM_SQUEEZE', 0) == 1 or row.get('BB_Expanding', 0) == 1: score += 2
    return score


def calculate_structural_score(row) -> float:
    """[레거시 호환] 단일 row dict → STRUCT_SCORE"""
    df = pd.DataFrame([row])
    return float(_vec_structural_score(df).iloc[0])


def calculate_timing_score(row) -> float:
    """[레거시 호환] 단일 row dict → TIMING_SCORE"""
    df = pd.DataFrame([row])
    return float(_vec_timing_score(df).iloc[0])


# ═══════════════════════════════════════════════════
#  2. 추세 분류 (REGIME) — 벡터화
# ═══════════════════════════════════════════════════

def detect_regime_row(row: pd.Series) -> str:
    """추세 단계 텍스트 분류 (단일 row, 레거시 호환)"""
    def _fv(key, default=0.0):
        try:
            val = row.get(key)
            if val is None or pd.isna(val): return default
            return float(val)
        except Exception:
            return default

    rel60 = _fv("rel_60d_%")
    slope = _fv("MACD_Slope_PCT") or _fv("MACD_Slope")
    rsi = _fv("RSI14", 50)

    if rel60 > 10 and slope > 0 and 50 <= rsi <= 70:
        return "① 강한 상승 추세"
    if rel60 > 5 and slope <= 0:
        return "② 상승 후 조정"
    if -5 <= rel60 <= 5:
        return "③ 박스 / 중립"
    if rel60 <= -5 and slope > 0:
        return "④ 바닥 반등 시도"
    return "⑤ 하락 / 약세"


def _vec_detect_regime(df: pd.DataFrame) -> pd.Series:
    """[Vectorized] 추세 단계 분류"""
    rel60 = _safe_col(df, 'rel_60d_%')
    slope = _safe_col(df, 'MACD_Slope_PCT')
    if 'MACD_Slope' in df.columns:
        slope = slope.where(slope != 0, _safe_col(df, 'MACD_Slope'))
    rsi = _safe_col(df, 'RSI14', 50)

    regime = pd.Series("⑤ 하락 / 약세", index=df.index)
    regime = regime.where(~((rel60 <= -5) & (slope > 0)), "④ 바닥 반등 시도")
    regime = regime.where(~((rel60 >= -5) & (rel60 <= 5)), "③ 박스 / 중립")
    regime = regime.where(~((rel60 > 5) & (slope <= 0)), "② 상승 후 조정")
    regime = regime.where(
        ~((rel60 > 10) & (slope > 0) & (rsi >= 50) & (rsi <= 70)),
        "① 강한 상승 추세"
    )
    return regime


# ═══════════════════════════════════════════════════
#  3. 곡선형 패널티 (변경 없음)
# ═══════════════════════════════════════════════════

def apply_curve_penalty(val, threshold, power=2.0, weight=1.0):
    if val <= threshold:
        return 0.0
    return ((val - threshold) ** power) * weight


# ═══════════════════════════════════════════════════
#  4. 상태 머신 (ROUTE) — 기존 호환 유지
# ═══════════════════════════════════════════════════

def determine_state(row, RouteState=None):
    """[정적 임계치] 레거시 호환"""
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
        t_score = float(row.get('TIMING_SCORE', row.get('TRIGGER_SCORE', 0)))
        is_squeeze = int(row.get('TTM_SQUEEZE', 0))
        vol_qual = float(row.get('Vol_Quality', 1.0))
        range_pos = float(row.get('Range_Pos', 0))

        if rsi >= 75 or r5 >= 20.0: return RouteState.OVERHEAT
        if (above_ma20 == 1 and slope > 0 and t_score >= 60
                and vol_qual >= 1.3 and range_pos >= 0.8):
            return RouteState.ATTACK
        if is_squeeze == 1 and above_ma20 == 1: return RouteState.ARMED
        if vol_qual >= 2.0: return RouteState.ARMED
        if float(row.get('Low_Trend_PCT', 0)) > 0: return RouteState.WAIT
        return RouteState.NEUTRAL
    except Exception:
        return RouteState.NEUTRAL


def determine_state_dynamic(row, thresholds: dict):
    """[동적 임계치] 레거시 호환"""
    try:
        def _get(k, default=0.0):
            val = row.get(k, default)
            try: return float(val) if not pd.isna(val) else default
            except Exception: return default

        rsi = _get('RSI14', 50)
        r1 = _get('ret_1d_%')
        r5 = _get('ret_5d_%')
        slope = _get('MACD_Slope_PCT')
        range_pos = _get('Range_Pos')
        vol_qual = _get('Vol_Quality', 1.0)
        t_score = _get('TIMING_SCORE')
        vol_z = _get('거래강도')
        low_trend = _get('Low_Trend_PCT')
        above_ma20 = int(_get('Above_MA20'))

        turnover = _get('거래대금(원)')
        frg_net = _get('외인순매수금액', _get('외인순매수'))
        ind_net = _get('개인순매수금액', _get('개인순매수'))

        _turnover_min = thresholds.get('turnover_min_valid', 50_000_000)
        _turnover_valid = turnover >= _turnover_min
        frg_ratio = (frg_net / turnover * 100) if _turnover_valid else 0.0
        ant_ratio = (ind_net / turnover * 100) if _turnover_valid else 0.0

        if _turnover_valid and r1 > 5.0 and frg_ratio < -20.0 and ant_ratio > 20.0:
            return "EXIT_WARNING"
        if vol_z >= 10.0 and r1 >= 10.0:
            return "EXIT_WARNING"
        if rsi >= 75 or r5 >= 25.0:
            return "OVERHEAT"

        vol_cut = thresholds.get('vol_q75', 1.2)
        range_cut = thresholds.get('range_q75', 0.8)
        if (slope > 0 and range_pos >= range_cut and vol_qual >= vol_cut
                and t_score >= 60 and above_ma20 == 1):
            if low_trend < -3.0: return "WAIT"
            return "ATTACK"

        is_squeeze = int(row.get('TTM_SQUEEZE', 0))
        if (is_squeeze == 1 or vol_qual >= 2.0) and above_ma20 == 1:
            if low_trend >= -3.0: return "ARMED"

        if low_trend > 0 or r1 > 0:
            return "WAIT"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


# ═══════════════════════════════════════════════════
#  5. 글로벌 스코어 통합 — [v6.0] 벡터화 적용
# ═══════════════════════════════════════════════════

def _calc_ml_weight(ml_series: pd.Series, macro_risk: str,
                    config=None) -> tuple:
    """ML 활성도 기반 동적 가중치 (변경 없음)"""
    cfg = config if isinstance(config, CollectorConfig) else DEFAULT_CONFIG

    ml = ml_series.fillna(0)
    ml_cov = float((ml > 0).mean())

    n = len(ml)
    if n >= 10:
        trim_k = max(1, int(n * cfg.trim_pct))
        ml_sorted = ml.sort_values().values
        ml_center = float(ml_sorted[trim_k:-trim_k].mean())
    else:
        ml_center = float(ml.mean())

    if ml_center <= cfg.ml_low or ml_cov < cfg.ml_cov_gate:
        w_a = 0.0
    elif ml_center >= cfg.ml_high:
        w_a = cfg.ml_max_weight
    else:
        w_a = cfg.ml_max_weight * (ml_center - cfg.ml_low) / (cfg.ml_high - cfg.ml_low)

    base_s, base_t = cfg.macro_weights.get(macro_risk, cfg.macro_weights.get("NORMAL", (0.40, 0.40)))

    rem = 1.0 - w_a
    st_sum = base_s + base_t
    w_s = rem * (base_s / st_sum)
    w_t = rem * (base_t / st_sum)

    total = w_s + w_t + w_a
    if total > 0:
        w_s /= total; w_t /= total; w_a /= total

    return round(w_s, 6), round(w_t, 6), round(w_a, 6)


def build_global_score(df: pd.DataFrame, macro_risk: str,
                       config=None) -> pd.DataFrame:
    """
    STRUCT + TIMING + AI → FINAL_SCORE 산출.
    ✅ [v6.0] apply(axis=1) 전면 제거 → 벡터 연산
    """
    cfg = config if isinstance(config, CollectorConfig) else DEFAULT_CONFIG
    x = df.copy()

    # ── [v6.0] 벡터화된 스코어링 ──
    x["EBS"] = _vec_ebs(x)
    x["PASS_EBS"] = (x["EBS"] >= cfg.ebs_pass_threshold).astype(int)

    x["STRUCT_SCORE"] = _vec_structural_score(x)
    x["TIMING_SCORE"] = _vec_timing_score(x)

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
