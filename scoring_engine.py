# -*- coding: utf-8 -*-
"""
scoring_engine.py — 종목 스코어링 + 상태 머신 엔진 (v6.0 Vectorized + v20.6.3 SSOT)
──────────────────────────────────────────────────────────────────────
[v6.0] apply(axis=1) 전면 제거 → 벡터 연산으로 리팩토링
  - calculate_ebs_independent  → _vec_ebs()
  - calculate_structural_score → _vec_structural_score()
  - calculate_timing_score     → _vec_timing_score()
  - 100종목 기준 약 30~50x 속도 향상
[v20.6.3] SSOT + Deterministic
  - _vec_determine_state_dynamic(): ROUTE 벡터 판정 (단건 함수 100% 일치)
  - generate_score_reasons(macro_risk): 장세 연동 임계치, numpy argsort 벡터화
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


def _vec_ebs(df: pd.DataFrame, config=None) -> pd.Series:
    """
    [Vectorized EBS] 5가지 펀더멘털 체크리스트 (0~10점)
    기존: df.apply(calculate_ebs_independent, axis=1)
    """
    cfg = config if isinstance(config, CollectorConfig) else DEFAULT_CONFIG
    score = pd.Series(0, index=df.index, dtype='int64')

    score += (_safe_col(df, 'Low_Trend_PCT') > 0).astype(int) * 2
    score += (_safe_col(df, 'Vol_Quality') >= cfg.indicator.vol_quality_min).astype(int) * 2
    score += (_safe_col(df, 'MACD_Slope_PCT') > 0).astype(int) * 2

    rsi = _safe_col(df, 'RSI14', 50)
    rsi_lo, rsi_hi = cfg.indicator.rsi_range
    score += ((rsi >= rsi_lo) & (rsi <= rsi_hi)).astype(int) * 2

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

    # [v4.0] scoring-overhaul: 곱연산 과락(Gate) 시스템
    # 핵심 지표가 최소 기준 미달이면 총점에 패널티 배수 적용
    gate_mult = pd.Series(1.0, index=df.index)
    vq_raw = _safe_col(df, 'Vol_Quality', 0.0)
    gate_mult = gate_mult * np.where(vq_raw < 0.5, 0.3, np.where(vq_raw < 0.8, 0.6, 1.0))
    mfi_raw = _safe_col(df, 'MFI14', 50)
    gate_mult = gate_mult * np.where(mfi_raw < 20, 0.3, np.where(mfi_raw < 30, 0.6, 1.0))
    tv = _safe_col(df, '거래대금(억원)', 0)
    if tv.sum() == 0:
        tv = _safe_col(df, '거래대금(억)', 0)
    gate_mult = gate_mult * np.where(tv < 10, 0.2, np.where(tv < 30, 0.5, 1.0))
    base = base * gate_mult

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


def _vec_timing_score(df: pd.DataFrame, config=None) -> pd.Series:
    """
    [Vectorized TIMING_SCORE] 매물대 + 기술적 + 섹터 보정 (0~100)
    기존: df.apply(calculate_timing_score, axis=1)
    """
    cfg = config if isinstance(config, CollectorConfig) else DEFAULT_CONFIG
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

    penalty += (rsi > cfg.indicator.rsi_penalty_threshold).astype(float) * 20
    penalty += (gap_pct > cfg.indicator.gap_pct_penalty_threshold).astype(float) * 10

    # 섹터 모멘텀 보너스 (데이터 있을 때만)
    sector_rank = _safe_col(df, 'SECTOR_RANK', 99)
    _sector_available = sector_rank.notna() & (sector_rank < 99)
    bonus += (_sector_available & (sector_rank <= 3)).astype(float) * 8
    bonus += (_sector_available & (sector_rank > 3) & (sector_rank <= 6)).astype(float) * 4

    return (std_trigger + bonus - penalty).clip(0, 100).round(1)


# ═══════════════════════════════════════════════════
#  레거시 호환용 (단일 row dict → 점수)
#  외부에서 row 단위로 호출하는 코드가 있을 수 있으므로 유지
# ═══════════════════════════════════════════════════

def calculate_ebs_independent(row, config=None) -> int:
    """[레거시 호환] 단일 row dict → EBS 점수 (v20.6.4: config SSOT)"""
    cfg = config if isinstance(config, CollectorConfig) else DEFAULT_CONFIG
    score = 0
    if row.get('Low_Trend_PCT', 0) > 0: score += 2
    if row.get('Vol_Quality', 0) >= cfg.indicator.vol_quality_min: score += 2
    if row.get('MACD_Slope_PCT', 0) > 0: score += 2
    rsi = row.get('RSI14', 50)
    rsi_lo, rsi_hi = cfg.indicator.rsi_range
    if rsi_lo <= rsi <= rsi_hi: score += 2
    if row.get('TTM_SQUEEZE', 0) == 1 or row.get('BB_Expanding', 0) == 1: score += 2
    return score


def calculate_structural_score(row) -> float:
    """[레거시 호환] 단일 row dict → STRUCT_SCORE"""
    df = pd.DataFrame([row])
    return float(_vec_structural_score(df).iloc[0])


def calculate_timing_score(row) -> float:
    """[레거시 호환] 단일 row dict → TIMING_SCORE"""
    df = pd.DataFrame([row])
    return float(_vec_timing_score(df, config=None).iloc[0])


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

def determine_state(row, RouteState=None, config=None):
    """[정적 임계치] 레거시 호환"""
    cfg = config if isinstance(config, CollectorConfig) else DEFAULT_CONFIG
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

        if rsi >= cfg.indicator.rsi_overheat or r5 >= cfg.indicator.route_overheat_ret5d: return RouteState.OVERHEAT
        if (above_ma20 == 1 and slope > 0
                and t_score >= cfg.indicator.timing_attack_threshold
                and vol_qual >= cfg.indicator.vol_quality_attack
                and range_pos >= 0.8):
            return RouteState.ATTACK
        if is_squeeze == 1 and above_ma20 == 1: return RouteState.ARMED
        if vol_qual >= cfg.indicator.route_armed_vol_quality: return RouteState.ARMED
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

        _cfg_ind = DEFAULT_CONFIG.indicator
        if _turnover_valid and r1 > _cfg_ind.route_exit_ret1d_flow and frg_ratio < _cfg_ind.route_exit_frg_ratio and ant_ratio > _cfg_ind.route_exit_ant_ratio:
            return "EXIT_WARNING"
        if vol_z >= _cfg_ind.route_exit_vol_z and r1 >= _cfg_ind.route_exit_ret1d:
            return "EXIT_WARNING"
        if rsi >= _cfg_ind.rsi_overheat or r5 >= _cfg_ind.route_overheat_ret5d:
            return "OVERHEAT"

        vol_cut = thresholds.get('vol_q75', 1.2)
        range_cut = thresholds.get('range_q75', 0.8)
        if (slope > 0 and range_pos >= range_cut and vol_qual >= vol_cut
                and t_score >= _cfg_ind.route_attack_timing_min and above_ma20 == 1):
            if low_trend < _cfg_ind.route_attack_low_trend_floor: return "WAIT"
            return "ATTACK"

        is_squeeze = int(row.get('TTM_SQUEEZE', 0))
        if (is_squeeze == 1 or vol_qual >= _cfg_ind.route_armed_vol_quality) and above_ma20 == 1:
            if low_trend >= _cfg_ind.route_attack_low_trend_floor: return "ARMED"

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
    # [v20.6.4] partial failure 방지: ML_SCORE가 정확히 0인 종목은
    # '미계산'일 수 있으므로, 활성 종목만으로 coverage/center 산출
    ml_active = ml[ml > 0]
    ml_cov = float(len(ml_active) / max(len(ml), 1))

    n = len(ml_active)
    if n >= 10:
        trim_k = max(1, int(n * cfg.trim_pct))
        ml_sorted = ml_active.sort_values().values
        ml_center = float(ml_sorted[trim_k:-trim_k].mean())
    elif n > 0:
        ml_center = float(ml_active.mean())
    else:
        ml_center = 0.0

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
    ✅ [v19.2] 가중치 투명성: W_STRUCT/W_TIMING/W_AI 컬럼 저장
    ✅ [v19.2] 축 유무 감지: SECTOR/ML 비활성 시 해당 보너스 자동 제외
    """
    cfg = config if isinstance(config, CollectorConfig) else DEFAULT_CONFIG
    x = df.copy()

    # ── [v19.2] 축 가용성 감지 ──
    _has_sector = "SECTOR_RANK" in x.columns and x["SECTOR_RANK"].notna().any()
    _has_ml = "ML_SCORE" in x.columns and (x["ML_SCORE"].fillna(0) != 0).any()

    # ── [v6.0] 벡터화된 스코어링 ──
    x["EBS"] = _vec_ebs(x, config=cfg)
    x["PASS_EBS"] = (x["EBS"] >= cfg.ebs_pass_threshold).astype(int)

    x["STRUCT_SCORE"] = _vec_structural_score(x)
    x["TIMING_SCORE"] = _vec_timing_score(x, config=cfg)

    if "ML_SCORE" not in x.columns:
        x["ML_SCORE"] = 0.0
    x["AI_SCORE"] = x["ML_SCORE"].clip(0, 100).round(1)

    w_s, w_t, w_a = _calc_ml_weight(x["ML_SCORE"], macro_risk, config=cfg)

    # [v19.2] ML 비활성 시 AI 가중치를 STRUCT/TIMING에 재배분
    if not _has_ml and w_a > 0:
        _redistribute = w_a
        w_a = 0.0
        w_s += _redistribute * 0.5
        w_t += _redistribute * 0.5
        # 재정규화
        _total = w_s + w_t
        if _total > 0:
            w_s /= _total
            w_t /= _total

    x["FINAL_SCORE"] = (
        (x["STRUCT_SCORE"] * w_s)
        + (x["TIMING_SCORE"] * w_t)
        + (x["AI_SCORE"] * w_a)
    ).round(1)

    x["DISPLAY_SCORE"] = x["FINAL_SCORE"]

    # [v19.2] 가중치 투명성: 오늘 어떤 비율로 계산됐는지 CSV에 저장
    x["W_STRUCT"] = round(w_s, 3)
    x["W_TIMING"] = round(w_t, 3)
    x["W_AI"] = round(w_a, 3)
    x["SCORING_AXES"] = (
        ("STRUCT+TIMING+AI" if _has_ml else "STRUCT+TIMING")
        + ("+SECTOR" if _has_sector else "")
    )

    return x

# ═══════════════════════════════════════════════════
#  6. [v20.6] 벡터화 ROUTE 판정
# ═══════════════════════════════════════════════════

def _vec_determine_state_dynamic(df: pd.DataFrame,
                                  thresholds: dict) -> pd.Series:
    """
    [v20.6] determine_state_dynamic의 완전 벡터화 버전.
    apply(axis=1) 제거 → 100종목 기준 ~20x 속도 향상.
    """
    def _col(name, default=0.0):
        return _safe_col(df, name, default)

    rsi       = _col('RSI14', 50)
    r1        = _col('ret_1d_%')
    r5        = _col('ret_5d_%')
    slope     = _col('MACD_Slope_PCT')
    range_pos = _col('Range_Pos')
    vol_qual  = _col('Vol_Quality', 1.0)
    t_score   = _col('TIMING_SCORE')
    vol_z     = _col('거래강도')
    low_trend = _col('Low_Trend_PCT')
    above_ma20 = _col('Above_MA20').astype(int)

    turnover  = _col('거래대금(원)')
    frg_net   = _col('외인순매수금액').where(
        _col('외인순매수금액') != 0, _col('외인순매수'))
    ind_net   = _col('개인순매수금액').where(
        _col('개인순매수금액') != 0, _col('개인순매수'))

    _turnover_min = thresholds.get('turnover_min_valid', 50_000_000)
    _turnover_valid = turnover >= _turnover_min

    frg_ratio = np.where(_turnover_valid, frg_net / turnover.replace(0, np.nan) * 100, 0.0)
    ant_ratio = np.where(_turnover_valid, ind_net / turnover.replace(0, np.nan) * 100, 0.0)

    vol_cut   = thresholds.get('vol_q75', 1.2)
    range_cut = thresholds.get('range_q75', 0.8)

    # ── 우선순위 높은 것부터 판정 (하위 조건이 상위 조건 덮어씀) ──
    route = pd.Series("NEUTRAL", index=df.index)

    # WAIT
    mask_wait = (low_trend > 0) | (r1 > 0)
    route = route.where(~mask_wait, "WAIT")

    # ARMED
    is_squeeze = _col('TTM_SQUEEZE').astype(int)
    _ci = DEFAULT_CONFIG.indicator
    mask_armed = ((is_squeeze == 1) | (vol_qual >= _ci.route_armed_vol_quality)) & (above_ma20 == 1) & (low_trend >= _ci.route_attack_low_trend_floor)
    route = route.where(~mask_armed, "ARMED")

    # ATTACK (low_trend 조건은 별도 downgrade에서 처리)
    mask_attack_base = (
        (slope > 0) & (range_pos >= range_cut) & (vol_qual >= vol_cut)
        & (t_score >= _ci.route_attack_timing_min) & (above_ma20 == 1)
    )
    route = route.where(~mask_attack_base, "ATTACK")

    # ATTACK → WAIT 다운그레이드 (low_trend 악화 시)
    mask_attack_downgrade = mask_attack_base & (low_trend < _ci.route_attack_low_trend_floor)
    route = route.where(~mask_attack_downgrade, "WAIT")

    # OVERHEAT
    mask_overheat = (rsi >= _ci.rsi_overheat) | (r5 >= _ci.route_overheat_ret5d)
    route = route.where(~mask_overheat, "OVERHEAT")

    # EXIT_WARNING
    mask_exit_vol = (vol_z >= _ci.route_exit_vol_z) & (r1 >= _ci.route_exit_ret1d)
    mask_exit_flow = (
        _turnover_valid & (r1 > _ci.route_exit_ret1d_flow)
        & (pd.Series(frg_ratio, index=df.index) < _ci.route_exit_frg_ratio)
        & (pd.Series(ant_ratio, index=df.index) > _ci.route_exit_ant_ratio)
    )
    mask_exit = mask_exit_vol | mask_exit_flow
    route = route.where(~mask_exit, "EXIT_WARNING")

    return route


# ═══════════════════════════════════════════════════
#  7. [v20.6] 점수 설명(Reason) 생성
# ═══════════════════════════════════════════════════

def generate_score_reasons(df: pd.DataFrame,
                           macro_risk: str = "NORMAL") -> pd.DataFrame:
    """
    [v20.6] FINAL_SCORE의 주요 기여/리스크 요인을 사람이 읽을 수 있게 생성.
    컬럼 추가: SCORE_REASON_TOP1, SCORE_REASON_TOP2, SCORE_RISK, ROUTE_REASON

    [v20.6.3] 장세 연동 임계치:
      NORMAL/BULL → 70점 이상이 "강점"
      CAUTION     → 60점 이상
      BEAR/CRITICAL → 50점 이상  (약장에서도 상위 축 설명 가능)
    """
    # ── 장세별 임계치 ──
    _STRENGTH_THRESHOLDS = {
        "BULL": 70, "NORMAL": 70,
        "CAUTION": 60,
        "BEAR": 50, "CRITICAL": 50,
    }
    strength_th = _STRENGTH_THRESHOLDS.get(macro_risk, 70)

    x = df.copy()
    n = len(x)

    reasons_top1 = pd.Series("", index=x.index)
    reasons_top2 = pd.Series("", index=x.index)
    risk_col     = pd.Series("", index=x.index)
    route_reason = pd.Series("", index=x.index)

    struct = _safe_col(x, 'STRUCT_SCORE')
    timing = _safe_col(x, 'TIMING_SCORE')
    ai     = _safe_col(x, 'AI_SCORE')
    rsi    = _safe_col(x, 'RSI14', 50)
    r5     = _safe_col(x, 'ret_5d_%')
    low_t  = _safe_col(x, 'Low_Trend_PCT')
    vq     = _safe_col(x, 'Vol_Quality', 1.0)
    mfi    = _safe_col(x, 'MFI14', 50)
    tv     = _safe_col(x, '거래대금(억원)')
    route  = x.get('ROUTE', pd.Series("", index=x.index)).astype(str)

    # ── 강점 판별 (실제 점수순 — 완전 벡터화, 장세 연동 임계치) ──
    axis_names = np.array(['STRUCT', 'TIMING', 'AI'])
    axis_vals  = np.column_stack([struct.values, timing.values, ai.values])  # (N, 3)

    # 장세별 임계치 적용
    axis_masked = np.where(axis_vals >= strength_th, axis_vals, -np.inf)

    # 행별 내림차순 argsort (큰 값이 앞으로)
    order = np.argsort(-axis_masked, axis=1)  # (N, 3)

    # 1위/2위 인덱스
    idx1 = order[:, 0]
    idx2 = order[:, 1]

    # 해당 축 값이 70점 이상인지 체크
    val1 = axis_masked[np.arange(len(idx1)), idx1]
    val2 = axis_masked[np.arange(len(idx2)), idx2]

    top1_labels = np.where(val1 > -np.inf,
                           np.char.add(axis_names[idx1], ' 강점'), '')
    top2_labels = np.where(val2 > -np.inf,
                           np.char.add(axis_names[idx2], ' 보조'), '')

    reasons_top1 = pd.Series(top1_labels, index=x.index)
    reasons_top2 = pd.Series(top2_labels, index=x.index)

    # 추가 강점 세분화 (3축 모두 70점 미만일 때 fallback)
    reasons_top1 = reasons_top1.where(
        ~((low_t > 2.0) & (reasons_top1 == "")), "저점추세 양호")
    reasons_top1 = reasons_top1.where(
        ~((vq >= 2.0) & (reasons_top1 == "")), "거래품질 우수")

    # ── 리스크 ──
    risk_col = risk_col.where(~(rsi >= 70), "RSI 과열")
    risk_col = risk_col.where(
        ~((r5 >= 15) & (risk_col == "")), "5일 급등")
    risk_col = risk_col.where(
        ~((tv < 30) & (risk_col == "")), "유동성 부족")
    risk_col = risk_col.where(
        ~((mfi < 25) & (risk_col == "")), "MFI 약세")
    risk_col = risk_col.where(
        ~((low_t < -5) & (risk_col == "")), "저점 이탈")

    # ── ROUTE 사유 ──
    route_reason = route_reason.where(
        ~(route == "EXIT_WARNING"), "수급/거래량 이상 감지")
    route_reason = route_reason.where(
        ~((route == "OVERHEAT") & (route_reason == "")), f"과열 (RSI/수익률)")
    route_reason = route_reason.where(
        ~((route == "ATTACK") & (route_reason == "")), "기술적 돌파 조건 충족")
    route_reason = route_reason.where(
        ~((route == "ARMED") & (route_reason == "")), "스퀴즈/품질 대기")
    route_reason = route_reason.where(
        ~((route == "WAIT") & (route_reason == "")), "추세 관망")
    route_reason = route_reason.where(
        ~((route == "NEUTRAL") & (route_reason == "")), "조건 미충족")

    x["SCORE_REASON_TOP1"] = reasons_top1
    x["SCORE_REASON_TOP2"] = reasons_top2
    x["SCORE_RISK"]        = risk_col
    x["ROUTE_REASON"]      = route_reason
    x["REASON_THRESHOLD"]  = strength_th  # [v20.6.3] 장세별 임계치 기록

    return x
