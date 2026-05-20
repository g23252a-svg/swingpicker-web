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


# ═══════════════════════════════════════════════════
#  [v22] ELITE_SCORE — TOP_PICK 이원화 (AGGRESSIVE/STABLE)
#  ─ 참조: SwingPicker_v22_Final_Consolidated.md §2.2.1
# ═══════════════════════════════════════════════════

# TOP_PICK positive gate — ROUTE가 이 집합에 있어야만 TOP_PICK 가능
# (기존 negative mask는 WAIT 통과를 허용해 누출 발생 — v22에서 차단)
TOP_PICK_ROUTES = frozenset({"ARMED", "ATTACK"})


def compute_elite_score(df: pd.DataFrame,
                         out_dir: str = None,
                         trade_ymd: str = None):
    """
    [v22] ELITE_SCORE + TOP_PICK 이원화.
    
    반환: (df_out, meta) 튜플.
      - meta.stable_funnel: STABLE 분기 단계별 통과 수
      - meta.aggressive_funnel: AGGRESSIVE 분기 단계별 통과 수
      - meta.sidecar_path: top_pick_funnel_{ymd}.json 경로 (저장 시)
    
    v21.5 공식 유지:
      ELITE = (S×30% + T×45% + AI×25%) × 밸런스배수 × RR게이트
    
    v22 추가:
      - TOP_PICK positive gate: ROUTE ∈ {ARMED, ATTACK} 필수
      - AGGRESSIVE: ELITE≥75 AND TP1_PCT≥15 (손익비 우선)
      - STABLE: ELITE≥70 AND 7≤TP1<15 AND BALANCE≥70 AND EST_WIN_RATE≥0.55 AND EST_WIN_RATE_MODE=="MATURE"
      - EST_WIN_RATE_MODE 없으면 CALIBRATION_MODE fallback (legacy 호환)
    """
    import numpy as np
    x = df.copy()

    s = x["STRUCT_SCORE"].fillna(0)
    t = x["TIMING_SCORE"].fillna(0)
    m = x["AI_SCORE"].fillna(x.get("ML_SCORE", pd.Series(0, index=x.index)).fillna(0))

    # (1) 3축 평균 (호환용)
    x["AXIS_MEAN"] = ((s + t + m) / 3).round(1)

    # (2) 3축 편차 → 밸런스 점수
    axis_gap = pd.concat([s, t, m], axis=1).max(axis=1) - pd.concat([s, t, m], axis=1).min(axis=1)
    x["AXIS_GAP"] = axis_gap.round(1)
    x["BALANCE_SCORE"] = (100 - axis_gap * 1.25).clip(0, 100).round(1)

    # (3) 현재가 기준 RR
    close = pd.to_numeric(x.get("종가", 0), errors="coerce").fillna(0)
    stop = pd.to_numeric(x.get("손절가", 0), errors="coerce").fillna(0)
    tp1 = pd.to_numeric(x.get("추천매도가1", 0), errors="coerce").fillna(0)
    buy = pd.to_numeric(x.get("추천매수가", 0), errors="coerce").fillna(0)

    risk = (close - stop).clip(lower=1)
    reward = (tp1 - close).clip(lower=0)
    rr_now = reward / risk
    x["RR_NOW_TP1"] = rr_now.round(2)

    # (4) 진입갭
    entry_gap = ((close - buy).abs() / buy.clip(lower=1) * 100)
    x["ENTRY_GAP_PCT"] = entry_gap.round(1)

    # ═══ ELITE 공식 (v21.5) — 유지 ═══
    weighted_axis = s * 0.30 + t * 0.45 + m * 0.25
    bal_mult = 0.8 + 0.2 * (x["BALANCE_SCORE"] / 100)
    rr_gate = pd.Series(np.where(rr_now >= 0.8, 1.0, 0.3), index=x.index)
    elite = (weighted_axis * bal_mult * rr_gate).round(1)

    # 하드 게이트
    elite = elite.where(close > stop, 0)
    elite = elite.where(close < tp1, 0)
    x["ELITE_SCORE"] = elite

    # ELITE 사유
    def _reason(row):
        parts = []
        if row.get("BALANCE_SCORE", 0) >= 80:
            parts.append("밸런스우수")
        if row.get("RR_NOW_TP1", 0) >= 2.0:
            parts.append(f"RR{row['RR_NOW_TP1']:.1f}")
        elif row.get("RR_NOW_TP1", 0) >= 1.0:
            parts.append(f"RR{row['RR_NOW_TP1']:.1f}")
        if row.get("ENTRY_GAP_PCT", 99) <= 2.0:
            parts.append("진입적정")
        if row.get("AXIS_MEAN", 0) >= 80:
            parts.append("3축고점")
        r = row.get("ROUTE", "")
        if r == "ATTACK":
            parts.append("돌입")
        elif r == "ARMED":
            parts.append("대기")
        return " + ".join(parts) if parts else ""

    x["ELITE_REASON"] = x.apply(_reason, axis=1)

    # ═══ TOP_PICK — v22 이원화 ═══
    _pass_ebs = x.get("PASS_EBS", pd.Series(1, index=x.index)).fillna(1).astype(int)
    _turnover = pd.to_numeric(x.get("거래대금(억원)", 0), errors="coerce").fillna(0)
    _tp1_pct = ((tp1 - close) / close.clip(lower=1) * 100).round(1)
    x["TP1_PCT"] = _tp1_pct

    # [v22] ROUTE positive gate — WAIT/OVERHEAT/CARRY 등 자동 탈락
    _route_active = x["ROUTE"].isin(TOP_PICK_ROUTES)

    # 캘리브레이션 성숙도 — STABLE 활성화 조건
    # 우선순위: EST_WIN_RATE_MODE (v22 정식) → CALIBRATION_MODE (legacy) → "FALLBACK"
    _cal_mode = x.get(
        "EST_WIN_RATE_MODE",
        x.get("CALIBRATION_MODE", pd.Series("FALLBACK", index=x.index))
    )
    _cal_mature = (_cal_mode == "MATURE")

    # EST_WIN_RATE (누락이면 0 — STABLE 조건 자동 실패)
    _est_wr = pd.to_numeric(
        x.get("EST_WIN_RATE", pd.Series(0, index=x.index)),
        errors="coerce"
    ).fillna(0)

    # [v22.3] RR_NOW_TP1 hard gate — 손익비 1.0 미만 TOP_PICK 차단
    # 평가 피드백 96.5점 핵심 항목: "STABLE 타입이라도 RR<1.0이면 추천 자격 X"
    _rr_now = pd.to_numeric(
        x.get("RR_NOW_TP1", pd.Series(0, index=x.index)),
        errors="coerce"
    ).fillna(0)

    # 공통 하드게이트
    _hard_gate = (
        _route_active
        & (close > stop)
        & (close < tp1)
        & (_pass_ebs == 1)
        & (_turnover >= 50)
        & (entry_gap <= 5.0)
        & (_rr_now >= 1.0)  # [v22.3] 손익비 하한 강제
    )

    # AGGRESSIVE: 손익비 우선 (TP1 15%+)
    _aggressive = (
        _hard_gate
        & (x["ELITE_SCORE"] >= 75)
        & (s >= 80)
        & (t >= 70)
        & (_tp1_pct >= 15)
        & (x["BALANCE_SCORE"] >= 50)
    )

    # STABLE: 승률·밸런스 우선 (TP1 7~15, MATURE만)
    _stable = (
        _hard_gate
        & (x["ELITE_SCORE"] >= 70)
        & (_tp1_pct >= 7) & (_tp1_pct < 15)
        & (x["BALANCE_SCORE"] >= 70)
        & (_est_wr >= 0.55)
        & _cal_mature
    )

    # TOP_PICK = AGGRESSIVE OR STABLE (겹치면 AGGRESSIVE 우선)
    x["TOP_PICK"] = (_aggressive | _stable).astype(int)
    x["TOP_PICK_TYPE"] = np.where(
        _aggressive, "AGGRESSIVE",
        np.where(_stable, "STABLE", "")
    )

    # ═══════════════════════════════════════════════════
    # [v3.9.22a] BUY_NOW_* shadow 컬럼 — TOP_PICK은 건드리지 않음
    # ═══════════════════════════════════════════════════
    # 목적: "TOP_PICK 후보 중 즉시 매수 적합 여부" 분리 신호.
    #
    # 설계 원칙 (평가 명시):
    # 1. TOP_PICK 의미 무변경 — 기존 모든 소비처(market/portfolio/briefing/
    #    backtest/swap) baseline 보존
    # 2. HARD BLOCK + SOFT RISK SCORE 2단 구조 —
    #    AND 게이트 일괄 적용 시 0건화 위험 (검증: 패치 원본 10일/15건 → 1건)
    # 3. CSV/JSON에만 기록 — UI 노출은 v3.9.22b에서 별도 진행
    # 4. ROUTE fallback 차단은 auto_backtest.py에서 별도 채택
    #
    # 근거 데이터 (backtest_top3_trades_20260519):
    #   - 전체 116건: 평균 -2.12%, 51 LOSS, 1~2일 손절 23건
    #   - TOP_PICK이 점수만 높고 진입 타이밍 필터가 약함
    # ═══════════════════════════════════════════════════

    # 측정용 컬럼 (없으면 NaN, fillna로 보수적 처리)
    def _num_col(name: str, default=np.nan):
        return pd.to_numeric(
            x.get(name, pd.Series(default, index=x.index)),
            errors="coerce",
        )

    _r1 = _num_col("ret_1d_%")
    _r5 = _num_col("ret_5d_%")
    _vwap_gap = _num_col("VWAP_GAP")
    _poc_gap = _num_col("POC_GAP")
    _res_near = _num_col("RES_RATIO_NEAR")
    _mfi = _num_col("MFI14")
    _range_pos = _num_col("Range_Pos")
    # [v3.9.22c] 추가 입력 컬럼
    _ebs_num = pd.to_numeric(
        x.get("EBS", pd.Series(8, index=x.index)),
        errors="coerce",
    ).fillna(8)
    _disparity = pd.to_numeric(
        x.get("이격도", pd.Series(0, index=x.index)),
        errors="coerce",
    ).fillna(0)
    _entry_risk_level = (
        x.get("ENTRY_RISK_LEVEL", pd.Series("", index=x.index))
        .astype(str).str.strip().str.upper()
    )

    # ─── HARD BLOCK — 걸리면 절대 즉시매수 금지 ───
    # 기존 _hard_gate(_route_active, EBS, turnover, entry_gap≤5, RR≥1.0)는
    # 이미 TOP_PICK이 통과한 조건이라 중복 검사 안 함.
    # BUY_NOW 전용 HARD BLOCK은 진입 타이밍 위험만 다룸.
    #
    # [v3.9.22c-2/3] 추가 HARD BLOCK:
    #   - ENTRY_RISK_LEVEL=RED → 무조건 AVOID (B_red shadow RWF 5/5 통과)
    #   - 현대해상형 reversal-risk (단기 급등 + 음봉 + 과열)
    _bn_hard_block = (
        (entry_gap > 3.0)            # 추천가에서 3% 이상 떠있음
        | (_rr_now < 1.10)           # RR 턱걸이 차단
        | (_r5 > 25.0)               # 5일 +25% 이상 과열 추격 차단
        # [v3.9.22c-2] ENTRY_RISK_LEVEL=RED → AVOID
        | (_entry_risk_level == "RED")
        # [v3.9.22c-3] 현대해상형 reversal-risk (단기 급등 + 음봉)
        | ((_r5 > 20.0) & (_r1 < -5.0))
        | ((_disparity > 15.0) & (_r1 < -3.0))
    ).fillna(False)

    # ─── SOFT RISK SCORE — 감점 누적 ───
    # 각 위험 신호별 감점 (0~100 점수 차감)
    _risk_calc = pd.Series(0.0, index=x.index)

    # 칼날잡기 위험: 전일 -5% 이상 음봉
    _risk_calc = _risk_calc + (_r1 < -5.0).fillna(False).astype(float) * 20.0
    # 추격 위험: VWAP에서 너무 떠있음
    _risk_calc = _risk_calc + (_vwap_gap > 35.0).fillna(False).astype(float) * 25.0
    _risk_calc = _risk_calc + (
        (_vwap_gap > 20.0) & (_vwap_gap <= 35.0)
    ).fillna(False).astype(float) * 10.0
    # 매물대 이탈/과열: POC에서 너무 떠있음
    _risk_calc = _risk_calc + (_poc_gap > 80.0).fillna(False).astype(float) * 25.0
    _risk_calc = _risk_calc + (
        (_poc_gap > 40.0) & (_poc_gap <= 80.0)
    ).fillna(False).astype(float) * 10.0
    # 자금 과열: MFI 과매수
    _risk_calc = _risk_calc + (_mfi > 82.0).fillna(False).astype(float) * 15.0
    # 박스 하단/무기력
    _risk_calc = _risk_calc + (_range_pos < 0.40).fillna(False).astype(float) * 10.0
    # 손익비 약함 (1.10~1.20 경고 구간)
    _risk_calc = _risk_calc + (
        (_rr_now >= 1.10) & (_rr_now < 1.20)
    ).fillna(False).astype(float) * 10.0
    # 저항 여지 부족
    _risk_calc = _risk_calc + (_res_near < 0.03).fillna(False).astype(float) * 10.0

    # [v3.9.22c-2] ENTRY_RISK_LEVEL=ORANGE — soft penalty
    # 평가 명시: ORANGE 단독은 -10~-15 감점, 추가 위험 조합 시 AVOID는 HARD BLOCK
    _risk_calc = _risk_calc + (_entry_risk_level == "ORANGE").astype(float) * 15.0

    # [v3.9.22c-3] 현대해상형 reversal-risk (HARD 못 미치는 약한 신호)
    # 이격도 > 10 (HMA20 대비 10% 이상 떠있음)
    _risk_calc = _risk_calc + (
        (_disparity > 10.0) & (_disparity <= 15.0)
    ).fillna(False).astype(float) * 15.0
    # ret_5d 단기 급등 + 전일 음봉 (조정 시작)
    _risk_calc = _risk_calc + (
        (_r5 > 15.0) & (_r1 < -3.0) & ~((_r5 > 20.0) & (_r1 < -5.0))
    ).fillna(False).astype(float) * 15.0
    # MFI 과열 + 단기 급등
    _risk_calc = _risk_calc + (
        (_mfi > 75.0) & (_r5 > 15.0) & (_r1 < 0)
    ).fillna(False).astype(float) * 10.0

    _risk_calc = _risk_calc.clip(0, 100)

    # ─── BUY_NOW_SCORE = 100 - risk ───
    # HARD BLOCK 걸리면 무조건 0점
    _buy_now_score = (100.0 - _risk_calc).where(~_bn_hard_block, 0.0)

    x["BUY_NOW_SCORE"] = _buy_now_score.round(1)

    # ─── BUY_NOW_GRADE — 3단계 분류 ───
    # 🟢 매수 적합 ≥ 70
    # 🟡 관찰/눌림 대기 50~69
    # 🔴 추격 금지 < 50
    _grade = np.where(
        _buy_now_score >= 70.0, "BUY",
        np.where(_buy_now_score >= 50.0, "WATCH", "AVOID")
    )
    x["BUY_NOW_GRADE"] = _grade

    # ─── BUY_NOW_PASS — boolean (BUY 등급만) ───
    x["BUY_NOW_PASS"] = (x["BUY_NOW_GRADE"] == "BUY").astype(int)

    # ─── BUY_NOW_REASON — 어느 위험이 잡혔는지 (디버깅/UI 툴팁용) ───
    def _build_reason(row_idx):
        reasons = []
        if _bn_hard_block.iloc[row_idx]:
            if entry_gap.iloc[row_idx] > 3.0:
                reasons.append(f"진입괴리 {entry_gap.iloc[row_idx]:.1f}%")
            if _rr_now.iloc[row_idx] < 1.10:
                reasons.append(f"RR {_rr_now.iloc[row_idx]:.2f}")
            r5v = _r5.iloc[row_idx]
            if pd.notna(r5v) and r5v > 25.0:
                reasons.append(f"5일 +{r5v:.0f}%")
            # [v3.9.22c-2/3] 신규 HARD BLOCK 사유
            erl = _entry_risk_level.iloc[row_idx]
            if erl == "RED":
                reasons.append("진입위험 RED")
            r1v = _r1.iloc[row_idx]
            if (pd.notna(r5v) and pd.notna(r1v)
                    and r5v > 20.0 and r1v < -5.0):
                reasons.append(f"급등후 음봉 ({r5v:.0f}%/{r1v:.0f}%)")
            disp_v = _disparity.iloc[row_idx]
            if pd.notna(disp_v) and disp_v > 15.0 and pd.notna(r1v) and r1v < -3.0:
                reasons.append(f"이격도 {disp_v:.0f}↑ + 음봉")
        # SOFT 신호도 핵심만 표시
        r1v = _r1.iloc[row_idx]
        if pd.notna(r1v) and r1v < -5.0:
            reasons.append(f"전일 {r1v:.1f}%")
        vwapv = _vwap_gap.iloc[row_idx]
        if pd.notna(vwapv) and vwapv > 20.0:
            reasons.append(f"VWAP {vwapv:.0f}↑")
        pocv = _poc_gap.iloc[row_idx]
        if pd.notna(pocv) and pocv > 40.0:
            reasons.append(f"POC {pocv:.0f}↑")
        # [v3.9.22c-2] ORANGE 소프트 사유
        erl = _entry_risk_level.iloc[row_idx]
        if erl == "ORANGE":
            reasons.append("진입위험 ORANGE")
        return " · ".join(reasons[:3])  # 최대 3개

    x["BUY_NOW_REASON"] = [_build_reason(i) for i in range(len(x))]

    # ═══════════════════════════════════════════════════
    # [v3.9.22a 미니패치 2] 결측 critical 컬럼 보호
    # ═══════════════════════════════════════════════════
    # 평가 명시: VWAP_GAP/POC_GAP/MFI14/Range_Pos 중 일부 결측 시 위험 신호를
    # 못 잡고 BUY가 나올 수 있음. critical 컬럼 7개 중 2+ 결측이면 WATCH로 강등.
    _critical_cols_for_buy_now = [
        "ENTRY_GAP_PCT", "RR_NOW_TP1", "ret_5d_%",
        "VWAP_GAP", "POC_GAP", "MFI14", "Range_Pos",
    ]
    _missing_count = pd.Series(0, index=x.index)
    for _col in _critical_cols_for_buy_now:
        if _col not in x.columns:
            _missing_count = _missing_count + 1
        else:
            _missing_count = _missing_count + pd.to_numeric(
                x[_col], errors="coerce"
            ).isna().astype(int)

    _data_insufficient = _missing_count >= 2

    if _data_insufficient.any():
        # WATCH로 강등 (HARD BLOCK이 이미 AVOID로 만든 행은 그대로 둠)
        _was_avoid = x["BUY_NOW_GRADE"] == "AVOID"
        _to_downgrade = _data_insufficient & ~_was_avoid

        # SCORE는 max 60으로 캡
        x.loc[_to_downgrade, "BUY_NOW_SCORE"] = np.minimum(
            x.loc[_to_downgrade, "BUY_NOW_SCORE"].astype(float), 60.0
        )
        x.loc[_to_downgrade, "BUY_NOW_GRADE"] = "WATCH"
        x.loc[_to_downgrade, "BUY_NOW_PASS"] = 0
        # REASON 앞에 "데이터 부족" 추가
        for _idx in x.index[_to_downgrade]:
            _existing = str(x.at[_idx, "BUY_NOW_REASON"] or "")
            _new = "데이터 부족"
            if _existing:
                _new = f"{_new} · {_existing}"
            x.at[_idx, "BUY_NOW_REASON"] = _new

    # ═══════════════════════════════════════════════════
    # [v3.9.22a 미니패치 1] BUY_NOW_ELIGIBLE — TOP_PICK AND BUY_NOW_PASS
    # ═══════════════════════════════════════════════════
    # 평가 명시: BUY_NOW_PASS는 전체 종목에 찍혀서 TOP_PICK=0이지만 BUY_NOW_PASS=1
    # 인 종목이 CSV에 생김. PRIME 회원이 CSV 보면 오해 가능.
    # UI/CSV 소비처에서는 반드시 ELIGIBLE을 사용해야 함.
    x["BUY_NOW_ELIGIBLE"] = (
        (x["TOP_PICK"].astype(int) == 1)
        & (x["BUY_NOW_PASS"].astype(int) == 1)
    ).astype(int)

    # ═══════════════════════════════════════════════════
    # [v3.9.22c-3] EBS < 6 AND TOP_PICK != 1 → ELIGIBLE 강제 0
    # ═══════════════════════════════════════════════════
    # 평가 명시: 현대해상 같은 EBS 4/8 + TOP_PICK 미달 종목이
    # 다른 경로 (ELITE_LABEL 즉시진입 fallback)로 회원 화면에 노출되는 것을
    # ELIGIBLE 단에서도 한 번 더 차단.
    # 단 TOP_PICK=1은 이미 EBS≥5 게이트 통과한 종목이므로 영향 없음.
    _ebs_block = (_ebs_num < 6) & (x["TOP_PICK"].astype(int) != 1)
    x.loc[_ebs_block, "BUY_NOW_ELIGIBLE"] = 0

    # ─── 추가 액션 플래그 ───
    # NO_CHASE_FLAG: 추격 매수 금지 (VWAP/POC/5일 과열)
    x["NO_CHASE_FLAG"] = (
        ((_vwap_gap > 35.0) | (_poc_gap > 80.0) | (_r5 > 25.0))
        .fillna(False).astype(int)
    )
    # PULLBACK_WAIT_FLAG: 눌림 대기 권장 (HARD BLOCK 아니지만 SOFT 위험 있음)
    x["PULLBACK_WAIT_FLAG"] = (
        (~_bn_hard_block) & (_buy_now_score < 70.0) & (_buy_now_score >= 30.0)
    ).fillna(False).astype(int)

    # ═══════════════════════════════════════════════════════════════════
    # [v3.9.23a SHADOW] Anti-STRUCT Reversal — 측정 전용
    # ═══════════════════════════════════════════════════════════════════
    # 평가 명시: production 추천에 영향 없는 shadow 컬럼.
    # 22,646건 백테스트에서 발견된 "바닥 반등형" 알파 패턴 기록.
    #
    # ★ 절대 지킬 룰:
    #   1) TOP_PICK / BUY_NOW_ELIGIBLE / ROUTE / LDY_RANK 영향 없음
    #   2) 화면에 매수 신호로 표시 X (관리자 디버깅 라벨만)
    #   3) 4월 편향 가능성 / 데이터마이닝 위험 / 4월 시장국면 효과
    #      → 1~2주 shadow 측정 후 production 승격 결정
    #
    # 후보 기본 조건 (n=523, win=52.4%, ret +1.91%, Δalpha +11.3p):
    #   TIMING_SCORE >= 70 AND STRUCT_SCORE < 50
    #
    # 강화 조건 (n=61, win=59.0%, ret +4.83%, 단 4월 편향):
    #   + TIMING_SCORE >= 90
    #   + RR_NOW_TP1 < 0.5
    #
    # 주의 조건 (HARD EXCLUDE — 챔피언이라도 안 됨):
    #   - 이격도 > 15 (추격위험)
    #   - MFI14 > 82 (과열)
    #   - ret_1d > 10 (이미 점프)
    # ═══════════════════════════════════════════════════════════════════

    # FLAG: 기본 후보 조건만 (가장 안전한 기준)
    _struct_low = pd.to_numeric(
        x.get("STRUCT_SCORE", pd.Series(0, index=x.index)),
        errors="coerce",
    ).fillna(50)
    _timing_high = pd.to_numeric(
        x.get("TIMING_SCORE", pd.Series(0, index=x.index)),
        errors="coerce",
    ).fillna(0)

    # 주의 조건 — 챔피언이어도 빼야 할 종목 (이격도/MFI/단기점프)
    _anti_exclude = (
        (_disparity > 15.0)
        | (_mfi > 82.0)
        | (_r1 > 10.0)
    ).fillna(False)

    # FLAG: 기본 조건 통과 + 주의 조건 미해당
    x["ANTI_STRUCT_REVERSAL_FLAG"] = (
        (_timing_high >= 70.0)
        & (_struct_low < 50.0)
        & (~_anti_exclude)
    ).fillna(False).astype(int)

    # TYPE: 강도 분류
    # - BASIC: T≥70 AND S<50 (검증 1순위 — n=523 안정)
    # - STRONG: T≥85 AND S<50 (n=230)
    # - CHAMPION: T≥90 AND S<60 AND RR<0.5 (n=492 but 4월 편향)
    _type_basic = (_timing_high >= 70.0) & (_struct_low < 50.0) & ~_anti_exclude
    _type_strong = _type_basic & (_timing_high >= 85.0)
    _type_champion = (
        (_timing_high >= 90.0)
        & (_struct_low < 60.0)
        & (_rr_now < 0.5)
        & ~_anti_exclude
    )

    def _classify_type(idx):
        if not _type_basic.iloc[idx] and not _type_champion.iloc[idx]:
            return ""
        if _type_champion.iloc[idx]:
            return "CHAMPION"
        if _type_strong.iloc[idx]:
            return "STRONG"
        return "BASIC"

    x["ANTI_STRUCT_REVERSAL_TYPE"] = [
        _classify_type(i) for i in range(len(x))
    ]

    # SCORE: 0~100 추가 강도
    # BASIC=60, STRONG=75, CHAMPION=90, exclude=0
    _asr_score = pd.Series(0.0, index=x.index)
    _asr_score = _asr_score.mask(_type_basic, 60.0)
    _asr_score = _asr_score.mask(_type_strong, 75.0)
    _asr_score = _asr_score.mask(_type_champion, 90.0)
    x["ANTI_STRUCT_REVERSAL_SCORE"] = _asr_score

    # REASON: 디버깅/검증용 텍스트
    def _asr_reason(idx):
        if x["ANTI_STRUCT_REVERSAL_FLAG"].iloc[idx] == 0:
            # 왜 제외됐는지
            reasons = []
            if not (_timing_high.iloc[idx] >= 70.0):
                reasons.append(f"T={_timing_high.iloc[idx]:.0f}<70")
            if not (_struct_low.iloc[idx] < 50.0):
                reasons.append(f"S={_struct_low.iloc[idx]:.0f}≥50")
            if _anti_exclude.iloc[idx]:
                disp_v = _disparity.iloc[idx]
                mfi_v = _mfi.iloc[idx]
                r1_v = _r1.iloc[idx]
                if pd.notna(disp_v) and disp_v > 15.0:
                    reasons.append(f"이격도{disp_v:.0f}>15")
                elif pd.notna(mfi_v) and mfi_v > 82.0:
                    reasons.append(f"MFI{mfi_v:.0f}>82")
                elif pd.notna(r1_v) and r1_v > 10.0:
                    reasons.append(f"전일+{r1_v:.0f}%")
            return " · ".join(reasons[:2])
        # FLAG=1일 때 — 어떤 강화 조건 충족했나
        reasons = []
        t = _timing_high.iloc[idx]
        s = _struct_low.iloc[idx]
        rr = _rr_now.iloc[idx]
        reasons.append(f"T={t:.0f}·S={s:.0f}")
        if pd.notna(rr) and rr < 0.5:
            reasons.append(f"RR={rr:.2f}<0.5")
        elif pd.notna(rr) and rr < 1.0:
            reasons.append(f"RR={rr:.2f}<1.0")
        return " · ".join(reasons)

    x["ANTI_STRUCT_REVERSAL_REASON"] = [
        _asr_reason(i) for i in range(len(x))
    ]

    # funnel 메타 (단계별 통과 수 — 디버깅/튜닝용)
    def _funnel(masks_dict):
        return {k: int(v.sum()) for k, v in masks_dict.items()}

    aggressive_funnel = _funnel({
        "total": pd.Series(True, index=x.index),
        "route_active": _route_active,
        "hard_gate": _hard_gate,
        "elite_75": _hard_gate & (x["ELITE_SCORE"] >= 75),
        "struct_80": _hard_gate & (x["ELITE_SCORE"] >= 75) & (s >= 80),
        "timing_70": _hard_gate & (x["ELITE_SCORE"] >= 75) & (s >= 80) & (t >= 70),
        "tp1_15": _hard_gate & (x["ELITE_SCORE"] >= 75) & (s >= 80) & (t >= 70) & (_tp1_pct >= 15),
        "final": _aggressive,
    })
    stable_funnel = _funnel({
        "total": pd.Series(True, index=x.index),
        "route_active": _route_active,
        "hard_gate": _hard_gate,
        "elite_70": _hard_gate & (x["ELITE_SCORE"] >= 70),
        "tp1_7_15": _hard_gate & (x["ELITE_SCORE"] >= 70) & (_tp1_pct >= 7) & (_tp1_pct < 15),
        "balance_70": _hard_gate & (x["ELITE_SCORE"] >= 70) & (_tp1_pct >= 7) & (_tp1_pct < 15) & (x["BALANCE_SCORE"] >= 70),
        "wr_55": _hard_gate & (x["ELITE_SCORE"] >= 70) & (_tp1_pct >= 7) & (_tp1_pct < 15) & (x["BALANCE_SCORE"] >= 70) & (_est_wr >= 0.55),
        "mature": _hard_gate & (x["ELITE_SCORE"] >= 70) & (_tp1_pct >= 7) & (_tp1_pct < 15) & (x["BALANCE_SCORE"] >= 70) & (_est_wr >= 0.55) & _cal_mature,
        "final": _stable,
    })

    meta = {
        "aggressive_funnel": aggressive_funnel,
        "stable_funnel": stable_funnel,
        "top_pick_count": int(x["TOP_PICK"].sum()),
        "aggressive_count": int(_aggressive.sum()),
        "stable_count": int(_stable.sum()),
    }

    # funnel sidecar 저장 (trade_ymd + out_dir 제공 시)
    if out_dir and trade_ymd:
        try:
            import os, json
            sidecar = {
                "trade_ymd": trade_ymd,
                "total": len(x),
                **meta,
            }
            sidecar_path = os.path.join(out_dir, f"top_pick_funnel_{trade_ymd}.json")
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2, ensure_ascii=False, default=str)
            meta["sidecar_path"] = sidecar_path
        except Exception:
            pass   # sidecar 실패는 무해

    return x, meta
