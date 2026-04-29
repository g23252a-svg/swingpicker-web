# -*- coding: utf-8 -*-
"""
ticker_analyzer.py — analyze_ticker 분리 모듈 (SRP)
═══════════════════════════════════════════════════════
[v3.2] God Function(359줄) → 4개 단일 책임 함수 + 오케스트레이터

분리 원칙:
  1. prepare_ohlcv       — 데이터 정제 + 기본 필터링
  2. calculate_indicators — 기술적 지표 벡터 연산
  3. build_ticker_plan    — 진입/청산/수급 계산
  4. assemble_result      — 메타 병합 + 최종 딕셔너리 조립
  5. analyze_ticker_v2    — 위 4개를 호출하는 얇은 오케스트레이터

검증 방법:
  test_shadow_analyze.py --mode snapshot  (기존 코드로 golden 저장)
  → analyze_ticker를 analyze_ticker_v2로 교체
  → test_shadow_analyze.py --mode compare (100% 일치 확인)
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════
#  0. 중간 전달 객체 (함수 간 데이터 운반)
# ═══════════════════════════════════════════════════

@dataclass
class OHLCVContext:
    """정제된 OHLCV + 기본 메타"""
    code6: str
    ohlcv: pd.DataFrame
    c: pd.Series      # 종가
    h: pd.Series      # 고가
    l: pd.Series      # 저가
    o: pd.Series      # 시가
    v: pd.Series      # 거래량
    last_c: float
    tv_eok: float      # 거래대금(억원)
    mcap: float        # 시가총액(억원)


@dataclass
class Indicators:
    """기술적 지표 계산 결과"""
    # 구조적 상태
    low_trend_pct: float
    rsi: float
    rsi_rising: int
    vol_quality: float
    bb_bw_val: float
    bb_expanding: int
    range_pos: float
    bw_squeeze: int
    ttm_squeeze: int
    sqz_cnt: int
    mfi: float
    disp: float        # 이격도 (MA20 대비)

    # 수익률
    ret_1d: float
    ret_5: float
    ret_10: float
    ret_20: float
    ret_60: float
    ret_120: float

    # 트리거
    trigger_str: str

    # VWAP / SuperTrend / V-Power
    vwap_val: float
    vwap_gap: float
    st_val: float
    st_trend: int
    v_power: float
    vol_z: float

    # Swing / Candle
    swing_low_10: float
    dist_to_swing: float
    is_swing_support: bool

    # HMA / 주봉
    curr_hma: float
    hma_trend_up: bool
    is_above_w20: bool
    is_w20_up: bool

    # MACD
    slope_pct: float
    hist: pd.Series    # MACD histogram (build_plan에서 사용)

    # 매물대
    poc_p: float
    res_all: float
    res_near: float
    near_pct: float
    is_above_poc: int
    poc_gap: float

    # BB 시리즈 (중간 계산용)
    ma20: pd.Series
    bb_upper: pd.Series
    bb_lower: pd.Series
    atr_series: pd.Series

    # 갭/진입 필터용
    gap_pct_val: float

    # [Fix] 누락 컬럼 — scoring_engine / validation에서 기대하는 필드
    data_length: int = 0                 # OHLCV 행 수 (_data_length)
    consecutive_limit_up: int = 0        # 연속 상한가 횟수
    mtf_weekly_trend: int = 0            # 주봉 추세 (+1/0/-1)
    mtf_monthly_trend: int = 0           # 월봉 추세 (+1/0/-1)
    mtf_data_sufficient: int = 0         # MTF 데이터 충분 여부 (1/0)


# ═══════════════════════════════════════════════════
#  1. 데이터 정제 + 기본 필터링
# ═══════════════════════════════════════════════════

def prepare_ohlcv(
    t: str,
    ohlcv_df: pd.DataFrame,
    top_df: pd.DataFrame,
    mcap_map: Dict[str, float],
    lookback_days: int,
    min_mcap_eok: float,
    min_turnover_eok: float,
    get_mcap_fn,
) -> Optional[OHLCVContext]:
    """
    OHLCV 정제 + 거래대금/시총 필터링.
    
    Returns: OHLCVContext or None (필터 탈락)
    """
    code6 = str(t).zfill(6)
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 120:
        return None

    from stop_logic import sanitize_ohlcv
    ohlcv = ohlcv_df.tail(lookback_days).copy()
    ohlcv = sanitize_ohlcv(ohlcv)
    if len(ohlcv) < 60:
        return None

    # 데이터 타입 강제 변환
    price_cols = ["종가", "고가", "저가", "시가"]
    for col in price_cols:
        if col in ohlcv.columns:
            ohlcv[col] = pd.to_numeric(ohlcv[col], errors='coerce')
    ohlcv[price_cols] = ohlcv[price_cols].ffill()
    ohlcv = ohlcv.dropna(subset=["종가"])

    if "거래량" in ohlcv.columns:
        ohlcv["거래량"] = pd.to_numeric(ohlcv["거래량"], errors='coerce').fillna(0)

    if len(ohlcv) < 60:
        return None

    c = ohlcv["종가"]
    h = ohlcv["고가"]
    l = ohlcv["저가"]
    v = ohlcv["거래량"]
    o = ohlcv["시가"]
    last_c = float(c.iloc[-1])

    # 거래대금 추출
    tv_row = top_df.loc[top_df["종목코드"] == code6, "거래대금(원)"]
    tv_eok = float(tv_row.values[0]) / 1e8 if not tv_row.empty else 0.0

    if tv_eok <= 0:
        if "거래대금" in ohlcv.columns:
            try:
                tv_eok = float(pd.to_numeric(ohlcv["거래대금"].iloc[-1], errors='coerce')) / 1e8
            except (ValueError, TypeError):
                pass
        if tv_eok <= 0:
            try:
                tv_eok = (last_c * float(v.iloc[-1])) / 1e8
            except (ValueError, TypeError):
                pass

    mcap = get_mcap_fn(mcap_map, code6)
    if mcap_map and mcap > 0 and mcap < min_mcap_eok:
        return None
    if tv_eok < min_turnover_eok:
        return None

    return OHLCVContext(
        code6=code6, ohlcv=ohlcv,
        c=c, h=h, l=l, o=o, v=v,
        last_c=last_c, tv_eok=tv_eok, mcap=mcap,
    )


# ═══════════════════════════════════════════════════
#  2. 기술적 지표 계산
# ═══════════════════════════════════════════════════

def calculate_indicators(
    ctx: OHLCVContext,
    bb_period: int, bb_std: float, bb_squeeze_bw: float,
    kc_period: int, kc_atr_period: int, kc_mult: float,
    calc_rsi_fn, calc_mfi_fn, calc_atr_fn, calc_vwap_fn,
    calc_supertrend_fn, calc_hma_fn, check_candle_fn,
    calc_volume_profile_fn,
    ema_fn,
) -> Indicators:
    """모든 기술적 지표를 1회 벡터 연산으로 계산."""
    c, h, l, o, v = ctx.c, ctx.h, ctx.l, ctx.o, ctx.v
    ohlcv = ctx.ohlcv
    last_c = ctx.last_c

    # --- 벡터 사전 계산 ---
    ohlcv['gap_pct'] = (o / c.shift(1) - 1) * 100
    ohlcv['candle_rng'] = h - l
    ohlcv['upper_shadow_ratio'] = (h - c) / ohlcv['candle_rng'].replace(0, 1)
    ohlcv['vol_ma_5'] = v.rolling(window=5).mean()
    ohlcv['vol_ratio'] = v / ohlcv['vol_ma_5'].replace(0, 1)

    # (1) Low Trend
    min_l_prev = float(l.iloc[-20:-10].min())
    min_l_curr = float(l.iloc[-10:].min())
    low_trend_pct = (min_l_curr - min_l_prev) / min_l_prev * 100 if min_l_prev > 0 else 0.0

    # (2) RSI
    rsi_s = calc_rsi_fn(c, 14)
    rsi = float(rsi_s.iloc[-1])
    rsi_min_prev = float(rsi_s.iloc[-20:-10].min())
    rsi_min_curr = float(rsi_s.iloc[-10:].min())
    rsi_rising = 1 if rsi_min_curr > rsi_min_prev else 0

    # (3) Volume Quality
    is_red = c > o
    vol_red_avg = v[is_red].tail(20).mean()
    vol_blue_avg = v[~is_red].tail(20).mean()
    vol_quality = vol_red_avg / vol_blue_avg if vol_blue_avg > 0 else (1.5 if vol_red_avg > 0 else 1.0)

    # (4) Bollinger Bands
    ma20 = c.rolling(bb_period).mean()
    std20 = c.rolling(bb_period).std()
    bb_upper = ma20 + (bb_std * std20)
    bb_lower = ma20 - (bb_std * std20)
    bb_bw = ((bb_upper - bb_lower) / ma20.replace(0, np.nan)) * 100
    bb_bw_val = float(bb_bw.iloc[-1])
    bb_bw_prev = float(bb_bw.iloc[-5])
    bb_expanding = 1 if (bb_bw_val > bb_bw_prev * 1.05) and (bb_bw_val < 20) else 0

    # Range Position
    period_high_20 = float(h.tail(20).max())
    period_low_20 = float(l.tail(20).min())
    denom = period_high_20 - period_low_20
    range_pos = (last_c - period_low_20) / denom if denom > 0 else 0.5

    # Squeeze
    bw_squeeze = 1 if (np.isfinite(bb_bw_val) and bb_bw_val < bb_squeeze_bw) else 0
    atr_series = calc_atr_fn(h, l, c, kc_atr_period)
    kc_mid = ema_fn(c, kc_period)
    kc_upper = kc_mid + (kc_mult * atr_series)
    kc_lower = kc_mid - (kc_mult * atr_series)
    ttm_series = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    ttm_squeeze = 1 if bool(ttm_series.iloc[-1]) else 0
    sqz_cnt = int(ttm_series.iloc[-5:].sum())

    # MACD / MFI
    mfi = float(calc_mfi_fn(h, l, c, v, 14).iloc[-1])
    macd = ema_fn(c, 12) - ema_fn(c, 26)
    sig = ema_fn(macd, 9)
    hist = macd - sig

    # 이격도
    disp = (last_c / float(ma20.iloc[-1]) - 1.0) * 100

    # Returns
    def _ret(d):
        return (last_c / float(c.iloc[-(d + 1)]) - 1.0) * 100 if len(c) >= d + 1 else np.nan

    ret_1d_val = (last_c / float(c.iloc[-2]) - 1.0) * 100 if len(c) > 1 else 0.0
    ret_5 = _ret(5); ret_10 = _ret(10); ret_20 = _ret(20)
    ret_60 = _ret(60); ret_120 = _ret(120)

    # Triggers
    triggers = []
    if (low_trend_pct > 0.5) and (rsi_rising == 1) and (range_pos > 0.75) and (vol_quality > 1.1):
        if rsi < 70:
            triggers.append("🚀급등시동")
    if (-2 <= disp <= 3) and (c.iloc[-1] > o.iloc[-1]) and (low_trend_pct >= 0):
        triggers.append("⚡눌림회복")
    if (ttm_squeeze == 0) and (sqz_cnt >= 3) and (bb_expanding == 1):
        triggers.append("📦박스돌파")
    trigger_str = "/".join(triggers) if triggers else ""

    # VWAP / SuperTrend
    vwap_val = calc_vwap_fn(ohlcv.tail(60))  # [v4.0] 20일 윈도우 VWAP용 충분한 데이터
    vwap_gap = (last_c - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0.0
    st_series, st_dir = calc_supertrend_fn(h, l, c, 10, 3.0)
    st_val = float(st_series.iloc[-1])
    st_trend = int(st_dir.iloc[-1])

    # V-Power / Vol-Z
    tail5 = ohlcv.tail(5).copy()
    body = (tail5["종가"] - tail5["시가"]).abs()
    range_len = (tail5["고가"] - tail5["저가"]).replace(0, 1)
    sign = np.where(tail5["종가"] >= tail5["시가"], 1, -1)
    power_raw = (body / range_len) * tail5["거래량"] * sign
    avg_vol = tail5["거래량"].mean()
    v_power = power_raw.sum() / avg_vol if avg_vol > 0 else 0.0
    vol_z = float((v / v.rolling(20).mean().replace(0, np.nan)).iloc[-1]) if len(v) else 0.0

    # Candle / Swing
    _ = check_candle_fn(o, h, l, c)
    swing_low_10 = float(l.tail(10).min())
    dist_to_swing = (last_c - swing_low_10) / last_c * 100
    is_swing_support = (dist_to_swing < 5.0) and (last_c > swing_low_10)

    # HMA
    hma20 = calc_hma_fn(c, 20)
    curr_hma = float(hma20.iloc[-1]) if len(hma20) > 0 else 0
    prev_hma = float(hma20.iloc[-2]) if len(hma20) > 1 else 0
    hma_trend_up = curr_hma > prev_hma

    # 주봉
    is_above_w20 = False
    is_w20_up = False
    try:
        w_res = ohlcv.resample('W').last()
        w_ma = w_res['종가'].rolling(20).mean()
        is_above_w20 = w_res['종가'].iloc[-1] > w_ma.iloc[-1]
        is_w20_up = w_ma.iloc[-1] > w_ma.iloc[-2]
    except Exception:
        pass

    # MACD Slope
    slope = float(np.polyfit(np.arange(len(hist.tail(5))), hist.tail(5).values.astype(float), 1)[0]) if len(hist) >= 5 else 0.0
    slope_pct = (slope / last_c) * 100.0 if last_c > 0 else 0.0

    # 매물대
    poc_p, res_all, res_near, near_pct = calc_volume_profile_fn(ohlcv.tail(120))
    is_above_poc = 1 if (poc_p is not None and last_c > poc_p) else 0
    poc_gap = round((last_c - poc_p) / poc_p * 100, 2) if poc_p else 0

    gap_pct_val = float(ohlcv['gap_pct'].iloc[-1]) if 'gap_pct' in ohlcv.columns else 0.0

    # --- [Fix 1] 누락 컬럼 계산 ---
    # (a) 데이터 길이
    data_length = len(ohlcv)

    # (b) 연속 상한가 (종가 대비 +29% 이상을 상한가로 간주, 역순 카운트)
    _consecutive_limit_up = 0
    if len(c) >= 2:
        _daily_ret = c.pct_change() * 100
        for _r in _daily_ret.iloc[::-1]:
            if _r >= 29.0:
                _consecutive_limit_up += 1
            else:
                break
    consecutive_limit_up = _consecutive_limit_up

    # (c) MTF: 주봉/월봉 추세
    mtf_weekly_trend = 0
    mtf_monthly_trend = 0
    mtf_data_sufficient = 0
    try:
        # 주봉
        w_res = ohlcv.resample('W').last().dropna(subset=['종가'])
        if len(w_res) >= 26:
            w_ma = w_res['종가'].rolling(20).mean()
            if len(w_ma.dropna()) >= 2:
                mtf_weekly_trend = 1 if w_ma.iloc[-1] > w_ma.iloc[-2] else -1
        # 월봉
        m_res = ohlcv.resample('ME').last().dropna(subset=['종가'])
        if len(m_res) >= 12:
            m_ma = m_res['종가'].rolling(6).mean()
            if len(m_ma.dropna()) >= 2:
                mtf_monthly_trend = 1 if m_ma.iloc[-1] > m_ma.iloc[-2] else -1
        # 데이터 충분성
        if len(w_res) >= 26 and len(m_res) >= 12:
            mtf_data_sufficient = 1
    except Exception:
        pass

    return Indicators(
        low_trend_pct=low_trend_pct, rsi=rsi, rsi_rising=rsi_rising,
        vol_quality=vol_quality, bb_bw_val=bb_bw_val, bb_expanding=bb_expanding,
        range_pos=range_pos, bw_squeeze=bw_squeeze, ttm_squeeze=ttm_squeeze,
        sqz_cnt=sqz_cnt, mfi=mfi, disp=disp,
        ret_1d=ret_1d_val, ret_5=ret_5, ret_10=ret_10,
        ret_20=ret_20, ret_60=ret_60, ret_120=ret_120,
        trigger_str=trigger_str,
        vwap_val=vwap_val, vwap_gap=vwap_gap,
        st_val=st_val, st_trend=st_trend,
        v_power=v_power, vol_z=vol_z,
        swing_low_10=swing_low_10, dist_to_swing=dist_to_swing,
        is_swing_support=is_swing_support,
        curr_hma=curr_hma, hma_trend_up=hma_trend_up,
        is_above_w20=is_above_w20, is_w20_up=is_w20_up,
        slope_pct=slope_pct, hist=hist,
        poc_p=poc_p, res_all=res_all, res_near=res_near, near_pct=near_pct,
        is_above_poc=is_above_poc, poc_gap=poc_gap,
        ma20=ma20, bb_upper=bb_upper, bb_lower=bb_lower, atr_series=atr_series,
        gap_pct_val=gap_pct_val,
        data_length=data_length,
        consecutive_limit_up=consecutive_limit_up,
        mtf_weekly_trend=mtf_weekly_trend,
        mtf_monthly_trend=mtf_monthly_trend,
        mtf_data_sufficient=mtf_data_sufficient,
    )


# ═══════════════════════════════════════════════════
#  3. 진입/청산/수급 계산
# ═══════════════════════════════════════════════════

@dataclass
class TradePlanResult:
    """build_trade_plan 결과"""
    buy: float
    stop: float
    target: float
    tp2: float
    actual_stop_pct: float
    max_loss_pct: float
    rr_mult: float
    stop_reason: str
    entry_action: str
    position_pct: float
    exec_rule_id: str
    # 수급
    frg_net_val: int
    inst_net_val: int
    major_net: int
    # [v4.0] 도달 확률 기반 목표가 메타
    tp1_method: str = "RR_LEGACY"
    tp1_prob: int = 50
    tp2_method: str = "RR_LEGACY"
    tp2_prob: int = 35
    tp3: float = 0.0
    tp3_method: str = ""
    tp3_prob: int = 0
    # [Phase 3+4 v3] 추천 row 메타 (이전엔 stop_reason만 보존되고 plan_reason 등 누락됐음)
    plan_reason: str = ""           # NORMAL / GAP / SWING / EST_MCAP 등 (TradePlan.plan_reason)
    regime: str = "normal"           # normal / high_vol / low_vol
    time_stop_days: int = 0          # 0 = 비활성

    def to_recommend_row(self) -> Dict[str, Any]:
        """[Phase 3+4 v4] recommend_latest.csv용 한글+메타 키 생성 — SSOT.

        assemble_result()에서 row.update()로 병합. 이게 진짜 SSOT 강제:
          - 가격(추천매수가/손절가/추천매도가1/2/3)
          - 진입 제어(ENTRY_ACTION/POSITION_PCT/EXEC_RULE_ID)
          - 운영 메타(PLAN_REASON/STOP_PCT/MAX_LOSS_PCT/RR_MULT/REGIME/TIME_STOP_DAYS)
          - TP 도달 확률 메타(TP1_METHOD/PROB ... TP3_METHOD/PROB)
          - STOP_REASON

        모든 키는 validate_recommend_row()에 정의된 계약을 만족하도록 출력됨.

        나머지 ticker_analyzer 고유 컬럼들(시장/종목명/지표/수급 등)은
        assemble_result에서 별도로 매핑하여 row에 합쳐짐.
        """
        return {
            # ── 가격 (한글 SSOT) ──
            "추천매수가": self.buy,
            "손절가": self.stop,
            "추천매도가1": self.target,
            "추천매도가2": self.tp2,
            "추천매도가3": self.tp3 if self.tp3 > 0 else None,
            # ── 진입 제어 (영문 메타) ──
            "ENTRY_ACTION": self.entry_action,
            "POSITION_PCT": self.position_pct,
            # ── 운영 메타 (영문 — REQUIRED_RECOMMEND_META_KEYS) ──
            "PLAN_REASON": self.plan_reason,
            "STOP_PCT": round(self.actual_stop_pct, 2),
            "MAX_LOSS_PCT": round(self.max_loss_pct, 1),
            "RR_MULT": round(self.rr_mult, 1),
            "REGIME": self.regime,
            "TIME_STOP_DAYS": self.time_stop_days,
            # ── 체결 메타 ──
            "EXEC_RULE_ID": self.exec_rule_id,
            "STOP_REASON": self.stop_reason,
            # ── TP 도달 확률 메타 ──
            "TP1_METHOD": self.tp1_method, "TP1_PROB": self.tp1_prob,
            "TP2_METHOD": self.tp2_method, "TP2_PROB": self.tp2_prob,
            "TP3_METHOD": self.tp3_method if self.tp3 > 0 else "",
            "TP3_PROB": self.tp3_prob if self.tp3 > 0 else 0,
        }


def build_ticker_plan(
    ctx: OHLCVContext,
    ind: Indicators,
    inv_maps: Optional[Dict[str, Dict[str, int]]],
) -> TradePlanResult:
    """진입/청산가, 수급 데이터, 트레이딩 플랜 산출."""
    last_c = ctx.last_c
    code6 = ctx.code6

    # 수급
    major_net = 0
    frg_net_val = 0
    inst_net_val = 0
    major_ratio = 0.0
    if inv_maps:
        frg_net_val = inv_maps.get("frg", {}).get(code6, 0)
        inst_net_val = inv_maps.get("inst", {}).get(code6, 0)
        major_net = frg_net_val + inst_net_val
        tv_won = ctx.tv_eok * 1e8
        major_ratio = abs(major_net) / tv_won if tv_won > 0 else 0.0

    # 진입가 보수화
    buy = last_c
    if ind.disp >= 15.0:
        buy = float(ind.ma20.iloc[-1]) * 1.05
    elif ind.ret_1d >= 7.0:
        mid_body = (float(ctx.o.iloc[-1]) + float(ctx.c.iloc[-1])) / 2
        support_level = float(ctx.l.iloc[-1]) + (float(ctx.h.iloc[-1]) - float(ctx.l.iloc[-1])) * 0.3
        buy = max(mid_body, support_level)
    elif ind.ret_1d >= 3.0:
        buy = last_c * 0.985

    # Trade Plan (SSOT)
    from trade_plan import build_trade_plan as _build_plan
    from trade_plan import ExecRule as _ExecRule, estimate_slippage_bps as _est_slip
    from collector_config import DEFAULT_CONFIG as _cfg

    _slip_bps = _est_slip(ctx.tv_eok, _cfg)
    _exec_rule = _ExecRule(sl_slippage_bps=_slip_bps, tp_slippage_bps=max(5.0, _slip_bps * 0.3))

    atr_val = float(ind.atr_series.iloc[-1]) if len(ind.atr_series) else last_c * 0.03
    today_low_val = float(ctx.l.iloc[-1])

    _plan = _build_plan(
        buy=buy, atr_val=atr_val, last_c=last_c,
        mcap=ctx.mcap, tv_eok=ctx.tv_eok,
        today_low=today_low_val, gap_up_pct=ind.gap_pct_val,
        swing_low_10=ind.swing_low_10, dist_to_swing=ind.dist_to_swing,
        ret_1d=ind.ret_1d, gap_pct=ind.gap_pct_val,
        major_net=major_net, major_ratio=major_ratio,
        exec_rule=_exec_rule,
        # [v20.8] PolicyConfig end-to-end: 추가 방어 인자 전달
        rsi14=ind.rsi,
        consecutive_limit_up=ind.consecutive_limit_up,
    )

    # Time Stop
    if _cfg.time_stop_days > 0:
        import dataclasses as _dc
        _plan = _dc.replace(_plan,
            time_stop_days=_cfg.time_stop_days,
            time_stop_min_move_pct=_cfg.time_stop_min_move_pct,
            time_stop_extend_if_profit=_cfg.time_stop_extend_if_profit,
        )

    # [v4.0] 도달 확률 기반 목표가 — 기존 RR 배수를 기술적 저항 레벨로 교체
    _tp1_method = "RR_LEGACY"
    _tp1_prob = 50
    _tp2_method = "RR_LEGACY"
    _tp2_prob = 35
    _tp3_val = 0.0
    _tp3_method = ""
    _tp3_prob = 0
    _final_tp1 = _plan.tp1
    _final_tp2 = _plan.tp2 if _plan.tp2 else _plan.tp1 * 1.1

    try:
        from stop_logic import compute_realistic_targets as _crt
        _rt = _crt(
            ohlcv=ctx.ohlcv,
            entry=_plan.entry,
            stop=_plan.stop,
            poc_p=ind.poc_p if ind.poc_p else 0.0,
            res_ratio=ind.res_all,
            res_ratio_near=ind.res_near,
            use_tick=True,
        )
        if _rt.get("TP1", 0) > _plan.entry:
            _final_tp1 = _rt["TP1"]
            _tp1_method = _rt.get("TP1_METHOD", "RR_LEGACY")
            _tp1_prob = _rt.get("TP1_PROB", 50)
        if _rt.get("TP2", 0) > _final_tp1:  # TP2는 반드시 TP1보다 위
            _final_tp2 = _rt["TP2"]
            _tp2_method = _rt.get("TP2_METHOD", "RR_LEGACY")
            _tp2_prob = _rt.get("TP2_PROB", 35)
        elif _rt.get("TP2", 0) > _plan.entry and _rt.get("TP2", 0) <= _final_tp1:
            # TP2가 TP1 이하면, 기존 RR 기반 tp2를 유지
            pass
        if _rt.get("TP3", 0) > _final_tp2:  # TP3는 반드시 TP2보다 위
            _tp3_val = _rt["TP3"]
            _tp3_method = _rt.get("TP3_METHOD", "")
            _tp3_prob = _rt.get("TP3_PROB", 0)
    except (ImportError, Exception) as _tp_err:
        # [Phase 3+4 v4] silent → logger 변경. 추적 가능하게.
        # compute_realistic_targets 실패 시 RR_LEGACY로 fallback됨.
        import logging as _logging
        _logging.getLogger("ticker_analyzer").warning(
            f"compute_realistic_targets 실패 → RR_LEGACY fallback: "
            f"{ctx.code6} / {type(_tp_err).__name__}: {_tp_err}"
        )

    # RR 재계산 (새 TP1 기준)
    _risk = _plan.entry - _plan.stop

    # ── [v19.2] TP 단조성 강제: TP1 < TP2 < TP3 ──
    # 어떤 경로로 왔든, 최종 출력 직전에 단조 증가를 보장
    if _final_tp2 <= _final_tp1:
        _final_tp2 = max(_final_tp1 * 1.05, _final_tp1 + _risk * 0.5) if _risk > 0 else _final_tp1 * 1.05
        _tp2_method = _tp2_method if _tp2_method else "MONO_GUARD"
    if _tp3_val > 0 and _tp3_val <= _final_tp2:
        _tp3_val = max(_final_tp2 * 1.05, _final_tp2 + _risk * 0.5) if _risk > 0 else _final_tp2 * 1.05
        _tp3_method = _tp3_method if _tp3_method else "MONO_GUARD"

    # 호가 단위 반올림
    from stop_logic import ceil_to_tick as _ceil_tick
    _final_tp1 = float(_ceil_tick(_final_tp1))
    _final_tp2 = float(_ceil_tick(_final_tp2))
    if _tp3_val > 0:
        _tp3_val = float(_ceil_tick(_tp3_val))

    _new_rr = (_final_tp1 - _plan.entry) / _risk if _risk > 0 else _plan.rr_mult

    return TradePlanResult(
        buy=_plan.entry, stop=_plan.stop, target=_final_tp1,
        tp2=_final_tp2,
        actual_stop_pct=_plan.stop_pct, max_loss_pct=_plan.max_loss_pct,
        rr_mult=round(_new_rr, 1), stop_reason=_plan.plan_reason,
        entry_action=_plan.entry_action, position_pct=_plan.position_pct,
        exec_rule_id=_plan.exec_rule_id,
        frg_net_val=frg_net_val, inst_net_val=inst_net_val, major_net=major_net,
        tp1_method=_tp1_method, tp1_prob=_tp1_prob,
        tp2_method=_tp2_method, tp2_prob=_tp2_prob,
        tp3=_tp3_val, tp3_method=_tp3_method, tp3_prob=_tp3_prob,
        # [Phase 3+4 v3] 추천 row 메타 보존
        plan_reason=_plan.plan_reason,
        regime=_plan.regime,
        time_stop_days=_plan.time_stop_days,
    )


# ═══════════════════════════════════════════════════
#  4. 최종 딕셔너리 조립
# ═══════════════════════════════════════════════════

def assemble_result(
    ctx: OHLCVContext,
    ind: Indicators,
    plan: TradePlanResult,
    name_map: Dict[str, str],
    sector_map: Dict[str, str],
    top_df: pd.DataFrame,
    kospi_set: set,
    kosdaq_set: set,
    bench_map: Dict[str, Dict[int, float]],
) -> Dict[str, Any]:
    """메타 정보 병합 + 기존과 100% 동일한 딕셔너리 반환."""
    code6 = ctx.code6
    last_c = ctx.last_c

    sector = sector_map.get(code6, "기타")
    name = name_map.get(code6, code6)
    m_row = top_df.loc[top_df["종목코드"] == code6, "시장"]
    market = str(m_row.values[0]) if not m_row.empty else ("KOSPI" if code6 in kospi_set else "KOSDAQ")

    bench_dict = bench_map.get(market, {})
    idx_20 = bench_dict.get(20, np.nan)
    idx_60 = bench_dict.get(60, np.nan)
    idx_120 = bench_dict.get(120, np.nan)
    rel_20 = ind.ret_20 - idx_20 if np.isfinite(idx_20) and np.isfinite(ind.ret_20) else np.nan
    rel_60 = ind.ret_60 - idx_60 if np.isfinite(idx_60) and np.isfinite(ind.ret_60) else np.nan
    rel_120 = ind.ret_120 - idx_120 if np.isfinite(idx_120) and np.isfinite(ind.ret_120) else np.nan

    row = {
        "시장": market, "종목명": name, "종목코드": code6, "업종": sector, "종가": int(last_c),
        "거래대금(억원)": round(ctx.tv_eok, 2), "시가총액(억원)": round(ctx.mcap, 1),
        "거래대금(원)": round(float(ctx.tv_eok if ctx.tv_eok is not None and not np.isnan(ctx.tv_eok) else 0.0) * 1e8, 0),
        "RSI14": round(ind.rsi, 1), "MFI14": round(ind.mfi, 1), "이격도": round(ind.disp, 2),
        "BB_BW": round(ind.bb_bw_val, 2), "TTM_SQUEEZE": int(ind.ttm_squeeze),
        "TTM_SQUEEZE_CNT": ind.sqz_cnt, "BB_SQUEEZE_BW": int(ind.bw_squeeze),
        "ret_1d_%": round(ind.ret_1d, 2),
        "ret_5d_%": round(ind.ret_5, 2), "ret_10d_%": round(ind.ret_10, 2),
        "ret_20d_%": round(ind.ret_20, 2), "ret_60d_%": round(ind.ret_60, 2),
        "ret_120d_%": round(ind.ret_120, 2),
        "rel_20d_%": round(rel_20, 2), "rel_60d_%": round(rel_60, 2), "rel_120d_%": round(rel_120, 2),
        "RES_RATIO": round(ind.res_all, 3),
        "RES_RATIO_NEAR": round(ind.res_near, 3),
        "IS_ABOVE_POC": ind.is_above_poc,
        "POC_GAP": ind.poc_gap,
        "NEAR_THRES": round(ind.near_pct, 1),
        "Low_Trend_PCT": round(ind.low_trend_pct, 2),
        "RSI_Rising": int(ind.rsi_rising),
        "BB_Expanding": int(ind.bb_expanding),
        "Vol_Quality": round(ind.vol_quality, 2),
        "Range_Pos": round(ind.range_pos, 2),
        # [Phase 3+4 v4] 가격/체결/메타는 SSOT인 plan.to_recommend_row()에서 채움
        # (아래 row.update(plan.to_recommend_row()) 호출로 병합)
        "외인순매수": plan.frg_net_val,
        "기관순매수": plan.inst_net_val,
        "메이저순매수": plan.major_net,
        "TRIGGER": ind.trigger_str,
        "Above_MA20": 1 if ind.disp > 0 else 0,
        "SUPERTREND_DIR": ind.st_trend, "SUPERTREND_VAL": ind.st_val,
        "VWAP": int(ind.vwap_val), "VWAP_GAP": round(ind.vwap_gap, 2),
        "MACD_Slope_PCT": round(ind.slope_pct, 4),
        "거래강도": round(ind.vol_z, 2), "V_POWER": round(ind.v_power, 2),
        "IS_SWING_SUPPORT": ind.is_swing_support,
        "주봉20선_상회": "O" if ind.is_above_w20 else "X",
        "주봉추세": "▲" if ind.is_w20_up else "▼",
        "HMA20": int(ind.curr_hma),
        "HMA_Trend": "▲" if ind.hma_trend_up else "▼",
        "HMA_On": "O" if last_c > ind.curr_hma else "X",
        "OBV_Div": "X",
        # [Fix 1] 누락 컬럼 — scoring_engine / validation 연결
        "gap_pct": round(ind.gap_pct_val, 2),
        "_data_length": ind.data_length,
        "consecutive_limit_up": ind.consecutive_limit_up,
        "MTF_WEEKLY_TREND": ind.mtf_weekly_trend,
        "MTF_MONTHLY_TREND": ind.mtf_monthly_trend,
        "MTF_DATA_SUFFICIENT": ind.mtf_data_sufficient,
    }

    # [Phase 3+4 v4] SSOT 병합 — 가격/체결/운영 메타 18개 키
    # 이게 진짜 SSOT 강제: ticker_analyzer가 직접 매핑하지 않고
    # TradePlanResult.to_recommend_row()로 통일.
    row.update(plan.to_recommend_row())

    # [Phase 3+4 v4] 한글 키 계약 검증 — 환경변수로 제어
    # STRICT_RECOMMEND_CONTRACT=1: 위반 시 ValueError → 즉시 차단
    # STRICT_RECOMMEND_CONTRACT=0 (기본값): 위반 시 logger.warning만 → 운영 안전 우선
    # STRICT_RECOMMEND_CONTRACT=skip: 검증 자체 스킵 (긴급 상황만)
    #
    # 기본값을 0으로 둔 이유: 첫 배포는 무조건 관찰 모드부터.
    # 며칠 운영하며 [Recommend Contract] 위반 0건 확인 후
    # Railway에 STRICT_RECOMMEND_CONTRACT=1 설정해서 strict 활성화.
    import os as _os
    _strict = _os.environ.get("STRICT_RECOMMEND_CONTRACT", "0")
    if _strict != "skip":
        try:
            from trade_plan import validate_recommend_row as _validate
            _validate(row)
        except ValueError as _e:
            if _strict == "1":
                raise
            else:
                # 비-strict 모드: 경고만 찍고 진행 (운영 안정성 우선)
                import logging as _logging
                _logging.getLogger("ticker_analyzer").warning(
                    f"[Recommend Contract] {ctx.code6} 검증 실패 (STRICT={_strict}이라 진행): {_e}"
                )

    return row


# ═══════════════════════════════════════════════════
#  5. 오케스트레이터 (기존 analyze_ticker 대체)
# ═══════════════════════════════════════════════════

def analyze_ticker_v2(
    t: str, ohlcv_df: pd.DataFrame, top_df: pd.DataFrame, mcap_map: Dict[str, float],
    kospi_set: set, kosdaq_set: set, name_map: Dict[str, str], sector_map: Dict[str, str],
    bench_map: Dict[str, Dict[int, float]],
    inv_maps: Optional[Dict[str, Dict[str, int]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    [v3.2] analyze_ticker의 SRP 분리 버전.
    
    기존 analyze_ticker와 100% 동일한 딕셔너리를 반환.
    시그니처도 동일하여 drop-in replacement 가능.
    """
    # 필요한 함수 import (기존 collector.py의 전역 함수들)
    from collector import (
        get_mcap_eok_from_map, calc_rsi, calc_mfi, calc_atr,
        calc_vwap, calc_supertrend, calc_hma, check_candle_pattern,
        calc_volume_profile_v2, ema,
        LOOKBACK_DAYS, MIN_MCAP_EOK, MIN_TURNOVER_EOK,
        BB_PERIOD, BB_STD, BB_SQUEEZE_BW,
        KC_PERIOD, KC_ATR_PERIOD, KC_MULT,
    )

    # Step 1: 데이터 정제 + 필터링
    ctx = prepare_ohlcv(
        t, ohlcv_df, top_df, mcap_map,
        lookback_days=LOOKBACK_DAYS,
        min_mcap_eok=MIN_MCAP_EOK,
        min_turnover_eok=MIN_TURNOVER_EOK,
        get_mcap_fn=get_mcap_eok_from_map,
    )
    if ctx is None:
        return None

    # Step 2: 지표 계산
    ind = calculate_indicators(
        ctx,
        bb_period=BB_PERIOD, bb_std=BB_STD, bb_squeeze_bw=BB_SQUEEZE_BW,
        kc_period=KC_PERIOD, kc_atr_period=KC_ATR_PERIOD, kc_mult=KC_MULT,
        calc_rsi_fn=calc_rsi, calc_mfi_fn=calc_mfi, calc_atr_fn=calc_atr,
        calc_vwap_fn=calc_vwap, calc_supertrend_fn=calc_supertrend,
        calc_hma_fn=calc_hma, check_candle_fn=check_candle_pattern,
        calc_volume_profile_fn=calc_volume_profile_v2,
        ema_fn=ema,
    )

    # Step 3: 트레이딩 플랜
    plan = build_ticker_plan(ctx, ind, inv_maps)

    # Step 4: 결과 조립
    return assemble_result(
        ctx, ind, plan,
        name_map, sector_map, top_df,
        kospi_set, kosdaq_set, bench_map,
    )
