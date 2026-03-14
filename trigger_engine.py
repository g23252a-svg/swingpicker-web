# -*- coding: utf-8 -*-
"""
trigger_engine.py — 트리거 점수 + 매물대 분석 엔진 (v20.6.5)
═══════════════════════════════════════════════════════════════
[v20.6.5] collector.py에서 분리 — pipeline_score.py 순환 의존 해소

분리 대상:
  - calculate_trigger_score(): 트리거 점수 산출 (144줄)
  - calc_volume_profile_v2(): 매물대 분석 (54줄)

하위 호환:
  collector.py에서 `from trigger_engine import *`로 재수출
  → 기존 `from collector import calculate_trigger_score` 동작 유지
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def calculate_trigger_score(df: pd.DataFrame) -> float:
    """
    [v3.1 #3] 트리거 점수 — 사전 계산 최적화.

    이전 버전: _calc_raw_trigger(idx)마다 df.iloc[:idx+1]을 잘라 rolling/ewm을
    밑바닥부터 재계산 → O(N²) 낭비.

    개선: 전체 df에 대해 rolling/ewm을 1회만 계산하고,
    _calc_raw_trigger(idx)는 값만 인덱싱 → O(1).
    """
    if df is None or df.empty or len(df) < 30:
        return 0.0

    # ═══ 1. 전체 데이터에 대해 미리 계산 (1회만) ═══
    close = pd.to_numeric(df['종가'], errors='coerce').fillna(0)
    high = pd.to_numeric(df['고가'], errors='coerce').fillna(0)
    low = pd.to_numeric(df['저가'], errors='coerce').fillna(0)
    open_p = pd.to_numeric(df['시가'], errors='coerce').fillna(0)
    vol = pd.to_numeric(df['거래량'], errors='coerce').fillna(0)

    vol_ma20 = vol.rolling(20).mean().shift(1)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig

    # 외인 데이터 존재 여부
    has_foreign = '외인순매수' in df.columns
    if has_foreign:
        foreign_net = pd.to_numeric(df['외인순매수'], errors='coerce').fillna(0)

    def _calc_raw_trigger(idx):
        if idx < 25:
            return 0.0
        try:
            # ═══ 2. 값만 추출 — O(1) ═══
            c_curr = float(close.iloc[idx])
            h_curr = float(high.iloc[idx])
            l_curr = float(low.iloc[idx])
            o_curr = float(open_p.iloc[idx])
            v_curr = float(vol.iloc[idx])
            c_prev = float(close.iloc[idx - 1])

            if c_prev == 0:
                return 0.0

            # 기초 지표
            ret_pct = (c_curr / c_prev - 1) * 100
            candle_len = h_curr - l_curr

            upper_wick = h_curr - max(o_curr, c_curr)
            wick_ratio = upper_wick / candle_len if candle_len > 0 else 0.0
            range_pos = (c_curr - l_curr) / candle_len if candle_len > 0 else 0.5

            # 거래량 비율 (사전 계산된 MA20 사용)
            vm20 = float(vol_ma20.iloc[idx])
            if pd.isna(vm20) or vm20 == 0:
                vm20 = v_curr
            vol_ratio = v_curr / vm20

            # ---------------------------------------------------------
            # [Base Score] 기본 점수 (Max 90)
            # ---------------------------------------------------------

            # (1) 거래량 점수
            if vol_ratio < 0.5:
                score_vol = 5.0
            elif 0.5 <= vol_ratio < 1.2:
                score_vol = 5.0 + (vol_ratio - 0.5) * 50.0
            elif 1.2 <= vol_ratio <= 3.0:
                score_vol = 40.0
            elif 3.0 < vol_ratio <= 4.0:
                score_vol = 40.0 - (vol_ratio - 3.0) * 20.0
            else:
                score_vol = 20.0

            score_vol = max(0.0, min(40.0, score_vol))

            # (2) 돌파/추세 점수 (사전 계산된 sma20, std20 사용)
            s20 = float(sma20.iloc[idx])
            sd20 = float(std20.iloc[idx])

            score_breakout = 0.0
            if not (pd.isna(s20) or pd.isna(sd20) or sd20 == 0):
                bb_upper = s20 + (2 * sd20)
                if c_curr >= bb_upper:
                    score_breakout = 40.0
                elif c_curr >= s20:
                    score_breakout = 20.0

            # (3) 모멘텀 가속 (사전 계산된 MACD 히스토그램 사용)
            score_mom = 0.0
            if idx >= 1 and hist.iloc[idx] > hist.iloc[idx - 1]:
                score_mom = 10.0

            base = score_vol + score_breakout + score_mom

            # ---------------------------------------------------------
            # [Penalty] 감점 로직 (Cap: 60)
            # ---------------------------------------------------------
            penalty = 0.0

            if ret_pct >= 5.0:
                if wick_ratio >= 0.35:
                    penalty += 25.0
                elif wick_ratio >= 0.25:
                    penalty += 15.0

            if ret_pct >= 3.0 and range_pos < 0.6:
                penalty += 15.0

            if vol_ratio >= 3.0:
                if c_curr < o_curr:
                    penalty += 25.0
                elif wick_ratio > 0.3 or range_pos < 0.5:
                    penalty += 20.0

            if has_foreign:
                try:
                    frg_net = float(foreign_net.iloc[idx])
                    if ret_pct >= 5.0 and frg_net < 0 and vol_ratio > 1.5:
                        if range_pos < 0.6 or wick_ratio > 0.25:
                            penalty += 20.0
                except (IndexError, ValueError, TypeError):
                    pass  # 외국인 데이터 없는 종목은 정상

            penalty = min(penalty, 60.0)

            return max(0.0, base - penalty)

        except Exception:
            return 0.0

    last_idx = len(df) - 1
    score_today = _calc_raw_trigger(last_idx)
    score_yesterday = _calc_raw_trigger(last_idx - 1)

    final_score = (score_today * 0.7) + (score_yesterday * 0.3)
    return float(final_score)


def calc_volume_profile_v2(df_120: pd.DataFrame) -> Tuple[Optional[float], float, float, float]:
    """
    [v8.7 최종 정밀 버전] 매물대 분석 로직
    - Typical Price 사용 + 2~98% Percentile 컷
    - 데이터 길이에 따른 가변 Bins (15~20) 반영
    - 변동성(ATR%) 연동형 Near Resistance 범위 설정
    """
    if df_120.empty or len(df_120) < 15:
        return None, 0.0, 0.0, 0.0

    # 1. 기초 데이터 확보 (Typical Price)
    typ_price = (df_120['고가'] + df_120['저가'] + df_120['종가']) / 3
    curr_c = float(df_120['종가'].iloc[-1])

    # 2. 변동성(ATR%) 기반 근접 저항 범위 결정 (6% ~ 12% 가변)
    tr = pd.concat([(df_120['고가'] - df_120['저가']),
                    (df_120['고가'] - df_120['종가'].shift(1)).abs(),
                    (df_120['저가'] - df_120['종가'].shift(1)).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14).mean().iloc[-1] / curr_c) if curr_c > 0 else 0.03
    near_threshold = max(1.06, min(1.12, 1.0 + atr_pct * 2.0))

    # 3. 범위 설정 및 안전장치 (Percentile + Safety Clamp)
    l_min = typ_price.quantile(0.02)
    h_max = typ_price.quantile(0.98)
    l_min = min(l_min, curr_c * 0.97)
    h_max = max(h_max, curr_c * 1.03)

    if h_max <= l_min:
        return None, 0.0, 0.0, 0.0

    # 4. 가변 Bins 설정 (해상도 조절)
    bins = 20 if len(df_120) >= 100 else 15
    bin_size = (h_max - l_min) / bins
    bins_edges = [l_min + i * bin_size for i in range(bins + 1)]

    # 5. Histogram 생성
    volume_hist, _ = np.histogram(typ_price, bins=bins_edges, weights=df_120['거래량'])

    # 6. POC 및 매물 비중 산출
    max_bin_idx = np.argmax(volume_hist)
    poc_p = (bins_edges[max_bin_idx] + bins_edges[max_bin_idx + 1]) / 2

    total_vol = volume_hist.sum() if volume_hist.sum() > 0 else 1
    res_all_vol = 0.0
    res_near_vol = 0.0

    for i in range(bins):
        bin_center = (bins_edges[i] + bins_edges[i + 1]) / 2
        if bin_center > curr_c:
            res_all_vol += volume_hist[i]
            if bin_center <= curr_c * near_threshold:
                res_near_vol += volume_hist[i]

    return poc_p, (res_all_vol / total_vol), (res_near_vol / total_vol), (near_threshold - 1.0) * 100
