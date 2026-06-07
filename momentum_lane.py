# -*- coding: utf-8 -*-
"""
momentum_lane.py — v23.1 Momentum Lane (⚡ 모멘텀 후보 레인)

[배경] ROUTE별 forward 수익률 분석 결과, "과열(OVERHEAT)"로 분류돼 공식 매수에서
배제되던 종목군이 실제로는 가장 높은 전방수익률(T+3 +16.6%, 승률 84%)과 가장 통제된
하방(-6.5%)을 보였다. 그리고 그 OVERHEAT 중 v23.0 GUARD를 통과한 종목만 추리면
수익이 더 깨끗해진다(통과 +16.8% vs 탈락 +5.3%, 격차 +11.5%p).

[설계] 공식 매수 게이트(scoring_engine.TOP_PICK)는 한 글자도 건드리지 않고, 별도의
"⚡ 모멘텀 후보" 레인을 추가한다. 진입 조건:
    ROUTE == OVERHEAT  AND  GUARD_ALL_PASS  AND  (시장 위험회피 아님)
Tier A(RR>=1.2) = 실전 후보 / Tier B(RR<1.2) = 관찰(매수 아님). ⚡ RR 알파 레인의
Tier 개념과 일관.

[시장국면 게이트 — 비대칭 보험] 분석 기간(2026)에는 진짜 하락장이 없어 게이트의
보호효과를 데이터로 검증할 수 없었다. 따라서 게이트는 "알파 증대용 대칭 게이트"가
아니라 "꼬리위험 보험"으로 설계한다 — 평상시 항상 ON, *명백한 하락 전환*에만 OFF:
    close < MA20  AND  MA20 하락(5일전 대비)  AND  MA20 대비 -3% 이상 이탈
강세장에선 거의 항상 ON(무해), 미래 하락장에서만 과열주 레인을 자동 차단한다.

read-only: 입력 df의 기존 컬럼은 변경하지 않고 MOMENTUM_* 컬럼만 추가한다.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("momentum_lane")

# ── 기본 파라미터 (SSOT가 없을 때의 폴백; collector_config.MomentumLaneConfig가 우선) ──
# [중요] RR(손익비) 기준은 의도적으로 분류에서 제외한다. 백테스트 결과 OVERHEAT의
# 초과수익은 RR이 *낮은*(이미 강하게 오른) 종목에서 나왔다(모멘텀 역설): RR>=1.2로
# 거른 군 +2.4% vs RR<1.2 군 +18.2%. 따라서 Tier는 RR이 아니라 가드 반영 점수
# 랭크로 나눈다(상위 N = 실전, 그 외 = 관찰). 점수 상위5도 +16.4% 알파 유지 확인.
MOMENTUM_SOURCE_ROUTE = "OVERHEAT"
MOMENTUM_REQUIRE_GUARD = True    # GUARD_ALL_PASS 통과 의무
MOMENTUM_MAX_PICKS = 5           # 점수 랭크 상위 N개를 Tier A(실전 후보)로

# 시장국면(비대칭 보험) 임계
REGIME_MA_WINDOW = 20
REGIME_MA_SLOPE_LOOKBACK = 5
REGIME_DEVIATION_FLOOR = -0.03   # close가 MA20 대비 -3% 이상 이탈해야 risk_off

# 계약 컬럼 (check_contract_gate가 검증)
MOMENTUM_LANE_COLS: List[str] = [
    "MOMENTUM_LANE",          # int 0/1 : Tier A 실전 후보
    "MOMENTUM_WATCH",         # int 0/1 : Tier B 관찰 후보 (매수 아님)
    "MOMENTUM_LANE_TIER",     # str "A"/"B"/""
    "MOMENTUM_LANE_SCORE",    # float : 정렬용(가드 반영 점수)
    "MOMENTUM_LANE_RANK",     # int : 레인 내 순위(1=최상, 미진입=0)
    "MOMENTUM_LANE_REASON",   # str
]


# ───────────────────────── 안전 접근자 ─────────────────────────
def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _txt(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].astype("object").fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    """bool 플래그 안전 추출. 없으면 전부 False."""
    if col in df.columns:
        s = df[col]
        if s.dtype == bool:
            return s.fillna(False)
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(bool)
    return pd.Series(False, index=df.index)


# ───────────────────────── 시장국면 게이트 ─────────────────────────
def compute_market_risk_off(
    kospi_daily_path: Optional[str] = None,
    asof: Optional[str] = None,
    df_kospi: Optional[pd.DataFrame] = None,
) -> Tuple[bool, dict]:
    """KOSPI 일봉으로 '명백한 하락 전환' 여부 판정 (비대칭 보험).

    risk_off=True  → 모멘텀 레인 OFF
    risk_off=False → 레인 ON (평상시/데이터부족 디폴트)

    asof: 'YYYYMMDD'. 지정 시 그 날짜까지의 데이터로 판정(백테스트용). None이면 최신.
    """
    info = {"close": None, "ma20": None, "deviation": None,
            "ma20_slope_down": None, "reason": ""}
    try:
        if df_kospi is None:
            if not kospi_daily_path or not os.path.exists(kospi_daily_path):
                info["reason"] = "kospi_daily 없음 → 레인 ON(디폴트)"
                return False, info
            df_kospi = pd.read_csv(kospi_daily_path)
        k = df_kospi.copy()
        k["date"] = k["date"].astype(str)
        k = k.sort_values("date").reset_index(drop=True)
        if asof is not None:
            k = k[k["date"] <= str(asof)]
        if len(k) < (REGIME_MA_WINDOW // 2 + REGIME_MA_SLOPE_LOOKBACK):
            info["reason"] = "표본 부족 → 레인 ON(디폴트)"
            return False, info
        close = pd.to_numeric(k["close"], errors="coerce")
        ma = close.rolling(REGIME_MA_WINDOW, min_periods=REGIME_MA_WINDOW // 2).mean()
        c = float(close.iloc[-1])
        m = float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else None
        if m is None or m <= 0:
            info["reason"] = "MA20 미형성 → 레인 ON(디폴트)"
            return False, info
        m_prev = ma.iloc[-1 - REGIME_MA_SLOPE_LOOKBACK] if len(ma) > REGIME_MA_SLOPE_LOOKBACK else np.nan
        slope_down = bool(not pd.isna(m_prev) and m < float(m_prev))
        dev = c / m - 1.0
        info.update({"close": round(c, 2), "ma20": round(m, 2),
                     "deviation": round(dev, 4), "ma20_slope_down": slope_down})
        risk_off = bool((c < m) and slope_down and (dev <= REGIME_DEVIATION_FLOOR))
        if risk_off:
            info["reason"] = (f"하락전환 감지: close {c:.0f} < MA20 {m:.0f} "
                              f"({dev*100:.1f}%) & MA20 하락 → 모멘텀 레인 OFF")
        else:
            info["reason"] = (f"정상: close {c:.0f} vs MA20 {m:.0f} "
                              f"({dev*100:+.1f}%) → 모멘텀 레인 ON")
        return risk_off, info
    except Exception as e:  # 어떤 에러든 레인을 끄지 않는다(보수적이지 않게)
        info["reason"] = f"판정 실패({e}) → 레인 ON(디폴트)"
        return False, info


# ───────────────────────── 설정 해석 ─────────────────────────
class _FallbackLaneConfig:
    source_route = MOMENTUM_SOURCE_ROUTE
    require_guard = MOMENTUM_REQUIRE_GUARD
    max_picks = MOMENTUM_MAX_PICKS


def _resolve_config(config):
    if config is None:
        return _FallbackLaneConfig()
    # CollectorConfig facade(.momentum_lane) 또는 MomentumLaneConfig 직접 허용
    cand = getattr(config, "momentum_lane", config)
    for attr in ("source_route", "require_guard", "max_picks"):
        if not hasattr(cand, attr):
            return _FallbackLaneConfig()
    return cand


# ───────────────────────── 메인 엔진 ─────────────────────────
def apply_momentum_lane(
    df: pd.DataFrame,
    market_risk_off: bool = False,
    config=None,
) -> pd.DataFrame:
    """OVERHEAT × GUARD 통과 종목을 ⚡ 모멘텀 레인으로 선별. read-only(MOMENTUM_* 추가)."""
    cfg = _resolve_config(config)
    out = df.copy()
    n = len(out)

    # 컬럼 초기화
    out["MOMENTUM_LANE"] = 0
    out["MOMENTUM_WATCH"] = 0
    out["MOMENTUM_LANE_TIER"] = ""
    out["MOMENTUM_LANE_SCORE"] = 0.0
    out["MOMENTUM_LANE_RANK"] = 0
    out["MOMENTUM_LANE_REASON"] = ""

    if n == 0:
        return out

    # 시장 위험회피 시 레인 전체 OFF
    if market_risk_off:
        out["MOMENTUM_LANE_REASON"] = "시장 위험회피(하락 전환) — 모멘텀 레인 비활성"
        return out

    route = _txt(out, "ROUTE")
    is_source = route.str.upper() == str(cfg.source_route).upper()

    if cfg.require_guard:
        # GUARD_ALL_PASS 컬럼 자체가 없으면(가드 미적용 파이프라인) 레인 비활성
        if "GUARD_ALL_PASS" not in out.columns:
            out["MOMENTUM_LANE_REASON"] = np.where(
                is_source, "GUARD 미적용 — 모멘텀 레인 보류", "")
            return out
        guard_ok = _flag(out, "GUARD_ALL_PASS")
    else:
        guard_ok = pd.Series(True, index=out.index)

    rr = _num(out, "RR_NOW_TP1", np.nan)
    # 정렬 점수: 가드 반영 점수 우선, 없으면 ELITE_SCORE
    score = _num(out, "GUARDED_ELITE_SCORE", np.nan)
    score = score.where(~score.isna(), _num(out, "ELITE_SCORE", 0.0)).fillna(0.0)
    out["MOMENTUM_LANE_SCORE"] = score.round(1)

    # 레인 자격: 과열 + 가드통과 (RR 무관 — 모멘텀 역설 때문에 RR로 거르지 않음)
    in_lane = is_source & guard_ok
    lane_idx = out.index[in_lane]

    if len(lane_idx) > 0:
        # 가드 반영 점수 내림차순 랭크 → 상위 max_picks = Tier A(실전), 그 외 = Tier B(관찰)
        ranked = score.loc[lane_idx].rank(ascending=False, method="first").astype(int)
        topn = int(cfg.max_picks)
        a_idx = ranked[ranked <= topn].index
        b_idx = ranked[ranked > topn].index

        out.loc[a_idx, "MOMENTUM_LANE"] = 1
        out.loc[a_idx, "MOMENTUM_LANE_TIER"] = "A"
        out.loc[a_idx, "MOMENTUM_LANE_RANK"] = ranked.loc[a_idx].astype(int)

        out.loc[b_idx, "MOMENTUM_WATCH"] = 1
        out.loc[b_idx, "MOMENTUM_LANE_TIER"] = "B"

    # 사유 문자열
    is_a = out["MOMENTUM_LANE"] == 1
    is_b = out["MOMENTUM_WATCH"] == 1

    def _reason_row(i):
        rr_txt = "" if pd.isna(rr.iloc[i]) else f", RR {rr.iloc[i]:.2f}"
        if is_a.iloc[i]:
            return (f"⚡ 모멘텀 후보 (과열·가드통과, 강도 {int(out['MOMENTUM_LANE_RANK'].iloc[i])}위"
                    f"/점수 {score.iloc[i]:.0f}{rr_txt})")
        if is_b.iloc[i]:
            return f"모멘텀 관찰 (과열·가드통과, 상위{int(cfg.max_picks)} 밖 — 매수 아님)"
        if is_source.iloc[i] and not guard_ok.iloc[i]:
            return "과열이나 GUARD 미통과 — 레인 제외"
        return ""

    out["MOMENTUM_LANE_REASON"] = [_reason_row(i) for i in range(n)]
    return out


# ───────────────────────── 요약 ─────────────────────────
def momentum_summary(df: pd.DataFrame) -> dict:
    """로그/대시보드용 한 줄 요약."""
    if df is None or "MOMENTUM_LANE" not in df.columns:
        return {"tier_a": 0, "tier_b": 0, "top": None}
    a = int(pd.to_numeric(df["MOMENTUM_LANE"], errors="coerce").fillna(0).sum())
    b = int(pd.to_numeric(df.get("MOMENTUM_WATCH", 0), errors="coerce").fillna(0).sum())
    top = None
    la = df[df["MOMENTUM_LANE"] == 1]
    if len(la):
        name_col = "종목명" if "종목명" in la.columns else None
        idx = pd.to_numeric(la["MOMENTUM_LANE_SCORE"], errors="coerce").idxmax()
        top = str(la.loc[idx, name_col]) if name_col else str(idx)
    return {"tier_a": a, "tier_b": b, "top": top}

