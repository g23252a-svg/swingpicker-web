# -*- coding: utf-8 -*-
"""
market_regime.py — [v28] 시장 레짐 판정 (상승/중립/하락)
═══════════════════════════════════════════════════
목적:
  지수 방향과 시장 내부(시장폭)를 결합해 "지금이 어떤 장인가"를
  하나의 명시적 상태로 판정하고, 그 상태가 신규 진입 허용/차단과
  포지션 사이즈를 결정하게 한다.

설계 근거 (2026-02-26 ~ 2026-07-13, 98거래일 실측):
  - UP 레짐   : 게이트 통과 픽 평균 +5.1%/건 (n=22) — 알파의 대부분
  - NEUTRAL   : 평균 +0.1%/건 (n=14) — 본전 수준 → 사이즈 절반
  - DOWN      : 진입 자체가 손실 구간 → 차단 (v27 시장폭 게이트 일반화)
  - 랭킹은 레짐과 무관하게 POC 근접 순이 우월했다
    (UP 레짐에서도 점수순 +3.0% vs POC순 +5.1%) → 레짐은 GO/NO-GO와
    사이즈만 결정하고 랭킹에는 개입하지 않는다.

판정 규칙 (lookahead 없음 — t일 종가까지만 사용):
  UP      : KOSPI 종가 > MA20 AND 종가 > MA5 AND 시장폭 ≥ 50
  DOWN    : 시장폭 < 35 (지수 위치 무관 — 내부 붕괴 우선)
  NEUTRAL : 그 외
  UNKNOWN : 지수/시장폭 데이터 부족 → 보수적으로 NEUTRAL 취급
"""
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REGIME_UP = "UP"
REGIME_NEUTRAL = "NEUTRAL"
REGIME_DOWN = "DOWN"
REGIME_UNKNOWN = "UNKNOWN"

# 레짐별 신규 진입 사이즈 배수 (Kelly/추천 비중에 곱)
# [v32] DOWN 0.0 → 0.3: 알파 전면 게이트 도입으로 하락 레짐에서도
# 알파 최상위 10%만 축소 사이즈로 진입 허용(사용자 선택: 적응형·반감 사이즈).
# 실측 근거: risk_off=False & 내부약세(breadth<35) 알파 픽 +0.84%/승률 51%.
# 진짜 위험한 하락(risk_off=KOSPI<하락MA20)은 NEW_ENTRY_BLOCKED가 별도 하드블록.
REGIME_SIZE_MULT = {
    REGIME_UP: 1.0,
    REGIME_NEUTRAL: 0.5,
    REGIME_DOWN: 0.3,
    REGIME_UNKNOWN: 0.5,
}

# 판정 임계 (v28 검증값)
BREADTH_UP_MIN = 50.0
BREADTH_DOWN_MAX = 35.0

_KOSPI_CSV = "kospi_daily.csv"


def load_kospi_daily(data_dir: str = "data") -> Optional[pd.DataFrame]:
    """kospi_daily.csv 로드 → date 오름차순 DataFrame (close, ma5, ma20)."""
    path = os.path.join(data_dir, _KOSPI_CSV)
    if not os.path.exists(path):
        logger.warning(f"KOSPI 일봉 파일 없음: {path}")
        return None
    try:
        k = pd.read_csv(path)
        if "date" not in k.columns or "close" not in k.columns:
            logger.warning("kospi_daily.csv 스키마 불일치 (date/close 필요)")
            return None
        k = k[["date", "close"]].copy()
        k["date"] = k["date"].astype(str).str.replace("-", "").str.slice(0, 8)
        k["close"] = pd.to_numeric(k["close"], errors="coerce")
        k = k.dropna().sort_values("date").reset_index(drop=True)
        k["ma5"] = k["close"].rolling(5).mean()
        k["ma20"] = k["close"].rolling(20).mean()
        return k
    except Exception as e:
        logger.warning(f"KOSPI 일봉 로드 실패: {e}")
        return None


def compute_market_regime(
    trade_ymd: str,
    market_breadth: Optional[float],
    data_dir: str = "data",
    kospi_df: Optional[pd.DataFrame] = None,
) -> dict:
    """레짐 판정.

    Returns dict:
      regime          : UP / NEUTRAL / DOWN / UNKNOWN
      size_mult       : 신규 진입 사이즈 배수 (0.0 / 0.5 / 1.0)
      allow_new_entry : bool — DOWN이면 False
      reason          : 사람이 읽는 판정 근거 (프론트 표시용)
      kospi_close / kospi_ma20 / kospi_ma5 / breadth : 원자료
    """
    ymd = str(trade_ymd).replace("-", "")[:8]
    breadth = None
    try:
        if market_breadth is not None:
            breadth = float(market_breadth)
    except (TypeError, ValueError):
        breadth = None

    k = kospi_df if kospi_df is not None else load_kospi_daily(data_dir)
    close = ma5 = ma20 = None
    if k is not None and not k.empty:
        sub = k[k["date"] <= ymd]
        if not sub.empty:
            last = sub.iloc[-1]
            close = float(last["close"])
            ma5 = float(last["ma5"]) if pd.notna(last["ma5"]) else None
            ma20 = float(last["ma20"]) if pd.notna(last["ma20"]) else None

    def _pack(regime: str, reason: str) -> dict:
        return {
            "regime": regime,
            "size_mult": REGIME_SIZE_MULT[regime],
            "allow_new_entry": regime != REGIME_DOWN,
            "reason": reason,
            "kospi_close": close,
            "kospi_ma20": ma20,
            "kospi_ma5": ma5,
            "breadth": breadth,
        }

    # 데이터 부족 → UNKNOWN (보수적: 사이즈 절반)
    if breadth is None and (close is None or ma20 is None):
        return _pack(REGIME_UNKNOWN, "지수/시장폭 데이터 부족 — 보수 운용 (사이즈 50%)")

    # 내부 붕괴 우선 — 지수가 어디 있든 시장폭 붕괴면 DOWN
    if breadth is not None and breadth < BREADTH_DOWN_MAX:
        return _pack(
            REGIME_DOWN,
            f"하락 레짐 — 시장폭 {breadth:.0f}% (<{BREADTH_DOWN_MAX:.0f}%), "
            "상승 종목이 소수인 내부 약세장. 신규 진입 차단 (실측: 이 구간 진입은 손실 우세)",
        )

    up_trend = (
        close is not None and ma20 is not None and ma5 is not None
        and close > ma20 and close > ma5
    )
    if up_trend and breadth is not None and breadth >= BREADTH_UP_MIN:
        return _pack(
            REGIME_UP,
            f"상승 레짐 — KOSPI 20일선·5일선 위 + 시장폭 {breadth:.0f}% "
            f"(실측: 이 레짐 게이트 통과 픽 평균 +5.1%/건)",
        )

    parts = []
    if close is not None and ma20 is not None:
        parts.append("KOSPI 20일선 " + ("위" if close > ma20 else "아래"))
    if breadth is not None:
        parts.append(f"시장폭 {breadth:.0f}%")
    return _pack(
        REGIME_NEUTRAL,
        "중립 레짐 — " + ", ".join(parts) +
        " (실측: 본전 구간 → 진입 허용하되 사이즈 50%)",
    )


def inject_regime_columns(df: pd.DataFrame, regime_info: dict) -> pd.DataFrame:
    """recommend DataFrame에 레짐 컬럼 주입."""
    df["MARKET_REGIME"] = regime_info.get("regime", REGIME_UNKNOWN)
    df["REGIME_REASON"] = regime_info.get("reason", "")
    df["REGIME_SIZE_MULT"] = regime_info.get("size_mult", 0.5)
    df["REGIME_ALLOW_ENTRY"] = int(bool(regime_info.get("allow_new_entry", True)))
    return df
