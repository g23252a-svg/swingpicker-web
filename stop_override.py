# -*- coding: utf-8 -*-
"""
stop_override.py — [v23.2] 손절 -10% override (공식 신호 과타이트 손절 교정).

[근거] backtest_validation 정확 재현(-14.61%) 위에서 손절폭 ablation 수행:
추천손절가(-5~7%, 과타이트)가 현재 -14.6%의 주범. 손절을 진입가 -10%로 넓히면
같은 종목·같은 진입으로 +13.40% / MDD 40.5%→23.7% (단조 패턴: 넓힐수록↑,
-7%로 조이면 -42%). 두영님 실제 진입(추천가 -3~5% 눌림 지정가)에서도 +8.65~11.5%
로 -14.6%를 뒤집음을 별도 검증함.

[안전장치 — 2단 구조]
강세장 단일 구간 검증이라 "손절을 넓힌다 = 손실을 늦게 끊는다"가 진짜 하락장에선
+수익을 -로 뒤집는다(베어마켓 미검증, 데이터에 없음). 따라서:
  · compute_market_risk_off 가 베어 감지 → override OFF(추천손절 복귀)
                                        + 신규진입 차단(NEW_ENTRY_BLOCKED)
검증된 강세장만 켜고, 미검증 베어는 룰로 막는다.

[적용 범위]
공식 신호(TOP_PICK / BUY_NOW_ELIGIBLE)에만 적용. ⚡ 모멘텀 레인은 익일 진입 자체가
손실(-24.6%)임이 검증돼 정보 카드로만 두며 손절 override를 적용하지 않는다.

read-only: STOP_OVERRIDE_* / NEW_ENTRY_BLOCKED 컬럼만 추가. 원본 '손절가'는 보존.
SSOT: collector_config.StopOverrideConfig (없으면 아래 폴백 상수).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── 기본 파라미터 (SSOT가 없을 때의 폴백; collector_config.StopOverrideConfig가 우선) ──
STOP_OVERRIDE_PCT = 0.10                  # 진입가 대비 손절 폭 (검증된 값)
STOP_APPLY_OFFICIAL_ONLY = True           # 공식 신호에만
STOP_DISABLE_ON_RISK_OFF = True           # 베어 시 override OFF
STOP_BLOCK_NEW_ENTRY_ON_RISK_OFF = True   # 베어 시 신규진입 차단

STOP_OVERRIDE_COLS = [
    "STOP_OVERRIDE_ACTIVE",   # 이 행에 override 손절가가 적용됐는지
    "STOP_OVERRIDE_PRICE",    # 진입가(추천매수가) × (1 - stop_pct), 0이면 미적용
    "STOP_OVERRIDE_PCT",      # 적용된 손절 폭 (0.10)
    "NEW_ENTRY_BLOCKED",      # 베어 → True (신규 매수 금지)
    "STOP_OVERRIDE_REASON",   # 사람이 읽는 사유
]


# ───────────────────────── 안전한 컬럼 접근 ─────────────────────────
def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        s = df[col]
        if s.dtype == bool:
            return s.fillna(False)
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(bool)
    return pd.Series(False, index=df.index)


# ───────────────────────── 설정 해석 ─────────────────────────
class _FallbackStopConfig:
    enabled = True
    stop_pct = STOP_OVERRIDE_PCT
    apply_to_official_only = STOP_APPLY_OFFICIAL_ONLY
    disable_on_risk_off = STOP_DISABLE_ON_RISK_OFF
    block_new_entry_on_risk_off = STOP_BLOCK_NEW_ENTRY_ON_RISK_OFF


def _resolve_config(config):
    if config is None:
        return _FallbackStopConfig()
    # CollectorConfig facade(.stop_override) 또는 StopOverrideConfig 직접 허용
    cand = getattr(config, "stop_override", config)
    for attr in ("enabled", "stop_pct", "apply_to_official_only",
                 "disable_on_risk_off", "block_new_entry_on_risk_off"):
        if not hasattr(cand, attr):
            return _FallbackStopConfig()
    return cand


# ───────────────────────── 메인 엔진 ─────────────────────────
def apply_stop_override(
    df: pd.DataFrame,
    market_risk_off: bool = False,
    config=None,
) -> pd.DataFrame:
    """공식 신호의 손절가를 진입가 -stop_pct로 교정. read-only(STOP_OVERRIDE_* 추가).

    · 강세/중립: TOP_PICK/BUY_NOW_ELIGIBLE 행에 STOP_OVERRIDE_PRICE = 추천매수가×(1-stop_pct)
    · 베어(market_risk_off): override 미적용(추천손절 복귀) + NEW_ENTRY_BLOCKED=True
    원본 '손절가' 컬럼은 절대 건드리지 않는다(추천손절 보존).
    """
    cfg = _resolve_config(config)
    out = df.copy()
    n = len(out)

    # 컬럼 초기화 (항상 부여)
    out["STOP_OVERRIDE_ACTIVE"] = False
    out["STOP_OVERRIDE_PRICE"] = 0.0
    out["STOP_OVERRIDE_PCT"] = 0.0
    out["NEW_ENTRY_BLOCKED"] = False
    out["STOP_OVERRIDE_REASON"] = ""

    if n == 0 or not bool(getattr(cfg, "enabled", True)):
        return out

    stop_pct = float(getattr(cfg, "stop_pct", STOP_OVERRIDE_PCT))
    official_only = bool(getattr(cfg, "apply_to_official_only", True))
    disable_ro = bool(getattr(cfg, "disable_on_risk_off", True))
    block_ro = bool(getattr(cfg, "block_new_entry_on_risk_off", True))

    # ── 베어: override OFF + (옵션) 신규진입 차단 ──
    if market_risk_off and disable_ro:
        if block_ro:
            out["NEW_ENTRY_BLOCKED"] = True
            out["STOP_OVERRIDE_REASON"] = "risk_off: 추천손절 유지 + 신규진입 차단"
        else:
            out["STOP_OVERRIDE_REASON"] = "risk_off: 추천손절 유지"
        return out

    # ── 강세/중립: 공식 신호에 손절 override ──
    buy = _num(out, "추천매수가")
    if official_only:
        elig = _bool(out, "TOP_PICK") | _bool(out, "BUY_NOW_ELIGIBLE")
    else:
        elig = pd.Series(True, index=out.index)
    valid = elig & (buy > 0)

    if valid.any():
        override_price = (buy * (1.0 - stop_pct)).round(0)
        out.loc[valid, "STOP_OVERRIDE_ACTIVE"] = True
        out.loc[valid, "STOP_OVERRIDE_PRICE"] = override_price[valid]
        out.loc[valid, "STOP_OVERRIDE_PCT"] = stop_pct
        out.loc[valid, "STOP_OVERRIDE_REASON"] = (
            f"공식신호 손절 진입가-{stop_pct * 100:.0f}% (추천손절 과타이트 교정)"
        )
    return out


def stop_override_summary(df: pd.DataFrame) -> dict:
    """배지/로그용 요약."""
    if df is None or len(df) == 0:
        return {"active": 0, "blocked": 0, "stop_pct": STOP_OVERRIDE_PCT}
    act = int(_bool(df, "STOP_OVERRIDE_ACTIVE").sum())
    blk = int(_bool(df, "NEW_ENTRY_BLOCKED").sum())
    pct = 0.0
    if "STOP_OVERRIDE_PCT" in df.columns:
        s = pd.to_numeric(df["STOP_OVERRIDE_PCT"], errors="coerce").fillna(0.0)
        pos = s[s > 0]
        pct = float(pos.iloc[0]) if len(pos) else STOP_OVERRIDE_PCT
    return {"active": act, "blocked": blk, "stop_pct": pct}
