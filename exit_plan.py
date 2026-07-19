# -*- coding: utf-8 -*-
"""
exit_plan.py — v25.1 Exit Discipline Layer (청산 규율 · 표시 전용 SHADOW)
═══════════════════════════════════════════════════════════════════════════
유령체결을 걷어낸 '정직 로그'로 다시 본 시스템의 맨얼굴 = h5 평균 -4.27%/승률 33%.
문제는 종목 선택이 아니라 **청산 규율**이었다. 세 레버를 데이터로 확정해 카드에 표시한다.
공식 산식(TOP_PICK/BUY_NOW_ELIGIBLE)·기존 손절가 컬럼은 불변 — 규율 '권고선'만 덧붙인다.

■ 근거 (정직 h5 1,275건 · 장중 저가 경로 시뮬 · OHLCV · 2026-02-26~06-30)

  ① ROUTE 위험 등급 — 청산 이전에 '진입 자체'가 문제인 구간이 있다
     CARRY  n=193  평균 -17.6%  SL률 74%   ← 물타기/이월. 진입 재앙
     NEUTRAL n=187 평균 -4.9%   SL률 68%   ← 방향성 없음
     이 둘 제외 시 전체 -4.27% → -1.86% (승률 33→35%). '들어가지 말라'가 최대 알파.

  ② 손절폭 조임 — 현행 손절이 평균 -14.5%(거의 고정 -15%), stop_hit의 60%가 -10% 이하
     장중 저가 시뮬: 손절선 -15%→-7%로 좁히면 평균 -3.17%→-2.04%, -10%이하 꼬리 42%→0%.
     한 방 손실을 반으로. (좁혀서 승률은 소폭↓지만 기대값·꼬리 대폭 개선)

  ③ TP 앞당김 — 익절을 +10%에 두면(현행 목표는 더 멀다) 승률·기대값 동반 상승
     ①+② 위에 TP+10% 결합 시 평균 -0.05%/승률 41% (현행 -3.17%/32%). 합계손실 -3,161→-37%p.

  결합(정직 h5 전체): 현행 -3.17% → 규율적용 -1.08%/승률 35%. CARRY/NEUTRAL 제외군만 보면
  평균 -0.05%로 사실상 손익분기 — '검증된 자리에서 규율대로 나오면 안 깨진다'.

■ 이 모듈이 부여하는 표시 컬럼 (공식 컬럼 불변)
  EXIT_ROUTE_RISK   : 'HIGH_AVOID'(CARRY/NEUTRAL) / 'CAUTION'(ATTACK·OVERHEAT) / 'OK'
  EXIT_STOP_TIGHT   : 진입가 × (1 + stop_tight_pct/100)  — 권고 손절선(기본 -7%)
  EXIT_TP_QUICK     : 진입가 × (1 + tp_quick_pct/100)    — 권고 1차 익절선(기본 +10%)
  EXIT_PLAN_NOTE    : 사람이 읽는 한 줄 요약
  ※ D1 컷(D1_CHECKPOINT_PRICE)은 v24.7 모듈이 계속 담당. 여기서는 손절폭·TP·ROUTE만.

■ 실전 규율 (사장님용, 카드가 매일 계산해 표시)
  1) EXIT_ROUTE_RISK=HIGH_AVOID면 신규진입 보류 (검증된 최악 구간)
  2) 손절은 EXIT_STOP_TIGHT(-7%)에서 — -15%까지 절대 방치 금지
  3) EXIT_TP_QUICK(+10%) 도달 시 1차 익절
  4) D+1 종가 -4% 이탈 시 조기청산(v24.7)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("exit_plan")

EXIT_OUTPUT_COLS = ["EXIT_ROUTE_RISK", "EXIT_STOP_TIGHT", "EXIT_TP_QUICK",
                    "BE_TRIGGER_PRICE", "FAST_TP_PRICE", "EXIT_PLAN_NOTE"]

# [v30] 본전스탑 전환 트리거 (+5%) — 청산 그리드 검증 상위 8개 조합 전부에
# BE+5%가 포함됨 (표본 851건). +5% 도달 시 손절선을 진입가로 올려
# '이겼다가 -12%로 마감'(손절 트레이드의 51%가 +5% 선터치)을 차단한다.
BE_TRIGGER_PCT = 5.0

# [v33] 빠른 부분익절 — 알파 엣지는 첫 2일에 집중(일당 +0.97%p → 10일 +0.49%p 반감).
# OOS 실측(n=526): 2일차 +2% 이상 달린 픽은 이후 3일 평균 -0.90%(승률 40%) 되돌림.
# → 2일 내 +2% 도달 시 절반 익절 + 잔여분 손절선 본전(BE) 상향.
FAST_TP_PCT = 2.0
FAST_TP_DAY = 2

# [v32.1] 치료된 ROUTE 기준 위험 등급 — ATTACK은 이제 최고 신호(+3.70%p)라
# '과열/추격 주의' 대상이 아니다. OVERHEAT(-2.16%p 실측)만 CAUTION.
_HIGH_AVOID_ROUTES = {"CARRY", "NEUTRAL"}
_CAUTION_ROUTES_LEGACY = {"ATTACK", "OVERHEAT"}   # 치료 전(미검증일) 폴백
_CAUTION_ROUTES_HEALED = {"OVERHEAT"}


class _FallbackExitConfig:
    stop_tight_pct = -7.0      # 권고 손절폭 (정직 시뮬 최적 균형점)
    tp_quick_pct = 10.0        # 권고 1차 익절폭
    min_entry = 100.0
    exit_enabled = True

    def __post_init__(self):  # 폴백은 검증 생략
        pass


def _resolve(config):
    if config is None:
        try:
            from collector_config import DEFAULT_CONFIG
            if hasattr(DEFAULT_CONFIG, "exit"):
                return DEFAULT_CONFIG.exit
        except Exception:
            logger.warning("청산 규율 설정 로드 실패 — 폴백 설정 사용", exc_info=True)
        return _FallbackExitConfig()
    if hasattr(config, "stop_tight_pct"):
        return config
    if hasattr(config, "exit"):
        return config.exit
    return _FallbackExitConfig()


def add_exit_plan_columns(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """추천 df에 청산 규율 표시 컬럼 부여 (공식 컬럼/기존 손절가 불변)."""
    if df is None or len(df) == 0:
        return df
    cfg = _resolve(config)
    out = df.copy()
    if not getattr(cfg, "exit_enabled", True):
        return out

    stop_pct = float(getattr(cfg, "stop_tight_pct", -7.0))
    tp_pct = float(getattr(cfg, "tp_quick_pct", 10.0))
    min_entry = float(getattr(cfg, "min_entry", 100.0))

    # ROUTE 위험 등급 — [v32.1] 치료된 ROUTE(ROUTE_ALPHA_HEALED)에선 ATTACK이
    # 최고 신호이므로 CAUTION에서 제외. OVERHEAT(실측 -2.16%p)만 주의.
    route = out.get("ROUTE", pd.Series("", index=out.index)).astype(str).str.upper().str.strip()
    _healed = False
    if "ROUTE_ALPHA_HEALED" in out.columns:
        _healed = bool(pd.to_numeric(out["ROUTE_ALPHA_HEALED"], errors="coerce")
                       .fillna(0).astype(int).max() == 1)
    caution_routes = _CAUTION_ROUTES_HEALED if _healed else _CAUTION_ROUTES_LEGACY
    risk = pd.Series("OK", index=out.index, dtype="object")
    risk = risk.mask(route.isin(caution_routes), "CAUTION")
    risk = risk.mask(route.isin(_HIGH_AVOID_ROUTES), "HIGH_AVOID")
    out["EXIT_ROUTE_RISK"] = risk

    # 권고 손절/익절선 (진입가 기준)
    entry = pd.to_numeric(out.get("추천매수가", pd.Series(np.nan, index=out.index)), errors="coerce")
    valid = entry.notna() & (entry >= min_entry)
    out["EXIT_STOP_TIGHT"] = (entry * (1.0 + stop_pct / 100.0)).round(0).where(valid)
    out["EXIT_TP_QUICK"] = (entry * (1.0 + tp_pct / 100.0)).round(0).where(valid)
    # [v30] 본전스탑 전환가 — 이 가격 도달 시 손절선을 진입가(+수수료)로 상향
    out["BE_TRIGGER_PRICE"] = (entry * (1.0 + BE_TRIGGER_PCT / 100.0)).round(0).where(valid)
    # [v33] 빠른 부분익절선 — D+2 내 이 가격 도달 시 절반 익절 + 잔여 본전 상향.
    # 실측: 2일차 +2%↑ 픽의 이후 3일 -0.90%(되돌림) → 되돌림을 실현이익으로 전환.
    out["FAST_TP_PRICE"] = (entry * (1.0 + FAST_TP_PCT / 100.0)).round(0).where(valid)

    # 사람이 읽는 요약
    def _note(i):
        parts = []
        if risk.iat[i] == "HIGH_AVOID":
            parts.append("⛔ 고위험 루트(진입보류 권고)")
        elif risk.iat[i] == "CAUTION":
            parts.append("⚠️ 과열/추격 주의")
        if valid.iat[i]:
            st = int(out["EXIT_STOP_TIGHT"].iat[i]); tp = int(out["EXIT_TP_QUICK"].iat[i])
            be = int(out["BE_TRIGGER_PRICE"].iat[i])
            ft = int(out["FAST_TP_PRICE"].iat[i])
            parts.append(
                f"손절 {st:,}({stop_pct:.0f}%)·본전전환 {be:,}(+{BE_TRIGGER_PCT:.0f}%)"
                f"·1차익절 {tp:,}(+{tp_pct:.0f}%)"
            )
            parts.append(
                f"⚡ D+{FAST_TP_DAY}내 {ft:,}(+{FAST_TP_PCT:.0f}%) 도달 시 절반 익절"
                "+잔여 본전 상향 (실측: 초반 급등 후 되돌림 -0.9%)"
            )
        return " · ".join(parts)

    out["EXIT_PLAN_NOTE"] = [_note(i) for i in range(len(out))]
    return out


def exit_summary(df: pd.DataFrame) -> dict:
    if df is None or "EXIT_ROUTE_RISK" not in getattr(df, "columns", []):
        return {}
    vc = df["EXIT_ROUTE_RISK"].value_counts().to_dict()
    return {"high_avoid": int(vc.get("HIGH_AVOID", 0)),
            "caution": int(vc.get("CAUTION", 0)),
            "ok": int(vc.get("OK", 0))}
