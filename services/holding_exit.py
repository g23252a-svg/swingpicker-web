# -*- coding: utf-8 -*-
"""[v67] 보유 포지션에 청산 규율이 없다 — 진입가 고정 손절 SSOT.

■ 무엇이 있었나 (2026-08-24 배치 전수)
  사용자 실손실: 8월 중 +100만원까지 갔던 수익이 +30만원으로 줄었다(반납 70%).

  같은 날 배치의 CARRY(보유 관리) 58종목을 전수 조사했다.

    진입가(CARRY_FROM_CLOSE) 대비 현재     평균 -30.4% · 중위 -33.4% · 최악 -66.7%
    진입가 기준 -7% 손절선을 이미 관통      **55/58건**
    고정 손절을 지켰다면                   평균 -6.7% · 중위 -7.0% · 최악 -7.0%
    종목당 차이                            **+23.8%p** (페어드 t=+9.33, p<1e-6)

  개별 사례가 더 분명하다:
    이루온     진입 2026-05-14 → **다음날** 손절선 관통 → 오늘까지 101일 보유 · -66.7%
    HLB파나진  진입 2026-04-09 → 4/23 관통 → 오늘까지 137일 보유 · -54.7%
    온코크로스 진입 2026-04-28 → 4/30 관통 → -61.5%

■ 왜 그렇게 됐나 — 두 가지 고장이 겹쳐 있다
  ① 손절 기준선이 진입가에 고정되지 않는다
     CARRY 행은 매일 재분석되면서 `추천매수가`가 당일 종가로 재기준되는 경우가
     있다. `EXIT_STOP_TIGHT = 추천매수가 × 0.93`이므로 **손절선이 가격을 따라
     내려간다**. 한국화장품 실측: 종가 6,660→6,420→6,280→6,250 로 내리는 동안
     손절선도 6,194→5,971→5,840→5,812 로 같이 내려갔다. 이런 손절선은
     **원리적으로 발동할 수 없다.** 화면은 늘 "손절 -7%"라고 적는다.
  ② 관통해도 아무 일도 일어나지 않는다
     기준선이 고정된 나머지 행들은 손절선이 한참 전에 뚫렸는데도 CARRY로 계속
     남는다. 종결 지시를 내리는 코드가 없다.

■ 화면에도 없었다
  v57이 검증한 청산 규율(+3% 25% 익절 → 본전스톱 → +10% 1차익절 · -7% 손절)은
  실측에서 **승률 41.4%→59.8%, MDD -33.5%→-16.5%, 기대값 89% 보존**
  (격자탐색 30조합 · BH-FDR p=0.0001)이었다. 그런데 사용자가 결정을 내리는
  '오늘' 탭(`components/decision_center.py`)에는 exit·본전·익절 참조가 **0건**이다.
  종목 상세 페이지를 눌러 들어가야만 보인다.

  현재 데이터로 재현해도 방향은 같다(알파 퍼널 254건·26일, 상위10 기준):
    보유: 승률 42% · 최고점 +35.6% → 최종 **-3.2%** (반납 38.8%p) · MDD -28.6%
    규율: 승률 62% · 최고점 +34.8% → 최종 +17.7% (반납 17.2%p) · MDD **-16.3%**
  평균수익 차이는 유의하지 않다(Δ+0.65%p, t=0.97, p=0.34). 채택 근거는
  **반납률과 낙폭**이지 기대수익이 아니다 — 그렇게만 인용해야 한다.

■ 이 모듈이 하는 일
  보유 포지션의 손절·익절 기준을 **진입가에 고정해서** 계산하고, 오늘 무엇을
  해야 하는지를 한 줄로 만든다. 기준선은 절대 재기준하지 않는다.

■ 하지 않는 일
  · 자동으로 팔지 않는다(앱은 주문을 내지 않는다).
  · CARRY 목록에서 종목을 **지우지 않는다**. 사용자가 실제로 들고 있을 수 있고,
    목록에서 사라지면 오히려 관리가 안 된다. 대신 '진작 종결됐어야 함'을
    상태로 표시한다.
  · 손절폭(-7%)을 바꾸지 않는다. v57 격자탐색이 고른 값을 그대로 쓴다.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("holding_exit")

# 기준값은 pick_reliability(연구 정의)와 exit_plan(화면 표기)의 SSOT를 따른다.
try:                                                # pragma: no cover - 임포트 경로
    from services.pick_reliability import (
        BE_TRIGGER_PCT, FAST_TP_PCT, STOP_PCT_DISC, TP_QUICK_PCT,
    )
except Exception as e:                              # 라이브러리 부재 등
    logger.warning(f"[v67] 청산 상수 참조 실패 ({e}) → 폴백값 사용")
    STOP_PCT_DISC, FAST_TP_PCT, BE_TRIGGER_PCT, TP_QUICK_PCT = -0.07, 0.03, 0.05, 0.10

ANCHOR_COL = "CARRY_FROM_CLOSE"     # 보유 진입 기준가 — 재기준 금지
CLOSE_COL = "종가"
AGE_COL = "CARRY_AGE_DAYS"
ROUTE_COL = "ROUTE"

# 부여하는 컬럼
COL_ANCHOR = "HOLD_ANCHOR_PRICE"
COL_STOP = "HOLD_STOP_PRICE"
COL_RET = "HOLD_RET_PCT"
COL_STATE = "HOLD_EXIT_STATE"
COL_ACTION = "HOLD_EXIT_ACTION"

STATE_STOP = "STOP_BREACHED"        # 손절선 관통 — 진작 종결됐어야 한다
STATE_TP = "TAKE_PROFIT"            # 1차 익절선 도달
STATE_BE = "MOVE_TO_BE"             # 본전스톱 전환선 도달
STATE_FAST = "FAST_TP"              # 빠른 부분익절선 도달
STATE_HOLD = "HOLD"
STATE_UNKNOWN = "UNKNOWN"

# 심각도 — 화면 정렬용. 손절 관통이 가장 급하다.
SEVERITY = {STATE_STOP: 0, STATE_TP: 1, STATE_BE: 2, STATE_FAST: 3,
            STATE_HOLD: 8, STATE_UNKNOWN: 9}

ACTIONABLE = (STATE_STOP, STATE_TP, STATE_BE, STATE_FAST)

# 경계값 허용오차(%). 정확히 손절선에 걸린 포지션은 '걸렸다'로 센다 —
# 부동소수 때문에 -7.000000001%가 HOLD로 빠지면 규율이 한 칸씩 새어나간다.
_EPS_PCT = 1e-6


def _num(row: Mapping[str, Any], key: str) -> Optional[float]:
    if key not in row:
        return None
    v = row.get(key)
    try:
        if v is None:
            return None
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def anchor_price(row: Mapping[str, Any]) -> Optional[float]:
    """보유 진입 기준가. **당일 종가로 재기준하지 않는다.**

    이 함수가 종가를 폴백으로 쓰면 손절선이 가격을 따라 내려가는 그 버그가
    되살아난다. 기준가를 모르면 None을 돌려주고 판정을 포기한다.
    """
    v = _num(row, ANCHOR_COL)
    return v if v and v > 0 else None


def stop_price(row: Mapping[str, Any]) -> Optional[float]:
    a = anchor_price(row)
    return None if a is None else a * (1.0 + STOP_PCT_DISC)


def hold_return_pct(row: Mapping[str, Any]) -> Optional[float]:
    a, c = anchor_price(row), _num(row, CLOSE_COL)
    if a is None or c is None or c <= 0:
        return None
    return (c / a - 1.0) * 100.0


def exit_state(row: Mapping[str, Any]) -> str:
    """오늘 이 보유 종목이 어떤 상태인가. 판정 불가면 UNKNOWN."""
    r = hold_return_pct(row)
    if r is None:
        return STATE_UNKNOWN
    if r <= STOP_PCT_DISC * 100.0 + _EPS_PCT:
        return STATE_STOP
    if r >= TP_QUICK_PCT * 100.0 - _EPS_PCT:
        return STATE_TP
    if r >= BE_TRIGGER_PCT * 100.0 - _EPS_PCT:
        return STATE_BE
    if r >= FAST_TP_PCT * 100.0 - _EPS_PCT:
        return STATE_FAST
    return STATE_HOLD


def action_line(row: Mapping[str, Any]) -> str:
    """무엇을 해야 하는지 한 줄. 조치할 것이 없으면 빈 문자열."""
    st = exit_state(row)
    r = hold_return_pct(row)
    if st in (STATE_HOLD, STATE_UNKNOWN) or r is None:
        return ""
    name = str(row.get("종목명", "")).strip()
    age = _num(row, AGE_COL)
    age_txt = f" · 보유 {age:.0f}일" if age is not None and age > 0 else ""
    sp = stop_price(row)
    if st == STATE_STOP:
        return (f"⛔ {name} {r:+.1f}% — 손절선({sp:,.0f}원, 진입 {STOP_PCT_DISC:+.0%})을 "
                f"이미 지났다{age_txt}. 규율대로면 진작 종결됐어야 한다")
    if st == STATE_TP:
        return (f"💰 {name} {r:+.1f}% — 1차 익절선(+{TP_QUICK_PCT:.0%}) 도달{age_txt}. "
                f"익절 검토")
    if st == STATE_BE:
        return (f"🛡️ {name} {r:+.1f}% — 본전스톱 전환선(+{BE_TRIGGER_PCT:.0%}) 도달"
                f"{age_txt}. 손절선을 진입가로 올릴 것")
    return (f"⚡ {name} {r:+.1f}% — 빠른 익절선(+{FAST_TP_PCT:.0%}) 도달{age_txt}. "
            f"일부 익절 + 잔여 본전 상향 검토")


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """보유 상태 컬럼 부여. 결정 컬럼은 건드리지 않는다(표시 전용)."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    out[COL_ANCHOR] = [anchor_price(r) for _, r in out.iterrows()]
    out[COL_STOP] = [stop_price(r) for _, r in out.iterrows()]
    out[COL_RET] = [hold_return_pct(r) for _, r in out.iterrows()]
    out[COL_STATE] = [exit_state(r) for _, r in out.iterrows()]
    out[COL_ACTION] = [action_line(r) for _, r in out.iterrows()]
    return out


def holdings_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0 or ROUTE_COL not in df.columns:
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)
    return df[ROUTE_COL].astype(str).str.strip().str.upper().eq("CARRY")


def summary(df: pd.DataFrame) -> dict:
    """보유 전체 요약 — '오늘 정리해야 할 것이 몇 건인가'."""
    empty = {"n": 0, "actionable": 0, "counts": {}, "lines": [], "line": "",
             "worst_pct": None, "median_pct": None}
    if df is None or len(df) == 0:
        return empty
    held = df[holdings_mask(df)]
    if not len(held):
        return empty
    rets, states, lines = [], [], []
    for _, r in held.iterrows():
        st = exit_state(r)
        states.append(st)
        v = hold_return_pct(r)
        if v is not None:
            rets.append(v)
        ln = action_line(r)
        if ln:
            lines.append((SEVERITY.get(st, 9), v if v is not None else 0.0, ln))
    counts = {s: states.count(s) for s in set(states)}
    lines.sort(key=lambda x: (x[0], x[1]))
    n_stop = counts.get(STATE_STOP, 0)
    out = {
        "n": int(len(held)),
        "actionable": int(sum(counts.get(s, 0) for s in ACTIONABLE)),
        "counts": counts,
        "lines": [x[2] for x in lines],
        "worst_pct": float(min(rets)) if rets else None,
        "median_pct": float(np.median(rets)) if rets else None,
        "line": "",
    }
    if n_stop:
        out["line"] = (f"보유 {out['n']}종목 중 **{n_stop}종목이 손절선을 지났다** — "
                       f"중위 {out['median_pct']:+.1f}% · 최악 {out['worst_pct']:+.1f}%")
    elif out["actionable"]:
        out["line"] = (f"보유 {out['n']}종목 중 {out['actionable']}종목에 "
                       f"오늘 조치가 필요하다 — 중위 {out['median_pct']:+.1f}%")
    return out
