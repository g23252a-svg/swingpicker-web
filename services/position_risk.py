# -*- coding: utf-8 -*-
"""[v61] 이 픽에서 실제로 얼마를 잃는가 — 리스크 표기 SSOT.

■ 무엇이 있었나 (2026-08-13 사용자 실손실)
  씨어스(458870)는 2026-08-11 배치의 프로덕션 픽 1순위였다
  (LDY_RANK=1 · ALPHA_SCORE=97.8 · RR 4.63 · 켈리_수량 29주).
  엔진이 의도한 최대 손실은 **29주 × (25,250 − 23,200) = 59,450원**이다.
  사용자 실제 손실은 **-230,000원 = 의도의 3.9배**였다.

  그런데 화면이 사용자에게 보여준 리스크 숫자는 이랬다:

    <span class="lbl" title="포지션당 최대 손실율. 손절 발동 시 잃을 자본 비중">
      MAX_LOSS</span>  →  5.0%

  툴팁은 "손절 발동 시 잃을 자본 비중"을 약속한다. 그런데 이 값(MAX_LOSS_PCT)은
  `trade_plan.py`에서 **"시총 기반 캡"**으로 정의된 별개 값이고, 씨어스의 실제
  손절폭(STOP_PCT)은 **8.12%** — 화면 표기의 **1.62배**였다.

■ 그 캡은 거의 캡하지 않는다 (전수 실측)
  120개 배치 51,049행 중 **STOP_PCT > MAX_LOSS_PCT가 44,901행(88.0%)**,
  **120일 전부에서 발생**. TOP_PICK만 보면 **107/109행(98.2%)**.
  중위값은 손절폭 9.53% vs 캡 5.00%.
  → 선언된 캡이 사실상 한 번도 구속하지 않는 **죽은 게이트**(v56 유형)인데,
    그 값이 "최대 손실"이라는 이름으로 화면에 나가고 있었다.

  '최대손실 5%'로 포지션을 잡으면 실제 손절폭 9.5% 기준보다 **약 1.9배**
  크게 산다. 손실이 의도를 초과하는 경로가 여기 있다.

■ 원화 리스크가 화면에 아예 없었다
  UI 전체에서 "손절 시 몇 원을 잃는가"를 보여주는 곳이 없었다. 수량(29주)과
  금액(76만원)은 있는데, **잃을 금액(5.9만원)이 없다.** 포지션 크기를 정할 때
  기준이 되는 단 하나의 숫자가 빠져 있었던 것이다.

■ 이 모듈이 하는 일
  가격에서 직접 계산한 **실제 손절 손실**을 %와 원화로 제공하고, 시총 캡은
  캡이라고 부른다. 표기가 사실과 어긋나면 리포트로 잡는다.

■ 하지 않는 일
  손절가 산식을 바꾸지 않는다. 캡을 강제로 구속시키지도 않는다 — 그러면 손절
  위치가 전부 이동하고, 그 변경은 실측 검증 없이 할 수 없다. v61은 **계산이
  아니라 표기**를 사실에 맞춘다.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger("position_risk")

ENTRY_COL = "추천매수가"
STOP_COL = "손절가"
QTY_COLS = ("켈리_수량", "추천수량")
CLOSE_COL = "종가"
STOP_PCT_COL = "STOP_PCT"
CAP_COL = "MAX_LOSS_PCT"

# 화면에서 써야 하는 라벨. 캡을 손실이라고 부르지 않는다.
LOSS_LABEL = "손절 시 손실"
LOSS_TOOLTIP = ("손절가까지 밀렸을 때 실제로 잃는 금액과 비율. "
                "추천 수량 기준으로 계산한 값이다 — 포지션 크기를 정할 때 "
                "기준이 되는 숫자다.")
CAP_LABEL = "시총 손절폭 캡(참고)"
CAP_TOOLTIP = ("시가총액 기준으로 둔 손절폭 상한 '참고값'이다. **손절 시 잃는 "
               "금액이 아니다.** 실측 120개 배치에서 실제 손절폭이 이 캡을 "
               "88% 초과했다(중위 9.5% vs 캡 5.0%) — 구속력이 거의 없다.")


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
    if f != f:          # NaN
        return None
    return f


def entry_price(row: Mapping[str, Any]) -> Optional[float]:
    """손절가가 걸려 있는 기준가. 추천매수가가 없으면 종가로 폴백."""
    for key in (ENTRY_COL, CLOSE_COL):
        v = _num(row, key)
        if v and v > 0:
            return v
    return None


def quantity(row: Mapping[str, Any]) -> Optional[float]:
    for key in QTY_COLS:
        v = _num(row, key)
        if v is not None and v > 0:
            return v
    return None


def stop_loss_pct(row: Mapping[str, Any]) -> Optional[float]:
    """**가격에서 직접 계산한** 손절 손실률(%, 양수).

    STOP_PCT 컬럼을 그대로 믿지 않고 가격으로 검산한다. 둘이 어긋나면
    (v61 이전 화면이 그랬듯) 어느 쪽이 진실인지 알 수 없기 때문이다.
    """
    buy, stop = entry_price(row), _num(row, STOP_COL)
    if not buy or not stop or stop <= 0 or stop >= buy:
        return None
    return (buy - stop) / buy * 100.0


def stop_loss_won(row: Mapping[str, Any]) -> Optional[float]:
    """손절 시 잃는 **금액(원)** — 화면에 없던 그 숫자."""
    buy, stop, qty = entry_price(row), _num(row, STOP_COL), quantity(row)
    if not buy or not stop or qty is None or stop >= buy:
        return None
    return (buy - stop) * qty


def position_won(row: Mapping[str, Any]) -> Optional[float]:
    buy, qty = entry_price(row), quantity(row)
    if not buy or qty is None:
        return None
    return buy * qty


def cap_binds(row: Mapping[str, Any]) -> Optional[bool]:
    """시총 캡이 실제로 손절폭을 구속했는가. 판단 불가면 None."""
    cap = _num(row, CAP_COL)
    actual = stop_loss_pct(row)
    if cap is None or cap <= 0 or actual is None:
        return None
    return actual <= cap + 1e-9


def risk_line(row: Mapping[str, Any]) -> str:
    """한 줄 리스크 문장. 계산 불가한 부분은 말하지 않는다."""
    qty = quantity(row)
    pos = position_won(row)
    won = stop_loss_won(row)
    pct = stop_loss_pct(row)
    if qty is None or pos is None:
        return ""
    bits = [f"{qty:,.0f}주 · {pos/10000:,.1f}만원"]
    if won is not None and pct is not None:
        bits.append(f"{LOSS_LABEL} -{won:,.0f}원(-{pct:.1f}%)")
    return " · ".join(bits)


def risk_consistency(row: Mapping[str, Any]) -> dict:
    """표기가 사실과 어긋나는지 점검. (조용히 넘기지 않기 위한 리포트)"""
    rep = {"ok": True, "problems": [], "stop_loss_pct": stop_loss_pct(row),
           "stop_loss_won": stop_loss_won(row), "cap": _num(row, CAP_COL),
           "cap_binds": cap_binds(row)}
    actual = rep["stop_loss_pct"]
    declared = _num(row, STOP_PCT_COL)
    if actual is not None and declared is not None:
        if abs(actual - declared) > 0.5:
            rep["problems"].append(
                f"STOP_PCT({declared:.2f}%)와 가격 기반 손절폭({actual:.2f}%)이 다르다")
    cap = rep["cap"]
    if cap is not None and cap > 0 and actual is not None and actual > cap + 1e-9:
        rep["problems"].append(
            f"시총 캡({cap:.1f}%)보다 실제 손절폭({actual:.2f}%)이 크다 — "
            "캡을 '최대 손실'로 표기하면 리스크를 과소 표기한다")
    rep["ok"] = not rep["problems"]
    return rep
