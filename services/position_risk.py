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

■ [v66] 갭이 손절선을 조용히 코앞으로 당긴다
  손절가는 **추천매수가(전일 종가 기준)** 에 고정되고, 다음 날 실제 체결가로
  다시 기준을 잡지 않는다. 그래서 시가가 갭다운하면 계획한 손절폭이 산술적으로
  줄어든다. 화면은 계획값(-8%)만 보여주므로 사용자는 그 사실을 알 수 없다.

  2026-08-19 실측 (전날 8/18 배치 26건):
    다음날 시가 갭 중위 **-3.84%** → 실효 손절폭 중위 **-4.18%** (계획 -8.03%)
    여유 4% 미만 9/26건 · 시가가 이미 손절가 이하 1건
    · 세미파이브(공식 매수): 16,100 → 시가 15,410(-4.3%), 손절 15,370
      → 실효 손절폭 **-0.26%**. 다음 날 바로 손절 체결.
    · 아모텍(근접 1위): 12,330 → 시가 11,960, 손절 11,340 → 실효 -5.2%
      8/19 저가 11,410으로 손절선 위 0.6%까지 붙었다.

  게이트 통과 상위10 전수(204건·21일)에서 갭 구간별:
    갭 ≤ -3%  (n=10) 실효 중위 -2.74% · **손절률 90%** · 수익 -1.94%
    갭 -3~-1% (n=32) 실효 중위 -4.21% · 손절률 41% · 수익 +3.45%
    갭 -1~+1% (n=115) 실효 중위 -7.16% · 손절률 25% · 수익 +2.14%
  실효 손절폭 구간별 손절률: ≤-8% 30% · -8~-6% 33% · -6~-4% 35% · **-4~0% 47%**
  → 경고선을 **실효 -4%** 로 둔다(그 아래에서 손절률이 뚜렷이 올라간다).
     표본이 얇다(갭≤-3% n=10). 그래서 **진입을 막지 않고 표시만 한다.**

■ [v66] 하지 않는 일 (다시)
  손절가를 실제 체결가로 재기준하지 않는다. 재기준하면 손절폭은 -8%로 유지되지만
  손절가 자체가 내려가 손실 **금액**이 커진다 — 자금 흐름을 바꾸는 변경이므로
  실측 검정을 통과하기 전에는 하지 않는다(별건).
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


# ══════════════════════════════════════════════════════════════════
#  [v66] 갭 민감도 — 계획 손절폭이 실제로 얼마가 되는가
# ══════════════════════════════════════════════════════════════════
#
# 배치는 전날 21:30에 돌고 진입은 다음 날 시가다. 배치 시점에 실제 체결가는
# 알 수 없으므로 **시나리오**로 보여준다. 시나리오 갭은 임의로 고르지 않고
# 실측 분포에서 가져온다 — 사이징된 후보 112건의 다음날 시가 갭:
#   p5 -4.26% · p25 -1.28% · 중위 +0.63% · p75 +1.72% · p95 +6.26%
# 즉 아래 두 값은 "흔한 나쁜 날(4건 중 1건)"과 "드문 나쁜 날(20건 중 1건)"이다.
GAP_P25_PCT = -1.28
GAP_P05_PCT = -4.26
GAP_SCENARIOS = (GAP_P25_PCT, GAP_P05_PCT)

# 실효 손절폭이 이 선보다 얇으면 경고. 실측 근거는 모듈 docstring 참조
# (실효 -4~0% 구간 손절률 47% vs -8% 이하 30%).
THIN_STOP_PCT = -4.0

GAP_LABEL = "갭 시 손절폭"
GAP_TOOLTIP = (
    "손절가는 추천매수가에 고정되어 있고 실제 체결가로 다시 잡지 않는다. "
    "그래서 시가가 갭다운하면 계획한 손절폭이 그만큼 줄어든다. "
    "2026-08-19 실측: 갭 중위 -3.84%로 실효 손절폭이 -8.03% → -4.18%가 됐고, "
    "공식 매수 세미파이브는 실효 -0.26%까지 좁아져 다음 날 바로 손절됐다."
)


def planned_stop_pct(row: Mapping[str, Any]) -> Optional[float]:
    """계획 손절폭(%, 음수). 추천매수가 기준 — 배치가 의도한 값."""
    buy, stop = entry_price(row), _num(row, STOP_COL)
    if not buy or not stop or stop <= 0 or stop >= buy:
        return None
    return (stop - buy) / buy * 100.0


def effective_stop_pct(row: Mapping[str, Any],
                       fill_price: Optional[float]) -> Optional[float]:
    """**실제 체결가 기준** 손절폭(%, 음수). 체결가를 모르면 None.

    이것이 사용자가 실제로 감수하는 폭이다. 계획값과 다르면 다른 게 맞고,
    화면은 둘 다 보여줘야 한다.
    """
    stop = _num(row, STOP_COL)
    try:
        fill = float(fill_price) if fill_price is not None else None
    except (TypeError, ValueError):
        return None
    if fill is None or fill != fill or fill <= 0:
        return None
    if not stop or stop <= 0:
        return None
    return (stop - fill) / fill * 100.0


def stop_pct_at_gap(row: Mapping[str, Any], gap_pct: float) -> Optional[float]:
    """시가가 추천매수가 대비 gap_pct% 갭일 때의 실효 손절폭."""
    buy = entry_price(row)
    if not buy:
        return None
    return effective_stop_pct(row, buy * (1.0 + float(gap_pct) / 100.0))


def gap_sensitivity(row: Mapping[str, Any]) -> dict:
    """계획 손절폭과 시나리오별 실효 손절폭. 계산 불가하면 빈 dict."""
    plan = planned_stop_pct(row)
    if plan is None:
        return {}
    out = {"planned_pct": plan, "scenarios": []}
    for g in GAP_SCENARIOS:
        eff = stop_pct_at_gap(row, g)
        if eff is None:
            continue
        out["scenarios"].append({"gap_pct": g, "effective_pct": eff,
                                 "thin": eff > THIN_STOP_PCT})
    out["thin_at_p25"] = any(s["thin"] and s["gap_pct"] == GAP_P25_PCT
                             for s in out["scenarios"])
    out["thin_at_p05"] = any(s["thin"] and s["gap_pct"] == GAP_P05_PCT
                             for s in out["scenarios"])
    # 계획 자체가 이미 얇은 경우도 경고 대상이다 (세미파이브 -4.53%)
    out["thin_as_planned"] = plan > THIN_STOP_PCT
    return out


def gap_table_line(row: Mapping[str, Any]) -> str:
    """정보 표시 — 모든 종목에 붙는 사실. 경고가 아니다.

    갭이 나면 손절폭이 얼마가 되는지를 그냥 적는다. 배치 시점에는 실제
    체결가를 모르므로 이것이 화면이 정직하게 말할 수 있는 최대치다.
    """
    s = gap_sensitivity(row)
    if not s or not s.get("scenarios"):
        return ""
    bits = [f"갭 {sc['gap_pct']:.1f}% → {sc['effective_pct']:.1f}%"
            for sc in s["scenarios"]]
    return f"{GAP_LABEL}: 계획 {s['planned_pct']:.1f}% · " + " · ".join(bits)


def gap_risk_line(row: Mapping[str, Any]) -> str:
    """**경고** 한 줄. 경고할 것이 없으면 빈 문자열.

    경고선을 p05(-4.26%) 갭에 두면 실측상 22종목 중 20종목이 걸린다 — 그건
    경고가 아니라 배경 소음이고, v64에서 집중도 문턱을 조인 것과 같은 이유로
    쓸모가 없다. 그래서 **흔한 나쁜 날(p25, 갭 -1.28%)에도 이미 얇아지는
    종목**만 경고한다. 실측 적용 결과 8/18 배치 2/28건 · 8/20 배치 1/22건.
    (p05 시나리오 숫자는 경고 대신 gap_table_line으로 항상 보여준다.)
    """
    s = gap_sensitivity(row)
    if not s:
        return ""
    if s.get("thin_as_planned"):
        return (f"⚠️ {GAP_LABEL}: 계획 자체가 {s['planned_pct']:.1f}%로 "
                f"실효 {THIN_STOP_PCT:.0f}%보다 얇다 — 갭 없이도 손절선이 코앞이다")
    if not s.get("thin_at_p25"):
        return ""
    eff = next((sc["effective_pct"] for sc in s["scenarios"]
                if sc["gap_pct"] == GAP_P25_PCT), None)
    if eff is None:
        return ""
    return (f"⚠️ {GAP_LABEL}: 계획 {s['planned_pct']:.1f}%인데 흔한 갭"
            f"({GAP_P25_PCT:.1f}%, 4일 중 1일)만 나도 {eff:.1f}%로 좁아진다 — "
            f"실효 {THIN_STOP_PCT:.0f}% 아래 구간 손절률 47%(vs -8% 이하 30%)")
