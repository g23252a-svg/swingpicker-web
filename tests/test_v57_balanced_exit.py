# -*- coding: utf-8 -*-
"""[v57] 승률과 기대값 둘 다 — 부분익절 파라미터 재조정 (+2%·절반 → +3%·1/4).

■ 사장님 요구
  "승률도 중요하지만 어느정도 적정타협해서 승률과 수익률 2가지 토끼를
   가능하면 다 잡는 방향이 좋은데"

■ 문제 (v55/v57 실측)
  공식 픽은 우측 꼬리 분포다 — 15일 평균 +6.20%인데 중위 -2.70%,
  상위 2건(+78.6%/+52.0%)을 빼면 -2.89%. 즉 수익의 대부분이 드문 대박에서 온다.
  그래서 조기 부분익절 비율이 크면 유일한 수익원을 잘라낸다:
      기준(목표가까지 보유)  승률 41.4% · 평균 +1.26% · 복리 +33.8% · MDD -33.5%
      +2%·절반(구 권고값)    승률 65.5% · 평균 +0.55%(기준의 43%) · 복리 +14.2%
  승률은 크게 오르지만(McNemar p<0.0001) 기대값이 반감한다.

■ 격자탐색 (트리거 2~6% × 익절비율 25/33/50% × BE 유무 = 30조합, BH-FDR 보정)
  채택 = **+3% 트리거 · 25% 부분익절 · +5% 본전스톱**
    top-3 (n=87)   승률 41.4% → 59.8% (+18.4%p) · 평균 +1.12% (기준의 89%)
                   복리 +32.5% · MDD -16.5% · p_BH=0.0001
                   ΔIS +15.9%p · ΔOOS +27.8%p (양 구간 개선)
    top-5 (n=144)  승률 44.4% → 61.1% (+16.7%p) · 평균 88% 보존 · p<0.0001
                   ΔIS +14.9 · ΔOOS +23.3          ← 준-홀드아웃 재현
    민감도 평탄: 트리거 2%(p<0.0001)·3%(0.0001)·4%(0.002) 전부 유의,
                 익절비율 25%가 기대값 보존에 최적 → 칼날 위 최적화 아님
  → 승률 +18%p를 얻고 기대값은 11%만 내놓으며 낙폭은 절반이 된다.

■ 범위
  exit_plan은 표시 전용 레이어다(공식 산식·손절가 컬럼 불변). 바뀌는 것은
  화면이 권고하는 값과 그 문구뿐이며, 실제 청산은 사용자가 한다.
  pick_reliability는 '화면이 지시하는 것을 그대로 측정'해야 하므로 같은 값을 쓴다.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import exit_plan as ep  # noqa: E402
from services import pick_reliability as pr  # noqa: E402


# ── ① 채택값이 코드에 박혀 있는가 ──

def test_adopted_parameters():
    assert ep.FAST_TP_PCT == 3.0, "트리거 +3% (격자탐색 채택값)"
    assert ep.FAST_TP_FRACTION == 0.25, "부분익절 1/4 (꼬리 보존)"
    assert ep.FAST_TP_DAY == 2
    assert ep.BE_TRIGGER_PCT == 5.0


def test_reliability_matches_exit_plan():
    """화면 권고와 측정 기준이 어긋나면 통계가 다른 규칙을 재는 셈이 된다."""
    assert pr.FAST_TP_PCT * 100 == ep.FAST_TP_PCT
    assert pr.FAST_TP_FRACTION == ep.FAST_TP_FRACTION
    assert pr.FAST_TP_DAYS == ep.FAST_TP_DAY
    assert pr.BE_TRIGGER_PCT * 100 == ep.BE_TRIGGER_PCT


# ── ② 시뮬레이터가 새 비율대로 동작하는가 ──

def _fut(rows):
    return pd.DataFrame(rows, columns=["시가", "고가", "저가", "종가"])


def test_partial_exit_keeps_three_quarters_for_the_tail():
    """+3% 도달 후 크게 오르면 잔여 75%가 그 상승을 먹어야 한다.

    구값(절반)이면 상승분의 50%만 먹는다 — 꼬리 보존이 이 패치의 요점이다.
    """
    fut = _fut([(100, 104, 99, 103)] + [(103, 140, 102, 138)] * 4)
    v = pr._simulate(fut, "disc")
    # +3% 부분익절(0.25×3%) + 잔여 0.75가 +10% 1차익절선에서 청산
    assert abs(v - (0.25 * 0.03 + 0.75 * 0.10)) < 1e-9, v


def test_partial_exit_still_locks_a_win_on_reversal():
    """+3% 찍고 폭락해도 부분익절 + 본전스톱으로 손실이 얕다 (승률 방어)."""
    fut = _fut([(100, 104, 99, 103), (103, 103, 70, 71)] + [(71, 72, 70, 71)] * 3)
    v = pr._simulate(fut, "disc")
    assert v > 0, f"부분익절+본전스톱이면 플러스로 끝나야 한다: {v}"
    assert v < 0.02


def test_no_partial_exit_below_trigger():
    """+3% 미달이면 부분익절이 발동하지 않는다 (구값 +2%와 구분)."""
    fut = _fut([(100, 102.5, 99, 102)] * 5)
    v = pr._simulate(fut, "disc")
    assert abs(v - 0.02) < 1e-9, v


def test_trigger_only_within_two_days():
    """3일차에 +3%를 찍어도 부분익절은 없다 (D+2 창)."""
    fut = _fut([(100, 100.5, 99, 100), (100, 100.5, 99, 100),
                (100, 104, 99, 103), (103, 104, 102, 103), (103, 104, 102, 103)])
    v = pr._simulate(fut, "disc")
    assert abs(v - 0.03) < 1e-9, v


# ── ③ 화면 문구가 트레이드오프를 정직하게 말하는가 ──

def _note(**ov):
    row = {"추천매수가": 10000.0, "ROUTE": "ATTACK", "종목코드": "000001"}
    row.update(ov)
    out = ep.add_exit_plan_columns(pd.DataFrame([row]))
    return str(out.iloc[0]["EXIT_PLAN_NOTE"])


def test_note_shows_new_fraction_and_trigger():
    n = _note()
    assert "+3%" in n
    assert "25%" in n, f"1/4 익절이 문구에 없다: {n}"
    assert "절반 익절" not in n, "구 문구(절반)가 남아 있다"


def test_note_states_both_sides_of_the_tradeoff():
    n = _note()
    assert "41%→60%" in n or "41%→60" in n, f"승률 개선 수치 누락: {n}"
    assert "89" in n, f"기대수익 보존 수치 누락: {n}"
    assert "목표가(TP1~TP3)까지 보유" in n


def test_evidence_recorded_in_source():
    src = (ROOT / "exit_plan.py").read_text(encoding="utf-8")
    for token in ["p_BH=0.0001", "top-5", "민감도 평탄", "격자 탐색"]:
        assert token in src, f"근거 미기록: {token}"
    # 구값의 근거(v33)도 지운 게 아니라 남겨 비교 가능해야 한다
    assert "n=526" in src, "v33 구 근거를 삭제하면 왜 바꿨는지 알 수 없다"
