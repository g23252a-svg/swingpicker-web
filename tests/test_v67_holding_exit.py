# -*- coding: utf-8 -*-
"""[v67] 무엇을 사는지만 있고 무엇을 파는지가 없었다.

■ 사용자 실손실 (2026-08)
  8월 중 **+100만원**까지 갔던 수익이 **+30만원**으로 줄었다(반납 70%).

■ 전수 조사 (2026-08-24 배치, CARRY 58종목)
    진입가(CARRY_FROM_CLOSE) 대비  평균 -30.4% · 중위 -33.4% · 최악 -66.7%
    진입가 기준 -7% 손절선 관통     **55/58건**
    고정 손절을 지켰다면            평균 -6.7% · 중위 -7.0% · 최악 -7.0%
    종목당 차이                     **+23.8%p** (페어드 t=+9.33, p<1e-6)

    이루온     진입 2026-05-14 → **다음날** 손절선 관통 → 101일 보유 · -66.7%
    HLB파나진  진입 2026-04-09 → 4/23 관통 → 137일 보유 · -54.7%
    온코크로스 진입 2026-04-28 → 4/30 관통 → -61.5%

■ 고장 ① 손절 기준선이 진입가에 고정되지 않는다
  CARRY 행이 매일 재분석되면서 `추천매수가`가 당일 종가로 재기준되는 경우가
  있고, `EXIT_STOP_TIGHT = 추천매수가 × 0.93`이므로 **손절선이 가격을 따라
  내려간다.** 한국화장품 실측:
    종가   6,660 → 6,420 → 6,280 → 6,250
    손절선 6,194 → 5,971 → 5,840 → 5,812
  이런 손절선은 원리적으로 발동하지 않는다. 화면은 늘 "손절 -7%"라고 적는다.

■ 고장 ② 관통해도 아무 일도 없다
  기준선이 고정된 행들은 손절선이 한참 전에 뚫렸는데도 CARRY로 계속 남는다.
  종결을 지시하는 코드가 없었다.

■ 고장 ③ 검증된 청산 규율이 '오늘' 탭에 없었다
  v57 규율(+3% 25% 익절 → 본전스톱 → +10% 1차익절 · -7% 손절)은 격자탐색
  30조합·BH-FDR에서 승률 41.4%→59.8%, MDD -33.5%→-16.5%, 기대값 89% 보존으로
  채택됐다. 그런데 `components/decision_center.py`에 exit·본전·익절 참조가
  **0건**이었다 — 종목 상세를 눌러야만 보였다.
  현재 데이터 재현(알파 퍼널 254건·26일, 상위10):
    보유 승률 42% · 최고점 +35.6% → 최종 -3.2%(반납 38.8%p) · MDD -28.6%
    규율 승률 62% · 최고점 +34.8% → 최종 +17.7%(반납 17.2%p) · MDD -16.3%
  평균수익 차이는 유의하지 않다(Δ+0.65%p, t=0.97, p=0.34).
  **채택 근거는 반납률·낙폭이지 기대수익이 아니다.**

■ 이 파일이 고정하는 것
  1. 기준가는 **진입가**다. 종가로 폴백하면 실패한다(고장 ①의 재발).
  2. 기준가를 모르면 판정하지 않는다(UNKNOWN) — 추측해서 손절선을 만들지 않는다.
  3. 손절선 관통은 STOP_BREACHED이고 가장 급한 상태다.
  4. 실배치에서 손절 관통이 실제로 검출된다(전제가 바뀌면 실패).
  5. '오늘' 탭이 이 블록을 **실제로 그린다**(v64가 계산만 하고 안 그렸던 실패 방지).
  6. 파는 것이 사는 것보다 위에 온다.
  7. **자금 흐름·결정 컬럼은 바뀌지 않는다** — 표시 전용이다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.holding_exit as HE  # noqa: E402
import services.recommendation_quality as RQ  # noqa: E402

DATA = ROOT / "data"
DC_SRC = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")
PF_SRC = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")


def _row(anchor=10000.0, close=10000.0, route="CARRY", **ov):
    r = {"종목명": "T", "ROUTE": route, HE.ANCHOR_COL: anchor, "종가": close}
    r.update(ov)
    return r


# ══════════════════════════════════════════════════════════════════
#  1. 기준가는 진입가다 — 재기준 금지 (고장 ①)
# ══════════════════════════════════════════════════════════════════
class TestAnchorIsFixed:
    def test_anchor_is_the_entry_not_today_close(self):
        r = _row(anchor=6770.0, close=6250.0)
        assert HE.anchor_price(r) == 6770.0

    def test_stop_does_not_follow_price_down(self):
        """한국화장품 실측 재현 — 가격이 내려도 손절선은 그대로여야 한다."""
        stops = [HE.stop_price(_row(anchor=6770.0, close=c))
                 for c in (6660.0, 6420.0, 6280.0, 6250.0)]
        assert len(set(stops)) == 1, f"손절선이 가격을 따라 움직였다: {stops}"
        assert stops[0] == pytest.approx(6770.0 * 0.93)

    def test_no_close_fallback(self):
        """기준가가 없으면 종가로 대신하면 안 된다 — 그게 고장 ①이다."""
        assert HE.anchor_price({"종가": 5000.0}) is None
        assert HE.stop_price({"종가": 5000.0}) is None
        assert HE.exit_state({"종가": 5000.0}) == HE.STATE_UNKNOWN

    def test_zero_or_negative_anchor_is_none(self):
        for bad in (0, -1, float("nan"), None, "x"):
            assert HE.anchor_price(_row(anchor=bad)) is None

    def test_source_has_no_close_fallback(self):
        src = (ROOT / "services" / "holding_exit.py").read_text(encoding="utf-8")
        i = src.find("def anchor_price")
        j = src.find("def stop_price")
        body = src[i:j]
        assert HE.CLOSE_COL not in body.split('"""')[-1], \
            "anchor_price가 종가를 폴백으로 쓰고 있다 — 손절선이 가격을 따라간다"


# ══════════════════════════════════════════════════════════════════
#  2. 상태 판정
# ══════════════════════════════════════════════════════════════════
class TestExitState:
    @pytest.mark.parametrize("ret,expected", [
        (-30.0, HE.STATE_STOP), (-7.0, HE.STATE_STOP), (-6.9, HE.STATE_HOLD),
        (0.0, HE.STATE_HOLD), (2.9, HE.STATE_HOLD), (3.0, HE.STATE_FAST),
        (4.9, HE.STATE_FAST), (5.0, HE.STATE_BE), (9.9, HE.STATE_BE),
        (10.0, HE.STATE_TP), (25.0, HE.STATE_TP),
    ])
    def test_thresholds(self, ret, expected):
        assert HE.exit_state(_row(anchor=10000.0, close=10000.0 * (1 + ret / 100))) == expected

    def test_thresholds_come_from_ssot(self):
        from services.pick_reliability import (BE_TRIGGER_PCT, FAST_TP_PCT,
                                               STOP_PCT_DISC, TP_QUICK_PCT)
        assert (HE.STOP_PCT_DISC, HE.FAST_TP_PCT, HE.BE_TRIGGER_PCT, HE.TP_QUICK_PCT) \
            == (STOP_PCT_DISC, FAST_TP_PCT, BE_TRIGGER_PCT, TP_QUICK_PCT)

    def test_stop_is_the_most_urgent(self):
        assert HE.SEVERITY[HE.STATE_STOP] < min(
            HE.SEVERITY[s] for s in HE.SEVERITY if s != HE.STATE_STOP)

    def test_hold_and_unknown_produce_no_action(self):
        assert HE.action_line(_row(close=10000.0)) == ""
        assert HE.action_line({"종가": 100.0}) == ""

    def test_breach_line_says_it_should_have_closed(self):
        line = HE.action_line(_row(anchor=10000.0, close=3330.0, 종목명="이루온",
                                   CARRY_AGE_DAYS=15))
        assert "이루온" in line and "손절선" in line and "종결" in line
        assert "-66.7%" in line


# ══════════════════════════════════════════════════════════════════
#  3. 요약
# ══════════════════════════════════════════════════════════════════
class TestSummary:
    def _df(self):
        return pd.DataFrame([
            _row(anchor=10000.0, close=5000.0, 종목명="A"),
            _row(anchor=10000.0, close=11500.0, 종목명="B"),
            _row(anchor=10000.0, close=10000.0, 종목명="C"),
            _row(anchor=10000.0, close=9000.0, 종목명="D", route="ATTACK"),
        ])

    def test_only_carry_counted(self):
        s = HE.summary(self._df())
        assert s["n"] == 3, "CARRY 아닌 행이 보유로 셈됐다"

    def test_counts_and_actionable(self):
        s = HE.summary(self._df())
        assert s["counts"][HE.STATE_STOP] == 1
        assert s["counts"][HE.STATE_TP] == 1
        assert s["actionable"] == 2

    def test_lines_sorted_by_severity(self):
        s = HE.summary(self._df())
        assert s["lines"][0].startswith("⛔"), "손절 관통이 맨 위가 아니다"

    def test_breach_headline_when_any(self):
        s = HE.summary(self._df())
        assert "손절선을 지났다" in s["line"]

    def test_empty_is_quiet(self):
        for df in (None, pd.DataFrame(), pd.DataFrame({"ROUTE": ["ATTACK"]})):
            s = HE.summary(df)
            assert s["n"] == 0 and s["line"] == "" and s["lines"] == []

    def test_annotate_adds_columns_without_touching_decisions(self):
        df = self._df()
        df["PRODUCTION_BUY"] = [1, 0, 0, 1]
        df["켈리_수량"] = [10, 20, 30, 40]
        out = HE.annotate(df)
        for c in (HE.COL_ANCHOR, HE.COL_STOP, HE.COL_RET, HE.COL_STATE, HE.COL_ACTION):
            assert c in out.columns
        assert list(out["PRODUCTION_BUY"]) == [1, 0, 0, 1]
        assert list(out["켈리_수량"]) == [10, 20, 30, 40]


# ══════════════════════════════════════════════════════════════════
#  4. 화면·배치에 실제로 붙어 있는가 (v64가 놓쳤던 실패)
# ══════════════════════════════════════════════════════════════════
class TestActuallyWired:
    def test_today_tab_renders_holdings(self):
        assert "_render_holdings(summary)" in DC_SRC, \
            "보유 정리 블록을 계산만 하고 그리지 않는다"

    def test_summary_carries_holdings(self):
        assert '"holdings": _HE.summary(df)' in DC_SRC

    def test_selling_comes_before_buying(self):
        i_sell = DC_SRC.find("_render_holdings(summary)")
        i_buy = DC_SRC.find('ui.label("매수 가능 여부")')
        assert 0 < i_sell < i_buy, "파는 것이 사는 것보다 아래에 있다"

    def test_batch_annotates_and_logs(self):
        assert "holding_exit import annotate" in PF_SRC
        assert "[v67] 보유" in PF_SRC

    def test_batch_annotation_is_display_only(self):
        """결정 컬럼을 바꾸는 코드가 붙어 있지 않다."""
        i = PF_SRC.find("[v67] 보유 청산 규율")
        j = PF_SRC.find("[v58] 알파 실전 성적", i)
        block = PF_SRC[i:j]
        for col in ("PRODUCTION_BUY", "TOP_PICK", "켈리_수량", "ALPHA_ENTRY_OK"):
            assert col not in block, f"v67 블록이 결정 컬럼({col})을 건드린다"


# ══════════════════════════════════════════════════════════════════
#  5. 실데이터 — 전제가 바뀌면 실패한다
# ══════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not (DATA / "recommend_20260824.csv").exists(),
                    reason="실데이터 없음")
class TestRealBatch:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(DATA / "recommend_20260824.csv", encoding="utf-8-sig",
                           dtype={"종목코드": str}, low_memory=False)

    def test_breaches_are_detected(self, df):
        s = HE.summary(df)
        assert s["n"] >= 50, f"CARRY {s['n']}종목 — 전제(58종목)와 다르다"
        assert s["counts"].get(HE.STATE_STOP, 0) >= 40, \
            f"손절 관통 {s['counts'].get(HE.STATE_STOP, 0)}건 — 전수 조사(55건)와 다르다"

    def test_losses_are_far_past_the_stop(self, df):
        s = HE.summary(df)
        assert s["median_pct"] is not None and s["median_pct"] < -20, \
            f"보유 중위 {s['median_pct']}% — 조사 시점(-33.4%)과 크게 다르다"
        assert s["worst_pct"] < -50

    def test_decisions_unchanged_by_annotation(self, df):
        before = pd.to_numeric(df["PRODUCTION_BUY"], errors="coerce").fillna(0)
        out = HE.annotate(df)
        after = pd.to_numeric(out["PRODUCTION_BUY"], errors="coerce").fillna(0)
        assert int((before != after).sum()) == 0
        q0 = pd.to_numeric(df["켈리_수량"], errors="coerce").fillna(0).sum()
        q1 = pd.to_numeric(out["켈리_수량"], errors="coerce").fillna(0).sum()
        assert q0 == q1

    def test_guard_still_agrees_after_annotation(self, df):
        """주석을 붙인 뒤에도 품질 가드 판정이 같아야 한다."""
        a = RQ.apply_recommendation_quality_guard(
            df.drop(columns=["PRODUCTION_BUY"], errors="ignore"))
        b = RQ.apply_recommendation_quality_guard(
            HE.annotate(df).drop(columns=["PRODUCTION_BUY"], errors="ignore"))
        assert int((pd.to_numeric(a["PRODUCTION_BUY"], errors="coerce").fillna(0)
                    != pd.to_numeric(b["PRODUCTION_BUY"], errors="coerce").fillna(0)
                    ).sum()) == 0
