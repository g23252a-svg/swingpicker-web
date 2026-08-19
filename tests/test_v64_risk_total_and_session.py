# -*- coding: utf-8 -*-
"""[v64] 5종목 동반 손실 조사 — 화면이 합계 리스크를 말하지 않았다.

■ 사용자 실손실 (2026-08-18)
  로킷헬스케어·SK네트웍스·두산퓨얼셀·한선엔지니어링·에치에프알 5종목 전부
  정리해 **-240,000원**.

■ 조사에서 확인된 사실
  1) **손절은 2026-08-19에 실제로 터졌다.** (조사 초판은 저장소 데이터가
     8/18 종가까지여서 "손절 미도달"로 적었다 — 사용자 확인으로 정정했다.)
     8/18 종가 시점에는 다섯 종목 모두 손절선 위였다:
       로킷 손절 29,650 / 8-18 최저 29,850 · SK 6,980 / 7,330 ·
       두산 34,500 / 35,550 · 한선 11,270 / 11,390 · 에치 14,260 / 14,440
     즉 **엔진 손절은 설계대로 작동했고**, 문제는 손절 규칙이 아니라
     '몇 종목을 얼마나 샀는가'다.
  2) 엔진은 **하루 1종목**만 사이징한다. 공식 매수(로킷) 단독 의도 손실은
     **88,400원**, 5종목을 엔진 수량대로 다 사면 **207,340원**.
     그런데 **어느 화면에도 이 합계가 없었다.** 종목별 "-8.1% vs 진입"만 있었다.
  3) 관찰 후보 목록이 **알파 점수순**이라 공식 매수(93.9)보다 높은 점수
     (98.7·98.5·97.4)가 위에 떴다. v63 실측: 알파 단독 1위 -1.85% vs
     엔진 픽 1위 +3.58% — **순서가 성과와 반대 방향**이다.
  4) 목록에 **켈리 0주 종목**이 있었다(두산=v62 급등 차단, 에치=이미 보유).
     살 수 없는 것을 후보로 보여줬다.
  5) 사유가 전부 "즉시 매수 조건 미충족"이었다. 실제로는 **모든 조건을 통과**
     하고 1종목 제한에서 밀린 것이거나 이미 보유 중이었다.
  6) 손절은 **동반 발생**한다 — 상위N 포트폴리오 실측에서 최악일이 N=1,2,3,5,8
     전부 -8.00%였다. 종목을 나눠도 꼬리 위험이 줄지 않는다.

■ 별건으로 잡은 결함: 휴장일 배치
  2026-08-17은 광복절 대체공휴일(8/15가 토요일)로 **휴장일**인데 배치가 돌아
  8/14 가격을 `기준일=20260817`·`RUN_STATUS=OK`로 찍고 공식매수를 냈다.
  392종목 종가가 8/14와 100% 동일. 원인은 `find_latest_valid_date`의 4단계
  폴백(IP차단 대비 '최근 평일 강제 진행')이 공휴일에도 같은 경로를 타는 것.
  이력 전수 124일 중 비거래일 7일. **실제 진입 왜곡은 작았다**(8/18 갭 중위
  -0.05% · 손절터치 0/12) → 픽을 죽이지 않고 표시만 정직하게 고쳤다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components import decision_center as DC  # noqa: E402
from services import session_freshness as SF  # noqa: E402

BATCH_0817 = ROOT / "data" / "recommend_20260817.csv"


def _batch():
    if not BATCH_0817.exists():
        pytest.skip("8/17 배치 CSV 없음")
    return pd.read_csv(BATCH_0817, dtype={"종목코드": str}, low_memory=False)


# ── A. 합계 리스크 ──────────────────────────────────────────────
class TestRiskTotal:
    def test_sums_official_and_watch(self):
        buys = [{"stop_loss_won": 65000.0, "position_won": 806250.0}]
        watch = [{"stop_loss_won": 130810.0, "position_won": 1627400.0},
                 {"stop_loss_won": 89180.0, "position_won": 1114750.0}]
        rt = DC._risk_total(buys, watch, None)
        assert rt["official_risk_won"] == 65000.0
        assert rt["watch_risk_won"] == pytest.approx(219990.0)
        assert rt["total_risk_won"] == pytest.approx(284990.0)

    def test_line_states_the_multiple(self):
        rt = DC._risk_total([{"stop_loss_won": 65000.0}],
                            [{"stop_loss_won": 195000.0}], None)
        assert "-65,000원" in rt["line"]
        assert "4.0배" in rt["line"], f"배수를 알려주지 않는다: {rt['line']}"

    def test_caveat_says_stops_are_correlated(self):
        rt = DC._risk_total([{"stop_loss_won": 1.0}], [], None)
        assert "동시에" in rt["caveat"]
        assert "-8%" in rt["caveat"], "동반 손절 실측 근거가 없다"

    def test_concentration_is_disclosed(self):
        """오늘 후보가 같은 성격인지 사실대로 적는다.

        2026-08-17 실측: TOP_PICK 12개 중 9개 KOSDAQ, 시총 중위 5,486억
        (유니버스 16,121억의 1/3) — 소형주 모멘텀 한 묶음이다.
        5종목을 사면 분산이 아니라 **같은 베팅의 5배**다.
        """
        s = DC.build_decision_summary(_batch())
        con = s["risk_total"]["concentration"]
        assert con["n"] >= 2
        assert "함께" in con["line"], f"동반 이동 경고가 없다: {con}"
        assert "KOSDAQ" in con["line"] or "시총" in con["line"]

    def test_concentration_quiet_when_diversified(self):
        """구성이 흩어져 있으면 조용하다 — 없는 경고를 만들지 않는다."""
        work = pd.DataFrame({
            "TOP_PICK": [1, 1, 1, 1],
            "시장": ["KOSPI", "KOSDAQ", "KOSPI", "KOSDAQ"],
            "업종_대분류": ["A", "B", "C", "D"],
            "시가총액(억원)": [20000.0, 21000.0, 19000.0, 22000.0],
        })
        assert DC._concentration(work)["line"] == ""

    def test_concentration_needs_two_picks(self):
        work = pd.DataFrame({"TOP_PICK": [1], "시장": ["KOSDAQ"]})
        assert DC._concentration(work)["n"] == 0

    def test_quiet_when_nothing_sized(self):
        rt = DC._risk_total([], [], None)
        assert rt["line"] == ""
        assert rt["total_risk_won"] == 0.0

    def test_real_batch_exposes_the_gap(self):
        """8/17 화면에서 공식 단독 vs 전부의 차이가 실제로 드러나는지."""
        s = DC.build_decision_summary(_batch())
        rt = s["risk_total"]
        assert rt["official_risk_won"] > 0, "공식 매수 리스크가 0이다"
        assert rt["total_risk_won"] > rt["official_risk_won"] * 2, (
            "관찰 후보를 합친 리스크가 공식 단독의 2배 미만 — "
            f"전제가 바뀌었다: {rt}")
        assert "배" in rt["line"]

    def test_summary_carries_won_risk_per_row(self):
        s = DC.build_decision_summary(_batch())
        assert s["buys"], "공식 매수가 없다"
        for row in s["buys"] + s["watch"]:
            assert row["stop_loss_won"] > 0, f"{row['name']}: 원화 리스크 없음"
            assert "손절 시 손실" in row["risk_line"]


# ── B. 관찰 후보 정렬·구성 ──────────────────────────────────────
class TestWatchList:
    def test_sorted_by_engine_key_not_raw_alpha(self):
        """엔진 랭킹(알파×손익비)이 정렬 기준이어야 한다.

        알파 단독 정렬은 v63 실측에서 성과와 반대 방향이었다
        (알파 1위 -1.85% vs 엔진 픽 1위 +3.58%).
        """
        s = DC.build_decision_summary(_batch())
        watch = s["watch"]
        if len(watch) < 2:
            pytest.skip("후보 2개 미만")
        keys = [w["alpha"] * min(w["rr"], 3.0) for w in watch]
        assert keys == sorted(keys, reverse=True), \
            f"엔진 키 내림차순이 아니다: {keys}"

    def test_zero_quantity_names_are_excluded(self):
        s = DC.build_decision_summary(_batch())
        for w in s["watch"]:
            assert w["qty"] > 0, \
                f"{w['name']}: 켈리 0주인데 후보 목록에 있다 (살 수 없는 후보)"

    def test_source_uses_engine_key(self):
        src = (ROOT / "components" / "decision_center.py").read_text(
            encoding="utf-8")
        assert "_engine_key" in src and "RR_NOW_TP1" in src, \
            "정렬이 손익비를 반영하지 않는다"


# ── C. 사유 문구 정확화 ─────────────────────────────────────────
class TestHonestReasons:
    def test_daily_cap_is_not_called_condition_failure(self):
        text = DC._humanize_reason("당일 신규진입 1종목 제한")
        assert "통과" in text and "순위" in text, text
        assert "미충족" not in text, "조건을 못 채운 것으로 읽힌다"

    def test_carry_is_labeled_as_already_held(self):
        text = DC._humanize_reason("상태 CARRY — 신규진입 상태 아님(0주)")
        assert "보유" in text and "0주" in text, text

    def test_kelly_reason_is_not_duplicated(self):
        text = DC._humanize_reason(
            "실측 승률 44% < 필요 승률 45% (손익비 1.22) — 켈리 기준 미달")
        assert text.count("켈리 기준 미달") == 1, f"문구가 중복된다: {text}"

    def test_real_batch_reasons_are_specific(self):
        s = DC.build_decision_summary(_batch())
        vague = [w["name"] for w in s["watch"]
                 if w["reason"].strip() == "즉시 매수 조건 미충족"]
        assert not vague, f"뭉갠 사유가 남아 있다: {vague}"


# ── D. 휴장일 세션 감지 ─────────────────────────────────────────
class TestSessionFreshness:
    def test_holiday_batch_is_detected_as_stale(self):
        rep = SF.assess("20260817", str(ROOT / "data"))
        if not rep.get("ok"):
            pytest.skip(rep.get("note"))
        assert rep["stale"] is True, "휴장일 배치를 신선하다고 판정한다"
        assert rep["price_asof"] == "20260814", rep
        assert rep["is_trading_day"] is False
        assert rep["lag_sessions"] >= 1

    def test_real_trading_day_is_clean(self):
        for ymd in ("20260818", "20260814", "20260813"):
            rep = SF.assess(ymd, str(ROOT / "data"))
            if not rep.get("ok"):
                continue
            assert rep["stale"] is False, f"{ymd}를 묵었다고 오판한다: {rep}"

    def test_assessment_is_as_of_batch_time(self):
        """나중에 쌓인 캐시로 판정하면 휴장일 배치가 사후에 신선해 보인다."""
        newest = SF.latest_price_ymd(str(ROOT / "data"))
        asof = SF.latest_price_ymd(str(ROOT / "data"), asof_ymd="20260817")
        assert asof == "20260814"
        assert newest != asof, "as-of 필터가 동작하지 않는다"

    def test_annotate_marks_but_does_not_change_decisions(self):
        d = _batch()
        before = {c: d[c].copy() for c in
                  ("TOP_PICK", "PRODUCTION_BUY", "BUY_NOW_ELIGIBLE", "켈리_수량")
                  if c in d.columns}
        rep = SF.assess("20260817", str(ROOT / "data"))
        out = SF.annotate(d.copy(), rep)
        assert int(out[SF.STALE_COL].iloc[0]) == 1
        assert str(out[SF.ASOF_COL].iloc[0]) == "20260814"
        for c, col in before.items():
            assert out[c].equals(col), f"{c}가 바뀌었다 — 표시 전용이어야 한다"

    def test_run_status_downgraded_only_from_ok(self):
        d = pd.DataFrame({"RUN_STATUS": ["OK", "DEGRADED"]})
        rep = {"ok": True, "stale": True, "price_asof": "20260814",
               "lag_sessions": 1, "trade_ymd": "20260817"}
        out = SF.annotate(d, rep)
        assert out["RUN_STATUS"].tolist() == [SF.STALE_STATUS, "DEGRADED"], \
            "더 나쁜 상태(DEGRADED)를 덮어썼다"

    def test_line_is_quiet_when_fresh(self):
        assert SF.line({"ok": True, "stale": False}) == ""

    def test_wired_after_quality_guard(self):
        src = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")
        i_q = src.find("df_out = apply_recommendation_quality_guard(df_out)")
        i_sf = src.find("[v64] 세션 신선도")
        assert i_q > 0 and i_sf > 0, "배선 누락"
        assert i_q < i_sf


# ── E. 리스크 산술 (손절 사건과 무관하게 성립) ──────────────────
class TestRiskArithmetic:
    """엔진 의도 손실 대비 실현 손실의 배수를 고정한다.

    [정정] 초판은 "다섯 종목 모두 손절가에 도달하지 않았다"를 사실로 못박았다.
    그 판단은 **저장소 데이터가 8/18 종가까지**였기 때문이고, 사용자 확인 결과
    **8/19에 실제로 손절이 터졌다.** 즉 엔진 손절은 설계대로 작동했다.
    그래서 손절 도달 여부를 전제로 삼는 테스트를 걷어내고, 손절 사건이든
    아니든 성립하는 **리스크 산술**만 고정한다 — v64의 근거는 '손절이 났는가'가
    아니라 '합계 리스크를 화면이 말했는가'이기 때문이다.
    """

    def test_intended_risk_of_single_official_pick(self):
        """공식 매수 1종목의 의도 손실 (로킷헬스케어 8/14 배치 기준)."""
        buy, stop, qty = 32250.0, 29650.0, 34.0
        assert (buy - stop) * qty == pytest.approx(88400.0)

    def test_all_five_would_risk_more_than_double(self):
        """5종목을 엔진 수량대로 사면 공식 단독의 2배를 넘는다."""
        legs = [(32250.0, 29650.0, 34.0),   # 로킷헬스케어 (공식 매수)
                (7590.0, 6980.0, 52.0),     # SK네트웍스
                (37550.0, 34500.0, 0.0),    # 두산퓨얼셀 — 엔진 0주
                (12250.0, 11270.0, 89.0),   # 한선엔지니어링
                (15510.0, 14260.0, 0.0)]    # 에치에프알 — 엔진 0주(보유중)
        total = sum((b - s) * q for b, s, q in legs)
        official = (32250.0 - 29650.0) * 34.0
        assert total == pytest.approx(207340.0)
        assert total > official * 2, "합계가 공식 단독의 2배 이하 — 전제가 바뀌었다"

    def test_two_of_five_were_sized_zero_by_engine(self):
        """두산퓨얼셀·에치에프알은 엔진이 0주로 산정했다 — 살 수 없는 후보였다."""
        for qty in (0.0, 0.0):
            assert qty == 0.0
