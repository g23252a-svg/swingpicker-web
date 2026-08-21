# -*- coding: utf-8 -*-
"""[v66] 갭이 손절폭을 좁힌다 + 화면이 하지 않는 일을 한다고 적고 있었다.

■ 사용자 실손실 (2026-08-19 ~ 08-21)
  8/18 배치를 보고 산 종목들이 이틀에 걸쳐 연달아 손절됐다(아모텍·우리로 등).
  8/19 코스피는 **-5.80%** 폭락했다.

■ 결함 ① 갭다운이 손절선을 코앞으로 당긴다
  손절가는 **추천매수가(전일 종가 기준)** 에 고정되고 다음 날 실제 체결가로
  다시 잡지 않는다. 그래서 시가가 갭다운하면 계획 손절폭이 산술적으로 줄어든다.
  8/18 배치 26건 실측: 갭 중위 **-3.84%** → 실효 손절폭 중위 **-4.18%**
  (계획 -8.03%). 여유 4% 미만 9건, 시가가 이미 손절가 이하 1건.
    · 세미파이브(공식 매수) 16,100 → 시가 15,410, 손절 15,370
      → 실효 **-0.26%** · 다음 날 즉시 손절
    · 아모텍(근접 1위) 12,330 → 시가 11,960, 손절 11,340 → 실효 -5.2%
      8/19 저가 11,410으로 손절선 위 0.6%까지 붙었다
  전수(게이트 통과 상위10 · 204건/21일): 갭≤-3% 구간 **손절률 90%**(n=10),
  갭 -1~+1% 구간 25%(n=115). 실효 손절폭 -4~0% 구간 손절률 **47%** vs
  -8% 이하 30%.

■ 결함 ② 화면이 "신규 진입 차단"이라 적고 28종목을 사이징했다
  8/18·8/19 배치: MARKET_REGIME=DOWN · REGIME_ALLOW_ENTRY=0 ·
  REGIME_REASON="…신규 진입 차단(실측: 이 구간 진입은 손실 우세)" ·
  REGIME_SIZE_MULT=0.3 → **그런데 공식 매수 1건 + 사이징 후보 28건**.
  · REGIME_ALLOW_ENTRY는 쓰기만 하고 읽는 코드가 0건이다.
  · regime_ok 거부권은 레거시 분기에만 있고 알파 경로에는 v32가 뺐다.
  · REGIME_SIZE_MULT는 RECOMMENDED_WEIGHT_PCT(표시용)에만 곱해진다.
    실측: 8/18(배수 0.3) 공식픽 투입 95.0만원 > 8/20(배수 1.0) 81.0만원.
  · 문구의 근거도 틀렸다 — DOWN 6일 게이트 상위5 **+4.15%**(초과 +2.29%p,
    p=0.37)로 "손실 우세"와 부호가 반대다.

■ 결함 ③ v64가 계산만 하고 화면에 붙이지 못했다
  risk_line · risk_total · concentration을 전부 만들어 summary에 담았는데
  **렌더 코드가 없었다.** v64 테스트가 데이터 계층 함수만 검사해서 놓쳤다 —
  v58.1이 고쳤던 것과 같은 실패 방식이다.

■ 이 파일이 고정하는 것
  1. 실효 손절폭은 **실제 체결가** 기준으로 계산된다(계획값과 다르면 다르다).
  2. 갭 경고는 **선별적**이다 — p05(-4.26%) 기준이면 22종목 중 20종목이 걸려
     경고가 소음이 된다. 흔한 갭(p25)에도 얇아지는 종목만 경고한다.
  3. 8/18 세미파이브가 경고 대상이어야 한다(그 종목이 실제로 손절됐다).
  4. 레짐 문구가 집행되지 않는 차단을 '차단'이라 적지 않는다.
  5. VETO_ENFORCED 상수와 실제 배선이 일치한다 — 어긋나면 실패(죽은 게이트 재발).
  6. 세 렌더러가 화면에 실제로 붙어 있다(계산만 하고 안 그리면 실패).
  7. **자금 흐름은 바뀌지 않는다** — 결정 컬럼과 수량은 v66 전후가 같다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import market_regime as MR  # noqa: E402
import services.position_risk as PR  # noqa: E402
import services.recommendation_quality as RQ  # noqa: E402

DATA = ROOT / "data"
DC_SRC = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")

# 실제 8/18 배치 값 — 다음 날 손절된 공식 매수
SEMI = {"종목명": "세미파이브", "추천매수가": 16100.0, "손절가": 15370.0,
        "켈리_수량": 59.0}
SEMI_OPEN = 15410.0          # 8/19 실제 시가
AMO = {"종목명": "아모텍", "추천매수가": 12330.0, "손절가": 11340.0,
       "켈리_수량": 124.0}
AMO_OPEN = 11960.0
NORMAL = {"종목명": "이노스페이스", "추천매수가": 6280.0, "손절가": 5770.0,
          "켈리_수량": 129.0}


# ══════════════════════════════════════════════════════════════════
#  1. 실효 손절폭 계산
# ══════════════════════════════════════════════════════════════════
class TestEffectiveStop:
    def test_planned_matches_batch_intent(self):
        assert PR.planned_stop_pct(SEMI) == pytest.approx(-4.53, abs=0.01)
        assert PR.planned_stop_pct(AMO) == pytest.approx(-8.03, abs=0.01)

    def test_effective_uses_actual_fill(self):
        """세미파이브: 계획 -4.53%였는데 실제 체결 기준 -0.26%였다."""
        assert PR.effective_stop_pct(SEMI, SEMI_OPEN) == pytest.approx(-0.26, abs=0.01)
        assert PR.effective_stop_pct(AMO, AMO_OPEN) == pytest.approx(-5.18, abs=0.01)

    def test_effective_differs_from_planned(self):
        """둘이 같다고 가정하면 이번 사건을 설명할 수 없다."""
        assert PR.effective_stop_pct(SEMI, SEMI_OPEN) > PR.planned_stop_pct(SEMI)

    def test_gap_up_widens_the_stop(self):
        """갭업이면 반대로 넓어진다 — 부호가 한 방향으로만 움직이면 버그다."""
        assert PR.stop_pct_at_gap(NORMAL, +3.0) < PR.planned_stop_pct(NORMAL)

    def test_no_fill_price_is_none_not_zero(self):
        assert PR.effective_stop_pct(SEMI, None) is None
        assert PR.effective_stop_pct(SEMI, 0) is None
        assert PR.effective_stop_pct(SEMI, float("nan")) is None

    def test_missing_stop_is_none(self):
        assert PR.effective_stop_pct({"추천매수가": 1000.0}, 1000.0) is None

    def test_scenarios_come_from_measured_distribution(self):
        """시나리오 갭은 임의값이 아니라 실측 분위수여야 한다."""
        assert PR.GAP_P25_PCT == pytest.approx(-1.28, abs=0.01)
        assert PR.GAP_P05_PCT == pytest.approx(-4.26, abs=0.01)
        assert PR.GAP_P05_PCT < PR.GAP_P25_PCT < 0

    def test_sensitivity_is_monotone(self):
        s = PR.gap_sensitivity(NORMAL)
        gaps = [x["gap_pct"] for x in s["scenarios"]]
        effs = [x["effective_pct"] for x in s["scenarios"]]
        assert gaps == sorted(gaps, reverse=True), "시나리오가 갭 내림차순이 아니다"
        # 갭이 깊을수록 실효 손절폭은 0에 가까워진다(= 커진다) — 얇아진다는 뜻
        assert effs == sorted(effs), f"갭이 깊은데 손절폭이 얇아지지 않았다: {effs}"
        assert effs[-1] > effs[0]


# ══════════════════════════════════════════════════════════════════
#  2. 경고는 선별적이어야 한다 (흔한 경고는 의미를 잃는다)
# ══════════════════════════════════════════════════════════════════
class TestWarningIsSelective:
    def test_semifive_is_warned(self):
        """실제로 다음 날 손절된 종목이 경고 대상이 아니면 문턱이 틀렸다."""
        assert PR.gap_risk_line(SEMI), "세미파이브가 경고되지 않았다"

    def test_normal_stock_is_quiet(self):
        assert PR.gap_risk_line(NORMAL) == ""

    def test_info_line_exists_for_every_stock(self):
        """경고가 없어도 시나리오 숫자는 사실로 보여준다."""
        assert PR.gap_table_line(NORMAL)
        assert "갭" in PR.gap_table_line(NORMAL)

    def test_thin_as_planned_has_its_own_message(self):
        row = {"추천매수가": 10000.0, "손절가": 9800.0, "켈리_수량": 10}
        line = PR.gap_risk_line(row)
        assert "계획 자체가" in line

    @pytest.mark.skipif(not (DATA / "recommend_20260818.csv").exists(),
                        reason="실데이터 없음")
    def test_real_batch_warning_rate_is_low(self):
        """실배치에서 경고가 흔하면 소음이다. p05 기준이면 20/22가 걸린다."""
        for ymd, cap in (("20260818", 6), ("20260820", 6)):
            p = DATA / f"recommend_{ymd}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p, encoding="utf-8-sig", dtype={"종목코드": str},
                            low_memory=False)
            q = pd.to_numeric(d["켈리_수량"], errors="coerce").fillna(0)
            sel = d[q > 0]
            warned = sum(1 for _, r in sel.iterrows() if PR.gap_risk_line(r))
            assert 0 < warned <= cap, f"{ymd}: 경고 {warned}/{len(sel)}건 — 선별적이지 않다"

    @pytest.mark.skipif(not (DATA / "recommend_20260818.csv").exists(),
                        reason="실데이터 없음")
    def test_semifive_warned_in_the_real_batch(self):
        d = pd.read_csv(DATA / "recommend_20260818.csv", encoding="utf-8-sig",
                        dtype={"종목코드": str}, low_memory=False)
        row = d[d["종목명"].astype(str) == "세미파이브"]
        if not len(row):
            pytest.skip("세미파이브가 이 배치에 없다")
        assert PR.gap_risk_line(row.iloc[0]), \
            "실배치에서 손절된 공식 매수가 경고되지 않았다"


# ══════════════════════════════════════════════════════════════════
#  3. 레짐 문구가 사실이어야 한다
# ══════════════════════════════════════════════════════════════════
class TestRegimeTextIsTrue:
    def _down(self):
        return MR.compute_market_regime("20260818", 23.7, data_dir=str(DATA))

    def test_down_does_not_claim_a_block_it_does_not_enforce(self):
        r = self._down()
        assert r["regime"] == "DOWN"
        if not MR.VETO_ENFORCED:
            assert "차단되지 않는다" in r["reason"], r["reason"]
            assert "신규 진입 차단." not in r["reason"], \
                "집행하지 않는 차단을 차단이라 적고 있다"

    def test_down_does_not_cite_a_refuted_measurement(self):
        """'이 구간 진입은 손실 우세'는 현재 표본에서 부호가 반대다."""
        r = self._down()
        assert "손실 우세" not in r["reason"] or "확인되지 않았다" in r["reason"]

    def test_up_does_not_promise_an_unverifiable_return(self):
        r = MR.compute_market_regime("20260820", 60.0, data_dir=str(DATA))
        assert r["regime"] == "UP"
        assert "+5.1%" not in r["reason"], \
            "현재 표본으로 재현 불가한 수익 수치를 근거로 인용한다"

    def test_every_regime_states_what_the_multiplier_touches(self):
        for br, exp in ((23.7, "DOWN"), (45.0, "NEUTRAL"), (60.0, "UP")):
            r = MR.compute_market_regime("20260820", br, data_dir=str(DATA))
            assert r["regime"] == exp
            assert MR.MULT_APPLIES_TO in r["reason"] or "차단되지 않는다" in r["reason"]

    def test_veto_enforced_is_reported(self):
        assert self._down()["veto_enforced"] is MR.VETO_ENFORCED

    def test_columns_carry_enforcement_scope(self):
        info = self._down()
        out = MR.inject_regime_columns(pd.DataFrame({"종목코드": ["000001"]}), info)
        assert int(out["REGIME_VETO_ENFORCED"].iloc[0]) == int(MR.VETO_ENFORCED)
        assert out["REGIME_MULT_APPLIES_TO"].iloc[0] == MR.MULT_APPLIES_TO


# ══════════════════════════════════════════════════════════════════
#  4. 죽은 게이트 재발 방지 — 선언과 배선이 일치하는가
# ══════════════════════════════════════════════════════════════════
class TestDeadGateGuard:
    def _row(self, **ov):
        r = {"종목코드": "024060", "종목명": "T", "TOP_PICK": 1,
             "BUY_NOW_ELIGIBLE": 1, "ALPHA_GATE_ACTIVE": 1, "ALPHA_ENTRY_OK": 1,
             "ALPHA_SCORE": 100.0, "ALPHA_VALIDATED": 1, "RR_NOW_TP1": 5.0,
             "POC_GAP": 5.0, "MARKET_BREADTH": 23.7, "MACRO_RISK": "NORMAL",
             "DATA_FRESHNESS_OK": 1, "MARKET_REGIME": "DOWN", "ROUTE": "WAIT",
             "DISPLAY_SCORE": 60.0, "BUY_NOW_SCORE": 60.0}
        r.update(ov)
        return pd.DataFrame([r])

    def test_constant_matches_actual_wiring(self):
        """VETO_ENFORCED=False라면 DOWN에서 공식 매수가 실제로 나와야 한다.
        (반대로 True로 바꿔놓고 배선을 안 하면 이 검사가 실패한다.)"""
        out = RQ.apply_recommendation_quality_guard(self._row())
        produced = int(out.iloc[0]["PRODUCTION_BUY"]) == 1
        assert produced is (not MR.VETO_ENFORCED), (
            f"VETO_ENFORCED={MR.VETO_ENFORCED}인데 DOWN 레짐에서 "
            f"공식매수={produced} — 선언과 배선이 어긋난다")

    def test_up_regime_unaffected(self):
        out = RQ.apply_recommendation_quality_guard(
            self._row(MARKET_REGIME="UP", MARKET_BREADTH=60.0))
        assert int(out.iloc[0]["PRODUCTION_BUY"]) == 1

    def test_allow_entry_column_is_not_silently_trusted(self):
        """REGIME_ALLOW_ENTRY=0을 넣어도 판정이 바뀌지 않는다 — 그것이 현재
        사실이고, 이 테스트는 그 사실을 문서화한다. 나중에 집행을 붙이면
        VETO_ENFORCED와 함께 바꿔야 한다."""
        a = RQ.apply_recommendation_quality_guard(self._row(REGIME_ALLOW_ENTRY=0))
        b = RQ.apply_recommendation_quality_guard(self._row(REGIME_ALLOW_ENTRY=1))
        assert int(a.iloc[0]["PRODUCTION_BUY"]) == int(b.iloc[0]["PRODUCTION_BUY"])

    def test_stale_comment_is_corrected(self):
        """옛 주석은 'DOWN=0 (진입 자체 차단됨)'이라 단정했다. 둘 다 사실이
        아니다(배수는 0.3이고 진입은 막히지 않는다). 인용한 정정문은 남기되
        **현행 설명으로 서술된 문장**은 사라져야 한다."""
        src = (ROOT / "services" / "recommendation_quality.py").read_text(
            encoding="utf-8")
        assert "# [v28] 레짐 사이징: UP=100%, NEUTRAL/UNKNOWN=50%, DOWN=0" not in src, \
            "DOWN에서 진입한다는 사실과 어긋나는 주석이 현행 설명으로 남아 있다"
        assert "[v66 정정]" in src, "무엇이 왜 틀렸는지 기록이 없다"
        assert "켈리_수량에는 걸리지 않는다" in src


# ══════════════════════════════════════════════════════════════════
#  5. 화면에 실제로 붙어 있는가 (v64가 놓친 것)
# ══════════════════════════════════════════════════════════════════
class TestActuallyRendered:
    def test_per_stock_risk_is_rendered(self):
        assert "_render_risk_lines(stock)" in DC_SRC, \
            "종목별 리스크 줄을 계산만 하고 그리지 않는다"
        assert DC_SRC.count("_render_risk_lines(stock)") >= 2, \
            "공식 매수·관찰 후보 양쪽에 붙어야 한다"

    def test_total_risk_block_is_rendered(self):
        assert "_render_risk_total(summary)" in DC_SRC, \
            "합계 리스크를 계산만 하고 그리지 않는다 (v64가 이랬다)"

    def test_renderers_read_the_payload_keys(self):
        for key in ("risk_line", "gap_warn", "gap_table"):
            assert f'stock.get("{key}")' in DC_SRC, f"{key}를 읽지 않는다"
        assert 'rt.get("concentration")' in DC_SRC

    def test_payload_carries_gap_fields(self):
        assert '"gap_table": _PR.gap_table_line(row)' in DC_SRC
        assert '"gap_warn": _PR.gap_risk_line(row)' in DC_SRC


# ══════════════════════════════════════════════════════════════════
#  6. 자금 흐름 불변 — v66은 표시만 바꾼다
# ══════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not (DATA / "recommend_20260820.csv").exists(),
                    reason="실데이터 없음")
class TestNoMoneyFlowChange:
    def test_decisions_are_unchanged(self):
        d = pd.read_csv(DATA / "recommend_20260820.csv", encoding="utf-8-sig",
                        dtype={"종목코드": str}, low_memory=False)
        before = pd.to_numeric(d["PRODUCTION_BUY"], errors="coerce").fillna(0)
        out = RQ.apply_recommendation_quality_guard(
            d.drop(columns=["PRODUCTION_BUY"], errors="ignore"))
        after = pd.to_numeric(out["PRODUCTION_BUY"], errors="coerce").fillna(0)
        assert int((before != after).sum()) == 0, "v66이 결정을 바꿨다"

    def test_quantities_untouched_by_display_helpers(self):
        d = pd.read_csv(DATA / "recommend_20260820.csv", encoding="utf-8-sig",
                        dtype={"종목코드": str}, low_memory=False)
        q0 = pd.to_numeric(d["켈리_수량"], errors="coerce").fillna(0).sum()
        for _, r in d.head(50).iterrows():
            PR.gap_risk_line(r); PR.gap_table_line(r); PR.risk_line(r)
        q1 = pd.to_numeric(d["켈리_수량"], errors="coerce").fillna(0).sum()
        assert q0 == q1
