# -*- coding: utf-8 -*-
"""[v61] 리스크 표기가 사실과 달랐던 것 — 회귀 봉쇄.

■ 사용자 실손실 (2026-08-13)
  씨어스(458870)로 -230,000원. 이 종목은 2026-08-11 배치의 프로덕션 픽
  1순위였다(LDY_RANK=1 · ALPHA_SCORE=97.8 · RR 4.63 · 켈리_수량 29주).
  **엔진이 의도한 최대 손실은 59,450원** — 실현 손실은 그 3.9배였다.

■ 조사에서 확인된 것 / 확인되지 않은 것
  ✅ 화면은 `MAX_LOSS 5.0%`를 "손절 발동 시 잃을 자본 비중"이라는 툴팁과 함께
     보여줬다. 그런데 그 값은 `trade_plan`의 **시총 기반 캡**이고, 실제 손절폭은
     8.12%였다(1.62배). 120개 배치 51,049행 중 **88.0%에서 실제 손절폭이 캡을
     초과**했고 120일 전부에서 발생했다. TOP_PICK은 107/109행(98.2%).
  ✅ "손절 시 몇 원을 잃는가"는 UI 어디에도 없었다.
  ❌ 씨어스 유형(장기 하락·POC 아래)을 막는 브레이크 9개를 검정했으나 **전부
     유의하지 않았다**(|t|<2.0, p>0.10). 근거 없는 필터는 넣지 않았다.
  ❌ 가드 경고(GUARD_FORCE_EXIT_ALERT 등)를 화면에 띄우는 것도 기각했다 —
     유니버스 전체에서 **역예측**이었다(GUARD_KELLY_MULT=0 종목이 +1.90%p 더
     좋았다, t=2.11). v45의 판단이 더 큰 표본에서 재현됐다.

  → v61은 **예측을 개선하지 않는다. 표기를 사실에 맞춘다.**
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import position_risk as PR  # noqa: E402

REAL_CSV = ROOT / "data" / "recommend_20260811.csv"
CEERS = "458870"


def _row(**ov):
    base = {"추천매수가": 25250.0, "손절가": 23200.0, "켈리_수량": 29.0,
            "종가": 25250.0, "STOP_PCT": 8.12, "MAX_LOSS_PCT": 5.0}
    base.update(ov)
    return base


# ── A. 실제 손절 손실을 가격에서 직접 계산한다 ──────────────────
class TestActualStopLoss:
    def test_pct_from_prices_not_from_cap(self):
        r = _row()
        assert PR.stop_loss_pct(r) == pytest.approx(8.1188, abs=1e-3)
        assert PR.stop_loss_pct(r) != r["MAX_LOSS_PCT"]

    def test_won_amount_is_the_missing_number(self):
        assert PR.stop_loss_won(_row()) == pytest.approx(59450.0)

    def test_position_won(self):
        assert PR.position_won(_row()) == pytest.approx(732250.0)

    def test_quantity_falls_back_to_추천수량(self):
        r = _row(켈리_수량=0.0, 추천수량=10.0)
        assert PR.quantity(r) == 10.0
        assert PR.stop_loss_won(r) == pytest.approx(20500.0)

    def test_entry_falls_back_to_close(self):
        r = _row(); r.pop("추천매수가")
        assert PR.entry_price(r) == 25250.0

    def test_no_quantity_gives_pct_only(self):
        r = _row(켈리_수량=0.0)
        assert PR.stop_loss_pct(r) is not None
        assert PR.stop_loss_won(r) is None, "수량이 없으면 금액을 만들어내지 않는다"

    def test_bad_stop_returns_none_not_zero(self):
        for bad in (0.0, None, 30000.0):     # 0/결측/진입가 이상
            assert PR.stop_loss_pct(_row(손절가=bad)) is None
            assert PR.stop_loss_won(_row(손절가=bad)) is None

    def test_risk_line_is_quiet_when_uncomputable(self):
        assert PR.risk_line({"손절가": 1.0}) == ""


# ── B. 캡을 손실이라고 부르지 않는다 ───────────────────────────
class TestCapIsNotMaxLoss:
    def test_cap_does_not_bind_in_the_ceers_case(self):
        assert PR.cap_binds(_row()) is False

    def test_cap_binds_when_it_actually_binds(self):
        assert PR.cap_binds(_row(MAX_LOSS_PCT=12.0)) is True

    def test_cap_binds_none_when_undecidable(self):
        assert PR.cap_binds(_row(MAX_LOSS_PCT=0.0)) is None

    def test_consistency_report_flags_understated_risk(self):
        rep = PR.risk_consistency(_row())
        assert rep["ok"] is False
        assert any("과소 표기" in p for p in rep["problems"])

    def test_consistency_clean_when_cap_binds(self):
        rep = PR.risk_consistency(_row(MAX_LOSS_PCT=12.0))
        assert rep["ok"] is True, rep["problems"]

    def test_consistency_flags_stop_pct_mismatch(self):
        """STOP_PCT 컬럼과 가격 기반 계산이 어긋나면 잡는다."""
        rep = PR.risk_consistency(_row(STOP_PCT=3.0))
        assert any("STOP_PCT" in p for p in rep["problems"])

    def test_labels_do_not_call_the_cap_a_loss(self):
        assert "손실" in PR.LOSS_LABEL
        assert "캡" in PR.CAP_LABEL
        assert "손실이 아니다" in PR.CAP_TOOLTIP or "아니다" in PR.CAP_TOOLTIP


# ── C. 화면 배선 — 오표기가 되살아나면 실패한다 ─────────────────
class TestUIWiring:
    DETAIL = ROOT / "components" / "stock_detail_v2.py"
    CARDS = ROOT / "components" / "action_cards.py"

    def test_old_misleading_tooltip_is_gone(self):
        """'포지션당 최대 손실율. 손절 발동 시 잃을 자본 비중' + 시총 캡 조합 금지."""
        src = self.DETAIL.read_text(encoding="utf-8")
        assert "포지션당 최대 손실율" not in src, \
            "시총 캡을 '최대 손실율'로 설명하는 툴팁이 되살아났다"

    def test_detail_shows_won_denominated_loss(self):
        src = self.DETAIL.read_text(encoding="utf-8")
        assert "stop_loss_won" in src, "손절 시 손실 금액이 상세 화면에 없다"
        assert "_PR.LOSS_LABEL" in src and "_PR.CAP_LABEL" in src, \
            "라벨을 SSOT에서 가져오지 않는다"

    def test_cap_row_marks_when_not_binding(self):
        src = self.DETAIL.read_text(encoding="utf-8")
        assert "cap_binds" in src and "미구속" in src, \
            "캡이 구속하지 않는 사실을 화면에 표시하지 않는다"

    def test_position_pct_tooltip_says_split_not_weight(self):
        """POSITION_PCT는 진입 분할 계획이다 — 포트폴리오 비중이 아니다."""
        src = self.DETAIL.read_text(encoding="utf-8")
        assert "진입 분할" in src
        assert "100% = 풀포지션, 0% = 신규매수 부적합" not in src

    def test_action_card_shows_risk_line(self):
        src = self.CARDS.read_text(encoding="utf-8")
        assert "_risk_html(row)" in src, "카드 본문에 리스크 줄이 배선되지 않았다"
        assert "position_risk" in src

    def test_no_module_computes_stop_loss_pct_on_its_own(self):
        """손절 손실률 계산이 SSOT 밖에서 중복되면 다시 갈라진다."""
        for path in (self.DETAIL, self.CARDS):
            src = path.read_text(encoding="utf-8")
            # MAX_LOSS_PCT를 손실 표기에 직접 쓰는 패턴 금지
            for m in re.finditer(r"MAX_LOSS_PCT", src):
                head = src[max(0, m.start() - 200):m.start()]
                assert "손절 시" not in head, \
                    f"{path.name}: MAX_LOSS_PCT를 손절 손실로 쓰고 있다"


# ── D. 전제 고정 — 이 패치의 근거를 실데이터로 못 박는다 ─────────
class TestMeasuredPremise:
    def _real(self):
        if not REAL_CSV.exists():
            pytest.skip("실제 배치 CSV 없음")
        return pd.read_csv(REAL_CSV, dtype={"종목코드": str}, low_memory=False)

    def test_cap_is_exceeded_for_most_rows(self):
        """캡이 실제로 거의 구속하지 않는다는 전제(88%)를 고정."""
        d = self._real()
        if not {"STOP_PCT", "MAX_LOSS_PCT"} <= set(d.columns):
            pytest.skip("컬럼 없음")
        sp = pd.to_numeric(d["STOP_PCT"], errors="coerce")
        ml = pd.to_numeric(d["MAX_LOSS_PCT"], errors="coerce")
        v = sp.notna() & ml.notna() & (ml > 0)
        assert v.sum() > 50
        rate = float((sp[v] > ml[v] + 1e-9).mean())
        assert rate > 0.5, (
            f"캡 초과율이 {rate:.1%}로 낮아졌다 — 캡이 실제로 구속하기 시작했다면 "
            "표기 방식을 다시 검토해야 한다(이 테스트가 그 신호다)")

    def test_ceers_row_reproduces_the_understatement(self):
        d = self._real()
        d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
        s = d[d["종목코드"] == CEERS]
        if s.empty:
            pytest.skip("씨어스 행 없음")
        r = s.iloc[0].to_dict()
        assert int(r.get("TOP_PICK", 0)) == 1, "재현 전제(프로덕션 픽)가 깨졌다"
        actual = PR.stop_loss_pct(r)
        cap = float(r["MAX_LOSS_PCT"])
        assert actual is not None and actual > cap, \
            "실제 손절폭이 캡보다 크다는 재현 전제가 깨졌다"
        assert PR.stop_loss_won(r) == pytest.approx(59450.0, abs=1.0)

    def test_intended_risk_is_far_below_reported_loss(self):
        """엔진 의도 손실(약 5.9만원) 대비 실손실(23만원)의 배수를 기록으로 남긴다.

        이 격차가 v61의 존재 이유다 — 신호가 아니라 **포지션 크기 기준**이
        화면에 없었다.
        """
        d = self._real()
        d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
        s = d[d["종목코드"] == CEERS]
        if s.empty:
            pytest.skip("씨어스 행 없음")
        intended = PR.stop_loss_won(s.iloc[0].to_dict())
        assert intended is not None
        assert 230000 / intended > 3.0, "격차 전제가 바뀌었다"


# ── E. 기각한 후보를 기록으로 남긴다 (다시 제안되지 않게) ────────
class TestRejectedCandidates:
    """실측으로 기각한 것들. 근거 없이 되살아나면 이 문서가 근거가 된다.

    9개 브레이크 후보(알파분위≥85 모집단 1,015행 · 21일, 일별 paired t):
      ret120≤-40  t=-0.59 p=0.56 | ret60≤-30  t=+0.34 | ret20≤-20 t=-0.97
      POC 아래     t=+0.28        | POC_GAP≤-20 t=+0.28 | ret120&POC t=-0.81
      ret5≤-10    t=-0.65        | ret120&ret60 t=+0.26| LT_PCTL<40 t=+2.00(양수)
    → 전부 유의하지 않았고 LT_PCTL<40은 **양수**(차단하면 손해).

    가드 경고 재검증(유니버스 6,697행 · 21일):
      GUARD_KELLY_MULT=0  차 +1.90%p (t=2.11, p=0.048)  ← 역예측
      강제청산경고=True     차 +1.51%p (t=1.91, p=0.070)  ← 역예측
      알파분위≥85 안에서는 |t|≤0.25로 신호 없음
    → 화면에 경고로 띄우거나 사이징에 되살리면 해롭다. v45 판단 유지.
    """

    def test_no_unvalidated_long_downtrend_brake_was_added(self):
        """검증 실패한 브레이크가 게이트에 들어가지 않았는지 소스로 확인."""
        for name in ("services/recommendation_quality.py", "alpha_engine.py"):
            src = (ROOT / name).read_text(encoding="utf-8")
            for pat in (r"ret_120d_%.{0,20}<=\s*-4", r"ret_60d_%.{0,20}<=\s*-3"):
                assert not re.search(pat, src), \
                    f"{name}에 미검증 장기하락 브레이크가 들어갔다 ({pat})"

    def test_guard_mult_still_not_applied_to_sizing(self):
        """v45 결정 유지 — 가드 배수는 사이징에 곱하지 않는다."""
        src = (ROOT / "kelly_calibrator.py").read_text(encoding="utf-8")
        assert 'df["GUARD_KELLY_MULT_APPLIED"] = 0' in src, \
            "가드 배수가 사이징에 되살아났다 — v45/v61 실측과 반대 방향이다"

    def test_guard_warnings_not_wired_into_ui_as_warnings(self):
        """역예측 지표를 경고로 노출하지 않는다."""
        for name in ("components/action_cards.py", "components/decision_center.py"):
            src = (ROOT / name).read_text(encoding="utf-8")
            assert "GUARD_FORCE_EXIT_ALERT" not in src, \
                f"{name}에 역예측 가드 경고가 배선됐다"
