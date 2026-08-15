# -*- coding: utf-8 -*-
"""[v57] 알파 진입을 통과한 관망(WAIT)은 사이징을 막지 않는다.

■ 왜 바꿨나
  v32는 "ROUTE는 진입 게이트가 아니다"(ATTACK 알파 -2.9%p, p=0.0004)를 근거로
  ROUTE 진입 거부권을 제거하고 검증 알파를 진입 SSOT로 삼았다. 그런데 사이징
  레이어(kelly_calibrator.resize_kelly_with_alpha)는 알파 게이트 통과 후에도
  비활성 ROUTE의 수량을 0으로 **덮어썼고**(코드 주석이 스스로 "표시 관례 유지"라
  적는다), v54가 그 0원 상태를 PRODUCTION_BUY 조건으로 승격시키면서 **ROUTE
  거부권이 뒷문으로 복원**됐다.

■ 빈도 비용 (v55 워크포워드 패널 76일)
  품질 가드 통과 후보 429종목 중 WAIT가 65.3%(280) → 픽 가능일 29일 → 15일로 반감.

■ 수익 근거 (실현: t+1 시가 진입 · -8% 스톱 · t+5 종가 · 하루 1픽)
  현행(ARMED/ATTACK만): 15일 평균 +6.20% · 중위 **-2.70%** · 상위2제거 **-2.89%**
                        · OOS 4일 **-1.22%** (IS +8.89%에서 부호 역전)
  WAIT 면제:            29일 평균 +2.89% · 중위 **+1.70%** · 승률 **58.6%**
                        · 상위2제거 **+1.35%** · t=+1.89 p=0.069
                        · 부트CI95 **[+0.1, +5.9]** · IS +2.14% / OOS +3.15%
  종목 단위 '사이징 가능' 예측력: Welch p=0.296 · MW p=0.598 ·
                        각 상위2 제거 후 차이 0.16%p → **없음**

■ 정직하게 남기는 것
  · 평균은 오히려 낮아진다(+6.20% → +2.89%). 채택 사유는 중위·승률·이상치제거·
    OOS 네 지표가 모두 같거나 낫고 **빈도가 2배**라는 것이다.
  · 공통일 페어드는 무의미하다(15일 Δ-2.37%p, t=-0.32, p=0.756) — 개선의 원천은
    "같은 날 더 좋은 종목"이 아니라 **"더 많은 날 진입한다"**는 것이다.
  · p=0.069는 관례적 0.05를 넘지 않는다. 29일 · OOS 6일로 표본이 얇다.

■ 이 파일이 고정하는 것
  1. 면제 대상은 **WAIT 하나**다. 넓히려면 다시 재야 한다.
  2. 면제는 **알파 진입 통과에만** 붙는다 (ROUTE만으로 열리면 v32 이전 회귀).
  3. 두 레이어(켈리 · 공식매수 판정)가 **같은 규칙**을 쓴다 — 어긋나면 v54가
     없앤 '공식 매수인데 0주'가 되살아난다.
  4. v54 불변식 유지: 사이징 불가 상태는 공식 매수가 아니다(CARRY 등).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kelly_calibrator as kc  # noqa: E402
import services.recommendation_quality as rq  # noqa: E402

STILL_BLOCKED = ["CARRY", "OVERHEAT", "EXIT_WARNING", "BLOCKED", "NEUTRAL"]


def _row(**ov):
    r = {
        "종목코드": "024060", "종목명": "흥구석유",
        "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1,
        "ALPHA_GATE_ACTIVE": 1, "ALPHA_ENTRY_OK": 1,
        "ALPHA_SCORE": 100.0, "ALPHA_VALIDATED": 1,
        "RR_NOW_TP1": 5.06, "POC_GAP": 5.0, "MARKET_BREADTH": 55.0,
        "MACRO_RISK": "NORMAL", "DATA_FRESHNESS_OK": 1,
        "MARKET_REGIME": "NEUTRAL", "ROUTE": "WAIT",
        "DISPLAY_SCORE": 60.0, "BUY_NOW_SCORE": 60.0,
    }
    r.update(ov)
    return pd.DataFrame([r])


# ── 1. 면제 범위는 WAIT 하나 ──────────────────────────────────────
class TestExemptScope:
    def test_exempt_set_is_exactly_wait(self):
        """넓히는 것은 정책 변경이다 — 재측정 없이 늘어나면 실패한다."""
        assert set(kc._ALPHA_EXEMPT_INACTIVE_ROUTES) == {"WAIT"}

    def test_exempt_is_subset_of_inactive(self):
        assert set(kc._ALPHA_EXEMPT_INACTIVE_ROUTES) <= set(kc._KELLY_INACTIVE_ROUTES)

    def test_inactive_set_unchanged(self):
        """비활성 집합 자체는 v54와 동일 — 바뀐 것은 면제 규칙뿐이다."""
        assert set(kc._KELLY_INACTIVE_ROUTES) == {
            "WAIT", "OVERHEAT", "EXIT_WARNING", "CARRY", "BLOCKED", "NEUTRAL"}

    @pytest.mark.parametrize("route", STILL_BLOCKED)
    def test_other_inactive_routes_stay_blocked(self, route):
        """CARRY(보유 중·승률 26.7%)·청산·차단·과열은 그대로 막힌다."""
        out = rq.apply_recommendation_quality_guard(_row(ROUTE=route))
        assert int(out.iloc[0]["PRODUCTION_BUY"]) == 0
        assert not bool(rq._route_sizable_mask(_row(ROUTE=route)).iloc[0])


# ── 2. 면제는 알파 진입 통과에만 붙는다 ───────────────────────────
class TestAlphaConditioned:
    def test_wait_with_alpha_is_sizable(self):
        assert bool(rq._route_sizable_mask(_row(ROUTE="WAIT")).iloc[0])

    def test_wait_without_alpha_is_not_sizable(self):
        assert not bool(rq._route_sizable_mask(_row(ROUTE="WAIT", ALPHA_ENTRY_OK=0)).iloc[0])

    def test_wait_without_alpha_column_is_not_sizable(self):
        """컬럼이 아예 없으면(레거시 CSV) 면제하지 않는다 — 보수적 방향."""
        df = _row(ROUTE="WAIT").drop(columns=["ALPHA_ENTRY_OK"])
        assert not bool(rq._route_sizable_mask(df).iloc[0])

    def test_production_buy_follows(self):
        assert int(rq.apply_recommendation_quality_guard(
            _row(ROUTE="WAIT")).iloc[0]["PRODUCTION_BUY"]) == 1
        assert int(rq.apply_recommendation_quality_guard(
            _row(ROUTE="WAIT", ALPHA_ENTRY_OK=0)).iloc[0]["PRODUCTION_BUY"]) == 0


# ── 3. 두 레이어가 같은 규칙을 쓴다 (v54의 요지) ──────────────────
class TestTwoLayersAgree:
    @pytest.mark.parametrize("route", ["WAIT", "ATTACK", "ARMED"] + STILL_BLOCKED)
    @pytest.mark.parametrize("alpha_ok", [1, 0])
    def test_kelly_zero_mask_is_inverse_of_sizable(self, route, alpha_ok):
        """켈리가 0을 강제하는 행 == 공식 매수 판정이 사이징 불가로 보는 행."""
        df = _row(ROUTE=route, ALPHA_ENTRY_OK=alpha_ok)
        zero = bool(kc.kelly_route_zero_mask(df).iloc[0])
        sizable = bool(rq._route_sizable_mask(df).iloc[0])
        assert zero is (not sizable), (
            f"두 레이어 불일치: ROUTE={route} alpha={alpha_ok} "
            f"kelly_zero={zero} sizable={sizable}")

    def test_quality_guard_reads_kelly_ssot(self):
        """면제 집합의 원천은 kelly_calibrator다 (지연 import)."""
        assert rq._alpha_exempt_routes() == frozenset(
            str(r).upper() for r in kc._ALPHA_EXEMPT_INACTIVE_ROUTES)

    def test_zero_mask_safe_on_missing_route(self):
        assert not kc.kelly_route_zero_mask(pd.DataFrame({"종목코드": ["000001"]})).any()
        assert len(kc.kelly_route_zero_mask(pd.DataFrame())) == 0


# ── 4. 실제 사이징에서도 수량이 살아남는가 ────────────────────────
class TestKellySizing:
    def test_wait_with_alpha_keeps_quantity(self):
        """켈리 0원 강제가 WAIT+알파 행을 건드리지 않는다."""
        df = _row(ROUTE="WAIT")
        df["켈리_수량"] = 100
        df["켈리_금액(원)"] = 1_000_000
        zero = kc.kelly_route_zero_mask(df)
        assert not bool(zero.iloc[0])

    def test_carry_is_zeroed(self):
        df = _row(ROUTE="CARRY")
        assert bool(kc.kelly_route_zero_mask(df).iloc[0])


# ── 5. 실제 픽 재현 — 024060 (20260805) ──────────────────────────
def test_real_pick_20260805_is_reproduced():
    """v54 계약에서는 이 픽이 탈락했다. v57에서는 재현되어야 한다.
    (CSV가 없는 환경에서는 스킵 — 데이터 의존 테스트)"""
    p = ROOT / "data" / "recommend_20260805.csv"
    if not p.exists():
        pytest.skip("recommend_20260805.csv 없음 (배치 미실행 환경)")
    rec = pd.read_csv(p, encoding="utf-8-sig", dtype={"종목코드": str}, low_memory=False)
    if "ALPHA_ENTRY_OK" not in rec.columns:
        pytest.skip("이 CSV에 알파 컬럼이 없다 — 계약 재생 불가")
    g = rq.apply_recommendation_quality_guard(
        rec.drop(columns=["PRODUCTION_BUY"], errors="ignore"))
    picks = g[rq.production_buy_mask(g)]
    assert len(picks) == 1, f"픽 {len(picks)}건 (1건이어야 한다)"
    row = picks.iloc[0]
    assert str(row["종목코드"]).zfill(6) == "024060"
    assert str(row["ROUTE"]).strip().upper() == "WAIT"


def test_policy_version_bumped():
    """계약이 바뀌면 버전도 바뀌어야 한다 — 구 CSV 판정 구분에 쓰인다.

    [v62] 알파 진입 게이트에 급등 추격 브레이크(ALPHA_SURGE_OK)가 AND로
    추가되어 진입 계약이 바뀌었다. 그래서 v57 → v62로 올렸다.
    기대값을 갱신하는 것이 이 테스트의 목적이다(검사를 지우는 것이 아니다).
    """
    assert rq.POLICY_VERSION == "alpha_gate_v62"
    assert rq.POLICY_VERSION != "alpha_gate_v57", \
        "계약이 바뀌었는데 버전이 그대로다"
