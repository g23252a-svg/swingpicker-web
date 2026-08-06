# -*- coding: utf-8 -*-
"""[v54] 첫 실전 픽에서 드러난 3대 모순 — 024060 흥구석유 20260805.

v45~v53 패치 후 처음으로 실제 공식 매수가 나왔고, 그 한 종목이 세 개의
표시 모순을 동시에 드러냈다. 사용자가 화면만 보고 전부 찾아냈다.

■ ① '공식 매수'인데 0주 — 두 곳이 서로 모르고 각자 판정했다
    recommend_latest.csv 실측:
      PRODUCTION_BUY=1 · BUY_NOW_ELIGIBLE=1 · TOP_PICK=1
      ROUTE=WAIT · 켈리_수량=0 · 추천금액(만원)=0 · POSITION_PCT=100
      KELLY_ZERO_REASON="상태 WAIT — 신규진입 상태 아님(사이징 제외)"
    시장 탭은 '공식 매수 · 엄격 매수 기준 통과 · 최대 2% 비중',
    상세 탭은 '관망 — 신규매수 대상 아님'을 같은 종목에 동시 표시.
    원인: recommendation_quality는 ROUTE를 보지 않고 PRODUCTION_BUY를 정하고,
          kelly_calibrator.resize_kelly_with_alpha는 _KELLY_INACTIVE_ROUTES로
          사이징을 0으로 강제한다.
    고치는 길이 둘이고, 수익 실측으로는 갈리지 않았다.
      (A) 켈리의 ROUTE 거부권 제거 → 픽이 실제 매수 가능해진다
      (B) 사이징 불가 상태를 공식 매수라 부르지 않는다 → 픽이 줄어든다
    처음엔 (B)를 지지하는 실측으로 읽었다(사이징 가능 +4.09% vs 차단 -1.13%,
    n=55/1,440 · 69일). 분해하니 채택 기준 미달이었다:
      · ATTACK n=9 — v32가 검정한 축(ATTACK 알파 -2.9%p, p=0.0004)은 지금 풀에
        표본이 없다. v54 측정은 v32와 모순된 게 아니라 다른 것(ARMED)을 봤다.
      · +4%는 ARMED 효과(n=46·18일, t=+1.16, p=0.26). IS만 표본이 되고
        OOS는 검정 불가 → 'IS·OOS 양쪽 유의' 기준 미달.
      · 교란 큼: 사이징 가능 행은 RR 1.96 vs 3.32 (t=-8.63),
        POC -1.02 vs -7.90 (t=+4.88). 층화하면 셀이 빈다.
      · 날짜 클러스터 부트 CI95 [-1.82, +15.78] — 0 포함. 에피소드 6건.
    → 수익은 채택 사유가 아니다. (B)를 고른 근거는 위험 비대칭이다:
      켈리의 ROUTE 0원 정책 자체가 v32에서 '표시 관례'로 남은 미검증 규칙이라
      (A)는 검증 안 된 방향으로 실제 자금을 태운다. (B)는 새 위험을 만들지
      않고 라벨만 실행 가능성에 맞춘다. 되돌리기도 한 줄.
      비용: 같은 풀에서 공식 매수가 나오는 날 69일 중 18일(26%).

■ ② 승률이 화면마다 달랐다 — 둘 다 '실측 승률'
    ALPHA_WIN_PROB 0.389 (시장 탭 39%) : 검증 알파 십분위 OOS 승률.
      정의 = 진입 t+1 시가 → t+6 종가 > 0 비율.
    EST_WIN_RATE 0.291 (상세 탭 29%)   : ELITE_SCORE 버킷 체결원장 실현 승률.
    같은 라벨이면 하나가 거짓말로 읽힌다. 진입 SSOT는 v32 이후 검증 알파이고
    ELITE 축은 v52 감사에서 예측력이 확인되지 않았다(품질점수 랭킹 -1.29%/일).
    → 대표값 = 알파 십분위, 기준 문구(HONEST_PROB_BASIS)를 값과 함께 표시.

■ ③ 손익비 5.06:1이 기대값처럼 보였다
    같은 행의 TP1_PROB=12 · TP2_PROB=5 · TP3_PROB=2 (v42 실측 도달률).
    두 결과 근사 확률가중: 0.12×(+40.6%) + 0.88×(-8%) = -2.2% → 음수.
    게이트 통과 풀에서 RR 자체의 예측 서열도 확인되지 않았다:
      일별 IC 평균 -0.0097 (t=-0.27, p=0.79) · 구간 비단조
      (RR 3~5 +0.69% > RR 5+ -0.10%)
    → 도달률을 손익비와 같은 줄에, 도달률<25%이고 근사 기대값이 음수면 경고.
    주의: 과거 패널의 TP1_PROB는 v42 이전 method 고정 confidence(25~80,
    중위 80)라 v42 실측 도달률(46/23/12/5/2) 구간을 담고 있지 않다. 즉 이
    경고 경로의 수익 효과는 과거 데이터로 검증할 수 없다 — 근거는 실측이
    아니라 산술이다(비율에는 도달 확률이 들어 있지 않다).
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kelly_calibrator as kc  # noqa: E402
import services.recommendation_quality as rq  # noqa: E402
from services.reco_evidence import add_evidence_columns, build_evidence_row  # noqa: E402


# ══════════════════════════════════════════════
# ① 공식 매수 = 실행 가능
# ══════════════════════════════════════════════

def test_inactive_route_set_is_shared_with_kelly():
    """두 모듈이 다른 집합을 쓰면 다시 '살 수 없는 공식 매수'가 생긴다."""
    assert rq._inactive_routes() == frozenset(
        str(r).strip().upper() for r in kc._KELLY_INACTIVE_ROUTES)


def test_fallback_matches_kelly_set():
    """지연 import 실패 시 쓰는 폴백 리터럴도 같아야 한다."""
    assert rq._KELLY_INACTIVE_ROUTES_FALLBACK == frozenset(
        str(r).strip().upper() for r in kc._KELLY_INACTIVE_ROUTES)


def test_route_sizable_mask_blocks_inactive_routes():
    df = pd.DataFrame({"ROUTE": ["WAIT", "NEUTRAL", "CARRY", "OVERHEAT",
                                 "EXIT_WARNING", "BLOCKED"]})
    assert not rq._route_sizable_mask(df).any()


def test_route_sizable_mask_allows_active_routes():
    df = pd.DataFrame({"ROUTE": ["ATTACK", "ARMED", "attack", " Armed "]})
    assert rq._route_sizable_mask(df).all()


def test_route_sizable_mask_passes_when_unknown():
    """컬럼 없음·빈 값은 통과 — 구 CSV에서 픽을 죽이지 않는다."""
    assert rq._route_sizable_mask(pd.DataFrame({"종목코드": ["000001"]})).all()
    assert rq._route_sizable_mask(pd.DataFrame({"ROUTE": ["", None]})).all()


def _guard_row(**ov):
    """알파 게이트 경로로 PRODUCTION_BUY까지 가는 최소 행."""
    r = {
        "종목코드": "024060", "종목명": "흥구석유",
        "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1,
        "ALPHA_GATE_ACTIVE": 1, "ALPHA_ENTRY_OK": 1,
        "ALPHA_SCORE": 100.0, "ALPHA_VALIDATED": 1,
        "RR_NOW_TP1": 5.06, "POC_GAP": 5.0, "MARKET_BREADTH": 55.0,
        "MACRO_RISK": "NORMAL", "DATA_FRESHNESS_OK": 1,
        "MARKET_REGIME": "NEUTRAL", "ROUTE": "ATTACK",
        "DISPLAY_SCORE": 60.0, "BUY_NOW_SCORE": 60.0,
    }
    r.update(ov)
    return r


def test_active_route_still_becomes_production_buy():
    """거부권이 정상 픽을 죽이지 않는다 (회귀 방지)."""
    out = rq.apply_recommendation_quality_guard(pd.DataFrame([_guard_row()]))
    assert int(out.iloc[0]["PRODUCTION_BUY"]) == 1


def test_wait_route_is_not_production_buy():
    """024060 케이스 — 사이징이 0으로 강제되는 상태는 공식 매수가 아니다."""
    out = rq.apply_recommendation_quality_guard(
        pd.DataFrame([_guard_row(ROUTE="WAIT")]))
    assert int(out.iloc[0]["PRODUCTION_BUY"]) == 0
    assert int(out.iloc[0]["BUY_NOW_ELIGIBLE"]) == 0
    assert int(out.iloc[0]["QUALITY_GUARD_PASS"]) == 1, "근거 자체는 통과여야 한다"


def test_wait_route_reason_names_the_state_not_daily_cap():
    """'당일 1종목 제한'이라 적으면 거짓 사유다 — 제한 때문이 아니다."""
    out = rq.apply_recommendation_quality_guard(
        pd.DataFrame([_guard_row(ROUTE="WAIT")]))
    reason = str(out.iloc[0]["QUALITY_GUARD_REASON"])
    assert "WAIT" in reason and "신규진입 상태 아님" in reason
    assert "1종목 제한" not in reason


def test_daily_cap_reason_survives_for_sizable_runner_up():
    """사이징 가능한 2등은 여전히 '당일 1종목 제한'이 맞는 사유다."""
    df = pd.DataFrame([
        _guard_row(종목코드="000001", ALPHA_SCORE=100.0),
        _guard_row(종목코드="000002", ALPHA_SCORE=99.0),
    ])
    out = rq.apply_recommendation_quality_guard(df)
    assert int(out["PRODUCTION_BUY"].sum()) == 1
    loser = out[out["PRODUCTION_BUY"] == 0].iloc[0]
    assert "1종목 제한" in str(loser["QUALITY_GUARD_REASON"])


def test_inactive_route_does_not_steal_the_daily_pick():
    """WAIT 1등이 슬롯을 먹어 실행 가능한 2등이 탈락하면 안 된다."""
    df = pd.DataFrame([
        _guard_row(종목코드="024060", ROUTE="WAIT", ALPHA_SCORE=100.0),
        _guard_row(종목코드="000002", ROUTE="ATTACK", ALPHA_SCORE=90.0),
    ])
    out = rq.apply_recommendation_quality_guard(df).set_index("종목코드")
    assert int(out.loc["024060", "PRODUCTION_BUY"]) == 0
    assert int(out.loc["000002", "PRODUCTION_BUY"]) == 1


# ── ①-b 픽이 줄어든 대가를 화면이 설명하는가 ──
#   v54가 사이징 불가 상태를 공식 매수에서 뺀 결과 0개인 날이 늘어난다.
#   그 날 '아무것도 못 찾았다'로 읽히면 사용자는 엔진을 불신한다 —
#   실제로는 종목은 찾았고 타이밍 상태가 아닌 것이다.

def test_awaiting_mask_finds_evidence_passing_but_unsizable():
    out = rq.apply_recommendation_quality_guard(
        pd.DataFrame([_guard_row(ROUTE="WAIT")]))
    assert bool(rq.awaiting_execution_mask(out).iloc[0])


def test_awaiting_mask_excludes_the_official_pick():
    out = rq.apply_recommendation_quality_guard(pd.DataFrame([_guard_row()]))
    assert int(out.iloc[0]["PRODUCTION_BUY"]) == 1
    assert not bool(rq.awaiting_execution_mask(out).iloc[0])


def test_awaiting_mask_excludes_evidence_failures():
    """근거 자체가 미달인 종목을 '상태만 문제'라고 하면 거짓말이다."""
    out = rq.apply_recommendation_quality_guard(
        pd.DataFrame([_guard_row(ROUTE="WAIT", RR_NOW_TP1=0.4)]))
    assert int(out.iloc[0]["QUALITY_GUARD_PASS"]) == 0
    assert not bool(rq.awaiting_execution_mask(out).iloc[0])


def test_awaiting_mask_empty_frame_is_safe():
    assert len(rq.awaiting_execution_mask(pd.DataFrame())) == 0


def test_decision_summary_names_the_waiting_candidate():
    from components.decision_center import build_decision_summary

    df = rq.apply_recommendation_quality_guard(
        pd.DataFrame([_guard_row(ROUTE="WAIT")]))
    s = build_decision_summary(df)
    assert s["status"] == "CASH"
    assert s["production_count"] == 0
    assert len(s["awaiting"]) == 1
    assert "흥구석유" in s["subtitle"] and "WAIT" in s["subtitle"]
    assert "종목이 없는 게 아니라" in s["subtitle"]
    assert "통과한 종목이 없음" not in " ".join(s["blockers"]), "거짓 차단자"
    # 실제 차단자가 목록 1순위여야 한다 (다른 종목 사유가 위에 오면 안 됨)
    assert "흥구석유" in s["blockers"][0] and "0주" in s["blockers"][0]


def test_decision_summary_unchanged_when_pick_is_executable():
    from components.decision_center import build_decision_summary

    df = rq.apply_recommendation_quality_guard(pd.DataFrame([_guard_row()]))
    s = build_decision_summary(df)
    assert s["status"] == "BUY" and s["awaiting"] == []


# ══════════════════════════════════════════════
# ② 승률 하나 · 기준 명시
# ══════════════════════════════════════════════

_CASE_024060 = {
    "ALPHA_VALIDATED": 1, "ALPHA_SCORE": 100.0, "ALPHA_WIN_PROB": 0.3888,
    "EST_WIN_RATE": 0.291, "EST_WIN_RATE_N": 157, "EST_WIN_RATE_MODE": "MATURE",
}


def test_alpha_prob_is_the_representative_win_rate():
    e, _k, p, basis = build_evidence_row(dict(_CASE_024060))
    assert p == 38.9, f"대표 승률이 알파 십분위가 아니다: {p}"
    assert "알파" in basis and "5일" in basis
    assert "알파 십분위 실측 승률 39%" in e


def test_legacy_win_rate_is_kept_but_labeled_as_different_basis():
    e, _k, _p, _b = build_evidence_row(dict(_CASE_024060))
    assert "29%" in e, "레거시 값을 지우면 감사가 불가능하다"
    assert "기준 다름" in e


def test_win_rate_is_not_printed_twice():
    """같은 승률을 두 문장에 넣으면 서로 다른 값처럼 읽힌다."""
    e, _k, _p, _b = build_evidence_row(dict(_CASE_024060))
    assert e.count("39%") == 1, e


def test_legacy_basis_used_when_alpha_absent():
    e, _k, p, basis = build_evidence_row({
        "EST_WIN_RATE": 0.62, "EST_WIN_RATE_N": 120, "EST_WIN_RATE_MODE": "MATURE"})
    assert p == 62.0
    assert "점수구간" in basis
    assert "실측 승률 62%" in e


def test_basis_empty_when_no_win_rate_evidence():
    _e, _k, p, basis = build_evidence_row({"POC_GAP": 5.0})
    assert np.isnan(p) and basis == ""


def test_basis_column_is_written_in_bulk():
    out = add_evidence_columns(pd.DataFrame([dict(_CASE_024060)]))
    assert "HONEST_PROB_BASIS" in out.columns
    assert "알파" in str(out.loc[0, "HONEST_PROB_BASIS"])
    assert float(out.loc[0, "HONEST_PROB_PCT"]) == 38.9


def test_detail_dict_carries_basis():
    src = (ROOT / "components" / "stock_detail_v2.py").read_text(encoding="utf-8")
    assert '"honest_prob_basis"' in src
    assert "이 점수 구간의 과거 성적" not in src, "기준이 하드코딩으로 남아 있다"


# ══════════════════════════════════════════════
# ③ 손익비는 기대값이 아니다
# ══════════════════════════════════════════════

class _CapUi:
    id = "x"

    def __init__(self):
        self.htmls = []

    def html(self, s=""):
        self.htmls.append(str(s))
        return self

    def echart(self, *a, **k):
        return self

    def __getattr__(self, _n):
        return self

    def __call__(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if "nicegui" not in sys.modules:
    stub = types.ModuleType("nicegui")
    _u = _CapUi()
    stub.ui = _u
    stub.run = _u
    stub.app = _u
    sys.modules["nicegui"] = stub
    ctx = types.ModuleType("nicegui.context")
    ctx.client = type("C", (), {})()
    sys.modules["nicegui.context"] = ctx

import components.stock_detail_v2 as sd  # noqa: E402


def _rail(**ov):
    row = {
        "종목코드": "024060", "종목명": "흥구석유",
        "종가": 10000, "추천매수가": 10000, "손절가": 9200,
        "추천매도가1": 14060, "추천매도가2": 17000, "추천매도가3": 20000,
        "RR_NOW_TP1": 5.06, "TP1_PROB": 12, "TP2_PROB": 5, "TP3_PROB": 2,
        "ROUTE": "WAIT", "TOP_PICK": 1, "켈리_수량": 0, "TIME_STOP_DAYS": 7,
    }
    row.update(ov)
    cap = _CapUi()
    orig = sd.ui
    sd.ui = cap
    try:
        sd.render_v2_action_rail(sd.normalize_stock_row(row))
    finally:
        sd.ui = orig
    return cap.htmls[-1]


def test_action_rail_shows_tp1_reach_probability():
    html = _rail()
    assert "TP1 도달률" in html and "12%" in html


def test_action_rail_warns_when_probability_weighted_ev_is_negative():
    html = _rail()
    assert "확률 가중이 아닙니다" in html
    assert "기대값" in html


def test_ev_warning_does_not_read_as_bad_stock():
    """[v55] 경고가 '이 종목은 나쁘다'로 읽히면 실측과 어긋난다.

    v55 실측: TP1까지 먼(>20%) 후보가 오히려 좋았고(가까운-먼 -3.49%p,
    t=-1.74, p=0.094), 랭킹에서 RR 상한을 낮추면 해로웠다(min(RR,2)
    -2.07%p, 블록부트 양수 2%). 공식 픽은 드문 대박이 수익의 전부인
    우측 꼬리 분포이므로 먼 목표가 곧 꼬리가 사는 자리다.
    경고의 뜻은 '기대를 낮춰라'가 아니라 '한 번에 크게 걸지 말라'다.
    """
    html = _rail()
    assert "목표가가 먼 후보가 오히려 나았습니다" in html
    assert "분할·비중으로 대응" in html


def test_no_ev_warning_when_reach_probability_is_healthy():
    html = _rail(TP1_PROB=46)
    assert "TP1 도달률" in html
    assert "확률 가중이 아닙니다" not in html


def test_no_ev_warning_when_weighted_ev_is_positive():
    """도달률이 낮아도 근사 기대값이 양수면 경고하지 않는다."""
    html = _rail(TP1_PROB=24, 추천매도가1=15000, 손절가=9800)
    assert "확률 가중이 아닙니다" not in html


def test_reach_probability_absent_is_not_faked():
    html = _rail(TP1_PROB=0)
    assert "TP1 도달률" not in html


def test_rr_sublabel_declares_it_is_not_expected_value():
    src = (ROOT / "components" / "stock_detail_v2.py").read_text(encoding="utf-8")
    assert "확률 가중 아님" in src
    assert "t=-0.27" in src, "RR 무예측 실측 근거 미기록"
