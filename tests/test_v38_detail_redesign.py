# -*- coding: utf-8 -*-
"""[v38] 종목 상세 대시보드 근본 재설계 — 결정 히어로 + 접이식 관리자 상세.

사용자 요청: 화면 보고 하나씩 고치지 말고 재설계. 옛 구조(5뱃지 헤더 +
시나리오 3카드 + 최종판정 띠 + 투자자 가이드)가 같은 판단을 4곳에 흩어
놓고 관리자 원시데이터를 평면으로 쏟아내 조잡 → 결정 우선 + 접이식으로 재편.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _DummyUi:
    id = "x"

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
    stub.ui = _DummyUi()
    stub.run = _DummyUi()
    stub.app = _DummyUi()
    sys.modules["nicegui"] = stub

import components.stock_detail_v2 as sd  # noqa: E402

SRC = (ROOT / "components" / "stock_detail_v2.py").read_text(encoding="utf-8")


def _buy_row():
    return {
        "종목코드": "005930", "종목명": "삼성전자", "기준일": "20260722",
        "STRUCT_SCORE": 92, "TIMING_SCORE": 85, "AI_SCORE": 0,
        "DISPLAY_SCORE": 88, "BALANCE_SCORE": 90, "TRIGGER_SCORE": 80,
        "종가": 70000, "추천매수가": 70000, "손절가": 64400,
        "추천매도가1": 78000, "추천매도가2": 85000, "추천매도가3": 95000,
        "ROUTE": "ATTACK", "TOP_PICK": 1, "RR_NOW_TP1": 1.6,
        "켈리_수량": 30, "추천금액(만원)": 210, "TIME_STOP_DAYS": 7,
        "POSITION_PCT": 100, "HONEST_PROB_PCT": 46, "MARKET_REGIME": "NEUTRAL",
        "EBS": 7, "ENTRY_GAP_PCT": 0.0,
    }


def _watch_row():
    r = _buy_row()
    r.update({"종목명": "한국전력", "종목코드": "015760", "ROUTE": "WAIT",
              "TOP_PICK": 0, "RR_NOW_TP1": 2.04, "켈리_수량": 0,
              "추천금액(만원)": 0, "STRUCT_SCORE": 0, "TIMING_SCORE": 12.4})
    return r


# ── 결정 히어로 존재 & 옛 흩어진 판정 제거 ──

def test_assembly_uses_decision_hero_not_old_scatter():
    body = SRC.split("def render_stock_detail_v2_partial")[1].split("\n_NB_GREEN")[0]
    assert "render_v2_decision_hero(" in body
    # 옛 5뱃지 헤더·시나리오·최종판정 띠는 조립부에서 제거.
    assert "render_v2_header(" not in body
    assert "render_v2_scenarios(" not in body
    assert "render_v2_final_verdict(" not in body


def test_admin_detail_is_collapsible():
    body = SRC.split("def render_stock_detail_v2_partial")[1].split("\n_NB_GREEN")[0]
    assert "ui.expansion(" in body
    # 관리자 상세는 접이식 안에서 렌더 (점수·수급 등).
    assert "render_v2_scores(n)" in body
    assert "render_v2_supply(n)" in body


def test_price_and_chart_always_shown():
    body = SRC.split("def render_stock_detail_v2_partial")[1].split("\n_NB_GREEN")[0]
    # 접이식 밖(항상 노출)에 가격 플랜·차트.
    before_exp = body.split("ui.expansion(")[0]
    assert "render_v2_price_plan(n)" in before_exp
    assert "render_v2_chart(" in before_exp


# ── 히어로가 판정 SSOT를 반영 ──

def test_hero_renders_buy_and_watch():
    # 매수·관망 두 상태 모두 예외 없이 렌더.
    sd.render_v2_decision_hero(sd.normalize_stock_row(_buy_row()), rank=1, total=323,
                               timestamp="20260722")
    sd.render_v2_decision_hero(sd.normalize_stock_row(_watch_row()), rank=126, total=323,
                               timestamp="20260722")


def test_full_panel_renders_both_states():
    sd.render_stock_detail_v2_partial(_buy_row(), rank=1, total=323)
    sd.render_stock_detail_v2_partial(_watch_row(), rank=126, total=323)
