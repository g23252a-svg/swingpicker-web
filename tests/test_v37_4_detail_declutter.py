# -*- coding: utf-8 -*-
"""[v37.4] 상세 대시보드(stock_detail_v2) 정리 — 중복 제거·AI축 제거·판정 단일화.

사용자 보고: 풀 대시보드가 조잡함(수익률·핵심레벨·TP·판정이 3~4번씩 반복,
AI 0.0 죽은 축 잔존). 중복 섹션을 제거하고 AI 축을 걷어낸다.
"""
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _DummyUi:
    id = "x"

    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
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


def _row():
    return {
        "종목코드": "035720", "종목명": "카카오", "기준일": "20260722",
        "STRUCT_SCORE": 66.4, "TIMING_SCORE": 55.2, "AI_SCORE": 0.0,
        "DISPLAY_SCORE": 60.8, "FINAL_SCORE": 60.8, "ELITE_SCORE": 11.2,
        "BALANCE_SCORE": 17.0, "TRIGGER_SCORE": 54.2, "AXIS_GAP": 66.4,
        "종가": 35500, "추천매수가": 35500, "손절가": 32650,
        "추천매도가1": 36650, "추천매도가2": 49600, "추천매도가3": 58500,
        "ROUTE": "ATTACK", "RR_NOW_TP1": 0.40, "TIME_STOP_DAYS": 7,
        "켈리_수량": 0, "POSITION_PCT": 100,
    }


def test_panel_renders_without_error():
    # 스텁 UI로 전체 조립 함수가 예외 없이 도는지 (AI 축 제거 후 NameError 회귀 방지).
    sd.render_stock_detail_v2_partial(_row(), rank=270, total=323)


def test_assembly_drops_redundant_blocks():
    # 조립부에서 투자자 가이드·하단 4섹터 호출 제거 확인.
    import re
    m = re.search(r"def render_stock_detail_v2_partial\(.*?\n(.*?)\n# ═", SRC, re.S)
    body = m.group(1)
    assert "render_v2_right_guide(n)" not in body
    assert "render_v2_bottom_sectors(" not in body
    # [v38] 최종 판정은 결정 히어로로 흡수 — 단일 판정 SSOT 유지.
    assert "render_v2_decision_hero(" in body


def test_returns_levels_risk_keeps_only_risk():
    # A(수익률)·B(핵심레벨) 3분할 헤더 제거, 리스크/뉴스만 유지.
    assert "A. 수익률 현황" not in SRC
    assert "B. 핵심 레벨" not in SRC
    assert "v2-three-split" not in SRC.split("def render_v2_scenarios")[0].split(
        "def render_v2_returns_levels_risk")[1]


def test_ai_axis_removed_from_scores_and_radar():
    # 3축→2축, 레이더 5축→4축.
    assert "3축 점수" not in SRC
    assert "구조·타이밍 점수" in SRC
    # scores 그리드에 ai 셀 없음
    scores_body = SRC.split("def render_v2_price_plan")[0].split(
        "def render_v2_scores")[1]
    assert 'class="axis-cell ai"' not in scores_body
    # 레이더 축 목록에 AI 없음
    radar_body = SRC.split("def render_v2_right_axisgap")[0].split(
        "def render_v2_radar")[1]
    assert '"AI",' not in radar_body


def test_axis_mean_label_uses_two_axis_average():
    # AI=0 포함 3축 평균(오해 소지) 대신 2축 평균 표기.
    assert "_st_mean = (struct_s + timing_s) / 2.0" in SRC
