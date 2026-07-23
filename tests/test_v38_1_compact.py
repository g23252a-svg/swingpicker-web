# -*- coding: utf-8 -*-
"""[v38.1] 상세 콤팩트 조정 — 차트/보조지표 높이 축소 + 사다리 라벨 겹침 수정.

사용자 보고: 차트 아래가 너무 길고 콤팩트한 느낌이 없음. 가격 사다리에서
진입==현재가(GAP 0%)일 때 라벨이 정확히 겹침.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


_ui = _CapUi()
if "nicegui" not in sys.modules:
    stub = types.ModuleType("nicegui")
    stub.ui = _ui
    stub.run = _ui
    stub.app = _ui
    sys.modules["nicegui"] = stub
    ctx = types.ModuleType("nicegui.context")
    ctx.client = type("C", (), {})()
    sys.modules["nicegui.context"] = ctx

import components.stock_detail_v2 as sd  # noqa: E402

SRC = (ROOT / "components" / "stock_detail_v2.py").read_text(encoding="utf-8")


def test_main_chart_height_reduced():
    assert "height: 320px" in SRC
    assert "height: 400px" not in SRC  # 옛 큰 높이 제거


def test_subcharts_use_shorter_sparklines():
    # 스파크라인/히스토그램을 30px로 호출(콤팩트).
    assert "height=30" in SRC


def _row(**ov):
    r = {
        "종목코드": "015760", "종목명": "한국전력",
        "종가": 33900, "추천매수가": 33900, "손절가": 31150,
        "추천매도가1": 39500, "추천매도가2": 47650, "추천매도가3": 65300,
        "RR_NOW_TP1": 2.04, "ROUTE": "WAIT",
    }
    r.update(ov)
    return r


def _render_plan_html(row):
    """모듈 전역 ui가 다른 테스트 스텁일 수 있으므로 캡처 stub으로 직접 교체."""
    cap = _CapUi()
    orig = sd.ui
    sd.ui = cap
    try:
        sd.render_v2_price_plan(sd.normalize_stock_row(row))
    finally:
        sd.ui = orig
    return cap.htmls[-1]


def test_ladder_merges_entry_and_close_when_equal():
    html = _render_plan_html(_row())
    # GAP 0% → 병합된 단일 라벨.
    assert "현재/진입" in html
    # 개별 '현재가'/'진입' 중복 라벨은 없어야 함(겹침 방지).
    assert ">현재가<" not in html


def test_ladder_keeps_separate_rows_when_gap_nonzero():
    html = _render_plan_html(_row(추천매수가=33000))  # close 33900 vs entry 33000
    assert "현재/진입" not in html
    assert "현재가" in html and "진입" in html
