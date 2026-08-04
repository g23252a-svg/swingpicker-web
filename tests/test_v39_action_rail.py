# -*- coding: utf-8 -*-
"""[v39] 재혁신 — 한 화면 콤팩트: 결정 히어로 + 가로 액션 레일, 차트/상세는 접이식.

사용자: "뭐가 콤팩트해졌는지 모르겠다. 아예 재혁신. 차트 안 보여줘도 됨."
→ 세로 사다리(190px)를 가로 액션 레일로 대체, 차트를 접이식으로 이동.
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


def _row(**ov):
    r = {
        "종목코드": "015760", "종목명": "한국전력",
        "종가": 33900, "추천매수가": 33900, "손절가": 31150,
        "추천매도가1": 39500, "추천매도가2": 47650, "추천매도가3": 65300,
        "RR_NOW_TP1": 2.04, "ROUTE": "WAIT", "TOP_PICK": 0,
        "켈리_수량": 0, "TIME_STOP_DAYS": 7,
    }
    r.update(ov)
    return r


def _rail_html(row):
    cap = _CapUi()
    orig = sd.ui
    sd.ui = cap
    try:
        sd.render_v2_action_rail(sd.normalize_stock_row(row))
    finally:
        sd.ui = orig
    return cap.htmls[-1]


def test_action_rail_renders_core_only():
    html = _rail_html(_row())
    # 손절·TP1·현재·진입·RR·켈리·타임스톱·다음목표 한 카드에.
    for token in ["손절", "TP1", "현재", "진입", "손익비", "켈리", "타임스톱", "다음 목표"]:
        assert token in html, token


def test_action_rail_marks_current_between_stop_and_tp1():
    html = _rail_html(_row())
    # 현재가(33,900)는 손절(31,150)~TP1(39,500) 사이 → 마커 위치 0<pos<100.
    import re
    m = re.search(r"left:([\d.]+)%;top:50%", html)
    assert m and 0.0 < float(m.group(1)) < 100.0


def test_action_rail_shows_kelly_zero_with_reason():
    """[v53] 0주는 사유와 함께 표시해야 한다.

    v39는 '신규 부적합(0주)'로만 적었다. v45가 KELLY_ZERO_REASON을 만들어
    0주 사유를 분류했는데(실측 승률 vs 필요 승률 / 배분액<1주 가격 /
    상태 미달 / 가격정보 불완전) UI 배선이 빠져 화면에는 안 나왔다.
    '침묵의 0주 금지'가 v45의 목적이므로 사유 노출까지가 완결이다.
    """
    html = _rail_html(_row(켈리_수량=0))
    assert "0주" in html
    assert "0주 사유" in html, "사유 블록이 없다 — 침묵의 0주"


def test_action_rail_uses_engine_zero_reason_when_present():
    html = _rail_html(_row(켈리_수량=0,
                          KELLY_ZERO_REASON="실측 승률 41% < 필요 승률 36%"))
    assert "실측 승률 41%" in html


def test_action_rail_zero_reason_falls_back_without_column():
    """구 CSV(사유 컬럼 없음)에서도 최소 문구는 나와야 한다."""
    row = _row(켈리_수량=0)
    row.pop("KELLY_ZERO_REASON", None)
    html = _rail_html(row)
    assert "켈리 기준 미달" in html


def test_action_rail_no_zero_block_when_qty_positive():
    html = _rail_html(_row(켈리_수량=12))
    assert "0주 사유" not in html


def test_full_panel_v39_renders():
    sd.render_stock_detail_v2_partial(_row(), rank=126, total=323)


def _plan_html(row):
    cap = _CapUi()
    orig = sd.ui
    sd.ui = cap
    try:
        sd.render_v2_price_plan(sd.normalize_stock_row(row))
    finally:
        sd.ui = orig
    return cap.htmls[-1]


def test_price_levels_use_table_not_absolute_ladder():
    # [v39.1] 겹치던 절대위치 사다리 → 겹칠 수 없는 정렬 표.
    html = _plan_html(_row(추천매도가1=36650, 추천매도가2=49600, 추천매도가3=58500,
                           종가=35500, 추천매수가=35500, 손절가=32650))
    assert "가격 레벨" in html
    assert "position:absolute" not in html  # 절대위치 제거 → 겹침 불가
    # 모든 레벨이 각각의 행으로 존재.
    for lv in ["TP3", "TP2", "TP1", "본전전환", "손절"]:
        assert lv in html
