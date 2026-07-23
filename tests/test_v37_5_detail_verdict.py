# -*- coding: utf-8 -*-
"""[v37.5] 상세 대시보드 2차 정리 — 판정 모순·중복 가격행·오해 라벨 수정.

사용자 보고(한국전력 화면): 관망/WAIT·TOP_PICK 없음·KELLY 0인데
헤더는 '🌑 핵심매수', 최종판정은 '양호한 진입 후보'로 떠 모순.
+ 가격 사다리 아래 종가/매수가/손절/TP가 텍스트로 그대로 반복.
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


# ── 판정 게이트: TOP_PICK이 SSOT ──

def test_watch_stock_not_labeled_buy_candidate():
    # 한국전력 사례: 관망(top_pick False)·RR 좋아도 매수 후보 아님.
    label, _, icon = sd._verdict_grade(top_pick=False, qty=0, rr=2.04, is_overheat=False)
    assert "관망" in label
    assert "진입 후보" not in label
    assert icon == "—"


def test_kelly_zero_blocks_buy_even_if_top_pick():
    label, _, _ = sd._verdict_grade(top_pick=True, qty=0, rr=1.8, is_overheat=False)
    assert "관망" in label


def test_overheat_is_rejected_first():
    label, _, icon = sd._verdict_grade(top_pick=True, qty=10, rr=2.0, is_overheat=True)
    assert "부적합" in label and icon == "⚠️"


def test_real_buy_grades_by_rr():
    hi, _, _ = sd._verdict_grade(True, 10, 1.5, False)
    assert "콤보 등급" in hi
    mid, _, _ = sd._verdict_grade(True, 10, 1.1, False)
    assert mid == "양호한 진입 후보"
    low, _, _ = sd._verdict_grade(True, 10, 0.7, False)
    assert "관망" in low


# ── 헤더 오해 라벨 ──

def test_header_core_badge_relabeled_neutral():
    assert "🌑 핵심매수" not in SRC
    assert "🏷️ 콤보 등급" in SRC


# ── 가격행 중복 제거 ──

def test_price_rows_removed_from_main_body():
    plan = SRC.split("def _trend_arrow")[0].split("def render_v2_price_plan")[1]
    # 사다리 밖 상시 렌더 가격행(툴팁 포함) 제거 — 폴백(ladder 없을 때)만 잔존.
    assert '<span class="lbl" title="전일 종가 (기준일 마감가)">종가</span>' not in plan
    assert "목표 근거" in plan  # 컴팩트 근거 라인으로 대체


def test_celebratory_rocket_removed():
    verdict = SRC.split("def render_v2_radar")[0].split("def render_v2_final_verdict")[1]
    assert "🚀" not in verdict


# ── 전체 조립 렌더 (WAIT 종목) 스모크 ──

def test_full_panel_render_watch_stock():
    row = {
        "종목코드": "015760", "종목명": "한국전력", "기준일": "20260722",
        "STRUCT_SCORE": 0.0, "TIMING_SCORE": 12.4, "AI_SCORE": 0.0,
        "DISPLAY_SCORE": 6.2, "BALANCE_SCORE": 84.5, "TRIGGER_SCORE": 24.7,
        "종가": 33900, "추천매수가": 33900, "손절가": 31150,
        "추천매도가1": 39500, "추천매도가2": 47650, "추천매도가3": 65300,
        "ROUTE": "WAIT", "TOP_PICK": 0, "RR_NOW_TP1": 2.04,
        "켈리_수량": 0, "TIME_STOP_DAYS": 7, "POSITION_PCT": 100,
    }
    sd.render_stock_detail_v2_partial(row, rank=126, total=323)
