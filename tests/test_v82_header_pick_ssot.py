# -*- coding: utf-8 -*-
"""v82 — 종목탭 상단 '🏆 오늘의 실전 후보' 카드를 오늘탭 공식 매수와 같은 SSOT로.

사용자: "오늘탭과 종목탭이 추천이 안맞는데" (9/9 배치).
- 오늘탭: 공식 매수 카카오 (PRODUCTION_BUY=1, production_buy_mask)
- 종목탭 카드: '⚪ 오늘 실전 후보 없음' — pick_top1/pick_top3가 🛡️ 콤보·
  ✅ 즉시진입 **라벨**로만 뽑는데 2026년 배치 전부에서 그 라벨이 비어 있었다.
  바로 위 히어로 스트립은 같은 화면에서 "매수 후보 1종목"이라 적고 있었다.
"""
import os

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = open(os.path.join(ROOT, "components", "tab_stocks.py"),
           encoding="utf-8").read()


def _df(rows):
    base = {
        "종목코드": "000000", "종목명": "X", "ALPHA_SCORE": 80.0,
        "RR_NOW_TP1": 1.0, "ELITE_LABEL": "", "ELITE_RANK_SCORE": 50.0,
        "PRODUCTION_BUY": 0, "TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0,
        "ROUTE": "WAIT", "GAP_PCT": 0.0,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


class TestSourceWiring:
    def test_header_uses_contract_helper_not_label_pickers(self):
        i0 = SRC.index("def render_tab_stocks(")
        i1 = SRC.index('ui.label("🎯 AI & Quant 추천 종목")')
        blk = SRC[i0:i1]
        assert "top3_codes = header_pick_codes(df)" in blk
        assert "pick_top1(df)" not in blk and "pick_top3(df)" not in blk

    def test_card_receives_today_tab_watch_list(self):
        i0 = SRC.index("def render_tab_stocks(")
        blk = SRC[i0:]
        assert "build_decision_summary as _bds" in blk
        assert "watch_items=_watch_items" in blk


class TestHeaderPickCodes:
    def test_production_row_without_label_is_picked(self):
        """옛 경로(라벨)는 빈 목록, 새 경로(계약)는 공식 매수를 낸다."""
        from components.tab_stocks import header_pick_codes, pick_top3, pick_top1
        df = _df([{"종목코드": "035720", "종목명": "카카오",
                   "PRODUCTION_BUY": 1, "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1},
                  {"종목코드": "017960", "종목명": "한국카본", "ALPHA_SCORE": 95.0,
                   "RR_NOW_TP1": 2.5}])
        assert pick_top1(df) == [] and pick_top3(df) == [], "재현: 라벨 경로는 빈 목록"
        assert header_pick_codes(df) == ["035720"]

    def test_non_production_rows_never_enter_even_with_high_axis(self):
        from components.tab_stocks import header_pick_codes
        df = _df([{"종목코드": "000001", "종목명": "A", "ALPHA_SCORE": 99.0,
                   "RR_NOW_TP1": 3.0, "ELITE_LABEL": "🛡️ 콤보",
                   "ELITE_RANK_SCORE": 90.0, "BUY_NOW_ELIGIBLE": 1}])
        assert header_pick_codes(df) == []

    def test_order_is_engine_axis_not_alpha_alone(self):
        from components.tab_stocks import header_pick_codes
        # A: 알파 90 × RR 0.5 = 45 · B: 알파 80 × RR 2.0 = 160 → B 먼저
        df = _df([{"종목코드": "000001", "종목명": "A", "ALPHA_SCORE": 90.0,
                   "RR_NOW_TP1": 0.5, "PRODUCTION_BUY": 1},
                  {"종목코드": "000002", "종목명": "B", "ALPHA_SCORE": 80.0,
                   "RR_NOW_TP1": 2.0, "PRODUCTION_BUY": 1}])
        assert header_pick_codes(df) == ["000002", "000001"]

    def test_caps_at_three_and_dedupes(self):
        from components.tab_stocks import header_pick_codes
        rows = [{"종목코드": f"00000{i}", "종목명": f"S{i}", "PRODUCTION_BUY": 1,
                 "ALPHA_SCORE": 90.0 - i} for i in range(1, 6)]
        rows.append({"종목코드": "000001", "종목명": "S1dup", "PRODUCTION_BUY": 1,
                     "ALPHA_SCORE": 89.0})
        assert header_pick_codes(_df(rows)) == ["000001", "000002", "000003"]

    def test_legacy_csv_without_contract_falls_back_to_label_path(self):
        from components.tab_stocks import header_pick_codes, pick_top3
        df = _df([{"종목코드": "000009", "종목명": "L", "ELITE_LABEL": "🛡️ 콤보",
                   "ELITE_RANK_SCORE": 90.0}]).drop(
            columns=["PRODUCTION_BUY", "TOP_PICK", "BUY_NOW_ELIGIBLE"])
        assert pick_top3(df) == ["000009"]
        assert header_pick_codes(df) == ["000009"]

    def test_empty_inputs(self):
        from components.tab_stocks import header_pick_codes
        assert header_pick_codes(None) == []
        assert header_pick_codes(pd.DataFrame()) == []
        assert header_pick_codes(_df([])) == []


class TestSameAsDecisionCenter:
    """실배치 — 두 탭이 같은 종목을 같은 순서로 말한다."""

    @pytest.mark.parametrize("ymd", ["20260907", "20260909"])
    def test_real_batch_header_equals_today_tab_buys(self, ymd):
        path = os.path.join(ROOT, "data", f"recommend_{ymd}.csv")
        if not os.path.exists(path):
            pytest.skip(f"{path} 없음")
        from components.tab_stocks import header_pick_codes, compute_elite_labels
        from components.decision_center import build_decision_summary
        df = pd.read_csv(path, dtype={"종목코드": str})
        df["종목코드"] = df["종목코드"].str.zfill(6)
        df = compute_elite_labels(df)
        buys = [str(x["code"]).zfill(6) for x in build_decision_summary(df)["buys"]]
        assert buys, "이 배치는 공식 매수가 있는 날이어야 검정이 성립"
        assert header_pick_codes(df) == buys
