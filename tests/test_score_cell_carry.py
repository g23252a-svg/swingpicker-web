# -*- coding: utf-8 -*-
"""[v22.3.21] 테이블 '점수' 셀 — CARRY 과차감 모순 방지 테스트.

CARRY 보유종목은 STALE/legacy penalty 누적으로 DISPLAY_SCORE가 0~한자리까지 떨어져
'S 98인데 점수 0' 모순이 생긴다. _nb_score_cell은 그 경우 '보유'로 표시한다.
비-CARRY 또는 정상 차감이면 DISPLAY_SCORE를 그대로 보여준다.
NiceGUI 미설치 sandbox에서는 함수만 떼어 테스트(import 우회)."""
import sys, os, re, glob
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tab_stocks 전체 import는 nicegui를 끌어오므로, 두 순수함수만 추출해 테스트
_SRC = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "components", "tab_stocks.py"), encoding="utf-8").read()
_ns = {"pd": pd}
exec(re.search(r"def _nz\(.*?\n(?:    .*\n|\n)*", _SRC).group(0), _ns)
exec(re.search(r"def _nb_score_cell\(.*?\n(?:    .*\n|\n)*", _SRC).group(0), _ns)
_cell = _ns["_nb_score_cell"]


def test_carry_overpenalized_shows_holding():
    # FINAL 80인데 DISPLAY 0 (과차감) → '보유'
    assert _cell({"ROUTE": "CARRY", "FINAL_SCORE": 80.1, "DISPLAY_SCORE": 0}) == "보유"


def test_carry_normal_keeps_display():
    # CARRY지만 괴리 작으면(≤15) DISPLAY 그대로
    assert _cell({"ROUTE": "CARRY", "FINAL_SCORE": 62, "DISPLAY_SCORE": 55}) == "55"


def test_non_carry_always_keeps_display():
    # 비-CARRY는 괴리가 커도 DISPLAY 그대로 (CARRY 한정 규칙)
    assert _cell({"ROUTE": "WAIT", "FINAL_SCORE": 80, "DISPLAY_SCORE": 0}) == "0"
    assert _cell({"ROUTE": "ATTACK", "FINAL_SCORE": 90, "DISPLAY_SCORE": 88}) == "88"


def test_boundary_15_points():
    # 괴리 정확히 15면 그대로(>15만 보유), 15.1이면 보유
    assert _cell({"ROUTE": "CARRY", "FINAL_SCORE": 65, "DISPLAY_SCORE": 50}) == "50"   # 15 → 그대로
    assert _cell({"ROUTE": "CARRY", "FINAL_SCORE": 65.2, "DISPLAY_SCORE": 50}) == "보유"  # 15.2 → 보유


def test_missing_final_falls_back_to_display():
    # FINAL 없으면 DISPLAY 기준(괴리 0) → 그대로
    assert _cell({"ROUTE": "CARRY", "DISPLAY_SCORE": 42}) == "42"


def test_real_csv_carry_zero_scores_become_holding():
    files = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "recommend_2026*.csv")))
    if not files:
        import pytest; pytest.skip("recommend CSV 없음")
    df = pd.read_csv(files[-1], dtype={"종목코드": str})
    carry0 = df[(df["ROUTE"] == "CARRY")
                & (pd.to_numeric(df["FINAL_SCORE"], errors="coerce") >= 50)
                & (pd.to_numeric(df["DISPLAY_SCORE"], errors="coerce") == 0)]
    # FINAL≥50인데 DISPLAY=0인 CARRY는 전부 '보유'로
    for _, r in carry0.iterrows():
        assert _cell(r.to_dict()) == "보유", f"{r['종목명']} 보유표시 실패"
    # 비-CARRY는 '보유' 절대 안 나옴
    nonc = df[df["ROUTE"] != "CARRY"]
    assert not any(_cell(r.to_dict()) == "보유" for _, r in nonc.iterrows())
