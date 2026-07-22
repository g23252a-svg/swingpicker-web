# -*- coding: utf-8 -*-
"""[v37.2] 죽은 AI 축 표시 제거 회귀 방지.

배경: 구 LSTM/XGB(v19) 모델은 OOS 검증 기록이 없어 v28-4 정직 게이트가
AI_SCORE를 항상 0으로 처리(ML_STATUS=UNVALIDATED_MODEL). 0점 고정 컬럼이
남아 "AI점수 왜 항상 0이냐"는 혼란 + min(S,T,AI) 정렬 무력화를 유발했다.
검증 통과한 예측 점수는 🧠알파 컬럼이 담당한다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "components" / "tab_stocks.py").read_text(encoding="utf-8")


def test_ai_table_column_removed():
    assert '"name": "ai"' not in SRC
    assert '"ai":    f\'{_nz(r.get("AI_SCORE"' not in SRC


def test_alpha_column_still_present():
    # AI 제거가 알파(검증된 예측 점수) 컬럼까지 지우면 안 된다.
    assert '"name": "alpha"' in SRC


def test_axis_min_sort_excludes_dead_ai():
    # AI=0 고정이면 min(S,T,AI)=0 전 종목 동률 — 정렬 무력화 재발 방지.
    assert "3축최저순" not in SRC
    assert "S·T최저순" in SRC
    import re
    m = re.search(r'elif sort_mode\.value == "🧱 S·T최저순":\n(.*?)elif ', SRC, re.S)
    assert m and "AI_SCORE" not in m.group(1)


def test_ai_sublabels_removed():
    # 카드/상세 서브라벨의 "AI{...}" 표기 제거 확인.
    assert "AI{a_v" not in SRC
    assert "AI{ai_val" not in SRC
    assert '("AI", ai_val)' not in SRC
