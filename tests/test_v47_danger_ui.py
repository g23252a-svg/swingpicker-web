# -*- coding: utf-8 -*-
"""[v47] 위험구간 UI 노출 — '관망'과 '정리 검토'를 구분한다.

기존 판정은 게이트 탈락을 전부 회색 '관망'으로 뭉갰다. 보유자에게 '관망'과
'정리 검토'는 전혀 다른 지시인데, 실측 5일 -2.88%·승률 21.5%(20일 10.8%)인
구간이 '관망'으로 보이면 방치하게 된다. 위험구간을 최우선 판정으로 올린다.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import components.stock_detail_v2 as sd


def test_danger_zone_outranks_all_other_verdicts():
    """위험구간은 과열·관망·최고등급보다 우선한다."""
    # 다른 조건이 최상(TOP_PICK·수량·높은RR)이어도 위험구간이면 위험 판정
    label, color, icon = sd._verdict_grade(
        top_pick=True, qty=10, rr=2.0, is_overheat=False, danger_zone=True)
    assert icon == "⛔"
    assert "정리" in label
    # 과열보다도 우선
    label2, _, icon2 = sd._verdict_grade(
        top_pick=False, qty=0, rr=0.5, is_overheat=True, danger_zone=True)
    assert icon2 == "⛔", "과열이 위험구간을 덮어씀"


def test_default_preserves_legacy_verdicts():
    """danger_zone 미지정 시 기존 판정이 그대로여야 한다(회귀 방지)."""
    assert sd._verdict_grade(False, 0, 0.5, True)[2] == "⚠️"
    assert sd._verdict_grade(False, 0, 1.5, False)[2] == "—"
    assert sd._verdict_grade(True, 10, 1.5, False)[2] == "👑"
    assert sd._verdict_grade(True, 10, 1.1, False)[2] == "✓"


def test_danger_is_visually_distinct_from_watch():
    """관망(회색)과 위험구간(적색)이 같은 색이면 구분 실패."""
    _, danger_color, _ = sd._verdict_grade(False, 0, 1.0, False, danger_zone=True)
    _, watch_color, _ = sd._verdict_grade(False, 0, 1.0, False, danger_zone=False)
    assert danger_color != watch_color
    assert "#6B7280" not in danger_color, "위험구간이 관망 회색을 씀"


def test_row_mapping_exposes_danger_fields():
    """CSV 행 → 상세 dict 매핑에 위험구간 필드가 실려야 한다."""
    import inspect
    src = inspect.getsource(sd)
    assert '"danger_zone": _as_bool(row.get("DANGER_ZONE", 0))' in src
    assert 'DANGER_ZONE_REASON' in src


def test_hero_uses_danger_reason_text():
    src = (ROOT / "components" / "stock_detail_v2.py").read_text(encoding="utf-8")
    # 히어로가 위험구간을 최우선 분기로 처리하고 사유를 노출하는지
    assert "if danger_zone:" in src
    assert 'n.get("danger_reason"' in src
