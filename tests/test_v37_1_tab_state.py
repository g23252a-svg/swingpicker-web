# -*- coding: utf-8 -*-
"""[v37.1] 종목탭 상태 영속화 + sticky CSS per-client 주입 회귀 방지.

배경(사용자 실사용 보고):
  · 모바일에서 상태 필터 변경 → 타 앱 전환 → 복귀 시 필터 전부 초기화
    (60초 유예 초과 시 페이지 리로드 — 위젯이 하드코딩 기본값이었음)
  · 테이블 제목/헤더 고정이 재접속 클라이언트에서 사라짐
    (모듈 단위 CSS 주입 플래그 — 프로세스당 첫 클라이언트만 CSS 수신)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "components" / "tab_stocks.py").read_text(encoding="utf-8")


def test_no_module_level_sticky_flag():
    # 프로세스 단위 플래그 부활 방지 — 두 번째 클라이언트부터 CSS 누락됐던 버그.
    assert "_STICKY_CSS_INJECTED = False" not in SRC
    assert "global _STICKY_CSS_INJECTED" not in SRC


def test_sticky_css_guard_is_per_client():
    assert "_ldy_sticky_css_done" in SRC
    assert "ui.context.client" in SRC


def test_filters_restored_from_user_storage():
    # 저장 키와 복원 헬퍼가 실제 위젯 생성에 연결돼 있는지.
    assert 'app.storage.user.get("stocks_tab_state"' in SRC
    assert 'app.storage.user["stocks_tab_state"]' in SRC
    for key in ['"route"', '"label"', '"risk"', '"sort"', '"table_mode"', '"view_mode"', '"search"']:
        assert key in SRC, f"{key} 영속화 누락"


def test_restore_validates_against_current_options():
    # 옵션 개편 후 저장값이 무효면 기본값으로 — KeyError/빈 화면 방지.
    m = re.search(r"def _restore\(.*?\n(.*?)\n\n", SRC, re.S)
    assert m and "if val in valid else default" in m.group(1)


def test_build_view_persists_state():
    # 모든 필터 변경은 _build_view를 타므로 여기서 저장돼야 리로드 후 복원됨.
    m = re.search(r"def _build_view\(\):\n(.*?)def ", SRC, re.S)
    assert m and "_persist_tab_state()" in m.group(1)


def test_mobile_corner_cell_zindex():
    # 헤더행×종목명열 교차 셀 — 세로/가로 스크롤 모두에서 가려지지 않게.
    assert "thead th:nth-child(6)" in SRC and "z-index: 3" in SRC
