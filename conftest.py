# -*- coding: utf-8 -*-
"""conftest.py — pytest 공용 설정 및 fixture
═══════════════════════════════════════════════════
[v20.8] 스크립트형 테스트 → pytest 표준화
"""
import sys
import os

# ── 수집 제외: pytest가 아닌 스크립트형 파일 / 외부 의존성 필요 ──
# ── 수집 제외: pytest가 아닌 CLI 스크립트 ──
collect_ignore_glob = [
    "test_shadow_analyze.py",          # argparse CLI 도구 (golden snapshot 비교용)
]

# 프로젝트 루트를 sys.path에 추가 (모든 테스트에서 import 가능)
sys.path.insert(0, os.path.dirname(__file__))

# streamlit mock (dashboard / auth 관련 테스트에서 필요)
import types
if 'streamlit' not in sys.modules:
    mock_st = types.ModuleType('streamlit')

    class _MockSecrets(dict):
        def get(self, k, d=None):
            return d
    mock_st.secrets = _MockSecrets()
    sys.modules['streamlit'] = mock_st

# duckdb mock (설치 안 된 환경용)
if 'duckdb' not in sys.modules:
    mock_duckdb = types.ModuleType('duckdb')

    class _MockConn:
        def execute(self, q, p=None):
            return self
        def fetchone(self):
            return (1,)
        def fetchall(self):
            return []
        def close(self):
            pass
    mock_duckdb.connect = lambda *a, **k: _MockConn()
    sys.modules['duckdb'] = mock_duckdb
