# -*- coding: utf-8 -*-
"""test_db_singleton.py — DB 싱글톤 + 동시성 + TTL 테스트 (pytest 표준)
═══════════════════════════════════════════════════
실행: pytest test_db_singleton.py -v
"""
import os
import threading
import time as _time

import pytest

import db_utils
from db_utils import get_db, _reset_db_singleton, LDYDBManager


# ═══════════════════════════════════════════════
#  1. 싱글톤 동일 인스턴스
# ═══════════════════════════════════════════════
class TestSingleton:
    def setup_method(self):
        _reset_db_singleton()

    def test_two_calls_same_instance(self):
        db1 = get_db()
        db2 = get_db()
        assert db1 is db2

    def test_instance_type(self):
        assert isinstance(get_db(), LDYDBManager)

    def test_five_calls_same_instance(self):
        dbs = [get_db() for _ in range(5)]
        assert all(d is dbs[0] for d in dbs)


# ═══════════════════════════════════════════════
#  2. 쿼리 직렬화 락
# ═══════════════════════════════════════════════
class TestQueryLock:
    def setup_method(self):
        _reset_db_singleton()

    def test_execute_safe_method(self):
        assert hasattr(get_db(), 'execute_safe')

    def test_execute_safe_works(self):
        result = get_db().execute_safe("SELECT 1 as v")
        row = result.fetchone()
        assert row[0] == 1


# ═══════════════════════════════════════════════
#  3. 동시 호출 안전성
# ═══════════════════════════════════════════════
class TestConcurrency:
    def test_ten_threads_same_instance(self):
        _reset_db_singleton()
        instances = []
        errors = []

        def get_db_thread():
            try:
                instances.append(id(get_db()))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=get_db_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"errors={errors[:3]}"
        assert len(set(instances)) == 1, f"unique={len(set(instances))}"


# ═══════════════════════════════════════════════
#  4. TTL 기반 갱신
# ═══════════════════════════════════════════════
class TestTTL:
    def test_ttl_refresh(self):
        _reset_db_singleton()
        db_a = get_db()
        init_time_before = db_utils._db_init_time

        original_ttl = db_utils._DB_TTL_SECONDS
        db_utils._DB_TTL_SECONDS = 0.1
        original_monotonic = _time.monotonic
        fake_offset = [0.0]
        _time.monotonic = lambda: original_monotonic() + fake_offset[0]

        try:
            # TTL 전: 같은 인스턴스
            db_b = get_db()
            assert db_a is db_b

            # TTL 경과
            fake_offset[0] = 1.0
            db_c = get_db()
            assert db_a is db_c, "TTL 후에도 동일 인스턴스 (연결 유지)"
            assert db_utils._db_init_time > init_time_before, "init_time 갱신됨"
        finally:
            db_utils._DB_TTL_SECONDS = original_ttl
            _time.monotonic = original_monotonic


# ═══════════════════════════════════════════════
#  5. force_refresh
# ═══════════════════════════════════════════════
class TestForceRefresh:
    def test_force_refresh_new_instance(self):
        _reset_db_singleton()
        db_old = get_db()
        db_new = get_db(force_refresh=True)
        assert db_old is not db_new

    def test_after_refresh_returns_new(self):
        _reset_db_singleton()
        get_db()
        db_new = get_db(force_refresh=True)
        assert get_db() is db_new


# ═══════════════════════════════════════════════
#  6. collector SSOT 잠금
# ═══════════════════════════════════════════════
class TestCollectorSSOT:
    @pytest.fixture(autouse=True)
    def _load_src(self):
        self.collector_src = open(
            os.path.join(os.path.dirname(__file__), "collector.py")
        ).read()

    def test_no_direct_ldy_db_manager_call(self):
        assert self.collector_src.count("LDYDBManager()") == 0

    def test_uses_get_db(self):
        assert self.collector_src.count("get_db()") >= 1

    def teardown_method(self):
        _reset_db_singleton()
