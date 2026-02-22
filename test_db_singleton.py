"""DB 싱글톤 + 동시성 + TTL 테스트"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0

def test(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def run():
    global PASS, FAIL
    PASS, FAIL = 0, 0

    print("=" * 60)
    print("🧪 DB 싱글톤 + 동시성 + TTL 테스트")
    print("=" * 60)

    # duckdb mock (설치 안 된 환경용)
    import types
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

    import db_utils
    from db_utils import get_db, _reset_db_singleton, LDYDBManager

    # ═══ 1. 싱글톤 동일 인스턴스 ═══
    print("\n📐 1. 싱글톤 동일 인스턴스")
    _reset_db_singleton()

    db1 = get_db()
    db2 = get_db()
    test("get_db() 2회 → 동일 인스턴스", db1 is db2)
    test("LDYDBManager 타입", isinstance(db1, LDYDBManager))

    db3 = get_db()
    db4 = get_db()
    db5 = get_db()
    test("5회 호출 → 전부 동일", db1 is db3 is db4 is db5)

    # ═══ 2. _conn_lock 존재 ═══
    print("\n📐 2. 쿼리 직렬화 락")
    test("_conn_lock 존재", hasattr(db1, '_conn_lock'))
    test("execute_safe 메서드", hasattr(db1, 'execute_safe'))

    # execute_safe 동작 확인
    try:
        result = db1.execute_safe("SELECT 1 as v")
        row = result.fetchone()
        test("execute_safe 동작", row[0] == 1)
    except Exception as e:
        test("execute_safe 동작", False, str(e))

    # ═══ 3. 동시 호출 → 레이스 없음 ═══
    print("\n📐 3. 동시 호출 안전성")
    import threading

    _reset_db_singleton()
    instances = []
    errors = []

    def get_db_thread():
        try:
            db = get_db()
            instances.append(id(db))
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=get_db_thread) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    test("10 스레드 에러 없음", len(errors) == 0,
         f"errors={errors[:3]}")
    test("10 스레드 동일 인스턴스", len(set(instances)) == 1,
         f"unique={len(set(instances))}")

    # ═══ 4. TTL 갱신 ═══
    print("\n📐 4. TTL 기반 갱신")
    _reset_db_singleton()
    import time as _time

    db_a = get_db()
    init_time_before = db_utils._db_init_time

    # TTL을 1초로 줄여서 테스트
    original_ttl = db_utils._DB_TTL_SECONDS
    db_utils._DB_TTL_SECONDS = 0.1  # 0.1초

    # 가짜로 시간 경과 시뮬레이션
    original_monotonic = _time.monotonic
    fake_offset = [0.0]
    _time.monotonic = lambda: original_monotonic() + fake_offset[0]

    try:
        # TTL 전: 같은 인스턴스
        db_b = get_db()
        test("TTL 전: 동일 인스턴스", db_a is db_b)

        # TTL 경과
        fake_offset[0] = 1.0  # 1초 경과 (TTL 0.1초 초과)
        db_c = get_db()
        test("TTL 후: 여전히 동일 인스턴스 (연결 유지)", db_a is db_c)
        test("TTL 후: init_time 갱신됨",
             db_utils._db_init_time > init_time_before)

    finally:
        db_utils._DB_TTL_SECONDS = original_ttl
        _time.monotonic = original_monotonic

    # ═══ 5. force_refresh ═══
    print("\n📐 5. force_refresh")
    db_old = get_db()
    db_new = get_db(force_refresh=True)
    test("force_refresh → 새 인스턴스", db_old is not db_new)
    test("force_refresh 후 get_db → 새 인스턴스", get_db() is db_new)

    # ═══ 6. collector에서 LDYDBManager() 직접 호출 없음 ═══
    print("\n📐 6. collector SSOT 잠금")
    collector_src = open(os.path.join(os.path.dirname(__file__), "collector.py")).read()
    direct_calls = collector_src.count("LDYDBManager()")
    test("collector에서 LDYDBManager() 직접 호출 0건",
         direct_calls == 0,
         f"found {direct_calls} direct calls")

    # get_db 사용 확인
    get_db_calls = collector_src.count("get_db()")
    test("collector에서 get_db() 사용", get_db_calls >= 2,
         f"found {get_db_calls}")

    # cleanup
    _reset_db_singleton()

    # ── 결과 ──
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"🏁 결과: {PASS}/{total} 통과 ({FAIL} 실패)")
    if FAIL > 0:
        print("⚠️ 실패 항목이 있습니다!")
        sys.exit(1)
    else:
        print("🏆 ALL PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run()
