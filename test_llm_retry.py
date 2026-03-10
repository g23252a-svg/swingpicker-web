"""LLM Retry 테스트 — 429/RESOURCE_EXHAUSTED 방어"""
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
    print("🧪 LLM Retry 테스트 (429/RESOURCE_EXHAUSTED)")
    print("=" * 60)

    from llm_retry_utils import (
        _llm_call_with_retry, _is_retryable, _extract_retry_after,
    )
    import time

    # ═══ 1. _is_retryable 판정 ═══
    print("\n📐 1. 재시도 가능 에러 판정")

    # status_code 기반
    class Http429(Exception):
        status_code = 429
    class Http500(Exception):
        status_code = 500
    class Http400(Exception):
        status_code = 400
    class GrpcExhausted(Exception):
        grpc_status_code = "RESOURCE_EXHAUSTED"

    test("429 → retryable", _is_retryable(Http429()))
    test("503 → retryable", _is_retryable(type('E', (Exception,), {'status_code': 503})()))
    test("400 → NOT retryable", not _is_retryable(Http400()))
    test("500 → NOT retryable", not _is_retryable(Http500()))
    test("GRPC RESOURCE_EXHAUSTED → retryable", _is_retryable(GrpcExhausted()))
    test("문자열 '429' → retryable", _is_retryable(Exception("Error 429 Too Many Requests")))
    test("문자열 'quota' → retryable", _is_retryable(Exception("quota exceeded")))
    test("일반 에러 → NOT retryable", not _is_retryable(Exception("some random error")))
    test("ValueError → NOT retryable", not _is_retryable(ValueError("bad value")))

    # ═══ 2. _extract_retry_after ═══
    print("\n📐 2. Retry-After 추출")

    class WithRetryAfter(Exception):
        retry_after = 5.0
    class NoRetryAfter(Exception):
        pass

    test("retry_after 속성 → 5.0", _extract_retry_after(WithRetryAfter()) == 5.0)
    test("속성 없음 → None", _extract_retry_after(NoRetryAfter()) is None)
    test("일반 Exception → None", _extract_retry_after(Exception("test")) is None)

    # ═══ 3. (A) 429 두 번 후 성공 ═══
    print("\n📐 3. (A) 429 두 번 후 성공")
    call_count = [0]
    sleep_times = []

    original_sleep = time.sleep
    def mock_sleep(s):
        sleep_times.append(s)
        # 실제 sleep 안 함 (테스트 속도)

    time.sleep = mock_sleep
    try:
        def flaky_fn():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Http429("429 Too Many Requests")
            return "success"

        result = _llm_call_with_retry(flaky_fn, max_retries=3, base_delay=2.0, cap=30.0)
        test("최종 성공", result == "success")
        test("총 호출 3회", call_count[0] == 3, f"got {call_count[0]}")
        test("sleep 2회 호출", len(sleep_times) == 2, f"got {len(sleep_times)}")
        # exponential: 1차=2*2^0*jitter, 2차=2*2^1*jitter
        test("1차 sleep: 1~3초 범위", 0.5 < sleep_times[0] < 4.0,
             f"got {sleep_times[0]:.2f}")
        test("2차 sleep > 1차", sleep_times[1] > sleep_times[0] * 0.5,
             f"1st={sleep_times[0]:.2f}, 2nd={sleep_times[1]:.2f}")

    finally:
        time.sleep = original_sleep

    # ═══ 4. (B) 429 연속 max_retries 초과 ═══
    print("\n📐 4. (B) max_retries 초과 → raise")
    call_count2 = [0]
    sleep_times2 = []

    time.sleep = lambda s: sleep_times2.append(s)
    try:
        def always_429():
            call_count2[0] += 1
            raise Http429("429 always")

        raised = False
        try:
            _llm_call_with_retry(always_429, max_retries=2, base_delay=1.0)
        except Http429:
            raised = True

        test("max_retries 초과 → raise", raised)
        test("총 호출 3회 (initial + 2 retry)", call_count2[0] == 3, f"got {call_count2[0]}")
        test("sleep 2회", len(sleep_times2) == 2)

    finally:
        time.sleep = original_sleep

    # ═══ 5. (C) 비재시도 에러 즉시 raise ═══
    print("\n📐 5. (C) 비재시도 에러 → 즉시 raise")
    call_count3 = [0]

    def bad_request():
        call_count3[0] += 1
        raise Http400("Bad Request")

    raised_400 = False
    try:
        _llm_call_with_retry(bad_request, max_retries=3)
    except Http400:
        raised_400 = True

    test("400 → 즉시 raise (재시도 없음)", raised_400)
    test("호출 1회만", call_count3[0] == 1, f"got {call_count3[0]}")

    # ═══ 6. Retry-After 우선 ═══
    print("\n📐 6. Retry-After 우선 적용")
    ra_sleeps = []
    time.sleep = lambda s: ra_sleeps.append(s)
    call_count4 = [0]

    try:
        def retry_after_fn():
            call_count4[0] += 1
            if call_count4[0] <= 1:
                e = Http429("429")
                e.retry_after = 7.0
                raise e
            return "ok"

        result4 = _llm_call_with_retry(retry_after_fn, max_retries=2, base_delay=2.0, cap=30.0)
        test("Retry-After 성공", result4 == "ok")
        test("sleep = Retry-After(7.0)", abs(ra_sleeps[0] - 7.0) < 0.1,
             f"got {ra_sleeps[0]:.2f}")

    finally:
        time.sleep = original_sleep

    # ═══ 7. cap 제한 ═══
    print("\n📐 7. cap 제한")
    cap_sleeps = []
    time.sleep = lambda s: cap_sleeps.append(s)
    call_count5 = [0]

    try:
        def high_backoff():
            call_count5[0] += 1
            if call_count5[0] <= 1:
                raise Http429("429")
            return "ok"

        _llm_call_with_retry(high_backoff, max_retries=2, base_delay=100.0, cap=5.0)
        test("cap=5: sleep ≤ 7.5 (5*1.5 jitter max)", cap_sleeps[0] <= 7.5,
             f"got {cap_sleeps[0]:.2f}")

    finally:
        time.sleep = original_sleep

    # ═══ 8. 총 대기시간 상한 (total_timeout) ═══
    print("\n📐 8. 총 대기시간 상한")
    timeout_sleeps = []
    fake_clock = [0.0]  # 가짜 시계

    def mock_sleep_tick(s):
        timeout_sleeps.append(s)
        fake_clock[0] += s  # sleep한 만큼 시계 진행

    original_monotonic = time.monotonic
    time.sleep = mock_sleep_tick
    time.monotonic = lambda: fake_clock[0]
    call_count6 = [0]

    try:
        def always_429_slow():
            call_count6[0] += 1
            raise Http429("429")

        raised_timeout = False
        try:
            _llm_call_with_retry(always_429_slow, max_retries=10,
                                 base_delay=2.0, cap=30.0, total_timeout=5.0)
        except Http429:
            raised_timeout = True

        test("total_timeout=5s → 결국 raise", raised_timeout)
        total_slept = sum(timeout_sleeps)
        test("총 sleep ≤ total_timeout(5s)",
             total_slept <= 6.0,
             f"total_slept={total_slept:.2f}")
        test("max_retries 전에 timeout으로 중단", call_count6[0] < 11,
             f"calls={call_count6[0]}")

    finally:
        time.sleep = original_sleep
        time.monotonic = original_monotonic

    # ═══ 9. 스모크: 정상 호출 1회 → 즉시 리턴 ═══
    print("\n📐 9. 스모크(canary) 정상 호출")
    smoke_sleeps = []
    time.sleep = lambda s: smoke_sleeps.append(s)

    try:
        result_smoke = _llm_call_with_retry(lambda: "healthy", max_retries=3)
        test("정상 호출 → 즉시 성공", result_smoke == "healthy")
        test("sleep 0회", len(smoke_sleeps) == 0)
    finally:
        time.sleep = original_sleep

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
