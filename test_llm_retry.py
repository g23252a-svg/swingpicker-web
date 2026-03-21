# -*- coding: utf-8 -*-
"""test_llm_retry.py — LLM Retry 테스트 (pytest 표준)
═══════════════════════════════════════════════════
429/RESOURCE_EXHAUSTED 방어 검증
실행: pytest test_llm_retry.py -v
"""
import time

import pytest

from llm_retry_utils import (
    _llm_call_with_retry, _is_retryable, _extract_retry_after,
)


# ── 테스트용 예외 클래스 ──
class Http429(Exception):
    status_code = 429

class Http500(Exception):
    status_code = 500

class Http400(Exception):
    status_code = 400

class GrpcExhausted(Exception):
    grpc_status_code = "RESOURCE_EXHAUSTED"

class WithRetryAfter(Exception):
    retry_after = 5.0


# ═══════════════════════════════════════════════
#  1. _is_retryable 판정
# ═══════════════════════════════════════════════
class TestIsRetryable:
    def test_429_retryable(self):
        assert _is_retryable(Http429())

    def test_503_retryable(self):
        assert _is_retryable(type('E', (Exception,), {'status_code': 503})())

    def test_400_not_retryable(self):
        assert not _is_retryable(Http400())

    def test_500_not_retryable(self):
        assert not _is_retryable(Http500())

    def test_grpc_exhausted_retryable(self):
        assert _is_retryable(GrpcExhausted())

    def test_string_429_retryable(self):
        assert _is_retryable(Exception("Error 429 Too Many Requests"))

    def test_string_quota_retryable(self):
        assert _is_retryable(Exception("quota exceeded"))

    def test_generic_error_not_retryable(self):
        assert not _is_retryable(Exception("some random error"))

    def test_value_error_not_retryable(self):
        assert not _is_retryable(ValueError("bad value"))


# ═══════════════════════════════════════════════
#  2. _extract_retry_after
# ═══════════════════════════════════════════════
class TestExtractRetryAfter:
    def test_retry_after_attribute(self):
        assert _extract_retry_after(WithRetryAfter()) == 5.0

    def test_no_attribute_returns_none(self):
        assert _extract_retry_after(Exception()) is None

    def test_generic_exception_returns_none(self):
        assert _extract_retry_after(Exception("test")) is None


# ═══════════════════════════════════════════════
#  3. 429 두 번 후 성공
# ═══════════════════════════════════════════════
class TestRetrySuccess:
    def test_429_then_success(self, monkeypatch):
        sleep_times = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_times.append(s))
        call_count = [0]

        def flaky_fn():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Http429("429 Too Many Requests")
            return "success"

        result = _llm_call_with_retry(flaky_fn, max_retries=3, base_delay=2.0, cap=30.0)
        assert result == "success"
        assert call_count[0] == 3
        assert len(sleep_times) == 2
        assert 0.5 < sleep_times[0] < 4.0
        assert sleep_times[1] > sleep_times[0] * 0.5


# ═══════════════════════════════════════════════
#  4. max_retries 초과 → raise
# ═══════════════════════════════════════════════
class TestRetryExhausted:
    def test_always_429_raises(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        call_count = [0]

        def always_429():
            call_count[0] += 1
            raise Http429("429 always")

        with pytest.raises(Http429):
            _llm_call_with_retry(always_429, max_retries=2, base_delay=1.0)
        assert call_count[0] == 3  # initial + 2 retry


# ═══════════════════════════════════════════════
#  5. 비재시도 에러 → 즉시 raise
# ═══════════════════════════════════════════════
class TestNonRetryableError:
    def test_400_raises_immediately(self):
        call_count = [0]

        def bad_request():
            call_count[0] += 1
            raise Http400("Bad Request")

        with pytest.raises(Http400):
            _llm_call_with_retry(bad_request, max_retries=3)
        assert call_count[0] == 1


# ═══════════════════════════════════════════════
#  6. Retry-After 우선 적용
# ═══════════════════════════════════════════════
class TestRetryAfterHeader:
    def test_retry_after_honored(self, monkeypatch):
        ra_sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: ra_sleeps.append(s))
        call_count = [0]

        def retry_after_fn():
            call_count[0] += 1
            if call_count[0] <= 1:
                e = Http429("429")
                e.retry_after = 7.0
                raise e
            return "ok"

        result = _llm_call_with_retry(retry_after_fn, max_retries=2, base_delay=2.0, cap=30.0)
        assert result == "ok"
        assert abs(ra_sleeps[0] - 7.0) < 0.1


# ═══════════════════════════════════════════════
#  7. cap 제한
# ═══════════════════════════════════════════════
class TestCap:
    def test_sleep_respects_cap(self, monkeypatch):
        cap_sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: cap_sleeps.append(s))
        call_count = [0]

        def high_backoff():
            call_count[0] += 1
            if call_count[0] <= 1:
                raise Http429("429")
            return "ok"

        _llm_call_with_retry(high_backoff, max_retries=2, base_delay=100.0, cap=5.0)
        assert cap_sleeps[0] <= 7.5  # 5*1.5 jitter max


# ═══════════════════════════════════════════════
#  8. 총 대기시간 상한 (total_timeout)
# ═══════════════════════════════════════════════
class TestTotalTimeout:
    def test_total_timeout_stops_retries(self, monkeypatch):
        timeout_sleeps = []
        fake_clock = [0.0]

        def mock_sleep_tick(s):
            timeout_sleeps.append(s)
            fake_clock[0] += s

        monkeypatch.setattr(time, "sleep", mock_sleep_tick)
        monkeypatch.setattr(time, "monotonic", lambda: fake_clock[0])
        call_count = [0]

        def always_429_slow():
            call_count[0] += 1
            raise Http429("429")

        with pytest.raises(Http429):
            _llm_call_with_retry(always_429_slow, max_retries=10,
                                 base_delay=2.0, cap=30.0, total_timeout=5.0)
        assert sum(timeout_sleeps) <= 6.0
        assert call_count[0] < 11


# ═══════════════════════════════════════════════
#  9. 정상 호출 → 즉시 리턴
# ═══════════════════════════════════════════════
class TestSmokeNormal:
    def test_healthy_call_no_sleep(self, monkeypatch):
        smoke_sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: smoke_sleeps.append(s))
        result = _llm_call_with_retry(lambda: "healthy", max_retries=3)
        assert result == "healthy"
        assert len(smoke_sleeps) == 0
