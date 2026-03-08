# -*- coding: utf-8 -*-
"""
llm_retry_utils.py — LLM API 재시도 유틸리티
═══════════════════════════════════════════════════
[v1.0] news_engine.py에서 분리 — 독립 모듈

news_engine, dart_analyzer 등 모든 LLM 호출부에서 공유.
순환 의존 방지 목적.

사용법:
    from llm_retry_utils import _llm_call_with_retry, _is_retryable
"""

import logging
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """에러 객체에서 Retry-After 힌트 추출 (있으면 우선 사용)"""
    # google.api_core.exceptions 스타일
    if hasattr(exc, 'retry_after'):
        return float(exc.retry_after)
    # HTTP response 헤더 스타일
    if hasattr(exc, 'response') and hasattr(exc.response, 'headers'):
        ra = exc.response.headers.get('Retry-After')
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
    return None


def _is_retryable(exc: Exception) -> bool:
    """재시도 가능한 에러인지 판정 (status code 우선, 문자열 fallback)"""
    # status_code 속성 (google, requests 등)
    status = getattr(exc, 'code', None) or getattr(exc, 'status_code', None)
    if status is not None:
        try:
            code = int(status)
            return code in (429, 503, 529)
        except (ValueError, TypeError):
            pass
    # grpc status
    if hasattr(exc, 'grpc_status_code'):
        grpc_code = str(exc.grpc_status_code)
        if 'RESOURCE_EXHAUSTED' in grpc_code or 'UNAVAILABLE' in grpc_code:
            return True
    # 문자열 fallback (최후 수단)
    msg = str(exc)
    return any(kw in msg for kw in ('429', 'RESOURCE_EXHAUSTED', 'quota', 'rate limit'))


def _llm_call_with_retry(
    call_fn,
    max_retries: int = 3,
    base_delay: float = 2.0,
    cap: float = 30.0,
    total_timeout: float = 60.0,
):
    """
    LLM API 호출 + exponential backoff + jitter + Retry-After.
    - 429/RESOURCE_EXHAUSTED → 재시도
    - 400/500 등 비재시도 에러 → 즉시 raise
    - total_timeout: 전체 대기시간 상한 (초). 초과 시 마지막 에러 raise.
    """
    t_start = time.monotonic()

    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except Exception as e:
            if not _is_retryable(e):
                raise  # 비재시도 에러 → 즉시 전파

            if attempt >= max_retries:
                logger.warning(f"LLM 재시도 한도 초과 ({max_retries}회): {e}")
                raise

            # 총 대기시간 상한 체크
            elapsed = time.monotonic() - t_start
            remaining = total_timeout - elapsed
            if remaining <= 0:
                logger.warning(f"LLM 총 대기시간 상한 초과 ({total_timeout}s): {e}")
                raise

            # Retry-After 우선, 없으면 exponential backoff + jitter
            retry_after = _extract_retry_after(e)
            if retry_after and retry_after > 0:
                wait = min(retry_after, cap)
            else:
                wait = min(cap, base_delay * (2 ** attempt))
                wait *= (0.5 + random.random())  # jitter: 0.5x ~ 1.5x

            # remaining으로 한번 더 클램프
            wait = min(wait, remaining)

            logger.info(
                f"LLM 429/재시도 {attempt+1}/{max_retries}: "
                f"wait={wait:.1f}s, elapsed={elapsed:.1f}s/{total_timeout}s"
            )
            time.sleep(wait)
