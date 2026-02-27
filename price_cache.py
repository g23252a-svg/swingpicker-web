# -*- coding: utf-8 -*-
"""
price_cache.py — Circuit Breaker + In-Memory Price Cache (v1.0)
═══════════════════════════════════════════════════════════════
- TTL 5분 캐시: API 실패 시 Last Known Price 반환
- Circuit Breaker: 3회 연속 실패 시 30초 차단
- aiohttp 비동기 현재가 조회 (asyncio.gather 병렬화)
"""

import time
import logging
import asyncio
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger("price_cache")

# ── 캐시 구조 ──────────────────────────────────────
# { code: (price, timestamp) }
_CACHE: Dict[str, Tuple[int, float]] = {}
CACHE_TTL = 300  # 5분

# ── Circuit Breaker ──────────────────────────────
# { code: {"failures": int, "blocked_until": float} }
_CB_STATE: Dict[str, dict] = {}
CB_FAIL_THRESHOLD = 3    # 3회 연속 실패
CB_COOLDOWN_SEC   = 30   # 30초 차단


# ────────────────────────────────────────────────
def get_cached(code: str) -> Optional[int]:
    """TTL 이내 캐시 반환, 만료·없으면 None"""
    entry = _CACHE.get(code)
    if entry and (time.time() - entry[1]) < CACHE_TTL:
        return entry[0]
    return None


def set_cache(code: str, price: int):
    """가격 캐시 갱신 + 주기적 만료 GC"""
    _CACHE[code] = (price, time.time())
    # ✅ [Fix #2] 100회 쓸 때마다 만료 키 일괄 삭제 (메모리 누수 방지)
    # 한국 주식 ~2,500종목 × 캐시 항목 크기 ≈ 무시 가능하나, 장기 운영 시 방어
    if len(_CACHE) % 100 == 0:
        _gc_expired()


def _gc_expired():
    """TTL 만료 캐시 항목 + 오래된 Circuit Breaker 상태 일괄 삭제"""
    now = time.time()
    expired_keys = [k for k, (_, ts) in _CACHE.items() if now - ts >= CACHE_TTL]
    for k in expired_keys:
        _CACHE.pop(k, None)

    # CB_STATE: 차단 해제 후 실패 횟수도 0인 항목은 제거
    dead_cb = [
        k for k, cb in _CB_STATE.items()
        if cb.get("failures", 0) == 0 and now >= cb.get("blocked_until", 0)
    ]
    for k in dead_cb:
        _CB_STATE.pop(k, None)

    if expired_keys or dead_cb:
        logger.debug(f"🧹 GC: 캐시 {len(expired_keys)}건, CB {len(dead_cb)}건 정리")


def is_circuit_open(code: str) -> bool:
    """Circuit Breaker가 열려(차단) 있으면 True"""
    cb = _CB_STATE.get(code)
    if not cb:
        return False
    if time.time() < cb.get("blocked_until", 0):
        return True
    return False


def record_failure(code: str):
    """실패 기록 — 임계치 도달 시 Circuit Breaker 발동"""
    cb = _CB_STATE.setdefault(code, {"failures": 0, "blocked_until": 0})
    cb["failures"] += 1
    if cb["failures"] >= CB_FAIL_THRESHOLD:
        cb["blocked_until"] = time.time() + CB_COOLDOWN_SEC
        logger.warning(f"🔴 Circuit Breaker 발동: {code} ({CB_COOLDOWN_SEC}초 차단)")


def record_success(code: str):
    """성공 시 실패 카운터 초기화"""
    if code in _CB_STATE:
        _CB_STATE[code] = {"failures": 0, "blocked_until": 0}


# ────────────────────────────────────────────────
def fetch_with_cache(code: str, name: str, fdr_func) -> Tuple[str, str, int]:
    """
    캐시 → Circuit Breaker → API 순으로 조회.
    fdr_func(code) → int 현재가 (0이면 실패)
    """
    cached = get_cached(code)
    if cached is not None:
        return code, name, cached

    if is_circuit_open(code):
        last = _CACHE.get(code, (0, 0))[0]
        logger.debug(f"⚡ Circuit open, using last known: {code}={last}")
        return code, name, last

    try:
        price = fdr_func(code)
        if price and price > 0:
            set_cache(code, price)
            record_success(code)
            return code, name, price
        else:
            record_failure(code)
            return code, name, 0
    except Exception as e:
        logger.warning(f"API 오류 {code}: {e}")
        record_failure(code)
        last = _CACHE.get(code, (0, 0))[0]
        return code, name, last


# ────────────────────────────────────────────────
async def fetch_prices_async(targets: list, fdr_module) -> Dict[str, int]:
    """
    asyncio.gather로 여러 종목 현재가 비동기 병렬 조회.
    targets: [(code, name), ...]
    Returns: {code: price}
    """
    loop = asyncio.get_event_loop()

    def _fdr_price(code: str) -> int:
        """FDR 동기 호출 래퍼"""
        try:
            from datetime import datetime, timedelta
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            d = fdr_module.DataReader(str(code).zfill(6), start)
            if d is not None and not d.empty:
                return int(d.iloc[-1]["Close"])
        except Exception:
            pass
        return 0

    async def _fetch_one(code: str, name: str) -> Tuple[str, int]:
        # 캐시 먼저
        cached = get_cached(code)
        if cached is not None:
            return code, cached
        if is_circuit_open(code):
            return code, _CACHE.get(code, (0, 0))[0]
        # FDR을 thread pool에서 실행 (blocking I/O 비동기화)
        try:
            price = await loop.run_in_executor(None, _fdr_price, code)
            if price > 0:
                set_cache(code, price)
                record_success(code)
            else:
                record_failure(code)
            return code, price
        except Exception as e:
            record_failure(code)
            return code, _CACHE.get(code, (0, 0))[0]

    tasks = [_fetch_one(code, name) for code, name in targets]
    results = await asyncio.gather(*tasks)
    return dict(results)


# ────────────────────────────────────────────────
def cache_stats() -> dict:
    """디버그용 캐시 상태 요약"""
    now = time.time()
    valid   = sum(1 for p, ts in _CACHE.values() if now - ts < CACHE_TTL)
    expired = len(_CACHE) - valid
    open_cb = sum(1 for cb in _CB_STATE.values() if now < cb.get("blocked_until", 0))
    return {
        "total_cached":  len(_CACHE),
        "valid_entries": valid,
        "expired_gc_pending": expired,   # ✅ 다음 GC에서 지워질 항목
        "open_circuits": open_cb,
    }
