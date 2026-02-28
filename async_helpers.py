# async_helpers.py — NiceGUI 이벤트 루프 블로킹 방지 유틸리티 (v6.0)
# ═══════════════════════════════════════════════════════════════════
# 사용법:
#   from async_helpers import run_sync, run_cpu
#   df = await run_sync(pd.read_csv, "data.csv")
#   result = await run_cpu(heavy_function, arg1, arg2)
# ═══════════════════════════════════════════════════════════════════

import asyncio
import logging
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, Callable

T = TypeVar("T")
_logger = logging.getLogger("async_helpers")

# I/O 바운드: 네트워크, 파일, DB
_io_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nicegui-io")

# CPU 바운드: Pandas, 차트 렌더링
_cpu_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nicegui-cpu")


async def run_sync(func: Callable[..., T], *args, **kwargs) -> T:
    """동기 I/O 함수를 워커 스레드에서 실행 → 이벤트 루프 해방"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _io_pool,
        functools.partial(func, *args, **kwargs)
    )


async def run_cpu(func: Callable[..., T], *args, **kwargs) -> T:
    """CPU 바운드 함수를 별도 스레드에서 실행"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _cpu_pool,
        functools.partial(func, *args, **kwargs)
    )


# ═══════════════════════════════════════════
#  Graceful Shutdown — 재배포 시 데이터 꼬임 방지
# ═══════════════════════════════════════════

def shutdown_pools():
    """
    NiceGUI app.on_shutdown에서 호출.
    진행 중인 스레드 작업이 완료될 때까지 최대 10초 대기 후 종료.
    DB 커넥션도 함께 정리.
    """
    _logger.info("🛑 Graceful Shutdown 시작 — 스레드 풀 종료 대기...")

    _io_pool.shutdown(wait=True, cancel_futures=False)
    _logger.info("  ✅ I/O 풀 종료 완료")

    _cpu_pool.shutdown(wait=True, cancel_futures=False)
    _logger.info("  ✅ CPU 풀 종료 완료")

    # DB 커넥션 정리
    try:
        from db_utils import get_db, _gist_sync
        db = get_db()
        if db:
            # 남은 dirty 테이블 최종 flush
            with _gist_sync._lock:
                remaining = list(_gist_sync._dirty)
                _gist_sync._dirty.clear()
            for tbl in remaining:
                filename = "users_db.json" if tbl == "users" else "inquiries_db.json"
                _logger.info(f"  🔄 최종 Gist flush: {tbl}")
                db._do_gist_upload(tbl, filename)

            db.close()
            _logger.info("  ✅ DB 커넥션 종료 완료")
    except Exception as e:
        _logger.warning(f"  ⚠️ DB 종료 중 에러 (무시): {e}")

    _logger.info("🛑 Graceful Shutdown 완료")


def register_shutdown(nicegui_app):
    """
    main.py에서 한 줄로 등록:
        from async_helpers import register_shutdown
        register_shutdown(app)
    """
    nicegui_app.on_shutdown(shutdown_pools)
