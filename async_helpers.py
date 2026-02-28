# async_helpers.py — NiceGUI 이벤트 루프 블로킹 방지 유틸리티 (v6.0)
# ═══════════════════════════════════════════════════════════════════
# 사용법:
#   from async_helpers import run_sync, run_cpu
#   df = await run_sync(pd.read_csv, "data.csv")
#   result = await run_cpu(heavy_function, arg1, arg2)
# ═══════════════════════════════════════════════════════════════════

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar, Callable

T = TypeVar("T")

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
