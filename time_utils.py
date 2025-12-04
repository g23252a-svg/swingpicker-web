# -*- coding: utf-8 -*-
"""
time_utils.py
- 시간 관련 공통 유틸
- DB/로그는 UTC, 화면/로컬 표시는 KST 기준
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Union, Optional

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
    KST = ZoneInfo("Asia/Seoul")
except Exception:
    # 혹시 아주 구버전이면 일단 UTC로. (필요하면 pytz로 교체 가능)
    KST = timezone.utc


# ---------------------- 기본 now 함수 ---------------------- #
def now_utc() -> datetime:
    """
    UTC 기준 현재 시각 (tzinfo 포함)
    - DB 저장용, 로그 저장용
    """
    return datetime.now(timezone.utc)


def now_kst() -> datetime:
    """
    KST 기준 현재 시각 (tzinfo 포함)
    - 화면 표시, 디버깅용
    """
    return now_utc().astimezone(KST)


# ---------------------- 파싱 / 변환 ---------------------- #
DTLike = Union[datetime, str]


def parse_iso_dt(value: str) -> Optional[datetime]:
    """
    ISO 문자열을 datetime으로 변환.
    예: '2025-12-04T13:16:38Z', '2025-12-04 13:16:38+00:00'
    """
    if not value:
        return None

    s = value.strip()
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        # '2025-12-04 13:16:38' 처럼 tz 없는 경우도 허용
        dt = datetime.fromisoformat(s)
        return dt
    except Exception:
        return None


def to_kst(dt: DTLike) -> Optional[datetime]:
    """
    문자열 / naive datetime / aware datetime 아무거나 받아서
    KST timezone 이 들어간 datetime 으로 변환
    """
    if dt is None:
        return None

    if isinstance(dt, str):
        dt_parsed = parse_iso_dt(dt)
        if dt_parsed is None:
            return None
        dt = dt_parsed

    # tz 없으면 UTC 로 간주
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(KST)


def to_kst_str(dt: DTLike, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    KST 기준 문자열로 변환해서 리턴
    """
    k = to_kst(dt)
    if k is None:
        return ""
    return k.strftime(fmt)
