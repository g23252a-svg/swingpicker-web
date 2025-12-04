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
    # 혹시 매우 구버전일 경우 대비 (pytz 등으로 교체 가능)
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
    ISO 문자열(예: '2025-12-04T13:16:38Z', '2025-12-04 13:16:38+00:00')을 datetime으로 변환.
    - Z(UTC 표시)가 있으면 +00:00 으로 치환
    - 실패 시 None
    """
    if not value:
        return None

    s = value.strip()
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        # '2025-12-04 13:16:38' 처럼 tz 없는 경우도 fromisoformat 허용
        dt = datetime.fromisoformat(s)
        return dt
    except Exception:
        return None


def to_kst(dt: DTLike) -> Optional[datetime]:
    """
    아무 형태(문자열/naive datetime/aware datetime)든 받아서
    KST 타임존이 들어간 datetime으로 변환.

    규칙:
    - str 이면 parse_iso_dt 로 먼저 파싱
    - tzinfo 없는 naive datetime 이면 UTC 로 간주
    - tzinfo 있는 aware datetime 이면 그대로 KST 로 변환
    """
    if dt is None:
        return None

    # 문자열이면 먼저 파싱
    if isinstance(dt, str):
        dt_parsed = parse_iso_dt(dt)
        if dt_parsed is None:
            return None
        dt = dt_parsed

    # tz 정보 없으면 UTC 로 가정
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(KST)


def to_kst_str(dt: DTLike, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    KST 기준 문자열로 반환.
    - dt: datetime 또는 문자열
    - fmt: strftime 포맷
    """
    k = to_kst(dt)
    if k is None:
        return ""
    return k.strftime(fmt)
