# -*- coding: utf-8 -*-
"""
time_utils.py (v7.5 Upgrade)

Timezone utilities for LDY Pro Trader.
- Standardizes KST (UTC+9) conversions.
- Provides UTC/KST datetime helpers.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Optional
import pandas as pd

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))

def now_utc() -> datetime:
    """DB/CSV 저장용: 항상 UTC 기준 aware datetime"""
    return datetime.now(timezone.utc)

def now_kst() -> datetime:
    """로그/화면 표시용: 한국 시간대 기준 aware datetime"""
    return datetime.now(KST)

def now_utc_str() -> str:
    """JSON DB 저장용: UTC 현재 시각을 ISO 포맷 문자열로 반환"""
    return datetime.now(timezone.utc).isoformat()

def to_kst_str(value: Any, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    DB/CSV에 저장된 UTC 시간(문자열/Timestamp)을 화면 표시용 KST 문자열로 변환.
    - value: str, datetime, pandas.Timestamp 등
    - Naive datetime(타임존 정보 없음)은 UTC로 가정하고 KST로 변환.
    """
    if value is None or value == "":
        return ""

    try:
        # 1. Pandas를 이용해 안전하게 파싱
        ts = pd.to_datetime(value, errors="coerce")
        
        # 파싱 실패(NaT)인 경우 빈 문자열 반환
        if pd.isna(ts):
            return ""

        # 2. Timezone 처리
        if ts.tzinfo is None:
            # 타임존 정보가 없으면 UTC로 간주 (시스템 표준)
            ts = ts.tz_localize("UTC")
        else:
            # 타임존 정보가 있으면 일단 UTC로 변환하여 기준 통일
            ts = ts.tz_convert("UTC")

        # 3. KST로 최종 변환 및 포맷팅
        ts_kst = ts.tz_convert(KST)
        return ts_kst.strftime(fmt)

    except Exception:
        # 어떤 에러가 발생하더라도 UI를 깨뜨리지 않도록 빈 문자열 반환
        return ""
