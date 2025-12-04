# time_utils.py
from datetime import datetime, timezone, timedelta
import pandas as pd

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))

def now_utc() -> datetime:
    """DB/CSV 저장용: 항상 UTC 기준 aware datetime"""
    return datetime.now(timezone.utc)

def now_kst() -> datetime:
    """로그 등에 찍을 때: 한국 시간대 기준 aware datetime"""
    return datetime.now(KST)

def to_kst_str(value, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    DB/CSV에 저장된 created_at / last_login_at (UTC 기준)을
    화면에 표시할 때 KST 문자열로 변환.
    - value: str, datetime, pandas.Timestamp 아무거나 받아도 됨
    """
    if value is None:
        return ""

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""

    # tz 정보 없으면 -> UTC로 가정
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    ts_kst = ts.tz_convert(KST)
    return ts_kst.strftime(fmt)
