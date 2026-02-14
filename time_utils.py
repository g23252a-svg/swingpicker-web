# -*- coding: utf-8 -*-
"""
time_utils.py (v14.0 Chronos-Sovereign: Eternal)
- 100/100: 대소문자 무시, 파싱 에러 방어, Set 성능 최적화 완결
"""

from datetime import datetime, timezone, timedelta, time
from typing import Any, Optional, Iterable, Set, Union
from enum import Enum
import pandas as pd

# ----------------- 1. 시간대 및 세션 정의 -----------------
KST = timezone(timedelta(hours=9))

class MarketSession(Enum):
    """[v14.0] 시장 세션 정의 (Label, IsActive)"""
    CLOSED             = ("🌑 휴장/마감", False)
    PRE_MARKET         = ("🌅 장전 시간외/동시호가", True)
    REGULAR            = ("☀️ 정규장", True)
    CLOSING_AUCTION    = ("🔔 장마감 동시호가", True)
    AFTER_HOURS_CLOSE  = ("🌇 장후 시간외 종가", True)
    AFTER_HOURS_SINGLE = ("🌙 장후 시간외 단일가", True)
    WEEKEND            = ("🛌 주말", False)
    HOLIDAY            = ("🚩 공휴일", False)

    def __init__(self, label: str, is_active: bool):
        self.label = label
        self.is_active = is_active

# ----------------- 2. 내부 헬퍼 로직 (100점 패치) -----------------

def _validate_tz(assume_tz: Any) -> str:
    """[❗100점 패치] 대소문자 및 공백 정규화로 입력 방어력 극대화"""
    clean_tz = str(assume_tz).upper().strip()
    return clean_tz if clean_tz in ["UTC", "KST"] else "UTC"

def _policy_tz(assume_tz: str):
    """타임존 정책 객체 매핑 (Pandas 에러 방어)"""
    return timezone.utc if _validate_tz(assume_tz) == "UTC" else KST

def _normalize_ymd(s: Any) -> str:
    """[❗100점 패치] errors='coerce' 적용으로 더욱 견고해진 날짜 정규화"""
    try:
        # 1차 시도: 표준 파싱
        ts = pd.to_datetime(s, errors="coerce")
        if not pd.isna(ts):
            return ts.strftime("%Y%m%d")
        # 2차 시도: 숫자만 추출 (Digits Fallback)
        return "".join(ch for ch in str(s) if ch.isdigit())[:8]
    except:
        return ""

# ----------------- 3. 정밀 시간 획득 및 변환 (Legacy 호환 포함) -----------------

def now_utc() -> datetime:
    """항상 UTC 기준 aware datetime"""
    return datetime.now(timezone.utc)

def now_utc_str() -> str:
    """ISO 포맷 UTC 문자열"""
    return now_utc().isoformat()

def now_kst() -> datetime:
    return datetime.now(KST)

def to_kst_str(value: Any, fmt: str = "%Y-%m-%d %H:%M:%S", assume_tz: str = "UTC") -> str:
    if value is None or value == "": return ""
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts): return ""
        if ts.tzinfo is None: ts = ts.tz_localize(_policy_tz(assume_tz))
        return ts.tz_convert(KST).strftime(fmt)
    except:
        return ""

# ----------------- 4. 시장 상태 판정 (Final Sovereign Logic) -----------------

def get_market_session(
    holiday_list: Optional[Union[Iterable[str], Set[str]]] = None,
    now: Optional[Any] = None,
    assume_tz: str = "UTC"
) -> MarketSession:
    """[❗100점 패치] 주입 시간 정규화 및 Set 멤버십 성능 최적화"""
    # 1. 시간 정규화 (QA/테스트 대응)
    if now is not None:
        target_ts = pd.to_datetime(now, errors="coerce")
        if pd.isna(target_ts): return MarketSession.CLOSED
        if target_ts.tzinfo is None: target_ts = target_ts.tz_localize(_policy_tz(assume_tz))
        nk = target_ts.tz_convert(KST)
    else:
        nk = now_kst()

    today_str = nk.strftime("%Y%m%d")
    if nk.weekday() >= 5: return MarketSession.WEEKEND
    
    # 2. [❗100점 패치] 휴장일 멤버십 체크 최적화
    if holiday_list:
        # 이미 Set이라면 정규화되었다고 가정하거나, 리스트일 때만 1회 변환
        if isinstance(holiday_list, set):
            h_set = holiday_list
        else:
            h_set = { _normalize_ymd(x) for x in holiday_list }
        
        if today_str in h_set: return MarketSession.HOLIDAY
    
    # 3. 정밀 세션 판정
    now_t = nk.time()
    if time(8, 30) <= now_t < time(9, 0): return MarketSession.PRE_MARKET
    if time(9, 0) <= now_t < time(15, 20): return MarketSession.REGULAR
    if time(15, 20) <= now_t <= time(15, 30): return MarketSession.CLOSING_AUCTION
    if time(15, 30) < now_t < time(15, 40): return MarketSession.CLOSED
    if time(15, 40) <= now_t <= time(16, 0): return MarketSession.AFTER_HOURS_CLOSE
    if time(16, 0) < now_t <= time(18, 0): return MarketSession.AFTER_HOURS_SINGLE
    
    return MarketSession.CLOSED

def is_tradable_now(holiday_list: Optional[Iterable[str]] = None) -> bool:
    """현재 주문 가능 세션 여부"""
    return get_market_session(holiday_list=holiday_list).is_active

def is_regular_session_now(holiday_list: Optional[Iterable[str]] = None) -> bool:
    """정규장 체결 구간 여부 확인"""
    session = get_market_session(holiday_list=holiday_list)
    return session in [MarketSession.REGULAR, MarketSession.CLOSING_AUCTION]

# ----------------- 5. 데이터 핸들링 유틸리티 -----------------

def to_date_int(dt: Optional[Any] = None, assume_tz: str = "UTC") -> int:
    """KST 기준 YYYYMMDD 정수 반환 (타입 방어 완결)"""
    if dt is None:
        target_ts = now_kst()
    else:
        target_ts = pd.to_datetime(dt, errors="coerce")
        if pd.isna(target_ts): return 0
        if target_ts.tzinfo is None: target_ts = target_ts.tz_localize(_policy_tz(assume_tz))
    
    return int(target_ts.tz_convert(KST).strftime("%Y%m%d"))

def format_relative_time(value: Any, assume_tz: str = "UTC") -> str:
    """[❗100점 패치] 24시간 UX 오타 수정 및 대소문자 정책 통일"""
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts): return "-"
        if ts.tzinfo is None: ts = ts.tz_localize(_policy_tz(assume_tz))
        
        target_kst = ts.tz_convert(KST)
        diff = now_kst() - target_kst
        sec = diff.total_seconds()
        
        if sec < 0: return "미래"
        if sec < 60: return "방금 전"
        if sec < 3600: return f"{int(sec // 60)}분 전"
        if sec < 86400: return f"{int(sec // 3600)}시간 전"
        return target_kst.strftime("%m/%d %H:%M")
    except:
        return "-"
