# -*- coding: utf-8 -*-
"""naver_aftermarket.py - 시간외 단일가 수집 + 날짜별 보관 [v85]

[v20.6.4] sidecar 방식 추가 — recommend CSV 원본 오염 방지 (유지)

[v85] 수집 결함 수정. **연구용 데이터 수집·보관만** 한다 — 선정/가드 로직은 없다.
  시간외 등락을 매수 취소 가드로 쓰자는 안은 검증에서 근거가 없었다(-2% 버킷의
  다음 갭이 오히려 +). 이 파일은 나중에 그 질문을 제대로 다시 물을 수 있도록
  원자료를 빠짐없이, 기준을 분리해서 남기는 것까지만 한다.

  (a) 'final != 분석종가' 행만 쓰던 필터 제거 — 요청한 **모든 행**을 AH_STATUS 와 함께
      기록한다(이전엔 ~72%가 조용히 빠져 '변동 없음'과 '체결 없음'이 구분되지 않았다).
  (b) overPrice 가 없을 때 closePrice 를 시간외종가로 복사하던 폴백 제거 — 체결이
      없으면 시간외종가는 NaN 이다. 원값은 NAVER_CLOSE / AH_PRICE_RAW 로 따로 남긴다.
  (c) price_snapshot_latest.csv '종가'를 시간외가로 덮어쓰던 블록 제거 — 그 파일은
      커밋되고 portfolio/validation/position_tracker 의 가격 폴백으로 읽힌다.
  (d) 분석종가(= recommend '종가')는 pykrx→Naver fchart(수정주가 일봉)의 마지막 봉이라
      KRX 공식 종가가 아니다(09-14 이후 KRX 종가와 일치 ~14%). 그 기준으로 나눈
      변동률은 시간외 움직임을 상쇄한다. KRX_CLOSE(price_snapshot_{ymd}.csv) 기준
      AH_CHG_VS_KRX 를 별도로 남긴다.
  (e) data/aftermarket_prices_{ymd}.csv 날짜별 보관(원자적 쓰기) + latest 동일 사본.
      auto_collect.yml 의 `git add -f data/*.csv` 가 그대로 커밋한다.
  (f) 레거시 update_csv_with_aftermarket() 삭제 — 호출처가 없고 recommend CSV 와
      스냅샷을 직접 덮어쓰는 함수였다.

출력 컬럼 (aftermarket_prices_{ymd}.csv / aftermarket_prices_latest.csv, 행 = recommend 종목)
  종목코드            6자리 문자열
  종목명              recommend 의 종목명
  분석종가            recommend '종가' (Naver 차트 일봉 마지막 봉 — KRX 공식 종가 아닐 수 있음)
  시간외종가          시간외 체결가. AH_STATUS == OK 일 때만 값, 아니면 NaN (closePrice 복사 금지)
  변동률              (시간외종가/분석종가 - 1) * 100, % 단위 소수 2자리 — v20.6.4 와 같은
                      기준(분석종가)을 유지해 과거 latest 이력과 이어 볼 수 있게 했다.
                      체결이 없으면 NaN. ⚠️ 09-14 이후 분석종가가 이미 시간외를 반영한
                      것으로 보여 ~0 에 몰린다 — 연구에는 AH_CHG_VS_KRX 를 쓴다.
  AH_STATUS           OK(TRADE_YMD 의 시간외 체결가 있음) / NO_PRINT(수집 성공, TRADE_YMD 체결
                      없음 — AH_PRINT_DATE_MATCH=False 면 AH_PRICE_RAW 에 다른 날 체결가가 남아
                      있을 수 있다) / FETCH_FAIL(값을 못 얻음 — 파싱 불가 overPrice 포함)
  AH_FINAL            True = 이 행의 결과(OK 가격 또는 NO_PRINT)가 확정치.
                      수집 시각이 TRADE_YMD 20:00 KST 이후 ~ 다음 평일 08:00 KST 이전이고
                      FETCH_FAIL 이 아니며 AH_PRINT_DATE_MATCH 가 False 가 아닐 때.
                      (KRX 시간외 단일가 16:00~18:00, NXT 애프터마켓 ~20:00 — 그 전 수집은
                      부분 체결가일 수 있다). ⚠️ 수동 실행이 20:00 전에 recommend_{T}.csv 를
                      만들면 저녁 정기 실행이 스킵돼 그 날은 확정 보관본이 안 생길 수 있다
                      (워크플로 가드) — FETCH_WINDOW 로 확인한다.
  FETCH_WINDOW        BEFORE_SESSION / INTRADAY / AH_SESSION / FINAL_EVENING /
                      FINAL_OVERNIGHT / NEXT_SESSION / NOT_FETCHED
  TRADE_YMD           배치 기준일 (YYYYMMDD)
  FETCHED_AT_KST      행별 수집 시각 'YYYY-MM-DD HH:MM:SS' (수집 안 한 행은 빈칸)
  NAVER_CLOSE         Naver basic closePrice (숫자 파싱; 파싱 불가면 NaN)
  NAVER_CLOSE_STR     closePrice 받은 그대로의 문자열 (형식 확인/재파싱용)
  AH_PRICE_RAW        overMarketPriceInfo.overPrice (숫자 파싱, 0 포함; 없거나 파싱 불가면 NaN).
                      다른 날 체결가(AH_PRINT_DATE_MATCH=False)여도 지우지 않는다.
  AH_PRICE_STR        overPrice 받은 그대로의 문자열 — '61,500원' 같은 형식 변화가 와도
                      원값이 남는다(그 행은 FETCH_FAIL/PARSE).
  AH_PRINT_DATE_MATCH AH_TRADED_AT 의 날짜(KST) == TRADE_YMD. 타임스탬프를 못 읽으면 빈칸
                      (판정 안 함). False 면 다른 날 체결가 → OK 가 아니라 NO_PRINT,
                      시간외종가/변동률/AH_CHG_* NaN, AH_FINAL False (원값은 유지).
  KRX_CLOSE           data/price_snapshot_{TRADE_YMD}.csv '종가' — 없으면 NaN.
                      latest 스냅샷 폴백은 하지 않는다(날짜가 다를 수 있다).
  KRX_SNAP_SRC        FDR / PYKRX / MISSING / UNKNOWN — 스냅샷 '시장' 값으로 추정한 출처.
                      KONEX·KOSDAQ GLOBAL 이 있으면 FDR 폴백 경로(collector.save_price_snapshot)
                      다. FDR 캐시는 배치 시점에 미확정(장중 캡처)일 수 있다 → CLOSE_MATCH 로 교차확인.
  AH_CHG_VS_KRX       시간외종가 / KRX_CLOSE - 1 (비율)
  AH_CHG_VS_ANALYSIS  시간외종가 / 분석종가 - 1 (비율) — 변동률/100 과 같은 값(반올림 전)
  CLOSE_MATCH         NAVER_CLOSE == KRX_CLOSE (둘 다 있을 때만, 아니면 빈칸)
  NAVER_MARKET_STATUS marketStatus 원문
  NAVER_TRADED_AT     localTradedAt 원문
  AH_SESSION_TYPE     overMarketPriceInfo.tradingSessionType 원문
  AH_MARKET_STATUS    overMarketPriceInfo.overMarketStatus 원문
  AH_TRADED_AT        overMarketPriceInfo.localTradedAt 원문 — 날짜가 TRADE_YMD 와 다르면
                      예전 체결가일 수 있다(거래 없는 종목). 값을 지우지 않고 원문만 남긴다.
  AH_NAVER_FLUCT_RAW  overMarketPriceInfo.fluctuationsRatio 원문(%; Naver 가 정한 기준가 대비)
  NAVER_TRADE_STOP    최상위 tradeStopType (dict 면 name → code → JSON 순으로 짧게)
  AH_TRADE_STOP       overMarketPriceInfo.tradeStopType (같은 방식)
  AH_FAIL_REASON      FETCH_FAIL 사유 HTTP_xxx / TIMEOUT / NETWORK / JSON / SCHEMA / EMPTY /
                      PARSE(overPrice 가 있는데 숫자로 못 읽음) / BUDGET / CIRCUIT
  AH_EXTRA            예상 밖 필드(스키마 확인용, 보통 빈칸) — NXT 별도 블록 등이 생기면 여기 남는다.
                      항상 올바른 JSON (길면 키를 덜어내고 "_truncated": 덜어낸 개수)
  SCHEMA_VER          'v85'

날짜별 보관본 덮어쓰기: 같은 TRADE_YMD 파일이 이미 있고 그 파일의 AH_FINAL 행 수가 이번
결과보다 많으면 날짜별 파일은 그대로 두고 latest 만 갱신한다(다음날 수동 재실행·Naver 장애
실행이 확정 보관본을 덮지 않게). 이 경우 latest 와 날짜별 파일은 다를 수 있다.

⚠️ 엔드포인트 스키마는 샌드박스에서 확인하지 못했다(프록시 403). 모든 필드는 .get 으로
   방어적으로 읽고 원문 문자열을 남긴다. 가격 원값(AH_PRICE_RAW/AH_PRICE_STR)은 어떤 경우에도
   지우지 않고, 판정에 쓰는 것은 AH_TRADED_AT 날짜 하나뿐이다(못 읽으면 판정 안 함) —
   첫 운영 배치에서 원문 컬럼으로 의미를 확인한다.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

NAVER_BASIC_URL = "https://m.stock.naver.com/api/stock/{code}/basic"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

SCHEMA_VER = "v85"

#: (connect, read) 초 — 기존 5초 read 유지, 연결 단계만 짧게.
REQUEST_TIMEOUT: Tuple[float, float] = (3.05, 5.0)
#: 기존 페이싱 유지 — 20건마다 0.5초.
PACE_EVERY = 20
PACE_SLEEP_SEC = 0.5
#: 전체 수집 시간 상한(초). 평소 ~360-410종목 × 수백 ms ≈ 1~3분. Naver 장애 시
#: 종목당 타임아웃이 누적돼 배치를 붙잡지 않도록 — 초과분은 FETCH_FAIL(BUDGET).
TIME_BUDGET_SEC = 600.0
#: 연속 '계통' 실패(타임아웃·네트워크·JSON·HTTP 403/429/5xx 등) 이 횟수면 차단기가 열린다.
#: 종목 단위 실패(HTTP_404·SCHEMA·PARSE)는 세지 않는다.
MAX_CONSEC_FAIL = 15
#: [v85] 반개방(half-open) 차단기 — 열리면 이만큼 쉬고 1건 시험 요청. 성공하면 재개,
#: 실패하면 다시 쉼. 최대 MAX_CIRCUIT_TRIPS 번까지(쉬는 시간도 TIME_BUDGET 에 포함) →
#: 짧은 429/5xx 버스트 하나로 그날 나머지 전 종목을 잃지 않게. 그래도 안 되면 CIRCUIT.
CIRCUIT_COOLDOWN_SEC = 30.0
MAX_CIRCUIT_TRIPS = 3
_NON_SYSTEMIC_FAIL = {"HTTP_404", "SCHEMA", "PARSE"}

STATUS_OK = "OK"
STATUS_NO_PRINT = "NO_PRINT"
STATUS_FETCH_FAIL = "FETCH_FAIL"
STATUSES = (STATUS_OK, STATUS_NO_PRINT, STATUS_FETCH_FAIL)

#: 확정 판정 경계 (KST). KRX 시간외 단일가 16:00~18:00, NXT 애프터마켓 15:30~20:00,
#: NXT 프리마켓 08:00~ — 그래서 [거래일 20:00, 다음 평일 08:00) 만 확정이다.
AH_SESSION_START = dtime(15, 30)
AH_FINAL_FROM = dtime(20, 0)
NEXT_SESSION_OPEN = dtime(8, 0)

OUTPUT_COLUMNS = [
    "종목코드", "종목명", "분석종가", "시간외종가", "변동률",
    "AH_STATUS", "AH_FINAL", "FETCH_WINDOW", "TRADE_YMD", "FETCHED_AT_KST",
    "NAVER_CLOSE", "NAVER_CLOSE_STR", "AH_PRICE_RAW", "AH_PRICE_STR", "AH_PRINT_DATE_MATCH",
    "KRX_CLOSE", "KRX_SNAP_SRC",
    "AH_CHG_VS_KRX", "AH_CHG_VS_ANALYSIS", "CLOSE_MATCH",
    "NAVER_MARKET_STATUS", "NAVER_TRADED_AT",
    "AH_SESSION_TYPE", "AH_MARKET_STATUS", "AH_TRADED_AT", "AH_NAVER_FLUCT_RAW",
    "NAVER_TRADE_STOP", "AH_TRADE_STOP",
    "AH_FAIL_REASON", "AH_EXTRA", "SCHEMA_VER",
]
_PRICE_COLUMNS = ("분석종가", "시간외종가", "NAVER_CLOSE", "AH_PRICE_RAW", "KRX_CLOSE")
# 결과 dict 의 문자열 필드 (fetch_fn 결과 정규화 때 str 로 맞춘다)
_STR_FIELDS = (
    "fail_reason", "naver_close_str", "ah_price_str",
    "naver_market_status", "naver_traded_at",
    "ah_session_type", "ah_market_status", "ah_traded_at", "ah_fluct_raw",
    "naver_trade_stop", "ah_trade_stop", "extra",
)

# overMarketPriceInfo 안에서 이미 다루거나(원문 컬럼) 의도적으로 버리는 키
_KNOWN_OVER_KEYS = {
    "overPrice", "tradingSessionType", "overMarketStatus", "localTradedAt",
    "fluctuationsRatio", "compareToPreviousClosePrice", "compareToPreviousPrice",
    "tradeStopType",
}
# 최상위에서 스키마 변화(NXT 별도 블록 등)를 잡기 위해 살펴볼 키 패턴
_EXTRA_TOP_KEY_RE = re.compile(r"(?i)(nxt|over|after|session)")
_EXTRA_MAX_LEN = 300
#: [v85] 원화 가격 형식만 — '61,500' (3자리 콤마 묶음) 또는 '61500', 소수점 허용.
#:   '61.500,00'(유럽식)처럼 콤마를 지우면 다른 수가 되는 문자열은 받지 않는다(→ NaN).
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
#: '값 없음' 자리표시 — 이것들은 체결 없음(NO_PRINT), 그 밖의 파싱 불가 문자열은 PARSE 실패.
_PLACEHOLDERS = {"", "-", "--", "N/A", "NA", "NULL", "NONE", "NAN"}
#: 타임스탬프 앞부분 — '2026-10-16T18:00:00+09:00' / '20261016180000' / '2026.10.16' / ...Z
_TS_RE = re.compile(
    r"^\s*(\d{4})[-./]?(\d{2})[-./]?(\d{2})"
    r"(?:[T ]?(\d{2}):?(\d{2})(?::?(\d{2}))?(?:\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?)?")


# ─────────────────────────────────────────────────────────────
#  파싱
# ─────────────────────────────────────────────────────────────

def _num(v: Any) -> float:
    """'209,000' / 209000 / '0' → float. 빈값·'-'·음수·비숫자·이상한 콤마 → NaN."""
    if v is None or isinstance(v, bool):
        return math.nan
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        s = str(v).strip()
        if not _NUM_RE.fullmatch(s):   # '', '-', 'N/A', 음수, '61,500원' → NaN (예외 없이)
            return math.nan
        x = float(s.replace(",", ""))
    if math.isnan(x) or math.isinf(x) or x < 0:
        return math.nan
    return x


def _is_placeholder(v: Any) -> bool:
    """None / '' / '-' / 'N/A' … — '값 없음' 표시. 숫자형 NaN 도 포함."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip().upper() in _PLACEHOLDERS


def _json_capped(obj: Any, max_len: int = _EXTRA_MAX_LEN) -> str:
    """JSON 직렬화 + 길이 상한. 잘라도 **항상 올바른 JSON** 이다.

    dict 는 앞에서부터 들어가는 만큼 키를 넣고 "_truncated": 덜어낸 키 수 를 붙인다.
    list 는 앞부분 원소 + "…(+n)" 표시 원소. 그 밖에는 {"_truncated": 원래 길이}.
    """
    def _dump(o: Any) -> str:
        return json.dumps(o, ensure_ascii=False, separators=(",", ":"), default=str)

    s = _dump(obj)
    if len(s) <= max_len:
        return s
    if isinstance(obj, dict):
        items = list(obj.items())
        kept: Dict[str, Any] = {}
        for k, v in items:
            cand = dict(kept)
            cand[str(k)] = v
            cand["_truncated"] = len(items) - len(cand)
            if len(_dump(cand)) > max_len:
                break
            kept[str(k)] = v
        kept["_truncated"] = len(items) - len(kept)
        return _dump(kept)
    if isinstance(obj, list):
        kept_l: list = []
        for v in obj:
            cand_l = kept_l + [v, f"…(+{len(obj) - len(kept_l) - 1})"]
            if len(_dump(cand_l)) > max_len:
                break
            kept_l.append(v)
        return _dump(kept_l + [f"…(+{len(obj) - len(kept_l)})"])
    return _dump({"_truncated": len(s)})


def _raw_str(v: Any) -> str:
    """원문 보존용 문자열. 스칼라는 그대로 — dict/list 는 (상한 안의 올바른) JSON 으로."""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return _json_capped(v)
    return str(v)


def _status_str(v: Any) -> str:
    """상태 필드(tradeStopType 등) — dict 면 name → code → JSON 순으로 짧게.

    {"code":"1","text":"운영.Trading","name":"TRADING"} 를 행마다 통째로 남기면 하루 ~30KB 라
    식별자(name)만 남긴다. name/code 가 없으면 JSON 원문(상한 안)."""
    if isinstance(v, dict):
        for k in ("name", "code"):
            if _is_scalar(v.get(k)) and v.get(k) not in (None, ""):
                return str(v.get(k))
    return _raw_str(v)


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _extra_fields(d: dict, over: Optional[dict]) -> str:
    """예상 밖 필드를 짧은 JSON 으로. 스키마가 예상대로면 빈 문자열."""
    extra: Dict[str, Any] = {}
    if isinstance(over, dict):
        for k, v in over.items():
            if k not in _KNOWN_OVER_KEYS and _is_scalar(v):
                extra[f"over.{k}"] = v
    for k, v in d.items():
        if k == "overMarketPriceInfo" or not _EXTRA_TOP_KEY_RE.search(str(k)):
            continue
        if _is_scalar(v):
            extra[k] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if _is_scalar(v2):
                    extra[f"{k}.{k2}"] = v2
    if not extra:
        return ""
    return _json_capped(extra)


def print_date_kst(v: Any) -> str:
    """체결 타임스탬프 원문 → KST 날짜 'YYYYMMDD'. 못 읽으면 '' (판정 안 함).

    '2026-10-16T18:00:00+09:00' / '20261016180000' / '2026.10.16' / '2026-10-16T09:00:00Z'.
    시간대가 붙어 있으면 KST 로 바꿔 날짜를 잡는다(없으면 KST 로 본다).
    """
    if v is None or isinstance(v, (bool, dict, list)):
        return ""
    s = str(v).strip()
    # 숫자만인 값은 YYYYMMDD[HHMM[SS]] 길이만 — epoch(10·13자리)를 날짜로 오인하면 같은 날
    # 체결까지 '다른 날'로 떨어진다. 연도는 20xx 만.
    if s.isdigit() and len(s) not in (8, 12, 14):
        return ""
    m = _TS_RE.match(s)
    if not m or not m.group(1).startswith("20"):
        return ""
    y, mo, dd, hh, mi, ss, tz = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(dd), int(hh or 0), int(mi or 0), int(ss or 0))
        if tz:
            if tz == "Z":
                off = timedelta(0)
            else:
                t = tz[1:].replace(":", "")
                off = timedelta(hours=int(t[:2]), minutes=int(t[2:]))
                off = -off if tz[0] == "-" else off
            dt = dt.replace(tzinfo=timezone(off)).astimezone(KST)
    except ValueError as e:   # 2026-13-45 같은 달력 밖 값 / 범위 밖 오프셋
        logger.warning(f"[v85] 시간외 체결 타임스탬프 해석 불가 → 날짜 판정 안 함: {v!r} ({e})")
        return ""
    return dt.strftime("%Y%m%d")


def _empty_result(status: str, reason: str = "") -> dict:
    return {
        "status": status, "fail_reason": reason,
        "naver_close": math.nan, "ah_price_raw": math.nan,
        "naver_close_str": "", "ah_price_str": "",
        "naver_market_status": "", "naver_traded_at": "",
        "ah_session_type": "", "ah_market_status": "", "ah_traded_at": "",
        "ah_fluct_raw": "", "naver_trade_stop": "", "ah_trade_stop": "", "extra": "",
    }


def parse_basic_payload(d: Any) -> dict:
    """Naver /api/stock/{code}/basic JSON → 원값 필드 dict (순수 함수).

    - overPrice > 0  → OK
    - overPrice 없음/0/자리표시('', '-', 'N/A' …) → NO_PRINT (closePrice 는 NAVER_CLOSE 로만)
    - overPrice 가 있는데 숫자로 못 읽음('61,500원' 등) → FETCH_FAIL(PARSE) — '체결 없음'으로
      적으면 확정 NO_PRINT 로 둔갑하므로 값을 못 얻은 것으로 남기고, 원문은 ah_price_str 에.
    - closePrice 도 overMarketPriceInfo 도 없는 응답(오류 JSON 등) → FETCH_FAIL(SCHEMA)
    """
    if not isinstance(d, dict):
        return _empty_result(STATUS_FETCH_FAIL, "SCHEMA")
    over = d.get("overMarketPriceInfo")
    over = over if isinstance(over, dict) else None
    res = _empty_result(STATUS_NO_PRINT)
    res["naver_close"] = _num(d.get("closePrice"))
    res["naver_close_str"] = _raw_str(d.get("closePrice"))
    res["naver_market_status"] = _raw_str(d.get("marketStatus"))
    res["naver_traded_at"] = _raw_str(d.get("localTradedAt"))
    res["naver_trade_stop"] = _status_str(d.get("tradeStopType"))
    over_price = None
    if over is not None:
        over_price = over.get("overPrice")
        res["ah_price_raw"] = _num(over_price)
        res["ah_price_str"] = _raw_str(over_price)
        res["ah_session_type"] = _raw_str(over.get("tradingSessionType"))
        res["ah_market_status"] = _raw_str(over.get("overMarketStatus"))
        res["ah_traded_at"] = _raw_str(over.get("localTradedAt"))
        res["ah_fluct_raw"] = _raw_str(over.get("fluctuationsRatio"))
        res["ah_trade_stop"] = _status_str(over.get("tradeStopType"))
    res["extra"] = _extra_fields(d, over)
    ah = res["ah_price_raw"]
    if not math.isnan(ah) and ah > 0:
        res["status"] = STATUS_OK
    elif over is not None and math.isnan(ah) and not _is_placeholder(over_price):
        res["status"] = STATUS_FETCH_FAIL
        res["fail_reason"] = "PARSE"
    elif over is None and "closePrice" not in d:
        res["status"] = STATUS_FETCH_FAIL
        res["fail_reason"] = "SCHEMA"
    return res


def fetch_after_market_price(code: str, session: Any = None,
                             timeout: Any = REQUEST_TIMEOUT) -> dict:
    """한 종목의 Naver basic 응답을 원값 필드로. 예외를 밖으로 던지지 않는다.

    [v85] 이전 버전은 overPrice 가 없으면 final=closePrice 로 돌려줘 정규장 종가가
    시간외종가로 기록됐다. 이제 원값만 돌려주고, 판정은 status 로 한다.
    """
    url = NAVER_BASIC_URL.format(code=code)
    getter = session.get if session is not None else requests.get
    try:
        r = getter(url, timeout=timeout, headers=_HEADERS)
    except requests.exceptions.Timeout as e:
        logger.warning(f"[v85] 시간외 수집 타임아웃 {code}: {e}")
        return _empty_result(STATUS_FETCH_FAIL, "TIMEOUT")
    except requests.exceptions.RequestException as e:
        logger.warning(f"[v85] 시간외 수집 네트워크 실패 {code}: {e}")
        return _empty_result(STATUS_FETCH_FAIL, "NETWORK")
    status_code = getattr(r, "status_code", None)
    if status_code != 200:
        return _empty_result(STATUS_FETCH_FAIL, f"HTTP_{status_code}")
    try:
        d = r.json()
    except ValueError as e:  # json.JSONDecodeError / requests JSONDecodeError 모두 ValueError
        logger.warning(f"[v85] 시간외 응답 JSON 파싱 실패 {code}: {e}")
        return _empty_result(STATUS_FETCH_FAIL, "JSON")
    return parse_basic_payload(d)


def _normalize_fetch_result(res: Any) -> dict:
    """fetch_fn 결과를 표준 dict 로. 비었거나 모양이 다르면 FETCH_FAIL."""
    if not isinstance(res, dict) or not res:
        return _empty_result(STATUS_FETCH_FAIL, "EMPTY")
    out = _empty_result(STATUS_FETCH_FAIL, "")
    out.update({k: v for k, v in res.items() if k in out})
    if out["status"] not in STATUSES:
        out["status"] = STATUS_FETCH_FAIL
        out["fail_reason"] = out["fail_reason"] or "SCHEMA"
    out["naver_close"] = _num(out["naver_close"])
    out["ah_price_raw"] = _num(out["ah_price_raw"])
    for k in _STR_FIELDS:   # None/NaN → '' , 그 밖은 원문 문자열 (CSV 에 'nan' 이 찍히지 않게)
        v = out[k]
        out[k] = "" if v is None or (isinstance(v, float) and math.isnan(v)) else _raw_str(v)
    # OK 는 실제 시간외 체결가가 있을 때만 — 모양만 OK 인 결과를 믿지 않는다.
    if out["status"] == STATUS_OK and not (out["ah_price_raw"] > 0):
        out["status"] = STATUS_NO_PRINT
    return out


# ─────────────────────────────────────────────────────────────
#  확정(finality) 판정
# ─────────────────────────────────────────────────────────────

def _to_kst(ts: Any) -> datetime:
    """datetime/문자열 → KST aware datetime. naive 는 KST 로 본다."""
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KST)
    return dt.astimezone(KST)


def _next_weekday(d: date) -> date:
    n = d + timedelta(days=1)
    while n.weekday() >= 5:
        n += timedelta(days=1)
    return n


def aftermarket_finality(trade_ymd: str, fetched_at: Any) -> Tuple[bool, str]:
    """(확정 여부, FETCH_WINDOW). 기준: 거래일 T, 수집 시각 KST.

      T 이전 / T 08:00 전          BEFORE_SESSION   (직전 세션 값)
      T 08:00 ~ 15:30              INTRADAY         (장중 — 시간외 아님)
      T 15:30 ~ 20:00              AH_SESSION       (KRX 단일가 ~18:00, NXT ~20:00 진행 중)
      T 20:00 ~ 24:00              FINAL_EVENING    확정
      T+ ~ 다음 평일 08:00 전      FINAL_OVERNIGHT  확정 (새벽 회수 실행)
      다음 평일 08:00 이후         NEXT_SESSION     (다음 세션 프리마켓/장중 값일 수 있음)

    다음 세션은 '다음 평일'로 근사한다 — 평일 휴장일 아침은 NEXT_SESSION(미확정)으로
    보수적으로 떨어진다(틀려도 확정을 미확정으로만 틀린다).
    """
    t_day = datetime.strptime(str(trade_ymd)[:8], "%Y%m%d").date()
    ts = _to_kst(fetched_at)
    d, t = ts.date(), ts.time()
    if d < t_day:
        return False, "BEFORE_SESSION"
    if d == t_day:
        if t < NEXT_SESSION_OPEN:
            return False, "BEFORE_SESSION"
        if t < AH_SESSION_START:
            return False, "INTRADAY"
        if t < AH_FINAL_FROM:
            return False, "AH_SESSION"
        return True, "FINAL_EVENING"
    nxt_open = datetime.combine(_next_weekday(t_day), NEXT_SESSION_OPEN, tzinfo=KST)
    if ts < nxt_open:
        return True, "FINAL_OVERNIGHT"
    return False, "NEXT_SESSION"


# ─────────────────────────────────────────────────────────────
#  기준가 (KRX 스냅샷)
# ─────────────────────────────────────────────────────────────

def load_krx_close(data_dir: str, trade_ymd: str) -> Tuple[Dict[str, float], str]:
    """data/price_snapshot_{ymd}.csv → ({종목코드: 종가}, KRX_SNAP_SRC).

    날짜 파일만 읽는다 — price_snapshot_latest.csv 로 폴백하지 않는다(다른 날일 수 있음).
    """
    path = os.path.join(data_dir, f"price_snapshot_{trade_ymd}.csv")
    if not os.path.exists(path):
        logger.warning(f"[v85] KRX 기준 스냅샷 없음 → KRX_CLOSE NaN: {path}")
        return {}, "MISSING"
    try:
        snap = pd.read_csv(path, dtype={"종목코드": str}, encoding="utf-8-sig")
    except Exception as e:
        logger.warning(f"[v85] KRX 기준 스냅샷 읽기 실패 → KRX_CLOSE NaN: {path}: {e}")
        return {}, "MISSING"
    if "종목코드" not in snap.columns or "종가" not in snap.columns:
        logger.warning(f"[v85] KRX 기준 스냅샷 컬럼 이상 → KRX_CLOSE NaN: {list(snap.columns)[:8]}")
        return {}, "MISSING"
    codes = snap["종목코드"].astype(str).str.strip().str.zfill(6)
    closes = pd.to_numeric(snap["종가"], errors="coerce")
    m = {c: float(v) for c, v in zip(codes, closes) if pd.notna(v) and v > 0}
    if "시장" in snap.columns:
        mk = set(snap["시장"].dropna().astype(str).str.strip().unique())
        src = "FDR" if (mk - {"KOSPI", "KOSDAQ"}) else "PYKRX"
    else:
        src = "UNKNOWN"
    return m, src


# ─────────────────────────────────────────────────────────────
#  쓰기
# ─────────────────────────────────────────────────────────────

def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """같은 디렉터리 임시파일 → os.replace. 실패하면 기존 파일이 그대로 남는다."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".aftermarket_", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            df.to_csv(f, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _as_int_if_integral(s: pd.Series) -> pd.Series:
    """가격 컬럼을 '209000.0' 대신 '209000' 으로 쓰기 위해 — 정수값뿐일 때만 Int64."""
    x = pd.to_numeric(s, errors="coerce")
    nn = x.dropna()
    if len(nn) and (nn % 1 == 0).all():
        return x.round().astype("Int64")
    return x


def _ratio(a: float, b: float) -> float:
    if a is None or b is None or math.isnan(a) or math.isnan(b) or b <= 0:
        return math.nan
    return round(a / b - 1.0, 6)


def _resolve_trade_ymd(df: pd.DataFrame, csv_path: str) -> str:
    """trade_ymd 미지정 시 recommend 의 기준일/PRICE_ASOF 최빈값 → 파일명 → 오늘(KST)."""
    for col in ("기준일", "PRICE_ASOF"):
        if col in df.columns:
            v = df[col].dropna().astype(str).str.replace(r"\D", "", regex=True).str[:8]
            v = v[v.str.len() == 8]
            if len(v):
                return str(v.mode().iloc[0])
    m = re.search(r"(\d{8})", os.path.basename(str(csv_path)))
    if m:
        return m.group(1)
    today = datetime.now(KST).strftime("%Y%m%d")
    logger.warning(f"[v85] trade_ymd 를 recommend 에서 못 찾음 → 오늘(KST) {today} 사용")
    return today


def _existing_final_count(path: str) -> Optional[int]:
    """이미 있는 날짜별 보관본의 AH_FINAL=True 행 수. 없거나 못 읽으면 None(덮어써도 됨)."""
    if not os.path.exists(path):
        return None
    try:
        old = pd.read_csv(path, usecols=lambda c: c == "AH_FINAL", dtype=str,
                          encoding="utf-8-sig")
    except Exception as e:
        logger.warning(f"[v85] 기존 시간외 보관본 읽기 실패 → 새 결과로 덮어씀: {path}: {e}")
        return None
    if "AH_FINAL" not in old.columns:
        return None
    return int(old["AH_FINAL"].astype(str).str.strip().str.lower().eq("true").sum())


def _summary(n_rows=0, dated_path=None, latest_path=None, trade_ymd=None, error=None,
             counts=None, n_final=0, krx_src="", elapsed=0.0, dated_written=False) -> dict:
    return {
        "n_rows": int(n_rows),
        "counts": dict(counts or {s: 0 for s in STATUSES}),
        "n_ok": int((counts or {}).get(STATUS_OK, 0)),
        "n_final": int(n_final),
        "trade_ymd": trade_ymd,
        "dated_path": dated_path,
        "dated_written": bool(dated_written),
        "latest_path": latest_path,
        "krx_snap_src": krx_src,
        "elapsed_sec": round(float(elapsed), 1),
        "error": error,
    }


def build_aftermarket_archive(
    recommend_csv: str,
    trade_ymd: Optional[str] = None,
    out_dir: Optional[str] = None,
    *,
    latest_name: str = "aftermarket_prices_latest.csv",
    tag: Optional[str] = None,
    fetch_fn: Optional[Callable[[str], dict]] = None,
    now_fn: Optional[Callable[[], datetime]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    time_budget_sec: float = TIME_BUDGET_SEC,
    max_consec_fail: int = MAX_CONSEC_FAIL,
    circuit_cooldown_sec: float = CIRCUIT_COOLDOWN_SEC,
    max_circuit_trips: int = MAX_CIRCUIT_TRIPS,
) -> dict:
    """recommend 전 종목의 시간외 단일가를 수집해 날짜별 + latest 로 저장한다 [v85].

    recommend CSV 는 읽기만 한다(원본 불변). price_snapshot_latest.csv 는 건드리지 않는다.
    실패는 밖으로 던지지 않고 summary['error'] 로 돌려준다 — 배치를 막으면 안 된다.
    같은 날짜별 파일에 확정(AH_FINAL) 행이 더 많으면 날짜별은 보존하고 latest 만 쓴다.

    Returns: {n_rows, counts{OK,NO_PRINT,FETCH_FAIL}, n_ok, n_final, trade_ymd,
              dated_path, dated_written, latest_path, krx_snap_src, elapsed_sec, error}
    """
    t0 = monotonic_fn()
    try:
        df = pd.read_csv(recommend_csv, dtype=str, encoding="utf-8-sig")
    except Exception as e:
        logger.warning(f"[v85] recommend CSV 읽기 실패 — 시간외 수집 스킵: {e}")
        return _summary(error=f"read_fail: {e}")

    code_col = next((c for c in df.columns if c == "종목코드"), None) \
        or next((c for c in df.columns if "종목코드" in c or c == "code"), None)
    if code_col is None:
        logger.warning(f"[v85] recommend 에 종목코드 컬럼 없음 — 시간외 수집 스킵: {list(df.columns)[:8]}")
        return _summary(error="no_code_col")
    close_col = next((c for c in df.columns if c == "종가"), None) \
        or next((c for c in df.columns if c in ("close", "Close")), None)
    name_col = "종목명" if "종목명" in df.columns else None

    trade_ymd = str(trade_ymd).strip() if trade_ymd else _resolve_trade_ymd(df, recommend_csv)
    try:
        if not re.fullmatch(r"\d{8}", trade_ymd):
            raise ValueError("YYYYMMDD 8자리 아님")
        datetime.strptime(trade_ymd, "%Y%m%d")
    except ValueError as e:
        logger.warning(f"[v85] trade_ymd 형식 이상 — 시간외 수집 스킵: {trade_ymd!r} ({e})")
        return _summary(error=f"bad_trade_ymd: {trade_ymd!r}")
    out_dir = out_dir or os.path.dirname(os.path.abspath(recommend_csv))

    base = pd.DataFrame({"종목코드": df[code_col].astype(str).str.strip().str.zfill(6)})
    base["종목명"] = df[name_col].fillna("").astype(str) if name_col else ""
    base["분석종가"] = pd.to_numeric(df[close_col], errors="coerce") if close_col else math.nan
    n_req = len(base)
    base = base[base["종목코드"].str.fullmatch(r"[0-9A-Z]{6}", na=False)]
    base = base.drop_duplicates("종목코드", keep="first").reset_index(drop=True)
    if len(base) != n_req:
        logger.warning(f"[v85] 시간외 수집 대상 정리: {n_req} → {len(base)} (중복/비정상 코드 제외)")

    krx_map, krx_src = load_krx_close(out_dir, trade_ymd)
    now_fn = now_fn or (lambda: datetime.now(KST))

    sess = None
    if fetch_fn is None:
        sess = requests.Session()
        # 모듈 전역을 호출 시점에 찾는다 — 테스트의 patch("naver_aftermarket.fetch_after_market_price") 존중
        fetch_fn = lambda c: fetch_after_market_price(c, session=sess)  # noqa: E731

    logger.info(f"[v85] 시간외 수집 시작: {len(base)}종목 (기준일 {trade_ymd}, KRX 기준 {krx_src})")
    rows = []
    consec_fail = 0
    trips = 0
    n_date_mismatch = 0
    stop_reason = ""
    codes = base["종목코드"].tolist()
    names = base["종목명"].tolist()
    ana_closes = base["분석종가"].tolist()
    try:
        for i, code in enumerate(codes):
            fetched_at = None
            if not stop_reason:
                if monotonic_fn() - t0 > time_budget_sec:
                    stop_reason = "BUDGET"
                elif consec_fail >= max_consec_fail:
                    if trips >= max_circuit_trips:
                        stop_reason = "CIRCUIT"
                    else:
                        # 반개방: 쉬고 이번 종목을 시험 요청으로 — 성공하면 consec_fail 이 0 으로
                        trips += 1
                        logger.warning(f"[v85] 시간외 연속 실패 {consec_fail}회 — "
                                       f"{circuit_cooldown_sec:.0f}s 쉬고 시험 요청 "
                                       f"({trips}/{max_circuit_trips})")
                        sleep_fn(circuit_cooldown_sec)
                        if monotonic_fn() - t0 > time_budget_sec:
                            stop_reason = "BUDGET"
                if stop_reason:
                    logger.warning(f"[v85] 시간외 수집 중단({stop_reason}) — 남은 "
                                   f"{len(codes) - i}종목은 FETCH_FAIL 로 기록")
            if stop_reason:
                r = _empty_result(STATUS_FETCH_FAIL, stop_reason)
            else:
                try:
                    r = _normalize_fetch_result(fetch_fn(code))
                except Exception as e:
                    logger.warning(f"[v85] 시간외 수집 예외 {code}: {e}")
                    r = _empty_result(STATUS_FETCH_FAIL, "NETWORK")
                fetched_at = _to_kst(now_fn())
                if r["status"] == STATUS_FETCH_FAIL and r["fail_reason"] not in _NON_SYSTEMIC_FAIL:
                    consec_fail += 1
                else:
                    consec_fail = 0
                if (i + 1) % PACE_EVERY == 0:
                    sleep_fn(PACE_SLEEP_SEC)

            if fetched_at is not None:
                win_final, window = aftermarket_finality(trade_ymd, fetched_at)
            else:
                win_final, window = False, "NOT_FETCHED"
            # 체결 타임스탬프 날짜 대조 — 다른 날 체결가(거래 없는 종목의 예전 값, 새벽 회수 때
            # 이미 넘어간 블록)를 TRADE_YMD 의 시간외가로 적지 않는다. 원값은 그대로 둔다.
            p_date = print_date_kst(r["ah_traded_at"])
            date_match: Any = (p_date == trade_ymd) if p_date else None
            if date_match is False:
                n_date_mismatch += 1
                if r["status"] == STATUS_OK:
                    r["status"] = STATUS_NO_PRINT
            ok = r["status"] == STATUS_OK
            ah = r["ah_price_raw"] if ok else math.nan
            ana = float(ana_closes[i]) if pd.notna(ana_closes[i]) else math.nan
            krx = krx_map.get(code, math.nan)
            nclose = r["naver_close"]
            chg_ana = _ratio(ah, ana)
            if not math.isnan(nclose) and not math.isnan(krx):
                close_match: Any = bool(nclose == krx)
            else:
                close_match = None
            rows.append({
                "종목코드": code,
                "종목명": names[i],
                "분석종가": ana,
                "시간외종가": ah,
                "변동률": round(chg_ana * 100, 2) if not math.isnan(chg_ana) else math.nan,
                "AH_STATUS": r["status"],
                "AH_FINAL": bool(win_final and r["status"] != STATUS_FETCH_FAIL
                                 and date_match is not False),
                "FETCH_WINDOW": window,
                "TRADE_YMD": trade_ymd,
                "FETCHED_AT_KST": fetched_at.strftime("%Y-%m-%d %H:%M:%S") if fetched_at else "",
                "NAVER_CLOSE": nclose,
                "NAVER_CLOSE_STR": r["naver_close_str"],
                "AH_PRICE_RAW": r["ah_price_raw"],
                "AH_PRICE_STR": r["ah_price_str"],
                "AH_PRINT_DATE_MATCH": date_match,
                "KRX_CLOSE": krx,
                "KRX_SNAP_SRC": krx_src,
                "AH_CHG_VS_KRX": _ratio(ah, krx),
                "AH_CHG_VS_ANALYSIS": chg_ana,
                "CLOSE_MATCH": close_match,
                "NAVER_MARKET_STATUS": r["naver_market_status"],
                "NAVER_TRADED_AT": r["naver_traded_at"],
                "AH_SESSION_TYPE": r["ah_session_type"],
                "AH_MARKET_STATUS": r["ah_market_status"],
                "AH_TRADED_AT": r["ah_traded_at"],
                "AH_NAVER_FLUCT_RAW": r["ah_fluct_raw"],
                "NAVER_TRADE_STOP": r["naver_trade_stop"],
                "AH_TRADE_STOP": r["ah_trade_stop"],
                "AH_FAIL_REASON": r["fail_reason"] if r["status"] == STATUS_FETCH_FAIL else "",
                "AH_EXTRA": r["extra"],
                "SCHEMA_VER": SCHEMA_VER,
            })
    finally:
        if sess is not None:
            sess.close()

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for c in _PRICE_COLUMNS:
        out[c] = _as_int_if_integral(out[c])
    counts = {s: int((out["AH_STATUS"] == s).sum()) for s in STATUSES}
    n_final = int(out["AH_FINAL"].astype(bool).sum()) if len(out) else 0
    if counts[STATUS_FETCH_FAIL]:
        _reasons = out.loc[out["AH_STATUS"] == STATUS_FETCH_FAIL, "AH_FAIL_REASON"].value_counts()
        logger.warning(f"[v85] 시간외 수집 실패 {counts[STATUS_FETCH_FAIL]}/{len(out)}종목: "
                       f"{_reasons.to_dict()}")
    _parse_fail = out[(out["AH_STATUS"] == STATUS_FETCH_FAIL) & (out["AH_FAIL_REASON"] == "PARSE")]
    if len(_parse_fail):
        # 형식 변화 감지 — 첫 운영 배치에서 눈에 띄어야 한다(원문은 AH_PRICE_STR 에 남아 있음)
        logger.warning(f"[v85] overPrice 형식 해석 불가 {len(_parse_fail)}종목 — 예: "
                       f"{_parse_fail['AH_PRICE_STR'].head(3).tolist()}")
    _close_bad = out["NAVER_CLOSE"].isna() & ~out["NAVER_CLOSE_STR"].map(_is_placeholder)
    if _close_bad.any():
        logger.warning(f"[v85] closePrice 형식 해석 불가 {int(_close_bad.sum())}종목 — 예: "
                       f"{out.loc[_close_bad, 'NAVER_CLOSE_STR'].head(3).tolist()}")
    if n_date_mismatch:
        _mis = out["AH_PRINT_DATE_MATCH"].astype(str) == "False"
        logger.warning(f"[v85] 시간외 체결 날짜 ≠ {trade_ymd} {n_date_mismatch}종목 → OK 아님·미확정 "
                       f"(예: {out.loc[_mis, 'AH_TRADED_AT'].head(3).tolist()})")

    dated_name = f"aftermarket_prices_{trade_ymd}{f'_{tag}' if tag else ''}.csv"
    dated_path = os.path.join(out_dir, dated_name)
    latest_path = os.path.join(out_dir, latest_name)
    # 확정 보관본 보호 — 다음날 수동 재실행(NEXT_SESSION)·Naver 장애 실행이 확정 행이 더 많은
    # 기존 날짜별 파일을 덮지 않게. latest 는 항상 이번 실행 결과.
    old_final = _existing_final_count(dated_path)
    write_dated = old_final is None or n_final >= old_final
    if not write_dated:
        logger.warning(f"[v85] 기존 날짜별 시간외 보관본 유지 — 확정 {old_final}행 > 이번 {n_final}행 "
                       f"(창 {out['FETCH_WINDOW'].mode().iloc[0] if len(out) else '-'}): "
                       f"{dated_path} (latest 만 갱신)")
    try:
        if write_dated:
            _atomic_write_csv(out, dated_path)
        _atomic_write_csv(out, latest_path)
    except Exception as e:
        logger.warning(f"[v85] 시간외 보관 파일 쓰기 실패: {e}")
        return _summary(len(out), None, None, trade_ymd, f"write_fail: {e}", counts,
                        n_final, krx_src, monotonic_fn() - t0)
    elapsed = monotonic_fn() - t0
    logger.info(f"[v85] 시간외 보관 {len(out)}행 {counts} 확정 {n_final} "
                f"({elapsed:.0f}s) → {dated_path}{'' if write_dated else ' (기존 유지)'}")
    return _summary(len(out), dated_path, latest_path, trade_ymd, None, counts,
                    n_final, krx_src, elapsed, dated_written=write_dated)


def fetch_after_market_prices_sidecar(csv_path, sidecar_path, snap_path=None, trade_ymd=None):
    """[v20.6.4 호환 래퍼] recommend CSV 원본 불변. 시간외 가격은 sidecar CSV 에 저장.

    [v85] sidecar_path 와 같은 폴더에 aftermarket_prices_{ymd}.csv 도 쓴다.
    trade_ymd 를 안 주면 recommend 의 기준일/PRICE_ASOF 에서 찾는다.
    snap_path 는 무시한다 — 스냅샷을 더 이상 덮어쓰지 않는다.
    반환: 시간외 체결가가 있는(AH_STATUS == OK) 종목 수.
    """
    if snap_path:
        logger.info(f"[v85] snap_path 무시 — price_snapshot 은 더 이상 덮어쓰지 않음: {snap_path}")
    out_dir = os.path.dirname(os.path.abspath(sidecar_path))
    s = build_aftermarket_archive(csv_path, trade_ymd, out_dir,
                                  latest_name=os.path.basename(sidecar_path))
    return int(s.get("n_ok", 0))
