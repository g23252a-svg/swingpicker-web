# -*- coding: utf-8 -*-
"""[v64] 휴장일에 배치가 돌아 묵은 가격을 '오늘'로 찍는 것을 잡는다.

■ 무엇이 있었나 (2026-08-17 실측)
  2026-08-15가 토요일이라 8/17(월)은 광복절 대체공휴일 — **휴장일**이다.
  그런데 배치가 돌았고 다음을 냈다:
      기준일        = 20260817
      RUN_STATUS   = OK
      TOP_PICK     = 12건 · PRODUCTION_BUY = 1건 (로킷헬스케어 32,250원)
  그 32,250원은 **8/14 종가**다. 392종목 전부 8/14와 종가가 100% 동일했고,
  OHLCV 캐시에 8/17 거래일 자체가 없다. 시장폭·MA20위 비율·당일 중위등락까지
  8/14와 소수점까지 같았다.

■ 왜 막지 못했나
  `collector.find_latest_valid_date`의 4단계 폴백이 원인이다:
      # 4. 모든 확인이 실패하면(IP차단 등), 에러 내지 말고 '최근 평일'로 강제 진행
  IP 차단에서 배치가 죽지 않게 하려는 의도는 맞다. 그런데 **공휴일에도 같은
  경로를 탄다** — 8/17은 월요일이라 '평일' 조건을 통과하고, 그 뒤 어디에서도
  기준일과 실제 가격일을 비교하지 않았다. 그래서 묵은 데이터가 '오늘'로 나갔다.
  이력 전수: 배치 124일 중 **비거래일 배치 7일**, 그 중 8/17이 유일하게
  TOP_PICK 12건·공식매수 1건을 냈다(나머지는 0~2건).

■ 실제 피해는 이번엔 작았다 (정직하게)
  8/18(다음 거래일) 실제 시가 대비 진입 갭 중위 **-0.05%**, -2% 이하 1/12종목,
  **손절 터치 0/12**. 즉 이번 건은 진입 왜곡이 크지 않았다. 그래서 이 모듈은
  **픽을 죽이지 않는다** — 수익 근거 없이 자금 흐름을 바꾸지 않는다는 원칙
  (v45·v51·v61에서 지켜온 것)에 따른다.
  대신 **날짜가 거짓말하는 것**을 고친다: 가격 기준일을 사실대로 적고,
  세션이 묵었음을 표시하고, 로그에 남긴다. 날짜는 검증 대상이 아니라 사실이다.

■ 하는 일 / 하지 않는 일
  한다:    실제 가격일(PRICE_ASOF) 기록 · SESSION_STALE 표시 · 며칠 묵었는지 ·
           RUN_STATUS 강등(OK → STALE_SESSION) · 배치 로그 한 줄
  하지 않는다: TOP_PICK·PRODUCTION_BUY·켈리 수량을 바꾸지 않는다.
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger("session_freshness")

STALE_COL = "SESSION_STALE"
ASOF_COL = "PRICE_ASOF"
LAG_COL = "PRICE_LAG_SESSIONS"
RUN_STATUS_COL = "RUN_STATUS"
STALE_STATUS = "STALE_SESSION"


def latest_price_ymd(data_dir: str = "data",
                     asof_ymd: Optional[str] = None) -> Optional[str]:
    """수집된 OHLCV에서 **실제 마지막 거래일**을 읽는다 (YYYYMMDD).

    `asof_ymd`를 주면 **그 배치 시점에 보였던 캐시**만 본다. 이게 중요하다 —
    나중에 쌓인 캐시로 판정하면 8/17 배치가 사후적으로는 신선해 보인다
    (8/18 캐시가 8/18 가격을 갖고 있으니까). 배치가 그때 무엇을 보고 있었는지가
    판정 대상이다.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_2*.parquet")))
    if asof_ymd:
        ymd = str(asof_ymd)[:8]
        files = [p for p in files
                 if os.path.basename(p)[12:20] <= ymd]
    if not files:
        return None
    for path in reversed(files[-3:]):        # 최신 몇 개만 보면 충분
        try:
            d = pd.read_parquet(path, columns=["종목코드"])
        except Exception as e:
            logger.warning(f"[v64] OHLCV 캐시 읽기 실패 {path}: {e}")
            continue
        idx = pd.to_datetime(getattr(d, "index", None), errors="coerce")
        idx = idx[idx.notna()] if idx is not None else None
        if idx is None or len(idx) == 0:
            continue
        return str(idx.max().strftime("%Y%m%d"))
    return None


def trading_days(data_dir: str = "data", limit: int = 5,
                 asof_ymd: Optional[str] = None) -> list:
    """최근 거래일 목록 (오래된 순). 달력이 아니라 **실제 데이터**에서 얻는다."""
    files = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_2*.parquet")))
    if asof_ymd:
        files = [p for p in files
                 if os.path.basename(p)[12:20] <= str(asof_ymd)[:8]]
    days: set = set()
    for path in reversed(files[-limit:]):
        try:
            d = pd.read_parquet(path, columns=["종목코드"])
        except Exception:
            continue
        idx = pd.to_datetime(getattr(d, "index", None), errors="coerce")
        if idx is None:
            continue
        days |= {str(x.strftime("%Y%m%d")) for x in idx[idx.notna()]}
    return sorted(days)


def assess(trade_ymd: str, data_dir: str = "data") -> dict:
    """배치 기준일이 실제 가격일과 맞는지. 판단 불가면 ok=False로 남긴다."""
    rep = {"ok": False, "stale": False, "trade_ymd": str(trade_ymd or ""),
           "price_asof": None, "lag_sessions": 0, "is_trading_day": None,
           "note": ""}
    if not trade_ymd:
        rep["note"] = "기준일 없음"
        return rep
    asof = latest_price_ymd(data_dir, asof_ymd=trade_ymd)
    if not asof:
        rep["note"] = "OHLCV 캐시 없음 — 신선도 판단 불가"
        return rep
    rep["ok"] = True
    rep["price_asof"] = asof
    days = trading_days(data_dir, asof_ymd=trade_ymd)
    ymd = str(trade_ymd)[:8]
    rep["is_trading_day"] = (ymd in days) if days else None
    if asof < ymd:
        rep["stale"] = True
        later = [d for d in days if d > asof]
        rep["lag_sessions"] = max(1, len(later))
        rep["note"] = (f"기준일 {ymd}인데 실제 가격은 {asof} — "
                       f"휴장일이거나 수집 실패로 세션이 묵었다")
    return rep


def annotate(df: pd.DataFrame, report: Optional[dict]) -> pd.DataFrame:
    """표시·감사 컬럼만 부여한다. **결정 컬럼은 건드리지 않는다.**"""
    if df is None or len(df) == 0 or not report or not report.get("ok"):
        return df
    out = df
    out[ASOF_COL] = str(report.get("price_asof") or "")
    out[STALE_COL] = int(bool(report.get("stale")))
    out[LAG_COL] = int(report.get("lag_sessions") or 0)
    if report.get("stale") and RUN_STATUS_COL in out.columns:
        cur = out[RUN_STATUS_COL].astype("object")
        # DEGRADED 등 더 나쁜 상태는 덮지 않는다
        out[RUN_STATUS_COL] = cur.where(~cur.isin(["OK", "", None]), STALE_STATUS)
    return out


def line(report: Optional[dict]) -> str:
    """배치 로그용 한 줄. 정상이면 빈 문자열."""
    if not report or not report.get("ok"):
        return report.get("note", "") if report else ""
    if not report.get("stale"):
        return ""
    return (f"세션 묵음 — 기준일 {report['trade_ymd']} · 실제 가격일 "
            f"{report['price_asof']} ({report['lag_sessions']}세션 전) · "
            f"거래일 여부 {report.get('is_trading_day')} · "
            "가격·시장폭이 전 거래일 값이다")


def is_alarming(report: Optional[dict]) -> bool:
    return bool(report and report.get("ok") and report.get("stale"))
