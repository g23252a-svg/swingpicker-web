# -*- coding: utf-8 -*-
"""market_calendar.py — 오늘 장이 열렸는지 **데이터로** 판정한다 [v84]

## 왜

배치 가드(auto_collect.yml)는 주말과 장 마감 전만 거른다. 평일 휴장일에는 체인이
배치를 돌리고, 시세 소스는 **마지막 거래일의 봉**을 돌려준다. 그 결과 전일과 종가가
100% 같은 CSV가 새 기준일로 저장된다 — 팬텀 배치다. 2026년 실측 10건:

    03-02(삼일절 대체) · 05-01(근로자의날) · 05-05(어린이날) · 05-25(부처님오신날 대체)
    06-03(지방선거) · 07-17(원인 미상, 전일 복제) · 08-17(광복절 대체)
    09-24·09-25(추석) · 10-05(개천절 대체)

피해: 같은 픽이 이틀로 기록돼 픽 이력·그림자 3종·레인 성적이 중복 계산되고
(8/17 로킷헬스케어, 9/25 HD현대마린엔진, 10/5 삼성전자우), 화면은 휴장일을
"추천 기준일"로 보여줬다. alpha_live_report 는 v65에서 가격 기반으로 팬텀을 걸렀지만
나머지 소비자는 recommend_*.csv 의 존재 자체를 거래일로 믿었다.

## 규칙

달력을 하드코딩하지 않는다 — 대체공휴일·임시휴장은 매년 바뀐다. 대신 수집한 OHLCV의
**마지막 봉 날짜가 오늘인 종목 비율**로 판정한다. 거래일이면 거의 전 종목이 오늘 봉을
갖고, 휴장일이면 0에 가깝다. 문턱 50%는 그 둘 사이 어디에 둬도 같다(임계 민감도 없음).
표본이 너무 적으면(수집 실패) 판정을 보류하고 **진행**한다 — 이 가드가 거래일 산출을
막는 쪽으로 틀리면 안 된다.

사후 탐지(`phantom_batch_days`)는 연속 배치의 종가 일치율 ≥ 99% 로 잡는다 — 테스트
트립와이어와 정리 스크립트용.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("market_calendar")

#: 오늘 봉을 가진 종목 비율이 이 이하이면 휴장일로 본다.
HOLIDAY_MAX_SHARE = 0.50
#: 판정에 필요한 최소 종목 수 — 미달이면 판정 보류(진행).
MIN_TICKERS = 30
#: 연속 배치 종가 일치율이 이 이상이면 팬텀 배치.
PHANTOM_MATCH = 0.99
#: 휴장일 스킵 마커 파일 — 워크플로가 후속 스텝을 건너뛰는 근거.
SKIP_MARKER_FMT = "holiday_skip_{ymd}.json"
#: 정리된 팬텀 배치가 들어가는 하위 폴더 (recommend_2*.csv 글롭 밖).
PHANTOM_DIR = "phantom_batches"


def _last_bar_ymd(df) -> Optional[str]:
    """DataFrame 의 마지막 봉 날짜(YYYYMMDD). 인덱스가 날짜이거나 '날짜'/'Date' 컬럼."""
    if df is None or len(df) == 0:
        return None
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.Timestamp(df.index.max()).strftime("%Y%m%d")
        for c in ("날짜", "Date", "date"):
            if c in df.columns:
                d = pd.to_datetime(df[c], errors="coerce").max()
                return None if pd.isna(d) else pd.Timestamp(d).strftime("%Y%m%d")
        d = pd.to_datetime(pd.Series(df.index), errors="coerce").max()
        return None if pd.isna(d) else pd.Timestamp(d).strftime("%Y%m%d")
    except Exception:
        return None


def session_share(ohlcv_map: Dict[str, pd.DataFrame], trade_ymd: str) -> Tuple[float, int]:
    """(오늘 봉을 가진 종목 비율, 판정에 쓴 종목 수)."""
    n = hit = 0
    for df in (ohlcv_map or {}).values():
        y = _last_bar_ymd(df)
        if y is None:
            continue
        n += 1
        if y == str(trade_ymd):
            hit += 1
    return (hit / n if n else float("nan")), n


def is_trading_session(ohlcv_map: Dict[str, pd.DataFrame], trade_ymd: str) -> Tuple[bool, dict]:
    """오늘이 거래일인가. 표본 부족이면 True(진행) + reason='insufficient'."""
    share, n = session_share(ohlcv_map, trade_ymd)
    info = {"trade_ymd": str(trade_ymd), "share_today": (None if n == 0 else round(share, 4)),
            "tickers": n, "threshold": HOLIDAY_MAX_SHARE}
    if n < MIN_TICKERS:
        info["verdict"] = "insufficient"          # 판정 보류 — 거래일 산출을 막지 않는다
        return True, info
    if share <= HOLIDAY_MAX_SHARE:
        info["verdict"] = "holiday"
        return False, info
    info["verdict"] = "trading"
    return True, info


def skip_marker_path(data_dir: str, trade_ymd: str) -> str:
    return os.path.join(data_dir, SKIP_MARKER_FMT.format(ymd=trade_ymd))


def write_skip_marker(data_dir: str, info: dict) -> str:
    import json
    os.makedirs(data_dir, exist_ok=True)
    p = skip_marker_path(data_dir, info.get("trade_ymd", ""))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    return p


def phantom_batch_days(data_dir: str, match: float = PHANTOM_MATCH) -> List[str]:
    """연속 recommend 배치 중 전 배치와 종가가 match 이상 같은 날짜 목록."""
    prev: Optional[Tuple[str, pd.Series]] = None
    out: List[str] = []
    for f in sorted(glob.glob(os.path.join(data_dir, "recommend_2*.csv"))):
        m = re.search(r"recommend_(\d{8})\.csv$", os.path.basename(f))
        if not m:
            continue
        ymd = m.group(1)
        try:
            d = pd.read_csv(f, dtype={"종목코드": str},
                            usecols=lambda c: c in ("종목코드", "종가"), low_memory=False)
        except Exception as e:
            logger.warning("[v84] %s 읽기 실패 — 팬텀 판정 생략: %s", f, e)
            continue
        if "종목코드" not in d.columns or "종가" not in d.columns:
            continue
        cur = pd.to_numeric(d.assign(종목코드=d["종목코드"].astype(str).str.zfill(6))
                            .drop_duplicates("종목코드").set_index("종목코드")["종가"], errors="coerce")
        if prev is not None:
            common = cur.index.intersection(prev[1].index)
            if len(common) >= MIN_TICKERS:
                same = float((cur[common].values == prev[1][common].values).mean())
                if same >= match:
                    out.append(ymd)
        prev = (ymd, cur)
    return out
