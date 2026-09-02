# -*- coding: utf-8 -*-
"""investor_flow_full.py — 투자자별 수급을 전종목으로 [v79]

## 왜

기존 수급(`prefetch_flow`)은 KIS의 **순매수 상위 랭킹** 엔드포인트라 하루
30종목만 온다. 1,200종목 유니버스의 횡단면 특징으로는 쓸 수 없다 —
승자 프로파일 루프(v78)의 flow_* 특징이 NaN으로 남아 있는 이유다.

pykrx의 전종목 순매수(`get_market_net_purchases_of_equities`)는 시장×투자자
조합당 1호출로 전 종목이 온다. 하루 4호출(KOSPI·KOSDAQ × 외국인·기관합계).
과거 날짜도 조회되므로 백필도 같은 함수로 한다.

## 저장 형식

data/flow_full_{ymd}.parquet — 종목코드 · frg_eok(외인 순매수, 억) ·
inst_eok(기관합계 순매수, 억). 단위는 **억 원**으로 저장 시점에 확정한다
(v72의 교훈: 원/억 혼용이 가짜 신호를 만든다 — tv60 사건, p=0.0009가
단위 수정 후 p=0.81로 소멸).

## 주의

- 이 컨테이너(개발 환경)는 외부망이 막혀 있다. 실호출은 배치(GitHub Actions)
  환경에서만 성공한다 — 여기서의 검증은 저장/적재 왕복과 단위 계약뿐이다.
- 실패해도 배치를 죽이지 않는다. 수급은 있으면 좋고 없으면 NaN이다.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from typing import List, Optional

import pandas as pd

logger = logging.getLogger("investor_flow_full")

MARKETS = ("KOSPI", "KOSDAQ")
#: pykrx 투자자 명칭 → 저장 컬럼
INVESTORS = {"외국인": "frg_eok", "기관합계": "inst_eok"}
FILE_FMT = "flow_full_{ymd}.parquet"
#: 순매수 '거래대금' 컬럼 — pykrx가 원 단위로 준다.
_NET_COL = "순매수거래대금"
_WON_PER_EOK = 1e8


def fetch_day(ymd: str) -> Optional[pd.DataFrame]:
    """전종목 순매수 1일치. 네트워크 불가·응답 이상이면 None (예외 안 던짐)."""
    try:
        from pykrx import stock as _krx
    except Exception as e:
        logger.warning("[v79] pykrx 임포트 실패: %s", e)
        return None
    # 1.0.51: 둘 다 있고 시그니처가 같다. _by_ticker가 정식 이름이다.
    _fn = getattr(_krx, "get_market_net_purchases_of_equities_by_ticker", None) \
        or getattr(_krx, "get_market_net_purchases_of_equities", None)
    if _fn is None:
        logger.warning("[v79] pykrx에 순매수 함수가 없다")
        return None
    frames = []
    for market in MARKETS:
        for inv, col in INVESTORS.items():
            try:
                d = _fn(ymd, ymd, market, inv)
            except Exception as e:
                logger.warning("[v79] %s %s %s 조회 실패: %s", ymd, market, inv, e)
                return None          # 부분 데이터는 저장하지 않는다 — 반쪽 수급은 편향
            if d is None or len(d) == 0 or _NET_COL not in d.columns:
                logger.warning("[v79] %s %s %s 응답 비었음", ymd, market, inv)
                return None
            f = d.reset_index()
            code_col = "티커" if "티커" in f.columns else f.columns[0]
            f = f[[code_col, _NET_COL]].rename(columns={code_col: "종목코드", _NET_COL: col})
            f["종목코드"] = f["종목코드"].astype(str).str.zfill(6)
            f[col] = pd.to_numeric(f[col], errors="coerce") / _WON_PER_EOK   # 원 → 억
            frames.append(f.set_index("종목코드"))
    out = None
    for f in frames:
        out = f if out is None else out.combine_first(f)
    out = out.reset_index()
    # 같은 컬럼이 시장별로 두 프레임에 나뉘어 왔으므로 합집합이 곧 전종목이다.
    return out


def save_day(data_dir: str, ymd: str, df: pd.DataFrame) -> str:
    p = os.path.join(data_dir, FILE_FMT.format(ymd=ymd))
    df.to_parquet(p, index=False)
    return p


def collect(data_dir: str, ymd: str) -> Optional[str]:
    """야간 배치 진입점 — 이미 있으면 스킵(멱등), 성공 시 경로."""
    p = os.path.join(data_dir, FILE_FMT.format(ymd=ymd))
    if os.path.exists(p):
        return p
    df = fetch_day(ymd)
    if df is None or df.empty:
        return None
    return save_day(data_dir, ymd, df)


#: 야간 배치에서 따라잡을 최근 세션 수. 9/1 21:40 KST 실측: 당일 응답이 비었다
#: (KRX 집계 지연 또는 야간 차단). 다음 날 배치가 빠진 날을 채운다.
CATCHUP_SESSIONS = 5


def collect_recent(data_dir: str, sessions: List[str], n: int = CATCHUP_SESSIONS) -> dict:
    """최근 n개 세션 중 없는 날을 채운다. 반환: {ymd: 경로 or None}."""
    out = {}
    for y in sorted(sessions)[-n:]:
        out[y] = collect(data_dir, y)
    return out


def have_days(data_dir: str) -> List[str]:
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, "flow_full_2*.parquet"))):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if m:
            out.append(m.group(1))
    return out


def line(path: Optional[str], ymd: str, recent: Optional[dict] = None,
         data_dir: Optional[str] = None) -> str:
    dd = data_dir or (os.path.dirname(path) if path else None)
    total = len(have_days(dd)) if dd else 0
    filled = [y for y, p in (recent or {}).items() if p and y != ymd]
    tail = f" (누적 {total}일" + (f", 따라잡기 {len(filled)}일" if filled else "") + ")"
    if not path:
        return f"수급 전종목 {ymd} — 당일 응답 없음, 다음 배치에 따라잡기" + tail
    try:
        n = len(pd.read_parquet(path))
    except Exception:
        n = -1
    return f"수급 전종목 {ymd} — {n:,}종목 저장" + tail
