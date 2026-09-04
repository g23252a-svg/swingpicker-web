# -*- coding: utf-8 -*-
"""investor_flow_kis.py — 전종목 투자자별 순매수를 KIS 종목별 API로 [v79.2]

## 왜 KRX가 아니라 KIS인가

v79는 pykrx(KRX data 포털)로 전종목 순매수를 받으려 했다. 9/4 실배치 실측:
따라잡기 5세션(8/28~9/3) 전부 "응답 비었음", 로그에 JSONDecodeError 11건 —
**KRX가 GitHub Actions 러너에 데이터 대신 차단 페이지(HTML)를 준다.** 일주일
전 날짜도 안 되므로 집계 지연이 아니라 소스 자체가 막힌 것이다. OHLCV는
KIS·네이버 폴백이 있어 살고, 수급은 폴백이 없었다.

KIS '주식현재가 투자자'(FHKST01010900)는 종목당 1호출로 **최근 약 30거래일의
일별 외국인·기관·개인 순매수**를 준다. 1,200종목 × 1호출, 초당 20건 제한 →
2~3분. prefetch_flow 워크플로가 이미 KIS 자격과 토큰을 갖고 있으므로 거기서
같은 토큰으로 돈다(토큰 발급은 분당 1회 제한 — 새로 받지 않는다).

## 저장

data/flow_full_{ymd}.parquet — 종목코드 · frg_eok · inst_eok (억 원, v79와 동일
규격이라 winner_profile은 무변경). 응답의 30일 전부를 매번 다시 쓴다 —
당일 행은 잠정치일 수 있고 다음 날 확정치로 덮인다. 승자 프로파일은 신호일
(≥6세션 전)만 읽으므로 항상 확정치를 본다.

## 단위 — 실측으로 확정 (2026-09-04 첫 실전)

KIS 순매수거래대금(*_ntby_tr_pbmn)은 **백만원**이다. 첫 실전에서 원으로 가정해
/1e8 했더니 삼성전자 외인이 0.0002억으로 저장됐다. 원시값을 되돌려 보니
SK하이닉스 -381,452 · 삼성전자 +22,861 — 같은 KIS 계열인 랭킹 API 캐시
(flow_{ymd}.json, 백만원)와 같은 자릿수였다(-445,284 · +4,750). 백만원이면
하이닉스 -3,815억·삼성전자 +229억으로 실제 규모와 맞고, 원이나 천원이면
말이 안 된다. 그래서 /100 → 억. v72의 교훈 그대로 — 단위는 추측이 아니라
교차검증으로 확정하고 unit_note에 적는다.

대금 필드가 없으면 순매수량 × 종가(원)로 근사하고 그건 /1e8 이다.
"""
from __future__ import annotations

import glob
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("investor_flow_kis")

KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"
PATH = "/uapi/domestic-stock/v1/quotations/inquire-investor"
TR_ID = "FHKST01010900"
#: 초당 20건 제한 — 여유를 두고 12건/초.
SLEEP_SEC = 0.085
#: 처음 N종목이 전부 실패하면 자격/차단 문제 — 나머지를 두드리지 않는다.
ABORT_AFTER_CONSECUTIVE_FAIL = 20
FILE_FMT = "flow_full_{ymd}.parquet"
_WON_PER_EOK = 1e8
#: *_ntby_tr_pbmn 은 백만원 — 실측 교차검증으로 확정 (모듈 docstring).
_MILLION_WON_PER_EOK = 100.0


def universe_codes(data_dir: str) -> List[str]:
    """배치 캐시(상위 600) + v73 레인 캐시(601~1200)의 종목코드 합집합."""
    codes: set = set()
    for pat in ("ohlcv_cache_2*.parquet", "quiet_lane_ohlcv_2*.parquet"):
        c = [f for f in sorted(glob.glob(os.path.join(data_dir, pat)))
             if "latest" not in os.path.basename(f)]
        if not c:
            continue
        try:
            d = pd.read_parquet(c[-1], columns=["종목코드"])
            codes |= set(d["종목코드"].astype(str).str.zfill(6))
        except Exception as e:
            logger.warning("[v79.2] %s 코드 읽기 실패: %s", c[-1], e)
    return sorted(codes)


def _num(v) -> float:
    try:
        return float(str(v).replace(",", "").strip() or 0)
    except Exception:
        return float("nan")


def parse_rows(output: list) -> Tuple[List[dict], str]:
    """KIS output → [{ymd, frg_eok, inst_eok}], unit_note."""
    rows, note = [], "tr_pbmn(백만원)/100"
    for r in output or []:
        ymd = str(r.get("stck_bsop_date", "")).replace("-", "")
        if len(ymd) != 8:
            continue
        if "frgn_ntby_tr_pbmn" in r or "orgn_ntby_tr_pbmn" in r:
            frg = _num(r.get("frgn_ntby_tr_pbmn")) / _MILLION_WON_PER_EOK
            inst = _num(r.get("orgn_ntby_tr_pbmn")) / _MILLION_WON_PER_EOK
        else:                                   # 대금 필드가 없으면 수량×종가 근사
            px = _num(r.get("stck_clpr"))
            frg = _num(r.get("frgn_ntby_qty")) * px / _WON_PER_EOK
            inst = _num(r.get("orgn_ntby_qty")) * px / _WON_PER_EOK
            note = "ntby_qty×stck_clpr(원)/1e8 근사"
        rows.append({"ymd": ymd, "frg_eok": frg, "inst_eok": inst})
    return rows, note


def fetch_ticker(session, token: str, app_key: str, app_secret: str,
                 code: str, timeout: int = 10) -> Optional[list]:
    """한 종목의 일별 투자자 순매수. 실패면 None (예외 안 던짐)."""
    headers = {"Authorization": f"Bearer {token}", "appkey": app_key,
               "appsecret": app_secret, "tr_id": TR_ID,
               "content-type": "application/json; charset=utf-8"}
    params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}
    try:
        r = session.get(KIS_BASE_URL + PATH, headers=headers, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json()
        if str(j.get("rt_cd")) != "0":
            return None
        return j.get("output") or []
    except Exception:
        return None


def collect_universe(session, token: str, app_key: str, app_secret: str,
                     codes: List[str], data_dir: str,
                     sleep_sec: float = SLEEP_SEC,
                     abort_after: int = ABORT_AFTER_CONSECUTIVE_FAIL) -> dict:
    """전 종목 순회 → 날짜별 parquet 저장. 반환: 요약 dict."""
    per_day: Dict[str, List[dict]] = {}
    ok = fail = consec = 0
    unit_note = ""
    for i, code in enumerate(codes):
        out = fetch_ticker(session, token, app_key, app_secret, code)
        if out is None:
            fail += 1; consec += 1
            if consec >= abort_after and ok == 0:
                logger.warning("[v79.2] 처음 %d종목 연속 실패 — 자격/차단 의심, 중단", abort_after)
                break
            continue
        consec = 0; ok += 1
        rows, unit_note = parse_rows(out)
        for r in rows:
            per_day.setdefault(r["ymd"], []).append(
                {"종목코드": code, "frg_eok": r["frg_eok"], "inst_eok": r["inst_eok"]})
        if sleep_sec:
            time.sleep(sleep_sec)
    written = []
    os.makedirs(data_dir, exist_ok=True)
    for ymd, rows in sorted(per_day.items()):
        df = pd.DataFrame(rows, columns=["종목코드", "frg_eok", "inst_eok"])
        df.to_parquet(os.path.join(data_dir, FILE_FMT.format(ymd=ymd)), index=False)
        written.append(ymd)
    return {"tickers": len(codes), "ok": ok, "fail": fail,
            "days_written": written, "unit_note": unit_note}


def line(s: dict) -> str:
    d = s.get("days_written") or []
    span = f"{d[0]}~{d[-1]}" if d else "없음"
    return (f"수급 전종목(KIS) — 종목 {s.get('ok', 0)}/{s.get('tickers', 0)} 성공 · "
            f"저장 {len(d)}일 ({span}) · 단위 {s.get('unit_note') or '?'}")
