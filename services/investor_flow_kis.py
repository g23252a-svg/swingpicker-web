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

## [v79.3] 단위는 매 실행 자가 검증한다

위 확정의 근거였던 랭킹 API(외국인·기관 매매종목 **가집계**)는 잠정 집계라
같은 날·같은 종목의 값이 종목별 API와 크게 다를 수 있다(9/10 실측: 삼성전자
가집계 +134억 vs 종목별 -1.5조). 외부 기준을 믿는 대신 응답 안의 두 값을
맞춰 본다 — 같은 행의 순매수량(*_ntby_qty, 주) × 종가(stck_clpr, 원)는 단위가
확실하다. 전 종목 행에서 |qty×px(원)| / |tr_pbmn| 의 중위값이 10^6 근처면
백만원, 10^3이면 천원, 1이면 원으로 판정해 억으로 환산하고 unit_note에
비율까지 적는다. 수량 필드가 없으면 종전 확정값(백만원)을 쓴다.
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
#: [v79.3] tr_pbmn 단위 후보 — 원 단위당 배수 (원·천원·백만원).
_UNIT_CANDIDATES = {"원": 1.0, "천원": 1e3, "백만원": 1e6}
_DEFAULT_UNIT = "백만원"


def infer_unit(raw_pbmn: List[float], qty_px_won: List[float]) -> Tuple[str, Optional[float]]:
    """[v79.3] |qty×px(원)| / |tr_pbmn| 의 중위값으로 tr_pbmn 단위를 고른다.

    반환: (단위명, 중위비). 표본이 없거나 비율이 후보 어디에도 가깝지 않으면
    (기본 백만원, None/비율) — 판정 실패도 기록에 남긴다.
    """
    import math
    ratios = []
    for a, b in zip(raw_pbmn, qty_px_won):
        try:
            a = abs(float(a)); b = abs(float(b))
        except Exception:
            continue
        if a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b):
            ratios.append(b / a)
    if not ratios:
        return _DEFAULT_UNIT, None
    ratios.sort()
    med = ratios[len(ratios) // 2]
    best = min(_UNIT_CANDIDATES.items(), key=lambda kv: abs(math.log10(med) - math.log10(kv[1])))
    if abs(math.log10(med) - math.log10(best[1])) > 1.0:      # 10배 넘게 빗나가면 판정 보류
        return _DEFAULT_UNIT, med
    return best[0], med


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


def parse_rows(output: list, unit: str = _DEFAULT_UNIT) -> Tuple[List[dict], str]:
    """KIS output → [{ymd, frg_eok, inst_eok, _raw_frg, _raw_inst, _qtypx_frg, _qtypx_inst}], unit_note.

    unit 은 tr_pbmn 의 단위명(원·천원·백만원). _raw_*/_qtypx_* 는 collect 단계의
    단위 자가 검증용(infer_unit)이며 저장 컬럼이 아니다.
    """
    per_eok = _WON_PER_EOK / _UNIT_CANDIDATES[unit]
    rows, note = [], f"tr_pbmn({unit})/{per_eok:g}"
    for r in output or []:
        ymd = str(r.get("stck_bsop_date", "")).replace("-", "")
        if len(ymd) != 8:
            continue
        px = _num(r.get("stck_clpr"))
        q_frg = _num(r.get("frgn_ntby_qty")) * px if "frgn_ntby_qty" in r else float("nan")
        q_inst = _num(r.get("orgn_ntby_qty")) * px if "orgn_ntby_qty" in r else float("nan")
        if "frgn_ntby_tr_pbmn" in r or "orgn_ntby_tr_pbmn" in r:
            raw_f = _num(r.get("frgn_ntby_tr_pbmn")); raw_i = _num(r.get("orgn_ntby_tr_pbmn"))
            frg = raw_f / per_eok
            inst = raw_i / per_eok
        else:                                   # 대금 필드가 없으면 수량×종가 근사
            raw_f = raw_i = float("nan")
            frg = q_frg / _WON_PER_EOK
            inst = q_inst / _WON_PER_EOK
            note = "ntby_qty×stck_clpr(원)/1e8 근사"
        rows.append({"ymd": ymd, "frg_eok": frg, "inst_eok": inst,
                     "_raw_frg": raw_f, "_raw_inst": raw_i,
                     "_qtypx_frg": q_frg, "_qtypx_inst": q_inst})
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
    raw_all: List[float] = []; qtypx_all: List[float] = []
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
            for k_raw, k_q in (("_raw_frg", "_qtypx_frg"), ("_raw_inst", "_qtypx_inst")):
                raw_all.append(r.get(k_raw, float("nan"))); qtypx_all.append(r.get(k_q, float("nan")))
        if sleep_sec:
            time.sleep(sleep_sec)
    # [v79.3] 단위 자가 검증 — tr_pbmn 을 기본 단위(백만원)로 읽어 두고, 같은 행의
    #   수량×종가(원)와의 중위비로 실제 단위를 판정해 필요하면 전부 다시 환산한다.
    unit, med = infer_unit(raw_all, qtypx_all)
    if unit_note.startswith("tr_pbmn("):
        if med is None:
            unit_note += " · 자가검증 불가(수량 필드 없음) — 기본 단위 사용"
        else:
            unit_note += f" · 자가검증 qty×px/tr_pbmn 중위비 {med:.3g} → {unit}"
        if unit != _DEFAULT_UNIT:
            factor = _UNIT_CANDIDATES[_DEFAULT_UNIT] / _UNIT_CANDIDATES[unit]
            for rows_ in per_day.values():
                for x in rows_:
                    x["frg_eok"] = x["frg_eok"] / factor; x["inst_eok"] = x["inst_eok"] / factor
            unit_note = unit_note.replace(f"tr_pbmn({_DEFAULT_UNIT})", f"tr_pbmn({unit})", 1)
            logger.warning("[v79.3] tr_pbmn 단위가 기본(%s)이 아니라 %s 로 판정 — 재환산 (중위비 %.3g)",
                           _DEFAULT_UNIT, unit, med)
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
