# -*- coding: utf-8 -*-
"""pick_history.py — 이 종목이 몇 번째 추천인지, 앞선 추천은 어떻게 끝났는지 [v74]

## 왜

사용자가 물었다 — "오늘탭에 뜬 추천종목이 2일전과 똑같은데?"
맞았다. 알파 엔진 도입 후 공식픽 13건 중 **아주IB투자가 3회**(8/19·8/24·8/26),
로킷헬스케어 2회. 상위 3종목이 46%를 차지한다. 오늘 화면과 이틀 전 화면의
공식픽이 같은 종목이었다.

배치는 정상이다 — 종가도 켈리 수량도 매일 다시 계산된다. **같은 종목이
계속 이기는 것**이다. 원인은 편입 게이트가 '거래대금이 상위 600에 든 날'을
요구하는데, 반복적으로 거래가 터지는 종목은 계속 그 조건에 걸리기 때문이다.
그리고 그날이 바로 그 종목의 나쁜 날이다(편입일 -2.36% vs 제외일 -0.00%,
`docs/PREDICTIVE_POWER_20260827.md` §8).

재추천 쿨다운은 이미 검정해서 **기각**됐다(일평균 페어드 p=0.86). 즉 "또 뽑지
마라"에는 근거가 없다. 그래서 막지 않는다 — 대신 **보이게 한다.**

측정 가능한 공식픽 7건의 실제 성적: 평균 -5.98%, 승률 14%(1/7), 6건이 -8% 손절.
세 번째로 같은 종목을 사기 전에 앞의 두 번이 어떻게 끝났는지는 알아야 한다.

## 무엇을 보장하지 않는가

- 이 모듈은 **아무것도 막지 않는다.** `PRODUCTION_BUY`·켈리 수량·추천 목록 무변경.
- 과거 성적은 표시용이다. 표본이 한 자릿수라 통계가 아니다.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("pick_history")

CACHE_NAME = "pick_history_latest.json"

#: 실현수익 정의 — services.pick_reliability 와 같은 SSOT.
HOLD_DAYS = 5
STOP_PCT = -0.08
#: 이 이전 배치는 다른 엔진이라 섞지 않는다.
HISTORY_FROM = "20260101"
#: 표시할 과거 등장 최대 개수.
MAX_SHOWN = 4

COL_NTH = "PICK_NTH"              # 이번이 몇 번째 추천인가 (1 = 처음)
COL_PRIOR = "PICK_PRIOR_LINE"     # 앞선 추천의 결과 한 줄
COL_PRIOR_N = "PICK_PRIOR_N"      # 결과를 측정할 수 있었던 과거 건수
COL_PRIOR_AVG = "PICK_PRIOR_AVG"  # 그 평균 (비율)


def _sessions_and_prices(data_dir: str):
    """OHLCV 합집합 — 소멸·이탈 종목도 담긴 쪽을 쓴다."""
    p = os.path.join(data_dir, "ohlcv_union_hl.parquet")
    if os.path.exists(p):
        try:
            d = pd.read_parquet(p)
            d["Date"] = pd.to_datetime(d["Date"])
            d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
            return d
        except Exception as e:
            logger.warning("[v74] 합집합 읽기 실패: %s", e)
    c = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.parquet")))
    c = [f for f in c if "latest" not in os.path.basename(f)]
    if not c:
        return None
    try:
        d = pd.read_parquet(c[-1]).reset_index()
        d["Date"] = pd.to_datetime(d["Date"])
        d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
        return d
    except Exception as e:
        logger.warning("[v74] OHLCV 읽기 실패: %s", e)
        return None


def _realized(g: pd.DataFrame, ymd: str) -> Optional[float]:
    """진입 t+1 시가 · 장중 저가로 -8% 손절 · t+1+5 종가 청산.

    거래정지(시가/고가/저가가 0)는 그날 거래가 없었던 것이므로 손절 판정에서 뺀다.
    저가 0을 그대로 쓰면 무조건 손절 적중으로 잡힌다.
    """
    a = g.sort_values("Date")
    o = pd.to_numeric(a["시가"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(a["저가"], errors="coerce").to_numpy(float)
    cl = pd.to_numeric(a["종가"], errors="coerce").to_numpy(float)
    dts = a["Date"].to_numpy()
    i = np.searchsorted(dts, np.datetime64(pd.Timestamp(ymd)))
    if i >= len(dts) or dts[i] != np.datetime64(pd.Timestamp(ymd)):
        return None
    e = i + 1
    if e >= len(dts):
        return None
    halted = ~((o > 0) & (lo > 0) & (cl > 0))
    if halted[i] or halted[e]:
        return None
    entry = o[e]
    if not np.isfinite(entry) or entry <= 0:
        return None
    sp = entry * (1.0 + STOP_PCT)
    end = min(e + HOLD_DAYS, len(dts) - 1)
    win = slice(e, end + 1)
    live = ~halted[win]
    wl = lo[win][live]
    # 손절이 이미 터졌으면 창이 안 차도 결과는 확정이다.
    # 첫 판본은 창이 찰 때까지 '측정중'으로 뒀는데, 픽 대부분이 손절로
    # 끝나는 마당에 그건 결과를 며칠씩 숨기는 짓이다.
    if wl.size and (wl <= sp).any():
        return STOP_PCT
    if e + HOLD_DAYS > len(dts) - 1:
        return None                      # 아직 안 끝났고 손절도 안 터졌다
    ex = end
    while ex > e and halted[ex]:
        ex -= 1
    return float((cl[ex] - entry) / entry)


def load_pick_log(data_dir: str, upto_ymd: Optional[str] = None) -> pd.DataFrame:
    """과거 배치에서 **공식픽(PRODUCTION_BUY=1)** 이었던 기록."""
    rows: List[dict] = []
    for f in sorted(glob.glob(os.path.join(data_dir, "recommend_2*.csv"))):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if not m:
            continue
        ymd = m.group(1)
        if ymd < HISTORY_FROM or (upto_ymd and ymd >= upto_ymd):
            continue
        try:
            d = pd.read_csv(f, dtype={"종목코드": str})
        except Exception as e:
            logger.warning("[v74] %s 읽기 실패: %s", f, e)
            continue
        if "PRODUCTION_BUY" not in d.columns:
            continue
        pb = d[pd.to_numeric(d["PRODUCTION_BUY"], errors="coerce") == 1]
        for _, r in pb.iterrows():
            rows.append(dict(ymd=ymd,
                             종목코드=str(r["종목코드"]).zfill(6),
                             종목명=str(r.get("종목명", "")),
                             추천매수가=pd.to_numeric(r.get("추천매수가"), errors="coerce")))
    return pd.DataFrame(rows, columns=["ymd", "종목코드", "종목명", "추천매수가"])


def build(data_dir: str, trade_ymd: str) -> Dict[str, dict]:
    """{종목코드: {nth, prior:[{ymd, ret}], avg, line}}"""
    log = load_pick_log(data_dir, upto_ymd=trade_ymd)
    if log.empty:
        return {}
    px = _sessions_and_prices(data_dir)
    by_code = {c: g for c, g in px.groupby("종목코드")} if px is not None else {}
    out: Dict[str, dict] = {}
    for code, g in log.groupby("종목코드"):
        g = g.sort_values("ymd")
        prior = []
        for _, r in g.iterrows():
            pg = by_code.get(code)
            ret = _realized(pg, r["ymd"]) if pg is not None else None
            prior.append(dict(ymd=r["ymd"], ret=ret))
        done = [p["ret"] for p in prior if p["ret"] is not None]
        out[code] = dict(
            nth=len(g) + 1,
            prior=prior[-MAX_SHOWN:],
            n_measured=len(done),
            avg=(float(np.mean(done)) if done else None),
            종목명=str(g["종목명"].iloc[-1]),
        )
        out[code]["line"] = _line(out[code])
    return out


def _line(h: dict) -> str:
    n = h.get("nth", 1)
    if n <= 1:
        return ""
    parts = []
    for p in h.get("prior", []):
        d = f"{p['ymd'][4:6]}/{p['ymd'][6:8]}"
        parts.append(f"{d} {p['ret']*100:+.1f}%" if p.get("ret") is not None
                     else f"{d} 측정중")
    body = " · ".join(parts)
    done = [p["ret"] for p in h.get("prior", []) if p.get("ret") is not None]
    tail = ""
    if done:
        tail = (f" → 앞선 {len(done)}회 평균 {np.mean(done)*100:+.1f}%"
                f"{', 전부 손절' if all(r <= STOP_PCT + 1e-9 for r in done) else ''}")
    return f"이 종목 {n}번째 추천 ({body}){tail}"


def annotate(df: pd.DataFrame, hist: Dict[str, dict]) -> pd.DataFrame:
    """표시 컬럼만 붙인다. 결정 컬럼은 건드리지 않는다."""
    if df is None or df.empty or not hist:
        return df
    code = df["종목코드"].astype(str).str.zfill(6)
    df[COL_NTH] = code.map(lambda c: hist.get(c, {}).get("nth", 1))
    df[COL_PRIOR] = code.map(lambda c: hist.get(c, {}).get("line", ""))
    df[COL_PRIOR_N] = code.map(lambda c: hist.get(c, {}).get("n_measured", 0))
    df[COL_PRIOR_AVG] = code.map(lambda c: hist.get(c, {}).get("avg"))
    return df


def repeat_summary(hist: Dict[str, dict], codes) -> str:
    """오늘 목록 전체에 대한 한 줄. 없으면 빈 문자열."""
    # `codes or []` 는 pandas Series 에서 ValueError 를 낸다
    # ("truth value of a Series is ambiguous"). 이 세션에서 이미 두 번 밟은 함정.
    cs = [] if codes is None else list(codes)
    rep = [hist[c] for c in {str(x).zfill(6) for x in cs}
           if c in hist and hist[c].get("nth", 1) > 1]
    if not rep:
        return ""
    done = [r for h in rep for r in
            [p["ret"] for p in h.get("prior", []) if p.get("ret") is not None]]
    tail = (f" · 그 종목들의 앞선 추천 {len(done)}회 평균 "
            f"{np.mean(done)*100:+.1f}%") if done else ""
    return f"오늘 목록 중 {len(rep)}종목은 재추천이다{tail}."


def stop_reality_line(df: pd.DataFrame) -> str:
    """손절가의 실체 — 설계는 ATR 기반인데 실제로는 -8%에 붙어 있다.

    `docs/STOP_ANCHOR_A_REJECTED_20260827.md` §1: 최근 10배치에서 ATR×2가
    `adaptive_ceil_pct=8.0` 상한에 85~93% 잘리고, 손절폭이 -8%에 붙어있는
    비율이 평균 67%다. 손절선은 매일 종가를 따라 다시 계산된다 —
    싸게 사도 손절폭이 줄지 않는다는 뜻이다.
    """
    if df is None or df.empty:
        return ""
    buy = pd.to_numeric(df.get("추천매수가"), errors="coerce")
    stop = pd.to_numeric(df.get("손절가"), errors="coerce")
    ok = buy.notna() & stop.notna() & (buy > 0)
    if not ok.any():
        return ""
    pct = (stop[ok] - buy[ok]) / buy[ok] * 100
    pinned = float((pct <= -7.5).mean() * 100)
    return (f"손절폭 중위 {pct.median():.1f}% · -7.5%보다 넓은 비율 {pinned:.0f}% — "
            "설계는 ATR 기반이지만 상한 8%에 걸려 사실상 고정폭이고, "
            "손절선은 매일 종가를 따라 다시 계산된다(싸게 사도 손절폭이 줄지 않는다).")
