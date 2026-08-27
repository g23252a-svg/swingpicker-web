# -*- coding: utf-8 -*-
"""research/full_panel.py — 생존편향 없는 전체 유니버스 패널 [v72]

## 무엇이 달라지는가

`research/harness.py` 의 패널은 `ohlcv_cache_*.parquet` 에서 나온다. 그건
엔진의 `top_n=600`(거래대금 상위) 산출물이고 **중도 소멸 종목이 0건**이다.
그 위에서 "거래대금 낮은 종목이 낫다"는 신호가 계속 나왔지만, 저유동 구간이야말로
상장폐지가 몰리는 곳이라 신뢰할 수 없었다.

`data/price_snapshot_YYYYMMDD.csv` 는 다르다 — **매일 2,880종목**의 OHLC를 담고,
합집합 2,923종목에 OHLCV 캐시에 없는 2,330종목이 포함된다. 사라진 종목은
사라진 날 이후 파일에서 빠지므로 **생존편향이 없다**.

대가: **거래량·거래대금 컬럼이 없다.** 유동성 신호는 이 패널로 못 만든다.
대신 "저유동 신호가 생존편향인지"는 이 패널로 판정할 수 있다 —
같은 신호를 두 패널에서 재서 무너지는지 보면 된다.

## 데이터 무결성 (실측)

118개 스냅샷 중 버리는 것:
  - 거래일이 아닌 날 7개
  - **팬텀 세션 9개** — 전일 종가와 99% 이상 동일(휴장일 복사본).
    v65에서 같은 유형을 잡았다.
  - **깨진 파일 2개** (20260703 일치율 6.1%, 20260714 3.7%) — 어느 인접일과도
    맞지 않는다. ohlcv_cache 와 교차검증해서 잡았다.
남는 것 약 100일. 나머지 날은 ohlcv_cache 와 종가가 95%+ 일치한다.

## 거래정지 처리 (실측 5.51%)

`시가=고가=저가=0, 종가만 존재`인 행이 **18,687건(5.51%)** 있다 — 거래정지 종목이다
(금양·카프로·동양생명·본시스템즈 등). 첫 판본은 이걸 거르지 않았고,
손절 판정이 `저가 <= 손절가` 였기 때문에 **저가=0이 무조건 손절 적중**으로 잡혔다.
보유 창에 정지일이 하루라도 끼면 전부 -8%가 됐다.

처방: `halted` 로 표시하고
  - 정지일에는 **진입하지 않는다** (시가가 없으므로 애초에 불가능)
  - 손절 판정에서 정지일을 **건너뛴다** (그날은 거래 자체가 없다)
  - 청산일이 정지일이면 **마지막 유효 종가**를 쓴다
정지가 끝까지 안 풀리면 사실상 소멸이므로 소멸 처리에 맡긴다.

## 상장폐지 처리

보유 창(t+1 ~ t+H) 안에 종목이 사라지면 수익을 어떻게 셀 것인가.
정답이 없으므로 **양쪽을 다 낸다**:
  - `fwdH`        : 마지막 체결가로 청산했다고 본다 (낙관)
  - `fwdH_del100` : 상장폐지를 -100% 로 본다 (비관)
진실은 그 사이에 있다. 한쪽만 보고하는 것은 정직하지 않다.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "research", "_full_panel_cache.parquet")

HORIZONS = (1, 3, 5, 10, 20)
STOP_PCT = -0.08
IS_FRAC = 0.6

#: 교차검증에서 어느 인접일과도 맞지 않은 파일. 원인 불명이라 버린다.
BROKEN_SNAPSHOTS = {"20260703", "20260714"}
#: 전일 종가와 이 비율 이상 같으면 휴장일 복사본으로 본다 (v65와 동일 규율).
PHANTOM_RATIO = 0.99
PHANTOM_MIN_CODES = 50
#: KONEX 는 유동성이 사실상 없어 스윙 대상이 아니다.
EXCLUDE_MARKETS = {"KONEX"}


def _sessions_from_cache() -> Set[pd.Timestamp]:
    c = sorted(glob.glob(os.path.join(DATA, "ohlcv_cache_*.parquet")))
    c = [f for f in c if "latest" not in os.path.basename(f)]
    if not c:
        return set()
    px = pd.read_parquet(c[-1], columns=[])
    px = pd.read_parquet(c[-1]).reset_index()
    return set(pd.to_datetime(px["Date"].unique()))


def load_snapshots(data_dir: str = DATA) -> Dict[str, pd.DataFrame]:
    """유효한 price_snapshot 만 {ymd: DataFrame} 으로."""
    sessions = _sessions_from_cache()
    out: Dict[str, pd.DataFrame] = {}
    prev: Optional[Dict[str, float]] = None
    dropped = {"non_session": [], "phantom": [], "broken": []}
    for f in sorted(glob.glob(os.path.join(data_dir, "price_snapshot_2*.csv"))):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if not m:
            continue
        ymd = m.group(1)
        d = pd.read_csv(f, dtype={"종목코드": str})
        d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
        for c in ("시가", "고가", "저가", "종가"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["종가"])
        d = d[d["종가"] > 0]
        # 거래정지: 종가만 있고 OHL 이 0/결측 — 그날은 사고팔 수 없다.
        d["halted"] = ~((d["시가"] > 0) & (d["고가"] > 0) & (d["저가"] > 0))
        cur = dict(zip(d["종목코드"], d["종가"]))
        if ymd in BROKEN_SNAPSHOTS:
            dropped["broken"].append(ymd)
            prev = cur
            continue
        if sessions and pd.Timestamp(ymd) not in sessions:
            dropped["non_session"].append(ymd)
            prev = cur
            continue
        if prev:
            common = [k for k in cur if k in prev]
            if len(common) >= PHANTOM_MIN_CODES:
                same = sum(1 for k in common if cur[k] == prev[k]) / len(common)
                if same >= PHANTOM_RATIO:
                    dropped["phantom"].append(ymd)
                    prev = cur
                    continue
        out[ymd] = d
        prev = cur
    out["_dropped"] = dropped  # type: ignore[assignment]
    return out


def _derive(px: pd.DataFrame) -> pd.DataFrame:
    """OHLC만으로 계산되는 피처. 거래량이 없으므로 유동성 피처는 없다."""
    g = px.groupby("종목코드", sort=False)
    c, h, l, o = px["종가"], px["고가"], px["저가"], px["시가"]
    for n in (1, 3, 5, 10, 20, 60, 120):
        px[f"ret_{n}d"] = g["종가"].pct_change(n) * 100
    px["gap"] = (o / g["종가"].shift(1) - 1) * 100
    px["intraday"] = (c / o - 1) * 100
    px["hl_range"] = (h - l) / c * 100
    px["clv"] = np.where((h - l) > 0, ((c - l) - (h - c)) / (h - l), 0.0)
    for n in (5, 20, 60):
        px[f"vol_{n}d"] = g["종가"].transform(
            lambda s, n=n: s.pct_change().rolling(n).std()) * 100
    for n in (5, 20, 60, 120):
        ma = g["종가"].transform(lambda s, n=n: s.rolling(n).mean())
        px[f"dev_ma{n}"] = (c / ma - 1) * 100
    for n in (20, 60, 120):
        hi = g["고가"].transform(lambda s, n=n: s.rolling(n).max())
        lo = g["저가"].transform(lambda s, n=n: s.rolling(n).min())
        px[f"pos_{n}d"] = np.where(hi > lo, (c - lo) / (hi - lo), np.nan)
        px[f"from_hi{n}"] = (c / hi - 1) * 100
    up = g.apply(lambda x: x["종가"].diff().clip(lower=0).rolling(14).mean()
                 ).reset_index(0, drop=True)
    dn = g.apply(lambda x: (-x["종가"].diff().clip(upper=0)).rolling(14).mean()
                 ).reset_index(0, drop=True)
    px["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    sgn = np.sign(g["종가"].pct_change()).fillna(0)
    px["_s"] = sgn
    px["streak"] = g["_s"].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1) * sgn
    return px.drop(columns=["_s"])


def _realized(px: pd.DataFrame, last_session: pd.Timestamp) -> pd.DataFrame:  # noqa: C901
    """진입 t+1 시가 · 장중 저가로 -8% 손절 · t+1+H 종가 청산.

    보유 창 안에 종목이 사라지면 두 가지로 센다 — 마지막 체결가 청산(fwdH)과
    상장폐지 -100%(fwdH_del100). 어느 쪽도 진실이라 단정하지 않는다.
    """
    assert STOP_PCT < 0
    px = px.sort_values(["종목코드", "Date"])
    acc: Dict[str, list] = {}
    for code, gd in px.groupby("종목코드", sort=False):
        o = gd["시가"].to_numpy(float)
        lo = gd["저가"].to_numpy(float)
        cl = gd["종가"].to_numpy(float)
        hal = gd["halted"].to_numpy(bool)
        dts = gd["Date"].to_numpy()
        idx = gd.index.to_numpy()
        n = len(gd)
        gone = dts[-1] < np.datetime64(last_session)   # 관측 구간 끝 전에 사라짐
        for H in HORIZONS:
            a = np.full(n, np.nan)
            b = np.full(n, np.nan)
            ns = np.full(n, np.nan)
            dflag = np.zeros(n, bool)
            for t in range(n):
                e = t + 1
                if e >= n:
                    break
                if hal[t] or hal[e]:
                    continue                 # 신호일·진입일이 정지면 매매 불가
                entry = o[e]
                if not np.isfinite(entry) or entry <= 0:
                    continue
                sp = entry * (1.0 + STOP_PCT)
                end = min(e + H, n - 1)
                win = slice(e, end + 1)
                live = ~hal[win]              # 정지일은 손절 판정에서 제외
                w_lo = lo[win][live]
                hit = bool(w_lo.size) and bool((w_lo <= sp).any())
                # 청산일이 정지면 그 이전 마지막 유효 종가로
                ex = end
                while ex > e and hal[ex]:
                    ex -= 1
                truncated = (e + H) > (n - 1)
                raw = (cl[ex] - entry) / entry
                if not truncated or gone:
                    ns[t] = raw                            # 손절 없는 판본
                if hit:
                    a[t] = b[t] = STOP_PCT
                elif not truncated:
                    a[t] = b[t] = raw
                elif gone:
                    dflag[t] = True
                    a[t] = raw                             # 마지막 체결가 청산
                    b[t] = -1.0                            # 상장폐지 -100%
                    ns[t] = -1.0
                # truncated & not gone = 관측 구간 끝 → 측정 불가(NaN)
            acc.setdefault(f"fwd{H}", []).append(pd.Series(a, index=idx))
            acc.setdefault(f"fwd{H}_del100", []).append(pd.Series(b, index=idx))
            acc.setdefault(f"fwd{H}_nostop", []).append(pd.Series(ns, index=idx))
            acc.setdefault(f"fwd{H}_delisted", []).append(pd.Series(dflag, index=idx))
    for k, parts in acc.items():
        px[k] = pd.concat(parts)
    return px


def build(force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    snaps = load_snapshots()
    dropped = snaps.pop("_dropped")  # type: ignore[arg-type]
    parts = []
    for ymd, d in snaps.items():
        d = d.copy()
        d["Date"] = pd.Timestamp(ymd)
        parts.append(d)
    px = pd.concat(parts, ignore_index=True)
    px = px[~px["시장"].astype(str).isin(EXCLUDE_MARKETS)]
    px = px.sort_values(["종목코드", "Date"]).reset_index(drop=True)
    # 정지일의 OHL(0)이 파생 피처를 오염시키지 않게 결측 처리한다.
    # 종가는 남긴다 — 정지 중에도 마지막 체결가는 유효한 정보다.
    for c in ("시가", "고가", "저가"):
        px.loc[px["halted"], c] = np.nan
    px = _derive(px)
    last_session = px["Date"].max()
    px = _realized(px, last_session)
    # 후보풀 편입 여부
    inpool: Dict[pd.Timestamp, Set[str]] = {}
    for f in sorted(glob.glob(os.path.join(DATA, "recommend_2*.csv"))):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if not m:
            continue
        d = pd.read_csv(f, usecols=["종목코드"], dtype={"종목코드": str})
        inpool[pd.Timestamp(m.group(1))] = set(d["종목코드"].str.zfill(6))
    px["inpool"] = [c in inpool.get(d, set()) for c, d in zip(px["종목코드"], px["Date"])]
    px["in_universe"] = px["Date"].isin(inpool.keys())
    # 소멸 여부
    lastd = px.groupby("종목코드")["Date"].transform("max")
    px["delisted"] = lastd < last_session
    days = sorted(px["Date"].unique())
    cut = days[int(len(days) * IS_FRAC)]
    px["seg"] = np.where(px["Date"] < cut, "IS", "OOS")
    px.attrs["dropped"] = dropped
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    px.to_parquet(CACHE, index=False)
    return px


def load(force: bool = False) -> pd.DataFrame:
    return build(force=force)


def summary(px: Optional[pd.DataFrame] = None) -> str:
    px = load() if px is None else px
    nd = int(px.loc[px["delisted"], "종목코드"].nunique())
    return (f"[v72] 생존편향 없는 패널 {len(px):,}행 · {px['종목코드'].nunique():,}종목 · "
            f"{px['Date'].nunique()}일 ({px['Date'].min().date()}~{px['Date'].max().date()}) · "
            f"기간 중 소멸 {nd}종목 · 후보풀 편입률 {px['inpool'].mean()*100:.1f}%")
