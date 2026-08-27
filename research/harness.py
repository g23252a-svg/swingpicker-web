# -*- coding: utf-8 -*-
"""research/harness.py — 예측력 탐색 공용 하네스 [v72]

## 왜 하네스를 먼저 만드는가

이 세션에서만 방법론 오류를 세 번 냈다.
  1. 일평균 수익을 복리로 곱해 -65%/-80% 같은 값을 만들었다.
  2. 종목행 t검정(p=0.0205)을 유의하다고 봤는데 일평균 페어드는 p=0.86이었다.
     같은 날 픽은 상관되어 있으므로 **행 단위 검정은 자유도를 뻥튀기한다**.
  3. 앵커 손절가가 진입가보다 위에 있는 행에서 '즉시 이익 손절'이 잡혀
     가짜 +20.9%가 생겼는데 p=0.0002라 통과처럼 보였다.

병렬 탐색자가 각자 검정을 짜면 이 오류가 그대로 복제된다.
**측정과 검정은 여기 한 곳에만 있다.** 탐색자는 신호(Series)만 만든다.

## 사용

    from research.harness import load_panel, evaluate, PROTOCOL

    P = load_panel()                      # 캐시됨. 전체 유니버스 × 파생피처
    sig = -P["ret_5d"]                    # 예: 5일 하락폭이 클수록 높은 점수
    r = evaluate(sig, P, name="5일 반전", top_n=10)
    print(r["verdict"])                   # PASS / FAIL + 이유

## 판정 기준 (사전 등록)

`evaluate` 는 아래를 **전부** 통과해야 PASS 를 준다. 하나라도 어기면 FAIL.
  - 일평균 수익 > 0
  - 일평균 페어드 t검정 p < 0.05 (단측 아님 — 양측)
  - 부호뒤집기 순열 p < 0.05
  - 블록 부트스트랩(block=3) CI95 가 0 을 포함하지 않음
  - drop-top-2 후에도 양수 (상위 2일이 전부가 아님)
  - 10% 절사평균 양수
  - IS/OOS 부호 일치
  - 분기 4분할 중 3개 이상 양수
BH-FDR 은 여러 신호를 한 번에 낼 때 `bh_fdr()` 로 따로 건다.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "research", "_panel_cache.parquet")
POOL_CACHE = os.path.join(ROOT, "research", "_pool_cache.parquet")

# ── 실현수익 SSOT (services.pick_reliability 와 동일) ──────────
HOLD_DAYS = 5
STOP_PCT = -0.08
HORIZONS = (1, 3, 5, 10, 20)

#: 알파 엔진 도입 이후. 그 전 배치는 다른 엔진이라 섞지 않는다.
IS_FRAC = 0.6
SEED = 20260827

PROTOCOL = (
    "일평균>0 · paired t p<0.05 · 순열 p<0.05 · 블록부트 CI95 0 제외 · "
    "drop-top2 양수 · 10%절사 양수 · IS/OOS 부호일치 · 분기 3/4 양수"
)


# ══════════════════════════════════════════════════════════════
#  패널 구축
# ══════════════════════════════════════════════════════════════
def _latest_ohlcv() -> str:
    c = sorted(glob.glob(os.path.join(DATA, "ohlcv_cache_*.parquet")))
    c = [f for f in c if "latest" not in os.path.basename(f)]
    if not c:
        raise FileNotFoundError("ohlcv_cache_*.parquet 없음")
    return c[-1]


def _derive_features(px: pd.DataFrame) -> pd.DataFrame:
    """OHLCV만으로 계산되는 피처 — 전 종목·전일자에 존재한다.

    후보풀 밖 종목도 평가하려면 recommend_*.csv 의 106개 컬럼은 못 쓴다.
    (그 파일은 그날 편입된 종목만 담는다.)
    """
    g = px.groupby("종목코드", sort=False)
    c, h, l, o, v = px["종가"], px["고가"], px["저가"], px["시가"], px["거래량"]
    px["tv"] = c * v
    out = px
    for n in (1, 3, 5, 10, 20, 60, 120):
        out[f"ret_{n}d"] = g["종가"].pct_change(n) * 100
    out["gap"] = (o / g["종가"].shift(1) - 1) * 100
    out["intraday"] = (c / o - 1) * 100
    out["hl_range"] = (h - l) / c * 100
    out["clv"] = np.where((h - l) > 0, ((c - l) - (h - c)) / (h - l), 0.0)
    r1 = g["종가"].pct_change()
    for n in (5, 20, 60):
        out[f"vol_{n}d"] = g["종가"].transform(
            lambda s, n=n: s.pct_change().rolling(n).std()) * 100
    for n in (5, 20, 60, 120):
        ma = g["종가"].transform(lambda s, n=n: s.rolling(n).mean())
        out[f"dev_ma{n}"] = (c / ma - 1) * 100
    for n in (20, 60, 120):
        hi = g["고가"].transform(lambda s, n=n: s.rolling(n).max())
        lo = g["저가"].transform(lambda s, n=n: s.rolling(n).min())
        out[f"pos_{n}d"] = np.where(hi > lo, (c - lo) / (hi - lo), np.nan)
        out[f"from_hi{n}"] = (c / hi - 1) * 100
    # ── 거래대금은 **전부 억원 단위**로 통일한다 ──────────────
    # 처음 판본은 tv/tv20/tv60 을 원 단위로 두고 tv_eok 만 억으로 뒀다.
    # 그 결과 내가 `tv60 >= 50` 을 "50억"으로 읽었는데 실제로는 **50원**이라
    # 유니버스 583종목 중 569개가 통과했고, 거래대금 0~14억짜리 종목이
    # '상시 유동' 표본에 섞여 들어갔다. 단위를 섞어 두면 반드시 이 사고가 난다.
    out["tv_eok"] = out["tv"] / 1e8
    out["tv20_eok"] = g.apply(
        lambda d: (d["종가"] * d["거래량"] / 1e8).rolling(20).mean()
    ).reset_index(0, drop=True)
    out["tv60_eok"] = g.apply(
        lambda d: (d["종가"] * d["거래량"] / 1e8).rolling(60).median()
    ).reset_index(0, drop=True)
    out["tv_ratio"] = out["tv_eok"] / out["tv20_eok"]
    out["vol_ratio"] = v / g["거래량"].transform(lambda s: s.rolling(20).mean())
    # RSI14
    d = g["종가"].diff()
    up = g.apply(lambda x: x["종가"].diff().clip(lower=0).rolling(14).mean()
                 ).reset_index(0, drop=True)
    dn = g.apply(lambda x: (-x["종가"].diff().clip(upper=0)).rolling(14).mean()
                 ).reset_index(0, drop=True)
    out["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # 연속 상승/하락일
    sgn = np.sign(r1).fillna(0)
    out["_s"] = sgn
    out["streak"] = g["_s"].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1) * sgn
    out = out.drop(columns=["_s"])
    # 52주 대비 / 아모 (Amihud 비유동성)
    out["amihud"] = (r1.abs() / out["tv_eok"].replace(0, np.nan))
    out["amihud20"] = g.apply(
        lambda x: (x["종가"].pct_change().abs()
                   / (x["종가"] * x["거래량"] / 1e8).replace(0, np.nan)).rolling(20).mean()
    ).reset_index(0, drop=True)
    out = out.drop(columns=["tv"])          # 원 단위 컬럼은 아예 없앤다
    return out


def _realized(px: pd.DataFrame) -> pd.DataFrame:
    """진입 t+1 시가 · 장중 저가로 손절 판정 · t+1+H 종가 청산.

    손절가는 **항상 진입가 아래**다. 위에 두면 진입 첫날 저가가 즉시 닿아
    '이익 손절'이라는 가짜 수익이 생긴다(이번 세션 실사고).
    """
    assert STOP_PCT < 0, "손절폭은 음수여야 한다"
    px = px.sort_values(["종목코드", "Date"])
    out = {}
    for code, gd in px.groupby("종목코드", sort=False):
        o = gd["시가"].to_numpy(float)
        lo = gd["저가"].to_numpy(float)
        cl = gd["종가"].to_numpy(float)
        n = len(gd)
        idx = gd.index.to_numpy()
        for H in HORIZONS:
            r_stop = np.full(n, np.nan)
            r_raw = np.full(n, np.nan)
            for t in range(n):
                e = t + 1
                if e + H >= n:
                    break
                entry = o[e]
                if not np.isfinite(entry) or entry <= 0:
                    continue
                sp = entry * (1.0 + STOP_PCT)
                w_lo = lo[e:e + H + 1]
                exit_c = cl[e + H]
                r_raw[t] = (exit_c - entry) / entry
                r_stop[t] = STOP_PCT if (w_lo <= sp).any() else r_raw[t]
            out.setdefault(f"fwd{H}", []).append(pd.Series(r_stop, index=idx))
            out.setdefault(f"fwd{H}_nostop", []).append(pd.Series(r_raw, index=idx))
    for k, parts in out.items():
        px[k] = pd.concat(parts)
    return px


def build_panel(force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    px = pd.read_parquet(_latest_ohlcv()).reset_index()
    px["Date"] = pd.to_datetime(px["Date"])
    px["종목코드"] = px["종목코드"].astype(str).str.zfill(6)
    px = px.sort_values(["종목코드", "Date"]).reset_index(drop=True)
    px = _derive_features(px)
    px = _realized(px)
    # 시장 · 시총 · 후보풀 편입 여부
    last = pd.read_csv(os.path.join(DATA, "recommend_latest.csv"), dtype={"종목코드": str})
    last["종목코드"] = last["종목코드"].str.zfill(6)
    px["시장"] = px["종목코드"].map(dict(zip(last["종목코드"], last["시장"].astype(str))))
    px["mcap"] = px["종목코드"].map(dict(zip(
        last["종목코드"], pd.to_numeric(last["시가총액(억원)"], errors="coerce"))))
    inpool = {}
    for f in sorted(glob.glob(os.path.join(DATA, "recommend_2*.csv"))):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if not m:
            continue
        d = pd.read_csv(f, usecols=["종목코드"], dtype={"종목코드": str})
        inpool[pd.Timestamp(m.group(1))] = set(d["종목코드"].str.zfill(6))
    px["inpool"] = [c in inpool.get(d, set()) for c, d in zip(px["종목코드"], px["Date"])]
    px["has_batch"] = px["Date"].isin(inpool.keys())
    px = px[px["has_batch"]].copy()
    days = sorted(px["Date"].unique())
    cut = days[int(len(days) * IS_FRAC)]
    px["seg"] = np.where(px["Date"] < cut, "IS", "OOS")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    px.to_parquet(CACHE, index=False)
    return px


def load_panel(force: bool = False) -> pd.DataFrame:
    return build_panel(force=force)


def build_pool_panel(force: bool = False) -> pd.DataFrame:
    """후보풀 행 + recommend_*.csv 의 엔진 피처 106개."""
    if not force and os.path.exists(POOL_CACHE):
        return pd.read_parquet(POOL_CACHE)
    P = load_panel()
    common = None
    files = sorted(glob.glob(os.path.join(DATA, "recommend_2*.csv")))
    for f in files:
        c = set(pd.read_csv(f, nrows=0).columns)
        common = c if common is None else (common & c)
    common = sorted(common)
    parts = []
    for f in files:
        m = re.search(r"(\d{8})", os.path.basename(f))
        if not m:
            continue
        d = pd.read_csv(f, dtype={"종목코드": str}, usecols=common)
        d["종목코드"] = d["종목코드"].str.zfill(6)
        d["Date"] = pd.Timestamp(m.group(1))
        parts.append(d)
    R = pd.concat(parts, ignore_index=True)
    keep = [c for c in P.columns if c not in R.columns] + ["종목코드", "Date"]
    out = R.merge(P[keep], on=["종목코드", "Date"], how="inner")
    # recommend_*.csv 는 배치마다 같은 컬럼에 수치/문자가 섞여 들어온다
    # (예: STRATEGY_SCORE). parquet 로 굳히기 전에 정규화한다 —
    # 조용히 astype(str) 하면 숫자 피처가 문자열이 되어 탐색에서 사라진다.
    for c in out.columns:
        if out[c].dtype != object:
            continue
        num = pd.to_numeric(out[c], errors="coerce")
        out[c] = num if num.notna().mean() >= 0.5 else out[c].astype(str)
    out.to_parquet(POOL_CACHE, index=False)
    return out


# ══════════════════════════════════════════════════════════════
#  검정
# ══════════════════════════════════════════════════════════════
def _perm_p(x: np.ndarray, n: int = 20000, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return 1.0
    s = rng.choice([-1.0, 1.0], size=(n, len(x)))
    return float((np.abs((s * x).mean(1)) >= abs(x.mean())).mean())


def _block_ci(x: np.ndarray, block: int = 3, n: int = 10000,
              seed: int = SEED) -> tuple:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    L = len(x)
    if L < block + 1:
        return (np.nan, np.nan)
    nb = int(np.ceil(L / block))
    st = rng.integers(0, L - block + 1, size=(n, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :L]
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def daily_series(signal: pd.Series, panel: pd.DataFrame, top_n: int = 10,
                 ret_col: str = f"fwd{HOLD_DAYS}",
                 mask: Optional[pd.Series] = None,
                 min_names: int = 3) -> pd.Series:
    """신호 상위 top_n 종목의 그날 평균 실현수익. 인덱스는 날짜.

    top_n=None 이면 mask 를 통과한 전체를 동일가중.
    """
    d = pd.DataFrame({"dt": panel["Date"].values,
                      "sig": pd.to_numeric(signal, errors="coerce").values,
                      "ret": pd.to_numeric(panel[ret_col], errors="coerce").values})
    if mask is not None:
        d = d[np.asarray(mask, bool)]
    d = d.dropna(subset=["sig", "ret"])
    if d.empty:
        return pd.Series(dtype=float)
    if top_n is None:
        g = d.groupby("dt")["ret"].agg(["mean", "size"])
    else:
        d = d.sort_values(["dt", "sig"], ascending=[True, False])
        d = d.groupby("dt").head(top_n)
        g = d.groupby("dt")["ret"].agg(["mean", "size"])
    g = g[g["size"] >= min_names]
    return g["mean"].sort_index()


def assess(v: pd.Series, name: str = "") -> dict:
    """일별 수익 시계열 하나에 사전등록 프로토콜을 전부 건다."""
    x = pd.Series(v).dropna()
    if len(x) < 20:
        return dict(name=name, ok=False, verdict="FAIL", why="일수 부족", days=len(x))
    a = x.to_numpy(float)
    t, p = stats.ttest_1samp(a, 0.0)
    lo, hi = _block_ci(a)
    srt = np.sort(a)[::-1]
    dt2 = srt[2:].mean() if len(srt) > 4 else np.nan
    trim = stats.trim_mean(a, 0.1)
    h = len(a) // 2
    is_m, oos_m = a[:h].mean(), a[h:].mean()
    qs = np.array_split(np.arange(len(a)), 4)
    nq = int(sum(a[ix].mean() > 0 for ix in qs))
    pp = _perm_p(a)
    checks = {
        "mean>0": a.mean() > 0,
        "t p<0.05": p < 0.05,
        "perm p<0.05": pp < 0.05,
        "CI excl 0": np.isfinite(lo) and (lo > 0 or hi < 0),
        "drop-top2>0": np.isfinite(dt2) and dt2 > 0,
        "trim>0": trim > 0,
        "IS/OOS sign": np.sign(is_m) == np.sign(oos_m),
        "quarters>=3": nq >= 3,
    }
    failed = [k for k, ok in checks.items() if not ok]
    return dict(
        name=name, ok=not failed, verdict="PASS" if not failed else "FAIL",
        why="" if not failed else "; ".join(failed),
        days=len(a), mean=float(a.mean()), median=float(np.median(a)),
        win_days=float((a > 0).mean()), t=float(t), p=float(p), perm_p=pp,
        ci_lo=lo, ci_hi=hi, drop_top2=float(dt2) if np.isfinite(dt2) else None,
        trim=float(trim), is_mean=float(is_m), oos_mean=float(oos_m),
        quarters_pos=nq, checks=checks,
    )


def evaluate(signal: pd.Series, panel: pd.DataFrame, name: str = "",
             top_n: int = 10, ret_col: str = f"fwd{HOLD_DAYS}",
             mask: Optional[pd.Series] = None,
             baseline: bool = True) -> dict:
    """신호 하나를 끝까지 평가한다. 절대수익과 (원하면) 초과수익 둘 다."""
    v = daily_series(signal, panel, top_n=top_n, ret_col=ret_col, mask=mask)
    r = assess(v, name=name)
    r["top_n"] = top_n
    r["ret_col"] = ret_col
    if baseline and len(v):
        base = daily_series(pd.Series(0.0, index=panel.index), panel,
                            top_n=None, ret_col=ret_col, mask=mask)
        common = v.index.intersection(base.index)
        if len(common) >= 20:
            ex = assess((v.loc[common] - base.loc[common]), name=name + " (초과)")
            r["excess"] = {k: ex[k] for k in
                           ("mean", "p", "perm_p", "verdict", "why",
                            "is_mean", "oos_mean", "quarters_pos")}
            r["baseline_mean"] = float(base.loc[common].mean())
    return r


def bh_fdr(results: Sequence[dict], q: float = 0.05,
           key: str = "p") -> List[dict]:
    """여러 신호를 한 번에 낼 때의 다중검정 보정. p 오름차순으로 임계 비교."""
    rs = [dict(r) for r in results if np.isfinite(r.get(key, np.nan))]
    rs.sort(key=lambda r: r[key])
    M = len(rs)
    passed = -1
    for i, r in enumerate(rs, 1):
        r["bh_thr"] = q * i / M
        if r[key] <= r["bh_thr"]:
            passed = i
    for i, r in enumerate(rs, 1):
        r["bh_pass"] = i <= passed
    return rs


def fmt(r: dict) -> str:
    if not r.get("days"):
        return f"{r.get('name','?'):<28} {r.get('why','')}"
    return (f"{r['name']:<28} n={r['days']:>3} 일평균 {r['mean']*100:>+6.2f}% "
            f"승률 {r['win_days']*100:>4.1f}% p={r['p']:.4f} perm={r['perm_p']:.4f} "
            f"CI[{r['ci_lo']*100:+.2f},{r['ci_hi']*100:+.2f}] "
            f"dt2 {r['drop_top2']*100:>+5.2f}% IS/OOS {r['is_mean']*100:+.2f}/"
            f"{r['oos_mean']*100:+.2f} Q{r['quarters_pos']}/4  [{r['verdict']}]"
            + (f" ← {r['why']}" if r['why'] else ""))


def eval_signal(expr: str, panel: pd.DataFrame) -> pd.Series:
    """문자열 식을 신호로 평가한다.

    탐색자가 낸 결과를 **내가 직접 다시 돌려보기** 위한 창구다.
    숫자를 믿지 않고 식을 믿는다 — 같은 식을 같은 하네스로 재현할 수 없으면
    그 결과는 없는 것으로 친다.

    식 안에서 쓸 수 있는 것: P(패널), np, pd, 그리고 P의 컬럼들.
    """
    env = {"P": panel, "np": np, "pd": pd, "stats": stats}
    for c in panel.columns:
        if isinstance(c, str) and c.isidentifier():
            env[c] = panel[c]
    out = eval(expr, {"__builtins__": {}}, env)  # noqa: S307 — 연구용, 신뢰 경계 안
    s = pd.Series(out, index=panel.index) if not isinstance(out, pd.Series) else out
    return pd.to_numeric(s, errors="coerce")


def xs_rank(s: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """일별 횡단면 백분위 순위 — 날짜 효과를 제거한다."""
    return pd.to_numeric(s, errors="coerce").groupby(panel["Date"].values).rank(pct=True)


def xs_z(s: pd.Series, panel: pd.DataFrame) -> pd.Series:
    """일별 횡단면 표준화."""
    v = pd.to_numeric(s, errors="coerce")
    g = v.groupby(panel["Date"].values)
    return (v - g.transform("mean")) / (g.transform("std") + 1e-9)
