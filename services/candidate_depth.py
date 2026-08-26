# -*- coding: utf-8 -*-
"""[v69] 후보 목록을 어디서 끊어야 하는가 — 실측으로 정한다.

■ 왜 필요한가
  화면은 매일 사이징된 후보를 20종목 이상 낸다. 그런데 알파 시대 24일 실측에서
  **사이징된 후보를 전부 사면 일평균 -1.99%**(복리 -21.7% · MDD -21.8%)다.
  우위는 상단에 몰려 있고, 깊이 들어갈수록 유니버스보다 못해진다.

■ 순위 깊이별 실측 (2026-07-14~08-18 · 24일 · 유니버스 평균 +0.78%)
  퍼널 재구성 상위N을 균등 보유했을 때의 **유니버스 대비 초과**:

    N   초과      p      상위2제거   |  시총3000억+ 초과   p     상위2제거
    1  +2.05%p  0.143   +0.70      |   +2.79%p      0.067   +1.47
    2  +1.45%p  0.143   +0.64      |   +1.45%p      0.099   +0.64
    3  +0.41%p  0.607   -0.22      |   +1.16%p      0.099   +0.62
    4  +0.31%p  0.723   -0.64      |   +1.32%p      0.125   +0.46
    5  +0.30%p  0.720   -0.57      |   +0.85%p      0.324   +0.02
    6  +0.01%p  0.993   -0.67      |   +0.60%p      0.414   -0.10
    7  -0.26%p  0.685   -0.86      |   +0.41%p      0.543   -0.25
   11  -0.19%p  0.751   -0.55      |   -0.00%p      0.997   -0.37
   15  -0.47%p  0.458   -0.83      |   -0.09%p      0.898   -0.50

■ 정직하게 — 표본이 늘면서 **약해졌다**
  3일 전 21일 표본에서는 N=1 초과가 p=0.038로 0.05를 넘겼다. 24일로 늘리니
  **p=0.143**(필터 없음) / p=0.067(시총 필터)로 유의성을 잃었다. 즉 지금
  **어떤 N도 통계적으로 검증되지 않았다.** 이 모듈은 '검증된 최적 N'을
  주장하지 않는다 — **초과가 음수로 측정된 깊이를 감추지 않고 잘라낼 뿐**이다.

■ 자르는 규칙 (사전 등록 · 매일 재계산)
  depth        = 초과가 k<=N 전 구간에서 **양수**인 최대 N (바닥 1, 천장 10)
  robust_depth = 거기에 **상위2 제거 후에도 양수**인 최대 N
  둘 다 그날 누적 데이터로 다시 잰다. 상수로 박지 않는다 — 데이터가 바뀌면
  깊이도 바뀌어야 하고, 바뀐 사실이 화면에 남아야 한다.

■ 하지 않는 일
  · 켈리 수량·PRODUCTION_BUY를 바꾸지 않는다. **표시 깊이만** 정한다.
  · 시총 필터를 강제하지 않는다(검정 미통과 — v66 기록). 참고용으로만 잰다.
  · '이 N이 최적'이라고 말하지 않는다. 말할 수 있는 것은 '이 아래는 실측상
    유니버스보다 낫지 않다'뿐이다.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("candidate_depth")

CACHE_NAME = "candidate_depth_latest.json"
MAX_DEPTH = 10          # 천장 — 이보다 깊은 목록은 어떤 경우에도 내지 않는다
MIN_DEPTH = 1           # 바닥 — 측정이 안 되면 가장 보수적으로 1
MIN_DAYS = 8            # 이보다 적으면 깊이를 재지 않는다(폴백 사용)
FALLBACK_DEPTH = 3      # 측정 불가 시. v45 기록(top3 페어드)과 같은 값.
PROBE_N = 15            # 곡선을 어디까지 훑어볼 것인가
UNIVERSE_SAMPLE = 250   # 같은 날 기준선 표본


def _realized(by_code: dict, code: str, ymd: str,
              hold: int, stop: float) -> float:
    a = by_code.get(code)
    if a is None:
        return np.nan
    f = a[a["Date"] > pd.Timestamp(ymd)].head(hold)
    if len(f) < hold:
        return np.nan
    e = float(f.iloc[0]["시가"])
    if not e > 0:
        return np.nan
    for _, r in f.iterrows():
        if (float(r["저가"]) - e) / e <= stop:
            return stop * 100.0
    return (float(f.iloc[-1]["종가"]) / e - 1.0) * 100.0


def _panel(data_dir: str) -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """퍼널 재구성 상위N의 실현수익 패널 + 같은 날 유니버스 기준선."""
    try:
        from services.alpha_live_report import ALPHA_LIVE_FROM, _gate_rank_key
        from services.pick_reliability import (HOLD_DAYS, STOP_PCT_HOLD,
                                               _build_hl_union)
    except Exception as e:
        logger.warning(f"[v69] 의존 모듈 참조 실패: {e}")
        return None, None
    hl = _build_hl_union(data_dir)
    if hl is None or hl.empty:
        return None, None
    hl = hl.copy()
    hl["Date"] = pd.to_datetime(hl["Date"])
    by_code = {c: v.sort_values("Date") for c, v in hl.groupby("종목코드")}
    sessions = set(hl["Date"].dt.strftime("%Y%m%d"))
    rows, uni = [], {}
    for fp in sorted(glob.glob(os.path.join(data_dir, "recommend_2*.csv"))):
        ymd = os.path.basename(fp)[-12:-4]
        if not ymd.isdigit() or ymd not in sessions or ymd < ALPHA_LIVE_FROM:
            continue
        try:
            d = pd.read_csv(fp, encoding="utf-8-sig", dtype={"종목코드": str},
                            low_memory=False)
        except Exception as e:
            logger.warning(f"[v69] CSV 스킵 {ymd}: {e}")
            continue
        if "ALPHA_SCORE" not in d.columns:
            continue
        d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
        gk = _gate_rank_key(d)
        if gk is None or not gk.notna().any():
            continue
        sub = (d.assign(_k=gk).dropna(subset=["_k"])
               .sort_values("_k", ascending=False).head(PROBE_N))
        vs = [_realized(by_code, c, ymd, HOLD_DAYS, STOP_PCT_HOLD)
              for c in d["종목코드"].head(UNIVERSE_SAMPLE)]
        vs = [v for v in vs if np.isfinite(v)]
        if vs:
            uni[ymd] = float(np.mean(vs))
        for rank, (_, r) in enumerate(sub.iterrows(), 1):
            v = _realized(by_code, r["종목코드"], ymd, HOLD_DAYS, STOP_PCT_HOLD)
            if np.isfinite(v):
                rows.append({"ymd": ymd, "rank": rank, "ret": v})
    if not rows or not uni:
        return None, None
    return pd.DataFrame(rows), pd.Series(uni)


def measure(data_dir: str = "data") -> dict:
    """깊이별 초과수익 곡선 + 자를 지점. 매일 다시 잰다."""
    from scipy import stats
    panel, uni = _panel(data_dir)
    if panel is None or panel["ymd"].nunique() < MIN_DAYS:
        n = 0 if panel is None else int(panel["ymd"].nunique())
        return {"ok": False, "reason": f"측정일 {n}일 — {MIN_DAYS}일 미만",
                "depth": FALLBACK_DEPTH, "robust_depth": FALLBACK_DEPTH,
                "n_days": n, "curve": []}
    curve = []
    for N in range(1, PROBE_N + 1):
        sel = panel[panel["rank"] <= N]
        dm = sel.groupby("ymd")["ret"].mean()
        ex = (dm - uni.reindex(dm.index)).dropna()
        if len(ex) < 3:
            continue
        t = stats.ttest_1samp(ex, 0.0)
        srt = np.sort(ex.values)
        curve.append({
            "n": N, "days": int(len(ex)),
            "mean_pct": round(float(dm.mean()), 3),
            "excess_pct": round(float(ex.mean()), 3),
            "t": round(float(t.statistic), 3),
            "p": round(float(t.pvalue), 4),
            "excess_drop_top2": round(float(srt[:-2].mean()), 3) if len(srt) > 2 else None,
            "win_rate": round(float((dm > 0).mean()), 4),
        })
    if not curve:
        return {"ok": False, "reason": "곡선 산출 불가", "depth": FALLBACK_DEPTH,
                "robust_depth": FALLBACK_DEPTH, "n_days": 0, "curve": []}

    def _run(pred) -> int:
        d = 0
        for c in curve:
            if c["n"] > MAX_DEPTH or not pred(c):
                break
            d = c["n"]
        return max(MIN_DEPTH, d)

    depth = _run(lambda c: c["excess_pct"] > 0)
    robust = _run(lambda c: c["excess_pct"] > 0
                  and (c["excess_drop_top2"] is not None and c["excess_drop_top2"] > 0))
    out = {
        "ok": True,
        "asof": str(panel["ymd"].max()),
        "n_days": int(panel["ymd"].nunique()),
        "universe_mean_pct": round(float(uni.mean()), 3),
        "depth": int(depth),
        "robust_depth": int(min(robust, depth)),
        "curve": curve,
        "rule": (f"초과가 1~N 전 구간 양수인 최대 N (바닥 {MIN_DEPTH}·천장 {MAX_DEPTH}) · "
                 "robust는 상위2 제거 후에도 양수"),
        "caveat": ("어떤 N도 통계적으로 검증되지 않았다(최선 p≈0.14). "
                   "이 깊이는 '최적'이 아니라 '이 아래는 실측상 유니버스보다 "
                   "낫지 않다'는 선이다."),
    }
    first = curve[0]
    out["best_p"] = first["p"]
    return out


def save(data_dir: str = "data", trade_ymd: Optional[str] = None) -> dict:
    rep = measure(data_dir)
    names = [CACHE_NAME]
    if trade_ymd:
        names.append(f"candidate_depth_{str(trade_ymd)[:8]}.json")
    for n in names:
        try:
            with open(os.path.join(data_dir, n), "w", encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=1)
        except OSError as e:
            logger.warning(f"[v69] 깊이 리포트 저장 실패 {n}: {e}")
    return rep


def load(data_dir: str = "data") -> Optional[dict]:
    p = os.path.join(data_dir, CACHE_NAME)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"[v69] 깊이 리포트 로드 실패: {e}")
        return None


def effective_depth(report: Optional[dict]) -> int:
    """화면이 실제로 쓸 깊이. 리포트가 없으면 보수적 폴백."""
    if not report or not report.get("ok"):
        return FALLBACK_DEPTH
    d = report.get("depth")
    try:
        d = int(d)
    except (TypeError, ValueError):
        return FALLBACK_DEPTH
    return max(MIN_DEPTH, min(MAX_DEPTH, d))


def depth_line(report: Optional[dict]) -> str:
    """왜 이만큼만 보여주는지 한 줄. 근거 없이 자르지 않는다."""
    if not report or not report.get("ok"):
        return (f"관찰 후보를 상위 {FALLBACK_DEPTH}종목만 표시합니다 — "
                "깊이 측정에 필요한 표본이 아직 부족합니다")
    d = effective_depth(report)
    rb = int(report.get("robust_depth") or d)
    cur = {c["n"]: c for c in report.get("curve", [])}
    nxt = cur.get(d + 1)
    line = (f"관찰 후보 상위 {d}종목만 표시합니다 — 실측 {report['n_days']}일에서 "
            f"{d + 1}위 이하는 유니버스 대비 초과가 "
            + (f"{nxt['excess_pct']:+.2f}%p" if nxt else "0 이하")
            + "였습니다")
    if rb < d:
        c = cur.get(rb)
        if c:
            line += (f" · 이상치를 빼도 우위가 남는 구간은 상위 {rb}종목까지"
                     f"(초과 {c['excess_pct']:+.2f}%p)")
    return line
