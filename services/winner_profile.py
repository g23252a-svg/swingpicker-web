# -*- coding: utf-8 -*-
"""winner_profile.py — 매일 이긴 종목의 특징을 기록하고, 앞으로 쌓인 증거로만 검정한다 [v78]

## 왜

사용자: "매일 매일 승률이 높았던 종목에 특징을 분석해서 그 특성을 테스트 해서
유의미한 승률을 달성하면 본 추천종목개발에 편입하는거지"

맞는 루프다. v73(조용한 각성 레인)이 정확히 이 루프에서 나왔다. 문제는
**과거 데이터에서 승자의 특징을 찾으면 사후편향으로 죽는다**는 것 —
같은 패널에서 129개 신호를 검정해 전부 기각한 게 그 증거다. 그중엔
일별 t검정 p=0.0002짜리도 있었고, 단위 버그가 만든 '4분기 연속 PASS'도 있었다.

그래서 이 모듈은 반대로 간다:

1. **특징 목록을 지금 고정한다** (선등록 — 아래 FEATURES, 버전 관리).
2. 매일 밤, 5일 보유 창이 **막 닫힌** 코호트에 대해 승자/패자의 특징 차이를
   한 줄씩 기록한다. 오늘 등록된 특징은 내일 이후의 데이터로만 평가된다 —
   **구조적으로 표본 외**다.
3. 누적 증거가 선등록 문턱(아래 PROMOTION)을 넘은 특징만 다음 단계로 간다.

## 승격 프로토콜 (선등록 — 이 숫자를 사후에 바꾸면 이 루프 전체가 무효다)

  1단계 (특징 → 후보): 관측 ≥ MIN_DAYS(40)일 · 일별 IC 평균의 HAC(5) p<0.05
        · 전체 특징에 대한 BH-FDR(q=0.05) 생존 · 전/후반 부호 일치
  2단계 (후보 → 그림자 레인): v73과 같은 방식의 표시 전용 레인 20 거래일
  3단계 (레인 → 본선정 편입): 사용자 승인 — 코드가 결정하지 않는다

## 무엇을 바꾸지 않는가

이 모듈은 기록·검정만 한다. `PRODUCTION_BUY`·켈리·추천 목록 무변경.
실측 수익 정의는 SSOT(진입 t+1 시가 · -8% 장중 손절 · t+5 종가)를 그대로 쓴다
(`pick_history._realized` 재사용 — 정의를 복제하면 언젠가 어긋난다).

## 데이터 한계 (정직하게)

배치 OHLCV 캐시는 거래대금 상위 ~600 + v73 레인 601~1200 캐시를 합쳐
약 1,200종목이다. 전체 시장이 아니라 **관측된 유니버스 안의** 횡단면이다.
수급 특징(flow_*)은 v79 전종목 수집이 쌓이기 전까지 NaN으로 기록된다 —
NaN은 검정에서 자동 제외되고, 데이터가 생기면 그날부터 증거가 쌓인다.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.pick_history import _realized  # SSOT 실현수익 — 복제 금지

logger = logging.getLogger("winner_profile")

#: 보유기간 — SSOT와 동일해야 한다.
HOLD_DAYS = 5
#: 특징 스냅샷·실현수익 계산에 필요한 최소 이력.
MIN_HISTORY = 25
#: 하루 코호트 최소 표본 — 이보다 작으면 그날은 기록하지 않는다.
MIN_COHORT = 30
#: 승격 1단계 최소 관측일 (선등록).
MIN_DAYS = 40
#: BH-FDR q (선등록).
FDR_Q = 0.05

LOG_NAME = "winner_profile_log.parquet"
SUMMARY_NAME = "winner_profile_summary.json"

#: 특징 레지스트리 v1 — 2026-08-31 고정. 추가는 버전을 올려서만 한다(삭제 금지).
#:   각 항목: 이름 -> 설명. 계산은 _features()에 있다.
FEATURES: Dict[str, str] = {
    "ret_1d":        "신호일 당일 수익률(%)",
    "ret_5d":        "신호일 기준 5일 수익률(%)",
    "ret_20d":       "신호일 기준 20일 수익률(%)",
    "vol_ratio":     "신호일 거래량 / 직전 20일 평균",
    "volat_20d":     "일수익률 표준편차 20일(%)",
    "ma20_gap":      "종가 / MA20 - 1 (%)",
    "rsi14":         "RSI(14)",
    "hi60_gap":      "종가 / 60일 고점 - 1 (%)",
    "tv_eok":        "신호일 거래대금(억)",
    "tv_rank_pctl":  "관측 유니버스 내 거래대금 백분위(0=최소)",
    "flow_frg_str":  "외인 순매수 / 거래대금 (v79 수급 — 없으면 NaN)",
    "flow_inst_str": "기관 순매수 / 거래대금 (v79 수급 — 없으면 NaN)",
}
REGISTRY_VERSION = "v1-20260831"


# ── 데이터 적재 ────────────────────────────────────────────────────────────
def _load_universe_ohlcv(data_dir: str) -> Optional[pd.DataFrame]:
    """배치 캐시(상위 600) + v73 레인 캐시(601~1200) 합집합."""
    frames = []
    for pat in ("ohlcv_cache_2*.parquet", "quiet_lane_ohlcv_2*.parquet"):
        c = [f for f in sorted(glob.glob(os.path.join(data_dir, pat)))
             if "latest" not in os.path.basename(f)]
        if c:
            try:
                frames.append(pd.read_parquet(c[-1]).reset_index())
            except Exception as e:
                logger.warning("[v78] %s 읽기 실패: %s", c[-1], e)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d[d["Date"].notna()]                      # 레인 캐시의 결측 행 방어
    d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
    # 두 캐시에 같은 종목이 있으면 배치 캐시(먼저 읽은 쪽)를 남긴다.
    d = d.drop_duplicates(subset=["종목코드", "Date"], keep="first")
    for c in ("시가", "고가", "저가", "종가", "거래량"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.sort_values(["종목코드", "Date"])


def _load_flow(data_dir: str, ymd: str) -> Optional[pd.DataFrame]:
    """v79 전종목 수급 (없으면 None — 특징은 NaN으로 기록된다)."""
    p = os.path.join(data_dir, f"flow_full_{ymd}.parquet")
    if not os.path.exists(p):
        return None
    try:
        f = pd.read_parquet(p)
        f["종목코드"] = f["종목코드"].astype(str).str.zfill(6)
        return f
    except Exception as e:
        logger.warning("[v78] 수급 읽기 실패: %s", e)
        return None


def target_session(px: pd.DataFrame, trade_ymd: str) -> Optional[str]:
    """오늘 배치 시점에 5일 창이 막 닫힌 신호일 = trade_ymd 기준 HOLD_DAYS+1 세션 전."""
    sessions = sorted(px["Date"].dropna().dt.strftime("%Y%m%d").unique())
    if trade_ymd not in sessions:
        return None
    i = sessions.index(trade_ymd)
    j = i - (HOLD_DAYS + 1)
    return sessions[j] if j >= 0 else None


# ── 특징 계산 ─────────────────────────────────────────────────────────────
def _rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff().dropna().tail(n * 3)
    if len(d) < n:
        return np.nan
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    r = up.iloc[-1] / dn.iloc[-1] if dn.iloc[-1] > 0 else np.inf
    return float(100 - 100 / (1 + r))


def _features(g: pd.DataFrame, flow_row: Optional[pd.Series]) -> Optional[dict]:
    """신호일까지의 이력만 쓴다 — 미래 참조 금지. g는 신호일이 마지막 행."""
    if len(g) < MIN_HISTORY:
        return None
    c = g["종가"]; v = g["거래량"]
    last = g.iloc[-1]
    if not (last["종가"] > 0 and last["시가"] > 0):     # 신호일 거래정지 제외
        return None
    ret = c.pct_change() * 100
    v20 = v.iloc[-21:-1].mean()
    tv = float(last["종가"] * last["거래량"]) / 1e8
    out = {
        "ret_1d": float(ret.iloc[-1]),
        "ret_5d": float((c.iloc[-1] / c.iloc[-6] - 1) * 100) if len(c) >= 6 else np.nan,
        "ret_20d": float((c.iloc[-1] / c.iloc[-21] - 1) * 100) if len(c) >= 21 else np.nan,
        "vol_ratio": float(last["거래량"] / v20) if v20 and v20 > 0 else np.nan,
        "volat_20d": float(ret.iloc[-20:].std()),
        "ma20_gap": float((c.iloc[-1] / c.iloc[-20:].mean() - 1) * 100),
        "rsi14": _rsi(c),
        "hi60_gap": float((c.iloc[-1] / c.iloc[-60:].max() - 1) * 100),
        "tv_eok": tv,
        "tv_rank_pctl": np.nan,          # 아래에서 횡단면으로 채운다
        "flow_frg_str": np.nan,
        "flow_inst_str": np.nan,
    }
    if flow_row is not None and tv > 0:
        frg = pd.to_numeric(pd.Series([flow_row.get("frg_eok")]), errors="coerce").iloc[0]
        inst = pd.to_numeric(pd.Series([flow_row.get("inst_eok")]), errors="coerce").iloc[0]
        if pd.notna(frg):
            out["flow_frg_str"] = float(frg / tv)
        if pd.notna(inst):
            out["flow_inst_str"] = float(inst / tv)
    return out


# ── 일일 기록 ─────────────────────────────────────────────────────────────
def build_day(data_dir: str, trade_ymd: str) -> Optional[pd.DataFrame]:
    """5일 창이 막 닫힌 코호트의 특징×성과 한 줄 요약. 표본 부족이면 None."""
    px = _load_universe_ohlcv(data_dir)
    if px is None:
        return None
    sig_ymd = target_session(px, trade_ymd)
    if sig_ymd is None:
        return None
    sig_dt = pd.to_datetime(sig_ymd)
    flow = _load_flow(data_dir, sig_ymd)
    flow_by = {r["종목코드"]: r for _, r in flow.iterrows()} if flow is not None else {}

    rows = []
    for code, g in px.groupby("종목코드", sort=False):
        h = g[g["Date"] <= sig_dt]
        if len(h) < MIN_HISTORY or h["Date"].iloc[-1] != sig_dt:
            continue
        feats = _features(h.tail(70), flow_by.get(code))
        if feats is None:
            continue
        ret = _realized(g.reset_index(drop=True), sig_ymd)   # SSOT
        if ret is None:
            continue
        feats.update({"종목코드": code, "ret": float(ret)})
        rows.append(feats)
    if len(rows) < MIN_COHORT:
        return None
    d = pd.DataFrame(rows)
    d["tv_rank_pctl"] = d["tv_eok"].rank(pct=True) * 100
    d["win"] = (d["ret"] > 0).astype(float)

    recs = []
    for f in FEATURES:
        x = pd.to_numeric(d[f], errors="coerce")
        m = x.notna() & d["ret"].notna()
        n = int(m.sum())
        if n < MIN_COHORT:
            ic = np.nan; wl = np.nan
        else:
            ic = float(x[m].rank().corr(d.loc[m, "ret"].rank()))    # Spearman
            sd = x[m].std()
            wl = float((x[m & (d.win > 0)].mean() - x[m & (d.win == 0)].mean()) / sd) \
                if sd and sd > 0 and (m & (d.win > 0)).any() and (m & (d.win == 0)).any() else np.nan
        recs.append({"ymd": sig_ymd, "feature": f, "ic": ic,
                     "winner_gap": wl, "n": n,
                     "win_rate": float(d.loc[m, "win"].mean()) if n else np.nan})
    return pd.DataFrame(recs)


def append_log(data_dir: str, day: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """일일 기록을 로그에 붙인다. 같은 날짜가 이미 있으면 건드리지 않는다(멱등)."""
    p = os.path.join(data_dir, LOG_NAME)
    old = None
    if os.path.exists(p):
        try:
            old = pd.read_parquet(p)
        except Exception as e:
            logger.warning("[v78] 로그 읽기 실패 — 새로 시작: %s", e)
    if old is not None and day["ymd"].iloc[0] in set(old["ymd"]):
        return old, False
    new = pd.concat([old, day], ignore_index=True) if old is not None else day
    new.to_parquet(p, index=False)
    return new, True


# ── 누적 검정 (선등록 문턱) ────────────────────────────────────────────────
def _hac_p(x: np.ndarray, lag: int = HOLD_DAYS) -> Tuple[float, float]:
    """Newey-West 평균 검정 — 5일 창 겹침 때문에 일반 t는 못 쓴다."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan, np.nan
    e = x - x.mean()
    s = float((e * e).mean())
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * float((e[k:] * e[:-k]).mean())
    se = np.sqrt(max(s, 1e-18) / n)
    t = x.mean() / se
    from scipy import stats as _st
    return float(t), float(2 * _st.t.sf(abs(t), df=n - 1))


def evaluate(log: pd.DataFrame) -> List[dict]:
    """특징별 누적 성적 + 승격 1단계 판정. 문턱은 선등록값 — 바꾸지 마라."""
    out = []
    for f, g in log.groupby("feature"):
        ic = pd.to_numeric(g["ic"], errors="coerce").dropna().values
        n = len(ic)
        row = {"feature": f, "days": n, "ic_mean": float(np.mean(ic)) if n else np.nan,
               "t": np.nan, "p": np.nan, "sign_stable": False, "stage1": False}
        if n >= 8:
            row["t"], row["p"] = _hac_p(ic)
            half = n // 2
            row["sign_stable"] = bool(half >= 4 and
                                      np.sign(np.mean(ic[:half])) == np.sign(np.mean(ic[half:])))
        out.append(row)
    ps = [(i, r["p"]) for i, r in enumerate(out)
          if np.isfinite(r["p"]) and out[i]["days"] >= MIN_DAYS]
    if ps:                                            # BH-FDR — 관측 충분한 특징끼리
        ranked = sorted(ps, key=lambda x: x[1])
        m = len(ranked)
        passed = set()
        for k in range(m - 1, -1, -1):
            if ranked[k][1] <= (k + 1) / m * FDR_Q:
                passed = {i for i, _ in ranked[:k + 1]}
                break
        for i in passed:
            if out[i]["sign_stable"]:
                out[i]["stage1"] = True
    return sorted(out, key=lambda r: (r["p"] if np.isfinite(r["p"]) else 9.9))


def run_batch(data_dir: str, trade_ymd: str) -> Optional[dict]:
    """야간 배치 진입점 — 기록 + 누적 검정 + 요약 저장. 실패해도 조용히 None."""
    day = build_day(data_dir, trade_ymd)
    if day is None:
        return None
    log, added = append_log(data_dir, day)
    ev = evaluate(log)
    summary = {"registry": REGISTRY_VERSION, "trade_ymd": trade_ymd,
               "logged_ymd": day["ymd"].iloc[0], "added": added,
               "days_total": int(log["ymd"].nunique()),
               "protocol": {"min_days": MIN_DAYS, "fdr_q": FDR_Q,
                            "stage2": "그림자 레인 20거래일", "stage3": "사용자 승인"},
               "features": ev}
    try:
        with open(os.path.join(data_dir, SUMMARY_NAME), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    except OSError as e:
        logger.warning("[v78] 요약 저장 실패: %s", e)
    return summary


def line(summary: Optional[dict]) -> str:
    if not summary:
        return ""
    ev = summary.get("features") or []
    s1 = [r["feature"] for r in ev if r.get("stage1")]
    best = ev[0] if ev else None
    days = summary.get("days_total", 0)
    head = f"승자 프로파일 — 누적 {days}일 (필요 {MIN_DAYS}일)"
    if s1:
        return f"{head} · ⚑ 1단계 통과: {', '.join(s1)} — 그림자 레인 후보"
    if best and np.isfinite(best.get("p", np.nan)):
        return (f"{head} · 선두 {best['feature']} IC {best['ic_mean']:+.3f}"
                f" (HAC p={best['p']:.3f}) — 아직 후보 없음")
    return f"{head} · 검정 가능 특징 없음"
