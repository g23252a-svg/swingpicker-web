# -*- coding: utf-8 -*-
"""[v68] 과신 방지 캡이 8월 내내 작동하지 않았다 — 선언 승률의 진실 SSOT.

■ 무엇이 있었나
  monotonicity 하드게이트 `declared_vs_realized_gap_15pp`가 2026-08-25에 처음
  실패했다(gap 26.2%p > 15%p). 8/24까지는 통과(12.5%p)였고 main도 같은 날
  같은 이유로 실패한다 — 특정 패치가 만든 것이 아니라 **데이터가 진실을
  드러낸 것**이다.

    declared_wr_top_pick  0.4496   (화면·켈리가 쓰는 선언 승률)
    realized_wr_top_pick  0.1871   (같은 기간 실현, n=30)
    avg_ret_excess_pct   -5.54%

■ 왜 캡이 안 걸렸나 — 두 겹의 고장
  `kelly_calibrator._clip_est_win_rate_to_realized_bins`는 선언 승률을
  `실현 p_win + 14.5%p`로 캡핑하도록 **이미 설계돼 있었다.** 그런데:

  ① **픽이 사는 구간에 표본이 없었다.**
     엔진의 TOP_PICK은 ELITE_SCORE **[0, 50)** 구간에 몰린다
     (2026-08-25 실측: 20종목 중 19종목 · 중위 21.4 · 범위 6.9~58.2).
     그 구간의 winrate_table 표본은:
       08-11 n_raw=2  (sufficient=False, p_win=0.5000 ← 폴백 상수)
       08-14 n_raw=2  (동일)
       08-18 n_raw=2  (동일)
       08-21 n_raw=22 (sufficient=False, p_win=0.2500)
       08-24 n_raw=32 (sufficient=True,  p_win=0.1929)
       08-25 n_raw=43 (sufficient=True,  p_win=0.1669)
     캡은 `sufficient=True` + `n_raw>=30` bin만 신뢰하므로 **8/24 전까지
     커버하는 bin이 아예 없었다.** 그럴 때 이 함수는 **조용히 원본을 그대로
     돌려준다** — 캡이 걸리지 않았다는 사실이 어디에도 기록되지 않는다(v56형).
     그 사이 화면은 선언 승률 38~46%를 적었고, 같은 구간 실현은 16.7~19.3%였다.

  ② **캡이 항상 한 배치 늦다.**
     `compute_est_win_rate`는 `pipeline_calibrate`(라인 677)에서 돌고,
     winrate_table을 다시 만드는 `auto_calibrate`는 `pipeline_finalize`
     (라인 2110)에서 **그 뒤에** 돈다. 즉 캡은 늘 **전날 표**를 읽는다.
     그래서 08-24 배치는 08-21 표(n_raw=22, sufficient=False)를 보고 캡을
     건너뛰었고, 그날 CSV의 선언 승률 28건이 **전부** 당일 표 기준 캡을
     초과했다(최대 +11.8%p).

■ 왜 이것이 손실로 이어지는가 — 켈리가 이 값을 쓴다
  선언 승률은 표시용이 아니라 **사이징 입력**이다(f = p − (1−p)/b).
  당일 표 기준 캡을 적용했다면:
    08-24  선언 0.4501 → 0.3379 · 켈리 f 중위 0.2457 → 0.0910
           **수량 0.40배** · 26종목 중 6종목은 f≤0(진입 불가)
    08-25  선언 0.3320 → 0.3119 · 22종목 중 13종목이 f≤0
  즉 8월 하락 구간에서 포지션이 **2.5배 크게** 잡혔다.

■ 이 모듈이 하는 일
  선언 승률과 **같은 점수 구간의 실현 승률**을 나란히 계산하고, 캡이
  적용됐는지/왜 안 됐는지를 상태로 남긴다. 조용한 미적용을 없앤다.

■ 고장 ③ 캡이 읽는 표와 리포트가 읽는 표가 다르다
  두 파일 모두 `as_of=20260825`라고 적혀 있는데 내용이 전혀 다르다:

    bin        winrate_table_by_ELITE_SCORE_latest   winrate_table_latest
    [0,50)     p_win 0.1622  n_raw  41               0.1669  n_raw  43
    [50,60)    p_win 0.4500  n_raw   0 ← 폴백 상수    0.3750  n_raw   6
    [60,70)    p_win 0.7778  n_raw  16               0.4468  n_raw 121
    [70,80)    p_win 0.3253  n_raw  41               0.3747  n_raw 399
    [80,90)    p_win 0.3636  n_raw   9               0.4075  n_raw 234
    [90,100)   p_win 0.4500  n_raw   0 ← 폴백 상수    0.4000  n_raw  18
    합계 표본              107                                    841

  `_load_winrate_table_cached`는 `by_{method}_latest`를 **먼저** 찾으므로 캡은
  표본 107건짜리 얇은 표를 본다. 반면 monotonicity 리포트는 841건짜리 표를
  쓴다. 그래서 같은 배치에서 두 숫자가 어긋난다. 얇은 표에서는 [50,60)과
  [90,100)이 **n_raw=0에 p_win=0.45 폴백 상수**라, 그 구간 픽은 어떤 경우에도
  캡이 걸리지 않는다.

■ 하지 않는 일
  **EST_WIN_RATE를 바꾸지 않는다.** 켈리 수량도 건드리지 않는다. 캡을 지금
  강제하면 포지션이 0.4배가 되는데, 그 캡의 근거인 p_win이 **하락 구간에서만
  n=32~43으로 막 확보된 값**이라 국면 과적합 위험이 크다. 표시와 경보를
  먼저 정직하게 만들고, 강제 여부는 별도 검정으로 결정한다.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("winrate_truth")

SCORE_COL = "ELITE_SCORE"
DECLARED_COL = "EST_WIN_RATE"

# kelly_calibrator._clip_est_win_rate_to_realized_bins 와 같은 값이어야 한다.
# 어긋나면 이 모듈이 캡과 다른 것을 보고하게 된다 — 테스트가 동일성을 고정한다.
MAX_GAP = 0.145
MIN_N_RAW = 30

COL_DECLARED = "WR_DECLARED"
COL_REALIZED = "WR_REALIZED_BIN"
COL_BIN_N = "WR_BIN_N"
COL_BIN = "WR_BIN_RANGE"
COL_GAP = "WR_GAP_PP"
COL_CAP = "WR_CAP_VALUE"
COL_STATUS = "WR_CAP_STATUS"

STATUS_OK = "WITHIN_CAP"            # 선언이 캡 이하 — 정상
STATUS_OVER = "OVER_CAP"            # 캡이 있는데 선언이 그보다 높다(미적용)
STATUS_NO_BIN = "NO_SUFFICIENT_BIN"  # 커버하는 신뢰 bin이 없다 — 캡 판단 불가
STATUS_UNKNOWN = "UNKNOWN"


# 캡(kelly_calibrator._load_winrate_table_impl)이 실제로 찾는 순서.
# 이 순서를 그대로 흉내내야 '캡이 무엇을 보고 있는가'를 진단할 수 있다.
CAP_TABLE_ORDER = ("winrate_table_by_ELITE_SCORE_latest.json",
                   "winrate_table_latest.json")
# 리포트(daily_briefing)가 쓰는 순서.
REPORT_TABLE_ORDER = ("winrate_table_latest.json",)


def load_table(data_dir: str = "data",
               ymd: Optional[str] = None,
               prefer: str = "report") -> Optional[list]:
    """winrate_table 로드.

    prefer="cap"    — 과신 방지 캡이 실제로 읽는 순서(by_ELITE_SCORE 우선)
    prefer="report" — 리포트가 쓰는 표(기본)
    ymd를 주면 그 날짜 표를 먼저 본다 — 캡이 '전날 표'를 읽는 문제를
    진단하려면 날짜를 지정할 수 있어야 한다.
    """
    names = []
    if ymd:
        names.append(f"winrate_table_{str(ymd)[:8]}.json")
    names += list(CAP_TABLE_ORDER if prefer == "cap" else REPORT_TABLE_ORDER)
    names.append("winrate_table_latest.json")
    seen = set()
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        p = os.path.join(data_dir, n)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"[v68] winrate_table 읽기 실패 {n}: {e}")
            continue
        table = data.get("table", data) if isinstance(data, dict) else data
        if isinstance(table, list) and table:
            return table
    return None


def table_divergence(data_dir: str = "data") -> dict:
    """캡이 읽는 표와 리포트가 읽는 표가 어긋나는지. (고장 ③ 감지)

    둘 다 같은 as_of를 주장하면서 표본 수가 다르면, 캡은 리포트가 보는 것과
    **다른 현실**을 근거로 판단하고 있다는 뜻이다.
    """
    cap = load_table(data_dir, prefer="cap")
    rep = load_table(data_dir, prefer="report")
    out = {"diverged": False, "cap_n": None, "report_n": None, "line": ""}
    if not cap or not rep:
        return out
    cn = sum(float(r.get("n_raw") or 0) for r in cap)
    rn = sum(float(r.get("n_raw") or 0) for r in rep)
    out["cap_n"], out["report_n"] = cn, rn
    zero_bins = sum(1 for r in cap if float(r.get("n_raw") or 0) == 0)
    out["cap_zero_bins"] = zero_bins
    if abs(cn - rn) > 1e-9:
        out["diverged"] = True
        out["line"] = (f"과신 방지 캡이 보는 표본 {cn:.0f}건 vs 리포트가 보는 "
                       f"{rn:.0f}건 — 두 곳이 다른 표를 읽고 있습니다"
                       + (f" · 캡 쪽 {zero_bins}개 구간은 표본 0건(폴백 상수)"
                          if zero_bins else ""))
    return out


def latest_table_ymd(data_dir: str = "data") -> Optional[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "winrate_table_2*.json")))
    files = [f for f in files if "latest" not in os.path.basename(f)]
    if not files:
        return None
    stem = os.path.basename(files[-1]).replace("winrate_table_", "").replace(".json", "")
    return stem if stem.isdigit() else None


def _f(v) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def bin_for(score: Optional[float], table: Optional[list],
            require_sufficient: bool = True) -> Optional[dict]:
    """이 점수를 덮는 bin. 신뢰 조건을 못 맞추면 None(=캡 판단 불가)."""
    s = _f(score)
    if s is None or not table:
        return None
    for r in table:
        lo, hi = _f(r.get("score_lo")), _f(r.get("score_hi"))
        if lo is None or hi is None or not (lo <= s < hi):
            continue
        if require_sufficient:
            if not bool(r.get("sufficient")):
                return None
            n = _f(r.get("n_raw"))
            if n is None or n < MIN_N_RAW:
                return None
        return dict(r)
    return None


def cap_value(score: Optional[float], table: Optional[list]) -> Optional[float]:
    """캡 값. kelly_calibrator와 같은 산식이어야 한다."""
    b = bin_for(score, table)
    if b is None:
        return None
    p = _f(b.get("p_win"))
    if p is None:
        return None
    return min(0.85, max(0.30, p + MAX_GAP))


def assess(row: Mapping[str, Any], table: Optional[list]) -> dict:
    """한 종목의 선언 승률이 같은 구간 실현 대비 어떤 상태인가."""
    score = _f(row.get(SCORE_COL))
    declared = _f(row.get(DECLARED_COL))
    out = {"score": score, "declared": declared, "realized": None, "n_raw": None,
           "bin": None, "cap": None, "gap_pp": None, "status": STATUS_UNKNOWN}
    if declared is None or score is None:
        return out
    b = bin_for(score, table)
    if b is None:
        out["status"] = STATUS_NO_BIN
        loose = bin_for(score, table, require_sufficient=False)
        if loose:
            out["bin"] = f"[{_f(loose.get('score_lo')):.0f}, {_f(loose.get('score_hi')):.0f})"
            out["n_raw"] = _f(loose.get("n_raw"))
        return out
    p = _f(b.get("p_win"))
    out["realized"] = p
    out["n_raw"] = _f(b.get("n_raw"))
    out["bin"] = f"[{_f(b.get('score_lo')):.0f}, {_f(b.get('score_hi')):.0f})"
    out["cap"] = cap_value(score, table)
    if p is not None:
        out["gap_pp"] = (declared - p) * 100.0
    out["status"] = (STATUS_OVER if out["cap"] is not None
                     and declared > out["cap"] + 1e-9 else STATUS_OK)
    return out


def gap_line(row: Mapping[str, Any], table: Optional[list]) -> str:
    """화면 한 줄. 말할 것이 없으면 빈 문자열."""
    a = assess(row, table)
    d = a["declared"]
    if d is None:
        return ""
    if a["status"] == STATUS_NO_BIN:
        n = a["n_raw"]
        n_txt = f"표본 {n:.0f}건" if n is not None else "표본 없음"
        return (f"승률 {d*100:.0f}% — 이 점수 구간{' ' + a['bin'] if a['bin'] else ''}은 "
                f"{n_txt}으로 **검증되지 않았습니다**(과신 방지 캡 적용 불가)")
    r = a["realized"]
    if r is None:
        return ""
    n = a["n_raw"] or 0
    line = f"승률 {d*100:.0f}% · 같은 점수 구간 실측 {r*100:.0f}%(n={n:.0f})"
    if a["gap_pp"] is not None and a["gap_pp"] >= 5.0:
        mult = d / r if r > 0 else float("inf")
        line += f" — **{a['gap_pp']:.0f}%p 높습니다**"
        if np.isfinite(mult):
            line += f"({mult:.1f}배)"
    return line


def annotate(df: pd.DataFrame, table: Optional[list]) -> pd.DataFrame:
    """진단 컬럼 부여. **EST_WIN_RATE와 켈리 수량은 건드리지 않는다.**"""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    res = [assess(r, table) for _, r in out.iterrows()]
    out[COL_DECLARED] = [a["declared"] for a in res]
    out[COL_REALIZED] = [a["realized"] for a in res]
    out[COL_BIN_N] = [a["n_raw"] for a in res]
    out[COL_BIN] = [a["bin"] for a in res]
    out[COL_GAP] = [a["gap_pp"] for a in res]
    out[COL_CAP] = [a["cap"] for a in res]
    out[COL_STATUS] = [a["status"] for a in res]
    return out


def summary(df: pd.DataFrame, table: Optional[list],
            mask: Optional[pd.Series] = None) -> dict:
    """배치 요약 — '선언 승률이 실측보다 얼마나 높은가'."""
    empty = {"n": 0, "line": "", "status_counts": {}, "declared_mean": None,
             "realized_mean": None, "gap_mean_pp": None, "over_cap": 0,
             "no_bin": 0}
    if df is None or len(df) == 0:
        return empty
    sel = df if mask is None else df[mask]
    if not len(sel):
        return empty
    res = [assess(r, table) for _, r in sel.iterrows()]
    dec = [a["declared"] for a in res if a["declared"] is not None]
    rea = [a["realized"] for a in res if a["realized"] is not None]
    gaps = [a["gap_pp"] for a in res if a["gap_pp"] is not None]
    counts = {}
    for a in res:
        counts[a["status"]] = counts.get(a["status"], 0) + 1
    out = {
        "n": len(sel),
        "status_counts": counts,
        "over_cap": counts.get(STATUS_OVER, 0),
        "no_bin": counts.get(STATUS_NO_BIN, 0),
        "declared_mean": float(np.mean(dec)) if dec else None,
        "realized_mean": float(np.mean(rea)) if rea else None,
        "gap_mean_pp": float(np.mean(gaps)) if gaps else None,
        "line": "",
    }
    if out["no_bin"] == len(sel):
        out["line"] = (f"추천 {len(sel)}종목의 선언 승률은 **검증되지 않았습니다** — "
                       f"이 점수 구간에 신뢰할 표본(n≥{MIN_N_RAW})이 없어 "
                       f"과신 방지 캡이 적용되지 않았습니다")
    elif out["gap_mean_pp"] is not None and out["gap_mean_pp"] >= 5.0:
        out["line"] = (f"선언 승률 {out['declared_mean']*100:.0f}% vs "
                       f"같은 점수 구간 실측 {out['realized_mean']*100:.0f}% — "
                       f"평균 **{out['gap_mean_pp']:.0f}%p 과대**"
                       + (f" · 캡 초과 {out['over_cap']}/{len(sel)}종목"
                          if out["over_cap"] else ""))
    return out
