# -*- coding: utf-8 -*-
"""quiet_lane_track.py — v73 레인의 실전 성적을 매일 쌓는다 [v80]

v73(조용한 각성 레인)은 103일 백테스트 비용후 +3.05%(HAC p=0.0117)로 살아남은
유일한 신호다. 그러나 같은 데이터를 뒤져서 찾은 신호라 백테스트는 탐색의
산물일 수 있다 — **발견 이후(2026-08-28~) 표본만이 오염되지 않은 증거**다.

이 모듈은 매일 저장되는 quiet_breakout_{ymd}.json의 픽 5종목에 대해 SSOT
실현수익(진입 t+1 시가 · -8% 장중 손절 · t+5 종가)을 계산해 누적한다.
20 거래일이 차면 사용자가 판정한다(코드가 승격하지 않는다).

비용 0.51%(수수료 0.015%×2 + 거래세 0.18% + 슬리피지 0.3%)는 v73 검정과
같은 값을 뺀다 — 백테스트와 같은 잣대로 비교해야 한다.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from services.pick_history import HOLD_DAYS, _realized   # SSOT — 복제 금지
from services.winner_profile import _load_code_history, _load_universe_ohlcv, _realized_full_window

logger = logging.getLogger("quiet_lane_track")

#: 실전 검증 목표 거래일 — v73 문서와 동일.
TARGET_DAYS = 20
#: 왕복 비용(%) — v73 검정과 같은 값.
COST_PCT = 0.51
#: 백테스트 참고치 (v73 · 전체 103일 · 비용후).
BACKTEST_REF = {"ret_pct": 3.05, "hac_p": 0.0117, "days": 103}
#: 레인 실전 1일차 — 이 날짜 이전 json은 검증 표본이 아니다(배포 전 드라이런 등).
LIVE_FROM = "20260828"

SUMMARY_NAME = "quiet_lane_track_latest.json"


def _lane_days(data_dir: str) -> List[str]:
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, "quiet_breakout_2*.json"))):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if m and m.group(1) >= LIVE_FROM:
            out.append(m.group(1))
    return out


def _load_picks(data_dir: str, ymd: str) -> List[dict]:
    try:
        with open(os.path.join(data_dir, f"quiet_breakout_{ymd}.json"), encoding="utf-8") as f:
            r = json.load(f)
    except Exception as e:
        logger.warning("[v80] %s 레인 파일 읽기 실패: %s", ymd, e)
        return []
    if not r.get("ok"):
        return []
    return [p for p in (r.get("picks") or []) if p.get("종목코드")]


def measure(data_dir: str):
    """[v85] 레인 픽 한 줄씩 — (레인 날짜 목록, DataFrame[ymd, 종목코드, 종목명, ret, src, closed]).

    src: latest(최신 배치·레인 파일) | history(_load_code_history 폴백) | none(시세 없음).
    closed: 관측 유니버스 달력에서 픽일 뒤 세션이 1+HOLD_DAYS 개 이상(5일 창이 닫혔다).
    build() 의 측정 규칙은 전부 여기 있다 — 요약은 이 표의 집계일 뿐이다.

    [v85] 시세 = 최신 배치 ∪ 최신 레인 캐시(_load_universe_ohlcv — '날짜' 인덱스 버그 수정으로
    레인 행이 이제 들어온다) + 거기 없는 픽 종목은 그 종목을 담은 가장 최근 날짜 파일
    (_load_code_history). 레인은 하루 ~30%씩 바뀌어 최신 파일 둘만으로는 과거 픽 대부분이
    빠졌다(130픽 중 2픽 측정). 그래도 5일 창이 닫힌 뒤까지 어느 파일에도 남지 못한 종목
    (밴드 밖으로 밀려난 종목)은 측정되지 않는다 — 생존 편향은 줄었을 뿐 없어지지 않았다.

    [v85] 폴백 시리즈는 _realized_full_window 로 잰다 — 창(t+1…t+6)이 시리즈 안에 다 있을 때만.
    그냥 _realized 를 쓰면 밴드 이탈로 잘린 시리즈에서 손절(-8%)만 잡혀 손실 쪽으로 기운다.
    """
    days = _lane_days(data_dir)
    px = _load_universe_ohlcv(data_dir)
    by_code: Dict[str, pd.DataFrame] = (
        {c: g.reset_index(drop=True) for c, g in px.groupby("종목코드")} if px is not None else {})
    sessions = (pd.DatetimeIndex(px["Date"].dropna().unique()).sort_values().to_numpy()
                if px is not None and len(px) else np.array([], dtype="datetime64[ns]"))
    picks = [(ymd, p) for ymd in days for p in _load_picks(data_dir, ymd)]
    missing = {str(p["종목코드"]).zfill(6) for _, p in picks} - set(by_code)
    hist: Dict[str, pd.DataFrame] = {}
    if missing and days:
        hist = _load_code_history(data_dir, missing, since=days[0])
    rows = []
    for ymd, p in picks:
        code = str(p["종목코드"]).zfill(6)
        if code in by_code:
            ret, src = _realized(by_code[code], ymd), "latest"
        elif code in hist:
            ret, src = _realized_full_window(hist[code], ymd), "history"
        else:
            ret, src = None, "none"
        # 창이 닫혔는가 — 관측 유니버스 달력에서 픽일 뒤 세션이 1+HOLD_DAYS 개 이상
        n_after = (len(sessions) - int(np.searchsorted(sessions, np.datetime64(pd.Timestamp(ymd)), side="right"))
                   if len(sessions) else 0)
        rows.append({"ymd": ymd, "종목코드": code, "종목명": p.get("종목명", ""),
                     "ret": (None if ret is None else float(ret)), "src": src,
                     "closed": bool(n_after >= 1 + HOLD_DAYS)})
    return days, pd.DataFrame(rows, columns=["ymd", "종목코드", "종목명", "ret", "src", "closed"])


def build(data_dir: str) -> dict:
    """레인 픽 전수 × SSOT 실현수익 → 누적 요약 (측정 규칙은 measure() docstring).

    [v85] 측정 커버리지(창이 닫힌 픽 중 측정된 비율)와 '픽 과반이 측정된 날' 수도 남기고,
    '표본 충족' 판정은 과반 측정일이 TARGET_DAYS 를 채울 때만 낸다. 픽 5개 중 1개만 잰 날로
    20일을 채우면 그건 레인의 하루가 아니라 살아남은 종목 하나다 — 측정일만 찼으면 '판정 보류'.
    """
    days, df = measure(data_dir)
    measured = df[df["ret"].notna()].copy()
    daily = (measured.groupby("ymd")["ret"].mean() * 100 - COST_PCT) if len(measured) else pd.Series(dtype=float)
    per_day = (df.assign(m=df["ret"].notna()).groupby("ymd")["m"].agg(["size", "sum"])
               if len(df) else pd.DataFrame(columns=["size", "sum"]))
    majority = per_day[(per_day["sum"] * 2 > per_day["size"])] if len(per_day) else per_day
    closed = df[df["closed"]] if len(df) else df
    n_closed_measured = int(closed["ret"].notna().sum()) if len(closed) else 0
    out = {
        "live_from": LIVE_FROM, "target_days": TARGET_DAYS,
        "days_total": int(len(days)),
        "days_measured": int(daily.shape[0]),
        "days_majority": int(len(majority)),
        "picks_total": int(len(df)), "picks_measured": int(len(measured)),
        "picks_closed": int(len(closed)), "picks_closed_measured": n_closed_measured,
        "coverage_closed": (round(n_closed_measured / len(closed), 4) if len(closed) else None),
        "picks_measured_history": int((measured["src"] == "history").sum()) if len(measured) else 0,
        "avg_ret_pct_after_cost": (float(daily.mean()) if len(daily) else None),
        "win_rate": (float((measured["ret"] > 0).mean()) if len(measured) else None),
        "stop_rate": (float((measured["ret"] <= -0.079).mean()) if len(measured) else None),
        "positive_days": (int((daily > 0).sum()) if len(daily) else 0),
        "daily": [{"ymd": k, "ret_pct": round(float(v), 3),
                   "n_measured": int(per_day.loc[k, "sum"]), "n_picks": int(per_day.loc[k, "size"])}
                  for k, v in daily.items()],
        "backtest_ref": BACKTEST_REF, "cost_pct": COST_PCT,
        "verdict": "판정 전 — 20 거래일 후 사용자 판정",
    }
    if out["days_majority"] >= TARGET_DAYS:
        out["verdict"] = "표본 충족 — 판정 가능 (코드는 승격하지 않는다)"
    elif out["days_measured"] >= TARGET_DAYS:
        cov = out["coverage_closed"]
        out["verdict"] = (f"판정 보류 — 측정일 {out['days_measured']}일이지만 픽 과반이 측정된 날은 "
                          f"{out['days_majority']}/{TARGET_DAYS}일"
                          + (f" · 창 닫힌 픽 측정 {cov * 100:.0f}%" if cov is not None else "")
                          + " (밴드 이탈 종목 미측정 — 생존 편향)")
    return out


def save(data_dir: str, summary: dict) -> None:
    try:
        with open(os.path.join(data_dir, SUMMARY_NAME), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    except OSError as e:
        logger.warning("[v80] 레인 성적 저장 실패: %s", e)


def load(data_dir: str) -> Optional[dict]:
    p = os.path.join(data_dir, SUMMARY_NAME)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[v80] 레인 성적 읽기 실패: %s", e)
        return None


def run_batch(data_dir: str) -> dict:
    s = build(data_dir)
    save(data_dir, s)
    return s


def line(s: Optional[dict]) -> str:
    if not s:
        return ""
    d, t = s.get("days_measured", 0), s.get("target_days", TARGET_DAYS)
    head = f"조용한 각성 레인 실전 — 측정 {d}/{t}일 (기록 {s.get('days_total', 0)}일)"
    # [v85] 측정 커버리지 — 레인은 밴드 이탈이 잦아 창이 닫힌 픽도 상당수가 미측정이다.
    if s.get("picks_closed"):
        head += (f" · 과반 측정일 {s.get('days_majority', 0)} · 창 닫힌 픽 "
                 f"{s.get('picks_closed_measured', 0)}/{s['picks_closed']} 측정")
    if not d:
        return head + " · 첫 5일 창이 아직 안 닫혔습니다"
    return (f"{head} · 비용후 일평균 {s['avg_ret_pct_after_cost']:+.2f}% · "
            f"승률 {s['win_rate'] * 100:.0f}% · 손절 {s['stop_rate'] * 100:.0f}% · "
            f"양수일 {s['positive_days']}/{d} — 백테스트 {s['backtest_ref']['ret_pct']:+.2f}%와 비교")
