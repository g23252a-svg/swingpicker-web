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

from services.pick_history import _realized          # SSOT — 복제 금지
from services.winner_profile import _load_code_history, _load_universe_ohlcv

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


def build(data_dir: str) -> dict:
    """레인 픽 전수 × SSOT 실현수익 → 누적 요약.

    [v85] 시세 = 최신 배치 ∪ 최신 레인 캐시(_load_universe_ohlcv — '날짜' 인덱스 버그 수정으로
    레인 행이 이제 들어온다) + 거기 없는 픽 종목은 그 종목을 담은 가장 최근 날짜 파일
    (_load_code_history). 레인은 하루 ~30%씩 바뀌어 최신 파일 둘만으로는 과거 픽 대부분이
    빠졌다(130픽 중 2픽 측정). 그래도 5일 창이 닫힌 뒤까지 어느 파일에도 남지 못한 종목
    (밴드 밖으로 밀려난 종목)은 측정되지 않는다 — 생존 편향은 줄었을 뿐 없어지지 않았다.
    """
    days = _lane_days(data_dir)
    px = _load_universe_ohlcv(data_dir)
    by_code: Dict[str, pd.DataFrame] = (
        {c: g.reset_index(drop=True) for c, g in px.groupby("종목코드")} if px is not None else {})
    picks = [(ymd, p) for ymd in days for p in _load_picks(data_dir, ymd)]
    missing = {str(p["종목코드"]).zfill(6) for _, p in picks} - set(by_code)
    if missing and days:
        by_code.update(_load_code_history(data_dir, missing, since=days[0]))
    rows = []
    for ymd, p in picks:
        code = str(p["종목코드"]).zfill(6)
        g = by_code.get(code)
        ret = _realized(g, ymd) if g is not None else None
        rows.append({"ymd": ymd, "종목코드": code, "종목명": p.get("종목명", ""),
                     "ret": (None if ret is None else float(ret))})
    df = pd.DataFrame(rows, columns=["ymd", "종목코드", "종목명", "ret"])
    measured = df[df["ret"].notna()].copy()
    daily = (measured.groupby("ymd")["ret"].mean() * 100 - COST_PCT) if len(measured) else pd.Series(dtype=float)
    out = {
        "live_from": LIVE_FROM, "target_days": TARGET_DAYS,
        "days_total": int(len(days)),
        "days_measured": int(daily.shape[0]),
        "picks_total": int(len(df)), "picks_measured": int(len(measured)),
        "avg_ret_pct_after_cost": (float(daily.mean()) if len(daily) else None),
        "win_rate": (float((measured["ret"] > 0).mean()) if len(measured) else None),
        "stop_rate": (float((measured["ret"] <= -0.079).mean()) if len(measured) else None),
        "positive_days": (int((daily > 0).sum()) if len(daily) else 0),
        "daily": [{"ymd": k, "ret_pct": round(float(v), 3)} for k, v in daily.items()],
        "backtest_ref": BACKTEST_REF, "cost_pct": COST_PCT,
        "verdict": "판정 전 — 20 거래일 후 사용자 판정",
    }
    if out["days_measured"] >= TARGET_DAYS:
        out["verdict"] = "표본 충족 — 판정 가능 (코드는 승격하지 않는다)"
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
    if not d:
        return head + " · 첫 5일 창이 아직 안 닫혔습니다"
    return (f"{head} · 비용후 일평균 {s['avg_ret_pct_after_cost']:+.2f}% · "
            f"승률 {s['win_rate'] * 100:.0f}% · 손절 {s['stop_rate'] * 100:.0f}% · "
            f"양수일 {s['positive_days']}/{d} — 백테스트 {s['backtest_ref']['ret_pct']:+.2f}%와 비교")
