# -*- coding: utf-8 -*-
"""
backtest_validation.py  (v3.9.2 — 2026-05-16)
═══════════════════════════════════════════════════
v1과 달리 이번 버전은 3가지를 실제로 구현한다.

  (1) 일자별 Top3 실제 백테스트
      매일 compute_elite_labels → pick_top3() 을 돌려 실제 3종목을 뽑고,
      그 종목들만 horizon 10일 추적하여 성능 집계. (v3.7.10 이후 horizon 20→10 축소)

  (2) OHLC 기반 TP1/TP2/TP3 + Stop 터치 판정  ← v3.8.3 확장
      ohlcv_cache_*.parquet 에서 장중 고가/저가를 읽어 정확한 터치 판정.
      체결 이후 전체 horizon을 스캔해 TP ladder + 극값(max_high/min_low/max_close)
      을 동시에 기록한다. 첫 터치 outcome(WIN/LOSS/OPEN)의 의미는 v3.7.x와 불변.
      OHLC 없는 종목은 종가 폴백 (명시적 기록).

  (3) 최강 라벨 튜닝 그리드서치 (--tune 옵션)
      평균/밸런스/갭/RR 임계값을 여러 조합으로 돌려 최적 조합 실측.

[v3.9.2 — PRE_ENTRY_RISK shadow backtest]
  · 5/15 폭락에서 손실 4건 attribution 결과, 모든 손실 종목이 STRUCT 70~85
    + VWAP_GAP > 8 조합 또는 VWAP_GAP > 15 과열 신호를 갖고 있었음 확인.
  · compute_pre_entry_risk_shadow(): 진입 시점의 위험 플래그(STRUCT 70~85 &
    VWAP_GAP>8 = RED / STRUCT<90 & VWAP_GAP>15 = ORANGE)에 해당하는 종목을
    Top3 후보 풀에서 제외했을 때 성과 차이를 측정. 4개 룰 비교:
      Rule A: STRUCT 70~85 단독 제외 (= 기존 struct_risk_shadow와 동일, 비교 기준선)
      Rule B: STRUCT 70~85 AND VWAP_GAP>8 제외 (RED만)
      Rule C: STRUCT<90 AND VWAP_GAP>15 제외 (ORANGE만)
      Rule D: B + C 합집합 (RED + ORANGE)
  · 목표: 구성변경률은 낮으면서 손실 회피 효과는 큰 룰 발견.
  · **baseline / 추천 / 매수가 절대 불변** — out["pre_entry_risk_shadow"]에만 저장.
    production 미적용 (위험 플래그 CSV 컬럼 추가는 별도 패치).

[v3.9.1 — STRUCT risk shadow backtest]
  · compute_struct_risk_shadow(): STRUCT_SCORE 70~85 종목을 Top3 후보 풀에서
    제외했을 때의 성과를 병렬 계산. rolling walk-forward 3/3 fold ΔEV 양수,
    non_win 평균손익 3/3 개선으로 검증됨 (메커니즘: 손절 감소가 아니라
    LOSS+OPEN 비승리 구간의 EV 누수 축소).
  · **baseline / 추천 / 매수가 절대 불변** — out["struct_risk_shadow"]에만 저장.
    production exclude 미적용 (Top3 구성 변경률 30~50%로 전략 변경급).

[v3.9.0 — ENTRY_MODE shadow backtest]
  · compute_entry_mode_shadow(): NOT_FILLED 종목 중 STRUCT≥90 & VWAP_GAP≤8
    조건을 만족하는 것에 chase-entry(시가 ≤ entry×1.03)를 적용했으면
    어떻게 됐을지 병렬 계산. chase 체결가 기준으로 stop/RR 재산정.
  · **기존 baseline 백테스트는 절대 불변** — shadow 결과는 out["entry_mode_shadow"]
    별도 섹션에만 저장. summarize_trades / simulate_ohlc / daily_*_backtest 무수정.
  · downstream(tab_stocks/tab_perf/kelly/combo_optimizer) 영향 0.

[v3.8.3 — TP3 실측 학습 루프]
  · 추천 CSV의 추천매도가2/추천매도가3를 pick에 포함
  · simulate_ohlc/simulate_close_only 에 ladder 12필드 추가:
      tp1_hit/tp1_day, tp2_hit/tp2_day, tp3_hit/tp3_day,
      stop_hit/stop_day, tp{1,2,3}_before_stop, max_high_pct/min_low_pct/max_close_pct
  · summarize_trades 에 tp2_rate, tp3_rate, tp3_before_stop_rate, avg_max_high_pct 추가
  · 기존 컬럼/지표 의미 불변 → tab_stocks.py, tab_perf.py, kelly 등 downstream 호환

출력:
  data/backtest_validation_latest.json   (tab_stocks.py 헤더가 읽음)
  data/backtest_validation_{YYYYMMDD}.json
  data/backtest_top3_trades_{YYYYMMDD}.csv
  data/backtest_tuning_{YYYYMMDD}.json     (--tune 옵션 시)

실행:
  python3 backtest_validation.py
  python3 backtest_validation.py --tune
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import pandas as pd
except ImportError:
    print("❌ pandas 필수 — pip install pandas pyarrow")
    sys.exit(1)


# ═══════════════════════════════════════════════════
#  라벨링 로직 — tab_stocks.py와 동일 (복제본, 동기 필수)
# ═══════════════════════════════════════════════════

def _fnum(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def compute_axis_stats(row: dict) -> dict:
    s = _fnum(row.get("STRUCT_SCORE", 0))
    t = _fnum(row.get("TIMING_SCORE", 0))
    a = _fnum(row.get("AI_SCORE", 0))
    close = _fnum(row.get("종가", 0))
    entry = _fnum(row.get("추천매수가", 0))
    stop = _fnum(row.get("손절가", 0))
    tp1 = _fnum(row.get("추천매도가1", 0))
    rr = _fnum(row.get("RR_NOW_TP1", 0))

    # [v2 패치] RR이 CSV에 없으면 종가 기준으로 재계산 (과거 CSV 호환)
    # scoring_engine.py의 compute_elite_score와 동일 공식
    if rr <= 0 and close > 0 and stop > 0 and tp1 > 0:
        risk = max(close - stop, 1.0)
        reward = max(tp1 - close, 0.0)
        rr = reward / risk

    axis_mean = (s + t + a) / 3 if (s or t or a) else 0.0
    axis_min = min(s, t, a) if (s or t or a) else 0.0
    axis_gap = (max(s, t, a) - min(s, t, a)) if (s or t or a) else 100.0
    balance = max(0.0, 100.0 - axis_gap * 1.25)
    gap_pct = (abs(close - entry) / entry * 100) if entry > 0 else 999.0
    valid = (
        close > 0 and entry > 0 and stop > 0 and tp1 > 0
        and tp1 > entry and stop < entry
    )
    return {
        "S": s, "T": t, "AI": a,
        "axis_mean": axis_mean, "axis_min": axis_min,
        "balance": balance, "gap_pct": gap_pct,
        "rr_now": rr, "valid": valid,
    }


def elite_label(stats: dict, thresholds: Optional[dict] = None) -> str:
    """thresholds 파라미터로 튜닝 그리드서치 지원.

    기본값은 v3.7.6 walk-forward 20일 검증 통과 (평균70/밸70/갭3/RR0.8).
    """
    th = thresholds or {
        "strong_mean": 70.0, "strong_bal": 70.0,
        "strong_gap": 3.0,   "strong_rr": 0.8,
        "ok_min": 50.0, "ok_bal": 70.0, "ok_gap": 5.0,
        "chase_gap": 5.0, "chase_mean": 60.0,
    }
    if not stats["valid"]:
        return ""
    am, amn, bal, gap, rr = (
        stats["axis_mean"], stats["axis_min"], stats["balance"],
        stats["gap_pct"], stats["rr_now"],
    )
    if (am >= th["strong_mean"] and bal >= th["strong_bal"]
            and gap <= th["strong_gap"] and rr >= th["strong_rr"]):
        return "🏆 최강"
    if (amn >= th["ok_min"] and bal >= th["ok_bal"]
            and gap <= th["ok_gap"]):
        return "✅ 즉시진입"
    if gap > th["chase_gap"] and am >= th["chase_mean"]:
        return "⚠️ 추격"
    return ""


def rank_score(stats: dict, label: str) -> float:
    if not stats["valid"]:
        return -999.0
    base = stats["axis_mean"] * (stats["balance"] / 100.0)
    rr = stats["rr_now"]
    if rr < 0.5:   rr_mult = 0.3
    elif rr < 1.0: rr_mult = 0.7
    else:          rr_mult = 1.0
    if   label == "✅ 즉시진입": label_mult = 1.30
    elif label == "🏆 최강":     label_mult = 1.00
    elif label == "⚠️ 추격":     label_mult = 0.70
    else:                         label_mult = 0.50
    return base * rr_mult * label_mult


def pick_top1_codes(day_csv_rows: list, thresholds: Optional[dict] = None,
                    min_rank: float = 40.0) -> list:
    """[v3.7.11] 하루치 recommend rows → Top 1 (🏆 최강 중 rank_score 1위).

    백테스트 +11.49% (36일)를 달성한 전략의 핵심 함수.
    pick_top3_codes와 달리 "rank 1위 1종목"만 반환.
    """
    candidates = []
    for row in day_csv_rows:
        stats = compute_axis_stats(row)
        lbl = elite_label(stats, thresholds)
        if lbl != "🏆 최강":
            continue
        score = rank_score(stats, lbl)
        if score < min_rank:
            continue
        candidates.append({
            "code":   str(row.get("종목코드", "")).zfill(6),
            "name":   str(row.get("종목명", "")),
            "sector": str(row.get("업종", "")),
            "label":  lbl,
            "score":  score,
            "entry":  _fnum(row.get("추천매수가", 0)),
            "tp1":    _fnum(row.get("추천매도가1", 0)),
            "tp2":    _fnum(row.get("추천매도가2", 0)),  # v3.8.3
            "tp3":    _fnum(row.get("추천매도가3", 0)),  # v3.8.3
            "stop":   _fnum(row.get("손절가", 0)),
            "close":  _fnum(row.get("종가", 0)),
        })
    if not candidates:
        return []
    candidates.sort(key=lambda r: -r["score"])
    return [candidates[0]]


def pick_top3_codes(day_csv_rows: list, thresholds: Optional[dict] = None,
                    min_rank: float = 40.0) -> list:
    """하루치 recommend rows → Top 3 (v3.7.10: 🏆 최강만).

    ✅ 즉시진입은 백테스트 net 기준 -0.22%로 확인되어 제외.
    오직 🏆 최강(+1.28%)만 사용.
    """
    candidates = []
    for row in day_csv_rows:
        stats = compute_axis_stats(row)
        lbl = elite_label(stats, thresholds)
        if lbl != "🏆 최강":  # v3.7.10: ✅ 즉시진입 배제
            continue
        score = rank_score(stats, lbl)
        if score < min_rank:
            continue
        candidates.append({
            "code":   str(row.get("종목코드", "")).zfill(6),
            "name":   str(row.get("종목명", "")),
            "sector": str(row.get("업종", "")),
            "label":  lbl,
            "score":  score,
            "entry":  _fnum(row.get("추천매수가", 0)),
            "tp1":    _fnum(row.get("추천매도가1", 0)),
            "tp2":    _fnum(row.get("추천매도가2", 0)),  # v3.8.3
            "tp3":    _fnum(row.get("추천매도가3", 0)),  # v3.8.3
            "stop":   _fnum(row.get("손절가", 0)),
            "close":  _fnum(row.get("종가", 0)),
        })
    candidates.sort(key=lambda r: -r["score"])
    picked = []
    seen_sectors: set = set()
    for c in candidates:
        if len(picked) >= 3: break
        if c["sector"] and c["sector"] in seen_sectors:
            continue
        picked.append(c)
        if c["sector"]:
            seen_sectors.add(c["sector"])
    return picked


# ═══════════════════════════════════════════════════
#  OHLC 로더
# ═══════════════════════════════════════════════════

_OHLC_CACHE: dict = {}


def load_ohlc(data_dir: Path) -> pd.DataFrame:
    """모든 ohlcv_cache_*.parquet을 병합 — 최대 커버리지 확보.

    각 parquet에 과거 18개월치 OHLCV가 들어있으므로 단순히 concat해도 됨.
    중복은 (종목코드, Date) 조합으로 제거 — 최신 parquet 값 우선.
    """
    if "df" in _OHLC_CACHE:
        return _OHLC_CACHE["df"]
    parquets = sorted(glob.glob(str(data_dir / "ohlcv_cache_*.parquet")))
    if not parquets:
        return pd.DataFrame()

    # 오래된 것부터 로드하고 마지막에 drop_duplicates(keep='last')로 최신 덮어쓰기
    frames = []
    for p in parquets:
        try:
            sub = pd.read_parquet(p)
            sub = sub.reset_index()
            sub["종목코드"] = sub["종목코드"].astype(str).str.zfill(6)
            frames.append(sub)
        except Exception as e:
            print(f"    ⚠️ {os.path.basename(p)} 로드 실패: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # (종목코드, Date) 중복 제거 — 최신 parquet 값 유지
    df = df.drop_duplicates(subset=["종목코드", "Date"], keep="last")
    df = df.sort_values(["종목코드", "Date"]).reset_index(drop=True)

    _OHLC_CACHE["df"] = df
    return df


def _empty_ladder() -> dict:
    """v3.8.3: ladder 빈 상태. simulate_* 모든 return에 동일 키 구조 보장."""
    return {
        "tp1_hit": False, "tp1_day": 0,
        "tp2_hit": False, "tp2_day": 0,
        "tp3_hit": False, "tp3_day": 0,
        "stop_hit": False, "stop_day": 0,
        "tp1_before_stop": False,
        "tp2_before_stop": False,
        "tp3_before_stop": False,
        "max_high_pct": 0.0,
        "min_low_pct": 0.0,
        "max_close_pct": 0.0,
    }


def simulate_ohlc(code: str, entry_date: pd.Timestamp, entry: float, tp1: float,
                  stop: float, horizon: int, ohlc_df: pd.DataFrame,
                  fill_window: int = 3, tp2: float = 0.0, tp3: float = 0.0) -> dict:
    """OHLC 기반 TP ladder + Stop 터치 판정 — 체결 검증 포함.

    [v3.8.3] TP ladder 확장:
      - 체결 이후 horizon 전체를 스캔해 TP1/TP2/TP3/Stop 도달일과 극값을 기록한다.
      - 기존 outcome(WIN/LOSS/OPEN/NOT_FILLED/NODATA) 의미는 v3.7.x와 동일하게 유지
        (첫 터치 우선, 동일 바 TP+Stop 동시 도달 시 LOSS 보수적).
      - 새 ladder 필드들은 첫 터치 이후에도 누적 기록되므로 "Stop 맞고 그 뒤
        TP3까지 갔는지", "TP1 맞은 후 TP2/TP3까지 갔는지" 분석이 가능해진다.

    [v3.7.8] 체결 검증 단계 (변경 없음):
      추천일 다음날부터 fill_window일 안에 '저가 ≤ 추천매수가 ≤ 고가' 또는
      '시가 ≤ 추천매수가 + 2%' 조건 만족 시 체결. 못 되면 NOT_FILLED 반환.
    """
    nodata = {
        "outcome": "NODATA", "exit_price": 0.0, "days_held": 0,
        "method": "none", "fill_date": None,
        **_empty_ladder(),
    }
    if ohlc_df is None or ohlc_df.empty:
        return nodata
    code_df = ohlc_df[ohlc_df["종목코드"] == code]
    if code_df.empty:
        return nodata
    future = code_df[code_df["Date"] > entry_date].sort_values("Date")
    if future.empty:
        return nodata

    # ── 1단계: 체결 검증 (fill_window일 안에 지정가 체결됐는지) ──
    fill_idx = None
    fill_date = None
    for i, (_, bar) in enumerate(future.head(fill_window).iterrows()):
        op = _fnum(bar["시가"])
        high = _fnum(bar["고가"])
        low = _fnum(bar["저가"])
        if low <= 0 or high <= 0:
            continue
        # gap-up 시가 > entry+2%: 체결 불가
        if op <= entry * 1.02:
            if low <= entry:  # 장중 entry 도달
                fill_idx = i
                fill_date = bar["Date"]
                break
            elif op <= entry:  # 시가 체결
                fill_idx = i
                fill_date = bar["Date"]
                break

    if fill_idx is None:
        return {
            "outcome": "NOT_FILLED", "exit_price": 0.0, "days_held": 0,
            "method": "not_filled_gap_up", "fill_date": None,
            **_empty_ladder(),
        }

    # ── 2단계: 체결일부터 horizon 끝까지 단일 패스 ──
    # ladder는 끝까지 누적, outcome은 첫 터치에서 결정 (이후 변경 없음)
    tracking = future.iloc[fill_idx:fill_idx + horizon]
    ladder = _empty_ladder()
    ladder_max_h: Optional[float] = None
    ladder_min_l: Optional[float] = None
    ladder_max_c: Optional[float] = None
    outcome_state: Optional[tuple] = None  # (outcome, exit_price, days_held, method)

    if tracking.empty:
        return {
            "outcome": "NODATA", "exit_price": 0.0, "days_held": 0,
            "method": "none", "fill_date": str(fill_date)[:10],
            **ladder,
        }

    for i, (_, bar) in enumerate(tracking.iterrows()):
        op = _fnum(bar["시가"])
        high = _fnum(bar["고가"])
        low = _fnum(bar["저가"])
        close_p = _fnum(bar["종가"])
        if low <= 0 or high <= 0:
            continue

        day_n = i + 1

        # 극값 갱신 (entry 기준 %)
        if entry > 0:
            h_pct = (high / entry - 1.0) * 100.0
            l_pct = (low / entry - 1.0) * 100.0
            if ladder_max_h is None or h_pct > ladder_max_h:
                ladder_max_h = h_pct
            if ladder_min_l is None or l_pct < ladder_min_l:
                ladder_min_l = l_pct
            if close_p > 0:
                c_pct = (close_p / entry - 1.0) * 100.0
                if ladder_max_c is None or c_pct > ladder_max_c:
                    ladder_max_c = c_pct

        # ── Ladder 기록 (전체 horizon 누적) ──
        stop_today = stop > 0 and low <= stop
        stop_already = ladder["stop_hit"]
        # 보수적: 같은 날 TP과 Stop 동시 닿으면 stop 우선 → tp_before_stop = False
        tp_before_stop_today = (not stop_already) and (not stop_today)

        if tp1 > 0 and high >= tp1 and not ladder["tp1_hit"]:
            ladder["tp1_hit"] = True
            ladder["tp1_day"] = day_n
            ladder["tp1_before_stop"] = tp_before_stop_today
        if tp2 > 0 and high >= tp2 and not ladder["tp2_hit"]:
            ladder["tp2_hit"] = True
            ladder["tp2_day"] = day_n
            ladder["tp2_before_stop"] = tp_before_stop_today
        if tp3 > 0 and high >= tp3 and not ladder["tp3_hit"]:
            ladder["tp3_hit"] = True
            ladder["tp3_day"] = day_n
            ladder["tp3_before_stop"] = tp_before_stop_today
        if stop_today and not ladder["stop_hit"]:
            ladder["stop_hit"] = True
            ladder["stop_day"] = day_n

        # ── 첫 터치 outcome 결정 (v3.7.x 로직 그대로, outcome_state가 None일 때만) ──
        if outcome_state is None:
            if i == 0:
                # 체결일: 시가 gap 미사용, 고가/저가만 (기존)
                if low <= stop and high >= tp1:
                    outcome_state = ("LOSS", stop, 1, "ohlc_both_touched")
                elif high >= tp1:
                    outcome_state = ("WIN", tp1, 1, "ohlc_high_touch")
                elif low <= stop:
                    outcome_state = ("LOSS", stop, 1, "ohlc_low_touch")
            else:
                # 다음날부터 gap 처리 (기존)
                if op >= tp1:
                    outcome_state = ("WIN", op, day_n, "ohlc_gap_up")
                elif op <= stop:
                    outcome_state = ("LOSS", op, day_n, "ohlc_gap_down")
                elif low <= stop and high >= tp1:
                    outcome_state = ("LOSS", stop, day_n, "ohlc_both_touched")
                elif low <= stop:
                    outcome_state = ("LOSS", stop, day_n, "ohlc_low_touch")
                elif high >= tp1:
                    outcome_state = ("WIN", tp1, day_n, "ohlc_high_touch")

    # 극값 None → 0.0 방어
    ladder["max_high_pct"]  = round(ladder_max_h, 2) if ladder_max_h is not None else 0.0
    ladder["min_low_pct"]   = round(ladder_min_l, 2) if ladder_min_l is not None else 0.0
    ladder["max_close_pct"] = round(ladder_max_c, 2) if ladder_max_c is not None else 0.0

    fd = str(fill_date)[:10]

    # outcome 결정
    if outcome_state is None:
        # horizon 마감 = OPEN (마지막 종가)
        last_close = _fnum(tracking.iloc[-1]["종가"])
        if last_close <= 0:
            return {
                "outcome": "NODATA", "exit_price": 0.0, "days_held": horizon,
                "method": "none", "fill_date": fd, **ladder,
            }
        return {
            "outcome": "OPEN", "exit_price": last_close,
            "days_held": len(tracking), "method": "ohlc_horizon_close",
            "fill_date": fd, **ladder,
        }

    return {
        "outcome": outcome_state[0], "exit_price": outcome_state[1],
        "days_held": outcome_state[2], "method": outcome_state[3],
        "fill_date": fd, **ladder,
    }


def simulate_close_only(code: str, entry_idx: int, entry: float, tp1: float,
                        stop: float, horizon: int, days_dict_seq: list,
                        tp2: float = 0.0, tp3: float = 0.0) -> dict:
    """종가 폴백 — OHLC 없는 종목용.

    [v3.8.3] ladder 필드를 종가 기준으로 최소 기록.
      tp_hit_n = close >= tp_n (장중 고가 정보 없어서 보수적, 실제보다 낮게 잡힘)
      max_high_pct / min_low_pct 는 0.0 (정보 없음)
      max_close_pct 만 의미 있음
    """
    ladder = _empty_ladder()
    ladder_max_c: Optional[float] = None

    future = days_dict_seq[entry_idx + 1: entry_idx + 1 + horizon]
    outcome_state: Optional[tuple] = None

    for day_i, (_, day_data) in enumerate(future, 1):
        if code not in day_data:
            continue
        px = _fnum(day_data[code].get("종가", 0))
        if px <= 0:
            continue

        if entry > 0:
            c_pct = (px / entry - 1.0) * 100.0
            if ladder_max_c is None or c_pct > ladder_max_c:
                ladder_max_c = c_pct

        stop_today = stop > 0 and px <= stop
        stop_already = ladder["stop_hit"]
        tp_before_stop_today = (not stop_already) and (not stop_today)

        if tp1 > 0 and px >= tp1 and not ladder["tp1_hit"]:
            ladder["tp1_hit"] = True
            ladder["tp1_day"] = day_i
            ladder["tp1_before_stop"] = tp_before_stop_today
        if tp2 > 0 and px >= tp2 and not ladder["tp2_hit"]:
            ladder["tp2_hit"] = True
            ladder["tp2_day"] = day_i
            ladder["tp2_before_stop"] = tp_before_stop_today
        if tp3 > 0 and px >= tp3 and not ladder["tp3_hit"]:
            ladder["tp3_hit"] = True
            ladder["tp3_day"] = day_i
            ladder["tp3_before_stop"] = tp_before_stop_today
        if stop_today and not ladder["stop_hit"]:
            ladder["stop_hit"] = True
            ladder["stop_day"] = day_i

        if outcome_state is None:
            if px <= stop:
                outcome_state = ("LOSS", stop, day_i, "close_fallback")
            elif px >= tp1:
                outcome_state = ("WIN", tp1, day_i, "close_fallback")

    ladder["max_close_pct"] = round(ladder_max_c, 2) if ladder_max_c is not None else 0.0

    if outcome_state is not None:
        return {
            "outcome": outcome_state[0], "exit_price": outcome_state[1],
            "days_held": outcome_state[2], "method": outcome_state[3],
            **ladder,
        }
    if future:
        last = future[-1][1]
        if code in last:
            px = _fnum(last[code].get("종가", 0))
            if px > 0:
                return {
                    "outcome": "OPEN", "exit_price": px,
                    "days_held": horizon, "method": "close_fallback",
                    **ladder,
                }
    return {"outcome": "NODATA", "exit_price": 0.0, "days_held": 0,
            "method": "none", **ladder}


# ═══════════════════════════════════════════════════
#  데이터 로드
# ═══════════════════════════════════════════════════

def load_day_csv(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("종목코드", "")).strip():
                out.append(r)
    return out


def load_all_days(data_dir: Path) -> list:
    """[(ymd, rows_list, rows_dict)]"""
    files = sorted(glob.glob(str(data_dir / "recommend_2026*.csv")))
    out = []
    for fp in files:
        m = re.search(r"(\d{8})\.csv", fp)
        if not m: continue
        ymd = m.group(1)
        rows_list = load_day_csv(fp)
        rows_dict = {str(r.get("종목코드", "")).zfill(6): r for r in rows_list}
        out.append((ymd, rows_list, rows_dict))
    return out


# ═══════════════════════════════════════════════════
#  [핵심 #1] 일자별 Top3 백테스트
# ═══════════════════════════════════════════════════

# [v3.7.10] Horizon 20 → 10 (백테스트 기반)
# 4~7일이 스윗스팟 (평균 +5.13%), 8~14일에 마이너스 전환 (-2.66%) 확인
# 10일 내 TP1/Stop 미도달이면 OPEN으로 청산하는 것이 실전 수익 극대화
HORIZON = 10
# [v3.7.10] 거래비용 현실화: 왕복 수수료 0.03% + 증권거래세 0.18% ≈ 0.22%
# 이전 0.4%는 보수적 과다 설정 → 실전 근사치 0.22%로 조정
COST_PCT = 0.22

# [v3.9.2] 엔진 버전 단일 출처 — JSON 메타 + 디버그 추적용.
#   파일 헤더 docstring과 일치해야 함. 새 버전 올릴 때 함께 갱신.
BACKTEST_ENGINE_VERSION = "v3.9.2"


def daily_top1_backtest(days: list, ohlc_df: pd.DataFrame,
                         thresholds: Optional[dict] = None,
                         min_rank: float = 40.0) -> dict:
    """[v3.7.11] 매일 🏆 최강 Top1만 추적. 백테스트 검증용 메인 루프.

    daily_top3_backtest와 동일 구조지만 pick_top1_codes 사용.
    이 함수가 README에서 말하는 '+11.49%'의 증거.
    """
    trades = []
    daily_picks_log = []
    days_dict_seq = [(d[0], d[2]) for d in days]

    for i in range(1, len(days)):
        ymd, rows_list, rows_dict = days[i]
        top1 = pick_top1_codes(rows_list, thresholds, min_rank)
        daily_picks_log.append({
            "date": ymd, "n_picked": len(top1),
            "codes": [c["code"] for c in top1],
        })
        if not top1:
            continue

        entry_date = pd.Timestamp(ymd[:4] + "-" + ymd[4:6] + "-" + ymd[6:])
        for pick in top1:
            code = pick["code"]
            result = simulate_ohlc(code, entry_date, pick["entry"], pick["tp1"],
                                   pick["stop"], HORIZON, ohlc_df,
                                   tp2=pick.get("tp2", 0.0), tp3=pick.get("tp3", 0.0))
            if result["outcome"] == "NODATA":
                result = simulate_close_only(code, i, pick["entry"], pick["tp1"],
                                             pick["stop"], HORIZON, days_dict_seq,
                                             tp2=pick.get("tp2", 0.0),
                                             tp3=pick.get("tp3", 0.0))
            if result["outcome"] == "NODATA":
                continue

            # v3.8.3 ladder 필드 추출
            ladder_cols = {
                "tp2":              pick.get("tp2", 0.0),
                "tp3":              pick.get("tp3", 0.0),
                "tp1_hit":          result.get("tp1_hit", False),
                "tp1_day":          result.get("tp1_day", 0),
                "tp2_hit":          result.get("tp2_hit", False),
                "tp2_day":          result.get("tp2_day", 0),
                "tp3_hit":          result.get("tp3_hit", False),
                "tp3_day":          result.get("tp3_day", 0),
                "stop_hit":         result.get("stop_hit", False),
                "stop_day":         result.get("stop_day", 0),
                "tp1_before_stop":  result.get("tp1_before_stop", False),
                "tp2_before_stop":  result.get("tp2_before_stop", False),
                "tp3_before_stop":  result.get("tp3_before_stop", False),
                "max_high_pct":     result.get("max_high_pct", 0.0),
                "min_low_pct":      result.get("min_low_pct", 0.0),
                "max_close_pct":    result.get("max_close_pct", 0.0),
            }

            if result["outcome"] == "NOT_FILLED":
                trades.append({
                    "date": ymd, "code": code, "name": pick["name"],
                    "sector": pick["sector"], "label": pick["label"],
                    "rank_score": round(pick["score"], 2),
                    "entry": pick["entry"], "tp1": pick["tp1"], "stop": pick["stop"],
                    "outcome": "NOT_FILLED", "exit_price": 0.0, "days_held": 0,
                    "method": result["method"], "fill_date": "",
                    "ret_pct": 0.0, "net_pct": 0.0,
                    **ladder_cols,
                })
                continue

            ret_pct = (result["exit_price"] / pick["entry"] - 1) * 100
            trades.append({
                "date": ymd, "code": code, "name": pick["name"],
                "sector": pick["sector"], "label": pick["label"],
                "rank_score": round(pick["score"], 2),
                "entry": pick["entry"], "tp1": pick["tp1"], "stop": pick["stop"],
                "outcome": result["outcome"],
                "exit_price": result["exit_price"],
                "days_held": result["days_held"],
                "method": result["method"],
                "fill_date": result.get("fill_date", "") or "",
                "ret_pct": round(ret_pct, 2),
                "net_pct": round(ret_pct - COST_PCT, 2),
                **ladder_cols,
            })

    return {"trades": trades, "daily_picks": daily_picks_log}


def daily_top3_backtest(days: list, ohlc_df: pd.DataFrame,
                        thresholds: Optional[dict] = None,
                        min_rank: float = 40.0) -> dict:
    """매일 pick_top3이 뽑은 3종목을 horizon일 추적.

    루프 조건 (v3.7.6):
    - OHLC가 recommend 날짜보다 뒤까지 있으면 마지막 recommend 날짜에서도 트레이딩 가능
    - 추적 중 horizon을 못 채우면 OHLC 마지막 날 종가로 마감 (simulate_ohlc 내부 처리)
    - 즉 len(days)-1까지 전부 시도, 단 0번째(=hist base)는 제외
    """
    trades = []
    daily_picks_log = []
    days_dict_seq = [(d[0], d[2]) for d in days]

    # 루프 종료점: horizon 뺀 만큼이 아니라 전체 (OHLC로 커버됨)
    for i in range(1, len(days)):
        ymd, rows_list, rows_dict = days[i]
        top3 = pick_top3_codes(rows_list, thresholds, min_rank)
        daily_picks_log.append({
            "date": ymd, "n_picked": len(top3),
            "codes": [c["code"] for c in top3],
        })
        if not top3:
            continue

        entry_date = pd.Timestamp(ymd[:4] + "-" + ymd[4:6] + "-" + ymd[6:])
        for pick in top3:
            code = pick["code"]
            result = simulate_ohlc(code, entry_date, pick["entry"], pick["tp1"],
                                   pick["stop"], HORIZON, ohlc_df,
                                   tp2=pick.get("tp2", 0.0), tp3=pick.get("tp3", 0.0))
            if result["outcome"] == "NODATA":
                result = simulate_close_only(code, i, pick["entry"], pick["tp1"],
                                             pick["stop"], HORIZON, days_dict_seq,
                                             tp2=pick.get("tp2", 0.0),
                                             tp3=pick.get("tp3", 0.0))
            if result["outcome"] == "NODATA":
                continue

            # v3.8.3 ladder 필드 추출 (top1과 동일 구조)
            ladder_cols = {
                "tp2":              pick.get("tp2", 0.0),
                "tp3":              pick.get("tp3", 0.0),
                "tp1_hit":          result.get("tp1_hit", False),
                "tp1_day":          result.get("tp1_day", 0),
                "tp2_hit":          result.get("tp2_hit", False),
                "tp2_day":          result.get("tp2_day", 0),
                "tp3_hit":          result.get("tp3_hit", False),
                "tp3_day":          result.get("tp3_day", 0),
                "stop_hit":         result.get("stop_hit", False),
                "stop_day":         result.get("stop_day", 0),
                "tp1_before_stop":  result.get("tp1_before_stop", False),
                "tp2_before_stop":  result.get("tp2_before_stop", False),
                "tp3_before_stop":  result.get("tp3_before_stop", False),
                "max_high_pct":     result.get("max_high_pct", 0.0),
                "min_low_pct":      result.get("min_low_pct", 0.0),
                "max_close_pct":    result.get("max_close_pct", 0.0),
            }

            # [v3.7.8] NOT_FILLED — 추천매수가에 체결 못 된 경우는 거래 안 된 것
            # 거래 기록에는 남기되, ret_pct=0으로 성과 통계에서 자연스럽게 중성
            if result["outcome"] == "NOT_FILLED":
                trades.append({
                    "date":        ymd,
                    "code":        code,
                    "name":        pick["name"],
                    "sector":      pick["sector"],
                    "label":       pick["label"],
                    "rank_score":  round(pick["score"], 2),
                    "entry":       pick["entry"],
                    "tp1":         pick["tp1"],
                    "stop":        pick["stop"],
                    "outcome":     "NOT_FILLED",
                    "exit_price":  0.0,
                    "days_held":   0,
                    "method":      result["method"],
                    "fill_date":   "",
                    "ret_pct":     0.0,
                    "net_pct":     0.0,
                    **ladder_cols,
                })
                continue

            ret_pct = (result["exit_price"] / pick["entry"] - 1) * 100
            trades.append({
                "date":        ymd,
                "code":        code,
                "name":        pick["name"],
                "sector":      pick["sector"],
                "label":       pick["label"],
                "rank_score":  round(pick["score"], 2),
                "entry":       pick["entry"],
                "tp1":         pick["tp1"],
                "stop":        pick["stop"],
                "outcome":     result["outcome"],
                "exit_price":  result["exit_price"],
                "days_held":   result["days_held"],
                "method":      result["method"],
                "fill_date":   result.get("fill_date", "") or "",
                "ret_pct":     round(ret_pct, 2),
                "net_pct":     round(ret_pct - COST_PCT, 2),
                **ladder_cols,
            })

    return {"trades": trades, "daily_picks": daily_picks_log}


def simulate_capital_portfolio(trades: list, initial_capital: float = 10_000_000,
                                max_positions: int = 3) -> dict:
    """자본 기반 포트폴리오 시뮬레이션 (v3.7.8 신규).

    실전에 가까운 시뮬:
    - 초기 자본: initial_capital
    - 동시 보유 상한: max_positions
    - 이미 보유 중인 종목 → 중복 진입 스킵
    - NOT_FILLED → 스킵 (체결 못 함)
    - 체결일 기준 포지션 할당: 당시 가용 자본 / 빈 슬롯 수
    - Exit 시점: entry_date + days_held 영업일
    - Exit 시 capital에 실현손익 반영

    반환:
      curve: 체결 완료 시점마다 [date, action, code, ret_pct, capital] 기록
      summary 필드들
    """
    if not trades:
        return None

    # 거래를 체결일 기준으로 정렬 (원본 오염 방지 위해 복사)
    import pandas as pd
    valid_trades = []
    for t_orig in trades:
        if t_orig["outcome"] == "NOT_FILLED":
            continue
        t = dict(t_orig)  # 얕은 복사
        if not t.get("fill_date"):
            fd = pd.Timestamp(t["date"][:4] + "-" + t["date"][4:6] + "-" + t["date"][6:])
            fd = fd + pd.tseries.offsets.BDay(1)
            t["_fill_date_ts"] = fd
        else:
            t["_fill_date_ts"] = pd.Timestamp(t["fill_date"])
        t["_entry_date_ts"] = pd.Timestamp(t["date"][:4] + "-" + t["date"][4:6] + "-" + t["date"][6:])
        valid_trades.append(t)

    valid_trades.sort(key=lambda t: (t["_fill_date_ts"], t["_entry_date_ts"]))

    # 포지션 슬롯 관리: 각 슬롯은 {code, fill_date, exit_date, invested, current_value}
    open_positions = {}  # code → {...}
    capital = initial_capital
    curve = []
    audit_log = []  # [v3.7.14] 신호별 skip/execute 이유 상세 기록
    n_filled = 0
    n_skipped_duplicate = 0
    n_skipped_not_filled = sum(1 for t in trades if t["outcome"] == "NOT_FILLED")
    n_skipped_full = 0

    # [v3.7.14] NOT_FILLED 건도 audit에 기록 (왜 배제됐는지 보여주기)
    for t in trades:
        if t["outcome"] == "NOT_FILLED":
            audit_log.append({
                "signal_date": t["date"],
                "code": t["code"],
                "name": t["name"],
                "label": t["label"],
                "action": "SKIP",
                "skip_reason": "NOT_FILLED",
                "skip_reason_detail": "추천매수가 gap-up으로 체결 불가",
                "fill_date": "",
                "open_positions_count": 0,
                "held_codes": "",
            })

    for t in valid_trades:
        fill_date = t["_fill_date_ts"]
        code = t["code"]
        exit_date = fill_date + pd.tseries.offsets.BDay(max(t["days_held"], 1))

        # 먼저 fill_date 이전/당일에 Exit된 포지션 정리
        expired = [c for c, p in open_positions.items() if p["exit_date"] <= fill_date]
        for c in expired:
            p = open_positions[c]
            # [v3.7.9] net (비용 차감) 기준으로 실현 — gross/net 통일
            net_ret = p["ret_pct"] - COST_PCT
            realized = p["invested"] * (net_ret / 100)
            # 원금 + 실현손익 → 자본으로 회수
            capital += p["invested"] + realized
            del open_positions[c]
            total_invested = sum(pos["invested"] for pos in open_positions.values())
            curve.append({
                "date": str(p["exit_date"])[:10],
                "action": "EXIT",
                "code": c,
                "name": p["name"],
                "ret_pct_gross": round(p["ret_pct"], 2),
                "ret_pct_net": round(net_ret, 2),
                "realized_pl": round(realized, 0),
                "cash": round(capital, 0),
                "invested_total": round(total_invested, 0),
                "total_assets": round(capital + total_invested, 0),
                "open_positions": len(open_positions),
            })

        # [v3.7.14] 현재 시점 열린 포지션 스냅샷 (audit용)
        held_now = ",".join(sorted(open_positions.keys()))
        open_count = len(open_positions)

        # 중복 보유 체크
        if code in open_positions:
            n_skipped_duplicate += 1
            audit_log.append({
                "signal_date": t["date"],
                "code": code,
                "name": t["name"],
                "label": t["label"],
                "action": "SKIP",
                "skip_reason": "SAME_TICKER_ALREADY_HELD",
                "skip_reason_detail": "이미 보유 중",
                "fill_date": str(fill_date)[:10],
                "open_positions_count": open_count,
                "held_codes": held_now,
            })
            continue

        # 슬롯 풀 체크 — 최대 max_positions개 동시 보유
        if len(open_positions) >= max_positions:
            n_skipped_full += 1
            audit_log.append({
                "signal_date": t["date"],
                "code": code,
                "name": t["name"],
                "label": t["label"],
                "action": "SKIP",
                "skip_reason": "SLOT_FULL",
                "skip_reason_detail": f"최대 {max_positions}포지션 다 찼음",
                "fill_date": str(fill_date)[:10],
                "open_positions_count": open_count,
                "held_codes": held_now,
            })
            continue

        # [v3.7.10 버그 수정] 투자금 = 총자산 / max_positions
        # 이전 버그: `capital / empty_slots` → 청산 후 재진입 시 자본 유휴 발생
        # 예: 1000만 시작 → 3개 진입 → 1개 청산(336만 회수) → 재진입 시 336/3=112만만 투자
        # 수정 후: 항상 총자산/3씩 투자 → 자본 활용도 일관
        total_assets_now = capital + sum(pos["invested"] for pos in open_positions.values())
        invested = min(total_assets_now / max_positions, capital)  # 현금 부족하면 현금만큼
        if invested <= 0:
            n_skipped_full += 1
            continue
        capital -= invested

        open_positions[code] = {
            "name": t["name"],
            "fill_date": fill_date,
            "exit_date": exit_date,
            "invested": invested,
            "ret_pct": t["ret_pct"],
        }
        n_filled += 1
        total_invested = sum(pos["invested"] for pos in open_positions.values())
        curve.append({
            "date": str(fill_date)[:10],
            "action": "ENTER",
            "code": code,
            "name": t["name"],
            "ret_pct_gross": 0,
            "ret_pct_net": 0,
            "realized_pl": 0,
            "cash": round(capital, 0),
            "invested_total": round(total_invested, 0),
            "total_assets": round(capital + total_invested, 0),
            "open_positions": len(open_positions),
        })
        # [v3.7.14] audit 로그에 EXECUTE 기록
        audit_log.append({
            "signal_date": t["date"],
            "code": code,
            "name": t["name"],
            "label": t["label"],
            "action": "EXECUTE",
            "skip_reason": "EXECUTED",
            "skip_reason_detail": "",
            "fill_date": str(fill_date)[:10],
            "open_positions_count": open_count,
            "held_codes": held_now,
        })

    # 남은 포지션 최종 청산 (horizon 종가 기준)
    for c in list(open_positions.keys()):
        p = open_positions[c]
        net_ret = p["ret_pct"] - COST_PCT
        realized = p["invested"] * (net_ret / 100)
        capital += p["invested"] + realized
        del open_positions[c]
        total_invested = sum(pos["invested"] for pos in open_positions.values())
        curve.append({
            "date": str(p["exit_date"])[:10],
            "action": "EXIT_FINAL",
            "code": c,
            "name": p["name"],
            "ret_pct_gross": round(p["ret_pct"], 2),
            "ret_pct_net": round(net_ret, 2),
            "realized_pl": round(realized, 0),
            "cash": round(capital, 0),
            "invested_total": round(total_invested, 0),
            "total_assets": round(capital + total_invested, 0),
            "open_positions": len(open_positions),
        })

    # 통계
    total_return = (capital - initial_capital) / initial_capital * 100

    # [v3.7.8] MDD: total_assets 기준 (현금+보유원금 · 평가손익은 보수적으로 생략)
    total_assets_series = [e.get("total_assets", initial_capital) for e in curve]
    peak = initial_capital
    max_dd = 0.0
    for ta in total_assets_series:
        if ta > peak:
            peak = ta
        if peak > 0:
            dd = (peak - ta) / peak * 100
            if dd > max_dd:
                max_dd = dd

    # 일별 종가 total_assets (같은 날 여러 이벤트면 마지막 값)
    daily_ta = {}
    for event in curve:
        daily_ta[event["date"]] = event.get("total_assets", initial_capital)
    sorted_dates = sorted(daily_ta.keys())
    n_pos_days = 0
    n_all_days = len(sorted_dates)
    prev_ta = initial_capital
    for d in sorted_dates:
        ta = daily_ta[d]
        if ta > prev_ta:
            n_pos_days += 1
        prev_ta = ta

    return {
        "initial_capital": initial_capital,
        "final_capital": round(capital, 0),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "n_filled": n_filled,
        "n_skipped_duplicate": n_skipped_duplicate,
        "n_skipped_not_filled": n_skipped_not_filled,
        "n_skipped_slot_full": n_skipped_full,
        "positive_day_rate": round(n_pos_days / n_all_days, 4) if n_all_days > 0 else 0,
        "curve": curve,
        # [v3.7.14] 감사 로그 및 스킵 이유 집계
        "audit_log": audit_log,
        "skip_reasons_summary": {
            "NOT_FILLED": n_skipped_not_filled,
            "SAME_TICKER_ALREADY_HELD": n_skipped_duplicate,
            "SLOT_FULL": n_skipped_full,
            "EXECUTED": n_filled,
            "total_signals": len(trades),
        },
    }


# ═══════════════════════════════════════════════════
#  [v3.9.0] ENTRY_MODE shadow backtest
# ═══════════════════════════════════════════════════
# 목적: 기존 백테스트가 NOT_FILLED로 버린 종목 중, 강한 종목(STRUCT≥90)이
#       다음날 갭상승으로 못 잡힌 케이스(P1 패턴)에 chase-entry를 적용했으면
#       어떻게 됐을지 병렬 계산. **baseline 백테스트는 절대 불변.**
#
# provisional rule (2026-05-14 시뮬레이션 기반 — simulate_entry_mode_v2.py):
#   - 대상: NOT_FILLED & STRUCT_SCORE≥90 & VWAP_GAP≤8
#   - chase 체결: 다음날 시가 ≤ entry × (1 + CHASE_CAP_PCT/100)
#   - chase 체결가 = 다음날 시가
#   - chase stop = chase_price × (1 - CHASE_STOP_PCT/100)   ← B_fixed6 방식
#   - rr_chase = (tp1 - chase_price) / (chase_price - chase_stop)
#   - rr_chase ≥ RR_CHASE_MIN 일 때만 chase 체결로 인정
# 시뮬 결과: cap+3% / stop-6% / VWAP≤8 → sum_ret +70%, stop_rate 0% (n=8)
ENTRY_MODE_SHADOW_ENABLED = True
CHASE_CAP_PCT = 3.0          # 다음날 시가가 entry +3% 이내면 chase
CHASE_STOP_PCT = 6.0         # chase 체결가 기준 -6% 손절
RR_CHASE_MIN = 0.8           # chase 손익비 하한
SHADOW_STRUCT_FLOOR = 90.0   # STRUCT_SCORE 하한
SHADOW_VWAP_GAP_CAP = 8.0    # VWAP_GAP 상한 (이 초과면 chase 금지)


def _load_recommend_features(days: list) -> dict:
    """recommend rows에서 (date, code) → {STRUCT_SCORE, VWAP_GAP} 맵 생성.

    days는 load_all_days() 결과: [(ymd, rows_list, rows_dict), ...]
    trade record에는 STRUCT/VWAP_GAP이 없으므로 원본 recommend에서 다시 읽는다.
    """
    feat = {}
    for ymd, rows_list, _ in days:
        for row in rows_list:
            code = str(row.get("종목코드", "")).zfill(6)
            feat[(ymd, code)] = {
                "STRUCT_SCORE": _fnum(row.get("STRUCT_SCORE", 0)),
                "VWAP_GAP":     _fnum(row.get("VWAP_GAP", 0)),
            }
    return feat


def compute_entry_mode_shadow(trades: list, ohlc_df: pd.DataFrame,
                              days: list) -> dict:
    """NOT_FILLED 종목에 chase-entry를 적용했을 때의 shadow 성과 계산.

    [중요] 이 함수는 baseline trades를 변경하지 않는다. 읽기만 한다.
    반환된 dict는 out["entry_mode_shadow"]에만 저장된다.

    Returns
    -------
    dict with keys:
      enabled, rule, n_not_filled_total, n_shadow_eligible,
      extra_fills, extra_wins, extra_losses, extra_opens,
      extra_sum_ret, extra_avg_ret, shadow_stop_rate,
      avg_rr_chase, shadow_trades (list of per-trade detail)
    """
    if not ENTRY_MODE_SHADOW_ENABLED:
        return {"enabled": False}

    rule_str = (
        f"NOT_FILLED & STRUCT≥{SHADOW_STRUCT_FLOOR:.0f} & VWAP_GAP≤{SHADOW_VWAP_GAP_CAP:.0f} "
        f"& chase_cap=+{CHASE_CAP_PCT:.0f}% & chase_stop=-{CHASE_STOP_PCT:.0f}% "
        f"& rr_chase≥{RR_CHASE_MIN}"
    )

    feat_map = _load_recommend_features(days)
    not_filled = [t for t in trades if t.get("outcome") == "NOT_FILLED"]

    shadow_trades = []
    n_eligible = 0
    for t in not_filled:
        ymd = str(t.get("date", ""))
        code = str(t.get("code", "")).zfill(6)
        entry = _fnum(t.get("entry", 0))
        tp1 = _fnum(t.get("tp1", 0))
        if entry <= 0 or tp1 <= 0:
            continue

        feat = feat_map.get((ymd, code), {})
        struct = feat.get("STRUCT_SCORE", 0.0)
        vwap_gap = feat.get("VWAP_GAP", 0.0)

        # provisional rule 게이트
        if struct < SHADOW_STRUCT_FLOOR:
            continue
        if vwap_gap > SHADOW_VWAP_GAP_CAP:
            continue
        n_eligible += 1

        # OHLC 다음날 시가 확인
        entry_date = pd.Timestamp(ymd[:4] + "-" + ymd[4:6] + "-" + ymd[6:])
        code_df = ohlc_df[ohlc_df["종목코드"] == code]
        future = code_df[code_df["Date"] > entry_date].sort_values("Date").head(HORIZON)
        if future.empty:
            continue
        d1 = future.iloc[0]
        next_open = _fnum(d1["시가"])
        if next_open <= 0:
            continue

        # chase 체결 조건: 시가가 chase cap 이내
        chase_limit = entry * (1 + CHASE_CAP_PCT / 100.0)
        if next_open > chase_limit:
            continue  # cap 초과 → chase 못 함

        chase_price = next_open
        chase_stop = chase_price * (1 - CHASE_STOP_PCT / 100.0)
        if chase_stop >= chase_price:
            continue

        # rr_chase 필터
        reward = tp1 - chase_price
        risk = chase_price - chase_stop
        rr_chase = reward / risk if risk > 0 else 0.0
        if rr_chase < RR_CHASE_MIN:
            continue

        # chase 체결가 + chase_stop 기준 TP1/Stop 첫 터치 판정
        outcome = "OPEN"
        exit_price = chase_price
        days_held = 0
        for i, (_, bar) in enumerate(future.iterrows()):
            op = _fnum(bar["시가"])
            high = _fnum(bar["고가"])
            low = _fnum(bar["저가"])
            if low <= 0 or high <= 0:
                continue
            day_n = i + 1
            if i == 0:
                # chase 체결 당일: 고가/저가만
                if low <= chase_stop and high >= tp1:
                    outcome, exit_price, days_held = "LOSS", chase_stop, day_n
                    break
                if high >= tp1:
                    outcome, exit_price, days_held = "WIN", tp1, day_n
                    break
                if low <= chase_stop:
                    outcome, exit_price, days_held = "LOSS", chase_stop, day_n
                    break
            else:
                if op >= tp1:
                    outcome, exit_price, days_held = "WIN", op, day_n
                    break
                if op <= chase_stop:
                    outcome, exit_price, days_held = "LOSS", op, day_n
                    break
                if low <= chase_stop and high >= tp1:
                    outcome, exit_price, days_held = "LOSS", chase_stop, day_n
                    break
                if low <= chase_stop:
                    outcome, exit_price, days_held = "LOSS", chase_stop, day_n
                    break
                if high >= tp1:
                    outcome, exit_price, days_held = "WIN", tp1, day_n
                    break
        else:
            # horizon 마감 = OPEN
            last_close = _fnum(future.iloc[-1]["종가"])
            if last_close > 0:
                exit_price = last_close
                days_held = len(future)

        ret_pct = (exit_price / chase_price - 1) * 100 - COST_PCT

        shadow_trades.append({
            "date": ymd, "code": code, "name": t.get("name", ""),
            "sector": t.get("sector", ""),
            "STRUCT_SCORE": round(struct, 1),
            "VWAP_GAP": round(vwap_gap, 2),
            "baseline_outcome": "NOT_FILLED",
            "entry": entry, "tp1": tp1,
            "chase_price": round(chase_price, 1),
            "chase_stop": round(chase_stop, 1),
            "rr_chase": round(rr_chase, 3),
            "shadow_outcome": outcome,
            "shadow_exit_price": round(exit_price, 1),
            "shadow_ret_pct": round(ret_pct, 2),
            "shadow_days_held": days_held,
        })

    # 집계
    extra_fills = len(shadow_trades)
    extra_wins = sum(1 for s in shadow_trades if s["shadow_outcome"] == "WIN")
    extra_losses = sum(1 for s in shadow_trades if s["shadow_outcome"] == "LOSS")
    extra_opens = sum(1 for s in shadow_trades if s["shadow_outcome"] == "OPEN")
    extra_sum_ret = round(sum(s["shadow_ret_pct"] for s in shadow_trades), 2)
    extra_avg_ret = round(extra_sum_ret / extra_fills, 2) if extra_fills > 0 else 0.0
    shadow_stop_rate = round(extra_losses / extra_fills, 4) if extra_fills > 0 else 0.0
    avg_rr_chase = (
        round(sum(s["rr_chase"] for s in shadow_trades) / extra_fills, 3)
        if extra_fills > 0 else 0.0
    )

    return {
        "enabled": True,
        "rule": rule_str,
        "n_not_filled_total": len(not_filled),
        "n_shadow_eligible": n_eligible,       # rule 게이트 통과 (chase 체결 전)
        "extra_fills": extra_fills,            # 실제 chase 체결된 수
        "extra_wins": extra_wins,
        "extra_losses": extra_losses,
        "extra_opens": extra_opens,
        "extra_sum_ret": extra_sum_ret,
        "extra_avg_ret": extra_avg_ret,
        "shadow_stop_rate": shadow_stop_rate,
        "avg_rr_chase": avg_rr_chase,
        "shadow_trades": shadow_trades,
        # production 후보 판정 (참고용 — 자동 적용 아님)
        "production_candidate": bool(
            extra_fills >= 8
            and extra_wins > extra_losses
            and extra_sum_ret > 0
            and avg_rr_chase >= RR_CHASE_MIN
        ),
    }


# ═══════════════════════════════════════════════════
#  [v3.9.1] STRUCT risk shadow backtest
# ═══════════════════════════════════════════════════
# 목적: STRUCT_SCORE 70~85 종목을 Top3 후보 풀에서 제외했으면 어땠을지
#       병렬 계산. **baseline / 추천 / 매수가 절대 불변.**
#
# 검증 근거 (simulate_struct_risk_shadow.py --mode rwf, 2026-05-14):
#   rolling walk-forward 3-fold — 3/3 fold 모두 ΔEV 양수
#   (weak 평균 +2.90, favorable +0.11), non_win 평균손익 3/3 개선.
#   메커니즘: 손절률 감소가 아니라 LOSS+OPEN 비승리 구간의 EV 누수 축소.
#
# production 미적용 이유: Top3 구성 변경률 30~50% — 단순 필터가 아니라
#   전략 변경급. shadow 측정으로 상승장 표본 추가 확보 후 재판단.
STRUCT_RISK_SHADOW_ENABLED = True
STRUCT_RISK_BAND_LO = 70.0   # STRUCT_SCORE 위험 구간 하한
STRUCT_RISK_BAND_HI = 85.0   # STRUCT_SCORE 위험 구간 상한 (이 미만)


def _pick_top3_exclude_struct_band(day_csv_rows: list,
                                   thresholds: Optional[dict] = None,
                                   min_rank: float = 40.0) -> list:
    """pick_top3_codes 로직 복제 — STRUCT 70~85 종목만 후보에서 제외.

    pick_top3_codes를 수정하지 않기 위해 여기서 재구현 (shadow 전용).
    원본과 동일한 elite_label / rank_score / 섹터 dedup 사용.
    """
    candidates = []
    for row in day_csv_rows:
        stats = compute_axis_stats(row)
        lbl = elite_label(stats, thresholds)
        if lbl != "🏆 최강":
            continue
        # ── STRUCT 70~85 제외 (shadow rule) ──
        if STRUCT_RISK_BAND_LO <= stats["S"] < STRUCT_RISK_BAND_HI:
            continue
        score = rank_score(stats, lbl)
        if score < min_rank:
            continue
        candidates.append({
            "code":   str(row.get("종목코드", "")).zfill(6),
            "name":   str(row.get("종목명", "")),
            "sector": str(row.get("업종", "")),
            "label":  lbl,
            "score":  score,
            "entry":  _fnum(row.get("추천매수가", 0)),
            "tp1":    _fnum(row.get("추천매도가1", 0)),
            "tp2":    _fnum(row.get("추천매도가2", 0)),
            "tp3":    _fnum(row.get("추천매도가3", 0)),
            "stop":   _fnum(row.get("손절가", 0)),
            "close":  _fnum(row.get("종가", 0)),
        })
    candidates.sort(key=lambda r: -r["score"])
    picked = []
    seen_sectors: set = set()
    for c in candidates:
        if len(picked) >= 3:
            break
        if c["sector"] and c["sector"] in seen_sectors:
            continue
        picked.append(c)
        if c["sector"]:
            seen_sectors.add(c["sector"])
    return picked


def _summarize_shadow_trades(trades: list) -> dict:
    """STRUCT shadow 전용 요약 — non_win 평균손익 포함.

    summarize_trades를 수정하지 않기 위해 shadow 전용 경량 요약을 별도 구현.
    핵심: ev, tp1_rate, stop_rate, fill_rate, avg_non_win.
    """
    filled = [t for t in trades if t["outcome"] != "NOT_FILLED"]
    n_all = len(trades)
    n = len(filled)
    if n == 0:
        return {"n_all": n_all, "n_filled": 0, "note": "no_filled"}
    wins = [t for t in filled if t["outcome"] == "WIN"]
    losses = [t for t in filled if t["outcome"] == "LOSS"]
    opens = [t for t in filled if t["outcome"] == "OPEN"]
    tp1_r = len(wins) / n
    stop_r = len(losses) / n
    open_r = len(opens) / n
    avg_win = sum(t["ret_pct"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t["ret_pct"] for t in losses) / len(losses) if losses else 0.0
    avg_open = sum(t["ret_pct"] for t in opens) / len(opens) if opens else 0.0
    ev = (tp1_r * avg_win + stop_r * avg_loss + open_r * avg_open) - COST_PCT
    non_win = losses + opens
    avg_non_win = (
        sum(t["ret_pct"] for t in non_win) / len(non_win) if non_win else 0.0
    )
    avg_all = sum(t["ret_pct"] for t in filled) / n
    return {
        "n_all": n_all,
        "n_filled": n,
        "fill_rate": round(n / n_all, 4) if n_all > 0 else 0.0,
        "tp1_rate": round(tp1_r, 4),
        "stop_rate": round(stop_r, 4),
        "open_rate": round(open_r, 4),
        "ev": round(ev, 2),
        "avg_non_win": round(avg_non_win, 2),
        "avg_all": round(avg_all, 2),
    }


def compute_struct_risk_shadow(days: list, ohlc_df: pd.DataFrame,
                               baseline_trades: list,
                               thresholds: Optional[dict] = None,
                               min_rank: float = 40.0) -> dict:
    """STRUCT 70~85 제외 shadow 백테스트.

    [중요] baseline_trades 는 읽기 전용 — 변경하지 않는다.
    shadow Top3를 자체적으로 다시 뽑아 동일 OHLC로 백테스트한 뒤,
    baseline 대비 ΔEV / Δnon_win / 구성변경률 등을 계산해 dict로 반환.
    반환값은 out["struct_risk_shadow"]에만 저장된다.

    measurement-only: ranking / picks / buy price / production 추천에
    어떤 영향도 주지 않는다.
    """
    if not STRUCT_RISK_SHADOW_ENABLED:
        return {"enabled": False}

    rule_str = (
        f"exclude STRUCT_SCORE {STRUCT_RISK_BAND_LO:.0f}~{STRUCT_RISK_BAND_HI:.0f} "
        f"from Top3 candidate pool"
    )

    # ── shadow Top3 백테스트 (STRUCT 70~85 제외) ──
    shadow_trades = []
    shadow_picks_by_day = {}
    days_dict_seq = [(d[0], d[2]) for d in days]

    for i in range(1, len(days)):
        ymd, rows_list, _ = days[i]
        top3 = _pick_top3_exclude_struct_band(rows_list, thresholds, min_rank)
        shadow_picks_by_day[ymd] = [c["code"] for c in top3]
        if not top3:
            continue
        entry_date = pd.Timestamp(ymd[:4] + "-" + ymd[4:6] + "-" + ymd[6:])
        for pick in top3:
            code = pick["code"]
            result = simulate_ohlc(code, entry_date, pick["entry"], pick["tp1"],
                                   pick["stop"], HORIZON, ohlc_df,
                                   tp2=pick.get("tp2", 0.0), tp3=pick.get("tp3", 0.0))
            if result["outcome"] == "NODATA":
                result = simulate_close_only(code, i, pick["entry"], pick["tp1"],
                                             pick["stop"], HORIZON, days_dict_seq,
                                             tp2=pick.get("tp2", 0.0),
                                             tp3=pick.get("tp3", 0.0))
            if result["outcome"] == "NODATA":
                continue
            if result["outcome"] == "NOT_FILLED":
                shadow_trades.append({
                    "date": ymd, "code": code, "outcome": "NOT_FILLED",
                    "ret_pct": 0.0,
                })
                continue
            ret_pct = (result["exit_price"] / pick["entry"] - 1) * 100
            shadow_trades.append({
                "date": ymd, "code": code, "outcome": result["outcome"],
                "ret_pct": round(ret_pct, 2),
            })

    # ── baseline trades 요약 (읽기 전용) ──
    base_summ = _summarize_shadow_trades(baseline_trades)
    shadow_summ = _summarize_shadow_trades(shadow_trades)

    if base_summ.get("n_filled", 0) == 0 or shadow_summ.get("n_filled", 0) == 0:
        return {
            "enabled": True, "rule": rule_str,
            "note": "insufficient_filled_trades",
            "baseline": base_summ, "shadow": shadow_summ,
        }

    # ── 구성 변경률 (baseline picks vs shadow picks) ──
    # baseline picks는 trade record의 (date, code)에서 역산
    base_picks_by_day: dict = {}
    for t in baseline_trades:
        base_picks_by_day.setdefault(str(t.get("date", "")), set()).add(
            str(t.get("code", "")).zfill(6))
    changed = 0
    total = 0
    for ymd, codes in shadow_picks_by_day.items():
        bc = base_picks_by_day.get(ymd, set())
        if bc or codes:
            total += 1
            if set(codes) != bc:
                changed += 1
    changed_pick_rate = round(changed / total, 4) if total > 0 else 0.0

    d_ev = round(shadow_summ["ev"] - base_summ["ev"], 2)
    d_tp1 = round(shadow_summ["tp1_rate"] - base_summ["tp1_rate"], 4)
    d_stop = round(shadow_summ["stop_rate"] - base_summ["stop_rate"], 4)
    d_fill = round(shadow_summ["fill_rate"] - base_summ["fill_rate"], 4)
    d_non_win = round(shadow_summ["avg_non_win"] - base_summ["avg_non_win"], 2)

    # single_backtest_ok — 단일 백테스트에서 조건 만족 여부.
    # ★ 이것은 "좋은 신호"일 뿐 "production 승인"이 아니다. ★
    # production_candidate 는 항상 False — RWF(rolling walk-forward) 검증이
    # 리포트에 배선되기 전까지는 운영 적용 금지.
    # RWF 검증은 simulate_struct_risk_shadow.py --mode rwf 가 담당하며,
    # 그 결과는 rwf_validated / rwf_folds_passed / rwf_avg_delta_ev 에 채워질 예정.
    single_backtest_ok = bool(
        d_ev > 0
        and d_non_win > 0
        and d_fill >= -0.07
        and changed_pick_rate <= 0.50
    )

    return {
        "enabled": True,
        "mode": "single_backtest_shadow",
        "rule": rule_str,
        "baseline_ev": base_summ["ev"],
        "shadow_ev": shadow_summ["ev"],
        "delta_ev": d_ev,
        "baseline_tp1_rate": base_summ["tp1_rate"],
        "shadow_tp1_rate": shadow_summ["tp1_rate"],
        "delta_tp1_rate": d_tp1,
        "baseline_stop_rate": base_summ["stop_rate"],
        "shadow_stop_rate": shadow_summ["stop_rate"],
        "delta_stop_rate": d_stop,
        "baseline_fill_rate": base_summ["fill_rate"],
        "shadow_fill_rate": shadow_summ["fill_rate"],
        "delta_fill_rate": d_fill,
        # ★ 핵심 메커니즘 지표 — 손절 감소가 아니라 비승리 구간 EV 누수 축소
        "baseline_non_win_avg_ret": base_summ["avg_non_win"],
        "shadow_non_win_avg_ret": shadow_summ["avg_non_win"],
        "delta_non_win_avg_ret": d_non_win,
        "changed_pick_rate": changed_pick_rate,
        "n_baseline_filled": base_summ["n_filled"],
        "n_shadow_filled": shadow_summ["n_filled"],
        # ── production 게이트 — B안: 단일 백테스트 통과 ≠ 운영 승인 ──
        "single_backtest_ok": single_backtest_ok,
        "rwf_required": True,
        "production_candidate": False,   # RWF 배선 전까지 항상 False
        "production_gate_reason": "single_backtest_only__rwf_required",
        # ── RWF 검증 결과 placeholder (외부 스크립트가 채울 예정) ──
        "rwf_validated": False,
        "rwf_folds_passed": None,
        "rwf_avg_delta_ev": None,
        "note": (
            "measurement-only; does not alter ranking, picks, buy price, "
            "or production recommendations. single_backtest_ok is a positive "
            "signal, NOT a production approval. RWF fold validation via "
            "simulate_struct_risk_shadow.py --mode rwf."
        ),
    }


# ═══════════════════════════════════════════════════
#  [v3.9.2] PRE_ENTRY_RISK shadow backtest
# ═══════════════════════════════════════════════════
# 배경: 5/15 폭락에서 손실 4건 attribution 결과
#   삼성증권 5/11: STRUCT 79.4 / VWAP_GAP 18.7  → −10.28%
#   삼성E&A  5/11: STRUCT 76.3 / VWAP_GAP 26.6  → −13.02%
#   삼성E&A  5/12: STRUCT 86.6 / VWAP_GAP 17.8  → −8.42%
#   HD현대   5/13: STRUCT 85.8 / VWAP_GAP 19.6  → −8.27%
# 4건 모두 (STRUCT 70~85 ∨ STRUCT 85~90 경계) + VWAP_GAP > 15 과열 조합.
# 즉 STRUCT 70~85 단독보다 (STRUCT + VWAP_GAP) 결합이 더 정확한 위험 신호.
#
# 4개 룰 비교 — 어느 룰이 구성변경률 적으면서 손실 회피 효과 큰가:
#   Rule A: STRUCT 70~85 단독 제외             (= 기존 struct_risk_shadow와 동일)
#   Rule B: STRUCT 70~85 AND VWAP_GAP > 8 제외  (RED — 본인 정의)
#   Rule C: STRUCT < 90  AND VWAP_GAP > 15 제외 (ORANGE — 본인 정의)
#   Rule D: B ∪ C 합집합                        (RED + ORANGE)
#
# **baseline / 추천 / 매수가 절대 불변** — out["pre_entry_risk_shadow"]에만 저장.
# production exclude 미적용 — 위험 플래그 CSV 컬럼 추가는 별도 패치.
PRE_ENTRY_RISK_SHADOW_ENABLED = True
PRE_RISK_STRUCT_LO = 70.0   # RED 하한
PRE_RISK_STRUCT_HI = 85.0   # RED 상한 (이 미만)
PRE_RISK_VWAP_RED = 8.0     # RED VWAP_GAP 임계
PRE_RISK_STRUCT_TOP = 90.0  # ORANGE STRUCT 상한 (이 미만)
PRE_RISK_VWAP_ORANGE = 15.0  # ORANGE VWAP_GAP 임계


def _classify_pre_entry_risk(struct_score: float, vwap_gap: float) -> str:
    """진입 시점의 위험 레벨 분류.
    
    Returns: "RED" / "ORANGE" / "GREEN"
      RED:    STRUCT 70~85 AND VWAP_GAP > 8   (위험구간 + 과열)
      ORANGE: STRUCT < 90 AND VWAP_GAP > 15   (강한 종목 아닌데 VWAP 멀리)
              (RED와 겹치면 RED 우선)
      GREEN:  나머지
    """
    in_red = (PRE_RISK_STRUCT_LO <= struct_score < PRE_RISK_STRUCT_HI
              and vwap_gap > PRE_RISK_VWAP_RED)
    if in_red:
        return "RED"
    in_orange = (struct_score < PRE_RISK_STRUCT_TOP
                 and vwap_gap > PRE_RISK_VWAP_ORANGE)
    if in_orange:
        return "ORANGE"
    return "GREEN"


def _make_risk_exclude_fn(rule: str):
    """룰 이름 → "이 종목 제외해야 하는가" 판정 함수.
    
    True = 제외. False = 유지.
    """
    def _A_struct_only(s, v):
        return PRE_RISK_STRUCT_LO <= s < PRE_RISK_STRUCT_HI
    def _B_red_only(s, v):
        return (PRE_RISK_STRUCT_LO <= s < PRE_RISK_STRUCT_HI
                and v > PRE_RISK_VWAP_RED)
    def _C_orange_only(s, v):
        return (s < PRE_RISK_STRUCT_TOP and v > PRE_RISK_VWAP_ORANGE)
    def _D_red_or_orange(s, v):
        return _B_red_only(s, v) or _C_orange_only(s, v)
    return {
        "A_struct70_85": _A_struct_only,
        "B_red":         _B_red_only,
        "C_orange":      _C_orange_only,
        "D_red_orange":  _D_red_or_orange,
    }[rule]


def _pick_top3_with_risk_exclude(day_csv_rows: list, exclude_fn,
                                  thresholds: Optional[dict] = None,
                                  min_rank: float = 40.0) -> list:
    """pick_top3_codes 로직 복제 — exclude_fn이 True 반환하는 종목만 제외.
    
    원본 pick_top3_codes 무수정. shadow 전용.
    """
    candidates = []
    for row in day_csv_rows:
        stats = compute_axis_stats(row)
        lbl = elite_label(stats, thresholds)
        if lbl != "🏆 최강":
            continue
        struct = stats["S"]
        vwap_gap = _fnum(row.get("VWAP_GAP", 0))
        if exclude_fn(struct, vwap_gap):
            continue  # shadow rule에 의한 제외
        score = rank_score(stats, lbl)
        if score < min_rank:
            continue
        candidates.append({
            "code":   str(row.get("종목코드", "")).zfill(6),
            "name":   str(row.get("종목명", "")),
            "sector": str(row.get("업종", "")),
            "label":  lbl,
            "score":  score,
            "entry":  _fnum(row.get("추천매수가", 0)),
            "tp1":    _fnum(row.get("추천매도가1", 0)),
            "tp2":    _fnum(row.get("추천매도가2", 0)),
            "tp3":    _fnum(row.get("추천매도가3", 0)),
            "stop":   _fnum(row.get("손절가", 0)),
            "close":  _fnum(row.get("종가", 0)),
        })
    candidates.sort(key=lambda r: -r["score"])
    picked = []
    seen_sectors: set = set()
    for c in candidates:
        if len(picked) >= 3:
            break
        if c["sector"] and c["sector"] in seen_sectors:
            continue
        picked.append(c)
        if c["sector"]:
            seen_sectors.add(c["sector"])
    return picked


def _run_shadow_top3(days: list, ohlc_df: pd.DataFrame, exclude_fn,
                     thresholds: Optional[dict] = None,
                     min_rank: float = 40.0) -> tuple:
    """한 룰에 대해 shadow Top3 백테스트. returns (trades, picks_by_day)."""
    trades = []
    picks_by_day = {}
    days_dict_seq = [(d[0], d[2]) for d in days]
    for i in range(1, len(days)):
        ymd, rows_list, _ = days[i]
        top3 = _pick_top3_with_risk_exclude(rows_list, exclude_fn,
                                             thresholds, min_rank)
        picks_by_day[ymd] = [c["code"] for c in top3]
        if not top3:
            continue
        entry_date = pd.Timestamp(ymd[:4] + "-" + ymd[4:6] + "-" + ymd[6:])
        for pick in top3:
            code = pick["code"]
            result = simulate_ohlc(code, entry_date, pick["entry"], pick["tp1"],
                                   pick["stop"], HORIZON, ohlc_df,
                                   tp2=pick.get("tp2", 0.0),
                                   tp3=pick.get("tp3", 0.0))
            if result["outcome"] == "NODATA":
                result = simulate_close_only(code, i, pick["entry"], pick["tp1"],
                                             pick["stop"], HORIZON, days_dict_seq,
                                             tp2=pick.get("tp2", 0.0),
                                             tp3=pick.get("tp3", 0.0))
            if result["outcome"] == "NODATA":
                continue
            if result["outcome"] == "NOT_FILLED":
                trades.append({"date": ymd, "code": code,
                               "outcome": "NOT_FILLED", "ret_pct": 0.0})
                continue
            ret_pct = (result["exit_price"] / pick["entry"] - 1) * 100
            trades.append({"date": ymd, "code": code,
                           "outcome": result["outcome"],
                           "ret_pct": round(ret_pct, 2)})
    return trades, picks_by_day


def compute_pre_entry_risk_shadow(days: list, ohlc_df: pd.DataFrame,
                                  baseline_trades: list,
                                  thresholds: Optional[dict] = None,
                                  min_rank: float = 40.0) -> dict:
    """4개 위험 제외 룰 비교 shadow 백테스트.
    
    [중요] baseline_trades 는 읽기 전용. shadow Top3를 룰마다 다시 뽑아
    백테스트한 뒤 baseline 대비 ΔEV / 구성변경률 비교.
    
    measurement-only: ranking / picks / buy price / production 추천에
    어떤 영향도 주지 않는다.
    """
    if not PRE_ENTRY_RISK_SHADOW_ENABLED:
        return {"enabled": False}

    rules = ["A_struct70_85", "B_red", "C_orange", "D_red_orange"]
    rule_desc = {
        "A_struct70_85": f"STRUCT {PRE_RISK_STRUCT_LO:.0f}~{PRE_RISK_STRUCT_HI:.0f} (struct_risk와 동일, 비교 기준)",
        "B_red":         f"STRUCT {PRE_RISK_STRUCT_LO:.0f}~{PRE_RISK_STRUCT_HI:.0f} AND VWAP_GAP>{PRE_RISK_VWAP_RED:.0f}",
        "C_orange":      f"STRUCT<{PRE_RISK_STRUCT_TOP:.0f} AND VWAP_GAP>{PRE_RISK_VWAP_ORANGE:.0f}",
        "D_red_orange":  f"B ∪ C",
    }

    base_summ = _summarize_shadow_trades(baseline_trades)
    if base_summ.get("n_filled", 0) == 0:
        return {"enabled": True, "note": "baseline_no_filled_trades"}

    base_picks_by_day: dict = {}
    for t in baseline_trades:
        base_picks_by_day.setdefault(str(t.get("date", "")), set()).add(
            str(t.get("code", "")).zfill(6))

    results_by_rule = {}
    for rule in rules:
        ex_fn = _make_risk_exclude_fn(rule)
        shadow_trades, shadow_picks = _run_shadow_top3(
            days, ohlc_df, ex_fn, thresholds, min_rank)
        shadow_summ = _summarize_shadow_trades(shadow_trades)

        if shadow_summ.get("n_filled", 0) == 0:
            results_by_rule[rule] = {
                "description": rule_desc[rule],
                "note": "shadow_no_filled_trades",
            }
            continue

        # 구성 변경률
        changed = 0
        total = 0
        for ymd, codes in shadow_picks.items():
            bc = base_picks_by_day.get(ymd, set())
            if bc or codes:
                total += 1
                if set(codes) != bc:
                    changed += 1
        changed_pick_rate = round(changed / total, 4) if total > 0 else 0.0

        d_ev = round(shadow_summ["ev"] - base_summ["ev"], 2)
        d_tp1 = round(shadow_summ["tp1_rate"] - base_summ["tp1_rate"], 4)
        d_stop = round(shadow_summ["stop_rate"] - base_summ["stop_rate"], 4)
        d_fill = round(shadow_summ["fill_rate"] - base_summ["fill_rate"], 4)
        d_non_win = round(shadow_summ["avg_non_win"] - base_summ["avg_non_win"], 2)

        # 단일 백테스트 통과 — production_candidate 아님
        single_ok = bool(
            d_ev > 0
            and d_non_win > 0
            and d_fill >= -0.07
            and changed_pick_rate <= 0.50
        )

        results_by_rule[rule] = {
            "description": rule_desc[rule],
            "shadow_ev": shadow_summ["ev"],
            "delta_ev": d_ev,
            "shadow_tp1_rate": shadow_summ["tp1_rate"],
            "delta_tp1_rate": d_tp1,
            "shadow_stop_rate": shadow_summ["stop_rate"],
            "delta_stop_rate": d_stop,
            "shadow_fill_rate": shadow_summ["fill_rate"],
            "delta_fill_rate": d_fill,
            "shadow_non_win_avg_ret": shadow_summ["avg_non_win"],
            "delta_non_win_avg_ret": d_non_win,
            "changed_pick_rate": changed_pick_rate,
            "n_shadow_filled": shadow_summ["n_filled"],
            "single_backtest_ok": single_ok,
        }

    # 최적 룰 추천 — sum_ret 아니라 (ΔEV * efficiency) 같은 단순 지표 안 만들고,
    # 정직하게 후보만 표시. 판단은 RWF + 사람이.
    best_by_ev = None
    best_by_efficiency = None  # ΔEV / changed_pick_rate (변경 1%당 EV 개선)
    for rule, r in results_by_rule.items():
        if "delta_ev" not in r:
            continue
        if best_by_ev is None or r["delta_ev"] > results_by_rule[best_by_ev]["delta_ev"]:
            best_by_ev = rule
        eff = r["delta_ev"] / max(r["changed_pick_rate"], 0.01)
        if best_by_efficiency is None:
            best_by_efficiency = (rule, eff)
        elif eff > best_by_efficiency[1]:
            best_by_efficiency = (rule, eff)

    return {
        "enabled": True,
        "mode": "single_backtest_shadow",
        "baseline_ev": base_summ["ev"],
        "baseline_tp1_rate": base_summ["tp1_rate"],
        "baseline_stop_rate": base_summ["stop_rate"],
        "baseline_fill_rate": base_summ["fill_rate"],
        "baseline_non_win_avg_ret": base_summ["avg_non_win"],
        "n_baseline_filled": base_summ["n_filled"],
        "rules": results_by_rule,
        "best_by_delta_ev": best_by_ev,
        "best_by_efficiency": best_by_efficiency[0] if best_by_efficiency else None,
        # ── production 게이트: 항상 False (별도 RWF + 사람 검토 필요) ──
        "rwf_required": True,
        "production_candidate": False,
        "production_gate_reason": "single_backtest_only__rwf_required",
        "rwf_validated": False,
        "note": (
            "measurement-only; does not alter ranking, picks, buy price, "
            "or production recommendations. Compares 4 risk-exclusion rules "
            "to find one with minimal pick-change while maximizing loss "
            "avoidance. RWF validation required before production."
        ),
    }


def summarize_trades(trades: list) -> dict:
    # NOT_FILLED는 별도 집계 — EV는 체결된 거래만
    not_filled = [t for t in trades if t["outcome"] == "NOT_FILLED"]
    filled = [t for t in trades if t["outcome"] != "NOT_FILLED"]
    n_all = len(trades)
    n = len(filled)
    if n == 0:
        return {"n": 0, "n_all": n_all, "n_not_filled": len(not_filled),
                "note": "no_filled_trades",
                "ladder": {"ladder_n": 0, "note": "no_filled_trades"},
                # v3.8.3 top-level alias (no-filled 케이스 — UI KeyError 방지)
                "tp2_rate": 0.0,
                "tp3_rate": 0.0,
                "tp3_before_stop_rate": 0.0,
                "avg_max_high_pct": 0.0,
                "avg_max_high_pct_tp3_miss": 0.0}
    wins = [t for t in filled if t["outcome"] == "WIN"]
    losses = [t for t in filled if t["outcome"] == "LOSS"]
    opens = [t for t in filled if t["outcome"] == "OPEN"]
    tp1_r = len(wins) / n
    stop_r = len(losses) / n
    open_r = len(opens) / n
    avg_tp1 = sum(t["ret_pct"] for t in wins) / len(wins) if wins else 0.0
    avg_stop = sum(t["ret_pct"] for t in losses) / len(losses) if losses else 0.0
    avg_open = sum(t["ret_pct"] for t in opens) / len(opens) if opens else 0.0
    avg_all = sum(t["ret_pct"] for t in filled) / n
    ev = (tp1_r * avg_tp1 + stop_r * avg_stop + open_r * avg_open) - COST_PCT
    ohlc_count = sum(1 for t in filled if t["method"].startswith("ohlc"))
    fill_rate = n / n_all if n_all > 0 else 1.0

    # ── v3.8.3: TP ladder 집계 ──
    # ladder 필드가 있는 trade에만 집계 (예전 trade list와의 호환 위해)
    ladder_trades = [t for t in filled if "tp3_hit" in t]
    ladder_n = len(ladder_trades)
    if ladder_n > 0:
        # 도달률 (체결된 거래 기준, 손절 무관 — 전체 horizon 내 도달 여부)
        tp1_hit_n = sum(1 for t in ladder_trades if t.get("tp1_hit"))
        tp2_hit_n = sum(1 for t in ladder_trades if t.get("tp2_hit"))
        tp3_hit_n = sum(1 for t in ladder_trades if t.get("tp3_hit"))
        # tp3 has valid target? (추천매도가3 = 0/NaN인 종목은 분모 제외)
        ladder_tp3_avail = [t for t in ladder_trades if _fnum(t.get("tp3", 0)) > 0]
        ladder_tp2_avail = [t for t in ladder_trades if _fnum(t.get("tp2", 0)) > 0]
        # 손절 이전 도달 (실전 트레이딩 기준)
        tp1_before_stop_n = sum(1 for t in ladder_trades if t.get("tp1_before_stop"))
        tp2_before_stop_n = sum(1 for t in ladder_trades if t.get("tp2_before_stop"))
        tp3_before_stop_n = sum(1 for t in ladder_trades if t.get("tp3_before_stop"))
        # 전환율 (계단별)
        tp1_to_tp2 = (
            sum(1 for t in ladder_trades if t.get("tp1_hit") and t.get("tp2_hit"))
            / tp1_hit_n if tp1_hit_n > 0 else 0.0
        )
        tp2_to_tp3 = (
            sum(1 for t in ladder_trades if t.get("tp2_hit") and t.get("tp3_hit"))
            / tp2_hit_n if tp2_hit_n > 0 else 0.0
        )
        # 극값 평균
        avg_max_high = (
            sum(_fnum(t.get("max_high_pct", 0)) for t in ladder_trades) / ladder_n
        )
        avg_min_low = (
            sum(_fnum(t.get("min_low_pct", 0)) for t in ladder_trades) / ladder_n
        )
        # TP3 실패 종목(=tp3_hit False)의 평균 최대상승률
        tp3_miss = [t for t in ladder_trades if not t.get("tp3_hit")]
        avg_max_high_tp3_miss = (
            sum(_fnum(t.get("max_high_pct", 0)) for t in tp3_miss) / len(tp3_miss)
            if tp3_miss else 0.0
        )
        ladder_summary = {
            "ladder_n": ladder_n,
            "tp1_reach_rate":   round(tp1_hit_n / ladder_n, 4),
            "tp2_reach_rate":   round(tp2_hit_n / len(ladder_tp2_avail), 4) if ladder_tp2_avail else 0.0,
            "tp3_reach_rate":   round(tp3_hit_n / len(ladder_tp3_avail), 4) if ladder_tp3_avail else 0.0,
            "tp1_before_stop_rate": round(tp1_before_stop_n / ladder_n, 4),
            "tp2_before_stop_rate": round(tp2_before_stop_n / len(ladder_tp2_avail), 4) if ladder_tp2_avail else 0.0,
            "tp3_before_stop_rate": round(tp3_before_stop_n / len(ladder_tp3_avail), 4) if ladder_tp3_avail else 0.0,
            "tp1_to_tp2_conv":  round(tp1_to_tp2, 4),
            "tp2_to_tp3_conv":  round(tp2_to_tp3, 4),
            "avg_max_high_pct": round(avg_max_high, 2),
            "avg_min_low_pct":  round(avg_min_low, 2),
            "avg_max_high_pct_tp3_miss": round(avg_max_high_tp3_miss, 2),
            "n_tp2_avail":      len(ladder_tp2_avail),
            "n_tp3_avail":      len(ladder_tp3_avail),
        }
    else:
        ladder_summary = {"ladder_n": 0, "note": "no_ladder_data"}

    return {
        "n": n,
        "n_all_picks": n_all,
        "n_not_filled": len(not_filled),
        "fill_rate": round(fill_rate, 4),
        "tp1_rate": round(tp1_r, 4),
        "stop_rate": round(stop_r, 4),
        "open_rate": round(open_r, 4),
        "avg_tp1": round(avg_tp1, 2),
        "avg_stop": round(avg_stop, 2),
        "avg_open": round(avg_open, 2),
        "avg_all": round(avg_all, 2),
        "ev": round(ev, 2),
        "ohlc_coverage": round(ohlc_count / n, 3),
        "ladder": ladder_summary,
        # v3.8.3 top-level alias — downstream(combo_optimizer/tab_perf/check_contract_gate)
        # 에서 summary["ladder"]["..."] 깊은 lookup 없이 직접 접근하도록 노출.
        # 주의: tp1_rate(=first-touch WIN rate, 기존 의미) vs tp2_rate/tp3_rate(=ladder
        # reach rate, 신규 의미) 는 의미가 다르다. 같은 어휘지만 다른 정의임.
        "tp2_rate":                  ladder_summary.get("tp2_reach_rate", 0.0),
        "tp3_rate":                  ladder_summary.get("tp3_reach_rate", 0.0),
        "tp3_before_stop_rate":      ladder_summary.get("tp3_before_stop_rate", 0.0),
        "avg_max_high_pct":          ladder_summary.get("avg_max_high_pct", 0.0),
        "avg_max_high_pct_tp3_miss": ladder_summary.get("avg_max_high_pct_tp3_miss", 0.0),
    }


# ═══════════════════════════════════════════════════
#  [핵심 #2] 최강 임계값 튜닝 그리드서치
# ═══════════════════════════════════════════════════

def tune_strong_thresholds(days: list, ohlc_df: pd.DataFrame) -> list:
    results = []
    grid = []
    for sm in [65, 70, 75, 80]:
        for sb in [70, 75, 80]:
            for sg in [3, 5, 7, 10]:
                for sr in [0.8, 1.0, 1.2]:
                    grid.append({
                        "strong_mean": sm, "strong_bal": sb,
                        "strong_gap": sg,  "strong_rr": sr,
                        "ok_min": 50.0, "ok_bal": 70.0, "ok_gap": 5.0,
                        "chase_gap": 5.0, "chase_mean": 60.0,
                    })

    print(f"    그리드: {len(grid)}개 조합")
    for idx, th in enumerate(grid):
        result = daily_top3_backtest(days, ohlc_df, thresholds=th, min_rank=40.0)
        summary = summarize_trades(result["trades"])
        if summary.get("n", 0) < 20:
            continue
        # 라벨별 분리
        strong_trades = [t for t in result["trades"] if t["label"] == "🏆 최강"]
        strong_summary = summarize_trades(strong_trades)
        results.append({
            "thresholds": {k: v for k, v in th.items() if k.startswith("strong")},
            "top3_summary": summary,
            "strong_only": strong_summary,
        })
        if (idx + 1) % 24 == 0:
            print(f"    진행: {idx+1}/{len(grid)}")
    # 종합 EV 기준 정렬
    results.sort(key=lambda r: -r["top3_summary"]["ev"])
    return results


def walk_forward_validate(days: list, ohlc_df: pd.DataFrame,
                          horizon_override: Optional[int] = None) -> dict:
    """Walk-forward 오버피팅 검증.

    전체 기간을 2등분:
      - IN-SAMPLE (앞): 튜닝 그리드서치로 최적 threshold 탐색
      - OUT-OF-SAMPLE (뒤): 그 threshold로 성능 재측정

    IS에서 좋은 조합이 OOS에서도 EV+면 일반화 O,
    OOS에서 EV 뒤집히면 오버피팅 증거.

    Horizon이 기본값(20)이면 IS/OOS 각각 샘플 확보 어려움.
    horizon_override=5 등으로 줄여서 일반화 가능성 점검 가능.
    """
    global HORIZON
    original_horizon = HORIZON
    if horizon_override:
        HORIZON = horizon_override

    try:
        n = len(days)
        # v3.7.6: 루프 종료점 제거로 horizon 못 채워도 거래 가능 → 최소 8일만 필요
        if n < 8:
            return {
                "error": f"데이터 부족 (총 {n}일, 최소 8일 필요)",
            }
        mid = n // 2
        is_days = days[:mid]
        oos_days = days[mid:]
        print(f"    horizon={HORIZON}일")
        print(f"    IS: {is_days[0][0]} ~ {is_days[-1][0]} ({len(is_days)}일)")
        print(f"    OOS: {oos_days[0][0]} ~ {oos_days[-1][0]} ({len(oos_days)}일)")

        print(f"    IS 튜닝 그리드서치...")
        is_tuning = tune_strong_thresholds(is_days, ohlc_df)
        if not is_tuning:
            return {"error": "IS 튜닝 결과 없음 (샘플 부족)",
                    "is_period": f"{is_days[0][0]} ~ {is_days[-1][0]}",
                    "oos_period": f"{oos_days[0][0]} ~ {oos_days[-1][0]}",
                    "horizon_used": HORIZON}

        print(f"    OOS 검증 (IS Top 5 조합)...")
        wf_results = []
        for i, candidate in enumerate(is_tuning[:5]):
            is_th = candidate["thresholds"]
            full_th = {
                **is_th,
                "ok_min": 50.0, "ok_bal": 70.0, "ok_gap": 5.0,
                "chase_gap": 5.0, "chase_mean": 60.0,
            }
            oos_result = daily_top3_backtest(oos_days, ohlc_df,
                                              thresholds=full_th, min_rank=40.0)
            oos_summary = summarize_trades(oos_result["trades"])
            is_ev = candidate["top3_summary"].get("ev", 0)
            oos_ev = oos_summary.get("ev", 0)
            oos_n = oos_summary.get("n", 0)

            # [v3.7.14] OOS도 자본 시뮬로 재측정 — 실전 운용 가능성 관점
            oos_capital = simulate_capital_portfolio(
                oos_result["trades"], initial_capital=10_000_000, max_positions=3
            )
            oos_capital_return = oos_capital.get("total_return_pct", 0) if oos_capital else 0
            oos_capital_mdd = oos_capital.get("max_drawdown_pct", 0) if oos_capital else 0
            oos_capital_n_exec = oos_capital.get("n_filled", 0) if oos_capital else 0

            wf_results.append({
                "rank_in_is": i + 1,
                "thresholds": is_th,
                # 신호 기준 (알파 품질)
                "is_summary": candidate["top3_summary"],
                "oos_summary": oos_summary,
                "generalizes": (oos_n >= 5 and oos_ev > 0),
                "decay": round(is_ev - oos_ev, 2),
                # [v3.7.14] 자본 기준 (실전 운용 가능성)
                "oos_capital": {
                    "total_return_pct": oos_capital_return,
                    "max_drawdown_pct": oos_capital_mdd,
                    "n_executed": oos_capital_n_exec,
                },
                "capital_generalizes": (oos_capital_n_exec >= 3 and oos_capital_return > 0),
            })

        return {
            "is_period": f"{is_days[0][0]} ~ {is_days[-1][0]}",
            "oos_period": f"{oos_days[0][0]} ~ {oos_days[-1][0]}",
            "is_days": len(is_days),
            "oos_days": len(oos_days),
            "horizon_used": HORIZON,
            "results": wf_results,
            # [v3.7.14] 최상단 요약 — 신호/자본 둘 다
            "walkforward_signal_summary": {
                "n_results": len(wf_results),
                "n_generalizes_signal": sum(1 for r in wf_results if r.get("generalizes")),
                "avg_oos_ev": round(
                    sum(r["oos_summary"].get("ev", 0) for r in wf_results) / len(wf_results), 2
                ) if wf_results else 0,
            },
            "walkforward_capital_summary": {
                "n_results": len(wf_results),
                "n_generalizes_capital": sum(
                    1 for r in wf_results if r.get("capital_generalizes")
                ),
                "avg_oos_return": round(
                    sum(r["oos_capital"].get("total_return_pct", 0) for r in wf_results)
                    / len(wf_results), 2
                ) if wf_results else 0,
                "avg_oos_mdd": round(
                    sum(r["oos_capital"].get("max_drawdown_pct", 0) for r in wf_results)
                    / len(wf_results), 2
                ) if wf_results else 0,
            },
        }
    finally:
        HORIZON = original_horizon


def rolling_walk_forward(days: list, ohlc_df: pd.DataFrame,
                          n_folds: int = 3,
                          is_ratio: float = 0.6,
                          horizon_override: Optional[int] = None) -> dict:
    """Rolling walk-forward — 단일 split 대신 여러 구간 반복 검증 (v3.7.8).

    방식:
      전체 기간을 n_folds 개 폴드로 나눔.
      각 폴드마다 앞 is_ratio 만큼을 IS, 나머지를 OOS로 사용.
      폴드 간에는 시작점이 밀려가는 방식 (expanding or sliding).

    장세가 바뀔 때도 엔진이 안정적으로 작동하는지 진짜 검증 가능.
    """
    global HORIZON
    original_horizon = HORIZON
    if horizon_override:
        HORIZON = horizon_override

    try:
        n = len(days)
        if n < 20:
            return {"error": f"데이터 부족 (총 {n}일, rolling은 최소 20일 권장)"}

        # 폴드 크기 = 전체를 n_folds로 나눈 뒤 IS 비율 적용
        # expanding 방식: 폴드마다 IS 구간이 점점 커짐
        fold_size = n // (n_folds + 1)  # 마지막 OOS 확보 위해 +1
        folds = []
        for i in range(n_folds):
            # expanding IS: 0 ~ fold_size * (i+1)
            is_end = fold_size * (i + 1)
            # IS 내부에서 is_ratio만큼 훈련, 나머지(폴드사이즈)를 OOS
            is_start = 0
            oos_start = is_end
            oos_end = min(is_end + fold_size, n)
            if oos_end - oos_start < 3:
                continue
            folds.append({
                "fold": i + 1,
                "is_range": (is_start, is_end),
                "oos_range": (oos_start, oos_end),
            })

        print(f"    rolling folds: {len(folds)}개 (horizon={HORIZON})")
        fold_results = []
        for f in folds:
            is_days = days[f["is_range"][0]:f["is_range"][1]]
            oos_days = days[f["oos_range"][0]:f["oos_range"][1]]
            print(f"    fold {f['fold']}: IS {is_days[0][0]}~{is_days[-1][0]} "
                  f"({len(is_days)}일) → OOS {oos_days[0][0]}~{oos_days[-1][0]} "
                  f"({len(oos_days)}일)")

            # 이 폴드에서는 그리드서치 대신 '현재 기본 threshold' 또는 
            # 간이 튜닝 Top 1만 사용 (속도 위해)
            is_tuning = tune_strong_thresholds(is_days, ohlc_df)
            if not is_tuning:
                fold_results.append({
                    "fold": f["fold"],
                    "is_period": f"{is_days[0][0]}~{is_days[-1][0]}",
                    "oos_period": f"{oos_days[0][0]}~{oos_days[-1][0]}",
                    "error": "IS 샘플 부족",
                })
                continue

            best = is_tuning[0]
            is_th = best["thresholds"]
            full_th = {
                **is_th,
                "ok_min": 50.0, "ok_bal": 70.0, "ok_gap": 5.0,
                "chase_gap": 5.0, "chase_mean": 60.0,
            }
            oos_result = daily_top3_backtest(oos_days, ohlc_df,
                                              thresholds=full_th, min_rank=40.0)
            oos_summary = summarize_trades(oos_result["trades"])
            is_ev = best["top3_summary"].get("ev", 0)
            oos_ev = oos_summary.get("ev", 0)
            oos_n = oos_summary.get("n", 0)
            fold_results.append({
                "fold": f["fold"],
                "is_period": f"{is_days[0][0]}~{is_days[-1][0]}",
                "oos_period": f"{oos_days[0][0]}~{oos_days[-1][0]}",
                "best_thresholds": is_th,
                "is_ev": is_ev,
                "oos_ev": oos_ev,
                "oos_n": oos_n,
                "generalizes": (oos_n >= 3 and oos_ev > 0),
                "decay": round(is_ev - oos_ev, 2),
            })

        # 종합
        if fold_results:
            valid = [f for f in fold_results if "error" not in f]
            n_gen = sum(1 for f in valid if f.get("generalizes"))
            avg_is = sum(f["is_ev"] for f in valid) / len(valid) if valid else 0
            avg_oos = sum(f["oos_ev"] for f in valid) / len(valid) if valid else 0
            summary = {
                "n_folds": len(fold_results),
                "n_valid": len(valid),
                "n_generalizes": n_gen,
                "avg_is_ev": round(avg_is, 2),
                "avg_oos_ev": round(avg_oos, 2),
                "robust": n_gen >= len(valid) * 0.6,  # 60% 이상 일반화
            }
        else:
            summary = {"error": "폴드 결과 없음"}

        return {
            "horizon_used": HORIZON,
            "n_folds_requested": n_folds,
            "is_ratio": is_ratio,
            "summary": summary,
            "folds": fold_results,
        }
    finally:
        HORIZON = original_horizon


# ═══════════════════════════════════════════════════
#  리포트 빌드 & 저장
# ═══════════════════════════════════════════════════

def build_report(days: list, ohlc_df: pd.DataFrame, tune: bool = False,
                 walkforward: bool = False, rolling: bool = False) -> dict:
    date_range = [days[0][0][:4] + "-" + days[0][0][4:6] + "-" + days[0][0][6:],
                  days[-1][0][:4] + "-" + days[-1][0][4:6] + "-" + days[-1][0][6:]] if days else ["", ""]

    out: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "version": BACKTEST_ENGINE_VERSION,        # [v3.9.2] 단일 출처 사용
        "engine_version": BACKTEST_ENGINE_VERSION, # [v3.9.2] 명시적 alias
        "horizon_bdays": HORIZON,
        "trade_cost_pct": COST_PCT,
        "days_covered": len(days),
        "ohlc_coverage_note": (
            f"OHLC parquet 종목수: "
            f"{ohlc_df['종목코드'].nunique() if not ohlc_df.empty else 0} — "
            f"이 종목들만 장중 고가/저가 터치 판정, 나머지는 종가 폴백"
        ),
        # [v3.7.14] 구조화된 methodology 블록 (모든 검증에 공통 적용)
        "methodology": {
            "horizon_days": HORIZON,
            "fill_window_days": 3,
            "fee_pct_roundtrip": COST_PCT,
            "max_positions_top1": 1,
            "max_positions_top3": 3,
            "dedup_same_ticker": True,
            "reentry_after_exit": True,
            "selection_mode": "top1_first_then_top3_fallback",
            "date_range": date_range,
            "description": (
                "매일 pick_top1()로 🏆 최강 1위 추천 → fill_window 3일 내 체결 검증 "
                "→ OHLC 기반 TP1/Stop 터치 판정 (장중 고가/저가) → horizon 10일 미도달시 종가 마감. "
                "자본 시뮬은 max_positions=1, 이미 보유 중이면 스킵, exit 후 재진입 가능."
            ),
        },
    }

    print("  ▶ 기본 threshold로 일자별 Top3 백테스트...")
    bt = daily_top3_backtest(days, ohlc_df, thresholds=None, min_rank=40.0)
    out["daily_top3_backtest"] = summarize_trades(bt["trades"])
    out["daily_picks_log"] = bt["daily_picks"]
    out["all_trades"] = bt["trades"]

    # [v3.9.0] ENTRY_MODE shadow — baseline 불변, 병렬 계산만
    print("  ▶ [v3.9.0] ENTRY_MODE shadow 백테스트 (NOT_FILLED chase 시뮬)...")
    try:
        out["entry_mode_shadow"] = compute_entry_mode_shadow(
            bt["trades"], ohlc_df, days)
        _sh = out["entry_mode_shadow"]
        if _sh.get("enabled"):
            print(f"     shadow: eligible {_sh['n_shadow_eligible']} → "
                  f"chase 체결 {_sh['extra_fills']} "
                  f"(WIN {_sh['extra_wins']} / LOSS {_sh['extra_losses']}), "
                  f"sum_ret {_sh['extra_sum_ret']}%, "
                  f"production_candidate={_sh['production_candidate']}")
    except Exception as e:
        print(f"     ⚠ entry_mode_shadow 계산 실패 (baseline 영향 없음): {e}")
        out["entry_mode_shadow"] = {"enabled": False, "error": str(e)}

    # [v3.9.1] STRUCT risk shadow — baseline 불변, STRUCT 70~85 제외 병렬 계산
    print("  ▶ [v3.9.1] STRUCT risk shadow 백테스트 (STRUCT 70~85 제외 시뮬)...")
    try:
        out["struct_risk_shadow"] = compute_struct_risk_shadow(
            days, ohlc_df, bt["trades"], thresholds=None, min_rank=40.0)
        _sr = out["struct_risk_shadow"]
        if _sr.get("enabled") and "delta_ev" in _sr:
            print(f"     shadow: ΔEV {_sr['delta_ev']:+.2f} "
                  f"(baseline {_sr['baseline_ev']} → shadow {_sr['shadow_ev']}), "
                  f"Δnon_win {_sr['delta_non_win_avg_ret']:+.2f}, "
                  f"구성변경 {_sr['changed_pick_rate']*100:.1f}%, "
                  f"single_backtest_ok={_sr['single_backtest_ok']} "
                  f"(production_candidate={_sr['production_candidate']} — RWF 검증 필요)")
    except Exception as e:
        print(f"     ⚠ struct_risk_shadow 계산 실패 (baseline 영향 없음): {e}")
        out["struct_risk_shadow"] = {"enabled": False, "error": str(e)}

    # [v3.9.2] PRE_ENTRY_RISK shadow — 4개 룰 비교 (RED/ORANGE 제외 효과)
    print("  ▶ [v3.9.2] PRE_ENTRY_RISK shadow 백테스트 (4개 룰 비교)...")
    try:
        out["pre_entry_risk_shadow"] = compute_pre_entry_risk_shadow(
            days, ohlc_df, bt["trades"], thresholds=None, min_rank=40.0)
        _pr = out["pre_entry_risk_shadow"]
        if _pr.get("enabled") and "rules" in _pr:
            print(f"     baseline EV: {_pr['baseline_ev']:+.2f}")
            for rule_name in ["A_struct70_85", "B_red", "C_orange", "D_red_orange"]:
                r = _pr["rules"].get(rule_name, {})
                if "delta_ev" in r:
                    mark = "★" if rule_name == _pr.get("best_by_efficiency") else " "
                    print(f"     {mark} {rule_name:14s}: ΔEV {r['delta_ev']:+.2f}, "
                          f"Δnon_win {r['delta_non_win_avg_ret']:+.2f}, "
                          f"구성변경 {r['changed_pick_rate']*100:.1f}%, "
                          f"ok={r['single_backtest_ok']}")
            print(f"     best_by_ΔEV: {_pr.get('best_by_delta_ev')} / "
                  f"best_by_efficiency: {_pr.get('best_by_efficiency')} "
                  f"(production_candidate={_pr['production_candidate']})")
    except Exception as e:
        print(f"     ⚠ pre_entry_risk_shadow 계산 실패 (baseline 영향 없음): {e}")
        out["pre_entry_risk_shadow"] = {"enabled": False, "error": str(e)}

    by_label: dict[str, list] = defaultdict(list)
    for t in bt["trades"]:
        by_label[t["label"]].append(t)
    out["by_label_top3"] = {
        lbl: summarize_trades(sub) for lbl, sub in by_label.items()
    }

    # [v3.7.14] Top1 백테스트 — 신호 성과 vs 실집행 성과 완전 분리
    # signal_top1 = "알파 품질"   (백테스트 trades 기준)
    # capital_top1 = "실전 운용 가능성" (simulate_capital_portfolio 기준)
    print("  ▶ [Top1] 신호 성과 백테스트 (pick_top1 기반)...")
    bt1 = daily_top1_backtest(days, ohlc_df, thresholds=None, min_rank=40.0)

    # 기존 키도 유지 (하위호환)
    out["daily_top1_backtest"] = summarize_trades(bt1["trades"])
    out["all_trades_top1"] = bt1["trades"]
    out["daily_picks_log_top1"] = bt1["daily_picks"]

    # [v3.7.14] 명시적 signal_top1 블록 — 신호 품질 지표만
    t1_summary = summarize_trades(bt1["trades"])
    out["signal_top1"] = {
        "label": "신호 성과 (알파 품질)",
        "description": "pick_top1로 뽑힌 모든 추천을 독립적으로 horizon 추적한 결과. "
                        "자본 제약 없음 — 순수 시그널 품질 측정용.",
        "n_signals_total": t1_summary.get("n_all_picks", 0),
        "n_filled": t1_summary.get("n", 0),
        "fill_rate": t1_summary.get("fill_rate", 0),
        "tp1_rate": t1_summary.get("tp1_rate", 0),
        "stop_rate": t1_summary.get("stop_rate", 0),
        "open_rate": t1_summary.get("open_rate", 0),
        "avg_all_pct": t1_summary.get("avg_all", 0),
        "ev_net_pct": t1_summary.get("ev", 0),
    }

    if tune:
        print("  ▶ 최강 임계값 튜닝 그리드서치...")
        tuning = tune_strong_thresholds(days, ohlc_df)
        out["tuning_results"] = tuning[:20]
        out["tuning_best"] = tuning[0] if tuning else None

    if walkforward:
        print("  ▶ Walk-forward 오버피팅 검증...")
        wf = walk_forward_validate(days, ohlc_df)

        def _wf_failed(w):
            return (w is None or "error" in w or not w.get("results"))

        if _wf_failed(wf):
            print("    → horizon=20 실패, horizon=10으로 재시도")
            wf = walk_forward_validate(days, ohlc_df, horizon_override=10)
        if _wf_failed(wf):
            print("    → horizon=10 실패, horizon=5로 재시도")
            wf = walk_forward_validate(days, ohlc_df, horizon_override=5)
        if _wf_failed(wf):
            print("    → horizon=5도 실패, horizon=3으로 최종 재시도")
            wf = walk_forward_validate(days, ohlc_df, horizon_override=3)
        out["walk_forward"] = wf

    if rolling:
        print("  ▶ Rolling walk-forward (3 folds, horizon 10)...")
        # [v3.7.13] 메인/walk-forward와 horizon 일관성 유지 (이전엔 5로 축소돼있었음)
        rwf = rolling_walk_forward(days, ohlc_df, n_folds=3,
                                     horizon_override=10)
        out["rolling_walk_forward"] = rwf

    return out


def save_report(report: dict, data_dir: Path):
    today = datetime.now().strftime("%Y%m%d")
    light = {k: v for k, v in report.items()
             if k not in ("all_trades", "daily_picks_log")}
    for name in (f"backtest_validation_{today}.json",
                 "backtest_validation_latest.json"):
        with open(data_dir / name, "w", encoding="utf-8") as f:
            json.dump(light, f, ensure_ascii=False, indent=2)

    if report.get("all_trades"):
        trades_path = data_dir / f"backtest_top3_trades_{today}.csv"
        keys = list(report["all_trades"][0].keys())
        with open(trades_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(report["all_trades"])
        print(f"  · {trades_path.name} ({len(report['all_trades'])}건)")

    # [v3.7.11] Top1 거래 로그 별도 CSV
    if report.get("all_trades_top1"):
        trades1_path = data_dir / f"backtest_top1_trades_{today}.csv"
        trades1_latest = data_dir / "backtest_top1_trades_latest.csv"
        keys = list(report["all_trades_top1"][0].keys())
        for p in (trades1_path, trades1_latest):
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(report["all_trades_top1"])
        print(f"  · {trades1_path.name} ({len(report['all_trades_top1'])}건)")

    if report.get("tuning_results"):
        tune_path = data_dir / f"backtest_tuning_{today}.json"
        with open(tune_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": report["generated_at"],
                "best": report.get("tuning_best"),
                "top20": report.get("tuning_results"),
            }, f, ensure_ascii=False, indent=2)
        print(f"  · {tune_path.name}")

    # [v3.7.7] Walk-forward 결과를 별도 JSON으로 — 주장이 아닌 증거
    wf = report.get("walk_forward")
    if wf and wf.get("results"):
        wf_path = data_dir / f"backtest_walkforward_{today}.json"
        wf_latest = data_dir / "backtest_walkforward_latest.json"
        # [v3.7.15] 메인과 동일한 구조화 methodology dict 재사용 + 기법 설명 추가
        main_methodology = report.get("methodology", {})
        wf_methodology = dict(main_methodology) if isinstance(main_methodology, dict) else {}
        wf_methodology["validation_type"] = "walk_forward_single_split"
        wf_methodology["validation_description"] = (
            "전체 기간을 시간순 2등분: IS(앞절반) 튜닝 → OOS(뒤절반) 재측정. "
            "IS Top 5 조합이 OOS에서 EV+ 유지하면 일반화, EV 음수로 뒤집히면 오버피팅 증거."
        )
        out = {
            "generated_at": report["generated_at"],
            "methodology": wf_methodology,
            **wf,
        }
        for p in (wf_path, wf_latest):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        generalizes_n = sum(1 for r in wf["results"] if r.get("generalizes"))
        total_n = len(wf["results"])
        print(f"  · {wf_path.name} (일반화 {generalizes_n}/{total_n})")

    # [v3.7.8] Rolling walk-forward 별도 JSON 저장
    rwf = report.get("rolling_walk_forward")
    if rwf and rwf.get("folds"):
        rwf_path = data_dir / f"backtest_rolling_{today}.json"
        rwf_latest = data_dir / "backtest_rolling_latest.json"
        # [v3.7.15] 메인과 동일한 구조화 methodology dict 재사용
        main_methodology = report.get("methodology", {})
        rwf_methodology = dict(main_methodology) if isinstance(main_methodology, dict) else {}
        rwf_methodology["validation_type"] = "rolling_walk_forward_expanding"
        rwf_methodology["n_folds_target"] = rwf.get("n_folds_requested", 3)
        rwf_methodology["is_ratio"] = rwf.get("is_ratio", 0.6)
        rwf_methodology["validation_description"] = (
            "Expanding rolling walk-forward: 전체 기간을 n_folds 구간으로 나누고, "
            "각 폴드마다 이전 전체 구간을 IS로 튜닝 → 다음 구간에서 OOS 측정. "
            "여러 장세에 걸친 robust 엔진인지 확인."
        )
        rwf_out = {
            "generated_at": report["generated_at"],
            "methodology": rwf_methodology,
            **rwf,
        }
        for p in (rwf_path, rwf_latest):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(rwf_out, f, ensure_ascii=False, indent=2)
        s = rwf.get("summary", {})
        if s.get("n_valid"):
            print(f"  · {rwf_path.name} "
                  f"(일반화 {s['n_generalizes']}/{s['n_valid']} · "
                  f"평균 OOS EV {s.get('avg_oos_ev', 0):+.2f}%)")

    # [v3.7.7] 일자별 포트폴리오 수익률 CSV — "하루 3종목 묶음 EV"
    if report.get("all_trades"):
        trades = report["all_trades"]
        by_date: dict = defaultdict(list)
        for t in trades:
            by_date[t["date"]].append(t)
        daily_port = []
        for date, day_trades in sorted(by_date.items()):
            n = len(day_trades)
            if n == 0:
                continue
            # 동일 비중 포트폴리오 수익률 = 평균 ret_pct
            port_ret = sum(t["ret_pct"] for t in day_trades) / n
            port_net = sum(t["net_pct"] for t in day_trades) / n
            n_win = sum(1 for t in day_trades if t["outcome"] == "WIN")
            n_loss = sum(1 for t in day_trades if t["outcome"] == "LOSS")
            n_notfilled = sum(1 for t in day_trades if t["outcome"] == "NOT_FILLED")
            daily_port.append({
                "date": date,
                "n_picks": n,
                "n_filled": n - n_notfilled,
                "codes": ",".join(t["code"] for t in day_trades),
                "names": ",".join(t["name"] for t in day_trades),
                "labels": ",".join(t["label"] for t in day_trades),
                "n_wins": n_win,
                "n_losses": n_loss,
                "n_not_filled": n_notfilled,
                "portfolio_ret_pct": round(port_ret, 2),
                "portfolio_net_pct": round(port_net, 2),
            })
        if daily_port:
            port_path = data_dir / f"backtest_daily_portfolio_{today}.csv"
            port_latest = data_dir / "backtest_daily_portfolio_latest.csv"
            keys = list(daily_port[0].keys())
            for p in (port_path, port_latest):
                with open(p, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(daily_port)
            # [v3.7.9] 전체 포트폴리오 통계 — gross/net 둘 다
            avg_port_gross = sum(d["portfolio_ret_pct"] for d in daily_port) / len(daily_port)
            avg_port_net   = sum(d["portfolio_net_pct"] for d in daily_port) / len(daily_port)
            # "플러스 마감"은 net 기준이 정직 (비용 차감 후에도 흑자인 날)
            n_pos_days_net = sum(1 for d in daily_port if d["portfolio_net_pct"] > 0)
            n_pos_days_gross = sum(1 for d in daily_port if d["portfolio_ret_pct"] > 0)
            print(f"  · {port_path.name} ({len(daily_port)}일)")
            print(f"    → 일평균 포트 gross {avg_port_gross:+.2f}% / net {avg_port_net:+.2f}% · "
                  f"플러스마감(net) {n_pos_days_net}/{len(daily_port)}일")
            # 주 리포트에도 요약 추가 (UI에서도 읽을 수 있도록) — net 기준
            summary = {
                "n_days": len(daily_port),
                "avg_daily_portfolio_ret_gross": round(avg_port_gross, 2),
                "avg_daily_portfolio_ret_net":   round(avg_port_net, 2),
                "avg_daily_portfolio_ret":       round(avg_port_net, 2),  # 하위호환: 기본은 net
                "n_positive_days":      n_pos_days_net,
                "n_positive_days_gross": n_pos_days_gross,
                "positive_rate": round(n_pos_days_net / len(daily_port), 4),
                "cost_basis": "net (비용 차감 후)",
            }
            light["daily_portfolio_summary"] = summary

    # [v3.7.8] 자본 기반 포트폴리오 시뮬 — 중복 보유 제외 + 자본 곡선
    if report.get("all_trades"):
        cap_sim = simulate_capital_portfolio(report["all_trades"],
                                             initial_capital=10_000_000,
                                             max_positions=3)
        if cap_sim:
            cap_path = data_dir / f"backtest_capital_curve_{today}.csv"
            cap_latest = data_dir / "backtest_capital_curve_latest.csv"
            if cap_sim["curve"]:
                keys = list(cap_sim["curve"][0].keys())
                for p in (cap_path, cap_latest):
                    with open(p, "w", encoding="utf-8", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=keys)
                        w.writeheader()
                        w.writerows(cap_sim["curve"])
                print(f"  · {cap_path.name} ({len(cap_sim['curve'])}일)")
                print(f"    → [Top3] 기간수익 {cap_sim['total_return_pct']:+.2f}% · "
                      f"MDD {cap_sim['max_drawdown_pct']:.2f}%")
                light["capital_portfolio"] = {
                    "initial_capital": cap_sim["initial_capital"],
                    "final_capital": cap_sim["final_capital"],
                    "total_return_pct": cap_sim["total_return_pct"],
                    "max_drawdown_pct": cap_sim["max_drawdown_pct"],
                    "n_trades_filled": cap_sim["n_filled"],
                    "n_skipped_duplicate": cap_sim["n_skipped_duplicate"],
                    "n_skipped_not_filled": cap_sim["n_skipped_not_filled"],
                    "positive_day_rate": cap_sim["positive_day_rate"],
                    "cost_basis": f"net (거래당 {COST_PCT}% 차감)",
                }

    # [v3.7.11] Top1 전용 자본 시뮬 — README "+11.49%" 의 증거
    if report.get("all_trades_top1"):
        cap1 = simulate_capital_portfolio(report["all_trades_top1"],
                                           initial_capital=10_000_000,
                                           max_positions=1)  # Top1은 동시 1포지션
        if cap1 and cap1.get("curve"):
            cap1_path = data_dir / f"backtest_top1_capital_curve_{today}.csv"
            cap1_latest = data_dir / "backtest_top1_capital_curve_latest.csv"
            keys = list(cap1["curve"][0].keys())
            for p in (cap1_path, cap1_latest):
                with open(p, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(cap1["curve"])
            print(f"  · {cap1_path.name} ({len(cap1['curve'])}이벤트)")
            print(f"    → [Top1] 기간수익 {cap1['total_return_pct']:+.2f}% · "
                  f"MDD {cap1['max_drawdown_pct']:.2f}% · "
                  f"체결 {cap1['n_filled']}건")

            # [v3.7.14] audit log CSV 별도 저장 (왜 스킵됐는지 이유별)
            if cap1.get("audit_log"):
                audit_path = data_dir / f"backtest_top1_execution_audit_{today}.csv"
                audit_latest = data_dir / "backtest_top1_execution_audit_latest.csv"
                keys = list(cap1["audit_log"][0].keys())
                for p in (audit_path, audit_latest):
                    with open(p, "w", encoding="utf-8", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=keys)
                        w.writeheader()
                        w.writerows(cap1["audit_log"])
                print(f"  · {audit_path.name} ({len(cap1['audit_log'])}이벤트)")
                # 스킵 이유 요약 출력
                sr = cap1.get("skip_reasons_summary", {})
                print(f"    → EXECUTED {sr.get('EXECUTED',0)} · "
                      f"NOT_FILLED {sr.get('NOT_FILLED',0)} · "
                      f"SAME_TICKER_HELD {sr.get('SAME_TICKER_ALREADY_HELD',0)} · "
                      f"SLOT_FULL {sr.get('SLOT_FULL',0)}")

            # [v3.7.13] 신호 vs 실체결 차이 — 과신 방지 리포트
            daily_top1 = report.get("daily_top1_backtest", {})
            n_signal_all = daily_top1.get("n_all_picks", daily_top1.get("n", 0))
            n_signal_filled = daily_top1.get("n", 0)
            n_capital_filled = cap1["n_filled"]
            n_slot_full = cap1.get("n_skipped_slot_full", 0)
            gap_reason = (
                f"신호 {n_signal_filled}건 → 자본시뮬 {n_capital_filled}건 "
                f"(슬롯풀 {n_slot_full}건 · 미체결 {cap1['n_skipped_not_filled']}건)"
            )
            print(f"    → [Top1 체결 차이] {gap_reason}")
            light["capital_portfolio_top1"] = {
                "label": "실집행 성과 (실전 운용 가능성)",
                "description": "Top1 신호를 max_positions=1 자본 시뮬에 실제 적용한 결과. "
                                "겹치는 시그널과 미체결 때문에 신호보다 표본이 작음.",
                "initial_capital": cap1["initial_capital"],
                "final_capital": cap1["final_capital"],
                "total_return_pct": cap1["total_return_pct"],
                "max_drawdown_pct": cap1["max_drawdown_pct"],
                "n_trades_filled": cap1["n_filled"],
                "n_skipped_not_filled": cap1["n_skipped_not_filled"],
                "n_skipped_slot_full": n_slot_full,
                "positive_day_rate": cap1["positive_day_rate"],
                "cost_basis": f"net (거래당 {COST_PCT}% 차감)",
                "note": "Top1 전용 - 동시 1포지션, 자본 100% 투자",
                # [v3.7.13] 신호 vs 실체결 gap
                "signal_vs_capital_gap": {
                    "n_signals_total": n_signal_all,
                    "n_signals_filled_ok": n_signal_filled,
                    "n_capital_executed": n_capital_filled,
                    "execution_rate": (
                        round(n_capital_filled / n_signal_filled, 3)
                        if n_signal_filled > 0 else 0
                    ),
                    "explanation": gap_reason,
                },
                # [v3.7.14] 스킵 이유별 집계
                "skip_reasons_summary": cap1.get("skip_reasons_summary", {}),
            }

    # [v3.7.8] rolling walk-forward 요약을 latest JSON에도 포함 (UI 표시용)
    if report.get("rolling_walk_forward"):
        rwf = report["rolling_walk_forward"]
        if rwf.get("summary"):
            light["rolling_summary"] = rwf["summary"]

    # [v3.7.14] Confidence badge 자동 판정 — 과신 방지 핵심
    # HIGH: 실행표본 30+ AND rolling robust
    # MEDIUM: 실행표본 10+ AND rolling 폴드 3+
    # LOW: 그 이하 (표본 부족)
    cap1_light = light.get("capital_portfolio_top1", {})
    rwf_summary = light.get("rolling_summary", {})
    wf_cap_summary = None
    if report.get("walk_forward") and isinstance(report["walk_forward"], dict):
        wf_cap_summary = report["walk_forward"].get("walkforward_capital_summary", {})

    n_exec = cap1_light.get("n_trades_filled", 0)
    n_valid_folds = rwf_summary.get("n_valid", 0)
    robust = rwf_summary.get("robust", False)
    wf_cap_generalizes = (wf_cap_summary or {}).get("n_generalizes_capital", 0)

    if n_exec >= 30 and robust:
        conf_level = "HIGH"
        conf_color = "green"
    elif n_exec >= 10 and n_valid_folds >= 3:
        conf_level = "MEDIUM"
        conf_color = "yellow"
    else:
        conf_level = "LOW"
        conf_color = "red"

    reason_parts = [f"실집행 표본 {n_exec}건"]
    if robust:
        reason_parts.append("rolling robust ✅")
    else:
        reason_parts.append(f"rolling 미확정 ({n_valid_folds}폴드)")
    if wf_cap_summary:
        reason_parts.append(f"wf capital 일반화 {wf_cap_generalizes}/5")

    light["confidence"] = {
        "level": conf_level,
        "color": conf_color,
        "reason": " · ".join(reason_parts),
        "executed_trades": n_exec,
        "rolling_robust": robust,
        "rolling_valid_folds": n_valid_folds,
        "walkforward_capital_generalizes": wf_cap_generalizes,
        "threshold_rule": {
            "HIGH": "executed >= 30 AND rolling robust",
            "MEDIUM": "executed >= 10 AND rolling folds >= 3",
            "LOW": "else (표본 부족)",
        },
    }
    print(f"  · confidence: {conf_level} — {light['confidence']['reason']}")

    # latest JSON 최종 업데이트
    if daily_port or report.get("all_trades") or report.get("rolling_walk_forward"):
        with open(data_dir / "backtest_validation_latest.json",
                  "w", encoding="utf-8") as f:
            json.dump(light, f, ensure_ascii=False, indent=2)


def load_latest_report(data_dir: Path = None) -> Optional[dict]:
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    p = data_dir / "backtest_validation_latest.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="최강 임계값 그리드서치")
    ap.add_argument("--walkforward", "--wf", action="store_true",
                    help="Walk-forward 오버피팅 검증 (IS→OOS 단일 split)")
    ap.add_argument("--rolling", action="store_true",
                    help="Rolling walk-forward (3 folds 반복 검증)")
    args = ap.parse_args()

    here = Path(__file__).parent
    data_dir = here / "data"
    if not data_dir.exists():
        print(f"❌ data/ 없음: {data_dir}")
        sys.exit(1)

    print(f"▶ v2 백테스트 시작 ({data_dir})")
    print(f"  · OHLC parquet 로드")
    ohlc_df = load_ohlc(data_dir)
    if not ohlc_df.empty:
        print(f"    → {ohlc_df['종목코드'].nunique()}종목 · "
              f"{ohlc_df['Date'].min()} ~ {ohlc_df['Date'].max()}")
    else:
        print("    → 없음, 종가 폴백만")

    print(f"  · recommend CSV 로드")
    days = load_all_days(data_dir)
    print(f"    → {len(days)}일치 ({days[0][0]} ~ {days[-1][0]})")

    report = build_report(days, ohlc_df, tune=args.tune,
                          walkforward=args.walkforward,
                          rolling=args.rolling)
    print(f"  · 리포트 저장")
    save_report(report, data_dir)

    print("\n" + "═" * 70)
    print("  🎯 일자별 Top3 백테스트 결과 (매일 뽑힌 3종목만)")
    print("═" * 70)
    s = report["daily_top3_backtest"]
    if s.get("n", 0) > 0:
        print(f"  총 거래: {s['n']}건  (OHLC 커버리지: {s['ohlc_coverage']*100:.0f}%)")
        print(f"  TP1 도달률: {s['tp1_rate']*100:>5.1f}%   손절률: {s['stop_rate']*100:>5.1f}%   미확정: {s['open_rate']*100:>5.1f}%")
        print(f"  평균 승수익: {s['avg_tp1']:>+6.2f}%   평균 패손실: {s['avg_stop']:>+6.2f}%")
        print(f"  평균 전체수익: {s['avg_all']:>+6.2f}%   비용후 EV: {s['ev']:>+6.2f}%")
    else:
        print("  (Top3 거래 없음)")

    print("\n  ─ 라벨별 분리 ─")
    for lbl, s in (report.get("by_label_top3") or {}).items():
        if s.get("n", 0) >= 3:
            print(f"    {lbl:<12} N={s['n']:>3}  TP1={s['tp1_rate']*100:>5.1f}%  EV={s['ev']:>+6.2f}%")

    if report.get("tuning_best"):
        print("\n" + "═" * 70)
        print("  🔧 최강 임계값 튜닝 Top 5")
        print("═" * 70)
        for i, r in enumerate(report["tuning_results"][:5], 1):
            th = r["thresholds"]
            ts = r["top3_summary"]
            ss = r.get("strong_only", {})
            n_str = ss.get("n", 0)
            print(f"  #{i}  평균≥{th['strong_mean']:.0f} 밸≥{th['strong_bal']:.0f} "
                  f"갭≤{th['strong_gap']:.0f}% RR≥{th['strong_rr']:.1f}  →  "
                  f"Top3전체 N={ts['n']:>3} TP1={ts['tp1_rate']*100:>5.1f}% EV={ts['ev']:>+6.2f}%  "
                  f"(🏆만 N={n_str})")

    # Walk-forward 결과
    wf = report.get("walk_forward")
    if wf and "results" in wf:
        print("\n" + "═" * 70)
        print("  🚶 Walk-Forward 오버피팅 검증")
        print("═" * 70)
        print(f"  IS: {wf['is_period']} ({wf['is_days']}일)  →  "
              f"OOS: {wf['oos_period']} ({wf['oos_days']}일)")
        print(f"\n  {'IS순위':<7}{'조건':<32}{'IS EV':>9}{'OOS EV':>9}{'OOS N':>7}{'일반화':>8}")
        print("  " + "─" * 70)
        for r in wf["results"]:
            th = r["thresholds"]
            cond = f"평균≥{th['strong_mean']} 밸≥{th['strong_bal']} 갭≤{th['strong_gap']}% RR≥{th['strong_rr']:.1f}"
            is_ev = r["is_summary"].get("ev", 0)
            oos_ev = r["oos_summary"].get("ev", 0)
            oos_n = r["oos_summary"].get("n", 0)
            mark = "✅" if r["generalizes"] else "❌"
            print(f"  #{r['rank_in_is']:<6}{cond:<32}{is_ev:>+7.2f}%{oos_ev:>+7.2f}%"
                  f"{oos_n:>7}   {mark}")
        generalizes_any = any(r["generalizes"] for r in wf["results"])
        if generalizes_any:
            print(f"\n  ✅ IS Top 5 중 일부가 OOS에서도 EV+ — 일반화 증거 있음")
        else:
            print(f"\n  ⚠️ IS Top 5 중 OOS에서 EV+ 없음 — 오버피팅 가능성")

    # Rolling walk-forward 출력
    rwf = report.get("rolling_walk_forward")
    if rwf and rwf.get("folds"):
        print("\n" + "═" * 70)
        print("  🔁 Rolling Walk-Forward (여러 구간 반복 검증)")
        print("═" * 70)
        for f in rwf["folds"]:
            if "error" in f:
                print(f"  폴드 {f['fold']}: {f['error']}")
                continue
            mark = "✅" if f["generalizes"] else "❌"
            print(f"  폴드 {f['fold']}  IS {f['is_period']} → OOS {f['oos_period']}  "
                  f"IS EV {f['is_ev']:+.2f}% → OOS EV {f['oos_ev']:+.2f}% "
                  f"(N={f['oos_n']})  {mark}")
        s = rwf.get("summary", {})
        if s.get("n_valid"):
            robust_mark = "✅" if s.get("robust") else "⚠️"
            print(f"\n  {robust_mark} 종합: {s['n_generalizes']}/{s['n_valid']} 폴드 일반화 · "
                  f"평균 IS EV {s['avg_is_ev']:+.2f}% · 평균 OOS EV {s['avg_oos_ev']:+.2f}%")


if __name__ == "__main__":
    main()
