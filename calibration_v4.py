# -*- coding: utf-8 -*-
"""
calibration_v4.py — v4.0 Phase 1: 세그먼트 조건부 캘리브레이션 (캘리브레이션 천장 제거)

문제 (v3.9.28):
  EST_WIN_RATE가 ELITE_SCORE 단일축 글로벌 룩업이라 ~0.51에 캡.
  scoring_engine STABLE 게이트(EST_WIN_RATE>=0.55)가 구조적으로 영구 미달 → STABLE 사망
  → 약세장에서 TOP_PICK이 AGGRESSIVE 단독 의존 → 공식추천 0개.

해결 (v4.0 Phase 1):
  1) 다축 세그먼트 캘리브레이션: (ELITE 버킷 × segment_cols) 조합별 승률
  2) 경험적 베이즈 수축: 표본 적은 셀은 글로벌 prior로 수렴 (과적합 차단)
  3) 상대 STABLE 게이트: 절대 0.55 대신 "당일 calibrated 상위 분위 + prior 마진"

설계 원칙:
  - SHADOW ONLY. 본 모듈은 scoring_engine의 TOP_PICK / EST_WIN_RATE를 변경하지 않는다.
    add_v4_shadow_columns()는 *_V4 / *_V4_SHADOW 컬럼만 추가한다.
  - 기존 kelly_calibrator의 시간감쇠(_time_weight, half_life 90d)를 재사용한다.
  - 본선 승격은 별도 PR + 백테스트(combo_optimizer) 통과 후.

pytest tests/test_calibration_v4.py -v
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("calibration_v4")

# ── 기존 인프라 재사용 (없으면 standalone fallback) ──────────────────────────
try:
    from kelly_calibrator import _time_weight as _kc_time_weight  # type: ignore
except Exception:  # pragma: no cover - 테스트/단독 실행 대비
    _kc_time_weight = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. 기본 파라미터 (v23 GDD 계승)
# ─────────────────────────────────────────────────────────────────────────────
ELITE_BUCKETS: List[Tuple[float, float]] = [
    (0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101),
]
DEFAULT_SEGMENT_COLS: List[str] = ["ACTION_TIER", "MACRO_REGIME_MODE"]
PRIOR_STRENGTH_K: float = 30.0      # Beta 사전강도 (셀 표본이 이만큼 쌓여야 prior 절반 극복)
GLOBAL_PRIOR_FALLBACK: float = 0.50  # 데이터 없을 때 prior
MIN_EFFECTIVE_N: float = 8.0        # 셀 "충분" 판정 (n_effective)
HALF_LIFE_DAYS: int = 90

# 상대 STABLE 게이트
STABLE_QUANTILE: float = 0.75       # 당일 calibrated WR 상위 25%
STABLE_PRIOR_MARGIN: float = 0.03   # AND prior + 3%p
STABLE_MIN_N: float = 20.0          # AND 셀 표본 하한


# ─────────────────────────────────────────────────────────────────────────────
# 2. 시간 감쇠 (kelly_calibrator 재사용 / fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _time_weight(rec_dates: pd.Series, half_life_days: int = HALF_LIFE_DAYS,
                 asof_date: Optional[str] = None) -> pd.Series:
    if _kc_time_weight is not None:
        try:
            return _kc_time_weight(rec_dates, half_life_days=half_life_days, asof_date=asof_date)
        except TypeError:
            return _kc_time_weight(rec_dates, half_life_days)
    # standalone fallback (테스트용): 지수 감쇠
    dt = pd.to_datetime(rec_dates, errors="coerce")
    asof = pd.to_datetime(asof_date) if asof_date else dt.max()
    age = (asof - dt).dt.days.fillna(half_life_days * 3).clip(lower=0)
    lam = np.log(2) / half_life_days
    return np.exp(-lam * age)


def _excess_win(
    df: pd.DataFrame, ret_col: str, date_col: str, horizon_col: str,
    benchmark_returns=None,
) -> Tuple[pd.Series, str]:
    """초과(벤치 대비) 승리 지표 {0,1} 계산.

    raw win( ret>0 )은 상승장에서 ~0.65로 부풀려져 '엣지'가 아니다.
    여기서는 같은 (날짜[,horizon]) 시장 대비 초과 여부를 본다.

    benchmark_returns:
      - None(기본): 당일 횡단면 평균수익을 시장 프록시로 사용(day-relative).
        같은 날 평균보다 나았으면 excess-win. 시장 베타 인플레이션 제거.
      - callable(date, horizon)->float 또는 dict[(date,horizon)]->float:
        진짜 지수(코스피/코스닥) 전방수익을 주면 지수 대비 초과로 계산.
    """
    ret = pd.to_numeric(df[ret_col], errors="coerce")
    if benchmark_returns is not None:
        def _bench_of(r):
            key_d = str(r.get(date_col, ""))
            key_h = r.get(horizon_col, None)
            if callable(benchmark_returns):
                return benchmark_returns(key_d, key_h)
            return (benchmark_returns.get((key_d, key_h))
                    if isinstance(benchmark_returns, dict) else None)
        bench = df.apply(_bench_of, axis=1)
        bench = pd.to_numeric(bench, errors="coerce")
        src = "external_benchmark"
    else:
        grp = [date_col] + ([horizon_col] if horizon_col in df.columns else [])
        bench = df.groupby(grp)[ret_col].transform("mean")
        src = "day_relative_mean"
    excess = ret - bench
    return (excess > 0).astype(float), src


def _elite_bucket(score: float) -> str:
    for lo, hi in ELITE_BUCKETS:
        if lo <= score < hi:
            return f"{int(lo)}-{int(hi)}"
    return "0-50"


def _norm_segment_value(v) -> str:
    s = str(v).strip().upper()
    return s if s and s not in {"NAN", "NONE", ""} else "ALL"


# ─────────────────────────────────────────────────────────────────────────────
# 3. 세그먼트 테이블 빌더 (순수 함수 — 테스트 가능)
# ─────────────────────────────────────────────────────────────────────────────
def build_segmented_table(
    trades: pd.DataFrame,
    *,
    score_col: str = "ELITE_SCORE",
    win_col: str = "is_win",
    date_col: str = "rec_date",
    segment_cols: Optional[Sequence[str]] = None,
    prior_k: float = PRIOR_STRENGTH_K,
    min_effective_n: float = MIN_EFFECTIVE_N,
    half_life_days: int = HALF_LIFE_DAYS,
    asof_ymd: Optional[str] = None,
    win_basis: str = "absolute",
    ret_col: str = "ret_pct",
    horizon_col: str = "horizon",
    benchmark_returns=None,
    lookup_col: Optional[str] = None,
) -> Dict:
    """과거 체결 로그 → (ELITE 버킷 × segment_cols) 세그먼트 승률 테이블.

    경험적 베이즈 수축:  p_shrunk = (wins_eff + k·p0) / (n_eff + k)
      - p0 = 전체 시간감쇠 승률 (글로벌 prior)
      - 표본 적은 셀은 p0로 수렴(과적합 방지), 표본 많은 셀만 신호 발현.

    win_basis:
      - "absolute"(기본): win_col(ret>0)을 그대로 사용. 상승장에서 ~0.65로 부풀려짐.
      - "excess": ret_col − 시장(또는 당일 평균) 초과 여부로 재정의. 시장 베타 제거 →
        prior가 ~0.5로 정직해지고 종목 간 분리력↑. (구독자 화면 과대표기 방지)

    lookup_col: 추론(score_segment) 시 recommend row에서 버킷을 읽을 컬럼명.
        미지정 시 score_col. per-trade 로그는 'score'(method별)지만 recommend row는
        'DISPLAY_SCORE'를 쓰므로, 라이브 테이블은 lookup_col='DISPLAY_SCORE'로 둔다.

    trades 에 segment_cols 가 없으면 자동으로 'ALL'로 폴백한다(스키마 안전).
    """
    seg_cols = list(segment_cols) if segment_cols is not None else list(DEFAULT_SEGMENT_COLS)
    n_raw_total = int(len(trades)) if trades is not None else 0
    meta = {
        "version": "v4.0-phase1",
        "method": "SEGMENTED_EB",
        "win_basis": win_basis,
        "score_col": score_col,
        "lookup_col": lookup_col or score_col,
        "prior_k": prior_k,
        "min_effective_n": min_effective_n,
        "half_life_days": half_life_days,
        "segment_cols": seg_cols,
        "n_raw_total": n_raw_total,
    }

    _has_win = win_col in (trades.columns if trades is not None else [])
    _has_ret = ret_col in (trades.columns if trades is not None else [])
    _usable = (win_basis == "excess" and _has_ret) or (win_basis != "excess" and _has_win)
    if trades is None or len(trades) == 0 or score_col not in trades.columns or not _usable:
        meta["global_prior"] = GLOBAL_PRIOR_FALLBACK
        meta["is_sufficient"] = False
        return {"meta": meta, "table": []}

    df = trades.copy()
    df["_w"] = _time_weight(df.get(date_col, pd.Series(index=df.index)),
                            half_life_days=half_life_days, asof_date=asof_ymd)
    df["_w"] = pd.to_numeric(df["_w"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if win_basis == "excess":
        df["_win"], _bench_src = _excess_win(df, ret_col, date_col, horizon_col, benchmark_returns)
        meta["benchmark_source"] = _bench_src
    else:
        df["_win"] = pd.to_numeric(df[win_col], errors="coerce").fillna(0.0).clip(0, 1)
        meta["benchmark_source"] = "none_absolute"
    df["_bucket"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).map(_elite_bucket)

    # 글로벌 prior (시간감쇠)
    w_tot = float(df["_w"].sum())
    p0 = float((df["_w"] * df["_win"]).sum() / w_tot) if w_tot > 0 else GLOBAL_PRIOR_FALLBACK
    meta["global_prior"] = round(p0, 4)

    # 존재하는 segment_cols 만 사용
    use_cols = [c for c in seg_cols if c in df.columns]
    meta["segment_cols_used"] = use_cols
    for c in use_cols:
        df[c] = df[c].map(_norm_segment_value)

    group_keys = ["_bucket"] + use_cols
    table: List[Dict] = []
    for keys, g in df.groupby(group_keys, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n_eff = float(g["_w"].sum())
        wins_eff = float((g["_w"] * g["_win"]).sum())
        p_raw = wins_eff / n_eff if n_eff > 0 else p0
        p_shrunk = (wins_eff + prior_k * p0) / (n_eff + prior_k)
        seg = {"_bucket": keys[0]}
        for i, c in enumerate(use_cols):
            seg[c] = keys[i + 1]
        table.append({
            "segment_key": "|".join(str(k) for k in keys),
            "bucket": keys[0],
            "segment": {c: seg[c] for c in use_cols},
            "p_win": round(p_shrunk, 4),
            "p_win_raw": round(p_raw, 4),
            "n_raw": int(len(g)),
            "n_effective": round(n_eff, 1),
            "sufficient": bool(n_eff >= min_effective_n),
        })

    meta["is_sufficient"] = any(t["sufficient"] for t in table)
    meta["n_segments"] = len(table)
    return {"meta": meta, "table": table}


# ─────────────────────────────────────────────────────────────────────────────
# 4. 세그먼트 룩업
# ─────────────────────────────────────────────────────────────────────────────
def _build_lookup(table: Dict) -> Tuple[Dict[str, Dict], List[str], float]:
    meta = table.get("meta", {})
    p0 = float(meta.get("global_prior", GLOBAL_PRIOR_FALLBACK))
    seg_cols = list(meta.get("segment_cols_used", meta.get("segment_cols", [])))
    lut = {row["segment_key"]: row for row in table.get("table", [])}
    return lut, seg_cols, p0


def _lookup_col_from_table(table: Dict) -> str:
    meta = table.get("meta", {})
    return str(meta.get("lookup_col") or meta.get("score_col") or "ELITE_SCORE")


def score_segment(row: pd.Series, table: Dict) -> Tuple[float, float, str, bool]:
    """(p_win_v4, n_effective, segment_key, sufficient) 반환. 셀 없으면 prior로 폴백.

    버킷 축은 테이블이 만들어진 기준(meta.lookup_col → score_col)을 따른다.
    recommend row에 그 컬럼이 없으면 DISPLAY_SCORE → ELITE_SCORE 순으로 폴백한다.
    (과거 버그: 무조건 ELITE_SCORE로 버킷팅 → 테이블 생성 기준과 어긋남)
    """
    lut, seg_cols, p0 = _build_lookup(table)
    score_col = _lookup_col_from_table(table)
    score_value = row.get(score_col, None)
    if score_value is None or (isinstance(score_value, float) and pd.isna(score_value)):
        score_value = row.get("DISPLAY_SCORE", row.get("ELITE_SCORE", 0))
    bucket = _elite_bucket(
        pd.to_numeric(pd.Series([score_value]), errors="coerce").fillna(0.0).iloc[0]
    )
    parts = [bucket] + [_norm_segment_value(row.get(c, "ALL")) for c in seg_cols]
    key = "|".join(parts)
    hit = lut.get(key)
    if hit is not None:
        return float(hit["p_win"]), float(hit["n_effective"]), key, bool(hit["sufficient"])
    # 셀 미존재 → prior 폴백 (수축의 극한값과 동일)
    return round(p0, 4), 0.0, key + "·prior", False


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHADOW 컬럼 부여 (본선 무변경)
# ─────────────────────────────────────────────────────────────────────────────
def _num(df: pd.DataFrame, names: Sequence[str], default: float = 0.0) -> pd.Series:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _mature_mask(df: pd.DataFrame) -> pd.Series:
    for c in ("CALIBRATION_MODE", "EST_WIN_RATE_MODE"):
        if c in df.columns:
            return df[c].astype(str).str.upper().eq("MATURE")
    return pd.Series(True, index=df.index)


def relative_stable_gate(
    p_win: pd.Series, n_eff: pd.Series, prior: float,
    *, quantile: float = STABLE_QUANTILE, prior_margin: float = STABLE_PRIOR_MARGIN,
    min_n: float = STABLE_MIN_N, mask: Optional[pd.Series] = None,
) -> pd.Series:
    """상대 STABLE 게이트 — 절대 0.55 대신 당일 분위 + prior 마진 + 표본 하한."""
    if mask is None:
        mask = pd.Series(True, index=p_win.index)
    pool = p_win[mask & (n_eff >= min_n)]
    if len(pool) >= 4:
        thr_q = float(np.quantile(pool, quantile))
    else:  # 표본 부족 → prior+margin 만으로 (분위 게이트 비활성)
        thr_q = prior + prior_margin
    thr = max(thr_q, prior + prior_margin)
    return (p_win >= thr) & (n_eff >= min_n) & mask


def add_v4_shadow_columns(df: pd.DataFrame, table: Dict) -> pd.DataFrame:
    """recommend DF에 v4 캘리브레이션 SHADOW 컬럼을 추가한다 (본선 컬럼 무변경).

    추가 컬럼:
      EST_WIN_RATE_V4, EST_WIN_RATE_V4_N, EST_WIN_RATE_V4_SEGMENT,
      EST_WIN_RATE_V4_SUFFICIENT, STABLE_GATE_V4_PASS,
      TOP_PICK_STABLE_V4_SHADOW, TOP_PICK_V4_SHADOW
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    _, _, p0 = _build_lookup(table)

    res = out.apply(lambda r: score_segment(r, table), axis=1, result_type="expand")
    out["EST_WIN_RATE_V4"] = res[0].astype(float).round(4)
    out["EST_WIN_RATE_V4_N"] = res[1].astype(float).round(1)
    out["EST_WIN_RATE_V4_SEGMENT"] = res[2].astype(str)
    out["EST_WIN_RATE_V4_SUFFICIENT"] = res[3].astype(int)

    # 상대 STABLE 게이트 (MATURE 풀 기준)
    mature = _mature_mask(out)
    gate = relative_stable_gate(out["EST_WIN_RATE_V4"], out["EST_WIN_RATE_V4_N"], p0, mask=mature)
    out["STABLE_GATE_V4_PASS"] = gate.astype(int)

    # STABLE 구조 전제 (scoring_engine STABLE 게이트와 동일, WR만 상대화)
    elite = _num(out, ["ELITE_SCORE", "DISPLAY_SCORE"])
    tp1 = _num(out, ["TP1_PCT", "TP1_pct"])
    balance = _num(out, ["BALANCE_SCORE"])
    stable_struct = (elite >= 70) & (tp1 >= 7) & (tp1 < 15) & (balance >= 70) & mature
    out["TOP_PICK_STABLE_V4_SHADOW"] = (stable_struct & gate).astype(int)

    # TOP_PICK_V4_SHADOW = 기존 TOP_PICK(AGGRESSIVE 포함) OR 새 STABLE_V4
    base_tp = (out["TOP_PICK"].astype(str).str.upper().isin({"1", "1.0", "TRUE"})
               if "TOP_PICK" in out.columns else pd.Series(False, index=out.index))
    out["TOP_PICK_V4_SHADOW"] = (base_tp | (out["TOP_PICK_STABLE_V4_SHADOW"] == 1)).astype(int)

    n_new = int(out["TOP_PICK_V4_SHADOW"].sum() - base_tp.sum())
    logger.info("[v4.0-p1] STABLE_V4 shadow가 추가로 살린 후보: %d개 (prior=%.3f)", n_new, p0)
    return out


__all__ = [
    "build_segmented_table",
    "score_segment",
    "relative_stable_gate",
    "add_v4_shadow_columns",
    "ELITE_BUCKETS",
    "DEFAULT_SEGMENT_COLS",
]
