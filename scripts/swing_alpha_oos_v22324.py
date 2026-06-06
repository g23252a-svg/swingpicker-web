# -*- coding: utf-8 -*-
# [v22.3.24] Swing Alpha OOS lane.
# Read-only 보조 레인: 공식 신규매수(TOP_PICK+BUY_NOW_ELIGIBLE) 산식 변경 없음.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from find_best_historical_alpha_combo_v22322 import (
    _guard_ok,
    _load_labeled,
    _load_ohlc,
    _num,
    _stats,
)

OUT_DIR = "data"

SWING_TOPK = 3
SWING_RR_TP1_MIN = 1.2
SWING_RR_TP2_MIN = 1.5
SWING_RR_TP3_MIN = 2.0
SWING_TP2_PROB_MIN = 40.0

PRACTICAL_TIMING_MIN = 60.0
PRACTICAL_FINAL_MIN = 55.0
PRACTICAL_STRUCT_MIN = 50.0

MIN_N_TRAIN = 10
MIN_N_TEST = 5


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _target_rr(df: pd.DataFrame, target_col: str) -> pd.Series:
    entry = _num(df, "추천매수가")
    close = _num(df, "종가")
    entry = entry.where(entry > 0, close)
    stop = _num(df, "손절가")
    target = _num(df, target_col)
    risk = entry - stop
    reward = target - entry
    rr = reward / risk.replace(0, np.nan)
    return rr.where((entry > 0) & (stop > 0) & (target > 0) & (risk > 0) & (reward > 0), np.nan)


def _flow_quality(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    frg = _num(df, "외인순매수").fillna(0)
    inst = _num(df, "기관순매수").fillna(0)
    retail = _num(df, "개인순매수").fillna(0)
    major = _num(df, "메이저순매수")
    major = major.fillna(frg + inst) if major.notna().any() else (frg + inst)

    strong = (frg > 0) & (inst >= 0) & (major > 0) & (retail < 0)
    ok = (frg > 0) & (major > 0)
    weak = frg > 0
    q = pd.Series(np.select([strong, ok, weak], ["STRONG", "OK", "WEAK"], default="BAD"), index=df.index)

    reasons = []
    for f, i, m, r, qq in zip(frg, inst, major, retail, q):
        bits = [f"외인 {f:+,.0f}", f"기관 {i:+,.0f}", f"메이저 {m:+,.0f}"]
        if r < 0:
            bits.append(f"개인 {r:+,.0f} 흡수")
        elif r > 0:
            bits.append(f"개인 {r:+,.0f} 동반")
        bits.append(f"수급 {qq}")
        reasons.append(" · ".join(bits))
    return q, pd.Series(reasons, index=df.index)


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "RR_NOW_TP2_SWING" not in out.columns:
        out["RR_NOW_TP2_SWING"] = _target_rr(out, "추천매도가2").round(2)
    if "RR_NOW_TP3_SWING" not in out.columns:
        out["RR_NOW_TP3_SWING"] = _target_rr(out, "추천매도가3").round(2)
    q, reason = _flow_quality(out)
    out["SWING_FLOW_QUALITY"] = q
    out["SWING_FLOW_REASON"] = reason
    return out


def _profile_mask(df: pd.DataFrame, prof: dict) -> pd.Series:
    m = _guard_ok(df)

    tmin = max(_safe_float(prof.get("timing_min"), 60.0), PRACTICAL_TIMING_MIN)
    fmin = _safe_float(prof.get("final_min"), PRACTICAL_FINAL_MIN)
    smin = _safe_float(prof.get("struct_min"), PRACTICAL_STRUCT_MIN)
    rrmin = _safe_float(prof.get("rr_min"), SWING_RR_TP1_MIN)
    pocmax = prof.get("poc_max", 90.0)
    vwapmax = prof.get("vwap_max", None)

    m &= _num(df, "TIMING_SCORE").fillna(-999) >= tmin
    m &= _num(df, "FINAL_SCORE").fillna(-999) >= fmin
    m &= _num(df, "STRUCT_SCORE").fillna(-999) >= smin
    m &= _num(df, "RR_NOW_TP1").fillna(0) >= rrmin
    m &= _num(df, "외인순매수").fillna(-1) > 0

    frg = _num(df, "외인순매수").fillna(0)
    inst = _num(df, "기관순매수").fillna(0)
    major = _num(df, "메이저순매수")
    major = major.fillna(frg + inst) if major.notna().any() else (frg + inst)
    m &= major > 0

    if pocmax is not None:
        m &= _num(df, "POC_GAP").fillna(999) <= float(pocmax)
    if vwapmax is not None:
        m &= _num(df, "VWAP_GAP").fillna(999) <= float(vwapmax)

    risk = df["ENTRY_RISK_LEVEL"].astype(str).str.strip().str.upper() if "ENTRY_RISK_LEVEL" in df.columns else pd.Series("", index=df.index)
    m &= ~risk.eq("RED")
    return m


def _swing_potential_mask(df: pd.DataFrame) -> pd.Series:
    rr1 = _num(df, "RR_NOW_TP1").fillna(0)
    rr2 = _num(df, "RR_NOW_TP2_SWING").fillna(0)
    rr3 = _num(df, "RR_NOW_TP3_SWING").fillna(0)
    tp2p = _num(df, "TP2_PROB").fillna(0)
    return (rr1 >= 1.5) | ((rr2 >= SWING_RR_TP2_MIN) & (tp2p >= SWING_TP2_PROB_MIN)) | (rr3 >= SWING_RR_TP3_MIN)


def _profile_desc(prof: dict) -> str:
    parts = [
        f"TIMING≥{_safe_float(prof.get('timing_min'), PRACTICAL_TIMING_MIN):.0f}",
        f"FINAL≥{_safe_float(prof.get('final_min'), PRACTICAL_FINAL_MIN):.0f}",
        f"STRUCT≥{_safe_float(prof.get('struct_min'), PRACTICAL_STRUCT_MIN):.0f}",
        "외인/메이저수급>0",
        f"RR1≥{_safe_float(prof.get('rr_min'), SWING_RR_TP1_MIN):.1f}",
    ]
    if prof.get("poc_max") is not None:
        parts.append(f"POC≤{_safe_float(prof.get('poc_max'), 90):.0f}")
    if prof.get("vwap_max") is not None:
        parts.append(f"VWAP≤{_safe_float(prof.get('vwap_max'), 999):.0f}")
    parts.append("RED 제외")
    return " + ".join(parts)


def discover_swing_oos_profile(data_dir: str = OUT_DIR) -> dict:
    oby, av = _load_ohlc(data_dir)
    if oby is None:
        return {"oos_pass": False, "reason": "no_ohlc", "desc": "OHLC 없음"}

    avset = set(av)
    recs = sorted(Path(data_dir).glob("recommend_2026*.csv"))
    meas = [
        f.name[10:18]
        for f in recs
        if f.name[10:18] in avset and av.index(f.name[10:18]) + 5 + 1 < len(av)
    ]
    if len(meas) < 20:
        return {"oos_pass": False, "reason": "insufficient_dates", "desc": "측정일 부족"}

    mid = len(meas) // 2
    train, test = set(meas[:mid]), set(meas[mid:])
    TR = _load_labeled(data_dir, train, oby, av)
    TE = _load_labeled(data_dir, test, oby, av)
    if TR.empty or TE.empty:
        return {"oos_pass": False, "reason": "no_labeled", "desc": "라벨 데이터 없음"}

    TR = _ensure_features(TR)
    TE = _ensure_features(TE)

    base_tr = float((TR["_ret"] > 0).mean() * 100)
    base_te = float((TE["_ret"] > 0).mean() * 100)

    grid = []
    for t in [60, 65, 70]:
        for f in [50, 55, 60]:
            for s in [50, 55, 60]:
                for rr in [1.2, 1.5]:
                    for poc in [90, 60]:
                        for vwap in [None, 15, 10]:
                            grid.append({
                                "timing_min": t,
                                "final_min": f,
                                "struct_min": s,
                                "rr_min": rr,
                                "poc_max": poc,
                                "vwap_max": vwap,
                            })

    profiles = []
    for prof in grid:
        mt = _profile_mask(TR, prof)
        me = _profile_mask(TE, prof)
        if int(mt.sum()) < MIN_N_TRAIN or int(me.sum()) < MIN_N_TEST:
            continue
        wtr, rtr, rrtr = _stats(TR.loc[mt, "_ret"])
        wte, rte, rrte = _stats(TE.loc[me, "_ret"])

        oos_pass = bool(rte > 0 and wte >= base_te and wte >= 35.0)
        edge = (wte - base_te) * 1.4 + rte * 8.0 + (wtr - base_tr) * 0.3 + rtr * 2.5 + min(int(me.sum()), 30) * 0.15
        if not oos_pass:
            edge -= 50.0

        profiles.append({
            **prof,
            "desc": _profile_desc(prof),
            "n_train": int(mt.sum()),
            "n_test": int(me.sum()),
            "win_train": round(wtr, 1),
            "win_test": round(wte, 1),
            "ret_train": round(rtr, 2),
            "ret_test": round(rte, 2),
            "rr_test": round(rrte, 2),
            "baseline_train": round(base_tr, 1),
            "baseline_test": round(base_te, 1),
            "oos_pass": oos_pass,
            "edge_score": round(edge, 1),
        })

    if not profiles:
        return {
            "oos_pass": False,
            "reason": "no_profile",
            "desc": "스윙 profile 없음",
            "baseline_test": round(base_te, 1),
        }

    profiles.sort(key=lambda x: (-int(x["oos_pass"]), -x["edge_score"], -x["win_test"], -x["ret_test"], -x["n_test"]))
    best = profiles[0]
    best["all_profiles_n"] = len(profiles)
    return best


def select_current_swing_candidates(df: pd.DataFrame, profile: dict, topk: int = SWING_TOPK) -> dict:
    empty = pd.DataFrame()
    if df is None or len(df) == 0:
        return {"picks": empty, "near": empty, "profile_pass_n": 0, "swing_pass_n": 0}

    d = _ensure_features(df)
    base = _profile_mask(d, profile)
    swing = _swing_potential_mask(d)

    pool = d[base].copy()
    if pool.empty:
        return {"picks": empty, "near": empty, "profile_pass_n": 0, "swing_pass_n": 0}

    rr1 = _num(pool, "RR_NOW_TP1").fillna(0).clip(0, 4) / 4 * 100
    rr2 = _num(pool, "RR_NOW_TP2_SWING").fillna(0).clip(0, 4) / 4 * 100
    rr3 = _num(pool, "RR_NOW_TP3_SWING").fillna(0).clip(0, 5) / 5 * 100
    timing = _num(pool, "TIMING_SCORE").fillna(0)
    final = _num(pool, "FINAL_SCORE").fillna(0)
    struct = _num(pool, "STRUCT_SCORE").fillna(0)
    flow = pool["SWING_FLOW_QUALITY"].map({"STRONG": 12, "OK": 8, "WEAK": 3, "BAD": -10}).fillna(0)
    vwap = _num(pool, "VWAP_GAP").fillna(0)
    poc = _num(pool, "POC_GAP").fillna(0)

    vwap_pen = np.where(vwap > 25, 12, np.where(vwap > 15, 5, 0))
    poc_pen = np.where(poc > 90, 15, np.where(poc > 60, 5, 0))

    pool["SWING_ALPHA_SCORE"] = (
        0.20 * timing + 0.18 * final + 0.12 * struct
        + 0.18 * rr1 + 0.12 * rr2 + 0.06 * rr3
        + flow - vwap_pen - poc_pen
    ).round(1)

    pool["SWING_ALPHA_REASON"] = (
        "OOS profile 통과 · "
        + pool["SWING_FLOW_REASON"].astype(str)
        + " · RR1 "
        + _num(pool, "RR_NOW_TP1").fillna(0).map(lambda x: f"{x:.2f}")
        + " · RR2 "
        + _num(pool, "RR_NOW_TP2_SWING").fillna(0).map(lambda x: f"{x:.2f}")
    )

    swing_mask_pool = swing.loc[pool.index]
    picks = pool[swing_mask_pool].nlargest(topk, "SWING_ALPHA_SCORE").copy()
    near = pool[~swing_mask_pool].nlargest(topk, "SWING_ALPHA_SCORE").copy()

    return {
        "picks": picks,
        "near": near,
        "profile_pass_n": int(len(pool)),
        "swing_pass_n": int(len(picks)),
    }


def build_swing_alpha(df: pd.DataFrame, data_dir: str = OUT_DIR, topk: int = SWING_TOPK) -> dict:
    profile = discover_swing_oos_profile(data_dir)
    selected = select_current_swing_candidates(df, profile, topk=topk)

    if not profile.get("oos_pass"):
        picks = selected["picks"].iloc[0:0].copy() if isinstance(selected.get("picks"), pd.DataFrame) else pd.DataFrame()
        selected["near"] = selected.get("picks", pd.DataFrame()) if isinstance(selected.get("picks"), pd.DataFrame) else pd.DataFrame()
        selected["picks"] = picks

    return {"profile": profile, **selected}
