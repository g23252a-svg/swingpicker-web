# -*- coding: utf-8 -*-
"""
combo_gate_v4.py — v4.0 공식픽 게이트 백테스트 (USE_CALIBRATION_V4 × TP1_BAND)

목적: v4 excess 캘리브레이션 + STABLE의 TP1 밴드 완화가 공식 추천(TOP_PICK)
      포트폴리오의 EV·승률(excess)·개수·MDD·보류현금효과를 어떻게 바꾸는지 측정.

설계: docs/DESIGN_v4_gate_backtest.md
원칙: read-only. scoring_engine / TOP_PICK 라이브 산식 무변경. combo_optimizer.py 무수정.
      AGGRESSIVE는 저장된 TOP_PICK_TYPE 재사용, STABLE 분기만 재시뮬.

리뷰 반영(필수): ① baseline 재현율 테스트  ② MDD 일별 포트폴리오 기준
                ③ baseline_sel reindex 안전  ④ look-ahead 방지(IS-only 테이블)
리뷰 반영(권장): TP1_BAND (6,18) 추가 · (3,30) exploratory · 검증케이스 종목코드 고정

pytest tests/test_v4_gate_backtest.py -v
python -m combo_gate_v4            # 67일×snapshot 실행 → data/v4_gate_backtest_latest.json
"""
from __future__ import annotations

import glob
import json
import logging
import os
from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from calibration_v4 import score_segment, relative_stable_gate, _build_lookup, build_segmented_table

logger = logging.getLogger("combo_gate_v4")

# 검증 케이스 — 종목명 대신 종목코드 고정 (실 CSV 확인: 2026-05-30)
VALIDATION_CASES = [
    {"name": "에스엔시스",   "code": "0008Z0", "date": "20260507", "note": "day-4 진입 케이스"},
    {"name": "신세계I&C",   "code": "035510", "date": "20260501", "note": "day-10 보유 방치 케이스"},
]

# TP1 밴드 격자 — 첫째가 현행 baseline. (3,30)은 exploratory only.
TP1_BANDS = [(7, 15), (6, 18), (5, 20), (5, 25), (3, 30)]
EXPLORATORY_BANDS = {(3, 30)}


# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로더 (STABLE 재현용 컬럼 포함)
# ─────────────────────────────────────────────────────────────────────────────
def _load_trade_rows_v4(data_dir: str = "data", horizon: int = 5) -> pd.DataFrame:
    """recommend_2026*.csv × horizon일 후 price_snapshot 매칭 → 거래결과 + STABLE 필드."""
    rec_files = sorted(glob.glob(os.path.join(data_dir, "recommend_2026*.csv")))
    snap_files = sorted(glob.glob(os.path.join(data_dir, "price_snapshot_2026*.csv")))
    snap_dates = [os.path.basename(f).replace("price_snapshot_", "").replace(".csv", "")
                  for f in snap_files]

    rows = []
    for rf in rec_files:
        rec_ymd = os.path.basename(rf).replace("recommend_", "").replace(".csv", "")
        if rec_ymd not in snap_dates:
            continue
        fidx = snap_dates.index(rec_ymd) + horizon
        if fidx >= len(snap_dates):
            continue
        try:
            rec = pd.read_csv(rf, dtype={"종목코드": str}, encoding="utf-8-sig")
            snap = pd.read_csv(os.path.join(data_dir, f"price_snapshot_{snap_dates[fidx]}.csv"),
                               dtype={"종목코드": str}, encoding="utf-8-sig")
        except Exception:
            continue
        rec["종목코드"] = rec["종목코드"].str.zfill(6)
        snap["종목코드"] = snap["종목코드"].str.zfill(6)
        fc_map = dict(zip(snap["종목코드"], pd.to_numeric(snap["종가"], errors="coerce")))

        for _, r in rec.iterrows():
            code = r["종목코드"]
            entry = float(pd.to_numeric(r.get("추천매수가", r.get("종가", 0)), errors="coerce") or 0)
            fc = fc_map.get(code, np.nan)
            if entry <= 0 or pd.isna(fc):
                continue
            rows.append({
                "code": code,
                "ret": (fc / entry - 1) * 100,
                "win": 1 if fc > entry else 0,
                "trade_date": rec_ymd,
                # STABLE 재현 필드
                "ELITE": float(r.get("ELITE_SCORE", 0) or 0),
                "BALANCE": float(r.get("BALANCE_SCORE", 0) or 0),
                "TP1_PCT": float(r.get("TP1_PCT", 0) or 0),
                "RR": float(r.get("RR_NOW_TP1", 0) or 0),
                "turnover": float(r.get("거래대금(억원)", 0) or 0),
                "entry_gap": abs(float(pd.to_numeric(r.get("ENTRY_GAP_PCT", r.get("GAP_PCT", 99)),
                                                     errors="coerce") or 99)),
                "MATURE": str(r.get("CALIBRATION_MODE", r.get("EST_WIN_RATE_MODE", ""))).upper() == "MATURE",
                "EST_WR": float(r.get("EST_WIN_RATE", 0) or 0),
                "TOP_PICK": 1 if str(r.get("TOP_PICK", "0")) in ("1", "1.0", "True") else 0,
                "TOP_PICK_TYPE": str(r.get("TOP_PICK_TYPE", "")),
                "DISPLAY_SCORE": float(r.get("DISPLAY_SCORE", 0) or 0),
            })
    df = pd.DataFrame(rows)
    df.attrs["matched_days"] = df["trade_date"].nunique() if len(df) else 0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. 공식픽 시뮬레이터 (AGGRESSIVE=저장값, STABLE=재유도)
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_official_pick(df: pd.DataFrame, use_v4: bool, tp1_band,
                            v4_table: Optional[Dict] = None) -> pd.Series:
    """주어진 설정의 '공식 TOP_PICK' 불리언 Series.

    재구성 불가/고정 픽(AGGRESSIVE + 구버전/No-Buy-Breaker 등 STABLE 외 경로로 채택된
    저장 TOP_PICK)은 그대로 carry-through 하고, **STABLE 분기만** 설정대로 재유도한다.
    → baseline↔v4 차이가 STABLE 분기에만 국한됨(효과 격리). baseline 재현율도 ~100%.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype=bool)
    lo, hi = tp1_band

    # carry: 저장 TOP_PICK 중 STABLE-type이 아닌 것 (AGGRESSIVE/legacy/breaker 고정)
    carry = (df["TOP_PICK"].astype(int) == 1) & (df["TOP_PICK_TYPE"].astype(str) != "STABLE")

    # STABLE 구조 전제 (TP1 밴드만 가변)
    stable_struct = (
        (df["ELITE"] >= 70) & (df["TP1_PCT"] >= lo) & (df["TP1_PCT"] < hi)
        & (df["BALANCE"] >= 70) & df["MATURE"].astype(bool)
        & (df["RR"] >= 1.0) & (df["turnover"] >= 50) & (df["entry_gap"] <= 5.0)
    )
    if use_v4 and v4_table is not None and v4_table.get("table"):
        res = df.apply(lambda r: score_segment(r, v4_table), axis=1, result_type="expand")
        p_v4 = res[0].astype(float)
        n_v4 = res[1].astype(float)
        _, _, p0 = _build_lookup(v4_table)
        wr_gate = relative_stable_gate(p_v4, n_v4, p0, mask=df["MATURE"].astype(bool))
    else:
        wr_gate = df["EST_WR"] >= 0.55

    return (carry | (stable_struct & wr_gate)).reindex(df.index).fillna(False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. baseline 재현율 (리뷰 필수 #0)
# ─────────────────────────────────────────────────────────────────────────────
def baseline_reproduction_rate(df: pd.DataFrame, v4_table: Optional[Dict] = None) -> Dict:
    """저장된 TOP_PICK==1 vs _simulate_official_pick(False,(7,15)) 일치율.
       95% 미만이면 이후 v4 비교 신뢰 불가."""
    if len(df) == 0:
        return {"agreement_pct": 0.0, "n": 0, "stored_top_pick": 0, "simulated": 0}
    sim = _simulate_official_pick(df, use_v4=False, tp1_band=(7, 15), v4_table=v4_table)
    stored = df["TOP_PICK"].astype(int) == 1
    agree = (sim.astype(int) == stored.astype(int)).mean() * 100
    return {
        "agreement_pct": round(float(agree), 1),
        "n": int(len(df)),
        "stored_top_pick": int(stored.sum()),
        "simulated": int(sim.sum()),
        "false_pos": int((sim & ~stored).sum()),
        "false_neg": int((~sim & stored).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. 콤보 평가 (리뷰 필수 ②③, 권장 ④ 반영)
# ─────────────────────────────────────────────────────────────────────────────
def _evaluate_gate_combo(df: pd.DataFrame, use_v4: bool, tp1_band, v4_table,
                         baseline_sel: pd.Series, min_samples: int = 5) -> Dict:
    baseline_sel = baseline_sel.reindex(df.index).fillna(False)            # ② index 안전
    sel = _simulate_official_pick(df, use_v4, tp1_band, v4_table)
    sub = df[sel]
    n = int(sel.sum())

    # ③ 명시적 정렬로 excess 승률
    day_mean = df.groupby("trade_date")["ret"].transform("mean")
    if n:
        day_mean_sub = day_mean.reindex(sub.index)
        wr_abs = (sub["ret"] > 0).mean() * 100
        wr_exc = (sub["ret"] > day_mean_sub).mean() * 100
        avg_ret = sub["ret"].mean()
        wins, losses = sub[sub["ret"] > 0]["ret"], sub[sub["ret"] <= 0]["ret"]
        aw = wins.mean() if len(wins) else 0.0
        al = abs(losses.mean()) if len(losses) else 0.0
        p = wr_abs / 100
        ev = p * aw - (1 - p) * al
        # ① MDD: 일별 포트폴리오(균등비중) 기준
        daily = sub.groupby("trade_date")["ret"].mean().sort_index()
        eq = daily.cumsum()
        mdd = float((eq.cummax() - eq).max()) if len(daily) else 0.0
        daily_pnl = float(daily.sum())   # 일별 평균수익 합 (자본정규화 근사)
    else:
        wr_abs = wr_exc = avg_ret = ev = mdd = aw = al = daily_pnl = 0.0

    # 보류 현금효과 (vs baseline)
    dropped = df[baseline_sel & ~sel]["ret"]
    added = df[~baseline_sel & sel]["ret"]
    avoided_loss = float(-dropped[dropped < 0].sum())   # +면 이득
    missed_gain = float(-dropped[dropped > 0].sum())    # -면 손해
    delta_pnl = float(added.sum() - dropped.sum())       # 단순 합계(개수민감 — n과 함께 해석)

    # baseline 일별 평균수익 대비 델타 (개수 영향 적은 지표)
    base_sub = df[baseline_sel]
    base_daily = base_sub.groupby("trade_date")["ret"].mean().sort_index() if len(base_sub) else pd.Series(dtype=float)
    delta_daily_pnl = round(daily_pnl - float(base_daily.sum()), 2)
    delta_avg_ret = round(avg_ret - (base_sub["ret"].mean() if len(base_sub) else 0.0), 2)

    return {
        "use_v4": use_v4, "tp1_band": list(tp1_band),
        "exploratory": tuple(tp1_band) in EXPLORATORY_BANDS,
        "n": n, "n_days": int(sub["trade_date"].nunique()) if n else 0,
        "win_rate": round(wr_abs, 1), "win_rate_excess": round(wr_exc, 1),
        "avg_ret": round(avg_ret, 2), "ev": round(ev, 2), "mdd": round(mdd, 2),
        "daily_pnl": round(daily_pnl, 2),
        "avoided_loss": round(avoided_loss, 2), "missed_gain": round(missed_gain, 2),
        "delta_pnl_vs_baseline": round(delta_pnl, 2),       # 단순합 (n_added/n_dropped와 함께 봄)
        "delta_avg_ret_vs_baseline": delta_avg_ret,
        "delta_daily_pnl_vs_baseline": delta_daily_pnl,     # 본선 판정 우선 지표
        "n_dropped": int((baseline_sel & ~sel).sum()),
        "n_added": int((~baseline_sel & sel).sum()),
    }


def _v4_gate_combos():
    return list(product([False, True], TP1_BANDS))


# ─────────────────────────────────────────────────────────────────────────────
# 5. look-ahead 방지 v4 테이블 빌드 (IS-only)
# ─────────────────────────────────────────────────────────────────────────────
def _filter_log_before(log: pd.DataFrame, asof_ymd: str) -> pd.DataFrame:
    """청산완료(exit < asof)된 trade만 — 미래참조 차단 (kelly_calibrator와 동일 원칙)."""
    if log is None or len(log) == 0 or "rec_date" not in log.columns:
        return log
    asof = pd.to_datetime(str(asof_ymd), format="%Y%m%d", errors="coerce")
    rec = pd.to_datetime(log["rec_date"].astype(str), format="%Y%m%d", errors="coerce")
    hor = pd.to_numeric(log.get("horizon", 5), errors="coerce").fillna(5).astype(int)
    valid = rec.notna()
    exit_np = np.busday_offset(rec[valid].values.astype("datetime64[D]"),
                               hor[valid].values, roll="forward")
    exit_dt = pd.Series(pd.to_datetime(exit_np), index=rec[valid].index)
    keep = exit_dt[exit_dt < asof].index
    return log.loc[keep].copy()


def _build_v4_table_asof(data_dir: str, asof_ymd: Optional[str]) -> Optional[Dict]:
    """라이브 정렬(method=DISPLAY_SCORE, horizon=5, excess) v4 테이블을 asof 이전 데이터로 빌드."""
    try:
        from kelly_calibrator import load_per_trade_log
        log = load_per_trade_log(data_dir)
    except Exception as e:
        logger.warning("per-trade 로그 로드 실패: %s", e)
        return None
    if log is None or len(log) == 0:
        return None
    if asof_ymd:
        log = _filter_log_before(log, asof_ymd)
    if "method" in log.columns:
        log = log[log["method"].astype(str) == "DISPLAY_SCORE"].copy()
    if "horizon" in log.columns:
        log = log[pd.to_numeric(log["horizon"], errors="coerce") == 5].copy()
    if len(log) == 0:
        return None
    return build_segmented_table(log, score_col="score", win_col="win", ret_col="ret_pct",
                                 segment_cols=[], win_basis="excess",
                                 lookup_col="DISPLAY_SCORE", asof_ymd=asof_ymd)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 검증 케이스 추적
# ─────────────────────────────────────────────────────────────────────────────
def _trace_validation_cases(df: pd.DataFrame, v4_table: Optional[Dict]) -> List[Dict]:
    out = []
    for c in VALIDATION_CASES:
        m = df["trade_date"].astype(str).eq(c["date"]) & df["code"].astype(str).str.zfill(6).eq(c["code"])
        sub = df[m]
        if len(sub) == 0:
            out.append({**c, "found": False})
            continue
        r = sub.iloc[[0]]
        base = bool(_simulate_official_pick(r, False, (7, 15), v4_table).iloc[0])
        v4_w = bool(_simulate_official_pick(r, True, (5, 20), v4_table).iloc[0])
        out.append({**c, "found": True, "ret": round(float(r["ret"].iloc[0]), 2),
                    "tp1_pct": float(r["TP1_PCT"].iloc[0]),
                    "stored_top_pick": int(r["TOP_PICK"].iloc[0]),
                    "baseline_picked": base, "v4_widened_picked": v4_w})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. 드라이버
# ─────────────────────────────────────────────────────────────────────────────
def run_v4_gate_backtest(data_dir: str = "data", horizon: int = 5,
                         oos_ratio: float = 0.3, save: bool = True) -> Dict:
    df = _load_trade_rows_v4(data_dir, horizon)
    if len(df) == 0:
        logger.warning("매칭 데이터 없음")
        return {}

    # 전체구간 테이블(in-sample 상한, look-ahead 있음) — 진단용
    full_table = _build_v4_table_asof(data_dir, asof_ymd=None)
    repro = baseline_reproduction_rate(df, full_table)

    baseline_sel = (df["TOP_PICK"].astype(int) == 1)   # 실제 과거 결정 = 비교 기준
    full_combos = [_evaluate_gate_combo(df, uv, band, full_table, baseline_sel)
                   for uv, band in _v4_gate_combos()]

    # IS/OOS look-ahead-safe 평가
    dates = sorted(df["trade_date"].unique())
    oos = {}
    if len(dates) >= 6:
        split = int(len(dates) * (1 - oos_ratio))
        split_date = dates[split]
        is_dates, oos_dates = set(dates[:split]), set(dates[split:])
        is_df = df[df["trade_date"].isin(is_dates)].copy()
        oos_df = df[df["trade_date"].isin(oos_dates)].copy()
        v4_is_table = _build_v4_table_asof(data_dir, asof_ymd=split_date)  # IS-only
        base_oos = (oos_df["TOP_PICK"].astype(int) == 1)
        oos = {
            "split_date": split_date, "n_is": len(is_df), "n_oos": len(oos_df),
            "v4_table_built_from": "IS-only (look-ahead safe)",
            "combos": [_evaluate_gate_combo(oos_df, uv, band, v4_is_table, base_oos)
                       for uv, band in _v4_gate_combos()],
        }

    cases = _trace_validation_cases(df, full_table)

    result = {
        "baseline_reproduction": repro,
        "full_sample": {"combos": full_combos, "lookahead_note": "전체구간 테이블 — in-sample 상한"},
        "oos": oos,
        "validation_cases": cases,
        "meta": {
            "horizon": horizon, "total_trades": len(df),
            "matched_days": df.attrs.get("matched_days", 0),
            "v4_table_meta": (full_table or {}).get("meta", {}),
            "tp1_bands": [list(b) for b in TP1_BANDS],
            "exploratory_bands": [list(b) for b in EXPLORATORY_BANDS],
            "promotion_rule": "baseline 대비 EV↑ AND excess승률↑ AND MDD 비악화 AND "
                              "delta_daily_pnl≥0 을 full+OOS 모두 만족 (exploratory 밴드 제외)",
        },
    }
    if save:
        with open(os.path.join(data_dir, "v4_gate_backtest_latest.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    return result


if __name__ == "__main__":
    import pprint
    pprint.pprint(run_v4_gate_backtest())
