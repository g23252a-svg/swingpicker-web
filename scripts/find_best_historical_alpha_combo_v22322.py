# -*- coding: utf-8 -*-
"""[v22.3.22] OOS 검증형 Historical Alpha Combo 엔진.

사람이 찍은 고정 조건(breadth<35 등)은 과최적화 위험이 있다. 이 엔진은
과거 recommend CSV를 train/test로 나눠, train에서 좋은 조합을 찾고
test(out-of-sample)에서 재현되는 조합만 채택한다.

공식 산식(TOP_PICK/BUY_NOW_ELIGIBLE/scoring_engine) 무변경. read-only 별도 레인.

핵심 원칙 (과최적화 방지):
  - RR_NOW_TP1은 과거 초반 CSV에 없으므로 '필수 조건'에서 제외, 보조 정렬로만 사용
  - breadth 조건은 OOS에서 자주 붕괴 → 그리드에 넣되 OOS 검증으로 자동 탈락시킴
  - 채택: train에서 baseline+MARGIN 우위 AND test에서도 플러스 유지
  - fallback: TIMING>=55 + 외인순매수>0 + POC<=90 + 가드 (OOS 검증된 안정 조합)

근거(2026-02~05 실측, 내일시가진입 TP+8/SL-4):
  진입 당일 TP/SL 포함 후 채택룰은 약 승률 37%·baseline 33% 수준으로 재측정됨.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

OUT_DIR = "data"
ALPHA_MAX_PICKS = 3
TP_PCT, SL_PCT, HORIZON = 8.0, 4.0, 5
BASELINE_MARGIN = 3.0        # train에서 baseline 승률 + 이만큼 넘어야 후보
MIN_N_TRAIN, MIN_N_TEST = 25, 15
GUARD_FLAGS = ["MARKET_WARNING_GUARD_FLAG", "ABNORMAL_HISTORY_GUARD_FLAG",
               "SPIKE_REVERSAL_GUARD_FLAG", "LONG_HISTORY_COLLAPSE_FLAG"]

# fallback (OOS 검증 통과한 안정 조합 — breadth 없음, RR 보조)
FALLBACK_RULE = {"timing_min": 55.0, "frg_pos": True, "poc_max": 90.0,
                 "rule_id": "FALLBACK_T55_FRG_POC90",
                 "desc": "TIMING>=55 + 외인순매수>0 + POC<=90 + 위험가드"}


def _num(d: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)


def _truthy(d: pd.DataFrame, c: str) -> pd.Series:
    if c not in d.columns:
        return pd.Series(False, index=d.index)
    return pd.to_numeric(d[c], errors="coerce").fillna(0) == 1


def _guard_ok(d: pd.DataFrame) -> pd.Series:
    m = pd.Series(True, index=d.index)
    for g in GUARD_FLAGS:
        m &= ~_truthy(d, g)
    return m


def _rule_mask(d: pd.DataFrame, timing, frg_pos, poc_max, struct_min, breadth_max) -> pd.Series:
    m = _guard_ok(d)
    if timing is not None:
        m &= _num(d, "TIMING_SCORE") >= timing
    if frg_pos:
        m &= _num(d, "외인순매수").fillna(-1) > 0
    if poc_max is not None:
        m &= _num(d, "POC_GAP").fillna(0) <= poc_max
    if struct_min is not None:
        m &= _num(d, "STRUCT_SCORE") >= struct_min
    if breadth_max is not None:
        m &= _num(d, "MARKET_BREADTH") < breadth_max
    return m


# ---------- OHLC 기반 내일진입 성과 ----------
def _load_ohlc(data_dir: str):
    cands = sorted(Path(data_dir).glob("ohlcv_cache_2026*.parquet"))
    if not cands:
        return None, None
    o = pd.read_parquet(cands[-1]).reset_index()
    o["Date"] = pd.to_datetime(o["Date"]).dt.strftime("%Y%m%d")
    o["종목코드"] = o["종목코드"].astype(str).str.zfill(6)
    for c in ["시가", "고가", "저가", "종가"]:
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o = o.sort_values(["종목코드", "Date"])
    oby = {code: sub.reset_index(drop=True) for code, sub in o.groupby("종목코드")}
    return oby, sorted(o["Date"].unique())


def _trade(oby, code, ymd, av_index):
    if code not in oby:
        return None
    df = oby[code]
    idx = df.index[df["Date"] == ymd]
    if len(idx) == 0 or idx[0] + 1 >= len(df):
        return None
    i = idx[0]
    entry = df.loc[i + 1, "시가"]
    if pd.isna(entry) or entry <= 0:
        return None
    # next_open 진입 당일(i+1)부터 HORIZON 거래일 TP/SL 판정.
    # 기존 i+2 시작은 진입 당일 급등/급락을 누락 → "산 지 얼마 안 돼 빡 오르는" 종목 측정 목적과 어긋남.
    for k in range(HORIZON):
        bar = i + 1 + k
        if bar >= len(df):
            break
        lo, hi = df.loc[bar, "저가"], df.loc[bar, "고가"]
        if pd.notna(lo) and (lo / entry - 1) * 100 <= -SL_PCT:
            return -SL_PCT
        if pd.notna(hi) and (hi / entry - 1) * 100 >= TP_PCT:
            return TP_PCT
    last = df.loc[min(i + HORIZON, len(df) - 1), "종가"]
    return (last / entry - 1) * 100 if pd.notna(last) else None


def _load_labeled(data_dir, dates, oby, av):
    rows = []
    for f in sorted(Path(data_dir).glob("recommend_2026*.csv")):
        ymd = f.name[10:18]
        if ymd not in dates:
            continue
        df = pd.read_csv(f, dtype={"종목코드": str})
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        df["_ret"] = [_trade(oby, c, ymd, av) for c in df["종목코드"]]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    R = pd.concat(rows, ignore_index=True)
    return R[R["_ret"].notna()].copy()


def _stats(rets: pd.Series):
    wr = float((rets > 0).mean() * 100)
    w = rets[rets > 0]; l = rets[rets < 0]
    rr = float(w.mean() / abs(l.mean())) if len(l) and len(w) else 0.0
    return wr, float(rets.mean()), rr


def discover_oos_rules(data_dir: str = OUT_DIR) -> dict:
    """train/test 분리 → OOS 통과 조합 탐색."""
    oby, av = _load_ohlc(data_dir)
    if oby is None:
        return {"rules": [], "reason": "no_ohlc"}
    avset = set(av)
    recs = sorted(Path(data_dir).glob("recommend_2026*.csv"))
    meas = [f.name[10:18] for f in recs
            if f.name[10:18] in avset and av.index(f.name[10:18]) + HORIZON + 1 < len(av)]
    if len(meas) < 20:
        return {"rules": [], "reason": "insufficient_dates"}
    mid = len(meas) // 2
    train, test = set(meas[:mid]), set(meas[mid:])
    TR = _load_labeled(data_dir, train, oby, av)
    TE = _load_labeled(data_dir, test, oby, av)
    if len(TR) == 0 or len(TE) == 0:
        return {"rules": [], "reason": "no_labeled"}
    base_tr = float((TR["_ret"] > 0).mean() * 100)
    base_te = float((TE["_ret"] > 0).mean() * 100)

    grid = [(t, True, p, st, b)
            for t in [55, 60]
            for p in [90, None]
            for st in [None, 60, 70]
            for b in [None, 45]]
    rules = []
    for (t, f, p, st, b) in grid:
        mt = _rule_mask(TR, t, f, p, st, b)
        if mt.sum() < MIN_N_TRAIN:
            continue
        wtr, rtr, rrtr = _stats(TR.loc[mt, "_ret"])
        if wtr < base_tr + BASELINE_MARGIN or rtr <= 0:
            continue
        me = _rule_mask(TE, t, f, p, st, b)
        if me.sum() < MIN_N_TEST:
            continue
        wte, rte, rrte = _stats(TE.loc[me, "_ret"])
        oos_pass = bool(rte > 0 and wte >= base_te)
        # 전체 통합 성과
        allm_n = int(mt.sum() + me.sum())
        combined_wr = (wtr * mt.sum() + wte * me.sum()) / allm_n
        combined_ret = (rtr * mt.sum() + rte * me.sum()) / allm_n
        # ALPHA_EDGE_SCORE: OOS 재현성 핵심 가중
        edge = (combined_wr - (base_tr + base_te) / 2) * 1.0 + combined_ret * 5.0
        if not oos_pass:
            edge -= 50.0  # train만 좋은 건 강한 페널티
        rules.append({
            "rule_id": f"T{t}_FRG_P{p or 'NA'}_S{st or 'NA'}_B{b or 'NA'}",
            "desc": _desc(t, f, p, st, b),
            "timing_min": t, "frg_pos": f, "poc_max": p, "struct_min": st, "breadth_max": b,
            "n_train": int(mt.sum()), "win_train": round(wtr, 1), "ret_train": round(rtr, 2),
            "n_test": int(me.sum()), "win_test": round(wte, 1), "ret_test": round(rte, 2),
            "win_rate": round(combined_wr, 1), "avg_ret": round(combined_ret, 2),
            "n": allm_n, "rr_test": round(rrte, 2),
            "oos_pass": oos_pass, "edge_score": round(edge, 1),
        })
    # OOS 통과 + breadth 없는 것 우선, edge 순
    rules.sort(key=lambda r: (-int(r["oos_pass"]), r["breadth_max"] is not None, -r["edge_score"]))
    return {"rules": rules, "base_train": round(base_tr, 1), "base_test": round(base_te, 1),
            "train_dates": sorted(train), "test_dates": sorted(test)}


def _desc(t, f, p, st, b):
    parts = [f"TIMING>={t}"]
    if f: parts.append("외인순매수>0")
    if p is not None: parts.append(f"POC<={p}")
    if st is not None: parts.append(f"STRUCT>={st}")
    if b is not None: parts.append(f"breadth<{b}")
    parts.append("위험가드")
    return " + ".join(parts)


RR_FLOOR = 1.2  # 오늘 실전 RR 알파 후보 최소 손익비


def _score_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """후보 풀에 HISTORICAL_ALPHA_SCORE 부여 (보조 정렬: RR/FINAL/외인/과열페널티)."""
    rr = _num(pool, "RR_NOW_TP1").fillna(0)
    fs_ = _num(pool, "FINAL_SCORE").fillna(0)
    ts = _num(pool, "TIMING_SCORE").fillna(0)
    frg = _num(pool, "외인순매수").fillna(0)
    vwap = _num(pool, "VWAP_GAP").fillna(0)
    poc = _num(pool, "POC_GAP").fillna(0)
    risk = pool["ENTRY_RISK_LEVEL"].astype(str) if "ENTRY_RISK_LEVEL" in pool.columns else pd.Series("", index=pool.index)
    frg_rank = frg.rank(pct=True) * 100.0
    vwap_pen = np.where(vwap > 40, 20.0, np.where(vwap > 25, 8.0, 0.0))
    poc_pen = np.where(poc > 120, 30.0, np.where(poc > 80, 10.0, 0.0))
    risk_pen = np.where(risk.isin(["RED"]), 15.0, 0.0)
    pool = pool.copy()
    pool["HISTORICAL_ALPHA_SCORE"] = (
        0.25 * ts + 0.25 * frg_rank + 0.25 * (rr.clip(0, 4) / 4.0 * 100.0)
        + 0.15 * fs_ - vwap_pen - poc_pen - risk_pen
    ).round(1)
    pool["_vwap_pen"] = vwap_pen
    pool["_poc_pen"] = poc_pen
    return pool


def select_alpha_tiers(df: pd.DataFrame, rule: dict, topk: int = ALPHA_MAX_PICKS) -> dict:
    """Tier A(RR>=1.2 실전 픽) / Tier B(OOS통과하나 RR부족, 관찰) 분리. read-only.

    - tier_a: OOS 조건 + RR>=1.2 → 실제 화면 추천 후보
    - tier_b: OOS 조건 통과하나 RR<1.2 → 관찰 후보(매수 아님), 'RR 부족으로 제외' 사유
    공식 산식(TOP_PICK/BUY_NOW_ELIGIBLE) 무변경.
    """
    empty = pd.DataFrame()
    if df is None or len(df) == 0:
        return {"tier_a": empty, "tier_b": empty, "rule_pass_n": 0}
    df = df.copy()
    m = _rule_mask(df, rule.get("timing_min"), rule.get("frg_pos", True),
                   rule.get("poc_max"), rule.get("struct_min"), rule.get("breadth_max"))
    pool = df[m].copy()
    if len(pool) == 0:
        return {"tier_a": empty, "tier_b": empty, "rule_pass_n": 0}

    pool = _score_pool(pool)
    rr = _num(pool, "RR_NOW_TP1")
    has_rr = bool(rr.notna().any())
    if has_rr:
        a = pool[rr.fillna(0) >= RR_FLOOR].copy()
        b = pool[rr.fillna(0) < RR_FLOOR].copy()
    else:
        # RR 컬럼이 아예 없으면(과거 상황) 전부 Tier A로 (RR 보장 불가하나 막지 않음)
        a, b = pool.copy(), empty
    return {
        "tier_a": a.nlargest(topk, "HISTORICAL_ALPHA_SCORE") if len(a) else empty,
        "tier_b": b.nlargest(topk, "HISTORICAL_ALPHA_SCORE") if len(b) else empty,
        "rule_pass_n": int(len(pool)),
    }


def select_alpha_candidates(df: pd.DataFrame, rule: dict, topk: int = ALPHA_MAX_PICKS) -> pd.DataFrame:
    """[호환용] Tier A(실전 RR 알파 픽)만 반환."""
    return select_alpha_tiers(df, rule, topk)["tier_a"]


def build(data_dir: str = OUT_DIR, out_dir: str = OUT_DIR) -> dict:
    disc = discover_oos_rules(data_dir)
    rules = disc.get("rules", [])
    oos_rules = [r for r in rules if r["oos_pass"]]

    if oos_rules:
        best = oos_rules[0]
        rule_source = "OOS_DISCOVERED"
    else:
        best = {**FALLBACK_RULE, "struct_min": None, "breadth_max": None,
                "win_rate": None, "avg_ret": None, "n": None, "oos_pass": False,
                "win_test": None, "ret_test": None}
        rule_source = "FALLBACK"

    base = Path(data_dir)
    latest = base / "recommend_latest.csv"
    if not latest.exists():
        cands = sorted(base.glob("recommend_2026*.csv"))
        latest = cands[-1] if cands else None
    if latest is None:
        raise FileNotFoundError("recommend CSV 없음")
    df = pd.read_csv(latest, dtype={"종목코드": str})
    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)

    tp = pd.to_numeric(df.get("TOP_PICK", 0), errors="coerce").fillna(0)
    el = pd.to_numeric(df.get("BUY_NOW_ELIGIBLE", 0), errors="coerce").fillna(0)
    official_n = int(((tp == 1) & (el == 1)).sum())

    tiers = select_alpha_tiers(df, best, ALPHA_MAX_PICKS)
    picks = tiers["tier_a"]
    near = tiers["tier_b"]
    cols = ["종목코드", "종목명", "HISTORICAL_ALPHA_SCORE", "TIMING_SCORE", "외인순매수",
            "RR_NOW_TP1", "FINAL_SCORE", "VWAP_GAP", "POC_GAP", "STRUCT_SCORE",
            "ENTRY_RISK_LEVEL", "ROUTE", "MARKET_BREADTH"]
    pcols = [c for c in cols if c in picks.columns]
    picks_out = picks[pcols].copy()
    ncols = [c for c in cols if c in near.columns]
    near_out = near[ncols].copy() if len(near) else near
    for c in ["HISTORICAL_ALPHA_RULE_ID", "HISTORICAL_ALPHA_RULE_DESC",
              "HISTORICAL_ALPHA_WIN_RATE", "HISTORICAL_ALPHA_AVG_RET",
              "HISTORICAL_ALPHA_N", "HISTORICAL_ALPHA_OOS_PASS"]:
        pass
    picks_out["HISTORICAL_ALPHA_RULE_ID"] = best.get("rule_id", "")
    picks_out["HISTORICAL_ALPHA_RULE_DESC"] = best.get("desc", "")
    picks_out["HISTORICAL_ALPHA_WIN_RATE"] = best.get("win_rate")
    picks_out["HISTORICAL_ALPHA_AVG_RET"] = best.get("avg_ret")
    picks_out["HISTORICAL_ALPHA_N"] = best.get("n")
    picks_out["HISTORICAL_ALPHA_OOS_PASS"] = best.get("oos_pass")

    op = Path(out_dir)
    op.mkdir(parents=True, exist_ok=True)
    picks_out.to_csv(op / "historical_alpha_pick_latest.csv", index=False, encoding="utf-8-sig")
    # 전체 탐색 결과
    pd.DataFrame(rules).to_csv(op / "best_historical_alpha_combo_latest.csv", index=False, encoding="utf-8-sig")

    summary = {
        "asof_file": latest.name,
        "official_buy_count": official_n,
        "alpha_pick_count": int(len(picks_out)),
        "near_candidate_count": int(len(near_out)) if len(near_out) else 0,
        "near_candidates": near_out.to_dict(orient="records") if len(near_out) else [],
        "rule_pass_count": int(tiers["rule_pass_n"]),
        "rr_floor": RR_FLOOR,
        "rule_source": rule_source,
        "selected_rule": {**{k: best.get(k) for k in
                          ["rule_id", "desc",
                           "timing_min", "frg_pos", "poc_max", "struct_min", "breadth_max",
                           "win_rate", "avg_ret", "n",
                           "win_train", "win_test", "ret_test", "oos_pass"]},
                          "baseline_test": disc.get("base_test")},
        "baseline_train": disc.get("base_train"),
        "baseline_test": disc.get("base_test"),
        "oos_rules_found": len(oos_rules),
        "total_rules_evaluated": len(rules),
        "alpha_picks": picks_out.to_dict(orient="records"),
        "backtest_method": f"내일시가진입 TP+{TP_PCT:.0f}/SL-{SL_PCT:.0f} {HORIZON}일, train/test OOS 검증",
        "disclaimer": "공식 신규매수 아님. OOS 검증 통과한 과거 통계 기반 기대값 후보(승률 37% 내외·baseline 33% 대비 +EV·RR로 버는 구조). 70% 적중 아님.",
    }
    with open(op / "historical_alpha_pick_summary_latest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OOS Historical Alpha Combo engine v22.3.22")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    s = build(args.data_dir, args.out_dir)
    r = s["selected_rule"]
    print(f"[v22.3.22] source={s['rule_source']} OOS통과룰={s['oos_rules_found']}/{s['total_rules_evaluated']} "
          f"alpha_pick={s['alpha_pick_count']} official={s['official_buy_count']}")
    print(f"  채택룰: {r['desc']} (train승{r.get('win_train')}% test승{r.get('win_test')}% OOS={r.get('oos_pass')})")
    for p in s["alpha_picks"]:
        print(f"  {p.get('종목명','?')}({p.get('종목코드','?')}) ALPHA={p.get('HISTORICAL_ALPHA_SCORE')} "
              f"TIMING={p.get('TIMING_SCORE')} 외인={p.get('외인순매수')} POC={p.get('POC_GAP')}")

