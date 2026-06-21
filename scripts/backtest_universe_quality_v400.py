# -*- coding: utf-8 -*-
"""
backtest_universe_quality_v400.py — v4.0 Universe Cutoff 사전검증 (read-only)

목적: 추천 universe에 시가총액(보조: 거래대금) 하한을 그었을 때
      forward EV·승률·MDD가 얼마나 개선되고 coverage가 얼마나 줄어드는지 측정.
      → v4.0 V4_UNIVERSE_OK cutoff의 데이터 근거 확정용. 공식 산식 변경 없음.

[핵심 — 룩어헤드 제거]
  recommend CSV의 ret_5d_%/ret_10d_%/ret_20d_% 는 '추천 시점 과거 수익률'이다
  (검증 완료). 이를 기준으로 쓰면 백테스트가 무의미해진다.
  따라서 forward 수익률을 price_snapshot으로 직접 계산한다:
    진입가 = 추천 다음 영업일(T+1) 시가(익일) 또는 추천일 종가(보조)
    청산가 = 진입 후 horizon 영업일 뒤 종가

[측정] 시총 cutoff 후보별로:
    n, coverage%, win%, avg_ret%, EV%, EV_delta_vs_all%, MDD%
  horizon = 5d(1순위) / 10d / 20d (보조 확인)
  진입 = 익일 / 종가

[판정 가드] n>=min_n AND coverage>=min_coverage 인 cutoff만 신뢰.

사용:
    python backtest_universe_quality_v400.py --data-dir data --out-dir data
    python backtest_universe_quality_v400.py --horizons 5,10,20 --entry next_open
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

MCAP_COL = "시가총액(억원)"
TURNOVER_COL = "거래대금(억원)"

# 시총 cutoff 후보 (억원) — 1천억~2조
MCAP_CUTOFFS = [0, 1_000, 2_000, 3_000, 5_000, 7_000, 10_000, 15_000, 20_000]
# 거래대금 cutoff 후보 (억원) — 보조
TURNOVER_CUTOFFS = [0, 30, 50, 100, 200]


def _snap_dates(d):
    return [os.path.basename(f).replace("price_snapshot_", "").replace(".csv", "")
            for f in sorted(glob.glob(os.path.join(d, "price_snapshot_2026*.csv")))]


def _snap(d, ymd):
    p = os.path.join(d, f"price_snapshot_{ymd}.csv")
    if not os.path.exists(p):
        return {}
    s = pd.read_csv(p, dtype={"종목코드": str}, encoding="utf-8-sig")
    s.columns = [c.replace("\ufeff", "") for c in s.columns]
    s["종목코드"] = s["종목코드"].astype(str).str.zfill(6)
    return {c: (o, k) for c, o, k in zip(
        s["종목코드"], pd.to_numeric(s.get("시가"), errors="coerce"),
        pd.to_numeric(s.get("종가"), errors="coerce"))}


def build_panel(data_dir, horizon, entry_mode):
    """추천 × forward수익률 + 시총/거래대금 패널."""
    dates = _snap_dates(data_dir)
    d2i = {d: i for i, d in enumerate(dates)}
    out = []
    for rf in sorted(glob.glob(os.path.join(data_dir, "recommend_2026*.csv"))):
        ymd = os.path.basename(rf)[10:18]
        if ymd not in d2i:
            continue
        i = d2i[ymd]
        ei = i + 1 if entry_mode == "next_open" else i
        xi = ei + horizon
        if xi >= len(dates) or ei >= len(dates):
            continue
        es, xs = _snap(data_dir, dates[ei]), _snap(data_dir, dates[xi])
        if not es or not xs:
            continue
        try:
            rec = pd.read_csv(rf, dtype={"종목코드": str}, encoding="utf-8-sig")
        except Exception:
            continue
        rec["종목코드"] = rec["종목코드"].astype(str).str.zfill(6)
        has_mc = MCAP_COL in rec.columns
        has_to = TURNOVER_COL in rec.columns
        is_top = ("TOP_PICK" in rec.columns)
        for _, r in rec.iterrows():
            code = r["종목코드"]
            e, x = es.get(code), xs.get(code)
            if e is None or x is None:
                continue
            ep = e[0] if entry_mode == "next_open" else e[1]
            if not ep or pd.isna(ep) or ep <= 0 or pd.isna(x[1]):
                continue
            out.append({
                "date": ymd,
                "ret": (x[1] - ep) / ep * 100.0,
                "mcap": pd.to_numeric(r.get(MCAP_COL), errors="coerce") if has_mc else np.nan,
                "turnover": pd.to_numeric(r.get(TURNOVER_COL), errors="coerce") if has_to else np.nan,
                "top_pick": int(pd.to_numeric(r.get("TOP_PICK", 0), errors="coerce") or 0) if is_top else 0,
            })
    return pd.DataFrame(out)


def _metrics(sub, baseline_ev):
    if len(sub) == 0:
        return None
    r = sub["ret"]
    p = (r > 0).mean()
    wins, losses = r[r > 0], r[r <= 0]
    aw = wins.mean() if len(wins) else 0.0
    al = abs(losses.mean()) if len(losses) else 0.0
    ev = p * aw - (1 - p) * al
    # 일별 균등비중 포트폴리오 MDD
    daily = sub.groupby("date")["ret"].mean().sort_index()
    eq = daily.cumsum()
    mdd = float((eq.cummax() - eq).max()) if len(daily) else 0.0
    return dict(n=int(len(sub)), win=round(p * 100, 1),
                avg=round(r.mean(), 2), ev=round(ev, 2),
                ev_delta=round(ev - baseline_ev, 2), mdd=round(mdd, 2),
                ndays=int(sub["date"].nunique()))


def run(data_dir, out_dir, horizons, entries, pool, min_n, min_cov):
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []

    for entry_mode in entries:
        for horizon in horizons:
            panel = build_panel(data_dir, horizon, entry_mode)
            if len(panel) == 0:
                continue
            if pool == "top_pick":
                panel = panel[panel["top_pick"] == 1]
            total_n = len(panel)
            if total_n == 0:
                continue
            base = _metrics(panel, 0.0)
            base_ev = base["ev"]

            # 시총 cutoff
            for cut in MCAP_CUTOFFS:
                seg = panel[panel["mcap"] >= cut]
                m = _metrics(seg, base_ev)
                if m is None:
                    continue
                cov = round(m["n"] / total_n * 100, 1)
                ok = (m["n"] >= min_n) and (cov >= min_cov)
                all_rows.append({
                    "axis": "mcap", "cutoff": cut, "horizon": f"ret_{horizon}d_%",
                    "entry": entry_mode, "pool": pool, "coverage_pct": cov,
                    "reliable": ok, **m,
                })
            # 거래대금 cutoff (보조)
            for cut in TURNOVER_CUTOFFS:
                seg = panel[panel["turnover"] >= cut]
                m = _metrics(seg, base_ev)
                if m is None:
                    continue
                cov = round(m["n"] / total_n * 100, 1)
                ok = (m["n"] >= min_n) and (cov >= min_cov)
                all_rows.append({
                    "axis": "turnover", "cutoff": cut, "horizon": f"ret_{horizon}d_%",
                    "entry": entry_mode, "pool": pool, "coverage_pct": cov,
                    "reliable": ok, **m,
                })

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, "validation_v4_universe_cutoff_latest.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 콘솔 요약 — 5d/익일/시총 1순위
    print("█" * 92)
    print(f"  v4.0 Universe Cutoff 사전검증  ·  pool={pool}  ·  forward 수익률  ·  가드 n≥{min_n} cov≥{min_cov}%")
    print("█" * 92)
    for entry_mode in entries:
        prim = df[(df["horizon"] == "ret_5d_%") & (df["entry"] == entry_mode)
                  & (df["axis"] == "mcap")]
        if len(prim) == 0:
            continue
        label = "익일 진입" if entry_mode == "next_open" else "종가 진입"
        print(f"\n▶ 시총 cutoff · ret_5d · {label}")
        print(f"  {'cutoff(억)':>10s}{'n':>7s}{'cov%':>7s}{'승률':>8s}{'EV':>8s}"
              f"{'ΔEV':>8s}{'MDD':>8s}{'신뢰':>6s}")
        for _, r in prim.sort_values("cutoff").iterrows():
            tag = "✅" if r["reliable"] else "⚠️"
            print(f"  {int(r['cutoff']):>10,}{r['n']:>7,}{r['coverage_pct']:>6.1f}%"
                  f"{r['win']:>7.1f}%{r['ev']:>+7.2f}%{r['ev_delta']:>+7.2f}%"
                  f"{r['mdd']:>7.2f}%{tag:>6s}")

    # 권고 cutoff: EV 최대가 아니라 "첫 플러스 전환점"을 균형형 shadow 후보로 둔다.
    # 이유: EV 최대화만 하면 coverage를 과도하게 희생하는 2조 이상 컷이 자동 선택될 수 있다.
    rec = df[(df["horizon"] == "ret_5d_%") & (df["entry"] == "next_open")
             & (df["axis"] == "mcap") & (df["reliable"]) & (df["cutoff"] > 0)].copy()

    balanced = None
    aggressive = None
    selection_rule = None
    if len(rec):
        rec = rec.sort_values("cutoff")
        # 1순위: coverage 50% 이상을 유지하면서 EV가 처음으로 0 이상 전환되는 최소 cutoff
        cand = rec[(rec["ev"] >= 0) & (rec["coverage_pct"] >= 50.0)]
        if len(cand):
            b = cand.iloc[0]
            selection_rule = "first_positive_ev_with_coverage_ge_50"
        else:
            # 2순위: coverage 50%를 만족하지 못해도 EV가 처음으로 0 이상 전환되는 최소 cutoff
            cand = rec[rec["ev"] >= 0]
            if len(cand):
                b = cand.iloc[0]
                selection_rule = "first_positive_ev"
            else:
                # 3순위: 플러스 전환점이 없으면 ΔEV 최대값을 참고 후보로만 둔다.
                b = rec.sort_values(["ev_delta", "ev"], ascending=[False, False]).iloc[0]
                selection_rule = "max_ev_delta_fallback_no_positive_ev"

        balanced = {
            "cutoff_eok": int(b["cutoff"]),
            "ev": float(b["ev"]),
            "ev_delta_vs_no_cut": float(b["ev_delta"]),
            "win": float(b["win"]),
            "coverage_pct": float(b["coverage_pct"]),
            "n": int(b["n"]),
            "selection_rule": selection_rule,
        }

        a = rec.sort_values(["ev_delta", "ev"], ascending=[False, False]).iloc[0]
        aggressive = {
            "cutoff_eok": int(a["cutoff"]),
            "ev": float(a["ev"]),
            "ev_delta_vs_no_cut": float(a["ev_delta"]),
            "win": float(a["win"]),
            "coverage_pct": float(a["coverage_pct"]),
            "n": int(a["n"]),
            "selection_rule": "max_ev_delta_aggressive_reference",
        }

    summary = {
        "version": "v4_universe_cutoff_preflight",
        "mode": "read_only_preflight",
        "official_formula_unchanged": True,
        "do_not_auto_hard_block": True,
        "note": "forward 수익률(price_snapshot 직접계산). CSV ret_Nd_%는 과거값이라 미사용.",
        "pool": pool, "min_n": min_n, "min_coverage_pct": min_cov,
        "recommended_mcap_cutoff": balanced,
        "balanced_shadow_mcap_cutoff": balanced,
        "aggressive_reference_mcap_cutoff": aggressive,
        "guard_note": "reliable=True (n>=min_n & coverage>=min_cov) 인 행만 cutoff 근거로 사용. 공식 차단 전 2~4주 shadow 검증 필요.",
    }
    json_path = os.path.join(out_dir, "validation_v4_universe_cutoff_latest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "█" * 92)
    if balanced:
        print(f"  균형형 shadow cutoff: 시총 ≥{balanced['cutoff_eok']:,}억  "
              f"→ EV {balanced['ev']:+.2f}% (ΔEV {balanced['ev_delta_vs_no_cut']:+.2f}%p) "
              f"· 승률 {balanced['win']}% · coverage {balanced['coverage_pct']}% · n {balanced['n']:,} "
              f"· rule={balanced['selection_rule']}")
        if aggressive and aggressive["cutoff_eok"] != balanced["cutoff_eok"]:
            print(f"  공격형 참고 cutoff: 시총 ≥{aggressive['cutoff_eok']:,}억  "
                  f"→ EV {aggressive['ev']:+.2f}% (ΔEV {aggressive['ev_delta_vs_no_cut']:+.2f}%p) "
                  f"· coverage {aggressive['coverage_pct']}%")
    else:
        print("  권고 cutoff 없음 — 신뢰 가드(n/coverage) 통과 구간 부족")
    print(f"  산출: {csv_path}")
    print(f"        {json_path}")
    print("  ⚠️ 이 결과는 단일 양극화장 구간이다. 10d/20d·종가진입에서도 무너지지 않는지 함께 볼 것.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--horizons", default="5,10,20")
    ap.add_argument("--entry", default="both", choices=["next_open", "rec_close", "both"])
    ap.add_argument("--pool", default="all", choices=["all", "top_pick"])
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--min-coverage", type=float, default=30.0)
    a = ap.parse_args()
    horizons = [int(h) for h in a.horizons.split(",")]
    entries = ["next_open", "rec_close"] if a.entry == "both" else [a.entry]
    run(a.data_dir, a.out_dir, horizons, entries, a.pool, a.min_n, a.min_coverage)


if __name__ == "__main__":
    main()
