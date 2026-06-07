# -*- coding: utf-8 -*-
"""
momentum_lane_backtest.py — v23.1 Momentum Lane 전방수익률 검증.

과거 recommend 스냅샷 × N일 후 price_snapshot 매칭으로, 모멘텀 레인(과열·가드통과)
Tier A(실전)/Tier B(관찰)의 실제 전방수익률과 시장국면 게이트 발생률을 측정한다.
guard_backtest_v23.py의 매칭 패턴을 재사용한다.

사용:
    python momentum_lane_backtest.py                  # horizon=3
    python momentum_lane_backtest.py --horizon 5
    python momentum_lane_backtest.py --data-dir data --horizon 3
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _snap_dates(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "price_snapshot_2026*.csv")))
    dates = [os.path.basename(f).replace("price_snapshot_", "").replace(".csv", "")
             for f in files]
    return dates


def _future_close_map(data_dir, ymd):
    p = os.path.join(data_dir, f"price_snapshot_{ymd}.csv")
    if not os.path.exists(p):
        return {}
    s = pd.read_csv(p, dtype={"종목코드": str}, encoding="utf-8-sig")
    s["종목코드"] = s["종목코드"].astype(str).str.zfill(6)
    return dict(zip(s["종목코드"], pd.to_numeric(s["종가"], errors="coerce")))


def build_rows(data_dir, horizon):
    """모멘텀 레인 + 전방수익률 거래행."""
    try:
        from guard_system import apply_guard_system
        from momentum_lane import apply_momentum_lane, compute_market_risk_off
    except Exception as e:
        print(f"import 실패: {e}", file=sys.stderr)
        return pd.DataFrame()

    kospi_path = os.path.join(data_dir, "kospi_daily.csv")
    df_kospi = pd.read_csv(kospi_path) if os.path.exists(kospi_path) else None

    snap_dates = _snap_dates(data_dir)
    d2i = {d: i for i, d in enumerate(snap_dates)}
    rec_files = sorted(glob.glob(os.path.join(data_dir, "recommend_2026*.csv")))

    rows = []
    risk_off_days = 0
    total_days = 0
    for rf in rec_files:
        ymd = os.path.basename(rf)[10:18]
        if ymd not in d2i:
            continue
        fidx = d2i[ymd] + horizon
        if fidx >= len(snap_dates):
            continue
        fmap = _future_close_map(data_dir, snap_dates[fidx])
        if not fmap:
            continue

        risk_off, _ = compute_market_risk_off(df_kospi=df_kospi, asof=ymd)
        total_days += 1
        if risk_off:
            risk_off_days += 1

        rec = pd.read_csv(rf, dtype={"종목코드": str}, encoding="utf-8-sig")
        rec["종목코드"] = rec["종목코드"].astype(str).str.zfill(6)
        try:
            rec = apply_guard_system(rec)
            rec = apply_momentum_lane(rec, market_risk_off=risk_off)
        except Exception:
            continue

        for _, r in rec.iterrows():
            entry = pd.to_numeric(r.get("추천매수가", r.get("종가", 0)), errors="coerce")
            fc = fmap.get(r["종목코드"], np.nan)
            if not entry or entry <= 0 or pd.isna(fc):
                continue
            ret = (fc - entry) / entry * 100.0
            tier = "A" if int(r.get("MOMENTUM_LANE", 0)) == 1 else (
                "B" if int(r.get("MOMENTUM_WATCH", 0)) == 1 else "")
            rows.append({"date": ymd, "tier": tier, "ret": ret,
                         "risk_off": bool(risk_off)})

    out = pd.DataFrame(rows)
    out.attrs["risk_off_days"] = risk_off_days
    out.attrs["total_days"] = total_days
    return out


def _stats(s):
    if len(s) == 0:
        return None
    return dict(n=len(s), win=(s > 0).mean() * 100, mean=s.mean(),
                med=s.median(), p5=np.percentile(s, 5), mx=s.max())


def run(data_dir, horizon):
    df = build_rows(data_dir, horizon)
    if len(df) == 0:
        print("거래 표본 없음 (recommend/price_snapshot 매칭 실패)")
        return

    rd = df.attrs.get("risk_off_days", 0)
    td = df.attrs.get("total_days", 0)

    print("=" * 72)
    print(f"  Momentum Lane 백테스트  (T+{horizon}, 매칭 {len(df):,}건)")
    print("=" * 72)
    print(f"  시장국면 게이트: risk_off {rd}/{td}일 "
          f"({(rd/td*100 if td else 0):.0f}%) — 그날 레인 OFF\n")

    print(f"  {'그룹':18s}{'n':>7s}{'승률':>9s}{'평균':>9s}{'중앙':>8s}{'최악5%':>9s}{'최고':>8s}")
    for tier, label in [("A", "⚡Tier A(실전)"), ("B", "Tier B(관찰)")]:
        st = _stats(df[df["tier"] == tier]["ret"])
        if st:
            print(f"  {label:18s}{st['n']:7,}{st['win']:8.1f}%{st['mean']:+8.2f}%"
                  f"{st['med']:+7.2f}%{st['p5']:+8.2f}%{st['mx']:+7.1f}%")
    # 레인 미진입(과열 외 + 가드탈락)
    st_out = _stats(df[df["tier"] == ""]["ret"])
    if st_out:
        print(f"  {'레인 외 전체':18s}{st_out['n']:7,}{st_out['win']:8.1f}%"
              f"{st_out['mean']:+8.2f}%{st_out['med']:+7.2f}%{st_out['p5']:+8.2f}%{st_out['mx']:+7.1f}%")

    a = df[df["tier"] == "A"]["ret"]
    base = df[df["tier"] == ""]["ret"]
    if len(a) and len(base):
        print(f"\n  ▶ Tier A − 레인외: {a.mean() - base.mean():+.2f}%p "
              f"(승률 {(a > 0).mean() * 100 - (base > 0).mean() * 100:+.1f}%p)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--horizon", type=int, default=3)
    args = ap.parse_args()
    run(args.data_dir, args.horizon)


if __name__ == "__main__":
    main()

