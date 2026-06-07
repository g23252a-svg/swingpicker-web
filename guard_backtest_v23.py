# -*- coding: utf-8 -*-
"""
guard_backtest_v23.py — v23.0 GUARD ON/OFF A/B 백테스트
═══════════════════════════════════════════════════
과거 recommend 스냅샷 × N일 후 price_snapshot 매칭으로 진입후보의 N일 전방수익률을
계산하고, GUARD 통과군(PASS) vs 가드 탈락군(BLOCK/경보/감점)을 비교한다.

핵심 질문: "GUARD가 거르는 종목이 실제로 더 나빴는가?"
  → PASS군 평균수익률 ≥ REJECT군 평균수익률 이면 가드가 음(-)의 알파를 제거한 것.

실행:
    python guard_backtest_v23.py                 # horizon=3, 기본 data/
    python guard_backtest_v23.py --horizon 5
    python guard_backtest_v23.py --data-dir data --horizon 3 --routes ATTACK,ARMED

combo_optimizer의 recommend↔price_snapshot 매칭 패턴을 재사용하되, GUARD ON/OFF에 특화.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd

try:
    from guard_system import apply_guard_system
    from collector_config import DEFAULT_CONFIG
except Exception as e:  # pragma: no cover
    raise SystemExit(f"guard_system/collector_config import 실패: {e}")


def _snap_dates(data_dir: str):
    snap_files = sorted(glob.glob(os.path.join(data_dir, "price_snapshot_2026*.csv")))
    dates = [
        os.path.basename(f).replace("price_snapshot_", "").replace(".csv", "")
        for f in snap_files
    ]
    return dates


def build_guarded_trade_rows(data_dir: str, horizon: int,
                             routes: List[str]) -> pd.DataFrame:
    """recommend × (T+horizon) snapshot 매칭 + GUARD 적용 → 거래행 DataFrame."""
    rec_files = sorted(glob.glob(os.path.join(data_dir, "recommend_2026*.csv")))
    snap_dates = _snap_dates(data_dir)
    rows = []
    matched_days = 0

    for rf in rec_files:
        rec_ymd = os.path.basename(rf).replace("recommend_", "").replace(".csv", "")
        if rec_ymd not in snap_dates:
            continue
        idx = snap_dates.index(rec_ymd)
        fidx = idx + horizon
        if fidx >= len(snap_dates):
            continue
        try:
            rec = pd.read_csv(rf, dtype={"종목코드": str}, encoding="utf-8-sig")
            snap = pd.read_csv(
                os.path.join(data_dir, f"price_snapshot_{snap_dates[fidx]}.csv"),
                dtype={"종목코드": str}, encoding="utf-8-sig",
            )
        except Exception:
            continue

        rec["종목코드"] = rec["종목코드"].astype(str).str.zfill(6)
        snap["종목코드"] = snap["종목코드"].astype(str).str.zfill(6)

        # 진입 후보만 (ROUTE 게이트) — GUARD ON/OFF 비교 대상 universe
        if "ROUTE" in rec.columns:
            rec = rec[rec["ROUTE"].astype(str).isin(routes)].copy()
        if rec.empty:
            continue

        # GUARDED_ELITE 계산을 위해 ELITE_SCORE 없으면 DISPLAY_SCORE proxy 주입
        if "ELITE_SCORE" not in rec.columns and "DISPLAY_SCORE" in rec.columns:
            rec["ELITE_SCORE"] = pd.to_numeric(rec["DISPLAY_SCORE"], errors="coerce")

        # GUARD 적용 (shadow only — TOP_PICK 재게이트 영향 배제)
        guarded = apply_guard_system(
            rec, config=DEFAULT_CONFIG, kospi_ret_1d=None
        )

        future_close = dict(zip(
            snap["종목코드"], pd.to_numeric(snap["종가"], errors="coerce")
        ))
        matched_days += 1

        for _, r in guarded.iterrows():
            code = r["종목코드"]
            entry = float(pd.to_numeric(
                r.get("추천매수가", r.get("종가", 0)), errors="coerce") or 0)
            fc = future_close.get(code, np.nan)
            if entry <= 0 or pd.isna(fc):
                continue
            ret = (fc / entry - 1) * 100
            block = bool(r.get("GUARD_BLOCK", False))
            alert = bool(r.get("GUARD_FORCE_EXIT_ALERT", False))
            pen = float(r.get("GUARD_PENALTY_TOTAL", 0) or 0)
            rr_mult = float(r.get("GUARD_RR_MULT", 1.0) or 1.0)
            # PASS = 차단X · 경보X · 감점0 · RR배수 정상
            is_pass = (not block) and (not alert) and (pen == 0) and (rr_mult >= 1.0)
            rows.append({
                "ret": ret,
                "win": 1 if fc > entry else 0,
                "trade_date": rec_ymd,
                "guard_pass": is_pass,
                "guard_block": block,
                "guard_alert": alert,
                "guard_penalty": pen,
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    df.attrs["matched_days"] = matched_days
    return df


def _stats(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "avg_ret": 0.0, "ev": 0.0,
                "worst": 0.0, "best": 0.0}
    wr = sub["win"].mean() * 100
    avg = sub["ret"].mean()
    wins = sub[sub["ret"] > 0]["ret"]
    losses = sub[sub["ret"] <= 0]["ret"]
    aw = wins.mean() if len(wins) else 0.0
    al = abs(losses.mean()) if len(losses) else 0.0
    wd = wr / 100
    ev = wd * aw - (1 - wd) * al
    return {
        "n": int(n), "win_rate": round(wr, 1), "avg_ret": round(avg, 2),
        "ev": round(ev, 2), "worst": round(sub["ret"].min(), 1),
        "best": round(sub["ret"].max(), 1),
    }


def run(data_dir: str, horizon: int, routes: List[str]) -> dict:
    df = build_guarded_trade_rows(data_dir, horizon, routes)
    if df.empty:
        return {"error": "거래 표본 없음 (recommend/price_snapshot 매칭 실패)"}

    off = _stats(df)                          # OFF = 전체 후보 (가드 미적용)
    on = _stats(df[df["guard_pass"]])         # ON  = 가드 통과만
    rej = _stats(df[~df["guard_pass"]])       # 가드 탈락군
    blocked = _stats(df[df["guard_block"]])
    alerted = _stats(df[df["guard_alert"]])

    return {
        "matched_days": int(df.attrs.get("matched_days", 0)),
        "horizon": horizon, "routes": routes,
        "OFF_all": off, "ON_pass": on, "REJECT": rej,
        "G1G4_blocked": blocked, "G5_alerted": alerted,
        "lift_avg_ret": round(on["avg_ret"] - off["avg_ret"], 2),
        "lift_win_rate": round(on["win_rate"] - off["win_rate"], 1),
        "reject_minus_pass_ret": round(rej["avg_ret"] - on["avg_ret"], 2),
    }


def _fmt(label: str, s: dict) -> str:
    return (f"  {label:<14} n={s['n']:>4}  승률 {s['win_rate']:>5.1f}%  "
            f"평균 {s['avg_ret']:>+6.2f}%  EV {s['ev']:>+6.2f}  "
            f"[최악 {s['worst']:>+6.1f} / 최고 {s['best']:>+6.1f}]")


def print_report(res: dict) -> None:
    print("\n" + "═" * 72)
    print(f"🛡️  v23.0 GUARD ON/OFF A/B 백테스트  (horizon={res['horizon']}일, "
          f"ROUTE={','.join(res['routes'])}, 매칭일수 {res['matched_days']})")
    print("═" * 72)
    if "error" in res:
        print("  ⚠️", res["error"]); return
    print(_fmt("OFF (전체)", res["OFF_all"]))
    print(_fmt("ON (가드통과)", res["ON_pass"]))
    print(_fmt("REJECT (탈락)", res["REJECT"]))
    print("  " + "-" * 68)
    print(_fmt("└ G1/G4 차단", res["G1G4_blocked"]))
    print(_fmt("└ G5 추세경보", res["G5_alerted"]))
    print("  " + "-" * 68)
    lift = res["lift_avg_ret"]
    rmp = res["reject_minus_pass_ret"]
    print(f"  📈 ON−OFF 평균수익률 리프트: {lift:+.2f}%p  "
          f"(승률 {res['lift_win_rate']:+.1f}%p)")
    print(f"  🔻 탈락군−통과군 수익률 차: {rmp:+.2f}%p  "
          f"{'→ 가드가 음의 알파 제거 ✅' if rmp < 0 else '→ 추가 점검 필요 ⚠️'}")
    print("═" * 72 + "\n")


def main():
    ap = argparse.ArgumentParser(description="v23 GUARD ON/OFF A/B 백테스트")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--routes", default="ATTACK,ARMED",
                    help="콤마구분 ROUTE 화이트리스트")
    args = ap.parse_args()
    routes = [x.strip() for x in args.routes.split(",") if x.strip()]
    res = run(args.data_dir, args.horizon, routes)
    print_report(res)


if __name__ == "__main__":
    main()
