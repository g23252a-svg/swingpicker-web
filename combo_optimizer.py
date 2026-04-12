# -*- coding: utf-8 -*-
"""
combo_optimizer.py — 지표 조합 최적화 엔진 (v21.3)
═══════════════════════════════════════════════════
과거 추천 + 실현 수익률 데이터를 기반으로
최고 승률 지표 조합을 자동 탐색.

매일 파이프라인 종료 후 실행 → optimal_filter_latest.json 저장
→ 대시보드/TOP_PICK에서 활용
"""
import glob
import json
import logging
import os
from itertools import product

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_combo_optimization(
    data_dir: str,
    horizon: int = 3,
    min_samples: int = 10,
    top_n: int = 10,
) -> dict:
    """
    조합 최적화 실행.

    Args:
        data_dir: data/ 디렉토리 경로
        horizon: 보유일수 (3 or 5)
        min_samples: 최소 샘플 수
        top_n: 상위 N개 조합 저장

    Returns:
        {"best": {...}, "top_combos": [...], "meta": {...}}
    """
    # 1) 데이터 로드: 추천 CSV + N일 후 스냅샷 매칭
    rec_files = sorted(glob.glob(os.path.join(data_dir, "recommend_2026*.csv")))
    snap_files = sorted(glob.glob(os.path.join(data_dir, "price_snapshot_2026*.csv")))
    snap_dates = [
        os.path.basename(f).replace("price_snapshot_", "").replace(".csv", "")
        for f in snap_files
    ]

    rows = []
    matched_days = 0

    for rf in rec_files:
        rec_ymd = os.path.basename(rf).replace("recommend_", "").replace(".csv", "")
        if rec_ymd not in snap_dates:
            continue
        idx = snap_dates.index(rec_ymd)
        future_idx = idx + horizon
        if future_idx >= len(snap_dates):
            continue

        try:
            rec = pd.read_csv(rf, dtype={"종목코드": str}, encoding="utf-8-sig")
            snap = pd.read_csv(
                os.path.join(data_dir, f"price_snapshot_{snap_dates[future_idx]}.csv"),
                dtype={"종목코드": str}, encoding="utf-8-sig",
            )
        except Exception:
            continue

        rec["종목코드"] = rec["종목코드"].str.zfill(6)
        snap["종목코드"] = snap["종목코드"].str.zfill(6)
        future_close = dict(zip(snap["종목코드"], pd.to_numeric(snap["종가"], errors="coerce")))

        matched_days += 1
        for _, r in rec.iterrows():
            code = r["종목코드"]
            entry = float(pd.to_numeric(r.get("추천매수가", r.get("종가", 0)), errors="coerce") or 0)
            fc = future_close.get(code, np.nan)
            if entry <= 0 or pd.isna(fc):
                continue

            rows.append({
                "ret": (fc / entry - 1) * 100,
                "win": 1 if fc > entry else 0,
                "S": float(r.get("STRUCT_SCORE", 0) or 0),
                "T": float(r.get("TIMING_SCORE", 0) or 0),
                "AI": float(r.get("AI_SCORE", r.get("ML_SCORE", 0)) or 0),
                "ROUTE": str(r.get("ROUTE", "")),
                "SCORE": float(r.get("DISPLAY_SCORE", 0) or 0),
            })

    if not rows:
        logger.warning("combo_optimizer: 매칭 데이터 없음")
        return {}

    df = pd.DataFrame(rows)
    total_wr = df["win"].mean() * 100

    # 2) 조합 그리드 탐색
    combos = list(product(
        [60, 70, 80, 90],                                  # S_min
        [50, 60, 70, 80],                                  # T_min
        [40, 50, 60, 70],                                  # AI_min
        [["ATTACK", "ARMED"], ["ATTACK", "ARMED", "WAIT"]],  # ROUTE
    ))

    results = []
    for s_min, t_min, ai_min, routes in combos:
        sub = df[
            (df["S"] >= s_min)
            & (df["T"] >= t_min)
            & (df["AI"] >= ai_min)
            & (df["ROUTE"].isin(routes))
        ]
        n = len(sub)
        if n < min_samples:
            continue

        wr = sub["win"].mean() * 100
        avg_ret = sub["ret"].mean()
        results.append({
            "S_min": int(s_min),
            "T_min": int(t_min),
            "AI_min": int(ai_min),
            "routes": routes,
            "n": int(n),
            "win_rate": round(wr, 1),
            "avg_ret": round(avg_ret, 2),
        })

    if not results:
        logger.warning("combo_optimizer: 유효 조합 없음")
        return {}

    # 3) 정렬 — 승률 우선, 동점 시 수익률
    results.sort(key=lambda x: (-x["win_rate"], -x["avg_ret"]))
    best = results[0]
    top_combos = results[:top_n]

    output = {
        "best": best,
        "top_combos": top_combos,
        "meta": {
            "total_trades": len(df),
            "total_win_rate": round(total_wr, 1),
            "matched_days": matched_days,
            "horizon": horizon,
            "min_samples": min_samples,
        },
    }

    # 4) 저장
    try:
        from datetime import datetime
        output["generated_at"] = datetime.now().isoformat()

        out_path = os.path.join(data_dir, "optimal_filter_latest.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        logger.info(
            f"🎯 최적 조합: S≥{best['S_min']} T≥{best['T_min']} AI≥{best['AI_min']} "
            f"| {'+'.join(best['routes'])} | {best['n']}건 승률 {best['win_rate']}%"
        )
    except Exception as e:
        logger.warning(f"optimal_filter 저장 실패: {e}")

    return output


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    result = run_combo_optimization(data_dir)
    if result:
        b = result["best"]
        print(f"\n🎯 최적 조합: S≥{b['S_min']} T≥{b['T_min']} AI≥{b['AI_min']}")
        print(f"   ROUTE: {'+'.join(b['routes'])}")
        print(f"   {b['n']}건 | 승률 {b['win_rate']}% | 수익 {b['avg_ret']:+.2f}%")
