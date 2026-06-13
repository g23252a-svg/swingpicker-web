# -*- coding: utf-8 -*-
"""
scripts/backtest_data_integrity.py — v24.2 DATA_INTEGRITY 소급 백테스트
═══════════════════════════════════════════════════
v24.1 무결성 게이트(P0-C)를 과거 추천 이력 전체에 소급 적용해
"무결성 위반/폭등 플래그 종목을 샀다면 어떻게 됐나"를 수치로 만든다.
→ demote_official 승격/기각 결정의 근거 (v24.2 작업 #1).

방법론
  [as-of 감사]  추천일 D의 ohlcv_cache_D.parquet(그날 파이프라인이 실제로
    본 데이터)에서 종목별 D 이하 구간을 잘라 audit_ohlcv_window를 돌린다.
    최신 데이터로 재감사하면 이후 수정주가 반영으로 당시의 점프가 사라질 수
    있으므로, 반드시 '그날의 캐시'를 쓴다. 임계값은 전부
    collector_config.DataIntegrityConfig(SSOT)에서 읽는다 — 하드코딩 금지.
  [forward 수익률]  combo_optimizer._load_trade_rows와 동일 관례:
    entry = 추천매수가(없으면 종가), exit = D+h번째 price_snapshot의 종가
    (h = snapshot 거래일 스텝). 호라이즌 기본 5/10/20.
  [그룹 비교]  CLEAN vs DI_BAD(무결성 위반) vs SURGE(폭등 플래그) — 
    demote_official은 '무결성 위반'에만 작용하므로 SURGE와 분리 집계한다.
    핵심 헤드라인: BUY_NOW_ELIGIBLE==1 ∩ DI_BAD 의 forward 성과.

산출물 (data/ 에 저장 — 추천 산식·파이프라인 무변경, 순수 분석)
  data_integrity_asof_panel.csv      : (rec_ymd × 종목) as-of 감사 패널
                                       → combo_optimizer DQ 변수의 입력
  backtest_data_integrity_latest.json: 그룹 통계 전체

실행
  python scripts/backtest_data_integrity.py                 # data/ 자동 탐색
  python scripts/backtest_data_integrity.py --data-dir data --horizons 5 10 20
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, PROJECT_ROOT)

from data_integrity import audit_ohlcv_window  # noqa: E402

logger = logging.getLogger("backtest_data_integrity")

PANEL_FILENAME = "data_integrity_asof_panel.csv"
RESULT_FILENAME = "backtest_data_integrity_latest.json"

PANEL_COLS = [
    "rec_ymd", "종목코드", "종목명", "DI_OK", "DI_REASON", "DI_NBAD",
    "SURGE_FLAG", "FLAGGED", "AUDITED",
]

# forward 수익률 이상치 경계 — |ret|가 이를 넘으면 D~D+h 사이 수정주가
# 단절(병합/감자 등) 가능성이 높아 robust 통계에서 별도 분리한다.
OUTLIER_RET_PCT = 200.0


# ═══════════════════════════════════════════════════
#  설정 로드 (SSOT)
# ═══════════════════════════════════════════════════
def _load_di_cfg():
    """DataIntegrityConfig를 SSOT에서 로드 (실패 시 모듈 폴백값과 동일)."""
    try:
        from collector_config import DEFAULT_CONFIG
        return DEFAULT_CONFIG.data_integrity
    except Exception as e:
        logger.warning("collector_config 로드 실패 — data_integrity 폴백 기본값 사용: %s", e)
        return None


def _cfg(di, name, default):
    return getattr(di, name, default) if di is not None else default


# ═══════════════════════════════════════════════════
#  Phase A — as-of 감사 패널
# ═══════════════════════════════════════════════════
def build_asof_panel(data_dir: str, *, rebuild: bool = False) -> pd.DataFrame:
    """추천 이력 전체에 무결성 게이트를 소급 적용한 패널을 만든다.

    각 추천일 D에 대해 그날의 ohlcv_cache_D.parquet으로 as-of 감사를 수행.
    캐시 없는 날 / 캐시에 없는 종목은 AUDITED=False (FLAGGED 판정은 surge만).
    """
    panel_path = os.path.join(data_dir, PANEL_FILENAME)
    if (not rebuild) and os.path.exists(panel_path):
        try:
            cached = pd.read_csv(panel_path, dtype={"종목코드": str, "rec_ymd": str},
                                 encoding="utf-8-sig")
            if list(cached.columns) == PANEL_COLS and not cached.empty:
                logger.info("기존 패널 재사용: %s (%d행) — --rebuild로 재생성 가능",
                            PANEL_FILENAME, len(cached))
                return cached
        except Exception as e:
            logger.warning("기존 패널 읽기 실패 — 재생성: %s", e)

    di = _load_di_cfg()
    window = int(_cfg(di, "window", 20))
    jump_limit = float(_cfg(di, "jump_limit_pct", 45.0))
    max_bad = int(_cfg(di, "max_bad_bars", 0))
    surge_th = float(_cfg(di, "surge_ret10_pct", 300.0))

    rec_files = sorted(glob.glob(os.path.join(data_dir, "recommend_2026*.csv")))
    rows: List[dict] = []
    n_days_audited = 0

    for rf in rec_files:
        rec_ymd = os.path.basename(rf).replace("recommend_", "").replace(".csv", "")
        cache_path = os.path.join(data_dir, f"ohlcv_cache_{rec_ymd}.parquet")

        try:
            rec = pd.read_csv(rf, dtype={"종목코드": str}, encoding="utf-8-sig",
                              low_memory=False)
        except Exception as e:
            logger.warning("추천 CSV 읽기 실패 — 스킵: %s (%s)", rf, e)
            continue
        if "종목코드" not in rec.columns:
            continue
        rec["종목코드"] = rec["종목코드"].astype(str).str.zfill(6)

        # 그날의 캐시 → 종목별 as-of OHLCV
        groups: Dict[str, pd.DataFrame] = {}
        asof_ts = pd.to_datetime(rec_ymd, format="%Y%m%d", errors="coerce")
        if os.path.exists(cache_path) and pd.notna(asof_ts):
            try:
                cache = pd.read_parquet(cache_path)
                cache_codes = cache["종목코드"].astype(str).str.zfill(6)
                for code, g in cache.groupby(cache_codes):
                    g = g.sort_index()
                    groups[code] = g[g.index <= asof_ts]
                n_days_audited += 1
            except Exception as e:
                logger.warning("OHLCV 캐시 읽기 실패 — %s 감사 생략: %s", rec_ymd, e)

        r10_series = (
            pd.to_numeric(rec["ret_10d_%"], errors="coerce").fillna(0.0)
            if "ret_10d_%" in rec.columns
            else pd.Series(0.0, index=rec.index)
        )

        for idx, r in rec.iterrows():
            code = r["종목코드"]
            name = str(r.get("종목명", ""))
            surge = bool(float(r10_series.loc[idx]) > surge_th)

            ohlcv = groups.get(code)
            if ohlcv is not None and len(ohlcv) > 0:
                ok, reason, nbad = audit_ohlcv_window(
                    ohlcv, window=window, jump_limit_pct=jump_limit,
                    max_bad_bars=max_bad,
                )
                audited = not str(reason).startswith("SKIP")
            else:
                ok, reason, nbad, audited = True, "SKIP:no_ohlcv", 0, False

            rows.append({
                "rec_ymd": rec_ymd,
                "종목코드": code,
                "종목명": name,
                "DI_OK": bool(ok),
                "DI_REASON": reason,
                "DI_NBAD": int(nbad),
                "SURGE_FLAG": surge,
                "FLAGGED": bool((not ok) or surge),
                "AUDITED": bool(audited),
            })

    panel = pd.DataFrame(rows, columns=PANEL_COLS)
    try:
        panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
        logger.info("패널 저장: %s (%d행 · 감사일 %d일)", panel_path, len(panel), n_days_audited)
    except Exception as e:
        logger.warning("패널 저장 실패 (분석은 계속): %s", e)
    return panel


# ═══════════════════════════════════════════════════
#  Phase B — forward 수익률 결합 (combo_optimizer 관례)
# ═══════════════════════════════════════════════════
def build_trades(data_dir: str, panel: pd.DataFrame,
                 horizons: List[int]) -> pd.DataFrame:
    """추천 행 × 호라이즌 forward 수익률 + 패널 플래그 결합."""
    snap_files = sorted(glob.glob(os.path.join(data_dir, "price_snapshot_2026*.csv")))
    snap_dates = [
        os.path.basename(f).replace("price_snapshot_", "").replace(".csv", "")
        for f in snap_files
    ]
    snap_close_cache: Dict[str, Dict[str, float]] = {}

    def _snap_close(ymd: str) -> Optional[Dict[str, float]]:
        if ymd in snap_close_cache:
            return snap_close_cache[ymd]
        try:
            s = pd.read_csv(os.path.join(data_dir, f"price_snapshot_{ymd}.csv"),
                            dtype={"종목코드": str}, encoding="utf-8-sig")
            s["종목코드"] = s["종목코드"].astype(str).str.zfill(6)
            m = dict(zip(s["종목코드"], pd.to_numeric(s["종가"], errors="coerce")))
        except Exception as e:
            logger.warning("snapshot 읽기 실패: %s (%s)", ymd, e)
            m = None
        snap_close_cache[ymd] = m
        return m

    pkey = panel.copy()
    pkey["rec_ymd"] = pkey["rec_ymd"].astype(str)
    pkey = pkey.set_index(["rec_ymd", "종목코드"])
    rec_files = sorted(glob.glob(os.path.join(data_dir, "recommend_2026*.csv")))
    rows: List[dict] = []

    def _flag01(v) -> int:
        """0/1·True/False·'True'/'False' 혼재 컬럼을 0/1로 정규화."""
        if isinstance(v, str):
            return 1 if v.strip().lower() in ("true", "1", "y", "yes") else 0
        try:
            return 1 if float(v) > 0 else 0
        except (TypeError, ValueError):
            return 0

    for rf in rec_files:
        rec_ymd = os.path.basename(rf).replace("recommend_", "").replace(".csv", "")
        if rec_ymd not in snap_dates:
            continue
        base_idx = snap_dates.index(rec_ymd)

        future_maps = {}
        for h in horizons:
            fi = base_idx + h
            if fi < len(snap_dates):
                future_maps[h] = _snap_close(snap_dates[fi])

        if not future_maps:
            continue

        try:
            rec = pd.read_csv(rf, dtype={"종목코드": str}, encoding="utf-8-sig",
                              low_memory=False)
        except Exception:
            continue
        rec["종목코드"] = rec["종목코드"].astype(str).str.zfill(6)

        for _, r in rec.iterrows():
            code = r["종목코드"]
            entry = float(pd.to_numeric(
                r.get("추천매수가", r.get("종가", 0)), errors="coerce") or 0)
            if entry <= 0:
                continue

            try:
                p = pkey.loc[(rec_ymd, code)]
                if isinstance(p, pd.DataFrame):
                    p = p.iloc[0]
            except KeyError:
                continue

            row = {
                "rec_ymd": rec_ymd,
                "종목코드": code,
                "종목명": str(r.get("종목명", "")),
                "ROUTE": str(r.get("ROUTE", r.get("상태", ""))),
                "TOP_PICK": _flag01(r.get("TOP_PICK", 0)),
                "BUY_NOW_ELIGIBLE": _flag01(r.get("BUY_NOW_ELIGIBLE", 0)),
                "MOMENTUM_LANE": _flag01(r.get("MOMENTUM_LANE", 0)),
                "DI_OK": bool(p["DI_OK"]),
                "DI_REASON": str(p["DI_REASON"]),
                "SURGE_FLAG": bool(p["SURGE_FLAG"]),
                "FLAGGED": bool(p["FLAGGED"]),
                "AUDITED": bool(p["AUDITED"]),
            }
            got_any = False
            for h in horizons:
                m = future_maps.get(h)
                fc = m.get(code, np.nan) if m else np.nan
                ret = (fc / entry - 1.0) * 100.0 if (pd.notna(fc) and entry > 0) else np.nan
                row[f"ret_h{h}"] = ret
                got_any = got_any or pd.notna(ret)
            if got_any:
                rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════
#  Phase C — 그룹 통계
# ═══════════════════════════════════════════════════
def _grp_stats(sub: pd.DataFrame, col: str) -> Optional[dict]:
    s = pd.to_numeric(sub[col], errors="coerce").dropna()
    if len(s) == 0:
        return None
    out_mask = s.abs() > OUTLIER_RET_PCT
    robust = s[~out_mask]
    base = robust if len(robust) >= 3 else s
    return {
        "n": int(len(s)),
        "win_rate": round(float((s > 0).mean() * 100), 1),
        "mean": round(float(base.mean()), 2),
        "median": round(float(s.median()), 2),
        "p10": round(float(s.quantile(0.10)), 2),
        "min": round(float(s.min()), 2),
        "n_outlier_200": int(out_mask.sum()),
    }


def _reason_class(reason: str, surge: bool) -> str:
    r = str(reason)
    tags = [t for t in ("V1", "V2", "V3") if t + ":" in r]
    if surge:
        tags.append("SURGE")
    if not tags:
        return "CLEAN"
    return "+".join(tags) if len(tags) > 1 else tags[0]


def _episodes(flag_df: pd.DataFrame) -> int:
    """플래그 행을 (종목, 연속 추천일 묶음) 에피소드 수로 환산 — 정직한 n."""
    if flag_df.empty:
        return 0
    n_ep = 0
    for _, g in flag_df.groupby("종목코드"):
        dts = pd.to_datetime(g["rec_ymd"], format="%Y%m%d", errors="coerce").sort_values()
        if dts.isna().all():
            n_ep += 1
            continue
        gaps = dts.diff().dt.days.fillna(99)
        n_ep += int((gaps > 7).sum()) + 1 if len(dts) else 0
    return n_ep


def summarize(trades: pd.DataFrame, horizons: List[int]) -> dict:
    """CLEAN vs DI_BAD vs SURGE 그룹 통계 (서브셋·사유별 포함)."""
    t = trades.copy()
    t["DI_BAD"] = ~t["DI_OK"].astype(bool)
    t["REASON_CLASS"] = [
        _reason_class(r, s) for r, s in zip(t["DI_REASON"], t["SURGE_FLAG"])
    ]

    subsets = {
        "ALL": t,
        "BUY_NOW_ELIGIBLE": t[t["BUY_NOW_ELIGIBLE"] == 1],
        "TOP_PICK": t[t["TOP_PICK"] == 1],
        "MOMENTUM_LANE": t[t["MOMENTUM_LANE"] == 1],
    }
    groups = {
        "CLEAN": lambda d: d[~d["FLAGGED"]],
        "FLAGGED_ANY": lambda d: d[d["FLAGGED"]],
        "DI_BAD": lambda d: d[d["DI_BAD"]],
        "SURGE": lambda d: d[d["SURGE_FLAG"]],
        "DI_BAD_ONLY": lambda d: d[d["DI_BAD"] & ~d["SURGE_FLAG"]],
    }

    result: dict = {
        "generated_at": datetime.now().isoformat(),
        "horizons": horizons,
        "outlier_ret_pct": OUTLIER_RET_PCT,
        "coverage": {
            "n_trades": int(len(t)),
            "n_audited": int(t["AUDITED"].sum()),
            "n_flagged": int(t["FLAGGED"].sum()),
            "n_di_bad": int(t["DI_BAD"].sum()),
            "n_surge": int(t["SURGE_FLAG"].sum()),
            "flagged_episodes": _episodes(t[t["FLAGGED"]]),
            "di_bad_episodes": _episodes(t[t["DI_BAD"]]),
        },
        "subsets": {},
        "reason_breakdown": {},
    }

    for sname, sdf in subsets.items():
        block: dict = {"n": int(len(sdf))}
        for gname, gf in groups.items():
            gd = gf(sdf)
            block[gname] = {
                "n": int(len(gd)),
                **{f"h{h}": _grp_stats(gd, f"ret_h{h}") for h in horizons},
            }
        result["subsets"][sname] = block

    for rc, g in t.groupby("REASON_CLASS"):
        result["reason_breakdown"][rc] = {
            "n": int(len(g)),
            **{f"h{h}": _grp_stats(g, f"ret_h{h}") for h in horizons},
        }

    return result


def trace_case(trades: pd.DataFrame, panel: pd.DataFrame, code: str,
               horizons: List[int]) -> List[dict]:
    """특정 종목(예: 에이프로젠 007460)의 소급 감사·수익률 추적."""
    rows = []
    pt = panel[panel["종목코드"] == code]
    tt = trades[trades["종목코드"] == code].set_index("rec_ymd") if not trades.empty else pd.DataFrame()
    for _, p in pt.iterrows():
        d = {
            "rec_ymd": str(p["rec_ymd"]),
            "DI_OK": bool(p["DI_OK"]),
            "DI_REASON": str(p["DI_REASON"]) if pd.notna(p["DI_REASON"]) else "",
            "SURGE_FLAG": bool(p["SURGE_FLAG"]),
        }
        if len(tt) and p["rec_ymd"] in tt.index:
            tr = tt.loc[p["rec_ymd"]]
            if isinstance(tr, pd.DataFrame):
                tr = tr.iloc[0]
            for h in horizons:
                v = tr.get(f"ret_h{h}")
                d[f"ret_h{h}"] = round(float(v), 2) if pd.notna(v) else None
        rows.append(d)
    return rows


# ═══════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="DATA_INTEGRITY 소급 백테스트 (v24.2)")
    ap.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--rebuild", action="store_true", help="as-of 패널 재생성")
    ap.add_argument("--trace", default="007460", help="추적 종목코드 (기본: 에이프로젠)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    panel = build_asof_panel(args.data_dir, rebuild=args.rebuild)
    if panel.empty:
        print("❌ 패널 비어 있음 — recommend/ohlcv_cache 데이터 확인 필요")
        return 1

    trades = build_trades(args.data_dir, panel, args.horizons)
    if trades.empty:
        print("❌ forward 매칭 0건 — price_snapshot 데이터 확인 필요")
        return 1

    result = summarize(trades, args.horizons)
    result["trace"] = {args.trace: trace_case(trades, panel, args.trace, args.horizons)}

    out_path = os.path.join(args.data_dir, RESULT_FILENAME)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 결과 저장: {out_path}")
    except Exception as e:
        logger.warning("결과 저장 실패: %s", e)

    # ── 콘솔 리포트 ──
    cov = result["coverage"]
    print("")
    print("═" * 64)
    print(" DATA_INTEGRITY 소급 백테스트 — v24.1 게이트 × 추천 이력")
    print("═" * 64)
    print(f" 매칭 거래행 {cov['n_trades']}건 · as-of 감사 {cov['n_audited']}건")
    print(f" 플래그 {cov['n_flagged']}건 (에피소드 {cov['flagged_episodes']}) · "
          f"무결성위반 {cov['n_di_bad']}건 (에피소드 {cov['di_bad_episodes']}) · "
          f"폭등 {cov['n_surge']}건")
    for sname in ("ALL", "BUY_NOW_ELIGIBLE", "TOP_PICK", "MOMENTUM_LANE"):
        blk = result["subsets"][sname]
        print("")
        print(f" [{sname}] n={blk['n']}")
        hdr = "  {:<13}{:>6}" + "".join("  h{:<2}: 평균{:>8} 승률{:>6} p10{:>8}".format(h, "", "", "") for h in args.horizons)
        for gname in ("CLEAN", "DI_BAD", "SURGE", "FLAGGED_ANY"):
            g = blk[gname]
            parts = [f"  {gname:<13}n={g['n']:>5}"]
            for h in args.horizons:
                st = g.get(f"h{h}")
                if st:
                    parts.append(f" | h{h}: {st['mean']:+7.2f}% 승률 {st['win_rate']:>5.1f}% p10 {st['p10']:+8.2f}%")
                else:
                    parts.append(f" | h{h}: (표본없음)")
            print("".join(parts))
    print("")
    print(" [사유별]")
    for rc, st in sorted(result["reason_breakdown"].items()):
        if rc == "CLEAN":
            continue
        h0 = st.get(f"h{args.horizons[0]}")
        msg = f"  {rc:<10} n={st['n']:>5}"
        if h0:
            msg += f" | h{args.horizons[0]} 평균 {h0['mean']:+.2f}% 승률 {h0['win_rate']}%"
        print(msg)
    tr = result["trace"].get(args.trace, [])
    if tr:
        print("")
        print(f" [추적: {args.trace}] {len(tr)}회 등장")
        for d in tr[:6]:
            print(f"  {d['rec_ymd']} DI_OK={d['DI_OK']} SURGE={d['SURGE_FLAG']} "
                  f"REASON={d['DI_REASON'][:40]}")
    print("═" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
