# -*- coding: utf-8 -*-
"""
axis_ic_report.py — [v28] 점수 축 예측력 야간 감사
═══════════════════════════════════════════════════
문제의식:
  AI 축에는 신뢰도 게이트(시간순 OOS 검증)가 있지만 STRUCT/TIMING 등
  룰 기반 축은 무조건 신뢰하는 비대칭이 있었다. 실측 결과 이 축들도
  하락장에서 역방향으로 작동했다 (81일 평균 IC: STRUCT -0.042 t=-2.9,
  ELITE -0.059 t=-4.0 / 상승장 4월엔 FINAL +0.052 t=+2.1).

역할:
  파이프라인 말미에 최근 N일의 일별 횡단면 Spearman IC
  (점수 순위 vs 5일 선행수익률 순위)를 계산해
  data/axis_ic_report_{ymd}.json (+ _latest)로 저장하고,
  유의미한 역주행(t < -2) 축은 경고 로그를 남긴다.

주의:
  - stale CARRY 행 제외 + CSV 종가와 실제 종가 1% 이내 일치 행만 사용
  - 선행수익률이 필요하므로 최근 5거래일은 표본에서 자동 제외됨
"""
from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AXES = (
    "STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE",
    "FINAL_SCORE", "ELITE_SCORE", "POC_GAP",
)
# 방향성: +1 = 높을수록 좋다고 설계된 축, -1 = 낮을수록 좋다고 설계된 지표.
# 판정은 (설계 방향 × IC)로 평가 — POC_GAP은 음(-)의 IC가 정상 작동이다.
AXIS_ORIENTATION = {
    "STRUCT_SCORE": +1, "TIMING_SCORE": +1, "AI_SCORE": +1,
    "FINAL_SCORE": +1, "ELITE_SCORE": +1,
    "POC_GAP": -1,
}
HORIZON_BDAYS = 5
LOOKBACK_DAYS = 60          # 최근 60거래일 롤링
MIN_CROSS_SECTION = 30      # 일별 최소 표본
IC_WARN_T = -2.0            # 역주행 경고 임계


def _spearman(a: pd.Series, b: pd.Series) -> float:
    ra, rb = a.rank(), b.rank()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_axis_ic_report(data_dir: str = "data",
                           lookback_days: int = LOOKBACK_DAYS) -> dict:
    """최근 lookback_days의 축별 IC 요약 dict 반환."""
    pq = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.parquet")))
    if not pq:
        return {"ok": False, "reason": "ohlcv_cache 없음"}
    try:
        ohlcv = pd.read_parquet(pq[-1])
    except Exception as e:
        return {"ok": False, "reason": f"ohlcv 로드 실패: {e}"}
    ohlcv.index = pd.to_datetime(ohlcv.index)
    px = ohlcv.pivot_table(index=ohlcv.index, columns="종목코드",
                           values="종가", aggfunc="last").sort_index()
    fwd = px.shift(-HORIZON_BDAYS) / px - 1

    files = sorted(glob.glob(os.path.join(data_dir, "recommend_2*.csv")))[-lookback_days:]
    ics: dict = {a: [] for a in AXES}
    n_days = 0
    for f in files:
        ymd = os.path.basename(f)[-12:-4]
        try:
            d = pd.to_datetime(ymd)
        except ValueError:
            continue
        if d not in fwd.index:
            continue
        try:
            rec = pd.read_csv(f, encoding="utf-8-sig",
                              dtype={"종목코드": str}, low_memory=False)
        except Exception:
            continue
        if "종목코드" not in rec.columns:
            continue
        rec["종목코드"] = rec["종목코드"].str.zfill(6)
        if "ROUTE" in rec.columns:
            rec = rec[rec["ROUTE"] != "CARRY"]
        real = px.loc[d]
        csv_close = pd.to_numeric(rec.get("종가"), errors="coerce")
        rec = rec[np.isclose(csv_close, rec["종목코드"].map(real), rtol=0.01)]
        fwd_ret = rec["종목코드"].map(fwd.loc[d])
        mask_base = fwd_ret.notna()
        if mask_base.sum() < MIN_CROSS_SECTION:
            continue
        n_days += 1
        for a in AXES:
            if a not in rec.columns:
                continue
            v = pd.to_numeric(rec[a], errors="coerce")
            m = mask_base & v.notna()
            if a == "AI_SCORE":
                m &= v != 0  # 프로덕션 배제 상태의 0점은 제외
            if m.sum() < MIN_CROSS_SECTION:
                continue
            ic = _spearman(v[m], fwd_ret[m])
            if not np.isnan(ic):
                ics[a].append(ic)

    axes_out = {}
    warnings_out = []
    for a in AXES:
        arr = np.array(ics[a], dtype=float)
        if len(arr) < 10:
            axes_out[a] = {"n_days": int(len(arr)), "verdict": "표본부족"}
            continue
        mean = float(arr.mean())
        t = float(mean / (arr.std(ddof=1) / np.sqrt(len(arr)))) if arr.std(ddof=1) > 0 else 0.0
        orient = AXIS_ORIENTATION.get(a, +1)
        t_eff = t * orient  # 설계 방향 기준 t
        verdict = "역주행" if t_eff < IC_WARN_T else ("유효" if t_eff > 2.0 else "무신호")
        axes_out[a] = {
            "n_days": int(len(arr)),
            "mean_ic": round(mean, 4),
            "t_stat": round(t, 2),
            "orientation": orient,
            "pct_positive": round(float((arr > 0).mean()), 3),
            "verdict": verdict,
        }
        if verdict == "역주행":
            _dir = "점수가 높을수록 수익률 낮음" if orient > 0 else "낮을수록 좋아야 하는 지표가 반대로 작동"
            msg = f"⚠️ 축 역주행: {a} IC {mean:+.4f} (t={t:.1f}, {len(arr)}일) — {_dir}"
            warnings_out.append(msg)
            logger.warning(msg)

    return {
        "ok": True,
        "horizon_bdays": HORIZON_BDAYS,
        "lookback_days": lookback_days,
        "n_days_used": n_days,
        "axes": axes_out,
        "warnings": warnings_out,
    }


def save_axis_ic_report(data_dir: str = "data", trade_ymd: str = None) -> dict:
    """리포트 계산 + JSON 저장 (dated + latest). 반환: 리포트 dict."""
    report = compute_axis_ic_report(data_dir)
    try:
        names = [f"axis_ic_report_latest.json"]
        if trade_ymd:
            names.append(f"axis_ic_report_{str(trade_ymd)[:8]}.json")
        for n in names:
            with open(os.path.join(data_dir, n), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=1)
    except Exception as e:
        logger.warning(f"axis IC 리포트 저장 실패: {e}")
    return report
