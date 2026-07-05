# -*- coding: utf-8 -*-
"""
honest_prob.py — v24.9 정직 확률 레이어 (단기 상승확률의 진실 표시)
═══════════════════════════════════════════════════════════════════
'단기간 상승확률이 높은 종목'을 만들려는 연구의 정직한 결론을 제품에 반영한다.

■ 연구 결과 (2026-07-03 · OHLCV 실경로 · 종가진입 · walk-forward · 29,703표본/84일)
  1. 24개 피처 × 2거래일 수익률 스피어만: 안정 피처 전무에 가깝고(최대 |ρ|≈0.05,
     전부 음의 '회피' 방향), 모멘텀(r1/r3/r5·RSI)은 반기마다 부호 반전 — 랭킹 불가.
  2. 시장레짐(KOSPI 3일) 조건부 확률: IS→OOS 캘리브레이션 붕괴
     (≤-4% 버킷 69%→42%, -4~-2% 버킷 30%→84% 역전) — 레짐 예측 확률도 불가.
  3. 회피 컴포짓 OOS 십분위 스프레드 0.4%p — 극단 필터도 무효.
  → 이 유니버스·단기 지평에서 '종목별 상승확률'의 검증 가능한 엣지는 현재 없다.
  상세: REPORT_v249_srp_null_result.md

■ 그래서 이 모듈이 하는 것 (지어내지 않는다)
  · SRP_BASE_PROB_2D : 최근 lookback(기본 20)거래일 유니버스의 '2거래일 후 종가 상승'
    실측 비율(%). 예측 모델이 아니라 기술통계 — 현재 장세의 기저율을 보여준다.
    (직전 검증에서 이 기저율이 51%→37%로 변한 것이 개별 종목 요인보다 압도적이었다)
  · SRP_EDGE_STATUS  : 'NO_EDGE_VALIDATED' — 종목 간 유의차가 검증되지 않았음을
    데이터로 명시. 재검증에서 엣지가 확인되기 전까지 이 상태가 진실이다.
  · SRP_NOTE         : 사용자용 한 줄 설명.
  공식 컬럼(TOP_PICK/BUY_NOW_ELIGIBLE)은 불변. 실패 시 조용히 생략(파이프라인 무해).

■ 검증에서 살아남아 실행에 쓸 수 있는 것 (이 모듈 밖, 기배포)
  · D+1 컷 -4% (v24.7): 종가진입 기준 평균 +1.1~1.6%p, -10% 꼬리 절반.
  · 지정가 미체결율 실측 37% (v24.8 백필): '추천매수가 대기'의 절반 가까이는 안 잡힌다.
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger("honest_prob")

SRP_OUTPUT_COLS = ["SRP_BASE_PROB_2D", "SRP_EDGE_STATUS", "SRP_NOTE"]
EDGE_STATUS = "NO_EDGE_VALIDATED"   # 재검증 통과 시에만 변경할 것
_H = 2                              # 지평(거래일)
_DEFAULT_LOOKBACK = 20


def _latest_ohlcv(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.parquet")))
    return files[-1] if files else None


def compute_market_base_prob(data_dir: str = "data",
                             lookback: int = _DEFAULT_LOOKBACK,
                             horizon: int = _H) -> dict:
    """최근 lookback 거래일 추천 유니버스의 '+horizon 거래일 종가 상승' 실측 비율.

    반환: {prob_pct, n_obs, n_days, asof, ok} — 데이터 부족/부재 시 ok=False.
    순수 기술통계: 결과가 확정된(종가 +horizon 존재) 날만 사용, 미래정보 없음.
    """
    out = {"prob_pct": np.nan, "n_obs": 0, "n_days": 0, "asof": "", "ok": False}
    pq = _latest_ohlcv(data_dir)
    if pq is None:
        return out
    try:
        o = pd.read_parquet(pq).reset_index()
        dc = "Date" if "Date" in o.columns else o.columns[0]
        o["code"] = o["종목코드"].astype(str).str.zfill(6)
        o["d"] = pd.to_datetime(o[dc]).dt.strftime("%Y%m%d").astype(int)
        o = o.sort_values(["code", "d"])
        close = o.pivot_table(index="d", columns="code", values="종가", aggfunc="last")
        fwd_up = (close.shift(-horizon) > close)          # 각 (일, 종목) 상승여부
        has_fwd = close.shift(-horizon).notna() & close.notna()

        rec_files = sorted(glob.glob(os.path.join(data_dir, "recommend_2026*.csv")))
        # 결과 확정된 추천일만, 최근 lookback개
        days = [int(f[-12:-4]) for f in rec_files if int(f[-12:-4]) in close.index]
        days = [d for d in days if bool(has_fwd.loc[d].any())][-lookback:]
        if len(days) < 5:
            return out
        n_up = n_tot = 0
        for f in rec_files:
            d = int(f[-12:-4])
            if d not in days:
                continue
            codes = pd.read_csv(f, usecols=["종목코드"], dtype={"종목코드": str})["종목코드"].str.zfill(6)
            codes = [c for c in codes if c in close.columns]
            m = has_fwd.loc[d, codes]
            n_tot += int(m.sum())
            n_up += int(fwd_up.loc[d, codes][m].sum())
        if n_tot < 200:
            return out
        out.update(prob_pct=round(n_up / n_tot * 100, 1), n_obs=n_tot,
                   n_days=len(days), asof=str(max(days)), ok=True)
        return out
    except Exception as e:  # 어떤 실패도 파이프라인을 막지 않는다
        logger.warning(f"honest_prob 기저확률 계산 실패: {e}")
        return out


def add_srp_columns(df: pd.DataFrame, data_dir: str = "data",
                    lookback: int = _DEFAULT_LOOKBACK) -> pd.DataFrame:
    """추천 df에 정직 확률 컬럼 부여 (공식 컬럼 불변, 실패 시 원본 그대로)."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    base = compute_market_base_prob(data_dir, lookback=lookback)
    if base["ok"]:
        out["SRP_BASE_PROB_2D"] = base["prob_pct"]
        out["SRP_NOTE"] = (f"최근 {base['n_days']}거래일 유니버스 실측 2일 상승률 "
                           f"{base['prob_pct']}% (n={base['n_obs']:,}) · "
                           f"종목별 유의차 미검증")
    else:
        out["SRP_BASE_PROB_2D"] = np.nan
        out["SRP_NOTE"] = "기저확률 계산불가(데이터 부족) · 종목별 유의차 미검증"
    out["SRP_EDGE_STATUS"] = EDGE_STATUS
    return out


def srp_summary(df: pd.DataFrame) -> dict:
    if df is None or "SRP_BASE_PROB_2D" not in getattr(df, "columns", []):
        return {}
    v = pd.to_numeric(df["SRP_BASE_PROB_2D"], errors="coerce")
    return {"base_prob_2d": float(v.iloc[0]) if len(v) and np.isfinite(v.iloc[0]) else None,
            "status": EDGE_STATUS}
