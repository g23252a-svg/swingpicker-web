# -*- coding: utf-8 -*-
"""[v58] 알파 엔진 실전 성적 야간 리포트 — 표본을 자동으로 쌓는다.

■ 왜 만드는가
  v29(2026-07-14)로 알파 엔진이 실전에 들어간 뒤, "요즘 알파가 맞고 있나"를
  물을 때마다 임시 스크립트를 짜서 재야 했다. 그 결과 2026-08-10에도
  "8/5~8/10 구간은 아직 측정 불가"라고만 답할 수 있었다 — 데이터는 있는데
  누적된 성적표가 없었기 때문이다.
  axis_ic_report(v28)가 룰 기반 축을 매일 감사하는 것과 같은 이유로,
  진입 SSOT인 알파 축도 매일 자기 성적을 남겨야 한다.

■ 무엇을 남기는가 (배치마다 재계산 — 멱등)
  ① IC   : 그날 ALPHA_SCORE와 선행수익의 스피어만 순위상관.
           표본이 얇을 때 수익 평균보다 훨씬 안정적이라 **1차 지표**로 둔다.
  ② 상위N: 알파 top1/3/5/10의 실현수익 + 같은 날 유니버스 평균 대비 초과.
  ③ 스톱 : 상위 종목이 -8% 장중 스톱에 걸린 비율. 2026-07 폭락 구간에서
           알파 1위가 13일 중 6일 스톱 체결이었고, 그것이 IC가 양수인데도
           수익이 음수였던 주된 경로였다. 원인을 구분해 기록한다.
  ④ 레짐 : risk_off 여부별 분리. 같은 엔진이 하락 구간과 급반등 구간에서
           정반대로 작동한 것이 실측됐다(7/15 IC +0.44 vs 7/29 IC -0.29).
           뭉쳐서 평균 내면 그 사실이 사라진다.

■ 실현수익 정의는 프로덕션과 같아야 한다
  진입 t+1 시가 · -8% 장중 스톱 · t+h 종가.
  상수는 services.pick_reliability에서 읽어 SSOT를 유지한다(어긋나면 화면이
  약속하는 것과 다른 것을 재게 된다 — v55.3에서 같은 실수를 한 번 고쳤다).

■ 과신 방지
  표본이 얇다(도입 후 영업일 기준 수십 일). 그래서 리포트는 항상 n을 함께
  담고, 판정을 내리지 않는다. 경고는 **IC가 유의하게 음수일 때만** 낸다 —
  수익 평균이 음수인 것은 폭락 구간에서 정상일 수 있기 때문이다.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("alpha_live_report")

ALPHA_LIVE_FROM = "20260714"        # v29 알파 엔진 실전 투입일
HORIZONS = (3, 5)
TOP_NS = (1, 3, 5, 10)
LOOKBACK_DAYS = 120                 # 최근 N개 배치 파일만 (초기 구간은 알파 컬럼 자체가 없다)
MIN_ROWS_PER_DAY = 10               # 이보다 적은 날은 횡단면 통계가 무의미
IC_WARN_T = -2.0                    # IC가 유의하게 음수면 경고 (역주행)
IC_WARN_MIN_DAYS = 5                # t검정이 불가할 때 '일관성' 경로의 최소 표본

CACHE_NAME = "alpha_live_report_latest.json"


def _stop_pct() -> float:
    """실현 정의의 스톱 폭 — pick_reliability와 같은 값이어야 한다."""
    try:
        from services.pick_reliability import STOP_PCT_HOLD

        return float(STOP_PCT_HOLD)
    except Exception as e:
        logger.warning(f"STOP_PCT_HOLD 참조 실패 ({e}) → 폴백 -0.08")
        return -0.08


def _load_snapshots(data_dir: str) -> tuple[dict, list]:
    px = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "price_snapshot_2*.csv"))):
        ymd = os.path.basename(f)[-12:-4]
        if not ymd.isdigit():
            continue
        try:
            t = pd.read_csv(f, encoding="utf-8-sig", dtype={"종목코드": str},
                            low_memory=False)
        except Exception:
            continue
        if "종목코드" not in t.columns or "종가" not in t.columns:
            continue
        t["종목코드"] = t["종목코드"].astype(str).str.zfill(6)
        px[ymd] = t.set_index("종목코드")
    return px, sorted(px)


def _realized(px: dict, days: list, code: str, ymd: str, h: int,
              stop: float) -> float:
    """진입 t+1 시가 · 스톱 장중 체결 · t+h 종가."""
    try:
        i = days.index(ymd)
    except ValueError:
        return np.nan
    if i + h >= len(days):
        return np.nan                      # 선행수익 미확정
    d1 = px[days[i + 1]]
    if code not in d1.index:
        return np.nan
    row = d1.loc[code]
    entry = pd.to_numeric(pd.Series([row.get("시가")]), errors="coerce").iloc[0]
    if not (np.isfinite(entry) and entry > 0):
        entry = pd.to_numeric(pd.Series([row.get("종가")]), errors="coerce").iloc[0]
    if not (np.isfinite(entry) and entry > 0):
        return np.nan
    stop_px = entry * (1.0 + stop)
    for j in range(i + 1, i + h + 1):
        dd = px[days[j]]
        if code not in dd.index:
            continue
        lo = pd.to_numeric(pd.Series([dd.loc[code].get("저가")]), errors="coerce").iloc[0]
        if np.isfinite(lo) and lo <= stop_px:
            return float(stop)             # 스톱 체결
    dh = px[days[i + h]]
    if code not in dh.index:
        return np.nan
    c = pd.to_numeric(pd.Series([dh.loc[code].get("종가")]), errors="coerce").iloc[0]
    if not np.isfinite(c):
        return np.nan
    return float(c) / float(entry) - 1.0


def _risk_off_map(data_dir: str) -> dict:
    """momentum_lane과 같은 정의로 risk_off 일자 맵."""
    path = os.path.join(data_dir, "kospi_daily.csv")
    if not os.path.exists(path):
        return {}
    try:
        from momentum_lane import REGIME_DEVIATION_FLOOR as FLOOR
    except Exception:
        FLOOR = -0.03
    try:
        k = pd.read_csv(path, encoding="utf-8-sig")
        k.columns = [str(c).strip().lstrip("﻿") for c in k.columns]
        k["date"] = k["date"].astype(str).str[:8]
        k = k.sort_values("date")
        c = pd.to_numeric(k["close"], errors="coerce")
        ma = c.rolling(20).mean()
        ro = (c < ma) & (ma < ma.shift(5)) & ((c / ma - 1.0) <= FLOOR)
        return dict(zip(k["date"], ro.fillna(False)))
    except Exception as e:
        logger.warning(f"risk_off 맵 생성 실패: {e}")
        return {}


def _agg(vals: list) -> dict:
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean_pct": round(100 * float(v.mean()), 3),
        "median_pct": round(100 * float(np.median(v)), 3),
        "win_rate": round(float((v > 0).mean()), 4),
        "cum_pct": round(100 * float(v.sum()), 2),
    }


def _tstat(vals: list) -> Optional[dict]:
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size < 3 or float(np.std(v, ddof=1)) == 0.0:
        return None
    from scipy import stats

    t, p = stats.ttest_1samp(v, 0.0)
    return {"n": int(v.size), "t": round(float(t), 3), "p": round(float(p), 4)}


def compute_alpha_live_report(data_dir: str = "data") -> dict:
    """알파 실전 성적을 배치 CSV + 가격 스냅샷에서 재계산 (멱등)."""
    from scipy import stats

    stop = _stop_pct()
    px, days = _load_snapshots(data_dir)
    if len(days) < 5:
        return {"ok": False, "reason": "가격 스냅샷 부족"}
    ro_map = _risk_off_map(data_dir)

    files = sorted(glob.glob(os.path.join(data_dir, "recommend_2*.csv")))[-LOOKBACK_DAYS:]
    per_day, skipped = [], 0
    for f in files:
        ymd = os.path.basename(f)[-12:-4]
        if not ymd.isdigit() or ymd < ALPHA_LIVE_FROM:
            continue
        try:
            rec = pd.read_csv(f, encoding="utf-8-sig", dtype={"종목코드": str},
                              low_memory=False)
        except Exception as e:
            logger.warning(f"[v58] CSV 스킵 {ymd}: {e}")
            skipped += 1
            continue
        if "ALPHA_SCORE" not in rec.columns or "종목코드" not in rec.columns:
            skipped += 1
            continue
        rec = rec.copy()
        rec["종목코드"] = rec["종목코드"].astype(str).str.zfill(6)
        rec["_a"] = pd.to_numeric(rec["ALPHA_SCORE"], errors="coerce")
        rec = rec[rec["_a"].notna()]
        if len(rec) < MIN_ROWS_PER_DAY:
            continue

        row_out = {"ymd": ymd, "pool": int(len(rec)),
                   "risk_off": bool(ro_map.get(ymd, False))}
        any_h = False
        for h in HORIZONS:
            rets = np.array([_realized(px, days, c, ymd, h, stop)
                             for c in rec["종목코드"]], dtype=float)
            ok = np.isfinite(rets)
            if ok.sum() < MIN_ROWS_PER_DAY:
                continue                    # 선행수익 미확정 지평
            any_h = True
            sub = rec.loc[ok].assign(_ret=rets[ok]).sort_values("_a", ascending=False)
            ic = _spearman_safe(sub["_a"], sub["_ret"])
            hd = {"n_scored": int(len(sub)),
                  "ic": None if ic is None else round(ic, 4),
                  "universe_pct": round(100 * float(sub["_ret"].mean()), 3)}
            for k in TOP_NS:
                head = sub.head(k)
                hd[f"top{k}_pct"] = round(100 * float(head["_ret"].mean()), 3)
                hd[f"top{k}_excess_pct"] = round(
                    100 * float(head["_ret"].mean() - sub["_ret"].mean()), 3)
                hd[f"top{k}_stop_rate"] = round(
                    float(np.isclose(head["_ret"], stop).mean()), 4)
            hd["top1_name"] = str(sub["종목명"].iloc[0]) if "종목명" in sub.columns else ""
            # [v63] 엔진이 실제로 고르는 규칙(퍼널 통과 + 알파×RR)으로도 계산
            gk = _gate_rank_key(rec.loc[ok])
            if gk is not None and gk.notna().any():
                gsub = (rec.loc[ok].assign(_ret=rets[ok], _k=gk)
                        .dropna(subset=["_k"]).sort_values("_k", ascending=False))
                hd["gated_pool"] = int(len(gsub))
                for k in TOP_NS:
                    ghead = gsub.head(k)
                    if len(ghead) == 0:
                        continue
                    hd[f"gated_top{k}_pct"] = round(
                        100 * float(ghead["_ret"].mean()), 3)
                    hd[f"gated_top{k}_excess_pct"] = round(
                        100 * float(ghead["_ret"].mean() - sub["_ret"].mean()), 3)
                    hd[f"gated_top{k}_stop_rate"] = round(
                        float(np.isclose(ghead["_ret"], stop).mean()), 4)
                if "종목명" in gsub.columns:
                    hd["gated_top1_name"] = str(gsub["종목명"].iloc[0])
            else:
                hd["gated_pool"] = 0
            row_out[f"h{h}"] = hd
        if any_h:
            per_day.append(row_out)

    if not per_day:
        return {"ok": False, "reason": "선행수익이 확정된 배치일 없음",
                "files_scanned": len(files), "files_skipped": skipped,
                "alpha_live_from": ALPHA_LIVE_FROM}

    out = {
        "ok": True,
        "alpha_live_from": ALPHA_LIVE_FROM,
        "asof": per_day[-1]["ymd"],
        "n_days": len(per_day),
        "files_scanned": len(files),
        "files_skipped": skipped,
        "stop_pct": stop,
        "definition": f"진입 t+1 시가 · {stop:.0%} 장중 스톱 · t+h 종가",
        "horizons": {},
        "per_day": per_day[-40:],
        "caveat": ("표본이 얇다(도입 후 수십 영업일). IC를 1차 지표로 보고, "
                   "수익 평균은 스톱 체결률·레짐과 함께 읽을 것. "
                   "이 리포트는 판정을 내리지 않는다."),
    }

    for h in HORIZONS:
        rows = [r for r in per_day if f"h{h}" in r]
        if not rows:
            continue
        hh = f"h{h}"
        ics = [r[hh]["ic"] for r in rows if r[hh].get("ic") is not None]
        blk = {
            "n_days": len(rows),
            "ic_mean": round(float(np.mean(ics)), 4) if ics else None,
            "ic_median": round(float(np.median(ics)), 4) if ics else None,
            "ic_positive_days": int(sum(1 for x in ics if x > 0)),
            "ic_t": _tstat(ics),
            "universe": _agg([r[hh]["universe_pct"] / 100 for r in rows]),
        }
        for k in TOP_NS:
            blk[f"top{k}"] = _agg([r[hh][f"top{k}_pct"] / 100 for r in rows])
            ex = [r[hh][f"top{k}_excess_pct"] / 100 for r in rows]
            blk[f"top{k}"]["excess_mean_pct"] = round(100 * float(np.mean(ex)), 3)
            blk[f"top{k}"]["excess_t"] = _tstat(ex)
            blk[f"top{k}"]["stop_rate"] = round(
                float(np.mean([r[hh][f"top{k}_stop_rate"] for r in rows])), 4)
            # [v63] 엔진이 실제로 고르는 규칙 기준 — 재구성된 날만 집계한다
            grows = [r for r in rows if r[hh].get(f"gated_top{k}_pct") is not None]
            if len(grows) >= 3:
                blk[f"gated_top{k}"] = _agg(
                    [r[hh][f"gated_top{k}_pct"] / 100 for r in grows])
                gex = [r[hh][f"gated_top{k}_excess_pct"] / 100 for r in grows]
                blk[f"gated_top{k}"]["excess_mean_pct"] = round(
                    100 * float(np.mean(gex)), 3)
                blk[f"gated_top{k}"]["excess_t"] = _tstat(gex)
                blk[f"gated_top{k}"]["stop_rate"] = round(float(np.mean(
                    [r[hh][f"gated_top{k}_stop_rate"] for r in grows])), 4)
                blk[f"gated_top{k}"]["n_days"] = len(grows)
        # 레짐 분리 — 같은 엔진이 국면별로 정반대일 수 있다
        for lab, sel in (("risk_off", True), ("normal", False)):
            sub = [r for r in rows if bool(r["risk_off"]) is sel]
            if len(sub) >= 3:
                sics = [r[hh]["ic"] for r in sub if r[hh].get("ic") is not None]
                blk[f"by_{lab}"] = {
                    "n_days": len(sub),
                    "ic_mean": round(float(np.mean(sics)), 4) if sics else None,
                    "top1": _agg([r[hh]["top1_pct"] / 100 for r in sub]),
                    "universe": _agg([r[hh]["universe_pct"] / 100 for r in sub]),
                }
        out["horizons"][hh] = blk

    # 경고는 IC 역주행에만 (수익 음수는 폭락 구간에서 정상일 수 있다)
    #
    # [v58] t검정 하나만 보면 **가장 명백한 역주행을 놓친다**. IC가 매일 거의
    #   똑같이 음수면 분산이 0에 가까워 t가 정의되지 않고(None), 그래서
    #   "완벽하게 일관되게 틀리는" 최악의 경우에 경고가 안 났다.
    #   (신규 테스트 test_warns_when_alpha_is_inverted가 이걸 잡아냈다.)
    #   → t 경로와 별개로 '일관성' 경로를 둔다: 양수일이 하나도 없고 평균이
    #     음수이며 표본이 최소한 있으면 t와 무관하게 경고한다.
    warns = []
    for hh, blk in out["horizons"].items():
        t = (blk.get("ic_t") or {}).get("t")
        ic_mean = blk.get("ic_mean")
        nd = blk.get("n_days", 0)
        if t is not None and t <= IC_WARN_T:
            warns.append(f"{hh}: 알파 IC 역주행 (평균 {ic_mean:+.3f}, "
                         f"t={t:+.2f}, n={nd}일) — 진입 축 재검증 필요")
        elif (ic_mean is not None and ic_mean < 0 and nd >= IC_WARN_MIN_DAYS
              and blk.get("ic_positive_days", 0) == 0):
            warns.append(f"{hh}: 알파 IC 역주행 (평균 {ic_mean:+.3f}, "
                         f"양수일 0/{nd} — t검정 불가/무의미) — 진입 축 재검증 필요")
    out["warnings"] = warns
    return out


def _spearman_safe(a: pd.Series, b: pd.Series) -> Optional[float]:
    from scipy import stats

    if len(a) < 5 or a.nunique() < 3 or b.nunique() < 3:
        return None
    try:
        r = stats.spearmanr(a.values, b.values).correlation
        return None if r is None or not np.isfinite(r) else float(r)
    except Exception:
        return None


def save_alpha_live_report(data_dir: str = "data",
                           trade_ymd: Optional[str] = None) -> dict:
    """리포트 계산 + JSON 저장 (dated + latest). 반환: 리포트 dict."""
    report = compute_alpha_live_report(data_dir)
    try:
        names = [CACHE_NAME]
        if trade_ymd:
            names.append(f"alpha_live_report_{str(trade_ymd)[:8]}.json")
        for n in names:
            with open(os.path.join(data_dir, n), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=1)
    except Exception as e:
        logger.warning(f"알파 실전 리포트 저장 실패: {e}")
    return report


def load_alpha_live_report(data_dir: str = "data") -> Optional[dict]:
    p = os.path.join(data_dir, CACHE_NAME)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        return r if r.get("ok") else None
    except Exception as e:
        logger.warning(f"알파 실전 리포트 로드 실패: {e}")
        return None


# [v63] 화면에 뜨는 "1위"가 **엔진이 추천하지 않는 종목**이었다.
#   기존 top1은 전체 풀을 ALPHA_SCORE만으로 정렬한 1등이다. 그런데 실제 픽은
#   알파 문턱 통과 + 저점추세 당일 분위>30 + 리스크가드 + (v62)진입일 급등 제외를
#   거친 뒤 **알파 × 손익비**로 고른다(recommendation_quality의 rank_key).
#   그 차이가 크다 — v58.1이 이 줄을 결정 센터의 "이 엔진의 지난 성적"으로
#   띄우는데, 2026-08-17 시점 h5 기준 raw 1위는 -1.84%(유니버스 +0.75%)였고
#   같은 기간 퍼널 재구성 1위는 +3.01%였다. 즉 **엔진 성적을 실제보다 나쁘게**
#   표기하고 있었다(v61에서 고친 '표기가 사실과 다름'과 같은 유형, 방향만 반대).
#   그래서 두 축을 나란히 낸다: IC는 모델 진단이므로 전체 풀 기준을 유지하고,
#   상위N 수익은 **엔진이 실제로 고르는 규칙**으로도 계산한다.
_LT_PCTL_FLOOR_PCT = 30.0      # alpha_engine._LT_PCTL_FLOOR 와 같은 값
_SURGE_PCT = 5.0               # alpha_engine._SURGE_CHASE_PCT 와 같은 값


def _num_col(rec: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """컬럼이 있을 때만 숫자 Series를 준다.

    pd.to_numeric(None)은 **스칼라 NaN**을 돌려주므로 컬럼 부재를 그대로
    넘기면 뒤에서 Series 메서드가 터진다(실제로 이 함수 첫 구현이 그랬다).
    """
    if name not in rec.columns:
        return None
    v = pd.to_numeric(rec[name], errors="coerce")
    return v if v.notna().any() else None


def _gate_rank_key(rec: pd.DataFrame) -> Optional[pd.Series]:
    """엔진 퍼널을 통과한 행에 대해 알파×손익비 랭킹 키. 재구성 불가면 None.

    문턱·바닥값은 alpha_engine과 같은 값을 쓴다. 어긋나면 이 리포트가 엔진과
    다른 것을 측정하게 된다 — 그게 바로 v63에서 고친 결함이다.
    """
    a = _num_col(rec, "ALPHA_SCORE")
    if a is None:
        return None
    ok = a.notna()
    thr = _num_col(rec, "ALPHA_ENTRY_THRESHOLD")
    if thr is not None:
        ok &= a >= thr
    # 저점추세 당일 분위 (컬럼이 있으면 그대로, 없으면 원값으로 분위 계산)
    ltp = _num_col(rec, "LOW_TREND_PCTL")
    if ltp is None:
        lt = _num_col(rec, "Low_Trend_PCT")
        ltp = lt.rank(pct=True) * 100 if lt is not None else None
    if ltp is not None:
        ok &= ltp.isna() | (ltp > _LT_PCTL_FLOOR_PCT)
    # v62 급등 추격 제외 (구 배치엔 SURGE 컬럼이 없으므로 ret_1d로 재구성)
    r1 = _num_col(rec, "ret_1d_%")
    if r1 is None:
        r1 = _num_col(rec, "등락률")   # Series에 `or`를 쓰면 진리값이 모호해진다
    if r1 is not None:
        ok &= r1.isna() | (r1 < _SURGE_PCT)
    # 리스크 가드
    if "ENTRY_RISK_GATE_OK" in rec.columns:
        ok &= rec["ENTRY_RISK_GATE_OK"].astype(str).str.lower().isin(
            ["true", "1", "1.0"])
    if int(ok.sum()) == 0:
        return None
    rr = _num_col(rec, "RR_NOW_TP1")
    rr = (rr.fillna(0.0).clip(0.0, 3.0) if rr is not None
          else pd.Series(1.0, index=rec.index))
    return (a * rr).where(ok)


def alpha_live_line(report: Optional[dict], h: int = 5) -> str:
    """배치 로그·화면용 한 줄 요약 (없으면 빈 문자열)."""
    if not report or not report.get("ok"):
        return ""
    blk = (report.get("horizons") or {}).get(f"h{h}")
    if not blk:
        return ""
    ic_t = (blk.get("ic_t") or {}).get("t")
    # [v63] 헤드라인은 **엔진이 실제로 고르는 규칙**(퍼널+알파×RR) 기준으로 낸다.
    #   예전에는 전체 풀을 알파 점수만으로 정렬한 1등을 "1위"라고 적었는데,
    #   그건 엔진이 추천하지 않는 종목이다(v63 주석 참고). 재구성이 안 되는
    #   구 배치만 있으면 raw 기준으로 표기하되 그 사실을 밝힌다.
    g1 = blk.get("gated_top1")
    t1 = blk.get("top1") or {}
    head = (f"알파 실전 {blk['n_days']}일(h={h}): IC {blk.get('ic_mean'):+.3f}"
            + (f" t={ic_t:+.2f}" if ic_t is not None else "")
            + f" · 양수일 {blk.get('ic_positive_days')}/{blk['n_days']}")
    if g1:
        return (head
                + f" · 엔진 1위 {g1.get('mean_pct', 0):+.2f}%"
                + f"(초과 {g1.get('excess_mean_pct', 0):+.2f}%p"
                + f" · 스톱체결 {100*g1.get('stop_rate', 0):.0f}%"
                + f" · {g1.get('n_days', 0)}일)")
    return (head
            + f" · 알파점수 단독 1위 {t1.get('mean_pct', 0):+.2f}%"
            + f"(초과 {t1.get('excess_mean_pct', 0):+.2f}%p"
            + f" · 스톱체결 {100*t1.get('stop_rate', 0):.0f}%)"
            + " · 엔진 픽 기준 재구성 불가(구 배치)")
