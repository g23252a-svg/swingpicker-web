# -*- coding: utf-8 -*-
"""[v55] 공식 매수 픽의 실현 분포 — 화면이 확신을 주는 만큼 근거가 있는가.

■ [v57] 이 docstring의 숫자는 **이 모듈의 산출이 아니다** — 출처를 구분한다
  아래 '패널 76일' 수치는 v55 연구 하니스(워크포워드 패널에서 원천 피처로
  퍼널을 다시 계산)의 결과다. 반면 이 모듈은 과거 recommend_*.csv에 **저장된**
  판정 컬럼을 읽어 재생하므로, 그 컬럼이 없는 날(ALPHA_ENTRY_OK는 116일 중
  16일에만 존재)에는 같은 퍼널을 재현할 수 없다. 실제로 v56 시점의 이 모듈
  산출은 `{"ok": false, "재현된 공식 매수 픽 없음"}`이었는데 docstring은
  '픽 15일 · 평균 +6.20%'를 사실처럼 적고 있었다 — v55.4/v56에서 잡은 것과
  같은 계열(문서 ≠ 실제 산출)이다. 두 숫자를 섞어 읽지 말 것:
    · 패널 수치 = 정책 판단용(연구 하니스, 원천 피처 재계산)
    · 이 모듈 산출 = 화면 표시용(저장 컬럼 재생, 계약·컬럼 가용성에 종속)

■ 왜 만드는가
  v54까지 화면은 픽 하나에 대해 '품질 87 · 손익비 5.06 · 승률 39%'처럼
  **확신을 주는 숫자만** 보여줬다. 그런데 v55에서 프로덕션 퍼널을 그대로
  재현해 측정하니 분포가 이렇다(패널 76일 · 2026-04~07, 실현 정의는 아래와 동일):

      픽 15일 · 평균 +6.20% · 중위 **-2.70%** · 승률 46.7% · sd 25.18 · 왜도 2.10
      상위 2건 +78.6% / +52.0%  →  이 2건을 빼면 평균 **-2.89%**
      하위 5건은 전부 -8.0%(스톱 체결)

  즉 이 전략은 승률로 버는 게 아니라 **드문 대박으로 버는 우측 꼬리 전략**이다.
  평균만 보여주면 '이기는 전략'으로 읽히고, 사용자는 한 픽에 큰돈을 넣는다.
  중위값이 음수라는 사실을 같이 보여주는 것이 정직하다.

■ 무엇을 계산하는가
  과거 recommend_*.csv에 **현재 계약(apply_recommendation_quality_guard)을
  재적용**해 그날의 공식 매수를 다시 뽑고, 실제 OHLCV로 실현수익을 계산한다.
  과거 CSV의 PRODUCTION_BUY 값을 그대로 쓰지 않는 이유: 그 값은 당시 계약
  (v32~v53)의 산물이고, 지금 화면이 약속하는 것은 v54 계약이다. 계약이 바뀌면
  통계도 다시 계산해야 한다 — 115개 파일 중 PRODUCTION_BUY 컬럼이 있는 것은
  18개뿐이기도 하다.

  두 청산 규칙을 함께 낸다. 화면이 두 가지를 동시에 지시하고 있기 때문이다:
    hold  : 진입 t+1 시가 · -8% 장중 스톱 · t+5 종가 청산   (목표가까지 보유)
    disc  : exit_plan 표시 규율 — 2일내 +3% 1/4익절 · +5% 본전스톱 ·
            +10% 1차익절 · -7% 손절                        (화면 권고 규율)
  [v57] 이 규율의 파라미터는 '승률과 기대값 둘 다'를 목표로 재조정됐다.
  격자탐색(트리거 2~6% × 익절비율 25/33/50% × BE 유무 = 30조합, BH-FDR 보정):
    +2%·절반(구값) 승률 41.4%→65.5% 이지만 평균 +1.26%→+0.55% (기준의 43%)
    +3%·1/4(채택)  승률 41.4%→**59.8%** · 평균 **+1.12%(89% 보존)** ·
                   MDD -33.5%→-16.5% · p_BH=0.0001 · ΔIS +15.9 / ΔOOS +27.8
                   top-5(n=144) 재현: 승률 +16.7%p · 평균 88% 보존
  즉 승률 +18%p를 얻고 기대값은 11%만 내놓는다. 두 규칙을 계속 함께 내보내
  '보유 vs 규율'의 트레이드오프를 사용자가 직접 볼 수 있게 한다.

■ 이 숫자를 어떻게 쓰면 안 되는가
  표본이 얇다(수십 일). '중위가 음수니 이 전략은 손해'도, '평균 +6%니 좋다'도
  단정할 수 없다. 그래서 출력에 n·신뢰구간·이상치 의존도를 항상 함께 담는다.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("pick_reliability")

CACHE_NAME = "pick_reliability_latest.json"
_UNION_HL = "ohlcv_union_hl.parquet"
_HL_COLS = ["시가", "고가", "저가", "종가"]

HOLD_DAYS = 5
STOP_PCT_HOLD = -0.08      # 실현수익 정의(연구 기준과 동일)
STOP_PCT_DISC = -0.07      # exit_plan 표시 권고
# [v57] exit_plan의 권고값과 같아야 한다 — 화면이 지시하는 것을 그대로 측정한다.
#   격자탐색 결과 +3% 트리거·1/4 익절이 승률(+18%p)과 기대값(89% 보존)을 함께
#   잡는 지점이었다(top-3 n=87 p_BH=0.0001 · top-5 n=144 재현).
#   테스트(test_v57_balanced_exit.py)가 exit_plan과의 동일성을 고정한다.
FAST_TP_PCT = 0.03
FAST_TP_DAYS = 2
FAST_TP_FRACTION = 0.25
BE_TRIGGER_PCT = 0.05      # 본전스톱 전환
TP_QUICK_PCT = 0.10        # 1차 익절
MIN_DAYS = 8               # 이보다 적으면 통계로 내보내지 않는다


# ─────────────────────────────────────────────────────────
#  OHLCV 합집합 (고가·저가 포함) — 생존편향 방지 + 증분 갱신
# ─────────────────────────────────────────────────────────
def _build_hl_union(data_dir: str) -> Optional[pd.DataFrame]:
    """일자별 캐시의 합집합. v50과 같은 이유로 합집합이어야 한다.

    각 ohlcv_cache_YYYYMMDD.parquet은 '그날의 유니버스'만 담으므로 최신
    파일 하나만 읽으면 그 사이 탈락한 종목이 사라진다(생존편향).
    """
    files = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.parquet")))
    files = [f for f in files if "latest" not in os.path.basename(f)]
    if not files:
        return None
    path = os.path.join(data_dir, _UNION_HL)
    have = None
    if os.path.exists(path):
        try:
            have = pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"[v55] HL 합집합 재생성 (읽기 실패: {e})")
            have = None
    todo = files if have is None else files[-3:]   # 증분: 최근 3일만 병합
    parts = [] if have is None else [have]
    for f in todo:
        ymd = os.path.basename(f).replace("ohlcv_cache_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f)
        except Exception as e:
            logger.warning(f"[v55] 캐시 스킵 {ymd}: {e}")
            continue
        if "종목코드" not in d.columns:
            continue
        keep = ["종목코드"] + [c for c in _HL_COLS if c in d.columns]
        d = d[keep].copy()
        d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
        if isinstance(d.index, pd.DatetimeIndex):
            d["Date"] = d.index
        else:
            try:
                d["Date"] = pd.to_datetime(ymd, format="%Y%m%d")
            except ValueError:
                continue
        parts.append(d)
    if not parts:
        return None
    big = pd.concat(parts, ignore_index=True).dropna(subset=["Date", "종목코드"])
    big = big.drop_duplicates(subset=["종목코드", "Date"], keep="last")
    big = big.sort_values(["종목코드", "Date"]).reset_index(drop=True)
    try:
        big.to_parquet(path, index=False)
    except Exception as e:
        logger.warning(f"[v55] HL 합집합 저장 실패 (계속): {e}")
    return big


# ─────────────────────────────────────────────────────────
#  실현수익 시뮬레이션
# ─────────────────────────────────────────────────────────
def _simulate(fut: pd.DataFrame, rule: str) -> Optional[float]:
    """fut: 추천일 다음 영업일부터의 OHLC. 진입은 첫 행 시가."""
    if fut is None or len(fut) < 2:
        return None
    entry = float(fut.iloc[0]["시가"])
    if not np.isfinite(entry) or entry <= 0:
        return None
    days = fut.iloc[0:HOLD_DAYS]
    stop = entry * (1 + (STOP_PCT_HOLD if rule == "hold" else STOP_PCT_DISC))
    realized, weight, half_done = 0.0, 1.0, False
    for i in range(len(days)):
        hi = float(days.iloc[i].get("고가", np.nan))
        lo = float(days.iloc[i].get("저가", np.nan))
        if not np.isfinite(hi) or not np.isfinite(lo):
            continue
        # 보수: 같은 날 스톱과 익절이 모두 닿으면 스톱 우선
        if lo <= stop:
            return realized + weight * (stop / entry - 1)
        if rule == "disc":
            if (not half_done) and i < FAST_TP_DAYS and hi >= entry * (1 + FAST_TP_PCT):
                realized += FAST_TP_FRACTION * FAST_TP_PCT
                weight, half_done = 1.0 - FAST_TP_FRACTION, True
                stop = max(stop, entry)
            if hi >= entry * (1 + BE_TRIGGER_PCT):
                stop = max(stop, entry)
            if hi >= entry * (1 + TP_QUICK_PCT):
                return realized + weight * TP_QUICK_PCT
    close = float(days.iloc[-1]["종가"])
    if not np.isfinite(close):
        return None
    return realized + weight * (close / entry - 1)


def _stats(vals: list) -> dict:
    s = pd.Series([v for v in vals if v is not None and np.isfinite(v)])
    if len(s) == 0:
        return {}
    top2 = float(s.nlargest(min(2, len(s))).sum())
    ex2 = s.nsmallest(max(len(s) - 2, 1)).mean()
    out = {
        "n": int(len(s)),
        "mean_pct": round(float(s.mean()) * 100, 2),
        "median_pct": round(float(s.median()) * 100, 2),
        "win_rate_pct": round(float((s > 0).mean()) * 100, 1),
        "sd_pct": round(float(s.std(ddof=1)) * 100, 2) if len(s) > 1 else None,
        "best_pct": round(float(s.max()) * 100, 2),
        "worst_pct": round(float(s.min()) * 100, 2),
        "top2_share_pct": (round(100 * top2 / float(s.sum()), 0)
                           if abs(float(s.sum())) > 1e-9 else None),
        "mean_ex_top2_pct": round(float(ex2) * 100, 2),
    }
    # 부트스트랩 신뢰구간 — 표본이 얇다는 사실을 숫자로 남긴다
    if len(s) >= MIN_DAYS:
        rng = np.random.default_rng(20260806)
        bs = [float(rng.choice(s.values, len(s), replace=True).mean()) for _ in range(2000)]
        out["mean_ci95_pct"] = [round(float(np.percentile(bs, 2.5)) * 100, 2),
                                round(float(np.percentile(bs, 97.5)) * 100, 2)]
    return out


def compute_pick_reliability(data_dir: str = "data", max_days: int = 200) -> dict:
    """과거 공식 매수 픽을 현재 계약으로 재현해 실현 분포를 계산."""
    from services.recommendation_quality import (
        POLICY_VERSION, apply_recommendation_quality_guard, production_buy_mask,
    )

    hl = _build_hl_union(data_dir)
    if hl is None or hl.empty:
        return {"ok": False, "reason": "OHLCV 캐시 없음"}
    by_code = {c: g.sort_values("Date").reset_index(drop=True)
               for c, g in hl.groupby("종목코드")}
    # [v65] 휴장일 배치는 직전 세션 가격으로 돌아간 복사본이다. 그 스냅샷의
    #   픽은 직전 영업일 픽과 같은 종목이므로, 측정일로 세면 같은 픽이 두 번
    #   집계된다(전수 7일: 20260302·0501·0505·0525·0603·0717·0817).
    #   OHLCV 캐시의 날짜 집합이 실제 세션 달력이므로 그것으로 판별한다.
    sessions = set(pd.to_datetime(hl["Date"]).dt.strftime("%Y%m%d"))
    non_session = []

    files = sorted(glob.glob(os.path.join(data_dir, "recommend_2*.csv")))[-max_days:]
    rows, skipped = [], 0
    for f in files:
        ymd = os.path.basename(f)[-12:-4]
        try:
            asof = pd.to_datetime(ymd, format="%Y%m%d")
        except ValueError:
            continue
        if ymd not in sessions:
            non_session.append(ymd)
            continue
        try:
            rec = pd.read_csv(f, encoding="utf-8-sig", dtype={"종목코드": str},
                              low_memory=False)
        except Exception as e:
            logger.warning(f"[v55] CSV 스킵 {ymd}: {e}")
            skipped += 1
            continue
        if "종목코드" not in rec.columns or len(rec) == 0:
            continue
        rec["종목코드"] = rec["종목코드"].astype(str).str.zfill(6)
        try:
            # 현재 계약을 재적용 — 과거 PRODUCTION_BUY 값은 당시 계약의 산물이다
            guarded = apply_recommendation_quality_guard(
                rec.drop(columns=["PRODUCTION_BUY"], errors="ignore"))
        except Exception as e:
            logger.warning(f"[v55] 가드 재적용 실패 {ymd}: {e}")
            skipped += 1
            continue
        picks = guarded[production_buy_mask(guarded)]
        for _, r in picks.iterrows():
            code = str(r["종목코드"]).zfill(6)
            g = by_code.get(code)
            if g is None:
                continue
            fut = g[g["Date"] > asof].head(HOLD_DAYS + 1)
            rows.append({
                "date": ymd, "code": code,
                "name": str(r.get("종목명", "")),
                "hold": _simulate(fut, "hold"),
                "disc": _simulate(fut, "disc"),
            })

    if not rows:
        # [v55] 표본 0은 '데이터 없음'이 아니라 그 자체가 사실이다 —
        # 2026-07~08 실측: 코스피가 MA20 대비 최대 -20% 이탈한 폭락으로
        # risk_off 하드블록이 최근 16영업일 중 15일 전 종목 신규진입을 차단했다.
        # 그 기간에는 공식 매수가 나올 수 없었으므로 표본도 없다. 화면이
        # '통계 없음'이 아니라 '차단되어 표본이 없다'고 말해야 한다.
        return {"ok": False, "reason": "기간 내 재현된 공식 매수 픽 없음",
                "files_scanned": len(files),
                "non_session_days_skipped": len(non_session),
                "policy_version": POLICY_VERSION,
                "block_note": "risk_off 등 하드블록으로 픽이 없었던 기간일 수 있음"}
    df = pd.DataFrame(rows)
    mature = df[df["hold"].notna()]
    out = {
        "ok": True,
        "policy_version": POLICY_VERSION,
        "asof": df["date"].max(),
        "files_scanned": len(files),
        "files_skipped": skipped,
        "pick_days": int(df["date"].nunique()),
        "non_session_days_skipped": len(non_session),
        "matured_picks": int(len(mature)),
        "horizon_days": HOLD_DAYS,
        "hold": _stats(df["hold"].tolist()),
        "disc": _stats(df["disc"].tolist()),
        "definition": {
            "hold": f"진입 t+1 시가 · {STOP_PCT_HOLD*100:.0f}% 장중 스톱 · t+{HOLD_DAYS} 종가",
            "disc": (f"exit_plan 표시 규율 — {FAST_TP_DAYS}일내 "
                     f"+{FAST_TP_PCT*100:.0f}% {FAST_TP_FRACTION:.0%}익절 · "
                     f"+{BE_TRIGGER_PCT*100:.0f}% 본전스톱 · +{TP_QUICK_PCT*100:.0f}% 1차익절 · "
                     f"{STOP_PCT_DISC*100:.0f}% 손절"),
        },
        "caveat": ("표본이 얇다. 평균은 소수 대박에 크게 의존하므로 "
                   "중위값·상위2건 기여도·신뢰구간을 함께 볼 것."),
    }
    return out


def save_pick_reliability(data_dir: str = "data", max_days: int = 200) -> dict:
    res = compute_pick_reliability(data_dir, max_days=max_days)
    try:
        with open(os.path.join(data_dir, CACHE_NAME), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[v55] 신뢰도 캐시 저장 실패: {e}")
    return res


def load_pick_reliability(data_dir: str = "data") -> Optional[dict]:
    path = os.path.join(data_dir, CACHE_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            res = json.load(f)
    except Exception as e:
        logger.warning(f"[v55] 신뢰도 캐시 읽기 실패: {e}")
        return None
    return res if res.get("ok") else None


def reliability_line(res: Optional[dict]) -> str:
    """화면 한 줄 요약 — 확신 대신 분포를 말한다."""
    if not res or not res.get("ok"):
        return ""
    h = res.get("hold") or {}
    n = h.get("n")
    if not n or n < MIN_DAYS:
        return (f"공식 매수 실현 표본 {n or 0}건 — 통계로 말하기엔 아직 적습니다"
                if n else "")
    parts = [f"과거 공식 매수 {n}건 실측(목표가까지 보유 기준)",
             f"중위 {h['median_pct']:+.1f}%",
             f"평균 {h['mean_pct']:+.1f}%",
             f"승률 {h['win_rate_pct']:.0f}%"]
    if h.get("top2_share_pct") is not None and h["top2_share_pct"] >= 50:
        parts.append(f"평균의 {h['top2_share_pct']:.0f}%가 상위 2건에서 나옴")
    return " · ".join(parts)


if __name__ == "__main__":   # pragma: no cover
    import pprint
    logging.basicConfig(level=logging.INFO)
    pprint.pprint(save_pick_reliability("data"))
    print("\n" + reliability_line(load_pick_reliability("data")))
