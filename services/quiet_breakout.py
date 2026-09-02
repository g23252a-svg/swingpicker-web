# -*- coding: utf-8 -*-
"""quiet_breakout.py — 조용한 종목의 거래량 각성 레인 [v73]

## 무엇인가

현행 엔진이 **보지 않는 구간**(거래대금 순위 601~1200위)에서,
**거래량이 갑자기 붙은** 종목을 고르는 별도 레인이다.

`PRODUCTION_BUY` 도 켈리 사이징도 건드리지 않는다. 기존 추천은 그대로 두고
이 레인의 결과를 **따로 기록·표시**한다. 근거가 아무리 좋아도 엔진의 뿌리
(`top_n=600`)를 하루아침에 뒤집는 것은 위험하고, 실전 표본을 쌓아 비교하는 것이
먼저이기 때문이다.

## 왜 이 구간인가 (실측, `docs/PREDICTIVE_POWER_20260827.md`)

거래대금 순위와 5일 수익은 **역방향**이다. IC(부호 반전):
```
IS   +0.108  양수일  84.4%  HAC p<0.0001  분기 4/4
OOS  +0.181  양수일 100.0%  HAC p<0.0001  분기 4/4
```
그런데 이 효과는 **현행 1~600 안에서는 작동하지 않는다**
(IS -0.010 p=0.57 / OOS +0.065 — 부호 불일치). 601~1200 구간에서만 산다
(IS +0.081 / OOS +0.128, 둘 다 p<0.0001).

## 왜 거래량 급증인가

상위600에서는 거래량 급증이 **늦은 관심**이라 되돌린다(엔진이 지는 이유).
조용한 구간에서는 **이른 관심**이다. 60개 랭커를 훑어 세 구간 전부에서
재현된 것은 이것뿐이었고, 무작위 대조군은 정상적으로 0 근처였다.

**이 모듈의 명세 그대로** (밴드 601~1200 · vol_ratio ≥ 1.5 · 거래일 ≥ 15/20 ·
동결 배제 · N=5 · 5일 보유 · -8% 손절 · 비용 0.51% 차감), 생존편향 없는 105일:
```
IS   64일  절대 +2.37%  비용후 +1.86%(p=0.137)  초과 +2.87%(p=0.0065)  분기 3/4
OOS  39일  절대 +5.52%  비용후 +5.01%(p=0.032)  초과 +5.09%(p=0.0105)  분기 3/4
전체 103일 절대 +3.56%  비용후 +3.05%(p=0.0117) 초과 +3.71%(p=0.0004)  분기 3/4
                                                        drop-top2 +2.20%
워밍업 351일(생존편향 있음) 비용후 +0.89%(p=0.042)
```
네 구간 전부 비용차감 후 양수. 초과수익은 IS·OOS·전체 모두 p<0.05.
문턱 민감도도 평평하다 — vol_ratio 1.0/1.2/1.5/2.0/3.0 에서 +2.78~3.96%.

실행성: 픽 거래대금 중위 14.6억(하위10% 4.6억) — 100만원 매수는 0.07% 수준.
현행 엔진이 이미 보고 있는 비율은 **7.6%** 뿐이다. 소멸종목 노출 271종목 중 1종목.

## 믿지 말아야 할 것

- 절대수익의 p값은 IS 에서 0.137 로 약하다. 강한 것은 초과수익 쪽이다.
- 양수일이 53%다. 소수 대박이 끌어올리는 구조다.
  다만 drop-top-2 는 +2.20%로 상위 2일을 빼도 남는다.
- 60개 피처를 훑어 찾은 것이다. 세 구간 재현·대조군·N 단조성·경제적 서사가
  뒷받침하지만, 다중검정 위험이 완전히 사라진 것은 아니다.
- **그래서 병렬 레인이다.** 실전 20일 이상 쌓인 뒤 현행과 비교해서 판단한다.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("quiet_breakout")

CACHE_NAME = "quiet_breakout_latest.json"

# ── 검정으로 고정된 명세 (SSOT) ────────────────────────────────
#: 거래대금 순위 밴드. 이 밖에서는 효과가 없다.
RANK_LO, RANK_HI = 600, 1200
#: 랭킹 기준 — 당일 거래량 / 20일 평균 거래량.
VOL_WINDOW = 20
#: 화면에 올릴 종목 수. N=5 가 절대·비용후 모두 가장 강했다.
TOP_N = 5
#: 거래량 각성의 최소 배수. 이보다 낮으면 '각성'이라 부르지 않는다.
MIN_VOL_RATIO = 1.5
#: 가격이 얼어붙은 종목 배제 — 정지는 레인지·비율 지표를 통째로 왜곡한다.
FROZEN_MIN_LIVE_DAYS = 18       # 최근 20일 중 살아있어야 하는 날
FROZEN_MIN_VOL_PCT = 0.5        # 20일 수익률 표준편차(%) 하한
#: 20일 창 안에서 실제로 거래된 날의 최소 수.
#: 거래량이 0이던 종목이 오늘 거래되면 vol_ratio 가 20배로 튀는데, 그건
#: '조용한 각성'이 아니라 **거래 재개**다. 성질이 전혀 다르므로 배제한다.
MIN_ACTIVE_DAYS = 15
#: 지표를 믿을 수 있는 최소 이력.
MIN_HISTORY = 25
#: 검정된 보유·청산 조건 (표시용 — 이 레인은 주문을 내지 않는다).
HOLD_DAYS = 5
STOP_PCT = -0.08


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def prepare(ohlcv: pd.DataFrame) -> Optional[dict]:
    """한 종목의 일봉에서 레인 판정에 필요한 값만 뽑는다.

    거래정지(시가/고가/저가가 0)는 그날을 '거래 없음'으로 본다 — 저가 0을
    그대로 쓰면 손절·레인지 지표가 전부 망가진다.
    """
    if ohlcv is None or len(ohlcv) < MIN_HISTORY:
        return None
    d = ohlcv.tail(120).copy()
    for c in ("시가", "고가", "저가", "종가", "거래량"):
        if c not in d.columns:
            return None
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["종가"])
    d = d[d["종가"] > 0]
    if len(d) < MIN_HISTORY:
        return None
    halted = ~((d["시가"] > 0) & (d["고가"] > 0) & (d["저가"] > 0))
    live20 = int((~halted).tail(VOL_WINDOW).sum())
    ret = d["종가"].pct_change()
    vol20 = float(ret.tail(VOL_WINDOW).std() * 100) if len(ret) >= VOL_WINDOW else float("nan")
    v = d["거래량"]
    tail = v.tail(VOL_WINDOW)
    active = int((tail > 0).sum())
    vmean = float(tail.mean())
    vtoday = float(v.iloc[-1])
    if not np.isfinite(vmean) or vmean <= 0 or active < MIN_ACTIVE_DAYS:
        return None
    close = float(d["종가"].iloc[-1])
    return dict(
        vol_ratio=vtoday / vmean,
        live20=live20,
        active20=active,
        vol_20d=vol20,
        frozen=bool(halted.iloc[-1]) or live20 < FROZEN_MIN_LIVE_DAYS
        or (np.isfinite(vol20) and vol20 < FROZEN_MIN_VOL_PCT),
        종가=close,
        ret_1d=float(ret.iloc[-1] * 100) if len(ret) else float("nan"),
        ret_5d=float((close / d["종가"].iloc[-6] - 1) * 100) if len(d) > 6 else float("nan"),
        ret_60d=float((close / d["종가"].iloc[-61] - 1) * 100) if len(d) > 61 else float("nan"),
        tv_eok=close * vtoday / 1e8,
    )


def build(rank_table: pd.DataFrame,
          ohlcv_map: Dict[str, pd.DataFrame],
          name_map: Optional[Dict[str, str]] = None,
          top_n: int = TOP_N) -> dict:
    """레인을 만든다.

    rank_table: 종목코드 · 거래대금(원) — **전 시장 순위를 매길 수 있는 표**.
                현행 배치의 `top_df`(상위 600)로는 안 된다. 601~1200위가 필요하다.
    ohlcv_map:  종목코드 → 일봉 DataFrame (시가/고가/저가/종가/거래량)
    """
    if rank_table is None or rank_table.empty:
        return dict(ok=False, reason="rank_table 없음")
    t = rank_table.copy()
    if "종목코드" not in t.columns:
        return dict(ok=False, reason="종목코드 컬럼 없음")
    tv_col = next((c for c in t.columns if "거래대금" in str(c)), None)
    if tv_col is None:
        return dict(ok=False, reason="거래대금 컬럼 없음")
    t["종목코드"] = t["종목코드"].astype(str).str.zfill(6)
    t["_tv"] = pd.to_numeric(t[tv_col], errors="coerce")
    t = t.dropna(subset=["_tv"])
    t["rank"] = t["_tv"].rank(ascending=False, method="min")
    band = t[(t["rank"] > RANK_LO) & (t["rank"] <= RANK_HI)]
    if band.empty:
        return dict(ok=False, reason=f"밴드 비어있음 (표에 {len(t)}종목 — "
                                     f"{RANK_HI}위까지 담긴 표가 필요하다)",
                    universe_size=int(len(t)))

    rows: List[dict] = []
    skipped = {"no_ohlcv": 0, "short_history": 0, "frozen": 0, "low_vol_ratio": 0}
    for code in band["종목코드"]:
        px = ohlcv_map.get(code)
        if px is None:
            skipped["no_ohlcv"] += 1
            continue
        f = prepare(px)
        if f is None:
            skipped["short_history"] += 1
            continue
        if f["frozen"]:
            skipped["frozen"] += 1
            continue
        if f["vol_ratio"] < MIN_VOL_RATIO:
            skipped["low_vol_ratio"] += 1
            continue
        rows.append(dict(종목코드=code,
                         종목명=(name_map or {}).get(code, ""),
                         거래대금순위=int(band.loc[band["종목코드"] == code, "rank"].iloc[0]),
                         **f))
    if not rows:
        return dict(ok=False, reason="조건을 통과한 종목 없음",
                    universe_size=int(len(t)), band_size=int(len(band)), skipped=skipped)
    df = pd.DataFrame(rows).sort_values("vol_ratio", ascending=False)
    picks = df.head(top_n)
    return dict(
        ok=True,
        universe_size=int(len(t)),
        band_size=int(len(band)),
        candidates=int(len(df)),
        skipped=skipped,
        top_n=int(top_n),
        picks=[{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in r.items()} for r in picks.to_dict("records")],
        spec=dict(rank_band=[RANK_LO, RANK_HI], min_vol_ratio=MIN_VOL_RATIO,
                  vol_window=VOL_WINDOW, hold_days=HOLD_DAYS, stop_pct=STOP_PCT),
        caveat=("검증 중인 병렬 레인이다. PRODUCTION_BUY·사이징과 무관하며 "
                "주문을 내지 않는다. 절대수익 근거는 105일 p=0.0089(N=5)로 "
                "강하지 않고 양수일이 53%다 — 실전 20일 이상 쌓인 뒤 판단한다."),
    )


def save(data_dir: str, ymd: str, report: dict) -> bool:
    """레인 결과 저장. **성공본(ok=True)을 실패본으로 덮지 않는다.**

    [v80] 2026-08-29(토) 새벽 지연 발화한 cron이 8/28 산출물을 다시 만들며
    KRX 야간차단 폴백(359종목 표)으로 레인을 돌려 ok=False를 냈고, 그것이
    #675의 정상 결과(71후보·5픽)를 덮었다 — 실전 검증 1일차가 사라졌다.
    git 이력(d3f2df39)에서 복구했다. 같은 날짜에 이미 ok=True가 있으면
    실패본은 버린다. 반환값: 실제로 썼는지.
    """
    try:
        os.makedirs(data_dir, exist_ok=True)
        day_p = os.path.join(data_dir, f"quiet_breakout_{ymd}.json")
        if not report.get("ok") and os.path.exists(day_p):
            try:
                with open(day_p, encoding="utf-8") as f:
                    if json.load(f).get("ok"):
                        logger.warning("[v73] %s 성공본이 있어 실패본을 버린다: %s",
                                       ymd, report.get("reason"))
                        return False
            except Exception:
                pass          # 기존 파일이 깨졌으면 덮어도 잃을 게 없다
        for n in (f"quiet_breakout_{ymd}.json", CACHE_NAME):
            with open(os.path.join(data_dir, n), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=1)
        return True
    except OSError as e:
        logger.warning("[v73] 레인 저장 실패 (계속): %s", e)
        return False


def load(data_dir: str, ymd: Optional[str] = None) -> Optional[dict]:
    for n in ([f"quiet_breakout_{ymd}.json"] if ymd else []) + [CACHE_NAME]:
        p = os.path.join(data_dir, n)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("[v73] 레인 읽기 실패 %s: %s", p, e)
    return None


def line(report: Optional[dict]) -> str:
    """한 줄 요약 — 화면용."""
    if not report:
        return "[검증중] 조용한 각성 레인 — 데이터 없음"
    if not report.get("ok"):
        return f"[검증중] 조용한 각성 레인 — 산출 못 함 ({report.get('reason', '?')})"
    n = len(report.get("picks", []))
    return (f"[검증중] 조용한 각성 {n}종목 — 거래대금 {RANK_LO+1}~{RANK_HI}위 중 "
            f"거래량 {MIN_VOL_RATIO:.1f}배 이상 급증 (후보 {report.get('candidates', 0)}종목). "
            f"현행 추천과 별개이며 주문 대상이 아니다.")


# ══════════════════════════════════════════════════════════════
#  배치 진입점
# ══════════════════════════════════════════════════════════════
"""■ 왜 별도 시세 캐시를 쓰는가

`collector.prepare_ohlcv_data` 는 편하지만 **`ohlcv_cache_YYYYMMDD.parquet` 를
갱신한다.** 거기에 601~1200위를 넣으면 그 캐시의 의미가 '상위 600'에서
'상위 1200'으로 바뀌고, 그걸 읽는 `pick_reliability` · `candidate_depth` ·
`alpha_live_report` 의 산출이 조용히 달라진다. 이 레인은 아무것도 바꾸지
않기로 했으므로 **자기 캐시**(`quiet_lane_ohlcv_YYYYMMDD.parquet`)를 쓴다.
"""

LANE_CACHE = "quiet_lane_ohlcv_{ymd}.parquet"
#: 한 번 실행에서 새로 받을 최대 종목 수. 배치 시간이 폭증하지 않게 막는다.
MAX_FETCH = 700
LANE_COLS = ["시가", "고가", "저가", "종가", "거래량"]


def _lane_cache_path(data_dir: str, ymd: str) -> str:
    return os.path.join(data_dir, LANE_CACHE.format(ymd=ymd))


def _load_lane_cache(data_dir: str, ymd: str) -> Dict[str, pd.DataFrame]:
    p = _lane_cache_path(data_dir, ymd)
    if not os.path.exists(p):
        return {}
    try:
        d = pd.read_parquet(p)
    except Exception as e:
        logger.warning("[v73] 레인 캐시 읽기 실패 %s: %s", p, e)
        return {}
    if "종목코드" not in d.columns:
        return {}
    return {c: g.drop(columns=["종목코드"]) for c, g in d.groupby("종목코드")}


def _save_lane_cache(data_dir: str, ymd: str, m: Dict[str, pd.DataFrame]) -> None:
    if not m:
        return
    try:
        parts = []
        for c, g in m.items():
            gg = g.copy()
            gg["종목코드"] = c
            parts.append(gg)
        pd.concat(parts).to_parquet(_lane_cache_path(data_dir, ymd))
    except Exception as e:
        logger.warning("[v73] 레인 캐시 저장 실패 (계속): %s", e)


def run_batch(trade_ymd: str, start_ymd: str, end_ymd: str,
              ohlcv_map: Optional[Dict[str, pd.DataFrame]] = None,
              name_map: Optional[Dict[str, str]] = None,
              data_dir: str = "data",
              max_fetch: int = MAX_FETCH) -> dict:
    """배치에서 호출한다. **어떤 실패도 배치를 깨뜨리지 않는다.**

    현행 파이프라인이 쓰는 것은 아무것도 바꾸지 않는다 — 순위표를 넓게 한 번
    더 받고, 부족한 시세만 레인 전용 캐시로 채운 뒤, JSON 하나를 남긴다.
    """
    try:
        from collector import pick_top_by_trading_value, safe_ohlcv_by_date
    except Exception as e:
        logger.warning("[v73] collector 참조 실패 — 레인 건너뜀: %s", e)
        return dict(ok=False, reason=f"collector import: {type(e).__name__}")
    try:
        rank = pick_top_by_trading_value(trade_ymd, RANK_HI)
    except Exception as e:
        logger.warning("[v73] 순위표 수집 실패 — 레인 건너뜀: %s", e)
        return dict(ok=False, reason=f"rank fetch: {type(e).__name__}: {e}"[:200])
    if rank is None or rank.empty:
        return dict(ok=False, reason="순위표 비어있음")

    t = rank.copy()
    t["종목코드"] = t["종목코드"].astype(str).str.zfill(6)
    tv_col = next((c for c in t.columns if "거래대금" in str(c)), None)
    if tv_col is None:
        return dict(ok=False, reason="거래대금 컬럼 없음")
    t["_r"] = pd.to_numeric(t[tv_col], errors="coerce").rank(ascending=False, method="min")
    need = t.loc[(t["_r"] > RANK_LO) & (t["_r"] <= RANK_HI), "종목코드"].tolist()

    have: Dict[str, pd.DataFrame] = dict(_load_lane_cache(data_dir, trade_ymd))
    for c in need:
        if c in have:
            continue
        src = (ohlcv_map or {}).get(c)
        if src is not None and not src.empty:
            have[c] = src[[x for x in LANE_COLS if x in src.columns]].copy()
    missing = [c for c in need if c not in have]
    fetched = failed = 0
    for c in missing[:max_fetch]:
        try:
            d = safe_ohlcv_by_date(start_ymd, end_ymd, c)
        except Exception:
            failed += 1
            continue
        if d is None or d.empty:
            failed += 1
            continue
        have[c] = d[[x for x in LANE_COLS if x in d.columns]].copy()
        fetched += 1
    _save_lane_cache(data_dir, trade_ymd, {c: have[c] for c in need if c in have})

    rep = build(t.rename(columns={tv_col: "거래대금(원)"}), have, name_map=name_map)
    rep.update(trade_ymd=trade_ymd, fetched=fetched, fetch_failed=failed,
               still_missing=max(0, len(missing) - max_fetch))
    save(data_dir, trade_ymd, rep)
    logger.info("[v73] %s", line(rep))
    return rep
