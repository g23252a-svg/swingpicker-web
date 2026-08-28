# -*- coding: utf-8 -*-
"""portfolio_budget.py — 계좌 전체가 감당할 수 있는 만큼만 배분한다 [v76]

## 왜

켈리는 종목마다 **따로** 분율을 낸다. 합계를 보는 사람이 없었다.

`kelly_calibrator.size_positions`는 행별로
`f_safe = clip(f_raw * 0.5, 0, max_allocation=0.25)` 를 구하고
`켈리_금액(원) = total_capital * f_safe` 를 찍는다. 종목당 상한 25%는 있는데
**하루 합계 상한도, 보유 중인 이전 포지션과의 합산도 없다.**

배치 121일 실측(계좌 1,000만원 가정):

| | 값 |
|---|---|
| 하루 배분 합계 중위 | 14.5% |
| 하루 배분 합계 최대 | **183.8%** (2026-04-24, 28종목) |
| 합계가 100%를 넘은 날 | 19/121일 (16%) |
| 5일 보유 중첩 노출 중위 | **133.0%** |
| 5일 보유 중첩 노출 최대 | **721.6%** |
| 중첩 300% 초과일 | 28/117일 |

보유기간이 5일이므로 t일에 산 포지션은 t+4일까지 살아 있다. 그런데 t+1,
t+2… 배치는 이전 포지션을 모르는 채로 계좌 전액을 다시 배분한다. 그래서
화면의 `켈리_수량`은 **집행 불가능한 숫자**다 — 현금 계좌로는 살 수 없다.

실현수익으로 환산하면(66일, 사이징 행 566건):

| 규칙 | 누적 | 최악일 | MDD |
|---|---|---|---|
| 현행 (예산 제약 없음) | -17,088,377원 | -1,193,981원 | **-173.1%** |
| 총노출 100% 상한 | -5,795,703원 | -738,314원 | -59.2% |
| 총노출 100% + 종목 10% | -5,778,555원 | -723,062원 | -59.2% |
| 총노출 60% + 종목 10% | -3,576,764원 | -433,837원 | -36.6% |

MDD -173%는 **파산**이다. 계좌가 0을 지나 빚으로 간다.

## 왜 '남은 예산 선착순'이 아니라 날짜별 트랜치인가

총노출 상한만 두면 한 날이 계좌를 다 먹고 다음 나흘을 굶긴다. 실측에서
2026-08-25 배치 하나가 100%를 가져가 이후 4개 배치의 배분이 0이 됐다.
그래서 진입일 코호트마다 `BOOK_CAP / HOLD_DAYS`씩만 배정한다:

| 규칙 | 누적 | 최악일 | MDD | 굶은날 | 집행일 |
|---|---|---|---|---|---|
| 현행 | -17,088,377원 | -1,193,981원 | -173.1% | 0 | 66 |
| 총100% 선착순 | -5,795,703원 | -738,314원 | -59.2% | **8** | 58 |
| 총100%+종목10% 선착순 | -5,778,555원 | -723,062원 | -59.2% | **8** | 58 |
| **+ 일일 트랜치 20%** | **-5,230,692원** | **-160,000원** | **-52.6%** | **0** | **66** |

트랜치가 모든 축에서 낫다. 최악일이 -723,062원 → -160,000원으로 4.5배
줄고, 굶는 날이 사라지고, 66일 전부 집행 가능해진다.

## 무엇을 고치고 무엇을 고치지 않는가

이 레이어는 **예산만** 본다. 어느 종목을 살지는 건드리지 않는다.

- 추천 목록·`PRODUCTION_BUY`·`TOP_PICK`·순위 **무변경**
- 분율을 **줄이기만** 한다. 어떤 행도 원래보다 커지지 않는다(`scale <= 1`).
- 원래 값은 `KELLY_FRACTION_RAW`·`켈리_금액_RAW(원)`에 보존한다.

전략의 기대수익이 음수인 건 이 모듈이 고칠 수 있는 문제가 아니다
(원인은 편입 게이트의 역예측성 — `docs/PREDICTIVE_POWER_20260827.md` §8).
예산 레이어가 하는 일은 **음수 기대값에 레버리지를 얹지 않는 것**뿐이다.
그것만으로 MDD가 -173% → -59%로 줄고, 계좌가 살아남는다.

## 손절이 -8%를 지키지 못하는 부분

같이 실측했다. 손절 발동 100,012건 중 **9.5%(9,488건)는 다음 시가가 이미
손절가 아래**였다. 그날은 -8%에 못 판다.

| | 값 |
|---|---|
| 갭 관통 시 실제 체결 평균 | **-10.59%** |
| 중위 | -9.47% |
| -15% 이하 체결 | 628건 |
| -20% 이하 체결 | 223건 |
| 최악 | -79.17% |

평균수익에 주는 영향은 -0.088%p로 작다. 문제는 **꼬리**다. 종목당 25%를
넣은 상태에서 -79% 체결이 한 번 나오면 계좌의 20%가 하루에 사라진다.
종목당 상한을 낮춰야 하는 이유는 기대값이 아니라 이 꼬리다.

`worst_case_pct()` 는 이 실측을 반영한 "실제로 각오해야 하는 손실"을 준다.
"-8% 손절"이라는 화면 문구는 시장이 지켜주는 약속이 아니다.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio_budget")

#: 동시 보유 총노출 상한 (계좌 배수). 현금 계좌는 1.0을 넘을 수 없다.
BOOK_CAP = 1.00
#: 종목당 상한. 켈리 자체 캡(0.25)보다 낮다 — 근거는 갭 관통 꼬리(위 표).
NAME_CAP = 0.10
#: 보유기간 — 실현수익 SSOT와 같은 값이어야 한다.
HOLD_DAYS = 5
#: 진입일 코호트 하나가 가져갈 수 있는 최대 — 한 날이 계좌를 다 먹지 않게 한다.
DAY_CAP = BOOK_CAP / HOLD_DAYS
#: 이 이전 배치는 컬럼 규격이 달라 노출 합산에 섞지 않는다.
HISTORY_FROM = "20260101"
#: 1주도 못 사는 행을 떨어뜨리고 재분배하는 반복 횟수 상한.
MAX_REDISTRIBUTE_PASSES = 12

#: 손절 발동 시 다음 시가가 이미 손절가 아래였던 비율 (실측 100,012건).
GAP_THROUGH_RATE = 0.095
#: 그때의 평균 체결 수익률 (실측).
GAP_FILL_MEAN = -0.1059

COL_SCALE = "BUDGET_SCALE"          # 이 배치에 적용된 축소 배수 (<=1)
COL_USED = "BUDGET_USED_PCT"        # 이전 포지션이 쓰고 있는 노출 (계좌 %)
COL_ROOM = "BUDGET_ROOM_PCT"        # 남은 예산 (계좌 %)
COL_REASON = "BUDGET_REASON"        # 사람이 읽는 사유
COL_RAW_F = "KELLY_FRACTION_RAW"    # 축소 전 분율
COL_RAW_AMT = "켈리_금액_RAW(원)"    # 축소 전 금액

_F = "KELLY_FRACTION"
_AMT = "켈리_금액(원)"
_QTY = "켈리_수량"
_BUY = "추천매수가"


def _num(s, default=0.0) -> pd.Series:
    """숫자화 — `pd.to_numeric(None)`은 스칼라 NaN을 준다. Series로 감싼다."""
    return pd.to_numeric(pd.Series(s), errors="coerce").fillna(default)


def worst_case_pct(stop_pct: float = -0.08,
                   gap_rate: float = GAP_THROUGH_RATE,
                   gap_fill: float = GAP_FILL_MEAN) -> Dict[str, float]:
    """'-8% 손절'이 실제로 뜻하는 손실.

    Returns
    -------
    dict
        ``nominal`` 선언 손절폭, ``expected`` 갭 확률을 반영한 기대 체결,
        ``gap_rate`` 갭 관통 확률, ``gap_fill`` 관통 시 평균 체결.
    """
    sp = float(stop_pct)
    gr = float(np.clip(gap_rate, 0.0, 1.0))
    gf = min(float(gap_fill), sp)          # 관통이 손절보다 좋을 수는 없다
    return {"nominal": sp,
            "expected": sp * (1.0 - gr) + gf * gr,
            "gap_rate": gr,
            "gap_fill": gf}


def _sessions_before(data_dir: str, trade_ymd: str, n: int) -> List[str]:
    """`trade_ymd` 직전 배치 n개의 날짜. 파일이 있는 날만 센다."""
    out: List[str] = []
    for f in sorted(glob.glob(os.path.join(data_dir, "recommend_2*.csv"))):
        m = re.search(r"(\d{8})", os.path.basename(f))
        if not m:
            continue
        y = m.group(1)
        if HISTORY_FROM <= y < str(trade_ymd):
            out.append(y)
    return out[-n:] if n > 0 else []


def _day_exposure(data_dir: str, ymd: str) -> Optional[Tuple[float, bool]]:
    """그날 배치가 요구한 노출과 '이미 예산이 적용된 배치인가'."""
    p = os.path.join(data_dir, f"recommend_{ymd}.csv")
    try:
        d = pd.read_csv(p, usecols=lambda c: c in (_F, _AMT, COL_SCALE))
    except Exception as e:
        logger.warning("[v76] %s 노출 읽기 실패: %s", p, e)
        return None
    budgeted = COL_SCALE in d.columns
    if _F in d.columns:
        return float(_num(d[_F]).clip(lower=0).sum()), budgeted
    if _AMT in d.columns:
        # 구 배치에 분율이 없으면 금액/기본 계좌로 되돌린다.
        return float(_num(d[_AMT]).clip(lower=0).sum()) / 10_000_000.0, budgeted
    return None


def live_exposure(data_dir: str, trade_ymd: str,
                  hold_days: int = HOLD_DAYS,
                  book_cap: float = BOOK_CAP,
                  day_cap: Optional[float] = DAY_CAP) -> Tuple[float, Dict[str, float]]:
    """아직 보유 중인 이전 포지션이 쓰고 있는 노출 (계좌 배수).

    직전 ``hold_days - 1``개 배치의 분율을 더한다. 손절로 이미 청산된 건은
    빼지 않는다 — 예산은 **보수적으로** 잡는다.

    예산 레이어 도입 이전 배치는 집행 불가능한 분율(합계 최대 184%)을 그대로
    적고 있다. 그 숫자를 '보유 중'으로 세면 신규 배분이 영구히 0이 된다.
    그건 실제로 보유한 적 없는 포지션이다. 그래서 ``BUDGET_SCALE`` 컬럼이
    없는(=예산 미적용) 과거 배치는 **그날 예산 레이어가 허용했을 만큼으로
    되돌려** 센다. 워밍업을 두고 순차 재생해 값이 수렴하게 한다.
    """
    hold = max(0, int(hold_days) - 1)
    if hold <= 0:
        return 0.0, {}
    window = _sessions_before(data_dir, trade_ymd, hold + 2 * int(hold_days))
    committed: Dict[str, float] = {}
    for i, y in enumerate(window):
        got = _day_exposure(data_dir, y)
        if got is None:
            continue
        raw, budgeted = got
        if budgeted:
            committed[y] = raw                    # 이미 예산이 적용된 값
            continue
        prior = sum(committed.get(w, 0.0) for w in window[max(0, i - hold):i])
        room = max(0.0, float(book_cap) - prior)
        if day_cap is not None:
            room = min(room, float(day_cap))      # 재생도 같은 규칙을 쓴다
        committed[y] = min(raw, room)             # 그날 허용됐을 만큼
    tail = window[-hold:]
    detail = {y: committed[y] for y in tail if y in committed}
    return float(sum(detail.values())), detail


def apply(df: pd.DataFrame, data_dir: str, trade_ymd: str,
          book_cap: float = BOOK_CAP, name_cap: float = NAME_CAP,
          total_capital: int = 10_000_000,
          hold_days: int = HOLD_DAYS,
          day_cap: Optional[float] = DAY_CAP) -> Tuple[pd.DataFrame, dict]:
    """예산 상한 안으로 사이징을 축소한다. **늘리지 않는다.**

    Returns
    -------
    (df, info)
        ``info`` 는 ``{used, room, raw, scale, capped, n, applied}``.
    """
    info = {"used": 0.0, "room": float(book_cap), "raw": 0.0, "scale": 1.0,
            "capped": 0, "n": 0, "sub_one": 0, "applied": False}
    if df is None or len(df) == 0 or _F not in df.columns:
        return df, info

    out = df.copy()
    f0 = _num(out[_F]).clip(lower=0.0)
    if COL_RAW_F not in out.columns:
        out[COL_RAW_F] = f0.values
    if _AMT in out.columns and COL_RAW_AMT not in out.columns:
        out[COL_RAW_AMT] = _num(out[_AMT]).astype(int).values

    # ① 종목당 상한
    f1 = f0.clip(upper=float(name_cap))
    info["capped"] = int((f0 > float(name_cap)).sum())

    # ② 보유 중인 이전 포지션이 먹은 예산
    used, _detail = live_exposure(data_dir, trade_ymd, hold_days=hold_days,
                                  book_cap=book_cap, day_cap=day_cap)
    room = max(0.0, float(book_cap) - used)
    if day_cap is not None:
        # 한 진입일 코호트가 가져갈 수 있는 몫으로 한 번 더 제한한다.
        room = min(room, float(day_cap))

    # ③ 남은 예산 안으로 비례 축소 — 축소만 한다
    raw = float(f1.sum())
    scale = 1.0 if raw <= room else (room / raw if raw > 0 else 0.0)
    scale = float(min(1.0, max(0.0, scale)))
    f2 = f1 * scale

    # 반올림은 상한을 넘길 수 있다(0.066666 → 0.0667 × 3 = 0.2001). 예산은 내림.
    f2 = pd.Series(np.floor(f2.values * 1e4) / 1e4, index=f2.index)
    out[_F] = f2.values
    amt = (float(total_capital) * f2).astype(int)
    if _AMT in out.columns:
        out[_AMT] = amt.values
    # ④ 집행 가능하게 만든다 — 트랜치를 26종목에 뿌리면 1종목당 배분이
    #    1주 값보다 작아진다(실측 2026-08-26: 사이징 26건 중 19건이 0주).
    #    0주는 배분이 아니라 낭비다. 못 사는 행을 하나씩 떨어뜨리고 그 몫을
    #    남은 행에 **원래 비율 그대로** 재분배한다. 순위를 새로 만들지 않는다 —
    #    사이징 대상 안에서 엔진의 순위에는 정보가 없다(HAC p=0.73).
    #    떨어뜨릴 행은 비중이 가장 작은 쪽, 동률이면 비싼 쪽부터.
    _sub_one = np.zeros(len(out), dtype=bool)
    if _QTY in out.columns and _BUY in out.columns:
        buy = _num(out[_BUY]).values
        w0 = f1.values
        live = w0 > 0
        alloc = np.zeros(len(out))
        for _ in range(int(live.sum()) + 1):
            w = np.where(live, w0, 0.0)
            tw = w.sum()
            if tw <= 0:
                alloc = np.zeros(len(out))
                break
            # 떨어진 행의 몫을 남은 행에 얹지 않는다 — 그건 엔진이 요청한 것보다
            # 크게 거는 새 베팅이고, 사이징 대상 안에서 순위에는 정보가 없다.
            # w0 는 이미 종목상한이 걸린 값이므로 이 한 줄이 상한·요구액을 함께 막는다.
            alloc = np.minimum(w / tw * room, w0)
            alloc = np.floor(alloc * 1e4) / 1e4
            a_won = (float(total_capital) * alloc).astype(np.int64)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(buy > 0, a_won / np.where(buy > 0, buy, 1), 0.0)
            bad = live & (np.floor(q) < 1)
            if not bad.any():
                break
            # 가장 약한 행 하나만 떨군다 — 한 번에 다 버리면 과소집행이 된다.
            order = np.lexsort((-buy, w))[::1]
            drop = next((i for i in order if bad[i]), None)
            if drop is None:
                break
            live[drop] = False
        f2 = pd.Series(np.where(live, alloc, 0.0), index=f2.index)
        out[_F] = f2.values
        amt = (float(total_capital) * f2).astype(int)
        if _AMT in out.columns:
            out[_AMT] = amt.values
        with np.errstate(divide="ignore", invalid="ignore"):
            qty = np.where(buy > 0, amt.values / np.where(buy > 0, buy, 1), 0)
        out[_QTY] = np.floor(qty).astype(int)      # 내림 — 예산을 넘지 않는다
        # 그래도 못 사는 행이 남으면(전 종목이 트랜치보다 비쌀 때) 사실대로 적는다.
        _sub_one = (out[_QTY].values <= 0) & (f2.values > 0)

    if scale >= 1.0 and info["capped"] == 0:
        reason = "예산 여유 — 축소 없음"
    elif room <= 0:
        reason = (f"보유 포지션이 계좌를 이미 다 쓰고 있음 "
                  f"(사용 {used * 100:.0f}% / 상한 {book_cap * 100:.0f}%) — 신규 배분 0")
    else:
        bits = []
        if info["capped"]:
            bits.append(f"종목당 {name_cap * 100:.0f}% 상한 적용 {info['capped']}건")
        if scale < 1.0:
            _lim = ("일일 배정" if day_cap is not None and room <= float(day_cap) + 1e-12
                    else f"총노출 {book_cap * 100:.0f}%")
            bits.append(f"{_lim} {room * 100:.0f}% 맞추려 {scale * 100:.0f}%로 축소")
        reason = " · ".join(bits)

    if _sub_one.any():
        _txt = np.array([
            f"예산 배분 {int(a):,}원 < 1주 {int(bp):,}원 — 이번 회차 수량 0"
            for a, bp in zip(amt.values, _num(out[_BUY]).values if _BUY in out.columns
                             else np.zeros(len(out)))], dtype=object)
        _r = np.where(_sub_one, _txt, np.asarray(reason, dtype=object))
        out[COL_REASON] = _r
        if "KELLY_ZERO_REASON" in out.columns:
            out["KELLY_ZERO_REASON"] = np.where(
                _sub_one, _txt, out["KELLY_ZERO_REASON"].astype(object).values)
    else:
        out[COL_REASON] = reason
    info["sub_one"] = int(_sub_one.sum())
    out[COL_SCALE] = round(scale, 4)
    out[COL_USED] = round(used * 100.0, 1)
    out[COL_ROOM] = round(room * 100.0, 1)
    info.update(used=used, room=room, raw=raw, scale=scale,
                n=int((f2 > 0).sum()), applied=True)
    return out, info


def line(info: dict) -> str:
    """로그 한 줄."""
    if not info or not info.get("applied"):
        return ""
    return (f"예산 — 보유 {info['used'] * 100:.0f}% · 여유 {info['room'] * 100:.0f}% · "
            f"신규요구 {info['raw'] * 100:.0f}% → 배수 {info['scale']:.2f} "
            f"(사이징 {info['n']}건, 종목상한 적용 {info['capped']}건"
            + (f", 1주 미만 {info['sub_one']}건" if info.get('sub_one') else "") + ")")


def stop_worst_line(stop_pct: float = -0.08) -> str:
    """화면용 — 손절이 실제로 뜻하는 손실."""
    w = worst_case_pct(stop_pct)
    return (f"손절 {w['nominal'] * 100:.0f}%는 약속이 아니다 — "
            f"{w['gap_rate'] * 100:.0f}%는 시가가 이미 손절가 아래로 열려 "
            f"평균 {w['gap_fill'] * 100:.1f}%에 체결됐다 "
            f"(각오할 평균 {w['expected'] * 100:.1f}%)")
