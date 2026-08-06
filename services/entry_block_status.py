# -*- coding: utf-8 -*-
"""[v55] 신규진입이 막힌 이유를 정량으로 — '왜 추천이 없냐'의 사실적 답.

■ 무엇이 실제로 일어났나 (2026-07~08 실측)
  사용자는 수 주간 "왜 추천이 한 번도 없냐"고 물었고, v45~v54는 라벨·UI·사유
  배선을 고쳤다. 그런데 v55에서 프로덕션 CSV를 직접 파보니 원인은 표시가 아니었다:

      최근 16영업일 중 15일이 NEW_ENTRY_BLOCKED = 전 종목
      사유: "risk_off: 추천손절 유지 + 신규진입 차단"

  코스피가 MA20 대비 최대 **-20.1%**(7/29) 이탈한 진짜 폭락이었고,
  `momentum_lane.compute_market_risk_off`가 설계대로 전면 차단한 것이다.
  8/5 단 하루 이탈이 -1.13%로 좁혀져 risk_off가 풀렸고, 그날 TOP_PICK 20개가
  처음 나왔다(사용자가 "드디어 첫 추천종목"이라고 한 그 날).

  즉 제품이 고장난 게 아니라 방어가 작동했다. 문제는 **화면이 그 사실을
  수치로 말하지 않은 것**이다. v34가 risk_off 스트립을 넣었지만 '얼마나 깊은
  차단이고 언제 풀리는지'가 없어서 사용자에게는 '추천 없음'과 구별되지 않았다.

■ 이 모듈이 하는 일
  차단 상태를 숫자로 만든다: 현재 이탈%, 해제선까지 남은 거리, 연속 차단일수,
  최근 60영업일 차단 비율, 마지막으로 풀린 날. 계산은 kospi_daily 하나로 끝난다
  (프로덕션 판정과 같은 함수를 쓰므로 화면과 엔진이 어긋나지 않는다).

■ 차단이 옳은가에 대해서는 단정하지 않는다
  v55 실측: 패널 내 차단일(6일, 전부 OOS)의 반사실 픽은 오히려 +3.15%
  (승률 66.7%)로 허용일 +2.43%보다 나빴다는 증거가 없다
  (Welch -0.72%p t=-0.20 p=0.85 · 순열 p=0.60 · 블록부트 CI [-6.49,+6.59]).
  그러나 깊은 폭락 구간(이탈 -8~-20%, 7/28~8/4)은 아직 선행수익이 없어
  평가 자체가 불가능하다. v49 기록대로 이 축은 창 길이에 따라 부호가 뒤집힌
  이력이 있다(105일 창 +3.07% t=+4.05 → 397일 창 -1.37%).
  따라서 차단 규칙은 손대지 않고, 상태만 정직하게 노출한다.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("entry_block_status")

# momentum_lane과 같은 값이어야 한다 — 다르면 화면과 엔진이 어긋난다.
# 테스트(test_v55_entry_block.py)가 동일성을 검사한다.
MA_WINDOW = 20
MA_SLOPE_LOOKBACK = 5
DEVIATION_FLOOR = -0.03


def _load_kospi(data_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(data_dir, "kospi_daily.csv")
    if not os.path.exists(path):
        return None
    try:
        k = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        logger.warning(f"[v55] kospi_daily 읽기 실패: {e}")
        return None
    if "date" not in k.columns or "close" not in k.columns:
        return None
    k = k[["date", "close"]].copy()
    k["date"] = pd.to_datetime(k["date"].astype(str).str.replace("-", "").str[:8],
                               format="%Y%m%d", errors="coerce")
    k["close"] = pd.to_numeric(k["close"], errors="coerce")
    k = k.dropna().sort_values("date").reset_index(drop=True)
    return k if len(k) >= MA_WINDOW // 2 + MA_SLOPE_LOOKBACK else None


def compute_entry_block_status(data_dir: str = "data") -> dict:
    """차단 상태를 수치로. 실패해도 화면을 막지 않도록 항상 dict를 돌려준다."""
    out = {
        "ok": False, "risk_off": None, "deviation_pct": None,
        "unlock_gap_pct": None, "streak_days": None, "block_rate_60d_pct": None,
        "last_open_date": None, "close": None, "ma20": None,
        "ma20_falling": None, "reason": "",
    }
    k = _load_kospi(data_dir)
    if k is None:
        out["reason"] = "코스피 일봉 데이터 부족 — 차단 상태를 계산할 수 없음"
        return out
    k["ma"] = k["close"].rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    k["ma_prev"] = k["ma"].shift(MA_SLOPE_LOOKBACK)
    k["dev"] = k["close"] / k["ma"] - 1.0
    k["risk_off"] = ((k["close"] < k["ma"]) & (k["ma"] < k["ma_prev"])
                     & (k["dev"] <= DEVIATION_FLOOR))
    last = k.iloc[-1]
    if not np.isfinite(last["ma"]) or last["ma"] <= 0:
        out["reason"] = "MA20 미형성 — 차단 없음으로 취급"
        return out

    ro = bool(last["risk_off"])
    dev = float(last["dev"])
    # 해제선: 이탈이 floor보다 얕아지면(또는 MA20 위로) 풀린다
    unlock_gap = (DEVIATION_FLOOR - dev) * 100.0   # 양수면 그만큼 더 올라야 함
    flags = k["risk_off"].astype(bool)
    streak = 0
    for v in reversed(flags.tolist()):
        if v:
            streak += 1
        else:
            break
    last60 = flags.tail(60)
    open_days = k.loc[~flags, "date"]
    out.update({
        "ok": True,
        "risk_off": ro,
        "close": round(float(last["close"]), 1),
        "ma20": round(float(last["ma"]), 1),
        "ma20_falling": bool(last["ma"] < last["ma_prev"]) if np.isfinite(last["ma_prev"]) else None,
        "deviation_pct": round(dev * 100, 2),
        "unlock_gap_pct": round(max(unlock_gap, 0.0), 2) if ro else 0.0,
        "streak_days": int(streak),
        "block_rate_60d_pct": round(float(last60.mean()) * 100, 0) if len(last60) else None,
        "last_open_date": (open_days.iloc[-1].strftime("%Y-%m-%d")
                           if len(open_days) else None),
        "asof": last["date"].strftime("%Y-%m-%d"),
    })
    out["reason"] = block_reason_line(out)
    return out


def block_reason_line(st: Optional[dict]) -> str:
    """사용자용 한 줄 — 추상적 경고 대신 숫자와 해제 조건."""
    if not st or not st.get("ok"):
        return ""
    if not st.get("risk_off"):
        return (f"신규진입 차단 없음 (코스피 {st['close']:,.0f} · "
                f"20일선 대비 {st['deviation_pct']:+.1f}%)")
    bits = [f"폭락 방어로 전 종목 신규진입 차단 {st['streak_days']}일 연속",
            f"코스피 {st['close']:,.0f} · 20일선 {st['ma20']:,.0f} 대비 "
            f"{st['deviation_pct']:+.1f}%"]
    if st.get("unlock_gap_pct") is not None:
        bits.append(f"해제선(-3%)까지 {st['unlock_gap_pct']:.1f}%p 남음")
    if st.get("block_rate_60d_pct") is not None:
        bits.append(f"최근 60일 중 {st['block_rate_60d_pct']:.0f}% 차단")
    if st.get("last_open_date"):
        bits.append(f"마지막 해제 {st['last_open_date']}")
    return " · ".join(bits)
