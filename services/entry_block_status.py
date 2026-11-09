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


_RECO_RE = None  # 지연 컴파일


def _latest_batch_csv(data_dir: str, batch_ymd: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """[v86] 화면이 보는 배치 CSV — batch_ymd가 있으면 그 날, 없으면 가장 최근 recommend_YYYYMMDD."""
    import re
    global _RECO_RE
    if _RECO_RE is None:
        _RECO_RE = re.compile(r"^recommend_(\d{8})\.csv$")
    try:
        names = os.listdir(data_dir)
    except OSError as e:
        logger.warning(f"[v86] {data_dir} 목록 실패: {e}")
        return None, None
    days = sorted(m.group(1) for m in (_RECO_RE.match(n) for n in names) if m)
    if batch_ymd:
        b = str(batch_ymd).replace("-", "")[:8]
        return (os.path.join(data_dir, f"recommend_{b}.csv"), b) if b in days else (None, b)
    if not days:
        return None, None
    return os.path.join(data_dir, f"recommend_{days[-1]}.csv"), days[-1]


_STATE_COLS = ("NEW_ENTRY_BLOCKED", "STOP_OVERRIDE_REASON", "RISK_OFF_ASOF", "RISK_OFF_STALE")


def _truthy(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    t = s.astype(str).str.strip().str.lower()
    return t.isin({"1", "1.0", "true", "t", "yes", "y"})


def recorded_block_state(df: pd.DataFrame) -> Optional[dict]:
    """[v86] 배치가 실제로 기록한 risk_off 판정. 컬럼이 없으면 None(=재계산 필요).

    판정식은 components/decision_center._risk_off_state와 같다:
    STOP_OVERRIDE_REASON에 'risk_off'가 있고 NEW_ENTRY_BLOCKED가 절반 이상.
    """
    if df is None or len(df) == 0:
        return None
    if "NEW_ENTRY_BLOCKED" not in df.columns or "STOP_OVERRIDE_REASON" not in df.columns:
        return None
    n_blk = int(_truthy(df["NEW_ENTRY_BLOCKED"]).sum())
    has_ro = bool(df["STOP_OVERRIDE_REASON"].astype(str).str.contains("risk_off", case=False).any())
    st = {"risk_off": bool(has_ro and n_blk >= max(1, int(len(df) * 0.5))),
          "risk_off_asof": None, "stale": None}
    if "RISK_OFF_ASOF" in df.columns:
        v = df["RISK_OFF_ASOF"].dropna().astype(str).str.replace(r"\.0$", "", regex=True)
        v = v[v.str.fullmatch(r"\d{8}")]
        if len(v):
            st["risk_off_asof"] = str(v.mode().iloc[0])
    if "RISK_OFF_STALE" in df.columns:
        st["stale"] = bool(_truthy(df["RISK_OFF_STALE"]).any())
    return st


def _read_state_cols(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in _STATE_COLS,
                           dtype={"RISK_OFF_ASOF": str}, low_memory=False)
    except Exception as e:
        logger.warning(f"[v86] 배치 CSV 상태 컬럼 읽기 실패({path}): {e}")
        return None


def _signal_table(k: pd.DataFrame) -> pd.DataFrame:
    k = k.copy()
    k["ma"] = k["close"].rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    k["ma_prev"] = k["ma"].shift(MA_SLOPE_LOOKBACK)
    k["dev"] = k["close"] / k["ma"] - 1.0
    k["risk_off"] = ((k["close"] < k["ma"]) & (k["ma"] < k["ma_prev"])
                     & (k["dev"] <= DEVIATION_FLOOR))
    return k


def compute_entry_block_status(data_dir: str = "data", df: Optional[pd.DataFrame] = None,
                               batch_ymd: Optional[str] = None) -> dict:
    """차단 상태를 수치로. 실패해도 화면을 막지 않도록 항상 dict를 돌려준다.

    [v86] 차단 여부는 **배치 CSV에 기록된 판정**을 쓴다(재계산하지 않는다).
      예전엔 kospi_daily의 마지막 행으로 다시 계산했는데, 배치는 하루 묵은 종가로 판정했고
      kospi는 배치 뒤에 갱신돼 flip일마다 화면과 CSV가 어긋났다(10/28·10/30·11/02·11/03).
      · df(배치 진행 중의 df_out) 또는 batch_ymd/최신 recommend_YYYYMMDD.csv에서 읽는다.
      · 수치(이탈%·연속일 등)는 RISK_OFF_ASOF(판정에 쓴 KOSPI 날짜) 기준으로 계산한다.
      · RISK_OFF_ASOF가 없는 옛 CSV: 배치일 기준으로 계산하고, 기록된 판정과 다르면
        직전 세션(옛 배치의 1일 지연 판정)을 시도해 맞는 쪽 날짜를 쓴다(asof_inferred).
      · 상태 컬럼이 아예 없으면 배치일 기준 재계산, 배치 CSV도 없으면 kospi 최신 행(옛 동작).
    """
    out = {
        "ok": False, "risk_off": None, "deviation_pct": None,
        "unlock_gap_pct": None, "streak_days": None, "block_rate_60d_pct": None,
        "last_open_date": None, "close": None, "ma20": None,
        "ma20_falling": None, "reason": "",
        "source": None, "batch_ymd": None, "risk_off_asof": None, "stale": None,
    }
    # ── 배치가 기록한 판정 ──
    rec = None
    b_ymd = str(batch_ymd).replace("-", "")[:8] if batch_ymd else None
    if df is not None:
        rec = recorded_block_state(df)
    else:
        path, b = _latest_batch_csv(data_dir, b_ymd)
        b_ymd = b_ymd or b
        if path:
            rec = recorded_block_state(_read_state_cols(path))
    out["batch_ymd"] = b_ymd

    k = _load_kospi(data_dir)
    if k is None:
        if rec is not None:
            # 수치는 없어도 기록된 판정은 말한다
            out.update({"risk_off": rec["risk_off"], "source": "batch_csv",
                        "risk_off_asof": rec.get("risk_off_asof"), "stale": rec.get("stale")})
        out["reason"] = "코스피 일봉 데이터 부족 — 차단 상태를 계산할 수 없음"
        return out
    k = _signal_table(k)
    ymd = k["date"].dt.strftime("%Y%m%d")

    # ── 수치 기준일 결정 ──
    asof = None
    inferred = False
    if rec is not None and rec.get("risk_off_asof"):
        asof = rec["risk_off_asof"]
    elif b_ymd:
        upto = k[ymd <= b_ymd]
        if len(upto):
            asof = upto["date"].iloc[-1].strftime("%Y%m%d")
            if rec is not None and len(upto) >= 2 and bool(upto["risk_off"].iloc[-1]) != rec["risk_off"] \
                    and bool(upto["risk_off"].iloc[-2]) == rec["risk_off"]:
                asof = upto["date"].iloc[-2].strftime("%Y%m%d")
                inferred = True
    kk = k[ymd <= asof] if asof else k
    if kk.empty:
        kk = k
    last = kk.iloc[-1]
    if not np.isfinite(last["ma"]) or last["ma"] <= 0:
        out["reason"] = "MA20 미형성 — 차단 없음으로 취급"
        return out

    ro_calc = bool(last["risk_off"])
    ro = rec["risk_off"] if rec is not None else ro_calc
    dev = float(last["dev"])
    # 해제선: 이탈이 floor보다 얕아지면(또는 MA20 위로) 풀린다
    unlock_gap = (DEVIATION_FLOOR - dev) * 100.0   # 양수면 그만큼 더 올라야 함
    flags = kk["risk_off"].astype(bool)
    streak = 0
    for v in reversed(flags.tolist()):
        if v:
            streak += 1
        else:
            break
    if ro and streak == 0:
        streak = 1  # 기록된 차단인데 재계산이 다르면 최소 1일로 말한다
    last60 = flags.tail(60)
    open_days = kk.loc[~flags, "date"]
    kospi_last = k["date"].iloc[-1].strftime("%Y%m%d")
    out.update({
        "ok": True,
        "risk_off": bool(ro),
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
        "source": ("batch_csv" if rec is not None else
                   ("recompute_batch" if b_ymd else "recompute_latest")),
        "risk_off_asof": last["date"].strftime("%Y%m%d"),
        "asof_inferred": inferred,
        "stale": (rec.get("stale") if rec is not None and rec.get("stale") is not None
                  else (bool(b_ymd) and last["date"].strftime("%Y%m%d") < b_ymd)),
        "recompute_disagrees": bool(rec is not None and ro_calc != rec["risk_off"]),
        "kospi_newer_than_batch": bool(b_ymd and kospi_last > b_ymd),
    })
    out["reason"] = block_reason_line(out)
    return out


def block_reason_line(st: Optional[dict]) -> str:
    """사용자용 한 줄 — 추상적 경고 대신 숫자와 해제 조건."""
    if not st or not st.get("ok"):
        return ""
    # [v86] 판정에 쓴 코스피가 배치일보다 묵었으면 그 사실을 같이 말한다
    _stale_note = ""
    if st.get("stale") and st.get("asof"):
        _stale_note = f" · 판정 기준 코스피 {st['asof']} (배치일보다 이전 종가)"
    if not st.get("risk_off"):
        return (f"신규진입 차단 없음 (코스피 {st['close']:,.0f} · "
                f"20일선 대비 {st['deviation_pct']:+.1f}%){_stale_note}")
    bits = [f"폭락 방어로 전 종목 신규진입 차단 {st['streak_days']}일 연속",
            f"코스피 {st['close']:,.0f} · 20일선 {st['ma20']:,.0f} 대비 "
            f"{st['deviation_pct']:+.1f}%"]
    if st.get("unlock_gap_pct") is not None:
        bits.append(f"해제선(-3%)까지 {st['unlock_gap_pct']:.1f}%p 남음")
    if st.get("block_rate_60d_pct") is not None:
        bits.append(f"최근 60일 중 {st['block_rate_60d_pct']:.0f}% 차단")
    if st.get("last_open_date"):
        bits.append(f"마지막 해제 {st['last_open_date']}")
    return " · ".join(bits) + _stale_note
