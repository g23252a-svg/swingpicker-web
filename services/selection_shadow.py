# -*- coding: utf-8 -*-
"""selection_shadow.py — 선별 변형 3종의 실전 성적을 매일 나란히 쌓는다 [v83]

## 왜

v83 점검(docs/SELECTION_AUDIT_V83.md)은 고변동 차단 게이트를 실전에 넣었다.
게이트의 근거는 IS/OOS 양쪽에서 강건하지만, **랭킹 축**(알파×손익비 vs 저변동 최저)은
알파 시대 18일 표본으로는 갈리지 않는다. 그래서 랭킹은 바꾸지 않고, 대신 세 변형의
'그날 1등'을 매일 기록해 발견 이후 표본으로만 비교한다 — v73 레인·v78 루프와 같은
규율(코드는 승격하지 않는다, 20 거래일 뒤 사용자가 판정한다).

    legacy : v83 이전 엔진 — 고변동 게이트 없이 알파 문턱·저점추세 통과 후 알파×손익비 1등
    live   : 현행 공식픽 (PRODUCTION_BUY=1) — v83 게이트 포함
    lowvol : 알파 문턱·저점추세 없이 리스크가드·급등·변동성 게이트만 통과한 후보 중 ATR% 최저

실현수익은 pick_history._realized(SSOT: 진입 t+1 시가 · -8% 장중 손절 · t+5 종가)로 잰다.
이 모듈은 아무것도 막지 않는다 — PRODUCTION_BUY·켈리·목록 무변경.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from services.pick_history import _realized          # SSOT — 복제 금지
from services.winner_profile import _hac_p, _load_universe_ohlcv

logger = logging.getLogger("selection_shadow")

VARIANTS = ("legacy", "live", "lowvol")
LOG_NAME = "selection_shadow_log.parquet"
SUMMARY_NAME = "selection_shadow_latest.json"
TARGET_DAYS = 20
#: 게이트 실전 1일차 — 이 날짜 이전 기록은 검증 표본이 아니다.
LIVE_FROM = "20260914"
#: 켈리가 0주로 만드는 상태 — recommendation_quality._inactive_routes 와 같은 취지의 근사.
_INACTIVE_ROUTES = ("OVERHEAT", "CARRY", "NEUTRAL", "CASH", "EXIT", "WEAK")


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    return (pd.to_numeric(df[c], errors="coerce") if c in df.columns
            else pd.Series(np.nan, index=df.index))


def _flag(df: pd.DataFrame, c: str, default: bool) -> pd.Series:
    if c not in df.columns:
        return pd.Series(default, index=df.index)
    return _num(df, c).fillna(1 if default else 0).astype(bool)


def variant_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """세 변형의 후보 집합. 공통 바닥 = 리스크가드·전면차단 아님·급등 아님·손익비≥1.3·POC·상태."""
    risk = _flag(df, "ENTRY_RISK_GATE_OK", False)
    blocked = _flag(df, "NEW_ENTRY_BLOCKED", False)
    surge = _flag(df, "ALPHA_SURGE_OK", True)
    lt = _flag(df, "ALPHA_LT_OK", True)
    vol = _flag(df, "ALPHA_VOL_OK", True)
    bnp = _flag(df, "BUY_NOW_PASS", True)
    rr = _num(df, "RR_NOW_TP1")
    poc = _num(df, "POC_GAP")
    route = df["ROUTE"].astype(str).str.upper() if "ROUTE" in df.columns else pd.Series("", index=df.index)
    a = _num(df, "ALPHA_SCORE"); thr = _num(df, "ALPHA_ENTRY_THRESHOLD")
    base = (risk & ~blocked & surge & bnp & (rr >= 1.30).fillna(False)
            & (poc.isna() | (poc <= 20.0)) & ~route.isin(_INACTIVE_ROUTES))
    return {
        "legacy": base & lt & (a >= thr).fillna(False),
        "live": _flag(df, "PRODUCTION_BUY", False),
        "lowvol": base & vol & _num(df, "V23_ATR_Pct").notna(),
    }


def pick_today(df: pd.DataFrame, trade_ymd: str) -> List[dict]:
    """세 변형의 '오늘 1등' 한 행씩. 후보가 없으면 변형은 빠진다(현금)."""
    if df is None or df.empty or "종목코드" not in df.columns:
        return []
    m = variant_masks(df)
    a = _num(df, "ALPHA_SCORE").fillna(0.0)
    rr = _num(df, "RR_NOW_TP1").fillna(0.0).clip(0.0, 3.0)
    key = {"legacy": a * rr * 1_000 + a, "live": a * rr * 1_000 + a,
           "lowvol": -_num(df, "V23_ATR_Pct")}
    out = []
    for v in VARIANTS:
        sub = df[m[v]]
        if sub.empty:
            continue
        k = key[v].reindex(sub.index)
        if k.notna().sum() == 0:
            continue
        i = k.idxmax()
        r = df.loc[i]
        out.append({"ymd": trade_ymd, "variant": v,
                    "종목코드": str(r.get("종목코드", "")).zfill(6),
                    "종목명": str(r.get("종목명", "")),
                    "alpha": float(a.loc[i]), "rr": float(rr.loc[i]),
                    "atr_pct": float(_num(df, "V23_ATR_Pct").loc[i]) if "V23_ATR_Pct" in df else np.nan,
                    "vol_pctl": float(_num(df, "VOL_GATE_PCTL").loc[i]) if "VOL_GATE_PCTL" in df else np.nan})
    return out


def append_log(data_dir: str, rows: List[dict]) -> bool:
    """(ymd, variant) 단위 멱등 추가. 같은 날 재실행은 덮지 않는다."""
    if not rows:
        return False
    p = os.path.join(data_dir, LOG_NAME)
    new = pd.DataFrame(rows)
    if os.path.exists(p):
        try:
            old = pd.read_parquet(p)
        except Exception as e:
            logger.warning("[v83] 그림자 로그 읽기 실패 — 덮어쓰지 않음: %s", e)
            return False
        ymd = str(rows[0]["ymd"])
        if ((old["ymd"].astype(str) == ymd)).any():
            return False
        new = pd.concat([old, new], ignore_index=True)
    os.makedirs(data_dir, exist_ok=True)
    new.to_parquet(p, index=False)
    return True


def load_log(data_dir: str) -> pd.DataFrame:
    p = os.path.join(data_dir, LOG_NAME)
    if not os.path.exists(p):
        return pd.DataFrame(columns=["ymd", "variant", "종목코드", "종목명"])
    try:
        return pd.read_parquet(p)
    except Exception as e:
        logger.warning("[v83] 그림자 로그 읽기 실패: %s", e)
        return pd.DataFrame(columns=["ymd", "variant", "종목코드", "종목명"])


def build(data_dir: str) -> dict:
    """로그 전수 × SSOT 실현수익 → 변형별 요약 + 페어드 차이."""
    log = load_log(data_dir)
    log = log[log["ymd"].astype(str) >= LIVE_FROM] if len(log) else log
    px = _load_universe_ohlcv(data_dir)
    by_code: Dict[str, pd.DataFrame] = (
        {c: g.reset_index(drop=True) for c, g in px.groupby("종목코드")} if px is not None else {})
    rets = []
    for _, r in log.iterrows():
        g = by_code.get(str(r["종목코드"]).zfill(6))
        v = _realized(g, str(r["ymd"])) if g is not None else None
        rets.append(None if v is None else float(v))
    log = log.assign(ret=rets)
    out: dict = {"live_from": LIVE_FROM, "target_days": TARGET_DAYS,
                 "days_total": int(log["ymd"].nunique()) if len(log) else 0,
                 "variants": {}, "paired": {}}
    daily: Dict[str, pd.Series] = {}
    for v in VARIANTS:
        s = log[(log["variant"] == v) & log["ret"].notna()]
        d = s.groupby("ymd")["ret"].mean() * 100 if len(s) else pd.Series(dtype=float)
        daily[v] = d
        t, p = _hac_p(d.values.astype(float)) if len(d) else (np.nan, np.nan)
        out["variants"][v] = {
            "days": int(len(d)), "avg_ret_pct": (float(d.mean()) if len(d) else None),
            "win_rate": (float((s["ret"] > 0).mean()) if len(s) else None),
            "stop_rate": (float((s["ret"] <= -0.079).mean()) if len(s) else None),
            "hac_t": (None if not np.isfinite(t) else float(t)),
            "hac_p": (None if not np.isfinite(p) else float(p)),
        }
    for a, b in (("live", "legacy"), ("lowvol", "live")):
        dd = (daily[a] - daily[b]).dropna()
        t, p = _hac_p(dd.values.astype(float)) if len(dd) else (np.nan, np.nan)
        out["paired"][f"{a}-{b}"] = {
            "days": int(len(dd)), "diff_pct": (float(dd.mean()) if len(dd) else None),
            "positive_days": int((dd > 0).sum()) if len(dd) else 0,
            "hac_t": (None if not np.isfinite(t) else float(t)),
            "hac_p": (None if not np.isfinite(p) else float(p)),
        }
    n_live = out["variants"]["live"]["days"]
    out["verdict"] = ("표본 충족 — 판정 가능 (코드는 승격하지 않는다)"
                      if n_live >= TARGET_DAYS else "판정 전 — 20 거래일 후 사용자 판정")
    return out


def save(data_dir: str, summary: dict) -> None:
    try:
        with open(os.path.join(data_dir, SUMMARY_NAME), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    except OSError as e:
        logger.warning("[v83] 그림자 요약 저장 실패: %s", e)


def load(data_dir: str) -> Optional[dict]:
    p = os.path.join(data_dir, SUMMARY_NAME)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[v83] 그림자 요약 읽기 실패: %s", e)
        return None


def run_batch(df: pd.DataFrame, data_dir: str, trade_ymd: str) -> dict:
    """오늘 3변형 픽 기록 → 누적 성적 요약 저장. 반환: 요약(+ today)."""
    today = pick_today(df, trade_ymd)
    added = append_log(data_dir, today)
    s = build(data_dir)
    s["today"] = today
    s["added"] = bool(added)
    save(data_dir, s)
    return s


def line(s: Optional[dict]) -> str:
    if not s:
        return ""
    v = s.get("variants") or {}
    n = (v.get("live") or {}).get("days", 0)
    head = f"선별 그림자 3종 — 측정 {n}/{s.get('target_days', TARGET_DAYS)}일 (기록 {s.get('days_total', 0)}일)"
    today = s.get("today") or []
    tl = " · ".join(f"{t['variant']} {t['종목명']}" for t in today)
    if not n:
        return head + (f" · 오늘 {tl}" if tl else "") + " · 첫 5일 창이 아직 안 닫혔습니다"
    parts = []
    for k, lab in (("live", "현행(v83 게이트)"), ("legacy", "구엔진(게이트 없음)"), ("lowvol", "저변동 최저")):
        x = v.get(k) or {}
        if x.get("days"):
            parts.append(f"{lab} {x['avg_ret_pct']:+.2f}%/승률 {x['win_rate'] * 100:.0f}%/손절 {x['stop_rate'] * 100:.0f}%")
    pr = (s.get("paired") or {}).get("live-legacy") or {}
    tail = ""
    if pr.get("days"):
        tail = f" · 현행−구엔진 {pr['diff_pct']:+.2f}%p ({pr['positive_days']}/{pr['days']}일 우위)"
    return head + " · " + " · ".join(parts) + tail
