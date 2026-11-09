# -*- coding: utf-8 -*-
"""selection_shadow.py — 선별 변형의 실전 성적을 매일 나란히 쌓는다 [v83 · v84 · v86]

## 왜

v83 점검(docs/SELECTION_AUDIT_V83.md)은 고변동 차단 게이트를 실전에 넣었다.
게이트의 근거는 IS/OOS 양쪽에서 강건하지만, **랭킹 축**(알파×손익비 vs 저변동 최저)은
알파 시대 18일 표본으로는 갈리지 않는다. 그래서 랭킹은 바꾸지 않고, 대신 변형들의
'그날 1등'을 매일 기록해 발견 이후 표본으로만 비교한다 — v73 레인·v78 루프와 같은
규율(코드는 승격하지 않는다).

    legacy : v83 이전 엔진 — 고변동 게이트 없이 알파 문턱·저점추세 통과 후 알파×손익비 1등
    live   : 현행 공식픽 (PRODUCTION_BUY=1) — v83 게이트 포함
    lowvol : 알파 문턱·저점추세 없이 리스크가드·급등·변동성 게이트만 통과한 후보 중 ATR% 최저
    ext_lowtv : [v84] 캐시 ∪ 레인(≈1,200종목)에서 선등록 5특징(거래대금·20일수익·60일고점이격·
                RSI·거래량비) 분위 평균 최저 — 승자 프로파일 전방 로그가 가리키는 방향의 기록
    comp_lowvol : [v86] lowvol 풀에서 pct(ALPHA_SCORE)+pct(-ATR%) 최고 1종목
    no_alpha    : [v86] 현행 풀에서 알파 문턱·저점추세 조건만 뺀 풀(리스크가드·급등·v83 변동성·
                  품질가드·사이징 가능 ROUTE) — 현행 랭킹키(알파×clip(RR)×1000+품질점수) 1등
    thin_skip   : [v86] 현행과 같은 픽, 단 그날 TOP_PICK 수 ≤ 5 이면 현금

## [v86] 선등록 판정 (docs/SHADOW_PREREG_V86.md)

v86 점검(랭킹·손실 원인 연구, 독립 재현 완료)의 결론은 "프로덕션 랭킹·게이트는 그대로,
대신 후보를 고정 규칙 그림자로 새 표본에서만 판정"이다. 그래서:

  · 판정은 REGISTERED_FROM(20261109) **이후** 새 배치만 센다. 그 전 기록은 전부 '탐색'
    (lowvol·ext_lowtv 의 기존 18~22일 포함 — 그 표본이 가설을 만들었으므로 재사용 금지).
  · 후보 vs live 페어드, risk_off 아닌 정상일·양쪽 모두 픽이 기록된 날의 **처음 40일 고정창**
    (실현 여부로 창을 고르지 않는다 — 40일 전부 실현될 때까지 '측정 중'). 내려진 판정은 동결.
  · 승격 후보 = 40일 HAC(5) p < 0.05 · 차이 > 0 · 앞/뒤 20일 반기 부호 같음 ·
    후보 평균(매매일 0.25% 비용 차감) > 0. 하나라도 못 넘으면 기각. 40일 전은 '측정 중 n/40'.
  · risk_off 차단일에도 모든 변형의 반사실 픽을 blocked=True 로 기록한다 — 같은 날끼리
    비교하기 위해서다. 요약은 차단일/정상일을 따로 낸다. 판정은 정상일만.

실현수익은 pick_history._realized(SSOT: 진입 t+1 시가 · -8% 장중 손절 · t+5 종가)로 잰다.
이 모듈은 아무것도 막지 않는다 — PRODUCTION_BUY·켈리·목록·랭킹·게이트 문턱 무변경.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from services.pick_history import _realized          # SSOT — 복제 금지
from services.winner_profile import (_hac_p, _load_code_history, _load_universe_ohlcv,
                                     _realized_full_window)

logger = logging.getLogger("selection_shadow")

VARIANTS = ("legacy", "live", "lowvol", "ext_lowtv", "comp_lowvol", "no_alpha", "thin_skip")
#: [v86] 선등록 후보 — live 와 페어드로 판정한다. legacy 는 참고(구엔진)라 '탐색' 고정.
CANDIDATES = ("lowvol", "ext_lowtv", "comp_lowvol", "no_alpha", "thin_skip")
LOG_NAME = "selection_shadow_log.parquet"
SUMMARY_NAME = "selection_shadow_latest.json"
#: 게이트 실전 1일차 — 이 날짜 이전 기록은 표본이 아니다(탐색 집계에도 넣지 않는다).
LIVE_FROM = "20260914"
#: 켈리가 0주로 만드는 상태 — recommendation_quality._inactive_routes 와 같은 취지의 근사.
_INACTIVE_ROUTES = ("OVERHEAT", "CARRY", "NEUTRAL", "CASH", "EXIT", "WEAK")

# ── [v86] 선등록 판정 규칙 — 결과를 보고 바꾸면 이 실험은 무효다 ──────────────
PREREG_VERSION = "v86-20261109"
#: 판정 표본 시작 — v86 머지 후 첫 배치. 이 날 이전 기록은 '탐색'으로만 보고한다.
REGISTERED_FROM = "20261109"
#: 정상일(비 risk_off) 페어드 고정창 길이 — 처음 40일로만 판정한다(선택적 중단 방지).
PREREG_DAYS = 40
PREREG_ALPHA = 0.05
PREREG_HAC_LAG = 5
#: 왕복 거래비용(%) — 후보가 실제로 매매한 날에만 차감한다(현금일 0).
PREREG_COST_PCT = 0.25
#: thin_skip — 그날 TOP_PICK(알파 게이트 통과) 수가 이 값 이하면 현금.
THIN_POOL_MAX = 5
#: risk_off 판정 — 배치 전 행이 NEW_ENTRY_BLOCKED 면 시장 전체 차단일로 본다
#: (정상일에도 7월방어·회복방어·손절 오버라이드로 일부 행이 차단될 수 있어 '전부'로 가른다).
_DAY_BLOCK_FRAC = 0.99
STATUS_EXPLORE = "탐색"
STATUS_PROMOTE = "승격 후보"
STATUS_REJECT = "기각"

# ── [v84] 확장 유니버스 변형 'ext_lowtv' ────────────────────────────────────
# v78 승자 프로파일 전방 로그(선등록 v1-20260831, 2026-09-14 기준 20일)에서 부호가
# 안정적이고 p<0.05 인 특징 다섯 개 — 전부 '낮을수록 좋다':
#   tv_rank_pctl IC -0.216 (HAC t -17.0) · ret_20d -0.112 (t -5.5) · hi60_gap -0.143 (t -3.6)
#   rsi14 -0.062 (t -3.2) · vol_ratio -0.046 (t -2.2)
# 즉 '거래대금 하위 · 20일 하락 · 60일 고점에서 멀고 · RSI 낮고 · 거래량이 터지지 않은'
# 종목이 5일 뒤 더 낫다. 가장 센 축(거래대금)은 상위600 풀 **안**에서는 정보가 없고
# (풀 내 IC -0.001), 601~1200 레인 유니버스 대 풀의 차이에서 온다 — 8/27 편입 결함과
# 같은 방향이다. 그래서 이 변형은 배치 캐시 ∪ 레인 캐시(≈1,200종목) 전체에서 다섯
# 특징의 분위 평균이 가장 낮은 1종목을 고른다. 프로토콜상 1단계(40일)가 아직 안 찼으므로
# **승격이 아니라 기록**이다 — 다섯 특징과 부호를 여기 고정해(v1-20261006) 앞으로 쌓이는
# 표본으로만 판정한다. 결과를 보고 특징을 바꾸면 이 변형은 무효다.
#
# [v85] 주의 — 10/06~10/16 기록분(7행)은 '≈1,200'이 아니라 배치 캐시 ≈600 에서 골랐다.
#   winner_profile._load_universe_ohlcv 가 레인 캐시('날짜' 인덱스) 행을 전부 버리던 버그다.
#   수정 후 같은 날짜를 시점별 파일로 다시 고르면 7일 모두 다른 종목(전부 레인 종목)이
#   나온다(유니버스 599~609 → 1,184~1,193). 기록은 고치지 않는다 — 판정 때 구분할 것.
EXT_REGISTRY = "v1-20261006"
EXT_FEATURES = ("tv_rank_pctl", "ret_20d", "hi60_gap", "rsi14", "vol_ratio")
#: 소액 계좌 체결 가능성 — 레인 유니버스 거래대금 p10이 3.4억(10/2 실측).
EXT_MIN_TV_EOK = 3.0
EXT_MIN_PRICE = 1000.0
EXT_MIN_HISTORY = 60

#: 화면 라벨
LABELS = {"live": "현행", "legacy": "구엔진", "lowvol": "저변동 최저",
          "ext_lowtv": "확장유니버스 저거래대금", "comp_lowvol": "알파+저변동 복합",
          "no_alpha": "알파문턱 제외", "thin_skip": "얇은 풀 현금"}
SHORT = {"lowvol": "저변동", "ext_lowtv": "확장", "comp_lowvol": "복합",
         "no_alpha": "무알파", "thin_skip": "얇은풀"}


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    return (pd.to_numeric(df[c], errors="coerce") if c in df.columns
            else pd.Series(np.nan, index=df.index))


def _flag(df: pd.DataFrame, c: str, default: bool) -> pd.Series:
    if c not in df.columns:
        return pd.Series(default, index=df.index)
    s = df[c]
    if s.dtype == object:                              # [v86] CSV 의 'True'/'False' 문자열 대응
        s = s.map(lambda x: {"true": 1, "false": 0}.get(x.strip().lower(), x)
                  if isinstance(x, str) else x)
    return pd.to_numeric(s, errors="coerce").fillna(1 if default else 0).astype(bool)


def day_blocked(df: pd.DataFrame) -> bool:
    """[v86] risk_off 시장 전체 차단일인가 — 전 행 NEW_ENTRY_BLOCKED."""
    if df is None or df.empty or "NEW_ENTRY_BLOCKED" not in df.columns:
        return False
    return bool(_flag(df, "NEW_ENTRY_BLOCKED", False).mean() >= _DAY_BLOCK_FRAC)


def _route(df: pd.DataFrame, counterfactual: bool) -> pd.Series:
    """ROUTE — 반사실(차단일)이면 봉인 전 원 판정(ROUTE_PRE_BLOCK)을 되살린다."""
    r = (df["ROUTE"].astype(str).str.upper().str.strip() if "ROUTE" in df.columns
         else pd.Series("", index=df.index))
    if counterfactual and "ROUTE_PRE_BLOCK" in df.columns:
        pre = df["ROUTE_PRE_BLOCK"].fillna("").astype(str).str.upper().str.strip()
        r = r.where(pre.eq("") | pre.eq("NAN"), pre)
    return r


def _sizable(route: pd.Series, entry_ok: pd.Series) -> pd.Series:
    """recommendation_quality._route_sizable_mask 와 같은 규칙 (SSOT 참조, 폴백 리터럴)."""
    try:
        from services.recommendation_quality import _alpha_exempt_routes, _inactive_routes
        inactive, exempt = _inactive_routes(), _alpha_exempt_routes()
    except Exception as e:                             # 순환 import 등 — 동일 리터럴로 폴백
        logger.warning("[v86] ROUTE 사이징 규칙 참조 실패 — 폴백: %s", e)
        inactive = frozenset({"WAIT", "OVERHEAT", "EXIT_WARNING", "CARRY", "BLOCKED", "NEUTRAL"})
        exempt = frozenset({"WAIT"})
    return route.eq("") | ~route.isin(inactive) | (entry_ok & route.isin(exempt))


def _pools(df: pd.DataFrame, counterfactual: bool = False) -> Dict[str, pd.Series]:
    """[v86] 현행 퍼널을 기록 컬럼으로 재구성한 풀.

    counterfactual=True(차단일)면 NEW_ENTRY_BLOCKED·MACRO_RISK 를 무시하고 ROUTE_PRE_BLOCK 을 쓴다.
    7월방어·회복방어 플래그는 그대로 둔다(행 단위 차단이라 risk_off 와 별개).

      alpha    : 알파 게이트(=TOP_PICK 재구성) — 리스크가드·알파≥문턱·저점추세·급등·v83 변동성
      live     : alpha ∧ BUY_NOW_PASS ∧ 공통 품질가드 ∧ 사이징 가능 ROUTE
      no_alpha : live 에서 '알파≥문턱'과 '저점추세'만 뺀 풀
    """
    risk = _flag(df, "ENTRY_RISK_GATE_OK", False)
    row_blk = (pd.Series(False, index=df.index) if counterfactual
               else _flag(df, "NEW_ENTRY_BLOCKED", False))
    july = _flag(df, "JULY_PROFIT_BLOCK_FLAG", False)
    rec = _flag(df, "PROFIT_RECOVERY_BLOCK_FLAG", False)
    lt = _flag(df, "ALPHA_LT_OK", True)
    surge = _flag(df, "ALPHA_SURGE_OK", True)
    vol = _flag(df, "ALPHA_VOL_OK", True)
    bnp = _flag(df, "BUY_NOW_PASS", True)
    a = _num(df, "ALPHA_SCORE"); thr = _num(df, "ALPHA_ENTRY_THRESHOLD")
    rr = _num(df, "RR_NOW_TP1"); poc = _num(df, "POC_GAP")
    macro = (df["MACRO_RISK"].astype(str).str.upper().str.strip() if "MACRO_RISK" in df.columns
             else pd.Series("NORMAL", index=df.index))
    # [v86] 반사실(차단일)에서는 매크로 경보도 무시한다 — MACRO_RISK=CRITICAL 은 그날 전 행에
    #   걸리는 시장 전체 신호(=risk_off 의 원인)라, 남겨 두면 live·no_alpha·thin_skip 만 그날
    #   빈 픽이 돼 차단일 요약이 변형마다 다른 날짜를 평균하게 된다(실측: 15 차단일 중 3일).
    macro_bad = (pd.Series(False, index=df.index) if counterfactual
                 else macro.isin(["WARNING", "CRITICAL"]))
    common = (~macro_bad
              & ~_flag(df, "ABNORMAL_HISTORY_GUARD_FLAG", False) & ~rec & ~july
              & _flag(df, "DATA_FRESHNESS_OK", True) & _flag(df, "INPUT_INTEGRITY_OK", True)
              & (poc.isna() | (poc <= 20.0)) & (rr >= 1.30).fillna(False))
    entry_core = risk & ~row_blk & ~july & ~rec & surge & vol
    alpha_ok = entry_core & a.notna() & (a >= thr).fillna(False) & lt
    noalpha_ok = entry_core & a.notna()
    route = _route(df, counterfactual)
    return {
        "alpha": alpha_ok,
        "live": alpha_ok & bnp & common & _sizable(route, alpha_ok),
        "no_alpha": noalpha_ok & bnp & common & _sizable(route, noalpha_ok),
    }


def variant_masks(df: pd.DataFrame, counterfactual: bool = False) -> Dict[str, pd.Series]:
    """변형별 후보 집합. 공통 바닥 = 리스크가드·전면차단 아님·급등 아님·손익비≥1.3·POC·상태.

    [v86] counterfactual=True 는 risk_off 차단일의 반사실 — 차단을 무시하고 봉인 전 ROUTE 로
    같은 규칙을 적용한다. 이때 live 는 PRODUCTION_BUY(전부 0)가 아니라 재구성 풀이다.
    """
    risk = _flag(df, "ENTRY_RISK_GATE_OK", False)
    blocked = (pd.Series(False, index=df.index) if counterfactual
               else _flag(df, "NEW_ENTRY_BLOCKED", False))
    surge = _flag(df, "ALPHA_SURGE_OK", True)
    lt = _flag(df, "ALPHA_LT_OK", True)
    vol = _flag(df, "ALPHA_VOL_OK", True)
    bnp = _flag(df, "BUY_NOW_PASS", True)
    rr = _num(df, "RR_NOW_TP1")
    poc = _num(df, "POC_GAP")
    route = _route(df, counterfactual)
    a = _num(df, "ALPHA_SCORE"); thr = _num(df, "ALPHA_ENTRY_THRESHOLD")
    base = (risk & ~blocked & surge & bnp & (rr >= 1.30).fillna(False)
            & (poc.isna() | (poc <= 20.0)) & ~route.isin(_INACTIVE_ROUTES))
    lowvol = base & vol & _num(df, "V23_ATR_Pct").notna()
    p = _pools(df, counterfactual)
    return {
        "legacy": base & lt & (a >= thr).fillna(False),
        "live": p["live"] if counterfactual else _flag(df, "PRODUCTION_BUY", False),
        "lowvol": lowvol,
        "comp_lowvol": lowvol,
        "no_alpha": p["no_alpha"],
    }


def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def _keys(df: pd.DataFrame, masks: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """변형별 랭킹키(클수록 1등). 현행 키 = 알파×clip(RR,0,3)×1000 + 품질점수(없으면 알파)."""
    a = _num(df, "ALPHA_SCORE").fillna(0.0)
    rr = _num(df, "RR_NOW_TP1").fillna(0.0).clip(0.0, 3.0)
    tie = _num(df, "QUALITY_GUARD_SCORE").fillna(0.0) if "QUALITY_GUARD_SCORE" in df.columns else a
    prod = a * rr * 1_000 + tie
    atr = _num(df, "V23_ATR_Pct")
    m = masks["comp_lowvol"]
    comp = pd.Series(np.nan, index=df.index)
    if m.any():                                        # 분위는 그날 풀 안에서
        comp.loc[m] = _pct(a[m]) + _pct(-atr[m])
    return {"legacy": a * rr * 1_000 + a, "live": prod, "lowvol": -atr,
            "comp_lowvol": comp, "no_alpha": prod}


def _row(df: pd.DataFrame, i, ymd: str, variant: str, blocked: bool, pool_n: int) -> dict:
    r = df.loc[i]
    return {"ymd": ymd, "variant": variant,
            "종목코드": str(r.get("종목코드", "")).zfill(6),
            "종목명": str(r.get("종목명", "")),
            "alpha": float(_num(df, "ALPHA_SCORE").fillna(0.0).loc[i]),
            "rr": float(_num(df, "RR_NOW_TP1").fillna(0.0).clip(0.0, 3.0).loc[i]),
            "atr_pct": float(_num(df, "V23_ATR_Pct").loc[i]),
            "vol_pctl": float(_num(df, "VOL_GATE_PCTL").loc[i]),
            "blocked": float(bool(blocked)), "cash": 0.0, "pool_n": float(pool_n)}


def top_pick_count(df: pd.DataFrame, counterfactual: bool = False) -> int:
    """thin_skip 기준 — 정상일은 기록된 TOP_PICK 합, 차단일은 재구성 알파 게이트 통과 수."""
    if counterfactual or "TOP_PICK" not in df.columns:
        return int(_pools(df, True)["alpha"].sum())
    return int(_flag(df, "TOP_PICK", False).sum())


def ext_lowtv_pick(data_dir: str, trade_ymd: str, px: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """[v84] 확장 유니버스(캐시 ∪ 레인)에서 선등록 5특징 분위 평균 최저 1종목."""
    from services.winner_profile import _features
    if px is None:
        px = _load_universe_ohlcv(data_dir)
    if px is None or px.empty:
        return None
    dt = pd.to_datetime(str(trade_ymd))
    rows = []
    for code, g in px.groupby("종목코드", sort=False):
        h = g[g["Date"] <= dt]
        if len(h) < EXT_MIN_HISTORY or h["Date"].iloc[-1] != dt:
            continue                                   # 오늘 봉이 없는 종목(정지·상폐)은 제외
        f = _features(h.tail(70), None)
        if f is None or f["tv_eok"] < EXT_MIN_TV_EOK or float(h["종가"].iloc[-1]) < EXT_MIN_PRICE:
            continue
        f["종목코드"] = str(code).zfill(6)
        rows.append(f)
    if len(rows) < 30:
        return None
    d = pd.DataFrame(rows)
    d["tv_rank_pctl"] = d["tv_eok"].rank(pct=True) * 100
    ranks = pd.concat([pd.to_numeric(d[c], errors="coerce").rank(pct=True) for c in EXT_FEATURES], axis=1)
    d["_score"] = ranks.mean(axis=1, skipna=False)
    d = d.dropna(subset=["_score"])
    if d.empty:
        return None
    r = d.loc[d["_score"].idxmin()]
    return {"ymd": str(trade_ymd), "variant": "ext_lowtv", "종목코드": r["종목코드"], "종목명": "",
            "alpha": float("nan"), "rr": float("nan"), "atr_pct": float("nan"),
            "vol_pctl": float("nan"), "ext_score": float(r["_score"]), "tv_eok": float(r["tv_eok"])}


def pick_today(df: pd.DataFrame, trade_ymd: str, data_dir: Optional[str] = None) -> List[dict]:
    """변형별 '오늘 1등' 한 행씩. 후보가 없으면 변형은 빠진다(현금).

    data_dir 가 주어지면 [v84] 확장 유니버스 변형(ext_lowtv)도 함께 고른다.
    [v86] risk_off 차단일이면 모든 변형을 반사실로 고르고 blocked=1 로 표시한다.
          thin_skip 이 현금을 택한 날은 cash=1 행(종목코드 빈값)으로 남겨 수익 0으로 잰다.
    """
    if df is None or df.empty or "종목코드" not in df.columns:
        return []
    blk = day_blocked(df)
    m = variant_masks(df, counterfactual=blk)
    key = _keys(df, m)
    out = []
    live_row = None
    for v in ("legacy", "live", "lowvol", "comp_lowvol", "no_alpha"):   # 배치 df 기반 변형
        sub = df[m[v]]
        if sub.empty:
            continue
        k = key[v].reindex(sub.index)
        if k.notna().sum() == 0:
            continue
        row = _row(df, k.idxmax(), trade_ymd, v, blk, len(sub))
        out.append(row)
        if v == "live":
            live_row = row
    # [v86] thin_skip — 현행과 같은 픽, 얇은 풀이면 현금. 현행 픽이 없으면 같이 빠진다.
    if live_row is not None and (blk or "TOP_PICK" in df.columns):
        n_top = top_pick_count(df, counterfactual=blk)
        if n_top <= THIN_POOL_MAX:
            out.append({"ymd": trade_ymd, "variant": "thin_skip", "종목코드": "", "종목명": "현금",
                        "alpha": float("nan"), "rr": float("nan"), "atr_pct": float("nan"),
                        "vol_pctl": float("nan"), "blocked": float(blk), "cash": 1.0,
                        "pool_n": float(n_top)})
        else:
            out.append({**live_row, "variant": "thin_skip", "pool_n": float(n_top)})
    if data_dir:
        try:
            e = ext_lowtv_pick(data_dir, trade_ymd)
        except Exception as ex:                       # 변형 하나가 깨져도 나머지 기록은 산다
            logger.warning("[v84] ext_lowtv 선정 실패 — 생략: %s", ex)
            e = None
        if e is not None:
            nm = df.loc[df["종목코드"].astype(str).str.zfill(6) == e["종목코드"], "종목명"]
            e["종목명"] = str(nm.iloc[0]) if len(nm) else _krx_name(data_dir, e["종목코드"])
            e["blocked"] = float(blk); e["cash"] = 0.0
            out.append(e)
    return out


def _krx_name(data_dir: str, code: str) -> str:
    """[v84.1] 풀 밖 종목(레인 유니버스)의 이름 — 최신 krx_codes_*.csv 에서. 없으면 코드."""
    import glob as _g
    files = sorted(_g.glob(os.path.join(data_dir, "krx_codes_2*.csv")))
    if not files:
        return code
    try:
        k = pd.read_csv(files[-1], dtype=str, usecols=["종목코드", "종목명"])
        hit = k.loc[k["종목코드"].astype(str).str.zfill(6) == str(code).zfill(6), "종목명"]
        return str(hit.iloc[0]) if len(hit) else code
    except Exception as e:
        logger.warning("[v84.1] krx_codes 이름 조회 실패: %s", e)
        return code


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


def _csv_day_blocked(data_dir: str, ymd: str) -> bool:
    """[v86] blocked 컬럼이 없던 v83~v85 기록 보강 — 그날 recommend CSV 로 risk_off 여부."""
    p = os.path.join(data_dir, f"recommend_{ymd}.csv")
    if not os.path.exists(p):
        return False
    try:
        d = pd.read_csv(p, usecols=lambda c: c == "NEW_ENTRY_BLOCKED", low_memory=False)
    except Exception as e:
        logger.warning("[v86] %s 차단 여부 조회 실패 — 정상일로 간주: %s", ymd, e)
        return False
    return day_blocked(d)


def _stats(s: pd.DataFrame) -> dict:
    """한 변형·한 구간의 일별 성적. s: ymd·ret(분수)·cash 컬럼."""
    if not len(s):
        return {"days": 0, "avg_ret_pct": None, "win_rate": None, "stop_rate": None,
                "hac_t": None, "hac_p": None}
    d = s.groupby("ymd")["ret"].mean().sort_index() * 100
    t, p = _hac_p(d.values.astype(float), lag=PREREG_HAC_LAG)
    traded = s[s["cash"] < 0.5]
    return {"days": int(len(d)), "avg_ret_pct": float(d.mean()),
            "win_rate": (float((traded["ret"] > 0).mean()) if len(traded) else None),
            "stop_rate": (float((traded["ret"] <= -0.079).mean()) if len(traded) else None),
            "hac_t": (None if not np.isfinite(t) else float(t)),
            "hac_p": (None if not np.isfinite(p) else float(p))}


def _paired(dd: pd.Series) -> dict:
    dd = dd.dropna().sort_index()
    t, p = _hac_p(dd.values.astype(float), lag=PREREG_HAC_LAG) if len(dd) else (np.nan, np.nan)
    return {"days": int(len(dd)), "diff_pct": (float(dd.mean()) if len(dd) else None),
            "positive_days": int((dd > 0).sum()),
            "hac_t": (None if not np.isfinite(t) else float(t)),
            "hac_p": (None if not np.isfinite(p) else float(p))}


def prereg_verdict(cand: pd.Series, live: pd.Series, traded: Optional[pd.Series] = None,
                   cand_days=None, live_days=None) -> dict:
    """[v86] 선등록 판정 — 순수 함수. 입력은 REGISTERED_FROM 이후 **정상일**의 일별 수익(%).

    cand·live: ymd 인덱스 Series(미실현 날은 빠지거나 NaN). traded: 후보가 실제 매매한 날 True.
    cand_days·live_days: 각 변형이 **픽을 기록한** 정상일(실현 여부 무관). 없으면 실현일로 대신한다.

    창은 '양쪽 모두 픽이 기록된 날'의 날짜순 처음 PREREG_DAYS 일로 고정한다 — 실현 여부로
    고르면 손절(-8%)은 즉시, 나머지는 t+6 에 확정되므로 미확정 앞날을 건너뛰고 일찍 손절된
    뒷날을 끌어와 손실 쪽으로 기운 창이 되고, 앞날이 확정되면 창이 움직여 판정이 뒤집힌다.
    창 안의 모든 날이 양쪽 실현될 때까지는 '측정 중 (실현 수)/40' 이다.
    """
    c_ok, l_ok = cand.dropna(), live.dropna()
    cd = set(map(str, cand_days if cand_days is not None else c_ok.index))
    ld = set(map(str, live_days if live_days is not None else l_ok.index))
    window = sorted(cd & ld)[:PREREG_DAYS]
    j = pd.concat({"c": c_ok, "l": l_ok}, axis=1)
    j.index = j.index.map(str)
    j = j.reindex(window).dropna()
    n = int(len(j))
    if len(window) < PREREG_DAYS or n < PREREG_DAYS:
        res = {"status": f"측정 중 {n}/{PREREG_DAYS}", "n": n, "decided": False,
               "logged": int(len(window))}
        if window:
            res["window"] = [window[0], window[-1]]
        return res
    dd = j["c"] - j["l"]
    t, p = _hac_p(dd.values.astype(float), lag=PREREG_HAC_LAG)
    h = PREREG_DAYS // 2
    h1, h2 = float(dd.iloc[:h].mean()), float(dd.iloc[h:].mean())
    if traded is not None:
        tr0 = traded.copy(); tr0.index = tr0.index.map(str)
        tr = tr0.reindex(j.index).fillna(True).astype(bool)
    else:
        tr = pd.Series(True, index=j.index)
    net = float((j["c"] - PREREG_COST_PCT * tr.astype(float)).mean())
    diff = float(dd.mean())
    checks = {"diff_pos": diff > 0,
              "hac_p": bool(np.isfinite(p) and p < PREREG_ALPHA),
              "halves_same_sign": bool(np.sign(h1) == np.sign(h2) == np.sign(diff) and diff != 0),
              "net_after_cost_pos": net > 0}
    ok = all(checks.values())
    return {"status": STATUS_PROMOTE if ok else STATUS_REJECT, "n": n, "decided": True,
            "logged": int(len(window)), "window": [str(j.index[0]), str(j.index[-1])],
            "diff_pct": diff, "hac_t": (float(t) if np.isfinite(t) else None),
            "hac_p": (float(p) if np.isfinite(p) else None),
            "half1_pct": h1, "half2_pct": h2, "cand_net_pct": net, "checks": checks}


def _frozen_verdict(prev: Optional[dict], c: str) -> Optional[dict]:
    """[v86] 한 번 내려진 판정은 동결 — 같은 PREREG_VERSION 이면 이전 판정을 그대로 쓴다
    (이후 시세 캐시 보정으로 창 안 수익이 바뀌어도 판정이 뒤집히지 않게)."""
    if not prev or prev.get("prereg_version") != PREREG_VERSION:
        return None
    v = (prev.get("verdicts") or {}).get(c)
    if isinstance(v, dict) and v.get("decided") and v.get("status") in (STATUS_PROMOTE, STATUS_REJECT):
        return {**v, "frozen": True}
    return None


def build(data_dir: str) -> dict:
    """로그 전수 × SSOT 실현수익 → 변형별 요약(탐색/선등록 × 정상일/차단일) + 페어드 + 판정."""
    log = load_log(data_dir)
    log = log[log["ymd"].astype(str) >= LIVE_FROM].copy() if len(log) else log.copy()
    for c in ("blocked", "cash"):
        if c not in log.columns:
            log[c] = np.nan
    log["ymd"] = log["ymd"].astype(str)
    log["cash"] = pd.to_numeric(log["cash"], errors="coerce").fillna(0.0)
    blk = pd.to_numeric(log["blocked"], errors="coerce")
    if blk.isna().any():                               # v83~v85 기록 — CSV 로 보강
        miss = sorted(log.loc[blk.isna(), "ymd"].unique())
        fill = {y: float(_csv_day_blocked(data_dir, y)) for y in miss}
        blk = blk.fillna(log["ymd"].map(fill))
    log["blocked"] = blk.fillna(0.0)
    px = _load_universe_ohlcv(data_dir) if len(log) else None
    by_code: Dict[str, pd.DataFrame] = (
        {c: g.reset_index(drop=True) for c, g in px.groupby("종목코드")} if px is not None else {})
    # [v85] '날짜' 버그 수정으로 ext_lowtv 가 레인(601~1200위) 종목을 고르게 됐다 — 레인은 하루
    #   ~30%씩 바뀌므로 최신 파일에 없는 픽 종목은 그 종목을 담은 가장 최근 날짜 파일에서 잰다.
    #   변형 간 같은 잣대를 쓰려고 모든 변형에 적용한다(현재 로그 57행에서는 측정 수 변화 0).
    #   폴백 시리즈는 _realized_full_window 로 잰다 — 밴드 이탈로 잘린 시리즈에 _realized 를 쓰면
    #   손절(-8%)만 잡히고 나머지는 영영 미측정이라, 레인 종목을 고르는 ext_lowtv 만 손실 쪽으로
    #   기운 채 live(배치 캐시 종목)와 페어 비교된다.
    hist: Dict[str, pd.DataFrame] = {}
    if len(log):
        # [v86] 현금 행(종목코드 '')은 이력 조회 대상이 아니다
        codes = log.loc[log["cash"] < 0.5, "종목코드"].astype(str).str.strip()
        missing = set(codes[(codes != "") & (codes.str.lower() != "nan")].str.zfill(6)) - set(by_code)
        if missing:
            hist = _load_code_history(data_dir, missing, since=str(log["ymd"].astype(str).min()))
    rets = []
    for _, r in log.iterrows():
        if float(r["cash"]) >= 0.5:
            rets.append(0.0)                           # [v86] 현금 = 수익 0
            continue
        c = str(r["종목코드"]).zfill(6)
        if c in by_code:
            v = _realized(by_code[c], str(r["ymd"]))
        elif c in hist:
            v = _realized_full_window(hist[c], str(r["ymd"]))
        else:
            v = None
        rets.append(None if v is None else float(v))
    log = log.assign(ret=pd.to_numeric(pd.Series(rets, index=log.index, dtype=object), errors="coerce"))
    meas = log[log["ret"].notna()]
    out: dict = {"live_from": LIVE_FROM, "prereg_version": PREREG_VERSION,
                 "registered_from": REGISTERED_FROM, "prereg_days": PREREG_DAYS,
                 "prereg_rule": {"alpha": PREREG_ALPHA, "hac_lag": PREREG_HAC_LAG,
                                 "cost_pct": PREREG_COST_PCT, "thin_pool_max": THIN_POOL_MAX},
                 "days_total": int(log["ymd"].nunique()) if len(log) else 0,
                 "days_blocked": int(log.loc[log["blocked"] >= 0.5, "ymd"].nunique()) if len(log) else 0,
                 "variants": {}, "paired": {}, "statuses": {}}
    seg = {"exploratory": meas["ymd"] < REGISTERED_FROM, "registered": meas["ymd"] >= REGISTERED_FROM}
    blk_m = {"normal": meas["blocked"] < 0.5, "blocked": meas["blocked"] >= 0.5}
    daily: Dict[tuple, pd.Series] = {}
    traded: Dict[tuple, pd.Series] = {}
    for v in VARIANTS:
        mv = meas["variant"] == v
        normal_all = meas[mv & blk_m["normal"]]
        rec = {**_stats(normal_all),                   # 상단 필드 = LIVE_FROM 이후 정상일 전체(호환)
               "segments": {}}
        for sk, sm in seg.items():
            rec["segments"][sk] = {}
            for bk, bm in blk_m.items():
                s = meas[mv & sm & bm]
                rec["segments"][sk][bk] = _stats(s)
                daily[(v, sk, bk)] = (s.groupby("ymd")["ret"].mean() * 100 if len(s)
                                      else pd.Series(dtype=float))
                traded[(v, sk, bk)] = (s.groupby("ymd")["cash"].mean() < 0.5 if len(s)
                                       else pd.Series(dtype=bool))
        out["variants"][v] = rec
    for a, b in [("live", "legacy")] + [(c, "live") for c in CANDIDATES]:
        pr = {}
        for sk in seg:
            for bk in blk_m:
                pr[f"{sk}_{bk}"] = _paired(daily[(a, sk, bk)] - daily[(b, sk, bk)])
        nl = pr["exploratory_normal"]                  # 호환 필드 = 탐색기 정상일
        out["paired"][f"{a}-{b}"] = {**nl, "segments": pr}
    # [v86] 판정 창은 '픽이 기록된' 등록 후 정상일로 정한다(실현 여부 무관) — prereg_verdict 참고
    reg_log = (log[(log["ymd"] >= REGISTERED_FROM) & (log["blocked"] < 0.5)]
               if len(log) else log)
    logged = {v: sorted(reg_log.loc[reg_log["variant"] == v, "ymd"].unique()) if len(reg_log) else []
              for v in VARIANTS}
    prev = load(data_dir)
    verdicts = {}
    for c in CANDIDATES:
        verdicts[c] = _frozen_verdict(prev, c) or prereg_verdict(
            daily[(c, "registered", "normal")], daily[("live", "registered", "normal")],
            traded[(c, "registered", "normal")], cand_days=logged[c], live_days=logged["live"])
        out["statuses"][c] = verdicts[c]["status"]
    out["statuses"]["legacy"] = STATUS_EXPLORE
    out["verdicts"] = verdicts
    out["verdict"] = (f"선등록 판정({PREREG_VERSION}) — {REGISTERED_FROM} 이후 정상일 페어드 처음 "
                      f"{PREREG_DAYS}일 · HAC({PREREG_HAC_LAG}) p<{PREREG_ALPHA} · 앞뒤 반기 같은 부호 · "
                      f"비용 {PREREG_COST_PCT}% 차감 후 평균>0 · 그 전 기록은 탐색 (코드는 승격하지 않는다)")
    return out


def _clean(o):
    """[v86] JSON 직렬화 — NaN/Inf → null, numpy 스칼라 → 파이썬."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if o is None or isinstance(o, (str, int)):
        return o
    if o is pd.NaT:
        return None
    return str(o)


def save(data_dir: str, summary: dict) -> None:
    try:
        with open(os.path.join(data_dir, SUMMARY_NAME), "w", encoding="utf-8") as f:
            json.dump(_clean(summary), f, ensure_ascii=False, indent=1, allow_nan=False)
    except (OSError, ValueError) as e:
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
    """오늘 변형 픽 기록 → 누적 성적 요약 저장. 반환: 요약(+ today)."""
    today = pick_today(df, trade_ymd, data_dir=data_dir)
    added = append_log(data_dir, today)
    s = build(data_dir)
    s["today"] = today
    s["today_blocked"] = bool(day_blocked(df)) if df is not None else False
    s["added"] = bool(added)
    save(data_dir, s)
    return s


def status_line(s: Optional[dict]) -> str:
    """[v86] 후보별 선등록 상태 한 줄 — '승격 후보'가 있으면 맨 앞."""
    if not s:
        return ""
    st = s.get("statuses") or {}
    if not st:
        return ""
    groups: Dict[str, List[str]] = {}
    for c in CANDIDATES:                               # 같은 상태끼리 묶어 짧게
        if c in st:
            groups.setdefault(str(st[c]), []).append(SHORT.get(c, c))
    order = sorted(groups, key=lambda g: (g != STATUS_PROMOTE, g == STATUS_REJECT, g))
    parts = [f"{g} {'·'.join(groups[g])}" for g in order]
    return f"선등록({s.get('registered_from', REGISTERED_FROM)}~) " + " | ".join(parts)


def line(s: Optional[dict]) -> str:
    if not s:
        return ""
    n_var = len(VARIANTS)
    head = (f"선별 그림자 {n_var}종 — 기록 {s.get('days_total', 0)}일"
            f"(차단일 {s.get('days_blocked', 0)}) · " + (status_line(s) or "선등록 판정 전"))
    today = s.get("today") or []
    tl = " · ".join(f"{LABELS.get(t['variant'], t['variant'])} {t.get('종목명', '')}"
                    for t in today if t.get("variant") in ("live", "comp_lowvol", "no_alpha", "thin_skip"))
    if tl:
        head += f" · 오늘{'(차단일 반사실)' if s.get('today_blocked') else ''} {tl}"
    v = s.get("variants") or {}
    parts = []
    for k in ("live", "lowvol", "ext_lowtv"):
        x = (((v.get(k) or {}).get("segments") or {}).get("exploratory") or {}).get("normal") or {}
        if x.get("days") and x.get("avg_ret_pct") is not None:
            parts.append(f"{LABELS[k]} {x['avg_ret_pct']:+.2f}%/{x['days']}일")
    if parts:
        head += " · 탐색기(판정 미반영) 정상일 " + " · ".join(parts)
    return head
