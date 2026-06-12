# -*- coding: utf-8 -*-
"""
data_integrity.py — v24.1 P0-C 데이터 무결성 게이트 (SSOT)
═══════════════════════════════════════════════════
OHLC 무결성 감사 + 이상 폭등주 플래그(P0-B 흡수)를 단일 모듈에서 적용한다.

배경 (에이프로젠 -66.7% 손절 사건)
  v24 P0-A(손절 하드캡)·P0-B(ret_10d>300% 모멘텀 제외)는 '증상'을 막았지만,
  근본 원인 후보인 OHLC 왜곡(수정주가 단절, 고가<저가 등)은 파이프라인 어느
  단계에서도 검사되지 않았다 — stop_logic.sanitize_ohlcv는 0원/음수 행 제거만
  수행한다. 본 모듈은 추천 행 단위로 최근 N봉의 무결성을 감사해 컬럼으로
  남기고, 왜곡 종목을 모멘텀 레인에서 제외한다. 사건 재발 시 런타임 디버깅
  없이 recommend CSV의 DATA_INTEGRITY_REASON만으로 원인 추적이 가능해진다.

설계 원칙 (guard_system.py / stop_override.py와 동일)
  1. 순수 함수: 입력 df → DATA_INTEGRITY_* 컬럼 부여한 df 반환 (in-place 아님).
  2. 임계값은 전부 collector_config.DataIntegrityConfig(SSOT)에서 주입.
  3. 데이터 누락에 강건 — OHLCV가 없으면 감사 SKIP(OK 취급) + 사유 기록.
  4. 공식 매수 산식 무변경(기본값): TOP_PICK / BUY_NOW_ELIGIBLE은 건드리지
     않는다. demote_official=True(기본 False)일 때만 '무결성 실패' 종목의
     BUY_NOW_ELIGIBLE을 0으로 강등한다 (TOP_PICK 자체는 진단용으로 보존).
  5. enabled=False여도 ABNORMAL_SURGE_FLAG(P0-B)는 항상 동작한다 — v24의
     기존 보호를 어떤 설정에서도 끄지 않기 위함.

검사 항목 (audit_ohlcv_window)
  V1 가격 불변식:  고가 < max(시가,종가) / 저가 > min(시가,종가) / 고가 < 저가
  V2 비양수 가격:  시·고·저·종 중 0 이하 (sanitize_ohlcv 누락분 2차 방어)
  V3 종가 점프:    |1일 종가 변화율| > jump_limit_pct (기본 45%)
                   — KRX 가격제한폭 ±30%를 정규 거래로 넘을 수 없음
                   → 수정주가 미반영·병합/감자 단절·데이터 오류 의심.
                   (거래재개 갭 등 '합법' 점프도 지표 창을 오염시키므로 플래그가 맞다)
  위반 봉 수 > max_bad_bars(기본 0) 이면 NOT OK.

산출 컬럼 (DATA_INTEGRITY_COLS — check_contract_gate step 13이 검증)
  DATA_INTEGRITY_OK      : bool — 무결성 통과 여부 (감사 불가 시 True + SKIP 사유)
  DATA_INTEGRITY_REASON  : str  — 위반 요약 (예: "V3:종가점프1582%x2")
  DATA_INTEGRITY_NBAD    : int  — 검사 창 내 위반 봉 수
  ABNORMAL_SURGE_FLAG    : bool — ret_10d_% > surge_ret10_pct (기존 P0-B 흡수)

소비처
  pipeline_calibrate 말미 1회 호출 (v24 P0-B 인라인 블록 대체).
  '무결성 실패 ∪ 폭등 플래그' → MOMENTUM_LANE=0 (기존 P0-B 동작 보존+확장).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("data_integrity")

# 산출 컬럼 계약 — check_contract_gate.py(step 13)가 이 목록을 검증한다.
DATA_INTEGRITY_COLS = [
    "DATA_INTEGRITY_OK",
    "DATA_INTEGRITY_REASON",
    "DATA_INTEGRITY_NBAD",
    "ABNORMAL_SURGE_FLAG",
]

# config 부재 시 폴백 기본값 (collector_config.DataIntegrityConfig와 동일해야 함)
_FALLBACK = {
    "enabled": True,
    "window": 20,
    "jump_limit_pct": 45.0,
    "max_bad_bars": 0,
    "surge_ret10_pct": 300.0,
    "demote_official": False,
}


# ── safe accessors ──────────────────────────────────────────────
def _price_series(df: pd.DataFrame, kor: str, eng: str) -> Optional[pd.Series]:
    """한글/영문 컬럼 동시 지원 가격 시리즈 (없으면 None)."""
    if kor in df.columns:
        return pd.to_numeric(df[kor], errors="coerce")
    if eng in df.columns:
        return pd.to_numeric(df[eng], errors="coerce")
    return None


def _cfg_get(cfg, name: str):
    """DataIntegrityConfig 필드 안전 조회 (cfg=None이어도 동작)."""
    return getattr(cfg, name, _FALLBACK[name]) if cfg is not None else _FALLBACK[name]


def _resolve_cfg(config):
    """CollectorConfig(또는 None)에서 data_integrity 하위 설정을 꺼낸다."""
    di = getattr(config, "data_integrity", None) if config is not None else None
    if di is not None:
        return di
    try:
        from collector_config import DataIntegrityConfig
        return DataIntegrityConfig()
    except Exception as e:  # collector_config 부재(단위테스트 등) — 폴백 기본값 사용
        logger.debug("DataIntegrityConfig 로드 실패 (폴백 기본값 사용): %s", e)
        return None


# ── 핵심 1: 단일 종목 OHLC 무결성 감사 (순수 함수) ────────────────
def audit_ohlcv_window(
    ohlcv: Optional[pd.DataFrame],
    *,
    window: int = 20,
    jump_limit_pct: float = 45.0,
    max_bad_bars: int = 0,
) -> Tuple[bool, str, int]:
    """최근 window봉의 OHLC 무결성을 감사한다.

    Returns:
        (ok, reason, n_bad)
        ok     : 위반 봉 수 <= max_bad_bars
        reason : "" (완전 정상) / "SKIP:..." (감사 불가) / "V1:..;V3:.." (위반 요약)
        n_bad  : 검사 창 내 위반 봉 수
    """
    if ohlcv is None or len(ohlcv) == 0:
        return True, "SKIP:no_data", 0

    df = ohlcv.tail(int(max(2, window)))
    o = _price_series(df, "시가", "Open")
    h = _price_series(df, "고가", "High")
    l = _price_series(df, "저가", "Low")
    c = _price_series(df, "종가", "Close")

    if c is None or c.notna().sum() < 2:
        return True, "SKIP:no_close", 0

    reasons = []
    bad = pd.Series(False, index=df.index)

    # V2 비양수 가격 — sanitize_ohlcv가 제거했어야 하나 2차 방어
    nonpos = pd.Series(False, index=df.index)
    for s in (o, h, l, c):
        if s is not None:
            nonpos = nonpos | (s <= 0).fillna(False)
    if bool(nonpos.any()):
        reasons.append("V2:비양수가격x{}".format(int(nonpos.sum())))
        bad = bad | nonpos

    # V1 가격 불변식 — 시·고·저·종 4개가 모두 있을 때만 판정
    if all(s is not None for s in (o, h, l, c)):
        hi_bad = (h < np.maximum(o, c)).fillna(False)
        lo_bad = (l > np.minimum(o, c)).fillna(False)
        hl_bad = (h < l).fillna(False)
        inv = hi_bad | lo_bad | hl_bad
        if bool(inv.any()):
            reasons.append("V1:OHLC불변식x{}".format(int(inv.sum())))
            bad = bad | inv

    # V3 종가 점프 — 창 내 인접 거래일 |변화율| > 한도
    jump = c.pct_change().abs() * 100.0
    jmask = (jump > float(jump_limit_pct)).fillna(False)
    if bool(jmask.any()):
        worst = float(jump[jmask].max())
        reasons.append("V3:종가점프{:.0f}%x{}".format(worst, int(jmask.sum())))
        bad = bad | jmask

    n_bad = int(bad.sum())
    ok = n_bad <= int(max_bad_bars)
    reason = ";".join(reasons)
    return ok, reason, n_bad


# ── 핵심 2: 추천 df 일괄 적용 (순수 함수) ─────────────────────────
def apply_data_integrity(
    df: pd.DataFrame,
    ohlcv_map: Optional[Dict[str, pd.DataFrame]] = None,
    config=None,
) -> pd.DataFrame:
    """추천 df에 무결성/폭등 컬럼을 부여하고 모멘텀 레인에서 제외한다.

    - ABNORMAL_SURGE_FLAG: ret_10d_% > surge_ret10_pct (enabled와 무관 — P0-B 보존)
    - DATA_INTEGRITY_*  : 행별 OHLCV 감사 (enabled=False면 SKIP:disabled)
    - MOMENTUM_LANE     : (무결성 실패 ∪ 폭등) 행을 0으로 (컬럼 있을 때만)
    - BUY_NOW_ELIGIBLE  : demote_official=True일 때만 '무결성 실패' 행을 0으로
    """
    out = df.copy()
    cfg = _resolve_cfg(config)
    enabled = bool(_cfg_get(cfg, "enabled"))
    window = int(_cfg_get(cfg, "window"))
    jump_limit = float(_cfg_get(cfg, "jump_limit_pct"))
    max_bad = int(_cfg_get(cfg, "max_bad_bars"))
    surge_th = float(_cfg_get(cfg, "surge_ret10_pct"))
    demote = bool(_cfg_get(cfg, "demote_official"))

    # 0) 빈 df — 계약 컬럼만 채우고 반환
    if out.empty:
        out["DATA_INTEGRITY_OK"] = pd.Series(dtype="bool")
        out["DATA_INTEGRITY_REASON"] = pd.Series(dtype="object")
        out["DATA_INTEGRITY_NBAD"] = pd.Series(dtype="int64")
        out["ABNORMAL_SURGE_FLAG"] = pd.Series(dtype="bool")
        return out

    # 1) [P0-B 보존] 이상 폭등 플래그 — enabled와 무관하게 항상 계산
    if "ret_10d_%" in out.columns:
        r10 = pd.to_numeric(out["ret_10d_%"], errors="coerce").fillna(0.0)
    else:
        r10 = pd.Series(0.0, index=out.index)
    surge = (r10 > surge_th).fillna(False)
    out["ABNORMAL_SURGE_FLAG"] = surge.astype(bool)

    # 2) 행 단위 OHLC 무결성 감사
    omap = ohlcv_map or {}
    code_col = "종목코드" if "종목코드" in out.columns else None
    ok_list, reason_list, nbad_list = [], [], []
    for _, row in out.iterrows():
        if not enabled:
            ok_list.append(True)
            reason_list.append("SKIP:disabled")
            nbad_list.append(0)
            continue
        code = str(row[code_col]).zfill(6) if code_col else ""
        ohlcv = omap.get(code)
        if ohlcv is None or len(ohlcv) == 0:
            ok_list.append(True)
            reason_list.append("SKIP:no_ohlcv")
            nbad_list.append(0)
            continue
        ok, reason, n_bad = audit_ohlcv_window(
            ohlcv, window=window, jump_limit_pct=jump_limit, max_bad_bars=max_bad,
        )
        ok_list.append(bool(ok))
        reason_list.append(reason)
        nbad_list.append(int(n_bad))

    out["DATA_INTEGRITY_OK"] = pd.Series(ok_list, index=out.index).astype(bool)
    out["DATA_INTEGRITY_REASON"] = pd.Series(reason_list, index=out.index).astype(str)
    out["DATA_INTEGRITY_NBAD"] = pd.Series(nbad_list, index=out.index).astype(int)

    # 3) 모멘텀 레인 제외 — (무결성 실패 ∪ 폭등). 기존 P0-B 동작 보존 + 확장
    integ_bad = ~out["DATA_INTEGRITY_OK"].astype(bool)
    exclude = (integ_bad | surge).fillna(False)
    if "MOMENTUM_LANE" in out.columns and bool(exclude.any()):
        out.loc[exclude, "MOMENTUM_LANE"] = 0

    # 4) [기본 OFF] 무결성 실패 종목의 BUY_NOW 강등 — 공식 산식 보존 원칙
    if enabled and demote and "BUY_NOW_ELIGIBLE" in out.columns and bool(integ_bad.any()):
        out.loc[integ_bad, "BUY_NOW_ELIGIBLE"] = 0

    return out


# ── 요약 (파이프라인 로그용) ─────────────────────────────────────
def data_integrity_summary(df: pd.DataFrame) -> dict:
    """apply_data_integrity 적용 후 df에서 요약 통계를 뽑는다."""
    if df is None or df.empty:
        return {"n_integrity_bad": 0, "n_surge": 0, "n_momentum_excluded": 0, "n_audited": 0}
    try:
        ok = df.get("DATA_INTEGRITY_OK")
        reason = df.get("DATA_INTEGRITY_REASON")
        surge = df.get("ABNORMAL_SURGE_FLAG")
        n_bad = int((~ok.astype(bool)).sum()) if ok is not None else 0
        n_surge = int(surge.astype(bool).sum()) if surge is not None else 0
        n_audited = 0
        if reason is not None:
            n_audited = int((~reason.astype(str).str.startswith("SKIP")).sum())
        n_excluded = 0
        if ok is not None and surge is not None:
            n_excluded = int(((~ok.astype(bool)) | surge.astype(bool)).sum())
        return {
            "n_integrity_bad": n_bad,
            "n_surge": n_surge,
            "n_momentum_excluded": n_excluded,
            "n_audited": n_audited,
        }
    except Exception as e:
        logger.debug("data_integrity_summary 실패 (무해): %s", e)
        return {"n_integrity_bad": 0, "n_surge": 0, "n_momentum_excluded": 0, "n_audited": 0}
