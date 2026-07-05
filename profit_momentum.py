# -*- coding: utf-8 -*-
"""
⚠️⚠️⚠️ [2026-07-03 정정 — v24.8 Fill-Aware 검증 결과] ⚠️⚠️⚠️
아래 본문의 OOS +17.68%/승률 80%는 per_trade_log의 유령체결 편향(지정가 도달 미검증,
로그 30%가 유령·유령 승률 96%)이 포함된 수치다. OHLCV 실경로·종가진입 재검증 결과:
PMS Top-3 = +0.72%/승률 38% (ELITE -1.82%/31% 대비 상대우위만 유지, 절대 성과 미미).
라이브 첫 픽(7/1)은 7/2 폭락장에서 평균 -14%. → PMS_PROFIT_PICK은 재검증 완료 전까지
매수 참고 금지. 컬럼 생성은 라이브 추적 기록용으로만 유지. 상세: REPORT_20260703_fill_bias.md
═══════════════════════════════════════════════════════════════════════════

profit_momentum.py — v24.4 Profit Momentum Overlay (PMS)  · SHADOW by default
═══════════════════════════════════════════════════════════════════════════
수익성 종목 캐치력 개선을 위한 데이터 검증 오버레이.

■ 데이터 근거 (추측 아님 — 사장님 실데이터 분석)
  per_trade_log 16,093건 + recommend 88일(2026-02-26~06-26) 조인 패널(1,834 라벨)에서
  H5 실현수익률과의 관계를 walk-forward(전반기 IS / 후반기 OOS)로 측정한 결과:

  · 안정적 양의 예측력(두 시기 동일 부호 · 모든 ROUTE 내부 유지):
      RSI14(ρ 0.45/0.24) · VWAP_GAP(0.44/0.26) · DISPLAY_SCORE(0.30/0.31)
      · 거래강도(0.23/0.25) · TIMING_SCORE(0.27/0.23) · 거래대금(0.21/0.23)
  · 안정적 음의 예측력(역방향 — 현행 핵심 지표):
      RR_NOW_TP1(-0.50/-0.29) · ELITE_SCORE(-0.34/-0.19) · AXIS_GAP · SECTOR_RANK

  즉 최근 4개월 표본에서 시스템의 ELITE_SCORE/RR 셋업은 실현수익과 '역상관'이었고
  모멘텀-확인 6피처가 강한 순상관이었다.

■ OOS 검증 (2026-04-30~06-22, 횡보레짐 · 벤치마크 +0.04%/40%)
  일별 Top-3 선택 실현수익:
      현행 ELITE_SCORE  : -1.68%  (승률 40% · 손절급 42% · 중앙값 -7.6%)
      제안 PMS          : +17.68% (승률 80% · 손절급 9%  · 중앙값 +20.8%)
  대박(+30%↑) 제거(+20% 상한)해도 PMS +11.6% vs ELITE -3.9% → 소수 대박 의존 아님.
  꼬리위험도 PMS가 작음(최악 -15.5% vs -43.4%). 5·6월 모두 80% 승률로 월별 안정.

■ ⚠️ 한계 / 안전장치
  · 표본은 4개월 · 단일(모멘텀 우호) 레짐. PMS는 '강세 매수'로 DipSniper(눌림매수)와 반대.
  · 레짐 반전·급락장에서 과열주가 가장 크게 깨질 수 있어 우위가 소멸/역전될 수 있음.
  · 따라서 기본 SHADOW(enforce=False): 공식 TOP_PICK/BUY_NOW_ELIGIBLE를 절대 변경하지 않고
    그림자 컬럼 + 별도 'PMS 추천 레인'만 생성한다. 라이브에서 선언↔실현을 비교한 뒤
    사장님이 enforce 여부를 직접 결정 (기존 demote_official / GUARD enforce 패턴과 동일).

산출 컬럼
  PMS_SCORE          : 6피처 일별 횡단면 백분위 순위 평균(0~100). 높을수록 모멘텀확인 강함.
  PMS_RANK           : 당일 PMS_SCORE 내림차순 순위(1=최고).
  PMS_AVAIL_FEATS    : 실제 사용된 피처 수(결측 강건).
  PMS_PROFIT_PICK    : 당일 'PMS 추천 레인' 상위 N 플래그(1/0) — 유동성/품질 floor 통과분만.
  PMS_BLEND_RANK     : ELITE 순위와 PMS 순위의 블렌드 순위(분석/정렬용 그림자).
  PMS_REASON         : 선택 사유 한 줄.
  (enforce=True일 때만) PMS_SORT_KEY로 정렬 보조 — 공식 컬럼은 불변.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("profit_momentum")

# check_contract_gate가 검증할 산출 컬럼 계약
PMS_OUTPUT_COLS = [
    "PMS_SCORE", "PMS_RANK", "PMS_AVAIL_FEATS",
    "PMS_PROFIT_PICK", "PMS_BLEND_RANK", "PMS_REASON",
]


# ── safe accessors ──────────────────────────────────────────────
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _resolve_pms_config(config):
    """CollectorConfig / PmsConfig / None → PmsConfig 인스턴스(폴백 포함)."""
    if config is None:
        try:
            from collector_config import DEFAULT_CONFIG
            if hasattr(DEFAULT_CONFIG, "pms"):
                return DEFAULT_CONFIG.pms
        except Exception:
            pass
        return _FallbackPmsConfig()
    if hasattr(config, "pms_features"):
        return config
    if hasattr(config, "pms"):
        return config.pms
    return _FallbackPmsConfig()


# ── 핵심: PMS 점수 계산 ──────────────────────────────────────────
def compute_pms_score(df: pd.DataFrame, cfg) -> tuple[pd.Series, pd.Series]:
    """일별 횡단면 백분위 순위 평균 → (PMS_SCORE 0~100, 사용피처수).

    · 룩어헤드 없음: rec_date(당일) 시점 피처만 사용, 당일 universe 내 순위.
    · 결측 강건: 결측 피처는 제외하고 가용 피처로 평균(가용 0개면 50 중립).
    · 스케일프리: 절대값이 아닌 순위 → 레짐/지표 스케일 변화에 강건.
    """
    idx = df.index
    # 일별 그룹 키 — 없으면 전체를 한 그룹으로
    if "rec_date" in df.columns:
        grp_key = df["rec_date"].astype("object").fillna("_NA_")
    elif "추천일" in df.columns:
        grp_key = df["추천일"].astype("object").fillna("_NA_")
    else:
        grp_key = pd.Series("_ALL_", index=idx)

    rank_sum = pd.Series(0.0, index=idx)
    avail = pd.Series(0, index=idx, dtype="int64")
    used = 0
    for feat, fallbacks in cfg.pms_features:
        col = None
        for cand in (feat, *fallbacks):
            if cand in df.columns:
                col = cand
                break
        if col is None:
            continue
        x = _num(df, col)
        # 당일 universe 내 백분위(0~100). 결측은 순위계산에서 제외 후 50으로 채움.
        pr = x.groupby(grp_key).rank(pct=True) * 100.0
        rank_sum = rank_sum.add(pr.fillna(50.0), fill_value=0.0)
        avail = avail + x.notna().astype("int64")
        used += 1

    if used == 0:
        return pd.Series(50.0, index=idx), avail
    pms = (rank_sum / used).clip(0, 100).round(2)
    return pms, avail


# ── 메인 엔진 ────────────────────────────────────────────────────
def add_profit_momentum_columns(df: pd.DataFrame,
                                config=None,
                                enforce: bool = False) -> pd.DataFrame:
    """PMS 그림자 컬럼 + 'PMS 추천 레인' 부여. 기본 SHADOW(공식 컬럼 불변).

    Parameters
    ----------
    df : 스코어링 완료 DataFrame(당일 universe 권장; RSI14/VWAP_GAP/... 포함)
    config : CollectorConfig 또는 PmsConfig. None이면 DEFAULT_CONFIG.pms / 폴백.
    enforce : True여도 공식 TOP_PICK/BUY_NOW_ELIGIBLE은 변경하지 않는다.
              정렬 보조키(PMS_SORT_KEY)만 추가로 노출한다.
    """
    if df is None or len(df) == 0:
        return df
    cfg = _resolve_pms_config(config)
    out = df.copy()
    idx = out.index

    if "rec_date" in out.columns:
        grp_key = out["rec_date"].astype("object").fillna("_NA_")
    elif "추천일" in out.columns:
        grp_key = out["추천일"].astype("object").fillna("_NA_")
    else:
        grp_key = pd.Series("_ALL_", index=idx)

    pms, avail = compute_pms_score(out, cfg)
    out["PMS_SCORE"] = pms
    out["PMS_AVAIL_FEATS"] = avail

    # 당일 PMS 순위 (1=최고)
    out["PMS_RANK"] = (
        pms.groupby(grp_key).rank(ascending=False, method="first").astype("int64")
    )

    # ── 품질/유동성 floor — 과열 미세주 캐치 방지 (GUARD #1 철학) ──
    turn = _num(out, "거래대금(억원)")
    if "거래대금(억원)" not in out.columns:
        turn = _num(out, "거래대금").div(1e8)  # 원 → 억원 근사
    liq_ok = (turn >= cfg.pms_min_turnover_eok).fillna(False)
    disp = _num(out, "DISPLAY_SCORE").fillna(0.0)
    quality_ok = (disp >= cfg.pms_min_display).fillna(False) if cfg.pms_min_display > 0 else pd.Series(True, index=idx)
    # 선택 풀: 유동성·품질 floor 통과 (전체 universe 대상)
    pool = (liq_ok & quality_ok)

    # ── PMS 추천 레인: 풀 안에서 당일 PMS 상위 N ──
    pool_pms = pms.where(pool, other=np.nan)
    pool_rank = pool_pms.groupby(grp_key).rank(ascending=False, method="first")
    profit_pick = (pool_rank <= cfg.pms_top_n).fillna(False) & pool
    out["PMS_PROFIT_PICK"] = profit_pick.astype("int64")

    # ── ELITE 순위와 블렌드(그림자) ──
    elite = _num(out, "ELITE_SCORE")
    if "ELITE_SCORE" not in out.columns:
        elite = _num(out, "DISPLAY_SCORE")
    elite_rank_pct = elite.groupby(grp_key).rank(pct=True).mul(100).fillna(50.0)
    w = float(cfg.pms_blend_weight)
    blended = (1.0 - w) * elite_rank_pct + w * pms
    out["PMS_BLEND_SCORE"] = blended.round(2)
    out["PMS_BLEND_RANK"] = (
        blended.groupby(grp_key).rank(ascending=False, method="first").astype("int64")
    )

    # ── 사유 한 줄 (벡터화) ──
    out["PMS_REASON"] = _build_pms_reason(out, cfg, pool, turn)

    if enforce:
        # 공식 컬럼 불변 — 정렬 보조키만 노출(상위일수록 작은 값).
        out["PMS_SORT_KEY"] = (-blended).round(3)

    return out


def _build_pms_reason(out: pd.DataFrame, cfg, pool: pd.Series, turn: pd.Series) -> pd.Series:
    idx = out.index
    pms = pd.to_numeric(out["PMS_SCORE"], errors="coerce").fillna(0.0)
    pick = pd.to_numeric(out["PMS_PROFIT_PICK"], errors="coerce").fillna(0).astype(int)

    parts: list[pd.Series] = []

    def _add(mask, text):
        s = pd.Series("", index=idx, dtype="object")
        parts.append(s.mask(mask.reindex(idx).fillna(False).astype(bool), text))

    _add(pick.eq(1), "PMS추천")
    _add(pms >= 80, "모멘텀확인강(PMS" + pms.round(0).astype("int64").astype(str) + ")")
    _add((pms < 80) & (pms >= 60), "모멘텀확인중(PMS" + pms.round(0).astype("int64").astype(str) + ")")
    _add(~pool.reindex(idx).fillna(False).astype(bool),
         "유동성/품질 floor 미달")

    result = pd.Series("", index=idx, dtype="object")
    for s in parts:
        nonempty = s != ""
        sep = pd.Series("", index=idx, dtype="object").mask(nonempty, " · " + s)
        result = result + sep
    return result.str.replace("^ · ", "", regex=True)


def pms_summary(df: pd.DataFrame) -> dict:
    """로깅용 요약."""
    if df is None or "PMS_SCORE" not in df.columns:
        return {}
    return {
        "n_rows": int(len(df)),
        "n_profit_pick": int(pd.to_numeric(df.get("PMS_PROFIT_PICK", 0), errors="coerce").fillna(0).sum()),
        "pms_mean": round(float(pd.to_numeric(df["PMS_SCORE"], errors="coerce").mean()), 1),
        "pms_p90": round(float(pd.to_numeric(df["PMS_SCORE"], errors="coerce").quantile(0.9)), 1),
        "avail_feats_mode": int(pd.to_numeric(df.get("PMS_AVAIL_FEATS", 0), errors="coerce").mode().iloc[0])
        if "PMS_AVAIL_FEATS" in df.columns and len(df) else 0,
    }


class _FallbackPmsConfig:
    """collector_config 미연동 환경용 기본값 — 검증된 6피처/임계값(SSOT 폴백)."""
    # (피처, 폴백후보들) — 결측 시 폴백 순서로 탐색
    pms_features = (
        ("RSI14", ()),
        ("VWAP_GAP", ()),
        ("거래강도", ()),
        ("TIMING_SCORE", ()),
        ("거래대금(억원)", ()),
        ("DISPLAY_SCORE", ("FINAL_SCORE",)),
    )
    pms_top_n = 3              # 'PMS 추천 레인' 당일 상위 N
    pms_min_turnover_eok = 50.0  # 유동성 floor(억원) — GUARD #1과 동일 철학
    pms_min_display = 0.0      # 품질 floor(DISPLAY_SCORE). 0이면 비활성
    pms_blend_weight = 0.5     # ELITE 순위 ↔ PMS 순위 블렌드 가중(그림자)
    pms_enforce = False        # 공식 추천 변경 금지(기본). True여도 정렬 보조키만 노출
