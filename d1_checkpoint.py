# -*- coding: utf-8 -*-
"""
d1_checkpoint.py — v24.7 D+1 Checkpoint (조기 손실컷) · SHADOW 표시 전용
═══════════════════════════════════════════════════════════════════════
진입 다음 날(D+1) 종가가 진입가 대비 임계(-4%) 이탈 시 조기청산을 권고하는
규칙을 추천 데이터에 '가격'으로 명시한다. 공식 산식/청산 시뮬은 변경하지 않는다.

■ 데이터 근거 (per_trade_log 16,549건 · h1↔h5 페어 1,878건 · 2026-02-26~06-24)
  h5 손익 분해: TP +13,772%p vs 손절 -7,886%p — 손절이 번 것의 87%를 반납.
  D+1 종가 손익 구간 → D+5 결말이 완벽 단조:
      D1 ≤ -7% → h5 승률 4.5% · 손절전이 86%   (회생 거의 불가)
      D1 ≥ +7% → h5 평균 +16.7% · 승률 85%     (달리는 놈은 계속)
  D1 ≤ -4% (201건): 즉시청산 -6.7% vs 5일보유 -9.3% → 조기청산 +2.5%p/건,
      이들의 63%는 결국 평균 -13.5% 손절까지 진행.

■ walk-forward / 서브셋 안정성
  IS(전반 강세장) 이득 +0.03%p — 비용 거의 0
  OOS(후반 험한장) 이득 +3.94%p/건 · 손절전이 79% — 강한 방어
  밸런스≥70 좋은 종목에서도 +4.27%p (좋은 종목도 예외 없음)
  → 강세장 공짜 · 하락장 방어의 '비대칭 보험' 프로파일.
  전체 h1-hold 풀(1,366건) 적용 시 총 +511%p (+0.37%p/건).

■ 컷 선택: -4%
  -2%는 개선 +0.44%p로 최대지만 발동 25%로 잦아 왕복수수료(0.22%)·재진입 부담.
  -4%는 발동 15%로 개선 +0.37%p 유지 — robust. (D1CheckpointConfig.d1_cut_pct)

■ 적용 범위 (이 모듈이 하는 것 / 안 하는 것)
  한다: 추천 df에 D1_CHECKPOINT_PCT / D1_CHECKPOINT_PRICE / D1_CHECKPOINT_REASON
        표시 컬럼 부여. UI(액션 카드 등)가 '진입 시 익일 종가 X원 이탈 → 청산'으로 안내.
  안 한다: TOP_PICK/BUY_NOW_ELIGIBLE/손절가/시뮬레이터 변경 — 청산 시뮬 반영은
        validation_engine 쪽 별도 단계로 진행 권고(리포트 참조).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("d1_checkpoint")

D1_OUTPUT_COLS = ["D1_CHECKPOINT_PCT", "D1_CHECKPOINT_PRICE", "D1_CHECKPOINT_REASON"]


class _FallbackD1Config:
    """collector_config 미연동 환경용 기본값 (SSOT 폴백)."""
    d1_cut_pct = -4.0          # D+1 종가가 진입가 대비 이 % 이하 → 조기청산 권고
    d1_min_entry = 100.0       # 진입가 유효 최소값(원) — 미만이면 계산 생략
    d1_enabled = True          # 표시 컬럼 생성 여부


def _resolve_config(config):
    if config is None:
        try:
            from collector_config import DEFAULT_CONFIG
            if hasattr(DEFAULT_CONFIG, "d1"):
                return DEFAULT_CONFIG.d1
        except Exception:
            logger.warning("D+1 설정 로드 실패 — 폴백 설정 사용", exc_info=True)
        return _FallbackD1Config()
    if hasattr(config, "d1_cut_pct"):
        return config
    if hasattr(config, "d1"):
        return config.d1
    return _FallbackD1Config()


def add_d1_checkpoint_columns(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """추천 df에 D+1 체크포인트 표시 컬럼 부여 (공식 컬럼 불변).

    D1_CHECKPOINT_PRICE = 추천매수가 × (1 + d1_cut_pct/100), 호가단위 무시(원 단위 반올림).
    추천매수가가 없거나 무효면 해당 행은 NaN/빈 사유로 남긴다(결측 강건).
    """
    if df is None or len(df) == 0:
        return df
    cfg = _resolve_config(config)
    out = df.copy()

    if not getattr(cfg, "d1_enabled", True):
        return out

    entry = pd.to_numeric(out.get("추천매수가", pd.Series(np.nan, index=out.index)),
                          errors="coerce")
    valid = entry.notna() & (entry >= float(getattr(cfg, "d1_min_entry", 100.0)))

    pct = float(cfg.d1_cut_pct)
    price = (entry * (1.0 + pct / 100.0)).round(0)

    out["D1_CHECKPOINT_PCT"] = np.where(valid, pct, np.nan)
    out["D1_CHECKPOINT_PRICE"] = price.where(valid)

    reason = pd.Series("", index=out.index, dtype="object")
    fmt_price = price.where(valid).map(
        lambda v: f"{int(v):,}" if pd.notna(v) else "")
    reason = reason.mask(
        valid,
        "진입 후 D+1 종가 " + fmt_price + "원(" + f"{pct:.0f}" + "%) 이탈 시 익일 조기청산 권고"
    )
    out["D1_CHECKPOINT_REASON"] = reason

    return out


def d1_summary(df: pd.DataFrame) -> dict:
    if df is None or "D1_CHECKPOINT_PRICE" not in df.columns:
        return {}
    p = pd.to_numeric(df["D1_CHECKPOINT_PRICE"], errors="coerce")
    return {"n_rows": int(len(df)), "n_with_checkpoint": int(p.notna().sum())}
