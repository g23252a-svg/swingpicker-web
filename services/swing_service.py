"""Read-only bridge from the swing model to the existing official-buy contract."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from services.snapshot_integrity import normalize_ymd, snapshot_date

logger = logging.getLogger(__name__)
MODEL_VERSION = "swing_net_profit_v1"


def attach_predictions(df: pd.DataFrame, predictions: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Never set TOP_PICK, PRODUCTION_BUY, sizing, or legacy alpha flags."""
    out = df.copy()
    if out.empty or "종목코드" not in out:
        return out
    if not isinstance(report, dict):
        return out
    asof = normalize_ymd(report.get("asof"))
    if (report.get("version") != MODEL_VERSION or not asof
            or asof != snapshot_date(out)):
        return out
    for name in ("PROB", "SCORE", "RANK"):
        out[f"SWING_{name}"] = np.nan
    out["SWING_STATUS"] = "UNAVAILABLE"
    out["SWING_ASOF"] = asof
    out["SWING_MODEL_VERSION"] = MODEL_VERSION
    out["SWING_REASON"] = "당일 가격 또는 학습 이력 부족"
    valid = report.get("validated") is True and report.get("trade_validated") is True
    out["SWING_TRADE_VALIDATED"] = int(report.get("trade_validated") is True)
    out["SWING_VALIDATION_DAYS"] = int(report.get("oos_days", 0))
    out["SWING_OOS_HIT_RATE"] = report.get("hit_rate", np.nan)
    out["SWING_OOS_NET_RETURN"] = report.get("net_return_mean", np.nan)
    out["SWING_OOS_UNIVERSE_NET_RETURN"] = report.get("universe_net_return_mean", np.nan)
    out["SWING_OOS_COVERAGE"] = report.get("outcome_coverage", np.nan)
    out["SWING_VALIDATION_REASON"] = (report.get("reason", "검증 결과 없음") + " · "
                                       + report.get("trade_reason", "수익 우위 미확인"))
    if predictions.empty or not {"code", "probability", "rank", "score", "reason"}.issubset(predictions):
        return out
    picks = predictions.copy()
    picks["code"] = picks.code.astype(str).str.zfill(6)
    if picks.code.duplicated().any():
        logger.warning("스윙 예측 중복 종목 — 결과 미사용")
        return out
    picks = picks.set_index("code")
    codes = out["종목코드"].astype(str).str.zfill(6)
    probability = pd.to_numeric(codes.map(picks.probability), errors="coerce")
    score = pd.to_numeric(codes.map(picks.score), errors="coerce")
    rank = pd.to_numeric(codes.map(picks["rank"]), errors="coerce")
    usable = (np.isfinite(probability) & probability.between(0, 1)
              & np.isfinite(score) & score.between(0, 100)
              & np.isfinite(rank) & rank.ge(1))
    out.loc[usable, "SWING_STATUS"] = "VALIDATED" if valid else "RESEARCH"
    out.loc[usable, "SWING_PROB"] = probability[usable] if valid else np.nan
    out.loc[usable, "SWING_SCORE"] = score[usable]
    out.loc[usable, "SWING_RANK"] = rank[usable]
    out.loc[usable, "SWING_REASON"] = codes.map(picks.reason)[usable]
    return out


def enrich_from_snapshot(df: pd.DataFrame, data_dir="data") -> pd.DataFrame:
    """A single atomic document binds predictions and validation from the same run."""
    path = Path(data_dir) / "swing_snapshot_latest.json"
    if not path.exists():
        return df
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return attach_predictions(df, pd.DataFrame(payload["predictions"]), payload["validation"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("스윙 예측 읽기 실패 — 기존 결과 유지: %s", exc)
        return df
