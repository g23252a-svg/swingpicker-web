#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/analyze_realized_edge.py
================================
실전 수익률 예측력 개선용 read-only 분석기.

목적
----
추천 당시 recommend_YYYYMMDD.csv의 특징값과 이후 backtest_top*_trades_*.csv의
실현 수익률을 결합해, 어떤 조건이 실제 TP1/승률/평균수익을 올렸는지 자동으로
검증한다.

중요 원칙
---------
- BUY_NOW_ELIGIBLE, TOP_PICK, scoring_engine.py는 절대 수정하지 않는다.
- 이 스크립트는 shadow/research 전용이다.
- 특정 조건을 하드 게이트로 승격하려면 최소 표본/월별 일관성 검증이 별도로 필요하다.

사용 예
-------
python scripts/analyze_realized_edge.py
python scripts/analyze_realized_edge.py --data-dir data --min-n 8
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_NUMERIC_FEATURES = [
    "FINAL_SCORE", "DISPLAY_SCORE", "ELITE_SCORE",
    "STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE", "ML_SCORE",
    "ML_RAW_SCORE", "ML_EFFECTIVE_SCORE",
    "AXIS_MEAN", "AXIS_GAP", "BALANCE_SCORE",
    "RR_NOW_TP1", "ENTRY_GAP_PCT", "gap_pct",
    "VWAP_GAP", "POC_GAP", "RES_RATIO", "RES_RATIO_NEAR",
    "RSI14", "MFI14", "MACD_Slope_PCT", "Vol_Quality", "Range_Pos",
    "거래대금(억원)", "시가총액(억원)",
    "BUY_NOW_SCORE", "ANTI_STRUCT_REVERSAL_SCORE",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "ROUTE", "TOP_PICK", "TOP_PICK_TYPE", "BUY_NOW_GRADE", "BUY_NOW_PASS",
    "BUY_NOW_ELIGIBLE", "ENTRY_RISK_LEVEL", "MACRO_RISK", "PASS_EBS",
    "EBS", "NO_CHASE_FLAG", "PULLBACK_WAIT_FLAG", "ANTI_STRUCT_REVERSAL_FLAG",
]


@dataclass(frozen=True)
class EdgeSummary:
    n: int
    win_rate_pct: float
    avg_net_pct: float
    median_net_pct: float
    tp1_before_stop_rate_pct: float
    stop_hit_rate_pct: float
    not_filled_rate_pct: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "win_rate_pct": round(self.win_rate_pct, 2),
            "avg_net_pct": round(self.avg_net_pct, 2),
            "median_net_pct": round(self.median_net_pct, 2),
            "tp1_before_stop_rate_pct": round(self.tp1_before_stop_rate_pct, 2),
            "stop_hit_rate_pct": round(self.stop_hit_rate_pct, 2),
            "not_filled_rate_pct": round(self.not_filled_rate_pct, 2),
        }


def normalize_code(value: object) -> str:
    """종목코드를 6자리 문자열로 정규화한다."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    return digits.zfill(6)[-6:]


def _date_from_name(path: Path, prefix: str) -> str | None:
    match = re.search(rf"{re.escape(prefix)}_(\d{{8}})\.csv$", path.name)
    if not match:
        return None
    return match.group(1)


def _as_ymd(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return digits[:8]
    return digits


def _bool_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    normalized = series.map(
        lambda x: False if pd.isna(x) else str(x).lower() in {"1", "true", "t", "yes", "y"}
    )
    return float(normalized.mean() * 100.0)


def summarize(df: pd.DataFrame) -> EdgeSummary:
    """실현 수익률/승률/TP1/손절 요약."""
    if df.empty:
        return EdgeSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ret = pd.to_numeric(df.get("realized_net_pct", pd.Series(dtype=float)), errors="coerce")
    ret_nonnull = ret.dropna()
    n = int(len(df))
    win = pd.to_numeric(df.get("win", pd.Series([0] * n)), errors="coerce").fillna(0)
    not_filled = pd.to_numeric(df.get("not_filled", pd.Series([0] * n)), errors="coerce").fillna(0)

    return EdgeSummary(
        n=n,
        win_rate_pct=float(win.mean() * 100.0) if n else 0.0,
        avg_net_pct=float(ret_nonnull.mean()) if len(ret_nonnull) else 0.0,
        median_net_pct=float(ret_nonnull.median()) if len(ret_nonnull) else 0.0,
        tp1_before_stop_rate_pct=_bool_rate(df.get("tp1_before_stop", pd.Series(dtype=bool))),
        stop_hit_rate_pct=_bool_rate(df.get("stop_hit", pd.Series(dtype=bool))),
        not_filled_rate_pct=float(not_filled.mean() * 100.0) if n else 0.0,
    )


def load_recommend_features(data_dir: Path) -> pd.DataFrame:
    """recommend_YYYYMMDD.csv 전체를 읽어 추천 당시 특징값을 만든다."""
    frames: list[pd.DataFrame] = []
    for path in sorted(data_dir.glob("recommend_*.csv")):
        rec_date = _date_from_name(path, "recommend")
        if rec_date is None:
            continue
        df = pd.read_csv(path)
        if "종목코드" not in df.columns:
            continue
        df = df.copy()
        df["signal_date"] = rec_date
        df["code_norm"] = df["종목코드"].map(normalize_code)
        keep = ["signal_date", "code_norm", "종목명"]
        keep.extend([c for c in DEFAULT_NUMERIC_FEATURES if c in df.columns])
        keep.extend([c for c in DEFAULT_CATEGORICAL_FEATURES if c in df.columns])
        keep = list(dict.fromkeys(keep))
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame(columns=["signal_date", "code_norm"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["signal_date", "code_norm"], keep="last")
    return out


def load_trade_results(data_dir: Path) -> pd.DataFrame:
    """backtest_top1/top3 trade 결과를 중복 제거해서 읽는다."""
    frames: list[pd.DataFrame] = []
    patterns = ["backtest_top1_trades_*.csv", "backtest_top3_trades_*.csv"]
    for pattern in patterns:
        source_set = "TOP1" if "top1" in pattern else "TOP3"
        for path in sorted(data_dir.glob(pattern)):
            if path.name.endswith("_latest.csv"):
                # dated 파일들과 중복되므로 latest는 제외한다.
                continue
            df = pd.read_csv(path)
            if "date" not in df.columns or "code" not in df.columns:
                continue
            df = df.copy()
            df["source_set"] = source_set
            df["signal_date"] = df["date"].map(_as_ymd)
            df["code_norm"] = df["code"].map(normalize_code)
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["source_set", "signal_date", "code_norm"])

    out = pd.concat(frames, ignore_index=True)
    # TOP1 is a strict subset of TOP3 in the cumulative backtest exports.  The
    # former implementation counted the same signal twice and overstated n.
    # Prefer TOP3 for overlapping rows, while retaining TOP1-only fixtures.
    out["_source_priority"] = out["source_set"].map({"TOP3": 0, "TOP1": 1}).fillna(9)
    out = out.sort_values("_source_priority", kind="mergesort")
    subset = ["signal_date", "code_norm", "method", "fill_date", "outcome"]
    subset = [c for c in subset if c in out.columns]
    out = out.drop_duplicates(subset=subset, keep="first").drop(columns=["_source_priority"])

    ret_col = "net_pct" if "net_pct" in out.columns else "ret_pct"
    out["realized_net_pct"] = pd.to_numeric(out.get(ret_col), errors="coerce")
    outcome = out.get("outcome", pd.Series([""] * len(out))).fillna("").astype(str).str.upper()
    out["win"] = ((out["realized_net_pct"] > 0) | (outcome == "WIN")).astype(int)
    out["not_filled"] = outcome.eq("NOT_FILLED").astype(int)
    return out


def build_edge_dataset(data_dir: Path) -> pd.DataFrame:
    """추천 당시 feature와 이후 실현 결과를 결합한다."""
    trades = load_trade_results(data_dir)
    recs = load_recommend_features(data_dir)
    if trades.empty or recs.empty:
        return pd.DataFrame()
    merged = trades.merge(recs, on=["signal_date", "code_norm"], how="left", suffixes=("", "_rec"))
    return merged


def _safe_float(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_numeric_slices(
    df: pd.DataFrame,
    features: Iterable[str] = DEFAULT_NUMERIC_FEATURES,
    min_n: int = 8,
) -> pd.DataFrame:
    """숫자 feature별 상/하위 slice의 실현 수익률 edge를 계산한다."""
    baseline = summarize(df)
    rows: list[dict] = []
    if df.empty:
        return pd.DataFrame(rows)

    for feature in features:
        if feature not in df.columns:
            continue
        series = pd.to_numeric(df[feature], errors="coerce")
        valid = df[series.notna()].copy()
        if len(valid) < min_n * 2:
            continue
        values = pd.to_numeric(valid[feature], errors="coerce")
        for q in (0.25, 0.50, 0.75):
            threshold = _safe_float(values.quantile(q))
            if threshold is None:
                continue
            masks = {
                "LOW_OR_EQ": values <= threshold,
                "HIGH": values > threshold,
            }
            for direction, mask in masks.items():
                sub = valid[mask]
                if len(sub) < min_n:
                    continue
                summary = summarize(sub)
                return_alpha = summary.avg_net_pct - baseline.avg_net_pct
                win_alpha = summary.win_rate_pct - baseline.win_rate_pct
                stop_alpha = summary.stop_hit_rate_pct - baseline.stop_hit_rate_pct
                verdict = "NEUTRAL"
                if return_alpha >= 1.0 and win_alpha >= 3.0:
                    verdict = "EDGE"
                elif return_alpha <= -1.0 and (win_alpha <= -3.0 or stop_alpha >= 3.0):
                    verdict = "RISK"
                rows.append({
                    "kind": "numeric",
                    "feature": feature,
                    "direction": direction,
                    "threshold": round(float(threshold), 4),
                    "rule": f"{feature} <= {threshold:.4g}" if direction == "LOW_OR_EQ" else f"{feature} > {threshold:.4g}",
                    "verdict": verdict,
                    **summary.as_dict(),
                    "return_alpha_pct": round(return_alpha, 2),
                    "win_alpha_pp": round(win_alpha, 2),
                    "stop_alpha_pp": round(stop_alpha, 2),
                })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        by=["verdict", "return_alpha_pct", "win_alpha_pp", "n"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def evaluate_categorical_slices(
    df: pd.DataFrame,
    features: Iterable[str] = DEFAULT_CATEGORICAL_FEATURES,
    min_n: int = 8,
) -> pd.DataFrame:
    """범주형 feature별 값 slice의 실현 수익률 edge를 계산한다."""
    baseline = summarize(df)
    rows: list[dict] = []
    if df.empty:
        return pd.DataFrame(rows)

    for feature in features:
        if feature not in df.columns:
            continue
        values = df[feature].fillna("<NA>").astype(str)
        for value, sub in df.groupby(values):
            if len(sub) < min_n:
                continue
            summary = summarize(sub)
            return_alpha = summary.avg_net_pct - baseline.avg_net_pct
            win_alpha = summary.win_rate_pct - baseline.win_rate_pct
            stop_alpha = summary.stop_hit_rate_pct - baseline.stop_hit_rate_pct
            verdict = "NEUTRAL"
            if return_alpha >= 1.0 and win_alpha >= 3.0:
                verdict = "EDGE"
            elif return_alpha <= -1.0 and (win_alpha <= -3.0 or stop_alpha >= 3.0):
                verdict = "RISK"
            rows.append({
                "kind": "categorical",
                "feature": feature,
                "direction": "EQ",
                "threshold": value,
                "rule": f"{feature} == {value}",
                "verdict": verdict,
                **summary.as_dict(),
                "return_alpha_pct": round(return_alpha, 2),
                "win_alpha_pp": round(win_alpha, 2),
                "stop_alpha_pp": round(stop_alpha, 2),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        by=["verdict", "return_alpha_pct", "win_alpha_pp", "n"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def build_report(df: pd.DataFrame, slices: pd.DataFrame, min_n: int) -> dict:
    baseline = summarize(df).as_dict()
    edge = slices[slices["verdict"] == "EDGE"].head(10) if not slices.empty else pd.DataFrame()
    risk = slices[slices["verdict"] == "RISK"].sort_values(
        by=["return_alpha_pct", "win_alpha_pp"], ascending=[True, True]
    ).head(10) if not slices.empty else pd.DataFrame()
    ml_diagnostic = {
        "status": "NO_ML_SAMPLES",
        "feature": None,
        "n": 0,
        "spearman": None,
        "monthly_spearman": {},
        "production_eligible": False,
    }
    for feature in ("ML_EFFECTIVE_SCORE", "ML_RAW_SCORE", "ML_SCORE", "AI_SCORE"):
        if feature not in df.columns:
            continue
        score = pd.to_numeric(df[feature], errors="coerce")
        realized = pd.to_numeric(df.get("realized_net_pct"), errors="coerce")
        valid = score.notna() & realized.notna()
        if int(valid.sum()) < 10:
            continue
        corr = score[valid].corr(realized[valid], method="spearman")
        monthly = {}
        if "signal_date" in df.columns:
            month = df["signal_date"].astype(str).str[:6]
            for key in sorted(month[valid].unique()):
                mask = valid & month.eq(key)
                if int(mask.sum()) >= 5:
                    value = score[mask].corr(realized[mask], method="spearman")
                    monthly[str(key)] = None if pd.isna(value) else round(float(value), 4)
        stable_positive_months = sum(v is not None and v > 0 for v in monthly.values())
        eligible = bool(
            pd.notna(corr)
            and float(corr) >= 0.10
            and len(monthly) >= 3
            and stable_positive_months >= max(2, int(len(monthly) * 0.6))
        )
        ml_diagnostic = {
            "status": "VALIDATED" if eligible else "UNVALIDATED",
            "feature": feature,
            "n": int(valid.sum()),
            "spearman": None if pd.isna(corr) else round(float(corr), 4),
            "monthly_spearman": monthly,
            "production_eligible": eligible,
        }
        break

    return {
        "version": "realized_edge_audit_v1",
        "purpose": "실전 수익률 예측력 개선을 위한 read-only alpha slice 검증",
        "policy": {
            "production_effect": "none",
            "hard_gate_change": False,
            "min_n": min_n,
            "deduplication": "TOP3 preferred over overlapping TOP1 signal",
            "promotion_rule": "n>=30, 월별 일관성, return_alpha>0, stop_alpha<=0 검증 전까지 공식 매수식에 반영 금지",
        },
        "baseline": baseline,
        "ml_diagnostic": ml_diagnostic,
        "top_edge_slices": edge.to_dict(orient="records") if not edge.empty else [],
        "top_risk_slices": risk.to_dict(orient="records") if not risk.empty else [],
    }


def write_outputs(df: pd.DataFrame, slices: pd.DataFrame, report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slices_path = out_dir / "realized_edge_slices_latest.csv"
    report_path = out_dir / "realized_edge_report_latest.json"
    dataset_path = out_dir / "realized_edge_dataset_latest.csv"
    df.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    slices.to_csv(slices_path, index=False, encoding="utf-8-sig")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="실전 수익률 edge slice 분석기")
    parser.add_argument("--data-dir", default="data", help="recommend/backtest CSV 폴더")
    parser.add_argument("--out-dir", default="data", help="리포트 출력 폴더")
    parser.add_argument("--min-n", type=int, default=8, help="slice 최소 표본 수")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    df = build_edge_dataset(data_dir)
    if df.empty:
        print("⚠️ 분석 가능한 recommend/backtest 결합 데이터가 없습니다.")
        return 2

    numeric = evaluate_numeric_slices(df, min_n=args.min_n)
    categorical = evaluate_categorical_slices(df, min_n=args.min_n)
    slices = pd.concat([numeric, categorical], ignore_index=True) if not categorical.empty else numeric
    report = build_report(df, slices, args.min_n)
    write_outputs(df, slices, report, out_dir)

    print("📈 Realized Edge Audit 완료")
    print(f"  표본 수: {report['baseline']['n']}")
    print(f"  평균 수익률: {report['baseline']['avg_net_pct']}%")
    print(f"  승률: {report['baseline']['win_rate_pct']}%")
    print(f"  slice 수: {len(slices)}")
    print(f"  리포트: {out_dir / 'realized_edge_report_latest.json'}")
    print(f"  slice CSV: {out_dir / 'realized_edge_slices_latest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
