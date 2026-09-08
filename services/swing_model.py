"""Close-time swing-profit probabilities with purged, calibrated walk-forward validation.

The label is cost-adjusted profit from next-open entry, up to five trading
sessions of holding and an 8% stop with gap-through execution. No score is a probability until OOS gates pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from services.swing_features import FEATURES, load_feature_panel

logger = logging.getLogger(__name__)
MODEL_VERSION = "swing_net_profit_v1"
TARGET_DEFINITION = "익일 시가 진입·최대 5거래일 보유·손절 및 비용 후 수익"


@dataclass(frozen=True)
class SwingPolicy:
    min_train_days: int = 60
    purge_sessions: int = 5
    min_calibration_days: int = 10
    calibration_days: int = 15
    test_block_days: int = 20
    max_folds: int = 4
    min_oos_days: int = 40
    min_auc: float = 0.53
    min_hit_lift: float = 0.03
    max_calibration_error: float = 0.06
    top_k: int = 5
    round_trip_cost: float = 0.003  # configurable cost + slippage scenario, not a tax quote
    min_turnover: float = 1_000_000_000  # KRW, known at signal close
    max_day_move: float = 0.25


POLICY = SwingPolicy()


def eligible_panel(panel: pd.DataFrame, policy=POLICY) -> pd.DataFrame:
    out = panel.copy()
    mask = (np.isfinite(out["close"]) & out["close"].gt(0)
            & out["volume"].gt(0) & out["turnover"].ge(policy.min_turnover)
            & out["ret_1d"].abs().lt(policy.max_day_move))
    return out.loc[mask].sort_values(["date", "code"]).reset_index(drop=True)


def purged_split(panel: pd.DataFrame, test_start, test_end, policy=POLICY):
    """All model/calibrator labels end strictly before their next partition."""
    days = pd.DatetimeIndex(sorted(panel.loc[panel.date < test_start, "date"].unique()))
    if len(days) < policy.min_train_days + policy.calibration_days + policy.purge_sessions:
        return (panel.iloc[:0],) * 3
    cal_start = days[-policy.calibration_days]
    ready = panel.get("_label_ready", panel.label_end)
    train = panel[(panel.date < cal_start) & (ready < cal_start)]
    cal = panel[(panel.date >= cal_start) & (panel.date < test_start)
                & (ready < test_start)]
    if (train.date.nunique() < policy.min_train_days
            or cal.date.nunique() < policy.min_calibration_days):
        return (panel.iloc[:0],) * 3
    test = panel[(panel.date >= test_start) & (panel.date <= test_end)]
    return train, cal, test


def _x(frame):
    return frame[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)


def _fit(train, calibration):
    """Fixed equal-weight ensemble, no test-set model/threshold selection."""
    if train.target_up.nunique() < 2 or calibration.target_up.nunique() < 2:
        raise ValueError("학습/확률 보정 구간에 두 방향 표본이 필요합니다")
    linear = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(C=0.1, max_iter=400, random_state=19))
    trees = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=7, max_depth=3, min_samples_leaf=80,
        l2_regularization=10.0, learning_rate=0.05,
        early_stopping=False, random_state=19)
    # Equal total weight per trading session prevents crowded days dominating.
    weight = 1 / train.groupby("date")["date"].transform("size")
    weight /= weight.mean()
    with threadpool_limits(limits=2):
        linear.fit(_x(train), train.target_up.astype(int),
                   logisticregression__sample_weight=weight)
        trees.fit(_x(train), train.target_up.astype(int), sample_weight=weight)
        raw = (linear.predict_proba(_x(calibration))[:, 1]
               + trees.predict_proba(_x(calibration))[:, 1]) / 2
        calibrator = LogisticRegression(C=1.0, random_state=19)
        cw = 1 / calibration.groupby("date")["date"].transform("size")
        cw /= cw.mean()
        calibrator.fit(_logit(raw), calibration.target_up.astype(int), sample_weight=cw)
    return linear, trees, calibrator


def _logit(p):
    p = np.clip(np.asarray(p), 0.001, 0.999)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _predict(model, frame):
    linear, trees, calibrator = model
    with threadpool_limits(limits=2):
        raw = (linear.predict_proba(_x(frame))[:, 1]
               + trees.predict_proba(_x(frame))[:, 1]) / 2
        return calibrator.predict_proba(_logit(raw))[:, 1]


def _block_ci(values, block=5, repetitions=1000):
    """Moving-block bootstrap over days, not correlated individual stocks."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return [None, None]
    rng = np.random.default_rng(19)
    width = min(block, len(values))
    starts = rng.integers(0, len(values), size=(repetitions, int(np.ceil(len(values) / width))))
    indices = (starts[:, :, None] + np.arange(width)) % len(values)
    means = values[indices.reshape(repetitions, -1)[:, :len(values)]].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def summarize_oos(oos: pd.DataFrame, policy=POLICY) -> dict:
    if oos.empty:
        return {"validated": False, "trade_validated": False, "oos_days": 0,
                "reason": "시간순 검증 표본 부족"}
    work = oos.copy()
    work["net_return"] = work.trade_return - policy.round_trip_cost
    ranked = work.sort_values(["date", "probability", "code"], ascending=[True, False, True])
    picks = ranked.groupby("date", sort=True).head(policy.top_k)
    selected = picks.groupby("date").agg(hit=("target_up", "mean"),
        net=("net_return", "mean"), direction_return=("forward_return", "mean"))
    universe = work.groupby("date").agg(hit=("target_up", "mean"), net=("net_return", "mean"))
    excess = selected.net - universe.net
    hit_lift = selected.hit - universe.hit
    metrics = work[np.isfinite(work.target_up) & np.isfinite(work.probability)].copy()
    if metrics.empty:
        return {"validated": False, "trade_validated": False, "oos_days": 0,
                "reason": "검증 결과 미확정", "unresolved_selected": len(picks)}
    per_day = 1 / metrics.groupby("date")["date"].transform("size")
    auc = float(roc_auc_score(metrics.target_up, metrics.probability, sample_weight=per_day)) if metrics.target_up.nunique() == 2 else 0.5
    brier = float(brier_score_loss(metrics.target_up, metrics.probability, sample_weight=per_day))
    baseline = float(brier_score_loss(metrics.target_up, metrics.baseline_probability, sample_weight=per_day))
    bins = pd.cut(metrics.probability, np.linspace(0, 1, 11), include_lowest=True)
    calibration = []
    ece = 0.0
    for _, group in metrics.groupby(bins, observed=True):
        weights = per_day.loc[group.index]
        predicted = float(np.average(group.probability, weights=weights))
        observed = float(np.average(group.target_up, weights=weights))
        ece += abs(predicted - observed) * weights.sum() / per_day.sum()
        calibration.append({"n": len(group), "predicted": predicted, "observed": observed})
    net_ci = _block_ci(selected.net)
    excess_ci = _block_ci(excess)
    unresolved = int(picks.trade_return.isna().sum())
    coverage = float(work.trade_return.notna().mean())
    complete_coverage = unresolved == 0 and coverage >= .98
    enough = len(selected) >= policy.min_oos_days
    probability_ok = (enough and complete_coverage and auc >= policy.min_auc and brier < baseline
                      and float(hit_lift.mean()) >= policy.min_hit_lift
                      and ece <= policy.max_calibration_error)
    trade_ok = (probability_ok and net_ci[0] is not None and net_ci[0] > 0
                and excess_ci[0] is not None and excess_ci[0] > 0)
    checks = {"sample": enough, "outcome_coverage": complete_coverage, "auc": auc >= policy.min_auc,
              "brier_skill": brier < baseline,
              "top5_hit_lift": float(hit_lift.mean()) >= policy.min_hit_lift,
              "calibration": ece <= policy.max_calibration_error}
    checks = {name: bool(passed) for name, passed in checks.items()}
    return {"validated": bool(probability_ok), "trade_validated": bool(trade_ok),
            "outcome_coverage": coverage, "unresolved_selected": unresolved,
            "return_scope": "관측 가능한 선택 코호트 평균; 미확정 결과가 있으면 검증 실패",
            "oos_days": len(selected), "oos_rows": len(work), "pick_rows": len(picks),
            "period_start": work.date.min().strftime("%Y%m%d"),
            "period_end": work.date.max().strftime("%Y%m%d"),
            "auc": auc, "brier": brier, "baseline_brier": baseline,
            "calibration_error": float(ece), "calibration_bins": calibration,
            "hit_rate": _finite_or_none(selected.hit.mean()), "universe_hit_rate": _finite_or_none(universe.hit.mean()),
            "hit_lift": _finite_or_none(hit_lift.mean()), "net_return_mean": _finite_or_none(selected.net.mean()),
            "universe_net_return_mean": _finite_or_none(universe.net.mean()),
            "net_return_ci95": net_ci, "excess_return_mean": _finite_or_none(excess.mean()),
            "excess_return_ci95": excess_ci,
            "cost_sensitivity": {str(bp): _finite_or_none(selected.net.mean() + policy.round_trip_cost - bp / 10000)
                                 for bp in (20, 30, 40)},
            "trade_checks": {"net_ci_positive": net_ci[0] is not None and net_ci[0] > 0,
                             "excess_ci_positive": excess_ci[0] is not None and excess_ci[0] > 0},
            "trade_reason": "비용 후 수익 우위 통과" if trade_ok else "비용 후 수익 우위 미확인",
            "checks": checks, "reason": "확률 검증 통과" if probability_ok else
                "검증 미달: " + ", ".join(k for k, passed in checks.items() if not passed)}


def train_and_predict(panel: pd.DataFrame, asof_ymd: str, policy=POLICY):
    """Returns research/validated predictions and a fully OOS audit report."""
    asof = pd.Timestamp(asof_ymd)
    eligible = eligible_panel(panel.loc[panel.date <= asof], policy)
    eligible["target_up"] = np.where(eligible.trade_return.notna(),
                                      (eligible.trade_return > policy.round_trip_cost).astype(float), np.nan)
    eligible["_label_ready"] = eligible.label_end
    if "label_available_on" in eligible:
        eligible["_label_ready"] = eligible[["label_end", "label_available_on"]].max(axis=1, skipna=False)
    known = eligible[eligible._label_ready.le(asof) & eligible.target_up.notna()
                     & eligible.trade_return.notna()].copy()
    known = known[np.isfinite(known.trade_return)]
    mature = eligible[eligible.label_end.le(asof)]
    days = pd.DatetimeIndex(sorted(mature.date.unique()))
    first = policy.min_train_days + policy.calibration_days + policy.purge_sessions
    starts = list(range(first, len(days), policy.test_block_days))[-policy.max_folds:]
    predictions, folds = [], []
    for start in starts:
        stop = min(start + policy.test_block_days, len(days)) - 1
        train, cal, _ = purged_split(known, days[start], days[stop], policy)
        # Rank the complete signal-time universe BEFORE inspecting outcomes.
        test = mature[(mature.date >= days[start]) & (mature.date <= days[stop])].copy()
        if train.empty or cal.empty or test.empty:
            continue
        try:
            model = _fit(train, cal)
        except ValueError as exc:
            logger.warning("스윙 모델 fold 생략: %s", exc)
            continue
        result = test[["date", "code", "target_up", "forward_return", "trade_return"]].copy()
        result.loc[~test._label_ready.le(asof),
                   ["target_up", "forward_return", "trade_return"]] = np.nan
        result["probability"] = _predict(model, test)
        result["baseline_probability"] = cal.groupby("date").target_up.mean().mean()
        predictions.append(result)
        folds.append({"train_label_available": train._label_ready.max().strftime("%Y%m%d"),
                      "cal_label_available": cal._label_ready.max().strftime("%Y%m%d"),
                      "train_label_end": train.label_end.max().strftime("%Y%m%d"),
                      "cal_start": cal.date.min().strftime("%Y%m%d"),
                      "cal_label_end": cal.label_end.max().strftime("%Y%m%d"),
                      "test_start": test.date.min().strftime("%Y%m%d"),
                      "test_end": test.date.max().strftime("%Y%m%d")})
    oos = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    report = summarize_oos(oos, policy)
    report.update({"label_quality": panel.attrs.get("label_quality", {}), "version": MODEL_VERSION, "asof": asof.strftime("%Y%m%d"),
                   "target_definition": TARGET_DEFINITION, "policy": asdict(policy),
                   "features": list(FEATURES), "folds": folds,
                   "execution_definition": "익일 시가 진입→5번째 보유 거래일 종가, -8% 손절·갭 체결·비용 반영",
                   "universe_definition": "날짜별 저장 추천 모집단, 유동성 기준 적용",
                   "limitations": ["저장 추천 모집단 밖 종목은 평가하지 않음",
                                    "중첩 보유 코호트 평균수익, 계좌 누적수익이 아님",
                                    "시가 체결 가정, 거래비용·슬리피지 왕복0.3% 시나리오"]})
    current = eligible[eligible.date.eq(asof)].copy()
    fit_days = pd.DatetimeIndex(sorted(known.date.unique()))
    if len(fit_days) < first or current.empty:
        report.update(validated=False, trade_validated=False, reason="학습 이력 또는 당일 가격 부족")
        return pd.DataFrame(), report, oos
    # Deployment calibrator uses mature labels through asof; no unknown label is filled.
    cal_start = fit_days[-policy.calibration_days]
    train = known[(known.date < cal_start) & (known._label_ready < cal_start)]
    cal = known[known.date >= cal_start]
    report["live_fit"] = {"train_days": train.date.nunique(), "calibration_days": cal.date.nunique(),
                          "train_label_available": str(train._label_ready.max()),
                          "calibration_start": str(cal.date.min()),
                          "calibration_label_available": str(cal._label_ready.max())}
    if (train.date.nunique() < policy.min_train_days
            or cal.date.nunique() < policy.min_calibration_days):
        report.update(validated=False, trade_validated=False, reason="라벨 누수 제거 후 학습/보정 이력 부족")
        return pd.DataFrame(), report, oos
    try:
        model = _fit(train, cal)
    except ValueError as exc:
        report.update(validated=False, trade_validated=False, reason=str(exc))
        return pd.DataFrame(), report, oos
    current["probability"] = _predict(model, current)
    current = current.sort_values(["probability", "code"], ascending=[False, True]).copy()
    current["rank"] = np.arange(1, len(current) + 1)
    current["score"] = current.probability.rank(pct=True, method="average") * 100
    current["reason"] = [f"오늘 {r.ret_1d:+.1%} · 5일 {r.ret_5d:+.1%} · 거래량 {r.relative_volume_20d:.1f}배"
                         for r in current.itertuples()]
    return current[["code", "probability", "rank", "score", "reason"]], report, oos


def _atomic_text(path: Path, text: str, *, overwrite=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
            name = f.name
            f.write(text)
        if overwrite:
            os.replace(name, path)
        else:
            try:
                os.link(name, path)
            except FileExistsError:
                logger.debug("최초 스윙 발표 보존: %s", path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def run(data_dir="data", asof_ymd=None, policy=POLICY, current_codes=None):
    if not asof_ymd:
        raise ValueError("명시적인 종가 기준일이 필요합니다")
    panel = load_feature_panel(data_dir, asof_ymd, current_codes=current_codes)
    picks, report, oos = train_and_predict(panel, asof_ymd, policy)
    directory = Path(data_dir)
    _atomic_text(directory / "swing_predictions_latest.csv", picks.to_csv(index=False))
    _atomic_text(directory / "swing_validation_latest.json",
                 json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if not oos.empty:
        _atomic_text(directory / ".swing_cache" / "oos_latest.csv", oos.to_csv(index=False))
        ranked = oos.sort_values(["date", "probability", "code"], ascending=[True, False, True])
        top = ranked.groupby("date").head(policy.top_k)
        daily = top.groupby("date").agg(selected=("code", "size"), known=("trade_return", "count"),
            hit_rate=("target_up", "mean"), gross_return=("trade_return", "mean"))
        daily["net_return"] = daily.gross_return - policy.round_trip_cost
        daily["universe_net_return"] = oos.groupby("date").trade_return.mean() - policy.round_trip_cost
        daily["excess_return"] = daily.net_return - daily.universe_net_return
        _atomic_text(directory / "swing_oos_daily_latest.csv", daily.to_csv())
    snapshot = json.dumps(
        {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
         "validation": report, "predictions": picks.to_dict("records")},
        ensure_ascii=False, indent=2, allow_nan=False)
    _atomic_text(directory / "swing_snapshot_latest.json", snapshot)
    archive = directory / f"swing_snapshot_{asof_ymd}.json"
    # Freeze the first publication for prospective performance checks.
    # A rerun may update latest, but must not rewrite yesterday's decisions.
    if not picks.empty:
        _atomic_text(archive, snapshot, overwrite=False)
    return picks, report
