#!/usr/bin/env python3
"""Reproduce one fixed expected-net-return experiment; never train a live model.

Usage from the repository root:
    python scripts/audit_swing_expected_return.py --data-dir data --asof 20260907

The output is a research comparison, not evidence to promote this strategy.
Do not tune parameters against this already-inspected OOS period. OHLCV loading
may refresh derived caches, but no raw source or production model is replaced.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--asof", default="20260907")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    # Works from /tmp during review and from scripts/ after adoption.
    candidates = (data_dir.parent, Path(__file__).resolve().parents[1], Path.cwd())
    for candidate in candidates:
        if (candidate / "services" / "swing_features.py").exists():
            sys.path.insert(0, str(candidate))
            break
    from services.swing_features import FEATURES, load_feature_panel
    from services.swing_model import SwingPolicy, eligible_panel, purged_split, _block_ci

    # Frozen before the original experiment; do not search these values.
    policy = SwingPolicy(
        min_train_days=60, purge_sessions=5, min_calibration_days=10,
        calibration_days=15, test_block_days=20, max_folds=4,
        min_oos_days=40, min_auc=0.53, min_hit_lift=0.03,
        max_calibration_error=0.06, top_k=5, round_trip_cost=0.003,
        min_turnover=1_000_000_000, max_day_move=0.25,
    )
    started = time.time()
    asof = pd.Timestamp(args.asof)
    stamp = asof.strftime("%Y%m%d")
    panel = load_feature_panel(str(data_dir), stamp)
    print(f"Panel: {len(panel)} rows / {panel.date.nunique()} dates", flush=True)
    eligible = eligible_panel(panel.loc[panel.date <= asof], policy)
    eligible["target_up"] = np.where(
        eligible.trade_return.notna(),
        (eligible.trade_return > policy.round_trip_cost).astype(float), np.nan,
    )
    eligible["_label_ready"] = eligible[["label_end", "label_available_on"]].max(
        axis=1, skipna=False
    )
    known = eligible[eligible._label_ready.le(asof) & np.isfinite(eligible.trade_return)].copy()
    mature = eligible[eligible.label_end.le(asof)].copy()
    dates = pd.DatetimeIndex(sorted(mature.date.unique()))
    first = policy.min_train_days + policy.calibration_days + policy.purge_sessions
    starts = list(range(first, len(dates), policy.test_block_days))[-policy.max_folds:]

    def x(frame):
        return frame[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)

    def day_weights(frame):
        weight = 1 / frame.groupby("date").date.transform("size")
        return weight / weight.mean()

    def metrics(frame):
        # Select before examining future outcomes: missing trades cannot be
        # silently replaced by lower-ranked, observable winners.
        ranked = frame.sort_values(["date", "predicted_net", "code"], ascending=[True, False, True])
        picks = ranked.groupby("date").head(policy.top_k)
        selected = picks.groupby("date").agg(net=("net", "mean"), hit=("target_up", "mean"))
        universe = frame.groupby("date").agg(net=("net", "mean"), hit=("target_up", "mean"))
        observed = frame[np.isfinite(frame.net)]
        weights = 1 / observed.groupby("date").date.transform("size")
        ics = []
        for _, group in observed.groupby("date"):
            if len(group) >= 20 and group.predicted_net.nunique() > 1 and group.net.nunique() > 1:
                ics.append(float(group.predicted_net.corr(group.net, method="spearman")))
        auc = (float(roc_auc_score(observed.target_up, observed.predicted_net, sample_weight=weights))
               if observed.target_up.nunique() == 2 else None)
        return {
            "days": int(frame.date.nunique()), "rows": len(frame), "selected_rows": len(picks),
            "selected_unresolved": int(picks.net.isna().sum()),
            "overall_outcome_coverage": float(frame.net.notna().mean()),
            "observed_selected_net_mean": float(selected.net.mean()),
            "observed_selected_hit_rate": float(selected.hit.mean()),
            "observed_universe_net_mean": float(universe.net.mean()),
            "observed_universe_hit_rate": float(universe.hit.mean()),
            "observed_excess_mean": float((selected.net - universe.net).mean()),
            "selected_net_block_ci95": _block_ci(selected.net),
            "excess_block_ci95": _block_ci(selected.net - universe.net),
            "mean_daily_ic": float(np.mean(ics)) if ics else None,
            "profit_direction_auc": auc,
            "predicted_net_mean": float(np.average(observed.predicted_net, weights=weights)),
            "observed_net_mean": float(np.average(observed.net, weights=weights)),
            "mean_absolute_error": float(np.average(abs(observed.predicted_net - observed.net), weights=weights)),
            "mean_baseline_absolute_error": float(np.average(abs(observed.baseline_net - observed.net), weights=weights)),
        }

    outputs, folds = [], []
    for number, position in enumerate(starts, 1):
        stop = min(position + policy.test_block_days, len(dates)) - 1
        train, calibration, _ = purged_split(known, dates[position], dates[stop], policy)
        test = mature[(mature.date >= dates[position]) & (mature.date <= dates[stop])].copy()
        if train.empty or calibration.empty or test.empty:
            continue
        ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
        tree = HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=7, max_depth=3, min_samples_leaf=80,
            l2_regularization=10.0, learning_rate=0.05, early_stopping=False, random_state=19,
        )
        weights = day_weights(train)
        target = train.trade_return - policy.round_trip_cost
        with threadpool_limits(limits=2):
            ridge.fit(x(train), target, ridge__sample_weight=weights)
            tree.fit(x(train), target, sample_weight=weights)
            calibration_raw = (ridge.predict(x(calibration)) + tree.predict(x(calibration))) / 2
            # A held-out mean-bias adjustment changes expected-return levels,
            # never stock ranks. No sign reversal or threshold selection.
            bias = float(np.average(
                calibration.trade_return - policy.round_trip_cost - calibration_raw,
                weights=day_weights(calibration),
            ))
            prediction = (ridge.predict(x(test)) + tree.predict(x(test))) / 2 + bias
        result = test[["date", "code", "trade_return", "target_up"]].copy()
        result.loc[~test._label_ready.le(asof), ["trade_return", "target_up"]] = np.nan
        result["net"] = result.trade_return - policy.round_trip_cost
        result["predicted_net"] = prediction
        result["baseline_net"] = float(calibration.groupby("date").trade_return.mean().mean() - policy.round_trip_cost)
        result["fold"] = number
        outputs.append(result)
        fold = {
            "fold": number, "train_days": int(train.date.nunique()),
            "calibration_days": int(calibration.date.nunique()),
            "train_label_available": train._label_ready.max().strftime("%Y%m%d"),
            "cal_start": calibration.date.min().strftime("%Y%m%d"),
            "cal_label_available": calibration._label_ready.max().strftime("%Y%m%d"),
            "test_start": test.date.min().strftime("%Y%m%d"),
            "test_end": test.date.max().strftime("%Y%m%d"),
            "calibration_mean_bias": bias, **metrics(result),
        }
        folds.append(fold)
        print(json.dumps(fold, ensure_ascii=False), flush=True)
    if not outputs:
        raise SystemExit("Insufficient history for the fixed purged experiment")
    oos = pd.concat(outputs, ignore_index=True)
    summary = metrics(oos)
    reference_path = data_dir / "swing_validation_latest.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    report = {
        "asof": stamp, "policy": asdict(policy), "features": list(FEATURES),
        "specification": "Ridge alpha100 + HGB squared-error depth3 leaf7 minleaf80 iter100 L2=10 lr=.05, equal weights; daily equal sample totals; calibration mean-bias correction only",
        "target": "D+1 open to D+5 close, -8% stop with actual opening gap execution, 0.003 cost deducted",
        "selection": "Top5 ranked before missing outcomes filtered; no rank threshold or hyperparameter search",
        "overall": summary, "folds": folds, "probability_model_reference": reference,
        "limitations": [
            "Exploratory comparison after observing the failed probability model; not untouched holdout approval",
            "Unknown selected outcomes disqualify any profitability claim",
            "Overlapping holding-cohort means, not portfolio compounding",
        ],
        "seconds": round(time.time() - started, 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"swing_expected_return_oos_{stamp}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    oos.to_csv(args.output_dir / f"swing_expected_return_oos_{stamp}.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
