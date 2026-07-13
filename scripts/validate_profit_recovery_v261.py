#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the v26.1 fill-aware profit-recovery validation.

This is deliberately narrower than the dashboard backtest.  It evaluates the
exact production candidate mask and ranking, one signal per day, using only
future OHLC bars after the signal date.

Execution assumptions
---------------------
* recommended limit must be touched within the next three sessions;
* an opening gap above +2% is not chased;
* the complete exit horizon is ten sessions after the signal;
* stop -6%, target +10%, round-trip cost 0.22%;
* if stop and target are both touched on one bar, stop wins;
* unfilled orders count as zero-return signal days, not profitable trades.

The report is evidence for a small, guarded deployment—not a return promise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.recommendation_quality import (  # noqa: E402
    MIN_MARKET_BREADTH,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    profit_recovery_candidate_mask,
    profit_recovery_rank_score,
)


FILL_WINDOW_DAYS = 3
HOLD_DAYS = 10
COST_PCT = 0.22
MAX_OPEN_GAP_PCT = 2.0
TRAIN_RATIO = 0.65


def _code(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits.zfill(6)[-6:] if digits else ""


def _date_from_path(path: Path) -> str | None:
    match = re.fullmatch(r"recommend_(\d{8})\.csv", path.name)
    return match.group(1) if match else None


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def load_daily_signals(data_dir: Path, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    eligible_dates = 0
    total_dates = 0
    for path in sorted(data_dir.glob("recommend_*.csv")):
        signal_date = _date_from_path(path)
        if signal_date is None or not (start <= signal_date <= end):
            continue
        frame = pd.read_csv(path, dtype={"종목코드": str}, low_memory=False)
        if frame.empty or "종목코드" not in frame.columns:
            continue
        total_dates += 1
        setup = profit_recovery_candidate_mask(frame)
        breadth = pd.to_numeric(
            frame.get("MARKET_BREADTH", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0)
        candidates = frame[setup & breadth.ge(MIN_MARKET_BREADTH)].copy()
        if candidates.empty:
            continue
        eligible_dates += 1
        candidates["_policy_rank"] = profit_recovery_rank_score(candidates)
        candidates["_row_order"] = np.arange(len(candidates))
        winner = candidates.sort_values(
            ["_policy_rank", "DISPLAY_SCORE", "_row_order"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0]
        rows.append(
            {
                "signal_date": signal_date,
                "code": _code(winner["종목코드"]),
                "name": str(winner.get("종목명", "")),
                "entry": _num(winner, "추천매수가"),
                "rank_score": _num(winner, "_policy_rank"),
                "breadth": _num(winner, "MARKET_BREADTH"),
            }
        )
    cadence = {
        "available_signal_dates": total_dates,
        "eligible_signal_dates": eligible_dates,
        "cadence_pct": round(eligible_dates / total_dates * 100, 2) if total_dates else 0.0,
    }
    return pd.DataFrame(rows), cadence


def load_price_histories(data_dir: Path, codes: set[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Use the newest cache containing each selected code.

    A single latest universe can omit a historical candidate that later lost
    liquidity.  Walking caches newest-to-oldest avoids silently converting it
    to an unfilled order.
    """
    histories: dict[str, pd.DataFrame] = {}
    sources: list[str] = []
    unresolved = set(codes)
    for path in sorted(data_dir.glob("ohlcv_cache_*.parquet"), reverse=True):
        if not unresolved:
            break
        frame = pd.read_parquet(path)
        if frame.empty or "종목코드" not in frame.columns:
            continue
        normalized = frame["종목코드"].map(_code)
        present = unresolved.intersection(set(normalized.unique()))
        if not present:
            continue
        frame = frame.copy()
        # The OHLC cache index is a repeated trading date, so .loc with the
        # normalized Series would create a many-to-many expansion.  Assign by
        # position first, then filter.
        frame["code"] = normalized.to_numpy()
        work = frame.loc[frame["code"].isin(present)].copy()
        work = work.reset_index()
        date_col = "Date" if "Date" in work.columns else work.columns[0]
        work["date"] = pd.to_datetime(work[date_col], errors="coerce")
        for code, group in work.groupby("code"):
            histories[code] = group.sort_values("date").drop_duplicates("date", keep="last")
        unresolved -= present
        sources.append(path.name)
    return histories, sorted(unresolved)


def simulate_signal(signal: pd.Series, history: pd.DataFrame | None) -> dict:
    base = signal.to_dict()
    base.update(
        {
            "fill_status": "UNKNOWN",
            "fill_date": "",
            "exit_date": "",
            "exit_type": "NO_DATA",
            "net_return_pct": np.nan,
        }
    )
    entry = float(signal["entry"])
    if history is None or history.empty or entry <= 0:
        return base

    signal_dt = pd.to_datetime(signal["signal_date"], format="%Y%m%d")
    future = history[history["date"] > signal_dt].head(HOLD_DAYS).copy()
    if len(future) < HOLD_DAYS:
        base["exit_type"] = "IMMATURE"
        return base

    fill_row_index = None
    fill_price = None
    for idx, row in future.head(FILL_WINDOW_DAYS).iterrows():
        day_open = _num(row, "시가")
        day_low = _num(row, "저가", np.inf)
        if day_open > entry * (1 + MAX_OPEN_GAP_PCT / 100):
            continue
        if day_low <= entry:
            fill_row_index = idx
            fill_price = min(day_open, entry) if day_open > 0 else entry
            break

    if fill_row_index is None or fill_price is None:
        base.update({"fill_status": "UNFILLED", "exit_type": "UNFILLED", "net_return_pct": 0.0})
        return base

    fill_pos = int(np.flatnonzero(future.index.to_numpy() == fill_row_index)[0])
    exit_window = future.iloc[fill_pos:]
    stop_price = entry * (1 - STOP_LOSS_PCT / 100)
    target_price = entry * (1 + TAKE_PROFIT_PCT / 100)
    fill_date = pd.Timestamp(exit_window.iloc[0]["date"])
    exit_date = pd.Timestamp(exit_window.iloc[-1]["date"])
    exit_price = _num(exit_window.iloc[-1], "종가")
    exit_type = "TIME"

    for _, row in exit_window.iterrows():
        day_open = _num(row, "시가")
        day_low = _num(row, "저가", np.inf)
        day_high = _num(row, "고가")
        exit_date = pd.Timestamp(row["date"])
        if day_open > 0 and day_open <= stop_price:
            exit_price = day_open
            exit_type = "GAP_STOP"
            break
        stop_hit = day_low <= stop_price
        target_hit = day_high >= target_price
        if stop_hit:
            exit_price = stop_price
            exit_type = "STOP_SAME_BAR" if target_hit else "STOP"
            break
        if target_hit:
            exit_price = target_price
            exit_type = "TARGET"
            break

    net = (exit_price / fill_price - 1) * 100 - COST_PCT
    base.update(
        {
            "fill_status": "FILLED",
            "fill_date": fill_date.strftime("%Y%m%d"),
            "fill_price": round(fill_price, 2),
            "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2),
            "exit_date": exit_date.strftime("%Y%m%d"),
            "exit_price": round(exit_price, 2),
            "exit_type": exit_type,
            "net_return_pct": round(float(net), 4),
        }
    )
    return base


def _longest_loss_streak(values: pd.Series) -> int:
    longest = current = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"signal_days": 0, "filled": 0, "avg_filled_net_pct": None}
    returns = pd.to_numeric(trades["net_return_pct"], errors="coerce").fillna(0)
    filled = trades[trades["fill_status"].eq("FILLED")].copy()
    filled_returns = pd.to_numeric(filled["net_return_pct"], errors="coerce").dropna()
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "signal_days": int(len(trades)),
        "filled": int(len(filled)),
        "fill_rate_pct": round(len(filled) / len(trades) * 100, 2),
        "avg_signal_day_net_pct": round(float(returns.mean()), 4),
        "avg_filled_net_pct": round(float(filled_returns.mean()), 4) if len(filled_returns) else None,
        "median_filled_net_pct": round(float(filled_returns.median()), 4) if len(filled_returns) else None,
        "win_rate_pct": round(float((filled_returns > 0).mean() * 100), 2) if len(filled_returns) else None,
        "sum_signal_net_pct": round(float(returns.sum()), 4),
        "max_drawdown_pct_points": round(float(drawdown.min()), 4),
        "longest_loss_streak": _longest_loss_streak(filled_returns),
        "best_trade_pct": round(float(filled_returns.max()), 4) if len(filled_returns) else None,
        "worst_trade_pct": round(float(filled_returns.min()), 4) if len(filled_returns) else None,
    }


def bootstrap_mean_ci(values: pd.Series, iterations: int = 5000) -> list[float] | None:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) < 2:
        return None
    rng = np.random.default_rng(261)
    means = rng.choice(array, size=(iterations, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [round(float(low), 4), round(float(high), 4)]


def build_report(trades: pd.DataFrame, cadence: dict, start: str, end: str, unresolved: list[str]) -> dict:
    ordered = trades.sort_values("signal_date").reset_index(drop=True)
    split_idx = max(1, int(len(ordered) * TRAIN_RATIO))
    train = ordered.iloc[:split_idx]
    test = ordered.iloc[split_idx:]
    full_summary = summarize(ordered)
    train_summary = summarize(train)
    test_summary = summarize(test)

    monthly = {}
    for month, group in ordered.groupby(ordered["signal_date"].str[:6]):
        monthly[str(month)] = summarize(group)

    filled_returns = ordered.loc[
        ordered["fill_status"].eq("FILLED"), "net_return_pct"
    ]
    train_avg = train_summary.get("avg_filled_net_pct")
    test_avg = test_summary.get("avg_filled_net_pct")
    enough = full_summary.get("filled", 0) >= 30 and test_summary.get("filled", 0) >= 10
    both_positive = (
        train_avg is not None and test_avg is not None and train_avg > 0 and test_avg > 0
    )
    deployment_status = "LIMITED_DEPLOYMENT" if enough and both_positive else "RESEARCH_ONLY"

    return {
        "version": "profit_recovery_validation_v261",
        "deployment_status": deployment_status,
        "warning": "과거 수익은 미래 수익을 보장하지 않으며 표본이 작아 종목당 2~3% 비중만 허용",
        "period": {"start": start, "mature_end": end},
        "execution": {
            "fill_window_days": FILL_WINDOW_DAYS,
            "max_open_gap_pct": MAX_OPEN_GAP_PCT,
            "horizon_days_after_signal": HOLD_DAYS,
            "stop_loss_pct": STOP_LOSS_PCT,
            "take_profit_pct": TAKE_PROFIT_PCT,
            "round_trip_cost_pct": COST_PCT,
            "same_bar_policy": "STOP_FIRST",
            "unfilled_signal_return_pct": 0.0,
        },
        "selection": {
            "max_daily_picks": 1,
            "minimum_market_breadth_pct": MIN_MARKET_BREADTH,
            "ranking": "30% DISPLAY + 25% FINAL + 20% STRUCT + 10% TIMING + 10% BUY_NOW + 5% MFI",
            "legacy_macro_note": "과거 CRITICAL 라벨은 절대 환율 수준 버그가 있어 핵심 백테스트에서 제외; v26.1은 실제 5일 환율 급등만 별도 차단",
        },
        "cadence": cadence,
        "split": {
            "method": "chronological_unique_signal_days",
            "train_ratio": TRAIN_RATIO,
            "train": train_summary,
            "test": test_summary,
        },
        "full": full_summary,
        "monthly": monthly,
        "bootstrap_filled_mean_95pct_ci": bootstrap_mean_ci(filled_returns),
        "unresolved_price_codes": unresolved,
        "gates": {
            "minimum_samples": enough,
            "train_and_test_mean_positive": both_positive,
            "cadence_not_permanently_zero": cadence.get("cadence_pct", 0) >= 40,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out-dir", default=str(ROOT / "reports"))
    parser.add_argument("--start", default="20260410")
    parser.add_argument("--end", default="", help="미입력 시 최신 OHLC에서 10거래일 전")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    cache_files = sorted(data_dir.glob("ohlcv_cache_*.parquet"))
    if not cache_files:
        print("OHLCV cache not found", file=sys.stderr)
        return 2
    latest = pd.read_parquet(cache_files[-1], columns=["종목코드"])
    market_dates = pd.DatetimeIndex(latest.index).drop_duplicates().sort_values()
    if len(market_dates) <= HOLD_DAYS:
        print("Not enough market dates", file=sys.stderr)
        return 2
    mature_end = args.end or market_dates[-(HOLD_DAYS + 1)].strftime("%Y%m%d")

    signals, cadence = load_daily_signals(data_dir, args.start, mature_end)
    if signals.empty:
        print("No eligible policy signals", file=sys.stderr)
        return 2
    histories, unresolved = load_price_histories(data_dir, set(signals["code"]))
    trades = pd.DataFrame(
        [simulate_signal(row, histories.get(row["code"])) for _, row in signals.iterrows()]
    )
    report = build_report(trades, cadence, args.start, mature_end, unresolved)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "profit_recovery_v261_validation.json"
    trades_path = out_dir / "profit_recovery_v261_trades.csv"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"trades={trades_path}")
    return 0 if report["deployment_status"] == "LIMITED_DEPLOYMENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
