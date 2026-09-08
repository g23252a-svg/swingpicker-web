"""Monitor frozen swing forecasts without retraining, reselection or trading.

The figures describe overlapping forecast cohorts, never an account equity
curve or recorded executions. Unknown selected outcomes remain in the cohort;
the sixth-ranked stock cannot replace an unobserved top-five outcome.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.swing_features import HOLD_SESSIONS, STOP_PCT, load_feature_panel

LOGGER = logging.getLogger(__name__)
_SNAPSHOT = re.compile(r"^swing_snapshot_(\d{8})\.json$")
_PRICES = re.compile(r"^ohlcv_cache_(\d{8})\.parquet$")


def _date(value):
    result = pd.to_datetime(str(value), errors="coerce")
    return None if pd.isna(result) else result.normalize().tz_localize(None)


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _code(value):
    text = re.sub(r"\.0$", "", str(value).strip().upper()).zfill(6)
    return text if re.fullmatch(r"[0-9A-Z]{6}", text) else None


def _files(directory, pattern):
    found = []
    for path in directory.iterdir() if directory.exists() else []:
        match = pattern.fullmatch(path.name)
        if match:
            date = _date(match.group(1))
            if date is not None:
                found.append((date, path))
    return sorted(found)


def _observed_sessions(directory, asof, panel):
    """Use actual OHLCV index dates, including sessions without recommendations."""
    files = [(day, path) for day, path in _files(directory, _PRICES) if day <= asof]
    for day, path in reversed(files):
        try:
            indexed = pd.read_parquet(path, columns=[])
            if isinstance(indexed.index, pd.DatetimeIndex):
                dates = pd.DatetimeIndex(indexed.index).normalize().tz_localize(None)
            else:
                raw = pd.read_parquet(path)
                column = next((name for name in ("date", "Date", "날짜", "일자") if name in raw), None)
                if column is None:
                    continue
                dates = pd.DatetimeIndex(pd.to_datetime(raw[column], errors="coerce")).normalize().tz_localize(None)
            dates = dates[dates.notna() & (dates <= day) & (dates <= asof)]
            if len(dates):
                return dates.unique().sort_values()
        except (OSError, ValueError, ImportError, KeyError) as exc:
            LOGGER.warning("Cannot read market-session dates from %s: %s", path.name, exc)
    # Recommendation membership is not a complete market calendar. Returning
    # no calendar keeps entry timing unknown instead of guessing a weekday.
    return pd.DatetimeIndex([])


def _timing_status(generated_at, entry_day):
    if not generated_at:
        return "unknown_timestamp"
    try:
        generated = pd.Timestamp(generated_at)
    except (ValueError, TypeError):
        return "unknown_timestamp"
    if pd.isna(generated) or generated.tzinfo is None:
        return "unknown_timestamp"
    if entry_day is None:
        return "pending_entry_session"
    entry_open = entry_day.tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)
    return "prospective" if generated <= entry_open else "retrospective"


def _summary(rows):
    completed = [row for row in rows if row["outcome_status"] == "completed"]
    returns = [row["net_return"] for row in completed]
    return {
        "selected": len(rows), "completed": len(completed),
        "pending": sum(row["outcome_status"] == "pending" for row in rows),
        "unresolved": sum(row["outcome_status"] == "unresolved" for row in rows),
        "known_cohort_mean_net_return": sum(returns) / len(returns) if returns else None,
        "known_cohort_hit_rate": sum(value > 0 for value in returns) / len(returns) if returns else None,
        "stop_threshold_loss_count": sum(row["trade_return"] <= STOP_PCT + 1e-9 for row in completed),
        "research_forecasts": sum(row["forecast_mode"] == "research" for row in rows),
    }


def _selected_predictions(snapshot, top_k):
    raw = snapshot.get("predictions")
    if not isinstance(raw, list):
        raise ValueError("predictions must be an archived list")
    selected, codes, ranks = [], set(), set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("prediction record is not an object")
        code, rank = _code(item.get("code")), _number(item.get("rank"))
        if code is None or rank is None or rank < 1 or rank != int(rank):
            raise ValueError("prediction code/rank is invalid")
        if code in codes or rank in ranks:
            raise ValueError("duplicate archived code/rank makes original selection ambiguous")
        codes.add(code)
        ranks.add(rank)
        if rank <= top_k:
            selected.append({**item, "code": code, "rank": int(rank)})
    return sorted(selected, key=lambda item: item["rank"])


def build_performance_report(data_dir="data", asof_ymd=None):
    directory = Path(data_dir)
    archives = _files(directory, _SNAPSHOT)
    if asof_ymd is None:
        available = _files(directory, _PRICES) or archives
        if not available:
            raise ValueError("No dated OHLCV cache or frozen swing forecast is available")
        asof = available[-1][0]
    else:
        asof = _date(asof_ymd)
        if asof is None:
            raise ValueError("asof must be a valid YYYYMMDD or ISO date")
    panel = load_feature_panel(directory, asof.strftime("%Y%m%d"))
    sessions = _observed_sessions(directory, asof, panel)
    outcomes = {}
    for row in panel.to_dict("records"):
        code, day = _code(row.get("code")), _date(row.get("date"))
        if code and day is not None:
            outcomes[(day, code)] = row
    result_days, all_rows, errors = [], [], []
    for signal_day, path in archives:
        if signal_day > asof:
            continue
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            validation = snapshot["validation"]
            if not isinstance(validation, dict) or _date(validation.get("asof")) != signal_day:
                raise ValueError("archive filename and validation.asof must match")
            policy = validation.get("policy", {})
            if not isinstance(policy, dict):
                raise ValueError("archived policy must be an object")
            top_k = _number(policy.get("top_k", 5))
            cost = _number(policy.get("round_trip_cost", .003))
            if top_k is None or top_k < 1 or top_k != int(top_k) or cost is None or not 0 <= cost < 1:
                raise ValueError("archived selection/cost policy is invalid")
            selected = _selected_predictions(snapshot, int(top_k))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({"file": path.name, "reason": str(exc)})
            continue
        after = sessions[sessions > signal_day]
        entry_day = after[0] if len(after) else None
        expected_end = after[HOLD_SESSIONS - 1] if len(after) >= HOLD_SESSIONS else None
        timing = _timing_status(snapshot.get("generated_at_utc"), entry_day)
        probability_gate = validation.get("validated") is True
        trade_gate = validation.get("trade_validated") is True
        forecast_mode = "validated_forecast" if probability_gate and trade_gate else "research"
        daily_rows = []
        for item in selected:
            observed = outcomes.get((signal_day, item["code"]), {})
            label_end = _date(observed.get("label_end")) or expected_end
            label_available = _date(observed.get("label_available_on"))
            gross = _number(observed.get("trade_return"))
            completed_window = label_end is not None and label_end <= asof
            ready = completed_window and (label_available is None or label_available <= asof)
            if ready and gross is not None:
                outcome_status, reason = "completed", "observed mature outcome"
            elif completed_window and (label_available is None or label_available <= asof):
                outcome_status, reason = "unresolved", "mature selected forecast has no executable observed outcome"
            else:
                outcome_status, reason = "pending", "holding window or outcome availability is not yet complete"
            known = outcome_status == "completed"
            record = {
                "signal_date": signal_day.strftime("%Y%m%d"), "code": item["code"], "rank": item["rank"],
                "probability": _number(item.get("probability")), "score": _number(item.get("score")),
                "reason": str(item.get("reason", "")), "forecast_mode": forecast_mode,
                "forecast_timing": timing, "is_recorded_trade": False,
                "outcome_status": outcome_status, "outcome_reason": reason,
                "entry_date": entry_day.strftime("%Y%m%d") if entry_day is not None else None,
                "label_end": label_end.strftime("%Y%m%d") if label_end is not None else None,
                "label_available_on": label_available.strftime("%Y%m%d") if label_available is not None else None,
                "round_trip_cost": cost, "trade_return": gross if known else None,
                "net_return": gross - cost if known else None,
                "net_profit": bool(gross > cost) if known else None,
            }
            daily_rows.append(record)
        result_days.append({
            "signal_date": signal_day.strftime("%Y%m%d"), "version": validation.get("version"),
            "generated_at_utc": snapshot.get("generated_at_utc"), "forecast_timing": timing,
            "validated": probability_gate, "trade_validated": trade_gate, "forecast_mode": forecast_mode,
            "top_k": int(top_k), "round_trip_cost": cost, "summary": _summary(daily_rows), "picks": daily_rows,
        })
        all_rows.extend(daily_rows)
    prospective = [row for row in all_rows if row["forecast_timing"] == "prospective"]
    retrospective = [row for row in all_rows if row["forecast_timing"] == "retrospective"]
    unknown = [row for row in all_rows if row["forecast_timing"] not in ("prospective", "retrospective")]
    return {
        "asof": asof.strftime("%Y%m%d"), "forecast_days": len(result_days),
        "first_forecast_date": result_days[0]["signal_date"] if result_days else None,
        "last_forecast_date": result_days[-1]["signal_date"] if result_days else None,
        "summary": _summary(all_rows), "prospective_summary": _summary(prospective),
        "retrospective_summary": _summary(retrospective), "timing_unknown_summary": _summary(unknown),
        "execution_definition": "익일 시가 진입, 진입일 포함 최대 5거래일, -8% 손절과 갭 체결, 저장된 왕복비용 반영",
        "return_units": "fraction; 0.01 = 1%", "days": result_days, "archive_errors": errors,
        "limitations": [
            "저장 당시 순위를 그대로 평가한 예측 코호트이며 실제 체결·공식 매수 내역이 아닙니다.",
            "보유 기간이 겹치는 코호트 평균으로 계좌 누적수익·연환산수익을 뜻하지 않습니다.",
            "미성숙·미확인 결과는 수익률과 승률에서 제외되며 손실 0%로 대체하지 않습니다.",
            "미확인 상위 종목을 차순위로 교체하지 않습니다. 알려진 결과만의 평균에는 결측 편향이 남습니다.",
            "생성 시각이 실제 익일 장 시작 이전임을 확인한 예측만 prospective_summary에 포함합니다.",
            "두 검증 기준 중 하나라도 실패한 저장 예측은 연구용이며, 통과한 예측도 실제 거래로 간주하지 않습니다.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--asof", help="Observation cutoff YYYYMMDD; defaults to latest dated price cache")
    parser.add_argument("--output", type=Path, help="Optional JSON output path; otherwise prints JSON")
    args = parser.parse_args(argv)
    result = build_performance_report(args.data_dir, args.asof)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        print(content)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(content)
        os.replace(temporary, args.output)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
