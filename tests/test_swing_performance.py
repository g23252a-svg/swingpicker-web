"""Frozen rankings, censoring and prospective timing must survive monitoring."""
import json

import pandas as pd
import pytest

from scripts import check_swing_performance as monitor


def _snapshot(directory, day="20260907", generated="2026-09-07T19:00:00+00:00", **overrides):
    snapshot = {
        "generated_at_utc": generated,
        "validation": {"asof": day, "version": "frozen-v1", "validated": False, "trade_validated": False,
                       "policy": {"top_k": 5, "round_trip_cost": .003}},
        "predictions": [{"code": f"{i:06d}", "rank": i, "probability": .9 - i / 100,
                         "score": 100 - i, "reason": f"original-{i}"} for i in range(1, 7)],
    }
    snapshot.update(overrides)
    (directory / f"swing_snapshot_{day}.json").write_text(json.dumps(snapshot), encoding="utf-8")
    return snapshot


def _outcomes(day="20260907", end="20260914", available="20260914", returns=None):
    values = returns if returns is not None else [.10, -.08, .02, .03, .04, 3.0]
    return pd.DataFrame([
        {"date": pd.Timestamp(day), "code": f"{i:06d}", "label_end": pd.Timestamp(end) if end else pd.NaT,
         "label_available_on": pd.Timestamp(available) if available else pd.NaT, "trade_return": value}
        for i, value in enumerate(values, 1)
    ])


def _sources(monkeypatch, panel, sessions=None):
    monkeypatch.setattr(monitor, "load_feature_panel", lambda *args: panel)
    if sessions is None:
        sessions = pd.bdate_range("2026-09-07", "2026-09-21")
    monkeypatch.setattr(monitor, "_observed_sessions", lambda directory, asof, panel: sessions[sessions <= asof])


def test_missing_first_choice_is_not_replaced_by_sixth(tmp_path, monkeypatch):
    _snapshot(tmp_path)
    _sources(monkeypatch, _outcomes(returns=[float("nan"), .01, .02, .03, .04, 3.0]))
    result = monitor.build_performance_report(tmp_path, "20260914")
    summary = result["summary"]
    assert summary["selected"] == 5
    assert summary["completed"] == 4
    assert summary["unresolved"] == 1
    assert summary["pending"] == 0
    assert summary["known_cohort_mean_net_return"] == pytest.approx(.025 - .003)
    assert [row["rank"] for row in result["days"][0]["picks"]] == [1, 2, 3, 4, 5]
    assert summary["research_forecasts"] == 5
    assert all(not row["is_recorded_trade"] for row in result["days"][0]["picks"])


def test_pending_forecasts_are_neither_losses_nor_zero_returns(tmp_path, monkeypatch):
    _snapshot(tmp_path)
    _sources(monkeypatch, _outcomes(end=None, available=None, returns=[float("nan")] * 6))
    result = monitor.build_performance_report(tmp_path, "20260908")
    assert result["summary"]["pending"] == 5
    assert result["summary"]["completed"] == 0
    assert result["summary"]["known_cohort_mean_net_return"] is None
    assert result["summary"]["known_cohort_hit_rate"] is None
    assert all(row["net_return"] is None and row["net_profit"] is None for row in result["days"][0]["picks"])


def test_late_availability_stays_pending_even_after_planned_exit(tmp_path, monkeypatch):
    _snapshot(tmp_path)
    _sources(monkeypatch, _outcomes(available="20260915"))
    result = monitor.build_performance_report(tmp_path, "20260914")
    assert result["summary"]["pending"] == 5
    assert result["summary"]["completed"] == 0


def test_research_and_retrospective_forecasts_are_not_live_evidence(tmp_path, monkeypatch):
    _snapshot(tmp_path, day="20260907", generated="2026-09-08T00:30:00+00:00")  # entry 09:30 KST
    _snapshot(tmp_path, day="20260908", generated="2026-09-08T19:00:00+00:00")  # before next open
    _snapshot(tmp_path, day="20260909", generated=None)
    panel = pd.concat([
        _outcomes(day="20260907", end="20260914"),
        _outcomes(day="20260908", end="20260915", available="20260915"),
        _outcomes(day="20260909", end="20260916", available="20260916"),
    ], ignore_index=True)
    _sources(monkeypatch, panel)
    result = monitor.build_performance_report(tmp_path, "20260917")
    assert result["summary"]["selected"] == 15
    assert result["prospective_summary"]["selected"] == 5
    assert result["retrospective_summary"]["selected"] == 5
    assert result["timing_unknown_summary"]["selected"] == 5
    assert [day["forecast_timing"] for day in result["days"]] == ["retrospective", "prospective", "unknown_timestamp"]
    assert all(day["forecast_mode"] == "research" for day in result["days"])


def test_both_original_gates_are_required_and_cost_is_frozen(tmp_path, monkeypatch):
    original = _snapshot(tmp_path)
    original["validation"]["validated"] = True
    original["validation"]["policy"]["round_trip_cost"] = .004
    original["validation"]["policy"]["top_k"] = 2
    _snapshot(tmp_path, validation=original["validation"])
    _sources(monkeypatch, _outcomes(returns=[.003, -.15, .20, .20, .20, 3.0]))
    result = monitor.build_performance_report(tmp_path, "20260914")
    assert result["summary"]["selected"] == 2
    assert result["summary"]["known_cohort_hit_rate"] == 0
    assert result["summary"]["stop_threshold_loss_count"] == 1
    assert result["days"][0]["forecast_mode"] == "research"
    assert result["days"][0]["picks"][0]["net_return"] == pytest.approx(-.001)


def test_future_archive_is_ignored_and_missing_entry_calendar_is_unknown(tmp_path, monkeypatch):
    _snapshot(tmp_path)
    _snapshot(tmp_path, day="20260915")
    _sources(monkeypatch, _outcomes(end=None, available=None), sessions=pd.DatetimeIndex([]))
    result = monitor.build_performance_report(tmp_path, "20260907")
    assert result["forecast_days"] == 1
    assert result["prospective_summary"]["selected"] == 0
    assert result["days"][0]["forecast_timing"] == "pending_entry_session"


def test_market_calendar_uses_observed_sessions_instead_of_weekday_guess(tmp_path, monkeypatch):
    _snapshot(tmp_path, generated="2026-09-08T19:00:00+00:00")
    # Deliberately absent September 8: an exchange holiday cannot become entry.
    sessions = pd.DatetimeIndex(["2026-09-07", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14", "2026-09-15"])
    pd.DataFrame({"close": 100}, index=sessions.rename("Date")).to_parquet(tmp_path / "ohlcv_cache_20260915.parquet")
    monkeypatch.setattr(monitor, "load_feature_panel", lambda *args: _outcomes(end="20260915", available="20260915"))
    result = monitor.build_performance_report(tmp_path)
    assert result["asof"] == "20260915"
    assert result["days"][0]["picks"][0]["entry_date"] == "20260909"
    assert result["prospective_summary"]["selected"] == 5


def test_invalid_archive_is_reported_without_reconstructing_its_ranks(tmp_path, monkeypatch):
    original = _snapshot(tmp_path)
    original["predictions"][0]["rank"] = 2
    _snapshot(tmp_path, predictions=original["predictions"])
    _sources(monkeypatch, _outcomes())
    result = monitor.build_performance_report(tmp_path, "20260914")
    assert result["forecast_days"] == 0
    assert len(result["archive_errors"]) == 1
    assert "duplicate" in result["archive_errors"][0]["reason"]


def test_cli_writes_strict_json_and_does_not_modify_snapshot(tmp_path, monkeypatch, capsys):
    _snapshot(tmp_path)
    path = tmp_path / "swing_snapshot_20260907.json"
    original = path.read_bytes()
    _sources(monkeypatch, _outcomes(end=None, available=None, returns=[float("nan")] * 6))
    output = tmp_path / "report.json"
    assert monitor.main(["--data-dir", str(tmp_path), "--asof", "20260908", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["summary"]["completed"] == 0
    assert "NaN" not in output.read_text()
    assert path.read_bytes() == original
    assert str(output) in capsys.readouterr().out
