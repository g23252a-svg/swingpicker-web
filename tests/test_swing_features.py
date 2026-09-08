"""Causal feature and executable five-session swing-label contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services import swing_features as sf


def _rows(days, codes=("000001", "0009K0")):
    return pd.DataFrame([
        {"date": day, "code": code, "open": 100.0, "high": 101.0,
         "low": 99.0, "close": 100.0, "volume": 1000.0}
        for code in codes for day in days
    ])


def _cache(directory, day, rows):
    names = {"date": "Date", "code": "종목코드", "open": "시가", "high": "고가",
             "low": "저가", "close": "종가", "volume": "거래량"}
    rows.rename(columns=names).set_index("Date").to_parquet(directory / f"ohlcv_cache_{day:%Y%m%d}.parquet")


def _rec(directory, day, codes=("000001", "0009K0")):
    pd.DataFrame({"종목코드": codes}).to_csv(directory / f"recommend_{day:%Y%m%d}.csv", index=False)


@pytest.fixture
def archive(tmp_path):
    days = pd.bdate_range("2026-01-01", periods=27)
    rows = _rows(days)
    _cache(tmp_path, days[20], rows.loc[rows.date.le(days[20])])
    _rec(tmp_path, days[20])
    return tmp_path, days, rows


def _signal(panel, day, code="000001"):
    return panel.loc[panel.date.eq(day) & panel.code.eq(code)].iloc[0]


def test_five_sessions_not_close_direction_and_gap_cost(archive):
    directory, days, rows = archive
    selected = rows.code.eq("000001") & rows.date.gt(days[20])
    rows.loc[selected, ["open", "high", "low", "close"]] = [109, 110, 104, 105]
    rows.loc[rows.code.eq("000001") & rows.date.eq(days[21]), ["open", "high", "low", "close"]] = [110, 111, 108, 109]
    # D+6 must not affect the five-session outcome.
    rows.loc[rows.code.eq("000001") & rows.date.eq(days[26]), ["open", "high", "low", "close"]] = [105, 132, 104, 130]
    _cache(directory, days[26], rows)
    result = sf.load_feature_panel(directory, f"{days[26]:%Y%m%d}")
    row = _signal(result, days[20])
    assert row.label_end == days[25]
    assert row.forward_return == pytest.approx(105 / 110 - 1)
    assert row.trade_return == pytest.approx(105 / 110 - 1)
    assert row.target_up == 0  # D+1 close is up 9%, but actual swing loses.
    assert result.attrs["label_definition"]["hold_sessions"] == 5


def test_open_gap_stop_is_worse_than_eight_percent(archive):
    directory, days, rows = archive
    mask = rows.code.eq("000001") & rows.date.eq(days[22])
    rows.loc[mask, ["open", "high", "low", "close"]] = [85, 90, 80, 88]
    _cache(directory, days[25], rows)
    row = _signal(sf.load_feature_panel(directory, str(days[25].date())), days[20])
    assert row.trade_return == pytest.approx(-0.15)
    assert row.forward_return == pytest.approx(0.0)
    assert row.label_end == days[25]


def test_intraday_stop_waits_for_complete_window(archive):
    directory, days, rows = archive
    rows.loc[rows.code.eq("000001") & rows.date.eq(days[21]), "low"] = 90
    _cache(directory, days[21], rows)
    early = _signal(sf.load_feature_panel(directory, f"{days[21]:%Y%m%d}"), days[20])
    assert pd.isna(early.trade_return)
    assert pd.isna(early.label_end)
    _cache(directory, days[25], rows)
    late = _signal(sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}"), days[20])
    assert late.trade_return == pytest.approx(-0.08)
    assert late.label_end == days[25]


def test_missing_next_market_bar_never_uses_next_ticker_bar(archive):
    directory, days, rows = archive
    rows = rows.loc[~(rows.code.eq("000001") & rows.date.eq(days[21]))]
    _cache(directory, days[25], rows)
    result = sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}")
    row = _signal(result, days[20])
    assert pd.isna(row.trade_return)
    assert row.label_end == days[25]
    assert result.attrs["label_quality"]["missing_entry"] == 1


def test_missing_inside_holding_window_remains_unknown(archive):
    directory, days, rows = archive
    rows = rows.loc[~(rows.code.eq("000001") & rows.date.eq(days[23]))]
    _cache(directory, days[25], rows)
    result = sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}")
    assert pd.isna(_signal(result, days[20]).trade_return)
    assert result.attrs["label_quality"]["missing_bar"] == 1


def test_price_discontinuity_does_not_become_a_winning_label(archive):
    directory, days, rows = archive
    rows.loc[rows.code.eq("000001") & rows.date.eq(days[23]), ["open", "high", "low", "close"]] = [150, 155, 149, 153]
    _cache(directory, days[25], rows)
    result = sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}")
    assert pd.isna(_signal(result, days[20]).target_up)
    assert result.attrs["label_quality"]["price_discontinuity"] == 1


@pytest.mark.parametrize("kind", ["upper", "lower"])
def test_locked_limit_bar_has_no_assured_fill(archive, kind):
    directory, days, rows = archive
    if kind == "upper":
        mask = rows.code.eq("000001") & rows.date.eq(days[21])
        rows.loc[mask, ["open", "high", "low", "close"]] = 130.0
    else:
        mask = rows.code.eq("000001") & rows.date.eq(days[22])
        rows.loc[mask, ["open", "high", "low", "close"]] = 70.0
    _cache(directory, days[25], rows)
    result = sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}")
    assert pd.isna(_signal(result, days[20]).trade_return)
    assert result.attrs["label_quality"][f"unfillable_{kind}_limit"] == 1


def test_future_filenames_rows_and_later_revisions_do_not_change_features(archive):
    directory, days, rows = archive
    # Even a past-dated file cannot smuggle later OHLCV rows into the panel.
    _cache(directory, days[20], rows)
    before = sf.load_feature_panel(directory, f"{days[20]:%Y%m%d}")
    changed = rows.copy()
    changed.loc[changed.date.le(days[20]), ["open", "high", "low", "close"]] *= 2
    _cache(directory, days[25], changed)
    _rec(directory, days[25], codes=("0009K0",))
    exact_cutoff = sf.load_feature_panel(directory, f"{days[20]:%Y%m%d}")
    pd.testing.assert_frame_equal(before, exact_cutoff)
    later = sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}")
    pd.testing.assert_frame_equal(
        before[["date", "code", "close"] + sf.FEATURES],
        later.loc[later.date.eq(days[20]), ["date", "code", "close"] + sf.FEATURES].reset_index(drop=True),
    )
    assert set(later.loc[later.date.eq(days[25]), "code"]) == {"0009K0"}
    assert before.target_up.isna().all()
    assert np.isfinite(before[sf.FEATURES]).all().all()


def test_daily_membership_and_current_override_are_separate(archive):
    directory, days, rows = archive
    _rec(directory, days[20], codes=("0009K0",))
    _cache(directory, days[25], rows)
    _rec(directory, days[25], codes=("0009K0",))
    result = sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}", current_codes=[1])
    assert set(result.loc[result.date.eq(days[20]), "code"]) == {"0009K0"}
    assert set(result.loc[result.date.eq(days[25]), "code"]) == {"000001"}
    assert result.loc[result.date.eq(days[25]), "target_up"].isna().all()


def test_late_backfilled_labels_report_availability(archive):
    directory, days, rows = archive
    # D+5 bar becomes available only in the D+6 snapshot.
    _cache(directory, days[26], rows)
    result = sf.load_feature_panel(directory, f"{days[26]:%Y%m%d}")
    row = _signal(result, days[20])
    assert row.label_end == days[25]
    assert row.label_available_on == days[26]


def test_cache_is_reused(archive, monkeypatch):
    directory, days, rows = archive
    _cache(directory, days[25], rows)
    before = sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}")
    monkeypatch.setattr(sf, "_read_bars", lambda *args: pytest.fail("unchanged archive should use union cache"))
    after = sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}")
    pd.testing.assert_frame_equal(before, after)
    assert (directory / ".swing_cache" / "ohlcv_union.parquet").exists()
    assert (directory / ".swing_cache" / "fingerprint.json").exists()
    assert not list(directory.glob("swing_ohlcv_union*"))


def test_late_source_cannot_create_past_features(archive):
    directory, days, rows = archive
    _cache(directory, days[20], rows.loc[rows.date.le(days[20]) & rows.code.eq("0009K0")])
    _cache(directory, days[25], rows)
    result = sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}", current_codes=["000001"])
    assert set(result.loc[result.date.eq(days[20]), "code"]) == {"0009K0"}
    assert set(result.loc[result.date.eq(days[25]), "code"]) == {"000001"}


def test_missing_archive_returns_typed_empty_panel(tmp_path):
    result = sf.load_feature_panel(tmp_path, "20260101")
    assert result.empty
    assert set(sf.FEATURES).issubset(result.columns)
    assert pd.api.types.is_datetime64_ns_dtype(result.date)


def test_bad_ohlcv_and_zero_volume_cannot_become_entry(archive):
    directory, days, rows = archive
    rows.loc[rows.code.eq("000001") & rows.date.eq(days[21]), "high"] = 0
    rows.loc[rows.code.eq("0009K0") & rows.date.eq(days[21]), "volume"] = 0
    _cache(directory, days[25], rows)
    result = sf.load_feature_panel(directory, f"{days[25]:%Y%m%d}")
    assert result.trade_return.isna().all()
