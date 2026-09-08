"""Point-in-time OHLCV features and five-session swing outcomes.

The recommendation archive defines membership separately on every signal date.
The first valid archived observation wins, so a later cache cannot rewrite old
features. A bar is usable only once its containing snapshot was available.
All returns are fractions (0.01 means 1%), prices/turnover are KRW, volume is
shares, relative volume is a ratio, and breadth/close location are in [0, 1].
``forward_return`` measures next-open-to-D+5-close return without a stop;
``trade_return`` applies an 8% stop, including worse opening-gap fills.
Both returns are before costs. The entry day counts as holding session one.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
HOLD_SESSIONS = 5
STOP_PCT = -0.08
FEATURES = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "gap_1d",
    "intraday_return", "range_pct", "close_location", "ma5_distance",
    "ma20_distance", "volatility_20d", "relative_volume_20d", "log_turnover",
    "universe_ret_1d", "universe_ret_5d", "relative_ret_1d", "relative_ret_5d",
    "universe_breadth",
]
_OUTPUT = [
    "date", "code", "label_end", "label_available_on", "target_up", "forward_return", "trade_return",
    "close", "volume", "turnover",
] + FEATURES
_BARS = ["date", "code", "open", "high", "low", "close", "volume", "turnover", "available_on"]
_CACHE_VERSION = 1
_CACHE_DIRECTORY = ".swing_cache"
_CACHE_NAME = "ohlcv_union.parquet"
_MANIFEST_NAME = "fingerprint.json"
_ALIASES = {
    "종목코드": "code", "Code": "code", "ticker": "code",
    "Date": "date", "날짜": "date", "일자": "date",
    "시가": "open", "고가": "high", "저가": "low", "종가": "close",
    "거래량": "volume", "거래대금": "turnover", "거래대금(원)": "turnover",
    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
}


def _empty_panel() -> pd.DataFrame:
    result = pd.DataFrame({col: pd.Series(dtype="float64") for col in _OUTPUT})
    result["code"] = pd.Series(dtype="object")
    for col in ("date", "label_end", "label_available_on"):
        result[col] = pd.Series(dtype="datetime64[ns]")
    return result[_OUTPUT]


def _codes(series: pd.Series) -> pd.Series:
    codes = series.astype("string").str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    codes = codes.str.zfill(6)
    return codes.where(codes.str.fullmatch(r"[0-9A-Z]{6}", na=False))


def _dated_files(directory: Path, prefix: str, suffix: str, cutoff: pd.Timestamp) -> list[tuple[Path, pd.Timestamp]]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{8}}){re.escape(suffix)}$")
    result = []
    for path in sorted(directory.glob(f"{prefix}*{suffix}")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        day = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
        if pd.notna(day) and day <= cutoff:
            result.append((path, day))
    return result


def _read_bars(path: Path, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    raw = raw.rename(columns=_ALIASES)
    if "date" not in raw.columns:
        if not isinstance(raw.index, pd.DatetimeIndex) and raw.index.name not in ("Date", "date", "날짜", "일자"):
            raise ValueError("OHLCV cache has no date column or date index")
        raw = raw.reset_index().rename(columns={raw.index.name or "index": "date"})
    required = ["date", "code", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(raw.columns):
        raise ValueError("OHLCV cache is missing required OHLCV columns")
    bars = raw[required + (["turnover"] if "turnover" in raw else [])].copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    if bars["date"].dt.tz is not None:
        bars["date"] = bars["date"].dt.tz_localize(None)
    bars["code"] = _codes(bars["code"])
    for col in ("open", "high", "low", "close", "volume"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    finite = np.isfinite(bars[["open", "high", "low", "close", "volume"]]).all(axis=1)
    valid = (
        finite & bars["code"].notna() & bars["date"].notna()
        & bars["date"].le(snapshot_date)
        & bars[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & bars["volume"].gt(0)
        & bars["high"].ge(bars[["open", "close", "low"]].max(axis=1))
        & bars["low"].le(bars[["open", "close", "high"]].min(axis=1))
    )
    bars = bars.loc[valid].copy()
    proxy_turnover = bars["close"] * bars["volume"]
    if "turnover" in bars:
        value = pd.to_numeric(bars["turnover"], errors="coerce")
        bars["turnover"] = value.where(np.isfinite(value) & value.gt(0), proxy_turnover)
    else:
        bars["turnover"] = proxy_turnover
    bars["available_on"] = snapshot_date
    return bars.drop_duplicates(["date", "code"], keep="first")[_BARS]


def _fingerprint(files: list[tuple[Path, pd.Timestamp]]) -> list[list]:
    return [[path.name, path.stat().st_size, path.stat().st_mtime_ns] for path, _ in files]


def _save_union(directory: Path, bars: pd.DataFrame, manifest: dict) -> None:
    """Best-effort atomic derived cache; a read-only data directory also works."""
    temporary = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for filename, write in (
            (_CACHE_NAME, lambda path: bars.to_parquet(path, index=False)),
            (_MANIFEST_NAME, lambda path: Path(path).write_text(json.dumps(manifest), encoding="utf-8")),
        ):
            fd, path = tempfile.mkstemp(prefix=".swing_", dir=directory)
            os.close(fd)
            temporary.append(path)
            write(path)
            os.replace(path, directory / filename)
    except (OSError, ValueError, ImportError) as exc:
        LOGGER.warning("Swing derived cache was not saved: %s", exc)
    finally:
        for path in temporary:
            if os.path.exists(path):
                os.unlink(path)


def _load_union(directory: Path, files: list[tuple[Path, pd.Timestamp]]) -> pd.DataFrame:
    cache_directory = directory / _CACHE_DIRECTORY
    fingerprints = _fingerprint(files)
    union = pd.DataFrame(columns=_BARS)
    start = 0
    try:
        manifest = json.loads((cache_directory / _MANIFEST_NAME).read_text(encoding="utf-8"))
        previous = manifest.get("files", [])
        if manifest.get("version") == _CACHE_VERSION and previous and fingerprints[:len(previous)] == previous:
            cached = pd.read_parquet(cache_directory / _CACHE_NAME)
            if set(_BARS).issubset(cached.columns) and len(cached) == manifest.get("rows"):
                union = cached[_BARS]
                start = len(previous)
    except (OSError, ValueError, ImportError, KeyError) as exc:
        LOGGER.debug("Rebuilding swing OHLCV union cache at %s: %s", cache_directory, exc)
    if start == len(files) and start:
        return union
    complete = True
    for path, snapshot_date in files[start:]:
        try:
            bars = _read_bars(path, snapshot_date)
        except (OSError, ValueError, ImportError, KeyError, TypeError) as exc:
            LOGGER.warning("Skipping unreadable swing OHLCV snapshot %s: %s", path.name, exc)
            complete = False
            continue
        if bars.empty:
            continue
        union = pd.concat([union, bars], ignore_index=True) if not union.empty else bars
        union = union.drop_duplicates(["date", "code"], keep="first")
    if complete and not union.empty:
        _save_union(cache_directory, union, {"version": _CACHE_VERSION, "files": fingerprints, "rows": len(union)})
    return union


def _membership(directory: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for path, date in _dated_files(directory, "recommend_", ".csv", cutoff):
        try:
            raw = pd.read_csv(path, usecols=lambda name: name in ("종목코드", "code"), dtype="string")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            LOGGER.warning("Skipping unreadable swing membership %s: %s", path.name, exc)
            continue
        if raw.empty or len(raw.columns) == 0:
            continue
        codes = _codes(raw.iloc[:, 0]).dropna().drop_duplicates()
        parts.append(pd.DataFrame({"date": date, "code": codes}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "code"])


def _swing_outcomes(panel: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Require a fully matured market window, even when an early stop is known.

    Missing/invalid bars are not forward-filled. A position with an already
    observed stop can still have a known return if later ticker bars disappear;
    it enters training only after its originally planned D+5 window has elapsed.
    """
    sessions = pd.Index(bars["date"].drop_duplicates().sort_values())
    prices = bars.set_index(["date", "code"])
    future = []
    for offset in range(1, HOLD_SESSIONS + 1):
        mapping = dict(zip(sessions[:-offset], sessions[offset:]))
        dates = panel["date"].map(mapping)
        keys = pd.MultiIndex.from_arrays([dates, panel["code"]], names=["date", "code"])
        day = prices.reindex(keys)[["open", "high", "low", "close", "volume", "available_on"]].reset_index(drop=True)
        future.append(day)
        if offset == HOLD_SESSIONS:
            panel["label_end"] = dates.to_numpy()
    # A later archive can fill missing historical outcome bars. Purged training
    # must wait for that archive, as well as the originally scheduled exit day.
    available = pd.concat([day["available_on"] for day in future], axis=1).max(axis=1)
    panel["label_available_on"] = available.to_numpy()
    panel["label_available_on"] = panel[["label_end", "label_available_on"]].max(axis=1)
    panel.loc[panel["label_end"].isna(), "label_available_on"] = pd.NaT

    n = len(panel)
    gross = np.full(n, np.nan)
    raw = np.full(n, np.nan)
    status = np.full(n, "immature", dtype=object)
    matured = panel["label_end"].notna().to_numpy()
    opens = np.column_stack([day["open"].to_numpy(float) for day in future])
    highs = np.column_stack([day["high"].to_numpy(float) for day in future])
    lows = np.column_stack([day["low"].to_numpy(float) for day in future])
    closes = np.column_stack([day["close"].to_numpy(float) for day in future])
    signal_close = panel["close"].to_numpy(float)
    for row in np.flatnonzero(matured):
        entry = opens[row, 0]
        if not np.isfinite(entry):
            status[row] = "missing_entry"
            continue
        first_gap = entry / signal_close[row] - 1.0
        if first_gap >= 0.285 and highs[row, 0] == lows[row, 0]:
            status[row] = "unfillable_upper_limit"
            continue
        stop = entry * (1.0 + STOP_PCT)
        previous_close = signal_close[row]
        status[row] = "complete"
        stopped = False
        raw_valid = True
        for step in range(HOLD_SESSIONS):
            op, hi, lo, cl = opens[row, step], highs[row, step], lows[row, step], closes[row, step]
            if not np.isfinite(op):
                raw_valid = False
                if not stopped:
                    status[row] = "missing_bar"
                    break
                continue
            # No fabricated split adjustment or extreme-price return labels.
            # A discontinuity after an already completed stop cannot change
            # that realized exit, but invalidates its no-stop comparison.
            abnormal = (abs(op / previous_close - 1.0) > 0.35
                        or abs(cl / previous_close - 1.0) > 0.35
                        or abs(cl / op - 1.0) > 0.35)
            if abnormal:
                raw_valid = False
                if not stopped:
                    status[row] = "price_discontinuity"
                    break
            if not stopped and lo <= stop:
                # A locked lower-limit session provides no assured sell fill.
                if hi == lo and op / previous_close - 1.0 <= -0.285:
                    status[row] = "unfillable_lower_limit"
                    raw_valid = False
                    break
                gross[row] = min(op, stop) / entry - 1.0
                stopped = True
                status[row] = "stopped"
            previous_close = cl
        if status[row] == "complete":
            gross[row] = closes[row, -1] / entry - 1.0
        if raw_valid and status[row] in ("complete", "stopped"):
            raw[row] = closes[row, -1] / entry - 1.0
    panel["trade_return"] = gross
    panel["forward_return"] = raw
    panel["target_up"] = np.where(np.isfinite(gross), (gross > 0).astype(float), np.nan)
    panel.attrs["label_quality"] = pd.Series(status).value_counts().astype(int).to_dict()
    panel.attrs["label_definition"] = {
        "hold_sessions": HOLD_SESSIONS, "stop_pct": STOP_PCT,
        "entry": "D+1 open", "scheduled_exit": "D+5 close",
        "costs_included": False, "early_stop_training": "wait until scheduled exit",
    }
    return panel


def load_feature_panel(data_dir: str | os.PathLike, asof_ymd: str, current_codes=None) -> pd.DataFrame:
    """Build historical/current signal rows using only archives through ``asof``.

    Requires 21 trailing valid bars (20 completed daily returns). Enter at D+1
    open and exit by D+5 close: five *market-observed sessions*, including entry.
    Missing bars/zero-volume bars, unfillable limit entries, and >35% price
    discontinuities remain unknown. Latest rows are returned with NaN outcomes.
    """
    cutoff = pd.to_datetime(str(asof_ymd), errors="coerce")
    if pd.isna(cutoff):
        raise ValueError("asof_ymd must be a valid YYYYMMDD or ISO date")
    cutoff = cutoff.normalize().tz_localize(None)
    directory = Path(data_dir)
    membership = _membership(directory, cutoff)
    if current_codes is not None:
        codes = _codes(pd.Series(list(current_codes), dtype="string")).dropna().drop_duplicates()
        membership = pd.concat([
            membership.loc[membership["date"].ne(cutoff)],
            pd.DataFrame({"date": cutoff, "code": codes}),
        ], ignore_index=True)
    files = _dated_files(directory, "ohlcv_cache_", ".parquet", cutoff)
    if membership.empty or not files:
        return _empty_panel()
    bars = _load_union(directory, files)
    if bars.empty:
        return _empty_panel()
    bars = bars.loc[bars["date"].le(cutoff) & bars["available_on"].le(cutoff)].copy()
    bars = bars.sort_values(["code", "date"]).reset_index(drop=True)
    grouped = bars.groupby("code", sort=False)
    previous_close = grouped["close"].shift(1)
    for days in (1, 3, 5, 10, 20):
        bars[f"ret_{days}d"] = bars["close"] / grouped["close"].shift(days) - 1.0
    bars["gap_1d"] = bars["open"] / previous_close - 1.0
    bars["intraday_return"] = bars["close"] / bars["open"] - 1.0
    bars["range_pct"] = (bars["high"] - bars["low"]) / previous_close
    spread = bars["high"] - bars["low"]
    bars["close_location"] = ((bars["close"] - bars["low"]) / spread).where(spread.gt(0), 0.5)
    for days in (5, 20):
        ma = grouped["close"].transform(lambda values: values.rolling(days, min_periods=days).mean())
        bars[f"ma{days}_distance"] = bars["close"] / ma - 1.0
    bars["volatility_20d"] = bars.groupby("code", sort=False)["ret_1d"].transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=1)
    )
    volume_mean = grouped["volume"].transform(lambda values: values.shift(1).rolling(20, min_periods=20).mean())
    bars["relative_volume_20d"] = bars["volume"] / volume_mean
    bars["log_turnover"] = np.log1p(bars["turnover"])
    # Prevent later backfills from becoming historical features, including any
    # of the trailing bars needed by the longest lookback.
    bars["_available_ns"] = bars["available_on"].astype("int64")
    latest_input = bars.groupby("code", sort=False)["_available_ns"].transform(
        lambda values: values.rolling(21, min_periods=21).max()
    )
    feature_available = latest_input.le(bars["date"].astype("int64"))

    panel = bars.loc[feature_available].copy()
    panel = panel.merge(membership, on=["date", "code"], how="inner", validate="one_to_one")
    if panel.empty:
        return _empty_panel()
    # Cross-sectional context also uses that date's archived membership only.
    by_day = panel.groupby("date", sort=False)
    for days in (1, 5):
        name = f"ret_{days}d"
        panel[f"universe_{name}"] = by_day[name].transform("mean")
        panel[f"relative_{name}"] = panel[name] - panel[f"universe_{name}"]
    panel["universe_breadth"] = panel["ret_1d"].gt(0).groupby(panel["date"]).transform("mean")
    panel[FEATURES] = panel[FEATURES].replace([np.inf, -np.inf], np.nan)
    panel = _swing_outcomes(panel, bars)
    return panel[_OUTPUT].sort_values(["date", "code"]).reset_index(drop=True)
