"""Regression tests for snapshot refresh, input validation and point-in-time dates."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import numpy as np
import pandas as pd
import pytest

from services import data_store as ds
from services import session_freshness as sf
from services.snapshot_integrity import (
    normalize_ymd, validate_snapshot, snapshot_date, freshness_summary,
)
from services.recommendation_quality import apply_recommendation_quality_guard


def frame(day="20260904", **overrides):
    row = {"종목코드": "005930", "종목명": "테스트", "종가": 10000,
           "추천매수가": 10000, "손절가": 9500, "추천매도가1": 11000,
           "기준일": day, "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1,
           "ALPHA_GATE_ACTIVE": 1, "ALPHA_ENTRY_OK": 1, "ALPHA_SCORE": 95,
           "RR_NOW_TP1": 2, "ROUTE": "ATTACK", "DATA_FRESHNESS_OK": 1}
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.fixture
def store_env(tmp_path, monkeypatch):
    path = tmp_path / "recommend_latest.csv"
    monkeypatch.setattr(ds, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ds, "RECOMMEND_PATH", str(path))
    monkeypatch.setattr(ds, "REMOTE_CSV_URL", "https://example.invalid/latest.csv")
    def unavailable(*a, **kw):
        raise ConnectionError("offline")
    monkeypatch.setattr(ds.requests, "get", unavailable)
    return ds.DataStore(), path


def remote(monkeypatch, data):
    response = SimpleNamespace(content=data.to_csv(index=False).encode(),
                               raise_for_status=lambda: None)
    monkeypatch.setattr(ds.requests, "get", lambda *a, **kw: response)


@pytest.mark.parametrize("value,expected", [
    (20260904, "20260904"), (20260904.0, "20260904"),
    ("2026-09-04", "20260904"), ("20260230", ""),
    ("2026", ""), (np.nan, ""), (None, ""), (0, ""),
])
def test_date_parsing(value, expected):
    assert normalize_ymd(value) == expected


def test_schema_code_and_batch_validation():
    assert validate_snapshot(frame(종목코드="5930")).iloc[0]["종목코드"] == "005930"
    for bad in [frame().drop(columns="종가"), frame(종목코드="nan"),
                pd.concat([frame(), frame()], ignore_index=True),
                frame(기준일="broken"), frame(trade_date="20260903")]:
        with pytest.raises(ValueError):
            validate_snapshot(bad)
    assert snapshot_date(frame().drop(columns="기준일")) == ""
    assert validate_snapshot(frame(종목코드="0126Z0")).iloc[0]["종목코드"] == "0126Z0"


@pytest.mark.parametrize("field,value", [
    ("종가", np.inf), ("종가", -1), ("종가", np.nan),
    ("RR_NOW_TP1", np.inf), ("RR_NOW_TP1", "broken"),
    ("손절가", 11000), ("추천매도가1", 9000), ("ALPHA_SCORE", np.inf),
])
def test_invalid_inputs_cannot_become_official_buy(field, value):
    good = apply_recommendation_quality_guard(frame())
    assert good.iloc[0]["PRODUCTION_BUY"] == 1
    bad = apply_recommendation_quality_guard(frame(**{field: value}))
    assert bad.iloc[0]["PRODUCTION_BUY"] == 0
    assert bad.iloc[0]["BUY_NOW_ELIGIBLE"] == 0
    assert bad.iloc[0]["RECOMMENDED_WEIGHT_PCT"] == 0
    assert "입력 오류" in bad.iloc[0]["QUALITY_GUARD_REASON"]


def test_manual_refresh_downloads_even_when_local_exists(store_env, monkeypatch):
    store, path = store_env
    frame("20260903").to_csv(path, index=False)
    remote(monkeypatch, frame())
    result = store.refresh(force_remote=True)
    assert result["ok"] and result["source"] == "remote"
    assert store.data_ts == "20260904"
    assert snapshot_date(pd.read_csv(path)) == "20260904"
    assert not list(path.parent.glob("*.tmp"))


def test_older_remote_does_not_replace_newer_local(store_env, monkeypatch):
    store, path = store_env
    frame().to_csv(path, index=False)
    before = path.read_bytes()
    remote(monkeypatch, frame("20260903"))
    assert store.refresh(force_remote=True)["source"] == "local"
    assert path.read_bytes() == before
    assert store.data_ts == "20260904"


def test_invalid_remote_cannot_poison_cache(store_env, monkeypatch):
    store, path = store_env
    frame().to_csv(path, index=False)
    before = path.read_bytes()
    remote(monkeypatch, pd.DataFrame({"error": ["bad response"]}))
    result = store.refresh(force_remote=True)
    assert not result["ok"]
    assert store.loaded and path.read_bytes() == before
    assert "기존 데이터" in result["message"]


def test_outage_retains_last_good_memory(store_env):
    store, path = store_env
    frame().to_csv(path, index=False)
    assert store.refresh()["ok"]
    before = store.scored
    path.unlink()
    result = store.refresh(force_remote=True)
    assert not result["ok"] and result["source"] == "memory"
    pd.testing.assert_frame_equal(before, store.scored)
    assert store.data_ts == "20260904"


def test_no_data_does_not_claim_success(store_env):
    store, _ = store_env
    assert not store.refresh(force_remote=True)["ok"]
    assert not store.loaded and store.data_ts == ""


def test_unknown_date_is_never_replaced_by_today(store_env):
    store, path = store_env
    frame().drop(columns="기준일").to_csv(path, index=False)
    store.refresh()
    assert store.data_ts == "확인 불가"


def test_failed_atomic_write_keeps_old_file(store_env, monkeypatch):
    store, path = store_env
    frame("20260903").to_csv(path, index=False)
    before = path.read_bytes()
    remote(monkeypatch, frame())
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(ds.os, "replace", fail)
    result = store.refresh(force_remote=True)
    assert store.data_ts == "20260904"  # validated data usable in memory
    assert result["errors"] and path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))


def test_scored_setter_and_getter_do_not_leak_mutability():
    store = ds.DataStore()
    source = frame()
    store.scored = source
    source.loc[0, "종가"] = 1
    copy = store.scored
    copy.loc[0, "종가"] = 2
    assert store.scored.iloc[0]["종가"] == 10000


def test_refreshes_are_serialized(store_env, monkeypatch):
    store, path = store_env
    entered, release = threading.Event(), threading.Event()
    calls = []
    def get(*a, **kw):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(content=frame().to_csv(index=False).encode(),
                               raise_for_status=lambda: None)
    monkeypatch.setattr(ds.requests, "get", get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(store.refresh, True)
        assert entered.wait(5)
        second = pool.submit(store.refresh, True)
        assert len(calls) == 1
        release.set()
        assert first.result()["ok"] and second.result()["ok"]
    assert snapshot_date(pd.read_csv(path)) == "20260904"


def test_cache_asof_filters_future_rows_and_normalizes_dates(tmp_path, monkeypatch):
    cache = tmp_path / "ohlcv_cache_20260904.parquet"
    cache.touch()
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pd.DataFrame(
        {"종목코드": ["005930"] * 2},
        index=pd.to_datetime(["2026-09-03", "2026-09-07"])))
    assert sf.latest_price_ymd(str(tmp_path), "2026-09-04") == "20260903"
    assert sf.trading_days(str(tmp_path), asof_ymd="20260904") == ["20260903"]
    assert sf.assess("2026-09-04", str(tmp_path))["stale"]


def test_numeric_row_index_is_not_a_price_date(tmp_path, monkeypatch):
    (tmp_path / "ohlcv_cache_20260904.parquet").touch()
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pd.DataFrame({"종목코드": ["005930"]}))
    assert sf.latest_price_ymd(str(tmp_path)) is None
    assert not sf.assess("nonsense", str(tmp_path))["ok"]


def test_freshness_display_never_claims_live_prices():
    result = freshness_summary(frame(PRICE_ASOF="20260903"))
    assert result["stale"] and result["price_date"] == "20260903"
    assert result["batch_date"] == "20260904"
    assert freshness_summary(frame())["unknown"]


def test_bad_high_ranked_candidate_does_not_hide_valid_candidate():
    candidates = pd.concat([
        frame(ALPHA_SCORE=100, RR_NOW_TP1=np.inf),
        frame(종목코드="000660", ALPHA_SCORE=90),
    ], ignore_index=True)
    out = apply_recommendation_quality_guard(candidates)
    assert out.PRODUCTION_BUY.tolist() == [0, 1]


def test_quality_failure_is_reported_to_refresh_ui(store_env, monkeypatch):
    from services import recommendation_quality
    store, path = store_env
    frame().to_csv(path, index=False)
    def fail(*a, **kw):
        raise RuntimeError("quality unavailable")
    monkeypatch.setattr(recommendation_quality, "apply_recommendation_quality_guard", fail)
    result = store.refresh()
    assert not result["ok"] and "신규매수" in result["message"]
    assert store.scored.PRODUCTION_BUY.sum() == 0


def test_same_date_remote_correction_replaces_local(store_env, monkeypatch):
    store, path = store_env
    frame().to_csv(path, index=False)
    remote(monkeypatch, frame(종가=10100))
    assert store.refresh(force_remote=True)["source"] == "remote"
    assert store.scored.iloc[0]["종가"] == 10100
