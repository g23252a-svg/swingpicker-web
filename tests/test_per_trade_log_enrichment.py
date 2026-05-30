# -*- coding: utf-8 -*-
"""v4.0 per-trade 로그 축 보강 + 스키마 마이그레이션 회귀 테스트.

pytest tests/test_per_trade_log_enrichment.py -v

대상: kelly_calibrator (save_per_trade_log / _ensure_per_trade_schema / load_per_trade_log)
- 신규 축 컬럼(MACRO_REGIME_MODE/ACTION_TIER/ROUTE/TOP_PICK_TYPE)이 스키마에 있음
- 구버전(14컬럼) 파일에 신규행 append 시 자동 마이그레이션(구행 NaN 백필, 정렬 안 깨짐)
- 마이그레이션 멱등 · 축 값 passthrough · 하위호환(축 없는 행도 저장)
"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from kelly_calibrator import (
    PER_TRADE_COLS, save_per_trade_log, load_per_trade_log, _ensure_per_trade_schema,
)

OLD_COLS = ["rec_date", "code", "method", "topk", "horizon", "score",
            "entry_price", "exit_price", "stop_price", "target_price",
            "ret_pct", "win", "exit_type", "b_ratio"]


def _old_row(code="000001", method="POSITION_TRACK"):
    return {c: (code if c == "code" else method if c == "method"
                else "20260501" if c == "rec_date" else 1) for c in OLD_COLS}


def _new_row(code="000002", method="POSITION_TRACK"):
    r = _old_row(code, method)
    r.update(MACRO_REGIME_MODE="FX_HIGH_AND_INTERNAL_WEAK", ACTION_TIER="",
             ROUTE="ARMED", TOP_PICK_TYPE="AGGRESSIVE")
    return r


def test_schema_has_new_axes():
    for c in ["MACRO_REGIME_MODE", "ACTION_TIER", "ROUTE", "TOP_PICK_TYPE"]:
        assert c in PER_TRADE_COLS


def test_migration_old_file_then_append_new(tmp_path):
    """구버전 14컬럼 파일 + 신규행 append → 18컬럼, 구행 NaN, 정렬 안 깨짐."""
    d = str(tmp_path)
    path = os.path.join(d, "per_trade_log.csv")
    pd.DataFrame([_old_row("000001"), _old_row("000003")]).to_csv(
        path, index=False, encoding="utf-8-sig")  # 구버전 파일 (14컬럼)

    save_per_trade_log(d, [_new_row("000002", method="POSITION_TRACK")], asof_ymd="20260502")

    df = load_per_trade_log(d)
    # 컬럼이 신규 스키마로 확장
    for c in ["MACRO_REGIME_MODE", "ROUTE", "TOP_PICK_TYPE"]:
        assert c in df.columns
    # 행 수 = 구 2 + 신규 1 (정렬 안 깨졌으면 code가 살아있음)
    codes = set(df["code"].astype(str))
    assert {"000001", "000003", "000002"} <= codes
    # 구행은 신규 축이 결측, 신규행은 값 보존
    new = df[df["code"] == "000002"].iloc[0]
    assert new["ROUTE"] == "ARMED" and new["TOP_PICK_TYPE"] == "AGGRESSIVE"
    old = df[df["code"] == "000001"].iloc[0]
    assert pd.isna(old["ROUTE"]) or old["ROUTE"] in ("", "nan")


def test_migration_idempotent(tmp_path):
    """이미 신규 스키마면 마이그레이션 no-op (재호출해도 행/컬럼 불변)."""
    d = str(tmp_path)
    path = os.path.join(d, "per_trade_log.csv")
    pd.DataFrame([_new_row("000002")]).reindex(columns=PER_TRADE_COLS).to_csv(
        path, index=False, encoding="utf-8-sig")
    before = pd.read_csv(path, dtype={"code": str})
    _ensure_per_trade_schema(path)
    _ensure_per_trade_schema(path)
    after = pd.read_csv(path, dtype={"code": str})
    assert list(before.columns) == list(after.columns)
    assert len(before) == len(after)


def test_axis_passthrough_through_save_load(tmp_path):
    """축 값이 save→load 왕복에서 보존."""
    d = str(tmp_path)
    save_per_trade_log(d, [_new_row("000009", method="POSITION_TRACK")], asof_ymd="20260503")
    df = load_per_trade_log(d)
    r = df[df["code"] == "000009"].iloc[0]
    assert r["MACRO_REGIME_MODE"] == "FX_HIGH_AND_INTERNAL_WEAK"
    assert r["ROUTE"] == "ARMED"


def test_backward_compat_rows_without_axes(tmp_path):
    """축 없는(구형) 행을 save해도 reindex로 결측 채워 저장된다."""
    d = str(tmp_path)
    save_per_trade_log(d, [_old_row("000004", method="POSITION_TRACK")], asof_ymd="20260504")
    df = load_per_trade_log(d)
    assert "000004" in set(df["code"].astype(str))
    assert "ROUTE" in df.columns  # 컬럼은 존재(값은 결측)


def test_real_log_migration_smoke():
    """실제 per_trade_log.csv 복사본 마이그레이션 smoke. 없으면 skip.
    마이그레이션은 reindex만(행 손실 0). dedup은 load 시점에 별도 발생."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(repo, "data", "per_trade_log.csv")
    if not os.path.exists(src):
        import pytest
        pytest.skip("per_trade_log.csv 없음")
    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, "per_trade_log.csv")
        raw_before = pd.read_csv(src, dtype={"code": str})
        raw_before.to_csv(dst, index=False, encoding="utf-8-sig")
        _ensure_per_trade_schema(dst)
        raw_after = pd.read_csv(dst, dtype={"code": str})
        # 마이그레이션: 행 손실 0, 컬럼은 신규 스키마
        assert len(raw_after) == len(raw_before)
        assert list(raw_after.columns) == list(PER_TRADE_COLS)
        # load는 dedup → raw 이하, 신규 축 컬럼 존재
        loaded = load_per_trade_log(d)
        assert len(loaded) <= len(raw_after)
        for c in ["MACRO_REGIME_MODE", "ROUTE", "TOP_PICK_TYPE", "ACTION_TIER"]:
            assert c in loaded.columns
