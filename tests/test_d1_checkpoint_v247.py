# -*- coding: utf-8 -*-
"""
test_d1_checkpoint_v247.py — v24.7 D+1 Checkpoint 회귀/안전 테스트.

보증
  1. 공식 컬럼(TOP_PICK/BUY_NOW_ELIGIBLE/손절가) 불변.
  2. 컷 가격 = 추천매수가 × (1 + pct/100), 원 단위 반올림.
  3. 결측/무효 진입가 강건(NaN·0·소액), 빈 df 안전.
  4. config 토글(d1_enabled=False → 컬럼 미생성), 커스텀 cut 반영.
  5. 사유 문자열 포맷.
"""
import numpy as np
import pandas as pd
import pytest

from d1_checkpoint import add_d1_checkpoint_columns, d1_summary, D1_OUTPUT_COLS
from collector_config import D1CheckpointConfig, CollectorConfig


def _df(**kw):
    base = dict(종목명=["A", "B", "C"], 종목코드=["000001", "000002", "000003"],
                추천매수가=[10000.0, 25550.0, np.nan],
                손절가=[9000.0, 22000.0, 5000.0],
                TOP_PICK=[1, 0, 0], BUY_NOW_ELIGIBLE=[1, 0, 0])
    base.update(kw)
    return pd.DataFrame(base)


def test_output_columns_and_price():
    out = add_d1_checkpoint_columns(_df())
    for c in D1_OUTPUT_COLS:
        assert c in out.columns
    # 10000 × 0.96 = 9600 / 25550 × 0.96 = 24528
    assert out.loc[0, "D1_CHECKPOINT_PRICE"] == 9600
    assert out.loc[1, "D1_CHECKPOINT_PRICE"] == 24528
    assert out.loc[0, "D1_CHECKPOINT_PCT"] == -4.0


def test_official_columns_unchanged():
    df = _df()
    tp, buy, stop = df.TOP_PICK.tolist(), df.BUY_NOW_ELIGIBLE.tolist(), df.손절가.tolist()
    out = add_d1_checkpoint_columns(df)
    assert out.TOP_PICK.tolist() == tp
    assert out.BUY_NOW_ELIGIBLE.tolist() == buy
    assert out.손절가.tolist() == stop  # 손절가 절대 불변


def test_invalid_entry_graceful():
    out = add_d1_checkpoint_columns(_df())
    # 행2: 추천매수가 NaN → 컷 NaN, 사유 빈문자
    assert pd.isna(out.loc[2, "D1_CHECKPOINT_PRICE"])
    assert out.loc[2, "D1_CHECKPOINT_REASON"] == ""


def test_min_entry_floor():
    df = _df(추천매수가=[50.0, 10000.0, 200.0])  # 50 < min 100 → 제외
    out = add_d1_checkpoint_columns(df)
    assert pd.isna(out.loc[0, "D1_CHECKPOINT_PRICE"])
    assert out.loc[2, "D1_CHECKPOINT_PRICE"] == 192  # 200×0.96


def test_custom_cut_and_disable():
    cfg = D1CheckpointConfig(d1_cut_pct=-3.0)
    out = add_d1_checkpoint_columns(_df(), config=cfg)
    assert out.loc[0, "D1_CHECKPOINT_PRICE"] == 9700  # ×0.97
    off = add_d1_checkpoint_columns(_df(), config=D1CheckpointConfig(d1_enabled=False))
    assert "D1_CHECKPOINT_PRICE" not in off.columns


def test_facade_config_path():
    c = CollectorConfig(d1=D1CheckpointConfig(d1_cut_pct=-5.0))
    out = add_d1_checkpoint_columns(_df(), config=c)
    assert out.loc[0, "D1_CHECKPOINT_PRICE"] == 9500


def test_config_validation():
    with pytest.raises(ValueError):
        D1CheckpointConfig(d1_cut_pct=-20.0)
    with pytest.raises(ValueError):
        D1CheckpointConfig(d1_cut_pct=0.0)


def test_reason_format():
    out = add_d1_checkpoint_columns(_df())
    r = out.loc[0, "D1_CHECKPOINT_REASON"]
    assert "9,600원" in r and "-4%" in r and "조기청산" in r


def test_empty_and_none():
    assert add_d1_checkpoint_columns(pd.DataFrame()) is not None
    assert add_d1_checkpoint_columns(None) is None


def test_summary():
    s = d1_summary(add_d1_checkpoint_columns(_df()))
    assert s["n_rows"] == 3 and s["n_with_checkpoint"] == 2
