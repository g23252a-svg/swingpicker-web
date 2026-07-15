# -*- coding: utf-8 -*-
"""test_honest_prob_v249.py — v24.9 정직 확률 레이어 테스트."""
import os

import numpy as np
import pandas as pd
import pytest

import honest_prob as hp


def _make_data_dir(tmp_path, n_days=30, up_ratio=0.4):
    """합성 parquet + recommend: 종목 절반은 2일 뒤 상승, 나머지는 하락하도록 구성."""
    d = tmp_path / "data"; d.mkdir(parents=True)
    bd = pd.bdate_range("2026-04-01", periods=n_days + 3)
    days = [int(t.strftime("%Y%m%d")) for t in bd]
    codes = [f"{i:06d}" for i in range(1, 26)]        # 25종목(최소표본 200 충족)
    n_up = int(len(codes) * up_ratio)
    rows = []
    for c_i, code in enumerate(codes):
        price = 10000.0
        for di, day in enumerate(days):
            rows.append({"Date": pd.Timestamp(str(day)), "종목코드": code,
                         "시가": price, "고가": price * 1.01, "저가": price * 0.99,
                         "종가": price, "거래량": 1000, "등락률": 0.0})
            price *= (1.01 if c_i < n_up else 0.99)   # 매일 +1% / -1%
    pd.DataFrame(rows).set_index("Date").to_parquet(d / "ohlcv_cache_20260601.parquet")
    for day in days[:n_days]:
        pd.DataFrame({"종목코드": codes}).to_csv(d / f"recommend_{day}.csv",
                                              index=False, encoding="utf-8-sig")
    return str(d), n_up / len(codes) * 100


def test_base_prob_math(tmp_path):
    data_dir, expect = _make_data_dir(tmp_path, up_ratio=0.4)
    b = hp.compute_market_base_prob(data_dir, lookback=20)
    assert b["ok"] and b["n_days"] == 20
    assert abs(b["prob_pct"] - expect) < 0.01, f"기대 {expect}% vs {b['prob_pct']}%"


def test_lookback_and_min_samples(tmp_path):
    data_dir, _ = _make_data_dir(tmp_path, n_days=30, up_ratio=0.6)
    b = hp.compute_market_base_prob(data_dir, lookback=10)
    assert b["ok"] and b["n_days"] == 10 and abs(b["prob_pct"] - 60.0) < 0.01
    # 표본 부족 → ok=False
    small, _ = _make_data_dir(tmp_path / "s", n_days=3)
    assert hp.compute_market_base_prob(small)["ok"] is False


def test_add_columns_and_official_unchanged(tmp_path):
    data_dir, expect = _make_data_dir(tmp_path)
    df = pd.DataFrame({"종목명": ["A", "B"], "TOP_PICK": [1, 0],
                       "BUY_NOW_ELIGIBLE": [0, 1]})
    out = hp.add_srp_columns(df, data_dir)
    for c in hp.SRP_OUTPUT_COLS:
        assert c in out.columns
    assert out["TOP_PICK"].tolist() == [1, 0]
    assert out["BUY_NOW_ELIGIBLE"].tolist() == [0, 1]
    assert (out["SRP_EDGE_STATUS"] == "NO_EDGE_VALIDATED").all()
    assert abs(out["SRP_BASE_PROB_2D"].iloc[0] - expect) < 0.01
    assert "종목별 유의차 미검증" in out["SRP_NOTE"].iloc[0]


def test_missing_data_graceful(tmp_path):
    empty = tmp_path / "nodata"; empty.mkdir()
    df = pd.DataFrame({"종목명": ["A"], "TOP_PICK": [0]})
    out = hp.add_srp_columns(df, str(empty))
    assert np.isnan(out["SRP_BASE_PROB_2D"].iloc[0])
    assert "계산불가" in out["SRP_NOTE"].iloc[0]
    assert out["TOP_PICK"].tolist() == [0]


def test_empty_df_safe(tmp_path):
    assert hp.add_srp_columns(pd.DataFrame(), "data") is not None
    assert hp.add_srp_columns(None, "data") is None


def test_card_shows_base_prob():
    from components.action_cards import build_cards_html
    df = pd.DataFrame([{"종목명": "테스트", "종목코드": "123456", "종가": 10000,
                        "추천매수가": 10000, "손절가": 9000, "추천매도가1": 12000,
                        "RSI14": 60, "DISPLAY_SCORE": 70, "ROUTE": "WAIT",
                        "거래대금(억원)": 200, "SRP_BASE_PROB_2D": 39.6,
                        "PRODUCTION_BUY": 1, "ACTION_DECISION": "BUY"}])
    h = build_cards_html(df)
    assert "시장기저 2일 상승률" in h and "40%" in h or "39" in h
    # 컬럼 없으면 표시 생략
    h2 = build_cards_html(df.drop(columns=["SRP_BASE_PROB_2D"]))
    assert "시장기저" not in h2


def test_summary():
    df = pd.DataFrame({"SRP_BASE_PROB_2D": [39.6, 39.6]})
    s = hp.srp_summary(df)
    assert s["base_prob_2d"] == 39.6 and s["status"] == "NO_EDGE_VALIDATED"
    assert hp.srp_summary(pd.DataFrame()) == {}
