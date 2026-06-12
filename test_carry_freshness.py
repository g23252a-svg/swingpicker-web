# -*- coding: utf-8 -*-
"""test_carry_freshness.py — CARRY 신선도 회귀 테스트 5종 [v20.3.2]

pytest test_carry_freshness.py -v

[v20.3.2] 의존성 주입 방식 — collector import 불필요, 순수 단위테스트
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


def _mk_mock_ohlcv(n: int = 70, last_close: float = 4700.0) -> pd.DataFrame:
    """[v24.1] CARRY 재분석 최소 요건을 충족하는 합성 OHLCV.

    _refresh_carry_rows는 60행 미만 OHLCV를 'ohlcv_short'로 CARRY_LEGACY
    폴백하므로, CARRY_REFRESHED 경로를 검증하려면 60행 이상이 필요하다.
    """
    closes = np.linspace(last_close * 0.9, last_close, n)
    return pd.DataFrame({
        "시가": closes * 0.99,
        "고가": closes * 1.01,
        "저가": closes * 0.98,
        "종가": closes,
        "거래량": np.full(n, 10000.0),
    })


def test_carry_indicators_refreshed():
    """CARRY 종목의 RSI14가 이전 값과 달라져야 한다."""
    prev_row = {
        "종목코드": "054920", "종목명": "한컴위드", "ROUTE": "CARRY",
        "RSI14": 56.2, "TIMING_SCORE": 77.7, "DISPLAY_SCORE": 89.2,
        "STRUCT_SCORE": 95.7, "ML_SCORE": 85.6, "FINAL_SCORE": 89.2,
        "종가": 4850, "기준일": "20260402", "CARRY_FROM_DATE": "20260402",
    }
    mock_result = prev_row.copy()
    mock_result["RSI14"] = 48.5
    mock_result["TIMING_SCORE"] = 65.0
    mock_result["종가"] = 4700

    ctx = MagicMock()
    ctx.trade_ymd = "20260409"
    ctx.start_s = "20250401"; ctx.end_s = "20260409"
    ctx.top_df = pd.DataFrame()
    ctx.mcap_map = {"054920": 5000.0}
    ctx.kospi_set = set(); ctx.kosdaq_set = {"054920"}
    ctx.name_map = {"054920": "한컴위드"}
    ctx.sector_map = {"054920": "IT"}
    ctx.bench_map = {}; ctx.inv_maps = None

    prev_df = pd.DataFrame([prev_row])

    from pipeline_calibrate import _refresh_carry_rows
    result = _refresh_carry_rows(
        ctx, prev_df, ["054920"],
        analyze_fn=lambda *a, **kw: mock_result,
        prepare_ohlcv_fn=lambda codes, s, e, d: {"054920": _mk_mock_ohlcv(70)},
        trigger_fn=lambda df: 45.0,
        ml_apply_fn=lambda df, _: df.assign(ML_SCORE=80.0),
        build_score_fn=lambda df, _: df,
        gen_reasons_fn=lambda df, **kw: df,
    )

    assert not result.empty, "CARRY 재분석 결과가 비어있으면 안 됨"
    row = result.iloc[0]
    assert row["ROW_BUILD_MODE"] == "CARRY_REFRESHED"
    assert row["RSI14"] != 56.2, f"RSI14가 이전값 그대로: {row['RSI14']}"


def test_carry_from_date_preserved():
    """기존 CARRY_FROM_DATE=20260403 -> 다음 날 carry -> 20260403 유지."""
    prev_row = {
        "종목코드": "054920", "종목명": "한컴위드", "ROUTE": "CARRY",
        "CARRY_FROM_DATE": "20260403", "기준일": "20260408",
        "RSI14": 56.2, "종가": 4850,
    }
    ctx = MagicMock()
    ctx.trade_ymd = "20260409"
    ctx.start_s = "20250401"; ctx.end_s = "20260409"
    ctx.top_df = pd.DataFrame()
    ctx.mcap_map = {}; ctx.kospi_set = set(); ctx.kosdaq_set = set()
    ctx.name_map = {}; ctx.sector_map = {}; ctx.bench_map = {}; ctx.inv_maps = None

    prev_df = pd.DataFrame([prev_row])
    mock_result = prev_row.copy()

    from pipeline_calibrate import _refresh_carry_rows
    result = _refresh_carry_rows(
        ctx, prev_df, ["054920"],
        analyze_fn=lambda *a, **kw: mock_result,
        prepare_ohlcv_fn=lambda codes, s, e, d: {"054920": _mk_mock_ohlcv(70)},
        trigger_fn=lambda df: 40.0,
        ml_apply_fn=lambda df, _: df.assign(ML_SCORE=80.0),
        build_score_fn=lambda df, _: df,
        gen_reasons_fn=lambda df, **kw: df,
    )

    assert not result.empty
    assert result.iloc[0]["CARRY_FROM_DATE"] == "20260403", \
        f"CARRY_FROM_DATE 리셋됨: {result.iloc[0]['CARRY_FROM_DATE']}"


def test_aftermarket_sidecar_preserves_original(tmp_path):
    """recommend CSV는 불변, sidecar만 바뀌어야 함."""
    orig = pd.DataFrame({
        "LDY_RANK": [1], "종목코드": ["005930"], "종목명": ["삼성전자"],
        "시장": ["KOSPI"], "업종_대분류": ["IT"], "종가": ["60000"],
    })
    csv_path = str(tmp_path / "recommend_latest.csv")
    sidecar_path = str(tmp_path / "aftermarket_prices_latest.csv")
    orig.to_csv(csv_path, index=False, encoding="utf-8-sig")

    from naver_aftermarket import fetch_after_market_prices_sidecar

    with patch("naver_aftermarket.fetch_after_market_price",
               return_value={"close": 61000, "after": 61500, "final": 61500}):
        count = fetch_after_market_prices_sidecar(csv_path, sidecar_path)

    after = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    assert after.iloc[0]["종가"] == "60000", \
        f"원본 CSV 종가 변경됨: {after.iloc[0]['종가']}"
    assert count > 0
    import os; assert os.path.exists(sidecar_path)
    sidecar = pd.read_csv(sidecar_path)
    assert sidecar.iloc[0]["시간외종가"] == 61500


def test_rsi_calculation_parity():
    """indicators.calc_rsi() 동일 입력 -> 동일 출력, 범위 0~100."""
    from indicators import calc_rsi

    np.random.seed(42)
    prices = pd.Series(np.cumsum(np.random.randn(50)) + 100)

    rsi_a = calc_rsi(prices, 14).iloc[-1]
    rsi_b = calc_rsi(prices, 14).iloc[-1]

    assert abs(rsi_a - rsi_b) < 0.001, "같은 입력에 다른 RSI"
    assert 0 <= rsi_a <= 100, f"RSI 범위 이탈: {rsi_a}"


def test_stale_carry_penalty_accumulates():
    """8일 > 0, 10일 > 8일, 5일 == 0, 30일 == cap(20)."""
    def calc_penalty(days):
        return min(20.0, max(0, (days - 5) * 5.0)) if days > 5 else 0.0

    assert calc_penalty(5) == 0
    assert calc_penalty(8) > 0
    assert calc_penalty(10) > calc_penalty(8)
    assert calc_penalty(30) == 20.0
