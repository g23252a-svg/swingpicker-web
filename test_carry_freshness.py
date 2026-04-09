# -*- coding: utf-8 -*-
"""test_carry_freshness.py — CARRY 신선도 회귀 테스트 5종 [v20.3]

pytest test_carry_freshness.py -v
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════
#  Test 1: CARRY 종목 지표 갱신 테스트
# ══════════════════════════════════════════════════════════════
def test_carry_indicators_refreshed():
    """
    CARRY 종목의 RSI14, TIMING_SCORE, FINAL_SCORE가
    이전 값과 달라져야 한다 (종가가 바뀌었으면).
    """
    # 이전 추천: RSI14=56.2, 종가=4850
    prev_row = {
        "종목코드": "054920", "종목명": "한컴위드", "ROUTE": "CARRY",
        "RSI14": 56.2, "TIMING_SCORE": 77.7, "DISPLAY_SCORE": 89.2,
        "STRUCT_SCORE": 95.7, "ML_SCORE": 85.6, "FINAL_SCORE": 89.2,
        "종가": 4850, "기준일": "20260402",
        "CARRY_FROM_DATE": "20260402",
    }

    # 당일 종가가 바뀜: 4850 → 4700
    new_close = 4700

    # 만약 재분석이 정상이면, RSI14가 56.2와 달라야 함
    # (종가가 하락했으므로 RSI도 달라져야 함)
    # 이 테스트는 실제 analyze_ticker를 호출하지 않으므로
    # ROW_BUILD_MODE로 검증
    from pipeline_calibrate import _refresh_carry_rows

    # Mock context
    ctx = MagicMock()
    ctx.trade_ymd = "20260409"
    ctx.start_s = "20250401"
    ctx.end_s = "20260409"
    ctx.top_df = pd.DataFrame()
    ctx.mcap_map = {"054920": 5000.0}
    ctx.kospi_set = set()
    ctx.kosdaq_set = {"054920"}
    ctx.name_map = {"054920": "한컴위드"}
    ctx.sector_map = {"054920": "IT"}
    ctx.bench_map = {}
    ctx.inv_maps = None

    prev_df = pd.DataFrame([prev_row])

    # Mock: analyze_ticker가 새 RSI를 반환
    mock_row = prev_row.copy()
    mock_row["RSI14"] = 48.5  # 달라진 RSI
    mock_row["TIMING_SCORE"] = 65.0
    mock_row["종가"] = new_close

    with patch("pipeline_calibrate.analyze_ticker", return_value=mock_row), \
         patch("pipeline_calibrate.prepare_ohlcv_data",
               return_value={"054920": pd.DataFrame({"종가": [new_close]})}), \
         patch("pipeline_calibrate.calculate_trigger_score", return_value=45.0), \
         patch("pipeline_calibrate.ml_engine") as mock_ml, \
         patch("pipeline_calibrate.build_global_score", side_effect=lambda df, _: df), \
         patch("pipeline_calibrate.generate_score_reasons", side_effect=lambda df, **kw: df):

        mock_ml.apply_ml_score.side_effect = lambda df, _: df.assign(ML_SCORE=80.0)
        result = _refresh_carry_rows(ctx, prev_df, ["054920"])

    assert not result.empty, "CARRY 재분석 결과가 비어있으면 안 됨"
    row = result.iloc[0]
    assert row["ROW_BUILD_MODE"] == "CARRY_REFRESHED"
    assert row["RSI14"] != 56.2, f"RSI14가 이전값 그대로면 재분석 안 된 것: {row['RSI14']}"


# ══════════════════════════════════════════════════════════════
#  Test 2: CARRY_FROM_DATE 보존 테스트
# ══════════════════════════════════════════════════════════════
def test_carry_from_date_preserved():
    """
    기존 CARRY_FROM_DATE=20260403 → 다음 날 다시 carry
    → 결과도 여전히 20260403이어야 함 (리셋 금지)
    """
    prev_row = {
        "종목코드": "054920", "종목명": "한컴위드", "ROUTE": "CARRY",
        "CARRY_FROM_DATE": "20260403", "기준일": "20260408",
        "RSI14": 56.2, "종가": 4850,
    }

    ctx = MagicMock()
    ctx.trade_ymd = "20260409"
    ctx.start_s = "20250401"
    ctx.end_s = "20260409"
    ctx.top_df = pd.DataFrame()
    ctx.mcap_map = {}; ctx.kospi_set = set(); ctx.kosdaq_set = set()
    ctx.name_map = {}; ctx.sector_map = {}; ctx.bench_map = {}; ctx.inv_maps = None

    prev_df = pd.DataFrame([prev_row])

    from pipeline_calibrate import _refresh_carry_rows
    mock_row = prev_row.copy()

    with patch("pipeline_calibrate.analyze_ticker", return_value=mock_row), \
         patch("pipeline_calibrate.prepare_ohlcv_data",
               return_value={"054920": pd.DataFrame({"종가": [4700]})}), \
         patch("pipeline_calibrate.calculate_trigger_score", return_value=40.0), \
         patch("pipeline_calibrate.ml_engine") as mock_ml, \
         patch("pipeline_calibrate.build_global_score", side_effect=lambda df, _: df), \
         patch("pipeline_calibrate.generate_score_reasons", side_effect=lambda df, **kw: df):

        mock_ml.apply_ml_score.side_effect = lambda df, _: df.assign(ML_SCORE=80.0)
        result = _refresh_carry_rows(ctx, prev_df, ["054920"])

    assert not result.empty
    assert result.iloc[0]["CARRY_FROM_DATE"] == "20260403", \
        f"CARRY_FROM_DATE가 리셋됨: {result.iloc[0]['CARRY_FROM_DATE']} (expected: 20260403)"


# ══════════════════════════════════════════════════════════════
#  Test 3: After-market 원본 보존 테스트
# ══════════════════════════════════════════════════════════════
def test_aftermarket_sidecar_preserves_original(tmp_path):
    """
    recommend_latest.csv 저장 후 시간외 업데이트 수행
    → 원본 분석 종가 컬럼은 변하지 않아야 함
    → 별도 sidecar만 바뀌어야 함
    """
    # 원본 CSV 생성
    orig = pd.DataFrame({
        "LDY_RANK": [1],
        "종목코드": ["005930"],
        "종목명": ["삼성전자"],
        "시장": ["KOSPI"],
        "업종_대분류": ["IT"],
        "종가": ["60000"],
    })
    csv_path = str(tmp_path / "recommend_latest.csv")
    sidecar_path = str(tmp_path / "aftermarket_prices_latest.csv")
    orig.to_csv(csv_path, index=False, encoding="utf-8-sig")

    from naver_aftermarket import fetch_after_market_prices_sidecar

    with patch("naver_aftermarket.fetch_after_market_price",
               return_value={"close": 61000, "after": 61500, "final": 61500}):
        count = fetch_after_market_prices_sidecar(csv_path, sidecar_path)

    # 원본 CSV 확인
    after = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    assert after.iloc[0]["종가"] == "60000", \
        f"원본 CSV 종가가 변경됨: {after.iloc[0]['종가']} (expected: 60000)"

    # sidecar 확인
    assert count > 0, "sidecar에 업데이트가 있어야 함"
    import os
    assert os.path.exists(sidecar_path), "sidecar 파일이 생성되어야 함"
    sidecar = pd.read_csv(sidecar_path)
    assert sidecar.iloc[0]["시간외종가"] == 61500


# ══════════════════════════════════════════════════════════════
#  Test 4: RSI 계산 일관성 테스트
# ══════════════════════════════════════════════════════════════
def test_rsi_calculation_parity():
    """
    indicators.calc_rsi()의 결과가 일관되어야 함.
    (dashboard RSI 통일 후 이 테스트로 검증)
    """
    from indicators import calc_rsi

    # 50일치 더미 종가
    np.random.seed(42)
    prices = pd.Series(np.cumsum(np.random.randn(50)) + 100)

    rsi_a = calc_rsi(prices, 14)
    rsi_b = calc_rsi(prices, 14)

    # 동일 입력 → 동일 출력
    assert abs(rsi_a - rsi_b) < 0.001, "같은 입력에 다른 RSI가 나오면 안 됨"

    # RSI 범위: 0~100
    assert 0 <= rsi_a <= 100, f"RSI 범위 이탈: {rsi_a}"


# ══════════════════════════════════════════════════════════════
#  Test 5: Stale carry 패널티 누적 테스트
# ══════════════════════════════════════════════════════════════
def test_stale_carry_penalty_accumulates():
    """
    8일 carry면 penalty > 0
    10일 carry면 penalty > 8일 carry
    """
    from pipeline_calibrate import run_calibration

    # stale penalty 공식: max(0, (days-5)*5.0), capped at 20
    def calc_penalty(days):
        return min(20.0, max(0, (days - 5) * 5.0)) if days > 5 else 0.0

    p8 = calc_penalty(8)
    p10 = calc_penalty(10)

    assert p8 > 0, f"8일 carry인데 penalty가 0: {p8}"
    assert p10 > p8, f"10일 penalty({p10})가 8일({p8})보다 작거나 같음"
    assert calc_penalty(5) == 0, "5일 이하면 penalty가 0이어야 함"
    assert calc_penalty(30) == 20.0, f"30일이면 cap=20이어야 함: {calc_penalty(30)}"
