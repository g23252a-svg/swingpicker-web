
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.no_buy_breaker_backtest_v3926 import run_backtest

def _write_ohlcv_parquet(data: Path, code: str = "000001", n_days: int = 12,
                         entry_touch_low: float = 99.0, base: float = 100.0,
                         daily_gain: float = 1.0):
    """v25 엔진용 합성 OHLCV — 5월 1일부터 연속 n_days, 매일 저가가 entry(100)에 도달."""
    import pandas as pd
    dates = pd.to_datetime([f"2026-05-{d:02d}" for d in range(1, n_days + 1)])
    rows = []
    for i, ts in enumerate(dates):
        c = base + daily_gain * i
        rows.append({"Date": ts, "종목코드": code, "시가": base, "고가": c + 3,
                     "저가": entry_touch_low, "종가": c, "거래량": 1000})
    pd.DataFrame(rows).set_index("Date").to_parquet(data / "ohlcv_cache_20260601.parquet")


def _recommend_row(code="000001", name="테스트", close=100.0, **overrides):
    row = {
        "종목코드": code,
        "종목명": name,
        "종가": close,
        "추천매수가": 100.0,
        "손절가": 94.0,
        "추천매도가1": 110.0,
        "ROUTE": "ARMED",
        "TOP_PICK": 0,
        "BUY_NOW_ELIGIBLE": 0,
        "BUY_NOW_PASS": 1,
        "PASS_EBS": 1,
        "거래대금(억원)": 100,
        "ENTRY_GAP_PCT": 0.0,
        "RR_NOW_TP1": 1.3,
        "STRUCT_SCORE": 95,
        "TIMING_SCORE": 65,
        "AI_SCORE": 80,
        "FINAL_SCORE": 82,
        "ELITE_SCORE": 74,
        "ENTRY_RISK_LEVEL": "GREEN",
        "VWAP_GAP": 4.0,
        "POC_GAP": 5.0,
        "MFI14": 60,
        "ret_1d_%": 0.5,
        "ret_5d_%": 3.0,
    }
    row.update(overrides)
    return row


def test_direct_recommend_close_fills_missing_backtest_trade_result(tmp_path: Path):
    """[v25 갱신] OHLCV 체결검증 엔진 — 체결·보유 완결 시 realized로 집계된다."""
    data = tmp_path / "data"
    out = tmp_path / "out"
    data.mkdir()

    dates = ["20260501", "20260502", "20260503", "20260504", "20260505", "20260506"]
    for d in dates:
        pd.DataFrame([_recommend_row(close=100)]).to_csv(
            data / f"recommend_{d}.csv", index=False, encoding="utf-8-sig")
    _write_ohlcv_parquet(data, n_days=12)  # 모든 rec의 fill+5일 보유 완결

    payload = run_backtest(str(data), str(out))
    trades = pd.read_csv(out / "no_buy_breaker_trades_latest.csv")
    rules = pd.read_csv(out / "no_buy_breaker_rules_latest.csv")

    assert payload["realized_rows"] > 0
    assert (trades["FILL_STATUS"] == "FILLED").all()
    assert trades["RET_5D"].notna().all()
    assert (trades["FILL_DAY"] == 1).all()          # D+1 저가 99 ≤ entry 100
    assert "CANDIDATE_N" in rules.columns
    assert "REALIZED_N" in rules.columns
    assert "VALIDATION_COVERAGE" in rules.columns
    assert (rules["ENGINE"] == "FILL_AWARE_OHLCV_V25").all()


def test_pending_outcome_not_counted_as_realized(tmp_path: Path):
    """[v25 갱신] 보유창 미완결(미래 데이터 부족)은 PENDING — realized/N에서 제외."""
    data = tmp_path / "data"
    out = tmp_path / "out"
    data.mkdir()

    for d in ["20260501", "20260502", "20260503"]:
        pd.DataFrame([_recommend_row(close=100)]).to_csv(
            data / f"recommend_{d}.csv", index=False, encoding="utf-8-sig")
    _write_ohlcv_parquet(data, n_days=5)  # fill은 되지만 fill+4일 종가가 없음

    payload = run_backtest(str(data), str(out))
    cand = pd.read_csv(out / "no_buy_breaker_candidates_latest.csv")
    trades = pd.read_csv(out / "no_buy_breaker_trades_latest.csv")
    rules = pd.read_csv(out / "no_buy_breaker_rules_latest.csv")

    assert payload["realized_rows"] == 0
    assert payload["pending_rows"] > 0
    assert (cand["FILL_STATUS"] == "PENDING").all()
    assert len(trades) == 0                          # trades = realized 전용
    assert (rules["N"] == 0).all()
    assert set(rules["DECISION"]) == {"REJECT_INSUFFICIENT_SAMPLE"}
