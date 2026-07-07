# -*- coding: utf-8 -*-
"""
test_nbb_certifier_v250.py — v25.0 Fill-Aware 룰 인증기 테스트.

보증
  1. [PASS 개통] 정직 기준(N≥20·승률·알파·OOS)을 실제로 충족하면
     PASS_PRODUCTION_GATE가 발급되고, 파이프라인 로더 기준으로 파싱된다.
  2. [OOS 안전조건] 전체 성과가 좋아도 후반기(OOS) 승률<50이면 REJECT_OOS_WEAK.
  3. [유령 차단] 창 완결·미도달 후보는 UNFILLED로 분리 — N에서 제외, N_UNFILLED 집계.
  4. [스키마] 선두 17컬럼 순서 불변(구버전 호환) + ENGINE 태그.
  5. [강건성] OHLCV parquet 부재 시에도 정상 산출(전룰 REJECT_INSUFFICIENT_SAMPLE).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.no_buy_breaker_backtest_v3926 import (
    run_backtest, RULE_SCHEMA_COLS, MIN_N, ENGINE_TAG,
)

LEGACY_17 = [
    "RULE_ID", "RULE_DESC", "CANDIDATE_N", "REALIZED_N", "PENDING_N",
    "VALIDATION_COVERAGE", "N", "WIN_RATE_5D", "AVG_RET_5D", "MEDIAN_RET_5D",
    "AVG_ALPHA_5D", "STOP_RISK_RATE", "MAX_LOSS_AVG", "RECENT_N",
    "RECENT_AVG_RET_5D", "DECISION", "REJECT_REASON",
]


def _rule_d_only_row(code: str, close: float = 100.0) -> dict:
    """RULE_D만 매칭 (A: STRUCT<90 / B: FINAL<80 / C: STRUCT<95 로 차단)."""
    return {
        "종목코드": code, "종목명": f"T{code}", "종가": close,
        "추천매수가": 100.0, "손절가": 94.0, "추천매도가1": 110.0,
        "ROUTE": "ARMED", "TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0,
        "BUY_NOW_PASS": 1, "PASS_EBS": 1, "거래대금(억원)": 100,
        "ENTRY_GAP_PCT": 0.0, "RR_NOW_TP1": 1.3,
        "STRUCT_SCORE": 80, "TIMING_SCORE": 65, "AI_SCORE": 80,
        "FINAL_SCORE": 78, "ELITE_SCORE": 70, "ENTRY_RISK_LEVEL": "GREEN",
        "VWAP_GAP": 4.0, "POC_GAP": 5.0, "MFI14": 60,
        "ret_1d_%": 0.5, "ret_5d_%": 3.0,
    }


def _write_stock_path(rows: list, code: str, start_day: int, rets: float,
                      n_days: int = 8, entry: float = 100.0):
    """code의 연속 n_days OHLC — D+1 저가로 체결, 5일째 종가 = entry*(1+rets/100)."""
    exit_close = entry * (1.0 + rets / 100.0)
    for k in range(n_days):
        day = start_day + k
        close = entry if k == 0 else (exit_close if k >= 5 else entry + 0.5)
        low = entry - 1.0 if k >= 1 else entry            # D+1부터 체결 가능
        rows.append({"Date": pd.Timestamp(f"2026-05-{day:02d}"), "종목코드": code,
                     "시가": entry, "고가": max(close, entry) + 2, "저가": low,
                     "종가": close, "거래량": 1000})


def _build_universe(tmp_path: Path, rets_by_day: list):
    """rec day i(1-base)마다 고유 종목 1개 — 성과를 rets_by_day[i-1]로 지정."""
    data = tmp_path / "data"; data.mkdir()
    ohlc_rows = []
    for i, r in enumerate(rets_by_day, start=1):
        code = f"{i:06d}"
        pd.DataFrame([_rule_d_only_row(code)]).to_csv(
            data / f"recommend_202605{i:02d}.csv", index=False, encoding="utf-8-sig")
        _write_stock_path(ohlc_rows, code, start_day=i, rets=r)
    pd.DataFrame(ohlc_rows).set_index("Date").to_parquet(data / "ohlcv_cache_20260701.parquet")
    # KOSPI: 전 기간 평평 → fwd5 = 0 → ALPHA == RET
    kd = pd.DataFrame({"date": [int(f"202605{d:02d}") for d in range(1, 32)],
                       "close": [1000.0] * 31})
    kd.to_csv(data / "kospi_daily.csv", index=False, encoding="utf-8-sig")
    return data


def test_pass_production_gate_opens_on_honest_evidence(tmp_path: Path):
    # 초반 4패(-1%) + 이후 20승(+8%) = N 24 · 승률 83 · OOS(후반12) 전승
    rets = [-1.0] * 4 + [8.0] * 20
    data = _build_universe(tmp_path, rets)
    payload = run_backtest(str(data), str(tmp_path / "out"))
    rules = pd.read_csv(tmp_path / "out" / "no_buy_breaker_rules_latest.csv")
    d = rules[rules.RULE_ID == "RULE_D_ROUTE_ARMED_CLEAN_ENTRY"].iloc[0]

    assert d["DECISION"] == "PASS_PRODUCTION_GATE"
    assert int(d["N"]) == 24 >= MIN_N
    assert d["WIN_RATE_5D"] > 55 and d["AVG_RET_5D"] > 0 and d["AVG_ALPHA_5D"] > 0
    assert int(d["OOS_N"]) >= 8 and d["OOS_WIN_RATE_5D"] >= 50
    # 다른 룰은 후보 0 → 표본부족
    others = rules[rules.RULE_ID != "RULE_D_ROUTE_ARMED_CLEAN_ENTRY"]
    assert set(others["DECISION"]) == {"REJECT_INSUFFICIENT_SAMPLE"}
    # 파이프라인 로더 기준(DECISION==PASS & N≥MIN_N) 파싱 → 1개 개통
    passed = rules[(rules.DECISION == "PASS_PRODUCTION_GATE") & (rules.N >= MIN_N)]
    assert len(passed) == 1
    assert payload["realized_rows"] == 24 and payload["unfilled_rows"] == 0


def test_oos_weakness_blocks_pass(tmp_path: Path):
    # 전반 12승(+8) + 후반 5승7패 → 전체 승률 70·평균>0이지만 OOS 승률 41.7 → 차단
    rets = [8.0] * 12 + ([8.0] * 5 + [-1.0] * 7)
    data = _build_universe(tmp_path, rets)
    run_backtest(str(data), str(tmp_path / "out"))
    rules = pd.read_csv(tmp_path / "out" / "no_buy_breaker_rules_latest.csv")
    d = rules[rules.RULE_ID == "RULE_D_ROUTE_ARMED_CLEAN_ENTRY"].iloc[0]
    assert d["WIN_RATE_5D"] > 55 and d["AVG_RET_5D"] > 0     # 전체는 통과권
    assert d["DECISION"] == "REJECT_OOS_WEAK"
    assert d["REJECT_REASON"] == "OOS_WIN_RATE_LOW"


def test_unfilled_counted_and_excluded(tmp_path: Path):
    data = tmp_path / "data"; data.mkdir()
    pd.DataFrame([_rule_d_only_row("000001")]).to_csv(
        data / "recommend_20260501.csv", index=False, encoding="utf-8-sig")
    rows = []
    for k in range(8):  # 저가가 entry(100) 위 → 절대 미체결, 창은 완결
        rows.append({"Date": pd.Timestamp(f"2026-05-{k+1:02d}"), "종목코드": "000001",
                     "시가": 106, "고가": 110, "저가": 105, "종가": 108, "거래량": 1000})
    pd.DataFrame(rows).set_index("Date").to_parquet(data / "ohlcv_cache_20260701.parquet")

    payload = run_backtest(str(data), str(tmp_path / "out"))
    rules = pd.read_csv(tmp_path / "out" / "no_buy_breaker_rules_latest.csv")
    d = rules[rules.RULE_ID == "RULE_D_ROUTE_ARMED_CLEAN_ENTRY"].iloc[0]
    assert payload["unfilled_rows"] == 1 and payload["realized_rows"] == 0
    assert int(d["N_UNFILLED"]) == 1 and int(d["N"]) == 0
    assert d["FILL_RATE_PCT"] == 0.0
    assert d["DECISION"] == "REJECT_INSUFFICIENT_SAMPLE"


def test_schema_first17_order_and_engine(tmp_path: Path):
    data = _build_universe(tmp_path, [8.0] * 3)
    run_backtest(str(data), str(tmp_path / "out"))
    rules = pd.read_csv(tmp_path / "out" / "no_buy_breaker_rules_latest.csv")
    assert list(rules.columns[:17]) == LEGACY_17
    assert list(rules.columns) == RULE_SCHEMA_COLS
    assert (rules["ENGINE"] == ENGINE_TAG).all()


def test_graceful_without_parquet(tmp_path: Path):
    data = tmp_path / "data"; data.mkdir()
    pd.DataFrame([_rule_d_only_row("000001")]).to_csv(
        data / "recommend_20260501.csv", index=False, encoding="utf-8-sig")
    payload = run_backtest(str(data), str(tmp_path / "out"))
    rules = pd.read_csv(tmp_path / "out" / "no_buy_breaker_rules_latest.csv")
    assert set(rules["DECISION"]) == {"REJECT_INSUFFICIENT_SAMPLE"}
    assert payload["realized_rows"] == 0
    assert (tmp_path / "out" / "no_buy_breaker_backtest_latest.json").exists()
