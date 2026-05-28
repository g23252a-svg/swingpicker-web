# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from pipeline_finalize import apply_evidence_gated_no_buy_breaker, get_no_buy_breaker_rule_mask
from scripts.no_buy_breaker_backtest_v3926 import run_backtest


def _candidate_row(**overrides):
    row = {
        "종목코드": "000001",
        "종목명": "테스트",
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
        "ALPHA_%": 1.5,
    }
    row.update(overrides)
    return row


def test_no_buy_breaker_rejects_without_validated_rule():
    df = pd.DataFrame([_candidate_row()])
    out = apply_evidence_gated_no_buy_breaker(df, rules=[])
    assert int(out.loc[0, "TOP_PICK"]) == 0
    assert int(out.loc[0, "BUY_NOW_ELIGIBLE"]) == 0
    assert out.loc[0, "NO_BUY_BREAKER_DECISION"] == "REJECT_NO_VALIDATED_RULE"


def test_no_buy_breaker_rejects_insufficient_n_rule_even_when_candidate_matches():
    df = pd.DataFrame([_candidate_row()])
    rules = [{"rule_id": "RULE_A_STRUCT90_TIMING60", "n": 0, "win_rate_5d": 80, "avg_ret_5d": 5, "avg_alpha_5d": 3}]
    out = apply_evidence_gated_no_buy_breaker(df, rules=rules)
    assert int(out.loc[0, "TOP_PICK"]) == 0
    assert int(out.loc[0, "BUY_NOW_ELIGIBLE"]) == 0
    assert out.loc[0, "NO_BUY_BREAKER_DECISION"] == "REJECT_NO_VALIDATED_RULE"


def test_no_buy_breaker_promotes_max_one_when_rule_is_validated():
    df = pd.DataFrame([
        _candidate_row(종목코드="000001", 종목명="A", FINAL_SCORE=82, RR_NOW_TP1=1.3),
        _candidate_row(종목코드="000002", 종목명="B", FINAL_SCORE=78, RR_NOW_TP1=2.0),
    ])
    rules = [{"rule_id": "RULE_A_STRUCT90_TIMING60", "n": 25, "win_rate_5d": 64, "avg_ret_5d": 3.2, "avg_alpha_5d": 1.1}]
    out = apply_evidence_gated_no_buy_breaker(df, rules=rules)
    assert int(((out["TOP_PICK"].astype(int) == 1) & (out["BUY_NOW_ELIGIBLE"].astype(int) == 1)).sum()) == 1
    winner = out[out["TOP_PICK"].astype(int) == 1].iloc[0]
    assert winner["종목코드"] == "000001"
    assert winner["TOP_PICK_TYPE"] == "NO_BUY_BREAKER_VALIDATED"
    assert winner["NO_BUY_BREAKER_DECISION"] == "ALLOW_MAX_ONE_OFFICIAL_PICK"
    assert winner["NO_BUY_BREAKER_RULE_ID"] == "RULE_A_STRUCT90_TIMING60"


def test_no_buy_breaker_skips_when_existing_official_buy_exists():
    df = pd.DataFrame([_candidate_row(TOP_PICK=1, BUY_NOW_ELIGIBLE=1)])
    rules = [{"rule_id": "RULE_A_STRUCT90_TIMING60", "n": 25, "win_rate_5d": 64, "avg_ret_5d": 3.2, "avg_alpha_5d": 1.1}]
    out = apply_evidence_gated_no_buy_breaker(df, rules=rules)
    assert int(out.loc[0, "TOP_PICK"]) == 1
    assert int(out.loc[0, "BUY_NOW_ELIGIBLE"]) == 1
    assert out.loc[0, "NO_BUY_BREAKER_DECISION"] == "SKIP_EXISTING_OFFICIAL_BUY"


def test_rule_mask_blocks_red_or_orange_risk():
    df = pd.DataFrame([_candidate_row(ENTRY_RISK_LEVEL="RED"), _candidate_row(ENTRY_RISK_LEVEL="GREEN")])
    mask = get_no_buy_breaker_rule_mask(df, "RULE_A_STRUCT90_TIMING60")
    assert mask.tolist() == [False, True]


def test_backtest_requires_evidence_before_pass(tmp_path: Path):
    data = tmp_path / "data"
    out_dir = tmp_path / "out"
    data.mkdir()
    # N=3이면 실현수익률이 좋아도 PASS가 아니라 표본 부족이어야 한다.
    trade_rows = []
    for i in range(3):
        ymd = f"2026050{i+1}"
        df = pd.DataFrame([_candidate_row(종목코드="000001")])
        df.to_csv(data / f"recommend_{ymd}.csv", index=False, encoding="utf-8-sig")
        trade_rows.append({"date": ymd, "code": "000001", "net_pct": 5.0, "stop_hit": False})
    pd.DataFrame(trade_rows).to_csv(data / "backtest_top1_trades_20260599.csv", index=False, encoding="utf-8-sig")
    payload = run_backtest(str(data), str(out_dir))
    rules = pd.read_csv(out_dir / "no_buy_breaker_rules_latest.csv")
    assert payload["no_buy_days"] == 3
    assert payload["realized_rows"] == 12  # 4개 후보 룰이 각 날짜에서 후보를 만든다
    assert "PASS_PRODUCTION_GATE" not in set(rules["DECISION"])
    assert set(rules["DECISION"]).issuperset({"REJECT_INSUFFICIENT_SAMPLE"})
    assert (out_dir / "no_buy_breaker_backtest_latest.json").exists()
