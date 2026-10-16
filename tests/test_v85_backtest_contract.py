# -*- coding: utf-8 -*-
"""[v85] 백테스트를 기록된 공식 매수(PRODUCTION_BUY) 기준으로 재구성 + 동결 감지.

고정하는 것:
  1. selector 기본값은 'legacy' — tune/walk-forward/rolling 호출부 무변경.
  2. contract 선택은 그날 CSV에 **기록된** PRODUCTION_BUY만 (재구성 금지, v65).
     컬럼 없는 CSV = 판정 불가(현금일 아님), 컬럼 있고 픽 없음 = 현금일.
  3. 성숙도 — 3봉 전 미체결은 PENDING, horizon 미완료 무터치는 PENDING,
     NOT_FILLED·PENDING은 승률/EV에 0.0으로 섞지 않는다.
  4. save_report 원자성 — 계산 중 예외면 이전 latest JSON 그대로, 거래 0건이어도
     daily_port NameError 없음, 섞인 키로 DictWriter가 터지지 않음.
  5. freshness/stale 판정, 성적표 제목(옛/새 JSON), 하드코딩 '99일' 제거.
  6. official_buy_validation·tab_perf가 최신·확정 스냅샷을 고른다.
  7. 실데이터: CSV에서 직접 센 공식픽 일수 == 선택기 결과 (JSON 미사용).
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_validation as bv  # noqa: E402


class _DummyUi:
    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if "nicegui" not in sys.modules:
    _stub = types.ModuleType("nicegui")
    _stub.ui = _DummyUi()
    _stub.run = _DummyUi()
    _stub.app = _DummyUi()
    sys.modules["nicegui"] = _stub


# ─────────────────────────── 합성 데이터 헬퍼 ───────────────────────────

def _row(code, pb=None, name="종목", entry=10000.0, stop=9500.0, tp1=11000.0,
         alpha=50.0, rr=1.0, **extra):
    r = {
        "종목코드": code, "종목명": name, "업종": "업종" + code[-1],
        "종가": str(entry), "추천매수가": str(entry), "손절가": str(stop),
        "추천매도가1": str(tp1), "추천매도가2": "0", "추천매도가3": "0",
        "ALPHA_SCORE": str(alpha), "RR_NOW_TP1": str(rr),
        "STRUCT_SCORE": "0", "TIMING_SCORE": "0", "AI_SCORE": "0",
    }
    if pb is not None:
        r["PRODUCTION_BUY"] = str(pb)
    r.update({k: str(v) for k, v in extra.items()})
    return r


def _day(ymd, rows):
    return (ymd, rows, {r["종목코드"]: r for r in rows})


def _ohlc(code, bars):
    """bars: [(YYYY-MM-DD, open, high, low, close)]"""
    return pd.DataFrame([
        {"종목코드": code, "Date": pd.Timestamp(d), "시가": o, "고가": h, "저가": lo, "종가": c}
        for d, o, h, lo, c in bars
    ])


def _bdays(start, n):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


# ─────────────────────────── 1. selector 기본값 ───────────────────────────

def test_selector_default_is_legacy():
    for fn in (bv.daily_top1_backtest, bv.daily_top3_backtest):
        assert inspect.signature(fn).parameters["selector"].default == bv.SELECTOR_LEGACY == "legacy"


def test_internal_legacy_callers_pin_legacy_selector():
    src = inspect.getsource(bv.tune_strong_thresholds)
    assert "selector=SELECTOR_LEGACY" in src
    assert "selector=SELECTOR_LEGACY" in inspect.getsource(bv.walk_forward_validate)
    assert "selector=SELECTOR_LEGACY" in inspect.getsource(bv.rolling_walk_forward)
    # nightly 헤드라인만 contract를 명시
    assert "selector=SELECTOR_CONTRACT" in inspect.getsource(bv.build_report)


def test_unknown_selector_raises():
    with pytest.raises(ValueError):
        bv.daily_top1_backtest([], pd.DataFrame(), selector="bogus")


def test_legacy_picker_untouched_ignores_production_buy():
    # 라벨 경로는 PRODUCTION_BUY를 보지 않는다 — AI 0인 행은 라벨이 안 나온다 (0713 이후 실상)
    rows = [_row("000001", pb=1)]
    assert bv.pick_top1_codes(rows) == []
    assert bv.pick_top3_codes(rows) == []


# ─────────────────────────── 2. recorded-only ───────────────────────────

def test_contract_picks_only_recorded_production_buy():
    rows = [
        _row("000001", pb=0, TOP_PICK=1, BUY_NOW_ELIGIBLE=1),   # 재구성하면 픽이지만 기록은 0
        _row("000002", pb=1, alpha=40, rr=1.0),
        _row("000003", pb="1.0", alpha=90, rr=2.0),
        _row("000004", pb=""),
    ]
    status, picks = bv.classify_contract_day(rows)
    assert status == bv.DAY_PICK
    assert [p["code"] for p in picks] == ["000003", "000002"]   # 엔진 축(알파×RR) 내림차순
    assert all(p["selection_source"] == bv.SOURCE_PRODUCTION_BUY for p in picks)
    assert all(p["label"] == bv.CONTRACT_LABEL for p in picks)
    assert [p["code"] for p in bv.select_official_picks(rows, max_n=1)] == ["000003"]


def test_recorded_flag_semantics_match_production_buy_mask():
    from services.recommendation_quality import production_buy_mask
    vals = ["1", "1.0", "true", "True", "y", "0", "0.0", "", "nan", "false", "2"]
    rows = [_row(f"{i:06d}", pb=v) for i, v in enumerate(vals, 1)]
    flags = bv._recorded_production_buy_flags(rows)
    expect = production_buy_mask(pd.DataFrame({"PRODUCTION_BUY": vals})).tolist()
    assert flags == expect


def test_no_column_is_undeterminable_not_fallback():
    # TOP_PICK & BUY_NOW_ELIGIBLE가 있어도 PRODUCTION_BUY 컬럼이 없으면 판정 불가
    rows = [_row("000001", TOP_PICK=1, BUY_NOW_ELIGIBLE=1)]
    status, picks = bv.classify_contract_day(rows)
    assert status == bv.DAY_UNDETERMINABLE and picks == []
    assert bv.select_official_picks(rows) is None
    assert bv.classify_contract_day([])[0] == bv.DAY_UNDETERMINABLE


def test_column_present_without_pick_is_cash_day():
    rows = [_row("000001", pb=0), _row("000002", pb=0)]
    status, picks = bv.classify_contract_day(rows)
    assert status == bv.DAY_CASH and picks == []
    assert bv.select_official_picks(rows) == []


def test_daily_log_marks_undeterminable_cash_and_pick_days():
    days = [
        _day("20260701", [_row("000009")]),                 # hist base (루프 제외)
        _day("20260702", [_row("000001")]),                 # 판정 불가
        _day("20260703", [_row("000001", pb=0)]),           # 현금
        _day("20260706", [_row("000001", pb=1)]),           # 공식픽
    ]
    res = bv.daily_top1_backtest(days, pd.DataFrame(), selector="contract")
    st = {e["date"]: e["day_status"] for e in res["daily_picks"]}
    assert st == {"20260702": "undeterminable", "20260703": "cash", "20260706": "pick"}
    assert [e["n_picked"] for e in res["daily_picks"]] == [0, 0, 1]


def test_load_all_days_glob_covers_2027_and_skips_non_batches(tmp_path):
    for name in ("recommend_20261230.csv", "recommend_20270104.csv",
                 "recommend_latest.csv", "recommend_20270105_backup.csv"):
        pd.DataFrame([{"종목코드": "000001", "종목명": "x"}]).to_csv(tmp_path / name, index=False)
    days = bv.load_all_days(tmp_path)
    assert [d[0] for d in days] == ["20261230", "20270104"]


# ─────────────────────────── 3. 성숙도 ───────────────────────────

def test_maturity_not_filled_needs_three_bars():
    d = _bdays("2026-08-03", 5)
    # 신호 8/3, 이후 봉 전부 갭업(open > entry*1.02) → 체결 불가
    gap = [(x, 10500, 10600, 10400, 10500) for x in d[1:]]
    sig = pd.Timestamp(d[0])
    one = bv.simulate_ohlc("000001", sig, 10000, 11000, 9500, 10, _ohlc("000001", gap[:1]),
                           market_bars_after=1)
    assert one["outcome"] == "PENDING" and one["matured"] is False
    three = bv.simulate_ohlc("000001", sig, 10000, 11000, 9500, 10, _ohlc("000001", gap[:3]),
                             market_bars_after=3)
    assert three["outcome"] == "NOT_FILLED" and three["matured"] is True
    # 종목 봉은 1개뿐이어도 시장은 3봉 지남(거래정지 등) → 미체결 확정
    halted = bv.simulate_ohlc("000001", sig, 10000, 11000, 9500, 10, _ohlc("000001", gap[:1]),
                              market_bars_after=3)
    assert halted["outcome"] == "NOT_FILLED"


def test_maturity_open_needs_full_horizon_but_touch_is_final():
    d = _bdays("2026-08-03", 13)
    sig = pd.Timestamp(d[0])
    flat = [(x, 10000, 10100, 9900, 10000) for x in d[1:]]           # 체결, 무터치
    partial = bv.simulate_ohlc("000001", sig, 10000, 11000, 9500, 10,
                               _ohlc("000001", flat[:5]), market_bars_after=5)
    assert partial["outcome"] == "PENDING" and partial["fill_date"] == d[1]
    full = bv.simulate_ohlc("000001", sig, 10000, 11000, 9500, 10,
                            _ohlc("000001", flat[:10]), market_bars_after=10)
    assert full["outcome"] == "OPEN" and full["matured"] is True
    # 둘째 날 손절 터치 — 봉 2개뿐이어도 결과 확정
    stop2 = flat[:1] + [(d[2], 9800, 9850, 9400, 9500)]
    loss = bv.simulate_ohlc("000001", sig, 10000, 11000, 9500, 10,
                            _ohlc("000001", stop2), market_bars_after=2)
    assert loss["outcome"] == "LOSS" and loss["matured"] is True


def test_maturity_no_bars_yet_is_pending_and_legacy_unchanged():
    sig = pd.Timestamp("2026-08-03")
    prev = _ohlc("000001", [("2026-07-31", 10000, 10100, 9900, 10000)])
    r = bv.simulate_ohlc("000001", sig, 10000, 11000, 9500, 10, prev, market_bars_after=0)
    assert r["outcome"] == "PENDING" and r["method"] == "pending_no_bars"
    # market_bars_after=None → 예전 동작 (NODATA / 부분 horizon OPEN)
    assert bv.simulate_ohlc("000001", sig, 10000, 11000, 9500, 10, prev)["outcome"] == "NODATA"
    d = _bdays("2026-08-03", 6)
    flat = _ohlc("000001", [(x, 10000, 10100, 9900, 10000) for x in d[1:]])
    assert bv.simulate_ohlc("000001", sig, 10000, 11000, 9500, 10, flat)["outcome"] == "OPEN"


def test_newest_official_pick_appears_as_pending_trade():
    # 신호 당일 밤(가격 봉 0개) — 예전엔 NODATA로 버려져 최신 공식픽이 사라졌다
    days = [_day("20260731", [_row("000009")]), _day("20260803", [_row("000001", pb=1)])]
    ohlc = _ohlc("000001", [("2026-07-31", 10000, 10100, 9900, 10000),
                            ("2026-08-03", 10000, 10100, 9900, 10000)])
    res = bv.daily_top1_backtest(days, ohlc, selector="contract")
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t["outcome"] == "PENDING" and t["net_pct"] is None and t["matured"] is False


# ─────────────────────── 4. NOT_FILLED·PENDING 집계 제외 ───────────────────────

def _trade(outcome, ret=None, date="20260803", code="000001"):
    pick = {"code": code, "name": "x", "sector": "s", "label": bv.CONTRACT_LABEL,
            "score": 1.0, "entry": 100.0, "tp1": 110.0, "stop": 95.0,
            "selection_source": bv.SOURCE_PRODUCTION_BUY}
    exit_px = 100.0 * (1 + (ret or 0) / 100)
    res = {"outcome": outcome, "exit_price": exit_px, "days_held": 3,
           "method": "ohlc_high_touch", "fill_date": "2026-08-04",
           "matured": outcome != "PENDING", "bars_after_signal": 5}
    return bv._make_trade_row(date, pick, res)


def test_not_filled_and_pending_not_counted_as_zero():
    trades = [_trade("WIN", 10.0), _trade("LOSS", -5.0, code="000002"),
              _trade("NOT_FILLED", code="000003"), _trade("PENDING", code="000004")]
    nf = trades[2]
    assert nf["ret_pct"] is None and nf["net_pct"] is None and nf["exit_price"] == 0.0
    s = bv.summarize_trades(trades)
    assert s["n"] == 2                       # 체결 확정만
    assert s["n_all_picks"] == 3             # 결과가 정해진 신호 (PENDING 제외)
    assert s["n_not_filled"] == 1 and s["n_pending"] == 1
    assert s["n_signals_incl_pending"] == 4
    assert s["win_rate"] == 0.5              # NOT_FILLED 0.0이 섞였다면 1/3
    assert s["ev"] == round((0.5 * 10.0 + 0.5 * -5.0) - bv.COST_PCT, 2)
    cap = bv.simulate_capital_portfolio(trades, max_positions=1)
    assert cap["n_filled"] + cap["n_skipped_slot_full"] + cap["n_skipped_duplicate"] == 2
    assert cap["skip_reasons_summary"]["PENDING"] == 1


def test_trade_rows_have_uniform_keys():
    rows = [_trade(o, 1.0, code=f"00000{i}") for i, o in
            enumerate(("WIN", "LOSS", "OPEN", "NOT_FILLED", "PENDING", "NODATA"))]
    for r in rows:
        assert tuple(r.keys()) == bv.TRADE_FIELDS


def test_csv_text_tolerates_mixed_keys():
    text = bv._csv_text([{"a": 1}, {"a": 2, "selection_source": "PRODUCTION_BUY"}])
    lines = text.strip().splitlines()
    assert lines[0] == "a,selection_source"
    assert lines[1] == "1," and lines[2] == "2,PRODUCTION_BUY"


# ─────────────────────────── 5. save_report 원자성 ───────────────────────────

def _mini_report(trades):
    return {
        "generated_at": "2026-10-16T21:00:00",
        "methodology": {"selection_mode": "production_contract"},
        "all_trades": trades, "daily_picks_log": [],
        "all_trades_top1": trades, "daily_top1_backtest": bv.summarize_trades(trades),
        "signal_top1": {}, "freshness": {"stale": False},
    }


def test_save_report_no_trades_has_no_nameerror(tmp_path):
    light = bv.save_report(_mini_report([]), tmp_path, today="20261016")
    j = json.loads((tmp_path / "backtest_validation_latest.json").read_text(encoding="utf-8"))
    assert j["confidence"]["level"] == "LOW"
    assert light["confidence"] == j["confidence"]
    assert (tmp_path / "backtest_validation_20261016.json").exists()


def test_save_report_crash_leaves_previous_json_intact(tmp_path, monkeypatch):
    prev = {"marker": "previous", "capital_portfolio_top1": {"total_return_pct": -38.0}}
    latest = tmp_path / "backtest_validation_latest.json"
    latest.write_text(json.dumps(prev), encoding="utf-8")

    def _boom(*a, **k):
        raise RuntimeError("simulated crash in capital sim")

    monkeypatch.setattr(bv, "simulate_capital_portfolio", _boom)
    with pytest.raises(RuntimeError):
        bv.save_report(_mini_report([_trade("WIN", 5.0)]), tmp_path, today="20261016")
    assert json.loads(latest.read_text(encoding="utf-8")) == prev
    # 계산 단계에서 터졌으므로 어떤 산출물도 생기지 않았다
    assert sorted(p.name for p in tmp_path.iterdir()) == ["backtest_validation_latest.json"]


def test_atomic_write_failure_keeps_old_file_and_cleans_tmp(tmp_path, monkeypatch):
    target = tmp_path / "x.json"
    target.write_text("OLD", encoding="utf-8")

    def _fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(bv.os, "replace", _fail_replace)
    with pytest.raises(OSError):
        bv._atomic_write_text(target, "NEW")
    assert target.read_text(encoding="utf-8") == "OLD"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_save_report_writes_full_json_and_selection_source(tmp_path):
    trades = [_trade("WIN", 10.0), _trade("NOT_FILLED", code="000002", date="20260804"),
              _trade("PENDING", code="000003", date="20260805")]
    bv.save_report(_mini_report(trades), tmp_path, today="20261016")
    j = json.loads((tmp_path / "backtest_validation_latest.json").read_text(encoding="utf-8"))
    for k in ("capital_portfolio", "capital_portfolio_top1", "confidence", "daily_portfolio_summary"):
        assert k in j
    # 결과 확정 거래가 없는 날(NOT_FILLED·PENDING만)은 일별 포트에서 빠진다
    assert j["daily_portfolio_summary"]["n_days"] == 1
    df = pd.read_csv(tmp_path / "backtest_top1_trades_20261016.csv", dtype={"code": str})
    assert list(df.columns) == list(bv.TRADE_FIELDS)
    assert set(df["selection_source"]) == {"PRODUCTION_BUY"}
    assert df.loc[df["outcome"] == "NOT_FILLED", "net_pct"].isna().all()


def test_confidence_ignores_legacy_rolling_for_contract_headline():
    rep = _mini_report([_trade("WIN", 5.0)])
    rep["rolling_walk_forward"] = {"summary": {"n_valid": 3, "robust": True, "n_generalizes": 3}}
    light, _outputs, _notes = bv.build_report_outputs(rep, "20261016")
    assert light["confidence"]["rolling_valid_folds"] == 0
    assert light["confidence"]["rolling_robust"] is False
    assert light["rolling_summary"]["selection"] == "legacy_label"


# ─────────────────────────── 6. freshness / stale ───────────────────────────

def _contract_days():
    return [
        _day("20260710", [_row("000009")]),
        _day("20260713", [_row("000001")]),                 # 판정 불가
        _day("20260714", [_row("000001", pb=0)]),
        _day("20260715", [_row("000001", pb=1)]),
        _day("20260716", [_row("000002", pb=1)]),
        _day("20260717", [_row("000002", pb=0)]),
    ]


def test_freshness_fresh_when_every_pick_has_a_trade():
    days = _contract_days()
    trades = [{"date": "20260715", "outcome": "LOSS"}, {"date": "20260716", "outcome": "PENDING"}]
    fr = bv.compute_freshness(days, trades)
    assert fr["latest_batch_date"] == "20260717"
    assert fr["last_signal_date"] == "20260716"
    assert fr["contract_pick_days"] == 2 and fr["contract_cash_days"] == 2
    assert fr["undeterminable_days"] == 1
    assert fr["contract_pick_days_unselected"] == 0
    assert fr["pending"] == 1
    assert fr["stale"] is False and fr["stale_reason"] == ""


def test_freshness_stale_when_backtest_skips_official_picks():
    fr = bv.compute_freshness(_contract_days(), trades=[])     # 동결된 라벨 경로 재현
    assert fr["contract_pick_days_unselected"] == 2
    assert fr["unselected_dates"] == ["20260715", "20260716"]
    assert fr["stale"] is True and "백테스트에 없음" in fr["stale_reason"]


def test_freshness_stale_on_trailing_undeterminable_and_price_lag():
    days = _contract_days() + [_day("20260720", [_row("000003")])]   # 컬럼 사라짐
    trades = [{"date": "20260715", "outcome": "LOSS"}, {"date": "20260716", "outcome": "LOSS"}]
    fr = bv.compute_freshness(days, trades)
    assert fr["trailing_undeterminable_batches"] == 1 and fr["stale"] is True
    import numpy as np
    cal = np.array(["2026-07-10"], dtype="datetime64[ns]")
    fr2 = bv.compute_freshness(_contract_days(), trades, calendar=cal)
    assert fr2["price_lag_batches"] == 5 and fr2["stale"] is True
    assert "가격 데이터" in fr2["stale_reason"]


def test_freshness_lag_rule():
    days = [_day("20260710", [_row("000009")])]
    days += [_day(f"202608{d:02d}", [_row("000001", pb=1 if d in (3, 10) else 0)])
             for d in (3, 4, 5, 6, 7, 10)]
    # 마지막 신호 0803, 최신 공식픽 0810 → 0804·0805·0806·0807·0810 = 5배치 지연 > 3
    fr = bv.compute_freshness(days, [{"date": "20260803", "outcome": "LOSS"}])
    assert fr["lag_batches_to_last_contract_pick"] == 5
    assert fr["stale"] is True


# ─────────────────────────── 7. official_ssot ───────────────────────────

def test_official_ssot_matches_pick_history_realized():
    from services.pick_history import _realized
    d = _bdays("2026-08-03", 12)
    a = [(x, 100, 101, 99, 100) for x in d]
    a[1] = (d[1], 100, 101, 91, 95)                       # t+1 진입 100 → 저가 91 ≤ 92 손절
    b = [(x, 100, 104, 99, 100 + i) for i, x in enumerate(d)]  # 무손절 → t+6 종가
    px = pd.concat([_ohlc("000001", a), _ohlc("000002", b)], ignore_index=True)
    log = [{"date": "20260803", "codes": ["000001"], "day_status": "pick"},
           {"date": "20260804", "codes": ["000002"], "day_status": "pick"},
           {"date": d[-1].replace("-", ""), "codes": ["000002"], "day_status": "pick"},
           {"date": "20260805", "codes": [], "day_status": "cash"}]
    out = bv.compute_official_ssot(log, [], px)
    assert out["n_official"] == 3 and out["n_measured"] == 2 and out["n_pending"] == 1
    exp_a = _realized(px[px["종목코드"] == "000001"], "20260803") * 100
    exp_b = _realized(px[px["종목코드"] == "000002"], "20260804") * 100
    assert exp_a == pytest.approx(-8.0)
    assert [p["ret_pct"] for p in out["picks"][:2]] == [round(exp_a, 2), round(exp_b, 2)]
    assert out["stop_rate"] == 0.5 and out["win_rate"] == 0.5
    assert out["mean_pct"] == round((round(exp_a, 2) + round(exp_b, 2)) / 2, 2)


# ─────────────────────────── 8. UI 헬퍼 (옛/새 JSON) ───────────────────────────

_OLD_JSON = {   # v3.9.2 형식 — freshness 없음, 라벨 전략, 0709에서 신호 멈춤
    "days_covered": 153,
    "methodology": {"date_range": ["2026-02-26", "2026-10-16"], "horizon_days": 10,
                    "fill_window_days": 3, "fee_pct_roundtrip": 0.22},
    "signal_top1": {"n_signals_total": 33, "n_filled": 27, "ev_net_pct": -0.46, "tp1_rate": 0.259},
    "capital_portfolio_top1": {"n_trades_filled": 11, "total_return_pct": -38.02},
    "daily_picks_log_top1": ([{"date": "20260709", "n_picked": 1, "codes": ["161890"]}]
                             + [{"date": f"2026{m:02d}{d:02d}", "n_picked": 0, "codes": []}
                                for m in (8, 9) for d in range(1, 29)]),
    "all_trades_top1": [{"date": "20260709", "code": "161890"}],
    "confidence": {"level": "LOW"},
}

_NEW_JSON = {
    "days_covered": 153,
    "methodology": {"date_range": ["2026-02-26", "2026-10-16"],
                    "contract_date_range": ["2026-07-13", "2026-10-16"],
                    "horizon_days": 10, "fill_window_days": 3, "fee_pct_roundtrip": 0.22},
    "signal_top1": {"n_signals_total": 38, "n_filled": 35, "ev_net_pct": -4.2, "tp1_rate": 0.029},
    "official_ssot": {"n_official": 38, "n_measured": 38, "mean_pct": -2.51, "median_pct": -6.87,
                      "win_rate": 0.342, "stop_rate": 0.5, "n_pending": 0},
    "capital_portfolio_top1": {"n_trades_filled": 12, "total_return_pct": -52.67},
    "freshness": {"latest_batch_date": "20261016", "last_signal_date": "20261013",
                  "contract_pick_days": 38, "contract_days_covered": 61,
                  "first_determinable_date": "20260713", "stale": False, "stale_reason": ""},
    "confidence": {"level": "LOW"},
}


def test_title_and_freshness_old_json_flagged_stale():
    import components.tab_stocks as ts
    fr = ts._scorecard_freshness(_OLD_JSON, batch_dates=None)
    assert fr["basis"] == "legacy" and fr["stale"] is True
    assert fr["last_signal"] == "20260709"
    title = ts._validation_scorecard_title(_OLD_JSON, fr)
    assert "옛 라벨" in title and "갱신 정지" in title and "153개 배치" in title
    assert "99일" not in title


def test_title_and_freshness_new_json():
    import components.tab_stocks as ts
    batches = ["20261014", "20261015", "20261016"]
    fr = ts._scorecard_freshness(_NEW_JSON, batches)
    assert fr["basis"] == "contract" and fr["stale"] is False
    title = ts._validation_scorecard_title(_NEW_JSON, fr)
    assert title.startswith("📊 공식 매수 성적표 — 07/13~10/16")
    assert "공식 매수 38일" in title and "갱신 정지" not in title
    # nightly 실패로 리포트가 3배치 밀림 → stale
    fr2 = ts._scorecard_freshness(_NEW_JSON, batches + ["20261019", "20261020", "20261021"])
    assert fr2["stale"] is True and "반영 못함" in fr2["reason"]
    # 리포트 자체가 stale이라고 말하면 그대로
    bad = dict(_NEW_JSON, freshness=dict(_NEW_JSON["freshness"], stale=True, stale_reason="X"))
    assert ts._scorecard_freshness(bad, batches)["reason"].startswith("X")


def test_headline_uses_official_ssot_then_falls_back():
    import components.tab_stocks as ts
    hm = ts._headline_metrics(_NEW_JSON)
    assert hm["basis"] == "official_ssot" and hm["ev"] == -2.51 and hm["tp1_rate"] == 0.029
    hm_old = ts._headline_metrics(_OLD_JSON)
    assert hm_old["basis"] == "signal_top1" and hm_old["ev"] == -0.46
    assert ts._headline_metrics({})["basis"] == "none"


def test_scorecard_renders_old_new_and_empty():
    import components.tab_stocks as ts
    for bt in (_OLD_JSON, _NEW_JSON, {}):
        ts._render_validation_scorecard(bt)
        ts._render_validation_scorecard(bt, fresh={"stale": True, "reason": "r", "basis": "legacy"})
    ts._render_member_summary(capital_top1=_NEW_JSON["capital_portfolio_top1"],
                              signal_top1=_NEW_JSON["signal_top1"], daily_top1={},
                              confidence={"level": "LOW"},
                              official_ssot=_NEW_JSON["official_ssot"], stale=True)


def test_hardcoded_99_day_title_removed():
    src = (ROOT / "components" / "tab_stocks.py").read_text(encoding="utf-8")
    assert "지난 99일 성적표" not in src


# ─────────────────── 9. downstream — 최신·확정 스냅샷 우선 ───────────────────

def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_official_buy_validation_prefers_newest_matured(tmp_path):
    from scripts.official_buy_validation import build_official_buy_validation
    _write(tmp_path / "recommend_20260805.csv", [
        {"종목코드": "000001", "종목명": "공식", "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1, "PRODUCTION_BUY": 1},
        {"종목코드": "000002", "종목명": "재구성만", "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1, "PRODUCTION_BUY": 0},
    ])
    _write(tmp_path / "recommend_20260806.csv", [
        {"종목코드": "000003", "종목명": "미체결", "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1, "PRODUCTION_BUY": 1},
    ])
    # D+1 밤 옛 형식 스냅샷: 봉 1개짜리 '미체결'(0.0)·1일 OPEN
    _write(tmp_path / "backtest_top1_trades_20260806.csv", [
        {"date": "20260805", "code": "000001", "outcome": "OPEN", "net_pct": 1.5},
    ])
    _write(tmp_path / "backtest_top1_trades_20260807.csv", [
        {"date": "20260805", "code": "000001", "outcome": "LOSS", "net_pct": -5.2, "matured": True},
        {"date": "20260806", "code": "000003", "outcome": "NOT_FILLED", "net_pct": 0.0, "matured": True},
    ])
    # 더 최신 파일이지만 측정중(PENDING) — 확정 결과가 이긴다
    _write(tmp_path / "backtest_top1_trades_20260808.csv", [
        {"date": "20260805", "code": "000001", "outcome": "PENDING", "net_pct": None, "matured": False},
    ])
    df, payload = build_official_buy_validation(tmp_path, tmp_path)
    s = payload["summary"]
    assert s["official_buy_signals"] == 2           # 기록된 PRODUCTION_BUY만 (000002 제외)
    assert s["official_buy_results"] == 1           # NOT_FILLED 0.0은 결과 아님
    assert s["official_buy_avg_net_pct"] == -5.2    # 첫 스냅샷 +1.5가 아니라 확정 -5.2
    assert s["official_buy_not_filled"] == 1
    v = df.set_index("code")["official_verdict"].to_dict()
    assert v["000001"] == "LOSS_OR_FLAT" and v["000003"] == "NOT_FILLED"


def test_tab_perf_prefers_newest_matured_snapshot():
    from components.tab_perf import _prefer_newest_matured
    t = pd.DataFrame([
        {"date": 20260805, "code": 1, "outcome": "OPEN", "net_pct": 1.5, "_file_rank": 0},
        {"date": 20260805, "code": 1, "outcome": "LOSS", "net_pct": -5.2, "matured": True, "_file_rank": 1},
        {"date": 20260805, "code": 1, "outcome": "PENDING", "net_pct": None, "matured": False, "_file_rank": 2},
        {"date": 20260806, "code": 2, "outcome": "PENDING", "net_pct": None, "matured": False, "_file_rank": 2},
    ])
    out = _prefer_newest_matured(t)
    got = {(r.date, r.code): r.outcome for r in out.itertuples()}
    assert got == {(20260805, 1): "LOSS", (20260806, 2): "PENDING"}
    assert "_file_rank" not in out.columns


def test_tab_perf_basis_card_copy_reflects_contract():
    src = (ROOT / "components" / "tab_perf.py").read_text(encoding="utf-8")
    assert "현재는 별도 누적 전이라 데이터 준비 중" not in src
    assert "PRODUCTION_BUY" in src


# ─────────────────────────── 10. 실데이터 ───────────────────────────

DATA = ROOT / "data"


def _real_csvs():
    files = sorted(p for p in DATA.glob("recommend_2*.csv")
                   if re.match(r"^recommend_2\d{7}\.csv$", p.name))
    if len(files) < 10:
        pytest.skip("실데이터 recommend CSV 없음")
    return files


def test_real_contract_pick_days_match_recorded_csvs():
    """CSV를 pandas + production_buy_mask로 독립 집계 → 선택기와 같아야 한다 (JSON 미사용)."""
    from services.recommendation_quality import production_buy_mask
    files = _real_csvs()
    days = bv.load_all_days(DATA)
    by_date = {d[0]: d for d in days}
    n_pick = n_cash = n_undet = 0
    window_pick = 0
    for f in files[1:]:      # 0번째 = hist base (백테스트 루프와 같은 범위)
        ymd = f.name[10:18]
        df = pd.read_csv(f, dtype={"종목코드": str}, encoding="utf-8-sig", low_memory=False)
        df = df[df["종목코드"].astype(str).str.strip() != ""]
        status, picks = bv.classify_contract_day(by_date[ymd][1])
        if "PRODUCTION_BUY" not in df.columns:
            assert status == bv.DAY_UNDETERMINABLE, ymd
            n_undet += 1
            continue
        exp = sorted(df.loc[production_buy_mask(df), "종목코드"].astype(str).str.zfill(6))
        assert sorted(p["code"] for p in picks) == exp, ymd
        if exp:
            n_pick += 1
            if "20260716" <= ymd <= "20261015":
                window_pick += 1
        else:
            n_cash += 1
    fr = bv.compute_freshness(days, trades=[])
    assert (fr["contract_pick_days"], fr["contract_cash_days"], fr["undeterminable_days"]) == \
        (n_pick, n_cash, n_undet)
    if n_pick:
        # 동결된 라벨 경로(거래 0건)는 반드시 stale로 잡힌다
        assert fr["stale"] is True and fr["contract_pick_days_unselected"] == n_pick
        # 기록된 공식픽 날마다 거래가 있으면 stale 아님
        trades = [{"date": d, "outcome": "LOSS"} for d in
                  [x[0] for x in days[1:] if bv.classify_contract_day(x[1])[0] == bv.DAY_PICK]]
        fr2 = bv.compute_freshness(days, trades=trades)
        assert fr2["contract_pick_days_unselected"] == 0 and fr2["stale"] is False
    # 0716~1015 창의 공식픽 일수는 CSV 집계와 일치 (2026-10 조사 시점 38일)
    assert window_pick == sum(
        1 for x in days[1:] if "20260716" <= x[0] <= "20261015"
        and bv.classify_contract_day(x[1])[0] == bv.DAY_PICK)


def test_real_contract_backtest_selects_every_recorded_pick():
    """contract 선택기는 기록된 공식픽 날마다 Top1을 뽑는다 (가격 없이 선택만 검증)."""
    _real_csvs()
    days = bv.load_all_days(DATA)
    res = bv.daily_top1_backtest(days, pd.DataFrame(), selector="contract")
    picked = {e["date"] for e in res["daily_picks"] if e["n_picked"] > 0}
    recorded = {x[0] for x in days[1:] if bv.classify_contract_day(x[1])[0] == bv.DAY_PICK}
    assert picked == recorded
    legacy = bv.daily_top1_backtest(days, pd.DataFrame(), selector="legacy")
    assert all(e["day_status"] == bv.DAY_LEGACY for e in legacy["daily_picks"])


# ─────────────────── 11. 리뷰 후속 — 달력 캐시 · shadow 성숙도 · 하류 카드 ───────────────────

def test_market_calendar_has_no_identity_cache_stale_reuse():
    """같은 프레임을 제자리에서 바꾸면 달력도 바뀌어야 한다 (예전 (id,len) 캐시 회귀)."""
    assert not hasattr(bv, "_CALENDAR_CACHE")
    df = pd.DataFrame({"종목코드": ["000009"] * 12,
                       "Date": pd.bdate_range("2026-09-02", periods=12),
                       "시가": 1.0, "고가": 1.0, "저가": 1.0, "종가": 1.0})
    days = [_day("20260831", [_row("000001", pb=0)]),
            _day("20260901", [_row("000001", pb=1, entry=100.0, stop=90.0, tp1=110.0)])]
    r1 = bv.daily_top1_backtest(days, df, selector="contract")
    assert r1["trades"][0]["bars_after_signal"] == 12
    # 같은 객체·같은 길이로 신호 '이전' 봉만 남기도록 제자리 변경
    df["Date"] = pd.bdate_range("2026-08-14", periods=12)
    assert pd.Timestamp(bv._market_calendar(df).max()) == df["Date"].max()
    r2 = bv.daily_top1_backtest(days, df, selector="contract")
    t = r2["trades"][0]
    # 예전 캐시: 낡은 달력(이후 12봉)으로 NODATA 확정 → 지금: 이후 봉 0 → 측정중
    assert t["outcome"] == "PENDING" and t["bars_after_signal"] == 0


def test_explicit_calendar_argument_is_used():
    d = _bdays("2026-08-03", 13)
    ohlc = _ohlc("000001", [(x, 10000, 10100, 9900, 10000) for x in d[1:]])
    days = [_day("20260731", [_row("000009")]), _day("20260803", [_row("000001", pb=1)])]
    # 종목 봉 12개(무터치) — 넘긴 달력이 '이후 봉 5개'라고 하면 그 달력을 따른다
    short_cal = pd.to_datetime(pd.Series(d[:6])).to_numpy(dtype="datetime64[ns]")
    r = bv.daily_top1_backtest(days, ohlc, selector="contract", calendar=short_cal)
    assert r["trades"][0]["bars_after_signal"] == 12      # max(종목 봉, 시장 봉)
    r_cal = bv._market_bars_after(short_cal, pd.Timestamp(d[0]))
    assert r_cal == 5
    # calendar 미지정 → ohlc_df 자체 달력(12봉) → 무터치 horizon 완료 → OPEN
    r0 = bv.daily_top1_backtest(days, ohlc, selector="contract")
    assert r0["trades"][0]["outcome"] == "OPEN"
    src = inspect.getsource(bv.build_report)
    assert "calendar = _market_calendar(ohlc_df)" in src
    assert src.count("calendar=calendar") >= 6


def _shadow_fixture():
    """오래된 신호(확정 LOSS) + 최근 신호(체결 후 무터치 5봉 = 미성숙)."""
    d = _bdays("2026-08-03", 25)
    bars = []
    for x in d[1:]:
        if x == d[2]:
            bars.append((x, 9800, 9850, 9400, 9500))       # 첫 신호 손절
        else:
            bars.append((x, 10000, 10100, 9900, 10000))
    ohlc = _ohlc("000001", bars)
    recent = d[19]                                           # 이후 봉 5개
    pick = {"code": "000001", "name": "x", "sector": "", "label": "🏆 최강",
            "score": 50.0, "entry": 10000.0, "tp1": 11000.0, "tp2": 0.0,
            "tp3": 0.0, "stop": 9500.0, "close": 10000.0}
    ymds = ["20260731", d[0].replace("-", ""), recent.replace("-", "")]
    days = [_day(y, [_row("000001")]) for y in ymds]
    return days, ohlc, pick


def test_shadow_runs_use_same_maturity_as_legacy_baseline(monkeypatch):
    days, ohlc, pick = _shadow_fixture()

    def picker(rows, *a, **k):
        return [dict(pick)]

    monkeypatch.setattr(bv, "pick_top3_codes", picker)
    monkeypatch.setattr(bv, "_pick_top3_exclude_struct_band", picker)
    monkeypatch.setattr(bv, "_pick_top3_with_risk_exclude", picker)
    base = bv.daily_top3_backtest(days, ohlc, selector="legacy")["trades"]
    assert [t["outcome"] for t in base] == ["LOSS", "PENDING"]

    sr = bv.compute_struct_risk_shadow(days, ohlc, base)
    # 같은 픽 → 같은 규칙이면 ΔEV 0 (예전엔 shadow만 부분 horizon OPEN을 확정으로 셌다)
    assert sr["delta_ev"] == 0 and sr["n_baseline_filled"] == sr["n_shadow_filled"] == 1
    assert sr["changed_pick_rate"] == 0
    pe = bv.compute_pre_entry_risk_shadow(days, ohlc, base)
    for rule in ("A_struct70_85", "B_red", "C_orange", "D_red_orange"):
        assert pe["rules"][rule]["delta_ev"] == 0, rule
        assert pe["rules"][rule]["n_shadow_filled"] == 1, rule

    trades, _ = bv._run_shadow_top3(days, ohlc, lambda s, v: False)
    assert [t["outcome"] for t in trades] == ["LOSS", "PENDING"]
    assert trades[1]["ret_pct"] is None


def test_shadow_pick_not_filled_only_after_fill_window():
    d = _bdays("2026-08-03", 5)
    gap = _ohlc("000001", [(x, 10500, 10600, 10400, 10500) for x in d[1:2]])
    pick = {"code": "000001", "entry": 10000.0, "tp1": 11000.0, "stop": 9500.0}
    sig = pd.Timestamp(d[0])
    early = bv._simulate_shadow_pick("20260803", 1, pick, sig, 1, gap, [])
    assert early["outcome"] == "PENDING"
    late = bv._simulate_shadow_pick("20260803", 1, pick, sig, 3, gap, [])
    assert late["outcome"] == "NOT_FILLED"


def test_entry_mode_shadow_partial_horizon_is_pending():
    d = _bdays("2026-08-03", 12)
    feat_days = [_day("20260803", [_row("000001", STRUCT_SCORE=95, VWAP_GAP=2)])]
    nf = dict(_trade("NOT_FILLED"))
    nf.update({"entry": 10000.0, "tp1": 11000.0})
    bars = [(x, 10100, 10200, 10000, 10100) for x in d[1:]]      # chase 체결, 무터치
    part = bv.compute_entry_mode_shadow([nf], _ohlc("000001", bars[:5]), feat_days)
    assert part["n_pending"] == 1 and part["extra_fills"] == 0
    full = bv.compute_entry_mode_shadow([nf], _ohlc("000001", bars[:10]), feat_days)
    assert full["n_pending"] == 0 and full["extra_fills"] == 1 and full["extra_opens"] == 1


def test_official_buy_cash_verdict_not_filled_is_not_pending(tmp_path):
    from scripts.official_buy_validation import build_official_buy_validation
    _write(tmp_path / "recommend_20260805.csv", [
        {"종목코드": "000001", "종목명": "보류", "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 0, "PRODUCTION_BUY": 0},
        {"종목코드": "000002", "종목명": "보류2", "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 0, "PRODUCTION_BUY": 0},
        {"종목코드": "000003", "종목명": "보류3", "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 0, "PRODUCTION_BUY": 0},
    ])
    _write(tmp_path / "backtest_top3_trades_20260812.csv", [
        {"date": "20260805", "code": "000001", "outcome": "NOT_FILLED", "net_pct": None, "matured": True},
        {"date": "20260805", "code": "000002", "outcome": "NODATA", "net_pct": None, "matured": False},
    ])
    df, payload = build_official_buy_validation(tmp_path, tmp_path)
    v = df.set_index("code")["cash_verdict"].to_dict()
    assert v == {"000001": "NOT_FILLED", "000002": "NO_DATA", "000003": "PENDING"}
    assert payload["summary"]["top_pick_holdout_results"] == 0
    assert payload["result_model"]["name"] == "limit_fill"


def test_tab_perf_top1_loader_drops_frozen_legacy_rows(tmp_path, monkeypatch):
    import components.tab_perf as tp
    legacy = [{"date": 20260301 + i, "code": i + 1, "fill_date": "2026-03-05",
               "outcome": "LOSS", "net_pct": -3.0, "exit_price": 100} for i in range(3)]
    for f in ("20261013", "20261014"):
        _write(tmp_path / f"backtest_top1_trades_{f}.csv", legacy)
    contract = [{"date": 20260805, "code": 9, "fill_date": "2026-08-06", "outcome": "LOSS",
                 "net_pct": -8.0, "exit_price": 92, "selection_source": "PRODUCTION_BUY",
                 "matured": True},
                {"date": 20260806, "code": 8, "fill_date": "", "outcome": "NOT_FILLED",
                 "net_pct": None, "exit_price": 0, "selection_source": "PRODUCTION_BUY",
                 "matured": True}]
    _write(tmp_path / "backtest_top1_trades_20261016.csv", contract)
    monkeypatch.setattr(tp, "DATA_DIR", str(tmp_path))
    t = tp._load_top1_trades()
    assert sorted(t["code"].tolist()) == [8, 9]
    assert t["net_pct_num"].dropna().tolist() == [-8.0]
    # 공식 매수 행이 하나도 없으면 예전처럼 전부 (옛 파일만 있는 배포 직후)
    os.remove(tmp_path / "backtest_top1_trades_20261016.csv")
    _write(tmp_path / "backtest_top1_trades_20261015.csv", legacy)
    assert len(tp._load_top1_trades()) == 3
    assert tp._keep_contract_rows(pd.DataFrame()).empty


def test_tab_perf_official_card_labels_model_and_shows_ssot(monkeypatch):
    import components.tab_perf as tp
    j = {"summary": {"official_buy_results": 35, "official_buy_win_rate": 14.29,
                     "official_buy_avg_net_pct": -4.2},
         "result_model": {"name": "limit_fill", "label": "제시가 지정가 모델",
                          "definition": "제시가 3일 체결창 · 10영업일"}}
    bt = {"official_ssot": {"n_measured": 38, "mean_pct": -2.51, "win_rate": 0.3421}}
    model, ssot = tp._official_card_basis_lines(j, bt)
    assert "지정가 모델" in model and "3일 체결창" in model
    assert "38건" in ssot and "-2.51%" in ssot and "34.2%" in ssot
    # 옛 official JSON(result_model 없음)·옛 backtest JSON(official_ssot 없음)
    model0, ssot0 = tp._official_card_basis_lines({"summary": {}}, {})
    assert "지정가 모델" in model0 and ssot0 is None
    assert tp._official_card_basis_lines(None, None)[1] is None
    assert tp._official_card_basis_lines(j, {"official_ssot": {"n_measured": 0}})[1] is None
    # 스텁 ui로 카드 렌더 (새/옛/빈 JSON 모두 예외 없이)
    for jj, bb in ((j, bt), ({"summary": {}}, {}), ({}, {})):
        monkeypatch.setattr(tp, "_load_official_buy_validation", lambda jj=jj: jj)
        monkeypatch.setattr(tp, "_load_backtest_validation", lambda bb=bb: bb)
        tp._render_official_buy_validation_card()
