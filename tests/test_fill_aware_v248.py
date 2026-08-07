# -*- coding: utf-8 -*-
"""
test_fill_aware_v248.py — v24.8 Fill-Aware Validation 테스트.

보증
  1. [통합] make_rank_validation_report: 유령체결(entry 미도달) 종목은
     rank_validation 집계(N)와 per_trade_log에서 제외되고 N_UNFILLED로 분리 카운트.
  2. [통합] 체결 종목은 fill_status='filled' + fill_day 기록, 성과는 기존 로직 그대로.
  3. [통합] FillConfig 끄면(enabled=False) 기존 동작(전원 포함, unchecked).
  4. [단위] backfill fill_check_row: 창 내 저가 도달/미도달/데이터부족(unknown) 판정.
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

# collector의 무거운 의존성(torch) 스텁 — 테스트 환경 한정
#
# [v55.4] 이 스텁이 세션 전역으로 새어나가 다른 테스트를 망가뜨렸다.
#   FEATURE_COLS = [] 인 가짜 ml_engine이 sys.modules에 남으면,
#   뒤에 도는 test_policy_consistency의 계약 테스트가
#   `from ml_engine import FEATURE_COLS`에서 ImportError를 못 보므로
#   스킵하지 않고 빈 목록과 비교해 실패한다
#   (tests/ + 루트 테스트를 한 프로세스로 돌릴 때만 재현 — 2026-08 실측).
#   → ① 스텁임을 표시해 하류가 진짜로 착각하지 않게 하고
#     ② 모듈 테스트가 끝나면 원상복구한다(teardown_module).
_STUB_INSTALLED = False
try:  # pragma: no cover
    import ml_engine  # noqa
except Exception:  # pragma: no cover
    _m = types.ModuleType("ml_engine")
    _m.FEATURE_COLS = []
    _m.__is_test_stub__ = True          # feature_contract.is_stub_ml_engine가 본다
    _m.__getattr__ = lambda n: (lambda *a, **k: None)
    sys.modules["ml_engine"] = _m
    _STUB_INSTALLED = True


def teardown_module(module):  # pragma: no cover - 테스트 격리용
    """심어둔 스텁을 제거해 세션 전역 오염을 막는다."""
    if _STUB_INSTALLED and getattr(sys.modules.get("ml_engine"), "__is_test_stub__", False):
        del sys.modules["ml_engine"]

import collector  # noqa: E402
from scripts.backfill_fill_status import fill_check_row  # noqa: E402


# ── 합성 데이터 빌더 ─────────────────────────────────────────────
DAYS = ["20260601", "20260602", "20260603", "20260604", "20260605"]

# 종목 A(000001): entry 10000 — D+1 저가 9900로 체결, SL/TP 미접촉 → hold
# 종목 B(000002): entry 20000 — 계속 상승, 저가가 entry에 절대 미도달 → 유령
PRICES = {
    # ymd: {code: (open, high, low, close)}
    "20260601": {"000001": (10100, 10300, 10000, 10200), "000002": (20500, 21000, 20300, 20800)},
    "20260602": {"000001": (10100, 10200, 9900, 10100),  "000002": (21000, 22000, 20800, 21800)},
    "20260603": {"000001": (10150, 10500, 10050, 10400), "000002": (22000, 23500, 21900, 23200)},
    "20260604": {"000001": (10450, 10700, 10300, 10600), "000002": (23500, 25000, 23300, 24800)},
    "20260605": {"000001": (10600, 10800, 10500, 10700), "000002": (25000, 26500, 24900, 26000)},
}


def _build_out_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for ymd in DAYS:
        rows = []
        for code, (o, h, l, c) in PRICES[ymd].items():
            rows.append({"종목코드": code, "종목명": f"T{code}", "시장": "KOSPI",
                         "종가": c, "시가": o, "저가": l, "고가": h})
        pd.DataFrame(rows).to_csv(d / f"price_snapshot_{ymd}.csv", index=False, encoding="utf-8-sig")
    rec = pd.DataFrame([
        {"종목코드": "000001", "종목명": "T000001", "종가": 10200,
         "추천매수가": 10000, "손절가": 9000, "추천매도가1": 12000, "SCOREX": 90},
        {"종목코드": "000002", "종목명": "T000002", "종가": 20800,
         "추천매수가": 20000, "손절가": 18000, "추천매도가1": 30000, "SCOREX": 80},
    ])
    rec.to_csv(d / "recommend_20260601.csv", index=False, encoding="utf-8-sig")
    return str(d)


def _run(out_dir):
    collector.make_rank_validation_report(
        out_dir=out_dir, asof_ymd="20260605", lookback_trading_days=10,
        horizons=[3], topks=[3], methods=["SCOREX"],
    )
    rv = pd.read_csv(os.path.join(out_dir, "rank_validation_latest.csv"))
    pt_path = os.path.join(out_dir, "per_trade_log.csv")
    pt = pd.read_csv(pt_path) if os.path.exists(pt_path) else pd.DataFrame()
    return rv, pt


class _CfgProxy:
    """DEFAULT_CONFIG 위임 + fill만 오버라이드."""
    def __init__(self, real, fill_ns):
        self._real, self._fill = real, fill_ns
    def __getattr__(self, n):
        if n == "fill":
            return self._fill
        return getattr(self._real, n)


# ── 통합: 유령체결 제외 ──────────────────────────────────────────
def test_phantom_excluded_from_aggregates_and_log(tmp_path):
    out = _build_out_dir(tmp_path)
    rv, pt = _run(out)
    row = rv[(rv["METHOD"] == "SCOREX") & (rv["H(영업일)"] == 3)].iloc[0]
    assert row["N"] == 1, "체결분(A)만 집계돼야 함"
    assert row["N_UNFILLED"] == 1, "유령(B) 분리 카운트"
    assert abs(row["FILL_RATE_%"] - 50.0) < 0.1
    # 성과는 A 기준: entry 10000 → D+3 종가 10600 = +6%
    assert abs(row["AVG_RET_%"] - 6.0) < 0.5
    assert row["WIN_RATE_%"] == 100.0
    # per_trade: A만, filled/fill_day=1 (D+1 저가 9900 ≤ 10000)
    assert len(pt) == 1 and pt.iloc[0]["code"] in (1, "000001")
    assert pt.iloc[0]["fill_status"] == "filled"
    assert int(pt.iloc[0]["fill_day"]) == 1
    assert pt.iloc[0]["exit_type"] == "hold_close"


def test_disabled_restores_legacy_behavior(tmp_path, monkeypatch):
    out = _build_out_dir(tmp_path)
    fill_off = types.SimpleNamespace(fill_aware_enabled=False, fill_window_days=3)
    monkeypatch.setattr(collector, "DEFAULT_CONFIG",
                        _CfgProxy(collector.DEFAULT_CONFIG, fill_off))
    rv, pt = _run(out)
    row = rv[(rv["METHOD"] == "SCOREX") & (rv["H(영업일)"] == 3)].iloc[0]
    assert row["N"] == 2, "끄면 유령 포함(기존 동작)"
    assert row["N_UNFILLED"] == 0
    assert len(pt) == 2
    assert set(pt["fill_status"]) == {"unchecked"}


def test_fill_gate_blocks_sl_tp_before_fill(tmp_path):
    """유령(B)은 TP 30000 미도달이지만, 도달했더라도 미체결이면 tp_hit 금지 —
    여기선 간접 검증: B가 로그에 없고 stop/tp율이 A(hold)만 반영돼 0%."""
    out = _build_out_dir(tmp_path)
    rv, _ = _run(out)
    row = rv[(rv["METHOD"] == "SCOREX") & (rv["H(영업일)"] == 3)].iloc[0]
    assert row["STOP_HIT_RATE_%"] == 0.0 and row["TP_HIT_RATE_%"] == 0.0


# ── 단위: backfill 판정 ──────────────────────────────────────────
def _lut(code="000001", rows=None):
    return {code: np.array(rows, dtype=float)}


def test_backfill_filled_first_touch_day():
    lut = _lut(rows=[[20260601, 10000], [20260602, 9950], [20260603, 9800]])
    s, d = fill_check_row(lut, "000001", 20260601, entry=9900, horizon=5)
    assert s == "filled" and d == 2  # D+2 저가 9800 ≤ 9900 (D+1 9950은 미도달)


def test_backfill_unfilled_when_window_complete():
    lut = _lut(rows=[[20260601, 10000], [20260602, 10100], [20260603, 10200], [20260604, 10300]])
    s, d = fill_check_row(lut, "000001", 20260601, entry=9900, horizon=5)
    assert s == "unfilled" and d == 0  # 3일 창 완결 + 미도달


def test_backfill_unknown_incomplete_window_or_missing():
    lut = _lut(rows=[[20260601, 10000], [20260602, 10100]])  # 창(3) 미완결
    s, _ = fill_check_row(lut, "000001", 20260601, entry=9000, horizon=5)
    assert s == "unknown"
    s, _ = fill_check_row(lut, "999999", 20260601, entry=10000, horizon=5)  # 코드 없음
    assert s == "unknown"
    s, _ = fill_check_row(lut, "000001", 20260601, entry=np.nan, horizon=5)  # entry 무효
    assert s == "unknown"
    s, _ = fill_check_row(lut, "000001", 20260601, entry=10000, horizon=0)  # h0
    assert s == "unknown"


def test_backfill_window_capped_by_horizon():
    # h=1이면 D+1만 검사 — D+2에 도달해도 unfilled여야 함 (창 완결 기준은 min(3,h)=1)
    lut = _lut(rows=[[20260601, 10000], [20260602, 10100], [20260603, 9500]])
    s, d = fill_check_row(lut, "000001", 20260601, entry=9900, horizon=1)
    assert s == "unfilled" and d == 0
