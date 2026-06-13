# -*- coding: utf-8 -*-
"""tests/test_backtest_data_integrity_v242.py — v24.2 소급 백테스트 회귀 테스트.

고정하는 계약:
  1. build_asof_panel: as-of 감사가 그날의 캐시로 수행되고, 캐시 없으면 무해 SKIP
  2. build_trades: forward 수익률 = combo_optimizer 관례 (추천매수가 → D+h snapshot 종가)
  3. _flag01: 0/1·True/False 문자열 혼재 컬럼 정규화
  4. summarize: CLEAN/DI_BAD/SURGE 분리 집계 (demote 판단은 DI_BAD 기준)
  5. combo_optimizer: 패널 없으면 DQ=0·128조합 → 기존과 완전 동일 (하위호환)
  6. combo_optimizer: 패널 있으면 DQ merge + dq_exclude=1이 플래그 행 제외
  7. 패널 CSV 재로딩 시 rec_ymd dtype(str) 보존 — 매칭 0건 회귀 방지
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtest_data_integrity import (  # noqa: E402
    build_asof_panel,
    build_trades,
    summarize,
    PANEL_FILENAME,
)
import combo_optimizer  # noqa: E402


# ── 합성 데이터 픽스처 ───────────────────────────────────────────
def _mk_ohlcv_long(code: str, closes, end="2026-01-02") -> pd.DataFrame:
    """추천일(end)에 마지막 봉이 오도록 앵커링 — as-of 컷에 잘리지 않게."""
    c = pd.Series(closes, dtype="float64")
    idx = pd.bdate_range(end=end, periods=len(c))
    return pd.DataFrame({
        "종목코드": code,
        "시가": (c * 0.99).values, "고가": (c * 1.02).values,
        "저가": (c * 0.97).values, "종가": c.values,
        "거래량": 10000.0, "등락률": 0.0,
    }, index=pd.Index(idx, name="Date"))


@pytest.fixture
def data_dir(tmp_path):
    """추천 1일 + 캐시 + snapshot 2일짜리 미니 data/ 디렉터리.

    종목: 정상주(000001) / 왜곡주(000002, 고가<종가) / 폭등주(000003, ret_10d 1500%)
    """
    d = tmp_path / "data"
    d.mkdir()
    rec_ymd = "20260102"

    # 추천 CSV — combo 그리드(ATTACK, S/T/AI 충족) 통과하도록 점수 부여
    rec = pd.DataFrame([
        {"종목코드": "000001", "종목명": "정상주", "ret_10d_%": 10.0,
         "추천매수가": 1000.0, "종가": 1000.0, "TOP_PICK": 1,
         "BUY_NOW_ELIGIBLE": "True", "MOMENTUM_LANE": 0,
         "STRUCT_SCORE": 90, "TIMING_SCORE": 80, "AI_SCORE": 70,
         "ROUTE": "ATTACK", "DISPLAY_SCORE": 90},
        {"종목코드": "000002", "종목명": "왜곡주", "ret_10d_%": 5.0,
         "추천매수가": 2000.0, "종가": 2000.0, "TOP_PICK": 0,
         "BUY_NOW_ELIGIBLE": "False", "MOMENTUM_LANE": 0,
         "STRUCT_SCORE": 90, "TIMING_SCORE": 80, "AI_SCORE": 70,
         "ROUTE": "ATTACK", "DISPLAY_SCORE": 85},
        {"종목코드": "000003", "종목명": "폭등주", "ret_10d_%": 1500.0,
         "추천매수가": 500.0, "종가": 500.0, "TOP_PICK": 0,
         "BUY_NOW_ELIGIBLE": 0, "MOMENTUM_LANE": 1,
         "STRUCT_SCORE": 90, "TIMING_SCORE": 80, "AI_SCORE": 70,
         "ROUTE": "ATTACK", "DISPLAY_SCORE": 80},
    ])
    rec.to_csv(d / f"recommend_{rec_ymd}.csv", index=False, encoding="utf-8-sig")

    # OHLCV 캐시 (그날 파이프라인이 본 데이터)
    clean = _mk_ohlcv_long("000001", np.linspace(900, 1000, 30))
    bad = _mk_ohlcv_long("000002", np.linspace(1900, 2000, 30))
    bad.iloc[-1, bad.columns.get_loc("고가")] = 100.0  # 고가 < 종가 위반
    surge = _mk_ohlcv_long("000003", np.linspace(450, 500, 30))
    pd.concat([clean, bad, surge]).to_parquet(d / f"ohlcv_cache_{rec_ymd}.parquet")

    # snapshot — D, D+1 (h=1 매칭)
    pd.DataFrame({"종목코드": ["000001", "000002", "000003"],
                  "종목명": ["정상주", "왜곡주", "폭등주"],
                  "종가": [1000.0, 2000.0, 500.0]}).to_csv(
        d / f"price_snapshot_{rec_ymd}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"종목코드": ["000001", "000002", "000003"],
                  "종목명": ["정상주", "왜곡주", "폭등주"],
                  "종가": [1100.0, 1800.0, 550.0]}).to_csv(
        d / "price_snapshot_20260103.csv", index=False, encoding="utf-8-sig")
    return str(d)


# ── 1~2. 패널 + forward 수익률 ──────────────────────────────────
class TestHarness:
    def test_asof_panel_detection(self, data_dir):
        panel = build_asof_panel(data_dir)
        p = panel.set_index("종목명")
        assert bool(p.loc["정상주", "DI_OK"]) and not bool(p.loc["정상주", "SURGE_FLAG"])
        assert not bool(p.loc["왜곡주", "DI_OK"])
        assert "V1" in str(p.loc["왜곡주", "DI_REASON"])
        assert bool(p.loc["폭등주", "SURGE_FLAG"]) and bool(p.loc["폭등주", "FLAGGED"])

    def test_panel_reload_keeps_str_dtypes(self, data_dir):
        """캐시 재로딩 시 rec_ymd가 int로 굳어 매칭 0건 되던 회귀 방지."""
        build_asof_panel(data_dir)                      # 1차: 생성+저장
        panel2 = build_asof_panel(data_dir)             # 2차: CSV 재사용
        assert panel2["rec_ymd"].dtype == object
        trades = build_trades(data_dir, panel2, [1])
        assert len(trades) == 3, "재로딩 패널로 forward 매칭 실패"

    def test_forward_return_convention(self, data_dir):
        panel = build_asof_panel(data_dir)
        trades = build_trades(data_dir, panel, [1]).set_index("종목명")
        # 정상주: 1000 → 1100 = +10%
        assert abs(trades.loc["정상주", "ret_h1"] - 10.0) < 1e-6
        # 왜곡주: 2000 → 1800 = -10%
        assert abs(trades.loc["왜곡주", "ret_h1"] - (-10.0)) < 1e-6

    def test_flag01_mixed_types(self, data_dir):
        """BUY_NOW_ELIGIBLE 'True'/'False' 문자열 정규화."""
        panel = build_asof_panel(data_dir)
        trades = build_trades(data_dir, panel, [1]).set_index("종목명")
        assert int(trades.loc["정상주", "BUY_NOW_ELIGIBLE"]) == 1
        assert int(trades.loc["왜곡주", "BUY_NOW_ELIGIBLE"]) == 0
        assert int(trades.loc["폭등주", "MOMENTUM_LANE"]) == 1

    def test_summarize_groups(self, data_dir):
        panel = build_asof_panel(data_dir)
        trades = build_trades(data_dir, panel, [1])
        res = summarize(trades, [1])
        cov = res["coverage"]
        assert cov["n_di_bad"] == 1 and cov["n_surge"] == 1
        a = res["subsets"]["ALL"]
        assert a["CLEAN"]["n"] == 1 and a["DI_BAD"]["n"] == 1 and a["SURGE"]["n"] == 1
        # DI_BAD(왜곡주) h1 = -10%
        assert abs(a["DI_BAD"]["h1"]["mean"] - (-10.0)) < 1e-6

    def test_missing_cache_is_harmless_skip(self, tmp_path):
        d = tmp_path / "data"; d.mkdir()
        pd.DataFrame([{"종목코드": "000001", "종목명": "고아주", "ret_10d_%": 1.0,
                       "종가": 100.0}]).to_csv(
            d / "recommend_20260105.csv", index=False, encoding="utf-8-sig")
        panel = build_asof_panel(str(d))
        assert len(panel) == 1
        assert bool(panel.iloc[0]["DI_OK"]) and not bool(panel.iloc[0]["AUDITED"])
        assert str(panel.iloc[0]["DI_REASON"]).startswith("SKIP")


# ── 5~6. combo_optimizer DQ 통합 ────────────────────────────────
class TestComboDQ:
    def test_all_combos_backward_compat(self):
        """기본(include_dq=False): 128조합 · 전부 dq=0 — 기존과 동일."""
        combos = combo_optimizer._all_combos()
        assert len(combos) == 128
        assert all(len(c) == 5 and c[4] == 0 for c in combos)
        combos_dq = combo_optimizer._all_combos(include_dq=True)
        assert len(combos_dq) == 256

    def test_evaluate_combo_dq_exclude(self):
        df = pd.DataFrame({
            "ret": [10.0, -50.0, 5.0], "win": [1, 0, 1],
            "S": [90, 90, 90], "T": [80, 80, 80], "AI": [70, 70, 70],
            "ROUTE": ["ATTACK"] * 3, "SCORE": [90] * 3,
            "trade_date": ["20260102"] * 3, "code": ["A", "B", "C"],
            "DQ": [0, 1, 0],
        })
        off = combo_optimizer._evaluate_combo(df, 60, 50, 40, ["ATTACK"], min_samples=1)
        on = combo_optimizer._evaluate_combo(df, 60, 50, 40, ["ATTACK"], min_samples=1,
                                             dq_exclude=1)
        assert off["n"] == 3 and on["n"] == 2
        assert on["avg_ret"] > off["avg_ret"]  # 플래그(-50%) 제외 효과

    def test_load_trade_rows_dq_merge(self, data_dir):
        build_asof_panel(data_dir)  # 패널 생성 → merge 대상
        df = combo_optimizer._load_trade_rows(data_dir, horizon=1)
        assert "DQ" in df.columns and "code" in df.columns
        m = df.set_index("code")["DQ"]
        assert int(m["000001"]) == 0
        assert int(m["000002"]) == 1  # 무결성 위반
        assert int(m["000003"]) == 1  # 폭등 플래그

    def test_load_trade_rows_without_panel(self, data_dir):
        """패널 없으면 DQ=0 전부 — dq_exclude 조합이 no-op (하위호환)."""
        panel_path = os.path.join(data_dir, PANEL_FILENAME)
        if os.path.exists(panel_path):
            os.remove(panel_path)
        df = combo_optimizer._load_trade_rows(data_dir, horizon=1)
        assert "DQ" in df.columns and int(df["DQ"].sum()) == 0
        a = combo_optimizer._evaluate_combo(df, 60, 50, 40, ["ATTACK"], min_samples=1)
        b = combo_optimizer._evaluate_combo(df, 60, 50, 40, ["ATTACK"], min_samples=1,
                                            dq_exclude=1)
        assert a == b
