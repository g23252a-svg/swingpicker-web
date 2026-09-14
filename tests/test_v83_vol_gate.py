# -*- coding: utf-8 -*-
"""[v83] 고변동 차단 게이트 + 선별 그림자 3종.

■ 근거 (docs/SELECTION_AUDIT_V83.md)
  생존편향 없는 전체 유니버스 패널에 후보풀 48,580행·114일을 붙여 잰 당일 횡단면
  변동성(ATR%)의 5일 실현수익 랭크IC: 전체 -0.252 (HAC t -5.54) · IS -0.213 (t -5.48)
  / OOS -0.274 (t -4.13) · 무손절도 같은 부호(-0.203, t -3.30) · 월별 7/7 음수.
  당일 5분위: OOS Q1 +0.11%(손절 25%) … Q4 -2.34%(67%) Q5 -2.52%(72%).
  현행 공식픽(알파 시대 측정가능 14건)은 변동성 분위 중위 0.79 — 14건 중 11건 손절,
  평균 -5.61%(풀 대비 -4.38%p). 같은 알파×손익비 랭킹에 상위 40% 차단만 얹으면
  퍼널 후보 18일에서 1등 평균 -6.15% → -1.00%, 손절 78% → 33%.

■ 규칙
  ALPHA_VOL_OK = V23_ATR_Pct 당일 분위 ≤ 0.60 (결측·표본<30은 통과 = 게이트 비활성).
  TOP_PICK 은 이 조건을 AND 로 추가한다. 랭킹 축은 바꾸지 않는다.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import alpha_engine as AE  # noqa: E402
from services import selection_shadow as SS  # noqa: E402

AE_SRC = (ROOT / "alpha_engine.py").read_text(encoding="utf-8")
PF_SRC = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")
DC_SRC = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")


def _frame(n_low: int = 30, n_high: int = 20, atr_col: bool = True):
    """게이트가 활성화되는 프레임 — 전부 알파·리스크·저점추세 통과, ATR%만 다르다.

    앞 n_low 행은 저변동(ATR 1~3%), 뒤 n_high 행은 고변동(ATR 10~20%).
    """
    rows = []
    for i in range(n_low + n_high):
        atr = 0.01 + 0.02 * (i / max(n_low - 1, 1)) if i < n_low else 0.10 + 0.10 * ((i - n_low) / max(n_high - 1, 1))
        r = {"종목코드": f"{i:06d}", "종목명": f"종목{i}",
             "ALPHA_VALIDATED": 1, "ALPHA_SCORE": 99.0 - i * 0.01,
             # 저점추세는 당일 분위라 값이 갈리면 하위 30%가 LT 게이트에 걸린다 — 전부 동률로
             "ENTRY_RISK_GATE_OK": True, "Low_Trend_PCT": 100.0,
             "ret_1d_%": 0.0, "MARKET_REGIME": "NEUTRAL",
             "TOP_PICK": 1, "종가": 10000.0, "손절가": 9200.0, "RR_NOW_TP1": 2.0}
        if atr_col:
            r["V23_ATR_Pct"] = atr
        rows.append(r)
    return pd.DataFrame(rows)


class TestVolGate:
    def test_constants(self):
        assert AE._VOL_GATE_PCTL == 0.60 and AE._VOL_GATE_COL == "V23_ATR_Pct"
        assert AE._VOL_GATE_MIN_SAMPLE == 30

    def test_top40_blocked_bottom60_pass(self):
        df = _frame(30, 20)
        out = AE.apply_alpha_entry_gate(df)
        ok = out["ALPHA_VOL_OK"].astype(int)
        assert ok.iloc[:30].sum() == 30, "저변동 60%는 전부 통과"
        assert ok.iloc[30:].sum() == 0, "고변동 40%는 전부 차단"
        assert out["TOP_PICK"].iloc[30:].sum() == 0
        assert out["TOP_PICK"].iloc[:30].sum() == 30
        assert (out["ALPHA_VOL_RULE"] == "atr_pctl<=0.60").all()
        assert out["VOL_GATE_PCTL"].iloc[-1] == pytest.approx(100.0)

    def test_blocked_row_explains_itself(self):
        out = AE.apply_alpha_entry_gate(_frame(30, 20))
        reason = str(out["ALPHA_ENTRY_BLOCK_REASON"].iloc[-1])
        assert "변동성 당일 상위" in reason and "v83" in reason
        assert str(out["ALPHA_ENTRY_BLOCK_REASON"].iloc[0]) == ""

    def test_missing_atr_value_passes(self):
        df = _frame(30, 20)
        df.loc[45, "V23_ATR_Pct"] = np.nan          # 고변동 구간의 한 행이 결측
        out = AE.apply_alpha_entry_gate(df)
        assert int(out.loc[45, "ALPHA_VOL_OK"]) == 1, "결측은 통과 — 데이터 이슈가 차단으로 번지지 않게"
        assert int(out.loc[46, "ALPHA_VOL_OK"]) == 0

    def test_no_atr_column_disables_gate(self):
        out = AE.apply_alpha_entry_gate(_frame(30, 20, atr_col=False))
        assert out["ALPHA_VOL_OK"].astype(int).sum() == 50
        assert (out["ALPHA_VOL_RULE"] == "inactive(sample)").all()
        assert out["TOP_PICK"].sum() == 50

    def test_small_sample_disables_gate(self):
        out = AE.apply_alpha_entry_gate(_frame(10, 10))     # 표본 20 < 30
        assert out["ALPHA_VOL_OK"].astype(int).sum() == 20
        assert (out["ALPHA_VOL_RULE"] == "inactive(sample)").all()

    def test_gate_is_relative_not_absolute(self):
        """같은 ATR 값이라도 그날 풀에서 상위 40%면 차단 — 절대 문턱이 아니다."""
        df = _frame(30, 20)
        df["V23_ATR_Pct"] = df["V23_ATR_Pct"] / 10.0      # 전부 10배 낮아도
        out = AE.apply_alpha_entry_gate(df)
        assert out["ALPHA_VOL_OK"].iloc[30:].sum() == 0

    def test_entry_ok_is_and_of_vol(self):
        """구현 배선 — entry_ok 식에 vol_ok가 AND로 들어가 있다."""
        i = AE_SRC.index("entry_ok = (risk_ok & ascore.notna()")
        assert "& vol_ok" in AE_SRC[i:i + 200]


@pytest.mark.skipif(not (ROOT / "data" / "recommend_20260911.csv").exists(), reason="실데이터 없음")
class TestRealBatch:
    def test_0911_official_pick_was_high_vol_and_is_now_blocked(self):
        """9/11 배치: 공식픽 대덕전자는 변동성 상위 22%였다 — 게이트 후 TOP_PICK에서 빠진다."""
        d = pd.read_csv(ROOT / "data" / "recommend_20260911.csv", dtype={"종목코드": str}, low_memory=False)
        out = AE.apply_alpha_entry_gate(d)
        row = out[out["종목명"].astype(str).str.contains("대덕전자")].iloc[0]
        assert float(row["VOL_GATE_PCTL"]) > 60.0
        assert int(row["ALPHA_VOL_OK"]) == 0 and int(row["TOP_PICK"]) == 0
        # 게이트는 풀의 약 40%를 자르되 TOP_PICK 을 전멸시키지 않는다
        blocked = int((out["ALPHA_VOL_OK"] == 0).sum())
        assert 0.30 * len(out) < blocked < 0.50 * len(out)
        assert int(out["TOP_PICK"].sum()) >= 5

    def test_0909_official_pick_low_vol_survives(self):
        d = pd.read_csv(ROOT / "data" / "recommend_20260909.csv", dtype={"종목코드": str}, low_memory=False)
        out = AE.apply_alpha_entry_gate(d)
        row = out[out["종목코드"].str.zfill(6) == "035720"].iloc[0]   # 카카오 · 분위 18%
        assert int(row["ALPHA_VOL_OK"]) == 1 and int(row["TOP_PICK"]) == 1


# ══ 선별 그림자 ═══════════════════════════════════════════════════════
def _shadow_df():
    """legacy·live·lowvol 이 서로 다른 종목을 고르도록 설계."""
    rows = [
        # A: 알파 최고·고변동 → 게이트 차단 (legacy 만 고른다)
        {"종목코드": "000001", "종목명": "A", "ALPHA_SCORE": 99, "ALPHA_ENTRY_THRESHOLD": 85, "RR_NOW_TP1": 3.0,
         "V23_ATR_Pct": 0.20, "ALPHA_VOL_OK": 0, "PRODUCTION_BUY": 0},
        # B: 알파 통과·중변동 → 현행 공식픽
        {"종목코드": "000002", "종목명": "B", "ALPHA_SCORE": 95, "ALPHA_ENTRY_THRESHOLD": 85, "RR_NOW_TP1": 2.5,
         "V23_ATR_Pct": 0.05, "ALPHA_VOL_OK": 1, "PRODUCTION_BUY": 1},
        # C: 알파 미달·최저변동 → lowvol 만 고른다
        {"종목코드": "000003", "종목명": "C", "ALPHA_SCORE": 40, "ALPHA_ENTRY_THRESHOLD": 85, "RR_NOW_TP1": 1.5,
         "V23_ATR_Pct": 0.01, "ALPHA_VOL_OK": 1, "PRODUCTION_BUY": 0},
        # D: 리스크가드 탈락 — 어디에도 못 든다
        {"종목코드": "000004", "종목명": "D", "ALPHA_SCORE": 99, "ALPHA_ENTRY_THRESHOLD": 85, "RR_NOW_TP1": 3.0,
         "V23_ATR_Pct": 0.005, "ALPHA_VOL_OK": 1, "PRODUCTION_BUY": 0, "ENTRY_RISK_GATE_OK": 0},
    ]
    df = pd.DataFrame(rows)
    df["ENTRY_RISK_GATE_OK"] = df["ENTRY_RISK_GATE_OK"].fillna(1)
    df["NEW_ENTRY_BLOCKED"] = 0; df["ALPHA_SURGE_OK"] = 1; df["ALPHA_LT_OK"] = 1
    df["BUY_NOW_PASS"] = 1; df["POC_GAP"] = 5.0; df["ROUTE"] = "WAIT"
    return df


class TestShadowPicks:
    def test_three_variants_pick_different_rows(self):
        p = {x["variant"]: x["종목명"] for x in SS.pick_today(_shadow_df(), "20260914")}
        assert p == {"legacy": "A", "live": "B", "lowvol": "C"}

    def test_risk_gate_failure_excluded_everywhere(self):
        df = _shadow_df(); df.loc[df["종목명"] == "D", "V23_ATR_Pct"] = 0.001
        p = {x["variant"]: x["종목명"] for x in SS.pick_today(df, "20260914")}
        assert "D" not in p.values()

    def test_no_candidates_means_cash(self):
        df = _shadow_df(); df["ENTRY_RISK_GATE_OK"] = 0; df["PRODUCTION_BUY"] = 0
        assert SS.pick_today(df, "20260914") == []

    def test_append_is_idempotent_per_day(self, tmp_path):
        rows = SS.pick_today(_shadow_df(), "20260914")
        assert SS.append_log(str(tmp_path), rows) is True
        assert SS.append_log(str(tmp_path), rows) is False
        assert SS.append_log(str(tmp_path), SS.pick_today(_shadow_df(), "20260915")) is True
        log = SS.load_log(str(tmp_path))
        assert len(log) == 6 and log["ymd"].nunique() == 2

    def test_build_without_prices_reports_zero_measured(self, tmp_path):
        SS.append_log(str(tmp_path), SS.pick_today(_shadow_df(), "20260914"))
        s = SS.build(str(tmp_path))
        assert s["days_total"] == 1
        assert s["variants"]["live"]["days"] == 0
        assert "판정 전" in s["verdict"]
        assert SS.line({**s, "today": SS.pick_today(_shadow_df(), "20260914")}).startswith("선별 그림자 3종")

    def test_build_measures_realized_and_pairs(self, tmp_path, monkeypatch):
        """가격이 있으면 SSOT 실현수익으로 변형별·페어드 요약을 낸다."""
        SS.append_log(str(tmp_path), SS.pick_today(_shadow_df(), "20260901"))
        dates = pd.bdate_range("20260825", periods=15)
        frames = []
        for code, path in (("000001", -0.10), ("000002", 0.02), ("000003", 0.01)):
            px = 10000 * (1 + path) ** np.arange(len(dates))
            frames.append(pd.DataFrame({"Date": dates, "종목코드": code, "시가": px, "고가": px * 1.01,
                                        "저가": px * 0.99, "종가": px}))
        monkeypatch.setattr(SS, "_load_universe_ohlcv", lambda _d: pd.concat(frames, ignore_index=True))
        monkeypatch.setattr(SS, "LIVE_FROM", "20260901")
        s = SS.build(str(tmp_path))
        v = s["variants"]
        assert v["legacy"]["days"] == 1 and v["legacy"]["avg_ret_pct"] == pytest.approx(-8.0)
        assert v["live"]["avg_ret_pct"] > 0 and v["lowvol"]["avg_ret_pct"] > 0
        assert s["paired"]["live-legacy"]["diff_pct"] > 0
        assert "현행−구엔진" in SS.line(s)


class TestWiring:
    def test_pipeline_runs_shadow_after_winner_profile(self):
        i_wp = PF_SRC.index("_WP.run_batch(OUT_DIR, trade_ymd)")
        i_ss = PF_SRC.index("_SS.run_batch(df_out, OUT_DIR, trade_ymd)")
        assert i_ss > i_wp

    def test_screen_shows_shadow_line(self):
        assert '"shadow_line"' in DC_SRC and "_ss.line(_ss.load(data_dir))" in DC_SRC
        assert 'payload.get("shadow_line")' in DC_SRC
