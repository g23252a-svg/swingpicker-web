# -*- coding: utf-8 -*-
"""[v86] 그림자 선등록 리뷰 회귀 — 판정 창 고정·동결, 차단일 매크로 반사실, v85 측정 보존."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import selection_shadow as SS  # noqa: E402
from tests.test_v86_shadow import _df, _days, _picks, _prices  # noqa: E402

DATA = ROOT / "data"


# ══ 판정 창 — 실현 여부가 아니라 '기록된 날'로 고정 ═══════════════════════
class TestWindowByLoggedDays:
    def _case(self, pending_live):
        """45일 기록. 36~39일째 live 미실현(pending) · 40~44일째는 양쪽 모두 일찍 손절."""
        ix = _days(45)
        rng = np.random.default_rng(7)
        live = pd.Series(rng.normal(0, 1, 45), index=ix)
        cand = live + 2.0 + rng.normal(0, 0.3, 45)
        live.iloc[40:] = -8.0; cand.iloc[40:] = -8.0
        if pending_live:
            live.iloc[36:40] = np.nan
        return cand, live, ix

    def test_pending_earlier_day_blocks_decision(self):
        cand, live, ix = self._case(pending_live=True)
        v = SS.prereg_verdict(cand, live, cand_days=ix, live_days=ix)
        assert not v["decided"] and v["status"] == "측정 중 36/40"
        assert v["window"] == [ix[0], ix[39]], "창은 뒷날의 이른 손절로 밀리지 않는다"

    def test_window_stable_once_settled(self):
        cand, live, ix = self._case(pending_live=False)
        v = SS.prereg_verdict(cand, live, cand_days=ix, live_days=ix)
        assert v["decided"] and v["window"] == [ix[0], ix[39]]
        assert v["status"] == "승격 후보"

    def test_window_only_days_logged_in_both_arms(self):
        ix = _days(45)
        cand = pd.Series(1.0, index=ix); live = pd.Series(0.0, index=ix)
        v = SS.prereg_verdict(cand, live, cand_days=ix[2:], live_days=ix)
        assert v["window"][0] == ix[2] and v["logged"] == 40

    def test_legacy_call_without_days_unchanged(self):
        ix = _days(39)
        v = SS.prereg_verdict(pd.Series(1.0, index=ix), pd.Series(0.0, index=ix))
        assert v["status"] == "측정 중 39/40"


class TestBuildWindowAndFreeze:
    def _write(self, tmp_path, monkeypatch, n_days, px):
        monkeypatch.setattr(SS, "_load_universe_ohlcv", lambda _d: px)
        monkeypatch.setattr(SS, "LIVE_FROM", "20260901")
        monkeypatch.setattr(SS, "REGISTERED_FROM", "20260902")
        days = [d.strftime("%Y%m%d") for d in pd.bdate_range("20260902", periods=n_days)]
        for y in days:
            SS.append_log(str(tmp_path), [
                {"ymd": y, "variant": "live", "종목코드": "000001", "종목명": "", "cash": 0.0, "blocked": 0.0},
                {"ymd": y, "variant": "lowvol", "종목코드": "000002", "종목명": "", "cash": 0.0, "blocked": 0.0}])
        return days

    def test_unrealized_tail_keeps_measuring(self, tmp_path, monkeypatch):
        # 시세 45봉 → 기록 44일 중 마지막 ~6일은 t+5 미도래로 미실현
        px = _prices([("000001", 0.002), ("000002", 0.01)], start="20260901", n=45)
        self._write(tmp_path, monkeypatch, 44, px)
        s = SS.build(str(tmp_path))
        v = s["verdicts"]["lowvol"]
        assert not v["decided"] and v["logged"] == 40 and v["n"] < 40
        assert s["statuses"]["lowvol"].startswith("측정 중")

    def test_decided_verdict_frozen_across_batches(self, tmp_path, monkeypatch):
        px = _prices([("000001", 0.002), ("000002", 0.01)], start="20260901", n=60)
        self._write(tmp_path, monkeypatch, 44, px)
        s1 = SS.build(str(tmp_path))
        assert s1["verdicts"]["lowvol"]["decided"]
        SS.save(str(tmp_path), s1)
        # 시세 캐시가 보정돼 창 안 수익이 뒤집혀도 판정은 그대로
        px2 = _prices([("000001", 0.01), ("000002", -0.01)], start="20260901", n=60)
        monkeypatch.setattr(SS, "_load_universe_ohlcv", lambda _d: px2)
        s2 = SS.build(str(tmp_path))
        assert s2["verdicts"]["lowvol"]["frozen"] is True
        assert s2["statuses"]["lowvol"] == s1["statuses"]["lowvol"]

    def test_freeze_ignored_for_other_prereg_version(self, tmp_path, monkeypatch):
        prev = {"prereg_version": "v0-old", "verdicts": {"lowvol": {"decided": True, "status": "기각"}}}
        assert SS._frozen_verdict(prev, "lowvol") is None
        prev["prereg_version"] = SS.PREREG_VERSION
        assert SS._frozen_verdict(prev, "lowvol")["frozen"] is True
        prev["verdicts"]["lowvol"] = {"decided": False, "status": "측정 중 3/40"}
        assert SS._frozen_verdict(prev, "lowvol") is None


# ══ 차단일 반사실 — 매크로 경보도 무시해 같은 날을 비교 ═══════════════════
class TestMacroCounterfactual:
    def test_critical_blocked_day_logs_all_variants(self):
        df = _df(n_top=8, blocked=True); df["MACRO_RISK"] = "CRITICAL"
        p = _picks(df)
        assert {"legacy", "live", "lowvol", "comp_lowvol", "no_alpha", "thin_skip"} <= set(p)
        assert p["live"]["종목명"] == "A"

    def test_normal_day_macro_guard_kept(self):
        df = _df(); df["MACRO_RISK"] = "WARNING"
        p = _picks(df)
        assert "no_alpha" not in p, "정상일 매크로 경보는 그대로 존중한다"


# ══ v85 측정 보존 — 레인 이탈 종목은 이력 폴백으로 잰다 ════════════════════
class TestV85MeasurementPreserved:
    def test_missing_code_measured_via_history_and_cash_not_looked_up(self, tmp_path, monkeypatch):
        px = _prices([("000001", 0.01)], start="20260901", n=30)
        hx = _prices([("000777", 0.005)], start="20260901", n=30)
        monkeypatch.setattr(SS, "_load_universe_ohlcv", lambda _d: px)
        asked = {}

        def _hist(_d, codes, since=None):
            asked["codes"] = set(codes)
            return {c: g.reset_index(drop=True) for c, g in hx.groupby("종목코드")}
        monkeypatch.setattr(SS, "_load_code_history", _hist)
        monkeypatch.setattr(SS, "LIVE_FROM", "20260901")
        SS.append_log(str(tmp_path), [
            {"ymd": "20260903", "variant": "live", "종목코드": "000001", "종목명": "", "cash": 0.0, "blocked": 0.0},
            {"ymd": "20260903", "variant": "ext_lowtv", "종목코드": "000777", "종목명": "", "cash": 0.0,
             "blocked": 0.0},
            {"ymd": "20260903", "variant": "thin_skip", "종목코드": "", "종목명": "현금", "cash": 1.0,
             "blocked": 0.0}])
        s = SS.build(str(tmp_path))
        assert asked["codes"] == {"000777"}, "현금 행은 이력 조회 대상이 아니다"
        assert s["variants"]["ext_lowtv"]["days"] == 1
        assert s["variants"]["thin_skip"]["avg_ret_pct"] == 0.0


# ══ 실데이터 — CRITICAL 차단일에도 live 반사실이 나온다 ═══════════════════
@pytest.mark.parametrize("ymd", ["20261023", "20261102", "20261106"])
def test_real_critical_blocked_days_have_live_counterfactual(ymd):
    p = DATA / f"recommend_{ymd}.csv"
    if not p.exists():
        pytest.skip("실데이터 없음")
    df = pd.read_csv(p, low_memory=False, dtype={"종목코드": str})
    if not SS.day_blocked(df):
        pytest.skip("차단일 아님")
    picks = {r["variant"] for r in SS.pick_today(df, ymd)}
    assert {"live", "no_alpha", "thin_skip", "lowvol", "comp_lowvol"} <= picks
