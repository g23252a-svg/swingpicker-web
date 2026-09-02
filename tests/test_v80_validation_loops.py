# -*- coding: utf-8 -*-
"""v80 — 레인 실전 성적 추적 · 성공본 보호 · 주말 지연 발화 스킵 · 검증 루프 화면."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from services import quiet_breakout as QB
from services import quiet_lane_track as QT


# ── 합성: OHLCV + 레인 픽 ────────────────────────────────────────────────
def _mk(data_dir, n_days=30, codes=("000001", "000002", "000003"), trend=0.0):
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2026-08-28", periods=n_days)
    rows = []
    for c in codes:
        px = 10_000 * np.cumprod(1 + rng.normal(trend, 0.01, n_days))
        for j, dt in enumerate(dates):
            rows.append({"Date": dt, "종목코드": c, "시가": px[j], "고가": px[j] * 1.01,
                         "저가": px[j] * 0.99, "종가": px[j], "거래량": 100_000, "등락률": 0.0})
    pd.DataFrame(rows).set_index("Date").to_parquet(
        os.path.join(data_dir, "ohlcv_cache_20260901.parquet"))
    return [d.strftime("%Y%m%d") for d in dates]


def _lane_json(data_dir, ymd, codes, ok=True, reason=None):
    rep = {"ok": ok, "trade_ymd": ymd,
           "picks": [{"종목코드": c, "종목명": f"N{c}", "거래대금순위": 800, "vol_ratio": 2.0}
                     for c in codes] if ok else [], "reason": reason}
    with open(os.path.join(data_dir, f"quiet_breakout_{ymd}.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False)
    return rep


class TestLaneTrack:
    def test_closed_windows_only(self, tmp_path):
        d = str(tmp_path); days = _mk(d)
        _lane_json(d, days[10], ["000001", "000002"])      # 창 닫힘
        _lane_json(d, days[-2], ["000003"])                 # 아직 안 닫힘
        s = QT.build(d)
        assert s["days_total"] == 2 and s["days_measured"] == 1
        assert s["picks_total"] == 3 and s["picks_measured"] == 2

    def test_cost_subtracted_like_backtest(self, tmp_path):
        d = str(tmp_path); days = _mk(d, trend=0.0)
        _lane_json(d, days[10], ["000001"])
        s = QT.build(d)
        raw = s["daily"][0]["ret_pct"] + QT.COST_PCT
        assert s["avg_ret_pct_after_cost"] == pytest.approx(raw - QT.COST_PCT)
        assert s["cost_pct"] == 0.51

    def test_pre_live_files_ignored(self, tmp_path):
        d = str(tmp_path); days = _mk(d)
        _lane_json(d, "20260701", ["000001"])                # 배포 전 드라이런
        _lane_json(d, days[10], ["000001"])
        assert QT.build(d)["days_total"] == 1

    def test_failed_lane_days_do_not_count(self, tmp_path):
        d = str(tmp_path); days = _mk(d)
        _lane_json(d, days[10], [], ok=False, reason="밴드 비어있음")
        s = QT.build(d)
        assert s["picks_total"] == 0

    def test_verdict_flips_at_target(self, tmp_path):
        d = str(tmp_path); days = _mk(d, n_days=60)
        for y in days[5:5 + QT.TARGET_DAYS]:
            _lane_json(d, y, ["000001"])
        s = QT.build(d)
        assert s["days_measured"] >= QT.TARGET_DAYS
        assert "판정 가능" in s["verdict"] and "승격하지 않는다" in s["verdict"]

    def test_line_and_roundtrip(self, tmp_path):
        d = str(tmp_path); days = _mk(d)
        _lane_json(d, days[10], ["000001"])
        s = QT.run_batch(d)
        assert QT.load(d)["days_measured"] == s["days_measured"]
        assert "1/20일" in QT.line(s) and "백테스트" in QT.line(s)
        assert QT.line(None) == ""


class TestNoClobber:
    def test_failed_report_cannot_overwrite_success(self, tmp_path):
        d = str(tmp_path)
        good = {"ok": True, "picks": [{"종목코드": "000001"}]}
        assert QB.save(d, "20260828", good) is True
        bad = {"ok": False, "reason": "밴드 비어있음"}
        assert QB.save(d, "20260828", bad) is False
        assert QB.load(d, "20260828")["ok"] is True, "8/29 사고 재발 — 성공본이 덮였다"

    def test_success_can_overwrite_failure(self, tmp_path):
        d = str(tmp_path)
        QB.save(d, "20260828", {"ok": False, "reason": "x"})
        assert QB.save(d, "20260828", {"ok": True, "picks": []}) is True
        assert QB.load(d, "20260828")["ok"] is True

    def test_success_overwrites_success(self, tmp_path):
        """정상 재실행(같은 날 두 번 성공)은 최신본을 쓴다."""
        d = str(tmp_path)
        QB.save(d, "20260828", {"ok": True, "picks": [], "v": 1})
        QB.save(d, "20260828", {"ok": True, "picks": [], "v": 2})
        assert QB.load(d, "20260828")["v"] == 2


class TestWorkflow:
    def _wf(self):
        return open(".github/workflows/auto_collect.yml", encoding="utf-8").read()

    def test_weekend_schedule_skipped_before_csv_check(self):
        src = self._wf()
        i_weekend = src.index('date +%u)" -ge 6')
        i_csv = src.index('-f "data/recommend_${TODAY}.csv"')
        i_dispatch = src.index('"workflow_dispatch" ]')
        assert i_dispatch < i_weekend < i_csv, "dispatch → 주말 → CSV 순이어야 한다"

    def test_main_cron_off_busy_slot(self):
        d = yaml.safe_load(self._wf())
        crons = [c["cron"] for c in d[True]["schedule"]]
        assert "23 11 * * 1-5" in crons and "40 13 * * 1-5" in crons
        for c in crons:
            assert not c.startswith(("0 ", "5 ")), f"정각 근처 혼잡 슬롯: {c}"

    def test_backfill_workflow_is_manual_only(self):
        d = yaml.safe_load(open(".github/workflows/backfill_flow.yml", encoding="utf-8"))
        assert list(d[True].keys()) == ["workflow_dispatch"]
        assert "start" in d[True]["workflow_dispatch"]["inputs"]


class TestScreen:
    def test_payload_survives_empty(self, tmp_path):
        from components.decision_center import _validation_payload
        p = _validation_payload(pd.DataFrame(), str(tmp_path))
        assert p["lane_picks"] == [] and p["budget_line"] == ""
        assert any("약속이 아니다" in s for s in p["stop_lines"])   # 갭 실측은 상수라 항상 뜬다

    def test_payload_reads_lane_and_budget(self, tmp_path):
        from components.decision_center import _validation_payload
        d = str(tmp_path); days = _mk(d)
        _lane_json(d, days[-1], ["000001", "000002"])
        QB.save(d, days[-1], json.load(open(os.path.join(d, f"quiet_breakout_{days[-1]}.json"))))
        QT.run_batch(d)
        df = pd.DataFrame({"BUDGET_SCALE": [0.2], "BUDGET_USED_PCT": [80.0],
                           "BUDGET_ROOM_PCT": [20.0], "KELLY_FRACTION": [0.05],
                           "켈리_수량": [3], "추천매수가": [10000], "손절가": [9200]})
        p = _validation_payload(df, d)
        assert len(p["lane_picks"]) == 2 and p["lane_picks"][0]["rank"] == 800
        assert "보유 중 80%" in p["budget_line"] and "오늘 배정 20%" in p["budget_line"]
        assert any("손절폭" in s for s in p["stop_lines"])

    def test_render_wired(self):
        src = open("components/decision_center.py", encoding="utf-8").read()
        assert "_render_validation_loops(_validation_payload(df))" in src
        assert "주문에 반영되지 않음" in src

    def test_finalize_runs_track_after_lane(self):
        src = open("pipeline_finalize.py", encoding="utf-8").read()
        assert src.index("_qb.run_batch") < src.index("_QT.run_batch") < src.index("_IFF.collect")
