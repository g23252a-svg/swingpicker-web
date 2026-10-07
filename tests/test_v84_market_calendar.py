# -*- coding: utf-8 -*-
"""[v84] 휴장일 가드 · 팬텀 배치 정리 · 확장 유니버스 그림자.

■ 팬텀 배치 — 2026년 10건 실측 (전 배치와 종가 100% 동일)
  03-02 · 05-01 · 05-05 · 05-25 · 06-03 · 07-17 · 08-17 · 09-24 · 09-25 · 10-05
  평일 휴장일(대체공휴일·선거·추석)에 체인이 배치를 돌리고 시세 소스가 마지막 거래일
  봉을 돌려줘 생겼다. 같은 픽이 이틀로 기록돼 픽 이력·그림자·레인 성적이 중복됐다.

■ 규칙
  수집 OHLCV의 '오늘 봉 보유 종목 비율' ≤ 50% → 휴장일 → 산출물 안 씀 + 마커.
  표본 < 30 → 판정 보류(진행). 달력 하드코딩 없음.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import market_calendar as MC  # noqa: E402
from services import selection_shadow as SS  # noqa: E402

PF_SRC = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")
WF_SRC = (ROOT / ".github" / "workflows" / "auto_collect.yml").read_text(encoding="utf-8")


def _ohlcv_map(n: int, last_ymd: str, n_today: int | None = None, index_kind: str = "index"):
    """n 종목 중 n_today 종목만 last_ymd 봉을 갖고 나머지는 하루 전까지."""
    n_today = n if n_today is None else n_today
    out = {}
    for i in range(n):
        end = pd.Timestamp(last_ymd) if i < n_today else pd.Timestamp(last_ymd) - pd.tseries.offsets.BDay(1)
        dates = pd.bdate_range(end=end, periods=30)
        df = pd.DataFrame({"시가": 100.0, "고가": 101.0, "저가": 99.0, "종가": 100.0, "거래량": 1000}, index=dates)
        if index_kind == "column":
            df = df.reset_index().rename(columns={"index": "날짜"})
        out[f"{i:06d}"] = df
    return out


class TestTradingSessionGuard:
    def test_full_coverage_is_trading_day(self):
        ok, info = MC.is_trading_session(_ohlcv_map(50, "20261006"), "20261006")
        assert ok and info["verdict"] == "trading" and info["share_today"] == 1.0

    def test_holiday_has_no_today_bars(self):
        """대체휴일: 모든 종목의 마지막 봉이 전 거래일 → 휴장."""
        ok, info = MC.is_trading_session(_ohlcv_map(50, "20261002"), "20261005")
        assert not ok and info["verdict"] == "holiday" and info["share_today"] == 0.0

    def test_threshold_is_not_knife_edge(self):
        """일부 종목만 오늘 봉이 없어도(수집 누락) 거래일이다 — 50% 문턱."""
        ok, _ = MC.is_trading_session(_ohlcv_map(50, "20261006", n_today=40), "20261006")
        assert ok
        ok2, _ = MC.is_trading_session(_ohlcv_map(50, "20261006", n_today=10), "20261006")
        assert not ok2

    def test_small_sample_never_blocks(self):
        """표본 부족은 진행 — 이 가드가 거래일 산출을 막는 쪽으로 틀리면 안 된다."""
        ok, info = MC.is_trading_session(_ohlcv_map(10, "20261002"), "20261005")
        assert ok and info["verdict"] == "insufficient"
        ok2, _ = MC.is_trading_session({}, "20261005")
        assert ok2

    def test_date_column_frames_supported(self):
        ok, info = MC.is_trading_session(_ohlcv_map(40, "20261002", index_kind="column"), "20261005")
        assert not ok and info["tickers"] == 40

    def test_empty_frames_ignored(self):
        m = _ohlcv_map(40, "20261006"); m["999999"] = pd.DataFrame(); m["999998"] = None
        ok, info = MC.is_trading_session(m, "20261006")
        assert ok and info["tickers"] == 40

    def test_marker_written(self, tmp_path):
        _, info = MC.is_trading_session(_ohlcv_map(50, "20261002"), "20261005")
        p = MC.write_skip_marker(str(tmp_path), info)
        assert os.path.basename(p) == "holiday_skip_20261005.json"
        assert json.load(open(p, encoding="utf-8"))["verdict"] == "holiday"


class TestPhantomDetection:
    def _write(self, d: Path, ymd: str, closes):
        pd.DataFrame({"종목코드": [f"{i:06d}" for i in range(len(closes))], "종가": closes}).to_csv(
            d / f"recommend_{ymd}.csv", index=False)

    def test_duplicate_closes_flagged_and_normal_days_not(self, tmp_path):
        base = list(np.arange(1000, 1000 + 40 * 10, 10, dtype=float))
        self._write(tmp_path, "20261001", base)
        self._write(tmp_path, "20261002", [c * 1.01 for c in base])          # 정상
        self._write(tmp_path, "20261005", [c * 1.01 for c in base])          # 10/2 복제 → 팬텀
        self._write(tmp_path, "20261006", [c * 1.02 for c in base])
        assert MC.phantom_batch_days(str(tmp_path)) == ["20261005"]

    def test_small_overlap_not_judged(self, tmp_path):
        self._write(tmp_path, "20261001", [1.0] * 10)
        self._write(tmp_path, "20261002", [1.0] * 10)
        assert MC.phantom_batch_days(str(tmp_path)) == []

    @pytest.mark.skipif(not (ROOT / "data" / "recommend_20260911.csv").exists(), reason="실데이터 없음")
    def test_repo_data_has_no_phantom_batches(self):
        """트립와이어 — 팬텀 배치는 data/phantom_batches/ 로 옮겨 글롭 밖에 둔다."""
        assert MC.phantom_batch_days(str(ROOT / "data")) == []

    @pytest.mark.skipif(not (ROOT / "data" / MC.PHANTOM_DIR).exists(), reason="정리 폴더 없음")
    def test_known_phantoms_are_archived(self):
        got = {p.name for p in (ROOT / "data" / MC.PHANTOM_DIR).glob("recommend_*.csv")}
        for ymd in ("20260817", "20260924", "20260925", "20261005"):
            assert f"recommend_{ymd}.csv" in got


class TestWiring:
    def test_pipeline_guard_runs_before_any_output(self):
        i_def = PF_SRC.index("def finalize_outputs(")
        i_guard = PF_SRC.index("_MC.is_trading_session(", i_def)
        i_save = PF_SRC.index("df_out.to_csv(op_d", i_def)
        assert i_def < i_guard < i_save
        blk = PF_SRC[i_guard:i_guard + 900]
        assert "write_skip_marker" in blk and "return" in blk

    def test_workflow_gates_post_steps_on_holiday(self):
        assert 'echo "holiday=true" >> "$GITHUB_OUTPUT"' in WF_SRC
        assert WF_SRC.count("steps.collect.outputs.holiday != 'true'") == 6
        # 커밋 스텝은 게이트하지 않는다 — 마커·캐시는 남겨야 다음 날 추적이 맞다
        i_commit = WF_SRC.index("- name: Commit & Push CSV")
        assert "steps.collect.outputs.holiday" not in WF_SRC[i_commit:i_commit + 200]


# ══ 확장 유니버스 그림자 (ext_lowtv) ═══════════════════════════════════════
def _universe(n_days: int = 80):
    """세 종목: 조용·하락·저거래대금(A) / 급등·고거래대금(B) / 중간(C) + 패딩 30종목."""
    dates = pd.bdate_range("20260601", periods=n_days)
    frames = []
    def mk(code, path_pct, vol, last_vol_mult=1.0):
        px = 10000 * (1 + path_pct) ** np.arange(n_days)
        v = np.full(n_days, vol, dtype=float); v[-1] *= last_vol_mult
        frames.append(pd.DataFrame({"Date": dates, "종목코드": code, "시가": px, "고가": px * 1.01,
                                    "저가": px * 0.99, "종가": px, "거래량": v}))
    mk("000001", -0.002, 50_000)            # A: 하락·저거래대금·RSI 낮음
    mk("000002", +0.004, 2_000_000, 4.0)    # B: 상승·고거래대금·거래량 폭발
    mk("000003", +0.000, 300_000)           # C
    for i in range(4, 34):
        mk(f"{i:06d}", 0.0005 * (i % 5), 200_000 + i * 10_000)
    return pd.concat(frames, ignore_index=True)


class TestExtLowTv:
    def test_registry_is_fixed(self):
        assert SS.EXT_REGISTRY == "v1-20261006"
        assert SS.EXT_FEATURES == ("tv_rank_pctl", "ret_20d", "hi60_gap", "rsi14", "vol_ratio")

    def test_picks_quiet_beaten_down_low_turnover(self):
        px = _universe()
        e = SS.ext_lowtv_pick("unused", px["Date"].max().strftime("%Y%m%d"), px=px)
        assert e is not None and e["종목코드"] == "000001"
        assert e["variant"] == "ext_lowtv" and 0 <= e["ext_score"] <= 1

    def test_requires_today_bar_and_min_turnover(self):
        px = _universe()
        px = px[~((px["종목코드"] == "000001") & (px["Date"] == px["Date"].max()))]   # A는 오늘 봉 없음(정지)
        e = SS.ext_lowtv_pick("unused", px["Date"].max().strftime("%Y%m%d"), px=px)
        assert e is not None and e["종목코드"] != "000001"

    def test_too_few_names_means_cash(self):
        px = _universe(); px = px[px["종목코드"].isin(["000001", "000002", "000003"])]
        assert SS.ext_lowtv_pick("unused", px["Date"].max().strftime("%Y%m%d"), px=px) is None

    def test_pick_today_adds_variant_only_with_data_dir(self, monkeypatch):
        df = pd.DataFrame([{"종목코드": "000001", "종목명": "A", "PRODUCTION_BUY": 1, "ALPHA_SCORE": 90,
                            "ALPHA_ENTRY_THRESHOLD": 85, "RR_NOW_TP1": 2.0, "V23_ATR_Pct": 0.05,
                            "ENTRY_RISK_GATE_OK": 1, "ALPHA_VOL_OK": 1}])
        px = _universe()
        monkeypatch.setattr(SS, "_load_universe_ohlcv", lambda _d: px)
        ymd = px["Date"].max().strftime("%Y%m%d")
        without = {x["variant"] for x in SS.pick_today(df, ymd)}
        with_ = {x["variant"] for x in SS.pick_today(df, ymd, data_dir="x")}
        assert "ext_lowtv" not in without and "ext_lowtv" in with_
        row = [x for x in SS.pick_today(df, ymd, data_dir="x") if x["variant"] == "ext_lowtv"][0]
        assert row["종목명"] == "A", "배치 df에 있는 종목이면 이름을 붙인다"

    def test_name_falls_back_to_krx_codes_for_lane_stock(self, tmp_path, monkeypatch):
        """[v84.1] 풀 밖 종목은 배치 df에 이름이 없다 — krx_codes 에서 찾고, 없으면 코드."""
        px = _universe(); ymd = px["Date"].max().strftime("%Y%m%d")
        monkeypatch.setattr(SS, "_load_universe_ohlcv", lambda _d: px)
        df = pd.DataFrame([{"종목코드": "000002", "종목명": "B", "PRODUCTION_BUY": 1, "ALPHA_SCORE": 90,
                            "ALPHA_ENTRY_THRESHOLD": 85, "RR_NOW_TP1": 2.0, "V23_ATR_Pct": 0.05,
                            "ENTRY_RISK_GATE_OK": 1, "ALPHA_VOL_OK": 1}])
        ext = [x for x in SS.pick_today(df, ymd, data_dir=str(tmp_path)) if x["variant"] == "ext_lowtv"][0]
        assert ext["종목코드"] == "000001" and ext["종목명"] == "000001"        # krx_codes 없음 → 코드
        pd.DataFrame({"종목코드": ["000001"], "종목명": ["서산"]}).to_csv(tmp_path / "krx_codes_20261006.csv", index=False)
        ext = [x for x in SS.pick_today(df, ymd, data_dir=str(tmp_path)) if x["variant"] == "ext_lowtv"][0]
        assert ext["종목명"] == "서산"

    def test_build_pairs_ext_with_live(self, tmp_path, monkeypatch):
        px = _universe(); ymd = "20260910"
        monkeypatch.setattr(SS, "_load_universe_ohlcv", lambda _d: px)
        monkeypatch.setattr(SS, "LIVE_FROM", "20260901")
        SS.append_log(str(tmp_path), [{"ymd": ymd, "variant": "live", "종목코드": "000002", "종목명": "B"},
                                      {"ymd": ymd, "variant": "ext_lowtv", "종목코드": "000001", "종목명": "A"}])
        s = SS.build(str(tmp_path))
        assert "ext_lowtv-live" in s["paired"] and s["variants"]["ext_lowtv"]["days"] == 1
        assert "확장유니버스" in SS.line(s)
