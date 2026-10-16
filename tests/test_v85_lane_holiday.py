# -*- coding: utf-8 -*-
"""[v85] 레인 캐시 '날짜' 인덱스 버그 + 휴장일 팬텀 캐시.

■ 레인 버그 (v78~v84)
  레인 캐시(quiet_lane_ohlcv_*)는 pykrx 원본 그대로 '날짜' 인덱스로 저장된다. 그런데
  winner_profile._load_universe_ohlcv 는 reset_index() 뒤 'Date' 만 읽어 레인 행의 날짜가
  전부 NaT → 결측 방어 필터에서 **레인 전체가 버려졌다**. 영향:
    - selection_shadow.ext_lowtv: '≈1,200종목' 이 실제로는 배치 캐시 ≈600 (실측 599~609 → 수정 후 ~1,190)
    - quiet_lane_track: 레인 픽 130개 중 2개만 측정 (수정 + 최근 파일 폴백 후 46개, 20일)
    - winner_profile.build_day: 코호트가 ≈600
  레인은 하루 ~30%씩 바뀌므로, 최신 파일에 없는 픽 종목은 그 종목을 담은 가장 최근 날짜
  파일에서 잰다(_load_code_history).

■ 휴장일 팬텀 캐시
  v84 가드(finalize_outputs)는 collector 가 ohlcv_cache_{ymd} 를 저장한 **뒤에** 돈다 →
  10/09(휴장)에도 전일 복제 캐시가 커밋됐다. 저장 전에 같은 판정을 한다(ohlcv_store.
  cache_write_allowed). 'holiday' 만 막고, 표본 부족(insufficient)·판정 실패는 저장한다.
  같은 실행의 CARRY 재수집(같은 trade_ymd, 종목 몇 개)도 기억된 휴장 판정으로 막는다.
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

from services import ohlcv_store as OS  # noqa: E402
from services import quiet_lane_track as QT  # noqa: E402
from services import selection_shadow as SS  # noqa: E402
from services import winner_profile as WP  # noqa: E402

DATA = ROOT / "data"


# ── 합성: 실제 작성기 경로로 배치/레인 캐시를 쓴다 ─────────────────────────────
def _series(end: str, n: int, base: float, drift: float, vol: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    c = np.round(base * np.cumprod(1 + rng.normal(drift, 0.01, n))).astype(np.int64)
    df = pd.DataFrame({"시가": c, "고가": np.round(c * 1.01).astype(np.int64),
                       "저가": np.round(c * 0.99).astype(np.int64), "종가": c,
                       "거래량": np.full(n, vol, dtype=np.int64)}, index=idx)
    df.index.name = "날짜"
    return df


def _write_main(d, ymd, data):
    from data_source import OHLCVCache
    m = {c: g.assign(등락률=0.0) for c, g in data.items()}
    OHLCVCache(out_dir=str(d)).save(ymd, m)


def _write_lane(d, ymd, data):
    from services import quiet_breakout as QB
    QB._save_lane_cache(str(d), ymd, data)


def _lane_json(d, ymd, codes):
    rep = {"ok": True, "trade_ymd": ymd,
           "picks": [{"종목코드": c, "종목명": f"N{c}"} for c in codes]}
    with open(os.path.join(str(d), f"quiet_breakout_{ymd}.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False)


class TestLaneDateIndex:
    def test_lane_rows_are_loaded(self, tmp_path):
        main = {f"1{i:05d}": _series("2026-10-16", 80, 10_000, 0.001, 300_000, i) for i in range(5)}
        lane = {f"2{i:05d}": _series("2026-10-16", 80, 5_000, 0.0, 80_000, 100 + i) for i in range(4)}
        lane["100000"] = _series("2026-10-16", 80, 1, 0.0, 1, 999)        # 양쪽에 있는 종목
        _write_main(tmp_path, "20261016", main)
        _write_lane(tmp_path, "20261016", lane)
        assert pd.read_parquet(tmp_path / "quiet_lane_ohlcv_20261016.parquet").index.name == "날짜"
        px = WP._load_universe_ohlcv(str(tmp_path))
        codes = set(px["종목코드"])
        assert codes == set(main) | set(lane)                               # 수정 전: set(main) 뿐
        assert px["Date"].notna().all() and str(px["Date"].dtype).startswith("datetime64")
        lane_only = px[px["종목코드"].str.startswith("2")]
        assert len(lane_only) == 4 * 80
        # 같은 (종목, 날짜) 가 둘 다 있으면 배치 캐시 값을 남긴다(기존 규칙)
        both = px[px["종목코드"] == "100000"]
        assert len(both) == 80 and (both["종가"] > 100).all()

    def test_with_date_col(self):
        a = pd.DataFrame({"날짜": [1], "x": [2]})
        assert list(WP._with_date_col(a).columns) == ["Date", "x"]
        b = pd.DataFrame({"Date": [1]})
        assert WP._with_date_col(b) is b
        assert WP._with_date_col(pd.DataFrame({"x": [1]})) is None

    def test_ext_lowtv_now_sees_lane_universe(self, tmp_path):
        """선등록 5특징(거래대금·20일수익·60일고점이격·RSI·거래량비)이 모두 낮은 쪽은 레인 종목."""
        main = {f"1{i:05d}": _series("2026-10-16", 80, 20_000, 0.004, 900_000, i) for i in range(30)}
        lane = {f"2{i:05d}": _series("2026-10-16", 80, 5_000, -0.004, 80_000, 100 + i) for i in range(30)}
        _write_main(tmp_path, "20261016", main)
        _write_lane(tmp_path, "20261016", lane)
        e = SS.ext_lowtv_pick(str(tmp_path), "20261016")
        assert e is not None and e["종목코드"] in lane                       # 수정 전: 레인이 안 보여 main 에서 골랐다


class TestHistoryFallback:
    def test_load_code_history_newest_file_wins(self, tmp_path):
        old = {"200001": _series("2026-10-07", 60, 5_000, 0.0, 80_000, 1)}
        new = {"200001": _series("2026-10-09", 62, 5_000, 0.0, 80_000, 1)}
        _write_lane(tmp_path, "20261007", old)
        _write_lane(tmp_path, "20261009", new)
        _write_lane(tmp_path, "20261013", {"200002": _series("2026-10-13", 60, 9_000, 0.0, 80_000, 2)})
        h = WP._load_code_history(str(tmp_path), ["200001", "999999"])
        assert set(h) == {"200001"}
        assert h["200001"]["Date"].max() == pd.Timestamp("2026-10-09")
        assert len(h["200001"]) == 62
        # since 이후 파일만
        assert WP._load_code_history(str(tmp_path), ["200001"], since="20261010") == {}

    def test_same_day_main_preferred(self, tmp_path):
        _write_lane(tmp_path, "20261009", {"300001": _series("2026-10-09", 30, 5_000, 0.0, 1, 1)})
        _write_main(tmp_path, "20261009", {"300001": _series("2026-10-09", 30, 7_000, 0.0, 2, 2)})
        h = WP._load_code_history(str(tmp_path), ["300001"])
        assert (h["300001"]["거래량"] == 2).all()

    def test_lane_track_measures_churned_pick(self, tmp_path):
        """10/01 레인 픽 — 10/13 이후 밴드에서 빠졌지만 10/12 레인 파일에 t+6 까지 있다."""
        days = pd.bdate_range(end="2026-10-16", periods=40)
        sig = days[-12].strftime("%Y%m%d")
        mainu = {f"1{i:05d}": _series("2026-10-16", 40, 10_000, 0.0, 300_000, i) for i in range(3)}
        _write_main(tmp_path, "20261016", mainu)
        churned = _series(days[-6].strftime("%Y-%m-%d"), 34, 5_000, 0.002, 80_000, 7)
        _write_lane(tmp_path, days[-6].strftime("%Y%m%d"), {"200007": churned})
        _write_lane(tmp_path, "20261016", {"200008": _series("2026-10-16", 40, 5_000, 0.0, 80_000, 8)})
        _lane_json(tmp_path, sig, ["200007", "200008", "999999"])
        s = QT.build(str(tmp_path))
        assert s["picks_total"] == 3
        assert s["picks_measured"] == 2                                     # 200007(폴백) + 200008(최신 레인)
        from services.pick_history import _realized
        g = WP._with_date_col(pd.read_parquet(
            tmp_path / f"quiet_lane_ohlcv_{days[-6].strftime('%Y%m%d')}.parquet").reset_index())
        g = g[g["종목코드"] == "200007"].reset_index(drop=True)
        assert _realized(g, sig) is not None

    def test_shadow_build_measures_churned_lane_pick(self, tmp_path):
        days = pd.bdate_range(end="2026-10-16", periods=40)
        sig = days[-12].strftime("%Y%m%d")
        _write_main(tmp_path, "20261016", {f"1{i:05d}": _series("2026-10-16", 40, 10_000, 0.0, 300_000, i)
                                           for i in range(3)})
        _write_lane(tmp_path, days[-6].strftime("%Y%m%d"),
                    {"200007": _series(days[-6].strftime("%Y-%m-%d"), 34, 5_000, 0.002, 80_000, 7)})
        log = pd.DataFrame([{"ymd": sig, "variant": "ext_lowtv", "종목코드": "200007", "종목명": "A"},
                            {"ymd": sig, "variant": "live", "종목코드": "100001", "종목명": "B"}])
        log.to_parquet(tmp_path / SS.LOG_NAME, index=False)
        s = SS.build(str(tmp_path))
        assert s["variants"]["ext_lowtv"]["days"] == 1 and s["variants"]["live"]["days"] == 1


class TestRealLane:
    def test_real_lane_rows_loaded(self, real_data_mirror):
        lanes = sorted(p for p in DATA.glob("quiet_lane_ohlcv_2*.parquet"))
        mains = sorted(p for p in DATA.glob("ohlcv_cache_2*.parquet"))
        if not lanes or not mains:
            pytest.skip("실데이터 레인/배치 캐시 없음")
        d = real_data_mirror(mains[-1].name, lanes[-1].name)
        px = WP._load_universe_ohlcv(d)
        main_codes = set(pd.read_parquet(mains[-1], columns=["종목코드"])["종목코드"].astype(str).str.zfill(6))
        lane_raw = pd.read_parquet(lanes[-1])
        lane_codes = set(lane_raw["종목코드"].astype(str).str.zfill(6))
        assert set(px["종목코드"]) == main_codes | lane_codes
        lane_only = lane_codes - main_codes
        assert len(lane_only) > 100                                         # 수정 전: 레인 전용 종목 0개
        n_lane_only_rows = int(lane_raw["종목코드"].astype(str).str.zfill(6).isin(lane_only).sum())
        assert int(px["종목코드"].isin(lane_only).sum()) == n_lane_only_rows

    def test_real_lane_track_measures_lane_only_picks(self, real_data_mirror):
        lanes = sorted(DATA.glob("quiet_lane_ohlcv_2*.parquet"))
        mains = sorted(DATA.glob("ohlcv_cache_2*.parquet"))
        if not lanes or not mains or not list(DATA.glob("quiet_breakout_2*.json")):
            pytest.skip("실데이터 레인 없음")
        d = real_data_mirror("ohlcv_cache_2*.parquet", "quiet_lane_ohlcv_2*.parquet", "quiet_breakout_2*.json")
        from services.pick_history import _realized
        lane = WP._with_date_col(pd.read_parquet(lanes[-1]).reset_index())
        lane["종목코드"] = lane["종목코드"].astype(str).str.zfill(6)
        lane["Date"] = pd.to_datetime(lane["Date"])
        main_codes = set(pd.read_parquet(mains[-1], columns=["종목코드"])["종목코드"].astype(str).str.zfill(6))
        n = 0
        for ymd in QT._lane_days(d):
            for p in QT._load_picks(d, ymd):
                c = str(p["종목코드"]).zfill(6)
                if c in main_codes:
                    continue
                g = lane[lane["종목코드"] == c].reset_index(drop=True)
                if len(g) and _realized(g, ymd) is not None:
                    n += 1
        if n == 0:
            pytest.skip("현재 데이터에 최신 레인 파일로 잴 수 있는 레인 전용 픽이 없다")
        s = QT.build(d)
        assert s["picks_measured"] >= n                                     # 수정 전: 레인 전용 픽 0개 측정


# ══ 휴장일 팬텀 캐시 ═════════════════════════════════════════════════════════
def _map(n: int, last: str) -> dict:
    return {f"{i:06d}": _series(last, 30, 1000, 0.0, 100, i) for i in range(n)}


class TestCacheWriteAllowed:
    @pytest.fixture(autouse=True)
    def _fresh_memo(self, monkeypatch):
        monkeypatch.setattr(OS, "_HOLIDAY_YMDS", set())

    def test_trading_day_saves(self):
        ok, info = OS.cache_write_allowed(_map(40, "2026-10-08"), "20261008")
        assert ok and info["verdict"] == "trading"

    def test_holiday_blocks_and_is_remembered(self):
        ok, info = OS.cache_write_allowed(_map(40, "2026-10-08"), "20261009")
        assert not ok and info["verdict"] == "holiday"
        # 같은 실행의 CARRY 재수집 — 종목 3개(표본 부족)라도 같은 날짜면 막는다
        ok2, info2 = OS.cache_write_allowed(_map(3, "2026-10-08"), "20261009")
        assert not ok2 and info2.get("memo")

    def test_insufficient_sample_still_saves(self):
        """판정 보류 = 진행 (v84 규칙) — 거래일 캐시를 버리는 쪽으로 틀리면 안 된다."""
        ok, info = OS.cache_write_allowed(_map(5, "2026-10-08"), "20261009")
        assert ok and info["verdict"] == "insufficient"

    def test_judgement_failure_still_saves(self, monkeypatch):
        from services import market_calendar as MC

        def boom(*a, **k):
            raise RuntimeError("x")
        monkeypatch.setattr(MC, "is_trading_session", boom)
        ok, info = OS.cache_write_allowed(_map(40, "2026-10-08"), "20261009")
        assert ok and info["verdict"] == "error"


class TestCollectorSaveSite:
    """collector.prepare_ohlcv_data 저장 지점 — 네트워크 없이 수집 함수를 바꿔 끼운다."""

    @pytest.fixture
    def col(self, monkeypatch):
        import collector
        monkeypatch.setattr(OS, "_HOLIDAY_YMDS", set())
        saved = []
        monkeypatch.setattr(collector, "load_ohlcv_cache", lambda ymd: {})
        monkeypatch.setattr(collector, "save_ohlcv_cache", lambda ymd, m: saved.append((ymd, len(m))))
        return collector, saved

    def _fetch(self, last):
        return lambda s, e, code: _series(last, 30, 1000, 0.0, 100, int(code))

    def test_holiday_run_writes_no_cache(self, col, monkeypatch):
        collector, saved = col
        monkeypatch.setattr(collector, "safe_ohlcv_by_date", self._fetch("2026-10-08"))
        m = collector.prepare_ohlcv_data([f"{i:06d}" for i in range(40)], "20250101", "20261009", "20261009")
        assert len(m) == 40 and saved == []                                 # 메모리 맵은 그대로 — 저장만 안 한다
        # 같은 실행의 CARRY 재수집(종목 3개) — 표본 부족이지만 휴장 판정이 기억돼 저장 안 함
        collector.prepare_ohlcv_data(["000001", "000002", "000003"], "20250101", "20261009", "20261009")
        assert saved == []

    def test_trading_day_and_small_sample_still_write(self, col, monkeypatch):
        collector, saved = col
        monkeypatch.setattr(collector, "safe_ohlcv_by_date", self._fetch("2026-10-08"))
        collector.prepare_ohlcv_data([f"{i:06d}" for i in range(40)], "20250101", "20261008", "20261008")
        assert saved == [("20261008", 40)]
        collector.prepare_ohlcv_data(["000001", "000002"], "20250101", "20261013", "20261013")
        assert saved[-1] == ("20261013", 2)                                  # insufficient → 저장

    def test_guard_sits_at_the_save_site(self):
        src = (ROOT / "collector.py").read_text(encoding="utf-8")
        i = src.index("def prepare_ohlcv_data(")
        body = src[i:src.index("\ndef ", i + 10)]
        j = body.index("save_ohlcv_cache(trade_ymd, ohlcv_map)")
        assert "cache_write_allowed(ohlcv_map, trade_ymd)" in body[:j]

    def test_lane_cache_written_after_holiday_guard(self):
        """레인 캐시는 finalize_outputs 의 휴장일 return 뒤에서만 쓰인다 — 팬텀 레인 없음."""
        src = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")
        i = src.index("def finalize_outputs(")
        guard = src.index("is_trading_session", i)
        lane = src.index("quiet_breakout", i)
        assert guard < lane


class TestRealPhantomCaches:
    def test_post_v84_holiday_left_a_phantom_main_cache(self):
        """10/09(휴장) — 기존 커밋된 팬텀 캐시. 이번 PR 은 옮기지 않는다(목록만 보고)."""
        f9 = DATA / "ohlcv_cache_20261009.parquet"
        f8 = DATA / "ohlcv_cache_20261008.parquet"
        if not (f9.exists() and f8.exists() and (DATA / "holiday_skip_20261009.json").exists()):
            pytest.skip("실데이터 없음")
        a = pd.read_parquet(f8).reset_index()
        b = pd.read_parquet(f9).reset_index()
        assert b["Date"].max() == a["Date"].max() == pd.Timestamp("2026-10-08")
        m = b.merge(a, on=["종목코드", "Date"], suffixes=("_b", "_a"))
        assert len(m) == len(b)
        for c in ("시가", "고가", "저가", "종가", "거래량"):
            assert (m[c + "_b"] == m[c + "_a"]).all()
        assert not (DATA / "quiet_lane_ohlcv_20261009.parquet").exists()     # 레인은 가드 뒤라 없음
