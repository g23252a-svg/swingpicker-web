# -*- coding: utf-8 -*-
"""[v85] OHLCV 버전 저장소 (services/ohlcv_store.py) — 그림자 이중 기록 단계.

■ 무엇을 지키나
  - 저장소(v = 바뀐 행, m = 그날 파일 구성)만으로 어떤 날짜 파일이든 **원본과 정확히 같은
    프레임**(인덱스 이름·dtype·컬럼 순서·행 순서·arrow 스키마 메타데이터)으로 복원된다.
  - 신규 행 / 수정주가 재작성(분할) / 당일 봉(시가 0) 다음 날 확정 / NaN / 같은 밤 재수집
    교체 / m 없는 밤 무시 / 레인 캐시('날짜' 인덱스) / 두 스키마 변형(string+ns, large_string+us).
  - 저장소 파일 이름은 소비자 글롭(ohlcv_cache_* · quiet_lane_ohlcv_* · recommend_*)에
    절대 걸리지 않는다.
  - 실데이터: 표본 밤(첫 밤·휴장 복제·부분 유니버스·분할 재작성 밤·마지막 밤)을 임시 미러로
    저장소에 넣고 원본과 정확히 같은지, 신규/개정 행 수가 parquet 에서 직접 계산한 값과 같은지.
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import ohlcv_store as OS  # noqa: E402

DATA = ROOT / "data"
WF_SRC = (ROOT / ".github" / "workflows" / "auto_collect.yml").read_text(encoding="utf-8")


# ── 합성 데이터: 실제 작성기와 같은 경로로 쓴다 ─────────────────────────────
def _bars(end: str, n: int, base: float = 1000.0, seed: int = 0) -> pd.DataFrame:
    """종목별 고정 시리즈(2025-06-02 ~ 2026-12-31 영업일)에서 end 까지 마지막 n 봉.
    같은 (seed, 날짜)는 어느 밤에 만들어도 같은 값이다 — 밤마다 바뀌는 건 새 날짜뿐."""
    rng = np.random.default_rng(seed)
    full = pd.bdate_range("2025-06-02", "2026-12-31")
    c = np.round(base * np.cumprod(1 + rng.normal(0, 0.02, len(full)))).astype(np.int64)
    vol = rng.integers(1_000, 90_000, len(full)).astype(np.int64)
    df = pd.DataFrame({"시가": c - 3, "고가": c + 7, "저가": c - 9, "종가": c, "거래량": vol}, index=full)
    df = df[df.index <= pd.Timestamp(end)].tail(n).copy()
    df["등락률"] = df["종가"].pct_change().mul(100).round(4)
    df.index.name = "날짜"                                  # pykrx 원본 인덱스 이름
    return df


def _write_main(d: Path, ymd: str, data: dict) -> str:
    """collector 경로 그대로 — data_source.OHLCVCache.save."""
    from data_source import OHLCVCache
    OHLCVCache(out_dir=str(d)).save(ymd, data)
    return str(d / f"ohlcv_cache_{ymd}.parquet")


def _write_lane(d: Path, ymd: str, data: dict) -> str:
    """레인 경로 그대로 — quiet_breakout._save_lane_cache ('날짜' 인덱스)."""
    from services import quiet_breakout as QB
    QB._save_lane_cache(str(d), ymd, {c: g[["시가", "고가", "저가", "종가", "거래량"]] for c, g in data.items()})
    return str(d / f"quiet_lane_ohlcv_{ymd}.parquet")


def _universe(end: str, codes=("005930", "000660", "035420"), n: int = 40) -> dict:
    # 코드 순서를 일부러 비정렬로 — 원본은 블록 순서가 정렬돼 있지 않다(실측)
    return {c: _bars(end, n, base=1000 + 500 * i, seed=i) for i, c in enumerate(codes)}


def _assert_exact(store_data_dir: str, ymd: str, kind: str, path: str, store_dir=None):
    got = OS.snapshot(store_data_dir, ymd, kind, store_dir=store_dir)
    want = pd.read_parquet(path)
    pd.testing.assert_frame_equal(got, want, check_exact=True)
    assert list(got.columns) == list(want.columns)
    assert got.index.name == want.index.name


def _v(d: Path, kind: str, ymd: str) -> pd.DataFrame:
    return pd.read_parquet(d / "ohlcv_store" / kind / "v" / f"{ymd}.parquet")


def _meta(d: Path, kind: str, ymd: str) -> dict:
    return OS._read_m(str(d / "ohlcv_store" / kind / "m" / f"{ymd}.parquet"))[1]


class TestSyntheticRoundTrip:
    def test_new_rows_only_and_exact(self, tmp_path):
        d = tmp_path
        u1 = _universe("2026-10-05", codes=("035420", "005930", "000660"))
        p1 = _write_main(d, "20261005", u1)
        u2 = {c: _bars("2026-10-06", 41, base=1000 + 500 * i, seed=i) for i, c in enumerate(u1)}
        p2 = _write_main(d, "20261006", u2)
        r = OS.ingest(str(d), kind="main")
        assert r["main"]["ingested"] == ["20261005", "20261006"] and not r["main"]["failed"]
        _assert_exact(str(d), "20261005", "main", p1)
        _assert_exact(str(d), "20261006", "main", p2)
        v2 = _v(d, "main", "20261006")
        # 둘째 밤은 새 날짜(10/06) 3종목만 — 나머지 40일은 첫 밤과 같은 값
        assert len(v2) == 3 and set(v2["date"]) == {pd.Timestamp("2026-10-06")}
        assert not v2["rev"].any()
        m = OS.members(str(d), "20261006")
        assert list(m["code"]) == ["035420", "005930", "000660"]       # 파일 블록 순서 보존
        assert m["n"].tolist() == [41, 41, 41]
        assert OS.verify(str(d), "all", kind="main", deep=True)["ok"]

    def test_split_rewrite_keeps_old_snapshot(self, tmp_path):
        """5:1 액면병합 — 셋째 밤에 한 종목 이력 전체가 새 기준으로 재작성된다."""
        d = tmp_path
        u = _universe("2026-10-05")
        p1 = _write_main(d, "20261005", u)
        u2 = {c: _bars("2026-10-06", 41, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        p2 = _write_main(d, "20261006", u2)
        u3 = {c: _bars("2026-10-07", 42, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        g = u3["000660"]
        for c in ("시가", "고가", "저가", "종가"):
            g[c] = g[c] * 5
        g["거래량"] = g["거래량"] // 5
        p3 = _write_main(d, "20261007", u3)
        OS.ingest(str(d), kind="main")
        v3 = _v(d, "main", "20261007")
        rev = v3[v3["rev"]]
        assert set(rev["code"]) == {"000660"} and len(rev) == 41      # 기존 41일 전부 개정
        assert (~v3["rev"]).sum() == 3                                 # 새 날짜 3종목
        for y, p in (("20261005", p1), ("20261006", p2), ("20261007", p3)):
            _assert_exact(str(d), y, "main", p)                        # 옛 스냅샷은 옛 기준 그대로
        assert _meta(d, "main", "20261007")["rev_codes"] == ["000660"]

    def test_same_day_placeholder_bar_revised_next_day(self, tmp_path):
        """당일 봉이 시가·고가·저가 0, 거래량 과소로 들어오고 다음 날 확정된다(실측 333쌍)."""
        d = tmp_path
        u = _universe("2026-10-05")
        u["005930"].iloc[-1, :3] = 0
        u["005930"].iloc[-1, u["005930"].columns.get_loc("거래량")] = 17
        p1 = _write_main(d, "20261005", u)
        u2 = {c: _bars("2026-10-06", 41, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        p2 = _write_main(d, "20261006", u2)
        OS.ingest(str(d), kind="main")
        v2 = _v(d, "main", "20261006")
        r = v2[v2["rev"]]
        assert len(r) == 1 and r["code"].iloc[0] == "005930" and r["date"].iloc[0] == pd.Timestamp("2026-10-05")
        _assert_exact(str(d), "20261005", "main", p1)                 # 자리표시 봉 그대로
        _assert_exact(str(d), "20261006", "main", p2)

    def test_nan_handling(self, tmp_path):
        """NaN==NaN 은 새 버전이 아니다. NaN → 숫자는 개정. 숫자 → NaN 도 개정."""
        d = tmp_path
        u = _universe("2026-10-05")
        p1 = _write_main(d, "20261005", u)
        u2 = {c: g.copy() for c, g in u.items()}
        u2["035420"].iloc[0, u2["035420"].columns.get_loc("등락률")] = 0.5     # NaN → 숫자
        u2["000660"].iloc[5, u2["000660"].columns.get_loc("등락률")] = np.nan  # 숫자 → NaN
        p2 = _write_main(d, "20261006", u2)
        OS.ingest(str(d), kind="main")
        v2 = _v(d, "main", "20261006")
        assert len(v2) == 2 and v2["rev"].all()
        assert pd.read_parquet(p1)["등락률"].isna().sum() == 3          # 각 종목 첫 행 NaN
        _assert_exact(str(d), "20261005", "main", p1)
        _assert_exact(str(d), "20261006", "main", p2)

    def test_gap_inside_code_and_unsorted_blocks(self, tmp_path):
        """둘째 밤에 한 종목의 중간 날짜가 빠져도(구간 둘) 정확히 복원된다."""
        d = tmp_path
        u = _universe("2026-10-05", codes=("900110", "000020", "500000"))
        p1 = _write_main(d, "20261005", u)
        u2 = {c: _bars("2026-10-06", 41, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        u2["000020"] = u2["000020"].drop(u2["000020"].index[10])
        p2 = _write_main(d, "20261006", u2)
        r = OS.ingest(str(d), kind="main")
        assert not r["main"]["failed"]
        m = OS.members(str(d), "20261006")
        assert (m["code"] == "000020").sum() == 2                     # 구간 두 개
        _assert_exact(str(d), "20261005", "main", p1)
        _assert_exact(str(d), "20261006", "main", p2)

    def test_duplicate_keys_night_fails_cleanly(self, tmp_path):
        """(종목, 날짜) 중복 파일은 표현할 수 없다 → 그 밤만 빠지고(m 없음) 다음 밤은 정상."""
        d = tmp_path
        u = _universe("2026-10-05")
        _write_main(d, "20261005", u)
        bad = pd.read_parquet(d / "ohlcv_cache_20261005.parquet")
        bad = pd.concat([bad, bad.iloc[:2]])
        bad.to_parquet(d / "ohlcv_cache_20261006.parquet")
        u3 = {c: _bars("2026-10-07", 42, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        p3 = _write_main(d, "20261007", u3)
        r = OS.ingest(str(d), kind="main")
        assert [f["ymd"] for f in r["main"]["failed"]] == ["20261006"]
        assert OS.nights(str(d), "main") == ["20261005", "20261007"]
        assert not (d / "ohlcv_store" / "main" / "v" / "20261006.parquet").exists()
        _assert_exact(str(d), "20261007", "main", p3)
        v = OS.verify(str(d), "all", kind="main")
        assert v["main"]["missing_store"] == ["20261006"] and not v["ok"]
        # 증분 재실행은 과거 누락을 다시 시도하지 않고 보고만 한다
        r2 = OS.ingest(str(d), kind="main")
        assert r2["main"]["skipped_older"] == ["20261006"] and not r2["main"]["ingested"]


class TestRerunAndMarkers:
    def test_same_night_rerun_replaces_only_that_night(self, tmp_path):
        d = tmp_path
        u = _universe("2026-10-05")
        p1 = _write_main(d, "20261005", u)
        u2 = {c: _bars("2026-10-06", 41, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        p2 = _write_main(d, "20261006", u2)
        OS.ingest(str(d), kind="main")
        v1_before = (d / "ohlcv_store" / "main" / "v" / "20261005.parquet").stat().st_mtime_ns
        m2 = d / "ohlcv_store" / "main" / "m" / "20261006.parquet"
        m2_before = m2.stat().st_mtime_ns
        # 같은 내용 재실행 → 아무것도 다시 쓰지 않는다
        r = OS.ingest(str(d), kind="main")
        assert r["main"]["unchanged"] == ["20261006"] and not r["main"]["replaced"]
        assert m2.stat().st_mtime_ns == m2_before
        # 수동 재수집: 같은 밤 파일이 다른 당일 거래량으로 다시 쓰였다
        u2["005930"].iloc[-1, u2["005930"].columns.get_loc("거래량")] += 1234
        p2 = _write_main(d, "20261006", u2)
        r = OS.ingest(str(d), kind="main")
        assert r["main"]["replaced"] == ["20261006"] and not r["main"]["failed"]
        assert (d / "ohlcv_store" / "main" / "v" / "20261005.parquet").stat().st_mtime_ns == v1_before
        _assert_exact(str(d), "20261005", "main", p1)
        _assert_exact(str(d), "20261006", "main", p2)
        v2 = _v(d, "main", "20261006")
        assert len(v2) == 3                                            # 교체 — 누적되지 않는다

    def test_missing_m_file_night_is_ignored(self, tmp_path):
        d = tmp_path
        u = _universe("2026-10-05")
        p1 = _write_main(d, "20261005", u)
        u2 = {c: _bars("2026-10-06", 41, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        _write_main(d, "20261006", u2)
        OS.ingest(str(d), kind="main")
        # 커밋 마커(m) 없이 v 만 남은 밤 = 중간에 죽은 실행 → 읽을 때 무시
        (d / "ohlcv_store" / "main" / "m" / "20261006.parquet").unlink()
        assert OS.nights(str(d), "main") == ["20261005"]
        with pytest.raises(OS.StoreError):
            OS.snapshot(str(d), "20261006", "main")
        _assert_exact(str(d), "20261005", "main", p1)
        # 고아 v 에 쓰레기 값이 있어도 다른 밤 복원에 섞이지 않는다
        junk = _v(d, "main", "20261006")
        junk["종가"] = -1
        pq.write_table(pa.Table.from_pandas(junk, preserve_index=False),
                       d / "ohlcv_store" / "main" / "v" / "20261006.parquet")
        assert (OS.latest_union(str(d), "main")["종가"] > 0).all()
        # 다음 ingest 가 그 밤을 다시 적는다
        r = OS.ingest(str(d), kind="main")
        assert r["main"]["ingested"] == ["20261006"]
        assert OS.verify(str(d), "all", kind="main", deep=True)["ok"]

    def test_stray_tmp_files_are_cleaned(self, tmp_path):
        d = tmp_path
        _write_main(d, "20261005", _universe("2026-10-05"))
        OS.ingest(str(d), kind="main")
        stray = d / "ohlcv_store" / "main" / "v" / f".20261006.parquet{OS._TMP_TAG}123"
        stray.write_bytes(b"x")
        OS.ingest(str(d), kind="main")
        assert not stray.exists()

    def test_older_night_changed_needs_full_mode(self, tmp_path):
        """증분(None)은 과거 밤 변경을 건드리지 않는다. 'all' 은 그 밤부터 다시 계산한다."""
        d = tmp_path
        u = _universe("2026-10-05")
        _write_main(d, "20261005", u)
        u2 = {c: _bars("2026-10-06", 41, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        _write_main(d, "20261006", u2)
        u3 = {c: _bars("2026-10-07", 42, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        _write_main(d, "20261007", u3)
        OS.ingest(str(d), kind="main")
        u2["000660"].iloc[3, 0] += 1
        p2 = _write_main(d, "20261006", u2)
        OS.ingest(str(d), kind="main")
        assert not OS.verify(str(d), "20261006", kind="main")["ok"]
        r = OS.ingest(str(d), ymd="all", kind="main")
        assert r["main"]["replaced"] == ["20261006", "20261007"]
        assert OS.verify(str(d), "all", kind="main", deep=True)["ok"]
        _assert_exact(str(d), "20261006", "main", p2)


class TestSchemaAndLane:
    def test_lane_date_index_exact(self, tmp_path):
        d = tmp_path
        u = _universe("2026-10-05", codes=("123450", "067890"))
        p1 = _write_lane(d, "20261005", u)
        assert pd.read_parquet(p1).index.name == "날짜"
        u2 = {c: _bars("2026-10-06", 41, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        p2 = _write_lane(d, "20261006", u2)
        r = OS.ingest(str(d), kind="lane")
        assert r["lane"]["ingested"] == ["20261005", "20261006"]
        _assert_exact(str(d), "20261005", "lane", p1)
        _assert_exact(str(d), "20261006", "lane", p2)
        assert _meta(d, "lane", "20261006")["date_col"] == "날짜"
        assert OS.nights(str(d), "main") == []                         # 종류는 서로 독립

    def test_both_schema_variants(self, tmp_path):
        """실측 두 변형: string + timestamp[ns] (24개) / large_string + timestamp[us] (140개)."""
        d = tmp_path
        u = _universe("2026-10-05")
        p1 = _write_main(d, "20261005", u)
        t = pq.read_table(p1)
        fields = []
        for f in t.schema:
            if f.name == "종목코드":
                fields.append(pa.field(f.name, pa.string()))
            elif pa.types.is_timestamp(f.type):
                fields.append(pa.field(f.name, pa.timestamp("ns")))
            else:
                fields.append(f)
        t_ns = t.cast(pa.schema(fields, metadata=t.schema.metadata))
        pq.write_table(t_ns, p1)
        fields = [pa.field(f.name, pa.large_string()) if f.name == "종목코드"
                  else (pa.field(f.name, pa.timestamp("us")) if pa.types.is_timestamp(f.type) else f)
                  for f in t.schema]
        u2 = {c: _bars("2026-10-06", 41, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        p2 = _write_main(d, "20261006", u2)
        t2 = pq.read_table(p2)
        pq.write_table(t2.cast(pa.schema(fields, metadata=t2.schema.metadata)), p2)
        OS.ingest(str(d), kind="main")
        assert pq.read_schema(p1).field("종목코드").type == pa.string()
        assert pq.read_schema(p2).field("종목코드").type == pa.large_string()
        for y, p in (("20261005", p1), ("20261006", p2)):
            _assert_exact(str(d), y, "main", p)
            # arrow 수준: 스키마(메타데이터 포함)까지 같다
            rebuilt = OS._snapshot_table(str(d / "ohlcv_store"), "main", y)
            assert rebuilt.schema.equals(pq.read_schema(p), check_metadata=True)

    def test_hydrate_writes_identical_frames_and_never_overwrites(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        u = _universe("2026-10-05")
        p1 = _write_main(d, "20261005", u)
        _write_lane(d, "20261005", u)
        OS.ingest(str(d))
        out = tmp_path / "hyd"
        w = OS.hydrate(str(d), str(out))
        assert sorted(os.path.basename(x) for x in w) == ["ohlcv_cache_20261005.parquet",
                                                          "quiet_lane_ohlcv_20261005.parquet"]
        pd.testing.assert_frame_equal(pd.read_parquet(out / "ohlcv_cache_20261005.parquet"),
                                      pd.read_parquet(p1), check_exact=True)
        # 같은 pyarrow 로 쓴 원본이면 바이트까지 같다 (pandas.to_parquet = pyarrow + snappy)
        assert (out / "ohlcv_cache_20261005.parquet").read_bytes() == Path(p1).read_bytes()
        assert OS.hydrate(str(d), str(out)) == []                      # 덮어쓰지 않는다
        assert len(OS.hydrate(str(d), str(out), overwrite=True)) == 2


class TestNamingAndWorkflow:
    def test_store_files_never_match_consumer_globs(self, tmp_path):
        d = tmp_path
        u = _universe("2026-10-05")
        _write_main(d, "20261005", u)
        _write_lane(d, "20261005", u)
        OS.ingest(str(d))
        store_files = [p for p in glob.glob(str(d / "ohlcv_store" / "**" / "*"), recursive=True)
                       if os.path.isfile(p)]
        assert len(store_files) == 4
        for pat in ("ohlcv_cache_*", "quiet_lane_ohlcv_*", "recommend_*", "ohlcv_cache_2*.parquet"):
            hits = glob.glob(str(d / pat)) + glob.glob(str(d / "**" / pat), recursive=True)
            assert not any("ohlcv_store" in h for h in hits), (pat, hits)
        assert OS.STORE_DIRNAME == "ohlcv_store"

    def test_workflow_shadow_step_and_commit(self):
        i_shadow = WF_SRC.index("python -m services.ohlcv_store shadow")
        i_commit = WF_SRC.index("- name: Commit & Push CSV")
        i_collect = WF_SRC.index("python collector.py")
        assert i_collect < i_shadow < i_commit
        step = WF_SRC[WF_SRC.rindex("- name:", 0, i_shadow):i_commit]
        assert "continue-on-error: true" in step
        assert "steps.collect.outputs.holiday != 'true'" in step
        assert "steps.guard.outputs.skip != 'true'" in step
        assert "--log data/ohlcv_store/verify_log.jsonl" in step
        commit = WF_SRC[i_commit:]
        assert "if [ -d data/ohlcv_store ]; then git add -f data/ohlcv_store" in commit
        assert "git add -f data/*.parquet" in commit                   # 날짜 파일은 그대로 커밋(그림자)

    def test_hydrate_not_wired_into_any_workflow(self):
        for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
            assert "ohlcv_store hydrate" not in wf.read_text(encoding="utf-8"), wf.name

    def test_dockerignore_excludes_store(self):
        txt = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        assert "data/ohlcv_store/" in [ln.strip() for ln in txt.splitlines()]


class TestCliAndShadow:
    def test_shadow_writes_log_line_and_summary(self, tmp_path, capsys):
        d = tmp_path
        u = _universe("2026-10-05")
        _write_main(d, "20261005", u)
        _write_lane(d, "20261005", u)
        u2 = {c: _bars("2026-10-06", 41, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        _write_main(d, "20261006", u2)
        _write_lane(d, "20261006", u2)
        log = d / "ohlcv_store" / "verify_log.jsonl"
        rc = OS.main(["shadow", "--data-dir", str(d), "--log", str(log)])
        assert rc == 0
        lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
        assert [x["kind"] for x in lines] == ["main", "lane"]
        for x in lines:
            for k in ("ymd", "kind", "ok", "rows", "n_new", "n_rev", "seconds"):
                assert k in x, k
            assert x["ok"] is True and x["ymd"] == "20261006" and x["ingested"] == 2
            assert x["also_verified"] == 1 and x["also_exact"] == 1
        out = capsys.readouterr().out
        assert "[v85 ohlcv_store] main: 대상 20261006" in out and "정확 복원" in out
        # 다음 밤 — 증분 1밤
        u3 = {c: _bars("2026-10-07", 42, base=1000 + 500 * i, seed=i) for i, c in enumerate(u)}
        _write_main(d, "20261007", u3)
        rc = OS.main(["shadow", "--data-dir", str(d), "--log", str(log)])
        last = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()][-2:]
        assert rc == 0
        assert last[0]["ok"] and last[0]["ingested"] == 1 and last[0]["n_new"] == 3
        assert last[0]["n_rev"] == 0
        # 레인은 '최신 레인 파일' 기준 — 레인 10/07 파일이 없으면 10/06 을 다시 확인할 뿐
        assert last[1]["ymd"] == "20261006" and last[1]["ok"] and last[1]["ingested"] == 0

    def test_shadow_reports_mismatch_without_raising(self, tmp_path, monkeypatch):
        d = tmp_path
        _write_main(d, "20261005", _universe("2026-10-05"))
        monkeypatch.setattr(OS, "_verify_one", lambda *a, **k: (False, "주입된 불일치"))
        r = OS.shadow(str(d), log_path=str(d / "log.jsonl"))
        assert not r["ok"]
        line = json.loads((d / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert line["ok"] is False and "주입된 불일치" in json.dumps(line, ensure_ascii=False)

    def test_cli_ingest_verify_stats(self, tmp_path, capsys):
        d = tmp_path
        _write_main(d, "20261005", _universe("2026-10-05"))
        assert OS.main(["ingest", "--data-dir", str(d), "--kind", "main"]) == 0
        assert OS.main(["verify", "--data-dir", str(d), "--kind", "main", "--deep"]) == 0
        assert OS.main(["stats", "--data-dir", str(d)]) == 0
        out = capsys.readouterr().out
        assert "verify main: 정확 1/1" in out
        with pytest.raises(ValueError):
            OS.ingest(str(d), ymd="newest")


# ── 실데이터 (data/ 가 없으면 skip) ─────────────────────────────────────────
_REAL_MAIN = ["20260226", "20260227", "20260302", "20260304", "20260520", "20260923"]
_REAL_LANE_N = 3


def _real_files(kind: str):
    pat = "ohlcv_cache_2*.parquet" if kind == "main" else "quiet_lane_ohlcv_2*.parquet"
    return sorted(p for p in DATA.glob(pat) if OS._DATED_RX[kind].match(p.name))


def _expected_new_rev(paths, vcols):
    """pandas 로 직접 계산한 기대값 — 마지막 파일의 (신규, 개정) 행 수."""
    prev = []
    for p in paths[:-1]:
        d = pd.read_parquet(p).reset_index()
        d = d.rename(columns={d.columns[0]: "date", "종목코드": "code"})
        prev.append(d[["code", "date"] + vcols])
    base = pd.concat(prev, ignore_index=True).drop_duplicates(["code", "date"], keep="last")
    last = pd.read_parquet(paths[-1]).reset_index()
    last = last.rename(columns={last.columns[0]: "date", "종목코드": "code"})[["code", "date"] + vcols]
    m = last.merge(base, on=["code", "date"], how="left", suffixes=("", "_o"), indicator=True)
    new = int(m["_merge"].eq("left_only").sum())
    both = m["_merge"].eq("both")
    diff = np.zeros(len(m), bool)
    for c in vcols:
        a = m[c].to_numpy(float); b = m[c + "_o"].to_numpy(float)
        diff |= ~((a == b) | (np.isnan(a) & np.isnan(b)))
    return new, int((both & diff).sum())


class TestRealData:
    def test_real_sample_nights_exact(self, tmp_path):
        main = {p.name[12:20]: p for p in _real_files("main")}
        if not main or not all(y in main for y in _REAL_MAIN):
            pytest.skip("실데이터 날짜 캐시 없음")
        nights = _REAL_MAIN + [max(main)]                              # 첫 밤·휴장 복제·부분·분할 전후·마지막
        mir = tmp_path / "mirror"
        mir.mkdir()
        for y in nights:
            (mir / main[y].name).symlink_to(main[y])
        st = tmp_path / "store"
        r = OS.ingest(str(mir), kind="main", store_dir=str(st))
        assert r["main"]["ingested"] == sorted(nights) and not r["main"]["failed"]
        v = OS.verify(str(mir), "all", kind="main", store_dir=str(st), deep=True)
        assert v["ok"] and v["main"]["exact"] == len(nights)
        # 휴장 복제 밤(20260302 = 0227 복제): 새 값이 없다
        n0302, r0302 = _expected_new_rev([main[y] for y in sorted(nights) if y <= "20260302"],
                                         ["시가", "고가", "저가", "종가", "거래량", "등락률"])
        meta = OS._read_m(str(st / "main" / "m" / "20260302.parquet"))[1]
        assert (meta["n_new"], meta["n_rev"]) == (n0302, r0302) == (0, 0)
        # 분할 재작성 밤(20260923): 신규·개정 수가 parquet 에서 직접 계산한 값과 같다
        paths = [main[y] for y in sorted(nights) if y <= "20260923"]
        exp_new, exp_rev = _expected_new_rev(paths, ["시가", "고가", "저가", "종가", "거래량", "등락률"])
        meta = OS._read_m(str(st / "main" / "m" / "20260923.parquet"))[1]
        assert (meta["n_new"], meta["n_rev"]) == (exp_new, exp_rev)
        assert exp_rev > 0
        # 084650 (5:1 병합) — 0520 과 0923 의 같은 날짜 종가가 다르면 0923 v 에 개정으로 있어야 한다
        a = pd.read_parquet(main["20260520"]).reset_index()
        b = pd.read_parquet(main["20260923"]).reset_index()
        a = a[a["종목코드"].astype(str) == "084650"]; b = b[b["종목코드"].astype(str) == "084650"]
        if len(a) and len(b):
            j = a.merge(b, on=a.columns[0], suffixes=("_a", "_b"))
            changed = int((j["종가_a"] != j["종가_b"]).sum())
            v923 = pd.read_parquet(st / "main" / "v" / "20260923.parquet")
            assert int((v923["rev"] & (v923["code"] == "084650")).sum()) >= changed > 0

    def test_real_lane_nights_exact(self, tmp_path):
        lane = _real_files("lane")
        if len(lane) < _REAL_LANE_N:
            pytest.skip("실데이터 레인 캐시 없음")
        pick = [lane[0], lane[len(lane) // 2], lane[-1]]
        mir = tmp_path / "mirror"
        mir.mkdir()
        for p in pick:
            (mir / p.name).symlink_to(p)
        st = tmp_path / "store"
        r = OS.ingest(str(mir), kind="lane", store_dir=str(st))
        assert len(r["lane"]["ingested"]) == 3 and not r["lane"]["failed"]
        v = OS.verify(str(mir), "all", kind="lane", store_dir=str(st), deep=True)
        assert v["ok"] and v["lane"]["exact"] == 3
        for p in pick:
            y = p.name[17:25]
            got = OS.snapshot(str(mir), y, "lane", store_dir=str(st))
            assert got.index.name == "날짜"
            pd.testing.assert_frame_equal(got, pd.read_parquet(p), check_exact=True)

    def test_real_latest_union_matches_files(self, tmp_path):
        """저장소의 latest-wins 합집합 = 같은 파일들을 pandas 로 이어 붙여 keep='last' 한 것."""
        main = _real_files("main")
        if len(main) < 3:
            pytest.skip("실데이터 날짜 캐시 없음")
        pick = [main[-3], main[-2], main[-1]]
        mir = tmp_path / "mirror"
        mir.mkdir()
        for p in pick:
            (mir / p.name).symlink_to(p)
        st = tmp_path / "store"
        OS.ingest(str(mir), kind="main", store_dir=str(st))
        u = OS.latest_union(str(mir), "main", cols=["시가", "고가", "저가", "종가"], store_dir=str(st))
        frames = []
        for p in pick:
            d = pd.read_parquet(p, columns=["종목코드", "시가", "고가", "저가", "종가"]).reset_index()
            frames.append(d.rename(columns={d.columns[0]: "Date"}))
        e = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(["종목코드", "Date"], keep="last")
             .sort_values(["종목코드", "Date"]).reset_index(drop=True))
        u = u.sort_values(["종목코드", "Date"]).reset_index(drop=True)
        assert len(u) == len(e)
        assert (u["종목코드"].astype(str).to_numpy() == e["종목코드"].astype(str).to_numpy()).all()
        assert (u["Date"].to_numpy() == pd.to_datetime(e["Date"]).astype("datetime64[ns]").to_numpy()).all()
        for c in ("시가", "고가", "저가", "종가"):
            assert (u[c].to_numpy() == e[c].to_numpy(float)).all(), c
