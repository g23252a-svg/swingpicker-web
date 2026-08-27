# -*- coding: utf-8 -*-
"""v71 — 시점별 유니버스 복원 + 시세 백필 도구 테스트."""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import universe_history as uh
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "backfill_universe_ohlcv",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "backfill_universe_ohlcv.py"))
bf = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(bf)


# ── 픽스처 ────────────────────────────────────────────────────
def _write_snap(d, ymd, codes, names=None):
    df = pd.DataFrame({uh.CODE_COL: [str(c).zfill(6) for c in codes],
                       uh.NAME_COL: [(names or {}).get(str(c).zfill(6), f"종목{c}") for c in codes]})
    df.to_csv(os.path.join(d, f"krx_codes_{ymd}.csv"), index=False)


def _bulk(d, ymds, codes):
    for y in ymds:
        _write_snap(d, y, codes)


@pytest.fixture
def uni(tmp_path):
    """정상 20스냅샷, 코드 1000개."""
    d = tmp_path / "data"; d.mkdir()
    codes = [f"{i:06d}" for i in range(1000)]
    ymds = [f"202601{i:02d}" for i in range(1, 21)]
    _bulk(str(d), ymds, codes)
    return str(d), codes, ymds


# ── 스냅샷 로딩 ───────────────────────────────────────────────
def test_loads_all_snapshots(uni):
    d, codes, ymds = uni
    s = uh.load_snapshots(d)
    assert len(s) == len(ymds)
    assert all(x.valid for x in s)
    assert s[0].codes == set(codes)


def test_codes_are_zero_padded(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    pd.DataFrame({uh.CODE_COL: ["5930", "660"], uh.NAME_COL: ["a", "b"]}).to_csv(
        d / "krx_codes_20260101.csv", index=False)
    s = uh.load_snapshots(str(d))
    assert s[0].codes == {"005930", "000660"}


def test_non_numeric_codes_dropped(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    pd.DataFrame({uh.CODE_COL: ["005930", "XXXXXX", ""], uh.NAME_COL: ["a", "b", "c"]}).to_csv(
        d / "krx_codes_20260101.csv", index=False)
    assert uh.load_snapshots(str(d))[0].codes == {"005930"}


def test_missing_code_column_marked_invalid(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    pd.DataFrame({"엉뚱한칸": ["005930"]}).to_csv(d / "krx_codes_20260101.csv", index=False)
    s = uh.load_snapshots(str(d))
    assert s[0].valid is False and s[0].reason == "missing_code_column"


def test_unreadable_file_recorded_not_skipped(tmp_path):
    """읽기 실패를 조용히 건너뛰면 '스냅샷이 없었다'와 구분되지 않는다."""
    d = tmp_path / "data"; d.mkdir()
    (d / "krx_codes_20260101.csv").write_bytes(b"\xff\xfe\x00bad")
    s = uh.load_snapshots(str(d))
    assert len(s) == 1 and s[0].valid is False


# ── 붕괴 스냅샷 방어 (핵심) ───────────────────────────────────
def test_degenerate_snapshot_is_dropped(uni):
    d, codes, ymds = uni
    _write_snap(d, "20260121", codes[:50])          # 5% 만 수집된 날
    s = uh.mark_degenerate(uh.load_snapshots(d))
    bad = [x for x in s if not x.valid]
    assert [x.ymd for x in bad] == ["20260121"]
    assert "degenerate" in bad[0].reason


def test_consecutive_outage_still_caught(uni):
    """롤링 중위였다면 못 잡던 케이스 — 연속 장애 구간."""
    d, codes, ymds = uni
    for y in ["20260121", "20260122", "20260123", "20260124", "20260125", "20260126"]:
        _write_snap(d, y, codes[:50])
    s = uh.mark_degenerate(uh.load_snapshots(d))
    bad = {x.ymd for x in s if not x.valid}
    assert bad == {"20260121", "20260122", "20260123", "20260124", "20260125", "20260126"}


def test_reference_size_converges_past_contamination(uni):
    d, codes, _ = uni
    for i in range(8):                               # 8/28 오염
        _write_snap(d, f"202602{i+1:02d}", codes[:40])
    ref = uh.reference_size([len(x.codes) for x in uh.load_snapshots(d)])
    assert ref == pytest.approx(1000, abs=1)


def test_iteration_needed_when_median_itself_is_dragged_down(tmp_path):
    """단순 전역 중위로는 못 잡는 배치 — 중간 크기 오염이 중위를 끌어내린다.

    100×8 · 450×4 · 1000×9 → 최초 중위 450, 문턱 225라 450짜리가 살아남는다.
    한 번 절사하면 중위가 1000으로 올라가 문턱 500이 되어 450짜리가 걸린다.
    """
    d = tmp_path / "data"; d.mkdir()
    i = 0
    for n, cnt in ((100, 8), (450, 4), (1000, 9)):
        for _ in range(cnt):
            i += 1
            _write_snap(str(d), f"2026{i:04d}"[:8].ljust(8, "0"),
                        [f"{j:06d}" for j in range(n)])
    snaps = uh.load_snapshots(str(d))
    sizes = [len(x.codes) for x in snaps]
    assert uh.reference_size(sizes, passes=0) == pytest.approx(450, abs=1)
    assert uh.reference_size(sizes) == pytest.approx(1000, abs=1)
    dropped = {len(x.codes) for x in uh.mark_degenerate(snaps) if not x.valid}
    assert dropped == {100, 450}


def test_healthy_universe_never_dropped(uni):
    d, _, _ = uni
    s = uh.mark_degenerate(uh.load_snapshots(d))
    assert all(x.valid for x in s)


def test_gradual_universe_shrink_not_flagged(tmp_path):
    """유니버스가 서서히 줄어드는 것은 붕괴가 아니다."""
    d = tmp_path / "data"; d.mkdir()
    for i in range(20):
        _write_snap(str(d), f"202601{i+1:02d}", [f"{j:06d}" for j in range(1000 - i * 10)])
    s = uh.mark_degenerate(uh.load_snapshots(str(d)))
    assert all(x.valid for x in s)


def test_alarm_logged_when_too_many_dropped(tmp_path, caplog):
    d = tmp_path / "data"; d.mkdir()
    for i in range(10):
        _write_snap(str(d), f"202601{i+1:02d}", [f"{j:06d}" for j in range(1000)])
    for i in range(6):
        _write_snap(str(d), f"202602{i+1:02d}", [f"{j:06d}" for j in range(10)])
    with caplog.at_level("WARNING"):
        uh.mark_degenerate(uh.load_snapshots(str(d)))
    assert any("붕괴로 버렸다" in r.getMessage() for r in caplog.records)


def test_degenerate_snapshot_does_not_create_fake_gap(uni):
    """붕괴일을 살려두면 전 종목이 그날 '사라진' 것으로 잡힌다."""
    d, codes, _ = uni
    _write_snap(d, "20260121", codes[:50])
    _bulk(d, ["20260122", "20260123"], codes)
    h = uh.build(d)
    assert (h["gap_max"] == 0).all()


# ── 소멸 판정 ─────────────────────────────────────────────────
def test_delisting_detected(uni):
    d, codes, _ = uni
    rest = codes[:-1]
    _bulk(d, ["20260121", "20260122", "20260123", "20260124"], rest)
    h = uh.build(d).set_index("종목코드")
    assert h.loc[codes[-1], "status"] == uh.STATUS_DELISTED
    assert h.loc[codes[-1], "delisted_ymd"] == "20260120"
    assert h.loc[codes[0], "status"] == uh.STATUS_LISTED


def test_absence_shorter_than_confirm_is_unconfirmed(uni):
    """확인 창이 모자라면 소멸이라고 단정하지 않는다."""
    d, codes, _ = uni
    _bulk(d, ["20260121", "20260122"], codes[:-1])   # 2개만 부재 (<CONFIRM_SNAPSHOTS=3)
    h = uh.build(d).set_index("종목코드")
    assert h.loc[codes[-1], "status"] == uh.STATUS_UNCONFIRMED
    assert h.loc[codes[-1], "delisted_ymd"] == ""


def test_confirm_boundary_exact(uni):
    d, codes, _ = uni
    _bulk(d, ["20260121", "20260122", "20260123"], codes[:-1])
    h = uh.build(d).set_index("종목코드")
    assert h.loc[codes[-1], "status"] == uh.STATUS_DELISTED


def test_still_present_on_last_snapshot_is_listed(uni):
    d, codes, _ = uni
    h = uh.build(d)
    assert (h["status"] == uh.STATUS_LISTED).all()


def test_reappearance_is_not_delisting(uni):
    d, codes, _ = uni
    _bulk(d, ["20260121", "20260122", "20260123", "20260124"], codes[:-1])
    _bulk(d, ["20260125", "20260126", "20260127"], codes)
    h = uh.build(d).set_index("종목코드")
    assert h.loc[codes[-1], "status"] == uh.STATUS_LISTED
    assert h.loc[codes[-1], "gap_max"] == 4


def test_new_listing_has_later_first_ymd(uni):
    d, codes, _ = uni
    newc = "999999"
    _bulk(d, ["20260121", "20260122"], codes + [newc])
    h = uh.build(d).set_index("종목코드")
    assert h.loc[newc, "first_ymd"] == "20260121"
    assert h.loc[codes[0], "first_ymd"] == "20260101"


def test_universe_is_union_not_last_snapshot(uni):
    """마지막 스냅샷만 쓰면 소멸 종목이 통째로 빠진다 — 생존편향의 정체."""
    d, codes, _ = uni
    _bulk(d, ["20260121", "20260122", "20260123", "20260124"], codes[:-5])
    h = uh.build(d)
    assert len(h) == len(codes)
    assert (h["status"] == uh.STATUS_DELISTED).sum() == 5


def test_empty_dir_returns_empty_frame(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    h = uh.build(str(d))
    assert h.empty and list(h.columns)[0] == "종목코드"


def test_all_degenerate_returns_empty(tmp_path):
    d = tmp_path / "data"; d.mkdir()
    # 전부 같은 크기면 기준 대비 미달이 없으므로 살아남아야 한다
    for i in range(5):
        _write_snap(str(d), f"202601{i+1:02d}", ["000001"])
    assert not uh.build(str(d)).empty


def test_membership_excludes_invalid(uni):
    d, codes, _ = uni
    _write_snap(d, "20260121", codes[:50])
    m = uh.membership(uh.mark_degenerate(uh.load_snapshots(d)))
    assert "20260121" not in m
    assert len(m) == 20


# ── 커버리지 / 결손 ───────────────────────────────────────────
def test_coverage_counts_missing(uni):
    d, codes, _ = uni
    px = pd.DataFrame({"종목코드": codes[:10], "종가": [1] * 10},
                      index=pd.to_datetime(["2026-01-01"] * 10))
    px.index.name = "Date"
    px.to_parquet(os.path.join(d, "ohlcv_cache_20260120.parquet"))
    c = uh.coverage(d)
    assert c["universe"] == 1000 and c["have_ohlcv"] == 10 and c["missing"] == 990


def test_coverage_counts_backfill_store(uni):
    d, codes, _ = uni
    s = os.path.join(d, "universe_ohlcv"); os.makedirs(s)
    for c in codes[:7]:
        pd.DataFrame({"종가": [1]}, index=pd.to_datetime(["2026-01-01"])).to_parquet(
            os.path.join(s, f"{c}.parquet"))
    assert uh.coverage(d)["have_ohlcv"] == 7


def test_missing_codes_excludes_present(uni):
    d, codes, _ = uni
    s = os.path.join(d, "universe_ohlcv"); os.makedirs(s)
    pd.DataFrame({"종가": [1]}, index=pd.to_datetime(["2026-01-01"])).to_parquet(
        os.path.join(s, f"{codes[0]}.parquet"))
    m = uh.missing_codes(d)
    assert codes[0] not in m and len(m) == 999


def test_delisted_ohlcv_gap_is_reported(uni):
    """소멸 종목의 시세가 없다는 사실 자체가 보고되어야 한다."""
    d, codes, _ = uni
    _bulk(d, ["20260121", "20260122", "20260123", "20260124"], codes[:-3])
    c = uh.coverage(d)
    assert c["delisted"] == 3 and c["delisted_with_ohlcv"] == 0


def test_summary_line_mentions_counts(uni):
    d, _, _ = uni
    s = uh.summary_line(d)
    assert "유니버스 1,000종목" in s and "붕괴" in s


def test_snapshot_report_shape(uni):
    d, _, _ = uni
    r = uh.snapshot_report(d)
    assert list(r.columns) == ["ymd", "n", "valid", "reason"] and len(r) == 20


# ── 백필 도구 ─────────────────────────────────────────────────
def _raw(rows=5, change_scale="pct"):
    idx = pd.date_range("2026-01-01", periods=rows, freq="D")
    ch = [0.0123] * rows if change_scale == "ratio" else [1.23] * rows
    return pd.DataFrame({"Open": [100] * rows, "High": [110] * rows, "Low": [90] * rows,
                         "Close": [105] * rows, "Volume": [1000] * rows, "Change": ch}, index=idx)


def _fake_fetch(rows=5, change_scale="pct"):
    unit = bf.UNIT_RATIO if change_scale == "ratio" else bf.UNIT_PERCENT

    def f(code, start, end):
        return _raw(rows, change_scale), unit
    return f


def test_normalize_matches_cache_schema():
    d = bf.normalize(_raw(), "1")
    assert list(d.columns) == bf.OUT_COLS
    assert d.index.name == "Date"
    assert d["종목코드"].iloc[0] == "000001"


def test_normalize_converts_ratio_change_to_percent():
    d = bf.normalize(_raw(change_scale="ratio"), "000001", change_unit=bf.UNIT_RATIO)
    assert d["등락률"].iloc[0] == pytest.approx(1.23, abs=1e-6)


def test_normalize_keeps_percent_change_as_is():
    """전부 +1.23%인 종목 — 값으로 단위를 추측하던 판본은 여기서 123%를 만들었다."""
    d = bf.normalize(_raw(change_scale="pct"), "000001", change_unit=bf.UNIT_PERCENT)
    assert d["등락률"].iloc[0] == pytest.approx(1.23, abs=1e-6)


def test_normalize_derive_ignores_source_change():
    """기본값은 출처의 등락률을 안 쓰고 종가에서 다시 계산한다."""
    raw = _raw(change_scale="ratio")
    raw.loc[raw.index[2], "Close"] = 210.0          # 105 → 210 = +100%
    d = bf.normalize(raw, "000001")
    assert pd.isna(d["등락률"].iloc[0])
    assert d["등락률"].iloc[2] == pytest.approx(100.0, abs=1e-6)


def test_normalize_rejects_unknown_unit():
    with pytest.raises(ValueError):
        bf.normalize(_raw(), "000001", change_unit="어쩌구")


def test_fetch_declares_unit_not_guesses():
    """단위는 값이 아니라 출처가 정한다 — 회귀 방지."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "backfill_universe_ohlcv.py"), encoding="utf-8").read()
    assert "abs().max()" not in src


def test_normalize_drops_zero_and_nan_close():
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    raw = pd.DataFrame({"Open": [1, 1, 1], "High": [1, 1, 1], "Low": [1, 1, 1],
                        "Close": [100, 0, None], "Volume": [1, 1, 1]}, index=idx)
    assert len(bf.normalize(raw, "000001")) == 1


def test_normalize_empty_returns_empty_with_columns():
    d = bf.normalize(pd.DataFrame(), "000001")
    assert d.empty and list(d.columns) == bf.OUT_COLS


def test_run_writes_parquet_and_manifest(uni):
    d, codes, _ = uni
    r = bf.run(d, limit=3, sleep=0, fetcher=_fake_fetch())
    assert r == dict(attempted=3, ok=3, empty=0, error=0)
    man = bf.load_manifest(d)
    assert len(man) == 3 and all(v["status"] == bf.ST_OK for v in man.values())
    assert os.path.exists(os.path.join(d, "universe_ohlcv", f"{codes[0]}.parquet"))


def test_run_is_resumable(uni):
    d, _, _ = uni
    bf.run(d, limit=3, sleep=0, fetcher=_fake_fetch())
    r = bf.run(d, limit=3, sleep=0, fetcher=_fake_fetch())
    assert r["attempted"] == 3
    assert len(bf.load_manifest(d)) == 6


def test_run_records_errors_without_stopping(uni):
    d, _, _ = uni
    calls = {"n": 0}

    def flaky(code, start, end):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return _fake_fetch()(code, start, end)

    r = bf.run(d, limit=3, sleep=0, fetcher=flaky)
    assert r["ok"] == 2 and r["error"] == 1
    man = bf.load_manifest(d)
    assert any(v["status"] == bf.ST_ERROR and "boom" in v["error"] for v in man.values())


def test_failed_codes_skipped_until_retry_flag(uni):
    d, _, _ = uni

    def broken(code, start, end):
        raise RuntimeError("nope")

    bf.run(d, limit=2, sleep=0, fetcher=broken)
    again = bf.run(d, limit=2, sleep=0, fetcher=broken)
    assert {m["code"] for m in bf.load_manifest(d).values()} != set()
    # 실패한 2종목을 건너뛰고 '다음' 2종목을 시도했어야 한다
    assert len(bf.load_manifest(d)) == 4
    retry = bf.run(d, limit=2, sleep=0, retry_failed=True, fetcher=_fake_fetch())
    assert retry["ok"] == 2


def test_empty_fetch_recorded_as_empty(uni):
    d, _, _ = uni
    r = bf.run(d, limit=2, sleep=0, fetcher=lambda c, s, e: (pd.DataFrame(), bf.UNIT_PERCENT))
    assert r["empty"] == 2 and r["ok"] == 0
    assert not os.path.exists(os.path.join(d, "universe_ohlcv", "000000.parquet"))


def test_dry_run_writes_nothing(uni):
    d, _, _ = uni
    r = bf.run(d, limit=5, sleep=0, dry_run=True, fetcher=_fake_fetch())
    assert r["dry_run"] is True and r["attempted"] == 5
    assert not os.path.exists(os.path.join(d, "universe_ohlcv"))


def test_manifest_survives_corruption(uni):
    d, _, _ = uni
    os.makedirs(os.path.join(d, "universe_ohlcv"), exist_ok=True)
    with open(bf.manifest_path(d), "w") as fh:
        fh.write("{not json")
    assert bf.load_manifest(d) == {}


def test_status_reports_remaining(uni):
    d, _, _ = uni
    bf.run(d, limit=4, sleep=0, fetcher=_fake_fetch())
    s = bf.status(d)
    assert "남은 작업" in s and "ok 4" in s


def test_backfilled_store_reduces_missing(uni):
    d, _, _ = uni
    before = uh.coverage(d)["missing"]
    bf.run(d, limit=6, sleep=0, fetcher=_fake_fetch())
    assert uh.coverage(d)["missing"] == before - 6


def test_not_wired_into_nightly_batch():
    """야간 배치 시간 폭증 방지 — pipeline_finalize가 이 도구를 부르면 안 된다."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "pipeline_finalize.py"), encoding="utf-8").read()
    assert "backfill_universe_ohlcv" not in src


# ══════════════════════════════════════════════════════════════════
#  data/ 오염 가드 (conftest) — 2026-08-27 실사고 회귀
# ══════════════════════════════════════════════════════════════════
class TestDataDirtyGuard:
    """`git status`에 data/ohlcv_union_hl.parquet이 수정된 채 남았던 사고.

    내용은 HEAD와 완전히 동일했다 — 데이터는 그대로인데 바이트만 다시 쓰였다.
    범인은 v65의 실데이터 회귀 테스트가 진짜 data/를 넘긴 것이었고,
    pick_reliability는 설계상 <data_dir>/ohlcv_union_hl.parquet에 캐싱한다.
    """

    @staticmethod
    def _conftest_src():
        return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conftest.py"),
                    encoding="utf-8").read()

    def test_guard_helper_is_not_a_second_hook(self):
        """같은 이름의 pytest 훅을 한 모듈에 두 번 정의하면 앞엣것이 사라진다.

        처음 판본이 pytest_runtest_teardown을 두 번 정의해 모듈 누출 가드를
        통째로 덮어썼다. 최상위 함수 이름에 중복이 없어야 한다.
        """
        import ast
        from collections import Counter
        tree = ast.parse(self._conftest_src())
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        dup = [k for k, v in Counter(names).items() if v > 1]
        assert dup == [], f"conftest 최상위 함수 중복 정의: {dup}"

    def test_module_leak_guard_still_present(self):
        """data/ 가드를 붙이면서 기존 모듈 누출 가드를 죽이지 않았는가."""
        src = self._conftest_src()
        assert "테스트 격리 위반" in src
        assert "_check_data_dirty(item)" in src

    def test_scan_detects_size_change(self, tmp_path, monkeypatch):
        import conftest as ct
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        f = tmp_path / "x.parquet"
        f.write_bytes(b"a")
        before = ct._scan_data()
        f.write_bytes(b"bb")
        assert ct._scan_data() != before

    def test_scan_detects_same_size_rewrite(self, tmp_path, monkeypatch):
        """실제 사고가 이 모양이었다 — 내용은 같고 바이트만 다시 쓰였다.

        크기만 보는 가드는 이걸 통과시킨다. mtime 을 함께 봐야 한다.
        """
        import time
        import conftest as ct
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        f = tmp_path / "x.parquet"
        f.write_bytes(b"aaaa")
        before = ct._scan_data()
        time.sleep(0.01)
        f.write_bytes(b"bbbb")                      # 같은 크기, 다른 내용
        after = ct._scan_data()
        assert after[f.name][1] == before[f.name][1], "크기는 같아야 하는 시나리오"
        assert after != before, "같은 크기 재작성을 놓쳤다 — mtime 을 봐야 한다"

    def test_scan_ignores_symlinks(self, tmp_path, monkeypatch):
        """미러의 심링크가 원본으로 오인되면 안 된다."""
        import conftest as ct
        real = tmp_path / "real"; real.mkdir()
        (real / "a.parquet").write_bytes(b"a")
        mirror = tmp_path / "mirror"; mirror.mkdir()
        (mirror / "a.parquet").symlink_to(real / "a.parquet")
        monkeypatch.setattr(ct, "_DATA_DIR", mirror)
        assert ct._scan_data() == {}

    def test_mirror_reads_pass_through(self, real_data_mirror):
        d = real_data_mirror("krx_codes_*.csv")
        got = sorted(os.listdir(d))
        assert got and all(n.startswith("krx_codes_") for n in got)
        assert os.path.islink(os.path.join(d, got[0]))

    def test_mirror_write_lands_in_tmp_not_repo(self, real_data_mirror):
        d = real_data_mirror("krx_codes_*.csv")
        out = os.path.join(d, "ohlcv_union_hl.parquet")
        pd.DataFrame({"a": [1]}).to_parquet(out)
        assert os.path.exists(out)
        repo_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        assert os.path.dirname(os.path.abspath(out)) != os.path.abspath(repo_data)

    def test_mirror_is_empty_without_patterns(self, real_data_mirror):
        assert os.listdir(real_data_mirror()) == []

    def test_v65_holiday_test_uses_mirror(self):
        """회귀 방지 — 이 테스트가 다시 진짜 data/ 를 넘기면 실패한다."""
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "test_v65_official_track_record.py"), encoding="utf-8").read()
        i = src.index("def test_real_data_skips_known_holidays")
        body = src[i:i + 800]
        assert "real_data_mirror(" in body
        assert "compute_pick_reliability(str(DATA))" not in body
