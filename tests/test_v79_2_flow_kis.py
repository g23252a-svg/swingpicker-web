# -*- coding: utf-8 -*-
"""v79.2 — KIS 종목별 투자자 API로 전종목 순매수 (KRX 차단 대응)."""
import os

import pandas as pd
import pytest

from services import investor_flow_kis as K


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status; self._b = body or {}
    def json(self):
        return self._b


class _Sess:
    """가짜 requests.Session — 호출 기록 + 종목별 응답."""
    def __init__(self, table, status=200):
        self.table = table; self.status = status; self.calls = []
    def get(self, url, headers=None, params=None, timeout=None):
        code = params["FID_INPUT_ISCD"]; self.calls.append((code, headers["tr_id"]))
        body = self.table.get(code)
        if body is None:
            return _Resp(500)
        return _Resp(self.status, body)


def _rows(days, frg_won, inst_won):
    return [{"stck_bsop_date": d, "stck_clpr": "10000",
             "frgn_ntby_tr_pbmn": str(frg_won), "orgn_ntby_tr_pbmn": str(inst_won)}
            for d in days]


def _out(days, frg_won, inst_won):
    """KIS 응답 봉투 — rt_cd 0 + output 리스트."""
    return {"rt_cd": "0", "output": _rows(days, frg_won, inst_won)}


class TestParse:
    def test_won_to_eok(self):
        rows, note = K.parse_rows(_rows(["20260903"], 250_000_000, -1_000_000))
        assert rows[0]["frg_eok"] == pytest.approx(2.5)
        assert rows[0]["inst_eok"] == pytest.approx(-0.01)
        assert "원" in note and "1e8" in note

    def test_qty_fallback_is_labelled(self):
        rows, note = K.parse_rows([{"stck_bsop_date": "20260903", "stck_clpr": "20000",
                                    "frgn_ntby_qty": "500", "orgn_ntby_qty": "-100"}])
        assert rows[0]["frg_eok"] == pytest.approx(500 * 20000 / 1e8)
        assert "근사" in note, "근사치는 출처가 말해야 한다 — 단위 추측 금지"

    def test_bad_dates_skipped(self):
        rows, _ = K.parse_rows([{"stck_bsop_date": "", "frgn_ntby_tr_pbmn": "1"}])
        assert rows == []


class TestCollect:
    def test_writes_one_file_per_day_all_tickers(self, tmp_path):
        days = ["20260901", "20260902", "20260903"]
        sess = _Sess({"000001": _out(days, 1e8, 2e8), "000002": _out(days, -3e8, 0)})
        s = K.collect_universe(sess, "tok", "k", "s", ["000001", "000002"], str(tmp_path), sleep_sec=0)
        assert s["ok"] == 2 and s["days_written"] == days
        d = pd.read_parquet(tmp_path / "flow_full_20260902.parquet")
        assert set(d["종목코드"]) == {"000001", "000002"}
        assert d.set_index("종목코드").loc["000002", "frg_eok"] == pytest.approx(-3.0)
        assert list(d.columns) == ["종목코드", "frg_eok", "inst_eok"], "v79 규격 유지 — winner_profile 무변경"

    def test_uses_investor_tr_id_and_token(self, tmp_path):
        sess = _Sess({"000001": _out(["20260903"], 1, 1)})
        K.collect_universe(sess, "tok", "k", "s", ["000001"], str(tmp_path), sleep_sec=0)
        assert sess.calls == [("000001", K.TR_ID)]

    def test_partial_failure_keeps_going(self, tmp_path):
        sess = _Sess({"000001": _out(["20260903"], 1e8, 0)})      # 000002는 500
        s = K.collect_universe(sess, "tok", "k", "s", ["000002", "000001"], str(tmp_path), sleep_sec=0)
        assert s["ok"] == 1 and s["fail"] == 1
        assert os.path.exists(tmp_path / "flow_full_20260903.parquet")

    def test_abort_when_everything_fails_early(self, tmp_path):
        codes = [f"{i:06d}" for i in range(100)]
        sess = _Sess({})                                            # 전부 500
        s = K.collect_universe(sess, "tok", "k", "s", codes, str(tmp_path), sleep_sec=0, abort_after=20)
        assert len(sess.calls) == 20, "자격/차단이면 나머지 80종목을 두드리면 안 된다"
        assert s["days_written"] == []

    def test_rt_cd_error_counts_as_fail(self, tmp_path):
        sess = _Sess({"000001": {"rt_cd": "1", "msg1": "만료"}})
        # _Sess는 body를 그대로 주므로 rt_cd!=0 경로를 탄다
        s = K.collect_universe(sess, "tok", "k", "s", ["000001"], str(tmp_path), sleep_sec=0)
        assert s["fail"] == 1

    def test_overwrites_with_latest(self, tmp_path):
        """당일 잠정치는 다음 실행의 확정치로 덮인다."""
        sess1 = _Sess({"000001": _out(["20260903"], 1e8, 0)})
        K.collect_universe(sess1, "tok", "k", "s", ["000001"], str(tmp_path), sleep_sec=0)
        sess2 = _Sess({"000001": _out(["20260903"], 5e8, 0)})
        K.collect_universe(sess2, "tok", "k", "s", ["000001"], str(tmp_path), sleep_sec=0)
        d = pd.read_parquet(tmp_path / "flow_full_20260903.parquet")
        assert d["frg_eok"].iloc[0] == pytest.approx(5.0)


class TestUniverse:
    def test_codes_from_both_caches(self, tmp_path):
        pd.DataFrame({"종목코드": ["1", "000002"], "종가": [1, 2]}).to_parquet(tmp_path / "ohlcv_cache_20260903.parquet")
        pd.DataFrame({"종목코드": ["000003"], "종가": [3]}).to_parquet(tmp_path / "quiet_lane_ohlcv_20260903.parquet")
        pd.DataFrame({"종목코드": ["999999"], "종가": [9]}).to_parquet(tmp_path / "ohlcv_cache_latest.parquet")
        assert K.universe_codes(str(tmp_path)) == ["000001", "000002", "000003"]

    def test_empty_when_no_cache(self, tmp_path):
        assert K.universe_codes(str(tmp_path)) == []


class TestWinnerProfileIntegration:
    def test_flow_file_is_readable_by_winner_profile(self, tmp_path):
        from services import winner_profile as W
        sess = _Sess({"000001": _out(["20260903"], 2e8, -1e8)})
        K.collect_universe(sess, "tok", "k", "s", ["000001"], str(tmp_path), sleep_sec=0)
        f = W._load_flow(str(tmp_path), "20260903")
        assert f is not None and f.iloc[0]["frg_eok"] == pytest.approx(2.0)


class TestWiring:
    def test_prefetch_reuses_token_and_calls_collector(self):
        src = open("prefetch_flow.py", encoding="utf-8").read()
        assert "def fetch_flow(ymd, token=None)" in src
        assert "_collect_full_universe(_tok)" in src
        assert src.count("_kis_get_token(") >= 2      # 정의 1 + main에서 1회 발급
    def test_workflow_commits_parquet_and_installs_pyarrow(self):
        y = open(".github/workflows/prefetch_flow.yml", encoding="utf-8").read()
        assert "data/flow_full_*.parquet" in y and "pyarrow" in y
