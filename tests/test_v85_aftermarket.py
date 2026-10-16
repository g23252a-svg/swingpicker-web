# -*- coding: utf-8 -*-
"""[v85] 시간외 단일가 수집 수정 + 날짜별 보관 (naver_aftermarket).

■ 고친 결함 (v20.6.4 sidecar)
  (a) final != 분석종가 인 행만 기록 → ~72% 가 조용히 빠졌다 → 이제 전 종목 + AH_STATUS
  (b) overPrice 없으면 closePrice 를 시간외종가로 복사 → 이제 NaN
  (c) price_snapshot_latest.csv '종가'를 시간외가로 덮어씀 → 이제 건드리지 않음
  (d) 분석종가(Naver 차트) 기준 변동률만 → KRX_CLOSE(price_snapshot_{ymd}) 기준 별도
  (e) latest 만 → aftermarket_prices_{ymd}.csv 날짜별 보관
  확정: 거래일 20:00 KST(NXT 애프터마켓 종료) 이후 ~ 다음 평일 08:00 전만 AH_FINAL.

Naver 는 샌드박스에서 막혀 있어(프록시 403) 응답은 전부 모의다.
"""
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import naver_aftermarket as NA  # noqa: E402

KST = timezone(timedelta(hours=9))
DATA = ROOT / "data"


def _kst(y, mo, d, h, mi=0, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=KST)


def _payload(close="61,000", over_price=None, over=True, **over_extra):
    """Naver basic 응답 모양(관측 불가 — 알려진 필드명 기준)."""
    d = {
        "itemCode": "005930", "stockName": "삼성전자",
        "closePrice": close, "marketStatus": "CLOSE",
        "localTradedAt": "2026-10-16T15:30:00+09:00",
        "imageCharts": {"day": "https://example/x.png"},
    }
    if over:
        o = {
            "tradingSessionType": "AFTER_MARKET",
            "overMarketStatus": "CLOSE",
            "localTradedAt": "2026-10-16T18:00:00+09:00",
            "fluctuationsRatio": "0.82",
            "compareToPreviousClosePrice": "500",
            "compareToPreviousPrice": {"code": "2", "text": "상승", "name": "RISING"},
        }
        if over_price is not None:
            o["overPrice"] = over_price
        o.update(over_extra)
        d["overMarketPriceInfo"] = o
    return d


class _Resp:
    def __init__(self, status_code=200, payload=None, raw=None):
        self.status_code = status_code
        self._payload = payload
        self._raw = raw

    def json(self):
        if self._raw is not None:
            return json.loads(self._raw)  # 깨진 JSON → json.JSONDecodeError(ValueError)
        return self._payload


class _Session:
    """requests.Session 대역 — 코드별 응답 또는 예외."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def get(self, url, timeout=None, headers=None):
        self.calls.append((url, timeout))
        code = re.search(r"/stock/(\w+)/basic", url).group(1)
        v = self.table[code]
        if isinstance(v, BaseException):
            raise v
        return v


# ─────────────────────────────────────────────────────────────
#  파싱 — 원값 보존, closePrice 폴백 없음
# ─────────────────────────────────────────────────────────────

class TestParse:
    def test_over_price_present_is_ok_with_raw_fields(self):
        r = NA.parse_basic_payload(_payload(close="61,000", over_price="61,500"))
        assert r["status"] == "OK"
        assert r["ah_price_raw"] == 61500.0
        assert r["naver_close"] == 61000.0
        assert r["ah_session_type"] == "AFTER_MARKET"
        assert r["ah_market_status"] == "CLOSE"
        assert r["ah_traded_at"] == "2026-10-16T18:00:00+09:00"
        assert r["naver_traded_at"] == "2026-10-16T15:30:00+09:00"
        assert r["naver_market_status"] == "CLOSE"
        assert r["ah_fluct_raw"] == "0.82"
        assert r["extra"] == "", "알려진 스키마면 AH_EXTRA 는 비어야 한다"

    def test_over_price_zero_is_no_print_not_close(self):
        r = NA.parse_basic_payload(_payload(close="61,000", over_price="0"))
        assert r["status"] == "NO_PRINT"
        assert r["ah_price_raw"] == 0.0
        assert r["naver_close"] == 61000.0

    def test_over_block_absent_is_no_print(self):
        r = NA.parse_basic_payload(_payload(close="61,000", over=False))
        assert r["status"] == "NO_PRINT"
        assert math.isnan(r["ah_price_raw"])
        assert r["naver_close"] == 61000.0
        assert r["ah_session_type"] == ""

    def test_over_price_key_missing_or_garbage_is_no_print(self):
        for op in (None, "", "-", "N/A"):
            r = NA.parse_basic_payload(_payload(over_price=op))
            assert r["status"] == "NO_PRINT", op
            assert math.isnan(r["ah_price_raw"]), op

    def test_numeric_types_accepted(self):
        r = NA.parse_basic_payload({"closePrice": 61000, "overMarketPriceInfo": {"overPrice": 61500}})
        assert r["status"] == "OK" and r["ah_price_raw"] == 61500.0

    def test_error_json_is_fetch_fail_schema(self):
        r = NA.parse_basic_payload({"code": "NOT_FOUND", "message": "x"})
        assert r["status"] == "FETCH_FAIL" and r["fail_reason"] == "SCHEMA"
        assert NA.parse_basic_payload(["not", "a", "dict"])["status"] == "FETCH_FAIL"
        assert NA.parse_basic_payload(None)["status"] == "FETCH_FAIL"

    def test_unknown_fields_captured_in_extra(self):
        d = _payload(over_price="61,500", venue="NXT")
        d["nxtOverMarketPriceInfo"] = {"overPrice": "61,700", "tradingSessionType": "AFTER_MARKET",
                                       "nested": {"x": 1}}
        r = NA.parse_basic_payload(d)
        extra = json.loads(r["extra"])
        assert extra["over.venue"] == "NXT"
        assert extra["nxtOverMarketPriceInfo.overPrice"] == "61,700"
        assert "imageCharts" not in r["extra"]
        assert r["status"] == "OK" and r["ah_price_raw"] == 61500.0


# ─────────────────────────────────────────────────────────────
#  HTTP — 네트워크/JSON/HTTP 오류는 예외가 아니라 FETCH_FAIL
# ─────────────────────────────────────────────────────────────

class TestFetch:
    def test_fetch_ok_uses_timeout_and_endpoint(self):
        s = _Session({"005930": _Resp(200, _payload(over_price="61,500"))})
        r = NA.fetch_after_market_price("005930", session=s)
        assert r["status"] == "OK" and r["ah_price_raw"] == 61500.0
        url, timeout = s.calls[0]
        assert url == "https://m.stock.naver.com/api/stock/005930/basic"
        assert timeout == NA.REQUEST_TIMEOUT and timeout is not None

    def test_malformed_json(self):
        s = _Session({"005930": _Resp(200, raw="<html>blocked</html>")})
        r = NA.fetch_after_market_price("005930", session=s)
        assert r["status"] == "FETCH_FAIL" and r["fail_reason"] == "JSON"

    def test_requests_jsondecodeerror_is_caught(self):
        resp = requests.models.Response()
        resp.status_code = 200
        resp._content = b"{not json"
        r = NA.fetch_after_market_price("005930", session=_Session({"005930": resp}))
        assert r["status"] == "FETCH_FAIL" and r["fail_reason"] == "JSON"

    def test_http_error(self):
        r = NA.fetch_after_market_price("005930", session=_Session({"005930": _Resp(404)}))
        assert r["status"] == "FETCH_FAIL" and r["fail_reason"] == "HTTP_404"

    @pytest.mark.parametrize("exc,reason", [
        (requests.exceptions.ConnectionError("down"), "NETWORK"),
        (requests.exceptions.ReadTimeout("slow"), "TIMEOUT"),
        (requests.exceptions.ConnectTimeout("slow"), "TIMEOUT"),
        (requests.exceptions.ProxyError("403"), "NETWORK"),
    ])
    def test_network_errors(self, exc, reason):
        r = NA.fetch_after_market_price("005930", session=_Session({"005930": exc}))
        assert r["status"] == "FETCH_FAIL" and r["fail_reason"] == reason
        assert math.isnan(r["ah_price_raw"]) and math.isnan(r["naver_close"])


# ─────────────────────────────────────────────────────────────
#  확정 창 — KRX 단일가 16~18시, NXT 애프터마켓 ~20시
# ─────────────────────────────────────────────────────────────

class TestFinality:
    # 20261016 = 금요일
    @pytest.mark.parametrize("ts,final,window", [
        (_kst(2026, 10, 15, 22, 0), False, "BEFORE_SESSION"),
        (_kst(2026, 10, 16, 7, 59), False, "BEFORE_SESSION"),
        (_kst(2026, 10, 16, 10, 0), False, "INTRADAY"),
        (_kst(2026, 10, 16, 15, 29), False, "INTRADAY"),
        (_kst(2026, 10, 16, 16, 30), False, "AH_SESSION"),
        (_kst(2026, 10, 16, 17, 0), False, "AH_SESSION"),   # KRX 단일가 진행 중
        (_kst(2026, 10, 16, 18, 30), False, "AH_SESSION"),  # NXT 애프터마켓 진행 중
        (_kst(2026, 10, 16, 19, 59, 59), False, "AH_SESSION"),
        (_kst(2026, 10, 16, 20, 0), True, "FINAL_EVENING"),
        (_kst(2026, 10, 16, 23, 59), True, "FINAL_EVENING"),
        (_kst(2026, 10, 17, 0, 30), True, "FINAL_OVERNIGHT"),   # 토 새벽
        (_kst(2026, 10, 17, 10, 0), True, "FINAL_OVERNIGHT"),   # 토 낮 — 다음 세션은 월
        (_kst(2026, 10, 19, 7, 59), True, "FINAL_OVERNIGHT"),   # 월 08시 전
        (_kst(2026, 10, 19, 8, 0), False, "NEXT_SESSION"),      # 월 NXT 프리마켓 시작
        (_kst(2026, 10, 20, 3, 0), False, "NEXT_SESSION"),
    ])
    def test_windows_friday(self, ts, final, window):
        assert NA.aftermarket_finality("20261016", ts) == (final, window)

    def test_weekday_next_morning(self):
        # 20261014 = 수요일 → 목 07:30 확정, 목 08:30 미확정
        assert NA.aftermarket_finality("20261014", _kst(2026, 10, 15, 5, 30)) == (True, "FINAL_OVERNIGHT")
        assert NA.aftermarket_finality("20261014", _kst(2026, 10, 15, 8, 30)) == (False, "NEXT_SESSION")

    def test_utc_and_naive_inputs(self):
        utc = datetime(2026, 10, 16, 11, 30, tzinfo=timezone.utc)   # = 20:30 KST
        assert NA.aftermarket_finality("20261016", utc) == (True, "FINAL_EVENING")
        utc2 = datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc)    # = 18:00 KST
        assert NA.aftermarket_finality("20261016", utc2) == (False, "AH_SESSION")
        # naive 는 KST 로 본다
        assert NA.aftermarket_finality("20261016", datetime(2026, 10, 16, 21, 0)) == (True, "FINAL_EVENING")
        assert NA.aftermarket_finality("20261016", "2026-10-16 17:00:00") == (False, "AH_SESSION")


# ─────────────────────────────────────────────────────────────
#  보관 — 전 종목 기록, KRX 기준, 스냅샷 불변, 날짜별 파일
# ─────────────────────────────────────────────────────────────

YMD = "20261016"


def _write_env(d: Path, with_dated_snapshot=True):
    rec = pd.DataFrame({
        "LDY_RANK": ["1", "2", "3", "4", "5"],
        "종목코드": ["002710", "005930", "000660", "035720", "123456"],
        "종목명": ["TCC스틸", "삼성전자", "SK하이닉스", "카카오", "가짜"],
        "종가": ["13270", "209000", "180000", "40000", "1000"],   # 분석종가 (Naver 차트)
        "기준일": [YMD] * 5,
        "ROUTE": ["WAIT"] * 5,
    })
    rec_path = d / "recommend_latest.csv"
    rec.to_csv(rec_path, index=False, encoding="utf-8-sig")
    snap = pd.DataFrame({
        "종목코드": ["002710", "005930", "000660", "035720", "999999"],
        "종목명": ["TCC스틸", "삼성전자", "SK하이닉스", "카카오", "기타"],
        "시장": ["KOSPI", "KOSPI", "KOSPI", "KOSPI", "KONEX"],
        "종가": [12940, 209000, 181000, 40000, 500],                # KRX 종가
        "시가": [12800, 208000, 180000, 39900, 500],
    })
    if with_dated_snapshot:
        snap.to_csv(d / f"price_snapshot_{YMD}.csv", index=False, encoding="utf-8-sig")
    # latest 는 다른 날 값 — 폴백되면 안 된다 (그리고 바이트 불변이어야 한다)
    latest = snap.copy()
    latest["종가"] = [1, 1, 1, 1, 1]
    latest.to_csv(d / "price_snapshot_latest.csv", index=False, encoding="utf-8-sig")
    return rec_path


def _fetch_table():
    return {
        # OK: 시간외가 분석종가와 다름
        "002710": {"status": "OK", "naver_close": 12940, "ah_price_raw": 13250,
                   "ah_session_type": "AFTER_MARKET", "ah_traded_at": "2026-10-16T18:00:00+09:00"},
        # OK: 시간외가 == 분석종가 → v20.6.4 는 이 행을 버렸다
        "005930": {"status": "OK", "naver_close": 209000, "ah_price_raw": 209000},
        # NO_PRINT: overPrice 없음 → closePrice 가 시간외종가로 들어가면 안 된다
        "000660": {"status": "NO_PRINT", "naver_close": 181000, "ah_price_raw": float("nan")},
        # NO_PRINT: overPrice 0
        "035720": {"status": "NO_PRINT", "naver_close": 40000, "ah_price_raw": 0},
        # FETCH_FAIL: 깨진 JSON
        "123456": {"status": "FETCH_FAIL", "fail_reason": "JSON"},
    }


def _run(d, rec_path, now=_kst(2026, 10, 16, 21, 5), **kw):
    table = _fetch_table()
    kw.setdefault("fetch_fn", lambda c: dict(table[c]))
    return NA.build_aftermarket_archive(
        str(rec_path), YMD, str(d), now_fn=lambda: now, sleep_fn=lambda s: None, **kw)


def _read(p):
    return pd.read_csv(p, dtype={"종목코드": str, "TRADE_YMD": str}, encoding="utf-8-sig")


class TestArchive:
    def test_all_rows_written_with_status(self, tmp_path):
        rec_path = _write_env(tmp_path)
        s = _run(tmp_path, rec_path)
        assert s["error"] is None
        out = _read(tmp_path / f"aftermarket_prices_{YMD}.csv")
        assert out["종목코드"].tolist() == ["002710", "005930", "000660", "035720", "123456"]
        assert out["AH_STATUS"].tolist() == ["OK", "OK", "NO_PRINT", "NO_PRINT", "FETCH_FAIL"]
        assert s["counts"] == {"OK": 2, "NO_PRINT": 2, "FETCH_FAIL": 1}
        assert s["n_rows"] == 5 and s["n_ok"] == 2
        assert set(out["AH_STATUS"]) <= {"OK", "NO_PRINT", "FETCH_FAIL"}
        assert list(out.columns[:5]) == ["종목코드", "종목명", "분석종가", "시간외종가", "변동률"]
        assert list(out.columns) == NA.OUTPUT_COLUMNS
        assert (out["TRADE_YMD"] == YMD).all() and (out["SCHEMA_VER"] == "v85").all()
        assert out.loc[4, "AH_FAIL_REASON"] == "JSON"

    def test_no_print_never_copies_close(self, tmp_path):
        rec_path = _write_env(tmp_path)
        _run(tmp_path, rec_path)
        out = _read(tmp_path / f"aftermarket_prices_{YMD}.csv").set_index("종목코드")
        for c in ("000660", "035720", "123456"):
            assert pd.isna(out.loc[c, "시간외종가"]), c
            assert pd.isna(out.loc[c, "변동률"]), c
            assert pd.isna(out.loc[c, "AH_CHG_VS_KRX"]), c
            assert pd.isna(out.loc[c, "AH_CHG_VS_ANALYSIS"]), c
        assert out.loc["000660", "NAVER_CLOSE"] == 181000
        assert pd.isna(out.loc["000660", "AH_PRICE_RAW"])
        assert out.loc["035720", "AH_PRICE_RAW"] == 0

    def test_krx_baseline_and_analysis_baseline_separate(self, tmp_path):
        rec_path = _write_env(tmp_path)
        _run(tmp_path, rec_path)
        out = _read(tmp_path / f"aftermarket_prices_{YMD}.csv").set_index("종목코드")
        r = out.loc["002710"]
        assert r["분석종가"] == 13270 and r["KRX_CLOSE"] == 12940 and r["시간외종가"] == 13250
        assert r["AH_CHG_VS_KRX"] == pytest.approx(13250 / 12940 - 1, abs=1e-6)
        assert r["AH_CHG_VS_ANALYSIS"] == pytest.approx(13250 / 13270 - 1, abs=1e-6)
        # 변동률 = 분석종가 기준 % (v20.6.4 정의 유지)
        assert r["변동률"] == round((13250 / 13270 - 1) * 100, 2)
        assert r["KRX_SNAP_SRC"] == "FDR"            # KONEX 가 섞인 스냅샷
        # 교차확인 NAVER_CLOSE vs KRX_CLOSE
        assert bool(r["CLOSE_MATCH"]) is True
        assert out.loc["000660", "KRX_CLOSE"] == 181000
        assert pd.isna(out.loc["123456", "KRX_CLOSE"])  # 스냅샷에 없는 종목
        assert pd.isna(out.loc["123456", "CLOSE_MATCH"])

    def test_close_match_false_when_differs(self, tmp_path):
        rec_path = _write_env(tmp_path)
        table = _fetch_table()
        table["002710"]["naver_close"] = 13270
        _run(tmp_path, rec_path, fetch_fn=lambda c: dict(table[c]))
        out = _read(tmp_path / f"aftermarket_prices_{YMD}.csv").set_index("종목코드")
        assert str(out.loc["002710", "CLOSE_MATCH"]) == "False"

    def test_dated_and_latest_identical_and_atomic(self, tmp_path):
        rec_path = _write_env(tmp_path)
        s = _run(tmp_path, rec_path)
        dated = tmp_path / f"aftermarket_prices_{YMD}.csv"
        latest = tmp_path / "aftermarket_prices_latest.csv"
        assert s["dated_path"] == str(dated) and s["latest_path"] == str(latest)
        assert dated.read_bytes() == latest.read_bytes()
        assert not list(tmp_path.glob(".aftermarket_*.tmp")), "임시파일이 남으면 안 된다"
        # 가격은 '209000.0' 이 아니라 정수로 써진다
        assert "209000.0" not in dated.read_text(encoding="utf-8-sig")

    def test_tag_goes_into_dated_name(self, tmp_path):
        rec_path = _write_env(tmp_path)
        s = _run(tmp_path, rec_path, tag="exp")
        assert s["dated_path"].endswith(f"aftermarket_prices_{YMD}_exp.csv")

    def test_recommend_and_snapshot_latest_untouched(self, tmp_path):
        rec_path = _write_env(tmp_path)
        snl = tmp_path / "price_snapshot_latest.csv"
        snd = tmp_path / f"price_snapshot_{YMD}.csv"
        before = (rec_path.read_bytes(), snl.read_bytes(), snd.read_bytes())
        _run(tmp_path, rec_path)
        NA.fetch_after_market_prices_sidecar(
            str(rec_path), str(tmp_path / "aftermarket_prices_latest.csv"), str(snl))
        assert (rec_path.read_bytes(), snl.read_bytes(), snd.read_bytes()) == before

    def test_missing_dated_snapshot(self, tmp_path):
        rec_path = _write_env(tmp_path, with_dated_snapshot=False)
        s = _run(tmp_path, rec_path)
        assert s["error"] is None and s["krx_snap_src"] == "MISSING"
        out = _read(tmp_path / f"aftermarket_prices_{YMD}.csv").set_index("종목코드")
        assert out["KRX_CLOSE"].isna().all(), "latest 스냅샷으로 폴백하면 안 된다"
        assert out["AH_CHG_VS_KRX"].isna().all()
        assert out["CLOSE_MATCH"].isna().all()
        assert (out["KRX_SNAP_SRC"] == "MISSING").all()
        # 기준가가 없어도 원값과 분석종가 기준은 남는다
        assert out.loc["002710", "시간외종가"] == 13250
        assert out.loc["002710", "AH_STATUS"] == "OK"
        assert out.loc["002710", "AH_CHG_VS_ANALYSIS"] == pytest.approx(13250 / 13270 - 1, abs=1e-6)

    @pytest.mark.parametrize("now,final,window", [
        (_kst(2026, 10, 16, 17, 0), False, "AH_SESSION"),
        (_kst(2026, 10, 16, 19, 30), False, "AH_SESSION"),
        (_kst(2026, 10, 16, 21, 5), True, "FINAL_EVENING"),
        (_kst(2026, 10, 17, 5, 40), True, "FINAL_OVERNIGHT"),
    ])
    def test_final_flag_per_row(self, tmp_path, now, final, window):
        rec_path = _write_env(tmp_path)
        s = _run(tmp_path, rec_path, now=now)
        out = _read(tmp_path / f"aftermarket_prices_{YMD}.csv").set_index("종목코드")
        assert (out["FETCH_WINDOW"] == window).all()
        assert (out["FETCHED_AT_KST"] == now.strftime("%Y-%m-%d %H:%M:%S")).all()
        ok_or_np = out["AH_STATUS"] != "FETCH_FAIL"
        assert (out.loc[ok_or_np, "AH_FINAL"] == final).all()
        assert not out.loc["123456", "AH_FINAL"], "FETCH_FAIL 은 확정이 아니다"
        # 미확정이어도 가격 원값은 지우지 않는다 (플래그로만 표시)
        assert out.loc["002710", "시간외종가"] == 13250
        assert s["n_final"] == (4 if final else 0)

    def test_fetch_exception_nonfatal(self, tmp_path):
        rec_path = _write_env(tmp_path)

        def boom(code):
            raise requests.exceptions.ConnectionError("naver down")
        s = _run(tmp_path, rec_path, fetch_fn=boom)
        assert s["error"] is None and s["counts"]["FETCH_FAIL"] == 5
        out = _read(tmp_path / "aftermarket_prices_latest.csv")
        assert len(out) == 5 and out["시간외종가"].isna().all()
        assert set(out["AH_FAIL_REASON"]) == {"NETWORK"}

    def test_circuit_breaker_stops_fetching(self, tmp_path):
        rec_path = _write_env(tmp_path)
        calls = []

        def fail(code):
            calls.append(code)
            return {"status": "FETCH_FAIL", "fail_reason": "TIMEOUT"}
        s = _run(tmp_path, rec_path, fetch_fn=fail, max_consec_fail=2)
        assert len(calls) == 2
        out = _read(tmp_path / "aftermarket_prices_latest.csv")
        assert len(out) == 5, "중단돼도 모든 행은 기록된다"
        assert out["AH_FAIL_REASON"].tolist() == ["TIMEOUT", "TIMEOUT", "CIRCUIT", "CIRCUIT", "CIRCUIT"]
        assert out["FETCH_WINDOW"].tolist()[2:] == ["NOT_FETCHED"] * 3
        assert s["counts"]["FETCH_FAIL"] == 5

    def test_per_code_404_does_not_trip_circuit(self, tmp_path):
        rec_path = _write_env(tmp_path)
        s = _run(tmp_path, rec_path, max_consec_fail=1,
                 fetch_fn=lambda c: {"status": "FETCH_FAIL", "fail_reason": "HTTP_404"})
        out = _read(tmp_path / "aftermarket_prices_latest.csv")
        assert set(out["AH_FAIL_REASON"]) == {"HTTP_404"}

    def test_time_budget(self, tmp_path):
        rec_path = _write_env(tmp_path)
        clock = iter([0.0, 0.0, 1.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0])
        s = _run(tmp_path, rec_path, monotonic_fn=lambda: next(clock), time_budget_sec=10)
        out = _read(tmp_path / "aftermarket_prices_latest.csv")
        assert len(out) == 5
        assert out["AH_STATUS"].tolist()[:2] == ["OK", "OK"]
        assert out["AH_FAIL_REASON"].tolist()[2:] == ["BUDGET"] * 3

    def test_pacing_kept(self, tmp_path):
        rec = pd.DataFrame({"종목코드": [f"{i:06d}" for i in range(1, 46)], "종가": ["1000"] * 45,
                            "기준일": [YMD] * 45})
        p = tmp_path / "recommend_latest.csv"
        rec.to_csv(p, index=False, encoding="utf-8-sig")
        sleeps = []
        NA.build_aftermarket_archive(
            str(p), YMD, str(tmp_path), now_fn=lambda: _kst(2026, 10, 16, 21),
            sleep_fn=sleeps.append,
            fetch_fn=lambda c: {"status": "NO_PRINT", "naver_close": 1000})
        assert sleeps == [NA.PACE_SLEEP_SEC, NA.PACE_SLEEP_SEC]  # 20, 40번째

    def test_bad_fetch_shapes_become_fetch_fail(self, tmp_path):
        rec_path = _write_env(tmp_path)
        shapes = {"002710": {}, "005930": None, "000660": {"status": "WEIRD"},
                  "035720": {"status": "OK", "ah_price_raw": 0}, "123456": "x"}
        _run(tmp_path, rec_path, fetch_fn=lambda c: shapes[c])
        out = _read(tmp_path / "aftermarket_prices_latest.csv").set_index("종목코드")
        assert out.loc["002710", "AH_STATUS"] == "FETCH_FAIL"
        assert out.loc["005930", "AH_STATUS"] == "FETCH_FAIL"
        assert out.loc["000660", "AH_STATUS"] == "FETCH_FAIL"
        assert out.loc["035720", "AH_STATUS"] == "NO_PRINT", "가격 없는 OK 는 믿지 않는다"
        assert out["시간외종가"].isna().all()

    @pytest.mark.parametrize("bad", ["2026101", "2026-10-16", "20261399", "latest"])
    def test_bad_trade_ymd_returns_error(self, tmp_path, bad):
        rec_path = _write_env(tmp_path)
        s = NA.build_aftermarket_archive(str(rec_path), bad, str(tmp_path),
                                         fetch_fn=lambda c: {"status": "NO_PRINT"})
        assert s["error"] and s["n_rows"] == 0
        assert not (tmp_path / "aftermarket_prices_latest.csv").exists()

    def test_unreadable_recommend_returns_error(self, tmp_path):
        s = NA.build_aftermarket_archive(str(tmp_path / "nope.csv"), YMD, str(tmp_path))
        assert s["error"] and s["n_rows"] == 0
        assert not (tmp_path / "aftermarket_prices_latest.csv").exists()

    def test_legacy_wrapper_resolves_ymd_from_recommend(self, tmp_path):
        rec_path = _write_env(tmp_path)
        table = _fetch_table()
        from unittest.mock import patch
        with patch("naver_aftermarket.fetch_after_market_price",
                   side_effect=lambda c, **kw: dict(table[c])):
            n = NA.fetch_after_market_prices_sidecar(
                str(rec_path), str(tmp_path / "aftermarket_prices_latest.csv"))
        assert n == 2
        out = _read(tmp_path / f"aftermarket_prices_{YMD}.csv")
        assert len(out) == 5 and out["KRX_CLOSE"].notna().sum() == 4


# ─────────────────────────────────────────────────────────────
#  호출부 — 스냅샷 latest 를 더 이상 넘기지 않는다
# ─────────────────────────────────────────────────────────────

def test_pipeline_call_site_does_not_pass_snapshot_latest():
    src = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")
    i = src.index("from naver_aftermarket import")
    j = src.index("except ImportError", i)
    block = src[i:j]
    assert "build_aftermarket_archive(op_l, trade_ymd, OUT_DIR" in block
    assert "price_snapshot_latest" not in block
    assert "fetch_after_market_prices_sidecar" not in block


def test_legacy_snapshot_writer_removed():
    src = (ROOT / "naver_aftermarket.py").read_text(encoding="utf-8")
    assert "def update_csv_with_aftermarket" not in src
    # 스냅샷 파일에 쓰는 코드가 없어야 한다 (읽기 전용 load_krx_close 만)
    assert "snap.to_csv" not in src


def test_workflow_commits_dated_archive():
    """auto_collect.yml 의 `git add -f data/*.csv` 가 data/aftermarket_prices_{ymd}.csv 를 잡는다."""
    wf = (ROOT / ".github" / "workflows" / "auto_collect.yml").read_text(encoding="utf-8")
    m = re.search(r"git add -f ([^\n|]*)", wf)
    assert m and "data/*.csv" in m.group(1).split()
    import fnmatch
    assert fnmatch.fnmatch(f"data/aftermarket_prices_{YMD}.csv", "data/*.csv")


# ─────────────────────────────────────────────────────────────
#  실데이터 — 실제 recommend/price_snapshot 파일 모양 그대로 (Naver 만 모의)
# ─────────────────────────────────────────────────────────────

def _latest_real_pair():
    if not DATA.is_dir():
        return None
    recs = sorted(DATA.glob("recommend_2*.csv"))
    for p in reversed(recs):
        m = re.fullmatch(r"recommend_(\d{8})\.csv", p.name)
        if m and (DATA / f"price_snapshot_{m.group(1)}.csv").exists():
            return m.group(1)
    return None


def test_real_data_every_row_written_with_krx_baseline(tmp_path):
    ymd = _latest_real_pair()
    if ymd is None:
        pytest.skip("data/recommend_{ymd}.csv + price_snapshot_{ymd}.csv 쌍 없음")
    shutil.copy(DATA / f"recommend_{ymd}.csv", tmp_path / f"recommend_{ymd}.csv")
    shutil.copy(DATA / f"price_snapshot_{ymd}.csv", tmp_path / f"price_snapshot_{ymd}.csv")
    rec = pd.read_csv(tmp_path / f"recommend_{ymd}.csv", dtype=str, encoding="utf-8-sig")
    snap = pd.read_csv(tmp_path / f"price_snapshot_{ymd}.csv", dtype={"종목코드": str},
                       encoding="utf-8-sig")
    codes = rec["종목코드"].astype(str).str.strip().str.zfill(6)
    codes = codes[codes.str.fullmatch(r"[0-9A-Z]{6}")].drop_duplicates().tolist()
    ana = dict(zip(rec["종목코드"].astype(str).str.zfill(6), pd.to_numeric(rec["종가"], errors="coerce")))
    krx = dict(zip(snap["종목코드"].astype(str).str.zfill(6), pd.to_numeric(snap["종가"], errors="coerce")))
    order = {c: i for i, c in enumerate(codes)}

    def fetch(c):
        # 짝수: 시간외가 == 분석종가 (v20.6.4 는 전부 버렸다) / 홀수: 체결 없음
        if order[c] % 2 == 0:
            return {"status": "OK", "naver_close": krx.get(c), "ah_price_raw": ana[c]}
        return {"status": "NO_PRINT", "naver_close": krx.get(c)}

    rec_bytes = (tmp_path / f"recommend_{ymd}.csv").read_bytes()
    s = NA.build_aftermarket_archive(
        str(tmp_path / f"recommend_{ymd}.csv"), ymd, str(tmp_path), fetch_fn=fetch,
        now_fn=lambda: datetime.strptime(ymd, "%Y%m%d").replace(hour=21, tzinfo=KST),
        sleep_fn=lambda s: None)
    assert (tmp_path / f"recommend_{ymd}.csv").read_bytes() == rec_bytes
    out = _read(tmp_path / f"aftermarket_prices_{ymd}.csv")
    assert out["종목코드"].tolist() == codes, "모든 요청 행이 recommend 순서대로 기록돼야 한다"
    assert s["counts"]["OK"] == (len(codes) + 1) // 2
    assert out["AH_FINAL"].all()
    exp_krx = pd.Series([krx.get(c, float("nan")) for c in codes], dtype=float)
    pd.testing.assert_series_equal(out["KRX_CLOSE"].astype(float), exp_krx, check_names=False)
    okm = out["AH_STATUS"] == "OK"
    exp_chg = (pd.Series([ana[c] for c in codes], dtype=float) / exp_krx - 1).round(6)
    exp_chg[~okm] = float("nan")
    pd.testing.assert_series_equal(out["AH_CHG_VS_KRX"].astype(float), exp_chg,
                                   check_names=False, atol=1e-9)
    assert out.loc[~okm, "시간외종가"].isna().all()
    assert (out.loc[okm & out["분석종가"].notna(), "변동률"] == 0).all()
    # 짝수 행은 시간외 == 분석종가 → NAVER_CLOSE(=KRX) 교차확인은 KRX_CLOSE 가 있으면 True
    has = out["KRX_CLOSE"].notna()
    assert out.loc[has, "CLOSE_MATCH"].astype(str).eq("True").all()
