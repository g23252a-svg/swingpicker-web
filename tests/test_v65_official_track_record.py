# -*- coding: utf-8 -*-
"""[v65] 성적표가 '실제로 산 것'을 재는가 — 재구성 대신 기록을 읽는다.

■ 무엇을 발견했나 (2026-08-19, v64 머지 직후 검산)
  v63은 "성적표가 엔진이 고르지 않는 종목을 재고 있다"를 고쳤다. 그 결과
  화면 헤드라인이 `엔진 1위 +3.58%(초과 +2.84%p)`로 바뀌었다. 그런데 그
  숫자를 배치 CSV로 되짚으니 **여전히 실제 픽의 성적이 아니었다.**

  ① 픽이 없던 날에도 픽을 만들어 냈다
     재구성 축(gated_topN)은 후보 풀(중위 20종목) 상단을 **매일** 산다.
     2026-07-14~08-12 측정 21일 전부에서 성적이 생겼다. 같은 구간에서 배치가
     실제로 기록한 공식 매수(PRODUCTION_BUY)는 **3일**뿐이었다. 겹치는 3일 중
     종목까지 같은 날은 2일(08-05 흥구석유 ○ · 08-11 씨어스 ○ ·
     08-12 클로봇 vs 삼천당제약 ✗).
     → 18/21일은 살 수 없었던 날의 수익이었다.

  ② 휴장일 스냅샷을 세션으로 셌다
     price_snapshot은 휴장일에도 생성되며 직전 세션을 그대로 복사한다.
     전수 점검에서 9일이 직전 세션과 종가 100% 동일했다(20260416·0430·0501·
     0505·0525·0602·0603·**0717**·**0817**). 0817은 공통 2,872종목의
     시가·저가·종가가 **전부** 같았다.
     유령 세션은 t+1 진입가를 추천일 당일 가격으로 만들고, t+h 지평을 한 세션
     줄이고, 같은 날을 일별 평균에 두 번 넣어 |t|를 부풀린다.

■ 무엇을 고쳤나
  · 기록된 결정을 읽는다 — PRODUCTION_BUY는 배치가 남긴 결정 그 자체이므로
    재구성 대상이 아니다. 픽 0건인 날은 0건으로 센다.
  · 유령 세션을 세션 목록과 측정일에서 제외한다(alpha_live_report).
    pick_reliability는 OHLCV 캐시의 날짜 집합을 세션 달력으로 써서 같은 일을
    한다(전수 7일: 20260302·0501·0505·0525·0603·0717·0817).
  · 재구성 축은 지우지 않고 남기되, 블록 안에 '공식 매수 발생 여부를 재현하지
    않는다'와 픽 없던 날 수를 함께 적는다.

■ 수정 후 실측 (2026-08-18 데이터)
  h5: 측정 19→17일 · IC +0.0884(t=2.25) → +0.0926(t=2.14, p=0.0485)
      공식픽 **1건/17일** → 통계 미산출(최소 3일 미달)
      재구성 축은 오히려 좋아졌다(+3.58% → +4.92%, 초과 +3.94%p, p=0.024).
      **좋아진 숫자도 공식 성적이 아니다** — 16/17일이 픽 없던 날이다.
  pick_reliability: 픽 발생일 7 → 6일 (유령 배치일 1건 중복 제거)

■ 이 파일이 고정하는 것
  1. 유령 세션은 세션이 아니다. 연속 복사본도 전부 제외된다.
  2. 공식 매수 축은 **기록만** 읽는다. 컬럼이 없으면 '0건'이 아니라 '판정 불가'다.
  3. 재구성 축은 자기가 재구성임을 블록 안에서 밝힌다 — 못을 빼면 실패한다.
  4. 헤드라인은 공식 성적을 먼저 말하고, 표본이 얇으면 얇다고 말한다.
     재구성 숫자를 공식 성적 자리에 적으면 실패한다.
  5. 픽 발생일 3일 미만이면 평균을 만들지 않는다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.alpha_live_report as alr  # noqa: E402
import services.pick_reliability as pr  # noqa: E402

DATA = ROOT / "data"


def _snap(rows):
    """price_snapshot 한 장을 종목코드 인덱스 프레임으로."""
    return pd.DataFrame(rows).set_index("종목코드")


def _mkpx(n_codes=60, closes=None):
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    closes = closes if closes is not None else [1000 + i for i in range(n_codes)]
    return _snap([{"종목코드": c, "종가": v, "시가": v, "고가": v, "저가": v}
                  for c, v in zip(codes, closes)])


# ══════════════════════════════════════════════════════════════════
#  1. 유령 세션 — 휴장일 복사본은 세션이 아니다
# ══════════════════════════════════════════════════════════════════
class TestPhantomSessions:
    def test_exact_copy_is_dropped(self):
        a = _mkpx()
        px = {"20260814": a, "20260817": a.copy()}
        real, dropped = alr._drop_phantom_sessions(px, ["20260814", "20260817"])
        assert real == ["20260814"]
        assert dropped == ["20260817"]

    def test_different_prices_are_kept(self):
        px = {"20260814": _mkpx(), "20260818": _mkpx(closes=[1500 + i for i in range(60)])}
        real, dropped = alr._drop_phantom_sessions(px, ["20260814", "20260818"])
        assert real == ["20260814", "20260818"]
        assert dropped == []

    def test_thin_overlap_is_not_judged_phantom(self):
        """공통 종목이 적으면 '동일'을 근거로 삼지 않는다 — 보수적 방향."""
        few = alr.DUP_SESSION_MIN_CODES - 1
        a = _mkpx(n_codes=few)
        px = {"20260814": a, "20260817": a.copy()}
        real, dropped = alr._drop_phantom_sessions(px, ["20260814", "20260817"])
        assert dropped == [], "표본이 얇은데 유령으로 단정했다"
        assert real == ["20260814", "20260817"]

    def test_consecutive_copies_all_dropped(self):
        """연휴는 복사본이 연달아 온다. 비교 기준을 복사본으로 갱신하면
        둘째 복사본이 '직전과 다르다'며 살아남는다 — 그 버그를 막는다."""
        a = _mkpx()
        px = {"20260501": a, "20260502": a.copy(), "20260503": a.copy(),
              "20260504": _mkpx(closes=[1200 + i for i in range(60)])}
        real, dropped = alr._drop_phantom_sessions(px, sorted(px))
        assert dropped == ["20260502", "20260503"], f"연속 복사본 처리 실패: {dropped}"
        assert real == ["20260501", "20260504"]

    def test_partial_change_below_ratio_is_kept(self):
        """일부만 바뀐 날은 실제 세션이다(거래는 있었다)."""
        n = 60
        base = [1000 + i for i in range(n)]
        moved = list(base)
        for i in range(5):            # 5/60 ≈ 8.3% 변동 > (1-0.99)
            moved[i] = base[i] + 50
        px = {"20260814": _mkpx(closes=base), "20260818": _mkpx(closes=moved)}
        real, dropped = alr._drop_phantom_sessions(px, ["20260814", "20260818"])
        assert dropped == []

    def test_order_is_preserved(self):
        px = {y: _mkpx(closes=[1000 + i + k for i in range(60)])
              for k, y in enumerate(["20260810", "20260811", "20260812"])}
        real, _ = alr._drop_phantom_sessions(px, ["20260810", "20260811", "20260812"])
        assert real == ["20260810", "20260811", "20260812"]

    def test_missing_close_column_is_skipped_not_crashed(self):
        px = {"20260814": pd.DataFrame({"x": [1]}).set_index("x")}
        real, dropped = alr._drop_phantom_sessions(px, ["20260814"])
        assert real == [] and dropped == []

    @pytest.mark.skipif(not (DATA / "price_snapshot_20260817.csv").exists(),
                        reason="실데이터 스냅샷 없음")
    def test_real_data_holidays_are_found(self):
        """실측: 0717·0817이 유령으로 잡혀야 한다(둘 다 알파 측정 구간 안)."""
        px, days = alr._load_snapshots(str(DATA))
        real, dropped = alr._drop_phantom_sessions(px, days)
        assert "20260817" in dropped, f"0817을 못 잡았다: {dropped}"
        assert "20260717" in dropped, f"0717을 못 잡았다: {dropped}"
        assert set(real).isdisjoint(dropped)
        assert len(real) + len(dropped) <= len(days)


# ══════════════════════════════════════════════════════════════════
#  2. 공식 매수 축은 기록만 읽는다
# ══════════════════════════════════════════════════════════════════
class TestOfficialMaskReadsRecordOnly:
    def test_numeric_column(self):
        df = pd.DataFrame({"PRODUCTION_BUY": [1, 0, 1]})
        assert list(alr._official_mask(df)) == [True, False, True]

    def test_string_column(self):
        df = pd.DataFrame({"PRODUCTION_BUY": ["True", "False", "yes"]})
        assert list(alr._official_mask(df)) == [True, False, True]

    def test_missing_column_is_unknown_not_zero(self):
        """구 배치를 '픽 0건'으로 세면 없는 성적을 있다고 말하게 된다."""
        assert alr._official_mask(pd.DataFrame({"종목코드": ["000001"]})) is None

    def test_does_not_reconstruct_from_top_pick(self):
        """production_buy_mask의 레거시 폴백(TOP_PICK & BUY_NOW_ELIGIBLE)은
        재구성이다. 공식 성적 축에서는 쓰지 않는다."""
        df = pd.DataFrame({"TOP_PICK": [1], "BUY_NOW_ELIGIBLE": [1]})
        assert alr._official_mask(df) is None, "재구성 폴백이 되살아났다"

    def test_all_zero_is_zero_picks_not_unknown(self):
        m = alr._official_mask(pd.DataFrame({"PRODUCTION_BUY": [0, 0]}))
        assert m is not None and int(m.sum()) == 0


# ══════════════════════════════════════════════════════════════════
#  3. 리포트 계약 — 재구성은 스스로 재구성이라고 밝힌다
# ══════════════════════════════════════════════════════════════════
def _panel(tmp: Path, pick_days=(), n_codes=60, n_days=12):
    """합성 패널. pick_days에 든 날만 PRODUCTION_BUY=1을 기록한다.

    날짜는 ALPHA_LIVE_FROM 이후여야 한다 — 그 앞은 리포트가 알파 미도입
    구간으로 보고 전부 건너뛴다(첫 구현이 이 함정에 빠졌다).
    """
    tmp.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(alr.ALPHA_LIVE_FROM)
    days = [(start + pd.Timedelta(days=i)).strftime("%Y%m%d")
            for i in range(n_days)]
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    price = {c: 10000.0 for c in codes}
    for di, ymd in enumerate(days):
        rows = []
        for ci, c in enumerate(codes):
            drift = 0.004 * ((n_codes - ci) - n_codes / 2) / (n_codes / 2)
            prev = price[c]
            # 결정적 지터 — 완전 동일한 수열은 t검정에서 분산 0이 되어
            # scipy가 정밀도 경고를 낸다. 난수는 쓰지 않는다(재현성).
            jitter = ((di * 7 + ci * 13) % 11) * 0.03
            p = prev * (1.0 + drift) + jitter
            price[c] = p
            rows.append({"종목코드": c, "종목명": f"T{c}", "시장": "KOSPI",
                         "시가": round(prev, 2), "종가": round(p, 2),
                         "고가": round(max(prev, p) * 1.01, 2),
                         "저가": round(min(prev, p) * 0.99, 2)})
        pd.DataFrame(rows).to_csv(tmp / f"price_snapshot_{ymd}.csv",
                                  index=False, encoding="utf-8-sig")
    for ymd in days:
        rows = []
        for ci, c in enumerate(codes):
            rows.append({"종목코드": c, "종목명": f"T{c}", "ROUTE": "WAIT",
                         "ALPHA_SCORE": float(100 - ci),
                         "RR_NOW_TP1": 2.0,
                         "PRODUCTION_BUY": 1 if (ymd in pick_days and ci == 0) else 0})
        pd.DataFrame(rows).to_csv(tmp / f"recommend_{ymd}.csv",
                                  index=False, encoding="utf-8-sig")
    return tmp, days


class TestReportContract:
    def test_reconstruction_block_declares_itself(self, tmp_path):
        d, _ = _panel(tmp_path / "data")
        r = alr.compute_alpha_live_report(str(d))
        assert r["ok"], r
        found = 0
        for blk in r["horizons"].values():
            for k in alr.TOP_NS:
                g = blk.get(f"gated_top{k}")
                if not g:
                    continue
                found += 1
                assert g["is_reconstruction"] is True
                assert "재구성" in g["axis"]
                assert "재현하지 않는다" in g["caveat"]
                assert "pick_days_covered" in g
                assert "no_pick_days_counted" in g
        assert found > 0, "재구성 축이 아예 생기지 않았다 — 계약 검증 불가"

    def test_official_block_is_not_reconstruction(self, tmp_path):
        d, _ = _panel(tmp_path / "data")
        r = alr.compute_alpha_live_report(str(d))
        for blk in r["horizons"].values():
            ob = blk["official"]
            assert ob["is_reconstruction"] is False
            assert "공식 매수" in ob["axis"]

    def test_zero_pick_days_are_counted_as_zero(self, tmp_path):
        """픽이 하나도 없는 패널에서 공식 성적이 만들어지면 안 된다."""
        d, _ = _panel(tmp_path / "data", pick_days=())
        r = alr.compute_alpha_live_report(str(d))
        for blk in r["horizons"].values():
            ob = blk["official"]
            assert ob["pick_days_declared"] == 0
            assert ob["no_pick_days"] == ob["days_recorded"] > 0
            assert "on_pick_days" not in ob
            assert "reason" in ob

    def test_day_counts_are_an_identity(self, tmp_path):
        d, days = _panel(tmp_path / "data")
        d, _ = _panel(tmp_path / "data", pick_days=tuple(days[1:5]))
        r = alr.compute_alpha_live_report(str(d))
        for blk in r["horizons"].values():
            ob = blk["official"]
            assert (ob["pick_days_declared"] + ob["no_pick_days"]
                    == ob["days_recorded"]), ob
            assert ob["days_pick_unmeasured"] >= 0
            assert ob["pick_days"] + ob["days_pick_unmeasured"] == ob["pick_days_declared"]

    def test_official_stats_appear_once_enough_pick_days(self, tmp_path):
        d, days = _panel(tmp_path / "data")
        d, _ = _panel(tmp_path / "data", pick_days=tuple(days[1:6]))
        r = alr.compute_alpha_live_report(str(d))
        got = [blk["official"] for blk in r["horizons"].values()
               if "on_pick_days" in blk["official"]]
        assert got, "픽 5일인데 공식 성적이 산출되지 않았다"
        for ob in got:
            assert ob["pick_days"] >= alr.OFFICIAL_MIN_PICK_DAYS
            cash = ob["all_days_cash"]
            assert cash["n"] == ob["pick_days"] + ob["no_pick_days"]
            assert "현금" in cash["note"]

    def test_below_minimum_makes_no_average(self, tmp_path):
        d, days = _panel(tmp_path / "data")
        d, _ = _panel(tmp_path / "data", pick_days=tuple(days[1:3]))
        r = alr.compute_alpha_live_report(str(d))
        for blk in r["horizons"].values():
            ob = blk["official"]
            if ob["pick_days"] < alr.OFFICIAL_MIN_PICK_DAYS:
                assert "on_pick_days" not in ob
                assert "all_days_cash" not in ob

    def test_sessions_block_reports_phantoms(self, tmp_path):
        d, days = _panel(tmp_path / "data", n_codes=alr.DUP_SESSION_MIN_CODES + 10)
        # 마지막 날을 그대로 복사한 유령 스냅샷을 하나 넣는다
        ghost = (pd.Timestamp(days[-1]) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        last = pd.read_csv(d / f"price_snapshot_{days[-1]}.csv",
                           encoding="utf-8-sig", dtype={"종목코드": str})
        last.to_csv(d / f"price_snapshot_{ghost}.csv", index=False,
                    encoding="utf-8-sig")
        r = alr.compute_alpha_live_report(str(d))
        s = r["sessions"]
        assert s["phantom_dropped"] >= 1
        assert ghost in s["phantom_days"]
        assert s["real"] + s["phantom_dropped"] <= s["snapshots"]
        assert "휴장" in s["rule"]

    def test_phantom_batch_day_is_not_measured(self, tmp_path):
        """유령 세션 날의 배치는 직전일 픽의 복사 — 측정일에서 빠져야 한다."""
        d, days = _panel(tmp_path / "data", n_codes=alr.DUP_SESSION_MIN_CODES + 10)
        ghost, prev = days[5], days[4]
        src = pd.read_csv(d / f"price_snapshot_{prev}.csv",
                          encoding="utf-8-sig", dtype={"종목코드": str})
        src.to_csv(d / f"price_snapshot_{ghost}.csv", index=False,
                   encoding="utf-8-sig")      # 직전일 복사 = 유령 세션
        r = alr.compute_alpha_live_report(str(d))
        assert ghost in r["sessions"]["phantom_days"]
        assert ghost not in [x["ymd"] for x in r["per_day"]]


# ══════════════════════════════════════════════════════════════════
#  4. 헤드라인 — 공식 성적을 먼저, 없으면 없다고
# ══════════════════════════════════════════════════════════════════
def _fake(official=None, gated=None, n_days=17):
    blk = {"n_days": n_days, "ic_mean": 0.09, "ic_median": 0.1,
           "ic_positive_days": 13, "ic_t": {"n": n_days, "t": 2.14, "p": 0.048},
           "universe": {"n": n_days, "mean_pct": 0.98}}
    blk["official"] = official if official is not None else {
        "axis": "공식 매수(기록된 결정)", "is_reconstruction": False,
        "days_recorded": n_days, "pick_days": 1, "pick_days_declared": 1,
        "no_pick_days": n_days - 1, "days_pick_unmeasured": 0,
        "reason": "픽 발생일 1일 — 3일 미만이라 통계를 내지 않는다"}
    if gated is not None:
        blk["gated_top1"] = gated
    return {"ok": True, "horizons": {"h5": blk}}


class TestHeadline:
    def test_leads_with_official_when_available(self):
        line = alr.alpha_live_line(_fake(official={
            "axis": "공식 매수(기록된 결정)", "is_reconstruction": False,
            "days_recorded": 17, "pick_days": 5, "pick_days_declared": 5,
            "no_pick_days": 12, "days_pick_unmeasured": 0,
            "on_pick_days": {"n": 5, "mean_pct": 2.1, "excess_mean_pct": 1.3,
                             "stop_rate": 0.2},
            "all_days_cash": {"n": 17, "mean_pct": 0.62, "note": "x"}}), 5)
        assert "공식픽 5건/17일" in line
        assert "+2.10%" in line and "+1.30%p" in line
        assert "현금포함 +0.62%/일" in line

    def test_says_thin_sample_instead_of_reconstruction_number(self):
        """표본이 얇을 때 재구성 숫자를 대신 적으면 v63과 같은 오표기다."""
        line = alr.alpha_live_line(_fake(gated={
            "mean_pct": 4.915, "excess_mean_pct": 3.938, "n_days": 17,
            "no_pick_days_counted": 16, "stop_rate": 0.23}), 5)
        assert "성적 표본 부족" in line
        assert "4.91" not in line and "3.93" not in line, \
            f"재구성 숫자가 공식 성적 자리에 들어갔다: {line}"

    def test_reconstruction_fallback_is_labelled(self):
        """공식 기록이 아예 없는 구 배치에서만 재구성을 보여주고, 그때도
        '픽 없던 날 포함'을 명시한다."""
        r = _fake(official={"axis": "공식 매수(기록된 결정)",
                            "is_reconstruction": False, "days_recorded": 0,
                            "pick_days": 0, "pick_days_declared": 0,
                            "no_pick_days": 0, "days_pick_unmeasured": 0},
                  gated={"mean_pct": 4.9, "excess_mean_pct": 3.9, "n_days": 17,
                         "no_pick_days_counted": 16, "stop_rate": 0.23})
        line = alr.alpha_live_line(r, 5)
        assert "후보 풀 상단" in line
        assert "픽 없던 날 16일 포함" in line
        assert "엔진 1위" not in line, "재구성을 '엔진 1위'로 다시 부르고 있다"

    def test_empty_report_is_silent(self):
        assert alr.alpha_live_line(None) == ""
        assert alr.alpha_live_line({"ok": False}) == ""
        assert alr.alpha_live_line({"ok": True, "horizons": {}}) == ""

    def test_ic_is_always_present(self):
        for r in (_fake(), _fake(gated={"mean_pct": 1.0, "excess_mean_pct": 0.1,
                                        "n_days": 5, "no_pick_days_counted": 4,
                                        "stop_rate": 0.0})):
            assert "IC" in alr.alpha_live_line(r, 5)


# ══════════════════════════════════════════════════════════════════
#  5. pick_reliability — 같은 유형의 중복 계상
# ══════════════════════════════════════════════════════════════════
class TestPickReliabilitySessions:
    def test_non_session_batch_day_is_skipped(self, tmp_path):
        """OHLCV 캐시에 없는 날짜의 배치는 휴장 복사본 — 세지 않는다."""
        d = tmp_path / "data"
        d.mkdir()
        dates = pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03",
                                "2026-07-06", "2026-07-07", "2026-07-08",
                                "2026-07-09"])
        rows = []
        for i, dt in enumerate(dates):
            rows.append({"종목코드": "000001", "Date": dt, "시가": 1000 + i,
                         "고가": 1020 + i, "저가": 990 + i, "종가": 1010 + i})
        pd.DataFrame(rows).set_index("Date").to_parquet(
            d / "ohlcv_cache_20260709.parquet")
        rec = pd.DataFrame([{"종목코드": "000001", "종목명": "A",
                             "PRODUCTION_BUY": 1}])
        rec.to_csv(d / "recommend_20260701.csv", index=False, encoding="utf-8-sig")
        rec.to_csv(d / "recommend_20260704.csv", index=False, encoding="utf-8-sig")
        res = pr.compute_pick_reliability(str(d))
        assert res.get("non_session_days_skipped", 0) >= 1, res
        if res.get("ok"):
            # 07-04는 토요일 — 캐시에 세션이 없으므로 측정일에서 빠진다
            assert res["pick_days"] == 1, res

    @pytest.mark.skipif(not (DATA / "recommend_20260817.csv").exists(),
                        reason="실데이터 없음")
    def test_real_data_skips_known_holidays(self):
        res = pr.compute_pick_reliability(str(DATA))
        assert res.get("non_session_days_skipped", 0) >= 1, res


# ══════════════════════════════════════════════════════════════════
#  6. 실데이터 회귀 — 리포트가 스스로 과대표기하지 않는가
# ══════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not (DATA / "recommend_20260818.csv").exists(),
                    reason="실데이터 없음")
class TestRealData:
    @pytest.fixture(scope="class")
    def report(self):
        return alr.compute_alpha_live_report(str(DATA))

    def test_phantom_days_excluded_from_measurement(self, report):
        assert report["ok"], report
        measured = {x["ymd"] for x in report["per_day"]}
        assert "20260817" not in measured
        assert "20260717" not in measured

    def test_official_pick_days_do_not_exceed_recorded(self, report):
        for blk in report["horizons"].values():
            ob = blk["official"]
            assert 0 <= ob["pick_days_declared"] <= ob["days_recorded"]

    def test_reconstruction_admits_its_phantom_pick_days(self, report):
        """실데이터에서 재구성 축은 픽 없던 날을 대량 포함한다 — 그 수가
        블록에 적혀 있어야 한다(적혀 있지 않으면 다시 공식 성적으로 읽힌다)."""
        seen = False
        for blk in report["horizons"].values():
            g = blk.get("gated_top1")
            if not g:
                continue
            seen = True
            assert g["no_pick_days_counted"] > 0, g
            assert g["pick_days_covered"] + g["no_pick_days_counted"] == g["n_days"]
        assert seen

    def test_headline_does_not_promise_official_record_it_lacks(self, report):
        line = alr.alpha_live_line(report, 5)
        assert line
        ob = report["horizons"]["h5"]["official"]
        if "on_pick_days" not in ob:
            assert "성적 표본 부족" in line, line
