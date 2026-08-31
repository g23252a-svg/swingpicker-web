# -*- coding: utf-8 -*-
"""v78 승자 프로파일 루프 + v79 수급 전종목 — 선등록·전방누적·멱등."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from services import investor_flow_full as F
from services import winner_profile as W


# ── 합성 유니버스 ─────────────────────────────────────────────────────────
def _mk_ohlcv(data_dir, n_codes=40, n_days=60, seed=7, ymd0="2026-06-01"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(ymd0, periods=n_days)
    rows = []
    for i in range(n_codes):
        px = 10_000 * np.cumprod(1 + rng.normal(0.001, 0.03, n_days))
        px = np.round(px, 0)
        for j, dt in enumerate(dates):
            o = px[j] * (1 + rng.normal(0, 0.004))
            rows.append({"Date": dt, "종목코드": f"{i:06d}",
                         "시가": round(o), "고가": round(max(o, px[j]) * 1.01),
                         "저가": round(min(o, px[j]) * 0.99), "종가": px[j],
                         "거래량": int(rng.integers(50_000, 500_000)),
                         "등락률": 0.0})
    d = pd.DataFrame(rows).set_index("Date")
    d.to_parquet(os.path.join(data_dir, "ohlcv_cache_20260831.parquet"))
    return sorted(dt.strftime("%Y%m%d") for dt in dates)


@pytest.fixture
def uni(tmp_path):
    days = _mk_ohlcv(str(tmp_path))
    return str(tmp_path), days


# ── 레지스트리 동결 ────────────────────────────────────────────────────────
class TestRegistry:
    def test_frozen_v1(self):
        """선등록 목록 — 여기서 특징을 빼면 루프의 전제(사전 고정)가 깨진다.
        추가는 REGISTRY_VERSION을 올려서만 한다."""
        assert W.REGISTRY_VERSION == "v1-20260831"
        assert set(W.FEATURES) == {
            "ret_1d", "ret_5d", "ret_20d", "vol_ratio", "volat_20d",
            "ma20_gap", "rsi14", "hi60_gap", "tv_eok", "tv_rank_pctl",
            "flow_frg_str", "flow_inst_str"}

    def test_promotion_thresholds_frozen(self):
        assert W.MIN_DAYS == 40 and W.FDR_Q == 0.05 and W.HOLD_DAYS == 5


# ── 일일 기록 ─────────────────────────────────────────────────────────────
class TestBuildDay:
    def test_target_is_closed_window(self, uni):
        d, days = uni
        px = W._load_universe_ohlcv(d)
        assert W.target_session(px, days[-1]) == days[-1 - (W.HOLD_DAYS + 1)]

    def test_day_has_all_features_once(self, uni):
        d, days = uni
        day = W.build_day(d, days[-1])
        assert day is not None
        assert set(day["feature"]) == set(W.FEATURES)
        assert len(day) == len(W.FEATURES)

    def test_flow_nan_until_v79_lands(self, uni):
        d, days = uni
        day = W.build_day(d, days[-1])
        flow = day[day.feature.isin(["flow_frg_str", "flow_inst_str"])]
        assert flow["ic"].isna().all()

    def test_small_cohort_refused(self, tmp_path):
        _mk_ohlcv(str(tmp_path), n_codes=5)
        assert W.build_day(str(tmp_path), "20260831") is None

    def test_no_lookahead_in_features(self, uni):
        """특징은 신호일까지의 이력만 본다 — 신호일 이후 데이터를 바꿔도 불변."""
        d, days = uni
        day1 = W.build_day(d, days[-1])
        p = os.path.join(d, "ohlcv_cache_20260831.parquet")
        px = pd.read_parquet(p).reset_index()
        sig = pd.to_datetime(days[-1 - (W.HOLD_DAYS + 1)])
        px.loc[px["Date"] > sig, "거래량"] = 1          # 미래 조작
        px.set_index("Date").to_parquet(p)
        day2 = W.build_day(d, days[-1])
        f1 = day1[day1.feature == "vol_ratio"]["ic"].iloc[0]
        f2 = day2[day2.feature == "vol_ratio"]["ic"].iloc[0]
        assert f1 == pytest.approx(f2), "특징이 미래를 봤다"


# ── 로그 멱등 + 전방 누적 ──────────────────────────────────────────────────
class TestLog:
    def test_idempotent(self, uni):
        d, days = uni
        s1 = W.run_batch(d, days[-1]); s2 = W.run_batch(d, days[-1])
        assert s1["added"] is True and s2["added"] is False
        log = pd.read_parquet(os.path.join(d, W.LOG_NAME))
        assert not log.duplicated(["ymd", "feature"]).any()

    def test_summary_carries_protocol(self, uni):
        d, days = uni
        W.run_batch(d, days[-1])
        s = json.load(open(os.path.join(d, W.SUMMARY_NAME)))
        assert s["protocol"]["min_days"] == 40
        assert s["registry"] == W.REGISTRY_VERSION


# ── 승격 판정 ─────────────────────────────────────────────────────────────
def _fake_log(feature_ics: dict, n_days=50):
    rows = []
    for i in range(n_days):
        for f, ic in feature_ics.items():
            rows.append({"ymd": f"202601{i:02d}", "feature": f,
                         "ic": ic + np.random.default_rng(i).normal(0, 0.02),
                         "winner_gap": 0.0, "n": 100, "win_rate": 0.4})
    return pd.DataFrame(rows)


class TestPromotion:
    def test_strong_stable_feature_passes_stage1(self):
        log = _fake_log({"vol_ratio": 0.10, "rsi14": 0.001})
        ev = {r["feature"]: r for r in W.evaluate(log)}
        assert ev["vol_ratio"]["stage1"] is True
        assert ev["rsi14"]["stage1"] is False

    def test_insufficient_days_never_pass(self):
        log = _fake_log({"vol_ratio": 0.30}, n_days=W.MIN_DAYS - 1)
        ev = {r["feature"]: r for r in W.evaluate(log)}
        assert ev["vol_ratio"]["stage1"] is False, "40일 미만은 아무리 세도 후보 아님"

    def test_sign_flip_never_passes(self):
        rows = []
        for i in range(50):
            ic = 0.15 if i < 25 else -0.15         # 전/후반 부호 반전
            rows.append({"ymd": f"202601{i:02d}", "feature": "vol_ratio",
                         "ic": ic, "winner_gap": 0, "n": 100, "win_rate": .4})
        ev = W.evaluate(pd.DataFrame(rows))
        assert ev[0]["stage1"] is False

    def test_line_mentions_stage1(self):
        s = {"days_total": 55, "features": [
            {"feature": "vol_ratio", "stage1": True, "ic_mean": .1, "p": .001, "days": 55}]}
        assert "1단계 통과" in W.line(s) and "vol_ratio" in W.line(s)
        assert W.line(None) == ""


# ── v79 수급 저장 계약 ────────────────────────────────────────────────────
class TestFlowFull:
    def test_roundtrip_and_units(self, tmp_path):
        df = pd.DataFrame({"종목코드": ["000001", "000002"],
                           "frg_eok": [12.5, -3.0], "inst_eok": [0.5, 7.0]})
        p = F.save_day(str(tmp_path), "20260831", df)
        back = pd.read_parquet(p)
        assert list(back.columns) == ["종목코드", "frg_eok", "inst_eok"]
        assert back.frg_eok.abs().max() < 1e6, "억 단위여야 한다 — 원이면 신호가 조작된다"

    def test_collect_idempotent(self, tmp_path, monkeypatch):
        df = pd.DataFrame({"종목코드": ["000001"], "frg_eok": [1.0], "inst_eok": [2.0]})
        F.save_day(str(tmp_path), "20260831", df)
        called = []
        monkeypatch.setattr(F, "fetch_day", lambda ymd: called.append(ymd) or df)
        p = F.collect(str(tmp_path), "20260831")
        assert p and not called, "이미 있으면 네트워크를 부르면 안 된다"

    def test_fetch_failure_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(F, "fetch_day", lambda ymd: None)
        assert F.collect(str(tmp_path), "20260901") is None

    def test_winner_profile_picks_up_flow(self, uni):
        d, days = uni
        sig = days[-1 - (W.HOLD_DAYS + 1)]
        codes = [f"{i:06d}" for i in range(40)]
        F.save_day(d, sig, pd.DataFrame({
            "종목코드": codes, "frg_eok": np.linspace(-5, 5, 40),
            "inst_eok": np.linspace(2, -2, 40)}))
        day = W.build_day(d, days[-1])
        flow = day[day.feature == "flow_frg_str"]
        assert flow["n"].iloc[0] >= W.MIN_COHORT
        assert np.isfinite(flow["ic"].iloc[0]), "수급이 생기면 그날부터 검정에 들어간다"


# ── 배선 ─────────────────────────────────────────────────────────────────
class TestWiring:
    def test_finalize_calls_both(self):
        src = open("pipeline_finalize.py", encoding="utf-8").read()
        assert "_WP.run_batch(OUT_DIR, trade_ymd)" in src
        assert "_IFF.collect(OUT_DIR, trade_ymd)" in src

    def test_flow_runs_before_winner_profile(self):
        """수급을 먼저 수집해야 같은 배치에서 특징으로 쓸 수 있다."""
        src = open("pipeline_finalize.py", encoding="utf-8").read()
        assert src.index("_IFF.collect") < src.index("_WP.run_batch")


# ── 승격 문턱의 각 축이 실제로 걸러내는지 (뮤테이션 표적) ──────────────────
def _log_from_series(x, feature="vol_ratio", extra=None):
    rows = [{"ymd": f"2026{i:04d}", "feature": feature, "ic": float(v),
             "winner_gap": 0.0, "n": 100, "win_rate": 0.4}
            for i, v in enumerate(x)]
    if extra is not None:
        rows += extra
    return pd.DataFrame(rows)


class TestPromotionAxes:
    def test_sign_flip_blocks_even_when_significant(self):
        """전반 음수·후반 양수인데 전체 평균은 유의 — 부호안정 축이 없으면 통과해버린다."""
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.normal(-0.02, 0.15, 25), rng.normal(0.35, 0.15, 25)])
        ev = {r["feature"]: r for r in W.evaluate(_log_from_series(x))}
        r = ev["vol_ratio"]
        assert r["p"] < 0.05 and r["days"] >= W.MIN_DAYS   # 유의한데
        assert r["sign_stable"] is False
        assert r["stage1"] is False, "부호가 뒤집힌 특징이 승격됐다"

    def test_bh_fdr_stricter_than_raw_p(self):
        """p=0.027짜리는 단독으론 유의지만, 특징 12개 중 2등이면 BH 컷(0.0083)에 걸린다."""
        rng = np.random.default_rng(123)
        borderline = rng.normal(0, 0.05, 50) + 0.005          # HAC p≈0.027
        strong = np.abs(rng.normal(0.5, 0.05, 50))            # 1등 (통과 마땅)
        extra = []
        for j, f in enumerate(sorted(W.FEATURES - {"vol_ratio", "rsi14"}
                                     if isinstance(W.FEATURES, set)
                                     else set(W.FEATURES) - {"vol_ratio", "rsi14"})):
            noise = np.random.default_rng(200 + j).normal(0, 0.1, 50)
            extra += [{"ymd": f"2026{i:04d}", "feature": f, "ic": float(v),
                       "winner_gap": 0.0, "n": 100, "win_rate": 0.4}
                      for i, v in enumerate(noise)]
        log = pd.concat([_log_from_series(strong, "vol_ratio"),
                         _log_from_series(borderline, "rsi14"),
                         pd.DataFrame(extra) if extra else pd.DataFrame()],
                        ignore_index=True)
        ev = {r["feature"]: r for r in W.evaluate(log)}
        assert ev["vol_ratio"]["stage1"] is True
        assert 0.0083 < ev["rsi14"]["p"] < 0.05, f"픽스처 붕괴: p={ev['rsi14']['p']}"
        assert ev["rsi14"]["stage1"] is False, \
            "생 p<0.05로 통과시켰다 — BH-FDR가 죽었다 (12개를 재면 1개는 우연히 나온다)"

    def test_hac_blocks_overlap_inflation(self):
        """5일 창 겹침형 자기상관(값이 5일씩 지속) — 순진 t는 p=0.006, HAC는 0.15."""
        rng = np.random.default_rng(7)
        x = np.repeat(rng.normal(0.05, 0.15, 12), 5)          # n=60, 유효표본 12
        from scipy import stats as _st
        assert _st.ttest_1samp(x, 0).pvalue < 0.01            # 순진 t라면 통과했을 것
        t, p = W._hac_p(x)
        assert p > 0.1, "겹침 보정이 죽었다 — HZ3 사건(p=0.0002→HAC 0.11) 재발 경로"

    def test_build_day_uses_ssot_realized(self, uni, monkeypatch):
        """실현수익은 pick_history._realized(SSOT) 하나만 쓴다 — 종가수익으로
        바꾸면 손절·거래정지 처리가 전부 사라진다."""
        d, days = uni
        monkeypatch.setattr(W, "_realized", lambda g, ymd: 0.777)
        day = W.build_day(d, days[-1])
        measured = day[day["n"] >= W.MIN_COHORT]          # flow 특징(n=0)은 제외
        assert len(measured) > 0
        assert (measured["win_rate"] == 1.0).all(), "SSOT 함수를 우회해 수익을 직접 계산했다"
