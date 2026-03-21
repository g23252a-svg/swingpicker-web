# -*- coding: utf-8 -*-
"""test_kelly_calibrator.py — Kelly Calibrator 단위 테스트 (pytest 표준)
═══════════════════════════════════════════════════
실행: pytest test_kelly_calibrator.py -v
"""
import os
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

from kelly_calibrator import (
    save_per_trade_log,
    load_per_trade_log,
    build_calibration_table,
    calibrated_win_rate,
    kelly_fraction,
    _bayesian_win_rate,
    _time_weight,
    _fallback_linear,
    _normalize_ymd,
    apply_kelly_calibrated,
    PRIOR_WIN_RATE,
)
import kelly_calibrator as kc


@pytest.fixture()
def tmpdir_clean():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def _big_trades(n=200):
    np.random.seed(42)
    trades = []
    for i in range(n):
        score = np.random.uniform(50, 100)
        win = 1 if np.random.random() < (0.3 + score * 0.005) else 0
        trades.append({
            "rec_date": f"2026{(i%12+1):02d}{(i%28+1):02d}",
            "code": f"{i:06d}", "method": "RANK_SCORE", "topk": 5,
            "horizon": 5, "score": round(score, 1), "entry_price": 10000,
            "exit_price": 10500 if win else 9500, "stop_price": 9500,
            "target_price": 11000, "ret_pct": 5.0 if win else -5.0,
            "win": win, "exit_type": "tp_hit" if win else "stop_hit", "b_ratio": 2.0,
        })
    return trades


# ═══════════════════════════════════════════════
#  1. per-trade 로그 저장
# ═══════════════════════════════════════════════
class TestPerTradeLog:
    def test_save_and_dedup(self, tmpdir_clean):
        trades = [
            {"rec_date": "20260210", "code": "005930", "method": "RANK_SCORE",
             "topk": 5, "horizon": 5, "score": 85.0, "entry_price": 60000,
             "exit_price": 63000, "stop_price": 57000, "target_price": 66000,
             "ret_pct": 5.0, "win": 1, "exit_type": "tp_hit", "b_ratio": 2.0},
            {"rec_date": "20260210", "code": "000660", "method": "RANK_SCORE",
             "topk": 5, "horizon": 5, "score": 72.0, "entry_price": 130000,
             "exit_price": 125000, "stop_price": 123000, "target_price": 140000,
             "ret_pct": -3.85, "win": 0, "exit_type": "stop_hit", "b_ratio": 1.43},
        ]
        path = save_per_trade_log(tmpdir_clean, trades, "20260215")
        assert os.path.exists(path)
        assert len(load_per_trade_log(tmpdir_clean)) == 2

        # 중복 저장 → load 시 dedup (append-only 파일이므로 raw는 4행이지만 load는 2행)
        save_per_trade_log(tmpdir_clean, trades, "20260215")
        assert len(load_per_trade_log(tmpdir_clean)) == 2

        # 새 데이터 추가
        new_trade = [{"rec_date": "20260211", "code": "005930", "method": "RANK_SCORE",
                      "topk": 5, "horizon": 5, "score": 78.0, "entry_price": 61000,
                      "exit_price": 59000, "stop_price": 58000, "target_price": 65000,
                      "ret_pct": -3.28, "win": 0, "exit_type": "hold_close", "b_ratio": 1.33}]
        save_per_trade_log(tmpdir_clean, new_trade, "20260215")
        assert len(load_per_trade_log(tmpdir_clean)) == 3


# ═══════════════════════════════════════════════
#  2. 베이지안 승률
# ═══════════════════════════════════════════════
class TestBayesianWinRate:
    def test_all_wins_pulled_down(self):
        p = _bayesian_win_rate(np.ones(10), np.ones(10))
        assert p < 1.0 and p > 0.5

    def test_all_losses_pulled_up(self):
        p = _bayesian_win_rate(np.zeros(10), np.ones(10))
        assert p > 0.0 and p < PRIOR_WIN_RATE

    def test_empty_returns_prior(self):
        p = _bayesian_win_rate(np.array([]), np.array([]))
        assert abs(p - PRIOR_WIN_RATE) < 0.01


# ═══════════════════════════════════════════════
#  3. 시간 가중
# ═══════════════════════════════════════════════
class TestTimeWeight:
    def test_recency_ordering(self):
        dates = pd.Series(["20260220", "20260110", "20251001"])
        w = _time_weight(dates, half_life_days=90)
        assert w[0] > w[1] > w[2]
        assert all(w > 0)


# ═══════════════════════════════════════════════
#  4. 캘리브레이션 테이블 빌드
# ═══════════════════════════════════════════════
class TestCalibrationTable:
    @pytest.fixture()
    def cal_dir(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades(), "20260220")
        return tmpdir_clean

    def test_table_not_empty(self, cal_dir):
        cal = build_calibration_table(cal_dir)
        assert len(cal) > 0
        assert "p_calibrated" in cal.columns

    def test_higher_score_higher_winrate(self, cal_dir):
        cal = build_calibration_table(cal_dir)
        if len(cal) < 2:
            pytest.skip("not enough bins")
        lo = cal[cal["score_lo"] == 50]
        hi = cal[cal["score_hi"] > 90]
        if not lo.empty and not hi.empty:
            assert hi["p_calibrated"].iloc[0] > lo["p_calibrated"].iloc[0]

    def test_json_saved(self, cal_dir):
        build_calibration_table(cal_dir)
        assert os.path.exists(os.path.join(cal_dir, "calibration_table.json"))


# ═══════════════════════════════════════════════
#  5. calibrated_win_rate 조회
# ═══════════════════════════════════════════════
class TestCalibratedWinRate:
    @pytest.fixture()
    def cal_dir(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades(), "20260220")
        build_calibration_table(tmpdir_clean)
        return tmpdir_clean

    def test_valid_range(self, cal_dir):
        p = calibrated_win_rate(85.0, cal_dir, method="RANK_SCORE", horizon=5)
        assert 0 < p < 1

    def test_fallback_on_missing_dir(self):
        p = calibrated_win_rate(80.0, "/nonexistent/path")
        assert 0.3 <= p <= 0.85


# ═══════════════════════════════════════════════
#  6. kelly_fraction
# ═══════════════════════════════════════════════
class TestKellyFraction:
    def test_positive_edge(self):
        f = kelly_fraction(0.6, 2.0)
        assert f > 0 and f <= 0.25

    def test_bad_conditions_zero(self):
        assert kelly_fraction(0.3, 0.5) == 0.0

    def test_zero_prob_zero(self):
        assert kelly_fraction(0.0, 2.0) == 0.0


# ═══════════════════════════════════════════════
#  7. fallback 선형 추정
# ═══════════════════════════════════════════════
class TestFallbackLinear:
    def test_score_60(self):
        assert abs(_fallback_linear(60) - 0.4) < 0.05

    def test_score_80(self):
        assert abs(_fallback_linear(80) - 0.6) < 0.05

    def test_floor(self):
        assert _fallback_linear(0) >= 0.3

    def test_ceiling(self):
        assert _fallback_linear(100) <= 0.85


# ═══════════════════════════════════════════════
#  8. method/horizon 분리
# ═══════════════════════════════════════════════
class TestMethodSeparation:
    def test_ldy_vs_rank(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades(), "20260220")
        np.random.seed(99)
        ldy_trades = []
        for i in range(100):
            score = np.random.uniform(60, 95)
            win = 1 if np.random.random() < 0.35 else 0
            ldy_trades.append({
                "rec_date": f"2026{(i%12+1):02d}{(i%28+1):02d}",
                "code": f"{i+500:06d}", "method": "LDY_SCORE", "topk": 5,
                "horizon": 3, "score": round(score, 1), "entry_price": 10000,
                "exit_price": 10500 if win else 9500, "stop_price": 9500,
                "target_price": 11000, "ret_pct": 5.0 if win else -5.0,
                "win": win, "exit_type": "tp_hit" if win else "stop_hit", "b_ratio": 2.0,
            })
        save_per_trade_log(tmpdir_clean, ldy_trades, "20260220")
        cal = build_calibration_table(tmpdir_clean)
        methods = cal["method"].unique().tolist()
        assert "RANK_SCORE" in methods
        assert "LDY_SCORE" in methods


# ═══════════════════════════════════════════════
#  9. 룩어헤드 방지 (asof_ymd)
# ═══════════════════════════════════════════════
class TestLookAheadPrevention:
    def test_asof_reduces_n(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades(), "20260220")
        cal_early = build_calibration_table(tmpdir_clean, asof_ymd="20260115")
        cal_all = build_calibration_table(tmpdir_clean)
        if not cal_early.empty and not cal_all.empty:
            assert cal_early["n_raw"].sum() < cal_all["n_raw"].sum()

    def test_far_future_asof_empty(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades(), "20260220")
        cal = build_calibration_table(tmpdir_clean, asof_ymd="20250101")
        assert cal.empty


# ═══════════════════════════════════════════════
#  10. 재현성 (asof_date 고정)
# ═══════════════════════════════════════════════
class TestReproducibility:
    def test_same_asof_same_weights(self):
        dates = pd.Series(["20260210", "20260110", "20251001"])
        w1 = _time_weight(dates, half_life_days=90, asof_date="20260222")
        w2 = _time_weight(dates, half_life_days=90, asof_date="20260222")
        assert np.allclose(w1, w2)

    def test_different_asof_different_weights(self):
        dates = pd.Series(["20260210", "20260110", "20251001"])
        w1 = _time_weight(dates, half_life_days=90, asof_date="20260222")
        w3 = _time_weight(dates, half_life_days=90, asof_date="20260301")
        assert not np.allclose(w1, w3)


# ═══════════════════════════════════════════════
#  11. min_effective_n 적용
# ═══════════════════════════════════════════════
class TestMinEffectiveN:
    def test_high_min_n_empty(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades(), "20260220")
        cal = build_calibration_table(tmpdir_clean, min_effective_n=9999.0)
        assert cal.empty or len(cal) == 0

    def test_low_min_n_more_bins(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades(), "20260220")
        cal_all = build_calibration_table(tmpdir_clean)
        cal_loose = build_calibration_table(tmpdir_clean, min_effective_n=0.1)
        assert len(cal_loose) >= len(cal_all)


# ═══════════════════════════════════════════════
#  12. topk 중복키 분리
# ═══════════════════════════════════════════════
class TestTopkDedup:
    def test_different_topk_separate_rows(self, tmpdir_clean):
        trades = [
            {"rec_date": "20260210", "code": "005930", "method": "RANK_SCORE",
             "topk": 3, "horizon": 5, "score": 85.0, "entry_price": 60000,
             "exit_price": 63000, "stop_price": 57000, "target_price": 66000,
             "ret_pct": 5.0, "win": 1, "exit_type": "tp_hit", "b_ratio": 2.0},
            {"rec_date": "20260210", "code": "005930", "method": "RANK_SCORE",
             "topk": 5, "horizon": 5, "score": 85.0, "entry_price": 60000,
             "exit_price": 63000, "stop_price": 57000, "target_price": 66000,
             "ret_pct": 5.0, "win": 1, "exit_type": "tp_hit", "b_ratio": 2.0},
        ]
        save_per_trade_log(tmpdir_clean, trades, "20260215")
        df = pd.read_csv(os.path.join(tmpdir_clean, "per_trade_log.csv"))
        assert len(df) == 2


# ═══════════════════════════════════════════════
#  13-15. 캐시/보간/정규화
# ═══════════════════════════════════════════════
class TestCacheAndInterpolation:
    def test_load_asof_propagation(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades(), "20260220")
        kc._CAL_CACHE = None
        kc._CAL_CACHE_KEY = None
        cal_asof = kc.load_calibration_table(tmpdir_clean, asof_ymd="20260115", force_reload=True)
        cal_all = kc.load_calibration_table(tmpdir_clean, asof_ymd=None, force_reload=True)
        if not cal_asof.empty and not cal_all.empty:
            assert cal_asof["n_raw"].sum() < cal_all["n_raw"].sum()

    def test_interpolation_monotonicity(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades(), "20260220")
        build_calibration_table(tmpdir_clean)
        kc._CAL_CACHE = None
        kc._CAL_CACHE_KEY = None
        p70 = calibrated_win_rate(65.0, tmpdir_clean, method="RANK_SCORE", horizon=5)
        p80 = calibrated_win_rate(85.0, tmpdir_clean, method="RANK_SCORE", horizon=5)
        p75 = calibrated_win_rate(75.0, tmpdir_clean, method="RANK_SCORE", horizon=5)
        assert p75 > 0
        if p70 != p80:
            assert min(p70, p80) - 0.05 <= p75 <= max(p70, p80) + 0.05


# ═══════════════════════════════════════════════
#  14. asof_date 파싱 안전성
# ═══════════════════════════════════════════════
class TestAsofParsing:
    def test_yyyymmdd(self):
        dates = pd.Series(["20260210"])
        w = _time_weight(dates, 90, asof_date="20260222")
        assert np.isfinite(w[0])

    def test_dashed_format(self):
        dates = pd.Series(["20260210"])
        w1 = _time_weight(dates, 90, asof_date="20260222")
        w2 = _time_weight(dates, 90, asof_date="2026-02-22")
        assert np.allclose(w1, w2)

    def test_invalid_date_no_crash(self):
        dates = pd.Series(["20260210"])
        w = _time_weight(dates, 90, asof_date="not_a_date")
        assert np.isfinite(w[0])


# ═══════════════════════════════════════════════
#  16. asof_ymd 정규화
# ═══════════════════════════════════════════════
class TestNormalizeYmd:
    def test_yyyymmdd(self):
        assert _normalize_ymd("20260222") == "20260222"

    def test_dashed(self):
        assert _normalize_ymd("2026-02-22") == "20260222"

    def test_slashed(self):
        assert _normalize_ymd("2026/02/22") == "20260222"

    def test_none(self):
        assert _normalize_ymd(None) is None

    def test_cache_key_consistency(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades()[:50], "20260220")
        kc._CAL_CACHE = None
        kc._CAL_CACHE_KEY = None
        kc.load_calibration_table(tmpdir_clean, asof_ymd="2026-02-22", force_reload=True)
        key_a = kc._CAL_CACHE_KEY
        kc.load_calibration_table(tmpdir_clean, asof_ymd="20260222")
        key_b = kc._CAL_CACHE_KEY
        assert key_a == key_b


# ═══════════════════════════════════════════════
#  17. apply_kelly_calibrated asof_ymd
# ═══════════════════════════════════════════════
class TestApplyKellyCalibrated:
    def test_no_crash_with_asof(self, tmpdir_clean):
        save_per_trade_log(tmpdir_clean, _big_trades(), "20260220")
        build_calibration_table(tmpdir_clean)

        df = pd.DataFrame([{
            "TOTAL_SCORE": 80, "추천매수가": 10000, "손절가": 9500,
            "추천매도가1": 11000, "추천수량": 0, "추천금액(만원)": 0.0,
        }])
        kc._CAL_CACHE = None
        kc._CAL_CACHE_KEY = None
        result = apply_kelly_calibrated(df.copy(), tmpdir_clean, asof_ymd="20260221")
        assert result["켈리_수량"].iloc[0] >= 0
