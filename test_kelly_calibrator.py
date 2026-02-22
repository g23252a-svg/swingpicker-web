"""Kelly Calibrator 단위 테스트"""

import sys, os
import numpy as np
import pandas as pd
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from kelly_calibrator import (
    save_per_trade_log,
    build_calibration_table,
    calibrated_win_rate,
    kelly_fraction,
    _bayesian_win_rate,
    _time_weight,
    _fallback_linear,
    PRIOR_WIN_RATE,
)

PASS = 0
FAIL = 0

def test(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def run_tests():
    global PASS, FAIL
    PASS, FAIL = 0, 0
    tmpdir = tempfile.mkdtemp()

    print("=" * 60)
    print("🧪 Kelly Calibrator 테스트")
    print("=" * 60)

    # ── 1. per-trade 로그 저장 ──
    print("\n📐 1. per-trade 로그 저장")
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
    path = save_per_trade_log(tmpdir, trades, "20260215")
    test("파일 생성", os.path.exists(path))

    df_log = pd.read_csv(path)
    test("2건 저장", len(df_log) == 2)

    # append 테스트 (중복 제거)
    save_per_trade_log(tmpdir, trades, "20260215")  # 동일 데이터
    df_log2 = pd.read_csv(path)
    test("중복 제거 (여전히 2건)", len(df_log2) == 2)

    # 새 데이터 추가
    new_trade = [
        {"rec_date": "20260211", "code": "005930", "method": "RANK_SCORE",
         "topk": 5, "horizon": 5, "score": 78.0, "entry_price": 61000,
         "exit_price": 59000, "stop_price": 58000, "target_price": 65000,
         "ret_pct": -3.28, "win": 0, "exit_type": "hold_close", "b_ratio": 1.33},
    ]
    save_per_trade_log(tmpdir, new_trade, "20260215")
    df_log3 = pd.read_csv(path)
    test("append → 3건", len(df_log3) == 3)

    # ── 2. 베이지안 승률 ──
    print("\n📐 2. 베이지안 승률")
    # 100% 승률 표본 10개 + prior → 45% 쪽으로 당겨짐
    wins_all = np.ones(10)
    weights_eq = np.ones(10)
    p_bay = _bayesian_win_rate(wins_all, weights_eq)
    test("100% 승률+prior → < 1.0", p_bay < 1.0, f"p={p_bay:.3f}")
    test("100% 승률+prior → > 0.5", p_bay > 0.5, f"p={p_bay:.3f}")

    # 0% 승률 표본 10개 → prior 쪽으로 당겨짐
    wins_none = np.zeros(10)
    p_bay0 = _bayesian_win_rate(wins_none, weights_eq)
    test("0% 승률+prior → > 0", p_bay0 > 0.0, f"p={p_bay0:.3f}")
    test("0% 승률+prior → < 0.45", p_bay0 < PRIOR_WIN_RATE, f"p={p_bay0:.3f}")

    # 표본 0개 → prior 그대로
    p_empty = _bayesian_win_rate(np.array([]), np.array([]))
    test("표본 0 → prior", abs(p_empty - PRIOR_WIN_RATE) < 0.01, f"p={p_empty:.3f}")

    # ── 3. 시간 가중 ──
    print("\n📐 3. 시간 가중")
    dates = pd.Series(["20260220", "20260110", "20251001"])
    w = _time_weight(dates, half_life_days=90)
    test("최근 > 오래된", w[0] > w[1] > w[2], f"weights={w}")
    test("가중치 > 0", all(w > 0))

    # ── 4. 캘리브레이션 테이블 빌드 ──
    print("\n📐 4. 캘리브레이션 테이블 빌드")
    # 충분한 표본 생성
    np.random.seed(42)
    big_trades = []
    for i in range(200):
        score = np.random.uniform(50, 100)
        win = 1 if np.random.random() < (0.3 + score * 0.005) else 0
        big_trades.append({
            "rec_date": f"2026{(i%12+1):02d}{(i%28+1):02d}",
            "code": f"{i:06d}",
            "method": "RANK_SCORE",
            "topk": 5,
            "horizon": 5,
            "score": round(score, 1),
            "entry_price": 10000,
            "exit_price": 10500 if win else 9500,
            "stop_price": 9500,
            "target_price": 11000,
            "ret_pct": 5.0 if win else -5.0,
            "win": win,
            "exit_type": "tp_hit" if win else "stop_hit",
            "b_ratio": 2.0,
        })
    tmpdir2 = tempfile.mkdtemp()
    save_per_trade_log(tmpdir2, big_trades, "20260220")
    cal = build_calibration_table(tmpdir2)
    test("테이블 비어있지 않음", len(cal) > 0, f"len={len(cal)}")
    test("p_calibrated 컬럼 존재", "p_calibrated" in cal.columns)

    # 높은 점수 구간이 낮은 점수 구간보다 승률 높아야 함
    if len(cal) >= 2:
        lo_bin = cal[cal["score_lo"] == 50]
        hi_bin = cal[cal["score_hi"] > 90]
        if len(lo_bin) > 0 and len(hi_bin) > 0:
            p_lo = lo_bin["p_calibrated"].iloc[0]
            p_hi = hi_bin["p_calibrated"].iloc[0]
            test("고점수 p > 저점수 p", p_hi > p_lo,
                 f"hi={p_hi:.3f}, lo={p_lo:.3f}")

    # JSON 저장 확인
    json_path = os.path.join(tmpdir2, "calibration_table.json")
    test("JSON 저장", os.path.exists(json_path))

    # ── 5. calibrated_win_rate 조회 ──
    print("\n📐 5. calibrated_win_rate 조회")
    p85 = calibrated_win_rate(85.0, tmpdir2, method="RANK_SCORE", horizon=5)
    test("p(85) > 0", p85 > 0, f"p={p85:.3f}")
    test("p(85) < 1", p85 < 1, f"p={p85:.3f}")

    # 캘리브레이션 없는 디렉토리 → fallback
    p_fb = calibrated_win_rate(80.0, "/nonexistent/path")
    test("fallback 사용", 0.3 <= p_fb <= 0.85, f"p={p_fb:.3f}")

    # ── 6. kelly_fraction ──
    print("\n📐 6. kelly_fraction")
    f1 = kelly_fraction(0.6, 2.0)  # p=60%, b=2
    test("f > 0", f1 > 0, f"f={f1:.3f}")
    test("f ≤ 0.25", f1 <= 0.25)

    f_bad = kelly_fraction(0.3, 0.5)  # 승률 낮고 손익비 낮으면 0
    test("나쁜 조건 → f=0", f_bad == 0.0, f"f={f_bad:.3f}")

    f_zero = kelly_fraction(0.0, 2.0)
    test("p=0 → f=0", f_zero == 0.0)

    # ── 7. fallback 선형 추정 ──
    print("\n📐 7. fallback 선형 추정 (하위호환)")
    test("score=60 → ~0.4", abs(_fallback_linear(60) - 0.4) < 0.05)
    test("score=80 → ~0.6", abs(_fallback_linear(80) - 0.6) < 0.05)
    test("score=0 → ≥ 0.3", _fallback_linear(0) >= 0.3)
    test("score=100 → ≤ 0.85", _fallback_linear(100) <= 0.85)

    # ── 8. method/horizon 분리 ──
    print("\n📐 8. method/horizon 분리")
    # LDY_SCORE 데이터 추가
    ldy_trades = []
    for i in range(100):
        score = np.random.uniform(60, 95)
        win = 1 if np.random.random() < 0.35 else 0  # 의도적으로 낮은 승률
        ldy_trades.append({
            "rec_date": f"2026{(i%12+1):02d}{(i%28+1):02d}",
            "code": f"{i+500:06d}",
            "method": "LDY_SCORE",
            "topk": 5,
            "horizon": 3,
            "score": round(score, 1),
            "entry_price": 10000,
            "exit_price": 10500 if win else 9500,
            "stop_price": 9500,
            "target_price": 11000,
            "ret_pct": 5.0 if win else -5.0,
            "win": win,
            "exit_type": "tp_hit" if win else "stop_hit",
            "b_ratio": 2.0,
        })
    save_per_trade_log(tmpdir2, ldy_trades, "20260220")
    cal2 = build_calibration_table(tmpdir2)
    methods_in_cal = cal2["method"].unique().tolist()
    test("RANK_SCORE in table", "RANK_SCORE" in methods_in_cal)
    test("LDY_SCORE in table", "LDY_SCORE" in methods_in_cal)

    # LDY(낮은 승률) < RANK(높은 승률) — 같은 점수대 비교
    rank_80 = cal2[(cal2["method"] == "RANK_SCORE") & (cal2["score_lo"] == 80)]
    ldy_80 = cal2[(cal2["method"] == "LDY_SCORE") & (cal2["score_lo"] == 80)]
    if len(rank_80) > 0 and len(ldy_80) > 0:
        test("RANK p > LDY p (같은 점수대)",
             rank_80["p_calibrated"].iloc[0] > ldy_80["p_calibrated"].iloc[0],
             f"rank={rank_80['p_calibrated'].iloc[0]:.3f}, ldy={ldy_80['p_calibrated'].iloc[0]:.3f}")

    # ── 9. 룩어헤드 방지 (asof_ymd) ──
    print("\n📐 9. 룩어헤드 방지 (asof_ymd)")
    # tmpdir2에 20260101~20260228 데이터가 있음
    # asof=20260115 → 1월14일 이전만 사용
    cal_early = build_calibration_table(tmpdir2, asof_ymd="20260115")
    cal_all = build_calibration_table(tmpdir2)  # asof 없음 = 전체
    if not cal_early.empty and not cal_all.empty:
        n_early = cal_early["n_raw"].sum()
        n_all = cal_all["n_raw"].sum()
        test("asof 필터 → n 감소", n_early < n_all,
             f"early={n_early}, all={n_all}")
    else:
        test("asof 필터 → 빈 테이블 (기대대로)", cal_early.empty)

    # 미래만 있는 asof → 빈 테이블
    cal_future = build_calibration_table(tmpdir2, asof_ymd="20250101")
    test("미래 asof → 빈 테이블", cal_future.empty)

    # ── 10. 재현성 (_time_weight asof_date 고정) ──
    print("\n📐 10. 재현성 (asof_date 고정)")
    dates10 = pd.Series(["20260210", "20260110", "20251001"])
    w1 = _time_weight(dates10, half_life_days=90, asof_date="20260222")
    w2 = _time_weight(dates10, half_life_days=90, asof_date="20260222")
    test("동일 asof → 동일 가중치", np.allclose(w1, w2))

    # 다른 asof → 다른 가중치
    w3 = _time_weight(dates10, half_life_days=90, asof_date="20260301")
    test("다른 asof → 다른 가중치", not np.allclose(w1, w3))

    # ── 11. min_effective_n 적용 ──
    print("\n📐 11. min_effective_n 적용")
    # 매우 높은 min_effective_n → 대부분 제외
    cal_strict = build_calibration_table(tmpdir2, min_effective_n=9999.0)
    test("높은 min_n → 빈 테이블", cal_strict.empty or len(cal_strict) == 0)

    # 낮은 min_effective_n → 더 많은 구간
    cal_loose = build_calibration_table(tmpdir2, min_effective_n=0.1)
    test("낮은 min_n → 구간 많음", len(cal_loose) >= len(cal_all),
         f"loose={len(cal_loose)}, all={len(cal_all)}")

    # ── 12. topk 중복키 분리 ──
    print("\n📐 12. topk 중복키 분리")
    topk_trades = [
        {"rec_date": "20260210", "code": "005930", "method": "RANK_SCORE",
         "topk": 3, "horizon": 5, "score": 85.0, "entry_price": 60000,
         "exit_price": 63000, "stop_price": 57000, "target_price": 66000,
         "ret_pct": 5.0, "win": 1, "exit_type": "tp_hit", "b_ratio": 2.0},
        {"rec_date": "20260210", "code": "005930", "method": "RANK_SCORE",
         "topk": 5, "horizon": 5, "score": 85.0, "entry_price": 60000,
         "exit_price": 63000, "stop_price": 57000, "target_price": 66000,
         "ret_pct": 5.0, "win": 1, "exit_type": "tp_hit", "b_ratio": 2.0},
    ]
    tmpdir3 = tempfile.mkdtemp()
    save_per_trade_log(tmpdir3, topk_trades, "20260215")
    df_topk = pd.read_csv(os.path.join(tmpdir3, "per_trade_log.csv"))
    test("topk 다르면 2건 유지", len(df_topk) == 2,
         f"len={len(df_topk)}")
    shutil.rmtree(tmpdir3, ignore_errors=True)

    # ── 13. load_calibration_table asof_ymd 전파 ──
    print("\n📐 13. load_calibration_table asof_ymd 전파")
    # tmpdir2는 아직 살아있음 (big_trades + ldy_trades)
    # 캐시 무효화
    import kelly_calibrator as kc
    kc._CAL_CACHE = None
    kc._CAL_CACHE_KEY = None

    cal_load_asof = kc.load_calibration_table(tmpdir2, asof_ymd="20260115", force_reload=True)
    cal_load_all = kc.load_calibration_table(tmpdir2, asof_ymd=None, force_reload=True)
    if not cal_load_asof.empty and not cal_load_all.empty:
        test("load asof → n 감소",
             cal_load_asof["n_raw"].sum() < cal_load_all["n_raw"].sum())
    else:
        test("load asof → 빈 (기대대로)", cal_load_asof.empty)

    # 날짜별 스냅샷 저장 확인
    snap_path = os.path.join(tmpdir2, "calibration_table_20260115.json")
    # 빈 테이블이면 스냅샷 안 만들어짐 → 존재 여부는 테이블 비어있으면 스킵
    if not cal_load_asof.empty:
        test("날짜별 스냅샷 저장", os.path.exists(snap_path))

    # ── 14. asof_date 파싱 안전성 ──
    print("\n📐 14. asof_date 파싱 안전성")
    # 다양한 포맷 테스트
    dates14 = pd.Series(["20260210"])
    w_yyyymmdd = _time_weight(dates14, 90, asof_date="20260222")
    w_dashed = _time_weight(dates14, 90, asof_date="2026-02-22")
    test("YYYYMMDD 파싱", np.isfinite(w_yyyymmdd[0]))
    test("YYYY-MM-DD 파싱", np.isfinite(w_dashed[0]))
    test("두 포맷 동일 결과", np.allclose(w_yyyymmdd, w_dashed))

    # 잘못된 날짜 → now() fallback (크래시 없음)
    w_bad = _time_weight(dates14, 90, asof_date="not_a_date")
    test("잘못된 날짜 → 크래시 없음", np.isfinite(w_bad[0]))

    # ── 15. 선형 보간 (계단→연속) ──
    print("\n📐 15. 선형 보간 (구간 경계)")
    kc._CAL_CACHE = None
    kc._CAL_CACHE_KEY = None
    # 70점과 80점 구간 경계(75점)에서 보간이 되는지
    p70 = calibrated_win_rate(65.0, tmpdir2, method="RANK_SCORE", horizon=5)
    p80 = calibrated_win_rate(85.0, tmpdir2, method="RANK_SCORE", horizon=5)
    p75 = calibrated_win_rate(75.0, tmpdir2, method="RANK_SCORE", horizon=5)
    test("p(75) 존재", p75 > 0)
    # 보간이면 중간값이어야 함 (또는 bin 매칭이면 그 bin의 값)
    if p70 != p80:
        # 완전 동일하지 않다면 보간 효과
        test("p 단조성 (70≤75≤80 또는 근처)",
             min(p70, p80) - 0.05 <= p75 <= max(p70, p80) + 0.05,
             f"p70={p70:.3f}, p75={p75:.3f}, p80={p80:.3f}")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    shutil.rmtree(tmpdir2, ignore_errors=True)

    # ── 16. asof_ymd 정규화 (캐시키/파일명 통일) ──
    print("\n📐 16. asof_ymd 정규화")
    from kelly_calibrator import _normalize_ymd
    test("YYYYMMDD → 그대로", _normalize_ymd("20260222") == "20260222")
    test("YYYY-MM-DD → YYYYMMDD", _normalize_ymd("2026-02-22") == "20260222")
    test("YYYY/MM/DD → YYYYMMDD", _normalize_ymd("2026/02/22") == "20260222")
    test("None → None", _normalize_ymd(None) is None)

    # 캐시키 동일성: 다른 포맷이지만 같은 날짜
    tmpdir4 = tempfile.mkdtemp()
    save_per_trade_log(tmpdir4, big_trades[:50], "20260220")
    kc._CAL_CACHE = None
    kc._CAL_CACHE_KEY = None
    cal_a = kc.load_calibration_table(tmpdir4, asof_ymd="2026-02-22", force_reload=True)
    key_a = kc._CAL_CACHE_KEY
    cal_b = kc.load_calibration_table(tmpdir4, asof_ymd="20260222")  # 캐시 히트 기대
    key_b = kc._CAL_CACHE_KEY
    test("다른 포맷 → 동일 캐시키", key_a == key_b,
         f"a={key_a}, b={key_b}")
    shutil.rmtree(tmpdir4, ignore_errors=True)

    # ── 17. apply_kelly_calibrated asof_ymd 전달 ──
    print("\n📐 17. apply_kelly_calibrated asof_ymd")
    from kelly_calibrator import apply_kelly_calibrated
    tmpdir5 = tempfile.mkdtemp()
    save_per_trade_log(tmpdir5, big_trades, "20260220")
    build_calibration_table(tmpdir5)

    test_df = pd.DataFrame([{
        "TOTAL_SCORE": 80, "추천매수가": 10000, "손절가": 9500,
        "추천매도가1": 11000, "추천수량": 0, "추천금액(만원)": 0.0,
    }])
    kc._CAL_CACHE = None
    kc._CAL_CACHE_KEY = None
    result_df = apply_kelly_calibrated(test_df.copy(), tmpdir5, asof_ymd="20260221")
    test("asof 전달 → 켈리 수량 계산됨", result_df["켈리_수량"].iloc[0] >= 0)
    test("크래시 없음", True)
    shutil.rmtree(tmpdir5, ignore_errors=True)

    # ── 결과 ──
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"🏁 결과: {PASS}/{total} 통과 ({FAIL} 실패)")
    if FAIL > 0:
        print("⚠️ 실패 항목이 있습니다!")
        sys.exit(1)
    else:
        print("🏆 ALL PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
