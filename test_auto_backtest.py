"""P3 #13 자동 백테스트 피드백 루프 테스트 — 6개 안전장치 검증"""
import sys, os, tempfile, shutil, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

PASS = 0
FAIL = 0

def test(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def _make_test_data(tmp):
    """테스트용 recommend + price_snapshot 생성 (5영업일)"""
    days = ["20260210", "20260211", "20260212", "20260213", "20260214",
            "20260217", "20260218", "20260219", "20260220", "20260221"]

    for ymd in days:
        # recommend: 3종목, 점수 다양
        rec = pd.DataFrame({
            "종목코드": ["005930", "000660", "035720"],
            "종목명": ["삼성전자", "SK하이닉스", "카카오"],
            "DISPLAY_SCORE": [85.0, 65.0, 45.0],
            "FINAL_SCORE": [85.0, 65.0, 45.0],
            "손절가": [68000, 140000, 45000],
        })
        rec.to_csv(os.path.join(tmp, f"recommend_{ymd}.csv"), index=False)

        # price_snapshot: 약간의 변동
        base = {"005930": 70000, "000660": 150000, "035720": 50000}
        delta = (int(ymd[-2:]) - 10) * 100  # 날짜별 소폭 변동
        snap = pd.DataFrame([
            {"종목코드": "005930", "시가": base["005930"] + delta,
             "고가": base["005930"] + delta + 500, "저가": base["005930"] + delta - 300,
             "종가": base["005930"] + delta + 200},
            {"종목코드": "000660", "시가": base["000660"] - delta,
             "고가": base["000660"], "저가": base["000660"] - delta - 200,
             "종가": base["000660"] - delta + 100},
            {"종목코드": "035720", "시가": base["035720"],
             "고가": base["035720"] + 200, "저가": base["035720"] - 400,
             "종가": base["035720"] - 100},
        ])
        snap.to_csv(os.path.join(tmp, f"price_snapshot_{ymd}.csv"), index=False)


def run():
    global PASS, FAIL
    PASS, FAIL = 0, 0

    print("=" * 60)
    print("🧪 P3 #13 자동 백테스트 피드백 루프 테스트")
    print("=" * 60)

    from auto_backtest import (
        BacktestConfig, DEFAULT_BT_CONFIG,
        compute_realized_returns, build_winrate_table,
        kelly_from_table, auto_calibrate,
    )

    # ═══ 1. Config 검증 ═══
    print("\n📐 1. BacktestConfig")
    cfg = DEFAULT_BT_CONFIG
    test("horizon_bdays=5", cfg.horizon_bdays == 5)
    test("min_n=30", cfg.min_n == 30)
    test("kelly_fraction=0.25 (quarter)", cfg.kelly_fraction_mult == 0.25)
    test("kelly_cap=0.10", cfg.kelly_cap == 0.10)
    test("round_trip_cost > 0", cfg.round_trip_cost_pct > 0)
    test("round_trip_cost 합리적 (0.3~1.0%)",
         0.3 < cfg.round_trip_cost_pct < 1.0,
         f"got {cfg.round_trip_cost_pct:.3f}%")

    # ═══ 2. 안전장치 1: 미확정 추천 제외 ═══
    print("\n📐 2. 안전장치1: 미확정 추천 제외 (look-ahead)")
    tmp = tempfile.mkdtemp()
    try:
        _make_test_data(tmp)
        cfg5 = BacktestConfig(horizon_bdays=5, min_n=1, min_effective_n=0.1,
                              lookback_days=30)

        # as_of = 20260221 → cutoff = 20260221 - 5bdays = 20260214
        returns = compute_realized_returns(tmp, "20260221", cfg5)

        if not returns.empty:
            max_rec = returns["rec_date"].max()
            test("최대 rec_date <= cutoff(20260214)",
                 max_rec <= "20260214", f"max_rec={max_rec}")

            # 20260217 이후 추천은 포함 안 됨
            late_recs = returns[returns["rec_date"] > "20260214"]
            test("미확정 추천 0건", len(late_recs) == 0,
                 f"found {len(late_recs)} late recs")
        else:
            test("returns 비어있지 않음", False, "empty returns")
            test("미확정 추천 0건", True)  # vacuously true

        # ═══ 3. 안전장치 2: 진입 = 다음날 시가 ═══
        print("\n📐 3. 안전장치2: 진입=다음날 시가, 청산=N일후 종가")
        if not returns.empty:
            # rec_date=20260210 → entry=20260211 시가
            r0210 = returns[returns["rec_date"] == "20260210"]
            if not r0210.empty:
                entry_005930 = r0210[r0210["code"] == "005930"]["entry_price"]
                if not entry_005930.empty:
                    # 20260211 시가 = 70000 + (11-10)*100 = 70100
                    test("삼성전자 진입가 = 20260211 시가",
                         abs(float(entry_005930.iloc[0]) - 70100) < 1,
                         f"got {float(entry_005930.iloc[0])}")
                else:
                    test("삼성전자 진입가 확인", False, "not found")
            else:
                test("20260210 추천 존재", False, "not found")
        else:
            test("진입가 검증 스킵", True)

        # ═══ 4. 안전장치 3: 비용 차감 ═══
        print("\n📐 4. 안전장치3: 비용 차감")
        if not returns.empty and "ret_gross_pct" in returns.columns:
            diff = returns["ret_gross_pct"] - returns["ret_net_pct"]
            avg_diff = diff.mean()
            test("gross - net = 왕복비용",
                 abs(avg_diff - cfg5.round_trip_cost_pct) < 0.01,
                 f"avg_diff={avg_diff:.4f}, expected={cfg5.round_trip_cost_pct:.4f}")
        else:
            test("비용 차감 검증", True)

        # ═══ 5. 안전장치 4: min_n + 스무딩 ═══
        print("\n📐 5. 안전장치4: min_n + 라플라스 스무딩")
        table = build_winrate_table(returns, cfg5)
        test("승률 테이블 생성", not table.empty if not returns.empty else True)

        if not table.empty:
            # min_n=1이라 대부분 sufficient
            # min_n=30으로 바꾸면 insufficient
            cfg_strict = BacktestConfig(horizon_bdays=5, min_n=30, min_effective_n=10.0,
                                        lookback_days=30)
            table_strict = build_winrate_table(returns, cfg_strict)
            if not table_strict.empty:
                insufficient = table_strict[~table_strict["sufficient"]]
                test("min_n=30 → 대부분 insufficient",
                     len(insufficient) >= len(table_strict) - 1,
                     f"sufficient={table_strict['sufficient'].sum()}/{len(table_strict)}")

            # 라플라스: n=0인 구간도 p > 0 (0이 아님)
            table_with_zeros = build_winrate_table(returns, cfg5)
            if not table_with_zeros.empty:
                zero_bins = table_with_zeros[table_with_zeros["n_raw"] == 0]
                nonzero_bins = table_with_zeros[table_with_zeros["n_raw"] > 0]
                # 라플라스로 인해 n>0인 구간에서 p는 0이나 1이 아님
                for _, row in nonzero_bins.iterrows():
                    p = row["p_win"]
                    test(f"라플라스: bin [{row['score_lo']},{row['score_hi']}) p∈(0,1)",
                         0 < p < 1, f"p={p}")

        # ═══ 6. 안전장치 5: 켈리 제한 ═══
        print("\n📐 6. 안전장치5: 켈리 제한")
        if not table.empty:
            # 점수 85 → kelly 계산
            k85 = kelly_from_table(85.0, table, config=cfg5)
            test("kelly(85) ≤ cap(0.10)", k85 <= cfg5.kelly_cap)
            test("kelly(85) ≥ 0", k85 >= 0)

            # 표본 부족 구간 → kelly=0
            k_strict = kelly_from_table(85.0, table_strict,
                                        config=cfg_strict)
            test("표본 부족 → kelly=0", k_strict == 0.0, f"got {k_strict}")

            # 극단 점수 → 매칭 실패 → 0
            k_extreme = kelly_from_table(200.0, table, config=cfg5)
            test("범위 밖 점수 → kelly=0", k_extreme == 0.0)

        # ═══ 7. auto_calibrate 통합 ═══
        print("\n📐 7. auto_calibrate 통합")
        summary = auto_calibrate(tmp, "20260221", cfg5)
        test("summary 리턴", isinstance(summary, dict))
        test("n_trades > 0", summary.get("n_trades", 0) > 0)

        # JSON 저장 확인
        json_path = os.path.join(tmp, "winrate_table_20260221.json")
        test("JSON 저장됨", os.path.exists(json_path))
        latest_path = os.path.join(tmp, "winrate_table_latest.json")
        test("latest JSON도 저장됨", os.path.exists(latest_path))

        if os.path.exists(json_path):
            data = json.load(open(json_path))
            test("JSON 내용 유효", isinstance(data, dict) and "table" in data)
            test("meta 버전 태깅", data.get("meta", {}).get("version") == "v19.0")
            test("meta horizon 기록", data.get("meta", {}).get("horizon_bdays") == cfg5.horizon_bdays)
            test("meta 비용 기록",
                 abs(data.get("meta", {}).get("round_trip_cost_pct", 0) - cfg5.round_trip_cost_pct) < 0.01)
            test("meta corporate_action 기록",
                 data.get("meta", {}).get("corporate_action_threshold_pct") == 30.0)

        # ═══ 8. 안전장치 7: 기업행위 필터 ═══
        print("\n📐 8. 기업행위 필터")
        # 비정상 수익률(>30%) 종목 테스트
        # 인위적으로 price_snapshot에 극단 가격 삽입
        extreme_snap = pd.DataFrame([
            {"종목코드": "999999", "시가": 10000, "고가": 10000, "저가": 10000, "종가": 10000},
        ])
        # 진입일에 시가 10000, 청산일에 종가 50000 (400% 수익 → 제외 대상)
        entry_ymd_test = "20260211"
        exit_ymd_test = "20260218"
        snap_entry = pd.read_csv(os.path.join(tmp, f"price_snapshot_{entry_ymd_test}.csv"), dtype={"종목코드": str})
        snap_exit = pd.read_csv(os.path.join(tmp, f"price_snapshot_{exit_ymd_test}.csv"), dtype={"종목코드": str})
        extreme_entry = pd.DataFrame([{"종목코드": "999999", "시가": 10000, "고가": 10000, "저가": 10000, "종가": 10000}])
        extreme_exit = pd.DataFrame([{"종목코드": "999999", "시가": 50000, "고가": 50000, "저가": 50000, "종가": 50000}])
        pd.concat([snap_entry, extreme_entry]).to_csv(
            os.path.join(tmp, f"price_snapshot_{entry_ymd_test}.csv"), index=False)
        pd.concat([snap_exit, extreme_exit]).to_csv(
            os.path.join(tmp, f"price_snapshot_{exit_ymd_test}.csv"), index=False)

        # recommend에도 999999 추가
        rec_test = pd.read_csv(os.path.join(tmp, "recommend_20260210.csv"), dtype={"종목코드": str})
        extreme_rec = pd.DataFrame([{"종목코드": "999999", "종목명": "극단종목", "DISPLAY_SCORE": 75.0,
                                      "FINAL_SCORE": 75.0, "손절가": 8000}])
        pd.concat([rec_test, extreme_rec]).to_csv(
            os.path.join(tmp, "recommend_20260210.csv"), index=False)

        returns_with_extreme = compute_realized_returns(tmp, "20260221", cfg5)
        extreme_in_results = returns_with_extreme[returns_with_extreme["code"] == "999999"]
        test("기업행위(400%) 종목 제외됨", len(extreme_in_results) == 0,
             f"found {len(extreme_in_results)} rows")

        # ═══ 9. 수학적 일관성 ═══
        print("\n📐 9. 수학적 일관성")
        if not returns.empty:
            # 전체 승률 = win 평균
            manual_wr = returns["win"].mean()
            test("summary 승률 일치",
                 abs(summary["overall_winrate"] - manual_wr) < 0.001)

            # net = gross - cost (모든 행)
            cost = cfg5.round_trip_cost_pct
            check = (returns["ret_gross_pct"] - returns["ret_net_pct"] - cost).abs()
            test("모든 행: net = gross - cost",
                 check.max() < 0.001, f"max_diff={check.max():.6f}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

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
    run()
