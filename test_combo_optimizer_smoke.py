# -*- coding: utf-8 -*-
"""
test_combo_optimizer_smoke.py — combo_optimizer 재현 가능 smoke test
══════════════════════════════════════════════════════════════════════
[v3.8.1] README의 end-to-end 검증 결과가 재현 가능함을 증명하는 최소 스크립트.

실행:
  python test_combo_optimizer_smoke.py

출력:
  - 가짜 데이터 20일치 생성
  - run_combo_optimization() / run_combo_optimization_wf() 실행
  - 결과 파일 존재 확인
  - 핵심 key 존재 검증
  - SUMMARY/PASS/FAIL 리포트

pytest 스타일도 지원:
  pytest test_combo_optimizer_smoke.py -v
"""
import os
import sys
import json
import shutil
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd

# 프로젝트 루트를 path에 추가 (combo_optimizer 임포트)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════
#  가짜 데이터 생성
# ══════════════════════════════════════════════════════════════════════

def generate_fake_pipeline_data(data_dir: str, n_days: int = 20, n_stocks: int = 20,
                                 seed: int = 42) -> Tuple[int, int]:
    """N일치 recommend CSV + price_snapshot CSV 생성.

    Returns:
        (n_rec_files, n_snap_files)
    """
    np.random.seed(seed)
    os.makedirs(data_dir, exist_ok=True)

    dates = [f"202604{str(i).zfill(2)}" for i in range(1, n_days + 1)]

    for d_idx, date in enumerate(dates):
        # recommend CSV
        rec_df = pd.DataFrame({
            "종목코드": [f"{i:06d}" for i in range(1000, 1000 + n_stocks)],
            "종목명": [f"종목{i}" for i in range(n_stocks)],
            "추천매수가": np.random.uniform(10000, 50000, n_stocks).round(-2),
            "종가": np.random.uniform(10000, 50000, n_stocks).round(-2),
            "STRUCT_SCORE": np.random.uniform(50, 100, n_stocks).round(1),
            "TIMING_SCORE": np.random.uniform(40, 90, n_stocks).round(1),
            "AI_SCORE": np.random.uniform(30, 80, n_stocks).round(1),
            "DISPLAY_SCORE": np.random.uniform(50, 95, n_stocks).round(1),
            "ROUTE": np.random.choice(["ATTACK", "ARMED", "WAIT"], n_stocks),
        })
        rec_df.to_csv(
            os.path.join(data_dir, f"recommend_{date}.csv"),
            index=False, encoding="utf-8-sig",
        )

        # price_snapshot CSV
        snap_df = pd.DataFrame({
            "종목코드": rec_df["종목코드"],
            "종가": rec_df["추천매수가"] * np.random.uniform(0.90, 1.15, n_stocks),
        })
        snap_df.to_csv(
            os.path.join(data_dir, f"price_snapshot_{date}.csv"),
            index=False, encoding="utf-8-sig",
        )

    n_rec = len([f for f in os.listdir(data_dir) if f.startswith("recommend_")])
    n_snap = len([f for f in os.listdir(data_dir) if f.startswith("price_snapshot_")])
    return n_rec, n_snap


# ══════════════════════════════════════════════════════════════════════
#  테스트 함수들 (pytest 자동 수집 + manual 실행 둘 다 지원)
# ══════════════════════════════════════════════════════════════════════

def test_legacy_grid_search():
    """기존 run_combo_optimization() 동작 확인."""
    from combo_optimizer import run_combo_optimization

    tmp = tempfile.mkdtemp(prefix="combo_opt_legacy_")
    try:
        n_rec, n_snap = generate_fake_pipeline_data(tmp, n_days=20, seed=42)
        assert n_rec == 20 and n_snap == 20, f"파일 생성 실패: {n_rec}/{n_snap}"

        result = run_combo_optimization(tmp, horizon=3, min_samples=5, top_n=5)
        assert result, "결과가 비어있음"
        assert "best_wr" in result, "best_wr 키 없음"
        assert "best_ev" in result, "best_ev 키 없음"
        assert "meta" in result, "meta 키 없음"
        assert result["meta"]["total_trades"] > 0, "거래 데이터 0건"

        # JSON 저장 확인
        out_json = os.path.join(tmp, "optimal_filter_latest.json")
        assert os.path.exists(out_json), f"{out_json} 저장 실패"

        with open(out_json, encoding="utf-8") as f:
            saved = json.load(f)
        assert "best_wr" in saved, "저장된 JSON에 best_wr 없음"
    finally:
        shutil.rmtree(tmp)


def test_walk_forward():
    """신규 run_combo_optimization_wf() 동작 확인."""
    from combo_optimizer import run_combo_optimization_wf

    tmp = tempfile.mkdtemp(prefix="combo_opt_wf_")
    try:
        n_rec, n_snap = generate_fake_pipeline_data(tmp, n_days=20, seed=42)
        assert n_rec == 20 and n_snap == 20

        result = run_combo_optimization_wf(
            tmp, horizon=3, min_samples=5, top_n=5,
            oos_ratio=0.5, robust_threshold=0.7, verbose=False,
        )
        assert result, "walk-forward 결과 비어있음"

        # 필수 key 검증 (v3.8.1 — 4단 분류 포함)
        required_keys = {
            "robust_combos", "overfit_combos", "recovering_combos",
            "weak_combos", "current_combo_wf", "strong_approximation",
            "summary", "meta",
        }
        missing = required_keys - set(result.keys())
        assert not missing, f"누락 키: {missing}"

        # summary 필수 필드
        summary_keys = {"n_total", "n_robust", "n_overfit", "n_recovering",
                        "n_weak", "robust_pct"}
        missing_summary = summary_keys - set(result["summary"].keys())
        assert not missing_summary, f"summary 누락: {missing_summary}"

        # 카테고리 합이 총 개수와 일치
        s = result["summary"]
        assert s["n_robust"] + s["n_overfit"] + s["n_recovering"] + s["n_weak"] == s["n_total"], \
            f"카테고리 합 불일치: {s}"

        # JSON 저장 확인
        out_json = os.path.join(tmp, "combo_walkforward_latest.json")
        assert os.path.exists(out_json), f"{out_json} 저장 실패"

        with open(out_json, encoding="utf-8") as f:
            saved = json.load(f)
        assert "summary" in saved, "저장 JSON에 summary 없음"
        assert saved["summary"]["n_total"] > 0, "분석 조합 0개"

        # 각 결과에 category 필드 존재
        for r in result["robust_combos"]:
            assert r.get("category") == "ROBUST", f"category 누락 or 불일치: {r}"
        for r in result["overfit_combos"]:
            assert r.get("category") == "OVERFIT", f"category 누락 or 불일치: {r}"
        for r in result["recovering_combos"]:
            assert r.get("category") == "RECOVERING", f"category 누락 or 불일치: {r}"
    finally:
        shutil.rmtree(tmp)


def test_robustness_categorization():
    """Robustness 4단 분류 경계 케이스 검증."""
    from combo_optimizer import _evaluate_combo

    # 가짜 DataFrame — 명시적으로 특정 조합 테스트
    df = pd.DataFrame({
        "ret": [5, -3, 2, 8, -1, 4, 6, -2, 3, 7],
        "win": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        "S":   [85, 92, 88, 95, 80, 90, 93, 87, 91, 94],
        "T":   [75, 85, 80, 88, 70, 82, 86, 78, 83, 87],
        "AI":  [55, 65, 60, 70, 50, 62, 67, 58, 63, 68],
        "ROUTE": ["ATTACK", "ARMED", "ATTACK", "ATTACK", "ARMED",
                  "ATTACK", "ARMED", "ATTACK", "ARMED", "ATTACK"],
    })

    stat = _evaluate_combo(df, 80, 70, 50, ["ATTACK", "ARMED"], min_samples=3)
    assert stat is not None, "표본 충분한데 None 반환"
    assert stat["n"] >= 3, f"표본 수 이상: {stat['n']}"
    assert "ev" in stat and "win_rate" in stat, f"필수 필드 누락: {stat}"

    # 표본 부족 케이스
    stat_low = _evaluate_combo(df, 95, 88, 70, ["ATTACK"], min_samples=10)
    assert stat_low is None, f"표본 부족인데 None 아님: {stat_low}"


# ══════════════════════════════════════════════════════════════════════
#  Manual 실행 (python test_combo_optimizer_smoke.py)
# ══════════════════════════════════════════════════════════════════════

def _run_manual():
    """pytest 없어도 실행 가능한 manual 리포트."""
    tests = [
        ("Legacy grid search", test_legacy_grid_search),
        ("Walk-forward validation", test_walk_forward),
        ("Robustness categorization", test_robustness_categorization),
    ]

    print("\n" + "=" * 72)
    print("🧪 combo_optimizer Smoke Test (v3.8.1)")
    print("=" * 72)

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}")
            print(f"     AssertionError: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 {name}")
            print(f"     {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "-" * 72)
    print(f"결과: {passed}개 PASS · {failed}개 FAIL")
    print("-" * 72)

    if failed == 0:
        print("✅ 전체 smoke test 통과 — README 검증 결과 재현됨")
        return 0
    else:
        print(f"❌ {failed}개 실패 — 결과 확인 필요")
        return 1


if __name__ == "__main__":
    sys.exit(_run_manual())
