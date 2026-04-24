# -*- coding: utf-8 -*-
"""
test_monotonicity_report_v22.py — daily_briefing.generate_monotonicity_report 스모크
[v22] CI gate 운영 안정성 보장 — fixture로 happy/HARD-fail/edge 케이스 검증.
"""
import json
import os
import pandas as pd
import pytest


def _write_recommend(path: str, rows):
    """recommend_latest.csv fixture 생성"""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_winrate_table(path: str, table, sufficient=True):
    """winrate_table_by_ELITE_SCORE_latest.json fixture 생성"""
    obj = {"meta": {"is_sufficient": sufficient}, "table": table}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


# ════════════════════════════════════════════════════════════
#  Happy path — 모든 데이터 정상 + HARD PASS / SOFT OK
# ════════════════════════════════════════════════════════════

def test_happy_path_all_pass(tmp_path):
    """정상 데이터 → HARD PASS, SOFT OK"""
    from daily_briefing import generate_monotonicity_report

    # recommend_latest: TOP_PICK 2건 모두 ATTACK/ARMED + TP1>0
    _write_recommend(tmp_path / "recommend_latest.csv", [
        {"종목코드": "005930", "ROUTE": "ATTACK", "TOP_PICK": 1,
         "TP1_PCT": 15.2, "EST_WIN_RATE": 0.62},
        {"종목코드": "000660", "ROUTE": "ARMED", "TOP_PICK": 1,
         "TP1_PCT": 10.5, "EST_WIN_RATE": 0.58},
        {"종목코드": "373220", "ROUTE": "WAIT", "TOP_PICK": 0,
         "TP1_PCT": 8.0, "EST_WIN_RATE": 0.45},
    ])

    # winrate_table: 단조 증가 + 양수 excess
    _write_winrate_table(tmp_path / "winrate_table_by_ELITE_SCORE_latest.json", [
        {"score_lo": 0, "score_hi": 50, "p_win": 0.40, "n_raw": 100,
         "sufficient": True, "avg_ret_net_pct": -0.5,
         "avg_ret_excess_pct": 0.3},
        {"score_lo": 50, "score_hi": 70, "p_win": 0.52, "n_raw": 150,
         "sufficient": True, "avg_ret_net_pct": 1.2,
         "avg_ret_excess_pct": 1.5},
        {"score_lo": 70, "score_hi": 90, "p_win": 0.65, "n_raw": 80,
         "sufficient": True, "avg_ret_net_pct": 3.5,
         "avg_ret_excess_pct": 3.8},
    ])

    report = generate_monotonicity_report(str(tmp_path), "20260424")

    assert report["ci_hard_all_pass"] is True
    assert report["top_pick_count"] == 2
    # 두 축 모두 계산됐는지
    assert report["declared_wr_top_pick"] is not None
    assert report["declared_wr_active"] is not None
    assert report["declared_vs_realized_gap_top_pick"] is not None
    # 호환 alias
    assert report["declared_vs_realized_gap"] == report["declared_vs_realized_gap_top_pick"]

    # 파일 저장됐는지
    assert (tmp_path / "monotonicity_report_20260424.json").exists()
    assert (tmp_path / "monotonicity_report_latest.json").exists()


# ════════════════════════════════════════════════════════════
#  HARD FAIL — TOP_PICK이 ROUTE=WAIT (positive gate 누출)
# ════════════════════════════════════════════════════════════

def test_hard_fail_route_leak(tmp_path):
    """TOP_PICK=1인데 ROUTE=WAIT → top_pick_route_positive FAIL"""
    from daily_briefing import generate_monotonicity_report

    _write_recommend(tmp_path / "recommend_latest.csv", [
        {"종목코드": "005930", "ROUTE": "WAIT", "TOP_PICK": 1,
         "TP1_PCT": 12.0, "EST_WIN_RATE": 0.55},
    ])

    report = generate_monotonicity_report(str(tmp_path), "20260424")

    assert report["ci_hard_all_pass"] is False
    fail_gates = [g for g in report["ci_hard"] if g["status"] == "FAIL"]
    assert any(g["gate"] == "top_pick_route_positive" for g in fail_gates)


# ════════════════════════════════════════════════════════════
#  HARD FAIL — TP1_PCT <= 0
# ════════════════════════════════════════════════════════════

def test_hard_fail_negative_tp1(tmp_path):
    """TOP_PICK=1인데 TP1_PCT=0 → top_pick_tp1_positive FAIL"""
    from daily_briefing import generate_monotonicity_report

    _write_recommend(tmp_path / "recommend_latest.csv", [
        {"종목코드": "005930", "ROUTE": "ATTACK", "TOP_PICK": 1,
         "TP1_PCT": 0.0, "EST_WIN_RATE": 0.55},
    ])

    report = generate_monotonicity_report(str(tmp_path), "20260424")
    fail_gates = [g for g in report["ci_hard"] if g["status"] == "FAIL"]
    assert any(g["gate"] == "top_pick_tp1_positive" for g in fail_gates)


# ════════════════════════════════════════════════════════════
#  HARD FAIL — 선언-실현 갭 > 15%p (TOP_PICK 기준)
# ════════════════════════════════════════════════════════════

def test_hard_fail_large_gap(tmp_path):
    """TOP_PICK 선언 0.75 vs 실현 0.40 → gap=35%p FAIL"""
    from daily_briefing import generate_monotonicity_report

    _write_recommend(tmp_path / "recommend_latest.csv", [
        {"종목코드": "005930", "ROUTE": "ATTACK", "TOP_PICK": 1,
         "TP1_PCT": 15.0, "EST_WIN_RATE": 0.75},   # 과대 선언
    ])
    # 실현 0.40 (모든 bin이 낮은 승률)
    _write_winrate_table(tmp_path / "winrate_table_by_ELITE_SCORE_latest.json", [
        {"score_lo": 0, "score_hi": 50, "p_win": 0.35, "n_raw": 100,
         "sufficient": True, "avg_ret_net_pct": 0.1,
         "avg_ret_excess_pct": 0.1},
        {"score_lo": 50, "score_hi": 100, "p_win": 0.42, "n_raw": 100,
         "sufficient": True, "avg_ret_net_pct": 0.2,
         "avg_ret_excess_pct": 0.2},
    ])

    report = generate_monotonicity_report(str(tmp_path), "20260424")
    gap = report["declared_vs_realized_gap_top_pick"]
    assert gap is not None and gap > 0.15
    fail_gates = [g for g in report["ci_hard"] if g["status"] == "FAIL"]
    assert any(g["gate"] == "declared_vs_realized_gap_15pp" for g in fail_gates)


# ════════════════════════════════════════════════════════════
#  Edge — recommend_latest.csv 없음 (collector 미실행)
# ════════════════════════════════════════════════════════════

def test_no_recommend_file_graceful(tmp_path):
    """recommend_latest.csv 없으면 SKIP gates + 파일은 생성"""
    from daily_briefing import generate_monotonicity_report

    report = generate_monotonicity_report(str(tmp_path), "20260424")

    # 크래시 없이 생성됐는지
    assert "ci_hard" in report
    assert "ci_soft" in report
    assert (tmp_path / "monotonicity_report_latest.json").exists()
    # top_pick_count == 0 (recommend 없음)
    assert report.get("top_pick_count", 0) == 0


# ════════════════════════════════════════════════════════════
#  Edge — winrate_table 없음 (배포 첫날)
# ════════════════════════════════════════════════════════════

def test_no_winrate_table(tmp_path):
    """winrate_table 없으면 monotonicity 무효 + gap SKIP"""
    from daily_briefing import generate_monotonicity_report

    _write_recommend(tmp_path / "recommend_latest.csv", [
        {"종목코드": "005930", "ROUTE": "ATTACK", "TOP_PICK": 1,
         "TP1_PCT": 15.0, "EST_WIN_RATE": 0.55},
    ])

    report = generate_monotonicity_report(str(tmp_path), "20260424")

    # winrate_table 무효 → realized_wr None → gap SKIP
    assert report["elite_monotonicity"]["valid"] is False
    skip_gates = [g for g in report["ci_hard"] if g["status"] == "SKIP"]
    assert any(g["gate"] == "declared_vs_realized_gap_15pp" for g in skip_gates)
    # 다른 HARD gate (route, tp1)는 통과
    assert report["ci_hard_all_pass"] is True


# ════════════════════════════════════════════════════════════
#  Edge — 0건 TOP_PICK (모든 게이트 PASS or SKIP)
# ════════════════════════════════════════════════════════════

def test_zero_top_pick(tmp_path):
    """TOP_PICK 0건이면 route/tp1 게이트 자동 PASS"""
    from daily_briefing import generate_monotonicity_report

    _write_recommend(tmp_path / "recommend_latest.csv", [
        {"종목코드": "005930", "ROUTE": "WAIT", "TOP_PICK": 0,
         "TP1_PCT": 5.0, "EST_WIN_RATE": 0.40},
    ])

    report = generate_monotonicity_report(str(tmp_path), "20260424")

    assert report["top_pick_count"] == 0
    # 0건이면 route/tp1 위반 자체가 불가능 → PASS
    pass_gates = [g for g in report["ci_hard"] if g["status"] == "PASS"]
    gate_names = [g["gate"] for g in pass_gates]
    assert "top_pick_route_positive" in gate_names
    assert "top_pick_tp1_positive" in gate_names


# ════════════════════════════════════════════════════════════
#  TOP_PICK 기준 vs Active 기준 분리 검증
# ════════════════════════════════════════════════════════════

def test_top_pick_vs_active_split(tmp_path):
    """declared_wr_top_pick과 declared_wr_active가 다른 값으로 기록되는지"""
    from daily_briefing import generate_monotonicity_report

    # Active(ATTACK/ARMED) 5종목, TOP_PICK은 그 중 1종목
    _write_recommend(tmp_path / "recommend_latest.csv", [
        {"종목코드": "005930", "ROUTE": "ATTACK", "TOP_PICK": 1,
         "TP1_PCT": 16.0, "EST_WIN_RATE": 0.70},   # TOP_PICK만 0.70
        {"종목코드": "000660", "ROUTE": "ATTACK", "TOP_PICK": 0,
         "TP1_PCT": 12.0, "EST_WIN_RATE": 0.50},
        {"종목코드": "373220", "ROUTE": "ARMED", "TOP_PICK": 0,
         "TP1_PCT": 10.0, "EST_WIN_RATE": 0.50},
        {"종목코드": "207940", "ROUTE": "ARMED", "TOP_PICK": 0,
         "TP1_PCT": 8.0, "EST_WIN_RATE": 0.50},
        {"종목코드": "005380", "ROUTE": "ATTACK", "TOP_PICK": 0,
         "TP1_PCT": 7.0, "EST_WIN_RATE": 0.50},
    ])

    report = generate_monotonicity_report(str(tmp_path), "20260424")

    # TOP_PICK 평균: 단일 0.70
    assert report["declared_wr_top_pick"] == 0.70
    # Active 평균: (0.70 + 0.50*4) / 5 = 0.54
    assert abs(report["declared_wr_active"] - 0.54) < 0.001
    # 둘이 달라야 함
    assert report["declared_wr_top_pick"] != report["declared_wr_active"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
