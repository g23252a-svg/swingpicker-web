# tests/test_v22_3_1_ops_reports.py
"""
v22.3.1 회귀 테스트 — 운영 리포트 일관성
==========================================

평가 피드백 핵심 항목:
  1. save_health() 실행 시 run_health_latest.json 생성 확인
  2. generate_monotonicity_report() 실행 시 latest 생성 확인
  3. TOP_PICK RR<1 존재 시 monotonicity hard fail 확인

이 테스트는 운영 산출물의 신뢰성을 보장한다 — 코드만 막아도 리포트가
감시 안 하면 다음 배포에서 회귀 발생 가능.
"""
import os
import json
import tempfile
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────
# Test 1: save_health() 가 run_health_latest.json 도 생성하는가
# ──────────────────────────────────────────────────────────────
def test_save_health_creates_latest_file():
    """save_health()이 run_health_{ymd}.json + run_health_latest.json 둘 다 생성해야 함"""
    from run_health import RunHealth, save_health

    health = RunHealth(
        status="OK",
        reasons=[],
        warnings=[],
        data_freshness_ok=True,
        checks={"MCAP": True, "BENCH": True},
        confidence_score=100,
        macro_risk="NORMAL",
        market_breadth=55.0,
        axis_status={},
        fallback_count=0,
    )
    # max_allowed_route는 status/confidence 기반 자동 계산 property (read-only)
    # OK + confidence_score 100 → "ATTACK" 자동

    with tempfile.TemporaryDirectory() as tmpdir:
        save_health(health, tmpdir, "20260501")

        dated_path = os.path.join(tmpdir, "run_health_20260501.json")
        latest_path = os.path.join(tmpdir, "run_health_latest.json")

        assert os.path.exists(dated_path), "dated 파일 미생성"
        assert os.path.exists(latest_path), "latest 파일 미생성 — v22.3 패치 미적용"

        # 두 파일 내용 동일 + trade_ymd 필드 있어야 함
        with open(dated_path, encoding='utf-8') as f:
            dated = json.load(f)
        with open(latest_path, encoding='utf-8') as f:
            latest = json.load(f)

        assert latest.get("trade_ymd") == "20260501", "latest에 trade_ymd 필드 없음"
        assert latest.get("status") == "OK"
        assert latest == dated, "dated와 latest 내용 불일치"


# ──────────────────────────────────────────────────────────────
# Test 2: monotonicity_report 가 RR<1 존재 시 HARD FAIL
# ──────────────────────────────────────────────────────────────
def _create_mock_recommend_csv(tmpdir, trade_ymd, top_picks_data):
    """recommend_{ymd}.csv 와 recommend_latest.csv 모두 생성"""
    df = pd.DataFrame([{
        "종목코드": f"00{i:04d}",
        "종목명": f"TEST{i}",
        "ROUTE": d.get("ROUTE", "ARMED"),
        "TOP_PICK": d["TOP_PICK"],
        "TP1_PCT": d.get("TP1_PCT", 10.0),
        "RR_NOW_TP1": d.get("RR_NOW_TP1", 1.5),
        "EST_WIN_RATE": d.get("EST_WIN_RATE", 0.60),
    } for i, d in enumerate(top_picks_data)])

    dated = os.path.join(tmpdir, f"recommend_{trade_ymd}.csv")
    latest = os.path.join(tmpdir, "recommend_latest.csv")
    df.to_csv(dated, index=False, encoding="utf-8-sig")
    df.to_csv(latest, index=False, encoding="utf-8-sig")
    return dated


def test_monotonicity_report_hard_fail_on_rr_lt_1():
    """TOP_PICK 중 RR_NOW_TP1 < 1.0 종목 있으면 monotonicity ci_hard FAIL"""
    from daily_briefing import generate_monotonicity_report

    with tempfile.TemporaryDirectory() as tmpdir:
        # TOP_PICK 2개: RR 1.5(통과), RR 0.87(차단되어야 할 것이 들어왔음 — 가짜 케이스)
        _create_mock_recommend_csv(tmpdir, "20260501", [
            {"TOP_PICK": 1, "RR_NOW_TP1": 1.50, "TP1_PCT": 10.0},
            {"TOP_PICK": 1, "RR_NOW_TP1": 0.87, "TP1_PCT": 8.0},
            {"TOP_PICK": 0, "RR_NOW_TP1": 2.00, "TP1_PCT": 12.0},
        ])

        report = generate_monotonicity_report(tmpdir, "20260501")

        # ci_hard 안에 RR 게이트 있어야 함
        rr_gates = [g for g in report.get("ci_hard", []) if g.get("gate") == "top_pick_rr_now_tp1_1"]
        assert len(rr_gates) == 1, f"top_pick_rr_now_tp1_1 게이트 없음: {report.get('ci_hard')}"
        assert rr_gates[0]["status"] == "FAIL", \
            f"RR<1 종목이 있는데 PASS로 판정: {rr_gates[0]}"

        # 통계 메트릭 확인
        assert report["top_pick_rr_lt1_count"] == 1
        assert report["top_pick_min_rr"] == 0.87

        # ci_hard_all_pass 는 False여야 함
        assert report["ci_hard_all_pass"] is False, \
            "RR<1 차단 미작동 — ci_hard_all_pass=True"


def test_monotonicity_report_hard_pass_when_all_rr_ok():
    """모든 TOP_PICK이 RR≥1.0이면 PASS"""
    from daily_briefing import generate_monotonicity_report

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_recommend_csv(tmpdir, "20260501", [
            {"TOP_PICK": 1, "RR_NOW_TP1": 1.50, "TP1_PCT": 10.0},
            {"TOP_PICK": 1, "RR_NOW_TP1": 1.08, "TP1_PCT": 8.0},
            {"TOP_PICK": 0, "RR_NOW_TP1": 0.50, "TP1_PCT": 12.0},  # 비-TOP_PICK은 무시
        ])

        report = generate_monotonicity_report(tmpdir, "20260501")
        rr_gates = [g for g in report.get("ci_hard", []) if g.get("gate") == "top_pick_rr_now_tp1_1"]
        assert len(rr_gates) == 1
        assert rr_gates[0]["status"] == "PASS", f"RR≥1 모두인데 FAIL: {rr_gates[0]}"
        assert report["top_pick_rr_lt1_count"] == 0
        assert report["top_pick_min_rr"] == 1.08


# ──────────────────────────────────────────────────────────────
# Test 3: monotonicity_report 가 latest 도 생성하는가
# ──────────────────────────────────────────────────────────────
def test_monotonicity_report_creates_latest():
    """generate_monotonicity_report()이 latest 파일도 만들어야 함"""
    from daily_briefing import generate_monotonicity_report

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_recommend_csv(tmpdir, "20260501", [
            {"TOP_PICK": 1, "RR_NOW_TP1": 1.50},
        ])

        generate_monotonicity_report(tmpdir, "20260501")

        dated = os.path.join(tmpdir, "monotonicity_report_20260501.json")
        latest = os.path.join(tmpdir, "monotonicity_report_latest.json")

        assert os.path.exists(dated), "dated 파일 미생성"
        assert os.path.exists(latest), "latest 파일 미생성"

        with open(dated, encoding='utf-8') as f:
            d = json.load(f)
        with open(latest, encoding='utf-8') as f:
            l = json.load(f)

        # 핵심 키들 일치
        assert d.get("ci_hard_all_pass") == l.get("ci_hard_all_pass")
        assert d.get("top_pick_count") == l.get("top_pick_count")
