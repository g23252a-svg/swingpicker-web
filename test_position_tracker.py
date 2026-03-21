# -*- coding: utf-8 -*-
"""test_position_tracker.py — P3 #14 포지션 트래킹 & 자동 알림 (pytest 표준)
═══════════════════════════════════════════════════
실행: pytest test_position_tracker.py -v
"""
import os
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

from position_tracker import (
    Position, load_positions, save_positions, save_to_history,
    detect_events, TrackEvent, track_open_positions,
    register_from_recommendations, record_closed_to_tradelog,
    _make_event_key,
)


def _setup(tmp):
    """테스트 포지션 + 가격 데이터."""
    positions = [
        Position(code="005930", name="삼성전자", entry_ymd="20260210",
                 entry_px=70000, stop_px=66000, stop_px_initial=66000,
                 take_px1=77000, take_px2=84000, trailing_high=70000),
        Position(code="000660", name="SK하이닉스", entry_ymd="20260210",
                 entry_px=150000, stop_px=140000, stop_px_initial=140000,
                 take_px1=165000, take_px2=180000, trailing_high=150000),
        Position(code="035720", name="카카오", entry_ymd="20260210",
                 entry_px=50000, stop_px=47000, stop_px_initial=47000,
                 take_px1=55000, take_px2=60000, trailing_high=50000),
    ]
    save_positions(tmp, positions)
    snap = pd.DataFrame([
        {"종목코드": "005930", "시가": 67000, "고가": 68000, "저가": 65000, "종가": 65500},
        {"종목코드": "000660", "시가": 152000, "고가": 155000, "저가": 151000, "종가": 153000},
        {"종목코드": "035720", "시가": 58000, "고가": 62000, "저가": 57000, "종가": 61000},
    ])
    snap.to_csv(os.path.join(tmp, "price_snapshot_20260215.csv"), index=False)


@pytest.fixture()
def tracker_dir():
    tmp = tempfile.mkdtemp()
    _setup(tmp)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════
#  1. 포지션 SSOT 저장소
# ═══════════════════════════════════════════════
class TestPositionSSOT:
    def test_load_positions(self, tracker_dir):
        positions = load_positions(tracker_dir)
        assert len(positions) == 3
        assert positions[0].entry_px == 70000

    def test_all_open(self, tracker_dir):
        positions = load_positions(tracker_dir)
        assert all(p.status == "OPEN" for p in positions)
        assert all(len(p.alerted_events) == 0 for p in positions)

    def test_field_completeness(self, tracker_dir):
        p = load_positions(tracker_dir)[0]
        assert p.stop_px_initial == 66000
        assert p.trailing_high == 70000


# ═══════════════════════════════════════════════
#  2. 이벤트 감지 규칙
# ═══════════════════════════════════════════════
class TestEventDetection:
    def test_stop_hit(self):
        p = Position(code="TEST", name="테스트", entry_ymd="20260210",
                     entry_px=10000, stop_px=9000, stop_px_initial=9000,
                     take_px1=12000, take_px2=14000, trailing_high=10000)
        events, p_u = detect_events(p, today_close=8900, today_high=10000,
                                    today_low=8800, check_ymd="20260215")
        assert any(e.event_type == "STOP_HIT" for e in events)
        assert p_u.status == "CLOSED_STOP"
        assert p_u.realized_pnl_pct < 0

    def test_no_stop(self):
        p = Position(code="TEST2", name="테스트2", entry_ymd="20260210",
                     entry_px=10000, stop_px=9000, stop_px_initial=9000,
                     take_px1=12000, take_px2=14000, trailing_high=10000)
        events, p_u = detect_events(p, today_close=9100, today_high=10000,
                                    today_low=9000, check_ymd="20260215")
        assert not any(e.event_type == "STOP_HIT" for e in events)
        assert p_u.status == "OPEN"

    def test_tp1_hit(self):
        p = Position(code="TEST3", name="테스트3", entry_ymd="20260210",
                     entry_px=10000, stop_px=9000, stop_px_initial=9000,
                     take_px1=12000, take_px2=14000, trailing_high=10000)
        events, p_u = detect_events(p, today_close=12500, today_high=13000,
                                    today_low=12000, check_ymd="20260215")
        assert any(e.event_type == "TP1_HIT" for e in events)
        assert p_u.status == "OPEN"  # TP2 미도달

    def test_tp2_hit(self):
        p = Position(code="TEST4", name="테스트4", entry_ymd="20260210",
                     entry_px=10000, stop_px=9000, stop_px_initial=9000,
                     take_px1=12000, take_px2=14000, trailing_high=10000)
        events, p_u = detect_events(p, today_close=14500, today_high=15000,
                                    today_low=14000, check_ymd="20260215")
        assert any(e.event_type == "TP2_HIT" for e in events)
        assert p_u.status == "CLOSED_TP"
        assert p_u.realized_pnl_pct > 0

    def test_drawdown_warning(self):
        p = Position(code="TEST5", name="테스트5", entry_ymd="20260210",
                     entry_px=10000, stop_px=8000, stop_px_initial=8000,
                     take_px1=15000, take_px2=20000, trailing_high=10000)
        events, _ = detect_events(p, today_close=9400, today_high=9500,
                                  today_low=9300, check_ymd="20260215")
        assert any(e.event_type == "WARN_DRAWDOWN" for e in events)


# ═══════════════════════════════════════════════
#  3. 알림 중복 방지
# ═══════════════════════════════════════════════
class TestIdempotency:
    def test_no_duplicate_alerts(self):
        p = Position(code="DUP", name="중복테스트", entry_ymd="20260210",
                     entry_px=10000, stop_px=9000, stop_px_initial=9000,
                     take_px1=15000, take_px2=20000, trailing_high=10000)
        events_r1, p_r1 = detect_events(p, today_close=9400, today_high=9500,
                                        today_low=9300, check_ymd="20260215")
        assert len(events_r1) > 0
        p_r1.alerted_events.extend([e.event_key for e in events_r1])

        events_r2, _ = detect_events(p_r1, today_close=9400, today_high=9500,
                                     today_low=9300, check_ymd="20260215")
        assert len(events_r2) == 0

    def test_new_day_new_event(self):
        p = Position(code="DUP2", name="일자별", entry_ymd="20260210",
                     entry_px=10000, stop_px=9000, stop_px_initial=9000,
                     take_px1=15000, take_px2=20000, trailing_high=10000)
        events_r1, p_r1 = detect_events(p, today_close=9400, today_high=9500,
                                        today_low=9300, check_ymd="20260215")
        p_r1.alerted_events.extend([e.event_key for e in events_r1])
        events_r3, _ = detect_events(p_r1, today_close=9300, today_high=9400,
                                     today_low=9200, check_ymd="20260216")
        assert len(events_r3) > 0


# ═══════════════════════════════════════════════
#  4. 기업행위 필터
# ═══════════════════════════════════════════════
class TestCorporateAction:
    def test_extreme_rise_flagged(self):
        p = Position(code="CA", name="기업행위", entry_ymd="20260210",
                     entry_px=10000, stop_px=9000, stop_px_initial=9000,
                     take_px1=12000, take_px2=14000, trailing_high=10000,
                     last_close_px=10000)
        events, p_u = detect_events(p, today_close=50000, today_high=50000,
                                    today_low=50000, check_ymd="20260215")
        assert any(e.event_type == "CORPORATE_ACTION" for e in events)
        assert not any(e.event_type == "STOP_HIT" for e in events)
        assert p_u.status == "OPEN"


# ═══════════════════════════════════════════════
#  5. track_open_positions 통합
# ═══════════════════════════════════════════════
class TestTrackOpenPositions:
    def test_integrated_tracking(self, tracker_dir):
        result = track_open_positions(tracker_dir, "20260215")
        assert result["checked"] == 3
        assert result["events"] > 0
        assert result["closed"] > 0

        remaining = load_positions(tracker_dir)
        assert len(remaining) < 3
        assert os.path.exists(os.path.join(tracker_dir, "positions_history.json"))

    def test_second_run_no_extra_events(self, tracker_dir):
        track_open_positions(tracker_dir, "20260215")
        result2 = track_open_positions(tracker_dir, "20260215")
        assert result2["events"] == 0


# ═══════════════════════════════════════════════
#  7. 추천→포지션 자동 등록
# ═══════════════════════════════════════════════
class TestRegisterFromRecommendations:
    def test_skip_duplicate_register_new(self, tracker_dir):
        rec = pd.DataFrame({
            "종목코드": ["005930", "999999"],
            "종목명": ["삼성전자", "신규종목"],
            "매수가": [72000, 30000],
            "손절가": [68000, 28000],
            "TP1": [80000, 35000],
            "TP2": [88000, 40000],
        })
        n_reg = register_from_recommendations(tracker_dir, rec, "20260215", top_n=5)
        assert n_reg == 1
        codes = [p.code for p in load_positions(tracker_dir)]
        assert "999999" in codes


# ═══════════════════════════════════════════════
#  8. #13 calibration 연결
# ═══════════════════════════════════════════════
class TestCalibrationConnection:
    def test_per_trade_log(self, tracker_dir):
        closed_test = [Position(
            code="TEST", name="테스트", entry_ymd="20260210", entry_px=10000,
            stop_px=9000, stop_px_initial=9000, take_px1=12000, take_px2=14000,
            status="CLOSED_STOP", close_ymd="20260215", close_px=8900,
            close_reason="STOP_HIT", realized_pnl_pct=-11.0, trailing_high=10000,
        )]
        n_recorded = record_closed_to_tradelog(tracker_dir, closed_test)
        assert n_recorded == 1

        log_path = os.path.join(tracker_dir, "per_trade_log.csv")
        assert os.path.exists(log_path)
        log_df = pd.read_csv(log_path)
        assert "STOP_HIT" in log_df["exit_type"].values
