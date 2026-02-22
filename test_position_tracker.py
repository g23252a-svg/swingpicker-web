"""P3 #14 포지션 트래킹 & 자동 알림 테스트"""
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


def _setup(tmp):
    """테스트 포지션 + 가격 데이터"""
    from position_tracker import Position, save_positions

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

    # 가격: 삼성 손절, SK 정상, 카카오 TP2 도달
    snap = pd.DataFrame([
        {"종목코드": "005930", "시가": 67000, "고가": 68000, "저가": 65000, "종가": 65500},
        {"종목코드": "000660", "시가": 152000, "고가": 155000, "저가": 151000, "종가": 153000},
        {"종목코드": "035720", "시가": 58000, "고가": 62000, "저가": 57000, "종가": 61000},
    ])
    snap.to_csv(os.path.join(tmp, "price_snapshot_20260215.csv"), index=False)


def run():
    global PASS, FAIL
    PASS, FAIL = 0, 0

    print("=" * 60)
    print("🧪 P3 #14 포지션 트래킹 & 자동 알림 테스트")
    print("=" * 60)

    from position_tracker import (
        Position, load_positions, save_positions, save_to_history,
        detect_events, TrackEvent, track_open_positions,
        register_from_recommendations, record_closed_to_tradelog,
        _make_event_key,
    )

    # ═══ 1. 포지션 SSOT ═══
    print("\n📐 1. 포지션 SSOT 저장소")
    tmp = tempfile.mkdtemp()
    try:
        _setup(tmp)
        positions = load_positions(tmp)
        test("포지션 로드 3개", len(positions) == 3)
        test("삼성전자 진입가", positions[0].entry_px == 70000)
        test("status=OPEN", all(p.status == "OPEN" for p in positions))
        test("alerted_events 빈 리스트", all(len(p.alerted_events) == 0 for p in positions))

        # 필드 완전성
        p = positions[0]
        test("stop_px_initial 보존", p.stop_px_initial == 66000)
        test("trailing_high 초기값", p.trailing_high == 70000)

        # ═══ 2. 이벤트 감지 규칙 ═══
        print("\n📐 2. 이벤트 감지 규칙 (A: 수학/규칙 테스트)")

        # (a) 손절 히트: 종가 ≤ stop_px
        p_stop = Position(code="TEST", name="테스트", entry_ymd="20260210",
                          entry_px=10000, stop_px=9000, stop_px_initial=9000,
                          take_px1=12000, take_px2=14000, trailing_high=10000)
        events, p_updated = detect_events(p_stop, today_close=8900, today_high=10000,
                                          today_low=8800, check_ymd="20260215")
        test("손절: 종가≤SL → STOP_HIT", any(e.event_type == "STOP_HIT" for e in events))
        test("손절: status=CLOSED_STOP", p_updated.status == "CLOSED_STOP")
        test("손절: realized_pnl 음수", p_updated.realized_pnl_pct < 0)

        # (b) 손절 안 됨: 종가 > stop_px
        p_safe = Position(code="TEST2", name="테스트2", entry_ymd="20260210",
                          entry_px=10000, stop_px=9000, stop_px_initial=9000,
                          take_px1=12000, take_px2=14000, trailing_high=10000)
        events2, p2 = detect_events(p_safe, today_close=9100, today_high=10000,
                                    today_low=9000, check_ymd="20260215")
        test("안전: 종가>SL → STOP_HIT 없음",
             not any(e.event_type == "STOP_HIT" for e in events2))
        test("안전: status=OPEN", p2.status == "OPEN")

        # (c) TP1 도달
        p_tp1 = Position(code="TEST3", name="테스트3", entry_ymd="20260210",
                         entry_px=10000, stop_px=9000, stop_px_initial=9000,
                         take_px1=12000, take_px2=14000, trailing_high=10000)
        events3, p3 = detect_events(p_tp1, today_close=12500, today_high=13000,
                                    today_low=12000, check_ymd="20260215")
        test("TP1: 종가≥TP1 → TP1_HIT", any(e.event_type == "TP1_HIT" for e in events3))
        test("TP1: status=OPEN 유지 (TP2 미도달)", p3.status == "OPEN")

        # (d) TP2 도달 → 청산
        p_tp2 = Position(code="TEST4", name="테스트4", entry_ymd="20260210",
                         entry_px=10000, stop_px=9000, stop_px_initial=9000,
                         take_px1=12000, take_px2=14000, trailing_high=10000)
        events4, p4 = detect_events(p_tp2, today_close=14500, today_high=15000,
                                    today_low=14000, check_ymd="20260215")
        test("TP2: 종가≥TP2 → TP2_HIT", any(e.event_type == "TP2_HIT" for e in events4))
        test("TP2: status=CLOSED_TP", p4.status == "CLOSED_TP")
        test("TP2: realized_pnl 양수", p4.realized_pnl_pct > 0)

        # (e) 드로다운 경고
        p_dd = Position(code="TEST5", name="테스트5", entry_ymd="20260210",
                        entry_px=10000, stop_px=8000, stop_px_initial=8000,
                        take_px1=15000, take_px2=20000, trailing_high=10000)
        events5, _ = detect_events(p_dd, today_close=9400, today_high=9500,
                                   today_low=9300, check_ymd="20260215")
        test("드로다운 -6%: WARN_DRAWDOWN",
             any(e.event_type == "WARN_DRAWDOWN" for e in events5))

        # ═══ 3. 알림 중복 방지 (B: 운영 테스트) ═══
        print("\n📐 3. 알림 중복 방지 (Idempotency)")

        # 동일 포지션 2번 체크 → 알림 1번만
        p_dup = Position(code="DUP", name="중복테스트", entry_ymd="20260210",
                         entry_px=10000, stop_px=9000, stop_px_initial=9000,
                         take_px1=15000, take_px2=20000, trailing_high=10000)

        # 1차 실행
        events_r1, p_r1 = detect_events(p_dup, today_close=9400, today_high=9500,
                                        today_low=9300, check_ymd="20260215")
        n_events_r1 = len(events_r1)
        # event_key 등록 (발송 시뮬레이션)
        p_r1.alerted_events.extend([e.event_key for e in events_r1])

        # 2차 실행 (동일 데이터)
        events_r2, p_r2 = detect_events(p_r1, today_close=9400, today_high=9500,
                                        today_low=9300, check_ymd="20260215")
        test("1차: 이벤트 발생", n_events_r1 > 0)
        test("2차: 중복 이벤트 0건", len(events_r2) == 0,
             f"got {len(events_r2)} events")

        # 다른 날짜면 새 이벤트
        events_r3, _ = detect_events(p_r2, today_close=9300, today_high=9400,
                                     today_low=9200, check_ymd="20260216")
        test("다른 날: 새 이벤트 발생", len(events_r3) > 0)

        # ═══ 4. 기업행위 필터 ═══
        print("\n📐 4. 기업행위 필터")
        p_ca = Position(code="CA", name="기업행위", entry_ymd="20260210",
                        entry_px=10000, stop_px=9000, stop_px_initial=9000,
                        take_px1=12000, take_px2=14000, trailing_high=10000,
                        last_close_px=10000)
        # 400% 상승 → 기업행위 의심
        events_ca, p_ca_u = detect_events(p_ca, today_close=50000, today_high=50000,
                                          today_low=50000, check_ymd="20260215")
        test("기업행위: CORPORATE_ACTION 이벤트",
             any(e.event_type == "CORPORATE_ACTION" for e in events_ca))
        test("기업행위: STOP_HIT 아님",
             not any(e.event_type == "STOP_HIT" for e in events_ca))
        test("기업행위: status=OPEN 유지", p_ca_u.status == "OPEN")

        # ═══ 5. track_open_positions 통합 ═══
        print("\n📐 5. 통합 트래킹")
        # 재설정
        _setup(tmp)
        result = track_open_positions(tmp, "20260215")
        test("checked=3", result["checked"] == 3)
        test("events > 0", result["events"] > 0)
        test("closed > 0", result["closed"] > 0, f"got {result['closed']}")

        # 저장 확인
        remaining = load_positions(tmp)
        test("미청산 포지션 < 3", len(remaining) < 3)

        # 히스토리
        hist_path = os.path.join(tmp, "positions_history.json")
        test("히스토리 저장됨", os.path.exists(hist_path))

        # ═══ 6. 2회 실행 중복 방지 (B: 스케줄 테스트) ═══
        print("\n📐 6. 2회 실행 중복 방지")
        result2 = track_open_positions(tmp, "20260215")
        test("2회차: 추가 이벤트 0건 (이미 처리됨)",
             result2["events"] == 0,
             f"got {result2['events']}")

        # ═══ 7. 추천→포지션 등록 ═══
        print("\n📐 7. 추천→포지션 자동 등록")
        rec = pd.DataFrame({
            "종목코드": ["005930", "999999"],
            "종목명": ["삼성전자", "신규종목"],
            "매수가": [72000, 30000],
            "손절가": [68000, 28000],
            "TP1": [80000, 35000],
            "TP2": [88000, 40000],
        })
        n_reg = register_from_recommendations(tmp, rec, "20260215", top_n=5)
        test("중복(삼성) 스킵 + 신규 1개", n_reg == 1)

        all_pos = load_positions(tmp)
        codes = [p.code for p in all_pos]
        test("999999 등록됨", "999999" in codes)

        # ═══ 8. #13 calibration 연결 ═══
        print("\n📐 8. #13 calibration 연결")
        # 청산 포지션 → per_trade_log
        closed_test = [Position(
            code="TEST", name="테스트", entry_ymd="20260210", entry_px=10000,
            stop_px=9000, stop_px_initial=9000, take_px1=12000, take_px2=14000,
            status="CLOSED_STOP", close_ymd="20260215", close_px=8900,
            close_reason="STOP_HIT", realized_pnl_pct=-11.0, trailing_high=10000,
        )]
        n_recorded = record_closed_to_tradelog(tmp, closed_test)
        test("per_trade_log 기록", n_recorded == 1)

        log_path = os.path.join(tmp, "per_trade_log.csv")
        test("per_trade_log.csv 존재", os.path.exists(log_path))
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            test("exit_type=STOP_HIT", "STOP_HIT" in log_df["exit_type"].values)

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
