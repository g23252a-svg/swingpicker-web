# -*- coding: utf-8 -*-
"""[v43] 위기 봉인 — 신규진입 차단 중 매수검토(ATTACK/ARMED) 라벨 봉인.

진단(7월 폭락, 코스피 7,284→6,023 -17%): 공식 추천(TOP_PICK)은 차단을
존중해 2주간 0개였는데, ROUTE 라벨은 차단을 무시하고 매수검토 10~23개를
계속 표시 — 이 라벨 추종 실측 -3.1%(7/23분 -7.7%)가 사용자 손실의 실체.

역사 실측(워크포워드 패널): 차단일 ATTACK 승률 7% vs 비차단일 50%
(일별 t=-3.96), 같은 날 유니버스 대비도 열세(t=-2.37) — 절대·상대 모두
무가치. 패널 내 차단→해제 전환일 0회로 '해제 직전 반등 놓침' 반박 불성립.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_engine import recompute_route_with_alpha


def _row(**ov):
    # 치료된 ATTACK 조건 충족 행 (알파 상위·MA20위·저점추세·비과열)
    r = {
        "종목코드": "005930", "ROUTE": "WAIT", "ALPHA_VALIDATED": 1,
        "ALPHA_SCORE": 95.0, "MARKET_REGIME": "NEUTRAL",
        "Above_MA20": 1, "Low_Trend_PCT": 5.0,
        "ret_5d_%": 3.0, "MFI14": 55.0, "RSI14": 60.0, "TTM_SQUEEZE": 0,
    }
    r.update(ov)
    return r


def _run(rows):
    return recompute_route_with_alpha(pd.DataFrame([_row(**r) for r in rows]))


def test_blocked_attack_is_sealed_to_wait():
    out = _run([{"NEW_ENTRY_BLOCKED": 1}])
    r = out.iloc[0]
    assert r["ROUTE"] == "WAIT"
    assert r["ROUTE_PRE_BLOCK"] == "ATTACK"       # 원 판정 보관 (해제 시 복귀 근거)
    assert "봉인" in r["ROUTE_REASON"]
    assert not bool(r["IS_ACTIVE"])
    assert bool(r["IS_WATCH"])                     # 관찰 목록에서는 계속 보임


def test_unblocked_attack_untouched():
    out = _run([{"NEW_ENTRY_BLOCKED": 0}])
    r = out.iloc[0]
    assert r["ROUTE"] == "ATTACK"
    assert r["ROUTE_PRE_BLOCK"] == ""
    assert bool(r["IS_ACTIVE"])


def test_blocked_armed_also_sealed():
    out = _run([{"ALPHA_SCORE": 75.0, "NEW_ENTRY_BLOCKED": 1}])   # ARMED 구간(문턱-15)
    r = out.iloc[0]
    assert r["ROUTE"] == "WAIT"
    assert r["ROUTE_PRE_BLOCK"] == "ARMED"


def test_other_block_flags_also_seal():
    for col in ["JULY_PROFIT_BLOCK_FLAG", "PROFIT_RECOVERY_BLOCK_FLAG"]:
        out = _run([{col: 1}])
        assert out.iloc[0]["ROUTE"] == "WAIT", col


def test_carry_preserved_even_when_blocked():
    # 보유 이월은 신규진입이 아니므로 봉인 대상 아님.
    out = _run([{"ROUTE": "CARRY", "NEW_ENTRY_BLOCKED": 1}])
    assert out.iloc[0]["ROUTE"] == "CARRY"


def test_unvalidated_leaves_route_alone():
    out = _run([{"ALPHA_VALIDATED": 0, "ROUTE": "ATTACK", "NEW_ENTRY_BLOCKED": 1}])
    assert out.iloc[0]["ROUTE"] == "ATTACK"       # 미검증이면 치료 자체를 안 함


def test_wait_overheat_not_marked_pre_block():
    # 봉인은 ATTACK/ARMED에만 — 원래 WAIT/과열은 ROUTE_PRE_BLOCK 비움.
    out = _run([{"ALPHA_SCORE": 30.0, "NEW_ENTRY_BLOCKED": 1}])
    r = out.iloc[0]
    assert r["ROUTE"] == "WAIT"
    assert r["ROUTE_PRE_BLOCK"] == ""
