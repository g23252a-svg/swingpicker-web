# -*- coding: utf-8 -*-
"""[v44] 폭락 추격 경고 + ARMED 라벨 정직화.

사용자 질문("삼성전자 오늘 엄청 반등") 실측 검증 결과 — 1.5년 주가 패널
(유동성 200억+, 차단 기간, 10일 전방수익):
  · MA20 이격 -15% 이하          -3.57% (승률 41%)
  · 과매도(RSI<35) & 이격 -15%↓   -2.14% (승률 44%)
  · 과매도 & 대형(거래대금1000억+)  -1.30% (승률 50%)
  · 반면 구조 유지(MA20위+저점상승) +5.68% (승률 62%)
→ '많이 빠졌으니 반등'이 아니라 '구조가 살아있나'가 갈림. 낙폭 과대 추격 경고.

ARMED 라벨: 저점상승을 요구하지 않는 ARMED 구조는 정상장 +0.54%/승률46%
(n=15,597)로 ATTACK 구조 +2.02%/48%(n=13,866)의 1/4 — '진입대기'라는 이름이
실측 가치를 과대표현하므로 '구조미확인'으로 정직화.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_engine import recompute_route_with_alpha
from components.ui_terms import ROUTE_LABELS


def _row(**ov):
    r = {
        "종목코드": "005930", "종목명": "삼성전자", "ROUTE": "WAIT",
        "ALPHA_VALIDATED": 1, "ALPHA_SCORE": 87.7, "MARKET_REGIME": "NEUTRAL",
        "Above_MA20": 0, "Low_Trend_PCT": -23.4, "이격도": -21.5,
        "ret_5d_%": -23.3, "MFI14": 35.7, "RSI14": 31.9, "TTM_SQUEEZE": 0,
    }
    r.update(ov)
    return r


def _run(rows):
    return recompute_route_with_alpha(pd.DataFrame([_row(**r) for r in rows]))


# ── 폭락 추격 경고 ──

def test_deep_oversold_gets_chase_warning():
    # 삼성전자 실제 케이스 (이격 -21.5%, RSI 31.9)
    out = _run([{}])
    r = out.iloc[0]
    assert r["CRASH_CHASE_WARN"] == 1
    assert r["ROUTE"] == "WAIT"
    assert "낙폭 과대" in r["ROUTE_REASON"]
    assert "41%" in r["ROUTE_REASON"]      # 실측 승률 근거 노출


def test_mild_pullback_no_warning():
    out = _run([{"이격도": -6.0, "RSI14": 45.0, "Low_Trend_PCT": -3.0}])
    r = out.iloc[0]
    assert r["CRASH_CHASE_WARN"] == 0
    assert "낙폭 과대" not in r["ROUTE_REASON"]


def test_oversold_alone_without_deep_drop_no_warning():
    # RSI만 낮고 이격은 얕으면 경고 아님 — 기준은 '낙폭'(이격 -15%↓)
    out = _run([{"이격도": -5.0, "RSI14": 28.0, "Low_Trend_PCT": -2.0}])
    assert out.iloc[0]["CRASH_CHASE_WARN"] == 0


def test_warning_falls_back_to_low_trend_when_dev_missing():
    df = pd.DataFrame([_row()]).drop(columns=["이격도"])
    out = recompute_route_with_alpha(df)
    assert out.iloc[0]["CRASH_CHASE_WARN"] == 1   # 저점추세 -23.4로 판정


def test_carry_not_warned():
    out = _run([{"ROUTE": "CARRY"}])
    r = out.iloc[0]
    assert r["ROUTE"] == "CARRY"
    assert r["CRASH_CHASE_WARN"] == 0             # 보유 관리엔 추격 경고 무의미


def test_seal_reason_wins_over_chase_warning():
    # 봉인(v43)은 MA20 위 구조라 원래 낙폭 조건에 안 걸리지만, 사유 우선순위 고정.
    out = _run([{"Above_MA20": 1, "Low_Trend_PCT": 3.0, "이격도": 5.0,
                 "ret_5d_%": 2.0, "NEW_ENTRY_BLOCKED": 1}])
    r = out.iloc[0]
    assert r["ROUTE_PRE_BLOCK"] == "ATTACK"
    assert "봉인" in r["ROUTE_REASON"]


# ── ARMED 라벨 정직화 ──

def test_armed_label_is_honest():
    assert ROUTE_LABELS["ARMED"] == "구조미확인"
    assert ROUTE_LABELS["ATTACK"] == "매수검토"     # 매수 후보는 ATTACK만


def test_armed_label_consistent_across_ui():
    from components.stock_detail_v2 import ROUTE_KR_MAP
    assert ROUTE_KR_MAP["ARMED"] == "구조미확인"
    src = (ROOT / "components" / "tab_stocks.py").read_text(encoding="utf-8")
    assert "진입대기" not in src, "tab_stocks에 옛 ARMED 라벨 잔존"
