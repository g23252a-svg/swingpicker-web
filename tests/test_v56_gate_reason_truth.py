# -*- coding: utf-8 -*-
"""[v56] 탈락 사유가 실제 원인을 지목해야 한다 — 알파 100점에 '알파 미달'.

■ 사용자가 화면에서 찾아낸 것 (2026-08-06 배치)
  '가장 가까운 관찰 후보' 3개의 사유가 이렇게 떴다:
    1. 흥구석유 024060 · AI 알파 **100.0점** → "AI 알파 진입 기준 미달"   ← 거짓
    2. 비에이치 090460 · 알파 99.3점        → "저점추세 당일 하위 12%"    ← 맞음
    3. 파세코  037070 · 알파 **98.2점**     → "AI 알파 진입 기준 미달"   ← 거짓
  진입선은 85점이다. 100점이 85점 미달일 수는 없다.

■ 실제 원인 (CSV 실측)
  278행 **전부** NEW_ENTRY_BLOCKED=True, STOP_OVERRIDE_REASON=
  "risk_off: 추천손절 유지 + 신규진입 차단". 코스피가 20일선 대비 -5.0% 이탈해
  전 종목 신규진입이 보류된 상태였다. 세 종목 모두 알파·자리·리스크가드를
  통과했거나(1·3번) 통과 후 LT만 미달(2번)이었고, 최상위 차단자는 시장 차단이다.

■ 결함의 구조
  apply_alpha_entry_gate는 4조건 AND(리스크가드 · 알파문턱 · ~차단 · 저점추세)을
  boolean 하나로만 내보냈다. 그래서 하류(recommendation_quality._reasons)가 사유를
  **재추론**했고, 알파·LT 어느 것도 걸리지 않으면 '진입 조건 미달'로 뭉갰다.
  그 문구를 decision_center._humanize_reason이 'AI 알파 진입 기준 미달'로
  **번역**하면서 알파 100점 종목에 알파 탓을 붙였다.
  → 판정하는 모듈이 사유를 남긴다(ALPHA_ENTRY_BLOCK_REASON). 재추론 금지.

■ 덤으로 드러난 라벨 거짓
  섹션 제목 '가장 가까운 관찰 후보 / 조건에 가까운 순서'인데 실제 정렬키는
  ALPHA_SCORE 내림차순이고, 전 종목 차단일에는 '가까운' 것 자체가 없다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alpha_engine as ae  # noqa: E402
from components.decision_center import _humanize_reason  # noqa: E402
from services.recommendation_quality import apply_recommendation_quality_guard  # noqa: E402


def _row(**ov):
    """알파 게이트를 통과하는 기본 행 (검증 통과 상태)."""
    r = {
        "종목코드": "024060", "종목명": "흥구석유",
        "ALPHA_VALIDATED": 1, "ALPHA_SCORE": 100.0, "ALPHA_ENTRY_THRESHOLD": 85.0,
        "ENTRY_RISK_GATE_OK": True, "NEW_ENTRY_BLOCKED": 0,
        "JULY_PROFIT_BLOCK_FLAG": 0, "PROFIT_RECOVERY_BLOCK_FLAG": 0,
        "Low_Trend_PCT": 999.0,   # 더미보다 확실히 높게 — 기본은 LT 통과
        "MARKET_REGIME": "NEUTRAL", "ROUTE": "ATTACK",
        "종가": 10000.0, "손절가": 9200.0, "추천매도가1": 14000.0,
        "추천매수가": 10000.0, "거래대금(억원)": 200.0, "POC_GAP": -5.0,
        "RR_NOW_TP1": 5.0,
    }
    r.update(ov)
    return r


def _gate(rows):
    """저점추세 분위 계산에 표본이 필요하므로 더미 행을 채워 넣는다.

    더미의 Low_Trend_PCT는 0~34 구간에 둔다. 검사 대상은 통과시킬 때 999,
    LT 탈락을 보려면 -999를 준다 — 분위가 상대값이므로 기준을 명확히 고정한다.
    """
    pad = [dict(_row(종목코드=f"9{i:05d}", 종목명=f"더미{i}", ALPHA_SCORE=10.0,
                     Low_Trend_PCT=float(i)))
           for i in range(ae._LT_PCTL_MIN_SAMPLE + 5)]
    df = pd.DataFrame(rows + pad)
    return ae.apply_alpha_entry_gate(df).iloc[:len(rows)]


# ══════════════════════════════════════════
# ① 게이트가 자기 사유를 남기는가
# ══════════════════════════════════════════

def test_passing_row_has_empty_reason():
    g = _gate([_row()])
    assert int(g.iloc[0]["ALPHA_ENTRY_OK"]) == 1
    assert g.iloc[0]["ALPHA_ENTRY_BLOCK_REASON"] == ""


def test_market_block_is_named_not_blamed_on_alpha():
    """알파 100점인데 '알파 미달'이라고 적으면 거짓이다 (사용자가 찾아낸 결함)."""
    g = _gate([_row(NEW_ENTRY_BLOCKED=1)])
    reason = g.iloc[0]["ALPHA_ENTRY_BLOCK_REASON"]
    assert "시장 전체 신규진입 차단" in reason
    assert "알파" not in reason, f"알파 100점에 알파 탓: {reason}"


def test_alpha_shortfall_names_the_score():
    g = _gate([_row(ALPHA_SCORE=70.0)])
    assert "알파 70점 (진입선 85점 미달)" in g.iloc[0]["ALPHA_ENTRY_BLOCK_REASON"]


def test_low_trend_shortfall_names_the_percentile():
    """비에이치 케이스 — 알파는 통과, 저점추세만 하위권."""
    g = _gate([_row(Low_Trend_PCT=-999.0)])
    r = g.iloc[0]["ALPHA_ENTRY_BLOCK_REASON"]
    assert "저점추세 당일 하위" in r
    assert "알파" not in r, f"알파는 통과했는데 알파 탓: {r}"


def test_risk_gate_shortfall_is_named():
    g = _gate([_row(ENTRY_RISK_GATE_OK=False)])
    assert "진입 자리·손익비 가드 미달" in g.iloc[0]["ALPHA_ENTRY_BLOCK_REASON"]


def test_multiple_failures_are_all_listed_most_decisive_first():
    """비에이치는 전체 차단 + 저점추세 미달 둘 다였다 — 하나만 적으면 정보 손실."""
    g = _gate([_row(NEW_ENTRY_BLOCKED=1, Low_Trend_PCT=-999.0)])
    r = g.iloc[0]["ALPHA_ENTRY_BLOCK_REASON"]
    assert r.index("시장 전체 신규진입 차단") < r.index("저점추세"), r
    assert "·" in r


def test_reason_column_exists_even_when_alpha_unvalidated():
    df = pd.DataFrame([_row(ALPHA_VALIDATED=0)])
    out = ae.apply_alpha_entry_gate(df)
    assert int(out.iloc[0]["ALPHA_ENTRY_OK"]) == 0
    # 미검증 경로는 조기 반환이라 사유 컬럼이 없어도 하류가 폴백으로 처리한다
    assert "ALPHA_ENTRY_OK" in out.columns


# ══════════════════════════════════════════
# ② 품질 가드가 그 사유를 그대로 쓰는가
# ══════════════════════════════════════════

def test_guard_uses_gate_reason_verbatim():
    g = _gate([_row(NEW_ENTRY_BLOCKED=1)])
    q = apply_recommendation_quality_guard(g)
    assert "시장 전체 신규진입 차단" in str(q.iloc[0]["QUALITY_GUARD_REASON"])
    assert "알파" not in str(q.iloc[0]["QUALITY_GUARD_REASON"])


def test_guard_falls_back_for_legacy_csv_without_reason_column():
    """구 CSV엔 사유 컬럼이 없다 — 그래도 최상위 차단자는 말해야 한다."""
    g = _gate([_row(NEW_ENTRY_BLOCKED=1)]).drop(columns=["ALPHA_ENTRY_BLOCK_REASON"])
    q = apply_recommendation_quality_guard(g)
    r = str(q.iloc[0]["QUALITY_GUARD_REASON"])
    assert "시장 전체 신규진입 차단" in r, r


def test_legacy_fallback_no_longer_blames_alpha_blindly():
    """알파·LT·차단 모두 아닌 경우의 폴백도 '알파 미달'이라 하지 않는다."""
    g = _gate([_row(ENTRY_RISK_GATE_OK=False)]).drop(
        columns=["ALPHA_ENTRY_BLOCK_REASON"])
    r = str(apply_recommendation_quality_guard(g).iloc[0]["QUALITY_GUARD_REASON"])
    assert "알파 기준" not in r
    assert "진입 조건 미달" in r


# ══════════════════════════════════════════
# ③ 화면 번역이 거짓을 만들지 않는가
# ══════════════════════════════════════════

def test_humanize_stops_calling_it_alpha_shortfall():
    out = _humanize_reason("진입 조건 미달 (알파·자리 외 사유)")
    assert "알파 기준은 통과" in out
    assert out != "AI 알파 진입 기준 미달"


def test_humanize_market_block_says_not_this_stocks_fault():
    out = _humanize_reason("시장 전체 신규진입 차단 (폭락 방어)")
    assert "이 종목만의 문제가 아님" in out


def test_humanize_keeps_real_alpha_shortfall():
    """진짜 알파 미달은 그대로 알파 미달로 보여야 한다."""
    assert "알파 70점" in _humanize_reason("알파 70점 (진입선 85점 미달)")


def test_watch_section_label_matches_actual_sort_key():
    """'조건에 가까운 순서'는 거짓 — 실제는 ALPHA_SCORE 내림차순."""
    src = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "조건에 가까운 순서이며" not in code
    assert "AI 알파 점수 순" in code
    assert "전 종목 신규진입이 보류돼" in code


# ══════════════════════════════════════════
# ④ 실제 배치 데이터 회귀 (있을 때만)
# ══════════════════════════════════════════

def test_real_batch_20260806_reasons_are_truthful():
    path = ROOT / "data" / "recommend_20260806.csv"
    if not path.exists():
        return
    d = pd.read_csv(path, encoding="utf-8-sig", dtype={"종목코드": str}, low_memory=False)
    d["종목코드"] = d["종목코드"].str.zfill(6)
    q = apply_recommendation_quality_guard(ae.apply_alpha_entry_gate(d))
    for code in ["024060", "037070"]:      # 알파 100.0 / 98.2 — 문턱 85 통과
        r = q[q["종목코드"] == code]
        if not len(r):
            continue
        row = r.iloc[0]
        assert float(row["ALPHA_SCORE"]) >= float(row["ALPHA_ENTRY_THRESHOLD"])
        reason = str(row["QUALITY_GUARD_REASON"])
        assert "알파" not in reason, f"{code} 알파 통과인데 알파 탓: {reason}"
        assert "시장 전체 신규진입 차단" in reason
