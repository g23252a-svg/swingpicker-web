# -*- coding: utf-8 -*-
"""[v45] 픽 전달 경로 방어 — 결측 컬럼 크래시 수정 + 분산 권고 정정.

사용자 질문("개선을 여러 번 했는데 한 번도 수익종목 제안이 없었다 — 진짜
나오게 바꾼 게 맞나")을 검증하다 발견한 두 결함:

1) ALPHA_VALIDATED / 임의 지표 컬럼이 없는 배치에서 게이트 함수가
   AttributeError('int'/'bool' object has no attribute 'fillna')로 죽었고,
   finalize의 try/except가 이를 삼켜 ROUTE 치료·알파 게이트가 조용히 스킵됐다.
   → 픽이 나올 수 있는 날에도 게이트가 통째로 건너뛰어질 수 있는 경로.

2) v37이 넣은 '당일 3~5종목 분산' 권고는 구 게이트(알파≥80, 저점추세 미적용)
   풀에서 측정한 값. 현행 게이트(알파≥85 + 저점추세≥0)로 재측정하면 정반대:
     top1 +2.86%/일 승률 50% (누적 +183%) vs top3 -0.12%/47%
     페어드 t=-2.92 p=0.005 → 당일 분산이 유의하게 열위.
   화면 권고가 엔진의 실측 최적 동작(하루 1종목)과 모순되던 것을 정정.

검증 결과(참고): 7/30 배치에서 차단 플래그만 해제하면
  알파≥85 44개 → 저점추세≥0 22개 → 리스크게이트 8개 → TOP_PICK 8개
  → BUY_NOW_ELIGIBLE 7개 → PRODUCTION_BUY 1개 (정상 생성).
즉 픽이 안 나온 원인은 게이트 과잉이 아니라 폭락방어(risk_off) 단일 병목.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import alpha_engine as ae


# ── ① 결측 컬럼 크래시 방어 ──

_MINIMAL = {"종목코드": "005930", "ROUTE": "WAIT", "ALPHA_SCORE": 90.0,
            "Above_MA20": 1, "Low_Trend_PCT": 5.0}


def test_route_heal_survives_missing_alpha_validated():
    # ALPHA_VALIDATED 컬럼 자체가 없는 배치 (구 CSV/degraded)
    out = ae.recompute_route_with_alpha(pd.DataFrame([_MINIMAL]))
    assert out.iloc[0]["ROUTE"] == "WAIT"        # 미검증 → 치료 스킵, 크래시 없음


def test_gate_survives_missing_alpha_validated():
    out = ae.apply_alpha_entry_gate(pd.DataFrame([_MINIMAL]))
    assert int(out.iloc[0]["ALPHA_GATE_ACTIVE"]) == 0


def test_route_heal_survives_missing_indicator_columns():
    # TTM_SQUEEZE / ret_5d_% / MFI14 / RSI14 / 이격도 전부 없는 배치
    row = dict(_MINIMAL); row["ALPHA_VALIDATED"] = 1; row["MARKET_REGIME"] = "NEUTRAL"
    out = ae.recompute_route_with_alpha(pd.DataFrame([row]))
    assert out.iloc[0]["ROUTE"] in ("ATTACK", "ARMED", "WAIT", "OVERHEAT")
    assert "CRASH_CHASE_WARN" in out.columns


def test_validated_flag_helper_is_column_safe():
    assert ae._alpha_validated_flag(pd.DataFrame()) is False
    assert ae._alpha_validated_flag(pd.DataFrame([{"a": 1}])) is False
    assert ae._alpha_validated_flag(pd.DataFrame([{"ALPHA_VALIDATED": 1}])) is True
    # 문자열/결측 혼재도 안전
    assert ae._alpha_validated_flag(
        pd.DataFrame([{"ALPHA_VALIDATED": None}, {"ALPHA_VALIDATED": "1"}])) is True


def test_full_gate_chain_runs_on_minimal_batch():
    # 크래시 없이 게이트 체인 통과 (픽 여부와 무관하게 경로가 살아있어야 함)
    row = dict(_MINIMAL); row["ALPHA_VALIDATED"] = 1
    df = pd.DataFrame([row])
    out = ae.apply_alpha_entry_gate(ae.recompute_route_with_alpha(df))
    assert "TOP_PICK" in out.columns and "ALPHA_ENTRY_OK" in out.columns


# ── ② 분산 권고 정정 ──

def test_stale_diversification_claim_removed():
    src = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")
    assert "48%→56%" not in src, "구 게이트 기준의 잘못된 승률 주장 잔존"
    assert "3~5종목 분산 (실측" not in src
    assert "하루 1종목" in src and "당일 분산은 실측 열위" in src


# ── ③ 켈리 사이징 정상화 (침묵의 0주 제거) ──

def _kelly_row(**ov):
    r = {
        "종목코드": "005930", "종목명": "테스트", "ROUTE": "ATTACK",
        "ALPHA_GATE_ACTIVE": 1, "ALPHA_VALIDATED": 1,
        "ALPHA_SCORE": 95.0, "ALPHA_WIN_PROB": 0.41,
        "ALPHA_ENTRY_THRESHOLD": 85.0, "ELITE_SCORE": 20.0,
        "추천매수가": 10000.0, "손절가": 9200.0, "추천매도가1": 11600.0,
    }
    r.update(ov)
    return pd.DataFrame([r])


def test_guard_multiplier_no_longer_zeroes_sizing():
    """가드 배수는 실측 역선택 — 사이징에서 분리(게이트 통과 종목 +1.12% vs 허용 -2.76%)."""
    import kelly_calibrator as kc
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        blocked = kc.apply_kelly_calibrated(
            _kelly_row(GUARD_KELLY_MULT=0.0), out_dir=td, method="ELITE_SCORE",
            kelly_multiplier=0.5, max_allocation=0.25, min_score_threshold=60.0)
        assert float(blocked.iloc[0]["KELLY_FRACTION"]) > 0, "가드 0배가 여전히 사이징을 죽임"
        # 손익비도 가드로 훼손되지 않아야 한다(진단 정보 보존).
        assert float(blocked.iloc[0]["KELLY_FINAL_B"]) > 0


def test_alpha_empirical_b_updated_to_measured():
    src = (ROOT / "kelly_calibrator.py").read_text(encoding="utf-8")
    assert "_ALPHA_EMPIRICAL_B_5D = 1.81" in src, "현행 게이트 실측 b(1.81) 미반영"


def test_zero_size_always_has_reason():
    """0주면 반드시 사유가 있어야 한다 — '침묵의 0주' 금지."""
    import kelly_calibrator as kc
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # 목표가가 진입가와 같아 손익비 0 → 0주 + 사유
        out = kc.apply_kelly_calibrated(
            _kelly_row(추천매도가1=10000.0), out_dir=td, method="ELITE_SCORE",
            kelly_multiplier=0.5, max_allocation=0.25, min_score_threshold=60.0)
        assert int(out.iloc[0]["켈리_수량"]) == 0
        assert str(out.iloc[0]["KELLY_ZERO_REASON"]).strip() != ""


def test_need_win_rate_exposed():
    import kelly_calibrator as kc
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = kc.apply_kelly_calibrated(
            _kelly_row(ALPHA_WIN_PROB=0.20), out_dir=td, method="ELITE_SCORE",
            kelly_multiplier=0.5, max_allocation=0.25, min_score_threshold=60.0)
        need = float(out.iloc[0]["KELLY_NEED_WIN_RATE"])
        assert 0.0 < need < 1.0
        assert "필요 승률" in str(out.iloc[0]["KELLY_ZERO_REASON"])


def test_inactive_route_zero_has_reason():
    import kelly_calibrator as kc
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = kc.resize_kelly_with_alpha(_kelly_row(ROUTE="WAIT"), td)
        assert int(out.iloc[0]["켈리_수량"]) == 0
        assert "신규진입 상태 아님" in str(out.iloc[0]["KELLY_ZERO_REASON"])
