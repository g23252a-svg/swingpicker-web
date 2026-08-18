# -*- coding: utf-8 -*-
"""[v63] 어제 넣은 v62가 만든 회귀 + 화면 성적이 엔진 성적이 아니었던 것.

■ A. v62가 만든 회귀 (내가 만든 결함)
  v62에서 '픽인데 TOP_PICK=0'이라던 라벨 모순을 고치려고 퍼널 주석을 **알파
  게이트 직후**에 재계산하게 했다. 그런데 그 뒤에 품질게이트가 '당일 신규진입
  1종목 제한'으로 PRODUCTION_BUY를 잘라낸다. 그래서 라벨이 **반대 방향으로**
  낡았다 — 2026-08-17 배치 실측 10건:
      OFFICIAL_FUNNEL_STAGE   = "OFFICIAL_BUY"      ← 공식 매수라고 말하는데
      PRODUCTION_BUY          = 0
      ACTION_DECISION         = WATCH
      OFFICIAL_BLOCK_REASON_2 = "TOP_PICK + BUY_NOW_ELIGIBLE"
      BUY_NOW_ELIGIBLE        = 0                   ← 근거로 든 값이 0
  (v62 이전에는 'TOP_PICK=0'이라고 했다 — 11건. 한쪽을 막고 반대쪽을 열었다.)
  → 라벨은 PRODUCTION_BUY를 바꾸는 **마지막 단계(품질게이트) 뒤**에 굳어야 한다.

■ B. 화면의 "이 엔진의 지난 성적"이 엔진 성적이 아니었다
  v58 리포트의 top1은 전체 풀을 ALPHA_SCORE만으로 정렬한 1등이다. 실제 픽은
  알파 문턱 + 저점추세 분위>30 + 리스크가드 + (v62)급등 제외를 통과한 뒤
  **알파 × 손익비**로 고른다. v58.1이 이 줄을 결정 센터에 띄우고 있었다.
  2026-08-17 시점 실측:
      h5  알파점수 단독 1위  -1.85%   ← 화면에 이렇게 떴다
          엔진 픽 1위      +3.58%  (유니버스 +0.75% · 초과 +2.83%p)
      h3  알파 단독 -1.68%  /  엔진 픽 +0.92%
  즉 **엔진 성적을 5.4%p 나쁘게** 표기했다(v61에서 고친 '표기가 사실과 다름'과
  같은 유형, 방향만 반대). 부수적으로 게이트가 나쁜 꼬리를 실제로 걷어낸다는
  증거이기도 하다.

■ C. 기각 기록 — 하루 1종목 제한 완화
  8/17 배치에서 6종목이 모든 관문을 통과했는데 1개만 채택된다. 완화를 검정했다
  (재구성 후보풀 841행·23일, 실현수익 SSOT):
      상위1 +3.01% / 상위2 +2.64% / 상위3 +2.49% / 상위5 +2.73% / 상위8 +2.07%
      상위3 - 상위1 = -0.52%p (t=-0.61, p=0.55) — 유의하지 않고 방향도 불리
      위험 대비 수익(평균/표준편차): 0.369(N=1) → 0.343(N=3) — 악화
      최악일은 모든 N에서 -8.00% 동일(동반 손절) — 분산이 꼬리를 못 막는다
  → 수익도 위험조정도 근거 없음. 제한 유지.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BATCH = ROOT / "data" / "recommend_20260817.csv"


def _load():
    if not BATCH.exists():
        pytest.skip("8/17 배치 CSV 없음")
    d = pd.read_csv(BATCH, dtype={"종목코드": str}, low_memory=False)
    d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
    return d


def _num(df, c):
    return (pd.to_numeric(df[c], errors="coerce").fillna(0)
            if c in df.columns else pd.Series(0, index=df.index))


def _contradictions(df):
    """라벨과 결정이 어긋난 건수 (양방향)."""
    st = df.get("OFFICIAL_FUNNEL_STAGE", pd.Series("", index=df.index)).astype(str)
    r1 = df.get("OFFICIAL_BLOCK_REASON_1", pd.Series("", index=df.index)).astype(str)
    r2 = df.get("OFFICIAL_BLOCK_REASON_2", pd.Series("", index=df.index)).astype(str)
    a = int(((st == "OFFICIAL_BUY") & (_num(df, "PRODUCTION_BUY") == 0)).sum())
    b = int((r2.str.contains("BUY_NOW_ELIGIBLE")
             & (_num(df, "BUY_NOW_ELIGIBLE") == 0)).sum())
    c = int(((_num(df, "TOP_PICK") == 1)
             & r1.str.contains("TOP_PICK=0")).sum())
    return {"official_buy_but_not_bought": a, "cited_value_is_zero": b,
            "pick_but_labeled_not_pick": c, "total": a + b + c}


# ── A. 라벨은 품질게이트 뒤에 굳는다 ────────────────────────────
class TestLabelAfterQualityGuard:
    FIN = ROOT / "pipeline_finalize.py"

    def test_reannotation_is_after_quality_guard(self):
        src = self.FIN.read_text(encoding="utf-8")
        i_gate = src.find("df_out = _alpha_gate(df_out)")
        i_q = src.find("df_out = apply_recommendation_quality_guard(df_out)")
        i_re = src.find("[v63] 공식 퍼널 라벨 재계산")
        assert i_gate > 0 and i_q > 0 and i_re > 0, "배선 누락"
        assert i_gate < i_q < i_re, (
            "재계산이 품질게이트보다 앞이다 — v62가 만든 회귀가 되살아났다 "
            "(품질게이트가 PRODUCTION_BUY를 바꾸는 마지막 단계다)")

    def test_v62_placement_is_gone(self):
        """알파 게이트 직후 재계산(v62 배치)이 남아 있으면 안 된다."""
        src = self.FIN.read_text(encoding="utf-8")
        assert "[v62] 공식 퍼널 라벨 재계산" not in src, \
            "v62의 잘못된 위치가 그대로 남아 있다"

    def test_reannotation_protects_contract_columns(self):
        src = self.FIN.read_text(encoding="utf-8")
        blk = src[src.find("[v63] 공식 퍼널 라벨 재계산"):][:3000]
        assert "원복" in blk
        for c in ("TOP_PICK", "BUY_NOW_ELIGIBLE", "PRODUCTION_BUY"):
            assert c in blk, f"{c} 보호가 없다"

    def test_real_batch_had_both_directions_of_contradiction(self):
        """전제 재현 — 8/17 원본에는 A·B 모순이 실제로 있었다."""
        d = _load()
        c = _contradictions(d)
        assert c["official_buy_but_not_bought"] > 0, "재현 전제가 깨졌다"
        assert c["cited_value_is_zero"] > 0

    def test_reannotation_clears_all_contradictions(self):
        d = _load()
        from pipeline_finalize import add_official_buy_funnel_columns
        out = add_official_buy_funnel_columns(
            d.copy(), macro_risk="NORMAL", market_breadth=57.0, macro_msg="")
        c = _contradictions(out)
        assert c["total"] == 0, f"재계산 후에도 모순 남음: {c}"

    def test_reannotation_does_not_change_decisions(self):
        d = _load()
        from pipeline_finalize import add_official_buy_funnel_columns
        out = add_official_buy_funnel_columns(
            d.copy(), macro_risk="NORMAL", market_breadth=57.0, macro_msg="")
        for c in ("TOP_PICK", "BUY_NOW_ELIGIBLE", "PRODUCTION_BUY", "BUY_NOW_GRADE"):
            if c in d.columns:
                assert d[c].equals(out[c]), f"{c}가 바뀌었다 — 표시 전용이어야 한다"


# ── B. 성적 표기는 엔진이 실제로 고르는 규칙으로 ────────────────
class TestTrackRecordMeasuresTheEngine:
    def test_gate_rank_key_applies_the_funnel(self):
        from services.alpha_live_report import _gate_rank_key, _SURGE_PCT
        rec = pd.DataFrame({
            "ALPHA_SCORE":           [99.0, 98.0, 97.0, 50.0, 96.0],
            "ALPHA_ENTRY_THRESHOLD": [90.0] * 5,
            "LOW_TREND_PCTL":        [80.0, 10.0, 70.0, 90.0, 75.0],
            "ret_1d_%":              [0.0,  0.0,  9.0,  0.0,  1.0],
            "ENTRY_RISK_GATE_OK":    [True, True, True, True, False],
            "RR_NOW_TP1":            [2.0,  2.0,  2.0,  2.0,  2.0],
        })
        key = _gate_rank_key(rec)
        assert key is not None
        ok = key.notna()
        assert bool(ok.iloc[0]), "정상 후보가 탈락했다"
        assert not bool(ok.iloc[1]), "저점추세 하위가 통과했다"
        assert not bool(ok.iloc[2]), f"진입일 +9%(>{_SURGE_PCT}) 급등이 통과했다"
        assert not bool(ok.iloc[3]), "알파 문턱 미달이 통과했다"
        assert not bool(ok.iloc[4]), "리스크가드 미달이 통과했다"

    def test_rank_key_is_alpha_times_rr(self):
        from services.alpha_live_report import _gate_rank_key
        rec = pd.DataFrame({
            "ALPHA_SCORE": [90.0, 95.0],
            "RR_NOW_TP1":  [3.0,  1.0],
        })
        key = _gate_rank_key(rec)
        assert key.iloc[0] > key.iloc[1], "손익비가 랭킹에 반영되지 않았다"

    def test_rr_is_clipped_at_three(self):
        from services.alpha_live_report import _gate_rank_key
        rec = pd.DataFrame({"ALPHA_SCORE": [90.0, 90.0],
                            "RR_NOW_TP1": [3.0, 99.0]})
        key = _gate_rank_key(rec)
        assert key.iloc[0] == key.iloc[1], "RR 상한(3)이 적용되지 않았다"

    def test_missing_columns_do_not_crash(self):
        """pd.to_numeric(None)은 스칼라 NaN을 준다 — 첫 구현이 여기서 터졌다."""
        from services.alpha_live_report import _gate_rank_key
        assert _gate_rank_key(pd.DataFrame({"ALPHA_SCORE": [90.0] * 3})) is not None
        assert _gate_rank_key(pd.DataFrame({"기타": [1, 2]})) is None

    def test_thresholds_match_alpha_engine(self):
        """리포트가 엔진과 다른 문턱을 쓰면 다른 것을 측정한다 — 그게 이 결함이었다."""
        import alpha_engine as AE
        from services import alpha_live_report as ALR
        assert ALR._SURGE_PCT == AE._SURGE_CHASE_PCT
        assert ALR._LT_PCTL_FLOOR_PCT == AE._LT_PCTL_FLOOR * 100

    def test_line_headlines_engine_pick_not_raw_alpha(self):
        from services.alpha_live_report import alpha_live_line
        rep = {"ok": True, "horizons": {"h5": {
            "n_days": 18, "ic_mean": 0.0967, "ic_positive_days": 15,
            "ic_t": {"t": 2.386},
            "top1": {"mean_pct": -1.845, "excess_mean_pct": -2.59, "stop_rate": 0.33},
            "gated_top1": {"mean_pct": 3.577, "excess_mean_pct": 2.83,
                           "stop_rate": 0.28, "n_days": 18},
        }}}
        line = alpha_live_line(rep, 5)
        assert "엔진 1위 +3.58%" in line, f"엔진 픽 기준이 헤드라인이 아니다: {line}"
        assert "-1.84" not in line, "알파 단독 1위를 성적으로 표기한다"

    def test_line_is_explicit_when_gate_cannot_be_rebuilt(self):
        from services.alpha_live_report import alpha_live_line
        rep = {"ok": True, "horizons": {"h5": {
            "n_days": 6, "ic_mean": 0.05, "ic_positive_days": 4,
            "ic_t": {"t": 1.0},
            "top1": {"mean_pct": -1.0, "excess_mean_pct": -1.5, "stop_rate": 0.2},
        }}}
        line = alpha_live_line(rep, 5)
        assert "알파점수 단독" in line and "재구성 불가" in line, \
            "재구성 불가를 밝히지 않고 raw 값을 성적처럼 적는다"

    def test_real_report_shows_gap_between_raw_and_engine(self):
        """실데이터 전제 고정 — 두 축이 실제로 크게 다르다."""
        import json
        p = ROOT / "data" / "alpha_live_report_latest.json"
        if not p.exists():
            pytest.skip("리포트 없음")
        blk = (json.loads(p.read_text(encoding="utf-8"))
               .get("horizons", {}).get("h5"))
        if not blk or "gated_top1" not in blk:
            pytest.skip("아직 gated 축이 없는 구 리포트")
        raw = blk["top1"]["mean_pct"]
        eng = blk["gated_top1"]["mean_pct"]
        assert eng > raw, (
            "엔진 픽이 알파 단독보다 나쁘다 — 게이트의 존재 근거를 재검토해야 한다")


# ── C. 기각 기록 — 하루 1종목 제한 완화 ─────────────────────────
class TestDailyCapStays:
    def test_single_pick_selection_unchanged(self):
        src = (ROOT / "services" / "recommendation_quality.py").read_text(
            encoding="utf-8")
        assert 'rank(method="first", ascending=False).eq(1)' in src, \
            "하루 1종목 선정이 바뀌었다 — 완화 근거는 실측으로 기각됐다 " \
            "(상위3-상위1 = -0.52%p, t=-0.61, p=0.55 · 위험조정 0.369→0.343)"

    def test_rejection_is_documented(self):
        src = (ROOT / "tests" / "test_v63_label_and_track_record.py").read_text(
            encoding="utf-8")
        for token in ("t=-0.61", "0.369", "-8.00%"):
            assert token in src, f"기각 근거 '{token}'가 기록되지 않았다"
