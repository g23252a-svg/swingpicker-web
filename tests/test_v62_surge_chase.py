# -*- coding: utf-8 -*-
"""[v62] 진입일 급등 추격 차단 + 픽인데 '픽 아님'이라던 라벨 모순 수정.

■ A. 급등 추격 브레이크 — 실측 검증을 통과한 첫 구조 브레이크
  v61에서 씨어스 유형(장기 하락·POC 아래) 브레이크 9개를 검정해 **전부 기각**
  했다(|t|<2.0). 그 과정에서 우연히 다른 축이 걸렸고, 그것만 강건성 전수 검정을
  통과했다 — **진입일 등락률**이다.

  알파 상위(당일 분위≥85) 1,015행·21일, 실현수익 정의는 SSOT
  (진입 t+1 시가 · -8% 장중 손절 · t+5 종가). 겹치지 않는 구간별 5일 실현수익:
      ret_1d < 0%  554행  +2.01%  승률 56.9%
      0~3%         301행  +1.17%  승률 52.8%
      3~5%          71행  +0.80%  승률 53.5%
      5~7%          41행  -1.51%  승률 46.3%   ← 부호 전환
      ≥7%           48행  -1.26%  승률 37.5%   (중위 -2.16%)
  단조 용량-반응이고 +5%에서 꺾인다.

  차단군 vs 잔여군 일별 paired t (5일): -2.99%p · t=-3.24 · p=0.0071
    · BH-FDR q=0.046 (임계 4종 × 호라이즌 3종 = 12개 가족)
    · 블록 부트스트랩 CI95 [-4.19%, -1.43%] — 0 배제
    · 이상치 강건: 중위 -3.06% · 10% 절사 -2.99% · 상위2일 제외 -2.15%
    · IS/OOS 부호 일치 (IS -4.21% t=-4.93 / OOS -2.23% t=-1.62)
    · 10일 호라이즌 동방향 -3.23% (t=-3.19, q=0.046)
    · 대리변수 아님 — 이격도<10 통제 -1.89%(t=-1.96), RSI<70 통제 -2.38%(t=-3.02)
  기존 방어가 못 잡던 공백: NO_CHASE_FLAG는 VWAP -35%/POC +80%/5일 +25% 같은
  극단값에서만 물리고, CRASH_CHASE_WARN은 반대 방향(폭락 추격)이다.
  픽 빈도(90일): TOP_PICK 117건 중 7건(6.0%) 제거 · PRODUCTION_BUY 4건 중 0건 ·
  **픽이 0이 되는 날 0일** (v57 교훈).

■ B. 라벨 모순
  TOP_PICK=1·PRODUCTION_BUY=1인 행에 OFFICIAL_BLOCK_REASON_1="TOP_PICK=0",
  OFFICIAL_FUNNEL_STAGE="ENTRY_READY_BUT_NOT_TOP_PICK"이 붙어 있었다
  (2026-08-11 씨어스 458870 · 2026-08-13 큐리옥스 445680 실측).
  퍼널 주석이 알파 게이트보다 앞에서 한 번만 돌아 라벨이 결정보다 먼저 굳었다.
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

import alpha_engine as AE  # noqa: E402

BATCH_0813 = ROOT / "data" / "recommend_20260813.csv"
DOOSAN = "336260"          # 두산퓨얼셀 — 8/13 진입일 +6.42%
CURIOX = "445680"          # 큐리옥스 — 8/13 진입일 -6.71%


def _frame(rows):
    """게이트가 활성화되는 최소 프레임 (검증 통과 + 리스크 가드 통과)."""
    base = []
    for i, ov in enumerate(rows):
        r = {"종목코드": f"{i:06d}", "종목명": f"종목{i}",
             "ALPHA_VALIDATED": 1, "ALPHA_SCORE": 99.0,
             # 저점추세는 **당일 분위** 기준이라 pad보다 높게 둬야 lt_ok가 통과한다
             "ENTRY_RISK_GATE_OK": True, "Low_Trend_PCT": 100.0,
             "ret_1d_%": 0.0, "MARKET_REGIME": "NEUTRAL",
             "TOP_PICK": 1, "종가": 10000.0, "손절가": 9200.0}
        r.update(ov)
        base.append(r)
    df = pd.DataFrame(base)
    # 분위 안정화 표본 미달을 피하려고 저점추세 표본을 채운다
    pad = pd.DataFrame([{**base[0], "종목코드": f"9{i:05d}", "종목명": f"pad{i}",
                         "Low_Trend_PCT": float(i), "ret_1d_%": 0.0,
                         "ALPHA_SCORE": 1.0, "TOP_PICK": 0}
                        for i in range(40)])
    return pd.concat([df, pad], ignore_index=True), len(base)


# ── A. 급등 추격 브레이크 ───────────────────────────────────────
class TestSurgeBrake:
    def test_threshold_constant_is_five(self):
        assert AE._SURGE_CHASE_PCT == 5.0

    def test_surge_blocks_entry(self):
        df, n = _frame([{"ret_1d_%": 6.5}])
        out = AE.apply_alpha_entry_gate(df)
        assert int(out["SURGE_CHASE_FLAG"].iloc[0]) == 1
        assert int(out["ALPHA_SURGE_OK"].iloc[0]) == 0
        assert int(out["ALPHA_ENTRY_OK"].iloc[0]) == 0
        assert int(out["TOP_PICK"].iloc[0]) == 0

    def test_below_threshold_passes(self):
        df, n = _frame([{"ret_1d_%": 4.9}])
        out = AE.apply_alpha_entry_gate(df)
        assert int(out["SURGE_CHASE_FLAG"].iloc[0]) == 0
        assert int(out["ALPHA_ENTRY_OK"].iloc[0]) == 1

    def test_boundary_is_inclusive(self):
        """+5.0% 정확히도 차단 — 구간 실측에서 5~7%가 음수였다."""
        df, n = _frame([{"ret_1d_%": 5.0}])
        out = AE.apply_alpha_entry_gate(df)
        assert int(out["SURGE_CHASE_FLAG"].iloc[0]) == 1

    def test_negative_day_passes(self):
        """진입일 하락은 차단 사유가 아니다 — 실측에서 오히려 +2.01%였다."""
        df, n = _frame([{"ret_1d_%": -6.71}])
        out = AE.apply_alpha_entry_gate(df)
        assert int(out["SURGE_CHASE_FLAG"].iloc[0]) == 0
        assert int(out["ALPHA_ENTRY_OK"].iloc[0]) == 1

    def test_missing_ret1d_passes(self):
        """결측은 통과 — 데이터 이슈가 전면 차단으로 번지면 안 된다."""
        df, n = _frame([{"ret_1d_%": np.nan}])
        out = AE.apply_alpha_entry_gate(df)
        assert int(out["SURGE_CHASE_FLAG"].iloc[0]) == 0
        assert int(out["ALPHA_ENTRY_OK"].iloc[0]) == 1

    def test_falls_back_to_등락률(self):
        df, n = _frame([{"ret_1d_%": np.nan, "등락률": 7.0}])
        df = df.drop(columns=["ret_1d_%"])
        df["등락률"] = df.get("등락률", pd.Series(0.0, index=df.index)).fillna(0.0)
        df.loc[0, "등락률"] = 7.0
        out = AE.apply_alpha_entry_gate(df)
        assert int(out["SURGE_CHASE_FLAG"].iloc[0]) == 1

    def test_no_column_at_all_passes(self):
        df, n = _frame([{"ret_1d_%": 6.0}])
        df = df.drop(columns=["ret_1d_%"])
        out = AE.apply_alpha_entry_gate(df)
        assert int(out["SURGE_CHASE_FLAG"].iloc[0]) == 0

    def test_block_reason_states_the_measurement(self):
        df, n = _frame([{"ret_1d_%": 6.42}])
        out = AE.apply_alpha_entry_gate(df)
        reason = str(out["ALPHA_ENTRY_BLOCK_REASON"].iloc[0])
        assert "6.4" in reason and "급등" in reason
        assert "승률" in reason or "실측" in reason, \
            "차단 사유에 근거가 없다 — 사용자가 왜 막혔는지 알 수 없다"

    def test_rule_string_recorded(self):
        df, n = _frame([{"ret_1d_%": 1.0}])
        out = AE.apply_alpha_entry_gate(df)
        assert "ret_1d" in str(out["ALPHA_SURGE_RULE"].iloc[0])

    def test_brake_is_anded_not_overriding(self):
        """급등이 아니어도 다른 조건이 막으면 여전히 막힌다."""
        df, n = _frame([{"ret_1d_%": 0.0, "ALPHA_SCORE": 1.0}])
        out = AE.apply_alpha_entry_gate(df)
        assert int(out["SURGE_CHASE_FLAG"].iloc[0]) == 0
        assert int(out["ALPHA_ENTRY_OK"].iloc[0]) == 0

    def test_inactive_gate_does_not_block(self):
        """알파 미검증 배치에서는 게이트가 개입하지 않는다(레거시 폴백)."""
        df, n = _frame([{"ret_1d_%": 9.0}])
        df["ALPHA_VALIDATED"] = 0
        out = AE.apply_alpha_entry_gate(df)
        assert int(pd.to_numeric(out.get("ALPHA_GATE_ACTIVE", 0),
                                 errors="coerce").fillna(0).iloc[0]) == 0

    def test_documented_measurement_is_present(self):
        """근거 없는 문턱이 되지 않도록 실측 수치를 소스에 남긴다."""
        src = (ROOT / "alpha_engine.py").read_text(encoding="utf-8")
        block = src[src.find("_SURGE_CHASE_PCT") - 2200:src.find("_SURGE_CHASE_PCT") + 200]
        for token in ("q=0.046", "t=-3.24", "IS/OOS", "픽이 0이 되는 날 0일"):
            assert token in block, f"검증 근거 '{token}'가 주석에 없다"


# ── B. 실데이터 회귀 (8/13 배치) ────────────────────────────────
class TestRealBatch:
    def _batch(self):
        if not BATCH_0813.exists():
            pytest.skip("8/13 배치 CSV 없음")
        d = pd.read_csv(BATCH_0813, dtype={"종목코드": str}, low_memory=False)
        d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
        return d

    def test_surge_pick_is_removed_and_clean_pick_kept(self):
        d = self._batch()
        out = AE.apply_alpha_entry_gate(d.copy())
        tp = pd.to_numeric(out["TOP_PICK"], errors="coerce").fillna(0)
        dsn = out[out["종목코드"] == DOOSAN]
        cur = out[out["종목코드"] == CURIOX]
        if dsn.empty or cur.empty:
            pytest.skip("대상 종목 부재")
        assert float(dsn["ret_1d_%"].iloc[0]) >= AE._SURGE_CHASE_PCT
        assert int(dsn["TOP_PICK"].iloc[0]) == 0, "급등 추격 픽이 남았다"
        assert float(cur["ret_1d_%"].iloc[0]) < 0
        assert int(cur["TOP_PICK"].iloc[0]) == 1, "정상 픽이 함께 죽었다"

    def test_pick_count_reduced_but_not_zeroed(self):
        d = self._batch()
        before = int(pd.to_numeric(d["TOP_PICK"], errors="coerce").fillna(0).sum())
        out = AE.apply_alpha_entry_gate(d.copy())
        after = int(pd.to_numeric(out["TOP_PICK"], errors="coerce").fillna(0).sum())
        assert after < before, "브레이크가 아무것도 막지 않았다 (죽은 게이트)"
        assert after > 0, "픽이 전멸했다 — v57에서 고친 유형의 회귀"

    def test_brake_is_not_a_dead_gate(self):
        """mutation: 문턱을 올리면 차단이 사라져야 한다."""
        d = self._batch()
        orig = AE._SURGE_CHASE_PCT
        try:
            AE._SURGE_CHASE_PCT = 999.0
            out = AE.apply_alpha_entry_gate(d.copy())
            assert int(pd.to_numeric(out["SURGE_CHASE_FLAG"],
                                     errors="coerce").fillna(0).sum()) == 0
        finally:
            AE._SURGE_CHASE_PCT = orig

    def test_historical_replay_never_zeroes_a_pick_day(self):
        """90일 실측 전제 고정 — 브레이크가 픽 없는 날을 만들지 않는다."""
        import glob
        zeroed = 0
        checked = 0
        for f in sorted(glob.glob(str(ROOT / "data" / "recommend_2026*.csv")))[-40:]:
            d = pd.read_csv(f, dtype={"종목코드": str}, low_memory=False)
            if not {"TOP_PICK", "ret_1d_%"} <= set(d.columns):
                continue
            tp = pd.to_numeric(d["TOP_PICK"], errors="coerce").fillna(0) == 1
            if not tp.any():
                continue
            checked += 1
            r1 = pd.to_numeric(d["ret_1d_%"], errors="coerce")
            surge = (r1 >= AE._SURGE_CHASE_PCT).fillna(False)
            if int((tp & ~surge).sum()) == 0:
                zeroed += 1
        if checked == 0:
            pytest.skip("검사 가능한 배치 없음")
        assert zeroed == 0, f"{zeroed}/{checked}일에서 픽이 전멸한다"


# ── C. 라벨 모순 (픽인데 '픽 아님') ─────────────────────────────
class TestFunnelLabelContradiction:
    FIN = ROOT / "pipeline_finalize.py"

    def test_reannotation_runs_after_alpha_gate(self):
        src = self.FIN.read_text(encoding="utf-8")
        i_gate = src.find("df_out = _alpha_gate(df_out)")
        i_re = src.find("[v62] 공식 퍼널 라벨 재계산")
        assert i_gate > 0 and i_re > 0, "배선 누락"
        assert i_gate < i_re, "재계산이 알파 게이트보다 먼저다 — 낡은 값이 그대로 남는다"

    def test_reannotation_protects_contract_columns(self):
        src = self.FIN.read_text(encoding="utf-8")
        blk = src[src.find("[v62] 공식 퍼널 라벨 재계산"):][:2600]
        assert "원복" in blk and "BUY_NOW_ELIGIBLE" in blk, \
            "계약 컬럼 보호 규약이 없다"

    def test_contradiction_existed_in_real_batches(self):
        """전제 재현 — 수정 전 CSV에는 모순이 실제로 있었다."""
        found = 0
        for path, code in ((ROOT / "data" / "recommend_20260813.csv", CURIOX),
                           (ROOT / "data" / "recommend_20260811.csv", "458870")):
            if not path.exists():
                continue
            d = pd.read_csv(path, dtype={"종목코드": str}, low_memory=False)
            if "OFFICIAL_BLOCK_REASON_1" not in d.columns:
                continue
            d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
            s = d[d["종목코드"] == code]
            if s.empty:
                continue
            r = s.iloc[0]
            if int(r.get("TOP_PICK", 0)) == 1 and \
                    "TOP_PICK=0" in str(r.get("OFFICIAL_BLOCK_REASON_1", "")):
                found += 1
        if found == 0:
            pytest.skip("해당 배치 부재")
        assert found >= 1

    def test_no_contradiction_after_reannotation(self):
        """재계산 함수를 직접 적용하면 모순이 사라진다."""
        if not BATCH_0813.exists():
            pytest.skip("배치 없음")
        from pipeline_finalize import add_official_buy_funnel_columns
        d = pd.read_csv(BATCH_0813, dtype={"종목코드": str}, low_memory=False)
        out = add_official_buy_funnel_columns(
            d.copy(), macro_risk="NORMAL", market_breadth=57.0, macro_msg="")
        tp = pd.to_numeric(out["TOP_PICK"], errors="coerce").fillna(0) == 1
        bad = tp & out["OFFICIAL_BLOCK_REASON_1"].astype(str).str.contains("TOP_PICK=0")
        assert int(bad.sum()) == 0, \
            f"재계산 후에도 '픽인데 TOP_PICK=0' 행이 {int(bad.sum())}건 남았다"


# ── D. v61에서 기각한 것들이 몰래 들어오지 않았는지 ──────────────
class TestRejectedStillRejected:
    def test_no_long_downtrend_brake(self):
        """장기 하락 브레이크는 v61에서 기각됐다(t=-0.59). 되살아나면 실패."""
        src = (ROOT / "alpha_engine.py").read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        for pat in (r"ret_120d_%.{0,24}<=\s*-", r"ret_60d_%.{0,24}<=\s*-",
                    r"IS_ABOVE_POC.{0,12}==\s*0"):
            assert not re.search(pat, code), \
                f"기각된 브레이크가 코드에 들어갔다 ({pat})"

    def test_lt_floor_unchanged(self):
        """LT_PCTL<40 차단은 실측 역방향(t=+2.00)이었다 — 문턱 유지."""
        assert AE._LT_PCTL_FLOOR == 0.30
