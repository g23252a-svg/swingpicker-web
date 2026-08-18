# -*- coding: utf-8 -*-
"""[v60] 보유(CARRY) 행이 보강 단계를 건너뛰어 점수가 3개월간 비어 있던 것 — 회귀 봉쇄.

v59가 남긴 질문("결측 30행이 어느 경로로 늦게 합류하는가")의 답은 **늦게 합류가
아니었다** — 파이프라인이 CARRY 행을 **더 짧은 경로로 다시 만든다.**

  Stage 3 run_scoring    업종 분류·섹터 모멘텀·수급·시총기준일
                         + LDY/TOTAL/RANK_SCORE·벤치_60d_* 대입 (457행)
  Stage 4 enrich_news    NEWS_SCORE
  Stage 5 run_calibration ★ CARRY 행이 concat으로 합류 (511행)
  Stage 6 finalize

붕괴 사슬 (실측)
  녹십자 006280   08/05 FRESH LDY_SCORE=39.8 → 08/06 CARRY_REFRESHED ∅
  지아이텍 382480  05/14 FRESH → 05/15 CARRY ∅ → 08/07 여전히 ∅ (2개월 반)
  2026-08-07 배치 업종 결측 30종목 = **100% CARRY** (FRESH 285행은 결측 0)

이 파일이 막는 것
  A) 복원  — DISPLAY_SCORE 별칭 3종·섹터 집계·배치 스칼라가 되살아난다
  B) 정직  — 돌리지 않은 단계(뉴스·전략)는 **채우지 않는다**
  C) 무해  — FRESH 행 값·dtype이 바뀌지 않는다
  D) 계약  — FRESH엔 있고 CARRY엔 없는 컬럼은 선언된 목록에만 있어야 한다
             (3개월 전 이 게이트가 있었으면 바로 걸렸다)
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import carry_backfill as cb  # noqa: E402

REAL_CSV = ROOT / "data" / "recommend_20260807.csv"


def _frame(rows):
    return pd.DataFrame(rows)


def _mini():
    """FRESH 2행 + CARRY 2행. CARRY는 보강 컬럼이 비어 있다."""
    return _frame([
        {"종목코드": "000001", "종목명": "프레시가", "ROW_BUILD_MODE": "FRESH",
         "DISPLAY_SCORE": 70.0, "LDY_SCORE": 70.0, "TOTAL_SCORE": 70.0,
         "RANK_SCORE": 70.0, "업종": "의약품 제조업", "업종_대분류": "바이오·의약품",
         "업종_상세": "의약품 제조업", "SECTOR_RET_5D": 3.0, "SECTOR_RS": 1.5,
         "SECTOR_RANK": 2.0, "NEWS_SCORE": 0.0, "개인순매수": 10.0},
        {"종목코드": "000002", "종목명": "프레시나", "ROW_BUILD_MODE": "FRESH",
         "DISPLAY_SCORE": 60.0, "LDY_SCORE": 60.0, "TOTAL_SCORE": 60.0,
         "RANK_SCORE": 60.0, "업종": "특수 목적용 기계", "업종_대분류": "조선·기계·설비",
         "업종_상세": "특수 목적용 기계", "SECTOR_RET_5D": -1.0, "SECTOR_RS": -0.4,
         "SECTOR_RANK": 9.0, "NEWS_SCORE": 0.0, "개인순매수": -5.0},
        {"종목코드": "006280", "종목명": "녹십자", "ROW_BUILD_MODE": "CARRY_REFRESHED",
         "DISPLAY_SCORE": 36.7, "LDY_SCORE": np.nan, "TOTAL_SCORE": np.nan,
         "RANK_SCORE": np.nan, "업종": "의약품 제조업", "업종_대분류": np.nan,
         "업종_상세": np.nan, "SECTOR_RET_5D": np.nan, "SECTOR_RS": np.nan,
         "SECTOR_RANK": np.nan, "NEWS_SCORE": np.nan, "개인순매수": np.nan},
        {"종목코드": "382480", "종목명": "지아이텍", "ROW_BUILD_MODE": "CARRY_LEGACY",
         "DISPLAY_SCORE": 21.0, "LDY_SCORE": np.nan, "TOTAL_SCORE": np.nan,
         "RANK_SCORE": np.nan, "업종": "특수 목적용 기계", "업종_대분류": np.nan,
         "업종_상세": np.nan, "SECTOR_RET_5D": np.nan, "SECTOR_RS": np.nan,
         "SECTOR_RANK": np.nan, "NEWS_SCORE": np.nan, "개인순매수": np.nan},
    ])


# ── A. 복원 ────────────────────────────────────────────────────
class TestRestore:
    def test_display_aliases_restored(self):
        out, rep = cb.backfill_carry_rows(_mini())
        c = cb.carry_mask(out)
        for col in cb.DISPLAY_ALIASES:
            assert out.loc[c, col].notna().all(), f"{col} 복원 안 됨"
            assert np.allclose(
                pd.to_numeric(out.loc[c, col]),
                pd.to_numeric(out.loc[c, "DISPLAY_SCORE"])), \
                f"{col}이 DISPLAY_SCORE와 다르다 — 별칭 정의 위반"
        assert rep["filled"]["LDY_SCORE"] == 2

    def test_alias_definition_matches_pipeline(self):
        """별칭 정의의 SSOT가 pipeline_calibrate라는 사실을 소스로 고정.

        저쪽이 별칭을 그만두면 이 백필도 근거를 잃는다. 그때는 이 테스트가
        먼저 실패해서 백필 정의를 다시 보게 만든다.
        """
        src = (ROOT / "pipeline_calibrate.py").read_text(encoding="utf-8")
        for col in cb.DISPLAY_ALIASES:
            assert f'df_out["{col}"] = df_out["{cb.DISPLAY_COL}"]' in src, \
                f"{col}이 더는 {cb.DISPLAY_COL}의 별칭이 아니다 — 백필 근거 재검토 필요"

    def test_sector_agg_taken_from_same_sector_not_recomputed(self):
        """섹터 집계는 **같은 섹터의 FRESH 행**에서 가져온다.

        캐리 2~3행만으로 평균을 다시 내면 '섹터 평균'이 아니라 '그 몇 행의
        평균'이 된다 — v59에서 고친 가짜 섹터와 같은 종류의 오류다.
        """
        out, _ = cb.backfill_carry_rows(_mini())
        nok = out[out["종목명"] == "녹십자"].iloc[0]
        gi = out[out["종목명"] == "지아이텍"].iloc[0]
        assert float(nok["SECTOR_RET_5D"]) == 3.0    # 바이오·의약품 FRESH 값
        assert float(gi["SECTOR_RET_5D"]) == -1.0    # 조선·기계·설비 FRESH 값
        assert float(nok["SECTOR_RANK"]) == 2.0
        assert float(gi["SECTOR_RANK"]) == 9.0

    def test_batch_scalars_restored(self):
        out, rep = cb.backfill_carry_rows(
            _mini(), bench_map={"KOSPI": {60: 11.1}, "KOSDAQ": {60: 4.4}},
            mcap_ymd="20260807")
        c = cb.carry_mask(out)
        assert (pd.to_numeric(out.loc[c, "벤치_60d_KOSPI_%"]) == 11.1).all()
        assert (out.loc[c, "시총기준일"].astype(str) == "20260807").all()

    def test_sector_detail_from_raw(self):
        out, _ = cb.backfill_carry_rows(_mini())
        c = cb.carry_mask(out)
        assert (out.loc[c, "업종_상세"].astype(str)
                == out.loc[c, "업종"].astype(str)).all()

    def test_existing_carry_values_not_overwritten(self):
        """스냅샷에서 물려받은 값이 있으면 그것이 그 시점의 진실이다."""
        df = _mini()
        df.loc[df["종목명"] == "지아이텍", "LDY_SCORE"] = 99.0
        out, _ = cb.backfill_carry_rows(df)
        gi = out[out["종목명"] == "지아이텍"].iloc[0]
        assert float(gi["LDY_SCORE"]) == 99.0, "기존 값을 덮어썼다"

    def test_idempotent(self):
        out1, _ = cb.backfill_carry_rows(_mini())
        out2, rep2 = cb.backfill_carry_rows(out1)
        assert sum((rep2.get("filled") or {}).values()) == 0
        pd.testing.assert_frame_equal(out1, out2)

    def test_no_carry_rows_is_noop(self):
        df = _mini()
        df["ROW_BUILD_MODE"] = "FRESH"
        out, rep = cb.backfill_carry_rows(df)
        assert rep["carry_rows"] == 0
        assert not rep["filled"]

    def test_missing_display_score_reported_not_silent(self):
        df = _mini().drop(columns=["DISPLAY_SCORE"])
        out, rep = cb.backfill_carry_rows(df)
        assert "없음" in rep["note"], "DISPLAY_SCORE 부재를 조용히 넘겼다"


# ── B. 정직: 돌리지 않은 단계는 채우지 않는다 ───────────────────
class TestHonesty:
    def test_news_score_not_fabricated(self):
        """NEWS_SCORE=0.0은 '확인했고 특이사항 없음'이라는 관측값이다.

        뉴스 단계를 돌리지 않은 캐리 행에 그것을 넣으면 하지 않은 확인을
        했다고 말하는 것이다.
        """
        out, rep = cb.backfill_carry_rows(_mini())
        c = cb.carry_mask(out)
        assert out.loc[c, "NEWS_SCORE"].isna().all(), \
            "뉴스를 수집하지 않았는데 NEWS_SCORE를 만들어냈다"
        assert "NEWS_SCORE" in rep["left_blank"]
        assert rep["left_blank"]["NEWS_SCORE"]["rows"] == 2

    def test_blank_reason_is_recorded(self):
        df = _mini()
        df["NEWS_REASON"] = np.nan
        df["HAS_NEWS"] = True
        out, _ = cb.backfill_carry_rows(df)
        c = cb.carry_mask(out)
        assert (out.loc[c, "NEWS_REASON"].astype(str)
                == cb.CARRY_BLANK_REASON).all()
        assert (~out.loc[c, "HAS_NEWS"].astype(bool)).all()

    def test_carry_state_always_has_a_reason(self):
        """상태가 CARRY인데 이유가 없으면 화면이 설명 없이 보유를 표시한다.

        실측(2026-08-12 배치): legacy 38행은 파이프라인이 사유를 넣지만
        CARRY_REFRESHED 12행은 공백이었다 — 재분석에 성공한 행인데도.
        """
        df = _mini()
        df["ROUTE_REASON"] = np.nan
        out, rep = cb.backfill_carry_rows(df)
        c = cb.carry_mask(out)
        assert out.loc[c, "ROUTE_REASON"].notna().all(), "캐리 사유가 비어 있다"
        nok = out[out["종목명"] == "녹십자"].iloc[0]        # CARRY_REFRESHED
        gi = out[out["종목명"] == "지아이텍"].iloc[0]        # CARRY_LEGACY
        assert "재분석 완료" in str(nok["ROUTE_REASON"])
        assert "스냅샷" in str(gi["ROUTE_REASON"])
        assert rep["filled"]["ROUTE_REASON"] == 2

    def test_existing_route_reason_kept(self):
        """파이프라인이 이미 넣은 문구를 덮어쓰지 않는다."""
        df = _mini()
        # pandas 3에서는 float64 컬럼에 문자열을 대입하면 TypeError다 —
        # 처음부터 object로 만든다.
        df["ROUTE_REASON"] = pd.Series([None] * len(df), dtype="object")
        df.loc[df["종목명"] == "지아이텍", "ROUTE_REASON"] = "캐리 재계산 실패: legacy snapshot"
        out, _ = cb.backfill_carry_rows(df)
        gi = out[out["종목명"] == "지아이텍"].iloc[0]
        assert str(gi["ROUTE_REASON"]) == "캐리 재계산 실패: legacy snapshot"

    def test_strategy_columns_not_fabricated(self):
        df = _mini()
        for col in ("STRATEGY", "STRATEGY_SCORE", "STRATEGY_HORIZON"):
            df[col] = np.nan
        out, rep = cb.backfill_carry_rows(df)
        c = cb.carry_mask(out)
        for col in ("STRATEGY", "STRATEGY_SCORE", "STRATEGY_HORIZON"):
            assert out.loc[c, col].isna().all(), f"{col}을 만들어냈다"
            assert col in rep["left_blank"]

    def test_individual_flow_not_zero_filled_without_map(self):
        """개인순매수 0은 '순매수 없음'이라는 관측값이지 미수집이 아니다."""
        out, rep = cb.backfill_carry_rows(_mini(), individual_net_map=None)
        c = cb.carry_mask(out)
        assert out.loc[c, "개인순매수"].isna().all()
        assert "개인순매수" not in rep["filled"]

    def test_individual_flow_filled_when_map_given(self):
        out, rep = cb.backfill_carry_rows(
            _mini(), individual_net_map={"006280": -1234.0})
        nok = out[out["종목명"] == "녹십자"].iloc[0]
        gi = out[out["종목명"] == "지아이텍"].iloc[0]
        assert float(nok["개인순매수"]) == -1234.0
        assert pd.isna(gi["개인순매수"]), "맵에 없는 종목을 0으로 채웠다"

    def test_log_line_states_intentional_blanks(self):
        out, rep = cb.backfill_carry_rows(_mini())
        line = cb.carry_backfill_line(rep)
        assert "캐리 2/4행" in line
        assert "복원" in line
        assert "의도적 공백" in line, "채우지 않은 사실이 로그에 없다"

    def test_quiet_when_nothing_to_do(self):
        df = _mini()
        df["ROW_BUILD_MODE"] = "FRESH"
        assert cb.carry_backfill_line(cb.backfill_carry_rows(df)[1]) == ""


# ── C. 무해: FRESH 행과 dtype을 건드리지 않는다 ─────────────────
class TestNoSideEffects:
    def test_fresh_rows_unchanged(self):
        df = _mini()
        out, _ = cb.backfill_carry_rows(
            df, bench_map={"KOSPI": {60: 1.0}}, mcap_ymd="20260807")
        f = ~cb.carry_mask(df)
        for col in df.columns:
            assert df.loc[f, col].equals(out.loc[f, col]), \
                f"FRESH 행의 {col}이 바뀌었다"

    def test_dtypes_preserved(self):
        """숫자 컬럼이 조용히 object가 되면 뒤쪽 연산·차트가 엉뚱해진다.

        v55.4에서 pandas 2/3 dtype 계약으로 같은 유형의 사고를 겪었고,
        v59의 repair_sector도 str→object로 열화시켰다(v60에서 함께 고쳤다).
        """
        df = _mini()
        out, _ = cb.backfill_carry_rows(
            df, bench_map={"KOSPI": {60: 1.0}}, mcap_ymd="20260807")
        changed = [(c, str(df[c].dtype), str(out[c].dtype))
                   for c in df.columns if df[c].dtype != out[c].dtype]
        assert not changed, f"dtype이 열화됐다: {changed}"

    def test_sector_repair_preserves_dtype(self):
        """v59 repair_sector의 dtype 열화 회귀 가드."""
        from services import sector_repair as sr
        df = pd.DataFrame({
            "종목명": ["녹십자", "삼성바이오"],
            "업종": ["의약품 제조업", "의약품 제조업"],
            "업종_대분류": pd.Series([None, "바이오·의약품"], dtype="object"),
        })
        df["업종_대분류"] = df["업종_대분류"].astype("str")
        out, _ = sr.repair_sector(df)
        assert out["업종_대분류"].dtype == df["업종_대분류"].dtype, \
            "repair_sector가 dtype을 열화시켰다"

    def test_empty_frame_safe(self):
        out, rep = cb.backfill_carry_rows(pd.DataFrame())
        assert rep["ok"] is False and "빈" in rep["note"]

    def test_missing_build_mode_treated_as_fresh(self):
        df = _mini().drop(columns=["ROW_BUILD_MODE"])
        out, rep = cb.backfill_carry_rows(df)
        assert rep["carry_rows"] == 0


# ── D. 계약 게이트: 3개월 전에 걸렸어야 했던 검사 ────────────────
class TestFreshCarryColumnContract:
    def _real(self):
        if not REAL_CSV.exists():
            pytest.skip("실제 배치 CSV 없음")
        return pd.read_csv(REAL_CSV, dtype={"종목코드": str}, low_memory=False)

    def test_real_batch_restores_all_target_columns(self):
        d = self._real()
        c0 = d["ROW_BUILD_MODE"] != "FRESH"
        targets = (list(cb.DISPLAY_ALIASES) + list(cb.SECTOR_AGG_COLS)
                   + ["업종_대분류", "업종_상세", "벤치_60d_KOSPI_%", "시총기준일"])
        before = {k: int(d.loc[c0, k].isna().sum()) for k in targets if k in d.columns}
        assert sum(before.values()) > 0, "이 CSV엔 복원할 결측이 없다 (표본 부적합)"
        out, rep = cb.backfill_carry_rows(
            d, bench_map={"KOSPI": {60: 10.0}, "KOSDAQ": {60: 4.0}},
            mcap_ymd="20260807")
        c1 = cb.carry_mask(out)
        after = {k: int(out.loc[c1, k].isna().sum()) for k in targets if k in out.columns}
        assert sum(after.values()) == 0, f"복원 후에도 결측 남음: {after}"

    # FRESH가 이 비율 이상 채우는 컬럼만 '무조건 산출'로 본다.
    #   조건부 사유 문자열(DANGER_ZONE_REASON·ENTRY_EDGE_REASON·ROUTE_PRE_BLOCK
    #   ·CAL_HOLD_REASON 등)은 **플래그가 켜질 때만** 채워진다. 실측 FRESH
    #   충전율이 2.5~14.7%로, 캐리 행이 비어 있는 것은 조건 미충족이지 누락이
    #   아니다. 그것까지 잡으면 게이트가 매일 거짓 경보를 낸다.
    UNCONDITIONAL_MIN_FILL = 0.90
    # 캐리 충전율이 FRESH의 이 비율 미만이면 위반. 실제 구멍(0.27배)은 잡고
    # 조건부 컬럼의 소폭 차이(0.88~0.98배)는 통과시키는 지점.
    CARRY_FILL_RATIO_FLOOR = 0.70

    @classmethod
    def _violations(cls, frame, carry) -> set:
        """FRESH가 항상 채우는데 CARRY 충전율이 크게 낮은 미선언 컬럼.

        '전부 비었을 때만' 잡으면 실제 구멍을 놓친다 — 2026-08-07 배치의
        LDY_SCORE는 캐리 41행 중 30행만 비어 있었고(legacy 11행은 옛 스냅샷에서
        값을 물려받았다), 그 부분 결측이 바로 화면을 망친 원인이었다.
        그래서 **충전율**로 비교한다.
        """
        fresh = ~carry
        bad = set()
        for col in frame.columns:
            ff = float(frame.loc[fresh, col].notna().mean())
            if ff < cls.UNCONDITIONAL_MIN_FILL:
                continue                       # 조건부 컬럼 — 계약 대상 아님
            cf = float(frame.loc[carry, col].notna().mean())
            # [v63] **상대 기준**으로 본다. 절대 90% 선으로 자르면 FRESH 충전율이
            #   90% 근처인 조건부 컬럼에서 오탐이 난다 — 2026-08-17 배치 실측:
            #   SCORE_REASON_TOP1(FRESH 91.9% / CARRY 81.2%) · 추천매도가3와
            #   TP3_METHOD(91.0% / 89.6%)가 걸렸는데 이들은 FRESH 자체가 ~91%인
            #   조건부 컬럼이고 캐리 특이 누락이 아니다.
            #   실제 구멍은 규모가 다르다(v60: FRESH 100% / CARRY 26.8% = 0.27배).
            if cf >= ff * cls.CARRY_FILL_RATIO_FLOOR:
                continue
            bad.add(col)
        return bad - set(cb.INTENTIONALLY_BLANK)

    def test_unconditional_columns_reach_carry_rows(self):
        """FRESH가 **항상** 채우는 컬럼은 CARRY에도 있어야 한다.

        이 게이트가 v60 이전에 있었다면 LDY_SCORE·SECTOR_* 구멍이 처음 생긴
        2026-05월에 바로 실패했다. Stage 3/4에 새 컬럼을 추가하고 캐리 배선을
        잊으면 여기서 잡힌다.
        """
        d = self._real()
        out, _ = cb.backfill_carry_rows(
            d, bench_map={"KOSPI": {60: 10.0}, "KOSDAQ": {60: 4.0}},
            mcap_ymd="20260807")
        c = cb.carry_mask(out)
        f = ~c
        if not bool(c.any()) or not bool(f.any()):
            pytest.skip("FRESH/CARRY 양쪽이 필요")
        undeclared = sorted(self._violations(out, c))
        assert not undeclared, (
            "FRESH가 항상 채우는데 CARRY엔 전부 없는 미선언 컬럼 "
            f"{len(undeclared)}개 — 캐리 배선 누락이거나 "
            f"INTENTIONALLY_BLANK에 사유와 함께 등록해야 한다: {undeclared}")

    def test_gate_would_have_caught_the_v60_hole(self):
        """게이트가 죽은 게이트가 아님 — 복원을 끄면 실제로 실패해야 한다."""
        d = self._real()
        c = d["ROW_BUILD_MODE"] != "FRESH"
        if not bool(c.any()):
            pytest.skip("캐리 행 필요")
        undeclared = sorted(self._violations(d, c))
        assert undeclared, ("복원 전 원본에서도 위반이 안 잡힌다 — "
                            "게이트가 아무것도 검사하지 않는다")
        for expected in ("LDY_SCORE", "TOTAL_SCORE", "RANK_SCORE"):
            assert expected in undeclared, \
                f"{expected} 구멍을 게이트가 놓친다"

    def test_borderline_conditional_columns_not_flagged(self):
        """[v63] FRESH 충전율이 90% 근처인 조건부 컬럼을 오탐하지 않는다.

        절대 기준(CARRY<90%)일 때 2026-08-17 배치에서 SCORE_REASON_TOP1·
        추천매도가3·TP3_METHOD 3개가 거짓 위반으로 잡혔다. 상대 기준으로
        바꿔 해소했고, 이 테스트가 되살아남을 막는다.
        """
        frame = pd.DataFrame({
            "ROW_BUILD_MODE": ["FRESH"] * 100 + ["CARRY_LEGACY"] * 100,
            # FRESH 91% / CARRY 89% — 조건부 컬럼의 정상적 편차
            "cond": ([1.0] * 91 + [np.nan] * 9) + ([1.0] * 89 + [np.nan] * 11),
            # FRESH 100% / CARRY 27% — 진짜 구멍
            "hole": ([1.0] * 100) + ([1.0] * 27 + [np.nan] * 73),
        })
        bad = self._violations(frame, cb.carry_mask(frame))
        assert "cond" not in bad, "조건부 컬럼을 오탐한다"
        assert "hole" in bad, "진짜 구멍을 놓친다"

    def test_conditional_reason_columns_are_not_flagged(self):
        """조건부 사유 컬럼이 거짓 경보를 내지 않는지 (오탐 회귀 가드).

        DANGER_ZONE_REASON은 DANGER_ZONE=1일 때만 채워진다. 캐리 행이 위험구간
        조건을 만족하지 않아 비어 있는 것은 **정상**이다. (실측 확인: 위험구간
        판정은 캐리 합류 뒤 Stage 6에서 돌므로 캐리 행도 평가 대상이며,
        컬럼이 존재하고 알파가 검증된 6일 배치에서 재계산 결과가 프로덕션
        기록과 동일하게 0건이었다 — 판정 누락이 아니라 조건 미충족이다.)
        """
        d = self._real()
        c = d["ROW_BUILD_MODE"] != "FRESH"
        for col in ("DANGER_ZONE_REASON", "ENTRY_EDGE_REASON",
                    "ROUTE_PRE_BLOCK", "CAL_HOLD_REASON"):
            if col not in d.columns:
                continue
            assert d.loc[~c, col].notna().mean() < self.UNCONDITIONAL_MIN_FILL, \
                f"{col}이 무조건 산출로 바뀌었다 — 계약 대상에 넣어야 한다"

    def test_danger_zone_is_evaluated_after_carry_join(self):
        """위험구간 판정이 캐리 합류 **뒤에** 돌아야 보유 종목도 평가된다.

        Stage 5(run_calibration)에서 캐리가 붙고 Stage 6(finalize)에서
        apply_alpha_entry_gate → flag_danger_zone이 돈다. 이 순서가 뒤집히면
        보유 종목은 평가되지 않은 채 기본값 0("안전")으로 남는다.
        """
        col = (ROOT / "collector.py").read_text(encoding="utf-8")
        assert col.find("run_calibration(ctx)") < col.find("finalize_outputs(ctx)")
        fin = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")
        assert "apply_alpha_entry_gate" in fin
        ae = (ROOT / "alpha_engine.py").read_text(encoding="utf-8")
        assert "flag_danger_zone(" in ae

    def test_intentionally_blank_entries_carry_a_reason(self):
        for col, why in cb.INTENTIONALLY_BLANK.items():
            assert isinstance(why, str) and len(why) >= 10, \
                f"{col}의 미채움 사유가 부실하다"


# ── E. 배선·게이트 생존 ────────────────────────────────────────
class TestWiring:
    def test_backfill_runs_after_carry_concat(self):
        """순서가 뒤바뀌면 이 패치는 아무 일도 하지 않는다.

        캐리 행이 df_out에 붙기 **전에** 백필이 돌면 대상이 0건이다.
        """
        src = (ROOT / "pipeline_calibrate.py").read_text(encoding="utf-8")
        i_concat = src.find("df_out = pd.concat([df_out, _cd]")
        i_backfill = src.find("backfill_carry_rows(")
        assert i_concat > 0, "캐리 concat을 찾지 못했다"
        assert i_backfill > 0, "backfill_carry_rows 호출이 없다 — 배선 누락"
        assert i_concat < i_backfill, \
            "백필이 캐리 concat보다 먼저 호출된다 — 대상이 0건이 된다"

    def test_backfill_runs_before_finalize_stage(self):
        """finalize(Stage 6)의 트리맵·켈리가 복원된 값을 보아야 한다."""
        src = (ROOT / "collector.py").read_text(encoding="utf-8")
        assert src.find("run_calibration(ctx)") < src.find("finalize_outputs(ctx)")

    def test_wired_call_passes_batch_context(self):
        """스칼라(벤치·시총기준일·수급)를 넘기지 않으면 그 컬럼은 복원되지 않는다."""
        src = (ROOT / "pipeline_calibrate.py").read_text(encoding="utf-8")
        call = src[src.find("backfill_carry_rows("):][:600]
        for kw in ("bench_map=", "mcap_ymd=", "individual_net_map="):
            assert kw in call, f"배선에서 {kw}를 넘기지 않는다"

    def test_is_alarming_is_not_a_dead_gate(self):
        clean = cb.backfill_carry_rows(_mini())[1]
        assert cb.is_alarming(clean) is False
        broken = dict(clean)
        broken["still_blank"] = {"NEW_COL": clean["carry_rows"]}
        assert cb.is_alarming(broken) is True, \
            "미복원 컬럼이 남아도 경고하지 않는다 — 죽은 게이트"

    def test_still_blank_reported_for_unrestorable(self):
        """DISPLAY_SCORE가 없으면 별칭을 복원할 수 없고, 그 사실이 남아야 한다."""
        df = _mini().drop(columns=["DISPLAY_SCORE"])
        _, rep = cb.backfill_carry_rows(df)
        assert set(cb.DISPLAY_ALIASES) <= set(rep["still_blank"])
        assert cb.is_alarming(rep) is True


# ── F. 붕괴 사슬 회귀 (실제 데이터) ─────────────────────────────
class TestDecayChainRegression:
    def test_greencross_row_recovers_score(self):
        if not REAL_CSV.exists():
            pytest.skip("실제 배치 CSV 없음")
        d = pd.read_csv(REAL_CSV, dtype={"종목코드": str}, low_memory=False)
        d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
        before = d[d["종목코드"] == "006280"]
        if before.empty:
            pytest.skip("녹십자 행 없음")
        assert before["LDY_SCORE"].isna().all(), "재현 전제(결측)가 깨졌다"
        out, _ = cb.backfill_carry_rows(d)
        r = out[out["종목코드"] == "006280"].iloc[0]
        assert pd.notna(r["LDY_SCORE"])
        assert float(r["LDY_SCORE"]) == float(r["DISPLAY_SCORE"])
        assert str(r["업종_대분류"]).strip() not in ("", "nan", "None")

    def test_carry_rows_no_longer_look_like_zero_score(self):
        """트리맵은 `color="LDY_SCORE"`로 칠한다. NaN→0 강제 때문에 보유 종목이
        점수 0으로 보였다. 복원 후에는 0이 아니어야 한다.
        """
        if not REAL_CSV.exists():
            pytest.skip("실제 배치 CSV 없음")
        d = pd.read_csv(REAL_CSV, dtype={"종목코드": str}, low_memory=False)
        out, _ = cb.backfill_carry_rows(d)
        c = cb.carry_mask(out)
        v = pd.to_numeric(out.loc[c, "LDY_SCORE"], errors="coerce")
        assert v.notna().all()
        assert (v > 0).any(), "복원했는데도 전부 0점이다"

    def test_treemap_color_source_is_ldy_score(self):
        """위 회귀의 전제(트리맵이 LDY_SCORE로 칠한다)를 소스로 고정."""
        src = (ROOT / "chart_components.py").read_text(encoding="utf-8")
        assert 'color="LDY_SCORE"' in src or "color='LDY_SCORE'" in src


# ── G. v59에서 내가 과장했던 주장의 실측 기록 ────────────────────
class TestSizingContaminationMagnitude:
    """v59 PR에서 '실제 주문 수량이 오염된다'고 썼다. 실측하니 절반만 맞았다.

    · 업종 결측 행 자신은 **켈리_수량 0** — 그 행들이 주문을 받은 적은 없다
    · 그러나 가짜 '?' 섹터가 **정상 픽의 섹터 순위를 밀어** 배수를 흔들었다:
      사이징이 실제로 돈 6일 전수에서 배수 차이 2.2~6.7%p(최대 7.5%p),
      정규화 후 최종 수량 영향 최대 5.24%
    · 8/05 기업은행은 12주 → v59 로직이면 11.72주 (반올림하면 동일)
    이 클래스는 그 계산을 코드로 고정해, 다음에 같은 주장을 할 때 근거가 되게 한다.
    """

    def test_missing_sector_rows_never_received_an_order(self):
        if not REAL_CSV.exists():
            pytest.skip("실제 배치 CSV 없음")
        d = pd.read_csv(REAL_CSV, dtype={"종목코드": str}, low_memory=False)
        m = d["업종_대분류"].isna()
        if not bool(m.any()):
            pytest.skip("결측 없음")
        qty = pd.to_numeric(d.loc[m, "켈리_수량"], errors="coerce").fillna(0)
        assert (qty == 0).all(), \
            "업종 결측 행이 주문 수량을 받았다 — v59 설명을 다시 써야 한다"

    def test_fake_bucket_shifts_ranks_of_known_sectors(self):
        """가짜 섹터를 하나 추가하면 정상 섹터의 순위 백분위가 밀린다."""
        r5 = pd.Series([10.0, 10.0, -5.0, -5.0, 2.0, 2.0])
        sec = pd.Series(["A", "A", "B", "B", None, None])
        old = sec.astype("object").fillna("?")
        rank_old = r5.groupby(old).transform("mean").rank(pct=True)
        kn = sec.notna()
        mn = pd.Series(np.nan, index=r5.index)
        mn.loc[kn] = r5[kn].groupby(sec[kn].astype(str)).transform("mean")
        rank_new = mn.rank(pct=True)
        moved = (rank_new[kn] - rank_old[kn]).abs()
        assert (moved > 0).any(), \
            "가짜 섹터가 정상 섹터 순위를 흔들지 않는다 — v59 근거가 무너진다"
