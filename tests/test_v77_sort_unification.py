# -*- coding: utf-8 -*-
"""v77 — 종목탭 기본 정렬을 오늘탭·공식픽 선정과 같은 축으로 통일.

사용자: "오늘탭 추천종목과 종목탭 정렬순서가 다른데 뭐가 1픽인거야"
8/28 배치 실측: 오늘탭 2위 현대차 vs 종목탭 2위 삼성전자. 원인은
종목탭 '알파순'이 ALPHA_SCORE 단독이었기 때문 — 엔진 선정 축은
알파×손익비다(v35: 품질점수 -1.29%/일 vs 알파×손익비 +2.07%/일).
"""
import os

import numpy as np
import pandas as pd
import pytest

from services.alpha_live_report import funnel_rank_key

SRC = open(os.path.join(os.path.dirname(__file__), "..",
                        "components", "tab_stocks.py"), encoding="utf-8").read()


class TestSourceWiring:
    def test_alpha_sort_branch_calls_helper(self):
        """'알파순' 분기가 공용 헬퍼 하나만 호출한다."""
        i_branch = SRC.index('sort_mode.value == "🧠 알파순"')
        i_next = SRC.index('sort_mode.value == "🔢 점수순"')
        assert "_sort_by_engine_axis(fdf)" in SRC[i_branch:i_next]


class TestEngineAxisHelper:
    """행동 검증 — 주석이 아니라 실제 정렬 결과를 잰다."""

    def _df(self):
        # 알파 단독 순서와 알파×RR 순서가 갈리게 설계:
        # A: 알파 90 · RR 0.5 → 키 45 | B: 알파 80 · RR 2.0 → 키 160
        return pd.DataFrame({
            "종목코드": ["000001", "000002", "000003"],
            "종목명": ["A", "B", "C"],
            "ALPHA_SCORE": [90.0, 80.0, 70.0],
            "RR_NOW_TP1": [0.5, 2.0, 1.0],
        })

    def test_sorts_by_alpha_times_rr_not_alpha_alone(self):
        from components.tab_stocks import _sort_by_engine_axis
        out = _sort_by_engine_axis(self._df())
        assert list(out["종목명"]) == ["B", "C", "A"], \
            "알파 단독이면 A>B>C — 엔진 축(알파×RR)이면 B(160)>C(70)>A(45)"

    def test_funnel_failures_go_last_ordered_by_alpha(self):
        from components.tab_stocks import _sort_by_engine_axis
        df = self._df()
        # 문턱 75: A(90)·B(80) 중 A는 문턱 95로 개별 미달 — 실제로는
        # threshold가 행별 값이므로 A(90<95) 미달, C(70<75) 미달, B만 통과.
        df["ALPHA_ENTRY_THRESHOLD"] = [95.0, 75.0, 75.0]
        out = _sort_by_engine_axis(df)
        # 통과자 B 먼저, 미통과 그룹은 알파 순: A(90) > C(70)
        assert list(out["종목명"]) == ["B", "A", "C"]

    def test_legacy_without_alpha_falls_back(self):
        from components.tab_stocks import _sort_by_engine_axis
        df = pd.DataFrame({"종목명": ["X", "Y"],
                           "ELITE_RANK_SCORE": [10.0, 20.0]})
        out = _sort_by_engine_axis(df)
        assert list(out["종목명"]) == ["Y", "X"]

    def test_empty_and_scoreless_df_survive(self):
        from components.tab_stocks import _sort_by_engine_axis
        assert len(_sort_by_engine_axis(pd.DataFrame({"종목명": []}))) == 0
        df = pd.DataFrame({"종목명": ["Z"]})
        assert list(_sort_by_engine_axis(df)["종목명"]) == ["Z"]


class TestSameAxisAsDecisionCenter:
    """합성 데이터에서 두 탭의 정렬 결과가 같은 순서를 내는지."""

    def _df(self):
        # 알파 단독 순서와 알파×RR 순서가 갈리게 설계:
        # A: 알파 90 · RR 0.5 → 키 45 | B: 알파 80 · RR 2.0 → 키 160
        return pd.DataFrame({
            "종목코드": ["000001", "000002", "000003"],
            "종목명": ["A", "B", "C"],
            "ALPHA_SCORE": [90.0, 80.0, 70.0],
            "RR_NOW_TP1": [0.5, 2.0, 1.0],
        })

    def test_engine_key_flips_alpha_only_order(self):
        df = self._df()
        k = pd.to_numeric(funnel_rank_key(df), errors="coerce")
        order = list(df.assign(_k=k).sort_values("_k", ascending=False)["종목명"])
        assert order == ["B", "C", "A"], \
            "알파×RR 축이면 B(160) > C(70) > A(45)여야 한다"

    def test_funnel_failures_sort_last(self):
        """퍼널 미통과(문턱 미달) 행은 키가 NaN — 뒤로 간다."""
        df = self._df()
        df["ALPHA_ENTRY_THRESHOLD"] = [75.0, 75.0, 75.0]   # C(70)는 미달
        k = pd.to_numeric(funnel_rank_key(df), errors="coerce")
        assert np.isnan(k.iloc[2])
        assert k.iloc[0] > 0 and k.iloc[1] > 0


@pytest.mark.skipif(not os.path.exists("data/recommend_20260828.csv"),
                    reason="실데이터 없음")
class TestRealBatch:
    def test_top5_matches_decision_center_axis(self, real_data_mirror):
        """8/28 실배치에서 두 축의 상위 5가 일치한다 (고정 데이터 — 결정적)."""
        d = real_data_mirror("recommend_20260828.csv")
        df = pd.read_csv(os.path.join(d, "recommend_20260828.csv"),
                         dtype={"종목코드": str})
        tp = df[pd.to_numeric(df.get("TOP_PICK", 0),
                              errors="coerce").fillna(0) == 1].copy()
        ek = pd.to_numeric(funnel_rank_key(tp), errors="coerce")
        al = pd.to_numeric(tp["ALPHA_SCORE"], errors="coerce")
        new_order = list(tp.assign(_e=ek, _a=al)
                         .sort_values(["_e", "_a"], ascending=False,
                                      na_position="last")["종목명"].head(5))
        rr = pd.to_numeric(tp["RR_NOW_TP1"], errors="coerce").fillna(0).clip(0, 3)
        dc_order = list(tp.assign(_k=al.clip(lower=0) * rr)
                        .sort_values("_k", ascending=False)["종목명"].head(5))
        assert new_order == dc_order
