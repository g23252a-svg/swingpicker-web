# -*- coding: utf-8 -*-
"""[v70] 화면이 퍼널 안에서 **성적이 나쁜 쪽**을 골라내고 있었다.

■ 사용자 질문
  "오늘탭에서 추천한 1위 종목과 3개 근접추천을 샀다면?"
  화면이 실제로 보여준 목록을 그대로 재현해 측정한 결과(11일 · 32건):
    공식 + 근접3  이긴 날 **18%** · 일평균 **-5.34%** · 복리 -45.8%
  같은 기간 퍼널 재구성 상위N은 -2.61%였다. **두 목록이 다른 집합**이었고
  겹침은 32종목 중 17종목(53%)뿐이었다.

■ 원인 — 목록 조건이 역선택이었다
  종전 조건: `ACTION_DECISION == "WATCH"` AND `켈리_수량 > 0`
  알파 시대 24일 · 퍼널 통과 상위20(395건) 실측:

    퍼널 통과 전체      395건  승률 50.4% · 평균 +0.47%
      └ 켈리 수량>0      45건  승률 **26.7%** · 평균 **-3.46%**  ← 화면이 보여준 것
      └ 켈리 수량=0     350건  승률 **53.4%** · 평균 **+0.97%**  ← 화면이 버린 것

  같은 날 페어드(11일): 퍼널 상위2 − 종전 상위2 = **+3.02%p**
    t=+1.73 · p=0.115 · IS +3.81 / OOS +2.37(부호 일치) ·
    상위2 제거 +1.20(생존) · 중위 +0.00

■ 기전
  켈리 f = p − (1−p)/b 이므로 **f>0은 선언 승률이 높은 행을 고른다.**
  그런데 그 선언 승률이 과대하다(v68: 선언 33% vs 같은 점수 구간 실측 17%).
  즉 `수량>0` 필터는 **모델이 가장 과신하는 종목**을 고르고 있었다.

■ 정직하게
  p=0.115로 **통계적으로 검증되지 않았다**(11일). 그래도 바꾼 이유는
  ⑴ IS/OOS 부호가 같고 이상치 제거에도 남으며 ⑵ 기전이 설명되고
  ⑶ 이 목록은 '매수 아님'으로 표시되는 **표시 전용**이라 되돌리기 쉽다.
  화면에 p값과 표본을 함께 적어 검증됐다고 읽히지 않게 한다.

■ v64의 취지는 배지로 보존한다
  v64가 `수량>0`을 넣은 이유는 '살 수 없는 것을 살 수 있는 것처럼 보이게
  하지 않기'였다. 목록에서 **빼는 대신** `0주 · 사이징 불가` 배지로 말한다.

■ 이 파일이 고정하는 것
  1. 관찰 후보 모집단은 **퍼널 통과군**이다(수량 조건으로 거르지 않는다).
  2. 살 수 없는 종목은 목록에 있되 **배지로 표시**된다.
  3. 화면과 리포트가 **같은 함수**로 순위를 매긴다(괴리 재발 방지).
  4. 바꾼 근거와 **p값**이 화면에 남는다.
  5. 퍼널 키를 못 구하면 종전 조건으로 안전하게 되돌아간다.
  6. 공식 매수·결정 컬럼·수량은 바뀌지 않는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.alpha_live_report as ALR  # noqa: E402

DATA = ROOT / "data"
DC_SRC = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
#  1. 화면과 리포트가 같은 함수를 쓴다
# ══════════════════════════════════════════════════════════════════
class TestSharedRanking:
    def test_public_wrapper_exists(self):
        assert hasattr(ALR, "funnel_rank_key")

    def test_wrapper_matches_internal(self):
        d = pd.DataFrame({"ALPHA_SCORE": [95.0, 80.0, 99.0],
                          "RR_NOW_TP1": [2.0, 3.0, 1.5],
                          "ALPHA_ENTRY_THRESHOLD": [85.0] * 3})
        a = ALR.funnel_rank_key(d)
        b = ALR._gate_rank_key(d)
        assert (a.fillna(-1) == b.fillna(-1)).all()

    def test_screen_uses_the_wrapper(self):
        assert "funnel_rank_key" in DC_SRC, \
            "화면이 리포트와 다른 규칙으로 목록을 뽑으면 v63의 괴리가 되살아난다"

    def test_screen_no_longer_filters_by_quantity(self):
        i = DC_SRC.find("if _funnel_key is not None")
        j = DC_SRC.find("buys = [_stock_payload", i)
        block = DC_SRC[i:j]
        assert "_fk" in block
        assert 'action.eq("WATCH") & (_qty > 0)' in block, \
            "폴백 경로가 사라졌다 — 퍼널 키 실패 시 안전망이 필요하다"


# ══════════════════════════════════════════════════════════════════
#  2. 살 수 없는 종목은 빼지 않고 표시한다 (v64 취지 보존)
# ══════════════════════════════════════════════════════════════════
class TestSizableBadge:
    def test_payload_carries_sizable(self):
        assert '"sizable": bool((_PR.quantity(row) or 0.0) > 0)' in DC_SRC

    def test_card_renders_the_badge(self):
        assert '"0주 · 사이징 불가"' in DC_SRC
        i = DC_SRC.find('"0주 · 사이징 불가"')
        assert DC_SRC.rfind('stock.get("sizable"', 0, i) > 0

    def test_zero_qty_is_not_dropped_from_the_pool(self):
        """v64는 목록에서 뺐다. v70은 배지로 말한다 — 조건이 되살아나면 실패."""
        i = DC_SRC.find("if _funnel_key is not None")
        j = DC_SRC.find("else:", i)
        primary = DC_SRC[i:j]
        assert "_qty > 0" not in primary, "퍼널 경로에 수량 필터가 다시 들어갔다"


# ══════════════════════════════════════════════════════════════════
#  3. 근거와 p값이 화면에 남는다
# ══════════════════════════════════════════════════════════════════
class TestEvidenceShown:
    def test_pool_line_exists(self):
        assert '"watch_pool_line"' in DC_SRC
        assert 'summary["watch_pool_line"]' in DC_SRC

    def test_line_states_the_pvalue_and_sample(self):
        i = DC_SRC.find('"watch_pool_line"')
        block = DC_SRC[i:i + 900]
        assert "p=0.115" in block, "p값 없이 바꾼 근거를 적으면 검증된 것으로 읽힌다"
        assert "검증된 것은 아닙니다" in block
        assert "26.7%" in block and "50.4%" in block

    def test_source_records_the_mechanism(self):
        i = DC_SRC.find("[v70] 관찰 후보 모집단을")
        assert i > 0
        block = DC_SRC[i:i + 1400]
        for token in ("50.4%", "26.7%", "53.4%", "+3.02%p", "p=0.115"):
            assert token in block, f"근거 {token}이 소스에 기록되지 않았다"


# ══════════════════════════════════════════════════════════════════
#  4. 실배치 동작
# ══════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not (DATA / "recommend_20260825.csv").exists(),
                    reason="실데이터 없음")
class TestRealBatch:
    @pytest.fixture(scope="class")
    def summary(self):
        from components.decision_center import build_decision_summary
        d = pd.read_csv(DATA / "recommend_20260825.csv", encoding="utf-8-sig",
                        dtype={"종목코드": str}, low_memory=False)
        return build_decision_summary(d), d

    def test_watch_comes_from_the_funnel(self, summary):
        s, d = summary
        assert s["watch"], "관찰 후보가 비었다"
        key = ALR.funnel_rank_key(d)
        top = set(d.assign(_k=key).dropna(subset=["_k"])
                  .sort_values("_k", ascending=False)
                  .head(10)["종목명"].astype(str))
        for w in s["watch"]:
            assert str(w["name"]) in top, f"{w['name']}이 퍼널 상위에 없다"

    def test_official_pick_unchanged(self, summary):
        s, d = summary
        from services.recommendation_quality import production_buy_mask
        assert len(s["buys"]) == int(production_buy_mask(d).fillna(False).sum())

    def test_watch_excludes_official(self, summary):
        s, _ = summary
        names = {b["name"] for b in s["buys"]}
        assert not (names & {w["name"] for w in s["watch"]})

    def test_depth_still_applies(self, summary):
        s, _ = summary
        assert len(s["watch"]) <= s["watch_depth"]

    def test_pool_line_rendered_with_data(self, summary):
        s, _ = summary
        assert "p=0.115" in (s.get("watch_pool_line") or "")


# ══════════════════════════════════════════════════════════════════
#  5. 폴백 — 퍼널 키를 못 구하면 안전하게 종전 규칙
# ══════════════════════════════════════════════════════════════════
class TestFallback:
    def test_no_alpha_column_falls_back(self):
        from components.decision_center import build_decision_summary
        d = pd.DataFrame({
            "종목코드": ["000001", "000002"], "종목명": ["A", "B"],
            "PRODUCTION_BUY": [0, 0], "ACTION_DECISION": ["WATCH", "WATCH"],
            "켈리_수량": [10, 0], "DISPLAY_SCORE": [70.0, 60.0],
            "종가": [1000, 2000], "추천매수가": [1000, 2000],
            "손절가": [930, 1860], "ROUTE": ["WAIT", "WAIT"],
        })
        s = build_decision_summary(d)
        assert isinstance(s.get("watch"), list)
        assert s.get("watch_pool_line") == "", \
            "퍼널 키가 없는데 바꿨다고 적으면 거짓말이다"

    def test_empty_frame_is_safe(self):
        from components.decision_center import build_decision_summary
        s = build_decision_summary(pd.DataFrame())
        assert s["watch"] == []
