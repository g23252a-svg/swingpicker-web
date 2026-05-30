# -*- coding: utf-8 -*-
"""[v22.3.21] No-Buy 카드 FOMO-safety 테스트.

pytest tests/test_no_buy_card.py -v

대상: components.buy_now_badge.build_no_buy_card_model (순수 로직, NiceGUI 불필요)
원칙: official_buy=False면 목표가/매수가를 매수 CTA처럼 노출하지 않는다(백엔드 안전신호 보호).
"""
import sys, os, re, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.buy_now_badge import build_no_buy_card_model, NO_BUY_ELITE_PASS


def _no_buy_row(**over):
    """NC형 매수보류 행 (TOP_PICK=0, MACRO CRITICAL, 큰 목표 +21.8%)."""
    r = dict(종목명="NC", STRUCT_SCORE=98, TIMING_SCORE=98, AI_SCORE=39, BALANCE_SCORE=26,
             ELITE_SCORE=70.8, MACRO_RISK="CRITICAL", ROUTE="ARMED", ENTRY_RISK_LEVEL="",
             추천매수가=289500, 추천매도가1=352500, TP1_PCT=21.8, RR_NOW_TP1=1.9,
             TOP_PICK=0, BUY_NOW_GRADE="WATCH", BUY_NOW_ELIGIBLE=0)
    r.update(over)
    return r


def _official_buy_row(**over):
    r = _no_buy_row(MACRO_RISK="LOW", ELITE_SCORE=82.0, AI_SCORE=70, BALANCE_SCORE=60,
                    TOP_PICK=1, BUY_NOW_GRADE="BUY", BUY_NOW_ELIGIBLE=1)
    r.update(over)
    return r


def test_no_buy_card_does_not_promote_target_price():
    m = build_no_buy_card_model(_no_buy_row())
    assert m["official_buy"] is False
    assert m["promote_target"] is False


def test_no_buy_card_uses_rest_tone_not_buy_cta():
    m = build_no_buy_card_model(_no_buy_row())
    assert m["headline_tone"] == "rest"
    assert m["cta"] is None
    assert "쉬어갑니다" in m["headline"]
    assert not any(p in m["headline"] for p in ["매수하세요", "지금 매수", "매수 검토", "매수 추천", "사세요"])
    assert "금지" not in m["headline"]          # '금지' 톤도 지양


def test_no_buy_card_hides_large_entry_price():
    m = build_no_buy_card_model(_no_buy_row())
    assert m["price_treatment"] == "footnote"   # hero 아님
    # 매수가/목표가 숫자가 hero 영역(헤드라인/서브텍스트)에 노출되면 실패
    hero = m["headline"] + " " + m["subtext"]
    assert "289,500" not in hero and "289500" not in hero
    assert "352,500" not in hero and "352500" not in hero
    # 가격은 오직 회색 각주에만
    assert "289,500" in m["target_footnote"]
    assert "오늘은 매수 대상이 아닙니다" in m["target_footnote"]


def test_admin_expander_contains_raw_engine_values():
    m = build_no_buy_card_model(_no_buy_row(), is_admin=True)
    raw = m["admin_raw"]
    for k in ["ELITE_SCORE", "ROUTE", "MACRO_RISK", "ENTRY_RISK_LEVEL",
              "STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE", "BALANCE_SCORE",
              "TOP_PICK", "BUY_NOW_ELIGIBLE", "BUY_NOW_GRADE"]:
        assert k in raw, f"admin_raw에 {k} 없음"
    assert raw["MACRO_RISK"] == "CRITICAL"
    # 비관리자 모드는 raw 비어있음
    assert build_no_buy_card_model(_no_buy_row(), is_admin=False)["admin_raw"] == {}


def test_no_buy_card_no_strong_profit_copy_in_main():
    """핵심: official_buy=False면 '+21.8% 목표' 같은 강한 수익 유도 문구가
       메인 카드(헤드라인/서브텍스트)에 나오면 실패. 각주에만 허용."""
    m = build_no_buy_card_model(_no_buy_row())
    main = m["headline"] + " " + m["subtext"] + " " + m["closest_note"]
    # 수익률 패턴(+NN.N% / +NN%) 이 메인에 등장하면 실패
    assert not re.search(r"\+\s*\d+(\.\d+)?\s*%", main), f"메인에 수익률 노출: {main!r}"
    # 목표 % 는 각주에만
    assert "+21.8%" in m["target_footnote"]
    assert m["promote_target"] is False


def test_official_buy_promotes_target_positive_control():
    """양성 대조: official_buy=True면 목표 hero + 매수 CTA 허용 (규칙이 조건부임을 보장)."""
    m = build_no_buy_card_model(_official_buy_row())
    assert m["official_buy"] is True
    assert m["promote_target"] is True
    assert m["price_treatment"] == "hero"
    assert m["cta"] is not None
    assert m["headline_tone"] == "buy"


def test_axes_pass_fail_computed():
    m = build_no_buy_card_model(_no_buy_row())
    by = {a["key"]: a for a in m["axes"]}
    assert by["struct"]["pass"] is True   # 98 >= 80
    assert by["timing"]["pass"] is True   # 98 >= 70
    assert by["ai"]["pass"] is False      # 39 < 60
    assert by["balance"]["pass"] is False # 26 < 50


def test_blockers_ranked_market_first():
    m = build_no_buy_card_model(_no_buy_row())
    assert m["blockers"][0]["kind"] == "market"
    assert m["blockers"][0]["rank"] == 1
    kinds = [b["kind"] for b in m["blockers"]]
    assert "score" in kinds  # 70.8 < 75 → 점수 부족도 사유


def test_real_nc_row_if_present():
    """실 recommend CSV에 매수보류(TOP_PICK=0 또는 ELIGIBLE=0) 행이 있으면 no-buy 처리 확인."""
    files = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "recommend_2026*.csv")))
    if not files:
        import pytest
        pytest.skip("recommend CSV 없음")
    import pandas as pd
    df = pd.read_csv(files[-1], dtype={"종목코드": str})
    # official_buy 아닐 행 하나 — 대부분 그러함
    row = df.iloc[0].to_dict()
    m = build_no_buy_card_model(row)
    if not m["official_buy"]:
        assert m["promote_target"] is False
        assert m["price_treatment"] in ("footnote", "hidden")
        assert m["headline_tone"] == "rest"


def test_rendered_html_keeps_price_in_footnote_only():
    """렌더 HTML에서도 official_buy=False면 가격이 hero가 아닌 각주(ⓘ)에만 나오는지.
       NiceGUI 없으면 skip(운영 환경 전용)."""
    try:
        from components.stock_detail_v2 import _nb_card_html
    except Exception:
        import pytest
        pytest.skip("NiceGUI 미설치 — 운영 환경 전용")
    html = _nb_card_html(build_no_buy_card_model(_no_buy_row()))
    foot_idx = html.find("ⓘ")
    assert foot_idx > 0, "각주(ⓘ)가 없음"
    head = html[:foot_idx]                       # 각주 이전(=hero/배너/막대/사유)
    assert "352,500" not in head and "+21.8%" not in head, "목표가/수익률이 각주 이전에 노출됨"
    assert "352,500" in html[foot_idx:]          # 가격은 각주에만
