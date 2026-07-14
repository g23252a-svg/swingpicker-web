# -*- coding: utf-8 -*-
"""
test_action_cards_v245.py
═══════════════════════════════════════════════════
v24.5 액션 카드 렌더러 회귀 테스트.

보증
  1. build_cards_html이 streamlit 없이 동작(오프라인 렌더)·예외 안 던짐.
  2. verdict 분기: 매수구간/눌림대기/추격주의 가격 경계 정확.
  3. 가격 사다리 좌표 2~98% 클램프, 결측 강건.
  4. PMS 레인 분리(PMS_PROFIT_PICK==1만), 컬럼 없으면 공식 레인만.
  5. 필터(공식매수/관찰후보/전체/PMS레인)와 구버전 가격필터 동작.
  6. PRODUCTION_BUY가 아닌 카드는 공식 추천으로 표시하지 않음.
  7. 스코프 CSS·sp- 프리픽스(전역 충돌 방지).
"""
import pandas as pd
import pytest

from components.action_cards import (
    build_cards_html, _verdict, _ladder_html, _gauge_html, _col,
    PULLBACK_WAIT_MAX_PCT, OVERHEAT_RSI,
)


def _row(**kw):
    base = dict(종목명="테스트", 종목코드="123456", 종가=10000, 추천매수가=10000,
                손절가=9000, 추천매도가1=12000, 추천매도가2=14000, RSI14=60, DISPLAY_SCORE=70,
                ELITE_SCORE=50, ROUTE="WAIT", **{"거래대금(억원)": 200})
    base.update(kw)
    return pd.Series(base)


# ── verdict 경계 ─────────────────────────────────────────────────
def test_verdict_buy_zone_at_entry():
    v = _verdict(_row(종가=10000, 추천매수가=10000))
    assert v["key"] == "buy" and v["label"] == "매수 구간"


def test_verdict_buy_zone_below_entry():
    v = _verdict(_row(종가=9500, 추천매수가=10000))  # 진입가 아래
    assert v["key"] == "buy"


def test_verdict_pullback_wait():
    # 진입가 위 +8% (경계 12% 이내), 비과열 → 눌림 대기
    v = _verdict(_row(종가=10800, 추천매수가=10000, RSI14=60, ROUTE="WAIT"))
    assert v["key"] == "wait" and v["label"] == "눌림 대기"


def test_verdict_chase_warning_far():
    # 진입가 위 +20% → 추격 주의
    v = _verdict(_row(종가=12000, 추천매수가=10000))
    assert v["key"] == "avoid" and v["label"] == "추격 주의"


def test_verdict_chase_warning_overheat():
    # 갭 작아도 OVERHEAT + 고RSI → 추격 주의
    v = _verdict(_row(종가=10500, 추천매수가=10000, RSI14=80, ROUTE="OVERHEAT"))
    assert v["key"] == "avoid"


def test_verdict_boundary_exact():
    # 경계 정확히 +12% → 눌림(<=), +12.1% → 추격
    assert _verdict(_row(종가=11200, 추천매수가=10000, RSI14=60))["key"] == "wait"
    assert _verdict(_row(종가=11210, 추천매수가=10000, RSI14=60))["key"] == "avoid"


def test_verdict_no_entry():
    v = _verdict(_row(추천매수가=0))
    assert v["key"] == "wait" and "관찰" in v["label"]


# ── 가격 사다리 ──────────────────────────────────────────────────
def test_ladder_has_four_marks():
    html = _ladder_html(_row(손절가=9000, 추천매수가=10000, 종가=10500, 추천매도가1=12000))
    for lab in ["손절", "진입", "현재", "1차 익절"]:
        assert lab in html
    assert "+20.0%" in html
    assert "sp-ladder" in html


def test_ladder_positions_clamped():
    # 현재가가 목표보다 위여도 좌표는 3~97% 안
    html = _ladder_html(_row(손절가=9000, 추천매수가=10000, 종가=15000, 추천매도가1=12000))
    import re
    pcts = [float(x) for x in re.findall(r"left:([\d.]+)%", html)]
    assert all(3.0 <= p <= 97.0 for p in pcts)


def test_ladder_empty_when_insufficient():
    assert _ladder_html(_row(손절가=0, 추천매수가=0, 종가=0, 추천매도가1=0)) == ""


# ── 게이지 ───────────────────────────────────────────────────────
def test_gauge_dashoffset_range():
    assert "stroke-dashoffset" in _gauge_html(0, "#fff")
    assert "stroke-dashoffset" in _gauge_html(100, "#fff")
    assert "85" in _gauge_html(85, "#fff")  # 점수 텍스트


# ── 피드 빌드 / 레인 ─────────────────────────────────────────────
def _frame(n=6, with_pms=False):
    rows = []
    for i in range(n):
        r = dict(종목명=f"종목{i}", 종목코드=f"{i:06d}", 종가=10000 + i * 100,
                 추천매수가=10000, 손절가=9000, 추천매도가1=12000, 추천매도가2=14000,
                 RSI14=50 + i, DISPLAY_SCORE=90 - i, ELITE_SCORE=40 + i,
                 ROUTE="WAIT", PRODUCTION_BUY=1, ACTION_DECISION="BUY",
                 **{"거래대금(억원)": 100 + i * 50})
        if with_pms:
            r["PMS_PROFIT_PICK"] = 1 if i < 2 else 0
            r["PMS_SCORE"] = 95 - i * 3
        rows.append(r)
    return pd.DataFrame(rows)


def test_build_feed_offline_no_streamlit():
    html = build_cards_html(_frame(5))
    assert "sp-feed" in html and html.count("sp-card") == 5
    assert "공식 매수" in html


def test_pms_lane_separation():
    html = build_cards_html(_frame(6, with_pms=True), show_pms=True, filter_mode="전체")
    assert "PMS 추천 레인" in html
    assert "편향 재검증 중" in html  # [v24.8] PMS 경고 라벨로 변경
    # PMS pick 2개는 PMS 카드로
    assert html.count("sp-pms-card") == 2


def test_no_pms_column_official_only():
    html = build_cards_html(_frame(4, with_pms=False))
    assert "PMS 추천 레인" not in html
    assert html.count("sp-card") == 4


def test_filter_buy_zone_only():
    df = _frame(6)
    df.loc[0, "종가"] = 10000   # 매수구간
    df.loc[1, "종가"] = 13000   # 추격
    html = build_cards_html(df, filter_mode="매수 구간")
    # 매수 구간만 → 추격/눌림 제외
    assert "추격 주의" not in html


def test_filter_pms_lane():
    html = build_cards_html(_frame(6, with_pms=True), show_pms=True, filter_mode="PMS 레인")
    assert "공식 매수" not in html
    assert "PMS 추천 레인" in html


def test_non_production_rows_are_never_called_official():
    df = _frame(3)
    df["PRODUCTION_BUY"] = 0
    df["ACTION_DECISION"] = "CASH"
    html = build_cards_html(df)
    assert "오늘 공식 매수 0개" in html
    assert "품질게이트 통과" not in html
    assert html.count("sp-card") == 0


def test_watch_candidates_require_explicit_view_and_warning():
    df = _frame(3)
    df["PRODUCTION_BUY"] = 0
    df["ACTION_DECISION"] = "WATCH"
    df["QUALITY_GUARD_REASON"] = "즉시매수 기준 미달"
    html = build_cards_html(df, filter_mode="관찰 후보")
    assert "관찰 후보" in html
    assert "매수 추천 아님" in html
    assert "품질게이트 통과" not in html
    assert html.count("sp-card") == 3


def test_legacy_official_fallback_is_conservative():
    df = _frame(1).drop(columns=["PRODUCTION_BUY"])
    df["TOP_PICK"] = 1
    df["BUY_NOW_ELIGIBLE"] = 1
    html = build_cards_html(df)
    assert "품질게이트 통과" in html


def test_pms_is_hidden_from_default_action_view():
    html = build_cards_html(_frame(6, with_pms=True))
    assert "PMS 추천 레인" not in html


def test_empty_frame_safe():
    html = build_cards_html(pd.DataFrame())
    assert "sp-feed" in html  # 예외 없이 빈 피드


# ── 안전/스코프 ──────────────────────────────────────────────────
def test_missing_columns_graceful():
    # 최소 컬럼만
    df = pd.DataFrame([{"종목명": "X", "종목코드": "000001"}])
    html = build_cards_html(df)
    assert "오늘 공식 매수 0개" in html  # 예외 없이 보수적 렌더


def test_scoped_css_prefix():
    from components.action_cards import _CSS
    # 모든 클래스가 sp- 프리픽스 (전역 충돌 방지)
    assert ".sp-" in _CSS
    # 전역 태그 셀렉터 오염 없음 (body/div 단독 스타일 금지)
    for bad in ["\nbody{", "\ndiv{", "\n*{"]:
        assert bad not in _CSS


def test_col_safe_parsing():
    r = pd.Series({"a": "1,234", "b": "12.5%", "c": "nan", "d": None})
    assert _col(r, "a") == 1234.0
    assert _col(r, "b") == 12.5
    assert _col(r, "c", default=99) == 99
    assert _col(r, "x", default=7) == 7  # 없는 컬럼


# ── v24.6: NiceGUI 경로 ─────────────────────────────────────────
def test_include_css_toggle():
    df = _frame(3)
    no_css = build_cards_html(df, include_css=False)
    with_css = build_cards_html(df, include_css=True)
    assert "<style>" not in no_css
    assert with_css.startswith("<style>")
    assert "sp-card" in with_css  # 카드 본문도 포함


def test_nicegui_renderer_graceful_without_nicegui():
    # nicegui 미설치 환경에서 예외 없이 None 반환(기존 화면 보존)
    from components.action_cards import render_action_cards_nicegui
    assert render_action_cards_nicegui(_frame(3)) is None
    assert render_action_cards_nicegui(pd.DataFrame()) is None  # 빈 df도 안전
