"""The home shortlist must distinguish calibrated probability from a score."""
import pandas as pd
import pytest

from components.swing_board import build_swing_board, _signal_text, render_swing_board


def row(code="005930", **kwargs):
    return {"종목코드": code, "종목명": f"종목 {code}", "종가": 70000, "기준일": "20260904",
            "PRODUCTION_BUY": 0, "SWING_STATUS": "VALIDATED",
            "SWING_TRADE_VALIDATED": 1,
            "SWING_ASOF": "20260904", "SWING_RANK": 1,
            "SWING_PROB": 0.63, "SWING_SCORE": 90.0, **kwargs}


@pytest.mark.parametrize("status,prob", [
    ("RESEARCH", 0.97), ("VALIDATED", float("nan")),
    ("VALIDATED", float("inf")), ("VALIDATED", -0.1), ("VALIDATED", 1.3),
])
def test_unvalidated_or_invalid_probability_is_never_rendered_as_percent(status, prob):
    board = build_swing_board(pd.DataFrame([row(SWING_STATUS=status, SWING_PROB=prob)]))
    assert board["mode"] == "FALLBACK"
    assert board["rows"] == []
    stock = board["research_rows"][0]
    assert stock["probability"] is None
    assert _signal_text(stock) == ("90.0점", "검증 중 · 상대순위")


@pytest.mark.parametrize("trade_validated", [0, False, "false", None, float("nan")])
def test_forecast_accuracy_without_net_return_validation_remains_research(trade_validated):
    df = pd.DataFrame([row(SWING_STATUS="VALIDATED", SWING_TRADE_VALIDATED=trade_validated,
                           SWING_OOS_NET_RETURN=-.025)])
    board = build_swing_board(df)
    assert board["mode"] == "FALLBACK"
    assert board["rows"] == []
    stock = board["research_rows"][0]
    assert stock["status"] == "RESEARCH"
    assert stock["probability"] is None
    assert _signal_text(stock) == ("90.0점", "검증 중 · 상대순위")


def test_missing_trade_validation_fails_closed():
    df = pd.DataFrame([row()]).drop(columns=["SWING_TRADE_VALIDATED"])
    board = build_swing_board(df)
    assert board["rows"] == []
    assert board["research_rows"][0]["probability"] is None


def test_model_rank_does_not_promote_research_to_official_or_change_source():
    df = pd.DataFrame([
        row("000002", SWING_RANK=2, SWING_PROB=.62, PRODUCTION_BUY=1),
        row("000001", SWING_RANK=1, SWING_PROB=.81, SWING_STATUS="RESEARCH"),
    ], index=[8, 8])
    before = df.copy(deep=True)
    board = build_swing_board(df)
    assert [r["code"] for r in board["rows"]] == ["000002"]
    assert board["rows"][0]["official"] is True
    assert [r["code"] for r in board["research_rows"]] == ["000001"]
    assert board["research_rows"][0]["official"] is False
    assert _signal_text(board["rows"][0]) == ("62.0%", "비용 후 스윙 수익 확률")
    pd.testing.assert_frame_equal(df, before)


def test_new_unavailable_batch_cannot_resurrect_old_validated_probability():
    df = pd.DataFrame([
        row("000001", SWING_ASOF="20260903"),
        row("000002", SWING_ASOF="20260904", SWING_STATUS="UNAVAILABLE"),
    ])
    summary = {"watch": [{"code": "000002", "name": "최신 관찰", "alpha_win_prob": .99}]}
    board = build_swing_board(df, summary)
    assert board["mode"] == "FALLBACK"
    assert [r["code"] for r in board["rows"]] == ["000002"]
    assert _signal_text(board["rows"][0]) == ("기존 선별", "스윙 확률 미제공")


@pytest.mark.parametrize("asof", ["20260903", "20260905", "", "invalid"])
def test_forecast_date_must_match_current_batch(asof):
    board = build_swing_board(pd.DataFrame([row(SWING_ASOF=asof)]))
    assert board["mode"] == "FALLBACK"
    assert board["rows"] == []


def test_alphanumeric_krx_code_survives_model_and_legacy_fallback():
    df = pd.DataFrame([row("0126Z0", PRODUCTION_BUY=1)])
    board = build_swing_board(df)
    assert board["rows"][0]["code"] == "0126Z0"
    assert board["rows"][0]["official"] is True
    board = build_swing_board(df.drop(columns=["SWING_STATUS"]),
                              {"buys": [{"code": "0126Z0", "name": "삼성에피스홀딩스"}]})
    assert board["rows"][0]["code"] == "0126Z0"
    assert board["rows"][0]["official"] is True


def test_model_rows_are_unique_bounded_and_stable_on_tied_ranks():
    rows = [row(f"{i:06d}", SWING_RANK=1) for i in range(12, 0, -1)]
    rows += [row("000001", SWING_RANK=1), row("bad", SWING_RANK=1)]
    board = build_swing_board(pd.DataFrame(rows), limit=100)
    assert board["total"] == 12
    assert [r["code"] for r in board["rows"]] == [f"{i:06d}" for i in range(1, 11)]
    assert [r["rank"] for r in board["rows"]] == list(range(1, 11))


def test_legacy_official_contract_and_fallback_are_preserved():
    df = pd.DataFrame([{"종목코드": "000001", "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1}])
    summary = {"buys": [{"code": "000001", "name": "공식"}],
               "watch": [{"code": "000001"}, {"code": "000002", "name": "관찰"}]}
    board = build_swing_board(df, summary)
    assert [r["official"] for r in board["rows"]] == [True, False]
    assert all(r["probability"] is None for r in board["rows"])
    assert build_swing_board(pd.DataFrame())["rows"] == []


def test_real_nicegui_board_starts_with_five_and_toggle_shows_ten():
    from nicegui import ui
    df = pd.DataFrame([row(f"{i:06d}", SWING_RANK=i) for i in range(1, 11)])
    with ui.column() as holder:
        render_swing_board(df, {"production_count": 0})
    try:
        assert len([el for el in holder.descendants() if el.tag == "nicegui-link"]) == 5
        labels = [str(getattr(el, "text", "")) for el in holder.descendants()]
        assert "63.0%" in labels
        assert "90.0점" not in labels
        assert "관찰 · 매수 아님" in labels
        toggle = next(el for el in holder.descendants() if el.tag == "q-btn-toggle")
        toggle.set_value(10)
        assert len([el for el in holder.descendants() if el.tag == "nicegui-link"]) == 10
        assert all(el.value is False for el in holder.descendants() if el.tag == "nicegui-expansion")
    finally:
        holder.delete()


def test_failed_model_never_replaces_existing_shortlist_and_is_collapsed():
    from nicegui import ui
    df = pd.DataFrame([
        row("000001", 종목명="실패 모델 1위", SWING_STATUS="RESEARCH", SWING_TRADE_VALIDATED=0,
            SWING_OOS_NET_RETURN=-.0268, SWING_OOS_UNIVERSE_NET_RETURN=-.0105,
            SWING_OOS_COVERAGE=.91, SWING_VALIDATION_DAYS=40,
            SWING_VALIDATION_REASON="평균 비용 후 수익 우위 미확인"),
        row("000002", 종목명="기존 선별 후보", SWING_STATUS="UNAVAILABLE", SWING_RANK=float("nan")),
    ])
    summary = {"production_count": 0, "watch": [{"code": "000002", "name": "기존 선별 후보"}]}
    board = build_swing_board(df, summary)
    assert board["mode"] == "FALLBACK"
    assert [r["code"] for r in board["rows"]] == ["000002"]
    assert [r["code"] for r in board["research_rows"]] == ["000001"]
    assert board["oos_net_return"] == -.0268
    assert board["oos_coverage"] == .91
    with ui.column() as holder:
        render_swing_board(df, summary)
    try:
        research = next(el for el in holder.descendants() if el.tag == "nicegui-expansion")
        assert research.value is False
        assert research.text == "연구 모델 순위 · 검증 결과"
        research_elements = set(research.descendants())
        primary_links = [el.text for el in holder.descendants()
                         if el.tag == "nicegui-link" and el not in research_elements]
        assert primary_links == ["기존 선별 후보"]
        labels = [str(getattr(el, "text", "")) for el in holder.descendants()]
        assert "신규 모델 검증 미달 · 기존 선별 사용" in labels
        assert "63.0%" not in labels
        assert "90.0점" in labels
        assert any("-2.68%" in label for label in labels)
        assert any("-1.05%" in label for label in labels)
    finally:
        holder.delete()


def test_today_renders_shortlist_before_collapsed_legacy_details(monkeypatch):
    from nicegui import ui
    from components import decision_center as dc
    summary = {"status": "CASH", "action_label": "기존 행동 지시", "action_detail": "상세 내용",
               "next_check": "다음 확인", "production_count": 0, "gates": [], "buys": [],
               "watch": [], "blockers": [], "risk_off": {}, "holdings": {}, "track_record": []}
    monkeypatch.setattr(dc, "build_decision_summary", lambda df: summary)
    monkeypatch.setattr(dc, "_validation_payload", lambda df: {})
    with ui.column() as holder:
        dc.render_decision_center(pd.DataFrame([row()]))
    try:
        labels = [str(getattr(el, "text", "")) for el in holder.descendants()]
        assert labels.index("종목 005930") < labels.index("기존 행동 지시")
        expansions = [el for el in holder.descendants() if el.tag == "nicegui-expansion"]
        assert len(expansions) == 4  # forecast methodology plus three legacy sections
        assert all(el.value is False for el in expansions)
    finally:
        holder.delete()
