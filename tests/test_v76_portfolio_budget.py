# -*- coding: utf-8 -*-
"""v76 포트폴리오 예산 — 축소만 하고, 상한을 넘기지 않고, 목록을 건드리지 않는다."""
import os
import numpy as np
import pandas as pd
import pytest

from services import portfolio_budget as B


# ── 픽스처 ────────────────────────────────────────────────────────────────
def _csv(d, ymd, fracs, budgeted=False, amounts=None):
    n = len(fracs)
    row = {"종목코드": [f"{i:06d}" for i in range(n)],
           "KELLY_FRACTION": fracs,
           "켈리_금액(원)": amounts if amounts is not None
                            else [int(f * 10_000_000) for f in fracs],
           "켈리_수량": [10] * n,
           "추천매수가": [10_000] * n}
    if budgeted:
        row[B.COL_SCALE] = [1.0] * n
    pd.DataFrame(row).to_csv(os.path.join(d, f"recommend_{ymd}.csv"), index=False)


@pytest.fixture
def data_dir(tmp_path):
    return str(tmp_path)


def _df(fracs, buy=10_000):
    n = len(fracs)
    return pd.DataFrame({"종목코드": [f"{i:06d}" for i in range(n)],
                         "KELLY_FRACTION": fracs,
                         "켈리_금액(원)": [int(f * 10_000_000) for f in fracs],
                         "켈리_수량": [999] * n,
                         "추천매수가": [buy] * n,
                         "PRODUCTION_BUY": [1] * n,
                         "TOP_PICK": [1] * n})


# ── 핵심 불변식 ────────────────────────────────────────────────────────────
def test_never_increases_any_position(data_dir):
    """어떤 행도 원래 분율보다 커지지 않는다."""
    df = _df([0.01, 0.02, 0.03])
    out, info = B.apply(df, data_dir, "20260828")
    assert (out["KELLY_FRACTION"].values <= df["KELLY_FRACTION"].values + 1e-9).all()
    assert info["scale"] <= 1.0


def test_small_book_untouched(data_dir):
    """예산에 여유가 있으면 손대지 않는다."""
    df = _df([0.01, 0.02])
    out, info = B.apply(df, data_dir, "20260828")
    assert info["scale"] == 1.0
    np.testing.assert_allclose(out["KELLY_FRACTION"].values, [0.01, 0.02])
    assert "축소 없음" in out[B.COL_REASON].iloc[0]


def test_day_cap_enforced(data_dir):
    """이전 포지션이 없어도 하루 배정(20%)을 넘지 못한다."""
    out, info = B.apply(_df([0.25] * 8), data_dir, "20260828")
    assert out["KELLY_FRACTION"].sum() <= B.DAY_CAP + 1e-9


def test_name_cap_enforced(data_dir):
    """종목당 상한이 먼저 걸린다."""
    out, info = B.apply(_df([0.25, 0.01]), data_dir, "20260828")
    assert info["capped"] == 1
    assert out["KELLY_FRACTION"].max() <= B.NAME_CAP + 1e-9


def test_book_cap_never_exceeded_across_cohorts(data_dir):
    """5개 코호트가 겹쳐도 총노출이 100%를 넘지 않는다."""
    for y in ["20260821", "20260824", "20260825", "20260826", "20260827"]:
        _csv(data_dir, y, [0.20], budgeted=True)
    used, _ = B.live_exposure(data_dir, "20260828")
    out, info = B.apply(_df([0.25] * 5), data_dir, "20260828")
    assert used + out["KELLY_FRACTION"].sum() <= B.BOOK_CAP + 1e-9


def test_rounding_floors_not_rounds(data_dir):
    """반올림이 상한을 넘기면 안 된다 (0.066666 × 3 = 0.2001)."""
    out, _ = B.apply(_df([0.10] * 3), data_dir, "20260828")
    assert out["KELLY_FRACTION"].sum() <= B.DAY_CAP + 1e-9


def test_qty_floored_not_rounded(data_dir):
    """수량은 내림 — 예산을 넘겨 사면 안 된다."""
    out, _ = B.apply(_df([0.10] * 3, buy=33_333), data_dir, "20260828")
    amt = out["켈리_금액(원)"].values
    assert (out["켈리_수량"].values * 33_333 <= amt + 1e-6).all()


# ── 목록 불변 ─────────────────────────────────────────────────────────────
def test_recommendation_list_unchanged(data_dir):
    """추천 목록·PRODUCTION_BUY·TOP_PICK·행 순서 무변경."""
    df = _df([0.25, 0.10, 0.05])
    out, _ = B.apply(df, data_dir, "20260828")
    assert list(out["종목코드"]) == list(df["종목코드"])
    assert list(out["PRODUCTION_BUY"]) == list(df["PRODUCTION_BUY"])
    assert list(out["TOP_PICK"]) == list(df["TOP_PICK"])
    assert len(out) == len(df)


def test_raw_preserved(data_dir):
    """축소 전 값이 보존된다."""
    df = _df([0.25, 0.10])
    out, _ = B.apply(df, data_dir, "20260828")
    np.testing.assert_allclose(out[B.COL_RAW_F].values, [0.25, 0.10])
    assert out[B.COL_RAW_AMT].iloc[0] == 2_500_000


def test_raw_not_overwritten_on_second_apply(data_dir):
    """두 번 적용해도 원값은 첫 값을 유지한다."""
    out1, _ = B.apply(_df([0.25]), data_dir, "20260828")
    out2, _ = B.apply(out1, data_dir, "20260828")
    assert out2[B.COL_RAW_F].iloc[0] == 0.25


# ── 이전 노출 재생 ─────────────────────────────────────────────────────────
def test_legacy_unbudgeted_days_are_clamped(data_dir):
    """예산 이전 배치의 집행 불가능한 분율(184%)을 그대로 세지 않는다."""
    _csv(data_dir, "20260826", [0.25] * 8)          # 200% 요구, BUDGET_SCALE 없음
    used, detail = B.live_exposure(data_dir, "20260828")
    assert detail["20260826"] <= B.DAY_CAP + 1e-9
    assert used <= B.BOOK_CAP + 1e-9


def test_budgeted_day_taken_as_is(data_dir):
    """이미 예산이 적용된 배치는 재축소하지 않는다."""
    _csv(data_dir, "20260826", [0.07], budgeted=True)
    _, detail = B.live_exposure(data_dir, "20260828")
    assert detail["20260826"] == pytest.approx(0.07)


def test_only_hold_window_counts(data_dir):
    """보유기간을 벗어난 배치는 노출에서 빠진다."""
    for y in ["20260801", "20260821", "20260824", "20260825", "20260826"]:
        _csv(data_dir, y, [0.05], budgeted=True)
    _, detail = B.live_exposure(data_dir, "20260828")
    assert "20260801" not in detail
    assert len(detail) == B.HOLD_DAYS - 1


def test_future_batches_excluded(data_dir):
    """당일·이후 배치는 '보유 중'이 아니다."""
    _csv(data_dir, "20260828", [0.20], budgeted=True)
    _csv(data_dir, "20260831", [0.20], budgeted=True)
    _, detail = B.live_exposure(data_dir, "20260828")
    assert "20260828" not in detail and "20260831" not in detail


def test_amount_only_legacy_csv(data_dir):
    """분율 컬럼이 없던 구 배치는 금액에서 되돌린다."""
    pd.DataFrame({"켈리_금액(원)": [1_500_000]}).to_csv(
        os.path.join(data_dir, "recommend_20260826.csv"), index=False)
    _, detail = B.live_exposure(data_dir, "20260828")
    assert detail["20260826"] == pytest.approx(0.15)


def test_no_room_gives_zero(data_dir):
    """예산이 다 찼으면 신규 배분은 0이고 사유를 적는다."""
    out, info = B.apply(_df([0.10]), data_dir, "20260828",
                        book_cap=0.0, day_cap=None)
    assert out["KELLY_FRACTION"].iloc[0] == 0
    assert out["켈리_수량"].iloc[0] == 0
    assert "신규 배분 0" in out[B.COL_REASON].iloc[0]


# ── 견고성 ────────────────────────────────────────────────────────────────
def test_empty_and_missing_columns(data_dir):
    for df in (pd.DataFrame(), pd.DataFrame({"종목코드": ["000001"]}), None):
        out, info = B.apply(df, data_dir, "20260828")
        assert info["applied"] is False


def test_nan_fraction_is_zero(data_dir):
    df = _df([0.05, 0.05]); df.loc[0, "KELLY_FRACTION"] = np.nan
    out, _ = B.apply(df, data_dir, "20260828")
    assert out["KELLY_FRACTION"].iloc[0] == 0


def test_negative_fraction_clipped(data_dir):
    df = _df([-0.5, 0.05])
    out, _ = B.apply(df, data_dir, "20260828")
    assert (out["KELLY_FRACTION"] >= 0).all()


def test_unreadable_csv_does_not_crash(data_dir):
    with open(os.path.join(data_dir, "recommend_20260826.csv"), "w") as f:
        f.write("\x00\x00 not,a,csv\n\"unclosed")
    used, _ = B.live_exposure(data_dir, "20260828")
    assert used >= 0


def test_zero_buy_price_no_division_error(data_dir):
    out, _ = B.apply(_df([0.05], buy=0), data_dir, "20260828")
    assert out["켈리_수량"].iloc[0] == 0


# ── 갭 관통 ───────────────────────────────────────────────────────────────
def test_worst_case_is_worse_than_nominal():
    """갭을 반영하면 각오할 손실이 선언 손절폭보다 나쁘다."""
    w = B.worst_case_pct(-0.08)
    assert w["expected"] < w["nominal"]
    assert w["gap_fill"] <= w["nominal"]


def test_worst_case_gap_fill_cannot_beat_stop():
    """관통 체결이 손절보다 좋을 수는 없다."""
    w = B.worst_case_pct(-0.08, gap_fill=-0.01)
    assert w["gap_fill"] == pytest.approx(-0.08)
    assert w["expected"] == pytest.approx(-0.08)


def test_worst_case_zero_gap_rate():
    w = B.worst_case_pct(-0.08, gap_rate=0.0)
    assert w["expected"] == pytest.approx(-0.08)


def test_stop_worst_line_mentions_numbers():
    s = B.stop_worst_line(-0.08)
    assert "-8%" in s and "약속이 아니다" in s


def test_line_empty_when_not_applied():
    assert B.line({"applied": False}) == ""
    assert B.line({}) == ""


# ── 회귀: 실제 배치 파일로 ──────────────────────────────────────────────────
def test_real_batch_stays_within_book(real_data_mirror):
    """실배치 CSV로 돌려도 총노출이 100%를 넘지 않는다."""
    import glob
    d = real_data_mirror("recommend_2*.csv")
    files = sorted(glob.glob(os.path.join(d, "recommend_2*.csv")))
    if not files:
        pytest.skip("배치 CSV 없음")
    ymd = os.path.basename(files[-1])[10:18]
    df = pd.read_csv(files[-1], dtype={"종목코드": str})
    if "KELLY_FRACTION" not in df.columns:
        pytest.skip("KELLY_FRACTION 없는 배치")
    used, _ = B.live_exposure(d, ymd)
    out, info = B.apply(df, d, ymd)
    assert used + out["KELLY_FRACTION"].sum() <= B.BOOK_CAP + 1e-6
    assert (out["KELLY_FRACTION"].values
            <= pd.to_numeric(df["KELLY_FRACTION"], errors="coerce").fillna(0).values + 1e-9).all()


# ── 재분배: 0주는 배분이 아니다 ─────────────────────────────────────────────
def test_no_zero_share_rows_when_redistributable(data_dir):
    """1주도 못 사는 행을 남기지 않는다 — 못 사면 떨어뜨리고 재분배."""
    df = _df([0.02] * 20, buy=100_000)          # 20종목 × 10만원
    out, _ = B.apply(df, data_dir, "20260828")
    live = out["KELLY_FRACTION"] > 0
    assert int(((out["켈리_수량"] == 0) & live).sum()) == 0
    assert int((out["켈리_수량"] > 0).sum()) >= 1


def test_redistribute_keeps_budget(data_dir):
    """재분배해도 트랜치를 넘지 않는다."""
    df = _df([0.02] * 20, buy=100_000)
    out, _ = B.apply(df, data_dir, "20260828")
    assert out["KELLY_FRACTION"].sum() <= B.DAY_CAP + 1e-9
    spend = (out["켈리_수량"] * out["추천매수가"]).sum()
    assert spend <= B.DAY_CAP * 10_000_000 + 1


def test_redistribute_never_exceeds_name_cap(data_dir):
    """재분배가 종목당 상한을 뚫으면 안 된다."""
    df = _df([0.25] * 3, buy=1_000)
    out, _ = B.apply(df, data_dir, "20260828")
    assert out["KELLY_FRACTION"].max() <= B.NAME_CAP + 1e-9


def test_redistribute_never_increases_over_raw(data_dir):
    """재분배해도 원래 분율을 넘지 않는다."""
    df = _df([0.03, 0.02, 0.01], buy=400_000)
    raw = df["KELLY_FRACTION"].values.copy()
    out, _ = B.apply(df, data_dir, "20260828")
    assert (out["KELLY_FRACTION"].values <= raw + 1e-9).all()


def test_all_names_unaffordable_gives_nothing(data_dir):
    """요청 분율로 1주도 못 사면 배분하지 않는다 — 부풀리지 않는다."""
    df = _df([0.05], buy=50_000_000)
    out, info = B.apply(df, data_dir, "20260828")
    assert out["켈리_수량"].iloc[0] == 0
    assert out["KELLY_FRACTION"].iloc[0] == 0


def test_dropped_weight_is_not_reassigned(data_dir):
    """떨어진 행의 몫을 남은 행에 얹지 않는다 (새 베팅 금지)."""
    df = _df([0.03, 0.02, 0.01], buy=400_000)
    raw = df["KELLY_FRACTION"].values.copy()
    out, _ = B.apply(df, data_dir, "20260828")
    assert (out["KELLY_FRACTION"].values <= raw + 1e-9).all()


def test_redistribute_terminates_on_pathological_input(data_dir):
    """가격이 제각각이어도 무한루프에 빠지지 않는다."""
    df = _df([0.01] * 30, buy=1)
    df["추천매수가"] = [10 ** (i % 8) for i in range(30)]
    out, _ = B.apply(df, data_dir, "20260828")
    assert out["KELLY_FRACTION"].sum() <= B.DAY_CAP + 1e-9


def test_book_cap_binds_when_tighter_than_day_cap(data_dir):
    """보유 노출이 많으면 남은 예산이 일일 트랜치보다 작아진다 — 그때는 총노출이 건다."""
    for y, f in [("20260821", 0.30), ("20260824", 0.30),
                 ("20260825", 0.30), ("20260826", 0.05)]:
        _csv(data_dir, y, [f], budgeted=True)
    used, _ = B.live_exposure(data_dir, "20260828")
    assert used == pytest.approx(0.95)
    out, info = B.apply(_df([0.10] * 3, buy=1_000), data_dir, "20260828")
    assert info["room"] == pytest.approx(0.05)      # 0.20 이 아니라 0.05
    assert out["KELLY_FRACTION"].sum() <= 0.05 + 1e-9
    assert used + out["KELLY_FRACTION"].sum() <= B.BOOK_CAP + 1e-9


def test_name_cap_binds_inside_tranche(data_dir):
    """트랜치에 여유가 있어도 한 종목이 종목상한을 넘지 못한다."""
    out, _ = B.apply(_df([0.25, 0.001], buy=1_000), data_dir, "20260828")
    assert out["KELLY_FRACTION"].iloc[0] <= B.NAME_CAP + 1e-9


def test_total_never_exceeds_requested_sum(data_dir):
    """요구 합계보다 크게 배분하지 않는다."""
    df = _df([0.01, 0.02], buy=1_000)
    out, _ = B.apply(df, data_dir, "20260828")
    assert out["KELLY_FRACTION"].sum() <= 0.03 + 1e-9
