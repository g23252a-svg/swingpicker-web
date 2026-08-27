# -*- coding: utf-8 -*-
"""v74 — 재추천 이력 표시 + 배치 종가를 '현재'라고 부르지 않기.

사용자 신고 두 건이 출발점이다.
  (1) "오늘탭에 뜬 추천종목이 2일전과 똑같은데?"
      → 맞았다. 공식픽 13건 중 아주IB투자 3회, 로킷헬스케어 2회.
  (2) "현재가도 27일 종가랑 안맞아"
      → 맞았다. 8/27 16:12에 화면이 "현재 3,360"을 띄웠는데 그건 8/26 종가였고
        실제 8/27 종가는 3,480이었다. 수집은 정상이었고 **라벨이 거짓말**이었다.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from services import pick_history as ph


def _rec(d, ymd, picks):
    """picks: [(code, name)] — PRODUCTION_BUY=1 인 행."""
    rows = [{"종목코드": c, "종목명": n, "PRODUCTION_BUY": 1, "추천매수가": 1000,
             "손절가": 920} for c, n in picks]
    rows += [{"종목코드": "999999", "종목명": "기타", "PRODUCTION_BUY": 0,
              "추천매수가": 1000, "손절가": 920}]
    pd.DataFrame(rows).to_csv(os.path.join(d, f"recommend_{ymd}.csv"),
                              index=False, encoding="utf-8-sig")


def _px(d, code, dates, o, h, l, c):
    return pd.DataFrame({"종목코드": code, "Date": pd.to_datetime(dates),
                         "시가": o, "고가": h, "저가": l, "종가": c})


def _write_px(d, frames):
    pd.concat(frames, ignore_index=True).to_parquet(
        os.path.join(d, "ohlcv_union_hl.parquet"))


@pytest.fixture
def env(tmp_path):
    d = str(tmp_path)
    days = [f"202603{i:02d}" for i in range(1, 21)]
    dts = pd.to_datetime(days)
    n = len(days)
    _write_px(d, [
        _px(d, "000001", dts, [1000] * n, [1050] * n, [980] * n, [1000] * n),   # 평탄
        _px(d, "000002", dts, [1000] * n, [1010] * n,
            [1000] * 5 + [900] + [1000] * (n - 6), [1000] * n),                  # 6일차 급락
    ])
    return d, days


# ── 재등장 카운트 ─────────────────────────────────────────────
def test_first_pick_has_no_line(env):
    d, days = env
    _rec(d, days[0], [("000001", "가")])
    h = ph.build(d, days[1])
    assert h["000001"]["nth"] == 2          # 다음 배치 기준으로는 2번째
    _rec(d, days[1], [])
    assert ph.build(d, days[0]) == {}       # 그 이전에는 이력 없음


def test_counts_prior_picks_only(env):
    d, days = env
    for y in (days[0], days[2], days[4]):
        _rec(d, y, [("000001", "가")])
    assert ph.build(d, days[3])["000001"]["nth"] == 3    # 0,2 → 이번이 3번째
    assert ph.build(d, days[5])["000001"]["nth"] == 4


def test_today_batch_not_counted_as_prior(env):
    """오늘 배치를 과거로 세면 '3번째'가 '4번째'가 된다."""
    d, days = env
    for y in (days[0], days[2], days[4]):
        _rec(d, y, [("000001", "가")])
    assert ph.build(d, days[4])["000001"]["nth"] == 3


def test_non_production_rows_ignored(env):
    d, days = env
    _rec(d, days[0], [])                    # PRODUCTION_BUY=0 만 있음
    assert ph.build(d, days[2]) == {}


def test_line_empty_for_first_time(env):
    d, days = env
    _rec(d, days[0], [("000001", "가")])
    h = ph.build(d, days[1])
    h["000001"]["nth"] = 1
    assert ph._line(h["000001"]) == ""


# ── 실현수익 ──────────────────────────────────────────────────
def test_flat_stock_measures_zero(env):
    d, days = env
    _rec(d, days[0], [("000001", "가")])
    h = ph.build(d, days[10])
    assert h["000001"]["prior"][0]["ret"] == pytest.approx(0.0, abs=1e-9)


def test_stop_detected(env):
    d, days = env
    _rec(d, days[3], [("000002", "나")])     # 진입 days[4], 6일차(index5) 저가 900
    h = ph.build(d, days[12])
    assert h["000002"]["prior"][0]["ret"] == ph.STOP_PCT


def test_stop_reported_before_window_completes(env):
    """손절이 이미 터졌으면 5일 창이 안 차도 결과는 확정이다.

    첫 판본은 창이 찰 때까지 '측정중'으로 뒀는데, 픽 대부분이 손절로
    끝나는 마당에 그건 결과를 며칠씩 숨기는 짓이었다.
    """
    d, days = env
    dts = pd.to_datetime(days[:7])           # days[5]까지만 존재 + 1
    _write_px(d, [_px(d, "000003", dts, [1000] * 7, [1010] * 7,
                      [1000] * 5 + [900, 900], [1000] * 7)])
    _rec(d, days[3], [("000003", "다")])
    h = ph.build(d, days[6])
    assert h["000003"]["prior"][0]["ret"] == ph.STOP_PCT


def test_unfinished_without_stop_is_none(env):
    d, days = env
    dts = pd.to_datetime(days[:7])
    _write_px(d, [_px(d, "000004", dts, [1000] * 7, [1010] * 7, [995] * 7, [1000] * 7)])
    _rec(d, days[4], [("000004", "라")])      # 진입 days[5], 창이 안 참
    assert ph.build(d, days[6])["000004"]["prior"][0]["ret"] is None


def test_halted_entry_not_measured(env):
    """거래정지(시가/고가/저가=0)면 살 수 없다. 저가 0을 손절로 세면 안 된다."""
    d, days = env
    dts = pd.to_datetime(days)
    n = len(days)
    o = [1000] * n; h_ = [1010] * n; l = [995] * n
    o[4] = h_[4] = l[4] = 0                  # 진입일 정지
    _write_px(d, [_px(d, "000005", dts, o, h_, l, [1000] * n)])
    _rec(d, days[3], [("000005", "마")])
    assert ph.build(d, days[12])["000005"]["prior"][0]["ret"] is None


def test_halted_signal_day_not_measured(env):
    """신호일에 정지였다면 그날 가격을 근거로 추천할 수 없었다.

    진입일 정지는 시가=0 이라 어차피 걸리지만, **신호일** 정지는 그렇지 않다 —
    이 구멍을 변이 검정이 찾아냈다.
    """
    d, days = env
    dts = pd.to_datetime(days)
    n = len(days)
    o = [1000] * n; h_ = [1010] * n; l = [995] * n
    o[3] = h_[3] = l[3] = 0                  # 신호일(days[3])만 정지, 진입일은 정상
    _write_px(d, [_px(d, "000007", dts, o, h_, l, [1000] * n)])
    _rec(d, days[3], [("000007", "사")])
    assert ph.build(d, days[12])["000007"]["prior"][0]["ret"] is None


def test_halt_inside_window_not_counted_as_stop(env):
    d, days = env
    dts = pd.to_datetime(days)
    n = len(days)
    o = [1000] * n; h_ = [1010] * n; l = [995] * n
    o[7] = h_[7] = l[7] = 0                  # 보유 중 하루 정지
    _write_px(d, [_px(d, "000006", dts, o, h_, l, [1000] * n)])
    _rec(d, days[3], [("000006", "바")])
    r = ph.build(d, days[15])["000006"]["prior"][0]["ret"]
    assert r is not None and r != ph.STOP_PCT


def test_missing_price_data_is_none(env):
    d, days = env
    _rec(d, days[0], [("777777", "없음")])
    assert ph.build(d, days[5])["777777"]["prior"][0]["ret"] is None


# ── 문구 ──────────────────────────────────────────────────────
def test_line_shows_count_and_prior_results(env):
    d, days = env
    for y in (days[0], days[2]):
        _rec(d, y, [("000001", "가")])
    line = ph.build(d, days[10])["000001"]["line"]
    assert "3번째 추천" in line and "+0.0%" in line


def test_line_marks_all_stopped(env):
    d, days = env
    _rec(d, days[3], [("000002", "나")])
    line = ph.build(d, days[15])["000002"]["line"]
    assert "전부 손절" in line and "-8.0%" in line


def test_line_shows_pending_as_measuring(env):
    d, days = env
    dts = pd.to_datetime(days[:7])
    _write_px(d, [_px(d, "000004", dts, [1000] * 7, [1010] * 7, [995] * 7, [1000] * 7)])
    _rec(d, days[4], [("000004", "라")])
    assert "측정중" in ph.build(d, days[6])["000004"]["line"]


def test_line_caps_shown_history(env):
    d, days = env
    for y in days[:8]:
        _rec(d, y, [("000001", "가")])
    h = ph.build(d, days[10])
    assert h["000001"]["nth"] == 9
    assert len(h["000001"]["prior"]) == ph.MAX_SHOWN


# ── annotate / 요약 ───────────────────────────────────────────
def test_annotate_adds_display_columns_only(env):
    d, days = env
    _rec(d, days[0], [("000001", "가")])
    h = ph.build(d, days[5])
    df = pd.DataFrame({"종목코드": ["000001", "000009"], "PRODUCTION_BUY": [1, 0],
                       "켈리_수량": [10, 0]})
    before = set(df.columns)
    out = ph.annotate(df.copy(), h)
    added = set(out.columns) - before
    assert added == {ph.COL_NTH, ph.COL_PRIOR, ph.COL_PRIOR_N, ph.COL_PRIOR_AVG}
    assert out.loc[0, ph.COL_NTH] == 2 and out.loc[1, ph.COL_NTH] == 1
    assert list(out["PRODUCTION_BUY"]) == [1, 0]
    assert list(out["켈리_수량"]) == [10, 0]


def test_annotate_survives_empty(env):
    assert ph.annotate(pd.DataFrame(), {}).empty
    df = pd.DataFrame({"종목코드": ["000001"]})
    assert ph.annotate(df, {}) is df


def test_repeat_summary_accepts_series(env):
    """`codes or []` 는 pandas Series 에서 ValueError 를 낸다 — 이 세션에서 밟은 함정."""
    d, days = env
    _rec(d, days[0], [("000001", "가")])
    h = ph.build(d, days[5])
    s = pd.Series(["000001", "000009"])
    assert "1종목" in ph.repeat_summary(h, s)
    assert ph.repeat_summary(h, None) == ""
    assert ph.repeat_summary({}, s) == ""


def test_stop_reality_line_reports_pinning():
    df = pd.DataFrame({"추천매수가": [1000, 1000, 1000],
                       "손절가": [920, 921, 960]})
    l = ph.stop_reality_line(df)
    assert "67%" in l and "매일 종가를 따라" in l


def test_stop_reality_line_empty_on_garbage():
    assert ph.stop_reality_line(pd.DataFrame()) == ""
    assert ph.stop_reality_line(pd.DataFrame({"추천매수가": [0], "손절가": [0]})) == ""


# ── 화면 연결 ─────────────────────────────────────────────────
def _src(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


def test_pick_nth_survives_missing_column():
    """`pd.to_numeric(None)` 은 NaN 이고 `NaN or 1` 에서 NaN 이 참이라
    int(NaN) 이 터진다 — 컬럼 없는 배치에서 화면 전체가 죽었다."""
    from components.decision_center import _pick_nth
    assert _pick_nth(pd.Series({"종목코드": "000001"})) == 1
    assert _pick_nth(pd.Series({ph.COL_NTH: None})) == 1
    assert _pick_nth(pd.Series({ph.COL_NTH: float("nan")})) == 1
    assert _pick_nth(pd.Series({ph.COL_NTH: "이상한값"})) == 1
    assert _pick_nth(pd.Series({ph.COL_NTH: 0})) == 1
    assert _pick_nth(pd.Series({ph.COL_NTH: 3})) == 3
    assert _pick_nth(pd.Series({ph.COL_NTH: "4"})) == 4


def test_decision_center_renders_repeat_line():
    src = _src("components/decision_center.py")
    assert "_render_repeat_line" in src
    assert src.count("_render_repeat_line(stock)") >= 2, "매수/관찰 카드 둘 다에 있어야 한다"
    assert '"pick_prior"' in src and '"pick_nth"' in src


def test_batch_wiring_is_display_only():
    src = _src("pipeline_finalize.py")
    i = src.index("[v74] 사용자 신고")
    blk = src[i:i + 1400]
    assert "_PH.annotate" in blk and "except Exception" in blk
    assert "PRODUCTION_BUY =" not in blk and "켈리_수량" not in blk


def test_pick_history_touches_no_decision_column():
    src = _src("services/pick_history.py")
    for forbidden in ('df["PRODUCTION_BUY"]', "켈리_수량", "추천수량", 'df["ROUTE"]'):
        assert forbidden not in src


# ── 배치 종가를 '현재'라고 부르지 않는다 ──────────────────────
def test_close_is_not_labeled_as_current():
    """8/27 16:12 화면이 '현재 3,360'을 띄웠는데 그건 8/26 종가였다.
    실제 8/27 종가는 3,480 — 3.5% 차이."""
    src = _src("components/stock_detail_v2.py")
    assert "현재 {_won(close)}" not in src, "배치 종가를 '현재'라고 쓰고 있다"
    assert "{_as_of_label} {_won(close)}" in src


def test_as_of_label_uses_batch_date():
    src = _src("components/stock_detail_v2.py")
    assert '"as_of": _safe_str(row.get("기준일", "")' in src
    assert "_as_of_label" in src and "배치 종가" in src


@pytest.mark.parametrize("ymd,expect", [
    ("20260826", "08/26 종가"), ("", "배치 종가"), ("abc", "배치 종가"),
    ("2026082", "배치 종가"),
])
def test_as_of_label_logic(ymd, expect):
    _ao = str(ymd or "").strip()
    label = (f"{_ao[4:6]}/{_ao[6:8]} 종가"
             if len(_ao) == 8 and _ao.isdigit() else "배치 종가")
    assert label == expect
