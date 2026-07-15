# -*- coding: utf-8 -*-
"""[v31] 종목 탭 재구성 — 히어로 스트립 렌더 스모크."""
import sys
import types

import pandas as pd


class _DummyUi:
    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if "nicegui" not in sys.modules:
    stub = types.ModuleType("nicegui")
    stub.ui = _DummyUi()
    stub.run = _DummyUi()
    stub.app = _DummyUi()
    sys.modules["nicegui"] = stub

from components.tab_stocks import _render_hero_strip  # noqa: E402


def _df(**cols):
    base = {"종목코드": ["000001"], "종목명": ["테스트"]}
    base.update({k: [v] for k, v in cols.items()})
    return pd.DataFrame(base)


def test_hero_strip_with_v28_columns():
    df = _df(MARKET_REGIME="DOWN", REGIME_REASON="시장폭 22%",
             MARKET_BREADTH=22.5, ALPHA_VALIDATED=1)
    _render_hero_strip(df, {"status": "CASH", "official_count": 0})


def test_hero_strip_official_buy_day():
    df = _df(MARKET_REGIME="UP", MARKET_BREADTH=62.0, ALPHA_VALIDATED=0)
    _render_hero_strip(df, {"status": "OFFICIAL_BUY_AVAILABLE", "official_count": 2})


def test_hero_strip_legacy_csv_without_new_columns():
    # 레짐/시장폭/알파 컬럼이 없는 legacy CSV → 크래시 없이 렌더
    _render_hero_strip(_df(), {"official_count": 0})
    _render_hero_strip(_df(), None)


def test_hero_strip_nan_breadth():
    _render_hero_strip(_df(MARKET_BREADTH=float("nan")), {"official_count": 0})


def test_verification_excluded_mask_demotes_stale_rows():
    from components.tab_stocks import _verification_excluded_mask

    df = pd.DataFrame({
        "DISPLAY_SCORE": [85.0, 0.0, -5.0, 0.0, 0.0],
        "IS_REAL_HOLDING": [False, False, False, True, False],
        "ROUTE": ["WAIT", "WAIT", "NEUTRAL", "WAIT", "CARRY"],
    })
    m = _verification_excluded_mask(df)
    # 정상 점수 → 유지 / 0·음수 비보유 → 제외 / 실보유·CARRY → 유지
    assert m.tolist() == [False, True, True, False, False]


def test_verification_excluded_mask_empty_df():
    from components.tab_stocks import _verification_excluded_mask
    assert _verification_excluded_mask(pd.DataFrame()).empty


def test_table_alpha_display():
    from components.tab_stocks import _table_alpha_display

    # 검증 통과 + 승률 → "92 (41%)"
    r = pd.Series({"ALPHA_VALIDATED": 1, "ALPHA_SCORE": 92.3, "ALPHA_WIN_PROB": 0.4116})
    assert _table_alpha_display(r) == "92 (41%)"
    # 검증 통과 + 승률 없음 → 백분위만
    r2 = pd.Series({"ALPHA_VALIDATED": "1", "ALPHA_SCORE": 55.0})
    assert _table_alpha_display(r2) == "55"
    # 미검증 → 대시 (검증 통과했을 때만 신뢰하는 계약)
    r3 = pd.Series({"ALPHA_VALIDATED": 0, "ALPHA_SCORE": 92.3})
    assert _table_alpha_display(r3) == "—"
    # 컬럼 없음 (legacy) → 대시
    assert _table_alpha_display(pd.Series({"종목명": "x"})) == "—"
    # NaN 점수 → 대시
    import numpy as np
    r4 = pd.Series({"ALPHA_VALIDATED": 1, "ALPHA_SCORE": np.nan})
    assert _table_alpha_display(r4) == "—"
