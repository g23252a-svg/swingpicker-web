# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from components.tab_stocks import _apply_stock_search_filter


def _df():
    return pd.DataFrame([
        {"종목코드": "005930", "종목명": "삼성전자", "업종": "반도체 제조업"},
        {"종목코드": "000660", "종목명": "SK하이닉스", "업종": "반도체 제조업"},
        {"종목코드": "035420", "종목명": "NAVER", "업종": "소프트웨어 개발 및 공급업"},
    ])


def test_search_by_korean_stock_name():
    out = _apply_stock_search_filter(_df(), "삼성")
    assert list(out["종목명"]) == ["삼성전자"]


def test_search_by_code_with_padding():
    out = _apply_stock_search_filter(_df(), "5930")
    assert list(out["종목코드"]) == ["005930"]


def test_search_by_sector():
    out = _apply_stock_search_filter(_df(), "반도체")
    assert set(out["종목명"]) == {"삼성전자", "SK하이닉스"}


def test_empty_query_returns_original():
    df = _df()
    out = _apply_stock_search_filter(df, "")
    assert len(out) == len(df)
