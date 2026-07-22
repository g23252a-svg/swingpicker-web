# -*- coding: utf-8 -*-
"""[v36] 미국 지수 오버나이트 피처 — look-ahead 방지가 생명.

코스피-나스닥 연동(반도체 ~30% + 외국인 수급 + NQ 선물 24h) 실사용 관찰에서
출발. us_daily.csv를 알파 학습 패널에 날짜 조인하며, 한국 거래일 d의 피처는
반드시 us_date < d 인 마지막 미국 마감만 사용한다.
피처 가치는 야간 워크포워드 검증 게이트가 자동 판정(무가치→무시, 유해→미사용).
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import alpha_engine as ae
from collect_us_daily import merge_us_frames


def _write_us(tmp, rows):
    pd.DataFrame(rows).to_csv(os.path.join(tmp, "us_daily.csv"), index=False)


def _panel(ymds):
    return pd.DataFrame({"_ymd": ymds, "종목코드": ["A"] * len(ymds)})


def test_no_lookahead_korea_day_uses_strictly_prior_us_close():
    tmp = tempfile.mkdtemp()
    _write_us(tmp, dict(date=["20260716", "20260717", "20260720"],
                        ndx_close=[100.0, 98.0, 101.0],
                        spx_close=[50.0, 49.5, 50.5]))
    out = ae.inject_us_features(_panel(["20260717", "20260720", "20260721"]), tmp)
    r = out.set_index("_ymd")["US_NDX_RET_1D"]
    # 한국 7/17은 미국 7/17 마감(-2%)을 절대 보면 안 됨 (그날 밤에야 확정)
    assert pd.isna(r["20260717"])          # 미국 7/16이 최신 → RET_1D는 첫 행이라 NaN
    assert abs(r["20260720"] - (-2.0)) < 0.01   # 미국 7/17 마감 사용
    assert abs(r["20260721"] - 3.0612) < 0.01   # 미국 7/20 마감 사용


def test_same_us_date_never_used_even_if_exact_match():
    # merge_asof allow_exact_matches=False 검증 — 한국일 == 미국일이어도 금지
    tmp = tempfile.mkdtemp()
    _write_us(tmp, dict(date=["20260716", "20260717"],
                        ndx_close=[100.0, 90.0], spx_close=[50.0, 45.0]))
    out = ae.inject_us_features(_panel(["20260717"]), tmp)
    # 미국 7/17(-10%)이 아니라 7/16(첫 행 → NaN)이어야 함
    assert pd.isna(out.iloc[0]["US_NDX_RET_1D"])


def test_missing_us_file_graceful():
    out = ae.inject_us_features(_panel(["20260717"]), tempfile.mkdtemp())
    assert "US_NDX_RET_1D" not in out.columns   # 컬럼 미생성 (모델이 자연 미사용)


def test_feature_cols_not_banned():
    for c in ae.US_OVERNIGHT_COLS:
        assert c not in ae._BAN_COLS


def test_us_feature_table_computes_ma20_and_rets():
    tmp = tempfile.mkdtemp()
    dates = [f"202601{d:02d}" for d in range(1, 26)]
    _write_us(tmp, dict(date=dates,
                        ndx_close=list(np.linspace(100, 124, 25)),
                        spx_close=list(np.linspace(50, 62, 25))))
    t = ae._load_us_feature_table(tmp)
    assert set(ae.US_OVERNIGHT_COLS) <= set(t.columns)
    assert t["US_NDX_VS_MA20"].notna().sum() > 0   # MA20 형성 후 값 존재
    assert t["US_SPX_RET_1D"].notna().sum() >= 20


def test_collector_merge_outer_and_none_safe():
    m = merge_us_frames({
        "ndx_close": pd.DataFrame({"date": ["20260101", "20260102"],
                                   "close": [1.0, 2.0]}),
        "spx_close": pd.DataFrame({"date": ["20260102"], "close": [9.0]}),
    })
    assert list(m.columns) == ["date", "ndx_close", "spx_close"]
    assert len(m) == 2 and pd.isna(m.iloc[0]["spx_close"])
    assert merge_us_frames({"ndx_close": None}) is None
