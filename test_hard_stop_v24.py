# -*- coding: utf-8 -*-
"""test_hard_stop_v24.py — v24 P0-A 손절 하드캡 + P0-B 이상 폭등주 필터 검증."""
import pandas as pd
import stop_logic as SL
import trade_plan as TP


def test_hardcap_caps_overwide_stop():
    """floor를 타이트(5%)하게 주면 더 넓은 손절이 floor 이내로 캡된다."""
    cfg = SL.get_config()
    old = getattr(cfg, "hard_stop_floor_pct", 20.0)
    try:
        cfg.hard_stop_floor_pct = 5.0
        p = TP.build_trade_plan(buy=10000, atr_val=500, last_c=10000, mcap=50000)
        assert p.stop_pct <= 5.5, "손절폭 %.2f%% 가 floor 5%%를 넘음" % p.stop_pct
        assert p.stop >= 10000 * 0.94, "손절가가 floor 하한보다 낮음"
    finally:
        cfg.hard_stop_floor_pct = old


def test_hardcap_noop_within_floor():
    """floor 기본(20%)에서 정상 손절(-10%대)은 캡되지 않는다."""
    cfg = SL.get_config()
    old = getattr(cfg, "hard_stop_floor_pct", 20.0)
    try:
        cfg.hard_stop_floor_pct = 20.0
        p = TP.build_trade_plan(buy=10000, atr_val=500, last_c=10000, mcap=50000)
        assert p.stop_pct < 20.0
        assert "HARDCAP" not in (p.plan_reason or "")
    finally:
        cfg.hard_stop_floor_pct = old


def test_config_has_hard_stop_floor():
    """StopConfig에 hard_stop_floor_pct 필드가 존재한다."""
    cfg = SL.get_config()
    assert hasattr(cfg, "hard_stop_floor_pct")
    assert float(cfg.hard_stop_floor_pct) > 0


def test_abnormal_surge_excludes_momentum():
    """ret_10d > 300% 종목만 모멘텀 레인에서 제외된다 (P0-B 로직)."""
    df = pd.DataFrame({
        "ret_10d_%": [1582.0, 800.0, 250.0, 50.0],
        "MOMENTUM_LANE": [1, 1, 1, 1],
    })
    r10 = pd.to_numeric(df["ret_10d_%"], errors="coerce").fillna(0.0)
    abn = r10 > 300.0
    df["ABNORMAL_SURGE_FLAG"] = abn
    df.loc[abn, "MOMENTUM_LANE"] = 0
    assert df.loc[0, "MOMENTUM_LANE"] == 0  # 1582% 제외
    assert df.loc[1, "MOMENTUM_LANE"] == 0  # 800% 제외
    assert df.loc[2, "MOMENTUM_LANE"] == 1  # 250% 유지
    assert df.loc[3, "MOMENTUM_LANE"] == 1  # 50% 유지
