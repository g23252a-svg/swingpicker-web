# -*- coding: utf-8 -*-
"""tests/test_data_integrity_v241.py — v24.1 P0-C 데이터 무결성 게이트 회귀 테스트.

고정하는 계약:
  1. audit_ohlcv_window: V1(OHLC 불변식)·V2(비양수)·V3(종가 점프) 탐지
  2. 상한가 30% 연속은 오탐하지 않는다 (jump_limit 45% > KRX 30%)
  3. 검사 창(window) 밖의 위반은 무시한다
  4. apply_data_integrity: 산출 컬럼 계약 + 모멘텀 레인 제외 + P0-B 보존
  5. 공식 산식 보존: 기본 설정에서 TOP_PICK/BUY_NOW_ELIGIBLE 무변경
  6. demote_official=True일 때만 무결성 실패 행의 BUY_NOW_ELIGIBLE=0
  7. enabled=False여도 ABNORMAL_SURGE_FLAG(P0-B)는 동작
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_integrity import (  # noqa: E402
    DATA_INTEGRITY_COLS,
    apply_data_integrity,
    audit_ohlcv_window,
    data_integrity_summary,
)


# ── 헬퍼 ─────────────────────────────────────────────────────────
def _mk_ohlcv(closes, vol: float = 10000.0) -> pd.DataFrame:
    """내부 정합(고가≥시·종, 저가≤시·종)이 보장된 합성 OHLCV."""
    c = pd.Series(closes, dtype="float64")
    return pd.DataFrame({
        "시가": c * 0.99,
        "고가": c * 1.02,
        "저가": c * 0.97,
        "종가": c,
        "거래량": vol,
    })


def _mk_df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ── 1. audit_ohlcv_window: 위반 탐지 ────────────────────────────
class TestAuditDetection:
    def test_clean_passes(self):
        ok, reason, n_bad = audit_ohlcv_window(_mk_ohlcv(np.linspace(100, 120, 30)))
        assert ok and n_bad == 0 and reason == ""

    def test_high_below_close_detected(self):
        """V1: 고가 < 종가."""
        df = _mk_ohlcv(np.linspace(100, 120, 30))
        df.loc[df.index[-1], "고가"] = 60.0
        ok, reason, n_bad = audit_ohlcv_window(df)
        assert not ok and n_bad >= 1 and "V1" in reason

    def test_low_above_open_detected(self):
        """V1: 저가 > 시가."""
        df = _mk_ohlcv(np.linspace(100, 120, 30))
        df.loc[df.index[-1], "저가"] = 999.0
        ok, reason, _ = audit_ohlcv_window(df)
        assert not ok and "V1" in reason

    def test_high_below_low_detected(self):
        """V1: 고가 < 저가 (역전)."""
        df = _mk_ohlcv(np.linspace(100, 120, 30))
        df.loc[df.index[-2], "고가"] = 90.0
        df.loc[df.index[-2], "저가"] = 95.0
        # 시·종도 90~95 사이로 맞춰 순수 '고가<저가'만 위반
        df.loc[df.index[-2], "시가"] = 92.0
        df.loc[df.index[-2], "종가"] = 93.0
        ok, reason, _ = audit_ohlcv_window(df)
        assert not ok and "V1" in reason

    def test_nonpositive_price_detected(self):
        """V2: 0원/음수 가격 — sanitize 누락분 2차 방어."""
        df = _mk_ohlcv(np.linspace(100, 120, 30))
        df.loc[df.index[-3], "저가"] = 0.0
        ok, reason, _ = audit_ohlcv_window(df)
        assert not ok and "V2" in reason

    def test_close_jump_detected(self):
        """V3: 종가 점프 — 에이프로젠형 수정주가 단절 시나리오."""
        closes = list(np.linspace(100, 110, 29)) + [110 * 17.0]  # +1600%대 점프
        ok, reason, _ = audit_ohlcv_window(_mk_ohlcv(closes))
        assert not ok and "V3" in reason

    def test_jump_downward_detected(self):
        """V3: 하방 점프(감자·병합 단절)도 동일하게 탐지."""
        closes = list(np.linspace(100, 110, 29)) + [110 * 0.3]  # -70%
        ok, reason, _ = audit_ohlcv_window(_mk_ohlcv(closes))
        assert not ok and "V3" in reason


# ── 2. audit_ohlcv_window: 오탐 방지 ────────────────────────────
class TestAuditNoFalsePositive:
    def test_limit_up_streak_not_flagged(self):
        """상한가 +30% 연속 3일은 합법 거래 → 절대 플래그 금지."""
        closes = [100.0]
        for _ in range(3):
            closes.append(closes[-1] * 1.30)
        closes = [90.0] * 26 + closes  # 앞쪽 평탄 구간
        # 90→100 구간도 +11%라 한도 내
        ok, reason, n_bad = audit_ohlcv_window(_mk_ohlcv(closes), jump_limit_pct=45.0)
        assert ok and n_bad == 0, f"상한가 연속 오탐: {reason}"

    def test_violation_outside_window_ignored(self):
        """검사 창(window=20) 밖의 위반은 무시한다."""
        closes = list(np.linspace(100, 110, 40))
        df = _mk_ohlcv(closes)
        df.loc[df.index[5], "고가"] = 1.0  # 창 밖(40봉 중 6번째)
        ok, _, n_bad = audit_ohlcv_window(df, window=20)
        assert ok and n_bad == 0

    def test_max_bad_bars_tolerance(self):
        """max_bad_bars=1이면 위반 1봉은 허용."""
        df = _mk_ohlcv(np.linspace(100, 120, 30))
        df.loc[df.index[-1], "고가"] = 60.0
        ok0, _, _ = audit_ohlcv_window(df, max_bad_bars=0)
        ok1, _, _ = audit_ohlcv_window(df, max_bad_bars=1)
        assert (not ok0) and ok1

    def test_empty_and_none_skip(self):
        ok_n, reason_n, _ = audit_ohlcv_window(None)
        ok_e, reason_e, _ = audit_ohlcv_window(pd.DataFrame())
        assert ok_n and reason_n.startswith("SKIP")
        assert ok_e and reason_e.startswith("SKIP")

    def test_no_close_column_skip(self):
        df = pd.DataFrame({"거래량": [1, 2, 3]})
        ok, reason, _ = audit_ohlcv_window(df)
        assert ok and reason.startswith("SKIP")

    def test_english_columns_supported(self):
        """영문 OHLC 컬럼도 감사된다."""
        c = pd.Series(np.linspace(100, 120, 30))
        df = pd.DataFrame({"Open": c * 0.99, "High": c * 1.02, "Low": c * 0.97, "Close": c})
        df.loc[df.index[-1], "High"] = 50.0
        ok, reason, _ = audit_ohlcv_window(df)
        assert not ok and "V1" in reason


# ── 3. apply_data_integrity: 계약 + 통합 ────────────────────────
class TestApplyContract:
    def _base_df(self):
        return _mk_df([
            {"종목코드": "000001", "종목명": "정상주", "ret_10d_%": 12.0,
             "MOMENTUM_LANE": 1, "BUY_NOW_ELIGIBLE": 1, "TOP_PICK": 1},
            {"종목코드": "000002", "종목명": "폭등주", "ret_10d_%": 1582.0,
             "MOMENTUM_LANE": 1, "BUY_NOW_ELIGIBLE": 1, "TOP_PICK": 1},
            {"종목코드": "000003", "종목명": "왜곡주", "ret_10d_%": 20.0,
             "MOMENTUM_LANE": 1, "BUY_NOW_ELIGIBLE": 1, "TOP_PICK": 1},
        ])

    def _omap(self):
        clean = _mk_ohlcv(np.linspace(100, 110, 30))
        bad = _mk_ohlcv(np.linspace(100, 110, 30))
        bad.loc[bad.index[-1], "고가"] = 50.0
        return {"000001": clean, "000002": clean, "000003": bad}

    def test_output_columns_contract(self):
        out = apply_data_integrity(self._base_df(), ohlcv_map=self._omap())
        for col in DATA_INTEGRITY_COLS:
            assert col in out.columns, f"산출 컬럼 누락: {col}"

    def test_surge_flag_and_momentum_exclusion(self):
        """[P0-B 보존] ret_10d>300% → 플래그 + 모멘텀 제외."""
        out = apply_data_integrity(self._base_df(), ohlcv_map=self._omap()).set_index("종목명")
        assert bool(out.loc["폭등주", "ABNORMAL_SURGE_FLAG"])
        assert int(out.loc["폭등주", "MOMENTUM_LANE"]) == 0
        assert not bool(out.loc["정상주", "ABNORMAL_SURGE_FLAG"])
        assert int(out.loc["정상주", "MOMENTUM_LANE"]) == 1

    def test_integrity_bad_excluded_from_momentum(self):
        out = apply_data_integrity(self._base_df(), ohlcv_map=self._omap()).set_index("종목명")
        assert not bool(out.loc["왜곡주", "DATA_INTEGRITY_OK"])
        assert "V1" in str(out.loc["왜곡주", "DATA_INTEGRITY_REASON"])
        assert int(out.loc["왜곡주", "MOMENTUM_LANE"]) == 0

    def test_official_signals_preserved_by_default(self):
        """기본 설정(demote_official=False): 공식 신호 무변경."""
        out = apply_data_integrity(self._base_df(), ohlcv_map=self._omap()).set_index("종목명")
        for name in ("정상주", "폭등주", "왜곡주"):
            assert int(out.loc[name, "TOP_PICK"]) == 1
            assert int(out.loc[name, "BUY_NOW_ELIGIBLE"]) == 1

    def test_demote_official_when_enabled(self):
        """demote_official=True: '무결성 실패' 행만 BUY_NOW 강등, TOP_PICK 보존."""
        from collector_config import CollectorConfig, DataIntegrityConfig
        cfg = CollectorConfig(data_integrity=DataIntegrityConfig(demote_official=True))
        out = apply_data_integrity(self._base_df(), ohlcv_map=self._omap(), config=cfg).set_index("종목명")
        assert int(out.loc["왜곡주", "BUY_NOW_ELIGIBLE"]) == 0
        assert int(out.loc["왜곡주", "TOP_PICK"]) == 1  # 진단용 보존
        assert int(out.loc["정상주", "BUY_NOW_ELIGIBLE"]) == 1
        # 폭등주는 '무결성'은 정상이므로 강등 대상 아님 (surge ≠ integrity)
        assert int(out.loc["폭등주", "BUY_NOW_ELIGIBLE"]) == 1

    def test_disabled_skips_audit_but_keeps_surge(self):
        """enabled=False: 감사 SKIP, 단 P0-B 폭등 보호는 유지."""
        from collector_config import CollectorConfig, DataIntegrityConfig
        cfg = CollectorConfig(data_integrity=DataIntegrityConfig(enabled=False))
        out = apply_data_integrity(self._base_df(), ohlcv_map=self._omap(), config=cfg).set_index("종목명")
        assert bool(out.loc["왜곡주", "DATA_INTEGRITY_OK"])  # 감사 안 함
        assert str(out.loc["왜곡주", "DATA_INTEGRITY_REASON"]).startswith("SKIP")
        assert bool(out.loc["폭등주", "ABNORMAL_SURGE_FLAG"])  # P0-B 유지
        assert int(out.loc["폭등주", "MOMENTUM_LANE"]) == 0

    def test_missing_ohlcv_is_harmless_skip(self):
        out = apply_data_integrity(self._base_df(), ohlcv_map={}).set_index("종목명")
        assert bool(out.loc["왜곡주", "DATA_INTEGRITY_OK"])
        assert str(out.loc["왜곡주", "DATA_INTEGRITY_REASON"]) == "SKIP:no_ohlcv"

    def test_pure_function_no_inplace(self):
        df = self._base_df()
        before = df.copy(deep=True)
        _ = apply_data_integrity(df, ohlcv_map=self._omap())
        pd.testing.assert_frame_equal(df, before)

    def test_empty_df_contract(self):
        out = apply_data_integrity(pd.DataFrame())
        for col in DATA_INTEGRITY_COLS:
            assert col in out.columns
        assert out.empty

    def test_summary_keys(self):
        out = apply_data_integrity(self._base_df(), ohlcv_map=self._omap())
        s = data_integrity_summary(out)
        assert s["n_integrity_bad"] == 1
        assert s["n_surge"] == 1
        assert s["n_momentum_excluded"] == 2
        assert s["n_audited"] == 3


# ── 4. SSOT 검증 ────────────────────────────────────────────────
class TestConfigSSOT:
    def test_default_config_wired(self):
        from collector_config import DEFAULT_CONFIG
        di = DEFAULT_CONFIG.data_integrity
        assert di.enabled is True
        assert di.demote_official is False
        assert di.surge_ret10_pct == 300.0  # v24 P0-B 임계값 보존

    def test_jump_limit_must_exceed_krx_band(self):
        from collector_config import DataIntegrityConfig
        with pytest.raises(ValueError):
            DataIntegrityConfig(jump_limit_pct=25.0)  # 상한가 오탐 위험 → 거부
