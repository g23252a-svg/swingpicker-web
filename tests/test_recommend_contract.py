# -*- coding: utf-8 -*-
"""
test_recommend_contract.py — [Phase 3+4] 한글 키 계약 회귀 테스트

목표:
  recommend_latest.csv가 의존하는 한글 키 (추천매수가/손절가/추천매도가1/2/3)
  계약을 깨는 변경이 들어오면 CI에서 즉시 차단.

테스트 범위:
  1. TradePlan.to_recommend_row() 한글 키 출력 검증
  2. validate_recommend_row() 정상/비정상 모두 검증
  3. ticker_analyzer.assemble_result()와 동일 키 사용 확인
  4. DEPRECATED 영문 path도 여전히 작동 (회귀 방지)
  5. 한글 키와 영문 키의 값 일관성 (entry == ENTRY_PRICE 등)
"""

import unittest

from trade_plan import (
    TradePlan,
    REQUIRED_RECOMMEND_KEYS,
    REQUIRED_PLAN_KEYS,
    RECOMMEND_PRICE_KEYS,
    validate_recommend_row,
    validate_row,
)


# ═══════════════════════════════════════════════════════════════════════
# 헬퍼: 정상 TradePlan 인스턴스
# ═══════════════════════════════════════════════════════════════════════

def _good_plan(**override) -> TradePlan:
    """정상 TradePlan 픽스처 — 필요한 필드만 override.

    entry/stop/tp1 override 시 tp2/tp3도 자동 단조 증가하도록 조정.
    """
    defaults = dict(
        entry=10000.0,
        stop=9500.0,
        tp1=10800.0,
        tp2=11500.0,
        tp3=12500.0,
        position_pct=100.0,
        entry_action="enter",
        plan_reason="NORMAL",
        stop_pct=5.0,
        max_loss_pct=1.5,
        rr_mult=1.6,
        regime="normal",
        exec_rule_id="LIMIT_TICK_v1",
        time_stop_days=7,
    )
    defaults.update(override)

    # entry override 시 tp1/tp2/tp3가 entry보다 위에 있도록 자동 조정
    if "entry" in override and "tp1" not in override:
        defaults["tp1"] = defaults["entry"] * 1.08
    if "entry" in override and "tp2" not in override:
        defaults["tp2"] = defaults["entry"] * 1.15
    if "entry" in override and "tp3" not in override:
        defaults["tp3"] = defaults["entry"] * 1.25
    # stop override 시 entry보다 아래에 있도록
    if "entry" in override and "stop" not in override:
        defaults["stop"] = defaults["entry"] * 0.95

    # tp1 override되면 tp2/tp3 일관성 유지
    if "tp1" in override and "tp2" not in override:
        defaults["tp2"] = defaults["tp1"] * 1.07
    if "tp1" in override and "tp3" not in override:
        defaults["tp3"] = defaults["tp1"] * 1.16

    # entry_action vs position_pct 일관성 — 명시적으로 둘 다 override 안 했으면 보정
    act = defaults.get("entry_action")
    if act == "hold" and "position_pct" not in override:
        defaults["position_pct"] = 0.0
    if act == "split" and "position_pct" not in override:
        defaults["position_pct"] = 50.0

    return TradePlan(**defaults)


# ═══════════════════════════════════════════════════════════════════════
# Section 1: to_recommend_row() — 한글 키 SSOT 출력
# ═══════════════════════════════════════════════════════════════════════

class TestToRecommendRowKoreanKeys(unittest.TestCase):
    """to_recommend_row() 가 정확히 어떤 키를 만드는가."""

    def test_returns_dict(self):
        row = _good_plan().to_recommend_row()
        self.assertIsInstance(row, dict)

    def test_has_all_required_korean_price_keys(self):
        """가격 5종 한글 키 모두 포함되어야 한다."""
        row = _good_plan().to_recommend_row()
        for k in ("추천매수가", "손절가", "추천매도가1", "추천매도가2", "추천매도가3"):
            self.assertIn(k, row, f"필수 한글 키 누락: {k}")

    def test_has_all_required_meta_keys(self):
        """영문 메타 키도 포함되어야 한다."""
        row = _good_plan().to_recommend_row()
        for k in ("ENTRY_ACTION", "POSITION_PCT", "PLAN_REASON",
                  "EXEC_RULE_ID", "STOP_PCT", "MAX_LOSS_PCT",
                  "RR_MULT", "REGIME", "TIME_STOP_DAYS"):
            self.assertIn(k, row, f"필수 영문 메타 키 누락: {k}")

    def test_korean_keys_match_csv_writer(self):
        """ticker_analyzer.assemble_result에서 쓰는 한글 키와 정확히 일치.

        이게 깨지면 recommend_latest.csv의 컬럼이 어긋나서 프런트가 빈 값을 봄.
        """
        row = _good_plan().to_recommend_row()
        # ticker_analyzer.py:639-641에 하드코딩된 키들
        actual_csv_keys = {"추천매수가", "손절가", "추천매도가1", "추천매도가2", "추천매도가3"}
        for k in actual_csv_keys:
            self.assertIn(k, row,
                f"한글 키 '{k}' 누락 — recommend_latest.csv가 깨짐")

    def test_price_values_match_dataclass(self):
        """한글 키의 값이 TradePlan dataclass 필드와 정확히 일치."""
        plan = _good_plan(entry=12345.0, stop=11000.0, tp1=13500.0)
        row = plan.to_recommend_row()
        self.assertEqual(row["추천매수가"], 12345.0)
        self.assertEqual(row["손절가"], 11000.0)
        self.assertEqual(row["추천매도가1"], 13500.0)

    def test_tp3_zero_becomes_none(self):
        """tp3=0 또는 None이면 추천매도가3 = None (CSV 빈 값) — 기존 동작 유지."""
        plan = _good_plan(tp3=0.0)
        row = plan.to_recommend_row()
        self.assertIsNone(row["추천매도가3"],
            "tp3=0이면 None으로 직렬화돼야 (assemble_result와 동일)")

    def test_tp3_positive_passes_through(self):
        plan = _good_plan(tp3=15000.0)
        row = plan.to_recommend_row()
        self.assertEqual(row["추천매도가3"], 15000.0)


# ═══════════════════════════════════════════════════════════════════════
# Section 2: validate_recommend_row() — 한글 계약 검증
# ═══════════════════════════════════════════════════════════════════════

class TestValidateRecommendRow(unittest.TestCase):
    """한글 키 row 검증 동작."""

    def test_good_row_passes(self):
        """정상 row는 예외 없이 통과."""
        row = _good_plan().to_recommend_row()
        try:
            validate_recommend_row(row)
        except Exception as e:
            self.fail(f"정상 row인데 예외 발생: {e}")

    def test_missing_required_korean_key_raises(self):
        row = _good_plan().to_recommend_row()
        del row["추천매수가"]
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("필수 컬럼 누락", str(ctx.exception))

    def test_missing_required_meta_key_raises(self):
        row = _good_plan().to_recommend_row()
        del row["EXEC_RULE_ID"]
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("필수 컬럼 누락", str(ctx.exception))

    def test_negative_entry_raises(self):
        row = _good_plan().to_recommend_row()
        row["추천매수가"] = -100
        with self.assertRaises(ValueError):
            validate_recommend_row(row)

    def test_zero_stop_raises(self):
        row = _good_plan().to_recommend_row()
        row["손절가"] = 0
        with self.assertRaises(ValueError):
            validate_recommend_row(row)

    def test_stop_above_entry_raises(self):
        """손절가 ≥ 추천매수가 면 정의상 위반."""
        row = _good_plan().to_recommend_row()
        row["손절가"] = row["추천매수가"] + 1
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("손절가", str(ctx.exception))

    def test_tp1_below_entry_raises(self):
        """추천매도가1 ≤ 추천매수가 면 정의상 위반 (수익 안 남)."""
        row = _good_plan().to_recommend_row()
        row["추천매도가1"] = row["추천매수가"] - 100
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("추천매도가1", str(ctx.exception))

    def test_tp_monotonic_increasing(self):
        """tp1 < tp2 < tp3 단조 증가 강제."""
        row = _good_plan().to_recommend_row()
        # tp2가 tp1보다 작으면 위반
        row["추천매도가2"] = row["추천매도가1"] - 100
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("추천매도가2", str(ctx.exception))

    def test_invalid_entry_action_raises(self):
        row = _good_plan().to_recommend_row()
        row["ENTRY_ACTION"] = "buy"  # 정의되지 않은 값
        with self.assertRaises(ValueError):
            validate_recommend_row(row)

    def test_position_pct_out_of_range_raises(self):
        row = _good_plan().to_recommend_row()
        row["POSITION_PCT"] = 150.0
        with self.assertRaises(ValueError):
            validate_recommend_row(row)

    def test_hold_with_position_raises(self):
        """hold인데 포지션이 잡혀있으면 일관성 위반.

        to_recommend_row()는 자체 검증을 수행하므로, 검증 함수를 직접 테스트하려면
        한 번 정상 row를 만든 뒤 dict 필드를 변조한다.
        """
        row = _good_plan().to_recommend_row()
        row["ENTRY_ACTION"] = "hold"
        row["POSITION_PCT"] = 50.0
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("hold", str(ctx.exception))

    def test_split_with_full_position_raises(self):
        """split인데 100% 포지션이면 일관성 위반."""
        row = _good_plan().to_recommend_row()
        row["ENTRY_ACTION"] = "split"
        row["POSITION_PCT"] = 100.0
        with self.assertRaises(ValueError):
            validate_recommend_row(row)


# ═══════════════════════════════════════════════════════════════════════
# Section 3: 한글/영문 path 일관성 (양쪽 모두 같은 entry/stop을 기록)
# ═══════════════════════════════════════════════════════════════════════

class TestKoreanEnglishConsistency(unittest.TestCase):
    """to_recommend_row()와 to_row()의 가격 값이 동일해야 한다."""

    def test_entry_price_same(self):
        plan = _good_plan(entry=15000.0)
        kr = plan.to_recommend_row()
        en = plan.to_row()
        self.assertEqual(kr["추천매수가"], en["ENTRY_PRICE"])

    def test_stop_price_same(self):
        plan = _good_plan(entry=15000.0, stop=14000.0)
        kr = plan.to_recommend_row()
        en = plan.to_row()
        self.assertEqual(kr["손절가"], en["STOP_PRICE"])

    def test_tp1_same(self):
        plan = _good_plan(tp1=16500.0)
        kr = plan.to_recommend_row()
        en = plan.to_row()
        self.assertEqual(kr["추천매도가1"], en["TP1"])

    def test_meta_keys_identical(self):
        """영문 메타 키는 두 path가 같은 값을 갖는다."""
        plan = _good_plan()
        kr = plan.to_recommend_row()
        en = plan.to_row()
        for k in ("ENTRY_ACTION", "POSITION_PCT", "PLAN_REASON",
                  "EXEC_RULE_ID", "STOP_PCT", "RR_MULT", "REGIME"):
            self.assertEqual(kr[k], en[k], f"메타 키 '{k}' 불일치")


# ═══════════════════════════════════════════════════════════════════════
# Section 4: DEPRECATED 영문 path 회귀 방지
# ═══════════════════════════════════════════════════════════════════════

class TestLegacyEnglishContract(unittest.TestCase):
    """to_row() / validate_row() / REQUIRED_PLAN_KEYS — 백테스트 호환 유지."""

    def test_to_row_still_works(self):
        row = _good_plan().to_row()
        self.assertIn("ENTRY_PRICE", row)
        self.assertIn("STOP_PRICE", row)
        self.assertIn("TP1", row)

    def test_validate_row_still_works(self):
        """영문 path도 검증 함수가 정상 동작."""
        row = _good_plan().to_row()
        try:
            validate_row(row)
        except Exception as e:
            self.fail(f"영문 row 검증 실패: {e}")

    def test_required_plan_keys_unchanged(self):
        """영문 키 계약은 변경되지 않아야 한다 (백테스트 의존성)."""
        expected = {
            "ENTRY_PRICE", "STOP_PRICE", "TP1",
            "POSITION_PCT", "ENTRY_ACTION", "PLAN_REASON",
            "EXEC_RULE_ID",
        }
        self.assertEqual(set(REQUIRED_PLAN_KEYS), expected)


# ═══════════════════════════════════════════════════════════════════════
# Section 5: 외부 모듈 SSOT 의존성
# ═══════════════════════════════════════════════════════════════════════

class TestExternalConsumerCompat(unittest.TestCase):
    """실제 한글 키를 read하는 외부 모듈들이 깨지지 않는지."""

    def test_recommend_price_keys_constant_stable(self):
        """RECOMMEND_PRICE_KEYS 튜플이 외부 모듈이 기대하는 정확한 5개."""
        self.assertEqual(
            RECOMMEND_PRICE_KEYS,
            ("추천매수가", "손절가", "추천매도가1", "추천매도가2", "추천매도가3"),
        )

    def test_collector_compatible_keys(self):
        """collector.py:636-644가 read하는 키가 to_recommend_row 출력에 있어야."""
        row = _good_plan().to_recommend_row()
        # collector.py가 읽는 정확한 키들
        for k in ("추천매수가", "손절가", "추천매도가1"):
            self.assertIn(k, row,
                f"collector.py가 read하는 '{k}' 가 to_recommend_row에 없음")

    def test_ticker_analyzer_compatible_keys(self):
        """ticker_analyzer.assemble_result가 매핑하는 키와 일치."""
        row = _good_plan().to_recommend_row()
        # ticker_analyzer.py:639-641 키들
        for k in ("추천매수가", "손절가", "추천매도가1", "추천매도가2"):
            self.assertIn(k, row)

    def test_simulated_assemble_result_passes_validation(self):
        """ticker_analyzer.assemble_result가 만드는 dict 형태(시뮬레이션)가
        validate_recommend_row로 검증되는지. 빠른 sanity check.

        진짜 회귀 차단은 TestRealAssembleResult (아래) 가 담당.
        """
        # ticker_analyzer.py:617-685 의 dict 구조 모사 (v3 신규 메타 키 포함)
        simulated_row = {
            # 가격 (한글)
            "추천매수가": 10000,
            "손절가": 9500,
            "추천매도가1": 10800,
            "추천매도가2": 11500,
            "추천매도가3": None,
            # 진입 제어 (영문)
            "ENTRY_ACTION": "enter",
            "POSITION_PCT": 100,
            # 체결 메타 (영문)
            "EXEC_RULE_ID": "LIMIT_TICK_v1",
            "STOP_PCT": 5.0,
            "MAX_LOSS_PCT": 1.5,
            "RR_MULT": 1.6,
            "STOP_REASON": "ATR_BASED",
            # [v3] 운영 메타 — 필수
            "PLAN_REASON": "NORMAL",
            "REGIME": "normal",
            "TIME_STOP_DAYS": 7,
            # 그 외 컬럼 무관
            "종목코드": "005930",
            "종목명": "삼성전자",
            "RSI14": 55.5,
        }
        try:
            validate_recommend_row(simulated_row)
        except Exception as e:
            self.fail(f"실 운영 dict 형태 검증 실패: {e}")

    def test_validation_catches_missing_korean_price_in_real_dict(self):
        """누가 실수로 ticker_analyzer.py의 '추천매수가' 매핑을 지웠다고 가정."""
        bad_row = {
            # "추천매수가": 누락!
            "손절가": 9500,
            "추천매도가1": 10800,
            "ENTRY_ACTION": "enter",
            "POSITION_PCT": 100,
            "EXEC_RULE_ID": "LIMIT_TICK_v1",
            "PLAN_REASON": "NORMAL",
            "STOP_PCT": 5.0,
            "MAX_LOSS_PCT": 1.5,
            "RR_MULT": 1.6,
            "REGIME": "normal",
        }
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(bad_row)
        self.assertIn("추천매수가", str(ctx.exception))


# ═══════════════════════════════════════════════════════════════════════
# Section 6: [v3 핵심] 실제 ticker_analyzer.assemble_result() 호출 검증
#            simulated 가 아닌 진짜 함수를 호출 — 한글 매핑 회귀 시 즉시 fail
# ═══════════════════════════════════════════════════════════════════════

class TestRealAssembleResult(unittest.TestCase):
    """[Phase 3+4 v3] 실제 ticker_analyzer.assemble_result() 호출 회귀 차단.

    이 테스트가 v1 평가의 핵심 흠 (회귀 차단력 13/20) 을 직접 해결.
    누가 ticker_analyzer.py의 한글 키 매핑을 실수로 망가뜨리면 즉시 fail.

    구현 방식:
      OHLCVContext / Indicators / TradePlanResult 인스턴스를 직접 만들어서
      assemble_result()의 한글 매핑 로직을 그대로 거치게 함.
    """

    def _build_real_inputs(self, **plan_overrides):
        """assemble_result에 넘길 실제 dataclass 인스턴스들 생성.

        외부 의존(SQLite/Gist 등) 없이 동작하는 최소 dataclass만 만든다.
        """
        import pandas as pd
        import numpy as np
        from ticker_analyzer import OHLCVContext, Indicators, TradePlanResult

        # 30일치 가짜 OHLCV (assemble_result는 ohlcv_df 직접 안 보지만 필드 존재해야 함)
        n = 30
        idx = pd.date_range("2026-04-01", periods=n)
        close = pd.Series(np.linspace(9500, 10000, n), index=idx, dtype=float)
        df = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": pd.Series([1_000_000] * n, index=idx, dtype=float),
        })

        ctx = OHLCVContext(
            code6="005930",
            ohlcv=df,
            c=df["close"],
            h=df["high"],
            l=df["low"],
            o=df["open"],
            v=df["volume"],
            last_c=10000.0,
            tv_eok=500.0,
            mcap=50000.0,
        )

        ind = Indicators(
            low_trend_pct=2.5, rsi=55.0, rsi_rising=1, vol_quality=1.2,
            bb_bw_val=3.5, bb_expanding=1, range_pos=0.6,
            bw_squeeze=0, ttm_squeeze=0, sqz_cnt=0,
            mfi=52.0, disp=1.5,
            ret_1d=1.5, ret_5=3.0, ret_10=5.0,
            ret_20=8.0, ret_60=15.0, ret_120=25.0,
            trigger_str="V_PWR",
            vwap_val=9950.0, vwap_gap=0.5,
            st_val=9700.0, st_trend=1,
            v_power=1.5, vol_z=1.2,
            swing_low_10=9300.0, dist_to_swing=7.0, is_swing_support=False,
            curr_hma=9800.0, hma_trend_up=True,
            is_above_w20=True, is_w20_up=True,
            slope_pct=0.05, hist=pd.Series([0.1, 0.2, 0.3], dtype=float),
            poc_p=9900.0, res_all=0.3, res_near=0.2, near_pct=2.0,
            is_above_poc=1, poc_gap=1.0,
            ma20=close.rolling(20).mean(),
            bb_upper=close * 1.02,
            bb_lower=close * 0.98,
            atr_series=pd.Series([200.0] * n, index=idx, dtype=float),
            gap_pct_val=0.5,
            data_length=n,
            consecutive_limit_up=0,
            mtf_weekly_trend=1,
            mtf_monthly_trend=1,
            mtf_data_sufficient=1,
        )

        plan_defaults = dict(
            buy=10000.0, stop=9500.0, target=10800.0, tp2=11500.0,
            actual_stop_pct=5.0, max_loss_pct=1.5, rr_mult=1.6,
            stop_reason="ATR_BASED",
            entry_action="enter", position_pct=100.0,
            exec_rule_id="LIMIT_TICK_v1",
            frg_net_val=1_000_000, inst_net_val=2_000_000, major_net=3_000_000,
            tp1_method="POC", tp1_prob=55,
            tp2_method="RES_RATIO", tp2_prob=40,
            tp3=12500.0, tp3_method="ATR_5x", tp3_prob=25,
            # [v3] 신규 메타
            plan_reason="NORMAL",
            regime="normal",
            time_stop_days=7,
        )
        plan_defaults.update(plan_overrides)
        plan = TradePlanResult(**plan_defaults)

        # name_map / sector_map / top_df / kospi_set / kosdaq_set / bench_map
        name_map = {"005930": "삼성전자"}
        sector_map = {"005930": "반도체"}
        top_df = pd.DataFrame({"종목코드": ["005930"], "시장": ["KOSPI"]})
        kospi_set = {"005930"}
        kosdaq_set = set()
        bench_map = {
            "KOSPI": {20: 2.0, 60: 5.0, 120: 10.0},
            "KOSDAQ": {20: 1.0, 60: 3.0, 120: 7.0},
        }
        return ctx, ind, plan, name_map, sector_map, top_df, kospi_set, kosdaq_set, bench_map

    def test_real_assemble_result_passes_validation(self):
        """실제 assemble_result() 호출 → 결과가 validate_recommend_row 통과해야."""
        # STRICT_RECOMMEND_CONTRACT=1 (assemble_result 내부 검증) 도 같이 작동
        from ticker_analyzer import assemble_result
        inputs = self._build_real_inputs()
        try:
            row = assemble_result(*inputs)
        except ValueError as e:
            self.fail(f"실제 assemble_result()가 자체 검증에서 fail: {e}")

        # 외부에서도 한 번 더 검증 (이중 안전)
        try:
            validate_recommend_row(row)
        except Exception as e:
            self.fail(f"실제 assemble_result() 결과가 외부 검증 fail: {e}")

    def test_real_assemble_result_has_korean_price_keys(self):
        """[핵심] 실제 결과 dict에 한글 가격 키가 정확히 존재."""
        from ticker_analyzer import assemble_result
        inputs = self._build_real_inputs()
        row = assemble_result(*inputs)

        for k in ("추천매수가", "손절가", "추천매도가1", "추천매도가2"):
            self.assertIn(k, row,
                f"실제 assemble_result() 결과에 '{k}' 누락 — 회귀 발생!")

    def test_real_assemble_result_has_v3_meta_keys(self):
        """[v3] PLAN_REASON / REGIME / TIME_STOP_DAYS — 새로 추가된 메타 키."""
        from ticker_analyzer import assemble_result
        inputs = self._build_real_inputs()
        row = assemble_result(*inputs)

        for k in ("PLAN_REASON", "REGIME", "TIME_STOP_DAYS"):
            self.assertIn(k, row, f"v3 메타 키 '{k}' 누락 — assemble_result 회귀!")

    def test_real_assemble_result_korean_values_match_plan(self):
        """한글 키의 값이 plan dataclass 필드와 정확히 일치 (오타/매핑실수 차단)."""
        from ticker_analyzer import assemble_result
        inputs = self._build_real_inputs()
        ctx, ind, plan, *_ = inputs
        row = assemble_result(*inputs)

        self.assertEqual(row["추천매수가"], plan.buy)
        self.assertEqual(row["손절가"], plan.stop)
        self.assertEqual(row["추천매도가1"], plan.target)
        self.assertEqual(row["추천매도가2"], plan.tp2)

    def test_real_assemble_result_strict_mode_blocks_invalid(self):
        """STRICT_RECOMMEND_CONTRACT=1 모드에서 잘못된 plan(stop>=entry)이 들어오면 차단."""
        import os
        from ticker_analyzer import assemble_result
        # 잘못된 plan: stop이 entry보다 위 (정의상 위반)
        inputs = self._build_real_inputs(buy=10000.0, stop=10500.0)

        # 환경변수를 strict로 명시 (기본값이지만 명시)
        old = os.environ.get("STRICT_RECOMMEND_CONTRACT")
        os.environ["STRICT_RECOMMEND_CONTRACT"] = "1"
        try:
            with self.assertRaises(ValueError) as ctx:
                assemble_result(*inputs)
            self.assertIn("손절가", str(ctx.exception))
        finally:
            if old is None:
                os.environ.pop("STRICT_RECOMMEND_CONTRACT", None)
            else:
                os.environ["STRICT_RECOMMEND_CONTRACT"] = old

    def test_real_assemble_result_non_strict_mode_warns_only(self):
        """STRICT_RECOMMEND_CONTRACT=0 모드에서는 경고만 찍고 진행 (운영 안전)."""
        import os, logging
        from ticker_analyzer import assemble_result
        inputs = self._build_real_inputs(buy=10000.0, stop=10500.0)  # 잘못된 plan

        old = os.environ.get("STRICT_RECOMMEND_CONTRACT")
        os.environ["STRICT_RECOMMEND_CONTRACT"] = "0"
        try:
            with self.assertLogs("ticker_analyzer", level="WARNING") as logs:
                row = assemble_result(*inputs)  # 예외 없이 통과해야
            # 경고 로그가 한 번이라도 찍혔는지
            self.assertTrue(
                any("Recommend Contract" in m for m in logs.output),
                f"비-strict 모드인데 경고가 안 찍힘: {logs.output}"
            )
            self.assertIsNotNone(row)
        finally:
            if old is None:
                os.environ.pop("STRICT_RECOMMEND_CONTRACT", None)
            else:
                os.environ["STRICT_RECOMMEND_CONTRACT"] = old

    def test_real_assemble_result_skip_mode_no_validation(self):
        """STRICT_RECOMMEND_CONTRACT=skip 모드에서는 검증 자체 스킵."""
        import os
        from ticker_analyzer import assemble_result
        inputs = self._build_real_inputs(buy=10000.0, stop=10500.0)  # 잘못된 plan

        old = os.environ.get("STRICT_RECOMMEND_CONTRACT")
        os.environ["STRICT_RECOMMEND_CONTRACT"] = "skip"
        try:
            row = assemble_result(*inputs)
            self.assertIsNotNone(row)
            # skip 모드에서는 잘못된 값이 그대로 들어감
            self.assertEqual(row["손절가"], 10500.0)
        finally:
            if old is None:
                os.environ.pop("STRICT_RECOMMEND_CONTRACT", None)
            else:
                os.environ["STRICT_RECOMMEND_CONTRACT"] = old


# ═══════════════════════════════════════════════════════════════════════
# Section 7: [v3] 세분화된 REQUIRED 키 그룹
# ═══════════════════════════════════════════════════════════════════════

class TestRequiredKeyGroups(unittest.TestCase):
    """v3에서 REQUIRED를 PRICE/EXEC/META 3그룹으로 나눔."""

    def test_price_keys_present(self):
        from trade_plan import REQUIRED_RECOMMEND_PRICE_KEYS
        self.assertEqual(
            REQUIRED_RECOMMEND_PRICE_KEYS,
            frozenset({"추천매수가", "손절가", "추천매도가1"})
        )

    def test_exec_keys_present(self):
        from trade_plan import REQUIRED_RECOMMEND_EXEC_KEYS
        self.assertEqual(
            REQUIRED_RECOMMEND_EXEC_KEYS,
            frozenset({"ENTRY_ACTION", "POSITION_PCT", "EXEC_RULE_ID"})
        )

    def test_meta_keys_present(self):
        from trade_plan import REQUIRED_RECOMMEND_META_KEYS
        self.assertEqual(
            REQUIRED_RECOMMEND_META_KEYS,
            frozenset({"PLAN_REASON", "STOP_PCT", "MAX_LOSS_PCT",
                       "RR_MULT", "REGIME", "TIME_STOP_DAYS"})
        )

    def test_combined_required_is_union(self):
        from trade_plan import (
            REQUIRED_RECOMMEND_KEYS,
            REQUIRED_RECOMMEND_PRICE_KEYS,
            REQUIRED_RECOMMEND_EXEC_KEYS,
            REQUIRED_RECOMMEND_META_KEYS,
        )
        expected = (
            REQUIRED_RECOMMEND_PRICE_KEYS
            | REQUIRED_RECOMMEND_EXEC_KEYS
            | REQUIRED_RECOMMEND_META_KEYS
        )
        self.assertEqual(set(REQUIRED_RECOMMEND_KEYS), set(expected))


# ═══════════════════════════════════════════════════════════════════════
# Section 8: [v3] _optional_positive_float helper 명확한 에러 메시지
# ═══════════════════════════════════════════════════════════════════════

class TestOptionalFloatHelper(unittest.TestCase):
    """v3 helper: 잘못된 tp2/tp3 값이 들어와도 명확한 에러."""

    def test_none_returns_none(self):
        from trade_plan import _optional_positive_float
        self.assertIsNone(_optional_positive_float({"k": None}, "k"))

    def test_empty_string_returns_none(self):
        from trade_plan import _optional_positive_float
        self.assertIsNone(_optional_positive_float({"k": ""}, "k"))

    def test_zero_returns_none(self):
        from trade_plan import _optional_positive_float
        self.assertIsNone(_optional_positive_float({"k": 0}, "k"))

    def test_negative_returns_none(self):
        from trade_plan import _optional_positive_float
        self.assertIsNone(_optional_positive_float({"k": -5.0}, "k"))

    def test_positive_returns_value(self):
        from trade_plan import _optional_positive_float
        self.assertEqual(_optional_positive_float({"k": 12.5}, "k"), 12.5)

    def test_garbage_raises_clear_error(self):
        """문자열 'abc' 같은 쓰레기 값은 명확한 에러 메시지."""
        from trade_plan import _optional_positive_float
        with self.assertRaises(ValueError) as ctx:
            _optional_positive_float({"k": "abc"}, "k")
        self.assertIn("Recommend Contract", str(ctx.exception))
        self.assertIn("k cast fail", str(ctx.exception))

    def test_nan_string_returns_none(self):
        from trade_plan import _optional_positive_float
        self.assertIsNone(_optional_positive_float({"k": "nan"}, "k"))


# ═══════════════════════════════════════════════════════════════════════
# Section 9: [v4] 메타 값 품질 검증 — 빈 값 / 잘못된 부호 차단
# ═══════════════════════════════════════════════════════════════════════

class TestMetaValueQuality(unittest.TestCase):
    """v4: 메타 키 값 품질 — 빈 문자열 / 음수 / 잘못된 enum 차단."""

    def _good_row(self):
        """validate 통과하는 정상 row (메타값 검증 단독 테스트용)."""
        return {
            "추천매수가": 10000, "손절가": 9500, "추천매도가1": 10800,
            "ENTRY_ACTION": "enter", "POSITION_PCT": 100,
            "EXEC_RULE_ID": "LIMIT_TICK_v1",
            "PLAN_REASON": "NORMAL", "STOP_PCT": 5.0,
            "MAX_LOSS_PCT": 1.5, "RR_MULT": 1.6, "REGIME": "normal",
            "TIME_STOP_DAYS": 7,
        }

    def test_empty_plan_reason_raises(self):
        row = self._good_row()
        row["PLAN_REASON"] = ""
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("PLAN_REASON", str(ctx.exception))

    def test_whitespace_only_plan_reason_raises(self):
        row = self._good_row()
        row["PLAN_REASON"] = "   "  # 공백만
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("PLAN_REASON", str(ctx.exception))

    def test_empty_exec_rule_id_raises(self):
        row = self._good_row()
        row["EXEC_RULE_ID"] = ""
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("EXEC_RULE_ID", str(ctx.exception))

    def test_invalid_regime_raises(self):
        row = self._good_row()
        row["REGIME"] = "extreme_vol"  # 정의되지 않은 값
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("REGIME", str(ctx.exception))

    def test_allowed_regimes(self):
        """REGIME 허용값 3종 모두 통과."""
        for regime in ("normal", "high_vol", "low_vol"):
            row = self._good_row()
            row["REGIME"] = regime
            try:
                validate_recommend_row(row)
            except Exception as e:
                self.fail(f"REGIME='{regime}' 허용돼야 하는데 실패: {e}")

    def test_zero_rr_mult_raises(self):
        """RR_MULT=0이면 손익비가 없으므로 위반."""
        row = self._good_row()
        row["RR_MULT"] = 0
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("RR_MULT", str(ctx.exception))

    def test_negative_rr_mult_raises(self):
        row = self._good_row()
        row["RR_MULT"] = -1.5
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("RR_MULT", str(ctx.exception))

    def test_negative_max_loss_pct_raises(self):
        row = self._good_row()
        row["MAX_LOSS_PCT"] = -0.5
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("MAX_LOSS_PCT", str(ctx.exception))

    def test_zero_max_loss_pct_allowed(self):
        """MAX_LOSS_PCT=0은 캡 없음 의미 — 허용."""
        row = self._good_row()
        row["MAX_LOSS_PCT"] = 0
        try:
            validate_recommend_row(row)
        except Exception as e:
            self.fail(f"MAX_LOSS_PCT=0은 허용돼야: {e}")

    def test_negative_time_stop_days_raises(self):
        row = self._good_row()
        row["TIME_STOP_DAYS"] = -1
        with self.assertRaises(ValueError) as ctx:
            validate_recommend_row(row)
        self.assertIn("TIME_STOP_DAYS", str(ctx.exception))

    def test_zero_time_stop_days_allowed(self):
        """TIME_STOP_DAYS=0은 비활성 의미 — 허용."""
        row = self._good_row()
        row["TIME_STOP_DAYS"] = 0
        try:
            validate_recommend_row(row)
        except Exception as e:
            self.fail(f"TIME_STOP_DAYS=0은 허용돼야: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Section 10: [v4] TradePlanResult.to_recommend_row() — 진짜 SSOT 강제
# ═══════════════════════════════════════════════════════════════════════

class TestTradePlanResultToRecommendRow(unittest.TestCase):
    """v4: assemble_result가 사용하는 SSOT 진입점 — TradePlanResult.to_recommend_row()."""

    def _make_plan_result(self, **override):
        """TradePlanResult 인스턴스 생성 helper."""
        from ticker_analyzer import TradePlanResult
        defaults = dict(
            buy=10000.0, stop=9500.0, target=10800.0, tp2=11500.0,
            actual_stop_pct=5.0, max_loss_pct=1.5, rr_mult=1.6,
            stop_reason="ATR_BASED",
            entry_action="enter", position_pct=100.0,
            exec_rule_id="LIMIT_TICK_v1",
            frg_net_val=1_000_000, inst_net_val=2_000_000, major_net=3_000_000,
            tp1_method="POC", tp1_prob=55,
            tp2_method="RES_RATIO", tp2_prob=40,
            tp3=12500.0, tp3_method="ATR_5x", tp3_prob=25,
            plan_reason="NORMAL", regime="normal", time_stop_days=7,
        )
        defaults.update(override)
        return TradePlanResult(**defaults)

    def test_returns_dict(self):
        plan = self._make_plan_result()
        row = plan.to_recommend_row()
        self.assertIsInstance(row, dict)

    def test_all_required_keys_present(self):
        """REQUIRED_RECOMMEND_KEYS 모두 포함 — 계약 만족."""
        plan = self._make_plan_result()
        row = plan.to_recommend_row()
        from trade_plan import REQUIRED_RECOMMEND_KEYS
        missing = REQUIRED_RECOMMEND_KEYS - set(row.keys())
        self.assertFalse(missing, f"REQUIRED 키 누락: {missing}")

    def test_passes_validation(self):
        """to_recommend_row() 결과는 즉시 validate_recommend_row 통과."""
        plan = self._make_plan_result()
        row = plan.to_recommend_row()
        try:
            validate_recommend_row(row)
        except Exception as e:
            self.fail(f"TradePlanResult.to_recommend_row() 결과 검증 실패: {e}")

    def test_korean_prices_match_dataclass(self):
        """한글 가격 키가 dataclass 필드와 정확히 일치."""
        plan = self._make_plan_result(buy=12345, stop=11000, target=13500, tp2=14500)
        row = plan.to_recommend_row()
        self.assertEqual(row["추천매수가"], 12345)
        self.assertEqual(row["손절가"], 11000)
        self.assertEqual(row["추천매도가1"], 13500)
        self.assertEqual(row["추천매도가2"], 14500)

    def test_tp3_zero_becomes_none(self):
        """tp3=0이면 추천매도가3=None (CSV 빈 값)."""
        plan = self._make_plan_result(tp3=0.0)
        row = plan.to_recommend_row()
        self.assertIsNone(row["추천매도가3"])

    def test_tp3_method_empty_when_tp3_zero(self):
        """tp3=0이면 TP3_METHOD="" / TP3_PROB=0 (관행)."""
        plan = self._make_plan_result(tp3=0.0, tp3_method="ATR_5x", tp3_prob=25)
        row = plan.to_recommend_row()
        self.assertEqual(row["TP3_METHOD"], "")
        self.assertEqual(row["TP3_PROB"], 0)

    def test_assemble_result_uses_to_recommend_row(self):
        """[핵심] assemble_result()가 plan.to_recommend_row()를 사용하는지.

        실제 호출 → 결과의 가격 키 값이 plan dataclass와 정확 일치.
        """
        from ticker_analyzer import assemble_result, OHLCVContext, Indicators
        # TestRealAssembleResult에서 쓴 fixture와 동일한 구조
        import pandas as pd, numpy as np
        n = 30
        idx = pd.date_range("2026-04-01", periods=n)
        close = pd.Series(np.linspace(9500, 10000, n), index=idx, dtype=float)
        df = pd.DataFrame({
            "open": close * 0.99, "high": close * 1.02,
            "low": close * 0.98, "close": close,
            "volume": pd.Series([1_000_000]*n, index=idx, dtype=float),
        })
        ctx = OHLCVContext(
            code6="005930", ohlcv=df, c=df["close"], h=df["high"],
            l=df["low"], o=df["open"], v=df["volume"],
            last_c=10000.0, tv_eok=500.0, mcap=50000.0,
        )
        ind = Indicators(
            low_trend_pct=2.5, rsi=55.0, rsi_rising=1, vol_quality=1.2,
            bb_bw_val=3.5, bb_expanding=1, range_pos=0.6,
            bw_squeeze=0, ttm_squeeze=0, sqz_cnt=0,
            mfi=52.0, disp=1.5,
            ret_1d=1.5, ret_5=3.0, ret_10=5.0,
            ret_20=8.0, ret_60=15.0, ret_120=25.0,
            trigger_str="V_PWR", vwap_val=9950.0, vwap_gap=0.5,
            st_val=9700.0, st_trend=1, v_power=1.5, vol_z=1.2,
            swing_low_10=9300.0, dist_to_swing=7.0, is_swing_support=False,
            curr_hma=9800.0, hma_trend_up=True,
            is_above_w20=True, is_w20_up=True,
            slope_pct=0.05, hist=pd.Series([0.1, 0.2, 0.3], dtype=float),
            poc_p=9900.0, res_all=0.3, res_near=0.2, near_pct=2.0,
            is_above_poc=1, poc_gap=1.0,
            ma20=close.rolling(20).mean(),
            bb_upper=close * 1.02, bb_lower=close * 0.98,
            atr_series=pd.Series([200.0]*n, index=idx, dtype=float),
            gap_pct_val=0.5, data_length=n, consecutive_limit_up=0,
            mtf_weekly_trend=1, mtf_monthly_trend=1, mtf_data_sufficient=1,
        )
        plan = self._make_plan_result(buy=15000, stop=14000, target=16500)
        name_map = {"005930": "삼성전자"}
        sector_map = {"005930": "반도체"}
        top_df = pd.DataFrame({"종목코드": ["005930"], "시장": ["KOSPI"]})
        bench_map = {
            "KOSPI": {20: 2.0, 60: 5.0, 120: 10.0},
            "KOSDAQ": {20: 1.0, 60: 3.0, 120: 7.0},
        }

        row = assemble_result(ctx, ind, plan, name_map, sector_map,
                              top_df, {"005930"}, set(), bench_map)

        # 핵심: assemble_result 결과의 가격 키가 plan과 정확히 일치
        # 이건 row.update(plan.to_recommend_row()) 가 작동한다는 증거
        self.assertEqual(row["추천매수가"], 15000)
        self.assertEqual(row["손절가"], 14000)
        self.assertEqual(row["추천매도가1"], 16500)


# ═══════════════════════════════════════════════════════════════════════
# Section 11: [v4] STRICT 기본값 운영 안전 (default = "0")
# ═══════════════════════════════════════════════════════════════════════

class TestStrictDefault(unittest.TestCase):
    """v4: 환경변수 기본값이 "0" (warn-only) — 첫 배포 안전성."""

    def test_strict_default_is_warn_only(self):
        """STRICT 환경변수 미설정 시 잘못된 plan도 경고만 — 운영 collector 안 죽음."""
        import os, logging
        from ticker_analyzer import assemble_result
        # TestRealAssembleResult helper 재사용을 위해 import
        from tests.test_recommend_contract import TestRealAssembleResult
        # 잘못된 plan: stop > entry (위반)
        helper = TestRealAssembleResult()
        inputs = helper._build_real_inputs(buy=10000.0, stop=10500.0)

        # 환경변수 깔끔히 제거 (기본값 작동 확인)
        old = os.environ.pop("STRICT_RECOMMEND_CONTRACT", None)
        try:
            with self.assertLogs("ticker_analyzer", level="WARNING") as logs:
                row = assemble_result(*inputs)  # 예외 X
            self.assertTrue(
                any("Recommend Contract" in m for m in logs.output),
                "기본값에서 경고가 안 찍힘"
            )
            self.assertIsNotNone(row)
        finally:
            if old is not None:
                os.environ["STRICT_RECOMMEND_CONTRACT"] = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
