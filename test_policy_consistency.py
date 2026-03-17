# -*- coding: utf-8 -*-
"""
test_policy_consistency.py — 정책 일관성 + 경계값 + 축 상태 검증
═══════════════════════════════════════════════════════════════════
[v20.8] pytest -v test_policy_consistency.py

검증 범위:
  1. PolicyConfig SSOT — validation/stop_logic이 같은 source 참조
  2. 경계값 테스트 — 거래대금/RSI/갭 임계치 정확한 동작
  3. 축별 상태 메타 — run_health axis_status 정상 설정
  4. Feature Contract — ML schema 일관성
"""
import pytest
import pandas as pd
import numpy as np

from collector_config import DEFAULT_CONFIG, PolicyConfig


# ═══════════════════════════════════════════════════
#  1. PolicyConfig SSOT 검증
# ═══════════════════════════════════════════════════

class TestPolicySSOT:
    """validation.py와 stop_logic.py가 PolicyConfig를 참조하는지."""

    def test_policy_exists_in_config(self):
        """DEFAULT_CONFIG에 policy 필드 존재."""
        assert hasattr(DEFAULT_CONFIG, 'policy')
        assert isinstance(DEFAULT_CONFIG.policy, PolicyConfig)

    def test_policy_hash_deterministic(self):
        """같은 PolicyConfig → 같은 hash."""
        p = PolicyConfig()
        assert p.policy_hash() == p.policy_hash()
        assert len(p.policy_hash()) == 12

    def test_hard_block_uses_policy(self):
        """validation.py HARD_BLOCK_RULES가 PolicyConfig 값과 일치."""
        from validation import HARD_BLOCK_RULES
        p = DEFAULT_CONFIG.policy

        # 거래대금 규칙
        turnover_rules = [r for r in HARD_BLOCK_RULES if '거래대금' in r.name]
        for rule in turnover_rules:
            assert rule.threshold == p.hard_block_turnover_min_eok, \
                f"{rule.name}: {rule.threshold} != policy {p.hard_block_turnover_min_eok}"

        # RSI 규칙
        rsi_rules = [r for r in HARD_BLOCK_RULES if r.check == 'RSI14']
        for rule in rsi_rules:
            assert rule.threshold == p.hard_block_rsi_max

        # 급등 규칙
        surge_rules = [r for r in HARD_BLOCK_RULES if r.name == '연속급등']
        for rule in surge_rules:
            assert rule.threshold == p.hard_block_ret5d_max

    def test_entry_defense_uses_policy(self):
        """stop_logic.py ENTRY_DEFENSE_RULES가 PolicyConfig 값 참조."""
        from stop_logic import ENTRY_DEFENSE_RULES
        p = DEFAULT_CONFIG.policy

        # gap hold 규칙 검증
        gap_hold = [r for r in ENTRY_DEFENSE_RULES if r['name'] == 'gap_hold']
        assert len(gap_hold) == 1
        # 12.01% gap → hold 트리거
        test_row = {"gap_pct": p.entry_gap_hold_pct + 0.01}
        assert gap_hold[0]['condition'](test_row) is True
        # 12.0% gap → hold 아님 (> 조건)
        test_row2 = {"gap_pct": p.entry_gap_hold_pct}
        assert gap_hold[0]['condition'](test_row2) is False

        # 유동성 규칙 검증
        liq_rule = [r for r in ENTRY_DEFENSE_RULES if r['name'] == 'very_low_liquidity']
        assert len(liq_rule) == 1
        test_row3 = {"거래대금(억원)": p.entry_turnover_hold_eok - 0.01}
        assert liq_rule[0]['condition'](test_row3) is True


# ═══════════════════════════════════════════════════
#  2. 경계값 테스트 — Hard Block
# ═══════════════════════════════════════════════════

class TestHardBlockBoundary:
    """Hard Block 임계치 경계값 동작 검증."""

    def _make_df(self, **overrides):
        """기본 통과 종목 + override."""
        base = {
            "종목코드": "005930", "종목명": "테스트",
            "ret_5d_%": 5.0, "거래대금(억원)": 100.0, "거래대금(억)": 100.0,
            "gap_pct": 3.0, "RSI14": 55.0, "_data_length": 200,
            "consecutive_limit_up": 0,
        }
        base.update(overrides)
        return pd.DataFrame([base])

    def test_turnover_boundary(self):
        """거래대금: 29.99→차단, 30.0→통과, 30.01→통과."""
        from validation import apply_hard_blocks
        p = DEFAULT_CONFIG.policy

        # 미만 → 차단
        _, blocked = apply_hard_blocks(self._make_df(**{"거래대금(억원)": p.hard_block_turnover_min_eok - 0.01}))
        assert len(blocked) == 1

        # 정확히 → 통과 (lt 연산: < 30.0이면 차단, 30.0은 통과)
        passed, _ = apply_hard_blocks(self._make_df(**{"거래대금(억원)": p.hard_block_turnover_min_eok}))
        assert len(passed) == 1

        # 초과 → 통과
        passed, _ = apply_hard_blocks(self._make_df(**{"거래대금(억원)": p.hard_block_turnover_min_eok + 0.01}))
        assert len(passed) == 1

    def test_rsi_boundary(self):
        """RSI: 84.99→통과, 85.0→통과(gt), 85.01→차단."""
        from validation import apply_hard_blocks
        p = DEFAULT_CONFIG.policy

        passed, _ = apply_hard_blocks(self._make_df(RSI14=p.hard_block_rsi_max - 0.01))
        assert len(passed) == 1

        # 정확히 85.0 → 통과 (gt: > 85.0)
        passed, _ = apply_hard_blocks(self._make_df(RSI14=p.hard_block_rsi_max))
        assert len(passed) == 1

        # 85.01 → 차단
        _, blocked = apply_hard_blocks(self._make_df(RSI14=p.hard_block_rsi_max + 0.01))
        assert len(blocked) == 1

    def test_ret5d_upper_boundary(self):
        """5일 수익률: 39.99→통과, 40.01→차단."""
        from validation import apply_hard_blocks
        p = DEFAULT_CONFIG.policy

        passed, _ = apply_hard_blocks(self._make_df(**{"ret_5d_%": p.hard_block_ret5d_max - 0.01}))
        assert len(passed) == 1

        _, blocked = apply_hard_blocks(self._make_df(**{"ret_5d_%": p.hard_block_ret5d_max + 0.01}))
        assert len(blocked) == 1

    def test_ret5d_lower_boundary(self):
        """5일 급락: -24.99→통과, -25.01→차단."""
        from validation import apply_hard_blocks
        p = DEFAULT_CONFIG.policy

        passed, _ = apply_hard_blocks(self._make_df(**{"ret_5d_%": p.hard_block_ret5d_min + 0.01}))
        assert len(passed) == 1

        _, blocked = apply_hard_blocks(self._make_df(**{"ret_5d_%": p.hard_block_ret5d_min - 0.01}))
        assert len(blocked) == 1

    def test_gap_boundary(self):
        """갭: 14.99→통과, 15.01→차단."""
        from validation import apply_hard_blocks
        p = DEFAULT_CONFIG.policy

        passed, _ = apply_hard_blocks(self._make_df(gap_pct=p.hard_block_gap_max - 0.01))
        assert len(passed) == 1

        _, blocked = apply_hard_blocks(self._make_df(gap_pct=p.hard_block_gap_max + 0.01))
        assert len(blocked) == 1


# ═══════════════════════════════════════════════════
#  3. 경계값 테스트 — Entry Defense
# ═══════════════════════════════════════════════════

class TestEntryDefenseBoundary:
    """Entry Defense 임계치 경계값 동작 검증."""

    def test_gap_hold_vs_split_boundary(self):
        """갭 7.01→split, 12.01→hold."""
        from stop_logic import check_entry_defense
        p = DEFAULT_CONFIG.policy

        # split 구간
        result = check_entry_defense({"gap_pct": p.entry_gap_split_pct + 0.01})
        assert result["action"] == "split"

        # hold 구간
        result = check_entry_defense({"gap_pct": p.entry_gap_hold_pct + 0.01})
        assert result["action"] == "hold"

        # 정상 구간
        result = check_entry_defense({"gap_pct": p.entry_gap_split_pct - 0.01})
        assert result["action"] == "enter"

    def test_surge_hold_vs_split_boundary(self):
        """급등 10.01→split, 15.01→hold."""
        from stop_logic import check_entry_defense
        p = DEFAULT_CONFIG.policy

        result = check_entry_defense({"ret_1d_%": p.entry_surge_split_pct + 0.01})
        assert result["action"] == "split"

        result = check_entry_defense({"ret_1d_%": p.entry_surge_hold_pct + 0.01})
        assert result["action"] == "hold"

    def test_liquidity_hold(self):
        """거래대금 49.99억→hold, 50.0억→enter."""
        from stop_logic import check_entry_defense
        p = DEFAULT_CONFIG.policy

        result = check_entry_defense({"거래대금(억원)": p.entry_turnover_hold_eok - 0.01})
        assert result["action"] == "hold"

        result = check_entry_defense({"거래대금(억원)": p.entry_turnover_hold_eok + 0.01})
        assert result["action"] == "enter"

    def test_rsi_split(self):
        """RSI 80.01→split, 79.99→enter."""
        from stop_logic import check_entry_defense
        p = DEFAULT_CONFIG.policy

        result = check_entry_defense({"RSI14": p.entry_rsi_split + 0.01})
        assert result["action"] == "split"

        result = check_entry_defense({"RSI14": p.entry_rsi_split - 0.01})
        assert result["action"] == "enter"


# ═══════════════════════════════════════════════════
#  4. 축별 상태 메타 검증
# ═══════════════════════════════════════════════════

class TestAxisStatus:
    """run_health 축별 상태가 올바르게 설정되는지."""

    def _make_health_df(self, **col_overrides):
        """최소 DataFrame."""
        base = {"종목코드": "005930", "FINAL_SCORE": 70}
        base.update(col_overrides)
        return pd.DataFrame([base])

    def test_all_ok(self):
        """모든 데이터 정상 → 전 축 OK."""
        from run_health import check_run_health
        df = self._make_health_df(
            **{"시가총액(억원)": 5000, "rel_60d_%": 10,
               "NEWS_SCORE": 5.0, "SECTOR_RANK": 3,
               "AI_SCORE": 70, "TIMING_SCORE": 60}
        )
        h = check_run_health(df,
                             mcap_map={"005930": 5000},
                             bench_map={"KOSPI": {60: 5.0}},
                             inv_maps={"frg": {"005930": 100}, "inst": {}, "ant": {}})

        assert h.axis_status["MCAP"] == "OK"
        assert h.axis_status["BENCH"] == "OK"
        assert h.axis_status["FLOW"] == "OK"
        assert h.axis_status["ML"] == "OK"
        assert h.fallback_count == 0

    def test_flow_partial(self):
        """외인/기관 없고 개인만 → FLOW=PARTIAL."""
        from run_health import check_run_health
        df = self._make_health_df(**{"시가총액(억원)": 5000})
        h = check_run_health(df,
                             mcap_map={"005930": 5000},
                             bench_map={"KOSPI": {60: 5.0}},
                             inv_maps={"frg": {}, "inst": {}, "ant": {"005930": 50}})
        assert h.axis_status["FLOW"] == "PARTIAL"

    def test_ml_disabled(self):
        """AI_SCORE 전부 0 → ML=DISABLED."""
        from run_health import check_run_health
        df = self._make_health_df(AI_SCORE=0.0)
        h = check_run_health(df, mcap_map={"005930": 5000})
        assert h.axis_status["ML"] == "DISABLED"

    def test_inject_axis_columns(self):
        """inject_columns이 AXIS_* 컬럼을 추가하는지."""
        from run_health import RunHealth
        h = RunHealth()
        h.set_axis("MCAP", "OK")
        h.set_axis("ML", "DISABLED")

        df = pd.DataFrame({"종목코드": ["005930"]})
        df = h.inject_columns(df)
        assert "AXIS_MCAP" in df.columns
        assert df["AXIS_MCAP"].iloc[0] == "OK"
        assert df["AXIS_ML"].iloc[0] == "DISABLED"
        assert "FALLBACK_COUNT" in df.columns

    def test_auto_degrade_on_3_failures(self):
        """3개 축 이상 실패 시 자동 DEGRADED."""
        from run_health import check_run_health
        df = self._make_health_df()  # 최소 데이터 — 대부분 축 실패
        h = check_run_health(df, mcap_map=None, bench_map=None, inv_maps=None)
        # MCAP, BENCH, FLOW 3개 이상 실패
        assert h.status in ("DEGRADED", "CRITICAL")


# ═══════════════════════════════════════════════════
#  5. Feature Contract 검증
# ═══════════════════════════════════════════════════

class TestFeatureContract:
    """ML feature schema 일관성."""

    def test_contract_matches_ml_engine(self):
        """feature_contract.py와 ml_engine.py FEATURE_COLS 동일."""
        from feature_contract import FEATURE_CONTRACT
        try:
            from ml_engine import FEATURE_COLS
            assert list(FEATURE_CONTRACT.columns) == FEATURE_COLS
        except ImportError:
            pytest.skip("ml_engine import requires torch")

    def test_schema_hash_deterministic(self):
        from feature_contract import FEATURE_CONTRACT
        h1 = FEATURE_CONTRACT.schema_hash
        h2 = FEATURE_CONTRACT.schema_hash
        assert h1 == h2
        assert len(h1) == 12

    def test_validate_correct_columns(self):
        from feature_contract import FEATURE_CONTRACT, validate_features
        df = pd.DataFrame(columns=list(FEATURE_CONTRACT.columns))
        ok, errors = validate_features(df, "test")
        assert ok is True
        assert errors == []

    def test_validate_missing_column(self):
        from feature_contract import FEATURE_CONTRACT, validate_features
        cols = list(FEATURE_CONTRACT.columns)[:-1]  # 마지막 하나 빠짐
        df = pd.DataFrame(columns=cols)
        ok, errors = validate_features(df, "test")
        assert ok is False
        assert len(errors) >= 1

    def test_n_features(self):
        from feature_contract import FEATURE_CONTRACT
        assert FEATURE_CONTRACT.n_features == 16


# ═══════════════════════════════════════════════════
#  6. 정책 버전 추적
# ═══════════════════════════════════════════════════

class TestPolicyVersion:
    """정책 변경 시 version/hash가 바뀌는지."""

    def test_different_policy_different_hash(self):
        p1 = PolicyConfig()
        p2 = PolicyConfig(hard_block_turnover_min_eok=50.0)
        assert p1.policy_hash() != p2.policy_hash()

    def test_all_fields_affect_hash(self):
        """모든 필드 변경이 hash에 반영되는지."""
        p_base = PolicyConfig()
        # entry_rsi_split만 바꿔도 hash 달라짐
        p_rsi = PolicyConfig(entry_rsi_split=75.0)
        assert p_base.policy_hash() != p_rsi.policy_hash()
        # entry_surge_split_pct만 바꿔도 hash 달라짐
        p_surge = PolicyConfig(entry_surge_split_pct=8.0)
        assert p_base.policy_hash() != p_surge.policy_hash()

    def test_policy_in_snapshot(self):
        """snapshot()에 정책 임계치 포함."""
        snap = DEFAULT_CONFIG.snapshot()
        assert "hard_block_turnover_min_eok" in snap
        assert "entry_gap_hold_pct" in snap
        assert "policy_version" in snap


# ═══════════════════════════════════════════════════
#  7. trade_plan → PolicyConfig end-to-end
# ═══════════════════════════════════════════════════

class TestTradePlanSSoT:
    """trade_plan이 PolicyConfig 기반 check_entry_defense를 사용하는지."""

    def test_trade_plan_gap_hold(self):
        """갭 12%+ → hold 판정이 trade_plan에서도 동작."""
        from trade_plan import build_trade_plan
        p = DEFAULT_CONFIG.policy
        plan = build_trade_plan(
            buy=10000, atr_val=300, last_c=10000,
            gap_pct=p.entry_gap_hold_pct + 1,  # 13%
            ret_1d=0,
        )
        assert plan.entry_action == "hold"
        assert plan.position_pct == 0.0

    def test_trade_plan_gap_split(self):
        """갭 7~12% → split."""
        from trade_plan import build_trade_plan
        p = DEFAULT_CONFIG.policy
        plan = build_trade_plan(
            buy=10000, atr_val=300, last_c=10000,
            gap_pct=p.entry_gap_split_pct + 1,  # 8%
            ret_1d=0,
        )
        assert plan.entry_action == "split"

    def test_trade_plan_normal_enter(self):
        """정상 조건 → enter."""
        from trade_plan import build_trade_plan
        plan = build_trade_plan(
            buy=10000, atr_val=300, last_c=10000,
            gap_pct=2, ret_1d=1,
            tv_eok=100,  # 유동성 충분
        )
        assert plan.entry_action == "enter"

    def test_trade_plan_low_liquidity_hold(self):
        """거래대금 50억 미만 → hold."""
        from trade_plan import build_trade_plan
        p = DEFAULT_CONFIG.policy
        plan = build_trade_plan(
            buy=10000, atr_val=300, last_c=10000,
            gap_pct=2, ret_1d=1,
            tv_eok=p.entry_turnover_hold_eok - 1,  # 49억
        )
        assert plan.entry_action == "hold"

    def test_trade_plan_rsi_split(self):
        """RSI 80+ → split."""
        from trade_plan import build_trade_plan
        p = DEFAULT_CONFIG.policy
        plan = build_trade_plan(
            buy=10000, atr_val=300, last_c=10000,
            gap_pct=2, ret_1d=1,
            tv_eok=100,
            rsi14=p.entry_rsi_split + 1,  # 81
        )
        assert plan.entry_action == "split"


# ═══════════════════════════════════════════════════
#  8. ML_STATUS 검증
# ═══════════════════════════════════════════════════

class TestMLStatus:
    """ML 추론 경로별 ML_STATUS 기록 + run_health 연결 검증."""

    def test_feature_contract_n_features(self):
        """Feature Contract 16컬럼."""
        from feature_contract import FEATURE_CONTRACT
        assert FEATURE_CONTRACT.n_features == 16

    def test_feature_contract_schema_version(self):
        from feature_contract import FEATURE_CONTRACT
        assert FEATURE_CONTRACT.schema_version == "v20.8"

    def test_feature_contract_column_names(self):
        """Contract 컬럼이 ml_engine FEATURE_COLS와 일치."""
        from feature_contract import FEATURE_CONTRACT
        try:
            from ml_engine import FEATURE_COLS
            assert list(FEATURE_CONTRACT.columns) == FEATURE_COLS
        except ImportError:
            pytest.skip("ml_engine requires torch")

    def test_feature_contract_validate_order(self):
        """컬럼 순서 불일치 감지."""
        from feature_contract import FEATURE_CONTRACT
        # 순서를 바꾸면 실패해야 함
        cols_wrong_order = list(FEATURE_CONTRACT.columns)
        cols_wrong_order[0], cols_wrong_order[1] = cols_wrong_order[1], cols_wrong_order[0]
        ok, errs = FEATURE_CONTRACT.validate(cols_wrong_order, "test")
        assert not ok
        assert any("order" in e.lower() for e in errs)

    def test_ml_status_ok_in_health(self):
        """ML_STATUS=OK → run_health ML축 OK."""
        from run_health import check_run_health
        df = pd.DataFrame({
            "종목코드": ["005930"], "FINAL_SCORE": [70],
            "ML_STATUS": ["OK"], "시가총액(억원)": [5000],
        })
        h = check_run_health(df,
                             mcap_map={"005930": 5000},
                             bench_map={"KOSPI": {60: 5.0}},
                             inv_maps={"frg": {"005930": 100}, "inst": {}, "ant": {}})
        assert h.axis_status["ML"] == "OK"

    def test_ml_status_fail_in_health(self):
        """ML_STATUS=DIM_MISMATCH → run_health ML축 FAILED."""
        from run_health import check_run_health
        df = pd.DataFrame({
            "종목코드": ["005930"], "FINAL_SCORE": [70],
            "ML_STATUS": ["DIM_MISMATCH"], "시가총액(억원)": [5000],
        })
        h = check_run_health(df,
                             mcap_map={"005930": 5000},
                             bench_map={"KOSPI": {60: 5.0}},
                             inv_maps={"frg": {"005930": 100}, "inst": {}, "ant": {}})
        assert "FAILED" in h.axis_status["ML"]

    def test_ml_status_no_column_falls_back_to_ai_score(self):
        """ML_STATUS 없으면 AI_SCORE 기반 판정."""
        from run_health import check_run_health
        df = pd.DataFrame({
            "종목코드": ["005930"], "FINAL_SCORE": [70],
            "AI_SCORE": [0.0], "시가총액(억원)": [5000],
        })
        h = check_run_health(df, mcap_map={"005930": 5000})
        assert h.axis_status["ML"] == "DISABLED"
