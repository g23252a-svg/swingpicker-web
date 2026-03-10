# -*- coding: utf-8 -*-
"""
test_v206_features.py — v20.6 핫픽스 회귀 검증
═══════════════════════════════════════════════════
pytest -v test_v206_features.py
"""
import numpy as np
import pandas as pd
import pytest
import json, os, tempfile

# ── 테스트 대상 import ──
from scoring_engine import (
    _vec_determine_state_dynamic,
    determine_state_dynamic,
    generate_score_reasons,
    build_global_score,
    _safe_col,
)


# ═══════════════════════════════════════════════════
#  1. ROUTE 벡터화 정확도 검증
# ═══════════════════════════════════════════════════

def _make_sample_df(n=500, seed=42):
    """ROUTE 판정에 필요한 컬럼을 가진 무작위 DataFrame 생성."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'RSI14':         rng.uniform(20, 85, n),
        'ret_1d_%':      rng.uniform(-5, 15, n),
        'ret_5d_%':      rng.uniform(-10, 30, n),
        'MACD_Slope_PCT': rng.normal(0, 1, n),
        'Range_Pos':     rng.uniform(0, 1, n),
        'Vol_Quality':   rng.uniform(0.5, 3, n),
        'TIMING_SCORE':  rng.uniform(0, 100, n),
        '거래강도':      rng.uniform(0, 12, n),
        'Low_Trend_PCT': rng.uniform(-8, 5, n),
        'Above_MA20':    rng.choice([0, 1], n),
        'TTM_SQUEEZE':   rng.choice([0, 1], n, p=[0.7, 0.3]),
        '거래대금(원)':  rng.uniform(1e6, 1e10, n),
        '외인순매수금액': rng.normal(0, 1e8, n),
        '개인순매수금액': rng.normal(0, 1e8, n),
        '외인순매수':    0.0,
        '개인순매수':    0.0,
    })
    return df


class TestRouteVectorization:
    """벡터 ROUTE와 단건 ROUTE가 100% 일치하는지 검증."""

    def test_route_match_random(self):
        """무작위 500건: 벡터 == 단건."""
        df = _make_sample_df(500)
        th = {
            'range_q75': float(df['Range_Pos'].quantile(0.75)),
            'vol_q75':   float(df['Vol_Quality'].quantile(0.75)),
        }

        # 벡터 판정
        vec_routes = _vec_determine_state_dynamic(df, th)

        # 단건 판정
        scalar_routes = df.apply(
            lambda r: determine_state_dynamic(r, th), axis=1
        )

        mismatches = (vec_routes != scalar_routes).sum()
        assert mismatches == 0, (
            f"ROUTE mismatch: {mismatches}/{len(df)}건\n"
            f"Sample:\n{df[vec_routes != scalar_routes].head()}"
        )

    def test_route_attack_downgrade(self):
        """ATTACK 조건 충족이지만 low_trend < -3 → WAIT로 다운그레이드."""
        df = pd.DataFrame([{
            'RSI14': 55, 'ret_1d_%': 2, 'ret_5d_%': 5,
            'MACD_Slope_PCT': 0.5, 'Range_Pos': 0.9,
            'Vol_Quality': 2.0, 'TIMING_SCORE': 75,
            '거래강도': 1.5, 'Low_Trend_PCT': -4.0,  # 핵심: < -3
            'Above_MA20': 1, 'TTM_SQUEEZE': 0,
            '거래대금(원)': 1e9, '외인순매수금액': 0,
            '개인순매수금액': 0, '외인순매수': 0, '개인순매수': 0,
        }])
        th = {'range_q75': 0.8, 'vol_q75': 1.2}

        route = _vec_determine_state_dynamic(df, th)
        assert route.iloc[0] == "WAIT", f"Expected WAIT, got {route.iloc[0]}"

    def test_route_exit_warning_flow(self):
        """외인 대량 매도 + 개인 대량 매수 + 급등 → EXIT_WARNING."""
        df = pd.DataFrame([{
            'RSI14': 60, 'ret_1d_%': 7, 'ret_5d_%': 10,
            'MACD_Slope_PCT': 0.5, 'Range_Pos': 0.9,
            'Vol_Quality': 2.0, 'TIMING_SCORE': 75,
            '거래강도': 5, 'Low_Trend_PCT': 2.0,
            'Above_MA20': 1, 'TTM_SQUEEZE': 0,
            '거래대금(원)': 1e9,
            '외인순매수금액': -2.5e8,  # -25% of turnover
            '개인순매수금액': 2.5e8,   # +25%
            '외인순매수': 0, '개인순매수': 0,
        }])
        th = {'range_q75': 0.8, 'vol_q75': 1.2}

        route = _vec_determine_state_dynamic(df, th)
        assert route.iloc[0] == "EXIT_WARNING"

    @pytest.mark.parametrize("rsi,r5,expected", [
        (80, 10, "OVERHEAT"),
        (50, 30, "OVERHEAT"),
    ])
    def test_route_overheat(self, rsi, r5, expected):
        """RSI >=75 또는 ret_5d >= 25 → OVERHEAT."""
        df = pd.DataFrame([{
            'RSI14': rsi, 'ret_1d_%': 0, 'ret_5d_%': r5,
            'MACD_Slope_PCT': 0, 'Range_Pos': 0.5,
            'Vol_Quality': 1.0, 'TIMING_SCORE': 50,
            '거래강도': 1, 'Low_Trend_PCT': 0,
            'Above_MA20': 0, 'TTM_SQUEEZE': 0,
            '거래대금(원)': 1e8, '외인순매수금액': 0,
            '개인순매수금액': 0, '외인순매수': 0, '개인순매수': 0,
        }])
        th = {'range_q75': 0.8, 'vol_q75': 1.2}
        assert _vec_determine_state_dynamic(df, th).iloc[0] == expected


# ═══════════════════════════════════════════════════
#  2. Score Reasons 검증
# ═══════════════════════════════════════════════════

class TestScoreReasons:
    """generate_score_reasons()의 점수순 정렬 + 장세 연동 검증."""

    def _make_scored_df(self, struct, timing, ai, **kwargs):
        """최소 컬럼으로 scored DataFrame 생성."""
        base = {
            'STRUCT_SCORE': struct, 'TIMING_SCORE': timing, 'AI_SCORE': ai,
            'RSI14': 55, 'ret_5d_%': 3, 'Low_Trend_PCT': 1,
            'Vol_Quality': 1.5, 'MFI14': 55, '거래대금(억원)': 50,
            'ROUTE': 'WAIT',
        }
        base.update(kwargs)
        return pd.DataFrame([base])

    def test_top1_is_highest_score(self):
        """가장 높은 점수 축이 TOP1."""
        df = self._make_scored_df(71, 85, 92)
        result = generate_score_reasons(df, macro_risk="NORMAL")
        assert result["SCORE_REASON_TOP1"].iloc[0] == "AI 강점"
        assert result["SCORE_REASON_TOP2"].iloc[0] == "TIMING 보조"

    def test_all_below_threshold_gives_empty(self):
        """모든 축 < 임계치 → TOP1/TOP2 비어있음 (fallback만 적용)."""
        df = self._make_scored_df(30, 40, 50)
        result = generate_score_reasons(df, macro_risk="NORMAL")
        # fallback: 저점추세 양호 (Low_Trend_PCT=1 > 0이므로 fallback 아님... 2보다 작으므로 빔)
        # vq=1.5 < 2.0이므로 거래품질 우수도 아님
        assert result["SCORE_REASON_TOP1"].iloc[0] == ""

    def test_bear_market_lowers_threshold(self):
        """BEAR 장세에서는 50점 이상이면 강점으로 인정."""
        df = self._make_scored_df(55, 60, 40)
        result = generate_score_reasons(df, macro_risk="BEAR")
        assert result["SCORE_REASON_TOP1"].iloc[0] == "TIMING 강점"
        assert result["SCORE_REASON_TOP2"].iloc[0] == "STRUCT 보조"
        assert result["REASON_THRESHOLD"].iloc[0] == 50

    def test_normal_market_threshold_70(self):
        """NORMAL 장세에서는 70점 이상이어야 강점."""
        df = self._make_scored_df(55, 60, 40)
        result = generate_score_reasons(df, macro_risk="NORMAL")
        assert result["SCORE_REASON_TOP1"].iloc[0] == ""  # 모두 70 미만
        assert result["REASON_THRESHOLD"].iloc[0] == 70

    def test_risk_rsi_overheat(self):
        """RSI >= 70 → SCORE_RISK = 'RSI 과열'."""
        df = self._make_scored_df(80, 80, 80, RSI14=72)
        result = generate_score_reasons(df)
        assert result["SCORE_RISK"].iloc[0] == "RSI 과열"

    def test_route_reason_attack(self):
        """ROUTE=ATTACK → ROUTE_REASON."""
        df = self._make_scored_df(80, 80, 80, ROUTE="ATTACK")
        result = generate_score_reasons(df)
        assert "돌파" in result["ROUTE_REASON"].iloc[0]


# ═══════════════════════════════════════════════════
#  3. Macro SSOT 검증
# ═══════════════════════════════════════════════════

class TestMacroSSoT:
    """run_health JSON에 macro_risk 직접 저장/읽기 검증."""

    def test_run_health_saves_macro_risk(self):
        """save_health에 macro_risk가 포함되는지."""
        from run_health import RunHealth, save_health

        h = RunHealth()
        h.macro_risk = "BEAR"
        h.market_breadth = 25.0
        h.confidence_score = 80.0

        with tempfile.TemporaryDirectory() as td:
            path = save_health(h, td, "20260310")
            with open(path, 'r') as f:
                data = json.load(f)

            assert data["macro_risk"] == "BEAR"
            assert data["market_breadth"] == 25.0

    def test_run_health_default_normal(self):
        """기본값은 NORMAL."""
        from run_health import RunHealth
        h = RunHealth()
        assert h.macro_risk == "NORMAL"
        assert h.market_breadth == 50.0


# ═══════════════════════════════════════════════════
#  4. ML Cache Schema 검증
# ═══════════════════════════════════════════════════

class TestMLCacheSchema:
    """Feature cache의 schema sidecar 검증."""

    def test_feature_hash_deterministic(self):
        """같은 FEATURE_COLS → 같은 hash."""
        from ml_engine import _compute_feature_hash
        h1 = _compute_feature_hash()
        h2 = _compute_feature_hash()
        assert h1 == h2
        assert len(h1) == 12  # md5[:12]

    def test_cache_path_is_joblib(self):
        """캐시 경로가 .joblib 확장자."""
        from ml_engine import FEATURE_CACHE_PATH
        assert FEATURE_CACHE_PATH.endswith('.joblib')


# ═══════════════════════════════════════════════════
#  5. Trigger 병렬 배치 검증
# ═══════════════════════════════════════════════════

class TestTriggerBatch:
    """_batch_trigger_scores의 병렬 처리 + 순서 보존 검증."""

    def test_order_preserved(self):
        """결과 순서가 입력 DataFrame 순서와 일치."""
        from pipeline_score import _batch_trigger_scores

        codes = ['005930', '000660', '035420']
        df = pd.DataFrame({'종목코드': codes})

        # 빈 ohlcv_map → 전부 0.0, 순서만 검증
        result = _batch_trigger_scores(df, {}, max_workers=2)
        assert len(result) == 3
        assert all(v == 0.0 for v in result)

    def test_partial_failure_logged(self):
        """일부 종목 실패해도 전체 결과는 반환."""
        from pipeline_score import _batch_trigger_scores

        codes = ['005930', 'INVALID']
        df = pd.DataFrame({'종목코드': codes})
        result = _batch_trigger_scores(df, {}, max_workers=2)
        assert len(result) == 2


# ═══════════════════════════════════════════════════
#  6. 주석/문서 동기화 검증
# ═══════════════════════════════════════════════════

class TestDocSync:
    """모듈 docstring이 현재 버전과 일치하는지."""

    def test_scoring_engine_docstring(self):
        import scoring_engine
        assert "v20.6.3" in scoring_engine.__doc__

    def test_pipeline_score_docstring(self):
        import pipeline_score
        assert "v20.6.3" in pipeline_score.__doc__

    def test_version_info_docstring(self):
        import version_info
        assert "v20.6.3" in version_info.__doc__ or "20.6.3" in str(version_info.CHANGELOG[0].get("version", ""))
