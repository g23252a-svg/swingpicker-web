"""
test_ml_engine.py — ml_engine.py v17.1 핵심 방어 로직 단위테스트

실행 방법:
  pytest test_ml_engine.py -v          (pytest 설치 시)
  python test_ml_engine.py             (unittest 직접 실행)

테스트 항목:
  1. 모델 파일 없음 → ML_SCORE=0 (폭발 없이)
  2. 스케일러 파일만 없음 → ML_SCORE=0
  3. 피처 차원 불일치 → ML_SCORE=0 (안전모드)
  4. 피처 버전 메타 불일치 → 레거시 폴백 트리거
  5. 정상 추론 (mock 모델)
  6. add_technical_features 피처 수 검증
"""

import os, sys, json, tempfile, unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_engine


def _make_sample_df():
    """추론용 더미 데이터프레임"""
    return pd.DataFrame({
        "종목코드": ["005930", "000660", "035720"],
        "종목명": ["삼성전자", "SK하이닉스", "카카오"],
        "FINAL_SCORE": [75.0, 60.0, 50.0],
    })


def _make_ohlcv_map():
    """추론용 더미 OHLCV (100일치)"""
    dates = pd.bdate_range("2025-01-01", periods=100)
    result = {}
    for code in ["005930", "000660", "035720"]:
        np.random.seed(hash(code) % (2**31))
        base = 50000 + np.cumsum(np.random.randn(100) * 500)
        df = pd.DataFrame({
            "시가": base + np.random.randn(100) * 200,
            "고가": base + abs(np.random.randn(100) * 400),
            "저가": base - abs(np.random.randn(100) * 400),
            "종가": base,
            "거래량": np.random.randint(100000, 1000000, 100).astype(float),
        }, index=dates)
        result[code] = df
    return result


class _BaseTest(unittest.TestCase):
    """각 테스트 전/후 전역 캐시 초기화"""
    def setUp(self):
        ml_engine._loaded_lstm_model = None
        ml_engine._loaded_scaler = None
        ml_engine._loaded_xgb_model = None
        ml_engine._loaded_feature_cols = None

    def tearDown(self):
        ml_engine._loaded_lstm_model = None
        ml_engine._loaded_scaler = None
        ml_engine._loaded_xgb_model = None
        ml_engine._loaded_feature_cols = None


# ================================================================
# Test 1: 모델 파일 없음 → ML_SCORE=0
# ================================================================
class TestNoModelFile(_BaseTest):
    def test_no_model_returns_zero(self):
        """모델/스케일러 파일이 모두 없을 때 ML_SCORE=0, 에러 없음"""
        with patch.object(ml_engine, 'MODEL_PATH', '/nonexistent/model.pth'), \
             patch.object(ml_engine, 'LEGACY_MODEL_PATH', '/nonexistent/legacy.pth'), \
             patch.object(ml_engine, 'SCALER_PATH', '/nonexistent/scaler.pkl'), \
             patch.object(ml_engine, 'LEGACY_SCALER_PATH', '/nonexistent/legacy.pkl'):

            result = ml_engine.apply_ml_score(_make_sample_df(), _make_ohlcv_map())

        self.assertIn("ML_SCORE", result.columns)
        self.assertTrue((result["ML_SCORE"] == 0.0).all(),
                        f"모델 없을 때 0이 아님: {result['ML_SCORE'].tolist()}")


# ================================================================
# Test 2: 스케일러만 없음 → ML_SCORE=0
# ================================================================
class TestNoScalerFile(_BaseTest):
    def test_no_scaler_returns_zero(self):
        """스케일러 없이 모델만 있어도 안전 처리"""
        with tempfile.TemporaryDirectory() as tmp:
            fake_model = os.path.join(tmp, "model.pth")
            torch.save({}, fake_model)

            with patch.object(ml_engine, 'MODEL_PATH', fake_model), \
                 patch.object(ml_engine, 'SCALER_PATH', '/nonexistent/scaler.pkl'), \
                 patch.object(ml_engine, 'LEGACY_MODEL_PATH', '/nonexistent/leg.pth'), \
                 patch.object(ml_engine, 'LEGACY_SCALER_PATH', '/nonexistent/leg.pkl'):

                result = ml_engine.apply_ml_score(_make_sample_df(), _make_ohlcv_map())

        self.assertTrue((result["ML_SCORE"] == 0.0).all())


# ================================================================
# Test 3: 피처 차원 불일치 → ML_SCORE=0 (안전모드)
# ================================================================
class TestDimensionMismatch(_BaseTest):
    def test_dimension_mismatch_returns_zero(self):
        """
        스케일러: 6피처, 입력: 16피처 → ValueError → ML_SCORE=0
        (이전 버전은 스케일링 스킵 후 틀린 점수를 생성했음)
        """
        scaler_6 = StandardScaler()
        scaler_6.fit(np.random.randn(100, 6))  # 6차원

        model_16 = ml_engine.TradingAttnLSTM(16, 64, 2, 1)
        model_16.eval()

        ml_engine._loaded_lstm_model = model_16
        ml_engine._loaded_scaler = scaler_6  # 의도적 불일치!
        ml_engine._loaded_xgb_model = None

        result = ml_engine.apply_ml_score(_make_sample_df(), _make_ohlcv_map())

        self.assertIn("ML_SCORE", result.columns)
        self.assertTrue((result["ML_SCORE"] == 0.0).all(),
                        f"차원 불일치인데 0이 아님: {result['ML_SCORE'].tolist()}")


# ================================================================
# Test 4: 피처 버전 메타 불일치 → 폴백 트리거
# ================================================================
class TestFeatureVersionMismatch(_BaseTest):
    def test_meta_mismatch_returns_false(self):
        """저장된 meta의 feature_cols ≠ 코드의 FEATURE_COLS → False"""
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "meta.json")
            old_meta = {
                "feature_cols": ["Log_Ret", "Volume_Norm", "Low_Trend",
                                 "Vol_Quality", "Dist_MA20", "RSI"],
                "n_features": 6,
            }
            with open(meta_path, "w") as f:
                json.dump(old_meta, f)

            result = ml_engine._check_feature_version(meta_path)
            self.assertFalse(result, "피처 불일치인데 True 반환")

    def test_meta_match_returns_true(self):
        """피처 일치하면 True"""
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "meta.json")
            meta = {"feature_cols": ml_engine.FEATURE_COLS, "n_features": 16}
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            result = ml_engine._check_feature_version(meta_path)
            self.assertTrue(result)

    def test_no_meta_returns_true(self):
        """메타 파일 없으면 체크 스킵 (하위 호환)"""
        result = ml_engine._check_feature_version("/nonexistent/meta.json")
        self.assertTrue(result)


# ================================================================
# Test 5: 정상 추론 (모델+스케일러 일치)
# ================================================================
class TestNormalInference(_BaseTest):
    def test_matching_model_produces_valid_scores(self):
        """피처 차원 일치 시 0~100 범위 점수 생성"""
        n_feat = len(ml_engine.FEATURE_COLS)

        scaler = StandardScaler()
        scaler.fit(np.random.randn(200, n_feat))

        model = ml_engine.TradingAttnLSTM(n_feat, 64, 2, 1)
        model.eval()

        ml_engine._loaded_lstm_model = model
        ml_engine._loaded_scaler = scaler
        ml_engine._loaded_xgb_model = None

        result = ml_engine.apply_ml_score(_make_sample_df(), _make_ohlcv_map())

        self.assertIn("ML_SCORE", result.columns)
        scores = result["ML_SCORE"]
        self.assertTrue(((scores >= 0) & (scores <= 100)).all(),
                        f"범위 밖 점수: {scores.tolist()}")


# ================================================================
# Test 6: add_technical_features 피처 수 검증
# ================================================================
class TestFeatureEngineering(_BaseTest):
    def test_feature_count_matches_spec(self):
        """add_technical_features → 정확히 FEATURE_COLS 수만큼 컬럼"""
        dates = pd.bdate_range("2024-01-01", periods=100)
        np.random.seed(42)
        base = 50000 + np.cumsum(np.random.randn(100) * 500)
        df = pd.DataFrame({
            "Open": base + np.random.randn(100) * 200,
            "High": base + abs(np.random.randn(100) * 400),
            "Low": base - abs(np.random.randn(100) * 400),
            "Close": base,
            "Volume": np.random.randint(100000, 1000000, 100).astype(float),
        }, index=dates)

        result = ml_engine.add_technical_features(df)

        if not result.empty:
            self.assertEqual(list(result.columns), ml_engine.FEATURE_COLS)
            self.assertEqual(result.shape[1], len(ml_engine.FEATURE_COLS))

    def test_insufficient_data_returns_empty(self):
        """60일 미만 데이터 → 빈 DataFrame 반환 (에러 아님)"""
        dates = pd.bdate_range("2024-01-01", periods=30)
        df = pd.DataFrame({
            "Open": [100]*30, "High": [101]*30, "Low": [99]*30,
            "Close": [100]*30, "Volume": [1000]*30,
        }, index=dates)

        result = ml_engine.add_technical_features(df)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
