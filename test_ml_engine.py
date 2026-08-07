"""
test_ml_engine.py — ml_engine v19+ 핵심 방어 로직 단위테스트
[v20.0] 현재 API에 맞게 전면 동기화
"""
import os, sys, json, tempfile, unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_engine

# [v55.4] 파일 전체가 `import torch`로 막혀 있어, torch 없는 CI에서는 이 파일의
#   테스트가 **하나도** 돌지 않았다. 실제로 torch가 필요한 건 모델을 만들거나
#   저장하는 3개 클래스뿐이다 — 나머지(메타 버전 검증·피처 엔지니어링)는
#   torch 없이 돌려야 한다. 그래서 필요 여부를 클래스 단위로 명시한다.
TORCH_AVAILABLE = ml_engine.TORCH_AVAILABLE
_needs_torch = unittest.skipUnless(TORCH_AVAILABLE, "torch 미설치 — 모델 생성/저장 불가")
if TORCH_AVAILABLE:
    import torch

def _make_sample_df():
    return pd.DataFrame({"종목코드": ["005930","000660","035720"],
        "종목명": ["삼성전자","SK하이닉스","카카오"], "FINAL_SCORE": [75.0,60.0,50.0]})

def _make_ohlcv_map():
    dates = pd.bdate_range("2025-01-01", periods=100)
    result = {}
    for code in ["005930","000660","035720"]:
        np.random.seed(hash(code)%(2**31))
        base = 50000 + np.cumsum(np.random.randn(100)*500)
        result[code] = pd.DataFrame({"시가":base+np.random.randn(100)*200,
            "고가":base+abs(np.random.randn(100)*400),
            "저가":base-abs(np.random.randn(100)*400),
            "종가":base, "거래량":np.random.randint(100000,1000000,100).astype(float)}, index=dates)
    return result

class _Base(unittest.TestCase):
    def setUp(self):
        ml_engine._loaded_lstm_model = None
        ml_engine._loaded_scaler = None
        ml_engine._loaded_xgb_model = None
    def tearDown(self):
        ml_engine._loaded_lstm_model = None
        ml_engine._loaded_scaler = None
        ml_engine._loaded_xgb_model = None

@unittest.skipUnless(TORCH_AVAILABLE,
    "torch 미설치 시 apply_ml_score가 TORCH_MISSING으로 조기 반환 — "
    "이 테스트의 대상(모델 파일 부재 폴백) 경로에 도달할 수 없다")
class TestNoModel(_Base):
    def test_no_model_returns_zero(self):
        with patch.object(ml_engine,'MODEL_PATH','/x/m.pth'), \
             patch.object(ml_engine,'SCALER_PATH','/x/s.pkl'), \
             patch.object(ml_engine,'FALLBACK_PATHS',[]):
            r = ml_engine.apply_ml_score(_make_sample_df(), _make_ohlcv_map())
        self.assertTrue((r["ML_SCORE"]==0).all())

@_needs_torch
class TestNoScaler(_Base):
    def test_no_scaler_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.save({}, os.path.join(tmp,"m.pth"))
            with patch.object(ml_engine,'MODEL_PATH',os.path.join(tmp,"m.pth")), \
                 patch.object(ml_engine,'SCALER_PATH','/x/s.pkl'), \
                 patch.object(ml_engine,'FALLBACK_PATHS',[]):
                r = ml_engine.apply_ml_score(_make_sample_df(), _make_ohlcv_map())
        self.assertTrue((r["ML_SCORE"]==0).all())

@_needs_torch
class TestDimMismatch(_Base):
    def test_dimension_mismatch_returns_zero(self):
        s6 = StandardScaler(); s6.fit(np.random.randn(100,6))
        m16 = ml_engine.TradingAttnLSTM(16,64,2,1); m16.eval()
        ml_engine._loaded_lstm_model = m16
        ml_engine._loaded_scaler = s6
        r = ml_engine.apply_ml_score(_make_sample_df(), _make_ohlcv_map())
        self.assertTrue((r["ML_SCORE"]==0).all())

class TestFeatureVersion(_Base):
    def test_mismatch_returns_dict_match_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp,"meta.json")
            json.dump({"feature_cols":["A","B","C"],"n_features":3}, open(p,"w"))
            r = ml_engine._check_feature_version(p)
            self.assertIsInstance(r, dict)
            self.assertFalse(r["match"])
            self.assertGreater(len(r["missing_in_model"]),0)

    def test_match_returns_dict_match_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp,"meta.json")
            json.dump({"feature_cols":list(ml_engine.FEATURE_COLS)}, open(p,"w"))
            r = ml_engine._check_feature_version(p)
            self.assertTrue(r["match"])

    def test_no_meta_returns_match_false(self):
        r = ml_engine._check_feature_version("/nonexistent/meta.json")
        self.assertIsInstance(r, dict)
        self.assertFalse(r["match"])

@_needs_torch
class TestNormalInference(_Base):
    def test_valid_scores(self):
        n = len(ml_engine.FEATURE_COLS)
        s = StandardScaler(); s.fit(np.random.randn(200,n))
        m = ml_engine.TradingAttnLSTM(n,64,2,1); m.eval()
        ml_engine._loaded_lstm_model = m; ml_engine._loaded_scaler = s
        r = ml_engine.apply_ml_score(_make_sample_df(), _make_ohlcv_map())
        self.assertTrue(((r["ML_SCORE"]>=0)&(r["ML_SCORE"]<=100)).all())

class TestFeatureEng(_Base):
    def test_feature_count(self):
        dates = pd.bdate_range("2024-01-01",periods=100); np.random.seed(42)
        base = 50000+np.cumsum(np.random.randn(100)*500)
        df = pd.DataFrame({"Open":base,"High":base+200,"Low":base-200,"Close":base,
            "Volume":np.random.randint(100000,1000000,100).astype(float)}, index=dates)
        r = ml_engine.add_technical_features(df)
        if not r.empty:
            self.assertEqual(list(r.columns), list(ml_engine.FEATURE_COLS))

    def test_short_data_empty(self):
        df = pd.DataFrame({"Open":[100]*30,"High":[101]*30,"Low":[99]*30,
            "Close":[100]*30,"Volume":[1000]*30}, index=pd.bdate_range("2024-01-01",periods=30))
        self.assertTrue(ml_engine.add_technical_features(df).empty)

if __name__ == "__main__":
    unittest.main(verbosity=2)
