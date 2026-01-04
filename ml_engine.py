# ml_engine.py (v11.0 Upgrade: XGBoost & SHAP)
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import shap  # pip install shap 필요

# DB 매니저 (데이터 로딩용)
try:
    from db_utils import LDYDBManager
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# ---------------- 설 정 ----------------
MODEL_PATH = "data/trading_model_xgb.json"  # XGBoost는 json 저장 추천
FEATURE_COLS = [
    "RSI14", "MFI14", "이격도", "MACD_Slope_PCT", 
    "거래강도", "V_POWER", "NORM_RR", "BB_BW", "LDY_SCORE"
]
TARGET_RET_THRESHOLD = 3.0 # 목표 수익률 3% 이상이면 성공(1)

def load_training_data_from_db():
    """
    DuckDB에 저장된 과거 추천 기록과 가격 데이터를 조인하여 학습 데이터셋 생성
    (훨씬 빠르고 효율적임)
    """
    if not DB_AVAILABLE:
        print("⚠️ DB 모듈이 없어 CSV 파일 스캔 방식으로 대체합니다.")
        return None # 기존 방식 fallback 필요 시 구현

    try:
        db = LDYDBManager()
        # 1. 과거 추천 종목 가져오기
        # (실제로는 가격 데이터와 조인해서 수익률 계산 로직이 SQL로 들어가면 베스트)
        # 여기서는 간단히 '성공 여부' 컬럼이 DB에 있다고 가정하거나,
        # 혹은 별도 배치 작업으로 'target'을 업데이트해뒀다고 가정합니다.
        
        # [임시] DB에 라벨링된 데이터가 없으므로, 현재는 CSV 로딩 방식을 유지하되
        # 추후 DB 기반으로 고도화하는 것을 추천합니다.
        # 이번 버전에서는 기존 파일 스캔 방식을 개선하여 유지합니다.
        return None 
    except Exception as e:
        print(f"⚠️ DB 로딩 실패: {e}")
        return None

def build_dataset_from_history(data_dir="data"):
    """
    (기존 로직 유지 및 개선) 
    과거 recommend_*.csv 파일들을 긁어모아 학습 데이터셋(X, y) 생성
    """
    import glob
    import FinanceDataReader as fdr
    from datetime import datetime, timedelta

    print("🔄 [ML] 과거 데이터로부터 학습 데이터셋 구축 중... (XGBoost)")
    files = glob.glob(os.path.join(data_dir, "recommend_*.csv"))
    files = [f for f in files if "latest" not in f and len(os.path.basename(f).split('_')[1]) >= 8]
    files.sort(reverse=True)
    
    # 학습 데이터양을 늘리기 위해 최근 60일치 사용 (XGBoost는 데이터가 많을수록 좋음)
    files = files[:60] 

    all_data = []
    
    for f in files:
        try:
            ymd = os.path.basename(f).split('_')[1][:8]
            df = pd.read_csv(f)
            
            if not set(FEATURE_COLS).issubset(df.columns):
                continue
                
            # 상위 5개만 샘플링 (API 호출 제한 고려)
            top_df = df.head(5).copy() 
            
            # --- 수익률 조회 및 라벨링 ---
            # (속도를 위해 fdr 대신 로컬 price_snapshot이 있다면 그걸 쓰는 게 좋음)
            # 여기서는 fdr 유지하되 예외처리 강화
            next_week_date = (datetime.strptime(ymd, "%Y%m%d") + timedelta(days=7)).strftime("%Y-%m-%d")
            start_date_fmt = datetime.strptime(ymd, "%Y%m%d").strftime("%Y-%m-%d")

            for idx, row in top_df.iterrows():
                code = str(row['종목코드']).zfill(6)
                try:
                    # 5일치 데이터 조회
                    ohlcv = fdr.DataReader(code, start_date_fmt, next_week_date)
                    if len(ohlcv) < 2: continue
                    
                    entry = ohlcv['Close'].iloc[0]
                    high_max = ohlcv['High'].iloc[1:].max()
                    
                    if entry > 0:
                        ret = (high_max / entry - 1) * 100
                        target = 1 if ret >= TARGET_RET_THRESHOLD else 0
                        
                        feat = row[FEATURE_COLS].to_dict()
                        feat['target'] = target
                        all_data.append(feat)
                except:
                    continue
                    
        except Exception:
            continue

    return pd.DataFrame(all_data)

def train_model():
    """
    XGBoost 모델 학습 및 저장
    """
    df = build_dataset_from_history()
    if df.empty or len(df) < 100: # XGBoost는 데이터가 좀 더 필요함
        print("⚠️ [ML] 학습 데이터 부족(<100)으로 모델 생성을 건너뜁니다.")
        return

    X = df[FEATURE_COLS].fillna(0)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost 분류기 설정
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # 평가
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
        
    print(f"✅ [ML] XGBoost 학습 완료! (Acc: {acc:.2f}, AUC: {auc:.2f}) -> {MODEL_PATH}")

    # 모델 저장 (XGBoost 전용 포맷 또는 joblib)
    # joblib이 편하므로 joblib 사용 (버전 호환성 위해)
    joblib.dump(model, MODEL_PATH.replace(".json", ".pkl"))

def get_shap_explanation(model, X_row):
    """
    단일 샘플에 대한 SHAP 기여도 분석 (가장 영향력 큰 Feature 2개 반환)
    """
    try:
        # TreeExplainer 생성 (속도 이슈가 있다면 전역으로 뺴거나 생략)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
        
        # shap_values가 리스트인 경우(분류)와 배열인 경우 처리
        if isinstance(shap_values, list):
            vals = shap_values[1] # Class 1 (성공)에 대한 기여도
        else:
            vals = shap_values
            
        # 기여도가 큰 순서대로 인덱스 정렬
        top_indices = np.argsort(np.abs(vals))[::-1]
        
        reasons = []
        for i in range(2): # 상위 2개
            idx = top_indices[i]
            col_name = FEATURE_COLS[idx]
            val = vals[idx]
            effect = "긍정" if val > 0 else "부정"
            reasons.append(f"{col_name}({effect})")
            
        return ", ".join(reasons)
    except:
        return ""

def apply_ml_score(current_df):
    """
    현재 데이터프레임에 ML 점수 + 설명(Reason) 추가
    """
    pkl_path = MODEL_PATH.replace(".json", ".pkl")
    if not os.path.exists(pkl_path):
        current_df["ML_SCORE"] = 0
        current_df["ML_REASON"] = ""
        return current_df

    try:
        model = joblib.load(pkl_path)
        X = current_df[FEATURE_COLS].fillna(0)
        
        # 예측
        probs = model.predict_proba(X)[:, 1]
        current_df["ML_SCORE"] = (probs * 100).round(1)
        
        # [Optional] SHAP 설명 추가 (속도가 느려질 수 있으므로 상위 종목만 하거나 생략 가능)
        # 전체에 대해 루프를 돌면 느리므로, 여기서는 생략하거나 필요시 추가
        # current_df["ML_REASON"] = [get_shap_explanation(model, row) for _, row in X.iterrows()]
        current_df["ML_REASON"] = "" # 일단 비워둠 (속도 최적화)

        print("🤖 [ML] XGBoost 예측 점수 반영 완료")
        
    except Exception as e:
        print(f"⚠️ [ML] 점수 반영 실패: {e}")
        current_df["ML_SCORE"] = 0
        current_df["ML_REASON"] = ""

    return current_df
