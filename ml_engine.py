# ml_engine.py
import os
import glob
import joblib
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- 설 정 ----------------
MODEL_PATH = "data/trading_model_rf.pkl"
FEATURE_COLS = [
    "RSI14", "MFI14", "이격도", "MACD_Slope_PCT", 
    "거래강도", "V_POWER", "NORM_RR", "BB_BW", "LDY_SCORE"
]
TARGET_DAY_LOOKAHEAD = 5   # 5일 후 수익률 예측
TARGET_RET_THRESHOLD = 3.0 # 목표 수익률 3% 이상이면 성공(1)

def _get_target_return(code, start_date, days=5):
    """
    특정 종목의 추천일(start_date) 이후 N일간 최고 수익률 조회
    """
    try:
        # 넉넉하게 10일치 가져와서 영업일 기준 days만큼 자름
        end_date = (datetime.strptime(start_date, "%Y%m%d") + timedelta(days=20)).strftime("%Y-%m-%d")
        start_fmt = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
        
        df = fdr.DataReader(code, start_fmt, end_date)
        if len(df) < 2: return 0.0
        
        # 추천 다음날 시가(또는 당일 종가) 대비 N일 내 최고가
        entry_price = df['Close'].iloc[0]
        max_price = df['High'].iloc[1:days+1].max()
        
        if pd.isna(max_price) or entry_price == 0: return 0.0
        return (max_price / entry_price - 1) * 100
    except:
        return 0.0

def build_dataset_from_history(data_dir="data"):
    """
    과거 recommend_*.csv 파일들을 긁어모아 학습 데이터셋(X, y) 생성
    """
    print("🔄 [ML] 과거 데이터로부터 학습 데이터셋 구축 중... (시간이 걸릴 수 있습니다)")
    files = glob.glob(os.path.join(data_dir, "recommend_*.csv"))
    # 'latest' 제외하고 날짜 있는 파일만
    files = [f for f in files if "latest" not in f and len(os.path.basename(f).split('_')[1]) >= 8]
    files.sort(reverse=True)
    
    # 최근 20개 파일만 사용 (너무 오래된 데이터는 노이즈가 될 수 있음)
    files = files[:20] 

    all_data = []
    
    for f in files:
        try:
            ymd = os.path.basename(f).split('_')[1][:8] # YYYYMMDD 추출
            df = pd.read_csv(f)
            
            # 필수 컬럼 있는지 확인
            if not set(FEATURE_COLS).issubset(df.columns):
                continue
                
            # 라벨링 (수익률 조회)
            # 주의: API 호출량이 많으므로 실제로는 캐싱하거나 배치 처리가 필요
            # 여기서는 샘플링하여 상위 5개만 학습에 사용 (속도 최적화)
            top_df = df.head(5).copy() 
            
            for idx, row in top_df.iterrows():
                code = str(row['종목코드']).zfill(6)
                ret = _get_target_return(code, ymd, TARGET_DAY_LOOKAHEAD)
                
                # Feature + Label
                feat = row[FEATURE_COLS].to_dict()
                feat['target'] = 1 if ret >= TARGET_RET_THRESHOLD else 0
                all_data.append(feat)
                
        except Exception as e:
            print(f"⚠️ {f} 처리 중 에러: {e}")
            continue

    return pd.DataFrame(all_data)

def train_model():
    """
    데이터셋 구축 -> 모델 학습 -> 저장
    """
    df = build_dataset_from_history()
    if df.empty or len(df) < 50:
        print("⚠️ [ML] 학습 데이터 부족으로 모델 생성을 건너뜁니다. (최소 50개 필요)")
        return

    X = df[FEATURE_COLS].fillna(0)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ [ML] 모델 학습 완료! (정확도: {acc:.2f}) -> 저장: {MODEL_PATH}")

    joblib.dump(model, MODEL_PATH)

def apply_ml_score(current_df):
    """
    현재 수집된 데이터프레임에 ML 예측 점수(ML_SCORE) 추가
    """
    if not os.path.exists(MODEL_PATH):
        # 모델이 없으면 0점으로 처리
        current_df["ML_SCORE"] = 0
        return current_df

    try:
        model = joblib.load(MODEL_PATH)
        # Feature 준비 (결측치는 0으로 대체)
        X = current_df[FEATURE_COLS].fillna(0)
        
        # 성공 확률(Probability) 예측 (0~1)
        probs = model.predict_proba(X)[:, 1]
        
        # 100점 만점으로 변환
        current_df["ML_SCORE"] = (probs * 100).round(1)
        print("🤖 [ML] AI 예측 점수 반영 완료")
        
    except Exception as e:
        print(f"⚠️ [ML] 점수 반영 실패: {e}")
        current_df["ML_SCORE"] = 0

    return current_df
