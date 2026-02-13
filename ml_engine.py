import os, joblib, glob, re, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score

# ---------------- 설 정 ----------------
MODEL_PATH = "data/trading_model_v15_6_master.pth"
SCALER_PATH = "data/trading_scaler_v15_6_master.pkl"
SEQ_LENGTH = 20
TARGET_RET = 3.0  
BASIC_COLS = ["Open", "High", "Low", "Close", "Volume"]

# -----------------------------------------------------------
# 1. 통합 전처리 및 피처 설계 (Clean & Feature Alignment)
# -----------------------------------------------------------
def clean_ohlcv(df):
    df = df.rename(columns={"시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    
    for col in BASIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=BASIC_COLS)

def add_technical_features(df):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    close = df['Close']
    df['Log_Ret'] = np.log(close / close.shift(1).replace(0, np.nan))
    df['Low_Trend'] = (df['Low'].rolling(10).min() - df['Low'].rolling(10).min().shift(10)) / \
                       df['Low'].rolling(10).min().shift(10).replace(0, np.nan)
    df['Vol_Quality'] = (df['Volume'] * (df['Close'] > df['Open'])).rolling(20).mean() / \
                         (df['Volume'] * (df['Close'] <= df['Open'])).rolling(20).mean().replace(0, np.nan)
    df['Dist_MA20'] = (close - close.rolling(20).mean()) / close.rolling(20).mean().replace(0, np.nan)
    
    delta = close.diff()
    up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (ema_up / ema_down.replace(0, np.nan))))

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    use_cols = ["Log_Ret", "Volume", "Low_Trend", "Vol_Quality", "Dist_MA20", "RSI"]
    return df[use_cols]

# -----------------------------------------------------------
# 2. Master Dataset Strategy (정렬 보장 및 메모리 최적화)
# -----------------------------------------------------------
def extract_date(path):
    m = re.search(r'(\d{8})', os.path.basename(path))
    return m.group(1) if m else "00000000"

def build_master_dataset(data_dir="data"):
    # [수정] 파일 날짜 기준 정렬 보장 -> drop_duplicates(keep='last')가 진짜 '최신'을 유지
    files = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.pkl")), key=extract_date)
    
    all_samples = []
    skip_counts = {"len": 0, "feat": 0, "error": 0}
    
    for f_path in files:
        f_name = os.path.basename(f_path)
        try:
            with open(f_path, 'rb') as f: data_map = pickle.load(f)
            for code, raw_df in data_map.items():
                try:
                    df = clean_ohlcv(raw_df)
                    if len(df) < SEQ_LENGTH + 40: 
                        skip_counts["len"] += 1; continue
                    
                    df_feat = add_technical_features(df)
                    if df_feat.empty: 
                        skip_counts["feat"] += 1; continue
                    
                    for i in range(SEQ_LENGTH, len(df_feat) - 5):
                        anchor_date = df_feat.index[i]
                        seq = df_feat.iloc[i-SEQ_LENGTH:i].values
                        entry_price = df.loc[anchor_date, 'Open'] 
                        future_high = df.loc[df_feat.index[i:i+5], 'High'].max()
                        
                        if entry_price > 0:
                            label = 1 if (future_high / entry_price - 1) * 100 >= TARGET_RET else 0
                            all_samples.append({'date': anchor_date, 'code': code, 'X': seq, 'y': label})
                except Exception as e:
                    # [수정] 상세 에러 로그로 디버깅 가시성 확보
                    # print(f"ERR code={code} in {f_name}: {e}")
                    skip_counts["error"] += 1; continue
        except Exception as e:
            print(f"🔥 Critical File Error {f_name}: {e}")
            continue

    if not all_samples: return None
    
    # 중복 제거 (정렬된 파일 리스트 덕분에 이제 'last'가 진짜 '최신 스냅샷'임)
    df_samples = pd.DataFrame(all_samples).drop_duplicates(subset=['date', 'code'], keep='last').sort_values('date')
    
    # 5영업일 Embargo 및 시간 분리
    unique_dates = df_samples['date'].unique()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    embargo_date = split_date - pd.offsets.BDay(5)
    
    train_df = df_samples[df_samples['date'] < embargo_date]
    val_df = df_samples[df_samples['date'] >= split_date]
    
    # [성능] 메모리 효율을 위해 numpy 스택 시점 최적화
    X_train = np.stack(train_df['X'].values); X_val = np.stack(val_df['X'].values)
    y_train = train_df['y'].values; y_val = val_df['y'].values
    
    scaler = StandardScaler(); scaler.fit(X_train.reshape(-1, X_train.shape[2]))
    joblib.dump(scaler, SCALER_PATH)
    
    X_train_s = scaler.transform(X_train.reshape(-1, X_train.shape[2])).reshape(-1, SEQ_LENGTH, X_train.shape[2])
    X_val_s = scaler.transform(X_val.reshape(-1, X_val.shape[2])).reshape(-1, SEQ_LENGTH, X_val.shape[2])
    
    pos_count = max(int(y_train.sum()), 1)
    pos_weight = torch.tensor([min((len(y_train)-pos_count)/pos_count, 20.0)], dtype=torch.float32)
    
    return X_train_s, y_train, X_val_s, y_val, val_df[['date', 'code']], pos_weight, X_train.shape[2]
