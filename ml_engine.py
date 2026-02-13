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

# -----------------------------------------------------------
# 3. Training & Inference (Collector 연동용 함수)
# -----------------------------------------------------------

class StockDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

def train_model():
    """collector.py에서 호출하는 훈련 시작 함수"""
    data = build_master_dataset()
    if data is None: return
    X_tr, y_tr, X_val, y_val, meta_val, p_weight, in_dim = data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TradingAttnLSTM 클래스가 상단에 정의되어 있어야 합니다.
    model = TradingAttnLSTM(in_dim, 64, 2, 1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=p_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_prec = 0
    for epoch in range(15): # 빠른 동기화를 위해 epoch 조정
        model.train()
        for b_X, b_y in DataLoader(StockDataset(X_tr, y_tr), batch_size=128, shuffle=True):
            b_X, b_y = b_X.to(device), b_y.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ [ML] v15.6 Master 모델 학습 및 저장 완료")

def apply_ml_score(current_df, full_ohlcv_map):
    """collector.py의 main 함수에서 호출하여 AI 점수를 주입하는 함수"""
    current_df["ML_SCORE"] = 0.0
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return current_df

    try:
        device = torch.device('cpu')
        scaler = joblib.load(SCALER_PATH)
        # 훈련 시 사용한 input_dim 확인
        input_dim = scaler.mean_.shape[0]
        model = TradingAttnLSTM(input_dim, 64, 2, 1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        scores, indices = [], []
        for idx, row in current_df.iterrows():
            code = str(row['종목코드']).zfill(6)
            if code not in full_ohlcv_map: continue
            
            df = clean_ohlcv(full_ohlcv_map[code])
            if len(df) < SEQ_LENGTH: continue
            
            df_feat = add_technical_features(df)
            if df_feat.empty: continue
            
            seq_data = df_feat.iloc[-SEQ_LENGTH:].values
            if len(seq_data) == SEQ_LENGTH:
                indices.append(idx)
                scores.append(seq_data)

        if scores:
            X_arr = np.array(scores)
            N, L, D = X_arr.shape
            X_scaled = scaler.transform(X_arr.reshape(-1, D)).reshape(N, L, D)
            with torch.no_grad():
                logits = model(torch.tensor(X_scaled, dtype=torch.float32))
                probs = torch.sigmoid(logits).numpy().flatten()
                current_df.loc[indices, "ML_SCORE"] = (probs * 100).round(1)
                
    except Exception as e:
        print(f"⚠️ AI 점수 산출 실패: {e}")
    return current_df
