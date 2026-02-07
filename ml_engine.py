# ml_engine.py (v15.0: Stability & Cleanup)
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---------------- 설 정 ----------------
MODEL_PATH = "data/trading_model_lstm_attn_v15.pth"
SCALER_PATH = "data/trading_scaler_v15.pkl"
SEQ_LENGTH = 20
TARGET_RET = 3.0
BASIC_COLS = ["Open", "High", "Low", "Close", "Volume"]

# -----------------------------------------------------------
# 1. Feature Engineering
# -----------------------------------------------------------
def add_technical_features(df):
    if df.empty: return pd.DataFrame()
    df = df.copy()
    for c in BASIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    if len(df) < 30: return df
    
    close = df['Close']
    low = df['Low']
    high = df['High']
    volume = df['Volume']
    
    # 1. Higher Low Score
    min_l_prev = low.rolling(window=10).min().shift(10)
    min_l_curr = low.rolling(window=10).min()
    df['Low_Trend'] = (min_l_curr - min_l_prev) / min_l_prev.replace(0, np.nan)

    # 2. Volume Quality
    is_up = (df['Close'] > df['Open']).astype(int)
    vol_up = (volume * is_up).rolling(window=20).mean()
    vol_down = (volume * (1 - is_up)).rolling(window=20).mean()
    df['Vol_Quality'] = vol_up / vol_down.replace(0, np.nan)

    # 3. Range Pos
    period_high = high.rolling(window=20).max()
    period_low = low.rolling(window=20).min()
    df['Range_Pos'] = (close - period_low) / (period_high - period_low).replace(0, np.nan)

    # 4. MA Support
    ma20 = close.rolling(window=20).mean()
    df['Dist_MA20'] = (close - ma20) / ma20.replace(0, np.nan)

    # 5. Indicators
    std20 = close.rolling(window=20).std()
    df['BB_Width'] = (ma20 + 2 * std20 - (ma20 - 2 * std20)) / ma20.replace(0, np.nan)
    
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.rolling(window=14).mean() / down.rolling(window=14).mean().replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))

    df = df.fillna(0).replace([np.inf, -np.inf], 0)

    use_cols = ["Open", "High", "Low", "Close", "Volume", "Low_Trend", "Vol_Quality", "Range_Pos", "Dist_MA20", "BB_Width", "RSI"]
    for c in use_cols:
        if c not in df.columns: df[c] = 0.0
            
    return df[use_cols]

# -----------------------------------------------------------
# 2. Attention LSTM Model
# -----------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_scores = self.attn(lstm_output)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class TradingAttnLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TradingAttnLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers # ✅ 이 줄이 꼭 있어야 합니다!
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # ✅ self.num_layers를 사용해 동적으로 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        context, _ = self.attention(lstm_out)
        out = self.fc(context)
        return out

class StockDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

# -----------------------------------------------------------
# 3. Processing & Training
# -----------------------------------------------------------
def build_dataset_from_history(data_dir="data"):
    import glob
    import pickle
    files = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.pkl")), reverse=True)[:60]
    X_list, y_list = [], []
    for f_path in files:
        try:
            with open(f_path, 'rb') as f: data_map = pickle.load(f)
            for code, df in data_map.items():
                if len(df) < SEQ_LENGTH + 35: continue
                df = df.rename(columns={"시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"})
                df_feat = add_technical_features(df)
                if df_feat.empty: continue
                
                train_data = df_feat.iloc[:-5]
                target_data = df.iloc[-5:]
                seq_data = train_data.iloc[-SEQ_LENGTH:].values
                if len(seq_data) != SEQ_LENGTH: continue
                
                curr = train_data.iloc[-1]['Close']
                fut = target_data['High'].max()
                if curr > 0:
                    ret = (fut / curr - 1) * 100
                    X_list.append(seq_data)
                    y_list.append(1 if ret >= TARGET_RET else 0)
        except: continue

    if not X_list: return None, None, None
    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    N, L, D = X_arr.shape
    scaler = StandardScaler()
    X_reshaped = X_arr.reshape(-1, D)
    scaler.fit(X_reshaped)
    joblib.dump(scaler, SCALER_PATH)
    X_scaled = scaler.transform(X_reshaped).reshape(N, L, D)
    return X_scaled, y_arr, scaler

def train_model():
    X, y, scaler = build_dataset_from_history()
    if X is None: return
    dataset = StockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_dim = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingAttnLSTM(input_dim, 64, 2, 1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(20):
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)

def apply_ml_score(current_df, full_ohlcv_map):
    current_df["ML_SCORE"] = 0.0
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH): return current_df

    try:
        device = torch.device('cpu')
        scaler = joblib.load(SCALER_PATH)
        input_dim = scaler.mean_.shape[0]
        model = TradingAttnLSTM(input_dim, 64, 2, 1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        scores, indices = [], []
        for idx, row in current_df.iterrows():
            code = str(row['종목코드']).zfill(6)
            if code not in full_ohlcv_map: continue
            df = full_ohlcv_map[code]
            if len(df) < SEQ_LENGTH + 30: continue
            df = df.rename(columns={"시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"})
            df_feat = add_technical_features(df)
            if df_feat.empty: continue
            seq_data = df_feat.iloc[-SEQ_LENGTH:].values
            if len(seq_data) == SEQ_LENGTH:
                indices.append(idx)
                scores.append(seq_data)

        if scores:
            X_arr = np.array(scores)
            N, L, D = X_arr.shape
            if D != input_dim: return current_df
            X_scaled = scaler.transform(X_arr.reshape(-1, D)).reshape(N, L, D)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(X_tensor).cpu().numpy().flatten()
            final_scores = (outputs * 100).round(1)
            final_scores = np.clip(final_scores, 0, 100)
            current_df.loc[indices, "ML_SCORE"] = final_scores
            print(f"🤖 [ML] AI 점수 산출 완료 (Avg: {np.mean(final_scores):.1f})")
    except Exception as e:
        print(f"⚠️ [ML] 점수 반영 실패: {e}")
    return current_df
