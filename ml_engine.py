# ml_engine.py (v13.0: Attention LSTM + Technical Indicators)
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
MODEL_PATH = "data/trading_model_lstm_attn.pth"  # 모델 파일명 변경 (v13.0)
SCALER_PATH = "data/trading_scaler_v2.pkl"       # 스케일러 버전 변경
SEQ_LENGTH = 20   # 20일치 데이터로 예측
TARGET_RET = 3.0  # 3% 이상 상승 시 성공(1)

# 학습에 사용할 기본 컬럼 (전처리 전)
BASIC_COLS = ["Open", "High", "Low", "Close", "Volume"]

# -----------------------------------------------------------
# 1. Feature Engineering (보조지표 생성)
# -----------------------------------------------------------
def add_technical_features(df):
    """
    OHLCV 데이터프레임에 기술적 지표를 추가하고,
    NaN을 처리하여 반환합니다.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    
    # 숫자형 변환 및 결측치 0 처리
    for c in BASIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 데이터가 너무 적으면 지표 계산 불가 -> 원본 리턴 혹은 빈 DF 처리
    if len(df) < 30:
        return df

    close = df['Close']
    
    # 1) 로그 수익률 (가격의 절대값 대신 변화율 학습)
    # 0으로 나누기 방지 등을 위해 shift(1)이 0인 경우 처리 필요하나, 주가 데이터라 가정
    df['Log_Ret'] = np.log(close / close.shift(1).replace(0, np.nan)).fillna(0)
    
    # 2) RSI (14)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=14).mean()
    ma_down = down.rolling(window=14).mean()
    rs = ma_up / ma_down.replace(0, np.nan) # 0 나누기 방지
    df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))
    
    # 3) MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4) Bollinger Bands (20, 2)
    ma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    # 밴드폭 (변동성 지표)
    df['BB_Width'] = (ma20 + 2 * std20 - (ma20 - 2 * std20)) / ma20.replace(0, np.nan)
    # 밴드 내 위치 % (0=하단, 1=상단)
    df['BB_Pct'] = (close - (ma20 - 2 * std20)) / (4 * std20.replace(0, np.nan))
    
    # 5) 거래량 변화율
    df['Vol_Chg'] = df['Volume'].pct_change().fillna(0)
    
    # NaN 채우기 (앞부분 이동평균 계산 구간 등)
    df = df.fillna(0)
    
    # 무한대 값 제거 (log, div 0 등)
    df = df.replace([np.inf, -np.inf], 0)

    # 최종적으로 사용할 피처 컬럼들만 선택
    # 주의: 모델 입력 차원과 순서가 항상 일치해야 함
    use_cols = [
        "Open", "High", "Low", "Close", "Volume",   # 기본
        "Log_Ret", "RSI", "MACD", "MACD_Hist",      # 추세/모멘텀
        "BB_Width", "BB_Pct", "Vol_Chg"             # 변동성/거래량
    ]
    
    # 데이터프레임에 없는 컬럼이 있다면 0으로 생성 (안전장치)
    for c in use_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    return df[use_cols]

# -----------------------------------------------------------
# 2. Attention LSTM Model Definition
# -----------------------------------------------------------
class Attention(nn.Module):
    """LSTM 출력(Sequence)에 가중치를 부여하는 Attention Layer"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim)
        # 1. 각 시점(t)의 점수 계산
        attn_scores = self.attn(lstm_output)  # (batch, seq_len, 1)
        
        # 2. Softmax로 확률(가중치) 변환
        attn_weights = F.softmax(attn_scores, dim=1) # (batch, seq_len, 1)
        
        # 3. 가중합 (Context Vector)
        context = torch.sum(attn_weights * lstm_output, dim=1)  # (batch, hidden_dim)
        return context, attn_weights

class TradingAttnLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TradingAttnLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True -> (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Attention Layer
        self.attention = Attention(hidden_dim)
        
        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64), # 학습 안정화
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
            nn.Sigmoid() # 확률 출력 (0~1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # lstm_out: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention 적용 (마지막 hidden state만 쓰는 게 아니라, 전체 시퀀스를 가중합)
        context, _ = self.attention(lstm_out)
        
        # FC Layer
        out = self.fc(context)
        return out

class StockDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# -----------------------------------------------------------
# 3. Data Processing & Training
# -----------------------------------------------------------
def build_dataset_from_history(data_dir="data"):
    """과거 ohlcv_cache 파일들을 로드하여 Feature Engineering 후 데이터셋 구축"""
    import glob
    import pickle
    
    print("🔄 [ML] 데이터셋 구축 중... (v13.0: Technical Features + Attention)")
    
    files = glob.glob(os.path.join(data_dir, "ohlcv_cache_*.pkl"))
    files.sort(reverse=True)
    # 너무 오래된 데이터는 패턴이 다를 수 있으니 최근 60일치 사용 (늘림)
    files = files[:60] 

    X_list = []
    y_list = []
    
    # 스케일링용
    all_features_flatten = []

    for f_path in files:
        try:
            with open(f_path, 'rb') as f:
                data_map = pickle.load(f)
            
            for code, df in data_map.items():
                # 최소 데이터 길이 체크 (SEQ_LENGTH + 미래확인용 5일 + 지표계산용 여유 30일)
                if len(df) < SEQ_LENGTH + 35: 
                    continue
                
                # 컬럼 매핑
                df = df.rename(columns={"시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"})
                
                # --- Feature Engineering 적용 ---
                df_feat = add_technical_features(df)
                if df_feat.empty: continue
                
                # 데이터 추출 (학습용: D-5일 기준)
                # 입력 시퀀스: T-SEQ ~ T
                # 타겟: T ~ T+5 최고가 수익률
                
                # 마지막 5일은 정답 확인용으로 남겨둠
                train_data = df_feat.iloc[:-5] 
                target_data = df.iloc[-5:] # 미래 가격 확인용 (원본 df 사용)
                
                # 입력 데이터 (최근 SEQ_LENGTH 개)
                seq_data = train_data.iloc[-SEQ_LENGTH:].values
                if len(seq_data) != SEQ_LENGTH: continue
                
                # 라벨링
                current_close = train_data.iloc[-1]['Close'] # 기준 시점 종가
                future_high = target_data['High'].max()      # 향후 5일간 최고가
                
                if current_close > 0:
                    ret = (future_high / current_close - 1) * 100
                    target = 1 if ret >= TARGET_RET else 0
                    
                    X_list.append(seq_data)
                    y_list.append(target)
                    all_features_flatten.append(seq_data)
                    
        except Exception as e:
            continue

    if not X_list:
        return None, None, None

    # 배열 변환
    X_arr = np.array(X_list) # (N, SEQ_LENGTH, FEATURE_DIM)
    y_arr = np.array(y_list)
    
    # Scaling
    N, L, D = X_arr.shape
    scaler = StandardScaler()
    
    # 3D -> 2D 펼쳐서 학습
    X_reshaped = X_arr.reshape(-1, D)
    scaler.fit(X_reshaped)
    
    joblib.dump(scaler, SCALER_PATH)
    
    X_scaled = scaler.transform(X_reshaped).reshape(N, L, D)
    
    return X_scaled, y_arr, scaler

def train_model():
    """Attention LSTM 모델 학습"""
    X, y, scaler = build_dataset_from_history()
    
    if X is None or len(X) < 100:
        print("⚠️ [ML] 학습 데이터 부족(<100)으로 모델 생성을 건너뜁니다.")
        return

    # Dataset & DataLoader
    dataset = StockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 입력 차원 확인
    input_dim = X.shape[2] 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Attention LSTM 모델 생성
    model = TradingAttnLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🔄 [ML] Attention LSTM 학습 시작 (Device: {device}, Samples: {len(X)}, Dim: {input_dim})")
    
    model.train()
    epochs = 15 # Epoch 조금 늘림
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ [ML] Attention LSTM 모델 저장 완료 -> {MODEL_PATH}")


# -----------------------------------------------------------
# 4. Inference (Apply Score)
# -----------------------------------------------------------
def apply_ml_score(current_df, full_ohlcv_map):
    """
    현재 데이터프레임에 ML 점수 반영
    Feature Engineering -> Scaling -> Attention LSTM Inference
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        current_df["ML_SCORE"] = 0
        return current_df

    try:
        device = torch.device('cpu')
        scaler = joblib.load(SCALER_PATH)
        
        # 모델 차원 정보는 저장된게 없으므로, 임시로 데이터 하나 만들어보거나 
        # Feature Engineering 함수의 출력 차원을 알아야 함.
        # add_technical_features 함수는 고정된 12개 컬럼을 반환하므로 12로 하드코딩 가능하나,
        # 안전하게 더미 데이터로 체크
        input_dim = 12 # add_technical_features의 use_cols 개수
        
        model = TradingAttnLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        scores = []
        indices = []
        
        for idx, row in current_df.iterrows():
            code = str(row['종목코드']).zfill(6)
            if code not in full_ohlcv_map:
                continue
                
            df = full_ohlcv_map[code]
            if len(df) < SEQ_LENGTH + 30: # 지표 계산 여유분 필요
                continue
                
            # 컬럼 매핑
            df = df.rename(columns={"시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"})
            
            # Feature Engineering
            df_feat = add_technical_features(df)
            if df_feat.empty: continue
            
            # 최신 SEQ_LENGTH 데이터 추출
            seq_data = df_feat.iloc[-SEQ_LENGTH:].values
            
            if len(seq_data) == SEQ_LENGTH:
                indices.append(idx)
                scores.append(seq_data)

        if not scores:
            current_df["ML_SCORE"] = 0
            return current_df
            
        # Batch Inference
        X_arr = np.array(scores)
        N, L, D = X_arr.shape
        
        # Scaling
        X_reshaped = X_arr.reshape(-1, D)
        # scaler의 feature 개수와 D가 일치해야 함. 다르면 에러나고 try-except로 빠짐.
        X_scaled = scaler.transform(X_reshaped).reshape(N, L, D)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(X_tensor).cpu().numpy().flatten()
            
        final_scores = (outputs * 100).round(1)
        
        current_df.loc[indices, "ML_SCORE"] = final_scores
        print(f"🤖 [ML] {len(indices)}개 종목 AI(Attention) 점수 예측 완료 (Avg: {np.mean(final_scores):.1f}점)")
        
    except Exception as e:
        print(f"⚠️ [ML] 점수 반영 실패: {e}")
        # 모델 구조가 바뀌어서(dim 불일치) 에러날 수 있음 -> 재학습 필요 메시지
        if "size mismatch" in str(e) or "feature names" in str(e):
            print("💡 모델이나 스케일러가 이전 버전입니다. 'python collector.py'를 실행해 모델을 재학습해주세요.")
        current_df["ML_SCORE"] = 0

    return current_df
