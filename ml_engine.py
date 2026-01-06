# ml_engine.py (v12.0: PyTorch LSTM Time-Series)
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---------------- 설 정 ----------------
MODEL_PATH = "data/trading_model_lstm.pth"
SCALER_PATH = "data/trading_scaler.pkl"
SEQ_LENGTH = 20  # 20일치 패턴을 학습
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"] # 학습할 피처
TARGET_RET = 3.0 # 3% 이상 상승 시 성공(1)

# PyTorch LSTM 모델 정의
class TradingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TradingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Sigmoid() # 확률 출력 (0~1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 마지막 시점의 hidden state만 사용
        out = out[:, -1, :] 
        out = self.fc(out)
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

def build_dataset_from_history(data_dir="data"):
    """과거 ohlcv_cache 파일들을 로드하여 시계열 데이터셋 구축"""
    import glob
    import pickle
    
    print("🔄 [ML] 과거 데이터로부터 시계열 데이터셋 구축 중... (LSTM)")
    # 최근 ohlcv 캐시 파일들 찾기
    files = glob.glob(os.path.join(data_dir, "ohlcv_cache_*.pkl"))
    files.sort(reverse=True)
    files = files[:30] # 최근 30일치 캐시 사용

    X_list = []
    y_list = []
    
    # 스케일링을 위해 전체 데이터를 모을 리스트
    all_features = []

    for f_path in files:
        try:
            with open(f_path, 'rb') as f:
                data_map = pickle.load(f)
            
            for code, df in data_map.items():
                if len(df) < SEQ_LENGTH + 5: continue # 데이터 너무 짧으면 패스
                
                # OHLCV 컬럼 매핑 (한글 -> 영문)
                df = df.rename(columns={"시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"})
                if not set(FEATURE_COLS).issubset(df.columns): continue
                
                # 데이터 전처리 (결측치 제거 및 타입 변환)
                df = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # 샘플링 (모든 날짜를 다 쓰면 너무 많으니, 최근 시점 위주로)
                # 학습용: D-5일전 시점의 20일치 데이터 -> D일~D+5일 최고가 수익률 라벨링
                # (간단하게 구현하기 위해 마지막 시점 하나만 추출해서 학습 데이터로 씀)
                
                # 입력 시퀀스 (T-20 ~ T)
                seq_data = df.iloc[-(SEQ_LENGTH+5):-5][FEATURE_COLS].values
                if len(seq_data) != SEQ_LENGTH: continue
                
                # 라벨링 (T ~ T+5 사이 최고가 수익률)
                current_price = df.iloc[-6]['Close']
                future_high = df.iloc[-5:]['High'].max()
                
                if current_price > 0:
                    ret = (future_high / current_price - 1) * 100
                    target = 1 if ret >= TARGET_RET else 0
                    
                    X_list.append(seq_data)
                    y_list.append(target)
                    all_features.append(seq_data)
                    
        except Exception as e:
            continue

    if not X_list:
        return None, None, None

    # 데이터 스케일링 (학습 전 전체 분포로 스케일러 피팅)
    X_arr = np.array(X_list) # (N, 20, 5)
    y_arr = np.array(y_list)
    
    # 3D -> 2D로 펼쳐서 스케일링 후 다시 3D로
    N, L, D = X_arr.shape
    scaler = StandardScaler()
    X_reshaped = X_arr.reshape(-1, D)
    scaler.fit(X_reshaped)
    
    # 스케일러 저장
    joblib.dump(scaler, SCALER_PATH)
    
    # 정규화 적용
    X_scaled = scaler.transform(X_reshaped).reshape(N, L, D)
    
    return X_scaled, y_arr, scaler

def train_model():
    """LSTM 모델 학습"""
    X, y, scaler = build_dataset_from_history()
    
    if X is None or len(X) < 50:
        print("⚠️ [ML] 학습 데이터 부족(<50)으로 모델 생성을 건너뜁니다.")
        return

    # Dataset & DataLoader
    dataset = StockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 모델 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingLSTM(input_dim=len(FEATURE_COLS), hidden_dim=64, num_layers=2, output_dim=1).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🔄 [ML] LSTM 학습 시작 (Device: {device}, Samples: {len(X)})")
    
    model.train()
    epochs = 10  # Epoch 수
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    # 모델 저장
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ [ML] LSTM 모델 저장 완료 -> {MODEL_PATH}")


def apply_ml_score(current_df, full_ohlcv_map):
    """
    현재 데이터프레임에 ML 점수 반영
    - full_ohlcv_map: 종목별 전체 OHLCV 데이터 (시계열 추출용)
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        current_df["ML_SCORE"] = 0
        return current_df

    try:
        # 모델 & 스케일러 로드
        device = torch.device('cpu') # 추론은 CPU로 해도 충분
        scaler = joblib.load(SCALER_PATH)
        
        model = TradingLSTM(input_dim=len(FEATURE_COLS), hidden_dim=64, num_layers=2, output_dim=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        scores = []
        indices = []
        
        # 각 종목별로 시계열 데이터 추출
        for idx, row in current_df.iterrows():
            code = str(row['종목코드']).zfill(6)
            if code not in full_ohlcv_map:
                continue
                
            df = full_ohlcv_map[code]
            if df.empty or len(df) < SEQ_LENGTH:
                continue
                
            # 최신 20일치 데이터 추출
            df = df.rename(columns={"시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"})
            seq_data = df[FEATURE_COLS].iloc[-SEQ_LENGTH:].apply(pd.to_numeric, errors='coerce').fillna(0).values
            
            if len(seq_data) == SEQ_LENGTH:
                indices.append(idx)
                scores.append(seq_data)

        if not scores:
            current_df["ML_SCORE"] = 0
            return current_df
            
        # 일괄 추론 (Batch Inference)
        X_arr = np.array(scores) # (N, 20, 5)
        N, L, D = X_arr.shape
        
        # 스케일링
        X_reshaped = X_arr.reshape(-1, D)
        X_scaled = scaler.transform(X_reshaped).reshape(N, L, D)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(X_tensor).cpu().numpy().flatten()
            
        # 점수 매핑 (0~1 확률 -> 0~100점)
        final_scores = (outputs * 100).round(1)
        
        # DataFrame에 업데이트
        current_df.loc[indices, "ML_SCORE"] = final_scores
        print(f"🤖 [ML] {len(indices)}개 종목 AI 점수 예측 완료 (Avg: {np.mean(final_scores):.1f}점)")
        
    except Exception as e:
        print(f"⚠️ [ML] 점수 반영 실패: {e}")
        current_df["ML_SCORE"] = 0

    return current_df
