# ml_engine.py (v14.0: State Separation & Energy Quality)
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
MODEL_PATH = "data/trading_model_lstm_attn_v14.pth"  # 모델 버전 업
SCALER_PATH = "data/trading_scaler_v14.pkl"
SEQ_LENGTH = 20   # 20일치 데이터로 예측
TARGET_RET = 3.0  # 3% 이상 상승 시 성공(1)

# 학습에 사용할 기본 컬럼
BASIC_COLS = ["Open", "High", "Low", "Close", "Volume"]

# -----------------------------------------------------------
# 1. Feature Engineering (상태 분리 및 에너지 질 측정)
# -----------------------------------------------------------
def add_technical_features(df):
    """
    OHLCV 데이터프레임에 '살아있는 횡보'와 '죽은 횡보'를 구분하기 위한
    구조적(Structural) 피처를 추가합니다.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    
    # 숫자형 변환 및 결측치 0 처리
    for c in BASIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    if len(df) < 30:
        return df

    close = df['Close']
    low = df['Low']
    high = df['High']
    volume = df['Volume']
    
    # ---------------------------------------------------------
    # 1️⃣ Higher Low Score (저점 상승 강도) - 핵심: 추세가 살아있는가?
    # ---------------------------------------------------------
    # 최근 20일간의 저점들의 기울기를 계산 (단순화: 전반부 10일 최저점 vs 후반부 10일 최저점)
    min_l_prev = low.rolling(window=10).min().shift(10)
    min_l_curr = low.rolling(window=10).min()
    # 양수면 저점 상승 중, 음수면 저점 하락/붕괴
    df['Low_Trend'] = (min_l_curr - min_l_prev) / min_l_prev.replace(0, np.nan)

    # ---------------------------------------------------------
    # 2️⃣ Volume Quality (거래량의 질) - 핵심: 매집인가 이탈인가?
    # ---------------------------------------------------------
    # 횡보 중 양봉 거래량은 많고, 음봉 거래량은 적어야 "살아있는 횡보"
    # Close > Open (양봉)일 때의 거래량 vs 음봉일 때의 거래량 비율
    is_up = (df['Close'] > df['Open']).astype(int)
    vol_up = (volume * is_up).rolling(window=20).mean()
    vol_down = (volume * (1 - is_up)).rolling(window=20).mean()
    # 1보다 크면 매수세 우위, 1보다 작으면 매도세 우위
    df['Vol_Quality'] = vol_up / vol_down.replace(0, np.nan)

    # ---------------------------------------------------------
    # 3️⃣ Price Range Position (박스권 내 위치) - 핵심: 돌파 임박인가?
    # ---------------------------------------------------------
    # 최근 20일 고가-저가 박스권 내에서 현재 종가의 위치 (0~1)
    # 상단(0.8 이상)에서 놀고 있어야 돌파가 임박한 것 ("고가 놀이")
    period_high = high.rolling(window=20).max()
    period_low = low.rolling(window=20).min()
    df['Range_Pos'] = (close - period_low) / (period_high - period_low).replace(0, np.nan)

    # ---------------------------------------------------------
    # 4️⃣ MA Support (이평선 지지력) - 핵심: 가격이 MA 위에 얹혀있는가?
    # ---------------------------------------------------------
    # 20일 이평선과의 이격도. (너무 높으면 급등 피로감, 음수면 역배열/하락)
    # 0에 가깝지만 살짝 양수(0~5%)인 구간이 가장 좋은 눌림목
    ma20 = close.rolling(window=20).mean()
    df['Dist_MA20'] = (close - ma20) / ma20.replace(0, np.nan)

    # 5️⃣ 기존 보조지표 (변동성 체크용)
    # Bollinger Band Width (수렴 여부 확인)
    std20 = close.rolling(window=20).std()
    df['BB_Width'] = (ma20 + 2 * std20 - (ma20 - 2 * std20)) / ma20.replace(0, np.nan)
    
    # RSI (과매도/과매수 필터링)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.rolling(window=14).mean() / down.rolling(window=14).mean().replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))

    # NaN 채우기
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)

    # 최종 Feature Selection
    # [기본] + [구조적 피처] + [변동성]
    use_cols = [
        "Open", "High", "Low", "Close", "Volume",   # 0-4
        "Low_Trend",    # 5: 저점이 올라가는가? (Higher Low)
        "Vol_Quality",  # 6: 매수 거래량이 더 많은가? (Accumulation)
        "Range_Pos",    # 7: 박스권 상단에서 준비 중인가? (Ready to Break)
        "Dist_MA20",    # 8: 이평선 지지를 받고 있는가? (Support)
        "BB_Width",     # 9: 에너지가 응축되었는가? (Compression)
        "RSI"           # 10: 과열/침체 방지
    ]
    
    for c in use_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    return df[use_cols]

# -----------------------------------------------------------
# 2. Attention LSTM Model (구조 유지)
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
        # ✅ [Fix] 하드코딩된 '2' 제거 -> self.num_layers 사용
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
    import glob
    import pickle
    
    print("🔄 [ML] 데이터셋 구축 중... (v14.0: Structural Features)")
    
    files = glob.glob(os.path.join(data_dir, "ohlcv_cache_*.pkl"))
    files.sort(reverse=True)
    # 데이터 범위: 최근 60일 (패턴 변화 반영)
    files = files[:60] 

    X_list = []
    y_list = []
    
    for f_path in files:
        try:
            with open(f_path, 'rb') as f:
                data_map = pickle.load(f)
            
            for code, df in data_map.items():
                if len(df) < SEQ_LENGTH + 35: continue
                
                df = df.rename(columns={"시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"})
                
                # Feature Engineering
                df_feat = add_technical_features(df)
                if df_feat.empty: continue
                
                train_data = df_feat.iloc[:-5] 
                target_data = df.iloc[-5:]
                
                seq_data = train_data.iloc[-SEQ_LENGTH:].values
                if len(seq_data) != SEQ_LENGTH: continue
                
                # 라벨링: 단순 상승이 아니라 "유의미한 상승" (3% 이상)
                current_close = train_data.iloc[-1]['Close']
                future_high = target_data['High'].max()
                
                if current_close > 0:
                    ret = (future_high / current_close - 1) * 100
                    # 타겟 강화: 상승폭이 3% 이상이어야 함
                    target = 1 if ret >= TARGET_RET else 0
                    
                    X_list.append(seq_data)
                    y_list.append(target)
                    
        except Exception:
            continue

    if not X_list:
        return None, None, None

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
    
    if X is None:
        print("⚠️ 학습 데이터 부족")
        return

    dataset = StockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    input_dim = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TradingAttnLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🔄 [ML] 학습 시작 (Samples: {len(X)}, Dim: {input_dim})")
    
    model.train()
    epochs = 20 # Epoch 증가
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
    print("✅ [ML] 모델 업데이트 완료")

def apply_ml_score(current_df, full_ohlcv_map):
    # ✅ [Fix] 시작 시 무조건 0으로 초기화 (잔존 데이터 방지)
    current_df["ML_SCORE"] = 0.0

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return current_df

    try:
        # CPU 강제 (서버 부하 방지)
        device = torch.device('cpu')
        scaler = joblib.load(SCALER_PATH)
        
        input_dim = scaler.mean_.shape[0]
        
        # 모델 로드 시 파라미터 일치 확인
        model = TradingAttnLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        scores = []
        indices = []
        
        # 배치 처리를 위한 리스트업
        for idx, row in current_df.iterrows():
            code = str(row['종목코드']).zfill(6)
            if code not in full_ohlcv_map: continue
            df = full_ohlcv_map[code]
            if len(df) < SEQ_LENGTH + 30: continue
            
            # Feature Engineering은 collector와 공유하거나 여기서 재수행
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
            
            # 입력 차원 안전 장치
            if D != input_dim:
                print(f"⚠️ [ML] 차원 불일치 (Model: {input_dim}, Data: {D}) -> Skip")
                return current_df

            X_scaled = scaler.transform(X_arr.reshape(-1, D)).reshape(N, L, D)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                outputs = model(X_tensor).cpu().numpy().flatten()
            
            # ✅ [Fix] 점수 스케일링 및 클리핑 (0~100)
            final_scores = (outputs * 100).round(1)
            final_scores = np.clip(final_scores, 0, 100)
            
            current_df.loc[indices, "ML_SCORE"] = final_scores
            
            print(f"🤖 [ML] AI 점수 산출 완료 (Avg: {np.mean(final_scores):.1f}점, Count: {len(final_scores)})")
            
    except Exception as e:
        print(f"⚠️ [ML] 점수 반영 실패: {e}")
        # 실패 시 0점 유지

    return current_df
