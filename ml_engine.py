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
FEATURE_CACHE_PATH = "data/feature_cache_v16.pkl" 

def get_feature_cache():
    """피처 보급창에서 데이터를 로드"""
    if os.path.exists(FEATURE_CACHE_PATH):
        try:
            with open(FEATURE_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        except: return {}
    return {}

def save_feature_cache(cache_data):
    """최신 피처 데이터를 보급창에 저장"""
    with open(FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump(cache_data, f)
TARGET_RET = 3.0  
BASIC_COLS = ["Open", "High", "Low", "Close", "Volume"]


# --- [여기 추가] 당일 학습 여부 판독기 ---
from datetime import datetime

def is_trained_today(force=False):
    """오늘 날짜에 이미 모델과 스케일러가 생성되었는지 확인"""
    if force: return False
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return False
    
    # 모델 파일의 마지막 수정 시간 체크
    mtime = os.path.getmtime(MODEL_PATH)
    last_date = datetime.fromtimestamp(mtime).date()
    return last_date == datetime.now().date()
# --------------------------------------

# -----------------------------------------------------------
# 1. 모델 클래스 정의 (Attention + LSTM)
# -----------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_scores = self.attn(lstm_output) # (batch, seq, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1) # (batch, hidden)
        return context, attn_weights

class TradingAttnLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TradingAttnLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim) # [중요] Logits 반환을 위해 Sigmoid 제거
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        context, _ = self.attention(lstm_out)
        return self.fc(context)

# -----------------------------------------------------------
# 2. 데이터 전처리 및 피처 엔진
# -----------------------------------------------------------
def clean_ohlcv(df):
    """한글 컬럼 리네임 및 시계열 정합성 확보"""
    df = df.rename(columns={"시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for col in BASIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=BASIC_COLS)

def add_technical_features(df):
    """AI 학습용 핵심 특징량 산출"""
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
# 3. 데이터셋 생성 전략 (Master Strategy)
# -----------------------------------------------------------
def extract_date(path):
    m = re.search(r'(\d{8})', os.path.basename(path))
    return m.group(1) if m else "00000000"

def build_master_dataset(data_dir="data"):
    files = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.pkl")), key=extract_date)
    all_samples = []
    
    for f_path in files:
        try:
            with open(f_path, 'rb') as f: data_map = pickle.load(f)
            for code, raw_df in data_map.items():
                try:
                    df = clean_ohlcv(raw_df)
                    df_feat = add_technical_features(df)
                    if len(df_feat) < SEQ_LENGTH + 5: continue
                    
                    for i in range(SEQ_LENGTH, len(df_feat) - 5):
                        anchor_date = df_feat.index[i]
                        seq = df_feat.iloc[i-SEQ_LENGTH:i].values
                        entry_price = df.loc[anchor_date, 'Open'] 
                        future_high = df.loc[df_feat.index[i:i+5], 'High'].max()
                        
                        if entry_price > 0:
                            label = 1 if (future_high / entry_price - 1) * 100 >= TARGET_RET else 0
                            all_samples.append({'date': anchor_date, 'code': code, 'X': seq, 'y': label})
                except: continue
        except: continue

    if not all_samples: return None
    df_samples = pd.DataFrame(all_samples).drop_duplicates(subset=['date', 'code'], keep='last').sort_values('date')
    
    unique_dates = df_samples['date'].unique()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    embargo_date = split_date - pd.offsets.BDay(5)
    
    train_df = df_samples[df_samples['date'] < embargo_date]
    val_df = df_samples[df_samples['date'] >= split_date]
    
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
# 4. 학습 및 추론 인터페이스 (Collector 연동)
# -----------------------------------------------------------
class StockDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

# -----------------------------------------------------------
# 4. 학습 및 추론 인터페이스 (Collector 연동)
# -----------------------------------------------------------
def train_model(force=False):
    """collector.py 호출용: 모델 훈련 및 저장 (스마트 스킵 적용)"""
    
    # [수정된 부분] 당일 훈련 여부 체크
    if is_trained_today(force):
        print(f"✅ [SKIP] 오늘 이미 v15.6 Master 모델 학습이 완료되었습니다. (Force={force})")
        return

    print(f"🤖 AI 모델 최적화(v15.6 Master) 진행 중... (약 30분 소요)")
    
    data = build_master_dataset()
    if data is None: 
        print("⚠️ [ML] 학습할 데이터가 부족하여 중단합니다.")
        return
        
    X_tr, y_tr, X_val, y_val, meta_val, p_weight, in_dim = data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingAttnLSTM(in_dim, 64, 2, 1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=p_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # --- [v16.1 혈자리 3번: 실전 KPI 기반 학습 관리] ---
    best_kpi = 0.0
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=128)

    for epoch in range(30): # 지능 최적화를 위해 epoch을 30으로 상향 권장
        model.train()
        for b_X, b_y in DataLoader(StockDataset(X_tr, y_tr), batch_size=128, shuffle=True):
            b_X, b_y = b_X.to(device), b_y.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 🎯 [사령관의 눈] 매 Epoch 마다 실전 성능 검증
        model.eval()
        all_probs = []
        with torch.no_grad():
            for v_X, _ in val_loader:
                out = torch.sigmoid(model(v_X.to(device)))
                all_probs.extend(out.cpu().numpy().flatten())
        
        all_probs = np.array(all_probs)
        auc = roc_auc_score(y_val, all_probs)
        
        # 🎖️ 다중 구간 적중률 (Top 20/50/100) 평균 산출
        val_res = pd.DataFrame({'prob': all_probs, 'target': y_val})
        avg_precision = np.mean([
            val_res.sort_values('prob', ascending=False).head(k)['target'].mean() 
            for k in [20, 50, 100]
        ])
        
        # [Tactical KPI] 6:4 황금 비율 판정 (분별력 60% + 상위권 적중률 40%)
        commander_kpi = (0.6 * auc) + (0.4 * avg_precision)
        
        print(f"🎖️ Epoch {epoch} | Tactical KPI: {commander_kpi:.4f} (AUC: {auc:.3f}, Hit_Avg: {avg_precision:.2f})")
        
        # 최적의 지능을 발견했을 때만 모델 봉인(저장)
        if commander_kpi > best_kpi:
            best_kpi = commander_kpi
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✨ 전설적 지능 발견! v16.1 Sovereign 모델 업데이트 완료")
    
    print(f"✅ [ML] 최종 최적화 완료 (Best KPI: {best_kpi:.4f})")




def apply_ml_score(current_df, full_ohlcv_map):
    """[v16.1 패치] 0.1초 광속 추론 엔진"""
    if not os.path.exists(MODEL_PATH): return current_df

    cache = get_feature_cache()
    target_codes = current_df["종목코드"].unique()
    valid_inputs, codes = [], []
    new_cache_count = 0

    for code in target_codes:
        if code not in full_ohlcv_map: continue
        
        raw_df = full_ohlcv_map[code]
        last_date = str(raw_df.index[-1])
        
        # 🛡️ 캐시 히트: 이미 정찰된 데이터는 즉시 통과
        if code in cache and cache[code]['date'] == last_date:
            valid_inputs.append(cache[code]['seq'])
            codes.append(code)
        else:
            # 🔨 캐시 미스: 신규 데이터 정밀 정찰
            df = clean_ohlcv(raw_df)
            df_feat = add_technical_features(df)
            if len(df_feat) >= SEQ_LENGTH:
                seq = df_feat.iloc[-SEQ_LENGTH:].values
                valid_inputs.append(seq)
                codes.append(code)
                cache[code] = {'date': last_date, 'seq': seq}
                new_cache_count += 1

    if new_cache_count > 0: save_feature_cache(cache)
    if not valid_inputs: return current_df
    
    # [Tensor Flow] 하드웨어 가속 추론
    X_batch = torch.tensor(np.array(valid_inputs), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_batch)).cpu().numpy().flatten()
    
    score_map = dict(zip(codes, (probs * 100).round(1)))
    current_df["ML_SCORE"] = current_df["종목코드"].map(score_map).fillna(0.0)
    return current_df
