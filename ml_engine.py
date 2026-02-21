"""
ml_engine.py v17.0 — Ensemble AI Engine (LSTM+Attention + XGBoost)

개선사항 (v16.1 → v17.0):
  1. [피처 확장] 6개 → 16개 (수급·매물대·다중 타임프레임)
  2. [레이블 변경] 고가 기준 → 종가 기준 (실현 가능한 수익 기준)
  3. [앙상블] LSTM + XGBoost 가중 평균 (정밀도↑)
  4. [Focal Loss] 클래스 불균형에 강건한 손실함수
  5. [BugFix] apply_ml_score에서 모델 미로드 버그 수정
"""

import os, joblib, glob, re, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from datetime import datetime

# --- XGBoost (optional) ---
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

# ====================== 설정 ======================
MODEL_PATH       = "data/trading_model_v17.pth"
SCALER_PATH      = "data/trading_scaler_v17.pkl"
XGB_MODEL_PATH   = "data/trading_model_xgb_v17.pkl"
FEATURE_CACHE_PATH = "data/feature_cache_v17.pkl"

# 하위 호환: v15.6 모델이 있으면 폴백
LEGACY_MODEL_PATH  = "data/trading_model_v15_6_master.pth"
LEGACY_SCALER_PATH = "data/trading_scaler_v15_6_master.pkl"

SEQ_LENGTH  = 20
TARGET_RET  = 3.0        # 종가 기준 3% 상승 레이블
BASIC_COLS  = ["Open", "High", "Low", "Close", "Volume"]

# ====================== 피처 엔진 (16개) ======================
FEATURE_COLS = [
    "Log_Ret", "Volume_Norm", "Low_Trend", "Vol_Quality", "Dist_MA20",
    "RSI", "MFI", "MACD_Hist_Norm", "BB_Width", "ATR_Pct",
    "OBV_Slope", "Range_Pos", "Vol_Ratio_5", "Ret_5d", "Ret_20d",
    "Upper_Shadow_Ratio",
]


def get_feature_cache():
    if os.path.exists(FEATURE_CACHE_PATH):
        try:
            with open(FEATURE_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    return {}


def save_feature_cache(cache_data):
    with open(FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump(cache_data, f)


def is_trained_today(force=False):
    if force:
        return False
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return False
    mtime = os.path.getmtime(MODEL_PATH)
    return datetime.fromtimestamp(mtime).date() == datetime.now().date()


def clean_ohlcv(df):
    """한글 컬럼 리네임 및 정합성 확보"""
    df = df.rename(columns={
        "시가": "Open", "고가": "High", "저가": "Low",
        "종가": "Close", "거래량": "Volume"
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    for col in BASIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=BASIC_COLS)


def add_technical_features(df):
    """
    [v17.0] 16개 피처 산출 — 기존 6개 + 수급/매물/다중타임프레임 10개 추가
    """
    if len(df) < 60:
        return pd.DataFrame()

    df = df.copy()
    c, h, l, o, v = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']

    # === 기존 호환 피처 ===
    df['Log_Ret'] = np.log(c / c.shift(1).replace(0, np.nan))

    # Volume 정규화 (20일 평균 대비 비율 → log 변환)
    vol_ma20 = v.rolling(20).mean().replace(0, np.nan)
    df['Volume_Norm'] = np.log1p(v / vol_ma20)

    df['Low_Trend'] = (l.rolling(10).min() - l.rolling(10).min().shift(10)) / \
                       l.rolling(10).min().shift(10).replace(0, np.nan)

    is_up = c > o
    vol_up_sum = (v * is_up.astype(float)).rolling(20).sum()
    vol_dn_sum = (v * (~is_up).astype(float)).rolling(20).sum()
    up_cnt = is_up.rolling(20).sum().replace(0, np.nan)
    dn_cnt = (~is_up).rolling(20).sum().replace(0, np.nan)
    vol_up_avg = vol_up_sum / up_cnt
    vol_dn_avg = vol_dn_sum / dn_cnt
    df['Vol_Quality'] = (vol_up_avg / vol_dn_avg.replace(0, np.nan)).clip(0, 5)

    ma20 = c.rolling(20).mean()
    df['Dist_MA20'] = (c - ma20) / ma20.replace(0, np.nan)

    # RSI (EMA 방식, 0~1 정규화)
    delta = c.diff()
    up_d = delta.clip(lower=0)
    down_d = -1 * delta.clip(upper=0)
    ema_up = up_d.ewm(com=13, adjust=False).mean()
    ema_down = down_d.ewm(com=13, adjust=False).mean()
    df['RSI'] = (100 - (100 / (1 + (ema_up / ema_down.replace(0, np.nan))))) / 100.0

    # === 신규 10개 피처 ===

    # MFI (Money Flow Index) — 자금 흐름 (0~1)
    tp = (h + l + c) / 3
    rmf = tp * v
    pos_flow = np.where(tp.diff() > 0, rmf, 0)
    neg_flow = np.where(tp.diff() < 0, rmf, 0)
    pos_sum = pd.Series(pos_flow, index=c.index).rolling(14).sum()
    neg_sum = pd.Series(neg_flow, index=c.index).rolling(14).sum().replace(0, 1)
    df['MFI'] = (100 - (100 / (1 + pos_sum / neg_sum))) / 100.0

    # MACD Histogram (주가 대비 % 정규화)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    df['MACD_Hist_Norm'] = hist / c.replace(0, np.nan) * 100

    # Bollinger Band Width (변동성)
    std20 = c.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    df['BB_Width'] = (bb_upper - bb_lower) / ma20.replace(0, np.nan)

    # ATR % (변동성)
    tr = pd.concat([
        (h - l),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df['ATR_Pct'] = atr14 / c.replace(0, np.nan)

    # OBV Slope (스마트 머니 방향)
    obv_sign = np.sign(c.diff()).fillna(0)
    obv = (obv_sign * v).cumsum()
    obv_ma5 = obv.rolling(5).mean()
    obv_ma20 = obv.rolling(20).mean()
    df['OBV_Slope'] = ((obv_ma5 - obv_ma20) / obv_ma20.abs().replace(0, np.nan)).clip(-2, 2)

    # Range Position (박스권 위치, 0~1)
    h20 = h.rolling(20).max()
    l20 = l.rolling(20).min()
    denom = (h20 - l20).replace(0, np.nan)
    df['Range_Pos'] = (c - l20) / denom

    # 거래량 비율 (5일 MA 대비, log)
    vol_ma5 = v.rolling(5).mean().replace(0, np.nan)
    df['Vol_Ratio_5'] = np.log1p(v / vol_ma5)

    # 5일/20일 수익률
    df['Ret_5d'] = c.pct_change(5)
    df['Ret_20d'] = c.pct_change(20)

    # 윗꼬리 비율 (분배봉 감지)
    candle_range = (h - l).replace(0, np.nan)
    body_top = pd.concat([c, o], axis=1).max(axis=1)
    df['Upper_Shadow_Ratio'] = (h - body_top) / candle_range

    # 정리
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
    return df[FEATURE_COLS]


# ====================== 모델 아키텍처 ======================

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        scores = self.attn(lstm_output)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights


class TradingAttnLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        ctx, _ = self.attention(out)
        return self.fc(ctx)


# ====================== Focal Loss ======================

class FocalLoss(nn.Module):
    """
    Focal Loss — 쉬운 샘플의 가중치를 줄여 어려운(소수) 클래스에 집중
    gamma=2.0: 잘 맞추는 샘플의 영향을 크게 줄임
    alpha=0.7: 양성 클래스(상승) 가중치
    """
    def __init__(self, alpha=0.7, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ====================== 모델 로딩 (전역 캐시) ======================

_loaded_lstm_model = None
_loaded_scaler = None
_loaded_xgb_model = None


def load_model():
    """
    [BugFix] 디스크에서 모델+스케일러 로드
    기존: model이 train_model 로컬 변수 → 학습 스킵 시 apply_ml_score 실패
    수정: 전역 캐시에 로드하여 언제든 추론 가능
    """
    global _loaded_lstm_model, _loaded_scaler, _loaded_xgb_model

    # LSTM 모델 (v17 우선, 없으면 v15.6 폴백)
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH
    scaler_path = SCALER_PATH if os.path.exists(SCALER_PATH) else LEGACY_SCALER_PATH

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            _loaded_scaler = joblib.load(scaler_path)

            # 피처 차원 자동 감지
            if hasattr(_loaded_scaler, 'n_features_in_'):
                in_dim = _loaded_scaler.n_features_in_
            else:
                in_dim = len(FEATURE_COLS)

            device = torch.device('cpu')
            _loaded_lstm_model = TradingAttnLSTM(in_dim, 64, 2, 1).to(device)
            state = torch.load(model_path, map_location=device, weights_only=True)
            _loaded_lstm_model.load_state_dict(state)
            _loaded_lstm_model.eval()
            print(f"✅ [ML] LSTM 모델 로드 완료: {model_path} (features={in_dim})")
        except Exception as e:
            print(f"⚠️ [ML] LSTM 모델 로드 실패: {e}")
            _loaded_lstm_model = None
            _loaded_scaler = None

    # XGBoost (있으면 로드)
    if XGB_OK and os.path.exists(XGB_MODEL_PATH):
        try:
            _loaded_xgb_model = joblib.load(XGB_MODEL_PATH)
            print(f"✅ [ML] XGBoost 모델 로드 완료: {XGB_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ [ML] XGBoost 로드 실패: {e}")
            _loaded_xgb_model = None


# ====================== 데이터셋 빌드 ======================

def extract_date(path):
    m = re.search(r'(\d{8})', os.path.basename(path))
    return m.group(1) if m else "00000000"


def build_master_dataset(data_dir="data"):
    """
    [v17.0] 종가 기준 레이블 + 확장 피처

    레이블 변경 이유:
      기존: 5일 내 '고가' 기준 3% → 순간 찍고 하락해도 양성 → 실현 불가 수익
      수정: 5일 내 '종가' 기준 3% → 장 마감까지 유지되는 실현 가능한 수익
    """
    files = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.pkl")), key=extract_date)
    all_samples = []

    for f_path in files:
        try:
            with open(f_path, 'rb') as f:
                data_map = pickle.load(f)
            for code, raw_df in data_map.items():
                try:
                    df = clean_ohlcv(raw_df)
                    df_feat = add_technical_features(df)
                    if len(df_feat) < SEQ_LENGTH + 6:
                        continue

                    for i in range(SEQ_LENGTH, len(df_feat) - 6):
                        anchor_date = df_feat.index[i]
                        seq = df_feat.iloc[i - SEQ_LENGTH:i].values

                        # 다음날 시가로 진입
                        entry_idx = i + 1
                        if entry_idx >= len(df):
                            continue
                        entry_price = df.iloc[entry_idx]['Open']
                        if entry_price <= 0:
                            continue

                        # [v17.0 핵심] 5영업일 종가 중 최대 수익률
                        future_closes = df.iloc[entry_idx:entry_idx + 5]['Close']
                        if future_closes.empty:
                            continue
                        max_close_ret = (future_closes.max() / entry_price - 1) * 100

                        label = 1 if max_close_ret >= TARGET_RET else 0
                        all_samples.append({
                            'date': anchor_date, 'code': code,
                            'X': seq, 'y': label
                        })
                except Exception:
                    continue
        except Exception:
            continue

    if not all_samples:
        return None

    df_samples = pd.DataFrame(all_samples) \
        .drop_duplicates(subset=['date', 'code'], keep='last') \
        .sort_values('date')

    # 시간 기반 분할 (미래 누출 방지)
    unique_dates = df_samples['date'].unique()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    embargo_date = split_date - pd.offsets.BDay(5)

    train_df = df_samples[df_samples['date'] < embargo_date]
    val_df = df_samples[df_samples['date'] >= split_date]

    if len(train_df) < 100 or len(val_df) < 50:
        print(f"⚠️ [ML] 데이터 부족: train={len(train_df)}, val={len(val_df)}")
        return None

    X_train = np.stack(train_df['X'].values)
    X_val = np.stack(val_df['X'].values)
    y_train = train_df['y'].values
    y_val = val_df['y'].values

    # 스케일러 피팅
    scaler = StandardScaler()
    n_feat = X_train.shape[2]
    scaler.fit(X_train.reshape(-1, n_feat))
    joblib.dump(scaler, SCALER_PATH)

    X_train_s = scaler.transform(X_train.reshape(-1, n_feat)).reshape(-1, SEQ_LENGTH, n_feat)
    X_val_s = scaler.transform(X_val.reshape(-1, n_feat)).reshape(-1, SEQ_LENGTH, n_feat)

    pos_ratio = y_train.mean()
    print(f"📊 [ML] 양성 비율: {pos_ratio:.2%} (train={len(y_train)}, val={len(y_val)})")

    return X_train_s, y_train, X_val_s, y_val, val_df[['date', 'code']], n_feat


# ====================== Dataset ======================

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


# ====================== 학습 ======================

def train_model(force=False):
    """[v17.0] LSTM + XGBoost 앙상블 학습"""

    if is_trained_today(force):
        print("✅ [SKIP] 오늘 이미 v17 모델 학습이 완료되었습니다.")
        return

    print("🤖 AI 모델 v17.0 학습 시작 (LSTM+Attention + XGBoost)...")

    data = build_master_dataset()
    if data is None:
        print("⚠️ [ML] 학습 데이터 부족으로 중단합니다.")
        return

    X_tr, y_tr, X_val, y_val, meta_val, in_dim = data

    # ============= (1) LSTM 학습 =============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingAttnLSTM(in_dim, 64, 2, 1).to(device)
    criterion = FocalLoss(alpha=0.7, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_kpi = 0.0
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=128)

    for epoch in range(30):
        model.train()
        for b_X, b_y in DataLoader(StockDataset(X_tr, y_tr), batch_size=128, shuffle=True):
            b_X = b_X.to(device)
            b_y = b_y.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # 검증
        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in val_loader:
                v_X = batch[0].to(device)
                out = torch.sigmoid(model(v_X))
                all_probs.extend(out.cpu().numpy().flatten())

        all_probs = np.array(all_probs)

        try:
            auc = roc_auc_score(y_val, all_probs)
        except ValueError:
            auc = 0.5

        # 다중 구간 적중률 (상위 20/50/100)
        val_res = pd.DataFrame({'prob': all_probs, 'target': y_val})
        hit_rates = []
        for k in [20, 50, 100]:
            if len(val_res) >= k:
                hit_rates.append(
                    val_res.sort_values('prob', ascending=False).head(k)['target'].mean()
                )
        avg_precision = np.mean(hit_rates) if hit_rates else 0.0

        # 적중률 비중 50% + AUC 비중 50% (상위권 정밀도 중시)
        kpi = (0.5 * auc) + (0.5 * avg_precision)

        if epoch % 5 == 0 or kpi > best_kpi:
            print(f"  Epoch {epoch:2d} | KPI: {kpi:.4f} "
                  f"(AUC: {auc:.3f}, HitAvg: {avg_precision:.2%})")

        if kpi > best_kpi:
            best_kpi = kpi
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"✅ [LSTM] 학습 완료 (Best KPI: {best_kpi:.4f})")

    # ============= (2) XGBoost 학습 =============
    if XGB_OK:
        print("🌲 XGBoost 앙상블 학습 시작...")
        # XGBoost는 마지막 시점의 피처 벡터만 사용
        X_tr_xgb = X_tr[:, -1, :]
        X_val_xgb = X_val[:, -1, :]

        pos_count = max(int(y_tr.sum()), 1)
        neg_count = len(y_tr) - pos_count

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg_count / pos_count,
            eval_metric='logloss',
            early_stopping_rounds=20,
            random_state=42,
            verbosity=0,
        )
        xgb_model.fit(
            X_tr_xgb, y_tr,
            eval_set=[(X_val_xgb, y_val)],
            verbose=False
        )
        joblib.dump(xgb_model, XGB_MODEL_PATH)

        xgb_probs = xgb_model.predict_proba(X_val_xgb)[:, 1]
        try:
            xgb_auc = roc_auc_score(y_val, xgb_probs)
        except ValueError:
            xgb_auc = 0.5
        print(f"✅ [XGBoost] 학습 완료 (AUC: {xgb_auc:.3f})")

        # 피처 중요도 Top 5
        importance = dict(zip(FEATURE_COLS[:in_dim], xgb_model.feature_importances_))
        top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top 피처: {', '.join(f'{k}({v:.2f})' for k, v in top5)}")

    # 전역 캐시에 로드
    load_model()
    print("✅ [ML] v17.0 전체 학습 파이프라인 완료!")


# ====================== 추론 ======================

def apply_ml_score(current_df, full_ohlcv_map):
    """
    [v17.0] LSTM + XGBoost 앙상블 추론

    BugFix: 모델이 메모리에 없으면 디스크에서 자동 로드
    앙상블: LSTM 60% + XGBoost 40%
    """
    global _loaded_lstm_model, _loaded_scaler, _loaded_xgb_model

    # [BugFix] 모델이 메모리에 없으면 디스크에서 로드
    if _loaded_lstm_model is None or _loaded_scaler is None:
        load_model()

    # 모델 파일 자체가 없는 경우
    if _loaded_lstm_model is None or _loaded_scaler is None:
        print("⚠️ [ML] 모델 파일 없음. ML_SCORE=0 으로 진행.")
        current_df["ML_SCORE"] = 0.0
        return current_df

    cache = get_feature_cache()
    target_codes = current_df["종목코드"].unique()
    valid_inputs, codes = [], []
    new_cache_count = 0

    for code in target_codes:
        if code not in full_ohlcv_map:
            continue

        raw_df = full_ohlcv_map[code]
        last_date = str(raw_df.index[-1])

        if code in cache and cache[code].get('date') == last_date:
            valid_inputs.append(cache[code]['seq'])
            codes.append(code)
        else:
            df = clean_ohlcv(raw_df)
            df_feat = add_technical_features(df)
            if len(df_feat) >= SEQ_LENGTH:
                seq = df_feat.iloc[-SEQ_LENGTH:].values
                valid_inputs.append(seq)
                codes.append(code)
                cache[code] = {'date': last_date, 'seq': seq}
                new_cache_count += 1

    if new_cache_count > 0:
        save_feature_cache(cache)

    if not valid_inputs:
        current_df["ML_SCORE"] = 0.0
        return current_df

    X_raw = np.array(valid_inputs)

    # 스케일러 적용
    n_feat = X_raw.shape[2]
    scaler_dim = getattr(_loaded_scaler, 'n_features_in_', n_feat)

    if n_feat != scaler_dim:
        # [v8.5 Fix] 피처 차원 불일치 → 의미 없는 추론 방지, 안전 폴백
        print(f"⚠️ [ML] 피처 차원 불일치 (데이터={n_feat}, 모델={scaler_dim}). ML_SCORE=0 폴백.")
        current_df["ML_SCORE"] = 0.0
        return current_df

    try:
        X_scaled = _loaded_scaler.transform(
            X_raw.reshape(-1, n_feat)
        ).reshape(-1, SEQ_LENGTH, n_feat)
    except Exception as e:
        print(f"⚠️ [ML] 스케일링 실패: {e}. ML_SCORE=0 폴백.")
        current_df["ML_SCORE"] = 0.0
        return current_df

    # --- LSTM 추론 ---
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    _loaded_lstm_model.eval()
    with torch.no_grad():
        lstm_probs = torch.sigmoid(_loaded_lstm_model(X_tensor)).cpu().numpy().flatten()

    # --- XGBoost 추론 (앙상블) ---
    final_probs = lstm_probs
    if _loaded_xgb_model is not None:
        try:
            X_xgb = X_scaled[:, -1, :]  # 마지막 시점 피처
            xgb_probs = _loaded_xgb_model.predict_proba(X_xgb)[:, 1]
            # 앙상블: LSTM 60% + XGBoost 40%
            final_probs = lstm_probs * 0.6 + xgb_probs * 0.4
        except Exception as e:
            print(f"⚠️ [ML] XGBoost 추론 실패, LSTM 단독: {e}")

    score_map = dict(zip(codes, (final_probs * 100).round(1)))
    current_df["ML_SCORE"] = current_df["종목코드"].map(score_map).fillna(0.0)

    # 상위 5개 로그
    top5 = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:5]
    if top5:
        print(f"🧠 [ML] Top5: {', '.join(f'{c}({s})' for c, s in top5)}")

    return current_df
