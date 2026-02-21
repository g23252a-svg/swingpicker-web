"""
ml_engine.py v18.3 — Elite Ensemble AI Engine (최종 안정화)

v17.0 → v18.3 주요 변경:
  ──────────────────────────────────────────────────────────────
  1. [다중 타겟] 3%/5%/7% → High 기준 soft label (0/0.33/0.67/1.0)
  2. [회귀 손실] SmoothL1 + 이중 가중 (계단형 tier × pos_weight 적응형)
     - tier: 1+2*target (7%+에 3배 가중)
     - pos: (1-pos_ratio)/pos_ratio (양성 희소 시 자동 보정)
  3. [자신감 적응형 하이브리드] prob×w_prob + 샤프닝 퍼센타일×w_pct
     - prob_std 기반 w_pct: 0.3~0.6 (하한 0.3으로 변별력 바닥 보장)
     - 퍼센타일 ^1.2 샤프닝: 상위권 점수 차이 확대
  4. [시퀀스 40일] + 과적합 방어 (dropout/WD/ES)
  5. [v18.3] 인덱스 안전 매핑: get_indexer(method="pad")
     - anchor_date가 df에 없을 때 가장 가까운 이전 영업일로 매칭
     - 데이터 손실 최소화
  6. [하위 호환] v18.2→v18.0→v17→v15.6 다단계 폴백
  ──────────────────────────────────────────────────────────────
"""

import os, joblib, glob, re, pickle, threading
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
# v18.2 경로
MODEL_PATH       = "data/trading_model_v18_3.pth"
SCALER_PATH      = "data/trading_scaler_v18_3.pkl"
XGB_MODEL_PATH   = "data/trading_model_xgb_v18_3.pkl"
FEATURE_CACHE_PATH = "data/feature_cache_v18_3.pkl"
META_PATH        = "data/trading_meta_v18_3.json"

# 하위 호환: v18.2 → v18.0 → v17 → v15.6 순으로 폴백
FALLBACK_PATHS = [
    ("data/trading_model_v18_2.pth", "data/trading_scaler_v18_2.pkl", "data/trading_model_xgb_v18_2.pkl"),
    ("data/trading_model_v18.pth", "data/trading_scaler_v18.pkl", "data/trading_model_xgb_v18.pkl"),
    ("data/trading_model_v17.pth", "data/trading_scaler_v17.pkl", "data/trading_model_xgb_v17.pkl"),
    ("data/trading_model_v15_6_master.pth", "data/trading_scaler_v15_6_master.pkl", None),
]

# [v18.0 핵심] 시퀀스 40일 — 중기 구조 학습
SEQ_LENGTH  = 40

# [v18.0 핵심] 다중 타겟 — soft label
TARGET_TIERS = [3.0, 5.0, 7.0]   # 달성 시 0.33, 0.67, 1.0
TARGET_RET   = 3.0               # 하위 호환용 (v17 레이블 기준)

BASIC_COLS  = ["Open", "High", "Low", "Close", "Volume"]

# ====================== 피처 엔진 (16개, v17 호환) ======================
FEATURE_COLS = [
    "Log_Ret", "Volume_Norm", "Low_Trend", "Vol_Quality", "Dist_MA20",
    "RSI", "MFI", "MACD_Hist_Norm", "BB_Width", "ATR_Pct",
    "OBV_Slope", "Range_Pos", "Vol_Ratio_5", "Ret_5d", "Ret_20d",
    "Upper_Shadow_Ratio",
]

# Thread safety for model loading
_model_lock = threading.Lock()


def get_feature_cache():
    # v18.3 → v18.2 → v18 → v17 캐시 순으로 탐색
    for path in [FEATURE_CACHE_PATH,
                 "data/feature_cache_v18_2.pkl",
                 "data/feature_cache_v18.pkl",
                 "data/feature_cache_v17.pkl"]:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                continue
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
    [v17.0 호환] 16개 피처 산출
    v18에서 피처 자체는 변경하지 않음 (모델 구조만 개선)
    """
    if len(df) < 60:
        return pd.DataFrame()

    df = df.copy()
    c, h, l, o, v = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']

    # === 기존 호환 피처 ===
    df['Log_Ret'] = np.log(c / c.shift(1).replace(0, np.nan))

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

    delta = c.diff()
    up_d = delta.clip(lower=0)
    down_d = -1 * delta.clip(upper=0)
    ema_up = up_d.ewm(com=13, adjust=False).mean()
    ema_down = down_d.ewm(com=13, adjust=False).mean()
    df['RSI'] = (100 - (100 / (1 + (ema_up / ema_down.replace(0, np.nan))))) / 100.0

    # === 신규 10개 피처 ===
    tp = (h + l + c) / 3
    rmf = tp * v
    pos_flow = np.where(tp.diff() > 0, rmf, 0)
    neg_flow = np.where(tp.diff() < 0, rmf, 0)
    pos_sum = pd.Series(pos_flow, index=c.index).rolling(14).sum()
    neg_sum = pd.Series(neg_flow, index=c.index).rolling(14).sum().replace(0, 1)
    df['MFI'] = (100 - (100 / (1 + pos_sum / neg_sum))) / 100.0

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    df['MACD_Hist_Norm'] = hist / c.replace(0, np.nan) * 100

    std20 = c.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    df['BB_Width'] = (bb_upper - bb_lower) / ma20.replace(0, np.nan)

    tr = pd.concat([
        (h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df['ATR_Pct'] = atr14 / c.replace(0, np.nan)

    obv_sign = np.sign(c.diff()).fillna(0)
    obv = (obv_sign * v).cumsum()
    obv_ma5 = obv.rolling(5).mean()
    obv_ma20 = obv.rolling(20).mean()
    df['OBV_Slope'] = ((obv_ma5 - obv_ma20) / obv_ma20.abs().replace(0, np.nan)).clip(-2, 2)

    h20 = h.rolling(20).max()
    l20 = l.rolling(20).min()
    denom = (h20 - l20).replace(0, np.nan)
    df['Range_Pos'] = (c - l20) / denom

    vol_ma5 = v.rolling(5).mean().replace(0, np.nan)
    df['Vol_Ratio_5'] = np.log1p(v / vol_ma5)

    df['Ret_5d'] = c.pct_change(5)
    df['Ret_20d'] = c.pct_change(20)

    candle_range = (h - l).replace(0, np.nan)
    body_top = pd.concat([c, o], axis=1).max(axis=1)
    df['Upper_Shadow_Ratio'] = (h - body_top) / candle_range

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
    """
    [v18.0] 과적합 방어 강화
    - dropout 0.2 → 0.3 (LSTM 내부)
    - fc dropout 0.3 → 0.4
    - BatchNorm 유지
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.3)    # 0.2→0.3
        self.attention = Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),      # 0.3→0.4
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        ctx, _ = self.attention(out)
        return self.fc(ctx)


# ====================== 손실 함수 ======================

class SoftTargetLoss(nn.Module):
    """
    [v18.1] Soft Label용 회귀 손실 (SmoothL1 + 클래스 가중)

    왜 Focal이 아닌 회귀인가:
    - Focal Loss는 hard 0/1 분류에서 "어려운 샘플 집중"이 목적
    - Soft label(0/0.33/0.67/1.0)에 BCE를 쓰면 가중이 왜곡됨
      (0.33을 "33% 확률로 양성"으로 해석하여 gradient가 애매해짐)
    - SmoothL1(Huber)은 타겟이 연속값일 때 자연스럽고,
      이상치(극단 오차)에도 안정적

    pos_weight: 양성 샘플(>0)에 추가 가중치를 줘서
    "전부 0으로 예측하면 편해지는" 편향을 방지
    """
    def __init__(self, pos_ratio: float = 0.5, beta: float = 0.2):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction='none')
        # pos_weight: 양성이 적으면 가중치 높임 (Focal의 alpha 역할)
        self.pos_weight = max(1.0, min(5.0, (1 - pos_ratio) / max(pos_ratio, 0.1)))
        print(f"   📐 SoftTargetLoss: beta={beta}, "
              f"pos_weight={self.pos_weight:.2f} (pos_ratio={pos_ratio:.2%})")

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)  # 0~1 범위
        base_loss = self.smooth_l1(preds, targets)
        # [v18.3] 이중 가중: 계단형(tier) × 데이터 적응형(pos)
        #
        # tier_w: soft label 값에 비례하여 "7%+ 종목"을 더 강하게 학습
        #   target=0.0→1.0, 0.33→1.66, 0.67→2.34, 1.0→3.0
        #
        # pos_w: 양성 비율이 적으면 양성 전체에 가중치를 높여서
        #   "전부 0으로 예측하면 편해지는" 편향 방지 (Focal alpha 역할)
        #
        # 곱하면: 양성이 희소할수록 + 강한 종목일수록 loss 기여 ↑
        tier_w = 1.0 + 2.0 * targets
        pos_w = torch.where(targets > 0, self.pos_weight, 1.0)
        weights = (tier_w * pos_w).clamp(max=10.0)  # 폭주 방지 (양성 극단 희소 시)
        return (base_loss * weights).mean()


# ====================== 모델 로딩 (Thread-Safe) ======================

_loaded_lstm_model = None
_loaded_scaler = None
_loaded_xgb_model = None
_loaded_seq_length = SEQ_LENGTH   # 로드된 모델의 시퀀스 길이 추적


def load_model():
    """
    [v18.0] Thread-safe 모델 로딩 + 다단계 폴백
    v18 → v17 → v15.6 순으로 탐색
    """
    global _loaded_lstm_model, _loaded_scaler, _loaded_xgb_model, _loaded_seq_length

    with _model_lock:
        # 이미 로드됐으면 스킵
        if _loaded_lstm_model is not None and _loaded_scaler is not None:
            return

        # 탐색 순서: v18 → v17 → v15.6
        candidates = [
            (MODEL_PATH, SCALER_PATH, XGB_MODEL_PATH, SEQ_LENGTH),
        ]
        for m, s, x in FALLBACK_PATHS:
            # v17은 SEQ=20, v15.6도 SEQ=20
            candidates.append((m, s, x, 20))

        for model_path, scaler_path, xgb_path, seq_len in candidates:
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                continue

            try:
                scaler = joblib.load(scaler_path)
                in_dim = getattr(scaler, 'n_features_in_', len(FEATURE_COLS))

                device = torch.device('cpu')
                model = TradingAttnLSTM(in_dim, 64, 2, 1).to(device)
                state = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(state)
                model.eval()

                _loaded_lstm_model = model
                _loaded_scaler = scaler
                _loaded_seq_length = seq_len
                print(f"✅ [ML] LSTM 로드: {model_path} (features={in_dim}, seq={seq_len})")

                # XGBoost (있으면)
                if xgb_path and XGB_OK and os.path.exists(xgb_path):
                    try:
                        _loaded_xgb_model = joblib.load(xgb_path)
                        print(f"✅ [ML] XGBoost 로드: {xgb_path}")
                    except Exception as e:
                        print(f"⚠️ [ML] XGBoost 로드 실패: {e}")
                        _loaded_xgb_model = None

                return  # 성공하면 즉시 탈출

            except Exception as e:
                print(f"⚠️ [ML] {model_path} 로드 실패: {e}")
                continue

        print("⚠️ [ML] 사용 가능한 모델이 없습니다.")


# ====================== 데이터셋 빌드 ======================

def extract_date(path):
    m = re.search(r'(\d{8})', os.path.basename(path))
    return m.group(1) if m else "00000000"


def _compute_soft_label(entry_price, future_highs):
    """
    [v18.1] 다중 타겟 Soft Label (High 기준)

    왜 Close가 아닌 High인가:
    - Close max: 장중에 7% 갔다가 종가에 2%로 내려와도 "미달" → 모델이 놓침
    - High max: 실제로 TP(익절가)에 도달했는지를 판별 → 실전 TP/SL과 정합
    - 추천 시스템이 "이 종목이 N% 갈 수 있는가"를 판별하는 것이므로
      장중 최고점 기준이 목적에 부합

    레이블 맵:
    - 7%+ 달성 → 1.0  (최강)
    - 5%+ 달성 → 0.67 (강)
    - 3%+ 달성 → 0.33 (보통)
    - 미달     → 0.0
    """
    if future_highs.empty or entry_price <= 0:
        return None

    max_high_ret = (future_highs.max() / entry_price - 1) * 100

    if max_high_ret >= TARGET_TIERS[2]:   # 7%+
        return 1.0
    elif max_high_ret >= TARGET_TIERS[1]: # 5%+
        return 0.67
    elif max_high_ret >= TARGET_TIERS[0]: # 3%+
        return 0.33
    else:
        return 0.0


def build_master_dataset(data_dir="data"):
    """
    [v18.2] High 기준 Soft Label + SEQ_LENGTH=40 + 인덱스 정렬 보장

    변경점 (v17 → v18.2):
    - 레이블: 0/1 binary → 0.0/0.33/0.67/1.0 soft label
    - 기준: Close max → High max (TP 도달 여부와 정합)
    - 시퀀스: 20→40일 (중기 구조 학습)
    - [v18.2 치명 Fix] df_feat 인덱스 → df 인덱스 정확 매핑
      (rolling/NaN drop으로 앞부분이 잘린 df_feat의 i와 df의 i가 달라지는 문제 해결)
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
                    if len(df_feat) < SEQ_LENGTH + 1:
                        continue

                    # [v18.2] 시퀀스: anchor_date 포함 (i-SEQ+1 ~ i+1)
                    for i in range(SEQ_LENGTH - 1, len(df_feat)):
                        anchor_date = df_feat.index[i]
                        seq = df_feat.iloc[i - SEQ_LENGTH + 1:i + 1].values  # anchor 포함

                        if len(seq) != SEQ_LENGTH:
                            continue

                        # [v18.3] anchor_date → df 위치 안전 매핑
                        # get_loc은 정확 매치만 → 영업일/타임존 차이로 KeyError 다발
                        # get_indexer(method="pad")는 "이전 가장 가까운 날짜"로 매칭
                        pos = df.index.get_indexer([anchor_date], method="pad")
                        anchor_pos = int(pos[0])
                        if anchor_pos < 0:
                            continue

                        entry_idx = anchor_pos + 1
                        if entry_idx + 5 > len(df):
                            continue

                        entry_price = float(df.iloc[entry_idx]['Open'])
                        if entry_price <= 0:
                            continue

                        # [v18.1] High 기준 Soft Label 산출
                        future_highs = df.iloc[entry_idx:entry_idx + 5]['High']
                        label = _compute_soft_label(entry_price, future_highs)
                        if label is None:
                            continue

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
    y_train = train_df['y'].values.astype(np.float32)   # soft label은 float
    y_val = val_df['y'].values.astype(np.float32)

    # 스케일러 피팅
    scaler = StandardScaler()
    n_feat = X_train.shape[2]
    scaler.fit(X_train.reshape(-1, n_feat))
    joblib.dump(scaler, SCALER_PATH)

    X_train_s = scaler.transform(X_train.reshape(-1, n_feat)).reshape(-1, SEQ_LENGTH, n_feat)
    X_val_s = scaler.transform(X_val.reshape(-1, n_feat)).reshape(-1, SEQ_LENGTH, n_feat)

    # [v18.0] soft label 통계 (양성 = label > 0)
    pos_ratio = (y_train > 0).mean()
    tier_dist = {
        '0 (미달)': (y_train == 0.0).mean(),
        '0.33 (3%+)': ((y_train > 0) & (y_train <= 0.34)).mean(),
        '0.67 (5%+)': ((y_train > 0.34) & (y_train <= 0.68)).mean(),
        '1.0 (7%+)': (y_train > 0.68).mean(),
    }
    print(f"📊 [ML] Soft Label 분포 (train={len(y_train)}, val={len(y_val)}):")
    for k, v in tier_dist.items():
        print(f"   {k}: {v:.1%}")
    print(f"   양성 비율(>0): {pos_ratio:.2%}")

    return X_train_s, y_train, X_val_s, y_val, val_df[['date', 'code']], n_feat, pos_ratio


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
    """
    [v18.1] LSTM + XGBoost 앙상블 학습

    개선점:
    - SoftTargetLoss (회귀 손실 + 동적 pos_weight)
    - Soft label 다중 타겟 (0/0.33/0.67/1.0)
    - Early stopping (5에폭 patience)
    - weight_decay 1e-4 → 5e-4
    """
    if is_trained_today(force):
        print("✅ [SKIP] 오늘 이미 v18.3 모델 학습이 완료되었습니다.")
        return

    print("🤖 AI 모델 v18.3 학습 시작 (변별력 강화 + 인덱스 정렬 + 과적합 방어)...")

    data = build_master_dataset()
    if data is None:
        print("⚠️ [ML] 학습 데이터 부족으로 중단합니다.")
        return

    X_tr, y_tr, X_val, y_val, meta_val, in_dim, pos_ratio = data

    # ============= (1) LSTM 학습 =============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingAttnLSTM(in_dim, 64, 2, 1).to(device)

    # [v18.1 핵심] Soft Label → 회귀 손실 (SmoothL1 + pos_weight)
    criterion = SoftTargetLoss(pos_ratio=pos_ratio, beta=0.2)

    # [v18.0] weight_decay 강화 (과적합 방어)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_kpi = 0.0
    patience_counter = 0
    PATIENCE = 5           # [v18.0] Early stopping patience
    MAX_EPOCHS = 40        # 30→40 (early stopping이 방어해주므로 여유 확보)

    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=128)

    # [v18.0] validation용 binary label (AUC 계산에 필요)
    y_val_binary = (y_val > 0).astype(np.float32)  # soft label → 0/1

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for b_X, b_y in DataLoader(StockDataset(X_tr, y_tr), batch_size=128, shuffle=True):
            b_X = b_X.to(device)
            b_y = b_y.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
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
            auc = roc_auc_score(y_val_binary, all_probs)
        except ValueError:
            auc = 0.5

        # 다중 구간 적중률 (상위 20/50/100)
        val_res = pd.DataFrame({'prob': all_probs, 'target': y_val_binary})
        hit_rates = []
        for k in [20, 50, 100]:
            if len(val_res) >= k:
                hit_rates.append(
                    val_res.sort_values('prob', ascending=False).head(k)['target'].mean()
                )
        avg_precision = np.mean(hit_rates) if hit_rates else 0.0

        # [v18.1] 강등급(5%+) 적중률 — soft label 기준 명확화
        y_val_strong = (y_val >= 0.67).astype(np.float32)  # 0.67="5%+", 1.0="7%+"
        val_res_strong = pd.DataFrame({'prob': all_probs, 'target': y_val_strong})
        strong_hit = 0.0
        if len(val_res_strong) >= 20:
            strong_hit = val_res_strong.sort_values('prob', ascending=False).head(20)['target'].mean()

        # [v18.2 보너스] Top-20 soft label 평균 (추천 시스템 지표)
        # → 상위 20개 종목의 실제 soft label 평균이 높을수록 좋은 모델
        val_res_soft = pd.DataFrame({'prob': all_probs, 'soft_y': y_val})
        top20_soft_avg = 0.0
        if len(val_res_soft) >= 20:
            top20_soft_avg = val_res_soft.sort_values('prob', ascending=False) \
                .head(20)['soft_y'].mean()

        # [v18.2] KPI: AUC 20% + 이진 적중률 25% + 강등급 적중률 25% + Top-20 soft 평균 30%
        # - AUC: 회귀 타겟과 결이 다르므로 비중 축소 (40→20%)
        # - top20_soft_avg: "상위 추천 종목이 실제로 얼마나 강한가"를 직접 측정
        kpi = (0.20 * auc) + (0.25 * avg_precision) + (0.25 * strong_hit) + (0.30 * top20_soft_avg)

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 5 == 0 or kpi > best_kpi:
            print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f} | KPI: {kpi:.4f} "
                  f"(AUC: {auc:.3f}, Hit: {avg_precision:.2%}, "
                  f"Strong: {strong_hit:.2%}, SoftTop20: {top20_soft_avg:.3f})")

        if kpi > best_kpi:
            best_kpi = kpi
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1

        # [v18.0] Early Stopping
        if patience_counter >= PATIENCE:
            print(f"🛑 [Early Stop] {PATIENCE}에폭 연속 개선 없음. "
                  f"Best KPI: {best_kpi:.4f} (epoch {epoch - PATIENCE})")
            break

    print(f"✅ [LSTM] 학습 완료 (Best KPI: {best_kpi:.4f})")

    # ============= (2) XGBoost 학습 =============
    if XGB_OK:
        print("🌲 XGBoost 앙상블 학습 시작...")
        X_tr_xgb = X_tr[:, -1, :]   # 마지막 시점 피처
        X_val_xgb = X_val[:, -1, :]

        # [v18.0] soft label → binary로 변환 (XGBoost는 분류)
        y_tr_binary = (y_tr > 0).astype(np.float32)

        pos_count = max(int(y_tr_binary.sum()), 1)
        neg_count = len(y_tr_binary) - pos_count

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
            X_tr_xgb, y_tr_binary,
            eval_set=[(X_val_xgb, y_val_binary)],
            verbose=False
        )
        joblib.dump(xgb_model, XGB_MODEL_PATH)

        xgb_probs = xgb_model.predict_proba(X_val_xgb)[:, 1]
        try:
            xgb_auc = roc_auc_score(y_val_binary, xgb_probs)
        except ValueError:
            xgb_auc = 0.5
        print(f"✅ [XGBoost] 학습 완료 (AUC: {xgb_auc:.3f})")

        importance = dict(zip(FEATURE_COLS[:in_dim], xgb_model.feature_importances_))
        top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top 피처: {', '.join(f'{k}({v:.2f})' for k, v in top5)}")

    # 메타데이터 저장 (하위 호환용)
    _save_meta(in_dim, pos_ratio, best_kpi)

    # 전역 캐시에 로드
    _loaded_lstm_model = None  # 강제 재로드
    _loaded_scaler = None
    load_model()
    print("✅ [ML] v18.3 전체 학습 파이프라인 완료!")


def _save_meta(in_dim, pos_ratio, best_kpi):
    """학습 메타데이터 저장 (디버깅 및 하위 호환 판별용)"""
    import json
    meta = {
        "version": "18.3",
        "seq_length": SEQ_LENGTH,
        "n_features": in_dim,
        "feature_cols": FEATURE_COLS[:in_dim],
        "target_tiers": TARGET_TIERS,
        "pos_ratio": round(float(pos_ratio), 4),
        "best_kpi": round(float(best_kpi), 4),
        "trained_at": datetime.now().isoformat(),
    }
    try:
        with open(META_PATH, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ====================== 추론 ======================

def apply_ml_score(current_df, full_ohlcv_map):
    """
    [v18.0] 하이브리드 스코어링 — 확률 50% + 퍼센타일 50%

    핵심 변경:
    - 기존: score = prob * 100 → 60~80대 쏠림, 변별 불가
    - 변경: score = 0.5 * (prob*100) + 0.5 * (rank_percentile*100)
      → 절대 확률: 시장 전체가 위험할 때 모두 낮아짐 (과신 방지)
      → 상대 순위: 구조적으로 상위/하위가 항상 벌어짐 (변별 보장)
    """
    global _loaded_lstm_model, _loaded_scaler, _loaded_xgb_model, _loaded_seq_length

    # Thread-safe 모델 로딩
    if _loaded_lstm_model is None or _loaded_scaler is None:
        load_model()

    if _loaded_lstm_model is None or _loaded_scaler is None:
        print("⚠️ [ML] 모델 파일 없음. ML_SCORE=0 으로 진행.")
        current_df["ML_SCORE"] = 0.0
        return current_df

    # 현재 로드된 모델의 시퀀스 길이 사용
    seq_len = _loaded_seq_length

    cache = get_feature_cache()
    target_codes = current_df["종목코드"].unique()
    valid_inputs, codes = [], []
    new_cache_count = 0

    for code in target_codes:
        if code not in full_ohlcv_map:
            continue

        raw_df = full_ohlcv_map[code]

        # [v18.3 Fix] clean 후 last_date 잡기 — raw_df는 미정렬/중복 가능
        df = clean_ohlcv(raw_df)
        if df.empty:
            continue
        last_date = str(df.index[-1])

        # 캐시 키에 seq_length 포함 (v17/v18 공존 대응)
        cache_key = f"{code}_s{seq_len}"

        if cache_key in cache and cache[cache_key].get('date') == last_date:
            valid_inputs.append(cache[cache_key]['seq'])
            codes.append(code)
        else:
            df_feat = add_technical_features(df)
            if len(df_feat) >= seq_len:
                seq = df_feat.iloc[-seq_len:].values
                valid_inputs.append(seq)
                codes.append(code)
                cache[cache_key] = {'date': last_date, 'seq': seq}
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
        print(f"⚠️ [ML] 피처 차원 불일치 (데이터={n_feat}, 모델={scaler_dim}). ML_SCORE=0 폴백.")
        current_df["ML_SCORE"] = 0.0
        return current_df

    try:
        X_scaled = _loaded_scaler.transform(
            X_raw.reshape(-1, n_feat)
        ).reshape(-1, seq_len, n_feat)
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
            X_xgb = X_scaled[:, -1, :]
            xgb_probs = _loaded_xgb_model.predict_proba(X_xgb)[:, 1]
            final_probs = lstm_probs * 0.6 + xgb_probs * 0.4
        except Exception as e:
            print(f"⚠️ [ML] XGBoost 추론 실패, LSTM 단독: {e}")

    # ─────────────────────────────────────────────
    # [v18.1 핵심] 하이브리드 스코어링 (자신감 적응형)
    # ─────────────────────────────────────────────
    prob_scores = final_probs * 100.0   # 절대 확률 (0~100)

    # 퍼센타일 (상대 순위, 0~100)
    rank_series = pd.Series(final_probs).rank(pct=True) * 100.0
    pct_scores = rank_series.values

    # [v18.3] 모델 자신감(prob 분산) 기반 동적 가중치
    #
    # 설계 의도(변별력 vs 과신 균형):
    # - std 작으면: 모델이 구분 못함 → 확률이 평준화되어 있음
    #   → 퍼센타일을 약간 줄이되(과신 방지), 완전히 죽이지는 않음(변별 유지)
    #   → w_pct 하한 0.3 (기존 0.2에서 상향)
    # - std 크면: 모델이 잘 구분함 → 퍼센타일로 추가 변별해도 안전
    prob_std = float(np.std(final_probs))
    w_pct = float(np.clip((prob_std - 0.05) / 0.10, 0.3, 0.6))  # 하한 0.3
    w_prob = 1.0 - w_pct

    # [v18.3] 퍼센타일 상위 샤프닝 (지수 1.2)
    # → 상위권은 점수 차이가 더 벌어지고, 하위권은 뭉침
    # → "변별력"이 필요한 곳(상위 추천 후보)에서 더 효과적
    pct_raw = rank_series.values  # 0~100
    pct_scores = (pct_raw / 100.0) ** 1.2 * 100.0  # 상위 벌림, 0~100 스케일 유지

    hybrid_scores = (w_prob * prob_scores + w_pct * pct_scores).round(1)

    score_map = dict(zip(codes, hybrid_scores))
    current_df["ML_SCORE"] = current_df["종목코드"].map(score_map).fillna(0.0)

    # --- 진단 로그 ---
    if codes:
        all_scores = np.array(list(score_map.values()))
        prob_mean = prob_scores.mean()
        print(f"🧠 [ML] 분포: prob_avg={prob_mean:.1f}, prob_std={prob_std:.3f}, "
              f"w_prob={w_prob:.2f}/w_pct={w_pct:.2f}")
        print(f"   hybrid: min={all_scores.min():.1f} / "
              f"med={np.median(all_scores):.1f} / "
              f"max={all_scores.max():.1f}")

        # 상위 5개
        top5 = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"🧠 [ML] Top5: {', '.join(f'{c}({s})' for c, s in top5)}")

        # [v18.0] 변별력 진단: 상위/하위 20%의 prob 차이
        n = len(all_scores)
        if n >= 10:
            top20_prob = np.sort(prob_scores)[-max(n//5, 1):].mean()
            bot20_prob = np.sort(prob_scores)[:max(n//5, 1)].mean()
            spread = top20_prob - bot20_prob
            print(f"   📐 변별력: top20%_prob={top20_prob:.1f}, "
                  f"bot20%_prob={bot20_prob:.1f}, spread={spread:.1f}")

    return current_df
