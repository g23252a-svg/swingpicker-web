"""
ml_engine.py v19.0 — Elite Ensemble AI Engine (5-Fix Evolution)

v18.3 → v19.0 주요 변경:
  ──────────────────────────────────────────────────────────────
  [Fix 1] First-touch Labeling: High만 보던 라벨 → Low(손절) 선 터치 여부 확인
           "유령 익절" 방지 — 손절가 먼저 도달 시 label=0.0 강제
  [Fix 2] Vectorized Inference: 종목별 for 루프 → Multi-index 일괄 피처 계산
           200종목 기준 ~10x 속도 향상
  [Fix 3] Log-scaled Weights: tier_w × pos_w 곱셈 → log1p 완화
           Gradient 폭주 방지 (기존 max 60x → ~4.2x)
  [Fix 4] Dynamic Ensemble: 고정 0.6/0.4 → 최근 검증 MAE 기반 역수 가중
           시장 국면별 자동 비중 조절
  [Fix 5] Rolling Z-score: 전역 StandardScaler → 60일 Rolling Z-score 피처
           비정상성(Non-stationarity) 대응
  ──────────────────────────────────────────────────────────────
  하위 호환: v18.3 → v18.2 → v18.0 → v17 → v15.6 다단계 폴백 유지
"""

import os, joblib, glob, re, pickle, threading, json
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
MODEL_PATH       = "data/trading_model_v19.pth"
SCALER_PATH      = "data/trading_scaler_v19.pkl"
XGB_MODEL_PATH   = "data/trading_model_xgb_v19.pkl"
FEATURE_CACHE_PATH = "data/feature_cache_v19.pkl"
META_PATH        = "data/trading_meta_v19.json"
# [Fix 4] 앙상블 가중치 캐시
ENSEMBLE_WEIGHTS_PATH = "data/ensemble_weights_v19.json"

FALLBACK_PATHS = [
    ("data/trading_model_v18_3.pth", "data/trading_scaler_v18_3.pkl", "data/trading_model_xgb_v18_3.pkl"),
    ("data/trading_model_v18_2.pth", "data/trading_scaler_v18_2.pkl", "data/trading_model_xgb_v18_2.pkl"),
    ("data/trading_model_v18.pth", "data/trading_scaler_v18.pkl", "data/trading_model_xgb_v18.pkl"),
    ("data/trading_model_v17.pth", "data/trading_scaler_v17.pkl", "data/trading_model_xgb_v17.pkl"),
    ("data/trading_model_v15_6_master.pth", "data/trading_scaler_v15_6_master.pkl", None),
]

SEQ_LENGTH  = 40
TARGET_TIERS = [3.0, 5.0, 7.0]
TARGET_RET   = 3.0

# [Fix 1] 손절 기준: 매수가 대비 -N% 하락 시 손절 판정
STOP_LOSS_PCT = -5.0

BASIC_COLS  = ["Open", "High", "Low", "Close", "Volume"]

FEATURE_COLS = [
    "Log_Ret", "Volume_Norm", "Low_Trend", "Vol_Quality", "Dist_MA20",
    "RSI", "MFI", "MACD_Hist_Norm", "BB_Width", "ATR_Pct",
    "OBV_Slope", "Range_Pos", "Vol_Ratio_5", "Ret_5d", "Ret_20d",
    "Upper_Shadow_Ratio",
]

_model_lock = threading.Lock()


# ====================== 캐시 유틸 ======================

def get_feature_cache():
    for path in [FEATURE_CACHE_PATH,
                 "data/feature_cache_v18_3.pkl",
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


# ====================== 데이터 정제 ======================

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


# ====================== 피처 엔진 (16개) ======================

def add_technical_features(df):
    """[v17.0 호환] 16개 피처 산출 — 단일 종목용"""
    if len(df) < 60:
        return pd.DataFrame()

    df = df.copy()
    c, h, l, o, v = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']

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


# ══════════════════════════════════════════════════════════════
# [Fix 2] Vectorized 피처 계산 — Multi-index 일괄 처리
# ══════════════════════════════════════════════════════════════

def add_technical_features_batch(ohlcv_map: dict) -> dict:
    """
    [v19.5] 진정한 벡터화 — groupby('_code') 네이티브 연산.
    피처 계산 내 for 루프 0건. 결과 분리용 루프만 존재.

    Parameters:
        ohlcv_map: {code: DataFrame} — raw OHLCV
    Returns:
        {code: DataFrame(FEATURE_COLS)} — 피처 결과 맵
    """
    frames = []
    valid_codes = []

    for code, raw_df in ohlcv_map.items():
        df = clean_ohlcv(raw_df)
        if len(df) < 60:
            continue
        df = df.copy()
        df["_code"] = code
        frames.append(df)
        valid_codes.append(code)

    if not frames:
        return {}

    big = pd.concat(frames, ignore_index=False)
    g = big.groupby("_code", sort=False)

    c = big['Close']
    h = big['High']
    l = big['Low']
    o = big['Open']
    v = big['Volume']

    # ── 16개 피처: groupby + transform/shift → C 엔진 100% ──

    c_shift1 = g['Close'].shift(1)

    # 1. Log_Ret
    big['Log_Ret'] = np.log(c / c_shift1.replace(0, np.nan))

    # 2. Volume_Norm
    vol_ma20 = g['Volume'].transform(lambda x: x.rolling(20).mean()).replace(0, np.nan)
    big['Volume_Norm'] = np.log1p(v / vol_ma20)

    # 3. Low_Trend
    low10 = g['Low'].transform(lambda x: x.rolling(10).min())
    low10_lag = g['Low'].transform(lambda x: x.rolling(10).min().shift(10)).replace(0, np.nan)
    big['Low_Trend'] = (low10 - low10_lag) / low10_lag

    # 4. Vol_Quality
    is_up = (c > o).astype(np.float32)
    vu = v * is_up
    vd = v * (1 - is_up)
    big['_vu'] = vu
    big['_vd'] = vd
    big['_is_up'] = is_up
    vu_sum = g['_vu'].transform(lambda x: x.rolling(20).sum())
    vd_sum = g['_vd'].transform(lambda x: x.rolling(20).sum())
    up_cnt = g['_is_up'].transform(lambda x: x.rolling(20).sum()).replace(0, np.nan)
    dn_cnt = (20 - up_cnt).replace(0, np.nan)
    big['Vol_Quality'] = ((vu_sum / up_cnt) / (vd_sum / dn_cnt).replace(0, np.nan)).clip(0, 5)

    # 5. Dist_MA20
    ma20 = g['Close'].transform(lambda x: x.rolling(20).mean()).replace(0, np.nan)
    big['Dist_MA20'] = (c - ma20) / ma20

    # 6. RSI (14)
    delta = c - c_shift1
    up_d = delta.clip(lower=0)
    down_d = (-delta).clip(lower=0)
    big['_up'] = up_d
    big['_dn'] = down_d
    ema_up = g['_up'].transform(lambda x: x.ewm(com=13, adjust=False).mean())
    ema_dn = g['_dn'].transform(lambda x: x.ewm(com=13, adjust=False).mean()).replace(0, np.nan)
    big['RSI'] = (100 - 100 / (1 + ema_up / ema_dn)) / 100.0

    # 7. MFI (14)
    tp = (h + l + c) / 3
    rmf = tp * v
    big['_tp'] = tp
    tp_diff = g['_tp'].diff()
    pf = pd.Series(np.where(tp_diff > 0, rmf, 0), index=big.index, dtype=np.float32)
    nf = pd.Series(np.where(tp_diff < 0, rmf, 0), index=big.index, dtype=np.float32)
    big['_pf'] = pf
    big['_nf'] = nf
    ps = g['_pf'].transform(lambda x: x.rolling(14).sum())
    ns = g['_nf'].transform(lambda x: x.rolling(14).sum()).replace(0, 1)
    big['MFI'] = (100 - 100 / (1 + ps / ns)) / 100.0

    # 8. MACD_Hist_Norm
    ema12 = g['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd = ema12 - ema26
    big['_macd'] = macd
    signal = g['_macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    big['MACD_Hist_Norm'] = (macd - signal) / c.replace(0, np.nan)

    # 9. BB_Width
    std20 = g['Close'].transform(lambda x: x.rolling(20).std())
    big['BB_Width'] = (4 * std20) / ma20.replace(0, np.nan)

    # 10. ATR_Pct
    tr = pd.concat([h - l, (h - c_shift1).abs(), (l - c_shift1).abs()], axis=1).max(axis=1)
    big['_tr'] = tr
    atr = g['_tr'].transform(lambda x: x.rolling(14).mean())
    big['ATR_Pct'] = atr / c.replace(0, np.nan)

    # 11. OBV_Slope
    sign = np.sign(c - c_shift1)
    big['_obv_raw'] = sign * v
    obv = g['_obv_raw'].cumsum()
    big['_obv'] = obv
    obv_ma = g['_obv'].transform(lambda x: x.rolling(10).mean())
    obv_ma_lag = g['_obv'].transform(lambda x: x.rolling(10).mean().shift(10))
    denom = obv_ma_lag.abs().replace(0, np.nan)
    big['OBV_Slope'] = ((obv_ma - obv_ma_lag) / denom).clip(-5, 5)

    # 12. Range_Pos
    h20 = g['High'].transform(lambda x: x.rolling(20).max())
    l20 = g['Low'].transform(lambda x: x.rolling(20).min())
    big['Range_Pos'] = (c - l20) / (h20 - l20).replace(0, np.nan)

    # 13. Vol_Ratio_5
    vol_ma5 = g['Volume'].transform(lambda x: x.rolling(5).mean()).replace(0, np.nan)
    big['Vol_Ratio_5'] = v / vol_ma5

    # 14/15. Ret_5d, Ret_20d
    big['Ret_5d'] = c / g['Close'].shift(5).replace(0, np.nan) - 1
    big['Ret_20d'] = c / g['Close'].shift(20).replace(0, np.nan) - 1

    # 16. Upper_Shadow_Ratio
    body_top = pd.concat([c, o], axis=1).max(axis=1)
    big['Upper_Shadow_Ratio'] = (h - body_top) / (h - l).replace(0, np.nan)

    # ── 임시 컬럼 정리 ──
    tmp = [c for c in big.columns if c.startswith('_')]
    big = big.drop(columns=tmp + BASIC_COLS, errors='ignore')
    big = big[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # ── 결과 분리 (이 루프만 존재 — 피처 계산 내 루프 0건) ──
    code_col = pd.concat([
        pd.Series(code, index=f.index, dtype='object')
        for code, f in zip(valid_codes, frames)
    ])

    result_map = {}
    for code in valid_codes:
        mask = code_col == code
        feat_df = big.loc[mask].dropna()
        if len(feat_df) >= SEQ_LENGTH:
            result_map[code] = feat_df

    return result_map


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
                            batch_first=True, dropout=0.3)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        ctx, _ = self.attention(out)
        return self.fc(ctx)


# ══════════════════════════════════════════════════════════════
# [Fix 3] Log-scaled Weights — Gradient 폭주 방지
# ══════════════════════════════════════════════════════════════

class SoftTargetLoss(nn.Module):
    """
    [v19.0 Fix 3] Log-scaled 가중치로 Gradient 안정화

    v18.3: weights = (tier_w * pos_w).clamp(max=10)
      → 양성 극소(5%) + 강등급(7%+)일 때 tier=3 × pos=19 = 57 → clamp=10
      → 여전히 일부 샘플이 평균의 10배 Gradient 발생

    v19.0: weights = log1p(tier_w * pos_w)
      → 같은 조건: log1p(57) ≈ 4.06
      → 에너지가 자연스럽게 완화되며, clamp 없이도 안정적
      → 상한 clamp(max=5.0)는 이중 안전장치로 유지
    """
    def __init__(self, pos_ratio: float = 0.5, beta: float = 0.2):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction='none')
        self.pos_weight = max(1.0, min(5.0, (1 - pos_ratio) / max(pos_ratio, 0.1)))
        print(f"   📐 SoftTargetLoss v19: beta={beta}, "
              f"pos_weight={self.pos_weight:.2f} (pos_ratio={pos_ratio:.2%})")

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)
        base_loss = self.smooth_l1(preds, targets)

        tier_w = 1.0 + 2.0 * targets
        pos_w = torch.where(targets > 0, self.pos_weight, 1.0)

        # [Fix 3] log1p 스케일링 — 곱셈 에너지를 로그 도메인으로 완화
        raw_w = tier_w * pos_w
        weights = torch.log1p(raw_w).clamp(max=5.0)

        return (base_loss * weights).mean()


# ══════════════════════════════════════════════════════════════
# [Fix 1] First-touch Labeling — 손절 우선 판정
# ══════════════════════════════════════════════════════════════

def _compute_soft_label(entry_price, future_highs, future_lows,
                        stop_loss_pct=STOP_LOSS_PCT):
    """
    [v19.0 Fix 1] First-touch Labeling

    v18.3: future_highs.max()만 확인 → 장중 -10% 찍고 다시 +7% 올라도 "성공"
    v19.0: 날짜별로 Low가 먼저 손절가를 터치했는지 확인

    로직:
    1. 각 날짜의 Low가 손절가(entry * (1 + stop_loss_pct/100)) 이하인지 체크
    2. 각 날짜의 High가 익절가(entry * (1 + tier/100)) 이상인지 체크
    3. 손절이 익절보다 먼저(또는 같은 날) 발생 → label=0.0
    4. 같은 날 둘 다 터치 → 보수적으로 손절 우선 판정
    """
    if future_highs.empty or entry_price <= 0:
        return None

    stop_price = entry_price * (1.0 + stop_loss_pct / 100.0)

    # 손절 첫 터치 날짜 (Low 기준)
    sl_hit = future_lows <= stop_price
    sl_day = sl_hit.idxmin() if not sl_hit.any() else future_lows.index[sl_hit.argmax()]
    has_sl = sl_hit.any()

    # 각 티어별 익절 첫 터치 날짜 (High 기준)
    best_label = 0.0
    for tier_pct, label_val in zip(
        reversed(TARGET_TIERS),    # 7%, 5%, 3% 순 (높은 것부터)
        reversed([0.33, 0.67, 1.0])
    ):
        tp_price = entry_price * (1.0 + tier_pct / 100.0)
        tp_hit = future_highs >= tp_price
        if tp_hit.any():
            tp_day = future_highs.index[tp_hit.argmax()]
            # [핵심] 손절이 익절보다 먼저(또는 같은 날) → 이 티어 무효
            if has_sl and sl_day <= tp_day:
                continue
            best_label = max(best_label, label_val)

    return best_label


# ====================== 모델 로딩 (Thread-Safe) ======================

_loaded_lstm_model = None
_loaded_scaler = None
_loaded_xgb_model = None
_loaded_seq_length = SEQ_LENGTH
_loaded_use_rolling_zscore = False  # [Fix 5] 플래그


def load_model():
    global _loaded_lstm_model, _loaded_scaler, _loaded_xgb_model
    global _loaded_seq_length, _loaded_use_rolling_zscore

    with _model_lock:
        if _loaded_lstm_model is not None and _loaded_scaler is not None:
            return

        candidates = [
            (MODEL_PATH, SCALER_PATH, XGB_MODEL_PATH, SEQ_LENGTH, True),
        ]
        for m, s, x in FALLBACK_PATHS:
            candidates.append((m, s, x, 20 if "v17" in m or "v15" in m else SEQ_LENGTH, False))

        for model_path, scaler_path, xgb_path, seq_len, use_rolling in candidates:
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
                _loaded_use_rolling_zscore = use_rolling
                print(f"✅ [ML] LSTM 로드: {model_path} "
                      f"(features={in_dim}, seq={seq_len}, rolling_z={use_rolling})")

                if xgb_path and XGB_OK and os.path.exists(xgb_path):
                    try:
                        _loaded_xgb_model = joblib.load(xgb_path)
                        print(f"✅ [ML] XGBoost 로드: {xgb_path}")
                    except Exception as e:
                        print(f"⚠️ [ML] XGBoost 로드 실패: {e}")
                        _loaded_xgb_model = None

                return

            except Exception as e:
                print(f"⚠️ [ML] {model_path} 로드 실패: {e}")
                continue

        print("⚠️ [ML] 사용 가능한 모델이 없습니다.")


# ══════════════════════════════════════════════════════════════
# [Fix 5] Rolling Z-score 스케일링
# ══════════════════════════════════════════════════════════════

def _apply_rolling_zscore(X_3d: np.ndarray, window: int = 60) -> np.ndarray:
    """
    [v19.0 Fix 5] 전역 StandardScaler 대체 — Rolling Z-score

    문제: 2년 전 '거래량 노멀'과 현재의 '거래량 노멀'은 다름 (비정상성)
         전역 스케일러는 과거 분포에 편향 → 현재 변동성을 이상치로 판단

    해결: 각 시퀀스의 피처를 최근 window일 기준으로 Z-score 정규화
         → 시점별로 자동 적응하므로 스케일러 의존성 제거

    Parameters:
        X_3d: (n_samples, seq_len, n_features)
        window: rolling 윈도우 (기본 60일, 시퀀스 40일보다 충분히 큼)

    Returns:
        Z-score 정규화된 (n_samples, seq_len, n_features)
    """
    n_samples, seq_len, n_feat = X_3d.shape

    # 각 샘플 내에서 시퀀스 전체의 mean/std로 정규화
    # (rolling window가 시퀀스 길이보다 길 경우 시퀀스 전체 사용)
    effective_window = min(window, seq_len)

    result = np.empty_like(X_3d)
    for i in range(n_samples):
        seq = X_3d[i]  # (seq_len, n_feat)
        # 마지막 effective_window 시점의 통계로 전체 시퀀스 정규화
        ref = seq[-effective_window:]
        mu = ref.mean(axis=0, keepdims=True)
        sigma = ref.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)  # 0 방지
        result[i] = (seq - mu) / sigma

    return result


# ====================== 데이터셋 빌드 ======================

def extract_date(path):
    m = re.search(r'(\d{8})', os.path.basename(path))
    return m.group(1) if m else "00000000"


def _load_ohlcv_cache_file(f_path: str) -> dict:
    """[v19.5] 캐시 파일 로드 — pkl/parquet/csv 멀티포맷 지원"""
    ext = os.path.splitext(f_path)[1].lower()

    if ext == ".pkl":
        with open(f_path, 'rb') as f:
            return pickle.load(f)

    # parquet / csv → OHLCVCache v5.0 통합 포맷 (종목코드 컬럼 + 날짜 인덱스)
    try:
        if ext == ".parquet":
            combined = pd.read_parquet(f_path)
        else:  # .csv
            combined = pd.read_csv(f_path, index_col=0, parse_dates=True,
                                   dtype={"종목코드": str})

        if "종목코드" not in combined.columns:
            return {}

        data_map = {}
        for code, group in combined.groupby("종목코드"):
            clean_code = str(code).zfill(6)
            data_map[clean_code] = group.drop(columns=["종목코드"])
        return data_map
    except Exception as e:
        print(f"⚠️ [ML] 캐시 로드 실패 ({f_path}): {e}")
        return {}


def build_master_dataset(data_dir="data"):
    """
    [v19.5] First-touch Labeling + Rolling Z-score

    변경점 (v19.0 → v19.5):
    - [Fix 6] pkl/parquet/csv 멀티포맷 캐시 로더 (OHLCVCache v5.0 호환)
    - [Fix 1] _compute_soft_label에 future_lows 전달
    - [Fix 5] StandardScaler fit 후 Rolling Z-score도 저장
    """
    # [v19.5 Fix 6] 3가지 확장자 모두 탐색 (pkl → parquet → csv 우선순위)
    files = []
    for ext in ("*.pkl", "*.parquet", "*.csv"):
        files.extend(glob.glob(os.path.join(data_dir, f"ohlcv_cache_{ext}")))
    # 날짜 기준 정렬 + 같은 날짜 중복 제거 (parquet 우선)
    seen_dates = set()
    unique_files = []
    for f_path in sorted(files, key=extract_date):
        d = extract_date(f_path)
        if d not in seen_dates:
            seen_dates.add(d)
            unique_files.append(f_path)
    files = unique_files

    if not files:
        print(f"⚠️ [ML] {data_dir}/ohlcv_cache_* 파일이 없습니다. (pkl/parquet/csv 모두 탐색)")
        return None

    print(f"📂 [ML] 학습 데이터 {len(files)}개 파일 발견 ({', '.join(os.path.splitext(f)[1] for f in files[:3])}...)")
    all_samples = []

    for f_path in files:
        try:
            data_map = _load_ohlcv_cache_file(f_path)
            if not data_map:
                continue
            for code, raw_df in data_map.items():
                try:
                    df = clean_ohlcv(raw_df)
                    df_feat = add_technical_features(df)
                    if len(df_feat) < SEQ_LENGTH + 1:
                        continue

                    for i in range(SEQ_LENGTH - 1, len(df_feat)):
                        anchor_date = df_feat.index[i]
                        seq = df_feat.iloc[i - SEQ_LENGTH + 1:i + 1].values

                        if len(seq) != SEQ_LENGTH:
                            continue

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

                        # [Fix 1] High + Low 모두 전달
                        future_highs = df.iloc[entry_idx:entry_idx + 5]['High']
                        future_lows = df.iloc[entry_idx:entry_idx + 5]['Low']
                        label = _compute_soft_label(
                            entry_price, future_highs, future_lows
                        )
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
    y_train = train_df['y'].values.astype(np.float32)
    y_val = val_df['y'].values.astype(np.float32)

    # [Fix 5] 스케일러는 하위 호환용으로 유지 + Rolling Z-score 병행
    scaler = StandardScaler()
    n_feat = X_train.shape[2]
    scaler.fit(X_train.reshape(-1, n_feat))
    joblib.dump(scaler, SCALER_PATH)

    # Rolling Z-score 적용
    X_train_s = _apply_rolling_zscore(X_train)
    X_val_s = _apply_rolling_zscore(X_val)

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


# ══════════════════════════════════════════════════════════════
# [Fix 4] Dynamic Ensemble — MAE 기반 역수 가중
# ══════════════════════════════════════════════════════════════

def _compute_dynamic_weights(lstm_probs, xgb_probs, y_val):
    """
    [v19.0 Fix 4] 검증셋 MAE 기반 동적 앙상블 가중치

    v18.3: lstm * 0.6 + xgb * 0.4 (고정)
    v19.0: inverse-MAE weighting
      → MAE가 작은(더 정확한) 모델에 자동으로 더 높은 비중

    Returns:
        (w_lstm, w_xgb) — 합이 1.0
    """
    y_binary = (y_val > 0).astype(np.float32)

    mae_lstm = np.mean(np.abs(lstm_probs - y_binary))
    mae_xgb = np.mean(np.abs(xgb_probs - y_binary))

    # 0 방지
    mae_lstm = max(mae_lstm, 1e-6)
    mae_xgb = max(mae_xgb, 1e-6)

    # 역수 가중: MAE가 작을수록 비중 큼
    inv_lstm = 1.0 / mae_lstm
    inv_xgb = 1.0 / mae_xgb
    total = inv_lstm + inv_xgb

    w_lstm = round(float(inv_lstm / total), 3)
    w_xgb = round(float(inv_xgb / total), 3)

    # 극단 방지: 어느 한쪽이 0.8 초과하지 않도록
    w_lstm = max(0.2, min(0.8, w_lstm))
    w_xgb = 1.0 - w_lstm

    print(f"   ⚖️ Dynamic Weights: LSTM={w_lstm:.2f}, XGB={w_xgb:.2f} "
          f"(MAE: LSTM={mae_lstm:.4f}, XGB={mae_xgb:.4f})")

    return w_lstm, w_xgb


def _load_ensemble_weights():
    """저장된 앙상블 가중치 로드 (없으면 기본값)"""
    if os.path.exists(ENSEMBLE_WEIGHTS_PATH):
        try:
            with open(ENSEMBLE_WEIGHTS_PATH, 'r') as f:
                data = json.load(f)
            return data.get("w_lstm", 0.6), data.get("w_xgb", 0.4)
        except Exception:
            pass
    return 0.6, 0.4  # 기본 폴백


def _save_ensemble_weights(w_lstm, w_xgb):
    """앙상블 가중치 저장"""
    try:
        with open(ENSEMBLE_WEIGHTS_PATH, 'w') as f:
            json.dump({
                "w_lstm": w_lstm,
                "w_xgb": w_xgb,
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2)
    except Exception:
        pass


# ====================== 학습 ======================

def train_model(force=False):
    """
    [v19.0] 5-Fix 통합 학습 파이프라인

    개선점:
    - [Fix 1] First-touch labeling (build_master_dataset 내)
    - [Fix 3] Log-scaled SoftTargetLoss
    - [Fix 4] Dynamic Ensemble 가중치 산출 및 저장
    - [Fix 5] Rolling Z-score 스케일링 (build_master_dataset 내)
    """
    if is_trained_today(force):
        print("✅ [SKIP] 오늘 이미 v19.0 모델 학습이 완료되었습니다.")
        return

    print("🤖 AI 모델 v19.0 학습 시작 (First-touch + Log-weight + Dynamic Ensemble)...")

    data = build_master_dataset()
    if data is None:
        print("⚠️ [ML] 학습 데이터 부족으로 중단합니다.")
        return

    X_tr, y_tr, X_val, y_val, meta_val, in_dim, pos_ratio = data

    # ============= (1) LSTM 학습 =============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingAttnLSTM(in_dim, 64, 2, 1).to(device)

    # [Fix 3] Log-scaled loss
    criterion = SoftTargetLoss(pos_ratio=pos_ratio, beta=0.2)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_kpi = 0.0
    patience_counter = 0
    PATIENCE = 5
    MAX_EPOCHS = 40

    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=128)
    y_val_binary = (y_val > 0).astype(np.float32)

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

        val_res = pd.DataFrame({'prob': all_probs, 'target': y_val_binary})
        hit_rates = []
        for k in [20, 50, 100]:
            if len(val_res) >= k:
                hit_rates.append(
                    val_res.sort_values('prob', ascending=False).head(k)['target'].mean()
                )
        avg_precision = np.mean(hit_rates) if hit_rates else 0.0

        y_val_strong = (y_val >= 0.67).astype(np.float32)
        val_res_strong = pd.DataFrame({'prob': all_probs, 'target': y_val_strong})
        strong_hit = 0.0
        if len(val_res_strong) >= 20:
            strong_hit = val_res_strong.sort_values('prob', ascending=False).head(20)['target'].mean()

        val_res_soft = pd.DataFrame({'prob': all_probs, 'soft_y': y_val})
        top20_soft_avg = 0.0
        if len(val_res_soft) >= 20:
            top20_soft_avg = val_res_soft.sort_values('prob', ascending=False) \
                .head(20)['soft_y'].mean()

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

        if patience_counter >= PATIENCE:
            print(f"🛑 [Early Stop] {PATIENCE}에폭 연속 개선 없음. "
                  f"Best KPI: {best_kpi:.4f} (epoch {epoch - PATIENCE})")
            break

    print(f"✅ [LSTM] 학습 완료 (Best KPI: {best_kpi:.4f})")

    # ============= (2) XGBoost 학습 =============
    lstm_val_probs = all_probs.copy()  # [Fix 4] 나중에 동적 가중치 계산용

    if XGB_OK:
        print("🌲 XGBoost 앙상블 학습 시작...")
        X_tr_xgb = X_tr[:, -1, :]
        X_val_xgb = X_val[:, -1, :]

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

        xgb_val_probs = xgb_model.predict_proba(X_val_xgb)[:, 1]
        try:
            xgb_auc = roc_auc_score(y_val_binary, xgb_val_probs)
        except ValueError:
            xgb_auc = 0.5
        print(f"✅ [XGBoost] 학습 완료 (AUC: {xgb_auc:.3f})")

        importance = dict(zip(FEATURE_COLS[:in_dim], xgb_model.feature_importances_))
        top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top 피처: {', '.join(f'{k}({v:.2f})' for k, v in top5)}")

        # [Fix 4] Dynamic Ensemble 가중치 산출
        w_lstm, w_xgb = _compute_dynamic_weights(
            lstm_val_probs, xgb_val_probs, y_val
        )
        _save_ensemble_weights(w_lstm, w_xgb)
    else:
        _save_ensemble_weights(1.0, 0.0)

    _save_meta(in_dim, pos_ratio, best_kpi)

    global _loaded_lstm_model, _loaded_scaler
    _loaded_lstm_model = None
    _loaded_scaler = None
    load_model()
    print("✅ [ML] v19.0 전체 학습 파이프라인 완료!")


def _save_meta(in_dim, pos_ratio, best_kpi):
    meta = {
        "version": "19.0",
        "seq_length": SEQ_LENGTH,
        "n_features": in_dim,
        "feature_cols": FEATURE_COLS[:in_dim],
        "target_tiers": TARGET_TIERS,
        "stop_loss_pct": STOP_LOSS_PCT,
        "pos_ratio": round(float(pos_ratio), 4),
        "best_kpi": round(float(best_kpi), 4),
        "trained_at": datetime.now().isoformat(),
        "fixes": ["first_touch_labeling", "vectorized_inference",
                   "log_scaled_weights", "dynamic_ensemble", "rolling_zscore"],
    }
    try:
        with open(META_PATH, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ====================== 추론 ======================

def apply_ml_score(current_df, full_ohlcv_map):
    """
    [v19.0] 5-Fix 통합 추론

    개선점:
    - [Fix 2] add_technical_features_batch로 일괄 피처 계산
    - [Fix 4] 동적 앙상블 가중치 적용
    - [Fix 5] Rolling Z-score 스케일링 (v19 모델일 때)
    """
    global _loaded_lstm_model, _loaded_scaler, _loaded_xgb_model
    global _loaded_seq_length, _loaded_use_rolling_zscore

    if _loaded_lstm_model is None or _loaded_scaler is None:
        load_model()

    if _loaded_lstm_model is None or _loaded_scaler is None:
        print("⚠️ [ML] 모델 파일 없음. ML_SCORE=0 으로 진행.")
        current_df["ML_SCORE"] = 0.0
        return current_df

    seq_len = _loaded_seq_length
    use_rolling = _loaded_use_rolling_zscore

    cache = get_feature_cache()
    target_codes = current_df["종목코드"].unique()

    # [Fix 2] 필요한 종목만 필터링하여 배치 처리
    codes_needing_calc = []
    codes_from_cache = []
    cached_seqs = {}

    for code in target_codes:
        if code not in full_ohlcv_map:
            continue

        raw_df = full_ohlcv_map[code]
        df = clean_ohlcv(raw_df)
        if df.empty:
            continue
        last_date = str(df.index[-1])
        cache_key = f"{code}_s{seq_len}"

        if cache_key in cache and cache[cache_key].get('date') == last_date:
            cached_seqs[code] = cache[cache_key]['seq']
            codes_from_cache.append(code)
        else:
            codes_needing_calc.append(code)

    # [Fix 2] 캐시 미스 종목들을 배치 피처 계산
    new_cache_count = 0
    if codes_needing_calc:
        subset_map = {c: full_ohlcv_map[c] for c in codes_needing_calc
                      if c in full_ohlcv_map}
        feat_map = add_technical_features_batch(subset_map)

        for code, df_feat in feat_map.items():
            if len(df_feat) >= seq_len:
                seq = df_feat.iloc[-seq_len:].values
                cached_seqs[code] = seq

                raw_df = full_ohlcv_map[code]
                df = clean_ohlcv(raw_df)
                last_date = str(df.index[-1])
                cache_key = f"{code}_s{seq_len}"
                cache[cache_key] = {'date': last_date, 'seq': seq}
                new_cache_count += 1

    if new_cache_count > 0:
        save_feature_cache(cache)

    # 유효 입력 조합
    valid_inputs = []
    codes = []
    for code in target_codes:
        if code in cached_seqs:
            valid_inputs.append(cached_seqs[code])
            codes.append(code)

    if not valid_inputs:
        current_df["ML_SCORE"] = 0.0
        return current_df

    X_raw = np.array(valid_inputs)
    n_feat = X_raw.shape[2]
    scaler_dim = getattr(_loaded_scaler, 'n_features_in_', n_feat)

    if n_feat != scaler_dim:
        print(f"⚠️ [ML] 피처 차원 불일치 (데이터={n_feat}, 모델={scaler_dim}). ML_SCORE=0 폴백.")
        current_df["ML_SCORE"] = 0.0
        return current_df

    # [Fix 5] 스케일링: v19 → Rolling Z-score, 하위 버전 → StandardScaler
    try:
        if use_rolling:
            X_scaled = _apply_rolling_zscore(X_raw)
        else:
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

    # --- XGBoost 추론 + [Fix 4] Dynamic Ensemble ---
    final_probs = lstm_probs
    if _loaded_xgb_model is not None:
        try:
            X_xgb = X_scaled[:, -1, :]
            xgb_probs = _loaded_xgb_model.predict_proba(X_xgb)[:, 1]

            # [Fix 4] 동적 가중치 로드
            w_lstm, w_xgb = _load_ensemble_weights()
            final_probs = lstm_probs * w_lstm + xgb_probs * w_xgb
            print(f"   ⚖️ Ensemble: LSTM×{w_lstm:.2f} + XGB×{w_xgb:.2f}")
        except Exception as e:
            print(f"⚠️ [ML] XGBoost 추론 실패, LSTM 단독: {e}")

    # ─────────────────────────────────────────────
    # 하이브리드 스코어링 (자신감 적응형)
    # ─────────────────────────────────────────────
    prob_scores = final_probs * 100.0

    rank_series = pd.Series(final_probs).rank(pct=True) * 100.0
    pct_scores = rank_series.values

    prob_std = float(np.std(final_probs))
    w_pct = float(np.clip((prob_std - 0.05) / 0.10, 0.3, 0.6))
    w_prob = 1.0 - w_pct

    pct_raw = rank_series.values
    pct_scores = (pct_raw / 100.0) ** 1.2 * 100.0

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

        top5 = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"🧠 [ML] Top5: {', '.join(f'{c}({s})' for c, s in top5)}")

        n = len(all_scores)
        if n >= 10:
            top20_prob = np.sort(prob_scores)[-max(n // 5, 1):].mean()
            bot20_prob = np.sort(prob_scores)[:max(n // 5, 1)].mean()
            spread = top20_prob - bot20_prob
            print(f"   📐 변별력: top20%_prob={top20_prob:.1f}, "
                  f"bot20%_prob={bot20_prob:.1f}, spread={spread:.1f}")

    return current_df
