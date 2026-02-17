"""
LDY Pro Trader Collector v8.0 (Macro Filter & Smart Regime)
- v8.0: 매크로(환율/나스닥) 필터링 추가, 시장 위험도에 따른 EBS/종목수 동적 조절
- v7.5: 최근 저점(Swing Low) 기반 스마트 손절 보정 + 매수세 강도(V-Power) 팩터 추가
...
"""

import os
import io
import time
import math
import pickle  # ✅ [v7.0 추가] 데이터 직렬화/캐싱용
from typing import Dict, Any, Optional, Callable, Tuple, List

import ml_engine  # ✅ [v10.0 추가]
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import re
from glob import glob
try:
    from pykrx import stock  # optional
    PYKRX_OK = True
except Exception:
    stock = None
    PYKRX_OK = False
from tqdm import tqdm
import FinanceDataReader as fdr
import asyncio  # 비동기 실행용
from db_utils import LDYDBManager       # [신규] DB 매니저
from async_crawler import AsyncNewsFetcher # [신규] 비동기 크롤러
import random
import dart_analyzer  # ✅ [2단계] DART 분석기 추가

from time_utils import now_kst, now_utc, KST
from concurrent.futures import ThreadPoolExecutor, as_completed
# 👇👇 [수정] 클래스 정의를 지우고, schema.py에서 가져오도록 변경 👇👇
from schema import RouteState



# [v9.0 추가] LLM 및 뉴스 크롤링용 라이브러리
try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False
    print("⚠️ BeautifulSoup4가 설치되지 않았습니다. (pip install beautifulsoup4)")

# ▼▼▼ [여기] 아래 코드를 삽입하세요 (secrets.toml 강제 로드) ▼▼▼
def load_secrets_to_env():
    """
    Streamlit 없이 실행될 때 .streamlit/secrets.toml 파일을 읽어 
    환경변수(os.environ)에 로드하는 함수
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        secrets_path = os.path.join(base_dir, ".streamlit", "secrets.toml")
        
        if os.path.exists(secrets_path):
            with open(secrets_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 주석이나 빈 줄 건너뛰기
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    # [Section] 헤더 건너뛰기
                    if line.startswith("[") and line.endswith("]"):
                        continue
                        
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    
                    if key and val:
                        os.environ[key] = val
                        # print(f"🔑 Key Loaded: {key}") 
    except Exception as e:
        print(f"⚠️ Secrets 로드 실패: {e}")

# 실행 초기화 시점에 키 로드
load_secrets_to_env()
# ▲▲▲ [여기] 까지 삽입 ▲▲▲

# LLM API 키 설정 (환경변수 또는 직접 입력)
# Google Gemini (무료 티어 가능) 또는 OpenAI 사용 권장
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        LLM_AVAILABLE = True
    else:
        LLM_AVAILABLE = False
except ImportError:
    LLM_AVAILABLE = False

# [보안 설정]
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_ID = os.environ.get("TG_ID")

# ------------------------------- 설정 -------------------------------

LOOKBACK_DAYS = 250          # 과거 데이터 조회 일수
BENCH_LOOKBACK_DAYS = 60     # 벤치마크 상대강도 기준 일수
TOP_N = 600                  # 거래대금 상위 N개 종목
MIN_TURNOVER_EOK = 50        # 최소 거래대금 (억원)
MIN_MCAP_EOK = 1000          # 최소 시총 (억원)
RSI_LOW, RSI_HIGH = 45, 65   # RSI 적정 구간
PASS_EBS = 4                 # EBS (룰 기반 스코어) 통과 기준
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data")
UTF8 = "utf-8-sig"

# 병렬 처리 (너무 크게 잡으면 KRX 쿼리/네트워크에 부담)
MAX_WORKERS = int(os.environ.get("LDY_WORKERS", "4"))  # 8 -> 4

# 변동성 축소/스퀴즈 감지
BB_PERIOD = 20
BB_STD = 2

# (1) 단순 Bandwidth(%) 기준 (정보용/보조)
BB_SQUEEZE_BW = 10.0  # Bandwidth(%) < 10%면 '협의 스퀴즈'로 표시

# (2) John Carter의 TTM Squeeze (진성 스퀴즈)
#     - Bollinger Band(표준편차)가 Keltner Channel(ATR) 안으로 완전히 들어갈 때
KC_PERIOD = 20
KC_ATR_PERIOD = 20
KC_MULT = 1.5

# 스퀴즈 보너스(가산점) — v6.9부터는 'TTM Squeeze'를 주 신호로 사용
BONUS_BB_SQUEEZE_SCORE = 3.0
BONUS_BB_SQUEEZE_ENTRY = 4.0
# 섹터 모멘텀 보정은 "업종 평균점수"보다 "업종 수익률"이 더 직관적이라 교체
# (기존 W_SECTOR = 0.05 그대로 사용해도 OK)

# [가중치]
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = 0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = 6.0, 6.0, 4.0
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 4.0, 2.0

# 🔹 v6.7 추가: 업종 보너스 + 과도 손절 패널티
W_SECTOR = 0.05          # 섹터 강도 보정 (최대 +5점 수준)
P_BIG_SL = 3.0           # 손절 폭이 너무 큰 종목 패널티

# ------------------------------- 유틸 -------------------------------

def log(msg: str) -> None:
    print(f"[{now_kst().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def wma(s: pd.Series, period: int) -> pd.Series:
    """
    [v9.0] 가중 이동 평균 (HMA 계산용)
    """
    weights = np.arange(1, period + 1)
    
    def _calc(x):
        return np.dot(x, weights) / weights.sum()
    
    # raw=True로 속도 최적화
    return s.rolling(period).apply(_calc, raw=True)

def calc_hma(s: pd.Series, period: int) -> pd.Series:
    """
    [v9.0] Hull Moving Average (HMA)
    - 반응 속도가 빠르고 휩소가 적음
    """
    if len(s) < period:
        return pd.Series(np.nan, index=s.index)

    half_length = int(period / 2)
    sqrt_length = int(math.sqrt(period))

    wma_half = wma(s, half_length)
    wma_full = wma(s, period)

    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt_length)

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    [v9.0] On-Balance Volume (OBV)
    - 주가 등락에 따른 거래량 누적 지표 (스마트 머니 추적)
    - 공식: 주가 상승 시 거래량 더하기, 하락 시 빼기
    """
    # 전일 대비 등락 부호 (-1, 0, 1)
    change = np.sign(close.diff()).fillna(0)
    # 부호 * 거래량 누적 합계
    obv = (change * volume).cumsum()
    return obv

# -------------------- [여기까지] --------------------

def _safe_sum(x: pd.Series) -> float:
    return pd.to_numeric(x, errors="coerce").fillna(0).sum()

def nz_num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # down=0 && up>0 => 100, up=0 && down>0 => 0, 둘 다 0 => 50
    both_zero = (roll_up == 0) & (roll_down == 0)
    rsi = rsi.where(~both_zero, 50)
    rsi = rsi.where(~((roll_down == 0) & (roll_up != 0)), 100)
    rsi = rsi.where(~((roll_up == 0) & (roll_down != 0)), 0)

    return rsi  # ✅ 이거 반드시 필요

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [(high - low),
         (high - close.shift(1)).abs(),
         (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def calc_supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    SuperTrend 지표 계산 (초기 NaN 예외처리 적용)
    """
    atr = calc_atr(high, low, close, period)

    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # 결과 배열 초기화 (NaN으로 시작)
    st_out = [np.nan] * len(close)
    trend_out = [1] * len(close)

    vals_c = close.values
    vals_bu = basic_upper.values
    vals_bl = basic_lower.values

    # 유효한 ATR 값이 나오는 시점부터 계산 시작
    # period 값 인덱스부터 데이터가 있다고 가정
    start_idx = period
    if start_idx >= len(close):
        # 데이터가 너무 짧은 경우 예외 처리
        return pd.Series(st_out, index=close.index), pd.Series(trend_out, index=close.index)

    # 초기값 설정 (첫 유효값 기준)
    final_upper = vals_bu[start_idx]
    final_lower = vals_bl[start_idx]
    curr_trend = 1

    st_out[start_idx] = final_lower
    trend_out[start_idx] = 1

    for i in range(start_idx + 1, len(close)):
        # 1. Upper Band 계산
        if (vals_bu[i] < final_upper) or (vals_c[i-1] > final_upper):
            final_upper = vals_bu[i]

        # 2. Lower Band 계산
        if (vals_bl[i] > final_lower) or (vals_c[i-1] < final_lower):
            final_lower = vals_bl[i]

        # 3. 추세 결정
        prev_trend = trend_out[i-1]

        if prev_trend == 1: # 상승 중
            if vals_c[i] < final_lower:
                curr_trend = -1
                final_upper = vals_bu[i] # Reset
            else:
                curr_trend = 1
        else: # 하락 중
            if vals_c[i] > final_upper:
                curr_trend = 1
                final_lower = vals_bl[i] # Reset
            else:
                curr_trend = -1

        trend_out[i] = curr_trend
        st_out[i] = final_upper if curr_trend == -1 else final_lower

    return pd.Series(st_out, index=close.index), pd.Series(trend_out, index=close.index)


def calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * vol
    pos = np.where(tp.diff() > 0, rmf, 0)
    neg = np.where(tp.diff() < 0, rmf, 0)
    pos_s = pd.Series(pos, index=close.index).rolling(period).sum()
    neg_s = pd.Series(neg, index=close.index).rolling(period).sum().replace(0, 1)
    return 100 - (100 / (1 + (pos_s / neg_s)))


def calc_vwap(df: pd.DataFrame) -> float:
    """
    주어진 기간(DataFrame) 동안의 거래량 가중 평균 가격(VWAP) 계산
    Typical Price = (High + Low + Close) / 3
    VWAP = Sum(Typical Price * Volume) / Sum(Volume)
    """
    if df.empty:
        return 0.0

    v = df['거래량']
    tp = (df['고가'] + df['저가'] + df['종가']) / 3

    vol_sum = v.sum()
    if vol_sum == 0:
        return 0.0

    return (tp * v).sum() / vol_sum

def check_candle_pattern(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> List[str]:
    """
    최근 캔들 패턴(망치형, 장악형) 감지
    """
    if len(c) < 2:
        return []

    patterns = []

    # 마지막 캔들 기준 (오늘)
    curr_o, curr_h, curr_l, curr_c = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    # 전일 캔들
    prev_o, prev_h, prev_l, prev_c = o.iloc[-2], h.iloc[-2], l.iloc[-2], c.iloc[-2]

    # 1. 망치형 (Hammer): 하락 추세 바닥에서 긴 아랫꼬리 + 작은 몸통
    # (여기서는 추세 판단 없이 캔들 모양만 봅니다)
    body = abs(curr_c - curr_o)
    upper_shadow = curr_h - max(curr_c, curr_o)
    lower_shadow = min(curr_c, curr_o) - curr_l

    # 조건: 아랫꼬리가 몸통의 2배 이상 & 윗꼬리는 몸통의 0.5배 이하 & 몸통이 아주 작지는 않음
    if (lower_shadow >= body * 2) and (upper_shadow <= body * 0.5) and (body > 0):
        patterns.append("망치형")

    # 2. 상승 장악형 (Bullish Engulfing): 전일 음봉 -> 금일 양봉이 전일 몸통을 감쌈
    is_prev_red = prev_c < prev_o
    is_curr_green = curr_c > curr_o

    if is_prev_red and is_curr_green:
        # 금일 시가가 전일 종가보다 낮거나 같고, 금일 종가가 전일 시가보다 높거나 같음
        # (몸통이 이전 몸통을 완전히 덮음)
        if (curr_o <= prev_c) and (curr_c >= prev_o):
            patterns.append("장악형")

    return patterns

def round_to_tick(price: float) -> int:
    if price < 2000: t = 1
    elif price < 5000: t = 5
    elif price < 20000: t = 10
    elif price < 50000: t = 50
    elif price < 200000: t = 100
    elif price < 500000: t = 500
    else: t = 1000
    return int(round(price / t) * t)

def tick_size(price: float) -> int:
    if price < 2000: return 1
    if price < 5000: return 5
    if price < 20000: return 10
    if price < 50000: return 50
    if price < 200000: return 100
    if price < 500000: return 500
    return 1000

def add_sector_momentum(df: pd.DataFrame, group_col: str = "업종_대분류") -> Tuple[pd.DataFrame, pd.Series]:
    """
    [v9.0] 섹터 주도주 로직 강화
    - 단순 등락률(Ret)뿐만 아니라 시장 대비 초과수익(RS, Relative Strength)을 반영
    - 시장이 하락해도 버티거나 오르는 '진짜 주도 섹터' 발굴
    """
    # 필수 컬럼 체크
    if group_col not in df.columns:
        df["SECTOR_RS"] = np.nan
        df["SECTOR_RANK"] = np.nan
        return df, pd.Series(dtype=float)

    # 1. 단순 모멘텀 (최근 5일 평균 수익률)
    col_ret = "ret_5d_%" if "ret_5d_%" in df.columns else "등락률"
    g_ret = df.groupby(group_col)[col_ret].mean()
    
    # 2. [핵심] 시장 대비 초과 수익 (20일 평균 상대강도)
    # analyze_ticker에서 계산된 'rel_20d_%' (종목수익률 - 지수수익률) 활용
    col_rs = "rel_20d_%" if "rel_20d_%" in df.columns else col_ret
    g_rs = df.groupby(group_col)[col_rs].mean()
    
    # 3. 종합 섹터 점수 산출 (RS에 가중치 60%, 단순수익 40%)
    # RS가 높아야 진짜 주도주임
    sector_score = (g_ret * 0.4) + (g_rs * 0.6)
    sector_score = sector_score.sort_values(ascending=False)
    
    # 4. 데이터프레임에 매핑
    df["SECTOR_RET_5D"] = df[group_col].map(g_ret)
    df["SECTOR_RS"] = df[group_col].map(g_rs)   # RS 지표 저장
    df["SECTOR_RANK"] = df[group_col].map(sector_score.rank(ascending=False, method="min"))
    
    return df, sector_score


def compute_market_breadth(df: pd.DataFrame) -> Dict[str, float]:
    """
    20일선 상회 비율(%) = 시장 온도
    analyze_ticker에서 Above_MA20을 넣는 전제
    """
    out = {}
    if "Above_MA20" not in df.columns:
        return {"ALL": np.nan, "KOSPI": np.nan, "KOSDAQ": np.nan}

    for m in ["KOSPI", "KOSDAQ"]:
        sub = df[df["시장"] == m]
        out[m] = round(float(sub["Above_MA20"].mean() * 100), 1) if len(sub) else np.nan

    out["ALL"] = round(float(df["Above_MA20"].mean() * 100), 1) if len(df) else np.nan
    return out


def label_market_temp(breadth_all: float) -> str:
    if not np.isfinite(breadth_all):
        return "🌫 N/A"
    if breadth_all >= 65:
        return "🔥 과열"
    if breadth_all <= 35:
        return "🧊 침체"
    return "🌤 중립"

def _backoff_sleep(i: int, base: float = 0.35, cap: float = 2.0) -> None:
    # i=0,1 일 때만 의미 (총 2회 재시도)
    jitter = random.uniform(0.85, 1.15)
    time.sleep(min(cap, base * (2 ** i)) * jitter)

def run_reality_check(out_dir: str, trade_ymd: str) -> None:
    """
    전일(가장 최근) recommend_YYYYMMDD*.csv의 상위 추천들이
    오늘 종가 스냅샷 기준으로 얼마나 움직였는지 검증 CSV 생성
    - data/reality_check_YYYYMMDD.csv
    - data/reality_check_latest.csv
    """
    try:
        # 오늘 종가 스냅샷
        snap_path = os.path.join(out_dir, f"price_snapshot_{trade_ymd}.csv")
        if not os.path.exists(snap_path):
            return
        snap = pd.read_csv(snap_path, dtype={"종목코드": str})
        snap["종목코드"] = snap["종목코드"].astype(str).str.zfill(6)
        close_map = dict(zip(snap["종목코드"], pd.to_numeric(snap["종가"], errors="coerce")))

        # 과거 recommend 파일 중 가장 최근(오늘 제외)
        files = [f for f in os.listdir(out_dir) if f.startswith("recommend_") and f.endswith(".csv")]
        cand = []
        for f in files:
            # recommend_YYYYMMDD or recommend_YYYYMMDD_tag
            core = f.replace("recommend_", "").replace(".csv", "")
            ymd = core.split("_")[0]
            if len(ymd) == 8 and ymd.isdigit() and ymd != trade_ymd:
                cand.append((ymd, f))
        if not cand:
            return
        cand.sort(reverse=True)
        prev_ymd, prev_file = cand[0]

        prev = pd.read_csv(os.path.join(out_dir, prev_file), dtype={"종목코드": str})
        prev["종목코드"] = prev["종목코드"].astype(str).str.zfill(6)

        # 상위 30개만 체크(너무 커지면 파일만 무거워짐)
        prev = prev.head(30).copy()

        prev["오늘종가"] = prev["종목코드"].map(close_map)
        prev["전일추천매수가"] = pd.to_numeric(prev.get("추천매수가", np.nan), errors="coerce")
        prev["전일→오늘_수익률%"] = (prev["오늘종가"] / prev["전일추천매수가"] - 1.0) * 100

        prev["검증기준일"] = trade_ymd
        prev["비교대상추천일"] = prev_ymd

        out1 = os.path.join(out_dir, f"reality_check_{trade_ymd}.csv")
        out2 = os.path.join(out_dir, "reality_check_latest.csv")
        prev.to_csv(out1, index=False, encoding=UTF8)
        prev.to_csv(out2, index=False, encoding=UTF8)
        log(f"🧪 Reality Check 저장 완료 → {out1}")
    except Exception as e:
        log(f"⚠️ Reality Check 실패: {e}")


def _ymd8_to_dash(s: str) -> str:
    s = str(s)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s

def _pykrx_df(fn, *args, **kwargs):
    """PYKRX 호출을 안전하게 감싸고 실패 시 None (+2회 재시도/backoff)"""
    if (not PYKRX_OK) or (stock is None):
        return None

    last_err = None
    for i in range(3):  # 총 3번 시도 = 최초 1 + 재시도 2
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if i < 2:
                _backoff_sleep(i)
            else:
                return None

def safe_ohlcv_by_date(start_ymd: str, end_ymd: str, code: str) -> Optional[pd.DataFrame]:
    df = _pykrx_df(stock.get_market_ohlcv_by_date, start_ymd, end_ymd, code)

    if df is None or getattr(df, "empty", True):
        df = None
        for i in range(3):  # 1 + 2회 재시도
            try:
                df = fdr.DataReader(str(code).zfill(6), _ymd8_to_dash(start_ymd), _ymd8_to_dash(end_ymd))
                break
            except Exception:
                if i < 2:
                    _backoff_sleep(i)
                else:
                    df = None

        if df is not None and not getattr(df, "empty", True):
            df = df.rename(columns={
                "Open": "시가", "High": "고가", "Low": "저가", "Close": "종가", "Volume": "거래량"
            })

    return df

def safe_ohlcv_by_ticker(ymd: str, market: str) -> Optional[pd.DataFrame]:
    """market 단위(전종목) ohlcv는 FDR로 1:1 대체가 어려워서 pykrx만 안전 호출"""
    return _pykrx_df(stock.get_market_ohlcv_by_ticker, ymd, market=market)

def safe_market_cap_by_ticker(ymd: str, market: str) -> Optional[pd.DataFrame]:
    return _pykrx_df(stock.get_market_cap_by_ticker, ymd, market=market)

def safe_ticker_list(ymd: str, market: str) -> List[str]:
    out = _pykrx_df(stock.get_market_ticker_list, ymd, market=market)
    return list(out) if out is not None else []

def safe_ticker_name(ticker: str) -> Optional[str]:
    if (not PYKRX_OK) or (stock is None):
        return None
    try:
        return stock.get_market_ticker_name(ticker)
    except Exception:
        return None


# ------------------------------- 매크로(Macro) 필터 -------------------------------

def check_macro_env(trade_ymd: str) -> Tuple[str, str, int, int]:
    """
    [v8.0] 환율(USD/KRW) 및 나스닥(IXIC) 매크로 지표 분석
    
    Returns:
        risk_level (str): 'CRITICAL', 'HIGH', 'NORMAL'
        summary_msg (str): 텔레그램 출력용 요약 메시지
        adj_ebs (int): 조정된 PASS_EBS (기본값 PASS_EBS)
        rec_limit (int): 텔레그램 추천 종목 수 (기본 5)
    """
    try:
        # 날짜 설정 (데이터 수신 지연 고려하여 넉넉히 10일 전부터 조회)
        end_dt = datetime.strptime(trade_ymd, "%Y%m%d")
        start_dt = end_dt - timedelta(days=10)
        start_s = start_dt.strftime("%Y-%m-%d")

        # 1. USD/KRW 환율 조회
        df_usd = fdr.DataReader('USD/KRW', start_s)
        curr_usd = df_usd['Close'].iloc[-1]
        prev_usd = df_usd['Close'].iloc[-2]
        usd_chg = (curr_usd - prev_usd) / prev_usd * 100

        # 2. 나스닥(IXIC) 조회
        df_nas = fdr.DataReader('IXIC', start_s)
        curr_nas = df_nas['Close'].iloc[-1]
        prev_nas = df_nas['Close'].iloc[-2]
        nas_chg = (curr_nas - prev_nas) / prev_nas * 100

        # 기본 설정값 로드
        adj_ebs = PASS_EBS  # 기본값 4
        rec_limit = 5       # 기본 Top 5
        msgs = []
        risk_score = 0

        # [로직 1] 환율 체크 (1400원 이상 or +0.5% 급등)
        if curr_usd >= 1400:
            msgs.append(f"💸 고환율({int(curr_usd)}원)")
            risk_score += 1
        elif usd_chg >= 0.5:
            msgs.append(f"💸 환율급등(+{usd_chg:.2f}%)")
            risk_score += 1

        # [로직 2] 나스닥 체크 (-2.0% 급락)
        if nas_chg <= -2.0:
            msgs.append(f"📉 나스닥급락({nas_chg:.2f}%)")
            risk_score += 2  # 나스닥 급락은 더 큰 위험으로 간주

        # [결과 판정]
        risk_level = "NORMAL"

        if risk_score >= 2:
            # 위험도 높음: 보수적 진입 + 종목 수 축소
            risk_level = "CRITICAL"
            adj_ebs = PASS_EBS + 1  # 기준 점수 상향 (예: 4 -> 5점)
            rec_limit = 3           # 추천 수 축소 (5 -> 3개)

        elif risk_score == 1:
            # 위험도 중간: 보수적 진입만
            risk_level = "HIGH"
            adj_ebs = PASS_EBS + 1
            rec_limit = 5

        # 요약 메시지 생성
        summary = f"🌍 매크로: {risk_level}"
        if msgs:
            summary += f" ({', '.join(msgs)})"
            if risk_level != "NORMAL":
                summary += f"\n   → 🛡️ 보수적 대응 (EBS {adj_ebs}점↑ / Top{rec_limit})"

        log(f"🔍 매크로 분석 완료: {summary}")
        return risk_level, summary, adj_ebs, rec_limit

    except Exception as e:
        log(f"⚠️ 매크로 데이터 수집 실패: {e}")
        return "NORMAL", "🌍 매크로: N/A (데이터 수집 실패)", PASS_EBS, 5


# ------------------------------- v7.0 OHLCV Caching System -------------------------------

def load_ohlcv_cache(ymd: str) -> Dict[str, pd.DataFrame]:
    """
    해당 날짜의 OHLCV 캐시 파일을 로드합니다.
    경로: data/ohlcv_cache_YYYYMMDD.pkl
    """
    cache_path = os.path.join(OUT_DIR, f"ohlcv_cache_{ymd}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            log(f"📂 OHLCV 캐시 로드 완료: {len(data)}개 종목 ({cache_path})")
            return data
        except Exception as e:
            log(f"⚠️ 캐시 파일 로드 실패(재수집 진행): {e}")
    return {}

def save_ohlcv_cache(ymd: str, data: Dict[str, pd.DataFrame]) -> None:
    """
    OHLCV 데이터를 캐시 파일로 저장합니다.
    """
    ensure_dir(OUT_DIR)
    cache_path = os.path.join(OUT_DIR, f"ohlcv_cache_{ymd}.pkl")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        log(f"💾 OHLCV 캐시 저장 완료: {len(data)}개 종목")
    except Exception as e:
        log(f"⚠️ 캐시 저장 실패: {e}")

def prepare_ohlcv_data(
    tickers: List[str], 
    start_ymd: str, 
    end_ymd: str, 
    trade_ymd: str
) -> Dict[str, pd.DataFrame]:
    """
    [v7.0 핵심]
    1. 로컬 캐시 확인
    2. 없는 종목만 병렬 수집 (safe_ohlcv_by_date 재사용)
    3. 캐시 업데이트 및 저장
    4. 전체 데이터 맵 반환
    """
    # 1) 캐시 로드
    ohlcv_map = load_ohlcv_cache(trade_ymd)

    # 2) 수집 필요한 종목 필터링
    #    (이미 캐시에 있고, 데이터가 비어있지 않은 것만 유효)
    targets = []
    for t in tickers:
        code = str(t).zfill(6)
        if code not in ohlcv_map or ohlcv_map[code] is None or ohlcv_map[code].empty:
            targets.append(code)

    if not targets:
        log("✨ 모든 데이터가 캐시에 있습니다. 수집을 건너뜁니다.")
        return ohlcv_map

    log(f"🔄 {len(targets)}개 종목 데이터 수집 시작 (캐시 미적중)...")

    # 3) 병렬 수집 (기존 safe_ohlcv_by_date 활용)
    #    MAX_WORKERS 활용
    collected_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Future -> Code 매핑
        future_to_code = {
            executor.submit(safe_ohlcv_by_date, start_ymd, end_ymd, code): code 
            for code in targets
        }

        for future in tqdm(as_completed(future_to_code), total=len(targets), desc="Fetching OHLCV"):
            code = future_to_code[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    ohlcv_map[code] = df
                    collected_count += 1
                else:
                    # 실패 시 None 저장하여 재시도 방지 or 빈 DF 저장
                    ohlcv_map[code] = pd.DataFrame() 
            except Exception as e:
                log(f"⚠️ {code} 수집 중 에러: {e}")
                ohlcv_map[code] = pd.DataFrame()

    # 4) 캐시 저장
    if collected_count > 0:
        save_ohlcv_cache(trade_ymd, ohlcv_map)

    return ohlcv_map


# ------------------------------- Rank Validation (RANK_SCORE 검증) -------------------------------

def _list_snapshot_days(out_dir: str) -> List[str]:
    paths = glob(os.path.join(out_dir, "price_snapshot_*.csv"))
    days = []
    for p in paths:
        b = os.path.basename(p)
        m = re.match(r"price_snapshot_(\d{8})\.csv$", b)
        if m:
            days.append(m.group(1))
    return sorted(list(set(days)))

def _load_close_map(out_dir: str, ymd: str) -> Dict[str, float]:
    path = os.path.join(out_dir, f"price_snapshot_{ymd}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={"종목코드": str})
    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    close = pd.to_numeric(df["종가"], errors="coerce")
    return dict(zip(df["종목코드"], close))

def _next_trade_day(trade_days: List[str], ymd: str, offset: int) -> Optional[str]:
    try:
        i = trade_days.index(ymd)
    except ValueError:
        return None
    j = i + offset
    if 0 <= j < len(trade_days):
        return trade_days[j]
    return None

def _pick_recommend_file_per_day(out_dir: str) -> Dict[str, str]:
    files = [os.path.basename(p) for p in glob(os.path.join(out_dir, "recommend_*.csv"))]
    mp: Dict[str, List[str]] = {}
    for f in files:
        core = f.replace("recommend_", "").replace(".csv", "")
        ymd = core.split("_")[0]
        if len(ymd) == 8 and ymd.isdigit():
            mp.setdefault(ymd, []).append(f)

    pick: Dict[str, str] = {}
    for ymd, flist in mp.items():
        # 태그 없는 recommend_YYYYMMDD.csv 우선
        plain = [f for f in flist if f == f"recommend_{ymd}.csv"]
        pick[ymd] = plain[0] if plain else sorted(flist)[0]
    return pick

def make_rank_validation_report(
    out_dir: str,
    asof_ymd: str,
    lookback_trading_days: int = 60,
    horizons: List[int] = [1, 3, 5],
    topks: List[int] = [1, 3, 5, 10],
    methods: List[str] = ["RANK_SCORE", "ENTRY_SCORE", "LDY_SCORE"],
) -> None:
    """
    과거 recommend + 이후 price_snapshot을 이용해
    '상위 K개가 H영업일 후에 얼마나 올랐나'를 승률/수익률로 검증한다.

    출력:
    - data/rank_validation_{asof_ymd}.csv (상세)
    - data/rank_validation_summary_{asof_ymd}.csv (요약)
    - data/rank_validation_latest.csv
    - data/rank_validation_summary_latest.csv
    """
    try:
        ensure_dir(out_dir)

        trade_days = _list_snapshot_days(out_dir)
        if not trade_days:
            log("⚠️ rank validation: price_snapshot이 없어 리포트 생략")
            return

        rec_map = _pick_recommend_file_per_day(out_dir)
        if not rec_map:
            log("⚠️ rank validation: recommend 파일이 없어 리포트 생략")
            return

        # 검증 대상 날짜: 스냅샷 캘린더 기준 최근 N 거래일
        tail_days = trade_days[-lookback_trading_days:]
        target_days = [d for d in tail_days if d in rec_map]

        rows = []

        for rec_ymd in target_days:
            rec_path = os.path.join(out_dir, rec_map[rec_ymd])
            try:
                df = pd.read_csv(rec_path, dtype={"종목코드": str})
            except Exception:
                continue

            if df is None or df.empty:
                continue

            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)

            # 필수 숫자
            df["추천매수가"] = pd.to_numeric(df.get("추천매수가", np.nan), errors="coerce")
            df["종가"] = pd.to_numeric(df.get("종가", np.nan), errors="coerce")

            # 추천매수가가 없으면 종가로 대체(최소한 검증은 되게)
            entry_px = df["추천매수가"].where(df["추천매수가"].notna(), df["종가"])

            for h in horizons:
                future_ymd = _next_trade_day(trade_days, rec_ymd, h)
                if not future_ymd:
                    continue

                close_map_future = _load_close_map(out_dir, future_ymd)
                if not close_map_future:
                    continue

                # MDD 계산 위해 중간 스냅샷도 로드(TopK만이니 부담 적음)
                mid_days = []
                for k in range(1, h + 1):
                    dd = _next_trade_day(trade_days, rec_ymd, k)
                    if dd:
                        mid_days.append(dd)

                mid_close_maps = [(_d, _load_close_map(out_dir, _d)) for _d in mid_days]

                for method in methods:
                    if method not in df.columns:
                        continue

                    dfx = df.copy()
                    dfx[method] = pd.to_numeric(dfx[method], errors="coerce").fillna(-1e9)
                    dfx = dfx.sort_values(method, ascending=False)

                    for topk in topks:
                        pick = dfx.head(topk).copy()
                        if pick.empty:
                            continue

                        codes = pick["종목코드"].tolist()

                        # epx를 float로 안전하게
                        epx = pd.to_numeric(entry_px.loc[pick.index], errors="coerce").values.astype(float)

                        # H일 후 종가
                        fut_close = np.array([close_map_future.get(c, np.nan) for c in codes], dtype=float)

                        # ✅ ret safe-divide (inf 방지)
                        ret = np.full_like(fut_close, np.nan, dtype=float)
                        ok_ret = np.isfinite(fut_close) & np.isfinite(epx) & (epx > 0)
                        ret[ok_ret] = (fut_close[ok_ret] / epx[ok_ret] - 1.0) * 100.0

                        # MDD(최저 종가 기반, 추천매수가 대비)
                        min_close = np.full_like(fut_close, np.nan)
                        for _d, cmap in mid_close_maps:
                            arr = np.array([cmap.get(c, np.nan) for c in codes], dtype=float)
                            if np.isnan(min_close).all():
                                min_close = arr
                            else:
                                min_close = np.nanmin(np.vstack([min_close, arr]), axis=0)

                        # ✅ mdd safe-divide (inf 방지)
                        mdd = np.full_like(fut_close, np.nan, dtype=float)
                        ok_mdd = np.isfinite(min_close) & np.isfinite(epx) & (epx > 0)
                        mdd[ok_mdd] = (min_close[ok_mdd] / epx[ok_mdd] - 1.0) * 100.0

                        # ✅ (D-2) n 정의 + 샘플 0이면 skip
                        valid = np.isfinite(ret) & np.isfinite(mdd)
                        n = int(valid.sum())
                        if n == 0:
                            continue

                        r = ret[valid]
                        md = mdd[valid]

                        rows.append({
                            "추천일": rec_ymd,
                            "비교종가일": future_ymd,
                            "H(영업일)": h,
                            "METHOD": method,
                            "TOPK": topk,
                            "N": n,
                            "WIN_RATE_%": round(float((r > 0).mean() * 100), 1),
                            "AVG_RET_%": round(float(np.nanmean(r)), 2),
                            "MED_RET_%": round(float(np.nanmedian(r)), 2),
                            "HIT_2%_%": round(float((r >= 2).mean() * 100), 1),
                            "HIT_5%_%": round(float((r >= 5).mean() * 100), 1),
                            "AVG_MDD_%": round(float(np.nanmean(md)), 2),
                            "WORST_MDD_%": round(float(np.nanmin(md)), 2),
                        })

        if not rows:
            log("⚠️ rank validation: 계산 가능한 샘플이 없어 리포트 생략")
            return

        detail = pd.DataFrame(rows)

        # 요약(가중치: 샘플 수 N)
        def _wavg(g, col):
            w = g["N"].values
            x = g[col].values
            return float(np.nansum(x * w) / np.nansum(w))

        grp = detail.groupby(["METHOD", "TOPK", "H(영업일)"], as_index=False)
        summary = grp.apply(lambda g: pd.Series({
            "TOTAL_N": int(g["N"].sum()),
            "WIN_RATE_%": round(_wavg(g, "WIN_RATE_%"), 1),
            "AVG_RET_%": round(_wavg(g, "AVG_RET_%"), 2),
            "MED_RET_%": round(float(np.nanmedian(g["MED_RET_%"].values)), 2),
            "HIT_2%_%": round(_wavg(g, "HIT_2%_%"), 1),
            "HIT_5%_%": round(_wavg(g, "HIT_5%_%"), 1),
            "AVG_MDD_%": round(_wavg(g, "AVG_MDD_%"), 2),
            "WORST_MDD_%": round(float(np.nanmin(g["WORST_MDD_%"].values)), 2),
        }), include_groups=False).reset_index(drop=True)

        detail_path = os.path.join(out_dir, f"rank_validation_{asof_ymd}.csv")
        summ_path = os.path.join(out_dir, f"rank_validation_summary_{asof_ymd}.csv")
        detail_latest = os.path.join(out_dir, "rank_validation_latest.csv")
        summ_latest = os.path.join(out_dir, "rank_validation_summary_latest.csv")

        detail.to_csv(detail_path, index=False, encoding=UTF8)
        summary.to_csv(summ_path, index=False, encoding=UTF8)
        detail.to_csv(detail_latest, index=False, encoding=UTF8)
        summary.to_csv(summ_latest, index=False, encoding=UTF8)

        log(f"📊 Rank Validation 저장 완료 → {detail_path}")
        log(f"📊 Rank Validation Summary 저장 완료 → {summ_path}")

    except Exception as e:
        log(f"⚠️ rank validation 실패: {e}")

# ------------------------------- 거래일/시총 -------------------------------

def _has_ohlcv_and_mcap(ymd: str) -> bool:
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            o = safe_ohlcv_by_ticker(ymd, market=m)
            if o is not None and not o.empty and "거래대금" in o.columns and _safe_sum(o["거래대금"]) > 0:
                return True
        except Exception:
            pass
    return False

def find_latest_valid_date(check_fn, max_back_days: int = 14) -> str:
    """
    [수정됨] 데이터 확인(check_fn)이 모두 실패하더라도(IP차단 등),
    최후의 수단으로 '가장 최근의 평일(금요일)'을 반환하여 
    일요일/휴장일 에러를 방지합니다.
    """
    now = now_kst()
    
    # 1. 장 마감 전이면 어제로 설정
    if now.hour < 18:
        now -= timedelta(days=1)

    # 2. 주말이면 금요일로 조정 (1차 보정)
    while now.weekday() >= 5: # 5:토, 6:일
        now -= timedelta(days=1)
    
    # 최후의 보루: 탐색이 다 실패했을 때 사용할 '최근 평일' 저장
    fallback_date = now.strftime("%Y%m%d")

    # 3. 데이터 존재 여부 확인 (최대 N일 과거까지)
    d = now
    for _ in range(max_back_days):
        # 탐색 중 주말 만나면 스킵
        while d.weekday() >= 5:
            d -= timedelta(days=1)
            
        ymd = d.strftime("%Y%m%d")
        
        # 데이터가 있다고 확인되면 그 날짜 반환
        if check_fn(ymd):
            log(f"✅ 유효한 거래일 확정: {ymd}")
            return ymd
        
        d -= timedelta(days=1)

    # 4. 모든 확인이 실패하면(IP차단 등), 에러 내지 말고 '최근 평일'로 강제 진행
    log(f"⚠️ 날짜 확인 실패(IP차단 가능성) -> 최근 평일({fallback_date})로 강제 설정")
    return fallback_date

def resolve_trade_date(force_ymd: Optional[str] = None) -> str:
    """
    - force_ymd가 주어지면 그 날짜에서 유효한 가장 가까운 영업일을 탐색
    - 없으면 오늘 기준으로 자동 탐색
    """
    if force_ymd:
        try:
            base = datetime.strptime(force_ymd, "%Y%m%d").date()
        except Exception:
            log(f"⚠️ 잘못된 날짜 형식(YYYYMMDD 아님): {force_ymd}, 자동 탐색으로 전환")
            return find_latest_valid_date(_has_ohlcv_and_mcap, max_back_days=10)

        def _check(ymd: str) -> bool:
            return _has_ohlcv_and_mcap(ymd)

        d = base
        last_ymd = d.strftime("%Y%m%d")
        for _ in range(10):
            ymd = d.strftime("%Y%m%d")
            if _check(ymd):
                return ymd
            d -= timedelta(days=1)
            last_ymd = d.strftime("%Y%m%d")
        return last_ymd

    return find_latest_valid_date(_has_ohlcv_and_mcap, max_back_days=10)

def build_mcap_map(ref_ymd: Optional[str] = None) -> Tuple[Dict[str, float], str]:
    # ✅ (1) 함수 첫 줄 방어코드: PYKRX 자체가 없으면 "빈 맵"으로 안전 종료
    if (not PYKRX_OK) or (stock is None):
        use = ref_ymd or now_kst().strftime("%Y%m%d")
        log("⚠️ PYKRX 사용 불가 → 시총 맵 비활성(빈 맵 반환)")
        return {}, use

    # ✅ (2) _check_mcap을 safe wrapper 기반으로 교체
    def _check_mcap(ymd: str) -> bool:
        a = safe_market_cap_by_ticker(ymd, market="KOSPI")
        b = safe_market_cap_by_ticker(ymd, market="KOSDAQ")
        return (a is not None and not a.empty) or (b is not None and not b.empty)

    use: Optional[str] = None

    # 1순위: ref_ymd 그대로 시도
    if ref_ymd and _check_mcap(ref_ymd):
        use = ref_ymd

    # 2순위: 자동 탐색
    if use is None:
        use = find_latest_valid_date(_check_mcap, max_back_days=10)

    try:
        parts = []
        a = safe_market_cap_by_ticker(use, market="KOSPI")
        b = safe_market_cap_by_ticker(use, market="KOSDAQ")
        if a is not None and not a.empty: parts.append(a)
        if b is not None and not b.empty: parts.append(b)

        df = pd.concat(parts) if parts else pd.DataFrame()
        if df.empty:
            log(f"⚠️ 시가총액 맵이 비어 있음(use={use}), 빈 맵 반환")
            return {}, use

        df["Code"] = df.index.astype(str).str.zfill(6)
        mcap_map = dict(zip(df["Code"], df["시가총액"] / 1e8))  # 억원
        return mcap_map, use

    except Exception as e:
        log(f"⚠️ 시가총액 맵 생성 실패({use}): {e}")
        return {}, use

def get_mcap_eok_from_map(mcap_map: Dict[str, float], ticker: str) -> float:
    return float(mcap_map.get(str(ticker).zfill(6), 0))

# ------------------------------- 업종 맵핑 -------------------------------

def get_fallback_sector_map() -> Dict[str, str]:
    return {
        "005930": "전기전자", "000660": "전기전자", "373220": "전기전자", "207940": "의약품",
        "005380": "운수장비", "005935": "전기전자", "068270": "의약품", "000270": "운수장비",
        "105560": "금융업", "005490": "철강금속", "035420": "서비스업", "035720": "서비스업",
        "006400": "전기전자", "051910": "화학", "012330": "화학", "028260": "유통업",
        "055550": "금융업", "086790": "금융업", "032830": "금융업", "003550": "화학",
        "015760": "전기가스업", "034020": "기계", "010120": "전기전자", "323410": "서비스업",
        "259960": "서비스업", "011200": "운수창고", "000810": "금융업", "018260": "서비스업",
        "010130": "철강금속", "009150": "전기전자", "033780": "금융업", "017670": "통신업",
        "329180": "운수장비", "096770": "화학", "003490": "운수창고", "030200": "통신업",
        "316140": "금융업", "000100": "의약품", "251270": "서비스업", "024110": "금융업",
        "036570": "서비스업", "086280": "운수창고", "090430": "화학", "010950": "화학",
        "009540": "운수장비", "267260": "전기전자", "042700": "전기전자", "010620": "화학",
        "138040": "금융업", "034730": "서비스업", "241560": "화학", "000150": "기계",
        "298040": "전기전자", "108490": "기계", "466100": "기계", "437730": "운수장비",
        "098460": "기계", "277810": "기계", "352820": "서비스업", "253450": "서비스업"
    }

def get_sector_map_krx() -> Dict[str, str]:
    """
    KIND(상장법인 목록) 기준 업종 맵 생성
    - corpList.do?method=download 는 사실상 HTML 테이블이므로 read_html 사용
    - '종목코드', '업종' 기준으로 맵 구성
    """
    ensure_dir(OUT_DIR)
    cache_path = os.path.join(OUT_DIR, "sector_map_krx.csv")

    # 1) 캐시 먼저 시도
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype=str)
            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
            df["업종"] = df["업종"].fillna("기타")
            log(f"📁 KIND 업종 캐시 로드 성공: {len(df)} rows")
            return dict(zip(df["종목코드"], df["업종"]))
        except Exception as e:
            log(f"⚠️ KIND 업종 캐시 로드 실패. 재다운로드 시도: {e}")

    # 2) 웹에서 다시 다운로드
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
    try:
        # KIND는 POST 로 파라미터 넣어 요청하는 게 가장 안정적
        data = {
            "method": "download",
            "orderMode": "1",      # 정렬 기준
            "orderStat": "D",      # 내림차순
            "searchType": "13",    # 상장법인
            "fiscalYearEnd": "all",
            "location": "all",
        }
        resp = requests.post(
            url,
            data=data,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
        )
        resp.raise_for_status()

        dfs = pd.read_html(io.BytesIO(resp.content), header=0)
        if not dfs:
            log("⚠️ KIND 테이블 파싱 실패: 테이블이 비어 있음")
            return {}

        df = dfs[0]
        df.columns = [str(c).strip() for c in df.columns]  # ✅ 여기 추가

        if "종목코드" not in df.columns or "업종" not in df.columns:
            log(f"⚠️ KIND CSV 컬럼 이상: {df.columns.tolist()}")
            return {}

        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        df["업종"] = df["업종"].replace("", np.nan).fillna("기타")

        # 필요 컬럼만 저장
        df_out = df[["종목코드", "업종"]].copy()
        df_out.to_csv(cache_path, index=False, encoding=UTF8)

        log(f"✅ KIND 업종 다운로드/파싱 완료 ({len(df_out)} rows)")
        return dict(zip(df_out["종목코드"], df_out["업종"]))

    except Exception as e:
        log(f"❌ KIND 업종 다운로드 실패(최종): {e}")
        return {}

def get_sector_map_fdr() -> Dict[str, str]:
    """
    FDR 기반 업종 맵
    - FDR에 'Sector' / 'Wics' / 'Industry' 같은 컬럼이 있을 때만 사용
    - 'Dept'(우량기업부, 기술성장 기업부 등)는 업종으로 취급하지 않음
    - KIND가 메인이고, FDR는 진짜로 '보조용'이라서 과하게 안 씀
    """
    ensure_dir(OUT_DIR)
    # 🔥 예전 sector_map_fdr.csv 대신 v2 캐시를 새로 쓴다
    cache_path = os.path.join(OUT_DIR, "sector_map_fdr_v2.csv")

    # 1) 캐시 먼저 시도
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype=str)
            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
            df["업종"] = df["업종"].fillna("기타")
            log(f"📁 FDR 업종 캐시(v2) 로드 성공: {len(df)} rows")
            return dict(zip(df["종목코드"], df["업종"]))
        except Exception as e:
            log(f"⚠️ FDR 업종 캐시(v2) 로드 실패. 재생성 시도: {e}")

    # 2) FDR에서 새로 생성
    try:
        df = fdr.StockListing("KRX")

        # 코드 컬럼 찾기
        code_col = None
        for c in ("Symbol", "Code", "ISU_CD"):
            if c in df.columns:
                code_col = c
                break
        if code_col is None:
            log(f"⚠️ FDR 코드 컬럼을 찾을 수 없음: {df.columns.tolist()}")
            return {}

        df[code_col] = df[code_col].astype(str).str.zfill(6)

        # ✅ 업종 후보 컬럼 (Dept는 일부러 제외!)
        sector_col = None
        for c in ("업종", "Sector", "Wics", "Industry"):
            if c in df.columns:
                sector_col = c
                break

        # 그런 컬럼이 하나도 없으면, FDR 업종 맵은 아예 안 쓴다
        if sector_col is None:
            log(f"⚠️ FDR에 업종/섹터 컬럼 없음 → FDR 업종 맵 사용 안 함: {df.columns.tolist()}")
            return {}

        df_out = df[[code_col, sector_col]].rename(
            columns={code_col: "종목코드", sector_col: "업종"}
        )

        # FDR에서 내려오는 이상한 값(기업부 계열)은 전부 '기타'로 처리
        bad_vals = {"기술성장 기업부", "우량기업부", "중견기업부", "기타 기업부"}
        df_out["업종"] = (
            df_out["업종"]
            .replace("", np.nan)
            .fillna("기타")
            .apply(lambda x: "기타" if str(x).strip() in bad_vals else x)
        )

        df_out.to_csv(cache_path, index=False, encoding=UTF8)
        log(f"✅ FDR 업종(v2) 생성 및 캐시 저장: {len(df_out)} rows")
        return dict(zip(df_out["종목코드"], df_out["업종"]))

    except Exception as e:
        log(f"❌ FDR 업종 생성 실패(최종): {e}")
        return {}

def load_sector_override() -> Dict[str, str]:
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, "sector_override.csv")
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path, dtype=str)
        if "종목코드" not in df.columns or "업종" not in df.columns:
            log(f"⚠️ sector_override.csv 컬럼 이상: {df.columns.tolist()}")
            return {}
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        df["업종"] = df["업종"].fillna("기타")
        log(f"📁 업종 Override 로드: {len(df)} rows")
        return dict(zip(df["종목코드"], df["업종"]))
    except Exception as e:
        log(f"⚠️ 업종 Override 로드 실패: {e}")
        return {}

def build_sector_map() -> Dict[str, str]:
    # 1) 메인: KIND 업종
    kind_map = get_sector_map_krx()

    # 2) 서브: FDR 업종 (v2)
    fdr_map = get_sector_map_fdr()

    # 3) 하드코딩 fallback + 사용자 override
    fallback = get_fallback_sector_map()
    override = load_sector_override()

    sector_map: Dict[str, str] = {}

    # 🔹 1순위: KIND 그대로 넣기
    sector_map.update(kind_map)

    # 🔹 2순위: FDR – KIND가 없거나 '기타'일 때만 보충
    for code, sec in fdr_map.items():
        cur = sector_map.get(code)
        if (cur is None) or (str(cur).strip() == "") or (str(cur).strip() == "기타"):
            sector_map[code] = sec

    # 🔹 3순위: fallback – 여전히 비어 있는 코드만 채우기
    for code, sec in fallback.items():
        sector_map.setdefault(code, sec)

    # 🔹 4순위: 최종 수동 Override가 최상위
    sector_map.update(override)

    log(f"ℹ️ 최종 업종 맵 크기: {len(sector_map)}개 (KIND 우선 + FDR 보조 + fallback + override)")

    # 디버그용 샘플 몇 개 찍어보면 확인하기 좋음
    for test in ["005930", "000660", "035420", "005490"]:
        if test in sector_map:
            log(f"   - {test} 업종 = {sector_map[test]}")

    return sector_map

# ------------------------------- 업종 대분류 (시각화용) -------------------------------

def classify_big_sector(name: str, detailed: str) -> str:
    """
    KRX 세부업종(detailed) + 종목명(name)을 기반으로
    대분류 업종을 만들어준다.
    """
    t = (detailed or "").strip()


    # KRX 구형 업종명(전기전자/운수장비 등) fallback 대응
    if any(k in t for k in ["전기전자", "의약품", "운수장비", "철강금속", "화학", "금융업", "서비스업", "유통업", "통신업", "전기가스업", "운수창고"]):
        mapping = {
            "전기전자": "IT/전기전자",
            "의약품": "바이오·의약품",
            "운수장비": "자동차·모빌리티",
            "철강금속": "철강·금속",
            "화학": "화학·소재",
            "금융업": "금융",
            "서비스업": "서비스 기타",
            "유통업": "유통·소비재",
            "통신업": "IT/전기전자",
            "전기가스업": "인프라·에너지",
            "운수창고": "운송·물류",
        }
        for k, v in mapping.items():
            if k in t:
                return v


    # 2차전지
    if any(k in t for k in ["2차전지", "이차전지", "이차 전지", "전지"]):
        return "2차전지"
    if any(k in name for k in ["에코프로", "엘앤에프", "퓨처엠", "에너지솔루션", "SDI", "에스디아이"]):
        return "2차전지"

    # 반도체
    if "반도체" in t:
        return "반도체"
    if any(k in name for k in ["하이닉스", "DB하이텍", "한미반도체", "티씨케이", "덕산네오룩스"]):
        return "반도체"

    # 인터넷/플랫폼·게임 (먼저 체크)
    if any(k in t for k in ["포털", "인터넷"]) or any(
        k in name for k in ["네이버", "NAVER", "카카오", "크래프톤", "넷마블", "엔씨소프트"]
    ):
        return "인터넷/플랫폼·게임"

    # IT/전기전자
    if any(k in t for k in [
        "전자부품", "전자 제품", "전기장비", "컴퓨터",
        "통신 및 방송 장비", "자료처리", "소프트웨어", "정보 서비스"
    ]):
        return "IT/전기전자"

    # 자동차·모빌리티
    if any(k in t for k in ["자동차", "운수장비", "차량부품"]) or any(
        k in name for k in ["현대차", "기아", "만도", "현대모비스", "HL클라테크", "롯데렌탈"]
    ):
        return "자동차·모빌리티"

    # 조선·기계·설비
    if any(k in t for k in ["조선", "기계", "선박", "보트 건조업", "산업용 장비", "펌프", "밸브", "터빈"]):
        return "조선·기계·설비"

    # 철강·금속
    if any(k in t for k in ["철강", "1차 금속", "비철금속", "금속가공"]):
        return "철강·금속"

    # 화학·소재
    if any(k in t for k in ["화학", "플라스틱 제품", "고무제품", "합성수지", "섬유제품"]):
        return "화학·소재"

    # 바이오·의약품
    if any(k in t for k in ["의약품", "제약", "생명공학", "의료기기"]):
        return "바이오·의약품"
    if any(k in name for k in ["셀트리온", "삼성바이오로직스", "HLB"]):
        return "바이오·의약품"

    # 금융
    if any(k in t for k in ["은행", "증권", "보험", "기타 금융업", "금융 지원 서비스"]):
        return "금융"

    # 건설·부동산
    if any(k in t for k in ["건설", "주택", "부동산", "토목"]):
        return "건설·부동산"

    # 유통·소비재
    if any(k in t for k in ["도소매", "소매업", "유통업", "전자상거래"]) or any(
        k in t for k in ["음·식료품", "음료", "식품", "의복", "패션", "화장품"]
    ):
        return "유통·소비재"

    # 운송·물류
    if any(k in t for k in ["운수", "물류", "항공운송", "해상운송", "창고업", "택배"]):
        return "운송·물류"

    # 인프라·에너지 (전력/가스/전력장비 포함)
    if any(k in t for k in ["전기가스", "수도", "발전", "송전", "에너지 공급"]):
        return "인프라·에너지"
    if "전동기, 발전기 및 전기 변환 · 공급 · 제어 장치 제조업" in t:
        return "인프라·에너지"

    # 미디어·콘텐츠
    if any(k in t for k in ["방송업", "영화", "비디오물", "출판", "광고업"]):
        return "미디어·콘텐츠"

    # 서비스 기타
    if any(k in t for k in ["서비스업", "사업 지원 서비스", "기타 개인 서비스"]):
        return "서비스 기타"

    return "기타"


# ------------------------------- 벤치마크 (지수 20/60/120일 수익률) -------------------------------

def get_benchmark_returns(trade_ymd: str) -> Dict[str, Dict[int, float]]:
    """
    KOSPI, KOSDAQ 지수의 20일, 60일, 120일 수익률을 한 번에 계산
    Returns: {'KOSPI': {20: 1.5, 60: -2.0, 120: 5.1}, ...}
    """
    try:
        end = datetime.strptime(trade_ymd, "%Y%m%d").date()
    except Exception:
        return {}

    # 120일 영업일 확보를 위해 넉넉히 200일 전부터 조회 (휴장일 고려)
    start = end - timedelta(days=200 * 2) 

    res: Dict[str, Dict[int, float]] = {"KOSPI": {}, "KOSDAQ": {}}
    index_map = {"KOSPI": "KS11", "KOSDAQ": "KQ11"}
    periods = [20, 60, 120]

    for market, symbol in index_map.items():
        try:
            df = fdr.DataReader(symbol, start, end)
            if df is None or df.empty or "Close" not in df.columns:
                continue

            close = df["Close"].dropna()
            last = float(close.iloc[-1])

            for p in periods:
                if len(close) > p:
                    # p일 전 종가
                    base = float(close.iloc[-(p + 1)])
                    if base > 0:
                        ret = (last / base - 1.0) * 100
                        res[market][p] = round(ret, 2)
                    else:
                        res[market][p] = np.nan
                else:
                    res[market][p] = np.nan

        except Exception as e:
            log(f"⚠️ 벤치마크({market}) 수익률 계산 실패: {e}")
            continue

    return res

# ------------------------------- 기타 수집 로직 -------------------------------

def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    """
    [수정됨] JSONDecodeError 대응 및 3차 백업(단순 리스트) 추가
    """
    # ------------------------------------------------------------
    # 1. 1차 시도: pykrx (IP 차단 시 실패 확률 높음)
    # ------------------------------------------------------------
    try:
        frames = []
        for m in ["KOSPI", "KOSDAQ"]:
            df = safe_ohlcv_by_ticker(date_yyyymmdd, market=m)
            if df is not None and not df.empty:
                df = df.reset_index()
                code_col = next((c for c in df.columns if "코드" in str(c) or "티커" in str(c)), None)
                if code_col:
                    df = df.rename(columns={code_col: "종목코드"})
                if "거래대금" in df.columns:
                    df = df.rename(columns={"거래대금": "거래대금(원)"})
                
                if "종목코드" in df.columns and "거래대금(원)" in df.columns:
                    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
                    df["시장"] = m
                    df["거래대금(원)"] = pd.to_numeric(df["거래대금(원)"], errors="coerce").fillna(0)
                    frames.append(df[["종목코드", "시장", "거래대금(원)"]])

        if frames:
            df_all = pd.concat(frames, ignore_index=True)
            if not df_all.empty:
                log(f"✅ KRX(pykrx) 수집 성공 ({len(df_all)}종목)")
                return df_all.sort_values("거래대금(원)", ascending=False).head(top_n)
    except Exception as e:
        log(f"⚠️ KRX(pykrx) 수집 실패: {e}")

    # ------------------------------------------------------------
    # 2. 2차 시도: FinanceDataReader (JSON 에러 방어)
    # ------------------------------------------------------------
    log("🔄 2차 시도: FDR(FinanceDataReader)로 수집 시도...")
    try:
        import json
        # FDR StockListing은 날짜 인자가 없으면 최근일 기준 조회
        df_krx = fdr.StockListing("KRX")
        
        rename_map = {"Code": "종목코드", "Amount": "거래대금(원)", "Market": "시장"}
        df_krx = df_krx.rename(columns=rename_map)
        df_krx["종목코드"] = df_krx["종목코드"].astype(str).str.zfill(6)
        
        # 거래대금 컬럼이 없으면(FDR 버전에 따라 다름) 0으로 채움
        if "거래대금(원)" not in df_krx.columns:
            df_krx["거래대금(원)"] = 0 
            
        df_krx = df_krx[df_krx["시장"].isin(["KOSPI", "KOSDAQ"])].copy()
        log(f"✅ FDR 수집 성공 ({len(df_krx)}종목)")
        return df_krx.sort_values("거래대금(원)", ascending=False).head(top_n)

    except (json.decoder.JSONDecodeError, ValueError, Exception) as e:
        log(f"⚠️ FDR 수집 실패 (IP차단 의심): {e}")

    # ------------------------------------------------------------
    # 3. 3차 시도 (최후의 보루): 섹터 맵 활용
    # ------------------------------------------------------------
    log("🚨 3차 시도: KIND 섹터 맵을 이용해 종목 리스트만 복구합니다.")
    try:
        # 이미 로드된/저장된 sector_map_krx.csv가 있다면 활용
        sec_map = get_sector_map_krx()
        if sec_map:
            # 리스트만 만들고 거래대금은 0으로 설정 (분석 단계에서 걸러지더라도 프로그램은 안 죽음)
            rows = [{"종목코드": k, "종목명": "Unknown", "시장": "KOSPI", "거래대금(원)": 0} for k in sec_map.keys()]
            df_fallback = pd.DataFrame(rows)
            log(f"✅ KIND 섹터 맵 기반 복구 성공 ({len(df_fallback)}종목)")
            # 거래대금이 0이라 Top N 정렬은 의미 없지만, 분석은 진행 가능
            return df_fallback.head(top_n)
            
    except Exception as e:
        log(f"❌ 3차 복구 실패: {e}")

    raise RuntimeError(f"모든 데이터 수집 수단 실패 ({date_yyyymmdd})")

def get_market_sets(d: str) -> Tuple[set, set]:
    try:
        kospi = set(safe_ticker_list(d, market="KOSPI"))
        kosdaq = set(safe_ticker_list(d, market="KOSDAQ"))
        return kospi, kosdaq
    except Exception:
        return set(), set()

def get_name_map_cached(d: str) -> Dict[str, str]:
    """
    종목코드 -> 종목명 매핑 생성 (FDR 우선 사용 -> 실패 시 pykrx 시도)
    """
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, f"krx_codes_{d}.csv")
    
    # 1. 이미 저장된 캐시 파일이 있으면 로드
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str)
            return dict(zip(df["종목코드"], df["종목명"]))
        except Exception:
            pass

    log("🔄 종목명 매핑 정보 생성 중... (FDR 우선)")
    name_map = {}

    # 2. FDR로 전체 종목명 한방에 가져오기 (가장 확실하고 빠름)
    try:
        # KRX 전체 상장 종목 리스트 다운로드
        df_fdr = fdr.StockListing("KRX")
        
        # FDR 버전에 따라 컬럼명이 Code/Symbol로 다를 수 있음
        code_col = "Code" if "Code" in df_fdr.columns else ("Symbol" if "Symbol" in df_fdr.columns else None)
        
        if code_col and "Name" in df_fdr.columns:
            for _, row in df_fdr.iterrows():
                code = str(row[code_col]).strip().zfill(6)
                name = str(row["Name"]).strip()
                name_map[code] = name
            log(f"✅ FDR 종목명 확보 완료: {len(name_map)}개")
    except Exception as e:
        log(f"⚠️ FDR 종목명 조회 실패: {e}")

    # 3. FDR 실패했거나 누락된 게 있다면 pykrx로 보완 (기존 방식)
    # (FDR이 성공했다면 이 부분은 거의 실행되지 않음)
    if (not name_map) and PYKRX_OK and (stock is not None):
        log("🔄 FDR 데이터 부족 -> pykrx로 개별 조회 시도 (느림)")
        for m in ["KOSPI", "KOSDAQ"]:
            try:
                tickers = safe_ticker_list(d, market=m)
                for t in tickers:
                    code = str(t).zfill(6)
                    # 이미 FDR로 확보했으면 패스
                    if code in name_map:
                        continue
                        
                    nm = safe_ticker_name(t)
                    if nm:
                        name_map[code] = nm
                    else:
                        name_map[code] = code # 이름 없으면 코드라도 사용
                    time.sleep(0.001) 
            except:
                pass

    # 4. 결과 저장 및 반환
    if name_map:
        rows = [{"종목코드": c, "종목명": n} for c, n in name_map.items()]
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding=UTF8)
        return name_map
    
    return {}

def save_price_snapshot(trade_ymd: str, name_map: Dict[str, str]) -> None:
    """
    trade_ymd 기준으로 KOSPI/KOSDAQ 전 종목 '종가' 스냅샷을 저장한다.
    - data/price_snapshot_YYYYMMDD.csv
    - data/price_snapshot_latest.csv
    - DuckDB: price_snapshots 테이블
    """
    ensure_dir(OUT_DIR)
    frames: List[pd.DataFrame] = []

    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = safe_ohlcv_by_ticker(trade_ymd, market=m)
            if df is None or df.empty:
                continue

            df = df.reset_index()

            # 코드 컬럼 찾기
            code_col = None
            for c in df.columns:
                if "티커" in str(c) or "코드" in str(c) or "종목코드" in str(c):
                    code_col = c
                    break

            if code_col is None or "종가" not in df.columns:
                log(f"⚠️ 가격 스냅샷({m}) 컬럼 이상: {df.columns.tolist()}")
                continue

            df["종목코드"] = df[code_col].astype(str).str.zfill(6)
            df["시장"] = m
            df["종목명"] = df["종목코드"].map(name_map).fillna("")

            frames.append(df[["종목코드", "종목명", "시장", "종가"]])
        except Exception as e:
            log(f"⚠️ 가격 스냅샷({m}) 수집 실패: {e}")
            continue

    if not frames:
        log(f"❌ 가격 스냅샷 생성 실패: 데이터 없음({trade_ymd})")
        return

    snap = pd.concat(frames, ignore_index=True)

    # 1. 기존 CSV 저장 (유지)
    dated = os.path.join(OUT_DIR, f"price_snapshot_{trade_ymd}.csv")
    latest = os.path.join(OUT_DIR, "price_snapshot_latest.csv")
    snap.to_csv(dated, index=False, encoding=UTF8)
    snap.to_csv(latest, index=False, encoding=UTF8)
    log(f"💾 가격 스냅샷 CSV 저장 완료")

    # 🔥 [2. DuckDB 저장 추가] --------------------------
    try:
        db = LDYDBManager()
        db.save_snapshot(snap, trade_ymd)
        db.close()
    except Exception as e:
        log(f"⚠️ 스냅샷 DB 저장 실패: {e}")
    # --------------------------------------------------


# ------------------------------- AI 코멘트 / 스코어 -------------------------------

def fetch_naver_news_headlines(code: str, days: int = 2) -> List[str]:
    """
    [v9.0] 네이버 금융 종목별 뉴스/공시 제목 크롤링
    """
    if not BS4_OK: return []
    
    headlines = []
    try:
        url = f"https://finance.naver.com/item/news_news.naver?code={code}&page=1"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=3)
        
        soup = BeautifulSoup(resp.content.decode('euc-kr', 'replace'), "html.parser")
        titles = soup.select("td.title > a")
        dates = soup.select("td.date")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for t, d in zip(titles, dates):
            article_date_str = d.text.strip()
            try:
                if len(article_date_str) > 10:
                    a_date = datetime.strptime(article_date_str, "%Y.%m.%d %H:%M")
                else:
                    a_date = datetime.strptime(article_date_str, "%Y.%m.%d")
                
                if a_date >= cutoff_date:
                    subject = t.text.strip()
                    if "특징주" in subject or "공시" in subject or "체결" in subject or "계약" in subject:
                        headlines.append(subject)
            except:
                continue
    except Exception as e:
        pass
        
    return list(set(headlines))[:10]

def analyze_sentiment_llm(stock_name: str, headlines: List[str]) -> Tuple[float, str]:
    """
    [v9.0] LLM을 이용해 뉴스 헤드라인의 호재/악재 여부를 판별
    Returns: (점수 -5 ~ +5, "요약 코멘트")
    """
    if not LLM_AVAILABLE or not headlines:
        return 0.0, ""

    # 프롬프트 엔지니어링
    news_text = "\n".join(headlines)
    prompt = f"""
    당신은 주식 시장 전문가입니다. 아래는 '{stock_name}' 종목의 최근 뉴스입니다.
    이 뉴스들이 주가에 미칠 영향을 분석하여 점수(-5:매우악재 ~ 0:중립 ~ +5:매우호재)와
    한 줄 요약을 JSON 형식으로 답하세요.
    
    [뉴스 목록]
    {news_text}
    
    [응답 형식]
    {{"score": 점수, "reason": "핵심재료 요약"}}
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # JSON 파싱 (간이)
        import json
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        data = json.loads(text)
        score = float(data.get("score", 0))
        reason = data.get("reason", "")
        
        return score, reason
        
    except Exception as e:
        # log(f"⚠️ LLM 분석 실패: {e}")
        return 0.0, ""


def generate_ai_comment(
    mfi: float, rsi: float, slope: float, disp: float, score: float, 
    ttm_squeeze: int = 0, bw_squeeze: int = 0, bb_bw: float = np.nan, squeeze_cnt: int = 0,
    is_swing_support: bool = False, v_power: float = 0.0
) -> str:
    comment = ""

    # 1. 스퀴즈 상태
    if int(ttm_squeeze) >= 1:
        if squeeze_cnt >= 5: 
            comment += f"🌪️ {squeeze_cnt}일 연속 응축 중! 폭발 임박. "
        else: 
            comment += f"🌪️ 에너지 응축 시작 ({squeeze_cnt}일차). "
    elif int(bw_squeeze) >= 1:
        if np.isfinite(bb_bw): 
            comment += f"🔇 변동성 축소 (BW {float(bb_bw):.1f}%). "
        else: 
            comment += "🔇 변동성 축소 구간. "

    # 2. 구조적 지지/저항 (v7.5)
    if is_swing_support:
        comment += "🧱 구조적 지지선(Swing Low) 근접. "

    # 3. 매수세 강도 (v7.5)
    if v_power >= 1.5:
        comment += "💪 매수세 장악(Power Buying). "
    elif v_power <= 0.6:
        comment += "📉 매도세 우위. "

    # 4. 수급 (MFI)
    if mfi >= 70: 
        comment += "💰 외국인/기관 수급 집중. "
    elif mfi >= 60: 
        comment += "💸 자금 유입 지속. "

    # 5. 모멘텀 (MACD Slope)
    if slope > 0.02: 
        comment += "🚀 단기 모멘텀 가속. "
    elif slope > 0.005: 
        comment += "📈 상승 모멘텀 개선. "

    # 6. 이격도
    if -2 <= disp <= 2: 
        comment += "✅ 20일선 눌림목 구간."
    elif disp > 5: 
        comment += "⚠️ 단기 급등 조정 주의."
    elif disp < -5: 
        comment += "📉 과매도 기술적 반등 기대."

    # 7. 종합 점수 평가
    if score >= 90: 
        comment += " (강력 매수)"
    elif score >= 80: 
        comment += " (매수 유효)"   

    # ✅ [v7.4 수정] 스퀴즈 일수 포함 코멘트
    if int(ttm_squeeze) >= 1:
        if squeeze_cnt >= 5:
            comment += f"🌪️ {squeeze_cnt}일 연속 응축 중! 폭발 임박 (Hot Zone). "
        else:
            comment += f"🌪️ 에너지 응축 시작 ({squeeze_cnt}일차). "
    elif int(bw_squeeze) >= 1:
        if np.isfinite(bb_bw):
            comment += f"🔇 변동성 축소 (BW {float(bb_bw):.1f}%). "
        else:
            comment += "🔇 변동성 축소 구간. "

    if mfi >= 70:
        comment += "💰 외국인/기관 수급 집중. "
    elif mfi >= 60:
        comment += "💸 자금 유입 지속. "

    if slope > 0.02:
        comment += "🚀 단기 모멘텀 가속. "
    elif slope > 0.005:
        comment += "📈 상승 모멘텀 개선. "
    elif slope > 0:
        comment += "↗️ 약한 플러스 모멘텀. "

    if -2 <= disp <= 2:
        comment += "✅ 20일선 눌림목 구간."
    elif disp > 5:
        comment += "⚠️ 단기 급등 조정 주의."
    elif disp < -5:
        comment += "📉 과매도 기술적 반등 기대."

    if score >= 90:
        comment += " (강력 매수)"
    elif score >= 80:
        comment += " (매수 유효)"

    return comment if comment else "특이사항 없음."

def cap_q(s: pd.Series, q: int = 90, floor: float = 1.0) -> float:
    c = np.nanpercentile(nz_num(s), q)
    return float(max(c, floor)) if np.isfinite(c) else floor

def pct_norm_pos(s: pd.Series, q: int = 90, floor: float = 1.0) -> pd.Series:
    s = nz_num(s).clip(lower=0)
    return np.clip(s / cap_q(s, q, floor), 0, 1)

def safe_quantile(s, q, fallback=0.0):
    """
    Pandas Series에서 안전하게 분위수를 계산합니다.
    데이터가 없거나 에러 발생 시 fallback(기본값)을 반환하여 시스템 정지를 막습니다.
    """
    if s is None:
        return fallback
    try:
        # 데이터가 Series 형태이고 비어있지 않은지 확인
        if hasattr(s, 'empty') and s.empty:
            return fallback
        
        v = s.quantile(q)
        # 결과가 NaN이면 fallback 반환, 아니면 float로 변환
        return fallback if pd.isna(v) else float(v)
    except:
        return fallback

def inv_dist_norm(dist: pd.Series, cap: float) -> pd.Series:
    return np.clip(1 - (nz_num(dist) / cap), 0, 1)

def detect_regime_row(row: pd.Series) -> str:
    """
    추세 단계(REGIME)를 텍스트로 분류
    - rel_60d_% : 60일 초과수익(α)
    - MACD_Slope : 단기 모멘텀 기울기
    - RSI14 : 과매수/과매도 판단
    """
    # ✅ [수정됨] 0.0 값을 제대로 가져오도록 로직 변경
    def _fv(key: str, default: float = 0.0) -> float:
        try:
            val = row.get(key)
            if val is None or pd.isna(val):
                return default
            return float(val)
        except Exception:
            return default

    rel60 = _fv("rel_60d_%", 0.0)
    slope = _fv("MACD_Slope_PCT", 0.0)
    if slope == 0.0:
        slope = _fv("MACD_Slope", 0.0)
    rsi = _fv("RSI14", 50.0)

    # ① 강한 상승 추세
    if rel60 > 10 and slope > 0 and 50 <= rsi <= 70:
        return "① 강한 상승 추세"

    # ② 상승 후 조정 구간 (상대강도는 높은데 모멘텀 둔화)
    if rel60 > 5 and slope <= 0:
        return "② 상승 후 조정"

    # ③ 박스 / 중립
    if -5 <= rel60 <= 5:
        return "③ 박스 / 중립"

    # ④ 바닥 반등 시도 (상대강도는 약하지만 모멘텀 플러스 전환)
    if rel60 <= -5 and slope > 0:
        return "④ 바닥 반등 시도"

    # ⑤ 하락 / 약세
    return "⑤ 하락 / 약세"

# --- [새로 추가 1] 곡선형 패널티 계산 함수 ---
def apply_curve_penalty(val, threshold, power=2.0, weight=1.0):
    """
    [v15.0] 곡선형 패널티 (Curved Penalty)
    임계치(threshold)를 넘어가면 감점 폭이 제곱(power)으로 커짐
    """
    if val <= threshold:
        return 0.0
    # (초과분 ^ power) * 가중치
    return ((val - threshold) ** power) * weight


# --- [새로 추가 2] 상태 머신 기반 ROUTE 결정 함수 (기존 route_tag 대체) ---
def determine_state(row):
    """
    [Strict] ATTACK 조건에 '가격 위치(고점 근접)' 추가
    """
    try:
        # 데이터 추출
        rsi = float(row.get('RSI14', 50))
        r5 = float(row.get('ret_5d_%', 0))
        above_ma20 = int(row.get('Above_MA20', 0))
        slope = float(row.get('MACD_Slope_PCT', 0))
        t_score = float(row.get('TRIGGER_SCORE', 0))
        is_squeeze = int(row.get('TTM_SQUEEZE', 0))
        vol_qual = float(row.get('Vol_Quality', 1.0))
        
        # ✅ [New] Range Position (20일 고가 대비 현재가 위치) 확인
        # range_pos >= 0.85 (상위 15% 구간) 이어야 진짜 돌파 임박
        range_pos = float(row.get('Range_Pos', 0)) 

        # 1. 과열 (OVERHEAT)
        if rsi >= 75 or r5 >= 20.0:
            return RouteState.OVERHEAT

        # 2. 집중 공략 (ATTACK): 조건 강화
        # - 20일선 위 & 모멘텀 양수 & 트리거 60점 이상
        # - 수급 질(Vol_Qual) 1.3 이상
        # - [추가] 박스권 상단(Range_Pos >= 0.8)에 있어야 함 (바닥에서 기는 종목 제외)
        if above_ma20 == 1 and slope > 0 and t_score >= 60 and vol_qual >= 1.3:
            if range_pos >= 0.8:  # 👈 추가된 조건: 신고가 영역 근접
                return RouteState.ATTACK

        # 3. 발사 임박 (ARMED)
        if is_squeeze == 1 and above_ma20 == 1:
            return RouteState.ARMED
        if vol_qual >= 2.0:
            return RouteState.ARMED

        # 4. 추세 대기 (WAIT)
        if float(row.get('Low_Trend_PCT', 0)) > 0:
            return RouteState.WAIT

        return RouteState.NEUTRAL

    except Exception as e:
        return RouteState.NEUTRAL

# -----------------------------------------------------------
# [New] Scoring Engine v15.1 (Safety Patched)
# -----------------------------------------------------------

def calculate_ebs_independent(row) -> int:
    """[독립형 EBS] 5가지 펀더멘털 체크리스트 (0~10점)"""
    score = 0
    if row.get('Low_Trend_PCT', 0) > 0: score += 2
    if row.get('Vol_Quality', 0) >= 1.1: score += 2
    if row.get('MACD_Slope_PCT', 0) > 0: score += 2
    rsi = row.get('RSI14', 50)
    if 45 <= rsi <= 70: score += 2
    if row.get('TTM_SQUEEZE', 0) == 1 or row.get('BB_Expanding', 0) == 1: score += 2
    return score

def calculate_structural_score(row) -> float:
    """[STRUCT_SCORE] 종목의 기초 체력 (0~100)"""
    def _norm(val, max_val): return min(max(val / max_val, 0), 1)
    
    trend_score = _norm(row.get('Low_Trend_PCT', 0), 3.0) * 40
    mfi_score = _norm(row.get('MFI14', 50) - 30, 40) * 15 
    vq_score = _norm(row.get('Vol_Quality', 0) - 0.8, 1.2) * 15
    range_score = _norm(row.get('Range_Pos', 0), 1.0) * 15
    
    disp = row.get('이격도', 0)
    if 0 <= disp <= 5: disp_score = 15
    elif disp < 0: disp_score = 5
    else: disp_score = max(15 - (disp - 5), 0)
    
    base = trend_score + mfi_score + vq_score + range_score + disp_score
    penalty = 0
    if row.get('Above_MA20', 0) == 0: penalty += 20
    
    return max(0.0, base - penalty) # 하한 0점 고정

def calculate_timing_score(row) -> float:
    """
    [v8.7 최종 정밀 버전] TIMING_SCORE 계산
    - 매물대(Volume Profile) 지표 반영 (연속 스케일 방식)
    - NaN 전염 방지용 Safety Float 적용
    - 과열 및 갭 패널티 유지
    """
    # --- [내부 방어용 헬퍼 함수] ---
    def _safe_float(x, default=0.0):
        try:
            if x is None: return default
            val = float(x)
            return default if np.isnan(val) else val
        except: return default

    # 1. 기초 데이터 로드 (Main에서 보존한 Raw Score 사용)
    raw = _safe_float(row.get('RAW_TRIGGER_SCORE', row.get('TRIGGER_SCORE', 0)))
    std_trigger = min(raw / 90.0 * 100.0, 100.0)
    
    bonus = 0
    penalty = 0

    # 2. 매물대(Volume Profile) 보정 로직
    res_all = _safe_float(row.get('RES_RATIO', 0))
    res_near = _safe_float(row.get('RES_RATIO_NEAR', 0))
    poc_gap = _safe_float(row.get('POC_GAP', 0))
    is_above = int(_safe_float(row.get('IS_ABOVE_POC', 0)))

    if is_above == 1:
        # [보너스] 상단 전체 매물이 적을수록 최대 +12점
        bonus += max(0, 12 * (1 - min(res_all, 0.30) / 0.30))
        # [보너스] 근접 저항이 5% 미만으로 매우 얇으면 추가 +3점
        if res_near < 0.05: bonus += 3
        # [과열 차감] POC 대비 너무 멀면(12% 초과) 보너스 일부 회수 (음수 방지)
        if poc_gap > 12: 
            bonus = max(0, bonus - 4)
    else:
        # [패널티] 상단 매물(저항)이 두꺼울수록 최대 -15점
        penalty += min(15, 15 * (min(res_all, 0.45) / 0.45))
        # [패널티] 당장 머리 위(Near) 저항이 20%를 넘으면 추가 -5점
        if res_near > 0.20: penalty += 5

    # 3. 기존 기술적 보너스 (TTM Squeeze, Supertrend)
    if int(_safe_float(row.get('TTM_SQUEEZE', 0))) == 1: bonus += 10
    if int(_safe_float(row.get('SUPERTREND_DIR', 0))) == 1: bonus += 5
    
    # 4. 기존 기술적 패널티 (RSI 과열, 갭 상승)
    rsi = _safe_float(row.get('RSI14', 50))
    gap_pct = _safe_float(row.get('gap_pct', 0))
    
    if rsi > 75: penalty += 20
    if gap_pct > 5.0: penalty += 10 # 갭이 너무 크면 추격 리스크 감점
    
    # ✅ 최종 점수 0~100 안전 Clamp (음수 방지 및 상한 고정)
    return max(0.0, min(std_trigger + bonus - penalty, 100.0))

def determine_state_dynamic(row, thresholds: dict):
    """
    [v8.5 Elite Patch] 수급 상대 비중 기반 Toxic Filter 및 동적 상태 결정 로직
    - 기존 고정 수치(10억) 필터를 '거래대금 대비 비중(%)'으로 전면 개편
    - 지표 불일치 및 급락 추세 방어 로직 강화
    """
    try:
        # 1. 기초 데이터 추출 (안전한 float 변환)
        def _get(k, default=0.0):
            val = row.get(k, default)
            try: return float(val) if not pd.isna(val) else default
            except: return default

        # 기술적 지표
        rsi = _get('RSI14', 50)
        r1 = _get('ret_1d_%', 0)
        r5 = _get('ret_5d_%', 0)
        slope = _get('MACD_Slope_PCT', 0)
        range_pos = _get('Range_Pos', 0)
        vol_qual = _get('Vol_Quality', 1.0)
        t_score = _get('TIMING_SCORE', 0)
        vol_z = _get('거래강도', 0)
        low_trend = _get('Low_Trend_PCT', 0)
        above_ma20 = int(_get('Above_MA20', 0))

        # 수급 데이터 및 거래대금
        turnover = _get('거래대금(원)', 1.0) # ZeroDivision 방지 위해 기본값 1.0
        frg_net = _get('외인순매수', 0)
        ind_net = _get('개인순매수', 0)

        # 2. [Elite] 상대적 수급 비중 계산
        # 전체 거래대금 중 외인/개인이 차지하는 순매수 비중 (%)
        frg_ratio = (frg_net / turnover) * 100 if turnover > 0 else 0
        ant_ratio = (ind_net / turnover) * 100 if turnover > 0 else 0

        # -----------------------------------------------------------
        # 🚨 [Elite TOXIC Filter] 설거지 및 폭탄 돌리기 정밀 감지
        # -----------------------------------------------------------
        is_toxic = False
        
        # [Case 1] 전형적인 설거지: 주가는 오르는데 외인은 거래의 20%를 던지고 개인이 다 받음
        if r1 > 5.0 and frg_ratio < -20.0 and ant_ratio > 20.0:
            is_toxic = True
            
        # [Case 2] 수급 불명확 + 광기: 거래량이 평소의 10배 이상 터지며 10% 급등 (단기 고점 징후)
        elif vol_z >= 10.0 and r1 >= 10.0:
            is_toxic = True

        if is_toxic:
            return "EXIT_WARNING" # ☠️ 절대 진입 금지 (폭탄 돌리기)

        # -----------------------------------------------------------
        # 3. 상태 결정 파이프라인 (Tiered Logic)
        # -----------------------------------------------------------
        
        # (1) 과열 (Absolute Cut)
        if rsi >= 75 or r5 >= 25.0:
            return "OVERHEAT"

        # (2) 집중 공략 (ATTACK) 조건: 추세+수급+타이밍 삼박자
        vol_cut = thresholds.get('vol_q75', 1.2)
        range_cut = thresholds.get('range_q75', 0.8)
        
        if (slope > 0 
            and range_pos >= range_cut
            and vol_qual >= vol_cut
            and t_score >= 60
            and above_ma20 == 1):
            
            # [추세 붕괴 방어] 저점이 -3% 이상 깨진 종목은 ATTACK 금지
            if low_trend < -3.0: 
                return "WAIT" 
            return "ATTACK"

        # (3) 발사 임박 (ARMED): 변동성 응축 및 돌파 대기
        is_squeeze = int(row.get('TTM_SQUEEZE', 0))
        if (is_squeeze == 1 or vol_qual >= 2.0) and above_ma20 == 1:
            if low_trend >= -3.0: # 추세가 살아있을 때만
                return "ARMED"
        
        # (4) 추세 대기 (WAIT): 저점이 높아지거나 반등 시도
        if low_trend > 0 or r1 > 0:
            return "WAIT"

        # (5) 중립 (NEUTRAL)
        return "NEUTRAL"

    except Exception as e:
        # 에러 발생 시 안전하게 중립 반환
        return "NEUTRAL"




def build_global_score(df: pd.DataFrame, macro_risk: str) -> pd.DataFrame:
    x = df.copy()
    
    # [핵심] EBS 독립 체크리스트 계산 (정예군 편성의 기준점)
    x["EBS"] = x.apply(calculate_ebs_independent, axis=1)
    
    # 기초 체력 및 타이밍 계산
    x["STRUCT_SCORE"] = x.apply(calculate_structural_score, axis=1).round(1)
    x["TIMING_SCORE"] = x.apply(calculate_timing_score, axis=1).round(1)
    
    if "ML_SCORE" not in x.columns: x["ML_SCORE"] = 0.0
    x["AI_SCORE"] = x["ML_SCORE"].clip(0, 100).round(1)
    
    # 매크로 가중치 적용 (순수 엔진 실력)
    if macro_risk == "CRITICAL": w_s, w_t, w_a = 0.6, 0.2, 0.2
    elif macro_risk == "HIGH": w_s, w_t, w_a = 0.5, 0.3, 0.2
    else: w_s, w_t, w_a = 0.4, 0.4, 0.2
        
    x["FINAL_SCORE"] = ((x["STRUCT_SCORE"] * w_s) + 
                        (x["TIMING_SCORE"] * w_t) + 
                        (x["AI_SCORE"] * w_a)).round(1)
    
    # 실전용 점수 초기화
    x["DISPLAY_SCORE"] = x["FINAL_SCORE"]
    
    return x
# ------------------------------- 텔레그램 (업그레이드) -------------------------------

def get_naver_theme_tags(code: str) -> str:
    """
    [v8.5] 네이버 금융에서 '동일업종(섹터)' 정보를 크롤링하여 해시태그로 반환
    """
    try:
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        # 타임아웃을 짧게 주어 전체 속도 저하 방지
        resp = requests.get(url, headers=headers, timeout=2)
        if resp.status_code != 200:
            return ""

        # 네이버 금융은 EUC-KR 인코딩 사용 가능성 있음
        text = resp.content.decode('euc-kr', 'replace')

        # 정규식으로 '동일업종비교' 링크 텍스트 추출 (BeautifulSoup 없이 가볍게 처리)
        # 패턴: <a href="/sise/sise_group_detail.naver?type=upjong...">업종명</a>
        match = re.search(r'sise_group_detail\.naver\?type=upjong[^>]*>([^<]+)</a>', text)
        if match:
            sector_name = match.group(1).strip()
            # 공백을 언더바(_)로 치환하여 해시태그화
            return f"#{sector_name.replace(' ', '_')}"

    except Exception:
        pass
    return ""

def send_telegram_auto(
    df: pd.DataFrame, 
    trade_ymd: str, 
    market_summary: str = "", 
    limit_count: int = 5  # 기본값 5
) -> None:
    log(f"📨 텔레그램 발송 시작 (Top {limit_count})...")

    if not TG_TOKEN or not TG_ID:
        log("⚠️ TG_TOKEN / TG_ID 미설정, 발송 생략")
        return

    # 숫자 포맷팅 헬퍼
    def _fmt_int(x):
        try:
            if pd.isna(x): return "N/A"                 
            return f"{int(float(x)):,}"
        except: return "N/A"            

    try:
        # 상위 종목 자르기
        top_picks = df.head(limit_count).reset_index(drop=True)
        trade_date = datetime.strptime(trade_ymd, "%Y%m%d").strftime('%Y-%m-%d')

        # [v8.5] 타이틀 업데이트
        msg = f"🔥 [LDY v8.5] 추천 Top {limit_count} ({trade_date})\n"

        if market_summary:
            msg += f"{market_summary}\n"
        msg += "-" * 30 + "\n\n"

        for i, row in top_picks.iterrows():
            rank = i + 1
            name = row['종목명']
            code = row['종목코드']
            route = row.get('ROUTE', '전략없음')
            buy = row.get('추천매수가', np.nan)
            score = row.get('LDY_SCORE', 0)
            comment = row.get('AI_COMMENT', '')
            news_reason = row.get('NEWS_REASON', '') # 추가됨

            # ✅ [v8.5 추가] 네이버 금융 실시간 테마/섹터 크롤링
            theme_tag = get_naver_theme_tags(code)

            # 기존 내부 분류(업종_대분류)가 있다면 함께 표기
            big_sector = row.get('업종_대분류', '')
            if big_sector and big_sector != '기타':
                # 중복 방지: 크롤링한 태그와 대분류가 다를 때만 둘 다 표시
                if theme_tag and (big_sector not in theme_tag):
                    theme_tag = f"#{big_sector} {theme_tag}"
                elif not theme_tag:
                    theme_tag = f"#{big_sector}"

            # 코멘트 길이 제한
            if len(comment) > 140:
                comment = comment[:140] + "…"

            # 메시지 포맷팅 (테마 태그 추가됨)
            msg += f"{rank}. {name} ({code}) {theme_tag}\n"
            msg += f"   🌡점수: {float(score):.1f}점\n"
            msg += f"   🎯전략: {route}\n"
            if news_reason and news_reason != "뉴스없음":
                msg += f"   📰재료: {news_reason}\n"
            msg += f"   💬AI: {comment}\n"

            # ✅ [v8.5 추가] 자금 관리(비중) 정보 표시
            qty = row.get('추천수량', 0)
            amt = row.get('추천금액(만원)', 0)
            if qty > 0:
                msg += f"   💰비중: {qty}주 (약 {amt}만원)\n"

            msg += f"   🔵매수: {_fmt_int(buy)}\n"
            msg += (
                f"   🔴손절: {_fmt_int(row.get('손절가'))} / "
                f"🟢목표: {_fmt_int(row.get('추천매도가1'))}\n\n"
            )

        # 메시지 길이 제한 (텔레그램 4096자 제한 대비)
        MAX_TG_LEN = 3800
        if len(msg) > MAX_TG_LEN:
            msg = msg[:MAX_TG_LEN] + "\n...\n(메시지 길이 제한으로 일부 생략)"

        resp = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_ID, "text": msg},
            timeout=15
        )
        resp.raise_for_status()
        log("🚀 텔레그램 전송 완료")

    except Exception as e:
        log(f"⚠️ 텔레그램 전송 실패: {e}")

# ------------------------------- 티커 분석 -------------------------------

# ---------------- util (analyze_ticker 위로 올리기) ----------------
def floor_to_tick_by(price: float, tick: int) -> int:
    if price is None or not np.isfinite(price) or tick <= 0:
        return 0
    return int(math.floor(float(price) / tick) * tick)

def ceil_to_tick_by(price: float, tick: int) -> int:
    if price is None or not np.isfinite(price) or tick <= 0:
        return 0
    return int(math.ceil(float(price) / tick) * tick)

def floor_to_tick(price: float) -> int:
    if price is None or not np.isfinite(price):
        return 0
    tk = tick_size(float(price))
    return floor_to_tick_by(float(price), tk)

def ceil_to_tick(price: float) -> int:
    if price is None or not np.isfinite(price):
        return 0
    tk = tick_size(float(price))
    return ceil_to_tick_by(float(price), tk)

# [v18.1] fetch_naver_news_headlines 중복 정의 제거 (1676줄에 원본 유지)
# 현재 메인 흐름은 AsyncNewsFetcher(async_crawler.py)를 사용함

# [v16.1 Patch] 투자자별 순매수 데이터 수집 함수 추가
def fetch_investor_net_buying(ymd: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    해당 일자의 외국인, 기관, 개인 순매수금액(원)을 가져옵니다.
    Returns: (외인_맵, 기관_맵, 개인_맵)
    """
    if (not PYKRX_OK) or (stock is None):
        return {}, {}, {}

    frg, inst, ant = {}, {}, {}
    
    try:
        log(f"💰 수급 데이터(투자자별 매매동향) 수집 중... ({ymd})")
        
        # 1. 외국인
        df_f = stock.get_market_net_purchases_of_equities_by_ticker(ymd, ymd, "ALL", "외국인")
        if df_f is not None:
            for code, row in df_f.iterrows():
                frg[str(code).zfill(6)] = int(row['순매수거래대금'])
        time.sleep(0.3) 

        # 2. 기관
        df_i = stock.get_market_net_purchases_of_equities_by_ticker(ymd, ymd, "ALL", "기관합계")
        if df_i is not None:
            for code, row in df_i.iterrows():
                inst[str(code).zfill(6)] = int(row['순매수거래대금'])
        time.sleep(0.3)

        # 3. 개인
        df_a = stock.get_market_net_purchases_of_equities_by_ticker(ymd, ymd, "ALL", "개인")
        if df_a is not None:
            for code, row in df_a.iterrows():
                ant[str(code).zfill(6)] = int(row['순매수거래대금'])
                
    except Exception as e:
        log(f"⚠️ 수급 데이터 수집 실패: {e}")
    
    return frg, inst, ant

# ------------------------------- Trigger Score Calculation -------------------------------

def calculate_trigger_score(df: pd.DataFrame) -> float:
    """
    [Updated v9.0] Safety Clamp 추가 + 최종 완성
    - Vol Score 계산 후 안전 범위(0~40) 강제 (이상치 방지)
    - Vol > 4.0 구간 점수 상향(20점) 및 선형 보간 유지
    """
    if df is None or df.empty or len(df) < 30:
        return 0.0

    def _calc_raw_trigger(idx):
        if idx < 25: return 0.0

        try:
            subset = df.iloc[:idx+1]
            
            # 1. 데이터 안전 추출
            close = pd.to_numeric(subset['종가'], errors='coerce').fillna(0)
            high = pd.to_numeric(subset['고가'], errors='coerce').fillna(0)
            low = pd.to_numeric(subset['저가'], errors='coerce').fillna(0)
            open_p = pd.to_numeric(subset['시가'], errors='coerce').fillna(0)
            vol = pd.to_numeric(subset['거래량'], errors='coerce').fillna(0)

            c_curr = float(close.iloc[-1])
            h_curr = float(high.iloc[-1])
            l_curr = float(low.iloc[-1])
            o_curr = float(open_p.iloc[-1])
            v_curr = float(vol.iloc[-1])
            c_prev = float(close.iloc[-2])

            if c_prev == 0: return 0.0

            # 2. 기초 지표 계산
            ret_pct = (c_curr / c_prev - 1) * 100
            candle_len = h_curr - l_curr
            
            # 윗꼬리 비율 & 종가 위치
            upper_wick = (h_curr - max(o_curr, c_curr))
            wick_ratio = upper_wick / candle_len if candle_len > 0 else 0.0
            range_pos = (c_curr - l_curr) / candle_len if candle_len > 0 else 0.5
            
            # 거래량 비율
            vol_ma20_series = vol.rolling(window=20).mean().shift(1)
            vol_ma20 = float(vol_ma20_series.iloc[-1])
            if pd.isna(vol_ma20) or vol_ma20 == 0: vol_ma20 = v_curr 
            
            vol_ratio = v_curr / vol_ma20

            # ---------------------------------------------------------
            # [Base Score] 기본 점수 (Max 90)
            # ---------------------------------------------------------
            
            # (1) 거래량 점수 (Linear Interpolation)
            if vol_ratio < 0.5:
                score_vol = 5.0
            elif 0.5 <= vol_ratio < 1.2:
                score_vol = 5.0 + (vol_ratio - 0.5) * 50.0
            elif 1.2 <= vol_ratio <= 3.0:
                score_vol = 40.0
            elif 3.0 < vol_ratio <= 4.0:
                score_vol = 40.0 - (vol_ratio - 3.0) * 20.0
            else:
                score_vol = 20.0 # 대장주 보호

            # [New] Safety Clamp: 혹시 모를 수치 튐 방지
            score_vol = max(0.0, min(40.0, score_vol))

            # (2) 돌파/추세 점수 (std20 0 방어)
            sma20 = close.rolling(20).mean().iloc[-1]
            std20 = close.rolling(20).std().iloc[-1]
            
            score_breakout = 0.0
            if not (pd.isna(sma20) or pd.isna(std20) or std20 == 0):
                bb_upper = sma20 + (2 * std20)
                if c_curr >= bb_upper: score_breakout = 40.0
                elif c_curr >= sma20: score_breakout = 20.0
            
            # (3) 모멘텀 가속
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            sig = macd.ewm(span=9, adjust=False).mean()
            hist = macd - sig
            
            score_mom = 0.0
            if hist.iloc[-1] > hist.iloc[-2]:
                score_mom = 10.0

            base = score_vol + score_breakout + score_mom 

            # ---------------------------------------------------------
            # [Penalty] 감점 로직 (Cap: 60)
            # ---------------------------------------------------------
            penalty = 0.0

            # 1. 분배봉
            if ret_pct >= 5.0:
                if wick_ratio >= 0.35: penalty += 25.0 
                elif wick_ratio >= 0.25: penalty += 15.0

            # 2. 약한 종가
            if ret_pct >= 3.0 and range_pos < 0.6:
                penalty += 15.0

            # 3. 거래량 폭발 설거지
            if vol_ratio >= 3.0:
                if c_curr < o_curr: penalty += 25.0 
                elif wick_ratio > 0.3 or range_pos < 0.5: penalty += 20.0

            # 4. 외인 이탈 결합
            if '외인순매수' in subset.columns:
                try:
                    frg_net = float(subset['외인순매수'].iloc[-1])
                    if ret_pct >= 5.0 and frg_net < 0 and vol_ratio > 1.5:
                        if range_pos < 0.6 or wick_ratio > 0.25:
                            penalty += 20.0
                except: pass

            penalty = min(penalty, 60.0)

            return max(0.0, base - penalty)
            
        except Exception:
            return 0.0

    last_idx = len(df) - 1
    score_today = _calc_raw_trigger(last_idx)
    score_yesterday = _calc_raw_trigger(last_idx - 1)

    final_score = (score_today * 0.7) + (score_yesterday * 0.3)
    return float(final_score)


def calc_volume_profile_v2(df_120: pd.DataFrame):
    """
    [v8.7 최종 정밀 버전] 매물대 분석 로직
    - Typical Price 사용 + 2~98% Percentile 컷
    - 데이터 길이에 따른 가변 Bins (15~20) 반영
    - 변동성(ATR%) 연동형 Near Resistance 범위 설정
    """
    if df_120.empty or len(df_120) < 15:
        return None, 0.0, 0.0, 0.0

    # 1. 기초 데이터 확보 (Typical Price)
    typ_price = (df_120['고가'] + df_120['저가'] + df_120['종가']) / 3
    curr_c = float(df_120['종가'].iloc[-1])
    
    # 2. 변동성(ATR%) 기반 근접 저항 범위 결정 (6% ~ 12% 가변)
    tr = pd.concat([(df_120['고가'] - df_120['저가']), 
                    (df_120['고가'] - df_120['종가'].shift(1)).abs(), 
                    (df_120['저가'] - df_120['종가'].shift(1)).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14).mean().iloc[-1] / curr_c) if curr_c > 0 else 0.03
    near_threshold = max(1.06, min(1.12, 1.0 + atr_pct * 2.0))

    # 3. 범위 설정 및 안전장치 (Percentile + Safety Clamp)
    l_min = typ_price.quantile(0.02)
    h_max = typ_price.quantile(0.98)
    l_min = min(l_min, curr_c * 0.97) # 현재가를 범위 안에 포함
    h_max = max(h_max, curr_c * 1.03)
    
    if h_max <= l_min: return None, 0.0, 0.0, 0.0

    # 4. 가변 Bins 설정 (해상도 조절)
    bins = 20 if len(df_120) >= 100 else 15
    bin_size = (h_max - l_min) / bins
    bins_edges = [l_min + i * bin_size for i in range(bins + 1)]
    
    # 5. Histogram 생성
    volume_hist, _ = np.histogram(typ_price, bins=bins_edges, weights=df_120['거래량'])
    
    # 6. POC 및 매물 비중 산출
    max_bin_idx = np.argmax(volume_hist)
    poc_p = (bins_edges[max_bin_idx] + bins_edges[max_bin_idx + 1]) / 2
    
    total_vol = volume_hist.sum() if volume_hist.sum() > 0 else 1
    res_all_vol = 0.0
    res_near_vol = 0.0
    
    for i in range(bins):
        bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
        if bin_center > curr_c:
            res_all_vol += volume_hist[i]
            if bin_center <= curr_c * near_threshold:
                res_near_vol += volume_hist[i]
                
    return poc_p, (res_all_vol / total_vol), (res_near_vol / total_vol), (near_threshold - 1.0) * 100

def analyze_ticker(
    t: str, ohlcv_df: pd.DataFrame, top_df: pd.DataFrame, mcap_map: Dict[str, float],
    kospi_set: set, kosdaq_set: set, name_map: Dict[str, str], sector_map: Dict[str, str],
    bench_map: Dict[str, Dict[int, float]],
) -> Optional[Dict[str, Any]]:
    code6 = str(t).zfill(6)
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 120: return None
    ohlcv = ohlcv_df.tail(LOOKBACK_DAYS).copy()
    
    # --- [데이터 타입 강제 변환: 여기서부터 교체] ---
    # 데이터프레임의 컬럼 자체를 숫자로 변환해야 매물대 함수가 작동합니다.
    for col in ["종가", "고가", "저가", "거래량", "시가"]:
        ohlcv[col] = pd.to_numeric(ohlcv[col], errors='coerce').fillna(0)
    
    # 변수 할당 (기존 로직 호환용)
    c = ohlcv["종가"]
    h = ohlcv["고가"]
    l = ohlcv["저가"]
    v = ohlcv["거래량"]
    o = ohlcv["시가"]
    
    last_c = float(c.iloc[-1])

    # --- [기본 필터링] 거래대금 및 시총 ---
    tv_row = top_df.loc[top_df["종목코드"] == code6, "거래대금(원)"]
    tv_eok = 0.0
    if not tv_row.empty:
        tv_eok = float(tv_row.values[0]) / 1e8

    if tv_eok <= 0:
        if "거래대금" in ohlcv.columns:
            try: tv_eok = float(pd.to_numeric(ohlcv["거래대금"].iloc[-1], errors='coerce')) / 1e8
            except: pass
        if tv_eok <= 0:
            try: tv_eok = (last_c * float(v.iloc[-1])) / 1e8
            except: pass

    mcap = get_mcap_eok_from_map(mcap_map, code6)
    if mcap_map and mcap > 0 and mcap < MIN_MCAP_EOK: return None
    if tv_eok < MIN_TURNOVER_EOK: return None

    # --- [지표 계산] ---
    ma20 = c.rolling(BB_PERIOD).mean()
    ma60 = c.rolling(60).mean()
    # 👇👇 [P2 최적화] 여기에 이 코드를 삽입하세요 👇👇
    # -----------------------------------------------------------
    # [Vectorization] 추후 연산을 위한 컬럼 미리 계산
    # -----------------------------------------------------------
    # 1. 갭 상승률 (시가 / 어제종가 - 1)
    ohlcv['gap_pct'] = (o / c.shift(1) - 1) * 100
    
    # 2. 캔들 모양 벡터 (윗꼬리 비율)
    ohlcv['candle_rng'] = h - l
    ohlcv['upper_shadow_ratio'] = (h - c) / ohlcv['candle_rng'].replace(0, 1)
    
    # 3. 거래량 급증 여부 (5일 평균 대비)
    ohlcv['vol_ma_5'] = v.rolling(window=5).mean()
    ohlcv['vol_ratio'] = v / ohlcv['vol_ma_5'].replace(0, 1)
    # -----------------------------------------------------------
    # 👆👆 여기까지 삽입 👆👆
    
    # 🔥 [v14.1] 1. 구조적 상태 변화 (Structural State)
    # (1) Low Trend: 저점 상승 여부 (Trend Check)
    min_l_prev = float(l.iloc[-20:-10].min())
    min_l_curr = float(l.iloc[-10:].min())
    # 양수면 저점 상승 중(에너지 축적), 음수면 저점 붕괴(무효화 대상)
    low_trend_pct = (min_l_curr - min_l_prev) / min_l_prev * 100 if min_l_prev > 0 else 0.0

    # (2) RSI Trend: RSI 저점 상승 여부 (Momentum Check)
    rsi_s = calc_rsi(c, 14)
    rsi = float(rsi_s.iloc[-1])
    rsi_min_prev = float(rsi_s.iloc[-20:-10].min())
    rsi_min_curr = float(rsi_s.iloc[-10:].min())
    # RSI 저점이 높아지고 있는가? (다이버전스 전조)
    rsi_rising = 1 if rsi_min_curr > rsi_min_prev else 0

    # (3) Volume Quality: 매집 강도
    is_red = c > o
    vol_red_avg = v[is_red].tail(20).mean()
    vol_blue_avg = v[~is_red].tail(20).mean()
    vol_quality = vol_red_avg / vol_blue_avg if vol_blue_avg > 0 else (1.5 if vol_red_avg > 0 else 1.0)

    # (4) BB Expansion: 변동성 확장 조짐 (BB Width가 바닥찍고 고개 드는지)
    std20 = c.rolling(BB_PERIOD).std()
    bb_upper = ma20 + (BB_STD * std20)
    bb_lower = ma20 - (BB_STD * std20)
    bb_bw = ((bb_upper - bb_lower) / ma20.replace(0, np.nan)) * 100
    bb_bw_val = float(bb_bw.iloc[-1])
    bb_bw_prev = float(bb_bw.iloc[-5]) # 5일 전 폭
    # 최근 폭이 과거보다 살짝 넓어지기 시작함 (확장 초입)
    bb_expanding = 1 if (bb_bw_val > bb_bw_prev * 1.05) and (bb_bw_val < 20) else 0

    # 박스권 위치 (Time Compression)
    period_high_20 = float(h.tail(20).max())
    period_low_20 = float(l.tail(20).min())
    denom = period_high_20 - period_low_20
    range_pos = (last_c - period_low_20) / denom if denom > 0 else 0.5

    # 기존 지표 (Squeeze 등)
    bw_squeeze = 1 if (np.isfinite(bb_bw_val) and bb_bw_val < BB_SQUEEZE_BW) else 0
    atr_kc_series = calc_atr(h, l, c, KC_ATR_PERIOD)
    kc_mid = ema(c, KC_PERIOD)
    kc_upper = kc_mid + (KC_MULT * atr_kc_series)
    kc_lower = kc_mid - (KC_MULT * atr_kc_series)
    ttm_series = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    ttm_squeeze = 1 if (bool(ttm_series.iloc[-1])) else 0
    sqz_cnt = int(ttm_series.iloc[-5:].sum()) # 최근 5일 중 스퀴즈 횟수

    # MACD / MFI
    mfi = float(calc_mfi(h, l, c, v, 14).iloc[-1])
    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9); hist = macd - sig
    hist_val = float(hist.iloc[-1])
    
    # Returns
    def _ret(d): return (last_c / float(c.iloc[-(d + 1)]) - 1.0) * 100 if len(c) >= d + 1 else np.nan
    ret_5 = _ret(5); ret_10 = _ret(10); ret_20 = _ret(20); ret_60 = _ret(60); ret_120 = _ret(120)

    # --- Trigger Logic (진입 엔진 분리) ---
    triggers = []
    
    # 1. 🚀 급등 시동 (Ignition)
    # 조건: 구조적 저점 상승 + RSI 저점 상승 + 상단 돌파 시도 + 매집
    if (low_trend_pct > 0.5) and (rsi_rising == 1) and (range_pos > 0.75) and (vol_quality > 1.1):
        if rsi < 70: triggers.append("🚀급등시동")
    
    # 2. ⚡ 눌림 회복 (Safe Pullback)
    # 조건: 20일선 지지 + 양봉 + 스퀴즈 이력
    disp = (last_c / float(ma20.iloc[-1]) - 1.0) * 100
    if (-2 <= disp <= 3) and (c.iloc[-1] > o.iloc[-1]) and (low_trend_pct >= 0):
        triggers.append("⚡눌림회복")

    # 3. 📦 박스 돌파 (Squeeze Break) - 변동성 확장 감지
    if (ttm_squeeze == 0) and (sqz_cnt >= 3) and (bb_expanding == 1):
         triggers.append("📦박스돌파")

    trigger_str = "/".join(triggers) if triggers else ""

    # VWAP & SuperTrend
    vwap_val = calc_vwap(ohlcv.tail(5))
    vwap_gap = (last_c - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0.0
    st_series, st_dir = calc_supertrend(h, l, c, 10, 3.0)
    st_val = float(st_series.iloc[-1]); st_trend = int(st_dir.iloc[-1])
    
    # V-Power
    tail5 = ohlcv.tail(5).copy()
    body = (tail5["종가"] - tail5["시가"]).abs()
    range_len = (tail5["고가"] - tail5["저가"]).replace(0, 1)
    sign = np.where(tail5["종가"] >= tail5["시가"], 1, -1)
    power_raw = (body / range_len) * tail5["거래량"] * sign
    avg_vol = tail5["거래량"].mean()
    v_power = power_raw.sum() / avg_vol if avg_vol > 0 else 0.0
    vol_z = float((v / v.rolling(20).mean().replace(0, np.nan)).iloc[-1]) if len(v) else 0.0

    # 캔들패턴
    candle_patterns = check_candle_pattern(o, h, l, c)

    # Swing Low Support Check
    swing_low_10 = float(l.tail(10).min())
    dist_to_swing = (last_c - swing_low_10) / last_c * 100
    is_swing_support = (dist_to_swing < 5.0) and (last_c > swing_low_10)

    # 🔥 [Fix A 준비: 1일 등락률 계산]
    prev_c = float(c.iloc[-2]) if len(c) > 1 else last_c
    ret_1d = (last_c / prev_c - 1.0) * 100

    # 👇👇 [여기부터 교체] 기존 '# 매매가 산정' 및 '# Fix B' 로직 제거 후 붙여넣기 👇👇
    # -------------------------------------------------------------
    # [Fix v9.0] 진입가 보수화 + 조건부 손절 바닥(Method A)
    # -------------------------------------------------------------
    buy = last_c
    
    # 이격도(disp)가 아직 정의 안됐을 경우를 대비해 안전 계산
    # (보통 위쪽 trigger logic에서 계산되지만, 안전하게 한 번 더 확인)
    if 'disp' not in locals():
        disp = (last_c / float(ma20.iloc[-1]) - 1.0) * 100

    # 1. 과열 이격도 (20일선 괴리율 15% 이상) -> 5일선/20일선 지지 대기
    if disp >= 15.0:
        buy = float(ma20.iloc[-1]) * 1.05 
        
    # 2. [핵심] 당일 급등주(7%↑) 추격 방지
    elif ret_1d >= 7.0:
        mid_body = (float(o.iloc[-1]) + float(c.iloc[-1])) / 2
        support_level = float(l.iloc[-1]) + (float(h.iloc[-1]) - float(l.iloc[-1])) * 0.3
        buy = max(mid_body, support_level)
    
    # 3. 적당한 상승
    elif ret_1d >= 3.0:
        buy = last_c * 0.985

    # 손절가/목표가 재계산
    atr_val = float(atr_kc_series.iloc[-1]) if len(atr_kc_series) else last_c * 0.03
    
    # 기본 손절: ATR 2.0배
    stop = buy - (2.0 * atr_val) 
    
    # [New] 급등주 2차 방어선 (Method A: ATR 비율 기반 선택)
    if ret_1d >= 7.0:
        today_low = float(l.iloc[-1])
        atr_pct = atr_val / today_low if today_low > 0 else 0.0

        # 변동성 크면(3.5% 이상) 숨 쉴 공간(0.8 ATR)을 주고,
        # 변동성 작으면 3% 룰을 적용해 칼손절 유도
        if atr_pct >= 0.035: 
            smart_floor = today_low - (0.8 * atr_val)
        else:
            smart_floor = today_low * 0.97

        # 기본 ATR 스탑이 바닥선보다 너무 깊게 내려가면(=손실이 너무 크면),
        # 스마트 바닥선으로 끌어올려 강제 청산
        if buy > smart_floor:
            stop = max(stop, smart_floor)

    # 구조적 지지선 보정 (Swing Low)
    if (dist_to_swing < 8.0) and (swing_low_10 > 0): 
        stop = max(stop, swing_low_10 * 0.97)
        
    target = buy + (buy - stop) * 2.5
    
    buy = round_to_tick(buy); stop = floor_to_tick(stop); target = ceil_to_tick(target)
    # 👆👆 [여기까지 교체] 👆👆

    # 메타 정보 (기존 유지)
    sector = sector_map.get(code6, "기타")
    name = name_map.get(code6, code6)
    m_row = top_df.loc[top_df["종목코드"] == code6, "시장"]
    market = str(m_row.values[0]) if not m_row.empty else ("KOSPI" if code6 in kospi_set else "KOSDAQ")
    bench_dict = bench_map.get(market, {})
    idx_20 = bench_dict.get(20, np.nan); idx_60 = bench_dict.get(60, np.nan); idx_120 = bench_dict.get(120, np.nan)
    rel_20 = ret_20 - idx_20 if np.isfinite(idx_20) and np.isfinite(ret_20) else np.nan
    rel_60 = ret_60 - idx_60 if np.isfinite(idx_60) and np.isfinite(ret_60) else np.nan
    rel_120 = ret_120 - idx_120 if np.isfinite(idx_120) and np.isfinite(ret_120) else np.nan
    
    # HMA
    hma20 = calc_hma(c, 20)
    curr_hma = float(hma20.iloc[-1]) if len(hma20) > 0 else 0
    prev_hma = float(hma20.iloc[-2]) if len(hma20) > 1 else 0
    hma_trend_up = curr_hma > prev_hma

    # 주봉 (간단 체크)
    is_above_w20 = False; is_w20_up = False
    try:
        w_res = ohlcv.resample('W').last()
        w_ma = w_res['종가'].rolling(20).mean()
        is_above_w20 = w_res['종가'].iloc[-1] > w_ma.iloc[-1]
        is_w20_up = w_ma.iloc[-1] > w_ma.iloc[-2]
    except: pass

    # MACD Slope
    slope = float(np.polyfit(np.arange(len(hist.tail(5))), hist.tail(5).values.astype(float), 1)[0]) if len(hist) >= 5 else 0.0
    slope_pct = (slope / last_c) * 100.0 if last_c > 0 else 0.0

    # [BugFix] 이전에 존재하던 두 번째 buy/stop/target 계산 블록 제거
    # 첫 번째 블록(Fix v9.0: 급등주 추격방지 + 스마트 손절 + R:R 2.5)만 사용

    # --- [이 코드가 빠져있으면 분석 결과에 반영되지 않습니다] ---
    # 매물대 분석 실행 (최근 120일 데이터 사용)
    poc_p, res_all, res_near, near_pct = calc_volume_profile_v2(ohlcv.tail(120))
    
    # POC 위치 버그 방지 및 이격도 계산
    is_above_poc = 1 if (poc_p is not None and last_c > poc_p) else 0
    poc_gap = round((last_c - poc_p) / poc_p * 100, 2) if poc_p else 0



    return {
        "시장": market, "종목명": name, "종목코드": code6, "업종": sector, "종가": int(last_c),
        "거래대금(억원)": round(tv_eok, 2), "시가총액(억원)": round(mcap, 1),
        "RSI14": round(rsi, 1), "MFI14": round(mfi, 1), "이격도": round(disp, 2),
        "BB_BW": round(bb_bw_val, 2), "TTM_SQUEEZE": int(ttm_squeeze), "TTM_SQUEEZE_CNT": sqz_cnt,
        "BB_SQUEEZE_BW": int(bw_squeeze),
        "ret_1d_%": round(ret_1d, 2),  # 👈 [New] 당일 등락률 추가
        "ret_5d_%": round(ret_5, 2), "ret_10d_%": round(ret_10, 2), "ret_20d_%": round(ret_20, 2),
        "ret_60d_%": round(ret_60, 2), "ret_120d_%": round(ret_120, 2),
        "rel_20d_%": round(rel_20, 2), "rel_60d_%": round(rel_60, 2), "rel_120d_%": round(rel_120, 2),

        "RES_RATIO": round(res_all, 3),        # 상단 전체 매물 비중
        "RES_RATIO_NEAR": round(res_near, 3),   # 상단 근접(8%) 매물 비중
        "IS_ABOVE_POC": is_above_poc,          # 매물대(POC) 돌파 여부
        "POC_GAP": poc_gap,                    # POC와의 이격도 (%)
        "NEAR_THRES": round(near_pct, 1),       # 가변 NEAR 범위 (%)
        
        
        # 🔥 [Step 1 & 2 핵심 지표]
        "Low_Trend_PCT": round(low_trend_pct, 2), # 무효화 체크용 (음수면 아웃)
        "RSI_Rising": int(rsi_rising),            # 상태 개선 체크용
        "BB_Expanding": int(bb_expanding),        # 상태 개선 체크용
        "Vol_Quality": round(vol_quality, 2),
        "Range_Pos": round(range_pos, 2),
        "추천매수가": buy, "손절가": stop, "추천매도가1": target, "추천매도가2": target * 1.1,
        "TRIGGER": trigger_str, # Entry Engine 분리
        "Above_MA20": 1 if disp > 0 else 0,
        "SUPERTREND_DIR": st_trend, "SUPERTREND_VAL": st_val,
        "VWAP": int(vwap_val), "VWAP_GAP": round(vwap_gap, 2),
        "MACD_Slope_PCT": round(slope_pct, 4),
        "거래강도": round(vol_z, 2), "V_POWER": round(v_power, 2),
        "IS_SWING_SUPPORT": is_swing_support,
        "주봉20선_상회": "O" if is_above_w20 else "X",
        "주봉추세": "▲" if is_w20_up else "▼",
        "HMA20": int(curr_hma), "HMA_Trend": "▲" if hma_trend_up else "▼", "HMA_On": "O" if last_c > curr_hma else "X",
        "OBV_Div": "X" 
    }

# 👇👇 [여기 사이에 함수를 통째로 붙여넣으세요] 👇👇

# [v11.0 신규 추가] 켈리 공식을 이용한 최적 비중 계산
def apply_kelly_betting(df: pd.DataFrame, total_capital: int = 10_000_000) -> pd.DataFrame:
    """
    Kelly Criterion을 적용하여 종목별 최적 매수 금액/수량을 다시 계산합니다.
    - 승률(p) 추정: TOTAL_SCORE(0~100)를 30%~70% 확률로 매핑
    - 손익비(b): (1차목표가 - 매수가) / (매수가 - 손절가)
    """
    log("💰 [Money Management] 켈리 공식(Kelly Criterion)으로 비중 최적화 중...")
    
    # 1. 안전 장치 (Half-Kelly 사용)
    KELLY_MULTIPLIER = 0.5   # 켈리 비율의 50%만 진입 (변동성 관리)
    MAX_ALLOCATION = 0.25    # 🔥 최대 비중 25%로 강화 (분산 투자 유도)
    def _calc_row(row):
        try:
            # 데이터 추출
            score = float(row.get("TOTAL_SCORE", 0))
            buy = float(row.get("추천매수가", 0))
            stop = float(row.get("손절가", 0))
            target = float(row.get("추천매도가1", 0))
            
            if buy <= 0 or stop <= 0 or target <= 0: return 0, 0

            # (1) 손익비(b) 계산 (Reward / Risk)
            risk = buy - stop
            reward = target - buy
            if risk <= 0: return 0, 0
            
            b = reward / risk 
            
            # (2) 승률(p) 추정 (60점=40%, 80점=60%, 100점=80%)
            p = 0.4 + (max(score, 0) - 60) * 0.01
            p = min(max(p, 0.3), 0.85)

            # 점수가 너무 낮으면(60점 미만) 켈리 적용 위험 -> 0
            if score < 60:
                return 0, 0

            # (3) 켈리 공식: f = p - (1-p)/b
            q = 1 - p
            f = p - (q / b)
            
            # (4) Half-Kelly 적용 및 최대 비중 제한
            f_safe = f * KELLY_MULTIPLIER
            f_final = min(max(f_safe, 0.0), MAX_ALLOCATION)
            
            # (5) 최종 금액 및 수량 산출
            final_amt = int(total_capital * f_final)
            final_qty = int(final_amt / buy)
            
            return final_qty, final_amt

        except Exception:
            return 0, 0

    # DataFrame에 적용
    res = df.apply(_calc_row, axis=1, result_type='expand')
    df["켈리_수량"] = res[0]
    df["켈리_금액(원)"] = res[1]
    
    # 켈리 결과가 유의미하면 기존 추천 수량을 덮어씀
    mask = df["켈리_수량"] > 0
    df.loc[mask, "추천수량"] = df.loc[mask, "켈리_수량"]
    df.loc[mask, "추천금액(만원)"] = (df.loc[mask, "켈리_금액(원)"] / 10000).round(1)
    
    return df

# ------------------------------- 메인 실행 -------------------------------

def main(
    trade_date: Optional[str] = None,
    top_n: Optional[int] = None,
    enable_telegram: bool = True,
    tag: Optional[str] = None,
) -> None:
    log("🚀 LDY Collector v10.0 (AI Powered) 시작...")

    # ----------------------------------------------------------------------
    # 🔥 [v15.6] 스마트 훈련 스킵 (당일 이미 학습했다면 30분 절약)
    # ----------------------------------------------------------------------
    if ml_engine.is_trained_today():
        log("✅ [SKIP] 오늘 이미 v15.6 Master 모델 학습이 완료되었습니다.")
    else:
        log("🤖 AI 모델 최적화(v15.6 Master) 진행 중... (약 30분 소요)")
        try:
            ml_engine.train_model() 
        except Exception as e:
            log(f"⚠️ 모델 학습 실패: {e}")
    
    # ✅ 여기에 넣기 (resolve_trade_date() 호출 전에!)
    if not PYKRX_OK:
        log("❌ pykrx 미사용 환경에서는 거래대금 TopN/스냅샷 생성이 불가합니다.")
        return

    # 1) 먼저 거래 기준일 결정
    trade_ymd = resolve_trade_date(trade_date)

    # 🔥 [v8.0 추가] 매크로 필터 적용 ---------------------------------
    macro_risk, macro_msg, new_ebs, rec_limit_cnt = check_macro_env(trade_ymd)

    # 전역 변수 PASS_EBS를 동적으로 수정 (주의: ThreadPoolExecutor 사용 시에도 global 변경은 반영됨)
    global PASS_EBS
    PASS_EBS = new_ebs
    log(f"⚙️ 매크로 필터 적용: PASS_EBS={PASS_EBS}, Telegram_Limit={rec_limit_cnt}")


    # 2) 그 날짜를 기준으로 시총 맵 생성 시도
    mcap_map, mcap_ymd = build_mcap_map(trade_ymd)

    log(f"📅 거래 기준일: {trade_ymd} (mcap ref: {mcap_ymd})")

    # [변경] 3) 20/60/120일 벤치마크 수익률 맵 생성
    bench_map = get_benchmark_returns(trade_ymd)

    def _fmt(v): return f"{v:.2f}%" if isinstance(v, (int, float)) else "N/A"

    k_60 = bench_map.get("KOSPI", {}).get(60)
    q_60 = bench_map.get("KOSDAQ", {}).get(60)
    log(f"📈 벤치마크(60d): KOSPI {_fmt(k_60)}, KOSDAQ {_fmt(q_60)}")


    use_top_n = top_n or TOP_N
    log(f"📊 거래대금 상위 N: {use_top_n}")

    # 4) 거래대금 상위 종목 리스트
    top_df = pick_top_by_trading_value(trade_ymd, use_top_n)

    # 5) 시총 프리필터
    if mcap_map:
        top_df["시가총액(억원)"] = top_df["종목코드"].map(
            lambda c: get_mcap_eok_from_map(mcap_map, c)
        )

        before_cnt = len(top_df)
        s_mcap = pd.to_numeric(top_df["시가총액(억원)"], errors="coerce").fillna(0)

        # 1차 필터
        top_df_f = top_df[s_mcap >= MIN_MCAP_EOK].copy()
        after_cnt = len(top_df_f)
        log(f"📊 시총 필터 적용: {before_cnt} → {after_cnt}개 (기준 {MIN_MCAP_EOK}억)")

        # 💥 모두 0개면 → 기준 완화해서 한 번 더 시도
        if after_cnt == 0 and before_cnt > 0:
            relaxed = MIN_MCAP_EOK / 10
            log(f"⚠️ 시총 필터 결과 0개 → 임시 기준 완화 ({relaxed}억)")
            top_df_f = top_df[s_mcap >= relaxed].copy()

        top_df = top_df_f
    else:
        log("⚠️ mcap_map 비어 있음 → 시총 사전 필터 생략")
        top_df["시가총액(억원)"] = 0.0

    # 6) 여기부터는 공통 분석 파이프라인
    tickers = top_df["종목코드"].tolist()

    kospi_set, kosdaq_set = get_market_sets(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)
    # 🔹 전체 종목 가격 스냅샷 저장
    save_price_snapshot(trade_ymd, name_map)
    sector_map = build_sector_map()

    # 날짜 계산
    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(
        days=LOOKBACK_DAYS * 2 + 60
    )
    start_s, end_s = start_dt.strftime("%Y%m%d"), trade_ymd

    # 🔥 [v7.0 핵심] 데이터 일괄 수집 및 캐싱 (여기가 바뀌었습니다!)
    # 분석 전에 모든 데이터를 미리 준비합니다.
    full_ohlcv_map = prepare_ohlcv_data(tickers, start_s, end_s, trade_ymd)

    rows: List[Dict[str, Any]] = []
    err_cnt = 0

    # 분석 시작 (데이터는 이미 full_ohlcv_map에 있음)
    if MAX_WORKERS <= 1:
        # 단일 스레드 처리
        for t in tqdm(tickers, desc="Analyzing"):
            code6 = str(t).zfill(6)
            try:
                # 맵에서 데이터 꺼내기
                df_t = full_ohlcv_map.get(code6)

                # 데이터 넘겨주기 (start_s, end_s 아님!)
                row = analyze_ticker(
                    t, df_t, 
                    top_df, mcap_map,
                    kospi_set, kosdaq_set, name_map, sector_map,
                    bench_map     # ✅ 수정: bench_map으로 변경
                )
                if row is not None:
                    rows.append(row)
            except Exception as e:
                err_cnt += 1
                # log(f"⚠️ {code6} 처리 중 오류 발생: {e}")
                continue
    else:
        # 🔥 [수정 대상] 병렬 처리 (CPU 연산 분산)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = []
            for t in tickers:
                code6 = str(t).zfill(6)
                df_t = full_ohlcv_map.get(code6)

                # 데이터가 없으면 스킵
                if df_t is None or df_t.empty:
                    continue

                futs.append(ex.submit(
                    analyze_ticker,
                    t, df_t, # DataFrame 전달
                    top_df, mcap_map,
                    kospi_set, kosdaq_set, name_map, sector_map,
                    bench_map # ✅ [수정 완료] bench_ret_60 -> bench_map 으로 변경
                ))

            for fut in tqdm(as_completed(futs), total=len(futs), desc="Analyzing"):
                try:
                    row = fut.result()
                    if row is not None:
                        rows.append(row)
                except Exception as e:
                    err_cnt += 1
                    # log(f"⚠️ 병렬 처리 중 오류: {e}")

    if err_cnt > 0:
        log(f"⚠️ 분석 중 오류 발생/데이터 부족 종목 수: {err_cnt}건")

    if not rows:
        log("⚠️ 필터를 통과한 종목이 없습니다. (시장 상황 악화 또는 데이터 부족)")
        # 빈 데이터프레임을 만들어서 텔레그램이라도 보내거나, 그냥 정상 종료
        if enable_telegram:
             send_telegram_auto(pd.DataFrame(), trade_ymd, market_summary="⚠️ 필터 통과 종목 없음 (휴장일/데이터부족)", limit_count=0)
        return

    df_raw = pd.DataFrame(rows)

    if not df_raw.empty:
        # -----------------------------------------------------------
        # 1. [데이터 무결성] 수급 데이터 매핑 및 단위 보정 (TOXIC 오탐 방지)
        # -----------------------------------------------------------
        map_frg, map_inst, map_ant = fetch_investor_net_buying(trade_ymd)
        df_raw["외인순매수"] = df_raw["종목코드"].map(map_frg).fillna(0)
        df_raw["기관순매수"] = df_raw["종목코드"].map(map_inst).fillna(0)
        df_raw["개인순매수"] = df_raw["종목코드"].map(map_ant).fillna(0)
        df_raw["메이저순매수"] = df_raw["외인순매수"] + df_raw["기관순매수"]

        # 🚨 [100점 포인트] '원' 단위 컬럼 강제 생성
        # determine_state_dynamic의 분모를 채워 EXIT_WARNING 오작동을 원천 차단합니다.
        if "거래대금(원)" not in df_raw.columns:
            tv_eok = pd.to_numeric(df_raw.get("거래대금(억원)", 0), errors="coerce").fillna(0.0)
            df_raw["거래대금(원)"] = (tv_eok * 100_000_000).astype(float)

        # -----------------------------------------------------------
        # 2. [전략 보강] 업종 분류 및 시장 온도 측정
        # -----------------------------------------------------------
        if "업종" in df_raw.columns:
            df_raw["업종_상세"] = df_raw["업종"]
            df_raw["업종_대분류"] = df_raw.apply(lambda r: classify_big_sector(str(r.get("종목명", "")), str(r.get("업종", ""))), axis=1)

        df_raw, _ = add_sector_momentum(df_raw, "업종_대분류")
        breadth = compute_market_breadth(df_raw)
        mkt_temp = label_market_temp(breadth.get("ALL", np.nan))
        log(f"🌡 시장 온도: {mkt_temp} (Breadth: {breadth.get('ALL', 0)}%) -> 동적 가중치 적용")

        # -----------------------------------------------------------
        # 3. [통합 엔진] AI 엔진 동기화 및 3원화 스코어링 빌드
        # -----------------------------------------------------------
        log("🧠 AI 엔진 동기화 및 통합 스코어링 시작...")
        
        # AI 점수 주입
        df_out = ml_engine.apply_ml_score(df_raw, full_ohlcv_map)
        df_out["ML_SCORE"] = pd.to_numeric(df_out.get("ML_SCORE", 0.0), errors='coerce').fillna(0.0).clip(0, 100)

        # 실시간 차트 신호(Raw Trigger) 산출
        trigger_list = []
        for idx, row in df_out.iterrows():
            code = str(row['종목코드']).zfill(6)
            ohlcv_df = full_ohlcv_map.get(code)
            ts = calculate_trigger_score(ohlcv_df) if ohlcv_df is not None and not ohlcv_df.empty else 0.0
            trigger_list.append(ts)
        
        df_out['TRIGGER_SCORE'] = trigger_list
        df_out['RAW_TRIGGER_SCORE'] = df_out['TRIGGER_SCORE'] 

        # 최종 통합 스코어링 실행 (v15.1 엔진: EBS/STRUCT/TIMING/FINAL 일괄 산출)
        df_out = build_global_score(df_out, macro_risk)

        # -----------------------------------------------------------
        # 4. [전술 상태] 동적 임계값 기반 ROUTE 결정
        # -----------------------------------------------------------
        thresholds = {
            "range_q75": safe_quantile(df_out.get("Range_Pos"), 0.75, 0.8),
            "vol_q75": safe_quantile(df_out.get("Vol_Quality"), 0.75, 1.2)
        }
        df_out["ROUTE"] = df_out.apply(lambda r: determine_state_dynamic(r, thresholds), axis=1)
        df_out["상태"] = df_out["ROUTE"]

        # -----------------------------------------------------------
        # 5. [정예군 편성] Elite 120 대형 박제 (Placement Logic)
        # -----------------------------------------------------------
        ebs_val = pd.to_numeric(df_out.get("EBS", 0), errors='coerce').fillna(0)
        struct_val = pd.to_numeric(df_out.get("STRUCT_SCORE", 0), errors='coerce').fillna(0)
        
        # [BugFix] OVERHEAT/EXIT_WARNING 종목은 정예군에서 제외
        mask_safe = ~df_out["ROUTE"].isin(["OVERHEAT", "EXIT_WARNING"])
        mask_qual = (ebs_val >= 6) & (struct_val >= 60) & mask_safe
        
        df_prime = df_out[mask_qual].copy().sort_values(["TIMING_SCORE", "AI_SCORE"], ascending=False)
        df_normal = df_out[~mask_qual].copy().sort_values("FINAL_SCORE", ascending=False)
        
        # 정예군 상위 120개를 최상단에 고정 배치 (전술 대형 완성)
        df_out = pd.concat([df_prime.head(120), df_normal, df_prime.iloc[120:]], ignore_index=True)
        
        # -----------------------------------------------------------
        # 6. [최종 확정] 랭크 부여 및 날짜 메타데이터
        # -----------------------------------------------------------
        df_out["LDY_RANK"] = np.arange(1, len(df_out) + 1)
        df_out["기준일"] = trade_ymd
        df_out["시총기준일"] = mcap_ymd

    # -------------------------------------------------------------
    # 🔥 [v11.0 수정] 비동기 뉴스 수집 & LLM 분석 & DB 저장 통합
    # -------------------------------------------------------------
    if LLM_AVAILABLE:
        log("🧠 상위 10개 종목 심층 분석 중 (뉴스 + DART 공시)...")
        
        dart_key = os.environ.get("DART_API_KEY")
        dart_engine = dart_analyzer.DartAnalyzer(dart_api_key=dart_key)
        if not dart_key: log("⚠️ DART_API_KEY 미설정. 공시 분석 스킵.")

        target_indices = df_out.index[:10]
        target_codes = [str(df_out.loc[i, "종목코드"]).zfill(6) for i in target_indices]
        
        news_map = {}
        try:
            fetcher = AsyncNewsFetcher(max_concurrent=5)
            if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            news_map = asyncio.run(fetcher.fetch_all(target_codes))
        except Exception as e:
            log(f"⚠️ 뉴스 수집 중 오류: {e}")

        df_out["NEWS_SCORE"] = 0.0
        df_out["NEWS_REASON"] = "특이사항 없음"
        
        for idx in target_indices:
            code, name = str(df_out.loc[idx, "종목코드"]).zfill(6), df_out.loc[idx, "종목명"]
            
            # (A) 뉴스 감성 분석
            headlines = news_map.get(code, [])
            l_score, l_reason = analyze_sentiment_llm(name, headlines) if headlines else (0.0, "")
            
            # (B) DART 공시 분석
            d_score, d_reason = 0.0, ""
            if dart_engine.dart:
                disclosures = dart_engine.get_major_disclosures(code, days=3)
                if disclosures:
                    recent = disclosures[0]
                    d_score, d_reason = dart_engine.analyze_report(recent['rcept_no'], recent['report_nm'])
                    log(f"   📄 {name} DART 분석: {recent['report_nm']} -> {d_score}점")

            # (C) 점수 통합 (순수 FINAL은 보존, DISPLAY만 업데이트)
            event_val = np.clip(l_score + d_score, -10, 10)
            df_out.at[idx, "NEWS_SCORE"] = event_val
            
            # 🚨 핵심: 실전용 점수 산출
            old_final = df_out.at[idx, "FINAL_SCORE"]
            df_out.at[idx, "DISPLAY_SCORE"] = np.clip(old_final + event_val, 0, 100)
            
            # 이유 및 코멘트 업데이트
            reasons = [f"[공시]{d_reason}" for d_reason in [d_reason] if d_reason] + \
                      [f"[뉴스]{l_reason}" for l_reason in [l_reason] if l_reason and l_reason != "뉴스없음"]
            final_reason = " / ".join(reasons) if reasons else "특이사항 없음"
            df_out.at[idx, "NEWS_REASON"] = final_reason
            
            if reasons:
                cur_comment = str(df_out.at[idx, "AI_COMMENT"])
                df_out.at[idx, "AI_COMMENT"] = (cur_comment if cur_comment != "nan" else "") + f" 📢재료: {final_reason}"
    else:
        log("ℹ️ LLM 설정(API Key)이 없어 심층 분석을 건너뜁니다.")
        df_out["NEWS_SCORE"] = 0.0
        df_out["NEWS_REASON"] = "N/A"

    # -----------------------------------------------------------
    # [Step 7] UI 호환성 동기화 및 상태 플래그 확정
    # -----------------------------------------------------------
    # 1. 모든 가시성 점수를 DISPLAY_SCORE로 수렴
    df_out["LDY_SCORE"]   = df_out["DISPLAY_SCORE"]
    df_out["TOTAL_SCORE"] = df_out["DISPLAY_SCORE"]
    df_out["RANK_SCORE"]  = df_out["DISPLAY_SCORE"]

    # 2. 벤치마크 데이터 매핑
    df_out["벤치_60d_KOSPI_%"] = bench_map.get("KOSPI", {}).get(60, np.nan)
    df_out["벤치_60d_KOSDAQ_%"] = bench_map.get("KOSDAQ", {}).get(60, np.nan)

    # 3. 전략 활성 상태 플래그 (Code Matching 방식)
    df_out["IS_ACTIVE"]    = df_out["ROUTE"].isin(["ATTACK", "ARMED"])
    df_out["IS_NOW_ENTRY"] = df_out["ROUTE"] == "ATTACK"
    df_out["IS_WATCH"]     = df_out["ROUTE"] == "WAIT"

    # -----------------------------------------------------------
    # [Step 8] 데이터 저장 및 정합성 체크
    # -----------------------------------------------------------
    must_cols = [
        "종목코드", "종목명", "시장", "업종_대분류", "종가", "거래대금(억원)", "시가총액(억원)",
        "상태", "ROUTE", "IS_ACTIVE", "IS_NOW_ENTRY", "IS_WATCH",
        "DISPLAY_SCORE", "FINAL_SCORE", "STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE", "NEWS_SCORE",
        "추천매수가", "손절가", "추천매도가1", "추천매도가2", "TRIGGER", "V_POWER", "거래강도", 
        "VWAP", "POC_GAP", "NEWS_REASON", "TTM_SQUEEZE_CNT", "Low_Trend_PCT", "RSI14", "이격도"
    ]
    # 누락 컬럼 방어
    for c in must_cols:
        if c not in df_out.columns: df_out[c] = np.nan

    df_out = df_out[must_cols + [c for c in df_out.columns if c not in must_cols]]
    
    ensure_dir(OUT_DIR)
    out_path_dated = os.path.join(OUT_DIR, f"recommend_{trade_ymd}{f'_{tag}' if tag else ''}.csv")
    out_path_latest = os.path.join(OUT_DIR, "recommend_latest.csv")

    df_out.to_csv(out_path_dated, index=False, encoding=UTF8)
    df_out.to_csv(out_path_latest, index=False, encoding=UTF8)
    log(f"💾 저장 완료 ({len(df_out)}건) → {out_path_dated}")

    # DuckDB 저장
    try:
        db = LDYDBManager()
        db.save_recommendations(df_out, trade_ymd)
        db.close()
    except Exception as e: log(f"⚠️ DB 저장 실패: {e}")

    run_reality_check(OUT_DIR, trade_ymd)
    make_rank_validation_report(OUT_DIR, asof_ymd=trade_ymd, 
                                methods=["DISPLAY_SCORE", "FINAL_SCORE", "AI_SCORE"])

    # -----------------------------------------------------------
    # [Step 9] 텔레그램 리포트 발송
    # -----------------------------------------------------------
    if enable_telegram:
        summary_text = f"🌡 {mkt_temp} (Breadth: {breadth.get('ALL', 0)}%)"
        if macro_msg: summary_text += f"\n{macro_msg}"
        
        # 주도 섹터 정보 추가
        if "SECTOR_RANK" in df_out.columns:
            top_sectors = df_out.sort_values("SECTOR_RS", ascending=False)["업종_대분류"].unique()[:2]
            summary_text += f"\n🚀 주도: {' '.join(top_sectors)}"

        send_telegram_auto(df_out, trade_ymd, market_summary=summary_text, limit_count=rec_limit_cnt)
    else:
        log("✉️ 텔레그램 발송 생략 (옵션 설정)")
