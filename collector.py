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

from time_utils import now_kst, now_utc, KST
from concurrent.futures import ThreadPoolExecutor, as_completed

# [v9.0 추가] LLM 및 뉴스 크롤링용 라이브러리
try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False
    print("⚠️ BeautifulSoup4가 설치되지 않았습니다. (pip install beautifulsoup4)")

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
        })).reset_index(drop=True)

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

def find_latest_valid_date(check_fn, max_back_days: int = 10) -> str:
    now = now_kst()
    d = now.date()
    if now.hour < 18:
        d -= timedelta(days=1)

    last_checked = d.strftime("%Y%m%d")
    for _ in range(max_back_days):
        ymd = d.strftime("%Y%m%d")
        last_checked = ymd
        if check_fn(ymd):
            return ymd
        d -= timedelta(days=1)

    return last_checked

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
    frames: List[pd.DataFrame] = []

    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = safe_ohlcv_by_ticker(date_yyyymmdd, market=m)
            if df is None or df.empty:
                log(f"⚠️ {m} 거래대금 데이터 비어있음: {date_yyyymmdd}")
                continue

            df = df.reset_index()

            # 코드 컬럼명 통일
            code_col = None
            for c in df.columns:
                if ("티커" in str(c)) or ("코드" in str(c)) or (str(c) == "종목코드"):
                    code_col = c
                    break
            if code_col is None:
                log(f"⚠️ {m} 코드 컬럼을 찾을 수 없음: {df.columns.tolist()}")
                continue

            df = df.rename(columns={code_col: "종목코드"})

            # 거래대금 컬럼명 통일
            if "거래대금" in df.columns and "거래대금(원)" not in df.columns:
                df = df.rename(columns={"거래대금": "거래대금(원)"})

            if "거래대금(원)" not in df.columns:
                log(f"⚠️ {m} 거래대금 컬럼이 없음: {df.columns.tolist()}")
                continue

            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
            df["시장"] = m
            df["거래대금(원)"] = pd.to_numeric(df["거래대금(원)"], errors="coerce")

            # ✅ 여기서 frames에 넣어야 함 (이게 누락돼서 지금 에러 난 것)
            frames.append(df[["종목코드", "시장", "거래대금(원)"]])

        except Exception as e:
            log(f"⚠️ {m} 거래대금 수집 실패({date_yyyymmdd}): {e}")
            continue

    if not frames:
        raise RuntimeError("No Data from KRX (거래대금)")

    df_all = pd.concat(frames, ignore_index=True).dropna(subset=["거래대금(원)"])
    df_all["종목코드"] = df_all["종목코드"].astype(str).str.zfill(6)

    return df_all.sort_values("거래대금(원)", ascending=False).head(top_n)

def get_market_sets(d: str) -> Tuple[set, set]:
    try:
        kospi = set(safe_ticker_list(d, market="KOSPI"))
        kosdaq = set(safe_ticker_list(d, market="KOSDAQ"))
        return kospi, kosdaq
    except Exception:
        return set(), set()

def get_name_map_cached(d: str) -> Dict[str, str]:
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, f"krx_codes_{d}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str)
            return dict(zip(df["종목코드"], df["종목명"]))
        except Exception:
            pass

    rows: List[Dict[str, str]] = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            tickers = safe_ticker_list(d, market=m)
            for t in tickers:
                nm = safe_ticker_name(t) or str(t)
                rows.append({
                    "종목코드": str(t).zfill(6),
                    "종목명": nm
                })
                time.sleep(0.001)
        except Exception:
            pass

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding=UTF8)
        return dict(zip(df["종목코드"], df["종목명"]))
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

    return comment if comment else "특이사항 없음."

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

def route_tag(row: pd.Series) -> str:
    """
    v6.7 ROUTE 분류
    - BRK: 강한 돌파
    - SQZ: 진성 스퀴즈(폭발대기)
    - Watch: 상승 준비 / 관찰
    - REV: 역추세 반등 (지수·섹터 대비 바닥권에서 턴)
    - PULL: 눌림/중립
    """
    # ✅ [수정됨] 0.0 값을 안전하게 처리
    def _fv(key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
            val = row.get(key)
            if val is None or pd.isna(val):
                return default
            return float(val)
        except Exception:
            return default

    r5 = _fv("ret_5d_%", 0.0)
    r10 = _fv("ret_10d_%", 0.0)
    slope = _fv("MACD_Slope_PCT", 0.0)
    if slope == 0.0:
        slope = _fv("MACD_Slope", 0.0)
    ebs = _fv("EBS", 0.0)
    now_pct = _fv("Now%", 999.0)
    now_pct = _fv("Now%", 999.0) # 이제 0.0이 들어와도 999가 되지 않음!
    rr1 = _fv("RR1", 0.0)
    mfi = _fv("MFI14", 50.0)
    rel60 = _fv("rel_60d_%", 0.0)  # 60일 상대강도(α)
    rel60 = _fv("rel_60d_%", 0.0)
    bb_sq = _fv("TTM_SQUEEZE", _fv("BB_SQUEEZE", 0.0))

    # 1) 강한 돌파 BRK
    strong_break = (
        (r5 >= 3) and (r10 >= 5) and (slope > 0) and (ebs >= PASS_EBS)
        and (now_pct <= 10) and (mfi >= 55)
    )
    # RR1이 너무 나쁘면 BRK에서 제외
    if strong_break and np.isfinite(rr1) and rr1 < 0.6:
        strong_break = False

    if strong_break:
        return "🔼 BRK (돌파)"

    # 2) 역추세 반등 REV
    #    - 60일 상대강도는 약하지만, 단기 r5>0 + slope>0 + 과도한 갭 아님
    rev = (
        (rel60 <= -5.0) and   # 지수 대비 꽤 처졌던 종목
        (r5 >= 1.0) and       # 최근 5일은 플러스
        (slope > 0) and       # 상승 기울기
        (now_pct <= 10)       # 진입가에서 너무 멀지 않음
    )
    if rev:
        return "🔻 REV (역추세 반등)"

    # 2.5) 진성 스퀴즈(TTM) + 모멘텀 플러스 = 🔥 SQZ(폭발대기)
    # 2.5) 진성 스퀴즈(TTM)
    if (bb_sq >= 1) and (slope > 0) and (now_pct <= 10):
        return "🔥 SQZ (폭발대기)"

    # 3) Watch 영역
    watch = ((slope > 0) and (r5 > 0)) or ((ebs >= PASS_EBS) and (now_pct <= 8))
    if watch:
        if r5 >= 1.5 and slope > 0:
            return "🔺 Watch (관찰·돌파예상)"
        return "🔺 Watch (상승 준비)"

    # 4) 그 외는 기본적으로 PULL
    if r5 <= -2 and slope > 0:
        return "🔁 MR (반전)"

    return "↩️ PULL (눌림)"

# [수정 대상] build_global_score 함수 내부의 MOM 계산 부분을 아래로 교체하세요.
# (함수 전체를 다 바꿀 필요는 없고, mom_norm 계산 부분만 찾아서 바꿔도 되지만, 
#  안전을 위해 build_global_score 함수 전체 코드를 드립니다)

def build_global_score(lat: pd.DataFrame, market_temp: str = "🌤 중립") -> pd.DataFrame:
    x = lat.copy()
    # 가중치 설정 (기존과 동일)
    w = {"RR": 0.25, "T1": 0.18, "SL": 0.12, "NEAR": 0.12, "MOM": 0.10, "LIQ": 0.13, "TEC": 0.10}
    if "과열" in market_temp: w = {"RR": 0.15, "T1": 0.20, "SL": 0.10, "NEAR": 0.05, "MOM": 0.25, "LIQ": 0.15, "TEC": 0.10}
    elif "침체" in market_temp: w = {"RR": 0.30, "T1": 0.10, "SL": 0.20, "NEAR": 0.25, "MOM": 0.05, "LIQ": 0.05, "TEC": 0.05}

    def col_or_zero(df, col): return nz_num(df[col]) if col in df.columns else pd.Series(0.0, index=df.index)

    # 데이터 로드
    close = nz_num(x["종가"]); entry = nz_num(x["추천매수가"]).replace(0, np.nan)
    stop = nz_num(x["손절가"]); t1 = nz_num(x["추천매도가1"]); turn = nz_num(x["거래대금(억원)"])
    rsi = nz_num(x["RSI14"]).fillna(50)
    slope = nz_num(x["MACD_Slope_PCT"]) if "MACD_Slope_PCT" in x.columns else nz_num(x["MACD_Slope"])
    slope_floor = 0.01 if "MACD_Slope_PCT" in x.columns else 1.0
    volz = nz_num(x["거래강도"]).fillna(0); kairi = nz_num(x["이격도"])
    r5 = nz_num(x["ret_5d_%"]); r10 = nz_num(x["ret_10d_%"]); ebs = nz_num(x["EBS"]).fillna(0)
    rel20 = col_or_zero(x, "rel_20d_%"); rel60 = col_or_zero(x, "rel_60d_%"); rel120 = col_or_zero(x, "rel_120d_%")

    # ✅ [v7.5] V-Power 로드
    v_power = col_or_zero(x, "V_POWER")

    # 🔥 [v8.0] SuperTrend 데이터 로드
    st_dir = col_or_zero(x, "SUPERTREND_DIR")

    # 팩터 계산
    rr_den = (close - stop).where((close-stop)>0, np.nan)
    rr1 = (t1 - close) / rr_den
    now_gap = ((close - entry).abs() / entry * 100).fillna(0)
    t1_room = ((t1 - close) / close * 100)
    sl_pct = ((entry - stop) / entry * 100)

    rr_norm = pct_norm_pos(rr1, q=90, floor=1.0).fillna(0)
    t1_norm = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1).fillna(0)
    opt = 7.0; width = 3.0
    sl_norm = pd.Series(np.exp(-((sl_pct - opt) / width) ** 2), index=x.index).fillna(0)
    near_norm = inv_dist_norm(now_gap, cap=cap_q(now_gap, q=75, floor=1.0)).fillna(0)

    ers_bits = ((ebs >= PASS_EBS).astype(int) + (slope > 0).astype(int) + ((rsi >= RSI_LOW) & (rsi <= RSI_HIGH)).astype(int))
    ers_norm = np.clip(ers_bits / 3.0, 0, 1).fillna(0)
    slope_pos_norm = pct_norm_pos(slope, q=90, floor=slope_floor).fillna(0)
    mom_mid_norm = pct_norm_pos(r10.clip(lower=0), q=90, floor=1.0).fillna(0)
    rel20_norm = pct_norm_pos(rel20.clip(lower=0), q=90, floor=1.0).fillna(0)
    rel60_norm = pct_norm_pos(rel60.clip(lower=0), q=90, floor=1.0).fillna(0)
    rel120_norm = pct_norm_pos(rel120.clip(lower=0), q=90, floor=1.0).fillna(0)

    mom_norm = np.clip(
        0.30 * ers_norm + 0.20 * slope_pos_norm + 0.10 * mom_mid_norm +
        0.10 * rel20_norm + 0.20 * rel60_norm + 0.10 * rel120_norm, 0, 1
    ).fillna(0)

    if turn.notna().any():
        lo, hi = np.nanpercentile(turn, 30), np.nanpercentile(turn, 90)
        denom = max(hi - lo, 1e-9)
        liq_norm = np.clip((turn - lo) / denom, 0, 1).fillna(0)
        liq_low = (turn < lo).astype(float)
    else: liq_norm = 0.0; liq_low = 0.0

    vol_sweet = (1 - np.minimum((volz - 1).abs() / 3, 1)).clip(0, 1).fillna(0)
    kairi_abs = kairi.abs()
    kairi_norm = (1 - np.minimum(kairi_abs / cap_q(kairi_abs, q=80, floor=3.0), 1)).clip(0, 1).fillna(0)

    # ✅ [v7.5 수정] V-Power 점수화 (TEC 팩터 비중 조절: vol 0.5 + kairi 0.3 + vp 0.2)
    vp_norm = pct_norm_pos(v_power, q=90, floor=1.0).fillna(0)
    tec_norm = np.clip(0.5 * vol_sweet + 0.3 * kairi_norm + 0.2 * vp_norm, 0, 1).fillna(0)

    base_score = (
        100 * w["RR"] * rr_norm + 100 * w["T1"] * t1_norm + 100 * w["SL"] * sl_norm +
        100 * w["NEAR"] * near_norm + 100 * w["MOM"] * mom_norm + 100 * w["LIQ"] * liq_norm +
        100 * w["TEC"] * tec_norm
    )

    pen = pd.Series(0.0, index=x.index)
    pen += 15.0 * (entry.isna() | ~np.isfinite(entry)).astype(float)
    pen += 10.0 * (stop.isna() | (stop <= 0)).astype(float)
    pen += 10.0 * (t1.isna() | (t1 <= 0)).astype(float)
    pen += P_OVERHEAT_5D * np.clip((r5 - 10) / 10, 0, 1)
    pen += P_OVERHEAT_10D * np.clip((r10 - 25) / 25, 0, 1)
    pen += P_RSI_OUT * ((rsi < RSI_LOW) | (rsi > RSI_HIGH)).astype(float)
    pen += P_MACD_NEG * (slope < 0).astype(float)
    pen += P_NEAR_FAR * np.clip((now_gap - 15) / 15, 0, 1)
    pen += P_LIQ_LOW * liq_low
    pen += P_VOL_SPIKE * (volz > 3).astype(float)
    pen += P_BIG_SL * (sl_pct > 12).astype(float)

    prelim_score = np.clip(base_score - pen, 0, 100)
    sector_bonus = 0.0
    # -------------------- [v9.0 섹터 보너스 강화] --------------------
    # 단순 상승 섹터가 아니라, 시장 대비 강한(RS > 0) 섹터에 보너스 부여
    if "SECTOR_RS" in x.columns:
        try:
            # RS(상대강도)가 양수일 때만 가산점 (시장보다 강할 때)
            sec_rs = nz_num(x["SECTOR_RS"]).clip(lower=0)
            
            # 상위 10% 섹터에 최대 점수 부여
            sec_norm = pct_norm_pos(sec_rs, q=90, floor=1.0).fillna(0)
            
            # W_SECTOR 가중치 적용 (기본 5점 ~ 최대 10점 효과)
            sector_bonus = 100 * W_SECTOR * sec_norm
        except: pass
    # -------------------------------------------------------------

    final_score = np.clip(prelim_score + sector_bonus, 0, 100)

    # 🔥 [v8.0] SuperTrend 보너스 점수 추가 (+3점)
    # 추세가 상승(1)인 경우 점수 상향 -> 추세 지속성 신뢰
    st_bonus = (st_dir == 1).astype(float) * 3.0
    final_score = np.clip(final_score + st_bonus, 0, 100)

    bb_sq = col_or_zero(x, "BB_SQUEEZE").clip(0, 1)
    final_score = np.clip(final_score + BONUS_BB_SQUEEZE_SCORE * bb_sq, 0, 100)

    raw_entry = (0.3*near_norm + 0.18*rr_norm + 0.12*t1_norm + 0.1*sl_norm + 0.2*mom_norm + 0.1*(1-liq_low))
    entry_score = np.clip(100 * raw_entry + BONUS_BB_SQUEEZE_ENTRY * bb_sq, 0, 100)

    above = col_or_zero(x, "Above_MA20").clip(0, 1)
    overheat = (0.6 * np.clip((r5 - 8) / 8, 0, 1) + 0.4 * np.clip((r10 - 18) / 18, 0, 1))
    rank_score = (0.55 * (entry_score/100) + 0.30 * (final_score/100) + 0.10 * bb_sq + 0.05 * above - 0.20 * overheat)

    x["NORM_RR"] = rr_norm.round(2); x["NORM_T1"] = t1_norm.round(2); x["NORM_SL"] = sl_norm.round(2)
    x["NORM_NEAR"] = near_norm.round(2); x["NORM_MOM"] = mom_norm.round(2); x["NORM_LIQ"] = liq_norm.round(2)
    x["NORM_TEC"] = tec_norm.round(2)

    x["RR1"] = rr1; x["Now%"] = now_gap; x["LDY_SCORE"] = final_score.round(1)
    x["ENTRY_SCORE"] = entry_score.round(1); x["RANK_SCORE"] = np.clip(rank_score * 100, 0, 100).round(1)
    x["ROUTE"] = x.apply(route_tag, axis=1); x["REGIME"] = x.apply(detect_regime_row, axis=1)
    x["ST_DIR"] = st_dir # 디버깅용 저장

    # ✅ [v7.5 수정] AI 코멘트 생성 시 V_POWER 전달
    x["AI_COMMENT"] = x.apply(lambda row: generate_ai_comment(
        row.get("MFI14", 50), row.get("RSI14", 50), row.get("MACD_Slope_PCT", 0),
        row.get("이격도", 0), row.get("LDY_SCORE", 0), row.get("TTM_SQUEEZE", 0), 
        row.get("BB_SQUEEZE_BW", 0), row.get("BB_BW", np.nan), row.get("TTM_SQUEEZE_CNT", 0),
        row.get("IS_SWING_SUPPORT", False), row.get("V_POWER", 0.0)
    ), axis=1)

    # ✅ [v10.0 추가] ML 엔진을 통한 AI 점수 예측 및 반영
    # 먼저 ML 점수를 계산해옴
    x = ml_engine.apply_ml_score(x)
    
    # 기존 RANK_SCORE와 ML_SCORE를 7:3 비율로 섞어 'FINAL_RANK_SCORE' 생성
    # (모델이 없으면 ML_SCORE가 0이므로 기존 점수만 사용되게 방어 로직 필요)
    if "ML_SCORE" in x.columns and x["ML_SCORE"].sum() > 0:
        x["TOTAL_SCORE"] = (x["RANK_SCORE"] * 0.7) + (x["ML_SCORE"] * 0.3)
    else:
        x["TOTAL_SCORE"] = x["RANK_SCORE"]
        
    x["TOTAL_SCORE"] = x["TOTAL_SCORE"].round(1)
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

def fetch_naver_news_headlines(code: str, days: int = 2) -> List[str]:
    """
    [v9.0] 네이버 금융 종목별 뉴스/공시 제목 크롤링
    - 최근 'days'일 이내의 뉴스만 수집
    """
    if not BS4_OK: return []
    
    headlines = []
    try:
        # 네이버 금융 뉴스 리스트 URL
        url = f"https://finance.naver.com/item/news_news.naver?code={code}&page=1"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=3)
        
        # 인코딩 처리 (EUC-KR -> UTF-8)
        soup = BeautifulSoup(resp.content.decode('euc-kr', 'replace'), "html.parser")
        
        # 뉴스 테이블 파싱
        # 구조: <tr> <td class="title"> <a ...>제목</a> </td> <td class="date">...</td> </tr>
        titles = soup.select("td.title > a")
        dates = soup.select("td.date")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for t, d in zip(titles, dates):
            article_date_str = d.text.strip() # 예: '2024.05.21 14:00'
            try:
                # 날짜 파싱 (시간 포맷이 다양할 수 있어 예외처리)
                if len(article_date_str) > 10:
                    a_date = datetime.strptime(article_date_str, "%Y.%m.%d %H:%M")
                else:
                    a_date = datetime.strptime(article_date_str, "%Y.%m.%d")
                
                if a_date >= cutoff_date:
                    subject = t.text.strip()
                    # 중복/광고성 필터링 (간단 예시)
                    if "특징주" in subject or "공시" in subject or "체결" in subject or "계약" in subject:
                        headlines.append(subject)
            except:
                continue
                
    except Exception as e:
        pass # 뉴스 수집 실패는 조용히 넘김
        
    return list(set(headlines))[:10] # 중복 제거 후 최대 10개



# [수정 대상] analyze_ticker 함수 전체를 아래 코드로 교체하세요.

def analyze_ticker(
    t: str, ohlcv_df: pd.DataFrame, top_df: pd.DataFrame, mcap_map: Dict[str, float],
    kospi_set: set, kosdaq_set: set, name_map: Dict[str, str], sector_map: Dict[str, str],
    bench_map: Dict[str, Dict[int, float]],
) -> Optional[Dict[str, Any]]:
    code6 = str(t).zfill(6)
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 120: return None
    ohlcv = ohlcv_df.tail(LOOKBACK_DAYS).copy()
    c = ohlcv["종가"]; h = ohlcv["고가"]; l = ohlcv["저가"]; v = ohlcv["거래량"]; o = ohlcv["시가"]
    
    # 데이터 타입 안전 변환
    c = pd.to_numeric(ohlcv["종가"], errors='coerce')
    h = pd.to_numeric(ohlcv["고가"], errors='coerce')
    l = pd.to_numeric(ohlcv["저가"], errors='coerce')
    v = pd.to_numeric(ohlcv["거래량"], errors='coerce')
    o = pd.to_numeric(ohlcv["시가"], errors='coerce')
    
    last_c = float(c.iloc[-1])

    ma20 = c.rolling(BB_PERIOD).mean(); ma60 = c.rolling(60).mean(); ma120 = c.rolling(120).mean()
    # -------------------- [v9.0 HMA 로직 추가] --------------------
    # HMA 20일선 계산
    hma20 = calc_hma(c, 20)
    curr_hma = float(hma20.iloc[-1]) if len(hma20) > 0 else np.nan
    prev_hma = float(hma20.iloc[-2]) if len(hma20) > 1 else np.nan

    # 1) HMA 추세 (상승/하락)
    hma_trend_up = False
    if np.isfinite(curr_hma) and np.isfinite(prev_hma):
        hma_trend_up = curr_hma > prev_hma

    # 2) 현재가가 HMA 위에 있는지 (지지)
    above_hma = 1 if (np.isfinite(curr_hma) and last_c > curr_hma) else 0

    # 3) 골든크로스 (전일 아래 -> 금일 위)
    prev_c = float(c.iloc[-2])
    hma_cross_up = False
    if (prev_c < prev_hma) and (last_c > curr_hma):
        hma_cross_up = True

    # -------------------- [v9.0 OBV 다이버전스 로직] --------------------
    obv = calc_obv(c, v)
    obv_ma20 = obv.rolling(20).mean() # OBV 추세선

    # [매집 패턴 감지]
    term = 20
    is_obv_div = False

    if len(c) > term:
        # 20일 전 대비 주가 변동률
        price_chg_pct = (last_c - float(c.iloc[-(term+1)])) / float(c.iloc[-(term+1)])
        # 20일 전 대비 OBV 변동량
        obv_chg_val = float(obv.iloc[-1] - obv.iloc[-(term+1)])

        # 조건: 주가 횡보/약세(-inf ~ 3%) & OBV 증가 & OBV가 이평선 위
        if (price_chg_pct < 0.03) and (obv_chg_val > 0) and (float(obv.iloc[-1]) > float(obv_ma20.iloc[-1])):
            is_obv_div = True
    # ------------------------------------------------------------------
    # -------------------------------------------------------------
    std20 = c.rolling(BB_PERIOD).std()
    bb_upper = ma20 + (BB_STD * std20); bb_lower = ma20 - (BB_STD * std20)
    bb_bw = ((bb_upper - bb_lower) / ma20.replace(0, np.nan)) * 100
    bb_bw_val = float(bb_bw.iloc[-1]) if len(bb_bw) else np.nan
    bw_squeeze = 1 if (np.isfinite(bb_bw_val) and bb_bw_val < BB_SQUEEZE_BW) else 0

    atr_kc_series = calc_atr(h, l, c, KC_ATR_PERIOD)
    kc_mid = ema(c, KC_PERIOD)
    kc_upper = kc_mid + (KC_MULT * atr_kc_series); kc_lower = kc_mid - (KC_MULT * atr_kc_series)
    ttm_series = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    ttm_last = ttm_series.iloc[-1] if len(ttm_series) else False
    ttm_squeeze = 1 if (bool(ttm_last) and not pd.isna(ttm_last)) else 0
    bb_squeeze = int(ttm_squeeze)

    sqz_cnt = 0
    if ttm_squeeze == 1:
        vals = ttm_series.iloc[:-1].values[::-1]
        temp_cnt = 0
        for val in vals:
            if val: temp_cnt += 1
            else: break
        sqz_cnt = temp_cnt + 1

    # ✅ [v7.5] Swing Low & V-Power
    swing_low_10 = float(l.tail(10).min())
    dist_to_swing = (last_c - swing_low_10) / last_c * 100
    is_swing_support = (dist_to_swing < 5.0) and (last_c > swing_low_10)

    tail5 = ohlcv.tail(5).copy()
    body = (tail5["종가"] - tail5["시가"]).abs()
    range_len = (tail5["고가"] - tail5["저가"]).replace(0, 1)
    sign = np.where(tail5["종가"] >= tail5["시가"], 1, -1)
    power_raw = (body / range_len) * tail5["거래량"] * sign
    avg_vol = tail5["거래량"].mean()
    v_power = power_raw.sum() / avg_vol if avg_vol > 0 else 0.0

    # ✅ [v8.5 추가] VWAP (최근 5일 주간 수급 단가) 계산
    # 단기 스윙 관점이므로 최근 5일(일주일)간의 거래량 가중 평단을 구합니다.
    vwap_val = calc_vwap(ohlcv.tail(5))
    # 현재가가 VWAP 대비 얼마나 위에 있는지(%) - 양수면 강한 수급 지지
    vwap_gap = (last_c - vwap_val) / vwap_val * 100 if vwap_val > 0 else 0.0

    # 🔥 [v8.0] SuperTrend 계산
    st_series, st_dir = calc_supertrend(h, l, c, period=10, multiplier=3.0)
    st_val = float(st_series.iloc[-1])
    st_trend = int(st_dir.iloc[-1]) # 1: 상승, -1: 하락

    # 🔥 [v8.5 오류 수정] 캔들 패턴 감지 코드 추가
    candle_patterns = check_candle_pattern(o, h, l, c)

    above_ma20 = 1 if (np.isfinite(ma20.iloc[-1]) and last_c > float(ma20.iloc[-1])) else 0
    atr = float(atr_kc_series.iloc[-1]) if len(atr_kc_series) else last_c * 0.03
    rsi = float(calc_rsi(c, 14).iloc[-1])
    mfi = float(calc_mfi(h, l, c, v, 14).iloc[-1])

    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9); hist = macd - sig
    hist_tail = hist.dropna().tail(5)
    slope = float(np.polyfit(np.arange(len(hist_tail)), hist_tail.values.astype(float), 1)[0]) if len(hist_tail) >= 2 else 0.0
    slope_pct = (slope / last_c) * 100.0 if last_c > 0 else 0.0
    vol_z = float((v / v.rolling(20).mean().replace(0, np.nan)).iloc[-1]) if len(v) else 0.0
    disp = (last_c / float(ma20.iloc[-1]) - 1.0) * 100 if ma20.iloc[-1] else 0.0

    def _ret(days): return (last_c / float(c.iloc[-(days + 1)]) - 1.0) * 100 if len(c) >= days + 1 else np.nan
    ret_5 = _ret(5); ret_10 = _ret(10); ret_20 = _ret(20); ret_60 = _ret(60); ret_120 = _ret(120)

    tv_row = top_df.loc[top_df["종목코드"] == code6, "거래대금(원)"]
    if tv_row.empty: return None
    tv_eok = float(tv_row.values[0]) / 1e8
    mcap = get_mcap_eok_from_map(mcap_map, code6)
    if mcap_map and mcap <= 0: return None
    if tv_eok < MIN_TURNOVER_EOK: return None

    score = 0; reason = []

    # === [v10.0 추가] 주봉(Weekly) 대추세 확증 필터 시작 ===
    try:
        # 주봉 변환 로직
        logic = {'시가': 'first', '고가': 'max', '저가': 'min', '종가': 'last', '거래량': 'sum'}
        
        # 인덱스를 Datetime으로 복사 (리샘플링용)
        ohlcv_tmp = ohlcv.copy()
        if not isinstance(ohlcv_tmp.index, pd.DatetimeIndex):
            ohlcv_tmp.index = pd.to_datetime(ohlcv_tmp.index)
            
        df_w = ohlcv_tmp.resample('W').apply(logic)
        w_ma20 = df_w['종가'].rolling(window=20).mean()
        
        curr_w_c = float(df_w['종가'].iloc[-1])
        curr_w_ma = float(w_ma20.iloc[-1])
        prev_w_ma = float(w_ma20.iloc[-2]) if len(w_ma20) > 1 else curr_w_ma
        
        is_above_w20 = curr_w_c > curr_w_ma if np.isfinite(curr_w_ma) else False
        is_w20_up = curr_w_ma > prev_w_ma if np.isfinite(curr_w_ma) and np.isfinite(prev_w_ma) else False

        if is_above_w20:
            score += 1.5
            reason.append("주봉20선↑")
        if is_w20_up:
            score += 0.5
            reason.append("주봉추세우상향")
        if not is_above_w20:
            score -= 2.0  # 주봉 역배열 종목은 강력하게 감점
            reason.append("주봉역배열주의")
    except Exception:
        pass
    # === [v10.0 추가] 주봉(Weekly) 대추세 확증 필터 끝 ===
    # -------------------- [v9.0 HMA 가산점] --------------------
    # 전략 1: HMA 우상향 + 주가 지지 (+1.5점) -> 강력한 추세
    if hma_trend_up and above_hma:
        score += 1.5
        reason.append("HMA추세↑")
    
    # 전략 2: HMA 상향 돌파 (+1.0점) -> 진입 타이밍
    if hma_cross_up:
        score += 1.0
        reason.append("HMA돌파")

    # -------------------- [v9.0 OBV 가산점] --------------------
    if is_obv_div:
        score += 2.0
        reason.append("OBV매집")
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    if RSI_LOW <= rsi <= RSI_HIGH: score += 1; reason.append("RSI적정")
    if slope > 0: score += 1; reason.append("MACD상승")
    if -1 <= disp <= 5: score += 1; reason.append("20선근접")
    if vol_z > 1.2: score += 1; reason.append("거래량↑")
    if ma20.iloc[-1] > ma60.iloc[-1]: score += 1; reason.append("정배열(단)")
    if last_c > ma120.iloc[-1]: score += 1; reason.append("장기추세(120↑)")
    else: score -= 1
    if mfi > 60: score += 1; reason.append("자금유입")
    if hist.iloc[-1] > 0: score += 1; reason.append("MACD>Sig")

    # ✅ [v8.5 추가] VWAP 지지 여부 (수급 질적 분석)
    # 현재가가 최근 5일 평균 매물대(VWAP)보다 높다면, 
    # 최근 매수자들이 수익 구간이므로 악성 매물이 적다고 판단 (+1점)
    # ✅ [v8.5 추가] VWAP 지지 여부
    if last_c > vwap_val: 
        score += 1
        reason.append("VWAP상회")

    # ✅ [v8.5 추가] 캔들 패턴 가산점 (+1점)
    # 기술적 반전 신호(망치형)나 강력한 매수세(장악형)가 떴다면 신뢰도 상승
    # ✅ [v8.5 추가] 캔들 패턴 가산점
    if candle_patterns:
        score += 1
        reason.append(f"패턴({','.join(candle_patterns)})")

    buy = min(last_c, ma20.iloc[-1] * 1.03) if (ma20.iloc[-1] > 0 and last_c > ma20.iloc[-1]) else last_c
    atr_mult = 2.0
    if ttm_squeeze == 1 or (np.isfinite(bb_bw_val) and bb_bw_val < 12.0): atr_mult = 1.8
    is_high_vol = (vol_z > 2.5) or (ret_5 > 10.0) or (disp > 5.0)
    if is_high_vol: atr_mult = 2.5

    stop = buy - (atr_mult * atr)

    # ✅ [v7.5] 기존 스윙 로우 보정
    if (dist_to_swing < 12.0) and (stop > swing_low_10):
        stop = swing_low_10 * 0.99

    # 🔥 [v8.0 핵심] SuperTrend 기반 스마트 손절 보정
    # 상승 추세(trend==1)이고, SuperTrend 라인이 현재가보다 아래에 있다면
    # 기존 손절가(ATR/SwingLow)와 SuperTrend 중 '더 높은 가격'을 손절가로 채택
    if st_trend == 1 and st_val < last_c:
        # 단, 진입가(buy)보다는 낮아야 함 (방어 코드)
        if st_val < buy:
            stop = max(stop, st_val)

    limit_pct = 0.90 if is_high_vol else 0.93
    stop = max(stop, buy * limit_pct)
    stop = min(stop, buy * 0.98)

# -------------------- [v9.0 동적 이익 실현 (Dynamic Profit Taking)] --------------------
    # 기존 고정 RR 방식 대신, 변동성(ATR)과 거래강도(vol_z)에 따라 목표가를 탄력적으로 조절
    
    risk = buy - stop  # 확정된 1주당 리스크 금액 (1R)
    
    # 1. 기본 목표 배수 (점수가 높을수록 신뢰도↑ -> 목표가 상향)
    base_mult = 2.0 if score >= 8 else (1.5 if score >= 6 else 1.2)
    
    # 2. 동적 확장 계수 (Dynamic Boost)
    # 거래량이 평소보다 폭발적(vol_z > 2.0)이라면 추세가 강하므로 목표가를 확장
    dynamic_boost = 1.0
    if vol_z > 3.0: 
        dynamic_boost = 1.5    # 초강력 수급 -> 목표가 50% 추가 상향
    elif vol_z > 2.0: 
        dynamic_boost = 1.2    # 강한 수급 -> 목표가 20% 추가 상향
    
    # 3. ATR 기반 최소 목표치 보정
    # 리스크가 너무 작게 잡혔을 경우(손절이 너무 타이트할 때), 
    # 최소한 ATR(일평균진폭)의 2배만큼은 먹을 구간을 확보
    min_target_dist = atr * 2.0
    
    # 4. 최종 목표가 산출
    # (리스크 기반 목표) vs (ATR 기반 목표) 중 더 큰 값을 채택하여 조기 매도 방지
    target_dist = max(risk * base_mult * dynamic_boost, min_target_dist)
    
    t1 = buy + target_dist
    t2 = buy + (target_dist * 2.0) # 2차 목표가는 1차 수익폭의 2배 지점
    # -------------------------------------------------------------------------------------

    buy = round_to_tick(buy); tk = tick_size(buy)
    stop = floor_to_tick(stop)
    if buy - stop < tk: stop = buy - tk
    t1 = ceil_to_tick(t1); t2 = ceil_to_tick(t2)
    if t1 <= buy: t1 = buy + tk * 2

    # -----------------------------------------------------------
    # ✅ [v8.5 추가] 변동성 기반 비중 조절 (Money Management)
    # -----------------------------------------------------------
    # 가정: 총 운용 자금 1,000만원, 1회 거래당 최대 손실 2% 제한, 최대 비중 25%
    # (실제 운영 시에는 이 값들을 설정 변수나 인자로 빼는 것이 좋습니다)
    ACCOUNT_CAPITAL = 10_000_000 
    RISK_PER_TRADE = 0.02         # 2% 리스크 (20만원)
    MAX_POS_PCT = 0.25            # 종목당 최대 비중 25% (250만원)
    RISK_PER_TRADE = 0.02         
    MAX_POS_PCT = 0.25            

    loss_per_share = buy - stop   # 1주당 예상 손실금액
    loss_per_share = buy - stop   
    rec_qty = 0
    rec_amt = 0

    if loss_per_share > 0 and buy > 0:
        # 1. 리스크 기준 허용 수량 (손절 나가도 내 자산의 2%만 잃게 설정)
        risk_based_qty = (ACCOUNT_CAPITAL * RISK_PER_TRADE) / loss_per_share
        
        # 2. 최대 비중 제한 수량 (아무리 좋아도 내 자산의 25% 이상 매수 금지)
        max_cap_qty = (ACCOUNT_CAPITAL * MAX_POS_PCT) / buy
        
        # -> 둘 중 더 적은 수량을 최종 추천 수량으로 선정 (보수적 접근)
        rec_qty = int(min(risk_based_qty, max_cap_qty))
        rec_amt = int(rec_qty * buy)

    sector = sector_map.get(code6, "기타")
    name = name_map.get(code6, code6)
    m_row = top_df.loc[top_df["종목코드"] == code6, "시장"]
    market = str(m_row.values[0]) if not m_row.empty else ("KOSPI" if code6 in kospi_set else "KOSDAQ")
    bench_dict = bench_map.get(market, {})
    idx_20 = bench_dict.get(20, np.nan); idx_60 = bench_dict.get(60, np.nan); idx_120 = bench_dict.get(120, np.nan)
    rel_20 = ret_20 - idx_20 if np.isfinite(idx_20) and np.isfinite(ret_20) else np.nan
    rel_60 = ret_60 - idx_60 if np.isfinite(idx_60) and np.isfinite(ret_60) else np.nan
    rel_120 = ret_120 - idx_120 if np.isfinite(idx_120) and np.isfinite(ret_120) else np.nan

    return {
        "시장": market, "종목명": name, "종목코드": code6, "업종": sector, "종가": int(last_c),
        "거래대금(억원)": round(tv_eok, 2), "시가총액(억원)": round(mcap, 1) if mcap_map else np.nan,
        "RSI14": round(rsi, 1), "MFI14": round(mfi, 1), "이격도": round(disp, 2),
        # ✅ [v8.5 추가] VWAP 데이터 저장
        "VWAP": int(vwap_val), "VWAP_GAP": round(vwap_gap, 2),
        # ✅ [v8.5 추가] 감지된 캔들 패턴 저장
        "캔들패턴": ",".join(candle_patterns) if candle_patterns else "",
        "MACD_Hist": round(float(hist.iloc[-1]), 4), "MACD_Slope_PCT": round(slope_pct, 4),
        "거래강도": round(vol_z, 2), "V_POWER": round(v_power, 2),
        "SUPERTREND_VAL": int(st_val) if np.isfinite(st_val) else 0, "SUPERTREND_DIR": int(st_trend) if np.isfinite(st_trend) else 1,
        "BB_BW": round(bb_bw_val, 2) if np.isfinite(bb_bw_val) else np.nan,
        "BB_SQUEEZE_BW": int(bw_squeeze), "TTM_SQUEEZE": int(ttm_squeeze), "TTM_SQUEEZE_CNT": int(sqz_cnt),
        "BB_SQUEEZE": int(bb_squeeze), "Above_MA20": int(above_ma20), 
        "IS_SWING_SUPPORT": is_swing_support,
        "ret_5d_%": round(ret_5, 2) if np.isfinite(ret_5) else np.nan,
        "ret_10d_%": round(ret_10, 2) if np.isfinite(ret_10) else np.nan,
        "ret_20d_%": round(ret_20, 2) if np.isfinite(ret_20) else np.nan,
        "ret_60d_%": round(ret_60, 2) if np.isfinite(ret_60) else np.nan,
        "ret_120d_%": round(ret_120, 2) if np.isfinite(ret_120) else np.nan,
        "rel_20d_%": round(rel_20, 2) if np.isfinite(rel_20) else np.nan,
        "rel_60d_%": round(rel_60, 2) if np.isfinite(rel_60) else np.nan,
        "rel_120d_%": round(rel_120, 2) if np.isfinite(rel_120) else np.nan,
        # === [v10.0 추가] 주봉 지표 데이터 저장 ===
        "주봉20선_상회": "O" if is_above_w20 else "X",
        "주봉추세": "▲" if is_w20_up else "▼",
        # ========================================
        "EBS": int(score), "통과": "★" if score >= PASS_EBS else "", "근거": ", ".join(reason),
        "추천매수가": buy, "손절가": stop, "추천매도가1": t1, "추천매도가2": t2, "ATR_MULT": atr_mult,
        # ✅ [v8.5 추가] 자금 관리 필드 추가
        "추천수량": rec_qty, 
        "추천금액(만원)": round(rec_amt / 10000, 1),  # <--- 여기에 쉼표(,)가 꼭 필요합니다!
        
        # ✅ [v9.0 추가] HMA & OBV 지표
        "HMA20": int(curr_hma) if np.isfinite(curr_hma) else 0,
        "HMA_Trend": "▲" if hma_trend_up else "▼",
        "HMA_On": "O" if above_hma else "X",
        "OBV_Div": "O" if is_obv_div else "X",
    }



# ------------------------------- 메인 실행 -------------------------------

def main(
    trade_date: Optional[str] = None,
    top_n: Optional[int] = None,
    enable_telegram: bool = True,
    tag: Optional[str] = None,
) -> None:
    log("🚀 LDY Collector v10.0 (AI Powered) 시작...")

    # ✅ [v10.0 추가] 주말이거나, 모델 파일이 없으면 학습 시도
    # (매번 학습하면 느리므로 파일이 없을 때만 수행하거나 별도 인자로 제어 가능)
    if not os.path.exists(ml_engine.MODEL_PATH):
        log("🤖 모델이 없습니다. 과거 데이터로 학습을 시작합니다...")
        ml_engine.train_model()
    
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
        raise RuntimeError("No Result (필터를 모두 통과한 종목 없음)")

    df_raw = pd.DataFrame(rows)

    # 🔹 업종 대분류 컬럼 생성
    if "업종" in df_raw.columns:
        df_raw["업종_상세"] = df_raw["업종"]
        df_raw["업종_대분류"] = df_raw.apply(
            lambda r: classify_big_sector(
                str(r.get("종목명", "")),
                str(r.get("업종", "")),
            ),
            axis=1,
        )

    # ✅ v6.8 섹터 모멘텀 + 시장 브레드스
    df_raw, sector_rank = add_sector_momentum(df_raw, "업종_대분류")
    breadth = compute_market_breadth(df_raw)
    mkt_temp = label_market_temp(breadth.get("ALL", np.nan))

    # [로그 추가] 시장 온도 확인
    log(f"🌡 시장 온도: {mkt_temp} (Breadth: {breadth.get('ALL', 0)}%) -> 동적 가중치 적용")

    # [수정] 스코어링 (mkt_temp 전달)
    df_out = build_global_score(df_raw, market_temp=mkt_temp)

    df_out["MKT_BREADTH_ALL_%"] = breadth.get("ALL", np.nan)
    df_out["MKT_BREADTH_KOSPI_%"] = breadth.get("KOSPI", np.nan)
    df_out["MKT_BREADTH_KOSDAQ_%"] = breadth.get("KOSDAQ", np.nan)
    df_out["MKT_TEMP"] = mkt_temp

    # 섹터 랭킹 상위 5개
    try:
        top_secs = sector_rank.head(5)
        df_out["TOP_SECTORS_5D"] = " / ".join(
            [f"{i+1}.{k}({v:+.1f}%)" for i, (k, v) in enumerate(top_secs.items())]
        )
    except Exception:
        df_out["TOP_SECTORS_5D"] = ""

    def _regime_rank(val: str) -> int:
        s = str(val)
        if s.startswith("①"): return 1
        if s.startswith("②"): return 2
        if s.startswith("③"): return 3
        if s.startswith("④"): return 4
        if s.startswith("⑤"): return 5
        return 999

    df_out["REGIME_RANK"] = df_out["REGIME"].map(_regime_rank).fillna(999).astype(int)

    df_out = df_out.sort_values(
        ["RANK_SCORE", "ENTRY_SCORE", "LDY_SCORE", "거래대금(억원)"],
        ascending=[False, False, False, False]
    )

    df_out["LDY_RANK"] = np.arange(1, len(df_out) + 1)
    df_out["기준일"] = trade_ymd
    df_out["시총기준일"] = mcap_ymd

    # -------------------------------------------------------------
    # 🔥 [v11.0 수정] 비동기 뉴스 수집 & LLM 분석 & DB 저장 통합
    # -------------------------------------------------------------
    if LLM_AVAILABLE:
        log("🧠 상위 10개 종목 뉴스/재료 심층 분석 중 (Async & LLM)...")
        
        # 1. 분석 대상 선정 (상위 10개)
        target_indices = df_out.index[:10]
        # 코드 포맷팅 (005930 등)
        target_codes = [str(df_out.loc[i, "종목코드"]).zfill(6) for i in target_indices]
        
        # 2. 비동기 뉴스 수집 실행 (병렬 처리로 속도 향상)
        try:
            fetcher = AsyncNewsFetcher(max_concurrent=5)
            # 윈도우 환경에서 asyncio 오류 발생 시 loop 정책 설정이 필요할 수 있음
            # if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            news_map = asyncio.run(fetcher.fetch_all(target_codes))
        except Exception as e:
            log(f"⚠️ 뉴스 수집 중 오류: {e}")
            news_map = {}

        # 3. LLM 분석 및 결과 반영
        df_out["NEWS_SCORE"] = 0.0
        df_out["NEWS_REASON"] = ""
        
        for idx in target_indices:
            code = str(df_out.loc[idx, "종목코드"]).zfill(6)
            name = df_out.loc[idx, "종목명"]
            headlines = news_map.get(code, [])
            
            # 뉴스 있으면 LLM 분석, 없으면 스킵
            if headlines:
                l_score, l_reason = analyze_sentiment_llm(name, headlines)
            else:
                l_score, l_reason = 0.0, "뉴스없음"
            
            # 점수 반영 (기존 로직 유지)
            old_score = df_out.at[idx, "LDY_SCORE"]
            new_score = np.clip(old_score + l_score, 0, 100)
            df_out.at[idx, "LDY_SCORE"] = new_score
            df_out.at[idx, "NEWS_SCORE"] = l_score
            df_out.at[idx, "NEWS_REASON"] = l_reason
            
            # 코멘트에 추가
            if l_reason and l_reason != "뉴스없음":
                old_comment = str(df_out.at[idx, "AI_COMMENT"])
                if old_comment == "nan": old_comment = ""
                df_out.at[idx, "AI_COMMENT"] = f"{old_comment} 📰재료: {l_reason}"
                
    else:
        log("ℹ️ LLM 설정(API Key)이 없거나 라이브러리가 없어 뉴스 분석을 건너뜁니다.")
        df_out["NEWS_SCORE"] = 0.0
        df_out["NEWS_REASON"] = ""

    # 🚨 [수정 완료] bench_ret_60 -> bench_map 사용으로 변경
    # bench_map 구조: {'KOSPI': {20: 1.1, 60: 2.2, ...}, ...}
    df_out["벤치_60d_KOSPI_%"] = bench_map.get("KOSPI", {}).get(60, np.nan)
    df_out["벤치_60d_KOSDAQ_%"] = bench_map.get("KOSDAQ", {}).get(60, np.nan)

    if "ret_60d_%" in df_out.columns:
        df_out["60일_종목수익률_%"] = df_out["ret_60d_%"]
    if "idx_60d_%" in df_out.columns:
        df_out["60일_지수수익률_%"] = df_out["idx_60d_%"]
    if "rel_60d_%" in df_out.columns:
        df_out["60일_초과수익(α)_%"] = df_out["rel_60d_%"]

    # 👉 메타 요약 로그
    try:
        avg_score = float(df_out["LDY_SCORE"].mean())
        top10_avg = float(df_out["LDY_SCORE"].head(10).mean())
        log(f"📌 전체 종목 수: {len(df_out)}개")
        log(f"📌 평균 점수: {avg_score:.1f}, 상위 10개 평균: {top10_avg:.1f}")

        # 간단 통계
        route_counts = df_out["ROUTE"].value_counts()
        route_str = ", ".join([f"{k}: {v}개" for k, v in route_counts.items()])
        log(f"📌 전략별 분포: {route_str}")

    except Exception as e:
        log(f"⚠️ 메타 요약 계산 실패: {e}")

    # [수정 대상] main 함수 하단 must_cols 리스트 업데이트
    must_cols = [
        "종목코드","종목명","시장","업종","업종_상세","업종_대분류",
        "종가","거래대금(억원)","시가총액(억원)",
        "추천매수가","손절가","추천매도가1","추천매도가2",
        "RANK_SCORE","LDY_SCORE","ENTRY_SCORE",
        "NEWS_SCORE", "NEWS_REASON",  # 👈 여기에 추가하면 보기 좋습니다
        "ROUTE","REGIME",
        "ret_20d_%", "ret_120d_%", 
        "rel_20d_%", "rel_60d_%", "rel_120d_%",
        "TTM_SQUEEZE_CNT",
        # ✅ [v7.4 추가] 팩터 컬럼 추가
        "NORM_RR", "NORM_T1", "NORM_SL", "NORM_NEAR", 
        "NORM_MOM", "NORM_LIQ", "NORM_TEC"
    ]
    for c in must_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan

    df_out = df_out[must_cols + [c for c in df_out.columns if c not in must_cols]]    

    ensure_dir(OUT_DIR)
    date_tag = trade_ymd
    suffix = f"_{tag}" if tag else ""
    out_path_dated = os.path.join(OUT_DIR, f"recommend_{date_tag}{suffix}.csv")
    out_path_latest = os.path.join(OUT_DIR, "recommend_latest.csv")

    df_out.to_csv(out_path_dated, index=False, encoding=UTF8)
    df_out.to_csv(out_path_latest, index=False, encoding=UTF8)

    log(f"💾 저장 완료 ({len(df_out)}건) → {out_path_dated}")
    log(f"💾 최신 파일 업데이트 → {out_path_latest}")

    # 🔥 [DuckDB에 추천 결과 저장] --------------------------
    try:
        db = LDYDBManager()
        db.save_recommendations(df_out, trade_ymd)
        db.close()
    except Exception as e:
        log(f"⚠️ 추천 결과 DB 저장 실패: {e}")
    # -----------------------------------------------------

    run_reality_check(OUT_DIR, trade_ymd)
    make_rank_validation_report(OUT_DIR, asof_ymd=trade_ymd)

    # main 함수 마지막 부분
    if enable_telegram:
        # 시장 요약 문구 생성
        summary_text = f"🌡 {mkt_temp} (Breadth: {breadth.get('ALL', 0)}%)"

        # ✅ 매크로 메시지 추가
        if macro_msg:
            summary_text += f"\n{macro_msg}"

        if "TOP_SECTORS_5D" in df_out.columns:
            # 상위 2개 섹터만 간략히
            try:
                top_sec_str = df_out.iloc[0]["TOP_SECTORS_5D"].split(" / ")[:2]
                summary_text += f"\n🚀 주도: {' '.join(top_sec_str)}"
            except: pass

        send_telegram_auto(df_out, trade_ymd, market_summary=summary_text, limit_count=rec_limit_cnt)
    else:
        log("✉️ --no-telegram 옵션으로 인해 텔레그램 발송 생략")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LDY Pro Trader Collector v7.5")
    parser.add_argument(
        "--date",
        type=str,
        help="거래 기준일 (YYYYMMDD). 미지정 시 자동 탐색",
        default=None,
    )
    parser.add_argument(
        "--top",
        type=int,
        help=f"거래대금 상위 N개 종목 (기본 {TOP_N})",
        default=None,
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="텔레그램 발송 비활성화",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="출력 파일 이름 뒤에 붙일 태그",
        default=None,
    )

    args = parser.parse_args()

    try:
        main(
            trade_date=args.date,
            top_n=args.top,
            enable_telegram=not args.no_telegram,
            tag=args.tag,
        )
    except Exception as e:
        log(f"❌ Collector 실행 중 치명적 오류: {e}")
        raise
