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
import json
import pickle  # ✅ [v7.0 추가] 데이터 직렬화/캐싱용
from typing import Dict, Any, Optional, Callable, Tuple, List

import ml_engine  # ✅ [v10.0 추가]

# [v20.8] Feature Contract 사전 검증 — 수집 시작 전 스키마 일관성 확인
try:
    from feature_contract import FEATURE_CONTRACT as _FC
    _fc_cols_ok = (list(_FC.columns) == ml_engine.FEATURE_COLS)
    if not _fc_cols_ok:
        import logging as _lg
        _lg.getLogger("collector").warning(
            f"⚠️ Feature Contract ≠ ml_engine.FEATURE_COLS! "
            f"contract={_FC.n_features}cols, ml={len(ml_engine.FEATURE_COLS)}cols"
        )
except ImportError:
    pass  # feature_contract 미설치 환경 허용
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
import gc
import logging
# 👇👇 [수정] 클래스 정의를 지우고, schema.py에서 가져오도록 변경 👇👇
from schema import RouteState
import stop_logic as SL
from stop_logic import (
    calc_stop_price, calc_rr_multiplier, sanitize_ohlcv, adjust_by_flow,
    check_entry_filter, conservative_exit_price, conservative_tp_price,
)



# [v9.0 추가] LLM 및 뉴스 크롤링용 라이브러리
try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False
    print("⚠️ BeautifulSoup4가 설치되지 않았습니다. (pip install beautifulsoup4)")

# ▼▼▼ [여기] secrets.toml 강제 로드 (TOML 파서 사용) ▼▼▼
def load_secrets_to_env():
    """
    Streamlit 없이 실행될 때 .streamlit/secrets.toml 파일을 읽어
    환경변수(os.environ)에 로드. [section] 구조도 정상 파싱.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        secrets_path = os.path.join(base_dir, ".streamlit", "secrets.toml")

        if not os.path.exists(secrets_path):
            return

        # TOML 파서: py3.11+ tomllib, 아니면 tomli 폴백
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                # 폴백: 기존 줄 단위 파서 (최후의 수단)
                with open(secrets_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("[") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and val:
                            os.environ[key] = val
                return

        with open(secrets_path, "rb") as f:
            data = tomllib.load(f)

        # ✅ [v14] #19: 섹션 키 네임스페이스 + 충돌 방지
        for key, val in data.items():
            if isinstance(val, str):
                os.environ[key] = val
            elif isinstance(val, (int, float, bool)):
                os.environ[key] = str(val)
            elif isinstance(val, dict):
                # 섹션 내부 키: sub_key 우선, SECTION_SUBKEY로도 등록 (충돌 방지)
                section = key.upper()
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, (str, int, float, bool)):
                        sv = str(sub_val)
                        # sub_key 직접 등록 (기존 호환)
                        if sub_key not in os.environ:
                            os.environ[sub_key] = sv
                        # SECTION_SUBKEY로도 등록 (네임스페이스)
                        ns_key = f"{section}_{sub_key.upper()}"
                        os.environ[ns_key] = sv

    except Exception as e:
        print(f"⚠️ Secrets 로드 실패: {e}")

# 실행 초기화 시점에 키 로드
load_secrets_to_env()
# ▲▲▲ [여기] 까지 삽입 ▲▲▲

# ── [v20.2] 공용 모듈에서 import (SSOT) ──
from shared_log import (
    log, logger, ensure_dir, safe_quantile, RunContext,
    GEMINI_API_KEY, LLM_AVAILABLE, _USE_NEW_GENAI,
    PYKRX_OK, OUT_DIR, UTF8, BASE_DIR,
    LOOKBACK_DAYS, BENCH_LOOKBACK_DAYS, TOP_N,
    MIN_TURNOVER_EOK, MIN_MCAP_EOK, MAX_WORKERS,
)

# [보안 설정]
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_ID = os.environ.get("TG_ID")

# ------------------------------- 설정 (collector_config.py SSOT) -------------------------------
from collector_config import CollectorConfig, DEFAULT_CONFIG as _CFG, Route, Market
DEFAULT_CONFIG = _CFG  # [v20.6.5] rank validation 등 내부 참조용 alias

RSI_LOW, RSI_HIGH = _CFG.rsi_low, _CFG.rsi_high
PASS_EBS = _CFG.pass_ebs

BB_PERIOD = _CFG.bb_period
BB_STD = _CFG.bb_std
BB_SQUEEZE_BW = _CFG.bb_squeeze_bw
KC_PERIOD = _CFG.kc_period
KC_ATR_PERIOD = _CFG.kc_atr_period
KC_MULT = _CFG.kc_mult
BONUS_BB_SQUEEZE_SCORE = _CFG.bonus_bb_squeeze_score
BONUS_BB_SQUEEZE_ENTRY = _CFG.bonus_bb_squeeze_entry

W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = (
    _CFG.w_rr, _CFG.w_t1, _CFG.w_sl, _CFG.w_near, _CFG.w_mom, _CFG.w_liq, _CFG.w_tec)
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = _CFG.p_overheat_5d, _CFG.p_overheat_10d, _CFG.p_rsi_out
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = (
    _CFG.p_macd_neg, _CFG.p_near_far, _CFG.p_liq_low, _CFG.p_vol_spike)
P_BIG_SL = _CFG.p_big_sl

# ── 분할 모듈 import ──
from data_source import KRXDataSource, OHLCVCache, get_data_source
from macro_filter import (
    check_macro_env, compute_market_breadth, label_market_temp, get_benchmark_returns,
)
from news_engine import (
    fetch_naver_news_headlines, analyze_sentiment_llm,
    generate_ai_comment, get_naver_theme_tags,
)
from telegram_sender import send_telegram_auto
from validation import (
    list_snapshot_days as _list_snapshot_days,
    load_close_map as _load_close_map,
    load_price_maps as _load_price_maps,
    pick_recommend_file_per_day as _pick_recommend_file_per_day,
    run_reality_check,
)

# [v20.6.5] 분리 모듈 재수출 (하위 호환)
from trigger_engine import calculate_trigger_score, calc_volume_profile_v2
from investor_flow import fetch_investor_net_buying

# ------------------------------- 유틸 (shared_log에서 import) -------------------------------
# log(), ensure_dir(), safe_quantile(), RunContext → shared_log.py (v20.2 SSOT)

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

# [v10.5] tick 유틸 — stop_logic.py에서 통합 관리 (중복 제거)
# collector 내부에서 기존 이름으로 계속 사용 가능
from stop_logic import tick_size, round_to_tick, floor_to_tick, ceil_to_tick

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


# [v14 REMOVED → compute_market_breadth moved to module] (15 lines deleted)


# [v14 REMOVED → label_market_temp moved to module] (8 lines deleted)

# [v14 REMOVED → _backoff_sleep moved to module] (4 lines deleted)

# [v14 REMOVED → run_reality_check moved to module] (50 lines deleted)


def _ymd8_to_dash(s: str) -> str:
    s = str(s)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s

# ═══════════════════════════════════════════════════
#  [v14 호환 래퍼] data_source 클래스 → 기존 함수명 매핑
# ═══════════════════════════════════════════════════
_ds = get_data_source()
_cache = OHLCVCache(OUT_DIR, fmt=_CFG.cache_format, allow_legacy_pickle=True)


def safe_ohlcv_by_date(start_ymd: str, end_ymd: str, code: str) -> Optional[pd.DataFrame]:
    """종목 코드 기반 OHLCV 조회 (기간)"""
    return _ds.get_ohlcv(str(code).zfill(6), start_ymd, end_ymd)


def safe_ohlcv_by_ticker(ymd: str, market: str = "ALL") -> Optional[pd.DataFrame]:
    """일별 전종목 OHLCV 조회"""
    return _ds.get_ohlcv_by_ticker(ymd, market=market)


def safe_market_cap_by_ticker(ymd: str, market: str = "ALL") -> Optional[pd.DataFrame]:
    """일별 전종목 시가총액 조회"""
    return _ds.get_market_cap(ymd, market=market)


def safe_ticker_list(ymd: str, market: str = "KOSPI") -> list:
    """종목 코드 리스트 조회"""
    return _ds.get_ticker_list(ymd, market=market)


def safe_ticker_name(ticker: str) -> Optional[str]:
    """종목명 조회"""
    return _ds.get_ticker_name(ticker)


def load_ohlcv_cache(trade_ymd: str) -> Dict[str, pd.DataFrame]:
    """OHLCV 캐시 로드"""
    return _cache.load(trade_ymd)


def save_ohlcv_cache(trade_ymd: str, data: Dict[str, pd.DataFrame]) -> None:
    """OHLCV 캐시 저장"""
    _cache.save(trade_ymd, data)

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

    # [v3.2 #2] 병렬 수집 완료 후 메모리 정리 — 워커 잔여 참조 해제
    del future_to_code
    gc.collect()

    return ohlcv_map


# ------------------------------- Rank Validation (RANK_SCORE 검증) -------------------------------

# [v14 REMOVED → _list_snapshot_days moved to module] (9 lines deleted)

# [v14 REMOVED → _load_close_map moved to module] (9 lines deleted)


# [v14 REMOVED → _load_price_maps moved to module] (24 lines deleted)

def _next_trade_day(trade_days: List[str], ymd: str, offset: int) -> Optional[str]:
    try:
        i = trade_days.index(ymd)
    except ValueError:
        return None
    j = i + offset
    if 0 <= j < len(trade_days):
        return trade_days[j]
    return None

# [v14 REMOVED → _pick_recommend_file_per_day moved to module] (15 lines deleted)



# [v22.3.14] rank validation 행별 KOSPI forward return / alpha helper
_KOSPI_FORWARD_RET_CACHE: Dict[Tuple[str, int], Dict[str, float]] = {}


def _load_kospi_forward_return_map(out_dir: str, horizon: int) -> Dict[str, float]:
    """data/kospi_daily.csv에서 rec_date별 KOSPI forward return을 로드.

    scripts/collect_kospi_daily.py가 생성하는 ret_{h}d_% 컬럼을 사용한다.
    없으면 빈 dict를 반환해 기존 rank validation 동작을 유지한다.
    """
    key = (str(out_dir), int(horizon))
    if key in _KOSPI_FORWARD_RET_CACHE:
        return _KOSPI_FORWARD_RET_CACHE[key]

    path = os.path.join(out_dir, "kospi_daily.csv")
    col = f"ret_{int(horizon)}d_%"
    result: Dict[str, float] = {}
    if not os.path.exists(path):
        _KOSPI_FORWARD_RET_CACHE[key] = result
        return result

    try:
        kdf = pd.read_csv(path, dtype={"date": str})
        if "date" not in kdf.columns or col not in kdf.columns:
            _KOSPI_FORWARD_RET_CACHE[key] = result
            return result
        kdf["date"] = kdf["date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
        vals = pd.to_numeric(kdf[col], errors="coerce")
        for d, v in zip(kdf["date"], vals):
            if d and len(d) == 8 and np.isfinite(v):
                result[d] = float(v)
    except Exception as e:
        logger.debug(f"KOSPI forward return 로드 실패 ({path}, {col}): {e}")
        result = {}

    _KOSPI_FORWARD_RET_CACHE[key] = result
    return result


def _lookup_kospi_forward_return(out_dir: str, rec_ymd: str, horizon: int) -> float:
    """rec_ymd 기준 horizon 영업일 KOSPI 수익률(%). 없으면 NaN."""
    mp = _load_kospi_forward_return_map(out_dir, horizon)
    try:
        return float(mp.get(str(rec_ymd), np.nan))
    except (TypeError, ValueError):
        return float("nan")

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
        per_trade_rows = []  # [v13] Kelly 캘리브레이션용 per-trade 히스토리

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
            df["손절가"] = pd.to_numeric(df.get("손절가", np.nan), errors="coerce")
            df["추천매도가1"] = pd.to_numeric(df.get("추천매도가1", np.nan), errors="coerce")  # [v10.5] 익절 체결용

            # 추천매수가가 없으면 종가로 대체(최소한 검증은 되게)
            entry_px = df["추천매수가"].where(df["추천매수가"].notna(), df["종가"])
            stop_px = df["손절가"]
            target_px = df["추천매도가1"]  # [v10.5] 목표가 시리즈

            for h in horizons:
                future_ymd = _next_trade_day(trade_days, rec_ymd, h)
                if not future_ymd:
                    continue

                close_map_future = _load_close_map(out_dir, future_ymd)
                if not close_map_future:
                    continue

                # MDD 계산 위해 중간 스냅샷도 로드(TopK만이니 부담 적음)
                # [v10.3] _load_price_maps로 종가+저가+시가+고가 전부 로드
                mid_days = []
                for k in range(1, h + 1):
                    dd = _next_trade_day(trade_days, rec_ymd, k)
                    if dd:
                        mid_days.append(dd)

                # [v20.6.5] _load_price_maps → {code: {close/high/low/open: val}}
                # rank validation은 {close: {code: val}} 형태 필요 → 전치
                def _transpose_pmaps(raw):
                    out = {"close": {}, "high": {}, "low": {}, "open": {}}
                    for code, fields in raw.items():
                        if isinstance(fields, dict):
                            for k in out:
                                if k in fields:
                                    out[k][code] = fields[k]
                    return out

                mid_price_maps = [(_d, _transpose_pmaps(_load_price_maps(out_dir, _d))) for _d in mid_days]

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
                        spx = pd.to_numeric(stop_px.loc[pick.index], errors="coerce").values.astype(float)
                        tpx = pd.to_numeric(target_px.loc[pick.index], errors="coerce").values.astype(float)  # [v10.5]

                        # H일 후 종가
                        fut_close = np.array([close_map_future.get(c, np.nan) for c in codes], dtype=float)
                        fut_close[fut_close <= 0] = np.nan

                        # 기본 ret (H일 후 종가 기준 — TP/SL 미적용)
                        ret = np.full_like(fut_close, np.nan, dtype=float)
                        ok_ret = np.isfinite(fut_close) & np.isfinite(epx) & (epx > 0)
                        ret[ok_ret] = (fut_close[ok_ret] / epx[ok_ret] - 1.0) * 100.0

                        # ── [v10.5] TP(익절) + SL(손절) 장중 감지 ──
                        min_close = np.full_like(fut_close, np.nan)
                        stop_hit = np.zeros(len(codes), dtype=bool)
                        tp_hit = np.zeros(len(codes), dtype=bool)      # [v10.5]
                        exit_prices = np.full(len(codes), np.nan)
                        n_extreme = 0

                        for _d, pmaps in mid_price_maps:
                            arr_close = np.array([pmaps["close"].get(c, np.nan) for c in codes], dtype=float)
                            arr_low = np.array([pmaps["low"].get(c, np.nan) for c in codes], dtype=float)
                            arr_open = np.array([pmaps["open"].get(c, np.nan) for c in codes], dtype=float)
                            arr_high = np.array([pmaps["high"].get(c, np.nan) for c in codes], dtype=float)

                            arr_close[arr_close <= 0] = np.nan
                            arr_low[arr_low <= 0] = np.nan
                            arr_open[arr_open <= 0] = np.nan
                            arr_high[arr_high <= 0] = np.nan

                            from trade_plan import TradePlan, ExecRule, exec_bar as _exec_bar
                            _exec_rule = ExecRule()

                            for j in range(len(codes)):
                                if stop_hit[j] or tp_hit[j]:
                                    continue

                                open_j = float(arr_open[j]) if np.isfinite(arr_open[j]) else 0.0
                                low_j = float(arr_low[j]) if np.isfinite(arr_low[j]) else float(arr_close[j]) if np.isfinite(arr_close[j]) else 0.0
                                high_j = float(arr_high[j]) if np.isfinite(arr_high[j]) else 0.0
                                close_j = float(arr_close[j]) if np.isfinite(arr_close[j]) else 0.0

                                # SSOT 체결: exec_bar 한 줄로 SL/TP 모두 처리
                                _plan_j = TradePlan(
                                    entry=float(epx[j]) if np.isfinite(epx[j]) else 0.0,
                                    stop=float(spx[j]) if np.isfinite(spx[j]) else 0.0,
                                    tp1=float(tpx[j]) if np.isfinite(tpx[j]) else 0.0,
                                    exec_rule_id=_exec_rule.rule_id,
                                    time_stop_days=DEFAULT_CONFIG.time_stop_days,
                                    time_stop_min_move_pct=DEFAULT_CONFIG.time_stop_min_move_pct,
                                    time_stop_extend_if_profit=DEFAULT_CONFIG.time_stop_extend_if_profit,
                                )
                                bar_result = _exec_bar(
                                    _plan_j,
                                    bar_open=open_j if open_j > 0 else low_j,
                                    bar_high=high_j,
                                    bar_low=low_j,
                                    bar_close=close_j,
                                    rule=_exec_rule,
                                )

                                if bar_result.action == "stop_hit":
                                    stop_hit[j] = True
                                    exit_prices[j] = bar_result.fill_price
                                    if np.isfinite(epx[j]) and epx[j] > 0:
                                        ret[j] = bar_result.return_pct
                                elif bar_result.action == "tp_hit":
                                    tp_hit[j] = True
                                    exit_prices[j] = bar_result.fill_price
                                    if np.isfinite(epx[j]) and epx[j] > 0:
                                        ret[j] = bar_result.return_pct

                            # MDD 트래킹: 청산된 종목은 exit_px로 고정
                            arr_for_mdd = np.where(np.isfinite(arr_low), arr_low, arr_close)
                            for j in range(len(codes)):
                                if (stop_hit[j] or tp_hit[j]) and np.isfinite(exit_prices[j]):
                                    arr_for_mdd[j] = exit_prices[j]

                            if np.isnan(min_close).all():
                                min_close = arr_for_mdd
                            else:
                                min_close = np.nanmin(np.vstack([min_close, arr_for_mdd]), axis=0)

                        # ✅ mdd safe-divide (inf 방지)
                        mdd = np.full_like(fut_close, np.nan, dtype=float)
                        ok_mdd = np.isfinite(min_close) & np.isfinite(epx) & (epx > 0)
                        # [v22.3.8-B2] AVG_MDD_% 부호 오염 방지 — TP hit 시 exit_price가
                        # min_close에 반영되어 MDD가 양수화되는 케이스 방지. MDD는 정의상
                        # 0 이하이므로 0으로 clip. 추천 로직 / BUY_NOW_ELIGIBLE / ASR 영향 없음.
                        # (정확한 intrabar dip 복원은 v22.4-mdd-tracker-redesign에서 처리)
                        mdd[ok_mdd] = np.minimum(
                            (min_close[ok_mdd] / epx[ok_mdd] - 1.0) * 100.0, 0.0
                        )

                        # [v10.3] 극단 이벤트 분리: -30% 이하를 NaN으로 덮지 않고 별도 카운트
                        extreme_mask = mdd < -30.0
                        n_extreme = int(np.sum(extreme_mask & np.isfinite(mdd)))

                        # ✅ (D-2) n 정의 + 샘플 0이면 skip
                        valid_ret = np.isfinite(ret)
                        valid = valid_ret & np.isfinite(mdd)
                        n = int(valid.sum())
                        if n == 0:
                            continue

                        r = ret[valid]
                        md = mdd[valid]

                        # [v22.3.14] 행별 KOSPI forward return / ALPHA_%.
                        # rec_ymd 기준 ret_{h}d_%를 사용해 검증일별 정확 알파를 계산한다.
                        kospi_ret = _lookup_kospi_forward_return(out_dir, rec_ymd, h)
                        if np.isfinite(kospi_ret):
                            alpha_arr = r - float(kospi_ret)
                            alpha_avg = float(np.nanmean(alpha_arr)) if len(alpha_arr) else np.nan
                            alpha_win_rate = float((alpha_arr > 0).mean() * 100) if len(alpha_arr) else np.nan
                        else:
                            alpha_avg = np.nan
                            alpha_win_rate = np.nan

                        # [v10.5] hit rate는 ret 유효 기준 (mdd NaN이어도 체결은 된 경우 포함)
                        n_ret = int(valid_ret.sum())
                        stop_hit_rate = float(stop_hit[valid_ret].mean() * 100) if n_ret > 0 else 0.0
                        tp_hit_rate = float(tp_hit[valid_ret].mean() * 100) if n_ret > 0 else 0.0

                        # ── [v13] per-trade 히스토리 수집 (Kelly 캘리브레이션용) ──
                        for j in range(len(codes)):
                            if not np.isfinite(ret[j]) or not np.isfinite(epx[j]) or epx[j] <= 0:
                                continue
                            _score_j = float(pick[method].iloc[j]) if method in pick.columns else 0.0
                            _risk_j = float(epx[j] - spx[j]) if np.isfinite(spx[j]) and spx[j] > 0 else 0.0
                            _reward_j = float(tpx[j] - epx[j]) if np.isfinite(tpx[j]) and tpx[j] > 0 else 0.0
                            _b_j = (_reward_j / _risk_j) if _risk_j > 0 else 0.0
                            _exit_type = "stop_hit" if stop_hit[j] else ("tp_hit" if tp_hit[j] else "hold_close")
                            per_trade_rows.append({
                                "rec_date": rec_ymd,
                                "code": codes[j],
                                "method": method,
                                "topk": topk,
                                "horizon": h,
                                "score": round(_score_j, 1),
                                "entry_price": round(float(epx[j]), 0),
                                "exit_price": round(float(exit_prices[j]), 0) if np.isfinite(exit_prices[j]) else 0,
                                "stop_price": round(float(spx[j]), 0) if np.isfinite(spx[j]) else 0,
                                "target_price": round(float(tpx[j]), 0) if np.isfinite(tpx[j]) else 0,
                                "ret_pct": round(float(ret[j]), 2),
                                "win": 1 if ret[j] > 0 else 0,
                                "exit_type": _exit_type,
                                "b_ratio": round(_b_j, 2),
                                # [v4.0] 세그먼트 축 — pick(recommend slice)에 있으면 기록, 없으면 ""
                                "MACRO_REGIME_MODE": (pick["MACRO_REGIME_MODE"].iloc[j]
                                                      if "MACRO_REGIME_MODE" in pick.columns else ""),
                                "ACTION_TIER": (pick["ACTION_TIER"].iloc[j]
                                                if "ACTION_TIER" in pick.columns else ""),
                                "ROUTE": (pick["ROUTE"].iloc[j] if "ROUTE" in pick.columns else ""),
                                "TOP_PICK_TYPE": (pick["TOP_PICK_TYPE"].iloc[j]
                                                  if "TOP_PICK_TYPE" in pick.columns else ""),
                            })

                        rows.append({
                            "추천일": rec_ymd,
                            "비교종가일": future_ymd,
                            "H(영업일)": h,
                            "METHOD": method,
                            "TOPK": topk,
                            "N": n,
                            "WIN_RATE_%": round(float((r > 0).mean() * 100), 1),
                            "AVG_RET_%": round(float(np.nanmean(r)), 2),
                            "KOSPI_RET_%": round(float(kospi_ret), 2) if np.isfinite(kospi_ret) else np.nan,
                            "ALPHA_%": round(float(alpha_avg), 2) if np.isfinite(alpha_avg) else np.nan,
                            "ALPHA_WIN_RATE_%": round(float(alpha_win_rate), 1) if np.isfinite(alpha_win_rate) else np.nan,
                            "MED_RET_%": round(float(np.nanmedian(r)), 2),
                            "HIT_2%_%": round(float((r >= 2).mean() * 100), 1),
                            "HIT_5%_%": round(float((r >= 5).mean() * 100), 1),
                            "AVG_MDD_%": round(float(np.nanmean(md)), 2),
                            "WORST_MDD_%": round(float(np.nanmin(md)), 2),
                            "STOP_HIT_RATE_%": round(stop_hit_rate, 1),
                            "TP_HIT_RATE_%": round(tp_hit_rate, 1),
                            "VAR_95_%": round(float(np.sort(r)[:max(1, int(len(r)*0.05))].mean()), 2),
                            "PL_RATIO": round(float(r[r>0].mean() / abs(r[r<0].mean())) if (r<0).any() and (r>0).any() else 0.0, 2),
                            "MAX_CONSEC_LOSS": int(max((sum(1 for _ in g) for k, g in __import__('itertools').groupby(r < 0) if k), default=0)),
                            # [v10.3] 극단 이벤트 분리 추적 + 경고 플래그
                            "N_EXTREME": n_extreme,
                            "RISK_FLAG": "🔴" if (float(np.nanmin(md)) < -10.0 or n_extreme > 0) else ("🟡" if float(np.nanmin(md)) < -6.0 else "🟢"),
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
            "KOSPI_RET_%": round(_wavg(g, "KOSPI_RET_%"), 2) if "KOSPI_RET_%" in g.columns and g["KOSPI_RET_%"].notna().any() else np.nan,
            "ALPHA_%": round(_wavg(g, "ALPHA_%"), 2) if "ALPHA_%" in g.columns and g["ALPHA_%"].notna().any() else np.nan,
            "ALPHA_WIN_RATE_%": round(_wavg(g, "ALPHA_WIN_RATE_%"), 1) if "ALPHA_WIN_RATE_%" in g.columns and g["ALPHA_WIN_RATE_%"].notna().any() else np.nan,
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

        # ── [v13] per-trade 히스토리 저장 + 캘리브레이션 테이블 빌드 ──
        if per_trade_rows:
            try:
                from kelly_calibrator import save_per_trade_log, build_calibration_table
                pt_path = save_per_trade_log(out_dir, per_trade_rows, asof_ymd)
                if pt_path:
                    cal_df = build_calibration_table(out_dir, asof_ymd=asof_ymd)
                    n_cal = len(cal_df) if cal_df is not None else 0
                    log(f"📊 [v13] per-trade log: {len(per_trade_rows)}건 → {pt_path}")
                    log(f"📊 [v13] calibration table: {n_cal}개 구간 빌드 완료")
            except Exception as e:
                log(f"⚠️ [v13] per-trade/calibration 저장 실패: {e}")

        # ── [v10.5] 워크포워드 리스크 KPI 자동 리포트 ──
        try:
            from stop_logic import rolling_risk_kpi
            # RANK_SCORE Top5, H=3 기준으로 워크포워드 분석
            rk_filter = detail[
                (detail["METHOD"] == "RANK_SCORE") &
                (detail["TOPK"] == 5) &
                (detail["H(영업일)"] == 3)
            ]
            if len(rk_filter) >= 5:
                dates_arr = rk_filter["추천일"].values
                rets_arr = rk_filter["AVG_RET_%"].values
                roll_df = rolling_risk_kpi(dates_arr, rets_arr, window_days=10)
                if not roll_df.empty:
                    roll_path = os.path.join(out_dir, f"risk_kpi_rolling_{asof_ymd}.csv")
                    roll_latest = os.path.join(out_dir, "risk_kpi_rolling_latest.csv")
                    roll_df.to_csv(roll_path, index=False, encoding=UTF8)
                    roll_df.to_csv(roll_latest, index=False, encoding=UTF8)
                    # 경고 플래그 요약
                    n_red = (roll_df["stability_flag"] == "🔴").sum()
                    n_yellow = (roll_df["stability_flag"] == "🟡").sum()
                    n_green = (roll_df["stability_flag"] == "🟢").sum()
                    log(f"📊 워크포워드 KPI → {roll_path} "
                        f"(🟢{n_green} 🟡{n_yellow} 🔴{n_red})")
                    if n_red > 0:
                        log(f"⚠️ 🔴 위험 구간 {n_red}개 감지 → 손절 설계 재점검 필요!")
        except Exception as e:
            log(f"⚠️ 워크포워드 KPI 생성 실패: {e}")

    except Exception as e:
        log(f"⚠️ rank validation 실패: {e}")

# ------------------------------- 거래일/시총 -------------------------------

def _has_ohlcv_and_mcap(ymd: str) -> bool:
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            o = safe_ohlcv_by_ticker(ymd, market=m)
            if o is not None and not o.empty and "거래대금" in o.columns and _safe_sum(o["거래대금"]) > 0:
                return True
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"OHLCV 체크 실패 ({m}/{ymd}): {e}")
        except Exception as e:
            logger.warning(f"⚠️ OHLCV 체크 예기치 않은 오류 ({m}/{ymd}): {type(e).__name__}: {e}")

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

def _build_mcap_map_fdr() -> Dict[str, float]:
    """
    [v19.1] FDR 폴백: pykrx 차단 시 FDR StockListing의 Marcap 컬럼으로 시총 맵 생성
    Marcap 단위: 원 → 억원 변환
    """
    try:
        df_fdr = fdr.StockListing("KRX")
        if df_fdr is None or df_fdr.empty:
            return {}

        code_col = "Code" if "Code" in df_fdr.columns else ("Symbol" if "Symbol" in df_fdr.columns else None)
        mcap_col = "Marcap" if "Marcap" in df_fdr.columns else None

        if not code_col or not mcap_col:
            log(f"⚠️ FDR 시총 폴백: 필요 컬럼 없음 (cols={df_fdr.columns.tolist()})")
            return {}

        df_fdr[code_col] = df_fdr[code_col].astype(str).str.zfill(6)
        df_fdr[mcap_col] = pd.to_numeric(df_fdr[mcap_col], errors="coerce").fillna(0)
        # Marcap은 원 단위 → 억원 변환
        mcap_map = dict(zip(df_fdr[code_col], df_fdr[mcap_col] / 1e8))
        # 0 이하 제거
        mcap_map = {k: v for k, v in mcap_map.items() if v > 0}
        log(f"✅ [FDR 폴백] 시총 맵 생성 성공: {len(mcap_map)}개 종목")
        return mcap_map

    except Exception as e:
        log(f"⚠️ FDR 시총 폴백 실패: {e}")
        return {}


def build_mcap_map(ref_ymd: Optional[str] = None) -> Tuple[Dict[str, float], str]:
    use = ref_ymd or now_kst().strftime("%Y%m%d")

    # ── 1순위: pykrx ──
    if PYKRX_OK and stock is not None:
        def _check_mcap(ymd: str) -> bool:
            a = safe_market_cap_by_ticker(ymd, market="KOSPI")
            b = safe_market_cap_by_ticker(ymd, market="KOSDAQ")
            return (a is not None and not a.empty) or (b is not None and not b.empty)

        pykrx_use = None
        if ref_ymd and _check_mcap(ref_ymd):
            pykrx_use = ref_ymd
        if pykrx_use is None:
            pykrx_use = find_latest_valid_date(_check_mcap, max_back_days=10)

        try:
            parts = []
            a = safe_market_cap_by_ticker(pykrx_use, market="KOSPI")
            b = safe_market_cap_by_ticker(pykrx_use, market="KOSDAQ")
            if a is not None and not a.empty: parts.append(a)
            if b is not None and not b.empty: parts.append(b)

            df = pd.concat(parts) if parts else pd.DataFrame()
            if not df.empty:
                df["Code"] = df.index.astype(str).str.zfill(6)
                mcap_map = dict(zip(df["Code"], df["시가총액"] / 1e8))
                log(f"✅ [pykrx] 시총 맵 생성 성공: {len(mcap_map)}개 종목")
                # [v19.2] 성공 시 캐시 저장
                _save_mcap_cache(mcap_map, OUT_DIR)
                return mcap_map, pykrx_use
        except Exception as e:
            log(f"⚠️ pykrx 시총 맵 실패: {e}")

    # ── 2순위: FDR 폴백 (pykrx 차단/실패 시) ──
    log("🔄 pykrx 시총 실패 → FDR Marcap 폴백 시도...")
    fdr_map = _build_mcap_map_fdr()
    if fdr_map:
        _save_mcap_cache(fdr_map, OUT_DIR)
        return fdr_map, use

    # ── 3순위: [v19.2] 로컬 캐시 폴백 ──
    cached = _load_mcap_cache(OUT_DIR)
    if cached:
        log(f"📂 [폴백] 시총 캐시 로드 성공: {len(cached)}개 종목")
        return cached, use

    log(f"⚠️ 시총 맵 생성 실패 (pykrx+FDR+캐시 모두), 빈 맵 반환")
    return {}, use


def _save_mcap_cache(mcap_map: Dict[str, float], out_dir: str) -> None:
    """시총 맵을 JSON 캐시로 저장"""
    import json
    try:
        ensure_dir(out_dir)
        path = os.path.join(out_dir, "mcap_cache_latest.json")
        with open(path, "w") as f:
            json.dump(mcap_map, f)
        log(f"📂 [캐시] 시총 맵 저장: {len(mcap_map)}개 → {path}")
    except Exception as e:
        log(f"⚠️ 시총 캐시 저장 실패: {e}")


def _load_mcap_cache(out_dir: str) -> Dict[str, float]:
    """최근 저장된 시총 캐시 로드 + 다단계 폴백"""
    import json
    # 1순위: 전용 캐시
    try:
        path = os.path.join(out_dir, "mcap_cache_latest.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            if data:
                return data
    except Exception:
        pass

    # 2순위: 이전 recommend CSV에서 시총 컬럼 복구
    try:
        rec_path = os.path.join(out_dir, "recommend_latest.csv")
        if os.path.exists(rec_path):
            _df = pd.read_csv(rec_path, dtype={"종목코드": str}, usecols=["종목코드", "시가총액(억원)"])
            _df["종목코드"] = _df["종목코드"].astype(str).str.zfill(6)
            _df["시가총액(억원)"] = pd.to_numeric(_df["시가총액(억원)"], errors="coerce").fillna(0)
            _valid = _df[_df["시가총액(억원)"] > 0]
            if not _valid.empty:
                mcap = dict(zip(_valid["종목코드"], _valid["시가총액(억원)"]))
                log(f"📂 [폴백] recommend CSV에서 시총 복구: {len(mcap)}개")
                return mcap
    except Exception:
        pass

    # 3순위: [v20.0] 거래대금 기반 시총 추정 (API 전면 차단 시 최후 수단)
    # 거래대금은 OHLCV 캐시에서 항상 사용 가능
    try:
        rec_path = os.path.join(out_dir, "recommend_latest.csv")
        if os.path.exists(rec_path):
            _df = pd.read_csv(rec_path, dtype={"종목코드": str})
            _df["종목코드"] = _df["종목코드"].astype(str).str.zfill(6)
            tv_col = "거래대금(억원)" if "거래대금(억원)" in _df.columns else "거래대금(억)"
            if tv_col in _df.columns:
                _df["_tv"] = pd.to_numeric(_df[tv_col], errors="coerce").fillna(0)
                mcap = {}
                for _, row in _df.iterrows():
                    tv = row["_tv"]
                    # 거래대금 → 시총 추정 (경험적 매핑)
                    if tv >= 1000:
                        est = 100000  # 대형주
                    elif tv >= 500:
                        est = 60000
                    elif tv >= 200:
                        est = 30000
                    elif tv >= 50:
                        est = 10000
                    elif tv >= 15:
                        est = 5000
                    else:
                        est = 2000   # 소형주
                    mcap[row["종목코드"]] = est
                if mcap:
                    log(f"📂 [폴백] 거래대금 기반 시총 추정: {len(mcap)}개 (정확도 낮음, API 복구 필요)")
                    return mcap
    except Exception:
        pass

    return {}

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

# [v14 REMOVED → get_benchmark_returns moved to module] (45 lines deleted)

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
    # ------------------------------------------------------------
    # 3. 3차 시도 (최후의 보루): 전일자 주도주(recommend_latest) 복구
    # ------------------------------------------------------------
    log("🚨 3차 시도: API 차단 감지. 전일자 데이터를 이용해 주도주 풀을 복구합니다.")
    try:
        prev_path = os.path.join(OUT_DIR, "recommend_latest.csv")
        if os.path.exists(prev_path):
            df_prev = pd.read_csv(prev_path, dtype={"종목코드": str})
            df_prev["종목코드"] = df_prev["종목코드"].astype(str).str.zfill(6)

            # 어제 파일의 거래대금 단위를 원 단위로 통일
            if "거래대금(억원)" in df_prev.columns:
                df_prev["거래대금(원)"] = pd.to_numeric(df_prev["거래대금(억원)"], errors="coerce").fillna(0) * 1e8
            else:
                df_prev["거래대금(원)"] = pd.to_numeric(df_prev.get("거래대금(원)", 0), errors="coerce").fillna(0)

            # 시장 정보 보강 (price_snapshot 폴백)
            if "시장" not in df_prev.columns or df_prev["시장"].nunique() <= 1:
                _snap_path = os.path.join(OUT_DIR, "price_snapshot_latest.csv")
                if os.path.exists(_snap_path):
                    try:
                        _snap = pd.read_csv(_snap_path, dtype={"종목코드": str}, usecols=["종목코드", "시장"])
                        _snap_map = dict(zip(_snap["종목코드"].str.zfill(6), _snap["시장"]))
                        df_prev["시장"] = df_prev["종목코드"].map(_snap_map).fillna(df_prev.get("시장", "KOSPI"))
                    except Exception:
                        pass

            df_fallback = df_prev[["종목코드", "시장", "거래대금(원)"]].copy()
            _mkt_dist = dict(df_fallback["시장"].value_counts())
            log(f"✅ 전일자 데이터 기반 복구 성공 ({len(df_fallback)}종목, 시장분포: {_mkt_dist})")

            # 어제 기준 거래대금 순으로 정렬 후 상위 N개 반환
            return df_fallback.sort_values("거래대금(원)", ascending=False).head(top_n)

        else:
            raise FileNotFoundError("이전 추천 파일이 없습니다.")

    except Exception as e:
        log(f"❌ 3차 복구(전일자 데이터) 실패: {e}")

    # 4차: 진짜 최후의 보루 - 빈 DataFrame 반환으로 엉뚱한 매매 차단
    log("🚨 모든 데이터 수집 수단 실패. 빈 종목 풀 반환.")
    return pd.DataFrame(columns=["종목코드", "시장", "거래대금(원)"])

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
            cached_map = dict(zip(df["종목코드"], df["종목명"]))
            # sanity check: 종목명이 종목코드와 동일한 비율이 높으면 캐시 무효화
            if cached_map:
                _bad = sum(1 for c, n in cached_map.items() if c == n)
                if _bad / len(cached_map) > 0.3:
                    log(f"⚠️ 종목명 캐시 오염 감지 ({_bad}/{len(cached_map)}건이 코드=이름), 재생성")
                else:
                    return cached_map
        except (pd.errors.ParserError, KeyError, OSError) as e:
            logger.debug(f"종목명 캐시 파싱 실패, 재생성: {e}")

    log("🔄 종목명 매핑 정보 생성 중... (FDR 우선)")
    name_map = {}

    # 2. FDR로 전체 종목명 한방에 가져오기
    try:
        df_fdr = fdr.StockListing("KRX")

        if df_fdr is not None and not df_fdr.empty:
            # FDR 버전별 컬럼명 변형 모두 대응
            _code_candidates = ["Code", "Symbol", "Ticker", "ISU_SRT_CD", "종목코드"]
            _name_candidates = ["Name", "종목명", "ISU_ABBRV"]

            code_col = None
            for c in _code_candidates:
                if c in df_fdr.columns:
                    code_col = c
                    break

            name_col = None
            for c in _name_candidates:
                if c in df_fdr.columns:
                    name_col = c
                    break

            # 일부 FDR 버전은 종목코드가 index에 들어감
            if code_col is None and df_fdr.index.dtype == object:
                sample_idx = str(df_fdr.index[0]).strip()
                if sample_idx.isdigit() and len(sample_idx) == 6:
                    df_fdr = df_fdr.reset_index()
                    df_fdr.rename(columns={df_fdr.columns[0]: "_idx_code"}, inplace=True)
                    code_col = "_idx_code"

            if code_col and name_col:
                codes = df_fdr[code_col].astype(str).str.strip().str.zfill(6)
                names = df_fdr[name_col].astype(str).str.strip()
                name_map.update(dict(zip(codes, names)))
                log(f"✅ FDR 종목명 확보 완료: {len(name_map)}개 (code_col={code_col}, name_col={name_col})")
            else:
                log(f"⚠️ FDR 컬럼 매칭 실패: code_col={code_col}, name_col={name_col}, 실제 컬럼={df_fdr.columns.tolist()[:10]}")
        else:
            log("⚠️ FDR StockListing('KRX') 결과가 비어 있음")
    except Exception as e:
        log(f"⚠️ FDR 종목명 조회 실패: {e}")

    # 2-1. FDR 실패 시 KRX Open API 직접 호출 폴백
    if not name_map:
        log("🔄 FDR 실패 → KRX 직접 조회 폴백 시도...")
        try:
            _krx_url = "http://data.krx.co.kr/comm/bldAttend498/getJsonData.cmd"
            for _mkt_id in ["STK", "KSQ"]:  # KOSPI, KOSDAQ
                _payload = {
                    "bld": "dbms/MDC/STAT/standard/MDCSTAT01901",
                    "mktId": _mkt_id,
                    "share": "1",
                    "csvxls_is498": "false",
                }
                _headers = {"User-Agent": "Mozilla/5.0", "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiStat/default.cmd"}
                resp = requests.post(_krx_url, data=_payload, headers=_headers, timeout=15)
                if resp.ok:
                    _items = resp.json().get("OutBlock_1", [])
                    for item in _items:
                        _c = str(item.get("ISU_SRT_CD", "")).strip().zfill(6)
                        _n = str(item.get("ISU_ABBRV", "")).strip()
                        if _c and _n and _c != _n:
                            name_map[_c] = _n
            if name_map:
                log(f"✅ KRX 직접 조회 성공: {len(name_map)}개")
        except Exception as e:
            log(f"⚠️ KRX 직접 조회 실패: {e}")

    # 3. FDR/KRX 모두 실패 시 pykrx로 보완 (느림)
    if (not name_map) and PYKRX_OK and (stock is not None):
        log("🔄 FDR/KRX 모두 실패 -> pykrx로 개별 조회 시도 (느림)")
        for m in ["KOSPI", "KOSDAQ"]:
            try:
                tickers = safe_ticker_list(d, market=m)
                for t in tickers:
                    code = str(t).zfill(6)
                    if code in name_map:
                        continue
                        
                    nm = safe_ticker_name(t)
                    if nm:
                        name_map[code] = nm
                    else:
                        name_map[code] = code
                    time.sleep(0.001) 
            except Exception:  # [v20.6.4] bare except → typed
                pass

    # 3-1. 전부 실패 시 Naver API 병렬 조회 (최종 폴백)
    if not name_map:
        log("🔄 FDR/KRX/pykrx 모두 실패 → Naver API 병렬 조회 시도...")
        # 코드 목록 확보: 이전 recommend_latest.csv 또는 pykrx ticker_list
        _target_codes = set()
        _prev_path = os.path.join(OUT_DIR, "recommend_latest.csv")
        if os.path.exists(_prev_path):
            try:
                _prev = pd.read_csv(_prev_path, dtype={"종목코드": str}, usecols=["종목코드"])
                _target_codes.update(_prev["종목코드"].astype(str).str.zfill(6).tolist())
            except Exception:
                pass
        if not _target_codes and PYKRX_OK and stock is not None:
            for _m in ["KOSPI", "KOSDAQ"]:
                try:
                    _target_codes.update(str(t).zfill(6) for t in safe_ticker_list(d, market=_m))
                except Exception:
                    pass

        if _target_codes:
            def _fetch_name_naver(code):
                try:
                    r = requests.get(
                        f"https://m.stock.naver.com/api/stock/{code}/basic",
                        timeout=5, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    if r.ok:
                        nm = r.json().get("stockName", "")
                        if nm and nm != code:
                            return code, nm
                except Exception:
                    pass
                return code, None

            with ThreadPoolExecutor(max_workers=20) as pool:
                futures = {pool.submit(_fetch_name_naver, c): c for c in _target_codes}
                for fut in as_completed(futures):
                    code, nm = fut.result()
                    if nm:
                        name_map[code] = nm

            if name_map:
                log(f"✅ Naver API 병렬 조회 성공: {len(name_map)}개 ({len(_target_codes)}건 중)")
            else:
                log("⚠️ Naver API 병렬 조회도 실패")

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

            # ⑥ 스냅샷에도 sanitize 강제 — 0값/이상치 지뢰 방지
            df = sanitize_ohlcv(df)

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

            # [v10.2] 시가/저가/고가도 저장 → 백테스트 체결 현실성 확보
            save_cols = ["종목코드", "종목명", "시장", "종가"]
            for extra in ["시가", "저가", "고가"]:
                if extra in df.columns:
                    save_cols.append(extra)
            frames.append(df[save_cols])
        except Exception as e:
            log(f"⚠️ 가격 스냅샷({m}) 수집 실패: {e}")
            continue

    if not frames:
        # ── FDR 폴백: pykrx 실패 시 FDR StockListing으로 스냅샷 생성 ──
        log(f"🔄 pykrx 스냅샷 실패 → FDR 폴백 시도...")
        try:
            df_fdr = fdr.StockListing("KRX")
            if df_fdr is not None and not df_fdr.empty:
                code_col = "Code" if "Code" in df_fdr.columns else "Symbol"
                market_col = "Market" if "Market" in df_fdr.columns else None

                df_fdr["종목코드"] = df_fdr[code_col].astype(str).str.zfill(6)
                df_fdr["종목명"] = df_fdr["종목코드"].map(name_map).fillna(df_fdr.get("Name", ""))
                df_fdr["시장"] = df_fdr[market_col] if market_col else "KRX"

                col_map = {"Close": "종가", "Open": "시가", "Low": "저가", "High": "고가"}
                df_fdr = df_fdr.rename(columns={k: v for k, v in col_map.items() if k in df_fdr.columns})

                save_cols = ["종목코드", "종목명", "시장", "종가"]
                for extra in ["시가", "저가", "고가"]:
                    if extra in df_fdr.columns:
                        save_cols.append(extra)

                snap_fdr = df_fdr[save_cols].copy()
                snap_fdr["종가"] = pd.to_numeric(snap_fdr["종가"], errors="coerce").fillna(0)
                snap_fdr = snap_fdr[snap_fdr["종가"] > 0]

                if not snap_fdr.empty:
                    frames.append(snap_fdr)
                    log(f"✅ [FDR 폴백] 가격 스냅샷 {len(snap_fdr)}종목 확보")
        except Exception as e:
            log(f"⚠️ FDR 스냅샷 폴백 실패: {e}")

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
        from db_utils import get_db
        db = get_db()
        db.save_snapshot(snap, trade_ymd)
        # ✅ 싱글톤이므로 close() 하지 않음
    except Exception as e:
        log(f"⚠️ 스냅샷 DB 저장 실패: {e}")
    # --------------------------------------------------


# ------------------------------- AI 코멘트 / 스코어 -------------------------------

# [v14 REMOVED → fetch_naver_news_headlines moved to module] (36 lines deleted)

# [v14 REMOVED → analyze_sentiment_llm moved to module] (41 lines deleted)


# [v14 REMOVED → generate_ai_comment moved to module] (59 lines deleted)

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
    except Exception:  # [v20.6.4] bare except → typed
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


# ═══════════════════════════════════════════════════
#  스코어링/상태머신 — scoring_engine.py SSOT import
# ═══════════════════════════════════════════════════
try:
    from scoring_engine import (
        determine_state,
        determine_state_dynamic,
        calculate_ebs_independent,
        calculate_structural_score,
        calculate_timing_score,
        build_global_score,
        _calc_ml_weight,
    )
    log("✅ scoring_engine SSOT import 성공")
except ImportError as _ie:
    log(f"⚠️ scoring_engine import 실패: {_ie}")
    raise ImportError(
        "scoring_engine.py를 찾을 수 없습니다. "
        "collector.py와 같은 디렉토리에 scoring_engine.py가 있는지 확인하세요. "
        f"원본 오류: {_ie}"
    )



# ------------------------------- 텔레그램 (업그레이드) -------------------------------

# [v14 REMOVED → get_naver_theme_tags moved to module] (26 lines deleted)

# [v14 REMOVED → send_telegram_auto moved to module] (102 lines deleted)
# [v20.6.5 MOVED → investor_flow.py] fetch_investor_net_buying (55줄 제거)

# ------------------------------- Trigger Score Calculation -------------------------------

# [v20.6.5 MOVED → trigger_engine.py] calculate_trigger_score + calc_volume_profile_v2 (198줄 제거)

def analyze_ticker(
    t: str, ohlcv_df: pd.DataFrame, top_df: pd.DataFrame, mcap_map: Dict[str, float],
    kospi_set: set, kosdaq_set: set, name_map: Dict[str, str], sector_map: Dict[str, str],
    bench_map: Dict[str, Dict[int, float]],
    inv_maps: Optional[Dict[str, Dict[str, int]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    [v3.2] SRP 분리 — ticker_analyzer.py로 위임.
    
    기존 시그니처 100% 유지 (drop-in replacement).
    내부 로직은 4개 단일 책임 함수로 분리됨:
      1. prepare_ohlcv        → 데이터 정제 + 기본 필터링
      2. calculate_indicators  → 기술적 지표 계산
      3. build_ticker_plan     → 진입/청산/수급
      4. assemble_result       → 메타 병합 + 최종 딕셔너리
    """
    from ticker_analyzer import analyze_ticker_v2
    return analyze_ticker_v2(
        t, ohlcv_df, top_df, mcap_map,
        kospi_set, kosdaq_set, name_map, sector_map,
        bench_map, inv_maps,
    )

# [v22] 켈리 공식 — apply_kelly_calibrated wrapper 위임 (ELITE_SCORE 축 + empirical b_ratio)
def apply_kelly_betting(df: pd.DataFrame, total_capital: int = 10_000_000,
                        out_dir: str = "") -> pd.DataFrame:
    """
    Kelly Criterion 비중 최적화 — v22 SSOT 적용.
    
    [v22 변경]
    - 점수 축 ELITE_SCORE 우선 (랭킹·승률·Kelly 일치)
    - method 자동 동기화 (양방향)
    - b_ratio = min(planned, empirical) — 과대배팅 방지
    - empirical 미가용 시 planned×0.6 보수화
    - 관측 컬럼 4종: KELLY_PLANNED_B / EMPIRICAL_B / FINAL_B / FRACTION
    - 엔진 식별 컬럼: KELLY_ENGINE = "v22_calibrated" | "v13_fallback"
    
    [v22 v2 — inactive route 확장]
    - WAIT/OVERHEAT/EXIT_WARNING + CARRY/BLOCKED도 추천금액 0원 처리
    - CARRY는 기존 보유 관리용 → 신규 추천 베팅 금지가 보수적
    """
    log("💰 [Money Management v22] 켈리 — ELITE_SCORE + empirical b_ratio 적용 중...")
    
    # [v22] 비활성 ROUTE 정의 (한 곳에서 관리)
    INACTIVE_ROUTES = ["WAIT", "OVERHEAT", "EXIT_WARNING", "CARRY", "BLOCKED"]
    
    try:
        from kelly_calibrator import apply_kelly_calibrated
    except ImportError as e:
        log(f"⚠️ [v22] kelly_calibrator import 실패 ({e}) → v13 fallback")
        return _apply_kelly_betting_v13_fallback(df, total_capital, out_dir,
                                                   error=f"import: {e}",
                                                   inactive_routes=INACTIVE_ROUTES)
    
    try:
        # v22 wrapper 호출 (전체 DataFrame 처리 — 벡터화)
        df = apply_kelly_calibrated(
            df,
            out_dir=out_dir or ".",
            total_capital=total_capital,
            method="ELITE_SCORE",   # 명시적 ELITE 우선 (내부에서 score_col과 동기화)
            kelly_multiplier=0.5,
            max_allocation=0.25,
            min_score_threshold=60.0,
        )
    except Exception as e:
        # 런타임 에러만 잡음 (ImportError는 위에서 처리). 버그는 fallback으로 숨기되 흔적 남김.
        log(f"⚠️ [v22] apply_kelly_calibrated 런타임 실패 ({type(e).__name__}: {e}) → v13 fallback")
        return _apply_kelly_betting_v13_fallback(df, total_capital, out_dir,
                                                   error=f"{type(e).__name__}: {e}",
                                                   inactive_routes=INACTIVE_ROUTES)
    
    # ───────── v22 wrapper 성공 경로 ─────────
    
    # [v22 가드] 비활성 ROUTE는 0원 강제 (apply_kelly_calibrated가 score 60+이면 베팅하므로 추가 필터)
    try:
        from collector_config import Route
        _route_strs = [
            *INACTIVE_ROUTES,
            *[str(getattr(Route, r, "")) for r in INACTIVE_ROUTES],
        ]
    except ImportError:
        _route_strs = INACTIVE_ROUTES
    
    inactive_mask = df["ROUTE"].astype(str).isin(_route_strs)
    
    # [v22 v3] 비활성 ROUTE에서 모든 베팅 관련 컬럼 0 강제
    # apply_kelly_calibrated는 score 60+이면 ROUTE 무관 베팅하므로 추가 필터 필수.
    # 관측 컬럼(PLANNED_B/EMPIRICAL_B/FINAL_B)은 진단용이라 0으로 초기화 안 함 — 값은 보존,
    # 단 KELLY_FRACTION만 0으로 두면 "계산은 됐지만 베팅 안 함" 의미 보존.
    if inactive_mask.any():
        for col in ["켈리_수량", "켈리_금액(원)", "추천수량", "추천금액(만원)",
                    "KELLY_FRACTION"]:
            if col in df.columns:
                df.loc[inactive_mask, col] = 0
    
    # [v22] 엔진 식별 컬럼 — CSV에서 어떤 경로 탔는지 확인 가능
    df["KELLY_ENGINE"] = "v22_calibrated"
    df["KELLY_ERROR"] = ""
    
    # 관측 로그
    n_active = int((df.get("KELLY_FRACTION", pd.Series(0, index=df.index)).fillna(0) > 0).sum())
    n_emp = int(df.get("KELLY_EMPIRICAL_B", pd.Series(dtype=float)).notna().sum())
    n_inactive = int(inactive_mask.sum())
    log(f"✅ [v22 calibrated] Kelly 적용: 활성 {n_active}건, "
        f"empirical b 적용 {n_emp}건, 비활성ROUTE 0원 처리 {n_inactive}건")
    
    return df


def _apply_kelly_betting_v13_fallback(df: pd.DataFrame, total_capital: int = 10_000_000,
                                       out_dir: str = "",
                                       error: str = "",
                                       inactive_routes: List[str] = None) -> pd.DataFrame:
    """[v13 fallback] v22 wrapper 실패 시 안전망 — 기존 인라인 로직.
    
    [v22 v2] CSV 스키마 호환: KELLY_PLANNED_B/EMPIRICAL_B/FINAL_B/FRACTION/ENGINE/ERROR 컬럼 보장.
    fallback이어도 대시보드/스키마가 깨지지 않게 NaN으로라도 채움.
    """
    KELLY_MULTIPLIER = 0.5
    MAX_ALLOCATION = 0.25
    
    # 비활성 ROUTE 기본값
    if inactive_routes is None:
        inactive_routes = ["WAIT", "OVERHEAT", "EXIT_WARNING", "CARRY", "BLOCKED"]
    
    _use_cal = False
    try:
        from kelly_calibrator import calibrated_win_rate, kelly_fraction
        if out_dir:
            _use_cal = True
    except ImportError:
        pass
    
    # Route enum value 둘 다 인식
    try:
        from collector_config import Route
        _inactive_set = set(inactive_routes) | set(
            str(getattr(Route, r, "")) for r in inactive_routes
        )
    except ImportError:
        _inactive_set = set(inactive_routes)

    def _calc_row(row):
        try:
            _route = str(row.get("ROUTE", row.get("상태", "")))
            if _route in _inactive_set:
                return 0, 0, 0.0, 0.0   # qty, amt, planned_b, fraction

            # [v22 fallback] ELITE_SCORE 있으면 우선, 없으면 TOTAL_SCORE
            score = float(row.get("ELITE_SCORE", row.get("TOTAL_SCORE", 0)))
            buy = float(row.get("추천매수가", 0))
            stop = float(row.get("손절가", 0))
            target = float(row.get("추천매도가1", 0))

            if buy <= 0 or stop <= 0 or target <= 0:
                return 0, 0, 0.0, 0.0

            risk = buy - stop
            reward = target - buy
            if risk <= 0:
                return 0, 0, 0.0, 0.0

            b = reward / risk

            if _use_cal:
                p = calibrated_win_rate(score, out_dir)
            else:
                p = 0.4 + (max(score, 0) - 60) * 0.01
                p = min(max(p, 0.3), 0.85)

            if score < 60:
                return 0, 0, b, 0.0   # planned_b는 기록

            if _use_cal:
                f_final = kelly_fraction(p, b, KELLY_MULTIPLIER, MAX_ALLOCATION)
            else:
                q = 1 - p
                f = p - (q / b)
                f_safe = f * KELLY_MULTIPLIER
                f_final = min(max(f_safe, 0.0), MAX_ALLOCATION)

            final_amt = int(total_capital * f_final)
            final_qty = int(final_amt / buy)

            return final_qty, final_amt, b, f_final

        except Exception:
            return 0, 0, 0.0, 0.0

    # [v22 v2] expand 결과 4개 컬럼 — 기존 (qty, amt) 유지하면서 관측 컬럼도 함께
    res = df.apply(_calc_row, axis=1, result_type='expand')
    df["켈리_수량"] = res[0].astype(int)
    df["켈리_금액(원)"] = res[1].astype(int)
    
    # [v22 v2] 관측 컬럼 스키마 유지 — fallback에서도 채움
    df["KELLY_PLANNED_B"] = res[2].round(3)
    df["KELLY_EMPIRICAL_B"] = np.nan      # fallback에선 empirical 미사용
    df["KELLY_FINAL_B"] = res[2].round(3) # fallback에서 final = planned (보수화 미적용)
    df["KELLY_FRACTION"] = res[3].round(4)
    df["KELLY_ENGINE"] = "v13_fallback"
    df["KELLY_ERROR"] = (str(error)[:200] if error else "")[:200]

    # [v22 v3] 추천수량/추천금액 stale 방지 — 전체 0으로 초기화 후 양수 Kelly만 채움
    # fallback 진입 전 다른 단계에서 이미 들어있던 값이 있으면 비활성 ROUTE에 잔존 가능 → 차단
    df["추천수량"] = 0
    df["추천금액(만원)"] = 0

    mask = df["켈리_수량"] > 0
    df.loc[mask, "추천수량"] = df.loc[mask, "켈리_수량"]
    df.loc[mask, "추천금액(만원)"] = (df.loc[mask, "켈리_금액(원)"] / 10000).round(1)

    if _use_cal:
        log(f"✅ [v13 fallback] 캘리브레이션 기반 승률 적용됨 (engine=v13_fallback)")
    else:
        log(f"⚠️ [v13 fallback] 캘리브레이션 없음 → 선형 추정 (engine=v13_fallback)")

    return df

# ------------------------------- 메인 실행 -------------------------------

def main(
    trade_date: Optional[str] = None,
    top_n: Optional[int] = None,
    enable_telegram: bool = True,
    tag: Optional[str] = None,
) -> None:
    """
    LDY Collector v20.1 — Modular Pipeline Architecture
    ═══════════════════════════════════════════════════════
    987줄 → ~40줄 오케스트레이터 (로직은 pipeline_*.py에 분리)
    """
    from time_utils import now_kst
    from pipeline_data import ensure_ml_ready, load_reference_data
    from pipeline_analyze import analyze_universe
    from pipeline_score import run_scoring
    from pipeline_news import enrich_news
    from pipeline_calibrate import run_calibration
    from pipeline_finalize import finalize_outputs

    log("🚀 LDY Collector v20.1 (Modular Pipeline) 시작...")

    # [Stage 0] ML 학습 체크
    ensure_ml_ready()

    # [Stage 1] 참조 데이터 로딩
    ctx = load_reference_data(trade_date, top_n)
    ctx.enable_telegram = enable_telegram
    ctx.tag = tag

    # [Stage 2] 종목별 분석
    ctx = analyze_universe(ctx)
    if ctx.df_out is None or ctx.df_out.empty:
        log("⚠️ 분석 결과 없음. 종료.")
        return

    # [Stage 3] 스코어링 + 라우팅
    ctx = run_scoring(ctx)

    # [Stage 4] 뉴스/DART/LLM 분석
    ctx = enrich_news(ctx)

    # [Stage 5] 캘리브레이션 + 캐리오버
    ctx = run_calibration(ctx)

    # [Stage 6] 저장 + 발송 + 검증
    finalize_outputs(ctx)

    log(f">>> Collector 완료: {now_kst().strftime('%Y-%m-%d %H:%M:%S %Z')}")


if __name__ == "__main__":
    main()
