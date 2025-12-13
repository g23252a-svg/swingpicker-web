"""
LDY Pro Trader Collector v7.4 (Dynamic ATR & Regime Weighting)
- v7.4: 변동성 국면별 ATR Multiplier (1.8~2.5) 동적 적용
- v7.2: 시장 국면(Bull/Bear)에 따른 스코어링 가중치 동적 변화
- v7.1: 상대강도(RS) 20/60/120일 다변화
- v7.0: OHLCV 배치 수집 및 캐싱
...
"""

import os
import io
import time
import math
import pickle  # ✅ [v7.0 추가] 데이터 직렬화/캐싱용
from typing import Dict, Any, Optional, Callable, Tuple, List

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
import random

from time_utils import now_kst, now_utc, KST
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * vol
    pos = np.where(tp.diff() > 0, rmf, 0)
    neg = np.where(tp.diff() < 0, rmf, 0)
    pos_s = pd.Series(pos, index=close.index).rolling(period).sum()
    neg_s = pd.Series(neg, index=close.index).rolling(period).sum().replace(0, 1)
    return 100 - (100 / (1 + (pos_s / neg_s)))

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
    업종(대분류)별 5일 평균 수익률을 계산해서 컬럼으로 주입
    """
    if group_col not in df.columns or "ret_5d_%" not in df.columns:
        df["SECTOR_RET_5D"] = np.nan
        df["SECTOR_RANK"] = np.nan
        return df, pd.Series(dtype=float)

    g = df.groupby(group_col)["ret_5d_%"].mean().sort_values(ascending=False)
    df["SECTOR_RET_5D"] = df[group_col].map(g)
    df["SECTOR_RANK"] = df[group_col].map(g.rank(ascending=False, method="min"))
    return df, g


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

    dated = os.path.join(OUT_DIR, f"price_snapshot_{trade_ymd}.csv")
    latest = os.path.join(OUT_DIR, "price_snapshot_latest.csv")

    snap.to_csv(dated, index=False, encoding=UTF8)
    snap.to_csv(latest, index=False, encoding=UTF8)

    log(f"💾 가격 스냅샷 저장 완료 → {dated}")


# ------------------------------- AI 코멘트 / 스코어 -------------------------------

def generate_ai_comment(mfi: float, rsi: float, slope: float, disp: float, score: float, ttm_squeeze: int = 0, bw_squeeze: int = 0, bb_bw: float = np.nan, squeeze_cnt: int = 0) -> str:
    comment = ""

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
    def _fv(key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
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
    def _fv(key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
        except Exception:
            return default

    r5 = _fv("ret_5d_%", 0.0)
    r10 = _fv("ret_10d_%", 0.0)
    slope = _fv("MACD_Slope_PCT", 0.0)
    if slope == 0.0:
        slope = _fv("MACD_Slope", 0.0)
    ebs = _fv("EBS", 0.0)
    now_pct = _fv("Now%", 999.0)
    rr1 = _fv("RR1", 0.0)
    mfi = _fv("MFI14", 50.0)
    rel60 = _fv("rel_60d_%", 0.0)  # 60일 상대강도(α)
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
        (slope > 0) and
        (now_pct <= 10)       # 엔트리에서 너무 멀지 않음
    )
    if rev:
        return "🔻 REV (역추세 반등)"

    # 2.5) 진성 스퀴즈(TTM) + 모멘텀 플러스 = 🔥 SQZ(폭발대기)
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
    """
    v7.2: 시장 국면(market_temp)에 따른 동적 가중치 적용
    - 🔥 과열: MOM(모멘텀), T1(목표수익) 가중치 상향 (추세 추종)
    - 🧊 침체: RR(손익비), SL(손절폭), NEAR(타점) 가중치 상향 (방어적)
    """
    x = lat.copy()

    # 1. 시장 국면에 따른 가중치 설정 (Dynamic Weighting)
    # 기본값 (중립)
    w = {
        "RR": 0.25, "T1": 0.18, "SL": 0.12, "NEAR": 0.12,
        "MOM": 0.10, "LIQ": 0.13, "TEC": 0.10
    }
    
    if "과열" in market_temp:
        # 공격적: 추세와 거래량을 중시, 타점은 조금 높아도 OK
        w = {
            "RR": 0.15, "T1": 0.20, "SL": 0.10, "NEAR": 0.05,
            "MOM": 0.25, "LIQ": 0.15, "TEC": 0.10
        }
    elif "침체" in market_temp:
        # 방어적: 손익비와 타점 정밀도, 손절폭을 최우선
        w = {
            "RR": 0.30, "T1": 0.10, "SL": 0.20, "NEAR": 0.25,
            "MOM": 0.05, "LIQ": 0.05, "TEC": 0.05
        }

    def col_or_zero(df, col):
        return nz_num(df[col]) if col in df.columns else pd.Series(0.0, index=df.index)

    # 데이터 로드
    close = nz_num(x["종가"])
    entry = nz_num(x["추천매수가"]).replace(0, np.nan)
    stop = nz_num(x["손절가"])
    t1 = nz_num(x["추천매도가1"])
    turn = nz_num(x["거래대금(억원)"])
    rsi = nz_num(x["RSI14"]).fillna(50)
    
    slope = nz_num(x["MACD_Slope_PCT"]) if "MACD_Slope_PCT" in x.columns else nz_num(x["MACD_Slope"])
    slope_floor = 0.01 if "MACD_Slope_PCT" in x.columns else 1.0
        
    volz = nz_num(x["거래강도"]).fillna(0)
    kairi = nz_num(x["이격도"])
    r5 = nz_num(x["ret_5d_%"])
    r10 = nz_num(x["ret_10d_%"])
    ebs = nz_num(x["EBS"]).fillna(0)
    
    rel20 = col_or_zero(x, "rel_20d_%")
    rel60 = col_or_zero(x, "rel_60d_%")
    rel120 = col_or_zero(x, "rel_120d_%")

    # ----- 팩터 정규화 (Normalization) -----
    rr_den = (close - stop).where((close-stop)>0, np.nan)
    rr1 = (t1 - close) / rr_den
    now_gap = ((close - entry).abs() / entry * 100).fillna(0)
    t1_room = ((t1 - close) / close * 100)
    sl_pct = ((entry - stop) / entry * 100)
    
    rr_norm = pct_norm_pos(rr1, q=90, floor=1.0).fillna(0)
    t1_norm = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1).fillna(0)
    
    # 손절폭 점수: 최적 구간(7%)에서 멀어질수록 감점
    opt = 7.0; width = 3.0
    sl_norm = pd.Series(np.exp(-((sl_pct - opt) / width) ** 2), index=x.index).fillna(0)
    near_norm = inv_dist_norm(now_gap, cap=cap_q(now_gap, q=75, floor=1.0)).fillna(0)

    # MOM 팩터
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

    # LIQ / TEC 팩터
    if turn.notna().any():
        lo, hi = np.nanpercentile(turn, 30), np.nanpercentile(turn, 90)
        denom = max(hi - lo, 1e-9)
        liq_norm = np.clip((turn - lo) / denom, 0, 1).fillna(0)
        liq_low = (turn < lo).astype(float)
    else:
        liq_norm = 0.0; liq_low = 0.0

    vol_sweet = (1 - np.minimum((volz - 1).abs() / 3, 1)).clip(0, 1).fillna(0)
    kairi_abs = kairi.abs()
    kairi_norm = (1 - np.minimum(kairi_abs / cap_q(kairi_abs, q=80, floor=3.0), 1)).clip(0, 1).fillna(0)
    tec_norm = np.clip(0.6 * vol_sweet + 0.4 * kairi_norm, 0, 1).fillna(0)

    # ✅ [v7.2 핵심] 동적 가중치 적용
    base_score = (
        100 * w["RR"] * rr_norm +
        100 * w["T1"] * t1_norm +
        100 * w["SL"] * sl_norm +
        100 * w["NEAR"] * near_norm +
        100 * w["MOM"] * mom_norm +
        100 * w["LIQ"] * liq_norm +
        100 * w["TEC"] * tec_norm
    )

    # 패널티 적용
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

    # 섹터 보정
    sector_bonus = 0.0
    if "SECTOR_RET_5D" in x.columns:
        try:
            sec5 = nz_num(x["SECTOR_RET_5D"]).clip(lower=0)
            sec_norm = pct_norm_pos(sec5, q=85, floor=1.0).fillna(0)
            sector_bonus = 100 * W_SECTOR * sec_norm
        except: pass

    final_score = np.clip(prelim_score + sector_bonus, 0, 100)
    
    # Squeeze Bonus
    bb_sq = col_or_zero(x, "BB_SQUEEZE").clip(0, 1)
    final_score = np.clip(final_score + BONUS_BB_SQUEEZE_SCORE * bb_sq, 0, 100)

    # Entry / Rank Score
    raw_entry = (0.3*near_norm + 0.18*rr_norm + 0.12*t1_norm + 0.1*sl_norm + 0.2*mom_norm + 0.1*(1-liq_low))
    entry_score = np.clip(100 * raw_entry + BONUS_BB_SQUEEZE_ENTRY * bb_sq, 0, 100)

    above = col_or_zero(x, "Above_MA20").clip(0, 1)
    entry_s = pd.Series(entry_score, index=x.index)
    final_s = pd.Series(final_score, index=x.index)
    overheat = (0.6 * np.clip((r5 - 8) / 8, 0, 1) + 0.4 * np.clip((r10 - 18) / 18, 0, 1))

    rank_score = (0.55 * (entry_s/100) + 0.30 * (final_s/100) + 0.10 * bb_sq + 0.05 * above - 0.20 * overheat)

    # ✅ [v7.4 추가] 정규화 팩터 저장 (0.0 ~ 1.0) -> 점수 산출 근거 투명화
    x["NORM_RR"] = rr_norm.round(2)     # 손익비 매력도
    x["NORM_T1"] = t1_norm.round(2)     # 목표수익 여력
    x["NORM_SL"] = sl_norm.round(2)     # 손절폭 적절성
    x["NORM_NEAR"] = near_norm.round(2) # 타점 근접도
    x["NORM_MOM"] = mom_norm.round(2)   # 모멘텀 강도
    x["NORM_LIQ"] = liq_norm.round(2)   # 유동성 점수
    x["NORM_TEC"] = tec_norm.round(2)   # 기술적 완성도

    x["RR1"] = rr1
    x["Now%"] = now_gap
    x["LDY_SCORE"] = final_score.round(1)
    x["ENTRY_SCORE"] = entry_score.round(1)
    x["RANK_SCORE"] = np.clip(rank_score * 100, 0, 100).round(1)
    x["ROUTE"] = x.apply(route_tag, axis=1)
    x["REGIME"] = x.apply(detect_regime_row, axis=1)
    x["AI_COMMENT"] = x.apply(
        lambda row: generate_ai_comment(
            row.get("MFI14", 50), row.get("RSI14", 50), row.get("MACD_Slope_PCT", 0),
            row.get("이격도", 0), row.get("LDY_SCORE", 0),
            row.get("TTM_SQUEEZE", 0), row.get("BB_SQUEEZE_BW", 0), row.get("BB_BW", np.nan),
            row.get("TTM_SQUEEZE_CNT", 0)
        ), axis=1
    )
    return x

# ------------------------------- 텔레그램 -------------------------------

# send_telegram_auto 함수 수정 (인자 추가 및 내용 보강)
def send_telegram_auto(df: pd.DataFrame, trade_ymd: str, market_summary: str = "") -> None: # ✅ market_summary 인자 추가
    log("📨 텔레그램 발송 시작...")
    if not TG_TOKEN or not TG_ID:
        log("⚠️ TG_TOKEN / TG_ID 미설정, 발송 생략")
        return

    # ✅ 여기 추가 (텔레그램 출력 안전 포맷)
    def _fmt_int(x):
        try:
            if pd.isna(x): return "N/A"                
            return f"{int(float(x)):,}"
        except: return "N/A"            

    try:
        top5 = df.head(5).reset_index(drop=True)
        trade_date = datetime.strptime(trade_ymd, "%Y%m%d").strftime('%Y-%m-%d')

        # ✅ [v7.4 추가] 시장 요약 정보 상단 표시
        msg = f"🔥 [LDY v7.4] 추천 Top 5 ({trade_date})\n"
        if market_summary:
            msg += f"{market_summary}\n"
        msg += "-" * 30 + "\n\n"

        for i, row in top5.iterrows():
            rank = i + 1
            name = row['종목명']
            code = row['종목코드']
            route = row.get('ROUTE', '전략없음')
            buy = row.get('추천매수가', np.nan)
            score = row.get('LDY_SCORE', 0)
            comment = row.get('AI_COMMENT', '')
            # ✅ 개별 코멘트 길이 제한(권장 120~180)
            if len(comment) > 140:
                comment = comment[:140] + "…"

            msg += f"{rank}. {name} ({code})\n"
            msg += f"   🌡점수: {float(score):.1f}점\n"
            msg += f"   🎯전략: {route}\n"
            msg += f"   💬AI: {comment}\n"

            # ✅ 여기 변경
            msg += f"   🔵매수: {_fmt_int(buy)}\n"
            msg += (
                f"   🔴손절: {_fmt_int(row.get('손절가'))} / "
                f"🟢목표: {_fmt_int(row.get('추천매도가1'))}\n\n"
            )

        # ✅ 최종 msg 길이 제한(텔레그램 4096 대비 여유)
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



# [수정 대상] analyze_ticker 함수 전체를 아래 코드로 교체하세요.

def analyze_ticker(
    t: str,
    ohlcv_df: pd.DataFrame,
    top_df: pd.DataFrame,
    mcap_map: Dict[str, float],
    kospi_set: set,
    kosdaq_set: set,
    name_map: Dict[str, str],
    sector_map: Dict[str, str],
    bench_map: Dict[str, Dict[int, float]],
) -> Optional[Dict[str, Any]]:
    
    code6 = str(t).zfill(6)

    # 데이터 유효성 검사
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 120:
        return None
    
    # 250일치 사용
    ohlcv = ohlcv_df.tail(LOOKBACK_DAYS).copy()

    c = ohlcv["종가"]
    h = ohlcv["고가"]
    l = ohlcv["저가"]
    v = ohlcv["거래량"]
    last_c = float(c.iloc[-1])

    # 이동평균
    ma20 = c.rolling(BB_PERIOD).mean()
    ma60 = c.rolling(60).mean()
    ma120 = c.rolling(120).mean()

    # 볼린저 밴드 & TTM Squeeze
    std20 = c.rolling(BB_PERIOD).std()
    bb_upper = ma20 + (BB_STD * std20)
    bb_lower = ma20 - (BB_STD * std20)
    bb_bw_series = ((bb_upper - bb_lower) / ma20.replace(0, np.nan)) * 100
    bb_bw = float(bb_bw_series.iloc[-1]) if len(bb_bw_series) else np.nan
    bw_squeeze = 1 if (np.isfinite(bb_bw) and bb_bw < BB_SQUEEZE_BW) else 0

    atr_kc_series = calc_atr(h, l, c, KC_ATR_PERIOD)
    kc_mid = ema(c, KC_PERIOD)
    kc_upper = kc_mid + (KC_MULT * atr_kc_series)
    kc_lower = kc_mid - (KC_MULT * atr_kc_series)
    
    # TTM Squeeze 조건: BB가 KC 안에 포함
    ttm_series = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    ttm_last = ttm_series.iloc[-1] if len(ttm_series) else False
    ttm_squeeze = 1 if (bool(ttm_last) and not pd.isna(ttm_last)) else 0
    bb_squeeze = int(ttm_squeeze)

    # ✅ [v7.4 추가] Squeeze 연속 발생 일수 카운팅 (Tracking)
    sqz_cnt = 0
    if ttm_squeeze == 1:
        # 오늘이 스퀴즈라면, 과거로 거슬러 올라가며 연속된 True 개수를 셈
        # iloc[:-1]은 오늘 제외한 과거 데이터, [::-1]은 역순 정렬
        vals = ttm_series.iloc[:-1].values[::-1] 
        temp_cnt = 0
        for val in vals:
            if val:
                temp_cnt += 1
            else:
                break
        sqz_cnt = temp_cnt + 1 # 오늘 포함

    # 지표 계산
    above_ma20 = 1 if (np.isfinite(ma20.iloc[-1]) and last_c > float(ma20.iloc[-1])) else 0
    atr = float(atr_series.iloc[-1]) if 'atr_series' in locals() else float(calc_atr(h,l,c,14).iloc[-1])
    rsi = float(calc_rsi(c, 14).iloc[-1])
    mfi = float(calc_mfi(h, l, c, v, 14).iloc[-1])
    
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    hist = macd - sig
    
    hist_tail = hist.dropna().tail(5)
    if len(hist_tail) >= 2:
        x_idx = np.arange(len(hist_tail), dtype=float)
        slope = float(np.polyfit(x_idx, hist_tail.values.astype(float), 1)[0])
    else:
        slope = 0.0
    slope_pct = (slope / last_c) * 100.0 if last_c > 0 else 0.0

    vol_z_series = v / v.rolling(20).mean().replace(0, np.nan)
    vol_z = float(vol_z_series.iloc[-1]) if len(vol_z_series) else 0.0
    disp = (last_c / float(ma20.iloc[-1]) - 1.0) * 100 if ma20.iloc[-1] else 0.0

    def _ret(days):
        if len(c) >= days + 1:
            return (last_c / float(c.iloc[-(days + 1)]) - 1.0) * 100
        return np.nan

    ret_5 = _ret(5)
    ret_10 = _ret(10)
    ret_20 = _ret(20)
    ret_60 = _ret(60)
    ret_120 = _ret(120)

    # 필터링
    tv_row = top_df.loc[top_df["종목코드"] == code6, "거래대금(원)"]
    if tv_row.empty: return None
    tv_eok = float(tv_row.values[0]) / 1e8
    mcap = get_mcap_eok_from_map(mcap_map, code6)
    
    if mcap_map and mcap <= 0: return None
    if tv_eok < MIN_TURNOVER_EOK: return None

    # 점수 산정 (EBS)
    score = 0
    reason = []
    if RSI_LOW <= rsi <= RSI_HIGH: score += 1; reason.append("RSI적정")
    if slope > 0: score += 1; reason.append("MACD상승")
    if -1 <= disp <= 5: score += 1; reason.append("20선근접")
    if vol_z > 1.2: score += 1; reason.append("거래량↑")
    if ma20.iloc[-1] > ma60.iloc[-1]: score += 1; reason.append("정배열(단)")
    if last_c > ma120.iloc[-1]: score += 1; reason.append("장기추세(120↑)")
    else: score -= 1
    if mfi > 60: score += 1; reason.append("자금유입")
    if hist.iloc[-1] > 0: score += 1; reason.append("MACD>Sig")

    # 매매가 산정
    if np.isnan(atr) or atr <= 0: atr = last_c * 0.03
    buy = min(last_c, ma20.iloc[-1] * 1.03) if (ma20.iloc[-1] > 0 and last_c > ma20.iloc[-1]) else last_c
    
    # 동적 ATR (v7.4)
    atr_mult = 2.0
    if ttm_squeeze == 1 or (np.isfinite(bb_bw) and bb_bw < 12.0):
        atr_mult = 1.8
    is_high_vol = (vol_z > 2.5) or (ret_5 > 10.0) or (disp > 5.0)
    if is_high_vol:
        atr_mult = 2.5

    stop = buy - (atr_mult * atr)
    limit_pct = 0.90 if is_high_vol else 0.93
    stop = max(stop, buy * limit_pct) 
    stop = min(stop, buy * 0.98)
    
    risk = buy - stop
    rr_mult = 2.0 if score >= 8 else (1.5 if score >= 6 else 1.2)
    t1 = buy + risk * rr_mult
    t2 = buy + risk * (rr_mult * 2.0)

    # 틱 보정
    buy = round_to_tick(buy)
    tk = tick_size(buy)
    stop = floor_to_tick(stop)
    if buy - stop < tk: stop = buy - tk
    t1 = ceil_to_tick(t1)
    if t1 <= buy: t1 = buy + tk * 2
    t2 = ceil_to_tick(t2)

    # 메타 정보
    sector = sector_map.get(code6, "기타")
    name = name_map.get(code6, code6)
    m_row = top_df.loc[top_df["종목코드"] == code6, "시장"]
    market = str(m_row.values[0]) if not m_row.empty else ("KOSPI" if code6 in kospi_set else "KOSDAQ")

    bench_dict = bench_map.get(market, {})
    idx_20 = bench_dict.get(20, np.nan)
    idx_60 = bench_dict.get(60, np.nan)
    idx_120 = bench_dict.get(120, np.nan)

    rel_20 = ret_20 - idx_20 if np.isfinite(idx_20) and np.isfinite(ret_20) else np.nan
    rel_60 = ret_60 - idx_60 if np.isfinite(idx_60) and np.isfinite(ret_60) else np.nan
    rel_120 = ret_120 - idx_120 if np.isfinite(idx_120) and np.isfinite(ret_120) else np.nan

    return {
        "시장": market,
        "종목명": name,
        "종목코드": code6,
        "업종": sector,
        "종가": int(last_c),
        "거래대금(억원)": round(tv_eok, 2),
        "시가총액(억원)": round(mcap, 1) if mcap_map else np.nan,
        "RSI14": round(rsi, 1),
        "MFI14": round(mfi, 1),
        "이격도": round(disp, 2),
        "MACD_Hist": round(float(hist.iloc[-1]), 4),
        "MACD_Slope_PCT": round(slope_pct, 4),
        "거래강도": round(vol_z, 2),
        
        "BB_BW": round(bb_bw, 2) if np.isfinite(bb_bw) else np.nan,
        "BB_SQUEEZE_BW": int(bw_squeeze),
        "TTM_SQUEEZE": int(ttm_squeeze),
        "TTM_SQUEEZE_CNT": int(sqz_cnt), # ✅ [v7.4 추가] 스퀴즈 연속 일수
        "BB_SQUEEZE": int(bb_squeeze),
        "Above_MA20": int(above_ma20),

        "ret_5d_%": round(ret_5, 2) if np.isfinite(ret_5) else np.nan,
        "ret_10d_%": round(ret_10, 2) if np.isfinite(ret_10) else np.nan,
        "ret_20d_%": round(ret_20, 2) if np.isfinite(ret_20) else np.nan,
        "ret_60d_%": round(ret_60, 2) if np.isfinite(ret_60) else np.nan,
        "ret_120d_%": round(ret_120, 2) if np.isfinite(ret_120) else np.nan,

        "rel_20d_%": round(rel_20, 2) if np.isfinite(rel_20) else np.nan,
        "rel_60d_%": round(rel_60, 2) if np.isfinite(rel_60) else np.nan,
        "rel_120d_%": round(rel_120, 2) if np.isfinite(rel_120) else np.nan,

        "EBS": int(score),
        "통과": "★" if score >= PASS_EBS else "",
        "근거": ", ".join(reason),
        "추천매수가": buy,
        "손절가": stop,
        "추천매도가1": t1,
        "추천매도가2": t2,
        "ATR_MULT": atr_mult,
    }

    return row

# ------------------------------- 메인 실행 -------------------------------

def main(
    trade_date: Optional[str] = None,
    top_n: Optional[int] = None,
    enable_telegram: bool = True,
    tag: Optional[str] = None,
) -> None:
    log("🚀 LDY Collector v7.0 (Caching Enabled) 시작...")

    # ✅ 여기에 넣기 (resolve_trade_date() 호출 전에!)
    if not PYKRX_OK:
        log("❌ pykrx 미사용 환경에서는 거래대금 TopN/스냅샷 생성이 불가합니다.")
        return

    # 1) 먼저 거래 기준일 결정
    trade_ymd = resolve_trade_date(trade_date)

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
                    bench_ret_60
                )
                if row is not None:
                    rows.append(row)
            except Exception as e:
                err_cnt += 1
                # log(f"⚠️ {code6} 처리 중 오류 발생: {e}")
                continue
    else:
        # 병렬 처리 (CPU 연산 분산)
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
                    bench_ret_60
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
    df_out["벤치_60d_KOSPI_%"] = bench_ret_60.get("KOSPI", np.nan)
    df_out["벤치_60d_KOSDAQ_%"] = bench_ret_60.get("KOSDAQ", np.nan)

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
    
    run_reality_check(OUT_DIR, trade_ymd)
    make_rank_validation_report(OUT_DIR, asof_ymd=trade_ymd)
    
    # main 함수 마지막 부분
    if enable_telegram:
        # 시장 요약 문구 생성
        summary_text = f"🌡 {mkt_temp} (Breadth: {breadth.get('ALL', 0)}%)"
        if "TOP_SECTORS_5D" in df_out.columns:
            # 상위 2개 섹터만 간략히
            try:
                top_sec_str = df_out.iloc[0]["TOP_SECTORS_5D"].split(" / ")[:2]
                summary_text += f"\n🚀 주도: {' '.join(top_sec_str)}"
            except: pass
                
        send_telegram_auto(df_out, trade_ymd, market_summary=summary_text) # ✅ 인자 전달
    else:
        log("✉️ --no-telegram 옵션으로 인해 텔레그램 발송 생략")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LDY Pro Trader Collector v7.4")
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
