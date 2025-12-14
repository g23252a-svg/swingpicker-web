# collector.py
# -*- coding: utf-8 -*-
"""
LDY Pro Trader Collector v7.5 (Trend Hunter & VBO)
- v7.5: ADX(추세강도) 팩터 추가, Larry Williams VBO 기준가 산출, VWAP 괴리율 분석
- v7.4: 변동성 국면별 ATR Multiplier (1.8~2.5) 동적 적용
- v7.2: 시장 국면(Bull/Bear)에 따른 스코어링 가중치 동적 변화
"""

import os
import time
import math
import pickle
import random
import numpy as np
import pandas as pd
import requests
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ✅ time_utils가 있으므로 import 사용
from time_utils import now_kst, now_utc, KST

try:
    from pykrx import stock
    PYKRX_OK = True
except Exception:
    stock = None
    PYKRX_OK = False

# [보안 설정]
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_ID = os.environ.get("TG_ID")

# ------------------------------- 설정 -------------------------------

LOOKBACK_DAYS = 250          # 과거 데이터 조회 일수
TOP_N = 600                  # 거래대금 상위 N개 종목
MIN_TURNOVER_EOK = 50        # 최소 거래대금 (억원)
MIN_MCAP_EOK = 1000          # 최소 시총 (억원)
RSI_LOW, RSI_HIGH = 45, 65   # RSI 적정 구간
PASS_EBS = 4                 # EBS (룰 기반 스코어) 통과 기준
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data")
UTF8 = "utf-8-sig"

MAX_WORKERS = int(os.environ.get("LDY_WORKERS", "4"))

# [파라미터]
BB_PERIOD, BB_STD = 20, 2
KC_PERIOD, KC_ATR_PERIOD, KC_MULT = 20, 20, 1.5
BONUS_BB_SQUEEZE_SCORE = 3.0
BONUS_BB_SQUEEZE_ENTRY = 4.0

# [가중치 v7.5]
W_RR, W_T1, W_SL, W_NEAR = 0.20, 0.15, 0.10, 0.10
W_MOM, W_TRD, W_LIQ, W_TEC = 0.10, 0.10, 0.15, 0.10

P_OVERHEAT_5D, P_OVERHEAT_10D = 6.0, 6.0
P_RSI_OUT, P_MACD_NEG = 4.0, 4.0
P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 2.0
W_SECTOR = 0.05
P_BIG_SL = 3.0

# ------------------------------- 유틸 -------------------------------

def log(msg: str) -> None:
    print(f"[{now_kst().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def nz_num(s):
    return pd.to_numeric(s, errors="coerce")

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

# ------------------------------- 지표 계산 함수 -------------------------------

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # 0/0 처리
    rsi = rsi.fillna(50)
    return rsi

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_mfi(high, low, close, vol, period=14):
    tp = (high + low + close) / 3
    rmf = tp * vol
    pos = np.where(tp.diff() > 0, rmf, 0)
    neg = np.where(tp.diff() < 0, rmf, 0)
    pos_s = pd.Series(pos, index=close.index).rolling(period).sum()
    neg_s = pd.Series(neg, index=close.index).rolling(period).sum().replace(0, 1)
    return 100 - (100 / (1 + (pos_s / neg_s)))

# ✅ [v7.5] ADX
def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = abs(100 * (pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr))
    
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di).replace(0, 1)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

# ✅ [v7.5] VWAP
def calc_vwap(high, low, close, vol, period=20):
    tp = (high + low + close) / 3
    tp_v = tp * vol
    return tp_v.rolling(period).sum() / vol.rolling(period).sum().replace(0, np.nan)

def round_to_tick(price):
    if price < 2000: t = 1
    elif price < 5000: t = 5
    elif price < 20000: t = 10
    elif price < 50000: t = 50
    elif price < 200000: t = 100
    elif price < 500000: t = 500
    else: t = 1000
    return int(round(price / t) * t)

# ------------------------------- 데이터 수집 Core -------------------------------

def _backoff_sleep(i: int):
    time.sleep(min(2.0, 0.3 * (2 ** i)) * random.uniform(0.8, 1.2))

def _safe_pykrx_call(func, *args, **kwargs):
    if not PYKRX_OK or stock is None: return None
    for i in range(3):
        try:
            return func(*args, **kwargs)
        except:
            _backoff_sleep(i)
    return None

def safe_ohlcv_by_date(start_ymd: str, end_ymd: str, code: str) -> pd.DataFrame:
    # 1. PYKRX
    if PYKRX_OK:
        for i in range(3):
            try:
                df = stock.get_market_ohlcv_by_date(start_ymd, end_ymd, code)
                if df is not None and not df.empty:
                    return df.rename(columns={"날짜": "Date"})
            except: _backoff_sleep(i)
    
    # 2. FDR Fallback
    for i in range(3):
        try:
            s_dash = f"{start_ymd[:4]}-{start_ymd[4:6]}-{start_ymd[6:]}"
            e_dash = f"{end_ymd[:4]}-{end_ymd[4:6]}-{end_ymd[6:]}"
            df = fdr.DataReader(str(code).zfill(6), s_dash, e_dash)
            if df is not None and not df.empty:
                return df.rename(columns={"Open":"시가", "High":"고가", "Low":"저가", "Close":"종가", "Volume":"거래량"})
        except: _backoff_sleep(i)
    
    return pd.DataFrame()

# ------------------------------- 캐싱 시스템 -------------------------------

def load_ohlcv_cache(ymd: str) -> dict:
    path = os.path.join(OUT_DIR, f"ohlcv_cache_{ymd}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f: return pickle.load(f)
        except: pass
    return {}

def save_ohlcv_cache(ymd: str, data: dict):
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, f"ohlcv_cache_{ymd}.pkl")
    try:
        with open(path, "wb") as f: pickle.dump(data, f)
    except: pass

def prepare_ohlcv_data(tickers, start_ymd, end_ymd, trade_ymd):
    ohlcv_map = load_ohlcv_cache(trade_ymd)
    targets = [str(t).zfill(6) for t in tickers if str(t).zfill(6) not in ohlcv_map or ohlcv_map[str(t).zfill(6)].empty]
    
    if targets:
        log(f"🔄 {len(targets)}개 종목 데이터 수집 중...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_code = {executor.submit(safe_ohlcv_by_date, start_ymd, end_ymd, c): c for c in targets}
            for future in tqdm(as_completed(future_to_code), total=len(targets)):
                code = future_to_code[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        ohlcv_map[code] = df
                except: 
                    pass  # 👈 실패하면 저장하지 않고 건너뜀 (다음에 다시 시도하게)
        
        # 유효한 데이터가 있을 때만 저장
        save_ohlcv_cache(trade_ymd, ohlcv_map)
    
    return ohlcv_map

# ------------------------------- 메타 정보 유틸 -------------------------------

def resolve_trade_date(date_str=None):
    if date_str: return date_str
    
    # 최근 10일 중 데이터 있는 날 찾기
    d = now_kst().date()
    if now_kst().hour < 16: d -= timedelta(days=1)
    
    for _ in range(10):
        ymd = d.strftime("%Y%m%d")
        # 간단 체크: 코스피 시총 데이터가 있나?
        check = _safe_pykrx_call(stock.get_market_cap_by_ticker, ymd, market="KOSPI")
        if check is not None and not check.empty:
            return ymd
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")

def build_mcap_map(ymd):
    mm = {}
    if not PYKRX_OK: return mm, ymd
    try:
        for m in ["KOSPI", "KOSDAQ"]:
            df = _safe_pykrx_call(stock.get_market_cap_by_ticker, ymd, market=m)
            if df is not None:
                for c, r in df.iterrows():
                    mm[str(c).zfill(6)] = r["시가총액"] / 1e8
    except: pass
    return mm, ymd

def get_benchmark_returns(ymd):
    res = {"KOSPI": {}, "KOSDAQ": {}}
    try:
        end = datetime.strptime(ymd, "%Y%m%d").date()
        start = end - timedelta(days=300)
        for mkt, sym in [("KOSPI", "KS11"), ("KOSDAQ", "KQ11")]:
            df = fdr.DataReader(sym, start, end)
            if df is not None and not df.empty:
                curr = df["Close"].iloc[-1]
                for d in [20, 60, 120]:
                    if len(df) > d:
                        res[mkt][d] = (curr / df["Close"].iloc[-(d+1)] - 1) * 100
    except: pass
    return res

def pick_top_by_trading_value(ymd, n):
    dfs = []
    for m in ["KOSPI", "KOSDAQ"]:
        df = _safe_pykrx_call(stock.get_market_ohlcv_by_ticker, ymd, market=m)
        if df is not None:
            df = df.reset_index()
            # 컬럼명 처리
            cols = [c for c in df.columns if "티커" in c or "코드" in c]
            if cols:
                df = df.rename(columns={cols[0]: "종목코드"})
                if "거래대금" in df.columns:
                    df = df.rename(columns={"거래대금": "거래대금(원)"})
                df["시장"] = m
                df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
                dfs.append(df[["종목코드", "시장", "거래대금(원)"]])
    
    if not dfs: return pd.DataFrame()
    full = pd.concat(dfs)
    return full.sort_values("거래대금(원)", ascending=False).head(n)

def get_market_sets(ymd):
    kp, kq = set(), set()
    try:
        l1 = _safe_pykrx_call(stock.get_market_ticker_list, ymd, market="KOSPI")
        if l1: kp = set([str(x).zfill(6) for x in l1])
        l2 = _safe_pykrx_call(stock.get_market_ticker_list, ymd, market="KOSDAQ")
        if l2: kq = set([str(x).zfill(6) for x in l2])
    except: pass
    return kp, kq

def get_name_map(ymd):
    ensure_dir(OUT_DIR)
    f = os.path.join(OUT_DIR, f"krx_codes_{ymd}.csv")
    if os.path.exists(f):
        return dict(zip(pd.read_csv(f, dtype=str)["종목코드"], pd.read_csv(f)["종목명"]))
    
    nm = {}
    if PYKRX_OK:
        for m in ["KOSPI", "KOSDAQ"]:
            ts = _safe_pykrx_call(stock.get_market_ticker_list, ymd, market=m) or []
            for t in ts:
                n = stock.get_market_ticker_name(t)
                nm[str(t).zfill(6)] = n
        if nm:
            pd.DataFrame(list(nm.items()), columns=["종목코드", "종목명"]).to_csv(f, index=False, encoding=UTF8)
    return nm

def save_price_snapshot(ymd, name_map):
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        df = _safe_pykrx_call(stock.get_market_ohlcv_by_ticker, ymd, market=m)
        if df is not None:
            df = df.reset_index()
            cols = [c for c in df.columns if "티커" in c or "코드" in c]
            if cols:
                df = df.rename(columns={cols[0]: "종목코드"})
                df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
                df["종목명"] = df["종목코드"].map(name_map).fillna("")
                df["시장"] = m
                frames.append(df[["종목코드", "종목명", "시장", "종가"]])
    
    if frames:
        full = pd.concat(frames)
        ensure_dir(OUT_DIR)
        full.to_csv(os.path.join(OUT_DIR, f"price_snapshot_{ymd}.csv"), index=False, encoding=UTF8)
        full.to_csv(os.path.join(OUT_DIR, "price_snapshot_latest.csv"), index=False, encoding=UTF8)

# ------------------------------- 섹터 -------------------------------

def load_sector_override() -> dict:
    path = os.path.join(OUT_DIR, "sector_override.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str)
            return dict(zip(df["종목코드"], df["업종"]))
        except: pass
    return {}

def build_sector_map() -> dict:
    return load_sector_override()


def classify_big_sector(name, detail):
    t, n = str(detail), str(name)
    if any(x in n for x in ["삼성전자", "SK하이닉스", "DB하이텍"]): return "반도체"
    if any(x in n for x in ["에코프로", "엘앤에프", "LG에너지", "삼성SDI"]): return "2차전지"
    if any(x in n for x in ["현대차", "기아", "현대모비스"]): return "자동차"
    if any(x in n for x in ["NAVER", "카카오"]): return "플랫폼"
    if any(x in n for x in ["셀트리온", "HLB", "알테오젠"]): return "바이오"
    if any(x in n for x in ["한화에어로", "현대로템", "LIG넥스원"]): return "방산"
    if any(x in n for x in ["HD현대일렉", "LS ELECTRIC", "제룡전기"]): return "전력설비"
    
    if "반도체" in t: return "반도체"
    if "전지" in t or "2차" in t: return "2차전지"
    if "자동차" in t: return "자동차"
    if "제약" in t or "바이오" in t: return "바이오"
    if "게임" in t or "소프트웨어" in t: return "게임/SW"
    if "화학" in t: return "화학"
    if "철강" in t or "금속" in t: return "철강/금속"
    if "기계" in t or "조선" in t: return "기계/조선"
    if "금융" in t or "은행" in t: return "금융"
    return "기타"

def add_sector_momentum(df, col):
    if col not in df.columns or "ret_5d_%" not in df.columns:
        return df, pd.Series()
    counts = df[col].value_counts()
    valid = counts[counts >= 3].index
    sub = df[df[col].isin(valid)]
    ranks = sub.groupby(col)["ret_5d_%"].mean().sort_values(ascending=False)
    df["SECTOR_RET_5D"] = df[col].map(ranks)
    return df, ranks

def compute_market_breadth(df):
    res = {"ALL": 0}
    if "Above_MA20" in df.columns:
        res["ALL"] = df["Above_MA20"].mean() * 100
        for m in ["KOSPI", "KOSDAQ"]:
            sub = df[df["시장"] == m]
            if not sub.empty: res[m] = sub["Above_MA20"].mean() * 100
    return res

# ------------------------------- 분석 핵심 (v7.5) -------------------------------

def analyze_ticker(
    t, ohlcv_df, top_df, mcap_map, kospi_set, kosdaq_set, name_map, sector_map, bench_map
):
    code6 = str(t).zfill(6)
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 120: return None
    
    ohlcv = ohlcv_df.tail(LOOKBACK_DAYS).copy()
    c, h, l, v = ohlcv["종가"], ohlcv["고가"], ohlcv["저가"], ohlcv["거래량"]
    last_c = float(c.iloc[-1])

    # Basic
    ma20 = c.rolling(BB_PERIOD).mean()
    ma60 = c.rolling(60).mean()
    
    # BB & Squeeze
    std20 = c.rolling(BB_PERIOD).std()
    bb_upper = ma20 + (BB_STD * std20)
    bb_lower = ma20 - (BB_STD * std20)
    
    atr_series = calc_atr(h, l, c, KC_ATR_PERIOD)
    kc_mid = ema(c, KC_PERIOD)
    kc_upper = kc_mid + (KC_MULT * atr_series)
    kc_lower = kc_mid - (KC_MULT * atr_series)
    
    ttm_series = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    ttm_squeeze = 1 if ttm_series.iloc[-1] else 0
    
    sqz_cnt = 0
    if ttm_squeeze:
        vals = ttm_series.iloc[:-1].values[::-1]
        for val in vals:
            if val: sqz_cnt += 1
            else: break
        sqz_cnt += 1

    # v7.5 ADX & VWAP
    adx_series = calc_adx(h, l, c, 14)
    adx = float(adx_series.iloc[-1]) if not adx_series.empty else 0.0
    
    vwap_s = calc_vwap(h, l, c, v, 20)
    vwap_val = float(vwap_s.iloc[-1]) if not vwap_s.empty else last_c
    vwap_gap = ((last_c / vwap_val) - 1.0) * 100 if vwap_val > 0 else 0.0

    # Indicators
    rsi = float(calc_rsi(c, 14).iloc[-1])
    mfi = float(calc_mfi(h, l, c, v, 14).iloc[-1])
    
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    hist = macd - sig
    
    hist_tail = hist.dropna().tail(5)
    slope = np.polyfit(np.arange(len(hist_tail)), hist_tail.values, 1)[0] if len(hist_tail) >= 2 else 0.0
    slope_pct = (slope / last_c) * 100.0 if last_c > 0 else 0.0
    
    vol_z = float((v / v.rolling(20).mean()).iloc[-1]) if len(v) > 20 else 0.0
    disp = (last_c / float(ma20.iloc[-1]) - 1.0) * 100 if ma20.iloc[-1] else 0.0

    # Returns
    def _ret(d): return (last_c / float(c.iloc[-(d+1)]) - 1.0)*100 if len(c) > d else np.nan
    ret_5, ret_10, ret_20, ret_60, ret_120 = _ret(5), _ret(10), _ret(20), _ret(60), _ret(120)

    # Filter
    tv_row = top_df.loc[top_df["종목코드"] == code6, "거래대금(원)"]
    if tv_row.empty: return None
    tv_eok = float(tv_row.values[0]) / 1e8
    mcap = mcap_map.get(code6, 0)
    
    if mcap_map and mcap <= 0: return None
    if tv_eok < MIN_TURNOVER_EOK: return None

    # Score
    score = 0
    reason = []
    if RSI_LOW <= rsi <= RSI_HIGH: score += 1; reason.append("RSI")
    if slope > 0: score += 1; reason.append("Slope")
    if -1 <= disp <= 5: score += 1; reason.append("Disp")
    if vol_z > 1.2: score += 1; reason.append("Vol")
    if ma20.iloc[-1] > ma60.iloc[-1]: score += 1; reason.append("Arr")
    if mfi > 60: score += 1; reason.append("MFI")
    if hist.iloc[-1] > 0: score += 1; reason.append("MACD")
    
    # v7.5 ADX Bonus
    if adx >= 25 and adx_series.iloc[-1] > adx_series.iloc[-2]: 
        score += 1; reason.append("Trend+")

    # Trading Plan
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else last_c * 0.03
    buy = min(last_c, ma20.iloc[-1] * 1.03) if (ma20.iloc[-1] > 0 and last_c > ma20.iloc[-1]) else last_c
    
    atr_mult = 2.0
    if ttm_squeeze: atr_mult = 1.8
    if (vol_z > 2.5) or (ret_5 > 10.0) or (disp > 5.0): atr_mult = 2.5
    
    stop = buy - (atr_mult * atr_val)
    stop = max(stop, buy * 0.90)
    stop = min(stop, buy * 0.98)
    
    t1 = buy + (buy - stop) * 2.0
    t2 = buy + (buy - stop) * 4.0

    # v7.5 VBO Price
    today_range = float(h.iloc[-1]) - float(l.iloc[-1])
    vbo_price = last_c + (today_range * 0.5)

    buy = round_to_tick(buy)
    stop = round_to_tick(stop)
    t1 = round_to_tick(t1)
    t2 = round_to_tick(t2)
    vbo_price = round_to_tick(vbo_price)

    # Info
    sector = sector_map.get(code6, "기타")
    name = name_map.get(code6, code6)
    m_row = top_df.loc[top_df["종목코드"] == code6, "시장"]
    market = str(m_row.values[0]) if not m_row.empty else ("KOSPI" if code6 in kospi_set else "KOSDAQ")
    
    bench_dict = bench_map.get(market, {})
    rel_20 = ret_20 - bench_dict.get(20, 0) if np.isfinite(ret_20) else np.nan
    rel_60 = ret_60 - bench_dict.get(60, 0) if np.isfinite(ret_60) else np.nan
    rel_120 = ret_120 - bench_dict.get(120, 0) if np.isfinite(ret_120) else np.nan

    return {
        "시장": market, "종목명": name, "종목코드": code6, "업종": sector,
        "종가": int(last_c), "거래대금(억원)": round(tv_eok, 2), "시가총액(억원)": round(mcap, 1),
        "RSI14": round(rsi, 1), "MFI14": round(mfi, 1), "이격도": round(disp, 2),
        
        # ✅ 필수 컬럼 (TRD, Slope 등) 포함 보장
        "MACD_Slope": round(slope, 4),
        "MA20_GAP": round(disp, 2),
        "ADX": round(adx, 1), 
        "VWAP_Gap": round(vwap_gap, 2),
        
        "MACD_Slope_PCT": round(slope_pct, 4), "거래강도": round(vol_z, 2),
        "TTM_SQUEEZE": ttm_squeeze, "TTM_SQUEEZE_CNT": sqz_cnt,
        "Above_MA20": 1 if last_c > float(ma20.iloc[-1]) else 0,
        "ret_5d_%": round(ret_5, 2), "ret_10d_%": round(ret_10, 2),
        "ret_60d_%": round(rel_60, 2), "rel_120d_%": round(rel_120, 2),
        "rel_20d_%": round(rel_20, 2), "rel_60d_%": round(rel_60, 2), "rel_120d_%": round(rel_120, 2),
        "EBS": score, "통과": "★" if score >= PASS_EBS else "",
        "근거": ", ".join(reason),
        "추천매수가": buy, "손절가": stop, "추천매도가1": t1, "추천매도가2": t2,
        "VBO_Price": int(vbo_price),
        "ATR_MULT": atr_mult
    }

# ------------------------------- Score & Route -------------------------------

def cap_q(s, q=90, floor=1.0):
    c = np.nanpercentile(nz_num(s), q)
    return float(max(c, floor)) if np.isfinite(c) else floor

def pct_norm_pos(s, q=90, floor=1.0):
    s = nz_num(s).clip(lower=0)
    return np.clip(s / cap_q(s, q, floor), 0, 1)

def inv_dist_norm(dist, cap):
    return np.clip(1 - (nz_num(dist) / cap), 0, 1)

def detect_regime_row(row):
    adx = float(row.get("ADX", 0))
    slope = float(row.get("MACD_Slope_PCT", 0))
    rel60 = float(row.get("rel_60d_%", 0))
    
    if adx >= 25 and slope > 0: return "① 강한 상승 추세"
    if rel60 > 10 and slope <= 0: return "② 상승 후 조정"
    if adx < 20 and -5 <= rel60 <= 5: return "③ 박스 / 횡보"
    if rel60 <= -10 and slope > 0.05: return "④ 바닥 반등 시도"
    return "⑤ 하락 / 약세"

def route_tag(row):
    r5 = float(row.get("ret_5d_%", 0))
    slope = float(row.get("MACD_Slope_PCT", 0))
    ebs = float(row.get("EBS", 0))
    sqz = int(row.get("TTM_SQUEEZE", 0))
    adx = float(row.get("ADX", 0))
    vbo = float(row.get("VBO_Price", 0))
    close = float(row.get("종가", 0))

    if sqz == 1: return "🔥 SQZ (폭발대기)"
    if vbo > 0 and close > 0 and ((vbo - close)/close*100) < 1.0:
        return "⚡ VBO (돌파임박)"
    if adx >= 25 and slope > 0 and r5 >= 3: return "🔼 BRK (추세돌파)"
    if slope > 0 and r5 > 0: return "🔺 Watch (상승준비)"
    if r5 <= -3 and slope > 0: return "🔁 MR (눌림반등)"
    return "↩️ PULL (관망)"

def generate_ai_comment_v75(row):
    cmts = []
    sqz_cnt = int(row.get("TTM_SQUEEZE_CNT", 0))
    if sqz_cnt >= 1: cmts.append(f"🌪️{sqz_cnt}일 응축")
    
    adx = row.get("ADX", 0)
    if adx >= 30: cmts.append("💪강한 추세")
    elif adx >= 20: cmts.append("↗️추세 시작")
    
    vbo = row.get("VBO_Price", 0)
    close = row.get("종가", 0)
    if vbo > 0 and close > 0:
        gap = ((vbo - close) / close) * 100
        if gap < 2: cmts.append("🔔돌파 임박")

    if row.get("MFI14", 0) >= 65: cmts.append("💰수급 집중")
    if row.get("LDY_SCORE", 0) >= 90: cmts.append("⭐Top Pick")
    
    return " ".join(cmts) if cmts else "특이사항 없음"

def build_global_score(lat, market_temp="🌤 중립"):
    x = lat.copy()
    w = {
        "RR": W_RR, "T1": W_T1, "SL": W_SL, "NEAR": W_NEAR,
        "MOM": W_MOM, "TRD": W_TRD, "LIQ": W_LIQ, "TEC": W_TEC
    }
    
    if "과열" in market_temp:
        w.update({"RR": 0.15, "T1": 0.15, "MOM": 0.20, "TRD": 0.20, "NEAR": 0.05})
    elif "침체" in market_temp:
        w.update({"RR": 0.30, "SL": 0.20, "NEAR": 0.20, "TRD": 0.05, "MOM": 0.05})

    def col_or_zero(df, col):
        return nz_num(df[col]) if col in df.columns else pd.Series(0.0, index=df.index)

    close = nz_num(x["종가"])
    stop = nz_num(x["손절가"])
    t1 = nz_num(x["추천매도가1"])
    entry = nz_num(x["추천매수가"]).replace(0, np.nan)
    turn = nz_num(x["거래대금(억원)"])
    adx = nz_num(x["ADX"]).fillna(0)
    
    # 1. Money Mgmt
    rr_den = (close - stop).replace(0, np.nan)
    rr1 = (t1 - close) / rr_den
    rr_norm = pct_norm_pos(rr1, q=90, floor=1.0).fillna(0)
    t1_room = ((t1 - close) / close * 100)
    t1_norm = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1).fillna(0)
    sl_pct = ((entry - stop) / entry * 100)
    sl_norm = pd.Series(np.exp(-((sl_pct - 7.0) / 3.0) ** 2), index=x.index).fillna(0)
    
    # 2. Timing
    now_gap = ((close - entry).abs() / entry * 100).fillna(0)
    near_norm = inv_dist_norm(now_gap, cap=cap_q(now_gap, q=75, floor=1.0)).fillna(0)

    # 3. Momentum
    slope = nz_num(x["MACD_Slope_PCT"])
    r5 = nz_num(x["ret_5d_%"])
    slope_norm = pct_norm_pos(slope, q=90, floor=0.01).fillna(0)
    mom_mid_norm = pct_norm_pos(nz_num(x["ret_10d_%"]), q=90, floor=1.0).fillna(0)
    mom_norm = np.clip(0.6*slope_norm + 0.4*mom_mid_norm, 0, 1)

    # 4. Trend (v7.5)
    adx_norm = np.clip((adx - 15) / 35, 0, 1).fillna(0)
    rel60_norm = pct_norm_pos(col_or_zero(x, "rel_60d_%"), q=90, floor=1.0).fillna(0)
    vwap_gap = col_or_zero(x, "VWAP_Gap")
    vwap_norm = np.clip((vwap_gap + 2) / 7, 0, 1).fillna(0)
    trd_norm = np.clip(0.4*adx_norm + 0.3*rel60_norm + 0.3*vwap_norm, 0, 1)

    # 5. Liquidity
    if turn.notna().any():
        lo, hi = np.nanpercentile(turn, 30), np.nanpercentile(turn, 90)
        denom = max(hi - lo, 1e-9)
        liq_norm = np.clip((turn - lo) / denom, 0, 1).fillna(0)
        liq_low = (turn < lo).astype(float)
    else:
        liq_norm = 0.0; liq_low = 0.0

    # 6. Technical
    volz = nz_num(x["거래강도"]).fillna(0)
    kairi = nz_num(x["이격도"])
    vol_sweet = (1 - np.minimum((volz - 1).abs() / 3, 1)).clip(0, 1).fillna(0)
    kairi_norm = (1 - np.minimum(kairi.abs() / cap_q(kairi.abs(), 80, 3.0), 1)).clip(0, 1).fillna(0)
    tec_norm = np.clip(0.6 * vol_sweet + 0.4 * kairi_norm, 0, 1).fillna(0)

    # Base Score
    base_score = (
        100 * w["RR"] * rr_norm + 100 * w["T1"] * t1_norm + 100 * w["SL"] * sl_norm +
        100 * w["NEAR"] * near_norm + 100 * w["MOM"] * mom_norm + 100 * w["TRD"] * trd_norm +
        100 * w["LIQ"] * liq_norm + 100 * w["TEC"] * tec_norm
    )

    # Penalty
    pen = pd.Series(0.0, index=x.index)
    pen += P_OVERHEAT_5D * np.clip((r5 - 10) / 10, 0, 1)
    pen += P_RSI_OUT * ((nz_num(x["RSI14"]) < RSI_LOW) | (nz_num(x["RSI14"]) > RSI_HIGH)).astype(float)
    pen += P_MACD_NEG * (slope < 0).astype(float)
    pen += P_NEAR_FAR * np.clip((now_gap - 15) / 15, 0, 1)
    pen += P_LIQ_LOW * liq_low
    pen += P_BIG_SL * (sl_pct > 12).astype(float)

    prelim_score = np.clip(base_score - pen, 0, 100)

    # Sector Bonus
    sector_bonus = 0.0
    if "SECTOR_RET_5D" in x.columns:
        sec_norm = pct_norm_pos(nz_num(x["SECTOR_RET_5D"]), q=85, floor=1.0).fillna(0)
        sector_bonus = 100 * W_SECTOR * sec_norm

    final_score = np.clip(prelim_score + sector_bonus, 0, 100)
    bb_sq = col_or_zero(x, "TTM_SQUEEZE").clip(0, 1)
    final_score = np.clip(final_score + BONUS_BB_SQUEEZE_SCORE * bb_sq, 0, 100)

    # Save Factors
    x["NORM_RR"] = rr_norm.round(2)
    x["NORM_T1"] = t1_norm.round(2)
    x["NORM_SL"] = sl_norm.round(2)
    x["NORM_NEAR"] = near_norm.round(2)
    x["NORM_MOM"] = mom_norm.round(2)
    x["NORM_TRD"] = trd_norm.round(2)
    x["NORM_LIQ"] = liq_norm.round(2)
    x["NORM_TEC"] = tec_norm.round(2)

    x["LDY_SCORE"] = final_score.round(1)
    rank_score = (
        0.8 * (final_score/100) + 0.1 * bb_sq + 
        0.1 * col_or_zero(x, "Above_MA20") - 
        0.1 * np.clip((r5 - 10) / 10, 0, 1)
    )
    x["RANK_SCORE"] = np.clip(rank_score * 100, 0, 100).round(1)
    x["ENTRY_SCORE"] = x["LDY_SCORE"]

    x["ROUTE"] = x.apply(route_tag, axis=1)
    x["REGIME"] = x.apply(detect_regime_row, axis=1)
    x["AI_COMMENT"] = x.apply(lambda r: generate_ai_comment_v75(r), axis=1)

    return x

# ------------------------------- Telegram -------------------------------

def send_telegram_auto(df, trade_ymd, market_summary=""):
    if not TG_TOKEN or not TG_ID: return
    try:
        top5 = df.head(5)
        msg = f"🚀 [LDY v7.5] Trend Hunter ({trade_ymd})\n"
        if market_summary: msg += f"{market_summary}\n"
        msg += "-" * 25 + "\n\n"
        
        for i, row in top5.iterrows():
            msg += f"{i+1}. {row['종목명']} ({row['ROUTE']})\n"
            msg += f"   🌡{row['LDY_SCORE']}점 / ADX {row.get('ADX',0)}\n"
            msg += f"   ⚡VBO: {int(row.get('VBO_Price',0)):,}\n"
            msg += f"   🔵매수: {int(row.get('추천매수가',0)):,}\n\n"
            
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_ID, "text": msg}
        )
    except: pass

# ------------------------------- Main -------------------------------

def main(trade_date=None, top_n=None, enable_telegram=True, tag=None):
    log("🚀 LDY Collector v7.5 시작...")
    
    trade_ymd = resolve_trade_date(trade_date)
    mcap_map, mcap_ymd = build_mcap_map(trade_ymd)
    bench_map = get_benchmark_returns(trade_ymd)
    log(f"📅 기준일: {trade_ymd}")
    
    # 1. Top N
    top_df = pick_top_by_trading_value(trade_ymd, top_n or TOP_N)
    if top_df.empty:
        log("❌ 거래대금 상위 종목 수집 실패")
        return

    # 시총 필터
    if mcap_map:
        original_count = len(top_df)
        top_df["시가총액"] = top_df["종목코드"].map(lambda c: mcap_map.get(c, 0))
        filtered_df = top_df[top_df["시가총액"] >= MIN_MCAP_EOK]
        
        if filtered_df.empty:
            log(f"⚠️ 시총 필터 후 0개됨 (매칭 실패 가능성). 필터 해제하고 {original_count}개로 진행합니다.")
        else:
            top_df = filtered_df
    
    tickers = top_df["종목코드"].tolist()
    if not tickers:
        log("❌ 분석 대상 종목이 없습니다.")
        return

    log(f"📊 분석 대상: {len(tickers)}개")
    
    # 2. OHLCV
    s_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS*2+60)
    full_ohlcv_map = prepare_ohlcv_data(tickers, s_dt.strftime("%Y%m%d"), trade_ymd, trade_ymd)
    
    # 3. Analyze
    kospi_set, kosdaq_set = get_market_sets(trade_ymd)
    name_map = get_name_map(trade_ymd)
    save_price_snapshot(trade_ymd, name_map)
    sector_map = build_sector_map()
    
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = []
        for t in tickers:
            c6 = str(t).zfill(6)
            if c6 in full_ohlcv_map:
                futs.append(ex.submit(analyze_ticker, t, full_ohlcv_map[c6], top_df, mcap_map, kospi_set, kosdaq_set, name_map, sector_map, bench_map))
        
        for f in tqdm(as_completed(futs), total=len(futs), desc="Analyzing"):
            try:
                r = f.result()
                if r: rows.append(r)
            except: pass
            
    df_raw = pd.DataFrame(rows)
    if df_raw.empty:
        log("❌ 분석 결과 없음")
        return

    # 4. Sector & Breadth
    df_raw["업종_대분류"] = df_raw.apply(lambda r: classify_big_sector(r.get("종목명",""), r.get("업종","")), axis=1)
    df_raw, sec_rank = add_sector_momentum(df_raw, "업종_대분류")
    breadth = compute_market_breadth(df_raw)
    
    b_val = breadth.get("ALL", 0)
    mkt_temp = "🌤 중립"
    if b_val >= 65: mkt_temp = "🔥 과열"
    elif b_val <= 35: mkt_temp = "🧊 침체"
    
    log(f"🌡 시장 온도: {mkt_temp} (Breadth: {b_val:.1f}%)")
    
    # 5. Global Score
    df_out = build_global_score(df_raw, market_temp=mkt_temp)
    df_out = df_out.sort_values("LDY_SCORE", ascending=False)
    
    ensure_dir(OUT_DIR)
    out_path = os.path.join(OUT_DIR, "recommend_latest.csv")
    df_out.to_csv(out_path, index=False, encoding=UTF8)
    # 날짜별 백업
    df_out.to_csv(os.path.join(OUT_DIR, f"recommend_{trade_ymd}.csv"), index=False, encoding=UTF8)
    
    log(f"✅ 저장 완료 ({len(df_out)}개): {out_path}")
    
    if enable_telegram:
        send_telegram_auto(df_out, trade_ymd, f"🌡 {mkt_temp} (Breadth {b_val:.0f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str)
    parser.add_argument("--top", type=int)
    parser.add_argument("--no-telegram", action="store_true")
    args = parser.parse_args()
    
    main(trade_date=args.date, top_n=args.top, enable_telegram=not args.no_telegram)
