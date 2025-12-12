# -*- coding: utf-8 -*-
"""
LDY Pro Trader Collector v6.9.1 (Data Polish Patch)
- Base: v6.6 (안정적 데이터 수집, 업종 분류, 유틸리티)
- New Features:
  1. Dynamic Sector Momentum: 실시간 주도 섹터(5일 평균 등락률) 가산점
  2. Bollinger Bands Squeeze: 변동성 축소(Bandwidth < 10%) 감지 -> 폭발 임박 포착
  3. Market Breadth: 20일 이동평균선 상회 종목 비율로 시장 과열/침체 진단
  4. Telegram: 주도 섹터 및 시장 온도 정보 포함
  5. Price Tick Normalization: 추천매수가/손절/목표가를 KRX 호가단위에 맞춰 자동 라운딩
"""

import os
import io
import time
import math
from typing import Dict, Any, Optional, Callable, Tuple, List

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from pykrx import stock
from tqdm import tqdm
import FinanceDataReader as fdr

from time_utils import now_kst, now_utc, KST

# 가격(호가단위) 유틸
from price_utils import round_to_tick, krx_tick_size

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
OUT_DIR = "data"
UTF8 = "utf-8-sig"

# [가중치 v6.8 조정]
# 점수 스케일이 0~100으로 자연스럽게 퍼지도록 정규화/가중치 재조정
# (기존: 일부 가중치가 합 1.0 미만 + near/liq/rr 정규화가 빡빡해서 상단 점수가 60대에 갇히는 현상)
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = 0.18, 0.14, 0.10, 0.10, 0.14, 0.12, 0.12
W_SECTOR_MOM = 0.10
# [패널티]
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = 6.0, 6.0, 4.0
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 4.0, 2.0
P_BIG_SL = 3.0

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
    roll_down = down.rolling(period).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * vol
    pos = np.where(tp.diff() > 0, rmf, 0)
    neg = np.where(tp.diff() < 0, rmf, 0)
    pos_s = pd.Series(pos, index=close.index).rolling(period).sum()
    neg_s = pd.Series(neg, index=close.index).rolling(period).sum().replace(0, 1)
    return 100 - (100 / (1 + (pos_s / neg_s)))

def calc_bollinger(close: pd.Series, period: int = 20, k: float = 2.0):
    """ [New] 볼린저 밴드 및 밴드폭(Bandwidth) 계산 """
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + (std * k)
    lower = ma - (std * k)
    # Bandwidth: 밴드폭이 좁을수록(Squeeze) 변동성 폭발 임박
    bandwidth = ((upper - lower) / ma) * 100
    return upper, lower, bandwidth

def round_to_tick(price: float) -> int:
    if price < 2000: t = 1
    elif price < 5000: t = 5
    elif price < 20000: t = 10
    elif price < 50000: t = 50
    elif price < 200000: t = 100
    elif price < 500000: t = 500
    else: t = 1000
    return int(round(price / t) * t)

# ------------------------------- 데이터 수집 공통 -------------------------------

def _has_ohlcv_and_mcap(ymd: str) -> bool:
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            o = stock.get_market_ohlcv_by_ticker(ymd, market=m)
            if o is not None and not o.empty and "거래대금" in o.columns and _safe_sum(o["거래대금"]) > 0:
                return True
        except Exception:
            pass
    return False

def find_latest_valid_date(check_fn: Callable[[str], bool], max_back_days: int = 10) -> str:
    now = now_kst()
    d = now.date()
    if now.hour < 18:
        d -= timedelta(days=1)
    
    last_ymd = d.strftime("%Y%m%d")
    for _ in range(max_back_days):
        ymd = d.strftime("%Y%m%d")
        if check_fn(ymd):
            return ymd
        d -= timedelta(days=1)
        last_ymd = d.strftime("%Y%m%d")
    return last_ymd

def resolve_trade_date(force_ymd: Optional[str] = None) -> str:
    if force_ymd:
        try:
            base = datetime.strptime(force_ymd, "%Y%m%d").date()
            def _check(ymd: str) -> bool: return _has_ohlcv_and_mcap(ymd)
            d = base
            for _ in range(10):
                ymd = d.strftime("%Y%m%d")
                if _check(ymd): return ymd
                d -= timedelta(days=1)
            return d.strftime("%Y%m%d")
        except:
            return find_latest_valid_date(_has_ohlcv_and_mcap)
    return find_latest_valid_date(_has_ohlcv_and_mcap)

def build_mcap_map(ref_ymd: Optional[str] = None) -> Tuple[Dict[str, float], str]:
    def _check_mcap(ymd: str) -> bool:
        try:
            df = pd.concat([stock.get_market_cap_by_ticker(ymd, 'KOSPI'), stock.get_market_cap_by_ticker(ymd, 'KOSDAQ')])
            return not df.empty
        except: return False

    use = ref_ymd if ref_ymd and _check_mcap(ref_ymd) else find_latest_valid_date(_check_mcap)
    try:
        df = pd.concat([stock.get_market_cap_by_ticker(use, 'KOSPI'), stock.get_market_cap_by_ticker(use, 'KOSDAQ')])
        df['Code'] = df.index
        return dict(zip(df['Code'].astype(str), df['시가총액'] / 1e8)), use
    except:
        return {}, use

def get_mcap_eok_from_map(mcap_map: Dict[str, float], ticker: str) -> float:
    return float(mcap_map.get(str(ticker).zfill(6), 0))

# ------------------------------- 업종 매핑 (v6.6 유지) -------------------------------

def get_fallback_sector_map() -> Dict[str, str]:
    return {
        "005930": "전기전자", "000660": "전기전자", "373220": "전기전자", "207940": "의약품",
        "005380": "운수장비", "005935": "전기전자", "068270": "의약품", "000270": "운수장비",
        "105560": "금융업", "035420": "서비스업", "035720": "서비스업", "006400": "전기전자"
    }

def get_sector_map_krx() -> Dict[str, str]:
    ensure_dir(OUT_DIR)
    cache_path = os.path.join(OUT_DIR, "sector_map_krx.csv")
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype=str)
            return dict(zip(df["종목코드"].str.zfill(6), df["업종"].fillna("기타")))
        except: pass
        
    try:
        url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
        r = requests.post(url, data={"method": "download", "searchType": "13"}, headers={"User-Agent": "Mozilla/5.0"})
        df = pd.read_html(io.BytesIO(r.content), header=0)[0]
        df = df[["종목코드", "업종"]].copy()
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        df.to_csv(cache_path, index=False, encoding=UTF8)
        return dict(zip(df["종목코드"], df["업종"]))
    except: return {}

def get_sector_map_fdr() -> Dict[str, str]:
    ensure_dir(OUT_DIR)
    cache_path = os.path.join(OUT_DIR, "sector_map_fdr_v2.csv")
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype=str)
            return dict(zip(df["종목코드"].str.zfill(6), df["업종"].fillna("기타")))
        except: pass

    try:
        df = fdr.StockListing("KRX")
        cols = {"Symbol": "종목코드", "Code": "종목코드", "Sector": "업종", "Wics": "업종"}
        df = df.rename(columns=cols)
        if "종목코드" in df.columns and "업종" in df.columns:
            df = df[["종목코드", "업종"]].fillna("기타")
            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
            df.to_csv(cache_path, index=False, encoding=UTF8)
            return dict(zip(df["종목코드"], df["업종"]))
    except: pass
    return {}

def load_sector_override() -> Dict[str, str]:
    path = os.path.join(OUT_DIR, "sector_override.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str)
            return dict(zip(df["종목코드"].str.zfill(6), df["업종"].fillna("기타")))
        except: pass
    return {}

def build_sector_map() -> Dict[str, str]:
    s_map = {}
    s_map.update(get_sector_map_krx())
    
    fdr_map = get_sector_map_fdr()
    for k, v in fdr_map.items():
        if k not in s_map or s_map[k] in ["기타", ""]:
            s_map[k] = v
            
    fallback = get_fallback_sector_map()
    for k, v in fallback.items():
        if k not in s_map: s_map[k] = v
        
    s_map.update(load_sector_override())
    return s_map

def classify_big_sector(name: str, detailed: str) -> str:
    t = (detailed or "").strip()
    if any(k in t for k in ["2차전지", "전지"]) or any(k in name for k in ["에코프로", "엘앤에프", "에너지솔루션"]): return "2차전지"
    if "반도체" in t or any(k in name for k in ["하이닉스", "한미반도체"]): return "반도체"
    if any(k in t for k in ["인터넷", "게임"]) or any(k in name for k in ["NAVER", "카카오", "크래프톤"]): return "인터넷/플랫폼·게임"
    if any(k in t for k in ["자동차", "운수장비"]) or any(k in name for k in ["현대차", "기아"]): return "자동차·모빌리티"
    if any(k in t for k in ["조선", "기계", "방산"]): return "조선·기계·방산"
    if any(k in t for k in ["제약", "바이오", "의료"]): return "바이오·의약품"
    if any(k in t for k in ["금융", "은행", "증권", "보험"]): return "금융"
    if any(k in t for k in ["화장품", "음식료", "유통", "의복"]): return "소비재/유통"
    if any(k in t for k in ["전력", "에너지", "전선"]): return "전력·인프라"
    return "기타"

# ------------------------------- 벤치마크 -------------------------------

def get_index_60d_returns(trade_ymd: str, lookback: int = BENCH_LOOKBACK_DAYS) -> Dict[str, float]:
    try:
        end = datetime.strptime(trade_ymd, "%Y%m%d").date()
    except: return {}
    start = end - timedelta(days=lookback * 2)
    res = {}
    for m, s in [("KOSPI", "KS11"), ("KOSDAQ", "KQ11")]:
        try:
            df = fdr.DataReader(s, start, end)
            close = df["Close"].dropna()
            if len(close) > lookback:
                ret = (close.iloc[-1] / close.iloc[-(lookback + 1)] - 1.0) * 100
                res[m] = round(ret, 2)
        except: pass
    return res

# ------------------------------- 기타 수집 -------------------------------

def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m).reset_index()
            cols = [str(c) for c in df.columns]
            code_col = next((c for c in cols if "티커" in c or "코드" in c), "티커")
            val_col = next((c for c in cols if "거래대금" in c), "거래대금")
            df = df.rename(columns={code_col: "종목코드", val_col: "거래대금(원)"})
            frames.append(df[['종목코드', '거래대금(원)']])
        except: pass
    if not frames: raise RuntimeError("No Data from KRX")
    df_all = pd.concat(frames)
    df_all['종목코드'] = df_all['종목코드'].astype(str).str.zfill(6)
    return df_all.sort_values('거래대금(원)', ascending=False).head(top_n)

def get_market_sets(d: str) -> Tuple[set, set]:
    try:
        return set(stock.get_market_ticker_list(d, market='KOSPI')), set(stock.get_market_ticker_list(d, market='KOSDAQ'))
    except: return set(), set()

def get_name_map_cached(d: str) -> Dict[str, str]:
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, "krx_codes.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str)
            return dict(zip(df['종목코드'], df['종목명']))
        except: pass
    
    rows = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            ts = stock.get_market_ticker_list(d, market=m)
            for t in ts:
                rows.append({"종목코드": str(t).zfill(6), "종목명": stock.get_market_ticker_name(t)})
            time.sleep(0.1)
        except: pass
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding=UTF8)
        return dict(zip(df['종목코드'], df['종목명']))
    return {}

def save_price_snapshot(trade_ymd: str, name_map: Dict[str, str]) -> None:
    ensure_dir(OUT_DIR)
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(trade_ymd, market=m).reset_index()
            # 간단한 컬럼 매핑
            df.columns = [c if "종가" not in str(c) else "종가" for c in df.columns]
            code_col = next((c for c in df.columns if "티커" in str(c) or "코드" in str(c)), None)
            if code_col and "종가" in df.columns:
                df["종목코드"] = df[code_col].astype(str).str.zfill(6)
                df["종목명"] = df["종목코드"].map(name_map).fillna("")
                df["시장"] = m
                frames.append(df[["종목코드", "종목명", "시장", "종가"]])
        except: pass
    
    if frames:
        snap = pd.concat(frames, ignore_index=True)
        snap.to_csv(os.path.join(OUT_DIR, f"price_snapshot_{trade_ymd}.csv"), index=False, encoding=UTF8)
        snap.to_csv(os.path.join(OUT_DIR, "price_snapshot_latest.csv"), index=False, encoding=UTF8)

# ------------------------------- v6.8 핵심 분석 로직 -------------------------------

def detect_regime_row(row: pd.Series) -> str:
    rel60 = float(row.get("rel_60d_%", 0))
    slope = float(row.get("MACD_Slope", 0))
    rsi = float(row.get("RSI14", 50))

    if rel60 > 10 and slope > 0 and 50 <= rsi <= 70: return "① 강한 상승 추세"
    if rel60 > 5 and slope <= 0: return "② 상승 후 조정"
    if -5 <= rel60 <= 5: return "③ 박스 / 중립"
    if rel60 <= -5 and slope > 0: return "④ 바닥 반등 시도"
    return "⑤ 하락 / 약세"

def route_tag(row: pd.Series) -> str:
    """ v6.8: Squeeze 조건 추가 """
    r5 = float(row.get("ret_5d_%", 0))
    slope = float(row.get("MACD_Slope", 0))
    ebs = float(row.get("EBS", 0))
    now_pct = float(row.get("Now%", 99))
    bw = float(row.get("BandWidth", 100))
    rel60 = float(row.get("rel_60d_%", 0))

    # [New] Squeeze: 밴드폭 10% 미만 & 하락세 아님
    if bw < 10 and slope > -0.05 and r5 > -2:
        return "⚡ Squeeze (급변동 임박)"

    if (r5 >= 3) and (slope > 0) and (ebs >= PASS_EBS) and (now_pct <= 10):
        return "🔼 BRK (돌파)"
    
    if (rel60 <= -5.0) and (r5 >= 1.0) and (slope > 0) and (now_pct <= 10):
        return "🔻 REV (역추세 반등)"

    if ((slope > 0) and (r5 > 0)) or ((ebs >= PASS_EBS) and (now_pct <= 8)):
        return "🔺 Watch (상승 준비)"

    return "↩️ PULL (눌림)"

def build_global_score(df: pd.DataFrame) -> pd.DataFrame:
    """v6.8 점수 스케일 교정 버전
    - 증상: LDY_SCORE 상단이 60대에 갇혀 min_score=70 필터가 전부 비는 현상
    - 원인:
      1) 가중치 합이 1.0 미만(=최대점 자체가 낮아짐)
      2) near_norm(진입거리), rr_norm, liq_norm 정규화 기준이 과하게 빡빡함
      3) W_TEC 정의는 있는데 점수에 반영이 안 됨
    - 해결:
      - 가중치 합 1.0로 재정렬 + 기술(TEC) 점수 반영
      - log 유동성 + near/rr/mom 정규화 완화
    """
    x = df.copy()

    # 1) Sector momentum (5일 평균)
    if "업종_대분류" in x.columns and "ret_5d_%" in x.columns:
        x["Sector_Mom_5d"] = x.groupby("업종_대분류")["ret_5d_%"].transform("mean").fillna(0)
    else:
        x["Sector_Mom_5d"] = 0.0

    # 안전한 숫자화
    def _num(col, default=0.0):
        if col not in x.columns:
            return pd.Series(default, index=x.index, dtype=float)
        return nz_num(x[col]).fillna(default)

    rr1 = _num("RR1", 0)
    rsi = _num("RSI14", 50)
    macd_slope = _num("MACD_Slope", 0)
    bw = _num("BandWidth", 100)
    ebs = _num("EBS", 0)
    tv = _num("거래대금(억원)", 0)
    ret5 = _num("ret_5d_%", 0)
    ret10 = _num("ret_10d_%", 0)
    now_gap = _num("Now%", 99)

    # 2) 정규화 (0~1)
    # RR: 실제 RR이 2.0이면 '충분히 좋다'로 보고 2.0 기준 만점
    rr_norm = np.clip(rr1 / 2.0, 0, 1)

    # 목표/손절 여유
    t1_room = _num("추천매도가1", 0)
    close = _num("종가", 0).replace(0, np.nan)
    buy = _num("추천매수가", 0)

    # t1_room(%): (T1-현재가)/현재가
    t1_room_pct = nz_num((t1_room - close) / close * 100).fillna(0)
    # sl_room(%): (현재가-손절)/현재가
    sl = _num("손절가", 0)
    sl_room_pct = nz_num((close - sl) / close * 100).fillna(0)

    # 기존보다 완만하게 (상단 점수가 퍼지도록)
    t1_norm = np.clip(t1_room_pct / 18.0, 0, 1)
    sl_norm = np.clip(sl_room_pct / 12.0, 0, 1)

    # 진입거리: 10% 이내면 점수 유지(기존 5%는 너무 빡빡)
    near_norm = np.clip(1 - (now_gap / 10.0), 0, 1)

    # 개별 모멘텀: 10~12%면 충분히 강함으로 평가
    indiv_mom = np.clip(ret10 / 12.0, 0, 1)

    # 섹터 모멘텀: 3%면 만점(기존 5%는 빡빡)
    sector_mom_norm = np.clip(_num("Sector_Mom_5d", 0) / 3.0, 0, 1)

    # 유동성: log 스케일(500억 이하 구간에서 변별력 확보)
    liq_norm = np.clip(np.log1p(tv) / np.log1p(800), 0, 1)

    # 기술(TEC): EBS + RSI 적정 + MACD 양/음 + Squeeze 보너스
    tech_ebs = np.clip(ebs / 5.0, 0, 1)
    tech_rsi = np.clip(1 - (rsi - 55).abs() / 20.0, 0, 1)   # 55 근처 최고
    tech_macd = (macd_slope > 0).astype(float)
    tech_sq = np.clip((15 - bw) / 15.0, 0, 1)               # bw<=15면 보너스

    tech_norm = np.clip(
        0.4 * tech_ebs + 0.3 * tech_rsi + 0.2 * tech_macd + 0.1 * tech_sq,
        0, 1
    )

    # 3) 종합 점수
    base_score = 100 * (
        W_RR * rr_norm +
        W_T1 * t1_norm +
        W_SL * sl_norm +
        W_NEAR * near_norm +
        W_MOM * indiv_mom +
        W_SECTOR_MOM * sector_mom_norm +
        W_LIQ * liq_norm +
        W_TEC * tech_norm
    )

    # 4) 패널티
    pen = pd.Series(0.0, index=x.index)
    if "P_OVERHEAT_5D" in globals():
        pen += P_OVERHEAT_5D * (ret5 > 15).astype(int)
    if "P_OVERHEAT_10D" in globals():
        pen += P_OVERHEAT_10D * (ret10 > 25).astype(int)
    if "P_MACD_NEG" in globals():
        pen += P_MACD_NEG * (macd_slope < 0).astype(int)
    if "P_RSI_OUT" in globals():
        pen += P_RSI_OUT * ((rsi < 40) | (rsi > 75)).astype(int)
    if "P_BIG_SL" in globals():
        pen += P_BIG_SL * (sl_room_pct > 15).astype(int)

    final_score = np.clip(base_score - pen, 0, 100)

    # 5) 진입 매력도
    entry_score = np.clip(
        40 * near_norm + 30 * rr_norm + 30 * sector_mom_norm, 0, 100
    )

    x["LDY_SCORE"] = final_score.round(1)
    x["ENTRY_SCORE"] = entry_score.round(1)
    x["ROUTE"] = x.apply(route_tag, axis=1)
    x["REGIME"] = x.apply(detect_regime_row, axis=1)
    return x

def analyze_ticker(
    t: str, start_s: str, end_s: str, top_df: pd.DataFrame, 
    mcap_map: Dict, sector_map: Dict, name_map: Dict, bench_ret_60: Dict,
    kospi_set: set
) -> Optional[Dict]:
    code6 = str(t).zfill(6)
    
    ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
    if ohlcv is None or len(ohlcv) < 120: return None
    ohlcv = ohlcv.tail(LOOKBACK_DAYS)
    
    c, h, l, v = ohlcv["종가"], ohlcv["고가"], ohlcv["저가"], ohlcv["거래량"]
    
    ma20 = c.rolling(20).mean()
    atr = calc_atr(h, l, c, 14).iloc[-1]
    rsi = calc_rsi(c, 14).iloc[-1]
    mfi = calc_mfi(h, l, c, v, 14).iloc[-1]
    
    # MACD
    macd = ema(c, 12) - ema(c, 26)
    slope = (macd - ema(macd, 9)).diff().iloc[-1]

    # [New] Bollinger Bands
    ub, lb, bw = calc_bollinger(c)
    last_bw = bw.iloc[-1]

    last_c = c.iloc[-1]
    ret_5 = (last_c / c.iloc[-6] - 1.0) * 100 if len(c) > 5 else 0
    ret_10 = (last_c / c.iloc[-11] - 1.0) * 100 if len(c) > 10 else 0
    
    # Bench
    market = "KOSPI" if t in kospi_set else "KOSDAQ"
    ret_60 = (last_c / c.iloc[-(BENCH_LOOKBACK_DAYS+1)] - 1.0)*100 if len(c) > BENCH_LOOKBACK_DAYS else 0
    idx_60 = bench_ret_60.get(market, 0.0)
    rel_60 = ret_60 - idx_60

    # Meta
    tv_row = top_df.loc[top_df["종목코드"] == code6, "거래대금(원)"]
    if tv_row.empty: return None
    tv_eok = float(tv_row.values[0]) / 1e8
    if tv_eok < MIN_TURNOVER_EOK: return None
    mcap = get_mcap_eok_from_map(mcap_map, code6)
    if mcap < 10: mcap = 0 # 예외처리

    # Strategy Score
    score = 0
    reason = []
    if RSI_LOW <= rsi <= RSI_HIGH: score += 1
    if slope > 0: score += 1
    if last_c > ma20.iloc[-1]: score += 1
    if mfi > 60: score += 1; reason.append("수급↑")
    if last_bw < 10: reason.append("Squeeze") # 밴드폭 축소
    
    # Trading Plan
    # - KRX 호가단위(틱)에 맞춰서 추천가/손절/목표가를 “유효한 호가”로 정규화
    raw_buy = ma20.iloc[-1] if last_c > ma20.iloc[-1] else last_c
    if raw_buy > last_c * 1.05:
        raw_buy = last_c

    raw_stop = raw_buy - (2.0 * atr)
    if raw_stop < raw_buy * 0.90:
        raw_stop = raw_buy * 0.90  # 최대 손절 10% 제한

    rr_ratio = 2.0 if score >= 4 else 1.5
    raw_t1 = raw_buy + (raw_buy - raw_stop) * rr_ratio

    # (정책) buy=가장 가까운 틱 / stop=아래로 / t1=위로
    buy = round_to_tick(raw_buy, method="nearest")
    stop = round_to_tick(raw_stop, method="down")
    t1 = round_to_tick(raw_t1, method="up")

    # 라운딩으로 이상해지는 케이스 방어
    if buy is None:
        buy = int(last_c)
    if stop is None:
        stop = round_to_tick(buy * 0.90, method="down")
    if t1 is None:
        t1 = round_to_tick(buy * 1.10, method="up")

    # stop이 buy 이상으로 올라오는 케이스 방어
    if stop >= buy:
        stop = round_to_tick(buy * 0.95, method="down")

    # 목표가가 buy 이하로 내려오는 케이스 방어
    if t1 <= buy:
        t1 = round_to_tick(buy * 1.05, method="up")
    
    return {
        "종목코드": code6, "종목명": name_map.get(code6, code6),
        "기준일": end_s,
        "업종": sector_map.get(code6, "기타"),
        "종가": int(last_c), "거래대금(억원)": round(tv_eok, 1),
        "RSI14": round(rsi, 1), "MFI14": round(mfi, 1),
        "MACD_Slope": round(slope, 5), "BandWidth": round(last_bw, 1),
        "ret_5d_%": round(ret_5, 2), "ret_10d_%": round(ret_10, 2),
        "ret_60d_%": round(ret_60, 2), "rel_60d_%": round(rel_60, 2),
        "EBS": score, "근거": ",".join(reason),
        "추천매수가": int(buy), "손절가": int(stop), "추천매도가1": int(t1),
        "RR1": round((t1-buy)/(buy-stop), 2) if buy!=stop else 0,
        "Now%": round(abs(last_c-buy)/buy*100, 2),
        "ma20_above": 1 if last_c > ma20.iloc[-1] else 0 
    }

# ------------------------------- 메인 실행 -------------------------------

def send_telegram_v68(df: pd.DataFrame, ymd: str, market_heat: float, hot_sectors: str) -> None:
    if not TG_TOKEN or not TG_ID:
        log("⚠️ Telegram Token/ID 미설정")
        return

    top5 = df.head(5).reset_index(drop=True)
    msg = f"🚀 [LDY v6.9.1] Smart Pick ({ymd})\n"
    msg += f"🌡 시장온도: {market_heat:.0f}% (20선 상회)\n"
    msg += f"🔥 주도섹터: {hot_sectors}\n"
    msg += "-" * 30 + "\n\n"
    
    for i, row in top5.iterrows():
        icon = "⚡" if "Squeeze" in str(row["ROUTE"]) else "💎"
        msg += f"{i+1}. {row['종목명']} ({row['종목코드']}) {icon}\n"
        msg += f"   📊점수: {row['LDY_SCORE']} (섹터:{row['업종_대분류']})\n"
        msg += f"   🎯전략: {row['ROUTE']}\n"
        msg += f"   ⚡BandWidth: {row['BandWidth']}%\n"
        msg += f"   🔵매수: {row['추천매수가']:,} / 🔴손절: {row['손절가']:,}\n\n"
        
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", 
                      data={"chat_id": TG_ID, "text": msg})
        log("🚀 텔레그램 발송 완료")
    except Exception as e:
        log(f"⚠️ 텔레그램 발송 실패: {e}")

def main(trade_date: Optional[str] = None, top_n: Optional[int] = None, 
         enable_telegram: bool = True, tag: Optional[str] = None) -> None:
    log("🚀 LDY Collector v6.8 (Full Integrated) 시작...")
    
    # 1. 날짜 및 기초 데이터
    ymd = resolve_trade_date(trade_date)
    log(f"📅 기준일: {ymd}")
    
    mcap_map, mcap_ymd = build_mcap_map(ymd)
    bench_ret_60 = get_index_60d_returns(ymd)
    
    # 2. 종목 선정
    top_df = pick_top_by_trading_value(ymd, top_n or TOP_N)
    
    # 시총 필터
    if mcap_map:
        top_df["시가총액"] = top_df["종목코드"].map(lambda c: get_mcap_eok_from_map(mcap_map, c))
        top_df = top_df[top_df["시가총액"] >= MIN_MCAP_EOK].copy()
    
    tickers = top_df["종목코드"].tolist()
    log(f"📊 분석 대상: {len(tickers)}개 종목 (거래대금/시총 필터)")
    
    # 3. 메타 데이터 준비
    kospi_set, kosdaq_set = get_market_sets(ymd)
    name_map = get_name_map_cached(ymd)
    save_price_snapshot(ymd, name_map)
    sector_map = build_sector_map()
    
    # 4. 개별 종목 분석
    start_dt = datetime.strptime(ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS*2)
    start_s = start_dt.strftime("%Y%m%d")
    
    rows = []
    for t in tqdm(tickers, desc="Analyzing"):
        try:
            r = analyze_ticker(t, start_s, ymd, top_df, mcap_map, sector_map, name_map, bench_ret_60, kospi_set)
            if r:
                r["업종_대분류"] = classify_big_sector(r["종목명"], r["업종"])
                rows.append(r)
        except Exception:
            continue
            
    if not rows: raise RuntimeError("분석 결과 없음")
    df = pd.DataFrame(rows)
    
    # 5. 시장 분석 (Market Breadth)
    market_above_ma20 = df["ma20_above"].mean() * 100
    log(f"🌡️ 시장 온도(20일선 상회 비율): {market_above_ma20:.1f}%")
    
    # 6. 스코어링 및 섹터 모멘텀
    df = build_global_score(df)
    
    # 주도 섹터 (Top 3)
    top_sectors = df.groupby("업종_대분류")["ret_5d_%"].mean().sort_values(ascending=False).head(3)
    hot_sector_str = ", ".join([f"{k}({v:.1f}%)" for k,v in top_sectors.items()])
    log(f"🔥 현재 주도 섹터 Top3: {hot_sector_str}")
    
    # ------------------------------------------------------------------
    # 7. 정렬 및 저장 (점수순)
    df = df.sort_values(["LDY_SCORE", "ENTRY_SCORE"], ascending=[False, False])
    
    # (1) 기록 보관용 파일 저장 (기존 코드)
    date_tag = now_kst().strftime("%Y%m%d")
    suffix = f"_{tag}" if tag else ""
    out_path = os.path.join(OUT_DIR, f"recommend_{ymd}{suffix}_v6.8.csv")
    df.to_csv(out_path, index=False, encoding=UTF8)
    log(f"💾 [기록용] 저장 완료: {out_path}")

    # (2) [추가됨] 시스템 연동용 고정 파일명 저장 (Github Action/Dashboard용)
    latest_path = os.path.join(OUT_DIR, "recommend_latest.csv")
    df.to_csv(latest_path, index=False, encoding=UTF8)
    log(f"💾 [시스템용] 저장 완료: {latest_path}")
    # ------------------------------------------------------------------
    
    # 8. 텔레그램
    if enable_telegram:
        send_telegram_v68(df, ymd, market_above_ma20, hot_sector_str)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="YYYYMMDD")
    parser.add_argument("--top", type=int)
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--tag", type=str)
    args = parser.parse_args()
    
    main(args.date, args.top, not args.no_telegram, args.tag)
