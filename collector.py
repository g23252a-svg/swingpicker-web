# -*- coding: utf-8 -*-
"""
LDY Pro Trader Collector v6.5
- 업종 매핑: KIND(KRX) + FDR + Fallback + Override(사용자 CSV) 병합
- ROUTE: BRK / Watch / MR / PULL 다단계 분류 유지
- 5일/10일 수익률 + 60일 지수/상대강도 계산
- AI Narrative, Telegram 유지
- 안정성/성능, 스코어링, 구조, 출력 메타데이터 개선
"""

import os
import io
import time
import math
from typing import Dict, Any, Optional, Callable, Tuple, List

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from pykrx import stock
from tqdm import tqdm
import FinanceDataReader as fdr

# 👇 추가
from time_utils import now_kst, now_utc, KST

def create_user(uid: str, email: str) -> dict:
    return {
        "user_id": uid,
        "email": email,
        "created_at": now_utc(),      # DB/CSV에는 UTC로 저장
        "last_login_at": now_utc(),
    }

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

# [가중치]
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = 0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = 6.0, 6.0, 4.0
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 4.0, 2.0

# ------------------------------- 유틸 -------------------------------
# ✅ 새 log 함수
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
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(period).mean() / down.replace(0, np.nan).rolling(period).mean()
    return 100 - 100 / (1 + rs)

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

# ------------------------------- 거래일/시총 -------------------------------
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
    """
    공통 날짜 탐색 유틸:
    - 오늘 18시 이전이면 전일 기준
    - check_fn(YYYYMMDD)가 True인 가장 최근 날짜 반환
    """
    d = datetime.now(KST).date()
    if datetime.now(KST).hour < 18:
        d -= timedelta(days=1)

    last_ymd = d.strftime("%Y%m%d")
    for _ in range(max_back_days):
        ymd = d.strftime("%Y%m%d")
        if check_fn(ymd):
            return ymd
        d -= timedelta(days=1)
        last_ymd = d.strftime("%Y%m%d")
    return last_ymd

def resolve_trade_date() -> str:
    return find_latest_valid_date(_has_ohlcv_and_mcap, max_back_days=10)

def build_mcap_map() -> Tuple[Dict[str, float], str]:
    def _check_mcap(ymd: str) -> bool:
        try:
            df = pd.concat([
                stock.get_market_cap_by_ticker(ymd, market='KOSPI'),
                stock.get_market_cap_by_ticker(ymd, market='KOSDAQ')
            ])
            return not df.empty
        except Exception:
            return False

    use = find_latest_valid_date(_check_mcap, max_back_days=10)
    try:
        df = pd.concat([
            stock.get_market_cap_by_ticker(use, market='KOSPI'),
            stock.get_market_cap_by_ticker(use, market='KOSDAQ')
        ])
        if df.empty:
            return {}, use
        df['Code'] = df.index
        return dict(zip(df['Code'].astype(str), df['시가총액'] / 1e8)), use
    except Exception as e:
        log(f"⚠️ 시가총액 맵 생성 실패({use}): {e}")
        return {}, use

def get_mcap_eok_from_map(mcap_map: Dict[str, float], ticker: str) -> float:
    return float(mcap_map.get(str(ticker).zfill(6), 0))

# ------------------------------- 업종 맵핑 -------------------------------
def get_fallback_sector_map() -> Dict[str, str]:
    # 기존에 쓰던 하드코딩 맵 (메이저들 커버)
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
    KIND(KRX)에서 공식 업종 CSV를 받아오는 맵.
    data/sector_map_krx.csv 로 캐시.
    """
    ensure_dir(OUT_DIR)
    cache_path = os.path.join(OUT_DIR, "sector_map_krx.csv")

    # 1) 캐시 우선
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype=str)
            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
            df["업종"] = df["업종"].fillna("기타")
            log(f"📁 KIND 업종 캐시 로드 성공: {len(df)} rows")
            return dict(zip(df["종목코드"], df["업종"]))
        except Exception as e:
            log(f"⚠️ KIND 업종 캐시 로드 실패. 재다운로드 시도: {e}")

    # 2) KRX 공식 업종 CSV 다운로드
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), encoding="euc-kr")

        # KIND 형식 기준: 종목코드, 기업명, 업종, ...
        if "종목코드" not in df.columns or "업종" not in df.columns:
            log(f"⚠️ KIND CSV 컬럼 이상: {df.columns.tolist()}")
            return {}

        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        df["업종"] = df["업종"].replace("", np.nan).fillna("기타")

        df_out = df[["종목코드", "업종"]]
        df_out.to_csv(cache_path, index=False, encoding=UTF8)

        log(f"✅ KIND 업종 다운로드 완료 ({len(df_out)} rows)")
        return dict(zip(df_out["종목코드"], df_out["업종"]))

    except Exception as e:
        log(f"❌ KIND 업종 다운로드 실패: {e}")
        return {}

def get_sector_map_fdr() -> Dict[str, str]:
    """
    FinanceDataReader.StockListing('KRX') 기반 섹터/업종 맵.
    KIND에서 안 나오는 애들 보완용.
    """
    ensure_dir(OUT_DIR)
    cache_path = os.path.join(OUT_DIR, "sector_map_fdr.csv")

    # 1) 캐시 우선
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype=str)
            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
            df["업종"] = df["업종"].fillna("기타")
            log(f"📁 FDR 업종 캐시 로드 성공: {len(df)} rows")
            return dict(zip(df["종목코드"], df["업종"]))
        except Exception as e:
            log(f"⚠️ FDR 업종 캐시 로드 실패. 재생성 시도: {e}")

    # 2) FDR에서 전체 KRX 리스트
    try:
        df = fdr.StockListing("KRX")
        code_col = "Symbol" if "Symbol" in df.columns else (
            "Code" if "Code" in df.columns else None
        )
        if code_col is None:
            raise RuntimeError(f"Code column not found: {df.columns.tolist()}")

        df[code_col] = df[code_col].astype(str).str.zfill(6)

        # 섹터 후보 컬럼
        cand_cols = ["업종", "Sector", "Wics", "Industry"]
        sector_col = None
        for c in cand_cols:
            if c in df.columns:
                sector_col = c
                break
        if sector_col is None:
            raise RuntimeError(f"No sector column in FDR: {df.columns.tolist()}")

        df_out = df[[code_col, sector_col]].rename(
            columns={code_col: "종목코드", sector_col: "업종"}
        )
        df_out["업종"] = df_out["업종"].replace("", np.nan).fillna("기타")

        df_out.to_csv(cache_path, index=False, encoding=UTF8)
        log(f"✅ FDR 업종 생성 및 캐시 저장: {len(df_out)} rows")
        return dict(zip(df_out["종목코드"], df_out["업종"]))

    except Exception as e:
        log(f"❌ FDR 업종 생성 실패: {e}")
        return {}

def load_sector_override() -> Dict[str, str]:
    """
    사용자가 직접 관리하는 오버라이드 테이블.
    파일: data/sector_override.csv
    컬럼: 종목코드, 업종
    """
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
    """
    최종 업종 맵:
    1) FDR
    2) KIND
    3) fallback 하드맵
    4) override (최우선)
    """
    # 1) FDR
    fdr_map = get_sector_map_fdr()

    # 2) KIND
    kind_map = get_sector_map_krx()

    # 3) fallback
    fallback = get_fallback_sector_map()

    # 4) override
    override = load_sector_override()

    sector_map: Dict[str, str] = {}

    # FDR 우선 세팅
    sector_map.update(fdr_map)

    # KIND로 덮어쓰기 (공식 업종이 더 직관적인 경우 많음)
    sector_map.update(kind_map)

    # fallback은 비어있는 코드만 채우기
    for code, sec in fallback.items():
        sector_map.setdefault(code, sec)

    # override는 최종 우선순위
    sector_map.update(override)

    log(f"ℹ️ 최종 업종 맵 크기: {len(sector_map)}개 (FDR+KIND+fallback+override)")
    return sector_map

# ------------------------------- 벤치마크 (지수 60일 수익률) -------------------------------
def get_index_60d_returns(trade_ymd: str, lookback: int = BENCH_LOOKBACK_DAYS) -> Dict[str, float]:
    """
    KOSPI(KS11), KOSDAQ(KQ11) 60일 수익률 계산.
    - trade_ymd 기준으로 lookback+여유일 만큼 과거부터 가져와서
      마지막 종가 / (lookback+1 번째 종가) - 1 로 계산.
    """
    try:
        end = datetime.strptime(trade_ymd, "%Y%m%d").date()
    except Exception:
        return {}

    start = end - timedelta(days=lookback * 2)  # 비영업일 감안 여유

    res: Dict[str, float] = {}
    index_map = {
        "KOSPI": "KS11",
        "KOSDAQ": "KQ11",
    }

    for market, symbol in index_map.items():
        try:
            df = fdr.DataReader(symbol, start, end)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            close = df["Close"].dropna()
            if len(close) <= lookback:
                continue
            last = float(close.iloc[-1])
            base = float(close.iloc[-(lookback + 1)])
            if base <= 0:
                continue
            ret = (last / base - 1.0) * 100
            res[market] = round(ret, 2)
        except Exception as e:
            log(f"⚠️ 벤치마크({market}, {symbol}) 60일 수익률 계산 실패: {e}")
            continue

    return res

# ------------------------------- 기타 수집 로직 -------------------------------
def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m).reset_index()
            df.columns = [
                '종목코드' if ('티커' in str(c) or '코드' in str(c)) else c
                for c in df.columns
            ]
            df.columns = ['거래대금(원)' if c == '거래대금' else c for c in df.columns]
            frames.append(df[['종목코드', '거래대금(원)']])
        except Exception:
            pass
    if not frames:
        raise RuntimeError("No Data from KRX (거래대금)")
    df_all = pd.concat(frames)
    df_all['종목코드'] = df_all['종목코드'].astype(str).str.zfill(6)
    return df_all.sort_values('거래대금(원)', ascending=False).head(top_n)

def get_market_sets(d: str) -> Tuple[set, set]:
    try:
        return set(stock.get_market_ticker_list(d, market='KOSPI')), set(stock.get_market_ticker_list(d, market='KOSDAQ'))
    except Exception:
        return set(), set()

def get_name_map_cached(d: str) -> Dict[str, str]:
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, "krx_codes.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str)
            return dict(zip(df['종목코드'], df['종목명']))
        except Exception:
            pass

    rows: List[Dict[str, str]] = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            tickers = stock.get_market_ticker_list(d, market=m)
            for t in tickers:
                rows.append({
                    '종목코드': str(t).zfill(6),
                    '종목명': stock.get_market_ticker_name(t)
                })
                time.sleep(0.001)
        except Exception:
            pass
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding=UTF8)
        return dict(zip(df['종목코드'], df['종목명']))
    return {}

# ------------------------------- AI 코멘트 / 스코어 -------------------------------
def generate_ai_comment(mfi: float, rsi: float, slope: float, disp: float, score: float) -> str:
    comment = ""

    if mfi >= 70:
        comment += "💰 외국인/기관의 강력한 수급이 집중되고 있습니다. "
    elif mfi >= 60:
        comment += "💸 자금 유입이 꾸준히 이어지고 있습니다. "

    if slope > 100:
        comment += "🚀 상승 에너지가 폭발적으로 증가하는 중입니다. "
    elif slope > 0:
        comment += "📈 상승 추세가 견고하게 유지되고 있습니다. "

    if -2 <= disp <= 2:
        comment += "✅ 20일선 부근의 안전한 눌림목 구간입니다."
    elif disp > 5:
        comment += "⚠️ 단기 급등으로 인한 조정 가능성을 염두에 두세요."
    elif disp < -5:
        comment += "📉 과매도 구간으로 기술적 반등이 기대됩니다."

    if score >= 90:
        comment += " (강력 매수 추천)"
    elif score >= 80:
        comment += " (매수 유효)"

    return comment if comment else "특이사항 없음. 기술적 지표를 참고하세요."

# --- Scoring Logic ---
def cap_q(s: pd.Series, q: int = 90, floor: float = 1.0) -> float:
    c = np.nanpercentile(nz_num(s), q)
    return float(max(c, floor)) if np.isfinite(c) else floor

def pct_norm_pos(s: pd.Series, q: int = 90, floor: float = 1.0) -> pd.Series:
    s = nz_num(s).clip(lower=0)
    return np.clip(s / cap_q(s, q, floor), 0, 1)

def inv_dist_norm(dist: pd.Series, cap: float) -> pd.Series:
    return np.clip(1 - (nz_num(dist) / cap), 0, 1)

def route_tag(row: pd.Series) -> str:
    """
    ROUTE 분류:
    - 🔼 BRK (돌파): 최근 모멘텀 +, MACD 기울기 +, EBS 통과, 현재가-진입 괴리 과하지 않음
    - 🔺 Watch (상승 준비): 모멘텀/기울기 양호 or EBS 좋고 진입과 근접
    - 🔁 MR (반전): 단기 크게 밀리고, 기울기도 음수
    - ↩️ PULL (눌림): 그 외
    """
    def _fv(key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
        except Exception:
            return default

    r5 = _fv("ret_5d_%", 0.0)
    r10 = _fv("ret_10d_%", 0.0)
    slope = _fv("MACD_Slope", 0.0)
    ebs = _fv("EBS", 0.0)
    now_pct = _fv("Now%", 999.0)
    rr1 = _fv("RR1", 0.0)
    mfi = _fv("MFI14", 50.0)

    # 강한 돌파: 단기/중기 모멘텀 +, 수급, EBS, 위치 모두 양호
    strong_break = (
        (r5 >= 3) and (r10 >= 5) and (slope > 0) and (ebs >= PASS_EBS)
        and (now_pct <= 10) and (mfi >= 55)
    )
    if strong_break and rr1 and not np.isnan(rr1) and rr1 < 0.6:
        strong_break = False

    if strong_break:
        return "🔼 BRK (돌파)"

    # 관찰/상승 준비: 기울기 +, 단기 모멘텀 + 또는 EBS + 진입가 근접
    watch = ((slope > 0) and (r5 > 0)) or ((ebs >= PASS_EBS) and (now_pct <= 8))
    if watch:
        if r5 >= 1.5 and slope > 0:
            return "🔺 Watch (관찰·돌파예상)"
        return "🔺 Watch (상승 준비)"

    # 반전 후보: 단기 낙폭 크고, 모멘텀도 아직 음
    if r5 <= -2 and slope < 0:
        return "🔁 MR (반전)"

    # 기본은 눌림
    return "↩️ PULL (눌림)"

def build_global_score(lat: pd.DataFrame) -> pd.DataFrame:
    x = lat.copy()
    close = nz_num(x["종가"])
    entry = nz_num(x["추천매수가"])
    stop = nz_num(x["손절가"])
    t1 = nz_num(x["추천매도가1"])
    turn = nz_num(x["거래대금(억원)"])
    rsi = nz_num(x["RSI14"])
    slope = nz_num(x["MACD_Slope"])
    volz = nz_num(x["거래강도"])
    kairi = nz_num(x["이격도"])
    r5 = nz_num(x["ret_5d_%"])
    r10 = nz_num(x["ret_10d_%"])
    ebs = nz_num(x["EBS"]).fillna(0)
    rel60 = nz_num(x.get("rel_60d_%", 0.0))  # 6.5: 지수 대비 60일 상대강도

    # RR, 진입 괴리, 손절/목표 여유
    rr_den = (entry - stop)
    rr_den = rr_den.where(rr_den > 0, np.nan)  # 0 이하 RR은 무의미 → NaN
    rr1 = (t1 - entry) / rr_den
    now_gap = ((close - entry).abs() / entry * 100)
    t1_room = ((t1 - close) / close * 100)
    sl_room = ((close - stop) / close * 100)

    rr_norm = pct_norm_pos(rr1, q=90, floor=1.0).fillna(0)
    t1_norm = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1).fillna(0)
    sl_norm = np.clip(sl_room / cap_q(sl_room, q=90, floor=3.0), 0, 1).fillna(0)
    near_norm = inv_dist_norm(now_gap, cap=cap_q(now_gap, q=75, floor=1.0)).fillna(0)

    # 모멘텀/에너지
    ers_bits = (
        (ebs >= PASS_EBS).astype(int)
        + (slope > 0).astype(int)
        + ((rsi >= RSI_LOW) & (rsi <= RSI_HIGH)).astype(int)
    )
    ers_norm = np.clip(ers_bits / 3.0, 0, 1).fillna(0)
    slope_pos_norm = pct_norm_pos(slope, q=90, floor=1.0).fillna(0)
    mom_mid_norm = pct_norm_pos(r10.clip(lower=0), q=90, floor=1.0).fillna(0)

    # 6.5: 지수 대비 60일 상대강도(양수만 반영)
    rel60_pos_norm = pct_norm_pos(rel60.clip(lower=0), q=90, floor=1.0).fillna(0)

    mom_norm = np.clip(
        0.45 * ers_norm
        + 0.25 * slope_pos_norm
        + 0.20 * mom_mid_norm
        + 0.10 * rel60_pos_norm,
        0,
        1
    ).fillna(0)

    # 유동성
    if turn.notna().any():
        lo, hi = np.nanpercentile(turn, 30), np.nanpercentile(turn, 90)
        denom = max(hi - lo, 1e-9)
        liq_norm = np.clip((turn - lo) / denom, 0, 1).fillna(0)
        liq_low = (turn < lo).astype(float)
    else:
        liq_norm = 0.0
        liq_low = 0.0

    # 기술적 세부: 거래량 스윗 스팟, 이격도
    vol_sweet = (1 - np.minimum((volz - 1).abs() / 3, 1)).clip(0, 1).fillna(0)
    kairi_abs = kairi.abs()
    kairi_norm = (1 - np.minimum(kairi_abs / cap_q(kairi_abs, q=80, floor=3.0), 1)).clip(0, 1).fillna(0)
    tec_norm = np.clip(0.6 * vol_sweet + 0.4 * kairi_norm, 0, 1).fillna(0)

    # 기본 점수
    base_score = (
        100 * W_RR * rr_norm
        + 100 * W_T1 * t1_norm
        + 100 * W_SL * sl_norm
        + 100 * W_NEAR * near_norm
        + 100 * W_MOM * mom_norm
        + 100 * W_LIQ * liq_norm
        + 100 * W_TEC * tec_norm
    )

    # 패널티
    pen = pd.Series(0.0, index=x.index)

    # 5일/10일 과열
    pen += P_OVERHEAT_5D * np.clip((r5 - 10) / 10, 0, 1)
    pen += P_OVERHEAT_10D * np.clip((r10 - 25) / 25, 0, 1)

    # RSI 범위 밖, MACD 기울기 음수
    pen += P_RSI_OUT * ((rsi < RSI_LOW) | (rsi > RSI_HIGH)).astype(float)
    pen += P_MACD_NEG * (slope < 0).astype(float)

    # 진입 괴리 과도
    pen += P_NEAR_FAR * np.clip((now_gap - 15) / 15, 0, 1)

    # 유동성 너무 낮거나, 거래량 스파이크 과도
    pen += P_LIQ_LOW * liq_low
    pen += P_VOL_SPIKE * (volz > 3).astype(float)

    score = np.clip(base_score - pen, 0, 100)
    x["RR1"] = rr1
    x["Now%"] = now_gap
    x["LDY_SCORE"] = score.round(1)

    x["ROUTE"] = x.apply(route_tag, axis=1)

    x["AI_COMMENT"] = x.apply(
        lambda row: generate_ai_comment(
            row.get("MFI14", 50),
            row.get("RSI14", 50),
            row.get("MACD_Slope", 0),
            row.get("이격도", 0),
            row.get("LDY_SCORE", 0),
        ),
        axis=1,
    )

    return x

# ------------------------------- 텔레그램 -------------------------------
def send_telegram_auto(df: pd.DataFrame, trade_ymd: str) -> None:
    log("📨 텔레그램 발송 시작...")
    if not TG_TOKEN or not TG_ID:
        log("⚠️ TG_TOKEN / TG_ID 미설정, 발송 생략")
        return

    try:
        top5 = df.head(5).reset_index(drop=True)
        trade_date = datetime.strptime(trade_ymd, "%Y%m%d").strftime('%Y-%m-%d')
        msg = f"🔥 [LDY v6.5] 추천 Top 5 ({trade_date})\n"
        msg += "-" * 30 + "\n\n"

        for i, row in top5.iterrows():
            rank = i + 1
            name = row['종목명']
            code = row['종목코드']
            route = row.get('ROUTE', '전략없음')
            buy = row['추천매수가']
            score = row.get('LDY_SCORE', 0)
            comment = row.get('AI_COMMENT', '')

            # 6.5: 60일 지수/상대강도 상세 표시용
            rel60 = row.get('rel_60d_%', np.nan)   # 종목 - 지수 (α)
            ret60 = row.get('ret_60d_%', np.nan)   # 종목 60일 수익률
            idx60 = row.get('idx_60d_%', np.nan)   # 지수 60일 수익률

            msg += f"{rank}. {name} ({code})\n"
            msg += f"   🌡점수: {score:.1f}점\n"
            msg += f"   🎯전략: {route}\n"

            # 👉 60일 상대강도/수익률 구조적으로 표시
            if not pd.isna(rel60):
                msg += f"   📊60일 상대강도(α): {rel60:+.2f}%\n"
                if not pd.isna(ret60) and not pd.isna(idx60):
                    msg += (
                        f"      · 종목: {ret60:+.2f}%  /  "
                        f"지수: {idx60:+.2f}%\n"
                    )

            msg += f"   💬AI: {comment}\n"
            msg += f"   🔵매수: {buy:,}\n"
            msg += f"   🔴손절: {row['손절가']:,} / 🟢목표: {row['추천매도가1']:,}\n\n"

        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_ID, "text": msg}
        )
        log("🚀 텔레그램 전송 완료")
    except Exception as e:
        log(f"⚠️ 텔레그램 전송 실패: {e}")

# ------------------------------- 티커 분석 -------------------------------
def analyze_ticker(
    t: str,
    start_s: str,
    end_s: str,
    top_df: pd.DataFrame,
    mcap_map: Dict[str, float],
    kospi_set: set,
    kosdaq_set: set,
    name_map: Dict[str, str],
    sector_map: Dict[str, str],
    bench_ret_60: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """
    단일 티커에 대한 지표 계산 및 추천 라인 생성
    """
    code6 = str(t).zfill(6)

    ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 120:
        return None
    ohlcv = ohlcv.tail(LOOKBACK_DAYS)

    c = ohlcv["종가"]
    h = ohlcv["고가"]
    l = ohlcv["저가"]
    v = ohlcv["거래량"]

    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    ma120 = c.rolling(120).mean()

    atr_series = calc_atr(h, l, c, 14)
    atr = atr_series.iloc[-1]

    rsi_series = calc_rsi(c, 14)
    rsi = rsi_series.iloc[-1]

    mfi_series = calc_mfi(h, l, c, v, 14)
    mfi = mfi_series.iloc[-1]

    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    hist = macd - sig
    slope = hist.diff().iloc[-1]

    vol_z_series = v / v.rolling(20).mean()
    vol_z = vol_z_series.iloc[-1]

    disp_series = (c / ma20 - 1.0) * 100
    disp = disp_series.iloc[-1]

    last_c = c.iloc[-1]

    # 5일/10일 수익률 계산
    if len(c) >= 6:
        ret_5 = (last_c / c.iloc[-6] - 1.0) * 100
    else:
        ret_5 = 0.0
    if len(c) >= 11:
        ret_10 = (last_c / c.iloc[-11] - 1.0) * 100
    else:
        ret_10 = 0.0

    # 60일 수익률 (개별 종목)
    if len(c) >= BENCH_LOOKBACK_DAYS + 1:
        ret_60 = (last_c / c.iloc[-(BENCH_LOOKBACK_DAYS + 1)] - 1.0) * 100
    else:
        ret_60 = 0.0

    tv_row = top_df.loc[top_df["종목코드"] == code6, "거래대금(원)"]
    if tv_row.empty:
        return None
    tv_eok = float(tv_row.values[0]) / 1e8

    mcap = get_mcap_eok_from_map(mcap_map, code6)

    # 거래대금/시총 필터
    if tv_eok < MIN_TURNOVER_EOK:
        return None

    if mcap > 0 and mcap < MIN_MCAP_EOK:
        return None

    # EBS(룰 기반) 스코어
    score = 0
    reason: List[str] = []
    if RSI_LOW <= rsi <= RSI_HIGH:
        score += 1; reason.append("RSI적정")
    if slope > 0:
        score += 1; reason.append("MACD상승")
    if -1 <= disp <= 5:
        score += 1; reason.append("20선근접")
    if vol_z > 1.2:
        score += 1; reason.append("거래량↑")
    if ma20.iloc[-1] > ma60.iloc[-1]:
        score += 1; reason.append("정배열(단)")
    if last_c > ma120.iloc[-1]:
        score += 1; reason.append("장기추세(120↑)")
    else:
        score -= 1
    if mfi > 60:
        score += 1; reason.append("자금유입(MFI)")
    if hist.iloc[-1] > 0:
        score += 1; reason.append("MACD>Sig")

    try:
        atr = float(atr)
    except Exception:
        atr = 0.0
    if np.isnan(atr) or atr <= 0:
        atr = last_c * 0.03

    if ma20.iloc[-1] > 0 and last_c > ma20.iloc[-1]:
        buy = min(last_c, ma20.iloc[-1] * 1.03)
    else:
        buy = last_c

    stop = buy - (2.0 * atr)
    if stop < buy * 0.93:
        stop = buy * 0.93
    if stop >= buy * 0.97:
        stop = buy * 0.97

    risk = buy - stop
    if score >= 8:
        rr1_val, rr2_val = (2.0, 4.0)
    elif score >= 6:
        rr1_val, rr2_val = (1.5, 3.0)
    else:
        rr1_val, rr2_val = (1.2, 2.5)

    t1 = buy + risk * rr1_val
    t2 = buy + risk * rr2_val

    buy = round_to_tick(buy)
    stop = round_to_tick(stop)
    t1 = round_to_tick(t1)
    t2 = round_to_tick(t2)

    sector = sector_map.get(code6, "기타")
    name = name_map.get(code6, code6)

    # 시장 구분 및 지수 60일 수익률 매핑
    market = "KOSPI" if t in kospi_set else "KOSDAQ"
    idx_60 = float(bench_ret_60.get(market, 0.0))
    rel_60 = ret_60 - idx_60

    row: Dict[str, Any] = {
        "시장": market,
        "종목명": name,
        "종목코드": code6,
        "업종": sector,
        "종가": int(last_c),
        "거래대금(억원)": round(tv_eok, 2),
        "시가총액(억원)": round(mcap, 1),
        "RSI14": round(float(rsi), 1),
        "MFI14": round(float(mfi), 1),
        "이격도": round(float(disp), 2),
        "MACD_Hist": round(float(hist.iloc[-1]), 4),
        "MACD_Slope": round(float(slope), 5),
        "거래강도": round(float(vol_z), 2),
        "ret_5d_%": round(float(ret_5), 2),
        "ret_10d_%": round(float(ret_10), 2),
        "ret_60d_%": round(float(ret_60), 2),
        "idx_60d_%": round(float(idx_60), 2),
        "rel_60d_%": round(float(rel_60), 2),
        "EBS": int(score),
        "통과": "★" if score >= PASS_EBS else "",
        "근거": ", ".join(reason),
        "추천매수가": buy,
        "손절가": stop,
        "추천매도가1": t1,
        "추천매도가2": t2,
    }

    return row

# ------------------------------- 메인 실행 -------------------------------
def main() -> None:
    log("🚀 LDY Collector v6.5 시작...")
    mcap_map, mcap_ymd = build_mcap_map()
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일: {trade_ymd} (mcap ref: {mcap_ymd})")

    # 지수 60일 수익률 계산
    bench_ret_60 = get_index_60d_returns(trade_ymd, BENCH_LOOKBACK_DAYS)
    def _fmt(v: Optional[float]) -> str:
        return f"{v:.2f}%" if isinstance(v, (int, float)) else "N/A"

    log(
        "📈 60일 기준 지수 수익률 - "
        f"KOSPI: {_fmt(bench_ret_60.get('KOSPI'))}, "
        f"KOSDAQ: {_fmt(bench_ret_60.get('KOSDAQ'))}"
    )

    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)

    # 시총 필터를 상위 거래대금 리스트에 선반영
    if mcap_map:
        top_df["시가총액(억원)"] = top_df["종목코드"].map(lambda c: get_mcap_eok_from_map(mcap_map, c))
        before_cnt = len(top_df)
        top_df = top_df[top_df["시가총액(억원)"] >= MIN_MCAP_EOK].copy()
        after_cnt = len(top_df)
        log(f"📊 시총 필터 적용: {before_cnt} → {after_cnt}개 (MIN_MCAP_EOK={MIN_MCAP_EOK})")
    else:
        log("⚠️ mcap_map 비어 있음 → 시총 사전 필터 생략")
        top_df["시가총액(억원)"] = 0.0

    tickers = top_df["종목코드"].tolist()

    kospi_set, kosdaq_set = get_market_sets(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)
    sector_map = build_sector_map()

    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS * 2 + 60)
    start_s, end_s = start_dt.strftime("%Y%m%d"), trade_ymd

    rows: List[Dict[str, Any]] = []
    err_cnt = 0
    for t in tqdm(tickers, desc="Analyzing"):
        code6 = str(t).zfill(6)
        try:
            row = analyze_ticker(
                t, start_s, end_s, top_df, mcap_map,
                kospi_set, kosdaq_set, name_map, sector_map,
                bench_ret_60
            )
            if row is not None:
                rows.append(row)
        except Exception as e:
            err_cnt += 1
            log(f"⚠️ {code6} 처리 중 오류 발생: {e}")
            continue

    if err_cnt > 0:
        log(f"⚠️ 분석 중 오류 발생 종목 수: {err_cnt}건")

    if not rows:
        raise RuntimeError("No Result (필터를 모두 통과한 종목 없음)")

    df_raw = pd.DataFrame(rows)
    df_out = build_global_score(df_raw).sort_values(
        ["LDY_SCORE", "거래대금(억원)"],
        ascending=[False, False]
    )

    # 메타데이터 컬럼 추가
    df_out["기준일"] = trade_ymd
    df_out["시총기준일"] = mcap_ymd
    df_out["벤치_60d_KOSPI_%"] = bench_ret_60.get("KOSPI", np.nan)
    df_out["벤치_60d_KOSDAQ_%"] = bench_ret_60.get("KOSDAQ", np.nan)

    # 6.5: 60일 수익률/상대강도 한국어 컬럼 alias
    if "ret_60d_%" in df_out.columns:
        df_out["60일_종목수익률_%"] = df_out["ret_60d_%"]
    if "idx_60d_%" in df_out.columns:
        df_out["60일_지수수익률_%"] = df_out["idx_60d_%"]
    if "rel_60d_%" in df_out.columns:
        df_out["60일_초과수익(α)_%"] = df_out["rel_60d_%"]    

    ensure_dir(OUT_DIR)
    date_tag = datetime.now(KST).strftime("%Y%m%d")
    out_path_dated = os.path.join(OUT_DIR, f"recommend_{date_tag}.csv")
    out_path_latest = os.path.join(OUT_DIR, "recommend_latest.csv")

    df_out.to_csv(out_path_dated, index=False, encoding=UTF8)
    df_out.to_csv(out_path_latest, index=False, encoding=UTF8)

    log(f"💾 저장 완료 ({len(df_out)}건) → {out_path_dated}")
    log(f"💾 최신 파일 업데이트 → {out_path_latest}")

    send_telegram_auto(df_out, trade_ymd)

if __name__ == "__main__":
    main()
