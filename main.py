# -*- coding: utf-8 -*-
"""
LDY Pro Trader — NiceGUI Full Edition
═══════════════════════════════════════
Streamlit dashboard.py 4,500줄 → NiceGUI 전체 변환

Tab 1: 📊 시장 현황
Tab 2: 🔭 종목 분석 (테이블 + 칸반 + 캔들차트 + 상세)
Tab 3: 💼 내 자산 (포트폴리오 AI 진단)
Tab 4: 📮 문의 게시판
Tab 5: ⚖️ 이용 약관
Tab 6: 🧩 업데이트 노트
Tab 7: 📈 시스템 성과
Tab 8: 👑 회원 관리 (Admin)
"""

import os
import math
import glob
import time
import logging
import hashlib
import secrets
import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pandas as pd
import numpy as np
import requests
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from nicegui import ui, app

# ─── [v6.0] 비동기 래퍼 (UI 프리징 방지) ───
from async_helpers import run_sync, run_cpu, register_shutdown

# ─── 신규 모듈 (v13 개선) ───
try:
    from price_cache import fetch_with_cache, fetch_prices_async, cache_stats, set_cache
    PRICE_CACHE_OK = True
except ImportError:
    PRICE_CACHE_OK = False

try:
    from kelly_widget import render_kelly_calculator, render_portfolio_kelly_summary
    KELLY_OK = True
except ImportError:
    KELLY_OK = False

try:
    from trade_journal_tab import render_trade_journal_tab
    JOURNAL_OK = True
except ImportError:
    JOURNAL_OK = False

# ─── 기존 모듈 재사용 ───
from shared_utils import nz_num, safe_float, calc_hma as calc_hma_series
from chart_components import (
    plot_fear_greed_gauge, plot_sector_treemap,
    plot_sector_momentum_bar, plot_radar_chart,
    plot_score_waterfall, plot_ai_gauge_chart,
)

# Optional imports (graceful fallback)
FDR_OK = False
PYKRX_OK = False
try:
    import FinanceDataReader as fdr
    FDR_OK = True
except ImportError:
    pass

try:
    from pykrx import stock
    PYKRX_OK = True
except ImportError:
    pass

try:
    from indicators import calculate_supertrend
except ImportError:
    def calculate_supertrend(df): return df

try:
    import version_info
    from version_info import APP_VERSION, CHANGELOG
except Exception:
    APP_VERSION = "12.3.0"
    CHANGELOG = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ldy-nicegui")

# ═══════════════════════════════════════════
#  1. 설정 & 상수
# ═══════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RECOMMEND_PATH = os.path.join(DATA_DIR, "recommend_latest.csv")
REMOTE_CSV_URL = os.getenv(
    "LDY_RAW_URL",
    "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
)
NICEGUI_VERSION = "2.0.0-nicegui"
PRICE_PRO = 19000
PRICE_PRIME = 39000

# Auth 상수
MASTER_ADMIN_ID = "admin"
_raw_admin_pw = os.environ.get("MASTER_ADMIN_PW", "").strip()
_ADMIN_PW_HASH = hashlib.sha256(_raw_admin_pw.encode()).hexdigest() if _raw_admin_pw else ""
_ADMIN_PW_SET = bool(_raw_admin_pw)
del _raw_admin_pw  # ✅ [v14] #18: 평문 즉시 제거

SECURITY_QUESTIONS = [
    "선택하세요...", "가장 기억에 남는 여행지는?", "어릴 적 살던 동네 이름은?",
    "가장 좋아하는 보물 1호는?", "초등학교 담임 선생님 성함은?",
    "나의 좌우명은?", "부모님의 고향은 어디인가요?",
]
ALLOWED_DOMAINS = [
    "naver.com", "gmail.com", "daum.net", "hanmail.net",
    "kakao.com", "nate.com", "icloud.com", "outlook.com",
    "hotmail.com", "yahoo.com", "taiyoinkproducts.co.kr"
]

KST = timezone(timedelta(hours=9))

def now_kst():
    return datetime.now(KST)

def to_kst_str(value, fmt="%Y-%m-%d %H:%M:%S"):
    if not value or str(value).strip() in ("", "-", "None", "NaT"):
        return "-"
    try:
        dt = pd.to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert(KST).strftime(fmt)
    except Exception:
        return str(value)

# ═══════════════════════════════════════════
#  2. Auth 시스템 (st.session_state → app.storage)
# ═══════════════════════════════════════════
def _get_db():
    try:
        from db_utils import get_db
        return get_db()
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return None

def _verify_admin_pw(pw):
    if not _ADMIN_PW_HASH: return False
    return secrets.compare_digest(hashlib.sha256(pw.encode()).hexdigest(), _ADMIN_PW_HASH)

def _create_salt(): return secrets.token_hex(16)
def _hash_pw(pw, salt): return hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 100000).hex()
def _hash_ans(ans, salt): return _hash_pw(ans.strip().lower(), salt)

def normalize_email(email):
    email = email.strip().lower()
    if "@" not in email: return email
    local, domain = email.split("@", 1)
    if domain in ("gmail.com", "googlemail.com"):
        local = local.replace(".", "")
        if "+" in local: local = local.split("+")[0]
    return f"{local}@{domain}"

def check_pw_strength(pw):
    return len(pw) >= 8 and re.search(r"[a-z]", pw.lower()) and re.search(r"[0-9]", pw)

def get_current_user():
    return app.storage.user.get("profile")

def set_current_user(profile):
    if profile:
        app.storage.user["profile"] = profile
    else:
        app.storage.user.pop("profile", None)

def get_auth_status():
    user = get_current_user()
    if not user: return "guest"
    role = user.get("role", "free")
    if role == "admin": return "admin"
    expire = user.get("prime_expire_date")
    if expire:
        try:
            exp_dt = datetime.strptime(str(expire).split(" ")[0], "%Y-%m-%d")
            if exp_dt.date() >= datetime.now().date():
                return role
        except Exception: pass
    return "free" if role in ("pro", "prime") else role


# ═══════════════════════════════════════════
#  3. 데이터 저장소
# ═══════════════════════════════════════════
class DataStore:
    def __init__(self):
        self.scored = pd.DataFrame()
        self.data_ts = ""
        self.loaded = False

    def refresh(self):
        df = None

        # 1) 로컬 파일 시도
        if os.path.exists(RECOMMEND_PATH):
            try:
                df = pd.read_csv(RECOMMEND_PATH, dtype={"종목코드": str, "종목명": str})
                logger.info(f"📂 로컬 CSV 로드: {RECOMMEND_PATH}")
            except Exception as e:
                logger.warning(f"로컬 CSV 읽기 실패: {e}")

        # 2) 로컬 실패 → GitHub raw URL 폴백
        if df is None or df.empty:
            url = REMOTE_CSV_URL.strip()
            if url:
                try:
                    logger.info(f"🌐 원격 CSV 다운로드 시도: {url}")
                    r = requests.get(url, timeout=30,
                                     headers={"Cache-Control": "no-cache"})
                    r.raise_for_status()
                    df = pd.read_csv(io.BytesIO(r.content),
                                     encoding="utf-8-sig",
                                     dtype={"종목코드": str, "종목명": str})
                    # 로컬에 캐싱 (다음 로드 시 빠르게)
                    os.makedirs(DATA_DIR, exist_ok=True)
                    with open(RECOMMEND_PATH, "wb") as f:
                        f.write(r.content)
                    logger.info(f"✅ 원격 CSV 다운로드 성공 → 로컬 캐싱 완료 ({len(df)}건)")
                except Exception as e:
                    logger.warning(f"원격 CSV 다운로드 실패: {e}")

        if df is None or df.empty:
            logger.warning("❌ 로컬/원격 모두 데이터 로드 실패")
            return

        try:
            num_cols = [
                "FINAL_SCORE", "DISPLAY_SCORE", "STRUCT_SCORE",
                "TIMING_SCORE", "AI_SCORE", "ML_SCORE", "TOTAL_SCORE",
                "RANK_SCORE", "EBS", "RR1", "RSI14",
                "거래대금(억원)", "종가", "추천매수가", "손절가",
                "추천매도가1", "추천매도가2", "TARGET_ATR",
            ]
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            # ─── 종목명 오염 자동 복구 (종목명==종목코드인 경우) ───
            if "종목코드" in df.columns and "종목명" in df.columns:
                mask = df["종목명"].astype(str).str.match(r'^\d+$')
                if mask.any():
                    _fixed = False
                    _bad_count = mask.sum()

                    # ── 1순위: krx_names_latest.csv (collector가 recommend와 함께 항상 저장) ──
                    _names_paths = [
                        os.path.join(DATA_DIR, "krx_names_latest.csv"),
                        "data/krx_names_latest.csv",
                        "/app/data/krx_names_latest.csv",
                    ]
                    for _np in _names_paths:
                        if _fixed:
                            break
                        try:
                            if os.path.exists(_np):
                                _ndf = pd.read_csv(_np, dtype=str)
                                if "종목코드" in _ndf.columns and "종목명" in _ndf.columns:
                                    _c2n = dict(zip(
                                        _ndf["종목코드"].astype(str).str.zfill(6),
                                        _ndf["종목명"]
                                    ))
                                    # 코드==이름인 항목 제거
                                    _c2n = {c: n for c, n in _c2n.items() if c != n and n and not n.isdigit()}
                                    if _c2n:
                                        df.loc[mask, "종목명"] = (
                                            df.loc[mask, "종목코드"].astype(str).str.zfill(6)
                                            .map(_c2n)
                                            .fillna(df.loc[mask, "종목명"])
                                        )
                                        _still_bad = df["종목명"].astype(str).str.match(r'^\d+$').sum()
                                        if _still_bad < _bad_count:
                                            logger.info(f"🔧 종목명 오염 {_bad_count - _still_bad}/{_bad_count}건 복구 [krx_names: {_np}]")
                                            _fixed = (_still_bad == 0)
                                            mask = df["종목명"].astype(str).str.match(r'^\d+$')  # 마스크 갱신
                        except Exception as _e:
                            logger.debug(f"krx_names 로드 실패 ({_np}): {_e}")

                    # ── 2순위: GitHub raw에서 krx_names_latest.csv 다운로드 ──
                    if not _fixed and mask.any():
                        try:
                            _base = REMOTE_CSV_URL.rsplit("/", 1)[0]
                            _names_url = f"{_base}/krx_names_latest.csv"
                            _resp = requests.get(_names_url, timeout=10)
                            if _resp.ok and _resp.text.strip():
                                _ndf = pd.read_csv(io.StringIO(_resp.text), dtype=str)
                                if "종목코드" in _ndf.columns and "종목명" in _ndf.columns:
                                    _c2n = dict(zip(
                                        _ndf["종목코드"].astype(str).str.zfill(6),
                                        _ndf["종목명"]
                                    ))
                                    _c2n = {c: n for c, n in _c2n.items() if c != n and n and not n.isdigit()}
                                    if _c2n:
                                        df.loc[mask, "종목명"] = (
                                            df.loc[mask, "종목코드"].astype(str).str.zfill(6)
                                            .map(_c2n)
                                            .fillna(df.loc[mask, "종목명"])
                                        )
                                        _still_bad = df["종목명"].astype(str).str.match(r'^\d+$').sum()
                                        logger.info(f"🔧 종목명 오염 {_bad_count - _still_bad}/{_bad_count}건 복구 [GitHub krx_names]")
                                        _fixed = (_still_bad == 0)
                                        mask = df["종목명"].astype(str).str.match(r'^\d+$')
                        except Exception as _e:
                            logger.debug(f"GitHub krx_names 다운로드 실패: {_e}")

                    # ── 3순위: _ensure_krx_map (FDR 전체 목록) ──
                    if not _fixed and mask.any():
                        _ensure_krx_map()
                        if _KRX_NAME_MAP:
                            _code_to_name = {v: k for k, v in _KRX_NAME_MAP.items()}
                            df.loc[mask, "종목명"] = (
                                df.loc[mask, "종목코드"].astype(str).str.zfill(6)
                                .map(_code_to_name)
                                .fillna(df.loc[mask, "종목명"])
                            )
                            _still_bad = df["종목명"].astype(str).str.match(r'^\d+$').sum()
                            if _still_bad < _bad_count:
                                logger.info(f"🔧 종목명 오염 {_bad_count - _still_bad}/{_bad_count}건 복구 [KRX캐시]")
                                _fixed = (_still_bad == 0)
                                mask = df["종목명"].astype(str).str.match(r'^\d+$')

                    # ── 4순위 (최후 수단): Naver API 개별 조회 ──
                    if not _fixed and mask.any():
                        _naver_fixed = 0
                        _codes = df.loc[mask, "종목코드"].astype(str).str.zfill(6).unique()
                        logger.info(f"🔄 Naver API로 종목명 {len(_codes)}건 개별 조회 시도...")
                        for _c in _codes:
                            try:
                                _r = requests.get(
                                    f"https://m.stock.naver.com/api/stock/{_c}/basic",
                                    timeout=5,
                                    headers={"User-Agent": "Mozilla/5.0"}
                                )
                                if _r.ok:
                                    _name = _r.json().get("stockName", "")
                                    if _name and _name != _c:
                                        df.loc[(mask) & (df["종목코드"].astype(str).str.zfill(6) == _c), "종목명"] = _name
                                        _naver_fixed += 1
                            except Exception:
                                pass
                        if _naver_fixed:
                            logger.info(f"🔧 종목명 오염 {_naver_fixed}/{len(_codes)}건 복구 [Naver]")

                    # 최종 상태 로깅
                    _final_bad = df["종목명"].astype(str).str.match(r'^\d+$').sum()
                    if _final_bad > 0:
                        logger.warning(f"⚠️ 종목명 복구 불완전: {_final_bad}건 여전히 코드 상태")
            # ─── 종목명 복구 끝 ──────────────────────────────────

            primary = next((c for c in ["DISPLAY_SCORE", "FINAL_SCORE", "TOTAL_SCORE"] if c in df.columns and df[c].abs().sum() > 0), None)
            if primary:
                for alias in ["DISPLAY_SCORE", "TOTAL_SCORE", "LDY_SCORE", "RANK_SCORE"]:
                    df[alias] = df[primary]

            ts_col = next((c for c in ["trade_date", "DATA_DATE"] if c in df.columns), None)
            self.data_ts = str(df[ts_col].iloc[0]) if ts_col else now_kst().strftime("%Y-%m-%d")
            self.scored = df
            self.loaded = True
            logger.info(f"✅ 데이터 로드: {len(df)}종목, 기준일 {self.data_ts}")
        except Exception as e:
            logger.exception(f"데이터 로드 실패: {e}")

store = DataStore()


# ═══════════════════════════════════════════
#  4. 데이터 유틸리티 (dashboard.py에서 추출)
# ═══════════════════════════════════════════
def get_fear_greed(df):
    if df.empty or "DISPLAY_SCORE" not in df.columns:
        return 50, "데이터 부족"
    avg = df["DISPLAY_SCORE"].mean()
    score = min(max(avg, 0), 100)
    if score >= 80: label = "극단적 탐욕"
    elif score >= 60: label = "탐욕"
    elif score >= 40: label = "중립"
    elif score >= 20: label = "공포"
    else: label = "극단적 공포"
    return score, label

def get_code_map(df):
    if df.empty or "종목코드" not in df.columns or "종목명" not in df.columns:
        return {}
    return dict(zip(df["종목명"], df["종목코드"].astype(str).str.zfill(6)))

def _recent_trade_date() -> str:
    """최근 거래일 (주말/공휴일 보정) — YYYYMMDD 형식"""
    dt = now_kst()
    wd = dt.weekday()
    if wd == 5: dt -= timedelta(days=1)
    elif wd == 6: dt -= timedelta(days=2)
    return dt.strftime("%Y%m%d")


# KRX 전체 종목 캐시 (앱 시작 시 1번만 로드)
_KRX_NAME_MAP = {}

def _ensure_krx_map():
    """전체 종목 목록 로드 (FDR → GitHub CSV → 로컬 파일 순 폴백)"""
    global _KRX_NAME_MAP
    if _KRX_NAME_MAP:
        return

    # ── 방법 1: FDR (Railway 해외 IP에서 실패 가능) ──
    if FDR_OK:
        try:
            listing = fdr.StockListing("KRX")
            if listing is not None and not listing.empty:
                code_col = None
                for c in ["Code", "Symbol", "Ticker", "ISU_SRT_CD", "종목코드"]:
                    if c in listing.columns:
                        code_col = c
                        break
                name_col = None
                for c in ["Name", "종목명", "ISU_ABBRV"]:
                    if c in listing.columns:
                        name_col = c
                        break
                if code_col is None and listing.index.dtype == object:
                    sample_idx = str(listing.index[0]).strip()
                    if sample_idx.isdigit() and len(sample_idx) == 6:
                        listing = listing.reset_index()
                        listing.rename(columns={listing.columns[0]: "_idx_code"}, inplace=True)
                        code_col = "_idx_code"
                if code_col and name_col:
                    _KRX_NAME_MAP = dict(zip(listing[name_col], listing[code_col].astype(str).str.zfill(6)))
                    logger.info(f"✅ KRX 종목 캐시 [FDR]: {len(_KRX_NAME_MAP)}개")
                    return
                else:
                    logger.warning(f"⚠️ FDR 컬럼 매칭 실패: cols={listing.columns.tolist()[:10]}")
        except Exception as e:
            logger.warning(f"⚠️ FDR 로드 실패: {e}")

    # ── 방법 2: GitHub에서 krx_names_latest.csv 다운로드 ──
    try:
        _base = REMOTE_CSV_URL.rsplit("/", 1)[0]  # .../data
        _names_url = f"{_base}/krx_names_latest.csv"
        resp = requests.get(_names_url, timeout=10)
        if resp.ok and resp.text.strip():
            _df = pd.read_csv(io.StringIO(resp.text), dtype=str)
            if "종목코드" in _df.columns and "종목명" in _df.columns:
                _map = {}
                for _, row in _df.iterrows():
                    c = str(row["종목코드"]).strip().zfill(6)
                    n = str(row["종목명"]).strip()
                    if c != n and n:
                        _map[n] = c
                if _map:
                    _KRX_NAME_MAP = _map
                    logger.info(f"✅ KRX 종목 캐시 [GitHub]: {len(_KRX_NAME_MAP)}개")
                    return
    except Exception as e:
        logger.warning(f"⚠️ GitHub 종목명 다운로드 실패: {e}")

    # ── 방법 3: 로컬 파일 폴백 ──
    for _path in ["data/krx_names_latest.csv", "/app/data/krx_names_latest.csv"]:
        try:
            if os.path.exists(_path):
                _df = pd.read_csv(_path, dtype=str)
                if "종목코드" in _df.columns and "종목명" in _df.columns:
                    _map = {str(row["종목명"]).strip(): str(row["종목코드"]).strip().zfill(6)
                            for _, row in _df.iterrows()
                            if str(row["종목명"]).strip() != str(row["종목코드"]).strip()}
                    if _map:
                        _KRX_NAME_MAP = _map
                        logger.info(f"✅ KRX 종목 캐시 [로컬]: {len(_KRX_NAME_MAP)}개")
                        return
        except Exception:
            pass

    logger.warning("⚠️ KRX 종목 매핑 로드 완전 실패 — 종목명이 코드로 표시될 수 있음")


def find_code_by_name(name, code_map):
    """종목명 → 코드 검색 (scored → KRX 전체 캐시 순, pykrx 미사용)"""
    if name in code_map: return code_map[name]
    for k, v in code_map.items():
        if name in k or k in name: return v
    # scored에 없으면 FDR 기반 전체 종목 캐시에서 검색
    _ensure_krx_map()
    if name in _KRX_NAME_MAP:
        return _KRX_NAME_MAP[name]
    for k, v in _KRX_NAME_MAP.items():
        if name in k or k in name:
            return v
    return name

def get_stock_chart_data(code):
    """캔들차트 데이터 (FDR 기반)"""
    if not FDR_OK: return None
    try:
        code_str = str(code).zfill(6)
        start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        df = fdr.DataReader(code_str, start)
        if df is None or df.empty: return None

        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()

        std20 = df['Close'].rolling(20).std()
        df['BB_UPPER'] = df['MA20'] + 2.0 * std20
        df['BB_LOWER'] = df['MA20'] - 2.0 * std20

        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr20 = tr.rolling(20).mean()
        df['KC_UPPER'] = df['MA20'] + 1.5 * atr20
        df['KC_LOWER'] = df['MA20'] - 1.5 * atr20

        delta = df['Close'].diff()
        rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()
        df['RSI14_CHART'] = 100 - (100 / (1 + rs))

        try:
            df['HMA20'] = calc_hma_series(df['Close'], 20)
        except Exception:
            pass

        change = np.sign(df['Close'].diff()).fillna(0)
        df['OBV'] = (change * df['Volume']).cumsum()

        df = calculate_supertrend(df)

        try:
            logic_w = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df_w = df.resample('W').apply(logic_w)
            df_w['WMA20'] = df_w['Close'].rolling(20).mean()
            df['WEEKLY_MA20'] = df.index.map(
                lambda x: df_w.loc[df_w.index <= x, 'WMA20'].iloc[-1]
                if not df_w.loc[df_w.index <= x, 'WMA20'].empty else np.nan
            )
        except Exception:
            pass

        return df.tail(120)
    except Exception:
        logger.exception("get_stock_chart_data failed")
        return None


def _add_vpvr_trace(fig, df, price_bins: int = 25):
    """
    가격대별 거래량 히스토그램 (VPVR) — 캔들차트 우측에 투명 바 오버레이.
    매물대가 두꺼운 구간(황색)이 지지·저항선 역할을 시각적으로 납득시킨다.

    ✅ [Fix #3] 고래(Block Trade) 왜곡 방지:
      - 95th percentile clip: 상위 5% 극단치를 95% 값으로 제한
      - log1p 스케일: 이후 로그 변환으로 중간 매물대도 시각적으로 부각
      덕분에 기관 블록딜 한 번이 터져도 나머지 매물대가 점처럼 사라지지 않는다.
    """
    try:
        if "Volume" not in df.columns or df.empty:
            return
        lo = float(df["Low"].min())
        hi = float(df["High"].max())
        if hi <= lo:
            return

        bin_size = (hi - lo) / price_bins
        centers, volumes = [], []
        for i in range(price_bins):
            b_lo = lo + bin_size * i
            b_hi = b_lo + bin_size
            mid  = (b_lo + b_hi) / 2
            mask = (df["Close"] >= b_lo) & (df["Close"] < b_hi)
            vol  = int(df.loc[mask, "Volume"].sum())
            centers.append(mid)
            volumes.append(vol)

        if not volumes or max(volumes) == 0:
            return

        # ── ✅ Fix #3a: 95th percentile clip (블록딜·자전거래 극단치 제거) ──
        import numpy as np
        arr        = np.array(volumes, dtype=float)
        clip_ceil  = float(np.percentile(arr, 95))
        arr_clipped = np.clip(arr, 0, clip_ceil)

        # ── ✅ Fix #3b: log1p 스케일 (중간 매물대 시인성 확보) ──
        arr_log = np.log1p(arr_clipped)
        max_log = arr_log.max() if arr_log.max() > 0 else 1.0

        n = len(df)
        for mid, log_vol, raw_vol in zip(centers, arr_log, volumes):
            if log_vol == 0:
                continue
            bar_w   = (log_vol / max_log) * n * 0.15  # 전체 캔들 폭의 15%
            # 고거래량 강조 기준도 clip된 값 기준으로 (원본 arr 아닌 arr_clipped)
            is_hv   = raw_vol >= float(np.percentile(arr, 70))
            color   = "rgba(251,191,36,0.55)" if is_hv else "rgba(100,116,139,0.22)"
            x_start_idx = max(0, n - 1 - int(bar_w))
            x_start = df.index[x_start_idx]
            x_end   = df.index[-1]
            fig.add_shape(
                type="rect",
                x0=x_start, x1=x_end,
                y0=mid - bin_size * 0.45,
                y1=mid + bin_size * 0.45,
                fillcolor=color,
                line=dict(width=0),
                layer="below",
                row=1, col=1,
            )
    except Exception as e:
        logger.debug(f"VPVR 렌더 실패: {e}")


def _add_vpvr(fig, df, price_bins=30, row=1, col=1):
    """(하위 호환 래퍼 — _add_vpvr_trace로 위임)"""
    _add_vpvr_trace(fig, df, price_bins=price_bins)


def plot_candle_chart(df, code, name, entry=None, stop=None, target1=None, target2=None):
    """캔들차트 (dashboard.py plot_interactive_chart 변환)"""
    if df is None or df.empty:
        return go.Figure()

    df = df.copy()
    col_map = {"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    rows = 2
    has_rsi = "RSI14_CHART" in df.columns
    if has_rsi: rows += 1
    row_heights = [0.6] + [0.4 / (rows - 1)] * (rows - 1)

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights)

    # 🇰🇷 한국식: 상승=빨강, 하락=파랑
    C_UP, C_DOWN = '#EF4444', '#3B82F6'

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=C_UP, increasing_fillcolor=C_UP,
        decreasing_line_color=C_DOWN, decreasing_fillcolor=C_DOWN,
        showlegend=False
    ), row=1, col=1)

    if "MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color='#FFD700', width=1.5)), row=1, col=1)
    if "WEEKLY_MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["WEEKLY_MA20"], name="주봉20선", line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dot')), row=1, col=1)

    if "BB_UPPER" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], fill='tonexty', fillcolor='rgba(33,150,243,0.07)', line=dict(width=0), name="BB", hoverinfo='skip'), row=1, col=1)

    if "Trend" in df.columns and "SuperTrend" in df.columns:
        up = df[df["Trend"] == 1]["SuperTrend"]
        if not up.empty:
            fig.add_trace(go.Scatter(x=up.index, y=up, mode='lines', line=dict(color=C_UP, width=2), name='SuperTrend'), row=1, col=1)
        dn = df[df["Trend"] == -1]["SuperTrend"]
        if not dn.empty:
            fig.add_trace(go.Scatter(x=dn.index, y=dn, mode='lines', line=dict(color=C_DOWN, width=2), showlegend=False), row=1, col=1)

    def _add_hline(val, color, label, dash="dash"):
        if val is None: return
        try:
            v = float(str(val).replace(',', ''))
            if v > 0 and not math.isnan(v):
                fig.add_hline(y=v, line_dash=dash, line_color=color, line_width=1,
                              annotation_text=label, annotation_position="top right", annotation_font_color=color)
        except Exception:
            pass

    _add_hline(entry, "#2962FF", "Entry", "solid")
    _add_hline(stop, "#FF3B30", "Stop Loss")
    _add_hline(target1, "#00E676", "T1")
    _add_hline(target2, "#EAB308", "T2", "dashdot")

    cur_row = 2
    if "Volume" in df.columns:
        colors = [C_UP if c >= o else C_DOWN for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, opacity=0.6, showlegend=False), row=cur_row, col=1)
        cur_row += 1

    if has_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14_CHART"], name="RSI", line=dict(color='#E040FB', width=1.5)), row=cur_row, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, layer="below", row=cur_row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="blue", opacity=0.1, layer="below", row=cur_row, col=1)

    # VPVR: 가격대별 거래량 히스토그램 (우측)
    try:
        _add_vpvr_trace(fig, df)
    except Exception:
        pass

    fig.update_layout(
        title=dict(text=f"<b>{name}</b> ({code})", x=0.02),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        height=550, margin=dict(l=10, r=50, t=50, b=10),
        hovermode="x unified", showlegend=False,
        font_color="white",
    )
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.1)', side='right')
    return fig


def fetch_current_price(code, name):
    """현재가 조회 — 캐시 → Circuit Breaker → FDR 순 (Railway 해외 서버 대응)"""
    code_str = str(code).zfill(6) if str(code).isdigit() else ""

    def _fdr_fetch(c: str) -> int:
        """FDR 동기 조회"""
        if not FDR_OK or not c:
            return 0
        try:
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            d = fdr.DataReader(c, start)
            if d is not None and not d.empty:
                return int(d.iloc[-1]["Close"])
        except Exception:
            pass
        return 0

    # ── 캐시 + Circuit Breaker 경로 ──
    if PRICE_CACHE_OK and code_str:
        c, n, p = fetch_with_cache(code_str, name, _fdr_fetch)
        if p > 0:
            return c, n, p

    # ── 코드 없으면 KRX 캐시에서 검색 ──
    if FDR_OK and not code_str:
        _ensure_krx_map()
        found = _KRX_NAME_MAP.get(name)
        if not found:
            for k, v in _KRX_NAME_MAP.items():
                if name in k or k in name:
                    found = v
                    break
        if found:
            if PRICE_CACHE_OK:
                c, n, p = fetch_with_cache(found, name, _fdr_fetch)
                if p > 0:
                    return found, name, p
            else:
                p = _fdr_fetch(found)
                if p > 0:
                    return found, name, p

    # ── 직접 FDR (캐시 없는 fallback) ──
    if FDR_OK and code_str:
        p = _fdr_fetch(code_str)
        if p > 0:
            return code, name, p

    return code, name, 0


# Portfolio Gist I/O
def load_portfolio_file():
    token = os.environ.get("LDY_GIST_TOKEN", "")
    gist_id = os.environ.get("LDY_GIST_ID", "")
    if not token or not gist_id: return ""
    try:
        import requests
        r = requests.get(f"https://api.github.com/gists/{gist_id}", headers={"Authorization": f"token {token}"}, timeout=10)
        if r.ok:
            files = r.json().get("files", {})
            if "portfolio.txt" in files:
                return files["portfolio.txt"]["content"]
    except Exception:
        pass
    return ""

def save_portfolio_file(text_data):
    token = os.environ.get("LDY_GIST_TOKEN", "")
    gist_id = os.environ.get("LDY_GIST_ID", "")
    if not token or not gist_id: return False
    try:
        import requests
        r = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {token}"},
            json={"files": {"portfolio.txt": {"content": text_data}}},
            timeout=10,
        )
        return r.ok
    except Exception:
        return False

def load_inquiry_items():
    db = _get_db()
    return db.get_all_inquiries() if db else []

def save_inquiry_items(items):
    db = _get_db()
    return db.save_inquiries(items) if db else False


# ═══════════════════════════════════════════
#  5. 공통 UI 컴포넌트
# ═══════════════════════════════════════════
DARK_CSS = """
<style>
    body { background: #0d0d1a !important; }
    .nicegui-content { max-width: 1400px; margin: 0 auto; }
    .q-tab-panel { padding: 8px 0 !important; }
    .q-card { background: #1a1a2e !important; }
    .q-table { background: #1a1a2e !important; color: white !important; }
    .q-table th { color: #94A3B8 !important; }
    .kanban-col { min-width: 280px; flex: 1; }
    .kanban-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 12px; margin-bottom: 8px;
        transition: transform 0.2s; cursor: pointer;
    }
    .kanban-card:hover { transform: translateY(-2px); background: rgba(255,255,255,0.08); }
</style>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<!-- PWA -->
<link rel="manifest" href="/static/manifest.json">
<meta name="theme-color" content="#1a1a2e">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="LDY Trader">
<link rel="apple-touch-icon" href="/static/icon-192.png">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5, user-scalable=yes">
<script>
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/sw.js').then(r => console.log('SW registered'));
}
</script>
"""


def metric_card(title, value, delta="", positive=True):
    with ui.card().classes("p-4 min-w-[140px] bg-[#1a1a2e] border border-gray-700 rounded-xl"):
        ui.label(title).classes("text-xs text-gray-400 uppercase tracking-wide")
        ui.label(str(value)).classes("text-xl font-bold text-white mt-1")
        if delta:
            color = "text-green-400" if positive else "text-red-400"
            ui.label(str(delta)).classes(f"text-sm {color} mt-0.5")


def section_title(text):
    ui.label(text).classes("text-lg font-bold text-white mt-6 mb-2 border-b border-gray-700 pb-2")


def price_bar_html(stop, entry, close, t1, t2=0):
    points = [("손절", stop, "#EF4444"), ("매수", entry, "#3B82F6"), ("현재", close, "#FFFFFF")]
    if t1 > 0: points.append(("T1", t1, "#10B981"))
    if t2 > 0 and t2 != t1: points.append(("T2", t2, "#EAB308"))
    points.sort(key=lambda x: x[1])
    p_min, p_max = points[0][1] * 0.98, points[-1][1] * 1.02
    rng = p_max - p_min
    if rng <= 0: return ""

    html = '<div style="position:relative;height:55px;background:linear-gradient(90deg,rgba(239,68,68,0.15) 0%,rgba(16,185,129,0.15) 100%);border-radius:10px;margin:8px 0 20px 0;">'
    for label, price, color in points:
        pct = max(3, min((price - p_min) / rng * 100, 97))
        is_cur = label == "현재"
        sz = "14px" if is_cur else "10px"
        bdr = "2px solid #FFF" if is_cur else "none"
        fw = "bold" if is_cur else "normal"
        html += (
            f'<div style="position:absolute;left:{pct}%;top:50%;transform:translate(-50%,-50%);z-index:{"10" if is_cur else "5"};text-align:center;">'
            f'<div style="width:{sz};height:{sz};background:{color};border-radius:50%;border:{bdr};margin:0 auto;"></div>'
            f'<div style="font-size:10px;color:{color};white-space:nowrap;margin-top:3px;font-weight:{fw};">{label}<br>{int(price):,}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def _plotly_dark(fig, height=300):
    """Plotly 차트 다크 테마 적용"""
    if fig:
        fig.update_layout(
            height=height, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
            margin=dict(t=30, b=10, l=10, r=10),
        )
    return fig


# ═══════════════════════════════════════════
#  6. 로그인 페이지
# ═══════════════════════════════════════════
@ui.page('/login')
def login_page():
    ui.add_head_html(DARK_CSS.replace("1400px", "500px"))

    with ui.card().classes("w-full p-8 bg-[#1a1a2e] border border-gray-700 rounded-2xl mt-16"):
        ui.label("🔐 LDY Pro Trader").classes("text-2xl font-bold text-center text-white w-full mb-4")

        with ui.tabs().classes("w-full") as tabs:
            t_login = ui.tab("로그인")
            t_join = ui.tab("전략군 가입")
            t_recover = ui.tab("계정 복구")

        with ui.tab_panels(tabs, value=t_login).classes("w-full"):
            with ui.tab_panel(t_login):
                lid = ui.input("아이디 (또는 이메일)").classes("w-full")
                lpw = ui.input("비밀번호", password=True, password_toggle_button=True).classes("w-full")
                msg = ui.label("").classes("text-sm mt-2")

                async def do_login():
                    uid = lid.value.strip()
                    pw = lpw.value
                    if uid == MASTER_ADMIN_ID and _ADMIN_PW_SET and _verify_admin_pw(pw):
                        set_current_user({"id": "admin", "role": "admin", "nickname": "관리자"})
                        ui.navigate.to("/"); return
                    db = _get_db()
                    if not db:
                        msg.set_text("❌ DB 연결 실패"); msg.classes(replace="text-sm mt-2 text-red-400"); return
                    clean = normalize_email(uid)
                    u = db.get_user_by_id(clean)
                    h = _hash_pw(pw, u["salt"] if u else "dummy")
                    if u and h == u.get("password"):
                        if str(u.get("is_banned")).upper() in ["Y", "TRUE", "1"]:
                            msg.set_text("🚫 차단된 계정"); msg.classes(replace="text-sm mt-2 text-red-400"); return
                        # ──────────────────────────────────────────
                        # [Fix] 최근접속일 갱신
                        # ──────────────────────────────────────────
                        try:
                            db.update_login_timestamp(clean)
                        except Exception:
                            pass
                        set_current_user({"id": u["id"], "login_id": u["id"], "role": u.get("role", "free"),
                                          "nickname": u.get("nickname"), "prime_expire_date": u.get("prime_expire_date")})
                        ui.navigate.to("/")
                    else:
                        msg.set_text("아이디 또는 비밀번호 오류"); msg.classes(replace="text-sm mt-2 text-red-400")

                ui.button("로그인", on_click=do_login).classes("w-full mt-4").props("color=primary")
                ui.button("🔓 둘러보기 (게스트)", on_click=lambda: ui.navigate.to("/")).classes("w-full mt-2").props("flat")

            with ui.tab_panel(t_join):
                ui.label("👋 가입을 환영합니다!").classes("text-white mb-2")
                j_em = ui.input("이메일").classes("w-full")
                j_nk = ui.input("닉네임 (최대 8자)").classes("w-full")
                j_p1 = ui.input("비밀번호 (8자+, 영문/숫자)", password=True).classes("w-full")
                j_p2 = ui.input("비밀번호 확인", password=True).classes("w-full")
                j_q = ui.select({i: q for i, q in enumerate(SECURITY_QUESTIONS)}, value=0, label="보안 질문").classes("w-full")
                j_ans = ui.input("보안 질문 답변").classes("w-full")
                j_msg = ui.label("").classes("text-sm mt-2")

                async def do_join():
                    domain = j_em.value.split("@")[-1].lower() if "@" in j_em.value else ""
                    if domain not in ALLOWED_DOMAINS:
                        j_msg.set_text("🚫 허용 도메인 아님"); j_msg.classes(replace="text-sm mt-2 text-red-400"); return
                    if not check_pw_strength(j_p1.value):
                        j_msg.set_text("⚠️ 8자+영문+숫자"); j_msg.classes(replace="text-sm mt-2 text-red-400"); return
                    if j_p1.value != j_p2.value:
                        j_msg.set_text("비밀번호 불일치"); j_msg.classes(replace="text-sm mt-2 text-red-400"); return
                    db = _get_db()
                    if not db: j_msg.set_text("DB 오류"); return
                    salt = _create_salt()
                    ok, m = db.register_user(normalize_email(j_em.value), _hash_pw(j_p1.value, salt), salt, j_nk.value[:8], j_q.value, _hash_ans(j_ans.value, salt))
                    if ok:
                        j_msg.set_text("🎉 가입 성공! 로그인하세요."); j_msg.classes(replace="text-sm mt-2 text-green-400")
                    else:
                        j_msg.set_text(m); j_msg.classes(replace="text-sm mt-2 text-red-400")
                ui.button("가입 신청", on_click=do_join).classes("w-full mt-4").props("color=primary")

            with ui.tab_panel(t_recover):
                r_id = ui.input("이메일").classes("w-full")
                r_ans = ui.input("보안 답변").classes("w-full")
                r_pw = ui.input("새 비밀번호", password=True).classes("w-full")
                r_msg = ui.label("").classes("text-sm mt-2")

                async def do_recover():
                    db = _get_db()
                    if not db: r_msg.set_text("DB 오류"); return
                    u = db.get_user_by_id(normalize_email(r_id.value.strip()))
                    ok = False
                    if u and _hash_ans(r_ans.value, u["salt"]) == u.get("security_ans"):
                        if check_pw_strength(r_pw.value):
                            ns = _create_salt()
                            ok = db.update_user_password(normalize_email(r_id.value), _hash_pw(r_pw.value, ns), ns)
                    r_msg.set_text("✅ 변경 완료!" if ok else "정보 불일치")
                    r_msg.classes(replace=f"text-sm mt-2 {'text-green-400' if ok else 'text-red-400'}")
                ui.button("비밀번호 재설정", on_click=do_recover).classes("w-full mt-4").props("color=primary")


# ═══════════════════════════════════════════
#  7. 메인 페이지 (8개 탭)
# ═══════════════════════════════════════════
@ui.page('/')
async def index():
    ui.add_head_html(DARK_CSS)
    if not store.loaded:
        await run_sync(store.refresh)
    df = store.scored
    auth = get_auth_status()
    user = get_current_user()

    # ─── Hero Banner ───
    with ui.row().classes("w-full items-center justify-between px-4 py-3 rounded-xl mb-2 bg-gradient-to-r from-[#1a1a2e] via-[#16213e] to-[#0f3460]"):
        with ui.column().classes("gap-0"):
            ui.label("💎 LDY Pro Trader").classes("text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400").style("font-family:Outfit,sans-serif")
            ui.label(f"v{APP_VERSION} · NiceGUI Edition").classes("text-xs text-gray-400")
        with ui.row().classes("gap-2 items-center"):
            if user:
                ui.label(f"👋 {user.get('nickname', '')}").classes("text-white text-sm")
                ui.badge(auth.upper()).props(f'color={"green" if auth in ("admin","prime") else "blue" if auth == "pro" else "gray"}')
                ui.button("로그아웃", on_click=lambda: (set_current_user(None), ui.navigate.to("/login"))).props("flat dense").classes("text-white text-xs")
            else:
                ui.button("🔐 로그인", on_click=lambda: ui.navigate.to("/login")).props("flat dense").classes("text-white")
            ui.button("🔄", on_click=_do_refresh).props("flat round dense").classes("text-white")

    if df.empty:
        ui.label("⚠️ 데이터 없음 — data/recommend_latest.csv 확인").classes("text-yellow-400 text-lg p-8")
        return

    # ─── 탭 구성 ───
    with ui.tabs().classes("w-full text-white") as tabs:
        t1 = ui.tab("📊 시장")
        t2 = ui.tab("🔭 종목 분석")
        t3 = ui.tab("💼 내 자산")
        t4 = ui.tab("📮 문의")
        t5 = ui.tab("⚖️ 약관")
        t6 = ui.tab("🧩 업데이트")
        t7 = ui.tab("📈 성과")
        t9 = ui.tab("📓 매매 일지")
        if auth == "admin":
            t8 = ui.tab("👑 관리")

    with ui.tab_panels(tabs, value=t1).classes("w-full"):
        with ui.tab_panel(t1): render_tab1_market(df)
        with ui.tab_panel(t2): render_tab2_stocks(df, auth)
        with ui.tab_panel(t3): render_tab3_portfolio(df, auth)
        with ui.tab_panel(t4): render_tab4_inquiry(auth, user)
        with ui.tab_panel(t5): render_tab5_terms()
        with ui.tab_panel(t6): render_tab6_updates()
        with ui.tab_panel(t7): render_tab7_performance()
        with ui.tab_panel(t9):
            if JOURNAL_OK:
                render_trade_journal_tab(df_scored=df)
            else:
                ui.label("⚠️ trade_journal_tab 모듈 없음").classes("text-yellow-400")
        if auth == "admin":
            with ui.tab_panel(t8): render_tab8_admin()

    # 푸터
    ui.label(f"📅 데이터 기준: {store.data_ts} · ⚠️ 투자 판단은 본인 책임").classes("text-xs text-gray-500 text-center mt-8 mb-4")


async def _do_refresh():
    await run_sync(store.refresh)
    ui.notify("🔄 데이터 새로고침 완료!", type="positive")
    await ui.run_javascript("setTimeout(()=>location.reload(),500)")


# ═══════════════════════════════════════════
#  Tab 1: 시장 현황
# ═══════════════════════════════════════════
def _render_macro_sparklines():
    """
    글로벌 매크로 스파크라인 배너 (Bloomberg 터미널 스타일).
    FDR로 USD/KRW, NASDAQ, KOSPI, 미국 10년 국채금리 최근 20일 데이터.
    """
    if not FDR_OK:
        return

    MACRO_TICKERS = [
        ("USD/KRW",    "USD/KRW",   "#F59E0B"),
        ("NASDAQ",     "IXIC",      "#3B82F6"),
        ("KOSPI",      "KS11",      "#10B981"),
        ("US 10Y",     "US10YT=RR", "#E040FB"),
    ]

    with ui.card().classes("w-full p-3 bg-[#0d0d1a] border border-gray-700/50 rounded-xl mb-4"):
        ui.label("🌍 글로벌 매크로").classes("text-xs text-gray-400 mb-2")
        with ui.row().classes("w-full gap-3 flex-wrap"):
            for label, ticker, color in MACRO_TICKERS:
                _spark_card(label, ticker, color)


def _spark_card(label: str, ticker: str, color: str):
    """개별 스파크라인 카드 (20일)"""
    import plotly.graph_objects as go
    try:
        from nicegui import ui
    except ImportError:
        return

    with ui.card().classes("flex-1 min-w-[140px] p-2 bg-[#1a1a2e] border border-gray-700/50 rounded-lg"):
        val_label  = ui.label("—").classes("text-sm font-bold text-white")
        chg_label  = ui.label("—").classes("text-xs")
        chart_slot = ui.column().classes("w-full")

        async def _load():
            try:
                start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
                d = await run_sync(fdr.DataReader, ticker, start)
                if d is None or d.empty:
                    val_label.set_text("N/A")
                    return
                d = d.tail(20)
                last   = float(d["Close"].iloc[-1])
                prev   = float(d["Close"].iloc[-2]) if len(d) > 1 else last
                chg    = (last - prev) / prev * 100 if prev else 0

                # 표시 형식
                if ticker in ("USD/KRW",):
                    fmt = f"{last:,.1f}"
                elif "10Y" in ticker:
                    fmt = f"{last:.3f}%"
                else:
                    fmt = f"{last:,.2f}"

                val_label.set_text(f"{label}: {fmt}")
                chg_label.set_text(f"{chg:+.2f}%")
                chg_label.classes(replace="text-xs text-green-400" if chg >= 0 else "text-xs text-red-400")

                # 스파크라인 mini chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(d))),
                    y=d["Close"].tolist(),
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    fill="tozeroy",
                    fillcolor=f"{color}22",
                    showlegend=False,
                ))
                fig.update_layout(
                    height=50, margin=dict(t=0,b=0,l=0,r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                )
                chart_slot.clear()
                with chart_slot:
                    ui.plotly(fig).classes("w-full")
            except Exception as _e:
                val_label.set_text(f"{label}: 조회 실패")

        ui.timer(0.5, _load, once=True)


def render_tab1_market(df):
    fg_score, fg_label = get_fear_greed(df)

    # ── 매크로 스파크라인 배너 ──────────────────────────────
    _render_macro_sparklines()

    section_title("📡 시장 현황")
    with ui.row().classes("w-full gap-4 flex-wrap"):
        fg_icon = "🟢" if fg_score >= 50 else "🔴"
        metric_card("시장 심리", f"{fg_icon} {fg_label}", f"지수: {fg_score:.0f}/100", fg_score >= 50)

        if "ret_1d_%" in df.columns:
            avg_ret = df.head(20)["ret_1d_%"].mean()
            metric_card("Top20 평균 수익률", f"{avg_ret:+.2f}%", "전일 대비", avg_ret >= 0)

        total = len(df)
        armed = len(df[df.get("ROUTE", pd.Series()).str.contains("ARMED|ATTACK", na=False)]) if "ROUTE" in df.columns else 0
        metric_card("분석 종목", f"{total}개", f"ARMED/ATTACK: {armed}개")

    # 게이지 + 트리맵
    section_title("🌡️ 공포/탐욕 & 주도 섹터")
    with ui.row().classes("w-full gap-4 flex-wrap items-start"):
        with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
            fig_g = plot_fear_greed_gauge(fg_score)
            if fig_g:
                ui.plotly(_plotly_dark(fig_g, 280)).classes("w-full")

        with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
            ui.label("🔥 오늘의 주도 섹터").classes("text-sm font-bold text-white mb-2")
            if "업종" in df.columns:
                fig_m = plot_sector_treemap(df.head(50))
                if fig_m:
                    ui.plotly(_plotly_dark(fig_m, 280)).classes("w-full")
                else:
                    ui.label("섹터 데이터 부족").classes("text-gray-500")

    section_title("🚀 섹터 모멘텀 Top 10")
    fig_mom = plot_sector_momentum_bar(df)
    if fig_mom and len(fig_mom.data) > 0:
        ui.plotly(_plotly_dark(fig_mom, 350)).classes("w-full")
    else:
        ui.label("모멘텀 데이터 부족").classes("text-gray-500")


# ═══════════════════════════════════════════
#  Tab 2: 종목 분석
# ═══════════════════════════════════════════
def render_tab2_stocks(df, auth):
    section_title("🎯 AI & Quant 추천 종목")

    # ── 뷰모드 + 필터 + 액션 버튼 ──
    with ui.row().classes("w-full gap-4 items-center flex-wrap mb-4"):
        view_mode = ui.toggle(["📋 테이블", "🃏 칸반"], value="📋 테이블")
        route_filter = ui.select(["전체", "ATTACK", "ARMED", "WAIT", "NEUTRAL"], value="전체", label="상태").classes("min-w-[120px]")
        sort_mode = ui.toggle(["🔢 점수순", "🚦 상태순"], value="🔢 점수순")

        # ── 체크된 종목 → 매매일지 자동 추가 버튼 ──
        journal_btn_msg = ui.label("").classes("text-sm")

        def _add_checked_to_journal():
            """체크된 종목들을 매매일지에 자동 등록"""
            if not _selected_codes:
                ui.notify("⚠️ 종목을 먼저 체크하세요", type="warning")
                return
            if not JOURNAL_OK:
                ui.notify("⚠️ trade_journal_tab 모듈 없음", type="negative")
                return

            from trade_journal_tab import save_trade
            added = 0
            for code in _selected_codes:
                row = df[df["종목코드"].astype(str).str.zfill(6) == code]
                if row.empty:
                    continue
                r = row.iloc[0]
                tid = save_trade({
                    "stock_name":      str(r.get("종목명", "")),
                    "stock_code":      code,
                    "route":           str(r.get("ROUTE", "")),
                    "score":           safe_float(r.get("DISPLAY_SCORE", 0)),
                    "recommend_price": nz_num(r.get("추천매수가", 0)),
                    "actual_price":    nz_num(r.get("추천매수가", 0)),  # 체결가 = 추천매수가 (나중에 수정 가능)
                    "stop_price":      nz_num(r.get("손절가", 0)),
                    "target_price":    nz_num(r.get("추천매도가1", 0)),
                    "qty":             0,
                    "notes":           f"[자동등록] 점수 {safe_float(r.get('DISPLAY_SCORE', 0)):.0f} | {r.get('ROUTE', '')}",
                })
                if tid > 0:
                    added += 1

            if added > 0:
                ui.notify(f"✅ {added}건 매매일지에 추가 완료! (📓 매매 일지 탭에서 확인)", type="positive")
            else:
                ui.notify("❌ 추가 실패", type="negative")

        ui.button("📝 매매일지 추가", on_click=_add_checked_to_journal).props(
            "flat dense"
        ).classes("text-yellow-400 border border-yellow-700/50").tooltip(
            "체크한 종목들을 매매일지에 자동 등록합니다"
        )

        # CSV 다운로드 (Prime/Admin만)
        if auth in ("prime", "admin"):
            async def download_csv():
                csv_path = os.path.join(DATA_DIR, "recommend_latest.csv")
                if os.path.exists(csv_path):
                    ui.download(csv_path, f"ldy_recommend_{store.data_ts}.csv")
                else:
                    ui.notify("CSV 파일 없음", type="warning")
            ui.button("📥 CSV 다운로드", on_click=download_csv).props("flat dense").classes("text-green-400 ml-auto")

    table_area = ui.column().classes("w-full")
    detail_area = ui.column().classes("w-full mt-4")

    # ── 다중 선택 상태 추적 ──
    _selected_codes: list[str] = []

    def _filtered():
        fdf = df.copy()
        if route_filter.value != "전체" and "ROUTE" in fdf.columns:
            fdf = fdf[fdf["ROUTE"].astype(str).str.contains(route_filter.value, na=False)]
        if sort_mode.value == "🔢 점수순" and "DISPLAY_SCORE" in fdf.columns:
            fdf = fdf.sort_values("DISPLAY_SCORE", ascending=False)
        # 접근 제한
        if auth == "guest": fdf = fdf.head(3)
        elif auth == "free": fdf = fdf.head(5)
        elif auth == "pro": fdf = fdf.head(20)
        else: fdf = fdf.head(50)
        return fdf

    def _build_view():
        table_area.clear()
        _selected_codes.clear()
        show = _filtered()
        with table_area:
            if view_mode.value == "🃏 칸반":
                _render_kanban(show)
            else:
                _render_table(show)

    def _render_table(show):
        columns = [
            {"name": "route", "label": "상태", "field": "route", "align": "center"},
            {"name": "name", "label": "종목명", "field": "name", "align": "left"},
            {"name": "score", "label": "점수", "field": "score", "align": "center", "sortable": True},
            {"name": "close", "label": "현재가", "field": "close", "align": "right"},
            {"name": "buy", "label": "매수", "field": "buy", "align": "right"},
            {"name": "stop", "label": "손절", "field": "stop", "align": "right"},
            {"name": "t1", "label": "T1목표", "field": "t1", "align": "right"},
            {"name": "sector", "label": "업종", "field": "sector", "align": "left"},
        ]
        rows = []
        for _, r in show.iterrows():
            rows.append({
                "code": str(r.get("종목코드", "")).zfill(6),
                "route": str(r.get("ROUTE", "—")),
                "name": str(r.get("종목명", "—")),
                "score": f'{safe_float(r.get("DISPLAY_SCORE", 0)):.0f}',
                "close": f'{int(nz_num(r.get("종가", 0))):,}',
                "buy": f'{int(nz_num(r.get("추천매수가", 0))):,}',
                "stop": f'{int(nz_num(r.get("손절가", 0))):,}',
                "t1": f'{int(nz_num(r.get("추천매도가1", 0))):,}',
                "sector": str(r.get("업종", "—")),
            })
        tbl = ui.table(columns=columns, rows=rows, row_key="code", selection="multiple", pagination={"rowsPerPage": 15}).classes("w-full").props("dense dark flat bordered")

        def _on_selection_change(e):
            sel_rows = e.args.get("rows", [])
            _selected_codes.clear()
            _selected_codes.extend([r.get("code", "") for r in sel_rows])
            # 마지막 선택된 종목 상세 표시 (단일 클릭 시)
            if sel_rows:
                _on_stock_select_by_code(sel_rows[-1].get("code", ""), df)

        tbl.on("selection", _on_selection_change)

    def _render_kanban(show):
        if show.empty:
            ui.label("표시할 종목 없음").classes("text-gray-400")
            return
        df_atk = show[show['ROUTE'].astype(str).str.contains("ATTACK", case=False, na=False)] if "ROUTE" in show.columns else pd.DataFrame()
        df_arm = show[show['ROUTE'].astype(str).str.contains("ARMED", case=False, na=False)] if "ROUTE" in show.columns else pd.DataFrame()
        ex = df_atk.index.union(df_arm.index) if not df_atk.empty or not df_arm.empty else pd.Index([])
        df_watch = show[~show.index.isin(ex)]

        with ui.row().classes("w-full gap-4 flex-wrap items-start"):
            _kanban_col("🚀 ATTACK", df_atk, "#EF4444", df)
            _kanban_col("🔫 ARMED", df_arm, "#F59E0B", df)
            _kanban_col("👀 WATCH", df_watch, "#3B82F6", df)

    def _kanban_col(title, sub_df, color, full_df):
        with ui.column().classes("kanban-col min-w-[280px] flex-1"):
            ui.label(f"{title} ({len(sub_df)})").classes("text-white font-bold mb-2").style(f"border-bottom: 2px solid {color}")
            if sub_df.empty:
                ui.label("비어 있음").classes("text-gray-500 text-sm")
                return
            for _, r in sub_df.iterrows():
                score = safe_float(r.get("DISPLAY_SCORE", 0))
                buy = int(nz_num(r.get("추천매수가", 0)))
                stop = int(nz_num(r.get("손절가", 0)))
                t1 = int(nz_num(r.get("추천매도가1", 0)))
                sc = "#10B981" if score >= 80 else "#3B82F6" if score >= 60 else "#94A3B8"
                code = str(r.get("종목코드", "")).zfill(6)

                with ui.card().classes("p-3 mb-2 cursor-pointer bg-[rgba(255,255,255,0.05)] border border-[rgba(255,255,255,0.1)] rounded-xl hover:bg-[rgba(255,255,255,0.08)]"):
                    with ui.row().classes("justify-between items-center"):
                        ui.label(f"{r.get('종목명', '')}").classes("text-white font-bold text-sm")
                        ui.badge(f"{score:.0f}", color=sc).classes("text-xs")
                    if buy > 0:
                        ui.label(f"🎯 {buy:,}  🛡️ {stop:,}  🟢 {t1:,}").classes("text-xs text-gray-400 mt-1")
                    rr = safe_float(r.get("RR1", 0))
                    if rr > 0:
                        rc = "#10B981" if rr >= 2 else "#F59E0B" if rr >= 1 else "#EF4444"
                        ui.label(f"R:R {rr:.1f}").classes("text-xs mt-1").style(f"color:{rc}")

    def _on_stock_select_by_code(code, full_df):
        detail_area.clear()
        if not code: return
        row = full_df[full_df["종목코드"].astype(str).str.zfill(6) == code]
        if row.empty: return
        row = row.iloc[0]

        with detail_area:
            section_title(f"🔍 {row.get('종목명', '')} ({code}) 상세 분석")

            _close = nz_num(row.get("종가", 0))
            _entry = nz_num(row.get("추천매수가", 0))
            _stop = nz_num(row.get("손절가", 0))
            _t1 = nz_num(row.get("추천매도가1", 0))
            _t2 = nz_num(row.get("추천매도가2", 0))
            _atr = nz_num(row.get("TARGET_ATR", 0))

            # 목표가 카드
            if _close > 0 and _entry > 0:
                risk = _entry - _stop if _stop > 0 else 1
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    metric_card("🔴 손절가", f"{int(_stop):,}", f"{(_stop/_close-1)*100:+.1f}%" if _close > 0 else "", False)
                    metric_card("🔵 매수가", f"{int(_entry):,}", "시스템 추천")
                    if _t1 > 0:
                        rr1 = (_t1 - _entry) / risk if risk > 0 else 0
                        metric_card("🟢 T1 목표", f"{int(_t1):,}", f"+{(_t1/_close-1)*100:.1f}% (RR {rr1:.1f}:1)")
                    if _t2 > 0 and _t2 != _t1:
                        rr2 = (_t2 - _entry) / risk if risk > 0 else 0
                        metric_card("🟡 T2 목표", f"{int(_t2):,}", f"+{(_t2/_close-1)*100:.1f}% (RR {rr2:.1f}:1)")
                    if _atr > 0 and _atr != _t1:
                        metric_card("⚪ ATR 목표", f"{int(_atr):,}", f"+{(_atr/_close-1)*100:.1f}%")

            # 가격 바
            if _close > 0 and _stop > 0 and _t1 > 0:
                ui.html(price_bar_html(_stop, _entry, _close, _t1, _t2))

            # 캔들차트
            with ui.card().classes("w-full p-2 bg-[#1a1a2e] mt-2"):
                ui.label("🕯️ 캔들차트 로딩 중...").classes("text-gray-400")
                chart_holder = ui.column().classes("w-full")

                async def load_chart():
                    chart_holder.clear()
                    with chart_holder:
                        cdata = await run_sync(get_stock_chart_data, code)
                        if cdata is not None:
                            fig = plot_candle_chart(cdata, code, row.get("종목명", ""), _entry, _stop, _t1, _t2)
                            ui.plotly(fig).classes("w-full")
                        else:
                            ui.label("📉 차트 데이터 로드 실패 (FDR 미설치 또는 네트워크 오류)").classes("text-yellow-400")
                ui.timer(0.1, load_chart, once=True)

            # 레이더 + 워터폴
            with ui.row().classes("w-full gap-4 flex-wrap mt-4"):
                with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
                    try:
                        fig_r = plot_radar_chart(row)
                        if fig_r: ui.plotly(_plotly_dark(fig_r, 300)).classes("w-full")
                    except Exception:
                        ui.label("레이더 차트 오류").classes("text-gray-500")
                with ui.card().classes("flex-1 min-w-[300px] p-2 bg-[#1a1a2e]"):
                    try:
                        fig_w = plot_score_waterfall(row)
                        if fig_w: ui.plotly(_plotly_dark(fig_w, 300)).classes("w-full")
                    except Exception:
                        ui.label("워터폴 차트 오류").classes("text-gray-500")

            # ROUTE 배지
            rv = str(row.get("ROUTE", "NEUTRAL"))
            rc = {"ATTACK": "#EF4444", "ARMED": "#F59E0B", "WAIT": "#3B82F6", "NEUTRAL": "#6B7280"}.get(rv, "#6B7280")
            ui.label(f"⚡ 현재 상태: {rv}").classes("text-lg font-bold mt-4 px-4 py-2 rounded-lg text-white").style(f"background:{rc}")

            # ── Kelly 포지션 사이저 (종목별) ─────────────────
            if KELLY_OK:
                kelly_holder = ui.card().classes(
                    "w-full p-4 bg-[#1a1a2e] border border-yellow-700/40 rounded-xl mt-4"
                )
                render_kelly_calculator(row.to_dict(), kelly_holder)

    # 이벤트 바인딩
    for w in [view_mode, route_filter, sort_mode]:
        w.on("update:model-value", lambda _: _build_view())
    _build_view()


# ═══════════════════════════════════════════
#  Tab 3: 내 자산
# ═══════════════════════════════════════════
def render_tab3_portfolio(df, auth):
    if auth in ("guest", "free"):
        ui.label("🔒 내 자산 분석은 Pro 등급부터 이용 가능합니다.").classes("text-yellow-400 p-8")
        ui.label(f"🚀 Pro 이용권 ({PRICE_PRO:,}원/월) · 👑 Prime 이용권 ({PRICE_PRIME:,}원/월)").classes("text-gray-400")
        return

    section_title("💼 내 자산: AI 리밸런싱 & 진단")
    ui.label("👇 보유 종목을 입력하세요 (종목명:평단가:수량)").classes("text-xs text-gray-400 mb-2")

    # ── 포트폴리오 로드 우선순위: ①브라우저 저장 → ②Gist → ③기본값 ──
    saved_local = app.storage.user.get("portfolio_text", "")
    saved_gist = load_portfolio_file() if not saved_local else ""
    saved = saved_local or saved_gist or ""

    pf_input = ui.textarea("포트폴리오 입력", value=saved,
                           placeholder="종목명:평단가:수량 (줄바꿈 구분)\n예) 에코프로머티:67341:60").classes("w-full").props("rows=6")
    result_area = ui.column().classes("w-full mt-4")

    def _auto_save():
        """입력 변경 시 브라우저 스토리지에 자동 저장 (새로고침해도 유지)"""
        app.storage.user["portfolio_text"] = pf_input.value

    pf_input.on("blur", lambda _: _auto_save())  # 포커스 벗어날 때 자동 저장

    async def analyze():
        result_area.clear()
        text = pf_input.value.strip()
        if not text: return

        # 브라우저 + Gist 이중 저장
        app.storage.user["portfolio_text"] = text
        await run_sync(save_portfolio_file, text)
        ui.notify("💾 포트폴리오 저장됨", type="positive")

        code_map = get_code_map(df)
        targets = []
        cash_amt = 0.0

        for line in text.split("\n"):
            if ":" not in line: continue
            parts = line.split(":")
            if len(parts) < 3: continue
            try:
                nm = parts[0].strip()
                price = int(float(parts[1].replace(",", "").strip()))
                qty = int(float(parts[2].replace(",", "").strip()))
            except (ValueError, TypeError):
                continue
            if nm.upper() == "CASH" or "현금" in nm:
                cash_amt += price * qty
            else:
                real_code = find_code_by_name(nm, code_map) or nm
                targets.append((real_code, nm, price, qty))

        if not targets and cash_amt <= 0:
            with result_area:
                ui.label("입력된 종목이 없습니다.").classes("text-gray-400")
            return

        with result_area:
            ui.label("⚡ 시세 조회 중...").classes("text-gray-400")

        # ── 비동기 현재가 조회 (asyncio.gather) ──
        price_map = {}
        if PRICE_CACHE_OK and FDR_OK:
            try:
                price_results = await fetch_prices_async(
                    [(t[0], t[1]) for t in targets], fdr
                )
                price_map = price_results
            except Exception as _ae:
                logger.warning(f"async 조회 실패, ThreadPool fallback: {_ae}")

        # fallback: ThreadPoolExecutor
        if not price_map:
            def _fetch_all_prices():
                _pm = {}
                with ThreadPoolExecutor(max_workers=8) as ex:
                    futs = [ex.submit(fetch_current_price, t[0], t[1]) for t in targets]
                    for f in futs:
                        c, n, p = f.result()
                        _pm[c] = p
                return _pm
            price_map = await run_sync(_fetch_all_prices)

        total_eval = total_buy = 0.0
        pf_rows = []
        for code, name, avg, qty in targets:
            curr = price_map.get(code, 0)

            # 현재가 0이면 scored에 종가가 있는지 폴백
            if curr == 0 and not df.empty and '종가' in df.columns:
                match_p = df[df['종목코드'] == str(code).zfill(6)] if '종목코드' in df.columns else pd.DataFrame()
                if match_p.empty and '종목명' in df.columns:
                    match_p = df[df['종목명'] == name]
                if not match_p.empty:
                    curr = int(nz_num(match_p.iloc[0].get('종가', 0)))

            eval_amt = curr * qty
            buy_amt = avg * qty
            total_eval += eval_amt
            total_buy += buy_amt
            pct = (curr - avg) / avg * 100 if avg > 0 and curr > 0 else 0

            # AI 점수 매핑
            match = df[df['종목코드'] == str(code).zfill(6)] if not df.empty and '종목코드' in df.columns else pd.DataFrame()
            if match.empty and not df.empty and '종목명' in df.columns:
                match = df[df['종목명'] == name]
            score = float(match.iloc[0].get('DISPLAY_SCORE', 0)) if not match.empty else 0

            if score >= 80: advice, acolor = "💪강력홀딩", "#10B981"
            elif score >= 60: advice, acolor = "👌보유(양호)", "#3B82F6"
            elif score <= 40 and score > 0: advice, acolor = "⚠️교체권장", "#EF4444"
            elif score == 0 and curr == 0: advice, acolor = "❓시세조회 실패", "#EF4444"
            elif score == 0: advice, acolor = "❓시스템 외 종목", "#9CA3AF"
            else: advice, acolor = "👀관망", "#F59E0B"

            pf_rows.append({"종목명": name, "현재가": curr, "평단가": avg, "수량": qty,
                            "매입금": buy_amt, "평가금": eval_amt, "수익률": pct,
                            "점수": score, "AI조언": advice, "색상": acolor, "code": code})

        result_area.clear()
        with result_area:
            total_asset = total_eval + cash_amt
            total_invest = total_buy + cash_amt
            total_rate = (total_asset - total_invest) / total_invest * 100 if total_invest > 0 else 0

            with ui.row().classes("w-full gap-4 flex-wrap"):
                metric_card("총 평가금액", f"{int(total_asset):,}원")
                metric_card("총 매입금액", f"{int(total_invest):,}원")
                metric_card("총 평가손익", f"{int(total_asset - total_invest):+,}원", f"{total_rate:+.2f}%", total_rate >= 0)
                if cash_amt > 0:
                    metric_card("현금 비중", f"{cash_amt/total_asset*100:.1f}%" if total_asset > 0 else "0%", f"{int(cash_amt):,}원")

            # 종목별 카드
            section_title("🩺 AI 포트폴리오 진단")
            pf_rows.sort(key=lambda x: x["점수"])
            for r in pf_rows:
                with ui.card().classes("w-full p-4 mb-2 bg-[#1a1a2e] border border-gray-700 rounded-xl"):
                    with ui.row().classes("w-full justify-between items-center"):
                        with ui.column().classes("gap-0"):
                            ui.label(r["종목명"]).classes("text-white font-bold")
                            p_color = "text-red-400" if r["수익률"] > 0 else "text-blue-400"
                            ui.label(f"{r['수익률']:+.2f}%  |  평가금: {int(r['평가금']):,}원").classes(f"text-sm {p_color}")
                        with ui.column().classes("items-end gap-0"):
                            ui.label(r["AI조언"]).classes(f"text-sm font-bold").style(f"color:{r['색상']}")
                            if r["점수"] > 0:
                                ui.label(f"점수: {r['점수']:.0f}").classes("text-xs text-gray-400")

            # 파이 차트
            if pf_rows:
                pie_data = pf_rows.copy()
                if cash_amt > 0:
                    pie_data.append({"종목명": "현금", "평가금": cash_amt})
                fig = px.pie(pd.DataFrame(pie_data), values="평가금", names="종목명", title="📊 자산 구성", hole=0.4)
                ui.plotly(_plotly_dark(fig, 300)).classes("w-full")

            # ── Kelly 포지션 사이저 ──────────────────────────
            if KELLY_OK and pf_rows:
                kelly_section = ui.card().classes("w-full p-4 bg-[#1a1a2e] border border-yellow-700/40 rounded-xl mt-4")
                render_portfolio_kelly_summary(pf_rows, total_eval, kelly_section)

    ui.button("🤖 AI 진단 실행", on_click=analyze).classes("mt-4").props("color=primary")


# ═══════════════════════════════════════════
#  Tab 4: 문의 게시판
# ═══════════════════════════════════════════
def render_tab4_inquiry(auth, user):
    section_title("📮 문의 게시판")

    d_nick = user.get("nickname", "") if user else ""
    d_email = user.get("login_id", "") if user else ""

    ui.label("✏️ 문의 작성").classes("text-white font-bold mt-4")
    with ui.row().classes("w-full gap-4"):
        nick_in = ui.input("닉네임", value=d_nick).classes("flex-1")
        email_in = ui.input("이메일 (선택)", value=d_email).classes("flex-1")
    title_in = ui.input("제목", placeholder="문의 제목").classes("w-full")
    content_in = ui.textarea("내용", placeholder="자유롭게 남겨주세요.").classes("w-full").props("rows=5")
    inq_list = ui.column().classes("w-full mt-4")

    async def submit_inquiry():
        if not title_in.value.strip() or not content_in.value.strip():
            ui.notify("제목과 내용을 입력하세요.", type="warning"); return
        items = load_inquiry_items()
        items.append({
            "title": title_in.value.strip(), "content": content_in.value.strip(),
            "nickname": nick_in.value.strip() or "익명", "email": email_in.value.strip(),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })
        if save_inquiry_items(items):
            ui.notify("💌 문의 등록 완료!", type="positive")
            title_in.value = ""; content_in.value = ""
            _load_inquiries()
        else:
            ui.notify("등록 실패", type="negative")

    ui.button("💌 문의 등록", on_click=submit_inquiry).classes("mt-2").props("color=primary")

    def _load_inquiries():
        inq_list.clear()
        items = load_inquiry_items()
        with inq_list:
            ui.label("📂 최근 문의 내역").classes("text-white font-bold mt-4")
            if not items:
                ui.label("등록된 문의가 없습니다.").classes("text-gray-400")
                return
            for i, item in enumerate(reversed(items[-30:])):
                with ui.card().classes("w-full p-3 mb-2 bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                    with ui.row().classes("justify-between"):
                        ui.label(f"📌 {item.get('title', '-')}").classes("text-white font-bold text-sm")
                        if auth == "admin":
                            ui.button("🗑️", on_click=lambda it=item: _del_inquiry(it)).props("flat dense size=sm")
                    ui.label(item.get("content", "")).classes("text-gray-300 text-sm mt-1")
                    meta = f"{item.get('nickname', '익명')} · {to_kst_str(item.get('created_at'))}"
                    ui.label(meta).classes("text-xs text-gray-500 mt-1")

    def _del_inquiry(item):
        items = [x for x in load_inquiry_items() if x.get("created_at") != item.get("created_at")]
        save_inquiry_items(items)
        ui.notify("삭제됨"); _load_inquiries()

    _load_inquiries()


# ═══════════════════════════════════════════
#  Tab 5: 이용 약관
# ═══════════════════════════════════════════
def render_tab5_terms():
    section_title("⚖️ 이용 약관 / 투자 유의사항")
    terms_md = f"""
### 1. 서비스 성격
LDY Pro Trader는 **퀀트 지표 기반 데이터 분석 도구**로, 매수·매도·수익을 보장하는 리딩 서비스가 아닙니다.

### 2. 투자 책임
최종 투자 의사결정은 **전적으로 이용자 본인**의 판단이며, 손익은 모두 본인에게 귀속됩니다.

### 3. 데이터 한계
시장 데이터는 지연·오류·누락이 발생할 수 있으며, 과거 기반 지표는 미래와 괴리가 발생할 수 있습니다.

### 4. 이용권 정책
- **Guest**: 상위 3개 맛보기
- **Free**: 상위 5개 열람
- **Pro ({PRICE_PRO:,}원/월)**: 상위 20 종목 + 내 자산 분석
- **Prime ({PRICE_PRIME:,}원/월)**: 전체 종목 + CSV + 텔레그램

### 5. 한 줄 요약
> 👉 데이터와 퀀트는 도구일 뿐, 최종 책임은 언제나 본인에게 있다.
"""
    ui.markdown(terms_md).classes("text-gray-300")


# ═══════════════════════════════════════════
#  Tab 6: 업데이트 노트
# ═══════════════════════════════════════════
def render_tab6_updates():
    section_title("🧩 업데이트 노트")
    if not CHANGELOG:
        ui.label("등록된 업데이트 기록이 없습니다.").classes("text-gray-400")
        return

    latest = CHANGELOG[0]
    with ui.card().classes("w-full p-4 bg-green-900/30 border border-green-700 rounded-xl mb-4"):
        ui.label(f"현재 버전: v{APP_VERSION}").classes("text-green-400 font-bold")
        ui.label(f"최근: {latest.get('date')} · {latest.get('title')}").classes("text-green-300 text-sm")

    for i, log in enumerate(CHANGELOG):
        expanded = (i == 0)
        with ui.expansion(
            f"{'⭐ ' if i == 0 else ''}v{log['version']} · {log['date']} — {log['title']}",
            value=expanded
        ).classes("w-full mb-1"):
            for item in log.get("items", []):
                ui.label(f"• {item}").classes("text-gray-300 text-sm ml-4")


# ═══════════════════════════════════════════
#  Tab 7: 시스템 성과
# ═══════════════════════════════════════════
def render_tab7_performance():
    section_title("📈 시스템 성과 추세")

    pattern = os.path.join(DATA_DIR, "rank_validation_summary_*.csv")
    all_files = sorted(glob.glob(pattern))

    # 날짜별 파일 + latest 파일 모두 사용
    dfs = []
    for f in all_files:
        try:
            base = os.path.basename(f)
            ds = base.replace("rank_validation_summary_", "").replace(".csv", "")
            d = pd.read_csv(f)
            if "latest" in ds:
                # latest 파일은 오늘 날짜로 처리
                d['Date'] = pd.to_datetime(now_kst().strftime("%Y-%m-%d"))
            else:
                d['Date'] = pd.to_datetime(ds, format="%Y%m%d")
            dfs.append(d)
        except Exception:
            pass

    if not dfs:
        ui.label("축적된 성과 데이터가 부족합니다.").classes("text-gray-400 p-8"); return

    history = pd.concat(dfs, ignore_index=True).sort_values('Date')

    col_win, col_ret = 'WIN_RATE_%', 'AVG_RET_%'
    if col_win not in history.columns or col_ret not in history.columns:
        ui.label("필요 컬럼 없음").classes("text-gray-400"); return

    # 필터
    with ui.row().classes("w-full gap-4 flex-wrap mb-4"):
        methods = sorted(history['METHOD'].unique()) if 'METHOD' in history.columns else []
        def_m = 'RANK_SCORE' if 'RANK_SCORE' in methods else (methods[0] if methods else None)
        sel_m = ui.select(methods, value=def_m, label="Method").classes("min-w-[150px]") if methods else None

        topks = sorted(history['TOPK'].unique().tolist()) if 'TOPK' in history.columns else []
        def_k = 5 if 5 in topks else (topks[0] if topks else None)
        sel_k = ui.select([str(k) for k in topks], value=str(def_k), label="Top K").classes("min-w-[100px]") if topks else None

        holds = sorted(history['H(영업일)'].unique().tolist()) if 'H(영업일)' in history.columns else []
        def_h = 5 if 5 in holds else (holds[0] if holds else None)
        sel_h = ui.select([str(h) for h in holds], value=str(def_h), label="보유기간").classes("min-w-[100px]") if holds else None

    chart_area = ui.column().classes("w-full")

    def _build_chart():
        chart_area.clear()
        cdf = history.copy()
        if sel_m and sel_m.value: cdf = cdf[cdf['METHOD'] == sel_m.value]
        if sel_k and sel_k.value: cdf = cdf[cdf['TOPK'] == int(sel_k.value)]
        if sel_h and sel_h.value: cdf = cdf[cdf['H(영업일)'] == int(sel_h.value)]
        cdf = cdf.sort_values('Date').tail(30)

        with chart_area:
            if cdf.empty:
                ui.label("조건 맞는 데이터 없음").classes("text-gray-400"); return

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=cdf['Date'], y=cdf[col_win], name="승률(%)", marker_color='#FFA726', opacity=0.6), secondary_y=False)
            fig.add_trace(go.Scatter(x=cdf['Date'], y=cdf[col_ret], name="수익률(%)", mode='lines+markers', line=dict(color='#29B6F6', width=3)), secondary_y=True)
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', hovermode="x unified", legend=dict(orientation="h", y=1.1))
            fig.update_yaxes(title_text="승률(%)", range=[0, 100], secondary_y=False)
            fig.update_yaxes(title_text="수익률(%)", secondary_y=True)
            ui.plotly(fig).classes("w-full")

            with ui.row().classes("w-full gap-4 mt-2"):
                metric_card("평균 승률", f"{cdf[col_win].mean():.1f}%")
                metric_card("평균 수익률", f"{cdf[col_ret].mean():.2f}%")

    for w in [sel_m, sel_k, sel_h]:
        if w: w.on("update:model-value", lambda _: _build_chart())
    _build_chart()

    # ── [Phase 1-5] Research Workbench 확장 ──
    try:
        from research_tab import render_research_tab
        ui.separator().classes("my-6")
        render_research_tab(data_dir=DATA_DIR)
    except Exception as _rt_err:
        ui.label(f"⚠️ Research 탭 로드 실패: {_rt_err}").classes("text-gray-400")


# ═══════════════════════════════════════════
#  Tab 8: 회원 관리 (Admin)
# ═══════════════════════════════════════════
def render_tab8_admin():
    section_title("👑 회원 관리")
    db = _get_db()
    if not db:
        ui.label("❌ DB 연결 실패").classes("text-red-400"); return

    users = db.get_all_users()
    if not users:
        ui.label("등록된 회원 없음").classes("text-gray-400"); return

    ui.label(f"👥 총 가입자: {len(users)}명").classes("text-white mb-4")

    columns = [
        {"name": "email", "label": "이메일", "field": "email", "align": "left"},
        {"name": "nick", "label": "닉네임", "field": "nick"},
        {"name": "role", "label": "권한", "field": "role"},
        {"name": "status", "label": "상태", "field": "status"},
        {"name": "joined", "label": "가입일", "field": "joined"},
        {"name": "last", "label": "최근접속", "field": "last"},
    ]
    rows = []
    for u in users:
        rows.append({
            "email": u.get("login_id") or u.get("id", ""),
            "nick": u.get("nickname", ""),
            "role": u.get("role", "free").upper(),
            "status": "🚫차단" if u.get("is_banned") else "✅",
            "joined": to_kst_str(u.get("join_date"), "%Y-%m-%d"),
            "last": to_kst_str(u.get("last_login")),
        })

    ui.table(columns=columns, rows=rows, row_key="email", pagination={"rowsPerPage": 20}).classes("w-full").props("dense dark flat bordered")

    # 관리자 액션
    ui.separator().classes("my-4")
    with ui.row().classes("w-full gap-8 flex-wrap"):
        with ui.column().classes("flex-1"):
            ui.label("🛠️ 개별 회원 제어").classes("text-white font-bold mb-2")
            emails = [r["email"] for r in rows]
            sel_email = ui.select(emails, label="회원 선택").classes("w-full")
            sel_role = ui.select(["free", "pro", "prime", "admin"], label="등급 변경", value="free").classes("w-full")

            async def apply_role():
                if sel_email.value:
                    db = _get_db()
                    if db:
                        ok = db.update_user_role(sel_email.value, sel_role.value)
                        ui.notify(f"{'✅ 변경 완료' if ok else '❌ 실패'}")
                    else:
                        ui.notify("DB 연결 실패", type="negative")

            async def toggle_ban():
                if sel_email.value:
                    db = _get_db()
                    if db:
                        ok, msg = db.toggle_user_ban(sel_email.value)
                        ui.notify(msg)
                    else:
                        ui.notify("DB 연결 실패", type="negative")

            with ui.row().classes("gap-2 mt-2"):
                ui.button("등급 적용", on_click=apply_role).props("color=primary")
                ui.button("🚫 차단/해제", on_click=toggle_ban).props("color=negative")

        with ui.column().classes("flex-1"):
            ui.label("🎉 전체 이벤트").classes("text-white font-bold mb-2")
            ui.label("전 회원에게 체험권을 지급합니다.").classes("text-gray-400 text-sm mb-2")

            async def grant_trial():
                db = _get_db()
                if db:
                    ok, msg = db.grant_all_users_trial(7)
                    ui.notify(f"{'🎁 ' + msg if ok else '❌ ' + msg}")
                else:
                    ui.notify("DB 연결 실패", type="negative")

            ui.button("🎁 전원 7일 Prime 지급", on_click=grant_trial).props("color=positive")


# ═══════════════════════════════════════════
#  앱 실행
# ═══════════════════════════════════════════
# PWA 정적 파일 서빙
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.add_static_files("/static", STATIC_DIR)

if __name__ in {"__main__", "__mp_main__"}:
    store.refresh()
    register_shutdown(app)  # [v6.0] 재배포 시 스레드 풀 + DB + Gist 최종 flush
    ui.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        title=f"LDY Pro Trader v{APP_VERSION}",
        favicon="💎",
        dark=True,
        storage_secret=os.environ["STORAGE_SECRET"],  # [v2.0 #3] 강제 — 미설정 시 앱 시작 차단
        reload=False,
        show=False,
    )
