# -*- coding: utf-8 -*-
"""
LDY Pro Trader Dashboard v8.0 (Macro View & SuperTrend Chart)
- v7.5: 7-Factor 레이더 차트, 스마트 손절/매수세(V-Power) 시각화
- v7.0: 팩터 기반 분석, 스퀴즈 지속일(CNT) 표시, 켈트너 채널
"""

# ---------------------------
# import
# ---------------------------
import os, io, math, json, requests, logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import re
from typing import Optional, Dict, Any, Tuple

def normalize_code(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)      # 660.0 같은 거 제거
    s = re.sub(r"[^0-9]", "", s)    # 숫자만 남김
    return s.zfill(6) if s else ""  # 6자리로

# -------------------- [v9.0 유틸리티 추가] --------------------
def wma(s: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    def _calc(x):
        return np.dot(x, weights) / weights.sum()
    return s.rolling(period).apply(_calc, raw=True)

def calc_hma_series(s: pd.Series, period: int) -> pd.Series:
    """차트용 HMA 시리즈 계산"""
    if len(s) < period:
        return pd.Series(np.nan, index=s.index)
    half_length = int(period / 2)
    sqrt_length = int(math.sqrt(period))
    wma_half = wma(s, half_length)
    wma_full = wma(s, period)
    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt_length)
# -----------------------------------------------------------

def postprocess_codes(df: pd.DataFrame) -> pd.DataFrame:
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].apply(normalize_code)
    return df

# -----------------------------------------------------------
# [주의] 아래 import 구문들은 맨 앞줄(들여쓰기 없음)에 있어야 합니다.
# -----------------------------------------------------------

from auth_user import (
    render_auth_box, get_user, list_users, update_user_role,
    load_inquiry_items, save_inquiry_items, _now_utc_str,
    load_subscriptions_db, save_subscriptions_db,  # 👈 여기에 쉼표(,)가 꼭 있어야 합니다!
    toggle_user_ban
)
from plotly.subplots import make_subplots
from version_info import (
    PRIME_TG_JOIN_URL,
    APP_VERSION,
    CHANGELOG,
    get_version_label,
    get_latest_log,
)


# ---------------------------
# 로깅 설정
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ldy")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("LDY_DATA_DIR", os.path.join(BASE_DIR, "data"))
RECOMMEND_LATEST_PATH = os.path.join(DATA_DIR, "recommend_latest.csv")
REALITY_LATEST_PATH   = os.path.join(DATA_DIR, "reality_check_latest.csv")
RANKVAL_SUMM_LATEST   = os.path.join(DATA_DIR, "rank_validation_summary_latest.csv")
RANKVAL_DETAIL_LATEST = os.path.join(DATA_DIR, "rank_validation_latest.csv")
PRICE_SNAP_LATEST     = os.path.join(DATA_DIR, "price_snapshot_latest.csv")
SECTOR_KRX_CACHE      = os.path.join(DATA_DIR, "sector_map_krx.csv")
SECTOR_FDR_CACHE      = os.path.join(DATA_DIR, "sector_map_fdr_v2.csv")
os.makedirs(DATA_DIR, exist_ok=True)
REMOTE_RECOMMEND_URL = os.getenv("LDY_REMOTE_RECOMMEND_URL", "")



# ---------------------------
# [수정됨] 구독/권한(만료일) 관리 - Gist 연동
# ---------------------------
# (기존의 SUBS_DB_PATH 정의나 파일 open 코드는 모두 제거됨)

def load_subs_db():
    """auth_user.py의 Gist 로드 함수 사용 (로컬 파일 X)"""
    return load_subscriptions_db()

def save_subs_db(db):
    """auth_user.py의 Gist 저장 함수 사용 (로컬 파일 X)"""
    return save_subscriptions_db(db)

def set_subscription(email: str, role: str, days: int = 30):
    email = (email or "").strip()
    if not email:
        return

    db = load_subs_db()
    subs = db.get("subs", {})

    role = (role or "").lower().strip()

    # ✅ free/guest/빈값이면 아예 삭제
    if role in ("free", "guest", ""):
        subs.pop(email, None)
        db["subs"] = subs
        save_subs_db(db)
        return

    # ✅ admin은 만료 없음
    if role == "admin":
        subs[email] = {"role": "admin", "expire_at": "", "paid_at": ""}
        db["subs"] = subs
        save_subs_db(db)
        return

    # ✅ pro/prime만 만료일 유지
    today = now_kst().date()
    expire = today + timedelta(days=days)
    subs[email] = {
        "role": role,
        "paid_at": today.strftime("%Y-%m-%d"),
        "expire_at": expire.strftime("%Y-%m-%d"),
    }
    db["subs"] = subs
    save_subs_db(db)

def get_subscription(email):
    """이메일 기준 구독 정보 조회"""
    email = (email or "").strip()
    if not email:
        return None
    db = load_subs_db()
    return db.get("subs", {}).get(email)

# ---------------------------
# 시간 / 타임존 유틸 (UTC 저장 + KST 표기)
# ---------------------------
KST = timezone(timedelta(hours=9))

def now_utc() -> datetime:
    """DB/파일 저장용: 항상 UTC 기준 aware datetime"""
    return datetime.now(timezone.utc)

def now_kst() -> datetime:
    """화면/로그 표시용: 한국 시간(KST) 기준 aware datetime"""
    return datetime.now(KST)

def to_kst_str(value, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if value is None or value == "" or value == "NaT":
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""

    # 🔹 말도 안 되는 옛날 날짜(예: 1970년)는 버리기
    try:
        if ts.year < 2000:
            return ""
    except Exception:
        pass

    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc).tz_convert(KST)
    else:
        ts = ts.tz_convert(KST)

    return ts.strftime(fmt)

# =========================
# ✅ Loader / Cache Utils (Local 우선 → Remote fallback)
# - 붙여넣기 위치: def to_kst_str(...) 함수 "끝난 직후"
# =========================

def _mtime(path: str) -> int:
    try:
        return int(os.path.getmtime(path))
    except Exception:
        return 0

def _normalize_github_raw(url: str) -> str:
    if not isinstance(url, str):
        return ""
    u = url.strip()
    if not u:
        return ""
    if "github.com/" in u and "/blob/" in u:
        u = u.replace("https://github.com/", "https://raw.githubusercontent.com/")
        u = u.replace("/blob/", "/")
    return u

def _download_bytes(url: str, timeout: int = 30) -> bytes:
    u = _normalize_github_raw(url)
    if not u:
        raise ValueError("REMOTE url is empty")
    r = requests.get(
        u,
        timeout=timeout,
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
    )
    r.raise_for_status()
    return r.content

def _read_csv_bytes(b: bytes, enc: str = "utf-8-sig") -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(b), encoding=enc)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(b), encoding="utf-8")

def _read_csv_file(path: str, enc: str = "utf-8-sig") -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=enc)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")

@st.cache_data(ttl=600)
def load_csv_url(url: str) -> pd.DataFrame:
    url = normalize_github_raw(url)
    r = requests.get(
        url,
        timeout=30,
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
    )
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig")


def _atomic_write_bytes(path: str, b: bytes) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, path)

def _safe_read_csv(path: str, enc: str = "utf-8-sig", remote_url: str = "") -> pd.DataFrame:
    """
    ✅ 로컬 우선 → 원격 fallback
    - 로컬 파일이 있으면 그걸 먼저 읽는다
    - 로컬이 없거나/읽기 실패 시 remote_url(또는 REMOTE_RECOMMEND_URL)에서 다운로드
    - 원격 성공 시 로컬(path)에 저장해서 mtime 갱신 → cache 무효화 자동 유도
    """
    last_err = None

    # 1) Local first
    if path and os.path.exists(path):
        try:
            return _read_csv_file(path, enc=enc)
        except Exception as e:
            last_err = e

    # 2) Remote fallback
    url = (remote_url or "").strip()
    if not url:
        url = (REMOTE_RECOMMEND_URL or "").strip()

    if url:
        try:
            b = _download_bytes(url, timeout=30)
            df = _read_csv_bytes(b, enc=enc)
            if path:
                try:
                    _atomic_write_bytes(path, b)
                except Exception:
                    logger.exception("remote csv downloaded but local save failed: %s", path)
            return df
        except Exception as e:
            last_err = e

    # 3) Fail
    if last_err is not None:
        raise RuntimeError(f"_safe_read_csv failed (path={path}, url={url}): {last_err}") from last_err
    raise RuntimeError(f"_safe_read_csv failed (path={path}, url={url})")

@st.cache_data(ttl=600, show_spinner=False)
def _load_csv_cached(path: str, enc: str, remote_url: str, mtime_sig: int) -> pd.DataFrame:
    # mtime_sig는 "캐시 키" 역할. 파일이 바뀌면 자동으로 캐시 무효화됨.
    return _safe_read_csv(path=path, enc=enc, remote_url=remote_url)

def load_recommend_latest(local_path: str = None, remote_url: str = "") -> pd.DataFrame:
    """
    recommend_latest.csv 로드
    - local_path 기본값: RECOMMEND_LATEST_PATH
    - remote_url 비어있으면 REMOTE_RECOMMEND_URL 사용
    """
    p = local_path or RECOMMEND_LATEST_PATH
    sig = _mtime(p)
    return _load_csv_cached(path=p, enc="utf-8-sig", remote_url=remote_url, mtime_sig=sig)

@st.cache_data(ttl=600, show_spinner=False)
def load_price_ohlcv(code: str, start: Optional[str] = None) -> pd.DataFrame:
    """
    가격 OHLCV 로드 (FDR 우선)
    return: index=Date, columns=[Open,High,Low,Close,Volume]
    """
    if not FDR_OK or fdr is None:
        return pd.DataFrame()

    code6 = str(code).split(".")[0].strip()
    if code6.isdigit():
        code6 = code6.zfill(6)

    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    try:
        df = fdr.DataReader(code6, start)
        if df is None or df.empty:
            return pd.DataFrame()
        # FDR은 보통 이 컬럼들로 옴
        need = ["Open", "High", "Low", "Close", "Volume"]
        for c in need:
            if c not in df.columns:
                return pd.DataFrame()
        return df[need].copy()
    except Exception:
        logger.exception("load_price_ohlcv(FDR) failed: %s", code6)
        return pd.DataFrame()

def calc_bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    볼린저 밴드
    return: (mid, upper, lower)
    """
    s = pd.to_numeric(close, errors="coerce")
    mid = s.rolling(window).mean()
    std = s.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower

def calc_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI(14) 시리즈
    """
    s = pd.to_numeric(close, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0)
    down = (-delta.clip(upper=0))
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------------------------
# 오픈베타 영구 PRIME 사용자
# ---------------------------
BETA_PRIME_USERS = {
    "coolguyhaeng@naver.com",
    "kiljung87@nate.com",
    "coiil@naver.com",
    "quartzk123@gmail.com",
    "user5@example.com",
}

def sync_user_role_with_subscription(user):
    """
    로그인 시마다 호출해서
    - 만료일 지난 Pro/Prime → free 자동 다운그레이드
    - 유효한 구독이면 subs.role 기준으로 auth_status 리턴
    """
    if not user:
        return "free", None

    email = user.get("login_id", "")
    base_role = user.get("role", "free")

    # (1) 베타 PRIME 유저: 무조건 PRIME 취급
    if email in BETA_PRIME_USERS:
        try:
            if base_role != "prime":
                update_user_role(email, "prime")
        except Exception:
            logger.exception("beta prime sync failed")
        return "prime", "∞"

    # (2) 일반 구독자
    sub = get_subscription(email)
    if not sub:
        return base_role, None

    exp_str = sub.get("expire_at")
    try:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
    except Exception:
        return base_role, exp_str

    today = now_kst().date()
    # 만료일 지났으면 free로
    if today > exp_date and base_role in ["pro", "prime"]:
        try:
            update_user_role(email, "free")
        except Exception:
            logger.exception("auto downgrade failed")
        set_subscription(email, "free")
        return "free", exp_str

    return sub.get("role", base_role), exp_str

# 1. 라이브러리 로드 (외부 라이브러리 실패에 대비)
try:
    import FinanceDataReader as fdr
    FDR_OK = True
except Exception as e:
    fdr = None
    FDR_OK = False
    logger.warning("FinanceDataReader not available: %s", e)

try:
    from pykrx import stock  # optional
    PYKRX_OK = True
except Exception as e:
    stock = None
    PYKRX_OK = False
    logger.info("pykrx not available: %s", e)

# 2. 페이지 설정
st.set_page_config(
    page_title=f"LDY Pro Trader v{APP_VERSION}",
    layout="wide",
    page_icon="💎",
)

with st.sidebar:
    if st.button("🔄 데이터/캐시 강제 새로고침"):
        st.cache_data.clear()
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

st.title(f"🏆 LDY Pro Trader v{APP_VERSION} (Prime Top 100 + Role-based Daily Top)")
st.caption("AI Quant Analysis & Portfolio Manager — Scoring / Subscription / Portfolio")

st.warning(
    "⚠️ 투자 관련 유의사항\n\n"
    "LDY Pro Trader는 주식 투자 의사결정을 돕기 위한 **데이터·알고리즘 기반 분석 도구**입니다.\n"
    "제공되는 모든 정보는 일반적인 참고용 자료일 뿐이며, 특정 종목의 매수·매도, 수익 창출 또는 손실 회피를 보장하지 않습니다.\n\n"
    "실제 투자에 대한 최종 판단과 그에 따른 결과(수익·손실 포함)는 **전적으로 이용자 본인에게 귀속**되며,\n"
    "본 서비스 및 개발자는 어떠한 법적 책임도 부담하지 않습니다."
)

# 🔔 상단 업데이트 공지 (version_info 헬퍼 함수 사용)
log = get_latest_log()
if log:
    # 화면 상단 간단 버전 라벨
    st.caption(f"LDY Pro Trader v{get_version_label(include_build=False)}")  # 예: v6.6

    # 핵심 2~3줄만 요약
    top_items = log["items"][:3]
    bullets = "\n".join(f"- {item}" for item in top_items)

    st.info(
        f"✅ v{log['version']} 업데이트 ({log['date']})\n\n"
        f"**{log['title']}**\n\n"
        f"{bullets}\n\n"
        "자세한 변경사항은 **🧩 LDY Pro Trader 업데이트 노트** 탭에서 확인할 수 있습니다."
    )

# 3. 설정 관리 (Secrets -> Env -> Default 순서)
def get_conf(key, default_val):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except FileNotFoundError:
        pass
    return os.getenv(key, default_val)

# ----------------- 설정값 로딩 -----------------
RAW_SRC = get_conf(
    "LDY_RAW_URL",
    "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
)
LOCAL_RAW = get_conf("LDY_LOCAL_RAW", "data/recommend_latest.csv")
PORTFOLIO_FILE = get_conf("LDY_PORTFOLIO_FILE", "my_portfolio.json")

# 🔐 보안키
KEY_PRO   = get_conf("LDY_KEY_PRO",   "220577")
KEY_PRIME = get_conf("LDY_KEY_PRIME", "577220")
ADMIN_KEY = get_conf("LDY_ADMIN_KEY", "2022322")

# 💳 결제 계좌 정보
BANK_ACCOUNT = get_conf("LDY_BANK_ACCOUNT", "카카오뱅크 3333-22-2658701")
BANK_HOLDER  = get_conf("LDY_BANK_HOLDER",  "이OO")

# 📊 스코어링 상수
PASS_EBS          = float(get_conf("LDY_PASS_EBS",          4))
MIN_TURN_KOSPI    = float(get_conf("LDY_MIN_TURN_KOSPI",    200.0))
MIN_TURN_KOSDAQ   = float(get_conf("LDY_MIN_TURN_KOSDAQ",   100.0))
MIN_TURN_DEFAULT  = float(get_conf("LDY_MIN_TURN_DEFAULT",  100.0))

W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = (0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10)
P_OVERHEAT_5D  = 6.0
P_OVERHEAT_10D = 6.0
P_RSI_OUT      = 4.0
P_MACD_NEG     = 4.0
P_NEAR_FAR     = 4.0
P_LIQ_LOW      = 4.0
P_VOL_SPIKE    = 2.0
RSI_LOW, RSI_HIGH = 45, 65

# ---------------------------
# 유틸 함수
# ---------------------------
def z6(x):
    return str(x).zfill(6) if str(x).isdigit() else str(x)

def nz_num(s):
    return pd.to_numeric(s, errors="coerce")

def ensure_turnover(df):
    if "거래대금(억원)" not in df.columns and "거래대금(원)" in df.columns:
        df["거래대금(억원)"] = (nz_num(df["거래대금(원)"]) / 1e8).round(2)
    return df

def normalize_cols(df):
    return ensure_turnover(df)

def make_preview(df, n=5):
    if df is None or df.empty:
        return df

    # 1순위: collector가 박아준 최종 랭크
    if "LDY_RANK" in df.columns:
        return df.sort_values("LDY_RANK", ascending=True).head(n).copy()

    # 2순위: CSV 원본 순서 랭크
    if "_CSV_RANK" in df.columns:
        return df.sort_values("_CSV_RANK", ascending=True).head(n).copy()

    # 3순위: 점수 기반(오를만함을 RANK_SCORE로 본다면 이게 핵심)
    keys = [c for c in ["RANK_SCORE", "ENTRY_SCORE", "LDY_SCORE", "거래대금(억원)"] if c in df.columns]
    if keys:
        return df.sort_values(keys, ascending=[False]*len(keys)).head(n).copy()

    # fallback
    return df.head(n).copy()

# ---------------------------
# 유틸 함수
# ---------------------------


def send_telegram_msg(token, chat_id, message):
    if not token or not chat_id:
        return False, "토큰/ID 누락"
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        r = requests.post(url, data=data, timeout=10)
        r.raise_for_status()
        return True, "전송 완료"
    except Exception as e:
        logger.exception("Telegram send failed")
        return False, str(e)

@st.cache_data(ttl=3600)
def get_code_map():
    """
    종목명 → 6자리 코드 매핑
    1순위: pykrx (KRX 공식 종목명)
    2순위: FinanceDataReader(KRX) 보조
    - 공백 제거 / 두 가지 key(원문, 공백제거문) 모두 저장
    """
    mapping = {}

    # 1) pykrx 우선 (이게 KRX 종목명 그대로라서 제일 믿을 만함)
    if PYKRX_OK:
        try:
            today_dt = now_kst().date()

            # ✅ 최근 10일 안에서 "티커가 실제로 나오는 날짜"를 찾는다 (주말/휴장일 대응)
            today = None
            for i in range(10):
                ymd = (today_dt - timedelta(days=i)).strftime("%Y%m%d")
                try:
                    chk = stock.get_market_ticker_list(ymd, market="KOSPI")
                    if chk:  # 빈 리스트 아니면 그 날짜가 거래일
                        today = ymd
                        break
                except Exception:
                    pass

            if today is None:
                today = now_kst().strftime("%Y%m%d")
            for mkt in ["KOSPI", "KOSDAQ"]:
                tickers = stock.get_market_ticker_list(today, market=mkt)
                for t in tickers:
                    code = str(t).zfill(6)
                    name = stock.get_market_ticker_name(t)  # 예: '삼성SDI'
                    if not isinstance(name, str):
                        continue
                    name = name.strip()
                    if not name:
                        continue

                    # 그대로
                    mapping.setdefault(name, code)
                    # 공백 제거 버전도 추가 (예: 'HD현대일렉트릭', 'HD 현대일렉트릭')
                    mapping.setdefault(name.replace(" ", ""), code)
        except Exception as e:
            logger.exception("get_code_map via pykrx failed: %s", e)

    # 2) FDR 보조 (pykrx가 안 되거나 빠진 종목 채우기용)
    if FDR_OK:
        try:
            df = fdr.StockListing("KRX")
            df["Code"] = df["Code"].astype(str).str.zfill(6)
            for _, row in df.iterrows():
                name = str(row.get("Name", "")).strip()
                code = row["Code"]
                if not name:
                    continue

                # 기존에 없을 때만 채움 (pykrx 우선 유지)
                mapping.setdefault(name, code)
                mapping.setdefault(name.replace(" ", ""), code)
        except Exception as e:
            logger.exception("get_code_map via FDR failed: %s", e)

    return mapping


def find_code_by_name(name_or_code, code_map):
    """
    - 6자리 숫자 → 그대로 코드로 사용
    - '005930.KS', '005930.KQ' 같은 형식도 처리
    - '삼성SDI', '삼성 SDI', '삼성SDI(006400)' 같은 케이스까지 최대한 커버
    """
    x = str(name_or_code).strip()
    if not x:
        return None

    # 1) 6자리 숫자만 들어온 경우
    if x.isdigit():
        return x.zfill(6)

    # 2) '005930.KS' 같은 형식
    if "." in x:
        left = x.split(".")[0]
        if left.isdigit():
            return left.zfill(6)

    # 3) 괄호 안에 코드가 들어 있는 경우: '삼성SDI(006400)'
    m = re.search(r"(\d{6})", x)
    if m:
        return m.group(1)

    # 4) 이름 기반 매핑 (원문 → 공백 제거 순으로 시도)
    cand = code_map.get(x)
    if cand:
        return cand

    cand = code_map.get(x.replace(" ", ""))
    if cand:
        return cand

    return None


# ---------------------------
# 시장 상태 계산 (지수 + 로컬 fallback)
# ---------------------------

@st.cache_data(ttl=600)
def get_market_status_local(scored_df: pd.DataFrame):
    result = {}

    has_market_col = "시장" in scored_df.columns

    for mkt in ["KOSPI", "KOSDAQ"]:
        if has_market_col:
            sub = scored_df[scored_df["시장"] == mkt].copy()
        else:
            sub = scored_df.copy()  # 시장 구분 없으면 전체 대상으로

        if sub.empty:
            result[mkt] = ("데이터 없음", float("nan"))
            continue

        if "ret_5d_%" not in sub.columns:
            result[mkt] = ("데이터 부족", float("nan"))
            continue

        r5 = pd.to_numeric(sub["ret_5d_%"], errors="coerce").dropna()
        if r5.empty:
            result[mkt] = ("데이터 부족", float("nan"))
            continue

        avg_5d = float(r5.mean())
        status = "📈 상승장" if avg_5d > 0 else "📉 조정장"
        status_text = f"{status} (스코어 기반)"

        result[mkt] = (status_text, avg_5d)

    kp_stat, kp_diff = result.get("KOSPI", ("데이터 없음", float("nan")))
    kq_stat, kq_diff = result.get("KOSDAQ", ("데이터 없음", float("nan")))
    return kp_stat, kp_diff, kq_stat, kq_diff


@st.cache_data(ttl=600)
def get_market_status(scored_df: pd.DataFrame):
    """
    KOSPI / KOSDAQ 상태 조회
    1) FDR / pykrx 인덱스 데이터로 계산 시도
    2) 실패/오류면 scored_df 기반 로컬 계산으로 fallback
    """
    # scored_df가 없으면 바로 실패 처리
    if scored_df is None or scored_df.empty:
        return "데이터 없음", float("nan"), "데이터 없음", float("nan")

    # 1) FDR / pykrx 둘 다 안 되면 바로 로컬
    if not FDR_OK and not PYKRX_OK:
        return get_market_status_local(scored_df)

    def _via_fdr(ticker: str):
        if not FDR_OK:
            return None
        try:
            df = fdr.DataReader(ticker)
            return df if df is not None and not df.empty else None
        except Exception:
            logger.exception("FDR DataReader failed for %s", ticker)
            return None

    def _via_pykrx_index(ticker: str):
        if not PYKRX_OK:
            return None
        try:
            today = now_kst().strftime("%Y%m%d")
            start = (now_kst() - timedelta(days=365)).strftime("%Y%m%d")
            code = "1001" if ticker == "KS11" else "2001"
            df = stock.get_index_ohlcv_by_date(start, today, code)
            if df is None or df.empty:
                return None
            if "종가" in df.columns and "Close" not in df.columns:
                df = df.rename(columns={"종가": "Close"})
            return df
        except Exception:
            logger.exception("pykrx index fetch failed for %s", ticker)
            return None

    def _status_for(ticker: str):
        # ✅ 순차적으로 확인하도록 수정
        df = _via_fdr(ticker)
        if df is None or df.empty:
            df = _via_pykrx_index(ticker)
            
        if df is None or df.empty:
            return None

        df = df.tail(60)
        if "Close" not in df.columns:
            return None

        close = df["Close"]
        ma20 = close.rolling(20).mean().iloc[-1]
        curr = close.iloc[-1]
        if pd.isna(ma20) or ma20 == 0:
            return None

        diff = ((curr - ma20) / ma20) * 100
        status = "📈 상승장" if diff > 0 else "📉 조정장"

        # 전일 기준 표기
        try:
            last_date = df.index[-1].date()
        except Exception:
            last_date = pd.to_datetime(df.index[-1]).date()

        if last_date < now_kst().date():
            status += " (전일 기준)"

        return status, diff

    try:
        kp = _status_for("KS11")
        kq = _status_for("KQ11")
        if kp and kq:
            return kp[0], kp[1], kq[0], kq[1]
    except Exception:
        logger.exception("get_market_status index path failed")

    # 2) 실패 시 로컬 fallback (globals() 금지)
    return get_market_status_local(scored_df)


@st.cache_data(ttl=600)
def get_macro_metrics():
    """
    [v8.0] 환율(USD/KRW), 나스닥(IXIC) 조회
    """
    if not FDR_OK:
        return None

    metrics = {}
    try:
        # 1. 환율
        # 최근 7일치 가져와서 마지막 영업일 기준 등락 계산
        df_usd = fdr.DataReader("USD/KRW", (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"))
        if df_usd is not None and not df_usd.empty:
            curr = df_usd["Close"].iloc[-1]
            prev = df_usd["Close"].iloc[-2]
            metrics["USD"] = (curr, (curr - prev))

        # 2. 나스닥
        df_nas = fdr.DataReader("IXIC", (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"))
        if df_nas is not None and not df_nas.empty:
            curr = df_nas["Close"].iloc[-1]
            prev = df_nas["Close"].iloc[-2]
            metrics["IXIC"] = (curr, (curr - prev) / prev * 100)
            
    except Exception as e:
        logger.warning(f"Macro metrics failed: {e}")
        
    return metrics


@st.cache_data(ttl=600)
def get_fear_greed_index(scored_df: pd.DataFrame):
    """
    1순위: FDR KS11 지수 기반 공포/탐욕
    2순위: 실패 시 scored_df 기반 fallback
    """

    # -------- 1) 지수(FDR) 경로 --------
    try:
        if FDR_OK:
            df = fdr.DataReader("KS11")
            if df is not None and not df.empty:
                delta = df["Close"].diff()
                up = delta.clip(lower=0)
                down = (-delta.clip(upper=0))
                rs = up.rolling(14).mean() / down.rolling(14).mean()
                rsi = 100 - (100 / (1 + rs))
                current_rsi = float(rsi.iloc[-1])

                ma20 = df["Close"].rolling(20).mean()
                disparity = float(df["Close"].iloc[-1] / ma20.iloc[-1] * 100)

                score = current_rsi
                if disparity > 105:
                    score += 10
                elif disparity < 95:
                    score -= 10

                score = max(0.0, min(100.0, score))

                if score >= 75:
                    status = "매도 권장 (탐욕)"
                elif score >= 60:
                    status = "과열 구간"
                elif score <= 25:
                    status = "적극 매수 (공포)"
                elif score <= 40:
                    status = "침체 구간"
                else:
                    status = "중립 (관망)"

                return float(score), status + " (지수 기준)"
    except Exception as e:
        logger.exception("fear_greed FDR path failed: %s", e)

    # -------- 2) scored_df fallback 경로 --------
    try:
        if scored_df is None or scored_df.empty:
            return 50.0, "중립 (데이터 없음)"

        if "RSI14" not in scored_df.columns:
            return 50.0, "중립 (데이터 부족)"

        rsi = pd.to_numeric(scored_df["RSI14"], errors="coerce").dropna()
        if rsi.empty:
            return 50.0, "중립 (데이터 부족)"

        rsi_mid = float(rsi.median())

        gap_mean = 0.0
        if "MA20_GAP" in scored_df.columns:
            gap = pd.to_numeric(scored_df["MA20_GAP"], errors="coerce").dropna()
            if not gap.empty:
                gap_mean = float(gap.mean())

        score = rsi_mid
        if gap_mean > 5:
            score += 10
        elif gap_mean < -5:
            score -= 10

        score = max(0.0, min(100.0, score))

        if score >= 75:
            status = "매도 권장 (탐욕)"
        elif score >= 60:
            status = "과열 구간"
        elif score <= 25:
            status = "적극 매수 (공포)"
        elif score <= 40:
            status = "침체 구간"
        else:
            status = "중립 (관망)"

        return float(score), status + " (스코어 기준)"
    except Exception as e:
        logger.exception("fear_greed local fallback failed: %s", e)
        return 50.0, "중립 (지표 계산 오류)"


def plot_fear_greed_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "시장 공포/탐욕 지수", 'font': {'size': 20}},
        delta={
            'reference': 50,
            'increasing': {'color': "red"},
            'decreasing': {'color': "blue"}
        },
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 25], 'color': '#4D96FF'},
                {'range': [25, 45], 'color': '#87CEEB'},
                {'range': [45, 55], 'color': '#D3D3D3'},
                {'range': [55, 75], 'color': '#FFB347'},
                {'range': [75, 100], 'color': '#FF6B6B'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_sector_treemap(df_map):
    """
    섹터 트리맵:
    - '업종_대분류' 컬럼이 있으면 대분류 기준으로 묶고
    - 없으면 기존 '업종' 컬럼을 사용
    """
    if df_map is None or df_map.empty:
        return go.Figure()

    # 1) 섹터 키 선택 (대분류 우선)
    sector_key = "업종_대분류" if "업종_대분류" in df_map.columns else "업종"

    if sector_key not in df_map.columns:
        # 업종 정보 자체가 없으면 빈 figure 반환
        return go.Figure()

    # 2) 트리맵 생성
    fig = px.treemap(
        df_map,
        path=[sector_key, "종목명"],   # ✅ 최상단을 대분류로
        values="거래대금(억원)",
        color="LDY_SCORE",
        color_continuous_scale="RdYlGn",
        title="<b>🔥 시장 주도 섹터 지도</b>",
        custom_data=["LDY_SCORE", sector_key],
    )

    # 3) hover 텍스트
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b>"                # 종목명
            "<br>섹터: %{customdata[1]}"     # 업종_대분류
            "<br>점수: %{customdata[0]:.1f}"
            "<br>대금: %{value}억"
            "<extra></extra>"
        )
    )

    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=350)
    return fig

def plot_sector_momentum_bar(scored_df: pd.DataFrame):
    """
    섹터별 최근 모멘텀 (ret_5d_% or LDY_SCORE 평균) Top 10 바 차트
    """
    if scored_df is None or scored_df.empty:
        return go.Figure()

    # 섹터 컬럼
    if "업종_대분류" in scored_df.columns:
        sector_col = "업종_대분류"
    elif "업종" in scored_df.columns:
        sector_col = "업종"
    else:
        return go.Figure()

    metric = "ret_5d_%" if "ret_5d_%" in scored_df.columns else "LDY_SCORE"

    grp = (
        scored_df
        .dropna(subset=[sector_col, metric])
        .groupby(sector_col)[metric]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    if grp.empty:
        return go.Figure()

    values = grp.values
    labels = grp.index

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                text=[f"{v:.2f}%" if metric == "ret_5d_%" else f"{v:.2f}" for v in values],
                textposition="auto",
            )
        ]
    )
    title_metric = "5일 평균 수익률" if metric == "ret_5d_%" else "LDY 평균 점수"
    fig.update_layout(
        title=f"🚀 섹터 모멘텀 Top 10 ({title_metric})",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

def plot_regime_summary(scored_df: pd.DataFrame):
    """
    Regime 별 평균 성과(점수, 수익률) 분석 테이블 표시
    """
    if scored_df is None or scored_df.empty or "REGIME" not in scored_df.columns:
        return

    # 필요한 컬럼 확인
    cols = ["LDY_SCORE"]
    if "ret_5d_%" in scored_df.columns:
        cols.append("ret_5d_%")

    # 그룹화 및 평균 계산 (내림차순 정렬)
    try:
        grp = scored_df.groupby("REGIME")[cols].mean().sort_values("LDY_SCORE", ascending=False)
    except Exception:
        return

    # 컬럼명 변경 (화면 표시용)
    rename_map = {"LDY_SCORE": "평균 점수"}
    if "ret_5d_%" in cols:
        rename_map["ret_5d_%"] = "5일 수익률(%)"

    grp = grp.rename(columns=rename_map)

    st.markdown("##### 🧐 Regime 별 성과 분석 (평균)")

    # 스타일링: 점수는 파란색, 수익률은 빨강-초록 그라데이션
    st_style = grp.style.format("{:.2f}").background_gradient(cmap="Blues", subset=["평균 점수"])

    if "5일 수익률(%)" in grp.columns:
        st_style = st_style.background_gradient(cmap="RdYlGn", subset=["5일 수익률(%)"])

    st.dataframe(st_style, use_container_width=True)

    # 1위 코멘트
    if not grp.empty:
        top_name = grp.index[0]
        top_val = grp.iloc[0]["평균 점수"]
        st.caption(f"💡 현재 **'{top_name}'** 구간의 종목들이 평균 **{top_val:.1f}점**으로 가장 우수한 평가를 받고 있습니다.")

def calculate_supertrend(df, period=10, multiplier=3):
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean()

    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    trend = pd.Series(1, index=df.index)

    for i in range(period, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

        if trend.iloc[i-1] == 1:
            if close.iloc[i] < final_lower.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = 1
        else:
            if close.iloc[i] > final_upper.iloc[i-1]:
                trend.iloc[i] = 1
            else:
                trend.iloc[i] = -1

        supertrend.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]

    df['SuperTrend'] = supertrend
    df['Trend'] = trend
    return df

@st.cache_data(ttl=600)
def get_stock_chart_data(code):
    if not FDR_OK: return None
    try:
        code_str = str(code).zfill(6)
        # 넉넉히 1년치 가져오되, 차트엔 최근 100~150개만 표시하는 게 좋음
        start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        df = fdr.DataReader(code_str, start_date)
        if df is None or df.empty: return None

        # 이동평균
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()

        # 🔹 Bollinger Bands (20, 2.0)
        std20 = df['Close'].rolling(window=20).std()
        df['BB_UPPER'] = df['MA20'] + 2.0 * std20
        df['BB_LOWER'] = df['MA20'] - 2.0 * std20

        # 🔹 Keltner Channels (20, 1.5 ATR) - Collector v7.x와 동기화
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr20 = tr.rolling(window=20).mean()

        df['KC_UPPER'] = df['MA20'] + (1.5 * atr20)
        df['KC_LOWER'] = df['MA20'] - (1.5 * atr20)

        # 🔹 RSI(14)
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        df['RSI14_CHART'] = 100 - (100 / (1 + rs))
        # -------------------- [v9.0 HMA 추가] --------------------
        # HMA 20일선 계산 (캔들 차트에 표시용)
        df['HMA20'] = calc_hma_series(df['Close'], 20)
        # ---------------------------------------------------------
        # -------------------- [v9.0 OBV 계산 추가] --------------------
        # OBV: 주가 등락에 따른 거래량 누적
        change = np.sign(df['Close'].diff()).fillna(0)
        df['OBV'] = (change * df['Volume']).cumsum()
        # -------------------------------------------------------------

        
        # SuperTrend
        df = calculate_supertrend(df)

        # -------------------- [v10.0 추가] 주봉 20선 계산 --------------------
        # 일봉 데이터를 주봉으로 리샘플링하여 대추세선 산출
        logic_w = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        df_w = df.resample('W').apply(logic_w)
        df_w['WMA20'] = df_w['Close'].rolling(window=20).mean()
        
        # 일봉 데이터프레임에 주봉 20선 값을 매핑 (시각화용 점선)
        # 현재 일봉 날짜보다 작거나 같은 가장 최근의 주봉 20선 값을 가져옴
        df['WEEKLY_MA20'] = df.index.map(lambda x: df_w.loc[df_w.index <= x, 'WMA20'].iloc[-1] if not df_w.loc[df_w.index <= x, 'WMA20'].empty else np.nan)
        # ------------------------------------------------------------------

        # 최근 120일 데이터 반환
        return df.tail(120)
    except Exception:
        logger.exception("get_stock_chart_data failed")
        return None

def plot_radar_chart(row):
    """
    v7.5: 7-Factor (V-Power 포함) 레이더 차트
    """
    # 1) 팩터 데이터 확인 (Collector v7.5 이상)
    if "NORM_MOM" in row.index:
        stats = {
            "모멘텀(MOM)": row.get("NORM_MOM", 0) * 100,
            "가성비(RR)": row.get("NORM_RR", 0) * 100,
            "수익여력(T1)": row.get("NORM_T1", 0) * 100,
            "안전성(SL)": row.get("NORM_SL", 0) * 100,
            "타점(NEAR)": row.get("NORM_NEAR", 0) * 100,
            "유동성(LIQ)": row.get("NORM_LIQ", 0) * 100,
            "기술/세력(TEC)": row.get("NORM_TEC", 0) * 100, # ✅ v7.5 추가
        }
    else:
        # Fallback (구버전 데이터용)
        stats = {
            "모멘텀": min(100, (row.get("ret_5d_%", 0) + 5) * 10),
            "수급(MFI)": row.get("MFI14", 50),
            "가성비(RR)": min(100, row.get("RR1", 1) * 50),
            "안전성": 100 - (row.get("이격도", 0) * 2),
            "종합점수": row.get("LDY_SCORE", 0),
        }

    values = [max(0, min(100, v)) for v in stats.values()]
    keys = list(stats.keys())

    # 레이더 차트 닫기 위해 첫 번째 값 추가
    values += values[:1]
    keys += keys[:1]

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=keys,
            fill='toself',
            name=row.get('종목명', '종목'),
            # 🔥 색상 변경: 밝은 Cyan + 반투명 채우기
            line=dict(color='#00E5FF', width=3),
            fillcolor='rgba(0, 229, 255, 0.2)'
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 100],
                tickfont=dict(size=10, color='gray'),
                gridcolor='rgba(128,128,128,0.3)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, weight='bold'),
                gridcolor='rgba(128,128,128,0.3)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        height=300, # 높이 약간 증가
        margin=dict(l=40, r=40, t=30, b=30),
        title=dict(text="📊 7-Factor Analysis", x=0.5, y=0.95, font=dict(size=14))
    )
    return fig

# ---------------------------
# 차트 시각화 (거래량 추가)
# ---------------------------
def plot_interactive_chart(
    df: pd.DataFrame,
    code: str,
    name: str,
    entry=None,
    stop=None,
    target1=None,
    target2=None,
    vwap=None,          # ✅ [v8.5] VWAP 가격 인자 추가
    show_bb: bool = True,
    show_kc: bool = False,
    show_rsi: bool = False,
    show_vwap: bool = False, # ✅ [v8.5] VWAP 표시 여부
    show_hma: bool = False,  # ✅ [v9.0] HMA 표시 옵션 추가
    show_obv: bool = False,  # 👈 [v9.0] 추가
    
):
    if df is None or df.empty:
        return go.Figure()

    # OBV가 켜지면 행을 하나 더 늘림 (총 3개 or 4개)
    rows = 2
    if show_rsi: rows += 1
    if show_obv: rows += 1
    
    # 높이 비율 동적 할당
    if rows == 2: row_heights = [0.7, 0.3]
    elif rows == 3: row_heights = [0.6, 0.2, 0.2]
    else: row_heights = [0.5, 0.15, 0.15, 0.2] # 4행일 때

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights
    )

    # --- 🎨 색상 팔레트 정의 ---
    COLOR_UP = '#FF3B30'      # 밝은 빨강 (상승)
    COLOR_DOWN = '#007AFF'    # 밝은 파랑 (하락)
    COLOR_MA20 = '#FFD700'    # 황금색 (20일선)
    COLOR_BB = 'rgba(189, 195, 199, 0.5)'      # 은은한 회색 선 (BB)
    COLOR_BB_FILL = 'rgba(189, 195, 199, 0.1)' # 아주 연한 회색 채우기 (BB)
    COLOR_KC = '#E040FB'      # 형광 보라 (KC)
    COLOR_ENTRY = '#FF9F0A'   # 형광 오렌지 (진입가)
    COLOR_STOP = '#30D158'    # 형광 초록 (목표가 - 상승이라 초록 계열 사용)
    COLOR_LOSS = '#00B0FF'    # 형광 하늘 (손절가 - 파랑 캔들과 구분되는 하늘색)
    COLOR_VWAP = '#FF00FF'    # ✅ 형광 마젠타 (VWAP)

    # 1) 캔들 차트
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="주가",
            increasing={'line': {'color': COLOR_UP, 'width': 1.5}, 'fillcolor': COLOR_UP},
            decreasing={'line': {'color': COLOR_DOWN, 'width': 1.5}, 'fillcolor': COLOR_DOWN},
            hovertemplate="<b>%{x|%y/%m/%d}</b><br>시가: %{open:,.0f}<br>고가: %{high:,.0f}<br>저가: %{low:,.0f}<br>종가: %{close:,.0f}원<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1
    )

    # 2) MA20 (황금색 실선)
    if "MA20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["MA20"], 
                name="20일선", 
                line=dict(color=COLOR_MA20, width=2)
            ), 
            row=1, col=1
        )

    # -------------------- [v9.0 HMA 라인 추가] --------------------
    if show_hma and "HMA20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["HMA20"],
                name="HMA(20)",
                line=dict(color='#00BCD4', width=2.5), # 밝은 Cyan 색상
            ),
            row=1, col=1
        )
    # -------------------------------------------------------------

    # 3) 볼린저 밴드 (은은한 회색 영역)
    if show_bb:
        if "BB_UPPER" in df.columns and "BB_LOWER" in df.columns:
            # 상단
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_UPPER"], 
                name="BB 상단", 
                line=dict(width=1, color=COLOR_BB),
                showlegend=False
            ), row=1, col=1)
            # 하단 (채우기 포함)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_LOWER"], 
                name="BB 밴드", 
                line=dict(width=1, color=COLOR_BB), 
                fill='tonexty', 
                fillcolor=COLOR_BB_FILL,
                showlegend=True
            ), row=1, col=1)

    # 4) 켈트너 채널 (보라색 점선) - 볼린저 밴드와 확연히 구분
    if show_kc:
        if "KC_UPPER" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["KC_UPPER"], 
                name="KC 상단", 
                line=dict(width=1.5, dash='dot', color=COLOR_KC)
            ), row=1, col=1)
        if "KC_LOWER" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["KC_LOWER"], 
                name="KC 하단", 
                line=dict(width=1.5, dash='dot', color=COLOR_KC)
            ), row=1, col=1)

    # 5) SuperTrend (Trailing Stop Line) - v8.0 개선 (선 차트로 변경)
    if "Trend" in df.columns and "SuperTrend" in df.columns:
        # 상승 추세 (초록색 실선 - 지지선 역할)
        st_up = df[df["Trend"] == 1]["SuperTrend"]
        if not st_up.empty:
            fig.add_trace(go.Scatter(
                x=st_up.index, y=st_up,
                mode='lines',
                line=dict(color='#00E676', width=2), # Solid Green Line
                name='SuperTrend (Support)'
            ), row=1, col=1)

        # 하락 추세 (빨간색 점선 - 저항선 역할)
        st_down = df[df["Trend"] == -1]["SuperTrend"]
        if not st_down.empty:
            fig.add_trace(go.Scatter(
                x=st_down.index, y=st_down,
                mode='lines',
                line=dict(color='#FF4081', width=2, dash='dot'),
                name='SuperTrend (Resist)'
            ), row=1, col=1)

        # -------------------- [v10.0 추가] 주봉 20선 렌더링 --------------------
        # 굵은 주봉 20선 추가 (회색 점선으로 '심리적 지지선' 표시)
        if "WEEKLY_MA20" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df["WEEKLY_MA20"],
                name="주봉 20선",
                line=dict(color='rgba(100, 100, 100, 0.5)', width=3, dash='dashdot'),
                hovertemplate="주봉20선: %{y:,.0f}원<extra></extra>"
            ), row=1, col=1)  # <--- 여기서 괄호를 닫아주어야 에러가 나지 않습니다.
        # ---------------------------------------------------------------------

        current_row = 2

    # 6) 거래량 (항상 표시)
    if "Volume" in df.columns:
        colors = [COLOR_UP if c >= o else COLOR_DOWN for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="거래량", marker_color=colors, opacity=0.8, showlegend=False
        ), row=current_row, col=1)
        current_row += 1

    # 7) RSI
    if show_rsi and "RSI14_CHART" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI14_CHART"], name="RSI(14)", line=dict(color='#AB47BC', width=1.5)
        ), row=current_row, col=1)
        # 기준선 30/70
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70, line=dict(color="red", width=1, dash="dot"), row=current_row, col=1)
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30, line=dict(color="blue", width=1, dash="dot"), row=current_row, col=1)
        current_row += 1

    # 8) [v9.0] OBV 차트 (새로 추가됨)
    if show_obv and "OBV" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["OBV"], name="💰OBV", 
            line=dict(color='#2962FF', width=1.5), fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.1)'
        ), row=current_row, col=1)
        current_row += 1

    # 8) 가격 라인 (가시성 높은 형광색 사용)
    def _safe_float(v):
        try:
            vv = float(pd.to_numeric(v, errors="coerce"))
            return vv if np.isfinite(vv) and vv > 0 else None
        except Exception:
            return None

    entry_v = _safe_float(entry)
    stop_v = _safe_float(stop)
    t1_v = _safe_float(target1)
    vwap_v = _safe_float(vwap) # ✅

    if entry_v is not None:
        fig.add_hline(y=entry_v, line_dash="dash", line_color=COLOR_ENTRY, line_width=1.5, 
                      annotation_text=f"🚀진입: {int(entry_v):,}", annotation_font_color=COLOR_ENTRY, row=1, col=1)
    if stop_v is not None:
        fig.add_hline(y=stop_v, line_dash="dot", line_color=COLOR_LOSS, line_width=1.5,
                      annotation_text=f"🛡️손절: {int(stop_v):,}", annotation_font_color=COLOR_LOSS, row=1, col=1)
    if t1_v is not None:
        fig.add_hline(y=t1_v, line_dash="dot", line_color=COLOR_STOP, line_width=1.5,
                      annotation_text=f"💰목표: {int(t1_v):,}", annotation_font_color=COLOR_STOP, row=1, col=1)

    # ✅ [v8.5] VWAP 라인 추가
    if show_vwap and vwap_v:
        fig.add_hline(y=vwap_v, line_dash="solid", line_color=COLOR_VWAP, line_width=1.2, annotation_text=f"🟣VWAP: {int(vwap_v):,}", annotation_font_color=COLOR_VWAP, row=1, col=1)

    fig.update_layout(
        title=dict(text=f"{name} ({str(code).zfill(6)})", font=dict(size=16), x=0),
        xaxis_rangeslider_visible=False,
        height=700 if show_rsi else 550,
        margin=dict(l=10, r=10, t=80, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="pan",
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    return fig
    
def plot_risk_reward_bar(buy, stop, target1, target2):
    fig = go.Figure()
    try:
        loss_pct = int(((buy - stop) / buy) * 100)
    except Exception:
        loss_pct = 0
    fig.add_trace(
        go.Bar(
            y=["Price"],
            x=[max(buy - stop, 0)],
            orientation='h',
            name='Risk',
            marker=dict(color='red'),
            text=f"손절: {int(stop):,}원 (-{loss_pct}%)",
            textposition='auto',
        )
    )
    try:
        p1_pct = int(((target1 - buy) / buy) * 100)
    except Exception:
        p1_pct = 0
    fig.add_trace(
        go.Bar(
            y=["Price"],
            x=[max(target1 - buy, 0)],
            orientation='h',
            name='Reward 1',
            marker=dict(color='lightgreen'),
            text=f"1차: {int(target1):,}원 (+{p1_pct}%)",
            textposition='auto',
        )
    )
    try:
        p2_pct = int(((target2 - buy) / buy) * 100)
    except Exception:
        p2_pct = 0
    fig.add_trace(
        go.Bar(
            y=["Price"],
            x=[max(target2 - target1, 0)],
            orientation='h',
            name='Reward 2',
            marker=dict(color='green'),
            text=f"2차: {int(target2):,}원 (+{p2_pct}%)",
            textposition='auto',
        )
    )
    fig.update_layout(
        barmode='stack',
        showlegend=False,
        height=80,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
    
def plot_ai_consensus(df):
    """
    [v10.0] AI Score vs Rule Score 산점도
    - 우상단일수록 AI와 퀀트 모두가 추천하는 종목
    """
    if df is None or df.empty or "ML_SCORE" not in df.columns:
        return None

    # 데이터 전처리 (0점 제외)
    plot_df = df[(df["RANK_SCORE"] > 0) & (df["ML_SCORE"] > 0)].copy()
    if plot_df.empty: return None

    fig = px.scatter(
        plot_df,
        x="RANK_SCORE",
        y="ML_SCORE",
        color="TOTAL_SCORE",
        size="거래대금(억원)",
        hover_name="종목명",
        hover_data=["종목코드", "업종", "ROUTE"],
        color_continuous_scale="RdYlGn",
        title="<b>🧠 AI(세로) vs 퀀트(가로) 합의 지점</b>",
        labels={"RANK_SCORE": "퀀트(Rule) 점수", "ML_SCORE": "AI(ML) 예측 점수"}
    )

    # 기준선 (80점)
    fig.add_hline(y=80, line_dash="dot", line_color="gray", annotation_text="AI 강력매수")
    fig.add_vline(x=80, line_dash="dot", line_color="gray", annotation_text="퀀트 강력매수")

    # 우상단 강조 박스 (Hot Zone)
    fig.add_shape(type="rect",
        x0=80, y0=80, x1=100, y1=100,
        line=dict(color="red", width=2),
        fillcolor="rgba(255, 0, 0, 0.1)"
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[40, 105]),
        yaxis=dict(range=[40, 105])
    )
    return fig


# ---------------------------
# 데이터 로딩
# ---------------------------
def normalize_github_raw(url: str) -> str:
    if not isinstance(url, str):
        return url
    if "github.com/" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        url = url.replace("/blob/", "/")
    return url



@st.cache_data(ttl=600)
def load_csv_path(path: str, enc: str = "utf-8-sig") -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=enc)
    except UnicodeDecodeError:
        # utf-8-sig 실패 시 utf-8 재시도
        return pd.read_csv(path, encoding="utf-8")

def log_src(df, src):
    logger.info("Data Loaded: %s rows=%s", src, len(df) if df is not None else 0)

# ---------------------------
# 포트폴리오 저장소 설정 (Gist 연동)
# ---------------------------
# secrets.toml 또는 환경변수에 설정 필요
GIST_TOKEN = get_conf("LDY_GIST_TOKEN", "")
GIST_ID    = get_conf("LDY_GIST_ID", "")
GIST_FILENAME = "my_portfolio.json"

def load_portfolio_file():
    """1순위: Gist, 2순위: 로컬 파일"""
    # 1. Gist 로드 시도
    if GIST_TOKEN and GIST_ID:
        try:
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            r = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                # Gist 안에 해당 파일이 있는지 확인
                if GIST_FILENAME in data["files"]:
                    content = data["files"][GIST_FILENAME]["content"]
                    # {"data": "..."} 형태이므로 파싱 후 내부 데이터 반환
                    return json.loads(content).get("data", "")
        except Exception as e:
            logger.error(f"Gist Load Failed: {e}")

    # 2. 로컬 파일 로드 (Fallback)
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("data", "")
        except Exception:
            logger.exception("load_portfolio_file local failed")

    return ""

def save_portfolio_file(text_data):
    """Gist와 로컬 파일 모두에 저장"""
    success = False
    json_content = json.dumps({"data": text_data}, ensure_ascii=False)

    # 1. Gist 저장 시도
    if GIST_TOKEN and GIST_ID:
        try:
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            payload = {
                "files": {
                    GIST_FILENAME: {
                        "content": json_content
                    }
                }
            }
            # PATCH 요청으로 Gist 업데이트
            r = requests.patch(f"https://api.github.com/gists/{GIST_ID}", json=payload, headers=headers, timeout=5)
            if r.status_code == 200:
                success = True
                logger.info("Saved to Gist successfully")
            else:
                logger.error(f"Gist Save Error: {r.status_code} {r.text}")
        except Exception as e:
            logger.exception(f"Gist Save Failed: {e}")

    # 2. 로컬 파일 저장 (백업용)
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            f.write(json_content)
        success = True # 로컬이라도 저장되면 성공으로 간주
    except Exception:
        logger.exception("save_portfolio_file local failed")

    return success

# ---------------------------
# 스코어링 함수 (v6.4 스타일)
# ---------------------------
def liquidity_gate(x_turn, market):
    min_map = {
        "KOSPI": MIN_TURN_KOSPI,
        "KOSDAQ": MIN_TURN_KOSDAQ,
    }
    try:
        return nz_num(x_turn) >= market.map(min_map).fillna(MIN_TURN_DEFAULT)
    except Exception:
        return pd.Series(False, index=x_turn.index)

def build_global_score(lat, keep_order: bool = False):
    x = lat.copy()
    req = [
        "종가", "추천매수가", "손절가", "추천매도가1",
        "거래대금(억원)", "RSI14", "MACD_Slope", "거래강도",
        "이격도", "ret_5d_%", "ret_10d_%", "EBS",
        "MACD_Hist", "MFI14", "시장",
    ]
    for c in req:
        if c not in x.columns:
            x[c] = np.nan

    slope_col = "MACD_Slope" if "MACD_Slope" in x.columns and x["MACD_Slope"].notna().any() \
        else ("MACD_slope" if "MACD_slope" in x.columns else "MACD_Slope")
    kairi_col = "이격도" if "이격도" in x.columns and x["이격도"].notna().any() \
        else ("乖離%" if "乖離%" in x.columns else "이격도")
    vol_col = "거래강도" if "거래강도" in x.columns and x["거래강도"].notna().any() \
        else ("Vol_Z" if "Vol_Z" in x.columns else "거래강도")

    close = nz_num(x["종가"])
    entry = nz_num(x["추천매수가"])
    stop = nz_num(x["손절가"])
    t1 = nz_num(x["추천매도가1"])
    turn = nz_num(x["거래대금(억원)"])
    rsi = nz_num(x["RSI14"])
    slope = nz_num(x.get(slope_col, pd.Series(np.nan, index=x.index)))
    volz = nz_num(x.get(vol_col, pd.Series(np.nan, index=x.index)))
    kairi = nz_num(x.get(kairi_col, pd.Series(np.nan, index=x.index)))
    r5 = nz_num(x["ret_5d_%"])
    r10 = nz_num(x["ret_10d_%"])
    ebs = nz_num(x["EBS"]).fillna(0)

    rr_den = (entry - stop)
    rr_den = rr_den.where(rr_den > 0, np.nan)
    rr1 = (t1 - entry) / rr_den
    now_gap = ((close - entry).abs() / entry * 100)
    t1_room = ((t1 - close) / close * 100)
    sl_room = ((close - stop) / close * 100)

    def cap_q(s, q=90, f=1.0):
        arr = nz_num(s)
        arr = arr.replace([np.inf, -np.inf], np.nan)
        if arr.dropna().size == 0:
            return float(f)
        try:
            val = float(np.nanpercentile(arr.dropna(), q))
            return max(val, float(f))
        except Exception:
            return float(f)

    def pct_norm(s, q=90, f=1.0):
        s_num = nz_num(s).clip(lower=0)
        cap = cap_q(s_num, q, f)
        if cap == 0:
            return np.zeros_like(s_num)
        return np.clip(s_num / cap, 0, 1)

    def inv_dist_norm(dist, cap):
        cap_val = float(cap) if cap is not None and not np.isnan(cap) else 1.0
        return np.clip(1 - (nz_num(dist) / max(cap_val, 1e-9)), 0, 1)

    rr_norm = pct_norm(rr1, q=90, f=1.0).fillna(0)
    t1_norm = np.clip(t1_room / cap_q(t1_room, q=90, f=5.0), 0, 1).fillna(0)
    sl_norm = np.clip(sl_room / cap_q(sl_room, q=90, f=3.0), 0, 1).fillna(0)
    near_norm = inv_dist_norm(now_gap, cap=cap_q(now_gap, q=75, f=1.0)).fillna(0)

    ers_bits = (
        (ebs >= PASS_EBS).astype(int)
        + (slope > 0).astype(int)
        + ((rsi >= RSI_LOW) & (rsi <= RSI_HIGH)).astype(int)
    )
    ers_norm = np.clip(ers_bits / 3.0, 0, 1).fillna(0)
    slope_pos_norm = pct_norm(slope, q=90, f=1.0).fillna(0)
    mom_mid_norm = pct_norm(r10.clip(lower=0), q=90, f=1.0).fillna(0)
    mom_norm = np.clip(0.5 * ers_norm + 0.3 * slope_pos_norm + 0.2 * mom_mid_norm, 0, 1).fillna(0)

    if turn.notna().any():
        try:
            lo, hi = np.nanpercentile(turn.dropna(), 30), np.nanpercentile(turn.dropna(), 90)
            denom = max(hi - lo, 1e-9)
            liq_norm = np.clip((turn - lo) / denom, 0, 1).fillna(0)
            liq_low = (turn < lo).astype(float)
        except Exception:
            liq_norm = pd.Series(0.0, index=x.index)
            liq_low = pd.Series(0.0, index=x.index)
    else:
        liq_norm = pd.Series(0.0, index=x.index)
        liq_low = pd.Series(0.0, index=x.index)

    vol_sweet = (1 - np.minimum((volz - 1).abs() / 3, 1)).clip(0, 1).fillna(0)
    kairi_abs = kairi.abs()
    kairi_norm = (1 - np.minimum(kairi_abs / cap_q(kairi_abs, q=80, f=3.0), 1)).clip(0, 1).fillna(0)
    tec_norm = np.clip(0.6 * vol_sweet + 0.4 * kairi_norm, 0, 1).fillna(0)

    base_score = (
        100 * W_RR * rr_norm
        + 100 * W_T1 * t1_norm
        + 100 * W_SL * sl_norm
        + 100 * W_NEAR * near_norm
        + 100 * W_MOM * mom_norm
        + 100 * W_LIQ * liq_norm
        + 100 * W_TEC * tec_norm
    )

    pen = pd.Series(0.0, index=x.index)
    pen += P_OVERHEAT_5D * np.clip((r5 - 10) / 10, 0, 1).fillna(0)
    pen += P_OVERHEAT_10D * np.clip((r10 - 25) / 25, 0, 1).fillna(0)
    pen += P_RSI_OUT * ((rsi < RSI_LOW) | (rsi > RSI_HIGH)).astype(float)
    pen += P_MACD_NEG * (slope < 0).astype(float)
    pen += P_NEAR_FAR * np.clip((now_gap - 15) / 15, 0, 1).fillna(0)
    pen += P_LIQ_LOW * liq_low
    pen += P_VOL_SPIKE * (volz > 3).astype(float)

    score = np.clip(base_score - pen, 0, 100)

    x["RR1"] = rr1
    x["Now%"] = now_gap
    x["T1_ROOM%"] = t1_room
    x["SL_ROOM%"] = sl_room
    x["LDY_SCORE"] = score.round(1)

    x["_GATE_OK"] = liquidity_gate(
        x["거래대금(억원)"],
        x.get("시장", pd.Series(np.nan, index=x.index))
    ).fillna(False)

    if "MA20" in x.columns:
        x["MA20_GAP"] = ((nz_num(x["종가"]) / nz_num(x["MA20"]) - 1.0) * 100).replace([np.inf, -np.inf], np.nan)
    else:
        x["MA20_GAP"] = np.nan

    # ✅ 여기부터가 핵심
    if not keep_order:
        x = x.sort_values("LDY_SCORE", ascending=False, na_position="last")
        x["LDY_RANK"] = range(1, len(x) + 1)
    # keep_order=True면 CSV 순서 유지 (LDY_RANK는 밖에서 결정)

    if "AI_COMMENT" in x.columns:
        x["WHY"] = x["AI_COMMENT"]

    return x

# ---------------------------
# 동적 라우트(분포기반 임계값) 적용
# ---------------------------
def compute_dynamic_thresholds(df):
    thr = {}

    if 'ret_5d_%' in df.columns:
        s = pd.to_numeric(df['ret_5d_%'], errors='coerce')
        thr['r5_q75'] = float(np.nanpercentile(s.dropna(), 75)) if s.dropna().size > 0 else 1.0
    else:
        thr['r5_q75'] = 1.0

    slope_col = None
    if "MACD_Slope" in df.columns:
        slope_col = "MACD_Slope"
    elif "MACD_slope" in df.columns:
        slope_col = "MACD_slope"

    if slope_col:
        s = pd.to_numeric(df[slope_col], errors='coerce')
        thr['slope_q60'] = float(np.nanpercentile(s.dropna(), 60)) if s.dropna().size > 0 else 0.0
    else:
        thr['slope_q60'] = 0.0

    if 'EBS' in df.columns:
        s = pd.to_numeric(df['EBS'], errors='coerce')
        thr['ebs_q60'] = float(np.nanpercentile(s.dropna(), 60)) if s.dropna().size > 0 else PASS_EBS
    else:
        thr['ebs_q60'] = PASS_EBS

    if 'Now%' in df.columns:
        s = pd.to_numeric(df['Now%'], errors='coerce')
        thr['now_gap_q25'] = float(np.nanpercentile(s.dropna(), 25)) if s.dropna().size > 0 else 10.0
    else:
        thr['now_gap_q25'] = 10.0

    for k, v in list(thr.items()):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            thr[k] = 0.0

    return thr

def route_tag_dynamic(row, th):
    # ✅ [수정됨] 0 값을 안전하게 처리하는 헬퍼 함수
    def _get_val(key, default):
        val = row.get(key)
        if val is None or pd.isna(val):
            return default
        try:
            return float(val)
        except:
            return default

    r5 = _get_val("ret_5d_%", 0.0)

    # 🚨 [핵심 수정] CSV에는 'MACD_Slope_PCT'에 값이 들어있습니다. 이걸 먼저 가져와야 합니다.
    slope = _get_val("MACD_Slope_PCT", 0.0) 
    if slope == 0.0:
         slope = _get_val("MACD_Slope", 0.0) # Fallback

    ebs = _get_val("EBS", 0.0)
    now_pct = _get_val("Now%", 999.0) # 0이어도 999로 바뀌지 않음
    rr1 = _get_val("RR1", 0.0)
    ma20_gap = _get_val("MA20_GAP", 0.0)

    # 1) TTM Squeeze (폭발 대기)
    # TTM_SQUEEZE 컬럼이 1이면 무조건 SQZ 태그 우선
    is_sqz = _get_val("TTM_SQUEEZE", 0.0)
    if is_sqz == 1.0:
        return "🔥 SQZ (폭발대기)"

    # 2) 강한 돌파 BRK
    strong = (
        (r5 >= th['r5_q75'])
        and (slope >= th['slope_q60'])
        and (ebs >= th['ebs_q60'])
        and (now_pct <= th['now_gap_q25'])
    )
    if strong and rr1 >= 0.5:
        return "🔼 BRK (강력 돌파)"

    # 3) Watch 영역
    if (slope > 0 and r5 > 0) or (ebs >= th['ebs_q60'] and now_pct <= th['now_gap_q25'] * 1.5):
        if r5 >= max(1.0, th['r5_q75'] * 0.6) and slope > 0:
            return "🔺 Watch→BRK (관찰·돌파예상)"
        return "🔺 Watch (상승 준비)"

    # 4) 20일선 위 강세
    if ma20_gap > 1 and slope > 0 and ebs >= PASS_EBS:
        return "🔼 BRK (MA20상승)"

    return "↩️ PULL (눌림)"

# 👉 데이터 기준일 추론
def infer_data_timestamp(df_raw: pd.DataFrame):
    """
    recommend_latest.csv 안에서 '기준일', '날짜', 'Date' 같은 컬럼을 찾아
    가장 최신 날짜를 기준 시각으로 추출.
    - 2000년 이전, 오늘+1일 이후 값은 버림
    - YYYYMMDD 형태도 별도 처리
    """
    if df_raw is None or df_raw.empty:
        return None

    candidates = []
    now_utc_val = now_utc()

    # 1차: 일반 datetime 컬럼 후보
    date_cols = ["기준일자", "기준일", "날짜", "DATE", "Date", "date", "update_time", "updated_at"]
    for col in date_cols:
        if col in df_raw.columns:
            s = pd.to_datetime(df_raw[col], errors="coerce", utc=True)
            # 🔹 현실적인 범위만 허용
            s = s[(s.notna()) &
                  (s >= pd.Timestamp("2000-01-01", tz="UTC")) &
                  (s <= now_utc_val + pd.Timedelta(days=1))]
            if not s.empty:
                candidates.append(s.max())

    # 2차: YYYYMMDD 숫자/문자 컬럼 처리
    if not candidates:
        ymd_cols = ["기준일자", "기준일", "날짜", "DATE", "Date"]
        for col in ymd_cols:
            if col in df_raw.columns:
                raw = df_raw[col].astype(str).str.replace(r"[^0-9]", "", regex=True)
                s = pd.to_datetime(raw, format="%Y%m%d", errors="coerce", utc=True)
                s = s[(s.notna()) &
                      (s >= pd.Timestamp("2000-01-01", tz="UTC")) &
                      (s <= now_utc_val + pd.Timedelta(days=1))]
                if not s.empty:
                    candidates.append(s.max())

    if candidates:
        # 여러 후보가 있다면 가장 최신값 반환 (UTC)
        return max(candidates)

    return None
# 👈 데이터 기준일 추론 끝

@st.cache_data(ttl=300)
def reality_check_top(df_top: pd.DataFrame, data_ts, n: int = 5):
    """
    recommend_latest.csv 기준 상위 n개 추천 종목에 대해
    - 기준일 종가 vs 현재가 수익률
    - 평균 수익률 / 적중 개수
    를 계산해서 대시보드 상단에 보여줄 요약값을 리턴.
    """
    if df_top is None or df_top.empty or not FDR_OK:
        return None

    df = df_top.head(n).copy()
    results = []
    hit = 0
    cnt = 0

    for _, row in df.iterrows():
        code = str(row.get("종목코드", "")).zfill(6)
        name = row.get("종목명", code)
        base_price = pd.to_numeric(row.get("추천매수가", np.nan), errors="coerce")
        if pd.isna(base_price) or base_price <= 0:
            base_price = pd.to_numeric(row.get("종가", np.nan), errors="coerce")

        try:
            # 최근 7일 사이 데이터에서 마지막 종가 사용
            start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            df_price = fdr.DataReader(code, start)
            if df_price is None or df_price.empty:
                continue
            cur_price = float(df_price["Close"].iloc[-1])
        except Exception:
            continue

        if cur_price <= 0:
            continue

        cnt += 1
        ret_pct = (cur_price - base_price) / base_price * 100
        if ret_pct > 0:
            hit += 1
        results.append(ret_pct)

    if cnt == 0:
        return None

    avg_ret = float(np.mean(results))

    # 기준일 문자열
    if data_ts is not None:
        base_str = to_kst_str(data_ts, fmt="%m/%d")
    else:
        base_str = "기준일 미상"

    return {
        "base_str": base_str,
        "avg_ret": avg_ret,
        "hit": hit,
        "count": cnt,
    }

@st.cache_data(ttl=600, show_spinner=False)
def prepare_scored_data(raw_url, local_raw, pass_ebs):
    """
    - CSV 로드 (원격 → 실패 시 로컬)
    - normalize_cols
    - build_global_score(keep_order=True)로 계산은 하되 정렬은 여기서 통제
    - CSV에 랭크 컬럼이 있으면 그걸 LDY_RANK로 고정
    - ROUTE/TH 계산
    - base/top20 생성
    - data_ts / src_type 반환
    """
    df_raw = None
    src_type = "unknown"

    # 1) CSV 로드
    try:
        df_raw = load_csv_url(raw_url)
        df_raw = postprocess_codes(df_raw)   # ✅ 여기 한 줄 추가
        log_src(df_raw, "Remote")
        src_type = "remote"
    except Exception as e_remote:
        logger.warning("prepare_scored_data: Remote load failed: %s", e_remote)
        if local_raw and os.path.exists(local_raw):
            df_raw = load_csv_path(local_raw)
            log_src(df_raw, "Local")
            src_type = "local"

    if df_raw is None or df_raw.empty:
        raise RuntimeError("CSV를 원격/로컬 어디서도 불러오지 못했습니다.")

    # ✅ 2) CSV 순서/랭크 안정화(여기가 반드시 raise 밖이어야 함)
    df_raw = df_raw.copy().reset_index(drop=True)

    rank_col = None
    for c in ["LDY_RANK", "RANK", "rank", "순위", "랭크"]:
        if c in df_raw.columns:
            rank_col = c
            break

    if rank_col:
        df_raw["_CSV_RANK"] = pd.to_numeric(df_raw[rank_col], errors="coerce")
    else:
        df_raw["_CSV_RANK"] = np.arange(1, len(df_raw) + 1)

    df_raw["_CSV_ROW"] = np.arange(len(df_raw))

    # 3) 기준 시점 추출
    data_ts = infer_data_timestamp(df_raw)

    # 4) 스코어링 (정렬 금지)
    df = normalize_cols(df_raw)
    scored = build_global_score(df, keep_order=True).reset_index(drop=True)

    # ✅ df_raw 기준 랭크/행순서 붙이기 (row align 전제)
    scored["_CSV_RANK"] = df_raw["_CSV_RANK"].values
    scored["_CSV_ROW"]  = df_raw["_CSV_ROW"].values

    # 5) TH/ROUTE 계산 (1회만)
    TH = compute_dynamic_thresholds(scored)
    scored["ROUTE"] = scored.apply(lambda r: route_tag_dynamic(r, TH), axis=1).fillna("—")

    # ✅ 표시/기본 랭크는 CSV 기준 고정
    scored = scored.sort_values(["_CSV_RANK", "_CSV_ROW"], ascending=[True, True]).reset_index(drop=True)
    scored["LDY_RANK"] = pd.to_numeric(scored["_CSV_RANK"], errors="coerce")

    # 6) base/top20
    base = scored[(pd.to_numeric(scored["EBS"], errors="coerce") >= pass_ebs) & (scored["_GATE_OK"])].copy()
    if len(base) < 20:
        base = scored.head(20).copy()

    top20 = base.head(20).copy()
    top20["P_hit"] = (top20["LDY_SCORE"] / 100.0 * 0.8).clip(0, 1) * 100

    return scored, base, top20, TH, data_ts, src_type


# ---------------------------
# 메인 데이터 로드 (Status UX)
# ---------------------------

# 전역에서 쓸 수 있게 기준 시점 / 데이터 출처 변수 선언
DATA_TS = None
DATA_SRC = None   # remote/local 태그용

with st.status("🚀 시장 데이터를 분석하고 있습니다...", expanded=True) as status:
    status.write("📥 데이터 다운로드 및 스코어링 계산 중...")
    try:
        # 🔧 RAW_URL → RAW_SRC 로 수정
        scored, base, top20, TH, DATA_TS, DATA_SRC = prepare_scored_data(
            RAW_SRC,
            LOCAL_RAW,
            PASS_EBS,
        )

        # get_market_status / get_fear_greed_index fallback용


        status.write("🌊 동적 유동성 필터 적용 중...")
        status.update(label="✅ 분석 완료!", state="complete", expanded=False)
    except Exception as e:
        status.update(label="❌ 데이터 로드 실패", state="error")
        st.error(f"데이터 로드/스코어링 중 오류: {e}")
        st.stop()

# 첫 가입 직후 표시용 플래그
just_registered = st.session_state.pop("just_registered", False)

# ---------------------------
# Sidebar (Auth / Portfolio / Subscription)
# ---------------------------


with st.sidebar:
    user = render_auth_box(show_debug=False)

    if user is None:
        auth_status = "guest"
        expire_str = None
        st.caption("현재 상태: 🔒 Guest (비로그인)")
    else:
        auth_status, expire_str = sync_user_role_with_subscription(user)
        if auth_status != user.get("role"):
            user["role"] = auth_status
            st.session_state["ldy_current_user"] = user

        if expire_str:
            st.caption(f"현재 상태: **{auth_status.upper()}** (만료일: {expire_str})")
        else:
            st.caption(f"현재 상태: **{auth_status.upper()}**")

    st.divider()
    st.subheader("💎 프리미엄 이용권 안내")

    PRICE_PRO = 19000
    PRICE_PRIME = 39000

    # 🌱 Free
    with st.container():
        st.markdown("### 🌱 **Free (무료)**")
        st.markdown(
            "- ✅ **회원가입 후** 상위 **5개 종목** 조회 (Guest는 3개)\n"
            "- ✅ 시장 지표 / 섹터맵 열람\n"
            "- ❌ 내 포트폴리오 분석\n"
            "- ❌ CSV 다운로드 / 알림"
        )

    # 🚀 Pro 1개월 이용권
    with st.container():
        st.markdown(f"### 🚀 **Pro 1개월 이용권 ({PRICE_PRO:,}원)**")
        st.markdown(
            "실전 투자자용, **데이터 기반 종목 선별에 집중하고 싶은 투자자에게 추천드립니다.**\n\n"
            "- 🔓 필터 적용 **Top 20 종목** 열람\n"
            "- 💼 **내 자산(포트폴리오)** 수익률 분석\n"
            "- 📊 개별 종목 레이더 · 리스크/리워드 차트\n"
            "- ❌ CSV 다운로드\n"
            "- ❌ 텔레그램 알림"
        )

    # 👑 Prime 1개월 이용권
    with st.container():
        st.markdown(f"### 👑 **Prime 1개월 이용권 ({PRICE_PRIME:,}원)**")
        st.markdown(
            "전업 / 하이엔드 투자자용, **시장 전체 스코어를 풀로 열람하고 싶은 분께 권장드립니다.**\n\n"
            "- ✅ **전체 스코어링 종목** 열람\n"
            "- ✅ CSV 다운로드\n"
            "- ✅ 텔레그램 요약 알림 (Top 종목 브리핑)\n"
            "- ✅ 향후 고급 리포트 / 신규 기능 우선 적용"
        )

    # 🔹 PRIME 전용 텔레그램 채널 안내 (로그인 + PRIME 이상 전용)
    if auth_status in ["prime", "admin"]:
        if PRIME_TG_JOIN_URL:
            st.markdown("#### 🔔 PRIME 전용 텔레그램 채널")
            try:
                st.link_button(
                    "👑 PRIME 채널 입장하기",
                    PRIME_TG_JOIN_URL,
                    use_container_width=True,
                    type="primary",
                )
            except Exception:
                st.markdown(f"[👑 PRIME 채널 입장하기]({PRIME_TG_JOIN_URL})")
        else:
            st.caption("※ PRIME 전용 텔레그램 채널 URL이 아직 설정되지 않았습니다. (LDY_PRIME_JOIN_URL 환경변수 확인 요망)")
    else:
        st.caption("※ PRIME 등급이 되면 텔레그램 **전용 채널 입장 링크**가 열립니다.")

    # 💳 결제(입금) 안내
    st.markdown("#### 💳 결제(입금) 안내")
    st.markdown(
        "이 서비스는 **자동 결제가 없는 ‘1개월 이용권(30일 패스)’** 방식입니다.  \n"
        "원하실 때마다 1개월 단위로만 선결제하여 사용하실 수 있습니다.\n\n"
        f"- 입금계좌: **{BANK_ACCOUNT}**  \n"
        f"- 예금주: **{BANK_HOLDER}**  \n\n"
        "입금 후 **카카오톡 채널 또는 문의 게시판**에  \n"
        "👉 입금자명 / 이메일 / 희망 이용권(Pro 또는 Prime)  \n"
        "을 남겨 주세요.\n\n"
        "관리자가 입금 내역을 확인한 뒤, 해당 계정에 Pro / Prime 권한을 부여하며  \n"
        "**부여일로부터 30일간** 프리미엄 기능이 활성화됩니다.\n\n"
        "이용 기간이 종료된 후 계속 사용을 원하실 경우,  \n"
        "동일한 방식으로 다시 **1개월 이용권을 결제**해 주세요."
    )

    if user and expire_str:
        st.info(f"현재 이용권 만료 예정일: **{expire_str}**")

    kakao_url = "https://open.kakao.com/o/soKqY04h"
    try:
        st.link_button("👉 구독/입금 확인 문의 (카톡)", kakao_url, type="primary", use_container_width=True)
    except Exception:
        st.markdown(f"[👉 구독/입금 확인 문의 (카톡)]({kakao_url})")

    # Pro 이상 포트폴리오
    if auth_status in ["pro", "prime", "admin"]:
        st.divider()
        st.subheader("💼 내 자산 관리")
        saved_pf = load_portfolio_file()
        pf_input = st.text_area(
            "종목명:평단가:수량",
            value=saved_pf,
            placeholder="NAVER:261000:10",
            height=100,
        )
        if st.button("💾 저장/분석", key="pf_btn"):
            save_portfolio_file(pf_input)
            st.success("저장되었습니다")
    else:
        pf_input = ""

    # Prime 이상 텔레그램
    send_btn = False
    tg_token, tg_chat_id = "", ""
    if auth_status in ["prime", "admin"]:
        with st.expander("🔔 텔레그램 봇"):
            tg_token = st.text_input("Token", type="password")
            tg_chat_id = st.text_input("ChatID")
            send_btn = st.button("🚀 전송")

# 관리자 전용: 회원 권한 + 구독 만료일 관리
    if auth_status == "admin":
        st.divider()
        st.subheader("👑 회원 관리 (Admin)")

        users = list_users()
        
        # 1. 관리자 대시보드 통계 (DAU/WAU)
        if users:
            total_users = len(users)
            now_utc_dt = datetime.now(timezone.utc)
            dau_count = 0  # Daily Active Users
            wau_count = 0  # Weekly Active Users
            
            for u in users:
                last_s = u.get("last_login")
                if last_s:
                    try:
                        last_dt = datetime.fromisoformat(last_s.replace("Z", "+00:00"))
                        diff = now_utc_dt - last_dt
                        if diff < timedelta(days=1): dau_count += 1
                        if diff < timedelta(days=7): wau_count += 1
                    except: pass
            
            dau_pct = f"{dau_count/total_users*100:.1f}%"
            wau_pct = f"{wau_count/total_users*100:.1f}%"
            
            st.markdown(f"""
            <div style="background:rgba(128,128,128,0.1); padding:10px; border-radius:5px; margin-bottom:15px; font-size:0.9em;">
                <div>👥 <b>총 가입자:</b> {total_users}명</div>
                <div>🔥 <b>DAU (24h):</b> {dau_count}명 ({dau_pct})</div>
                <div>📅 <b>WAU (7일):</b> {wau_count}명 ({wau_pct})</div>
            </div>
            """, unsafe_allow_html=True)

        if not users:
            st.info("회원이 없습니다.")
        else:
            # 2. 회원 목록 테이블 (구독 만료일 연동 + 필터링 기능 추가)
            subs_db = load_subs_db() # subscriptions_db.json 로드
            subs = subs_db.get("subs", {})
            rows = []
            
            today = now_kst().date() # 오늘 날짜 (만료 여부 비교용)

            for u in users:
                email = u.get("login_id")
                role = u.get("role", "free")
                
                # 차단 여부
                is_banned = u.get("is_banned", False)
                
                # 만료일 확인 및 상태 결정
                expire_at_str = "-"
                is_expired = False
                
                if role == "admin":
                    expire_at_str = "∞ (Admin)"
                elif email in subs:
                    # 구독 DB에 정보가 있으면 가져옴
                    expire_at_str = subs[email].get("expire_at", "-")
                    try:
                        # 날짜 비교: 만료일이 오늘보다 이전이면 만료됨 처리
                        exp_date = datetime.strptime(expire_at_str, "%Y-%m-%d").date()
                        if exp_date < today:
                            is_expired = True
                    except:
                        pass
                
                # 상태 텍스트 결정 (우선순위: 차단 > 만료 > 정상)
                if is_banned:
                    status_txt = "🚫차단됨"
                elif is_expired:
                    status_txt = "❌만료됨"
                else:
                    status_txt = "✅정상"

                rows.append({
                    "Email": email,
                    "닉네임": u.get("nickname"),
                    "권한": role,
                    "만료일": expire_at_str,  # ✅ 추가됨
                    "상태": status_txt,       # ✅ 업데이트됨
                    "최근접속": to_kst_str(u.get("last_login")),
                    "_is_expired": is_expired # 필터링용 히든 컬럼
                })

            df_users = pd.DataFrame(rows)
            
            # 🔥 [UI 기능 추가] 만료 회원 필터링 & 검색 옵션
            c_filter1, c_filter2 = st.columns(2)
            with c_filter1:
                show_expired = st.checkbox("📉 만료된 회원만 보기")
            with c_filter2:
                search_query = st.text_input("🔍 이메일 검색", placeholder="user@example.com")

            # 필터 로직 적용
            if show_expired:
                df_users = df_users[df_users["_is_expired"] == True]
            
            if search_query:
                # 대소문자 구분 없이 검색
                df_users = df_users[df_users["Email"].str.contains(search_query, case=False, na=False)]

            # 최근 접속순 정렬
            if not df_users.empty and "최근접속" in df_users.columns:
                df_users = df_users.sort_values("최근접속", ascending=False)
                
            # 테이블 출력
            st.dataframe(
                df_users.drop(columns=["_is_expired"]), # 히든 컬럼 제외하고 출력
                use_container_width=True, 
                height=300,
                column_config={
                    "최근접속": st.column_config.TextColumn("최근접속", width="medium"),
                    "만료일": st.column_config.TextColumn("만료일", width="small"), # ✅ UI 표시 설정
                    "권한": st.column_config.TextColumn("권한", width="small"),
                    "상태": st.column_config.TextColumn("상태", width="small"),
                }
            )

            # 3. 통합 계정 제어 (권한 변경 + 차단)
            st.markdown("##### 🛠️ 계정 제어")
            
            # (만료된 회원도 리스트에 나와야 제어가 가능하므로 원본 리스트 사용 권장)
            target_list = df_users["Email"].tolist() if not df_users.empty else []
            target_email = st.selectbox("대상 회원 선택", options=target_list, key="admin_target_unified")
            
            c_adm1, c_adm2 = st.columns(2)
            
            # [왼쪽] 권한 변경
            with c_adm1:
                new_role = st.selectbox("권한", ["free", "pro", "prime", "admin"], key="admin_role_unified")
                if st.button("권한 적용", type="primary", use_container_width=True):
                    if update_user_role(target_email, new_role, user.get("login_id")):
                        set_subscription(target_email, new_role)
                        st.success(f"변경 완료: {new_role}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("변경 실패")
            
            # [오른쪽] 차단 토글
            with c_adm2:
                # 현재 차단 상태 확인
                current_ban = False
                if target_email:
                    # users 원본 딕셔너리에서 찾아야 정확함
                    u_target = next((u for u in users if u["login_id"] == target_email), None)
                    if u_target:
                        current_ban = u_target.get("is_banned", False)

                btn_label = "⭕ 차단 해제" if current_ban else "🚫 계정 차단"
                btn_type = "primary" if current_ban else "secondary"
                
                st.write("") 
                st.write("") 
                if st.button(btn_label, type=btn_type, use_container_width=True):
                    ok, msg = toggle_user_ban(target_email, user.get("login_id"))
                    if ok:
                        st.warning(msg) if not current_ban else st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)

# ---------------------------
# Telegram send
# ---------------------------
if send_btn and tg_token and tg_chat_id:
    msg = f"🔥 [LDY v{APP_VERSION}] 추천 Top 5 ({now_kst().strftime('%m/%d')})\n\n"
    for i in range(min(5, len(top20))):
        row = top20.iloc[i]
        msg += f"{i+1}. {row.get('종목명','-')} ({row.get('ROUTE','-')})\n"
        msg += f"   매수: {int(row.get('추천매수가',0)):,} / 손절: {int(row.get('손절가',0)):,}\n\n"
    ok, res = send_telegram_msg(tg_token, tg_chat_id, msg)
    if ok:
        st.toast("전송 완료!", icon="✅")
    else:
        st.error(f"전송 실패: {res}")

df_latest = load_recommend_latest(local_path=RECOMMEND_LATEST_PATH, remote_url=RAW_SRC)
user = get_user()
user_role = (user or {}).get("role", "guest")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 시장 (Market)",
        "🔭 종목 분석",
        "💼 내 자산",
        "📮 문의 게시판",
        "⚖️ 이용 약관 / 투자 유의사항",
        "🧩 LDY Pro Trader 업데이트 노트",
    ]
)

with tab1:
    # 🔥 v6.8 Reality Check: 지난 추천 성과 요약
    rc = reality_check_top(top20, DATA_TS, n=5)
    if rc is not None:
        msg = (
            f"📅 {rc['base_str']} 추천 Top {rc['count']} 기준, "
            f"현재 평균 수익률 **{rc['avg_ret']:+.2f}%** "
            f"(적중 {rc['hit']}/{rc['count']})"
        )
        st.success(msg)
    else:
        st.caption("※ FDR 데이터 또는 추천 데이터가 부족해 성과 검증을 표시할 수 없습니다.")

    kp_stat, kp_diff, kq_stat, kq_diff = get_market_status(scored)
    c1, c2 = st.columns(2)

    def _fmt_metric(stat, diff):
        bad_stats = {
            "데이터 없음",
            "데이터 오류",
            "데이터 소스 오류",
            "데이터 부족",
            "Unknown",
            "Error",
        }
        if stat in bad_stats or pd.isna(diff):
            friendly = "📡 지수 데이터 지연/점검 중"
            return friendly, "-", "off"

        delta_txt = f"{diff:.2f}%"
        delta_color = "off" if ("상승" in stat or diff >= 0) else "inverse"
        return stat, delta_txt, delta_color

    kp_value, kp_delta, kp_color = _fmt_metric(kp_stat, kp_diff)
    kq_value, kq_delta, kq_color = _fmt_metric(kq_stat, kq_diff)

    c1.metric("KOSPI", kp_value, kp_delta, delta_color=kp_color)
    c2.metric("KOSDAQ", kq_value, kq_delta, delta_color=kq_color)


    # 👇 [여기 삽입] 🔥 [v8.0] 매크로(환율/미증시) 메트릭 및 리스크 배너
    macro_data = get_macro_metrics()
    if macro_data:
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        
        # 환율
        if "USD" in macro_data:
            val, diff = macro_data["USD"]
            # 1400원 넘으면 경고색 (inverse: 빨강/파랑 반전 효과 활용 or 직접 지정)
            usd_color = "inverse" if val >= 1400 else "normal" 
            m1.metric("USD/KRW (환율)", f"{val:,.1f}원", f"{diff:+.1f}원", delta_color=usd_color)
            
        # 나스닥
        if "IXIC" in macro_data:
            val, pct = macro_data["IXIC"]
            # -2% 이상 하락 시 경고색
            nas_color = "inverse" if pct <= -2.0 else "normal"
            m2.metric("NASDAQ (나스닥)", f"{val:,.0f}", f"{pct:+.2f}%", delta_color=nas_color)
            
        # 리스크 상태 요약
        risk_msg = "✅ 평온 (Normal)"
        if "USD" in macro_data and macro_data["USD"][0] >= 1400:
            risk_msg = "⚠️ 주의 (고환율)"
        if "IXIC" in macro_data and macro_data["IXIC"][1] <= -2.0:
            risk_msg = "🚨 위험 (미증시 급락)"
            
        m3.metric("시장 리스크 모드", risk_msg)

    # 🔥 v6.5: 데이터 기준 시각 + 지표 모드 + 소스 태그 + 신선도 경고
    fg_score, fg_status = get_fear_greed_index(scored)

    info_lines = []

    # 0) 데이터 소스 태그
    if DATA_SRC == "remote":
        info_lines.append("📡 데이터 출처: **GitHub 원격 CSV** (실시간 반영)")
    elif DATA_SRC == "local":
        info_lines.append("📁 데이터 출처: **로컬 캐시 파일** (네트워크 장애 시 대체 사용)")
    else:
        info_lines.append("📡 데이터 출처: **알 수 없음** (코드/환경 확인 필요)")

    # 1) 추천 데이터 기준 일자
    if DATA_TS is not None:
        ts_date = to_kst_str(DATA_TS, fmt="%Y-%m-%d")
        if ts_date:
            info_lines.append(f"📅 추천 데이터 기준 일자: **{ts_date} (KST)**")

            # 신선도 경고 (기준일이 2일 이상 지났을 때)
            try:
                ts_kst = pd.to_datetime(DATA_TS).tz_convert(KST)
                days_diff = (now_kst().date() - ts_kst.date()).days
                if days_diff >= 2:
                    info_lines.append(
                        f"⚠️ 기준일이 **{days_diff}일** 지났습니다. "
                        "GitHub의 `recommend_latest.csv` 업데이트 여부를 확인해 주세요."
                    )
            except Exception:
                pass

    # 2) 지수/스코어 기준 여부 요약
    mode_bits = []

    if "스코어 기반" in str(kp_stat) or "스코어 기반" in str(kq_stat):
        mode_bits.append("시장 상태: 🔄 **로컬 스코어 기반 추정**")
    else:
        mode_bits.append("시장 상태: 📡 **지수(FDR/pykrx) 기준**")

    if "스코어 기준" in fg_status:
        mode_bits.append("공포/탐욕: 📊 **스코어 기준**")
    elif "지수 기준" in fg_status:
        mode_bits.append("공포/탐욕: 📈 **지수 기준**")

    if mode_bits:
        info_lines.append(" · ".join(mode_bits))

    # 3) KOSPI/KOSDAQ 퍼센트 계산 방식 설명 추가
    use_local_market = ("스코어 기반" in str(kp_stat)) or ("스코어 기반" in str(kq_stat))
    if use_local_market:
        info_lines.append(
            "※ KOSPI/KOSDAQ 퍼센트 값은 지수 데이터 장애 시 "
            "**최근 5영업일 평균 수익률**을 기반으로 한 로컬 추정치입니다."
        )
    else:
        info_lines.append(
            "※ KOSPI/KOSDAQ 퍼센트 값은 지수 종가와 **20일 이동평균선 괴리율(%)** 기준입니다."
        )

    if info_lines:
        st.caption("  \n".join(info_lines))


    st.divider()

    # 공포/탐욕 게이지 + 섹터맵
    c_gauge, c_map = st.columns([1, 1.5])
    # 🚨 [수정] 공포/탐욕 게이지와 섹터맵을 모바일에서 보기 좋게 변경
    # PC에서는 옆으로, 모바일에서는 위아래로 자연스럽게 배치되도록
    # Streamlit은 화면이 좁으면 자동으로 수직 배치하지만, 
    # [1, 1.5] 비율 강제보다는 1:1이 모바일에서 찌그러짐을 방지함.
    c_gauge, c_map = st.columns([1, 1]) 
    
    with c_gauge:
        st.plotly_chart(
            plot_fear_greed_gauge(fg_score),
            use_container_width=True,
            # 모바일에서 게이지가 너무 작아지지 않게 높이 약간 확보
            config={'staticPlot': True} # 터치 오동작 방지
        )
        st.caption(f"시장 공포/탐욕 지수 — {fg_status}")
    
    with c_map:
        st.markdown("##### 🔥 오늘의 주도 섹터")
        map_src = st.radio(
            "섹터맵 기준 데이터",
            options=["EBS/유동성 통과 종목", "전체 상위 Top 50"],
            horizontal=True,
            key="sector_data_src",
        )
        if "업종" in scored.columns:
            if map_src == "EBS/유동성 통과 종목":
                map_df = base.copy()
            else:
                map_df = scored.head(50).copy()
            fig = plot_sector_treemap(map_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("섹터 데이터 부족")
        else:
            st.info("섹터 정보 없음")

    st.divider()
    st.markdown("##### 🚀 섹터 모멘텀 Top 10")
    mom_fig = plot_sector_momentum_bar(scored)
    if mom_fig and len(mom_fig.data) > 0:
        st.plotly_chart(mom_fig, use_container_width=True)
    else:
        st.caption("※ 섹터 모멘텀을 계산할 수 있는 데이터가 부족합니다.")

    # 👇 [여기 추가!] 이 두 줄을 tab1 맨 마지막에 넣으세요
    st.divider()
    plot_regime_summary(scored)

def _to_num(x, default=np.nan):
    v = pd.to_numeric(x, errors="coerce")
    return default if pd.isna(v) else float(v)

def _to_int_str(x, default=0):
    v = pd.to_numeric(x, errors="coerce")
    return f"{int(v) if pd.notna(v) else default:,}"


with tab2:
    st.subheader("🎯 AI & Quant 추천 종목")

    # ✅ [v8.5] 회원가입 직후 Top 5 프리뷰
    if just_registered:
        st.success("🎉 첫 가입을 환영합니다! 오늘 기준 TOP 5 프리뷰를 먼저 보여드릴게요.")
        try:
            preview = make_preview(base, n=5)
        except Exception:
            preview = make_preview(scored, n=5)

        if not preview.empty:
            cols = ["종목명", "종목코드", "LDY_SCORE", "추천매수가", "손절가", "추천매도가1"]
            cols = [c for c in cols if c in preview.columns]
            
            prev_view = preview[cols].copy()
            fmt_cols = ["추천매수가", "손절가", "추천매도가1"]
            for c in fmt_cols:
                if c in prev_view.columns:
                    prev_view[c] = pd.to_numeric(prev_view[c], errors='coerce').fillna(0).apply(lambda x: f"{int(x):,}")

            if "종목명" in prev_view.columns:
                prev_view = prev_view.set_index("종목명")
            
            st.dataframe(prev_view, use_container_width=True)
        else:
            st.info("프리뷰로 표시할 종목이 없습니다.")
        
        st.session_state["just_registered"] = False
        st.divider()

    # ---------------------------
    # 필터링 위젯
    # ---------------------------
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        min_score = st.slider(
            "최소 퀀트(LDY) 점수",
            min_value=0, max_value=100, value=70, step=1, # AI 도입으로 기본값 조금 완화
            key="min_score",
        )

    with col_f2:
        def _route_order(r: str):
            s = str(r)
            if "SQZ" in s: return (0, s)
            if "BRK" in s: return (1, s)
            if "Watch" in s or "관찰" in s or "상승" in s: return (2, s)
            if "MR" in s: return (3, s)
            if "PULL" in s: return (4, s)
            return (5, s)

        all_routes = sorted(
            scored["ROUTE"].dropna().unique().tolist(),
            key=_route_order
        ) if "ROUTE" in scored.columns else []
        
        if all_routes:
            default_routes = [r for r in all_routes if "PULL" not in r] or all_routes
            sel_routes = st.multiselect(
                "전략 유형 (ROUTE)",
                options=all_routes,
                default=default_routes,
                key="route_filter",
            )
        else:
            sel_routes = []

    with col_f3:
        if "REGIME" in scored.columns:
            all_regimes = sorted(scored["REGIME"].dropna().unique().tolist())
            sel_regimes = st.multiselect(
                "추세 구분 (REGIME)",
                options=all_regimes,
                default=all_regimes,
                key="regime_filter",
            )
        else:
            sel_regimes = []

    use_only_gate = st.checkbox("EBS/유동성 통과만 사용", value=True, key="only_gate")

    # ---------------------------
    # 데이터 필터링 로직
    # ---------------------------
    if use_only_gate:
        if auth_status in ["prime", "admin"]:
            base_view = base.head(300).copy()
        else:
            base_view = top20.copy()
    else:
        if auth_status in ["prime", "admin"]:
            base_view = scored.copy()
        else:
            base_view = scored.head(50).copy()

    filtered = base_view.copy()
    filtered = filtered[filtered["LDY_SCORE"] >= min_score]

    if sel_routes:
        filtered = filtered[filtered["ROUTE"].isin(sel_routes)]
    if sel_regimes and "REGIME" in filtered.columns:
        filtered = filtered[filtered["REGIME"].isin(sel_regimes)]

    # 추가 필터 (Squeeze, SuperTrend, OBV, HMA)
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        show_only_squeeze = st.checkbox("🌪️ TTM Squeeze (폭발 대기)", key="chk_sqz_only")
        show_obv_only = st.checkbox("💰 OBV 매집 (다이버전스)", key="chk_obv")
    with c_sub2:
        show_supertrend_bull = st.checkbox("📈 SuperTrend 상승 추세", key="chk_st_bull")
        show_hma_up = st.checkbox("🚀 HMA 추세 상승", key="chk_hma")

    if show_only_squeeze and "TTM_SQUEEZE" in filtered.columns:
        filtered = filtered[filtered["TTM_SQUEEZE"] == 1]
    if show_supertrend_bull and "SUPERTREND_DIR" in filtered.columns:
        filtered = filtered[filtered["SUPERTREND_DIR"] == 1]
    if show_obv_only and "OBV_Div" in filtered.columns:
        filtered = filtered[filtered["OBV_Div"] == "O"]
    if show_hma_up and "HMA_Trend" in filtered.columns:
        filtered = filtered[filtered["HMA_Trend"] == "▲"]

    # 🔥 [v10.0] 정렬 기준 변경 (TOTAL_SCORE 우선)
    sort_col = "TOTAL_SCORE" if "TOTAL_SCORE" in filtered.columns else "LDY_SCORE"
    filtered = filtered.sort_values(
        [sort_col, "거래대금(억원)"], 
        ascending=[False, False]
    )

    # 권한별 노출 개수 제한
    if auth_status in ["pro", "prime", "admin"]:
        limit = 20 if auth_status == "pro" else 100
        view_df = filtered.head(limit)
        desc = f"{auth_status.upper()} 회원: AI 종합 랭킹 Top {limit} 열람 중"
        st.success(f"🥇 {desc}")
    else:
        limit = 5 if user else 3
        view_df = filtered.head(limit)
        user_type = "Free" if user else "Guest"
        st.info(f"✅ {user_type} 회원: 상위 {limit}개 열람 중 (Pro/Prime 업그레이드 시 더 많은 종목 확인 가능)")

    if view_df.empty:
        st.warning("조건에 맞는 종목이 없습니다. 필터를 조정해 보세요.")
    else:
        # ---------------------------------------------------------
        # 🔥 [v10.0 New] AI 컨센서스 차트 (리스트 위에 배치)
        # ---------------------------------------------------------
        if "ML_SCORE" in view_df.columns and view_df["ML_SCORE"].sum() > 0:
            with st.expander("🧠 AI Insight Matrix (터치하여 차트 열기)", expanded=True):
                ai_fig = plot_ai_consensus(view_df)
                if ai_fig:
                    st.plotly_chart(ai_fig, use_container_width=True)
                    st.caption("💡 **우상단 빨간 박스** 영역은 **AI와 퀀트 로직이 동시에 강력 매수**를 가리키는 종목입니다.")

        # ---------------------------
        # 상세 종목 분석 (SelectBox)
        # ---------------------------
        opts = view_df.apply(
            lambda r: f"{r.get('종목명','-')} ({r.get('종목코드','-')}) / {r.get('REGIME','-')}",
            axis=1
        ).tolist()
        sel = st.selectbox("종목 선택 (상세 분석)", opts)

        if sel:
            sel_idx = opts.index(sel)
            row = view_df.iloc[sel_idx]
            code = str(row.get("종목코드", "")).zfill(6)

            c1, c2 = st.columns([2, 1])

            with c1:
                with st.expander("⚙️ 차트 보조지표 설정", expanded=False):
                    c_opt1, c_opt2, c_opt3, c_opt4, c_opt5, c_opt6 = st.columns(6)
                    with c_opt1: show_bb = st.checkbox("볼린저", value=True, key=f"opt_bb_{code}")
                    with c_opt2: show_kc = st.checkbox("켈트너", value=True, key=f"opt_kc_{code}")
                    with c_opt3: show_rsi = st.checkbox("RSI", value=False, key=f"opt_rsi_{code}")
                    with c_opt4: show_vwap = st.checkbox("VWAP", value=True, key=f"opt_vwap_{code}")
                    with c_opt5: show_hma = st.checkbox("HMA", value=True, key=f"opt_hma_{code}")
                    with c_opt6: show_obv = st.checkbox("OBV", value=True, key=f"opt_obv_{code}")

                chart_df = get_stock_chart_data(code)
                if chart_df is None or getattr(chart_df, "empty", True):
                    st.info("차트 데이터 없음")
                else:
                    entry = pd.to_numeric(row.get("추천매수가", np.nan), errors="coerce")
                    stop  = pd.to_numeric(row.get("손절가", np.nan), errors="coerce")
                    t1    = pd.to_numeric(row.get("추천매도가1", np.nan), errors="coerce")
                    t2    = pd.to_numeric(row.get("추천매도가2", np.nan), errors="coerce")
                    vwap  = pd.to_numeric(row.get("VWAP", np.nan), errors="coerce")

                    if pd.isna(t2) and pd.notna(t1): t2 = float(t1) * 1.07

                    fig = plot_interactive_chart(
                        df=chart_df, code=str(code), name=row.get("종목명", "-"),
                        entry=entry, stop=stop, target1=t1, target2=t2, vwap=vwap,
                        show_bb=show_bb, show_kc=show_kc, show_rsi=show_rsi,
                        show_vwap=show_vwap, show_hma=show_hma, show_obv=show_obv
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                if auth_status in ["pro", "prime", "admin"]:
                    st.markdown(f"### {row.get('종목명','-')}")
                    
                    # 주봉 신호등
                    w_above = row.get("주봉20선_상회") == "O"
                    w_up = row.get("주봉추세") == "▲"
                    if w_above and w_up:
                        t_color, t_bg, t_msg = "#2E7D32", "#E8F5E9", "🟢 대세 상승 (Strong Bull)"
                    elif w_above:
                        t_color, t_bg, t_msg = "#EF6C00", "#FFF3E0", "🟡 추세 유지 (Watching)"
                    else:
                        t_color, t_bg, t_msg = "#C62828", "#FFEBEE", "🔴 대세 하락 (High Risk)"

                    st.markdown(f"""
                        <div style="background-color:{t_bg}; border-left: 5px solid {t_color}; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
                            <p style="margin:0; font-size:0.85em; color:#666;">주봉 대추세 확증</p>
                            <p style="margin:0; font-weight:bold; color:{t_color}; font-size:1.1em;">{t_msg}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # 레이더 차트
                    st.plotly_chart(plot_radar_chart(row), use_container_width=True)

                    # 뉴스 카드
                    news_score = pd.to_numeric(row.get("NEWS_SCORE", 0), errors="coerce")
                    news_reason = str(row.get("NEWS_REASON", "")).strip()
                    if news_reason and news_reason != "nan" and news_reason != "뉴스없음":
                        n_color = "#E8F5E9" if news_score >= 3 else ("#FFEBEE" if news_score <= -3 else "#F3E5F5")
                        n_border = "#43A047" if news_score >= 3 else ("#E53935" if news_score <= -3 else "#8E24AA")
                        n_icon = "🔥 호재" if news_score >= 3 else ("🚨 악재" if news_score <= -3 else "📢 이슈")
                        
                        st.markdown(f"""
                        <div style="background-color:{n_color}; border-left: 4px solid {n_border}; padding: 12px; border-radius: 4px; margin-bottom: 10px;">
                            <span style="color:{n_border}; font-weight:bold; font-size:0.9em;">{n_icon} (점수: {news_score})</span><br>
                            <span style="color:#333; font-size:0.95em;">{news_reason}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # AI 코멘트 & 뱃지
                    ai_cmt = row.get("AI_COMMENT", row.get("WHY", "-"))
                    badges = []
                    if row.get("IS_SWING_SUPPORT", False): badges.append("🛡️스마트지지")
                    if row.get("OBV_Div") == "O": badges.append("💰OBV매집")
                    if row.get("HMA_Trend") == "▲": badges.append("🚀HMA상승")
                    if row.get("HMA_On") == "O": badges.append("✅HMA지지")
                    patterns = str(row.get("캔들패턴", "")).strip()
                    if patterns and patterns != "nan": badges.append(f"🕯️{patterns}")

                    if badges:
                        st.markdown("".join([f"<span style='background-color:#E3F2FD; color:#1565C0; padding:4px 8px; border-radius:4px; margin-right:5px; font-size:0.85em; font-weight:600;'>{b}</span>" for b in badges]), unsafe_allow_html=True)
                        st.write("")
                    
                    st.info(f"💬 **AI:** {ai_cmt}")

                    # 자금 관리
                    rec_qty = pd.to_numeric(row.get("추천수량", 0), errors='coerce')
                    rec_amt = pd.to_numeric(row.get("추천금액(만원)", 0), errors='coerce')
                    if rec_qty > 0:
                        st.markdown(f"""
                            <div style="padding:10px; border-radius:5px; background-color:rgba(46, 125, 50, 0.1); border:1px solid #4caf50; margin-bottom:10px;">
                                <strong style="color:#2e7d32;">💰 자금 관리 (Risk 2%)</strong><br>
                                🎯 추천 비중: <b>{int(rec_qty)}주</b> (약 {rec_amt}만원)
                            </div>
                        """, unsafe_allow_html=True)

                    # 리스크/리워드 차트
                    rr_entry = _to_num(row.get("추천매수가", np.nan), np.nan)
                    rr_stop  = _to_num(row.get("손절가", np.nan), np.nan)
                    rr_t1    = _to_num(row.get("추천매도가1", np.nan), np.nan)
                    rr_t2    = _to_num(row.get("추천매도가2", np.nan), np.nan)
                    st.plotly_chart(plot_risk_reward_bar(rr_entry, rr_stop, rr_t1, rr_t2), use_container_width=True)

                else:
                    st.warning("🔒 상세 분석(레이더/자금관리/AI)은 **Pro 등급부터** 확인 가능합니다.")

                # 가격 메트릭
                c_a, c_b = st.columns(2)
                c_a.metric("진입가", _to_int_str(row.get("추천매수가", 0)))
                stop_label = "손절가 🛡️" if row.get("IS_SWING_SUPPORT", False) else "손절가"
                c_b.metric(stop_label, _to_int_str(row.get("손절가", 0)), delta="Stop", delta_color="inverse")

        st.divider()
        st.subheader("📋 Daily Top List (AI Powered)", anchor=False)
        safe_view = view_df.copy().reset_index(drop=True)

        if not safe_view.empty:
            if "종목명" in safe_view.columns:
                safe_view.set_index("종목명", inplace=True)

            for c in ["종가", "추천매수가", "손절가", "추천매도가1", "거래대금(억원)"]:
                if c in safe_view.columns:
                    safe_view[c] = pd.to_numeric(safe_view[c], errors='coerce').fillna(0).apply(lambda x: f"{int(x):,}")

            # ✅ 표시할 컬럼 정의 (ML_SCORE, TOTAL_SCORE 추가)
            cols = [
                "REGIME", "ROUTE", 
                "TOTAL_SCORE", "ML_SCORE", "LDY_SCORE", 
                "TTM_SQUEEZE_CNT",
                "업종", "종목코드",
                "종가", "추천매수가", "손절가", "추천매도가1",
            ]
            display_cols = [c for c in cols if c in safe_view.columns]

            # 컬럼 설정 (Column Config)
            cfg = {
                "TOTAL_SCORE": st.column_config.ProgressColumn(
                    "🏆종합", format="%.1f", min_value=0, max_value=100, width="small"
                ),
                "ML_SCORE": st.column_config.ProgressColumn(
                    "🤖AI", format="%.1f", min_value=0, max_value=100, width="small"
                ),
                "LDY_SCORE": st.column_config.NumberColumn(
                    "퀀트", format="%.1f"
                ),
                "TTM_SQUEEZE_CNT": st.column_config.NumberColumn(
                    "🌪️응축", help="TTM Squeeze 연속 발생 일수", format="%d일", width="small"
                ),
                "종가": st.column_config.TextColumn("현재가", width="small"),
                "추천매수가": st.column_config.TextColumn("매수", width="small"),
                "손절가": st.column_config.TextColumn("손절", width="small"),
                "추천매도가1": st.column_config.TextColumn("목표", width="small"),
                "ROUTE": st.column_config.TextColumn("전략", width="small"),
            }

            st.dataframe(
                safe_view[display_cols], 
                use_container_width=True, 
                column_config=cfg, 
                height=600
            )

    if auth_status in ["prime", "admin"]:
        csv = scored.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 전체 다운로드", csv, "ldy_rank.csv", "text/csv")
        
# ---------------------------
# 내 자산 (병렬 처리)
# ---------------------------
def fetch_current_price(code, name):
    """
    현재가 조회 함수 (FDR 우선 시도 -> 실패 시 pykrx 시도)
    """
    price = 0

    # 1차 시도: FinanceDataReader (속도가 빠름)
    if FDR_OK:
        try:
            # 최근 7일 데이터 조회 (휴장일 고려)
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            df = fdr.DataReader(str(code).zfill(6), start_date)

            if df is not None and not df.empty:
                price = int(df.iloc[-1]['Close'])
        except Exception:
            pass # FDR 실패 시 그냥 넘어감

    # 2차 시도: pykrx (FDR 실패 시 백업)
    if price == 0 and PYKRX_OK:
        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=7)

            df_k = stock.get_market_ohlcv_by_date(
                start_dt.strftime("%Y%m%d"), 
                end_dt.strftime("%Y%m%d"), 
                str(code).zfill(6)
            )

            if df_k is not None and not df_k.empty:
                if '종가' in df_k.columns:
                    price = int(df_k.iloc[-1]['종가'])
                elif 'Close' in df_k.columns:
                    price = int(df_k.iloc[-1]['Close'])
        except Exception:
            pass

    return code, name, price

with tab3:
    # 1) 권한 체크
    if auth_status in ["guest", "free"]:
        st.info("🔒 내 자산 분석은 Pro 등급부터 가능합니다.")
    else:
        st.subheader("💼 내 자산 관리 (엑셀형 에디터)")

        # 1. 데이터 로드 (Gist에서 가져오기)
        saved_str = load_portfolio_file()
        default_data = []
        
        # 저장된 데이터 파싱 (기존 텍스트 포맷 -> 리스트 변환)
        if saved_str:
            try:
                lines = saved_str.strip().split("\n")
                for line in lines:
                    if ":" in line:
                        parts = line.split(":")
                        if len(parts) >= 3:
                            # 콤마 제거 및 숫자 변환
                            try:
                                p_val = int(float(parts[1].replace(",","").strip()))
                                q_val = int(float(parts[2].replace(",","").strip()))
                                default_data.append({
                                    "종목명": parts[0].strip(),
                                    "평단가": p_val,
                                    "수량": q_val,
                                    "비고": ""
                                })
                            except: pass
            except: pass
        
        # 데이터가 없으면 빈 행 추가 (사용자 입력 유도용)
        if not default_data:
            default_data = [{"종목명": "", "평단가": 0, "수량": 0, "비고": ""}]

        # 2. 데이터 에디터 출력 (엑셀처럼 편집 가능)
        edited_df = st.data_editor(
            pd.DataFrame(default_data),
            num_rows="dynamic",  # 행 추가/삭제 허용
            use_container_width=True,
            key="portfolio_editor",
            column_config={
                "종목명": st.column_config.TextColumn(required=True),
                "평단가": st.column_config.NumberColumn(format="%d원", min_value=0, required=True),
                "수량": st.column_config.NumberColumn(format="%d주", min_value=0, required=True),
                "비고": st.column_config.TextColumn(width="small")
            }
        )

        # 3. 데이터 저장 및 분석 대상 생성
        targets = []
        cash_amt = 0.0
        save_lines = []
        
        # 코드 매핑 함수가 있으면 가져오기 (전역 함수 사용)
        code_map = get_code_map() if 'get_code_map' in globals() else {}

        # 에디터의 내용을 순회하며 저장용 문자열과 분석용 타겟 리스트 생성
        if edited_df is not None and not edited_df.empty:
            for _, row in edited_df.iterrows():
                nm = str(row.get("종목명", "")).strip()
                if not nm: continue
                
                try:
                    price = float(row.get("평단가", 0))
                    qty = int(row.get("수량", 0))
                except: continue

                # 저장 포맷 (종목:평단:수량) 생성
                save_lines.append(f"{nm}:{int(price)}:{int(qty)}")

                # 현금(CASH)인지 일반 종목인지 구분
                if nm.upper() == "CASH" or "현금" in nm:
                    cash_amt += price * qty
                else:
                    # 종목명을 코드로 변환 (매핑 실패 시 입력한 이름 그대로 사용)
                    # 전역 함수 find_code_by_name 사용
                    real_code = find_code_by_name(nm, code_map) or nm
                    targets.append((real_code, nm, price, qty))
            
            # 변경 사항이 있으면 자동 저장 (Gist/파일)
            new_save_str = "\n".join(save_lines)
            if new_save_str != saved_str:
                save_portfolio_file(new_save_str)

        # 4. 실시간 시세 조회 및 카드 출력
        if not targets and cash_amt <= 0:
             st.info("👆 위 표에 보유 종목명, 평단가, 수량을 입력하면 실시간으로 분석됩니다.")
        else:
            # 병렬 처리로 현재가 조회
            price_map = {}
            with st.spinner('⚡ 보유 종목 시세 조회 중...'):
                with ThreadPoolExecutor(max_workers=10) as executor:
                    # 전역 함수 fetch_current_price 사용
                    futures = [executor.submit(fetch_current_price, t[0], t[1]) for t in targets]
                    for future in futures:
                        c, n, p = future.result()
                        price_map[c] = p
            
            # 종목별 카드 출력
            cols = st.columns(3)
            total_eval = 0.0
            total_buy = 0.0
            rows_pf = []

            for idx, (code, name, avg, qty) in enumerate(targets):
                curr = price_map.get(code, 0)
                
                # 시세가 있고 코드가 숫자면 실제 종목명(KRX) 가져오기, 아니면 입력한 이름 사용
                real_name = name
                if PYKRX_OK and curr > 0 and str(code).isdigit():
                    try:
                        kn = stock.get_market_ticker_name(code)
                        if kn: real_name = kn
                    except: pass

                eval_amt = curr * qty
                buy_amt = avg * qty
                total_eval += eval_amt
                total_buy += buy_amt
                
                rows_pf.append({"name": real_name, "eval": eval_amt, "code": code})
                
                # 수익률 계산
                pct = (curr - avg) / avg * 100 if avg > 0 and curr > 0 else 0
                pnl = eval_amt - buy_amt
                
                color = "green" if pct > 0 else ("red" if pct < 0 else "gray")
                signal = "🟢" if pct > 0 else ("🔴" if pct < 0 else "⚪")

                # 카드 UI (3열 배치)
                with cols[idx % 3]:
                    with st.container(border=True):
                        c_main, c_pnl = st.columns([1.5, 1])
                        c_main.markdown(f"**{real_name}** {signal}")
                        c_main.caption(f"평단 {int(avg):,} / {qty}주")
                        c_pnl.markdown(f":{color}[{pct:+.2f}%]")
                        c_pnl.markdown(f"**{int(curr):,}원**" if curr > 0 else "확인불가")

            # 5. 전체 요약 (현금 포함)
            total_asset = total_eval + cash_amt
            total_invest = total_buy + cash_amt
            total_rate = (total_asset - total_invest) / total_invest * 100 if total_invest > 0 else 0
            
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("총 매수금(현금포함)", f"{int(total_invest):,}원")
            m2.metric("총 평가금(현금포함)", f"{int(total_asset):,}원")
            m3.metric("총 수익률", f"{total_rate:+.2f}%", 
                      delta=f"{int(total_asset - total_invest):,}원", 
                      delta_color="normal" if total_rate >= 0 else "inverse")

            # 6. 파이 차트 (자산 구성)
            if cash_amt > 0:
                rows_pf.append({"name": "💰 현금 (CASH)", "eval": cash_amt, "code": "CASH"})

            if total_asset > 0:
                df_chart = pd.DataFrame(rows_pf)
                if not df_chart.empty:
                    fig = px.pie(df_chart, values="eval", names="name", title="📊 자산 비중", hole=0.4)
                    fig.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("📮 문의 게시판")

    current_user = user if 'user' in globals() else None

    default_email = ""
    default_nick = ""
    if current_user:
        default_email = current_user.get("login_id", "")
        default_nick = current_user.get("nickname", "")

    st.markdown("#### ✏️ 문의 작성")

    with st.form("inquiry_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            nickname = st.text_input("닉네임", value=default_nick, placeholder="닉네임 또는 이름")
        with col_b:
            email = st.text_input("이메일 (선택)", value=default_email, placeholder="답변 받을 이메일 (선택)")

        title = st.text_input("제목", placeholder="문의 제목을 입력해 주세요.")
        content = st.text_area("내용", placeholder="사이트 사용 관련 문의를 자유롭게 남겨 주세요.", height=150)

        submitted = st.form_submit_button("💌 문의 등록")

    if submitted:
        if not title.strip() or not content.strip():
            st.error("제목과 내용을 모두 입력해 주세요.")
        else:
            # ✅ [수정됨] Gist에서 기존 목록을 불러옵니다.
            current_items = load_inquiry_items()

            # 새 문의 데이터 생성
            new_item = {
                "title": title.strip(),
                "content": content.strip(),
                "nickname": nickname.strip() or "익명",
                "email": email.strip(),
                "created_at": _now_utc_str(), # auth_user의 시간 함수 사용
            }

            # 리스트에 추가하고 Gist에 저장
            current_items.append(new_item)
            ok = save_inquiry_items(current_items)

            if ok:
                st.success("문의가 등록되었습니다. Gist에 저장 완료! 🙌")
                # 화면 갱신을 위해 rerun (Streamlit 버전에 따라 다름)
                try:
                    st.rerun()
                except:
                    pass
            else:
                st.error("저장 실패! (Gist 연동 오류 - 로그 확인 필요)")

    st.markdown("#### 📂 최근 문의 내역")

    # ✅ [수정됨] Gist에서 데이터를 불러와서 보여줍니다.
    inquiries = load_inquiry_items()

    if not inquiries:
        st.info("아직 등록된 문의가 없습니다.")
    else:
        # 최신순 정렬 (리스트 뒤집기)
        for item in reversed(inquiries[-50:]):
            box = st.container(border=True)
            with box:
                st.markdown(f"**제목:** {item.get('title', '-')}")

                # 날짜 포맷팅 (UTC -> KST 변환은 to_kst_str 함수가 있다면 사용, 없으면 그대로)
                date_str = item.get('created_at','-')
                if 'to_kst_str' in globals():
                    date_str = to_kst_str(date_str)

                meta = f"작성자: {item.get('nickname','익명')} · 작성일: {date_str}"
                if item.get("email"):
                    meta += f" · 이메일: {item.get('email')}"
                st.caption(meta)
                st.markdown(item.get("content", "").replace("\n", "  \n"))

with tab5:
    st.subheader("⚖️ 이용 약관 / 투자 유의사항")

    st.markdown("### 1. 서비스 성격")
    st.markdown(
        "- LDY Pro Trader는 **퀀트 지표 기반의 데이터 분석 도구**로, "
        "개별 종목의 매수·매도, 수익을 보장하는 리딩 서비스가 아닙니다.\n"
        "- 제공되는 모든 정보는 **교육 및 참고용**이며, "
        "투자 판단을 보조하는 **연구·리서치 자료**의 성격을 가집니다."
    )

    st.markdown("### 2. 투자 책임에 대한 안내")
    st.markdown(
        "- 실제 매수·매도 등 **최종 투자 의사결정**은 전적으로 이용자 본인의 판단입니다.\n"
        "- 투자 결과로 발생하는 **손익(수익, 손실, 기회비용 포함)**은 "
        "모두 이용자 본인에게 귀속되며, 본 서비스 및 개발자는 이에 대해 법적 책임을 지지 않습니다.\n"
        "- 본 서비스는 **미래 수익률, 특정 수익구간 달성, 손실 방지** 등을 어떠한 형태로도 보증하지 않습니다."
    )

    st.markdown("### 3. 데이터 및 지표 한계")
    st.markdown(
        "- 사용되는 시장 데이터는 외부 데이터 제공처 및 증권사 API, 공개 데이터 소스를 바탕으로 하며, "
        "지연·오류·누락이 발생할 수 있습니다.\n"
        "- 지표 및 스코어는 과거 데이터를 기반으로 계산되며, "
        "**향후 시장 상황과 괴리**가 발생할 수 있습니다.\n"
        "- 알고리즘 로직은 지속적으로 개선/업데이트될 수 있으며, "
        "이 과정에서 **종전 결과와 다른 스코어**가 나올 수 있습니다."
    )

    st.markdown("### 4. 이용권 및 계정 정책 (요약)")
    st.markdown(
        "- **Guest(비회원)** : 상위 3개 종목 맛보기.\n"
        "- **Free(회원)** : 상위 5개 종목 열람.\n"
        f"- **Pro 1개월 이용권 ({PRICE_PRO:,}원)** : 상위 20 종목, 내 자산 분석 기능 제공.\n"
        f"- **Prime 1개월 이용권 ({PRICE_PRIME:,}원)** : 전체 종목, CSV 다운로드, 텔레그램 알림 등 고급 기능 제공.\n"
        "- 자동 결제는 지원하지 않으며, 1개월 단위 선불 결제·연장 방식입니다.\n"
        "- 구체적인 결제/환불/이용 기간 정책은 별도 안내(카카오 채널, 약관 페이지 등)를 따릅니다."
    )

    st.markdown("### 5. 한 줄 요약")
    st.info("👉 **데이터와 퀀트는 도구일 뿐, 최종 책임은 언제나 본인에게 있다.**")

with tab6:
    st.subheader("🧩 LDY Pro Trader 업데이트 노트")

    if not CHANGELOG:
        st.info("아직 등록된 업데이트 기록이 없습니다.")
    else:
        latest = CHANGELOG[0]

        # 🔹 상단에 현재 버전 / 최근 업데이트 요약
        st.success(
            f"현재 버전: **v{APP_VERSION}**  \n"
            f"최근 업데이트: **{latest['date']} · {latest['title']}**"
        )

        st.markdown("---")

        # 🔹 버전별 상세 내역 (최신 버전은 기본 펼침)
        for idx, log in enumerate(CHANGELOG):
            header = f"v{log['version']} · {log['date']} — {log['title']}"
            is_latest = (idx == 0)

            with st.expander(
                f"⭐ {header}" if is_latest else header,
                expanded=is_latest,   # 최신 버전만 기본 펼침
            ):
                for item in log.get("items", []):
                    st.markdown(f"- {item}")
