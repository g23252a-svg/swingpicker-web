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

def postprocess_codes(df: pd.DataFrame) -> pd.DataFrame:
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].apply(normalize_code)
    return df


# load_inquiry_items, save_inquiry_items, _now_utc_str 를 추가했습니다.
from auth_user import (
    render_auth_box, get_user, list_users, update_user_role,
    load_inquiry_items, save_inquiry_items, _now_utc_str,
    load_subscriptions_db, save_subscriptions_db  # 👈 이 2개가 꼭 있어야 합니다!
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
        df = _via_fdr(ticker) or _via_pykrx_index(ticker)
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

        # SuperTrend
        df = calculate_supertrend(df)

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
    show_bb: bool = True,
    show_kc: bool = False,
    show_rsi: bool = False,
):
    if df is None or df.empty:
        return go.Figure()

    rows = 3 if show_rsi else 2
    row_heights = [0.6, 0.2, 0.2] if show_rsi else [0.7, 0.3]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
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
                line=dict(color='#FF4081', width=2, dash='dot'), # Dotted Red Line
                name='SuperTrend (Resist)'
            ), row=1, col=1)

    # 6) 거래량 (캔들 색상과 일치시키되 투명도 조절)
    if "Volume" in df.columns:
        colors = [
            COLOR_UP if c >= o else COLOR_DOWN 
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], 
            name="거래량", 
            marker_color=colors, 
            opacity=0.8,
            showlegend=False
        ), row=2, col=1)

    # 7) RSI (과매수/과매도 영역 강조)
    if show_rsi and "RSI14_CHART" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI14_CHART"], 
            name="RSI(14)", 
            line=dict(color='#AB47BC', width=1.5)
        ), row=3, col=1)

        # 기준선 30/70
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70,
                      line=dict(color="red", width=1, dash="dot"), row=3, col=1)
        fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30,
                      line=dict(color="blue", width=1, dash="dot"), row=3, col=1)

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

    if entry_v is not None:
        fig.add_hline(y=entry_v, line_dash="dash", line_color=COLOR_ENTRY, line_width=1.5, 
                      annotation_text=f"🚀진입: {int(entry_v):,}", annotation_font_color=COLOR_ENTRY, row=1, col=1)
    if stop_v is not None:
        fig.add_hline(y=stop_v, line_dash="dot", line_color=COLOR_LOSS, line_width=1.5,
                      annotation_text=f"🛡️손절: {int(stop_v):,}", annotation_font_color=COLOR_LOSS, row=1, col=1)
    if t1_v is not None:
        fig.add_hline(y=t1_v, line_dash="dot", line_color=COLOR_STOP, line_width=1.5,
                      annotation_text=f"💰목표: {int(t1_v):,}", annotation_font_color=COLOR_STOP, row=1, col=1)

    # 레이아웃 설정
    # 🟢 [붙여넣을 새 코드]
    # 레이아웃 설정 (모바일 최적화)
    fig.update_layout(
        title=dict(
            text=f"{name} ({str(code).zfill(6)})",
            font=dict(size=16),
            x=0,  # 제목은 왼쪽 유지
        ),
        xaxis_rangeslider_visible=False,
        height=700 if show_rsi else 550,
        # ✅ 상단 여백(t)을 50 -> 80으로 늘려 제목과 범례 사이 공간 확보
        margin=dict(l=10, r=10, t=80, b=10),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            # ✅ 범례를 오른쪽(x=1) 끝으로 정렬하여 제목(왼쪽)과 겹침 방지
            xanchor="right", x=1
        ),
        dragmode="pan",
    )

    # Y축 그리드 설정 (눈에 덜 띄게)
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
        st.subheader("👑 회원 권한 / 구독 관리 (Admin)")

        users = list_users()
        if not users:
            st.info("등록된 회원이 없습니다.")
        else:
            subs_db = load_subs_db()
            subs = subs_db.get("subs", {})

            rows = []
            today = now_kst().date()
            
            for u in users:
                email = u.get("login_id")
                sub = subs.get(email, {})
                
                # 1. 실제 권한 확인 (구독정보 없으면 기본 유저 권한 사용)
                current_role = sub.get("role") or u.get("role", "free")
                
                # 2. 만료일 문자열 가져오기
                exp_str = sub.get("expire_at", "")
                days_left = ""

                # 3. 권한별 표시 로직 분기
                if current_role in ["free", "guest"]:
                    exp_str = "-"
                    days_left = "-"
                elif current_role == "admin":
                    exp_str = "무제한"
                    days_left = "∞"
                else:
                    # Pro / Prime 인 경우
                    if exp_str:
                        try:
                            d_exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
                            remain = (d_exp - today).days
                            days_left = f"{remain}일"
                            
                            # (선택) 만료된 경우 표시
                            if remain < 0:
                                days_left = f"만료 ({remain}일)"
                        except Exception:
                            days_left = "날짜오류"
                    else:
                        # 권한은 있는데 날짜가 없는 경우 (DB 불일치 등)
                        exp_str = "미설정"
                        days_left = "?"

                rows.append({
                    "이메일": email,
                    "닉네임": u.get("nickname"),
                    "권한(auth_user)": u.get("role"),
                    "구독 역할(sub)": current_role,  # 수정됨
                    "만료일": exp_str,
                    "잔여일수": days_left,
                    "가입일": to_kst_str(u.get("created_at")),
                    "마지막 로그인": to_kst_str(u.get("last_login")),
                })

            df_users = pd.DataFrame(rows)
            st.dataframe(df_users, use_container_width=True, height=230)

            target_email = st.selectbox(
                "권한을 변경할 회원 선택",
                options=[u["이메일"] for u in rows],
                key="admin_target_user",
            )
            new_role = st.selectbox(
                "새 권한",
                options=["free", "pro", "prime", "admin"],
                index=1,
                key="admin_new_role",
            )

            if st.button("권한 변경 적용", key="btn_update_role"):
                ok = update_user_role(target_email, new_role)
                if ok:
                    set_subscription(target_email, new_role)
                    msg = f"{target_email} → {new_role} 으로 변경되었습니다."
                    if new_role in ["pro", "prime"]:
                        sub_info = get_subscription(target_email)
                        if sub_info:
                            msg += f" (만료일: {sub_info.get('expire_at')})"
                    st.success(msg + " (새로고침 후 반영)")
                else:
                    st.error("권한 변경에 실패했습니다.")

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
    st.subheader("🎯 추천 종목 필터")

    if just_registered:
        st.success("🎉 첫 가입을 환영합니다! 오늘 기준 TOP 5 프리뷰를 먼저 보여드릴게요.")
        try:
            preview = make_preview(base, n=5)
        except Exception:
            preview = make_preview(scored, n=5)

        if not preview.empty:
            cols = ["종목명", "종목코드", "LDY_SCORE", "추천매수가", "손절가", "추천매도가1"]
            cols = [c for c in cols if c in preview.columns]
            if "종목명" in cols:
                prev_view = preview[cols].set_index("종목명")
            else:
                prev_view = preview[cols]
            st.dataframe(prev_view, use_container_width=True)
        else:
            st.info("프리뷰로 표시할 종목이 없습니다.")
        st.session_state["just_registered"] = False
        st.divider()

    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        min_score = st.slider(
            "최소 LDY 점수",
            min_value=0, max_value=100, value=80, step=1,
            key="min_score",
        )

    # ROUTE 필터 (기존 유지)
    with col_f2:
        def _route_order(r: str):
            s = str(r)
            if "SQZ" in s:
                return (0, s)
            if "BRK" in s:
                return (1, s)
            if "Watch" in s or "관찰" in s or "상승" in s:
                return (2, s)
            if "MR" in s:
                return (3, s)
            if "PULL" in s:
                return (4, s)
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

    # 🔥 REGIME(추세) 필터 추가
    with col_f3:
        if "REGIME" in scored.columns:
            all_regimes = sorted(scored["REGIME"].dropna().unique().tolist())
            # 기본값: 전체 선택 (필요하면 ①~③만 기본 선택으로 바꿀 수도 있음)
            sel_regimes = st.multiselect(
                "추세 구분 (REGIME)",
                options=all_regimes,
                default=all_regimes,
                key="regime_filter",
            )
        else:
            sel_regimes = []

    # EBS / 유동성 통과여부 체크박스
    use_only_gate = st.checkbox(
        "EBS/유동성 통과만 사용",
        value=True,
        key="only_gate",
    )

    # 🔥 권한별로 Daily Top 모수 설정
    if use_only_gate:
        # ✅ EBS / 유동성 통과 종목만 보는 모드
        if auth_status in ["prime", "admin"]:
            # PRIME 이상: 통과 종목을 넉넉히 가져와서 그 중에서 Top 100 뽑기
            base_view = base.head(300).copy()   # 필요하면 .head(300) → base.copy()로 바꿔도 됨
        else:
            # guest / free / pro: 기존처럼 Top20만 모수
            base_view = top20.copy()
    else:
        # ✅ 전체 스코어링 종목에서 보는 모드
        if auth_status in ["prime", "admin"]:
            # PRIME 이상: 전체 스코어링 대상
            base_view = scored.copy()
        else:
            # guest / free / pro: 상위 50개까지만 모수
            base_view = scored.head(50).copy()

    filtered = base_view.copy()
    filtered = filtered[filtered["LDY_SCORE"] >= min_score]

    if sel_routes:
        filtered = filtered[filtered["ROUTE"].isin(sel_routes)]

    # 🔥 REGIME 필터 반영
    if sel_regimes and "REGIME" in filtered.columns:
        filtered = filtered[filtered["REGIME"].isin(sel_regimes)]

    # 👇👇👇 [여기] 아래 코드를 삽입하세요 👇👇👇

    # 🔥 [v7.0] 스퀴즈 종목만 보기 필터
    # (체크박스를 누르면 TTM_SQUEEZE == 1 인 종목만 남김)
    show_only_squeeze = st.checkbox("🌪️ TTM Squeeze (폭발 대기) 종목만 보기", key="chk_sqz_only")

    if show_only_squeeze and "TTM_SQUEEZE" in filtered.columns:
        filtered = filtered[filtered["TTM_SQUEEZE"] == 1]

    # 👇 [여기 삽입] 🔥 [v8.0] SuperTrend 상승 추세 필터 추가
    show_supertrend_bull = st.checkbox("📈 SuperTrend 상승 추세만 보기", key="chk_st_bull")

    if show_supertrend_bull and "SUPERTREND_DIR" in filtered.columns:
        # 1: 상승, -1: 하락 (Collector v8.0 기준)
        filtered = filtered[filtered["SUPERTREND_DIR"] == 1]


    if auth_status in ["pro", "prime", "admin"]:
        if auth_status == "pro":
            # Pro: 기존처럼 Top 20
            view_df = filtered.head(20)
            desc = "Pro 회원: 필터 적용 Top 20 종목 열람 중"
        else:
            # Prime 이상(Prime, Admin): Top 100
            view_df = filtered.head(100)
            desc = f"{auth_status.upper()} 회원: 필터 적용 Top 100 종목 열람 중"

        st.success(f"🥇 {desc}")
    else:
        if user is None:
            view_df = filtered.head(3)
            st.info(
                "🔐 현재는 **비로그인(게스트)** 상태라, 필터 적용 상위 **3개 종목**만 확인할 수 있습니다.\n\n"
                "✅ 지금 무료 회원가입하면 **상위 5개 종목**까지 바로 열람 가능합니다!"
            )
        else:
            view_df = filtered.head(5)
            st.info(
                "✅ Free 회원: 필터 적용 상위 **5개 종목**까지 열람 중입니다.\n"
                "📈 더 많은 종목과 CSV 다운로드, 알림 기능은 Pro / Prime 등급에서 제공됩니다."
            )
    if view_df.empty:
        st.warning("조건에 맞는 종목이 없습니다. 필터를 조정해 보세요.")
    else:
        opts = view_df.apply(
            lambda r: f"{r.get('종목명','-')} ({r.get('종목코드','-')}) / {r.get('REGIME','-')}",
            axis=1
        ).tolist()
        sel = st.selectbox("종목 선택", opts)
        # [dashboard.py] tab2 내부의 if sel: 블록 전체를 이것으로 교체하세요.
        if sel:
            sel_idx = opts.index(sel)
            row = view_df.iloc[sel_idx]
            code = str(row.get("종목코드", "")).zfill(6)

            # --- 여기서부터 들여쓰기(Indent) 주의 ---
            c1, c2 = st.columns([2, 1])


            with c1:  # 👈 여기가 c1, c2 정의와 같은 레벨이거나 안쪽이어야 함
                # 🔧 [수정됨] 모바일 최적화: Expander로 옵션 숨기기
                with st.expander("⚙️ 차트 보조지표 설정 (터치하여 열기)", expanded=False):
                    c_opt1, c_opt2, c_opt3 = st.columns(3)
                    with c_opt1:
                        show_bb = st.checkbox("볼린저 밴드", value=True, key=f"opt_bb_{code}")
                    with c_opt2:
                        show_kc = st.checkbox("켈트너 채널", value=True, key=f"opt_kc_{code}")
                    with c_opt3:
                        show_rsi = st.checkbox("RSI 표시", value=False, key=f"opt_rsi_{code}")

                chart_df = get_stock_chart_data(code)

                # 차트 데이터 유효성 체크
                if chart_df is None or getattr(chart_df, "empty", True):
                    st.info("차트 데이터 없음")
                else:
                    # 숫자형 안전 변환
                    entry = pd.to_numeric(row.get("추천매수가", np.nan), errors="coerce")
                    stop  = pd.to_numeric(row.get("손절가", np.nan), errors="coerce")
                    t1    = pd.to_numeric(row.get("추천매도가1", np.nan), errors="coerce")
                    t2    = pd.to_numeric(row.get("추천매도가2", np.nan), errors="coerce")

                    # 목표가2 없으면 자동 계산
                    if pd.isna(t2) and pd.notna(t1):
                        t2 = float(t1) * 1.07

                    # 차트 그리기 함수 호출 (파라미터 전달)
                    fig = plot_interactive_chart(
                        df=chart_df,
                        code=str(code),
                        name=row.get("종목명", "-"),
                        entry=entry,
                        stop=stop,
                        target1=t1,
                        target2=t2,
                        show_bb=show_bb,
                        show_kc=show_kc,  # 🔥 켈트너 채널 옵션 전달
                        show_rsi=show_rsi,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                if auth_status in ["pro", "prime", "admin"]:
                    st.markdown(f"### {row.get('종목명','-')}")
                    
                    # 📊 레이더 차트
                    st.plotly_chart(plot_radar_chart(row), use_container_width=True)

                    # 📝 AI 코멘트 & 스마트 손절 뱃지
                    ai_cmt = row.get("AI_COMMENT", row.get("WHY", "-"))
                    
                    # ✅ [v7.5] 스마트 손절 감지 표시
                    is_swing = row.get("IS_SWING_SUPPORT", False)
                    if is_swing:
                        st.success("🛡️ **스마트 지지선(Swing Low)** 감지됨! (손절가 보정)")
                    
                    st.info(f"💬 **AI:** {ai_cmt}")

                    # 리스크/리워드 바 차트
                    rr_entry = _to_num(row.get("추천매수가", np.nan), np.nan)
                    rr_stop  = _to_num(row.get("손절가", np.nan), np.nan)
                    rr_t1    = _to_num(row.get("추천매도가1", np.nan), np.nan)
                    rr_t2    = _to_num(row.get("추천매도가2", np.nan), np.nan)

                    st.plotly_chart(
                        plot_risk_reward_bar(rr_entry, rr_stop, rr_t1, rr_t2),
                        use_container_width=True,
                    )
                else:
                    st.warning("🔒 상세 분석(레이더 / 리스크-리워드 / AI 코멘트)은 **Pro 등급부터** 확인 가능합니다.")

                # 하단 가격 메트릭
                c_a, c_b = st.columns(2)
                c_a.metric("진입가", _to_int_str(row.get("추천매수가", 0)))
                
                # 손절가 표시에 뱃지 추가 가능 여부 확인
                stop_label = "손절가"
                if row.get("IS_SWING_SUPPORT", False):
                    stop_label += " 🛡️"
                
                c_b.metric(
                    stop_label,
                    _to_int_str(row.get("손절가", 0)),
                    delta="Stop",
                    delta_color="inverse",
                )

    st.divider()
    st.subheader("📋 Daily Top List (Squeeze Analysis)", anchor=False)
    safe_view = view_df.copy().reset_index(drop=True)

    if not safe_view.empty:
        if "종목명" in safe_view.columns:
            safe_view.set_index("종목명", inplace=True)

        # 숫자 포맷팅
        price_cols = ["종가", "추천매수가", "손절가", "추천매도가1", "거래대금(억원)"]
        for c in price_cols:
            if c in safe_view.columns:
                safe_view[c] = pd.to_numeric(safe_view[c], errors='coerce').fillna(0).apply(lambda x: f"{int(x):,}")

        if "LDY_SCORE" in safe_view.columns:
            safe_view["LDY_SCORE"] = pd.to_numeric(safe_view["LDY_SCORE"], errors='coerce').fillna(0)

        # ✅ 표시할 컬럼 정의 (SQUEEZE_CNT 추가)
        # ✅ 표시할 컬럼 정의 (SQUEEZE_CNT 추가)
        cols = [
            "REGIME", "ROUTE", 
            "TTM_SQUEEZE_CNT", # 🔥 [New] 스퀴즈 지속일
            "업종", "종목코드", "LDY_SCORE",
            "종가", "추천매수가", "손절가", "추천매도가1",
            "종목명"
        ]
        # 실제 존재하는 컬럼만 필터링
        cols = [c for c in cols if c in safe_view.columns]

        # 컬럼 설정 (Column Config)
        # 🚨 Streamlit의 column_config를 활용해 "작은 화면"용 라벨 설정
        cfg = {
            "LDY_SCORE": st.column_config.ProgressColumn(
                "점수", format="%.1f", min_value=0, max_value=100
            ),
            "TTM_SQUEEZE_CNT": st.column_config.NumberColumn(
                "🌪️응축(일)", help="TTM Squeeze 연속 발생 일수. 5일 이상이면 폭발 임박(Hot Zone)"
                "점수", format="%.0f", min_value=0, max_value=100, width="small" 
            ),
            "종가": st.column_config.TextColumn("현재가"),
            "추천매수가": st.column_config.TextColumn("진입가"),
            "손절가": st.column_config.TextColumn("손절가"),
            "추천매도가1": st.column_config.TextColumn("목표가"),
            "ROUTE": st.column_config.TextColumn("전략", width="small"),
            "종가": st.column_config.TextColumn("현재가", width="small"),
            "추천매수가": st.column_config.TextColumn("매수", width="small"),
            "손절가": st.column_config.TextColumn("손절", width="small"),
            "TTM_SQUEEZE_CNT": st.column_config.NumberColumn("응축", width="small"),
        }

        # 실제 존재하는 컬럼만 필터링 (인덱스인 종목명은 cols에서 제외)
        display_cols = [c for c in cols if c in safe_view.columns or c == "종목명"]
        if "종목명" in safe_view.index.names:
             display_cols = [c for c in display_cols if c != "종목명"]

        st.dataframe(
            safe_view[display_cols],     # ✅ 첫 번째 위치 인자 (데이터)
            use_container_width=True,    # ✅ 키워드 인자
            column_config=cfg,           # ✅ 키워드 인자
            height=500                   # ✅ 키워드 인자
        )
    else:
        st.info("표시할 종목 없음")

    if auth_status in ["prime", "admin"]:
        csv = scored.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 전체 다운로드",
            csv,
            "ldy_rank.csv",
            "text/csv",
        )

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

    # 2차 시도: pykrx (FDR 실패 시 백업, collector.py와 동일 로직)
    # price가 0이고 pykrx 라이브러리가 로드되어 있다면 시도
    if price == 0 and PYKRX_OK:
        try:
            # pykrx는 YYYYMMDD 형식을 씀
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=7)

            # 오늘 날짜까지 조회
            df_k = stock.get_market_ohlcv_by_date(
                start_dt.strftime("%Y%m%d"), 
                end_dt.strftime("%Y%m%d"), 
                str(code).zfill(6)
            )

            if df_k is not None and not df_k.empty:
                # '종가' 컬럼이 있는지 확인 (pykrx 버전에 따라 컬럼명이 다를 수 있음)
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
    elif pf_input:
        try:
            # 2) 종목 코드 매핑 준비
            code_map = get_code_map() if 'get_code_map' in globals() else {}
            if not code_map and FDR_OK:
                try:
                    df_krx = fdr.StockListing('KRX')
                    code_map = dict(
                        zip(df_krx['Name'], df_krx['Code'].astype(str).str.zfill(6))
                    )
                except Exception:
                    pass

            # 3) 포트폴리오 파싱
            targets = []     # 종목 리스트
            cash_amt = 0.0   # 현금(예수금) 총액

            lines = pf_input.strip().split('\n')
            for line in lines:
                if ":" not in line:
                    continue

                parts = [p.strip() for p in line.split(':')]
                if len(parts) != 3:
                    continue

                name_input, avg_str, qty_str = parts

                # ✅ CASH / 현금 라인 분리 처리
                if name_input.strip().upper().startswith("CASH") or "현금" in name_input:
                    try:
                        cash_amt += float(avg_str.replace(',', '')) * int(qty_str.replace(',', ''))
                    except Exception:
                        pass
                    continue

                # ✅ 일반 종목 라인
                code = str(name_input).strip()
                if not code.isdigit():
                    code = code_map.get(code, code)  # 종목명 → 코드 매핑
                code = str(code).zfill(6)

                try:
                    avg = float(avg_str.replace(',', ''))
                    qty = int(qty_str.replace(',', ''))
                except Exception:
                    continue

                targets.append((code, name_input, avg, qty))

            if not targets and cash_amt <= 0:
                st.warning("포트폴리오 입력 형식이 올바른지 확인해 주세요. 예시: `NAVER:261000:10` 또는 `CASH:1000000:1`")
                st.stop()

            # 4) 현재가 조회 (병렬 처리)
            price_map = {}
            with st.spinner('⚡ 실시간 시세를 조회 중입니다...'):
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [
                        executor.submit(fetch_current_price, t[0], t[1])
                        for t in targets
                    ]
                    for future in futures:
                        c, n, p = future.result()
                        price_map[c] = p

            # 5) 기본 지표 계산 (종목별/전체)
            cols_layout = st.columns(3)
            
            total_buy = 0.0
            total_eval = 0.0

            rows_pf = []  # 섹터/파이차트용

            for idx, (code, name_input, avg, qty) in enumerate(targets):
                cur_price = price_map.get(code, 0)
                real_name = (
                    stock.get_market_ticker_name(code)
                    if (PYKRX_OK and cur_price > 0)
                    else name_input
                )

                buy_amt = avg * qty
                eval_amt = cur_price * qty
                total_buy += buy_amt
                total_eval += eval_amt

                rows_pf.append(
                    {
                        "code": code,
                        "name": real_name,
                        "avg": avg,
                        "qty": qty,
                        "eval": eval_amt,
                    }
                )

                # 수익률/평가손익
                # 수익률/평가손익 계산
                if cur_price > 0:
                    profit_rate = (cur_price - avg) / avg * 100
                    pnl = eval_amt - buy_amt
                    if profit_rate > 0:
                        signal = "🟢 수익"
                    elif profit_rate > -3:
                        signal = "🟡 보합"
                    else:
                        signal = "🔴 손실"
                else:
                    signal = "❓ 확인불가"
                    profit_rate = 0
                    pnl = 0

                with cols_layout[idx % 3]:
                    st.metric(
                        label=f"{real_name} ({signal})",
                        value=f"{cur_price:,}원" if cur_price > 0 else "시세 없음",
                        delta=f"{profit_rate:+.2f}% ({int(pnl):,}원)",
                        delta_color="normal" if profit_rate >= 0 else "inverse",
                    )
                # 🚨 [수정됨] 모바일 최적화: 카드형 UI (컨테이너 사용)
                with st.container(border=True):
                    c_main, c_pnl = st.columns([1.5, 1])
                    with c_main:
                        st.markdown(f"**{real_name}**")
                        st.caption(f"평단: {int(avg):,} / 수량: {qty}")
                        if cur_price > 0:
                            st.markdown(f"현재: **{cur_price:,}원**")
                        else:
                            st.markdown("시세 확인 불가")
                    
                    with c_pnl:
                        # 수익률 색상 강조
                        color = "green" if profit_rate > 0 else "red"
                        if profit_rate == 0: color = "gray"
                        
                        # 우측에 수익률 크게 표시
                        st.markdown(f":{color}[**{profit_rate:+.2f}%**]")
                        st.markdown(f":{color}[{int(pnl):,}원]")

            st.divider()

            # 6) 전체 포트폴리오 요약
            c1, c2, c3 = st.columns(3)
            tot_rate = (
                (total_eval - total_buy) / total_buy * 100
                if total_buy > 0
                else 0
            )

            c1.metric("총 매수", f"{int(total_buy):,}원")
            c2.metric("총 평가", f"{int(total_eval):,}원")
            c3.metric(
                "총 수익률",
                f"{tot_rate:+.2f}%",
                f"{int(total_eval - total_buy):,}원",
                delta_color="normal" if tot_rate >= 0 else "inverse",
            )

            # 7) 현금 비중 계산
            total_asset = total_eval + cash_amt
            if cash_amt > 0 and total_asset > 0:
                cash_ratio = cash_amt / total_asset * 100
                st.info(
                    f"💰 현재 현금(예수금) 비중은 **{cash_ratio:.1f}%** 입니다.\n"
                    "시장 변동성에 따라 보통 **10~30%** 사이에서 조절하는 전략이 많이 사용됩니다."
                )

            # 8) 종목별 평가금액 비중 파이차트
            try:
                if total_eval > 0:
                    df_pf = pd.DataFrame(rows_pf)
                    df_pf = df_pf[df_pf["eval"] > 0]

                    if not df_pf.empty:
                        fig_pie = go.Figure(
                            data=[
                                go.Pie(
                                    labels=df_pf["name"],
                                    values=df_pf["eval"],
                                    hole=0.4,
                                )
                            ]
                        )
                        fig_pie.update_layout(
                            title="📊 종목별 평가 금액 비중",
                            height=300,
                            margin=dict(l=10, r=10, t=40, b=10),
                            showlegend=True,
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
            except Exception:
                logger.exception("portfolio pie chart failed")

            # 9) 포트폴리오 Health Check (섹터 편중 + 현금 비중)
            try:
                st.subheader("🏥 포트폴리오 건강검진", anchor=False)

                df_pf = pd.DataFrame(rows_pf)
                df_pf = df_pf[df_pf["eval"] > 0]

                # 섹터 컬럼 선택
                sector_col = None
                if "업종_대분류" in scored.columns:
                    sector_col = "업종_대분류"
                elif "업종" in scored.columns:
                    sector_col = "업종"

                if sector_col:
                    # 종목코드 → 섹터 매핑
                    sector_map = (
                        scored
                        .dropna(subset=[sector_col, "종목코드"])
                        .drop_duplicates("종목코드")
                        .set_index("종목코드")[sector_col]
                        .to_dict()
                    )

                    df_pf["섹터"] = df_pf["code"].map(sector_map).fillna("기타")

                    sector_grp = (
                        df_pf.groupby("섹터")["eval"]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    total_eval_safe = sector_grp.sum()

                    if total_eval_safe > 0:
                        # 섹터 비중 (%)
                        sector_ratio = (sector_grp / total_eval_safe * 100).round(1)

                        fig_sector = go.Figure(
                            data=[
                                go.Bar(
                                    x=sector_ratio.values,
                                    y=sector_ratio.index,
                                    orientation="h",
                                    text=[f"{v:.1f}%" for v in sector_ratio.values],
                                    textposition="auto",
                                )
                            ]
                        )
                        fig_sector.update_layout(
                            title="섹터별 비중 (평가금액 기준)",
                            height=320,
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        st.plotly_chart(fig_sector, use_container_width=True)

                        # 편중 진단 코멘트
                        top_sector = sector_ratio.index[0]
                        top_ratio = float(sector_ratio.iloc[0])

                        comment = f"현재 가장 큰 비중은 **{top_sector} ({top_ratio:.1f}%)** 입니다.  \n"

                        if top_ratio >= 60:
                            comment += "➡ 단일 섹터 비중이 60%를 넘어 **위험도가 상당히 높은 편**입니다. 분산을 강하게 추천합니다."
                        elif top_ratio >= 40:
                            comment += "➡ 특정 섹터 비중이 40% 이상으로 **다소 편중된 상태**입니다. 다른 섹터 편입을 고민해 볼 만합니다."
                        else:
                            comment += "➡ 섹터 비중이 비교적 고르게 분산되어 있습니다."

                        # 현금 비중까지 함께 코멘트
                        total_asset = total_eval + cash_amt
                        if cash_amt > 0 and total_asset > 0:
                            cash_ratio = cash_amt / total_asset * 100
                            comment += f"\n\n또한 현금(예수금) 비중은 약 **{cash_ratio:.1f}%** 입니다."

                            if cash_ratio < 5:
                                comment += " 변동성이 큰 장세에서는 다소 낮은 편입니다. 방어력을 조금 보강하는 것도 고려해 보세요."
                            elif cash_ratio > 40:
                                comment += " 상당히 보수적인 비중으로, 기회 포착 속도는 느려질 수 있지만 하락 방어에는 유리한 편입니다."

                        st.info(comment)
                    else:
                        st.caption("※ 섹터별 평가금액이 거의 없어 건강검진을 생략했습니다.")
                else:
                    st.caption("※ 스코어 데이터에 섹터 정보(업종, 업종_대분류)가 없어 건강검진을 생략했습니다.")
            except Exception:
                logger.exception("portfolio health check failed")
                st.caption("※ 포트폴리오 건강검진 중 오류가 발생했습니다.")
        except Exception as e:
            st.error(f"포트폴리오 분석 중 오류 발생: {e}")

    else:
        st.info("👈 사이드바에 포트폴리오를 입력하고 '저장/분석' 버튼을 누르세요.")

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
