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
import time
import uuid
import version_info
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import glob
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import re
from typing import Optional, Dict, Any, Tuple
from dart_analyzer import DartAnalyzer

# ─── 모듈 분리 import (v2.0) ───
from shared_utils import nz_num, wma, safe_float
from scoring_engine import build_global_score as _ssot_build_global_score  # [v20.6] SSOT
from shared_utils import calc_hma as calc_hma_series  # 기존 호출 호환
from chart_components import (
    plot_ai_gauge_chart, plot_fear_greed_gauge, plot_kelly_visual,
    plot_radar_chart, plot_score_waterfall,
    plot_sector_treemap, plot_sector_momentum_bar,
    plot_ai_consensus, plot_opportunity_map,
    add_volume_profile,
)




# 1. 앱 시작 부분 (상단)에 알림 추가
version_info.show_toast_notification()
# ---------------------------
# [v11.0] DB 연동 및 히스토리 차트 함수
# ---------------------------
import duckdb  # 👈 필수
from plotly.subplots import make_subplots

@st.cache_data(ttl=60)  # DB 조회는 1분 캐싱
def get_stock_history_from_db(code: str):
    """
    DuckDB(ldy_trader.db)에서 특정 종목의 과거 추천 내역(점수, 주가)을 조회
    """
    db_path = "ldy_trader.db"
    if not os.path.exists(db_path):
        return pd.DataFrame()

    try:
        conn = duckdb.connect(db_path, read_only=True)
        code_str = str(code).zfill(6)
        
        # 🚨 [교정] display_score를 주 지표로 가져오도록 변경
        query = f"""
            SELECT trade_date, close_price, display_score as ldy_score, final_score, ai_comment
            FROM daily_recommend
            WHERE code = '{code_str}'
            ORDER BY trade_date ASC
        """
        df = conn.execute(query).fetchdf()
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

def plot_score_history_chart(history_df, stock_name):
    """
    점수(LDY, RANK)와 주가(Close)를 이중축 차트로 시각화
    """
    if history_df is None or history_df.empty:
        return None

    # 날짜 처리
    df = history_df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 점수 (좌측 축)
    fig.add_trace(
        go.Scatter(
            x=df['trade_date'], y=df['ldy_score'],
            name="기초 점수 (LDY)",
            mode='lines+markers',
            line=dict(color='#29B6F6', width=2),
            marker=dict(size=6)
        ),
        secondary_y=False,
    )
    
    # 2. 랭크 점수 (좌측 축)
    if 'rank_score' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['trade_date'], y=df['rank_score'],
                name="랭킹 점수 (Rank)",
                mode='lines',
                line=dict(color='#FFA726', width=2, dash='dot'),
            ),
            secondary_y=False,
        )

    # 3. 주가 (우측 축)
    fig.add_trace(
        go.Scatter(
            x=df['trade_date'], y=df['close_price'],
            name="주가",
            mode='lines',
            line=dict(color='#BDBDBD', width=1),
            opacity=0.4
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=f"📈 {stock_name} - 점수 히스토리", font=dict(size=15)),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified"
    )
    
    # 축 설정
    fig.update_yaxes(title_text="점수", range=[0, 105], secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text="주가", showgrid=False, secondary_y=True)

    return fig

def get_route_color(route):
    """전략 상태(ROUTE)에 따른 시각적 배지 색상 반환 (v11.0 표준)"""
    colors = {
        "ATTACK": "#FF4B4B",  # 🚀 강력 레드
        "ARMED": "#FFA726",   # 🔫 오렌지
        "WAIT": "#29B6F6",    # 🧱 블루
        "OVERHEAT": "#757575", # ⚠️ 그레이
        "NEUTRAL": "#BDBDBD"  # ⚪️ 연그레이
    }
    r = str(route).upper()
    for key, color in colors.items():
        if key in r:
            return color
    return "#BDBDBD"

def normalize_code(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)      # 660.0 같은 거 제거
    s = re.sub(r"[^0-9]", "", s)    # 숫자만 남김
    return s.zfill(6) if s else ""  # 6자리로

# -------------------- [v9.0 유틸리티 추가] --------------------


def postprocess_codes(df: pd.DataFrame) -> pd.DataFrame:
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].apply(normalize_code)
    return df

# -----------------------------------------------------------
# [주의] 아래 import 구문들은 맨 앞줄(들여쓰기 없음)에 있어야 합니다.
# -----------------------------------------------------------

from auth_user import (
    render_auth_box,  
    get_user, 
    list_users, 
    update_user_role,
    load_inquiry_items, 
    save_inquiry_items, 
    _now_utc_str,
    load_subscriptions_db, 
    save_subscriptions_db,
    toggle_user_ban, 
    grant_all_users_trial
)

try:
    # 들여쓰기와 괄호 위치를 정밀하게 맞췄습니다.
    from version_info import (
        PRIME_TG_JOIN_URL,
        APP_VERSION,
        CHANGELOG,
        get_version_label,
        get_latest_log,
    )
except ImportError as e:
    # 보급로가 끊겼을 때를 대비한 비상 식량(Fallback)
    # st.error는 앱 시작 시점에 호출되면 사이드바 이전에 나타나므로 경고 로그로 대체하거나 유지
    st.warning(f"⚠️ 시스템 버전 정보 로드 실패 (기본값 가동): {e}")
    APP_VERSION = "12.3.0"
    PRIME_TG_JOIN_URL = "https://t.me/+DovDEluWnEJhOTY1"
    CHANGELOG = []
    def get_version_label(include_build=True): return "12.3.0"
    def get_latest_log(): return None


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
    RSI(14) 시리즈 — [v20.3] indicators.calc_rsi() SSOT 통일
    엔진/대시보드/차트 모두 동일한 Wilder RMA 기반 RSI 사용.
    """
    from indicators import calc_rsi
    return calc_rsi(pd.to_numeric(close, errors="coerce"), period)

# ---------------------------
# 오픈베타 영구 PRIME 사용자
# ---------------------------
BETA_PRIME_USERS = set()  # 👈 베타 테스터 목록 제거 (이제 모두 결제 필요)

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
    # (BETA_PRIME_USERS가 비어있으므로 이 로직은 이제 실행되지 않습니다)
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

# =====================================================================
# 📱 [v20.0.1] 모바일 UX 개선 — 세로 고정 + 탭 전환 유지
# =====================================================================
st.markdown("""
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
</head>
<style>
/* 📱 화면 세로 고정 (가로 전환 방지) */
@media screen and (orientation: landscape) and (max-width: 1024px) {
    html {
        transform: rotate(-90deg);
        transform-origin: left top;
        width: 100vh;
        height: 100vw;
        overflow-x: hidden;
        position: absolute;
        top: 100%;
        left: 0;
    }
}
</style>
<script>
// 📱 탭 전환 시 Streamlit WebSocket 유지 (튕김 방지)
(function() {
    // 1) visibilitychange로 탭 복귀 시 재연결 시도
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) {
            // 탭 복귀 시 — Streamlit이 이미 끊겼으면 부드럽게 복구
            try {
                const ws = window._stWebSocket || 
                    (window.parent && window.parent._stWebSocket);
                if (ws && ws.readyState > 1) {
                    // WebSocket이 닫힌 상태면 페이지 새로고침 (가장 안정적)
                    // 단, 3초 대기 후 — 즉시 리로드하면 깜빡임
                    setTimeout(function() {
                        if (document.querySelector('[data-testid="stException"]') ||
                            document.querySelector('.stError')) {
                            window.location.reload();
                        }
                    }, 2000);
                }
            } catch(e) {}
        }
    });

    // 2) 백그라운드에서 WebSocket keep-alive 핑
    setInterval(function() {
        if (document.hidden) {
            try {
                // Streamlit의 내부 WebSocket에 빈 메시지 전송 시도
                const frames = document.querySelectorAll('iframe');
                frames.forEach(function(f) {
                    try { f.contentWindow.postMessage('ping', '*'); } catch(e) {}
                });
            } catch(e) {}
        }
    }, 15000);  // 15초마다

    // 3) Screen Orientation API로 세로 고정 시도
    try {
        if (screen.orientation && screen.orientation.lock) {
            screen.orientation.lock('portrait').catch(function(){});
        }
    } catch(e) {}
})();
</script>
""", unsafe_allow_html=True)

# =====================================================================
# 🎨 [v18.0] Global UI Theme — Premium Dark Finance Dashboard
# =====================================================================
st.markdown("""
<style>
/* ─── Import Fonts ─── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Root Variables ─── */
:root {
    --accent-blue: #3B82F6;
    --accent-cyan: #06B6D4;
    --accent-green: #10B981;
    --accent-red: #EF4444;
    --accent-amber: #F59E0B;
    --accent-purple: #8B5CF6;
    --card-bg: rgba(255,255,255,0.03);
    --card-border: rgba(255,255,255,0.08);
    --card-hover: rgba(255,255,255,0.06);
    --glass-bg: rgba(255,255,255,0.05);
    --glass-border: rgba(255,255,255,0.1);
    --text-primary: #F1F5F9;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --gradient-blue: linear-gradient(135deg, #3B82F6, #06B6D4);
    --gradient-green: linear-gradient(135deg, #10B981, #34D399);
    --gradient-red: linear-gradient(135deg, #EF4444, #F97316);
    --gradient-purple: linear-gradient(135deg, #8B5CF6, #EC4899);
}

/* ─── General Typography ─── */
html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, sans-serif !important;
}
code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}
h1, h2, h3 { font-weight: 600 !important; letter-spacing: -0.02em; }

/* ─── Metric Cards ─── */
[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 16px 20px;
    transition: all 0.2s ease;
}
[data-testid="stMetric"]:hover {
    background: var(--card-hover);
    border-color: var(--accent-blue);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(59,130,246,0.15);
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.7;
}
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 1.4rem !important;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--card-bg);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--card-border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 8px 16px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: var(--gradient-blue) !important;
    color: white !important;
    font-weight: 600;
}

/* ─── Containers & Cards ─── */
[data-testid="stExpander"] {
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
div.stContainer, div[data-testid="stVerticalBlock"] > div[style*="border"] {
    border-radius: 12px !important;
    border-color: var(--card-border) !important;
}

/* ─── Buttons ─── */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    font-family: 'Outfit', sans-serif;
    transition: all 0.2s ease;
    border: 1px solid var(--card-border);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59,130,246,0.25);
}
.stButton > button[kind="primary"] {
    background: var(--gradient-blue);
    border: none;
}

/* ─── DataFrames ─── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--card-border);
}

/* ─── Progress bars ─── */
.stProgress > div > div > div {
    background: var(--gradient-blue);
    border-radius: 8px;
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--card-border);
}

/* ─── Hero Banner ─── */
.hero-banner {
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(6,182,212,0.08), rgba(139,92,246,0.06));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.08), transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 0;
    background: linear-gradient(135deg, #3B82F6, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 0.9rem;
    opacity: 0.6;
    margin-top: 6px;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: var(--accent-blue);
    margin-top: 10px;
}

/* ─── Kanban Cards ─── */
.kanban-header {
    padding: 10px 0;
    margin-bottom: 12px;
    text-align: center;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.02em;
    border-bottom: 3px solid;
    position: relative;
}
.kanban-header .count {
    font-weight: 400;
    opacity: 0.5;
    font-size: 0.85rem;
}
.kanban-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: all 0.2s ease;
}
.kanban-card:hover {
    background: var(--card-hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    border-color: rgba(255,255,255,0.15);
}
.kanban-stock-name {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 8px;
}
.kanban-stock-code {
    font-size: 0.75rem;
    opacity: 0.5;
    font-family: 'JetBrains Mono', monospace;
}
.kanban-scores {
    display: flex;
    gap: 12px;
    margin: 8px 0;
}
.kanban-score-item {
    font-size: 0.8rem;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(255,255,255,0.05);
}
.kanban-score-item b { font-size: 1rem; }
.kanban-price {
    font-size: 0.78rem;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.kanban-rr-bar {
    height: 4px;
    border-radius: 2px;
    background: rgba(255,255,255,0.08);
    margin-top: 8px;
    overflow: hidden;
}
.kanban-rr-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
}

/* ─── Route Badge ─── */
.route-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    padding: 16px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.route-badge::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 60%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05));
    pointer-events: none;
}
.route-badge .label {
    font-size: 0.8rem;
    opacity: 0.8;
    display: block;
    margin-bottom: 4px;
}
.route-badge .value {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 3px;
}

/* ─── Disclaimer Banner ─── */
.disclaimer-bar {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 0.78rem;
    line-height: 1.6;
    color: var(--text-secondary);
    margin-bottom: 16px;
}
.disclaimer-bar strong { color: var(--accent-amber); }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # 기존 버전 표시 코드 대신 아래 함수 사용
    version_info.render_sidebar_version_badge()
    if st.button("🔄 데이터/캐시 강제 새로고침"):
        st.cache_data.clear()
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

st.markdown(f"""
<div class="hero-banner">
    <p class="hero-title">💎 LDY Pro Trader v{APP_VERSION}</p>
    <p class="hero-subtitle">AI Quant Analysis & Portfolio Manager — Scoring · Subscription · Portfolio</p>
    <span class="hero-badge">🤖 LSTM + XGBoost Ensemble · 16-Feature AI Engine</span>
</div>
""", unsafe_allow_html=True)

with st.expander("⚠️ 투자 유의사항 (클릭하여 펼치기)", expanded=False):
    st.markdown("""
    <div class="disclaimer-bar">
        LDY Pro Trader는 <strong>데이터·알고리즘 기반 분석 도구</strong>입니다. 
        제공되는 정보는 참고용이며, 매수·매도 및 수익 보장을 하지 않습니다. 
        투자 판단과 결과는 <strong>전적으로 이용자 본인</strong>에게 귀속되며, 
        본 서비스 및 개발자는 법적 책임을 부담하지 않습니다.
    </div>
    """, unsafe_allow_html=True)

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

# 🔐 보안키 — 반드시 Streamlit Secrets 또는 환경변수에서 설정할 것
KEY_PRO   = get_conf("LDY_KEY_PRO",   "")
KEY_PRIME = get_conf("LDY_KEY_PRIME", "")
ADMIN_KEY = get_conf("LDY_ADMIN_KEY", "")

# 💳 결제 계좌 정보
BANK_ACCOUNT = get_conf("LDY_BANK_ACCOUNT", "카카오뱅크 3333-22-2658701")
BANK_HOLDER  = get_conf("LDY_BANK_HOLDER",  "이OO")

# 📊 스코어링 상수
PASS_EBS          = float(get_conf("LDY_PASS_EBS",          4))
MIN_TURN_KOSPI    = float(get_conf("LDY_MIN_TURN_KOSPI",    200.0))
MIN_TURN_KOSDAQ   = float(get_conf("LDY_MIN_TURN_KOSDAQ",   100.0))
MIN_TURN_DEFAULT  = float(get_conf("LDY_MIN_TURN_DEFAULT",  100.0))

# [v20.6] 독자 가중치 상수 제거 — scoring_engine.build_global_score()로 SSOT 통합
RSI_LOW, RSI_HIGH = 45, 65  # liquidity_gate 등 UI에서 참조

# ---------------------------
# 유틸 함수
# ---------------------------
def z6(x):
    return str(x).zfill(6) if str(x).isdigit() else str(x)


def ensure_turnover(df):
    if "거래대금(억원)" not in df.columns and "거래대금(원)" in df.columns:
        df["거래대금(억원)"] = (nz_num(df["거래대금(원)"]) / 1e8).round(2)
    return df

def normalize_cols(df):
    return ensure_turnover(df)

def make_preview(df, n=5):
    """ _CSV_RANK 숫자 변환 후 정렬 (우선순위: CSV순 > LDY순 > 점수순) """
    if df is None or df.empty: return df

    # 1순위: CSV가 준 랭크(_CSV_RANK) - 파일 저장 순서 존중
    if "_CSV_RANK" in df.columns:
        tmp = df.copy()
        tmp["_CSV_RANK"] = pd.to_numeric(tmp["_CSV_RANK"], errors="coerce")
        cols = ["_CSV_RANK"]
        if "_CSV_ROW" in tmp.columns: cols.append("_CSV_ROW")
        return tmp.sort_values(cols, ascending=True).head(n).copy()

    # 2순위: LDY_RANK
    if "LDY_RANK" in df.columns:
        return df.sort_values("LDY_RANK", ascending=True).head(n).copy()

    # 3순위: 점수 기반 (v15: Timing -> AI -> Money)
    sort_keys = []
    asc_opts = []
    for k in ["TIMING_SCORE", "AI_SCORE", "RANK_SCORE", "FINAL_SCORE", "거래대금(억원)"]:
        if k in df.columns:
            sort_keys.append(k)
            asc_opts.append(False)
            
    if sort_keys:
        return df.sort_values(sort_keys, ascending=asc_opts).head(n).copy()

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
    종목명 -> 종목코드 매핑
    1순위: Collector가 수집해둔 로컬 파일 (data/krx_codes_*.csv) 사용 -> 가장 빠르고 안전
    2순위: pykrx 라이브러리 (Live)
    3순위: FinanceDataReader (Live)
    """
    mapping = {}

    # 1. 로컬 캐시 파일 우선 로드 (Collector가 수집한 파일 활용)
    #    -> 외부 통신 없이 즉시 로드되므로 에러가 안 납니다.
    try:
        # data 폴더 내의 krx_codes_*.csv 파일 검색
        pattern = os.path.join(DATA_DIR, "krx_codes_*.csv")
        files = glob.glob(pattern)
        
        if files:
            # 가장 최근에 수정된 파일 선택
            latest_file = max(files, key=os.path.getmtime)
            df = pd.read_csv(latest_file, dtype=str)
            
            if "종목명" in df.columns and "종목코드" in df.columns:
                for _, row in df.iterrows():
                    name = str(row["종목명"]).strip()
                    code = str(row["종목코드"]).strip().zfill(6)
                    if name and code:
                        # 그대로 저장
                        mapping.setdefault(name, code)
                        # 공백 제거 버전도 저장 (예: 'HD현대일렉트릭', 'HD 현대일렉트릭')
                        mapping.setdefault(name.replace(" ", ""), code)
                
                # 로컬 로드 성공 시 로그 남기고 바로 반환 (외부 API 호출 생략)
                logger.info(f"✅ Loaded code map from local file: {os.path.basename(latest_file)} ({len(mapping)} items)")
                return mapping
    except Exception as e:
        logger.warning(f"Local code map load failed: {e}")

    # 2. pykrx (Live Fallback) - 로컬 파일 없을 때만 실행
    if PYKRX_OK:
        try:
            today_dt = now_kst().date()
            today = None
            # 최근 5일 내 거래일 찾기
            for i in range(5):
                ymd = (today_dt - timedelta(days=i)).strftime("%Y%m%d")
                try:
                    # 해당 날짜에 티커 리스트가 나오는지 확인
                    if stock.get_market_ticker_list(ymd, market="KOSPI"):
                        today = ymd
                        break
                except (ValueError, AttributeError): pass
            
            if today:
                for mkt in ["KOSPI", "KOSDAQ"]:
                    tickers = stock.get_market_ticker_list(today, market=mkt)
                    for t in tickers:
                        name = stock.get_market_ticker_name(t)
                        if not isinstance(name, str): continue
                        name = name.strip()
                        if not name: continue
                        
                        code = str(t).zfill(6)
                        
                        # 그대로 저장
                        mapping.setdefault(name, code)
                        # 공백 제거 버전도 저장
                        mapping.setdefault(name.replace(" ", ""), code)
        except Exception as e:
            logger.warning(f"pykrx code map failed: {e}")

    # 3. FDR (Last Resort) - 위 두 방법 다 실패했을 때만 실행
    if not mapping and FDR_OK:
        try:
            df = fdr.StockListing("KRX")
            for _, row in df.iterrows():
                name = str(row.get("Name", "")).strip()
                code = str(row.get("Code", "")).strip().zfill(6)
                if name:
                    # 그대로 저장
                    mapping.setdefault(name, code)
                    # 공백 제거 버전도 저장
                    mapping.setdefault(name.replace(" ", ""), code)
        except Exception as e:
            # FDR 에러는 로그만 남기고 무시 (앱 중단 방지)
            logger.warning(f"FDR code map failed: {e}")

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
                rsi = calc_rsi_series(df["Close"], 14)
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

        # 🔹 RSI(14) — [v20.3] indicators.calc_rsi SSOT 통일
        df['RSI14_CHART'] = calc_rsi_series(df['Close'], 14)
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


# -----------------------------------------------------------
# [v12.0 New] AI 게이지 & 켈리 자금관리 차트
# -----------------------------------------------------------






def plot_correlation_heatmap(df_target):
    """Top 종목들의 주가 상관관계 히트맵 (최근 60일 기준)"""
    if df_target is None or df_target.empty: return None
    
    targets = df_target.head(10)
    codes = targets['종목코드'].astype(str).str.zfill(6).tolist()
    names = targets['종목명'].tolist()
    
    price_data = {}
    for code, name in zip(codes, names):
        try:
            d = get_stock_chart_data(code)
            if d is not None and not d.empty:
                price_data[name] = d['Close'].tail(60)
        except Exception:
            continue
            
    if not price_data: return None
    
    df_prices = pd.DataFrame(price_data).dropna()
    if df_prices.shape[1] < 2: return None 
    
    df_corr = df_prices.corr()
    
    fig = px.imshow(
        df_corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="<b>🔗 Top 10 종목 간 상관관계 (Correlation)</b>"
    )
    fig.update_layout(
        height=400, margin=dict(t=50, b=10, l=10, r=10),
        xaxis_side="top"
    )
    return fig


def plot_interactive_chart(
    df: pd.DataFrame,
    code: str,
    name: str,
    entry=None,
    stop=None,
    target1=None,
    target2=None,
    target_atr=None,
    vwap=None,
    show_bb: bool = True,
    show_kc: bool = False,
    show_rsi: bool = False,
    show_vwap: bool = False,
    show_hma: bool = False,
    show_obv: bool = False,
    show_vp: bool = True,
):
    if df is None or df.empty:
        st.warning("차트 데이터 없음")
        return go.Figure()

    df = df.copy()
    col_map = {"시가":"Open", "고가":"High", "저가":"Low", "종가":"Close", "거래량":"Volume"}
    df.rename(columns={k:v for k,v in col_map.items() if k in df.columns}, inplace=True)

    # 행 높이 설정
    rows = 2
    if show_rsi: rows += 1
    if show_obv: rows += 1
    
    row_heights = [0.6] + [0.4 / (rows - 1)] * (rows - 1)

    fig = make_subplots(
        rows=rows, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, 
        row_heights=row_heights
    )

    # 🎨 [디자인] TradingView 스타일 컬러
    C_UP = '#089981'   # 모던 그린
    C_DOWN = '#F23645' # 모던 레드
    C_MA20 = '#FFD700' # 골드
    C_BB_FILL = 'rgba(33, 150, 243, 0.07)' # 아주 연한 파랑
    
    # 1. 캔들 차트
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color=C_UP, increasing_fillcolor=C_UP,
        decreasing_line_color=C_DOWN, decreasing_fillcolor=C_DOWN,
        showlegend=False
    ), row=1, col=1)

    # 2. 이동평균선
    if "MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA 20", line=dict(color=C_MA20, width=1.5)), row=1, col=1)
    
    if "WEEKLY_MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["WEEKLY_MA20"], name="주봉 20선", 
                                 line=dict(color='rgba(255, 255, 255, 0.4)', width=2, dash='dot')), row=1, col=1)

    # 3. 볼린저 밴드
    if show_bb and "BB_UPPER" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_UPPER"], line=dict(width=0), showlegend=False, hoverinfo='skip'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_LOWER"], fill='tonexty', fillcolor=C_BB_FILL, 
            line=dict(width=0), name="Bollinger", hoverinfo='skip'
        ), row=1, col=1)

    # 4. Keltner Channel
    if show_kc and "KC_UPPER" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["KC_UPPER"], line=dict(width=1, dash='dot', color='rgba(224, 64, 251, 0.5)'), name="KC High"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["KC_LOWER"], line=dict(width=1, dash='dot', color='rgba(224, 64, 251, 0.5)'), name="KC Low"), row=1, col=1)

    # 5. SuperTrend
    if "Trend" in df.columns and "SuperTrend" in df.columns:
        up = df[df["Trend"] == 1]["SuperTrend"]
        if not up.empty:
            fig.add_trace(go.Scatter(x=up.index, y=up, mode='lines', line=dict(color=C_UP, width=2), name='SuperTrend'), row=1, col=1)
        down = df[df["Trend"] == -1]["SuperTrend"]
        if not down.empty:
            fig.add_trace(go.Scatter(x=down.index, y=down, mode='lines', line=dict(color=C_DOWN, width=2), showlegend=False), row=1, col=1)

    # 6. 중요 가격 라인 (🔥 호출 방식 변경됨: 내부에서 변환)
    def _add_line(val, color, label, style="dash"):
        # 값이 없으면 패스
        if val is None: return
        if isinstance(val, float) and math.isnan(val): return

        # 리스트/시리즈 처리 (안전장치)
        if hasattr(val, '__len__') and not isinstance(val, str):
            try:
                if len(val) > 0:
                    val = val.iloc[0] if hasattr(val, 'iloc') else val[0]
                else:
                    return
            except Exception:
                pass

        # 문자열 변환 및 콤마 제거
        try:
            s_val = str(val).replace(',', '').strip()
            val_float = float(s_val)
            
            if val_float > 0:
                fig.add_hline(y=val_float, line_dash=style, line_color=color, line_width=1, 
                              annotation_text=label, annotation_position="top right", annotation_font_color=color)
        except Exception:
            return 

    # 🔥 [중요] 여기서 float()를 쓰지 말고 변수(entry, stop 등)를 그대로 넘겨야 합니다!
    _add_line(entry, "#2962FF", "Entry", "solid")
    _add_line(stop, "#FF3B30", "Stop Loss")
    _add_line(target1, "#00E676", "T1 목표")
    _add_line(target2, "#FFD600", "T2 목표", "dashdot")
    _add_line(target_atr, "rgba(150,150,150,0.5)", "ATR 참고", "dot")

    # 7. 거래량
    current_row = 2
    if "Volume" in df.columns:
        colors = [C_UP if c >= o else C_DOWN for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Vol", marker_color=colors, marker_line_width=0, opacity=0.6), row=current_row, col=1)
        current_row += 1

    # 8. RSI
    if show_rsi and "RSI14_CHART" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14_CHART"], name="RSI", line=dict(color='#E040FB', width=1.5)), row=current_row, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, layer="below", row=current_row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="blue", opacity=0.1, layer="below", row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        current_row += 1

    # 9. OBV
    if show_obv and "OBV" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color='#29B6F6', width=1.5), fill='tozeroy', fillcolor='rgba(41, 182, 246, 0.1)'), row=current_row, col=1)

    # 10. 매물대
    if show_vp:
        fig = add_volume_profile(fig, df)

    # 11. 최종 레이아웃
    fig.update_layout(
        title=dict(text=f"<b>{name}</b> <span style='font-size:12px;color:gray;'>({code})</span>", x=0.02, y=0.98),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        height=600 if show_rsi else 500,
        margin=dict(l=10, r=50, t=50, b=10),
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', side='right'),
    )
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.1)', side='right')

    return fig

    


def render_kanban_board(df):
    """
    [v18.0] Active 종목을 상태별 글래스모피즘 카드로 시각화 (Kanban View)
    """
    if df.empty:
        st.info("표시할 종목이 없습니다.")
        return

    col_attack, col_armed, col_watch = st.columns(3)
    
    df_attack = df[df['ROUTE'].astype(str).str.contains("ATTACK|공략|발사", case=False, na=False)]
    df_armed = df[df['ROUTE'].astype(str).str.contains("ARMED|임박|준비|BRK|돌파", case=False, na=False)]
    ex_indices = df_attack.index.union(df_armed.index)
    df_watch = df[~df.index.isin(ex_indices)]

    def _render_card(container, title, sub_df, color, icon, gradient):
        with container:
            st.markdown(
                f"<div class='kanban-header' style='border-color:{color};'>"
                f"{icon} {title} <span class='count'>({len(sub_df)})</span></div>", 
                unsafe_allow_html=True
            )
            
            if sub_df.empty:
                st.caption("비어 있음")
                
            for _, row in sub_df.iterrows():
                final_score = row.get('FINAL_SCORE', 0)
                trigger_score = row.get('TRIGGER_SCORE', 0)
                buy_p = int(pd.to_numeric(str(row.get('추천매수가', 0)).replace(',', ''), errors='coerce') or 0)
                stop_p = int(pd.to_numeric(str(row.get('손절가', 0)).replace(',', ''), errors='coerce') or 0)
                rr_val = row.get('RR1', 0) or 0
                rr_width = min(float(rr_val) / 3.0 * 100, 100)
                
                # 점수에 따른 뱃지 색상
                score_color = "#10B981" if final_score >= 80 else "#3B82F6" if final_score >= 60 else "#94A3B8"
                
                price_html = ""
                if buy_p > 0:
                    t1_p = int(pd.to_numeric(str(row.get('추천매도가1', 0)).replace(',', ''), errors='coerce') or 0)
                    t1_html = f" <span style='color:#10B981;opacity:0.8;'>🟢 {t1_p:,}</span>" if t1_p > 0 else ""
                    price_html = f"""
                    <div class='kanban-price'>
                        🎯 <b>{buy_p:,}</b> 
                        <span style='color:#EF4444;opacity:0.8;'>🛡️ {stop_p:,}</span>{t1_html}
                    </div>"""
                
                rr_html = ""
                if rr_val and float(rr_val) > 0:
                    rr_color = "#10B981" if float(rr_val) >= 2.0 else "#F59E0B" if float(rr_val) >= 1.0 else "#EF4444"
                    rr_html = f"""
                    <div style='display:flex;align-items:center;gap:8px;margin-top:6px;font-size:0.75rem;'>
                        <span style='opacity:0.5;'>R:R</span>
                        <div class='kanban-rr-bar' style='flex:1;'>
                            <div class='kanban-rr-fill' style='width:{rr_width}%;background:{rr_color};'></div>
                        </div>
                        <span style='color:{rr_color};font-weight:600;'>{float(rr_val):.1f}</span>
                    </div>"""
                
                st.markdown(f"""
                <div class='kanban-card'>
                    <div class='kanban-stock-name'>
                        {row['종목명']} <span class='kanban-stock-code'>{row['종목코드']}</span>
                    </div>
                    <div class='kanban-scores'>
                        <div class='kanban-score-item'>🏆 <b style='color:{score_color};'>{final_score:.0f}</b></div>
                        <div class='kanban-score-item'>🔥 <b>{trigger_score:.0f}</b></div>
                    </div>
                    {price_html}
                    {rr_html}
                </div>
                """, unsafe_allow_html=True)

    _render_card(col_attack, "진입 ATTACK", df_attack, "#EF4444", "🚀", "var(--gradient-red)")
    _render_card(col_armed, "준비 ARMED", df_armed, "#F59E0B", "🔫", "var(--gradient-purple)")
    _render_card(col_watch, "관찰 WATCH", df_watch, "#3B82F6", "👀", "var(--gradient-blue)")

# -------------------------------------------------------------
# 🔥 [v14.2 Dashboard Engine] Time-Aware State Machine
# -------------------------------------------------------------
def get_survival_days(current_codes: list, lookback: int = 15) -> dict:
    """
    최근 N일간의 데이터를 역추적하여 '상위권 생존 일수(Days Alive)'를 계산합니다.
    단순 등장이 아니라 '연속 생존' 여부가 핵심입니다.
    """
    days_map = {code: 1 for code in current_codes} # 신규 진입은 1일차
    
    # 최신순 정렬 (latest 제외, 날짜 역순)
    files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, "recommend_*.csv")) if "latest" not in f], reverse=True)
    
    # 오늘(0번) 제외, 어제(1번)부터 탐색
    history_files = files[1:lookback+1] 
    
    if not history_files:
        return days_map

    # 연속성 체크를 위한 임시 집합
    survivors = set(current_codes)

    for f_path in history_files:
        try:
            df_past = pd.read_csv(f_path, usecols=["종목코드"], dtype=str)
            past_set = set(df_past["종목코드"].apply(lambda x: str(x).zfill(6)))
            
            # 이번 과거 파일에도 살아남은 종목만 카운트 증가
            # 한 번이라도 끊기면(탈락했으면) 더 이상 카운트하지 않음 (연속성)
            next_survivors = set()
            for code in survivors:
                if code in past_set:
                    days_map[code] += 1
                    next_survivors.add(code)
            
            survivors = next_survivors
            if not survivors: break # 더 이상 생존자가 없으면 중단
            
        except Exception:
            continue
            
    return days_map



def augment_display_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    
    # 1. 생존일 계산 로직
    try:
        codes = df["종목코드"].astype(str).str.zfill(6).tolist()
        survival_map = get_survival_days(codes, lookback=20)
        df["생존일"] = df["종목코드"].apply(lambda x: survival_map.get(str(x).zfill(6), 1))
    except Exception:
        df["생존일"] = 1

    # 2. [핵심 수정] CSV의 텍스트 값("ATTACK")을 직접 확인하여 매핑
    # schema.py 의존성을 제거하여 매칭 오류 방지
    def map_route_to_ui(route_val):
        r = str(route_val).strip()
        
        # 1. 집중 공략 (ATTACK)
        if r == "ATTACK": 
            return "🚀 집중 공략 (Active)", 0
        
        # 2. 트리거 임박 (ARMED)
        if r == "ARMED": 
            return "🔫 트리거 임박 (Ready)", 10
        
        # 3. 과열 (OVERHEAT)
        if r == "OVERHEAT": 
            return "⛔ 과열/주의 (Caution)", 90
            
        # 4. 관찰 대기 (WAIT)
        if r == "WAIT":
            return "👀 추세 관찰 (Wait)", 50
            
        # 5. 기본값
        return "⚪ 일반 관망 (Neutral)", 60

    if "ROUTE" in df.columns:
        mapped = df["ROUTE"].apply(map_route_to_ui)
        df["상태"] = mapped.apply(lambda x: x[0])
        df["_STATE_SORT"] = mapped.apply(lambda x: x[1])
    else:
        df["상태"] = "⚪ 일반 관망 (Neutral)"
        df["_STATE_SORT"] = 60

    # 3. [중요] Active 플래그 기준 (명시적 텍스트 매칭)
    if "ROUTE" in df.columns:
        # ATTACK(공략)과 ARMED(임박) 두 가지만 Active로 분류
        df["IS_ACTIVE"] = df["ROUTE"].isin(["ATTACK", "ARMED"])
    else:
        # 컬럼 없을 경우 대비
        df["IS_ACTIVE"] = df["_STATE_SORT"] <= 20

    # 4. 제외 사유 매핑 (UI용)
    df["제외사유"] = np.where(
        df["_STATE_SORT"] == 90, "⚠️ 과열/급등주의",
        np.where(df["_STATE_SORT"] >= 50, "⏳ 모멘텀 부족", "-")
    )

    # 5. 추세 데코레이션
    def _deco_trend(val):
        try: v = float(val)
        except (ValueError, TypeError): return "-"
        if v > 1.0: return "📈 급상승"
        if v > 0.0: return "↗️ 우상향"
        if v == 0.0: return "➡️ 횡보"
        return "↘️ 붕괴"
    
    if "Low_Trend_PCT" in df.columns:
        df["추세"] = df["Low_Trend_PCT"].apply(_deco_trend)
    else:
        df["추세"] = "-"
        
    return df
# ---------------------------
# 데이터 로딩
# ---------------------------
def normalize_github_raw(url: str) -> str:
    """GitHub URL을 Raw 포맷으로 통일 (/blob/, /raw/ 모두 대응)"""
    if not isinstance(url, str):
        return ""
    u = url.strip()
    if not u:
        return ""

    u = u.replace("http://", "https://")

    # 이미 raw 도메인이면 그대로
    if "raw.githubusercontent.com/" in u:
        return u

    # github.com/user/repo/blob/branch/file -> raw 변환
    if "github.com/" in u and "/blob/" in u:
        u = u.replace("https://github.com/", "https://raw.githubusercontent.com/")
        u = u.replace("/blob/", "/")
        return u

    # github.com/user/repo/raw/branch/file -> raw 변환
    if "github.com/" in u and "/raw/" in u:
        u = u.replace("https://github.com/", "https://raw.githubusercontent.com/")
        u = u.replace("/raw/", "/")
        return u

    return u

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
    min_map = {"KOSPI": MIN_TURN_KOSPI, "KOSDAQ": MIN_TURN_KOSDAQ}
    turn = nz_num(x_turn)

    # 1) turn이 스칼라면 Series로 승격
    if isinstance(turn, pd.Series):
        turn_s = turn
    else:
        turn_s = pd.Series([turn], dtype="float64")

    # 2) market 처리 (Series vs Scalar)
    if isinstance(market, pd.Series):
        m = market.astype(str)
        if not m.index.equals(turn_s.index):
            fill_val = str(market.iloc[0]) if not market.empty else ""
            m = m.reindex(turn_s.index).fillna(fill_val)
    else:
        m = pd.Series([str(market)] * len(turn_s), index=turn_s.index, dtype="object")

    m = m.astype(str).str.strip().str.upper()
    th = m.map(min_map).fillna(MIN_TURN_DEFAULT)
    return turn_s >= th

# ---------------------------
# [Fix 3] Preview Sort Logic (숫자 정렬 + 순서 보장)
# ---------------------------

def build_global_score(lat, keep_order: bool = False, macro_risk: str = "NORMAL"):
    """
    [v20.6] SSOT 위임 — scoring_engine.build_global_score()에 단일 경로.
    Legacy 호환 시그니처 유지. Dashboard CASE 2 fallback 전용.
    """
    x = lat.copy()
    
    # scoring_engine SSOT 경로 (macro_risk 전달)
    x = _ssot_build_global_score(x, macro_risk=macro_risk)
    
    # UI 호환: LDY_SCORE 매핑
    x["LDY_SCORE"] = x.get("DISPLAY_SCORE", x.get("FINAL_SCORE", 0.0))
    
    # 게이트 + MA20 갭 (UI 전용)
    from shared_utils import nz_num
    if "_GATE_OK" not in x.columns:
        x["_GATE_OK"] = liquidity_gate(
            x.get("거래대금(억원)", 0),
            x.get("시장", pd.Series(np.nan, index=x.index))
        ).fillna(False)
    
    if "MA20" in x.columns:
        x["MA20_GAP"] = ((nz_num(x["종가"]) / nz_num(x["MA20"]) - 1.0) * 100).replace([np.inf, -np.inf], np.nan)
    else:
        x["MA20_GAP"] = np.nan
    
    if not keep_order:
        x = x.sort_values("LDY_SCORE", ascending=False, na_position="last")
        x["LDY_RANK"] = range(1, len(x) + 1)
    
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
        except (ValueError, TypeError):
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

    return "🔺 Watch (상승 준비)" # 기본값을 Watch로 격상

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
    [Updated v14.4]
    1. 데이터 로드 (로컬/원격)
    2. v8 스코어 체계 정밀 판정 (Prescored)
    3. 점수 컬럼 강제 숫자 변환 (Normalize 부작용 방지) - Fix 1
    4. 점수 0점 판정 로직 강화 (abs sum) - Fix 2
    5. ROUTE 유효성 체크 및 TH 용도 분리 - Fix 3
    """
    df_raw = None
    src_type = "unknown"

    # -----------------------------------------------------------
    # 1. 데이터 로드
    # -----------------------------------------------------------
    if local_raw and os.path.exists(local_raw):
        try:
            df_raw = load_csv_path(local_raw)
            if df_raw is not None and not df_raw.empty:
                df_raw = postprocess_codes(df_raw)
                log_src(df_raw, "Local (Priority)")
                src_type = "local"
        except Exception as e:
            logger.warning(f"Local load failed: {e}")

    if df_raw is None or df_raw.empty:
        try:
            df_raw = load_csv_url(raw_url)
            df_raw = postprocess_codes(df_raw)
            log_src(df_raw, "Remote (Fallback)")
            src_type = "remote"
        except Exception as e:
            logger.warning(f"Remote load failed: {e}")

    if df_raw is None or df_raw.empty:
        raise RuntimeError("데이터를 로컬/원격 어디서도 불러오지 못했습니다.")

    # -----------------------------------------------------------
    # 2. 데이터 전처리
    # -----------------------------------------------------------
    df_raw = df_raw.copy().reset_index(drop=True)

    # 랭크 정보 백업
    rank_col = next((c for c in ["LDY_RANK", "RANK", "rank", "순위"] if c in df_raw.columns), None)
    if rank_col:
        # 안전한 정렬을 위해 결측치는 아주 큰 수로 대체
        df_raw["_CSV_RANK"] = pd.to_numeric(df_raw[rank_col], errors="coerce").fillna(999999)
    else:
        df_raw["_CSV_RANK"] = np.arange(1, len(df_raw) + 1)
    
    df_raw["_CSV_ROW"] = np.arange(len(df_raw))
    
    data_ts = infer_data_timestamp(df_raw)
    
    # v8 스코어 체계 판정 (FINAL이 있거나, TOTAL+TRIGGER 조합 등)
    is_v8_scored = (
        ("FINAL_SCORE" in df_raw.columns) or
        (("TOTAL_SCORE" in df_raw.columns) and ("TRIGGER_SCORE" in df_raw.columns)) or
        (("RANK_SCORE" in df_raw.columns) and ("TRIGGER_SCORE" in df_raw.columns))
    )
    
    scored = None
    
    if is_v8_scored:
        # ✅ CASE 1: Prescored (v15.6 Master-Grade / v8.0+ Collector)
        scored = normalize_cols(df_raw) 
        
        # 1. [Fix 1] 지표 데이터 정합성 강화 (Numeric 캐스팅 & 단위 보정)
        # 모든 기술적 지표와 점수 컬럼을 숫자형으로 강제 변환합니다.
        score_cols = [
            "FINAL_SCORE", "DISPLAY_SCORE", "STRUCT_SCORE", "TIMING_SCORE", "AI_SCORE", 
            "ML_SCORE", "TOTAL_SCORE", "LDY_SCORE", "RANK_SCORE", "NEWS_SCORE", "EBS", "RR1",
            "V_POWER", "거래강도", "거래대금(억원)", "Low_Trend_PCT", "Range_Pos", "Vol_Quality"
        ]
        for c in score_cols:
            if c in scored.columns:
                scored[c] = pd.to_numeric(scored[c], errors="coerce").fillna(0.0)

        # 🚨 [수급 무결성] 수급 비중 계산용 '원' 단위 컬럼 부재 시 자동 생성
        # determine_state_dynamic의 분모가 깨지는 것을 원천 차단합니다.
        if "거래대금(원)" not in scored.columns:
            scored["거래대금(원)"] = (scored["거래대금(억원)"] * 100_000_000).astype(float)
        
        # 2. [Face-lifting] 전술적 점수 통합 (v11.0 UI-Elite 규격)
        # 🚨 핵심: 모든 랭킹 및 UI 지표를 '실전 점수(DISPLAY_SCORE)'로 통일합니다.
        # 뉴스/공시가 반영된 DISPLAY_SCORE가 없다면 엔진 점수인 FINAL_SCORE를 주 지표로 삼습니다.
        
        primary_metric = "DISPLAY_SCORE" if "DISPLAY_SCORE" in scored.columns else "FINAL_SCORE"

        if primary_metric in scored.columns:
            # [Fix 2] 점수 0점 판정 및 백업 복구 로직 (데이터 유실 방어)
            # 만약 주 지표가 통째로 0이라면, 백업 점수(TOTAL -> LDY -> RANK) 순으로 탐색합니다.
            if scored[primary_metric].abs().sum() == 0:
                for backup_col in ["TOTAL_SCORE", "LDY_SCORE", "RANK_SCORE"]:
                    if backup_col in scored.columns and scored[backup_col].abs().sum() > 0:
                        scored[primary_metric] = scored[backup_col]
                        logger.info(f"🔄 {primary_metric} 복구 완료 (Source: {backup_col})")
                        break

            # 🚨 [UI Unity] 모든 가시성 점수 컬럼을 주 지표로 강제 동기화
            # 이를 통해 텔레그램, 리스트, 상세 분석의 숫자가 100% 일치하게 됩니다.
            scored["TOTAL_SCORE"] = scored[primary_metric]
            scored["LDY_SCORE"]   = scored[primary_metric]
            scored["RANK_SCORE"]  = scored[primary_metric]
            
            # 구형 파일 대응: DISPLAY_SCORE 컬럼이 없었다면 생성
            if "DISPLAY_SCORE" not in scored.columns:
                scored["DISPLAY_SCORE"] = scored[primary_metric]

        # 3. 필수 컬럼 UI 결측 방어 (v15.1 호환용 Fallback)
        # 분석 엔진 버전에 관계없이 대시보드 UI가 깨지는 것을 방지하기 위해 기본값을 배정합니다.
        fallback_map = {
            "STRUCT_SCORE": "TOTAL_SCORE",   # 기초 체력
            "TIMING_SCORE": "TRIGGER_SCORE", # 진입 타점
            "AI_SCORE":     "ML_SCORE",      # AI 예측치
            "NEWS_SCORE":   0.0,             # 뉴스 가점
            "EBS":          0.0,             # 독립 체크리스트 (정예군 편성 핵심)
            "V_POWER":      0.0,             # 매수세 강도
            "Low_Trend_PCT": 0.0             # 저점 추세
        }

        for target_col, fallback in fallback_map.items():
            if target_col not in scored.columns:
                if isinstance(fallback, str):
                    scored[target_col] = scored.get(fallback, 0.0)
                else:
                    scored[target_col] = fallback

    else:
        # ⚠️ CASE 2: Legacy (점수 부재 시) -> scoring_engine SSOT 재계산
        df = normalize_cols(df_raw)
        
        # [v20.6] macro_risk: CSV 직접 → run_health 간접 → NORMAL fallback
        _resolved_macro = "NORMAL"
        if "MACRO_RISK" in df_raw.columns:
            # v20.6+ CSV에 직접 저장된 값 사용 (추론 불필요)
            _mr_vals = df_raw["MACRO_RISK"].dropna().unique()
            if len(_mr_vals) > 0:
                _resolved_macro = str(_mr_vals[0])
        else:
            # 구버전 CSV fallback: run_meta JSON 직접 → run_health JSON 직접 → 간접 추론
            try:
                import glob as _glob_mod
                # [v20.6.3] run_meta sidecar 우선 (가장 정확)
                _rm_files = sorted(_glob_mod.glob(os.path.join(DATA_DIR, "run_meta_*.json")), reverse=True)
                if _rm_files:
                    with open(_rm_files[0], 'r') as _rmf:
                        _rm_data = json.load(_rmf)
                    if "macro_risk" in _rm_data and _rm_data["macro_risk"]:
                        _resolved_macro = str(_rm_data["macro_risk"])
                else:
                    # run_health JSON fallback
                    _rh_files = sorted(_glob_mod.glob(os.path.join(DATA_DIR, "run_health_*.json")), reverse=True)
                    if _rh_files:
                        with open(_rh_files[0], 'r') as _rhf:
                            _rh_data = json.load(_rhf)
                        if "macro_risk" in _rh_data and _rh_data["macro_risk"]:
                            _resolved_macro = str(_rh_data["macro_risk"])
                        else:
                            _mar = _rh_data.get("max_allowed_route", "ATTACK")
                            if _mar == "WAIT":
                                _resolved_macro = "BEAR"
                            elif _mar == "ARMED":
                                _resolved_macro = "CAUTION"
            except Exception:
                pass
        
        scored = build_global_score(df, keep_order=True, macro_risk=_resolved_macro).reset_index(drop=True)
        
        # [v20.6] scoring_engine이 STRUCT/TIMING/AI_SCORE를 이미 생성 — 보존
        # FINAL_SCORE/DISPLAY_SCORE도 scoring_engine 산출물 그대로 사용
        if "FINAL_SCORE" in scored.columns:
            scored["LDY_SCORE"] = scored["FINAL_SCORE"]
            if "DISPLAY_SCORE" not in scored.columns:
                scored["DISPLAY_SCORE"] = scored["FINAL_SCORE"]
        else:
            # scoring_engine 호출 자체가 실패한 극단적 fallback
            scored["FINAL_SCORE"]   = scored.get("LDY_SCORE", 0.0)
            scored["DISPLAY_SCORE"] = scored.get("LDY_SCORE", 0.0)
            scored["STRUCT_SCORE"]  = scored.get("LDY_SCORE", 0.0)
            scored["TIMING_SCORE"]  = 0.0
            scored["AI_SCORE"]      = 0.0

    # 4. [Rank Restoration] 원본 파일의 배치 순서(Placement) 보존
    # CSV에 기록된 정예군(Top 120 등)의 전술적 순서를 최우선으로 존중합니다.
    if "_CSV_RANK" in df_raw.columns:
        scored["_CSV_RANK"] = df_raw["_CSV_RANK"].values
    scored["_CSV_ROW"] = df_raw["_CSV_ROW"].values

    # -----------------------------------------------------------
    # 3. DART 필터
    # -----------------------------------------------------------
    try:
        dart_key = get_conf("DART_API_KEY", "")
        gemini_key = get_conf("GEMINI_API_KEY", "")
        analyzer = DartAnalyzer(dart_api_key=dart_key, gemini_api_key=gemini_key)
        scored = analyzer.apply_dart_filter(scored)
        
        # DART 악재 반영
        if "DART_SCORE" in scored.columns:
            bad_mask = scored["DART_SCORE"] <= -4
            if bad_mask.any():
                scored.loc[bad_mask, ["FINAL_SCORE", "LDY_SCORE", "TOTAL_SCORE"]] = 0
    except Exception as e:
        logger.warning(f"DART Filter Failed: {e}")

    # -----------------------------------------------------------
    # 4. ROUTE 상태 결정 (Fix 3: Valid Check & TH Usage)
    # -----------------------------------------------------------
    has_valid_route = False
    
    if "ROUTE" in scored.columns:
        # 문자열 정리 ("nan", None 등 제거)
        route_clean = scored["ROUTE"].astype(str).str.strip()
        route_clean = route_clean.replace({"nan": "", "None": "", "NaN": "", "<NA>": ""})
        scored["ROUTE"] = route_clean
        
        # 유효한 라우트가 하나라도 있는지 확인
        has_valid_route = scored["ROUTE"].ne("").any()

    # 분포 통계(TH) 계산
    # 주: Prescored 데이터라도 '차트 시각화(분포선)' 등을 위해 대시보드 기준 TH 계산은 필요함
    TH = compute_dynamic_thresholds(scored)

    if not has_valid_route:
        # 라우트가 없으면 대시보드 로직으로 생성
        scored["ROUTE"] = scored.apply(lambda r: route_tag_dynamic(r, TH), axis=1).fillna("—")

    # -----------------------------------------------------------
    # 5. 정렬 및 필터링
    # -----------------------------------------------------------
    # 원본 CSV 순서 최우선 존중
    scored = scored.sort_values(["_CSV_RANK", "_CSV_ROW"], ascending=[True, True]).reset_index(drop=True)
    
    if "LDY_RANK" not in scored.columns or scored["LDY_RANK"].isnull().all():
        scored["LDY_RANK"] = range(1, len(scored) + 1)

    # Base 필터링
    ebs_s = pd.to_numeric(scored["EBS"], errors="coerce").fillna(0)
    cond_ebs = (ebs_s >= pass_ebs)
    
    if "_GATE_OK" not in scored.columns:
        scored["_GATE_OK"] = liquidity_gate(
            scored["거래대금(억원)"],
            scored.get("시장", pd.Series(np.nan, index=scored.index))
        ).fillna(False)
        
    base = scored[cond_ebs & scored["_GATE_OK"]].copy()
    
    if len(base) < 20:
        base = scored.head(20).copy()

    top20 = base.head(20).copy()
    
    # P_hit (UI용)
    if "P_hit" not in top20.columns:
        # 안전하게 숫자 변환 후 계산
        score_val = pd.to_numeric(top20["FINAL_SCORE"], errors='coerce').fillna(0)
        top20["P_hit"] = (score_val / 100.0 * 0.8).clip(0, 1) * 100

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
    user = render_auth_box()

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

        # 1. 회원 목록 불러오기
        users = list_users()
        
        # --- [진단 모드] Gist 연동 문제 추적 ---
        if not users:
            st.warning("⚠️ 회원 목록이 비어 있습니다.")
            with st.expander("🔍 Gist 연동 진단 (클릭하여 펼치기)", expanded=True):
                import db_utils as _dbm
                
                # Step 1: 시크릿 키 확인
                gist_id = _dbm.GIST_ID
                gist_token = _dbm.GIST_TOKEN
                st.write(f"**1. GIST_ID:** `{gist_id[:12]}...`" if gist_id else "**1. GIST_ID:** ❌ 없음")
                st.write(f"**2. GIST_TOKEN:** `{gist_token[:8]}...`" if gist_token else "**2. GIST_TOKEN:** ❌ 없음")
                
                if not gist_id or not gist_token:
                    st.error("🚫 Gist 인증 키가 없습니다. Streamlit Cloud Settings → Secrets에 LDY_GIST_ID, LDY_GIST_TOKEN을 확인하세요.")
                    st.code("st.secrets 키 목록: " + str(list(st.secrets.keys())))
                else:
                    # Step 2: Gist API 직접 호출
                    import requests as _req
                    try:
                        _url = f"https://api.github.com/gists/{gist_id}"
                        _resp = _req.get(_url, headers={"Authorization": f"token {gist_token}"}, timeout=10)
                        st.write(f"**3. Gist API 응답:** {_resp.status_code}")
                        
                        if _resp.status_code == 200:
                            _files = _resp.json().get("files", {})
                            st.write(f"**4. Gist 파일 목록:** {list(_files.keys())}")
                            
                            if "users_db.json" in _files:
                                _content = _files["users_db.json"]["content"]
                                st.write(f"**5. users_db.json 크기:** {len(_content)} bytes")
                                st.write(f"**6. 데이터 미리보기:** `{_content[:300]}...`")
                                
                                import json as _json
                                _data = _json.loads(_content)
                                st.write(f"**7. 데이터 타입:** `{type(_data).__name__}`, 항목 수: {len(_data) if isinstance(_data, (list, dict)) else 'N/A'}")
                                
                                if isinstance(_data, dict) and "users" in _data:
                                    st.write(f"**8. Dict 형식 (users 키 안에 {len(_data['users'])}명)**")
                                elif isinstance(_data, list) and len(_data) > 0:
                                    st.write(f"**8. List 형식 ({len(_data)}명), 첫 항목 키:** {list(_data[0].keys())}")
                                else:
                                    st.error(f"**8. ❌ 예상치 못한 데이터 형식**")
                                    
                                # Step 3: DB INSERT 직접 테스트
                                try:
                                    _db = _dbm.LDYDBManager()
                                    _count = _db.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                                    st.write(f"**9. DB users 테이블 행 수:** {_count}")
                                    if _count > 0:
                                        _sample = _db.conn.execute("SELECT id, nickname, role FROM users LIMIT 3").fetchall()
                                        st.write(f"**10. 샘플 데이터:** {_sample}")
                                    _db.close()
                                except Exception as _e:
                                    st.error(f"**9. DB 오류:** {_e}")
                            else:
                                st.error(f"❌ 'users_db.json' 파일이 Gist에 없습니다.")
                        elif _resp.status_code == 401:
                            st.error("🚫 Gist 토큰이 만료되었거나 권한이 없습니다.")
                        elif _resp.status_code == 404:
                            st.error("🚫 Gist ID가 잘못되었습니다.")
                        else:
                            st.error(f"❌ API 오류: {_resp.text[:300]}")
                    except Exception as _e:
                        st.error(f"❌ 네트워크 오류: {_e}")
        # --- [진단 모드 끝] ---
        
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
                    except (ValueError, TypeError): pass
            
            dau_pct = f"{dau_count/total_users*100:.1f}%"
            wau_pct = f"{wau_count/total_users*100:.1f}%"
            
            st.markdown(f"""
            <div style="display:flex; gap:12px; margin-bottom:16px;">
                <div style="flex:1; background:linear-gradient(135deg, rgba(59,130,246,0.12), rgba(59,130,246,0.04)); border:1px solid rgba(59,130,246,0.2); border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:0.75rem; opacity:0.6; text-transform:uppercase; letter-spacing:0.05em;">총 가입자</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#3B82F6;">{total_users}</div>
                </div>
                <div style="flex:1; background:linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.04)); border:1px solid rgba(239,68,68,0.2); border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:0.75rem; opacity:0.6; text-transform:uppercase; letter-spacing:0.05em;">DAU (24h)</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#EF4444;">{dau_count}</div>
                    <div style="font-size:0.7rem; opacity:0.5;">{dau_pct}</div>
                </div>
                <div style="flex:1; background:linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.04)); border:1px solid rgba(16,185,129,0.2); border-radius:12px; padding:16px; text-align:center;">
                    <div style="font-size:0.75rem; opacity:0.6; text-transform:uppercase; letter-spacing:0.05em;">WAU (7일)</div>
                    <div style="font-size:1.8rem; font-weight:700; color:#10B981;">{wau_count}</div>
                    <div style="font-size:0.7rem; opacity:0.5;">{wau_pct}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if not users:
            st.info("회원이 없습니다.")
        else:
            # 2. 회원 목록 테이블
            subs_db = load_subs_db()
            subs = subs_db.get("subs", {})
            rows = []
            
            today = now_kst().date()

            for u in users:
                email = u.get("login_id")
                role = u.get("role", "free")
                
                is_banned = u.get("is_banned", False)
                
                expire_at_str = "-"
                is_expired = False
                
                if role == "admin":
                    expire_at_str = "∞ (Admin)"
                elif email in subs:
                    expire_at_str = subs[email].get("expire_at", "-")
                    try:
                        exp_date = datetime.strptime(expire_at_str, "%Y-%m-%d").date()
                        if exp_date < today:
                            is_expired = True
                    except (ValueError, TypeError):
                        pass
                
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
                    "만료일": expire_at_str,
                    "상태": status_txt,
                    "최근접속": to_kst_str(u.get("last_login")),
                    "_is_expired": is_expired
                })

            df_users = pd.DataFrame(rows)
            
            c_filter1, c_filter2 = st.columns(2)
            with c_filter1:
                show_expired = st.checkbox("📉 만료된 회원만 보기")
            with c_filter2:
                search_query = st.text_input("🔍 이메일 검색", placeholder="user@example.com")

            if show_expired:
                df_users = df_users[df_users["_is_expired"] == True]
            
            if search_query:
                df_users = df_users[df_users["Email"].str.contains(search_query, case=False, na=False)]

            if not df_users.empty and "최근접속" in df_users.columns:
                df_users = df_users.sort_values("최근접속", ascending=False)
                
            st.dataframe(
                df_users.drop(columns=["_is_expired"]),
                use_container_width=True, 
                height=300,
                column_config={
                    "최근접속": st.column_config.TextColumn("최근접속", width="medium"),
                    "만료일": st.column_config.TextColumn("만료일", width="small"),
                    "권한": st.column_config.TextColumn("권한", width="small"),
                    "상태": st.column_config.TextColumn("상태", width="small"),
                }
            )

            # 3. 통합 계정 제어
            st.markdown("##### 🛠️ 계정 제어")
            
            target_list = df_users["Email"].tolist() if not df_users.empty else []
            target_email = st.selectbox("대상 회원 선택", options=target_list, key="admin_target_unified")
            
            c_adm1, c_adm2 = st.columns(2)
            
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
            
            with c_adm2:
                current_ban = False
                if target_email:
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

        # 4. [NEW] 이벤트 관리 (들여쓰기 수정됨)
        st.divider()
        st.markdown("##### 🎉 이벤트 관리")
        st.caption("버튼을 누르면 관리자를 제외한 **모든 회원**의 권한이 Prime으로 변경되고, 만료일이 오늘부터 **7일 후**로 설정됩니다.")
        
        if st.button("🎁 전원 7일 Prime 무료 지급", type="primary", use_container_width=True):
            ok, msg = grant_all_users_trial(days=7)
            if ok:
                st.balloons()
                st.success(msg)
                time.sleep(2)
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
# 👇 [안전 장치 코드 추가] FINAL_SCORE 컬럼이 없으면 임시로 생성 (에러 방지)
if df_latest is not None and not df_latest.empty:
    if "FINAL_SCORE" not in df_latest.columns:
        # FINAL_SCORE가 없으면 TOTAL_SCORE나 LDY_SCORE를 대신 사용
        fallback_score = "TOTAL_SCORE" if "TOTAL_SCORE" in df_latest.columns else "LDY_SCORE"
        df_latest["FINAL_SCORE"] = df_latest[fallback_score]
        
    if "TRIGGER_SCORE" not in df_latest.columns:
        # TRIGGER_SCORE가 없으면 0점으로 초기화
        df_latest["TRIGGER_SCORE"] = 0.0

    # ── [v19.2] Run Health 배너 ──
    _run_status = df_latest["RUN_STATUS"].iloc[0] if "RUN_STATUS" in df_latest.columns else None
    if _run_status == "DEGRADED":
        _degraded_reasons = str(df_latest["DEGRADED_REASONS"].iloc[0]) if "DEGRADED_REASONS" in df_latest.columns else ""
        _reason_labels = {
            "MCAP_EMPTY": "시가총액", "MCAP_ALL_ZERO": "시가총액",
            "BENCH_FAIL": "벤치마크", "BENCH_NAN": "벤치마크",
            "FLOW_ZERO": "수급(외인/기관)", "FLOW_PARTIAL": "수급(개인만)", "NEWS_OFF": "뉴스분석",
            "SECTOR_FAIL": "섹터분석",
        }
        _missing = []
        for _code, _label in _reason_labels.items():
            if _code in _degraded_reasons and _label not in _missing:
                _missing.append(_label)
        _miss_str = ", ".join(_missing) if _missing else _degraded_reasons
        st.warning(f"⚠️ **오늘 분석은 일부 데이터 없이 실행되었습니다** — 누락: {_miss_str}. 추천 정확도가 평소보다 낮을 수 있습니다.")
    elif _run_status == "CRITICAL":
        st.error("🔴 **오늘 분석에 심각한 데이터 오류가 감지되었습니다.** 추천을 참고만 하시고, 내일 결과를 확인해주세요.")

    # ── [v19.2.3] Run Health 상세 리포트 열기 ──
    if _run_status and _run_status != "OK":
        import json as _json
        from glob import glob as _glob
        _health_files = sorted(_glob(os.path.join(DATA_DIR, "run_health_*.json")), reverse=True)
        if _health_files:
            try:
                with open(_health_files[0], "r", encoding="utf-8") as _hf:
                    _health_data = _json.load(_hf)
                with st.expander("📋 실행 건강 리포트 (Run Health Detail)", expanded=False):
                    _h_cols = st.columns(3)
                    _status_emoji = {"OK": "🟢", "DEGRADED": "🟡", "CRITICAL": "🔴"}.get(_health_data.get("status", ""), "⚪")
                    _h_cols[0].metric("런 상태", f"{_status_emoji} {_health_data.get('status', '?')}")
                    _h_cols[1].metric("이슈 수", f"{len(_health_data.get('reasons', []))}건")
                    _h_cols[2].metric("데이터 신선도", "✅" if _health_data.get("data_freshness_ok", True) else "❌ 지연")

                    # 항목별 체크 결과
                    _checks = _health_data.get("checks", {})
                    if _checks:
                        _check_md = "| 항목 | 상태 |\n|---|---|\n"
                        _check_labels = {
                            "MCAP": "시가총액", "MCAP_EMPTY": "시가총액", "MCAP_ALL_ZERO": "시가총액(전행0)",
                            "BENCH": "벤치마크", "BENCH_FAIL": "벤치마크", "BENCH_NAN": "벤치마크(NaN)",
                            "FLOW": "수급(외인/기관)", "FLOW_ZERO": "수급(전행0)", "FLOW_PARTIAL": "수급(개인만)",
                            "NEWS": "뉴스분석", "NEWS_OFF": "뉴스(비활성)",
                            "SECTOR": "섹터분석", "SECTOR_FAIL": "섹터(실패)",
                            "TP_MONOTONIC": "목표가 단조성",
                        }
                        for _ck, _ok in _checks.items():
                            _label = _check_labels.get(_ck, _ck)
                            _mark = "✅ 정상" if _ok else "❌ 실패"
                            _check_md += f"| {_label} | {_mark} |\n"
                        st.markdown(_check_md)

                    # JSON 다운로드
                    st.download_button(
                        "📥 run_health.json 다운로드",
                        data=_json.dumps(_health_data, ensure_ascii=False, indent=2),
                        file_name=os.path.basename(_health_files[0]),
                        mime="application/json",
                    )
            except Exception:
                pass
user = get_user()
user_role = (user or {}).get("role", "guest")

# ═══════════════════════════════════════════════════
# [v22 UI Phase 2] 탭 구조 개선
#   - tab1 라벨: "📊 시장 (Market)" → "🏠 오늘의 추천"
#   - tab8 (회원관리): admin에게만 라벨 노출
# 권한 로직(auth_status/PRIME/결제)은 안 건드림 — 사이드바 그대로 유지
# ═══════════════════════════════════════════════════
is_admin = (auth_status == "admin")

_tab_labels = [
    "🏠 오늘의 추천",
    "🔭 종목 분석",
    "💼 내 자산",
    "📮 문의 게시판",
    "⚖️ 이용 약관 / 투자 유의사항",
    "🧩 LDY Pro Trader 업데이트 노트",
    "📈 시스템 성과 (Performance)",
]

if is_admin:
    _tab_labels.append("👑 회원관리")

_tabs = st.tabs(_tab_labels)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = _tabs[:7]
tab8 = _tabs[7] if is_admin else None

# ── [v20.0.1] 탭/종목 상태 자동 복원 (모바일 탭 전환 대응) ──
st.markdown("""
<script>
(function() {
    // ── 탭 클릭 시 localStorage에 저장 ──
    function setupTabTracking() {
        const tabs = document.querySelectorAll('[data-baseweb="tab"]');
        if (!tabs.length) { setTimeout(setupTabTracking, 500); return; }
        
        tabs.forEach(function(tab, idx) {
            tab.addEventListener('click', function() {
                localStorage.setItem('ldy_active_tab', idx);
            });
        });
        
        // ── 페이지 로드 시 저장된 탭으로 복원 ──
        const saved = localStorage.getItem('ldy_active_tab');
        if (saved !== null) {
            const tabIdx = parseInt(saved);
            if (tabIdx > 0 && tabIdx < tabs.length) {
                // 약간의 딜레이 후 클릭 (Streamlit 렌더링 완료 대기)
                setTimeout(function() { tabs[tabIdx].click(); }, 300);
            }
        }
    }
    
    // ── 종목 선택 시 localStorage에 저장 ──
    function setupStockTracking() {
        const observer = new MutationObserver(function() {
            const selectbox = document.querySelector('[data-testid="stSelectbox"]');
            if (selectbox) {
                const span = selectbox.querySelector('[data-baseweb="select"] span');
                if (span && span.textContent) {
                    const current = span.textContent.trim();
                    const prev = localStorage.getItem('ldy_selected_stock');
                    if (current && current !== prev && current !== '분석할 종목을 선택하세요') {
                        localStorage.setItem('ldy_selected_stock', current);
                    }
                }
            }
        });
        observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    }
    
    // ── 초기화 ──
    if (document.readyState === 'complete') {
        setupTabTracking();
        setupStockTracking();
    } else {
        window.addEventListener('load', function() {
            setupTabTracking();
            setupStockTracking();
        });
    }
})();
</script>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# [v22 UI] 헬퍼 함수 — tab1/tab2에서 사용
# ═══════════════════════════════════════════════════
def _to_num(x, default=np.nan):
    v = pd.to_numeric(x, errors="coerce")
    return default if pd.isna(v) else float(v)

def _to_int_str(x, default=0):
    v = pd.to_numeric(x, errors="coerce")
    return f"{int(v) if pd.notna(v) else default:,}"


# ═══════════════════════════════════════════════════
# [v22 UI] 오늘의 결론 카드 — 한눈에 보는 요약
# ═══════════════════════════════════════════════════
def render_today_summary_card(scored_df: pd.DataFrame):
    """대시보드 최상단에 '오늘의 결론' 카드 표시.
    
    3가지 시나리오:
      A) TOP_PICK >= 1   → 추천 카드 (큼직하게)
      B) TOP_PICK = 0, ARMED/ATTACK 있음 → 관찰 모드 + 가까운 후보
      C) TOP_PICK = 0, 활성 후보도 0    → 매수 신호 없음
    
    안전 설계:
      - try/except로 에러 시 카드 자체 미표시 (기존 화면 유지)
      - 컬럼 누락 시 graceful fallback
    """
    try:
        if scored_df is None or scored_df.empty:
            return
        
        # TOP_PICK 종목 추출 (다양한 표기 대응)
        top_picks = pd.DataFrame()
        if 'TOP_PICK' in scored_df.columns:
            tp_str = scored_df['TOP_PICK'].astype(str).str.strip().str.upper()
            tp_mask = tp_str.isin(['1', '1.0', 'TRUE', 'Y', 'YES'])
            top_picks = scored_df[tp_mask].copy()
        
        n_top = len(top_picks)
        
        # ── 시나리오 A: TOP_PICK 있음 ──
        if n_top >= 1:
            st.markdown("### 🏆 오늘의 추천")
            
            # 최대 3개까지만 표시
            top_picks = top_picks.head(3)
            cols = st.columns(len(top_picks))
            
            for i, (_, row) in enumerate(top_picks.iterrows()):
                with cols[i]:
                    name = row.get('종목명', 'N/A')
                    
                    # TOP_PICK_TYPE에 따라 이모지 분리
                    tp_type = str(row.get('TOP_PICK_TYPE', '')).upper()
                    if tp_type == 'AGGRESSIVE':
                        emoji = '🔥'
                        type_label = '공격형'
                    elif tp_type == 'STABLE':
                        emoji = '💎'
                        type_label = '안정형'
                    else:
                        emoji = '⭐'
                        type_label = '추천'
                    
                    st.markdown(f"#### {emoji} {name}")
                    st.caption(f"{type_label} · ELITE {_to_num(row.get('ELITE_SCORE'), 0):.1f}")
                    
                    # 가격 정보
                    buy = _to_num(row.get('추천매수가'))
                    tp1 = _to_num(row.get('추천매도가1'))
                    stop = _to_num(row.get('손절가'))
                    
                    if pd.notna(buy) and buy > 0:
                        st.markdown(f"**매수**: {int(buy):,}원")
                        if pd.notna(tp1) and tp1 > 0:
                            ret = (tp1 / buy - 1) * 100
                            st.markdown(f"**목표**: {int(tp1):,}원  `+{ret:.1f}%`")
                        if pd.notna(stop) and stop > 0:
                            loss = (stop / buy - 1) * 100
                            st.markdown(f"**손절**: {int(stop):,}원  `{loss:.1f}%`")
                    
                    # 추천 비중
                    amt = _to_num(row.get('추천금액(만원)'), 0)
                    if amt > 0:
                        st.markdown(f"**비중**: {amt:.0f}만원")
                    
                    # 승률
                    ewr = _to_num(row.get('EST_WIN_RATE'))
                    if pd.notna(ewr):
                        st.caption(f"예상 승률 {ewr:.0%}")
            
            st.divider()
            return
        
        # ── 시나리오 B/C: TOP_PICK 0건 ──
        # 활성 ROUTE 후보 찾기
        active = pd.DataFrame()
        if 'ROUTE' in scored_df.columns:
            active = scored_df[
                scored_df['ROUTE'].astype(str).str.strip().str.upper().isin(['ATTACK', 'ARMED'])
            ].copy()
        
        if len(active) > 0 and 'ELITE_SCORE' in active.columns:
            # ── 시나리오 B: 관찰 모드 ──
            top_candidate = active.sort_values('ELITE_SCORE', ascending=False).iloc[0]
            cand_name = top_candidate.get('종목명', 'N/A')
            cand_score = _to_num(top_candidate.get('ELITE_SCORE'), 0)
            cand_route = top_candidate.get('ROUTE', '')
            cand_tp1 = _to_num(top_candidate.get('TP1_PCT'), 0)
            
            # 부족한 점수 진단
            shortfall = ""
            struct = _to_num(top_candidate.get('STRUCT_SCORE'), 0)
            timing = _to_num(top_candidate.get('TIMING_SCORE'), 0)
            balance = _to_num(top_candidate.get('BALANCE_SCORE'), 0)
            if cand_score < 75 and cand_score >= 70:
                shortfall = f"ELITE {75 - cand_score:.1f}점 부족 (75 이상 필요)"
            elif struct > 0 and struct < 80:
                shortfall = f"STRUCT {80 - struct:.1f}점 부족 (80 이상 필요)"
            elif timing > 0 and timing < 70:
                shortfall = f"TIMING {70 - timing:.1f}점 부족"
            else:
                shortfall = "조건 일부 미달"
            
            st.warning(f"""
            ⏸️ **오늘은 관찰 모드** — 정식 추천 0건
            
            **가장 가까운 후보**: {cand_name} (점수 {cand_score:.1f}, {cand_route})
            
            └ {shortfall}, 목표수익 +{cand_tp1:.1f}%
            
            > 시스템이 신중하게 골라서 오늘은 통과한 종목이 없어요. 무리한 진입은 자제하시고 다음 기회를 기다리세요.
            """)
            
            with st.expander(f"📋 활성 후보 {len(active)}종목 더 보기"):
                show_cols = [c for c in ['종목명','ROUTE','ELITE_SCORE','TP1_PCT',
                                          'EST_WIN_RATE','추천금액(만원)']
                             if c in active.columns]
                top10 = active.sort_values('ELITE_SCORE', ascending=False).head(10)
                st.dataframe(top10[show_cols], use_container_width=True, hide_index=True)
        else:
            # ── 시나리오 C: 활성 후보 0 ──
            st.error("""
            🔴 **오늘은 매수 신호 없음** — 시장 약세
            
            ATTACK/ARMED 종목이 없어 신규 진입을 권하지 않습니다. 다음 거래일을 기다려주세요.
            """)
        
        st.divider()
    
    except Exception as e:
        # 카드 렌더링 실패해도 기존 화면은 유지
        # (디버그용 — 운영에선 silent fail)
        try:
            st.caption(f"_(오늘의 요약 카드 일시 비표시)_")
        except Exception:
            pass


with tab1:
    # ═══════════════════════════════════════════════════
    # [v22 UI Phase 2] 첫 화면 Hero 카드 — 한눈에 결론
    # ═══════════════════════════════════════════════════
    render_today_summary_card(scored)
    
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

    # ═══════════════════════════════════════════════════
    # [v22 UI Phase 2] 상세 정보는 expander로 접기 — 첫 화면 깔끔하게
    # 펼치면: 글로벌 매크로 / 엔진 상태 / 공포탐욕 / 섹터맵 / 모멘텀 / 지표 분석
    # ═══════════════════════════════════════════════════
    with st.expander("📊 시장 상세 분석 보기 (글로벌 매크로 · 섹터 · 지표 승률)", expanded=False):

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



with tab2:
    # [v22 UI] 오늘의 결론 카드 — 최상단 한눈에 보기
    render_today_summary_card(scored)
    
    st.subheader("🎯 AI & Quant 추천 종목")

    # ✅ [v8.5] 회원가입 직후 Top 5 프리뷰
    if just_registered:
        st.success("🎉 첫 가입을 환영합니다! 오늘 기준 TOP 5 프리뷰를 먼저 보여드릴게요.")
        try:
            preview = make_preview(base, n=5)
        except Exception:
            preview = make_preview(scored, n=5)

        if not preview.empty:
            cols_prev = ["종목명", "종목코드", "LDY_SCORE", "추천매수가", "손절가", "추천매도가1"]
            cols_prev = [c for c in cols_prev if c in preview.columns]

            prev_view = preview[cols_prev].copy()
            fmt_cols = ["추천매수가", "손절가", "추천매도가1"]
            for c in fmt_cols:
                if c in prev_view.columns:
                    prev_view[c] = pd.to_numeric(prev_view[c], errors='coerce').fillna(0).apply(lambda x: f"{int(x):,}")

            # 🛠️ 종목명 복구
            try:
                name_map = get_code_map()
                code_to_name = {v: k for k, v in name_map.items()}

                def _fix_prev_name(r):
                    nm = str(r.get("종목명", "")).strip()
                    cd = str(r.get("종목코드", "")).strip().zfill(6)
                    if not nm or nm.isdigit() or nm == cd:
                        return code_to_name.get(cd, nm)
                    return nm

                if "종목명" in prev_view.columns:
                    prev_view["종목명"] = prev_view.apply(_fix_prev_name, axis=1)
            except Exception:
                pass

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
            min_value=0, max_value=100, 
            value=0,  # 🔥 [수정] 기본값을 0으로 낮춤 (Active 종목이 점수 필터에 안 잘리게)
            step=1,
            key="min_score",
        )

    with col_f2:
        def _route_order(r: str):
            s = str(r)
            if "ATTACK" in s or "공략" in s: return (0, s) # ATTACK 우선
            if "ARMED" in s or "임박" in s: return (1, s)
            if "SQZ" in s: return (2, s)
            if "BRK" in s: return (3, s)
            if "Watch" in s or "관찰" in s or "상승" in s: return (4, s)
            return (5, s)

        all_routes = sorted(
            scored["ROUTE"].dropna().unique().tolist(),
            key=_route_order
        ) if "ROUTE" in scored.columns else []

        if all_routes:
            # 기본값: 전체 선택
            sel_routes = st.multiselect(
                "전략 유형 (ROUTE)",
                options=all_routes,
                default=[], # 기본은 전체 보기
                placeholder="전체 보기 (선택 시 필터링)",
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
                default=[],
                placeholder="전체 보기 (선택 시 필터링)",
                key="regime_filter",
            )
        else:
            sel_regimes = []

    use_only_gate = st.checkbox("EBS/유동성 통과만 사용", value=True, key="only_gate")

    # ---------------------------
    # 데이터 필터링 로직
    # ---------------------------
    # 1. 대상 데이터셋 선정
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

    # 2. 점수 필터링 (사용자가 0보다 크게 설정했을 때만 적용)
    if min_score > 0:
        filter_col = "FINAL_SCORE" if "FINAL_SCORE" in filtered.columns else (
            "TOTAL_SCORE" if "TOTAL_SCORE" in filtered.columns else (
                "RANK_SCORE" if "RANK_SCORE" in filtered.columns else "LDY_SCORE"
            )
        )
        if filter_col in filtered.columns:
            filtered[filter_col] = pd.to_numeric(filtered[filter_col], errors='coerce').fillna(0)
            filtered = filtered[filtered[filter_col] >= min_score]

    # 3. 라우트/리짐 필터링
    if sel_routes and "ROUTE" in filtered.columns:
        filtered = filtered[filtered["ROUTE"].isin(sel_routes)]
    if sel_regimes and "REGIME" in filtered.columns:
        filtered = filtered[filtered["REGIME"].isin(sel_regimes)]

    # 4. 추가 체크박스 필터
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

    # 5. 정렬 (Active 우선 로직 적용을 위해 여기서 미리 정렬하지 않음, 데이터 준비만)
    # (나중에 active/passive 나눌 때 정렬함)

    # 권한별 노출 개수 제한 메시지
    if auth_status in ["pro", "prime", "admin"]:
        limit = 20 if auth_status == "pro" else 100
        st.success(f"🥇 {auth_status.upper()} 회원: AI 종합 랭킹 Top {limit} 열람 중")
    else:
        limit = 5 if user else 3
        user_type = "Free" if user else "Guest"
        st.info(f"✅ {user_type} 회원: 상위 {limit}개 열람 중 (Pro/Prime 업그레이드 시 더 많은 종목 확인 가능)")
    

    full_df = augment_display_data(filtered.copy())
    
    # 변수 사전 초기화 (필터링 결과가 0개일 때 NameError 방지)
    active_df = pd.DataFrame()
    passive_df = pd.DataFrame()
    active_view = pd.DataFrame()
    passive_view = pd.DataFrame()

    # 사용자 정의 컬럼 리스트 및 디자인 설정 (cfg)
    cols = [
        "상태", "종목명", "생존일",
        "FINAL_SCORE",   # [1] 최종 판단
        "TOTAL_SCORE",   # [2] 기초 체력 (구조)
        "TRIGGER_SCORE", # [3] 단기 맥점 (타이밍)
        "추세", "종가", "추천매수가", "손절가", "추천매도가1", "업종"
    ]
    
    cfg = {
        "상태": st.column_config.TextColumn("Action", width="medium"),
        "종목명": st.column_config.TextColumn("종목명", width="medium", pinned=True),
        "생존일": st.column_config.ProgressColumn("Time", format="%d일", min_value=0, max_value=12),
        "FINAL_SCORE": st.column_config.ProgressColumn(
            "🥇최종순위", format="%.1f", min_value=0, max_value=100, width="small",
            help="구조(체력)와 타이밍(맥점)을 시장상황에 맞춰 가중 합산한 최종 점수"
        ),
        "TOTAL_SCORE": st.column_config.NumberColumn(
            "⚙️구조(체력)", format="%.0f", width="small", 
            help="차트/수급/AI가 판단한 종목의 기초 체력 (높을수록 튼튼함)"
        ),
        "TRIGGER_SCORE": st.column_config.NumberColumn(
            "🔥타이밍(맥)", format="%.0f", width="small", 
            help="단기 급등 임박 신호 (높을수록 당장 쏠 확률 높음)"
        ),
        "추세": st.column_config.TextColumn("Trend", width="small"),
        "종가": st.column_config.TextColumn("현재가", width="small"),
        "추천매수가": st.column_config.TextColumn("매수", width="small"),
        "손절가": st.column_config.TextColumn("손절", width="small"),
        "추천매도가1": st.column_config.TextColumn("T1목표", width="small"),
        "제외사유": st.column_config.TextColumn("제외사유", width="small")
    }

    if not full_df.empty:
        if "IS_ACTIVE" in full_df.columns:
            active_df  = full_df[full_df["IS_ACTIVE"] == True].copy()
            passive_df = full_df[full_df["IS_ACTIVE"] == False].copy()
        else:
            active_mask = full_df["상태"].astype(str).str.contains(r"🚀|🔫|👀|⭐️|🔋|🆕", na=False)
            active_df  = full_df[active_mask].copy()
            passive_df = full_df[~active_mask].copy()

        sort_mode = st.radio("정렬 기준", ["🚦 상태 우선 (행동순)", "🔢 점수 우선 (능력순)"], horizontal=True, label_visibility="collapsed", key="tab2_sort_mode")
        
        def _apply_sort(df_target):
            if df_target.empty: return df_target
            if sort_mode == "🚦 상태 우선 (행동순)" and "_STATE_SORT" in df_target.columns:
                return df_target.sort_values(by=["_STATE_SORT", "FINAL_SCORE", "TRIGGER_SCORE", "거래대금(억원)"], ascending=[True, False, False, False])
            else:
                s_col = "FINAL_SCORE" if "FINAL_SCORE" in df_target.columns else "TOTAL_SCORE"
                return df_target.sort_values(by=[s_col, "거래대금(억원)"], ascending=[False, False])

        active_df = _apply_sort(active_df)
        passive_df = _apply_sort(passive_df)
        active_view = active_df.head(limit).copy()
        passive_view = passive_df.copy()

        try:
            name_map = get_code_map(); code_to_name = {v: k for k, v in name_map.items()}
            def _fmt_display(df_target):
                if df_target.empty: return df_target
                if "종목명" in df_target.columns:
                    df_target["종목명"] = df_target.apply(
                        lambda r: code_to_name.get(str(r.get("종목코드","")).zfill(6), r.get("종목명")), axis=1
                    )
                for c in ["종가", "추천매수가", "손절가", "추천매도가1", "거래대금(억원)"]:
                    if c in df_target.columns:
                        df_target[c] = pd.to_numeric(df_target[c], errors="coerce").fillna(0).apply(lambda x: f"{int(x):,}")
                return df_target
            active_view = _fmt_display(active_view)
            passive_view = _fmt_display(passive_view)
        except Exception: pass

    if full_df.empty:
        st.warning("🧐 현재 모든 필터 조건을 만족하는 종목이 없습니다. 필터를 한두 개 해제해 보세요.")
    else:
        st.markdown("### 🔭 한눈에 보는 시장 지도")
        top3 = active_df.head(3)
        if not top3.empty:
            c_h = st.columns(3)
            for i, (idx, row) in enumerate(top3.iterrows()):
                with c_h[i]:
                    st.container(border=True).metric(f"🥇 Top {i+1}. {row['종목명']}", f"{row['FINAL_SCORE']}점", f"Trigger {row['TRIGGER_SCORE']}점")

        try:
            chart_data = pd.concat([active_df, passive_df.head(20)])
            if not chart_data.empty:
                fig_map = plot_opportunity_map(chart_data)
                if fig_map: st.plotly_chart(fig_map, use_container_width=True)
        except Exception: pass

        st.divider()
        st.markdown(f"### 🔥 집중 공략 후보 ({len(active_view)}개)")
        view_type = st.radio("보기 방식", ["📋 리스트", "🃏 칸반"], horizontal=True, label_visibility="collapsed", key="v_type_final")
        
        if view_type == "📋 리스트":
            st.dataframe(active_view[[c for c in cols if c in active_view.columns]], use_container_width=True, column_config=cfg, hide_index=True, height=500)
        else:
            render_kanban_board(active_view)

        if not passive_view.empty:
            with st.expander(f"💤 보류/제외 종목 ({len(passive_view)}개)"):
                st.dataframe(passive_view[[c for c in (cols+["제외사유"]) if c in passive_view.columns]], use_container_width=True, column_config=cfg, hide_index=True)

    st.divider()
    st.markdown("### 🔍 상세 정밀 분석 (Deep Dive)")
    
    target_list = []
    if not active_view.empty:
        target_list = active_view["종목명"].tolist()
    elif not passive_view.empty:
        target_list = passive_view["종목명"].tolist()
    
    if not target_list:
        st.info("💡 분석할 종목이 없습니다.")
    else:
        # [v20.0.1] 종목 선택 복원 — 탭 전환 후 돌아와도 유지
        _saved_stock = st.query_params.get("stock", "")
        _default_idx = 0
        if _saved_stock and _saved_stock in target_list:
            _default_idx = target_list.index(_saved_stock)

        selected_name = st.selectbox("분석할 종목을 선택하세요", target_list,
                                      index=_default_idx, key="dd_select_final")

        # 선택 변경 시 URL에 저장 (새로고침 후에도 복원)
        if selected_name and selected_name != _saved_stock:
            st.query_params["stock"] = selected_name
        sel_row = None
        if not active_df.empty:
            found = active_df[active_df["종목명"] == selected_name]
            if not found.empty: sel_row = found.iloc[0]
        if sel_row is None and not passive_df.empty:
            found = passive_df[passive_df["종목명"] == selected_name]
            if not found.empty: sel_row = found.iloc[0]

        if sel_row is not None:
            d1, d2 = st.columns([1.2, 1])
            with d1:
                code = str(sel_row['종목코드']).zfill(6)
                df_chart = get_stock_chart_data(code)
                if df_chart is not None:
                    fig_candle = plot_interactive_chart(df_chart, code, selected_name, entry=sel_row.get('추천매수가'), stop=sel_row.get('손절가'), target1=sel_row.get('추천매도가1'), target2=sel_row.get('추천매도가2'), target_atr=sel_row.get('TARGET_ATR'), show_vp=True)
                    st.plotly_chart(fig_candle, use_container_width=True)
            with d2:
                try:
                    fig_water = plot_score_waterfall(sel_row)
                    st.plotly_chart(fig_water, use_container_width=True)
                except Exception: pass
                st.plotly_chart(plot_radar_chart(sel_row), use_container_width=True)

            route_val = sel_row.get("ROUTE", "NEUTRAL")
            badge_color = get_route_color(route_val)
            st.markdown(f"""
                <div class="route-badge" style="background:{badge_color};">
                    <div>
                        <span class="label">현재 작전 상태</span>
                        <span class="value">{route_val}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # ──────────────────────────────────────────
            # [v12.0] 과학적 목표가 분석 카드
            # ──────────────────────────────────────────
            _close = nz_num(sel_row.get("종가", 0))
            _entry = nz_num(sel_row.get("추천매수가", 0))
            _stop  = nz_num(sel_row.get("손절가", 0))
            _t1    = nz_num(sel_row.get("추천매도가1", 0))
            _t2    = nz_num(sel_row.get("추천매도가2", 0))
            _t_atr = nz_num(sel_row.get("TARGET_ATR", 0))

            if _close > 0 and _entry > 0 and _stop > 0 and _t1 > 0:
                st.markdown("##### 🎯 과학적 목표가 분석 (Multi-Method Cluster)")

                # 손익비 계산
                risk = _entry - _stop
                rr_t1 = (_t1 - _entry) / risk if risk > 0 else 0
                rr_t2 = (_t2 - _entry) / risk if risk > 0 and _t2 > 0 else 0

                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("🔴 손절가", f"{int(_stop):,}", 
                          delta=f"{(_stop/_close - 1)*100:+.1f}%", delta_color="inverse")
                tc2.metric("🟢 T1 목표 (보수적)", f"{int(_t1):,}", 
                          delta=f"+{(_t1/_close - 1)*100:.1f}%  (RR {rr_t1:.1f}:1)")
                if _t2 > 0 and _t2 != _t1:
                    tc3.metric("🟡 T2 목표 (공격적)", f"{int(_t2):,}", 
                              delta=f"+{(_t2/_close - 1)*100:.1f}%  (RR {rr_t2:.1f}:1)")
                else:
                    tc3.metric("🟡 T2 목표", "—", delta="T1과 동일")
                if _t_atr > 0 and _t_atr != _t1:
                    tc4.metric("⚪ ATR 참고", f"{int(_t_atr):,}", 
                              delta=f"+{(_t_atr/_close - 1)*100:.1f}%")
                else:
                    tc4.metric("⚪ ATR 참고", "—")

                # 시각적 가격 바 (손절 ~ ATR 범위를 시각화)
                bar_prices = [("손절", _stop, "#FF3B30"), ("현재가", _close, "#FFFFFF"), 
                             ("매수", _entry, "#2962FF")]
                if _t1 > 0: bar_prices.append(("T1", _t1, "#00E676"))
                if _t2 > 0 and _t2 != _t1: bar_prices.append(("T2", _t2, "#FFD600"))
                if _t_atr > 0 and _t_atr != _t1 and _t_atr != _t2: bar_prices.append(("ATR", _t_atr, "#888888"))
                bar_prices.sort(key=lambda x: x[1])

                price_min = bar_prices[0][1] * 0.98
                price_max = bar_prices[-1][1] * 1.02
                price_range = price_max - price_min
                if price_range > 0:
                    bar_html = '<div style="position:relative; height:80px; background:linear-gradient(90deg, rgba(255,59,48,0.15) 0%, rgba(255,59,48,0.05) 30%, rgba(0,230,118,0.05) 70%, rgba(0,230,118,0.15) 100%); border-radius:8px; margin:8px 0 40px 0; overflow:visible;">'
                    for label, price, color in bar_prices:
                        pct = (price - price_min) / price_range * 100
                        pct = max(3, min(pct, 97))
                        is_current = label == "현재가"
                        dot_size = "14px" if is_current else "10px"
                        z_idx = "10" if is_current else "5"
                        border = "2px solid #FFF" if is_current else "none"
                        bar_html += f'''<div style="position:absolute; left:{pct}%; top:35%; transform:translate(-50%,-50%); z-index:{z_idx};">
                            <div style="width:{dot_size}; height:{dot_size}; background:{color}; border-radius:50%; border:{border}; margin:0 auto;"></div>
                            <div style="font-size:10px; color:{color}; text-align:center; white-space:nowrap; margin-top:4px; font-weight:{"bold" if is_current else "normal"}; line-height:1.3;">{label}<br>{int(price):,}</div>
                        </div>'''
                    bar_html += '</div>'
                    st.markdown(bar_html, unsafe_allow_html=True)

                # 익절 전략 안내
                if risk > 0 and _t1 > 0:
                    with st.expander("📋 단계별 익절 전략 안내", expanded=False):
                        strategy_md = f"""
| 단계 | 가격 | 현재가 대비 | 행동 |
|:---:|:---:|:---:|:---|
| 🔴 손절 | {int(_stop):,}원 | {(_stop/_close - 1)*100:+.1f}% | 전량 매도 (비협상) |
| 🟢 **T1 도달** | **{int(_t1):,}원** | **+{(_t1/_close - 1)*100:.1f}%** | **40% 1차 익절** (클러스터 수렴 저항) |"""
                        if _t2 > 0 and _t2 != _t1:
                            strategy_md += f"""
| 🟡 **T2 도달** | **{int(_t2):,}원** | **+{(_t2/_close - 1)*100:.1f}%** | **30% 2차 익절** (상위 클러스터) |"""
                        strategy_md += f"""
| ⚪ 잔여 | T2 이상 | — | 나머지 30% 트레일링 스탑 (고점 -8~10%) |

> 💡 T1은 볼린저밴드·SuperTrend·피보나치 등 **다수 지표가 수렴하는 기술적 저항대**입니다.  
> T2는 피보나치 확장·섹터 상위 성과 등 **더 높은 목표** 클러스터입니다.
"""
                        st.markdown(strategy_md)

            st.markdown("---")
            st.subheader("🧱 매물대 및 저항 데이터 분석 (Volume Profile)")
            def _gv(k):
                v = sel_row.get(k, 0.0)
                try: return float(v)
                except (ValueError, TypeError): return 0.0
            res_all, res_near, poc_gap, near_thres = _gv("RES_RATIO"), _gv("RES_RATIO_NEAR"), _gv("POC_GAP"), _gv("NEAR_THRES")
            is_above_poc = int(sel_row.get("IS_ABOVE_POC", 0) or 0)
            m1, m2, m3 = st.columns(3)
            m1.metric("상단 전체 매물 비중", f"{res_all*100:.1f}%", "🔴 저항강함" if res_all > 0.4 else "🟡 보통" if res_all > 0.2 else "🟢 매물진공")
            m2.metric(f"근접 저항 (위 {near_thres:.1f}%)", f"{res_near*100:.1f}%", "⚠️ 저항" if res_near > 0.2 else "🚀 돌파기대", delta_color="inverse")
            m3.metric("POC 대비 위치", f"{poc_gap:+.1f}%", "안착성공" if is_above_poc == 1 else "돌파필요")

            if res_all < 0.15 and is_above_poc == 1:
                st.success(f"🎯 **[전략]** 현재 주요 매물대 위에 안착했으며, 상단 매물이 매우 희박한 '매물 진공' 구간입니다. 탄력적인 시세가 기대됩니다.")
            elif res_near > 0.30:
                st.warning(f"⚠️ **[전략]** 현재가 바로 위에 매물 벽이 두꺼움(30%+). 거래량을 동반한 돌파 확인이 필요합니다.")

    st.divider()
    with st.expander("🧩 Top 종목 상관관계 점검 (분산투자 확인용)"):
        corr_target = active_df if not active_df.empty else passive_df
        if not corr_target.empty:
            if st.button("🚀 상관관계 분석 실행", key="btn_run_corr_final"):
                with st.spinner("분석 중..."):
                    fig_corr = plot_correlation_heatmap(corr_target)
                    if fig_corr: st.plotly_chart(fig_corr, use_container_width=True)
        else: st.info("표시할 종목이 없습니다.")
    
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
        st.info("🔒 내 자산 분석 및 리밸런싱 제안은 **Pro 등급**부터 가능합니다.")
    else:
        st.subheader("💼 내 자산: AI 리밸런싱 & 진단")

        # 1. 데이터 로드 (Gist/Local)
        saved_str = load_portfolio_file()
        default_data = []
        
        if saved_str:
            try:
                lines = saved_str.strip().split("\n")
                for line in lines:
                    if ":" in line:
                        parts = line.split(":")
                        if len(parts) >= 3:
                            try:
                                nm = parts[0].strip()
                                p_val = int(float(parts[1].replace(",","").strip()))
                                q_val = int(float(parts[2].replace(",","").strip()))
                                default_data.append({"종목명": nm, "평단가": p_val, "수량": q_val, "비고": ""})
                            except (ValueError, TypeError, IndexError): pass
            except Exception: pass
        
        if not default_data:
            default_data = [{"종목명": "", "평단가": 0, "수량": 0, "비고": ""}]

        # 2. 데이터 에디터
        st.caption("👇 현재 보유 중인 종목을 입력하세요. AI가 점수를 분석해 조언을 드립니다.")
        edited_df = st.data_editor(
            pd.DataFrame(default_data),
            num_rows="dynamic",
            use_container_width=True,
            key="portfolio_editor",
            column_config={
                "종목명": st.column_config.TextColumn(required=True),
                "평단가": st.column_config.NumberColumn(format="%d원", min_value=0, required=True),
                "수량": st.column_config.NumberColumn(format="%d주", min_value=0, required=True),
            }
        )

        # 3. 데이터 저장 및 분석 대상 추출
        targets = []
        cash_amt = 0.0
        save_lines = []
        code_map = get_code_map() if 'get_code_map' in globals() else {}

        if edited_df is not None and not edited_df.empty:
            for _, row in edited_df.iterrows():
                nm = str(row.get("종목명", "")).strip()
                if not nm: continue
                try:
                    price = float(row.get("평단가", 0))
                    qty = int(row.get("수량", 0))
                except (ValueError, TypeError): continue

                save_lines.append(f"{nm}:{int(price)}:{int(qty)}")

                if nm.upper() == "CASH" or "현금" in nm:
                    cash_amt += price * qty
                else:
                    real_code = find_code_by_name(nm, code_map) or nm
                    targets.append((real_code, nm, price, qty))
            
            new_save_str = "\n".join(save_lines)
            if new_save_str != saved_str:
                save_portfolio_file(new_save_str)

        # 4. AI 진단 로직 (현재가 조회 + 점수 매핑)
        if not targets and cash_amt <= 0:
             st.info("👆 보유 종목을 입력하면 AI가 진단을 시작합니다.")
        else:
            # 현재가 조회 (병렬)
            price_map = {}
            with st.spinner('⚡ 보유 종목 시세 및 AI 점수 조회 중...'):
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(fetch_current_price, t[0], t[1]) for t in targets]
                    for future in futures:
                        c, n, p = future.result()
                        price_map[c] = p
            
            total_eval = 0.0
            total_buy = 0.0
            pf_rows = []

            # 점수 데이터(scored)가 있다면 활용
            scored_df = scored if 'scored' in globals() else pd.DataFrame()

            for code, name, avg, qty in targets:
                curr = price_map.get(code, 0)
                
                # 종목명 보정
                real_name = name
                if PYKRX_OK and curr > 0 and str(code).isdigit():
                    try:
                        kn = stock.get_market_ticker_name(code)
                        if kn: real_name = kn
                    except Exception: pass

                eval_amt = curr * qty
                buy_amt = avg * qty
                total_eval += eval_amt
                total_buy += buy_amt
                
                pct = (curr - avg) / avg * 100 if avg > 0 and curr > 0 else 0.0
                
                # 🔥 [핵심] AI 점수 매핑 & 조언 생성
                ai_score = 0
                rank_score = 0
                advice = "관망"
                advice_color = "gray"
                
                if not scored_df.empty:
                    # 코드 기준 검색
                    match = scored_df[scored_df['종목코드'] == str(code).zfill(6)]
                    if not match.empty:
                        ai_score = float(match.iloc[0].get('ML_SCORE', 0))
                        rank_score = float(match.iloc[0].get('RANK_SCORE', 0))
                
                # 점수가 0이면 이름으로 한 번 더 검색 (Fallback)
                if ai_score == 0 and not scored_df.empty:
                     match_name = scored_df[scored_df['종목명'] == real_name]
                     if not match_name.empty:
                        ai_score = float(match_name.iloc[0].get('ML_SCORE', 0))
                        rank_score = float(match_name.iloc[0].get('RANK_SCORE', 0))

                # 종합 점수 (AI 60% + 퀀트 40%)
                final_s = 0
                if ai_score > 0:
                    final_s = (ai_score * 0.6) + (rank_score * 0.4)
                else:
                    final_s = rank_score # AI 점수 없으면 퀀트 점수만

                # 조언 로직
                if final_s >= 80:
                    advice = "💪강력홀딩/추매"
                    advice_color = "green"
                elif final_s >= 60:
                    advice = "👌보유(양호)"
                    advice_color = "blue"
                elif final_s == 0:
                    advice = "❓정보없음"
                    advice_color = "gray"
                elif final_s <= 40:
                    advice = "⚠️교체권장/매도"
                    advice_color = "red"
                else:
                    advice = "👀관망(중립)"
                    advice_color = "orange"

                pf_rows.append({
                    "종목명": real_name,
                    "현재가": curr,
                    "평단가": avg,
                    "수량": qty,
                    "매입금": buy_amt,
                    "평가금": eval_amt,
                    "평가손익": eval_amt - buy_amt,
                    "수익률": pct,
                    "점수": final_s,
                    "AI조언": advice,
                    "Color": advice_color,
                    "Code": code
                })

            # 5. 포트폴리오 카드 UI
            st.divider()
            
            # 전체 자산 요약
            total_asset = total_eval + cash_amt
            total_invest = total_buy + cash_amt
            total_rate = (total_asset - total_invest) / total_invest * 100 if total_invest > 0 else 0
            total_pnl = total_asset - total_invest
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("총 평가금액", f"{int(total_asset):,}원")
            m2.metric("총 매입금액", f"{int(total_invest):,}원")
            pnl_delta = f"{int(total_pnl):+,}원"
            m3.metric("총 평가손익", pnl_delta, delta=f"{total_rate:+.2f}%",
                       delta_color="normal" if total_pnl >= 0 else "inverse")
            if cash_amt > 0:
                cash_pct = cash_amt / total_asset * 100 if total_asset > 0 else 0
                m4.metric("현금 비중", f"{cash_pct:.1f}%", delta=f"{int(cash_amt):,}원")
            else:
                m4.metric("종목 수", f"{len(pf_rows)}개")

            # ── 종목별 손익 요약 테이블 ──
            if pf_rows:
                summary_df = pd.DataFrame(pf_rows)
                # 비중 계산
                summary_df["비중(%)"] = (summary_df["평가금"] / total_asset * 100).round(1) if total_asset > 0 else 0.0

                display_cols = ["종목명", "현재가", "평단가", "수량", "매입금", "평가금", "평가손익", "수익률", "비중(%)", "점수", "AI조언"]
                show_df = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()
                
                # 포맷팅
                for c in ["현재가", "평단가", "매입금", "평가금", "평가손익"]:
                    if c in show_df.columns:
                        show_df[c] = show_df[c].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "-")
                if "수익률" in show_df.columns:
                    show_df["수익률"] = show_df["수익률"].apply(lambda x: f"{x:+.2f}%")

                with st.expander("📋 종목별 손익 상세", expanded=True):
                    st.dataframe(show_df, use_container_width=True, hide_index=True)
                
                # ── 집중도 리스크 경고 ──
                if total_asset > 0:
                    max_weight = summary_df["비중(%)"].max()
                    max_stock = summary_df.loc[summary_df["비중(%)"].idxmax(), "종목명"]
                    if max_weight >= 40:
                        st.warning(f"⚠️ **집중 리스크**: {max_stock}이(가) 포트폴리오의 {max_weight:.1f}%를 차지합니다. 분산 투자를 권장합니다.")
                    
                    # 손실 종목 경고
                    loss_stocks = summary_df[summary_df["수익률"] < -10]
                    if not loss_stocks.empty:
                        loss_names = ", ".join(loss_stocks["종목명"].tolist())
                        st.error(f"🔴 **손절 점검 필요**: {loss_names} (수익률 -10% 이하)")

            st.markdown("##### 🩺 AI 포트폴리오 진단 결과")
            
            if not pf_rows:
                st.info("보유 종목 정보가 없습니다.")
            else:
                # 점수 낮은 순으로 정렬 (위험 종목 먼저 보기)
                pf_rows.sort(key=lambda x: x["점수"])

                for row in pf_rows:
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([2, 1.5, 2])
                        
                        # 왼쪽: 종목 정보
                        with c1:
                            st.markdown(f"**{row['종목명']}**")
                            p_color = "red" if row['수익률'] > 0 else "blue" # 국내장 컬러 (빨강=상승)
                            st.markdown(f"<span style='color:{p_color}; font-weight:bold;'>{row['수익률']:+.2f}%</span>", unsafe_allow_html=True)
                            st.caption(f"평가금: {int(row['평가금']):,}원")

                        # 가운데: AI 점수
                        with c2:
                            st.markdown("🤖 **AI Score**")
                            if row['점수'] > 0:
                                st.progress(min(row['점수'] / 100, 1.0))
                                st.caption(f"{row['점수']:.1f}점")
                            else:
                                st.caption("분석불가 (데이터부족)")

                        # 오른쪽: 조언 및 교체 제안
                        with c3:
                            st.markdown(f"**AI 의견:** :{row['Color']}[{row['AI조언']}]")
                            
                            # 교체 매매 제안 (점수가 40점 미만인 경우)
                            if row['점수'] > 0 and row['점수'] <= 40:
                                # Top 20 중 같은 업종이거나 점수 높은 종목 1개 추천
                                rec_stock = None
                                if not scored_df.empty:
                                    # 1순위: 같은 업종 내 1등
                                    # (업종 정보가 있으면 좋겠지만 없으면 전체 1등 추천)
                                    rec_stock = scored_df.iloc[0] 
                                    
                                if rec_stock is not None:
                                    rec_name = rec_stock.get('종목명', '추천주')
                                    rec_score = rec_stock.get('TOTAL_SCORE', 0)
                                    st.info(f"💡 **교체 추천:** {rec_name} ({rec_score:.0f}점)")
                            
                            elif row['점수'] >= 80:
                                st.success("🚀 추세가 강력합니다. 수익 극대화 구간!")

            # 6. 현금 비중 차트
            if cash_amt > 0:
                pf_rows.append({"종목명": "현금 (CASH)", "평가금": cash_amt})
            
            df_pie = pd.DataFrame(pf_rows)
            if not df_pie.empty:
                fig = px.pie(df_pie, values="평가금", names="종목명", title="📊 자산 구성", hole=0.4)
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
        # 1. 빈 칸 체크
        if not title.strip() or not content.strip():
            st.error("제목과 내용을 모두 입력해 주세요.")
        else:
            # 2. [핵심] 세션별 고유 제출 토큰 확인
            # 현재 입력된 내용을 조합해 '지문(Fingerprint)'을 만듭니다.
            submission_id = f"{nickname}_{title}_{content[:20]}"
            
            if st.session_state.get("last_sub_id") == submission_id:
                # 이미 처리된 토큰이면 조용히 무시하거나 안내만 함
                st.warning("이미 처리 중이거나 완료된 문의입니다.")
            else:
                with st.spinner("💌 전령이 성벽으로 달리고 있습니다..."):
                    # Gist에서 최신 목록 다시 로드 (충돌 방지)
                    current_items = load_inquiry_items()

                    # 3. 데이터 중복 검증 (장부 내 마지막 데이터와 비교)
                    is_really_duplicate = False
                    if current_items:
                        last_item = current_items[-1]
                        if (last_item.get('title') == title.strip() and 
                            last_item.get('content') == content.strip()):
                            is_really_duplicate = True

                    if is_really_duplicate:
                        st.info("이미 장부에 기록된 내용입니다.")
                        st.session_state["last_sub_id"] = submission_id # 토큰 갱신
                    else:
                        # 4. 새 데이터 등록
                        new_item = {
                            "title": title.strip(),
                            "content": content.strip(),
                            "nickname": nickname.strip() or "익명",
                            "email": email.strip(),
                            "created_at": _now_utc_str(),
                        }
                        
                        current_items.append(new_item)
                        ok = save_inquiry_items(current_items)

                        if ok:
                            # 5. [중요] 처리가 끝났음을 토큰으로 기록
                            st.session_state["last_sub_id"] = submission_id
                            st.success("문의가 성공적으로 등록되었습니다!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("통신 장애로 기록에 실패했습니다.")

    st.markdown("#### 📂 최근 문의 내역")

    # Gist에서 데이터를 불러옴
    inquiries = load_inquiry_items()

    if not inquiries:
        st.info("아직 등록된 문의가 없습니다.")
    else:
        # 최신순 정렬 (리스트 뒤집기)
        # enumerate를 사용하여 고유 key 생성
        for i, item in enumerate(reversed(inquiries[-50:])):
            box = st.container(border=True)
            with box:
                c_head, c_btn = st.columns([8, 1])
                
                with c_head:
                    st.markdown(f"**제목:** {item.get('title', '-')}")

                # 날짜 포맷팅
                date_str = item.get('created_at','-')
                if 'to_kst_str' in globals():
                    date_str_disp = to_kst_str(date_str)
                else:
                    date_str_disp = date_str

                meta = f"작성자: {item.get('nickname','익명')} · 작성일: {date_str_disp}"
                if item.get("email"):
                    meta += f" · 이메일: {item.get('email')}"
                
                with c_head:
                    st.caption(meta)
                    st.markdown(item.get("content", "").replace("\n", "  \n"))

                # 🔥 [추가된 기능] 관리자일 경우 삭제 버튼 표시
                with c_btn:
                    if auth_status == "admin":
                        # 버튼 키를 유니크하게 만들기 위해 인덱스와 날짜 조합
                        if st.button("🗑️", key=f"del_inq_{i}_{date_str}", help="글 삭제"):
                            # 삭제 로직: 원본 리스트에서 해당 created_at을 가진 항목 제거
                            new_list = [x for x in inquiries if x.get('created_at') != date_str]
                            save_inquiry_items(new_list)
                            st.toast("글이 삭제되었습니다.")
                            time.sleep(1)
                            st.rerun()

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
    version_info.show_recent_updates(limit=5)

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


# ---------------------------
# [Tab 7] 시스템 성과 히스토리 (System Performance Lab)
# ---------------------------
with tab7:
    st.subheader("📈 시스템 성과 추세 (System Performance Lab)")
    st.caption("과거 추천 종목들의 검증 데이터(Validation Summary)를 기반으로 시스템의 승률 변화를 추적합니다.")

    @st.cache_data(ttl=3600)
    def load_performance_history():
        # data 폴더의 모든 rank_validation_summary_*.csv 파일 로드
        pattern = os.path.join(DATA_DIR, "rank_validation_summary_*.csv")
        files = sorted(glob.glob(pattern))
        
        dfs = []
        for f in files:
            try:
                # 1. 파일명에서 날짜 추출 (예: rank_validation_summary_20251230.csv)
                base_name = os.path.basename(f)
                
                # 'latest' 파일은 중복되므로 제외
                if "latest" in base_name:
                    continue

                date_str = base_name.replace("rank_validation_summary_", "").replace(".csv", "")
                
                df = pd.read_csv(f)
                
                # 2. 날짜 컬럼 생성
                try:
                    dt = pd.to_datetime(date_str, format="%Y%m%d")
                    df['Date'] = dt
                except (ValueError, TypeError):
                    df['Date'] = date_str

                dfs.append(df)
            except Exception as e:
                pass
        
        if not dfs:
            return pd.DataFrame()
            
        # 모든 파일 합치기
        combined = pd.concat(dfs, ignore_index=True)
        
        # 날짜순 정렬
        if 'Date' in combined.columns:
            combined = combined.sort_values('Date')
            
        return combined

    # 데이터 로딩
    history_df = load_performance_history()

    if history_df.empty:
        st.info("📉 아직 축적된 성과 검증 데이터(history)가 충분하지 않습니다.")
    else:
        # 🛠️ [수정됨] 실제 컬럼명 확인 및 매핑
        # CSV 파일의 실제 컬럼: WIN_RATE_% , AVG_RET_%
        col_win = 'WIN_RATE_%'
        col_ret = 'AVG_RET_%'

        if col_win not in history_df.columns or col_ret not in history_df.columns:
            st.error(f"데이터 컬럼을 찾을 수 없습니다. (필요: {col_win}, {col_ret})")
            st.write("현재 컬럼 목록:", history_df.columns.tolist())
        else:
            # ---------------------------
            # 필터링 UI (복합 데이터 중 하나만 선택해서 추세 보기)
            # ---------------------------
            st.markdown("##### 🔍 지표 상세 필터")
            f_col1, f_col2, f_col3 = st.columns(3)
            
            # 1) 전략 선택 (Method)
            methods = sorted(history_df['METHOD'].unique()) if 'METHOD' in history_df.columns else []
            if methods:
                # 기본값으로 'RANK_SCORE' 우선 선택
                def_m = 'RANK_SCORE' if 'RANK_SCORE' in methods else methods[0]
                sel_method = f_col1.selectbox("스코어링 기준 (Method)", methods, index=methods.index(def_m))
            else:
                sel_method = None

            # 2) Top K 선택
            topks = sorted(history_df['TOPK'].unique()) if 'TOPK' in history_df.columns else []
            if topks:
                # 기본값 5
                def_k = 5 if 5 in topks else topks[0]
                sel_k = f_col2.selectbox("Top K (상위 N개)", topks, index=topks.index(def_k))
            else:
                sel_k = None

            # 3) 보유 기간 (H)
            holds = sorted(history_df['H(영업일)'].unique()) if 'H(영업일)' in history_df.columns else []
            if holds:
                # 기본값 5일
                def_h = 5 if 5 in holds else holds[0]
                sel_h = f_col3.selectbox("보유 기간 (H일)", holds, index=holds.index(def_h))
            else:
                sel_h = None
            
            # 필터링 적용
            chart_df = history_df.copy()
            if sel_method:
                chart_df = chart_df[chart_df['METHOD'] == sel_method]
            if sel_k:
                chart_df = chart_df[chart_df['TOPK'] == sel_k]
            if sel_h:
                chart_df = chart_df[chart_df['H(영업일)'] == sel_h]

            # ---------------------------
            # 차트 그리기
            # ---------------------------
            if chart_df.empty:
                st.warning("선택한 조건에 맞는 데이터가 없습니다.")
            else:
                # 날짜 기준 재정렬
                chart_df = chart_df.sort_values('Date')
                
                # 최근 30일/60일/전체 옵션
                # d_range = st.radio("조회 기간", ["최근 30건", "전체"], horizontal=True)
                # if d_range == "최근 30건":
                #     chart_df = chart_df.tail(30)
                chart_df = chart_df.tail(30) # 기본 30개만 보여주기

                # 이중축 차트
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 막대: 승률 (이미 % 단위임)
                fig.add_trace(
                    go.Bar(
                        x=chart_df['Date'], 
                        y=chart_df[col_win], 
                        name="승률(%)", 
                        marker_color='#FFA726', 
                        opacity=0.6,
                        text=chart_df[col_win].apply(lambda x: f"{x:.1f}%"),
                        textposition='auto'
                    ),
                    secondary_y=False
                )

                # 선: 평균 수익률
                fig.add_trace(
                    go.Scatter(
                        x=chart_df['Date'], 
                        y=chart_df[col_ret], 
                        name="평균수익률(%)", 
                        mode='lines+markers+text', 
                        line=dict(color='#29B6F6', width=3),
                        text=chart_df[col_ret].apply(lambda x: f"{x:.1f}%"),
                        textposition="top center"
                    ),
                    secondary_y=True
                )

                fig.update_layout(
                    title=dict(
                        text=f"<b>📊 {sel_method} (Top {sel_k}, {sel_h}일 보유) 성과 추이</b>",
                        font=dict(size=16)
                    ),
                    hovermode="x unified",
                    height=450,
                    legend=dict(orientation="h", y=1.1),
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                
                # 축 설정
                fig.update_yaxes(title_text="승률 (%)", range=[0, 100], secondary_y=False, showgrid=True)
                fig.update_yaxes(title_text="수익률 (%)", secondary_y=True, showgrid=False)

                st.plotly_chart(fig, use_container_width=True)

                # 요약 통계
                avg_win = chart_df[col_win].mean()
                avg_ret = chart_df[col_ret].mean()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("기간 평균 승률", f"{avg_win:.1f}%")
                m2.metric("기간 평균 수익률", f"{avg_ret:.2f}%")
                m3.caption(f"※ 최근 {len(chart_df)}건 기준")
                
                with st.expander("📄 상세 데이터 보기"):
                    # 보기 좋게 컬럼 정리
                    disp_cols = ['Date', 'METHOD', 'TOPK', 'H(영업일)', col_win, col_ret, 'TOTAL_N']
                    disp_df = chart_df[ [c for c in disp_cols if c in chart_df.columns] ].copy()
                    
                    # 날짜 포맷
                    if 'Date' in disp_df.columns:
                        disp_df['Date'] = disp_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x)
                        
                    st.dataframe(disp_df.sort_values('Date', ascending=False), use_container_width=True)

# [v22 UI Phase 2] admin이 아니면 tab8 = None이라 with 진입 차단 필수
if is_admin and tab8 is not None:
    with tab8:
        st.subheader("👑 회원 관리 (Admin)")
    
        # 1. 권한 체크 (보안 필수)
        if auth_status == "admin":
            users = list_users()
        
            # --- 상단 요약 통계 ---
            if users:
                total_users = len(users)
            
                # 간단한 가입자 통계 표시
                st.markdown(f"""
                <div style="background-color:rgba(0,0,0,0.05); padding:10px; border-radius:5px; margin-bottom:15px;">
                    👥 <b>총 가입자:</b> {total_users}명
                </div>
                """, unsafe_allow_html=True)
        
            # --- 메인 리스트 ---
            if not users:
                st.info("등록된 회원이 없습니다.")
            else:
                # 표시할 데이터 가공
                rows = []
                for u in users:
                    # 상태 메시지 꾸미기
                    is_banned = u.get("is_banned", False)
                    role = u.get("role", "free")
                
                    status_icon = "✅"
                    if is_banned: status_icon = "🚫차단"
                
                    rows.append({
                        "Email": u.get("login_id"),
                        "닉네임": u.get("nickname"),
                        "권한": role.upper(),
                        "상태": status_icon,
                        "가입일": to_kst_str(u.get("join_date")),
                        "최근접속": to_kst_str(u.get("last_login")),
                        "만료일": to_kst_str(u.get("prime_expire_date")).split(" ")[0] if u.get("prime_expire_date") else "-"
                    })
            
                df_users = pd.DataFrame(rows)
            
                # 최신 접속순 정렬
                if "최근접속" in df_users.columns:
                    df_users = df_users.sort_values("최근접속", ascending=False)

                # 테이블 출력
                st.dataframe(
                    df_users, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Email": st.column_config.TextColumn("이메일", width="medium"),
                        "최근접속": st.column_config.TextColumn("최근접속", width="small"),
                    }
                )
            
                st.divider()
            
                # --- [관리자 액션 패널] ---
                c_left, c_right = st.columns([1, 1])
            
                with c_left:
                    st.markdown("### 🛠️ 개별 회원 제어")
                    target_email = st.selectbox("회원 선택", df_users["Email"], key="adm_usr_sel")
                
                    c_act1, c_act2 = st.columns(2)
                    with c_act1:
                        new_role = st.selectbox("등급 변경", ["free", "pro", "prime", "admin"], key="adm_role_sel")
                        if st.button("등급 적용", use_container_width=True):
                            if update_user_role(target_email, new_role):
                                st.toast(f"✅ {target_email} -> {new_role} 변경 완료")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("변경 실패")
                            
                    with c_act2:
                        st.write("") # 간격
                        st.write("") 
                        if st.button("🚫 차단/해제", type="primary", use_container_width=True):
                            ok, msg = toggle_user_ban(target_email)
                            if ok:
                                st.toast(msg)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(msg)

                with c_right:
                    st.markdown("### 🎉 전체 이벤트")
                    st.caption("관리자를 제외한 모든 회원에게 체험권을 일괄 지급합니다.")
                
                    if st.button("🎁 전원 7일 Prime 무료 지급", type="primary", use_container_width=True):
                        ok, msg = grant_all_users_trial(days=7)
                        if ok:
                            st.balloons()
                            st.success(msg)
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(msg)

        else:
            st.error("🚫 관리자 권한이 필요합니다.")
