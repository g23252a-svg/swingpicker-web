# -*- coding: utf-8 -*-
"""
LDY Pro Trader v6.3 (Subscription Ready)
- 개선: 스코어링 로직을 Collector v6.4 수준으로 정교화
- 개선: 포트폴리오 분석 병렬 처리 (속도 향상)
- 개선: 차트에 거래량(Volume) 보조 지표 추가
- 개선: 데이터 로딩 상태 시각화 (st.status)
- 개선: 보안 설정 (st.secrets 우선 지원)
- 추가: Pro / Prime 유료 구독 + 1개월 만료일 관리
"""

import os, io, math, json, requests, logging
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------
# 로깅 설정
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ldy")

# ---------------------------
# 문의게시판 저장소 설정
# ---------------------------
INQUIRY_DB_PATH = os.path.join("data", "inquiries_db.json")

def load_inquiry_db():
    """문의글 DB 로드"""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(INQUIRY_DB_PATH):
        return {"inquiries": []}
    try:
        with open(INQUIRY_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
        if "inquiries" not in db:
            db["inquiries"] = []
        return db
    except Exception:
        return {"inquiries": []}

def save_inquiry_db(db):
    """문의글 DB 저장"""
    os.makedirs("data", exist_ok=True)
    with open(INQUIRY_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

# ---------------------------
# 구독/권한(만료일) 관리
# ---------------------------
SUBS_DB_PATH = os.path.join("data", "subscriptions_db.json")

def load_subs_db():
    """구독 DB 로드"""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(SUBS_DB_PATH):
        return {"subs": {}}
    try:
        with open(SUBS_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
        if "subs" not in db:
            db["subs"] = {}
        return db
    except Exception:
        return {"subs": {}}

def save_subs_db(db):
    """구독 DB 저장"""
    os.makedirs("data", exist_ok=True)
    with open(SUBS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def set_subscription(email, role):
    """
    관리자가 권한 변경할 때 1개월 만료일 저장
    - pro / prime : 오늘 기준 +30일
    - free / admin : 구독 만료 처리
    """
    email = (email or "").strip()
    if not email:
        return

    db = load_subs_db()
    subs = db.get("subs", {})
    today = now_kst().date()

    if role in ["pro", "prime"]:
        expire = today + timedelta(days=30)
        subs[email] = {
            "role": role,
            "paid_at": today.strftime("%Y-%m-%d"),
            "expire_at": expire.strftime("%Y-%m-%d"),
        }
    else:
        # free / admin 등으로 바뀌면 구독 종료로 간주
        if email in subs:
            subs[email]["role"] = role
            subs[email]["expire_at"] = today.strftime("%Y-%m-%d")

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
    """
    DB/auth_user 등에 저장된 시간을 KST 문자열로 변환
    - 타임존 없는 값: 기존에 KST로 저장되었다고 가정 → KST로 localize만 함 (시간 안 바뀜)
    - 타임존 있는 값: 해당 타임존에서 KST로 convert
    """
    if value is None or value == "" or value == "NaT":
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""

    if ts.tzinfo is None:
        ts = ts.tz_localize(KST)
    else:
        ts = ts.tz_convert(KST)

    return ts.strftime(fmt)

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
    FDR_OK = False
    logger.warning("FinanceDataReader not available: %s", e)

try:
    from pykrx import stock
    PYKRX_OK = True
except Exception as e:
    PYKRX_OK = False
    logger.info("pykrx not available: %s", e)

# 2. 페이지 설정
st.set_page_config(page_title="LDY Pro Trader v6.3", layout="wide", page_icon="💎")
st.title("🏆 LDY Pro Trader v6.3 (Enhanced Score + Subscription)")
st.caption("AI Quant Analysis & Portfolio Manager — Scoring / Subscription / Portfolio")

st.warning(
    "⚠️ 투자 관련 유의사항\n\n"
    "LDY Pro Trader는 주식 투자 의사결정을 돕기 위한 **데이터·알고리즘 기반 분석 도구**입니다.\n"
    "제공되는 모든 정보는 일반적인 참고용 자료일 뿐이며, 특정 종목의 매수·매도, 수익 창출 또는 손실 회피를 보장하지 않습니다.\n\n"
    "실제 투자에 대한 최종 판단과 그에 따른 결과(수익·손실 포함)는 **전적으로 이용자 본인에게 귀속**되며,\n"
    "본 서비스 및 개발자는 어떠한 법적 책임도 부담하지 않습니다."
)

# 3. 설정 관리 (Secrets -> Env -> Default 순서)
def get_conf(key, default_val):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except FileNotFoundError:
        pass
    return os.getenv(key, default_val)

RAW_URL = get_conf(
    "LDY_RAW_URL",
    "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
)
LOCAL_RAW = get_conf("LDY_LOCAL_RAW", "data/recommend_latest.csv")
PORTFOLIO_FILE = get_conf("LDY_PORTFOLIO_FILE", "my_portfolio.json")

# 보안키
KEY_PRO = get_conf("LDY_KEY_PRO", "220577")
KEY_PRIME = get_conf("LDY_KEY_PRIME", "577220")
ADMIN_KEY = get_conf("LDY_ADMIN_KEY", "2022322")

# 결제 계좌 정보 (전역 설정)
BANK_ACCOUNT = get_conf("LDY_BANK_ACCOUNT", "카카오뱅크 3333-22-2658701")
BANK_HOLDER  = get_conf("LDY_BANK_HOLDER", "이두영")

# 스코어링 상수
PASS_EBS = float(get_conf("LDY_PASS_EBS", 4))
MIN_TURN_KOSPI = float(get_conf("LDY_MIN_TURN_KOSPI", 200.0))
MIN_TURN_KOSDAQ = float(get_conf("LDY_MIN_TURN_KOSDAQ", 100.0))
MIN_TURN_DEFAULT = float(get_conf("LDY_MIN_TURN_DEFAULT", 100.0))

# 가중치
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = (0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10)

# 패널티
P_OVERHEAT_5D = 6.0
P_OVERHEAT_10D = 6.0
P_RSI_OUT = 4.0
P_MACD_NEG = 4.0
P_NEAR_FAR = 4.0
P_LIQ_LOW = 4.0
P_VOL_SPIKE = 2.0

# RSI 적정 구간
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
    if FDR_OK:
        try:
            df = fdr.StockListing('KRX')
            return dict(zip(df['Name'], df['Code'].astype(str).str.zfill(6)))
        except Exception as e:
            logger.exception("get_code_map failed")
    return {}

def find_code_by_name(name_or_code, code_map):
    name_or_code = str(name_or_code).strip()
    if name_or_code.isdigit():
        return name_or_code.zfill(6)
    return code_map.get(name_or_code, None)

@st.cache_data(ttl=600)
def get_market_status_local(scored_df: pd.DataFrame):
    """
    FDR/pykrx가 막혀도 동작하는 로컬 시장 상태 계산
    - 기준: 각 시장별 5일 수익률(ret_5d_%) 평균
    - 평균 > 0  → 상승장
      평균 ≤ 0 → 조정장
    """
    result = {}

    for mkt in ["KOSPI", "KOSDAQ"]:
        sub = scored_df[scored_df.get("시장", "") == mkt].copy()
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


    """
    KOSPI / KOSDAQ 상태 조회 (통합 래퍼)
    1) FDR / pykrx 인덱스 데이터로 계산 시도
    2) 실패하거나 데이터 오류면 scored DF 기반 로컬 계산으로 fallback
    """
    # -----------------------------
    # 1) 인덱스 데이터 기반 계산
    # -----------------------------
    if not FDR_OK and not PYKRX_OK:
        # 바로 로컬로
        if "scored" in globals():
            try:
                return get_market_status_local(globals()["scored"])
            except Exception:
                logger.exception("get_market_status_local fallback failed (no FDR/PYKRX)")
        return "데이터 소스 오류", float("nan"), "데이터 소스 오류", float("nan")

    def _via_fdr(ticker: str):
        """FinanceDataReader 경로"""
        if not FDR_OK:
            return None
        try:
            df = fdr.DataReader(ticker)
            if df is None or df.empty:
                return None
            return df
        except Exception:
            logger.exception("FDR DataReader failed for %s", ticker)
            return None

    def _via_pykrx_index(ticker: str):
        """pykrx 인덱스 경로 (KS11/KQ11 대응)"""
        if not PYKRX_OK:
            return None
        try:
            today = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

            # KS11(코스피 지수) → 1001, KQ11(코스닥 지수) → 2001
            code = "1001" if ticker == "KS11" else "2001"
            df = stock.get_index_ohlcv_by_date(start, today, code)
            if df is None or df.empty:
                return None

            # pykrx: '종가' 컬럼을 Close로 맞춰줌
            if "종가" in df.columns and "Close" not in df.columns:
                df = df.rename(columns={"종가": "Close"})
            return df
        except Exception:
            logger.exception("pykrx index fetch failed for %s", ticker)
            return None

    def _status_for(ticker: str):
        """단일 지수(KOSPI/KOSDAQ) 상태 계산"""
        df = _via_fdr(ticker)
        if df is None:
            df = _via_pykrx_index(ticker)

        if df is None or df.empty:
            return "데이터 오류", float("nan")

        # 최근 60개만 사용
        df = df.tail(60)

        if "Close" not in df.columns:
            return "데이터 부족", float("nan")

        close = df["Close"]
        ma20 = close.rolling(20).mean().iloc[-1]
        curr = close.iloc[-1]

        if pd.isna(ma20) or ma20 == 0:
            return "데이터 부족", float("nan")

        diff = ((curr - ma20) / ma20) * 100
        status = "📈 상승장" if diff > 0 else "📉 조정장"

        # 마지막 데이터 날짜 기준이 오늘보다 이전이면 "(전일 기준)" 붙이기
        last_idx = df.index[-1]
        try:
            last_date = last_idx.date()
        except Exception:
            last_date = pd.to_datetime(last_idx).date()

        today = datetime.now().date()
        if last_date < today:
            status += " (전일 기준)"

        return status, diff

    try:
        kp_stat, kp_diff = _status_for("KS11")
        kq_stat, kq_diff = _status_for("KQ11")

        # 인덱스 데이터가 정상이면 그대로 사용
        bad_stats = {
            "데이터 없음",
            "데이터 오류",
            "데이터 소스 오류",
            "데이터 부족",
            "Unknown",
            "Error",
        }
        if kp_stat not in bad_stats or kq_stat not in bad_stats:
            return kp_stat, kp_diff, kq_stat, kq_diff
    except Exception:
        logger.exception("get_market_status index path failed")

    # -----------------------------
    # 2) 로컬 scored DF 기반 fallback
    # -----------------------------
    if "scored" in globals():
        try:
            return get_market_status_local(globals()["scored"])
        except Exception:
            logger.exception("get_market_status_local fallback failed")

    return "데이터 소스 오류", float("nan"), "데이터 소스 오류", float("nan")

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    if not FDR_OK:
        return 50, "Neutral"
    try:
        df = fdr.DataReader('KS11')
        if df.empty:
            return 50, "Neutral"
        delta = df['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        rs = up.rolling(14).mean() / down.abs().rolling(14).mean()
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        ma20 = df['Close'].rolling(20).mean()
        disparity = (df['Close'] / ma20 * 100).iloc[-1]

        score = current_rsi
        if disparity > 105:
            score += 10
        elif disparity < 95:
            score -= 10
        score = max(0, min(100, score))

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
        return score, status
    except Exception:
        logger.exception("fear_greed failed")
        return 50, "Error"

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

def plot_sector_treemap(df):
    if '업종' not in df.columns:
        return None
    df_map = df.copy()
    df_map['업종'] = df_map['업종'].fillna('기타')
    df_map = df_map[df_map['업종'] != '기타']
    if df_map.empty:
        return None
    if 'LDY_SCORE' in df_map.columns:
        df_map['LDY_SCORE'] = pd.to_numeric(df_map['LDY_SCORE'], errors='coerce').fillna(0).round(1)
    fig = px.treemap(
        df_map,
        path=['업종', '종목명'],
        values='거래대금(억원)',
        color='LDY_SCORE',
        color_continuous_scale='RdYlGn',
        title="<b>🔥 시장 주도 섹터 지도</b>",
        custom_data=['LDY_SCORE']
    )
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>점수: %{customdata[0]:.1f}<br>대금: %{value}억<extra></extra>'
    )
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), height=350)
    return fig

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
    if not FDR_OK:
        return None
    try:
        code_str = str(code).zfill(6)
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        df = fdr.DataReader(code_str, start_date)
        if df is None or df.empty:
            return None
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df = calculate_supertrend(df)
        return df.tail(80)
    except Exception:
        logger.exception("get_stock_chart_data failed")
        return None

def plot_radar_chart(row):
    stats = {
        "모멘텀": min(100, (row.get("ret_5d_%", 0) + 5) * 10),
        "수급(MFI)": row.get("MFI14", 50),
        "가성비(RR)": min(100, row.get("RR1", 1) * 50),
        "안전성": 100 - (row.get("이격도", 0) * 2),
        "종합점수": row.get("LDY_SCORE", 0),
    }
    values = [max(0, min(100, v)) for v in stats.values()]
    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=list(stats.keys()),
            fill='toself',
            name=row.get('종목명', '종목')
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=250,
        margin=dict(l=30, r=30, t=20, b=20),
    )
    return fig

# ---------------------------
# 차트 시각화 (거래량 추가)
# ---------------------------
def plot_interactive_chart(df, code, name, entry, stop, target1, target2):
    if df is None or df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="주가",
        increasing_line_color='#ef5350',
        decreasing_line_color='#2979ff',
        hovertemplate="<b>%{x|%y/%m/%d}</b><br>종가: %{close:,}원<extra></extra>",
        showlegend=False
    ), row=1, col=1)

    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA20'], line=dict(color='orange', width=1.5), name='20일선'
        ), row=1, col=1)
    if 'MA60' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA60'], line=dict(color='purple', width=1.5), name='60일선'
        ), row=1, col=1)

    up = df[df['Trend'] == 1]
    down = df[df['Trend'] == -1]
    if not up.empty:
        fig.add_trace(go.Scatter(
            x=up.index, y=up['SuperTrend'], mode='markers', marker=dict(color='green', size=2), name='상승추세'
        ), row=1, col=1)
    if not down.empty:
        fig.add_trace(go.Scatter(
            x=down.index, y=down['SuperTrend'], mode='markers', marker=dict(color='red', size=2), name='하락추세'
        ), row=1, col=1)

    colors = ['#ef5350' if row['Close'] >= row['Open'] else '#2979ff' for idx, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], marker_color=colors, name="거래량", showlegend=False
    ), row=2, col=1)

    lines = [
        (entry, "🔵진입", "blue"),
        (stop, "🔴손절", "red"),
        (target1, "🟢목표1", "green"),
    ]
    for val, label, color in lines:
        try:
            if pd.notna(val) and val > 0:
                fig.add_hline(
                    y=val, line_dash="dash", line_color=color,
                    annotation_text=label, row=1, col=1
                )
        except Exception:
            pass

    fig.update_layout(
        title=f"{name} ({code})",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
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
@st.cache_data(ttl=600)
def load_csv_url(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=600)
def load_csv_path(path):
    return pd.read_csv(path, encoding="utf-8")

def log_src(df, src):
    logger.info("Data Loaded: %s rows=%s", src, len(df) if df is not None else 0)

def load_portfolio_file():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("data", "")
        except Exception:
            logger.exception("load_portfolio_file failed")
    return ""

def save_portfolio_file(text_data):
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump({"data": text_data}, f, ensure_ascii=False)
        return True
    except Exception:
        logger.exception("save_portfolio_file failed")
        return False

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

def build_global_score(lat):
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
    vol_col = "거래강도" if "거래강도" in x.columns and x["거래강도"].not나().any() \
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
            lo, hi = np.nanpercentile(turn.dropna(), 30), np.nanpercentile(turn.drop나(), 90)
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
    pen += P_OVERHEAT_10D * np.clip((r10 - 25) / 25, 0, 1).fill나(0)
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
        x["MA20_GAP"] = (
            (nz_num(x["종가"]) / nz_num(x["MA20"]) - 1.0) * 100
        ).replace([np.inf, -np.inf], np.nan)
    else:
        x["MA20_GAP"] = np.nan

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
        thr['ebs_q60'] = float(np.nanpercentile(s.drop나(), 60)) if s.drop나().size > 0 else PASS_EBS
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
    try:
        r5 = float(row.get("ret_5d_%", 0) or 0)
    except Exception:
        r5 = 0.0
    try:
        slope = float(row.get("MACD_Slope", row.get("MACD_slope", 0)) or 0)
    except Exception:
        slope = 0.0
    try:
        ebs = float(row.get("EBS", 0) or 0)
    except Exception:
        ebs = 0.0
    try:
        now_pct = float(row.get("Now%", 999) or 999)
    except Exception:
        now_pct = 999
    try:
        rr1 = float(row.get("RR1", 0) or 0)
    except Exception:
        rr1 = 0.0
    try:
        ma20_gap = float(row.get("MA20_GAP", 0) or 0)
    except Exception:
        ma20_gap = 0.0

    strong = (
        (r5 >= th['r5_q75'])
        and (slope >= th['slope_q60'])
        and (ebs >= th['ebs_q60'])
        and (now_pct <= th['now_gap_q25'])
    )
    if strong and rr1 >= 0.5:
        return "🔼 BRK (강력 돌파)"

    if (slope > 0 and r5 > 0) or (ebs >= th['ebs_q60'] and now_pct <= th['now_gap_q25'] * 1.5):
        if r5 >= max(1.0, th['r5_q75'] * 0.6) and slope > 0:
            return "🔺 Watch→BRK (관찰·돌파예상)"
        return "🔺 Watch (상승 준비)"

    if ma20_gap > 1 and slope > 0 and ebs >= PASS_EBS:
        return "🔼 BRK (MA20상승)"

    return "↩️ PULL (눌림)"

@st.cache_data(ttl=600)
def prepare_scored_data(raw_url, local_raw, pass_ebs):
    """
    - CSV 로드 (원격 → 실패 시 로컬)
    - normalize_cols
    - build_global_score
    - 동적 threshold + ROUTE
    - base / top20 / P_hit 계산
    """
    df_raw = None

    try:
        df_raw = load_csv_url(raw_url)
        log_src(df_raw, "Remote")
    except Exception as e_remote:
        logger.warning("prepare_scored_data: Remote load failed: %s", e_remote)
        if os.path.exists(local_raw):
            try:
                df_raw = load_csv_path(local_raw)
                log_src(df_raw, "Local")
            except Exception as e_local:
                logger.exception("prepare_scored_data: Local load failed: %s", e_local)

    if df_raw is None:
        raise RuntimeError("CSV를 원격/로컬 어디서도 불러오지 못했습니다.")

    df = normalize_cols(df_raw)
    latest = df.copy()
    scored = build_global_score(latest)

    TH = compute_dynamic_thresholds(scored)
    scored["ROUTE"] = scored.apply(
        lambda r: route_tag_dynamic(r, TH),
        axis=1
    ).fillna("—")

    base = scored[(scored["EBS"] >= pass_ebs) & (scored["_GATE_OK"])].copy()
    if len(base) < 20:
        base = scored.head(20)

    top20 = base.head(20).copy()
    top20["P_hit"] = (top20["LDY_SCORE"] / 100.0 * 0.8).clip(0, 1) * 100

    return scored, base, top20, TH

# ---------------------------
# 메인 데이터 로드 (Status UX)
# ---------------------------
with st.status("🚀 시장 데이터를 분석하고 있습니다...", expanded=True) as status:
    status.write("📥 데이터 다운로드 및 스코어링 계산 중...")
    try:
        scored, base, top20, TH = prepare_scored_data(
            RAW_URL,
            LOCAL_RAW,
            PASS_EBS,
        )
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
from auth_user import render_auth_box, list_users, update_user_role

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
    st.subheader("💎 프리미엄 구독 안내")

    PRICE_PRO = 19000
    PRICE_PRIME = 39000

    with st.container():
        st.markdown("### 🌱 **Free (무료)**")
        st.markdown(
            "- ✅ **회원가입 후** 상위 **5개 종목** 조회 (Guest는 3개)\n"
            "- ✅ 시장 지표/섹터맵 열람\n"
            "- ❌ 내 포트폴리오 분석\n"
            "- ❌ CSV 다운로드 / 알림"
        )

    with st.container():
        st.markdown(f"### 🚀 **Pro (월 {PRICE_PRO:,}원)**")
        st.markdown(
            "`실전 투자자용`\n"
            "- 🔓 필터 적용 **Top 20 종목** 열람\n"
            "- 💼 **내 자산(포트폴리오)** 수익률 분석\n"
            "- 📊 개별 종목 레이더·리스크/리워드 차트\n"
            "- ❌ CSV 다운로드\n"
            "- ❌ 텔레그램 알림"
        )

    with st.container():
        st.markdown(f"### 👑 **Prime (월 {PRICE_PRIME:,}원)**")
        st.markdown(
            "`전업 / 하이엔드 투자자`\n"
            "- ✅ **전체 스코어링 종목** 열람\n"
            "- ✅ CSV 다운로드\n"
            "- ✅ 텔레그램 요약 알림 (Top 종목 브리핑)\n"
            "- ✅ 향후 고급 리포트 / 기능 우선 적용"
        )

    st.markdown("#### 💳 결제(입금) 안내")
    st.markdown(
        f"- 입금계좌: **{BANK_ACCOUNT}**  \n"
        f"- 예금주: **{BANK_HOLDER}**  \n"
        "- 입금 후 카카오톡 채널 또는 문의 게시판에 **입금자명 / 이메일 / 희망 요금제(Pro/Prime)**를 남겨 주세요.  \n"
        "- 관리자가 입금 확인 후 **1개월 단위로 권한을 부여/연장**합니다."
    )

    if user and expire_str:
        st.info(f"현재 구독 만료 예정일: **{expire_str}**")

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
            today = datetime.now().date()
            for u in users:
                email = u.get("login_id")
                sub = subs.get(email, {})
                exp_str = sub.get("expire_at", "")
                days_left = ""
                if exp_str:
                    try:
                        d_exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
                        days_left = (d_exp - today).days
                    except Exception:
                        days_left = ""
                rows.append({
                    "이메일": email,
                    "닉네임": u.get("nickname"),
                    "권한(auth_user)": u.get("role"),
                    "구독 역할(sub)": sub.get("role", ""),
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
    msg = f"🔥 [LDY v6.3] 추천 Top 5 ({datetime.now().strftime('%m/%d')})\n\n"
    for i in range(min(5, len(top20))):
        row = top20.iloc[i]
        msg += f"{i+1}. {row.get('종목명','-')} ({row.get('ROUTE','-')})\n"
        msg += f"   매수: {int(row.get('추천매수가',0)):,} / 손절: {int(row.get('손절가',0)):,}\n\n"
    ok, res = send_telegram_msg(tg_token, tg_chat_id, msg)
    if ok:
        st.toast("전송 완료!", icon="✅")
    else:
        st.error(f"전송 실패: {res}")

# ---------------------------
# 메인 UI
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 시장 (Market)", "🔭 종목 분석", "💼 내 자산", "📮 문의 게시판", "⚖️ 이용 약관 / 투자 유의사항"]
)

with tab1:
    kp_stat, kp_diff, kq_stat, kq_diff = get_market_status()
    c1, c2 = st.columns(2)

    def _fmt_metric(stat, diff):
        """지수 상태/증감값을 안전하게 포맷"""
        bad_stats = {
            "데이터 없음",
            "데이터 오류",
            "데이터 소스 오류",
            "데이터 부족",
            "Unknown",
            "Error",
        }

        # 에러/데이터 부족이면 예쁜 문구로 통일
        if stat in bad_stats or pd.isna(diff):
            friendly = "📡 지수 데이터 지연/점검 중"
            return friendly, "-", "off"

        # 정상일 때만 % 표시
        delta_txt = f"{diff:.2f}%"
        delta_color = "off" if ("상승" in stat or diff >= 0) else "inverse"
        return stat, delta_txt, delta_color

    kp_value, kp_delta, kp_color = _fmt_metric(kp_stat, kp_diff)
    kq_value, kq_delta, kq_color = _fmt_metric(kq_stat, kq_diff)

    c1.metric("KOSPI", kp_value, kp_delta, delta_color=kp_color)
    c2.metric("KOSDAQ", kq_value, kq_delta, delta_color=kq_color)

    st.divider()
    c_gauge, c_map = st.columns([1, 1.5])
    with c_gauge:
        fg_score, fg_status = get_fear_greed_index()
        st.plotly_chart(
            plot_fear_greed_gauge(fg_score),
            use_container_width=True,
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

with tab2:
    st.subheader("🎯 추천 종목 필터")

    if just_registered:
        st.success("🎉 첫 가입을 환영합니다! 오늘 기준 TOP 5 프리뷰를 먼저 보여드릴게요.")
        try:
            preview = base.sort_values("LDY_SCORE", ascending=False).head(5).copy()
        except Exception:
            preview = scored.sort_values("LDY_SCORE", ascending=False).head(5).copy()

        if not preview.empty:
            cols = [
                "종목명", "종목코드", "LDY_SCORE",
                "추천매수가", "손절가", "추천매도가1"
            ]
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
    with col_f2:
        all_routes = sorted(
            scored["ROUTE"].dropna().unique().tolist()
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
        use_only_gate = st.checkbox(
            "EBS/유동성 통과만 사용",
            value=True,
            key="only_gate",
        )

    if use_only_gate:
        base_view = top20.copy()
    else:
        base_view = scored.sort_values(
            ["LDY_SCORE", "거래대금(억원)"],
            ascending=[False, False]
        ).head(50)

    filtered = base_view.copy()
    filtered = filtered[filtered["LDY_SCORE"] >= min_score]
    if sel_routes:
        filtered = filtered[filtered["ROUTE"].isin(sel_routes)]

    if auth_status in ["pro", "prime", "admin"]:
        view_df = filtered.head(20)
        st.success(
            f"🥇 {auth_status.upper()} 회원: 필터 적용 Top {len(view_df)} 종목 열람 중"
        )
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
            lambda r: f"{r.get('종목명','-')} ({r.get('종목코드','-')})",
            axis=1
        ).tolist()
        sel = st.selectbox("종목 선택", opts)
        if sel:
            sel_idx = opts.index(sel)
            row = view_df.iloc[sel_idx]
            code = row.get('종목코드', '')

            c1, c2 = st.columns([2, 1])
            with c1:
                chart_df = get_stock_chart_data(code)
                if chart_df is not None:
                    st.plotly_chart(
                        plot_interactive_chart(
                            chart_df, code,
                            row.get('종목명', '-'),
                            row.get('추천매수가', 0),
                            row.get('손절가', 0),
                            row.get('추천매도가1', 0),
                            row.get('추천매도가2', 0),
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("차트 데이터 없음")
            with c2:
                if auth_status in ["pro", "prime", "admin"]:
                    st.markdown(f"### {row.get('종목명','-')}")
                    st.plotly_chart(
                        plot_radar_chart(row),
                        use_container_width=True,
                    )
                    ai_cmt = row.get("AI_COMMENT", row.get("WHY", "-"))
                    st.info(f"💬 **AI:** {ai_cmt}")
                    st.plotly_chart(
                        plot_risk_reward_bar(
                            row.get('추천매수가', 0),
                            row.get('손절가', 0),
                            row.get('추천매도가1', 0),
                            row.get('추천매도가2', 0),
                        ),
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "🔒 상세 분석(레이더 / 리스크-리워드 / AI 코멘트)은 **Pro 등급부터** 확인 가능합니다."
                    )

                c_a, c_b = st.columns(2)
                c_a.metric("진입가", f"{int(row.get('추천매수가', 0)):,}")
                c_b.metric(
                    "손절가",
                    f"{int(row.get('손절가', 0)):,}",
                    delta="Stop",
                    delta_color="inverse",
                )

    st.divider()
    st.subheader("📋 Daily Top List", anchor=False)
    safe_view = view_df.copy().reset_index(drop=True)
    if not safe_view.empty:
        if "종목명" in safe_view.columns:
            safe_view.set_index("종목명", inplace=True)
        price_cols = [
            "종가", "추천매수가", "손절가", "추천매도가1", "추천매도가2", "거래대금(억원)",
        ]
        for c in price_cols:
            if c in safe_view.columns:
                safe_view[c] = pd.to_numeric(
                    safe_view[c], errors='coerce'
                ).fillna(0).apply(lambda x: f"{int(x):,}")
        if "LDY_SCORE" in safe_view.columns:
            safe_view["LDY_SCORE"] = pd.to_numeric(
                safe_view["LDY_SCORE"], errors='coerce'
            ).fill나(0)
        cols = [
            "ROUTE", "업종", "종목코드", "LDY_SCORE",
            "종가", "추천매수가", "손절가", "추천매도가1",
        ]
        cols = [c for c in cols if c in safe_view.columns]
        cfg = {
            "LDY_SCORE": st.column_config.ProgressColumn(
                "점수", format="%.1f", min_value=0, max_value=100
            ),
            "종가": st.column_config.TextColumn("현재가"),
            "추천매수가": st.column_config.TextColumn("진입가"),
            "손절가": st.column_config.TextColumn("손절가"),
            "추천매도가1": st.column_config.TextColumn("목표가"),
        }
        st.dataframe(
            safe_view[cols],
            use_container_width=True,
            column_config=cfg,
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
    if not FDR_OK:
        return code, name, 0
    try:
        df = fdr.DataReader(str(code), (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return code, name, 0
        return code, name, int(df.iloc[-1]['Close'])
    except Exception:
        return code, name, 0

with tab3:
    if auth_status in ["guest", "free"]:
        st.info("🔒 내 자산 분석은 Pro 등급부터 가능합니다.")
    elif pf_input:
        try:
            code_map = get_code_map() if 'get_code_map' in globals() else {}
            if not code_map and FDR_OK:
                try:
                    df_krx = fdr.StockListing('KRX')
                    code_map = dict(zip(df_krx['Name'], df_krx['Code'].astype(str).str.zfill(6)))
                except Exception:
                    pass

            targets = []
            lines = pf_input.strip().split('\n')
            for line in lines:
                if ":" not in line:
                    continue
                name_input, avg, qty = line.split(':')
                code = str(name_input).strip()
                if not code.isdigit():
                    code = code_map.get(code, code)
                code = str(code).zfill(6)
                targets.append((code, name_input, float(avg.replace(',', '')), int(qty.replace(',', ''))))

            price_map = {}
            with st.spinner('⚡ 실시간 시세를 조회 중입니다...'):
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(fetch_current_price, t[0], t[1]) for t in targets]
                    for future in futures:
                        c, n, p = future.result()
                        price_map[c] = p

            cols_layout = st.columns(3)
            total_buy = 0
            total_eval = 0

            for idx, (code, name_input, avg, qty) in enumerate(targets):
                cur_price = price_map.get(code, 0)
                real_name = stock.get_market_ticker_name(code) if (PYKRX_OK and cur_price > 0) else name_input

                if cur_price > 0:
                    profit_rate = (cur_price - avg) / avg * 100
                    if profit_rate > 0:
                        signal = "🟢 수익"
                    elif profit_rate > -3:
                        signal = "🟡 보합"
                    else:
                        signal = "🔴 손실"
                else:
                    signal = "❓ 확인불가"
                    profit_rate = 0

                buy_amt = avg * qty
                eval_amt = cur_price * qty
                total_buy += buy_amt
                total_eval += eval_amt

                with cols_layout[idx % 3]:
                    st.metric(
                        label=f"{real_name} ({signal})",
                        value=f"{cur_price:,}원",
                        delta=f"{profit_rate:+.2f}% ({int(eval_amt-buy_amt):,}원)",
                        delta_color="normal" if profit_rate >= 0 else "inverse",
                    )

            st.divider()
            c1, c2, c3 = st.columns(3)
            tot_rate = (total_eval - total_buy) / total_buy * 100 if total_buy > 0 else 0
            c1.metric("총 매수", f"{int(total_buy):,}원")
            c2.metric("총 평가", f"{int(total_eval):,}원")
            c3.metric(
                "총 수익",
                f"{tot_rate:+.2f}%",
                f"{int(total_eval-total_buy):,}원",
                delta_color="normal" if tot_rate >= 0 else "inverse",
            )
        except Exception as e:
            logger.exception("pf analysis failed")
            st.error(f"분석 실패: {e}")
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
            db = load_inquiry_db()
            inq_list = db.get("inquiries", [])

            inq_list.append({
                "title": title.strip(),
                "content": content.strip(),
                "nickname": nickname.strip() or "익명",
                "email": email.strip(),
                "created_at": now_kst().strftime("%Y-%m-%d %H:%M:%S"),
            })
            db["inquiries"] = inq_list
            save_inquiry_db(db)
            st.success("문의가 등록되었습니다. 가능한 한 빠르게 확인하겠습니다. 🙌")

    st.markdown("#### 📂 최근 문의 내역")

    db = load_inquiry_db()
    inquiries = db.get("inquiries", [])

    if not inquiries:
        st.info("아직 등록된 문의가 없습니다.")
    else:
        for item in reversed(inquiries[-50:]):
            box = st.container(border=True)
            with box:
                st.markdown(f"**제목:** {item.get('title', '-')}")
                meta = f"작성자: {item.get('nickname','익명')} · 작성일: {item.get('created_at','-')}"
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

    st.markdown("### 4. 구독 및 계정 정책 (요약)")
    st.markdown(
        "- **Guest(비회원)** : 상위 3개 종목 맛보기.\n"
        f"- **Free(회원)** : 상위 5개 종목 열람.\n"
        f"- **Pro (월 {PRICE_PRO:,}원)** : 상위 20 종목, 내 자산 분석 기능 제공.\n"
        f"- **Prime (월 {PRICE_PRIME:,}원)** : 전체 종목, CSV 다운로드, 텔레그램 알림 등 고급 기능 제공.\n"
        "- 구체적인 결제/환불/구독 해지 정책은 별도 안내(카카오 채널, 약관 페이지 등)를 따릅니다."
    )

    st.markdown("### 5. 한 줄 요약")
    st.info("👉 **데이터와 퀀트는 도구일 뿐, 최종 책임은 언제나 본인에게 있다.**")
