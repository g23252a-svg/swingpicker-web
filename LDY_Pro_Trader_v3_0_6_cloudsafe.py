import streamlit as st
import pandas as pd
import numpy as np
import io, os, time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====(옵션) pykrx가 import에 실패해도 앱이 죽지 않게=====
try:
    from pykrx import stock
    HAS_PYKRX = True
except Exception:
    HAS_PYKRX = False

# =========================
# 기본 세팅
# =========================
st.set_page_config(page_title="LDY Pro Trader v3.0.6 (CloudSafe FullScan)", layout="wide")
st.title("📈 LDY Pro Trader v3.0.6 (CloudSafe FullScan)")
st.caption("KOSPI+KOSDAQ 전종목 급등 초입 자동 스캐너 | Cloud 네트워크 차단 시에도 안전 폴백")

KST = timezone(timedelta(hours=9))
def ymd(d=None): 
    d = d or datetime.now(KST)
    return d.strftime("%Y%m%d")

def effective_ymd(use_prev_close: bool) -> str:
    now = datetime.now(KST)
    roll = now.replace(hour=9, minute=5, second=0, microsecond=0)
    base = (now.date() - timedelta(days=1)) if (use_prev_close or now < roll) else now.date()
    return base.strftime("%Y%m%d")

# =========================
# 지표 계산 함수
# =========================
def rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd_hist(close, fast=12, slow=26, sig=9):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig).mean()
    return macd - signal

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def zscore(x, window=20):
    return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-9)

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ 스캔 조건 (전종목용)")
lookback = int(st.sidebar.number_input("조회일수", 20, 252, 60))
turnover_min = st.sidebar.number_input("거래대금 하한 (억원)", 0, 5000, 50)
mcap_min = st.sidebar.number_input("시총 하한 (억원)", 0, 1000000, 1000)
rsi_min = st.sidebar.number_input("RSI 하한", 0, 100, 45)
rsi_max = st.sidebar.number_input("RSI 상한", 0, 100, 65)
score_pass = st.sidebar.number_input("통과점수", 0, 7, 4)
use_prev_close = st.sidebar.checkbox("전일 기준(장마감)", True)
st.sidebar.divider()

# =========================
# 데이터 로더 (3단계 폴백)
# 1) pykrx 라이브 수집 시도
# 2) /data/full_ohlcv.csv 자동 로드
# 3) 업로더로 CSV 받기 or 샘플 생성
# =========================
@st.cache_data(ttl=1800)
def load_full_ohlcv_via_pykrx(lookback: int) -> pd.DataFrame:
    """Cloud에서 막히면 빈 DF 반환 (죽지 않음)"""
    if not HAS_PYKRX:
        return pd.DataFrame()

    end = ymd()
    start = ymd(datetime.now(KST) - timedelta(days=int(lookback * 1.5)))
    try:
        kospi = stock.get_market_ticker_list(market="KOSPI")
        kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
    except Exception:
        return pd.DataFrame()

    tickers = (kospi or []) + (kosdaq or [])
    if not tickers:
        return pd.DataFrame()

    # 미리 KOSPI set 캐싱 (시장 라벨링용)
    kset = set(kospi)

    results = []
    def fetch(code):
        try:
            df = stock.get_market_ohlcv_by_date(start, end, code)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.reset_index().rename(columns={"날짜":"날짜"})
            df["종목명"] = stock.get_market_ticker_name(code)
            df["종목코드"] = code
            df["시장"] = "KOSPI" if code in kset else "KOSDAQ"
            # pykrx 표준 컬럼명을 본 앱 통일 스키마로 매핑
            # 기대 컬럼: 날짜, 시장, 종목명, 종목코드, 시가/고가/저가/종가, 거래량, 거래대금(억원), 시가총액(억원)
            # pykrx 거래대금은 원 단위가 아님 → get_market_ohlcv_by_date는 '거래대금'이 원화로 들어옴(일반적으로)
            df["거래대금(억원)"] = (df["거래대금"] / 1e8).round(2)
            # 시총은 별도 API 호출 부담 → 임시 NaN (필터에서 하한 쓰면 제거됨)
            df["시가총액(억원)"] = np.nan
            return df[["날짜","시장","종목명","종목코드","시가","고가","저가","종가","거래량","거래대금(억원)","시가총액(억원)"]]
        except Exception:
            return pd.DataFrame()

    # 병렬 수집
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch, c) for c in tickers]
        for f in as_completed(futures):
            r = f.result()
            if r is not None and not r.empty:
                results.append(r)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

def load_full_ohlcv_from_repo() -> pd.DataFrame:
    """리포지토리에 미리 넣어둔 CSV 자동 로드 (예: data/full_ohlcv.csv)"""
    candidates = [
        "data/full_ohlcv.csv",
        "full_ohlcv.csv",
        "data/ohlcv_latest.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                pass
    return pd.DataFrame()

def generate_sample_universe(n_tickers=200, lookback=60) -> pd.DataFrame:
    """랜덤 기반 샘플 종목 생성 (길이 mismatch 방지 완전 안전 버전)"""
    idx = pd.date_range(end=datetime.now(KST).date(), periods=lookback, freq="D")
    all_rows = []
    rng = np.random.default_rng(42)

    for i in range(n_tickers):
        base = float(rng.uniform(3_000, 150_000))
        close = pd.Series(base * (1 + rng.normal(0, 0.01, lookback)).cumprod(), index=idx)
        close = close.clip(500, None)
        high = close * (1 + rng.uniform(0.003, 0.02))
        low = close * (1 - rng.uniform(0.003, 0.02))
        vol = pd.Series(rng.normal(1.5e6, 5e5, lookback)).clip(1e5, None).round()
        tnov = (close.values * vol.values) / 1e8

        mk = "KOSPI" if i % 2 == 0 else "KOSDAQ"
        code = f"{i:06d}"
        name = f"SYM{i:03d}"
        mcap = float(rng.uniform(1500, 200000))  # ★스칼라로 고정★

        df = pd.DataFrame({
            "날짜": idx,
            "시장": mk,
            "종목명": name,
            "종목코드": code,
            "시가": (close * 0.995).round(0),
            "고가": high.round(0),
            "저가": low.round(0),
            "종가": close.round(0),
            "거래량": vol,
            "거래대금(억원)": np.round(tnov, 2),
            "시가총액(억원)": mcap,   # ★길이 동일하게 브로드캐스트 가능★
        })
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


# 1) pykrx 시도
with st.spinner("📊 KOSPI + KOSDAQ 전종목 불러오는 중..."):
    df_raw = load_full_ohlcv_via_pykrx(lookback)

# 2) 리포 CSV 자동 로드
if df_raw.empty:
    repo_df = load_full_ohlcv_from_repo()
    if not repo_df.empty:
        st.info("🔁 pykrx 접근 차단 감지 → 리포지토리 CSV로 대체합니다 (data/full_ohlcv.csv).")
        df_raw = repo_df

# 3) 사용자 업로드 or 샘플 생성
if df_raw.empty:
    st.warning("⚠️ Cloud 환경에서 pykrx가 차단되었습니다. 아래에서 CSV를 업로드하거나, 샘플 데이터로 테스트할 수 있습니다.")
    up = st.file_uploader("전종목 OHLCV CSV 업로드(컬럼 예: 날짜,시장,종목명,종목코드,시가,고가,저가,종가,거래량,거래대금(억원),시가총액(억원))")
    if up is not None:
        try:
            df_raw = pd.read_csv(up)
            st.success("✅ 업로드된 CSV를 사용합니다.")
        except Exception as e:
            st.error(f"CSV 로드 실패: {e}")

if df_raw.empty:
    st.info("🧪 샘플 유니버스(200종목)로 폴백하여 UI/로직 테스트를 진행합니다.")
    df_raw = generate_sample_universe(n_tickers=200, lookback=lookback)

# =========================
# 지표 계산 및 필터링
# =========================
def enrich(df):
    out = []
    for code, g in df.groupby("종목코드"):
        g = g.sort_values("날짜").copy()
        g["MA20"] = g["종가"].rolling(20).mean()
        g["乖離%"] = (g["종가"]/(g["MA20"]+1e-9)-1)*100
        g["RSI14"] = rsi(g["종가"], 14)
        g["MACD_hist"] = macd_hist(g["종가"])
        g["MACD_slope"] = g["MACD_hist"].diff()
        g["ATR14"] = atr(g["고가"], g["저가"], g["종가"], 14)
        g["Vol_Z"] = zscore(g["거래량"], 20)
        out.append(g)
    return pd.concat(out, ignore_index=True)

df = enrich(df_raw)
snap = df.sort_values("날짜").groupby(["시장","종목코드","종목명"]).tail(1)

# 점수화
snap["EBS"] = 0
snap.loc[snap["MACD_hist"] > 0, "EBS"] += 1
snap.loc[snap["MACD_slope"] > 0, "EBS"] += 1
snap.loc[snap["RSI14"].between(rsi_min, rsi_max), "EBS"] += 1
snap.loc[snap["乖離%"].between(0, 10), "EBS"] += 1
snap.loc[snap["Vol_Z"] >= 1.5, "EBS"] += 1

# 거래대금/시총 하한 (시총 NaN은 통과시키고 싶으면 fillna로 조정)
snap["시가총액(억원)"] = snap["시가총액(억원)"].fillna(mcap_min + 1)
picked = snap[
    (snap["거래대금(억원)"] >= turnover_min) &
    (snap["시가총액(억원)"] >= mcap_min) &
    (snap["EBS"] >= score_pass)
].sort_values(["EBS","거래대금(억원)","Vol_Z"], ascending=[False,False,False])

st.success(f"🔥 급등 초입 후보 {len(picked)}개 종목 발견!")

show_cols = ["시장","종목명","종목코드","종가","거래대금(억원)","시가총액(억원)",
             "乖離%","RSI14","MACD_hist","MACD_slope","Vol_Z","EBS"]
st.dataframe(picked[show_cols], use_container_width=True)

# 다운로드
csv = picked[show_cols].to_csv(index=False, encoding="utf-8-sig")
st.download_button("📥 CSV 다운로드", data=csv,
                   file_name=f"swingpicker_full_{effective_ymd(use_prev_close)}.csv", mime="text/csv")

# 안내
if not HAS_PYKRX:
    st.info("ℹ️ pykrx 미탑재 환경입니다. requirements.txt에 `pykrx`가 포함되어 있어야 합니다.")
st.caption("※ Cloud에서 KRX/Naver 접근이 차단되면 자동으로 CSV/샘플로 폴백합니다. 로컬 실행 시 pykrx가 정상 수집합니다.")
