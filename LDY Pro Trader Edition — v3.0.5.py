import streamlit as st
import pandas as pd
import numpy as np
import io, math, time
from datetime import datetime, timedelta, timezone

# =========================
# 기본 세팅
# =========================
st.set_page_config(page_title="Swing Picker Web v3.0.5 • LDY Pro Trader Edition", layout="wide")
st.title("📈 Swing Picker • v3.0.5 LDY Pro Trader")
st.caption("급등 초입 스코어 + ATR 리스크 + 포지션 사이징 | Made by LDY")

KST = timezone(timedelta(hours=9))
def effective_ymd(use_prev_close: bool) -> str:
    now = datetime.now(KST)
    roll = now.replace(hour=9, minute=5, second=0, microsecond=0)
    base = (now.date() - timedelta(days=1)) if (use_prev_close or now < roll) else now.date()
    return base.strftime("%Y%m%d")

# =========================
# 유틸 (지표)
# =========================
def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_dn = pd.Series(dn, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-9)
    return 100 - (100 / (1 + rs))

def macd_hist(close: pd.Series, fast=12, slow=26, sig=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return hist

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def zscore(x: pd.Series, window: int = 20):
    m = x.rolling(window).mean()
    s = x.rolling(window).std()
    return (x - m) / (s + 1e-9)

def nr7(high: pd.Series, low: pd.Series, window: int = 7):
    # 최근 n일 중 최저 고저폭(=NR7) 여부 (불리언)
    rng = high - low
    return rng == rng.rolling(window).min()

# =========================
# 사이드바 (프로 옵션)
# =========================
st.sidebar.header("⚙️ 스캔 조건 (LDY Pro)")
market_radio = st.sidebar.radio("시장", ["KOSPI","KOSDAQ","KOSPI+KOSDAQ"], index=2, horizontal=True)
markets = ["KOSPI","KOSDAQ"] if market_radio=="KOSPI+KOSDAQ" else [market_radio]

lookback = st.sidebar.number_input("조회일수", 20, 252, 60, step=1)
rec_count = st.sidebar.number_input("추천 종목 수", 1, 200, 15, step=1)

preset = st.sidebar.selectbox("유동성 프리셋", ["개잡주 배제 (50억↑)", "중형주 (100억↑)", "대형주 (300억↑)"], index=1)
turnover_min = {"개잡주 배제 (50억↑)":50, "중형주 (100억↑)":100, "대형주 (300억↑)":300}[preset]
vol_mult = st.sidebar.number_input("거래량 Z-score 하한", 0.0, 5.0, 1.5, step=0.1)

colp1, colp2 = st.sidebar.columns(2)
price_min = colp1.number_input("가격 ≥(원)", 0, 2_000_000, 1_000, step=100)
price_max = colp2.number_input("가격 ≤(원)", 0, 2_000_000, 1_000_000, step=1000)

colm1, colm2 = st.sidebar.columns(2)
mcap_min = colm1.number_input("시총 ≥(억원)", 0, 20_000_000, 1_000, step=10)
mcap_max = colm2.number_input("시총 ≤(억원)", 0, 20_000_000, 10_000_000, step=10)

st.sidebar.subheader("📈 초입 스코어 설정")
rsi_min = st.sidebar.number_input("RSI 저한", 0, 100, 45, step=1)
rsi_max = st.sidebar.number_input("RSI 상한", 0, 100, 65, step=1)
ma20_min = st.sidebar.number_input("MA20乖離 하한(%)", -50.0, 200.0, 0.0, step=0.5)
ma20_max = st.sidebar.number_input("MA20乖離 상한(%)", -50.0, 200.0, 10.0, step=0.5)
score_pass = st.sidebar.number_input("최소 통과점수(0~7)", 0, 7, 4, step=1)
macd_up_only = st.sidebar.checkbox("MACD 히스토그램 > 0 필수", True)

st.sidebar.subheader("🛡 리스크/실행")
acct_krw = st.sidebar.number_input("계좌 금액(원)", 0, 10_000_000_000, 30_000_000, step=1_000_000)
risk_pct = st.sidebar.number_input("트레이드당 리스크(%)", 0.1, 5.0, 1.0, step=0.1)
atr_mult = st.sidebar.number_input("손절 폭 (ATR배)", 0.5, 5.0, 1.5, step=0.1)
fee_bps = st.sidebar.number_input("수수료+슬리피지(베이시스포인트)", 0, 100, 10, step=1)  # 10bps=0.1%

use_prev_close = st.sidebar.checkbox("전일 기준(장마감 데이터)", True)
force_refresh  = st.sidebar.button("🔄 강제 새로고침")

blacklist = st.sidebar.text_area("블랙리스트(쉼표로 구분)", value="")
blk = [x.strip() for x in blacklist.split(",") if x.strip()]

st.write(f"🗓 기준일: {effective_ymd(use_prev_close)} | 프리셋: {preset} | Made by LDY")

if force_refresh:
    st.cache_data.clear()
    st.toast("캐시 초기화 완료", icon="✅")

# =========================
# 데이터 (샘플) — 실전은 pykrx로 치환
# 필요한 컬럼: 시장, 종목명, 종목코드, 날짜별 OHLCV(고가/저가/종가/거래량/거래대금), 시가총액
# =========================
@st.cache_data(ttl=1800)
def load_sample_ohlcv(lookback: int):
    # 최근 lookback일 더미 시계열 생성 (6개 종목)
    idx = pd.date_range(end=datetime.now(KST).date(), periods=lookback, freq="D")
    def mk(name, code, market, base=50000, vol=2e6, tnov=2000):
        np.random.seed(abs(hash(code)) % (10**6))
        close = pd.Series(base*(1+np.random.normal(0,0.01,lookback)).cumprod(), index=idx).clip(1000, None)
        high = close * (1 + np.random.uniform(0.005, 0.02, lookback))
        low  = close * (1 - np.random.uniform(0.005, 0.02, lookback))
        volu = pd.Series(np.random.normal(vol, vol*0.3, lookback)).clip(1e5, None).round()
        tnov_series = (close * volu / 1e8)  # 억원 단위 근사
        mcap = pd.Series(np.random.normal(20000, 5000, lookback)).clip(3000, None)  # 억원
        df = pd.DataFrame({
            "날짜": idx, "시장": market, "종목명": name, "종목코드": code,
            "종가": close.round(0), "고가": high.round(0), "저가": low.round(0),
            "거래량": volu, "거래대금(억원)": tnov_series.round(0), "시가총액(억원)": mcap.round(0)
        })
        return df

    dfs = [
        mk("LG전자","066570","KOSPI", base=95_000, vol=1.5e6, tnov=1500),
        mk("POSCO홀딩스","005490","KOSPI", base=550_000, vol=8e5, tnov=3500),
        mk("NAVER","035420","KOSPI", base=250_000, vol=1.2e6, tnov=1900),
        mk("에코프로","086520","KOSDAQ", base=700_000, vol=7e5, tnov=2800),
        mk("HLB","028300","KOSDAQ", base=120_000, vol=1.1e6, tnov=950),
        mk("한미사이언스","008930","KOSPI", base=40_000, vol=2.2e6, tnov=300),
    ]
    return pd.concat(dfs, ignore_index=True)

df_raw = load_sample_ohlcv(lookback)

# =========================
# 지표 계산
# =========================
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (mkt, code, name), g in df.groupby(["시장","종목코드","종목명"], as_index=False):
        g = g.sort_values("날짜").copy()
        g["MA20"] = g["종가"].rolling(20).mean()
        g["乖離%"] = (g["종가"] / (g["MA20"] + 1e-9) - 1.0) * 100.0
        g["RSI14"] = rsi(g["종가"], 14)
        g["MACD_hist"] = macd_hist(g["종가"])
        g["MACD_slope"] = g["MACD_hist"].diff()
        g["ATR14"] = atr(g["고가"], g["저가"], g["종가"], 14)
        g["Vol_Z"] = zscore(g["거래량"], 20)
        g["NR7"] = nr7(g["고가"], g["저가"], 7).astype(int)
        # 거래대금 퍼센타일(최근 60일 대비)
        g["Turnover_pct"] = g["거래대금(억원)"].rank(pct=True)
        out.append(g)
    return pd.concat(out, ignore_index=True)

df = df_raw[df_raw["시장"].isin(markets)].copy()
df = enrich(df)

# 최신일 스냅샷
snap = df.sort_values("날짜").groupby(["시장","종목코드","종목명"]).tail(1).copy()

# 하드필터(유동성·가격·시총·거래대금)
hard = (
    (snap["거래대금(억원)"] >= turnover_min) &
    (snap["종가"].between(price_min, price_max)) &
    (snap["시가총액(억원)"].between(mcap_min, mcap_max))
)
base = snap[hard].copy()

# Early Breakout Score (기본 5점)
base["EBS"] = 0
base.loc[base["MACD_hist"] > 0, "EBS"] += 1
base.loc[base["MACD_slope"] > 0, "EBS"] += 1
base.loc[base["RSI14"].between(rsi_min, rsi_max), "EBS"] += 1
base.loc[base["乖離%"].between(ma20_min, ma20_max), "EBS"] += 1
base.loc[base["Vol_Z"] >= vol_mult, "EBS"] += 1

# 보너스(+2): Turnover 상위, 변동성 축소
base.loc[base["Turnover_pct"] >= 0.70, "EBS"] += 1
# NR7 또는 ATR14가 과거 20일 하위 30%면 +1
# (스냅샷이라 단순화: NR7=1이면 +1)
base.loc[base["NR7"] == 1, "EBS"] += 1

if macd_up_only:
    base = base[base["MACD_hist"] > 0]

picked = base[base["EBS"] >= score_pass].copy()

# ===== 실행/리스크 =====
# 손절 = 종가 - (ATR14 * atr_mult)
fee = fee_bps / 10000.0
picked["손절단가"] = (picked["종가"] - picked["ATR14"] * atr_mult).clip(lower=1).round(0)
picked["손절폭"] = (picked["종가"] - picked["손절단가"]).clip(lower=1)

risk_amt = acct_krw * (risk_pct/100.0)
picked["추천수량"] = np.floor((risk_amt / (picked["손절폭"]*(1+fee))).clip(lower=0))
picked["예상투입"] = (picked["추천수량"] * picked["종가"]).round(0)

# R-멀티 목표가
picked["매수기준가"] = picked["종가"].round(0)  # 시장가 진입 가정(원하면 MA20 등으로 바꿔도 됨)
picked["1R"] = (picked["매수기준가"] + picked["손절폭"]*1).round(0)
picked["2R"] = (picked["매수기준가"] + picked["손절폭"]*2).round(0)
picked["3R"] = (picked["매수기준가"] + picked["손절폭"]*3).round(0)

# 보기 좋게 정렬
picked = picked.sort_values(["EBS","Turnover_pct","Vol_Z"], ascending=[False,False,False]).head(rec_count)

st.success(f"✅ 분석 완료! 추천 종목 {len(picked)}개 발견 (통과점수 ≥ {score_pass})")

disp_cols = [
    "시장","종목명","종목코드","종가","거래대금(억원)","시가총액(억원)",
    "RSI14","MACD_hist","MACD_slope","乖離%","Vol_Z","Turnover_pct","NR7",
    "ATR14","EBS",
    "매수기준가","손절단가","손절폭","1R","2R","3R",
    "추천수량","예상투입"
]
if picked.empty:
    st.warning("현재 조건에서 통과 종목 없음. (유동성 프리셋/점수/RSI/乖離 범위 조정 추천)")
else:
    st.dataframe(picked[disp_cols], use_container_width=True)

    # 다운로드 (CSV + 엑셀)
    csv_data = picked[disp_cols].to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 CSV 다운로드", data=csv_data,
                       file_name=f"swingpicker_pro_{effective_ymd(use_prev_close)}.csv",
                       mime="text/csv")

    try:
        import openpyxl  # ensure installed
        buf = io.BytesIO()
        picked[disp_cols].to_excel(buf, index=False)
        st.download_button("📊 엑셀(XLSX) 다운로드", data=buf.getvalue(),
                           file_name=f"swingpicker_pro_{effective_ymd(use_prev_close)}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.info("엑셀 저장을 쓰려면 requirements.txt에 `openpyxl` 추가하세요.")
