import streamlit as st
import pandas as pd
import numpy as np
import math, time, random, json, os
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pykrx import stock

# =========================
# 기본 세팅
# =========================
st.set_page_config(page_title="Swing Picker Web v3.0.4 (LDY Full Sync)", layout="wide")

GA_MEASUREMENT_ID = "G-3PLRGRT2RL"
st.markdown(f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_MEASUREMENT_ID}');
</script>
""", unsafe_allow_html=True)

st.title("📈 Swing Picker Web v3.0.4 (LDY FullSync)")
st.caption("거래대금 + 기술지표 기반 자동 스윙 종목 추천기 | Made by **LDY**")

# =========================
# KST 기준 전일/금일 판단
# =========================
KST = timezone(timedelta(hours=9))
def get_effective_trade_date(use_prev_close: bool) -> str:
    now_kst = datetime.now(KST)
    today = now_kst.date()
    rollover = now_kst.replace(hour=9, minute=5, second=0, microsecond=0)
    if use_prev_close or (now_kst < rollover):
        base = today - timedelta(days=1)
    else:
        base = today
    return base.strftime("%Y%m%d")

# =========================
# Sidebar 조건 패널
# =========================
def build_sidebar():
    st.sidebar.header("⚙️ 스캔 조건 (Made by LDY)")

    colA, colB = st.sidebar.columns(2)
    markets = colA.multiselect("시장", ["KOSPI","KOSDAQ"], default=["KOSPI","KOSDAQ"])
    lookback = colB.number_input("조회일수(LOOKBACK)", 30, 252, 63, step=1)

    col1, col2 = st.sidebar.columns(2)
    top_n_turnover = col1.number_input("거래대금 상위 N", 20, 2000, 120, step=10)
    rec_count      = col2.number_input("추천 종목 수", 1, 200, 10, step=1)

    st.sidebar.divider()
    st.sidebar.subheader("📊 가격/시총/거래대금")
    col3, col4 = st.sidebar.columns(2)
    price_min = col3.number_input("가격 ≥ (원)", 0, 1_000_000_000, 1000, step=100)
    price_max = col4.number_input("가격 ≤ (원)", 0, 1_000_000_000, 1_000_000, step=1000)

    col5, col6 = st.sidebar.columns(2)
    mcap_min = col5.number_input("시가총액 ≥ (억원)", 0, 10_000_000, 1000, step=10)
    mcap_max = col6.number_input("시가총액 ≤ (억원)", 0, 10_000_000, 10_000_000, step=10)

    col7, col8 = st.sidebar.columns(2)
    turnover_min = col7.number_input("거래대금 ≥ (억원)", 0, 10_000_000, 50, step=10)
    vol_multiple = col8.number_input("거래량배수 ≥", 0.1, 50.0, 1.50, step=0.05)

    st.sidebar.subheader("📈 기술지표 한계")
    col9, col10 = st.sidebar.columns(2)
    rr5_max  = col9.number_input("5일 수익률 ≤ %", -100.0, 200.0, 8.0, step=0.5)
    rr10_max = col10.number_input("10일 수익률 ≤ %", -100.0, 300.0, 15.0, step=0.5)

    col11, col12 = st.sidebar.columns(2)
    ma20_dev_min = col11.number_input("MA20乖離 ≥ %", -50.0, 200.0, -5.0, step=0.5)
    ma20_dev_max = col12.number_input("MA20乖離 ≤ %", -50.0, 200.0, 10.0, step=0.5)

    col13, col14 = st.sidebar.columns(2)
    rsi_min = col13.number_input("RSI14 ≥", 0.0, 100.0, 40.0, step=1.0)
    rsi_max = col14.number_input("RSI14 ≤", 0.0, 100.0, 75.0, step=1.0)

    macd_positive = st.sidebar.checkbox("MACD 히스토그램 > 0", True)
    hard_drop_5d  = st.sidebar.number_input("급락 기준 (5일 수익률 < %)", -50.0, 0.0, -10.0, step=0.5)

    st.sidebar.subheader("🚫 제외 규칙")
    ex_gap_up   = st.sidebar.checkbox("당일 갭상승 종목 제외", True)
    ex_gap_down = st.sidebar.checkbox("당일 갭하락 종목 제외", False)
    ex_limit_up = st.sidebar.checkbox("상한가/근접 제외", True)
    ex_limit_dn = st.sidebar.checkbox("하한가/근접 제외", True)
    ex_warn     = st.sidebar.checkbox("관리/거래정지/우선주/스팩/리츠 제외", True)

    st.sidebar.subheader("🧰 기타")
    use_prev_close = st.sidebar.checkbox("전일 기준(장 마감 데이터 기준)", True)
    force_refresh  = st.sidebar.button("🔄 강제 새로고침")

    blacklist = st.sidebar.text_area("블랙리스트(쉼표로 구분)", value="")
    blk = [x.strip() for x in blacklist.split(",") if x.strip()]

    return {
        "markets": markets,
        "lookback": lookback,
        "top_n_turnover": top_n_turnover,
        "rec_count": rec_count,
        "price_min": price_min,
        "price_max": price_max,
        "mcap_min": mcap_min,
        "mcap_max": mcap_max,
        "turnover_min": turnover_min,
        "vol_multiple": vol_multiple,
        "rr5_max": rr5_max,
        "rr10_max": rr10_max,
        "ma20_dev_min": ma20_dev_min,
        "ma20_dev_max": ma20_dev_max,
        "rsi_min": rsi_min,
        "rsi_max": rsi_max,
        "macd_positive": macd_positive,
        "hard_drop_5d": hard_drop_5d,
        "ex_gap_up": ex_gap_up,
        "ex_gap_down": ex_gap_down,
        "ex_limit_up": ex_limit_up,
        "ex_limit_dn": ex_limit_dn,
        "ex_warn": ex_warn,
        "use_prev_close": use_prev_close,
        "force_refresh": force_refresh,
        "blacklist": blk,
    }

# =========================
# Data Load (샘플)
# =========================
@st.cache_data(ttl=1800)
def load_sample_data(effective_ymd: str):
    """데모용 샘플 데이터"""
    data = {
        "종목명": ["한미사이언스", "HLB", "LG전자"],
        "종목코드": ["008930", "028300", "066570"],
        "현재가": [40900, 122000, 93500],
        "거래대금(억원)": [300, 950, 1120],
        "거래량배수": [3.2, 1.8, 2.4],
        "5일수익률%": [7.35, -3.2, 2.5],
        "10일수익률%": [11.9, -5.4, 4.1],
        "MA20乖離%": [6.72, -1.4, 3.8],
        "RSI14": [61.9, 44.2, 57.3],
        "MACD_hist": [279.9, -50.2, 10.5],
        "시가총액(억원)": [12000, 22000, 17000],
    }
    return pd.DataFrame(data)

# =========================
# 필터 엔진
# =========================
def apply_filters(df, cfg):
    q = (df["거래대금(억원)"] >= cfg["turnover_min"]) \
        & (df["거래량배수"] >= cfg["vol_multiple"]) \
        & (df["5일수익률%"] <= cfg["rr5_max"]) \
        & (df["10일수익률%"] <= cfg["rr10_max"]) \
        & (df["MA20乖離%"].between(cfg["ma20_dev_min"], cfg["ma20_dev_max"])) \
        & (df["RSI14"].between(cfg["rsi_min"], cfg["rsi_max"])) \
        & (df["현재가"].between(cfg["price_min"], cfg["price_max"])) \
        & (df["시가총액(억원)"].between(cfg["mcap_min"], cfg["mcap_max"]))

    if cfg["macd_positive"]:
        q &= (df["MACD_hist"] > 0)

    if cfg["blacklist"]:
        q &= ~(df["종목명"].isin(cfg["blacklist"]) | df["종목코드"].isin(cfg["blacklist"]))

    df_top = df.sort_values("거래대금(억원)", ascending=False).head(int(cfg["top_n_turnover"]))
    picked = df_top[q].copy()

    if {"5일수익률%","거래대금(억원)"}.issubset(picked.columns):
        picked = picked.sort_values(["5일수익률%","거래대금(억원)"], ascending=[True,False])

    return picked.head(int(cfg["rec_count"]))

# =========================
# Main
# =========================
cfg = build_sidebar()
effective_ymd = get_effective_trade_date(cfg["use_prev_close"])
st.write(f"🗓 기준일: {effective_ymd} | 데이터소스: pykrx | Made by **LDY**")

if cfg["force_refresh"]:
    st.cache_data.clear()
    st.toast("🔄 캐시 강제 초기화 완료!", icon="✅")

with st.spinner("데이터 수집 및 분석 중... (약 1~3분 소요)"):
    df = load_sample_data(effective_ymd)
    picked = apply_filters(df, cfg)
    time.sleep(1)

st.success(f"✅ 분석 완료! 추천 종목 {len(picked)}개 발견")
st.dataframe(picked, use_container_width=True)
