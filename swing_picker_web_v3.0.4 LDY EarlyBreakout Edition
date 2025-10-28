import streamlit as st
import pandas as pd
import numpy as np
import math, time, random, json, os, io
from datetime import datetime, timedelta, timezone
from pykrx import stock

# =========================
# 기본 세팅
# =========================
st.set_page_config(page_title="Swing Picker Web v3.0.4 (LDY EarlyBreakout Edition)", layout="wide")

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

st.title("📈 Swing Picker Web v3.0.4 • LDY EarlyBreakout Edition")
st.caption("급등 초입 캐치용 • 거래대금 + 기술지표 점수화 스캐너 | Made by **LDY**")

# =========================
# KST 기준 전일/금일 판단
# =========================
KST = timezone(timedelta(hours=9))
def get_effective_trade_date(use_prev_close: bool) -> str:
    now_kst = datetime.now(KST)
    today = now_kst.date()
    rollover = now_kst.replace(hour=9, minute=5, second=0, microsecond=0)
    base = (today - timedelta(days=1)) if (use_prev_close or now_kst < rollover) else today
    return base.strftime("%Y%m%d")

# =========================
# Sidebar 조건 패널 (개선)
# =========================
def build_sidebar():
    st.sidebar.header("⚙️ 스캔 조건 (Made by LDY)")

    # 시장 선택: 보기 깔끔하게 라디오
    market_choice = st.sidebar.radio(
        "시장",
        options=["KOSPI", "KOSDAQ", "KOSPI+KOSDAQ"],
        index=2,
        horizontal=True
    )
    if market_choice == "KOSPI":
        markets = ["KOSPI"]
    elif market_choice == "KOSDAQ":
        markets = ["KOSDAQ"]
    else:
        markets = ["KOSPI", "KOSDAQ"]

    # 조회일수 & 추천개수
    colA, colB = st.sidebar.columns(2)
    lookback = colA.number_input("조회일수(LOOKBACK)", 5, 252, 30, step=1)
    rec_count = colB.number_input("추천 종목 수", 1, 200, 10, step=1)

    st.sidebar.divider()

    # 거래대금 프리셋 (개잡주 배제/중형/대형)
    preset = st.sidebar.selectbox(
        "유동성 프리셋",
        ["개잡주 배제 (50억↑)", "중형주 중심 (100억↑)", "대형주 중심 (300억↑)"],
        index=0
    )
    preset_map = {
        "개잡주 배제 (50억↑)": 50,
        "중형주 중심 (100억↑)": 100,
        "대형주 중심 (300억↑)": 300,
    }
    default_turnover = preset_map[preset]

    st.sidebar.subheader("📊 가격/시총/거래대금")
    col1, col2 = st.sidebar.columns(2)
    price_min = col1.number_input("가격 ≥ (원)", 0, 1_000_000_000, 1_000, step=100)
    price_max = col2.number_input("가격 ≤ (원)", 0, 1_000_000_000, 1_000_000, step=1000)

    col3, col4 = st.sidebar.columns(2)
    mcap_min = col3.number_input("시가총액 ≥ (억원)", 0, 10_000_000, 1_000, step=10)
    mcap_max = col4.number_input("시가총액 ≤ (억원)", 0, 10_000_000, 10_000_000, step=10)

    col5, col6 = st.sidebar.columns(2)
    turnover_min = col5.number_input("거래대금 ≥ (억원)", 0, 10_000_000, default_turnover, step=10)
    vol_multiple = col6.number_input("거래량배수 ≥", 0.1, 50.0, 1.20, step=0.05)

    st.sidebar.subheader("📈 기술지표 한계")
    # 급등 '초입' 느낌: 과열 방지 상한만 두고, 하한은 느슨
    col7, col8 = st.sidebar.columns(2)
    rr5_max  = col7.number_input("5일 수익률 ≤ %", -100.0, 200.0, 20.0, step=0.5)
    rr10_max = col8.number_input("10일 수익률 ≤ %", -100.0, 300.0, 35.0, step=0.5)

    col9, col10 = st.sidebar.columns(2)
    ma20_dev_min = col9.number_input("MA20乖離 ≥ %", -50.0, 200.0, -5.0, step=0.5)
    ma20_dev_max = col10.number_input("MA20乖離 ≤ %", -50.0, 200.0, 10.0, step=0.5)

    col11, col12 = st.sidebar.columns(2)
    rsi_min = col11.number_input("RSI14 ≥", 0.0, 100.0, 35.0, step=1.0)
    rsi_max = col12.number_input("RSI14 ≤", 0.0, 100.0, 80.0, step=1.0)

    macd_positive = st.sidebar.checkbox("MACD 히스토그램 > 0", True)

    st.sidebar.subheader("🚫 제외 규칙")
    ex_warn     = st.sidebar.checkbox("관리/거래정지/우선주/스팩/리츠 제외", True)
    ex_limit_up = st.sidebar.checkbox("상한가/근접 제외", True)
    ex_limit_dn = st.sidebar.checkbox("하한가/근접 제외", True)

    st.sidebar.subheader("🧰 기타")
    use_prev_close = st.sidebar.checkbox("전일 기준(장 마감 데이터 기준)", True)
    force_refresh  = st.sidebar.button("🔄 강제 새로고침")

    st.sidebar.caption("💡 거래대금 상위 N은 내부에서 자동 조절 (유동성 프리셋에 따라 300~600 탐색).")

    blacklist = st.sidebar.text_area("블랙리스트(쉼표로 구분)", value="")
    blk = [x.strip() for x in blacklist.split(",") if x.strip()]

    return {
        "markets": markets,
        "market_choice": market_choice,
        "lookback": lookback,
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
        "ex_warn": ex_warn,
        "ex_limit_up": ex_limit_up,
        "ex_limit_dn": ex_limit_dn,
        "use_prev_close": use_prev_close,
        "force_refresh": force_refresh,
        "blacklist": blk,
        "preset": preset,
    }

# =========================
# Data Load (데모용 샘플)
# =========================
@st.cache_data(ttl=1800)
def load_sample_data(effective_ymd: str, markets: list[str], lookback: int):
    """
    실제 배포에선 pykrx로 치환.
    여기선 데모용 DF 컬럼 스펙만 맞춰둠.
    """
    data = {
        "시장": ["KOSDAQ","KOSDAQ","KOSPI","KOSPI","KOSDAQ","KOSPI"],
        "종목명": ["한미사이언스","HLB","LG전자","POSCO홀딩스","에코프로","NAVER"],
        "종목코드": ["008930","028300","066570","005490","086520","035420"],
        "현재가": [40900,122000,93500,558000,707000,255000],
        "거래대금(억원)": [300,950,1120,3500,2800,1900],
        "거래량배수": [3.2,1.8,2.4,1.3,1.6,1.25],
        "5일수익률%": [7.3,-3.2,2.5,12.0,18.0,5.5],
        "10일수익률%":[11.9,-5.4,4.1,22.0,30.0,9.2],
        "MA20乖離%":[6.7,-1.4,3.8,4.0,9.5,1.2],
        "RSI14":[61.9,44.2,57.3,66.0,69.0,55.0],
        "MACD_hist":[0.8,-0.3,0.2,0.9,1.1,0.4],  # 단위 스케일만 맞춤
        "시가총액(억원)":[12000,22000,17000,490000,190000,420000],
        # 아래 두 컬럼은 점수화에서 있으면 가점, 없으면 무시
        "RSI_slope":[+0.8,-0.2,+0.3,+0.6,+0.5,+0.2],       # RSI 증감 추세(최근-과거)
        "MACD_slope":[+0.1,-0.05,+0.02,+0.12,+0.15,+0.03],  # MACD 히스토그램 변화량
    }
    return pd.DataFrame(data)

# =========================
# 필터 & 스코어 엔진 (Early Breakout)
# =========================
def early_breakout_picker(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # 0) 시장, 블랙리스트 1차 필터
    df = df[df["시장"].isin(cfg["markets"])].copy()
    if cfg["blacklist"]:
        df = df[~(df["종목명"].isin(cfg["blacklist"]) | df["종목코드"].isin(cfg["blacklist"]))]

    # 1) 하드 필터 (개잡주/과열 제거의 최소선)
    hard = (
        (df["거래대금(억원)"] >= cfg["turnover_min"]) &
        (df["현재가"].between(cfg["price_min"], cfg["price_max"])) &
        (df["시가총액(억원)"].between(cfg["mcap_min"], cfg["mcap_max"])) &
        (df["거래량배수"] >= cfg["vol_multiple"]) &
        (df["5일수익률%"] <= cfg["rr5_max"]) &
        (df["10일수익률%"] <= cfg["rr10_max"]) &
        (df["MA20乖離%"].between(cfg["ma20_dev_min"], cfg["ma20_dev_max"])) &
        (df["RSI14"].between(cfg["rsi_min"], cfg["rsi_max"]))
    )
    if cfg["macd_positive"]:
        hard &= (df["MACD_hist"] > 0)
    base = df[hard].copy()
    if base.empty:
        return base

    # 2) 거래대금 상위 N(프리셋에 따라 탐색폭 확장)
    #    초입 캐치를 위해 상위 300~600까지는 열어둠
    top_span = 300 if cfg["preset"].startswith("개잡주") else (500 if "중형" in cfg["preset"] else 600)
    base = base.sort_values("거래대금(억원)", ascending=False).head(top_span)

    # 3) 점수화 (5점 만점, 3점 이상 통과)
    #    - MACD_hist > 0 : +1
    #    - MACD_slope > 0 : +1 (초입 가속감)
    #    - RSI14 45~65 : +1 (과열 전 박스 상단 돌파)
    #    - MA20乖離 0~10 : +1 (20MA 위 양호한 탄력)
    #    - 5일수익률 -2~20 : +1 (음봉 탈락 방지 + 과열 방지)
    base["score"] = 0
    base.loc[base["MACD_hist"] > 0, "score"] += 1
    if "MACD_slope" in base.columns:
        base.loc[base["MACD_slope"] > 0, "score"] += 1
    base.loc[base["RSI14"].between(45, 65), "score"] += 1
    base.loc[base["MA20乖離%"].between(0, 10), "score"] += 1
    base.loc[base["5일수익률%"].between(-2, cfg["rr5_max"]), "score"] += 1

    picked = base[base["score"] >= 3].copy()
    if picked.empty:
        return picked

    # 4) 정렬: 점수 ↓, 5일수익률(낮을수록 초입) ↑, 거래대금 ↓
    picked = picked.sort_values(
        by=["score", "5일수익률%", "거래대금(억원)"],
        ascending=[False, True, False]
    )

    # 5) 최종 추천 수 제한
    return picked.head(int(cfg["rec_count"]))

# =========================
# 메인 실행
# =========================
cfg = build_sidebar()
effective_ymd = get_effective_trade_date(cfg["use_prev_close"])
st.write(f"🗓 기준일: {effective_ymd} | 데이터소스: pykrx | Made by **LDY**")

if cfg["force_refresh"]:
    st.cache_data.clear()
    st.toast("🔄 캐시 강제 초기화 완료!", icon="✅")

with st.spinner("데이터 수집 및 분석 중... (약 1~2분)"):
    df_all = load_sample_data(effective_ymd, cfg["markets"], cfg["lookback"])
    picked = early_breakout_picker(df_all, cfg)
    time.sleep(0.5)

st.success(f"✅ 분석 완료! 추천 종목 {len(picked)}개 발견")

# 표 + 다운로드
if not picked.empty:
    st.dataframe(
        picked[
            ["시장","종목명","종목코드","현재가","거래대금(억원)","거래량배수",
             "5일수익률%","10일수익률%","MA20乖離%","RSI14","MACD_hist","시가총액(억원)","score"]
        ],
        use_container_width=True
    )
    # CSV 다운로드
    csv_data = picked.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 결과 CSV 다운로드",
        data=csv_data,
        file_name=f"swingpicker_{effective_ymd}.csv",
        mime="text/csv",
        help="추천 종목 리스트를 CSV로 저장합니다."
    )
    # 엑셀 다운로드
    buffer = io.BytesIO()
    picked.to_excel(buffer, index=False)
    st.download_button(
        label="📊 결과 엑셀(XLSX) 다운로드",
        data=buffer.getvalue(),
        file_name=f"swingpicker_{effective_ymd}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="추천 종목 리스트를 엑셀 파일로 저장합니다."
    )
else:
    st.warning("⚠️ 현재 조건에서 추천 결과가 없습니다. (프리셋/LOOKBACK/거래량배수 조정 권장)")
