# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.1.0 (Auto Update Viewer)
- GitHub raw CSV를 우선 로드, 실패 시 로컬 data/recommend_latest.csv 폴백
- 종목명/종목코드/추천가/EBS/근거 등을 바로 표로 표시
- '초입 후보만' 필터(EBS≥4), 거래대금 하한, 정렬/검색/TopN 제공
"""

import os
import io
import requests
import pandas as pd
import streamlit as st

# ------------------------- 기본 설정 -------------------------
st.set_page_config(
    page_title="LDY Pro Trader v3.1.0 (Auto Update)",
    layout="wide",
)
st.title("📈 LDY Pro Trader v3.1.0 (Auto Update)")
st.caption("매일 장마감 후 자동 업데이트되는 스윙 추천 종목 리스트 | Made by LDY")

# GitHub Raw CSV (본인 레포 경로에 맞게 필요시 수정)
RAW_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_PATH = "data/recommend_latest.csv"
PASS_SCORE = 4  # '초입' 기준 EBS 점수

# ------------------------- 로딩 유틸 -------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_remote_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    # raw는 UTF-8-SIG 아닐 수 있으니 pandas에게 맡김
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=300, show_spinner=False)
def load_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """표시에 필요한 컬럼이 없으면 안전하게 추가."""
    need_cols = [
        "시장","종목명","종목코드","종가","거래대금(억원)","시가총액(억원)",
        "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%",
        "EBS","통과","근거","추천매수가","추천매도가1","추천매도가2","손절가"
    ]
    for c in need_cols:
        if c not in df.columns:
            df[c] = None
    # 숫자형 캐스팅(정렬/필터 안정화)
    num_cols = [
        "종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope",
        "Vol_Z","ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------- 데이터 로드 -------------------------
src_msg = ""
df = None
try:
    df = load_remote_csv(RAW_URL)
    src_msg = f"✅ 데이터 로드: remote\n{RAW_URL}"
except Exception:
    if os.path.exists(LOCAL_PATH):
        df = load_local_csv(LOCAL_PATH)
        src_msg = f"✅ 데이터 로드: local\n{LOCAL_PATH}"
    else:
        st.error("❌ CSV를 찾을 수 없습니다. collector가 daily CSV를 생성했는지 확인하세요.")
        st.stop()

st.info(src_msg)
df = ensure_columns(df)

stamp = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M")
st.success(f"📅 추천 기준: {stamp} · 총 {len(df):,}개")

# ------------------------- 필터/정렬 UI -------------------------
with st.expander("🔍 보기/필터", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
    with c1:
        only_entry = st.checkbox("🚀 초입 후보만 (EBS≥4)", value=True)
    with c2:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c3:
        sort_key = st.selectbox(
            "정렬",
            ["EBS▼", "거래대금▼", "시가총액▼", "RSI▲", "RSI▼", "종가▲", "종가▼"],
            index=0
        )
    with c4:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)
    with c5:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

# ------------------------- 필터 적용 -------------------------
view = df.copy()

if only_entry:
    view = view[view["EBS"] >= PASS_SCORE]

view = view[view["거래대금(억원)"] >= float(min_turn)]

if q_text:
    q_low = q_text.strip().lower()
    # 부분 일치(종목명, 종목코드)
    name_hit = view["종목명"].fillna("").astype(str).str.lower().str.contains(q_low, na=False)
    code_hit = view["종목코드"].fillna("").astype(str).str.contains(q_low, na=False)
    view = view[name_hit | code_hit]

# 정렬
if sort_key == "EBS▼":
    view = view.sort_values(["EBS","거래대금(억원)"], ascending=[False, False])
elif sort_key == "거래대금▼":
    view = view.sort_values("거래대금(억원)", ascending=False)
elif sort_key == "시가총액▼":
    view = view.sort_values("시가총액(억원)", ascending=False, na_position="last")
elif sort_key == "RSI▲":
    view = view.sort_values("RSI14", ascending=True, na_position="last")
elif sort_key == "RSI▼":
    view = view.sort_values("RSI14", ascending=False, na_position="last")
elif sort_key == "종가▲":
    view = view.sort_values("종가", ascending=True, na_position="last")
elif sort_key == "종가▼":
    view = view.sort_values("종가", ascending=False, na_position="last")

# ------------------------- 표 출력 -------------------------
show_cols = [
    "통과","시장","종목명","종목코드",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "거래대금(억원)","시가총액(억원)",
    "EBS","근거",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
view = view[show_cols].head(int(topn))

st.dataframe(view, use_container_width=True, height=640)

# ------------------------- 다운로드 -------------------------
st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view.to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_entry_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ EBS 구성(급등 초입 로직)", expanded=False):
    st.markdown(
        """
- 기본 컷: 거래대금 ≥ **50억원**, 시가총액 ≥ **1,000억원** (collector에서 적용)
- 점수(0~7):
  1) RSI 45~65  
  2) MACD 히스토그램 기울기 > 0  
  3) 종가가 MA20 근처(-1%~+4%)  
  4) 상대거래량(20일) > 1.2  
  5) MA20 > MA60(상승 구조)  
  6) MACD 히스토그램 > 0  
  7) 5일 수익률 < 10%(과열 방지)  
- **통과(🚀초입)**: EBS ≥ 4  
- 추천가: ATR/MA 기반 보수적 가이드 (투자 판단 책임 본인)
        """
    )
