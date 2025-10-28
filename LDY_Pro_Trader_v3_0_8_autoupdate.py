# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.0.8 (Auto Update Viewer)
- daily_output/recommend_YYYYMMDD.csv 중 가장 최신 파일을 자동 로드하여 표시
- Cloud에서는 데이터 '표시 전용'만 담당 (수집/계산은 collector.py가 수행)
"""

import os
import glob
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LDY Pro Trader v3.0.8 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.0.8 (Auto Update)")
st.caption("매일 장마감 후 자동 업데이트되는 스윙 추천 종목 리스트 | Made by LDY")

DATA_DIR = "daily_output"

@st.cache_data(ttl=300)
def find_latest_file():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "recommend_*.csv")), reverse=True)
    return files[0] if files else None

@st.cache_data(ttl=300)
def load_csv(path: str):
    return pd.read_csv(path)

latest = find_latest_file()

colL, colR = st.columns([7,3])
with colR:
    st.info("Cloud는 표시전용입니다.\n\n실데이터 수집은 로컬/서버의 collector.py가 담당합니다.")

if latest is None:
    st.error("❌ 추천 데이터가 없습니다.\n\n로컬/서버에서 collector.py가 실행되어 CSV가 생성되어야 합니다.")
else:
    date_str = os.path.splitext(os.path.basename(latest))[0].split("_")[1]
    df = load_csv(latest)

    st.success(f"📅 추천 기준일: {date_str} · 총 {len(df)}개")
    # 뷰어 필터(가벼운 보기용)
    with st.expander("🔍 보기 필터"):
        min_ebs = st.slider("최소 EBS 점수", 0, 7, 4)
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
        df_view = df[(df["EBS"] >= min_ebs) & (df["거래대금(억원)"] >= min_turn)].copy()

    show_cols = [
        "시장","종목명","종목코드","종가","거래대금(억원)","시가총액(억원)",
        "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%",
        "EBS","추천매수가","추천매도가1","추천매도가2","손절가"
    ]
    st.dataframe(df_view[show_cols], use_container_width=True, height=560)

    # 다운로드
    st.download_button(
        "📥 전체 CSV 다운로드",
        data=df.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"recommend_{date_str}.csv",
        mime="text/csv"
    )

    st.caption("※ 매수/매도/손절 가격은 ATR과 MA20 기반의 보수적 가이드입니다(투자판단 책임 본인).")
