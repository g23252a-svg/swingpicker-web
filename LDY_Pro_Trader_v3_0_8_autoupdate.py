# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.0.8 (Auto Update Viewer)
- GitHub Actions가 커밋한 data/recommend_latest.csv(.gz)를 원격에서 즉시 로드
- Cloud는 '표시 전용' (수집·계산은 collector.py + Actions가 수행)
"""

import os
from io import BytesIO
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

# ---------------------------
# 기본 세팅
# ---------------------------
st.set_page_config(page_title="LDY Pro Trader v3.0.8 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.0.8 (Auto Update)")
st.caption("매일 장마감 후 자동 업데이트되는 스윙 추천 종목 리스트 | Made by LDY")

# ✅ 네 레포 RAW 경로 (필수)
RAW_BASE = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data"

KST = timezone(timedelta(hours=9))

# ---------------------------
# 유틸 & 로딩
# ---------------------------
@st.cache_data(ttl=300)
def read_remote_csv(basename: str = "recommend_latest"):
    """
    GitHub RAW에서 gz → csv 순으로 시도.
    실패 시, 로컬 data/ 폴더 파일로 폴백.
    반환: (df, source, used_path)
    """
    # 1) 원격 우선
    for ext, kwargs in [
        (".csv.gz", dict(compression="gzip")),
        (".csv", dict())
    ]:
        url = f"{RAW_BASE}/{basename}{ext}"
        try:
            df = pd.read_csv(url, low_memory=False, **kwargs)
            return df, "remote", url
        except Exception:
            pass

    # 2) 로컬 폴백 (개발용)
    for ext, kwargs in [
        (".csv.gz", dict(compression="gzip")),
        (".csv", dict())
    ]:
        path = os.path.join("data", f"{basename}{ext}")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, low_memory=False, **kwargs)
                return df, "local", path
            except Exception:
                pass

    return None, None, None


def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_krw_conversions(df: pd.DataFrame):
    """
    거래대금/시총 컬럼이 (억원)으로 없으면, 원 단위를 억원으로 변환해서 생성.
    """
    # 거래대금
    if "거래대금(억원)" not in df.columns:
        if "거래대금" in df.columns:
            df["거래대금(억원)"] = (pd.to_numeric(df["거래대금"], errors="coerce") / 1e8).round(2)
        elif "거래대금(원)" in df.columns:
            df["거래대금(억원)"] = (pd.to_numeric(df["거래대금(원)"], errors="coerce") / 1e8).round(2)

    # 시가총액
    if "시가총액(억원)" not in df.columns:
        if "시가총액" in df.columns:
            df["시가총액(억원)"] = (pd.to_numeric(df["시가총액"], errors="coerce") / 1e8).round(2)
        elif "시가총액(원)" in df.columns:
            df["시가총액(억원)"] = (pd.to_numeric(df["시가총액(원)"], errors="coerce") / 1e8).round(2)

    return df


def pick_existing(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]


# ---------------------------
# 데이터 로드
# ---------------------------
df, src, used_path = read_remote_csv("recommend_latest")

right = st.sidebar if st.sidebar else st
with right:
    st.markdown("### 상태")
    if src is None:
        st.error("❌ 추천 데이터가 없습니다.\n\ncollector.py + GitHub Actions가 `data/recommend_latest.csv`를 커밋해야 합니다.")
    else:
        st.success(f"✅ 데이터 로드: **{src}**\n\n`{used_path}`")
    if st.button("♻️ 강제 새로고침"):
        st.cache_data.clear()
        st.rerun()

if df is None:
    st.stop()

# ---------------------------
# 전처리(타입/결측/표준화)
# ---------------------------
df = add_krw_conversions(df)

# 숫자형 전환(가능한 것들)
numeric_candidates = [
    "종가", "거래대금(억원)", "시가총액(억원)",
    "RSI14", "乖離%", "MACD_hist", "MACD_slope",
    "Vol_Z", "ret_5d_%", "ret_10d_%", "EBS",
    "추천매수가", "추천매도가1", "추천매도가2", "손절가"
]
df = ensure_numeric(df, numeric_candidates)

# 추천 기준일 추정(있으면 표시)
basis = None
for cand in ["기준일", "date", "DATE"]:
    if cand in df.columns:
        basis = str(df[cand].iloc[0])
        break
if basis is None:
    basis = datetime.now(KST).strftime("%Y-%m-%d %H:%M")

# ---------------------------
# 필터 UI
# ---------------------------
with st.expander("🔍 보기/필터", expanded=True):
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])
    with c1:
        min_ebs = st.slider("최소 EBS", 0, 7, 4)
    with c2:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c3:
        sort_key = st.selectbox("정렬", ["EBS▼", "거래대금▼", "시가총액▼"])
    with c4:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)

    mask = pd.Series([True] * len(df))
    if "EBS" in df.columns:
        mask &= (df["EBS"] >= min_ebs)
    if "거래대금(억원)" in df.columns:
        mask &= (df["거래대금(억원)"] >= min_turn)

    view = df[mask].copy()

    if sort_key == "EBS▼" and "EBS" in view.columns:
        view = view.sort_values("EBS", ascending=False)
    elif sort_key == "거래대금▼" and "거래대금(억원)" in view.columns:
        view = view.sort_values("거래대금(억원)", ascending=False)
    elif sort_key == "시가총액▼" and "시가총액(억원)" in view.columns:
        view = view.sort_values("시가총액(억원)", ascending=False)

    view = view.head(topn).reset_index(drop=True)

# ---------------------------
# 테이블 표시
# ---------------------------
st.success(f"📅 추천 기준: {basis} · 총 {len(df):,}개 / 표시 {len(view):,}개")

base_cols = ["시장", "종목명", "종목코드", "종가", "거래대금(억원)", "시가총액(억원)"]
factor_cols = ["EBS", "RSI14", "乖離%", "MACD_hist", "MACD_slope", "Vol_Z", "ret_5d_%", "ret_10d_%"]
plan_cols = ["추천매수가", "추천매도가1", "추천매도가2", "손절가"]

show_cols = pick_existing(view, base_cols + factor_cols + plan_cols)
if not show_cols:
    st.warning("표시할 수 있는 표준 컬럼이 없습니다. CSV에 최소한의 컬럼(시장, 종목명/코드, 종가, 거래대금/시총, EBS 등)을 포함해 주세요.")
else:
    st.dataframe(view[show_cols], use_container_width=True, height=620)

# ---------------------------
# 다운로드
# ---------------------------
cL, cR = st.columns([1, 1])
with cL:
    st.download_button(
        "📥 (현재 필터) CSV 다운로드",
        data=view.to_csv(index=False, encoding="utf-8-sig"),
        file_name="recommend_filtered.csv",
        mime="text/csv",
    )
with cR:
    st.download_button(
        "📥 (원본 전체) CSV 다운로드",
        data=df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="recommend_latest.csv",
        mime="text/csv",
    )

st.caption("※ 매수/매도/손절 가격은 ATR/MA 기반의 보수적 가이드입니다. 투자 판단의 책임은 본인에게 있습니다.")
