# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.4 (Auto Update)
- 추천 CSV 원격 로드 → 이름맵 보강 → EV_SCORE 계산 → Top Picks 필터/정렬 제공
- NumberColumn 타입과 데이터 dtype을 엄격하게 맞춰 Streamlit 오류 방지
"""

import os, io, math, requests, numpy as np, pandas as pd, streamlit as st
from datetime import datetime

RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"

PASS_SCORE = 4

st.set_page_config(page_title="LDY Pro Trader v3.4 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.4 (Auto Update)")
st.caption("매일 장마감 후 자동 업데이트되는 스윙 추천 종목 리스트 | EV스코어·TopPick 내장")

# ---------------- IO ----------------
@st.cache_data(ttl=300)
def load_csv_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30); r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=300)
def load_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def log_src(df: pd.DataFrame, src: str, url_or_path: str):
    st.info(f"상태 ✅ 데이터 로드: {src}\n\n{url_or_path}")
    st.success(f"📅 표시시각: {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M')} · 행수: {len(df):,}")

# -------- name map (optional, 있으면 사용) --------
def z6(x) -> str:
    s = str(x); return s.zfill(6) if s.isdigit() else s

@st.cache_data(ttl=6*60*60)
def try_load_name_map() -> pd.DataFrame|None:
    for src, path in [("remote", CODES_URL), ("local", LOCAL_MAP)]:
        try:
            m = load_csv_url(path) if src=="remote" else load_csv_path(path)
            if {"종목코드","종목명"}.issubset(m.columns):
                m["종목코드"] = m["종목코드"].astype(str).map(z6)
                return m[["종목코드","종목명"]].drop_duplicates("종목코드")
        except Exception:
            pass
    return None

def apply_names(df: pd.DataFrame) -> pd.DataFrame:
    mp = try_load_name_map()
    if mp is not None:
        df["종목코드"] = df["종목코드"].astype(str).map(z6)
        if "종목명" not in df.columns: df["종목명"] = None
        df = df.merge(mp, on="종목코드", how="left", suffixes=("","_map"))
        df["종목명"] = df["종목명"].fillna(df["종목명_map"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_map")], errors="ignore")
    df["종목명"] = df["종목명"].fillna("(이름없음)")
    return df

# -------- EV_SCORE 계산 --------
def compute_evs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 필요한 컬럼이 없으면 계산 skip
    need = ["EBS","추천매수가","손절가","추천매도가1","종가","거래대금(억원)"]
    if not set(need).issubset(out.columns):
        for c in need:
            if c not in out.columns: out[c] = np.nan

    # 숫자 캐스팅
    num_cols = ["EBS","추천매수가","손절가","추천매도가1","추천매도가2","종가","거래대금(억원)",
                "NOW_ENTRY_%","STOP_BUF_%","T1_BUF_%","MIN_RR","NOW_TICKS"]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # collector가 계산했을 수도 있으나, 없으면 앱에서 계산
    miss_now = out["NOW_ENTRY_%"].isna().all() if "NOW_ENTRY_%" in out.columns else True
    if miss_now:
        out["NOW_ENTRY_%"] = (out["종가"] - out["추천매수가"]) / out["추천매수가"] * 100.0
    if "STOP_BUF_%" not in out.columns or out["STOP_BUF_%"].isna().all():
        out["STOP_BUF_%"] = (out["추천매수가"] - out["손절가"]) / out["추천매수가"] * 100.0
    if "T1_BUF_%" not in out.columns or out["T1_BUF_%"].isna().all():
        out["T1_BUF_%"] = (out["추천매도가1"] - out["추천매수가"]) / out["추천매수가"] * 100.0
    if "MIN_RR" not in out.columns or out["MIN_RR"].isna().all():
        out["MIN_RR"] = out["T1_BUF_%"] / out["STOP_BUF_%"]

    # EV_SCORE (0~100): EBS/RR/여유/근접/유동성 가중
    EBS_norm  = (out["EBS"] / 7.0).clip(0, 1)
    RR_norm   = (out["MIN_RR"] / 2.0).clip(0, 1)         # RR=2.0에서 만점
    STOP_norm = (out["STOP_BUF_%"] / 4.0).clip(0, 1)     # 4%에서 만점
    T1_norm   = (out["T1_BUF_%"]   / 8.0).clip(0, 1)     # 8%에서 만점
    PROX_norm = (1 - (out["NOW_ENTRY_%"].abs() / 3.0)).clip(0, 1)  # ±3% 이내가 만점
    LIQ_norm  = np.tanh((out["거래대금(억원)"].fillna(0)) / 500.0)  # 500억 넘어가면 점차 포화

    ev = (
        0.25 * EBS_norm +
        0.25 * RR_norm  +
        0.15 * STOP_norm +
        0.15 * T1_norm  +
        0.10 * PROX_norm +
        0.10 * LIQ_norm
    ) * 100.0
    out["EV_SCORE"] = np.round(ev, 1)
    return out

# -------- 데이터 로드 --------
try:
    df_raw = load_csv_url(RAW_URL); log_src(df_raw, "remote", RAW_URL)
except Exception:
    if os.path.exists(LOCAL_RAW):
        df_raw = load_csv_path(LOCAL_RAW); log_src(df_raw, "local", LOCAL_RAW)
    else:
        st.error("❌ CSV가 없습니다. Actions에서 collector가 data/recommend_latest.csv를 올렸는지 확인하세요.")
        st.stop()

df = apply_names(df_raw)
df = compute_evs(df)

# 숫자형 엄격 캐스팅 (Data Editor 타입 오류 방지)
int_cols  = ["EBS","NOW_TICKS"]
for c in int_cols:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")

float_cols = ["종가","추천매수가","손절가","추천매도가1","추천매도가2","거래대금(억원)","시가총액(억원)",
              "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%",
              "NOW_ENTRY_%","STOP_BUF_%","T1_BUF_%","MIN_RR","EV_SCORE"]
for c in float_cols:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

text_cols = ["시장","종목명","종목코드","근거","통과"]
for c in text_cols:
    if c in df.columns: df[c] = df[c].astype("string")

# ---------------- UI ----------------
st.toggle("보기 모드", key="view_mode_toggle", value=True, help="Top Picks / 전체 보기 전환")
mode = "Top Picks" if st.session_state["view_mode_toggle"] else "전체 보기"

with st.expander("🔍 보기/필터", expanded=True):
    c1,c2,c3 = st.columns([1,1,2])
    with c1:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 0, step=50)
    with c2:
        sort_key = st.selectbox("정렬", ["EV_SCORE▼","EBS▼","거래대금▼","종가▲","종가▼"], index=0)
    with c3:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)
    q = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

# Top Picks 조건
with st.expander("🛠 Top Picks 조건", expanded=(mode=="Top Picks")):
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        rr_min = st.slider("최소 RR(목표1/손절)", 1.00, 3.00, 1.20, step=0.05)
    with c2:
        stop_buf = st.slider("손절여유 ≥ (%)", 0.00, 5.00, 1.00, step=0.10)
    with c3:
        t1_buf = st.slider("목표1여유 ≥ (%)", 0.00, 10.00, 3.00, step=0.25)
    with c4:
        ers_min = st.slider("ERS ≥", 0.00, 3.00, 0.80, step=0.05,
                            help="ERS = (EBS/7) * MIN_RR 의 간단 지표(앱 내 계산)")
    with c5:
        prox = st.slider("Now 근접 밴드(±%)", 0.00, 3.00, 1.00, step=0.10)

view = df.copy()
view = view[(view["거래대금(억원)"] >= float(min_turn)) | view["거래대금(억원)"].isna()]

if q:
    qq = q.strip().lower()
    view = view[
        view["종목명"].fillna("").str.lower().str.contains(qq) |
        view["종목코드"].fillna("").str.contains(qq)
    ]

# ERS 계산(간단형)
view["ERS"] = (pd.to_numeric(view["EBS"], errors="coerce")/7.0) * pd.to_numeric(view["MIN_RR"], errors="coerce")

# 모드별 필터
if mode == "Top Picks":
    view = view[
        (view["MIN_RR"] >= rr_min) &
        (view["STOP_BUF_%"] >= stop_buf) &
        (view["T1_BUF_%"] >= t1_buf) &
        (view["ERS"] >= ers_min) &
        (view["NOW_ENTRY_%"].abs() <= prox)
    ]

# 정렬
def sorter(dfv, key):
    try:
        if key=="EV_SCORE▼": return dfv.sort_values(["EV_SCORE","거래대금(억원)"], ascending=[False,False])
        if key=="EBS▼":      return dfv.sort_values(["EBS","거래대금(억원)"], ascending=[False,False])
        if key=="거래대금▼":  return dfv.sort_values("거래대금(억원)", ascending=False)
        if key=="종가▲":      return dfv.sort_values("종가", ascending=True, na_position="last")
        if key=="종가▼":      return dfv.sort_values("종가", ascending=False, na_position="last")
    except Exception:
        pass
    return dfv

view = sorter(view, sort_key)

# 표시 컬럼
cols = [
    "통과","시장","종목명","종목코드",
    "EV_SCORE","EBS","ERS",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "NOW_ENTRY_%","NOW_TICKS","MIN_RR","STOP_BUF_%","T1_BUF_%",
    "거래대금(억원)","시가총액(억원)",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","근거"
]
for c in cols:
    if c not in view.columns: view[c] = np.nan

st.write(f"📋 총 {len(df):,}개 / 표시 {min(len(view), int(topn)):,}개")

view_fmt = view[cols].head(int(topn)).copy()

# 타입 재확인 (편집기 타입오류 방지)
for c in ["EBS","NOW_TICKS"]:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce").round(0).astype("Int64")
num_cols = [c for c in view_fmt.columns if c not in ["통과","시장","종목명","종목코드","근거"]]
for c in num_cols:
    view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce")

st.data_editor(
    view_fmt,
    width="stretch",
    height=640,
    hide_index=True,
    disabled=True,
    num_rows="fixed",
    column_config={
        # 텍스트
        "통과":     st.column_config.TextColumn(" "),
        "시장":     st.column_config.TextColumn("시장"),
        "종목명":   st.column_config.TextColumn("종목명"),
        "종목코드": st.column_config.TextColumn("종목코드"),
        "근거":     st.column_config.TextColumn("근거"),
        # 점수/평가
        "EV_SCORE": st.column_config.NumberColumn("EV_SCORE", format="%.1f"),
        "EBS":      st.column_config.NumberColumn("EBS",      format="%d"),
        "ERS":      st.column_config.NumberColumn("ERS",      format="%.2f"),
        # 가격/틱/퍼센트
        "종가":        st.column_config.NumberColumn("종가",        format="%,d"),
        "추천매수가":  st.column_config.NumberColumn("추천매수가",  format="%,d"),
        "손절가":      st.column_config.NumberColumn("손절가",      format="%,d"),
        "추천매도가1": st.column_config.NumberColumn("추천매도가1", format="%,d"),
        "추천매도가2": st.column_config.NumberColumn("추천매도가2", format="%,d"),
        "NOW_TICKS":   st.column_config.NumberColumn("Now-Entry(틱)", format="%d"),
        "NOW_ENTRY_%": st.column_config.NumberColumn("Now-Entry(%)",  format="%.2f"),
        "MIN_RR":      st.column_config.NumberColumn("RR(목표1/손절)", format="%.2f"),
        "STOP_BUF_%":  st.column_config.NumberColumn("손절여유(%)",   format="%.2f"),
        "T1_BUF_%":    st.column_config.NumberColumn("목표1여유(%)",  format="%.2f"),
        # 유동성/지표
        "거래대금(억원)": st.column_config.NumberColumn("거래대금(억원)", format="%,.0f"),
        "시가총액(억원)": st.column_config.NumberColumn("시가총액(억원)", format="%,.0f"),
        "RSI14":        st.column_config.NumberColumn("RSI14",        format="%.1f"),
        "乖離%":         st.column_config.NumberColumn("乖離%",         format="%.2f"),
        "MACD_hist":    st.column_config.NumberColumn("MACD_hist",    format="%.4f"),
        "MACD_slope":   st.column_config.NumberColumn("MACD_slope",   format="%.5f"),
        "Vol_Z":        st.column_config.NumberColumn("Vol_Z",        format="%.2f"),
        "ret_5d_%":     st.column_config.NumberColumn("ret_5d_%",     format="%.2f"),
        "ret_10d_%":    st.column_config.NumberColumn("ret_10d_%",    format="%.2f"),
    },
)

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view_fmt.to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_entry_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ 점수/지표 설명", expanded=False):
    st.markdown("""
- **EBS(0~7)**: RSI 45~65 / MACD상승 / MA20 근처 / 거래량증가 / 상승구조 / MACD>sig / 5d<10% 1점씩
- **EV_SCORE(0~100)**: EBS·RR·손절여유·목표1여유·근접성·유동성 종합 점수
- **RR(목표1/손절)**: (목표1여유%) / (손절여유%)
- **Now-Entry(%)**: (종가−추천매수)/추천매수×100, **Now-Entry(틱)**: 틱(10원) 기준 차이
- 컷(권장): 거래대금 ≥ 50억, 시총 ≥ 1,000억
""")
