# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.4.1 (Auto Update + EV Score + Top Picks)
- CSV: data/recommend_latest.csv (remote 우선)
- 이름맵: data/krx_codes.csv (remote/FDR/pykrx 폴백)
- Now-Entry(%)가 0%로 뭉개지지 않도록: float 계산 → 표시는 포맷만 반올림
- EV_SCORE/Top Picks 내장 (필터: 최소 RR, 손절여유, 목표1여유, ERS, Now 근접 밴드)
"""

import os, io, math, requests, numpy as np, pandas as pd, streamlit as st
from datetime import datetime

# optional deps
try:
    from pykrx import stock
    PYKRX_OK = True
except Exception:
    PYKRX_OK = False

try:
    import FinanceDataReader as fdr
    FDR_OK = True
except Exception:
    FDR_OK = False

st.set_page_config(page_title="LDY Pro Trader v3.4.1 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.4 (Auto Update)")
st.caption("매일 장마감 후 자동 업데이트되는 스윙 추천 종목 리스트 | EV스코어·TopPick 내장")

RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"
PASS_SCORE = 4

# ---------------- IO ----------------
@st.cache_data(ttl=300)
def load_csv_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=300)
def load_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def log_src(df: pd.DataFrame, src: str, url_or_path: str):
    st.info(f"상태 ✅ 데이터 로드: {src}\n\n{url_or_path}")
    st.success(f"📅 표시시각: {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M')} · 행수: {len(df):,}")

# --------------- utils --------------
def z6(x) -> str:
    s = str(x)
    return s.zfill(6) if s.isdigit() else s

def ensure_turnover(df: pd.DataFrame) -> pd.DataFrame:
    if "거래대금(억원)" not in df.columns:
        base = None
        if "거래대금(원)" in df.columns:
            base = pd.to_numeric(df["거래대금(원)"], errors="coerce")
        elif all(x in df.columns for x in ["거래량","종가"]):
            base = pd.to_numeric(df["거래량"], errors="coerce") * pd.to_numeric(df["종가"], errors="coerce")
        if base is not None:
            df["거래대금(억원)"] = (base/1e8).round(2)
    return df

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cmap = {
        "Date":"날짜","date":"날짜",
        "Code":"종목코드","티커":"종목코드","ticker":"종목코드",
        "Name":"종목명","name":"종목명",
        "Open":"시가","High":"고가","Low":"저가","Close":"종가","Volume":"거래량",
        "거래대금":"거래대금(원)","시가총액":"시가총액(원)"
    }
    for k,v in cmap.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})

    if "날짜" in df.columns:
        with pd.option_context('future.no_silent_downcasting', True):
            try: df["날짜"] = pd.to_datetime(df["날짜"])
            except: pass
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.replace(".0","", regex=False).map(z6)
    else:
        df["종목코드"] = None
    if "시장" not in df.columns:
        df["시장"] = "ALL"
    if "종목명" not in df.columns:
        df["종목명"] = None

    for c in ["시가","고가","저가","종가","거래량","거래대금(원)","시가총액(원)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = ensure_turnover(df)
    return df

# -------- name map (robust) --------
@st.cache_data(ttl=6*60*60)
def load_name_map() -> pd.DataFrame | None:
    # 1) repo의 data/krx_codes.csv 우선
    try:
        m = load_csv_url(CODES_URL)
        if {"종목코드","종목명"}.issubset(m.columns):
            m["종목코드"] = m["종목코드"].astype(str).map(z6)
            return m[["종목코드","종목명"]].drop_duplicates("종목코드")
    except Exception:
        pass
    if os.path.exists(LOCAL_MAP):
        try:
            m = load_csv_path(LOCAL_MAP)
            if {"종목코드","종목명"}.issubset(m.columns):
                m["종목코드"] = m["종목코드"].astype(str).map(z6)
                return m[["종목코드","종목명"]].drop_duplicates("종목코드")
        except Exception:
            pass

    # 2) FDR 폴백
    if FDR_OK:
        try:
            lst = fdr.StockListing("KRX")
            m = lst.rename(columns={"Code":"종목코드","Name":"종목명"})[["종목코드","종목명"]]
            m["종목코드"] = m["종목코드"].astype(str).map(z6)
            return m.drop_duplicates("종목코드")
        except Exception:
            pass

    # 3) pykrx 개별 조회
    if PYKRX_OK:
        today = datetime.now().strftime("%Y%m%d")
        rows = []
        try:
            for mk in ["KOSPI","KOSDAQ","KONEX"]:
                try:
                    lst = stock.get_market_ticker_list(today, market=mk)
                except Exception:
                    lst = []
                for t in lst:
                    try:
                        nm = stock.get_market_ticker_name(t)
                    except Exception:
                        nm = None
                    rows.append({"종목코드": str(t).zfill(6), "종목명": nm})
            m = pd.DataFrame(rows).dropna().drop_duplicates("종목코드")
            return m if len(m) else None
        except Exception:
            return None
    return None

def apply_names(df: pd.DataFrame) -> pd.DataFrame:
    mp = load_name_map()
    if mp is not None:
        df["종목코드"] = df["종목코드"].astype(str).map(z6)
        if "종목명" not in df.columns: df["종목명"] = None
        df = df.merge(mp, on="종목코드", how="left", suffixes=("","_map"))
        df["종목명"] = df["종목명"].fillna(df["종목명_map"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_map")], errors="ignore")
    df["종목명"] = df["종목명"].fillna("(이름없음)")
    return df

# -------- 데이터 로드 --------
try:
    df_raw = load_csv_url(RAW_URL); log_src(df_raw, "remote", RAW_URL)
except Exception:
    if os.path.exists(LOCAL_RAW):
        df_raw = load_csv_path(LOCAL_RAW); log_src(df_raw, "local", LOCAL_RAW)
    else:
        st.error("❌ CSV가 없습니다. Actions에서 collector가 data/recommend_latest.csv를 올렸는지 확인하세요.")
        st.stop()

df = normalize_cols(df_raw.copy())
df = apply_names(df)

# 숫자 캐스팅
num_cols = [
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist",
    "MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- 파생지표(진입성) ----------
# 근접도/여유/리스크-리워드
df["NOW_ENTRY_%"] = 100.0 * ((df["종가"] - df["추천매수가"]) / df["추천매수가"])
df["NOW_ENTRY_ABS_%"] = df["NOW_ENTRY_%"].abs()
df["STOP_GAP_%"] = 100.0 * ((df["추천매수가"] - df["손절가"]) / df["추천매수가"])
df["T1_GAP_%"]   = 100.0 * ((df["추천매도가1"] - df["추천매수가"]) / df["추천매수가"])
df["RR_T1"] = np.where(df["STOP_GAP_%"] > 0, df["T1_GAP_%"] / df["STOP_GAP_%"], np.nan)

# ERS(간이 EV): p_hit 추정 * T1_GAP - (1-p)*STOP_GAP
def p_hit_est(row):
    ebs = row.get("EBS", np.nan)
    macds = row.get("MACD_slope", np.nan)
    volz = row.get("Vol_Z", np.nan)
    p = 0.40
    if pd.notna(ebs):  p += 0.06 * min(max(ebs, 0), 7)
    if pd.notna(macds) and macds > 0: p += 0.02
    if pd.notna(volz) and volz > 1.2: p += 0.01
    return float(np.clip(p, 0.25, 0.85))

p = df.apply(p_hit_est, axis=1)
df["ERS_%"] = p * df["T1_GAP_%"] - (1 - p) * df["STOP_GAP_%"]

# EV_SCORE (표준화 가중 합)
def zsafe(s):
    s = pd.to_numeric(s, errors="coerce")
    m, v = np.nanmean(s), np.nanstd(s)
    if not np.isfinite(v) or v == 0: return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / v

df["EV_SCORE"] = (
    0.35 * zsafe(df["EBS"]) +
    0.25 * zsafe(df["RR_T1"]) +
    0.20 * zsafe(df["T1_GAP_%"]) +
   -0.10 * zsafe(df["NOW_ENTRY_ABS_%"]) +
    0.10 * zsafe(df["Vol_Z"])
)

# ---------- 보기 모드 ----------
mode = st.radio("보기 모드", ["Top Picks", "전체 보기"], horizontal=True)

with st.expander("🔍 보기/필터", expanded=True):
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,2])
    with c1:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c2:
        sort_key = st.selectbox("정렬", ["EV_SCORE▼","EBS▼","거래대금▼","RSI▲","RSI▼","종가▲","종가▼"], index=0)
    with c3:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)
    with c4:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")
    with c5:
        pass

view = df.copy()
view = view[view["거래대금(억원)"] >= float(min_turn)]
if q_text:
    q = q_text.strip().lower()
    view = view[
        view["종목명"].fillna("").astype(str).str.lower().str.contains(q) |
        view["종목코드"].fillna("").astype(str).str.contains(q)
    ]

if mode == "Top Picks":
    with st.expander("🛠 Top Picks 조건", expanded=True):
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1:
            min_rr = st.slider("최소 RR(목표1/손절)", 1.00, 3.00, 1.20, step=0.05)
        with c2:
            min_stop = st.slider("손절여유 ≥ (%)", 0.00, 5.00, 1.00, step=0.25)
        with c3:
            min_t1 = st.slider("목표1여유 ≥ (%)", 0.00, 10.00, 3.00, step=0.5)
        with c4:
            min_ers = st.slider("ERS ≥", 0.00, 3.00, 0.20, step=0.05)
        with c5:
            band = st.slider("Now 근접 밴드(±%)", 0.00, 3.00, 1.50, step=0.25)

    # 필터
    view = view[
        (view["RR_T1"] >= min_rr) &
        (view["STOP_GAP_%"] >= min_stop) &
        (view["T1_GAP_%"] >= min_t1) &
        (view["ERS_%"] >= min_ers) &
        (view["NOW_ENTRY_ABS_%"] <= band)
    ]

# 정렬
def safe_sort(dfv, key):
    try:
        if key=="EV_SCORE▼": return dfv.sort_values("EV_SCORE", ascending=False, na_position="last")
        if key=="EBS▼":      return dfv.sort_values(["EBS","거래대금(억원)"], ascending=[False,False])
        if key=="거래대금▼":  return dfv.sort_values("거래대금(억원)", ascending=False)
        if key=="RSI▲":      return dfv.sort_values("RSI14", ascending=True, na_position="last")
        if key=="RSI▼":      return dfv.sort_values("RSI14", ascending=False, na_position="last")
        if key=="종가▲":      return dfv.sort_values("종가", ascending=True, na_position="last")
        if key=="종가▼":      return dfv.sort_values("종가", ascending=False, na_position="last")
    except Exception:
        pass
    for alt in ["EV_SCORE","EBS","거래대금(억원)","종가"]:
        if alt in dfv.columns:
            return dfv.sort_values(alt, ascending=False, na_position="last")
    return dfv

view = safe_sort(view, sort_key)
view = view.head(int(topn))

st.write(f"📋 총 {len(df):,}개 / 표시 {len(view):,}개")

# ---------- 표 렌더링: 타입 안전 + 포맷 ----------
cols = [
    "시장","종목명","종목코드",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "NOW_ENTRY_%","NOW_ENTRY_ABS_%","STOP_GAP_%","T1_GAP_%","RR_T1","ERS_%","EV_SCORE",
    "거래대금(억원)","시가총액(억원)","EBS","근거",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
for c in cols:
    if c not in view.columns: view[c] = np.nan

# 표시는 별도 사본
vf = view[cols].copy()

# 가격류 → Int64 (NaN 허용 정수)
for c in ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]:
    if c in vf.columns:
        vf[c] = pd.to_numeric(vf[c], errors="coerce").round(0).astype("Int64")

# 나머지 수치 → float 유지(퍼센트/지표 포맷은 표시에서 처리)
for c in ["NOW_ENTRY_%","NOW_ENTRY_ABS_%","STOP_GAP_%","T1_GAP_%","RR_T1","ERS_%","EV_SCORE",
          "거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"]:
    if c in vf.columns:
        vf[c] = pd.to_numeric(vf[c], errors="coerce")

st.data_editor(
    vf,
    width="stretch",
    height=680,
    hide_index=True,
    disabled=True,
    num_rows="fixed",
    column_config={
        # 텍스트
        "시장":         st.column_config.TextColumn("시장"),
        "종목명":       st.column_config.TextColumn("종목명"),
        "종목코드":     st.column_config.TextColumn("종목코드"),
        "근거":         st.column_config.TextColumn("근거"),
        # 가격/정수
        "종가":          st.column_config.NumberColumn("종가",           format="%,d"),
        "추천매수가":    st.column_config.NumberColumn("추천매수가",     format="%,d"),
        "손절가":        st.column_config.NumberColumn("손절가",         format="%,d"),
        "추천매도가1":   st.column_config.NumberColumn("추천매도가1",    format="%,d"),
        "추천매도가2":   st.column_config.NumberColumn("추천매도가2",    format="%,d"),
        "EBS":          st.column_config.NumberColumn("EBS",            format="%d"),
        # 퍼센트/스코어
        "NOW_ENTRY_%":      st.column_config.NumberColumn("Now-Entry(%)",     format="%.2f%%"),
        "NOW_ENTRY_ABS_%":  st.column_config.NumberColumn("|Now-Entry|(%)",   format="%.2f%%"),
        "STOP_GAP_%":       st.column_config.NumberColumn("손절여유(%)",      format="%.2f%%"),
        "T1_GAP_%":         st.column_config.NumberColumn("목표1여유(%)",     format="%.2f%%"),
        "RR_T1":            st.column_config.NumberColumn("RR(목표1/손절)",   format="%.2f"),
        "ERS_%":            st.column_config.NumberColumn("ERS",              format="%.2f"),
        "EV_SCORE":         st.column_config.NumberColumn("EV_SCORE",         format="%.2f"),
        # 억원/지표
        "거래대금(억원)": st.column_config.NumberColumn("거래대금(억원)",  format="%,.0f"),
        "시가총액(억원)": st.column_config.NumberColumn("시가총액(억원)",  format="%,.0f"),
        "RSI14":        st.column_config.NumberColumn("RSI14",          format="%.1f"),
        "乖離%":         st.column_config.NumberColumn("乖離%",           format="%.2f"),
        "MACD_hist":    st.column_config.NumberColumn("MACD_hist",      format="%.4f"),
        "MACD_slope":   st.column_config.NumberColumn("MACD_slope",     format="%.5f"),
        "Vol_Z":        st.column_config.NumberColumn("Vol_Z",          format="%.2f"),
        "ret_5d_%":     st.column_config.NumberColumn("ret_5d_%",       format="%.2f"),
        "ret_10d_%":    st.column_config.NumberColumn("ret_10d_%",      format="%.2f"),
    },
)

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=vf.to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_entry_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ 점수/지표 설명"):
    st.markdown("""
**EBS(0~7)**: RSI 45~65 / MACD상승 / MA20 근처 / 거래량증가 / 상승구조(MA20>MA60) / MACD>sig / 5일수익<10%  
**Now-Entry(%)**: (종가−추천매수)/추천매수×100 → 0에 가까울수록 '지금 가격이 엔트리와 근접'  
**손절여유(%)**: (추천매수−손절)/추천매수×100  
**목표1여유(%)**: (목표1−추천매수)/추천매수×100  
**RR(목표1/손절)**: 목표1여유 ÷ 손절여유 (≥1.2 권장)  
**ERS**: p_hit 추정 기반 간이 기대값(높을수록 유리)  
**EV_SCORE**: EBS, RR, 목표여유, Now근접도, Vol_Z 표준화 가중합 (높을수록 우선순위↑)
""")
