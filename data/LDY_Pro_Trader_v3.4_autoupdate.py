# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.4 (Auto Update)
- 추천 CSV: data/recommend_latest.csv (remote 우선)
- Robust name map(FDR/pykrx 폴백) + 숫자 포맷 + EV 점수 + Top Picks
- Streamlit Data Editor 타입 호환성 보강
"""

import os, io, math, requests, numpy as np, pandas as pd, streamlit as st
from datetime import datetime

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

st.set_page_config(page_title="LDY Pro Trader v3.4 (Auto Update)", layout="wide")
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

# -------- name map (robust) --------
@st.cache_data(ttl=6*60*60)
def load_name_map() -> pd.DataFrame | None:
    # 1) repo codes
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
    # 2) FDR
    if FDR_OK:
        try:
            lst = fdr.StockListing("KRX")
            m = lst.rename(columns={"Code":"종목코드","Name":"종목명"})[["종목코드","종목명"]]
            m["종목코드"] = m["종목코드"].astype(str).map(z6)
            return m.drop_duplicates("종목코드")
        except Exception:
            pass
    # 3) pykrx
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

# -------------- EV/TopPick --------------
def compute_ev_fields(df: pd.DataFrame, now_band_pct: float = 3.0) -> pd.DataFrame:
    """RR/손절여유/목표여유/ERS/EV_SCORE/Now근접 계산"""
    out = df.copy()
    to_num = ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]
    for c in to_num:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # RR (목표1/손절)
    rr = (out["추천매도가1"] - out["추천매수가"]) / (out["추천매수가"] - out["손절가"])
    rr = rr.replace([np.inf, -np.inf], np.nan)
    out["RR"] = rr

    # 손절여유(%): (엔트리-손절)/엔트리 *100
    out["손절여유_%"] = (out["추천매수가"] - out["손절가"]) / out["추천매수가"] * 100

    # 목표1여유(%): (목표1-현재)/현재 *100
    out["목표1여유_%"] = (out["추천매도가1"] - out["종가"]) / out["종가"] * 100

    # ERS: EBS 정규화(3~7 → 0~1)
    ebs = pd.to_numeric(out.get("EBS", np.nan), errors="coerce")
    out["ERS"] = np.clip((ebs - 3) / 4, 0, 1)

    # Now 근접: 엔트리 대비 현재가 이격
    out["NOW_FROM_ENTRY_%"] = (out["종가"] - out["추천매수가"]) / out["추천매수가"] * 100
    if now_band_pct and now_band_pct > 0:
        out["NOW_IN_BAND"] = (out["NOW_FROM_ENTRY_%"].abs() <= now_band_pct).fillna(False)
        now_prox_norm = 1 - np.minimum(out["NOW_FROM_ENTRY_%"].abs() / now_band_pct, 1.0)
    else:
        out["NOW_IN_BAND"] = False
        now_prox_norm = 0.0

    # EV_SCORE (0~100)
    rr_clip   = np.clip(out["RR"] / 3.0, 0, 1)                     # RR 3배에서 상한
    sl_clip   = np.clip(out["손절여유_%"] / 5.0, 0, 1)             # 5%에서 상한
    t1_clip   = np.clip(out["목표1여유_%"] / 10.0, 0, 1)           # 10%에서 상한
    ers_clip  = np.clip(out["ERS"], 0, 1)
    now_clip  = np.clip(now_prox_norm, 0, 1)

    ev = 100*(0.35*rr_clip + 0.25*ers_clip + 0.20*t1_clip + 0.20*now_clip)
    out["EV_SCORE"] = ev.round(1)

    return out

def enforce_types_for_editor(df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit data_editor 타입 호환을 위해 Number/Int 컬럼 강제 캐스팅"""
    out = df.copy()
    # 정수형으로 보일 컬럼
    int_cols = ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]
    for c in int_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    # 실수형
    float_cols = ["거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z",
                  "ret_5d_%","ret_10d_%","RR","손절여유_%","목표1여유_%","ERS","NOW_FROM_ENTRY_%","EV_SCORE"]
    for c in float_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out

# -------- load raw --------
try:
    df_raw = load_csv_url(RAW_URL); log_src(df_raw, "remote", RAW_URL)
except Exception:
    if os.path.exists(LOCAL_RAW):
        df_raw = load_csv_path(LOCAL_RAW); log_src(df_raw, "local", LOCAL_RAW)
    else:
        st.error("❌ CSV가 없습니다. Actions에서 collector가 data/recommend_latest.csv를 올렸는지 확인하세요.")
        st.stop()

# 기본 정리
df = df_raw.copy()
df = ensure_turnover(df)
df = apply_names(df)

for c in ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z",
          "ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 보기 모드
tab1, tab2 = st.tabs(["🟢 Top Picks", "📋 전체 보기"])

with st.sidebar:
    st.subheader("🔍 보기/필터")
    min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 0, step=10)
    sort_key = st.selectbox("정렬", ["EV_SCORE▼","EBS▼","거래대금▼","시가총액▼","RSI▲","RSI▼","종가▲","종가▼"], index=0)
    topn = st.slider("표시 수(Top N)", 10, 500, 10, step=10)
    q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

with tab1:
    st.subheader("🛠 Top Picks 조건")
    c1,c2,c3 = st.columns(3)
    with c1:
        rr_min = st.slider("최소 RR(목표1/손절)", 1.00, 3.00, 1.00, step=0.25)
        ers_min = st.slider("ERS ≥", 0.00, 1.00, 0.00, step=0.05)
    with c2:
        sl_min = st.slider("손절여유 ≥ (%)", 0.00, 5.00, 0.00, step=0.25)
        t1_min = st.slider("목표1여유 ≥ (%)", 0.00, 10.00, 0.00, step=0.5)
    with c3:
        now_band = st.slider("Now 근접 밴드(±%)", 0.00, 3.00, 3.00, step=0.25)

# EV 필드 계산
df_ev = compute_ev_fields(df, now_band_pct=3.0)

def apply_common_filters(dfin: pd.DataFrame) -> pd.DataFrame:
    v = dfin.copy()
    if "거래대금(억원)" in v.columns:
        v = v[v["거래대금(억원)"] >= float(min_turn)]
    if q_text:
        q = q_text.strip().lower()
        v = v[
            v["종목명"].fillna("").astype(str).str.lower().str.contains(q) |
            v["종목코드"].fillna("").astype(str).str.contains(q)
        ]
    return v

def safe_sort(dfv: pd.DataFrame, key: str) -> pd.DataFrame:
    try:
        if key=="EV_SCORE▼" and "EV_SCORE" in dfv.columns:
            return dfv.sort_values(["EV_SCORE","EBS","거래대금(억원)"], ascending=[False, False, False])
        if key=="EBS▼" and "EBS" in dfv.columns:
            by = ["EBS"] + (["거래대금(억원)"] if "거래대금(억원)" in dfv.columns else [])
            return dfv.sort_values(by=by, ascending=[False]+[False]*(len(by)-1))
        if key=="거래대금▼" and "거래대금(억원)" in dfv.columns:
            return dfv.sort_values("거래대금(억원)", ascending=False)
        if key=="시가총액▼" and "시가총액(억원)" in dfv.columns:
            return dfv.sort_values("시가총액(억원)", ascending=False, na_position="last")
        if key=="RSI▲" and "RSI14" in dfv.columns:
            return dfv.sort_values("RSI14", ascending=True, na_position="last")
        if key=="RSI▼" and "RSI14" in dfv.columns:
            return dfv.sort_values("RSI14", ascending=False, na_position="last")
        if key=="종가▲" and "종가" in dfv.columns:
            return dfv.sort_values("종가", ascending=True, na_position="last")
        if key=="종가▼" and "종가" in dfv.columns:
            return dfv.sort_values("종가", ascending=False, na_position="last")
    except Exception:
        pass
    for alt in ["EV_SCORE","EBS","거래대금(억원)","시가총액(억원)","종가"]:
        if alt in dfv.columns:
            return dfv.sort_values(alt, ascending=False, na_position="last")
    return dfv

# -------- Top Picks --------
with tab1:
    v = df_ev.copy()
    v = apply_common_filters(v)

    # Top Picks 추가 필터
    v = v[
        (v["RR"] >= rr_min) &
        (v["손절여유_%"] >= sl_min) &
        (v["목표1여유_%"] >= t1_min) &
        (v["ERS"] >= ers_min)
    ]
    # Now 근접 밴드: 필수는 아니지만 기본 on 느낌
    v = v[(v["NOW_FROM_ENTRY_%"].abs() <= now_band) | v["NOW_FROM_ENTRY_%"].isna()]

    v = safe_sort(v, sort_key)
    v = v.head(int(topn))

    # 표시 컬럼
    cols = [
        "시장","종목명","종목코드","EV_SCORE",
        "EBS","RR","손절여유_%","목표1여유_%","ERS","NOW_FROM_ENTRY_%",
        "종가","추천매수가","손절가","추천매도가1","추천매도가2",
        "거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","근거"
    ]
    for c in cols:
        if c not in v.columns:
            v[c] = np.nan

    st.write(f"📋 총 {len(df_ev):,}개 / 표시 {min(len(v), int(topn)):,}개")
    v_fmt = enforce_types_for_editor(v[cols])

    st.data_editor(
        v_fmt,
        width="stretch", height=520, hide_index=True, disabled=True, num_rows="fixed",
        column_config={
            "시장": st.column_config.TextColumn("시장"),
            "종목명": st.column_config.TextColumn("종목명"),
            "종목코드": st.column_config.TextColumn("종목코드"),
            "근거": st.column_config.TextColumn("근거"),
            "EV_SCORE": st.column_config.NumberColumn("EV_SCORE", format="%.1f"),
            "EBS": st.column_config.NumberColumn("EBS", format="%d"),
            "RR": st.column_config.NumberColumn("RR(목표1/손절)", format="%.2f"),
            "손절여유_%": st.column_config.NumberColumn("손절여유(%)", format="%.2f"),
            "목표1여유_%": st.column_config.NumberColumn("목표1여유(%)", format="%.2f"),
            "ERS": st.column_config.NumberColumn("ERS", format="%.2f"),
            "NOW_FROM_ENTRY_%": st.column_config.NumberColumn("Now-Entry(%)", format="%.2f"),
            "종가": st.column_config.NumberColumn("종가", format="%,d"),
            "추천매수가": st.column_config.NumberColumn("추천매수가", format="%,d"),
            "손절가": st.column_config.NumberColumn("손절가", format="%,d"),
            "추천매도가1": st.column_config.NumberColumn("추천매도가1", format="%,d"),
            "추천매도가2": st.column_config.NumberColumn("추천매도가2", format="%,d"),
            "거래대금(억원)": st.column_config.NumberColumn("거래대금(억원)", format="%,.0f"),
            "시가총액(억원)": st.column_config.NumberColumn("시가총액(억원)", format="%,.0f"),
            "RSI14": st.column_config.NumberColumn("RSI14", format="%.1f"),
            "乖離%": st.column_config.NumberColumn("乖離%", format="%.2f"),
            "MACD_hist": st.column_config.NumberColumn("MACD_hist", format="%.4f"),
            "MACD_slope": st.column_config.NumberColumn("MACD_slope", format="%.5f"),
            "Vol_Z": st.column_config.NumberColumn("Vol_Z", format="%.2f"),
            "ret_5d_%": st.column_config.NumberColumn("ret_5d_%", format="%.2f"),
            "ret_10d_%": st.column_config.NumberColumn("ret_10d_%", format="%.2f"),
        },
    )

# -------- 전체 보기 --------
with tab2:
    v = apply_common_filters(df_ev)
    v = safe_sort(v, sort_key)
    v = v.head(int(topn))
    cols_all = [
        "시장","종목명","종목코드","EV_SCORE","EBS","RR","손절여유_%","목표1여유_%","ERS","NOW_FROM_ENTRY_%",
        "종가","추천매수가","손절가","추천매도가1","추천매도가2",
        "거래대금(억원)","시가총액(억원)",
        "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","근거"
    ]
    for c in cols_all:
        if c not in v.columns:
            v[c] = np.nan
    v_fmt = enforce_types_for_editor(v[cols_all])

    st.data_editor(
        v_fmt,
        width="stretch", height=520, hide_index=True, disabled=True, num_rows="fixed",
        column_config={
            "시장": st.column_config.TextColumn("시장"),
            "종목명": st.column_config.TextColumn("종목명"),
            "종목코드": st.column_config.TextColumn("종목코드"),
            "근거": st.column_config.TextColumn("근거"),
            "EV_SCORE": st.column_config.NumberColumn("EV_SCORE", format="%.1f"),
            "EBS": st.column_config.NumberColumn("EBS", format="%d"),
            "RR": st.column_config.NumberColumn("RR(목표1/손절)", format="%.2f"),
            "손절여유_%": st.column_config.NumberColumn("손절여유(%)", format="%.2f"),
            "목표1여유_%": st.column_config.NumberColumn("목표1여유(%)", format="%.2f"),
            "ERS": st.column_config.NumberColumn("ERS", format="%.2f"),
            "NOW_FROM_ENTRY_%": st.column_config.NumberColumn("Now-Entry(%)", format="%.2f"),
            "종가": st.column_config.NumberColumn("종가", format="%,d"),
            "추천매수가": st.column_config.NumberColumn("추천매수가", format="%,d"),
            "손절가": st.column_config.NumberColumn("손절가", format="%,d"),
            "추천매도가1": st.column_config.NumberColumn("추천매도가1", format="%,d"),
            "추천매도가2": st.column_config.NumberColumn("추천매도가2", format="%,d"),
            "거래대금(억원)": st.column_config.NumberColumn("거래대금(억원)", format="%,.0f"),
            "시가총액(억원)": st.column_config.NumberColumn("시가총액(억원)", format="%,.0f"),
            "RSI14": st.column_config.NumberColumn("RSI14", format="%.1f"),
            "乖離%": st.column_config.NumberColumn("乖離%", format="%.2f"),
            "MACD_hist": st.column_config.NumberColumn("MACD_hist", format="%.4f"),
            "MACD_slope": st.column_config.NumberColumn("MACD_slope", format="%.5f"),
            "Vol_Z": st.column_config.NumberColumn("Vol_Z", format="%.2f"),
            "ret_5d_%": st.column_config.NumberColumn("ret_5d_%", format="%.2f"),
            "ret_10d_%": st.column_config.NumberColumn("ret_10d_%", format="%.2f"),
        },
    )

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=df_ev.to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_ev_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ 점수/지표 설명"):
    st.markdown("""
- **EBS**(0~7): RSI 45~65 / MACD↑ / MA20 근처 / 거래량↑ / 상승구조(MA20>MA60) / MACD>0 / 5d<10%
- **RR(목표1/손절)**: (목표1−엔트리) / (엔트리−손절). 1.0 이상 권장
- **손절여유(%)**: (엔트리−손절)/엔트리 ×100 (≥ 2~3% 권장)
- **목표1여유(%)**: (목표1−현재)/현재 ×100
- **ERS**: EBS를 0~1로 정규화(3→0, 7→1)
- **Now-Entry(%)**: 현재가가 엔트리에 얼마나 가까운지(±3% 내 근접하면 진입 판단 용이)
- **EV_SCORE**(0~100): 0.35·RR_norm + 0.25·ERS + 0.20·T1_room + 0.20·Now_proximity
""")
