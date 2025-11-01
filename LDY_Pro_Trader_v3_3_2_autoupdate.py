# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.4 (Auto Update + EV/Regime/TopPick)
- 추천 CSV: data/recommend_latest.csv (remote 우선)
- 이름맵:   data/krx_codes.csv (remote 우선) → FDR → pykrx 순 폴백
- EV_SCORE / RR1 / Stop여유 / Target여유 / Now밴드거리 / REGIME_OK / EVENT_RISK / TopPick 표시
- 표 숫자에 천단위 콤마 적용
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

st.set_page_config(page_title="LDY Pro Trader v3.4", layout="wide")
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
    cmap = {"Date":"날짜","date":"날짜","Code":"종목코드","티커":"종목코드","ticker":"종목코드",
            "Name":"종목명","name":"종목명","Open":"시가","High":"고가","Low":"저가","Close":"종가","Volume":"거래량",
            "거래대금":"거래대금(원)","시가총액":"시가총액(원)"}
    for k,v in cmap.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})
    if "날짜" in df.columns:
        try: df["날짜"] = pd.to_datetime(df["날짜"])
        except: pass
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.replace(".0","",regex=False).map(z6)
    else:
        df["종목코드"] = None
    if "시장" not in df.columns: df["시장"]="ALL"
    if "종목명" not in df.columns: df["종목명"]=None
    for c in ["시가","고가","저가","종가","거래량","거래대금(원)","시가총액(원)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = ensure_turnover(df)
    return df

# -------- load raw --------
try:
    df = load_csv_url(RAW_URL); log_src(df, "remote", RAW_URL)
except Exception:
    if os.path.exists(LOCAL_RAW):
        df = load_csv_path(LOCAL_RAW); log_src(df, "local", LOCAL_RAW)
    else:
        st.error("❌ CSV가 없습니다. Actions에서 collector가 data/recommend_latest.csv를 올렸는지 확인하세요.")
        st.stop()

df = normalize_cols(df)

# 숫자 캐스팅
num_cols = ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z",
            "ret_5d_%","ret_10d_%","EBS","추천매수가","손절가","추천매도가1","추천매도가2",
            "RR1","Stop여유_%","Target1여유_%","Now밴드거리_%","EV_R","ERS","EV_SCORE"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ------------- UI -------------
left, right = st.columns([2,1])
with left:
    st.subheader("Top Picks / 전체 보기")
with right:
    st.markdown("")

with st.expander("🔍 보기/필터", expanded=True):
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,1.2])
    with c1:
        only_top = st.checkbox("🌟 Top Picks만", value=False)
        only_entry = st.checkbox("🚀 EBS≥4", value=("EBS" in df.columns))
    with c2:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c3:
        sort_key = st.selectbox("정렬", ["EV_SCORE▼","ERS▼","EBS▼","거래대금▼","RSI▲","RSI▼","종가▲","종가▼"], index=0)
    with c4:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)
    with c5:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")
    c6,c7,c8 = st.columns([1,1,1])
    with c6:
        hide_event = st.checkbox("이벤트 위험 제외", value=True if "EVENT_RISK" in df.columns else False)
    with c7:
        hide_bad_regime = st.checkbox("레짐 불리 제외", value=False)
    with c8:
        now_band_lim = st.slider("Now 근접 밴드(±%)", 0.0, 3.0, 1.0, 0.1)

view = df.copy()
if "TopPick" in view.columns and only_top:
    view = view[view["TopPick"]==True]
if only_entry and "EBS" in view.columns:
    view = view[view["EBS"] >= PASS_SCORE]
if "거래대금(억원)" in view.columns:
    view = view[view["거래대금(억원)"] >= float(min_turn)]
if "EVENT_RISK" in view.columns and hide_event:
    view = view[view["EVENT_RISK"] != True]
if "REGIME_OK" in view.columns and hide_bad_regime:
    view = view[view["REGIME_OK"] == True]
if "Now밴드거리_%" in view.columns and now_band_lim > 0:
    view = view[view["Now밴드거리_%"].abs() <= now_band_lim]
if q_text:
    q = q_text.strip().lower()
    view = view[
        view["종목명"].fillna("").astype(str).str.lower().str.contains(q) |
        view["종목코드"].fillna("").astype(str).str.contains(q)
    ]

def safe_sort(dfv, key):
    try:
        if key=="EV_SCORE▼" and "EV_SCORE" in dfv.columns: return dfv.sort_values("EV_SCORE", ascending=False)
        if key=="ERS▼" and "ERS" in dfv.columns: return dfv.sort_values("ERS", ascending=False)
        if key=="EBS▼" and "EBS" in dfv.columns: return dfv.sort_values(["EBS","거래대금(억원)"], ascending=[False,False])
        if key=="거래대금▼" and "거래대금(억원)" in dfv.columns: return dfv.sort_values("거래대금(억원)", ascending=False)
        if key=="RSI▲" and "RSI14" in dfv.columns: return dfv.sort_values("RSI14", ascending=True, na_position="last")
        if key=="RSI▼" and "RSI14" in dfv.columns: return dfv.sort_values("RSI14", ascending=False, na_position="last")
        if key=="종가▲" and "종가" in dfv.columns: return dfv.sort_values("종가", ascending=True, na_position="last")
        if key=="종가▼" and "종가" in dfv.columns: return dfv.sort_values("종가", ascending=False, na_position="last")
    except Exception:
        pass
    for alt in ["EV_SCORE","ERS","EBS","거래대금(억원)","종가"]:
        if alt in dfv.columns:
            return dfv.sort_values(alt, ascending=False, na_position="last")
    return dfv

view = safe_sort(view, sort_key)

# 배지/표 컬럼
if "EBS" in view.columns:
    view["통과"] = np.where(view["EBS"]>=PASS_SCORE, "🚀", "")
if "TopPick" in view.columns:
    view["Top"] = np.where(view["TopPick"]==True, "🌟", "")

cols = [
    "Top","통과","시장","종목명","종목코드",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "RR1","Stop여유_%","Target1여유_%","Now밴드거리_%",
    "EV_R","ERS","EV_SCORE",
    "거래대금(억원)","시가총액(억원)",
    "EBS","근거",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%",
    "REGIME_OK","EVENT_RISK"
]
for c in cols:
    if c not in view.columns: view[c]=np.nan

st.write(f"📋 총 {len(df):,}개 / 표시 {min(len(view), int(topn)):,}개")

# 숫자 포맷 처리
view_fmt = view[cols].head(int(topn)).copy()
for c in ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce").round(0).astype("Int64")
for c in ["거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%",
          "RR1","Stop여유_%","Target1여유_%","Now밴드거리_%","EV_R","ERS","EV_SCORE"]:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce")

st.data_editor(
    view_fmt,
    width="stretch",
    height=700,
    hide_index=True,
    disabled=True,
    num_rows="fixed",
    column_config={
        "Top":         st.column_config.TextColumn(" "),
        "통과":         st.column_config.TextColumn(" "),
        "시장":         st.column_config.TextColumn("시장"),
        "종목명":       st.column_config.TextColumn("종목명"),
        "종목코드":     st.column_config.TextColumn("코드"),
        "근거":         st.column_config.TextColumn("근거"),
        # 가격/정수
        "종가":          st.column_config.NumberColumn("종가",           format="%,d"),
        "추천매수가":    st.column_config.NumberColumn("추천매수가",     format="%,d"),
        "손절가":        st.column_config.NumberColumn("손절가",         format="%,d"),
        "추천매도가1":   st.column_config.NumberColumn("추천매도가1",    format="%,d"),
        "추천매도가2":   st.column_config.NumberColumn("추천매도가2",    format="%,d"),
        "EBS":          st.column_config.NumberColumn("EBS",            format="%d"),
        # 성능/리스크 지표
        "RR1":           st.column_config.NumberColumn("RR(목표1/손절)",   format="%.2f"),
        "Stop여유_%":     st.column_config.NumberColumn("손절여유(%)",      format="%.2f"),
        "Target1여유_%":  st.column_config.NumberColumn("목표1여유(%)",     format="%.2f"),
        "Now밴드거리_%":  st.column_config.NumberColumn("Now-Entry(%)",   format="%.2f"),
        "EV_R":          st.column_config.NumberColumn("EV_R",           format="%.3f"),
        "ERS":           st.column_config.NumberColumn("ERS",            format="%.3f"),
        "EV_SCORE":      st.column_config.NumberColumn("EV_SCORE",       format="%.3f"),
        # 체결성/규모
        "거래대금(억원)":  st.column_config.NumberColumn("거래대금(억원)",   format="%,.0f"),
        "시가총액(억원)":  st.column_config.NumberColumn("시가총액(억원)",   format="%,.0f"),
        # 기술지표
        "RSI14":        st.column_config.NumberColumn("RSI14",          format="%.1f"),
        "乖離%":         st.column_config.NumberColumn("乖離%",           format="%.2f"),
        "MACD_hist":    st.column_config.NumberColumn("MACD_hist",      format="%.4f"),
        "MACD_slope":   st.column_config.NumberColumn("MACD_slope",     format="%.5f"),
        "Vol_Z":        st.column_config.NumberColumn("Vol_Z",          format="%.2f"),
        "ret_5d_%":     st.column_config.NumberColumn("5일수익률(%)",      format="%.2f"),
        "ret_10d_%":    st.column_config.NumberColumn("10일수익률(%)",     format="%.2f"),
        # 플래그
        "REGIME_OK":    st.column_config.CheckboxColumn("레짐양호"),
        "EVENT_RISK":   st.column_config.CheckboxColumn("이벤트근접"),
    },
)

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view[cols].head(int(topn)).to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_top_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ 스코어/필터 설명", expanded=False):
    st.markdown("""
- **EBS(0~7)**: RSI 45~65 / MACD↑ / MA20 근처 / 거래량↑ / 상승구조 / MACD>0 / 5d<10%
- **RR1**: (목표1−엔트리) / (엔트리−스탑)
- **EV_R**: (RR1−1)/(RR1+1)  *(드리프트 0 가정 기대R)*
- **EV_SCORE**: EV_R × (EBS/7)
- **TopPick**: EBS≥4 ∧ RR1≥1.2 ∧ 손절여유≥1% ∧ 목표1여유≥6% ∧ (레짐 양호) ∧ (이벤트 위험 없음)에서  
  EV_SCORE/ERS/거래대금 순으로 **상관노출(20일 상관>0.65) 제한** 적용한 상위 후보
""")
