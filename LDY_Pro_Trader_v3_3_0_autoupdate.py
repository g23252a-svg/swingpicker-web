# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.3.1 (Auto Update + Name Enrich + Safe Sort/Turnover Fallback)
- GitHub raw CSV 우선 로드, 실패시 로컬 data/recommend_latest.csv 폴백
- CSV가 원시 OHLCV만 있어도 화면에서 RSI/MACD/ATR/MA/VolZ/수익률 → EBS/추천가 산출
- 종목명 없으면 pykrx로 실시간 매핑(6자리 0패딩 포함, 캐시)
- 거래대금(억원) 없을 때도 안전 보강: 거래대금(원) 또는 거래량*종가로 계산
- 정렬 시 컬럼 없으면 안전 폴백( KeyError 방지 )
- 'use_container_width' 경고 대응: width="stretch" 사용
"""

import os, io, math, requests, numpy as np, pandas as pd, streamlit as st
from datetime import datetime

# pykrx(선택): 없더라도 앱은 동작(이름 매핑만 생략)
try:
    from pykrx import stock
    PYKRX_OK = True
except Exception:
    PYKRX_OK = False

st.set_page_config(page_title="LDY Pro Trader v3.3.1 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.3.1 (Auto Update)")
st.caption("매일 장마감 후 자동 업데이트되는 스윙 추천 종목 리스트 | Made by LDY")

RAW_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_PATH = "data/recommend_latest.csv"
PASS_SCORE = 4

# ------------------------- IO -------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_remote_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=300, show_spinner=False)
def load_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def info_src(df: pd.DataFrame, src_text: str):
    st.info(f"상태\n✅ 데이터 로드: {src_text}\n\n{RAW_URL if 'remote' in src_text else LOCAL_PATH}")
    st.success(f"📅 추천 기준(표시 시각): {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M')} · 원시 행수: {len(df):,}")

# ------------------------- 지표 -------------------------
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi14(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd_features(close: pd.Series):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd_line - signal
    slope = hist.diff()
    return hist, slope

def atr14(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ------------------------- 정규화/보강 -------------------------
def z6(x) -> str:
    s = str(x)
    return s.zfill(6) if s.isdigit() else s

def ensure_turnover_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    거래대금(억원) 보강:
    1) 거래대금(억원) 있으면 유지
    2) 거래대금(원) 있으면 /1e8
    3) 거래량 & 종가 있으면 거래량*종가 /1e8
    """
    if "거래대금(억원)" not in df.columns:
        base = None
        if "거래대금(원)" in df.columns:
            base = pd.to_numeric(df["거래대금(원)"], errors="coerce")
        elif all(c in df.columns for c in ["거래량","종가"]):
            vol = pd.to_numeric(df["거래량"], errors="coerce")
            cls = pd.to_numeric(df["종가"], errors="coerce")
            base = vol * cls
        if base is not None:
            df["거래대금(억원)"] = (base / 1e8).round(2)
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼명 통일
    colmap = {
        "Date":"날짜","date":"날짜",
        "Code":"종목코드","티커":"종목코드","ticker":"종목코드",
        "Name":"종목명","name":"종목명",
        "Open":"시가","High":"고가","Low":"저가","Close":"종가","Volume":"거래량",
        "거래대금":"거래대금(원)",  # pykrx 일괄 대응
        "시가총액":"시가총액(원)"
    }
    for k,v in colmap.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})

    # 타입 캐스팅
    for c in ["시가","고가","저가","종가","거래량","거래대금(원)","시가총액(원)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 날짜
    if "날짜" in df.columns:
        try:
            df["날짜"] = pd.to_datetime(df["날짜"])
        except Exception:
            pass

    # 코드 6자리
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.replace(".0","", regex=False).map(z6)
    else:
        df["종목코드"] = None

    # 시장/종목명 기본값
    if "시장" not in df.columns:
        df["시장"] = "ALL"
    if "종목명" not in df.columns:
        df["종목명"] = None

    # 거래대금(억원) 보강
    df = ensure_turnover_cols(df)

    return df

@st.cache_data(ttl=300, show_spinner=True)
def enrich_from_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    must = {"종목코드","날짜","시가","고가","저가","종가"}
    if not must.issubset(set(raw.columns)):
        return raw
    raw = raw.sort_values(["종목코드","날짜"])
    g = raw.groupby("종목코드", group_keys=False)

    def _feat(x: pd.DataFrame):
        x = x.copy()
        x["MA20"] = x["종가"].rolling(20).mean()
        x["ATR14"] = atr14(x["고가"], x["저가"], x["종가"], 14)
        x["RSI14"] = rsi14(x["종가"], 14)
        hist, slope = macd_features(x["종가"])
        x["MACD_hist"] = hist
        x["MACD_slope"] = slope
        x["Vol_Z"] = (x["거래량"] - x["거래량"].rolling(20).mean()) / x["거래량"].rolling(20).std()
        x["乖離%"] = (x["종가"]/x["MA20"] - 1.0)*100
        x["ret_5d_%"] = (x["종가"]/x["종가"].shift(5) - 1.0)*100
        x["ret_10d_%"] = (x["종가"]/x["종가"].shift(10) - 1.0)*100

        last = x.iloc[-1:].copy()
        e = 0; why=[]
        def nz(v, fallback=-999): 
            return v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else fallback

        rsi_v = nz(last["RSI14"].iloc[0])
        c1 = 45 <= rsi_v <= 65; e += int(c1);  why.append("RSI 45~65" if c1 else "")
        c2 = nz(last["MACD_slope"].iloc[0]) > 0; e+=int(c2); why.append("MACD↑" if c2 else "")
        close, ma20 = last["종가"].iloc[0], last["MA20"].iloc[0]
        c3 = (not math.isnan(ma20)) and (0.99*ma20 <= close <= 1.04*ma20); e+=int(c3); why.append("MA20±4%" if c3 else "")
        c4 = nz(last["Vol_Z"].iloc[0]) > 1.2; e+=int(c4); why.append("VolZ>1.2" if c4 else "")
        m20_prev = x["MA20"].iloc[-2] if len(x)>=2 else np.nan
        c5 = (not math.isnan(m20_prev)) and (last["MA20"].iloc[0] - m20_prev > 0); e+=int(c5); why.append("MA20↑" if c5 else "")
        c6 = nz(last["MACD_hist"].iloc[0]) > 0; e+=int(c6); why.append("MACD>0" if c6 else "")
        r5 = last["ret_5d_%"].iloc[0]; c7 = (not math.isnan(r5)) and (r5 < 10); e+=int(c7); why.append("5d<10%" if c7 else "")

        last["EBS"] = e
        last["근거"] = " / ".join([w for w in why if w])

        atr = last["ATR14"].iloc[0]
        if math.isnan(atr) or math.isnan(ma20) or math.isnan(close) or atr <= 0:
            entry=t1=t2=stp=np.nan
        else:
            band_low, band_high = ma20 - 0.5*atr, ma20 + 0.5*atr
            entry = min(max(close, band_low), band_high)
            t1, t2, stp = entry + 1.0*atr, entry + 1.8*atr, entry - 1.2*atr
        last["추천매수가"] = round(entry,2) if not math.isnan(entry) else np.nan
        last["추천매도가1"] = round(t1,2) if not math.isnan(t1) else np.nan
        last["추천매도가2"] = round(t2,2) if not math.isnan(t2) else np.nan
        last["손절가"] = round(stp,2) if not math.isnan(stp) else np.nan
        return last

    out = g.apply(_feat).reset_index(drop=True)

    # 거래대금(억원) 최신행 보강
    if "거래대금(억원)" not in out.columns:
        # raw에서 최신행 추출 후 계산
        tail = raw.groupby("종목코드").tail(1).copy()
        tail = ensure_turnover_cols(tail)
        if "거래대금(억원)" in tail.columns:
            out = out.merge(tail[["종목코드","거래대금(억원)"]], on="종목코드", how="left")

    if "시가총액(억원)" not in out.columns:
        out["시가총액(억원)"] = np.nan
    if "시장" not in out.columns:
        out["시장"] = "ALL"
    return out

# ------------------------- 종목명 매핑 -------------------------
@st.cache_data(ttl=6*60*60, show_spinner=False)
def name_map_from_pykrx(codes: list[str]) -> dict:
    """pykrx로 코드→이름 매핑(캐시 6시간). 실패는 None."""
    if not PYKRX_OK:
        return {}
    out = {}
    for t in codes:
        try:
            out[t] = stock.get_market_ticker_name(t)
        except Exception:
            out[t] = None
    return out

def fill_names(df: pd.DataFrame) -> pd.DataFrame:
    if "종목코드" not in df.columns:
        return df
    need = df[df["종목명"].isna() | (df["종목명"]=="")]["종목코드"].dropna().astype(str).map(z6).unique().tolist()
    if len(need)==0:
        return df
    m = name_map_from_pykrx(need) if PYKRX_OK else {}
    if not m and not PYKRX_OK:
        st.warning("pykrx가 없어서 종목명 매핑을 건너뜁니다. (requirements.txt 확인)")
        df.loc[df["종목명"].isna() | (df["종목명"]==""), "종목명"] = "(이름없음)"
        return df
    df["종목명"] = df["종목명"].where(df["종목명"].notna() & (df["종목명"]!=""), df["종목코드"].map(m))
    df["종목명"] = df["종목명"].fillna("(이름없음)")
    return df

# ------------------------- 로드 & 가공 -------------------------
try:
    df_raw = load_remote_csv(RAW_URL)
    info_src(df_raw, "remote")
except Exception:
    if os.path.exists(LOCAL_PATH):
        df_raw = load_local_csv(LOCAL_PATH)
        info_src(df_raw, "local")
    else:
        st.error("❌ CSV를 찾을 수 없습니다. collector가 data/recommend_latest.csv를 올렸는지 확인하세요.")
        st.stop()

df_raw = normalize_columns(df_raw)

# EBS/추천가 존재 여부
has_ebs = "EBS" in df_raw.columns and df_raw["EBS"].notna().any()
has_reco = all(c in df_raw.columns for c in ["추천매수가","추천매도가1","추천매도가2","손절가"]) and \
           df_raw[["추천매수가","추천매도가1","추천매도가2","손절가"]].notna().any().any()

if has_ebs and has_reco:
    df = df_raw.copy()
else:
    with st.status("🧮 원시 OHLCV → 지표/점수/추천가 생성 중...", expanded=False):
        df = enrich_from_ohlcv(df_raw)

# 최신 일자만 집계
if "날짜" in df.columns:
    latest_by_code = df.sort_values(["종목코드","날짜"]).groupby("종목코드").tail(1).copy()
else:
    latest_by_code = df.copy()

# 종목명 매핑
with st.status("🏷️ 종목명 매핑 중...", expanded=False):
    latest_by_code = fill_names(latest_by_code)

# 안전 캐스팅 + 거래대금(억원) 최종 보강(한 번 더)
latest_by_code = ensure_turnover_cols(latest_by_code)
for c in ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가"]:
    if c in latest_by_code.columns:
        latest_by_code[c] = pd.to_numeric(latest_by_code[c], errors="coerce")

# ------------------------- UI -------------------------
with st.expander("🔍 보기/필터", expanded=True):
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,2])

    default_entry = True
    if "EBS" not in latest_by_code.columns or latest_by_code["EBS"].notna().sum()==0:
        default_entry = False
        st.warning("EBS 점수가 없어 ‘🚀 초입 후보만’ 필터를 자동 해제합니다. (원시 OHLCV 계산 실패/데이터 부족)")
    with c1:
        only_entry = st.checkbox("🚀 초입 후보만 (EBS≥4)", value=default_entry)
    with c2:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c3:
        sort_key = st.selectbox("정렬",
            ["EBS▼","거래대금▼","시가총액▼","RSI▲","RSI▼","종가▲","종가▼"],
            index=0 if "EBS" in latest_by_code.columns else 1)
    with c4:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)
    with c5:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

view = latest_by_code.copy()

# 필터
if only_entry and "EBS" in view.columns:
    view = view[view["EBS"] >= PASS_SCORE]
if "거래대금(억원)" in view.columns:
    view = view[view["거래대금(억원)"] >= float(min_turn)]

if q_text:
    q = q_text.strip().lower()
    name_hit = view["종목명"].fillna("").astype(str).str.lower().str.contains(q, na=False)
    code_hit = view["종목코드"].fillna("").astype(str).str.contains(q, na=False)
    view = view[name_hit | code_hit]

# 안전 정렬
def safe_sort(dfv: pd.DataFrame, key: str) -> pd.DataFrame:
    try:
        if key == "EBS▼" and "EBS" in dfv.columns:
            by = ["EBS"] + (["거래대금(억원)"] if "거래대금(억원)" in dfv.columns else [])
            return dfv.sort_values(by=by, ascending=[False] + [False]* (len(by)-1))
        if key == "거래대금▼" and "거래대금(억원)" in dfv.columns:
            return dfv.sort_values("거래대금(억원)", ascending=False)
        if key == "시가총액▼" and "시가총액(억원)" in dfv.columns:
            return dfv.sort_values("시가총액(억원)", ascending=False, na_position="last")
        if key == "RSI▲" and "RSI14" in dfv.columns:
            return dfv.sort_values("RSI14", ascending=True, na_position="last")
        if key == "RSI▼" and "RSI14" in dfv.columns:
            return dfv.sort_values("RSI14", ascending=False, na_position="last")
        if key == "종가▲" and "종가" in dfv.columns:
            return dfv.sort_values("종가", ascending=True, na_position="last")
        if key == "종가▼" and "종가" in dfv.columns:
            return dfv.sort_values("종가", ascending=False, na_position="last")
    except Exception:
        pass
    # 폴백: 가능한 컬럼 우선순위
    for alt in ["EBS","거래대금(억원)","시가총액(억원)","종가"]:
        if alt in dfv.columns:
            return dfv.sort_values(alt, ascending=False, na_position="last")
    return dfv

view = safe_sort(view, sort_key)

# 표시 컬럼
show_cols = [
    "통과","시장","종목명","종목코드",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "거래대금(억원)","시가총액(억원)",
    "EBS","근거",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
if "EBS" in view.columns:
    view["통과"] = np.where(view["EBS"]>=PASS_SCORE, "🚀", "")

for c in show_cols:
    if c not in view.columns:
        view[c] = np.nan

st.write(f"📋 총 {len(latest_by_code):,}개 / 표시 {min(len(view), int(topn)):,}개")
st.dataframe(view[show_cols].head(int(topn)), width="stretch", height=640)

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view[show_cols].head(int(topn)).to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_entry_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ EBS 구성(급등 초입 로직)", expanded=False):
    st.markdown(
        """
- 기본 컷(collector 권장): 거래대금 ≥ **50억원**, 시가총액 ≥ **1,000억원**
- 점수(0~7):
  1) RSI 45~65  
  2) MACD 히스토그램 기울기 > 0  
  3) 종가가 MA20 근처(-1%~+4%)  
  4) 상대거래량(20일) > 1.2  
  5) MA20 상승(기울기 > 0)  
  6) MACD 히스토그램 > 0  
  7) 5일 수익률 < 10%(과열 방지)  
- **통과(🚀초입)**: EBS ≥ 4  
- 추천가: ATR/MA 기반 보수적 가이드  
  - 엔트리: MA20±0.5×ATR 범위 내 스냅  
  - T1: +1.0×ATR, T2: +1.8×ATR, 손절: −1.2×ATR
        """
    )
