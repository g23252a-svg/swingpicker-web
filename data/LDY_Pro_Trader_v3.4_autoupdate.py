# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.4.1 (Auto Update + EV/ERS + Now/Entry Fix + Safe Editor)
- 추천 CSV: data/recommend_latest.csv (remote 우선)
- 이름맵:   data/krx_codes.csv (remote 우선) → FDR → pykrx 순 폴백
- EV_SCORE / ERS / Now-Entry / RR / Buffer 계산 및 Top Picks 내장
- Streamlit data_editor 타입/포맷 안전 보강
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

PASS_SCORE = 4  # EBS 통과 기준

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

def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi14(close: pd.Series, period=14):
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    au, ad = up.rolling(period).mean(), dn.rolling(period).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100/(1+rs)

def macd_feats(close: pd.Series):
    e12, e26 = ema(close,12), ema(close,26)
    macd = e12 - e26
    sig  = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd - sig
    return hist, hist.diff()

def atr14(h, l, c, period=14):
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

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

# --------- enrich from OHLCV ----------
@st.cache_data(ttl=300)
def enrich_from_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    need = {"종목코드","날짜","시가","고가","저가","종가"}
    if not need.issubset(set(raw.columns)):
        return raw
    raw = raw.sort_values(["종목코드","날짜"])
    g = raw.groupby("종목코드", group_keys=False)

    def _feat(x: pd.DataFrame):
        x = x.copy()
        x["MA20"] = x["종가"].rolling(20).mean()
        x["ATR14"] = atr14(x["고가"], x["저가"], x["종가"], 14)
        x["RSI14"] = rsi14(x["종가"])
        hist, slope = macd_feats(x["종가"]); x["MACD_hist"], x["MACD_slope"] = hist, slope
        x["Vol_Z"] = (x["거래량"] - x["거래량"].rolling(20).mean())/x["거래량"].rolling(20).std()
        x["乖離%"] = (x["종가"]/x["MA20"] - 1)*100
        x["ret_5d_%"]  = (x["종가"]/x["종가"].shift(5)  - 1)*100
        x["ret_10d_%"] = (x["종가"]/x["종가"].shift(10) - 1)*100

        last = x.iloc[-1:].copy()
        e, why = 0, []
        def nz(v): 
            return not (isinstance(v,float) and math.isnan(v))
        rsi = last["RSI14"].iloc[0];      c1 = nz(rsi) and 45<=rsi<=65;  e+=int(c1); why.append("RSI 45~65" if c1 else "")
        c2 = nz(last["MACD_slope"].iloc[0]) and last["MACD_slope"].iloc[0] > 0; e+=int(c2); why.append("MACD↑" if c2 else "")
        close, ma20 = last["종가"].iloc[0], last["MA20"].iloc[0]
        c3 = nz(ma20) and (0.99*ma20 <= close <= 1.04*ma20); e+=int(c3); why.append("MA20±4%" if c3 else "")
        c4 = nz(last["Vol_Z"].iloc[0]) and last["Vol_Z"].iloc[0] > 1.2; e+=int(c4); why.append("VolZ>1.2" if c4 else "")
        m20p = x["MA20"].iloc[-2] if len(x)>=2 else np.nan
        c5 = nz(m20p) and (last["MA20"].iloc[0] - m20p > 0); e+=int(c5); why.append("MA20↑" if c5 else "")
        c6 = nz(last["MACD_hist"].iloc[0]) and last["MACD_hist"].iloc[0] > 0; e+=int(c6); why.append("MACD>0" if c6 else "")
        r5 = last["ret_5d_%"].iloc[0];    c7 = nz(r5) and r5 < 10;        e+=int(c7); why.append("5d<10%" if c7 else "")
        last["EBS"] = e; last["근거"] = " / ".join([w for w in why if w])

        atr = last["ATR14"].iloc[0]
        if any([not nz(atr), not nz(ma20), not nz(close)]) or atr <= 0:
            entry=t1=t2=stp=np.nan
        else:
            band_lo, band_hi = ma20-0.5*atr, ma20+0.5*atr
            entry = min(max(close, band_lo), band_hi)
            t1, t2, stp = entry+1.0*atr, entry+1.8*atr, entry-1.2*atr
        last["추천매수가"] = round(entry,2) if not math.isnan(entry) else np.nan
        last["추천매도가1"] = round(t1,2)   if not math.isnan(t1)    else np.nan
        last["추천매도가2"] = round(t2,2)   if not math.isnan(t2)    else np.nan
        last["손절가"]     = round(stp,2)   if not math.isnan(stp)   else np.nan
        return last

    try:
        out = g.apply(_feat, include_groups=False).reset_index(drop=True)
    except TypeError:
        out = g.apply(_feat).reset_index(drop=True)

    tail = raw.groupby("종목코드").tail(1).copy()
    tail = ensure_turnover(tail)
    if "거래대금(억원)" in tail.columns:
        out = out.merge(tail[["종목코드","거래대금(억원)"]], on="종목코드", how="left")
    if "시가총액(억원)" not in out.columns:
        out["시가총액(억원)"] = np.nan
    if "시장" not in out.columns:
        out["시장"] = "ALL"
    return out

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

# -------- load raw --------
try:
    df_raw = load_csv_url(RAW_URL); log_src(df_raw, "remote", RAW_URL)
except Exception:
    if os.path.exists(LOCAL_RAW):
        df_raw = load_csv_path(LOCAL_RAW); log_src(df_raw, "local", LOCAL_RAW)
    else:
        st.error("❌ CSV가 없습니다. Actions에서 collector가 data/recommend_latest.csv를 올렸는지 확인하세요.")
        st.stop()

df_raw = normalize_cols(df_raw)

# 이미 완제품인지 체크
has_ebs  = "EBS" in df_raw.columns and df_raw["EBS"].notna().any()
has_reco = all(c in df_raw.columns for c in ["추천매수가","추천매도가1","추천매도가2","손절가"]) and \
           df_raw[["추천매수가","추천매도가1","추천매도가2","손절가"]].notna().any().any()

if has_ebs and has_reco:
    df = df_raw.copy()
else:
    with st.status("🧮 원시 OHLCV → 지표/점수/추천가 생성 중...", expanded=False):
        df = enrich_from_ohlcv(df_raw)

# 최신 행만
latest = df.sort_values(["종목코드","날짜"]).groupby("종목코드").tail(1) if "날짜" in df.columns else df.copy()

# 이름 매핑 (레포/ FDR / pykrx)
with st.status("🏷️ 종목명 매핑 중...", expanded=False):
    latest = apply_names(latest)

# 숫자 캐스팅 & 거래대금 보강
latest = ensure_turnover(latest)
for c in ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가"]:
    if c in latest.columns:
        latest[c] = pd.to_numeric(latest[c], errors="coerce")

# =================== 파생지표(Entry/Now/RR/EV/ERS) ===================
def clip01(v): 
    return np.minimum(np.maximum(v, 0.0), 1.0)

view = latest.copy()

# Now/Entry 안전 계산
view["NOW"]   = pd.to_numeric(view.get("종가", np.nan), errors="coerce")
view["ENTRY"] = pd.to_numeric(view.get("추천매수가", np.nan), errors="coerce")
view["STOP"]  = pd.to_numeric(view.get("손절가", np.nan), errors="coerce")
view["T1"]    = pd.to_numeric(view.get("추천매도가1", np.nan), errors="coerce")

den = view["ENTRY"].where(view["ENTRY"] > 0)
view["NOW_ENTRY_%"] = ((view["NOW"] - den) / den * 100)
view.loc[den.isna(), "NOW_ENTRY_%"] = np.nan

valid_rr = (view["ENTRY"] > 0) & (view["STOP"] > 0) & (view["ENTRY"] > view["STOP"])
view["RR"] = np.where(valid_rr, (view["T1"] - view["ENTRY"]) / (view["ENTRY"] - view["STOP"]), np.nan)

view["STOP_BUF_%"] = np.where(view["ENTRY"] > 0, (view["ENTRY"] - view["STOP"]) / view["ENTRY"] * 100, np.nan)
view["T1_BUF_%"]   = np.where(view["ENTRY"] > 0, (view["T1"] - view["ENTRY"]) / view["ENTRY"] * 100, np.nan)

# ERS: Entry Readiness Score (0~1)
# - 중립 RSI(55)에서 얼마나 가까운가, EBS(0~7) 정규화, Now-Entry 근접도
rsi = pd.to_numeric(view.get("RSI14", np.nan), errors="coerce")
ers_rsi = 1 - (np.abs(rsi - 55) / 15)               # 55±15 범위 → 0~1
ers_rsi = clip01(ers_rsi)

ebs = pd.to_numeric(view.get("EBS", np.nan), errors="coerce")
ers_ebs = clip01(ebs / 7.0)

near = 1 - (np.abs(view["NOW_ENTRY_%"]) / 3.0)      # ±3% 이내 선호
ers_near = clip01(near)

view["ERS"] = (0.4*ers_ebs + 0.4*ers_near + 0.2*ers_rsi).round(3)

# EV_SCORE: 종합 스코어 (0~100)
rr_scaled   = clip01(view["RR"] / 2.0)              # RR 2.0까지 선형
t1buf_scaled= clip01(view["T1_BUF_%"] / 8.0)        # 목표1 여유 8%까지 선형
stopbuf_scaled = clip01(view["STOP_BUF_%"] / 4.0)   # 손절여유 4%까지 선형

view["EV_SCORE"] = (
    100 * (0.50*view["ERS"].fillna(0)
           + 0.25*rr_scaled.fillna(0)
           + 0.15*t1buf_scaled.fillna(0)
           + 0.10*stopbuf_scaled.fillna(0))
).round(1)

# =================== UI ===================
mode = st.radio("보기 모드", ["Top Picks", "전체 보기"], horizontal=True, index=0)

with st.expander("🔍 보기/필터", expanded=True):
    c1,c2,c3 = st.columns([1,1,2])
    with c1:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 0, step=50)
    with c2:
        sort_key = st.selectbox("정렬", ["EV_SCORE▼","EBS▼","거래대금▼","시가총액▼","RSI▲","RSI▼","종가▲","종가▼"], index=0)
    with c3:
        topn = st.slider("표시 수(Top N)", 10, 500, 10, step=10)
    q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

now_band = 0.5
if mode == "Top Picks":
    with st.expander("🛠 Top Picks 조건", expanded=True):
        c1,c2,c3 = st.columns(3)
        with c1:
            min_rr = st.slider("최소 RR(목표1/손절)", 1.00, 3.00, 1.00, 0.05)
            min_stop = st.slider("손절여유 ≥ (%)", 0.00, 5.00, 0.00, 0.25)
        with c2:
            min_t1 = st.slider("목표1여유 ≥ (%)", 0.00, 10.00, 0.00, 0.5)
            min_ers = st.slider("ERS ≥", 0.00, 1.00, 0.00, 0.01)
        with c3:
            now_band = st.slider("Now 근접 밴드(±%)", 0.00, 3.00, 0.50, 0.05)

# 공통 필터링
view = view.copy()
if "거래대금(억원)" in view.columns:
    view = view[view["거래대금(억원)"] >= float(min_turn)]

if q_text:
    q = q_text.strip().lower()
    view = view[
        view["종목명"].fillna("").astype(str).str.lower().str.contains(q) |
        view["종목코드"].fillna("").astype(str).str.contains(q)
    ]

# 근접 플래그
view["NEAR_ENTRY"] = (view["NOW_ENTRY_%"].abs() <= float(now_band))

# Top Picks 고급 필터
if mode == "Top Picks":
    cond = (
        (view["EBS"] >= PASS_SCORE) &
        (view["NEAR_ENTRY"].fillna(False)) &
        (view["RR"] >= float(min_rr)) &
        (view["STOP_BUF_%"] >= float(min_stop)) &
        (view["T1_BUF_%"] >= float(min_t1)) &
        (view["ERS"] >= float(min_ers))
    )
    view = view[cond]

# 정렬
def safe_sort(dfv, key):
    try:
        if key=="EV_SCORE▼" and "EV_SCORE" in dfv.columns:
            return dfv.sort_values(["EV_SCORE","EBS","거래대금(억원)"], ascending=[False,False,False])
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

view = safe_sort(view, sort_key)

# 통과 마크
if "EBS" in view.columns:
    view["통과"] = np.where(view["EBS"]>=PASS_SCORE, "🚀", "")

# 보여줄 컬럼
cols = [
    "통과","시장","종목명","종목코드",
    "NOW","ENTRY","NOW_ENTRY_%","RR","STOP_BUF_%","T1_BUF_%","ERS","EV_SCORE","NEAR_ENTRY",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "거래대금(억원)","시가총액(억원)",
    "EBS","근거",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
for c in cols:
    if c not in view.columns: view[c] = np.nan

st.write(f"📋 총 {len(latest):,}개 / 표시 {min(len(view), int(topn)):,}개")

# ── 포맷팅용 캐스팅 ──
view_fmt = view[cols].head(int(topn)).copy()

# 정수형(천단위 콤마)
for c in ["NOW","ENTRY","종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce").round(0).astype("Int64")

# 실수형
for c in ["NOW_ENTRY_%","RR","STOP_BUF_%","T1_BUF_%","ERS","EV_SCORE",
          "거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"]:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce")

# --- column_config를 안전하게 구성 (존재할 때만) ---
cc = {}
def add_num(key, label, fmt):
    if key in view_fmt.columns:
        cc[key] = st.column_config.NumberColumn(label, format=fmt)
def add_txt(key, label):
    if key in view_fmt.columns:
        cc[key] = st.column_config.TextColumn(label)
def add_chk(key, label):
    if key in view_fmt.columns:
        # Boolean이 아닐 수 있어 to_bool 시도
        vb = view_fmt[key].astype("bool", errors="ignore") if hasattr(view_fmt[key], "astype") else view_fmt[key]
        view_fmt[key] = vb
        cc[key] = st.column_config.CheckboxColumn(label)

# 텍스트
add_txt("통과"," ")
add_txt("시장","시장")
add_txt("종목명","종목명")
add_txt("종목코드","종목코드")
add_txt("근거","근거")

# 체크박스
add_chk("NEAR_ENTRY", "Now 근접")

# 숫자 포맷
add_num("NOW",          "Now(종가)",         "%,d")
add_num("ENTRY",        "Entry(추천)",       "%,d")
add_num("NOW_ENTRY_%",  "Now↔Entry(%)",     "%.2f")
add_num("RR",           "RR(T1/Stop)",      "%.2f")
add_num("STOP_BUF_%",   "손절여유(%)",      "%.2f")
add_num("T1_BUF_%",     "목표1여유(%)",     "%.2f")
add_num("ERS",          "ERS",              "%.3f")
add_num("EV_SCORE",     "EV_SCORE",         "%.1f")

add_num("종가",          "종가",              "%,d")
add_num("추천매수가",    "추천매수가",        "%,d")
add_num("손절가",        "손절가",            "%,d")
add_num("추천매도가1",   "추천매도가1",       "%,d")
add_num("추천매도가2",   "추천매도가2",       "%,d")

add_num("거래대금(억원)", "거래대금(억원)",     "%,.0f")
add_num("시가총액(억원)", "시가총액(억원)",     "%,.0f")

add_num("EBS",          "EBS",              "%d")
add_num("RSI14",        "RSI14",            "%.1f")
add_num("乖離%",         "乖離%",              "%.2f")
add_num("MACD_hist",    "MACD_hist",        "%.4f")
add_num("MACD_slope",   "MACD_slope",       "%.5f")
add_num("Vol_Z",        "Vol_Z",            "%.2f")
add_num("ret_5d_%",     "ret_5d_%",         "%.2f")
add_num("ret_10d_%",    "ret_10d_%",        "%.2f")

st.data_editor(
    view_fmt,
    width="stretch",
    height=640,
    hide_index=True,
    disabled=True,
    num_rows="fixed",
    column_config=cc,
)

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view[cols].head(int(topn)).to_csv(index=False, encoding="utf-8-sig"),
    file_name=("ldy_top_picks.csv" if mode=="Top Picks" else "ldy_entry_candidates.csv"),
    mime="text/csv"
)

with st.expander("ℹ️ 점수/지표 설명", expanded=False):
    st.markdown("""
**EBS(0~7)**: RSI 45~65 / MACD↑ / MA20±4% / VolZ>1.2 / MA20↑ / MACD>0 / 5d<10%  
**ERS(0~1)**: 0.4·(EBS/7) + 0.4·(Now-Entry 근접도) + 0.2·(RSI 중립(55) 근접도)  
**EV_SCORE(0~100)**: 50%·ERS + 25%·RR(≤2 정규화) + 15%·T1 여유(≤8%) + 10%·손절여유(≤4%)  
**RR**: (목표1−Entry) / (Entry−손절)  
**Now↔Entry(%)**: Now가 Entry 대비 얼마나 이탈했는지(±%)
""")
