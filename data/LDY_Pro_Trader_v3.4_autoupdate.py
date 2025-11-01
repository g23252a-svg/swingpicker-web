# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.4 (Auto Update + EV Score + TopPick + Robust Types)
- 추천 CSV: data/recommend_latest.csv (remote 우선)
- 이름맵:   data/krx_codes.csv (remote 우선) → FDR → pykrx 순 폴백
- OHLCV만 와도 화면에서 지표/EBS/추천가 생성
- 거래대금(억원) 안전 보강, 안전 정렬
- 표 숫자에 천단위 콤마 적용 (Streamlit column_config)
- EV_SCORE/ERS/RR/Now-Entry 밴드 계산 + TopPick 뷰 & 필터
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
        x["MA60"] = x["종가"].rolling(60).mean()
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

    # 거래대금(억원) 최신행 보강
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

    # 2) FDR 폴백
    if FDR_OK:
        try:
            lst = fdr.StockListing("KRX")
            m = lst.rename(columns={"Code":"종목코드","Name":"종목명"})[["종목코드","종목명"]]
            m["종목코드"] = m["종목코드"].astype(str).map(z6)
            return m.drop_duplicates("종목코드")
        except Exception:
            pass

    # 3) pykrx 개별 조회(네트워크 차단 환경이면 실패 가능)
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
for c in ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가","MA20","MA60","ATR14"]:
    if c in latest.columns:
        latest[c] = pd.to_numeric(latest[c], errors="coerce")

# ---------- 파생지표 (RR/여유/ERS/EV_SCORE/TopPick 등) ----------
def compute_derived(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    # 기본 값/보호
    for c in ["종가","추천매수가","손절가","추천매도가1"]:
        if c not in x.columns: x[c] = np.nan
    E = pd.to_numeric(x["추천매수가"], errors="coerce")
    S = pd.to_numeric(x["손절가"], errors="coerce")
    T1= pd.to_numeric(x["추천매도가1"], errors="coerce")
    NOW = pd.to_numeric(x["종가"], errors="coerce")
    denom = (E - S).replace(0, np.nan)

    # RR1 (최소 RR)
    x["RR1"] = (T1 - E) / denom
    # Stop/Target 여유(%)
    x["Stop여유_%"]   = (E - S) / E * 100
    x["Target1여유_%"] = (T1 - E) / E * 100
    # Now vs Entry 거리(%)
    x["Now밴드거리_%"] = (NOW - E) / E * 100

    # ERS(Entry Readiness Score) 0~1
    ebs = pd.to_numeric(x.get("EBS", np.nan), errors="coerce").fillna(0.0)
    ebs_norm = (ebs / 7.0).clip(0,1)
    turn = pd.to_numeric(x.get("거래대금(억원)", np.nan), errors="coerce").fillna(0.0)
    vol_norm = (turn / 1000.0).clip(0,1)  # 1000억에서 포화
    # 구조: MA20>MA60 or '상승구조' 키워드
    ma20 = pd.to_numeric(x.get("MA20", np.nan), errors="coerce")
    ma60 = pd.to_numeric(x.get("MA60", np.nan), errors="coerce")
    has_up_struct = (ma20 > ma60)
    if "근거" in x.columns:
        has_up_struct = has_up_struct | x["근거"].astype(str).str.contains("상승구조", na=False)
    struct = has_up_struct.astype(float)
    # 엔트리 근접도(±3% 이내 가중)
    dist = (x["Now밴드거리_%"].abs() / 3.0).clip(lower=0)   # 3% 밖은 1.0 이상
    near = (1.0 - dist).clip(0,1)

    x["ERS"] = (0.4*ebs_norm + 0.2*vol_norm + 0.2*struct + 0.2*near).clip(0,1)

    # EV_SCORE (0~100): ERS, RR1, Stop여유 결합
    rr_norm = (x["RR1"] / 2.0).clip(0,1)             # RR1=2 → 1.0
    sb_norm = (x["Stop여유_%"] / 3.0).clip(0,1)      # Stop여유 3%에서 포화
    x["EV_SCORE"] = (100*(0.5*x["ERS"] + 0.3*rr_norm + 0.2*sb_norm)).round(1)

    # 체크박스 계열(없으면 기본값)
    if "REGIME_OK" not in x.columns:
        x["REGIME_OK"] = (ebs >= PASS_SCORE)  # 간단 버전: EBS 통과시 True
    if "EVENT_RISK" not in x.columns:
        x["EVENT_RISK"] = False

    # TopPick 규칙
    cond_toppick = (
        (ebs >= PASS_SCORE) &
        (x["RR1"] >= 1.5) &
        (x["Now밴드거리_%"].abs() <= 1.5) &
        (x["Stop여유_%"] >= 1.5) &
        (x["ERS"] >= 0.60) &
        (x["EV_SCORE"] >= 60)
    )
    x["TopPick"] = cond_toppick
    return x

latest = compute_derived(latest)

# ------------- UI -------------
mode = st.radio("보기 모드", ["Top Picks", "전체 보기"], horizontal=True, index=0)

with st.expander("🔍 보기/필터", expanded=True):
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,2])
    with c1:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c2:
        sort_key = st.selectbox("정렬", ["EV_SCORE▼","EBS▼","거래대금▼","시가총액▼","RSI▲","RSI▼","종가▲","종가▼"], index=0)
    with c3:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)
    with c5:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

with st.expander("🛠 Top Picks 조건", expanded=(mode=="Top Picks")):
    d1,d2,d3,d4,d5 = st.columns(5)
    with d1:
        rr_min = st.slider("최소 RR(목표1/손절)", 1.00, 3.00, 1.50, step=0.05)
    with d2:
        sb_min = st.slider("손절여유 ≥ (%)", 0.00, 5.00, 1.50, step=0.25)
    with d3:
        t1b_min = st.slider("목표1여유 ≥ (%)", 0.00, 10.00, 0.00, step=0.5)
    with d4:
        ers_min = st.slider("ERS ≥", 0.00, 1.00, 0.60, step=0.05)
    with d5:
        band_max = st.slider("Now 근접 밴드(±%)", 0.00, 3.00, 1.50, step=0.10)

view = latest.copy()

# 모드별 기본 필터
if mode == "Top Picks":
    view = view[view["TopPick"]]

# 공통 필터
if "거래대금(억원)" in view.columns:
    view = view[view["거래대금(억원)"] >= float(min_turn)]
if q_text:
    q = q_text.strip().lower()
    view = view[
        view["종목명"].fillna("").astype(str).str.lower().str.contains(q) |
        view["종목코드"].fillna("").astype(str).str.contains(q)
    ]

# TopPick 추가 슬라이더 필터 적용
if mode == "Top Picks":
    view = view[
        (view["RR1"] >= rr_min) &
        (view["Stop여유_%"] >= sb_min) &
        (view["Target1여유_%"] >= t1b_min) &
        (view["ERS"] >= ers_min) &
        (view["Now밴드거리_%"].abs() <= band_max)
    ]

def safe_sort(dfv, key):
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

view = safe_sort(view, sort_key)

if "EBS" in view.columns:
    view["통과"] = np.where(view["EBS"]>=PASS_SCORE, "🚀", "")

cols = [
    "TopPick","REGIME_OK","EVENT_RISK",
    "통과","시장","종목명","종목코드",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "RR1","Stop여유_%","Target1여유_%","Now밴드거리_%","ERS","EV_SCORE",
    "거래대금(억원)","시가총액(억원)",
    "EBS","근거",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
for c in cols:
    if c not in view.columns: view[c]=np.nan

st.write(f"📋 총 {len(latest):,}개 / 표시 {min(len(view), int(topn)):,}개")

# ── 숫자/타입 포맷(콤마 & 체크박스) 적용을 위한 캐스팅 ──
view_fmt = view[cols].head(int(topn)).copy()

# CP949-safe 열명 역매핑 (괴리_% → 乖離%)
if "괴리_%" in view_fmt.columns and "乖離%" not in view_fmt.columns:
    view_fmt = view_fmt.rename(columns={"괴리_%": "乖離%"})

# 가격/정수류 → Int64 (NaN 허용 정수)
for c in ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce").round(0).astype("Int64")

# 억원/지표류 → float
num_cols_all = [
    "RR1","Stop여유_%","Target1여유_%","Now밴드거리_%","ERS","EV_SCORE",
    "거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
for c in num_cols_all:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce")

# 체크박스 열 보장 + boolean 캐스팅
for b in ["REGIME_OK","EVENT_RISK","TopPick"]:
    if b not in view_fmt.columns:
        view_fmt[b] = False
    view_fmt[b] = (
        view_fmt[b]
        .replace({"True": True, "False": False, "true": True, "false": False})
        .astype("boolean")
        .fillna(False)
    )

# -------- 표 렌더링 --------
st.data_editor(
    view_fmt,
    width="stretch",
    height=640,
    hide_index=True,
    disabled=True,          # 읽기 전용 표
    num_rows="fixed",
    column_config={
        # 체크박스
        "TopPick":     st.column_config.CheckboxColumn("Top", help="규칙 충족 자동 선정"),
        "REGIME_OK":   st.column_config.CheckboxColumn("Regime", help="시장/구조 양호"),
        "EVENT_RISK":  st.column_config.CheckboxColumn("Event", help="이벤트 리스크"),
        # 텍스트
        "통과":         st.column_config.TextColumn(" "),
        "시장":         st.column_config.TextColumn("시장"),
        "종목명":       st.column_config.TextColumn("종목명"),
        "종목코드":     st.column_config.TextColumn("종목코드"),
        "근거":         st.column_config.TextColumn("근거"),
        # 가격/정수(콤마)
        "종가":          st.column_config.NumberColumn("종가",           format="%,d"),
        "추천매수가":    st.column_config.NumberColumn("추천매수가",     format="%,d"),
        "손절가":        st.column_config.NumberColumn("손절가",         format="%,d"),
        "추천매도가1":   st.column_config.NumberColumn("추천매도가1",    format="%,d"),
        "추천매도가2":   st.column_config.NumberColumn("추천매도가2",    format="%,d"),
        "EBS":          st.column_config.NumberColumn("EBS",            format="%d"),
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
        # EV 계열
        "RR1":          st.column_config.NumberColumn("RR(목표1/손절)",  format="%.2f", help="(T1-Entry)/(Entry-Stop)"),
        "Stop여유_%":    st.column_config.NumberColumn("손절여유(%)",      format="%.2f"),
        "Target1여유_%": st.column_config.NumberColumn("목표1여유(%)",     format="%.2f"),
        "Now밴드거리_%":  st.column_config.NumberColumn("Now-Entry(%)",   format="%.2f"),
        "ERS":          st.column_config.NumberColumn("ERS(0~1)",       format="%.2f"),
        "EV_SCORE":     st.column_config.NumberColumn("EV_SCORE",       format="%.1f"),
    },
)

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view[cols].head(int(topn)).to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_entry_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ 점수/지표 설명", expanded=False):
    st.markdown("""
**EBS(0~7)**: RSI 45~65 / MACD↑ / MA20±4% / VolZ>1.2 / MA20↑ / MACD>0 / 5d<10%  
**RR1**: (목표1−추천매수) / (추천매수−손절) — 손절 대비 목표 보상비  
**ERS(0~1)**: EBS·유동성·상승구조·엔트리 근접도 결합한 진입 준비도  
**EV_SCORE(0~100)**: 0.5·ERS + 0.3·RR + 0.2·손절여유 의 가중 합산 점수  
**TopPick**: EBS≥4, RR1≥1.5, |Now−Entry|≤1.5%, 손절여유≥1.5%, ERS≥0.60, EV_SCORE≥60 충족
""")
