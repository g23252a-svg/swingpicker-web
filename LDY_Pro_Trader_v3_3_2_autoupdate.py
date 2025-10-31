# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.3.4 (Auto Update + Robust Name Map + Number Format + 추격 진입 모드)
- 추천 CSV: data/recommend_latest.csv (remote 우선)
- 이름맵:   data/krx_codes.csv (remote 우선) → FDR → pykrx 순 폴백
- collector CSV(완제품) 또는 원시 OHLCV 모두 지원
- '풀백' + '추격' 2가지 진입 모드 지원
- 지금 진입 유효(목표가1 ≥ 종가) 필터 & RR(목표1/손절) 필터 제공
- 표 숫자(가격/억원)에 천단위 콤마 적용 (Streamlit column_config)
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

st.set_page_config(page_title="LDY Pro Trader v3.3.4 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.3.4 (Auto Update)")
st.caption("매일 장마감 후 자동 업데이트되는 스윙 추천 종목 리스트 | Made by LDY")

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
    st.success(f"📅 추천 기준(표시 시각): {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M')} · 원시 행수: {len(df):,}")

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

def round_to_tick(price: float) -> int:
    # 단순 10원 틱 적용(간이)
    if pd.isna(price): return np.nan
    return int(round(float(price) / 10.0) * 10)

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
for c in ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가","ATR14","MA20"]:
    if c in latest.columns:
        latest[c] = pd.to_numeric(latest[c], errors="coerce")

# ---------- 추격 모드(표시용) 계산 ----------
def infer_ma20_from_disp(c: float, disp_pct: float):
    if pd.isna(c) or pd.isna(disp_pct): return np.nan
    try:
        return float(c) / (1.0 + float(disp_pct)/100.0)
    except ZeroDivisionError:
        return np.nan

def infer_atr_proxy(row: pd.Series):
    # 우선 ATR14 있으면 그걸 사용
    if "ATR14" in row and pd.notna(row["ATR14"]) and row["ATR14"] > 0:
        return float(row["ATR14"])
    buy  = pd.to_numeric(row.get("추천매수가"), errors="coerce")
    stop = pd.to_numeric(row.get("손절가"), errors="coerce")
    t1   = pd.to_numeric(row.get("추천매도가1"), errors="coerce")
    t2   = pd.to_numeric(row.get("추천매도가2"), errors="coerce")
    cand = []
    if pd.notna(buy) and pd.notna(stop) and buy > stop:
        cand.append((buy - stop) / 1.5)     # buy - stop = 1.5 * ATR
    if pd.notna(buy) and pd.notna(t1) and t1 > buy:
        cand.append((t1 - buy) / 1.5)       # t1 - buy = 1.5 * ATR
    if pd.notna(buy) and pd.notna(t2) and t2 > buy:
        cand.append((t2 - buy) / 3.0)       # t2 - buy = 3 * ATR
    if not cand:
        return np.nan
    return float(np.nanmedian(cand))

def compute_chase_set(dfv: pd.DataFrame) -> pd.DataFrame:
    out = dfv.copy()
    c   = pd.to_numeric(out.get("종가"), errors="coerce")
    disp = pd.to_numeric(out.get("乖離%"), errors="coerce")
    ma20_est = pd.Series(np.nan, index=out.index)
    if "MA20" in out.columns and out["MA20"].notna().any():
        ma20_est = pd.to_numeric(out["MA20"], errors="coerce")
    else:
        ma20_est = c / (1.0 + disp/100.0)  # 乖離%로 MA20 추정

    atr_proxy = out.apply(infer_atr_proxy, axis=1)
    # chase 계산
    buy_chase  = c
    stop_chase = np.maximum(c - 1.2*atr_proxy, ma20_est*0.97)
    t1_chase   = c + 1.0*atr_proxy
    t2_chase   = c + 1.8*atr_proxy

    # 틱 반올림
    out["추천매수가(추격)"]  = buy_chase.round(0).astype("Int64")
    out["손절가(추격)"]      = pd.Series([round_to_tick(x) for x in stop_chase], index=out.index).astype("Int64")
    out["추천매도가1(추격)"] = pd.Series([round_to_tick(x) for x in t1_chase],   index=out.index).astype("Int64")
    out["추천매도가2(추격)"] = pd.Series([round_to_tick(x) for x in t2_chase],   index=out.index).astype("Int64")

    # RR(목표1/손절)
    # 풀백 RR
    pb_buy  = pd.to_numeric(out.get("추천매수가"), errors="coerce")
    pb_stop = pd.to_numeric(out.get("손절가"), errors="coerce")
    pb_t1   = pd.to_numeric(out.get("추천매도가1"), errors="coerce")
    out["RR(풀백)"] = np.where(
        (pb_buy.notna()) & (pb_stop.notna()) & (pb_t1.notna()) & (pb_buy > pb_stop),
        (pb_t1 - pb_buy) / (pb_buy - pb_stop),
        np.nan
    )
    # 추격 RR
    ch_stop = pd.to_numeric(out["손절가(추격)"], errors="coerce")
    ch_t1   = pd.to_numeric(out["추천매도가1(추격)"], errors="coerce")
    out["RR(추격)"] = np.where(
        (c.notna()) & (ch_stop.notna()) & (ch_t1.notna()) & (c > ch_stop),
        (ch_t1 - c) / (c - ch_stop),
        np.nan
    )
    # 보조: 추정 MA20
    out["MA20(추정)"] = ma20_est
    return out

latest = compute_chase_set(latest)

# ------------- UI -------------
with st.expander("🔍 보기/필터", expanded=True):
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,2])
    with c1:
        only_entry = st.checkbox("🚀 초입 후보만 (EBS≥4)", value=("EBS" in latest.columns))
    with c2:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c3:
        sort_key = st.selectbox("정렬", ["EBS▼","거래대금▼","시가총액▼","RSI▲","RSI▼","종가▲","종가▼"], index=0)
    with c4:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)
    with c5:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

    c6, c7, c8 = st.columns([1,1,1])
    with c6:
        mode = st.radio("진입 기준", ["풀백(기본)", "추격"], horizontal=True)
    with c7:
        only_now = st.checkbox("지금 진입 유효(목표가1 ≥ 종가)", value=False)
    with c8:
        min_rr = st.slider("최소 RR(목표1/손절)", 0.0, 3.0, 0.0, step=0.1)

view = latest.copy()
if only_entry and "EBS" in view.columns:
    view = view[view["EBS"] >= PASS_SCORE]
if "거래대금(억원)" in view.columns:
    view = view[view["거래대금(억원)"] >= float(min_turn)]
if q_text:
    q = q_text.strip().lower()
    view = view[
        view["종목명"].fillna("").astype(str).str.lower().str.contains(q) |
        view["종목코드"].fillna("").astype(str).str.contains(q)
    ]

def safe_sort(dfv, key):
    try:
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
    for alt in ["EBS","거래대금(억원)","시가총액(억원)","종가"]:
        if alt in dfv.columns:
            return dfv.sort_values(alt, ascending=False, na_position="last")
    return dfv

# 지금 진입 유효 / RR 필터
if only_now:
    base_t1 = view["추천매도가1(추격)"] if (mode=="추격" and "추천매도가1(추격)" in view.columns) else view["추천매도가1"]
    view = view[ pd.to_numeric(base_t1, errors="coerce") >= pd.to_numeric(view["종가"], errors="coerce") ]

if min_rr > 0:
    rr_col = "RR(추격)" if mode=="추격" else "RR(풀백)"
    if rr_col in view.columns:
        view = view[ pd.to_numeric(view[rr_col], errors="coerce") >= float(min_rr) ]

view = safe_sort(view, sort_key)

if "EBS" in view.columns:
    view["통과"] = np.where(view["EBS"]>=PASS_SCORE, "🚀", "")

# 표시 컬럼
base_cols = [
    "통과","시장","종목명","종목코드",
    "종가","거래대금(억원)","시가총액(억원)",
    "EBS","근거","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
pullback_cols = ["추천매수가","손절가","추천매도가1","추천매도가2","RR(풀백)"]
chase_cols    = ["추천매수가(추격)","손절가(추격)","추천매도가1(추격)","추천매도가2(추격)","RR(추격)"]

cols = base_cols + pullback_cols + [c for c in chase_cols if c in view.columns]

for c in cols:
    if c not in view.columns:
        view[c] = np.nan

st.write(f"📋 총 {len(latest):,}개 / 표시 {min(len(view), int(topn)):,}개")

# ── 숫자 포맷(콤마) 적용을 위한 캐스팅 ──
view_fmt = view[cols].head(int(topn)).copy()

# 가격/정수류 → Int64 (NaN 허용 정수)
int_like_cols = [
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "추천매수가(추격)","손절가(추격)","추천매도가1(추격)","추천매도가2(추격)","EBS"
]
for c in int_like_cols:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce").round(0).astype("Int64")

# 억원/지표류 → float
float_cols = ["거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","RR(풀백)","RR(추격)"]
for c in float_cols:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce")

st.data_editor(
    view_fmt,
    width="stretch",
    height=640,
    hide_index=True,
    disabled=True,          # 읽기 전용 표
    num_rows="fixed",
    column_config={
        # 텍스트
        "통과":         st.column_config.TextColumn(" "),
        "시장":         st.column_config.TextColumn("시장"),
        "종목명":       st.column_config.TextColumn("종목명"),
        "종목코드":     st.column_config.TextColumn("종목코드"),
        "근거":         st.column_config.TextColumn("근거"),
        # 가격/정수(콤마)
        "종가":          st.column_config.NumberColumn("종가",           format="%,d"),
        "추천매수가":    st.column_config.NumberColumn("추천매수가(풀백)",format="%,d"),
        "손절가":        st.column_config.NumberColumn("손절가(풀백)",    format="%,d"),
        "추천매도가1":   st.column_config.NumberColumn("목표가1(풀백)",   format="%,d"),
        "추천매도가2":   st.column_config.NumberColumn("목표가2(풀백)",   format="%,d"),
        "EBS":          st.column_config.NumberColumn("EBS",            format="%d"),
        # 억원/지표 (콤마·소수)
        "거래대금(억원)": st.column_config.NumberColumn("거래대금(억원)",  format="%,.0f"),
        "시가총액(억원)": st.column_config.NumberColumn("시가총액(억원)",  format="%,.0f"),
        "RSI14":        st.column_config.NumberColumn("RSI14",          format="%.1f"),
        "乖離%":         st.column_config.NumberColumn("乖離%",           format="%.2f"),
        "MACD_hist":    st.column_config.NumberColumn("MACD_hist",      format="%.4f"),
        "MACD_slope":   st.column_config.NumberColumn("MACD_slope",     format="%.5f"),
        "Vol_Z":        st.column_config.NumberColumn("Vol_Z",          format="%.2f"),
        "ret_5d_%":     st.column_config.NumberColumn("ret_5d_%",       format="%.2f"),
        "ret_10d_%":    st.column_config.NumberColumn("ret_10d_%",      format="%.2f"),
        # 추격 세트(콤마)
        "추천매수가(추격)":  st.column_config.NumberColumn("추천매수가(추격)",  format="%,d"),
        "손절가(추격)":      st.column_config.NumberColumn("손절가(추격)",      format="%,d"),
        "추천매도가1(추격)": st.column_config.NumberColumn("목표가1(추격)",     format="%,d"),
        "추천매도가2(추격)": st.column_config.NumberColumn("목표가2(추격)",     format="%,d"),
        # RR
        "RR(풀백)":     st.column_config.NumberColumn("RR(풀백: 목표1/손절)",   format="%.2f"),
        "RR(추격)":     st.column_config.NumberColumn("RR(추격: 목표1/손절)",   format="%.2f"),
    },
)

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view_fmt.to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_entry_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ EBS & 진입 로직", expanded=False):
    st.markdown("""
**EBS(0~7)**: RSI 45~65 / MACD↑ / MA20±4% / VolZ>1.2 / MA20↑ / MACD>0 / 5d<10%  
**통과 기준**: EBS ≥ 4  

**풀백 진입**: MA20±0.5ATR 부근 진입, T1=+1.0ATR, T2=+1.8ATR, 손절=−1.2ATR  
**추격 진입(표시용)**: 현재가 기준 T1=+1.0×ATR(추정), T2=+1.8×ATR(추정), 손절=max(현재가−1.2×ATR, MA20×0.97)  
- ATR이 CSV에 없으면, `추천가/손절/목표가` 관계식으로 ATR을 **추정**하고,  
- MA20이 없으면 乖離%로 MA20을 **역산**하여 사용합니다.  

**필터**  
- “지금 진입 유효”: (선택한 진입 모드의) `목표가1 ≥ 종가`  
- “최소 RR”: (목표1−진입) / (진입−손절) ≥ 설정값
""")
