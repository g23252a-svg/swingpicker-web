# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.3.3 (Auto Update + Robust Name Map + Number Format + EBS+)
- 추천 CSV: data/recommend_latest.csv (remote 우선)
- 이름맵:   data/krx_codes.csv (remote 우선) → FDR → pykrx 순 폴백
- OHLCV만 와도 화면에서 지표/EBS/추천가 생성
- 거래대금(억원) 안전 보강, 안전 정렬
- 표에 표시되는 숫자(가격/억원) 천단위 콤마 + 소수 포맷 정교화
- collector가 생성한 EBS_PLUS, HitProb_%(heuristic) 컬럼 자동 인식/표시/정렬
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

st.set_page_config(page_title="LDY Pro Trader v3.3.3 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.3.3 (Auto Update)")
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
        r5 = last["ret_5d_%"].iloc[0];    c7 = nz(r5) and r5 < 10;        e+=int(c7); why.append("5d<10%")
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

# 이름 매핑
with st.status("🏷️ 종목명 매핑 중...", expanded=False):
    latest = apply_names(latest)

# 숫자 캐스팅 & 거래대금 보강
latest = ensure_turnover(latest)
num_cols = ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z",
            "ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가",
            "EBS_PLUS","HitProb_%(heuristic)"]
for c in num_cols:
    if c in latest.columns:
        latest[c] = pd.to_numeric(latest[c], errors="coerce")

# ------------- UI -------------
with st.expander("🔍 보기/필터", expanded=True):
    c1,c2,c3,c4,c5 = st.columns([1,1,1.2,1,2])
    with c1:
        only_entry = st.checkbox("🚀 초입 후보만 (EBS≥4)", value=("EBS" in latest.columns))
    with c2:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c3:
        sort_opts = ["EBS+▼","확률▼","EBS▼","거래대금▼","시가총액▼","RSI▲","RSI▼","종가▲","종가▼"]
        sort_key = st.selectbox("정렬", sort_opts, index=0)
    with c4:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)
    with c5:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

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
        if key=="EBS+▼" and "EBS_PLUS" in dfv.columns:
            by = ["EBS_PLUS","EBS","거래대금(억원)"]
            by = [b for b in by if b in dfv.columns]
            return dfv.sort_values(by=by, ascending=[False]*len(by))
        if key=="확률▼" and "HitProb_%(heuristic)" in dfv.columns:
            by = ["HitProb_%(heuristic)","EBS_PLUS","거래대금(억원)"]
            by = [b for b in by if b in dfv.columns]
            return dfv.sort_values(by=by, ascending=[False]*len(by))
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
    for alt in ["EBS_PLUS","EBS","거래대금(억원)","시가총액(억원)","종가"]:
        if alt in dfv.columns:
            return dfv.sort_values(alt, ascending=False, na_position="last")
    return dfv

view = safe_sort(view, sort_key)

if "EBS" in view.columns:
    view["통과"] = np.where(view["EBS"]>=PASS_SCORE, "🚀", "")

# 표시 컬럼 구성 (있으면 보여주기)
base_cols = [
    "통과","시장","종목명","종목코드",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "거래대금(억원)","시가총액(억원)",
    "EBS","근거",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
extra_cols = [c for c in ["EBS_PLUS","HitProb_%(heuristic)"] if c in view.columns]
cols = [c for c in base_cols if c in view.columns] + extra_cols
for c in cols:
    if c not in view.columns: view[c]=np.nan

st.write(f"📋 총 {len(latest):,}개 / 표시 {min(len(view), int(topn)):,}개")

# ── 숫자 포맷(콤마) 적용을 위한 캐스팅 ──
view_fmt = view[cols].head(int(topn)).copy()

# 가격/정수류 → Int64 (NaN 허용 정수)
for c in ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce").round(0).astype("Int64")

# 억원/지표류 → float (collector와 소수 자리 맞춤)
if "거래대금(억원)" in view_fmt.columns:
    view_fmt["거래대금(억원)"] = pd.to_numeric(view_fmt["거래대금(억원)"], errors="coerce")  # collector: round(?, 2)
if "시가총액(억원)" in view_fmt.columns:
    view_fmt["시가총액(억원)"] = pd.to_numeric(view_fmt["시가총액(억원)"], errors="coerce")  # collector: round(?, 1)
for c in ["RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS_PLUS","HitProb_%(heuristic)"]:
    if c in view_fmt.columns:
        view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce")

# 컬럼 설정
colcfg = {
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
}
# 억원/지표 (콤마·소수) — collector의 반올림 자리수와 일치
if "거래대금(억원)" in view_fmt.columns:
    colcfg["거래대금(억원)"] = st.column_config.NumberColumn("거래대금(억원)", format="%,.2f")
if "시가총액(억원)" in view_fmt.columns:
    colcfg["시가총액(억원)"] = st.column_config.NumberColumn("시가총액(억원)", format="%,.1f")

for c, f in [
    ("RSI14","%.1f"),
    ("乖離%","%.2f"),
    ("MACD_hist","%.4f"),
    ("MACD_slope","%.5f"),
    ("Vol_Z","%.2f"),
    ("ret_5d_%","%.2f"),
    ("ret_10d_%","%.2f"),
    ("EBS_PLUS","%.1f"),
]:
    if c in view_fmt.columns:
        colcfg[c] = st.column_config.NumberColumn(c, format=f)

# 확률은 진행바로 가독성↑
if "HitProb_%(heuristic)" in view_fmt.columns:
    colcfg["HitProb_%(heuristic)"] = st.column_config.ProgressColumn(
        "HitProb(%)",
        min_value=0, max_value=100, format="%.0f"
    )

st.data_editor(
    view_fmt,
    width="stretch",
    height=640,
    hide_index=True,
    disabled=True,
    num_rows="fixed",
    column_config=colcfg,
)

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view[cols].head(int(topn)).to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_entry_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ EBS 구성(급등 초입 로직)", expanded=False):
    st.markdown("""
- 컷: 거래대금 ≥ **50억원**, 시총 ≥ **1,000억원** (collector 권장)
- 점수(0~7): RSI 45~65 / MACD↑ / MA20±4% / VolZ>1.2 / MA20↑ / MACD>0 / 5d<10%
- 통과: **EBS ≥ 4**
- 추천가: ATR/MA20 기반 (엔트리=MA20±0.5ATR, T1=+1.0ATR, T2=+1.8ATR, 손절=−1.2ATR)
- 참고: EBS+는 RS/베이스 품질/볼륨/레짐 등을 합성한 가중 점수(0~100), HitProb는 UI용 휴리스틱(%)
""")
