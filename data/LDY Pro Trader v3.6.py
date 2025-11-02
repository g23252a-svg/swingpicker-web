# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.5 (Auto Update, No Sliders, Top10 Only)
- 가중치/페널티 슬라이더 제거, TOP 10 즉시 노출
- EV_SCORE: 퍼센타일 정규화 + 고정 가중치(사용자 조정 불가)
- 추천 CSV: data/recommend_latest.csv (remote 우선)
- 이름맵:   data/krx_codes.csv (remote 우선) → FDR → pykrx 폴백
- OHLCV만 와도 화면에서 지표/EBS/추천가 생성
"""

import os, io, math, requests, numpy as np, pandas as pd, streamlit as st
from datetime import datetime

# ---------------- optional deps ----------------
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

# ---------------- page ----------------
st.set_page_config(page_title="LDY Pro Trader v3.5 (Top10)", layout="wide")
st.title("📈 LDY Pro Trader v3.5 (Top 10 Auto)")
st.caption("장마감 후 자동 업데이트 | 가중치/페널티 조정 없이 Top 10만 한눈에!")

# ---------------- constants ----------------
RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"
PASS_SCORE = 4  # EBS 통과 기준(고정)

# ---------------- IO helpers ----------------
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

# ---------------- utils ----------------
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

# --------- enrich from OHLCV (fallback) ----------
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

    # 2) FDR 폴백
    if FDR_OK:
        try:
            lst = fdr.StockListing("KRX")
            m = lst.rename(columns={"Code":"종목코드","Name":"종목명"})[["종목코드","종목명"]]
            m["종목코드"] = m["종목코드"].astype(str).map(z6)
            return m.drop_duplicates("종목코드")
        except Exception:
            pass

    # 3) pykrx 폴백
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

# ---------------- load raw ----------------
try:
    df_raw = load_csv_url(RAW_URL); log_src(df_raw, "remote", RAW_URL)
except Exception:
    if os.path.exists(LOCAL_RAW):
        df_raw = load_csv_path(LOCAL_RAW); log_src(df_raw, "local", LOCAL_RAW)
    else:
        st.error("❌ CSV가 없습니다. Actions에서 collector가 data/recommend_latest.csv를 올렸는지 확인하세요.")
        st.stop()

df_raw = normalize_cols(df_raw)

# 완제품인지 체크
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
for c in ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가"]:
    if c in latest.columns:
        latest[c] = pd.to_numeric(latest[c], errors="coerce")

# ---------------- scoring (고정식, 무슬라이더) ----------------
def _safe_pct_cap(s: pd.Series, q=90, floor=1.0):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return 1.0
    cap = np.nanpercentile(s, q)
    if not np.isfinite(cap) or cap <= 0:
        cap = floor
    return max(float(cap), floor)

def add_eval_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    for col in ["종가","추천매수가","손절가","추천매도가1","RSI14","MACD_slope","EBS"]:
        if col not in df.columns:
            df[col] = np.nan

    close = pd.to_numeric(df["종가"], errors="coerce")
    entry = pd.to_numeric(df["추천매수가"], errors="coerce")
    stop  = pd.to_numeric(df["손절가"], errors="coerce")
    t1    = pd.to_numeric(df["추천매도가1"], errors="coerce")

    rr_den = (entry - stop)
    rr1 = (t1 - entry) / rr_den.replace(0, np.nan)
    rr1 = rr1.mask((entry.isna()) | (stop.isna()) | (t1.isna()))
    df["RR1"] = rr1

    df["Now%"]   = ((close - entry).abs() / entry * 100).replace([np.inf,-np.inf], np.nan)
    df["T1여유%"] = ((t1 - close) / close * 100).replace([np.inf,-np.inf], np.nan)
    df["SL여유%"] = ((close - stop) / close * 100).replace([np.inf,-np.inf], np.nan)

    # ERS(0~3): EBS 통과 + MACD_slope>0 + RSI(45~65)
    ebs_ok  = (df.get("EBS", np.nan) >= PASS_SCORE).astype(int)
    macd_ok = (pd.to_numeric(df.get("MACD_slope"), errors="coerce") > 0).astype(int)
    rsi_ok  = ((pd.to_numeric(df.get("RSI14"), errors="coerce") >= 45) &
               (pd.to_numeric(df.get("RSI14"), errors="coerce") <= 65)).astype(int)
    df["ERS"] = (ebs_ok + macd_ok + rsi_ok).astype(float)

    # ---- 퍼센타일 기반 정규화(데이터 분포 자동적응) ----
    rr_cap   = _safe_pct_cap(df["RR1"],    q=90, floor=1.0)
    t1_cap   = _safe_pct_cap(df["T1여유%"], q=90, floor=5.0)
    sl_cap   = _safe_pct_cap(df["SL여유%"], q=90, floor=3.0)
    near_cap = _safe_pct_cap(df["Now%"],   q=75, floor=1.0)  # 근접도는 낮을수록 좋음 → 75p를 절단값

    rr_norm   = np.clip(df["RR1"] / rr_cap, 0, 1)
    t1_norm   = np.clip(df["T1여유%"] / t1_cap, 0, 1)
    sl_norm   = np.clip(df["SL여유%"] / sl_cap, 0, 1)
    near_norm = np.clip(1 - (df["Now%"] / near_cap), 0, 1)
    ers_norm  = np.clip(df["ERS"] / 3.0, 0, 1)

    # ---- 고정 가중치 (사용자 조정 불가) ----
    # 리워드/위험 비중을 가장 크게, 그다음 목표여유/손절여유/근접/ERS
    ev = 100 * (0.35*rr_norm + 0.25*t1_norm + 0.20*sl_norm + 0.15*near_norm + 0.05*ers_norm)
    df["EV_SCORE"] = np.round(ev, 1)
    return df

scored = add_eval_columns(latest)

# Top 10 추출 규칙(최소한의 퀄리티 게이트만 고정 적용)
tp = scored.copy()
tp = tp.dropna(subset=["추천매수가","손절가","추천매도가1","종가"])
tp = tp[tp["EBS"] >= PASS_SCORE]                 # EBS 통과
tp = tp.sort_values("EV_SCORE", ascending=False, na_position="last").head(10)

# 통과 마크 및 순위
tp["통과"] = np.where(tp["EBS"]>=PASS_SCORE, "🚀", "")
tp.insert(0, "순위", range(1, len(tp)+1))

# ---------------- 표 렌더링 ----------------
def cast_for_editor(df):
    df = df.copy()
    # 정수류
    for c in ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
    # 실수류
    for c in ["거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z",
              "ret_5d_%","ret_10d_%","EV_SCORE","ERS","RR1","Now%","T1여유%","SL여유%"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def column_config_for(df):
    cfg = {}
    def add(name, col):
        if name in df.columns: cfg[name]=col
    # text
    add("순위",       st.column_config.NumberColumn("순위", format="%d"))
    add("통과",       st.column_config.TextColumn(" "))
    add("시장",       st.column_config.TextColumn("시장"))
    add("종목명",     st.column_config.TextColumn("종목명"))
    add("종목코드",   st.column_config.TextColumn("종목코드"))
    add("근거",       st.column_config.TextColumn("근거"))
    # ints (comma)
    add("종가",        st.column_config.NumberColumn("종가",           format="%,d"))
    add("추천매수가",  st.column_config.NumberColumn("추천매수가",     format="%,d"))
    add("손절가",      st.column_config.NumberColumn("손절가",         format="%,d"))
    add("추천매도가1", st.column_config.NumberColumn("추천매도가1",    format="%,d"))
    add("추천매도가2", st.column_config.NumberColumn("추천매도가2",    format="%,d"))
    add("EBS",        st.column_config.NumberColumn("EBS",            format="%d"))
    # floats
    add("EV_SCORE",   st.column_config.NumberColumn("EV_SCORE",       format="%.1f"))
    add("ERS",        st.column_config.NumberColumn("ERS",            format="%.2f"))
    add("RR1",        st.column_config.NumberColumn("RR(목표1/손절)",  format="%.2f"))
    add("Now%",       st.column_config.NumberColumn("Now근접(%)",       format="%.2f"))
    add("T1여유%",    st.column_config.NumberColumn("목표1여유(%)",     format="%.2f"))
    add("SL여유%",    st.column_config.NumberColumn("손절여유(%)",      format="%.2f"))
    add("거래대금(억원)", st.column_config.NumberColumn("거래대금(억원)",  format="%,.0f"))
    add("시가총액(억원)", st.column_config.NumberColumn("시가총액(억원)",  format="%,.0f"))
    add("RSI14",      st.column_config.NumberColumn("RSI14",          format="%.1f"))
    add("乖離%",       st.column_config.NumberColumn("乖離%",           format="%.2f"))
    add("MACD_hist",  st.column_config.NumberColumn("MACD_hist",      format="%.4f"))
    add("MACD_slope", st.column_config.NumberColumn("MACD_slope",     format="%.5f"))
    add("Vol_Z",      st.column_config.NumberColumn("Vol_Z",          format="%.2f"))
    add("ret_5d_%",   st.column_config.NumberColumn("ret_5d_%",       format="%.2f"))
    add("ret_10d_%",  st.column_config.NumberColumn("ret_10d_%",      format="%.2f"))
    return cfg

st.subheader("🟢 Top 10 (자동 랭킹)", anchor=False)
top_cols = [
    "순위","통과","시장","종목명","종목코드",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "EV_SCORE","ERS","RR1","Now%","T1여유%","SL여유%",
    "거래대금(억원)","시가총액(억원)",
    "EBS","근거","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
for c in top_cols:
    if c not in tp.columns: tp[c]=np.nan

tp_fmt = cast_for_editor(tp[top_cols])
st.data_editor(
    tp_fmt,
    key="tbl_top10",
    width="stretch", height=560,
    hide_index=True, disabled=True, num_rows="fixed",
    column_config=column_config_for(tp_fmt),
)

# 다운로드 (Top10 / 전체랭크)
st.download_button(
    "📥 Top 10 다운로드 (CSV)",
    data=tp[top_cols].to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_top10.csv",
    mime="text/csv",
    key="dl_top10",
)

# 전체 랭크도 백그라운드 계산하여 파일로만 제공(화면은 Top10만)
rank_all = scored.copy()
rank_all = rank_all.dropna(subset=["추천매수가","손절가","추천매도가1","종가"])
rank_all = rank_all[rank_all["EBS"] >= PASS_SCORE]
rank_all = rank_all.sort_values("EV_SCORE", ascending=False, na_position="last")
rank_all.insert(0, "순위", range(1, len(rank_all)+1))

st.download_button(
    "📥 전체 랭킹 (CSV)",
    data=rank_all[top_cols].to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_full_rank.csv",
    mime="text/csv",
    key="dl_full",
)

with st.expander("ℹ️ 점수/지표 설명", expanded=False):
    st.markdown("""
**EBS(0~7)**: RSI 45~65 / MACD↑ / MA20±4% / VolZ>1.2 / MA20↑ / MACD>0 / 5d<10%  
**RR1**: (목표1−추천매수) / (추천매수−손절) — 보상/위험 비율  
**Now%**: 현재가 vs 추천매수 괴리(%) — 낮을수록 엔트리에 근접  
**T1여유%**: 목표1까지 여유(%)  
**SL여유%**: 손절까지 여유(%)  
**ERS(0~3)**: EBS 통과(≥4) + MACD_slope>0 + RSI 45~65 각 1점씩  
**EV_SCORE(0~100)**: 퍼센타일 정규화한 지표에 고정 가중치로 산출(사용자 조정 불가)  
- 가중치: RR 0.35, 목표여유 0.25, 손절여유 0.20, 근접도 0.15, ERS 0.05
""")
