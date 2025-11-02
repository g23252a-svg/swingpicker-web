# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.6 (Auto Update + EV + Top Picks + BUY_RANK)
- 추천 CSV: data/recommend_latest.csv (remote 우선)
- 이름맵:   data/krx_codes.csv (remote 우선) → FDR → pykrx
- EV_SCORE: near_band=0일 때 가중치 재배분(만점 100 유지), RR 동적 클립
- 🏆 종합순위(전체): BUY_SCORE(0~100), BUY_RANK(1=최상) 추가
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
st.set_page_config(page_title="LDY Pro Trader v3.6 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.6 (Auto Update)")
st.caption("매일 장마감 후 자동 업데이트 — EV스코어·TopPick·BUY_RANK(종합순위)")

# ---------------- constants ----------------
RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"
PASS_SCORE = 4  # EBS 합격선

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

# -------- name map --------
@st.cache_data(ttl=6*60*60)
def load_name_map() -> pd.DataFrame | None:
    # 1) repo
    try:
        m = load_csv_url(CODES_URL)
        if {"종목코드","종목명"}.issubset(m.columns):
            m["종목코드"] = m["종목코드"].astype(str).map(z6)
            return m[["종목코드","종목명"]].drop_duplicates("종목코드")
    except Exception:
        pass
    # 2) local
    if os.path.exists(LOCAL_MAP):
        try:
            m = load_csv_path(LOCAL_MAP)
            if {"종목코드","종목명"}.issubset(m.columns):
                m["종목코드"] = m["종목코드"].astype(str).map(z6)
                return m[["종목코드","종목명"]].drop_duplicates("종목코드")
        except Exception:
            pass
    # 3) FDR
    if FDR_OK:
        try:
            lst = fdr.StockListing("KRX")
            m = lst.rename(columns={"Code":"종목코드","Name":"종목명"})[["종목코드","종목명"]]
            m["종목코드"] = m["종목코드"].astype(str).map(z6)
            return m.drop_duplicates("종목코드")
        except Exception:
            pass
    # 4) pykrx
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

# 완제품 여부
has_ebs  = "EBS" in df_raw.columns and df_raw["EBS"].notna().any()
has_reco = all(c in df_raw.columns for c in ["추천매수가","추천매도가1","추천매도가2","손절가"]) and \
           df_raw[["추천매수가","추천매도가1","추천매도가2","손절가"]].notna().any().any()

df = df_raw.copy() if (has_ebs and has_reco) else enrich_from_ohlcv(df_raw)

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

# ---------------- helper: scoring ----------------
def compute_rr1_series(df: pd.DataFrame) -> pd.Series:
    entry = pd.to_numeric(df.get("추천매수가"), errors="coerce")
    stop  = pd.to_numeric(df.get("손절가"), errors="coerce")
    t1    = pd.to_numeric(df.get("추천매도가1"), errors="coerce")
    rr_den = (entry - stop)
    rr1 = (t1 - entry) / rr_den.replace(0, np.nan)
    rr1 = rr1.mask((entry.isna()) | (stop.isna()) | (t1.isna()))
    return rr1

def pct_norm(s: pd.Series, low=5, high=95, invert=False) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    lo = np.nanpercentile(x, low)
    hi = np.nanpercentile(x, high)
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        return pd.Series(np.nan, index=s.index)
    n = (x - lo) / (hi - lo)
    n = np.clip(n, 0, 1)
    return 1 - n if invert else n

def add_eval_columns(df_in: pd.DataFrame, near_band_pct: float, rr_clip: float = 3.0) -> pd.DataFrame:
    """RR1/여유%/ERS/EV_SCORE + near_band=0이면 가중치 재배분(만점 100 유지)"""
    df = df_in.copy()

    # 안전 기본값
    for col in ["종가","추천매수가","손절가","추천매도가1","RSI14","MACD_slope","EBS"]:
        if col not in df.columns:
            df[col] = np.nan

    close = pd.to_numeric(df["종가"], errors="coerce")
    entry = pd.to_numeric(df["추천매수가"], errors="coerce")
    stop  = pd.to_numeric(df["손절가"], errors="coerce")
    t1    = pd.to_numeric(df["추천매도가1"], errors="coerce")

    # RR1
    rr_den = (entry - stop)
    rr1 = (t1 - entry) / rr_den.replace(0, np.nan)
    rr1 = rr1.mask((entry.isna()) | (stop.isna()) | (t1.isna()))
    df["RR1"] = rr1

    # 근접/여유(%)
    df["Now%"]    = np.where((entry > 0) & np.isfinite(entry), (close.sub(entry).abs() / entry) * 100, np.nan)
    df["T1여유%"] = np.where((close > 0) & np.isfinite(close), (t1.sub(close) / close) * 100, np.nan)
    df["SL여유%"] = np.where((close > 0) & np.isfinite(close), ((close - stop) / close) * 100, np.nan)

    # ERS (0~3)
    ebs_ok  = (pd.to_numeric(df.get("EBS"), errors="coerce") >= PASS_SCORE).astype(int)
    macd_ok = (pd.to_numeric(df.get("MACD_slope"), errors="coerce") > 0).astype(int)
    rsi_val = pd.to_numeric(df.get("RSI14"), errors="coerce")
    rsi_ok  = ((rsi_val >= 45) & (rsi_val <= 65)).astype(int)
    df["ERS"] = (ebs_ok + macd_ok + rsi_ok).astype(float)

    # ----- EV 가중치 (near_band=0이면 재배분) -----
    w = dict(rr=0.35, sl=0.20, t1=0.20, near=0.15, ers=0.10)
    if not near_band_pct or near_band_pct <= 0:
        alive = w["rr"] + w["sl"] + w["t1"] + w["ers"]  # 0.85
        scale = 1.0 / alive
        w["rr"]*=scale; w["sl"]*=scale; w["t1"]*=scale; w["ers"]*=scale; w["near"]=0.0

    rr_cap   = rr_clip if (isinstance(rr_clip,(int,float)) and rr_clip>0) else 3.0
    rr_norm  = np.clip(df["RR1"], 0, rr_cap) / rr_cap
    sl_norm  = np.clip(df["SL여유%"]/5,  0, 1)
    t1_norm  = np.clip(df["T1여유%"]/10, 0, 1)
    near_norm= np.clip(1 - (df["Now%"] / near_band_pct), 0, 1) if (near_band_pct and near_band_pct>0) else 0.0
    ers_norm = np.clip(df["ERS"]/3, 0, 1)

    ev_raw = (w["rr"]*rr_norm + w["sl"]*sl_norm + w["t1"]*t1_norm + w["near"]*near_norm + w["ers"]*ers_norm)
    df["EV_SCORE"] = np.round(np.where(np.isfinite(ev_raw), ev_raw*100, 0), 1)
    return df

def compute_buy_score(df_in: pd.DataFrame, *, near_band_pct: float, rr_clip: float, weights: dict) -> pd.DataFrame:
    """
    BUY_SCORE(0~100) = 긍정(가중합) - 페널티, 분위수 정규화 활용
    weights keys: ev, rr, near, t1, sl, ers, liq, pen_macd, pen_rsi, pen_ext
    """
    d = add_eval_columns(df_in, near_band_pct=near_band_pct, rr_clip=rr_clip).copy()

    rr_n   = np.clip(d["RR1"], 0, rr_clip) / rr_clip
    near_n = pct_norm(d["Now%"],    low=5,  high=95, invert=True)
    t1_n   = pct_norm(d["T1여유%"], low=5,  high=95, invert=False)
    sl_n   = pct_norm(d["SL여유%"], low=5,  high=95, invert=False)
    ev_n   = pd.to_numeric(d["EV_SCORE"], errors="coerce")/100
    ers_n  = pd.to_numeric(d["ERS"],      errors="coerce")/3
    liq_n  = pct_norm(d["거래대금(억원)"], low=10, high=90, invert=False)

    macd_pen = (pd.to_numeric(d.get("MACD_slope"), errors="coerce") <= 0).astype(float)
    rsi      = pd.to_numeric(d.get("RSI14"), errors="coerce")
    rsi_pen  = (~((rsi >= 45) & (rsi <= 65))).astype(float)
    # 과열/이격 과대(乖離%) 페널티: 상단 꼬리 구간을 0~1로 매핑하여 차감
    ext_pen  = pct_norm(pd.to_numeric(d.get("乖離%"), errors="coerce").abs(), low=70, high=99, invert=False)

    pos = (
        weights["ev"]*ev_n.fillna(0)   +
        weights["rr"]*rr_n.fillna(0)   +
        weights["near"]*near_n.fillna(0) +
        weights["t1"]*t1_n.fillna(0)   +
        weights["sl"]*sl_n.fillna(0)   +
        weights["ers"]*ers_n.fillna(0) +
        weights["liq"]*liq_n.fillna(0)
    )
    neg = (
        weights["pen_macd"]*macd_pen.fillna(0) +
        weights["pen_rsi"] *rsi_pen.fillna(0)  +
        weights["pen_ext"] *ext_pen.fillna(0)
    )

    score = 100*np.clip(pos - neg, 0, None)
    d["BUY_SCORE"] = np.round(score, 1)
    d["BUY_RANK"]  = d["BUY_SCORE"].rank(method="min", ascending=False).astype("Int64")
    return d

def cast_for_editor(df):
    df = df.copy()
    int_like = ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS","BUY_RANK"]
    for c in int_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
    float_like = [
        "거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope",
        "Vol_Z","ret_5d_%","ret_10d_%","EV_SCORE","ERS","RR1","Now%","T1여유%","SL여유%","BUY_SCORE"
    ]
    for c in float_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def column_config_for(df):
    cfg = {}
    def add(name, col):
        if name in df.columns: cfg[name]=col
    add("통과",       st.column_config.TextColumn(" "))
    add("시장",       st.column_config.TextColumn("시장"))
    add("종목명",     st.column_config.TextColumn("종목명"))
    add("종목코드",   st.column_config.TextColumn("종목코드"))
    add("근거",       st.column_config.TextColumn("근거"))
    add("종가",        st.column_config.NumberColumn("종가",           format="%,d"))
    add("추천매수가",  st.column_config.NumberColumn("추천매수가",     format="%,d"))
    add("손절가",      st.column_config.NumberColumn("손절가",         format="%,d"))
    add("추천매도가1", st.column_config.NumberColumn("추천매도가1",    format="%,d"))
    add("추천매도가2", st.column_config.NumberColumn("추천매도가2",    format="%,d"))
    add("EBS",        st.column_config.NumberColumn("EBS",            format="%d"))
    add("거래대금(억원)", st.column_config.NumberColumn("거래대금(억원)",  format="%,.0f"))
    add("시가총액(억원)", st.column_config.NumberColumn("시가총액(억원)",  format="%,.0f"))
    add("RSI14",      st.column_config.NumberColumn("RSI14",          format="%.1f"))
    add("乖離%",       st.column_config.NumberColumn("乖離%",           format="%.2f"))
    add("MACD_hist",  st.column_config.NumberColumn("MACD_hist",      format="%.4f"))
    add("MACD_slope", st.column_config.NumberColumn("MACD_slope",     format="%.5f"))
    add("Vol_Z",      st.column_config.NumberColumn("Vol_Z",          format="%.2f"))
    add("ret_5d_%",   st.column_config.NumberColumn("ret_5d_%",       format="%.2f"))
    add("ret_10d_%",  st.column_config.NumberColumn("ret_10d_%",      format="%.2f"))
    add("EV_SCORE",   st.column_config.NumberColumn("EV_SCORE",       format="%.1f"))
    add("ERS",        st.column_config.NumberColumn("ERS",            format="%.2f"))
    add("RR1",        st.column_config.NumberColumn("RR1(목표1/손절)", format="%.2f"))
    add("Now%",       st.column_config.NumberColumn("Now 근접(%)",      format="%.2f"))
    add("T1여유%",    st.column_config.NumberColumn("목표1여유(%)",     format="%.2f"))
    add("SL여유%",    st.column_config.NumberColumn("손절여유(%)",      format="%.2f"))
    add("BUY_SCORE",  st.column_config.NumberColumn("BUY_SCORE",      format="%.1f"))
    add("BUY_RANK",   st.column_config.NumberColumn("BUY_RANK",       format="%d"))
    return cfg

def render_table(df, *, key: str, height=620):
    st.data_editor(
        df, key=key, width="stretch", height=height, hide_index=True,
        disabled=True, num_rows="fixed", column_config=column_config_for(df),
    )

def safe_sort(dfv, key):
    try:
        if key=="EV_SCORE▼" and "EV_SCORE" in dfv.columns:
            return dfv.sort_values("EV_SCORE", ascending=False, na_position="last")
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
    for alt in ["BUY_SCORE","EV_SCORE","EBS","거래대금(억원)","시가총액(억원)","종가"]:
        if alt in dfv.columns:
            return dfv.sort_values(alt, ascending=False, na_position="last")
    return dfv

# ---------------- Filters (공통) ----------------
with st.container():
    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 0, step=50, key="flt_turn")
    with c2:
        sort_key = st.selectbox("정렬", ["BUY_SCORE▼","EV_SCORE▼","EBS▼","거래대금▼","시가총액▼","RSI▲","RSI▼","종가▲","종가▼"], index=0, key="flt_sort")
    with c3:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10, key="flt_topn")
    with c4:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930", key="flt_query")

view_base = latest.copy()
if "거래대금(억원)" in view_base.columns:
    view_base = view_base[view_base["거래대금(억원)"] >= float(min_turn)]
if q_text:
    q = q_text.strip().lower()
    view_base = view_base[
        view_base["종목명"].fillna("").astype(str).str.lower().str.contains(q) |
        view_base["종목코드"].fillna("").astype(str).str.contains(q)
    ]

# RR 분포 기반 동적 상한
_rr_all = compute_rr1_series(view_base)
_rr_p95 = float(np.nanpercentile(_rr_all.dropna(), 95)) if _rr_all.dropna().size>0 else 1.5
_rr_slider_max = float(max(1.0, min(3.0, round(_rr_p95 + 0.5, 2))))  # 상한 3.0

# ---------------- Tabs ----------------
tab0, tab1, tab2 = st.tabs(["🏆 종합순위(전체)", "🟢 Top Picks", "📋 전체 보기"])

with tab0:
    st.subheader("🏆 종합순위(전체) — BUY_RANK", anchor=False)

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        near_band_for_buy = st.slider("Now 근접 밴드(±%)", 0.00, 3.00, 0.00, step=0.10, key="mr_near")
        rr_dyn_for_buy    = st.checkbox("RR 동적 클립(p95)", value=True, key="mr_rrdyn")
    with c2:
        rr_min_for_view   = st.slider("최소 RR(목표1/손절)", 0.50, _rr_slider_max, 0.80, step=0.10, key="mr_rrmin")
        ers_min_for_view  = st.slider("ERS ≥", 0.00, 3.00, 0.50, step=0.50, key="mr_ersmin")
    with c3:
        st.markdown("**⚖️ 스코어 가중치**")
        w_ev   = st.slider("EV",        0.0, 1.0, 0.30, 0.05, key="w_ev")
        w_rr   = st.slider("RR",        0.0, 1.0, 0.20, 0.05, key="w_rr")
        w_near = st.slider("근접",        0.0, 1.0, 0.15, 0.05, key="w_near")
        w_t1   = st.slider("목표1여유",     0.0, 1.0, 0.10, 0.05, key="w_t1")
        w_sl   = st.slider("손절여유",      0.0, 1.0, 0.10, 0.05, key="w_sl")
        w_ers  = st.slider("ERS",       0.0, 1.0, 0.10, 0.05, key="w_ers")
        w_liq  = st.slider("유동성(거래대금)", 0.0, 1.0, 0.05, 0.05, key="w_liq")
        st.markdown("**🚫 페널티 가중치**")
        p_macd = st.slider("MACD≤0",    0.0, 1.0, 0.15, 0.05, key="p_macd")
        p_rsi  = st.slider("RSI 이탈",    0.0, 1.0, 0.10, 0.05, key="p_rsi")
        p_ext  = st.slider("乖離 과열",    0.0, 1.0, 0.10, 0.05, key="p_ext")

    rr_clip_val = _rr_p95 if rr_dyn_for_buy and np.isfinite(_rr_p95) and _rr_p95>0 else 3.0
    weights = dict(ev=w_ev, rr=w_rr, near=w_near, t1=w_t1, sl=w_sl, ers=w_ers, liq=w_liq,
                   pen_macd=p_macd, pen_rsi=p_rsi, pen_ext=p_ext)

    scored_buy = compute_buy_score(
        view_base, near_band_pct=near_band_for_buy, rr_clip=rr_clip_val, weights=weights
    )

    # 기본 뷰 필터 (너무 이른/늦은 시그널 제거용 권장 필터)
    vb = scored_buy.copy()
    vb = vb.dropna(subset=["추천매수가","손절가","추천매도가1","종가"])
    if rr_min_for_view > 0:
        vb = vb[vb["RR1"] >= rr_min_for_view]
    if ers_min_for_view > 0:
        vb = vb[vb["ERS"] >= ers_min_for_view]
    if near_band_for_buy > 0:
        vb = vb[vb["Now%"] <= near_band_for_buy]

    # 랭크 및 정렬
    vb = vb.sort_values(["BUY_SCORE","EV_SCORE","거래대금(억원)"], ascending=[False,False,False])
    vb["BUY_RANK"] = vb["BUY_SCORE"].rank(method="min", ascending=False).astype("Int64")

    # 표 렌더링
    cols = [
        "BUY_RANK","BUY_SCORE","시장","종목명","종목코드",
        "종가","추천매수가","손절가","추천매도가1","추천매도가2",
        "RR1","Now%","T1여유%","SL여유%","EV_SCORE","ERS",
        "거래대금(억원)","시가총액(억원)","EBS","근거","RSI14","乖離%","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
    ]
    for c in cols:
        if c not in vb.columns: vb[c]=np.nan

    st.write(f"📋 총 {len(view_base):,}개 / 표시 {min(len(vb), int(topn)):,}개")
    render_table(cast_for_editor(vb[cols].head(int(topn))), key="tbl_master_rank_v36")

    st.download_button(
        "📥 종합순위 다운로드 (CSV)",
        data=vb[cols].to_csv(index=False, encoding="utf-8-sig"),
        file_name="ldy_buy_ranking.csv",
        mime="text/csv",
        key="dl_master_rank_v36",
    )

    # Top 3 요약
    if len(vb):
        st.markdown("---")
        top3 = vb.head(3)[["종목명","종목코드","BUY_RANK","BUY_SCORE","EV_SCORE","RR1","Now%","T1여유%","SL여유%","ERS","거래대금(억원)"]]
        st.write("🥇 Top 3 한눈에")
        render_table(cast_for_editor(top3), key="tbl_master_top3_v36", height=200)

with tab1:
    st.subheader("🛠 Top Picks 조건", anchor=False)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        rr_min = st.slider("최소 RR(목표1/손절)", 0.50, _rr_slider_max, 1.00, step=0.10, key="tp_rr")
        near_band = st.slider("Now 근접 밴드(±%)", 0.00, 3.00, 0.00, step=0.10, key="tp_near")
    with c2:
        sl_min = st.slider("손절여유 ≥ (%)", 0.00, 10.00, 0.00, step=0.50, key="tp_sl")
        t1_min = st.slider("목표1여유 ≥ (%)", 0.00, 15.00, 0.00, step=0.50, key="tp_t1")
    with c3:
        ers_min = st.slider("ERS ≥", 0.00, 3.00, 1.00, step=0.50, key="tp_ers")
        use_rr_dyn = st.checkbox("RR 동적 클립(p95)", value=True, key="tp_rr_dyn")

    rr_clip_val = _rr_p95 if use_rr_dyn and np.isfinite(_rr_p95) and _rr_p95>0 else 3.0
    scored = add_eval_columns(view_base, near_band_pct=near_band, rr_clip=rr_clip_val)

    tp = scored.copy()
    tp = tp[tp["EBS"] >= PASS_SCORE]
    tp = tp.dropna(subset=["추천매수가","손절가","추천매도가1","종가"])
    if rr_min > 0:  tp = tp[tp["RR1"] >= rr_min]
    if ers_min > 0: tp = tp[tp["ERS"] >= ers_min]
    if sl_min > 0:  tp = tp[tp["SL여유%"] >= sl_min]
    if t1_min > 0:  tp = tp[tp["T1여유%"] >= t1_min]
    if near_band > 0: tp = tp[tp["Now%"] <= near_band]

    tp = safe_sort(tp, sort_key).head(int(topn))
    if "EBS" in tp.columns:
        tp["통과"] = np.where(tp["EBS"]>=PASS_SCORE, "🚀", "")

    cols = [
        "통과","시장","종목명","종목코드",
        "종가","추천매수가","손절가","추천매도가1","추천매도가2",
        "EV_SCORE","ERS","RR1","Now%","T1여유%","SL여유%",
        "거래대금(억원)","시가총액(억원)","EBS","근거",
        "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
    ]
    for c in cols:
        if c not in tp.columns: tp[c]=np.nan

    st.write(f"📋 총 {len(view_base):,}개 / 표시 {min(len(tp), int(topn)):,}개")
    render_table(cast_for_editor(tp[cols]), key="tbl_top_picks_v36")
    st.download_button(
        "📥 Top Picks 다운로드 (CSV)",
        data=tp[cols].to_csv(index=False, encoding="utf-8-sig"),
        file_name="ldy_top_picks.csv",
        mime="text/csv",
        key="dl_top_picks_v36",
    )

with tab2:
    scored_all = add_eval_columns(view_base, near_band_pct=0.0, rr_clip=(_rr_p95 if _rr_p95>0 else 3.0))
    view = safe_sort(scored_all, sort_key).head(int(topn))
    if "EBS" in view.columns:
        view["통과"] = np.where(view["EBS"]>=PASS_SCORE, "🚀", "")

    cols = [
        "통과","시장","종목명","종목코드",
        "종가","추천매수가","손절가","추천매도가1","추천매도가2",
        "EV_SCORE","ERS","RR1","Now%","T1여유%","SL여유%",
        "거래대금(억원)","시가총액(억원)",
        "EBS","근거",
        "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
    ]
    for c in cols:
        if c not in view.columns: view[c]=np.nan

    st.write(f"📋 총 {len(view_base):,}개 / 표시 {min(len(view), int(topn)):,}개")
    render_table(cast_for_editor(view[cols]), key="tbl_full_view_v36")
    st.download_button(
        "📥 전체 보기 다운로드 (CSV)",
        data=view[cols].to_csv(index=False, encoding="utf-8-sig"),
        file_name="ldy_entry_candidates.csv",
        mime="text/csv",
        key="dl_full_view_v36",
    )

# ---------------- help ----------------
with st.expander("ℹ️ 점수/지표 설명", expanded=False):
    st.markdown("""
**BUY_SCORE(0~100)**: EV/RR/근접/목표1여유/손절여유/ERS/유동성(+) − MACD≤0/RSI이탈/과열乖離 페널티(−)  
**BUY_RANK**: BUY_SCORE 내림차순 랭킹 (1 = 최상)  
권장 필터: 거래대금 하한, RR≥0.8, ERS≥0.5, Now%≤2%  
""")
