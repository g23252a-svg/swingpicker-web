# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.5 — MOMO Top10 (No Sliders)
- 매일 장마감 후 업데이트된 CSV(recommend_latest.csv)를 기반으로
  EV_SCORE + MOMO_SCORE를 결합한 GLOBAL_SCORE로 단일 Top10 출력
- 급등 직전/직후 '폭발(momentum burst)' 신호를 강하게 반영
- 슬라이더/가중치 UI 제거, 고정 컷(거래대금, EBS)만 적용
"""

import os, io, math, requests, numpy as np, pandas as pd, streamlit as st
from datetime import datetime

# ---------- Optional deps (fallback용) ----------
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

# ---------- Page ----------
st.set_page_config(page_title="LDY Pro Trader v3.5 — MOMO Top10", layout="wide")
st.title("📈 LDY Pro Trader v3.5 — MOMO Top10")
st.caption("급등 추세 포착용 단일 Top10 | EV_SCORE × MOMO_SCORE")

# ---------- Constants ----------
RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"

PASS_SCORE_EBS = 4          # Top Picks 기본 컷
MIN_TURNOVER   = 100        # (억원) 유동성 컷(고정)
NEAR_BAND_DEF  = 1.5        # Now 근접도 밴드(%), EV_SCORE 내부에서 사용

# ---------- IO helpers ----------
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

# ---------- Utils ----------
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

# ---------- Enrich from OHLCV (fallback) ----------
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
        c2 = nz(last["MACD_slope"].iloc[0]) and last["MACD_slope"].iloc[0] > 0; e+=int(c2); why.append("MACD상승" if c2 else "")
        close, ma20 = last["종가"].iloc[0], last["MA20"].iloc[0]
        c3 = nz(ma20) and (0.99*ma20 <= close <= 1.04*ma20); e+=int(c3); why.append("MA20 근처" if c3 else "")
        c4 = nz(last["Vol_Z"].iloc[0]) and last["Vol_Z"].iloc[0] > 1.2; e+=int(c4); why.append("거래량증가" if c4 else "")
        m20p = x["MA20"].iloc[-2] if len(x)>=2 else np.nan
        c5 = nz(m20p) and (last["MA20"].iloc[0] - m20p > 0); e+=int(c5); why.append("상승구조" if c5 else "")
        c6 = nz(last["MACD_hist"].iloc[0]) and last["MACD_hist"].iloc[0] > 0; e+=int(c6); why.append("MACD>sig" if c6 else "")
        r5 = last["ret_5d_%"].iloc[0];    c7 = nz(r5) and r5 < 10;        e+=int(c7); why.append("과열아님" if c7 else "")
        last["EBS"] = e; last["근거"] = ", ".join([w for w in why if w])

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

# ---------- Name map ----------
@st.cache_data(ttl=6*60*60)
def load_name_map() -> pd.DataFrame | None:
    # 1) repo map
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

# ---------- EV_SCORE ----------
def add_eval_columns(df_in: pd.DataFrame, near_band_pct: float = NEAR_BAND_DEF) -> pd.DataFrame:
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

    df["Now%"]   = (close.sub(entry).abs() / entry * 100).replace([np.inf, -np.inf], np.nan)
    df["T1여유%"] = (t1.sub(close) / close * 100).replace([np.inf, -np.inf], np.nan)
    df["SL여유%"] = (close.sub(stop) / close * 100).replace([np.inf, -np.inf], np.nan)

    ebs_ok  = (pd.to_numeric(df.get("EBS"), errors="coerce") >= PASS_SCORE_EBS).astype(int)
    macd_ok = (pd.to_numeric(df.get("MACD_slope"), errors="coerce") > 0).astype(int)
    rsi_ok  = ((pd.to_numeric(df.get("RSI14"), errors="coerce") >= 45) & (pd.to_numeric(df.get("RSI14"), errors="coerce") <= 65)).astype(int)
    df["ERS"] = (ebs_ok + macd_ok + rsi_ok).astype(float)

    rr_norm   = np.clip(df["RR1"], 0, 3) / 3
    sl_norm   = np.clip(df["SL여유%"]/5, 0, 1)
    t1_norm   = np.clip(df["T1여유%"]/10, 0, 1)
    near_norm = 0.0
    if near_band_pct and near_band_pct > 0:
        near_norm = np.clip(1 - (df["Now%"] / near_band_pct), 0, 1)
    ers_norm  = np.clip(df["ERS"]/3, 0, 1)

    ev = 100*(0.35*rr_norm + 0.20*sl_norm + 0.20*t1_norm + 0.15*near_norm + 0.10*ers_norm)
    df["EV_SCORE"] = np.round(ev.fillna(0), 1)

    return df

# ---------- MOMO_SCORE ----------
def _scale_01(s, lo, hi):
    v = pd.to_numeric(s, errors="coerce")
    return np.clip((v - lo) / max(1e-9, (hi - lo)), 0, 1)

def _log_liq(x):
    v = pd.to_numeric(x, errors="coerce")
    return _scale_01(np.log1p(v*1e8), np.log1p(1e10), np.log1p(1.5e12))  # 100억~1500억

def add_momo_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    for c in ["종가","시가","고가","저가","RSI14","MACD_slope","乖離%","ret_5d_%","거래대금(억원)","Vol_Z"]:
        if c not in df.columns: df[c] = np.nan

    close = pd.to_numeric(df["종가"], errors="coerce")
    volz  = pd.to_numeric(df["Vol_Z"], errors="coerce")
    rsi   = pd.to_numeric(df["RSI14"], errors="coerce")
    kairi = pd.to_numeric(df["乖離%"], errors="coerce")
    r5    = pd.to_numeric(df["ret_5d_%"], errors="coerce")
    mslope= pd.to_numeric(df["MACD_slope"], errors="coerce")
    turn  = pd.to_numeric(df["거래대금(억원)"], errors="coerce")

    # (A) Breakout proxy
    bo_rsi   = _scale_01(rsi, 55, 70)
    bo_kairi = (kairi.between(2, 8)).astype(float) * _scale_01(kairi, 2, 8)
    bo_r5    = (r5.between(3, 12)).astype(float) * _scale_01(r5, 3, 12)
    breakout = (0.4*bo_rsi + 0.3*bo_kairi + 0.3*bo_r5)

    # (B) 거래대금/볼륨 확장
    volx = _scale_01(volz, 1.5, 4.0)
    liq  = _log_liq(turn)
    expansion = (0.6*volx + 0.4*liq)

    # (C) 트렌드 품질
    macd_ok = (mslope > 0).astype(float)
    rsi_mid = _scale_01(rsi, 50, 65)
    rsi_hot_penalty = (rsi > 75).astype(float)*0.4
    trend = np.clip(0.5*macd_ok + 0.5*rsi_mid - rsi_hot_penalty, 0, 1)

    # (D) squeeze→release (없으면 0)
    squeeze_release = 0.0
    if "BB_Width" in df.columns and "%B" in df.columns:
        bbw = _scale_01(df["BB_Width"], df["BB_Width"].quantile(0.05), df["BB_Width"].quantile(0.6))
        pb  = _scale_01(df["%B"], 0.8, 1.0)
        squeeze_release = (1 - bbw) * pb

    # (E) 페널티
    overhead_pen = ((kairi < -8).astype(float)*0.3 + (kairi > 12).astype(float)*0.3)
    low_liq_pen  = (turn < MIN_TURNOVER).astype(float)*0.4
    penalty = np.clip(overhead_pen + low_liq_pen, 0, 1)

    momo = 100*(0.35*breakout + 0.30*expansion + 0.25*trend + 0.10*squeeze_release)
    momo = momo * (1 - 0.6*penalty)
    df["MOMO_SCORE"] = np.round(momo.fillna(0), 1)
    return df

# ---------- GLOBAL_SCORE ----------
def add_global_score(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "EV_SCORE" not in df.columns:
        df["EV_SCORE"] = 0.0
    df = add_momo_columns(df)
    glob = 0.6*pd.to_numeric(df["MOMO_SCORE"], errors="coerce") + \
           0.4*pd.to_numeric(df["EV_SCORE"], errors="coerce")
    df["GLOBAL_SCORE"] = np.round(glob.fillna(0), 1)
    return df

# ---------- Table formatting ----------
def cast_for_editor(df):
    df = df.copy()
    int_like = ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]
    for c in int_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
    float_like = ["거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope",
                  "Vol_Z","ret_5d_%","ret_10d_%","EV_SCORE","MOMO_SCORE","GLOBAL_SCORE",
                  "ERS","RR1","Now%","T1여유%","SL여유%"]
    for c in float_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def column_config_for(df):
    cfg = {}
    def add(name, col):
        if name in df.columns: cfg[name]=col
    add("시장",       st.column_config.TextColumn("시장"))
    add("종목명",     st.column_config.TextColumn("종목명"))
    add("종목코드",   st.column_config.TextColumn("종목코드"))
    add("근거",       st.column_config.TextColumn("근거"))

    add("종가",        st.column_config.NumberColumn("종가",           format="%,d"))
    add("추천매수가",  st.column_config.NumberColumn("추천매수가",     format="%,d"))
    add("손절가",      st.column_config.NumberColumn("손절가",         format="%,d"))
    add("추천매도가1", st.column_config.NumberColumn("추천매도가1",    format="%,d"))
    add("추천매도가2", st.column_config.NumberColumn("추천매도가2",    format="%,d"))

    add("거래대금(억원)", st.column_config.NumberColumn("거래대금(억원)",  format="%,.0f"))
    add("시가총액(억원)", st.column_config.NumberColumn("시가총액(억원)",  format="%,.0f"))

    add("GLOBAL_SCORE", st.column_config.NumberColumn("GLOBAL",        format="%.1f"))
    add("MOMO_SCORE",   st.column_config.NumberColumn("MOMO",          format="%.1f"))
    add("EV_SCORE",     st.column_config.NumberColumn("EV",            format="%.1f"))

    add("RR1",        st.column_config.NumberColumn("RR(목표1/손절)",    format="%.2f"))
    add("T1여유%",    st.column_config.NumberColumn("목표1여유(%)",      format="%.2f"))
    add("SL여유%",    st.column_config.NumberColumn("손절여유(%)",      format="%.2f"))
    add("Now%",       st.column_config.NumberColumn("Now 근접(%)",       format="%.2f"))

    add("RSI14",      st.column_config.NumberColumn("RSI14",          format="%.1f"))
    add("MACD_slope", st.column_config.NumberColumn("MACD_slope",     format="%.5f"))
    add("Vol_Z",      st.column_config.NumberColumn("Vol_Z",          format="%.2f"))
    add("乖離%",       st.column_config.NumberColumn("乖離%",           format="%.2f"))
    add("ret_5d_%",   st.column_config.NumberColumn("5d수익(%)",       format="%.2f"))
    add("ret_10d_%",  st.column_config.NumberColumn("10d수익(%)",      format="%.2f"))
    add("EBS",        st.column_config.NumberColumn("EBS",            format="%d"))
    return cfg

def render_table(df, *, key: str, height=560):
    st.data_editor(
        df,
        key=key,
        width="stretch",
        height=height,
        hide_index=True,
        disabled=True,
        num_rows="fixed",
        column_config=column_config_for(df),
    )

# ---------- Load raw ----------
try:
    df_raw = load_csv_url(RAW_URL); log_src(df_raw, "remote", RAW_URL)
except Exception:
    if os.path.exists(LOCAL_RAW):
        df_raw = load_csv_path(LOCAL_RAW); log_src(df_raw, "local", LOCAL_RAW)
    else:
        st.error("❌ CSV가 없습니다. Actions에서 collector가 data/recommend_latest.csv를 올렸는지 확인하세요.")
        st.stop()

df_raw = normalize_cols(df_raw)

# 완제품(EBS/추천가) 여부 체크
has_ebs  = "EBS" in df_raw.columns and df_raw["EBS"].notna().any()
has_reco = all(c in df_raw.columns for c in ["추천매수가","추천매도가1","추천매도가2","손절가"]) and \
           df_raw[["추천매수가","추천매도가1","추천매도가2","손절가"]].notna().any().any()

if has_ebs and has_reco:
    df = df_raw.copy()
else:
    with st.status("🧮 원시 OHLCV → 지표/점수/추천가 생성 중...", expanded=False):
        df = enrich_from_ohlcv(df_raw)

latest = df.sort_values(["종목코드","날짜"]).groupby("종목코드").tail(1) if "날짜" in df.columns else df.copy()

with st.status("🏷️ 종목명 매핑 중...", expanded=False):
    latest = apply_names(latest)

latest = ensure_turnover(latest)
for c in ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가"]:
    if c in latest.columns:
        latest[c] = pd.to_numeric(latest[c], errors="coerce")

# ---------- Scoring & Ranking ----------
scored = add_eval_columns(latest, near_band_pct=NEAR_BAND_DEF)
scored = add_global_score(scored)

# 고정 컷: 거래대금, EBS
if "거래대금(억원)" in scored.columns:
    scored = scored[scored["거래대금(억원)"] >= MIN_TURNOVER]
if "EBS" in scored.columns:
    scored = scored[scored["EBS"] >= PASS_SCORE_EBS]

ranked = scored.sort_values(
    ["GLOBAL_SCORE","MOMO_SCORE","EV_SCORE","거래대금(억원)"],
    ascending=[False, False, False, False]
).head(10)

# ---------- View ----------
st.subheader("🔥 MOMO Top 10 (급등 추세 포착용)", anchor=False)
cols_out = [
    "시장","종목명","종목코드","종가",
    "GLOBAL_SCORE","MOMO_SCORE","EV_SCORE",
    "RR1","T1여유%","SL여유%","Now%",
    "RSI14","MACD_slope","Vol_Z","乖離%","ret_5d_%","거래대금(억원)",
    "EBS","근거",
]
for c in cols_out:
    if c not in ranked.columns: ranked[c]=np.nan

render_table(cast_for_editor(ranked[cols_out]), key="tbl_momo_top10")

st.download_button(
    "📥 MOMO Top 10 (CSV)",
    data=ranked[cols_out].to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_momo_top10.csv",
    mime="text/csv",
    key="dl_momo_top10",
)

with st.expander("ℹ️ 스코어 해석 가이드"):
    st.markdown("""
**GLOBAL_SCORE = 0.6·MOMO + 0.4·EV**  
- **MOMO_SCORE(0~100)**: 돌파(신고가 프록시), 거래대금/볼륨 확장, 트렌드 품질, 스퀴즈→확장(+), 과열/저유동(–)  
- **EV_SCORE(0~100)**: RR(목표1/손절), 손절여유·목표여유, Now 근접도(±1.5%), ERS(=EBS 컷+MACD_slope+RSI)  
**추천 운용**: Top10 중 **Now% ≤ 1.0~1.5**, **SL여유% ≥ 3**, **Vol_Z ≥ 2**, **RSI 55~70** 우선 검토.
""")
