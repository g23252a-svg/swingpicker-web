# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.4.2 (Auto Update + EV Score + Top Picks)
- 결측 추천가(엔트리/손절/목표1) 자동 보정: OHLCV가 부족해도 EV가 0만 찍히지 않도록 기본값 생성
- EV 게이트: MACD_hist/RSI가 NaN이어도 페널티로 반영(=0.90), slope≤0는 강펀치(×0.75)
- Streamlit DuplicateElementId 방지 유지
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
st.set_page_config(page_title="LDY Pro Trader v3.4.2 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.4.2 (Auto Update)")
st.caption("매일 장마감 후 자동 업데이트되는 스윙 추천 종목 리스트 | EV스코어·TopPick 내장")

# ---------------- constants ----------------
RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"
PASS_SCORE = 4

# 결측 보정 기본값(엔트리 기준 %)
DEF_T1_PCT = 0.06   # +6% 목표1
DEF_SL_PCT = 0.03   # -3% 손절
# ATR이 있으면 ATR 기반(엔트리±), 없으면 위 % 사용

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
            band_lo, band_hi = ma20 - 0.5*atr, ma20 + 0.5*atr
            base_entry = ma20
            entry = min(max(base_entry, band_lo), band_hi)
            t1, t2, stp = entry + 1.0*atr, entry + 1.8*atr, entry - 1.2*atr
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
    if FDR_OK:
        try:
            lst = fdr.StockListing("KRX")
            m = lst.rename(columns={"Code":"종목코드","Name":"종목명"})[["종목코드","종목명"]]
            m["종목코드"] = m["종목코드"].astype(str).map(z6)
            return m.drop_duplicates("종목코드")
        except Exception:
            pass
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

# -------- 결측 추천가 보정(핵심) --------
def fill_reco_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """추천매수가/손절가/목표1 결측 시 기본 규칙으로 자동 생성"""
    df = df.copy()
    for col in ["종가","추천매수가","손절가","추천매도가1","추천매도가2","ATR14","MA20"]:
        if col not in df.columns:
            df[col] = np.nan

    close = pd.to_numeric(df["종가"], errors="coerce")
    entry = pd.to_numeric(df["추천매수가"], errors="coerce")
    stop  = pd.to_numeric(df["손절가"], errors="coerce")
    t1    = pd.to_numeric(df["추천매도가1"], errors="coerce")
    t2    = pd.to_numeric(df["추천매도가2"], errors="coerce")
    atr   = pd.to_numeric(df.get("ATR14"), errors="coerce")
    ma20  = pd.to_numeric(df.get("MA20"), errors="coerce")

    # entry 결측 → MA20이 있으면 MA20을 ±0.5*ATR로 클램프, 없으면 close 사용
    use_atr = atr.notna() & (atr > 0) & ma20.notna()
    entry_calc = np.where(use_atr, np.clip(ma20, ma20 - 0.5*atr, ma20 + 0.5*atr), close)
    df.loc[entry.isna(), "추천매수가"] = np.round(entry_calc[entry.isna()], 0)

    # stop/t1 결측 → ATR 있으면 ATR 기반, 없으면 % 기반
    entry = pd.to_numeric(df["추천매수가"], errors="coerce")  # 업데이트된 entry 다시 로드
    use_atr = atr.notna() & (atr > 0) & entry.notna()

    stop_calc = np.where(use_atr, entry - 1.2*atr, entry * (1 - DEF_SL_PCT))
    t1_calc   = np.where(use_atr, entry + 1.0*atr, entry * (1 + DEF_T1_PCT))
    t2_calc   = np.where(use_atr, entry + 1.8*atr, entry * (1 + DEF_T1_PCT*1.8))

    df.loc[stop.isna() & entry.notna(), "손절가"]      = np.round(stop_calc[stop.isna() & entry.notna()], 0)
    df.loc[t1.isna()   & entry.notna(), "추천매도가1"] = np.round(t1_calc[t1.isna()   & entry.notna()], 0)
    df.loc[t2.isna()   & entry.notna(), "추천매도가2"] = np.round(t2_calc[t2.isna()   & entry.notna()], 0)

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

# 완제품 체크 → 미완이면 enrich 시도, 그래도 비면 보정 채움
has_ebs  = "EBS" in df_raw.columns and df_raw["EBS"].notna().any()
has_reco = all(c in df_raw.columns for c in ["추천매수가","추천매도가1","추천매도가2","손절가"]) and \
           df_raw[["추천매수가","추천매도가1","추천매도가2","손절가"]].notna().any().any()

if has_ebs and has_reco:
    df = df_raw.copy()
else:
    with st.status("🧮 원시 OHLCV → 지표/점수/추천가 생성 중...", expanded=False):
        df = enrich_from_ohlcv(df_raw)
    # 여전히 추천가가 비어있으면 기본 규칙으로 생성
    df = fill_reco_if_missing(df)

# 최신 행만
latest = df.sort_values(["종목코드","날짜"]).groupby("종목코드").tail(1) if "날짜" in df.columns else df.copy()

# 이름 매핑
with st.status("🏷️ 종목명 매핑 중...", expanded=False):
    latest = apply_names(latest)

# 숫자 캐스팅 & 거래대금 보강
latest = ensure_turnover(latest)
for c in ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가","ATR14","MA20"]:
    if c in latest.columns:
        latest[c] = pd.to_numeric(latest[c], errors="coerce")

# ---------------- EV score ----------------
def _clip01(x):
    try:
        if pd.isna(x): return 0.0
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0

def make_ev_score(df: pd.DataFrame) -> pd.Series:
    """
    기대값 기반 EV 점수 0~100.
    NaN은 보수적으로 페널티 처리.
    """
    rr1  = (pd.to_numeric(df.get("RR1"), errors="coerce") - 1.0) / (3.0 - 1.0)
    rr1  = np.vectorize(_clip01)(rr1)

    t1r  = np.vectorize(_clip01)(pd.to_numeric(df.get("T1여유%"), errors="coerce") / 8.0)   # 8%에서 1.0
    slr  = np.vectorize(_clip01)(pd.to_numeric(df.get("SL여유%"), errors="coerce") / 4.0)   # 4%에서 1.0
    ers  = np.vectorize(_clip01)(pd.to_numeric(df.get("ERS"), errors="coerce") / 3.0)
    near = np.vectorize(_clip01)(1.0 - (pd.to_numeric(df.get("Now%"), errors="coerce").abs() / 1.0))  # ±1% 근접=1
    liq  = np.vectorize(_clip01)(np.log10(pd.to_numeric(df.get("거래대금(억원)"), errors="coerce").fillna(0) + 1) / 3.0)

    base = 0.25*rr1 + 0.20*t1r + 0.15*slr + 0.20*ers + 0.10*near + 0.10*liq

    hist  = pd.to_numeric(df.get("MACD_hist"), errors="coerce")
    slope = pd.to_numeric(df.get("MACD_slope"), errors="coerce")
    rsi   = pd.to_numeric(df.get("RSI14"), errors="coerce")

    # NaN도 페널티: >조건을 만족하지 않으면 페널티로 간주
    cond_hist_pos = (hist > 0)
    cond_slope_pos = (slope > 0)
    cond_rsi_in = (rsi >= 45) & (rsi <= 68)

    gate = np.ones(len(df), dtype=float)
    gate *= np.where(cond_hist_pos.fillna(False), 1.00, 0.90)  # hist 양수 아니면 0.90
    gate *= np.where(cond_slope_pos.fillna(False), 1.00, 0.75) # slope 양수 아니면 0.75
    gate *= np.where(cond_rsi_in.fillna(False), 1.00, 0.90)    # RSI 범위 밖/NaN 0.90

    ev_raw = base * gate
    ev = (100.0 * ev_raw).clip(0, 100).round(1)

    # 상위 퍼센타일 기준 리스케일 (스코어가 과도하게 낮게 몰리는 것 완화)
    try:
        p95 = np.nanpercentile(ev, 95)
        if p95 > 0:
            ev = (ev * (95.0 / p95)).clip(0, 100).round(1)
    except Exception:
        pass

    return ev

# ---------------- helper: scoring ----------------
def add_eval_columns(df_in: pd.DataFrame, near_band_pct: float) -> pd.DataFrame:
    """RR1/여유%/ERS/EV_SCORE 계산 컬럼 추가"""
    df = df_in.copy()
    for col in ["종가","추천매수가","손절가","추천매도가1","RSI14","MACD_slope","MACD_hist","EBS","거래대금(억원)"]:
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

    df["Now%"]    = (close.sub(entry).abs() / entry * 100).replace([np.inf, -np.inf], np.nan)
    df["T1여유%"] = (t1.sub(close) / close * 100).replace([np.inf, -np.inf], np.nan)
    df["SL여유%"] = (close.sub(stop) / close * 100).replace([np.inf, -np.inf], np.nan)

    ebs_ok  = (pd.to_numeric(df.get("EBS"), errors="coerce") >= PASS_SCORE).astype(int)
    macd_ok = (pd.to_numeric(df.get("MACD_slope"), errors="coerce") > 0).astype(int)
    rsi_v   = pd.to_numeric(df.get("RSI14"), errors="coerce")
    rsi_ok  = ((rsi_v >= 45) & (rsi_v <= 65)).astype(int)
    df["ERS"] = (ebs_ok + macd_ok + rsi_ok).astype(float)

    df["EV_SCORE"] = make_ev_score(df)
    return df

def cast_for_editor(df):
    df = df.copy()
    int_like = ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]
    for c in int_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
    float_like = [
        "거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope",
        "Vol_Z","ret_5d_%","ret_10d_%","EV_SCORE","ERS","RR1","Now%","T1여유%","SL여유%"
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
    return cfg

def render_table(df, *, key: str, height=620):
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

# ---------------- Filters (공통) ----------------
with st.container():
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
    with c1:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 0, step=50, key="flt_turn")
    with c2:
        sort_key = st.selectbox("정렬", ["EV_SCORE▼","EBS▼","거래대금▼","시가총액▼","RSI▲","RSI▼","종가▲","종가▼"], index=0, key="flt_sort")
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
    for alt in ["EV_SCORE","EBS","거래대금(억원)","시가총액(억원)","종가"]:
        if alt in dfv.columns:
            return dfv.sort_values(alt, ascending=False, na_position="last")
    return dfv

# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["🟢 Top Picks", "📋 전체 보기"])

with tab1:
    st.subheader("🛠 Top Picks 조건", anchor=False)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        rr_min = st.slider("최소 RR(목표1/손절)", 1.00, 3.00, 1.00, step=0.25, key="tp_rr")
    with c2:
        ers_min = st.slider("ERS ≥", 0.00, 3.00, 1.00, step=0.50, key="tp_ers")
    with c3:
        sl_min = st.slider("손절여유 ≥ (%)", 0.00, 10.00, 0.00, step=0.50, key="tp_sl")
    with c4:
        t1_min = st.slider("목표1여유 ≥ (%)", 0.00, 15.00, 0.00, step=0.50, key="tp_t1")
    with c5:
        near_band = st.slider("Now 근접 밴드(±%)", 0.00, 3.00, 0.00, step=0.10, key="tp_near")

    scored = add_eval_columns(view_base, near_band)

    tp = scored.copy()
    tp = tp[tp["EBS"] >= PASS_SCORE]
    tp = tp.dropna(subset=["추천매수가","손절가","추천매도가1","종가"])

    if rr_min > 0:
        tp = tp[tp["RR1"] >= rr_min]
    if ers_min > 0:
        tp = tp[tp["ERS"] >= ers_min]
    if sl_min > 0:
        tp = tp[tp["SL여유%"] >= sl_min]
    if t1_min > 0:
        tp = tp[tp["T1여유%"] >= t1_min]
    if near_band > 0:
        tp = tp[tp["Now%"] <= near_band]

    tp = safe_sort(tp, sort_key).head(int(topn))

    if "EBS" in tp.columns:
        tp["통과"] = np.where(tp["EBS"]>=PASS_SCORE, "🚀", "")

    cols = [
        "통과","시장","종목명","종목코드",
        "종가","추천매수가","손절가","추천매도가1","추천매도가2",
        "EV_SCORE","ERS","RR1","Now%","T1여유%","SL여유%",
        "거래대금(억원)","시가총액(억원)",
        "EBS","근거",
        "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
    ]
    for c in cols:
        if c not in tp.columns: tp[c]=np.nan

    st.write(f"📋 총 {len(view_base):,}개 / 표시 {min(len(tp), int(topn)):,}개")
    tp_fmt = cast_for_editor(tp[cols])
    render_table(tp_fmt, key="tbl_top_picks")

    st.download_button(
        "📥 Top Picks 다운로드 (CSV)",
        data=tp[cols].to_csv(index=False, encoding="utf-8-sig"),
        file_name="ldy_top_picks.csv",
        mime="text/csv",
        key="dl_top_picks",
    )

with tab2:
    scored_all = add_eval_columns(view_base, near_band_pct=0.0)
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
    v_fmt = cast_for_editor(view[cols])
    render_table(v_fmt, key="tbl_full_view")

    st.download_button(
        "📥 전체 보기 다운로드 (CSV)",
        data=view[cols].to_csv(index=False, encoding="utf-8-sig"),
        file_name="ldy_entry_candidates.csv",
        mime="text/csv",
        key="dl_full_view",
    )

with st.expander("ℹ️ 점수/지표 설명", expanded=False):
    st.markdown("""
**EBS(0~7)**: RSI 45~65 / MACD↑ / MA20±4% / VolZ>1.2 / MA20↑ / MACD>0 / 5d<10%  
**RR1**: (목표1−추천매수) / (추천매수−손절)  
**Now%**: |현재가−추천매수|/추천매수×100  
**T1여유%**: (목표1−현재가)/현재가×100  
**SL여유%**: (현재가−손절)/현재가×100  
**ERS(0~3)**: EBS 통과(≥4) + MACD_slope>0 + RSI 45~65  
**EV_SCORE**: 0.25·RR + 0.20·T1여유 + 0.15·SL여유 + 0.20·ERS + 0.10·근접 + 0.10·유동성  
→ 이후 MACD_hist 양수 아님 ×0.90, MACD_slope≤0 ×0.75, RSI 범위 밖/NaN ×0.90, p95 리스케일
""")
