# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.4.0 (Auto Update + Now-Ready Score)
- 추천 CSV: data/recommend_latest.csv (remote 우선)
- 이름맵:   data/krx_codes.csv (remote 우선) → FDR → pykrx 순 폴백
- OHLCV만 와도 화면에서 지표/EBS/추천가 생성
- 숫자(가격/억원) 콤마 포맷 + 진입지수(ERS), RR, 근접도 추가
- Top Picks(지금 진입 유효 + 높은 ERS) 상단 강조
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

st.set_page_config(page_title="LDY Pro Trader v3.4.0 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.4.0 (Auto Update)")
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
for c in ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가","ATR14","MA20"]:
    if c in latest.columns:
        latest[c] = pd.to_numeric(latest[c], errors="coerce")

# ---------------- Now-Readiness metrics ----------------
def _clip(x, lo, hi): 
    return np.minimum(np.maximum(x, lo), hi)

def compute_now_metrics(dfv: pd.DataFrame, entry_mode: str) -> pd.DataFrame:
    out = dfv.copy()

    # 진입가 기준
    if entry_mode == "추격(현재가)":
        out["진입가_사용"] = out["종가"]
    else:
        out["진입가_사용"] = out["추천매수가"]

    # RR 계산 (목표1/손절)
    e = pd.to_numeric(out["진입가_사용"], errors="coerce")
    t1 = pd.to_numeric(out["추천매도가1"], errors="coerce")
    stp = pd.to_numeric(out["손절가"], errors="coerce")
    c = pd.to_numeric(out["종가"], errors="coerce")

    rr = (t1 - e) / (e - stp)
    rr = rr.where((e.notna()) & (t1.notna()) & (stp.notna()) & ((e - stp) > 0), np.nan)
    out["RR"] = rr

    # 근접도: ATR14가 있으면 |c-e| / (1.5*ATR) → 0(멀다)~1(가깝다)
    if "ATR14" in out.columns and out["ATR14"].notna().any():
        prox = 1.0 - (np.abs(c - e) / (1.5 * out["ATR14"]))
        out["근접도"] = _clip(prox, 0.0, 1.0)
    else:
        # 대안: |c-e|/e 를 2% 스케일에 맵핑
        prox = 1.0 - (np.abs(c - e) / (0.02 * e))
        out["근접도"] = _clip(prox, 0.0, 1.0)

    # 목표/손절 여유 (현재가 기준)
    out["목표여유_%"] = (t1 - c) / c
    out["손절여유_%"] = (c - stp) / c

    # 보조 스코어
    # RR 스코어(0~1): RR 0~2.5 구간 선형
    rr_score = _clip(rr / 2.5, 0, 1)

    # 마진 스코어(0~1): 목표여유 0~3% 구간 선형
    margin_score = _clip(out["목표여유_%"] / 0.03, 0, 1)

    # RSI 스코어(0~1): 55 중심 ±30 폭
    rsi = pd.to_numeric(out.get("RSI14", np.nan), errors="coerce")
    rsi_score = 1 - _clip(np.abs(rsi - 55) / 30.0, 0, 1)

    # 거래량/모멘텀 스코어
    volz = pd.to_numeric(out.get("Vol_Z", np.nan), errors="coerce")
    vol_score = _clip(volz / 1.5, 0, 1)  # Vol_Z≈1.5 이상이면 만점

    macd_h = pd.to_numeric(out.get("MACD_hist", np.nan), errors="coerce")
    macd_sl = pd.to_numeric(out.get("MACD_slope", np.nan), errors="coerce")
    mom_score = ((macd_sl > 0).astype(float) + (macd_h > 0).astype(float)) / 2.0

    ebs = pd.to_numeric(out.get("EBS", np.nan), errors="coerce")
    ebs_score = _clip(ebs / 7.0, 0, 1)

    # 진입지수 ERS (0~1)
    ERS = (
        0.35 * rr_score +
        0.25 * out["근접도"].fillna(0) +
        0.10 * margin_score.fillna(0) +
        0.10 * rsi_score.fillna(0) +
        0.05 * vol_score.fillna(0) +
        0.05 * mom_score.fillna(0) +
        0.10 * ebs_score.fillna(0)
    )
    out["진입지수"] = ERS

    # 지금 진입 유효(목표1 ≥ 현재가, 현재가 > 손절)
    out["지금진입유효"] = (out["목표여유_%"] > 0) & (out["손절여유_%"] > 0) & rr.notna()

    # 신호 레이블
    cond_now = out["지금진입유효"] & (out["진입지수"] >= 0.65)
    cond_wait = (out["진입지수"] >= 0.50)
    out["진입신호"] = np.select(
        [cond_now, cond_wait],
        ["✅ Now", "⚠️ 대기"],
        default="⛔ Pass"
    )
    return out

# ------------- UI -------------
with st.expander("🔍 보기/필터", expanded=True):
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,2])
    with c1:
        only_entry = st.checkbox("🚀 초입 후보만 (EBS≥4)", value=("EBS" in latest.columns))
    with c2:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c3:
        entry_mode = st.selectbox("진입가 기준", ["기본(추천매수)", "추격(현재가)"], index=0)
    with c4:
        min_rr = st.slider("최소 RR(목표1/손절)", 0.0, 3.0, 0.0, 0.1)
    with c5:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

c6,c7,c8 = st.columns([1,1,1])
with c6:
    sort_key = st.selectbox("정렬", ["진입지수▼","EBS▼","거래대금▼","시가총액▼","RSI▲","RSI▼","종가▲","종가▼"], index=0)
with c7:
    only_now = st.checkbox("지금 진입 유효만(목표1≥현재가 & 현재가>손절)", value=False)
with c8:
    min_ers = st.slider("최소 진입지수(0~1)", 0.0, 1.0, 0.0, 0.05)

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

# Now metrics
view = compute_now_metrics(view, entry_mode)

# RR 필터 & 지금 진입 유효 필터 & ERS 필터
view = view[ (view["RR"].fillna(-1) >= float(min_rr)) ]
if only_now:
    view = view[ view["지금진입유효"] ]
if min_ers > 0:
    view = view[ view["진입지수"].fillna(0) >= float(min_ers) ]

def safe_sort(dfv, key):
    try:
        if key=="진입지수▼" and "진입지수" in dfv.columns:
            return dfv.sort_values(["진입지수","RR","거래대금(억원)"], ascending=[False,False,False], na_position="last")
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
    for alt in ["진입지수","EBS","거래대금(억원)","시가총액(억원)","종가"]:
        if alt in dfv.columns:
            return dfv.sort_values(alt, ascending=False, na_position="last")
    return dfv

view = safe_sort(view, sort_key)

# Top Picks 강조
top_show = view.copy()
top_show = top_show[top_show["지금진입유효"]].sort_values(["진입지수","RR","거래대금(억원)"], ascending=[False,False,False]).head(5)

st.write(f"📋 총 {len(latest):,}개 / 필터 후 {len(view):,}개 표시")

if len(top_show):
    best = top_show.iloc[0]
    cA, cB = st.columns([2,3])
    with cA:
        st.success(f"🥇 **지금 베스트**: {best.get('종목명','?')} ({best.get('종목코드','')})")
        st.metric("진입지수(0~1)", f"{best.get('진입지수',0):.2f}")
        st.metric("RR(목표1/손절)", f"{best.get('RR',np.nan):.2f}")
        st.metric("근접도(0~1)", f"{best.get('근접도',0):.2f}")
    with cB:
        st.write("**Top 5 Now Picks**")
        cols_top = ["진입신호","종목명","종목코드","진입지수","RR","종가","추천매수가","손절가","추천매도가1","목표여유_%","손절여유_%"]
        for c in cols_top:
            if c not in top_show.columns: top_show[c]=np.nan
        small = top_show[cols_top].copy()
        small["목표여유_%"] = (small["목표여유_%"]*100).round(2)
        small["손절여유_%"] = (small["손절여유_%"]*100).round(2)
        st.dataframe(
            small,
            use_container_width=True,
            hide_index=True
        )

# ---- 표 본문 ----
cols = [
    "진입신호",
    "시장","종목명","종목코드",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "RR","근접도","진입지수","목표여유_%","손절여유_%",
    "거래대금(억원)","시가총액(억원)",
    "EBS","근거",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
for c in cols:
    if c not in view.columns: view[c]=np.nan

view_fmt = view[cols].copy()

# 타입 캐스팅/표시 포맷
for c in ["종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]:
    if c in view_fmt.columns: view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce").round(0).astype("Int64")
for c in ["RR","근접도","진입지수","목표여유_%","손절여유_%","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"]:
    if c in view_fmt.columns: view_fmt[c] = pd.to_numeric(view_fmt[c], errors="coerce")

st.data_editor(
    view_fmt,
    width="stretch",
    height=680,
    hide_index=True,
    disabled=True,
    num_rows="fixed",
    column_config={
        # 텍스트
        "진입신호":     st.column_config.TextColumn("신호"),
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
        # RR/근접/지수
        "RR":           st.column_config.NumberColumn("RR(목표1/손절)",  format="%.2f"),
        "근접도":        st.column_config.NumberColumn("근접도(0~1)",     format="%.2f"),
        "진입지수":      st.column_config.NumberColumn("진입지수(0~1)",   format="%.2f"),
        "목표여유_%":     st.column_config.NumberColumn("목표여유(%)",     format="%.2f"),
        "손절여유_%":     st.column_config.NumberColumn("손절여유(%)",     format="%.2f"),
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
    },
)

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view_fmt.to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_entry_candidates_now_ready.csv",
    mime="text/csv"
)

with st.expander("ℹ️ 지표/점수 설명", expanded=False):
    st.markdown("""
**RR(목표1/손절)** = (목표가1 − 진입가) / (진입가 − 손절가)  
**근접도(0~1)** = 현재가가 진입가에 얼마나 가까운지 (1에 가까울수록 좋음)  
**진입지수(0~1)** = RR·근접도·목표여유·RSI·거래량·모멘텀·EBS를 종합한 즉시 진입 적합도  
**지금진입유효** = 목표가1 ≥ 현재가 이고 현재가 > 손절가 인 경우
""")
