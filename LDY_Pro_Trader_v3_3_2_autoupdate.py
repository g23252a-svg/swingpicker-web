# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.3.4 (Auto Update + ERS + RR/여유 필터 + 콤마 포맷)
- 추천 CSV: data/recommend_latest.csv (remote 우선)
- 이름맵:   data/krx_codes.csv (remote 우선) → FDR → pykrx 순 폴백
- OHLCV만 와도 화면에서 지표/EBS/추천가 생성
- '진입 준비도(ERS)'와 RR/손절·목표 여유%를 계산해 Top Picks/Now 리스트 제공
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
st.caption("매일 장마감 후 자동 업데이트되는 스윙 추천 종목 리스트 | ERS/RR 필터 내장 — Made by LDY")

RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"

PASS_SCORE = 4  # EBS 컷

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
    # 1) repo data/krx_codes.csv
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

# 완제품 여부 확인
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

# ---------- 파생: RR/여유/ERS ----------
def compute_risk_fields(x: pd.DataFrame) -> pd.DataFrame:
    v = x.copy()
    # 기본 값 확보
    entry = v.get("추천매수가")
    stop  = v.get("손절가")
    t1    = v.get("추천매도가1")
    close = v.get("종가")

    # 결측 방어: 엔트리 없으면 종가 사용
    v["entry_used"] = np.where(pd.notna(entry), entry, close)

    # 손절/목표 결측 보정 (없으면 계산 불가 → NaN 유지)
    v["stop_used"]  = stop
    v["t1_used"]    = t1

    # 최소 스탑 폭 보정(앱 레벨): max(2%*entry, 50원) — ATR 미존재 환경 대비
    # 수집기에서 ATR기반 최소폭을 이미 적용했다면 보정이 걸리지 않을 수 있음
    v["min_stop_gap"] = v["entry_used"] * 0.02
    v["stop_used"] = np.where(
        pd.notna(v["stop_used"]) & pd.notna(v["entry_used"]),
        np.maximum(v["stop_used"], v["entry_used"] - np.maximum(v["min_stop_gap"], 50.0)),
        v["stop_used"]
    )

    # 여유% & RR
    v["손절여유_%"]  = (v["entry_used"] - v["stop_used"]) / v["entry_used"] * 100.0
    v["목표1여유_%"] = (v["t1_used"]    - v["entry_used"]) / v["entry_used"] * 100.0
    v["RR"] = np.where(
        (pd.notna(v["t1_used"]) & pd.notna(v["stop_used"]) & (v["entry_used"] > v["stop_used"])),
        (v["t1_used"] - v["entry_used"]) / (v["entry_used"] - v["stop_used"]),
        np.nan
    )

    # ERS (Entry Readiness Score, 0~1)
    # - rr_norm: RR 1.5~3.0 구간 선호
    rr_norm = ((v["RR"] - 1.5) / (3.0 - 1.5)).clip(lower=0, upper=1)

    # - near_entry: 엔트리 근접도 (±0.8% 이내면 만점)
    v["entry_gap_%"] = (v["entry_used"] - v["종가"]) / v["entry_used"] * 100.0
    near_entry = (1 - (v["entry_gap_%"].abs() / 0.8)).clip(lower=0, upper=1)

    # - trend_norm: EBS 0~7 → 0~1 정규화(컷 4↑ 가점)
    ebs = pd.to_numeric(v.get("EBS"), errors="coerce").fillna(0)
    trend_norm = (ebs.clip(lower=0, upper=7) / 7.0)

    # - vol_norm: Vol_Z 1.0~2.0 → 0~1 (없으면 0.5)
    volz = pd.to_numeric(v.get("Vol_Z"), errors="coerce")
    vol_norm = ((volz - 1.0) / 1.0).clip(lower=0, upper=1)
    vol_norm = vol_norm.fillna(0.5)

    v["ERS"] = (0.40 * rr_norm) + (0.30 * near_entry) + (0.20 * trend_norm) + (0.10 * vol_norm)
    return v

latest = compute_risk_fields(latest)

# ------------- UI -------------
with st.expander("🔍 보기/필터", expanded=True):
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,2])
    with c1:
        only_entry = st.checkbox("🚀 초입 후보만 (EBS≥4)", value=("EBS" in latest.columns))
    with c2:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c3:
        sort_key = st.selectbox("정렬", ["ERS▼","RR▼","EBS▼","거래대금▼","시가총액▼","종가▲","종가▼"], index=0)
    with c4:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)
    with c5:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

with st.expander("🛠 Top Picks 조건", expanded=True):
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        rr_min = st.slider("최소 RR", 1.0, 3.0, 1.8, 0.1)
    with c2:
        stop_min = st.slider("손절여유 ≥ (%)", 0.0, 5.0, 2.0, 0.1)
    with c3:
        tgt_min = st.slider("목표1여유 ≥ (%)", 0.0, 10.0, 4.0, 0.1)
    with c4:
        ers_min = st.slider("ERS ≥", 0.00, 1.00, 0.65, 0.01)
    with c5:
        band = st.slider("Now 근접 밴드(±%)", 0.2, 2.0, 0.8, 0.1)

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

# 정렬
def safe_sort(dfv, key):
    try:
        if key=="ERS▼" and "ERS" in dfv.columns:
            return dfv.sort_values(["ERS","RR","거래대금(억원)"], ascending=[False,False,False])
        if key=="RR▼" and "RR" in dfv.columns:
            return dfv.sort_values(["RR","ERS"], ascending=[False,False])
        if key=="EBS▼" and "EBS" in dfv.columns:
            by = ["EBS"] + (["거래대금(억원)"] if "거래대금(억원)" in dfv.columns else [])
            return dfv.sort_values(by=by, ascending=[False]+[False]*(len(by)-1))
        if key=="거래대금▼" and "거래대금(억원)" in dfv.columns:
            return dfv.sort_values("거래대금(억원)", ascending=False)
        if key=="시가총액▼" and "시가총액(억원)" in dfv.columns:
            return dfv.sort_values("시가총액(억원)", ascending=False, na_position="last")
        if key=="종가▲" and "종가" in dfv.columns:
            return dfv.sort_values("종가", ascending=True, na_position="last")
        if key=="종가▼" and "종가" in dfv.columns:
            return dfv.sort_values("종가", ascending=False, na_position="last")
    except Exception:
        pass
    for alt in ["ERS","RR","EBS","거래대금(억원)","시가총액(억원)","종가"]:
        if alt in dfv.columns:
            return dfv.sort_values(alt, ascending=False, na_position="last")
    return dfv

view = safe_sort(view, sort_key)

if "EBS" in view.columns:
    view["통과"] = np.where(view["EBS"]>=PASS_SCORE, "🚀", "")

# 공통 컬럼
base_cols = [
    "통과","시장","종목명","종목코드",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "거래대금(억원)","시가총액(억원)",
    "EBS","근거",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
risk_cols = ["ERS","RR","손절여유_%","목표1여유_%","entry_gap_%"]
derived_cols = ["entry_used","stop_used","t1_used"]

for c in base_cols + risk_cols + derived_cols:
    if c not in view.columns: view[c] = np.nan

st.write(f"📋 총 {len(latest):,}개 / 표시 {min(len(view), int(topn)):,}개")

# ── 숫자 포맷(콤마) 적용 ──
def cast_and_format(dfv: pd.DataFrame) -> pd.DataFrame:
    v = dfv.copy()
    # 정수류
    for c in ["종가","추천매수가","손절가","추천매도가1","추천매도가2","entry_used","stop_used","t1_used","EBS"]:
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce").round(0).astype("Int64")
    # 실수류
    for c in ["거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","ERS","RR","손절여유_%","목표1여유_%","entry_gap_%"]:
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce")
    return v

# ===== Top Picks (조건 충족) =====
qual = view.copy()
qual = qual[
    (qual["RR"] >= rr_min) &
    (qual["손절여유_%"] >= stop_min) &
    (qual["목표1여유_%"] >= tgt_min) &
    (qual["ERS"] >= ers_min)
].copy()
qual = qual.sort_values(["ERS","RR","거래대금(억원)"], ascending=[False,False,False])

st.subheader("⭐ Top Picks (조건 충족)")
qp = cast_and_format(qual[base_cols + risk_cols].head(int(topn)))
st.data_editor(
    qp,
    width="stretch",
    height=420,
    hide_index=True,
    disabled=True,
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
        "추천매수가":    st.column_config.NumberColumn("추천매수가",     format="%,d"),
        "손절가":        st.column_config.NumberColumn("손절가",         format="%,d"),
        "추천매도가1":   st.column_config.NumberColumn("추천매도가1",    format="%,d"),
        "추천매도가2":   st.column_config.NumberColumn("추천매도가2",    format="%,d"),
        "EBS":          st.column_config.NumberColumn("EBS",            format="%d"),
        # 억원/지표/파생
        "거래대금(억원)": st.column_config.NumberColumn("거래대금(억원)",  format="%,.0f"),
        "시가총액(억원)": st.column_config.NumberColumn("시가총액(억원)",  format="%,.0f"),
        "RSI14":        st.column_config.NumberColumn("RSI14",          format="%.1f"),
        "乖離%":         st.column_config.NumberColumn("乖離%",           format="%.2f"),
        "MACD_hist":    st.column_config.NumberColumn("MACD_hist",      format="%.4f"),
        "MACD_slope":   st.column_config.NumberColumn("MACD_slope",     format="%.5f"),
        "Vol_Z":        st.column_config.NumberColumn("Vol_Z",          format="%.2f"),
        "ret_5d_%":     st.column_config.NumberColumn("ret_5d_%",       format="%.2f"),
        "ret_10d_%":    st.column_config.NumberColumn("ret_10d_%",      format="%.2f"),
        "ERS":          st.column_config.NumberColumn("ERS",            format="%.2f"),
        "RR":           st.column_config.NumberColumn("RR",             format="%.2f"),
        "손절여유_%":     st.column_config.NumberColumn("손절여유(%)",     format="%.2f"),
        "목표1여유_%":    st.column_config.NumberColumn("목표1여유(%)",    format="%.2f"),
        "entry_gap_%":   st.column_config.NumberColumn("엔트리괴리(%)",     format="%.2f"),
    },
)

# ===== ✅ Now (엔트리 근접 + 조건 충족) =====
now_mask = view["entry_gap_%"].abs() <= band
now_df = view[now_mask].copy()
now_df = now_df[
    (now_df["RR"] >= rr_min) &
    (now_df["손절여유_%"] >= stop_min) &
    (now_df["목표1여유_%"] >= tgt_min) &
    (now_df["ERS"] >= ers_min)
].sort_values(["ERS","RR","거래대금(억원)"], ascending=[False,False,False])

st.subheader("✅ Now (엔트리 근접 & 조건 충족)")
npv = cast_and_format(now_df[base_cols + risk_cols].head(50))
st.data_editor(
    npv,
    width="stretch",
    height=320,
    hide_index=True,
    disabled=True,
    num_rows="fixed",
    column_config={
        "통과":         st.column_config.TextColumn(" "),
        "시장":         st.column_config.TextColumn("시장"),
        "종목명":       st.column_config.TextColumn("종목명"),
        "종목코드":     st.column_config.TextColumn("종목코드"),
        "근거":         st.column_config.TextColumn("근거"),
        "종가":          st.column_config.NumberColumn("종가",           format="%,d"),
        "추천매수가":    st.column_config.NumberColumn("추천매수가",     format="%,d"),
        "손절가":        st.column_config.NumberColumn("손절가",         format="%,d"),
        "추천매도가1":   st.column_config.NumberColumn("추천매도가1",    format="%,d"),
        "추천매도가2":   st.column_config.NumberColumn("추천매도가2",    format="%,d"),
        "EBS":          st.column_config.NumberColumn("EBS",            format="%d"),
        "거래대금(억원)": st.column_config.NumberColumn("거래대금(억원)",  format="%,.0f"),
        "시가총액(억원)": st.column_config.NumberColumn("시가총액(억원)",  format="%,.0f"),
        "RSI14":        st.column_config.NumberColumn("RSI14",          format="%.1f"),
        "乖離%":         st.column_config.NumberColumn("乖離%",           format="%.2f"),
        "MACD_hist":    st.column_config.NumberColumn("MACD_hist",      format="%.4f"),
        "MACD_slope":   st.column_config.NumberColumn("MACD_slope",     format="%.5f"),
        "Vol_Z":        st.column_config.NumberColumn("Vol_Z",          format="%.2f"),
        "ret_5d_%":     st.column_config.NumberColumn("ret_5d_%",       format="%.2f"),
        "ret_10d_%":    st.column_config.NumberColumn("ret_10d_%",      format="%.2f"),
        "ERS":          st.column_config.NumberColumn("ERS",            format="%.2f"),
        "RR":           st.column_config.NumberColumn("RR",             format="%.2f"),
        "손절여유_%":     st.column_config.NumberColumn("손절여유(%)",     format="%.2f"),
        "목표1여유_%":    st.column_config.NumberColumn("목표1여유(%)",    format="%.2f"),
        "entry_gap_%":   st.column_config.NumberColumn("엔트리괴리(%)",     format="%.2f"),
    },
)

# ===== 전체 테이블 =====
st.subheader("📋 전체 리스트")
cols_all = base_cols + risk_cols
view_fmt = cast_and_format(view[cols_all].head(int(topn)))
st.data_editor(
    view_fmt,
    width="stretch",
    height=640,
    hide_index=True,
    disabled=True,
    num_rows="fixed",
    column_config={
        "통과":         st.column_config.TextColumn(" "),
        "시장":         st.column_config.TextColumn("시장"),
        "종목명":       st.column_config.TextColumn("종목명"),
        "종목코드":     st.column_config.TextColumn("종목코드"),
        "근거":         st.column_config.TextColumn("근거"),
        "종가":          st.column_config.NumberColumn("종가",           format="%,d"),
        "추천매수가":    st.column_config.NumberColumn("추천매수가",     format="%,d"),
        "손절가":        st.column_config.NumberColumn("손절가",         format="%,d"),
        "추천매도가1":   st.column_config.NumberColumn("추천매도가1",    format="%,d"),
        "추천매도가2":   st.column_config.NumberColumn("추천매도가2",    format="%,d"),
        "EBS":          st.column_config.NumberColumn("EBS",            format="%d"),
        "거래대금(억원)": st.column_config.NumberColumn("거래대금(억원)",  format="%,.0f"),
        "시가총액(억원)": st.column_config.NumberColumn("시가총액(억원)",  format="%,.0f"),
        "RSI14":        st.column_config.NumberColumn("RSI14",          format="%.1f"),
        "乖離%":         st.column_config.NumberColumn("乖離%",           format="%.2f"),
        "MACD_hist":    st.column_config.NumberColumn("MACD_hist",      format="%.4f"),
        "MACD_slope":   st.column_config.NumberColumn("MACD_slope",     format="%.5f"),
        "Vol_Z":        st.column_config.NumberColumn("Vol_Z",          format="%.2f"),
        "ret_5d_%":     st.column_config.NumberColumn("ret_5d_%",       format="%.2f"),
        "ret_10d_%":    st.column_config.NumberColumn("ret_10d_%",      format="%.2f"),
        "ERS":          st.column_config.NumberColumn("ERS",            format="%.2f"),
        "RR":           st.column_config.NumberColumn("RR",             format="%.2f"),
        "손절여유_%":     st.column_config.NumberColumn("손절여유(%)",     format="%.2f"),
        "목표1여유_%":    st.column_config.NumberColumn("목표1여유(%)",    format="%.2f"),
        "entry_gap_%":   st.column_config.NumberColumn("엔트리괴리(%)",     format="%.2f"),
    },
)

st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view_fmt.to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_entry_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ 로직 설명 (EBS/ERS/RR)", expanded=False):
    st.markdown("""
**EBS (0~7)**  
- RSI 45~65 / MACD상승 / MA20±4% / 거래량증가(Vol_Z>1.2) / MA20↑ / MACD>sig / 5d<10%

**RR (Risk-Reward)**  
- RR = (T1−엔트리) / (엔트리−손절)  
- 앱 레벨에서 최소 스탑폭을 **max(2%*엔트리, 50원)** 으로 보정(수집기는 ATR 기반 최소폭 권장)

**ERS (0~1)**  
- 0.40×RR정규화(1.5~3.0) + 0.30×엔트리근접(±0.8%) + 0.20×EBS정규화 + 0.10×거래량 추세(Vol_Z)
- 기본 Top Picks 컷: RR≥1.8, 손절여유≥2%, 목표1여유≥4%, ERS≥0.65

**Now 섹션**  
- 엔트리괴리(|Entry−Close|/Entry) ≤ **±{band:.1f}%** & Top Picks 조건 충족
""".format(band=band))
