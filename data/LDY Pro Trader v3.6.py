# -*- coding: utf-8 -*-
"""
LDY Pro Trader v4.0 — Global Rank (Single Composite Score)
- 한 화면: '오늘의 GLOBAL TOP 10'만 고정 노출 (가중치/슬라이더 없음)
- 모든 지표(리스크·보상·유동성·모멘텀·과열/과매수·근접도 등) → 단일 점수 LDY_SCORE(0~100)
- LDY_RANK = LDY_SCORE 내림차순 순위 (1위가 가장 유망)
- 유동성 하드컷: KOSPI≥200억, KOSDAQ≥100억 (비면 자동 완화)
"""

import os, io, math, requests, numpy as np, pandas as pd, streamlit as st
from datetime import datetime

# -------- Optional deps (이름 맵 폴백용) --------
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

# -------- Page --------
st.set_page_config(page_title="LDY Pro Trader v4.0 — Global Rank", layout="wide")
st.title("🏆 LDY Pro Trader v4.0 — Global Rank")
st.caption("모든 지표를 단일 점수로 종합 → 1위가 가장 유망한 종목")

# -------- Constants --------
RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"

PASS_EBS = 4                 # 품질 게이트
MIN_TURN_KOSPI = 200.0       # 유동성 하드컷
MIN_TURN_KOSDAQ = 100.0
MIN_TURN_DEFAULT = 100.0

# 고정 가중치(합=1.0) — 조정 UI 없음
W_RR   = 0.25  # 보상대비위험 (RR1)
W_T1   = 0.18  # 목표1 여유
W_SL   = 0.12  # 손절 여유
W_NEAR = 0.12  # 현재가-추천가 근접
W_MOM  = 0.10  # 모멘텀(ERS+MACD_slope+RSI중심 보너스)
W_LIQ  = 0.13  # 유동성(거래대금 퍼센타일)
W_TEC  = 0.10  # 기술균형(VolZ 스윗스팟 +乖離 안정)

# 페널티(점수에서 직접 차감, 0~30 범위 가정)
P_OVERHEAT_5D = 6.0   # 단기 과열(ret_5d_%) 과도 시
P_OVERHEAT_10D= 6.0   # 중기 과열(ret_10d_%)
P_RSI_OUT     = 4.0   # RSI 45~65 벗어남
P_MACD_NEG    = 4.0   # MACD 기울기 음수
P_NEAR_FAR    = 4.0   # 엔트리에서 너무 멀어짐
P_LIQ_LOW     = 4.0   # 유동성 하위권
P_VOL_SPIKE   = 2.0   # 변동성 과도 스파이크(VolZ≫)

# -------- IO helpers --------
@st.cache_data(ttl=300)
def load_csv_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30); r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=300)
def load_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")

def log_src(df: pd.DataFrame, src: str, url_or_path: str):
    st.info(f"상태 ✅ 데이터 로드: {src}\n\n{url_or_path}")
    st.success(f"📅 표시시각: {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M')} · 행수: {len(df):,}")

# -------- Utils --------
def z6(x) -> str:
    s = str(x)
    return s.zfill(6) if s.isdigit() else s

def ensure_turnover(df: pd.DataFrame) -> pd.DataFrame:
    if "거래대금(억원)" not in df.columns:
        base = None
        if "거래대금(원)" in df.columns:
            base = pd.to_numeric(df["거래대금(원)"], errors="coerce")
        elif all(c in df.columns for c in ["거래량","종가"]):
            base = pd.to_numeric(df["거래량"], errors="coerce") * pd.to_numeric(df["종가"], errors="coerce")
        if base is not None:
            df["거래대금(억원)"] = (base/1e8).round(2)
    return df

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cmap = {
        "Date":"날짜","date":"날짜",
        "Code":"종목को드","티커":"종목코드","ticker":"종목코드",
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
    for c in ["시가","고가","저가","종가","거래량","거래대금(원)","시가총액(원)","거래대금(억원)","시가총액(억원)",
              "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS",
              "추천매수가","추천매도가1","추천매도가2","손절가"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return ensure_turnover(df)

# -------- 이름맵 --------
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
                lst = stock.get_market_ticker_list(today, market=mk) or []
                for t in lst:
                    nm = None
                    try: nm = stock.get_market_ticker_name(t)
                    except Exception: pass
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

# -------- Load --------
try:
    df_raw = load_csv_url(RAW_URL); log_src(df_raw, "remote", RAW_URL)
except Exception:
    if os.path.exists(LOCAL_RAW):
        df_raw = load_csv_path(LOCAL_RAW); log_src(df_raw, "local", LOCAL_RAW)
    else:
        st.error("❌ CSV가 없습니다. Actions 수집 상태를 확인하세요.")
        st.stop()

df = normalize_cols(df_raw)
df = apply_names(df)

# 최신행만 사용
latest = df.sort_values(["종목코드","날짜"]).groupby("종목코드").tail(1) if "날짜" in df.columns else df.copy()

# -------- 하드 유동성 컷 --------
def liquidity_gate(x: pd.Series, market: pd.Series) -> pd.Series:
    min_map = {"KOSPI": MIN_TURN_KOSPI, "KOSDAQ": MIN_TURN_KOSDAQ}
    mins = market.map(min_map).fillna(MIN_TURN_DEFAULT)
    return x >= mins

# -------- 정규화 유틸 --------
def cap_q(s: pd.Series, q=90, floor=1.0):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum()==0: return floor
    c = np.nanpercentile(s, q)
    if not np.isfinite(c) or c<=0: c=floor
    return max(float(c), floor)

def pct_norm_pos(s: pd.Series, q=90, floor=1.0):
    # 양수만 인정(음수→0), q-캡으로 0~1 스케일
    s = pd.to_numeric(s, errors="coerce").clip(lower=0)
    cap = cap_q(s, q=q, floor=floor)
    return np.clip(s / cap, 0, 1)

def inv_dist_norm(dist: pd.Series, cap):
    # 0일수록 1, cap 넘으면 0
    d = pd.to_numeric(dist, errors="coerce")
    return np.clip(1 - (d / cap), 0, 1)

# -------- Composite Score --------
def build_global_score(lat: pd.DataFrame) -> pd.DataFrame:
    x = lat.copy()

    # 필수 수치
    for c in ["종가","추천매수가","손절가","추천매도가1","거래대금(억원)","RSI14","MACD_slope","Vol_Z","乖離%","ret_5d_%","ret_10d_%","EBS"]:
        if c not in x.columns: x[c]=np.nan

    close = pd.to_numeric(x["종가"], errors="coerce")
    entry = pd.to_numeric(x["추천매수가"], errors="coerce")
    stop  = pd.to_numeric(x["손절가"], errors="coerce")
    t1    = pd.to_numeric(x["추천매도가1"], errors="coerce")
    turn  = pd.to_numeric(x["거래대금(억원)"], errors="coerce")
    rsi   = pd.to_numeric(x["RSI14"], errors="coerce")
    slope = pd.to_numeric(x["MACD_slope"], errors="coerce")
    volz  = pd.to_numeric(x["Vol_Z"], errors="coerce")
    kairi = pd.to_numeric(x["乖離%"], errors="coerce")
    r5    = pd.to_numeric(x["ret_5d_%"], errors="coerce")
    r10   = pd.to_numeric(x["ret_10d_%"], errors="coerce")
    ebs   = pd.to_numeric(x["EBS"], errors="coerce").fillna(0)

    # RR1, 여유, 근접
    rr_den = (entry - stop)
    rr1 = (t1 - entry) / rr_den.replace(0, np.nan)
    rr1 = rr1.mask(entry.isna() | stop.isna() | t1.isna())
    now_gap = (close - entry).abs() / entry * 100
    t1_room = (t1 - close) / close * 100
    sl_room = (close - stop) / close * 100

    # 정규화(상한=분포 기반, 과한 outlier 방지)
    rr_norm   = pct_norm_pos(rr1, q=90, floor=1.0)
    t1_norm   = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1)
    sl_norm   = np.clip(sl_room / cap_q(sl_room, q=90, floor=3.0), 0, 1)
    near_norm = inv_dist_norm(now_gap, cap=cap_q(now_gap, q=75, floor=1.0))
    # 모멘텀 묶음: ERS(=EBS≥4 + slope>0 + RSI in-band) + slope(양수부분) + RSI 중심보너스
    ers_bits = (ebs>=PASS_EBS).astype(int) + (slope>0).astype(int) + ((rsi>=45)&(rsi<=65)).astype(int)
    ers_norm = np.clip(ers_bits/3.0, 0, 1)
    slope_pos_norm = pct_norm_pos(slope, q=90, floor=1.0)
    rsi_center = 1 - np.minimum((rsi-55).abs()/10, 1)           # 55에 가까울수록 1 (±10 범위)
    rsi_center = rsi_center.clip(lower=0, upper=1).fillna(0)
    mom_norm = np.clip(0.5*ers_norm + 0.3*slope_pos_norm + 0.2*rsi_center, 0, 1)

    # 유동성: 거래대금 퍼센타일 스케일
    if turn.notna().any():
        lo = np.nanpercentile(turn, 30)
        hi = np.nanpercentile(turn, 90)
        span = max(hi - lo, 1e-9)
        liq_norm = np.clip((turn - lo) / span, 0, 1)
    else:
        liq_norm = pd.Series(0.0, index=x.index)

    # 기술 균형: VolZ 스윗스팟(≈1) +乖離 안정(절대乖離 낮을수록)
    vol_sweet = 1 - np.minimum((volz - 1).abs()/3, 1)           # 1에 가까울수록 좋음
    vol_sweet = vol_sweet.clip(0,1).fillna(0)
    kairi_norm = 1 - np.minimum(kairi.abs()/cap_q(kairi.abs(), q=80, floor=3.0), 1)
    kairi_norm = kairi_norm.clip(0,1).fillna(0)
    tec_norm = np.clip(0.6*vol_sweet + 0.4*kairi_norm, 0, 1)

    # 가중합 (0~100)
    base_score = 100*(W_RR*rr_norm + W_T1*t1_norm + W_SL*sl_norm + W_NEAR*near_norm + W_MOM*mom_norm + W_LIQ*liq_norm + W_TEC*tec_norm)

    # 페널티 (점수 차감)
    pen = pd.Series(0.0, index=x.index)

    # 단기/중기 과열
    pen += P_OVERHEAT_5D * np.clip((r5 - 10)/10, 0, 1)      # 5일 +10% 초과부터 최대 패널티
    pen += P_OVERHEAT_10D* np.clip((r10 - 20)/20, 0, 1)     # 10일 +20% 초과부터

    # RSI 밴드 이탈(45~65)
    rsi_out = (rsi < 45) | (rsi > 65)
    pen += P_RSI_OUT * rsi_out.astype(float)

    # MACD 기울기 음수
    pen += P_MACD_NEG * (slope < 0).astype(float)

    # 엔트리와 괴리 과다
    near_cap = cap_q(now_gap, q=75, floor=1.0)
    pen += P_NEAR_FAR * np.clip((now_gap - near_cap)/near_cap, 0, 1)

    # 유동성 하위권 (하위 20%)
    if turn.notna().any():
        p20 = np.nanpercentile(turn, 20)
        pen += P_LIQ_LOW * (turn < p20).astype(float)

    # 변동성 스파이크 (VolZ > 3)
    pen += P_VOL_SPIKE * (volz > 3).astype(float)

    score = np.clip(base_score - pen, 0, 100)
    x["RR1"]      = rr1
    x["Now%"]     = now_gap
    x["T1여유%"]   = t1_room
    x["SL여유%"]   = sl_room
    x["ERS"]      = ers_bits.astype(float)
    x["LDY_SCORE"]= score.round(1)

    # 하드 유동성 컷(기본), 비면 완화용 플래그
    gate_ok = liquidity_gate(x["거래대금(억원)"], x["시장"])
    x["_GATE_OK"] = gate_ok.fillna(False)

    # 랭크(내림차순)
    x = x.sort_values("LDY_SCORE", ascending=False, na_position="last")
    x["LDY_RANK"] = range(1, len(x)+1)
    return x

scored = build_global_score(latest)

# 기본 필터: 추천가/손절/목표1 모두 있어야 랭킹 포함
scored = scored.dropna(subset=["추천매수가","손절가","추천매도가1","종가"])

# 품질게이트(EBS) & 유동성 하드컷
base = scored[ (pd.to_numeric(scored.get("EBS"), errors="coerce") >= PASS_EBS) & (scored["_GATE_OK"]) ].copy()

# 비거나 Top10 미만이면 자동 완화(EBS≥3 + 유동성 완화: KOSPI150/KOSDAQ80)
if len(base) < 10:
    fb = scored[ pd.to_numeric(scored.get("EBS"), errors="coerce") >= (PASS_EBS-1) ].copy()
    # 완화 컷
    MIN_KOSPI_F, MIN_KOSDAQ_F = 150.0, 80.0
    mm = {"KOSPI": MIN_KOSPI_F, "KOSDAQ": MIN_KOSDAQ_F}
    fb["_min_turn"] = fb["시장"].map(mm).fillna(MIN_TURN_DEFAULT)
    fb = fb[ fb["거래대금(억원)"] >= fb["_min_turn"] ]
    base_codes = set(base["종목코드"])
    fill = fb[~fb["종목코드"].isin(base_codes)]
    base = pd.concat([base, fill]).sort_values("LDY_SCORE", ascending=False).head(50)

# 최종 Top 10
top10 = base.sort_values("LDY_SCORE", ascending=False, na_position="last").head(10).copy()
top10["통과"] = np.where(pd.to_numeric(top10.get("EBS"), errors="coerce") >= PASS_EBS, "🚀", "")

# -------- Render --------
def colcfg(df):
    cfg={}
    def add(k, col):
        if k in df.columns: cfg[k]=col
    add("LDY_RANK",  st.column_config.NumberColumn("RANK", format="%d"))
    add("통과",       st.column_config.TextColumn(" "))
    add("시장",       st.column_config.TextColumn("시장"))
    add("종목명",     st.column_config.TextColumn("종목명"))
    add("종목코드",   st.column_config.TextColumn("종목코드"))
    add("LDY_SCORE", st.column_config.NumberColumn("LDY_SCORE", format="%.1f"))
    add("종가",        st.column_config.NumberColumn("종가", format="%,d"))
    add("추천매수가",  st.column_config.NumberColumn("추천매수가", format="%,d"))
    add("손절가",      st.column_config.NumberColumn("손절가", format="%,d"))
    add("추천매도가1", st.column_config.NumberColumn("목표1", format="%,d"))
    add("추천매도가2", st.column_config.NumberColumn("목표2", format="%,d"))
    add("RR1",       st.column_config.NumberColumn("RR1", format="%.2f"))
    add("Now%",      st.column_config.NumberColumn("엔트리근접(%)", format="%.2f"))
    add("T1여유%",    st.column_config.NumberColumn("목표1여유(%)", format="%.2f"))
    add("SL여유%",    st.column_config.NumberColumn("손절여유(%)", format="%.2f"))
    add("ERS",       st.column_config.NumberColumn("ERS", format="%.0f"))
    add("거래대금(억원)", st.column_config.NumberColumn("거래대금(억원)", format="%,.0f"))
    add("시가총액(억원)", st.column_config.NumberColumn("시가총액(억원)", format="%,.0f"))
    add("RSI14",     st.column_config.NumberColumn("RSI14", format="%.1f"))
    add("乖離%",      st.column_config.NumberColumn("乖離%", format="%.2f"))
    add("MACD_slope",st.column_config.NumberColumn("MACD_slope", format="%.5f"))
    add("Vol_Z",     st.column_config.NumberColumn("Vol_Z", format="%.2f"))
    add("ret_5d_%",  st.column_config.NumberColumn("5일수익%", format="%.2f"))
    add("ret_10d_%", st.column_config.NumberColumn("10일수익%", format="%.2f"))
    add("EBS",       st.column_config.NumberColumn("EBS", format="%d"))
    add("근거",       st.column_config.TextColumn("근거"))
    return cfg

st.subheader("오늘의 GLOBAL TOP 10", anchor=False)
cols = ["LDY_RANK","통과","시장","종목명","종목코드","LDY_SCORE",
        "종가","추천매수가","손절가","추천매도가1","추천매도가2",
        "RR1","Now%","T1여유%","SL여유%","ERS",
        "거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS","근거"]
for c in cols:
    if c not in top10.columns: top10[c]=np.nan

# 형식 안정화
fmt = top10.copy()
int_cols = ["LDY_RANK","종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]
for c in int_cols:
    if c in fmt.columns:
        fmt[c] = pd.to_numeric(fmt[c], errors="coerce").round(0).astype("Int64")
float_cols = ["LDY_SCORE","RR1","Now%","T1여유%","SL여유%","거래대금(억원)","시가총액(억원)",
              "RSI14","乖離%","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"]
for c in float_cols:
    if c in fmt.columns:
        fmt[c] = pd.to_numeric(fmt[c], errors="coerce")

st.data_editor(
    fmt[cols],
    key="tbl_global_top10",
    width="stretch", height=520,
    hide_index=True, disabled=True, num_rows="fixed",
    column_config=colcfg(fmt),
)

st.download_button(
    "📥 GLOBAL TOP 10 (CSV)",
    data=top10[cols].to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_global_top10.csv",
    mime="text/csv",
    key="dl_global_top10",
)

# 전체 랭킹 CSV 다운만 제공(화면표시는 Top10만)
st.download_button(
    "📥 전체 랭킹 (CSV, 최대 2,000행)",
    data=scored.sort_values("LDY_SCORE", ascending=False).head(2000)[cols].to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_global_rank_full.csv",
    mime="text/csv",
    key="dl_global_full",
)

st.caption("※ 품질게이트: EBS≥4 + 유동성 하드컷 기본 / 후보가 부족하면 자동 완화(EBS≥3, 완화 유동성)")
