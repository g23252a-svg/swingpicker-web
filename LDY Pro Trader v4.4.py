# -*- coding: utf-8 -*-
"""
LDY Pro Trader v4.4 — Global Rank (Single Composite Score + Router + Explain + P_hit)
- 한 화면: '오늘의 GLOBAL TOP 10'만 고정 노출 (가중치/슬라이더 없음)
- 모든 지표 → 단일 점수 LDY_SCORE(0~100), LDY_RANK = 내림차순
- 전략 배지(ROUTE): BRK(돌파)/PULL(눌림)/TREND(추세)/MR(되돌림) 간단 라우팅
- 설명력: 컴포넌트별 기여도(점수)+패널티, WHY 문자열, P_hit(타격 확률 추정) 표시
- 유동성 하드컷: KOSPI≥200억, KOSDAQ≥100억 (후보 부족 시 자동 완화)
- 선택 로그 학습(없으면 자동 무시): data/trade_logs.csv(컬럼: code, hit(0/1), score(float))
"""

import os, io, math, json, requests, numpy as np, pandas as pd, streamlit as st
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
st.set_page_config(page_title="LDY Pro Trader v4.4 — Global Rank", layout="wide")
st.title("🏆 LDY Pro Trader v4.4 — Global Rank")
st.caption("모든 지표를 단일 점수로 종합 → 1위가 가장 유망한 종목 (고정 Top 10, 가중치 UI 없음)")

# -------- Constants --------
RAW_URL   = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_RAW = "data/recommend_latest.csv"
CODES_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/krx_codes.csv"
LOCAL_MAP = "data/krx_codes.csv"

PASS_EBS = 4                 # 품질 게이트
MIN_TURN_KOSPI = 200.0       # 유동성 하드컷
MIN_TURN_KOSDAQ = 100.0
MIN_TURN_DEFAULT = 100.0

# 고정 가중치(합=1.0) — UI 조정 없음
W_RR   = 0.25  # 보상대비위험 (RR1)
W_T1   = 0.18  # 목표1 여유
W_SL   = 0.12  # 손절 여유
W_NEAR = 0.12  # 현재가-추천가 근접
W_MOM  = 0.10  # 모멘텀(ERS+MACD_slope+RSI 중심)
W_LIQ  = 0.13  # 유동성(거래대금 퍼센타일)
W_TEC  = 0.10  # 기술균형(VolZ 스윗스팟 +乖離 안정)

# 페널티(점수에서 직접 차감)
P_OVERHEAT_5D = 6.0   # 5일 과열
P_OVERHEAT_10D= 6.0   # 10일 과열
P_RSI_OUT     = 4.0   # RSI 45~65 이탈
P_MACD_NEG    = 4.0   # MACD 기울기 음수
P_NEAR_FAR    = 4.0   # 엔트리 괴리 과다
P_LIQ_LOW     = 4.0   # 유동성 하위권
P_VOL_SPIKE   = 2.0   # VolZ 스파이크

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

def nz_num(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")

def ensure_turnover(df: pd.DataFrame) -> pd.DataFrame:
    if "거래대금(억원)" not in df.columns:
        base = None
        if "거래대금(원)" in df.columns:
            base = nz_num(df["거래대금(원)"])
        elif all(c in df.columns for c in ["거래량","종가"]):
            base = nz_num(df["거래량"]) * nz_num(df["종가"])
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
    # 숫자 캐스팅
    for c in ["시가","고가","저가","종가","거래량","거래대금(원)","시가총액(원)","거래대금(억원)","시가총액(억원)",
              "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS",
              "추천매수가","추천매도가1","추천매도가2","손절가"]:
        if c in df.columns:
            df[c] = nz_num(df[c])
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
                    try: nm = stock.get_market_ticker_name(t)
                    except Exception: nm = None
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

# -------- 로드 --------
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
latest = df.sort_values(["종목코드","날짜"]).groupby("종목코드").tail(1) if "날짜" in df.columns else df.copy()

# -------- 하드 유동성 컷 --------
def liquidity_gate(x_turn: pd.Series, market: pd.Series) -> pd.Series:
    min_map = {"KOSPI": MIN_TURN_KOSPI, "KOSDAQ": MIN_TURN_KOSDAQ}
    mins = market.map(min_map).fillna(MIN_TURN_DEFAULT)
    return nz_num(x_turn) >= mins

# -------- 정규화 유틸 --------
def cap_q(s: pd.Series, q=90, floor=1.0):
    s = nz_num(s)
    if s.notna().sum()==0: return floor
    c = np.nanpercentile(s, q)
    if not np.isfinite(c) or c<=0: c=floor
    return float(max(c, floor))

def pct_norm_pos(s: pd.Series, q=90, floor=1.0):
    s = nz_num(s).clip(lower=0)
    cap = cap_q(s, q=q, floor=floor)
    return np.clip(s / cap, 0, 1)

def inv_dist_norm(dist: pd.Series, cap):
    d = nz_num(dist)
    return np.clip(1 - (d / cap), 0, 1)

# -------- 전략 라우터(간이) --------
def route_tag(row) -> str:
    rsi = row.get("RSI14", np.nan)
    slope = row.get("MACD_slope", np.nan)
    kairi = row.get("乖離%", np.nan)
    r5 = row.get("ret_5d_%", np.nan)
    near = row.get("Now%", np.nan)

    # 기준: 돌파(BRK) / 눌림(PULL) / 추세(TREND) / 되돌림(MR)
    if pd.notna(r5) and pd.notna(near) and pd.notna(slope):
        if (r5 >= 3) and (near <= 0.7) and (slope > 0) and (abs(kairi) <= 6):
            return "🔼 BRK"
    if pd.notna(rsi) and pd.notna(near):
        if (45 <= rsi <= 60) and (near <= 1.0) and (abs(kairi) <= 5):
            return "↩️ PULL"
    if pd.notna(slope) and slope > 0 and pd.notna(r5) and r5 > 0 and abs(kairi) <= 7:
        return "📈 TREND"
    if pd.notna(rsi) and (rsi >= 67 or rsi <= 40):
        return "🔁 MR"
    return "—"

# -------- P_hit 교정(로그 있으면 이용) --------
@st.cache_data(ttl=300)
def load_trade_logs(path="data/trade_logs.csv"):
    if os.path.exists(path):
        try:
            d = pd.read_csv(path)
            # 기대 컬럼: code, hit(0/1), score(float)
            if {"code","hit","score"}.issubset(d.columns):
                d["hit"] = nz_num(d["hit"]).clip(0,1)
                d["score"] = nz_num(d["score"])
                return d.dropna(subset=["score","hit"])
        except Exception:
            return None
    return None

def calibrate_p_hit(raw_score: pd.Series, ers_norm: pd.Series) -> pd.Series:
    # 베이스: 0~100 점수 → 0~1 로짓형 매핑(완만)
    x = nz_num(raw_score).fillna(0)/100.0
    base = 1/(1 + np.exp(-4*(x-0.55)))   # 중심 55점 부근
    # ERS 보정(품질 시그널)
    e = np.clip(nz_num(ers_norm), 0, 1).fillna(0)
    base = np.clip(0.85*base + 0.15*e, 0, 1)

    logs = load_trade_logs()
    if logs is None or logs.empty:
        return base

    # 단순 플랫-빈 교정: 점수 10분위 → 실측 hit 비율로 재매핑
    df = pd.DataFrame({"score": x, "base": base})
    # 로그를 10분위로 집계
    logs = logs.copy()
    logs["bin"] = pd.qcut(logs["score"], q=10, duplicates="drop")
    by = logs.groupby("bin", as_index=False)["hit"].mean().rename(columns={"hit":"obs"})
    # 현재 x도 같은 방식으로 구간화
    try:
        ref = pd.qcut(x, q=len(by), duplicates="drop")
        ref = pd.DataFrame({"bin": ref})
        ref = ref.merge(by, on="bin", how="left")
        cal = ref["obs"].fillna(base)
        return cal.clip(0,1)
    except Exception:
        return base

# -------- Composite Score --------
def build_global_score(lat: pd.DataFrame) -> pd.DataFrame:
    x = lat.copy()

    # 필수 수치
    for c in ["종가","추천매수가","손절가","추천매도가1","거래대금(억원)","시가총액(억원)","RSI14","MACD_slope",
              "Vol_Z","乖離%","ret_5d_%","ret_10d_%","EBS","MACD_hist"]:
        if c not in x.columns: x[c]=np.nan

    close = nz_num(x["종가"])
    entry = nz_num(x["추천매수가"])
    stop  = nz_num(x["손절가"])
    t1    = nz_num(x["추천매도가1"])
    turn  = nz_num(x["거래대금(억원)"])
    rsi   = nz_num(x["RSI14"])
    slope = nz_num(x["MACD_slope"])
    volz  = nz_num(x["Vol_Z"])
    kairi = nz_num(x["乖離%"])
    r5    = nz_num(x["ret_5d_%"])
    r10   = nz_num(x["ret_10d_%"])
    ebs   = nz_num(x["EBS"]).fillna(0)

    # RR1, 여유, 근접
    rr_den = (entry - stop)
    rr1 = (t1 - entry) / rr_den.replace(0, np.nan)
    rr1 = rr1.mask(entry.isna() | stop.isna() | t1.isna())
    now_gap = (close - entry).abs() / entry * 100
    t1_room = (t1 - close) / close * 100
    sl_room = (close - stop) / close * 100

    # 정규화
    rr_norm   = pct_norm_pos(rr1, q=90, floor=1.0)
    t1_norm   = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1)
    sl_norm   = np.clip(sl_room / cap_q(sl_room, q=90, floor=3.0), 0, 1)
    near_norm = inv_dist_norm(now_gap, cap=cap_q(now_gap, q=75, floor=1.0))
    ers_bits = (ebs>=PASS_EBS).astype(int) + (slope>0).astype(int) + ((rsi>=45)&(rsi<=65)).astype(int)
    ers_norm = np.clip(ers_bits/3.0, 0, 1)
    slope_pos_norm = pct_norm_pos(slope, q=90, floor=1.0)
    rsi_center = 1 - np.minimum((rsi-55).abs()/10, 1)           # 55에 가까울수록 1 (±10)
    rsi_center = rsi_center.clip(0,1).fillna(0)
    mom_norm = np.clip(0.5*ers_norm + 0.3*slope_pos_norm + 0.2*rsi_center, 0, 1)

    # 유동성: 거래대금 퍼센타일 스케일
    if turn.notna().any():
        lo = np.nanpercentile(turn, 30) if np.isfinite(np.nanpercentile(turn.dropna(), 30)) else np.nanmin(turn)
        hi = np.nanpercentile(turn, 90) if np.isfinite(np.nanpercentile(turn.dropna(), 90)) else np.nanmax(turn)
        span = max(hi - lo, 1e-9)
        liq_norm = np.clip((turn - lo) / span, 0, 1)
    else:
        liq_norm = pd.Series(0.0, index=x.index)

    # 기술 균형: VolZ≈1 + |乖離| 작을수록 좋음
    vol_sweet = 1 - np.minimum((volz - 1).abs()/3, 1)
    vol_sweet = vol_sweet.clip(0,1).fillna(0)
    kairi_norm = 1 - np.minimum(kairi.abs()/cap_q(kairi.abs(), q=80, floor=3.0), 1)
    kairi_norm = kairi_norm.clip(0,1).fillna(0)
    tec_norm = np.clip(0.6*vol_sweet + 0.4*kairi_norm, 0, 1)

    # 가중합(0~100)
    c_rr   = 100*(W_RR*rr_norm)
    c_t1   = 100*(W_T1*t1_norm)
    c_sl   = 100*(W_SL*sl_norm)
    c_near = 100*(W_NEAR*near_norm)
    c_mom  = 100*(W_MOM*mom_norm)
    c_liq  = 100*(W_LIQ*liq_norm)
    c_tec  = 100*(W_TEC*tec_norm)
    base_score = c_rr + c_t1 + c_sl + c_near + c_mom + c_liq + c_tec

    # 페널티
    pen = pd.Series(0.0, index=x.index)
    pen += P_OVERHEAT_5D * np.clip((r5 - 10)/10, 0, 1)
    pen += P_OVERHEAT_10D* np.clip((r10 - 20)/20, 0, 1)
    pen += P_RSI_OUT * ((rsi < 45) | (rsi > 65)).astype(float)
    pen += P_MACD_NEG * (slope < 0).astype(float)
    near_cap = cap_q(now_gap, q=75, floor=1.0)
    pen += P_NEAR_FAR * np.clip((now_gap - near_cap)/near_cap, 0, 1)
    if turn.notna().any():
        p20 = np.nanpercentile(turn.dropna(), 20) if turn.dropna().size else -np.inf
        pen += P_LIQ_LOW * (turn < p20).astype(float)
    pen += P_VOL_SPIKE * (volz > 3).astype(float)

    score = np.clip(base_score - pen, 0, 100)

    # 결과 합치기
    x["RR1"]        = rr1
    x["Now%"]       = now_gap
    x["T1여유%"]     = t1_room
    x["SL여유%"]     = sl_room
    x["ERS"]        = ers_bits.astype(float)
    x["LDY_SCORE"]  = score.round(1)
    x["_cRR"] = c_rr.round(1); x["_cT1"] = c_t1.round(1); x["_cSL"] = c_sl.round(1)
    x["_cNEAR"] = c_near.round(1); x["_cMOM"] = c_mom.round(1); x["_cLIQ"] = c_liq.round(1); x["_cTEC"] = c_tec.round(1)
    x["_PEN"] = pen.round(1)

    # 라우터
    # Now% 등은 방금 만든 컬럼 사용
    x["ROUTE"] = (x.apply(route_tag, axis=1) if len(x) else "—")

    # 유동성 게이트
    x["_GATE_OK"] = liquidity_gate(x["거래대금(억원)"], x["시장"]).fillna(False)

    # 랭크
    x = x.sort_values("LDY_SCORE", ascending=False, na_position="last")
    x["LDY_RANK"] = range(1, len(x)+1)

    # WHY 문자열(요약)
    x["WHY"] = (
        "RR+" + x["_cRR"].fillna(0).astype(str) + ", "
        "T1+" + x["_cT1"].fillna(0).astype(str) + ", "
        "SL+" + x["_cSL"].fillna(0).astype(str) + ", "
        "NEAR+" + x["_cNEAR"].fillna(0).astype(str) + ", "
        "MOM+" + x["_cMOM"].fillna(0).astype(str) + ", "
        "LIQ+" + x["_cLIQ"].fillna(0).astype(str) + ", "
        "TEC+" + x["_cTEC"].fillna(0).astype(str) + ", "
        "PEN−" + x["_PEN"].fillna(0).astype(str)
    )
    return x

scored = build_global_score(latest)

# 랭킹 포함 조건: 추천/손절/목표1/종가 있어야
scored = scored.dropna(subset=["추천매수가","손절가","추천매도가1","종가"])

# 품질게이트 & 유동성
base = scored[(nz_num(scored.get("EBS")) >= PASS_EBS) & (scored["_GATE_OK"])].copy()

# 후보 부족 시 자동 완화
if len(base) < 10:
    fb = scored[nz_num(scored.get("EBS")) >= (PASS_EBS-1)].copy()
    MIN_KOSPI_F, MIN_KOSDAQ_F = 150.0, 80.0
    mm = {"KOSPI": MIN_KOSPI_F, "KOSDAQ": MIN_KOSDAQ_F}
    fb["_min_turn"] = fb["시장"].map(mm).fillna(MIN_TURN_DEFAULT)
    fb = fb[ nz_num(fb["거래대금(억원)"]) >= fb["_min_turn"] ]
    base_codes = set(base["종목코드"])
    fill = fb[~fb["종목코드"].isin(base_codes)]
    base = pd.concat([base, fill]).sort_values("LDY_SCORE", ascending=False).head(50)

# 최종 Top10
top10 = base.sort_values("LDY_SCORE", ascending=False, na_position="last").head(10).copy()
top10["통과"] = np.where(nz_num(top10.get("EBS")) >= PASS_EBS, "🚀", "")

# P_hit(타격확률) 추정
ers_norm_tmp = np.clip(nz_num(top10["ERS"])/3.0, 0, 1)
top10["P_hit"] = (calibrate_p_hit(top10["LDY_SCORE"], ers_norm_tmp) * 100).round(1)

# -------- Render --------
def colcfg(df):
    cfg={}
    def add(k, col):
        if k in df.columns: cfg[k]=col
    add("LDY_RANK",  st.column_config.NumberColumn("RANK", format="%d"))
    add("통과",       st.column_config.TextColumn(" "))
    add("ROUTE",     st.column_config.TextColumn("ROUTE"))
    add("시장",       st.column_config.TextColumn("시장"))
    add("종목명",     st.column_config.TextColumn("종목명"))
    add("종목코드",   st.column_config.TextColumn("종목코드"))
    add("LDY_SCORE", st.column_config.NumberColumn("LDY_SCORE", format="%.1f"))
    add("P_hit",     st.column_config.NumberColumn("P_hit(%)", format="%.1f"))
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
    add("WHY",       st.column_config.TextColumn("WHY(기여도 요약)"))
    # 내부 기여도 열은 다운로드에만 포함(테이블은 간결하게)
    return cfg

st.subheader("오늘의 GLOBAL TOP 10", anchor=False)

cols_show = [
    "LDY_RANK","통과","ROUTE","시장","종목명","종목코드",
    "LDY_SCORE","P_hit",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "RR1","Now%","T1여유%","SL여유%","ERS",
    "거래대금(억원)","시가총액(억원)",
    "RSI14","乖離%","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS",
    "WHY"
]
for c in cols_show:
    if c not in top10.columns: top10[c]=np.nan

# 형식 안정화
fmt = top10.copy()
int_cols = ["LDY_RANK","종가","추천매수가","손절가","추천매도가1","추천매도가2","EBS"]
for c in int_cols:
    if c in fmt.columns:
        fmt[c] = nz_num(fmt[c]).round(0).astype("Int64")
float_cols = ["LDY_SCORE","P_hit","RR1","Now%","T1여유%","SL여유%","거래대금(억원)","시가총액(억원)",
              "RSI14","乖離%","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"]
for c in float_cols:
    if c in fmt.columns:
        fmt[c] = nz_num(fmt[c])

st.data_editor(
    fmt[cols_show],
    key="tbl_global_top10_v44",
    width="stretch", height=560,
    hide_index=True, disabled=True, num_rows="fixed",
    column_config=colcfg(fmt),
)

# 다운로드 (설명용 내부 기여도 포함)
cols_download = cols_show + ["_cRR","_cT1","_cSL","_cNEAR","_cMOM","_cLIQ","_cTEC","_PEN"]
for c in cols_download:
    if c not in top10.columns: top10[c] = np.nan

st.download_button(
    "📥 GLOBAL TOP 10 (CSV)",
    data=top10[cols_download].to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_global_top10.csv",
    mime="text/csv",
    key="dl_global_top10_v44",
)

# ===== [FIX] 전체 랭킹 CSV: 누락 컬럼 자동 보강 후 내보내기 =====
def ensure_all_columns(df: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in wanted if c not in out.columns]
    # 디버그 겸 화면에 경고
    if missing:
        st.warning("⚠️ 내보내기 누락 컬럼 자동 보강: " + ", ".join(missing))
    for c in missing:
        out[c] = np.nan
    # 정렬된 고정 컬럼 순서로 반환
    return out[wanted]

# Top10에서 쓰던 고정 표시 컬럼(순서 그대로 재사용)
full_cols = [
    "LDY_RANK","통과","시장","종목명","종목코드","LDY_SCORE",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "RR1","Now%","T1여유%","SL여유%","ERS",
    "거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_slope",
    "Vol_Z","ret_5d_%","ret_10d_%","EBS","근거"
]

# 전체 랭킹 데이타 준비
export_df = scored.sort_values("LDY_SCORE", ascending=False, na_position="last").copy()

# '통과' 칼럼이 없는 경우(Top10에서만 만들었을 수 있음) → 전체에도 생성
if "통과" not in export_df.columns:
    export_df["통과"] = np.where(
        pd.to_numeric(export_df.get("EBS"), errors="coerce") >= PASS_EBS, "🚀", ""
    )

# 컬럼 자동 보강 후, 상위 N행만 내보내기
export_ready = ensure_all_columns(export_df, full_cols).head(2000)

st.download_button(
    "📥 전체 랭킹 (CSV, 최대 2,000행)",
    data=export_ready.to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_global_rank_full.csv",
    mime="text/csv",
    key="dl_global_full",
)
# ===============================================================

