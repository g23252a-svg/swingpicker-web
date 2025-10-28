# -*- coding: utf-8 -*-
"""
LDY Pro Trader v3.2.0 (Auto Update Viewer, self-enrich)
- GitHub raw CSV를 우선 로드, 실패 시 로컬 data/recommend_latest.csv 폴백
- CSV가 원시 OHLCV만 있어도, 이 화면에서 RSI/MACD/ATR/MA20/VolZ/수익률 계산 → EBS/추천가 생성
- EBS 컬럼이 없거나 전부 NaN이면 '초입 후보만' 필터 자동 해제
- 거래대금(원)만 있으면 '거래대금(억원)'으로 자동 변환
"""

import os, io, math, requests
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------- 기본 설정 -------------------------
st.set_page_config(page_title="LDY Pro Trader v3.2.0 (Auto Update)", layout="wide")
st.title("📈 LDY Pro Trader v3.2.0 (Auto Update)")
st.caption("매일 장마감 후 자동 업데이트되는 스윙 추천 종목 리스트 | Made by LDY")

RAW_URL = "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
LOCAL_PATH = "data/recommend_latest.csv"
PASS_SCORE = 4

# ------------------------- 로딩 -------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_remote_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=300, show_spinner=False)
def load_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def info_src(df: pd.DataFrame, src_text: str):
    st.info(f"상태\n✅ 데이터 로드: {src_text}\n\n{RAW_URL if 'remote' in src_text else LOCAL_PATH}")
    st.success(f"📅 추천 기준(표시 시각): {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M')} · 원시 행수: {len(df):,}")

# ------------------------- 지표 계산 유틸 -------------------------
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi14(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def macd_features(close: pd.Series):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd_line - signal
    slope = hist.diff()  # 히스토그램 기울기
    return hist, slope

def atr14(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ------------------------- 스키마 정리 -------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼 이름 표준화(가능한 매핑)
    colmap = {
        "Date": "날짜", "date": "날짜",
        "Code": "종목코드", "티커": "종목코드", "ticker": "종목코드",
        "Name": "종목명", "name": "종목명",
        "Open": "시가", "High": "고가", "Low": "저가", "Close": "종가",
        "Volume": "거래량",
        "거래대금": "거래대금(원)",
        "시가총액": "시가총액(원)"
    }
    for k, v in colmap.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # 거래대금(억원) 만들기
    if "거래대금(억원)" not in df.columns:
        if "거래대금(원)" in df.columns:
            df["거래대금(억원)"] = (pd.to_numeric(df["거래대금(원)"], errors="coerce") / 1e8).round(2)
        elif "거래대금" in df.columns:
            df["거래대금(억원)"] = (pd.to_numeric(df["거래대금"], errors="coerce") / 1e8).round(2)

    # 숫자 캐스팅
    for c in ["시가","고가","저가","종가","거래량","거래대금(억원)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 날짜 처리
    if "날짜" in df.columns:
        try:
            df["날짜"] = pd.to_datetime(df["날짜"])
        except Exception:
            pass

    # 필수 기본 컬럼 보강
    for c in ["시장","종목코드","종목명"]:
        if c not in df.columns:
            df[c] = None

    return df

# ------------------------- 원시 OHLCV → 스코어/추천가 생성 -------------------------
@st.cache_data(ttl=300, show_spinner=True)
def enrich_from_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    # 최소 필수: 날짜/코드/시가/고가/저가/종가/거래량, (거래대금(억원) 권장)
    must_cols = {"종목코드", "날짜", "시가", "고가", "저가", "종가"}
    if not must_cols.issubset(set(raw.columns)):
        return raw  # 못 만들면 원본 반환(뷰어는 보호 로직으로 표시)
    g = raw.sort_values(["종목코드","날짜"]).groupby("종목코드", group_keys=False)

    def _feat(group: pd.DataFrame):
        group = group.copy()
        # 지표
        group["MA20"] = group["종가"].rolling(20).mean()
        group["ATR14"] = atr14(group["고가"], group["저가"], group["종가"], 14)
        group["RSI14"] = rsi14(group["종가"], 14)
        hist, slope = macd_features(group["종가"])
        group["MACD_hist"] = hist
        group["MACD_slope"] = slope
        group["Vol_Z"] = (group["거래량"] - group["거래량"].rolling(20).mean()) / group["거래량"].rolling(20).std()
        group["乖離%"] = (group["종가"] / group["MA20"] - 1.0) * 100.0
        group["ret_5d_%"] = (group["종가"] / group["종가"].shift(5) - 1.0) * 100.0
        group["ret_10d_%"] = (group["종가"] / group["종가"].shift(10) - 1.0) * 100.0

        # EBS 7점제
        last = group.iloc[-1:].copy()
        conds = []
        ebs = 0
        # 1) RSI 45~65
        c1 = 45 <= (last["RSI14"].iloc[0] if not last["RSI14"].isna().iloc[0] else -999) <= 65
        ebs += int(c1);  conds.append("RSI 45~65" if c1 else "")
        # 2) MACD slope > 0
        c2 = (last["MACD_slope"].iloc[0] if not last["MACD_slope"].isna().iloc[0] else -999) > 0
        ebs += int(c2);  conds.append("MACD↑" if c2 else "")
        # 3) 종가가 MA20 -1% ~ +4%
        close = last["종가"].iloc[0]; ma20 = last["MA20"].iloc[0]
        c3 = (not np.isnan(ma20)) and (0.99*ma20 <= close <= 1.04*ma20)
        ebs += int(c3);  conds.append("MA20±4%" if c3 else "")
        # 4) Vol_Z > 1.2
        c4 = (last["Vol_Z"].iloc[0] if not last["Vol_Z"].isna().iloc[0] else -999) > 1.2
        ebs += int(c4);  conds.append("VolZ>1.2" if c4 else "")
        # 5) MA20 > MA60? (MA60 없으면 MA20 기울기>0로 대체)
        ma20_slope = last["MA20"].iloc[0] - group["MA20"].iloc[-2] if len(group) >= 2 else np.nan
        c5 = (not np.isnan(ma20_slope)) and (ma20_slope > 0)
        ebs += int(c5);  conds.append("MA20↑" if c5 else "")
        # 6) MACD_hist > 0
        c6 = (last["MACD_hist"].iloc[0] if not last["MACD_hist"].isna().iloc[0] else -999) > 0
        ebs += int(c6);  conds.append("MACD>0" if c6 else "")
        # 7) 5일 수익률 < 10% (과열 방지)
        r5 = last["ret_5d_%"].iloc[0]
        c7 = (not np.isnan(r5)) and (r5 < 10)
        ebs += int(c7);  conds.append("5d<10%" if c7 else "")

        last["EBS"] = ebs
        last["근거"] = " / ".join([c for c in conds if c])

        # 추천가 (보수적): 엔트리=MA20±0.5*ATR 범위 내로 스냅, T1=+1*ATR, T2=+1.8*ATR, 손절=-1.2*ATR
        atr = last["ATR14"].iloc[0]
        if np.isnan(atr) or np.isnan(ma20) or np.isnan(close) or atr <= 0:
            entry, t1, t2, stp = np.nan, np.nan, np.nan, np.nan
        else:
            band_low = ma20 - 0.5 * atr
            band_high = ma20 + 0.5 * atr
            entry = min(max(close, band_low), band_high)
            t1 = entry + 1.0 * atr
            t2 = entry + 1.8 * atr
            stp = entry - 1.2 * atr

        last["추천매수가"] = round(entry, 2) if not np.isnan(entry) else np.nan
        last["추천매도가1"] = round(t1, 2) if not np.isnan(t1) else np.nan
        last["추천매도가2"] = round(t2, 2) if not np.isnan(t2) else np.nan
        last["손절가"] = round(stp, 2) if not np.isnan(stp) else np.nan

        return last

    out = g.apply(_feat).reset_index(drop=True)

    # 거래대금(억원) 없으면 마지막날 기준 그룹합/평균 등으로 채우기(가능한 경우)
    if "거래대금(억원)" not in out.columns and "거래대금(원)" in raw.columns:
        tv_last = raw.sort_values(["종목코드","날짜"]).groupby("종목코드").tail(1)[["종목코드","거래대금(원)"]]
        tv_last["거래대금(억원)"] = (tv_last["거래대금(원)"]/1e8).round(2)
        out = out.merge(tv_last[["종목코드","거래대금(억원)"]], on="종목코드", how="left")

    # 시총(억원) 없으면 NaN 유지(collector에서 채우는 걸 권장)
    if "시가총액(억원)" not in out.columns:
        out["시가총액(억원)"] = np.nan
    # 시장 없으면 ALL로
    if "시장" not in out.columns:
        out["시장"] = "ALL"

    return out

# ------------------------- 데이터 로드 & 정규화 -------------------------
try:
    df_raw = load_remote_csv(RAW_URL)
    info_src(df_raw, "remote")
except Exception:
    if os.path.exists(LOCAL_PATH):
        df_raw = load_local_csv(LOCAL_PATH)
        info_src(df_raw, "local")
    else:
        st.error("❌ CSV를 찾을 수 없습니다. collector가 data/recommend_latest.csv를 올렸는지 확인하세요.")
        st.stop()

df_raw = normalize_columns(df_raw)

# ------------------------- 스키마 감지: EBS/추천가가 있나? -------------------------
has_ebs = "EBS" in df_raw.columns and df_raw["EBS"].notna().any()
has_reco = all(c in df_raw.columns for c in ["추천매수가","추천매도가1","추천매도가2","손절가"]) and df_raw[["추천매수가","추천매도가1","추천매도가2","손절가"]].notna().any().any()

if has_ebs and has_reco:
    df = df_raw.copy()
else:
    # 원시 OHLCV에서 스스로 생성
    with st.status("🧮 원시 OHLCV → 지표/점수/추천가 생성 중...", expanded=False):
        df = enrich_from_ohlcv(df_raw)

# 당일(또는 최신일) 한 줄 요약 뷰 만들기
if "날짜" in df.columns:
    latest_by_code = df.sort_values(["종목코드", "날짜"]).groupby("종목코드").tail(1).copy()
else:
    latest_by_code = df.copy()

# 최종 안전 캐스팅
for c in ["종가","거래대금(억원)","시가총액(억원)","RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%","EBS","추천매수가","추천매도가1","추천매도가2","손절가"]:
    if c in latest_by_code.columns:
        latest_by_code[c] = pd.to_numeric(latest_by_code[c], errors="coerce")

# ------------------------- UI: 필터/정렬 -------------------------
with st.expander("🔍 보기/필터", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
    # EBS 모두 NaN이면 자동 해제
    default_entry = True
    if "EBS" not in latest_by_code.columns or latest_by_code["EBS"].notna().sum() == 0:
        default_entry = False
        st.warning("EBS 점수가 없어 ‘🚀 초입 후보만’ 필터를 자동 해제합니다. (원시 OHLCV에서 계산 실패 또는 데이터 부족)")
    with c1:
        only_entry = st.checkbox("🚀 초입 후보만 (EBS≥4)", value=default_entry)
    with c2:
        min_turn = st.slider("최소 거래대금(억원)", 0, 5000, 50, step=10)
    with c3:
        sort_key = st.selectbox(
            "정렬",
            ["EBS▼", "거래대금▼", "시가총액▼", "RSI▲", "RSI▼", "종가▲", "종가▼"],
            index=0 if "EBS" in latest_by_code.columns else 1
        )
    with c4:
        topn = st.slider("표시 수(Top N)", 10, 500, 200, step=10)
    with c5:
        q_text = st.text_input("🔎 종목명/코드 검색", value="", placeholder="예: 삼성전자 또는 005930")

view = latest_by_code.copy()

# 종목명 없으면 코드만 표시(이름은 collector에서 맵파일 생성 권장)
if "종목명" not in view.columns or view["종목명"].isna().all():
    view["종목명"] = "(이름없음)"

# 필터들
if only_entry and "EBS" in view.columns:
    view = view[view["EBS"] >= PASS_SCORE]
if "거래대금(억원)" in view.columns:
    view = view[view["거래대금(억원)"] >= float(min_turn)]

if q_text:
    q = q_text.strip().lower()
    name_hit = view["종목명"].fillna("").astype(str).str.lower().str.contains(q, na=False)
    code_hit = view["종목코드"].fillna("").astype(str).str.contains(q, na=False)
    view = view[name_hit | code_hit]

# 정렬
if sort_key == "EBS▼" and "EBS" in view.columns:
    view = view.sort_values(["EBS","거래대금(억원)"], ascending=[False, False])
elif sort_key == "거래대금▼" and "거래대금(억원)" in view.columns:
    view = view.sort_values("거래대금(억원)", ascending=False)
elif sort_key == "시가총액▼" and "시가총액(억원)" in view.columns:
    view = view.sort_values("시가총액(억원)", ascending=False, na_position="last")
elif sort_key == "RSI▲" and "RSI14" in view.columns:
    view = view.sort_values("RSI14", ascending=True, na_position="last")
elif sort_key == "RSI▼" and "RSI14" in view.columns:
    view = view.sort_values("RSI14", ascending=False, na_position="last")
elif sort_key == "종가▲" and "종가" in view.columns:
    view = view.sort_values("종가", ascending=True, na_position="last")
elif sort_key == "종가▼" and "종가" in view.columns:
    view = view.sort_values("종가", ascending=False, na_position="last")

# ------------------------- 표 출력 -------------------------
show_cols = [
    "통과","시장","종목명","종목코드",
    "종가","추천매수가","손절가","추천매도가1","추천매도가2",
    "거래대금(억원)","시가총액(억원)",
    "EBS","근거",
    "RSI14","乖離%","MACD_hist","MACD_slope","Vol_Z","ret_5d_%","ret_10d_%"
]
# 통과 표시
if "EBS" in view.columns:
    view["통과"] = np.where(view["EBS"] >= PASS_SCORE, "🚀", "")

# 누락 컬럼 채움
for c in show_cols:
    if c not in view.columns:
        view[c] = np.nan

st.write(f"📋 총 {len(latest_by_code):,}개 / 표시 {min(len(view), int(topn)):,}개")
st.dataframe(view[show_cols].head(int(topn)), width="stretch", height=640)

# ------------------------- 다운로드 -------------------------
st.download_button(
    "📥 현재 보기 다운로드 (CSV)",
    data=view[show_cols].head(int(topn)).to_csv(index=False, encoding="utf-8-sig"),
    file_name="ldy_entry_candidates.csv",
    mime="text/csv"
)

with st.expander("ℹ️ EBS 구성(급등 초입 로직)", expanded=False):
    st.markdown(
        """
- 기본 컷(collector 권장): 거래대금 ≥ **50억원**, 시가총액 ≥ **1,000억원**
- 점수(0~7):
  1) RSI 45~65  
  2) MACD 히스토그램 기울기 > 0  
  3) 종가가 MA20 근처(-1%~+4%)  
  4) 상대거래량(20일) > 1.2  
  5) MA20 상승(기울기 > 0)  
  6) MACD 히스토그램 > 0  
  7) 5일 수익률 < 10%(과열 방지)  
- **통과(🚀초입)**: EBS ≥ 4  
- 추천가: ATR/MA 기반 보수적 가이드 (엔트리 = MA20±0.5*ATR 범위 내 스냅, T1=+1*ATR, T2=+1.8*ATR, 손절=-1.2*ATR)
        """
    )
