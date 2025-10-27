import streamlit as st
import streamlit.components.v1 as components

# ✅ 페이지 설정은 반드시 첫 Streamlit 호출로
st.set_page_config(page_title="Swing Picker Web v3.0.2 FullSync", layout="wide")

# --- GA4 ---
GA_MEASUREMENT_ID = "G-3PLRGRT2RL"
GA_SCRIPT = f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_MEASUREMENT_ID}');
  window._gtagReady = true;
</script>
"""
st.markdown(GA_SCRIPT, unsafe_allow_html=True)

# ✅ 화면 살아있는지 즉시 표시(임시)
st.write("✅ App loaded")
# swing_picker_web_v3.0.2_fullsync.py
# ✅ exe 완전 동일 로직 + 캐시/재시도/딜레이/주말보정 + Streamlit UI 버전

import streamlit as st
import pandas as pd
import datetime as dt
import math, time, os, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pykrx import stock

# ---------------- 기본 설정 ----------------
DEFAULTS = {
    "MARKETS": ["KOSPI", "KOSDAQ"],
    "TOP_TURNOVER": 120,
    "TOP_N": 10,
    "LOOKBACK_DAYS": 63,
    "MAX_WORKERS": 6,
    "USE_YESTERDAY": True,
    "VOL_RATIO_MIN": 1.5,
    "RET5_MAX": 8.0,
    "RET10_MAX": 15.0,
    "USE_MA20_SUPPORT": True,
    "USE_CANDLE_BODY": True,
    "USE_RSI_MACD": False,
    "USE_GOLDEN_CROSS": False,
    "USE_RSI_REBOUND": False,
    "EXCLUDE_HARD_DROP": False,
    "HARD_DROP_5D": -10.0,
    "REQUEST_DELAY_SEC": 0.22,
    "FUTURE_TIMEOUT_SEC": 9.0,
    "GET_OHLCV_MAX_RETRY": 3,
    "CACHE_DIR": "./cache",
}

# ---------------- 유틸 ----------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def last_trading_day(base_dt: dt.datetime) -> dt.datetime:
    d = base_dt
    while d.weekday() >= 5:  # 5=토, 6=일
        d -= dt.timedelta(days=1)
    return d

# ---------------- 캐시 + 안정화된 데이터 요청 ----------------
def get_ohlcv(code: str, start: str, end: str) -> pd.DataFrame:
    ensure_dir(DEFAULTS["CACHE_DIR"])
    cache_file = os.path.join(DEFAULTS["CACHE_DIR"], f"{code}_{start}_{end}.csv")

    # 캐시 읽기
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            if not df.empty:
                df["종목코드"] = code
                return df
        except:
            pass

    # 재시도 + 백오프
    delay = DEFAULTS["REQUEST_DELAY_SEC"]
    for attempt in range(DEFAULTS["GET_OHLCV_MAX_RETRY"]):
        try:
            df = stock.get_market_ohlcv_by_date(start, end, code)
            time.sleep(delay)
            if df is not None and not df.empty:
                df["종목코드"] = code
                df.to_csv(cache_file, index=False, encoding="utf-8-sig")
                return df
        except Exception as e:
            time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame()

def get_top_turnover_stocks(end: str, markets, top_turnover: int) -> pd.DataFrame:
    all_df = []
    for m in markets:
        try:
            df = stock.get_market_ohlcv_by_ticker(end, market=m)
            df["시장"] = m
            all_df.append(df)
            time.sleep(DEFAULTS["REQUEST_DELAY_SEC"])
        except Exception as e:
            print(f"[WARN] turnover fetch fail {m}: {e}")
    if not all_df:
        return pd.DataFrame()
    df_all = pd.concat(all_df)
    if "거래대금" not in df_all.columns:
        df_all["거래대금"] = df_all["종가"] * df_all["거래량"]
    df_all["거래대금(억)"] = df_all["거래대금"] / 1e8
    return df_all.sort_values("거래대금(억)", ascending=False).head(int(top_turnover))

# ---------------- 보조지표 ----------------
def rsi_series(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-9)
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def macd_series(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

# ---------------- 분석 로직 ----------------
def analyze_stock(df, code, name,
                  vol_ratio_min, ret5_max, ret10_max,
                  use_ma20_support, use_candle_body, use_rsi_macd,
                  use_golden_cross, use_rsi_rebound, exclude_hard_drop, hard_drop_5d):
    if df is None or len(df) < 40:
        return None
    try:
        c = df["종가"].astype(float)
        o = df["시가"].astype(float)
        h = df["고가"].astype(float)
        l = df["저가"].astype(float)
        v = df["거래량"].astype(float)

        ret5 = c.pct_change(5).iloc[-1] * 100
        ret10 = c.pct_change(10).iloc[-1] * 100
        v20 = v.iloc[-20:].mean()
        vr = (v.iloc[-3:].mean() / v20) if v20 > 0 else 0.0

        if exclude_hard_drop and ret5 < hard_drop_5d:
            return None

        ma5 = c.rolling(5).mean()
        ma20 = c.rolling(20).mean()

        price_above_ma = c.iloc[-1] >= ma20.iloc[-1] * 0.98 if ma20.iloc[-1] > 0 else False
        ma20_gap = (c.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1] * 100 if ma20.iloc[-1] > 0 else 0

        body_today = abs(c.iloc[-1] - o.iloc[-1]) / max(o.iloc[-1], 1e-9)
        body_yday = abs(c.iloc[-2] - o.iloc[-2]) / max(o.iloc[-2], 1e-9)
        strong_body = (body_today >= 0.02) or (body_yday >= 0.02)

        rsi = rsi_series(c, 14).iloc[-1]
        macd, sig, hist = macd_series(c)
        macd_hist = hist.iloc[-1]

        recent_low = l.iloc[-10:].min()
        rebound = (c.iloc[-1] - recent_low) / max(recent_low, 1e-9)

        conds = [ret5 <= ret5_max, ret10 <= ret10_max, vr >= vol_ratio_min, rebound >= -0.02, price_above_ma]

        if use_ma20_support:
            last3_gap = ((c.iloc[-3:] - ma20.iloc[-3:]) / ma20.iloc[-3:]).dropna()
            if len(last3_gap) >= 3:
                conds.append((last3_gap > -0.03).sum() >= 2 and (last3_gap < 0.15).all())

        if use_candle_body:
            conds.append(strong_body)

        if use_rsi_macd:
            conds.append(rsi <= 70 and macd_hist > 0)

        if use_rsi_rebound:
            conds.append((35 <= rsi <= 70) and (macd_hist > 0))

        if use_golden_cross:
            if len(ma5) > 1 and len(ma20) > 1:
                golden = (ma5.iloc[-2] <= ma20.iloc[-2]) and (ma5.iloc[-1] > ma20.iloc[-1])
                conds.append(golden)

        if all(conds):
            last = c.iloc[-1]
            buy = round(last * 0.98)
            sell = round(last * 1.10)
            return {
                "종목명": name,
                "종목코드": code,
                "현재가": f"{last:,}",
                "추천매수가": f"{buy:,}",
                "추천매도가": f"{sell:,}",
                "5일수익률(%)": f"{ret5:.2f}",
                "10일수익률(%)": f"{ret10:.2f}",
                "거래량배수": f"{vr:.2f}",
                "MA20乖離(%)": f"{ma20_gap:.2f}",
                "RSI14": f"{rsi:.1f}",
                "MACD_hist": f"{macd_hist:.4f}",
            }
    except Exception:
        return None

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Swing Picker Web v3.0.2 FullSync", layout="wide")
st.title("📈 Swing Picker Web v3.0.2 (FullSync)")
st.caption("거래대금 + 기술지표 기반 자동 스윙 종목 추천 (exe 완전 동일 로직)")

# ---- 입력 파라미터 ----
col1, col2, col3 = st.columns(3)
top_turnover = col1.number_input("거래대금 상위 N", 50, 300, DEFAULTS["TOP_TURNOVER"])
top_n = col2.number_input("추천 종목 수", 5, 50, DEFAULTS["TOP_N"])
lookback = col3.number_input("조회일수 (LOOKBACK)", 30, 180, DEFAULTS["LOOKBACK_DAYS"])

use_yesterday = st.checkbox("전일 기준 데이터 사용", value=True)
vol_ratio_min = st.number_input("거래량배수 ≥", 0.5, 10.0, DEFAULTS["VOL_RATIO_MIN"])
ret5_max = st.number_input("5일 수익률 ≤ %", -50.0, 50.0, DEFAULTS["RET5_MAX"])
ret10_max = st.number_input("10일 수익률 ≤ %", -100.0, 100.0, DEFAULTS["RET10_MAX"])

opt_ma20 = st.checkbox("MA20 지지", value=DEFAULTS["USE_MA20_SUPPORT"])
opt_body = st.checkbox("캔들바디", value=DEFAULTS["USE_CANDLE_BODY"])
opt_rsi_macd = st.checkbox("RSI/MACD", value=DEFAULTS["USE_RSI_MACD"])
opt_gc = st.checkbox("골든크로스(5/20)", value=DEFAULTS["USE_GOLDEN_CROSS"])
opt_rsi_reb = st.checkbox("RSI 반등", value=DEFAULTS["USE_RSI_REBOUND"])
opt_drop = st.checkbox("급락 배제", value=DEFAULTS["EXCLUDE_HARD_DROP"])
drop_5d = st.number_input("급락 기준 (5일 수익률 < %)", -50.0, 0.0, DEFAULTS["HARD_DROP_5D"])

# ---- 실행 버튼 ----
if st.button("스캔 시작 🚀"):
    st.info("데이터 수집 및 분석 중... (약 1~3분 소요)")
    today = dt.datetime.now()
    end_dt = last_trading_day(today - dt.timedelta(days=1 if use_yesterday else 0))
    end = end_dt.strftime("%Y%m%d")
    start = (end_dt - dt.timedelta(days=int(lookback))).strftime("%Y%m%d")

    df_top = get_top_turnover_stocks(end, DEFAULTS["MARKETS"], top_turnover)
    if df_top.empty:
        st.error("거래대금 상위 데이터를 불러올 수 없습니다.")
        st.stop()

    codes = list(df_top.index)
    results = []
    total = len(codes)
    prog = st.progress(0)
    st_text = st.empty()

    with ThreadPoolExecutor(max_workers=DEFAULTS["MAX_WORKERS"]) as ex:
        fut = {ex.submit(get_ohlcv, c, start, end): c for c in codes}
        for i, f in enumerate(as_completed(fut)):
            time.sleep(0.1)  # 안정화용 딜레이
            code = fut[f]
            try:
                df = f.result()
            except:
                df = pd.DataFrame()
            name = stock.get_market_ticker_name(code)
            res = analyze_stock(df, code, name, vol_ratio_min, ret5_max, ret10_max,
                                opt_ma20, opt_body, opt_rsi_macd,
                                opt_gc, opt_rsi_reb, opt_drop, drop_5d)
            if res:
                results.append(res)
            prog.progress((i+1)/total)
            st_text.text(f"{i+1}/{total} 종목 처리 중...")

    if not results:
        st.warning("조건을 만족하는 종목이 없습니다.")
    else:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(["거래량배수", "RSI14"], ascending=[False, True]).head(int(top_n))
        st.success(f"✅ 분석 완료! 추천 종목 {len(df_res)}개 발견")
        st.dataframe(df_res, use_container_width=True)
        csv = df_res.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 결과 CSV 다운로드", csv, "swingpicker_results.csv", "text/csv")
