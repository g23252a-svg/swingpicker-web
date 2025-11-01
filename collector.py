# -*- coding: utf-8 -*-
"""
LDY Pro Trader: Nightly Collector (KRX) — v3.4 (EV/Regime/De-corr/Event)
- 매일 장마감 후: 유동성 상위(TV 상위) 종목 n개 선정
- 각 종목 60거래일 OHLCV 수집 → 지표/점수(EBS) + 엔트리/스탑/목표(TP1/TP2)
- EV스코어/레짐게이팅/상관노출 제한/이벤트 락아웃(선택)까지 반영
- pykrx 시그니처 차이 회피: TV 상위 선정은 get_market_ohlcv_by_ticker()로 일원화
"""

import os, time, math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pykrx import stock

# ------------------------------- 설정 -------------------------------
KST = timezone(timedelta(hours=9))

LOOKBACK_DAYS = 60          # 조회일수
TOP_N = 600                 # 거래대금 상위 샘플 크기
MIN_TURNOVER_EOK = 50       # 거래대금 하한(억원)
MIN_MCAP_EOK = 1000         # 시총 하한(억원)
PASS_SCORE = 4              # 통과점수(EBS)
SLEEP_SEC = 0.05            # API call 간 딜레이(안정성)

# ── Top Picks 필터(즉시 체감 5가지) ──
MIN_RR1 = 1.20              # 목표1/손절 최소 R/R
MIN_STOP_GAP_PCT = 1.0      # 스탑여유(%) 하한
MIN_TGT1_GAP_PCT = 6.0      # 목표1여유(%) 하한
REGIME_STRICT = True        # 레짐 불리 시 제외(True) / 가중치만(0.8배) 적용(False)
EVENT_LOCKOUT_STRICT = True # 이벤트 근접 시 제외(True) / 배지만 달기(False)
CORR_THRESHOLD = 0.65       # 20일 수익률 상관 상한(초과면 같은 군집으로 간주)
MAX_TOPPICKS = 60           # Top Picks 최대 표기 수

OUT_DIR = "data"
UTF8 = "utf-8-sig"

# ------------------------------- 유틸 -------------------------------
def log(msg: str):
    print(f"[{datetime.now(KST)}] {msg}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def calc_rsi(close: pd.Series, period: int = 14):
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    au, ad = up.rolling(period).mean(), dn.rolling(period).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100/(1+rs)

def calc_atr(high, low, close, period: int = 14):
    prev = close.shift(1)
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def round_to_tick(price: float) -> int:
    return int(round(price / 10.0) * 10)  # KRX 10원 틱 단순화

# ------------------------------- 기준일 결정 -------------------------------
def resolve_trade_date() -> str:
    """
    장마감 집계 시차 고려: 당일 18시 이전이면 전일, 데이터 없으면 하루씩 뒤로.
    반환: 'YYYYMMDD'
    """
    now = datetime.now(KST)
    d = now.date()
    if now.hour < 18:
        d = d - timedelta(days=1)
    for _ in range(7):
        ymd = d.strftime("%Y%m%d")
        try:
            tmp = stock.get_market_ohlcv_by_ticker(ymd, market="KOSPI")
            if tmp is not None and not tmp.empty and "거래대금" in tmp.columns:
                return ymd
        except Exception:
            pass
        d = d - timedelta(days=1)
    return datetime.now(KST).strftime("%Y%m%d")

# ------------------------------- 상위 TV 선정 -------------------------------
def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m)
            if df is None or df.empty:
                continue
            df = df.reset_index()
            if "티커" in df.columns:
                df = df.rename(columns={"티커": "종목코드"})
            if "거래대금(원)" not in df.columns and "거래대금" in df.columns:
                df = df.rename(columns={"거래대금":"거래대금(원)"})
            frames.append(df[["종목코드","거래대금(원)"]])
        except Exception as e:
            log(f"⚠️ {m} TV 집계 실패: {e}")
    if not frames:
        raise RuntimeError("거래대금 상위 집계용 데이터 없음")
    tv_df = pd.concat(frames, ignore_index=True)
    tv_df["종목코드"] = tv_df["종목코드"].astype(str).str.zfill(6)
    tv_df["거래대금(원)"] = pd.to_numeric(tv_df["거래대금(원)"], errors="coerce").fillna(0)
    tv_df = tv_df.sort_values("거래대금(원)", ascending=False).head(top_n).reset_index(drop=True)
    return tv_df

def get_market_map(date_yyyymmdd: str):
    kospi = set(stock.get_market_ticker_list(date_yyyymmdd, market="KOSPI"))
    kosdaq = set(stock.get_market_ticker_list(date_yyyymmdd, market="KOSDAQ"))
    return kospi, kosdaq

def get_name_map_cached(date_yyyymmdd: str) -> dict:
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, "krx_codes.csv")
    mp = {}
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype={"종목코드":"string"})
            for _, r in df.iterrows():
                mp[str(r["종목코드"]).zfill(6)] = r.get("종목명","")
        except Exception:
            mp = {}
    if not mp:
        rows = []
        for m in ["KOSPI","KOSDAQ","KONEX"]:
            try:
                lst = stock.get_market_ticker_list(date_yyyymmdd, market=m)
            except Exception:
                lst = []
            for t in lst:
                try:
                    nm = stock.get_market_ticker_name(t)
                except Exception:
                    nm = ""
                rows.append({"종목코드": str(t).zfill(6), "종목명": nm, "시장": m})
                time.sleep(0.002)
        if rows:
            df = pd.DataFrame(rows).drop_duplicates("종목코드")
            df.to_csv(path, index=False, encoding=UTF8)
            mp = {str(r["종목코드"]).zfill(6): r["종목명"] for _, r in df.iterrows()}
    return mp

def get_mcap_eok(date_yyyymmdd: str, ticker: str) -> float:
    try:
        cap = stock.get_market_cap_by_date(date_yyyymmdd, date_yyyymmdd, ticker)
        return float(cap["시가총액"].iloc[0]) / 1e8
    except Exception:
        return np.nan

# ------------------------------- 레짐 게이팅 -------------------------------
def get_index_regime(date_yyyymmdd: str) -> dict:
    """
    KOSPI/KOSDAQ 120영업일 불러서 MA50 및 기울기 확인.
    return: {"KOSPI": (above50, slope>0), "KOSDAQ": (...)}
    """
    # pykrx 지수코드: KOSPI(1001), KOSDAQ(2001)
    start = (datetime.strptime(date_yyyymmdd, "%Y%m%d") - timedelta(days=240)).strftime("%Y%m%d")
    out = {}
    for code, key in [("1001","KOSPI"), ("2001","KOSDAQ")]:
        try:
            idx = stock.get_index_ohlcv_by_date(start, date_yyyymmdd, code)
            if idx is None or idx.empty: 
                out[key] = (True, True)  # 못 가져오면 관대 처리
                continue
            close = idx["종가"].astype(float)
            ma50 = close.rolling(50).mean()
            above = bool(close.iloc[-1] > ma50.iloc[-1])
            slope = bool(ma50.iloc[-1] - ma50.iloc[-5] > 0) if len(ma50) >= 55 else True
            out[key] = (above, slope)
        except Exception:
            out[key] = (True, True)
    return out

# ------------------------------- 이벤트 락아웃 -------------------------------
def load_event_calendar() -> pd.DataFrame | None:
    """
    선택 CSV: data/events.csv (컬럼: 종목코드,날짜[YYYY-MM-DD],이벤트)
    없으면 None
    """
    path = os.path.join(OUT_DIR, "events.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype={"종목코드":"string"})
        if "날짜" in df.columns:
            df["날짜"] = pd.to_datetime(df["날짜"]).dt.date
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        return df[["종목코드","날짜","이벤트"]].dropna()
    except Exception:
        return None

def is_event_locked(event_df: pd.DataFrame | None, ticker: str, trade_date: str) -> bool:
    if event_df is None: 
        return False
    d = datetime.strptime(trade_date, "%Y%m%d").date()
    week = pd.bdate_range(d - timedelta(days=5), d + timedelta(days=5)).date
    near = event_df[(event_df["종목코드"]==str(ticker).zfill(6)) & (event_df["날짜"].isin(week))]
    return len(near) > 0

# ------------------------------- CP949 안전 치환 -------------------------------
def make_cp949_safe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [c.replace("乖離%", "괴리_%") for c in df2.columns]
    if "근거" in df2.columns and df2["근거"].dtype == object:
        df2["근거"] = (
            df2["근거"]
            .str.replace("MACD↑","MACD상승",regex=False)
            .str.replace("거래량↑","거래량증가",regex=False)
            .str.replace("과열X","과열아님",regex=False)
        )
    return df2

# ------------------------------- 메인 -------------------------------
def main():
    log("전종목 수집 시작…")
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일 확정: {trade_ymd}")

    log("🔍 거래대금 상위 종목 선정 중…")
    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ TOP {len(tickers)} 종목 선정 완료")

    kospi_set, kosdaq_set = get_market_map(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)
    regime = get_index_regime(trade_ymd)    # {"KOSPI":(above,slope), "KOSDAQ":(above,slope)}
    events = load_event_calendar()

    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS*2)
    start_s = start_dt.strftime("%Y%m%d")
    end_s = trade_ymd

    rows = []
    prices_for_corr = {}  # 수익률 상관 계산용: {ticker: close_series}

    for i, t in enumerate(tickers, 1):
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty:
                continue
            ohlcv = ohlcv.reset_index().rename(columns={"index":"날짜"})
            ohlcv["날짜"] = pd.to_datetime(ohlcv["날짜"])
            ohlcv = ohlcv.tail(LOOKBACK_DAYS)

            close = ohlcv["종가"].astype(float)
            high  = ohlcv["고가"].astype(float)
            low   = ohlcv["저가"].astype(float)
            vol   = ohlcv["거래량"].astype(float)
            if len(close) < 20: 
                continue

            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            atr14 = calc_atr(high, low, close, 14)
            rsi14 = calc_rsi(close, 14)

            ema12 = ema(close,12); ema26 = ema(close,26)
            macd_line = ema12 - ema26
            macd_sig  = ema(macd_line, 9)
            macd_hist = macd_line - macd_sig
            macd_slope = macd_hist.diff()

            vol_z = vol / (vol.rolling(20).mean())
            disp  = (close/ma20 - 1.0)*100

            last = ohlcv.iloc[-1]
            c = float(last["종가"])
            v_z     = float(vol_z.iloc[-1]) if not np.isnan(vol_z.iloc[-1]) else np.nan
            rsi_v   = float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else np.nan
            macd_h  = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else np.nan
            macd_sl = float(macd_slope.iloc[-1]) if not np.isnan(macd_slope.iloc[-1]) else np.nan
            m20     = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else np.nan
            m60     = float(ma60.iloc[-1]) if not np.isnan(ma60.iloc[-1]) else np.nan
            atr     = float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else np.nan
            disp_v  = float(disp.iloc[-1]) if not np.isnan(disp.iloc[-1]) else np.nan
            ret5  = (close.pct_change(5 ).iloc[-1]*100) if len(close)>=6  else np.nan
            ret10 = (close.pct_change(10).iloc[-1]*100) if len(close)>=11 else np.nan

            mkt = "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타")
            name = name_map.get(str(t).zfill(6), "") or stock.get_market_ticker_name(t)
            tv_eok = float(top_df.loc[top_df["종목코드"]==t,"거래대금(원)"].values[0]) / 1e8
            mcap_eok = get_mcap_eok(trade_ymd, t)

            # 기본 컷
            if tv_eok < MIN_TURNOVER_EOK or (not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK):
                continue

            # EBS 점수
            score, reason = 0, []
            if 45 <= rsi_v <= 65: score+=1; reason.append("RSI 45~65")
            if macd_sl > 0:       score+=1; reason.append("MACD↑")
            if not np.isnan(disp_v) and -1.0 <= disp_v <= 4.0: score+=1; reason.append("MA20 근처")
            if v_z > 1.2:         score+=1; reason.append("거래량↑")
            if not np.isnan(m20) and not np.isnan(m60) and m20 > m60: score+=1; reason.append("상승구조")
            if macd_h > 0:        score+=1; reason.append("MACD>0")
            if not np.isnan(ret5) and ret5 < 10: score+=1; reason.append("5d<10%")

            if np.isnan(atr) or np.isnan(m20):
                continue

            # 브래킷 (엔트리/목표/스탑)
            band_lo, band_hi = m20-0.5*atr, m20+0.5*atr
            entry = min(max(c, band_lo), band_hi)
            stop  = entry - 1.5*atr
            tgt1  = entry + (entry - stop)*1.0
            tgt2  = entry + (entry - stop)*2.0

            entry = round_to_tick(entry)
            stop  = max(round_to_tick(stop), round_to_tick(m20*0.97))
            tgt1  = round_to_tick(tgt1)
            tgt2  = round_to_tick(tgt2)

            # R/R 및 여유
            rr1 = (tgt1 - entry) / (entry - stop) if (entry - stop) > 0 else np.nan
            stop_gap = (entry - stop)/entry*100
            tgt1_gap = (tgt1 - entry)/entry*100
            now_band = (c - entry)/entry*100  # 0이면 엔트리와 같은 수준

            # EV 스코어
            if not np.isnan(rr1) and rr1 > 0:
                ev_r = (rr1 - 1) / (rr1 + 1)     # 드리프트 0 가정 EV_R
            else:
                ev_r = np.nan
            ers = (score/7.0)*rr1 if (not np.isnan(rr1)) else np.nan
            ev_score = ev_r*(score/7.0) if (not np.isnan(ev_r)) else np.nan

            # 레짐 플래그
            above50, slope_up = regime.get(mkt, (True, True))
            regime_ok = bool(above50 and slope_up)

            # 이벤트 락아웃
            event_risk = is_event_locked(events, t, trade_ymd)

            # 상관계산용 저장
            prices_for_corr[t] = close.reset_index(drop=True)

            rows.append({
                "날짜": trade_ymd,
                "시장": mkt,
                "종목명": name,
                "종목코드": str(t).zfill(6),
                "종가": int(c),
                "거래대금(억원)": round(tv_eok,2),
                "시가총액(억원)": None if np.isnan(mcap_eok) else round(mcap_eok,1),
                "RSI14": None if np.isnan(rsi_v) else round(rsi_v,1),
                "乖離%": None if np.isnan(disp_v) else round(disp_v,2),
                "MACD_hist": None if np.isnan(macd_h) else round(macd_h,4),
                "MACD_slope": None if np.isnan(macd_sl) else round(macd_sl,5),
                "Vol_Z": None if np.isnan(v_z) else round(v_z,2),
                "ret_5d_%": None if np.isnan(ret5) else round(ret5,2),
                "ret_10d_%": None if np.isnan(ret10) else round(ret10,2),
                "EBS": int(score),
                "근거": ", ".join(reason),
                "추천매수가": entry,
                "손절가": stop,
                "추천매도가1": tgt1,
                "추천매도가2": tgt2,
                "RR1": None if np.isnan(rr1) else round(rr1,3),
                "Stop여유_%": None if np.isnan(stop_gap) else round(stop_gap,2),
                "Target1여유_%": None if np.isnan(tgt1_gap) else round(tgt1_gap,2),
                "Now밴드거리_%": None if np.isnan(now_band) else round(now_band,2),
                "EV_R": None if np.isnan(ev_r) else round(ev_r,3),
                "ERS": None if np.isnan(ers) else round(ers,3),
                "EV_SCORE": None if np.isnan(ev_score) else round(ev_score,3),
                "REGIME_OK": regime_ok,
                "EVENT_RISK": bool(event_risk),
            })
        except Exception as e:
            log(f"⚠️ {t} 처리 실패: {e}")
        time.sleep(SLEEP_SEC)

    if not rows:
        raise RuntimeError("수집 결과가 비었습니다.")

    df = pd.DataFrame(rows)

    # ── Top Picks 규칙(하한/레짐/이벤트) 적용 ──
    mask = (
        (df["EBS"] >= PASS_SCORE) &
        (pd.to_numeric(df["RR1"], errors="coerce") >= MIN_RR1) &
        (pd.to_numeric(df["Stop여유_%"], errors="coerce") >= MIN_STOP_GAP_PCT) &
        (pd.to_numeric(df["Target1여유_%"], errors="coerce") >= MIN_TGT1_GAP_PCT)
    )
    if REGIME_STRICT:
        mask &= df["REGIME_OK"].fillna(True)
    else:
        # 레짐 불리 시 EV_SCORE 0.8배
        df.loc[~df["REGIME_OK"].fillna(True), "EV_SCORE"] = df["EV_SCORE"] * 0.8

    if EVENT_LOCKOUT_STRICT:
        mask &= ~df["EVENT_RISK"].fillna(False)

    # ── 상관노출 제한(20일 수익률 상관) ──
    picks = df[mask].copy()
    # 수익률 행렬 구성
    rets = {}
    for t, s in prices_for_corr.items():
        if len(s) >= 21:
            rets[t] = s.pct_change().tail(20).reset_index(drop=True)
    ret_df = pd.DataFrame(rets).corr(min_periods=10) if rets else pd.DataFrame()

    # EV_SCORE → ERS → 거래대금 순 정렬
    picks = picks.sort_values(
        ["EV_SCORE","ERS","거래대금(억원)"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    accepted = []
    for _, r in picks.iterrows():
        t = r["종목코드"]
        ok = True
        for a in accepted:
            if not ret_df.empty and t in ret_df.columns and a in ret_df.index:
                corr = ret_df.loc[a, t]
                if pd.notna(corr) and corr > CORR_THRESHOLD:
                    ok = False
                    break
        if ok:
            accepted.append(t)
        if len(accepted) >= MAX_TOPPICKS:
            break

    df["TopPick"] = df["종목코드"].isin(accepted)
    # 최종 정렬: TopPick 우선 → EV_SCORE → 거래대금
    df = df.sort_values(
        ["TopPick","EV_SCORE","거래대금(억원)"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # 저장
    ensure_dir(OUT_DIR)
    path_day = os.path.join(OUT_DIR, f"recommend_{trade_ymd}.csv")
    path_latest = os.path.join(OUT_DIR, "recommend_latest.csv")
    df.to_csv(path_day, index=False, encoding=UTF8)
    df.to_csv(path_latest, index=False, encoding=UTF8)

    # (선택) CP949-safe 버전 추가 저장
    df_cp = make_cp949_safe(df)
    df_cp.to_csv(os.path.join(OUT_DIR, f"recommend_{trade_ymd}_cp949.csv"), index=False, encoding="cp949")

    log(f"💾 저장 완료: {path_day} / {path_latest} (+ cp949)")
    log(f"📊 TopPick: {df['TopPick'].sum()} / 전체 {len(df)}")

if __name__ == "__main__":
    main()
