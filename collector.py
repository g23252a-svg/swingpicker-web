# -*- coding: utf-8 -*-
"""
LDY Pro Trader: Nightly Collector (KRX) — v3.4.1
- 매일 장마감 후: 거래대금 상위 종목 N개 추출
- 각 종목 60거래일 OHLCV로 지표/EBS/추천가 산출
- **거래대금(억원) 컷은 per-ticker OHLCV의 (거래량×종가)/1e8 로 판단** ← 단위 혼선 제거
- 주말/휴일 기준일 보완, 스킵 사유 요약 로깅 추가
"""

import os
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pykrx import stock

# ------------------------------- 설정 -------------------------------
KST = timezone(timedelta(hours=9))
LOOKBACK_DAYS = 60
TOP_N = 600
MIN_TURNOVER_EOK = 50     # 컷: 거래대금(억원)
MIN_MCAP_EOK = 1000       # 컷: 시총(억원)
RSI_LOW, RSI_HIGH = 45, 65
PASS_SCORE = 4
SLEEP_SEC = 0.03
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
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.rolling(period).mean() / dn.rolling(period).mean().replace(0, np.nan)
    return 100 - 100/(1+rs)

def calc_atr(high, low, close, period: int = 14):
    prev = close.shift(1)
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def round_to_tick(price: float) -> int:
    return int(round(price / 10.0) * 10)

# ------------------------------- 기준일 결정 -------------------------------
def resolve_trade_date() -> str:
    """
    pykrx 일부 함수가 비영업일 입력 시 '직전 영업일' 데이터를 돌려줄 때가 있어
    안전하게 최근 7일을 뒤로 훑으며 '실데이터 존재'일을 고른다.
    """
    now = datetime.now(KST)
    d = now.date()
    # 오후 6시 이전엔 어제 데이터로 가정 (수집 지연 대비)
    if now.hour < 18:
        d = d - timedelta(days=1)

    for _ in range(7):
        ymd = d.strftime("%Y%m%d")
        try:
            df = stock.get_market_ohlcv_by_ticker(ymd, market="KOSPI")
            if df is not None and not df.empty and "거래대금" in df.columns:
                return ymd
        except Exception:
            pass
        d = d - timedelta(days=1)
    return (now.date() - timedelta(days=1)).strftime("%Y%m%d")

# ------------------------------- 상위 TV 선정 -------------------------------
def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    """
    랭킹용(정렬)으로만 사용. 단위 혼선 방지를 위해 컷 판정은 per-ticker OHLCV에서 수행한다.
    """
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m)
            if df is None or df.empty:
                continue
            df = df.reset_index()  # index: 티커
            if "티커" in df.columns:
                df.rename(columns={"티커": "종목코드"}, inplace=True)
            # pykrx는 '거래대금' 단위가 호출별로 다를 수 있다 → 여기선 단순 정렬용으로만 사용
            df.rename(columns={"거래대금": "거래대금_raw"}, inplace=True)
            frames.append(df[["종목코드", "거래대금_raw"]])
        except Exception as e:
            log(f"⚠️ {m} TV 집계 실패: {e}")

    if not frames:
        raise RuntimeError("거래대금 상위 집계에 사용할 데이터가 없습니다.")

    tv_df = pd.concat(frames, ignore_index=True)
    tv_df["종목코드"] = tv_df["종목코드"].astype(str).str.zfill(6)
    tv_df["거래대금_raw"] = pd.to_numeric(tv_df["거래대금_raw"], errors="coerce").fillna(0)
    tv_df = tv_df.sort_values("거래대금_raw", ascending=False).head(top_n).reset_index(drop=True)
    return tv_df

def get_market_map(date_yyyymmdd: str):
    kospi = set(stock.get_market_ticker_list(date_yyyymmdd, market="KOSPI"))
    kosdaq = set(stock.get_market_ticker_list(date_yyyymmdd, market="KOSDAQ"))
    return kospi, kosdaq

def get_name_map_cached(date_yyyymmdd: str) -> dict:
    ensure_dir(OUT_DIR)
    map_path = os.path.join(OUT_DIR, "krx_codes.csv")
    mp = {}
    if os.path.exists(map_path):
        try:
            df = pd.read_csv(map_path, dtype={"종목코드": "string"})
            for _, r in df.iterrows():
                mp[str(r["종목코드"]).zfill(6)] = r.get("종목명", "")
        except Exception:
            mp = {}

    if not mp:
        rows = []
        for m in ["KOSPI", "KOSDAQ", "KONEX"]:
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
            df.to_csv(map_path, index=False, encoding=UTF8)
            mp = {str(r["종목코드"]).zfill(6): r["종목명"] for _, r in df.iterrows()}
    return mp

def get_mcap_eok(date_yyyymmdd: str, ticker: str) -> float:
    try:
        cap = stock.get_market_cap_by_date(date_yyyymmdd, date_yyyymmdd, ticker)
        if cap is None or cap.empty:
            return np.nan
        return float(cap["시가총액"].iloc[0]) / 1e8
    except Exception:
        return np.nan

# ------------------------------- CP949 안전 치환 -------------------------------
def make_cp949_safe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [c.replace("乖離%", "괴리_%") for c in df2.columns]
    if "통과" in df2.columns:
        df2["통과"] = df2["통과"].replace({"🚀초입": "초입"})
    if "근거" in df2.columns and df2["근거"].dtype == object:
        df2["근거"] = (
            df2["근거"]
            .str.replace("MACD↑", "MACD상승", regex=False)
            .str.replace("거래량↑", "거래량증가", regex=False)
            .str.replace("과열X", "과열아님", regex=False)
        )
    return df2

# ------------------------------- 메인 로직 -------------------------------
def main():
    log("전종목 수집 시작…")

    # 1) 기준일
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일 확정: {trade_ymd}")

    # 2) 상위 거래대금 종목(정렬용)
    log("🔍 거래대금 상위 종목 선정 중…")
    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ TOP {len(tickers)} 종목 선정 완료")

    # 3) 시장/이름 맵
    kospi_set, kosdaq_set = get_market_map(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)

    # 4) per-ticker 수집/산출
    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS * 2)
    start_s = start_dt.strftime("%Y%m%d")
    end_s = trade_ymd

    rows = []
    # 진단 카운터
    c_len = c_turn = c_mcap = c_nan = c_calc = 0

    for i, t in enumerate(tickers, 1):
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty:
                c_len += 1
                continue

            ohlcv = ohlcv.reset_index().rename(columns={"index": "날짜"})
            # pykrx는 '날짜'가 index name인 경우가 많음 → 위 한 줄이면 충분

            # 최근 60거래일만
            ohlcv["날짜"] = pd.to_datetime(ohlcv["날짜"])
            ohlcv = ohlcv.tail(LOOKBACK_DAYS)
            if len(ohlcv) < 20:
                c_len += 1
                continue

            close = ohlcv["종가"].astype(float)
            high  = ohlcv["고가"].astype(float)
            low   = ohlcv["저가"].astype(float)
            vol   = ohlcv["거래량"].astype(float)

            # 거래대금(억원) — **단위 혼선 제거: 거래량×종가로 계산**
            last = ohlcv.iloc[-1]
            tv_eok = float(last["거래량"]) * float(last["종가"]) / 1e8

            # 시총(억원)
            mcap_eok = get_mcap_eok(trade_ymd, t)

            # 컷
            if tv_eok < MIN_TURNOVER_EOK:
                c_turn += 1
                continue
            if not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK:
                c_mcap += 1
                continue

            # 지표
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            atr14 = calc_atr(high, low, close, 14)
            rsi14 = calc_rsi(close, 14)

            ema12 = ema(close, 12)
            ema26 = ema(close, 26)
            macd_line   = ema12 - ema26
            macd_signal = ema(macd_line, 9)
            macd_hist   = macd_line - macd_signal
            macd_slope  = macd_hist.diff()

            disp  = (close / ma20 - 1.0) * 100  # 乖離%

            # 최종 값
            c = float(last["종가"])
            v_rsi   = rsi14.iloc[-1]
            v_mh    = macd_hist.iloc[-1]
            v_ms    = macd_slope.iloc[-1]
            v_ma20  = ma20.iloc[-1]
            v_ma60  = ma60.iloc[-1]
            v_atr   = atr14.iloc[-1]
            v_disp  = disp.iloc[-1]
            ret5  = (close.pct_change(5 ).iloc[-1] * 100) if len(close) >= 6  else np.nan
            ret10 = (close.pct_change(10).iloc[-1] * 100) if len(close) >= 11 else np.nan

            if any(map(lambda x: pd.isna(x), [v_ma20, v_atr, c])):
                c_nan += 1
                continue

            # EBS (급등 초입)
            score, reason = 0, []
            if RSI_LOW <= v_rsi <= RSI_HIGH: score += 1; reason.append("RSI 45~65")
            if v_ms > 0:                     score += 1; reason.append("MACD상승")
            if not pd.isna(v_disp) and -1.0 <= v_disp <= 4.0: score += 1; reason.append("MA20 근처")
            # Vol Z 대신 보수적: (최근/20일평균) 비율
            vol_z = vol.iloc[-1] / max(1.0, vol.rolling(20).mean().iloc[-1])
            if vol_z > 1.2:                  score += 1; reason.append("거래량증가")
            if v_ma20 > v_ma60:              score += 1; reason.append("상승구조")
            if v_mh > 0:                     score += 1; reason.append("MACD>sig")
            if not pd.isna(ret5) and ret5 < 10: score += 1; reason.append("과열아님")

            # 추천가 (MA20±0.5ATR 밴드 클램프)
            band_lo, band_hi = v_ma20 - 0.5 * v_atr, v_ma20 + 0.5 * v_atr
            buy  = min(max(c, band_lo), band_hi)
            stop = max(band_lo - 0.7 * v_atr, buy - 1.5 * v_atr)  # 살짝 여유
            tgt1 = buy + (buy - stop) * 1.0
            tgt2 = buy + (buy - stop) * 2.0

            buy  = round_to_tick(buy)
            stop = round_to_tick(stop)
            tgt1 = round_to_tick(tgt1)
            tgt2 = round_to_tick(tgt2)

            mkt = "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타")
            name = name_map.get(str(t).zfill(6), "") or stock.get_market_ticker_name(t)

            rows.append({
                "시장": mkt,
                "종목명": name,
                "종목코드": t,
                "종가": int(c),
                "거래대금(억원)": round(tv_eok, 2),   # ← OHLCV 기반
                "시가총액(억원)": None if np.isnan(mcap_eok) else round(mcap_eok, 1),
                "RSI14": None if pd.isna(v_rsi) else round(v_rsi, 1),
                "乖離%": None if pd.isna(v_disp) else round(v_disp, 2),
                "MACD_hist": None if pd.isna(v_mh) else round(v_mh, 4),
                "MACD_slope": None if pd.isna(v_ms) else round(v_ms, 5),
                "Vol_Z": round(float(vol_z), 2) if not pd.isna(vol_z) else None,
                "ret_5d_%": None if pd.isna(ret5) else round(float(ret5), 2),
                "ret_10d_%": None if pd.isna(ret10) else round(float(ret10), 2),
                "EBS": int(score),
                "통과": "초입" if score >= PASS_SCORE else "",
                "근거": ", ".join(reason),
                "추천매수가": buy,
                "추천매도가1": tgt1,
                "추천매도가2": tgt2,
                "손절가": stop,
            })
        except Exception as e:
            c_calc += 1
            log(f"⚠️ {t} 처리 실패: {e}")
        time.sleep(SLEEP_SEC)

    # 요약 로그
    log(f"요약) 수집 rows: {len(rows)}  | 스킵 길이<20: {c_len} | 거래대금컷: {c_turn} | 시총컷: {c_mcap} | NaN컷: {c_nan} | 예외: {c_calc}")

    if not rows:
        raise RuntimeError("수집 결과가 비었습니다.")

    df_out = pd.DataFrame(rows)
    # 정렬: EBS▼, 거래대금(억원)▼
    df_out = df_out.sort_values(["EBS", "거래대금(억원)"], ascending=[False, False]).reset_index(drop=True)

    # 저장
    ensure_dir(OUT_DIR)
    path_day_utf8    = os.path.join(OUT_DIR, f"recommend_{trade_ymd}.csv")
    path_latest_utf8 = os.path.join(OUT_DIR, "recommend_latest.csv")

    df_out.to_csv(path_day_utf8, index=False, encoding=UTF8)
    df_out.to_csv(path_latest_utf8, index=False, encoding=UTF8)
    log(f"✅ 저장 완료 → {path_day_utf8}, {path_latest_utf8}")
    log("끝.")

if __name__ == "__main__":
    main()
