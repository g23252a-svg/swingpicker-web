# -*- coding: utf-8 -*-
"""
LDY Pro Trader: Nightly Collector (KRX)
- '가장 최근 실제 거래일'을 자동 탐색해 그 날짜 기준으로 유동성 상위 N개 선정
- 각 종목 60거래일 OHLCV 수집 → 지표/점수(EBS) + 추천 매수/손절/매도 계산
- CSV 저장: data/recommend_YYYYMMDD.csv, data/recommend_latest.csv
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pykrx import stock

# ------------------------------- 설정 -------------------------------
KST = timezone(timedelta(hours=9))

LOOKBACK_DAYS = 60          # 조회 일수(캘린더 기준)
TOP_N = 600                 # 거래대금 상위 샘플 크기(300~800 권장)
MIN_TURNOVER_EOK = 50       # 거래대금 하한(억원)
MIN_MCAP_EOK = 1000         # 시총 하한(억원)
RSI_LOW, RSI_HIGH = 45, 65  # RSI 범위
PASS_SCORE = 4              # 통과점수(최종 EBS)
SLEEP_SEC = 0.02            # API call 간 딜레이
OUT_DIR = "data"            # 출력 디렉터리

# ------------------------------- 유틸 -------------------------------
def log(msg: str):
    print(f"[{datetime.now(KST)}] {msg}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def round_to_tick(price: float) -> int:
    # 간단 틱(10원) 반올림
    return int(round(price / 10.0) * 10)

# ------------------------------- 거래일/상위 N 추출 -------------------------------
def _prep_ohlcv_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """get_market_ohlcv_by_ticker 결과를 ['종목코드','거래대금(원)']로 정리"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["종목코드", "거래대금(원)"])
    df = df.reset_index().rename(columns={"티커": "종목코드"}, errors="ignore")
    if "종목코드" not in df.columns:
        df.insert(0, "종목코드", df.iloc[:, 0])
    cand = [c for c in df.columns if "거래대금" in str(c)]
    if not cand:
        # 최후수단: 종가*거래량 근사
        if "종가" in df.columns and "거래량" in df.columns:
            df["거래대금(원)"] = df["종가"].astype("float64") * df["거래량"].astype("float64")
        else:
            df["거래대금(원)"] = np.nan
    else:
        df["거래대금(원)"] = df[cand[0]].astype("float64")
    out = df[["종목코드", "거래대금(원)"]].copy()
    out["종목코드"] = out["종목코드"].astype(str).str.zfill(6)
    return out

def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    """KOSPI/KOSDAQ 당일 티커별 OHLCV에서 '거래대금(원)' 기준 상위 N 추출."""
    kospi = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market="KOSPI")
    kosdaq = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market="KOSDAQ")
    k1 = _prep_ohlcv_by_ticker(kospi)
    k2 = _prep_ohlcv_by_ticker(kosdaq)
    all_df = pd.concat([k1, k2], ignore_index=True)
    all_df = (
        all_df.dropna(subset=["거래대금(원)"])
              .sort_values("거래대금(원)", ascending=False)
              .head(top_n)
              .reset_index(drop=True)
    )
    return all_df

def find_last_trading_day_with_data(base_dt: datetime, max_back_days: int = 7) -> str:
    """
    오늘(또는 기준시각) 포함하여 뒤로 1일씩 물러나며
    '거래대금 합계 > 0' 인 날짜를 찾는다.
    반환: 'YYYYMMDD'
    """
    for i in range(max_back_days + 1):
        d = (base_dt - timedelta(days=i)).strftime("%Y%m%d")
        try:
            kospi = stock.get_market_ohlcv_by_ticker(d, market="KOSPI")
            kosdaq = stock.get_market_ohlcv_by_ticker(d, market="KOSDAQ")
            k1 = _prep_ohlcv_by_ticker(kospi)
            k2 = _prep_ohlcv_by_ticker(kosdaq)
            merged = pd.concat([k1, k2], ignore_index=True)
            tv_sum = float(merged["거래대금(원)"].fillna(0).sum())
            if tv_sum > 0:
                return d
        except Exception:
            pass
    # 못 찾으면 어제 날짜라도 반환
    return (base_dt - timedelta(days=1)).strftime("%Y%m%d")

# ------------------------------- 보조: 시장/이름/시총 -------------------------------
def get_market_map(date_yyyymmdd: str):
    kospi = set(stock.get_market_ticker_list(date_yyyymmdd, market="KOSPI"))
    kosdaq = set(stock.get_market_ticker_list(date_yyyymmdd, market="KOSDAQ"))
    return kospi, kosdaq

def get_name(ticker: str) -> str:
    try:
        nm = stock.get_market_ticker_name(ticker)
        return nm if nm is not None else ""
    except Exception:
        return ""

def get_mcap_eok(date_yyyymmdd: str, ticker: str) -> float:
    try:
        cap = stock.get_market_cap_by_date(date_yyyymmdd, date_yyyymmdd, ticker)
        if cap is None or cap.empty:
            return np.nan
        return float(cap["시가총액"].iloc[0]) / 1e8  # 억원
    except Exception:
        return np.nan

# ------------------------------- 메인 로직 -------------------------------
def main():
    log("전종목 수집 시작…")

    now = datetime.now(KST)
    # 1) '실제 거래 데이터가 존재하는 가장 최근 영업일' 확정
    trade_date = find_last_trading_day_with_data(now, max_back_days=7)
    log(f"📅 거래 기준일 확정: {trade_date}")

    # 2) 상위 N 선정 (trade_date 기준)
    log("🔍 거래대금 상위 종목 선정 중…")
    top_df = pick_top_by_trading_value(trade_date, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ TOP {len(tickers)} 종목 선정 완료")

    # 3) OHLCV 조회 구간 (trade_date를 end로 고정)
    end_dt = datetime.strptime(trade_date, "%Y%m%d").replace(tzinfo=KST)
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
    start_s = start_dt.strftime("%Y%m%d")
    end_s = trade_date

    kospi_set, kosdaq_set = get_market_map(trade_date)

    # 디버깅용 카운터
    cnt_turnover = cnt_mcap = cnt_nan_ind = 0

    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            # OHLCV 60일
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty:
                continue
            ohlcv = ohlcv.reset_index()
            if "날짜" not in ohlcv.columns:
                ohlcv.rename(columns={ohlcv.columns[0]: "날짜"}, inplace=True)
            ohlcv["날짜"] = pd.to_datetime(ohlcv["날짜"])

            close = ohlcv["종가"].astype(float)
            high  = ohlcv["고가"].astype(float)
            low   = ohlcv["저가"].astype(float)
            vol   = ohlcv["거래량"].astype(float)

            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            atr14 = calc_atr(high, low, close, 14)
            rsi14 = calc_rsi(close, 14)

            ema12 = ema(close, 12)
            ema26 = ema(close, 26)
            macd_line = ema12 - ema26
            macd_signal = ema(macd_line, 9)
            macd_hist = macd_line - macd_signal
            macd_slope = macd_hist.diff()

            vol_z = vol / (vol.rolling(20).mean())
            disp = (close / ma20 - 1.0) * 100  # 乖離%

            last = ohlcv.iloc[-1]
            c = float(last["종가"])
            v_z = float(vol_z.iloc[-1]) if not np.isnan(vol_z.iloc[-1]) else np.nan
            rsi_v = float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else np.nan
            macd_h = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else np.nan
            macd_sl = float(macd_slope.iloc[-1]) if not np.isnan(macd_slope.iloc[-1]) else np.nan
            m20 = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else np.nan
            m60 = float(ma60.iloc[-1]) if not np.isnan(ma60.iloc[-1]) else np.nan
            atr = float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else np.nan
            disp_v = float(disp.iloc[-1]) if not np.isnan(disp.iloc[-1]) else np.nan

            ret5 = (close.pct_change(5).iloc[-1] * 100) if len(close) >= 6 else np.nan
            ret10 = (close.pct_change(10).iloc[-1] * 100) if len(close) >= 11 else np.nan

            # 시장/종목명/시총/거래대금(억원)
            mkt = "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타")
            name = get_name(t) or t  # 이름 못 받으면 티커로 대체
            tv_eok = float(top_df.loc[top_df["종목코드"] == t, "거래대금(원)"].values[0]) / 1e8
            mcap_eok = get_mcap_eok(trade_date, t)

            # 필터
            if tv_eok < MIN_TURNOVER_EOK:
                cnt_turnover += 1
                continue
            if not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK:
                cnt_mcap += 1
                continue
            if np.isnan(atr) or np.isnan(m20):
                cnt_nan_ind += 1
                continue

            # EBS (급등 초입 스코어)
            score = 0
            reason = []
            if RSI_LOW <= rsi_v <= RSI_HIGH:
                score += 1; reason.append("RSI 45~65")
            if macd_sl > 0:
                score += 1; reason.append("MACD↑")
            if not np.isnan(disp_v) and -1.0 <= disp_v <= 4.0:
                score += 1; reason.append("MA20 근처")
            if v_z > 1.2:
                score += 1; reason.append("거래량↑")
            if not np.isnan(m20) and not np.isnan(m60) and m20 > m60:
                score += 1; reason.append("상승구조")
            if macd_h > 0:
                score += 1; reason.append("MACD>sig")
            if not np.isnan(ret5) and ret5 < 10:
                score += 1; reason.append("과열X")

            # 추천가(보수적)
            buy = min(c, m20 * 1.01)
            stop = buy - 1.5 * atr
            tgt1 = buy + (buy - stop) * 1.0   # R:R=1
            tgt2 = buy + (buy - stop) * 2.0   # R:R=2

            buy  = round_to_tick(buy)
            stop = max(round_to_tick(stop), round_to_tick(m20 * 0.97))
            tgt1 = round_to_tick(tgt1)
            tgt2 = round_to_tick(tgt2)

            rows.append({
                "시장": mkt,
                "종목명": name,
                "종목코드": t,
                "종가": int(c),
                "거래대금(억원)": round(tv_eok, 2),
                "시가총액(억원)": None if np.isnan(mcap_eok) else round(mcap_eok, 1),
                "RSI14": None if np.isnan(rsi_v) else round(rsi_v, 1),
                "乖離%": None if np.isnan(disp_v) else round(disp_v, 2),
                "MACD_hist": None if np.isnan(macd_h) else round(macd_h, 4),
                "MACD_slope": None if np.isnan(macd_sl) else round(macd_sl, 5),
                "Vol_Z": None if np.isnan(v_z) else round(v_z, 2),
                "ret_5d_%": None if np.isnan(ret5) else round(ret5, 2),
                "ret_10d_%": None if np.isnan(ret10) else round(ret10, 2),
                "EBS": int(score),
                "통과": "🚀초입" if score >= PASS_SCORE else "",
                "근거": ", ".join(reason),
                "추천매수가": buy,
                "추천매도가1": tgt1,
                "추천매도가2": tgt2,
                "손절가": stop,
            })
        except Exception as e:
            log(f"⚠️ {t} 처리 실패: {e}")
        time.sleep(SLEEP_SEC)

    if not rows:
        log(f"필터 컷 통계 ▶ 거래대금<{MIN_TURNOVER_EOK}억: {cnt_turnover}, "
            f"시총<{MIN_MCAP_EOK}억: {cnt_mcap}, 지표 NaN: {cnt_nan_ind}")
        raise RuntimeError("수집 결과가 비었습니다.")

    df_out = pd.DataFrame(rows).sort_values(
        ["EBS", "거래대금(억원)"], ascending=[False, False]
    ).reset_index(drop=True)

    ensure_dir(OUT_DIR)
    path_day = os.path.join(OUT_DIR, f"recommend_{trade_date}.csv")
    path_latest = os.path.join(OUT_DIR, "recommend_latest.csv")
    df_out.to_csv(path_day, index=False, encoding="utf-8-sig")
    df_out.to_csv(path_latest, index=False, encoding="utf-8-sig")
    log(f"💾 저장 완료: {path_day} (+ {path_latest})")

if __name__ == "__main__":
    main()
