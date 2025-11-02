# -*- coding: utf-8 -*-
"""
LDY Pro Trader: Nightly Collector (KRX, robust biz-day & mcap)
- 매일 장마감 후: 거래대금 상위(TV) 종목 선정(KOSPI+KOSDAQ)
- 각 종목 60거래일 OHLCV 수집 → 지표/점수(EBS) → 추천가/손절/목표가 산출
- '주말/휴일 ymd'로 인한 시총 0 프레임 방지: 영업일/시총 양수 검증 후 롤백
- mcap은 시장별 일괄 조회 → 억원 변환 맵으로 사용 (0/NaN은 무효 처리)
- 결과: data/recommend_YYYYMMDD.csv, data/recommend_latest.csv, data/krx_codes.csv
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

LOOKBACK_DAYS = 60          # 조회일수
TOP_N = 600                 # 거래대금 상위 샘플 크기(300~800 권장)
MIN_TURNOVER_EOK = 50       # 거래대금 하한(억원)
MIN_MCAP_EOK = 1000         # 시총 하한(억원)
RSI_LOW, RSI_HIGH = 45, 65  # RSI 범위
PASS_SCORE = 4              # 통과점수(최종 EBS)
SLEEP_SEC = 0.05            # API call 간 딜레이(안정성)
OUT_DIR = "data"
UTF8 = "utf-8-sig"

# ------------------------------- 유틸 -------------------------------
def log(msg: str):
    print(f"[{datetime.now(KST)}] {msg}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def calc_rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - 100 / (1 + rs)

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def round_to_tick(price: float) -> int:
    # 간단 틱(10원 단위)
    return int(round(price / 10.0) * 10)

def _safe_sum(x) -> float:
    return pd.to_numeric(x, errors="coerce").fillna(0).sum()

# ------------------------------- 영업일/시총 검증 -------------------------------
def _has_ohlcv_and_mcap(ymd: str) -> bool:
    """해당 ymd의 KOSPI/KOSDAQ 데이터가 '양수 합계'인지(0 프레임 아님) 확인."""
    # OHLCV 거래대금 합계 검사
    o_valid = False
    for m in ("KOSPI", "KOSDAQ"):
        try:
            o = stock.get_market_ohlcv_by_ticker(ymd, market=m)
        except Exception:
            o = None
        if o is not None and not o.empty and "거래대금" in o.columns and _safe_sum(o["거래대금"]) > 0:
            o_valid = True
            break

    # 시총 합계 검사
    m_valid = False
    for m in ("KOSPI", "KOSDAQ"):
        try:
            mc = stock.get_market_cap_by_ticker(ymd, market=m)
        except Exception:
            mc = None
        if mc is not None and not mc.empty and "시가총액" in mc.columns and _safe_sum(mc["시가총액"]) > 0:
            m_valid = True
            break

    return o_valid and m_valid

def resolve_trade_date() -> str:
    """
    장마감 집계 시차/휴일을 고려: 오늘 18시 이전이면 전일부터,
    이후에도 데이터가 '양수 합계'가 아닐 경우 하루씩 과거 영업일로 롤백.
    """
    now = datetime.now(KST)
    d = now.date()
    if now.hour < 18:
        d = d - timedelta(days=1)

    for _ in range(10):
        ymd = d.strftime("%Y%m%d")
        if _has_ohlcv_and_mcap(ymd):
            return ymd
        d = d - timedelta(days=1)
    # 최후의 보루
    return (now - timedelta(days=1)).strftime("%Y%m%d")

# ------------------------------- 시총 맵 -------------------------------
def build_mcap_map() -> tuple[dict, str]:
    """
    유효한 기준일을 찾아 KOSPI/KOSDAQ 시총을 일괄 조회해 억원 단위 dict로 반환.
    0/NaN 합계면 하루 롤백. 대표 샘플(005930)도 0이면 롤백.
    """
    today = datetime.now(KST).date()
    # 18시 이전이면 전일 기준으로 시작
    if datetime.now(KST).hour < 18:
        today = today - timedelta(days=1)

    d = today
    for _ in range(10):
        use = d.strftime("%Y%m%d")
        any_ok = False
        mcap: dict[str, float] = {}
        for m in ("KOSPI", "KOSDAQ"):
            try:
                df = stock.get_market_cap_by_ticker(use, market=m)
            except Exception:
                df = None
            if df is None or df.empty or "시가총액" not in df.columns:
                continue
            # 0 프레임 방지: 합계가 양수일 때만 반영
            raw = pd.to_numeric(df["시가총액"], errors="coerce")
            if raw.fillna(0).sum() <= 0:
                continue

            tickers = df.index.astype(str).str.zfill(6)
            vals = (raw / 1e8).astype(float)  # 억원
            mcap.update({t: float(v) for t, v in zip(tickers, vals)})
            any_ok = True

        if any_ok and len(mcap) > 0:
            sample = mcap.get("005930", np.nan)  # 삼성전자
            if not (isinstance(sample, (int, float)) and sample > 0):
                # 대표도 0/NaN → 하루 더 롤백
                d = d - timedelta(days=1)
                continue
            log(f"🧭 시총 기준일 확정: {use} · 종목수 {len(mcap):,}")
            log(f"   시총 샘플 005930: {mcap['005930']:.1f}억")
            return mcap, use

        d = d - timedelta(days=1)

    log("⚠️ 시총 기준일 확보 실패 → 빈 맵 반환")
    return {}, today.strftime("%Y%m%d")

def get_mcap_eok_from_map(mcap_map: dict, ticker: str) -> float:
    v = mcap_map.get(str(ticker).zfill(6))
    if v is None or (isinstance(v, (int, float)) and v <= 0):
        return np.nan  # 0은 NaN 처리
    return float(v)

# ------------------------------- 상위 TV 선정 -------------------------------
def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    frames = []
    for m in ("KOSPI", "KOSDAQ"):
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m)
        except Exception as e:
            log(f"⚠️ {m} TV 집계 실패: {e}")
            df = None
        if df is None or df.empty:
            continue
        df = df.reset_index()  # 티커 → 컬럼
        if "티커" in df.columns:
            df.rename(columns={"티커": "종목코드"}, inplace=True)
        if "거래대금(원)" not in df.columns and "거래대금" in df.columns:
            df.rename(columns={"거래대금": "거래대금(원)"}, inplace=True)
        frames.append(df[["종목코드", "거래대금(원)"]])

    if not frames:
        raise RuntimeError("거래대금 상위 집계에 사용할 데이터가 없습니다.")

    tv_df = pd.concat(frames, ignore_index=True)
    tv_df["종목코드"] = tv_df["종목코드"].astype(str).str.zfill(6)
    tv_df["거래대금(원)"] = pd.to_numeric(tv_df["거래대금(원)"], errors="coerce").fillna(0)
    tv_df = tv_df[tv_df["거래대금(원)"] > 0]  # 0원 거래(주말치) 제거
    tv_df = tv_df.sort_values("거래대금(원)", ascending=False).head(top_n).reset_index(drop=True)
    return tv_df

def get_market_sets(date_yyyymmdd: str):
    def _get(market):
        try:
            return set(stock.get_market_ticker_list(date_yyyymmdd, market=market))
        except Exception:
            return set()
    return _get("KOSPI"), _get("KOSDAQ")

# ------------------------------- 종목명 맵 -------------------------------
def get_name_map_cached(date_yyyymmdd: str) -> dict:
    ensure_dir(OUT_DIR)
    map_path = os.path.join(OUT_DIR, "krx_codes.csv")
    mp = {}
    # 캐시 읽기
    if os.path.exists(map_path):
        try:
            df = pd.read_csv(map_path, dtype={"종목코드": "string"}, encoding="utf-8")
            for _, r in df.iterrows():
                mp[str(r["종목코드"]).zfill(6)] = r.get("종목명", "")
        except Exception:
            mp = {}

    if mp:
        return mp

    # 신선 생성
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

# ------------------------------- 메인 로직 -------------------------------
def main():
    log("전종목 수집 시작…")

    # 0) 시총 기준일/맵 준비 (먼저 확보해서 필터 안정)
    mcap_map, mcap_ymd = build_mcap_map()
    log(f"🧭 시총 기준일 확정: {mcap_ymd} · 종목수 {len(mcap_map):,}")

    # 1) 거래 기준일 결정
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일 확정: {trade_ymd}")

    # 2) 상위 거래대금 종목
    log("🔍 거래대금 상위 종목 선정 중…")
    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ TOP {len(tickers)} 종목 선정 완료")

    # 3) 시장 구분/종목명 맵
    kospi_set, kosdaq_set = get_market_sets(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)

    # 4) 각 종목 OHLCV 60일 + 지표/점수/추천가
    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS * 2)
    start_s, end_s = start_dt.strftime("%Y%m%d"), trade_ymd

    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty:
                continue

            ohlcv = ohlcv.reset_index().rename(columns={"index": "날짜"})
            ohlcv["날짜"] = pd.to_datetime(ohlcv["날짜"])

            # 최근 60거래일만 사용
            ohlcv = ohlcv.tail(LOOKBACK_DAYS)
            if len(ohlcv) < 20:
                continue

            close = ohlcv["종가"].astype(float)
            high  = ohlcv["고가"].astype(float)
            low   = ohlcv["저가"].astype(float)
            vol   = ohlcv["거래량"].astype(float)

            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean()
            atr14 = calc_atr(high, low, close, 14)
            rsi14 = calc_rsi(close, 14)

            ema12 = ema(close, 12); ema26 = ema(close, 26)
            macd_line   = ema12 - ema26
            macd_signal = ema(macd_line, 9)
            macd_hist   = macd_line - macd_signal
            macd_slope  = macd_hist.diff()

            vol_z = vol / (vol.rolling(20).mean())
            disp  = (close / ma20 - 1.0) * 100  # 乖離%

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
            ret5  = (close.pct_change(5 ).iloc[-1] * 100) if len(close) >= 6  else np.nan
            ret10 = (close.pct_change(10).iloc[-1] * 100) if len(close) >= 11 else np.nan

            # 시장/이름/거래대금/시총
            mkt = "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타")
            name = name_map.get(str(t).zfill(6), "") or stock.get_market_ticker_name(t)
            tv_eok = float(top_df.loc[top_df["종목코드"] == t, "거래대금(원)"].values[0]) / 1e8
            mcap_eok = get_mcap_eok_from_map(mcap_map, t)

            # 필터: 개잡주 컷
            if tv_eok < MIN_TURNOVER_EOK or (not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK):
                continue

            # EBS 점수
            score = 0
            reason = []
            if RSI_LOW <= rsi_v <= RSI_HIGH:
                score += 1; reason.append("RSI 45~65")
            if macd_sl > 0:
                score += 1; reason.append("MACD상승")
            if not np.isnan(disp_v) and -1.0 <= disp_v <= 4.0:
                score += 1; reason.append("MA20 근처")
            if v_z > 1.2:
                score += 1; reason.append("거래량증가")
            if not np.isnan(m20) and not np.isnan(m60) and m20 > m60:
                score += 1; reason.append("상승구조")
            if macd_h > 0:
                score += 1; reason.append("MACD>sig")
            if not np.isnan(ret5) and ret5 < 10:
                score += 1; reason.append("과열아님")

            if np.isnan(atr) or np.isnan(m20) or atr <= 0:
                continue

            # 추천가/손절/목표 (보수적 엔트리)
            buy  = min(c, m20 * 1.01)
            stop = max(buy - 1.5 * atr, m20 * 0.97)
            tgt1 = buy + (buy - stop) * 1.0
            tgt2 = buy + (buy - stop) * 2.0

            buy  = round_to_tick(buy)
            stop = round_to_tick(stop)
            tgt1 = round_to_tick(tgt1)
            tgt2 = round_to_tick(tgt2)

            rows.append({
                "시장": mkt,
                "종목명": name,
                "종목코드": str(t).zfill(6),
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
                "통과": "초입" if score >= PASS_SCORE else "",
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
        raise RuntimeError("수집 결과가 비었습니다.")

    df_out = pd.DataFrame(rows).sort_values(
        ["EBS", "거래대금(억원)"], ascending=[False, False]
    ).reset_index(drop=True)

    ensure_dir(OUT_DIR)
    path_day_utf8    = os.path.join(OUT_DIR, f"recommend_{trade_ymd}.csv")
    path_latest_utf8 = os.path.join(OUT_DIR, "recommend_latest.csv")

    df_out.to_csv(path_day_utf8, index=False, encoding=UTF8)
    df_out.to_csv(path_latest_utf8, index=False, encoding=UTF8)

    log(f"💾 saved: {path_day_utf8} ({len(df_out):,} rows)")
    log(f"💾 saved: {path_latest_utf8} ({len(df_out):,} rows)")

if __name__ == "__main__":
    main()
