# -*- coding: utf-8 -*-
"""
LDY Pro Trader: Nightly Collector (KRX) v3.4.1
- 매일 장마감 후: 유동성 상위(TV 상위) 종목 n개 선정
- 각 종목 60거래일 OHLCV 수집 후 당일 스냅샷 지표/점수(EBS) 계산
- 추천매수/매도/손절 가격 컬럼 포함 CSV 저장 (UTF-8 BOM)
- 시총 NaN 방지: 시총맵을 전영업일까지 롤백해서 구성
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
LOOKBACK_DAYS = 60         # 조회일수
TOP_N = 600                # 거래대금 상위 샘플 크기(300~800 권장)
MIN_TURNOVER_EOK = 50      # 거래대금 하한(억원)
MIN_MCAP_EOK = 1000        # 시총 하한(억원)
RSI_LOW, RSI_HIGH = 45, 65 # RSI 범위
PASS_SCORE = 4             # 통과점수(최종 EBS)
SLEEP_SEC = 0.05           # API call 간 딜레이(안정성)
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
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calc_atr(high, low, close, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def round_to_tick(price: float) -> int:
    # 간단 틱(10원 단위)
    return int(round(price / 10.0) * 10)

# ------------------------------- 기준일 결정 -------------------------------
def _has_ohlcv_and_mcap(ymd: str) -> bool:
    ok1 = ok2 = False
    try:
        df1 = stock.get_market_ohlcv_by_ticker(ymd, market="KOSPI")
        ok1 = df1 is not None and not df1.empty
    except:
        pass
    try:
        df2 = stock.get_market_cap_by_ticker(ymd, market="KOSPI")
        ok2 = df2 is not None and not df2.empty and "시가총액" in df2.columns
    except:
        pass
    return ok1 and ok2

def resolve_trade_date() -> str:
    """
    장마감 집계 시차 + 주말/휴일 고려.
    OHLCV/시총 모두 나오는 가장 가까운 영업일을 뒤로 탐색.
    반환: 'YYYYMMDD'
    """
    now = datetime.now(KST)
    d = now.date()
    if now.hour < 18:
        d = d - timedelta(days=1)

    for _ in range(10):
        ymd = d.strftime("%Y%m%d")
        if _has_ohlcv_and_mcap(ymd):
            return ymd
        d -= timedelta(days=1)
    return (now - timedelta(days=1)).strftime("%Y%m%d")

# ------------------------------- 시총 맵 -------------------------------
def build_mcap_map(date_yyyymmdd: str) -> dict:
    """
    주말/휴일 등 공백을 대비해 최대 7일 뒤로 굴리며 시총 맵 구성.
    단위: 억원
    """
    use = date_yyyymmdd
    for _ in range(7):
        mcap = {}
        any_ok = False
        for mk in ["KOSPI", "KOSDAQ", "KONEX"]:
            try:
                df = stock.get_market_cap_by_ticker(use, market=mk)
                if df is None or df.empty or "시가총액" not in df.columns:
                    continue
                any_ok = True
                # index: 티커, columns: ... , "시가총액"
                part = (pd.to_numeric(df["시가총액"], errors="coerce") / 1e8)
                part = part.replace([np.inf, -np.inf], np.nan).dropna()
                for t, v in part.items():
                    mcap[str(t).zfill(6)] = float(v)
            except:
                pass
        if any_ok and len(mcap) > 0:
            log(f"🧭 시총 기준일 확정: {use} · 종목수 {len(mcap):,}")
            if "005930" in mcap:
                log(f"   시총 샘플 005930: {mcap['005930']:.1f}억")
            return mcap
        # 하루씩 뒤로
        use = (datetime.strptime(use, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    log("⚠️ 시총 맵 비어 있음 — 일부 종목 시총 NaN 가능")
    return {}

def get_mcap_eok_from_map(mcap_map: dict, ticker: str) -> float:
    v = mcap_map.get(str(ticker).zfill(6))
    return float(v) if v is not None else np.nan

# ------------------------------- 이름/시장 맵 -------------------------------
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

# ------------------------------- 상위 TV 선정 -------------------------------
def pick_top_by_trading_value(date_yyyymmdd: str, top_n: int) -> pd.DataFrame:
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m)
            if df is None or df.empty:
                continue
            df = df.reset_index()  # '티커' -> 컬럼
            if "티커" in df.columns:
                df.rename(columns={"티커": "종목코드"}, inplace=True)
            if "거래대금(원)" not in df.columns and "거래대금" in df.columns:
                df.rename(columns={"거래대금": "거래대금(원)"}, inplace=True)
            frames.append(df[["종목코드", "거래대금(원)"]])
        except Exception as e:
            log(f"⚠️ {m} TV 집계 실패: {e}")

    if not frames:
        raise RuntimeError("거래대금 상위 집계에 사용할 데이터가 없습니다.")

    tv_df = pd.concat(frames, ignore_index=True)
    tv_df["종목코드"] = tv_df["종목코드"].astype(str).str.zfill(6)
    tv_df["거래대금(원)"] = pd.to_numeric(tv_df["거래대금(원)"], errors="coerce").fillna(0)
    tv_df = tv_df.sort_values("거래대금(원)", ascending=False).head(top_n).reset_index(drop=True)
    return tv_df

# ------------------------------- CP949 안전 치환 -------------------------------
def make_cp949_safe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    # 컬럼명 치환
    df2.columns = [c.replace("乖離%", "괴리_%") for c in df2.columns]
    # 값 치환
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

    # 1) 기준일/시총맵
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일 확정: {trade_ymd}")
    mcap_map = build_mcap_map(trade_ymd)

    # 2) 상위 거래대금 종목
    log("🔍 거래대금 상위 종목 선정 중…")
    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ TOP {len(tickers)} 종목 선정 완료")

    # 3) 시장 구분/종목명 맵
    kospi_set, kosdaq_set = get_market_map(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)

    # 4) 각 종목 OHLCV 60일 + 지표/점수/추천가
    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS*2)
    start_s = start_dt.strftime("%Y%m%d")
    end_s = trade_ymd

    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty:
                continue

            ohlcv = ohlcv.reset_index().rename(columns={"index": "날짜"})
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

            ema12 = ema(close, 12)
            ema26 = ema(close, 26)
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

            mkt = "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타")
            name = name_map.get(str(t).zfill(6), "") or stock.get_market_ticker_name(t)
            tv_eok = float(top_df.loc[top_df["종목코드"] == t, "거래대금(원)"].values[0]) / 1e8
            mcap_eok = get_mcap_eok_from_map(mcap_map, t)  # 맵에서 가져오기

            # 필터: 개잡주 컷
            if tv_eok < MIN_TURNOVER_EOK or (not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK):
                continue

            # EBS 점수 (급등 '초입' 스코어)
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
                score += 1; reason.append("MACD>0")
            if not np.isnan(ret5) and ret5 < 10:
                score += 1; reason.append("5d<10%")

            if np.isnan(atr) or np.isnan(m20):
                continue

            # 보수적 엔트리/손절/목표
            buy  = min(c, m20 * 1.01)
            stop = buy - 1.5 * atr
            tgt1 = buy + (buy - stop) * 1.0
            tgt2 = buy + (buy - stop) * 2.0

            buy  = round_to_tick(buy)
            stop = max(round_to_tick(stop), round_to_tick(m20 * 0.97))
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
                "통과": "🚀" if score >= PASS_SCORE else "",
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

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["EBS", "거래대금(억원)"], ascending=[False, False]).reset_index(drop=True)

    ensure_dir(OUT_DIR)
    path_day_utf8    = os.path.join(OUT_DIR, f"recommend_{trade_ymd}.csv")
    path_latest_utf8 = os.path.join(OUT_DIR, "recommend_latest.csv")

    df_out.to_csv(path_day_utf8, index=False, encoding=UTF8)
    df_out.to_csv(path_latest_utf8, index=False, encoding=UTF8)

    # CP949 안전본(선택): 필요 시 주석 해제
    # safe = make_cp949_safe(df_out)
    # safe.to_csv(os.path.join(OUT_DIR, f"recommend_{trade_ymd}_cp949.csv"), index=False, encoding="cp949", errors="ignore")

    log(f"✅ 저장 완료: {path_day_utf8} · {path_latest_utf8}")
    log(f"   행수: {len(df_out):,}, 통과(EBS≥{PASS_SCORE}): {int((df_out['EBS']>=PASS_SCORE).sum()):,}")

if __name__ == "__main__":
    main()
