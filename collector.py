# -*- coding: utf-8 -*-
"""
LDY Pro Trader: Nightly Collector (KRX)  — EBS+ 강화 버전
- 장마감 후: 거래대금 상위 종목 선정 → 60거래일 OHLCV 지표/점수 계산
- 기본 EBS(7요소) + 정밀 필터(EBS+): RS, VCP/수렴, UDVR/OBV, 레짐 등
- 추천 매수/손절/목표가 산출, UTF-8/CP949 안전 CSV 동시 저장
- pykrx 시그니처 차이 회피: TV 상위는 get_market_ohlcv_by_ticker() 사용
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

LOOKBACK_DAYS = 60          # 종목별 사용 일수(60)
TOP_N = 600                 # 거래대금 상위 추출 개수
MIN_TURNOVER_EOK = 50       # 거래대금 하한(억원)
MIN_MCAP_EOK = 1000         # 시총 하한(억원)
RSI_LOW, RSI_HIGH = 45, 65  # RSI 통과 범위
PASS_SCORE = 4              # 기존 EBS 통과 점수
SLEEP_SEC = 0.04            # API 콜 간 딜레이
OUT_DIR = "data"
UTF8 = "utf-8-sig"

# EBS+ 가중치 (총점 0~100 스케일에 사용)
W1_EBS      = 6   # EBS
W2_RS       = 4   # RS_slope, RS_high50
W3_BASE     = 5   # VCP, 피봇근접, 볼륨돌파
W4_ACCUM    = 3   # UDVR, OBV_high
W5_WEEKLY   = 2   # 주봉 정렬(간이)
W6_REGIME   = 2   # 시장 레짐 보너스

# ------------------------------- 유틸 -------------------------------
def log(msg: str):
    print(f"[{datetime.now(KST)}] {msg}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def calc_rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - 100 / (1 + rs)

def calc_atr(high, low, close, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def round_to_tick(price: float) -> int:
    # 10원 틱 반올림
    return int(round(price / 10.0) * 10)

# ------------------------------- 기준일 결정 -------------------------------
def resolve_trade_date() -> str:
    """
    장마감 집계 시차 고려: 당일 18시 이전이면 전일로,
    데이터가 비면 하루씩 뒤로 가며 최근 영업일을 찾음.
    """
    now = datetime.now(KST)
    d = now.date()
    if now.hour < 18:
        d -= timedelta(days=1)

    for _ in range(7):
        ymd = d.strftime("%Y%m%d")
        try:
            tmp = stock.get_market_ohlcv_by_ticker(ymd, market="KOSPI")
            if tmp is not None and not tmp.empty and ("거래대금" in tmp.columns or "거래대금(원)" in tmp.columns):
                return ymd
        except Exception:
            pass
        d -= timedelta(days=1)
    # 최후의 보루
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
            if "거래대금(원)" not in df.columns:
                # pykrx 보통 '거래대금' 명칭 → 통일
                if "거래대금" in df.columns:
                    df = df.rename(columns={"거래대금": "거래대금(원)"})
            frames.append(df[["종목코드", "거래대금(원)"]])
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
        return float(cap["시가총액"].iloc[0]) / 1e8
    except Exception:
        return np.nan

# ------------------------------- 지수/레짐/RS -------------------------------
def find_index_code(date_yyyymmdd: str, contains: str = "코스피") -> str | None:
    """이름에 '코스피' 포함된 지수 코드 탐색."""
    try:
        codes = stock.get_index_ticker_list(date_yyyymmdd)
        for c in codes:
            try:
                nm = stock.get_index_ticker_name(c)
                if nm and contains in nm and "200" not in nm:
                    return c
            except Exception:
                continue
    except Exception:
        pass
    return None

def get_index_close(start_s: str, end_s: str, index_code: str | None) -> pd.Series | None:
    if not index_code:
        return None
    try:
        idx = stock.get_index_ohlcv_by_date(start_s, end_s, index_code)
        return idx["종가"].astype(float)
    except Exception:
        return None

def get_regime_signal(start_s: str, end_s: str, date_yyyymmdd: str) -> dict:
    """
    간이 레짐: 코스피/코스닥 각각 MA20 > MA60 여부.
    regime_good: 둘 중 하나라도 True면 True
    """
    out = {"kospi_ma_up": None, "kosdaq_ma_up": None, "regime_good": None, "kospi_close": None}
    kospi_code = find_index_code(date_yyyymmdd, "코스피")
    kosdaq_code = find_index_code(date_yyyymmdd, "코스닥")

    kospi_close = get_index_close(start_s, end_s, kospi_code)
    kosdaq_close = get_index_close(start_s, end_s, kosdaq_code)

    def ma_up(s: pd.Series):
        if s is None or len(s) < 60: return None
        ma20 = s.rolling(20).mean().iloc[-1]
        ma60 = s.rolling(60).mean().iloc[-1]
        if np.isnan(ma20) or np.isnan(ma60): return None
        return ma20 > ma60

    out["kospi_ma_up"] = ma_up(kospi_close)
    out["kosdaq_ma_up"] = ma_up(kosdaq_close)
    out["regime_good"] = any([x for x in [out["kospi_ma_up"], out["kosdaq_ma_up"]] if isinstance(x, (bool, np.bool_))])
    out["kospi_close"] = kospi_close  # RS 계산용으로 전달
    return out

# ------------------------------- CP949 안전 치환 -------------------------------
def make_cp949_safe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    # 컬럼명
    df2.columns = [c.replace("乖離%", "괴리_%") for c in df2.columns]
    # 값 치환
    if "근거" in df2.columns and df2["근거"].dtype == object:
        df2["근거"] = (
            df2["근거"]
            .str.replace("MACD>sig", "MACD상방", regex=False)
            .str.replace("MA20 근처", "MA20근접", regex=False)
        )
    return df2

# ------------------------------- 메인 로직 -------------------------------
def main():
    log("전종목 수집 시작…")

    # 1) 기준일
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일 확정: {trade_ymd}")

    # 2) 상위 거래대금
    log("🔍 거래대금 상위 종목 선정 중…")
    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()
    log(f"✅ TOP {len(tickers)} 종목 선정 완료")

    # 3) 시장/이름 맵, 레짐/지수
    kospi_set, kosdaq_set = get_market_map(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)

    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS * 2)
    start_s = start_dt.strftime("%Y%m%d")
    end_s = trade_ymd

    regime = get_regime_signal(start_s, end_s, trade_ymd)
    kospi_close = regime.get("kospi_close")
    regime_good = bool(regime.get("regime_good"))

    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty:
                continue
            ohlcv = ohlcv.reset_index().rename(columns={"index": "날짜"})
            ohlcv["날짜"] = pd.to_datetime(ohlcv["날짜"])

            # 최근 60거래일만
            ohlcv = ohlcv.tail(LOOKBACK_DAYS)
            if len(ohlcv) < 20:
                continue

            close = ohlcv["종가"].astype(float)
            high  = ohlcv["고가"].astype(float)
            low   = ohlcv["저가"].astype(float)
            vol   = ohlcv["거래량"].astype(float)

            ma20  = close.rolling(20).mean()
            ma60  = close.rolling(60).mean()
            atr14 = calc_atr(high, low, close, 14)
            rsi14 = calc_rsi(close, 14)

            e12 = ema(close, 12)
            e26 = ema(close, 26)
            macd_line   = e12 - e26
            macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
            macd_hist   = macd_line - macd_signal
            macd_slope  = macd_hist.diff()

            vol_avg20 = vol.rolling(20).mean()
            vol_z     = (vol - vol_avg20) / vol.rolling(20).std()
            disp      = (close / ma20 - 1.0) * 100

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
            ret5    = (close.pct_change(5).iloc[-1] * 100) if len(close) >= 6  else np.nan
            ret10   = (close.pct_change(10).iloc[-1] * 100) if len(close) >= 11 else np.nan

            mkt = "KOSPI" if t in kospi_set else ("KOSDAQ" if t in kosdaq_set else "기타")
            name = name_map.get(str(t).zfill(6), "") or stock.get_market_ticker_name(t)
            tv_eok = float(top_df.loc[top_df["종목코드"] == t, "거래대금(원)"].values[0]) / 1e8
            mcap_eok = get_mcap_eok(trade_ymd, t)

            # 1차 컷: 거래대금/시총
            if tv_eok < MIN_TURNOVER_EOK:
                continue
            if not np.isnan(mcap_eok) and mcap_eok < MIN_MCAP_EOK:
                continue

            # 기존 EBS (7요소)
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

            # 2차 컷: 최근 저점/쿨다운 (거짓 양성 감소)
            # 최근 60저점이 최근 10거래일 내면 제외
            try:
                low_pos = int(np.nanargmin(low.values[-60:])) if len(low) >= 60 else int(np.nanargmin(low.values))
                recent_low_too_soon = (len(low) - 1 - low_pos) <= 10
            except Exception:
                recent_low_too_soon = False

            # 전일 급등(+10%)이면 하루 쿨다운
            try:
                day_chg = close.pct_change().iloc[-1]
                big_gap_yday = day_chg > 0.10
            except Exception:
                big_gap_yday = False

            if recent_low_too_soon or big_gap_yday:
                # 정밀 후보군에서 제외
                continue

            # RS 컨펌 (지수 대비)
            rs_slope20, rs_high50 = None, None
            if kospi_close is not None and len(kospi_close) >= len(close):
                kc = kospi_close.iloc[-len(close):].reset_index(drop=True)
                rs_line = close.reset_index(drop=True) / kc
                if len(rs_line) >= 50:
                    rs_slope20 = rs_line.pct_change(20).iloc[-1]
                    rs_high50  = (rs_line.iloc[-50:].max() - rs_line.iloc[-1]) / rs_line.iloc[-1] <= 0.005  # 0.5% 이내

            # 베이스 품질: VCP/수렴, 피봇근접(20일 고점 -3%), 볼륨돌파(≥1.5x)
            vcp_ok = False
            if len(close) >= 40:
                atr20 = calc_atr(high, low, close, 20)
                if len(atr20.dropna()) >= 20:
                    wk = 5
                    seg = [atr20.iloc[-1 - wk * i: -wk * i if i > 0 else None].mean() for i in range(4)]
                    vcp_ok = all(seg[i] <= seg[i + 1] for i in range(len(seg) - 1))
            pivot20  = close.rolling(20).max().iloc[-1] if len(close) >= 20 else np.nan
            near_pivot = (not np.isnan(pivot20)) and (close.iloc[-1] >= pivot20 * 0.97)
            vol_boost  = (not np.isnan(vol_avg20.iloc[-1])) and (vol.iloc[-1] >= vol_avg20.iloc[-1] * 1.5)

            # 수급/축적: UDVR(50D), OBV 고점
            chg = close.diff()
            up_vol = vol.where(chg > 0, 0).rolling(50).sum()
            dn_vol = vol.where(chg < 0, 0).rolling(50).sum()
            try:
                udvr = float((up_vol / dn_vol.replace(0, np.nan)).iloc[-1])
            except Exception:
                udvr = np.nan
            udvr_ok = (not np.isnan(udvr)) and udvr > 1.3

            obv = (np.sign(chg).fillna(0) * vol).cumsum()
            obv_high = False
            if len(obv.dropna()) >= 50:
                mx = obv.rolling(50).max().iloc[-1]
                obv_high = (not np.isnan(mx)) and (obv.iloc[-1] >= mx * 0.995)

            # 간이 주봉 정렬(데이터 짧아 약식): 20일선 기울기 양수면 +1로 대체
            weekly_pos = False
            if len(ma20.dropna()) >= 2:
                weekly_pos = (ma20.iloc[-1] - ma20.iloc[-2]) > 0

            # EBS+ 점수 (0~100)
            ebs = int(score)
            pts = 0.0
            # EBS (정규화: 7점 만점 → 가중치)
            pts += W1_EBS * (ebs / 7)

            # RS 2요소
            if rs_slope20 is not None and rs_slope20 > 0: pts += W2_RS
            if rs_high50: pts += W2_RS

            # 베이스 품질 3요소
            if vcp_ok:     pts += W3_BASE
            if near_pivot: pts += W3_BASE
            if vol_boost:  pts += W3_BASE

            # 축적 2요소
            if udvr_ok:   pts += W4_ACCUM
            if obv_high:  pts += W4_ACCUM

            # 주봉
            if weekly_pos: pts += W5_WEEKLY

            # 레짐
            if regime_good: pts += W6_REGIME

            total_possible = W1_EBS * 1.0 + W2_RS * 2 + W3_BASE * 3 + W4_ACCUM * 2 + W5_WEEKLY * 1 + W6_REGIME * 1
            ebs_plus = round(100.0 * pts / total_possible, 1)

            # 휴리스틱 Hit 확률(과신 금지: UI용 참고치)
            hit_prob = round(max(15.0, min(95.0, 0.9 * ebs_plus - 10.0)), 1)

            # 추천가 (보수적 규칙)
            if np.isnan(atr) or np.isnan(m20) or atr <= 0:
                buy = tgt1 = tgt2 = stp = np.nan
            else:
                band_lo, band_hi = m20 - 0.5 * atr, m20 + 0.5 * atr
                entry = min(max(c, band_lo), band_hi)
                # 돌파형 엔트리(선택): pivot20 + n틱 → 필요시 교체
                buy  = round_to_tick(entry)
                stp  = max(round_to_tick(entry - 1.5 * atr), round_to_tick(m20 * 0.97))
                tgt1 = round_to_tick(entry + 1.0 * atr)
                tgt2 = round_to_tick(entry + 1.8 * atr)

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

                "EBS": ebs,
                "근거": ", ".join([r for r in reason if r]),

                # 정밀 필터 시그널
                "RS_slope20": None if rs_slope20 is None or np.isnan(rs_slope20) else round(float(rs_slope20), 4),
                "RS_high50": bool(rs_high50) if rs_high50 is not None else None,
                "VCP_ok": bool(vcp_ok),
                "Pivot_near": bool(near_pivot),
                "Vol_breakout": bool(vol_boost),
                "UDVR_50": None if np.isnan(udvr) else round(float(udvr), 2),
                "UDVR_ok": bool(udvr_ok),
                "OBV_high": bool(obv_high),
                "Weekly_pos": bool(weekly_pos),
                "Regime_good": bool(regime_good),

                # EBS+
                "EBS_PLUS": ebs_plus,
                "HitProb_%(heuristic)": hit_prob,

                # 추천가
                "추천매수가": buy,
                "추천매도가1": tgt1,
                "추천매도가2": tgt2,
                "손절가": stp,
            })
        except Exception as e:
            log(f"⚠️ {t} 처리 실패: {e}")
        time.sleep(SLEEP_SEC)

    if not rows:
        raise RuntimeError("수집 결과가 비었습니다.")

    df_out = pd.DataFrame(rows)
    # 기본 정렬: EBS+ → EBS → 거래대금
    sort_cols = [c for c in ["EBS_PLUS", "EBS", "거래대금(억원)"] if c in df_out.columns]
    df_out = df_out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    ensure_dir(OUT_DIR)
    path_day_utf8    = os.path.join(OUT_DIR, f"recommend_{trade_ymd}.csv")
    path_latest_utf8 = os.path.join(OUT_DIR, "recommend_latest.csv")

    df_out.to_csv(path_day_utf8, index=False, encoding=UTF8)
    df_out.to_csv(path_latest_utf8, index=False, encoding=UTF8)

    # CP949 안전본 저장(엑셀 호환)
    df_cp = make_cp949_safe(df_out)
    path_day_cp    = os.path.join(OUT_DIR, f"recommend_{trade_ymd}_cp949.csv")
    path_latest_cp = os.path.join(OUT_DIR, "recommend_latest_cp949.csv")
    df_cp.to_csv(path_day_cp, index=False, encoding="cp949", errors="replace")
    df_cp.to_csv(path_latest_cp, index=False, encoding="cp949", errors="replace")

    log(f"💾 저장 완료: {path_day_utf8} (+ latest, cp949 동시 저장)")

if __name__ == "__main__":
    main()
