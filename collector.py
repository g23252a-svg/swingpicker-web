# collector_v6_0.py
# -*- coding: utf-8 -*-
"""
LDY Pro Trader Collector v6.0
- 안정성, 예외처리, 지표정확성 개선
- calc_rsi / calc_mfi 안정화
- RR 계산 방어 및 ret_5d/ret_10d 반영
- build_mcap_map 예외·컬럼 방어, 로깅 강화
- Telegram 전송 안전성 개선
- (선택) 병렬 수집을 위한 구조 준비
"""

import os
import time
import math
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from pykrx import stock
from tqdm import tqdm
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------- 환경 / 설정 -------------------------------
KST = timezone(timedelta(hours=9))
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_ID = os.environ.get("TG_ID")

LOOKBACK_DAYS = 250          # 과거 조회 일수
TOP_N = 600                  # 거래대금 상위 n
MIN_TURNOVER_EOK = 50       # 최소 거래대금 (억원)
MIN_MCAP_EOK = 1000         # 최소 시가총액 (억원)
RSI_LOW, RSI_HIGH = 45, 65  # RSI 허용 범위
PASS_EBS = 4                # EBS 통과 기준
SLEEP_SEC = 0.01            # API rate control 기본 대기
OUT_DIR = "data"
UTF8 = "utf-8-sig"

# 가중치 / 패널티 (현재 Collector에서는 레이블용)
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = 0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = 6.0, 6.0, 4.0
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 4.0, 2.0

# ------------------------------- 유틸 -------------------------------
def log(msg: str):
    print(f"[{datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def nz_num(s):
    return pd.to_numeric(s, errors="coerce")

def round_to_tick(price):
    # tick rule: 한국증권시장 단위 근사 (간단 버전)
    if price < 2000: t = 1
    elif price < 5000: t = 5
    elif price < 20000: t = 10
    elif price < 50000: t = 50
    elif price < 200000: t = 100
    elif price < 500000: t = 500
    else: t = 1000
    try:
        return int(round(price / t) * t)
    except:
        return int(price or 0)

# ------------------------------- 지표 함수 (안정화) -------------------------------
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def calc_rsi(close, period=14):
    # Wilder's smoothing (EMA-like)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # use Wilder smoothing
    up_ewm = up.ewm(alpha=1/period, adjust=False).mean()
    down_ewm = down.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    rs = up_ewm / down_ewm
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_mfi(high, low, close, vol, period=14):
    # 안정적인 MFI 구현
    tp = (high + low + close) / 3
    rmf = tp * vol
    delta_tp = tp.diff()
    pos_rmf = rmf.where(delta_tp > 0, 0.0)
    neg_rmf = rmf.where(delta_tp < 0, 0.0).abs()
    pos_sum = pos_rmf.rolling(period).sum()
    neg_sum = neg_rmf.rolling(period).sum().replace(0, np.nan)
    rs = pos_sum / neg_sum
    mfi = 100 - (100 / (1 + rs))
    return mfi.fillna(50)

def calc_atr(high, low, close, period=14):
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# ------------------------------- 보조 정규화 함수 -------------------------------
def cap_q(s, q=90, floor=1.0):
    arr = nz_num(s)
    arr_nonan = arr.dropna()
    if arr_nonan.size == 0:
        return float(floor)
    try:
        val = float(np.nanpercentile(arr_nonan, q))
        if not np.isfinite(val):
            return float(floor)
        return max(val, float(floor))
    except Exception:
        return float(floor)

def pct_norm_pos(s, q=90, floor=1.0):
    s_num = nz_num(s).clip(lower=0)
    cap = cap_q(s_num, q=q, floor=floor)
    return np.clip(s_num / cap, 0, 1)

def inv_dist_norm(dist, cap):
    return np.clip(1 - (nz_num(dist) / (cap if cap != 0 else 1)), 0, 1)

# ------------------------------- 데이터 수집 보조 -------------------------------
def _has_ohlcv_and_mcap(ymd):
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            o = stock.get_market_ohlcv_by_ticker(ymd, market=m)
            if o is not None and not o.empty and "거래대금" in o.columns and np.nansum(o["거래대금"]) > 0:
                return True
        except Exception:
            continue
    return False

def resolve_trade_date():
    d = datetime.now(KST).date()
    # 장마감 이후 데이터 수집 가정: 오후 18시 이전이면 전일 데이터 기준
    if datetime.now(KST).hour < 18:
        d -= timedelta(days=1)
    for _ in range(14):
        ymd = d.strftime("%Y%m%d")
        if _has_ohlcv_and_mcap(ymd):
            return ymd
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")

def build_mcap_map():
    # 최근 10일 사이에서 시가총액 맵 생성 (억 단위로 표준화)
    for offset in range(0, 12):
        d = (datetime.now(KST).date() - timedelta(days=offset))
        ymd = d.strftime("%Y%m%d")
        try:
            df_kp = stock.get_market_cap_by_ticker(ymd, market='KOSPI')
            df_kq = stock.get_market_cap_by_ticker(ymd, market='KOSDAQ')
            df = pd.concat([df_kp, df_kq], axis=0)
            if df is None or df.empty:
                continue
            # 가능한 컬럼명들 방어
            mcap_col = next((c for c in ['시가총액', '시가총액(원)', '시가총액(억원)'] if c in df.columns), None)
            if mcap_col is None:
                # try default index values if available
                try:
                    vals = df.iloc[:, 0].astype(float)
                except Exception:
                    continue
            else:
                vals = df[mcap_col].astype(float)
                # 만약 원 단위이면 억 단위로 변환
                if vals.abs().max() > 1e6:
                    vals = vals / 1e8
            df['Code'] = df.index.astype(str).str.zfill(6)
            return dict(zip(df['Code'], vals.round(2).astype(float))), ymd
        except Exception as e:
            log(f"build_mcap_map: {ymd} failed -> {e}")
            continue
    log("build_mcap_map: no valid mcap found; returning empty map")
    return {}, ""

def get_mcap_eok_from_map(mcap_map, ticker):
    return float(mcap_map.get(str(ticker).zfill(6), 0.0))

def get_fallback_sector_map():
    # 간단한 하드코드 맵(원본 유지)
    return {
        "005930": "전기전자", "000660": "전기전자", "373220": "전기전자", "207940": "의약품",
        "005380": "운수장비", "005935": "전기전자", "068270": "의약품", "000270": "운수장비",
        # ... (필요시 확장)
    }

def get_sector_map():
    sector_map = get_fallback_sector_map()
    try:
        df = fdr.StockListing('KRX')
        target_cols = ['Sector', '업종', 'Industry', 'Wics']
        col = next((c for c in target_cols if c in df.columns), None)
        if col:
            code_col = 'Symbol' if 'Symbol' in df.columns else ('Code' if 'Code' in df.columns else None)
            if code_col:
                df[code_col] = df[code_col].astype(str).str.zfill(6)
                df = df.dropna(subset=[col])
                sector_map.update(dict(zip(df[code_col], df[col])))
    except Exception as e:
        log(f"get_sector_map: fallback used ({e})")
    return sector_map

def pick_top_by_trading_value(date_yyyymmdd, top_n):
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m).reset_index()
            # 컬럼명 방어
            df = df.rename(columns={df.columns[0]: '종목코드'})
            # 거래대금의 컬럼명을 찾음
            tv_col = next((c for c in df.columns if '거래대금' in c), None)
            if tv_col is None:
                continue
            df = df[['종목코드', tv_col]].rename(columns={tv_col: '거래대금(원)'})
            frames.append(df)
        except Exception as e:
            log(f"pick_top_by_trading_value {m} failed: {e}")
            continue
    if not frames:
        raise RuntimeError("No trading value data available")
    df = pd.concat(frames, ignore_index=True)
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    df['거래대금(원)'] = nz_num(df['거래대금(원)']).fillna(0)
    df = df.sort_values('거래대금(원)', ascending=False).head(top_n)
    return df

def get_market_sets(d):
    try:
        kp = set(stock.get_market_ticker_list(d, market='KOSPI'))
        kq = set(stock.get_market_ticker_list(d, market='KOSDAQ'))
        return kp, kq
    except Exception as e:
        log(f"get_market_sets failed: {e}")
        return set(), set()

def get_name_map_cached(d):
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, "krx_codes.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str)
            return dict(zip(df['종목코드'].astype(str).str.zfill(6), df['종목명']))
        except Exception:
            pass
    rows = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            tickers = stock.get_market_ticker_list(d, market=m)
            for t in tickers:
                try:
                    name = stock.get_market_ticker_name(t)
                    rows.append({'종목코드': str(t).zfill(6), '종목명': name})
                    time.sleep(0.001)
                except Exception:
                    continue
        except Exception as e:
            log(f"get_name_map_cached({m}) failed: {e}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding=UTF8)
        return dict(zip(df['종목코드'], df['종목명']))
    return {}

# ------------------------------- AI 코멘트 (간단 Rule-based) -------------------------------
def generate_ai_comment(mfi, rsi, slope, disp, score):
    comment = ""
    if mfi >= 70: comment += "💰 수급 강세(외국인/기관 유입). "
    elif mfi >= 60: comment += "💸 자금 유입 관찰. "
    if slope > 0.0: comment += "📈 MACD 히스토그램 상승(모멘텀). "
    if -2 <= disp <= 2: comment += "✅ 20일선 부근 눌림(진입 고려). "
    elif disp > 5: comment += "⚠️ 단기 과열(조정주의). "
    elif disp < -5: comment += "📉 과매도(반등 가능성). "
    if score >= 90: comment += "(강력 매수 추천)"
    elif score >= 80: comment += "(매수 유효)"
    return comment or "특이사항 없음. 지표 확인 요망."

# ------------------------------- 스코어링 함수 -------------------------------
def build_global_score(lat):
    x = lat.copy()
    # 필요한 컬럼 방어
    default_cols = ["종가","추천매수가","손절가","추천매도가1","거래대금(억원)","RSI14","MACD_Slope","거래강도",
                    "이격도","ret_5d_%","ret_10d_%","EBS","MACD_Hist","MFI14"]
    for c in default_cols:
        if c not in x.columns:
            x[c] = np.nan

    close = nz_num(x["종가"])
    entry = nz_num(x["추천매수가"])
    stop = nz_num(x["손절가"])
    t1 = nz_num(x["추천매도가1"])
    turn = nz_num(x["거래대금(억원)"])
    rsi = nz_num(x["RSI14"])
    slope = nz_num(x["MACD_Slope"])
    volz = nz_num(x["거래강도"])
    kairi = nz_num(x["이격도"])
    r5 = nz_num(x["ret_5d_%"])
    ebs = nz_num(x["EBS"]).fillna(0)

    # 안전한 RR 계산: 분모가 0 또는 nan 인 경우 처리
    rr_den = (entry - stop).replace([0, np.inf, -np.inf], np.nan)
    rr1 = ((t1 - entry) / rr_den).where(rr_den.notna())
    now_gap = ((close - entry).abs() / entry * 100).where(entry.notna())
    t1_room = ((t1 - close) / close * 100).where(close.notna())
    sl_room = ((close - stop) / close * 100).where(close.notna())

    rr_norm = pct_norm_pos(rr1, q=90, floor=1.0).fillna(0)
    t1_norm = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1).fillna(0)
    sl_norm = np.clip(sl_room / cap_q(sl_room, q=90, floor=3.0), 0, 1).fillna(0)
    near_norm = inv_dist_norm(now_gap, cap_q(now_gap, q=75, floor=1.0)).fillna(0)

    ers_bits = (ebs >= PASS_EBS).astype(int) + (slope > 0).astype(int) + ((rsi >= 45) & (rsi <= 65)).astype(int)
    ers_norm = np.clip(ers_bits / 3.0, 0, 1).fillna(0)
    slope_pos_norm = pct_norm_pos(slope, q=90, floor=1.0).fillna(0)
    mom_norm = np.clip(0.5 * ers_norm + 0.3 * slope_pos_norm, 0, 1).fillna(0)

    if turn.notna().any():
        lo, hi = np.nanpercentile(turn.dropna(), 30), np.nanpercentile(turn.dropna(), 90)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo == 0:
            liq_norm = 0.0
        else:
            liq_norm = np.clip((turn - lo) / max(hi - lo, 1e-9), 0, 1).fillna(0)
    else:
        liq_norm = 0.0

    vol_sweet = (1 - np.minimum((volz - 1).abs() / 3, 1)).clip(0, 1).fillna(0)
    kairi_norm = (1 - np.minimum(kairi.abs() / cap_q(kairi.abs(), q=80, floor=3.0), 1)).clip(0, 1).fillna(0)
    tec_norm = np.clip(0.6 * vol_sweet + 0.4 * kairi_norm, 0, 1).fillna(0)

    base_score = (100 * W_RR * rr_norm) + (100 * W_T1 * t1_norm) + (100 * W_SL * sl_norm) + \
                 (100 * W_NEAR * near_norm) + (100 * W_MOM * mom_norm) + (100 * W_LIQ * liq_norm) + (100 * W_TEC * tec_norm)

    pen = pd.Series(0.0, index=x.index)
    pen += P_OVERHEAT_5D * np.clip((r5 - 10) / 10, 0, 1)
    pen += P_RSI_OUT * ((rsi < 45) | (rsi > 65)).astype(float)
    pen += P_MACD_NEG * (slope < 0).astype(float)

    score = np.clip(base_score - pen, 0, 100)

    x["RR1"] = rr1
    x["Now%"] = now_gap
    x["LDY_SCORE"] = score.round(1)
    x["_GATE_OK"] = (nz_num(x["거래대금(억원)"]) >= 0).fillna(False)  # 실제 gate는 외부에서 검사
    x = x.sort_values("LDY_SCORE", ascending=False, na_position="last")
    x["LDY_RANK"] = range(1, len(x) + 1)

    # simple route (fallback)
    conditions = [(r5 >= 3) & (slope > 0), (rsi >= 40) & (rsi <= 60), (rsi <= 40)]
    choices = ["🔼 BRK (돌파)", "↩️ PULL (눌림)", "🔁 MR (반전)"]
    x["ROUTE"] = np.select(conditions, choices, default="—")

    # AI Comment
    x["AI_COMMENT"] = x.apply(lambda row: generate_ai_comment(row.get("MFI14", 50), row.get("RSI14", 50),
                                                              row.get("MACD_Slope", 0), row.get("이격도", 0),
                                                              row.get("LDY_SCORE", 0)), axis=1)
    return x

# ------------------------------- Telegram 전송 (안전) -------------------------------
def send_telegram_auto(df):
    log("📨 텔레그램 발송 시작...")
    if not TG_TOKEN or not TG_ID:
        log("TG_TOKEN 또는 TG_ID 미설정. 전송 스킵.")
        return
    try:
        top5 = df.head(5).reset_index(drop=True)
        trade_date = datetime.now(KST).strftime('%Y-%m-%d')
        msg = f"🔥 [LDY v6.0] 추천 Top 5 ({trade_date})\n\n"
        for i, row in top5.iterrows():
            rank = i + 1
            name = row.get('종목명', '')
            code = row.get('종목코드', '')
            route = row.get('ROUTE', '전략없음')
            buy = int(row.get('추천매수가', 0) or 0)
            stop = int(row.get('손절가', 0) or 0)
            t1 = int(row.get('추천매도가1', 0) or 0)
            comment = row.get('AI_COMMENT', '')
            msg += f"{rank}. {name} ({code})\n"
            msg += f"   🎯전략: {route}\n"
            msg += f"   💬AI: {comment}\n"
            msg += f"   🔵매수: {buy:,}\n"
            msg += f"   🔴손절: {stop:,} / 🟢목표: {t1:,}\n\n"
        resp = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                             data={"chat_id": TG_ID, "text": msg}, timeout=10)
        resp.raise_for_status()
        log("🚀 텔레그램 전송 완료")
    except Exception as e:
        log(f"⚠️ 텔레그램 전송 실패: {e}")

# ------------------------------- 메인 분석 루틴 -------------------------------
def analyze_ticker(t, start_s, end_s, top_df, name_map, sector_map, kospi_set, kosdaq_set, mcap_map):
    try:
        ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
        if ohlcv is None or ohlcv.empty or len(ohlcv) < 120:
            return None
        ohlcv = ohlcv.tail(LOOKBACK_DAYS)
        c = ohlcv['종가']
        h = ohlcv['고가']
        l = ohlcv['저가']
        v = ohlcv['거래량']
        ma20 = c.rolling(20).mean()
        ma60 = c.rolling(60).mean()
        ma120 = c.rolling(120).mean()
        atr = calc_atr(h, l, c, 14).iloc[-1]
        rsi = calc_rsi(c, 14).iloc[-1]
        mfi = calc_mfi(h, l, c, v, 14).iloc[-1]
        macd = ema(c, 12) - ema(c, 26)
        sig = ema(macd, 9)
        hist = macd - sig
        # slope 안정적 계산: 최근 5봉 회귀기울기
        hist_vals = hist.tail(5).values
        if len(hist_vals) >= 2 and np.all(np.isfinite(hist_vals)):
            try:
                slope = float(np.polyfit(np.arange(len(hist_vals)), hist_vals, 1)[0])
            except Exception:
                slope = hist.diff().iloc[-1] if len(hist) > 1 else 0.0
        else:
            slope = hist.diff().iloc[-1] if len(hist) > 1 else 0.0
        vol_z = (v / v.rolling(20).mean()).iloc[-1]
        disp = ((c / ma20 - 1.0) * 100).iloc[-1] if ma20.iloc[-1] != 0 else 0.0
        last_c = c.iloc[-1]
        # ret 계산
        ret_5d = c.pct_change(5).iloc[-1] * 100 if len(c) > 5 else np.nan
        ret_10d = c.pct_change(10).iloc[-1] * 100 if len(c) > 10 else np.nan
        tv_eok = float(top_df.loc[top_df["종목코드"] == t, "거래대금(원)"].values[0]) / 1e8 if not top_df.loc[top_df["종목코드"] == t, "거래대금(원)"].empty else 0.0
        mcap = get_mcap_eok_from_map(mcap_map, t)
        if tv_eok < MIN_TURNOVER_EOK or (mcap > 0 and mcap < MIN_MCAP_EOK):
            return None
        # 간단 점수 (rule-based)
        score = 0
        reasons = []
        if RSI_LOW <= rsi <= RSI_HIGH:
            score += 1; reasons.append("RSI적정")
        if slope > 0:
            score += 1; reasons.append("MACD상승")
        if -1 <= disp <= 5:
            score += 1; reasons.append("20선근접")
        if vol_z > 1.2:
            score += 1; reasons.append("거래량↑")
        if ma20.iloc[-1] > ma60.iloc[-1]:
            score += 1; reasons.append("정배열(단)")
        if last_c > ma120.iloc[-1]:
            score += 1; reasons.append("장기추세(120↑)")
        else:
            score -= 1
        if mfi > 60:
            score += 1; reasons.append("자금유입(MFI)")
        if hist.iloc[-1] > 0:
            score += 1; reasons.append("MACD>Sig")
        try:
            atr = float(atr)
        except:
            atr = max(1.0, last_c * 0.03)
        buy = min(last_c, ma20.iloc[-1] * 1.03) if ma20.iloc[-1] > 0 and last_c > ma20.iloc[-1] else last_c
        stop = buy - (2.0 * atr)
        if stop < buy * 0.93:
            stop = buy * 0.93
        if stop >= buy * 0.97:
            stop = buy * 0.97
        risk = buy - stop if buy - stop > 0 else max(1.0, buy * 0.01)
        rr1, rr2 = (2.0, 4.0) if score >= 8 else ((1.5, 3.0) if score >= 6 else (1.2, 2.5))
        t1 = buy + risk * rr1
        t2 = buy + risk * rr2
        buy = round_to_tick(buy); stop = round_to_tick(stop); t1 = round_to_tick(t1); t2 = round_to_tick(t2)
        sector = sector_map.get(str(t).zfill(6), "기타")
        return {
            "시장": "KOSPI" if t in kospi_set else "KOSDAQ",
            "종목명": name_map.get(str(t).zfill(6), str(t)),
            "종목코드": str(t).zfill(6),
            "업종": sector,
            "종가": int(last_c), "거래대금(억원)": round(tv_eok, 2),
            "시가총액(억원)": round(mcap, 1),
            "RSI14": round(float(rsi) if not np.isnan(rsi) else 50.0, 1),
            "MFI14": round(float(mfi) if not np.isnan(mfi) else 50.0, 1),
            "이격도": round(float(disp), 2),
            "MACD_Hist": round(float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0, 6),
            "MACD_Slope": round(float(slope), 6),
            "거래강도": round(float(vol_z) if not np.isnan(vol_z) else 0.0, 2),
            "ret_5d_%": round(float(ret_5d) if not np.isnan(ret_5d) else 0.0, 2),
            "ret_10d_%": round(float(ret_10d) if not np.isnan(ret_10d) else 0.0, 2),
            "EBS": int(score),
            "통과": "★" if score >= PASS_EBS else "",
            "근거": ", ".join(reasons),
            "추천매수가": int(buy), "손절가": int(stop), "추천매도가1": int(t1), "추천매도가2": int(t2)
        }
    except Exception as e:
        log(f"analyze_ticker {t} failed: {e}")
        return None

def main(parallel_workers=6, use_parallel=True):
    log("🚀 LDY Collector v6.0 시작...")
    mcap_map, mcap_ymd = build_mcap_map()
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일: {trade_ymd} (mcap ref: {mcap_ymd})")
    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].astype(str).str.zfill(6).tolist()
    kospi_set, kosdaq_set = get_market_sets(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)
    sector_map = get_sector_map()
    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS * 2 + 60)
    start_s, end_s = start_dt.strftime("%Y%m%d"), trade_ymd
    rows = []

    if use_parallel:
        log(f"병렬 수집 실행 (workers={parallel_workers})")
        with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
            futures = {ex.submit(analyze_ticker, t, start_s, end_s, top_df, name_map, sector_map, kospi_set, kosdaq_set, mcap_map): t for t in tickers}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Analyzing"):
                res = fut.result()
                if res:
                    rows.append(res)
                time.sleep(SLEEP_SEC)  # rate-control
    else:
        log("단일 스레드 수집 실행")
        for t in tqdm(tickers, desc="Analyzing"):
            res = analyze_ticker(t, start_s, end_s, top_df, name_map, sector_map, kospi_set, kosdaq_set, mcap_map)
            if res:
                rows.append(res)
            time.sleep(SLEEP_SEC)

    if not rows:
        raise RuntimeError("No Result after analysis")

    df_raw = pd.DataFrame(rows)
    df_out = build_global_score(df_raw).sort_values(["LDY_SCORE", "거래대금(억원)"], ascending=[False, False])
    ensure_dir(OUT_DIR)
    out_path = os.path.join(OUT_DIR, "recommend_latest.csv")
    df_out.to_csv(out_path, index=False, encoding=UTF8)
    log(f"💾 저장 완료: {out_path} ({len(df_out)}건)")
    send_telegram_auto(df_out)
    return df_out

if __name__ == "__main__":
    # 실행 옵션: 병렬 실행 여부 및 워커 수는 필요에 따라 조정
    try:
        df_result = main(parallel_workers=6, use_parallel=True)
        log("완료.")
    except Exception as e:
        log(f"치명적 오류 발생: {e}")
