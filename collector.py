# -*- coding: utf-8 -*-
"""
LDY Pro Trader Collector v6.0 (Sector Auto-Mapping Edition)
- 업종 자동 매핑 강화
    · FDR의 Sector/업종 컬럼 활용 (NaN을 억지로 '기타'로 채우지 않음)
    · 종목명 기반 간단 카테고리 추론 (ETF/ETN/리츠/스팩/우선주 등)
    · 끝까지 못 찾은 종목만 최종적으로 '기타' 부여
- AI Narrative / Scoring / Telegram 그대로 유지
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

# [보안 설정]
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_ID = os.environ.get("TG_ID")

# ------------------------------- 설정 -------------------------------
KST = timezone(timedelta(hours=9))
LOOKBACK_DAYS = 250          # 과거 데이터 조회 일수
TOP_N = 600                  # 거래대금 상위 N개 종목
MIN_TURNOVER_EOK = 50        # 최소 거래대금 (억원)
MIN_MCAP_EOK = 1000          # 최소 시총 (억원)
RSI_LOW, RSI_HIGH = 45, 65   # RSI 적정 구간
PASS_EBS = 4                 # EBS (룰 기반 스코어) 통과 기준
SLEEP_SEC = 0.05
OUT_DIR = "data"
UTF8 = "utf-8-sig"

# [가중치]
W_RR, W_T1, W_SL, W_NEAR, W_MOM, W_LIQ, W_TEC = 0.25, 0.18, 0.12, 0.12, 0.10, 0.13, 0.10
P_OVERHEAT_5D, P_OVERHEAT_10D, P_RSI_OUT = 6.0, 6.0, 4.0
P_MACD_NEG, P_NEAR_FAR, P_LIQ_LOW, P_VOL_SPIKE = 4.0, 4.0, 4.0, 2.0

# ------------------------------- 유틸 -------------------------------
def log(msg: str):
    print(f"[{datetime.now(KST)}] {msg}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _safe_sum(x):
    return pd.to_numeric(x, errors="coerce").fillna(0).sum()

def nz_num(s):
    return pd.to_numeric(s, errors="coerce")

def calc_rsi(close, period=14):
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(period).mean() / down.replace(0, np.nan).rolling(period).mean()
    return 100 - 100 / (1 + rs)

def calc_atr(high, low, close, period=14):
    tr = pd.concat(
        [(high - low),
         (high - close.shift(1)).abs(),
         (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def calc_mfi(high, low, close, vol, period=14):
    tp = (high + low + close) / 3
    rmf = tp * vol
    pos = np.where(tp.diff() > 0, rmf, 0)
    neg = np.where(tp.diff() < 0, rmf, 0)
    pos_s = pd.Series(pos).rolling(period).sum()
    neg_s = pd.Series(neg).rolling(period).sum().replace(0, 1)
    return 100 - (100 / (1 + (pos_s / neg_s)))

def round_to_tick(price):
    if price < 2000: t = 1
    elif price < 5000: t = 5
    elif price < 20000: t = 10
    elif price < 50000: t = 50
    elif price < 200000: t = 100
    elif price < 500000: t = 500
    else: t = 1000
    return int(round(price / t) * t)

# ------------------------------- 거래일/시총 -------------------------------
def _has_ohlcv_and_mcap(ymd):
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            o = stock.get_market_ohlcv_by_ticker(ymd, market=m)
            if o is not None and not o.empty and "거래대금" in o.columns and _safe_sum(o["거래대금"]) > 0:
                return True
        except:
            pass
    return False

def resolve_trade_date():
    """
    장 종료 이후 기준일 찾기 (최대 10일 뒤로)
    """
    d = datetime.now(KST).date()
    # 18시 이전이면 전일 기준
    if datetime.now(KST).hour < 18:
        d -= timedelta(days=1)
    for _ in range(10):
        ymd = d.strftime("%Y%m%d")
        if _has_ohlcv_and_mcap(ymd):
            return ymd
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")

def build_mcap_map():
    d = datetime.now(KST).date()
    if datetime.now(KST).hour < 18:
        d -= timedelta(days=1)
    for _ in range(10):
        use = d.strftime("%Y%m%d")
        try:
            df = pd.concat([
                stock.get_market_cap_by_ticker(use, market='KOSPI'),
                stock.get_market_cap_by_ticker(use, market='KOSDAQ')
            ])
            if not df.empty:
                df['Code'] = df.index
                return dict(zip(df['Code'], df['시가총액'] / 1e8)), use
        except:
            pass
        d -= timedelta(days=1)
    return {}, ""

def get_mcap_eok_from_map(mcap_map, ticker):
    return float(mcap_map.get(str(ticker).zfill(6), 0))

# ------------------------------- 업종 맵핑 -------------------------------
def get_fallback_sector_map():
    """
    최소한의 대형주 하드 맵핑 (문제 생길 때 백업용)
    """
    return {
        "005930": "전기전자", "000660": "전기전자", "373220": "전기전자", "207940": "의약품",
        "005380": "운수장비", "005935": "전기전자", "068270": "의약품", "000270": "운수장비",
        "105560": "금융업", "005490": "철강금속", "035420": "서비스업", "035720": "서비스업",
        "006400": "전기전자", "051910": "화학", "012330": "화학", "028260": "유통업",
        "055550": "금융업", "086790": "금융업", "032830": "금융업", "003550": "화학",
        "015760": "전기가스업", "034020": "기계", "010120": "전기전자", "323410": "서비스업",
        "259960": "서비스업", "011200": "운수창고", "000810": "금융업", "018260": "서비스업",
        "010130": "철강금속", "009150": "전기전자", "033780": "금융업", "017670": "통신업",
        "329180": "운수장비", "096770": "화학", "003490": "운수창고", "030200": "통신업",
        "316140": "금융업", "000100": "의약품", "251270": "서비스업", "024110": "금융업",
        "036570": "서비스업", "086280": "운수창고", "090430": "화학", "010950": "화학",
        "009540": "운수장비", "267260": "전기전자", "042700": "전기전자", "010620": "화학",
        "138040": "금융업", "034730": "서비스업", "241560": "화학", "000150": "기계",
        "298040": "전기전자", "108490": "기계", "466100": "기계", "437730": "운수장비",
        "098460": "기계", "277810": "기계", "352820": "서비스업", "253450": "서비스업"
    }

def fetch_fdr_sector():
    """
    FDR StockListing('KRX')에서 Sector/업종/Wics 등을 최대한 활용하여
    코드→업종 맵 생성 (NaN을 '기타'로 채우지 않음)
    """
    sector_map = {}
    try:
        df = fdr.StockListing('KRX')
        # 코드 컬럼 찾기
        code_col = 'Symbol' if 'Symbol' in df.columns else 'Code'
        df[code_col] = df[code_col].astype(str).str.zfill(6)

        # 업종 후보 컬럼
        cand_cols = ['업종', 'Sector', 'Industry', 'Wics', 'WICS']
        sector_col = None
        for c in cand_cols:
            if c in df.columns:
                sector_col = c
                break
        if sector_col is None:
            log("[WARN] FDR: 업종/섹터 컬럼을 찾지 못함")
            return sector_map

        tmp = df[[code_col, sector_col]].copy()
        tmp[sector_col] = tmp[sector_col].astype(str).str.strip()
        # 공백은 NaN 처리 후, 실제 값 있는 것만 사용
        tmp = tmp.replace({"": np.nan}).dropna(subset=[sector_col])

        sector_map.update(dict(zip(tmp[code_col], tmp[sector_col])))
        log(f"[INFO] FDR 업종 매핑 {len(sector_map)}건 로드")
    except Exception as e:
        log(f"[WARN] FDR 업종 수집 실패: {e}")
    return sector_map

def classify_by_name(code, name):
    """
    KIND/FDR 모두에서 업종을 못 찾은 종목에 대해
    종목명 기반으로 대략적인 카테고리 부여 (없으면 None)
    """
    if not isinstance(name, str):
        return None

    u = name.upper()
    if "ETF" in u:
        return "ETF"
    if "ETN" in u:
        return "ETN"
    if "SPAC" in u or "스팩" in name:
        return "SPAC"
    if "리츠" in name or "REIT" in u:
        return "리츠/부동산"
    if "우선주" in name or "/우" in name or "우B" in name or "우C" in name:
        return "우선주"
    # 나머지는 일반 주식으로 보고 None (마지막 단계에서만 '기타')
    return None

def get_sector_map_full(name_map):
    """
    최종 업종 맵 생성:
    1) fallback 하드맵
    2) FDR StockListing 업종/섹터
    3) 이름 기반 보완 (ETF/ETN/리츠/스팩/우선주 등)
    4) 그래도 없으면 '기타'
    """
    # 1) 하드코딩한 기본 맵
    sector_map = get_fallback_sector_map()

    # 2) FDR에서 업종 정보 보강
    log("📥 FDR 업종 맵 수집 중...")
    fdr_map = fetch_fdr_sector()
    for code, sec in fdr_map.items():
        # 이미 하드맵에 있으면 그대로 두고, 없거나 빈 값일 때만 덮어쓰기
        if code not in sector_map or sector_map[code] in ["", None]:
            sector_map[code] = sec

    # 3) 이름 기반으로 ETF/ETN/리츠/스팩/우선주 등 보완
    log("📥 이름 기반 업종 보완 중...")
    for code, name in name_map.items():
        code6 = str(code).zfill(6)
        cur = sector_map.get(code6, None)
        if cur in ["", None]:
            guess = classify_by_name(code6, name)
            if guess is not None:
                sector_map[code6] = guess

    # 4) 마지막으로도 여전히 없는 애들만 '기타'
    for code in name_map.keys():
        code6 = str(code).zfill(6)
        if code6 not in sector_map or sector_map[code6] in ["", None]:
            sector_map[code6] = "기타"

    return sector_map

# ------------------------------- 기타 수집 로직 -------------------------------
def pick_top_by_trading_value(date_yyyymmdd, top_n):
    frames = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_ohlcv_by_ticker(date_yyyymmdd, market=m).reset_index()
            # pykrx 버전에 따라 컬럼명이 약간 다를 수 있어 보정
            df.columns = ['종목코드' if '티커' in str(c) or '코드' in str(c) else c for c in df.columns]
            df.columns = ['거래대금(원)' if c == '거래대금' else c for c in df.columns]
            frames.append(df[['종목코드', '거래대금(원)']])
        except:
            pass
    if not frames:
        raise RuntimeError("No Data from KRX (거래대금)")
    df = pd.concat(frames)
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    return df.sort_values('거래대금(원)', ascending=False).head(top_n)

def get_market_sets(d):
    try:
        return set(stock.get_market_ticker_list(d, market='KOSPI')), set(stock.get_market_ticker_list(d, market='KOSDAQ'))
    except:
        return set(), set()

def get_name_map_cached(d):
    """
    종목코드 → 종목명 맵 (캐시 파일 사용)
    """
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, "krx_codes.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str)
            return dict(zip(df['종목코드'], df['종목명']))
        except:
            pass

    rows = []
    for m in ["KOSPI", "KOSDAQ"]:
        try:
            tickers = stock.get_market_ticker_list(d, market=m)
            for t in tickers:
                rows.append({
                    '종목코드': str(t).zfill(6),
                    '종목명': stock.get_market_ticker_name(t)
                })
                time.sleep(0.001)
        except:
            pass
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding=UTF8)
        return dict(zip(df['종목코드'], df['종목명']))
    return {}

# ------------------------------- AI 코멘트 / 스코어 -------------------------------
def generate_ai_comment(mfi, rsi, slope, disp, score):
    comment = ""

    # 1. 수급 분석
    if mfi >= 70:
        comment += "💰 외국인/기관의 강력한 수급이 집중되고 있습니다. "
    elif mfi >= 60:
        comment += "💸 자금 유입이 꾸준히 이어지고 있습니다. "

    # 2. 추세/모멘텀 분석
    if slope > 100:
        comment += "🚀 상승 에너지가 폭발적으로 증가하는 중입니다. "
    elif slope > 0:
        comment += "📈 상승 추세가 견고하게 유지되고 있습니다. "

    # 3. 위치 분석
    if -2 <= disp <= 2:
        comment += "✅ 20일선 부근의 안전한 눌림목 구간입니다."
    elif disp > 5:
        comment += "⚠️ 단기 급등으로 인한 조정 가능성을 염두에 두세요."
    elif disp < -5:
        comment += "📉 과매도 구간으로 기술적 반등이 기대됩니다."

    # 4. 점수 요약
    if score >= 90:
        comment += " (강력 매수 추천)"
    elif score >= 80:
        comment += " (매수 유효)"

    return comment if comment else "특이사항 없음. 기술적 지표를 참고하세요."

# --- Scoring Logic ---
def cap_q(s, q=90, floor=1.0):
    c = np.nanpercentile(nz_num(s), q)
    return float(max(c, floor)) if np.isfinite(c) else floor

def pct_norm_pos(s, q=90, floor=1.0):
    s = nz_num(s).clip(lower=0)
    return np.clip(s / cap_q(s, q, floor), 0, 1)

def inv_dist_norm(dist, cap):
    return np.clip(1 - (nz_num(dist) / cap), 0, 1)

def build_global_score(lat):
    x = lat.copy()
    close, entry, stop, t1 = nz_num(x["종가"]), nz_num(x["추천매수가"]), nz_num(x["손절가"]), nz_num(x["추천매도가1"])
    turn, rsi, slope, volz = nz_num(x["거래대금(억원)"]), nz_num(x["RSI14"]), nz_num(x["MACD_Slope"]), nz_num(x["거래강도"])
    kairi, r5, ebs = nz_num(x["이격도"]), nz_num(x["ret_5d_%"]), nz_num(x["EBS"]).fillna(0)

    rr_den = (entry - stop)
    rr1 = ((t1 - entry) / rr_den.replace(0, np.nan))
    now_gap = ((close - entry).abs() / entry * 100)
    t1_room = ((t1 - close) / close * 100)
    sl_room = ((close - stop) / close * 100)

    rr_norm = pct_norm_pos(rr1, q=90, floor=1.0).fillna(0)
    t1_norm = np.clip(t1_room / cap_q(t1_room, q=90, floor=5.0), 0, 1).fillna(0)
    sl_norm = np.clip(sl_room / cap_q(sl_room, q=90, floor=3.0), 0, 1).fillna(0)
    near_norm = inv_dist_norm(now_gap, cap=cap_q(now_gap, q=75, floor=1.0)).fillna(0)

    ers_bits = (ebs >= PASS_EBS).astype(int) + (slope > 0).astype(int) + ((rsi >= 45) & (rsi <= 65)).astype(int)
    ers_norm = np.clip(ers_bits / 3.0, 0, 1).fillna(0)
    slope_pos_norm = pct_norm_pos(slope, q=90, floor=1.0).fillna(0)
    mom_norm = np.clip(0.5 * ers_norm + 0.3 * slope_pos_norm, 0, 1).fillna(0)

    if turn.notna().any():
        lo, hi = np.nanpercentile(turn, 30), np.nanpercentile(turn, 90)
        liq_norm = np.clip((turn - lo) / max(hi - lo, 1e-9), 0, 1).fillna(0)
    else:
        liq_norm = 0.0

    vol_sweet = (1 - np.minimum((volz - 1).abs() / 3, 1)).clip(0, 1).fillna(0)
    kairi_norm = (1 - np.minimum(kairi.abs() / cap_q(kairi.abs(), q=80, floor=3.0), 1)).clip(0, 1).fillna(0)
    tec_norm = np.clip(0.6 * vol_sweet + 0.4 * kairi_norm, 0, 1).fillna(0)

    base_score = (
        100 * W_RR * rr_norm
        + 100 * W_T1 * t1_norm
        + 100 * W_SL * sl_norm
        + 100 * W_NEAR * near_norm
        + 100 * W_MOM * mom_norm
        + 100 * W_LIQ * liq_norm
        + 100 * W_TEC * tec_norm
    )

    pen = pd.Series(0.0, index=x.index)
    pen += P_OVERHEAT_5D * np.clip((r5 - 10) / 10, 0, 1)
    pen += P_RSI_OUT * ((rsi < 45) | (rsi > 65)).astype(float)
    pen += P_MACD_NEG * (slope < 0).astype(float)

    score = np.clip(base_score - pen, 0, 100)
    x["RR1"] = rr1
    x["Now%"] = now_gap
    x["LDY_SCORE"] = score.round(1)

    # ROUTE (전략 태그)
    conditions = [
        (r5 >= 3) & (slope > 0),
        (rsi >= 40) & (rsi <= 60),
        (rsi <= 40)
    ]
    choices = ["🔼 BRK (돌파)", "↩️ PULL (눌림)", "🔁 MR (반전)"]
    x["ROUTE"] = np.select(conditions, choices, default="—")

    # AI 코멘트
    x["AI_COMMENT"] = x.apply(lambda row: generate_ai_comment(
        row.get("MFI14", 50),
        row.get("RSI14", 50),
        row.get("MACD_Slope", 0),
        row.get("이격도", 0),
        row.get("LDY_SCORE", 0)
    ), axis=1)

    return x

# ------------------------------- 텔레그램 -------------------------------
def send_telegram_auto(df):
    log("📨 텔레그램 발송 시작...")
    if not TG_TOKEN or not TG_ID:
        log("⚠️ TG_TOKEN / TG_ID 미설정, 발송 생략")
        return

    try:
        top5 = df.head(5).reset_index(drop=True)
        trade_date = datetime.now(KST).strftime('%Y-%m-%d')
        msg = f"🔥 [LDY v6.0] 추천 Top 5 ({trade_date})\n"
        msg += "-" * 30 + "\n\n"

        for i, row in top5.iterrows():
            rank = i + 1
            name = row['종목명']
            code = row['종목코드']
            route = row.get('ROUTE', '전략없음')
            buy = row['추천매수가']
            comment = row.get('AI_COMMENT', '')

            msg += f"{rank}. {name} ({code})\n"
            msg += f"   🎯전략: {route}\n"
            msg += f"   💬AI: {comment}\n"
            msg += f"   🔵매수: {buy:,}\n"
            msg += f"   🔴손절: {row['손절가']:,} / 🟢목표: {row['추천매도가1']:,}\n\n"

        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_ID, "text": msg}
        )
        log("🚀 텔레그램 전송 완료")
    except Exception as e:
        log(f"⚠️ 텔레그램 전송 실패: {e}")

# ------------------------------- 메인 실행 -------------------------------
def main():
    log("🚀 LDY Collector v6.0 시작...")
    mcap_map, mcap_ymd = build_mcap_map()
    trade_ymd = resolve_trade_date()
    log(f"📅 거래 기준일: {trade_ymd} (mcap ref: {mcap_ymd})")

    # 1) 거래대금 상위 N개
    top_df = pick_top_by_trading_value(trade_ymd, TOP_N)
    tickers = top_df["종목코드"].tolist()

    # 2) 시장/종목명/업종 정보 준비
    kospi_set, kosdaq_set = get_market_sets(trade_ymd)
    name_map = get_name_map_cached(trade_ymd)
    sector_map = get_sector_map_full(name_map)

    start_dt = datetime.strptime(trade_ymd, "%Y%m%d") - timedelta(days=LOOKBACK_DAYS * 2 + 60)
    start_s, end_s = start_dt.strftime("%Y%m%d"), trade_ymd

    rows = []
    for t in tqdm(tickers, desc="Analyzing"):
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_s, end_s, t)
            if ohlcv is None or ohlcv.empty or len(ohlcv) < 120:
                continue
            ohlcv = ohlcv.tail(LOOKBACK_DAYS)

            c, h, l, v = ohlcv["종가"], ohlcv["고가"], ohlcv["저가"], ohlcv["거래량"]
            ma20, ma60, ma120 = c.rolling(20).mean(), c.rolling(60).mean(), c.rolling(120).mean()

            atr = calc_atr(h, l, c, 14).iloc[-1]
            rsi = calc_rsi(c, 14).iloc[-1]
            mfi = calc_mfi(h, l, c, v, 14).iloc[-1]

            macd = ema(c, 12) - ema(c, 26)
            sig = ema(macd, 9)
            hist = macd - sig
            slope = hist.diff().iloc[-1]

            vol_z = (v / v.rolling(20).mean()).iloc[-1]
            disp = ((c / ma20 - 1.0) * 100).iloc[-1]

            last_c = c.iloc[-1]

            tv_eok = float(top_df.loc[top_df["종목코드"] == t, "거래대금(원)"].values[0]) / 1e8
            mcap = get_mcap_eok_from_map(mcap_map, t)

            # 유동성 / 시총 필터
            if tv_eok < MIN_TURNOVER_EOK:
                continue
            if mcap > 0 and mcap < MIN_MCAP_EOK:
                continue

            score = 0
            reason = []
            if RSI_LOW <= rsi <= RSI_HIGH:
                score += 1; reason.append("RSI적정")
            if slope > 0:
                score += 1; reason.append("MACD상승")
            if -1 <= disp <= 5:
                score += 1; reason.append("20선근접")
            if vol_z > 1.2:
                score += 1; reason.append("거래량↑")
            if ma20.iloc[-1] > ma60.iloc[-1]:
                score += 1; reason.append("정배열(단)")
            if last_c > ma120.iloc[-1]:
                score += 1; reason.append("장기추세(120↑)")
            else:
                score -= 1
            if mfi > 60:
                score += 1; reason.append("자금유입(MFI)")
            if hist.iloc[-1] > 0:
                score += 1; reason.append("MACD>Sig")

            try:
                atr = float(atr)
            except:
                atr = 0.0
            if np.isnan(atr) or atr <= 0:
                atr = last_c * 0.03

            # 매수가
            if ma20.iloc[-1] > 0 and last_c > ma20.iloc[-1]:
                buy = min(last_c, ma20.iloc[-1] * 1.03)
            else:
                buy = last_c

            # 손절가
            stop = buy - (2.0 * atr)
            if stop < buy * 0.93:
                stop = buy * 0.93
            if stop >= buy * 0.97:
                stop = buy * 0.97

            risk = buy - stop
            if score >= 8:
                rr1, rr2 = (2.0, 4.0)
            elif score >= 6:
                rr1, rr2 = (1.5, 3.0)
            else:
                rr1, rr2 = (1.2, 2.5)
            t1 = buy + risk * rr1
            t2 = buy + risk * rr2

            buy = round_to_tick(buy)
            stop = round_to_tick(stop)
            t1 = round_to_tick(t1)
            t2 = round_to_tick(t2)

            code6 = str(t).zfill(6)
            sector = sector_map.get(code6, "기타")
            name = name_map.get(code6, code6)

            rows.append({
                "시장": "KOSPI" if t in kospi_set else "KOSDAQ",
                "종목명": name,
                "종목코드": code6,
                "업종": sector,
                "종가": int(last_c),
                "거래대금(억원)": round(tv_eok, 2),
                "시가총액(억원)": round(mcap, 1),
                "RSI14": round(rsi, 1),
                "MFI14": round(mfi, 1),
                "이격도": round(disp, 2),
                "MACD_Hist": round(hist.iloc[-1], 4),
                "MACD_Slope": round(slope, 5),
                "거래강도": round(vol_z, 2),
                "ret_5d_%": 0,
                "ret_10d_%": 0,
                "EBS": int(score),
                "통과": "★" if score >= PASS_EBS else "",
                "근거": ", ".join(reason),
                "추천매수가": buy,
                "손절가": stop,
                "추천매도가1": t1,
                "추천매도가2": t2
            })
        except Exception:
            # 개별 종목 에러는 무시하고 다음으로 진행
            continue

    if not rows:
        raise RuntimeError("No Result (필터를 모두 통과한 종목 없음)")

    df_raw = pd.DataFrame(rows)
    df_out = build_global_score(df_raw).sort_values(
        ["LDY_SCORE", "거래대금(억원)"],
        ascending=[False, False]
    )

    ensure_dir(OUT_DIR)
    out_path = os.path.join(OUT_DIR, "recommend_latest.csv")
    df_out.to_csv(out_path, index=False, encoding=UTF8)
    log(f"💾 저장 완료 ({len(df_out)}건) → {out_path}")

    send_telegram_auto(df_out)

if __name__ == "__main__":
    main()
