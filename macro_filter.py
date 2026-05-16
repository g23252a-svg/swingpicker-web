# -*- coding: utf-8 -*-
"""
macro_filter.py — 매크로 환경 판단 + 벤치마크
═══════════════════════════════════════════════════
[v14]  collector.py 분할 — 매크로 관련 함수 모음
[v2.0] 5건 리팩터링
  #1 risk_level 정수 매핑 (문자열 max() ASCII 버그 수정)
  #2 동적 조회 구간 (하드코딩 "2024-01-01" 제거, 50 영업일)
  #3 글로벌 시차 방어 (KR/US 영업일 정렬, iloc 안전 접근)
  #4 DRY: 공통 유틸 함수 (_parse_ymd, _safe_fdr_fetch)
  #5 Fail-safe 예외 처리 (데이터 실패 시 보수적 CAUTION 격상)
[v2.2]
  #1 FDR 모듈 레벨 1회 import (함수별 반복 import 제거)
  #2 데이터 신선도(Freshness) 체크 (stale 데이터 CAUTION 격상)
"""
import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from collector_config import CollectorConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# [v2.2 #1] 모듈 레벨 1회 import — 함수별 try/import 오버헤드 제거
try:
    import FinanceDataReader as fdr
    HAS_FDR = True
except ImportError:
    fdr = None  # type: ignore
    HAS_FDR = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    yf = None  # type: ignore
    HAS_YF = False


# ═══════════════════════════════════════════════════
#  공통 유틸리티 (#4 DRY)
# ═══════════════════════════════════════════════════

# [v2.0 #1] 리스크 레벨 정수 매핑 — 문자열 max() ASCII 버그 근절
_RISK_LEVELS = {"NORMAL": 0, "CAUTION": 1, "CRITICAL": 2}
_RISK_NAMES = {v: k for k, v in _RISK_LEVELS.items()}

# [v2.0 #2] 데이터 조회 영업일 수 (50 BDay ≈ 약 2.5개월)
_LOOKBACK_BDAYS = 50  # 기본 macro 분석용 (60일 미만 보유기간만 필요)

# [v3.9.15e + 2] 벤치마크 수익률 전용 lookback.
# get_benchmark_returns()가 [1, 3, 5, 10, 20, 60, 120] 보유기간 모두 계산하지만
# _LOOKBACK_BDAYS=50으로는 if len(close) > 60 통과 못 해서 60·120 키가 dict에
# 안 들어감 → bench_cache_latest.json에 60일 키 누락 → 슬라이더 41~60일 알파
# 미산출. 130일이면 60일 보유기간은 안정적으로 산출 가능 (120일은 데이터
# 양에 따라 조건부). KRX 영업일 기준 약 6.5개월 분량.
#
# 주의: services/benchmarks.py의 _BENCH_HOLD_MAP과 정합 유지.
_BENCH_LOOKBACK_BDAYS = 130


def _risk_max(current: str, candidate: str) -> str:
    """[v2.0 #1] 리스크 레벨 비교 — 정수 매핑 기반 (ASCII 함정 제거)

    Before: max("NORMAL", "CAUTION") → "NORMAL" (N > C in ASCII) ← 버그!
    After:  _RISK_LEVELS["CAUTION"]=1 > _RISK_LEVELS["NORMAL"]=0 → "CAUTION" ✅
    """
    cur_val = _RISK_LEVELS.get(current, 0)
    cand_val = _RISK_LEVELS.get(candidate, 0)
    return _RISK_NAMES[max(cur_val, cand_val)]


def _parse_ymd(trade_ymd: str) -> pd.Timestamp:
    """[v2.0 #4] 공통 날짜 파싱 (복붙 제거)"""
    ymd = str(trade_ymd).replace("-", "").replace("/", "")[:8]
    return pd.to_datetime(ymd, format="%Y%m%d")


def _calc_start_date(end_dt: pd.Timestamp, lookback_bdays: int = _LOOKBACK_BDAYS) -> str:
    """[v2.0 #2] 동적 시작일 — 하드코딩 '2024-01-01' 제거

    Before: "2024-01-01" 고정 → 2년치 데이터 매번 다운로드 + 2023년 백테스트 불가
    After:  end_dt에서 50 영업일 전으로 동적 계산 (약 2.5개월)
    """
    start_dt = end_dt - pd.offsets.BDay(lookback_bdays)
    return start_dt.strftime("%Y-%m-%d")


# [v2.2 #2] 데이터 신선도(Freshness) 허용 영업일 수
_MAX_STALE_BDAYS = 3


def _check_freshness(df: pd.DataFrame, end_dt: pd.Timestamp) -> Tuple[bool, int]:
    """[v2.2 #2] 데이터 신선도 체크 — stale 데이터 감지

    Returns: (is_fresh, stale_days)
      is_fresh: 최신 데이터가 end_dt 기준 _MAX_STALE_BDAYS 이내인지
      stale_days: end_dt와 최신 데이터 간 영업일 차이
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return False, 999

    last_date = df.index[-1]
    delta_days = (end_dt - last_date).days

    # 영업일 기준 환산 (주말 제외 근사)
    stale_bdays = max(0, int(delta_days * 5 / 7))

    return stale_bdays <= _MAX_STALE_BDAYS, stale_bdays


def _safe_fdr_fetch(
    symbol: str,
    end_dt: pd.Timestamp,
    lookback_bdays: int = _LOOKBACK_BDAYS,
    min_rows: int = 1,
) -> Optional[pd.DataFrame]:
    """[v2.0 #4+#5] 안전한 FDR 데이터 조회 — DRY + Fail-safe 통합

    - 동적 시작일 계산
    - min_rows 미달 시 None 반환
    - 예외 시 None (호출부에서 CAUTION 격상 판단)
    - [v2.1 #1] 미래 데이터 차단 (Look-ahead Bias 원천 봉쇄)
    """
    if not HAS_FDR:
        return None

    start_d = _calc_start_date(end_dt, lookback_bdays)
    end_d = end_dt.strftime("%Y-%m-%d")

    try:
        df = fdr.DataReader(symbol, start_d, end_d)
        if df is None or df.empty:
            return None

        # [v2.0 #3] 컬럼명 방어: 한글('종가') / 영문('Close') 모두 대응
        if "Close" not in df.columns and "종가" in df.columns:
            df = df.rename(columns={"종가": "Close", "시가": "Open",
                                     "고가": "High", "저가": "Low"})
        if "Close" not in df.columns:
            logger.warning(f"{symbol}: 'Close' 컬럼 없음 — {list(df.columns)}")
            return None

        # [v2.1 #1] 미래 데이터 차단: 백테스트 시 end_dt 이후 데이터 유입 방지
        if isinstance(df.index, pd.DatetimeIndex):
            df = df[df.index <= end_dt]
        elif df.index.dtype == "object":
            try:
                df.index = pd.to_datetime(df.index)
                df = df[df.index <= end_dt]
            except (ValueError, TypeError):
                pass  # 인덱스 변환 불가 시 원본 유지

        if len(df) < min_rows:
            return None

        return df
    except Exception as e:
        logger.warning(f"FDR 조회 실패 [{symbol}]: {e}")
        return None


# ═══════════════════════════════════════════════════
#  매크로 환경 판단
# ═══════════════════════════════════════════════════

def check_macro_env(
    trade_ymd: str,
    config: CollectorConfig = DEFAULT_CONFIG,
) -> Tuple[str, str, int, int]:
    """[v2.0] 환율/나스닥 기반 시장 위험도 판단.

    Returns: (risk_level, message, ebs_threshold, rec_limit)

    Fixes:
      #1 risk_level 정수 매핑 (ASCII max 버그 수정)
      #2 동적 조회 구간 (50 BDay)
      #3 시차 방어 (데이터 부족 시 안전 처리)
      #5 Fail-safe (API 실패 → CAUTION 격상, 침묵 금지)
    """
    if not HAS_FDR and not HAS_YF:
        return ("NORMAL", "FDR/yfinance 미설치", config.pass_ebs, config.rec_limit_default)

    end_dt = _parse_ymd(trade_ymd)
    risk_level = "NORMAL"
    messages = []
    data_failures = 0  # [v2.0 #5] 데이터 실패 카운터

    # ── 환율 (USD/KRW) ──
    fx = _safe_fdr_fetch("USD/KRW", end_dt, min_rows=1) if HAS_FDR else None
    if fx is None and HAS_YF:
        try:
            _yf = yf.download("KRW=X", period="1mo", interval="1d", progress=False, timeout=10)
            if _yf is not None and len(_yf) > 0:
                fx = _yf.rename(columns={"Close": "Close"} if "Close" in _yf.columns else {})
                logger.info("✅ yfinance 환율 폴백 성공")
        except Exception:
            pass
    if fx is not None:
        is_fresh, stale_days = _check_freshness(fx, end_dt)
        fx_last = float(fx["Close"].iloc[-1])
        fx_date = fx.index[-1].strftime("%m/%d") if isinstance(fx.index, pd.DatetimeIndex) else "?"

        if not is_fresh:
            # [v2.2 #2] stale 데이터 → 경고 + CAUTION 격상
            data_failures += 1
            messages.append(f"환율 데이터 {stale_days}일 지연 [{fx_date}] → stale")
            logger.warning(f"환율 데이터 stale: {stale_days} BDay 지연")
        elif fx_last >= config.macro_fx_critical:
            risk_level = "CRITICAL"
            messages.append(f"환율 {fx_last:.0f}원 [{fx_date}] (CRITICAL)")
        elif fx_last >= config.macro_fx_caution:
            risk_level = _risk_max(risk_level, "CAUTION")
            messages.append(f"환율 {fx_last:.0f}원 [{fx_date}] (주의)")
    else:
        data_failures += 1
        logger.warning("환율 데이터 조회 실패 — Fail-safe 적용")

    # ── 나스닥 전일 수익률 ──
    nq = _safe_fdr_fetch("IXIC", end_dt, min_rows=2) if HAS_FDR else None
    if nq is None and HAS_YF:
        try:
            _yf_nq = yf.download("^IXIC", period="1mo", interval="1d", progress=False, timeout=10)
            if _yf_nq is not None and len(_yf_nq) >= 2:
                nq = _yf_nq
                logger.info("✅ yfinance 나스닥 폴백 성공")
        except Exception:
            pass
    if nq is not None:
        close = nq["Close"].dropna()
        if len(close) >= 2:
            is_fresh, stale_days = _check_freshness(nq, end_dt)
            nq_ret = float(close.pct_change().iloc[-1] * 100)
            nq_date = close.index[-1].strftime("%m/%d") if isinstance(close.index, pd.DatetimeIndex) else "?"

            if not is_fresh:
                # [v2.2 #2] stale 데이터 → 수익률 판단 무시 + 경고
                data_failures += 1
                messages.append(f"나스닥 데이터 {stale_days}일 지연 [{nq_date}] → stale")
                logger.warning(f"나스닥 데이터 stale: {stale_days} BDay 지연")
            elif nq_ret <= config.macro_nasdaq_critical:
                risk_level = "CRITICAL"
                messages.append(f"나스닥 {nq_ret:+.1f}% [{nq_date}] (CRITICAL)")
            elif nq_ret <= config.macro_nasdaq_caution:
                risk_level = _risk_max(risk_level, "CAUTION")
                messages.append(f"나스닥 {nq_ret:+.1f}% [{nq_date}] (주의)")
        else:
            data_failures += 1
            logger.warning("나스닥 Close 데이터 부족 (< 2행)")
    else:
        data_failures += 1
        logger.warning("나스닥 데이터 조회 실패 — Fail-safe 적용")

    # [v2.0 #5] Fail-safe: 데이터 소스 하나라도 실패 시 보수적 CAUTION 격상
    if data_failures > 0 and risk_level == "NORMAL":
        risk_level = _risk_max(risk_level, "CAUTION")
        messages.append(f"데이터 조회 실패 {data_failures}건 → 보수적 CAUTION")

    # 위험도에 따라 EBS/추천수 조정
    ebs_thresh = config.pass_ebs
    rec_limit = config.rec_limit_default

    if risk_level == "CRITICAL":
        ebs_thresh = config.pass_ebs + 2
        rec_limit = config.rec_limit_caution
    elif risk_level == "CAUTION":
        ebs_thresh = config.pass_ebs + 1
        rec_limit = config.rec_limit_default

    msg = " | ".join(messages) if messages else "매크로 정상"
    return (risk_level, msg, ebs_thresh, rec_limit)


# ═══════════════════════════════════════════════════
#  시장 체온
# ═══════════════════════════════════════════════════

def compute_market_breadth(df: pd.DataFrame) -> Dict[str, float]:
    """시장 체온 (전종목 기준 상승/하락 비율)"""
    if df.empty:
        return {"ALL": 50.0}
    try:
        ret = df.get("ret_1d_%")
        if ret is None:
            return {"ALL": 50.0}
        up = (ret > 0).sum()
        total = len(ret.dropna())
        return {"ALL": round(up / max(total, 1) * 100, 1)}
    except Exception:
        return {"ALL": 50.0}


def label_market_temp(breadth_all: float) -> str:
    """시장 체온 텍스트"""
    if breadth_all >= 65:
        return "🔥 과열"
    elif breadth_all >= 55:
        return "🟢 활황"
    elif breadth_all >= 45:
        return "🟡 보통"
    elif breadth_all >= 35:
        return "🟠 냉각"
    else:
        return "🔵 침체"


# ═══════════════════════════════════════════════════
#  벤치마크 수익률
# ═══════════════════════════════════════════════════

def get_benchmark_returns(
    trade_ymd: str,
    config: CollectorConfig = DEFAULT_CONFIG,
) -> Dict[str, Dict[int, float]]:
    """[v2.0] 벤치마크(KOSPI/KOSDAQ) N일 수익률

    [v20.0] 3단계 폴백: FDR → yfinance → 캐시
    """
    end_dt = _parse_ymd(trade_ymd)
    result = {}

    # ── 1순위: FDR ──
    if HAS_FDR:
        for name, code in [("KOSPI", "KS11"), ("KOSDAQ", "KQ11")]:
            # [v3.9.15e + 2] _LOOKBACK_BDAYS(50) → _BENCH_LOOKBACK_BDAYS(130)
            # 60일 보유기간 수익률 계산을 위한 lookback 확장
            df = _safe_fdr_fetch(code, end_dt, lookback_bdays=_BENCH_LOOKBACK_BDAYS, min_rows=2)
            if df is None:
                logger.warning(f"벤치마크 {name}({code}) FDR 조회 실패")
                continue
            close = df["Close"].dropna()
            if len(close) < 2:
                continue
            last = float(close.iloc[-1])
            rets = {}
            for d in [1, 3, 5, 10, 20, 60, 120]:
                if len(close) > d:
                    rets[d] = round((last / float(close.iloc[-(d + 1)]) - 1) * 100, 2)
            result[name] = rets

    # ── 2순위: yfinance (FDR 실패 시) ──
    if len(result) < 2 and HAS_YF:
        logger.info("🔄 벤치마크 FDR 실패 → yfinance 폴백")
        for name, ticker in [("KOSPI", "^KS11"), ("KOSDAQ", "^KQ11")]:
            if name in result:
                continue
            try:
                # [v3.9.15e + 2] 6mo → 1y (60일 보유기간 안정 계산)
                yf_df = yf.download(ticker, period="1y", interval="1d",
                                     progress=False, timeout=10)
                if yf_df is not None and len(yf_df) > 20:
                    close = yf_df["Close"].dropna()
                    if hasattr(close, 'columns'):
                        close = close.iloc[:, 0]
                    last = float(close.iloc[-1])
                    rets = {}
                    for d in [1, 3, 5, 10, 20, 60, 120]:
                        if len(close) > d:
                            rets[d] = round((last / float(close.iloc[-(d + 1)]) - 1) * 100, 2)
                    result[name] = rets
                    logger.info(f"✅ yfinance 벤치마크 {name} 성공 ({len(close)}일)")
            except Exception as e:
                logger.warning(f"⚠️ yfinance {name} 실패: {e}")

    # ── 3순위: 로컬 캐시 ──
    if len(result) < 2:
        cached = _load_bench_cache()
        if cached:
            for k, v in cached.items():
                if k not in result:
                    result[k] = v
            logger.info(f"📂 벤치마크 캐시 폴백 사용: {list(cached.keys())}")

    # 성공 시 캐시 저장
    if result:
        _save_bench_cache(result)

    return result


def _save_bench_cache(bench_map: dict) -> None:
    """벤치마크 캐시 저장"""
    import json
    try:
        path = os.path.join(os.path.dirname(__file__), "data", "bench_cache_latest.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(bench_map, f)
    except Exception:
        pass


def _load_bench_cache() -> dict:
    """벤치마크 캐시 로드"""
    import json
    try:
        path = os.path.join(os.path.dirname(__file__), "data", "bench_cache_latest.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}
