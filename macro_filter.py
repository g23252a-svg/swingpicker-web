# -*- coding: utf-8 -*-
"""
macro_filter.py — 매크로 환경 판단 + 벤치마크
═══════════════════════════════════════════════════
[v14] #9 collector.py 분할 — 매크로 관련 함수 모음
"""
import os
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from collector_config import CollectorConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def check_macro_env(
    trade_ymd: str,
    config: CollectorConfig = DEFAULT_CONFIG,
) -> Tuple[str, str, int, int]:
    """
    환율/나스닥/VIX 기반 시장 위험도 판단.
    Returns: (risk_level, message, ebs_threshold, rec_limit)
    - risk_level: "NORMAL" / "CAUTION" / "CRITICAL"
    """
    try:
        import FinanceDataReader as fdr
    except ImportError:
        return ("NORMAL", "FDR 미설치", config.pass_ebs, config.rec_limit_default)

    risk_level = "NORMAL"
    messages = []
    ebs_thresh = config.pass_ebs
    rec_limit = config.rec_limit_default

    try:
        # 환율 (USD/KRW)
        end_d = f"{trade_ymd[:4]}-{trade_ymd[4:6]}-{trade_ymd[6:]}"
        fx = fdr.DataReader("USD/KRW", "2024-01-01", end_d)
        if fx is not None and not fx.empty:
            fx_last = float(fx["Close"].iloc[-1])
            if fx_last >= config.macro_fx_critical:
                risk_level = "CRITICAL"
                messages.append(f"환율 {fx_last:.0f}원 (CRITICAL)")
            elif fx_last >= config.macro_fx_caution:
                risk_level = max(risk_level, "CAUTION")
                messages.append(f"환율 {fx_last:.0f}원 (주의)")
    except Exception as e:
        logger.debug(f"환율 조회 실패: {e}")

    try:
        # 나스닥 전일 수익률
        end_d = f"{trade_ymd[:4]}-{trade_ymd[4:6]}-{trade_ymd[6:]}"
        nq = fdr.DataReader("IXIC", "2024-01-01", end_d)
        if nq is not None and len(nq) >= 2:
            nq_ret = (float(nq["Close"].iloc[-1]) / float(nq["Close"].iloc[-2]) - 1) * 100
            if nq_ret <= config.macro_nasdaq_critical:
                risk_level = "CRITICAL"
                messages.append(f"나스닥 {nq_ret:+.1f}% (CRITICAL)")
            elif nq_ret <= config.macro_nasdaq_caution:
                risk_level = max(risk_level, "CAUTION")
                messages.append(f"나스닥 {nq_ret:+.1f}% (주의)")
    except Exception as e:
        logger.debug(f"나스닥 조회 실패: {e}")

    # 위험도에 따라 EBS/추천수 조정
    if risk_level == "CRITICAL":
        ebs_thresh = config.pass_ebs + 2
        rec_limit = config.rec_limit_caution
    elif risk_level == "CAUTION":
        ebs_thresh = config.pass_ebs + 1
        rec_limit = config.rec_limit_default

    msg = " | ".join(messages) if messages else "매크로 정상"
    return (risk_level, msg, ebs_thresh, rec_limit)


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


def get_benchmark_returns(
    trade_ymd: str,
    config: CollectorConfig = DEFAULT_CONFIG,
) -> Dict[str, Dict[int, float]]:
    """벤치마크(KOSPI/KOSDAQ) N일 수익률"""
    try:
        import FinanceDataReader as fdr
    except ImportError:
        return {}

    result = {}
    for name, code in [("KOSPI", "KS11"), ("KOSDAQ", "KQ11")]:
        try:
            end_d = f"{trade_ymd[:4]}-{trade_ymd[4:6]}-{trade_ymd[6:]}"
            df = fdr.DataReader(code, "2024-01-01", end_d)
            if df is None or df.empty:
                continue
            close = df["Close"]
            last = float(close.iloc[-1])
            rets = {}
            for d in [1, 3, 5, 10, 20]:
                if len(close) > d:
                    rets[d] = round((last / float(close.iloc[-(d+1)]) - 1) * 100, 2)
            result[name] = rets
        except Exception:
            pass
    return result
