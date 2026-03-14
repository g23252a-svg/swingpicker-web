# -*- coding: utf-8 -*-
"""
shared_log.py — 공용 로깅 + 유틸 + 상수
═══════════════════════════════════════════
[v20.2] collector.py 의존 제거를 위한 공용 모듈
pipeline_* 파일들이 collector 대신 여기서 import

사용법:
    from shared_log import log, OUT_DIR, UTF8, ensure_dir, safe_quantile
"""

import os
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

# ── collector_config SSOT ──
from collector_config import DEFAULT_CONFIG as _CFG, Route

# ── 상수 ──
LOOKBACK_DAYS = _CFG.lookback_days
BENCH_LOOKBACK_DAYS = _CFG.bench_lookback_days
TOP_N = _CFG.top_n
MIN_TURNOVER_EOK = _CFG.min_turnover_eok
MIN_MCAP_EOK = _CFG.min_mcap_eok
OUT_DIR = _CFG.out_dir
BASE_DIR = _CFG.base_dir
UTF8 = "utf-8-sig"
MAX_WORKERS = int(os.environ.get("LDY_WORKERS", str(_CFG.max_workers)))

# ── pykrx 가용성 ──
try:
    from pykrx import stock  # noqa: F401
    PYKRX_OK = True
except Exception:
    PYKRX_OK = False

# ── LLM 가용성 ──
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LLM_AVAILABLE = False
_USE_NEW_GENAI = False

try:
    from google import genai as _genai_client  # noqa: F401
    if GEMINI_API_KEY:
        _USE_NEW_GENAI = True
        LLM_AVAILABLE = True
except ImportError:
    _genai_client = None

if not LLM_AVAILABLE:
    try:
        import google.generativeai as genai  # noqa: F401
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            LLM_AVAILABLE = True
    except ImportError:
        LLM_AVAILABLE = False

# ── 로깅 ──
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LDY_Collector")

# [v20.6.5] pykrx 내부 JSON parse 에러 로그 억제 (FDR 폴백이 처리)
logging.getLogger("pykrx").setLevel(logging.WARNING)


def log(msg: str) -> None:
    """기존 호환성 유지 래퍼"""
    logger.info(msg)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_quantile(s, q, fallback=0.0):
    """Pandas Series에서 안전하게 분위수 계산"""
    if s is None:
        return fallback
    try:
        if hasattr(s, 'empty') and s.empty:
            return fallback
        v = s.quantile(q)
        return fallback if pd.isna(v) else float(v)
    except Exception:
        return fallback


# ── RunContext ──
@dataclass
class RunContext:
    """매크로 필터 등 런타임에 동적으로 변하는 상태"""
    pass_ebs: float = _CFG.pass_ebs
    rec_limit_cnt: int = 20
    macro_risk: str = "NORMAL"
    macro_msg: str = ""
