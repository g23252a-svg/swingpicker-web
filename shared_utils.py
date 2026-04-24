# -*- coding: utf-8 -*-
"""
shared_utils.py — collector.py / dashboard.py 공용 유틸리티
─────────────────────────────────────────────────────────────
양쪽에서 복붙되어 있던 함수들을 단일 소스로 통합합니다.
"""
import math
import numpy as np
import pandas as pd
from typing import Any

# ───────────────────── 수치 안전 변환 ─────────────────────

def nz_num(s: Any) -> pd.Series:
    """문자열 혼합 Series → 숫자 변환 (실패 시 NaN)"""
    return pd.to_numeric(s, errors="coerce")


def safe_float(x, default: float = 0.0) -> float:
    """단일 값 → float 안전 변환 (NaN/None → default)"""
    try:
        if x is None:
            return default
        val = float(x)
        return default if (math.isnan(val) or math.isinf(val)) else val
    except Exception:
        return default


def _safe_sum(x: pd.Series) -> float:
    return pd.to_numeric(x, errors="coerce").fillna(0).sum()


def safe_quantile(s, q: float, fallback: float = 0.0) -> float:
    """Pandas Series 안전 분위수 (빈 데이터/에러 시 fallback)"""
    if s is None:
        return fallback
    try:
        if hasattr(s, 'empty') and s.empty:
            return fallback
        v = s.quantile(q)
        return fallback if pd.isna(v) else float(v)
    except Exception:
        return fallback


# ───────────────────── 이동평균 (공용) ─────────────────────

def ema(s: pd.Series, span: int) -> pd.Series:
    """지수 이동 평균"""
    return s.ewm(span=span, adjust=False).mean()


def wma(s: pd.Series, period: int) -> pd.Series:
    """가중 이동 평균 (HMA 계산 기반)"""
    weights = np.arange(1, period + 1)

    def _calc(x):
        return np.dot(x, weights) / weights.sum()

    return s.rolling(period).apply(_calc, raw=True)


def calc_hma(s: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average (HMA) — 빠른 반응 + 낮은 휩소"""
    if len(s) < period:
        return pd.Series(np.nan, index=s.index)

    half_length = int(period / 2)
    sqrt_length = int(math.sqrt(period))

    wma_half = wma(s, half_length)
    wma_full = wma(s, period)

    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt_length)


# ───────────────────── 정규화 / 클리핑 ─────────────────────

def cap_q(s: pd.Series, q: int = 90, floor: float = 1.0) -> float:
    c = np.nanpercentile(nz_num(s), q)
    return float(max(c, floor)) if np.isfinite(c) else floor


def pct_norm_pos(s: pd.Series, q: int = 90, floor: float = 1.0) -> pd.Series:
    s = nz_num(s).clip(lower=0)
    return np.clip(s / cap_q(s, q, floor), 0, 1)


def inv_dist_norm(dist: pd.Series, cap: float) -> pd.Series:
    return np.clip(1 - (nz_num(dist) / cap), 0, 1)

# ═══════════════════════════════════════════════════════════════
#  [v22] 설계 종료판 추가 함수 5종
#  ─ 참조: SwingPicker_v22_Final_Consolidated.md §2.1
# ═══════════════════════════════════════════════════════════════
import logging
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_v22_logger = logging.getLogger(__name__)


# ─── 2.1.1 ATR 단위 정규화 ──────────────────────────────────
def _normalize_atr_pct(v) -> float:
    """ATR을 decimal ratio로 정규화 (decimal/percentage 혼용 방어).

    경계: f >= 1.0이면 percentage(5.0=5%)로 간주하여 /100,
          f < 1.0이면 decimal(0.05=5%)로 간주하여 그대로.

    이유: 코드베이스 내 혼용 존재 —
      - ml_engine.add_technical_features_batch: ATR_Pct = atr/close → decimal
      - stop_logic.calc_stop_price: atr_pct = (atr/buy)*100 → percentage
    한국 주식 ATR 실전 분포 1~15% 기준, 1.0은 "1% (percentage)"로 해석하는 게
    1을 100%로 오해하는 것보다 훨씬 안전. 진짜 100% decimal은 실전 부재.
    """
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(f) or f < 0:
        return 0.0
    return f / 100.0 if f >= 1.0 else f


# ─── 2.1.1b 종목코드 정규화 ──────────────────────────────────
def _normalize_stock_code(c) -> str:
    """종목코드를 6자리 숫자 문자열로 정규화.
    
    처리:
      - None / pd.NA / NaN → ""
      - float("5930.0") → "5930" → "005930"
      - " 005930 " → "005930" (공백 제거)
      - 숫자 외 문자 제거 후 zfill(6)
      - 모든 문자 제거 후 빈 문자열이면 "" 반환 (zfill 안 함)
    
    market map / benchmark index / coverage 계산에서 공용 사용.
    """
    if c is None:
        return ""
    try:
        if pd.isna(c):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(c).strip()
    # float 문자열 방어: "5930.0" → "5930"
    if s.endswith(".0"):
        s = s[:-2]
    # 숫자만 추출
    s = "".join(ch for ch in s if ch.isdigit())
    return s.zfill(6) if s else ""


# ─── 2.1.2 시장(KOSPI/KOSDAQ) 매핑 ───────────────────────────
@lru_cache(maxsize=1)
def _load_market_map() -> dict:
    """KRX 종목코드 → 시장구분 매핑.
    
    우선순위:
      1. data/sector_map_krx.csv의 시장/MARKET 컬럼
      2. 최근 price_snapshot_*.csv의 시장/MARKET 컬럼 (code/ticker 대체명 지원)
      3. 빈 맵 (get_benchmark_index가 KOSPI 기본 반환)
    
    운영 주의: @lru_cache(maxsize=1)로 프로세스 수명 내 고정 캐시.
    collector가 sector_map_krx.csv를 갱신한 뒤 동일 프로세스에서 다시 읽어야
    한다면, 갱신 직후 `_load_market_map.cache_clear()` 호출 필요.
    """
    # 1순위: sector_map_krx.csv
    p = Path("data/sector_map_krx.csv")
    if p.exists():
        try:
            df = pd.read_csv(p, dtype={"종목코드": str})
            code_col = next((c for c in ["종목코드", "code", "ticker"] 
                             if c in df.columns), None)
            mkt_col = next((c for c in ["시장", "MARKET", "market"] 
                            if c in df.columns), None)
            if code_col and mkt_col:
                # [v22 defensive] 정규화된 코드로 통일
                df[code_col] = df[code_col].apply(_normalize_stock_code)
                # 빈 코드 (파싱 실패) 제거
                df = df[df[code_col] != ""]
                result = dict(zip(
                    df[code_col],
                    df[mkt_col].astype(str).str.upper().str.strip()
                ))
                _v22_logger.info(
                    f"✅ 시장 메타 로드: {len(result)}건 (sector_map_krx.csv)"
                )
                return result
        except Exception as e:
            _v22_logger.warning(f"sector_map_krx.csv 파싱 실패: {e}")
    
    # 2순위: 최근 price_snapshot 3개 시도
    for snap in sorted(Path("data").glob("price_snapshot_2*.csv"), reverse=True)[:3]:
        try:
            df = pd.read_csv(snap, dtype=str)   # 모든 컬럼 str로 읽어 정규화 안전
            code_col = next((c for c in ["종목코드", "code", "ticker"] 
                             if c in df.columns), None)
            mkt_col = next((c for c in ["시장", "MARKET", "market"] 
                            if c in df.columns), None)
            if code_col and mkt_col:
                df[code_col] = df[code_col].apply(_normalize_stock_code)
                df = df[df[code_col] != ""]
                result = dict(zip(
                    df[code_col],
                    df[mkt_col].astype(str).str.upper().str.strip()
                ))
                _v22_logger.info(
                    f"✅ 시장 메타 로드: {len(result)}건 ({snap.name})"
                )
                return result
        except Exception:
            continue
    
    _v22_logger.warning(
        "⚠️ 시장 구분 메타 없음 — 전종목 KOSPI 간주 (벤치 신뢰도 낮음)"
    )
    return {}


def get_benchmark_index(code) -> str:
    """종목코드 → 'KOSPI' | 'KOSDAQ'. 
    
    [v22 defensive] code는 int/float/str/None 모두 허용 — _normalize_stock_code로 처리.
    KOSDAQ GLOBAL도 KOSDAQ로. KONEX / UNKNOWN은 KOSPI 기본 (더 안정적 지수).
    """
    norm = _normalize_stock_code(code)
    if not norm:
        return "KOSPI"
    m = _load_market_map().get(norm, "UNKNOWN")
    return "KOSDAQ" if "KOSDAQ" in m else "KOSPI"


# ─── 2.1.2b Route 타입 정규화 ───────────────────────────────
def route_name(v) -> str:
    """Route enum / str / None → 표준 문자열 (예: 'ATTACK').
    
    [v22] Route(str, Enum) 이미 str과 비교 가능하지만, 명시적 normalizer로
    enum 갱신(e.g. IntEnum 전환) 등 미래 변경에 대비한 방어 헬퍼.
    
    사용 예:
        route_name(Route.ATTACK) == "ATTACK"     # True
        route_name("ATTACK") == "ATTACK"          # True
        route_name(None) == ""                    # True (안전 기본값)
    """
    if v is None:
        return ""
    # Enum이면 .value, str이면 그대로
    val = getattr(v, "value", v)
    return str(val).strip().upper()


# ─── 2.1.3 Wilson Lower Confidence Bound ─────────────────────
def wilson_lcb(wins: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval 하한 (기본 95% 신뢰구간).
    
    단조성 게이트에서 raw p_win 비교 대신 사용 — 표본 흔들림에 강함.
    """
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - margin) / denom


# ─── 2.1.4 IS_NOW_ENTRY 적응형 판정 ──────────────────────────
def compute_is_now_entry(close: float, entry: float,
                         atr_pct, mcap_eok: float) -> int:
    """현재가와 추천매수가의 근접도 판정.
    
    시총 기반 base + ATR 반영 + 0.8% hard cap.
    atr_pct는 decimal(0.05) OR percentage(5.0) 둘 다 허용 (내부 정규화).
    
    시총 기본폭:
      - 5조+ 대형주:      0.3%
      - 5천억~5조 중형주: 0.5%
      - 5천억 미만 소형주: 0.8%
    ATR 반영: max(base, ATR_decimal * 0.15), 최종 0.8% cap.
    """
    if entry <= 0:
        return 0
    atr = _normalize_atr_pct(atr_pct)

    if mcap_eok is None or mcap_eok <= 0:
        base_pct = 0.005   # 시총 미상 → 중형주 기준
    elif mcap_eok >= 50_000:
        base_pct = 0.003
    elif mcap_eok >= 5_000:
        base_pct = 0.005
    else:
        base_pct = 0.008
    
    adaptive_pct = min(0.008, max(base_pct, atr * 0.15))
    diff_pct = abs(close - entry) / max(entry, 1)
    return int(diff_pct <= adaptive_pct)


# ─── 2.1.5 Market map coverage 진단 ──────────────────────────
def compute_market_map_coverage(codes) -> dict:
    """종목코드 입력에 대한 시장 매핑 커버리지 통계.
    
    daily_briefing의 monotonicity_report에 포함되어
    avg_ret_excess 신뢰도 판단에 사용.
    
    [v22 defensive] 입력 처리:
      - None → 빈 입력으로 처리
      - pd.DataFrame → "종목코드" 컬럼 자동 추출
      - pd.Series → tolist()
      - list/tuple/set → 그대로
      - 그 외 iterable → list()
      - 빈 입력 → coverage=1.0, warning 없음 (휴장/장애일 false-alarm 방지)
      - 각 코드는 _normalize_stock_code 통과 후 dedup
    """
    # 입력 정규화 (Series ambiguity 방어)
    if codes is None:
        raw_codes = []
    elif isinstance(codes, pd.DataFrame):
        raw_codes = (codes["종목코드"].tolist() 
                     if "종목코드" in codes.columns else [])
    elif isinstance(codes, pd.Series):
        raw_codes = codes.tolist()
    elif isinstance(codes, (list, tuple, set)):
        raw_codes = list(codes)
    else:
        try:
            raw_codes = list(codes)
        except TypeError:
            raw_codes = []
    
    # 빈 입력 early-return — 운영 false-alarm 차단
    if len(raw_codes) == 0:
        return {
            "market_map_coverage": 1.0,
            "market_unknown_count": 0,
            "market_unknown_samples": [],
            "benchmark_mapping_warning": None,
        }
    
    mmap = _load_market_map()
    # 정규화 + 빈 코드 제외 + dedup + 정렬 (결정론적 출력)
    normed = [_normalize_stock_code(c) for c in raw_codes]
    code_list = sorted(set(c for c in normed if c))
    
    # 모든 코드가 정규화 후 빈 문자열이면 (예: NaN만 가득) 빈 입력 취급
    if not code_list:
        return {
            "market_map_coverage": 1.0,
            "market_unknown_count": 0,
            "market_unknown_samples": [],
            "benchmark_mapping_warning": None,
        }
    
    known = [c for c in code_list if c in mmap]
    unknown = [c for c in code_list if c not in mmap]
    coverage = len(known) / len(code_list)
    
    warning = None
    if coverage < 0.80:
        warning = (f"🚨 시장 매핑 커버리지 {coverage:.1%} < 80% — "
                   f"avg_ret_excess 신뢰도 낮음. collector 메타 갱신 필요.")
    elif coverage < 0.95:
        warning = (f"⚠️ 시장 매핑 커버리지 {coverage:.1%} — "
                   f"{len(unknown)}개 종목 KOSPI 기본 적용됨.")
    
    return {
        "market_map_coverage": round(coverage, 4),
        "market_unknown_count": len(unknown),
        "market_unknown_samples": unknown[:10],
        "benchmark_mapping_warning": warning,
    }
