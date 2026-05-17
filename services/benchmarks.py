"""
services/benchmarks.py
======================
[v3.9.15c] 벤치마크 (KOSPI/KOSDAQ) 데이터 SSOT.
[v3.9.15d] hold_days 슬라이더 → bench key 매핑 + lru_cache 추가.
[v3.9.15e] daily KOSPI CSV 가이드 보강 (ret_1d_% / ret_60d_% 누락 수정) +
           41~60일 구간 정직한 None 반환 (가짜 20일 매핑 절대 금지).

이전엔 tab_perf.py의 helper를 tab_backtest.py가 직접 import해서 사용 —
UI 컴포넌트끼리 helper를 당겨 쓰는 구조는 순환 import / 테스트 격리 문제 위험.

이 모듈로 분리하면 tab_perf / tab_backtest 둘 다 services를 통해서만 접근.

데이터 한계 (현재):
- bench_cache_latest.json은 "오늘 기준 N일 KOSPI 수익률" 1개 시점만
- 백테스트 전체 기간 일자별 KOSPI 시계열 데이터는 아직 없음
- 따라서 현재 알파는 '간이 알파' (전략 거래당 평균 vs KOSPI hold_days)
- 진짜 알파 (거래일별 매수/매도 시점 KOSPI 차감)는 daily KOSPI CSV 추가 후 가능
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Optional

_logger = logging.getLogger(__name__)

# DATA_DIR 자동 탐색 (어디서 import되든 동작)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_ROOT, "data")


def _get_bench_file_mtime() -> float:
    """bench_cache_latest.json의 최종 수정 시각.
    
    파일이 갱신되면 lru_cache를 무효화하기 위한 키.
    """
    dirs_to_try = [DATA_DIR, os.path.join(os.getcwd(), "data"), "data"]
    for d in dirs_to_try:
        path = os.path.join(d, "bench_cache_latest.json")
        if os.path.exists(path):
            try:
                return os.path.getmtime(path)
            except Exception:
                return 0.0
    return 0.0


@lru_cache(maxsize=4)
def _load_bench_cache_cached(mtime_key: float) -> dict:
    """[v3.9.15d] 내부 캐시 helper — mtime 기반 자동 무효화.
    
    mtime_key가 같으면 lru_cache가 같은 dict 반환.
    파일 갱신되면 mtime 변경 → 새 dict 로드.
    """
    dirs_to_try = [DATA_DIR, os.path.join(os.getcwd(), "data"), "data"]
    for d in dirs_to_try:
        path = os.path.join(d, "bench_cache_latest.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                result = {}
                for index_name, hold_data in raw.items():
                    if isinstance(hold_data, dict):
                        result[index_name] = {
                            int(k): float(v)
                            for k, v in hold_data.items()
                            if str(k).isdigit()
                        }
                _logger.info(
                    f"📊 벤치마크 로드: {list(result.keys())} "
                    f"(보유기간: {sorted(result.get('KOSPI', {}).keys())})"
                )
                return result
            except Exception as e:
                _logger.warning(f"bench_cache 로드 실패 ({path}): {e}")
                return {}
    _logger.info("bench_cache_latest.json 없음 — 알파 미표시")
    return {}


def load_bench_cache() -> dict:
    """bench_cache_latest.json 로드 — KOSPI/KOSDAQ 보유기간별 수익률.
    
    [v3.9.15d] lru_cache 적용 — 같은 mtime이면 재로드 안 함 (v3.9.16 프리셋
    비교표가 4번 호출해도 1번만 디스크 읽음).
    
    파일 형식:
        {
            "KOSPI": {"1": -0.0, "3": 1.36, "5": 4.58, "10": 10.53, "20": 19.06},
            "KOSDAQ": {...}
        }
    
    Returns:
        {"KOSPI": {1: -0.0, 5: 4.58, ...}, ...} (정수 키로 변환)
        파일 없으면 빈 dict
    """
    mtime = _get_bench_file_mtime()
    return _load_bench_cache_cached(mtime)


# [v3.9.15d] hold_days 슬라이더 → bench key 매핑
# tab_backtest._RET_MAP과 동일 임계값 사용 (정합성 보장)
# 슬라이더 12일을 그대로 bench.get(12)하면 None — 가장 가까운 key (10일)로 매핑해야
#
# [v3.9.15e] 41~60일은 60 키로 매핑하되, bench_cache_latest.json에 60 키가
# 없으면 정직하게 None을 반환할 것 (가짜로 20 키로 끌어쓰지 말 것).
# 이유: 50일 보유 종목 알파를 20일 KOSPI 수익률과 비교하면 systematic하게
# 왜곡된 거짓 알파가 됨. simple alpha 없음 < 거짓 alpha. 1·3·5·10·20·60일을
# cache 또는 daily CSV에 일관되게 갖춰서 해결할 것.
_BENCH_HOLD_MAP = [
    (3,   1),     # 1~3일   → KOSPI 1일
    (7,   5),     # 4~7일   → KOSPI 5일
    (15,  10),    # 8~15일  → KOSPI 10일
    (40,  20),    # 16~40일 → KOSPI 20일
    (999, 60),    # 41~+일  → KOSPI 60일 (cache 없으면 None — 가짜 매핑 금지)
]


def map_slider_to_bench_key(hold_days: int) -> int:
    """[v3.9.15d] 사용자 슬라이더 값을 가장 가까운 bench key로 매핑.
    
    예시:
        12일 → 10 (KOSPI 10일 수익률 사용)
        17일 → 20 (KOSPI 20일 수익률 사용)
        40일 → 20 (KOSPI 20일 수익률 사용)
        50일 → 60 (KOSPI 60일 수익률 — 데이터 없을 수 있음)
    
    tab_backtest._RET_MAP과 동일 임계값을 써서 전략 수익률 컬럼 매핑과
    KOSPI key 매핑이 정합.
    """
    for threshold, key in _BENCH_HOLD_MAP:
        if hold_days <= threshold:
            return key
    return 60  # fallback


def get_kospi_return(bench_data: dict, hold_days: int) -> Optional[float]:
    """특정 보유기간의 KOSPI 수익률 추출.
    
    [v3.9.15d] hold_days를 map_slider_to_bench_key()로 정규화 후 조회.
    예: 12일 슬라이더 → KOSPI 10일 수익률.
    
    Returns: KOSPI 수익률(%) 또는 None
    """
    if not bench_data or "KOSPI" not in bench_data:
        return None
    key = map_slider_to_bench_key(int(hold_days))
    return bench_data["KOSPI"].get(key)


# ═══════════════════════════════════════════════════════════════
# [v3.9.15d] 진짜 일자별 알파 — daily KOSPI CSV 추가 후 활성화
# ═══════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def _load_kospi_daily_cached(mtime_key: float):
    """일자별 KOSPI CSV 로드 (있으면).
    
    필요 형식: data/kospi_daily.csv
        date(YYYYMMDD), close, ret_1d_%, ret_5d_%, ret_10d_%, ret_20d_%
    
    Returns: DataFrame indexed by date, 또는 None
    """
    import pandas as pd
    dirs_to_try = [DATA_DIR, os.path.join(os.getcwd(), "data"), "data"]
    for d in dirs_to_try:
        path = os.path.join(d, "kospi_daily.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df["date"] = df["date"].astype(str).str.replace("-", "")
                df = df.set_index("date")
                _logger.info(f"📊 KOSPI daily 로드: {len(df)}일")
                return df
            except Exception as e:
                _logger.warning(f"kospi_daily.csv 로드 실패: {e}")
                return None
    return None


def load_kospi_daily():
    """daily KOSPI CSV 로드 (mtime 캐시 무효화).
    
    Returns: DataFrame or None.
    
    데이터가 없으면 None — 호출자는 fallback (간이 알파) 사용 권장.
    """
    dirs_to_try = [DATA_DIR, os.path.join(os.getcwd(), "data"), "data"]
    mtime = 0.0
    for d in dirs_to_try:
        path = os.path.join(d, "kospi_daily.csv")
        if os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                break
            except OSError as e:
                # mtime 조회 실패 (권한/fs 이슈) — cache 동작엔 영향 없음 (mtime=0 fallback)
                _logger.debug(f"[bench cache] {path} mtime 조회 실패: {e}")
    return _load_kospi_daily_cached(mtime)


def get_kospi_return_for_date(date_str: str, hold_days: int) -> Optional[float]:
    """[v3.9.15d] 진짜 거래일별 KOSPI 수익률.
    
    daily KOSPI CSV가 있으면 date_str(YYYYMMDD) 시점의 hold_days 보유 수익률.
    없으면 None (호출자가 간이 알파로 fallback).
    
    Args:
        date_str: "20260513" 형식
        hold_days: 슬라이더 값 (1~120)
    
    Returns: KOSPI 수익률(%) 또는 None
    
    [v3.9.15e] data/kospi_daily.csv 수집 가이드 — 필수 컬럼 누락 주의:
    
    슬라이더 1~3일 → ret_1d_% 필요 (없으면 1~3일 보유 시 None)
    슬라이더 4~7일 → ret_5d_%
    슬라이더 8~15일 → ret_10d_%
    슬라이더 16~40일 → ret_20d_%
    슬라이더 41~60일 → ret_60d_% 필요 (없으면 41~60일 보유 시 None)
    
    pykrx 예제:
        import pandas as pd
        from pykrx import stock
        df = stock.get_index_ohlcv("20210101", "20260516", "1001")  # KOSPI
        df = df.reset_index().rename(columns={"날짜": "date", "종가": "close"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        for n in [1, 3, 5, 10, 20, 60]:    # 1, 60 빼먹지 말 것
            df[f"ret_{n}d_%"] = df["close"].pct_change(n).shift(-n) * 100
        df.to_csv("data/kospi_daily.csv", index=False, encoding="utf-8-sig")
    
    원칙: 데이터 없으면 None 반환 — 호출자가 simple alpha로 fallback 하거나
    "(알파 없음)"으로 표시. 절대 다른 기간 KOSPI로 대체 매핑하지 말 것
    (50일 보유한 종목 알파를 20일 KOSPI와 비교하면 systematic하게 왜곡된
    거짓 알파가 됨).
    """
    df = load_kospi_daily()
    if df is None:
        return None
    # 보유기간 → 적절한 ret 컬럼
    key = map_slider_to_bench_key(hold_days)
    col = f"ret_{key}d_%"
    if col not in df.columns:
        # [v3.9.15e] 정직한 None — 가짜 컬럼 매핑 금지
        return None
    date_key = str(date_str).replace("-", "")
    if date_key not in df.index:
        return None
    val = df.loc[date_key, col]
    try:
        if val is None or (hasattr(val, "__class__") and val.__class__.__name__ == "float64" and val != val):
            # NaN 체크 (pd.isna 의존 회피)
            return None
        return float(val)
    except Exception:
        return None

