# -*- coding: utf-8 -*-
"""
data_source.py — 데이터소스 추상화 + Parquet 캐시
═══════════════════════════════════════════════════
[v14] #10 pickle→Parquet + #12 pykrx/FDR 추상화

설계 원칙:
- KRXDataSource 클래스가 pykrx → FDR → fallback 재시도를 한 곳에서 처리
- 캐시는 Parquet(기본) / pickle(레거시) 양쪽 지원, config.cache_format으로 전환
- collector.py에서는 data_source.get_ohlcv() 등 고수준 API만 호출
"""
import os
import time
import pickle
import logging
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  유틸
# ═══════════════════════════════════════════════════

def _backoff_sleep(attempt: int, base: float = 0.35, cap: float = 2.0) -> None:
    time.sleep(min(base * (2 ** attempt), cap))


def _pykrx_df(fn, *args, max_retries: int = 3, **kwargs) -> Optional[pd.DataFrame]:
    """pykrx 함수 호출 + 자동 재시도"""
    for i in range(max_retries):
        try:
            df = fn(*args, **kwargs)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.debug(f"pykrx retry {i+1}/{max_retries}: {e}")
            _backoff_sleep(i)
    return None


# ═══════════════════════════════════════════════════
#  KRXDataSource — 통합 데이터소스
# ═══════════════════════════════════════════════════

class KRXDataSource:
    """
    pykrx/FDR 통합 추상화.
    - get_ohlcv(), get_ohlcv_by_ticker(), get_market_cap() 등 고수준 API 제공
    - provider 우선순위: pykrx → FDR → empty fallback
    - 실패 시 자동 재시도 + 로깅
    """

    def __init__(self):
        try:
            from pykrx import stock as _stock
            self._pykrx = _stock
            self._pykrx_ok = True
        except ImportError:
            self._pykrx = None
            self._pykrx_ok = False

        try:
            import FinanceDataReader as _fdr
            self._fdr = _fdr
            self._fdr_ok = True
        except ImportError:
            self._fdr = None
            self._fdr_ok = False

    # ── OHLCV (종목별, 기간) ──
    def get_ohlcv(self, code: str, start_ymd: str, end_ymd: str) -> Optional[pd.DataFrame]:
        """종목 OHLCV 조회. pykrx → FDR fallback."""
        # pykrx
        if self._pykrx_ok:
            df = _pykrx_df(self._pykrx.get_market_ohlcv, start_ymd, end_ymd, code)
            if df is not None and not df.empty:
                return df

        # FDR fallback
        if self._fdr_ok:
            try:
                s = f"{start_ymd[:4]}-{start_ymd[4:6]}-{start_ymd[6:]}"
                e = f"{end_ymd[:4]}-{end_ymd[4:6]}-{end_ymd[6:]}"
                df = self._fdr.DataReader(code, s, e)
                if df is not None and not df.empty:
                    return df
            except Exception as ex:
                logger.debug(f"FDR fallback failed for {code}: {ex}")

        return None

    # ── 일별 전종목 OHLCV ──
    def get_ohlcv_by_ticker(self, ymd: str, market: str = "ALL") -> Optional[pd.DataFrame]:
        if self._pykrx_ok:
            return _pykrx_df(self._pykrx.get_market_ohlcv, ymd, market=market)
        return None

    # ── 시가총액 ──
    def get_market_cap(self, ymd: str, market: str = "ALL") -> Optional[pd.DataFrame]:
        if self._pykrx_ok:
            return _pykrx_df(self._pykrx.get_market_cap, ymd, market=market)
        return None

    # ── 종목 리스트 ──
    def get_ticker_list(self, ymd: str, market: str) -> List[str]:
        if self._pykrx_ok:
            try:
                return self._pykrx.get_market_ticker_list(ymd, market=market)
            except Exception:
                return []
        return []

    # ── 종목명 ──
    def get_ticker_name(self, ticker: str) -> Optional[str]:
        if self._pykrx_ok:
            try:
                name = self._pykrx.get_market_ticker_name(ticker)
                return name if name else None
            except Exception:
                pass
        if self._fdr_ok:
            try:
                listing = self._fdr.StockListing("KRX")
                row = listing[listing["Code"] == ticker]
                if not row.empty:
                    return str(row.iloc[0].get("Name", ""))
            except Exception:
                pass
        return None


# ═══════════════════════════════════════════════════
#  캐시 레이어 — Parquet(신규) / pickle(레거시)
# ═══════════════════════════════════════════════════

class OHLCVCache:
    """
    OHLCV 캐시 관리.
    - Parquet: 보안(RCE 없음) + 스키마 보존 + lazy load 가능
    - CSV: pyarrow 미설치 환경 fallback (안전, 느림)
    - pickle: 기존 호환용 (읽기만)
    """

    def __init__(self, out_dir: str, fmt: str = "parquet", allow_legacy_pickle: bool = False):
        self.out_dir = out_dir
        self.fmt = fmt  # "parquet" or "pickle"
        self.allow_legacy_pickle = allow_legacy_pickle
        self._has_parquet = self._check_parquet()

    @staticmethod
    def _check_parquet() -> bool:
        try:
            import pyarrow  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            pass
        return False

    def _safe_dir(self, ymd: str) -> str:
        return os.path.join(self.out_dir, f"ohlcv_cache_{ymd}")

    def _pickle_path(self, ymd: str) -> str:
        return os.path.join(self.out_dir, f"ohlcv_cache_{ymd}.pkl")

    def _ext(self) -> str:
        return ".parquet" if (self.fmt == "parquet" and self._has_parquet) else ".csv"

    def load(self, ymd: str) -> Dict[str, pd.DataFrame]:
        """캐시 로드. directory(parquet/csv) → pickle fallback."""
        cache_dir = self._safe_dir(ymd)
        if os.path.isdir(cache_dir):
            data = {}
            for f in os.listdir(cache_dir):
                code = None
                if f.endswith(".parquet"):
                    code = f.replace(".parquet", "")
                    try:
                        data[code] = pd.read_parquet(os.path.join(cache_dir, f))
                    except Exception as e:
                        logger.warning(f"Parquet 로드 실패 {code}: {e}")
                elif f.endswith(".csv"):
                    code = f.replace(".csv", "")
                    try:
                        data[code] = pd.read_csv(os.path.join(cache_dir, f))
                    except Exception as e:
                        logger.warning(f"CSV 로드 실패 {code}: {e}")
            if data:
                logger.info(f"📂 OHLCV 캐시 로드: {len(data)}개 종목 ({cache_dir})")
                return data

        # pickle fallback (읽기만, allow_legacy_pickle=True 필수)
        pkl_path = self._pickle_path(ymd)
        if os.path.exists(pkl_path):
            if not self.allow_legacy_pickle:
                logger.warning(
                    f"⚠️ pickle 캐시 발견({pkl_path}) but allow_legacy_pickle=False. "
                    "보안상 스킵. Parquet/CSV 재생성 필요."
                )
                return {}
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"📂 OHLCV pickle 캐시 로드(레거시): {len(data)}개 종목")
                return data
            except Exception as e:
                logger.warning(f"pickle 캐시 로드 실패: {e}")

        return {}

    def save(self, ymd: str, data: Dict[str, pd.DataFrame]) -> None:
        """캐시 저장. Parquet 우선, 불가 시 CSV fallback."""
        if not data:
            return

        cache_dir = self._safe_dir(ymd)
        os.makedirs(cache_dir, exist_ok=True)
        ext = self._ext()
        saved = 0

        for code, df in data.items():
            fpath = os.path.join(cache_dir, f"{code}{ext}")
            try:
                if ext == ".parquet":
                    df.to_parquet(fpath)
                else:
                    df.to_csv(fpath, index=True)
                saved += 1
            except Exception as e:
                logger.warning(f"캐시 저장 실패 {code}: {e}")

        logger.info(f"💾 OHLCV 캐시 저장({ext}): {saved}개 종목")

    def exists(self, ymd: str) -> bool:
        """캐시 존재 여부"""
        return os.path.isdir(self._safe_dir(ymd)) or os.path.exists(self._pickle_path(ymd))


# ── 싱글턴 인스턴스 ──
_data_source = None

def get_data_source() -> KRXDataSource:
    global _data_source
    if _data_source is None:
        _data_source = KRXDataSource()
    return _data_source
