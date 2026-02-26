# -*- coding: utf-8 -*-
"""
data_source.py — 데이터소스 추상화 + Parquet 캐시
═══════════════════════════════════════════════════
[v5.0] 4건 리팩터링:
  #1 Parquet 좁쌀 파일 → 단일 통합 파일 (I/O 2500회 → 1회)
  #2 공휴일 빈 DF 재시도 지옥 → 예외 시에만 재시도
  #3 날짜 하드코딩 슬라이싱 → _safe_ymd() 방어
  #4 get_ohlcv_by_ticker/get_market_cap FDR fallback 추가
"""
import os
import re
import time
import pickle
import logging
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  유틸
# ═══════════════════════════════════════════════════

def _backoff_sleep(attempt: int, base: float = 0.35, cap: float = 2.0) -> None:
    time.sleep(min(base * (2 ** attempt), cap))


def _safe_ymd(ymd: str) -> str:
    """[v5.0 #3] 어떤 날짜 형식이든 YYYYMMDD로 정규화

    "20251031", "2025-10-31", "2025.10.31" → "20251031"
    """
    clean = re.sub(r"[^0-9]", "", str(ymd).strip())
    return clean[:8]


def _safe_ymd_dash(ymd: str) -> str:
    """YYYYMMDD → YYYY-MM-DD (FDR용)"""
    c = _safe_ymd(ymd)
    return f"{c[:4]}-{c[4:6]}-{c[6:8]}" if len(c) >= 8 else ymd


def _pykrx_df(fn, *args, max_retries: int = 3, **kwargs) -> Optional[pd.DataFrame]:
    """[v5.0 #2] pykrx 함수 호출 — 예외 시에만 재시도, 빈 DF는 정상 반환

    공휴일/주말에 빈 DataFrame이 오는 것은 정상 응답.
    네트워크 오류(Exception)만 재시도 대상.
    """
    for i in range(max_retries):
        try:
            df = fn(*args, **kwargs)
            return df  # 빈 DF라도 예외가 아니면 정상 반환
        except Exception as e:
            logger.debug(f"pykrx API 에러 재시도 {i+1}/{max_retries}: {e}")
            _backoff_sleep(i)
    return None


# ═══════════════════════════════════════════════════
#  KRXDataSource — 통합 데이터소스
# ═══════════════════════════════════════════════════

class KRXDataSource:
    """
    pykrx/FDR 통합 추상화.
    - provider 우선순위: pykrx → FDR → empty fallback
    - [v5.0 #4] get_ohlcv_by_ticker, get_market_cap에도 FDR fallback 추가
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

        # FDR StockListing 캐시 (get_ohlcv_by_ticker/get_market_cap fallback용)
        self._fdr_listing_cache: Optional[pd.DataFrame] = None

    def _get_fdr_listing(self) -> Optional[pd.DataFrame]:
        """FDR KRX 전종목 데이터 캐시 (시총, 종가 포함)"""
        if self._fdr_listing_cache is not None:
            return self._fdr_listing_cache
        if not self._fdr_ok:
            return None
        try:
            listing = self._fdr.StockListing("KRX")
            if listing is not None and not listing.empty:
                self._fdr_listing_cache = listing
                return listing
        except Exception as e:
            logger.debug(f"FDR StockListing 실패: {e}")
        return None

    # ── OHLCV (종목별, 기간) ──
    def get_ohlcv(self, code: str, start_ymd: str, end_ymd: str) -> Optional[pd.DataFrame]:
        """종목 OHLCV 조회. pykrx → FDR fallback."""
        s_clean = _safe_ymd(start_ymd)
        e_clean = _safe_ymd(end_ymd)

        # pykrx
        if self._pykrx_ok:
            df = _pykrx_df(self._pykrx.get_market_ohlcv, s_clean, e_clean, code)
            if df is not None and not df.empty:
                return df

        # [v5.0 #3] FDR fallback — 안전한 날짜 변환
        if self._fdr_ok:
            try:
                s_dash = _safe_ymd_dash(start_ymd)
                e_dash = _safe_ymd_dash(end_ymd)
                df = self._fdr.DataReader(code, s_dash, e_dash)
                if df is not None and not df.empty:
                    return df
            except Exception as ex:
                logger.debug(f"FDR fallback failed for {code}: {ex}")

        return None

    # ── 일별 전종목 OHLCV ──
    def get_ohlcv_by_ticker(self, ymd: str, market: str = "ALL") -> Optional[pd.DataFrame]:
        """[v5.0 #4] pykrx → FDR StockListing fallback"""
        ymd_clean = _safe_ymd(ymd)

        if self._pykrx_ok:
            df = _pykrx_df(self._pykrx.get_market_ohlcv, ymd_clean, market=market)
            if df is not None and not df.empty:
                return df

        # FDR fallback — StockListing에서 종가/거래량 추출
        if self._fdr_ok:
            try:
                listing = self._get_fdr_listing()
                if listing is not None and not listing.empty:
                    col_map = {}
                    for src, dst in [("Close", "종가"), ("Open", "시가"), ("High", "고가"),
                                     ("Low", "저가"), ("Volume", "거래량"), ("Marcap", "시가총액")]:
                        if src in listing.columns:
                            col_map[src] = dst
                    if col_map:
                        code_col = "Code" if "Code" in listing.columns else "Symbol"
                        df = listing[[code_col] + list(col_map.keys())].copy()
                        df = df.rename(columns=col_map)
                        df = df.set_index(code_col)
                        df.index.name = None
                        logger.info(f"📊 FDR fallback: get_ohlcv_by_ticker ({len(df)}종목)")
                        return df
            except Exception as e:
                logger.debug(f"FDR ohlcv_by_ticker fallback 실패: {e}")

        return None

    # ── 시가총액 ──
    def get_market_cap(self, ymd: str, market: str = "ALL") -> Optional[pd.DataFrame]:
        """[v5.0 #4] pykrx → FDR StockListing fallback"""
        ymd_clean = _safe_ymd(ymd)

        if self._pykrx_ok:
            df = _pykrx_df(self._pykrx.get_market_cap, ymd_clean, market=market)
            if df is not None and not df.empty:
                return df

        # FDR fallback
        if self._fdr_ok:
            try:
                listing = self._get_fdr_listing()
                if listing is not None and "Marcap" in listing.columns:
                    code_col = "Code" if "Code" in listing.columns else "Symbol"
                    cols = [code_col, "Marcap"]
                    if "Stocks" in listing.columns:
                        cols.append("Stocks")
                    df = listing[cols].copy()
                    df = df.rename(columns={"Marcap": "시가총액"})
                    if "Stocks" in df.columns:
                        df = df.rename(columns={"Stocks": "상장주식수"})
                    df = df.set_index(code_col)
                    df.index.name = None
                    logger.info(f"📊 FDR fallback: get_market_cap ({len(df)}종목)")
                    return df
            except Exception as e:
                logger.debug(f"FDR market_cap fallback 실패: {e}")

        return None

    # ── 종목 리스트 ──
    def get_ticker_list(self, ymd: str, market: str) -> List[str]:
        ymd_clean = _safe_ymd(ymd)
        if self._pykrx_ok:
            try:
                return self._pykrx.get_market_ticker_list(ymd_clean, market=market)
            except Exception:
                pass
        # FDR fallback
        if self._fdr_ok:
            try:
                listing = self._get_fdr_listing()
                if listing is not None:
                    code_col = "Code" if "Code" in listing.columns else "Symbol"
                    return listing[code_col].astype(str).str.zfill(6).tolist()
            except Exception:
                pass
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
                listing = self._get_fdr_listing()
                if listing is not None:
                    code_col = "Code" if "Code" in listing.columns else "Symbol"
                    row = listing[listing[code_col] == ticker]
                    if not row.empty:
                        return str(row.iloc[0].get("Name", ""))
            except Exception:
                pass
        return None


# ═══════════════════════════════════════════════════
#  캐시 레이어 — 통합 Parquet(신규) / pickle(레거시)
# ═══════════════════════════════════════════════════

class OHLCVCache:
    """[v5.0 #1] OHLCV 캐시 — 단일 통합 파일

    Before: 종목당 1파일 × 2,500개 = 2,500회 File I/O
    After:  1개 통합 파일 = 1회 File I/O (수십 배 빠름)

    레거시 호환: 기존 폴더 캐시(개별 파일), pickle 캐시 읽기 지원
    """

    def __init__(self, out_dir: str, fmt: str = "parquet", allow_legacy_pickle: bool = False):
        self.out_dir = out_dir
        self.fmt = fmt
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

    def _ext(self) -> str:
        return ".parquet" if (self.fmt == "parquet" and self._has_parquet) else ".csv"

    def _unified_path(self, ymd: str) -> str:
        """통합 캐시 파일 경로"""
        return os.path.join(self.out_dir, f"ohlcv_cache_{ymd}{self._ext()}")

    def _legacy_dir(self, ymd: str) -> str:
        """레거시 폴더 캐시 경로"""
        return os.path.join(self.out_dir, f"ohlcv_cache_{ymd}")

    def _pickle_path(self, ymd: str) -> str:
        return os.path.join(self.out_dir, f"ohlcv_cache_{ymd}.pkl")

    def load(self, ymd: str) -> Dict[str, pd.DataFrame]:
        """캐시 로드 우선순위: 통합 파일 → 레거시 폴더 → pickle fallback"""

        # 1) 통합 파일 (신규 v5.0 포맷)
        unified = self._unified_path(ymd)
        if os.path.exists(unified):
            return self._load_unified(unified)

        # 2) 레거시 폴더 캐시 (개별 파일 — v4 이하)
        legacy_dir = self._legacy_dir(ymd)
        if os.path.isdir(legacy_dir):
            data = self._load_legacy_dir(legacy_dir)
            if data:
                return data

        # 3) pickle fallback (읽기만)
        return self._load_legacy_pickle(ymd)

    def _load_unified(self, path: str) -> Dict[str, pd.DataFrame]:
        """통합 파일에서 로드 → {code: df} dict (날짜 인덱스 보존)"""
        data = {}
        try:
            if path.endswith(".parquet"):
                combined = pd.read_parquet(path)
            else:
                # [v5.1 Fix] CSV: 날짜 인덱스(0번째 컬럼) 복원 + 종목코드 문자열 강제
                combined = pd.read_csv(path, index_col=0, parse_dates=True, dtype={"종목코드": str})

            if "종목코드" in combined.columns:
                for code, group in combined.groupby("종목코드"):
                    # [v5.1 Fix] 앞자리 0 증발 방어 + 종목코드 컬럼만 제거 (날짜 인덱스 유지!)
                    clean_code = str(code).zfill(6)
                    data[clean_code] = group.drop(columns=["종목코드"])
                logger.info(f"📂 OHLCV 통합 캐시 로드: {len(data)}개 종목")
        except Exception as e:
            logger.error(f"통합 캐시 로드 실패: {e}")
        return data

    def _load_legacy_dir(self, cache_dir: str) -> Dict[str, pd.DataFrame]:
        """레거시 폴더 캐시 (종목당 개별 파일)"""
        data = {}
        for f in os.listdir(cache_dir):
            code = None
            try:
                if f.endswith(".parquet"):
                    code = f.replace(".parquet", "")
                    data[code] = pd.read_parquet(os.path.join(cache_dir, f))
                elif f.endswith(".csv"):
                    code = f.replace(".csv", "")
                    data[code] = pd.read_csv(os.path.join(cache_dir, f))
            except Exception as e:
                logger.warning(f"레거시 캐시 로드 실패 {code}: {e}")
        if data:
            logger.info(f"📂 OHLCV 레거시 폴더 캐시 로드: {len(data)}개 종목")
        return data

    def _load_legacy_pickle(self, ymd: str) -> Dict[str, pd.DataFrame]:
        """pickle fallback (읽기만, 보안 옵션 체크)"""
        pkl_path = self._pickle_path(ymd)
        if not os.path.exists(pkl_path):
            return {}
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
        """[v5.0 #1] 단일 통합 파일로 저장 (I/O 1회)"""
        if not data:
            return

        os.makedirs(self.out_dir, exist_ok=True)
        cache_file = self._unified_path(ymd)

        try:
            # 전 종목 DataFrame을 하나로 병합
            combined = pd.concat(
                data.values(),
                keys=data.keys(),
                names=["종목코드", "Date"],  # 2번째 레벨 = 날짜 인덱스
            )
            # [v5.1 Fix] 종목코드만 컬럼으로, 날짜(Date) 인덱스는 그대로 유지!
            combined = combined.reset_index(level="종목코드")

            if self._ext() == ".parquet":
                combined.to_parquet(cache_file)
            else:
                # [v5.1 Fix] index=True → 날짜 인덱스를 CSV에 저장
                combined.to_csv(cache_file, index=True)

            logger.info(f"💾 OHLCV 통합 캐시 저장({self._ext()}): {len(data)}개 종목")
        except Exception as e:
            logger.error(f"통합 캐시 저장 실패: {e}")

    def exists(self, ymd: str) -> bool:
        """캐시 존재 여부 (통합 파일 or 레거시)"""
        return (
            os.path.exists(self._unified_path(ymd))
            or os.path.isdir(self._legacy_dir(ymd))
            or os.path.exists(self._pickle_path(ymd))
        )


# ── 싱글턴 인스턴스 ──
_data_source = None

def get_data_source() -> KRXDataSource:
    global _data_source
    if _data_source is None:
        _data_source = KRXDataSource()
    return _data_source
