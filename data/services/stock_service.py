# data/services/stock_service.py — 종목 데이터 & 지표 계산
"""
UI 의존성 0. FDR 호출 + pandas 기술적 지표 계산.

사용법:
    from data.services.stock_service import StockService
    svc = StockService()
    chart_df = svc.get_chart_data("005930")
"""
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger("stock_service")

# Optional imports (graceful fallback)
_FDR_OK = False
try:
    import FinanceDataReader as fdr
    _FDR_OK = True
except ImportError:
    fdr = None

try:
    from indicators import calculate_supertrend
except ImportError:
    def calculate_supertrend(df):
        return df

try:
    from shared_utils import calc_hma as calc_hma_series
except ImportError:
    calc_hma_series = None


class StockService:
    """종목 시세 조회 + 기술적 지표 계산"""

    # ── KRX 전체 종목 캐시 (앱 레벨 싱글톤) ──
    _krx_map: Dict[str, str] = {}

    def ensure_krx_map(self):
        """FDR 기반 전체 종목 목록 로드 (최초 1회)"""
        if self._krx_map or not _FDR_OK:
            return
        try:
            listing = fdr.StockListing("KRX")
            if listing is not None and not listing.empty:
                code_col = "Code" if "Code" in listing.columns else "Symbol"
                name_col = "Name" if "Name" in listing.columns else "종목명"
                if code_col in listing.columns and name_col in listing.columns:
                    StockService._krx_map = dict(
                        zip(listing[name_col], listing[code_col].astype(str).str.zfill(6))
                    )
                    logger.info(f"✅ KRX 종목 캐시: {len(self._krx_map)}개")
        except Exception as e:
            logger.warning(f"KRX 목록 로드 실패: {e}")

    def find_code(self, name: str, scored_map: Dict[str, str] = None) -> str:
        """종목명 → 코드 (scored → KRX 캐시 순)"""
        if scored_map:
            if name in scored_map:
                return scored_map[name]
            for k, v in scored_map.items():
                if name in k or k in name:
                    return v
        self.ensure_krx_map()
        if name in self._krx_map:
            return self._krx_map[name]
        for k, v in self._krx_map.items():
            if name in k or k in name:
                return v
        return name

    # ── 시세 조회 ──

    def fetch_current_price(self, code: str, name: str = "") -> Tuple[str, str, int]:
        """현재가 조회 → (code, name, price)"""
        code_str = str(code).zfill(6) if str(code).isdigit() else ""
        if _FDR_OK and code_str:
            try:
                start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                d = fdr.DataReader(code_str, start)
                if d is not None and not d.empty:
                    return code, name, int(d.iloc[-1]["Close"])
            except Exception:
                pass
        if _FDR_OK and not code_str:
            self.ensure_krx_map()
            found = self._krx_map.get(name)
            if not found:
                for k, v in self._krx_map.items():
                    if name in k or k in name:
                        found = v
                        break
            if found:
                try:
                    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                    d = fdr.DataReader(found, start)
                    if d is not None and not d.empty:
                        return found, name, int(d.iloc[-1]["Close"])
                except Exception:
                    pass
        return code, name, 0

    def fetch_prices_parallel(
        self, targets: List[Tuple[str, str]], max_workers: int = 8
    ) -> Dict[str, int]:
        """다수 종목 현재가 병렬 조회 → {code: price}"""
        result = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.fetch_current_price, t[0], t[1]) for t in targets]
            for f in futs:
                c, n, p = f.result()
                result[c] = p
        return result

    # ── 차트 데이터 + 지표 ──

    def get_chart_data(self, code: str, days: int = 400) -> Optional[pd.DataFrame]:
        """캔들차트 데이터 + 기술적 지표 (BB, KC, RSI, OBV, SuperTrend)"""
        if not _FDR_OK:
            return None
        try:
            code_str = str(code).zfill(6)
            start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            df = fdr.DataReader(code_str, start)
            if df is None or df.empty:
                return None

            df = self._add_moving_averages(df)
            df = self._add_bollinger(df)
            df = self._add_keltner(df)
            df = self._add_rsi(df)
            df = self._add_obv(df)
            df = self._add_hma(df)
            df = calculate_supertrend(df)
            df = self._add_weekly_ma(df)

            return df.tail(120)
        except Exception:
            logger.exception(f"get_chart_data failed: {code}")
            return None

    # ── 지표 계산 (private) ──

    @staticmethod
    def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        return df

    @staticmethod
    def _add_bollinger(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
        std = df["Close"].rolling(window).std()
        df["BB_UPPER"] = df["MA20"] + n_std * std
        df["BB_LOWER"] = df["MA20"] - n_std * std
        return df

    @staticmethod
    def _add_keltner(df: pd.DataFrame, mult: float = 1.5) -> pd.DataFrame:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr20 = tr.rolling(20).mean()
        df["KC_UPPER"] = df["MA20"] + mult * atr20
        df["KC_LOWER"] = df["MA20"] - mult * atr20
        return df

    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        df["RSI14_CHART"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def _add_obv(df: pd.DataFrame) -> pd.DataFrame:
        change = np.sign(df["Close"].diff()).fillna(0)
        df["OBV"] = (change * df["Volume"]).cumsum()
        return df

    @staticmethod
    def _add_hma(df: pd.DataFrame) -> pd.DataFrame:
        if calc_hma_series is not None:
            try:
                df["HMA20"] = calc_hma_series(df["Close"], 20)
            except Exception:
                pass
        return df

    @staticmethod
    def _add_weekly_ma(df: pd.DataFrame) -> pd.DataFrame:
        try:
            logic_w = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            df_w = df.resample("W").apply(logic_w)
            df_w["WMA20"] = df_w["Close"].rolling(20).mean()
            df["WEEKLY_MA20"] = df.index.map(
                lambda x: df_w.loc[df_w.index <= x, "WMA20"].iloc[-1]
                if not df_w.loc[df_w.index <= x, "WMA20"].empty
                else np.nan
            )
        except Exception:
            pass
        return df
