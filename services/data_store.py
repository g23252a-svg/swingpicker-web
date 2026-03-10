# -*- coding: utf-8 -*-
"""
data_store.py — 전역 주식 데이터 상태 관리
═══════════════════════════════════════════════════
DataStore (Thread-Safe 싱글턴) + KRX 종목 캐시 + 종목명 4단계 복구
"""
import io
import os
import logging
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

try:
    import FinanceDataReader as fdr
    FDR_OK = True
except ImportError:
    FDR_OK = False

_logger = logging.getLogger("ldy-nicegui")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RECOMMEND_PATH = os.path.join(DATA_DIR, "recommend_latest.csv")
REMOTE_CSV_URL = os.getenv(
    "LDY_RAW_URL",
    "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
)

KST = timezone(timedelta(hours=9))


def now_kst():
    return datetime.now(KST)


# ══════════════════════════════════════════════════════
#  KRX 전체 종목 캐시
# ══════════════════════════════════════════════════════
_KRX_NAME_MAP = {}


def _ensure_krx_map():
    """전체 종목 목록 로드 (FDR → GitHub CSV → 로컬 파일 순 폴백)"""
    global _KRX_NAME_MAP
    if _KRX_NAME_MAP:
        return

    # ── 방법 1: FDR (Railway 해외 IP에서 실패 가능) ──
    if FDR_OK:
        try:
            listing = fdr.StockListing("KRX")
            if listing is not None and not listing.empty:
                code_col = None
                for c in ["Code", "Symbol", "Ticker", "ISU_SRT_CD", "종목코드"]:
                    if c in listing.columns:
                        code_col = c
                        break
                name_col = None
                for c in ["Name", "종목명", "ISU_ABBRV"]:
                    if c in listing.columns:
                        name_col = c
                        break
                if code_col is None and listing.index.dtype == object:
                    sample_idx = str(listing.index[0]).strip()
                    if sample_idx.isdigit() and len(sample_idx) == 6:
                        listing = listing.reset_index()
                        listing.rename(columns={listing.columns[0]: "_idx_code"}, inplace=True)
                        code_col = "_idx_code"
                if code_col and name_col:
                    _KRX_NAME_MAP = dict(zip(listing[name_col], listing[code_col].astype(str).str.zfill(6)))
                    _logger.info(f"✅ KRX 종목 캐시 [FDR]: {len(_KRX_NAME_MAP)}개")
                    return
                else:
                    _logger.warning(f"⚠️ FDR 컬럼 매칭 실패: cols={listing.columns.tolist()[:10]}")
        except Exception as e:
            _logger.warning(f"⚠️ FDR 로드 실패: {e}")

    # ── 방법 2: GitHub에서 krx_names_latest.csv 다운로드 ──
    try:
        _base = REMOTE_CSV_URL.rsplit("/", 1)[0]
        _names_url = f"{_base}/krx_names_latest.csv"
        resp = requests.get(_names_url, timeout=10)
        if resp.ok and resp.text.strip():
            _df = pd.read_csv(io.StringIO(resp.text), dtype=str)
            if "종목코드" in _df.columns and "종목명" in _df.columns:
                _map = {}
                for _, row in _df.iterrows():
                    c = str(row["종목코드"]).strip().zfill(6)
                    n = str(row["종목명"]).strip()
                    if c != n and n:
                        _map[n] = c
                if _map:
                    _KRX_NAME_MAP = _map
                    _logger.info(f"✅ KRX 종목 캐시 [GitHub]: {len(_KRX_NAME_MAP)}개")
                    return
    except Exception as e:
        _logger.warning(f"⚠️ GitHub 종목명 다운로드 실패: {e}")

    # ── 방법 3: 로컬 파일 폴백 ──
    for _path in [os.path.join(DATA_DIR, "krx_names_latest.csv"),
                  "data/krx_names_latest.csv", "/app/data/krx_names_latest.csv"]:
        try:
            if os.path.exists(_path):
                _df = pd.read_csv(_path, dtype=str)
                if "종목코드" in _df.columns and "종목명" in _df.columns:
                    _map = {str(row["종목명"]).strip(): str(row["종목코드"]).strip().zfill(6)
                            for _, row in _df.iterrows()
                            if str(row["종목명"]).strip() != str(row["종목코드"]).strip()}
                    if _map:
                        _KRX_NAME_MAP = _map
                        _logger.info(f"✅ KRX 종목 캐시 [로컬]: {len(_KRX_NAME_MAP)}개")
                        return
        except Exception:
            pass

    _logger.warning("⚠️ KRX 종목 매핑 로드 완전 실패 — 종목명이 코드로 표시될 수 있음")


# ══════════════════════════════════════════════════════
#  DataStore (Thread-Safe 싱글턴)
# ══════════════════════════════════════════════════════
class DataStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._scored = pd.DataFrame()
        self.data_ts = ""
        self.loaded = False

    @property
    def scored(self):
        """읽기 시 항상 스냅샷 복사본 반환 — 쓰기 중 참조 꼬임 방지"""
        with self._lock:
            return self._scored.copy()

    @scored.setter
    def scored(self, value):
        with self._lock:
            self._scored = value

    def refresh(self):
        df = None

        # 1) 로컬 파일 시도
        if os.path.exists(RECOMMEND_PATH):
            try:
                df = pd.read_csv(RECOMMEND_PATH, dtype={"종목코드": str, "종목명": str})
                _logger.info(f"📂 로컬 CSV 로드: {RECOMMEND_PATH}")
            except Exception as e:
                _logger.warning(f"로컬 CSV 읽기 실패: {e}")

        # 2) 로컬 실패 → GitHub raw URL 폴백
        if df is None or df.empty:
            url = REMOTE_CSV_URL.strip()
            if url:
                try:
                    _logger.info(f"🌐 원격 CSV 다운로드 시도: {url}")
                    r = requests.get(url, timeout=30,
                                     headers={"Cache-Control": "no-cache"})
                    r.raise_for_status()
                    df = pd.read_csv(io.BytesIO(r.content),
                                     encoding="utf-8-sig",
                                     dtype={"종목코드": str, "종목명": str})
                    os.makedirs(DATA_DIR, exist_ok=True)
                    with open(RECOMMEND_PATH, "wb") as f:
                        f.write(r.content)
                    _logger.info(f"✅ 원격 CSV 다운로드 성공 → 로컬 캐싱 완료 ({len(df)}건)")
                except Exception as e:
                    _logger.warning(f"원격 CSV 다운로드 실패: {e}")

        if df is None or df.empty:
            _logger.warning("❌ 로컬/원격 모두 데이터 로드 실패")
            return

        try:
            num_cols = [
                "FINAL_SCORE", "DISPLAY_SCORE", "STRUCT_SCORE",
                "TIMING_SCORE", "AI_SCORE", "ML_SCORE", "TOTAL_SCORE",
                "RANK_SCORE", "EBS", "RR1", "RSI14",
                "거래대금(억원)", "종가", "추천매수가", "손절가",
                "추천매도가1", "추천매도가2", "TARGET_ATR",
            ]
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            # ─── 종목명 오염 자동 복구 (4단계) ───
            if "종목코드" in df.columns and "종목명" in df.columns:
                mask = df["종목명"].astype(str).str.match(r'^\d+$')
                if mask.any():
                    _fixed = False
                    _bad_count = mask.sum()

                    # 1순위: krx_names_latest.csv
                    _names_paths = [
                        os.path.join(DATA_DIR, "krx_names_latest.csv"),
                        "data/krx_names_latest.csv",
                        "/app/data/krx_names_latest.csv",
                    ]
                    for _np in _names_paths:
                        if _fixed:
                            break
                        try:
                            if os.path.exists(_np):
                                _ndf = pd.read_csv(_np, dtype=str)
                                if "종목코드" in _ndf.columns and "종목명" in _ndf.columns:
                                    _c2n = dict(zip(
                                        _ndf["종목코드"].astype(str).str.zfill(6),
                                        _ndf["종목명"]
                                    ))
                                    _c2n = {c: n for c, n in _c2n.items() if c != n and n and not n.isdigit()}
                                    if _c2n:
                                        df.loc[mask, "종목명"] = (
                                            df.loc[mask, "종목코드"].astype(str).str.zfill(6)
                                            .map(_c2n)
                                            .fillna(df.loc[mask, "종목명"])
                                        )
                                        _still_bad = df["종목명"].astype(str).str.match(r'^\d+$').sum()
                                        if _still_bad < _bad_count:
                                            _logger.info(f"🔧 종목명 오염 {_bad_count - _still_bad}/{_bad_count}건 복구 [krx_names: {_np}]")
                                            _fixed = (_still_bad == 0)
                                            mask = df["종목명"].astype(str).str.match(r'^\d+$')
                        except Exception as _e:
                            _logger.debug(f"krx_names 로드 실패 ({_np}): {_e}")

                    # 2순위: GitHub raw
                    if not _fixed and mask.any():
                        try:
                            _base = REMOTE_CSV_URL.rsplit("/", 1)[0]
                            _names_url = f"{_base}/krx_names_latest.csv"
                            _resp = requests.get(_names_url, timeout=10)
                            if _resp.ok and _resp.text.strip():
                                _ndf = pd.read_csv(io.StringIO(_resp.text), dtype=str)
                                if "종목코드" in _ndf.columns and "종목명" in _ndf.columns:
                                    _c2n = dict(zip(
                                        _ndf["종목코드"].astype(str).str.zfill(6),
                                        _ndf["종목명"]
                                    ))
                                    _c2n = {c: n for c, n in _c2n.items() if c != n and n and not n.isdigit()}
                                    if _c2n:
                                        df.loc[mask, "종목명"] = (
                                            df.loc[mask, "종목코드"].astype(str).str.zfill(6)
                                            .map(_c2n)
                                            .fillna(df.loc[mask, "종목명"])
                                        )
                                        _still_bad = df["종목명"].astype(str).str.match(r'^\d+$').sum()
                                        _logger.info(f"🔧 종목명 오염 {_bad_count - _still_bad}/{_bad_count}건 복구 [GitHub krx_names]")
                                        _fixed = (_still_bad == 0)
                                        mask = df["종목명"].astype(str).str.match(r'^\d+$')
                        except Exception as _e:
                            _logger.debug(f"GitHub krx_names 다운로드 실패: {_e}")

                    # 3순위: _ensure_krx_map (FDR 전체 목록)
                    if not _fixed and mask.any():
                        _ensure_krx_map()
                        if _KRX_NAME_MAP:
                            _code_to_name = {v: k for k, v in _KRX_NAME_MAP.items()}
                            df.loc[mask, "종목명"] = (
                                df.loc[mask, "종목코드"].astype(str).str.zfill(6)
                                .map(_code_to_name)
                                .fillna(df.loc[mask, "종목명"])
                            )
                            _still_bad = df["종목명"].astype(str).str.match(r'^\d+$').sum()
                            if _still_bad < _bad_count:
                                _logger.info(f"🔧 종목명 오염 {_bad_count - _still_bad}/{_bad_count}건 복구 [KRX캐시]")
                                _fixed = (_still_bad == 0)
                                mask = df["종목명"].astype(str).str.match(r'^\d+$')

                    # 4순위 (최후 수단): Naver API 병렬 조회 (ThreadPool)
                    if not _fixed and mask.any():
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        _codes = df.loc[mask, "종목코드"].astype(str).str.zfill(6).unique()
                        _logger.info(f"🔄 Naver API로 종목명 {len(_codes)}건 병렬 조회 시도...")
                        _code_to_name = {}

                        def _fetch_name(code):
                            try:
                                r = requests.get(
                                    f"https://m.stock.naver.com/api/stock/{code}/basic",
                                    timeout=5,
                                    headers={"User-Agent": "Mozilla/5.0"}
                                )
                                if r.ok:
                                    name = r.json().get("stockName", "")
                                    if name and name != code:
                                        return code, name
                            except Exception:
                                pass
                            return code, None

                        with ThreadPoolExecutor(max_workers=20) as pool:
                            futures = {pool.submit(_fetch_name, c): c for c in _codes}
                            for fut in as_completed(futures):
                                code, name = fut.result()
                                if name:
                                    _code_to_name[code] = name

                        if _code_to_name:
                            for code, name in _code_to_name.items():
                                df.loc[(mask) & (df["종목코드"].astype(str).str.zfill(6) == code), "종목명"] = name
                            _logger.info(f"🔧 종목명 오염 {len(_code_to_name)}/{len(_codes)}건 복구 [Naver 병렬]")

                    _final_bad = df["종목명"].astype(str).str.match(r'^\d+$').sum()
                    if _final_bad > 0:
                        _logger.warning(f"⚠️ 종목명 복구 불완전: {_final_bad}건 여전히 코드 상태")

            primary = next((c for c in ["DISPLAY_SCORE", "FINAL_SCORE", "TOTAL_SCORE"]
                           if c in df.columns and df[c].abs().sum() > 0), None)
            if primary:
                for alias in ["DISPLAY_SCORE", "TOTAL_SCORE", "LDY_SCORE", "RANK_SCORE"]:
                    df[alias] = df[primary]

            ts_col = next((c for c in ["trade_date", "DATA_DATE"] if c in df.columns), None)
            self.data_ts = str(df[ts_col].iloc[0]) if ts_col else now_kst().strftime("%Y-%m-%d")
            self.scored = df
            self.loaded = True
            _logger.info(f"✅ 데이터 로드: {len(df)}종목, 기준일 {self.data_ts}")
        except Exception as e:
            _logger.exception(f"데이터 로드 실패: {e}")


# 싱글턴 인스턴스
store = DataStore()
