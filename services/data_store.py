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
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from services.snapshot_integrity import snapshot_date, validate_snapshot

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
        self._refresh_lock = threading.Lock()
        self.last_refresh = {"ok": False, "source": "none", "message": "아직 불러오지 않았습니다"}
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
            self._scored = value.copy()

    def refresh(self, force_remote=False):
        """Serialize refreshes; publish only a fully validated snapshot."""
        with self._refresh_lock:
            return self._refresh(force_remote=force_remote)

    def _refresh(self, force_remote=False):
        candidates = []
        errors = []
        remote_failed = False
        if os.path.exists(RECOMMEND_PATH):
            try:
                local = validate_snapshot(pd.read_csv(
                    RECOMMEND_PATH, dtype={"종목코드": str, "종목명": str}))
                candidates.append((local, "local"))
            except Exception as exc:
                errors.append(f"로컬 데이터 검증 실패: {exc}")

        # A manual refresh really checks the remote source, even with a local cache.
        if force_remote or not candidates:
            try:
                if not REMOTE_CSV_URL.strip():
                    raise ValueError("원격 데이터 주소가 없습니다")
                response = requests.get(REMOTE_CSV_URL.strip(), timeout=30,
                                        headers={"Cache-Control": "no-cache"})
                response.raise_for_status()
                remote = validate_snapshot(pd.read_csv(
                    io.BytesIO(response.content), encoding="utf-8-sig",
                    dtype={"종목코드": str, "종목명": str}))
                candidates.append((remote, "remote"))
            except Exception as exc:
                remote_failed = True
                errors.append(f"원격 데이터 확인 실패: {exc}")

        with self._lock:
            if self.loaded:
                candidates.append((self._scored.copy(), "memory"))
        if not candidates:
            result = {"ok": False, "source": "none",
                      "message": "데이터를 불러오지 못했습니다", "errors": errors}
            with self._lock:
                self.last_refresh = result
            return result

        # Never roll back to an older (or undated) remote snapshot.
        priority = {"memory": 0, "local": 1, "remote": 2}
        df, source = max(candidates, key=lambda item: (
            snapshot_date(item[0]), priority[item[1]]))
        raw_df = df.copy()
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

            quality_failed = False
            # 배포 직후에도 새 production 계약을 즉시 적용한다. collector가
            # 다음 CSV를 만들기 전까지 기존 recommend_latest를 읽더라도 첫
            # 화면과 알림에서 연구 후보가 공식 매수로 승격되지 않는다.
            try:
                from services.recommendation_quality import apply_recommendation_quality_guard
                df = apply_recommendation_quality_guard(df)
            except Exception as exc:
                quality_failed = True
                errors.append("추천 품질 검증 실패")
                _logger.exception("최종 품질게이트 적용 실패 — 신규매수 안전 차단: %s", exc)
                df["PRODUCTION_BUY"] = 0
                df["BUY_NOW_ELIGIBLE"] = 0
                df["ACTION_DECISION"] = "CASH"
                df["RECOMMENDED_WEIGHT_PCT"] = 0.0
                df["QUALITY_GUARD_REASON"] = "품질게이트 실행 실패"

            data_ts = snapshot_date(df) or "확인 불가"
            if source == "remote":
                # Replace in one operation; a failed write never truncates the cache.
                temp_path = None
                try:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8-sig", newline="",
                        dir=DATA_DIR, suffix=".tmp", delete=False,
                    ) as handle:
                        temp_path = handle.name
                        raw_df.to_csv(handle, index=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temp_path, RECOMMEND_PATH)
                except OSError as exc:
                    errors.append(f"로컬 캐시 저장 실패: {exc}")
                    _logger.warning("캐시 저장 실패: %s", exc)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
            message = ("추천 품질 검증에 실패해 신규매수를 보류합니다"
                       if quality_failed else
                       "최신 데이터 확인에 실패해 기존 데이터를 표시합니다"
                       if remote_failed else
                       "원격 데이터를 확인했습니다" if source == "remote" else
                       "기존의 최신 데이터를 유지합니다")
            result = {"ok": not remote_failed and not quality_failed, "source": source,
                      "message": message, "data_date": data_ts, "errors": errors}
            with self._lock:
                self._scored = df.copy()
                self.data_ts = data_ts
                self.loaded = True
                self.last_refresh = result
            _logger.info("데이터 로드: %s종목, 기준일 %s (%s)", len(df), data_ts, source)
            return result
        except Exception as exc:
            _logger.exception("데이터 처리 실패: %s", exc)
            result = {"ok": False, "source": "memory" if self.loaded else "none",
                      "message": "데이터 처리 실패 — 이전 데이터를 유지합니다",
                      "errors": errors + [str(exc)]}
            with self._lock:
                self.last_refresh = result
            return result


# 싱글턴 인스턴스
store = DataStore()
