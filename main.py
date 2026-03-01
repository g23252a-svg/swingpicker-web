# -*- coding: utf-8 -*-
"""
LDY Pro Trader — NiceGUI Full Edition
═══════════════════════════════════════
Streamlit dashboard.py 4,500줄 → NiceGUI 전체 변환

Tab 1: 📊 시장 현황
Tab 2: 🔭 종목 분석 (테이블 + 칸반 + 캔들차트 + 상세)
Tab 3: 💼 내 자산 (포트폴리오 AI 진단)
Tab 4: 📮 문의 게시판
Tab 5: ⚖️ 이용 약관
Tab 6: 🧩 업데이트 노트
Tab 7: 📈 시스템 성과
Tab 8: 👑 회원 관리 (Admin)
"""

import os
import time
import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import requests
import io

from nicegui import ui, app

# ─── 비동기 래퍼 ───
from async_helpers import run_sync, run_cpu, register_shutdown

# ─── 서비스 & UI ───
from services.auth import (
    get_current_user, set_current_user, get_auth_status,
)
from components.ui_utils import DARK_CSS
from views.login_page import login_page  # noqa: F401 — @ui.page('/login') 등록

# ─── 탭 컴포넌트 ───
from components.tab_terms import render_tab_terms
from components.tab_updates import render_tab_updates
from components.tab_perf import render_tab_perf
from components.tab_inquiry import render_tab_inquiry
from components.tab_admin import render_tab_admin
from components.tab_market import render_tab_market
from components.tab_stocks import render_tab_stocks
from components.tab_portfolio import render_tab_portfolio

# ─── 매매일지 (선택) ───
try:
    from trade_journal_tab import render_trade_journal_tab
    JOURNAL_OK = True
except ImportError:
    JOURNAL_OK = False

# Optional imports
FDR_OK = False
try:
    import FinanceDataReader as fdr
    FDR_OK = True
except ImportError:
    pass

try:
    import version_info
    from version_info import APP_VERSION, CHANGELOG
except Exception:
    APP_VERSION = "12.3.0"
    CHANGELOG = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ldy-nicegui")

# ═══════════════════════════════════════════
#  설정 & 상수
# ═══════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RECOMMEND_PATH = os.path.join(DATA_DIR, "recommend_latest.csv")
REMOTE_CSV_URL = os.getenv(
    "LDY_RAW_URL",
    "https://raw.githubusercontent.com/g23252a-svg/swingpicker-web/main/data/recommend_latest.csv"
)

KST = timezone(timedelta(hours=9))


def now_kst():
    return datetime.now(KST)


# ═══════════════════════════════════════════
#  데이터 저장소
# ═══════════════════════════════════════════
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
            return self._scored.copy()  # ✅ Fix#1: 참조→복사본 (진짜 Thread-Safety)

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
                logger.info(f"📂 로컬 CSV 로드: {RECOMMEND_PATH}")
            except Exception as e:
                logger.warning(f"로컬 CSV 읽기 실패: {e}")

        # 2) 로컬 실패 → GitHub raw URL 폴백
        if df is None or df.empty:
            url = REMOTE_CSV_URL.strip()
            if url:
                try:
                    logger.info(f"🌐 원격 CSV 다운로드 시도: {url}")
                    r = requests.get(url, timeout=30,
                                     headers={"Cache-Control": "no-cache"})
                    r.raise_for_status()
                    df = pd.read_csv(io.BytesIO(r.content),
                                     encoding="utf-8-sig",
                                     dtype={"종목코드": str, "종목명": str})
                    # 로컬에 캐싱 (다음 로드 시 빠르게)
                    os.makedirs(DATA_DIR, exist_ok=True)
                    with open(RECOMMEND_PATH, "wb") as f:
                        f.write(r.content)
                    logger.info(f"✅ 원격 CSV 다운로드 성공 → 로컬 캐싱 완료 ({len(df)}건)")
                except Exception as e:
                    logger.warning(f"원격 CSV 다운로드 실패: {e}")

        if df is None or df.empty:
            logger.warning("❌ 로컬/원격 모두 데이터 로드 실패")
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

            # ─── 종목명 오염 자동 복구 (종목명==종목코드인 경우) ───
            if "종목코드" in df.columns and "종목명" in df.columns:
                mask = df["종목명"].astype(str).str.match(r'^\d+$')
                if mask.any():
                    _fixed = False
                    _bad_count = mask.sum()

                    # ── 1순위: krx_names_latest.csv (collector가 recommend와 함께 항상 저장) ──
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
                                    # 코드==이름인 항목 제거
                                    _c2n = {c: n for c, n in _c2n.items() if c != n and n and not n.isdigit()}
                                    if _c2n:
                                        df.loc[mask, "종목명"] = (
                                            df.loc[mask, "종목코드"].astype(str).str.zfill(6)
                                            .map(_c2n)
                                            .fillna(df.loc[mask, "종목명"])
                                        )
                                        _still_bad = df["종목명"].astype(str).str.match(r'^\d+$').sum()
                                        if _still_bad < _bad_count:
                                            logger.info(f"🔧 종목명 오염 {_bad_count - _still_bad}/{_bad_count}건 복구 [krx_names: {_np}]")
                                            _fixed = (_still_bad == 0)
                                            mask = df["종목명"].astype(str).str.match(r'^\d+$')  # 마스크 갱신
                        except Exception as _e:
                            logger.debug(f"krx_names 로드 실패 ({_np}): {_e}")

                    # ── 2순위: GitHub raw에서 krx_names_latest.csv 다운로드 ──
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
                                        logger.info(f"🔧 종목명 오염 {_bad_count - _still_bad}/{_bad_count}건 복구 [GitHub krx_names]")
                                        _fixed = (_still_bad == 0)
                                        mask = df["종목명"].astype(str).str.match(r'^\d+$')
                        except Exception as _e:
                            logger.debug(f"GitHub krx_names 다운로드 실패: {_e}")

                    # ── 3순위: _ensure_krx_map (FDR 전체 목록) ──
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
                                logger.info(f"🔧 종목명 오염 {_bad_count - _still_bad}/{_bad_count}건 복구 [KRX캐시]")
                                _fixed = (_still_bad == 0)
                                mask = df["종목명"].astype(str).str.match(r'^\d+$')

                    # ── 4순위 (최후 수단): Naver API 개별 조회 ──
                    if not _fixed and mask.any():
                        _naver_fixed = 0
                        _codes = df.loc[mask, "종목코드"].astype(str).str.zfill(6).unique()
                        logger.info(f"🔄 Naver API로 종목명 {len(_codes)}건 개별 조회 시도...")
                        for _c in _codes:
                            try:
                                _r = requests.get(
                                    f"https://m.stock.naver.com/api/stock/{_c}/basic",
                                    timeout=5,
                                    headers={"User-Agent": "Mozilla/5.0"}
                                )
                                if _r.ok:
                                    _name = _r.json().get("stockName", "")
                                    if _name and _name != _c:
                                        df.loc[(mask) & (df["종목코드"].astype(str).str.zfill(6) == _c), "종목명"] = _name
                                        _naver_fixed += 1
                            except Exception:
                                pass
                        if _naver_fixed:
                            logger.info(f"🔧 종목명 오염 {_naver_fixed}/{len(_codes)}건 복구 [Naver]")

                    # 최종 상태 로깅
                    _final_bad = df["종목명"].astype(str).str.match(r'^\d+$').sum()
                    if _final_bad > 0:
                        logger.warning(f"⚠️ 종목명 복구 불완전: {_final_bad}건 여전히 코드 상태")
            # ─── 종목명 복구 끝 ──────────────────────────────────

            primary = next((c for c in ["DISPLAY_SCORE", "FINAL_SCORE", "TOTAL_SCORE"] if c in df.columns and df[c].abs().sum() > 0), None)
            if primary:
                for alias in ["DISPLAY_SCORE", "TOTAL_SCORE", "LDY_SCORE", "RANK_SCORE"]:
                    df[alias] = df[primary]

            ts_col = next((c for c in ["trade_date", "DATA_DATE"] if c in df.columns), None)
            self.data_ts = str(df[ts_col].iloc[0]) if ts_col else now_kst().strftime("%Y-%m-%d")
            self.scored = df
            self.loaded = True
            logger.info(f"✅ 데이터 로드: {len(df)}종목, 기준일 {self.data_ts}")
        except Exception as e:
            logger.exception(f"데이터 로드 실패: {e}")

store = DataStore()


# ═══════════════════════════════════════════
#  KRX 종목 캐시
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
                    logger.info(f"✅ KRX 종목 캐시 [FDR]: {len(_KRX_NAME_MAP)}개")
                    return
                else:
                    logger.warning(f"⚠️ FDR 컬럼 매칭 실패: cols={listing.columns.tolist()[:10]}")
        except Exception as e:
            logger.warning(f"⚠️ FDR 로드 실패: {e}")

    # ── 방법 2: GitHub에서 krx_names_latest.csv 다운로드 ──
    try:
        _base = REMOTE_CSV_URL.rsplit("/", 1)[0]  # .../data
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
                    logger.info(f"✅ KRX 종목 캐시 [GitHub]: {len(_KRX_NAME_MAP)}개")
                    return
    except Exception as e:
        logger.warning(f"⚠️ GitHub 종목명 다운로드 실패: {e}")

    # ── 방법 3: 로컬 파일 폴백 ──
    for _path in ["data/krx_names_latest.csv", "/app/data/krx_names_latest.csv"]:
        try:
            if os.path.exists(_path):
                _df = pd.read_csv(_path, dtype=str)
                if "종목코드" in _df.columns and "종목명" in _df.columns:
                    _map = {str(row["종목명"]).strip(): str(row["종목코드"]).strip().zfill(6)
                            for _, row in _df.iterrows()
                            if str(row["종목명"]).strip() != str(row["종목코드"]).strip()}
                    if _map:
                        _KRX_NAME_MAP = _map
                        logger.info(f"✅ KRX 종목 캐시 [로컬]: {len(_KRX_NAME_MAP)}개")
                        return
        except Exception:
            pass

    logger.warning("⚠️ KRX 종목 매핑 로드 완전 실패 — 종목명이 코드로 표시될 수 있음")


# ═══════════════════════════════════════════
#  메인 페이지 (8개 탭)
# ═══════════════════════════════════════════
@ui.page('/')
async def index():
    ui.add_head_html(DARK_CSS)
    if not store.loaded:
        await run_sync(store.refresh)
    df = store.scored
    auth = get_auth_status()
    user = get_current_user()

    # ─── Hero Banner ───
    with ui.row().classes("w-full items-center justify-between px-4 py-3 rounded-xl mb-2 bg-gradient-to-r from-[#1a1a2e] via-[#16213e] to-[#0f3460]"):
        with ui.column().classes("gap-0"):
            ui.label("💎 LDY Pro Trader").classes("text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400").style("font-family:Outfit,sans-serif")
            ui.label(f"v{APP_VERSION} · NiceGUI Edition").classes("text-xs text-gray-400")
        with ui.row().classes("gap-2 items-center"):
            if user:
                ui.label(f"👋 {user.get('nickname', '')}").classes("text-white text-sm")
                ui.badge(auth.upper()).props(f'color={"green" if auth in ("admin","prime") else "blue" if auth == "pro" else "gray"}')
                ui.button("로그아웃", on_click=lambda: (set_current_user(None), ui.navigate.to("/login"))).props("flat dense").classes("text-white text-xs")
            else:
                ui.button("🔐 로그인", on_click=lambda: ui.navigate.to("/login")).props("flat dense").classes("text-white")
            ui.button("🔄", on_click=_do_refresh).props("flat round dense").classes("text-white")

    if df.empty:
        ui.label("⚠️ 데이터 없음 — data/recommend_latest.csv 확인").classes("text-yellow-400 text-lg p-8")
        return

    # ─── 탭 구성 ───
    with ui.tabs().classes("w-full text-white") as tabs:
        t1 = ui.tab("📊 시장")
        t2 = ui.tab("🔭 종목 분석")
        t3 = ui.tab("💼 내 자산")
        t4 = ui.tab("📮 문의")
        t5 = ui.tab("⚖️ 약관")
        t6 = ui.tab("🧩 업데이트")
        t7 = ui.tab("📈 성과")
        t9 = ui.tab("📓 매매 일지")
        if auth == "admin":
            t8 = ui.tab("👑 관리")

    with ui.tab_panels(tabs, value=t1).classes("w-full"):
        with ui.tab_panel(t1): render_tab_market(df)  # ✅ components/tab_market.py
        with ui.tab_panel(t2): render_tab_stocks(df, auth, store)  # ✅ components/tab_stocks.py
        with ui.tab_panel(t3): render_tab_portfolio(df, auth)  # ✅ components/tab_portfolio.py
        with ui.tab_panel(t4): render_tab_inquiry(auth, user)  # ✅ components/tab_inquiry.py
        with ui.tab_panel(t5): render_tab_terms()  # ✅ components/tab_terms.py
        with ui.tab_panel(t6): render_tab_updates()  # ✅ components/tab_updates.py
        with ui.tab_panel(t7): render_tab_perf()  # ✅ components/tab_perf.py
        with ui.tab_panel(t9):
            if JOURNAL_OK:
                render_trade_journal_tab(df_scored=df)
            else:
                ui.label("⚠️ trade_journal_tab 모듈 없음").classes("text-yellow-400")
        if auth == "admin":
            with ui.tab_panel(t8): render_tab_admin()  # ✅ components/tab_admin.py

    # 푸터
    ui.label(f"📅 데이터 기준: {store.data_ts} · ⚠️ 투자 판단은 본인 책임").classes("text-xs text-gray-500 text-center mt-8 mb-4")


async def _do_refresh():
    await run_sync(store.refresh)
    ui.notify("🔄 데이터 새로고침 완료!", type="positive")
    await ui.run_javascript("setTimeout(()=>location.reload(),500)")


# ═══════════════════════════════════════════
#  앱 실행
# ═══════════════════════════════════════════
# PWA 정적 파일 서빙
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.add_static_files("/static", STATIC_DIR)

if __name__ in {"__main__", "__mp_main__"}:
    store.refresh()
    register_shutdown(app)  # [v6.0] 재배포 시 스레드 풀 + DB + Gist 최종 flush
    ui.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        title=f"LDY Pro Trader v{APP_VERSION}",
        favicon="💎",
        dark=True,
        storage_secret=os.environ["STORAGE_SECRET"],  # [v2.0 #3] 강제 — 미설정 시 앱 시작 차단
        reload=False,
        show=False,
    )
