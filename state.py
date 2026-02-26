# state.py — 앱 전역 상태 관리
"""
NiceGUI 다중 사용자 환경에서의 상태 관리 전략:

┌─────────────────────────────────────────────┐
│  GlobalData (앱 레벨, 모든 사용자 공유)      │
│  - master_df: 전체 종목 DataFrame (ReadOnly)  │
│  - data_ts: 데이터 기준일                     │
│  - feature_flags: 활성화된 기능               │
│  ⚠️ 절대 사용자별 필터/선택 상태를 넣지 말 것 │
├─────────────────────────────────────────────┤
│  SessionState (세션 레벨, 사용자별 독립)      │
│  - filtered_df: 필터 적용된 결과              │
│  - selected_ticker: 선택한 종목               │
│  - portfolio_text: 포트폴리오 입력값          │
│  → NiceGUI app.storage.user 또는 클로저로 관리│
└─────────────────────────────────────────────┘

사용법:
    from state import global_data
    df = global_data.scored   # ReadOnly 접근
    global_data.refresh()     # 데이터 갱신
"""
import os
import logging
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger("state")

KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    return datetime.now(KST)


class GlobalData:
    """앱 레벨 공유 데이터 — ReadOnly (모든 사용자가 동일한 인스턴스를 봄)

    ⚠️ 규칙:
    - scored DataFrame은 refresh() 시에만 교체 (atomic swap)
    - 사용자별 필터/선택은 여기에 절대 저장하지 않음
    - 탭 컴포넌트에서는 읽기만 허용
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 공유 데이터 (ReadOnly — refresh()로만 교체)
        self._scored: pd.DataFrame = pd.DataFrame()
        self._data_ts: str = ""
        self._loaded: bool = False

    # ── ReadOnly 프로퍼티 (setter 없음 → 외부에서 수정 불가) ──

    @property
    def scored(self) -> pd.DataFrame:
        """전체 종목 scored DataFrame (ReadOnly 참조)

        원본 보호: refresh()의 atomic swap으로만 교체되므로
        탭에서 .head()/.query()/.loc[] 등은 새 객체를 반환하여 안전.
        직접 .iloc[0, col] = val 같은 in-place 수정은 금지.
        필터링이 필요하면: filtered = state.scored.query("ROUTE == 'ATTACK'")
        """
        return self._scored

    @property
    def data_ts(self) -> str:
        return self._data_ts

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ── 데이터 갱신 (Atomic Swap) ──

    def refresh(self, csv_path: str = None):
        """CSV 로드 → atomic swap (락 불필요 — Python GIL 보장)"""
        if csv_path is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            csv_path = os.path.join(data_dir, "recommend_latest.csv")

        if not os.path.exists(csv_path):
            logger.warning(f"CSV not found: {csv_path}")
            return

        try:
            # 임시 변수에 로드 (실패해도 기존 데이터 유지)
            new_df = pd.read_csv(csv_path, dtype={"종목코드": str})

            # 숫자 컬럼 정리
            num_cols = [
                "FINAL_SCORE", "DISPLAY_SCORE", "STRUCT_SCORE",
                "TIMING_SCORE", "AI_SCORE", "ML_SCORE", "TOTAL_SCORE",
                "RANK_SCORE", "EBS", "RR1", "RSI14",
                "거래대금(억원)", "종가", "추천매수가", "손절가",
                "추천매도가1", "추천매도가2", "TARGET_ATR",
            ]
            for c in num_cols:
                if c in new_df.columns:
                    new_df[c] = pd.to_numeric(new_df[c], errors="coerce").fillna(0)

            # 대표 점수 컬럼 동기화
            primary = next(
                (c for c in ["DISPLAY_SCORE", "FINAL_SCORE", "TOTAL_SCORE"]
                 if c in new_df.columns and new_df[c].abs().sum() > 0),
                None,
            )
            if primary:
                for alias in ["DISPLAY_SCORE", "TOTAL_SCORE", "LDY_SCORE", "RANK_SCORE"]:
                    new_df[alias] = new_df[primary]

            # 데이터 기준일
            ts_col = next(
                (c for c in ["trade_date", "DATA_DATE"] if c in new_df.columns),
                None,
            )
            new_ts = str(new_df[ts_col].iloc[0]) if ts_col else now_kst().strftime("%Y-%m-%d")

            # Atomic swap — Python GIL이 참조 교체를 원자적으로 보장
            self._scored = new_df
            self._data_ts = new_ts
            self._loaded = True

            logger.info(f"✅ 데이터 로드: {len(new_df)}종목, 기준일 {new_ts}")

        except Exception as e:
            logger.exception(f"데이터 로드 실패: {e}")


# ═══════════════════════════════════════════
#  세션 상태 헬퍼 (NiceGUI app.storage.user 기반)
# ═══════════════════════════════════════════

class SessionKeys:
    """app.storage.user 키 상수 — 오타 방지"""
    FILTERED_DF = "filtered_df"       # 사용자가 필터 적용한 결과
    SELECTED_TICKER = "selected_ticker"
    PORTFOLIO_TEXT = "portfolio_text"
    VIEW_MODE = "view_mode"
    ROUTE_FILTER = "route_filter"


def get_session_val(key: str, default=None):
    """NiceGUI 세션 스토리지에서 값 조회

    ⚠️ ui.page 핸들러 안에서만 호출 가능.
    백그라운드 태스크에서 호출하면 RuntimeError 대신 default 반환.
    """
    try:
        from nicegui import app, context
        # context가 살아있는지 확인 (request scope 체크)
        _ = context.client
        return app.storage.user.get(key, default)
    except Exception:
        return default


def set_session_val(key: str, value):
    """NiceGUI 세션 스토리지에 값 저장

    ⚠️ ui.page 핸들러 안에서만 호출 가능.
    """
    try:
        from nicegui import app, context
        _ = context.client
        app.storage.user[key] = value
    except Exception:
        pass


# ── 전역 인스턴스 (앱 전체에서 1개) ──
global_data = GlobalData()
