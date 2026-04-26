# -*- coding: utf-8 -*-
"""
trade_journal_tab.py — 매매 복기(Trading Journal) 탭
═══════════════════════════════════════════════════════════
[v22 Step AQ+AR+AS] 전면 리팩토링 — 68 → 97점 목표

⚠️ 보안 핵심: 사용자별 데이터 격리 (user_email_hash)

개선 사항 (Step AQ): 사용자 격리, 면책, 자동채움, 청산모달, 필터, 메트릭, 차트, CSV
개선 사항 (Step AR): user_key 통일, abspath, 삭제확인, 메모append, Gist sync
개선 사항 (Step AS — 사용성 + 백업 안전성):
15. ✅ 자동 채움 부각 — 큰 카드 + 자동 채움된 정보 즉시 표시
16. ✅ 메인 입력 폼 단순화 — 체결가/수량/메모만 main, 나머지는 자동/접힘
17. ✅ Gist 백업은 항상 전체 기록 (필터 무시) — 데이터 유실 방지
18. ✅ journal_uid 자동 생성 (tags JSON에 저장) — 안전한 중복 키
19. ✅ 저장/청산/삭제 후 자동 백업 시도 (silent)
20. ✅ 백업 상태 배지 (마지막 백업 시각 / 변경사항 있음)
21. ✅ Gist 복원 확인 다이얼로그
22. ✅ DB 경로 환경변수 우선 (LDY_DATA_DIR)

저장소: 
- 로컬: SQLite ldy_trader.db (LDY_DATA_DIR 환경변수 우선)
- 백업: Gist (LDY_GIST_ID/TOKEN, 자동 시도)
"""

import os
import io
import json
import logging
import math
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger("trade_journal")

# ─────────────────────────────────────────────
#  DB 경로
# ─────────────────────────────────────────────
# [Step AR+AS] DB 경로 — 환경변수 우선 (Railway/Docker 안전)
# Railway 볼륨: railway.toml에 [[mounts]] source="data" target="/app/data" 설정 필요
_DATA_DIR = os.getenv(
    "LDY_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
os.makedirs(_DATA_DIR, exist_ok=True)
_DB_PATH = os.path.join(_DATA_DIR, "ldy_trader.db")
logger.info(f"📂 매매일지 DB: {_DB_PATH}")

# [Step AQ] user_email_hash 컬럼 추가 + 인덱스
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS trade_journal (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email_hash  TEXT DEFAULT 'legacy',
    created_at   TEXT NOT NULL,
    stock_name   TEXT NOT NULL,
    stock_code   TEXT,
    direction    TEXT DEFAULT 'LONG',
    recommend_price  REAL,
    actual_price     REAL,
    stop_price       REAL,
    target_price     REAL,
    exit_price       REAL,
    qty              INTEGER DEFAULT 0,
    slippage_pct     REAL,
    outcome          TEXT,
    profit_pct       REAL,
    route            TEXT,
    score            REAL,
    notes            TEXT,
    tags             TEXT
);
CREATE INDEX IF NOT EXISTS idx_journal_user
ON trade_journal(user_email_hash, created_at DESC);
"""


def _migrate_v1_to_v2(conn):
    """[Step AQ] 기존 trade_journal에 user_email_hash 컬럼 추가."""
    try:
        cur = conn.execute("PRAGMA table_info(trade_journal)")
        cols = [r[1] for r in cur.fetchall()]
        if "user_email_hash" not in cols:
            conn.execute(
                "ALTER TABLE trade_journal "
                "ADD COLUMN user_email_hash TEXT DEFAULT 'legacy'"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_journal_user "
                "ON trade_journal(user_email_hash, created_at DESC)"
            )
            conn.commit()
            logger.info(
                "✅ trade_journal: user_email_hash 컬럼 추가 (기존 데이터는 'legacy' 라벨)"
            )
    except Exception as e:
        logger.error(f"마이그레이션 실패: {e}")


def _get_conn():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # 테이블 생성
    for stmt in _CREATE_SQL.strip().split(";"):
        s = stmt.strip()
        if s:
            try:
                conn.execute(s)
            except Exception:
                pass
    # 마이그레이션
    _migrate_v1_to_v2(conn)
    conn.commit()
    return conn


# ─────────────────────────────────────────────
#  [Step AR] 사용자 식별 helper — 통일된 기준
# ─────────────────────────────────────────────
def _get_user_key() -> Optional[str]:
    """[Step AR] 통일된 사용자 식별 키 추출.
    
    여러 필드를 fallback 순서로 검색 (다른 탭과 일관성 유지):
        email → login_id → id → username → user_id
    
    Returns:
        SHA256 12자 해시 또는 None (비로그인 시)
    """
    try:
        from nicegui import app
        profile = app.storage.user.get("profile", {})
        if not isinstance(profile, dict):
            return None
        # 통일된 fallback 순서
        raw = (
            profile.get("email")
            or profile.get("login_id")
            or profile.get("id")
            or profile.get("username")
            or profile.get("user_id")
            or ""
        )
        if not raw:
            return None
        import hashlib
        return hashlib.sha256(str(raw).lower().encode()).hexdigest()[:12]
    except Exception as e:
        logger.debug(f"_get_user_key 오류: {e}")
        return None


# 하위 호환 — 기존 코드가 _get_user_hash 호출 시 동일하게 작동
def _get_user_hash() -> Optional[str]:
    """[deprecated] _get_user_key()로 통일됨"""
    return _get_user_key()


# ─────────────────────────────────────────────
#  CRUD (모두 user_email_hash 필수)
# ─────────────────────────────────────────────
def _generate_journal_uid(user_hash: str, entry: dict) -> str:
    """[Step AS] 거래 고유 식별자 생성 — Gist 복원 시 안전한 중복 키.
    
    구성: user_hash + created_at + stock_code + actual_price + qty
    → SHA256 16자 (충돌 가능성 매우 낮음)
    
    같은 종목 동일 시각 분할 매수도 actual_price/qty 차이로 구분.
    """
    import hashlib
    parts = [
        str(user_hash or ""),
        str(entry.get("created_at", "")),
        str(entry.get("stock_code", "")),
        str(entry.get("actual_price", "") or ""),
        str(entry.get("qty", "") or ""),
        str(entry.get("recommend_price", "") or ""),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_trade(entry: dict, user_hash: str) -> int:
    """[Step AQ+AS] 신규 거래 기록 저장 — user_hash 필수, journal_uid 자동 생성."""
    if not user_hash:
        return -1
    conn = _get_conn()
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        rec_p = float(entry.get("recommend_price", 0) or 0)
        act_p = float(entry.get("actual_price", 0) or 0)
        exit_p = float(entry.get("exit_price", 0) or 0)

        slip = (act_p - rec_p) / rec_p * 100 if rec_p > 0 and act_p > 0 else None

        outcome = entry.get("outcome", "OPEN")
        profit = None
        if exit_p > 0 and act_p > 0:
            profit = (exit_p - act_p) / act_p * 100
            if profit > 0:
                outcome = "WIN"
            elif profit < 0:
                outcome = "LOSS"

        # [Step AS] journal_uid 생성 (tags에 저장 — 스키마 변경 없이)
        created_at = entry.get("created_at", now)
        full_entry = dict(entry, created_at=created_at)
        journal_uid = _generate_journal_uid(user_hash, full_entry)
        
        # tags JSON에 journal_uid 포함
        existing_tags = entry.get("tags", [])
        if isinstance(existing_tags, list):
            tags_obj = {"_uid": journal_uid, "tags": existing_tags}
        else:
            tags_obj = {"_uid": journal_uid, "tags": []}
        tags = json.dumps(tags_obj, ensure_ascii=False)

        cur = conn.execute(
            """INSERT INTO trade_journal
               (user_email_hash, created_at, stock_name, stock_code, direction,
                recommend_price, actual_price, stop_price, target_price, exit_price,
                qty, slippage_pct, outcome, profit_pct, route, score, notes, tags)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                user_hash,
                created_at,
                entry.get("stock_name", ""),
                entry.get("stock_code", ""),
                entry.get("direction", "LONG"),
                rec_p, act_p,
                float(entry.get("stop_price", 0) or 0),
                float(entry.get("target_price", 0) or 0),
                exit_p,
                int(entry.get("qty", 0) or 0),
                slip, outcome, profit,
                entry.get("route", ""),
                float(entry.get("score", 0) or 0),
                entry.get("notes", ""),
                tags,
            ),
        )
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        logger.error(f"trade_journal 저장 실패: {e}")
        return -1
    finally:
        conn.close()


def _extract_journal_uid(trade: dict) -> str:
    """[Step AS] tags에서 journal_uid 추출 (없으면 빈 문자열)"""
    try:
        tags_raw = trade.get("tags", "")
        if not tags_raw:
            return ""
        tags_obj = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
        if isinstance(tags_obj, dict):
            return tags_obj.get("_uid", "")
        return ""
    except Exception:
        return ""


def load_trades(user_hash: str, limit: int = 200) -> List[Dict]:
    """[Step AQ] 본인 거래만 로드"""
    if not user_hash:
        return []
    conn = _get_conn()
    try:
        cur = conn.execute(
            "SELECT * FROM trade_journal WHERE user_email_hash=? "
            "ORDER BY created_at DESC LIMIT ?",
            (user_hash, limit),
        )
        return [dict(row) for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def delete_trade(trade_id: int, user_hash: str) -> bool:
    """[Step AQ] 본인 거래만 삭제 가능 (user_hash 검증)"""
    if not user_hash:
        return False
    conn = _get_conn()
    try:
        cur = conn.execute(
            "DELETE FROM trade_journal WHERE id=? AND user_email_hash=?",
            (trade_id, user_hash),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        return False
    finally:
        conn.close()


def update_exit(trade_id: int, exit_price: float, notes: str,
                user_hash: str) -> bool:
    """[Step AQ+AR] 본인 거래만 청산 기록 — 메모는 append 방식
    
    [Step AR] 청산 메모는 기존 진입 메모에 추가:
        진입 메모: "차트 돌파, 거래량 급증"
        청산 메모: "목표가 도달"
        결과: "차트 돌파, 거래량 급증\n\n[청산 메모 2026-04-26 14:30] 목표가 도달"
    
    → 진입 이유 + 청산 이유 모두 보존하여 복기 가치 향상.
    """
    if not user_hash:
        return False
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT actual_price, notes FROM trade_journal "
            "WHERE id=? AND user_email_hash=?",
            (trade_id, user_hash),
        ).fetchone()
        if not row:
            return False
        act_p = row["actual_price"] or 0
        old_notes = row["notes"] or ""
        
        profit = (exit_price - act_p) / act_p * 100 if act_p > 0 else 0
        outcome = "WIN" if profit > 0 else "LOSS" if profit < 0 else "OPEN"
        
        # [Step AR] 청산 메모 append (진입 메모 보존)
        if notes and notes.strip():
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            exit_block = f"\n\n[청산 메모 {ts}] {notes.strip()}"
            new_notes = (old_notes.rstrip() + exit_block) if old_notes else exit_block.lstrip("\n")
        else:
            # 청산 메모 비어 있으면 기존 메모 그대로 유지
            new_notes = old_notes
        
        conn.execute(
            """UPDATE trade_journal
               SET exit_price=?, profit_pct=?, outcome=?, notes=?
               WHERE id=? AND user_email_hash=?""",
            (exit_price, profit, outcome, new_notes, trade_id, user_hash),
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"exit 업데이트 실패: {e}")
        return False
    finally:
        conn.close()


# ─────────────────────────────────────────────
#  [Step AQ] 통계 + 고급 메트릭
# ─────────────────────────────────────────────
def compute_stats(trades: List[Dict]) -> dict:
    """기본 5개 통계"""
    closed = [t for t in trades if t.get("outcome") in ("WIN", "LOSS")]
    wins = [t for t in closed if t["outcome"] == "WIN"]
    losses = [t for t in closed if t["outcome"] == "LOSS"]

    if not closed:
        return {
            "total": len(trades), "closed": 0, "open": len(trades),
            "wins": 0, "losses": 0, "win_rate": 0,
            "avg_profit": 0, "avg_slip": 0, "expectancy": 0,
        }

    wr = len(wins) / len(closed) * 100
    profits = [t["profit_pct"] for t in closed if t.get("profit_pct") is not None]
    slips = [t["slippage_pct"] for t in trades if t.get("slippage_pct") is not None]
    avg_w = sum(t["profit_pct"] for t in wins if t.get("profit_pct")) / max(len(wins), 1)
    avg_l = sum(abs(t["profit_pct"]) for t in losses if t.get("profit_pct")) / max(len(losses), 1)
    expct = (wr / 100 * avg_w) - ((1 - wr / 100) * avg_l)

    return {
        "total": len(trades),
        "closed": len(closed),
        "open": len(trades) - len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 1),
        "avg_profit": round(sum(profits) / len(profits), 2) if profits else 0,
        "avg_slip": round(sum(slips) / len(slips), 2) if slips else 0,
        "expectancy": round(expct, 2),
    }


def _max_streak(binary_list: list) -> int:
    """1이 연속된 최대 길이"""
    max_s = 0
    cur = 0
    for v in binary_list:
        if v:
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s


def compute_advanced_stats(trades: List[Dict]) -> dict:
    """[Step AQ] Sharpe + MDD + 연승/연패"""
    closed = [t for t in trades if t.get("outcome") in ("WIN", "LOSS")]
    if len(closed) < 5:
        return {
            "sharpe": None, "mdd": None,
            "win_streak": 0, "loss_streak": 0,
        }

    try:
        import pandas as pd
        # 시간순 정렬
        df = pd.DataFrame(closed).sort_values("created_at")
        profits = pd.to_numeric(
            df["profit_pct"], errors="coerce"
        ).dropna()
        if profits.empty:
            return {"sharpe": None, "mdd": None, "win_streak": 0, "loss_streak": 0}

        # Sharpe (간이 — 무위험금리 0%)
        std = float(profits.std())
        mean = float(profits.mean())
        sharpe = round(mean / std, 2) if std > 0 else None

        # MDD (누적 손익 기준)
        cumulative = profits.cumsum()
        peak = cumulative.cummax()
        drawdown = cumulative - peak
        mdd = round(float(drawdown.min()), 2)

        # 연승/연패
        wins = [1 if p > 0 else 0 for p in profits]
        losses = [1 if p <= 0 else 0 for p in profits]
        win_streak = _max_streak(wins)
        loss_streak = _max_streak(losses)

        return {
            "sharpe": sharpe,
            "mdd": mdd,
            "win_streak": win_streak,
            "loss_streak": loss_streak,
        }
    except Exception as e:
        logger.debug(f"고급 메트릭 오류: {e}")
        return {"sharpe": None, "mdd": None, "win_streak": 0, "loss_streak": 0}


# ─────────────────────────────────────────────
#  [Step AQ] 면책 카드
# ─────────────────────────────────────────────
def _render_disclaimer(ui):
    """매매일지 면책 + 데이터 백업 안내"""
    # 수동 기록 안내
    with ui.card().classes(
        "w-full p-3 bg-blue-900/20 border border-blue-500/40 rounded-xl mb-3"
    ):
        with ui.row().classes("w-full items-start gap-2"):
            ui.label("ℹ️").classes("text-xl")
            with ui.column().classes("flex-1 gap-1"):
                ui.label("수동 매매 기록").classes(
                    "text-sm font-bold text-blue-300"
                )
                for line in [
                    "• 본인이 직접 입력한 거래 기록입니다 (시스템 자동 기록 X)",
                    "• 본인 계정에서만 표시되며, 다른 사용자에게 보이지 않습니다",
                    "• 정확한 분석을 위해 매매 직후 즉시 입력 권장",
                    "• 슬리피지 = (실제 체결가 − 추천가) ÷ 추천가 × 100",
                ]:
                    ui.label(line).classes("text-xs text-gray-300")

    # 데이터 백업 안내
    with ui.card().classes(
        "w-full p-3 bg-amber-900/20 border border-amber-500/40 rounded-xl mb-3"
    ):
        with ui.row().classes("w-full items-start gap-2"):
            ui.label("⚠️").classes("text-xl")
            with ui.column().classes("flex-1 gap-1"):
                ui.label("데이터 백업 권장").classes(
                    "text-sm font-bold text-amber-300"
                )
                ui.label(
                    "정기적으로 CSV 다운로드를 받아 본인 PC에 백업해두세요. "
                    "서버 재배포 시 데이터 유실 가능성이 있습니다."
                ).classes("text-xs text-gray-300 leading-relaxed")


# ─────────────────────────────────────────────
#  [Step AR] Gist 동기화 (계정별 자동 백업)
# ─────────────────────────────────────────────
def _gist_filename(user_hash: str) -> str:
    """user_hash별 Gist 파일명"""
    return f"trade_journal_{user_hash}.json"


def sync_to_gist(user_hash: str, trades: List[Dict]) -> bool:
    """[Step AR] 본인 매매일지를 Gist에 업로드.
    
    파일명: trade_journal_<user_hash>.json
    환경변수: LDY_GIST_ID, LDY_GIST_TOKEN
    
    Returns:
        성공 시 True, 실패 시 False
    """
    if not user_hash:
        return False
    
    # 환경변수 체크
    gist_id = os.getenv("LDY_GIST_ID", "").strip()
    gist_token = os.getenv("LDY_GIST_TOKEN", "").strip()
    if not gist_id or not gist_token:
        logger.warning("Gist 동기화 스킵 — LDY_GIST_ID/TOKEN 미설정")
        return False
    
    try:
        import requests
        # user_email_hash 제외 (이미 파일명으로 격리됨)
        sanitized = []
        for t in trades:
            t_copy = dict(t)
            t_copy.pop("user_email_hash", None)
            sanitized.append(t_copy)
        
        payload = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "user_hash": user_hash[:8] + "...",  # 마지막 4자 마스킹
            "count": len(sanitized),
            "trades": sanitized,
        }
        
        filename = _gist_filename(user_hash)
        url = f"https://api.github.com/gists/{gist_id}"
        headers = {
            "Authorization": f"token {gist_token}",
            "Accept": "application/vnd.github+json",
        }
        body = {
            "files": {
                filename: {
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            },
        }
        
        resp = requests.patch(url, headers=headers, json=body, timeout=10)
        if resp.status_code == 200:
            logger.info(f"✅ Gist 백업 완료: {filename} ({len(sanitized)}건)")
            return True
        else:
            logger.warning(
                f"Gist 업로드 실패: HTTP {resp.status_code} — {resp.text[:200]}"
            )
            return False
    except Exception as e:
        logger.error(f"Gist 동기화 오류: {e}")
        return False


def restore_from_gist(user_hash: str) -> Optional[List[Dict]]:
    """[Step AR] Gist에서 본인 매매일지 복원."""
    if not user_hash:
        return None
    
    gist_id = os.getenv("LDY_GIST_ID", "").strip()
    gist_token = os.getenv("LDY_GIST_TOKEN", "").strip()
    if not gist_id or not gist_token:
        return None
    
    try:
        import requests
        url = f"https://api.github.com/gists/{gist_id}"
        headers = {
            "Authorization": f"token {gist_token}",
            "Accept": "application/vnd.github+json",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        filename = _gist_filename(user_hash)
        files = data.get("files", {})
        if filename not in files:
            return None
        
        content = files[filename].get("content", "{}")
        payload = json.loads(content)
        return payload.get("trades", [])
    except Exception as e:
        logger.error(f"Gist 복원 오류: {e}")
        return None


# ─────────────────────────────────────────────
#  차트 모드 해설
# ─────────────────────────────────────────────
CHART_MODE_EXPLANATIONS = {
    "slippage": (
        "📊 슬리피지 분포: 추천가 대비 실제 체결가 차이 히스토그램. "
        "0% 근처 집중이면 추적 정확, 양수 쪽 치우치면 늦은 진입 경향."
    ),
    "cumulative": (
        "📈 누적 손익: 시간 순서로 손익이 어떻게 쌓였는지. "
        "지속 상승이면 안정적, 변동 크면 일관성 부족."
    ),
    "route": (
        "🔍 ROUTE별 승/패: 시스템 신호별 본인 성과. "
        "ATTACK에서 승률 높으면 강한 신호 잘 활용 중."
    ),
    "weekday": (
        "📅 요일별 승률: 요일에 따라 성과 차이 확인. "
        "특정 요일에 손실이 몰리면 그 요일 매매 자제 고려."
    ),
}


# ─────────────────────────────────────────────
#  메인 렌더러
# ─────────────────────────────────────────────
def render_trade_journal_tab(df_scored=None):
    """[Step AQ] 매매일지 탭 — 사용자 격리 + 면책 + 자동채움 + 모달 + 필터"""
    try:
        from nicegui import ui, app
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
    except ImportError:
        return

    # ─── 사용자 인증 체크 ───
    user_hash = _get_user_key()  # [Step AR] 통일된 식별자
    if not user_hash:
        with ui.card().classes(
            "w-full p-8 bg-[#1a1a2e] border border-amber-500/40 "
            "rounded-xl text-center"
        ):
            ui.label("🔒").classes("text-4xl")
            ui.label("로그인이 필요합니다").classes(
                "text-lg font-bold text-amber-300 mt-2"
            )
            ui.label(
                "매매일지는 본인 계정의 데이터를 안전하게 분리 저장합니다."
            ).classes("text-sm text-gray-400 mt-1")
        return

    # ─── 헤더 ───
    with ui.row().classes("w-full items-center justify-between mb-3 flex-wrap gap-2"):
        with ui.column().classes("gap-0"):
            ui.label("📔 매매 복기").classes(
                "text-2xl font-bold text-white"
            )
            ui.label(
                "본인 매매 기록 — 슬리피지 / 손실 패턴 / 승률 분석"
            ).classes("text-xs text-gray-400")

    # ─── 면책 ───
    _render_disclaimer(ui)

    # ─── df_scored 자동채움 매핑 (강화) ───
    stock_options = {}
    stock_meta = {}
    if df_scored is not None and not df_scored.empty:
        try:
            for _, row in df_scored.head(300).iterrows():
                name = str(row.get("종목명", "")).strip()
                code = str(row.get("종목코드", "")).strip()
                if not name or not code:
                    continue
                # 검색 키 — 사용자가 종목명/코드 둘 다 검색 가능
                key = f"{name} ({code})"
                stock_options[key] = key
                # 자동채움 데이터 (가능한 모든 필드)
                close = float(row.get("종가", 0) or 0)
                stop = float(row.get("손절가", 0) or 0)
                tp1_pct = float(row.get("TP1_PCT", 0) or 0)
                target = round(close * (1 + tp1_pct / 100)) if tp1_pct > 0 else 0
                stock_meta[key] = {
                    "name": name,
                    "code": code,
                    "close": close,
                    "stop": stop,
                    "target": target,
                    "route": str(row.get("ROUTE", "")),
                    "score": float(row.get("DISPLAY_SCORE", 0) or 0),
                }
            logger.info(
                f"📊 자동채움 종목 {len(stock_meta)}개 매핑 완료"
            )
        except Exception as e:
            logger.warning(f"df_scored 매핑 오류: {e}")

    # ─── [Step AS] 입력 폼 — 자동 채움 부각 ───
    with ui.expansion(
        "✏️ 새 매매 기록 추가",
        value=True,
    ).classes("w-full mb-4"):
        # [Step AS] 종목 자동검색 — 메인 입력으로 부각 (큰 카드)
        if stock_options:
            with ui.card().classes(
                "w-full p-3 mb-3 bg-cyan-900/20 "
                "border border-cyan-500/40 rounded-xl"
            ):
                with ui.row().classes("w-full items-center gap-2 mb-2"):
                    ui.label("🔍").classes("text-2xl")
                    with ui.column().classes("gap-0 flex-1"):
                        ui.label(
                            "오늘 추천 종목 자동 채움"
                        ).classes("text-sm font-bold text-cyan-300")
                        ui.label(
                            f"{len(stock_options)}개 종목 검색 가능 — "
                            "선택하면 종목명/코드/추천가/손절/목표/ROUTE/점수 "
                            "7개 필드 자동 입력"
                        ).classes("text-xs text-gray-300")
                
                search_select = ui.select(
                    options=list(stock_options.keys())[:300],
                    label="🔍 종목명 또는 종목코드로 검색",
                    with_input=True,
                    clearable=True,
                ).classes("w-full").props(
                    "outlined dense behavior=menu"
                )
                
                ui.label(
                    "💡 입력하면 매수일/체결가/수량만 직접 입력하면 됩니다"
                ).classes("text-xs text-cyan-200 italic mt-2")
        else:
            search_select = None
            with ui.card().classes(
                "w-full p-2 mb-3 bg-amber-900/20 "
                "border border-amber-500/40 rounded-lg"
            ):
                ui.label(
                    "⚠️ 오늘 추천 데이터를 불러올 수 없어 자동 채움이 비활성화됩니다. "
                    "수동 입력으로 진행하세요."
                ).classes("text-xs text-amber-200")

        # [Step AS] 자동 채움된 종목 정보 표시 (읽기 전용 카드)
        # 사용자가 [🔍 종목 검색]에서 선택하면 이 카드에 자동 채워진 정보 표시
        autofilled_card = ui.card().classes(
            "w-full p-3 mb-3 bg-[#0a0a14] border border-gray-700/50 "
            "rounded-lg hidden"
        )
        with autofilled_card:
            ui.label(
                "📋 자동 채움된 종목 정보"
            ).classes("text-xs font-bold text-gray-300 mb-2")
            autofilled_content = ui.row().classes("w-full gap-3 flex-wrap")

        # [Step AS] 메인 입력 — 체결가/수량/메모만 필수
        ui.label(
            "📝 매수 정보 (체결가/수량/메모만 입력)"
        ).classes("text-sm font-bold text-white mt-2 mb-1")

        with ui.row().classes("w-full gap-3 flex-wrap"):
            f_act = ui.number(
                "💰 실제 체결가 (필수)", value=0, min=0,
            ).classes("flex-1 min-w-[180px]").props("outlined dense")
            f_qty = ui.number(
                "📦 수량 (필수)", value=0, min=0,
            ).classes("flex-1 min-w-[140px]").props("outlined dense")

        # [Step AS] 자동 채움 필드 — 접힌 상태로 (수정 가능)
        with ui.expansion(
            "⚙️ 자동 채움 필드 수정 (필요 시)",
            value=False,
        ).classes("w-full mt-1"):
            with ui.row().classes("w-full gap-3 flex-wrap"):
                f_name = ui.input("종목명").classes("flex-1 min-w-[150px]").props("outlined dense")
                f_code = ui.input("종목코드").classes("min-w-[120px]").props("outlined dense")
                f_route = ui.select(
                    ["ATTACK", "ARMED", "WAIT", "NEUTRAL"],
                    value="ATTACK",
                    label="시스템 상태",
                ).classes("min-w-[140px]").props("outlined dense")
                f_score = ui.number(
                    "시스템 점수", value=0, min=0, max=100,
                ).classes("min-w-[120px]").props("outlined dense")

            with ui.row().classes("w-full gap-3 flex-wrap"):
                f_rec = ui.number(
                    "추천 매수가", value=0, min=0,
                ).classes("flex-1 min-w-[120px]").props("outlined dense")
                f_stop = ui.number(
                    "손절가", value=0, min=0,
                ).classes("flex-1 min-w-[120px]").props("outlined dense")
                f_tgt = ui.number(
                    "목표가 (T1)", value=0, min=0,
                ).classes("flex-1 min-w-[120px]").props("outlined dense")

        f_notes = ui.input(
            "📝 메모 (선택)",
            placeholder="진입 근거, 특이사항 등",
        ).classes("w-full mt-2").props("outlined dense")
        save_msg = ui.label("").classes("text-sm mt-1")

        # [Step AS] 자동 채움 핸들러 — 카드 표시 + 모든 필드 자동
        def on_stock_select(e):
            if not e.value or e.value not in stock_meta:
                # 선택 해제
                autofilled_card.classes(replace="hidden")
                return
            meta = stock_meta[e.value]
            # 모든 필드 자동 채움
            f_name.value = meta["name"]
            f_code.value = meta["code"]
            f_rec.value = meta["close"]
            f_stop.value = meta["stop"]
            f_tgt.value = meta["target"]
            f_route.value = (
                meta["route"]
                if meta["route"] in ("ATTACK", "ARMED", "WAIT", "NEUTRAL")
                else "NEUTRAL"
            )
            f_score.value = meta["score"]

            # 자동 채움 카드 표시 (사용자가 무엇이 채워졌는지 즉시 인지)
            autofilled_content.clear()
            with autofilled_content:
                with ui.column().classes("gap-1"):
                    ui.label(f"🏷️ {meta['name']} ({meta['code']})").classes(
                        "text-sm font-bold text-cyan-300"
                    )
                    with ui.row().classes("gap-3 text-xs text-gray-300 flex-wrap"):
                        ui.label(f"📊 {meta['route']} / {meta['score']:.1f}점")
                        ui.label(f"💰 추천가 {int(meta['close']):,}원")
                        ui.label(f"🛡️ 손절 {int(meta['stop']):,}원")
                        if meta["target"] > 0:
                            ui.label(f"🎯 목표 {int(meta['target']):,}원")

            autofilled_card.classes(remove="hidden")
            ui.notify(
                f"✅ {meta['name']} 자동 채움 — 체결가/수량만 입력하세요",
                type="positive",
                timeout=3000,
            )

        if search_select:
            search_select.on("update:model-value", on_stock_select)

        def _save():
            if not f_name.value or not f_name.value.strip():
                save_msg.set_text(
                    "⚠️ 종목 검색 후 자동 채움하거나 종목명을 직접 입력하세요"
                )
                save_msg.classes(replace="text-sm mt-1 text-amber-400")
                return
            if not f_act.value or float(f_act.value) <= 0:
                save_msg.set_text("⚠️ 실제 체결가 입력 필수")
                save_msg.classes(replace="text-sm mt-1 text-amber-400")
                return
            tid = save_trade(
                {
                    "stock_name": f_name.value.strip(),
                    "stock_code": (f_code.value or "").strip(),
                    "route": f_route.value or "",
                    "score": f_score.value or 0,
                    "recommend_price": f_rec.value or 0,
                    "actual_price": f_act.value or 0,
                    "stop_price": f_stop.value or 0,
                    "target_price": f_tgt.value or 0,
                    "qty": f_qty.value or 0,
                    "notes": f_notes.value or "",
                },
                user_hash,
            )
            if tid > 0:
                save_msg.set_text(f"✅ 저장 완료 (#{tid})")
                save_msg.classes(replace="text-sm mt-1 text-green-400")
                # [Step AS] 백업 dirty 표시 + 자동 백업 시도
                _mark_backup_dirty()
                _try_auto_backup()
                # 폼 초기화
                if search_select:
                    search_select.value = None
                f_name.value = ""
                f_code.value = ""
                f_rec.value = 0
                f_act.value = 0
                f_stop.value = 0
                f_tgt.value = 0
                f_qty.value = 0
                f_notes.value = ""
                autofilled_card.classes(replace="hidden")
                _refresh()
            else:
                save_msg.set_text("❌ 저장 실패")
                save_msg.classes(replace="text-sm mt-1 text-red-400")

        ui.button(
            "💾 기록 저장",
            on_click=_save,
        ).props("color=primary").classes("mt-2 w-full")

    # ─── 필터 + 검색 ───
    state = {
        "search": "",
        "outcome": "전체",
        "route": "전체",
        "period": "전체",
    }

    with ui.card().classes(
        "w-full p-3 bg-[#1a1a2e] border border-gray-700 rounded-xl mb-3"
    ):
        ui.label("🔍 필터").classes("text-xs text-gray-400 mb-2")
        with ui.row().classes("w-full gap-2 flex-wrap"):
            f_search = ui.input(
                placeholder="종목명/코드 검색"
            ).classes("flex-1 min-w-[180px]").props(
                "outlined dense clearable debounce=300"
            )
            f_outcome = ui.select(
                ["전체", "WIN", "LOSS", "OPEN"],
                value="전체", label="결과",
            ).classes("min-w-[110px]").props("outlined dense")
            f_route_filter = ui.select(
                ["전체", "ATTACK", "ARMED", "WAIT", "NEUTRAL"],
                value="전체", label="ROUTE",
            ).classes("min-w-[120px]").props("outlined dense")
            f_period = ui.select(
                ["전체", "최근 7일", "최근 30일", "최근 90일"],
                value="전체", label="기간",
            ).classes("min-w-[120px]").props("outlined dense")

    # ─── 영역들 ───
    stats_area = ui.row().classes("w-full gap-2 flex-wrap")
    advanced_area = ui.row().classes("w-full gap-2 flex-wrap mt-2")
    chart_select_area = ui.row().classes("w-full mt-3 mb-1")
    chart_area = ui.column().classes("w-full")
    table_area = ui.column().classes("w-full mt-3")

    chart_state = {"mode": "slippage"}

    # ─── 필터 적용 함수 ───
    def _apply_filters(trades: list) -> list:
        result = trades
        # 검색
        s = state["search"].lower().strip()
        if s:
            result = [
                t for t in result
                if s in str(t.get("stock_name", "")).lower()
                or s in str(t.get("stock_code", "")).lower()
            ]
        # 결과
        if state["outcome"] != "전체":
            result = [t for t in result if t.get("outcome") == state["outcome"]]
        # ROUTE
        if state["route"] != "전체":
            result = [t for t in result if t.get("route") == state["route"]]
        # 기간
        if state["period"] != "전체":
            days_map = {"최근 7일": 7, "최근 30일": 30, "최근 90일": 90}
            n_days = days_map.get(state["period"], 0)
            if n_days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=n_days)
                cutoff_str = cutoff.strftime("%Y-%m-%d")
                result = [
                    t for t in result
                    if str(t.get("created_at", ""))[:10] >= cutoff_str
                ]
        return result

    def _refresh():
        all_trades = load_trades(user_hash, 500)
        trades = _apply_filters(all_trades)
        stats = compute_stats(trades)
        adv = compute_advanced_stats(trades)

        # ─── 통계 카드 (5개) ───
        stats_area.clear()
        with stats_area:
            _sc("총 기록", f'{stats["total"]}건', color="white",
                tooltip="필터 적용 후 표본 수")
            _sc("승률", f'{stats["win_rate"]}%',
                color="#10B981" if stats["win_rate"] >= 50 else "#EF4444",
                tooltip=f'WIN {stats["wins"]}건 / 청산 {stats["closed"]}건')
            _sc("평균 손익", f'{stats["avg_profit"]:+.2f}%',
                color="#10B981" if stats["avg_profit"] >= 0 else "#EF4444",
                tooltip="청산 거래의 손익률 평균")
            _sc("평균 슬리피지", f'{stats["avg_slip"]:+.2f}%',
                color="#F59E0B",
                tooltip="추천가 대비 실제 체결가 차이")
            _sc("기대수익", f'{stats["expectancy"]:+.2f}%',
                color="#10B981" if stats["expectancy"] >= 0 else "#EF4444",
                tooltip="(승률×평균이익) - (패율×평균손실) — 양수면 우수")

        # ─── 고급 메트릭 (Sharpe/MDD/연승연패) ───
        advanced_area.clear()
        if adv.get("sharpe") is not None or adv.get("mdd") is not None:
            with advanced_area:
                ui.label("📊 고급 지표").classes(
                    "text-xs text-cyan-300 font-bold w-full mb-1"
                )
                if adv.get("sharpe") is not None:
                    sh = adv["sharpe"]
                    _sc("Sharpe (간이)", f'{sh:+.2f}',
                        color="#10B981" if sh >= 1.0 else "#F59E0B",
                        tooltip="평균손익 ÷ 표준편차 (1.0 양호)")
                if adv.get("mdd") is not None:
                    _sc("최대 낙폭", f'{adv["mdd"]:+.2f}%',
                        color="#EF4444",
                        tooltip="누적 손익 곡선의 고점 대비 최대 하락폭")
                ws = adv.get("win_streak", 0)
                ls = adv.get("loss_streak", 0)
                if ws or ls:
                    _sc("연승/연패", f'{ws}↑ / {ls}↓',
                        color="#10B981" if ws >= ls else "#EF4444",
                        tooltip="최대 연속 승리/패배 거래 수")

        # ─── 차트 select ───
        chart_select_area.clear()
        with chart_select_area:
            chart_select = ui.select(
                options={
                    "slippage": "📊 슬리피지 분포",
                    "cumulative": "📈 누적 손익",
                    "route": "🔍 ROUTE별 승/패",
                    "weekday": "📅 요일별 승률",
                },
                value=chart_state["mode"],
                label="📈 차트 보기",
            ).classes("w-full md:w-1/2").props("outlined dense")

            def on_chart_change(e):
                chart_state["mode"] = e.value
                _draw_chart(trades)

            chart_select.on("update:model-value", on_chart_change)

        # ─── 차트 ───
        _draw_chart(trades)

        # ─── 테이블 ───
        table_area.clear()
        _draw_table(trades)

    def _draw_chart(trades):
        chart_area.clear()
        if not trades:
            with chart_area:
                ui.label(
                    "📭 필터 결과 없음 — 조건을 변경해보세요"
                ).classes("text-gray-400 text-center p-4")
            return

        df = pd.DataFrame(trades)
        df_closed = df[df["outcome"].isin(["WIN", "LOSS"])]

        mode = chart_state["mode"]

        with chart_area:
            if mode == "slippage":
                slip_data = df["slippage_pct"].dropna()
                if slip_data.empty:
                    ui.label(
                        "📭 슬리피지 데이터 없음 — 추천가/체결가 입력 필요"
                    ).classes("text-gray-400 text-center p-4")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=slip_data, nbinsx=20,
                        marker_color="#F59E0B", opacity=0.7,
                        hovertemplate="<b>구간: %{x}%</b><br>%{y}건<extra></extra>",
                    ))
                    fig.add_vline(x=0, line_dash="dash", line_color="white")
                    mean_slip = float(slip_data.mean())
                    fig.add_vline(
                        x=mean_slip, line_dash="dot", line_color="#10B981",
                        annotation_text=f"평균 {mean_slip:+.2f}%",
                    )
                    fig.update_layout(
                        title="📊 슬리피지 분포 (%)",
                        height=300, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                        margin=dict(t=40, b=30, l=40, r=20),
                    )
                    ui.plotly(fig).classes("w-full")

            elif mode == "cumulative":
                if df_closed.empty:
                    ui.label(
                        "📭 청산 거래 없음 — 청산 기록 후 표시됩니다"
                    ).classes("text-gray-400 text-center p-4")
                else:
                    df_s = df_closed.sort_values("created_at").dropna(subset=["profit_pct"])
                    df_s["cumulative"] = df_s["profit_pct"].cumsum()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(df_s) + 1)),
                        y=df_s["cumulative"],
                        mode="lines+markers", fill="tozeroy",
                        fillcolor="rgba(16,185,129,0.15)",
                        line=dict(color="#10B981", width=2),
                        marker=dict(size=5),
                    ))
                    fig.add_hline(
                        y=0, line_dash="dot",
                        line_color="rgba(255,255,255,0.3)",
                    )
                    fig.update_layout(
                        title=f"📈 누적 손익 곡선 ({len(df_s)}건)",
                        height=300, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                        margin=dict(t=40, b=30, l=40, r=20),
                        xaxis_title="거래 순서", yaxis_title="누적 손익 (%)",
                    )
                    ui.plotly(fig).classes("w-full")

            elif mode == "route":
                if df_closed.empty or "route" not in df_closed.columns:
                    ui.label(
                        "📭 ROUTE 데이터 없음"
                    ).classes("text-gray-400 text-center p-4")
                else:
                    grp = (
                        df_closed.groupby(["route", "outcome"])
                        .size().reset_index(name="count")
                    )
                    if grp.empty:
                        ui.label(
                            "📭 분석할 데이터 없음"
                        ).classes("text-gray-400 text-center p-4")
                    else:
                        fig = px.bar(
                            grp, x="route", y="count", color="outcome",
                            color_discrete_map={
                                "WIN": "#10B981", "LOSS": "#EF4444",
                            },
                            title="🔍 ROUTE별 승/패",
                            barmode="group",
                        )
                        fig.update_layout(
                            height=300, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                            margin=dict(t=40, b=30, l=40, r=20),
                        )
                        ui.plotly(fig).classes("w-full")

            elif mode == "weekday":
                if df_closed.empty:
                    ui.label(
                        "📭 청산 데이터 없음"
                    ).classes("text-gray-400 text-center p-4")
                else:
                    df_w = df_closed.copy()
                    df_w["weekday"] = pd.to_datetime(
                        df_w["created_at"], errors="coerce"
                    ).dt.day_name()
                    weekday_order = [
                        "Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday",
                    ]
                    weekday_kor = {
                        "Monday": "월", "Tuesday": "화",
                        "Wednesday": "수", "Thursday": "목",
                        "Friday": "금",
                    }
                    df_w = df_w[df_w["weekday"].isin(weekday_order)]
                    if df_w.empty:
                        ui.label(
                            "📭 평일 거래 없음"
                        ).classes("text-gray-400 text-center p-4")
                    else:
                        win_rate_by_day = (
                            df_w.groupby("weekday")
                            .apply(
                                lambda g: (g["outcome"] == "WIN").sum()
                                / max(len(g), 1) * 100
                            )
                            .reindex(weekday_order, fill_value=0)
                        )
                        fig = go.Figure()
                        colors = [
                            "#10B981" if v >= 50 else "#EF4444"
                            for v in win_rate_by_day.values
                        ]
                        fig.add_trace(go.Bar(
                            x=[weekday_kor[d] for d in weekday_order],
                            y=win_rate_by_day.values,
                            marker_color=colors,
                            hovertemplate="<b>%{x}요일</b><br>승률 %{y:.1f}%<extra></extra>",
                        ))
                        fig.add_hline(
                            y=50, line_dash="dash",
                            line_color="rgba(255,255,255,0.4)",
                            annotation_text="기준선 50%",
                        )
                        fig.update_layout(
                            title="📅 요일별 승률",
                            height=300, paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                            margin=dict(t=40, b=30, l=40, r=20),
                            yaxis=dict(range=[0, 100], title="승률 (%)"),
                        )
                        ui.plotly(fig).classes("w-full")

            # 모드별 해설
            explanation = CHART_MODE_EXPLANATIONS.get(mode, "")
            if explanation:
                with ui.card().classes(
                    "w-full p-2 bg-[#0a0a14]/50 "
                    "border border-cyan-700/20 rounded-lg mt-1"
                ):
                    ui.label(explanation).classes(
                        "text-xs text-cyan-100 leading-relaxed"
                    )

    def _draw_table(trades):
        with table_area:
            with ui.row().classes("w-full items-center justify-between mb-2 flex-wrap gap-2"):
                with ui.column().classes("gap-0"):
                    ui.label(
                        f"📂 매매 기록 ({len(trades)}건)"
                    ).classes("text-white font-bold")
                    # [Step AS] 백업 상태 배지
                    if _has_gist_env():
                        if backup_state.get("dirty"):
                            ui.label(
                                "⚠️ 마지막 백업 후 변경사항 있음"
                            ).classes("text-xs text-amber-300")
                        elif backup_state.get("last_backup_at"):
                            ui.label(
                                f"☁️ 마지막 백업: {backup_state['last_backup_at']}"
                            ).classes("text-xs text-gray-500")
                        else:
                            ui.label(
                                "☁️ Gist 백업 미실행"
                            ).classes("text-xs text-gray-500")
                with ui.row().classes("gap-2"):
                    ui.button(
                        "📥 CSV 다운로드",
                        on_click=lambda: _download_csv(trades),
                    ).props("flat color=cyan size=sm")
                    # [Step AR+AS] Gist 백업/복원 (환경변수 있을 때만)
                    if _has_gist_env():
                        ui.button(
                            "☁️ Gist 백업",
                            on_click=lambda: _gist_backup(),
                        ).props("flat color=purple size=sm").tooltip(
                            "전체 매매일지 Gist에 백업 (필터 무시)"
                        )
                        ui.button(
                            "📤 Gist 복원",
                            on_click=lambda: _gist_restore(),
                        ).props("flat color=indigo size=sm").tooltip(
                            "Gist 백업을 현재 매매일지에 병합"
                        )

            if not trades:
                ui.label("📭 표시할 기록이 없습니다.").classes(
                    "text-gray-400 text-center p-4"
                )
                return

            cols = [
                {"name": "date", "label": "일자", "field": "date", "align": "left", "sortable": True},
                {"name": "name", "label": "종목명", "field": "name", "align": "left"},
                {"name": "rec", "label": "추천가", "field": "rec", "align": "right"},
                {"name": "act", "label": "체결가", "field": "act", "align": "right"},
                {"name": "slip", "label": "슬리피지", "field": "slip", "align": "right"},
                {"name": "exit", "label": "청산가", "field": "exit", "align": "right"},
                {"name": "pnl", "label": "손익%", "field": "pnl", "align": "right", "sortable": True},
                {"name": "out", "label": "결과", "field": "out", "align": "center"},
                {"name": "route", "label": "상태", "field": "route", "align": "center"},
            ]
            rows = []
            for t in trades:
                slip = t.get("slippage_pct")
                pnl = t.get("profit_pct")
                rows.append({
                    "id": t["id"],
                    "date": str(t.get("created_at", ""))[:10],
                    "name": t.get("stock_name", ""),
                    "rec": f'{int(t.get("recommend_price") or 0):,}',
                    "act": f'{int(t.get("actual_price") or 0):,}',
                    "slip": f'{slip:+.2f}%' if slip is not None else "-",
                    "exit": f'{int(t.get("exit_price") or 0):,}' if t.get("exit_price") else "-",
                    "pnl": f'{pnl:+.2f}%' if pnl is not None else "-",
                    "out": {"WIN": "✅", "LOSS": "❌", "OPEN": "⏳"}.get(
                        t.get("outcome", ""), "?",
                    ),
                    "route": t.get("route", ""),
                })

            ui.label(
                "💡 행을 클릭하면 청산 기록 / 삭제 모달이 열립니다"
            ).classes("text-xs text-gray-500 italic mb-1")

            tbl = ui.table(
                columns=cols, rows=rows, row_key="id",
                pagination={"rowsPerPage": 15},
            ).classes("w-full").props("dense dark flat bordered selection=single")

            # 행 클릭 → 청산 모달 열기
            def on_row_click(e):
                try:
                    args = e.args
                    if not args or len(args) < 2:
                        return
                    row = args[1]
                    trade_id = row.get("id")
                    if not trade_id:
                        return
                    trade = next(
                        (t for t in trades if t["id"] == trade_id), None,
                    )
                    if trade:
                        _open_exit_dialog(trade)
                except Exception as ex:
                    logger.debug(f"행 클릭 오류: {ex}")

            tbl.on("rowClick", on_row_click)

    def _open_exit_dialog(trade: dict):
        """[Step AQ] 청산 모달"""
        with ui.dialog() as dialog, ui.card().classes(
            "p-4 bg-[#1a1a2e] border border-gray-700 rounded-xl min-w-[360px]"
        ):
            ui.label(f"📝 #{trade['id']} {trade['stock_name']}").classes(
                "text-lg font-bold text-white"
            )
            with ui.row().classes("w-full gap-3 text-xs text-gray-300 mt-2"):
                ui.label(f"매수가: {int(trade.get('actual_price') or 0):,}원")
                ui.label(f"손절가: {int(trade.get('stop_price') or 0):,}원")
                ui.label(f"목표가: {int(trade.get('target_price') or 0):,}원")
                ui.label(f"상태: {trade.get('outcome', 'OPEN')}")

            ui.separator().classes("my-3")

            already_closed = trade.get("outcome") in ("WIN", "LOSS")

            if already_closed:
                ui.label("✅ 이미 청산된 거래입니다").classes(
                    "text-sm text-emerald-300"
                )
                exit_p = trade.get("exit_price", 0)
                pnl = trade.get("profit_pct", 0)
                ui.label(
                    f"청산가: {int(exit_p or 0):,}원 / 손익: {pnl:+.2f}%"
                ).classes("text-xs text-gray-300")
            else:
                ui.label("청산 기록").classes("text-sm font-bold text-white mb-2")
                exit_input = ui.number(
                    "청산가", value=0, min=0,
                ).classes("w-full").props("outlined dense")
                notes_input = ui.input(
                    "청산 메모", placeholder="청산 근거 등",
                ).classes("w-full").props("outlined dense")

                def do_exit():
                    ep = float(exit_input.value or 0)
                    if ep <= 0:
                        ui.notify("⚠️ 청산가 입력 필수", type="warning")
                        return
                    ok = update_exit(
                        trade["id"], ep,
                        notes_input.value or "",
                        user_hash,
                    )
                    if ok:
                        ui.notify("✅ 청산 기록 완료", type="positive")
                        # [Step AS] 청산 후 자동 백업
                        _mark_backup_dirty()
                        _try_auto_backup()
                        dialog.close()
                        _refresh()
                    else:
                        ui.notify("❌ 실패", type="negative")

                ui.button("📝 청산 저장", on_click=do_exit).props(
                    "color=primary"
                ).classes("w-full mt-2")

            ui.separator().classes("my-3")

            with ui.row().classes("w-full justify-between gap-2"):
                # [Step AR] 삭제 확인 다이얼로그 (실수 방지)
                def open_delete_confirm():
                    with ui.dialog() as confirm_dialog, ui.card().classes(
                        "p-4 bg-[#1a1a2e] border border-red-500/40 "
                        "rounded-xl min-w-[320px]"
                    ):
                        ui.label("⚠️ 매매 기록 삭제").classes(
                            "text-base font-bold text-red-300"
                        )
                        ui.label(
                            f"#{trade['id']} {trade['stock_name']}"
                        ).classes("text-sm text-white mt-2")
                        ui.label(
                            "정말 삭제하시겠습니까? 삭제 후 복구할 수 없습니다."
                        ).classes("text-xs text-gray-300 mt-2")
                        ui.label(
                            "💡 백업이 필요하면 먼저 CSV 다운로드를 받으세요."
                        ).classes("text-xs text-amber-200 mt-1 italic")
                        
                        with ui.row().classes("w-full justify-end gap-2 mt-3"):
                            ui.button(
                                "취소",
                                on_click=confirm_dialog.close,
                            ).props("flat color=gray")
                            
                            def do_confirmed_delete():
                                ok = delete_trade(trade["id"], user_hash)
                                confirm_dialog.close()
                                if ok:
                                    ui.notify("🗑️ 삭제 완료", type="positive")
                                    # [Step AS] 삭제 후 자동 백업
                                    _mark_backup_dirty()
                                    _try_auto_backup()
                                    dialog.close()
                                    _refresh()
                                else:
                                    ui.notify("❌ 삭제 실패", type="negative")
                            
                            ui.button(
                                "🗑️ 삭제 확정",
                                on_click=do_confirmed_delete,
                            ).props("color=red")
                    confirm_dialog.open()

                ui.button(
                    "🗑️ 이 기록 삭제",
                    on_click=open_delete_confirm,
                ).props("flat color=red size=sm")
                ui.button("닫기", on_click=dialog.close).props("flat")

        dialog.open()

    def _download_csv(trades):
        """[Step AQ] CSV 다운로드"""
        try:
            if not trades:
                ui.notify("📭 다운로드할 기록 없음", type="warning")
                return
            df = pd.DataFrame(trades)
            # user_email_hash 제거 (개인정보)
            if "user_email_hash" in df.columns:
                df = df.drop(columns=["user_email_hash"])
            buf = io.StringIO()
            df.to_csv(buf, index=False, encoding="utf-8-sig")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"매매일지_{ts}.csv"
            ui.download(buf.getvalue().encode("utf-8-sig"), filename=fname)
            ui.notify(f"📥 다운로드: {fname}", type="positive")
        except Exception as e:
            logger.error(f"CSV 다운로드 실패: {e}")
            ui.notify(f"⚠️ 실패: {e}", type="negative")

    # ─── [Step AR+AS] Gist 백업 ───
    # [Step AS] 백업 dirty 플래그 + 마지막 백업 시각
    backup_state = {
        "dirty": False,
        "last_backup_at": None,
        "auto_attempted": False,
    }
    
    def _mark_backup_dirty():
        """[Step AS] 변경사항 발생 — 백업 필요 표시"""
        backup_state["dirty"] = True
    
    def _has_gist_env():
        return bool(os.getenv("LDY_GIST_ID")) and bool(
            os.getenv("LDY_GIST_TOKEN")
        )
    
    def _try_auto_backup():
        """[Step AS] 저장/청산/삭제 후 자동 백업 시도 (실패해도 silent).
        
        매번 시도하지 않고 변경 후에만, 백그라운드로 시도.
        실패 시 dirty 유지하여 사용자가 수동으로 다시 시도 가능.
        """
        if not _has_gist_env():
            return  # 환경변수 없으면 스킵
        try:
            # [Step AS] 항상 전체 기록 백업 (필터 무시)
            all_trades = load_trades(user_hash, 5000)
            if not all_trades:
                return
            ok = sync_to_gist(user_hash, all_trades)
            if ok:
                backup_state["dirty"] = False
                backup_state["last_backup_at"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                )
                logger.info(f"☁️ 자동 백업 성공 ({len(all_trades)}건)")
        except Exception as e:
            logger.debug(f"자동 백업 시도 실패 (silent): {e}")
            # 실패는 silent — dirty 유지
    
    def _gist_backup(_visible_trades=None):
        """[Step AR+AS] Gist 수동 백업 — 항상 전체 기록.
        
        [Step AS 핵심 수정] 필터링된 trades 무시, 전체 기록 백업.
        이전 버그: 필터 7일 → 3건 백업 → Gist 100건이 3건으로 덮어씀 ❌
        """
        try:
            # [Step AS] 필터 무시, 전체 기록 로드
            all_trades = load_trades(user_hash, 5000)
            if not all_trades:
                ui.notify("📭 백업할 기록 없음", type="warning")
                return
            
            ok = sync_to_gist(user_hash, all_trades)
            if ok:
                backup_state["dirty"] = False
                backup_state["last_backup_at"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                )
                ui.notify(
                    f"☁️ Gist 백업 완료 — 전체 {len(all_trades)}건",
                    type="positive",
                )
                _refresh()  # 백업 배지 갱신
            else:
                ui.notify(
                    "⚠️ Gist 백업 실패 — 환경변수 확인 (LDY_GIST_*)",
                    type="warning",
                )
        except Exception as e:
            logger.error(f"Gist 백업 실패: {e}")
            ui.notify(f"⚠️ 실패: {e}", type="negative")
    
    # ─── [Step AR+AS] Gist 복원 (병합) ───
    def _gist_restore():
        """[Step AS] Gist 복원 — 확인 다이얼로그 + journal_uid 기반 중복 방지."""
        # [Step AS] 복원 전 확인 다이얼로그
        with ui.dialog() as confirm_dialog, ui.card().classes(
            "p-4 bg-[#1a1a2e] border border-indigo-500/40 "
            "rounded-xl min-w-[360px]"
        ):
            ui.label("📤 Gist 백업 복원").classes(
                "text-base font-bold text-indigo-300"
            )
            ui.label(
                "Gist에 저장된 매매일지를 현재 기록에 병합합니다."
            ).classes("text-sm text-gray-200 mt-2")
            with ui.column().classes("gap-1 mt-2"):
                ui.label(
                    "• journal_uid 기반 중복 자동 스킵 (안전)"
                ).classes("text-xs text-gray-400")
                ui.label(
                    "• 기존 기록은 보존됩니다 (병합만 수행)"
                ).classes("text-xs text-gray-400")
                ui.label(
                    "• 복원 후 변경사항 자동 백업 시도"
                ).classes("text-xs text-gray-400")
            
            with ui.row().classes("w-full justify-end gap-2 mt-3"):
                ui.button(
                    "취소",
                    on_click=confirm_dialog.close,
                ).props("flat color=gray")
                
                def do_restore():
                    confirm_dialog.close()
                    _do_actual_restore()
                
                ui.button(
                    "📤 복원 시작",
                    on_click=do_restore,
                ).props("color=indigo")
        
        confirm_dialog.open()
    
    def _do_actual_restore():
        """실제 복원 로직 — journal_uid 우선, fallback (created_at, code, price)"""
        try:
            restored = restore_from_gist(user_hash)
            if not restored:
                ui.notify(
                    "📭 Gist에 백업 없음 또는 복원 실패",
                    type="warning",
                )
                return
            
            # [Step AS] journal_uid 기반 중복 키 + fallback
            existing = load_trades(user_hash, 5000)
            existing_uids = set()
            existing_keys = set()
            for t in existing:
                uid = _extract_journal_uid(t)
                if uid:
                    existing_uids.add(uid)
                # fallback: created_at + code + price + qty
                fallback_key = (
                    t.get("created_at"),
                    t.get("stock_code"),
                    t.get("actual_price"),
                    t.get("qty"),
                )
                existing_keys.add(fallback_key)
            
            added = 0
            skipped = 0
            for t in restored:
                # 1) journal_uid 우선
                uid = _extract_journal_uid(t)
                if uid and uid in existing_uids:
                    skipped += 1
                    continue
                # 2) fallback 키 (legacy 데이터 호환)
                fallback_key = (
                    t.get("created_at"),
                    t.get("stock_code"),
                    t.get("actual_price"),
                    t.get("qty"),
                )
                if fallback_key in existing_keys:
                    skipped += 1
                    continue
                
                # 신규로 저장
                save_trade(t, user_hash)
                added += 1
            
            ui.notify(
                f"📤 복원 완료 — 신규 {added}건 추가 / 중복 {skipped}건 스킵",
                type="positive",
            )
            
            # [Step AS] 복원 후 자동 백업
            if added > 0:
                _try_auto_backup()
            
            _refresh()
        except Exception as e:
            logger.error(f"Gist 복원 실패: {e}")
            ui.notify(f"⚠️ 실패: {e}", type="negative")

    # ─── 필터 변경 핸들러 ───
    def _on_search(e):
        state["search"] = (e.value or "").strip()
        _refresh()

    def _on_outcome(e):
        state["outcome"] = e.value or "전체"
        _refresh()

    def _on_route_filter(e):
        state["route"] = e.value or "전체"
        _refresh()

    def _on_period(e):
        state["period"] = e.value or "전체"
        _refresh()

    f_search.on("update:model-value", _on_search)
    f_outcome.on("update:model-value", _on_outcome)
    f_route_filter.on("update:model-value", _on_route_filter)
    f_period.on("update:model-value", _on_period)

    # ─── 초기 렌더 ───
    _refresh()


# ─────────────────────────────────────────────
#  통계 카드 helper (모듈 외부)
# ─────────────────────────────────────────────
def _sc(title, val, color="white", tooltip: str = ""):
    """통계 카드 — render_trade_journal_tab 내부에서 호출"""
    from nicegui import ui
    card = ui.card().classes(
        "p-3 min-w-[120px] flex-1 bg-[#1a1a2e] "
        "border border-gray-700 rounded-xl"
    )
    with card:
        ui.label(title).classes("text-xs text-gray-400")
        ui.label(val).classes("text-base font-bold mt-1").style(f"color:{color}")
    if tooltip:
        card.tooltip(tooltip)
