# -*- coding: utf-8 -*-
"""
tab_admin.py — 👑 회원 관리 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
[v22 Step AV+AW+AX] 전면 리팩토링 — 64 → 96점 목표

⚠️ 사고 방지 핵심:
- 이전: 전체 초기화/전원 Prime 지급/차단/강등이 1클릭 즉시 실행 → 사고 위험
- 현재: 모든 위험 작업에 RESET 입력/확인 다이얼로그/미리보기 카운트

개선 사항 (Step AV): 위험 보호 4종, 검색 5종, 자동채움, 상세 모달, 액션 로그, 추세, CSV, 다중선택
개선 사항 (Step AW): Gist 파일명 표준화, 입금확인 보존, payment 탐지, 차단 필터, 매출 주석 정리

개선 사항 (Step AX — 운영 데이터 모델 강화):
16. ✅ admin_actions Gist 백업 (TABLE_TO_GIST_FILE 매핑 + admin_actions_db.json)
   - Railway 재배포 시에도 감사 로그 유실 방지
   - _log_admin_action 호출 시 자동 mark_gist_dirty
17. ✅ PAYMENT_CONFIRMED 직접 SQL 조회 (전체 범위)
   - 이전: 최근 200개 스캔 → 운영 길어지면 누락 위험
   - 수정: _get_payment_confirmed_action(inquiry_id) 전용 함수
18. ✅ 전체 초기화 범위 세분화 (3가지 옵션)
   - A. 회원만 초기화 (문의 보존)
   - B. 회원 + 일반 문의 (결제/환불 보존) ⭐ 권장
   - C. 회원 + 모든 문의 (분쟁 기록 사라짐) 🚨

향후 작업 (백엔드 단계 — db_utils.py 마이그레이션 필요):
- inquiries 테이블 category/status/admin_reply/admin_reply_at 컬럼 추가
- 입금확인 → Prime 부여 통합 버튼 (위 컬럼 의존)
- payments 테이블 → 실제 매출 통계
"""
import asyncio
import io
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from nicegui import ui

_logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))


# ═══════════════════════════════════════════════════
#  [Step AV] 액션 로그 (admin_actions 테이블)
# ═══════════════════════════════════════════════════
_ADMIN_ACTIONS_SQL = """
CREATE TABLE IF NOT EXISTS admin_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_email TEXT NOT NULL,
    action_type TEXT NOT NULL,
    target_email TEXT,
    details TEXT,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_admin_actions_target
ON admin_actions(target_email, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_admin_actions_time
ON admin_actions(timestamp DESC);
"""


def _ensure_admin_actions_table(db):
    """[Step AV] admin_actions 테이블 생성 (멱등)"""
    try:
        for stmt in _ADMIN_ACTIONS_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                db._exec_sqlite(s)
    except Exception as e:
        _logger.debug(f"admin_actions 테이블 생성 실패: {e}")


def _log_admin_action(
    db,
    admin_email: str,
    action_type: str,
    target_email: str = "",
    details: dict = None,
):
    """[Step AV+AX] 관리자 작업 기록 — 감사 추적 + Gist 백업
    
    Args:
        admin_email: 작업한 관리자 이메일
        action_type: ROLE_CHANGE / BAN_TOGGLE / BULK_TRIAL / RESET_ALL / DOWNGRADE / DELETE_USER 등
        target_email: 대상 회원 (전체 작업 시 'ALL')
        details: 추가 정보 (이전 값, 새 값, 사유 등)
    
    [Step AX] 기록 후 자동 Gist 동기화 (admin_actions_db.json)
    → Railway 재배포 시에도 감사 로그 유실 방지
    """
    if not db:
        return
    try:
        _ensure_admin_actions_table(db)
        ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        details_json = json.dumps(details or {}, ensure_ascii=False)
        db._exec_sqlite(
            "INSERT INTO admin_actions "
            "(admin_email, action_type, target_email, details, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (admin_email, action_type, target_email or "", details_json, ts),
        )
        _logger.info(
            f"📝 ADMIN: {admin_email} → {action_type} → {target_email or 'ALL'} | {details}"
        )
        # [Step AX] Gist 백업 트리거 (60초 배치 동기화)
        try:
            if hasattr(db, "_mark_gist_dirty"):
                db._mark_gist_dirty("admin_actions")
        except Exception as e:
            _logger.debug(f"admin_actions Gist 동기화 마킹 실패: {e}")
    except Exception as e:
        _logger.error(f"액션 로그 저장 실패: {e}")


def _get_admin_actions(db, target_email: str = None, limit: int = 50) -> list:
    """[Step AV] 액션 로그 조회"""
    if not db:
        return []
    try:
        _ensure_admin_actions_table(db)
        if target_email:
            rows = db._exec_sqlite(
                "SELECT * FROM admin_actions WHERE target_email=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (target_email, limit),
                fetch=True,
            )
        else:
            rows = db._exec_sqlite(
                "SELECT * FROM admin_actions ORDER BY timestamp DESC LIMIT ?",
                (limit,),
                fetch=True,
            )
        return [dict(r) for r in (rows or [])]
    except Exception as e:
        _logger.debug(f"액션 로그 조회 실패: {e}")
        return []


def _get_payment_confirmed_action(db, inquiry_id: str) -> dict:
    """[Step AX] 특정 inquiry의 PAYMENT_CONFIRMED 액션 직접 조회.
    
    이전 버그 (Step AW): 최근 200개만 스캔 → 운영 길어지면 누락
    수정 (Step AX): SQL LIKE로 inquiry_id 직접 검색 (전체 범위)
    
    Args:
        inquiry_id: inquiry created_at 또는 id (details JSON에 저장됨)
    
    Returns:
        매칭되는 첫 번째 액션 dict 또는 빈 dict
    """
    if not db or not inquiry_id:
        return {}
    try:
        _ensure_admin_actions_table(db)
        # SQLite LIKE로 details JSON 안의 inquiry_id 검색
        # JSON 형태: {"inquiry_id": "2026-04-26 12:30:00", ...}
        like_pattern = f'%"inquiry_id": "{inquiry_id}"%'
        rows = db._exec_sqlite(
            "SELECT * FROM admin_actions "
            "WHERE action_type='PAYMENT_CONFIRMED' AND details LIKE ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (like_pattern,),
            fetch=True,
        )
        if rows:
            return dict(rows[0])
        return {}
    except Exception as e:
        _logger.debug(f"PAYMENT_CONFIRMED 조회 실패: {e}")
        return {}


# ═══════════════════════════════════════════════════
#  헬퍼
# ═══════════════════════════════════════════════════
def _get_db():
    try:
        from db_utils import get_db
        db = get_db()
        if db and hasattr(db, 'ensure_gist_loaded'):
            db.ensure_gist_loaded()
        return db
    except Exception:
        return None


def _to_kst_str(value, fmt="%Y-%m-%d %H:%M:%S"):
    if not value or str(value).strip() in ("", "-", "None", "NaT"):
        return "-"
    try:
        import pandas as pd
        dt = pd.to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert(KST).strftime(fmt)
    except Exception:
        return str(value)


def _safe_str(value, default: str = "") -> str:
    """[Step AZ] None / 빈 값 안전 처리.
    
    DB의 NULL은 dict.get(key, default)에서 default가 적용되지 않음
    (키가 있고 값이 None이기 때문). startswith/슬라이싱/lower 직전에 사용.
    """
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _get_current_admin_email() -> str:
    """[Step AV] 현재 로그인된 관리자 이메일 (액션 로그용)"""
    try:
        from services.auth import get_current_user
        u = get_current_user()
        if u:
            return u.get("login_id") or u.get("id", "unknown_admin")
    except Exception:
        pass
    return "unknown_admin"


def _compute_growth_stats(users: list) -> dict:
    """[Step AV] 가입 추세 통계 계산"""
    now = datetime.now(KST)
    week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    month_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    this_month_start = now.replace(day=1).strftime("%Y-%m-%d")
    
    new_this_week = 0
    new_this_month = 0
    new_last_30_days = 0
    
    for u in users:
        join = str(u.get("join_date", ""))[:10]
        if not join:
            continue
        if join >= week_ago:
            new_this_week += 1
        if join >= month_ago:
            new_last_30_days += 1
        if join >= this_month_start:
            new_this_month += 1
    
    # 전환율 (prime / total)
    total = len(users)
    primes = sum(
        1 for u in users
        if str(u.get("role", "")).lower() == "prime"
    )
    conversion = (primes / total * 100) if total > 0 else 0
    
    # 만료 임박 (7일 내)
    expiring_soon = 0
    for u in users:
        if str(u.get("role", "")).lower() != "prime":
            continue
        exp = u.get("prime_expire_date")
        if not exp:
            continue
        try:
            exp_dt = datetime.strptime(str(exp)[:10], "%Y-%m-%d")
            if 0 <= (exp_dt.date() - now.date()).days <= 7:
                expiring_soon += 1
        except Exception:
            pass
    
    return {
        "new_this_week": new_this_week,
        "new_this_month": new_this_month,
        "new_last_30_days": new_last_30_days,
        "conversion_rate": round(conversion, 1),
        "primes": primes,
        "expiring_soon": expiring_soon,
    }


# ═══════════════════════════════════════════════════
#  메인 렌더러
# ═══════════════════════════════════════════════════
def render_tab_admin():
    """[Step AV] Tab 8: 회원 관리 (Admin) — 위험 작업 보호 + 검색 + 액션 로그"""

    db = _get_db()
    if not db:
        ui.label("❌ DB 연결 실패").classes("text-red-400")
        return
    
    # [Step AV] admin_actions 테이블 보장
    _ensure_admin_actions_table(db)
    
    admin_email = _get_current_admin_email()
    
    users = db.get_all_users()
    
    # ─── 헤더 ───
    with ui.row().classes("w-full items-center justify-between mb-3 flex-wrap gap-2"):
        with ui.column().classes("gap-0"):
            ui.label("👑 회원 관리").classes(
                "text-2xl font-bold text-white"
            )
            ui.label(
                f"관리자: {admin_email}"
            ).classes("text-xs text-gray-400")

    # ─── [Step AV] 면책 안내 ───
    with ui.card().classes(
        "w-full p-3 bg-blue-900/20 border border-blue-500/40 rounded-xl mb-3"
    ):
        with ui.row().classes("w-full items-start gap-2"):
            ui.label("ℹ️").classes("text-xl")
            with ui.column().classes("flex-1 gap-1"):
                ui.label("개인정보 처리 안내").classes(
                    "text-sm font-bold text-blue-300"
                )
                for line in [
                    "• 모든 관리자 작업은 admin_actions 테이블에 자동 기록됩니다 (감사 추적)",
                    "• 회원 정보는 개인정보보호법에 따라 5년 보관됩니다",
                    "• 위험 작업(전체 초기화/일괄 등급/일괄 차단)은 추가 확인이 필요합니다",
                    "• CSV 다운로드 시 비밀번호/세션 토큰은 자동 제외됩니다",
                ]:
                    ui.label(line).classes("text-xs text-gray-300")

    if not users:
        ui.label("👥 등록된 회원 없음").classes("text-gray-400 mt-4")
        return

    # ─── 회원 데이터 빌드 ───
    from services.auth import compute_access_status

    def _build_rows(users_list):
        rows = []
        for u in users_list:
            role_raw = u.get("role", "free").upper()
            _, allowed, reason = compute_access_status(u)

            if reason == "admin":
                access_label = "🔑관리자"
            elif reason == "active_subscription":
                access_label = "✅활성"
            elif reason == "expired":
                access_label = "❌만료"
            elif reason == "banned":
                access_label = "🚫차단"
            else:
                access_label = "⚪무료"

            rows.append({
                "email": u.get("login_id") or u.get("id", ""),
                "nick": u.get("nickname", ""),
                "role": role_raw,
                "access": access_label,
                "expire": (
                    _to_kst_str(u.get("prime_expire_date"), "%Y-%m-%d")
                    if u.get("prime_expire_date") else "-"
                ),
                "status": "🚫차단" if u.get("is_banned") else "✅",
                "joined": _to_kst_str(u.get("join_date"), "%Y-%m-%d"),
                "last": _to_kst_str(u.get("last_login")),
                "_user": u,  # 원본 보관 (자동 채움/상세용)
            })
        return rows

    # ─── 통계 계산 ───
    stats = _compute_growth_stats(users)
    rows_all = _build_rows(users)
    
    _active = sum(1 for r in rows_all if r["access"] == "✅활성")
    _expired = sum(1 for r in rows_all if r["access"] == "❌만료")
    _free = sum(1 for r in rows_all if r["access"] == "⚪무료")
    _admin = sum(1 for r in rows_all if r["access"] == "🔑관리자")
    _banned = sum(1 for r in rows_all if r["access"] == "🚫차단")

    # ─── [Step AV] 통계 카드 (기본 5종 + 추세 4종) ───
    ui.label("📊 회원 현황").classes(
        "text-sm font-bold text-cyan-300 mb-2"
    )
    
    with ui.grid().classes("w-full gap-2 grid-cols-2 md:grid-cols-5 mb-3"):
        _stat_card("👥 전체", len(users), "white")
        _stat_card("✅ 활성", _active, "#10B981")
        _stat_card("❌ 만료", _expired, "#EF4444")
        _stat_card("⚪ 무료", _free, "#6B7280")
        _stat_card(
            "⚠️ 만료 임박 (7일)",
            stats["expiring_soon"],
            "#F59E0B" if stats["expiring_soon"] > 0 else "#6B7280",
        )

    # 가입 추세 4종
    ui.label("📈 가입 추세").classes(
        "text-sm font-bold text-cyan-300 mb-2"
    )
    with ui.grid().classes("w-full gap-2 grid-cols-2 md:grid-cols-4 mb-3"):
        _stat_card(
            "📅 이번 주 신규",
            stats["new_this_week"],
            "#3B82F6",
        )
        _stat_card(
            "📅 이번 달 신규",
            stats["new_this_month"],
            "#3B82F6",
        )
        _stat_card(
            "💎 Prime 회원",
            stats["primes"],
            "#A78BFA",
        )
        _stat_card(
            "🔄 전환율",
            f"{stats['conversion_rate']}%",
            "#10B981" if stats["conversion_rate"] >= 30 else "#F59E0B",
            tooltip="(Prime 회원 / 전체) × 100",
        )

    # ─── [Step AV] 검색 + 필터 ───
    state = {
        "search": "",
        "access": "전체",
        "role": "전체",
        "period": "전체",
        "banned": "전체",
    }

    with ui.card().classes(
        "w-full p-3 bg-[#1a1a2e] border border-gray-700 rounded-xl mb-3"
    ):
        ui.label("🔍 검색 / 필터").classes(
            "text-xs text-gray-400 mb-2"
        )
        with ui.row().classes("w-full gap-2 flex-wrap"):
            f_search = ui.input(
                placeholder="이메일/닉네임 검색",
            ).classes("flex-1 min-w-[200px]").props(
                "outlined dense clearable debounce=300"
            )
            f_access = ui.select(
                ["전체", "✅활성", "❌만료", "⚪무료", "🔑관리자", "🚫차단"],
                value="전체", label="실제 상태",
            ).classes("min-w-[120px]").props("outlined dense")
            f_role = ui.select(
                ["전체", "FREE", "PRIME", "ADMIN"],
                value="전체", label="DB 등급",
            ).classes("min-w-[120px]").props("outlined dense")
            f_period = ui.select(
                ["전체", "최근 7일 가입", "최근 30일 가입", "만료 임박 (7일)"],
                value="전체", label="기간",
            ).classes("min-w-[160px]").props("outlined dense")
            # [Step AW] 차단 필터 UI 추가 (state와 일치)
            f_banned = ui.select(
                ["전체", "정상", "차단"],
                value="전체", label="차단 여부",
            ).classes("min-w-[120px]").props("outlined dense")

    # ─── 영역 ───
    table_area = ui.column().classes("w-full")
    
    def _apply_filters(rows):
        result = list(rows)
        # 검색
        s = state["search"].lower().strip()
        if s:
            result = [
                r for r in result
                if s in str(r.get("email", "")).lower()
                or s in str(r.get("nick", "")).lower()
            ]
        # 실제 상태
        if state["access"] != "전체":
            result = [r for r in result if r["access"] == state["access"]]
        # DB 등급
        if state["role"] != "전체":
            result = [r for r in result if r["role"] == state["role"]]
        # 기간
        if state["period"] != "전체":
            now = datetime.now(KST)
            if state["period"] == "최근 7일 가입":
                cutoff = (now - timedelta(days=7)).strftime("%Y-%m-%d")
                result = [
                    r for r in result
                    if str(r.get("joined", ""))[:10] >= cutoff
                ]
            elif state["period"] == "최근 30일 가입":
                cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%d")
                result = [
                    r for r in result
                    if str(r.get("joined", ""))[:10] >= cutoff
                ]
            elif state["period"] == "만료 임박 (7일)":
                today = now.strftime("%Y-%m-%d")
                week_later = (now + timedelta(days=7)).strftime("%Y-%m-%d")
                result = [
                    r for r in result
                    if r["role"] == "PRIME"
                    and r["expire"] != "-"
                    and today <= r["expire"] <= week_later
                ]
        # [Step AW] 차단 필터
        if state["banned"] == "차단":
            result = [r for r in result if r["status"] == "🚫차단"]
        elif state["banned"] == "정상":
            result = [r for r in result if r["status"] == "✅"]
        return result

    def _refresh_table():
        table_area.clear()
        filtered = _apply_filters(rows_all)
        
        with table_area:
            with ui.row().classes("w-full items-center justify-between mb-2 flex-wrap gap-2"):
                ui.label(
                    f"📂 회원 목록 ({len(filtered)} / 전체 {len(rows_all)})"
                ).classes("text-white font-bold")
                with ui.row().classes("gap-2"):
                    ui.button(
                        "📥 CSV 다운로드",
                        on_click=lambda: _download_csv(filtered),
                    ).props("flat color=cyan size=sm").tooltip(
                        "현재 필터된 회원 목록 (비밀번호/토큰 제외)"
                    )

            if not filtered:
                ui.label(
                    "📭 필터 결과가 없습니다 — 조건을 변경해보세요"
                ).classes("text-gray-400 text-center p-4")
                return

            columns = [
                {"name": "email", "label": "이메일", "field": "email", "align": "left", "sortable": True},
                {"name": "nick", "label": "닉네임", "field": "nick"},
                {"name": "role", "label": "DB권한", "field": "role"},
                {"name": "access", "label": "실제상태", "field": "access"},
                {"name": "expire", "label": "구독만료", "field": "expire", "sortable": True},
                {"name": "status", "label": "차단", "field": "status"},
                {"name": "joined", "label": "가입일", "field": "joined", "sortable": True},
                {"name": "last", "label": "최근접속", "field": "last"},
                {"name": "actions", "label": "액션", "field": "actions", "align": "center"},
            ]
            
            display_rows = [
                {**{k: v for k, v in r.items() if k != "_user"}, "actions": "👁️"}
                for r in filtered
            ]
            
            ui.label(
                "💡 액션 버튼(👁️) 클릭 = 회원 상세  ·  "
                "체크박스 = 다중 선택 후 일괄 작업"
            ).classes("text-xs text-gray-500 italic mb-1")

            tbl = ui.table(
                columns=columns,
                rows=display_rows,
                row_key="email",
                pagination={"rowsPerPage": 20},
                selection="multiple",
            ).classes("w-full").props("dense dark flat bordered")
            
            # 액션 버튼 슬롯
            tbl.add_slot("body-cell-actions", """
                <q-td :props="props">
                    <q-btn flat dense round color="cyan" icon="visibility" size="sm"
                           @click="() => $parent.$emit('detail', props.row)">
                        <q-tooltip>회원 상세 / 액션 로그</q-tooltip>
                    </q-btn>
                </q-td>
            """)
            
            def on_detail(e):
                try:
                    row = e.args
                    if not row:
                        return
                    user_email = row.get("email")
                    user_obj = next(
                        (r["_user"] for r in filtered if r["email"] == user_email),
                        None,
                    )
                    if user_obj:
                        _open_user_detail(user_obj)
                except Exception as ex:
                    _logger.debug(f"상세 모달 오류: {ex}")
            
            tbl.on("detail", on_detail)
            
            # 다중 선택 액션 바
            with ui.row().classes("w-full mt-2 items-center gap-2 flex-wrap"):
                selection_label = ui.label("").classes(
                    "text-xs text-cyan-300"
                )
                
                def update_label():
                    n = len(tbl.selected) if tbl.selected else 0
                    selection_label.set_text(
                        f"☑️ {n}건 선택됨" if n > 0 else ""
                    )
                
                tbl.on("selection", lambda _: update_label())
                
                ui.button(
                    "🚫 일괄 차단",
                    on_click=lambda: _bulk_action(
                        tbl.selected, "BAN_TOGGLE", filtered,
                    ),
                ).props("flat color=red size=sm").tooltip(
                    "선택한 회원 모두 차단/해제 토글"
                )
                ui.button(
                    "⏰ 일괄 강등 (FREE)",
                    on_click=lambda: _bulk_action(
                        tbl.selected, "ROLE_DOWNGRADE", filtered,
                    ),
                ).props("flat color=amber size=sm").tooltip(
                    "선택한 회원 모두 FREE로 변경"
                )

    def _download_csv(rows):
        """[Step AV] CSV 다운로드 (개인정보 제외)"""
        try:
            import pandas as pd
            if not rows:
                ui.notify("📭 다운로드할 회원 없음", type="warning")
                return
            # 표시용 컬럼만 (비밀번호/토큰 제외)
            safe_data = []
            for r in rows:
                safe_data.append({
                    "email": r.get("email"),
                    "nickname": r.get("nick"),
                    "role": r.get("role"),
                    "access_status": r.get("access"),
                    "prime_expire": r.get("expire"),
                    "is_banned": r.get("status"),
                    "join_date": r.get("joined"),
                    "last_login": r.get("last"),
                })
            df = pd.DataFrame(safe_data)
            buf = io.StringIO()
            df.to_csv(buf, index=False, encoding="utf-8-sig")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"회원목록_{ts}.csv"
            ui.download(buf.getvalue().encode("utf-8-sig"), filename=fname)
            ui.notify(f"📥 다운로드: {fname}", type="positive")
            _log_admin_action(
                db, admin_email, "CSV_EXPORT", "ALL",
                {"count": len(rows)},
            )
        except Exception as e:
            ui.notify(f"⚠️ 실패: {e}", type="negative")

    def _bulk_action(selected_rows, action: str, filtered: list):
        """[Step AV] 일괄 작업 — 차단/강등 + 확인 다이얼로그"""
        if not selected_rows:
            ui.notify("⚠️ 회원을 선택하세요", type="warning")
            return
        emails = [r.get("email") for r in selected_rows if r.get("email")]
        if not emails:
            return
        
        action_label = {
            "BAN_TOGGLE": "차단/해제 토글",
            "ROLE_DOWNGRADE": "FREE로 강등",
        }.get(action, action)
        
        with ui.dialog() as cd, ui.card().classes(
            "p-4 bg-[#1a1a2e] border border-amber-500/40 "
            "rounded-xl min-w-[400px]"
        ):
            ui.label(f"⚠️ 일괄 작업: {action_label}").classes(
                "text-base font-bold text-amber-300"
            )
            ui.label(f"대상: {len(emails)}명").classes(
                "text-sm text-white mt-2"
            )
            
            # 대상 미리보기 (최대 10명)
            preview = ", ".join(emails[:10])
            if len(emails) > 10:
                preview += f" 외 {len(emails) - 10}명"
            ui.label(preview).classes(
                "text-xs text-gray-300 mt-1"
            )
            
            ui.label(
                "이 작업은 액션 로그에 기록되며, 회원에게 즉시 적용됩니다."
            ).classes("text-xs text-gray-400 mt-2")
            
            reason_input = ui.input(
                "사유 (액션 로그에 기록)",
                placeholder="일괄 작업 사유",
            ).classes("w-full mt-2").props("outlined dense")
            
            with ui.row().classes("w-full justify-end gap-2 mt-3"):
                ui.button("취소", on_click=cd.close).props("flat color=gray")
                
                async def do_bulk():
                    cd.close()
                    reason = (reason_input.value or "").strip()
                    
                    def _bulk_exec():
                        db_a = _get_db()
                        if not db_a:
                            return 0, "DB 연결 실패"
                        success = 0
                        for em in emails:
                            try:
                                if action == "BAN_TOGGLE":
                                    db_a.toggle_user_ban(em)
                                elif action == "ROLE_DOWNGRADE":
                                    db_a.update_user_role(em, "free")
                                success += 1
                                _log_admin_action(
                                    db_a, admin_email,
                                    action, em,
                                    {"reason": reason, "bulk": True},
                                )
                            except Exception as ex:
                                _logger.warning(
                                    f"일괄 작업 실패 ({em}): {ex}"
                                )
                        return success, f"{success}/{len(emails)}명 처리 완료"
                    
                    success, msg = await asyncio.to_thread(_bulk_exec)
                    if success > 0:
                        ui.notify(f"✅ {msg}", type="positive")
                        # 데이터 새로고침
                        nonlocal users, rows_all
                        users[:] = db.get_all_users()
                        rows_all[:] = _build_rows(users)
                        _refresh_table()
                    else:
                        ui.notify(f"❌ {msg}", type="negative")
                
                ui.button(
                    f"⚠️ {len(emails)}명에게 {action_label} 실행",
                    on_click=do_bulk,
                ).props("color=amber")
        cd.open()

    # ─── 회원 상세 모달 ───
    def _open_user_detail(user: dict):
        """[Step AV] 회원 상세 — 결제/문의 이력 + 액션 로그"""
        email = user.get("login_id") or user.get("id", "")
        nickname = user.get("nickname", "")
        
        with ui.dialog() as dialog, ui.card().classes(
            "p-4 bg-[#1a1a2e] border border-gray-700 rounded-xl "
            "min-w-[500px] max-w-[800px]"
        ):
            ui.label(f"👤 {nickname or email}").classes(
                "text-lg font-bold text-white"
            )
            ui.label(email).classes("text-xs text-gray-400")
            ui.separator().classes("my-2")
            
            # 기본 정보
            with ui.grid().classes("w-full gap-2 grid-cols-2"):
                _info_row("📋 등급", user.get("role", "free"))
                role_status = compute_access_status(user)[2]
                _info_row("🔐 실제 상태", role_status)
                _info_row(
                    "💎 만료일",
                    _to_kst_str(user.get("prime_expire_date"), "%Y-%m-%d")
                    if user.get("prime_expire_date") else "-",
                )
                _info_row(
                    "🚫 차단 여부",
                    "차단됨" if user.get("is_banned") else "정상",
                )
                _info_row(
                    "📅 가입일",
                    _to_kst_str(user.get("join_date"), "%Y-%m-%d"),
                )
                _info_row(
                    "🕐 최근 접속",
                    _to_kst_str(user.get("last_login")),
                )
            
            # 문의 이력
            ui.label("📨 문의 이력").classes(
                "text-sm font-bold text-cyan-300 mt-3 mb-1"
            )
            try:
                inquiries = db.get_all_inquiries()
                user_inquiries = [
                    q for q in inquiries
                    if q.get("id") == email or q.get("login_id") == email
                ][:5]
                if user_inquiries:
                    for q in user_inquiries:
                        with ui.card().classes(
                            "w-full p-2 bg-[#0a0a14] "
                            "border border-gray-700/50 rounded-lg mb-1"
                        ):
                            ui.label(
                                f"📌 {_safe_str(q.get('title'))[:80]}"
                            ).classes("text-xs text-white")
                            ui.label(
                                _to_kst_str(q.get("created_at"))
                            ).classes("text-xs text-gray-500")
                else:
                    ui.label("문의 이력 없음").classes(
                        "text-xs text-gray-500 italic"
                    )
            except Exception:
                ui.label("문의 조회 실패").classes("text-xs text-gray-500")
            
            # 액션 로그 (이 회원에 대한)
            ui.label("📋 관리자 작업 이력 (최근 10건)").classes(
                "text-sm font-bold text-cyan-300 mt-3 mb-1"
            )
            actions = _get_admin_actions(db, target_email=email, limit=10)
            if actions:
                for a in actions:
                    try:
                        details = json.loads(a.get("details", "{}") or "{}")
                    except Exception:
                        details = {}
                    with ui.card().classes(
                        "w-full p-2 bg-[#0a0a14] "
                        "border border-gray-700/50 rounded-lg mb-1"
                    ):
                        with ui.row().classes(
                            "w-full justify-between items-start"
                        ):
                            ui.label(
                                f"⚡ {a.get('action_type', '')}"
                            ).classes("text-xs text-amber-300 font-bold")
                            ui.label(
                                _safe_str(a.get("timestamp"))[:16]
                            ).classes("text-xs text-gray-500")
                        ui.label(
                            f"by {a.get('admin_email', '')[:20]}"
                        ).classes("text-xs text-gray-400")
                        if details:
                            detail_str = ", ".join(
                                f"{k}={v}" for k, v in list(details.items())[:3]
                            )
                            ui.label(detail_str).classes(
                                "text-xs text-gray-300"
                            )
            else:
                ui.label("작업 이력 없음").classes(
                    "text-xs text-gray-500 italic"
                )
            
            with ui.row().classes("w-full justify-end mt-3"):
                ui.button("닫기", on_click=dialog.close).props("flat")
        
        dialog.open()

    # ─── [Step AV] 개별 회원 제어 (현재 값 자동 채움) ───
    ui.separator().classes("my-4")
    ui.label("🛠️ 관리자 액션").classes(
        "text-lg font-bold text-white mb-3"
    )
    
    with ui.row().classes("w-full gap-3 flex-wrap"):
        # 개별 회원 제어
        with ui.card().classes(
            "flex-1 min-w-[300px] p-4 bg-[#1a1a2e] "
            "border border-gray-700 rounded-xl"
        ):
            ui.label("👤 개별 회원 제어").classes(
                "text-sm font-bold text-cyan-300 mb-2"
            )
            
            emails = [r["email"] for r in rows_all]
            sel_email = ui.select(
                options=emails, label="회원 선택", with_input=True,
            ).classes("w-full").props("outlined dense clearable")
            
            # 현재 값 표시
            current_info = ui.label("").classes(
                "text-xs text-gray-400 mt-1"
            )
            
            sel_role = ui.select(
                ["free", "prime", "admin"],
                label="등급 변경", value="free",
            ).classes("w-full mt-2").props("outlined dense")
            sel_days = ui.select(
                {
                    0: "만료일 없음", 7: "7일",
                    14: "14일", 30: "30일 (1개월)",
                    90: "90일 (3개월)", 180: "180일 (6개월)",
                    365: "365일 (1년)",
                },
                label="구독 기간", value=30,
            ).classes("w-full mt-2").props("outlined dense")
            
            # [Step AV] 회원 선택 시 현재 값 자동 채움
            def on_user_select(e):
                if not e.value:
                    current_info.set_text("")
                    sel_role.value = "free"
                    return
                user_obj = next(
                    (r["_user"] for r in rows_all
                     if r["email"] == e.value),
                    None,
                )
                if user_obj:
                    cur_role = str(user_obj.get("role", "free")).lower()
                    cur_expire = (
                        _to_kst_str(
                            user_obj.get("prime_expire_date"),
                            "%Y-%m-%d",
                        )
                        if user_obj.get("prime_expire_date") else "-"
                    )
                    cur_banned = "🚫차단" if user_obj.get("is_banned") else "✅정상"
                    current_info.set_text(
                        f"현재: {cur_role.upper()} / "
                        f"만료: {cur_expire} / {cur_banned}"
                    )
                    # select 값을 현재 값으로 (덮어쓰기 사고 방지)
                    sel_role.value = cur_role if cur_role in (
                        "free", "prime", "admin",
                    ) else "free"
            
            sel_email.on("update:model-value", on_user_select)
            
            async def apply_role_with_confirm():
                if not sel_email.value:
                    ui.notify("⚠️ 회원을 선택하세요", type="warning")
                    return
                
                target = sel_email.value
                new_role = sel_role.value
                days = sel_days.value or 0
                
                # 현재 값 가져오기
                user_obj = next(
                    (r["_user"] for r in rows_all if r["email"] == target),
                    None,
                )
                cur_role = (
                    str(user_obj.get("role", "free")).lower()
                    if user_obj else "free"
                )
                
                # [Step AV] 확인 다이얼로그
                with ui.dialog() as cd, ui.card().classes(
                    "p-4 bg-[#1a1a2e] border border-cyan-500/40 "
                    "rounded-xl min-w-[400px]"
                ):
                    ui.label("👤 회원 등급 변경 확인").classes(
                        "text-base font-bold text-cyan-300"
                    )
                    ui.label(f"대상: {target}").classes(
                        "text-sm text-white mt-2"
                    )
                    ui.label(
                        f"변경: {cur_role.upper()} → {new_role.upper()}"
                    ).classes("text-sm text-gray-200 mt-1")
                    if days > 0 and new_role == "prime":
                        ui.label(
                            f"만료일: {(datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')}"
                        ).classes("text-xs text-amber-300 mt-1")
                    
                    reason_input = ui.input(
                        "사유 (액션 로그)",
                        placeholder="등급 변경 사유",
                    ).classes("w-full mt-2").props("outlined dense")
                    
                    with ui.row().classes("w-full justify-end gap-2 mt-3"):
                        ui.button("취소", on_click=cd.close).props("flat")
                        
                        async def do_apply():
                            cd.close()
                            reason = (reason_input.value or "").strip()
                            
                            def _do():
                                db_a = _get_db()
                                if not db_a:
                                    return False, "DB 연결 실패"
                                if days > 0 and new_role == "prime":
                                    expire = (
                                        datetime.now() + timedelta(days=days)
                                    ).strftime("%Y-%m-%d")
                                    db_a.update_user_subscription(
                                        target, new_role, expire,
                                    )
                                    _log_admin_action(
                                        db_a, admin_email, "ROLE_CHANGE", target,
                                        {
                                            "from": cur_role, "to": new_role,
                                            "expire": expire, "reason": reason,
                                        },
                                    )
                                    return True, (
                                        f"✅ {target} → PRIME (만료: {expire})"
                                    )
                                else:
                                    ok = db_a.update_user_role(target, new_role)
                                    if ok:
                                        _log_admin_action(
                                            db_a, admin_email,
                                            "ROLE_CHANGE", target,
                                            {
                                                "from": cur_role,
                                                "to": new_role,
                                                "reason": reason,
                                            },
                                        )
                                    return ok, "✅ 변경 완료" if ok else "❌ 실패"
                            
                            ok, msg = await asyncio.to_thread(_do)
                            ui.notify(msg, type="positive" if ok else "negative")
                            if ok:
                                nonlocal users, rows_all
                                users[:] = db.get_all_users()
                                rows_all[:] = _build_rows(users)
                                _refresh_table()
                        
                        ui.button("✅ 변경 확정", on_click=do_apply).props(
                            "color=primary"
                        )
                cd.open()
            
            async def toggle_ban_with_confirm():
                if not sel_email.value:
                    ui.notify("⚠️ 회원을 선택하세요", type="warning")
                    return
                
                target = sel_email.value
                user_obj = next(
                    (r["_user"] for r in rows_all if r["email"] == target),
                    None,
                )
                is_banned = bool(user_obj.get("is_banned")) if user_obj else False
                action = "해제" if is_banned else "차단"
                
                with ui.dialog() as cd, ui.card().classes(
                    "p-4 bg-[#1a1a2e] border border-red-500/40 "
                    "rounded-xl min-w-[400px]"
                ):
                    ui.label(f"🚫 회원 {action} 확인").classes(
                        "text-base font-bold text-red-300"
                    )
                    ui.label(f"대상: {target}").classes(
                        "text-sm text-white mt-2"
                    )
                    ui.label(
                        f"현재 상태: {'🚫차단됨' if is_banned else '✅정상'}"
                    ).classes("text-xs text-gray-300 mt-1")
                    ui.label(
                        f"변경: {action} 처리"
                    ).classes("text-sm text-amber-300 mt-1")
                    
                    reason_input = ui.input(
                        "사유 (액션 로그)",
                        placeholder=f"{action} 사유",
                    ).classes("w-full mt-2").props("outlined dense")
                    
                    with ui.row().classes("w-full justify-end gap-2 mt-3"):
                        ui.button("취소", on_click=cd.close).props("flat")
                        
                        async def do_ban():
                            cd.close()
                            reason = (reason_input.value or "").strip()
                            
                            def _do():
                                db_a = _get_db()
                                if not db_a:
                                    return False, "DB 연결 실패"
                                ok, msg = db_a.toggle_user_ban(target)
                                if ok:
                                    _log_admin_action(
                                        db_a, admin_email,
                                        "BAN_TOGGLE", target,
                                        {
                                            "from": is_banned,
                                            "to": not is_banned,
                                            "reason": reason,
                                        },
                                    )
                                return ok, msg
                            
                            ok, msg = await asyncio.to_thread(_do)
                            ui.notify(msg, type="positive" if ok else "negative")
                            if ok:
                                nonlocal users, rows_all
                                users[:] = db.get_all_users()
                                rows_all[:] = _build_rows(users)
                                _refresh_table()
                        
                        ui.button(
                            f"🚫 {action} 확정", on_click=do_ban,
                        ).props("color=red")
                cd.open()
            
            with ui.row().classes("w-full gap-2 mt-3"):
                ui.button(
                    "등급 적용", on_click=apply_role_with_confirm,
                ).props("color=primary").classes("flex-1")
                ui.button(
                    "🚫 차단/해제", on_click=toggle_ban_with_confirm,
                ).props("color=negative").classes("flex-1")

        # ─── 전체 이벤트 + 만료 강등 ───
        with ui.card().classes(
            "flex-1 min-w-[300px] p-4 bg-[#1a1a2e] "
            "border border-gray-700 rounded-xl"
        ):
            ui.label("🎉 일괄 작업").classes(
                "text-sm font-bold text-cyan-300 mb-2"
            )
            
            # [Step AV] 전원 Prime — 인원수/금액 미리보기 + 확인
            async def grant_trial_with_confirm():
                non_admin = sum(
                    1 for u in users
                    if str(u.get("role", "")).lower() != "admin"
                )
                est_value = non_admin * 12000  # 1인당 14일 약 12000원
                
                with ui.dialog() as cd, ui.card().classes(
                    "p-4 bg-[#1a1a2e] border border-amber-500/40 "
                    "rounded-xl min-w-[400px]"
                ):
                    ui.label("🎁 전원 14일 Prime 지급").classes(
                        "text-base font-bold text-amber-300"
                    )
                    ui.label(f"대상: 관리자 제외 {non_admin}명").classes(
                        "text-sm text-white mt-2"
                    )
                    ui.label(
                        f"예상 가치: 약 {est_value:,}원 "
                        f"({non_admin}명 × 14일)"
                    ).classes("text-sm text-amber-200 mt-1")
                    ui.label(
                        "⚠️ 이 작업은 매출에 영향을 줄 수 있습니다."
                    ).classes("text-xs text-red-300 mt-2")
                    
                    reason_input = ui.input(
                        "사유 (액션 로그)",
                        placeholder="이벤트/혜택 사유",
                    ).classes("w-full mt-2").props("outlined dense")
                    
                    with ui.row().classes("w-full justify-end gap-2 mt-3"):
                        ui.button("취소", on_click=cd.close).props("flat")
                        
                        async def do_grant():
                            cd.close()
                            reason = (reason_input.value or "").strip()
                            
                            def _do():
                                db_a = _get_db()
                                if not db_a:
                                    return False, "DB 연결 실패"
                                ok, msg = db_a.grant_all_users_trial(14)
                                if ok:
                                    _log_admin_action(
                                        db_a, admin_email,
                                        "BULK_TRIAL", "ALL",
                                        {
                                            "days": 14, "count": non_admin,
                                            "value": est_value,
                                            "reason": reason,
                                        },
                                    )
                                return ok, msg
                            
                            ok, msg = await asyncio.to_thread(_do)
                            ui.notify(
                                f"{'🎁 ' + msg if ok else '❌ ' + msg}",
                                type="positive" if ok else "negative",
                            )
                            if ok:
                                nonlocal users, rows_all
                                users[:] = db.get_all_users()
                                rows_all[:] = _build_rows(users)
                                _refresh_table()
                        
                        ui.button(
                            f"🎁 {non_admin}명에게 지급 확정",
                            on_click=do_grant,
                        ).props("color=positive")
                cd.open()
            
            ui.button(
                "🎁 전원 14일 Prime 지급",
                on_click=grant_trial_with_confirm,
            ).props("color=positive outlined").classes("w-full")
            
            ui.separator().classes("my-3")
            
            # [Step AV] 만료 강등 — 미리보기
            async def run_downgrade_with_preview():
                # 만료 대상 미리보기
                today = datetime.now(KST).strftime("%Y-%m-%d")
                expired_users = []
                for u in users:
                    if str(u.get("role", "")).lower() != "prime":
                        continue
                    exp = u.get("prime_expire_date")
                    if not exp:
                        continue
                    if str(exp)[:10] < today:
                        expired_users.append(u)
                
                if not expired_users:
                    ui.notify(
                        "✅ 만료된 회원 없음", type="positive",
                    )
                    return
                
                with ui.dialog() as cd, ui.card().classes(
                    "p-4 bg-[#1a1a2e] border border-amber-500/40 "
                    "rounded-xl min-w-[400px]"
                ):
                    ui.label("⏰ 만료 회원 강등 확인").classes(
                        "text-base font-bold text-amber-300"
                    )
                    ui.label(
                        f"강등 대상: {len(expired_users)}명"
                    ).classes("text-sm text-white mt-2")
                    
                    # 미리보기 (최대 5명)
                    for u in expired_users[:5]:
                        em = u.get("login_id") or u.get("id", "")
                        exp = u.get("prime_expire_date", "")
                        ui.label(
                            f"  • {em} (만료: {str(exp)[:10]})"
                        ).classes("text-xs text-gray-300")
                    if len(expired_users) > 5:
                        ui.label(
                            f"  ... 외 {len(expired_users) - 5}명"
                        ).classes("text-xs text-gray-500")
                    
                    with ui.row().classes("w-full justify-end gap-2 mt-3"):
                        ui.button("취소", on_click=cd.close).props("flat")
                        
                        async def do_downgrade():
                            cd.close()
                            
                            def _do():
                                from services.auth import downgrade_expired_users
                                count, details = downgrade_expired_users()
                                if count > 0:
                                    db_a = _get_db()
                                    if db_a:
                                        _log_admin_action(
                                            db_a, admin_email,
                                            "DOWNGRADE_EXPIRED", "ALL",
                                            {
                                                "count": count,
                                                "details": details[:5],
                                            },
                                        )
                                return count, details
                            
                            count, details = await asyncio.to_thread(_do)
                            if count > 0:
                                ui.notify(
                                    f"⏰ {count}명 강등 완료",
                                    type="warning",
                                )
                                nonlocal users, rows_all
                                users[:] = db.get_all_users()
                                rows_all[:] = _build_rows(users)
                                _refresh_table()
                            else:
                                ui.notify(
                                    "✅ 강등할 회원 없음",
                                    type="positive",
                                )
                        
                        ui.button(
                            f"⏰ {len(expired_users)}명 강등 실행",
                            on_click=do_downgrade,
                        ).props("color=amber")
                cd.open()
            
            ui.button(
                "⏰ 만료 회원 강등",
                on_click=run_downgrade_with_preview,
            ).props("color=warning outlined").classes("w-full")
            
            ui.separator().classes("my-3")
            
            # [Step AV] 위험 구역 — 빨간 카드로 강조
            with ui.card().classes(
                "w-full p-3 bg-red-900/20 border border-red-500/60 "
                "rounded-lg mt-2"
            ):
                ui.label("🚨 위험 구역").classes(
                    "text-sm font-bold text-red-300 mb-2"
                )
                
                async def reset_all_with_protection():
                    """[Step AV+AX] RESET 입력 + 카운트 미리보기 + 사유 + 범위 선택"""
                    non_admin = sum(
                        1 for u in users
                        if str(u.get("role", "")).lower() != "admin"
                    )
                    inquiries = []
                    try:
                        inquiries = db.get_all_inquiries() or []
                    except Exception:
                        pass
                    n_inquiries = len(inquiries)
                    
                    # [Step AX+AZ] payment/refund 카테고리 분리 카운트
                    n_payment = sum(
                        1 for q in inquiries
                        if (
                            q.get("category") in ("payment", "refund")
                            or _safe_str(q.get("title")).startswith("[💳 입금확인]")
                            or _safe_str(q.get("title")).startswith("[환불]")
                        )
                    )
                    n_general = n_inquiries - n_payment
                    
                    with ui.dialog() as cd, ui.card().classes(
                        "p-4 bg-[#1a1a2e] border border-red-500/60 "
                        "rounded-xl min-w-[480px]"
                    ):
                        ui.label("🚨 초기화 범위 선택").classes(
                            "text-base font-bold text-red-300"
                        )
                        ui.label(
                            "초기화 범위를 신중히 선택하세요. "
                            "복구할 수 없습니다."
                        ).classes("text-sm text-gray-200 mt-2")
                        
                        # [Step AX] 범위 선택 (3가지 옵션)
                        scope_select = ui.select(
                            options={
                                "users_only": (
                                    f"A. 회원만 초기화 ({non_admin}명, 문의 보존)"
                                ),
                                "users_general": (
                                    f"B. 회원 + 일반 문의 "
                                    f"({non_admin}명 + {n_general}건, "
                                    f"결제/환불 문의 보존) ⭐ 권장"
                                ),
                                "users_all": (
                                    f"C. 회원 + 모든 문의 "
                                    f"({non_admin}명 + {n_inquiries}건, "
                                    f"분쟁 기록 사라짐) 🚨"
                                ),
                            },
                            value="users_general",
                            label="초기화 범위",
                        ).classes("w-full mt-2").props("outlined dense")
                        
                        # 범위별 미리보기 카드
                        preview_card = ui.card().classes(
                            "w-full p-2 mt-2 bg-[#0a0a14] "
                            "border border-amber-500/40 rounded-lg"
                        )
                        
                        def update_preview():
                            preview_card.clear()
                            scope = scope_select.value
                            with preview_card:
                                if scope == "users_only":
                                    ui.label(
                                        f"  • 회원 삭제: {non_admin}명"
                                    ).classes("text-xs text-red-200")
                                    ui.label(
                                        f"  • 문의 보존: {n_inquiries}건"
                                    ).classes("text-xs text-emerald-300")
                                    ui.label(
                                        "  💡 분쟁 대응 기록 모두 보존"
                                    ).classes("text-xs text-amber-200")
                                elif scope == "users_general":
                                    ui.label(
                                        f"  • 회원 삭제: {non_admin}명"
                                    ).classes("text-xs text-red-200")
                                    ui.label(
                                        f"  • 일반 문의 삭제: {n_general}건"
                                    ).classes("text-xs text-red-200")
                                    ui.label(
                                        f"  • 결제/환불 문의 보존: {n_payment}건 ⭐"
                                    ).classes("text-xs text-emerald-300")
                                else:  # users_all
                                    ui.label(
                                        f"  • 회원 삭제: {non_admin}명"
                                    ).classes("text-xs text-red-200")
                                    ui.label(
                                        f"  • 모든 문의 삭제: {n_inquiries}건"
                                    ).classes("text-xs text-red-200")
                                    ui.label(
                                        f"  ⚠️ 결제/환불 문의 {n_payment}건도 삭제 — 분쟁 위험"
                                    ).classes("text-xs text-red-300 font-bold")
                                ui.label(
                                    "  • 복구: 불가능"
                                ).classes("text-xs text-red-300 font-bold")
                        
                        scope_select.on(
                            "update:model-value",
                            lambda _: update_preview(),
                        )
                        update_preview()
                        
                        ui.label(
                            "💡 백업이 필요하면 먼저 CSV 다운로드를 받으세요."
                        ).classes("text-xs text-amber-200 mt-2 italic")
                        
                        ui.label(
                            "계속하려면 'RESET'을 정확히 입력하세요:"
                        ).classes("text-xs text-red-300 mt-2 font-bold")
                        reset_input = ui.input(
                            placeholder="RESET",
                        ).classes("w-full mt-1").props("outlined dense")
                        
                        reason_input = ui.input(
                            "사유 (필수, 액션 로그)",
                            placeholder="예: 베타 테스트 종료, 재시작",
                        ).classes("w-full mt-2").props("outlined dense")
                        
                        with ui.row().classes("w-full justify-end gap-2 mt-3"):
                            ui.button("취소", on_click=cd.close).props("flat")
                            
                            async def do_reset():
                                typed = (reset_input.value or "").strip()
                                reason = (reason_input.value or "").strip()
                                scope = scope_select.value or "users_general"
                                
                                if typed != "RESET":
                                    ui.notify(
                                        "⚠️ 'RESET'을 정확히 입력해야 합니다",
                                        type="warning",
                                    )
                                    return
                                if not reason:
                                    ui.notify(
                                        "⚠️ 사유 입력 필수 (감사 로그)",
                                        type="warning",
                                    )
                                    return
                                
                                cd.close()
                                ui.notify("⏳ 초기화 중...", type="info")
                                
                                def _do():
                                    db_a = _get_db()
                                    if not db_a:
                                        return False, "DB 연결 실패"
                                    try:
                                        # [Step AX] 범위별 삭제 카운트
                                        if scope == "users_only":
                                            inquiries_deleted = 0
                                            inquiry_action = "preserved_all"
                                        elif scope == "users_general":
                                            inquiries_deleted = n_general
                                            inquiry_action = "deleted_general_only"
                                        else:  # users_all
                                            inquiries_deleted = n_inquiries
                                            inquiry_action = "deleted_all"
                                        
                                        # 액션 로그 먼저 (삭제 전!)
                                        _log_admin_action(
                                            db_a, admin_email,
                                            "RESET_ALL", "ALL",
                                            {
                                                "scope": scope,
                                                "users_deleted": non_admin,
                                                "inquiries_deleted": inquiries_deleted,
                                                "inquiry_action": inquiry_action,
                                                "payment_preserved": n_payment if scope != "users_all" else 0,
                                                "reason": reason,
                                            },
                                        )
                                        # 회원 삭제 (모든 범위 공통)
                                        db_a._exec_sqlite(
                                            "DELETE FROM users "
                                            "WHERE role != 'admin'"
                                        )
                                        # [Step AX] 범위별 문의 삭제
                                        if scope == "users_general":
                                            # 결제/환불 보존, 일반만 삭제
                                            db_a._exec_sqlite(
                                                "DELETE FROM inquiries "
                                                "WHERE NOT (title LIKE '[💳 입금확인]%' "
                                                "OR title LIKE '[환불]%')"
                                            )
                                        elif scope == "users_all":
                                            # 모두 삭제
                                            db_a._exec_sqlite(
                                                "DELETE FROM inquiries"
                                            )
                                        # users_only는 inquiries 손대지 않음
                                        
                                        # [Step AW+AX] Gist 표준 파일명 (TABLE_TO_GIST_FILE)
                                        try:
                                            from db_utils import (
                                                USER_DB_FILE, INQUIRY_DB_FILE,
                                            )
                                        except ImportError:
                                            USER_DB_FILE = "users_db.json"
                                            INQUIRY_DB_FILE = "inquiries_db.json"
                                        u_ok = db_a._do_gist_upload(
                                            "users", USER_DB_FILE,
                                        )
                                        i_ok = True  # users_only면 동기화 불필요
                                        if scope != "users_only":
                                            i_ok = db_a._do_gist_upload(
                                                "inquiries", INQUIRY_DB_FILE,
                                            )
                                        if u_ok and i_ok:
                                            return True, (
                                                f"🔄 초기화 완료 "
                                                f"(범위: {scope})"
                                            )
                                        else:
                                            return True, (
                                                "⚠️ SQLite 삭제 완료, "
                                                "Gist 일부 실패"
                                            )
                                    except Exception as ex:
                                        return False, f"❌ 오류: {ex}"
                                
                                ok, msg = await asyncio.to_thread(_do)
                                ui.notify(
                                    msg,
                                    type="positive" if ok else "negative",
                                )
                                if ok:
                                    nonlocal users, rows_all
                                    users[:] = db.get_all_users()
                                    rows_all[:] = _build_rows(users)
                                    _refresh_table()
                            
                            ui.button(
                                "🚨 초기화 확정",
                                on_click=do_reset,
                            ).props("color=red")
                    cd.open()
                
                ui.button(
                    "🔄 초기화 (범위 선택)",
                    on_click=reset_all_with_protection,
                ).props("flat color=red").classes("w-full").tooltip(
                    "범위 선택 + RESET 입력 + 사유 필수"
                )
                ui.label(
                    "⚠️ 3가지 범위 / RESET 입력 / 사유 필수"
                ).classes("text-[10px] text-red-300 mt-1 text-center")

        # ─── 입금확인 + 액션 로그 보기 ───
        with ui.card().classes(
            "flex-1 min-w-[300px] p-4 bg-[#1a1a2e] "
            "border border-gray-700 rounded-xl"
        ):
            ui.label("💳 입금확인 / 📋 액션 로그").classes(
                "text-sm font-bold text-cyan-300 mb-2"
            )
            
            # 입금확인 대기
            payment_list = ui.column().classes("w-full")
            
            # [Step AW] 입금확인 처리 완료된 inquiry_id 추적
            # admin_actions 테이블에 PAYMENT_CONFIRMED 기록이 있으면 처리됨
            # → 삭제하지 않고 보존 (분쟁 대응)
            def _is_payment_confirmed(req: dict) -> bool:
                """[Step AW+AX] PAYMENT_CONFIRMED 액션 로그 존재 여부.
                
                [Step AX 수정] 최근 200개 스캔 → 직접 SQL 조회 (전체 범위)
                """
                try:
                    inquiry_id = req.get("created_at", "") or req.get("id", "")
                    if not inquiry_id:
                        return False
                    # [Step AX] _get_payment_confirmed_action으로 직접 조회
                    action = _get_payment_confirmed_action(db, inquiry_id)
                    return bool(action)
                except Exception:
                    return False
            
            async def _confirm_payment_request(req):
                """[Step AW S급 수정] 입금확인 처리 — 삭제 금지, 액션 로그로 보존
                
                이전 버그: save_inquiries(items)로 문의 삭제
                → 분쟁 시 기록 없음
                
                현재: admin_actions 테이블에 PAYMENT_CONFIRMED 기록
                → inquiries 보존, 처리 여부는 액션 로그로 판정
                """
                # 확인 다이얼로그
                with ui.dialog() as cd, ui.card().classes(
                    "p-4 bg-[#1a1a2e] border border-emerald-500/40 "
                    "rounded-xl min-w-[400px]"
                ):
                    ui.label("💳 입금확인 처리").classes(
                        "text-base font-bold text-emerald-300"
                    )
                    ui.label(
                        f"📌 {_safe_str(req.get('title'))[:60]}"
                    ).classes("text-sm text-white mt-2")
                    ui.label(
                        _safe_str(req.get("content"))[:200]
                    ).classes("text-xs text-gray-300 mt-1")
                    ui.label(
                        f"🕐 {_to_kst_str(req.get('created_at'))}"
                    ).classes("text-xs text-gray-500 mt-1")
                    
                    ui.separator().classes("my-2")
                    ui.label(
                        "💡 입금확인 후 해당 회원 등급을 PRIME으로 변경하세요. "
                        "처리 기록은 admin_actions에 영구 보존됩니다 (분쟁 대응)."
                    ).classes("text-xs text-amber-200 italic")
                    
                    reply_input = ui.input(
                        "처리 메모 (액션 로그)",
                        placeholder="예: 토스 30,000원 입금 확인, PRIME 30일 적용",
                    ).classes("w-full mt-2").props("outlined dense")
                    
                    with ui.row().classes("w-full justify-end gap-2 mt-3"):
                        ui.button("취소", on_click=cd.close).props("flat")
                        
                        async def do_confirm():
                            cd.close()
                            reply = (reply_input.value or "").strip()
                            
                            def _do():
                                db_d = _get_db()
                                if not db_d:
                                    return False
                                # [Step AW] 삭제하지 않고 admin_actions에만 기록
                                inquiry_id = (
                                    req.get("created_at", "")
                                    or req.get("id", "")
                                )
                                target_email = (
                                    req.get("id", "") or req.get("login_id", "")
                                )
                                _log_admin_action(
                                    db_d, admin_email,
                                    "PAYMENT_CONFIRMED",
                                    target_email,
                                    {
                                        "inquiry_id": inquiry_id,
                                        "title": _safe_str(req.get("title"))[:80],
                                        "admin_reply": reply,
                                        "category": req.get(
                                            "category", "payment"
                                        ),
                                    },
                                )
                                return True
                            
                            ok = await asyncio.to_thread(_do)
                            if ok:
                                ui.notify(
                                    "✅ 입금확인 기록 보존 완료 — "
                                    "분쟁 시 admin_actions에서 조회 가능",
                                    type="positive",
                                )
                                _load_payment_requests()
                            else:
                                ui.notify("❌ 처리 실패", type="negative")
                        
                        ui.button(
                            "✅ 입금확인 처리",
                            on_click=do_confirm,
                        ).props("color=positive")
                cd.open()

            def _load_payment_requests():
                payment_list.clear()
                db_p = _get_db()
                if not db_p:
                    return
                inquiries = db_p.get_all_inquiries()
                # [Step AW+AZ] payment 탐지 — None 안전
                pay_reqs_all = [
                    q for q in inquiries
                    if (
                        q.get("category") == "payment"
                        or _safe_str(q.get("title")).startswith("[💳 입금확인]")
                    )
                ]
                # [Step AW] 처리 완료/미처리 분리 (보존 + 가독성)
                pending = [
                    q for q in pay_reqs_all
                    if not _is_payment_confirmed(q)
                ]
                completed = [
                    q for q in pay_reqs_all
                    if _is_payment_confirmed(q)
                ]
                
                with payment_list:
                    # 대기 중
                    if not pending:
                        ui.label(
                            "✅ 대기 중인 입금확인 요청 없음"
                        ).classes("text-xs text-gray-500 italic")
                    else:
                        ui.label(
                            f"⏳ 대기 중 ({len(pending)}건)"
                        ).classes("text-xs text-amber-300 font-bold mb-1")
                        for req in reversed(pending[-10:]):
                            with ui.card().classes(
                                "w-full p-2 mb-1 bg-[#0f3460] "
                                "border border-amber-500/40 rounded-lg"
                            ):
                                with ui.row().classes(
                                    "w-full justify-between items-center"
                                ):
                                    ui.label(
                                        f"📌 {_safe_str(req.get('title'))[:40]}"
                                    ).classes("text-xs text-white")
                                    ui.button(
                                        "✅ 처리",
                                        on_click=lambda r=req: (
                                            _confirm_payment_request(r)
                                        ),
                                    ).props("flat dense size=sm color=green")
                                ui.label(
                                    _safe_str(req.get("content"))[:80]
                                ).classes("text-xs text-gray-300 mt-1")
                                ui.label(
                                    f"🕐 {_to_kst_str(req.get('created_at'))}"
                                ).classes("text-xs text-gray-500 mt-1")
                    
                    # 처리 완료 (보존 표시)
                    if completed:
                        ui.label(
                            f"✅ 처리 완료 ({len(completed)}건, 보존)"
                        ).classes("text-xs text-emerald-300 font-bold mt-3 mb-1")
                        for req in reversed(completed[-5:]):
                            with ui.card().classes(
                                "w-full p-2 mb-1 bg-[#0a0a14] "
                                "border border-emerald-700/30 rounded-lg "
                                "opacity-70"
                            ):
                                ui.label(
                                    f"✅ {_safe_str(req.get('title'))[:40]}"
                                ).classes("text-xs text-emerald-300")
                                ui.label(
                                    f"🕐 {_to_kst_str(req.get('created_at'))}"
                                ).classes("text-xs text-gray-500")
            
            _load_payment_requests()
            
            with ui.row().classes("gap-2 mt-1"):
                ui.button(
                    "🔄 새로고침",
                    on_click=_load_payment_requests,
                ).props("flat dense size=sm color=blue")
                ui.button(
                    "📋 전체 액션 로그",
                    on_click=lambda: _open_actions_log(),
                ).props("flat dense size=sm color=purple").tooltip(
                    "최근 50건 관리자 작업 이력"
                )

    # 액션 로그 전체 보기 모달
    def _open_actions_log():
        with ui.dialog() as dialog, ui.card().classes(
            "p-4 bg-[#1a1a2e] border border-gray-700 rounded-xl "
            "min-w-[600px] max-w-[900px]"
        ):
            ui.label("📋 관리자 작업 이력").classes(
                "text-lg font-bold text-white"
            )
            ui.label(
                "최근 50건 — 모든 위험 작업이 기록됩니다 (감사 추적)"
            ).classes("text-xs text-gray-400 mt-1")
            ui.separator().classes("my-2")
            
            actions = _get_admin_actions(db, limit=50)
            if not actions:
                ui.label("기록 없음").classes(
                    "text-gray-400 italic p-4"
                )
            else:
                for a in actions:
                    try:
                        details = json.loads(a.get("details", "{}") or "{}")
                    except Exception:
                        details = {}
                    
                    # 위험도 색상
                    action_type = a.get("action_type", "")
                    color = "amber"
                    if action_type in ("RESET_ALL", "BULK_TRIAL"):
                        color = "red"
                    elif action_type in ("BAN_TOGGLE", "DOWNGRADE_EXPIRED"):
                        color = "orange"
                    
                    with ui.card().classes(
                        f"w-full p-2 bg-[#0a0a14] "
                        f"border border-{color}-700/40 rounded-lg mb-1"
                    ):
                        with ui.row().classes(
                            "w-full justify-between items-center"
                        ):
                            ui.label(
                                f"⚡ {action_type}"
                            ).classes(f"text-sm text-{color}-300 font-bold")
                            ui.label(
                                _safe_str(a.get("timestamp"))[:19]
                            ).classes("text-xs text-gray-500")
                        with ui.row().classes(
                            "w-full justify-between items-center mt-1"
                        ):
                            ui.label(
                                f"by {a.get('admin_email', '')}"
                            ).classes("text-xs text-gray-400")
                            ui.label(
                                f"→ {a.get('target_email', '') or 'ALL'}"
                            ).classes("text-xs text-cyan-300")
                        if details:
                            detail_str = ", ".join(
                                f"{k}={v}"
                                for k, v in list(details.items())[:5]
                            )
                            ui.label(detail_str).classes(
                                "text-xs text-gray-300 mt-1"
                            )
            
            with ui.row().classes("w-full justify-end mt-3"):
                ui.button("닫기", on_click=dialog.close).props("flat")
        
        dialog.open()

    # ─── 필터 변경 핸들러 ───
    def _on_search(e):
        state["search"] = (e.value or "").strip()
        _refresh_table()

    def _on_access(e):
        state["access"] = e.value or "전체"
        _refresh_table()

    def _on_role(e):
        state["role"] = e.value or "전체"
        _refresh_table()

    def _on_period(e):
        state["period"] = e.value or "전체"
        _refresh_table()

    def _on_banned(e):
        state["banned"] = e.value or "전체"
        _refresh_table()

    f_search.on("update:model-value", _on_search)
    f_access.on("update:model-value", _on_access)
    f_role.on("update:model-value", _on_role)
    f_period.on("update:model-value", _on_period)
    f_banned.on("update:model-value", _on_banned)

    # ─── 초기 렌더 ───
    _refresh_table()


# ═══════════════════════════════════════════════════
#  헬퍼
# ═══════════════════════════════════════════════════
def _stat_card(title, value, color="white", tooltip: str = ""):
    """[Step AV] 통계 카드"""
    card = ui.card().classes(
        "p-3 bg-[#1a1a2e] border border-gray-700 rounded-xl"
    )
    with card:
        ui.label(title).classes("text-xs text-gray-400")
        ui.label(str(value)).classes(
            "text-lg font-bold mt-1"
        ).style(f"color:{color}")
    if tooltip:
        card.tooltip(tooltip)


def _info_row(label, value):
    """[Step AV] 회원 상세 정보 행"""
    with ui.column().classes("gap-0"):
        ui.label(label).classes("text-xs text-gray-400")
        ui.label(str(value)).classes("text-sm text-white")
