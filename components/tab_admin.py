# -*- coding: utf-8 -*-
"""
tab_admin.py — 👑 관리자 패널 (NiceGUI)
═══════════════════════════════════════════════════
[v1.0] Streamlit → NiceGUI 이식

기능:
  - 회원 목록 테이블 (최신 접속순)
  - 개별 회원: 등급 변경, 차단/해제
  - 전체 이벤트: 7일 Prime 무료 지급
  - 구독 관리: 만료일 설정

보안:
  - auth_status == "admin" 일 때만 전체 UI 렌더
"""
import logging
from datetime import datetime

import pandas as pd
from nicegui import ui, run

from db_utils import get_db

_logger = logging.getLogger(__name__)


def _to_kst_display(val) -> str:
    """UTC/datetime → KST 표시용"""
    try:
        from datetime import timedelta
        if val is None:
            return "-"
        if isinstance(val, datetime):
            kst = val + timedelta(hours=9)
            return kst.strftime("%Y-%m-%d %H:%M")
        s = str(val)[:19]
        if s in ("None", "NaT", "nan", ""):
            return "-"
        dt = datetime.strptime(s.replace("T", " "), "%Y-%m-%d %H:%M:%S")
        kst = dt + timedelta(hours=9)
        return kst.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(val)[:16] if val else "-"


# ═══════════════════════════════════════════════════
#  메인 렌더
# ═══════════════════════════════════════════════════

def render_tab_admin(auth_status: str):
    """Tab 8: 관리자 패널

    Args:
        auth_status: "guest" | "free" | "prime" | "pro" | "admin"
    """
    ui.label("👑 회원 관리 (Admin)").classes("text-2xl font-bold mb-4")

    if auth_status != "admin":
        with ui.card().classes("w-full p-8 text-center bg-red-50"):
            ui.icon("block", size="xl").classes("text-red-400")
            ui.label("🚫 관리자 권한이 필요합니다.").classes("text-lg text-red-600 mt-4")
        return

    # ── 전체 UI 컨테이너 (갱신용) ──
    admin_container = ui.column().classes("w-full")

    def refresh_admin():
        admin_container.clear()
        with admin_container:
            _render_admin_panel()

    refresh_admin()


def _render_admin_panel():
    """관리자 패널 내부 (갱신 가능)"""
    db = get_db()
    users = db.get_all_users()

    # ── 상단 요약 ──
    if users:
        with ui.card().classes("w-full p-4 bg-gray-50 mb-4"):
            ui.label(f"👥 총 가입자: {len(users)}명").classes("text-lg font-bold")
    else:
        ui.label("등록된 회원이 없습니다.").classes("text-gray-500")
        return

    # ── 회원 테이블 ──
    _render_user_table(users)

    ui.separator().classes("my-6")

    # ── 하단: 액션 패널 ──
    with ui.row().classes("w-full gap-6 flex-wrap"):
        with ui.column().classes("flex-1 min-w-[300px]"):
            _render_individual_control(users)

        with ui.column().classes("flex-1 min-w-[300px]"):
            _render_event_panel()


# ═══════════════════════════════════════════════════
#  회원 테이블
# ═══════════════════════════════════════════════════

def _render_user_table(users: list):
    """회원 목록 테이블"""
    columns = [
        {"name": "email", "label": "이메일", "field": "email", "align": "left", "sortable": True},
        {"name": "nick", "label": "닉네임", "field": "nick", "align": "left"},
        {"name": "role", "label": "권한", "field": "role", "align": "center", "sortable": True},
        {"name": "status", "label": "상태", "field": "status", "align": "center"},
        {"name": "join", "label": "가입일", "field": "join", "align": "center", "sortable": True},
        {"name": "last", "label": "최근접속", "field": "last", "align": "center", "sortable": True},
        {"name": "expire", "label": "만료일", "field": "expire", "align": "center"},
    ]

    rows = []
    for u in users:
        is_banned = u.get("is_banned", False)
        role = u.get("role", "free")
        expire_raw = u.get("prime_expire_date")

        rows.append({
            "email": u.get("login_id", u.get("id", "")),
            "nick": u.get("nickname", ""),
            "role": role.upper(),
            "status": "🚫차단" if is_banned else "✅",
            "join": _to_kst_display(u.get("join_date")),
            "last": _to_kst_display(u.get("last_login")),
            "expire": _to_kst_display(expire_raw).split(" ")[0] if expire_raw else "-",
        })

    # 최신 접속순
    rows.sort(key=lambda r: r.get("last", ""), reverse=True)

    ui.table(
        columns=columns,
        rows=rows,
        row_key="email",
    ).classes("w-full").props("dense flat")


# ═══════════════════════════════════════════════════
#  개별 회원 제어
# ═══════════════════════════════════════════════════

def _render_individual_control(users: list):
    """개별 회원 등급 변경 / 차단"""
    ui.label("🛠️ 개별 회원 제어").classes("text-lg font-semibold mb-2")

    emails = [u.get("login_id", u.get("id", "")) for u in users]
    sel_email = ui.select("회원 선택", options=emails, value=emails[0] if emails else None).classes("w-full")

    with ui.row().classes("gap-4 mt-3"):
        # ── 등급 변경 ──
        with ui.column():
            sel_role = ui.select(
                "등급 변경",
                options=["free", "pro", "prime", "admin"],
                value="free",
            ).classes("w-40")

            async def apply_role():
                if not sel_email.value:
                    return
                db = get_db()
                await run.io_bound(db.update_user_role, sel_email.value, sel_role.value)
                ui.notify(f"✅ {sel_email.value} → {sel_role.value} 변경 완료", type="positive")

            ui.button("등급 적용", on_click=apply_role, color="primary").props("dense")

        # ── 차단/해제 ──
        with ui.column():
            async def toggle_ban():
                if not sel_email.value:
                    return
                db = get_db()
                ok, msg = await run.io_bound(db.toggle_user_ban, sel_email.value)
                if ok:
                    ui.notify(msg, type="info")
                else:
                    ui.notify(msg, type="negative")

            ui.button("🚫 차단/해제", on_click=toggle_ban, color="red").props("dense")

    # ── 구독 설정 ──
    ui.separator().classes("my-4")
    ui.label("📅 구독 만료일 설정").classes("text-sm font-medium")

    with ui.row().classes("gap-2 items-end"):
        inp_expire = ui.input("만료일 (YYYY-MM-DD)", placeholder="2026-03-31").classes("w-48")

        async def set_subscription():
            if not sel_email.value or not inp_expire.value:
                ui.notify("회원과 만료일을 입력하세요", type="warning")
                return
            db = get_db()
            try:
                await run.io_bound(
                    db.update_user_subscription,
                    sel_email.value, sel_role.value, inp_expire.value,
                )
                ui.notify(f"✅ {sel_email.value} 구독 설정 완료", type="positive")
            except Exception as e:
                ui.notify(f"실패: {e}", type="negative")

        ui.button("적용", on_click=set_subscription).props("dense color=teal")


# ═══════════════════════════════════════════════════
#  전체 이벤트
# ═══════════════════════════════════════════════════

def _render_event_panel():
    """전체 회원 이벤트 (7일 Prime 무료 등)"""
    ui.label("🎉 전체 이벤트").classes("text-lg font-semibold mb-2")
    ui.label("관리자를 제외한 모든 회원에게 체험권을 일괄 지급합니다.").classes(
        "text-sm text-gray-500 mb-3"
    )

    async def grant_trial():
        db = get_db()
        ok, msg = await run.io_bound(db.grant_all_users_trial, 7)
        if ok:
            ui.notify(f"🎁 {msg}", type="positive")
        else:
            ui.notify(f"실패: {msg}", type="negative")

    ui.button(
        "🎁 전원 7일 Prime 무료 지급",
        on_click=grant_trial,
        color="primary",
    ).classes("text-lg px-6 py-2")
