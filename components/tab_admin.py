# -*- coding: utf-8 -*-
"""
tab_admin.py — 👑 회원 관리 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
관리자 전용: 회원 목록, 등급 변경, 차단/해제, 전체 이벤트
"""
import logging
from datetime import datetime, timedelta, timezone

from nicegui import ui

_logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))


def _get_db():
    try:
        from db_utils import get_db
        return get_db()
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


def render_tab_admin():
    """Tab 8: 회원 관리 (Admin)"""
    ui.label("👑 회원 관리").classes("text-2xl font-bold mb-4 text-white")

    db = _get_db()
    if not db:
        ui.label("❌ DB 연결 실패").classes("text-red-400")
        return

    users = db.get_all_users()
    if not users:
        ui.label("등록된 회원 없음").classes("text-gray-400")
        return

    ui.label(f"👥 총 가입자: {len(users)}명").classes("text-white mb-4")

    columns = [
        {"name": "email", "label": "이메일", "field": "email", "align": "left"},
        {"name": "nick", "label": "닉네임", "field": "nick"},
        {"name": "role", "label": "권한", "field": "role"},
        {"name": "status", "label": "상태", "field": "status"},
        {"name": "joined", "label": "가입일", "field": "joined"},
        {"name": "last", "label": "최근접속", "field": "last"},
    ]
    rows = []
    for u in users:
        rows.append({
            "email": u.get("login_id") or u.get("id", ""),
            "nick": u.get("nickname", ""),
            "role": u.get("role", "free").upper(),
            "status": "🚫차단" if u.get("is_banned") else "✅",
            "joined": _to_kst_str(u.get("join_date"), "%Y-%m-%d"),
            "last": _to_kst_str(u.get("last_login")),
        })

    ui.table(columns=columns, rows=rows, row_key="email",
             pagination={"rowsPerPage": 20}).classes("w-full").props("dense dark flat bordered")

    # 관리자 액션
    ui.separator().classes("my-4")
    with ui.row().classes("w-full gap-8 flex-wrap"):
        with ui.column().classes("flex-1"):
            ui.label("🛠️ 개별 회원 제어").classes("text-white font-bold mb-2")
            emails = [r["email"] for r in rows]
            sel_email = ui.select(emails, label="회원 선택").classes("w-full")
            sel_role = ui.select(["free", "pro", "prime", "admin"], label="등급 변경", value="free").classes("w-full")

            async def apply_role():
                if sel_email.value:
                    db = _get_db()
                    if db:
                        ok = db.update_user_role(sel_email.value, sel_role.value)
                        ui.notify(f"{'✅ 변경 완료' if ok else '❌ 실패'}")
                    else:
                        ui.notify("DB 연결 실패", type="negative")

            async def toggle_ban():
                if sel_email.value:
                    db = _get_db()
                    if db:
                        ok, msg = db.toggle_user_ban(sel_email.value)
                        ui.notify(msg)
                    else:
                        ui.notify("DB 연결 실패", type="negative")

            with ui.row().classes("gap-2 mt-2"):
                ui.button("등급 적용", on_click=apply_role).props("color=primary")
                ui.button("🚫 차단/해제", on_click=toggle_ban).props("color=negative")

        with ui.column().classes("flex-1"):
            ui.label("🎉 전체 이벤트").classes("text-white font-bold mb-2")
            ui.label("전 회원에게 체험권을 지급합니다.").classes("text-gray-400 text-sm mb-2")

            async def grant_trial():
                db = _get_db()
                if db:
                    ok, msg = db.grant_all_users_trial(7)
                    ui.notify(f"{'🎁 ' + msg if ok else '❌ ' + msg}")
                else:
                    ui.notify("DB 연결 실패", type="negative")

            ui.button("🎁 전원 7일 Prime 지급", on_click=grant_trial).props("color=positive")
