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
        {"name": "expire", "label": "구독만료", "field": "expire"},
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
            "expire": _to_kst_str(u.get("prime_expire_date"), "%Y-%m-%d") if u.get("prime_expire_date") else "-",
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
            sel_days = ui.select(
                {0: "만료일 없음", 7: "7일", 30: "30일 (1개월)", 90: "90일 (3개월)", 180: "180일 (6개월)", 365: "365일 (1년)"},
                label="구독 기간", value=30,
            ).classes("w-full")

            async def apply_role():
                if sel_email.value:
                    db = _get_db()
                    if db:
                        days = sel_days.value or 0
                        if days > 0 and sel_role.value in ("pro", "prime"):
                            from datetime import datetime as dt2
                            expire = (dt2.now() + timedelta(days=days)).strftime("%Y-%m-%d")
                            db.update_user_subscription(sel_email.value, sel_role.value, expire)
                            ui.notify(f"✅ {sel_email.value} → {sel_role.value.upper()} (만료: {expire})")
                        else:
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

            def _reset_all_members():
                db_r = _get_db()
                if db_r:
                    try:
                        db_r._exec_sqlite("DELETE FROM users WHERE role != 'admin'")
                        db_r._exec_sqlite("DELETE FROM inquiries")
                        db_r._mark_gist_dirty("users")
                        db_r._mark_gist_dirty("inquiries")
                        ui.notify("🔄 전체 회원 초기화 완료 (관리자 제외)")
                    except Exception as ex:
                        ui.notify(f"❌ 오류: {ex}")

            ui.button("🔄 전체 회원 초기화 (관리자 제외)", on_click=lambda: _reset_all_members()).props("color=red").tooltip("관리자 제외 전체 삭제")
            ui.button("🎁 전원 7일 Prime 지급", on_click=grant_trial).props("color=positive")

        # ── 입금확인 요청 목록 ──
        with ui.column().classes("flex-1"):
            ui.label("💳 입금확인 대기").classes("text-white font-bold mb-2")
            ui.label("멤버십 탭에서 요청된 입금확인 내역").classes("text-gray-400 text-sm mb-2")

            payment_list = ui.column().classes("w-full")

            def _load_payment_requests():
                payment_list.clear()
                db_p = _get_db()
                if not db_p:
                    return
                inquiries = db_p.get_all_inquiries()
                pay_reqs = [q for q in inquiries if q.get("title", "").startswith("[💳 입금확인]")]
                with payment_list:
                    if not pay_reqs:
                        ui.label("대기 중인 요청 없음").classes("text-gray-500 text-sm")
                        return
                    for req in reversed(pay_reqs[-10:]):
                        with ui.card().classes("w-full p-3 mb-2 bg-[#0f3460] border border-blue-700 rounded-lg"):
                            ui.label(f"📌 {req.get('title', '')}").classes("text-white font-bold text-sm")
                            ui.label(req.get("content", "")).classes("text-gray-300 text-xs mt-1 whitespace-pre-line")
                            ui.label(f"🕐 {_to_kst_str(req.get('created_at'))}").classes("text-xs text-gray-500 mt-1")

            _load_payment_requests()
            ui.button("🔄 새로고침", on_click=_load_payment_requests).props("flat dense size=sm color=blue")
