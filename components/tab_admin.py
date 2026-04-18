# -*- coding: utf-8 -*-
"""
tab_admin.py — 👑 회원 관리 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
관리자 전용: 회원 목록, 등급 변경, 차단/해제, 전체 이벤트, 회원 초기화
모든 네트워크/DB 호출은 asyncio.to_thread로 비동기 처리 (connection lost 방지)
"""
import asyncio
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

    db = _get_db()
    if not db:
        ui.label("❌ DB 연결 실패").classes("text-red-400")
        return

    users = db.get_all_users()

    if not users:
        ui.label("👥 총 가입자: 0명").classes("text-white mb-4")
        ui.label("등록된 회원 없음").classes("text-gray-400")
        return

    # ── 회원 데이터 빌드 (실제 접근 상태 판정) ──
    from services.auth import compute_access_status

    columns = [
        {"name": "email", "label": "이메일", "field": "email", "align": "left"},
        {"name": "nick", "label": "닉네임", "field": "nick"},
        {"name": "role", "label": "DB권한", "field": "role"},
        {"name": "access", "label": "실제상태", "field": "access"},
        {"name": "expire", "label": "구독만료", "field": "expire"},
        {"name": "status", "label": "차단", "field": "status"},
        {"name": "joined", "label": "가입일", "field": "joined"},
        {"name": "last", "label": "최근접속", "field": "last"},
    ]
    rows = []
    for u in users:
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
            "expire": _to_kst_str(u.get("prime_expire_date"), "%Y-%m-%d") if u.get("prime_expire_date") else "-",
            "status": "🚫차단" if u.get("is_banned") else "✅",
            "joined": _to_kst_str(u.get("join_date"), "%Y-%m-%d"),
            "last": _to_kst_str(u.get("last_login")),
        })

    # ── 상단 요약 ──
    ui.label(f"👑 회원 관리").classes("text-2xl font-bold mb-2 text-white")
    ui.label(f"👥 총 가입자: {len(users)}명").classes("text-white mb-2")
    _active = sum(1 for r in rows if r.get("access") == "✅활성")
    _expired = sum(1 for r in rows if r.get("access") == "❌만료")
    _free = sum(1 for r in rows if r.get("access") == "⚪무료")
    _admin = sum(1 for r in rows if r.get("access") == "🔑관리자")
    with ui.row().classes("gap-4 mb-4"):
        ui.badge(f"✅ 활성 {_active}", color="#10B981").classes("text-sm px-3 py-1")
        ui.badge(f"❌ 만료 {_expired}", color="#EF4444").classes("text-sm px-3 py-1")
        ui.badge(f"⚪ 무료 {_free}", color="#6B7280").classes("text-sm px-3 py-1")
        ui.badge(f"🔑 관리자 {_admin}", color="#3B82F6").classes("text-sm px-3 py-1")

    if rows:
        ui.table(columns=columns, rows=rows, row_key="email",
                 pagination={"rowsPerPage": 20}).classes("w-full").props("dense dark flat bordered")

    # ── 관리자 액션 ──
    ui.separator().classes("my-4")
    with ui.row().classes("w-full gap-8 flex-wrap"):

        # ── 개별 회원 제어 ──
        with ui.column().classes("flex-1"):
            ui.label("🛠️ 개별 회원 제어").classes("text-white font-bold mb-2")
            emails = [r["email"] for r in rows]
            sel_email = ui.select(emails, label="회원 선택").classes("w-full")
            sel_role = ui.select(["free", "prime", "admin"], label="등급 변경", value="free").classes("w-full")
            sel_days = ui.select(
                {0: "만료일 없음", 7: "7일", 14: "14일", 30: "30일 (1개월)", 90: "90일 (3개월)", 180: "180일 (6개월)", 365: "365일 (1년)"},
                label="구독 기간", value=30,
            ).classes("w-full")

            async def apply_role():
                if not sel_email.value:
                    return

                def _do():
                    db_a = _get_db()
                    if not db_a:
                        return False, "DB 연결 실패"
                    days = sel_days.value or 0
                    if days > 0 and sel_role.value == "prime":
                        expire = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
                        db_a.update_user_subscription(sel_email.value, sel_role.value, expire)
                        return True, f"✅ {sel_email.value} → PRIME (만료: {expire})"
                    else:
                        ok = db_a.update_user_role(sel_email.value, sel_role.value)
                        return ok, "✅ 변경 완료" if ok else "❌ 실패"

                ok, msg = await asyncio.to_thread(_do)
                ui.notify(msg, type="positive" if ok else "negative")

            async def toggle_ban():
                if not sel_email.value:
                    return

                def _do():
                    db_a = _get_db()
                    if not db_a:
                        return "DB 연결 실패"
                    ok, msg = db_a.toggle_user_ban(sel_email.value)
                    return msg

                msg = await asyncio.to_thread(_do)
                ui.notify(msg)

            with ui.row().classes("gap-2 mt-2"):
                ui.button("등급 적용", on_click=apply_role).props("color=primary")
                ui.button("🚫 차단/해제", on_click=toggle_ban).props("color=negative")

        # ── 전체 이벤트 + 초기화 ──
        with ui.column().classes("flex-1"):
            ui.label("🎉 전체 이벤트").classes("text-white font-bold mb-2")
            ui.label("전 회원에게 체험권을 지급합니다.").classes("text-gray-400 text-sm mb-2")

            async def grant_trial():
                def _do():
                    db_a = _get_db()
                    if not db_a:
                        return False, "DB 연결 실패"
                    return db_a.grant_all_users_trial(14)

                ok, msg = await asyncio.to_thread(_do)
                ui.notify(f"{'🎁 ' + msg if ok else '❌ ' + msg}")

            ui.button("🎁 전원 14일 Prime 지급", on_click=grant_trial).props("color=positive")

            ui.separator().classes("my-3")
            ui.label("⏰ 만료 관리").classes("text-yellow-400 font-bold mb-2")
            ui.label("만료된 PRIME 회원을 FREE로 자동 강등합니다.").classes("text-gray-400 text-sm mb-2")

            async def run_downgrade():
                def _do():
                    from services.auth import downgrade_expired_users
                    return downgrade_expired_users()

                count, details = await asyncio.to_thread(_do)
                if count > 0:
                    ui.notify(f"⏰ {count}명 만료 강등 완료!", type="warning")
                    for d in details:
                        ui.notify(d, type="info")
                else:
                    ui.notify("✅ 만료된 회원 없음", type="positive")

            ui.button("⏰ 만료 회원 강등", on_click=run_downgrade).props("color=warning outlined")

            ui.separator().classes("my-3")
            ui.label("⚠️ 위험 구역").classes("text-red-400 font-bold mb-2")

            async def _reset_all_members():
                ui.notify("⏳ 초기화 중...", type="info")

                def _do():
                    db_a = _get_db()
                    if not db_a:
                        return False, "DB 연결 실패"
                    try:
                        db_a._exec_sqlite("DELETE FROM users WHERE role != 'admin'")
                        db_a._exec_sqlite("DELETE FROM inquiries")
                        u_ok = db_a._do_gist_upload("users", "users.json")
                        i_ok = db_a._do_gist_upload("inquiries", "inquiries.json")
                        if u_ok and i_ok:
                            return True, "🔄 전체 회원 초기화 + Gist 동기화 완료!"
                        else:
                            return True, "⚠️ SQLite 삭제 완료, Gist 일부 실패"
                    except Exception as ex:
                        return False, f"❌ 오류: {ex}"

                ok, msg = await asyncio.to_thread(_do)
                ui.notify(msg, type="positive" if ok else "negative")

            ui.button("🔄 전체 회원 초기화 (관리자 제외)", on_click=_reset_all_members).props("color=red outlined").tooltip("관리자 제외 모든 회원 + 문의 삭제 + Gist 즉시 반영")

        # ── 입금확인 요청 목록 ──
        with ui.column().classes("flex-1"):
            ui.label("💳 입금확인 대기").classes("text-white font-bold mb-2")
            ui.label("멤버십 탭에서 요청된 입금확인 내역").classes("text-gray-400 text-sm mb-2")

            payment_list = ui.column().classes("w-full")

            async def _dismiss_payment(req):
                def _do():
                    db_d = _get_db()
                    if db_d:
                        items = [x for x in db_d.get_all_inquiries()
                                 if x.get("created_at") != req.get("created_at")]
                        db_d.save_inquiries(items)
                        return True
                    return False

                ok = await asyncio.to_thread(_do)
                if ok:
                    ui.notify("✅ 처리 완료", type="positive")
                _load_payment_requests()

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
                            with ui.row().classes("w-full justify-between items-center"):
                                ui.label(f"📌 {req.get('title', '')}").classes("text-white font-bold text-sm")
                                ui.button("✅ 처리완료", on_click=lambda r=req: _dismiss_payment(r)).props("flat dense size=sm color=green")
                            ui.label(req.get("content", "")).classes("text-gray-300 text-xs mt-1 whitespace-pre-line")
                            ui.label(f"🕐 {_to_kst_str(req.get('created_at'))}").classes("text-xs text-gray-500 mt-1")

            _load_payment_requests()
            ui.button("🔄 새로고침", on_click=_load_payment_requests).props("flat dense size=sm color=blue")
