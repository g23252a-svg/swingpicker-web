# -*- coding: utf-8 -*-
"""
tab_inquiry.py — 📮 문의 게시판 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
DB/Gist 기반 문의 CRUD. 관리자 삭제 가능.
"""
import logging
from datetime import datetime, timezone, timedelta

from nicegui import ui

_logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))


def _get_db():
    try:
        from db_utils import get_db
        return get_db()
    except Exception:
        return None


def _to_kst_str(value):
    """UTC 문자열 → KST 표시"""
    if not value or str(value).strip() in ("", "-", "None"):
        return "-"
    try:
        import pandas as pd
        dt = pd.to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert(KST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)[:16]


def _load_inquiry_items():
    db = _get_db()
    return db.get_all_inquiries() if db else []


def _save_inquiry_items(items):
    db = _get_db()
    return db.save_inquiries(items) if db else False


def render_tab_inquiry(auth, user):
    """Tab 4: 문의 게시판

    Args:
        auth: "guest" | "free" | "pro" | "prime" | "admin"
        user: 로그인 유저 정보 dict (nickname, login_id 등)
    """
    if user is None:
        user = {}

    ui.label("📮 문의 게시판").classes("text-2xl font-bold mb-4 text-white")

    d_nick = user.get("nickname", "") if user else ""
    d_email = user.get("login_id", "") if user else ""

    ui.label("✏️ 문의 작성").classes("text-white font-bold mt-4")
    with ui.row().classes("w-full gap-4"):
        nick_in = ui.input("닉네임", value=d_nick).classes("flex-1")
        email_in = ui.input("이메일 (선택)", value=d_email).classes("flex-1")
    title_in = ui.input("제목", placeholder="문의 제목").classes("w-full")
    content_in = ui.textarea("내용", placeholder="자유롭게 남겨주세요.").classes("w-full").props("rows=5")
    inq_list = ui.column().classes("w-full mt-4")

    async def submit_inquiry():
        if not title_in.value.strip() or not content_in.value.strip():
            ui.notify("제목과 내용을 입력하세요.", type="warning")
            return
        items = _load_inquiry_items()
        items.append({
            "title": title_in.value.strip(), "content": content_in.value.strip(),
            "nickname": nick_in.value.strip() or "익명", "email": email_in.value.strip(),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        })
        if _save_inquiry_items(items):
            ui.notify("💌 문의 등록 완료!", type="positive")
            title_in.value = ""
            content_in.value = ""
            _load_inquiries()
        else:
            ui.notify("등록 실패", type="negative")

    ui.button("💌 문의 등록", on_click=submit_inquiry).classes("mt-2").props("color=primary")

    def _load_inquiries():
        inq_list.clear()
        items = _load_inquiry_items()
        with inq_list:
            ui.label("📂 최근 문의 내역").classes("text-white font-bold mt-4")
            if not items:
                ui.label("등록된 문의가 없습니다.").classes("text-gray-400")
                return
            for i, item in enumerate(reversed(items[-30:])):
                with ui.card().classes("w-full p-3 mb-2 bg-[#1a1a2e] border border-gray-700 rounded-lg"):
                    with ui.row().classes("justify-between"):
                        ui.label(f"📌 {item.get('title', '-')}").classes("text-white font-bold text-sm")
                        if auth == "admin":
                            ui.button("🗑️", on_click=lambda it=item: _del_inquiry(it)).props("flat dense size=sm")
                    ui.label(item.get("content", "")).classes("text-gray-300 text-sm mt-1")
                    meta = f"{item.get('nickname', '익명')} · {_to_kst_str(item.get('created_at'))}"
                    ui.label(meta).classes("text-xs text-gray-500 mt-1")

    def _del_inquiry(item):
        items = [x for x in _load_inquiry_items() if x.get("created_at") != item.get("created_at")]
        _save_inquiry_items(items)
        ui.notify("삭제됨")
        _load_inquiries()

    _load_inquiries()
