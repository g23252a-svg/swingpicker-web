# -*- coding: utf-8 -*-
"""
tab_updates.py — 🧩 업데이트 노트 (NiceGUI)
═══════════════════════════════════════════════════
version_info.py의 CHANGELOG 데이터를 렌더링.
최신 버전은 기본 펼침, 나머지는 접힘.
"""
from nicegui import ui

try:
    from version_info import CHANGELOG, APP_VERSION
except ImportError:
    CHANGELOG = []
    APP_VERSION = "0.0.0"


def render_tab_updates():
    """Tab 6: 업데이트 노트"""
    ui.label("🧩 업데이트 노트").classes("text-2xl font-bold mb-4 text-white")

    if not CHANGELOG:
        ui.label("등록된 업데이트 기록이 없습니다.").classes("text-gray-400")
        return

    latest = CHANGELOG[0]

    # ── 현재 버전 배너 ──
    with ui.card().classes("w-full p-4 bg-green-900/30 border border-green-700 rounded-xl mb-4"):
        ui.label(f"현재 버전: v{APP_VERSION}").classes("text-green-400 font-bold")
        ui.label(f"최근: {latest.get('date')} · {latest.get('title')}").classes("text-green-300 text-sm")

    # ── 버전별 상세 ──
    for i, log in enumerate(CHANGELOG):
        is_latest = i == 0
        ver = log.get("version", "?")
        date = log.get("date", "")
        title = log.get("title", "")
        items = log.get("items", [])

        icon = "⭐" if is_latest else "📜"
        header = f"{icon} v{ver} · {date} — {title}"

        with ui.expansion(header, value=is_latest).classes("w-full mb-1"):
            for item in items:
                ui.label(f"• {item}").classes("text-gray-300 text-sm ml-4")
