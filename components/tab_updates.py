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
    ui.label("🧩 LDY Pro Trader 업데이트 노트").classes("text-2xl font-bold mb-4")

    if not CHANGELOG:
        with ui.card().classes("w-full p-6 bg-gray-50"):
            ui.label("아직 등록된 업데이트 기록이 없습니다.").classes("text-gray-500")
        return

    latest = CHANGELOG[0]

    # ── 현재 버전 배너 ──
    with ui.card().classes("w-full p-4 bg-green-50 border-l-4 border-green-500 mb-4"):
        ui.label(
            f"현재 버전: v{APP_VERSION}"
        ).classes("text-lg font-bold text-green-800")
        ui.label(
            f"최근 업데이트: {latest['date']} · {latest['title']}"
        ).classes("text-sm text-green-700")

    # ── 버전별 상세 ──
    for idx, log in enumerate(CHANGELOG):
        is_latest = idx == 0
        ver = log.get("version", "?")
        date = log.get("date", "")
        title = log.get("title", "")
        items = log.get("items", [])
        log_type = log.get("type", "patch")

        # 아이콘 선택
        icon = "⭐" if is_latest else ("🚀" if log_type == "major" else "🔧")
        header = f"{icon} v{ver} · {date} — {title}"

        with ui.expansion(header, value=is_latest).classes("w-full mb-1"):
            with ui.column().classes("pl-4 gap-1"):
                for item in items:
                    ui.label(f"• {item}").classes("text-sm leading-relaxed")
