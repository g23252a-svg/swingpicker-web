# -*- coding: utf-8 -*-
"""
site_banner.py — 🔔 사이트 전체 공지 배너
═══════════════════════════════════════════════════════════
[v22 Step AA] 운영 중 모든 페이지 상단에 표시되는 공지/경고 배너

용도:
- 패치/업데이트 작업 중 알림 ("일시적 오류 가능")
- 점검 중 알림 ("16:00~17:00 점검")
- 긴급 공지 ("결제 시스템 일시 중단")
- 마케팅 공지 ("신규 기능 출시")

환경변수로 제어 (운영 중 즉시 켜고 끄기 가능):
- SITE_BANNER_ENABLED: true/false (기본 false)
- SITE_BANNER_LEVEL: info/warning/error/maintenance
- SITE_BANNER_TITLE: 제목
- SITE_BANNER_MESSAGE: 본문
- SITE_BANNER_ACTION_URL: 선택, 클릭 시 이동할 URL
- SITE_BANNER_ACTION_LABEL: 선택, 버튼 라벨
"""
import logging
import os

from nicegui import ui

_logger = logging.getLogger(__name__)


def _read_env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("true", "1", "yes", "on", "y")


# 레벨별 색상/아이콘 매핑
_LEVEL_CONFIG = {
    "info": {
        "icon": "ℹ️",
        "border": "border-cyan-500/60",
        "bg": "from-cyan-900/30 to-blue-900/30",
        "title_color": "text-cyan-200",
        "msg_color": "text-cyan-100",
    },
    "warning": {
        "icon": "⚠️",
        "border": "border-amber-500/60",
        "bg": "from-amber-900/30 to-orange-900/30",
        "title_color": "text-amber-200",
        "msg_color": "text-amber-100",
    },
    "error": {
        "icon": "🚨",
        "border": "border-red-500/60",
        "bg": "from-red-900/30 to-rose-900/30",
        "title_color": "text-red-200",
        "msg_color": "text-red-100",
    },
    "maintenance": {
        "icon": "🛠️",
        "border": "border-purple-500/60",
        "bg": "from-purple-900/30 to-indigo-900/30",
        "title_color": "text-purple-200",
        "msg_color": "text-purple-100",
    },
}


def render_site_banner():
    """[Step AA] 사이트 공지 배너 — 활성화 시 페이지 상단에 표시.
    
    main.py 또는 dashboard.py 최상단에서 호출하세요.
    환경변수가 SITE_BANNER_ENABLED=false 이거나 비어있으면 아무것도 안 함.
    """
    enabled = _read_env_bool("SITE_BANNER_ENABLED", False)
    if not enabled:
        return
    
    level = (os.environ.get("SITE_BANNER_LEVEL", "info") or "info").lower()
    if level not in _LEVEL_CONFIG:
        level = "info"
    cfg = _LEVEL_CONFIG[level]
    
    title = os.environ.get("SITE_BANNER_TITLE", "").strip()
    message = os.environ.get("SITE_BANNER_MESSAGE", "").strip()
    action_url = os.environ.get("SITE_BANNER_ACTION_URL", "").strip()
    action_label = os.environ.get("SITE_BANNER_ACTION_LABEL", "자세히").strip()
    
    if not title and not message:
        return  # 내용이 비어 있으면 표시 X
    
    try:
        with ui.row().classes(
            f"w-full bg-gradient-to-r {cfg['bg']} "
            f"border-l-4 {cfg['border']} px-4 py-3 mb-3 "
            f"items-center gap-3 rounded-r-lg shadow-lg"
        ):
            ui.label(cfg["icon"]).classes("text-2xl")
            
            with ui.column().classes("flex-1 gap-0"):
                if title:
                    ui.label(title).classes(
                        f"text-sm font-bold {cfg['title_color']}"
                    )
                if message:
                    ui.label(message).classes(
                        f"text-xs {cfg['msg_color']} whitespace-pre-line"
                    )
            
            if action_url:
                ui.button(
                    action_label,
                    on_click=lambda url=action_url: ui.navigate.to(
                        url, new_tab=True
                    ),
                ).props("size=sm color=white outline").classes(
                    "shrink-0"
                )
    except Exception as e:
        _logger.warning(f"사이트 배너 렌더링 실패: {e}")


# ─── 빠른 프리셋 (환경변수 없이 코드에서 직접 호출 시 사용) ───
def show_patch_notice(message: str = ""):
    """패치 중 공지 빠른 호출"""
    msg = message or (
        "현재 시스템 개선 작업 중입니다. "
        "일부 기능이 일시적으로 작동하지 않을 수 있어요."
    )
    _force_show("warning", "🛠️ 시스템 패치 중", msg)


def show_maintenance_notice(message: str = ""):
    """점검 공지 빠른 호출"""
    msg = message or "정기 점검이 진행 중입니다."
    _force_show("maintenance", "🔧 점검 중", msg)


def _force_show(level: str, title: str, message: str):
    """환경변수 무시하고 강제 표시 (코드 내부 사용)"""
    cfg = _LEVEL_CONFIG.get(level, _LEVEL_CONFIG["info"])
    try:
        with ui.row().classes(
            f"w-full bg-gradient-to-r {cfg['bg']} "
            f"border-l-4 {cfg['border']} px-4 py-3 mb-3 "
            f"items-center gap-3 rounded-r-lg shadow-lg"
        ):
            ui.label(cfg["icon"]).classes("text-2xl")
            with ui.column().classes("flex-1 gap-0"):
                ui.label(title).classes(
                    f"text-sm font-bold {cfg['title_color']}"
                )
                ui.label(message).classes(
                    f"text-xs {cfg['msg_color']} whitespace-pre-line"
                )
    except Exception as e:
        _logger.warning(f"강제 배너 표시 실패: {e}")
