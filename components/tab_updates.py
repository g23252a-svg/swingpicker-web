# -*- coding: utf-8 -*-
"""
tab_updates.py — 🧩 업데이트 노트 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
[v22 Step AH] 전면 리팩토링 — 70 → 90점 목표

개선 사항:
1. ✅ 카테고리 자동 분류 (🆕 신규 / ⚡ 개선 / 🐛 버그 / 🛡️ 보안 / 📊 UX / 🏢 운영)
2. ✅ 검색 박스 (실시간 키워드 필터)
3. ✅ 카테고리 필터 토글
4. ✅ 히어로 카드 (최신 버전 + 핵심 3가지 + 통계)
5. ✅ 누적 통계 (전체 버전 / 카테고리별 변경 수)
6. ✅ v21 PDF 배너 정리 (조건부 + 우측 작게)
7. ✅ 모바일 가독성 (그룹 토글로 스크롤 단축)
"""
import html as _html
import os
import re

from nicegui import ui

try:
    from version_info import CHANGELOG, APP_VERSION
except ImportError:
    CHANGELOG = []
    APP_VERSION = "0.0.0"


# ═══════════════════════════════════════════════════
#  카테고리 자동 분류 시스템
# ═══════════════════════════════════════════════════
# [Step AI] 우선순위 재조정 — 결제/약관/개인정보 서비스 특성 반영
# 1) security: 보안/안정성 (방어/롤백/검증 등 — 가장 자주 발생)
# 2) business: 결제/환불/멤버십 (서비스 본질)
# 3) bug: 명확한 버그 수정만 (방어/롤백은 security로)
# 4) new: 신규 기능
# 5) performance: 성능/속도
# 6) improvement: 개선/리팩토링
# 7) ux: UI/UX
# 8) data: 데이터/분석
CATEGORY_RULES = [
    {
        "key": "security",
        "label": "🛡️ 보안 · 안정성",
        "color": "amber",
        # [Step AI] 차단/방어/롤백/재시도/fallback을 보안으로 이동
        "keywords": ["보안", "암호화", "검증", "인증", "동의", "약관", "개인정보",
                     "권한", "PCI", "SSL", "토큰", "해시", "잠금",
                     "감사 로그", "위변조", "탈출", "민감", "익명",
                     "차단", "방어", "롤백", "재시도", "fallback"],
        "emoji_hints": ["🛡️", "🔐", "🔒", "🔑", "📜", "🚨"],
    },
    {
        "key": "business",
        "label": "🏢 운영 · 비즈니스",
        "color": "blue",
        "keywords": ["사업자", "결제", "환불", "구독", "Prime", "멤버십", "토스",
                     "가맹점", "도메인", "통신판매", "관리자", "운영", "고객",
                     "문의", "카카오톡", "Telegram", "이메일 인증"],
        "emoji_hints": ["💎", "🏢", "💳", "💰", "📞", "📧", "📮", "👑"],
    },
    {
        "key": "bug",
        "label": "🐛 버그 수정",
        "color": "red",
        # [Step AI] 명확한 버그 키워드만 유지 (차단/방어/롤백/재시도/fallback 제거)
        "keywords": ["수정", "버그", "오류", "장애", "복구", "fix", "에러"],
        "emoji_hints": ["🐛"],
    },
    {
        "key": "new",
        "label": "🆕 신규 기능",
        "color": "green",
        "keywords": ["신규", "도입", "추가", "새로운", "출시", "오픈", "런칭",
                     "도입했", "새 기능", "활성화", "지원"],
        "emoji_hints": ["🆕", "🎉", "🚀", "💎", "🎁", "✨"],
    },
    {
        "key": "performance",
        "label": "⚡ 성능 · 속도",
        "color": "purple",
        "keywords": ["성능", "속도", "최적화", "캐시", "병렬", "지연", "throttle",
                     "재계산", "rebuild", "load", "응답", "처리량"],
        "emoji_hints": ["⚡", "🏎️", "💨"],
    },
    {
        "key": "improvement",
        "label": "📈 개선 · 강화",
        "color": "cyan",
        "keywords": ["개선", "강화", "확장", "리팩토링", "재설계", "통합",
                     "정확도", "정밀도", "보존", "안정", "구조", "체계화",
                     "자동화", "고도화", "보강"],
        "emoji_hints": ["📈", "🔄", "🔁", "🎯"],
    },
    {
        "key": "ux",
        "label": "📱 UX · 디자인",
        "color": "indigo",
        "keywords": ["UI", "UX", "디자인", "가독성", "표시", "테마", "다크",
                     "모바일", "반응형", "스타일", "버튼", "카드", "아이콘",
                     "tooltip", "안내", "메시지"],
        "emoji_hints": ["📱", "🎨", "💡", "👀", "🖼️"],
    },
    {
        "key": "data",
        "label": "📊 데이터 · 분석",
        "color": "teal",
        "keywords": ["데이터", "백테스트", "스코어", "지표", "ELITE", "팩터",
                     "벤치마크", "통계", "차트", "시그널", "ROUTE",
                     "AI", "TOP_PICK", "추천", "랭킹", "RR", "Kelly"],
        "emoji_hints": ["📊", "📉", "🧪"],
    },
]
# 어디에도 안 맞으면 fallback
DEFAULT_CATEGORY = {
    "key": "other",
    "label": "🔧 기타",
    "color": "gray",
    "keywords": [],
    "emoji_hints": [],
}

# [Step AJ] key → category 매핑 (수동 지정용)
CATEGORY_BY_KEY = {cat["key"]: cat for cat in CATEGORY_RULES}
CATEGORY_BY_KEY[DEFAULT_CATEGORY["key"]] = DEFAULT_CATEGORY


def _get_item_text(item) -> str:
    """[Step AJ] 항목에서 텍스트 추출 — dict/str 모두 지원.
    
    하위 호환:
    - str: 그대로 반환
    - dict: text 또는 message 필드 사용
    """
    if isinstance(item, dict):
        return str(item.get("text") or item.get("message") or "")
    return str(item) if item else ""


def _categorize_item(item) -> dict:
    """[Step AH+AJ] 변경 항목을 카테고리에 자동 분류.
    
    [Step AJ] 수동 category 필드 우선 — dict로 category 지정 시 자동 분류 스킵.
    
    예:
    "🐛 단순 텍스트"                              → 자동 분류
    {"text": "...", "category": "data"}           → 수동 분류 (data 강제)
    
    우선순위:
    1) item이 dict이고 category 필드가 유효하면 그대로 사용
    2) 이모지 힌트 (가장 정확)
    3) 키워드 매칭
    4) 기타 (DEFAULT_CATEGORY)
    """
    if not item:
        return DEFAULT_CATEGORY
    
    # [Step AJ] 1순위: 수동 category 필드 (dict)
    if isinstance(item, dict):
        manual_cat = item.get("category", "")
        if manual_cat and manual_cat in CATEGORY_BY_KEY:
            return CATEGORY_BY_KEY[manual_cat]
    
    text = _get_item_text(item)
    if not text:
        return DEFAULT_CATEGORY
    
    text_lower = text.lower()
    
    # 2순위: 이모지 힌트 (가장 정확)
    for cat in CATEGORY_RULES:
        for hint in cat["emoji_hints"]:
            if hint in text:
                return cat
    
    # 3순위: 키워드 매칭
    for cat in CATEGORY_RULES:
        for kw in cat["keywords"]:
            if kw.lower() in text_lower:
                return cat
    
    return DEFAULT_CATEGORY


# ═══════════════════════════════════════════════════
#  통계 계산
# ═══════════════════════════════════════════════════
def _compute_stats() -> dict:
    """[Step AH+AJ] 전체 CHANGELOG 통계 (dict 항목 지원)."""
    total_changes = 0
    by_category = {cat["key"]: 0 for cat in CATEGORY_RULES}
    by_category[DEFAULT_CATEGORY["key"]] = 0
    
    for log in CHANGELOG:
        for item in log.get("items", []):
            total_changes += 1
            cat = _categorize_item(item)  # dict/str 모두 자동 처리
            by_category[cat["key"]] = by_category.get(cat["key"], 0) + 1
    
    return {
        "total_versions": len(CHANGELOG),
        "total_changes": total_changes,
        "by_category": by_category,
    }


def _compute_version_stats(items: list) -> dict:
    """[Step AH+AJ] 단일 버전의 카테고리별 통계 (dict 항목 지원)."""
    stats = {cat["key"]: 0 for cat in CATEGORY_RULES}
    stats[DEFAULT_CATEGORY["key"]] = 0
    for item in items:
        cat = _categorize_item(item)  # dict/str 모두 자동 처리
        stats[cat["key"]] = stats.get(cat["key"], 0) + 1
    return stats


def _strip_markdown(text: str) -> str:
    """**bold** → bold (마크다운 굵게 표시 단순화)."""
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)


def _highlight_search(text: str, keyword: str) -> str:
    """[Step AJ] 검색어를 mark 태그로 강조 — escape 처리 필수.
    
    Args:
        text: 원본 텍스트 (escape 처리됨)
        keyword: 검색 키워드 (없으면 escape만 적용)
    
    Returns:
        HTML 안전한 문자열 (mark 태그 포함)
    """
    safe = _html.escape(text or "")
    if not keyword:
        return safe
    
    keyword = keyword.strip()
    if not keyword:
        return safe
    
    try:
        # 검색 키워드도 escape 후 정규식 escape
        safe_keyword = _html.escape(keyword)
        pattern = re.escape(safe_keyword)
        return re.sub(
            pattern,
            lambda m: (
                f"<mark style='background:#F59E0B;color:#111827;"
                f"border-radius:3px;padding:0 3px;font-weight:600'>"
                f"{m.group(0)}</mark>"
            ),
            safe,
            flags=re.IGNORECASE,
        )
    except Exception:
        # 정규식 오류 시 안전한 fallback
        return safe


# ═══════════════════════════════════════════════════
#  메인 렌더링
# ═══════════════════════════════════════════════════
def render_tab_updates():
    """[Step AH+AJ] 업데이트 노트 — 전면 리팩토링"""
    
    # [Step AJ] CHANGELOG 최신순 보장 — date 기준 내림차순 정렬
    # version_info.py의 순서가 어긋나도 안전하게 최신 표시
    sorted_changelog = sorted(
        CHANGELOG,
        key=lambda x: x.get("date", "0000-00-00"),
        reverse=True,
    )
    
    # ─── 헤더 ───
    with ui.row().classes("w-full items-center justify-between mb-3"):
        with ui.column().classes("gap-0"):
            ui.label("🧩 업데이트 노트").classes(
                "text-2xl font-bold text-white"
            )
            ui.label(
                f"v{APP_VERSION} · {len(CHANGELOG)}개 버전 누적"
            ).classes("text-xs text-gray-400")
        
        # v21 PDF 제안서 (작게, 조건부)
        pdf_paths = [
            "static/swingpicker_upgrade_v21.pdf",
            "/static/swingpicker_upgrade_v21.pdf",
        ]
        pdf_exists = any(
            os.path.exists(p.lstrip('/')) for p in pdf_paths
        )
        if pdf_exists:
            ui.link(
                "📄 v21 제안서",
                "/static/swingpicker_upgrade_v21.pdf",
                new_tab=True,
            ).classes(
                "text-xs text-indigo-400 hover:text-indigo-300 no-underline"
            ).style("padding: 6px 12px; border: 1px solid rgba(99,102,241,0.4); border-radius: 6px;")
    
    if not sorted_changelog:
        with ui.card().classes(
            "w-full p-8 bg-[#1a1a2e] border border-gray-700 rounded-lg "
            "items-center"
        ):
            ui.label("📭").classes("text-4xl mb-2")
            ui.label("등록된 업데이트 기록이 없습니다.").classes("text-gray-400")
        return
    
    latest = sorted_changelog[0]
    
    # ─── 히어로 카드 (최신 버전) ───
    _render_hero_card(latest)
    
    # ─── 누적 통계 카드 ───
    _render_stats_card()
    
    # ─── 검색 박스 + 필터 ───
    state = {"search": "", "category": "all"}
    
    search_input = ui.input(
        placeholder="🔍 검색 (예: 결제, 환불, 버그, AI, 백테스트...)"
    ).classes("w-full mt-3 mb-2").props(
        "outlined dense clearable debounce=300"
    )
    
    # [Step AJ] 카테고리 필터 — select 방식 (모바일 친화)
    # 9개 카테고리 → toggle은 모바일에서 너무 길어짐
    category_options = {"all": "📌 전체 카테고리"}
    for cat in CATEGORY_RULES:
        category_options[cat["key"]] = cat["label"]
    category_options[DEFAULT_CATEGORY["key"]] = DEFAULT_CATEGORY["label"]
    
    ui.select(
        options=category_options,
        value="all",
        label="카테고리 필터",
        on_change=lambda e: (
            state.update({"category": e.value}),
            _refresh(),
        ),
    ).classes("w-full mb-3").props("outlined dense")
    
    # 본문 영역
    content_area = ui.column().classes("w-full")
    
    def _refresh():
        content_area.clear()
        with content_area:
            _render_versions(state, sorted_changelog)
    
    def on_search_change(e):
        state["search"] = (e.value or "").strip()
        _refresh()
    
    search_input.on_value_change(on_search_change)
    
    _refresh()


def _render_hero_card(latest: dict):
    """[Step AH+AJ] 최신 버전 히어로 카드 (dict 항목 지원)"""
    items = latest.get("items", [])
    # [Step AJ] dict/str 모두 텍스트 추출 후 strip
    items_clean = [_strip_markdown(_get_item_text(i)) for i in items]
    version_stats = _compute_version_stats(items)
    
    # [Step AI] Top 5 highlight — 첫 5개 항목의 핵심 부분만 추출
    highlights = []
    for item in items_clean[:5]:
        # ** 제거된 텍스트에서 첫 줄/문장만 추출
        first_part = item.split(":", 1)[0].strip() if ":" in item else item
        if len(first_part) > 50:
            first_part = first_part[:50] + "..."
        highlights.append(first_part)
    
    with ui.card().classes(
        "w-full p-5 mb-4 rounded-xl border border-cyan-500/40"
    ).style(
        "background: linear-gradient(135deg, rgba(6,182,212,0.15) 0%, "
        "rgba(15,23,42,0.95) 50%, rgba(99,102,241,0.15) 100%);"
    ):
        # 헤더 (배지 + 제목)
        with ui.row().classes("w-full items-center gap-2 mb-2 flex-wrap"):
            ui.badge(f"v{latest.get('version', '?')}").props(
                "color=cyan"
            ).classes("text-sm font-bold")
            type_emoji = {"major": "🚀", "minor": "✨",
                          "patch": "🔧"}.get(latest.get("type", ""), "🎯")
            ui.badge(
                f"{type_emoji} {latest.get('type', 'release').upper()}"
            ).props("color=indigo").classes("text-xs")
            ui.label(latest.get("date", "")).classes(
                "text-sm text-gray-400 ml-2"
            )
        
        # 제목
        title = latest.get("title", "")
        # "vXX —" 같은 prefix 제거하여 깔끔하게
        title_clean = re.sub(r"^v\d+\.?\d*\s*—\s*", "", title)
        ui.label(f"⭐ {title_clean}").classes(
            "text-xl font-bold text-white mb-3"
        )
        
        # Top 3-5 highlights
        if highlights:
            ui.label("✨ 이번 업데이트 핵심").classes(
                "text-sm font-bold text-cyan-300 mb-1"
            )
            for h in highlights[:5]:
                ui.label(f"• {h}").classes(
                    "text-sm text-gray-200 ml-2 mb-0.5"
                )
            if len(items) > 5:
                ui.label(
                    f"  … 외 {len(items) - 5}개 변경사항"
                ).classes("text-xs text-gray-500 ml-2 italic")
        
        # 카테고리별 배지 통계
        ui.separator().classes("my-3")
        with ui.row().classes("w-full gap-2 flex-wrap"):
            for cat in CATEGORY_RULES:
                count = version_stats.get(cat["key"], 0)
                if count > 0:
                    ui.badge(f"{cat['label']} {count}").props(
                        f"color={cat['color']}"
                    ).classes("text-xs")
            other = version_stats.get(DEFAULT_CATEGORY["key"], 0)
            if other > 0:
                ui.badge(f"{DEFAULT_CATEGORY['label']} {other}").props(
                    "color=gray"
                ).classes("text-xs")


def _render_stats_card():
    """[Step AH] 누적 통계 카드"""
    stats = _compute_stats()
    
    with ui.card().classes(
        "w-full p-3 mb-3 bg-[#1a1a2e] border border-gray-700/50 rounded-lg"
    ):
        with ui.row().classes("w-full items-center justify-between flex-wrap gap-2"):
            with ui.column().classes("gap-0"):
                ui.label("📊 누적 변경사항").classes(
                    "text-sm font-bold text-gray-300"
                )
                ui.label(
                    f"{stats['total_versions']}개 버전  ·  "
                    f"{stats['total_changes']}개 변경"
                ).classes("text-xs text-gray-500")
            
            # 주요 카테고리 통계 (개수 기준 상위 4개)
            sorted_cats = sorted(
                CATEGORY_RULES,
                key=lambda c: stats["by_category"].get(c["key"], 0),
                reverse=True,
            )[:4]
            with ui.row().classes("gap-1 flex-wrap"):
                for cat in sorted_cats:
                    count = stats["by_category"].get(cat["key"], 0)
                    if count > 0:
                        ui.badge(f"{cat['label']} {count}").props(
                            f"color={cat['color']}"
                        ).classes("text-[10px]")


def _render_versions(state: dict, sorted_changelog: list = None):
    """[Step AH+AJ] 버전별 변경사항 렌더링 (검색/필터/하이라이트 적용).
    
    [Step AJ] 변경사항:
    - sorted_changelog 전달받기 (date 기준 최신순 정렬됨)
    - dict 항목 지원 ({text, category} 형태)
    - 검색어 하이라이트 (ui.html + mark 태그)
    """
    
    # [Step AJ] sorted_changelog 없으면 fallback (안전)
    if sorted_changelog is None:
        sorted_changelog = sorted(
            CHANGELOG,
            key=lambda x: x.get("date", "0000-00-00"),
            reverse=True,
        )
    
    search_raw = state.get("search", "")
    search = search_raw.lower()
    cat_filter = state["category"]
    
    # 결과 카운트
    visible_count = 0
    
    for i, log in enumerate(sorted_changelog):
        is_latest = (i == 0)
        ver = log.get("version", "?")
        date = log.get("date", "")
        title = log.get("title", "")
        items = log.get("items", [])
        
        # 항목별 카테고리 분류
        items_by_cat = {}
        for item in items:
            # [Step AJ] dict/str 모두 지원
            item_text = _get_item_text(item)
            item_clean = _strip_markdown(item_text)
            cat = _categorize_item(item)
            
            # 검색 필터
            if search and search not in item_clean.lower() and search not in title.lower():
                continue
            
            # 카테고리 필터
            if cat_filter != "all" and cat["key"] != cat_filter:
                continue
            
            items_by_cat.setdefault(cat["key"], {
                "label": cat["label"],
                "color": cat["color"],
                "items": [],
            })
            items_by_cat[cat["key"]]["items"].append(item_clean)
        
        # 매칭된 항목 없으면 스킵
        if not items_by_cat:
            continue
        
        visible_count += sum(len(v["items"]) for v in items_by_cat.values())
        
        # 헤더
        type_emoji = {"major": "🚀", "minor": "✨",
                      "patch": "🔧"}.get(log.get("type", ""), "📜")
        icon = "⭐" if is_latest else type_emoji
        total_in_ver = sum(len(v["items"]) for v in items_by_cat.values())
        header = f"{icon} v{ver} · {date} · {total_in_ver}개"
        if title:
            # 제목 짧게
            short_title = re.sub(r"^v\d+\.?\d*\s*—\s*", "", title)
            if len(short_title) > 60:
                short_title = short_title[:60] + "..."
            header += f" — {short_title}"
        
        with ui.expansion(header, value=is_latest).classes(
            "w-full mb-2"
        ).props("expand-icon=expand_more header-class=text-white"):
            
            # 카테고리별 그룹 표시 (CATEGORY_RULES 순서대로)
            ordered_keys = [c["key"] for c in CATEGORY_RULES] + [DEFAULT_CATEGORY["key"]]
            
            for cat_key in ordered_keys:
                if cat_key not in items_by_cat:
                    continue
                grp = items_by_cat[cat_key]
                
                # 카테고리 헤더
                with ui.row().classes("w-full items-center gap-2 mt-2 mb-1"):
                    ui.badge(
                        f"{grp['label']} ({len(grp['items'])})"
                    ).props(f"color={grp['color']}").classes("text-xs")
                
                # 항목 리스트
                for item in grp["items"]:
                    # [Step AJ] 검색어 하이라이트 (escape + mark)
                    highlighted = _highlight_search(item, search_raw)
                    ui.html(
                        f"<div class='text-sm text-gray-300 ml-3 mb-1 "
                        f"leading-relaxed'>• {highlighted}</div>"
                    )
    
    # 결과 없음 메시지
    if visible_count == 0:
        with ui.card().classes(
            "w-full p-6 bg-[#1a1a2e] border border-gray-700 rounded-lg "
            "items-center"
        ):
            ui.label("🔍").classes("text-4xl mb-2")
            ui.label("검색 결과가 없습니다.").classes(
                "text-gray-400 text-base font-bold"
            )
            search_hint = state.get("search", "")
            if search_hint:
                ui.label(f"검색어: '{search_hint}'").classes(
                    "text-xs text-gray-500 mt-1"
                )
            ui.label("다른 키워드를 시도하거나 카테고리 필터를 '전체'로 바꿔보세요.").classes(
                "text-xs text-gray-500 mt-1"
            )
    elif state["search"] or state["category"] != "all":
        ui.label(
            f"💡 {visible_count}개 변경사항이 매칭됨"
        ).classes("text-xs text-gray-500 italic mt-2 text-center")
