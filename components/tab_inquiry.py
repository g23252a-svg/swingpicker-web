# -*- coding: utf-8 -*-
"""
tab_inquiry.py — 📮 문의 게시판 (NiceGUI Dark Theme)
═══════════════════════════════════════════════════
[v22 Step Y] 전면 리팩토링 — 62 → 92점 목표

개선 사항:
1. ✅ 중복 등록 방지 (inquiry_id PRIMARY KEY + 더블클릭 방어)
2. ✅ Telegram 즉시 알림 (운영자 인지)
3. ✅ 카테고리 분류 (결제/환불/버그/기능/일반)
4. ✅ 답변 시스템 (관리자 답변 + 사용자 확인)
5. ✅ 상태 배지 (대기/처리중/답변완료)
6. ✅ 본인 문의 / 전체 문의 분리
7. ✅ 길이 제한 + 검증
8. ✅ 카카오톡 채널 안내
"""
import hashlib
import html as _html
import logging
import os
from datetime import datetime, timezone, timedelta

from nicegui import ui

_logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))

# ─── 환경변수 ───
BUSINESS_KAKAO = os.environ.get("BUSINESS_KAKAO", "")
BUSINESS_KAKAO_URL = os.environ.get("BUSINESS_KAKAO_URL", "")
BUSINESS_EMAIL = os.environ.get("BUSINESS_EMAIL", "support@swingpicker.com")
TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_ID = os.environ.get("TG_ID", "")

# ─── 길이 제한 ───
MAX_TITLE_LEN = 100
MAX_CONTENT_LEN = 2000
MIN_CONTENT_LEN = 5

# ─── 카테고리 (is_public: True=공개, False=본인+관리자만) ───
CATEGORIES = [
    {"value": "payment", "label": "💳 결제 문의", "color": "amber",
     "priority": 1, "is_public": False},
    {"value": "refund", "label": "💰 환불 요청", "color": "red",
     "priority": 1, "is_public": False},
    {"value": "bug", "label": "🐛 버그 신고", "color": "orange",
     "priority": 2, "is_public": False},
    {"value": "feature", "label": "💡 기능 제안", "color": "blue",
     "priority": 3, "is_public": True},
    {"value": "general", "label": "📝 일반 문의", "color": "gray",
     "priority": 4, "is_public": True},
]
CATEGORY_MAP = {c["value"]: c for c in CATEGORIES}

# 공개 카테고리 (비로그인도 볼 수 있는 — 게시판 성격)
PUBLIC_CATEGORIES = {c["value"] for c in CATEGORIES if c["is_public"]}

# ─── 상태 ───
STATUS_MAP = {
    "open": {"label": "🟡 대기 중", "color": "amber"},
    "in_progress": {"label": "🔵 처리 중", "color": "blue"},
    "replied": {"label": "✅ 답변 완료", "color": "emerald"},
    "closed": {"label": "⚫ 종료", "color": "gray"},
}


def _get_db():
    try:
        from db_utils import get_db
        db = get_db()
        if db and hasattr(db, 'ensure_gist_loaded'):
            db.ensure_gist_loaded()
        return db
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
        return str(value)[:19]


def _send_telegram(text: str):
    """[Step Y] Telegram 알림"""
    if not TG_TOKEN or not TG_ID:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(
            url,
            data={"chat_id": TG_ID, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
    except Exception as e:
        _logger.warning(f"Telegram 알림 실패: {e}")


def _generate_inquiry_id(email: str, title: str, content: str) -> str:
    """[Step Y] 안정적인 inquiry_id 생성 — 동일 내용 더블 클릭 시 동일 ID → 중복 방지"""
    # 5초 윈도우 — 동일 사용자가 동일 내용 5초 내 중복 등록 방지
    now_5sec = int(datetime.now().timestamp() / 5) * 5
    seed = f"{email}|{title}|{content}|{now_5sec}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def _get_user_email(user) -> str:
    """[Step Y] 사용자 이메일 안전 추출"""
    if not user:
        return ""
    return user.get("login_id") or user.get("id") or user.get("email") or ""


def _validate_inquiry(title: str, content: str) -> tuple:
    """[Step Y] 입력 검증 → (ok: bool, msg: str)"""
    title = (title or "").strip()
    content = (content or "").strip()
    
    if not title:
        return False, "제목을 입력해주세요."
    if not content:
        return False, "내용을 입력해주세요."
    if len(title) > MAX_TITLE_LEN:
        return False, f"제목은 {MAX_TITLE_LEN}자 이내로 작성해주세요."
    if len(content) < MIN_CONTENT_LEN:
        return False, f"내용은 {MIN_CONTENT_LEN}자 이상 작성해주세요."
    if len(content) > MAX_CONTENT_LEN:
        return False, f"내용은 {MAX_CONTENT_LEN}자 이내로 작성해주세요."
    return True, ""


# ═══════════════════════════════════════════════════
#  메인 렌더링
# ═══════════════════════════════════════════════════
def render_tab_inquiry(auth, user):
    """[Step Y] 문의 게시판 — 전면 리팩토링
    
    Args:
        auth: "guest" | "free" | "pro" | "prime" | "admin"
        user: 로그인 유저 정보 dict
    """
    if user is None:
        user = {}
    
    is_admin = (auth == "admin")
    user_email = _get_user_email(user)
    
    # ─── 헤더 ───
    with ui.row().classes("w-full items-center justify-between mb-4"):
        ui.label("📮 문의 게시판").classes(
            "text-2xl font-bold text-white"
        )
        # 관리자 통계
        if is_admin:
            db = _get_db()
            if db and hasattr(db, 'get_inquiry_stats'):
                stats = db.get_inquiry_stats()
                with ui.row().classes("gap-2"):
                    ui.badge(f"총 {stats['total']}").props("color=blue")
                    if stats['open'] > 0:
                        ui.badge(f"대기 {stats['open']}").props("color=orange")
                    if stats['replied'] > 0:
                        ui.badge(f"답변 {stats['replied']}").props("color=green")
    
    # ─── 카카오톡 안내 (긴급 문의용) ───
    if BUSINESS_KAKAO and BUSINESS_KAKAO_URL:
        with ui.card().classes(
            "w-full p-3 bg-gradient-to-r from-[#FEE500]/15 to-[#FEE500]/5 "
            "border border-[#FEE500]/50 rounded-lg mb-4"
        ):
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("💬").classes("text-2xl")
                with ui.column().classes("flex-1 gap-0"):
                    ui.label("긴급 문의는 카카오톡 채널로!").classes(
                        "text-sm font-bold text-white"
                    )
                    ui.label(
                        f"채널: {BUSINESS_KAKAO} · "
                        f"이메일: {BUSINESS_EMAIL}"
                    ).classes("text-xs text-gray-300")
                ui.button(
                    f"💬 {BUSINESS_KAKAO}",
                    on_click=lambda: ui.navigate.to(BUSINESS_KAKAO_URL, new_tab=True)
                ).props('size=sm color=amber unelevated').classes(
                    "text-black font-bold"
                )
    
    # ═══════════════════════════════════════════════════
    # 1. 문의 작성 폼
    # ═══════════════════════════════════════════════════
    with ui.card().classes(
        "w-full p-5 bg-[#1a1a2e] border border-cyan-500/30 rounded-xl mb-6"
    ):
        ui.label("✏️ 문의 작성").classes(
            "text-lg font-bold text-cyan-300 mb-3"
        )
        
        # 카테고리 선택
        # [Step AA] 비로그인은 공개 카테고리만 (결제/환불/버그는 로그인 필요)
        if user_email or is_admin:
            category_options = {c["value"]: c["label"] for c in CATEGORIES}
        else:
            category_options = {
                c["value"]: c["label"]
                for c in CATEGORIES if c["is_public"]
            }
        cat_select = ui.select(
            options=category_options,
            value="general",
            label="카테고리",
        ).classes("w-full mb-2").props("outlined")
        
        # 비로그인 안내
        if not user_email and not is_admin:
            ui.label(
                "💡 결제/환불/버그 문의는 로그인 후 작성해주세요. "
                "(개인정보 보호)"
            ).classes("text-xs text-amber-300 mb-2")
        
        # 닉네임 + 이메일 (로그인 시 자동 + readonly)
        with ui.row().classes("w-full gap-3"):
            d_nick = user.get("nickname", "") if user else ""
            d_email = user_email
            
            nick_in = ui.input(
                "닉네임",
                value=d_nick,
                placeholder="작성자 표시 이름",
            ).classes("flex-1").props("outlined dense")
            
            # [Step Z] 로그인 유저는 이메일 readonly (계정 이메일 강제)
            if user_email and not is_admin:
                email_in = ui.input(
                    "이메일 (계정 이메일)",
                    value=d_email,
                ).classes("flex-1").props("outlined dense readonly")
            else:
                # 비로그인 또는 관리자: 자유 입력
                placeholder = (
                    "답변 받을 이메일 (필수)"
                    if not is_admin
                    else "이메일 (선택, 관리자)"
                )
                email_in = ui.input(
                    "이메일",
                    value=d_email,
                    placeholder=placeholder,
                ).classes("flex-1").props("outlined dense")
        
        # 제목
        title_in = ui.input(
            "제목",
            placeholder=f"문의 제목 ({MAX_TITLE_LEN}자 이내)",
        ).classes("w-full mt-2").props("outlined dense")
        
        # 내용
        content_in = ui.textarea(
            "내용",
            placeholder=(
                f"자세한 내용을 작성해주세요 "
                f"({MIN_CONTENT_LEN}~{MAX_CONTENT_LEN}자)"
            ),
        ).classes("w-full mt-2").props("outlined rows=5")
        
        # 글자 수 표시 (실시간)
        char_count_label = ui.label(f"0 / {MAX_CONTENT_LEN}자").classes(
            "text-xs text-gray-500 mt-1"
        )
        
        def on_content_change(e):
            length = len(e.value or "")
            char_count_label.text = f"{length} / {MAX_CONTENT_LEN}자"
            if length > MAX_CONTENT_LEN:
                char_count_label.classes(replace="text-xs text-red-400 mt-1")
            else:
                char_count_label.classes(replace="text-xs text-gray-500 mt-1")
        
        content_in.on_value_change(on_content_change)
        
        # 제출 상태 (더블클릭 방지)
        submitting_state = {"value": False}
        
        async def submit_inquiry():
            # 더블 클릭 방어
            if submitting_state["value"]:
                ui.notify("처리 중입니다...", type="warning")
                return
            submitting_state["value"] = True
            submit_btn.props("disable loading")
            
            try:
                # 검증
                title = (title_in.value or "").strip()
                content = (content_in.value or "").strip()
                nickname = (nick_in.value or "").strip() or "익명"
                email = (email_in.value or "").strip()
                category = cat_select.value or "general"
                
                ok, msg = _validate_inquiry(title, content)
                if not ok:
                    ui.notify(msg, type="warning")
                    return
                
                # [Step Z] 이메일 필수화 (관리자는 예외)
                if not is_admin:
                    if not email:
                        ui.notify(
                            "답변 받을 이메일을 입력해주세요. "
                            "비로그인 시 이메일 없이는 답변 드릴 수 없습니다.",
                            type="warning",
                        )
                        return
                    if "@" not in email or "." not in email:
                        ui.notify("올바른 이메일 형식이 아닙니다.", type="warning")
                        return
                
                # 안정적인 inquiry_id 생성 (5초 윈도우 → 더블 클릭 시 동일 ID)
                inquiry_id = _generate_inquiry_id(email or nickname, title, content)
                created_at = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                
                # DB 저장
                db = _get_db()
                if not db or not hasattr(db, 'add_inquiry'):
                    ui.notify("DB 연결 실패", type="negative")
                    return
                
                ok = db.add_inquiry(
                    inquiry_id=inquiry_id,
                    email=email,
                    nickname=nickname,
                    title=title,
                    content=content,
                    created_at=created_at,
                    category=category,
                )
                
                if ok:
                    # [Step Z] Telegram HTML escape — 사용자 입력 안전 처리
                    cat_info = CATEGORY_MAP.get(category, CATEGORY_MAP["general"])
                    priority_emoji = "🚨" if cat_info["priority"] == 1 else "📮"
                    safe_nickname = _html.escape(nickname)
                    safe_email = _html.escape(email or "익명")
                    safe_title = _html.escape(title)
                    safe_content = _html.escape(content[:200])
                    safe_cat_label = _html.escape(cat_info["label"])
                    
                    _send_telegram(
                        f"{priority_emoji} <b>[신규 문의]</b>\n"
                        f"━━━━━━━━━━━━\n"
                        f"📌 분류: {safe_cat_label}\n"
                        f"👤 {safe_nickname} ({safe_email})\n"
                        f"📋 {safe_title}\n"
                        f"💬 {safe_content}"
                        f"{'...' if len(content) > 200 else ''}\n"
                        f"⏰ {created_at} UTC"
                    )
                    
                    ui.notify(
                        "💌 문의가 등록되었습니다! "
                        "영업일 1~3일 내 답변 드리겠습니다.",
                        type="positive",
                        timeout=5000,
                    )
                    title_in.value = ""
                    content_in.value = ""
                    char_count_label.text = f"0 / {MAX_CONTENT_LEN}자"
                    _refresh_list()
                else:
                    # add_inquiry False → 중복
                    ui.notify(
                        "⚠️ 동일한 문의가 이미 등록되어 있습니다 "
                        "(5초 내 중복 차단).",
                        type="warning",
                    )
            except Exception as e:
                _logger.error(f"문의 등록 오류: {e}", exc_info=True)
                ui.notify(f"등록 실패: {e}", type="negative")
            finally:
                submitting_state["value"] = False
                submit_btn.props(remove="disable loading")
        
        submit_btn = ui.button(
            "💌 문의 등록",
            on_click=submit_inquiry,
        ).classes("mt-3 w-full").props("color=primary size=lg")
    
    # ═══════════════════════════════════════════════════
    # 2. 문의 목록 (탭: 전체 / 내 문의)
    # ═══════════════════════════════════════════════════
    list_container = ui.column().classes("w-full")
    
    # 필터 상태
    filter_state = {
        "view": "my" if user_email and not is_admin else "all",
        "status_filter": "all",
        "category_filter": "all",
        "page": 0,
    }
    
    def _refresh_list():
        list_container.clear()
        with list_container:
            _render_inquiry_list(
                auth=auth,
                user_email=user_email,
                is_admin=is_admin,
                filter_state=filter_state,
                refresh_fn=_refresh_list,
            )
    
    _refresh_list()


# ═══════════════════════════════════════════════════
#  문의 목록 렌더링
# ═══════════════════════════════════════════════════
def _render_inquiry_list(auth, user_email, is_admin, filter_state, refresh_fn):
    """[Step Y+Z] 문의 목록 — 권한별 데이터 분리.
    
    [Step Z] 개인정보 보호:
    - 관리자: 전체 문의 보기 가능
    - 일반 유저: 본인 문의 + 공개 카테고리(general/feature)만
    - 비로그인: 공개 카테고리만 (작성은 별도 막음)
    
    결제/환불/버그는 본인 또는 관리자만 볼 수 있음.
    """
    
    db = _get_db()
    if not db:
        ui.label("DB 연결 실패").classes("text-red-400")
        return
    
    # ─── [Step Z] 권한별 데이터 조회 ───
    if is_admin:
        # 관리자: 전체 문의
        items = db.get_all_inquiries() if hasattr(db, 'get_all_inquiries') else []
    elif user_email:
        # 일반 로그인 유저: 본인 문의 + 공개 카테고리
        my_items = (
            db.get_user_inquiries(user_email)
            if hasattr(db, 'get_user_inquiries') else []
        )
        all_items = (
            db.get_all_inquiries() if hasattr(db, 'get_all_inquiries') else []
        )
        # view 모드에 따라 분기
        if filter_state["view"] == "my":
            items = my_items
        else:
            # 전체 문의 보기 = 본인 + 공개 카테고리만
            my_ids = {i.get("inquiry_id") for i in my_items}
            public_items = [
                i for i in all_items
                if i.get("category") in PUBLIC_CATEGORIES
                or i.get("inquiry_id") in my_ids
            ]
            items = public_items
    else:
        # 비로그인: 공개 카테고리만
        all_items = db.get_all_inquiries() if hasattr(db, 'get_all_inquiries') else []
        items = [
            i for i in all_items if i.get("category") in PUBLIC_CATEGORIES
        ]
    
    # 상태/카테고리 필터
    if filter_state["status_filter"] != "all":
        items = [i for i in items if i.get("status") == filter_state["status_filter"]]
    if filter_state["category_filter"] != "all":
        items = [i for i in items if i.get("category") == filter_state["category_filter"]]
    
    # ─── 필터 바 ───
    with ui.row().classes("w-full items-center gap-2 mb-3 flex-wrap"):
        # [Step Z] 일반 유저: 내 문의 / 공개 게시판 (결제/환불/버그는 비공개)
        if user_email and not is_admin:
            ui.toggle(
                {"my": "📂 내 문의", "all": "🌐 공개 게시판"},
                value=filter_state["view"],
                on_change=lambda e: (
                    filter_state.update({"view": e.value, "page": 0}),
                    refresh_fn(),
                ),
            ).props("dense")
        
        # 카테고리 필터 (관리자만)
        if is_admin:
            cat_filter_options = {"all": "📌 전체"}
            cat_filter_options.update({c["value"]: c["label"] for c in CATEGORIES})
            ui.select(
                options=cat_filter_options,
                value=filter_state["category_filter"],
                on_change=lambda e: (
                    filter_state.update({"category_filter": e.value, "page": 0}),
                    refresh_fn(),
                ),
            ).props("dense outlined").classes("min-w-[140px]")
            
            # 상태 필터
            status_options = {
                "all": "📌 전체 상태",
                "open": "🟡 대기 중",
                "in_progress": "🔵 처리 중",
                "replied": "✅ 답변 완료",
                "closed": "⚫ 종료",
            }
            ui.select(
                options=status_options,
                value=filter_state["status_filter"],
                on_change=lambda e: (
                    filter_state.update({"status_filter": e.value, "page": 0}),
                    refresh_fn(),
                ),
            ).props("dense outlined").classes("min-w-[140px]")
    
    # 헤더
    if is_admin:
        title_text = f"📂 전체 문의 ({len(items)}건)"
    elif filter_state["view"] == "my":
        title_text = f"📂 내 문의 ({len(items)}건)"
    else:
        title_text = (
            f"🌐 공개 게시판 ({len(items)}건) "
            f"· 결제/환불/버그는 본인+관리자만 열람"
        )
    ui.label(title_text).classes("text-white font-bold mt-2 text-sm")
    
    if not items:
        with ui.card().classes(
            "w-full p-8 bg-[#1a1a2e] border border-gray-700 rounded-lg "
            "items-center"
        ):
            ui.label("📭").classes("text-4xl mb-2")
            ui.label("등록된 문의가 없습니다.").classes("text-gray-400")
        return
    
    # ─── 페이지네이션 ───
    PAGE_SIZE = 10
    page = filter_state.get("page", 0)
    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    page_items = items[start:end]
    
    # ─── 카드 목록 ───
    for item in page_items:
        _render_inquiry_card(
            item=item,
            is_admin=is_admin,
            user_email=user_email,
            refresh_fn=refresh_fn,
        )
    
    # 페이지네이션 (10건 이상)
    if len(items) > PAGE_SIZE:
        total_pages = (len(items) + PAGE_SIZE - 1) // PAGE_SIZE
        with ui.row().classes("w-full justify-center gap-2 mt-4"):
            for p in range(total_pages):
                color = "primary" if p == page else "gray"
                btn_props = f"size=sm color={color}"
                if p != page:
                    btn_props += " flat"
                ui.button(
                    f"{p+1}",
                    on_click=lambda pp=p: (
                        filter_state.update({"page": pp}),
                        refresh_fn(),
                    ),
                ).props(btn_props)


def _render_inquiry_card(item, is_admin, user_email, refresh_fn):
    """[Step Y] 단일 문의 카드"""
    inquiry_id = item.get("inquiry_id", "")
    title = item.get("title", "-")
    content = item.get("content", "")
    nickname = item.get("nickname", "익명")
    email = item.get("email") or item.get("id", "")
    created_at = item.get("created_at", "")
    category = item.get("category", "general")
    status = item.get("status", "open")
    admin_reply = item.get("admin_reply", "")
    admin_reply_at = item.get("admin_reply_at", "")
    
    cat_info = CATEGORY_MAP.get(category, CATEGORY_MAP["general"])
    status_info = STATUS_MAP.get(status, STATUS_MAP["open"])
    is_my_inquiry = user_email and email and (user_email.lower() == email.lower())
    
    # 카드 색상 (본인 문의면 강조)
    border_class = (
        "border-cyan-500/40"
        if is_my_inquiry
        else "border-gray-700"
    )
    
    with ui.card().classes(
        f"w-full p-4 mb-3 bg-[#1a1a2e] {border_class} rounded-lg"
    ):
        # ─── 헤더 (배지 + 제목 + 본인 표시) ───
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            # 카테고리 배지
            ui.badge(cat_info["label"]).props(
                f"color={cat_info['color']}"
            ).classes("text-xs")
            
            # 상태 배지
            ui.badge(status_info["label"]).props(
                f"color={status_info['color']}"
            ).classes("text-xs")
            
            # 본인 문의 표시
            if is_my_inquiry:
                ui.badge("👤 내 글").props("color=cyan").classes("text-xs")
            
            # 관리자 삭제 버튼
            if is_admin:
                with ui.row().classes("ml-auto gap-1"):
                    ui.button(
                        "🗑️",
                        on_click=lambda iid=inquiry_id: _confirm_delete(iid, refresh_fn),
                    ).props("flat dense size=sm color=red")
        
        # 제목
        ui.label(f"📌 {title}").classes(
            "text-white font-bold text-base mt-1"
        )
        
        # 내용
        ui.label(content).classes(
            "text-gray-300 text-sm mt-1 whitespace-pre-line"
        )
        
        # 메타 정보
        meta = f"{nickname} · {_to_kst_str(created_at)}"
        ui.label(meta).classes("text-xs text-gray-500 mt-1")
        
        # ─── 관리자 답변 영역 ───
        if admin_reply:
            with ui.card().classes(
                "w-full p-3 mt-3 bg-emerald-900/20 border border-emerald-500/40 "
                "rounded-lg"
            ):
                with ui.row().classes("w-full items-center gap-2 mb-1"):
                    ui.label("💬 운영자 답변").classes(
                        "text-emerald-300 font-bold text-sm"
                    )
                    if admin_reply_at:
                        ui.label(f"· {_to_kst_str(admin_reply_at)}").classes(
                            "text-xs text-gray-500"
                        )
                ui.label(admin_reply).classes(
                    "text-gray-200 text-sm whitespace-pre-line"
                )
        
        # ─── 관리자 답변 입력 ───
        if is_admin and not admin_reply:
            with ui.expansion("✏️ 답변 작성", icon="reply").classes("w-full mt-2"):
                reply_input = ui.textarea(
                    "답변",
                    placeholder="사용자에게 보낼 답변을 작성해주세요.",
                ).classes("w-full").props("outlined rows=4")
                
                async def save_reply(iid=inquiry_id, ri=reply_input, t=title):
                    txt = (ri.value or "").strip()
                    if not txt:
                        ui.notify("답변 내용을 입력해주세요.", type="warning")
                        return
                    db = _get_db()
                    if db and hasattr(db, 'update_inquiry_reply'):
                        ok = db.update_inquiry_reply(iid, txt)
                        if ok:
                            # [Step Z] Telegram HTML escape
                            safe_t = _html.escape(t[:50])
                            safe_txt = _html.escape(txt[:150])
                            _send_telegram(
                                f"✅ <b>[관리자 답변 등록]</b>\n"
                                f"📋 {safe_t}\n"
                                f"💬 {safe_txt}"
                                f"{'...' if len(txt) > 150 else ''}"
                            )
                            ui.notify("답변이 등록되었습니다!", type="positive")
                            refresh_fn()
                        else:
                            ui.notify("답변 등록 실패", type="negative")
                
                with ui.row().classes("w-full justify-end gap-2 mt-2"):
                    ui.button(
                        "💌 답변 등록",
                        on_click=save_reply,
                    ).props("color=primary size=sm")


def _confirm_delete(inquiry_id, refresh_fn):
    """[Step Y] 삭제 확인 다이얼로그"""
    with ui.dialog() as dialog, ui.card().classes("p-4"):
        ui.label("정말 이 문의를 삭제하시겠어요?").classes(
            "text-lg font-bold text-white mb-2"
        )
        ui.label("이 작업은 되돌릴 수 없습니다.").classes(
            "text-sm text-red-400 mb-4"
        )
        with ui.row().classes("gap-2 justify-end"):
            ui.button(
                "취소", on_click=dialog.close
            ).props("flat color=gray")
            
            def do_delete():
                db = _get_db()
                if db and hasattr(db, 'delete_inquiry'):
                    if db.delete_inquiry(inquiry_id):
                        ui.notify("삭제되었습니다", type="positive")
                        dialog.close()
                        refresh_fn()
                    else:
                        ui.notify("삭제 실패", type="negative")
            
            ui.button(
                "🗑️ 삭제", on_click=do_delete
            ).props("color=red")
    dialog.open()
