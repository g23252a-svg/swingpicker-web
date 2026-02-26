# -*- coding: utf-8 -*-
"""
tab_inquiry.py — 📮 문의 게시판 (NiceGUI 컴포넌트)
═══════════════════════════════════════════════════
[v1.0] Streamlit → NiceGUI 이식

핵심 설계:
  1. 중복 제출 방지 (Idempotency)
     - 세션 단위 fingerprint 토큰
     - DB 마지막 항목과 내용 비교 (이중 방어)
  2. 관리자 삭제 기능
     - auth_status == "admin" 일 때만 🗑️ 버튼 노출
  3. 비동기 I/O
     - DB/Gist 쓰기는 run.io_bound → UI 블로킹 없음

구조:
  ┌─ 문의 작성 폼 (닉네임, 이메일, 제목, 내용)
  ├─ [문의 등록] 버튼 (중복 방어)
  └─ 최근 문의 내역 (최신순, 관리자 삭제 가능)
"""
import logging
from datetime import datetime, timezone

from nicegui import ui, run, app

from db_utils import get_db

_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  유틸
# ═══════════════════════════════════════════════════

def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _to_kst_display(utc_str: str) -> str:
    """UTC 문자열 → KST 표시용 (간이 변환)"""
    try:
        from datetime import timedelta
        dt = datetime.strptime(str(utc_str)[:19], "%Y-%m-%d %H:%M:%S")
        kst = dt + timedelta(hours=9)
        return kst.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(utc_str)[:16] if utc_str else "-"


def _get_session_val(key: str, default=None):
    """세션 스토리지 안전 접근"""
    try:
        return app.storage.user.get(key, default)
    except Exception:
        return default


def _set_session_val(key: str, value):
    """세션 스토리지 안전 저장"""
    try:
        app.storage.user[key] = value
    except Exception:
        pass


# ═══════════════════════════════════════════════════
#  메인 렌더 함수
# ═══════════════════════════════════════════════════

def render_tab_inquiry(auth_status: str, user_info: dict = None):
    """Tab 4: 문의 게시판

    Args:
        auth_status: "guest" | "free" | "prime" | "pro" | "admin"
        user_info: 로그인 유저 정보 dict (nickname, login_id 등)
    """
    if user_info is None:
        user_info = {}

    ui.label("📮 문의 게시판").classes("text-2xl font-bold mb-4")

    # ── 결과 표시 영역 (목록) ──
    list_container = ui.column().classes("w-full")

    # ── 1) 문의 작성 폼 ──
    _render_write_form(auth_status, user_info, list_container)

    ui.separator().classes("my-6")

    # ── 2) 최근 문의 내역 ──
    ui.label("📂 최근 문의 내역").classes("text-xl font-bold mb-2")
    _refresh_inquiry_list(list_container, auth_status)


# ═══════════════════════════════════════════════════
#  작성 폼
# ═══════════════════════════════════════════════════

def _render_write_form(auth_status: str, user_info: dict, list_container):
    """문의 작성 UI"""

    if auth_status == "guest":
        with ui.card().classes("w-full p-6 bg-gray-50"):
            ui.label("🔒 문의를 작성하려면 로그인이 필요합니다.").classes("text-gray-600")
        return

    ui.label("✏️ 문의 작성").classes("text-lg font-semibold mb-2")

    with ui.card().classes("w-full p-4"):
        with ui.row().classes("w-full gap-4"):
            inp_nick = ui.input(
                "닉네임",
                value=user_info.get("nickname", ""),
                placeholder="닉네임 또는 이름",
            ).classes("flex-1")
            inp_email = ui.input(
                "이메일 (선택)",
                value=user_info.get("login_id", ""),
                placeholder="답변 받을 이메일",
            ).classes("flex-1")

        inp_title = ui.input(
            "제목",
            placeholder="문의 제목을 입력해 주세요.",
        ).classes("w-full mt-2")

        inp_content = ui.textarea(
            "내용",
            placeholder="사이트 사용 관련 문의를 자유롭게 남겨 주세요.",
        ).classes("w-full mt-2").props("rows=5")

        submit_btn = ui.button("💌 문의 등록", color="primary").classes("mt-3")

        async def on_submit():
            title = (inp_title.value or "").strip()
            content = (inp_content.value or "").strip()
            nickname = (inp_nick.value or "").strip() or "익명"
            email = (inp_email.value or "").strip()

            # ── 빈칸 체크 ──
            if not title or not content:
                ui.notify("제목과 내용을 모두 입력해 주세요.", type="warning")
                return

            # ── 중복 제출 방지 #1: 세션 fingerprint ──
            fingerprint = f"{nickname}_{title}_{content[:30]}"
            last_fp = _get_session_val("last_inquiry_fp")
            if last_fp == fingerprint:
                ui.notify("이미 처리 중이거나 완료된 문의입니다.", type="warning")
                return

            submit_btn.disable()
            submit_btn.text = "⏳ 등록 중..."

            try:
                db = get_db()

                # ── 중복 제출 방지 #2: DB 마지막 항목과 비교 ──
                existing = await run.io_bound(db.get_all_inquiries)
                if existing:
                    last = existing[-1]
                    if last.get("title") == title and last.get("content") == content:
                        ui.notify("이미 장부에 기록된 내용입니다.", type="info")
                        _set_session_val("last_inquiry_fp", fingerprint)
                        return

                # ── 새 문의 등록 ──
                new_item = {
                    "email": email or nickname,
                    "nickname": nickname,
                    "title": title,
                    "content": content,
                    "created_at": _now_utc_str(),
                }
                existing.append(new_item)

                ok = await run.io_bound(db.save_inquiries, existing)

                if ok:
                    _set_session_val("last_inquiry_fp", fingerprint)
                    ui.notify("✅ 문의가 성공적으로 등록되었습니다!", type="positive")

                    # 입력 초기화
                    inp_title.value = ""
                    inp_content.value = ""

                    # 목록 갱신
                    _refresh_inquiry_list(list_container, auth_status)
                else:
                    ui.notify("통신 장애로 기록에 실패했습니다.", type="negative")

            except Exception as e:
                _logger.error(f"문의 등록 실패: {e}")
                ui.notify(f"오류: {e}", type="negative")

            finally:
                submit_btn.enable()
                submit_btn.text = "💌 문의 등록"

        submit_btn.on_click(on_submit)


# ═══════════════════════════════════════════════════
#  문의 목록
# ═══════════════════════════════════════════════════

def _refresh_inquiry_list(container, auth_status: str):
    """문의 목록을 (재)렌더링"""
    container.clear()

    with container:
        try:
            db = get_db()
            inquiries = db.get_all_inquiries()
        except Exception:
            inquiries = []

        if not inquiries:
            ui.label("아직 등록된 문의가 없습니다.").classes("text-gray-500 mt-4")
            return

        # 최신순 50개
        recent = list(reversed(inquiries[-50:]))

        for i, item in enumerate(recent):
            date_str = item.get("created_at", "-")
            date_disp = _to_kst_display(date_str)

            with ui.card().classes("w-full p-4 mb-2 border"):
                with ui.row().classes("w-full items-start justify-between"):
                    # ── 왼쪽: 제목 + 메타 + 내용 ──
                    with ui.column().classes("flex-1"):
                        ui.label(f"제목: {item.get('title', '-')}").classes(
                            "font-bold text-base"
                        )

                        meta_parts = [
                            f"작성자: {item.get('nickname', '익명')}",
                            f"작성일: {date_disp}",
                        ]
                        if item.get("email"):
                            meta_parts.append(f"이메일: {item['email']}")
                        ui.label(" · ".join(meta_parts)).classes("text-xs text-gray-400 mt-1")

                        content_text = item.get("content", "")
                        ui.label(content_text).classes("text-sm text-gray-700 mt-2 whitespace-pre-wrap")

                    # ── 오른쪽: 관리자 삭제 버튼 ──
                    if auth_status == "admin":
                        async def delete_inquiry(created_at=date_str):
                            try:
                                db_ = get_db()
                                all_items = await run.io_bound(db_.get_all_inquiries)
                                new_list = [
                                    x for x in all_items
                                    if x.get("created_at") != created_at
                                ]
                                await run.io_bound(db_.save_inquiries, new_list)
                                ui.notify("글이 삭제되었습니다.", type="info")
                                _refresh_inquiry_list(container, auth_status)
                            except Exception as e:
                                ui.notify(f"삭제 실패: {e}", type="negative")

                        ui.button(
                            icon="delete",
                            on_click=delete_inquiry,
                        ).props("flat dense color=red").tooltip("글 삭제")
