# -*- coding: utf-8 -*-
"""
login_page.py — 🔐 로그인 / 가입 / 계정 복구 페이지
═══════════════════════════════════════════════════
[v2] 가입 시 이메일 인증코드 검증 추가
"""
from nicegui import ui

from components.ui_utils import DARK_CSS
from services.auth import (
    MASTER_ADMIN_ID, ADMIN_PW_SET, SECURITY_QUESTIONS, ALLOWED_DOMAINS,
    get_db, verify_admin_pw, normalize_email, check_pw_strength,
    authenticate_user, set_current_user,
    create_salt, hash_pw, hash_ans,
)

try:
    from email_verify import send_verification_email, verify_code, is_configured as email_configured
    EMAIL_VERIFY_OK = True
except ImportError:
    EMAIL_VERIFY_OK = False


@ui.page('/login')
def login_page():
    ui.add_head_html(DARK_CSS.replace("1400px", "500px"))

    with ui.card().classes("w-full p-8 bg-[#1a1a2e] border border-gray-700 rounded-2xl mt-16"):
        ui.label("🔐 LDY Pro Trader").classes("text-2xl font-bold text-center text-white w-full mb-4")

        with ui.tabs().classes("w-full") as tabs:
            t_login = ui.tab("로그인")
            t_join = ui.tab("전략군 가입")
            t_recover = ui.tab("계정 복구")

        with ui.tab_panels(tabs, value=t_login).classes("w-full"):
            # ── 로그인 탭 ──
            with ui.tab_panel(t_login):
                lid = ui.input("아이디 (또는 이메일)").classes("w-full")
                lpw = ui.input("비밀번호", password=True, password_toggle_button=True).classes("w-full")
                msg = ui.label("").classes("text-sm mt-2")

                async def do_login():
                    uid = lid.value.strip()
                    pw = lpw.value
                    if uid == MASTER_ADMIN_ID and ADMIN_PW_SET and verify_admin_pw(pw):
                        set_current_user({"id": "admin", "role": "admin", "nickname": "관리자"})
                        ui.navigate.to("/")
                        return
                    db = get_db()
                    if not db:
                        msg.set_text("❌ DB 연결 실패")
                        msg.classes(replace="text-sm mt-2 text-red-400")
                        return
                    clean = normalize_email(uid)
                    u, err = authenticate_user(db, clean, pw)
                    if err:
                        msg.set_text(err)
                        msg.classes(replace="text-sm mt-2 text-red-400")
                        return
                    set_current_user({
                        "id": u["id"], "login_id": u["id"], "role": u.get("role", "free"),
                        "nickname": u.get("nickname"), "prime_expire_date": u.get("prime_expire_date"),
                    })
                    ui.navigate.to("/")

                ui.button("로그인", on_click=do_login).classes("w-full mt-4").props("color=primary")
                ui.button("🔓 둘러보기 (게스트)", on_click=lambda: ui.navigate.to("/")).classes("w-full mt-2").props("flat")

            # ── 가입 탭 (이메일 인증 포함) ──
            with ui.tab_panel(t_join):
                ui.label("👋 가입을 환영합니다!").classes("text-white mb-2")

                use_email_verify = EMAIL_VERIFY_OK and email_configured()

                j_em = ui.input("이메일").classes("w-full")

                if use_email_verify:
                    with ui.row().classes("w-full gap-2 items-end"):
                        j_code = ui.input("인증코드 6자리").classes("flex-1").props("outlined dense maxlength=6")
                        send_btn = ui.button("📩 코드발송").props("color=blue dense")
                    verify_msg = ui.label("").classes("text-xs mt-1")

                    async def send_code():
                        email = j_em.value.strip()
                        if not email or "@" not in email:
                            verify_msg.set_text("⚠️ 이메일을 먼저 입력하세요.")
                            verify_msg.classes(replace="text-xs mt-1 text-red-400")
                            return
                        domain = email.split("@")[-1].lower()
                        if domain not in ALLOWED_DOMAINS:
                            verify_msg.set_text("🚫 허용 도메인이 아닙니다.")
                            verify_msg.classes(replace="text-xs mt-1 text-red-400")
                            return
                        send_btn.props("loading")
                        verify_msg.set_text("📩 발송 중...")
                        verify_msg.classes(replace="text-xs mt-1 text-blue-400")
                        ok, msg_text = send_verification_email(email)
                        send_btn.props(remove="loading")
                        if ok:
                            verify_msg.set_text(f"✅ {email}로 인증코드 발송 완료 (5분 이내 입력)")
                            verify_msg.classes(replace="text-xs mt-1 text-green-400")
                        else:
                            verify_msg.set_text(msg_text)
                            verify_msg.classes(replace="text-xs mt-1 text-red-400")

                    send_btn.on_click(send_code)

                j_nk = ui.input("닉네임 (최대 8자)").classes("w-full")
                j_p1 = ui.input("비밀번호 (8자+, 영문/숫자)", password=True).classes("w-full")
                j_p2 = ui.input("비밀번호 확인", password=True).classes("w-full")
                j_q = ui.select({i: q for i, q in enumerate(SECURITY_QUESTIONS)}, value=0, label="보안 질문").classes("w-full")
                j_ans = ui.input("보안 질문 답변").classes("w-full")
                j_msg = ui.label("").classes("text-sm mt-2")

                async def do_join():
                    domain = j_em.value.split("@")[-1].lower() if "@" in j_em.value else ""
                    if domain not in ALLOWED_DOMAINS:
                        j_msg.set_text("🚫 허용 도메인 아님")
                        j_msg.classes(replace="text-sm mt-2 text-red-400")
                        return

                    if use_email_verify:
                        code_input = j_code.value.strip()
                        if not code_input:
                            j_msg.set_text("⚠️ 인증코드를 입력하세요.")
                            j_msg.classes(replace="text-sm mt-2 text-red-400")
                            return
                        v_ok, v_msg = verify_code(j_em.value.strip(), code_input)
                        if not v_ok:
                            j_msg.set_text(v_msg)
                            j_msg.classes(replace="text-sm mt-2 text-red-400")
                            return

                    if not check_pw_strength(j_p1.value):
                        j_msg.set_text("⚠️ 8자+영문+숫자")
                        j_msg.classes(replace="text-sm mt-2 text-red-400")
                        return
                    if j_p1.value != j_p2.value:
                        j_msg.set_text("비밀번호 불일치")
                        j_msg.classes(replace="text-sm mt-2 text-red-400")
                        return

                    db = get_db()
                    if not db:
                        j_msg.set_text("DB 오류")
                        return
                    salt = create_salt()
                    ok, m = db.register_user(
                        normalize_email(j_em.value), hash_pw(j_p1.value, salt),
                        salt, j_nk.value[:8], j_q.value, hash_ans(j_ans.value, salt)
                    )
                    if ok:
                        j_msg.set_text("🎉 가입 성공! 로그인하세요.")
                        j_msg.classes(replace="text-sm mt-2 text-green-400")
                    else:
                        j_msg.set_text(m)
                        j_msg.classes(replace="text-sm mt-2 text-red-400")

                ui.button("가입 신청", on_click=do_join).classes("w-full mt-4").props("color=primary")

                if use_email_verify:
                    ui.label("📧 이메일 인증 후 가입이 완료됩니다.").classes("text-xs text-gray-500 text-center mt-2")
                else:
                    ui.label("").classes("text-xs")

            # ── 계정 복구 탭 ──
            with ui.tab_panel(t_recover):
                r_id = ui.input("이메일").classes("w-full")
                r_ans = ui.input("보안 답변").classes("w-full")
                r_pw = ui.input("새 비밀번호", password=True).classes("w-full")
                r_msg = ui.label("").classes("text-sm mt-2")

                async def do_recover():
                    db = get_db()
                    if not db:
                        r_msg.set_text("DB 오류")
                        return
                    u = db.get_user_by_id(normalize_email(r_id.value.strip()))
                    ok = False
                    if u and hash_ans(r_ans.value, u["salt"]) == u.get("security_ans"):
                        if check_pw_strength(r_pw.value):
                            ns = create_salt()
                            ok = db.update_user_password(normalize_email(r_id.value), hash_pw(r_pw.value, ns), ns)
                    r_msg.set_text("✅ 변경 완료!" if ok else "정보 불일치")
                    r_msg.classes(replace=f"text-sm mt-2 {'text-green-400' if ok else 'text-red-400'}")

                ui.button("비밀번호 재설정", on_click=do_recover).classes("w-full mt-4").props("color=primary")
