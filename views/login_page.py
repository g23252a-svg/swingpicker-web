# -*- coding: utf-8 -*-
"""
login_page.py — 🔐 로그인 / 가입 / 계정 복구 페이지
═══════════════════════════════════════════════════
가입 시 이메일 인증코드 검증 + 14일 Prime 자동 부여
"""
from datetime import datetime, timedelta
from nicegui import ui

from components.ui_utils import DARK_CSS
from services.auth import (
    MASTER_ADMIN_ID, ADMIN_PW_SET, SECURITY_QUESTIONS, ALLOWED_DOMAINS,
    get_db, verify_admin_pw, normalize_email, check_pw_strength,
    authenticate_user, set_current_user,
    create_salt, hash_pw, hash_ans,
)

# ── 이메일 인증 (미설정 시 fallback) ──
try:
    from email_verify import send_verification_email, verify_code, is_configured as email_configured
except ImportError:
    email_configured = lambda: False
    send_verification_email = lambda e: (False, "미설정")
    verify_code = lambda e, c: (False, "미설정")

TRIAL_DAYS = 14  # 신규 가입 무료체험 일수


@ui.page('/login')
def login_page():
    ui.add_head_html(DARK_CSS.replace("1400px", "500px"))

    with ui.card().classes("w-full p-8 bg-[#1a1a2e] border border-gray-700 rounded-2xl mt-16"):
        ui.label("🔐 SwingPicker").classes("text-2xl font-bold text-center text-white w-full mb-4")

        with ui.tabs().classes("w-full") as tabs:
            t_login = ui.tab("로그인")
            t_join = ui.tab("회원가입")
            t_recover = ui.tab("계정 복구")

        with ui.tab_panels(tabs, value=t_login).classes("w-full"):
            # ══════════════════════════════════════
            #  로그인
            # ══════════════════════════════════════
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

            # ══════════════════════════════════════
            #  회원가입 (이메일 인증 + 14일 Prime)
            # ══════════════════════════════════════
            with ui.tab_panel(t_join):
                ui.label("👋 가입을 환영합니다!").classes("text-white mb-2")

                use_verify = email_configured()

                if use_verify:
                    ui.label("📧 이메일 인증 후 가입이 완료됩니다.").classes("text-gray-400 text-xs mb-2")

                # ── Step 1: 이메일 + 인증 ──
                j_em = ui.input("이메일", placeholder="example@gmail.com").classes("w-full")

                verified = {"done": False}  # 인증 완료 여부

                if use_verify:
                    verify_row = ui.row().classes("w-full gap-2 items-center")
                    code_section = ui.column().classes("w-full")
                    code_section.set_visibility(False)

                    with verify_row:
                        send_msg = ui.label("").classes("text-xs")

                        async def send_code():
                            email = j_em.value.strip()
                            if not email or "@" not in email:
                                ui.notify("이메일을 입력하세요.", type="warning")
                                return
                            domain = email.split("@")[-1].lower()
                            if domain not in ALLOWED_DOMAINS:
                                ui.notify("🚫 허용되지 않는 이메일 도메인입니다.", type="warning")
                                return
                            ok, m = send_verification_email(email)
                            send_msg.set_text(m)
                            send_msg.classes(replace=f"text-xs {'text-green-400' if ok else 'text-red-400'}")
                            if ok:
                                code_section.set_visibility(True)
                                ui.notify("📩 인증코드가 발송되었습니다!", type="positive")

                        ui.button("📩 코드발송", on_click=send_code).props("color=primary dense size=sm")

                    with code_section:
                        j_code = ui.input("인증코드 (6자리)", placeholder="123456").classes("w-full")
                        verify_msg = ui.label("").classes("text-xs")

                        async def check_code():
                            ok, m = verify_code(j_em.value.strip(), j_code.value.strip())
                            verify_msg.set_text(m)
                            verify_msg.classes(replace=f"text-xs {'text-green-400' if ok else 'text-red-400'}")
                            if ok:
                                verified["done"] = True
                                ui.notify("✅ 이메일 인증 완료!", type="positive")

                        ui.button("✅ 인증확인", on_click=check_code).props("color=green dense size=sm")

                # ── Step 2: 나머지 정보 ──
                j_nk = ui.input("닉네임 (최대 8자)").classes("w-full")
                j_p1 = ui.input("비밀번호 (8자+, 영문/숫자)", password=True).classes("w-full")
                j_p2 = ui.input("비밀번호 확인", password=True).classes("w-full")
                j_q = ui.select({i: q for i, q in enumerate(SECURITY_QUESTIONS)}, value=0, label="보안 질문").classes("w-full")
                j_ans = ui.input("보안 질문 답변").classes("w-full")
                j_msg = ui.label("").classes("text-sm mt-2")

                async def do_join():
                    email = j_em.value.strip()
                    domain = email.split("@")[-1].lower() if "@" in email else ""
                    if domain not in ALLOWED_DOMAINS:
                        j_msg.set_text("🚫 허용 도메인 아님")
                        j_msg.classes(replace="text-sm mt-2 text-red-400")
                        return

                    # 이메일 인증 확인
                    if use_verify and not verified["done"]:
                        j_msg.set_text("📧 이메일 인증을 먼저 완료해주세요.")
                        j_msg.classes(replace="text-sm mt-2 text-yellow-400")
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

                    clean_email = normalize_email(email)
                    salt = create_salt()
                    ok, m = db.register_user(
                        clean_email, hash_pw(j_p1.value, salt),
                        salt, j_nk.value[:8], j_q.value, hash_ans(j_ans.value, salt)
                    )
                    if ok:
                        # ✅ 가입 성공 → 14일 Prime 자동 부여
                        try:
                            expire = (datetime.now() + timedelta(days=TRIAL_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
                            db.update_user_subscription(clean_email, "prime", expire)
                            j_msg.set_text(f"🎉 가입 성공! {TRIAL_DAYS}일 Prime 체험권이 지급되었습니다. 로그인하세요!")
                        except Exception:
                            j_msg.set_text("🎉 가입 성공! 로그인하세요.")
                        j_msg.classes(replace="text-sm mt-2 text-green-400")
                    else:
                        j_msg.set_text(m)
                        j_msg.classes(replace="text-sm mt-2 text-red-400")

                ui.button("가입 신청", on_click=do_join).classes("w-full mt-4").props("color=primary")

                # ── 무료체험 안내 ──
                with ui.row().classes("w-full mt-3 justify-center"):
                    ui.label(f"🎁 신규 가입 시 {TRIAL_DAYS}일 Prime 무료체험 자동 지급!").classes("text-amber-400 text-xs")

            # ══════════════════════════════════════
            #  계정 복구
            # ══════════════════════════════════════
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
                    if u and hash_ans(r_ans.value, u["salt"]) == u.get("security_a_hash"):
                        if check_pw_strength(r_pw.value):
                            ns = create_salt()
                            ok = db.update_user_password(normalize_email(r_id.value), hash_pw(r_pw.value, ns), ns)
                    r_msg.set_text("✅ 변경 완료!" if ok else "정보 불일치")
                    r_msg.classes(replace=f"text-sm mt-2 {'text-green-400' if ok else 'text-red-400'}")

                ui.button("비밀번호 재설정", on_click=do_recover).classes("w-full mt-4").props("color=primary")
