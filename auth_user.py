# auth_user.py
# -*- coding: utf-8 -*-

import os
import json
import hashlib
import logging
import re
from typing import Tuple, Optional
from datetime import datetime, timezone

import requests
import streamlit as st
import secrets

# ----------------- 로깅 설정 -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("auth_user")

# 🔹 auth_user.py가 있는 폴더 기준으로 data 폴더 고정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USER_DB_PATH = os.path.join(DATA_DIR, "users_db.json")

CURRENT_USER_KEY = "ldy_current_user"
JUST_REGISTERED_KEY = "just_registered"

# ----------------- Secrets / Env 우선 -----------------
def _get_conf(key: str, default_val: str) -> str:
    """
    Streamlit secrets > 환경변수 > 기본값 순으로 설정값을 가져온다.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except FileNotFoundError:
        # 로컬 환경에서 .streamlit/secrets.toml 이 없을 수도 있음
        pass
    return os.getenv(key, default_val)

# 구독/관리용 키
KEY_PRO   = _get_conf("LDY_KEY_PRO",   "220577")
KEY_PRIME = _get_conf("LDY_KEY_PRIME", "577220")
ADMIN_KEY = _get_conf("LDY_ADMIN_KEY", "2022322")

# 🔹 Gist 관련 설정 (Streamlit secrets 또는 환경변수에서 읽음)
GIST_ID    = _get_conf("LDY_GIST_ID", "")
GIST_TOKEN = _get_conf("LDY_GIST_TOKEN", "")

# 디버그 로그 (Streamlit Cloud / Render 로그에서 확인용)
print("[auth_user] DEBUG GIST_ID =", GIST_ID)
print("[auth_user] DEBUG GIST_TOKEN set?", bool(GIST_TOKEN))

# ----------------- 공통 유틸 -----------------
def _now_utc_str() -> str:
    """
    DB에 저장용 공통 시간 포맷
    - 항상 UTC 기준, ISO8601 문자열로 저장
    - 예: 2025-12-04T10:22:11.123456+00:00
    """
    return datetime.now(timezone.utc).isoformat()

def _normalize_email(email: str) -> str:
    """
    이메일 키는 무조건 lowercase + trim
    """
    return (email or "").strip().lower()

# ----------------- 기본 로컬 DB 유틸 -----------------
def _ensure_data_dir() -> None:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

def _normalize_user_db_structure(db_raw: dict) -> dict:
    """
    - {"users": {...}} 형태든, 바로 {"a@b.com": {...}} 이든 모두 통일
    - 이메일 키는 전부 lower()
    - created_at / last_login / role / login_id 기본값도 보정
    - 대소문자 다른 중복 계정은 last_login 이 더 최신인 쪽을 남김
    """
    if not isinstance(db_raw, dict):
        return {"users": {}}

    raw_users = db_raw.get("users")
    if raw_users is None or not isinstance(raw_users, dict):
        raw_users = db_raw

    normalized_users = {}
    for key, user in raw_users.items():
        if not isinstance(user, dict):
            continue

        email_norm = _normalize_email(key)
        if not email_norm:
            continue

        u = dict(user)  # shallow copy

        # 필수 필드 보정
        u["login_id"] = _normalize_email(u.get("login_id", email_norm))
        u.setdefault("role", "free")

        now_utc = _now_utc_str()
        created = u.get("created_at")
        last = u.get("last_login")

        if not created and last:
            created = last
        elif not created:
            created = now_utc

        if not last:
            last = created

        u["created_at"] = created
        u["last_login"] = last

        # 닉네임 기본값
        if not u.get("nickname"):
            u["nickname"] = email_norm.split("@")[0]

        # 이미 같은 키가 있으면 last_login 기준으로 더 최신 것 사용
        existing = normalized_users.get(email_norm)
        if existing:
            if u["last_login"] > existing.get("last_login", ""):
                normalized_users[email_norm] = u
        else:
            normalized_users[email_norm] = u

    return {"users": normalized_users}

def _load_user_db_local() -> dict:
    """
    로컬 파일(users_db.json)에서 DB 로드 + 구조 정규화
    """
    _ensure_data_dir()
    if not os.path.exists(USER_DB_PATH):
        return {"users": {}}
    try:
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        db = _normalize_user_db_structure(raw)
        return db
    except Exception:
        logger.exception("[auth_user] local user DB load failed, return empty")
        return {"users": {}}

def _save_user_db_local(db: dict) -> None:
    """
    로컬 파일(users_db.json)에 DB 저장
    """
    _ensure_data_dir()
    try:
        with open(USER_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("[auth_user] local user DB save failed")

# ----------------- GitHub Gist 연동 유틸 -----------------
GIST_FILE_NAME = "users_db.json"   # Gist 안에 만들 파일 이름

def _extract_json_from_text(content: str) -> str:
    """
    Gist 파일에 'LDY Pro Trader user DB' 같은 설명 텍스트가 앞에 붙어 있어도
    실제 JSON 블록만 잘라내서 파싱할 수 있게 처리.
    """
    if not content:
        return "{}"
    s = content.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # JSON 형태가 아니면 빈 오브젝트로 처리
        return "{}"
    return s[start : end + 1]

def _load_user_db_from_gist() -> Optional[dict]:
    """
    GitHub Gist에서 users_db.json 내용을 읽어온다.
    - GIST_ID / GIST_TOKEN 없으면 None 리턴 (로컬로 fallback)
    """
    if not GIST_ID or not GIST_TOKEN:
        return None

    try:
        url = f"https://api.github.com/gists/{GIST_ID}"
        headers = {
            "Authorization": f"token {GIST_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        gist = resp.json()
        files = gist.get("files", {})
        file_obj = files.get(GIST_FILE_NAME)

        if not file_obj:
            logger.info("[auth_user] Gist에 '%s' 파일이 없어 빈 DB로 초기화합니다.", GIST_FILE_NAME)
            return {"users": {}}

        content = file_obj.get("content", "")
        if not content or not content.strip():
            return {"users": {}}

        json_text = _extract_json_from_text(content)
        raw = json.loads(json_text)
        db = _normalize_user_db_structure(raw)

        logger.info("[auth_user] Gist에서 user DB 로드 완료 (users=%d)", len(db.get("users", {})))
        return db

    except Exception as e:
        logger.exception("[auth_user] Gist user DB load 실패: %s", e)
        return None

def _save_user_db_to_gist(db: dict) -> bool:
    """
    GitHub Gist에 users_db.json 내용을 저장한다.
    - 실패해도 예외 던지지 않고 False 리턴
    """
    if not GIST_ID or not GIST_TOKEN:
        return False

    try:
        url = f"https://api.github.com/gists/{GIST_ID}"
        headers = {
            "Authorization": f"token {GIST_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        files = {
            GIST_FILE_NAME: {
                "content": json.dumps(db, ensure_ascii=False, indent=2)
            }
        }
        payload = {"files": files}

        resp = requests.patch(url, headers=headers, data=json.dumps(payload), timeout=10)
        resp.raise_for_status()
        logger.info("[auth_user] Gist에 user DB 저장 완료 (users=%d)", len(db.get("users", {})))
        return True

    except Exception as e:
        logger.exception("[auth_user] Gist user DB save 실패: %s", e)
        return False

# ----------------- 통합 DB 유틸 (Gist 우선, 로컬 백업) -----------------
# 👉 Gist API 호출 최적화를 위한 간단 캐시 (TTL)
_USER_DB_CACHE: Optional[dict] = None
_USER_DB_CACHE_TS: Optional[float] = None
_USER_DB_CACHE_TTL = 30  # 초 단위 (예: 30초)

def load_user_db() -> dict:
    """
    1순위: Gist에서 로드 (캐시 + TTL)
    2순위: 로컬 파일에서 로드
    """
    global _USER_DB_CACHE, _USER_DB_CACHE_TS

    now_ts = datetime.now(timezone.utc).timestamp()

    # 캐시가 있고, TTL 이내면 그대로 사용
    if _USER_DB_CACHE is not None and _USER_DB_CACHE_TS is not None:
        if now_ts - _USER_DB_CACHE_TS < _USER_DB_CACHE_TTL:
            return _USER_DB_CACHE

    # Gist 우선 시도
    db = _load_user_db_from_gist()
    if db is not None:
        # 구조 정규화는 _load_user_db_from_gist 내부에서 이미 수행
        _save_user_db_local(db)
        _USER_DB_CACHE = db
        _USER_DB_CACHE_TS = now_ts
        return db

    # Gist 실패 → 로컬
    logger.info("[auth_user] Gist 사용 불가, 로컬 user DB 사용")
    db = _load_user_db_local()
    _USER_DB_CACHE = db
    _USER_DB_CACHE_TS = now_ts
    return db

def save_user_db(db: dict) -> None:
    """
    - 항상 로컬에는 저장
    - Gist 설정이 되어 있으면 Gist에도 저장
    - 모듈 캐시도 항상 최신으로 갱신
    """
    global _USER_DB_CACHE, _USER_DB_CACHE_TS

    # 구조 정규화(혹시라도 잘못된 형태가 들어온 경우 대비)
    db = _normalize_user_db_structure(db)

    # 캐시 갱신
    _USER_DB_CACHE = db
    _USER_DB_CACHE_TS = datetime.now(timezone.utc).timestamp()

    # 로컬 백업
    _save_user_db_local(db)

    # Gist 동기화
    if GIST_ID and GIST_TOKEN:
        ok = _save_user_db_to_gist(db)
        if not ok:
            logger.warning("[auth_user] Gist 저장 실패, 로컬 파일만 최신 상태입니다.")

# ----------------- 비밀번호 해시 -----------------
def _create_salt(email: str) -> str:
    """
    비밀번호 해시에 사용할 솔트 생성
    - email은 굳이 섞지 않아도 되고, 독립 랜덤 값이면 충분
    """
    return secrets.token_hex(16)  # 32자리 hex 문자열 (128bit)

def _hash_password(password: str, salt: str) -> str:
    # 필요 시 sha256 반복 횟수를 늘려 강도 강화도 가능
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

# ----------------- 입력 검증 유틸 -----------------
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _validate_email(email: str) -> bool:
    if not email:
        return False
    return bool(EMAIL_REGEX.match(email))

def _validate_password(pw: str) -> Tuple[bool, str]:
    if len(pw) < 6:
        return False, "비밀번호는 최소 6자 이상이어야 합니다."
    return True, ""

# ----------------- 회원 가입 / 로그인 -----------------
def register_user(email: str, password: str, nickname: str, invite_code: str = ""):
    email_norm = _normalize_email(email)
    if not email_norm or not password:
        return False, "이메일 / 비밀번호를 입력해 주세요.", None

    if not _validate_email(email_norm):
        return False, "이메일 형식이 올바르지 않습니다.", None

    ok_pw, pw_msg = _validate_password(password)
    if not ok_pw:
        return False, pw_msg, None

    db = load_user_db()
    users = db.get("users", {})

    if email_norm in users:
        return False, "이미 존재하는 이메일입니다.", None

    # 기본 권한
    role = "free"
    invite_code = (invite_code or "").strip()

    if invite_code == ADMIN_KEY:
        role = "admin"
    elif invite_code == KEY_PRIME:
        role = "prime"
    elif invite_code == KEY_PRO:
        role = "pro"

    salt = _create_salt(email_norm)
    pw_hash = _hash_password(password, salt)
    now_utc = _now_utc_str()

    users[email_norm] = {
        "login_id": email_norm,
        "nickname": nickname or email_norm.split("@")[0],
        "role": role,
        "salt": salt,
        "password_hash": pw_hash,
        "created_at": now_utc,
        "last_login": now_utc,   # 가입 시점 = 첫 로그인 시점
    }

    db["users"] = users
    save_user_db(db)

    logger.info("[auth_user] 신규 회원가입: %s (role=%s)", email_norm, role)
    return True, f"회원가입 완료! 현재 권한: {role}", users[email_norm]

def authenticate_user(email: str, password: str):
    email_norm = _normalize_email(email)
    db = load_user_db()
    users = db.get("users", {})
    user = users.get(email_norm)

    # 보안 관점에서 이메일/비번 에러 메시지 통합
    if not user:
        return None, "이메일 또는 비밀번호가 일치하지 않습니다."

    salt = user.get("salt", "")
    pw_hash = user.get("password_hash", "")

    if _hash_password(password, salt) != pw_hash:
        return None, "이메일 또는 비밀번호가 일치하지 않습니다."

    # 로그인 성공 시점에 last_login을 UTC 기준으로 갱신
    now_str = _now_utc_str()
    user["last_login"] = now_str
    users[email_norm] = user
    db["users"] = users
    save_user_db(db)

    logger.info("[auth_user] 로그인 성공: %s", email_norm)
    return user, "로그인 성공"

def get_user():
    return st.session_state.get(CURRENT_USER_KEY)

# ----------------- Admin용 헬퍼 (dashboard.py에서 사용) -----------------
def list_users():
    """
    전체 유저 목록 리스트 반환 (dashboard용)
    """
    db = load_user_db()
    users = db.get("users", {})
    # 이메일 기준 정렬
    return [users[k] for k in sorted(users.keys())]

def update_user_role(email: str, new_role: str, acting_admin_email: Optional[str] = None) -> bool:
    """
    관리자가 회원 권한 변경
    - acting_admin_email: 실제 권한 변경을 수행하는 관리자 이메일
      (자기 자신을 free/prime 등으로 떨어뜨리는 실수 방지용)
    """
    email_norm = _normalize_email(email)
    if acting_admin_email:
        acting_admin_email = _normalize_email(acting_admin_email)

    db = load_user_db()
    users = db.get("users", {})
    if email_norm not in users:
        return False

    # ⛔ 자기 자신(admin)이 자기 권한을 admin이 아닌 걸로 바꾸는 것을 막기
    if acting_admin_email == email_norm:
        current_role = users[email_norm].get("role", "free")
        if current_role == "admin" and new_role != "admin":
            logger.warning("[auth_user] admin self-downgrade 차단: %s -> %s", email_norm, new_role)
            return False

    users[email_norm]["role"] = new_role
    db["users"] = users
    save_user_db(db)

    logger.info("[auth_user] 권한 변경: %s → %s", email_norm, new_role)
    return True

# ----------------- UI: 로그인 / 회원가입 박스 -----------------
def _render_admin_panel(current_user):
    """
    (선택) auth_user 내부에서도 간단 관리 패널 제공 가능
    """
    db = load_user_db()
    users = db.get("users", {})

    st.markdown("---")
    st.subheader("🛠 회원 권한 관리 (Admin - auth_user)")

    if not users:
        st.info("등록된 계정이 없습니다.")
        return

    email_list = sorted(users.keys())
    target_email = st.selectbox("계정 선택", email_list)
    if not target_email:
        return

    target_user = users[target_email]
    current_role = target_user.get("role", "free")
    st.write(f"선택된 계정: `{target_email}` (현재 권한: `{current_role}`)")

    role_options = ["free", "pro", "prime", "admin"]
    idx = role_options.index(current_role) if current_role in role_options else 0
    new_role = st.radio(
        "새 권한",
        role_options,
        index=idx,
        horizontal=True,
        key=f"admin_role_{target_email}",
    )

    if st.button("권한 변경 적용", key=f"admin_apply_{target_email}"):
        current_admin = current_user.get("login_id") if current_user else None
        ok = update_user_role(target_email, new_role, acting_admin_email=current_admin)
        if ok:
            st.success(f"{target_email} 권한이 `{new_role}` 로 변경되었습니다.")
        else:
            st.error("변경 실패 (자기 자신 admin 권한을 낮출 수는 없습니다.)")

def render_auth_box(show_debug: bool = False):
    """
    사이드바 로그인 / 회원가입 UI
    - show_debug=True 로 호출하면 Gist 연동 디버그 캡션을 표시
      (예: 개발 시에만 사용)
    """
    # 세션 키 초기화
    if CURRENT_USER_KEY not in st.session_state:
        st.session_state[CURRENT_USER_KEY] = None

    if JUST_REGISTERED_KEY not in st.session_state:
        st.session_state[JUST_REGISTERED_KEY] = False

    # 🔍 DB 디버그용 (Gist 연동 확인용) - 옵션
    if show_debug:
        try:
            _db = load_user_db()
            st.caption(
                f"DEBUG: GIST_ID={GIST_ID[:8]}..., "
                f"users={len(_db.get('users', {}))}, "
                f"from_gist={bool(GIST_ID and GIST_TOKEN)}"
            )
        except Exception as e:
            st.caption(f"DEBUG: load_user_db error = {e}")

    st.subheader("🔐 계정 로그인 / 회원가입")
    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    # ---------------- 로그인 탭 ----------------
    with tab_login:
        login_email = st.text_input("이메일", key="login_email")
        login_pw = st.text_input("비밀번호", type="password", key="login_pw")
        if st.button("로그인", key="btn_login"):
            user, msg = authenticate_user(login_email, login_pw)
            if user is None:
                st.error(msg)
            else:
                st.session_state[CURRENT_USER_KEY] = user
                st.success(f"{user['nickname']}님 환영합니다! ({user['role']})")
                # 로그인 시에는 첫 가입 플래그 초기화
                st.session_state[JUST_REGISTERED_KEY] = False

    # ---------------- 회원가입 탭 ----------------
    with tab_signup:
        reg_email = st.text_input("이메일", key="reg_email")
        reg_nick = st.text_input("닉네임 (선택)", key="reg_nick")
        reg_pw1 = st.text_input("비밀번호", type="password", key="reg_pw1")
        reg_pw2 = st.text_input("비밀번호 확인", type="password", key="reg_pw2")
        reg_code = st.text_input(
            "초대/구독 코드 (선택 - Pro/Prime/Admin 용)",
            type="password",
            key="reg_code",
        )

        # 💡 회원가입 전 혜택 홍보 문구
        st.info("✅ 지금 회원가입하면 **오늘 기준 상위 5개 추천 종목**까지 무료로 확인할 수 있습니다.")

        if st.button("회원가입", key="btn_register"):
            if reg_pw1 != reg_pw2:
                st.error("비밀번호가 서로 일치하지 않습니다.")
            else:
                ok, msg, new_user = register_user(
                    reg_email, reg_pw1, reg_nick, reg_code
                )
                if ok:
                    st.success(msg)
                    st.session_state[CURRENT_USER_KEY] = new_user
                    # 🔹 첫 가입 여부 플래그 ON
                    st.session_state[JUST_REGISTERED_KEY] = True
                else:
                    st.error(msg)

    # ---------------- 공통 로그인 상태 표시 ----------------
    user = get_user()
    if user:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"👤 {user['nickname']}님 로그인 중 (권한: `{user['role']}`)")
        with col2:
            if st.button("로그아웃", key="btn_logout"):
                st.session_state[CURRENT_USER_KEY] = None
                st.session_state[JUST_REGISTERED_KEY] = False
                st.success("로그아웃 되었습니다.")
                user = None

    return user
