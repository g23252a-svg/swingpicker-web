#!/usr/bin/env python3
"""
tests/test_auth_service.py — 실제 services.auth 함수 통합 테스트
═══════════════════════════════════════════════════════════════════════
v2.3 신규: 평가자의 핵심 지적("Auth 테스트가 실제 services.auth를 검증하지
않음")에 대응. test_auth_migration.py는 bcrypt 라이브러리 동작을 검증하고,
이 파일은 실제 운영 함수 authenticate_user, compute_access_status를 직접
호출해서 회귀를 차단합니다.

v2.4 추가: 실제 bcrypt 자동 마이그레이션 회귀 테스트.

실행:
  python -m pytest tests/test_auth_service.py -v
"""
from __future__ import annotations
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# bcrypt cost를 테스트용으로 낮춤 (services.auth가 bcrypt.hashpw 사용)
os.environ.setdefault("BCRYPT_COST", "4")

# nicegui / db_utils 가 로컬에 없을 수 있으니 mock으로 우회 — 우리가 테스트할
# authenticate_user / compute_access_status는 모듈 import 시점 외에는
# nicegui를 쓰지 않습니다.
sys.modules.setdefault("nicegui", MagicMock())
sys.modules.setdefault("db_utils", MagicMock())

import bcrypt  # noqa: E402

from services.auth import (  # noqa: E402  (위 mock 후 import 필수)
    authenticate_user,
    compute_access_status,
    create_salt,
    hash_pw,
    verify_password,
    _is_bcrypt_hash,
    hash_password_bcrypt,
    _DUMMY_BCRYPT_HASH,
    _dummy_password_check,
)


# 날짜 테스트 안정화: fixed_now를 통일 — datetime.now() 직접 사용 회피
FIXED_NOW = datetime(2026, 4, 29, 12, 0, 0)


class FakeDB:
    """authenticate_user가 기대하는 최소 DB 인터페이스."""

    def __init__(self, users: dict | None = None):
        self.users = users or {}
        self.timestamp_calls: list[str] = []
        self.update_should_raise: Exception | None = None

    def get_user_by_id(self, email: str):
        return self.users.get(email)

    def update_login_timestamp(self, email: str):
        self.timestamp_calls.append(email)
        if self.update_should_raise:
            raise self.update_should_raise


class FakeDBWithMigration(FakeDB):
    """update_user_password 메서드를 추가로 지원하는 DB — bcrypt 마이그레이션 검증용.
    실제 db_utils.SwingPickerDB와 동일한 인터페이스: update_user_password(email, pw_hash, salt).
    """

    def __init__(self, users: dict | None = None):
        super().__init__(users)
        self.password_updates: list[tuple[str, str, str]] = []

    def update_user_password(self, email: str, new_hash: str, salt: str = ""):
        if email in self.users:
            self.users[email]["password"] = new_hash
            self.users[email]["salt"] = salt
        self.password_updates.append((email, new_hash, salt))


def make_user(email="user@example.com", password="testpass123",
              role="free", is_banned="N", expire=None):
    """services.auth가 기대하는 user_row 형태 (legacy pbkdf2 hash)."""
    salt = create_salt()
    row = {
        "email": email,
        "salt": salt,
        "password": hash_pw(password, salt),
        "role": role,
        "is_banned": is_banned,
    }
    if expire is not None:
        row["prime_expire_date"] = expire
    return row


def make_bcrypt_user(email="modern@example.com", password="testpass123",
                     role="free", is_banned="N", expire=None):
    """이미 bcrypt로 저장된 신규 형식 유저."""
    row = {
        "email": email,
        "salt": "",  # bcrypt는 salt 분리 안 함
        "password": bcrypt.hashpw(
            password.encode(),
            bcrypt.gensalt(int(os.environ.get("BCRYPT_COST", "4"))),
        ).decode(),
        "role": role,
        "is_banned": is_banned,
    }
    if expire is not None:
        row["prime_expire_date"] = expire
    return row


# ═══════════════════════════════════════════════════════════════════════
# authenticate_user — 실제 운영 로그인 함수
# ═══════════════════════════════════════════════════════════════════════

class TestAuthenticateUser(unittest.TestCase):
    """services.auth.authenticate_user를 직접 호출하는 회귀 테스트 (legacy pbkdf2)."""

    def setUp(self):
        self.password = "validPass123!"
        self.user = make_user(email="alice@example.com", password=self.password)
        self.db = FakeDB({"alice@example.com": self.user})

    def test_correct_password_succeeds(self):
        user, err = authenticate_user(self.db, "alice@example.com", self.password)
        self.assertIsNotNone(user)
        self.assertIsNone(err)
        self.assertEqual(user["email"], "alice@example.com")

    def test_wrong_password_rejected(self):
        user, err = authenticate_user(self.db, "alice@example.com", "wrongPass")
        self.assertIsNone(user)
        self.assertIsNotNone(err)
        self.assertIn("오류", err)

    def test_nonexistent_user_rejected(self):
        user, err = authenticate_user(self.db, "ghost@example.com", "anything")
        self.assertIsNone(user)
        self.assertIsNotNone(err)
        # 같은 메시지 — username enumeration 방어
        self.assertIn("오류", err)

    def test_banned_user_blocked(self):
        """ban 유저는 비밀번호가 맞아도 차단 — 보안 회귀 차단의 핵심."""
        self.user["is_banned"] = "Y"
        user, err = authenticate_user(self.db, "alice@example.com", self.password)
        self.assertIsNone(user)
        self.assertIsNotNone(err)
        self.assertIn("차단", err)

    def test_banned_variants(self):
        """ban 플래그의 다양한 표현(Y/TRUE/1)을 모두 차단."""
        for ban_value in ["Y", "y", "TRUE", "true", "True", "1"]:
            with self.subTest(ban_value=ban_value):
                self.user["is_banned"] = ban_value
                user, err = authenticate_user(self.db, "alice@example.com", self.password)
                self.assertIsNone(user, f"{ban_value} 로 차단 실패")

    def test_login_updates_timestamp(self):
        """성공 로그인은 db.update_login_timestamp를 호출."""
        authenticate_user(self.db, "alice@example.com", self.password)
        self.assertIn("alice@example.com", self.db.timestamp_calls)

    def test_failed_login_no_timestamp(self):
        """실패 로그인은 timestamp 안 갱신."""
        authenticate_user(self.db, "alice@example.com", "wrongPass")
        self.assertNotIn("alice@example.com", self.db.timestamp_calls)

    def test_timestamp_failure_does_not_break_login(self):
        """timestamp 갱신이 DB 에러로 실패해도 로그인은 성공해야 함 (degradation)."""
        self.db.update_should_raise = RuntimeError("DB connection lost")
        user, err = authenticate_user(self.db, "alice@example.com", self.password)
        self.assertIsNotNone(user)
        self.assertIsNone(err)


# ═══════════════════════════════════════════════════════════════════════
# v2.6 신규: hash_password_bcrypt 헬퍼 + dummy bcrypt timing defense
# ═══════════════════════════════════════════════════════════════════════

class TestSignupHelpers(unittest.TestCase):
    """[v22.5] 신규 가입/비번변경용 bcrypt 헬퍼 검증."""

    def test_hash_password_bcrypt_creates_bcrypt_format(self):
        """hash_password_bcrypt는 $2b$ 형식의 hash를 만듦."""
        h = hash_password_bcrypt("anyPassword123")
        self.assertTrue(_is_bcrypt_hash(h),
                        "hash_password_bcrypt 결과가 bcrypt 형식이 아님")
        self.assertTrue(h.startswith("$2b$"))

    def test_hash_password_bcrypt_self_verifying(self):
        """hash_password_bcrypt 결과는 bcrypt.checkpw로 검증됨."""
        import bcrypt
        pw = "myStrongPass!2025"
        h = hash_password_bcrypt(pw)
        self.assertTrue(bcrypt.checkpw(pw.encode(), h.encode()))
        self.assertFalse(bcrypt.checkpw(b"wrong", h.encode()))

    def test_hash_password_bcrypt_different_salts(self):
        """같은 비밀번호도 매번 다른 hash (salt 자동 생성)."""
        pw = "samePassword"
        h1 = hash_password_bcrypt(pw)
        h2 = hash_password_bcrypt(pw)
        self.assertNotEqual(h1, h2,
                            "같은 비밀번호인데 hash가 동일 — salt 생성 누락")

    def test_signup_with_bcrypt_can_login(self):
        """[통합] hash_password_bcrypt로 만든 유저는 authenticate_user로 로그인됨."""
        password = "newSignupPass!"
        # 신규 가입 시뮬레이션 — bcrypt hash + salt=""
        user = {
            "email": "newuser@example.com",
            "password": hash_password_bcrypt(password),
            "salt": "",  # 신규 가입은 salt 컬럼 빈 값
            "is_banned": "N",
            "role": "free",
        }
        db = FakeDBWithMigration({user["email"]: user})

        u, err = authenticate_user(db, user["email"], password)
        self.assertIsNotNone(u)
        self.assertIsNone(err)
        # 신규 가입자는 이미 bcrypt이므로 추가 업그레이드 X
        self.assertEqual(len(db.password_updates), 0)


class TestDummyBcryptDefense(unittest.TestCase):
    """[v22.5] 사용자 부재 시 timing 공격 방어 — dummy도 bcrypt."""

    def test_dummy_hash_is_bcrypt_format(self):
        """모듈 레벨 _DUMMY_BCRYPT_HASH는 bcrypt 형식."""
        self.assertTrue(_DUMMY_BCRYPT_HASH.startswith(b"$2b$"))

    def test_dummy_password_check_does_not_raise(self):
        """_dummy_password_check는 어떤 input에도 예외 안 던짐."""
        # 정상 케이스
        _dummy_password_check("any_password")
        # 빈 문자열
        _dummy_password_check("")
        # 매우 긴 문자열
        _dummy_password_check("x" * 1000)
        # 유니코드
        _dummy_password_check("비밀번호🔐")

    def test_nonexistent_user_does_not_invoke_legacy_path(self):
        """사용자 부재 시 hash_pw(pbkdf2)가 아닌 bcrypt path를 탐.

        과거(v2.5 이전): hash_pw(password, "dummy") — pbkdf2
        현재(v2.6+):     bcrypt.checkpw(password, _DUMMY_BCRYPT_HASH)

        이 테스트는 hash_pw가 호출되지 않는 것을 mock으로 검증.
        """
        from unittest.mock import patch
        db = FakeDBWithMigration({})  # 빈 DB

        with patch("services.auth.hash_pw") as mock_hash_pw:
            with patch("services.auth.bcrypt.checkpw",
                       wraps=__import__("bcrypt").checkpw) as mock_checkpw:
                u, err = authenticate_user(db, "ghost@example.com", "anything")

                self.assertIsNone(u)
                self.assertIn("오류", err)
                # 핵심: dummy 검증은 bcrypt를 타야 함
                mock_checkpw.assert_called()
                # hash_pw는 사용자 부재 path에서 호출되지 않아야 함
                mock_hash_pw.assert_not_called()


class TestBcryptMigration(unittest.TestCase):
    """legacy pbkdf2 → bcrypt 자동 업그레이드 시나리오."""

    def setUp(self):
        self.password = "validPass123!"

    def test_is_bcrypt_hash_helper(self):
        self.assertTrue(_is_bcrypt_hash("$2b$04$abcdefg"))
        self.assertTrue(_is_bcrypt_hash("$2a$10$xxx"))
        self.assertTrue(_is_bcrypt_hash("$2y$12$yyy"))
        self.assertFalse(_is_bcrypt_hash("abcdef0123456789" * 4))  # pbkdf2 hex
        self.assertFalse(_is_bcrypt_hash(""))
        self.assertFalse(_is_bcrypt_hash(None))

    def test_legacy_pbkdf2_user_can_log_in(self):
        """기존 pbkdf2 유저는 변환 없이도 바로 로그인 가능."""
        user = make_user(email="legacy@example.com", password=self.password)
        db = FakeDBWithMigration({user["email"]: user})
        u, err = authenticate_user(db, user["email"], self.password)
        self.assertIsNotNone(u)
        self.assertIsNone(err)

    def test_legacy_user_password_upgraded_after_login(self):
        """legacy 로그인 후 password 필드가 bcrypt 형식으로 자동 업그레이드."""
        user = make_user(email="legacy@example.com", password=self.password)
        db = FakeDBWithMigration({user["email"]: user})
        # 업그레이드 전: pbkdf2 hex
        self.assertFalse(_is_bcrypt_hash(user["password"]))

        authenticate_user(db, user["email"], self.password)

        # 업그레이드 후: bcrypt
        self.assertTrue(_is_bcrypt_hash(user["password"]),
                        "password가 bcrypt로 업그레이드되지 않음")
        # DB 호출 확인
        self.assertEqual(len(db.password_updates), 1)
        self.assertEqual(db.password_updates[0][0], user["email"])
        self.assertTrue(db.password_updates[0][1].startswith("$2b$"))

    def test_legacy_user_can_login_again_after_migration(self):
        """업그레이드 후에도 다시 로그인 가능 (이번엔 bcrypt 경로)."""
        user = make_user(email="legacy@example.com", password=self.password)
        db = FakeDBWithMigration({user["email"]: user})

        u1, _ = authenticate_user(db, user["email"], self.password)
        self.assertIsNotNone(u1)

        u2, _ = authenticate_user(db, user["email"], self.password)
        self.assertIsNotNone(u2)

        # 업그레이드는 첫 번째 로그인에서만
        self.assertEqual(len(db.password_updates), 1)

    def test_bcrypt_user_logs_in_no_upgrade(self):
        """이미 bcrypt 유저는 추가 업그레이드 없음."""
        user = make_bcrypt_user(email="modern@example.com", password=self.password)
        db = FakeDBWithMigration({user["email"]: user})
        u, err = authenticate_user(db, user["email"], self.password)
        self.assertIsNotNone(u)
        self.assertIsNone(err)
        self.assertEqual(len(db.password_updates), 0)

    def test_wrong_password_legacy_format_no_upgrade(self):
        """legacy 유저 — 틀린 비밀번호는 거부 + 업그레이드 안 함."""
        user = make_user(email="legacy@example.com", password=self.password)
        db = FakeDBWithMigration({user["email"]: user})
        u, err = authenticate_user(db, user["email"], "wrong_pass")
        self.assertIsNone(u)
        self.assertEqual(len(db.password_updates), 0)
        # password 필드가 그대로 pbkdf2여야 함
        self.assertFalse(_is_bcrypt_hash(user["password"]))

    def test_wrong_password_bcrypt_format_rejected(self):
        """bcrypt 유저 — 틀린 비밀번호는 거부."""
        user = make_bcrypt_user(email="modern@example.com", password=self.password)
        db = FakeDBWithMigration({user["email"]: user})
        u, err = authenticate_user(db, user["email"], "wrong_pass")
        self.assertIsNone(u)

    def test_legacy_login_succeeds_when_db_lacks_update_method(self):
        """update_password_hash가 없는 DB여도 검증은 성공 (graceful degradation)."""
        user = make_user(email="legacy@example.com", password=self.password)
        db = FakeDB({user["email"]: user})  # 기본 FakeDB는 update_password_hash 없음
        u, err = authenticate_user(db, user["email"], self.password)
        self.assertIsNotNone(u)
        self.assertIsNone(err)
        # 업그레이드는 안 일어남 — 다음 로그인 때 다시 시도하게 됨
        self.assertFalse(_is_bcrypt_hash(user["password"]))

    def test_verify_password_direct_call(self):
        """verify_password를 직접 호출 — authenticate_user 우회 검증."""
        user = make_user(email="direct@example.com", password=self.password)
        db = FakeDBWithMigration({user["email"]: user})
        self.assertTrue(verify_password(db, user, self.password))
        self.assertFalse(verify_password(db, user, "wrong"))

    def test_user_without_salt_or_bcrypt_rejected(self):
        """salt도 없고 bcrypt도 아닌 잘못된 row는 안전하게 거부."""
        user = {"email": "broken@example.com", "password": "garbage", "salt": ""}
        db = FakeDBWithMigration({user["email"]: user})
        self.assertFalse(verify_password(db, user, self.password))

    def test_corrupted_bcrypt_hash_rejected(self):
        """깨진 bcrypt hash도 안전하게 거부 (예외 던지지 않음)."""
        user = {
            "email": "corrupt@example.com",
            "password": "$2b$invalid_corrupted_hash",
            "salt": "",
        }
        db = FakeDBWithMigration({user["email"]: user})
        # 예외 던지지 않고 False 반환해야 함
        self.assertFalse(verify_password(db, user, self.password))


# ═══════════════════════════════════════════════════════════════════════
# compute_access_status — 권한 판정 SSOT (now=fixed로 안정화)
# ═══════════════════════════════════════════════════════════════════════

class TestComputeAccessStatus(unittest.TestCase):
    """services.auth.compute_access_status 직접 호출. 권한 회귀 차단."""

    def test_no_user_is_guest(self):
        role, allowed, reason = compute_access_status(None, now=FIXED_NOW)
        self.assertEqual(role, "guest")
        self.assertFalse(allowed)
        self.assertEqual(reason, "no_user")

    def test_banned_blocks_premium_even_with_active_subscription(self):
        """ban이 활성 구독을 덮어써야 함 (보안 우선)."""
        future = (FIXED_NOW + timedelta(days=30)).strftime("%Y-%m-%d")
        row = {"is_banned": "Y", "role": "prime", "prime_expire_date": future}
        role, allowed, reason = compute_access_status(row, now=FIXED_NOW)
        self.assertEqual(role, "banned")
        self.assertFalse(allowed)
        self.assertEqual(reason, "banned")

    def test_admin_always_allowed(self):
        row = {"role": "admin"}
        role, allowed, reason = compute_access_status(row, now=FIXED_NOW)
        self.assertEqual(role, "admin")
        self.assertTrue(allowed)
        self.assertEqual(reason, "admin")

    def test_prime_active_subscription(self):
        future = (FIXED_NOW + timedelta(days=30)).strftime("%Y-%m-%d")
        row = {"role": "prime", "prime_expire_date": future}
        role, allowed, reason = compute_access_status(row, now=FIXED_NOW)
        self.assertEqual(role, "prime")
        self.assertTrue(allowed)
        self.assertEqual(reason, "active_subscription")

    def test_pro_active_subscription(self):
        future = (FIXED_NOW + timedelta(days=30)).strftime("%Y-%m-%d")
        row = {"role": "pro", "prime_expire_date": future}
        role, allowed, reason = compute_access_status(row, now=FIXED_NOW)
        self.assertEqual(role, "pro")
        self.assertTrue(allowed)

    def test_prime_expired_downgrades_to_free(self):
        """만료된 prime은 free로 — 결제 회귀의 핵심 케이스."""
        past = (FIXED_NOW - timedelta(days=1)).strftime("%Y-%m-%d")
        row = {"role": "prime", "prime_expire_date": past}
        role, allowed, reason = compute_access_status(row, now=FIXED_NOW)
        self.assertEqual(role, "free")
        self.assertFalse(allowed)
        self.assertEqual(reason, "expired")

    def test_prime_expires_today_still_allowed(self):
        """만료일 = 오늘이면 아직 허용 (마지막 날 권한 유지)."""
        today = FIXED_NOW.strftime("%Y-%m-%d")
        row = {"role": "prime", "prime_expire_date": today}
        role, allowed, reason = compute_access_status(row, now=FIXED_NOW)
        self.assertEqual(role, "prime")
        self.assertTrue(allowed)

    def test_prime_no_expire_date_falls_through_to_free(self):
        """prime인데 만료일 없음 — 안전하게 free 처리."""
        row = {"role": "prime"}
        role, allowed, reason = compute_access_status(row, now=FIXED_NOW)
        self.assertEqual(role, "free")

    def test_invalid_expire_date_safe_fallback(self):
        """파싱 불가능한 만료일 → 안전하게 free, 예외 던지지 않음."""
        row = {"role": "prime", "prime_expire_date": "not-a-date"}
        role, allowed, reason = compute_access_status(row, now=FIXED_NOW)
        self.assertEqual(role, "free")
        self.assertFalse(allowed)
        self.assertEqual(reason, "expire_parse_error")

    def test_free_user_no_premium(self):
        row = {"role": "free"}
        role, allowed, reason = compute_access_status(row, now=FIXED_NOW)
        self.assertEqual(role, "free")
        self.assertFalse(allowed)
        self.assertEqual(reason, "free")

    def test_unknown_role_treated_as_free(self):
        """알 수 없는 role도 안전 fallback."""
        row = {"role": "wizard"}
        role, allowed, reason = compute_access_status(row, now=FIXED_NOW)
        self.assertEqual(role, "free")
        self.assertFalse(allowed)


# ═══════════════════════════════════════════════════════════════════════
# Cross-cutting — authenticate_user + compute_access_status 조합
# ═══════════════════════════════════════════════════════════════════════

class TestAuthAndAccessIntegration(unittest.TestCase):
    """로그인 후 권한 판정까지 한 번에 — 실제 흐름 시뮬레이션."""

    def test_pro_user_full_flow(self):
        future = (FIXED_NOW + timedelta(days=30)).strftime("%Y-%m-%d")
        user_dict = make_user(role="pro", expire=future)
        db = FakeDBWithMigration({user_dict["email"]: user_dict})

        # 1. 로그인 (legacy → bcrypt 자동 업그레이드)
        u, err = authenticate_user(db, user_dict["email"], "testpass123")
        self.assertIsNone(err)
        self.assertIsNotNone(u)
        self.assertEqual(len(db.password_updates), 1)

        # 2. 권한 판정
        role, allowed, reason = compute_access_status(u, now=FIXED_NOW)
        self.assertEqual(role, "pro")
        self.assertTrue(allowed)

    def test_expired_prime_logs_in_but_no_premium(self):
        """만료 유저: 로그인은 되지만 premium 접근 차단 — 가장 흔한 운영 케이스."""
        past = (FIXED_NOW - timedelta(days=10)).strftime("%Y-%m-%d")
        user_dict = make_user(role="prime", expire=past)
        db = FakeDBWithMigration({user_dict["email"]: user_dict})

        u, err = authenticate_user(db, user_dict["email"], "testpass123")
        self.assertIsNone(err)
        self.assertIsNotNone(u)

        role, allowed, reason = compute_access_status(u, now=FIXED_NOW)
        self.assertEqual(role, "free")
        self.assertFalse(allowed)
        self.assertEqual(reason, "expired")

    def test_banned_user_cant_log_in_even_with_correct_password(self):
        """ban 유저 — 비밀번호 맞아도 로그인 불가."""
        user_dict = make_user(is_banned="Y")
        db = FakeDBWithMigration({user_dict["email"]: user_dict})
        u, err = authenticate_user(db, user_dict["email"], "testpass123")
        self.assertIsNone(u)
        self.assertIn("차단", err)

    def test_banned_legacy_user_does_not_migrate_password(self):
        """[v22.4 핵심] 차단된 legacy 계정은 비밀번호 맞아도 bcrypt 업그레이드 X.

        평가자가 지적한 가장 큰 보안 흐름 문제:
          이전(v2.4): 검증 → migration write → ban 거부 (write 발생함)
          현재(v2.5): ban 판정 → allow_upgrade=False로 검증 → ban 거부 (write 없음)
        """
        user = make_user(
            email="banned@example.com",
            password="testpass123",
            is_banned="Y",
        )
        db = FakeDBWithMigration({user["email"]: user})

        u, err = authenticate_user(db, user["email"], "testpass123")

        # 로그인은 차단됨
        self.assertIsNone(u)
        self.assertIn("차단", err)
        # 핵심: bcrypt migration write가 발생하면 안 됨
        self.assertEqual(len(db.password_updates), 0,
                         "차단 계정에 password update가 발생함 — 보안 흐름 위반")
        # password 필드도 그대로 pbkdf2 형식이어야 함
        self.assertFalse(_is_bcrypt_hash(user["password"]),
                         "차단 계정의 password가 bcrypt로 변경됨 — 보안 흐름 위반")

    def test_banned_user_wrong_password_no_write(self):
        """차단 계정 + 틀린 비번 — 당연히 write 없음."""
        user = make_user(email="banned@example.com", is_banned="Y")
        db = FakeDBWithMigration({user["email"]: user})
        u, err = authenticate_user(db, user["email"], "wrong")
        self.assertIsNone(u)
        self.assertEqual(len(db.password_updates), 0)

    def test_verify_password_allow_upgrade_false_skips_migration(self):
        """verify_password(allow_upgrade=False) 직접 호출 — 검증만 하고 write 없음."""
        user = make_user(email="test@example.com", password="testpass123")
        db = FakeDBWithMigration({user["email"]: user})

        # allow_upgrade=False로 호출
        result = verify_password(db, user, "testpass123", allow_upgrade=False)

        self.assertTrue(result)  # 검증 자체는 성공
        self.assertEqual(len(db.password_updates), 0)  # 하지만 write 없음
        self.assertFalse(_is_bcrypt_hash(user["password"]))  # password 그대로


if __name__ == "__main__":
    unittest.main(verbosity=2)

