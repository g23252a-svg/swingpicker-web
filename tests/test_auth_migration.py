#!/usr/bin/env python3
"""
tests/test_auth_migration.py — bcrypt 마이그레이션 테스트
═══════════════════════════════════════════════════════════
GitHub Actions에서 실행:
  BCRYPT_COST=4 python -m pytest tests/test_auth_migration.py -v

로컬에서 실행:
  BCRYPT_COST=4 python tests/test_auth_migration.py
"""
import hashlib
import os
import secrets
import time
import unittest

# bcrypt cost를 테스트용으로 낮춤
os.environ.setdefault("BCRYPT_COST", "4")

import bcrypt


class TestBcryptBasic(unittest.TestCase):
    """bcrypt 기본 동작 검증."""

    def test_hash_and_verify(self):
        pw = "test_password_2025!"
        cost = int(os.environ.get("BCRYPT_COST", "12"))
        hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt(cost)).decode()

        self.assertTrue(hashed.startswith("$2b$"))
        self.assertTrue(bcrypt.checkpw(pw.encode(), hashed.encode()))
        self.assertFalse(bcrypt.checkpw(b"wrong", hashed.encode()))

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_AUTH_TESTS") == "1",
        "slow perf test — set RUN_SLOW_AUTH_TESTS=1 to enable",
    )
    def test_cost_12_reasonable_time(self):
        """운영 cost=12 해싱이 3초 이내인지 확인 (slow perf — CI에서 흔들릴 수 있음)."""
        start = time.time()
        bcrypt.hashpw(b"benchmark", bcrypt.gensalt(12))
        elapsed = time.time() - start
        self.assertLess(elapsed, 3.0, f"cost=12 해싱 {elapsed:.2f}s — 너무 느림")

    def test_different_passwords_different_hashes(self):
        cost = 4
        h1 = bcrypt.hashpw(b"password1", bcrypt.gensalt(cost))
        h2 = bcrypt.hashpw(b"password2", bcrypt.gensalt(cost))
        self.assertNotEqual(h1, h2)

    def test_same_password_different_salts(self):
        cost = 4
        h1 = bcrypt.hashpw(b"same_pw", bcrypt.gensalt(cost))
        h2 = bcrypt.hashpw(b"same_pw", bcrypt.gensalt(cost))
        # 같은 비밀번호도 salt가 달라서 해시가 다름
        self.assertNotEqual(h1, h2)
        # 하지만 둘 다 검증은 통과
        self.assertTrue(bcrypt.checkpw(b"same_pw", h1))
        self.assertTrue(bcrypt.checkpw(b"same_pw", h2))


class TestLegacyMigration(unittest.TestCase):
    """기존 pbkdf2 → bcrypt 무중단 마이그레이션 검증."""

    def _make_legacy_hash(self, password: str, salt: str) -> str:
        """현재 main.py의 _hash_pw 방식 재현."""
        return hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 100_000
        ).hex()

    def _is_legacy(self, hashed: str) -> bool:
        return not hashed.startswith("$2b$")

    def test_legacy_hash_format(self):
        """기존 해시가 hex 문자열인지 확인."""
        salt = secrets.token_hex(16)
        h = self._make_legacy_hash("password123", salt)
        self.assertEqual(len(h), 64)  # SHA-256 hex = 64자
        self.assertTrue(self._is_legacy(h))

    def test_bcrypt_hash_format(self):
        """bcrypt 해시가 $2b$ 로 시작하는지 확인."""
        h = bcrypt.hashpw(b"password123", bcrypt.gensalt(4)).decode()
        self.assertTrue(h.startswith("$2b$"))
        self.assertFalse(self._is_legacy(h))

    def test_migration_flow(self):
        """전체 마이그레이션 플로우 시뮬레이션.

        1. 기존 유저: pbkdf2 해시로 저장됨
        2. 로그인 시도 → pbkdf2로 검증 성공
        3. bcrypt 해시로 업그레이드
        4. 다음 로그인 → bcrypt로 검증 성공
        """
        password = "my_secure_password_42!"
        salt = secrets.token_hex(16)

        # Step 1: 기존 pbkdf2 해시
        legacy_hash = self._make_legacy_hash(password, salt)
        self.assertTrue(self._is_legacy(legacy_hash))

        # Step 2: 로그인 시도 — 기존 방식으로 검증
        attempt_hash = self._make_legacy_hash(password, salt)
        self.assertEqual(legacy_hash, attempt_hash)  # 로그인 성공

        # Step 3: bcrypt로 업그레이드
        cost = int(os.environ.get("BCRYPT_COST", "4"))
        new_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(cost)).decode()
        self.assertFalse(self._is_legacy(new_hash))

        # Step 4: 다음 로그인 — bcrypt로 검증
        self.assertTrue(bcrypt.checkpw(password.encode(), new_hash.encode()))

        # 틀린 비밀번호 거부
        self.assertFalse(bcrypt.checkpw(b"wrong_pw", new_hash.encode()))

    def test_wrong_password_rejected_both_formats(self):
        """틀린 비밀번호가 두 포맷 모두에서 거부."""
        password = "correct"
        wrong = "incorrect"
        salt = secrets.token_hex(16)

        # pbkdf2
        legacy = self._make_legacy_hash(password, salt)
        wrong_legacy = self._make_legacy_hash(wrong, salt)
        self.assertNotEqual(legacy, wrong_legacy)

        # bcrypt
        bh = bcrypt.hashpw(password.encode(), bcrypt.gensalt(4))
        self.assertFalse(bcrypt.checkpw(wrong.encode(), bh))

    def test_security_answer_migration(self):
        """보안 질문 답변도 동일한 마이그레이션 적용."""
        answer = "서울 강남구"
        normalized = answer.strip().lower()
        salt = secrets.token_hex(16)

        # 기존 방식
        legacy = self._make_legacy_hash(normalized, salt)

        # bcrypt 방식
        cost = int(os.environ.get("BCRYPT_COST", "4"))
        new_hash = bcrypt.hashpw(normalized.encode(), bcrypt.gensalt(cost)).decode()
        self.assertTrue(bcrypt.checkpw(normalized.encode(), new_hash.encode()))


class TestTimingAttack(unittest.TestCase):
    """타이밍 공격 방어 검증."""

    @unittest.skipUnless(
        os.environ.get("RUN_SLOW_AUTH_TESTS") == "1",
        "slow timing test — CI runner 부하 따라 flaky. "
        "RUN_SLOW_AUTH_TESTS=1로 활성화",
    )
    def test_nonexistent_user_same_time(self):
        """존재하지 않는 유저도 해싱 수행하여 응답 시간 균등화."""
        cost = int(os.environ.get("BCRYPT_COST", "4"))

        # 존재하는 유저 검증 시간
        real_hash = bcrypt.hashpw(b"password", bcrypt.gensalt(cost))
        t1 = time.time()
        bcrypt.checkpw(b"password", real_hash)
        real_time = time.time() - t1

        # 존재하지 않는 유저 — dummy 해싱
        t2 = time.time()
        bcrypt.hashpw(b"dummy", bcrypt.gensalt(cost))
        fake_time = time.time() - t2

        # 시간 차이가 5배 이내 (완벽한 동일은 불가능)
        ratio = max(real_time, fake_time) / max(min(real_time, fake_time), 0.001)
        self.assertLess(ratio, 5.0,
                        f"타이밍 차이 {ratio:.1f}배 — 공격 가능성")


if __name__ == "__main__":
    unittest.main()
