# -*- coding: utf-8 -*-
"""test_security.py — P4 #18~#19 보안 테스트 (pytest 표준)
═══════════════════════════════════════════════════
실행: pytest test_security.py -v
"""
import os
import re
import tempfile
import shutil

import pytest


@pytest.fixture()
def src_files():
    """소스 파일 내용을 미리 로드."""
    base = os.path.dirname(__file__)
    return {
        "auth": open(os.path.join(base, "auth_user.py")).read(),
        "main": open(os.path.join(base, "main.py")).read(),
        "collector": open(os.path.join(base, "collector.py")).read(),
    }


# ═══════════════════════════════════════════════
#  1. #18: 평문 전역변수 제거
# ═══════════════════════════════════════════════
class TestPlaintextRemoval:
    def test_auth_no_master_pw_global(self, src_files):
        pw = re.findall(r'^MASTER_ADMIN_PW\s*=', src_files["auth"], re.MULTILINE)
        assert len(pw) == 0

    def test_auth_uses_pw_set(self, src_files):
        assert "_ADMIN_PW_SET" in src_files["auth"]

    def test_auth_del_raw_pw(self, src_files):
        assert "del _raw_pw" in src_files["auth"]


# ═══════════════════════════════════════════════
#  2. #18: 해시 비교 메커니즘
# ═══════════════════════════════════════════════
class TestHashComparison:
    def test_auth_compare_digest(self, src_files):
        assert "compare_digest" in src_files["auth"]

    def test_auth_pw_hash(self, src_files):
        assert "_ADMIN_PW_HASH" in src_files["auth"]


# ═══════════════════════════════════════════════
#  3. #19: tomllib 파서
# ═══════════════════════════════════════════════
class TestTomlParser:
    def test_tomllib_import(self, src_files):
        assert "import tomllib" in src_files["collector"]

    def test_tomli_fallback(self, src_files):
        assert "import tomli as tomllib" in src_files["collector"]

    def test_section_subkey_namespace(self, src_files):
        assert "ns_key" in src_files["collector"]
        assert "SECTION_" in src_files["collector"]


# ═══════════════════════════════════════════════
#  4. #19: secrets 파싱 동작
# ═══════════════════════════════════════════════
class TestSecretsParsing:
    @pytest.fixture()
    def secrets_env(self):
        """임시 secrets.toml + 환경변수 백업/복원."""
        tmp = tempfile.mkdtemp()
        streamlit_dir = os.path.join(tmp, ".streamlit")
        os.makedirs(streamlit_dir)
        with open(os.path.join(streamlit_dir, "secrets.toml"), "w") as f:
            f.write(
                'GLOBAL_KEY = "global_value"\n'
                'NUMERIC_KEY = 42\n\n'
                '[telegram]\nTG_TOKEN = "test_token_123"\nTG_ID = "99999"\n\n'
                '[auth]\nmaster_admin_pw = "secret_pw"\n'
            )

        keys = [
            "GLOBAL_KEY", "NUMERIC_KEY", "TG_TOKEN", "TG_ID",
            "TELEGRAM_TG_TOKEN", "TELEGRAM_TG_ID", "AUTH_MASTER_ADMIN_PW",
            "master_admin_pw",
        ]
        saved = {k: os.environ.pop(k, None) for k in keys}
        yield os.path.join(streamlit_dir, "secrets.toml")

        # 복원
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        for k in keys:
            os.environ.pop(k, None)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_toml_parsing(self, secrets_env):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(secrets_env, "rb") as f:
            data = tomllib.load(f)
        assert "GLOBAL_KEY" in data
        assert "telegram" in data
        assert data.get("NUMERIC_KEY") == 42

    def test_namespace_registration(self, secrets_env):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(secrets_env, "rb") as f:
            data = tomllib.load(f)
        for key, val in data.items():
            if isinstance(val, str):
                os.environ[key] = val
            elif isinstance(val, (int, float, bool)):
                os.environ[key] = str(val)
            elif isinstance(val, dict):
                section = key.upper()
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, (str, int, float, bool)):
                        sv = str(sub_val)
                        if sub_key not in os.environ:
                            os.environ[sub_key] = sv
                        os.environ[f"{section}_{sub_key.upper()}"] = sv

        assert os.environ.get("GLOBAL_KEY") == "global_value"
        assert os.environ.get("NUMERIC_KEY") == "42"
        assert os.environ.get("TG_TOKEN") == "test_token_123"
        assert os.environ.get("TELEGRAM_TG_TOKEN") == "test_token_123"
        assert os.environ.get("AUTH_MASTER_ADMIN_PW") == "secret_pw"


# ═══════════════════════════════════════════════
#  5. DB 싱글톤 사용 확인
# ═══════════════════════════════════════════════
class TestDBSingletonUsage:
    def test_auth_uses_get_db(self, src_files):
        s = src_files["auth"]
        assert "get_singleton_db" in s or "from db_utils import get_db" in s

    def test_auth_no_direct_ldy_db(self, src_files):
        assert src_files["auth"].count("LDYDBManager()") == 0
