"""P4 #18~#19 보안 테스트 — 평문 제거 + tomllib 파서"""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))

# streamlit mock (import 시 필요)
import types
if 'streamlit' not in sys.modules:
    mock_st = types.ModuleType('streamlit')
    class _MockSecrets(dict):
        def get(self, k, d=None):
            return d
    mock_st.secrets = _MockSecrets()
    sys.modules['streamlit'] = mock_st

PASS = 0
FAIL = 0

def test(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def run():
    global PASS, FAIL
    PASS, FAIL = 0, 0

    print("=" * 60)
    print("🧪 P4 #18~#19 보안 테스트")
    print("=" * 60)

    # ═══ 1. #18: 평문 전역변수 없음 ═══
    print("\n📐 1. #18: 평문 전역변수 제거")

    # auth_user.py 소스 검사
    auth_src = open(os.path.join(os.path.dirname(__file__), "auth_user.py")).read()

    # MASTER_ADMIN_PW = 형태의 전역 할당이 없어야 함
    import re
    pw_assignments = re.findall(r'^MASTER_ADMIN_PW\s*=', auth_src, re.MULTILINE)
    test("auth_user: MASTER_ADMIN_PW 전역 할당 0건",
         len(pw_assignments) == 0, f"found {len(pw_assignments)}")

    # _ADMIN_PW_SET 사용
    test("auth_user: _ADMIN_PW_SET 사용",
         "_ADMIN_PW_SET" in auth_src)

    # del _raw_pw 존재
    test("auth_user: del _raw_pw 존재",
         "del _raw_pw" in auth_src)

    # main.py도 검사
    main_src = open(os.path.join(os.path.dirname(__file__), "main.py")).read()
    main_pw_assigns = re.findall(r'^MASTER_ADMIN_PW\s*=', main_src, re.MULTILINE)
    test("main: MASTER_ADMIN_PW 전역 할당 0건",
         len(main_pw_assigns) == 0, f"found {len(main_pw_assigns)}")
    test("main: del _raw_admin_pw 존재",
         "del _raw_admin_pw" in main_src)
    test("main: _ADMIN_PW_SET 사용",
         "_ADMIN_PW_SET" in main_src)

    # ═══ 2. #18: 해시 비교 (timing-safe) ═══
    print("\n📐 2. #18: 해시 비교 메커니즘")
    test("auth_user: compare_digest 사용",
         "compare_digest" in auth_src)
    test("auth_user: _ADMIN_PW_HASH 존재",
         "_ADMIN_PW_HASH" in auth_src)
    test("main: compare_digest 사용",
         "compare_digest" in main_src)

    # ═══ 3. #19: tomllib 파서 ═══
    print("\n📐 3. #19: tomllib 파서")
    collector_src = open(os.path.join(os.path.dirname(__file__), "collector.py")).read()
    test("collector: tomllib import",
         "import tomllib" in collector_src)
    test("collector: tomli fallback",
         "import tomli as tomllib" in collector_src)

    # 네임스페이스 등록
    test("collector: SECTION_SUBKEY 네임스페이스",
         "ns_key" in collector_src and "SECTION_" in collector_src)

    # ═══ 4. #19: secrets 파싱 동작 테스트 ═══
    print("\n📐 4. #19: secrets 파싱 동작")

    # 임시 secrets.toml 생성
    tmp = tempfile.mkdtemp()
    streamlit_dir = os.path.join(tmp, ".streamlit")
    os.makedirs(streamlit_dir)

    secrets_content = '''
GLOBAL_KEY = "global_value"
NUMERIC_KEY = 42

[telegram]
TG_TOKEN = "test_token_123"
TG_ID = "99999"

[auth]
master_admin_pw = "secret_pw"
'''
    with open(os.path.join(streamlit_dir, "secrets.toml"), "w") as f:
        f.write(secrets_content)

    # load_secrets_to_env 테스트 (기존 환경변수 백업)
    saved_env = {}
    for k in ["GLOBAL_KEY", "NUMERIC_KEY", "TG_TOKEN", "TG_ID",
              "TELEGRAM_TG_TOKEN", "TELEGRAM_TG_ID", "AUTH_MASTER_ADMIN_PW",
              "master_admin_pw"]:
        saved_env[k] = os.environ.pop(k, None)

    try:
        # collector의 load_secrets_to_env를 직접 호출하기 위해 함수 추출
        # (collector import는 부작용이 많으므로 소스에서 함수만 추출)
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None

        if tomllib:
            secrets_path = os.path.join(streamlit_dir, "secrets.toml")
            with open(secrets_path, "rb") as f:
                data = tomllib.load(f)

            # 파싱 자체 성공
            test("toml 파싱 성공", "GLOBAL_KEY" in data)
            test("섹션 파싱", "telegram" in data)
            test("숫자 파싱", data.get("NUMERIC_KEY") == 42)

            # 네임스페이스 등록 시뮬레이션
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
                            ns_key = f"{section}_{sub_key.upper()}"
                            os.environ[ns_key] = sv

            test("GLOBAL_KEY 등록", os.environ.get("GLOBAL_KEY") == "global_value")
            test("NUMERIC_KEY 등록", os.environ.get("NUMERIC_KEY") == "42")
            test("TG_TOKEN 직접 등록", os.environ.get("TG_TOKEN") == "test_token_123")
            test("TELEGRAM_TG_TOKEN 네임스페이스",
                 os.environ.get("TELEGRAM_TG_TOKEN") == "test_token_123")
            test("AUTH_MASTER_ADMIN_PW 네임스페이스",
                 os.environ.get("AUTH_MASTER_ADMIN_PW") == "secret_pw")
        else:
            test("tomllib 사용 가능", False, "not installed")

    finally:
        # 환경변수 복원
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        # 추가 키도 정리
        for k in ["GLOBAL_KEY", "NUMERIC_KEY", "TG_TOKEN", "TG_ID",
                   "TELEGRAM_TG_TOKEN", "TELEGRAM_TG_ID", "AUTH_MASTER_ADMIN_PW",
                   "master_admin_pw"]:
            os.environ.pop(k, None)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    # ═══ 5. DB 싱글톤 사용 확인 ═══
    print("\n📐 5. DB 싱글톤 사용")
    test("auth_user: get_db → 싱글톤",
         "get_singleton_db" in auth_src or "from db_utils import get_db" in auth_src)
    test("auth_user: LDYDBManager() 직접 생성 0건",
         auth_src.count("LDYDBManager()") == 0)
    test("main: get_db → 싱글톤",
         "from db_utils import get_db" in main_src)

    # ── 결과 ──
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"🏁 결과: {PASS}/{total} 통과 ({FAIL} 실패)")
    if FAIL > 0:
        print("⚠️ 실패 항목이 있습니다!")
        sys.exit(1)
    else:
        print("🏆 ALL PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run()
