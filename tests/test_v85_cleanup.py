# -*- coding: utf-8 -*-
"""
tests/test_v85_cleanup.py — [v85-4] 정리 회귀 가드.

■ 무엇을 지키나
  1. 레거시 파일 삭제 — dashboard.py · backup_v205c_20260310/ · fix4_silent_exceptions.py ·
     *.patch 2개 · requirements_nicegui-3.txt 가 없고, 코드/배포/CI 어디서도 참조하지 않는다.
  2. README_*.md 2개 → docs/ 이동 — 루트에 남지 않고, README.md·MIGRATION_GUIDE.md 링크가 실제 파일로 풀린다.
  3. collector.py 조용한 예외 → 이름 있는 로거 "LDY_Collector" 경유.
     강제 실패(손상 캐시·읽기 불가 CSV·API 예외)로 핸들러를 실제로 태워 WARNING이 남는지,
     반환값(폴백 순서)은 그대로인지 확인한다. scoring_engine sidecar 실패도 같은 방식.
     ※ "silent 0건"은 check_silent_exceptions 의 좁은 정의(pass뿐/print뿐) 기준이다.
       return 폴백만 하는 핸들러(get_market_sets 등)는 이 게이트 밖이다 — 완전 무결 주장이 아님.
  4. 2027 시한폭탄 — 'recommend_2026*.csv' 같은 연도 고정 글롭이 코드에 남지 않는다.
     [v85 리뷰] '_2026*' 한 형태만이 아니라 'recommend_20260*'·'_2026??'·'_2026[01]'·'_202*'(10년 고정),
     분리 결합("recommend_" + "2026*.csv")·f-string 까지 잡는다 (가드 자체 회귀 테스트 포함).
     backtest_validation.py 의 기존 1건은 v85-1 소유 — (파일, 문자열) 정확 쌍만 예외로 두고,
     v85-1 머지로 그 문자열이 사라지면 stale 예외 테스트가 실패해 예외 삭제를 강제한다.
  ※ AST 가드는 파싱 실패 파일을 조용히 건너뛰지 않는다 — 실패 목록 자체를 assert 한다.

■ 격리
  collector 테스트는 collector.OUT_DIR 를 tmp_path 로 바꾸고 PYKRX_OK/stock/네트워크 함수를
  전부 monkeypatch 한다 — 실 data/ 를 건드리지 않고(conftest 가드), pykrx 유무(CI/로컬)와
  무관하게 같은 경로를 탄다.
"""
from __future__ import annotations

import ast
import glob
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DELETED = [
    "dashboard.py",
    "backup_v205c_20260310",
    "fix4_silent_exceptions.py",
    "july_profit_defense.patch",
    "profit_recovery_suite_v3929.patch",
    "requirements_nicegui-3.txt",
]
MOVED_READMES = [
    "README_july_profit_defense.md",
    "README_profit_recovery_suite_v3929.md",
]
_SKIP_DIRS = {".git", ".claude", "data", "__pycache__", ".pytest_cache", "node_modules",
              ".venv", "venv", "env", ".mypy_cache"}


def _iter_files(suffixes: tuple[str, ...], root: Path = ROOT):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in _SKIP_DIRS and not d.startswith(("backup_", "legacy_"))]
        for fn in fns:
            if fn.endswith(suffixes):
                yield Path(dp) / fn


def _parse(path: Path):
    """[v85 리뷰] 파싱 실패를 삼키지 않는다 — SyntaxError/UnicodeDecodeError 는 그대로 올린다."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _parse_tree(root: Path = ROOT):
    """(상대경로, AST) 목록 + 파싱 실패 목록.

    [v85 리뷰] 예전 _parse 는 실패 시 None 을 돌려 AST 가드(import·연도 글롭)가 그 파일을
    흔적 없이 건너뛰었다(cp949 파일, 다른 파이썬 버전 문법 등). 실패는 목록으로 모아
    호출부가 assert 한다 — 가드를 조용히 빠져나가는 파일이 없다.
    """
    parsed, unparsed = [], []
    for py in _iter_files((".py",), root):
        rel = py.relative_to(root).as_posix()
        try:
            parsed.append((rel, _parse(py)))
        except (SyntaxError, UnicodeDecodeError) as e:
            unparsed.append(f"{rel}: {type(e).__name__}: {e}")
    return parsed, unparsed


@pytest.fixture(scope="module")
def py_trees():
    """레포 전체 .py AST — 모듈 안에서 한 번만 파싱한다 (가드 두 개 + stale 예외 검사 공용)."""
    return _parse_tree(ROOT)


def test_parse_tree_reports_unparseable_files(tmp_path):
    """파싱 불가 파일(문법 오류·비 UTF-8)은 건너뛰지 않고 실패 목록에 남는다."""
    (tmp_path / "ok.py").write_text("import os\n", encoding="utf-8")
    (tmp_path / "bad_syntax.py").write_text("def f(:\n", encoding="utf-8")
    (tmp_path / "cp949.py").write_bytes("# 한글 주석\nx = 1\n".encode("cp949"))

    parsed, unparsed = _parse_tree(tmp_path)

    assert [rel for rel, _ in parsed] == ["ok.py"]
    assert sorted(u.split(":", 1)[0] for u in unparsed) == ["bad_syntax.py", "cp949.py"]
    assert any("SyntaxError" in u for u in unparsed) and any("UnicodeDecodeError" in u for u in unparsed)


# ═══════════════════════════════════════════════════════════════════
# 1. 레거시 파일 삭제
# ═══════════════════════════════════════════════════════════════════
class TestLegacyFilesRemoved:

    @pytest.mark.parametrize("name", DELETED)
    def test_absent(self, name):
        assert not (ROOT / name).exists(), f"{name} 이(가) 다시 생겼다 (v85 삭제 대상)"

    def test_no_python_imports_deleted_modules(self, py_trees):
        """dashboard / backup 패키지 / fix4 를 import·import_module·__import__ 하는 .py 가 없다."""
        banned = {"dashboard", "backup_v205c_20260310", "fix4_silent_exceptions"}
        trees, unparsed = py_trees
        assert not unparsed, "파싱 불가 .py — import 가드가 검사하지 못한다:\n" + "\n".join(unparsed)
        offenders = []
        for rel, tree in trees:
            for node in ast.walk(tree):
                mods: list[str] = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                elif (isinstance(node, ast.Call) and node.args
                      and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                    f = node.func
                    fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                    if fname in ("import_module", "__import__"):
                        mods = [node.args[0].value]
                for m in mods:
                    if m.split(".")[0] in banned:
                        offenders.append(f"{rel}:{node.lineno} → {m}")
        assert not offenders, "삭제된 모듈을 import 하는 코드:\n" + "\n".join(offenders)

    def test_no_deploy_or_ci_references(self):
        """배포·CI·의존성 설정이 삭제 파일을 가리키지 않는다 (가리키면 빌드/배치가 깨진다)."""
        targets: list[Path] = []
        targets += sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
        for name in ("Dockerfile", "railway.toml", "railway.json", "nixpacks.toml",
                     "Procfile", ".dockerignore", "run_collector.bat"):
            if (ROOT / name).exists():
                targets.append(ROOT / name)
        targets += sorted(ROOT.glob("requirements*.txt"))
        targets += sorted(ROOT.glob("*.sh")) + sorted((ROOT / "scripts").glob("*.sh"))
        assert targets, "검사 대상 설정 파일을 못 찾았다"
        offenders = []
        for t in targets:
            txt = t.read_text(encoding="utf-8", errors="replace")
            for name in DELETED + MOVED_READMES:
                if name in txt:
                    offenders.append(f"{t.relative_to(ROOT)} → {name}")
        assert not offenders, "삭제/이동 파일을 참조하는 배포·CI 설정:\n" + "\n".join(offenders)

    def test_production_entrypoint_untouched(self):
        """라이브 진입점은 main.py — Dockerfile CMD 가 그대로다."""
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "main.py" in dockerfile
        assert "dashboard" not in dockerfile


# ═══════════════════════════════════════════════════════════════════
# 2. README_*.md → docs/
# ═══════════════════════════════════════════════════════════════════
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _relative_links(md: Path) -> list[str]:
    out = []
    for target in _LINK_RE.findall(md.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        out.append(target.split("#", 1)[0])
    return out


class TestReadmesMoved:

    @pytest.mark.parametrize("name", MOVED_READMES)
    def test_moved_to_docs(self, name):
        assert not (ROOT / name).exists(), f"루트에 {name} 이 남아 있다"
        moved = ROOT / "docs" / name
        assert moved.exists(), f"docs/{name} 없음"
        # 내용 보존 — 원래 제목/핵심 룰이 그대로 있다
        txt = moved.read_text(encoding="utf-8")
        assert txt.lstrip().startswith("#")
        assert ("JULY_PROFIT" in txt) or ("PROFIT_RECOVERY" in txt)

    def test_no_readme_star_at_root(self):
        assert sorted(p.name for p in ROOT.glob("README_*.md")) == []
        assert (ROOT / "README.md").exists(), "README.md 는 루트에 남아야 한다"

    @pytest.mark.parametrize("md", ["README.md", "MIGRATION_GUIDE.md"])
    def test_relative_links_resolve(self, md):
        path = ROOT / md
        broken = [t for t in _relative_links(path) if not (path.parent / t).exists()]
        assert not broken, f"{md} 의 깨진 링크: {broken}"

    def test_readme_links_to_moved_docs(self):
        links = _relative_links(ROOT / "README.md")
        for name in MOVED_READMES:
            assert f"docs/{name}" in links, f"README.md 가 docs/{name} 를 가리키지 않는다"

    def test_any_md_link_to_readme_star_resolves(self):
        """어느 .md 든 README_*.md 를 가리키는 링크는 실제 파일로 풀려야 한다."""
        broken = []
        for md in _iter_files((".md",)):
            for t in _relative_links(md):
                if "README_" in t and not (md.parent / t).exists():
                    broken.append(f"{md.relative_to(ROOT)} → {t}")
        assert not broken, "\n".join(broken)

    def test_migration_guide_documents_removal_and_recovery(self):
        txt = (ROOT / "MIGRATION_GUIDE.md").read_text(encoding="utf-8")
        assert "[v85]" in txt and "dashboard.py" in txt
        assert "--diff-filter=D" in txt, "git 이력 복구 방법이 빠졌다"
        assert "Streamlit Cloud" in txt


# ═══════════════════════════════════════════════════════════════════
# 3. collector / scoring_engine 핸들러 로깅
# ═══════════════════════════════════════════════════════════════════
@pytest.fixture
def collector_mod():
    return pytest.importorskip("collector", reason="collector import 불가", exc_type=ImportError)


def _v85_records(caplog, logger_name: str, needle: str, level=logging.WARNING):
    return [r for r in caplog.records
            if r.name == logger_name and r.levelno == level and needle in r.getMessage()]


class TestCollectorHandlersLog:

    def test_module_logger_is_ldy_collector(self, collector_mod):
        assert collector_mod.logger.name == "LDY_Collector"

    def test_mcap_cache_corrupt_json_warns_then_falls_back(self, collector_mod, tmp_path, caplog):
        """1순위(JSON 캐시) 손상 → WARNING + 2순위(recommend CSV) 결과를 그대로 반환."""
        (tmp_path / "mcap_cache_latest.json").write_text("{corrupt", encoding="utf-8")
        pd.DataFrame({"종목코드": ["5930", "000660"], "시가총액(억원)": [4000000, 1500000]}).to_csv(
            tmp_path / "recommend_latest.csv", index=False)
        caplog.set_level(logging.WARNING, logger="LDY_Collector")

        got = collector_mod._load_mcap_cache(str(tmp_path))

        assert got == {"005930": 4000000, "000660": 1500000}
        assert _v85_records(caplog, "LDY_Collector", "[v85] 시총 캐시"), caplog.text

    def test_mcap_cache_all_tiers_fail_warns_and_returns_empty(self, collector_mod, tmp_path, caplog):
        """2·3순위 모두 읽기 실패(파일 대신 디렉터리) → 각각 WARNING, 반환은 예전처럼 {}."""
        (tmp_path / "recommend_latest.csv").mkdir()      # exists=True, read_csv 는 실패
        caplog.set_level(logging.WARNING, logger="LDY_Collector")

        got = collector_mod._load_mcap_cache(str(tmp_path))

        assert got == {}
        assert _v85_records(caplog, "LDY_Collector", "[v85] recommend_latest.csv 시총 복구 실패")
        assert _v85_records(caplog, "LDY_Collector", "[v85] 거래대금 기반 시총 추정 실패")

    def test_mcap_tier2_missing_column_warns_and_uses_tier3(self, collector_mod, tmp_path, caplog):
        pd.DataFrame({"종목코드": ["005930", "123456"], "거래대금(억원)": [1500, 10]}).to_csv(
            tmp_path / "recommend_latest.csv", index=False)
        caplog.set_level(logging.WARNING, logger="LDY_Collector")

        got = collector_mod._load_mcap_cache(str(tmp_path))

        assert got == {"005930": 100000, "123456": 2000}          # 3순위 거래대금 버킷 그대로
        assert _v85_records(caplog, "LDY_Collector", "[v85] recommend_latest.csv 시총 복구 실패")

    def test_market_label_enrich_failure_warns_keeps_labels(self, collector_mod, tmp_path,
                                                            monkeypatch, caplog):
        """3차 폴백에서 price_snapshot 시장 보강 실패 → WARNING, 전일 라벨 유지(반환 동일)."""
        def _boom(*a, **k):
            raise RuntimeError("blocked")
        monkeypatch.setattr(collector_mod, "OUT_DIR", str(tmp_path))
        monkeypatch.setattr(collector_mod, "safe_ohlcv_by_ticker", _boom)
        monkeypatch.setattr(collector_mod.fdr, "StockListing", _boom)
        pd.DataFrame({"종목코드": ["005930", "035720", "247540"],
                      "시장": ["KOSPI"] * 3,
                      "거래대금(억원)": [900, 500, 700]}).to_csv(tmp_path / "recommend_latest.csv", index=False)
        # 시장 컬럼이 없는 스냅샷 → usecols ValueError
        pd.DataFrame({"종목코드": ["005930"], "종가": [70000]}).to_csv(
            tmp_path / "price_snapshot_latest.csv", index=False)
        caplog.set_level(logging.WARNING, logger="LDY_Collector")

        out = collector_mod.pick_top_by_trading_value("20990101", 2)

        assert list(out["종목코드"]) == ["005930", "247540"]
        assert set(out["시장"]) == {"KOSPI"}
        assert _v85_records(caplog, "LDY_Collector", "[v85] price_snapshot_latest.csv 시장 보강 실패")

    def _isolate_name_map(self, mod, monkeypatch, tmp_path, *, pykrx: bool):
        def _boom(*a, **k):
            raise RuntimeError("blocked")
        monkeypatch.setattr(mod, "OUT_DIR", str(tmp_path))
        monkeypatch.setattr(mod.fdr, "StockListing", _boom)
        monkeypatch.setattr(mod.requests, "post", _boom)
        monkeypatch.setattr(mod.requests, "get", _boom)
        monkeypatch.setattr(mod, "PYKRX_OK", pykrx)
        monkeypatch.setattr(mod, "stock", object() if pykrx else None)

    def test_name_map_pykrx_partial_failure_warns(self, collector_mod, tmp_path, monkeypatch, caplog):
        """pykrx 종목명 루프가 KOSDAQ 에서 죽음 → 시장·누적 개수를 담은 WARNING, KOSPI 분은 반환."""
        self._isolate_name_map(collector_mod, monkeypatch, tmp_path, pykrx=True)

        def _tickers(d, market="KOSPI"):
            if market == "KOSDAQ":
                raise RuntimeError("kosdaq blocked")
            return ["005930", "000660"]
        monkeypatch.setattr(collector_mod, "safe_ticker_list", _tickers)
        monkeypatch.setattr(collector_mod, "safe_ticker_name", lambda t: f"N{t}")
        monkeypatch.setattr(collector_mod.time, "sleep", lambda s: None)
        caplog.set_level(logging.WARNING, logger="LDY_Collector")

        got = collector_mod.get_name_map_cached("20990101")

        assert got == {"005930": "N005930", "000660": "N000660"}
        recs = _v85_records(caplog, "LDY_Collector", "[v85] pykrx 종목명 조회 중단")
        assert len(recs) == 1 and "KOSDAQ" in recs[0].getMessage() and "누적 2개" in recs[0].getMessage()

    def test_name_map_naver_target_load_failures_warn(self, collector_mod, tmp_path, monkeypatch, caplog):
        """모든 소스 실패 + 대상 코드 로드 실패(recommend CSV·pykrx) → 각 WARNING, 반환 {}."""
        self._isolate_name_map(collector_mod, monkeypatch, tmp_path, pykrx=True)

        def _boom(*a, **k):
            raise RuntimeError("blocked")
        monkeypatch.setattr(collector_mod, "safe_ticker_list", _boom)
        (tmp_path / "recommend_latest.csv").mkdir()      # exists=True, read_csv 실패
        caplog.set_level(logging.WARNING, logger="LDY_Collector")

        got = collector_mod.get_name_map_cached("20990101")

        assert got == {}
        assert _v85_records(caplog, "LDY_Collector", "[v85] Naver 폴백 대상 코드 로드 실패 (recommend_latest.csv)")
        pk = _v85_records(caplog, "LDY_Collector", "[v85] Naver 폴백 대상 코드 로드 실패 (pykrx")
        assert {("KOSPI" in r.getMessage(), "KOSDAQ" in r.getMessage()) for r in pk} == {(True, False), (False, True)}
        assert not (tmp_path / "krx_codes_20990101.csv").exists()   # 빈 맵은 캐시하지 않는다(기존 동작)

    def test_name_map_naver_per_code_failure_is_debug(self, collector_mod, tmp_path, monkeypatch, caplog):
        """종목별 Naver 실패는 debug(최대 수천 줄 방지) — 이름 없는 코드는 예전처럼 빠진다."""
        self._isolate_name_map(collector_mod, monkeypatch, tmp_path, pykrx=False)
        pd.DataFrame({"종목코드": ["005930", "000660"]}).to_csv(tmp_path / "recommend_latest.csv", index=False)
        caplog.set_level(logging.DEBUG, logger="LDY_Collector")

        got = collector_mod.get_name_map_cached("20990101")

        assert got == {}
        dbg = _v85_records(caplog, "LDY_Collector", "[v85] Naver 종목명 조회 실패", level=logging.DEBUG)
        codes = {m.group(1) for r in dbg for m in [re.search(r"실패 (\d{6}):", r.getMessage())] if m}
        assert codes == {"005930", "000660"}
        assert not _v85_records(caplog, "LDY_Collector", "[v85] Naver 종목명 조회 실패")   # WARNING 홍수 없음

    @staticmethod
    def _fake_secrets(collector_mod, monkeypatch, payload: bytes) -> list[str]:
        """collector 가 여는 .streamlit/secrets.toml 만 메모리 바이트로 대체한다.

        [v85 리뷰] secrets.toml 은 gitignore 라 CI·워크트리에 없다 — exists 만 속이면 open() 이
        FileNotFoundError 를 내서 tomllib 파싱 경로에 닿지 못했다. open 을 collector 모듈
        전역으로만 가린다(builtins 는 그대로) — 다른 파일 open 은 실제 open 으로 통과.
        """
        import builtins
        import io
        opened: list[str] = []
        real_exists, real_open = os.path.exists, builtins.open

        def _is_secrets(p) -> bool:
            return str(p).replace("\\", "/").endswith(".streamlit/secrets.toml")

        def _open(p, mode="r", *a, **k):
            if _is_secrets(p):
                opened.append(mode)
                return io.BytesIO(payload)
            return real_open(p, mode, *a, **k)

        monkeypatch.setattr(os.path, "exists", lambda p: True if _is_secrets(p) else real_exists(p))
        monkeypatch.setattr(collector_mod, "open", _open, raising=False)
        return opened

    def test_secrets_loader_parse_failure_uses_named_logger(self, collector_mod, monkeypatch, caplog):
        """secrets.toml TOML 파싱 실패(실제 tomllib) → print 대신 "LDY_Collector" WARNING (루트 금지)."""
        pytest.importorskip("tomllib", reason="py3.11+ tomllib 경로 전용", exc_type=ImportError)
        opened = self._fake_secrets(collector_mod, monkeypatch, b'V85_CLEANUP_PROBE = [\n')
        monkeypatch.setenv("V85_CLEANUP_PROBE", "untouched")
        caplog.set_level(logging.WARNING, logger="LDY_Collector")

        collector_mod.load_secrets_to_env()

        assert opened == ["rb"], "tomllib 경로(open(..., 'rb'))에 닿지 못했다"
        recs = _v85_records(caplog, "LDY_Collector", "[v85] Secrets 로드 실패")
        assert len(recs) == 1 and "TOMLDecodeError" in recs[0].getMessage(), caplog.text
        assert not [r for r in caplog.records if r.name == "root" and "Secrets" in r.getMessage()]
        assert os.environ["V85_CLEANUP_PROBE"] == "untouched"     # 실패 시 환경변수 미주입(기존 동작)

    def test_secrets_loader_valid_toml_still_injects_env(self, collector_mod, monkeypatch, caplog):
        """정상 TOML 은 예전처럼 주입되고 경고가 없다 — 핸들러 변경이 성공 경로를 건드리지 않았다."""
        pytest.importorskip("tomllib", reason="py3.11+ tomllib 경로 전용", exc_type=ImportError)
        opened = self._fake_secrets(collector_mod, monkeypatch,
                                    b'V85_CLEANUP_PROBE = "ok"\n[sec]\nV85_CLEANUP_SUB = 7\n')
        for key in ("V85_CLEANUP_PROBE", "V85_CLEANUP_SUB", "SEC_V85_CLEANUP_SUB"):
            monkeypatch.setenv(key, "placeholder")     # teardown 이 원상(미설정)으로 되돌린다
        monkeypatch.delenv("V85_CLEANUP_SUB")
        caplog.set_level(logging.WARNING, logger="LDY_Collector")

        collector_mod.load_secrets_to_env()

        assert opened == ["rb"]
        assert os.environ["V85_CLEANUP_PROBE"] == "ok"
        assert os.environ["V85_CLEANUP_SUB"] == "7" and os.environ["SEC_V85_CLEANUP_SUB"] == "7"
        assert not _v85_records(caplog, "LDY_Collector", "[v85] Secrets 로드 실패")

    def test_secrets_handler_never_calls_root_logging(self):
        """루트 함수 logging.warning() 은 shared_log 의 basicConfig(INFO) 를 무력화한다 — 금지."""
        tree = _parse(ROOT / "collector.py")
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_secrets_to_env")
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "logging"):
                assert node.func.attr == "getLogger", f"루트 logging.{node.func.attr}() 호출 (line {node.lineno})"

    def test_no_gate_silent_handlers_in_collector_and_scoring(self):
        """check_silent_exceptions 정의(pass뿐/print뿐) 기준 0건 — baseline 없이도 회귀를 잡는다."""
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import check_silent_exceptions as cse
        finally:
            sys.path.remove(str(ROOT / "scripts"))
        for f in ("collector.py", "scoring_engine.py"):
            assert cse.scan_file(ROOT / f) == [], f


class TestScoringSidecarLogs:

    def _df(self):
        return pd.DataFrame([{
            "종목코드": "000001", "종목명": "테스트", "STRUCT_SCORE": 85.0, "TIMING_SCORE": 80.0,
            "AI_SCORE": 75.0, "ML_SCORE": 75.0, "FINAL_SCORE": 80.0, "ROUTE": "ATTACK",
            "EBS": 8, "PASS_EBS": 1, "종가": 10000, "추천매수가": 10000, "손절가": 9500,
            "추천매도가1": 11500, "거래대금(억원)": 100, "EST_WIN_RATE": 0.60,
            "EST_WIN_RATE_MODE": "MATURE", "CALIBRATION_MODE": "MATURE",
        }])

    def test_sidecar_write_failure_warns_and_result_unchanged(self, tmp_path, caplog):
        se = pytest.importorskip("scoring_engine", reason="scoring_engine import 불가", exc_type=ImportError)
        caplog.set_level(logging.WARNING, logger=se.logger.name)
        base_x, base_meta = se.compute_elite_score(self._df())

        x, meta = se.compute_elite_score(self._df(), out_dir=str(tmp_path / "missing_dir"),
                                         trade_ymd="20990101")

        pd.testing.assert_frame_equal(x, base_x)
        assert meta == base_meta and "sidecar_path" not in meta
        recs = _v85_records(caplog, se.logger.name, "[v85] top_pick_funnel_20990101.json 저장 실패")
        assert recs, caplog.text

    def test_sidecar_success_path_still_writes(self, tmp_path):
        se = pytest.importorskip("scoring_engine", reason="scoring_engine import 불가", exc_type=ImportError)
        _, meta = se.compute_elite_score(self._df(), out_dir=str(tmp_path), trade_ymd="20990101")
        assert Path(meta["sidecar_path"]).exists()


# ═══════════════════════════════════════════════════════════════════
# 4. 2027 시한폭탄 — 연도 고정 글롭 금지
# ═══════════════════════════════════════════════════════════════════
#: 배치 파일 접두어 (recommend_ / price_snapshot_ / ohlcv_cache_ …).
_BATCH_PREFIX = r"(?:recommend|price_snapshot|snapshot|ohlcv_cache|ohlcv)"
#: [v85 리뷰] 접두어 + 최소 10년 단위 연도('_202') 고정 + 같은 토큰 안 어딘가의 글롭 메타문자(* ? [).
#:   'recommend_2026*' 뿐 아니라 '_20260*'(1~9월)·'_2026??0*'·'_2026[01][0-9]*'·'_202*'(10년 고정)도 잡는다.
#:   'recommend_2*.csv'·'recommend_[0-9]…'(연도 무관)·'recommend_20261016.csv'(정확 파일명)는 통과.
_YEAR_FIXED_GLOB = re.compile(_BATCH_PREFIX + r"\w*?_20\d[^\s'\"]*?[*?\[]")
#: [v85 리뷰] 분리 결합형 — PREFIX + "2026*.csv" 처럼 연도로 시작하는 조각 + 글롭 메타문자.
#:   ('{}' 는 _static_str 이 비상수 조각 자리에 넣는 표시)
_YEAR_HEAD_GLOB = re.compile(r"(?:^|\{\})_?20\d[^\s'\"]*?[*?\[]")
#: (파일, 문자열) 정확 쌍만 예외 — 같은 파일에 새로 생기는 연도 고정 글롭은 그대로 잡힌다.
#: v85-1(백테스트 재구성) 브랜치가 소유·수정한다. 두 브랜치가 머지되면 이 문자열이 사라지므로
#: test_owned_elsewhere_exemptions_not_stale 가 실패한다(의도된 트립와이어) → 그때 이 항목을 지운다.
_OWNED_ELSEWHERE: set = set()   # [v85] v85/backtest 머지로 backtest_validation.py 글롭도 연도 무관 — 예외 없음
#: 이 파일 자신은 가드 회귀 테스트용 위반 샘플 문자열을 담고 있어 스캔에서 뺀다.
_SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()


def _static_str(node):
    """문자열 상수 · f-string · '+' 결합을 정적으로 펼친다 — 비상수 조각은 '{}' 자리표시.

    "data/recommend_" + "2026*.csv" 처럼 연도와 접두어가 서로 다른 상수에 나뉘어도
    결합 결과("data/recommend_2026*.csv")로 검사하기 위함.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
                       for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_str(node.left), _static_str(node.right)
        if left is None and right is None:
            return None
        return (left if left is not None else "{}") + (right if right is not None else "{}")
    return None


def _year_fixed_hits(tree) -> list[tuple[int, str]]:
    """코드(docstring·단독 문자열 문 제외)의 연도 고정 배치 글롭 (lineno, 문자열) 목록."""
    doc_ids = {id(n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Expr) and isinstance(n.value, (ast.Constant, ast.JoinedStr))}
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in doc_ids:
            continue
        s = _static_str(node)
        if s is None:
            continue
        if _YEAR_FIXED_GLOB.search(s) or _YEAR_HEAD_GLOB.search(s):
            hit = (node.lineno, s)
            if hit not in hits:
                hits.append(hit)
    return hits


class TestNoYearFixedGlob:

    def test_no_year_fixed_batch_glob_in_code(self, py_trees):
        trees, unparsed = py_trees
        assert not unparsed, "파싱 불가 .py — 연도 글롭 가드가 검사하지 못한다:\n" + "\n".join(unparsed)
        offenders = []
        for rel, tree in trees:
            if rel == _SELF:
                continue
            for lineno, s in _year_fixed_hits(tree):
                if (rel, s) in _OWNED_ELSEWHERE:
                    continue
                offenders.append(f"{rel}:{lineno} {s!r}")
        assert not offenders, "연도 고정 글롭 (2027년부터 새 파일을 못 본다):\n" + "\n".join(offenders)

    def test_owned_elsewhere_exemptions_not_stale(self, py_trees):
        """예외 쌍이 실제로 아직 코드에 있어야 한다 — 소유 브랜치(v85-1)가 고치면 예외를 지우게 강제.

        이 테스트가 실패하면 _OWNED_ELSEWHERE 에서 해당 항목을 지운다 (가드가 그 파일을 다시 전부 본다).
        """
        trees, _ = py_trees
        present = {(rel, s) for rel, tree in trees for _, s in _year_fixed_hits(tree)}
        stale = sorted(_OWNED_ELSEWHERE - present)
        assert not stale, ("소유 브랜치가 이미 고친 예외 — tests/test_v85_cleanup.py 의 _OWNED_ELSEWHERE 에서 "
                           "지워라 (남겨 두면 같은 문자열이 다시 들어와도 가드가 못 잡는다): " + repr(stale))

    @pytest.mark.parametrize("src", [
        'glob.glob("data/recommend_2026*.csv")',
        'glob.glob("data/price_snapshot_2027*.csv")',
        'glob.glob(os.path.join(d, "ohlcv_cache_2026*.parquet"))',
        'glob.glob("recommend_20260*.csv")',                  # 2026년 1~9월에 고정
        'glob.glob("recommend_2026??0*.csv")',
        'glob.glob("recommend_2026[01][0-9]*.csv")',
        'glob.glob("recommend_202*.csv")',                    # 2020년대에 고정
        'glob.glob("data/recommend_" + "2026*.csv")',          # 분리 결합 — 상수 하나씩은 무해해 보임
        'glob.glob(PREFIX + "2026*.csv")',
        'glob.glob(f"{d}/recommend_2026*.csv")',
        'glob.glob(f"{d}/recommend_" + "2026" + "*.csv")',
    ])
    def test_guard_catches_year_fixed_variants(self, src):
        """[v85 리뷰] 가드 자체 회귀 — '_2026*' 한 형태만 잡던 예전 정규식이 놓치던 변형들."""
        assert _year_fixed_hits(ast.parse(src)), f"가드가 놓친 연도 고정 글롭: {src}"

    @pytest.mark.parametrize("src", [
        'glob.glob(f"recommend_{_YMD8_GLOB}.csv")',
        'glob.glob(os.path.join(d, "recommend_" + "[0-9]" * 8 + ".csv"))',
        'glob.glob("recommend_2*.csv")',                       # 연도 무관 (v85-1 방식)
        'glob.glob("recommend_20*.csv")',                      # 세기 고정 — 2100년까지 무해
        'p = "data/recommend_20261016.csv"',                  # 정확 파일명 (글롭 아님)
        'p = "recommend_latest.csv"',
        'msg = "recommend_20261016.csv 없음 [skip]"',          # 공백 뒤 '[' 는 같은 토큰 아님
        'RE = re.compile(r"^recommend_(2\\d{7})\\.csv$")',
        'def f():\n    """예전엔 recommend_2026*.csv 를 글롭했다."""\n    return 1',   # docstring
    ])
    def test_guard_allows_year_agnostic_forms(self, src):
        assert not _year_fixed_hits(ast.parse(src)), f"오탐: {src}"

    def test_year_agnostic_glob_semantics(self, tmp_path):
        """새 글롭: 2026·2027 날짜 파일은 모두, *_latest·태그 파일은 제외."""
        import combo_optimizer
        for name in ("recommend_20261230.csv", "recommend_20270104.csv", "recommend_latest.csv",
                     "recommend_20270104_tag.csv", "price_snapshot_20270104.csv"):
            (tmp_path / name).write_text("x", encoding="utf-8")
        pat = os.path.join(str(tmp_path), f"recommend_{combo_optimizer._YMD8_GLOB}.csv")
        assert [os.path.basename(p) for p in sorted(glob.glob(pat))] == [
            "recommend_20261230.csv", "recommend_20270104.csv"]

    def test_snapshot_date_lists_cross_year(self, tmp_path):
        """백테스트 날짜축이 해를 넘겨도 이어진다 (예전 글롭은 2027 스냅샷을 버렸다)."""
        gb = pytest.importorskip("guard_backtest_v23", exc_type=ImportError)
        import momentum_lane_backtest as mlb
        for ymd in ("20261229", "20261230", "20270104"):
            (tmp_path / f"price_snapshot_{ymd}.csv").write_text("종목코드,종가\n", encoding="utf-8")
        (tmp_path / "price_snapshot_latest.csv").write_text("종목코드,종가\n", encoding="utf-8")
        want = ["20261229", "20261230", "20270104"]
        assert gb._snap_dates(str(tmp_path)) == want
        assert mlb._snap_dates(str(tmp_path)) == want

    def test_real_data_new_glob_selects_exactly_dated_batches(self):
        """실데이터: 새 글롭이 고르는 집합 == data/ 의 recommend_YYYYMMDD.csv 전부 (파일에서 직접 계산)."""
        data = ROOT / "data"
        names = os.listdir(data) if data.is_dir() else []
        dated = sorted(n for n in names if re.fullmatch(r"recommend_\d{8}\.csv", n))
        if not dated:
            pytest.skip("data/recommend_YYYYMMDD.csv 없음")
        import honest_prob
        got = sorted(os.path.basename(p) for p in glob.glob(
            os.path.join(str(data), f"recommend_{honest_prob._YMD8_GLOB}.csv")))
        assert got == dated
        # 비배치 파일(latest·태그)은 절대 섞이지 않는다
        assert all(re.fullmatch(r"recommend_\d{8}\.csv", n) for n in got)
