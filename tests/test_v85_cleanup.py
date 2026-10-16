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
     (backtest_validation.py 는 v85-1 담당 — 그 브랜치에서 처리하므로 여기서 제외)

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


def _iter_files(suffixes: tuple[str, ...]):
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in _SKIP_DIRS and not d.startswith(("backup_", "legacy_"))]
        for fn in fns:
            if fn.endswith(suffixes):
                yield Path(dp) / fn


def _parse(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


# ═══════════════════════════════════════════════════════════════════
# 1. 레거시 파일 삭제
# ═══════════════════════════════════════════════════════════════════
class TestLegacyFilesRemoved:

    @pytest.mark.parametrize("name", DELETED)
    def test_absent(self, name):
        assert not (ROOT / name).exists(), f"{name} 이(가) 다시 생겼다 (v85 삭제 대상)"

    def test_no_python_imports_deleted_modules(self):
        """dashboard / backup 패키지 / fix4 를 import·import_module·__import__ 하는 .py 가 없다."""
        banned = {"dashboard", "backup_v205c_20260310", "fix4_silent_exceptions"}
        offenders = []
        for py in _iter_files((".py",)):
            tree = _parse(py)
            if tree is None:
                continue
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
                        offenders.append(f"{py.relative_to(ROOT)}:{node.lineno} → {m}")
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

    def test_secrets_loader_failure_uses_named_logger(self, collector_mod, monkeypatch, caplog):
        """secrets.toml 파싱 실패 → print 대신 "LDY_Collector" WARNING (루트 logging.warning 금지)."""
        import tomllib
        real_exists = os.path.exists
        monkeypatch.setattr(os.path, "exists",
                            lambda p: True if str(p).endswith("secrets.toml") else real_exists(p))
        monkeypatch.setattr(tomllib, "load", lambda f: (_ for _ in ()).throw(ValueError("bad toml")))
        caplog.set_level(logging.WARNING, logger="LDY_Collector")

        collector_mod.load_secrets_to_env()

        assert _v85_records(caplog, "LDY_Collector", "[v85] Secrets 로드 실패")
        assert not [r for r in caplog.records if r.name == "root" and "Secrets" in r.getMessage()]

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
#: 'recommend_2026*.csv' / 'price_snapshot_2026*.csv' / 'ohlcv_cache_2026*.parquet' 류.
_YEAR_FIXED_GLOB = re.compile(r"_20\d\d\*")
#: v85-1(백테스트 재구성) 브랜치가 소유 — 그쪽에서 고친다. 머지 후 이 예외는 지워도 된다.
_OWNED_ELSEWHERE = {"backtest_validation.py"}


def _string_constants(tree):
    """코드에서 쓰이는 문자열 상수 — docstring·단독 문자열 문(설명문)은 제외."""
    doc_ids = {id(n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
            yield node.lineno, node.value


class TestNoYearFixedGlob:

    def test_no_year_fixed_batch_glob_in_code(self):
        offenders = []
        for py in _iter_files((".py",)):
            rel = py.relative_to(ROOT).as_posix()
            if rel in _OWNED_ELSEWHERE:
                continue
            tree = _parse(py)
            if tree is None:
                continue
            for lineno, s in _string_constants(tree):
                if _YEAR_FIXED_GLOB.search(s) and ("recommend" in s or "snapshot" in s or "ohlcv" in s):
                    offenders.append(f"{rel}:{lineno} {s!r}")
        assert not offenders, "연도 고정 글롭 (2027년부터 새 파일을 못 본다):\n" + "\n".join(offenders)

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
