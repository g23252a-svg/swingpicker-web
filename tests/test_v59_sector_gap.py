# -*- coding: utf-8 -*-
"""[v59] 업종 결측이 화면을 죽이고 사이징을 오염시킨 것 — 3층 회귀 봉쇄.

실제로 벌어진 일 (2026-08-10, 사용자 신고)
  시장 탭: `❌ 로딩 실패: ('None entries cannot have not-None children',
  업종_대분류 NaN 종목명 녹십자 ...)`

파고 든 결과 (2026-08-07 배치 326종목 실측)
  · 업종_대분류 결측 **30종목(9.2%)** — 8/06은 32/278(11.5%)
  · 그 30행은 업종(상세)은 있는데(녹십자='의약품 제조업') 업종_대분류·
    업종_상세·SECTOR_*·NEWS_SCORE·개인순매수·STRATEGY*·시총기준일·
    **LDY_SCORE·TOTAL_SCORE·RANK_SCORE**·벤치_60d_*·CALIBRATION_* 등
    18개 컬럼이 전부 비어 있었다 → 섹터/뉴스/전략 단계 뒤에 합류한 행들이다
  · ALPHA_SCORE·ROUTE·FINAL_SCORE·EBS·TOP_PICK은 채워져 있었다

세 층 모두 이 파일이 막는다
  A) 표시: plotly 트리맵이 결측 부모로 죽지 않는다
  B) 돈: 켈리 섹터 모멘텀이 결측을 **가짜 섹터 하나로 묶지 않는다**
     (종전 `astype(str).fillna("?")` → 무관한 30종목이 공통 배수를 받아
      ALPHA_SIZE_MULT → 켈리_수량이 오염됐다)
  C) 복구·가시성: 업종(상세)에서 같은 분류기로 복구하고, 결측 사실을
     조용히 넘기지 않는다
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import sector_repair as sr  # noqa: E402


# ── A. 표시: 트리맵이 죽지 않는다 ─────────────────────────────────
class TestTreemapSurvivesGaps:
    def _df(self, **ov):
        base = pd.DataFrame([
            {"업종_대분류": "의약품", "종목명": "삼성바이오",
             "거래대금(억원)": 100.0, "LDY_SCORE": 80.0},
            {"업종_대분류": np.nan, "종목명": "녹십자",
             "거래대금(억원)": 50.0, "LDY_SCORE": 70.0},
        ])
        for k, v in ov.items():
            base[k] = v
        return base

    def test_nan_parent_no_longer_crashes(self):
        """이것이 사용자가 본 크래시다."""
        from chart_components import plot_sector_treemap
        fig = plot_sector_treemap(self._df())
        assert fig is not None and len(fig.data) >= 1

    @pytest.mark.parametrize("blank", ["", "  ", "nan", "None", "-", None])
    def test_all_blank_forms_are_handled(self, blank):
        from chart_components import plot_sector_treemap
        df = self._df()
        df.loc[1, "업종_대분류"] = blank
        assert plot_sector_treemap(df) is not None

    def test_missing_values_and_colors(self):
        from chart_components import plot_sector_treemap
        df = self._df()
        df.loc[0, "거래대금(억원)"] = np.nan
        df.loc[1, "LDY_SCORE"] = np.nan
        assert plot_sector_treemap(df) is not None

    def test_all_zero_value_returns_empty_not_crash(self):
        from chart_components import plot_sector_treemap
        df = self._df()
        df["거래대금(억원)"] = 0.0
        fig = plot_sector_treemap(df)
        assert fig is not None and len(fig.data) == 0

    def test_real_batch_csv_renders(self):
        """실제 배치 CSV(결측 30종목 포함)로도 살아야 한다."""
        from chart_components import plot_sector_treemap
        p = ROOT / "data" / "recommend_20260807.csv"
        if not p.exists():
            pytest.skip("배치 CSV 없음")
        d = pd.read_csv(p, encoding="utf-8-sig", dtype={"종목코드": str},
                        low_memory=False)
        assert plot_sector_treemap(d.head(50)) is not None


# ── B. 돈: 결측이 가짜 섹터로 묶이지 않는다 ───────────────────────
class TestSizingNotPolluted:
    def test_unknown_sector_gets_neutral_multiplier(self):
        """무관한 결측 종목들이 서로의 수익률로 배수를 받으면 안 된다."""
        import kelly_calibrator as kc

        out = pd.DataFrame({
            "업종_대분류": pd.Series(["의약품", None, "반도체", None, "의약품"],
                                dtype="object"),
            "ret_5d_%": [1.0, 20.0, 2.0, 20.0, 1.0],
        })
        _ret5 = pd.to_numeric(out["ret_5d_%"], errors="coerce")
        _sec_raw = out["업종_대분류"]
        _sec = _sec_raw.astype("object").where(_sec_raw.notna(), None)
        _sec = _sec.map(lambda v: None if v is None or str(v).strip() in
                        ("", "nan", "None", "?", "미분류") else str(v).strip())
        known = _sec.notna()
        sec_mean = pd.Series(np.nan, index=out.index, dtype="float64")
        sec_mean.loc[known] = (_ret5.loc[known]
                               .groupby(_sec.loc[known]).transform("mean"))
        factor = (kc._SECMOM_LO + (kc._SECMOM_HI - kc._SECMOM_LO)
                  * sec_mean.rank(pct=True).fillna(0.5))
        assert pd.isna(sec_mean.iloc[1]), "결측이 섹터 평균을 받았다"
        assert abs(factor.iloc[1] - 1.0) < 1e-9, "결측이 중립 배수가 아니다"
        assert abs(factor.iloc[3] - 1.0) < 1e-9

    def test_kelly_source_no_longer_uses_fillna_placeholder(self):
        """`astype(str).fillna("?")` 패턴이 되살아나면 오염이 재발한다."""
        # 주석에는 '종전에 이랬다'는 기록이 남아 있어야 하므로 코드 라인만 본다
        raw = (ROOT / "kelly_calibrator.py").read_text(encoding="utf-8")
        code = "\n".join(l for l in raw.splitlines()
                         if not l.lstrip().startswith("#"))
        assert 'astype(str).fillna("?")' not in code, \
            "결측을 '?' 가짜 섹터로 묶는 패턴이 다시 들어왔다"
        src = raw
        assert "가짜 섹터" in src, "왜 이렇게 고쳤는지 기록이 사라졌다"


# ── C. 복구와 가시성 ──────────────────────────────────────────────
class TestRepair:
    def _mixed(self):
        return pd.DataFrame([
            {"종목명": "삼성전자", "업종": "전기전자", "업종_대분류": "IT/전기전자"},
            {"종목명": "녹십자", "업종": "의약품 제조업", "업종_대분류": np.nan},
            {"종목명": "제이씨케미칼", "업종": "기타 화학제품 제조업",
             "업종_대분류": ""},
        ])

    def test_repairs_from_detail_column(self):
        out, rep = sr.repair_sector(self._mixed())
        assert rep["ok"] and rep["missing_before"] == 2
        assert rep["repaired"] == 2, f"복구 실패: {rep}"
        assert rep["still_missing"] == 0
        assert not sr.blank_mask(out, "업종_대분류").any()

    def test_uses_same_classifier_as_pipeline(self):
        """다른 규칙으로 채우면 복구 행과 정상 행이 다른 기준으로 섞인다."""
        out, rep = sr.repair_sector(self._mixed())
        assert rep["classifier"] in ("sector_utils", "collector"), rep["classifier"]
        from sector_utils import classify_big_sector
        expect = classify_big_sector("녹십자", "의약품 제조업")
        got = out.loc[out["종목명"] == "녹십자", "업종_대분류"].iloc[0]
        assert got == expect

    def test_real_batch_repairs_all_30(self):
        p = ROOT / "data" / "recommend_20260807.csv"
        if not p.exists():
            pytest.skip("배치 CSV 없음")
        d = pd.read_csv(p, encoding="utf-8-sig", dtype={"종목코드": str},
                        low_memory=False)
        _, rep = sr.repair_sector(d)
        assert rep["missing_before"] > 0, "이 CSV에는 결측이 있었다"
        assert rep["repaired"] == rep["missing_before"], \
            f"복구 누락: {rep['repaired']}/{rep['missing_before']}"

    def test_no_detail_column_reports_failure_not_silence(self):
        df = pd.DataFrame([{"종목명": "X", "업종_대분류": np.nan}])
        out, rep = sr.repair_sector(df)
        assert rep["still_missing"] == 1
        assert sr.is_alarming(rep), "복구 실패인데 경고하지 않는다"

    def test_missing_is_reported_even_when_fully_repaired(self):
        """복구되더라도 '애초에 결측이 많았다'는 사실은 남아야 한다 —
        그 행들은 LDY_SCORE 등 17개 컬럼도 비어 있었다."""
        out, rep = sr.repair_sector(self._mixed())
        assert rep["repaired"] == 2 and rep["still_missing"] == 0
        line = sr.sector_repair_line(rep)
        assert "결측" in line and "복구" in line
        assert sr.is_alarming(rep), "결측 비율 66%인데 경고가 없다"

    def test_clean_frame_is_quiet(self):
        df = pd.DataFrame([{"종목명": "A", "업종": "전기전자",
                            "업종_대분류": "IT/전기전자"}])
        out, rep = sr.repair_sector(df)
        assert rep["missing_before"] == 0
        assert sr.sector_repair_line(rep) == ""
        assert not sr.is_alarming(rep)

    def test_does_not_invent_other_missing_columns(self):
        """업종만 복구한다 — 없는 점수(LDY_SCORE 등)를 만들어내지 않는다."""
        df = self._mixed()
        df["LDY_SCORE"] = [80.0, np.nan, np.nan]
        out, _ = sr.repair_sector(df)
        assert out["LDY_SCORE"].isna().sum() == 2, "없는 점수를 채웠다"

    def test_display_label_is_display_only(self):
        df = self._mixed()
        lab = sr.display_sector(df)
        assert lab.iloc[1] == sr.DISPLAY_UNKNOWN
        doc = sr.display_sector.__doc__ or ""
        assert "엔진 축에는 쓰지" in doc, "표시 전용이라는 경고가 사라졌다"

    def test_empty_frame_safe(self):
        out, rep = sr.repair_sector(pd.DataFrame())
        assert rep["ok"] is False and sr.sector_repair_line(rep) == ""


# ── D. 배치 배선 생존 ─────────────────────────────────────────────
def test_pipeline_calls_repair_before_kelly():
    """복구가 켈리 사이징보다 먼저 돌아야 오염이 막힌다."""
    src = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")
    assert "repair_sector" in src, "파이프라인이 복구를 호출하지 않는다"
    i_repair = src.index("repair_sector")
    i_kelly = src.index("resize_kelly_with_alpha")
    assert i_repair < i_kelly, "복구가 켈리 사이징보다 뒤에 있다 (오염 방지 실패)"

# ── E. 실행 순서 오염 봉쇄 (v59에서 실제로 터진 것) ───────────────
class TestNoStubPollution:
    """단독 실행은 통과하는데 전체 스위트에서 9건 실패했다.

    원인: tests/test_carry_exit_display_v22320.py가 빈 plotly 모듈을
    sys.modules에 심고 제거하지 않아, 뒤에 도는 테스트가 실제 plotly를 부분
    재import하며 **같은 클래스가 두 벌** 생겼다. 그 결과 px.treemap이 기본
    template을 검증할 때 다음으로 죽었다:
      ValueError: Invalid value of type 'plotly.graph_objs.layout._template.Template'
    v55.4의 ml_engine 스텁 오염과 같은 유형 — 같은 방식으로 막는다.
    """

    def test_plotly_in_sys_modules_is_real(self):
        import sys as _s
        pl = _s.modules.get("plotly")
        if pl is None:
            pytest.skip("plotly 미로드")
        assert not getattr(pl, "__is_test_stub__", False), \
            "plotly가 테스트 스텁으로 대체된 상태다 — 스텁 teardown 누락"
        assert getattr(pl, "__file__", None), "plotly에 __file__이 없다(스텁 의심)"

    def test_carry_test_cleans_up_its_stub(self):
        """이미 import된 모듈 객체로 검사한다 (다시 import하지 않는다).

        런타임에 테스트 모듈을 재 import하면 그 모듈의 최상위 import가 다시
        실행되어, 앞선 테스트가 남긴 sys.modules 상태에 따라 엉뚱하게 실패한다
        (실측: shared_utils가 부분 초기화 상태여서 ImportError). pytest는 수집
        단계에서 이미 모든 테스트 모듈을 import해 두므로 그것을 찾아 쓴다.
        """
        import sys as _s
        mod = _s.modules.get("test_carry_exit_display_v22320") or _s.modules.get(
            "tests.test_carry_exit_display_v22320")
        if mod is None:
            pytest.skip("carry 테스트 모듈 미수집 (단독 실행)")
        mod.teardown_module(mod)          # 이중 안전장치가 실제로 동작하는지
        left = _s.modules.get("plotly")
        assert left is None or not getattr(left, "__is_test_stub__", False)

    def test_thirdparty_isolation_guard_is_wired(self):
        """tests/conftest.py 가드가 실제로 물려 있는지 (선언만 있고 죽은 게이트 금지)."""
        import sys as _s
        cft = _s.modules.get("conftest") or _s.modules.get("tests.conftest")
        if cft is None:
            import importlib.util as _u
            spec = _u.spec_from_file_location(
                "_v59_conftest_probe", ROOT / "tests" / "conftest.py")
            cft = _u.module_from_spec(spec)
            spec.loader.exec_module(cft)
        assert "plotly" in cft.PROTECTED_PREFIXES
        assert "nicegui" in cft.PROTECTED_PREFIXES
        # 스냅샷이 비어 있으면 가드는 아무것도 검사하지 않는다(공허한 통과).
        if _s.modules.get("plotly") is not None and cft.__name__ in (
                "conftest", "tests.conftest"):
            assert cft._snapshot, "가드 스냅샷이 비어 있다 — 검사가 사실상 꺼져 있다"

    @staticmethod
    def _thirdparty_purges(src: str) -> set:
        """`del sys.modules[...]`을 **감싼 if 조건**이 서드파티 이름을 거르는지 본다.

        줄 단위 근접 검색은 오탐이 난다. 실제로 두 건이 걸렸다:
          - carry 테스트: 자기 스텁만 지운다(`__is_test_stub__` 조건) — 정상
          - pick_top1: 조건은 `mod == "components.tab_stocks"`이고 nicegui 코드가
            근처에 있을 뿐 — 정상
        그래서 del을 직접 지배하는 if의 test 노드만 검사한다.
        """
        import ast
        tree = ast.parse(src)
        libs = ("plotly", "nicegui")
        hits = set()

        def _targets_sys_modules(node) -> bool:
            for t in getattr(node, "targets", []):
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Attribute)
                        and t.value.attr == "modules"
                        and isinstance(t.value.value, ast.Name)
                        and t.value.value.id == "sys"):
                    return True
            return False

        def walk(node, guards):
            if isinstance(node, ast.Delete) and _targets_sys_modules(node):
                cond = " ".join(guards)
                for lib in libs:
                    # 'plotly' 뿐 아니라 startswith('plotly.') 형태도 잡는다
                    if any(tok in cond for tok in
                           (f"'{lib}'", f'"{lib}"', f"'{lib}.", f'"{lib}.')):
                        hits.add(lib)
                return
            if isinstance(node, ast.If):
                test_src = ast.unparse(node.test)
                for child in node.body:
                    walk(child, guards + [test_src])
                for child in node.orelse:
                    walk(child, guards)
                return
            for child in ast.iter_child_nodes(node):
                walk(child, guards)

        walk(tree, [])
        return hits

    def test_no_test_purges_thirdparty_from_sys_modules(self):
        """소스 수준 회귀 가드: 서드파티를 raw del 하는 테스트가 다시 생기면 잡는다."""
        offenders = []
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            for lib in sorted(self._thirdparty_purges(
                    path.read_text(encoding="utf-8"))):
                offenders.append(f"{path.name}: {lib}")
        assert not offenders, (
            "서드파티 모듈을 sys.modules에서 직접 삭제하는 테스트가 있다 "
            f"— 같은 클래스가 두 벌 생겨 스위트가 순서에 따라 깨진다: {offenders}")

    def test_purge_detector_actually_detects(self):
        """위 가드가 죽은 게이트가 아님을 확인 (mutation: 나쁜 패턴을 넣어본다)."""
        bad = (
            "import sys\n"
            "for mod in list(sys.modules):\n"
            "    if mod == 'plotly' or mod.startswith('nicegui.'):\n"
            "        del sys.modules[mod]\n"
        )
        assert self._thirdparty_purges(bad) == {"plotly", "nicegui"}
        good = (
            "import sys\n"
            "_N = ('plotly',)\n"
            "for n in _N:\n"
            "    if getattr(sys.modules.get(n), '__is_test_stub__', False):\n"
            "        del sys.modules[n]\n"
        )
        assert self._thirdparty_purges(good) == set(), "정상 스텁 정리를 오탐한다"

    def test_stub_installer_marks_and_removes(self):
        src = (ROOT / "tests" / "test_carry_exit_display_v22320.py").read_text(
            encoding="utf-8")
        assert "__is_test_stub__" in src, "스텁 표시가 없다"
        assert "def teardown_module" in src, "스텁 원복이 없다"
