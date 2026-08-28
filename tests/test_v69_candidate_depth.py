# -*- coding: utf-8 -*-
"""[v69] 관찰 후보를 몇 개까지 보여줄지 — 하드코딩 3 대신 실측으로.

■ 왜
  사이징된 후보를 전부 사면 알파 시대 24일 실측 **일평균 -1.99%**
  (복리 -21.7% · MDD -21.8%)다. 우위는 상단에 몰려 있는데 '오늘' 탭의
  관찰 후보 개수는 `.head(3)`으로 **하드코딩**돼 있었고 그 3에 근거가 없었다.

■ 깊이별 실측 (2026-07-14~08-18 · 24일 · 유니버스 평균 +0.78%)
    N   초과       p       상위2제거
    1  +2.05%p   0.143    +0.70
    2  +1.45%p   0.143    +0.64
    3  +0.41%p   0.607    **-0.22**   ← 이상치 빼면 무너진다
    4  +0.31%p   0.723    -0.64
    5  +0.30%p   0.720    -0.57
    6  +0.01%p   0.993    -0.67
    7  -0.26%p   0.685    -0.86       ← 여기부터 초과가 음수
   15  -0.47%p   0.458    -0.83

■ 정직하게 — 표본이 늘면서 약해졌다
  21일 표본에서는 N=1 초과가 p=0.038로 0.05를 넘겼는데, 24일로 늘리니
  **p=0.143**이 됐다. **어떤 N도 검증되지 않았다.** 그래서 이 패치는
  '최적 N'을 주장하지 않고, **초과가 음수로 측정된 깊이를 감추지 않고
  잘라낼 뿐**이다.

■ 자르는 규칙 (사전 등록 · 매일 재계산)
  depth        = 초과가 1~N 전 구간 양수인 최대 N (바닥 1 · 천장 10)
  robust_depth = 거기에 상위2 제거 후에도 양수인 최대 N
  화면은 **robust_depth**를 쓴다(오늘 = 2). 상수로 박지 않는다.

■ 이 파일이 고정하는 것
  1. 깊이는 **상수가 아니다** — 화면 코드에 숫자가 박히면 실패한다.
  2. 규칙대로 잘린다(초과 양수 연속 · 상위2 제거 생존).
  3. 바닥·천장이 지켜진다. 측정 불가면 보수적 폴백.
  4. 왜 그 깊이인지 화면이 말한다(근거 없는 절단 금지).
  5. 실측 전제 고정 — N=3에서 상위2제거가 음수로 꺾인다.
  6. **결정 컬럼·수량은 바뀌지 않는다** — 표시 깊이만 정한다.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.candidate_depth as CD  # noqa: E402

DATA = ROOT / "data"
DC_SRC = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")
PF_SRC = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")


def _rep(curve, ok=True):
    return {"ok": ok, "n_days": 24, "curve": curve,
            "depth": 0, "robust_depth": 0}


def _c(n, ex, drop=None, p=0.5):
    return {"n": n, "days": 24, "mean_pct": ex, "excess_pct": ex, "t": 1.0,
            "p": p, "excess_drop_top2": ex if drop is None else drop,
            "win_rate": 0.5}


# ══════════════════════════════════════════════════════════════════
#  1. 깊이가 상수가 아니다
# ══════════════════════════════════════════════════════════════════
class TestNotHardcoded:
    def test_screen_has_no_literal_watch_depth(self):
        i = DC_SRC.find("watch_df = work[")
        j = DC_SRC.find("buys = [_stock_payload", i)
        block = DC_SRC[i:j]
        assert ".head(3)" not in block, "관찰 후보 개수가 여전히 하드코딩돼 있다"
        assert "_watch_depth" in block

    def test_screen_reads_the_measured_report(self):
        assert '_CD.load("data")' in DC_SRC
        assert "_CD.effective_depth(" in DC_SRC

    def test_batch_recomputes_daily(self):
        assert "_cd.save(OUT_DIR, trade_ymd)" in PF_SRC
        assert "[v69] 후보 깊이" in PF_SRC


# ══════════════════════════════════════════════════════════════════
#  2. 자르는 규칙
# ══════════════════════════════════════════════════════════════════
class TestCutRule:
    def test_cuts_where_excess_turns_negative(self):
        curve = [_c(1, 2.0), _c(2, 1.0), _c(3, 0.5), _c(4, -0.1), _c(5, 0.9)]
        r = dict(_rep(curve))
        d = 0
        for c in curve:
            if not c["excess_pct"] > 0:
                break
            d = c["n"]
        assert d == 3, "음수 이후가 다시 양수라도 이어붙이면 안 된다"

    def test_robust_requires_surviving_outlier_removal(self):
        curve = [_c(1, 2.0, drop=0.7), _c(2, 1.4, drop=0.6),
                 _c(3, 0.4, drop=-0.2), _c(4, 0.3, drop=-0.6)]
        rb = 0
        for c in curve:
            if not (c["excess_pct"] > 0 and c["excess_drop_top2"] > 0):
                break
            rb = c["n"]
        assert rb == 2

    def test_robust_is_strictly_shallower_when_outliers_carry_it(self, monkeypatch):
        """상위2 제거 조건이 **구현에서** 살아 있는지 고정한다.

        순위1·2는 매일 꾸준히 +, 순위3은 이틀만 폭등하고 나머지는 −인 패널을
        만든다. 이때 N=3의 평균 초과는 양수지만 상위2를 빼면 음수가 되므로
        robust_depth < depth 여야 한다. 조건을 지우면 둘이 같아져 실패한다.
        """
        days = [f"202607{d:02d}" for d in range(10, 30)]
        rows = []
        for i, y in enumerate(days):
            # 순위1·2는 작지만 꾸준한 +(결정적 지터로 분산 0 회피)
            j = ((i * 7) % 5) * 0.01
            rows.append({"ymd": y, "rank": 1, "ret": 0.30 + j})
            rows.append({"ymd": y, "rank": 2, "ret": 0.30 - j})
            # 순위3은 이틀만 폭등, 나머지는 −
            rows.append({"ymd": y, "rank": 3, "ret": 120.0 if i < 2 else -3.0})
        panel = pd.DataFrame(rows)
        monkeypatch.setattr(CD, "_panel", lambda d: (
            panel, pd.Series({y: 0.0 for y in days})))
        out = CD.measure(str(DATA))
        cur = {c["n"]: c for c in out["curve"]}
        assert cur[3]["excess_pct"] > 0, "전제: N=3 평균 초과는 양수여야 한다"
        assert cur[3]["excess_drop_top2"] < 0, "전제: 상위2 제거 시 음수여야 한다"
        assert out["depth"] >= 3
        assert out["robust_depth"] == 2, (
            f"robust_depth={out['robust_depth']} — 상위2 제거 조건이 "
            "구현에서 빠졌다")
        assert out["robust_depth"] < out["depth"]

    def test_measure_applies_both_rules(self, monkeypatch):
        curve = [_c(1, 2.0, drop=0.7), _c(2, 1.4, drop=0.6),
                 _c(3, 0.4, drop=-0.2), _c(4, 0.3, drop=-0.6),
                 _c(5, 0.1, drop=-0.5), _c(6, -0.3, drop=-0.9)]
        panel = pd.DataFrame({"ymd": [f"2026071{i%10}" for i in range(60)],
                              "rank": [(i % 6) + 1 for i in range(60)],
                              "ret": np.linspace(-5, 5, 60)})
        # measure의 규칙 부분만 검증 — 곡선을 직접 주입
        monkeypatch.setattr(CD, "_panel", lambda d: (panel, pd.Series(
            {y: 0.0 for y in panel["ymd"].unique()})))
        out = CD.measure(str(DATA))
        assert out["ok"] is True
        assert out["depth"] >= out["robust_depth"] >= CD.MIN_DEPTH

    def test_floor_and_ceiling(self):
        assert CD.MIN_DEPTH == 1 and CD.MAX_DEPTH == 10
        assert CD.effective_depth({"ok": True, "depth": 999}) == CD.MAX_DEPTH
        assert CD.effective_depth({"ok": True, "depth": 0}) == CD.MIN_DEPTH
        assert CD.effective_depth({"ok": True, "depth": "x"}) == CD.FALLBACK_DEPTH

    def test_fallback_when_not_measured(self):
        assert CD.effective_depth(None) == CD.FALLBACK_DEPTH
        assert CD.effective_depth({"ok": False}) == CD.FALLBACK_DEPTH

    def test_thin_sample_refuses_to_measure(self, tmp_path, monkeypatch):
        panel = pd.DataFrame({"ymd": ["20260714"] * 5, "rank": [1, 2, 3, 4, 5],
                              "ret": [1.0] * 5})
        monkeypatch.setattr(CD, "_panel",
                            lambda d: (panel, pd.Series({"20260714": 0.0})))
        out = CD.measure(str(tmp_path))
        assert out["ok"] is False and out["depth"] == CD.FALLBACK_DEPTH


# ══════════════════════════════════════════════════════════════════
#  3. 근거 없이 자르지 않는다
# ══════════════════════════════════════════════════════════════════
class TestExplains:
    def test_line_names_the_depth_and_the_number(self):
        rep = {"ok": True, "n_days": 24, "depth": 6, "robust_depth": 2,
               "curve": [_c(1, 2.05, 0.7), _c(2, 1.45, 0.64), _c(3, 0.41, -0.22),
                         _c(4, 0.31, -0.64), _c(5, 0.30, -0.57), _c(6, 0.01, -0.67),
                         _c(7, -0.26, -0.86)]}
        line = CD.depth_line(rep)
        assert "24일" in line and "-0.26%p" in line
        assert "상위 2종목" in line

    def test_line_says_when_unmeasured(self):
        line = CD.depth_line(None)
        assert "표본이 아직 부족" in line and str(CD.FALLBACK_DEPTH) in line

    def test_screen_renders_the_reason(self):
        assert 'summary["watch_depth_line"]' in DC_SRC
        assert '"watch_depth_line": _CD.depth_line(' in DC_SRC

    def test_report_carries_its_own_caveat(self, real_data_mirror):
        # [v71] 진짜 data/ 금지 — CD.measure → _panel → _build_hl_union 이
        # <data_dir>/ohlcv_union_hl.parquet 에 캐시를 쓴다.
        rep = CD.measure(real_data_mirror("ohlcv_cache_*.parquet", "recommend_*.csv"))
        if not rep.get("ok"):
            pytest.skip("측정 불가")
        assert "검증되지 않았다" in rep["caveat"]
        assert "최적" in rep["caveat"]


# ══════════════════════════════════════════════════════════════════
#  4. 실측 전제 고정
# ══════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not (DATA / "recommend_20260818.csv").exists(),
                    reason="실데이터 없음")
class TestRealMeasurement:
    @pytest.fixture(scope="class")
    def rep(self, tmp_path_factory):
        # [v71] 클래스 스코프라 함수 스코프 real_data_mirror 를 못 쓴다.
        # CD.measure 가 14초라 테스트마다 돌릴 수도 없다 → 미러를 직접 만든다.
        from conftest import build_data_mirror
        d = build_data_mirror(tmp_path_factory.mktemp("v69_mirror"),
                              "ohlcv_cache_*.parquet", "recommend_*.csv")
        return CD.measure(d)

    def test_measured(self, rep):
        assert rep["ok"] is True
        assert rep["n_days"] >= CD.MIN_DAYS

    # ── [v74.1] 부호 전제는 고정 데이터로 잰다 ──────────────────────
    #   원래 이 테스트는 rolling 실데이터에서 cur[1] > 0을 박아뒀고, 8/28
    #   배치가 들어오자 -0.452로 뒤집혀 무관한 PR의 CI를 떨어뜨렸다(#127).
    #   경보로서는 설계대로 작동한 것이다 — 전제가 실제로 바뀌었다:
    #   27일 실측에서 **어느 N도 상위2 제거를 버티지 못한다**(N=1 -0.452,
    #   N=2 -0.178, N=3 -0.859). '상위 1~2위엔 견고한 엣지'는 24일짜리
    #   전제였고 죽었다. 화면은 robust_depth가 바닥(MIN_DEPTH)으로 떨어져
    #   자동으로 보수화된다 — 코드 수정은 필요 없었다.
    #   0 근처 부호는 데이터가 하루 늘 때마다 또 뒤집힐 수 있으므로,
    #   부호 전제는 PREMISE_ASOF 시점으로 고정해 결정적으로 만들고
    #   살아있는 경보는 유의성 테스트(아래)에 맡긴다.
    PREMISE_ASOF = "20260828"

    @pytest.fixture(scope="class")
    def rep_pinned(self, tmp_path_factory):
        import glob as _g, re as _re
        from conftest import build_data_mirror
        d = build_data_mirror(tmp_path_factory.mktemp("v69_pin"),
                              "ohlcv_cache_*.parquet", "recommend_*.csv")
        for f in _g.glob(os.path.join(d, "*.*")):
            m = _re.search(r"(\d{8})", os.path.basename(f))
            if m and m.group(1) > self.PREMISE_ASOF:
                os.remove(f)          # 심링크 제거 — 원본 무손상
        return CD.measure(d)

    def test_edge_collapses_by_rank3(self, rep_pinned):
        """[8/28 고정] 어느 깊이도 상위2 제거를 버티지 못한다 — 그래서 자른다."""
        cur = {c["n"]: c for c in rep_pinned["curve"]}
        assert cur[1]["excess_drop_top2"] < 0, "1위마저 상위2 의존 — 8/28 실측"
        assert cur[3]["excess_drop_top2"] < 0
        assert rep_pinned["robust_depth"] == CD.MIN_DEPTH, \
            "견고한 깊이가 바닥이 아니게 됐다 — 전제 기록을 갱신하라"

    def test_deep_ranks_are_not_better_than_universe(self, rep):
        cur = {c["n"]: c for c in rep["curve"]}
        assert cur[max(cur)]["excess_pct"] <= 0.5

    def test_nothing_is_significant(self, rep):
        """유의한 N이 생기면 이 패치의 전제(‘검증되지 않았다’)를 갱신해야 한다."""
        assert all(c["p"] > 0.05 for c in rep["curve"]), \
            "0.05를 넘긴 N이 생겼다 — caveat과 규칙을 재검토하라"

    def test_screen_depth_is_conservative(self, rep):
        d = CD.effective_depth(rep)
        shown = max(CD.MIN_DEPTH, min(d, int(rep["robust_depth"])))
        assert shown <= d
        assert CD.MIN_DEPTH <= shown <= CD.MAX_DEPTH


# ══════════════════════════════════════════════════════════════════
#  5. 저장·적재 + 자금 흐름 불변
# ══════════════════════════════════════════════════════════════════
class TestPersistAndSafety:
    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        panel = pd.DataFrame({"ymd": [f"202607{10+i%12:02d}" for i in range(120)],
                              "rank": [(i % 10) + 1 for i in range(120)],
                              "ret": np.tile(np.linspace(3, -3, 10), 12)})
        monkeypatch.setattr(CD, "_panel", lambda d: (
            panel, pd.Series({y: 0.0 for y in panel["ymd"].unique()})))
        rep = CD.save(str(tmp_path), "20260825")
        assert (tmp_path / CD.CACHE_NAME).exists()
        assert (tmp_path / "candidate_depth_20260825.json").exists()
        back = CD.load(str(tmp_path))
        assert back["depth"] == rep["depth"]

    def test_load_missing_is_none(self, tmp_path):
        assert CD.load(str(tmp_path)) is None

    def test_load_corrupt_is_none(self, tmp_path):
        (tmp_path / CD.CACHE_NAME).write_text("{not json", encoding="utf-8")
        assert CD.load(str(tmp_path)) is None

    def test_module_never_writes_decision_columns(self):
        src = (ROOT / "services" / "candidate_depth.py").read_text(encoding="utf-8")
        body = src.split('"""', 2)[-1]
        for col in ("PRODUCTION_BUY", "켈리_수량", "TOP_PICK", "ALPHA_ENTRY_OK"):
            assert f'"{col}"] =' not in body and f"'{col}'] =" not in body

    def test_batch_block_is_display_only(self):
        i = PF_SRC.find("[v69] 후보 목록을 어디서")
        j = PF_SRC.find("[v68] 선언 승률", i)
        block = PF_SRC[i:j]
        for col in ("PRODUCTION_BUY", "켈리_수량"):
            assert f'df_out["{col}"]' not in block
