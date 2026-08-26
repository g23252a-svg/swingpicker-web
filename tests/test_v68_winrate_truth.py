# -*- coding: utf-8 -*-
"""[v68] 과신 방지 캡이 8월 내내 조용히 작동하지 않았다.

■ 신호
  monotonicity 하드게이트 `declared_vs_realized_gap_15pp`가 2026-08-25에 처음
  FAIL(gap 26.2%p > 15%p). 8/24까지 통과(12.5%p)였고 main도 같은 날 같은
  이유로 실패한다 — 패치가 만든 것이 아니라 데이터가 진실을 드러낸 것이다.
    declared_wr_top_pick 0.4496 vs realized_wr_top_pick 0.1871 (n=30)

■ 고장 ① 픽이 사는 구간에 표본이 없어 캡이 붙지 않았다
  TOP_PICK은 ELITE_SCORE **[0, 50)**에 몰린다(08-25: 20종목 중 19 · 중위 21.4).
  그 구간 winrate_table 표본:
    08-11 n_raw=2  sufficient=False p_win=0.5000 ← 폴백 상수
    08-14 n_raw=2  (동일)          08-18 n_raw=2 (동일)
    08-21 n_raw=22 sufficient=False p_win=0.2500
    08-24 n_raw=32 sufficient=True  p_win=0.1929
    08-25 n_raw=43 sufficient=True  p_win=0.1669
  캡은 sufficient=True & n_raw>=30 bin만 신뢰하므로 8/24 전까지 커버 bin이
  아예 없었고, 그럴 때 `_clip_est_win_rate_to_realized_bins`는 **조용히 원본을
  돌려준다**(v56형 죽은 게이트). 그 사이 화면은 38~46%를 적었다.

■ 고장 ② 캡이 항상 한 배치 늦다
  compute_est_win_rate는 pipeline_calibrate(677)에서, winrate_table을 다시
  만드는 auto_calibrate는 pipeline_finalize(2110)에서 **그 뒤에** 돈다.
  그래서 08-24 배치는 08-21 표(n_raw=22, sufficient=False)를 보고 캡을
  건너뛰었고, 당일 표 기준으로는 28/28종목이 캡을 초과했다(최대 +11.8%p).

■ 왜 손실인가 — 켈리가 이 값을 쓴다
  f = p − (1−p)/b. 당일 표 기준 캡을 적용했다면
    08-24 선언 0.4501 → 0.3379 · 켈리 f 중위 0.2457 → 0.0910 = **수량 0.40배**
          26종목 중 6종목은 f≤0(진입 불가)
    08-25 선언 0.3320 → 0.3119 · 22종목 중 13종목이 f≤0
  8월 하락 구간에서 포지션이 약 2.5배 크게 잡혔다.

■ 이 파일이 고정하는 것
  1. 캡 산식·문턱이 kelly_calibrator와 **같다**(어긋나면 다른 것을 보고한다).
  2. 신뢰 bin이 없으면 **조용히 통과하지 않고** NO_SUFFICIENT_BIN으로 남는다.
  3. 날짜별 표를 지정해 읽을 수 있다(‘전날 표’ 문제를 진단하려면 필요하다).
  4. 실배치 전제 고정 — 08-21 전원 NO_SUFFICIENT_BIN · 08-24 전원 OVER_CAP.
  5. 화면·배치가 실제로 이 사실을 말한다(계산만 하고 안 쓰면 실패).
  6. **EST_WIN_RATE·켈리 수량은 바뀌지 않는다** — 진단 전용이다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.winrate_truth as WT  # noqa: E402

DATA = ROOT / "data"
DC_SRC = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")
PF_SRC = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")

TABLE = [
    {"score_lo": 0, "score_hi": 50, "p_win": 0.1669, "n_raw": 43, "sufficient": True},
    {"score_lo": 50, "score_hi": 60, "p_win": 0.3750, "n_raw": 6, "sufficient": False},
    {"score_lo": 60, "score_hi": 70, "p_win": 0.4468, "n_raw": 121, "sufficient": True},
    {"score_lo": 90, "score_hi": 100.01, "p_win": 0.4000, "n_raw": 18, "sufficient": False},
]


def _row(score=25.0, declared=0.45):
    return {"종목명": "T", WT.SCORE_COL: score, WT.DECLARED_COL: declared}


# ══════════════════════════════════════════════════════════════════
#  1. 캡 산식이 kelly_calibrator와 같은가
# ══════════════════════════════════════════════════════════════════
class TestCapMatchesSSOT:
    def test_constants_match_kelly_calibrator(self):
        import inspect
        import kelly_calibrator as kc
        sig = inspect.signature(kc._clip_est_win_rate_to_realized_bins)
        assert float(sig.parameters["max_gap"].default) == WT.MAX_GAP
        src = inspect.getsource(kc._clip_est_win_rate_to_realized_bins)
        assert f"< {WT.MIN_N_RAW}" in src or f"<{WT.MIN_N_RAW}" in src, \
            "n_raw 문턱이 어긋난다 — 이 모듈이 캡과 다른 것을 보고하게 된다"

    def test_cap_formula(self):
        # min(0.85, max(0.30, p_win + 0.145))
        assert WT.cap_value(25.0, TABLE) == pytest.approx(min(0.85, max(0.30, 0.1669 + 0.145)))
        assert WT.cap_value(65.0, TABLE) == pytest.approx(0.4468 + 0.145)

    def test_cap_floor_and_ceiling(self):
        low = [{"score_lo": 0, "score_hi": 50, "p_win": 0.01, "n_raw": 99, "sufficient": True}]
        assert WT.cap_value(10.0, low) == pytest.approx(0.30)
        high = [{"score_lo": 0, "score_hi": 50, "p_win": 0.99, "n_raw": 99, "sufficient": True}]
        assert WT.cap_value(10.0, high) == pytest.approx(0.85)

    def test_agrees_with_kelly_on_real_rows(self):
        """같은 입력에 대해 두 구현의 캡 판정이 일치해야 한다."""
        import kelly_calibrator as kc
        p = DATA / "recommend_20260825.csv"
        if not p.exists():
            pytest.skip("실데이터 없음")
        d = pd.read_csv(p, encoding="utf-8-sig", dtype={"종목코드": str}, low_memory=False)
        tp = d[pd.to_numeric(d["TOP_PICK"], errors="coerce").fillna(0) > 0]
        if not len(tp):
            pytest.skip("TOP_PICK 없음")
        # 캡이 **실제로 읽는** 표로 비교해야 한다. 리포트용 표로 비교하면
        # 두 구현이 아니라 두 파일을 비교하게 된다(고장 ③ 참고).
        table = WT.load_table(str(DATA), prefer="cap")
        if not table:
            pytest.skip("표 없음")
        kc._load_winrate_table_cached.cache_clear()
        sc = pd.to_numeric(tp[WT.SCORE_COL], errors="coerce").values
        wr = pd.to_numeric(tp[WT.DECLARED_COL], errors="coerce").values
        capped, _ = kc._clip_est_win_rate_to_realized_bins(
            sc, wr, str(DATA), method="ELITE_SCORE")
        mine = [WT.cap_value(s, table) for s in sc]
        for s, w, k, m in zip(sc, wr, capped, mine):
            expect = w if m is None else min(w, m)
            assert k == pytest.approx(expect, abs=1e-9), \
                f"score={s} 선언={w}: kelly={k} vs winrate_truth={expect}"


# ══════════════════════════════════════════════════════════════════
#  2. 조용한 미적용을 없앤다 (고장 ①)
# ══════════════════════════════════════════════════════════════════
class TestNoSilentPass:
    def test_insufficient_bin_is_reported_not_ignored(self):
        a = WT.assess(_row(score=55.0), TABLE)   # [50,60) n_raw=6 sufficient=False
        assert a["status"] == WT.STATUS_NO_BIN
        assert a["cap"] is None

    def test_insufficient_bin_still_names_the_bin_and_n(self):
        """왜 캡이 안 붙었는지 알 수 있어야 한다 — 침묵이 아니라 사유."""
        a = WT.assess(_row(score=55.0), TABLE)
        assert a["bin"] == "[50, 60)"
        assert a["n_raw"] == 6

    def test_n_raw_below_threshold_is_not_trusted(self):
        t = [{"score_lo": 0, "score_hi": 50, "p_win": 0.2, "n_raw": WT.MIN_N_RAW - 1,
              "sufficient": True}]
        assert WT.bin_for(10.0, t) is None
        assert WT.cap_value(10.0, t) is None

    def test_uncovered_score_is_no_bin(self):
        assert WT.assess(_row(score=85.0), TABLE)["status"] == WT.STATUS_NO_BIN

    def test_over_cap_is_flagged(self):
        a = WT.assess(_row(score=25.0, declared=0.45), TABLE)
        assert a["status"] == WT.STATUS_OVER
        assert a["gap_pp"] == pytest.approx((0.45 - 0.1669) * 100, abs=0.01)

    def test_within_cap_is_ok(self):
        assert WT.assess(_row(score=25.0, declared=0.28), TABLE)["status"] == WT.STATUS_OK

    def test_missing_inputs_are_unknown(self):
        assert WT.assess({"종목명": "x"}, TABLE)["status"] == WT.STATUS_UNKNOWN
        assert WT.assess(_row(), None)["status"] == WT.STATUS_NO_BIN

    def test_line_says_unverified_when_no_bin(self):
        line = WT.gap_line(_row(score=55.0, declared=0.47), TABLE)
        assert "검증되지 않았습니다" in line and "캡 적용 불가" in line

    def test_line_states_the_gap(self):
        line = WT.gap_line(_row(score=25.0, declared=0.45), TABLE)
        assert "실측 17%" in line and "높습니다" in line and "2.7배" in line


# ══════════════════════════════════════════════════════════════════
#  2-b. 캡이 읽는 표와 리포트가 읽는 표 (고장 ③)
# ══════════════════════════════════════════════════════════════════
class TestTableDivergence:
    def test_cap_and_report_orders_are_declared(self):
        import kelly_calibrator as kc
        import inspect
        src = inspect.getsource(kc._load_winrate_table_impl)
        assert "winrate_table_by_" in src and "winrate_table_latest" in src
        assert WT.CAP_TABLE_ORDER[0].startswith("winrate_table_by_"),             "캡의 탐색 순서를 흉내내지 못하면 캡이 무엇을 보는지 진단할 수 없다"

    @pytest.mark.skipif(
        not (DATA / "winrate_table_by_ELITE_SCORE_latest.json").exists(),
        reason="실데이터 없음")
    def test_real_divergence_is_detected(self):
        d = WT.table_divergence(str(DATA))
        assert d["diverged"] is True,             "두 표가 같아졌다면 고장 ③이 고쳐진 것이다 — 전제를 갱신하라"
        assert d["cap_n"] < d["report_n"],             f"캡 표본 {d['cap_n']} vs 리포트 {d['report_n']}"
        assert "다른 표를 읽고 있습니다" in d["line"]

    @pytest.mark.skipif(
        not (DATA / "winrate_table_by_ELITE_SCORE_latest.json").exists(),
        reason="실데이터 없음")
    def test_cap_table_has_zero_sample_fallback_bins(self):
        """n_raw=0인데 p_win=0.45가 적힌 구간 — 그 구간 픽은 절대 캡되지 않는다."""
        cap = WT.load_table(str(DATA), prefer="cap")
        zero = [r for r in cap if float(r.get("n_raw") or 0) == 0]
        assert zero, "표본 0건 구간이 사라졌다면 전제를 갱신하라"
        for r in zero:
            assert WT.bin_for((float(r["score_lo"]) + float(r["score_hi"])) / 2,
                              cap) is None

    def test_prefer_report_ignores_cap_only_file(self, tmp_path):
        import json
        d = tmp_path / "data"
        d.mkdir()
        thin = [{"score_lo": 0, "score_hi": 100, "p_win": 0.9, "n_raw": 99,
                 "sufficient": True}]
        rich = [{"score_lo": 0, "score_hi": 100, "p_win": 0.2, "n_raw": 99,
                 "sufficient": True}]
        (d / "winrate_table_by_ELITE_SCORE_latest.json").write_text(
            json.dumps({"table": thin}), encoding="utf-8")
        (d / "winrate_table_latest.json").write_text(
            json.dumps({"table": rich}), encoding="utf-8")
        assert WT.load_table(str(d), prefer="cap")[0]["p_win"] == 0.9
        assert WT.load_table(str(d), prefer="report")[0]["p_win"] == 0.2


# ══════════════════════════════════════════════════════════════════
#  3. 날짜별 표 (고장 ② 진단에 필요)
# ══════════════════════════════════════════════════════════════════
class TestDatedTable:
    @pytest.mark.skipif(not (DATA / "winrate_table_20260821.json").exists(),
                        reason="실데이터 없음")
    def test_dated_load_differs_from_latest(self):
        old = WT.load_table(str(DATA), "20260821")
        new = WT.load_table(str(DATA), "20260825")
        assert old and new
        b_old = next(r for r in old if float(r["score_lo"]) == 0)
        b_new = next(r for r in new if float(r["score_lo"]) == 0)
        assert float(b_old["n_raw"]) < float(b_new["n_raw"]), \
            "날짜별 표가 실제로 다르지 않다 — '전날 표' 문제를 진단할 수 없다"

    def test_missing_date_falls_back_to_latest(self, tmp_path):
        import json
        d = tmp_path / "data"
        d.mkdir()
        (d / "winrate_table_latest.json").write_text(
            json.dumps({"table": TABLE}), encoding="utf-8")
        assert WT.load_table(str(d), "29991231") is not None

    def test_no_table_returns_none(self, tmp_path):
        assert WT.load_table(str(tmp_path)) is None


# ══════════════════════════════════════════════════════════════════
#  4. 실배치 전제 고정
# ══════════════════════════════════════════════════════════════════
class TestRealBatches:
    def _tp(self, ymd):
        p = DATA / f"recommend_{ymd}.csv"
        if not p.exists():
            pytest.skip(f"{ymd} 없음")
        d = pd.read_csv(p, encoding="utf-8-sig", dtype={"종목코드": str}, low_memory=False)
        return d, pd.to_numeric(d["TOP_PICK"], errors="coerce").fillna(0) > 0

    def test_0821_had_no_trustworthy_bin(self):
        d, m = self._tp("20260821")
        s = WT.summary(d, WT.load_table(str(DATA), "20260821"), mask=m)
        assert s["no_bin"] == s["n"] > 0, \
            "08-21에 캡이 붙을 수 있었다면 전제가 바뀐 것이다"
        assert "검증되지 않았습니다" in s["line"]

    def test_0824_every_pick_exceeds_the_cap(self):
        d, m = self._tp("20260824")
        s = WT.summary(d, WT.load_table(str(DATA), "20260824"), mask=m)
        assert s["over_cap"] == s["n"] > 0, f"캡 초과 {s['over_cap']}/{s['n']}"
        assert s["gap_mean_pp"] is not None and s["gap_mean_pp"] > 20, \
            f"격차 {s['gap_mean_pp']}%p — 조사 시점(26%p)과 크게 다르다"

    def test_picks_live_in_the_low_score_bin(self):
        """캡이 못 붙은 이유의 전제 — 픽이 [0,50)에 몰린다."""
        d, m = self._tp("20260825")
        sc = pd.to_numeric(d.loc[m, WT.SCORE_COL], errors="coerce").dropna()
        assert (sc < 50).mean() >= 0.8, f"[0,50) 비율 {(sc < 50).mean():.2f}"

    def test_annotation_does_not_touch_money(self):
        d, _ = self._tp("20260825")
        out = WT.annotate(d, WT.load_table(str(DATA), "20260825"))
        for col in (WT.DECLARED_COL, "켈리_수량", "PRODUCTION_BUY", "TOP_PICK"):
            if col not in d.columns:
                continue
            a = pd.to_numeric(d[col], errors="coerce").fillna(-1)
            b = pd.to_numeric(out[col], errors="coerce").fillna(-1)
            assert int((a != b).sum()) == 0, f"{col}이 바뀌었다"
        for c in (WT.COL_STATUS, WT.COL_GAP, WT.COL_REALIZED, WT.COL_BIN_N):
            assert c in out.columns


# ══════════════════════════════════════════════════════════════════
#  5. 화면·배치가 실제로 말하는가
# ══════════════════════════════════════════════════════════════════
class TestActuallyWired:
    def test_today_tab_renders_it(self):
        assert "_render_winrate_truth(summary)" in DC_SRC
        assert '"winrate_truth": _winrate_truth(df)' in DC_SRC

    def test_screen_says_it_feeds_sizing(self):
        """'표시용 숫자'로 읽히면 안 된다 — 켈리 입력이라는 사실을 적는다."""
        i = DC_SRC.find("def _render_winrate_truth")
        j = DC_SRC.find("def _render_holdings", i)
        assert "켈리" in DC_SRC[i:j]

    def test_batch_annotates_and_logs(self):
        assert "winrate_truth as _wt" in PF_SRC
        assert "[v68] 선언 승률 검증" in PF_SRC

    def test_batch_block_is_diagnostic_only(self):
        i = PF_SRC.find("[v68] 선언 승률")
        j = PF_SRC.find("[v67] 보유 청산 규율", i)
        block = PF_SRC[i:j]
        assert "EST_WIN_RATE" not in block.split('"""')[-1] or "df_out[" not in block
        for col in ("켈리_수량", "PRODUCTION_BUY"):
            assert f'df_out["{col}"]' not in block, f"v68 블록이 {col}을 쓴다"

    def test_batch_alarms_on_table_divergence(self):
        assert "table_divergence(OUT_DIR)" in PF_SRC
        assert "🚨 [v68]" in PF_SRC

    def test_screen_shows_divergence(self):
        assert '_WT.table_divergence("data")' in DC_SRC
        assert '_dv.get("diverged")' in DC_SRC

    def test_batch_reads_the_current_day_table(self):
        """'전날 표' 문제를 진단하려면 배치는 당일 표를 지정해 읽어야 한다."""
        assert "_wt.load_table(OUT_DIR, trade_ymd)" in PF_SRC
