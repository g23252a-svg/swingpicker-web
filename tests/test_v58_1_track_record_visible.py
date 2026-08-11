# -*- coding: utf-8 -*-
"""[v58.1] 만들어놓고 아무도 못 보던 성적 기록을 화면에 올린다.

배경
  v55가 픽 실현 분포(`services/pick_reliability`)를, v58이 알파 실전 성적
  (`services/alpha_live_report`)을 **매일 산출**하게 만들었다. 그런데 grep 결과
  `reliability_line`·`alpha_live_line` 둘 다 **components/ 어디에서도 호출되지
  않았다** — JSON을 직접 열어야 볼 수 있었다. v53이 고친 것과 같은 유형
  (엔진은 바뀌었고 화면은 모른다)이다.

이 파일이 고정하는 것
  1. 배선 생존 — 결정 화면이 두 리포트를 실제로 읽는다. 리포트를 만들고
     화면이 안 읽으면 사용자에게는 없는 기능이다.
  2. 리포트가 없거나 ok=False면 **아무것도 표시하지 않는다**. 빈 자리를
     낙관적 문구로 채우면 그게 v55~v58에서 걷어낸 과신이다.
  3. 표시 문구에 **표본 n이 포함**된다. n을 숨긴 숫자는 약속처럼 읽힌다.
  4. 리포트가 깨져 있어도 화면이 죽지 않는다(관측 레이어가 결정 화면을
     끌어내리면 안 된다).
  5. 라벨이 '과거 관측 · 미래 약속 아님'을 명시한다.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import components.decision_center as dc  # noqa: E402


# ── 1. 배선 생존 ──────────────────────────────────────────────────
class TestWiring:
    def test_decision_center_reads_both_reports(self):
        src = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")
        assert "alpha_live_line" in src, "알파 실전 성적을 화면이 읽지 않는다"
        assert "reliability_line" in src, "픽 실현 분포를 화면이 읽지 않는다"

    def test_summary_carries_track_record_key(self):
        df = pd.DataFrame([{"종목코드": "000001", "종목명": "T", "ROUTE": "WAIT"}])
        s = dc.build_decision_summary(df)
        assert "track_record" in s, "summary에 track_record가 없다"
        assert isinstance(s["track_record"], list)

    def test_renderer_guards_on_empty(self):
        """빈 리스트일 때 렌더 블록이 조건부여야 한다(빈 제목만 뜨면 안 됨)."""
        src = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")
        assert 'summary.get("track_record")' in src
        assert "과거 관측" in src, "'미래 약속이 아니다'라는 표시가 없다"


# ── 2. 없는 것을 있다고 하지 않는다 ───────────────────────────────
class TestHonestEmpty:
    def test_missing_reports_render_nothing(self, tmp_path):
        assert dc._track_record_lines(str(tmp_path)) == []

    def test_not_ok_report_renders_nothing(self, tmp_path):
        (tmp_path / "alpha_live_report_latest.json").write_text(
            json.dumps({"ok": False, "reason": "표본 없음"}), encoding="utf-8")
        (tmp_path / "pick_reliability_latest.json").write_text(
            json.dumps({"ok": False, "reason": "표본 없음"}), encoding="utf-8")
        assert dc._track_record_lines(str(tmp_path)) == []

    def test_corrupt_report_does_not_break_screen(self, tmp_path):
        (tmp_path / "alpha_live_report_latest.json").write_text("{not json",
                                                               encoding="utf-8")
        (tmp_path / "pick_reliability_latest.json").write_text("{not json",
                                                              encoding="utf-8")
        assert dc._track_record_lines(str(tmp_path)) == []   # 예외가 새면 안 된다


# ── 3. 표시되는 숫자는 표본을 데리고 온다 ─────────────────────────
class TestSampleShown:
    def _write_alpha(self, tmp_path, n_days=13):
        rep = {
            "ok": True, "asof": "20260807", "n_days": n_days,
            "horizons": {"h5": {
                "n_days": n_days, "ic_mean": 0.1155, "ic_median": 0.1197,
                "ic_positive_days": 11, "ic_t": {"n": n_days, "t": 2.128, "p": 0.0548},
                "top1": {"n": n_days, "mean_pct": -3.009, "median_pct": -3.216,
                         "win_rate": 0.3846, "cum_pct": -39.1,
                         "excess_mean_pct": -1.816, "stop_rate": 0.4615},
            }},
        }
        (tmp_path / "alpha_live_report_latest.json").write_text(
            json.dumps(rep, ensure_ascii=False), encoding="utf-8")

    def test_line_includes_sample_size(self, tmp_path):
        self._write_alpha(tmp_path)
        lines = dc._track_record_lines(str(tmp_path))
        assert lines, "리포트가 정상인데 라인이 없다"
        line = lines[0]
        assert "13일" in line, f"표본 n이 문구에 없다: {line}"
        assert "IC" in line

    def test_line_shows_the_bad_news_too(self, tmp_path):
        """성적이 나쁠 때 그 사실이 문구에 남아야 한다 — 좋은 것만 고르면 과신이다."""
        self._write_alpha(tmp_path)
        line = dc._track_record_lines(str(tmp_path))[0]
        assert "-3.01" in line or "-3.0" in line, f"음수 수익이 숨겨졌다: {line}"
        assert "스톱체결" in line, f"스톱 체결률이 빠졌다: {line}"

    def test_no_promise_words(self, tmp_path):
        """'확실'·'보장' 같은 약속형 표현이 성적 문구에 들어가면 안 된다."""
        self._write_alpha(tmp_path)
        line = dc._track_record_lines(str(tmp_path))[0]
        for bad in ("보장", "확실", "무조건", "반드시 오"):
            assert bad not in line, f"약속형 표현 '{bad}': {line}"
