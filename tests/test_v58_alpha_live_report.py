# -*- coding: utf-8 -*-
"""[v58] 알파 실전 성적 야간 리포트 — 재는 것이 맞는지, 그리고 조용히 죽지 않는지.

배경
  v29(2026-07-14) 이후 "알파가 요즘 맞나"를 물을 때마다 임시 스크립트를 짜야 했고,
  그래서 2026-08-10에도 최근 구간을 답할 수 없었다. axis_ic_report(v28)가 룰
  기반 축을 매일 감사하는 것과 같은 이유로 진입 SSOT인 알파도 성적을 남긴다.

이 파일이 고정하는 것
  1. 실현수익 정의가 프로덕션과 같다 — 진입 t+1 시가 · -8% 장중 스톱 · t+h 종가.
     스톱 상수는 pick_reliability에서 읽는다(어긋나면 화면 약속과 다른 걸 잰다).
  2. IC 계산이 맞는 방향이다(알파가 실제로 맞는 합성 데이터에서 IC > 0).
  3. 선행수익이 확정되지 않은 날은 조용히 통과하지 않고 집계에서 빠진다.
  4. IC 역주행은 경고를 낸다. 반대로 **수익 평균이 음수인 것만으로는 경고하지
     않는다** — 폭락 구간에서는 정상일 수 있고, 그걸 경고로 만들면 늑대소년이 된다.
  5. 리포트에 n·정의·주의문이 항상 담긴다(v55.4/v56 계열: 숫자만 남기면 과신).
  6. 배치 배선이 살아 있다 — pipeline_finalize가 이 모듈을 호출한다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.alpha_live_report as alr  # noqa: E402

DAYS = ["20260714", "20260715", "20260716", "20260717", "20260720",
        "20260721", "20260722", "20260723", "20260724", "20260727"]


def _write_panel(tmp: Path, alpha_works: bool = True, n_codes: int = 20,
                 rec_days=None, drop_alpha_col: bool = False):
    """합성 배치 패널.

    alpha_works=True  → 알파 점수가 높을수록 실제로 더 오른다(IC>0 기대)
    alpha_works=False → 반대로 움직인다(IC<0 → 경고 기대)
    """
    tmp.mkdir(parents=True, exist_ok=True)
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    rec_days = rec_days or DAYS[:6]

    # 가격: 종목별 일정 일간수익률 — 알파 순위와 (역)상관
    base = 10000.0
    price = {c: base for c in codes}
    for di, ymd in enumerate(DAYS):
        rows = []
        for ci, c in enumerate(codes):
            # 알파 상위(작은 ci)가 더 오르게 / 반대로
            slope = (n_codes - ci) if alpha_works else (ci + 1)
            drift = 0.002 * (slope - n_codes / 2) / (n_codes / 2)
            p_prev = price[c]
            p = p_prev * (1.0 + drift)
            price[c] = p
            rows.append({"종목코드": c, "종목명": f"T{c}", "시장": "KOSPI",
                         "시가": round(p_prev, 1), "종가": round(p, 1),
                         "고가": round(max(p_prev, p) * 1.01, 1),
                         "저가": round(min(p_prev, p) * 0.995, 1)})
        pd.DataFrame(rows).to_csv(tmp / f"price_snapshot_{ymd}.csv",
                                  index=False, encoding="utf-8-sig")

    for ymd in rec_days:
        rows = []
        for ci, c in enumerate(codes):
            r = {"종목코드": c, "종목명": f"T{c}", "ROUTE": "WAIT"}
            if not drop_alpha_col:
                r["ALPHA_SCORE"] = float(100 - ci)      # ci=0 이 최상위
            rows.append(r)
        pd.DataFrame(rows).to_csv(tmp / f"recommend_{ymd}.csv",
                                  index=False, encoding="utf-8-sig")
    return tmp


# ── 1. 실현 정의 SSOT ─────────────────────────────────────────────
class TestDefinition:
    def test_stop_pct_comes_from_pick_reliability(self):
        from services.pick_reliability import STOP_PCT_HOLD
        assert alr._stop_pct() == float(STOP_PCT_HOLD)

    def test_stop_is_applied_intraday(self, tmp_path):
        """저가가 스톱을 찍으면 종가와 무관하게 스톱 수익이어야 한다."""
        d = tmp_path / "data"
        d.mkdir()
        rows_a = [{"종목코드": "000001", "종목명": "A", "시가": 1000,
                   "종가": 1000, "고가": 1010, "저가": 995}]
        # t+1: 시가 1000 진입, 저가 800(-20%) → 스톱, 종가는 1100로 회복
        rows_b = [{"종목코드": "000001", "종목명": "A", "시가": 1000,
                   "종가": 1100, "고가": 1120, "저가": 800}]
        rows_c = [{"종목코드": "000001", "종목명": "A", "시가": 1100,
                   "종가": 1200, "고가": 1210, "저가": 1090}]
        for ymd, rs in (("20260714", rows_a), ("20260715", rows_b), ("20260716", rows_c)):
            pd.DataFrame(rs).to_csv(d / f"price_snapshot_{ymd}.csv",
                                    index=False, encoding="utf-8-sig")
        px, days = alr._load_snapshots(str(d))
        got = alr._realized(px, days, "000001", "20260714", 2, -0.08)
        assert got == pytest.approx(-0.08), f"스톱이 적용되지 않았다: {got}"

    def test_unmatured_day_returns_nan(self, tmp_path):
        d = _write_panel(tmp_path / "data")
        px, days = alr._load_snapshots(str(d))
        # 마지막 날 + h 는 미래 → NaN (조용히 0으로 세면 안 된다)
        assert not np.isfinite(alr._realized(px, days, "000001", days[-1], 3, -0.08))

    def test_report_carries_definition_and_caveat(self, tmp_path):
        d = _write_panel(tmp_path / "data")
        r = alr.compute_alpha_live_report(str(d))
        assert r["ok"]
        assert "시가" in r["definition"] and "스톱" in r["definition"]
        assert r["caveat"] and "판정을 내리지 않는다" in r["caveat"]
        assert r["n_days"] >= 1 and r["alpha_live_from"] == alr.ALPHA_LIVE_FROM


# ── 2. IC 방향이 맞는가 ───────────────────────────────────────────
class TestIC:
    def test_positive_ic_when_alpha_works(self, tmp_path):
        d = _write_panel(tmp_path / "data", alpha_works=True)
        r = alr.compute_alpha_live_report(str(d))
        blk = r["horizons"]["h3"]
        assert blk["ic_mean"] > 0.5, f"알파가 맞는 데이터인데 IC가 낮다: {blk['ic_mean']}"
        assert blk["ic_positive_days"] == blk["n_days"]
        assert not r["warnings"], "정상 데이터에 경고가 붙었다"

    def test_warns_when_alpha_is_inverted(self, tmp_path):
        d = _write_panel(tmp_path / "data", alpha_works=False)
        r = alr.compute_alpha_live_report(str(d))
        blk = r["horizons"]["h3"]
        assert blk["ic_mean"] < 0
        assert r["warnings"], "IC 역주행인데 경고가 없다"
        assert "역주행" in r["warnings"][0]

    def test_negative_returns_alone_do_not_warn(self, tmp_path):
        """전 종목이 하락해도(폭락 구간) IC가 정상이면 경고하지 않는다."""
        d = tmp_path / "data"
        d.mkdir()
        codes = [f"{i:06d}" for i in range(1, 21)]
        price = {c: 10000.0 for c in codes}
        for ymd in DAYS:
            rows = []
            for ci, c in enumerate(codes):
                p_prev = price[c]
                # 전부 하락하지만 알파 상위가 덜 하락 → IC>0, 수익은 음수
                p = p_prev * (1.0 - 0.01 - 0.001 * ci)
                price[c] = p
                rows.append({"종목코드": c, "종목명": f"T{c}", "시가": round(p_prev, 1),
                             "종가": round(p, 1), "고가": round(p_prev, 1),
                             "저가": round(p * 0.999, 1)})
            pd.DataFrame(rows).to_csv(d / f"price_snapshot_{ymd}.csv",
                                      index=False, encoding="utf-8-sig")
        for ymd in DAYS[:6]:
            pd.DataFrame([{"종목코드": c, "종목명": f"T{c}",
                           "ALPHA_SCORE": float(100 - ci), "ROUTE": "WAIT"}
                          for ci, c in enumerate(codes)]).to_csv(
                d / f"recommend_{ymd}.csv", index=False, encoding="utf-8-sig")
        r = alr.compute_alpha_live_report(str(d))
        blk = r["horizons"]["h3"]
        assert blk["top1"]["mean_pct"] < 0, "수익은 음수여야 하는 설정"
        assert blk["ic_mean"] > 0
        assert not r["warnings"], "수익 음수만으로 경고하면 늑대소년이 된다"


# ── 3. 없는 것을 있다고 하지 않는다 ───────────────────────────────
class TestHonestEmpty:
    def test_no_alpha_column_is_not_ok(self, tmp_path):
        d = _write_panel(tmp_path / "data", drop_alpha_col=True)
        r = alr.compute_alpha_live_report(str(d))
        assert r["ok"] is False and "확정된 배치일 없음" in r["reason"]

    def test_pre_live_batches_are_excluded(self, tmp_path):
        """알파 도입 전(2026-07-14 이전) 배치는 집계에 들어가면 안 된다."""
        d = _write_panel(tmp_path / "data", rec_days=["20260714", "20260715"])
        old = pd.read_csv(d / "recommend_20260714.csv", encoding="utf-8-sig",
                          dtype={"종목코드": str})
        old.to_csv(d / "recommend_20260601.csv", index=False, encoding="utf-8-sig")
        r = alr.compute_alpha_live_report(str(d))
        ymds = [x["ymd"] for x in r["per_day"]]
        assert "20260601" not in ymds
        assert all(y >= alr.ALPHA_LIVE_FROM for y in ymds)

    def test_load_returns_none_when_not_ok(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / alr.CACHE_NAME).write_text(json.dumps({"ok": False}), encoding="utf-8")
        assert alr.load_alpha_live_report(str(d)) is None

    def test_line_is_empty_without_report(self):
        assert alr.alpha_live_line(None) == ""
        assert alr.alpha_live_line({"ok": False}) == ""


# ── 4. 저장·로드 왕복 + 한 줄 요약 ────────────────────────────────
class TestPersistence:
    def test_save_writes_dated_and_latest(self, tmp_path):
        d = _write_panel(tmp_path / "data")
        r = alr.save_alpha_live_report(str(d), "20260717")
        assert r["ok"]
        assert (d / alr.CACHE_NAME).exists()
        assert (d / "alpha_live_report_20260717.json").exists()
        back = alr.load_alpha_live_report(str(d))
        assert back and back["n_days"] == r["n_days"]

    def test_line_reports_ic_and_stop_rate(self, tmp_path):
        d = _write_panel(tmp_path / "data")
        r = alr.compute_alpha_live_report(str(d))
        line = alr.alpha_live_line(r, h=3)
        assert "IC" in line and "양수일" in line and "스톱체결" in line

    def test_idempotent(self, tmp_path):
        """매일 재계산해도 같은 입력이면 같은 결과 — 누적 드리프트 금지."""
        d = _write_panel(tmp_path / "data")
        a = alr.compute_alpha_live_report(str(d))
        b = alr.compute_alpha_live_report(str(d))
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── 5. 배치 배선이 살아 있는가 ────────────────────────────────────
def test_pipeline_finalize_calls_the_report():
    """리포트를 만들어놓고 배치에서 부르지 않으면 표본이 쌓이지 않는다."""
    src = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")
    assert "save_alpha_live_report" in src, "pipeline_finalize가 리포트를 호출하지 않는다"
    assert "alpha_live_line" in src, "배치 로그에 한 줄 요약이 남지 않는다"


def test_real_repo_report_runs():
    """실제 data/로도 예외 없이 돌아야 한다 (없으면 ok=False로 정직하게)."""
    r = alr.compute_alpha_live_report(str(ROOT / "data"))
    assert isinstance(r, dict) and "ok" in r
    if r["ok"]:
        assert r["n_days"] >= 1 and "h3" in r["horizons"] or "h5" in r["horizons"]
