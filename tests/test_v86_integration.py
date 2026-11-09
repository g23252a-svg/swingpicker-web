# -*- coding: utf-8 -*-
"""[v86] 통합 리뷰 후속 — 두 트랙(timing·shadow)을 합친 뒤에만 드러난 문제.

  ① 판단센터 🔒 띠: 판정 사유(entry_block_status)는 RISK_OFF_ASOF 기준인데
     해제선 줄·진행률(_risk_off_state)은 kospi 최신 행 기준이라, 16:30 일봉 수집 후
     다음 배치 전까지 매일 저녁 한 띠 안에서 두 날짜를 말했다.
  ② 화면의 kospi_daily가 기록된 판정일보다 묵으면 수치 기준일을 덮어쓰고 표시 없이 보였다.
  ③ 수동 백필 end가 주말·휴장일이면 신선도 검사가 거짓 실패했다.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v86_timing import CRASH, _write_kospi, _write_reco  # noqa: E402
from services import entry_block_status as ebs  # noqa: E402
from components import decision_center as dc  # noqa: E402


def _blocked_df(n=4):
    return pd.DataFrame({
        "종목코드": [f"{i:06d}" for i in range(n)],
        "NEW_ENTRY_BLOCKED": [True] * n,
        "STOP_OVERRIDE_REASON": ["risk_off: 추천손절 유지 + 신규진입 차단"] * n,
    })


def test_strip_line_uses_verdict_day_not_newer_kospi(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    k = _write_kospi(d, CRASH)
    b = k["date"].iloc[-1]
    _write_reco(d, b, blocked=True, asof=b, stale=False)
    # 다음 날 16:30 수집: 강한 반등 행이 배치보다 먼저 들어온다
    nxt = (pd.Timestamp(b) + pd.offsets.BDay(1)).strftime("%Y%m%d")
    kk = pd.concat([k, pd.DataFrame({"date": [nxt], "close": [3400.0]})])
    kk.to_csv(d / "kospi_daily.csv", index=False, encoding="utf-8-sig")
    monkeypatch.chdir(tmp_path)

    ro = dc._risk_off_state(_blocked_df())
    assert ro["active"] and ro["progress"] is not None
    ro["market"] = ebs.compute_entry_block_status()
    assert ro["market"]["kospi_newer_than_batch"] is True
    dc._align_risk_off_line(ro)
    mkt = ro["market"]
    assert f"{mkt['close']:,.0f}" in ro["line"], "해제선 줄은 판정일 종가를 말한다"
    assert "3,400" not in ro["line"]
    exp = max(0.0, min(100.0, (mkt["deviation_pct"] + 15.0) / 12.0 * 100.0))
    assert abs(ro["progress"] - exp) < 1e-9


def test_strip_drops_numbers_when_asof_unknown():
    ro = {"active": True, "line": "코스피 9,999 · ...", "progress": 100.0,
          "market": {"ok": False, "asof_unknown": True, "risk_off": True}}
    dc._align_risk_off_line(ro)
    assert ro["progress"] is None and "9,999" not in ro["line"]


def test_strip_untouched_when_inactive_or_no_market():
    ro = {"active": False, "line": "", "progress": None, "market": None}
    dc._align_risk_off_line(ro)
    assert ro == {"active": False, "line": "", "progress": None, "market": None}


def test_screen_kospi_older_than_recorded_asof_is_flagged(tmp_path):
    k = _write_kospi(tmp_path, CRASH)
    b = k["date"].iloc[-1]
    # 배치는 b 다음 거래일 종가로 판정했지만 화면의 kospi_daily는 b까지만 있다
    rec_asof = (pd.Timestamp(b) + pd.offsets.BDay(1)).strftime("%Y%m%d")
    _write_reco(tmp_path, rec_asof, blocked=True, asof=rec_asof, stale=False)
    st = ebs.compute_entry_block_status(str(tmp_path))
    assert st["kospi_behind_record"] is True
    assert st["risk_off_asof"] == rec_asof, "기록된 판정일을 덮어쓰지 않는다"
    assert "미반영" in st["reason"]


def test_screen_kospi_current_not_flagged(tmp_path):
    k = _write_kospi(tmp_path, CRASH)
    b = k["date"].iloc[-1]
    _write_reco(tmp_path, b, blocked=True, asof=b, stale=False)
    st = ebs.compute_entry_block_status(str(tmp_path))
    assert st["kospi_behind_record"] is False and "미반영" not in st["reason"]


def _freshness_script():
    wf = yaml.safe_load((ROOT / ".github/workflows/collect_kospi_daily.yml").read_text(encoding="utf-8"))
    steps = next(iter(wf["jobs"].values()))["steps"]
    run = next(s for s in steps if s.get("id") == "fresh")["run"]
    body = run.split('python -c "', 1)[1].rsplit('"', 1)[0]
    import textwrap
    return textwrap.dedent(body)


def _run_fresh(tmp_path, monkeypatch, last, end_in, today):
    import datetime as dt
    (tmp_path / "data").mkdir(exist_ok=True)
    pd.DataFrame({"date": [last]}).to_csv(tmp_path / "data/kospi_daily.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("END_INPUT", end_in)

    class _DT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.strptime(today, "%Y%m%d")
    fake = type(sys)("datetime")
    fake.__dict__.update(dt.__dict__)
    fake.datetime = _DT
    monkeypatch.setitem(sys.modules, "datetime", fake)
    try:
        exec(_freshness_script(), {"__name__": "__main__"})
        return 0
    except SystemExit as e:
        return int(e.code or 0)


def test_freshness_backfill_end_on_weekend_passes(tmp_path, monkeypatch):
    # end=20261108(일) → 기대는 20261106(금). 금요일까지 받았으면 통과.
    assert _run_fresh(tmp_path, monkeypatch, "20261106", "20261108", "20261111") == 0


def test_freshness_backfill_end_still_checked(tmp_path, monkeypatch):
    assert _run_fresh(tmp_path, monkeypatch, "20261104", "20261106", "20261111") == 1


def test_freshness_nightly_lag_fails(tmp_path, monkeypatch):
    assert _run_fresh(tmp_path, monkeypatch, "20261106", "", "20261111") == 1
