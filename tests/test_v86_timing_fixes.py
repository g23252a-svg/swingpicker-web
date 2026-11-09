# -*- coding: utf-8 -*-
"""[v86] timing 리뷰 후속 — 확인된 지적 5건의 회귀 테스트.

  ① KOSPI 종가가 아예 없으면(파일 없음·읽기 실패·asof 이하 행 없음) stale로 드러낸다
  ② 옛 CSV 추정이 기록과 맞지 않으면(날짜 미상 판정) 모순되는 수치를 말하지 않는다
  ③ 수동 과거 백필(--end + --allow-older)은 '묵은 결과 거부'에 막히지 않는다
  ④ 화면은 recommend_latest.csv(배포 앱이 실제로 갱신하는 파일)를 우선 읽는다
  ⑤ collect_kospi_daily.yml rebase 충돌 시 커밋을 버리고 초록불로 끝나지 않는다
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import momentum_lane as ml  # noqa: E402
from services import entry_block_status as ebs  # noqa: E402

DATA = ROOT / "data"
CRASH = list(np.linspace(3000, 3300, 40)) + list(np.linspace(3300, 2600, 25))
UP = list(np.linspace(2600, 3400, 60))


def _kospi_df(closes, start="2026-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": dates.strftime("%Y%m%d"), "close": closes})


# ── ① 종가 없음 = stale ──

def test_missing_kospi_file_is_stale(tmp_path):
    ro, info = ml.compute_market_risk_off(str(tmp_path / "nope.csv"), asof="20261106")
    assert ro is False, "판정(레인 ON 디폴트)은 그대로"
    assert info["stale"] is True and info["kospi_last"] is None
    assert info["sessions_behind"] is None
    assert info["reason"].startswith("KOSPI_STALE")


def test_unreadable_kospi_is_stale(tmp_path):
    p = tmp_path / "kospi_daily.csv"
    p.write_text("foo,bar\n1,2\n", encoding="utf-8")   # date 컬럼 없음 → 예외 경로
    ro, info = ml.compute_market_risk_off(str(p), asof="20261106")
    assert ro is False and info["stale"] is True
    assert "KOSPI_STALE" in info["reason"]


def test_no_row_before_asof_is_stale():
    ro, info = ml.compute_market_risk_off(df_kospi=_kospi_df(UP), asof="20250101")
    assert ro is False and info["stale"] is True
    assert info["reason"].startswith("KOSPI_STALE(no data")


def test_no_asof_missing_file_keeps_old_behavior(tmp_path):
    _, info = ml.compute_market_risk_off(str(tmp_path / "nope.csv"))
    assert info["stale"] is False and not info["reason"].startswith("KOSPI_STALE")


def test_missing_kospi_reaches_csv_and_health(tmp_path):
    from run_health import check_run_health
    from stop_override import apply_stop_override
    ro, info = ml.compute_market_risk_off(str(tmp_path / "nope.csv"), asof="20261106")
    df = pd.DataFrame({"종목코드": ["000001", "000002"], "TOP_PICK": [0, 0]})
    out = apply_stop_override(df, market_risk_off=ro, risk_off_info=info)
    assert out["RISK_OFF_STALE"].all() and (out["RISK_OFF_ASOF"] == "").all()
    h = check_run_health(df, trade_ymd="20261106", risk_off_info=info)
    assert "KOSPI_STALE" in h.reasons


def test_health_kospi_stale_when_asof_without_close():
    # info가 예전 형태(stale=False)여도 asof는 있고 kospi_last가 없으면 STALE
    from run_health import check_run_health
    df = pd.DataFrame({"종목코드": ["000001"]})
    h = check_run_health(df, trade_ymd="20261106",
                         risk_off_info={"asof": "20261106", "kospi_last": None, "stale": False})
    assert "KOSPI_STALE" in h.reasons


# ── ② 날짜 미상 판정 ──

def _write_reco(d: Path, ymd: str, blocked: bool, asof=None, n=4, name=None, with_date=False):
    df = pd.DataFrame({
        "종목코드": [f"{i:06d}" for i in range(n)],
        "NEW_ENTRY_BLOCKED": [blocked] * n,
        "STOP_OVERRIDE_REASON": ["risk_off: 추천손절 유지 + 신규진입 차단" if blocked else ""] * n,
    })
    if asof is not None:
        df["RISK_OFF_ASOF"] = asof
    if with_date:
        df["기준일"] = ymd
    df.to_csv(d / (name or f"recommend_{ymd}.csv"), index=False, encoding="utf-8-sig")


def test_old_csv_unknown_asof_drops_contradicting_numbers(tmp_path):
    # 배치일·직전 세션 모두 '차단' 신호인데 기록은 '차단 아님' → 더 묵은 종가로 판정된 것
    k = _kospi_df(CRASH)
    k.to_csv(tmp_path / "kospi_daily.csv", index=False)
    b = k["date"].iloc[-1]
    _write_reco(tmp_path, b, blocked=False)
    st = ebs.compute_entry_block_status(str(tmp_path), batch_ymd=b)
    assert st["risk_off"] is False and st["asof_unknown"] is True
    assert st["stale"] is True, "모름/False가 아니라 지연으로"
    assert st["streak_days"] is None, "연속일을 0/1로 끼워 맞추지 않는다"
    assert st["ok"] is False, "소비자가 수치를 쓰지 않게"
    assert "20일선" not in st["reason"] and "날짜 미상" in st["reason"]


def test_recorded_asof_disagreement_is_not_unknown(tmp_path):
    k = _kospi_df(CRASH)
    k.to_csv(tmp_path / "kospi_daily.csv", index=False)
    b = k["date"].iloc[-1]
    _write_reco(tmp_path, b, blocked=False, asof=b)
    st = ebs.compute_entry_block_status(str(tmp_path), batch_ymd=b)
    assert st["asof_unknown"] is False and st["ok"] is True


@pytest.mark.skipif(not (DATA / "recommend_20261013.csv").exists()
                    or not (DATA / "kospi_daily.csv").exists(), reason="실데이터 없음")
def test_real_20261013_frozen_kospi_batch_has_no_contradiction():
    # 20261013 배치는 20260917에 멈춘 kospi로 판정 — 재계산(-11.3%·2일 연속)과 모순
    st = ebs.compute_entry_block_status(str(DATA), batch_ymd="20261013")
    if not st.get("recompute_disagrees"):
        pytest.skip("이 데이터에선 재계산이 기록과 일치")
    assert st["asof_unknown"] is True and st["streak_days"] is None and st["stale"] is True
    assert "%" not in st["reason"]


# ── ③ 수동 백필 ──

def _fake_fetch(dates):
    def f(start, end):
        return pd.DataFrame({"date": dates, "close": np.linspace(100, 110, len(dates))})
    return f


def test_kospi_allow_older_permits_backfill(tmp_path, monkeypatch):
    import collect_kospi_daily as ck
    out = tmp_path / "kospi_daily.csv"
    pd.DataFrame({"date": ["20261105", "20261106"], "close": [1, 2]}).to_csv(out, index=False)
    old = pd.bdate_range("2026-08-01", "2026-09-17").strftime("%Y%m%d").tolist()
    monkeypatch.setattr(ck, "_fetch_pykrx", _fake_fetch(old))
    assert ck.collect_kospi_daily("20260801", "20260917", str(out)) is False
    assert ck.collect_kospi_daily("20260801", "20260917", str(out), allow_older=True) is True
    assert pd.read_csv(out, dtype={"date": str})["date"].max() == "20260917"


def test_us_allow_older_permits_backfill(tmp_path, monkeypatch):
    import collect_us_daily as cu
    out = tmp_path / "us_daily.csv"
    pd.DataFrame({"date": ["20261105"], "ndx_close": [1.0]}).to_csv(out, index=False)
    old = pd.DataFrame({"date": ["20260915", "20260916"], "ndx_close": [1.0, 2.0]})
    monkeypatch.setattr(cu, "_fetch_symbol", lambda *a, **k: None)
    monkeypatch.setattr(cu, "merge_us_frames", lambda frames: old.copy())
    assert cu.collect_us_daily("20260801", "20260916", str(out)) is False
    assert cu.collect_us_daily("20260801", "20260916", str(out), allow_older=True) is True


def test_kospi_workflow_passes_allow_older_only_with_end():
    wf = (ROOT / ".github" / "workflows" / "collect_kospi_daily.yml").read_text(encoding="utf-8")
    assert wf.count('--end ${{ inputs.end }} --allow-older') == 2
    auto = (ROOT / ".github" / "workflows" / "auto_collect.yml").read_text(encoding="utf-8")
    assert "--allow-older" not in auto, "야간 배치는 묵은 결과로 덮지 않는다"


# ── ④ recommend_latest 우선 ──

def test_screen_prefers_newer_recommend_latest(tmp_path):
    k = _kospi_df(CRASH + UP)
    k.to_csv(tmp_path / "kospi_daily.csv", index=False)
    old_b = k["date"].iloc[len(CRASH) - 1]
    new_b = k["date"].iloc[-1]
    _write_reco(tmp_path, old_b, blocked=True, asof=old_b)          # 배포 때 실려 온 묵은 날짜 파일
    _write_reco(tmp_path, new_b, blocked=False, asof=new_b,
                name="recommend_latest.csv", with_date=True)        # 원격에서 갱신된 latest
    st = ebs.compute_entry_block_status(str(tmp_path))
    assert st["risk_off"] is False and st["batch_ymd"] == new_b
    assert st["source_file"] == "recommend_latest.csv"


def test_screen_uses_dated_file_when_latest_older(tmp_path):
    k = _kospi_df(CRASH)
    k.to_csv(tmp_path / "kospi_daily.csv", index=False)
    b_old, b_new = k["date"].iloc[-3], k["date"].iloc[-1]
    _write_reco(tmp_path, b_old, blocked=False, asof=b_old,
                name="recommend_latest.csv", with_date=True)
    _write_reco(tmp_path, b_new, blocked=True, asof=b_new)
    st = ebs.compute_entry_block_status(str(tmp_path))
    assert st["risk_off"] is True and st["batch_ymd"] == b_new


def test_explicit_batch_ymd_ignores_latest(tmp_path):
    k = _kospi_df(CRASH + UP)
    k.to_csv(tmp_path / "kospi_daily.csv", index=False)
    old_b, new_b = k["date"].iloc[len(CRASH) - 1], k["date"].iloc[-1]
    _write_reco(tmp_path, old_b, blocked=True, asof=old_b)
    _write_reco(tmp_path, new_b, blocked=False, asof=new_b,
                name="recommend_latest.csv", with_date=True)
    st = ebs.compute_entry_block_status(str(tmp_path), batch_ymd=old_b)
    assert st["risk_off"] is True and st["batch_ymd"] == old_b


# ── ⑤ rebase 충돌 재시도 ──

def test_kospi_workflow_rebase_conflict_not_silent():
    wf = (ROOT / ".github" / "workflows" / "collect_kospi_daily.yml").read_text(encoding="utf-8")
    assert 'echo "::warning::rebase 실패 — 푸시 생략"; exit 0' not in wf
    assert "git reset --hard origin/main" in wf
    assert "scripts/daily_csv_newer.py" in wf
    assert "exit 1; }" in wf, "푸시 실패는 빨간불"


def test_daily_csv_newer(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"date": ["20261105", "20261106"]}).to_csv(a, index=False)
    pd.DataFrame({"date": ["20261105"]}).to_csv(b, index=False)
    script = str(ROOT / "scripts" / "daily_csv_newer.py")
    run = lambda x, y: subprocess.run([sys.executable, script, str(x), str(y)],
                                      capture_output=True).returncode
    assert run(a, b) == 0
    assert run(b, a) == 1, "원격이 더 새로우면 덮지 않는다"
    assert run(a, tmp_path / "none.csv") == 0
