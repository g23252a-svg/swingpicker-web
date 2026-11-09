# -*- coding: utf-8 -*-
"""[v86] 시점 정합 — risk_off는 배치 당일 종가로, 파일은 거래일로, 수급 캐시는 같은 날짜만.

■ v86 실측(git 이력 재현)
  · kospi_daily.csv는 배치 커밋 12~40분 뒤에 갱신돼 9/21 이후 33/33 배치가 하루 묵은
    종가로 risk_off를 판정했다. 기록된 NEW_ENTRY_BLOCKED가 배치일 기준 판정과 다른 날:
    10/13(파일이 20260917에 멈춤) · 10/28 · 10/30 · 11/02 · 11/03.
  · 화면(entry_block_status)은 kospi 최신 행으로 다시 계산해 flip일마다 CSV와 어긋났다.
  · 자정 넘긴 배치가 backtest_*_<시계날짜>를 써 _20261102/_20261104 세트가 사라졌고,
    flow_20261102.json도 같은 이유로 없었다(11/02 배치는 다른 날짜 latest 캐시로 FLOW=OK).
  판정 규칙·임계값은 바꾸지 않는다(지연이어도 막지 않음 — 정책 변경은 백테스트 필요).
"""
import json
import os
import subprocess
import sys
from datetime import datetime
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


def _kospi_df(closes, start="2026-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": dates.strftime("%Y%m%d"), "close": closes})


CRASH = list(np.linspace(3000, 3300, 40)) + list(np.linspace(3300, 2600, 25))
UP = list(np.linspace(2600, 3400, 60))


# ── ① compute_market_risk_off 신선도 정보 (판정은 그대로) ──

def test_fresh_when_last_equals_asof():
    k = _kospi_df(CRASH)
    last = k["date"].iloc[-1]
    ro, info = ml.compute_market_risk_off(df_kospi=k, asof=last)
    assert ro is True
    assert info["kospi_last"] == last and info["stale"] is False
    assert not info["reason"].startswith("KOSPI_STALE")


def test_stale_flag_and_reason_but_decision_unchanged():
    k = _kospi_df(CRASH)
    last = pd.Timestamp(k["date"].iloc[-1])
    later = (last + pd.offsets.BDay(2)).strftime("%Y%m%d")
    ro_fresh, _ = ml.compute_market_risk_off(df_kospi=k, asof=k["date"].iloc[-1])
    ro_stale, info = ml.compute_market_risk_off(df_kospi=k, asof=later)
    assert ro_stale == ro_fresh, "지연이어도 판정을 바꾸지 않는다(fail-closed 아님)"
    assert info["stale"] is True and info["sessions_behind"] == 2
    assert info["reason"].startswith("KOSPI_STALE")
    assert info["kospi_last"] == k["date"].iloc[-1]


def test_weekend_asof_is_not_stale():
    assert ml._sessions_between("20261106", "20261107") == 0   # 금 → 토
    assert ml._sessions_between("20261106", "20261108") == 0   # 금 → 일
    assert ml._sessions_between("20261106", "20261109") == 1   # 금 → 월
    assert ml._sessions_between("20261106", "20261106") == 0


def test_asof_uses_only_rows_up_to_asof():
    k = _kospi_df(UP + list(np.linspace(3400, 2700, 20)))
    mid = k["date"].iloc[59]
    ro, info = ml.compute_market_risk_off(df_kospi=k, asof=mid)
    assert ro is False and info["kospi_last"] == mid and info["stale"] is False


# ── ② stop_override가 판정 시점을 행에 남긴다 ──

def test_stop_override_records_risk_off_asof():
    from stop_override import apply_stop_override, RISK_OFF_STATE_COLS
    df = pd.DataFrame({"추천매수가": [1000, 2000], "TOP_PICK": [1, 0]})
    out = apply_stop_override(df, market_risk_off=True,
                              risk_off_info={"kospi_last": "20261102", "stale": True})
    for c in RISK_OFF_STATE_COLS:
        assert c in out.columns
    assert set(out["RISK_OFF_ASOF"]) == {"20261102"}
    assert out["RISK_OFF_STALE"].all()
    assert out["NEW_ENTRY_BLOCKED"].all()


def test_stop_override_without_info_keeps_old_schema():
    from stop_override import apply_stop_override
    out = apply_stop_override(pd.DataFrame({"추천매수가": [1000]}), market_risk_off=False)
    assert "RISK_OFF_ASOF" not in out.columns


def test_pipeline_calibrate_passes_trade_ymd():
    src = (ROOT / "pipeline_calibrate.py").read_text(encoding="utf-8")
    assert src.count("asof=ctx.trade_ymd") >= 2, "두 compute_market_risk_off 호출 모두 asof 필요"
    assert "risk_off_info=_ro2_info" in src
    assert "ctx.risk_off_info" in src


# ── ③ run_health: KOSPI_STALE · 수급 캐시 날짜 불일치 ──

def _health_df():
    return pd.DataFrame({"NEWS_SCORE": [1.0], "SECTOR_RANK": [1], "외인순매수": [1]})


def test_run_health_reports_kospi_stale():
    from run_health import check_run_health
    h = check_run_health(_health_df(), mcap_map={"a": 1}, bench_map={"a": 1},
                         inv_maps={"frg": {"a": 1}, "inst": {}, "ant": {}},
                         risk_off_info={"stale": True, "kospi_last": "20261102",
                                        "asof": "20261103", "reason": "KOSPI_STALE"})
    assert "KOSPI_STALE" in h.reasons and h.status in ("DEGRADED", "CRITICAL")


def test_run_health_fresh_kospi_adds_no_reason():
    from run_health import check_run_health
    h = check_run_health(_health_df(), mcap_map={"a": 1}, bench_map={"a": 1},
                         inv_maps={"frg": {"a": 1}, "inst": {}, "ant": {}},
                         risk_off_info={"stale": False, "kospi_last": "20261103"})
    assert "KOSPI_STALE" not in h.reasons and h.checks.get("KOSPI_FRESH") is True


def test_run_health_flow_not_ok_when_cache_date_mismatched():
    from run_health import check_run_health
    meta = {"ymd": "20261102", "source": "kis_cache", "cache_ymd": "20261103",
            "cache_ymd_mismatch": True}
    h = check_run_health(_health_df(), mcap_map={"a": 1}, bench_map={"a": 1},
                         inv_maps={"frg": {"a": 1}, "inst": {}, "ant": {}}, flow_meta=meta)
    assert h.axis_status["FLOW"] != "OK"
    assert h.checks.get("FLOW") is False
    assert any(w.startswith("FLOW_CACHE_STALE") for w in h.warnings)


def test_run_health_flow_ok_when_same_day_pykrx_rescued():
    from run_health import check_run_health
    meta = {"ymd": "20261102", "source": "pykrx", "cache_ymd": "20261103",
            "cache_ymd_mismatch": True}
    h = check_run_health(_health_df(), mcap_map={"a": 1}, bench_map={"a": 1},
                         inv_maps={"frg": {"a": 1}, "inst": {}, "ant": {}}, flow_meta=meta)
    assert h.axis_status["FLOW"] == "OK"


def test_pipeline_finalize_wires_health_meta():
    src = (ROOT / "pipeline_finalize.py").read_text(encoding="utf-8")
    assert 'risk_off_info=getattr(ctx, "risk_off_info", None)' in src
    assert 'flow_meta=getattr(ctx, "flow_meta", None)' in src
    assert "compute_entry_block_status(OUT_DIR, df=df_out, batch_ymd=trade_ymd)" in src


# ── ④ investor_flow: latest 캐시는 같은 날짜일 때만 ──

def _write_flow(d: Path, name: str, ymd: str):
    (d / "data").mkdir(exist_ok=True)
    (d / "data" / name).write_text(json.dumps(
        {"ymd": ymd, "frg": {"005930": 10}, "inst": {"005930": 5}, "ant": {}}), encoding="utf-8")


def test_flow_latest_with_other_date_is_not_used(tmp_path, monkeypatch):
    import investor_flow as inf
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(inf, "PYKRX_OK", False)
    _write_flow(tmp_path, "flow_cache_latest.json", "20261103")
    frg, inst, ant = inf.fetch_investor_net_buying("20261102")
    assert frg == {} and inst == {}
    meta = inf.last_fetch_meta()
    assert meta["cache_ymd_mismatch"] is True and meta["cache_ymd"] == "20261103"
    assert meta["source"] == "none"


def test_flow_latest_with_same_date_is_used(tmp_path, monkeypatch):
    import investor_flow as inf
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(inf, "PYKRX_OK", False)
    _write_flow(tmp_path, "flow_cache_latest.json", "20261102")
    frg, _, _ = inf.fetch_investor_net_buying("20261102")
    assert frg == {"005930": 10}
    assert inf.last_fetch_meta()["source"] == "kis_cache_latest"


def test_flow_dated_file_preferred(tmp_path, monkeypatch):
    import investor_flow as inf
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(inf, "PYKRX_OK", False)
    _write_flow(tmp_path, "flow_20261102.json", "20261102")
    _write_flow(tmp_path, "flow_cache_latest.json", "20261103")
    frg, _, _ = inf.fetch_investor_net_buying("20261102")
    assert frg == {"005930": 10}
    m = inf.last_fetch_meta()
    assert m["source"] == "kis_cache" and m["cache_ymd_mismatch"] is False


# ── ⑤ entry_block_status: 기록된 판정을 읽는다 ──

def _write_reco(d: Path, ymd: str, blocked: bool, asof=None, stale=None, n=4):
    df = pd.DataFrame({
        "종목코드": [f"{i:06d}" for i in range(n)],
        "NEW_ENTRY_BLOCKED": [blocked] * n,
        "STOP_OVERRIDE_REASON": ["risk_off: 추천손절 유지 + 신규진입 차단" if blocked else ""] * n,
    })
    if asof is not None:
        df["RISK_OFF_ASOF"] = asof
    if stale is not None:
        df["RISK_OFF_STALE"] = stale
    df.to_csv(d / f"recommend_{ymd}.csv", index=False, encoding="utf-8-sig")


def _write_kospi(d: Path, closes, start="2026-01-01"):
    k = _kospi_df(closes, start)
    k.to_csv(d / "kospi_daily.csv", index=False, encoding="utf-8-sig")
    return k


def test_screen_follows_recorded_state_not_latest_kospi(tmp_path):
    # kospi 최신 행은 폭락(차단) — 그러나 배치는 '차단 아님'으로 기록했다
    k = _write_kospi(tmp_path, CRASH)
    b = k["date"].iloc[-1]
    _write_reco(tmp_path, b, blocked=False, asof=k["date"].iloc[-2], stale=True)
    st = ebs.compute_entry_block_status(str(tmp_path))
    assert st["ok"] and st["risk_off"] is False
    assert st["source"] == "batch_csv"
    assert st["risk_off_asof"] == k["date"].iloc[-2], "수치는 판정에 쓴 날짜 기준"
    assert st["stale"] is True
    assert "판정 기준 코스피" in st["reason"]


def test_screen_blocked_when_recorded_blocked(tmp_path):
    k = _write_kospi(tmp_path, CRASH)
    b = k["date"].iloc[-1]
    _write_reco(tmp_path, b, blocked=True, asof=b, stale=False)
    st = ebs.compute_entry_block_status(str(tmp_path))
    assert st["risk_off"] is True and st["unlock_gap_pct"] > 0 and st["stale"] is False


def test_old_csv_without_asof_infers_previous_session(tmp_path):
    # 마지막 날 신호 ≠ 기록, 전날 신호 = 기록 → 옛 배치의 1일 지연 판정으로 추정
    closes = CRASH[:-1] + [3400.0]          # 마지막 날 급반등 → 배치일 신호는 '차단 아님'
    k = _write_kospi(tmp_path, closes)
    b = k["date"].iloc[-1]
    _write_reco(tmp_path, b, blocked=True)  # 옛 배치는 전날(차단) 종가로 판정
    st = ebs.compute_entry_block_status(str(tmp_path))
    assert st["risk_off"] is True and st["asof_inferred"] is True
    assert st["risk_off_asof"] == k["date"].iloc[-2]


def test_csv_without_state_cols_recomputes_asof_batch(tmp_path):
    k = _write_kospi(tmp_path, CRASH + list(np.linspace(2600, 3600, 10)))
    b = k["date"].iloc[len(CRASH) - 1]      # 폭락 끝 날짜의 배치, kospi는 그 뒤로 회복
    pd.DataFrame({"종목코드": ["000001"]}).to_csv(tmp_path / f"recommend_{b}.csv", index=False)
    st = ebs.compute_entry_block_status(str(tmp_path))
    assert st["source"] == "recompute_batch"
    assert st["risk_off"] is True, "최신 kospi(회복)가 아니라 배치일 기준"
    assert st["kospi_newer_than_batch"] is True


def test_df_argument_reads_in_batch_state(tmp_path):
    k = _write_kospi(tmp_path, CRASH)
    df = pd.DataFrame({"NEW_ENTRY_BLOCKED": [False] * 3, "STOP_OVERRIDE_REASON": [""] * 3,
                       "RISK_OFF_ASOF": [k["date"].iloc[-1]] * 3})
    st = ebs.compute_entry_block_status(str(tmp_path), df=df, batch_ymd=k["date"].iloc[-1])
    assert st["risk_off"] is False and st["source"] == "batch_csv"
    assert st["recompute_disagrees"] is True


def test_no_batch_csv_keeps_legacy_latest_recompute(tmp_path):
    _write_kospi(tmp_path, CRASH)
    st = ebs.compute_entry_block_status(str(tmp_path))
    assert st["source"] == "recompute_latest" and st["risk_off"] is True


# ── ⑥ 파일 날짜 = 거래일 ──

def test_batch_date_label_prefers_env(tmp_path, monkeypatch):
    import backtest_validation as bv
    (tmp_path / "recommend_20261103.csv").write_text("a\n1\n")
    monkeypatch.setenv("BATCH_TARGET", "20261102")
    assert bv.batch_date_label(tmp_path) == "20261102"


def test_batch_date_label_falls_back_to_newest_recommend(tmp_path, monkeypatch):
    import backtest_validation as bv
    monkeypatch.delenv("BATCH_TARGET", raising=False)
    for y in ("20261030", "20261102"):
        (tmp_path / f"recommend_{y}.csv").write_text("a\n1\n")
    (tmp_path / "recommend_latest.csv").write_text("a\n1\n")
    (tmp_path / "recommend_20261103_tag.csv").write_text("a\n1\n")
    assert bv.batch_date_label(tmp_path) == "20261102"


def test_save_report_default_date_is_batch_day(tmp_path, monkeypatch):
    import backtest_validation as bv
    monkeypatch.delenv("BATCH_TARGET", raising=False)
    (tmp_path / "recommend_20261102.csv").write_text("a\n1\n")
    # 최소 리포트 — v85 계약 테스트의 헬퍼를 재사용
    from tests.test_v85_backtest_contract import _mini_report
    bv.save_report(_mini_report([]), tmp_path)
    assert (tmp_path / "backtest_validation_20261102.json").exists()
    assert not (tmp_path / f"backtest_validation_{datetime.now():%Y%m%d}.json").exists() \
        or datetime.now().strftime("%Y%m%d") == "20261102"


@pytest.mark.parametrize("ts,win,ymd", [
    ("2026-11-02 21:00", "evening", "20261102"),
    ("2026-11-03 01:27", "overnight", "20261102"),   # 11/02 prefetch의 실제 발화 시각
    ("2026-11-09 03:00", "overnight", "20261106"),   # 월요일 새벽 → 금요일
    ("2026-11-03 15:40", "postclose", "20261103"),   # cron 35 6 이 제때 돈 경우
    ("2026-11-03 11:00", "market", None),
    ("2026-11-07 20:00", "weekend", None),
])
def test_prefetch_batch_target_windows(ts, win, ymd):
    import prefetch_flow as pf
    assert pf.batch_target(datetime.strptime(ts, "%Y-%m-%d %H:%M")) == (win, ymd)


def test_prefetch_matches_batch_target_sh_now():
    import prefetch_flow as pf
    try:
        r = subprocess.run(["bash", str(ROOT / "scripts" / "batch_target.sh")],
                           capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as e:
        pytest.skip(f"bash 실행 불가: {e}")
    kv = dict(line.split("=", 1) for line in r.stdout.strip().splitlines())
    from zoneinfo import ZoneInfo
    win, ymd = pf.batch_target(datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None))
    if win == "postclose":
        assert kv["BATCH_WINDOW"] == "market"
    else:
        assert win == kv["BATCH_WINDOW"] and (ymd or "") == kv["BATCH_TARGET"]


# ── ⑦ 수집기: end 이후 행 제거 · 묵은 결과로 덮어쓰지 않음 ──

def _fake_fetch(dates):
    def f(start, end):
        return pd.DataFrame({"date": dates, "close": np.linspace(100, 110, len(dates))})
    return f


def test_kospi_collector_trims_after_end(tmp_path, monkeypatch):
    import collect_kospi_daily as ck
    dates = pd.bdate_range("2026-10-01", "2026-11-06").strftime("%Y%m%d").tolist()
    monkeypatch.setattr(ck, "_fetch_pykrx", _fake_fetch(dates))
    out = tmp_path / "kospi_daily.csv"
    assert ck.collect_kospi_daily("20261001", "20261103", str(out))
    assert pd.read_csv(out, dtype={"date": str})["date"].max() == "20261103"


def test_kospi_collector_refuses_older_result(tmp_path, monkeypatch):
    import collect_kospi_daily as ck
    out = tmp_path / "kospi_daily.csv"
    pd.DataFrame({"date": ["20261105", "20261106"], "close": [1, 2]}).to_csv(out, index=False)
    old = pd.bdate_range("2026-08-01", "2026-09-17").strftime("%Y%m%d").tolist()
    monkeypatch.setattr(ck, "_fetch_pykrx", _fake_fetch(old))
    assert ck.collect_kospi_daily("20260801", "20261106", str(out)) is False
    assert pd.read_csv(out, dtype={"date": str})["date"].max() == "20261106", "기존 파일 유지"


# ── ⑧ 워크플로 배선 ──

def _wf(name):
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def test_auto_collect_refreshes_kospi_before_collector():
    s = _wf("auto_collect.yml")
    assert 'echo "BATCH_TARGET=${BATCH_TARGET}" >> "$GITHUB_ENV"' in s
    i_k = s.index("scripts/collect_kospi_daily.py --end")
    i_c = s.index("python collector.py")
    i_dep = s.index("pip install -r requirements.txt")
    assert i_dep < i_k < i_c
    assert "scripts/collect_us_daily.py --end" in s
    assert "data/kospi_daily.csv" in s[s.index("Commit & Push CSV"):]
    assert 'chore: add CSV ${BATCH_TARGET:-manual} (written' in s


def test_kospi_workflow_freshness_and_data_dated_message():
    s = _wf("collect_kospi_daily.yml")
    assert "신선도 검사" in s and "holiday_skip_" in s
    assert s.index("필수 컬럼 누락") < s.index("신선도 검사 (v86)")
    assert "$(date +%Y-%m-%d))" not in s
    assert "data ${DATA_DATE}" in s


def test_prefetch_workflow_uses_batch_target():
    s = _wf("prefetch_flow.yml")
    seg = s[s.index("Prefetch investor flow"):s.index("Show cache result")]
    assert "scripts/batch_target.sh" in seg and "export BATCH_TARGET" in seg
    assert "prefetch flow cache ${FLOW_YMD:-unknown} (written" in s


# ── ⑨ 실데이터 재현 (데이터 없으면 skip) ──

def _recorded(path):
    df = pd.read_csv(path, usecols=lambda c: c in ("NEW_ENTRY_BLOCKED", "STOP_OVERRIDE_REASON"),
                     low_memory=False)
    return ebs.recorded_block_state(df)


@pytest.mark.skipif(not (DATA / "kospi_daily.csv").exists(), reason="실데이터 없음")
def test_real_recorded_flags_vs_same_day_rule():
    """9/21~11/06 배치: 배치일 종가 규칙과 기록이 다른 날은 정확히 5일(연구 재현)."""
    k = pd.read_csv(DATA / "kospi_daily.csv")
    import re
    days = sorted(m.group(1) for m in (re.fullmatch(r"recommend_(\d{8})\.csv", p.name)
                                       for p in DATA.glob("recommend_[0-9]*.csv"))
                  if m and "20260921" <= m.group(1) <= "20261106")
    if len(days) < 25:
        pytest.skip("배치 CSV 부족")
    mism = []
    for b in days:
        rec = _recorded(DATA / f"recommend_{b}.csv")
        if rec is None:
            continue
        ro, _ = ml.compute_market_risk_off(df_kospi=k, asof=b)
        if ro != rec["risk_off"]:
            mism.append(b)
    assert mism == ["20261013", "20261028", "20261030", "20261102", "20261103"]


@pytest.mark.skipif(not (DATA / "recommend_20261030.csv").exists(), reason="실데이터 없음")
@pytest.mark.parametrize("b,blocked,asof", [
    ("20261028", True, "20261027"),
    ("20261030", False, "20261029"),
    ("20261102", True, "20261030"),
    ("20261103", False, "20261102"),
])
def test_real_screen_agrees_with_recorded_flip_days(b, blocked, asof):
    st = ebs.compute_entry_block_status(str(DATA), batch_ymd=b)
    assert st["risk_off"] is blocked
    assert st["risk_off_asof"] == asof and st["asof_inferred"] is True and st["stale"] is True


@pytest.mark.skipif(not (DATA / "kospi_daily.csv").exists(), reason="실데이터 없음")
def test_real_stale_freeze_would_be_flagged():
    """9/18~10/13 동결(20260917) 재현 — 10/13 배치에 18영업일 지연으로 표시된다."""
    k = pd.read_csv(DATA / "kospi_daily.csv")
    frozen = k[k["date"].astype(str) <= "20260917"]
    ro, info = ml.compute_market_risk_off(df_kospi=frozen, asof="20261013")
    assert info["stale"] is True and info["kospi_last"] == "20260917"
    assert info["sessions_behind"] >= 15
    ro_true, _ = ml.compute_market_risk_off(df_kospi=k, asof="20261013")
    assert ro is False and ro_true is True, "동결 데이터는 10/13 차단을 놓쳤다"
