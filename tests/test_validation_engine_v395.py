# -*- coding: utf-8 -*-
"""v3.9.5 shadow 완화 시계열 추적 회귀 가드.

pytest tests/test_validation_engine_v395.py -v

- shadow 완화 플래그 × 날짜별 forward 성과 시계열 생성
- 누적+최근 윈도우 추세로 PROMOTION/REJECT/TREND_IMPROVING 판정
- v3.9.4(carry-stale)/v3.9.3 산출물 하위호환
- shadow 컬럼 없는 legacy → NO_SHADOW_DATA, 무크래시
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation_engine_v395 import (  # noqa: E402
    build_validation_engine_v395,
    build_shadow_timeseries,
    build_shadow_relaxation_summary,
    _promotion_verdict,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _rec_row(code, name, entry_relax, score_relax, macro_relax=0):
    return {
        "종목코드": code, "종목명": name, "ROUTE": "ARMED",
        "TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0, "BUY_NOW_PASS": 0,
        "DISPLAY_SCORE": 65.0,
        "SHADOW_ENTRY_RELAXED_ELIGIBLE": entry_relax,
        "SHADOW_SCORE_RELAXED_ELIGIBLE": score_relax,
        "SHADOW_MACRO_RELAXED_ELIGIBLE": macro_relax,
    }


def _trade_row(date, code, net_pct):
    return {"date": date, "code": code, "net_pct": net_pct, "tp1_hit": False, "stop_hit": False}


def _build_dataset(tmp_path: Path):
    """진입조건 완화가 누적 수익(+) — PROMOTION_CANDIDATE가 나오도록 구성."""
    data = tmp_path / "data"
    data.mkdir()
    out = tmp_path / "out"

    # 2일치, 진입완화 플래그가 매일 3개씩. forward 양호(+2~+4).
    _write_csv(data / "recommend_20260528.csv", [
        _rec_row("100001", "엠케이전자", 1, 1),
        _rec_row("100002", "디바이스", 1, 0),
        _rec_row("100003", "프로이천", 1, 0),
        _rec_row("900001", "비완화", 0, 0),
    ])
    _write_csv(data / "recommend_20260529.csv", [
        _rec_row("100004", "한온시스템", 1, 1),
        _rec_row("100005", "에이치브이엠", 1, 0),
        _rec_row("100006", "삼아알미늄", 1, 0),
    ])
    _write_csv(data / "backtest_top1_trades_20260528.csv", [
        _trade_row("2026-05-28", "100001", 4.0),
        _trade_row("2026-05-28", "100002", 2.5),
        _trade_row("2026-05-28", "100003", 3.0),
        _trade_row("2026-05-28", "900001", -1.0),
    ])
    _write_csv(data / "backtest_top1_trades_20260529.csv", [
        _trade_row("2026-05-29", "100004", 3.5),
        _trade_row("2026-05-29", "100005", 2.0),
        _trade_row("2026-05-29", "100006", 2.5),
    ])
    return data, out


def test_timeseries_built_per_flag_per_date(tmp_path: Path):
    data, out = _build_dataset(tmp_path)
    row_level, ts, relax, summary = build_validation_engine_v395(data, out, recent_days=5)

    assert not ts.empty
    assert (out / "shadow_relaxation_timeseries_latest.csv").exists()

    entry = ts[ts["SHADOW_FLAG"] == "SHADOW_ENTRY_RELAXED_ELIGIBLE"]
    # 2개 날짜
    assert set(entry["SIGNAL_DATE"]) == {"20260528", "20260529"}
    d28 = entry[entry["SIGNAL_DATE"] == "20260528"].iloc[0]
    assert int(d28["N"]) == 3
    assert d28["AVG_RET_%"] > 0


def test_entry_relaxation_promotion_candidate(tmp_path: Path):
    """진입조건 완화가 누적 +수익·승률 100% → PROMOTION_CANDIDATE."""
    data, out = _build_dataset(tmp_path)
    _row, _ts, relax, summary = build_validation_engine_v395(data, out, recent_days=5)

    entry = relax[relax["SHADOW_FLAG"] == "SHADOW_ENTRY_RELAXED_ELIGIBLE"].iloc[0]
    assert entry["CUM_RESULT_N"] == 6
    assert entry["CUM_AVG_RET_%"] > 2.0
    assert entry["PROMOTION_VERDICT"] == "PROMOTION_CANDIDATE"


def test_summary_version_and_headline(tmp_path: Path):
    data, out = _build_dataset(tmp_path)
    _row, _ts, _relax, summary = build_validation_engine_v395(data, out)

    assert summary["version"] == "v3.9.5"
    sr = summary["shadow_relaxation"]
    assert sr["status"] == "OK"
    assert "SHADOW_ENTRY_RELAXED_ELIGIBLE" in sr["promotion_candidates"]

    payload = json.loads((out / "validation_engine_v395_latest.json").read_text(encoding="utf-8"))
    assert payload["version"] == "v3.9.5"
    assert "shadow_relaxation" in payload


def test_backward_compat_v394_v393_outputs(tmp_path: Path):
    """v3.9.5는 v3.9.4/v3.9.3 산출물을 그대로 생성해야 한다."""
    data, out = _build_dataset(tmp_path)
    build_validation_engine_v395(data, out)
    assert (out / "validation_engine_v393_latest.csv").exists()
    assert (out / "validation_engine_v394_latest.json").exists()
    assert (out / "validation_engine_v395_latest.json").exists()


def test_legacy_without_shadow_columns_is_safe(tmp_path: Path):
    """shadow 컬럼 없는 legacy → NO_SHADOW_DATA, 크래시 없음."""
    data = tmp_path / "data"
    data.mkdir()
    out = tmp_path / "out"
    _write_csv(data / "recommend_20260519.csv", [
        {"종목코드": "900001", "종목명": "레거시", "ROUTE": "ATTACK",
         "TOP_PICK": 1, "BUY_NOW_ELIGIBLE": 1, "DISPLAY_SCORE": 80.0},
    ])
    _write_csv(data / "backtest_top1_trades_20260519.csv", [
        {"date": "2026-05-19", "code": "900001", "net_pct": 1.0},
    ])
    row_level, ts, relax, summary = build_validation_engine_v395(data, out)
    assert ts.empty
    assert relax.empty
    assert summary["shadow_relaxation"]["status"] == "NO_SHADOW_DATA"
    assert not (out / "shadow_relaxation_timeseries_latest.csv").exists()


def test_promotion_verdict_thresholds():
    assert _promotion_verdict(3, 5.0, 5.0, 100.0) == "NEED_MORE_N"        # 표본 부족
    assert _promotion_verdict(10, 2.5, 2.5, 60.0) == "PROMOTION_CANDIDATE"
    assert _promotion_verdict(10, -2.0, -2.0, 30.0) == "REJECT_RELAXATION"
    # 누적은 약하지만 최근이 +1%p 이상 개선 & 최근≥0 → 회복 조짐
    assert _promotion_verdict(10, 0.2, 1.5, 50.0) == "TREND_IMPROVING"
    assert _promotion_verdict(10, 0.5, 0.6, 50.0) == "KEEP_TRACKING"


def test_empty_rowdf_safe():
    assert build_shadow_timeseries(pd.DataFrame()).empty
    assert build_shadow_relaxation_summary(pd.DataFrame()).empty
