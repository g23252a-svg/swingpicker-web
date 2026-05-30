# -*- coding: utf-8 -*-
"""v3.9.4 carry-stale 검증 회귀 가드.

pytest tests/test_validation_engine_v394.py -v

- DEAD/EXIT_SIGNAL=1 carry가 forward 부진이면 EXIT_SIGNAL_VALIDATED
- STAGE/EXIT_SIGNAL 분해표 + summary verdict 생성
- v3.9.3 산출물(no_buy/shadow/CSV) 하위호환 유지
- carry 컬럼 없는 legacy CSV → NO_CARRY_STALE_DATA (무크래시)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation_engine_v394 import (  # noqa: E402
    build_validation_engine_v394,
    build_carry_stale_validation,
    _carry_exit_grade,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _carry_row(code, name, stage, sig, age, ret):
    return {
        "종목코드": code, "종목명": name, "ROUTE": "CARRY",
        "TOP_PICK": 0, "BUY_NOW_ELIGIBLE": 0, "BUY_NOW_PASS": 0,
        "DISPLAY_SCORE": 70.0,
        "CARRY_STALE_STAGE": stage, "CARRY_EXIT_SIGNAL": sig,
        "CARRY_AGE_DAYS": age, "CARRY_RET_PCT": ret,
    }


def _trade_row(code, net_pct, stop_hit=False):
    return {"date": "2026-05-20", "code": code, "net_pct": net_pct,
            "tp1_hit": False, "stop_hit": stop_hit}


def _build_dataset(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    out = tmp_path / "out"

    # DEAD/EXIT_SIGNAL=1 carry 6건 (forward 부진), FRESH 2건(양호), STALE 1건
    recs = [
        _carry_row("100001", "신세계I&C", "DEAD", 1, 10, -9.0),
        _carry_row("100002", "한컴위드", "DEAD", 1, 12, -7.0),
        _carry_row("100003", "한텍", "DEAD", 1, 11, -8.0),
        _carry_row("100004", "한양디지텍", "DEAD", 1, 14, -6.0),
        _carry_row("100005", "에이피알", "DEAD", 1, 10, -5.0),
        _carry_row("100006", "에스엔시스", "DEAD", 1, 13, -4.0),
        _carry_row("200001", "웨이비스", "FRESH", 0, 2, 2.0),
        _carry_row("200002", "대우건설", "FRESH", 0, 1, 1.5),
        _carry_row("300001", "씨앤시", "STALE", 0, 8, -1.0),
    ]
    _write_csv(data / "recommend_20260520.csv", recs)

    # 실현 결과: DEAD는 계속 하락(-2~-6), FRESH는 상승(+2~+3)
    trades = [
        _trade_row("100001", -6.0, stop_hit=True),
        _trade_row("100002", -3.0),
        _trade_row("100003", -5.0, stop_hit=True),
        _trade_row("100004", -2.0),
        _trade_row("100005", -4.0),
        _trade_row("100006", -3.5),
        _trade_row("200001", 3.0),
        _trade_row("200002", 2.0),
        _trade_row("300001", -1.0),
    ]
    _write_csv(data / "backtest_top1_trades_20260520.csv", trades)
    return data, out


def test_dead_exit_signal_validated_when_forward_is_bad(tmp_path: Path):
    data, out = _build_dataset(tmp_path)
    row_level, no_buy, shadow, carry, summary = build_validation_engine_v394(data, out)

    # carry-stale 표 생성 + CSV
    assert not carry.empty
    assert (out / "carry_stale_validation_latest.csv").exists()

    # STAGE 분해: DEAD 6, FRESH 2, STALE 1
    stage = carry[carry["GROUP_KIND"] == "STAGE"].set_index("GROUP_VALUE")
    assert int(stage.loc["DEAD", "N"]) == 6
    assert int(stage.loc["FRESH", "N"]) == 2
    assert int(stage.loc["STALE", "N"]) == 1

    # EXIT_SIGNAL=1 그룹: forward 부진 → 가드 검증됨
    sig = carry[carry["GROUP_KIND"] == "EXIT_SIGNAL"].set_index("GROUP_VALUE")
    assert sig.loc["EXIT_SIGNAL=1", "GRADE_OR_HINT"] == "EXIT_SIGNAL_VALIDATED"
    assert sig.loc["EXIT_SIGNAL=1", "AVG_RET_%"] < 0


def test_summary_has_v394_version_and_verdict(tmp_path: Path):
    data, out = _build_dataset(tmp_path)
    _row, _nb, _sh, _carry, summary = build_validation_engine_v394(data, out)

    assert summary["version"] == "v3.9.4"
    verdict = summary["carry_stale_validation"]
    assert verdict["status"] == "OK"
    assert verdict["exit_signal_grade"] == "EXIT_SIGNAL_VALIDATED"
    # DEAD가 FRESH보다 forward가 나쁨 → 음수
    assert verdict["dead_minus_fresh_avg_ret_%"] < 0

    # JSON 파일도 기록
    payload = json.loads((out / "validation_engine_v394_latest.json").read_text(encoding="utf-8"))
    assert payload["version"] == "v3.9.4"
    assert "carry_stale_validation" in payload


def test_backward_compat_v393_outputs_still_produced(tmp_path: Path):
    """v3.9.4는 v3.9.3 산출물(no_buy/shadow/row CSV)을 그대로 생성해야 한다."""
    data, out = _build_dataset(tmp_path)
    row_level, no_buy, shadow, carry, summary = build_validation_engine_v394(data, out)

    assert isinstance(no_buy, pd.DataFrame)
    assert isinstance(shadow, pd.DataFrame)
    assert (out / "validation_engine_v393_latest.csv").exists()
    # v393 노트가 v394 노트에 누적
    assert any("v3.9.4" in n for n in summary["notes"])


def test_legacy_csv_without_carry_columns_is_safe(tmp_path: Path):
    """carry-stale 컬럼이 없는 legacy 데이터 → NO_CARRY_STALE_DATA, 크래시 없음."""
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

    row_level, no_buy, shadow, carry, summary = build_validation_engine_v394(data, out)
    assert carry.empty
    assert summary["carry_stale_validation"]["status"] == "NO_CARRY_STALE_DATA"
    # carry CSV는 비어있으므로 기록하지 않음
    assert not (out / "carry_stale_validation_latest.csv").exists()


def test_carry_exit_grade_thresholds():
    assert _carry_exit_grade(3, -5.0) == "NEED_MORE_N"       # 표본 부족
    assert _carry_exit_grade(10, -2.0) == "EXIT_SIGNAL_VALIDATED"
    assert _carry_exit_grade(10, 3.0) == "EXIT_TOO_AGGRESSIVE_WARNING"
    assert _carry_exit_grade(10, 0.0) == "INCONCLUSIVE"


def test_empty_rowdf_returns_empty():
    assert build_carry_stale_validation(pd.DataFrame()).empty
