# -*- coding: utf-8 -*-
import json
import pandas as pd

from kelly_calibrator import compute_est_win_rate


def test_est_win_rate_is_capped_by_realized_elite_bin(tmp_path):
    """EST_WIN_RATE는 실현 ELITE bin 대비 15%p 초과 과신 표시를 하지 않는다."""
    # calibration_table 쪽은 일부러 높게 만들어 원래라면 0.70 근처가 나오게 한다.
    cal = [
        {
            "method": "ELITE_SCORE",
            "horizon": 5,
            "score_lo": 70,
            "score_hi": 80,
            "score_center": 75,
            "p_calibrated": 0.70,
            "n_effective": 100,
            "n_raw": 100,
        },
        {
            "method": "ELITE_SCORE",
            "horizon": 5,
            "score_lo": 80,
            "score_hi": 90,
            "score_center": 85,
            "p_calibrated": 0.72,
            "n_effective": 100,
            "n_raw": 100,
        },
    ]
    (tmp_path / "calibration_table.json").write_text(
        json.dumps(cal, ensure_ascii=False),
        encoding="utf-8",
    )

    # monotonicity 기준 실현 승률 테이블.
    # 70~80 bin p_win=0.36이면 표시 승률은 0.36+0.145=0.505 이하로 캡.
    wt = {
        "meta": {"is_sufficient": True},
        "table": [
            {
                "score_lo": 70,
                "score_hi": 80,
                "p_win": 0.36,
                "n_raw": 120,
                "sufficient": True,
                "avg_ret_net_pct": 0.0,
                "avg_ret_excess_pct": 0.0,
            }
        ],
    }
    (tmp_path / "winrate_table_by_ELITE_SCORE_latest.json").write_text(
        json.dumps(wt, ensure_ascii=False),
        encoding="utf-8",
    )

    df = pd.DataFrame({"ELITE_SCORE": [75.0]})
    out = compute_est_win_rate(df, str(tmp_path), asof_ymd="20260525")

    assert out.loc[0, "EST_WIN_RATE"] <= 0.505
    assert bool(out.loc[0, "EST_WIN_RATE_REALIZED_CAP"]) is True


def test_est_win_rate_cap_does_not_require_buy_now_fields(tmp_path):
    """캡핑은 승률 표시용이며 BUY_NOW/TOP_PICK 컬럼 없이도 동작한다."""
    wt = {
        "meta": {"is_sufficient": True},
        "table": [
            {
                "score_lo": 70,
                "score_hi": 80,
                "p_win": 0.40,
                "n_raw": 100,
                "sufficient": True,
            }
        ],
    }
    (tmp_path / "winrate_table_by_ELITE_SCORE_latest.json").write_text(
        json.dumps(wt, ensure_ascii=False),
        encoding="utf-8",
    )

    df = pd.DataFrame({"ELITE_SCORE": [75.0]})
    out = compute_est_win_rate(df, str(tmp_path), asof_ymd="20260525")

    assert "BUY_NOW_ELIGIBLE" not in out.columns
    assert "TOP_PICK" not in out.columns
    assert "EST_WIN_RATE" in out.columns
