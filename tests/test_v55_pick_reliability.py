# -*- coding: utf-8 -*-
"""[v55] 픽 신뢰도 — 평균만 보여주면 과신을 부른다.

■ 왜 필요한가 (v55 실측)
  프로덕션 퍼널을 그대로 재현한 하루 1픽의 분포:
      15일 · 평균 +6.20% · 중위 **-2.70%** · 승률 46.7% · sd 25.18 · 왜도 2.10
      상위 2건 +78.6% / +52.0%  →  빼면 평균 -2.89%
      하위 5건 전부 -8.0%(스톱 체결)
  즉 승률로 버는 전략이 아니라 드문 대박으로 버는 우측 꼬리 전략이다.
  화면이 '품질 87 · 손익비 5.06 · 승률 39%'만 보여주면 사용자는 이걸
  '이기는 전략'으로 읽고 한 픽에 큰돈을 넣는다. 중위값과 이상치 의존도를
  같이 보여주는 것이 정직하다.

■ 계약 재적용이 핵심
  과거 CSV의 PRODUCTION_BUY는 당시 계약(v32~v53)의 산물이다(115개 파일 중
  컬럼이 있는 것은 18개뿐). 지금 화면이 약속하는 것은 v54 계약이므로
  현재 가드를 다시 돌려 픽을 재현한 뒤 실현수익을 계산한다.

■ 표본 0도 정보다
  2026-07~08은 코스피가 20일선 대비 최대 -20% 이탈한 폭락으로 risk_off가
  최근 16영업일 중 15일 전 종목 신규진입을 차단했다. 그 기간에는 공식 매수가
  나올 수 없었으므로 표본도 없다. '통계 없음'이 아니라 '차단되어 없음'이라고
  말해야 한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import pick_reliability as pr  # noqa: E402


def _fut(rows):
    """rows: (시가, 고가, 저가, 종가) 목록 — 진입일 포함."""
    return pd.DataFrame(rows, columns=["시가", "고가", "저가", "종가"])


# ── ① 청산 시뮬 정확성 ──

def test_hold_rule_exits_at_horizon_close():
    fut = _fut([(100, 105, 99, 104)] * pr.HOLD_DAYS)
    assert abs(pr._simulate(fut, "hold") - 0.04) < 1e-9


def test_hold_rule_stops_out_at_minus8():
    fut = _fut([(100, 101, 91, 92)] + [(92, 93, 90, 91)] * 4)
    assert abs(pr._simulate(fut, "hold") - (-0.08)) < 1e-9


def test_stop_takes_priority_over_target_same_day():
    """같은 날 스톱과 익절이 모두 닿으면 보수적으로 스톱."""
    fut = _fut([(100, 120, 85, 100)] * pr.HOLD_DAYS)
    assert pr._simulate(fut, "hold") < 0


def test_disc_rule_caps_gain_at_tp_quick():
    """화면 규율은 +10%에서 자른다 — 대박을 못 먹는다."""
    fut = _fut([(100, 180, 99, 175)] * pr.HOLD_DAYS)
    hold = pr._simulate(fut, "hold")
    disc = pr._simulate(fut, "disc")
    assert hold > 0.5, "보유 규칙은 꼬리를 먹는다"
    assert disc <= 0.11, f"화면 규율이 +10% 이상을 먹었다: {disc}"
    assert disc < hold


def test_disc_rule_half_exit_then_breakeven_stop():
    """2일 내 +2% 도달 → 절반 익절 + 잔여 본전스톱. 되돌려도 손실이 얕다."""
    fut = _fut([(100, 103, 99, 102), (102, 102, 80, 81)] + [(81, 82, 80, 81)] * 3)
    v = pr._simulate(fut, "disc")
    assert v > -0.02, f"본전스톱이 작동해야 한다: {v}"
    assert v < 0.02


def test_simulate_needs_two_rows():
    assert pr._simulate(_fut([(100, 101, 99, 100)]), "hold") is None
    assert pr._simulate(None, "hold") is None


def test_simulate_rejects_bad_entry():
    assert pr._simulate(_fut([(0, 1, 0, 1)] * 3), "hold") is None


# ── ② 통계 요약이 이상치 의존을 드러내는가 ──

def test_stats_expose_outlier_dependence():
    vals = [0.786, 0.52, 0.03, -0.08, -0.08, -0.08, -0.027, 0.01, -0.05, -0.02]
    st = pr._stats(vals)
    assert st["n"] == 10
    assert st["median_pct"] < 0, "중위가 음수라는 사실이 드러나야 한다"
    assert st["mean_pct"] > 0
    assert st["top2_share_pct"] >= 50
    assert st["mean_ex_top2_pct"] < st["mean_pct"]
    assert "mean_ci95_pct" in st and len(st["mean_ci95_pct"]) == 2


def test_stats_empty_is_safe():
    assert pr._stats([]) == {}
    assert pr._stats([None, float("nan")]) == {}


def test_small_sample_has_no_ci():
    st = pr._stats([0.01, 0.02, -0.03])
    assert st["n"] == 3 and "mean_ci95_pct" not in st


# ── ③ 화면 문구가 확신 대신 분포를 말하는가 ──

def test_line_reports_median_not_only_mean():
    res = {"ok": True, "hold": {"n": 20, "mean_pct": 6.2, "median_pct": -2.7,
                                "win_rate_pct": 46.7, "top2_share_pct": 88}}
    line = pr.reliability_line(res)
    assert "중위 -2.7%" in line
    assert "평균 +6.2%" in line
    assert "상위 2건" in line, "이상치 의존을 밝혀야 한다"


def test_line_admits_thin_sample():
    line = pr.reliability_line({"ok": True, "hold": {"n": 3}})
    assert "아직 적습니다" in line


def test_line_empty_when_unavailable():
    assert pr.reliability_line(None) == ""
    assert pr.reliability_line({"ok": False}) == ""


# ── ④ 표본 0을 '차단되어 없음'으로 말하는가 ──

def test_no_picks_says_blocked_not_missing(tmp_path):
    (tmp_path / "ohlcv_cache_20260805.parquet").unlink(missing_ok=True)
    pd.DataFrame({"종목코드": ["005930"], "시가": [100], "고가": [101],
                  "저가": [99], "종가": [100]}).to_parquet(
        tmp_path / "ohlcv_cache_20260805.parquet", index=False)
    # 픽이 나올 수 없는 CSV (TOP_PICK 없음)
    pd.DataFrame({"종목코드": ["005930"], "종목명": ["삼성전자"],
                  "TOP_PICK": [0], "BUY_NOW_ELIGIBLE": [0]}).to_csv(
        tmp_path / "recommend_20260805.csv", index=False, encoding="utf-8-sig")
    res = pr.compute_pick_reliability(str(tmp_path))
    assert res["ok"] is False
    assert "block_note" in res, "차단 때문일 수 있다는 사실을 남겨야 한다"
    assert "risk_off" in res["block_note"]


def test_reliability_definitions_are_recorded():
    src = (ROOT / "services" / "pick_reliability.py").read_text(encoding="utf-8")
    assert "중위 **-2.70%**" in src, "우측 꼬리 실측 근거 미기록"
    assert "t=-1.76" in src or "-2.10%p" in src
