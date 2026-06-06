# -*- coding: utf-8 -*-
from __future__ import annotations

from components.tab_perf import _shadow_reliability_eval


def test_shadow_reliability_blocks_tiny_sample():
    r = _shadow_reliability_eval(n=3, delta_ev=6.0, changed_rate=0.10, single_ok=True)
    assert r["promotion_ready"] is False
    assert "N<10" in r["blockers"]
    assert "표본 부족" in r["grade"]


def test_shadow_reliability_blocks_high_pick_churn():
    r = _shadow_reliability_eval(n=50, delta_ev=1.2, changed_rate=0.525, single_ok=True)
    assert r["promotion_ready"] is False
    assert "구성변경률>40%" in r["blockers"]


def test_shadow_reliability_allows_review_when_gate_clean():
    r = _shadow_reliability_eval(n=35, delta_ev=0.8, changed_rate=0.25, single_ok=True, rwf_ok=True)
    assert r["promotion_ready"] is True
    assert "승격 검토 가능" in r["verdict"]
