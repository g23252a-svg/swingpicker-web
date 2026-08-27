# -*- coding: utf-8 -*-
"""v73 — 조용한 각성 레인. 검증된 명세가 코드와 일치하는지, 그리고
이 레인이 현행 산출을 절대 건드리지 않는지를 고정한다."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from services import quiet_breakout as qb


def _px(n=40, vol=1000.0, last_vol=None, close=1000.0, halted_tail=0, flat=False,
        seed=0):
    """실제 밴드 종목을 흉내낸다 — vol_20d 중위가 3.7% 수준이어야
    FROZEN_MIN_VOL_PCT(0.5%) 문턱에 걸리지 않는다."""
    idx = pd.RangeIndex(n)
    if flat:
        c = np.full(n, close)
    else:
        r = np.random.default_rng(seed).normal(0, 0.03, n)   # 일 3% 변동
        c = close * np.exp(np.cumsum(r))
    d = pd.DataFrame({"시가": c * 0.99, "고가": c * 1.02, "저가": c * 0.98,
                      "종가": c, "거래량": np.full(n, vol)}, index=idx)
    if last_vol is not None:
        d.loc[d.index[-1], "거래량"] = last_vol
    for k in range(halted_tail):
        d.loc[d.index[-1 - k], ["시가", "고가", "저가"]] = 0.0
    return d


def _rank(n=1500, tv_top=1e12, tv_bot=1e8):
    codes = [f"{i:06d}" for i in range(n)]
    tv = np.linspace(tv_top, tv_bot, n)
    return pd.DataFrame({"종목코드": codes, "거래대금(원)": tv})


def _map(n=1500, spike_at=(), vol=1000.0, spike=5000.0, **kw):
    """밴드(601~1200) 밖 종목은 어차피 후보가 아니므로 만들지 않는다.
    다만 밴드 하한을 지우는 변이를 잡으려면 1~600 에도 급등을 넣을 수 있어야 하므로
    spike_at 로 지정된 코드는 밴드 밖이어도 만든다."""
    need = set(range(qb.RANK_LO, min(qb.RANK_HI, n))) | set(spike_at)
    return {f"{i:06d}": _px(vol=vol, last_vol=(spike if i in spike_at else None),
                            seed=i, **kw)
            for i in sorted(need)}


# ── 검정된 명세가 코드에 그대로 있는가 ────────────────────────
def test_spec_constants_match_research():
    """docs/PREDICTIVE_POWER_20260827.md §8~9 에서 검정된 값."""
    assert (qb.RANK_LO, qb.RANK_HI) == (600, 1200)
    assert qb.VOL_WINDOW == 20
    assert qb.TOP_N == 5
    assert qb.HOLD_DAYS == 5 and qb.STOP_PCT == -0.08


def test_module_documents_its_own_weakness():
    src = open(os.path.join(ROOT, "services", "quiet_breakout.py"), encoding="utf-8").read()
    for k in ("믿지 말아야 할 것", "양수일", "다중검정", "병렬 레인"):
        assert k in src, f"약점 기술 누락: {k}"


# ── 밴드 ──────────────────────────────────────────────────────
def test_only_band_ranks_are_considered():
    r = qb.build(_rank(), _map(spike_at=tuple(range(0, 1500, 7))))
    assert r["ok"]
    assert all(qb.RANK_LO < p["거래대금순위"] <= qb.RANK_HI for p in r["picks"])


def test_top600_spike_never_picked():
    """현행 유니버스(1~600)에서는 효과가 없다 — 아예 후보로 넣지 않는다."""
    r = qb.build(_rank(), _map(spike_at=tuple(range(0, 600))))
    assert r["ok"] is False and "없음" in r["reason"]


def test_rank_table_too_small_is_reported_not_crashed():
    r = qb.build(_rank(n=300), _map(n=300))
    assert r["ok"] is False
    assert "1200" in r["reason"] and r["universe_size"] == 300


def test_empty_rank_table():
    assert qb.build(pd.DataFrame(), {})["ok"] is False


def test_missing_columns_reported():
    assert qb.build(pd.DataFrame({"x": [1]}), {})["ok"] is False
    assert qb.build(pd.DataFrame({"종목코드": ["000001"]}), {})["ok"] is False


# ── 랭킹 ──────────────────────────────────────────────────────
def test_ranked_by_vol_ratio_descending():
    m = _map()
    for i, v in ((700, 9000.0), (800, 7000.0), (900, 5000.0), (1000, 3000.0)):
        m[f"{i:06d}"] = _px(seed=i, last_vol=v)
    r = qb.build(_rank(), m, top_n=4)
    assert [p["종목코드"] for p in r["picks"]] == ["000700", "000800", "000900", "001000"]
    assert r["picks"][0]["vol_ratio"] > r["picks"][-1]["vol_ratio"]


def test_min_vol_ratio_filters():
    m = _map()
    m["000700"] = _px(seed=700, last_vol=1000 * (qb.MIN_VOL_RATIO - 0.2))   # 문턱 아래
    m["000800"] = _px(seed=800, last_vol=1000 * (qb.MIN_VOL_RATIO + 0.2))   # 문턱 위
    r = qb.build(_rank(), m)
    assert [p["종목코드"] for p in r["picks"]] == ["000800"]
    assert r["skipped"]["low_vol_ratio"] > 0


def test_top_n_respected():
    m = _map(spike_at=tuple(range(601, 1200)))
    for n in (1, 3, 5, 20):
        assert len(qb.build(_rank(), m, top_n=n)["picks"]) == n


# ── 거래정지·동결 방어 ────────────────────────────────────────
def test_halted_today_excluded():
    m = _map()
    m["000700"] = _px(seed=700, last_vol=9000.0, halted_tail=1)
    m["000800"] = _px(seed=800, last_vol=5000.0)
    r = qb.build(_rank(), m)
    assert [p["종목코드"] for p in r["picks"]] == ["000800"]


def test_long_halt_excluded():
    """정지가 길면 레인지·비율 지표가 통째로 망가진다."""
    m = _map()
    m["000700"] = _px(seed=700, last_vol=9000.0, halted_tail=5)
    m["000800"] = _px(seed=800, last_vol=5000.0)
    assert [p["종목코드"] for p in qb.build(_rank(), m)["picks"]] == ["000800"]


def test_frozen_price_excluded():
    """가격이 전혀 안 움직이는 종목 — 상장폐지 직전의 전형."""
    m = _map()
    m["000700"] = _px(seed=700, last_vol=9000.0, flat=True)
    m["000800"] = _px(seed=800, last_vol=5000.0)
    r = qb.build(_rank(), m)
    assert [p["종목코드"] for p in r["picks"]] == ["000800"]
    assert r["skipped"]["frozen"] > 0


def test_zero_low_does_not_leak_into_output():
    m = _map()
    m["000700"] = _px(seed=700, last_vol=9000.0, halted_tail=2)
    r = qb.build(_rank(), m)
    assert all(p["종목코드"] != "000700" for p in r.get("picks", []))


# ── 이력·결측 ─────────────────────────────────────────────────
def test_short_history_excluded():
    m = _map()
    m["000700"] = _px(seed=700, n=qb.MIN_HISTORY - 1, last_vol=9000.0)
    m["000800"] = _px(seed=800, last_vol=5000.0)
    r = qb.build(_rank(), m)
    assert [p["종목코드"] for p in r["picks"]] == ["000800"]
    assert r["skipped"]["short_history"] > 0


def test_missing_ohlcv_counted():
    m = _map()
    del m["000700"]
    assert qb.build(_rank(), m)["skipped"]["no_ohlcv"] >= 1


def test_zero_volume_history_excluded():
    """20일간 거래량 0이던 종목이 오늘 거래되면 vol_ratio 가 20배로 튄다.
    그건 '조용한 각성'이 아니라 거래 재개다 — 성질이 다르므로 배제한다."""
    m = _map()
    m["000700"] = _px(seed=700, vol=0.0, last_vol=9000.0)
    m["000800"] = _px(seed=800, last_vol=5000.0)
    r = qb.build(_rank(), m)
    assert [p["종목코드"] for p in r["picks"]] == ["000800"]


def test_thin_volume_history_excluded():
    """거래일이 드문드문한 종목도 같은 이유로 뺀다."""
    m = _map()
    d = _px(seed=700, last_vol=9000.0)
    d.loc[d.index[-19:-1], "거래량"] = 0.0          # 최근 20일 중 2일만 거래
    m["000700"] = d
    m["000800"] = _px(seed=800, last_vol=5000.0)
    assert [p["종목코드"] for p in qb.build(_rank(), m)["picks"]] == ["000800"]


def test_prepare_returns_none_on_garbage():
    assert qb.prepare(None) is None
    assert qb.prepare(pd.DataFrame()) is None
    assert qb.prepare(pd.DataFrame({"종가": [1, 2, 3]})) is None


def test_nan_never_written_to_json(tmp_path):
    m = _map()
    m["000700"] = _px(seed=700, n=30, last_vol=9000.0)     # ret_60d 계산 불가
    r = qb.build(_rank(), m)
    qb.save(str(tmp_path), "20260826", r)
    txt = open(tmp_path / "quiet_breakout_20260826.json", encoding="utf-8").read()
    assert "NaN" not in txt and "Infinity" not in txt
    json.loads(txt)


# ── 저장·읽기 ─────────────────────────────────────────────────
def test_save_load_roundtrip(tmp_path):
    r = qb.build(_rank(), _map(spike_at=(700, 800)))
    qb.save(str(tmp_path), "20260826", r)
    assert qb.load(str(tmp_path), "20260826")["picks"] == r["picks"]
    assert qb.load(str(tmp_path))["picks"] == r["picks"]


def test_load_missing_returns_none(tmp_path):
    assert qb.load(str(tmp_path)) is None


def test_load_corrupt_returns_none(tmp_path):
    (tmp_path / qb.CACHE_NAME).write_text("{not json", encoding="utf-8")
    assert qb.load(str(tmp_path)) is None


def test_save_failure_does_not_raise():
    qb.save("/proc/nonexistent-dir-xyz", "20260826", {"ok": True})


# ── 화면 문구 ─────────────────────────────────────────────────
def test_line_marks_lane_as_unverified_and_not_orderable():
    l = qb.line(qb.build(_rank(), _map(spike_at=(700,))))
    assert "검증중" in l and "주문 대상이 아니다" in l


def test_line_handles_none_and_failure():
    assert "데이터 없음" in qb.line(None)
    assert "산출 못 함" in qb.line({"ok": False, "reason": "x"})


def test_report_carries_caveat():
    r = qb.build(_rank(), _map(spike_at=(700,)))
    assert "PRODUCTION_BUY" in r["caveat"] and "주문을 내지 않는다" in r["caveat"]
    assert r["spec"]["rank_band"] == [qb.RANK_LO, qb.RANK_HI]


# ── 현행 산출 불변 (이게 이 패치의 핵심 약속) ─────────────────
def test_lane_never_touches_decision_columns():
    src = open(os.path.join(ROOT, "services", "quiet_breakout.py"), encoding="utf-8").read()
    for forbidden in ("PRODUCTION_BUY\"", "PRODUCTION_BUY'", "켈리_수량", "추천수량",
                      "df_out[", "ROUTE ="):
        assert forbidden not in src, f"결정 컬럼을 건드린다: {forbidden}"


def test_batch_wiring_is_display_only():
    src = open(os.path.join(ROOT, "pipeline_finalize.py"), encoding="utf-8").read()
    i = src.index("[v73] 조용한 각성 레인")
    blk = src[i:i + 1200]
    assert "run_batch" in blk and "df_out" not in blk
    assert "except Exception" in blk, "배치를 깨뜨릴 수 있다"


def test_batch_wiring_logs_no_impact():
    src = open(os.path.join(ROOT, "pipeline_finalize.py"), encoding="utf-8").read()
    i = src.index("[v73] 조용한 각성 레인")
    assert "현행 산출에 영향 없음" in src[i:i + 1200]


def test_lane_uses_its_own_ohlcv_cache_not_the_main_one():
    """메인 캐시에 601~1200위를 넣으면 그걸 읽는 v55/v69/v58 산출이 조용히 달라진다.

    문서에서 이름을 **언급**하는 것은 괜찮다 — 실제로 **부르는지**를 본다.
    """
    import ast
    src = open(os.path.join(ROOT, "services", "quiet_breakout.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    called |= {n.func.attr for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names}
    for forbidden in ("prepare_ohlcv_data", "save_ohlcv_cache", "load_ohlcv_cache"):
        assert forbidden not in called, f"메인 캐시를 건드린다: {forbidden}()"
        assert forbidden not in imported, f"메인 캐시 함수를 import 한다: {forbidden}"
    assert qb.LANE_CACHE.startswith("quiet_lane_")


def test_run_batch_survives_no_network(tmp_path):
    r = qb.run_batch("20260826", "20250101", "20260826", data_dir=str(tmp_path))
    assert isinstance(r, dict) and "ok" in r


def test_max_fetch_is_bounded():
    assert 0 < qb.MAX_FETCH <= 1000
