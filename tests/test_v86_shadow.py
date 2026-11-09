# -*- coding: utf-8 -*-
"""[v86] 선별 그림자 선등록 실험 — 신규 변형 · 차단일 반사실 · 판정 규칙 · JSON.

docs/SHADOW_PREREG_V86.md 의 규칙이 코드에 고정돼 있는지, 그리고 신규 변형
(comp_lowvol · no_alpha · thin_skip)이 설계대로 고르는지 본다. 실데이터 테스트는
data/recommend_*.csv 로 재구성 현행 풀의 1등이 기록된 PRODUCTION_BUY 와 같은지 확인한다.
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import selection_shadow as SS  # noqa: E402

DATA = ROOT / "data"
DC_SRC = (ROOT / "components" / "decision_center.py").read_text(encoding="utf-8")
DOC = ROOT / "docs" / "SHADOW_PREREG_V86.md"


def _df(n_top=8, blocked=False):
    """A: 알파 통과·고RR(현행 1등) / B: 알파 통과·저변동 / C: 알파 미달·저변동·RR 높음 /
    D: 알파 미달·최저변동 / E: 변동성 게이트 탈락. 패딩 행으로 TOP_PICK 수를 맞춘다."""
    rows = [
        dict(종목코드="000001", 종목명="A", ALPHA_SCORE=95, RR_NOW_TP1=3.0, V23_ATR_Pct=0.06, PRODUCTION_BUY=1),
        dict(종목코드="000002", 종목명="B", ALPHA_SCORE=90, RR_NOW_TP1=1.5, V23_ATR_Pct=0.02, PRODUCTION_BUY=0),
        dict(종목코드="000003", 종목명="C", ALPHA_SCORE=80, RR_NOW_TP1=3.0, V23_ATR_Pct=0.03, PRODUCTION_BUY=0),
        dict(종목코드="000004", 종목명="D", ALPHA_SCORE=50, RR_NOW_TP1=1.4, V23_ATR_Pct=0.01, PRODUCTION_BUY=0),
        dict(종목코드="000005", 종목명="E", ALPHA_SCORE=99, RR_NOW_TP1=3.0, V23_ATR_Pct=0.20, PRODUCTION_BUY=0,
             ALPHA_VOL_OK=0),
    ]
    df = pd.DataFrame(rows)
    df["ALPHA_VOL_OK"] = df["ALPHA_VOL_OK"].fillna(1)
    df["ALPHA_ENTRY_THRESHOLD"] = 85
    df["ENTRY_RISK_GATE_OK"] = 1; df["ALPHA_SURGE_OK"] = 1; df["ALPHA_LT_OK"] = 1
    df["BUY_NOW_PASS"] = 1; df["POC_GAP"] = 5.0; df["ROUTE"] = "ARMED"
    df["QUALITY_GUARD_SCORE"] = 50.0
    df["TOP_PICK"] = ((df["ALPHA_SCORE"] >= 85) & (df["ALPHA_VOL_OK"] == 1)).astype(int)
    pad = n_top - int(df["TOP_PICK"].sum())
    if pad > 0:   # 리스크가드 통과·알파 통과지만 BUY_NOW_PASS=0 → TOP_PICK 수만 늘린다
        p = pd.DataFrame([dict(종목코드=f"9{i:05d}", 종목명=f"P{i}", ALPHA_SCORE=86, RR_NOW_TP1=1.0,
                               V23_ATR_Pct=0.05, PRODUCTION_BUY=0, ALPHA_VOL_OK=1, ALPHA_ENTRY_THRESHOLD=85,
                               ENTRY_RISK_GATE_OK=1, ALPHA_SURGE_OK=1, ALPHA_LT_OK=1, BUY_NOW_PASS=0,
                               POC_GAP=5.0, ROUTE="ARMED", QUALITY_GUARD_SCORE=10.0, TOP_PICK=1)
                          for i in range(pad)])
        df = pd.concat([df, p], ignore_index=True)
    df["NEW_ENTRY_BLOCKED"] = 0
    if blocked:    # risk_off — 전 행 차단, 매수 라벨 봉인, 공식픽·TOP_PICK 0
        df["NEW_ENTRY_BLOCKED"] = True
        df["ROUTE_PRE_BLOCK"] = df["ROUTE"]
        df["ROUTE"] = "WAIT"
        df["PRODUCTION_BUY"] = 0; df["TOP_PICK"] = 0
    return df


def _picks(df, ymd="20261110"):
    return {r["variant"]: r for r in SS.pick_today(df, ymd)}


# ══ 신규 변형 ═══════════════════════════════════════════════════════════
class TestNewVariants:
    def test_variant_registry(self):
        assert set(SS.VARIANTS) >= {"legacy", "live", "lowvol", "ext_lowtv",
                                    "comp_lowvol", "no_alpha", "thin_skip"}
        assert SS.CANDIDATES == ("lowvol", "ext_lowtv", "comp_lowvol", "no_alpha", "thin_skip")

    def test_picks_on_normal_day(self):
        p = _picks(_df())
        assert p["live"]["종목명"] == "A"
        assert p["lowvol"]["종목명"] == "D"                 # ATR 최저
        # comp: 풀 {A,B,C,D} 안 분위 — A 1.0+0.25, B 0.75+0.75, C 0.5+0.5, D 0.25+1.0 → B
        assert p["comp_lowvol"]["종목명"] == "B"
        assert p["no_alpha"]["종목명"] == "A"               # 현행 키 1등은 그대로 A
        assert all(r["blocked"] == 0.0 for r in p.values())

    def test_no_alpha_drops_only_alpha_threshold(self):
        df = _df(); df.loc[df["종목명"] == "A", "ALPHA_SCORE"] = 70   # A 알파 미달
        df.loc[df["종목명"] == "A", "PRODUCTION_BUY"] = 0
        p = _picks(df)
        assert p["no_alpha"]["종목명"] == "C"               # 80×3 = 240 > A 70×3, B 90×1.5
        assert p["no_alpha"]["종목명"] != "E"               # 변동성 게이트는 유지(E 99×3 최고키)
        df.loc[df["종목명"] == "C", "ENTRY_RISK_GATE_OK"] = 0
        assert _picks(df)["no_alpha"]["종목명"] != "C"      # 리스크가드도 유지

    def test_no_alpha_keeps_quality_guard(self):
        df = _df(); df.loc[df["종목명"] == "A", "RR_NOW_TP1"] = 1.2   # RR<1.3 → 품질가드 탈락
        assert _picks(df)["no_alpha"]["종목명"] != "A"

    def test_thin_skip_follows_live_on_thick_day(self):
        p = _picks(_df(n_top=8))
        assert p["thin_skip"]["종목코드"] == p["live"]["종목코드"] and p["thin_skip"]["cash"] == 0.0

    def test_thin_skip_cash_on_thin_day(self):
        p = _picks(_df(n_top=SS.THIN_POOL_MAX))
        assert p["thin_skip"]["cash"] == 1.0 and p["thin_skip"]["종목코드"] == ""
        assert p["thin_skip"]["pool_n"] == SS.THIN_POOL_MAX

    def test_thin_skip_absent_when_live_absent(self):
        df = _df(); df["PRODUCTION_BUY"] = 0
        assert "thin_skip" not in _picks(df)


# ══ 차단일 반사실 ══════════════════════════════════════════════════════
class TestBlockedCounterfactual:
    def test_day_blocked_detection(self):
        assert SS.day_blocked(_df(blocked=True)) is True
        partial = _df(); partial.loc[0, "NEW_ENTRY_BLOCKED"] = 1   # 행 단위 차단은 risk_off 아님
        assert SS.day_blocked(partial) is False

    def test_all_batch_variants_logged_flagged(self):
        p = _picks(_df(n_top=8, blocked=True))
        assert {"legacy", "live", "lowvol", "comp_lowvol", "no_alpha", "thin_skip"} <= set(p)
        assert all(r["blocked"] == 1.0 for r in p.values())
        assert p["live"]["종목명"] == "A", "봉인 전 ROUTE 로 현행 풀을 재구성한다"

    def test_blocked_thin_count_uses_rebuilt_gate(self):
        p = _picks(_df(n_top=3, blocked=True))
        assert p["thin_skip"]["cash"] == 1.0                # 기록 TOP_PICK=0 이 아니라 재구성 수

    def test_normal_day_ignores_row_blocks_only_on_risk_off(self):
        df = _df(); df.loc[df["종목명"] == "C", "NEW_ENTRY_BLOCKED"] = 1
        df.loc[df["종목명"] == "A", "ALPHA_SCORE"] = 70; df["PRODUCTION_BUY"] = 0
        assert _picks(df)["no_alpha"]["종목명"] != "C"      # 정상일 행 차단은 존중


# ══ 선등록 판정 ════════════════════════════════════════════════════════
def _days(n, start="20261109"):
    return [d.strftime("%Y%m%d") for d in pd.bdate_range(start, periods=n)]


class TestPreregVerdict:
    def test_rule_constants_fixed(self):
        assert SS.REGISTERED_FROM == "20261109"
        assert SS.PREREG_DAYS == 40 and SS.PREREG_ALPHA == 0.05
        assert SS.PREREG_HAC_LAG == 5 and SS.PREREG_COST_PCT == 0.25
        assert SS.THIN_POOL_MAX == 5
        assert not hasattr(SS, "TARGET_DAYS"), "20일 판정 문구는 폐기됐다"

    def test_measuring_before_40(self):
        ix = _days(39)
        v = SS.prereg_verdict(pd.Series(1.0, index=ix), pd.Series(0.0, index=ix))
        assert v["status"] == "측정 중 39/40" and not v["decided"]

    def test_promote_when_all_conditions_hold(self):
        ix = _days(40); rng = np.random.default_rng(0)
        live = pd.Series(rng.normal(-1.0, 2.0, 40), index=ix)
        cand = live + 2.0 + rng.normal(0, 0.5, 40)
        v = SS.prereg_verdict(cand, live)
        assert v["status"] == "승격 후보", v
        assert all(v["checks"].values())

    def test_reject_when_net_after_cost_not_positive(self):
        ix = _days(40); rng = np.random.default_rng(1)
        live = pd.Series(rng.normal(-3.0, 1.0, 40), index=ix)
        cand = pd.Series(0.2, index=ix) + rng.normal(0, 0.05, 40)    # 차이 크지만 0.2-0.25 < 0
        v = SS.prereg_verdict(cand, live)
        assert v["status"] == "기각" and v["checks"]["diff_pos"] and not v["checks"]["net_after_cost_pos"]

    def test_cash_days_cost_nothing(self):
        ix = _days(40); rng = np.random.default_rng(1)
        live = pd.Series(rng.normal(-3.0, 1.0, 40), index=ix)
        cand = pd.Series(0.2, index=ix) + rng.normal(0, 0.05, 40)
        traded = pd.Series([i % 4 == 0 for i in range(40)], index=ix)   # 매매 10일만 비용
        assert SS.prereg_verdict(cand, live, traded)["checks"]["net_after_cost_pos"]

    def test_reject_when_halves_disagree(self):
        ix = _days(40)
        live = pd.Series(0.0, index=ix)
        cand = pd.Series([3.0] * 20 + [-0.5] * 20, index=ix) + np.linspace(0, 0.01, 40)
        v = SS.prereg_verdict(cand, live)
        assert v["status"] == "기각" and not v["checks"]["halves_same_sign"]

    def test_fixed_window_ignores_days_after_40(self):
        ix = _days(60); rng = np.random.default_rng(2)
        live = pd.Series(rng.normal(0, 1, 60), index=ix)
        cand = live + np.r_[rng.normal(0, 1, 40), np.full(20, 50.0)]   # 41일째부터 '대박'
        v = SS.prereg_verdict(cand, live)
        assert v["window"] == [ix[0], ix[39]] and v["status"] == "기각"

    def test_only_days_with_both_arms(self):
        ix = _days(45)
        live = pd.Series(0.0, index=ix[:42])
        cand = pd.Series(1.0, index=ix[3:])
        assert SS.prereg_verdict(cand, live)["status"] == "측정 중 39/40"


# ══ build · JSON · 화면 ════════════════════════════════════════════════
def _prices(codes_paths, start="20260901", n=60):
    dates = pd.bdate_range(start, periods=n)
    frames = []
    for code, g in codes_paths:
        px = 10000 * (1 + g) ** np.arange(n)
        frames.append(pd.DataFrame({"Date": dates, "종목코드": code, "시가": px, "고가": px * 1.005,
                                    "저가": px * 0.995, "종가": px}))
    return pd.concat(frames, ignore_index=True)


class TestBuild:
    def _setup(self, tmp_path, monkeypatch):
        px = _prices([("000001", 0.01), ("000002", -0.002), ("000003", 0.003)])
        monkeypatch.setattr(SS, "_load_universe_ohlcv", lambda _d: px)
        monkeypatch.setattr(SS, "LIVE_FROM", "20260901")
        monkeypatch.setattr(SS, "REGISTERED_FROM", "20260910")
        rows = [
            # 탐색기(등록 전) 정상일
            {"ymd": "20260902", "variant": "live", "종목코드": "000001", "blocked": 0.0},
            {"ymd": "20260902", "variant": "lowvol", "종목코드": "000002", "blocked": 0.0},
            # 등록 후 정상일 — thin_skip 현금
            {"ymd": "20260911", "variant": "live", "종목코드": "000001", "blocked": 0.0},
            {"ymd": "20260911", "variant": "lowvol", "종목코드": "000003", "blocked": 0.0},
            {"ymd": "20260911", "variant": "thin_skip", "종목코드": "", "cash": 1.0, "blocked": 0.0},
            # 등록 후 차단일 — 반사실
            {"ymd": "20260914", "variant": "live", "종목코드": "000002", "blocked": 1.0},
            {"ymd": "20260914", "variant": "lowvol", "종목코드": "000003", "blocked": 1.0},
            # 등록 후 정상일 — lowvol 만 (live 현금) → 페어드 제외
            {"ymd": "20260915", "variant": "lowvol", "종목코드": "000003", "blocked": 0.0},
        ]
        for y in sorted({r["ymd"] for r in rows}):
            SS.append_log(str(tmp_path), [{**{"종목명": "", "cash": 0.0}, **r} for r in rows if r["ymd"] == y])
        return SS.build(str(tmp_path))

    def test_segments_split_exploratory_registered_and_blocked(self, tmp_path, monkeypatch):
        s = self._setup(tmp_path, monkeypatch)
        lv = s["variants"]["lowvol"]["segments"]
        assert lv["exploratory"]["normal"]["days"] == 1
        assert lv["registered"]["normal"]["days"] == 2
        assert lv["registered"]["blocked"]["days"] == 1
        assert s["days_blocked"] == 1

    def test_paired_only_both_arms(self, tmp_path, monkeypatch):
        s = self._setup(tmp_path, monkeypatch)
        sg = s["paired"]["lowvol-live"]["segments"]
        assert sg["registered_normal"]["days"] == 1        # 0915 은 live 가 없어 제외
        assert sg["registered_blocked"]["days"] == 1
        assert sg["exploratory_normal"]["days"] == 1

    def test_thin_cash_counts_zero_return(self, tmp_path, monkeypatch):
        s = self._setup(tmp_path, monkeypatch)
        t = s["variants"]["thin_skip"]["segments"]["registered"]["normal"]
        assert t["days"] == 1 and t["avg_ret_pct"] == 0.0
        assert t["win_rate"] is None                       # 현금일은 승률·손절률 표본이 아니다
        assert s["paired"]["thin_skip-live"]["segments"]["registered_normal"]["diff_pct"] < 0

    def test_statuses_and_verdict_text(self, tmp_path, monkeypatch):
        s = self._setup(tmp_path, monkeypatch)
        assert s["statuses"]["lowvol"] == "측정 중 1/40"
        assert s["statuses"]["legacy"] == "탐색"
        assert set(s["statuses"]) == set(SS.CANDIDATES) | {"legacy"}
        assert "선등록" in s["verdict"] and "20일" not in s["verdict"]

    def test_save_writes_strict_json(self, tmp_path, monkeypatch):
        s = self._setup(tmp_path, monkeypatch)
        s["today"] = [{"variant": "ext_lowtv", "alpha": float("nan"), "x": np.float64("inf"),
                       "n": np.int64(3), "b": np.bool_(True)}]
        SS.save(str(tmp_path), s)
        txt = (tmp_path / SS.SUMMARY_NAME).read_text(encoding="utf-8")
        assert "NaN" not in txt and "Infinity" not in txt

        def _bad(c):
            raise ValueError(c)
        d = json.loads(txt, parse_constant=_bad)
        assert d["today"][0]["alpha"] is None and d["today"][0]["n"] == 3 and d["today"][0]["b"] is True

    def test_old_rows_without_blocked_backfilled_from_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(SS, "_load_universe_ohlcv", lambda _d: None)
        monkeypatch.setattr(SS, "LIVE_FROM", "20260901")
        pd.DataFrame({"NEW_ENTRY_BLOCKED": [True] * 40}).to_csv(tmp_path / "recommend_20260903.csv", index=False)
        SS.append_log(str(tmp_path), [{"ymd": "20260903", "variant": "ext_lowtv", "종목코드": "000001", "종목명": ""}])
        SS.append_log(str(tmp_path), [{"ymd": "20260904", "variant": "ext_lowtv", "종목코드": "000001", "종목명": ""}])
        s = SS.build(str(tmp_path))
        assert s["days_blocked"] == 1 and s["days_total"] == 2

    def test_line_and_screen_show_statuses(self, tmp_path, monkeypatch):
        s = self._setup(tmp_path, monkeypatch)
        ln = SS.line(s)
        assert ln.startswith("선별 그림자") and "측정 중" in ln and "탐색기" in ln
        st = dict(s, statuses={**s["statuses"], "comp_lowvol": "승격 후보", "no_alpha": "기각"})
        sl = SS.status_line(st)
        assert sl.index("승격 후보") < sl.index("측정 중") < sl.index("기각")
        assert "_ss.status_line(_ss.load(data_dir))" in DC_SRC and '"shadow_status"' in DC_SRC

    def test_doc_matches_code(self):
        t = DOC.read_text(encoding="utf-8")
        for s in (SS.REGISTERED_FROM, SS.PREREG_VERSION, "40", "0.25%", "HAC", "comp_lowvol",
                  "no_alpha", "thin_skip", "측정 중 n/40", "승격 후보", "기각", "탐색"):
            assert s in t, s


# ══ 실데이터 — 재구성 현행 풀이 기록된 공식픽과 같은가 ═══════════════════
def _csvs(lo="20260914", hi="20261106"):
    out = []
    for f in sorted(glob.glob(str(DATA / "recommend_2*.csv"))):
        y = re.search(r"(\d{8})", os.path.basename(f)).group(1)
        if lo <= y <= hi:
            out.append((y, f))
    return out


@pytest.fixture(scope="module")
def days():
    from services.market_calendar import phantom_batch_days
    ph = set(phantom_batch_days(str(DATA)))
    out = []
    for y, f in _csvs():
        if y in ph:
            continue
        d = pd.read_csv(f, low_memory=False, dtype={"종목코드": str})
        if "ALPHA_ENTRY_OK" not in d.columns:
            continue
        d["종목코드"] = d["종목코드"].astype(str).str.zfill(6)
        out.append((y, d))
    if not out:
        pytest.skip("알파 시대 CSV 없음")
    return out


@pytest.mark.skipif(not _csvs(), reason="data/recommend_*.csv 없음")
class TestRealData:
    def test_rebuilt_live_top_equals_production_buy(self, days):
        n = 0
        for y, d in days:
            if SS.day_blocked(d):
                continue
            pb = d.loc[SS._flag(d, "PRODUCTION_BUY", False), "종목코드"].tolist()
            pool = SS._pools(d)["live"]
            key = SS._keys(d, SS.variant_masks(d))["live"]
            top = d.loc[key.where(pool).idxmax(), "종목코드"] if pool.any() else None
            assert (pb[0] if pb else None) == top, y
            n += 1
        assert n >= 5

    def test_rebuilt_alpha_gate_equals_top_pick_on_normal_days(self, days):
        for y, d in days:
            if not SS.day_blocked(d):
                assert SS.top_pick_count(d) == int(SS._pools(d)["alpha"].sum()), y

    def test_blocked_days_get_counterfactual_picks(self, days):
        blk = [(y, d) for y, d in days if SS.day_blocked(d)]
        if not blk:
            pytest.skip("차단일 없음")
        got = 0
        for y, d in blk:
            assert int(SS._flag(d, "PRODUCTION_BUY", False).sum()) == 0, y
            p = {r["variant"]: r for r in SS.pick_today(d, y)}
            assert all(r["blocked"] == 1.0 for r in p.values())
            got += "live" in p
        assert got >= len(blk) // 2, "차단일 대부분에서 반사실 현행 픽이 있어야 같은 날 비교가 된다"

    def test_new_variants_run_every_day(self, days):
        for y, d in days:
            p = {r["variant"] for r in SS.pick_today(d, y)}
            assert "comp_lowvol" in p or "lowvol" not in p, y
            if "live" in p:
                assert "thin_skip" in p, y
