# -*- coding: utf-8 -*-
"""[v29] 알파 엔진 테스트 — 검증 게이트·미사용 폴백·픽 바닥 필터."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import alpha_engine as ae
from backtest_validation import _v27_entry_gate_ok
from services.reco_evidence import build_evidence_row


def _synthetic_panel(n_days=90, n_stocks=80, signal=0.6, seed=0):
    """피처 F1이 선행수익률을 예측하는 합성 패널."""
    rng = np.random.default_rng(seed)
    rows = []
    days = pd.bdate_range("2026-01-05", periods=n_days).strftime("%Y%m%d")
    for d in days:
        f1 = rng.normal(size=n_stocks)
        noise = rng.normal(size=n_stocks)
        fwd = signal * f1 + noise
        for i in range(n_stocks):
            rows.append({"_ymd": d, "종목코드": f"{i:06d}",
                         "F1": f1[i], "F2": rng.normal(), "_fwd": fwd[i]})
    p = pd.DataFrame(rows)
    p["_y"] = p.groupby("_ymd")["_fwd"].rank(pct=True)
    return p


def test_walk_forward_validates_predictive_panel():
    p = _synthetic_panel(signal=0.6)
    r = ae.walk_forward_validate(p, ["F1", "F2"])
    assert r["ok"] and r["validated"]
    assert r["ic_t"] >= 2.0 and r["auc"] >= 0.52
    # 캘리브레이션 단조성 (상위 십분위 승률 > 하위)
    calib = {c["decile"]: c["win_rate"] for c in r["calibration"]}
    assert calib[9] > calib[0]


def test_walk_forward_rejects_noise_panel():
    p = _synthetic_panel(signal=0.0, seed=1)
    r = ae.walk_forward_validate(p, ["F1", "F2"])
    assert r["ok"]
    assert not r["validated"]  # 무신호 → 게이트 탈락


def test_score_today_unvalidated_is_unused(tmp_path):
    # meta 없음 → ALPHA_VALIDATED=0, 점수 NaN
    df = pd.DataFrame({"종목코드": ["000001"], "F1": [1.0]})
    out = ae.score_today(df, data_dir=str(tmp_path))
    assert int(out.loc[0, "ALPHA_VALIDATED"]) == 0
    assert np.isnan(out.loc[0, "ALPHA_SCORE"])


def test_score_today_failed_validation_meta(tmp_path):
    (tmp_path / ae.META_PATH).write_text(
        json.dumps({"ok": True, "validated": False}), encoding="utf-8")
    df = pd.DataFrame({"종목코드": ["000001"]})
    out = ae.score_today(df, data_dir=str(tmp_path))
    assert int(out.loc[0, "ALPHA_VALIDATED"]) == 0


def test_entry_gate_alpha_floor():
    base = {"POC_GAP": "5.0", "MARKET_BREADTH": "60"}
    # 검증 통과 + 하위 30% → 차단
    assert not _v27_entry_gate_ok({**base, "ALPHA_VALIDATED": "1", "ALPHA_SCORE": "12.0"})
    # 검증 통과 + 상위 → 통과
    assert _v27_entry_gate_ok({**base, "ALPHA_VALIDATED": "1", "ALPHA_SCORE": "85.0"})
    # 미검증 → 알파 무시하고 통과
    assert _v27_entry_gate_ok({**base, "ALPHA_VALIDATED": "0", "ALPHA_SCORE": "12.0"})
    # 컬럼 없음 (legacy) → 통과
    assert _v27_entry_gate_ok(base)


def test_evidence_shows_alpha_when_validated():
    e, k, _, _b = build_evidence_row({
        "ALPHA_VALIDATED": 1, "ALPHA_SCORE": 92.0, "ALPHA_WIN_PROB": 0.41,
    })
    assert "AI 알파모델 예측 상위 8%" in e
    assert "실측 승률 41%" in e


def test_evidence_flags_low_alpha_as_risk():
    _, k, _, _b = build_evidence_row({
        "ALPHA_VALIDATED": 1, "ALPHA_SCORE": 15.0, "ALPHA_WIN_PROB": 0.25,
    })
    assert "하위 15%" in k and "픽 제외" in k


def test_evidence_falls_back_to_honest_unused():
    e, _, _, _b = build_evidence_row({"ALPHA_VALIDATED": 0, "AI_SCORE": 0, "ML_TRUSTED": 0})
    assert "AI 예측 미사용" in e


def test_walk_forward_survives_infinite_returns():
    """[v31.8] 시가=0 글리치로 _fwd에 inf가 섞여도 검증이 오염되지 않아야 한다.

    실측 (2026-07-15): OHLCV 시가=0 86행 → Q5-Q1 스프레드 -inf →
    IC t=6.8·AUC 0.60인데도 validated=False → 알파가 하루 걸러 꺼짐.
    """
    p = _synthetic_panel(signal=0.6, seed=2)
    # 일부 행 수익률을 ±inf로 오염 (글리치 재현)
    idx = p.sample(frac=0.01, random_state=0).index
    p.loc[idx, "_fwd"] = np.inf
    p.loc[idx[: len(idx) // 2], "_fwd"] = -np.inf
    r = ae.walk_forward_validate(p, ["F1", "F2"])
    assert r["ok"]
    assert np.isfinite(r["q5q1_spread_pct"])
    assert r["validated"]  # 신호가 살아 있으므로 통과해야 정상
