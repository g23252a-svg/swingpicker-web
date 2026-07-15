# -*- coding: utf-8 -*-
"""
alpha_engine.py — [v29] 검증 통과가 조건인 알파 모델 (AI 축의 재건)
═══════════════════════════════════════════════════
목적:
  "점수가 높으면 실제로 오를 확률이 높은" 점수를 만든다.
  기존 축들의 실측 IC가 음(-)인 상황(STRUCT -0.042, ELITE -0.059)에서,
  과거 일별 추천 CSV(운영 중 생성 — 구조적으로 lookahead 없음)와
  OHLCV 선행수익률로 GBDT를 학습해 ALPHA_SCORE를 산출한다.

정직성 계약 (v26 ML reliability gate와 동일 철학):
  - 매일 밤 월 단위 워크포워드(과거로만 학습→다음 달 검증)를 다시 돌려
    OOS IC t≥2 AND AUC≥0.52 AND Q5-Q1>0 을 전부 통과해야만
    ALPHA_VALIDATED=1로 사용된다. 하나라도 실패하면 가중치 0 + 미사용 표기.
  - 검증 수치는 alpha_model_meta.json에 그대로 저장되고 근거 문장에 노출된다.

2026-07-14 최초 검증 (4개월 워크포워드, OOS 64일 / 22,174행):
  OOS IC +0.146 (t=+7.0) · AUC 0.540 · Q5-Q1 +4.11%/5일
  vs FINAL_SCORE IC -0.037 · ELITE_SCORE IC -0.065 (동일 OOS)
  픽 적용(알파 하위30% 제외): 평균 +3.51% → +4.20%/건

사용처:
  1. ALPHA_SCORE (당일 백분위 0~100) — 표시 순위·근거 문장
  2. 픽 바닥 필터 — 당일 알파 하위 30% 후보 제외 (backtest_validation)
  3. ALPHA_WIN_PROB — 알파 십분위별 실측 승률 캘리브레이션
"""
from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = "alpha_model.joblib"
META_PATH = "alpha_model_meta.json"

# ── 학습에서 제외할 컬럼 (가격 레벨·결과 파생·기존 합성점수) ──
_BAN_COLS = {
    "LDY_RANK", "종가", "추천매수가", "손절가",
    "추천매도가1", "추천매도가2", "추천매도가3",
    "켈리_수량", "추천금액(만원)", "VWAP", "SUPERTREND_VAL", "POC", "MA20", "MA5",
    "KELLY_PLANNED_B", "KELLY_EMPIRICAL_B", "KELLY_FINAL_B", "KELLY_FRACTION",
    "DISPLAY_SCORE", "FINAL_SCORE", "ELITE_SCORE",
    "ALPHA_SCORE", "ALPHA_WIN_PROB", "ALPHA_VALIDATED",
}
# 검증 통과 기준 (전부 충족해야 사용)
GATE_MIN_IC_T = 2.0
GATE_MIN_AUC = 0.52
GATE_MIN_SPREAD = 0.0
# 픽 바닥 필터 (당일 백분위)
ALPHA_FLOOR_PCT = 30.0

_HOLD_BDAYS = 5  # t+1 시가 진입 → t+1+5 종가 (실거래 기준)


def _load_ohlcv_panel(data_dir: str):
    pq = sorted(glob.glob(os.path.join(data_dir, "ohlcv_cache_*.parquet")))
    if not pq:
        return None, None
    o = pd.read_parquet(pq[-1])
    o.index = pd.to_datetime(o.index)
    close = o.pivot_table(index=o.index, columns="종목코드", values="종가", aggfunc="last").sort_index()
    open_ = o.pivot_table(index=o.index, columns="종목코드", values="시가", aggfunc="last").sort_index()
    # [v31.8] 가격 0/음수 글리치 방어 — 시가=0 행(거래정지 등)이 있으면
    # fwd가 ±inf로 오염되어 Q5-Q1 스프레드가 -inf → 검증이 통째로 탈락한다
    # (2026-07-15 실측: 시가=0 86행 → validated=False → 알파 미표시).
    close = close.where(close > 0)
    open_ = open_.where(open_ > 0)
    fwd = close.shift(-(_HOLD_BDAYS + 1)) / open_.shift(-1) - 1
    fwd = fwd.where(np.isfinite(fwd))
    return close, fwd


def _numeric_feature_cols(panel: pd.DataFrame) -> list:
    out = []
    for c in panel.columns:
        if c.startswith("_") or c in _BAN_COLS or c in ("종목코드", "종목명"):
            continue
        if c.endswith("가") or "수량" in c:
            continue
        v = pd.to_numeric(panel[c], errors="coerce")
        if v.notna().mean() > 0.7 and v.std() > 0:
            out.append(c)
    return out


def build_training_panel(data_dir: str = "data", max_days: int = 200):
    """과거 recommend CSV + OHLCV 선행수익률 패널.

    안전장치: CARRY 행 제외, CSV 종가와 실제 종가 1% 이내 일치 행만 사용
    (stale 가격 오염 차단). 선행수익률이 아직 없는 최근 영업일은 자동 제외.
    """
    close, fwd = _load_ohlcv_panel(data_dir)
    if close is None:
        return None
    parts = []
    files = sorted(glob.glob(os.path.join(data_dir, "recommend_2*.csv")))[-max_days:]
    for f in files:
        ymd = os.path.basename(f)[-12:-4]
        try:
            d = pd.to_datetime(ymd)
        except ValueError:
            continue
        if d not in fwd.index:
            continue
        try:
            rec = pd.read_csv(f, encoding="utf-8-sig", dtype={"종목코드": str}, low_memory=False)
        except Exception:
            continue
        if "종목코드" not in rec.columns or "종가" not in rec.columns:
            continue
        rec["종목코드"] = rec["종목코드"].str.zfill(6)
        if "ROUTE" in rec.columns:
            rec = rec[rec["ROUTE"] != "CARRY"]
        cc = pd.to_numeric(rec["종가"], errors="coerce")
        rec = rec[np.isclose(cc, rec["종목코드"].map(close.loc[d]), rtol=0.01)]
        rec["_fwd"] = rec["종목코드"].map(fwd.loc[d])
        rec = rec.dropna(subset=["_fwd"])
        rec = rec[np.isfinite(rec["_fwd"])]  # [v31.8] inf 방어 (이중 안전망)
        if len(rec) < 30:
            continue
        rec["_ymd"] = ymd
        parts.append(rec)
    if not parts:
        return None
    panel = pd.concat(parts, ignore_index=True)
    panel["_y"] = panel.groupby("_ymd")["_fwd"].rank(pct=True)
    return panel


def _make_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_depth=4, max_iter=200, learning_rate=0.05,
        min_samples_leaf=200, l2_regularization=1.0, random_state=42,
    )


def _fit(panel, feats, train_mask):
    """상수(3값 미만) 컬럼 제외 후 학습 → (model, 사용컬럼)."""
    X = panel[feats].apply(pd.to_numeric, errors="coerce")
    Xtr = X[train_mask]
    ok = [c for c in feats if Xtr[c].nunique(dropna=True) >= 3]
    model = _make_model()
    model.fit(Xtr[ok], panel.loc[train_mask, "_y"])
    return model, ok


def _fit_predict(panel, feats, train_mask, pred_mask):
    model, ok = _fit(panel, feats, train_mask)
    X = panel.loc[pred_mask, ok].apply(pd.to_numeric, errors="coerce")
    return model, ok, model.predict(X)


def walk_forward_validate(panel: pd.DataFrame, feats: list, min_train_days: int = 25) -> dict:
    """월 단위 워크포워드 → OOS 지표. 학습에 미래 정보 없음."""
    from sklearn.metrics import roc_auc_score
    panel = panel.copy()
    panel["_ym"] = panel["_ymd"].str[:6]
    months = sorted(panel["_ym"].unique())
    panel["_pred"] = np.nan
    for m in months[1:]:
        tr = panel["_ym"] < m
        te = panel["_ym"] == m
        if panel.loc[tr, "_ymd"].nunique() < min_train_days or te.sum() < 300:
            continue
        try:
            _, _, preds = _fit_predict(panel, feats, tr, te)
            panel.loc[te, "_pred"] = preds
        except Exception as e:
            logger.warning(f"워크포워드 {m} 학습 실패: {e}")
    oos = panel[panel["_pred"].notna()]
    if oos["_ymd"].nunique() < 20:
        return {"ok": False, "reason": f"OOS 일수 부족 ({oos['_ymd'].nunique()}일)"}
    ics, spreads = [], []
    for _, g in oos.groupby("_ymd"):
        # [v31.8] non-finite 수익률 방어 (0원 시가 글리치 등)
        g = g[np.isfinite(pd.to_numeric(g["_fwd"], errors="coerce"))]
        if len(g) < 50:
            continue
        ics.append(float(np.corrcoef(g["_pred"].rank(), g["_fwd"].rank())[0, 1]))
        q = pd.qcut(g["_pred"], 5, labels=False, duplicates="drop")
        _sp = float(g.loc[q == q.max(), "_fwd"].mean() - g.loc[q == 0, "_fwd"].mean())
        if np.isfinite(_sp):
            spreads.append(_sp)
    ics = np.array(ics)
    t = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))) if ics.std(ddof=1) > 0 else 0.0
    try:
        auc = float(roc_auc_score((oos["_fwd"] > 0).astype(int), oos["_pred"]))
    except ValueError:
        auc = float("nan")
    spread = float(np.mean(spreads) * 100)
    validated = bool(t >= GATE_MIN_IC_T and auc >= GATE_MIN_AUC and spread > GATE_MIN_SPREAD)
    # 알파 십분위 → 실측 승률 캘리브레이션 (OOS만 사용)
    calib = []
    try:
        oos = oos.copy()
        oos["_pctile"] = oos.groupby("_ymd")["_pred"].rank(pct=True) * 100
        oos["_dec"] = np.clip((oos["_pctile"] // 10).astype(int), 0, 9)
        for dec, g in oos.groupby("_dec"):
            calib.append({"decile": int(dec), "n": int(len(g)),
                          "win_rate": round(float((g["_fwd"] > 0).mean()), 4),
                          "avg_fwd_pct": round(float(g["_fwd"].mean() * 100), 3)})
    except Exception as e:
        logger.warning(f"캘리브레이션 실패: {e}")
    return {
        "ok": True, "validated": validated,
        "oos_days": int(oos["_ymd"].nunique()), "oos_rows": int(len(oos)),
        "mean_ic": round(float(ics.mean()), 4), "ic_t": round(t, 2),
        "auc": round(auc, 4), "q5q1_spread_pct": round(spread, 3),
        "gate": {"min_ic_t": GATE_MIN_IC_T, "min_auc": GATE_MIN_AUC, "min_spread": GATE_MIN_SPREAD},
        "calibration": calib,
    }


def train_and_save(data_dir: str = "data", trade_ymd: str = None) -> dict:
    """야간 학습: 패널 구축 → 워크포워드 검증 → (통과 시) 전체 학습·저장.

    반환 meta는 alpha_model_meta.json으로 저장.
    검증 실패 시 모델을 저장하지 않고 validated=False meta만 남긴다.
    """
    try:
        import joblib
    except ImportError:
        return {"ok": False, "reason": "joblib 미설치"}
    panel = build_training_panel(data_dir)
    if panel is None or panel["_ymd"].nunique() < 40:
        meta = {"ok": False, "validated": False,
                "reason": f"학습 데이터 부족 ({0 if panel is None else panel['_ymd'].nunique()}일)"}
        _save_meta(data_dir, meta)
        return meta
    feats = _numeric_feature_cols(panel)
    report = walk_forward_validate(panel, feats)
    meta = {"trained_ymd": str(trade_ymd or ""), "n_features": len(feats),
            "features": feats, **report}
    if report.get("ok") and report.get("validated"):
        try:
            model, ok_cols = _fit(panel, feats, panel.index >= 0)
            joblib.dump({"model": model, "features": ok_cols},
                        os.path.join(data_dir, MODEL_PATH))
            meta["model_saved"] = True
        except Exception as e:
            meta["model_saved"] = False
            meta["validated"] = False
            meta["reason"] = f"최종 학습 실패: {e}"
    else:
        meta["model_saved"] = False
    _save_meta(data_dir, meta)
    return meta


def _save_meta(data_dir: str, meta: dict):
    try:
        with open(os.path.join(data_dir, META_PATH), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    except Exception as e:
        logger.warning(f"알파 메타 저장 실패: {e}")


def load_meta(data_dir: str = "data") -> dict:
    try:
        with open(os.path.join(data_dir, META_PATH), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def score_today(df: pd.DataFrame, data_dir: str = "data") -> pd.DataFrame:
    """오늘 추천 df에 ALPHA_SCORE(당일 백분위 0~100)·ALPHA_WIN_PROB·ALPHA_VALIDATED 주입.

    검증 미통과/모델 없음 → ALPHA_VALIDATED=0, 점수 NaN (사용처에서 자동 무시).
    """
    out = df.copy()
    out["ALPHA_SCORE"] = np.nan
    out["ALPHA_WIN_PROB"] = np.nan
    out["ALPHA_VALIDATED"] = 0
    meta = load_meta(data_dir)
    if not meta.get("validated"):
        return out
    model_file = os.path.join(data_dir, MODEL_PATH)
    if not os.path.exists(model_file):
        return out
    try:
        import joblib
        bundle = joblib.load(model_file)
        model, feats = bundle["model"], bundle["features"]
        X = pd.DataFrame({c: pd.to_numeric(out.get(c), errors="coerce")
                          if c in out.columns else np.nan for c in feats},
                         index=out.index)
        preds = model.predict(X)
        pctile = pd.Series(preds, index=out.index).rank(pct=True) * 100
        out["ALPHA_SCORE"] = pctile.round(1)
        # 십분위 캘리브레이션 → 실측 승률
        calib = {c["decile"]: c for c in meta.get("calibration", [])}
        dec = np.clip((pctile // 10).astype(int), 0, 9)
        out["ALPHA_WIN_PROB"] = [
            calib.get(int(d), {}).get("win_rate", np.nan) for d in dec
        ]
        out["ALPHA_VALIDATED"] = 1
    except Exception as e:
        logger.warning(f"알파 예측 실패 (미사용 처리): {e}")
        out["ALPHA_SCORE"] = np.nan
        out["ALPHA_WIN_PROB"] = np.nan
        out["ALPHA_VALIDATED"] = 0
    return out
