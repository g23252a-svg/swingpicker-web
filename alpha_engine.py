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

# ═══════════════════════════════════════════════════════════════
# [v32] 알파 전면 진입 게이트 — ROUTE(ATTACK/ARMED) 거부권 대체
# ───────────────────────────────────────────────────────────────
# 근거(3~7월 워크포워드 OOS 실측, 미래정보 미사용):
#   · ROUTE ATTACK 알파 -2.9%p (t=-3.56, p=0.0004) — 5개월 전부 음수, 양 국면 음수
#   · ROUTE ARMED  알파  0.0%p (t=0.02, p=0.99) — 완전 노이즈
#   · ELITE_SCORE  IC   -0.06 (t=-5.30) — 점수 높을수록 수익 낮음(역상관)
#   · 알파 모델    IC   +0.19 (t=6.8, AUC 0.60), Q5-Q1 +5.05%p, 십분위 승률 24%→41% 단조
#   반사실: 알파 top3 +0.81% vs ROUTE ATK/ARM -5.91% vs 유니버스 -4.26%
# → ROUTE를 진입 게이트에서 제거하고 검증된 알파를 primary 게이트로.
#   레짐 적응형 백분위 문턱(사용자 선택: 상승30%/하락10%):
# [v40] NEUTRAL 80→85 상향 — OOS 실측: 알파≥80(+1.35%/승률59%) vs
#   ≥85(+1.62%/승률61%, t=2.08). 매수검토 풀 35→27개/일로 정제(품질↑).
#   상위 픽(top1~3)은 이미 고알파라 영향 없음 — 관찰 리스트 청소 효과.
ALPHA_GATE_THRESHOLD = {
    "UP": 70.0,       # 상승 국면 → 상위 30% (검증 범위 밖 — 기존 유지)
    "NEUTRAL": 85.0,  # 중립 국면 → 상위 15% [v40 상향: 80→85]
    "DOWN": 90.0,     # 하락 국면 → 상위 10% (진입 자체는 레짐 게이트가 추가 차단)
    "UNKNOWN": 85.0,  # 미상 → 중립 취급 [v40: 80→85]
}
ALPHA_GATE_DEFAULT_THRESHOLD = 85.0

# ═══════════════════════════════════════════════════════════════
# [v32.1] ROUTE 자체 치료 — 상태 판정을 데이터 방향으로 재설계
# ───────────────────────────────────────────────────────────────
# 진단(3~7월 OOS): 기존 ATTACK은 역방향 피처로 조립 —
#   거래품질 IC -0.10(t=-7.5), MACD기울기 -0.055(t=-4.7), 레인지위치 -0.012,
#   MFI -0.048, 타이밍 +0.004(무의미). 유일한 양(+) 구조는 저점추세 +0.031.
# 재설계: 강도=검증 알파, 상태=올바른 방향 기술구조(MA20 위 + 저점추세 상승 + 과열 제외).
# 검증(OOS 상태별 알파): 기존 ATTACK -4.64%p → 신 ATTACK +3.70%p(승률 36%),
#   ARMED -0.38→+1.56%p, OVERHEAT -0.55→-2.16%p(위험 정확 표시). 순서 정상화.
_ROUTE_ARMED_ALPHA_GAP = 15.0     # ARMED 문턱 = ATTACK 문턱 - 15
_ROUTE_OVERHEAT_RET5 = 15.0       # 5일 급등 → 과열(역신호)
_ROUTE_OVERHEAT_MFI = 85.0
_ROUTE_OVERHEAT_RSI = 78.0
# ROUTE를 재계산해도 보존할 상태(보유/청산/캐리 의미). 신규진입 상태만 재판정.
_ROUTE_PRESERVE = {"CARRY", "EXIT", "EXIT_WARNING", "BLOCKED"}

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
    # [v36] 미국 지수 오버나이트 피처 주입 (us_daily.csv 있을 때만)
    panel = inject_us_features(panel, data_dir, ymd_col="_ymd")
    return panel


# ═══════════════════════════════════════════════════════════════
# [v36] 미국 지수 오버나이트 피처 — 나스닥·S&P 전일 마감
# ───────────────────────────────────────────────────────────────
# 근거: 코스피-나스닥 연동(반도체 비중 ~30% + 외국인 수급 + NQ 선물 24h).
# look-ahead 방지가 핵심: 한국 거래일 d의 피처는 반드시 us_date < d 인
# 마지막 미국 마감만 사용 (미국 d일 마감은 한국 d+1 새벽에야 확정).
# 시장 공통 피처(종목 간 동일값)지만 GBDT가 종목 피처와의 상호작용으로
# 활용(예: 나스닥 약세 다음날엔 저베타/방어주 우위). 가치는 워크포워드
# 검증 게이트가 자동 판정 — 무가치하면 모델이 무시, 검증 실패 시 미사용.
US_OVERNIGHT_COLS = ["US_NDX_RET_1D", "US_NDX_RET_5D", "US_NDX_VS_MA20",
                   "US_SPX_RET_1D"]


def _load_us_feature_table(data_dir: str):
    """us_daily.csv → 미국 마감일 기준 피처 테이블 (없으면 None)."""
    path = os.path.join(data_dir, "us_daily.csv")
    if not os.path.exists(path):
        return None
    try:
        us = pd.read_csv(path)
        us["date"] = us["date"].astype(str).str.replace("-", "").str.slice(0, 8)
        us = us.sort_values("date").reset_index(drop=True)
        ndx = pd.to_numeric(us.get("ndx_close"), errors="coerce")
        spx = pd.to_numeric(us.get("spx_close"), errors="coerce")
        out = pd.DataFrame({"us_date": us["date"]})
        out["US_NDX_RET_1D"] = ndx.pct_change() * 100
        out["US_NDX_RET_5D"] = ndx.pct_change(5) * 100
        ma20 = ndx.rolling(20, min_periods=10).mean()
        out["US_NDX_VS_MA20"] = (ndx / ma20 - 1.0) * 100
        out["US_SPX_RET_1D"] = spx.pct_change() * 100
        return out.dropna(subset=["us_date"])
    except Exception as e:
        logger.warning(f"[v36] us_daily.csv 로드 실패 (피처 생략): {e}")
        return None


def inject_us_features(df: pd.DataFrame, data_dir: str,
                       ymd_col: str = "_ymd") -> pd.DataFrame:
    """한국 거래일별로 '그 전'(us_date < 한국일) 마지막 미국 피처를 조인.

    us_daily.csv 없거나 로드 실패 → 원본 그대로 (피처 컬럼 미생성; GBDT는
    학습 피처 목록에 없으면 자연히 미사용, 있는데 NaN이면 NaN 처리).
    """
    us = _load_us_feature_table(data_dir)
    if us is None or df is None or len(df) == 0 or ymd_col not in df.columns:
        return df
    out = df.copy()
    # merge_asof: 한국일 d에 대해 us_date <= d-1 (엄격히 과거) 마지막 행
    left = pd.DataFrame({"_k_ymd": sorted(out[ymd_col].astype(str).unique())})
    left["_k_dt"] = pd.to_datetime(left["_k_ymd"], format="%Y%m%d", errors="coerce")
    right = us.copy()
    right["_us_dt"] = pd.to_datetime(right["us_date"], format="%Y%m%d", errors="coerce")
    right = right.dropna(subset=["_us_dt"]).sort_values("_us_dt")
    left = left.dropna(subset=["_k_dt"]).sort_values("_k_dt")
    joined = pd.merge_asof(
        left, right, left_on="_k_dt", right_on="_us_dt",
        direction="backward", allow_exact_matches=False,  # us_date < 한국일 (look-ahead 금지)
    )
    feat_map = joined.set_index("_k_ymd")[US_OVERNIGHT_COLS]
    for c in US_OVERNIGHT_COLS:
        out[c] = out[ymd_col].astype(str).map(feat_map[c])
    return out


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


def score_today(df: pd.DataFrame, data_dir: str = "data",
                trade_ymd: str = None) -> pd.DataFrame:
    """오늘 추천 df에 ALPHA_SCORE(당일 백분위 0~100)·ALPHA_WIN_PROB·ALPHA_VALIDATED 주입.

    검증 미통과/모델 없음 → ALPHA_VALIDATED=0, 점수 NaN (사용처에서 자동 무시).
    [v36] trade_ymd가 주어지면 미국 지수 오버나이트 피처를 조인해 예측에 사용
    (학습과 동일하게 us_date < trade_ymd 만 — look-ahead 없음).
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
        # [v36] 미국 피처 주입 — 임시 _ymd 컬럼으로 조인 후 제거
        if trade_ymd and "US_NDX_RET_1D" not in out.columns:
            try:
                _tmp = out.copy()
                _tmp["_ymd"] = str(trade_ymd)
                _tmp = inject_us_features(_tmp, data_dir, ymd_col="_ymd")
                for c in US_OVERNIGHT_COLS:
                    if c in _tmp.columns:
                        out[c] = _tmp[c].values
            except Exception as _ue:
                logger.warning(f"[v36] 미국 피처 주입 실패 (NaN 처리): {_ue}")
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


def _alpha_threshold_series(df: pd.DataFrame) -> pd.Series:
    """레짐(MARKET_REGIME)별 알파 백분위 문턱 시리즈."""
    regime = (
        df.get("MARKET_REGIME", pd.Series("UNKNOWN", index=df.index))
        .astype(str).str.upper().str.strip()
    )
    return regime.map(ALPHA_GATE_THRESHOLD).fillna(ALPHA_GATE_DEFAULT_THRESHOLD)


def apply_alpha_entry_gate(df: pd.DataFrame) -> pd.DataFrame:
    """[v32] 알파를 primary 진입 게이트로 적용 — ROUTE 거부권 대체.

    검증 통과(ALPHA_VALIDATED==1)일 때만 작동한다:
      TOP_PICK = 리스크가드(ENTRY_RISK_GATE_OK) AND 알파백분위 ≥ 레짐문턱
                 AND 저점추세 유지(Low_Trend_PCT ≥ 0, 결측은 통과)   [v37]
    · ROUTE는 건드리지 않는다(타이밍 배지로만 존치).
    · ENTRY_RISK_GATE_OK가 없으면(레거시) close>stop 등 기본 가드만 재구성.
    · 미검증이면 TOP_PICK을 그대로 두고 ALPHA_GATE_ACTIVE=0 (레거시 폴백).

    주입 컬럼:
      ALPHA_GATE_ACTIVE, ALPHA_ENTRY_THRESHOLD, ALPHA_ENTRY_OK, ALPHA_LT_OK
    """
    out = df.copy()
    n = len(out)
    validated = (
        int(pd.to_numeric(out.get("ALPHA_VALIDATED", 0), errors="coerce")
            .fillna(0).astype(int).max()) == 1
        if n else False
    )
    thr = _alpha_threshold_series(out) if n else pd.Series(dtype=float)
    out["ALPHA_ENTRY_THRESHOLD"] = thr

    if not validated:
        out["ALPHA_GATE_ACTIVE"] = 0
        out["ALPHA_ENTRY_OK"] = 0
        return out

    ascore = pd.to_numeric(out.get("ALPHA_SCORE", np.nan), errors="coerce")

    # 리스크 가드 — scoring_engine이 심어둔 ENTRY_RISK_GATE_OK 우선 사용.
    if "ENTRY_RISK_GATE_OK" in out.columns:
        risk_ok = out["ENTRY_RISK_GATE_OK"].fillna(False).astype(bool)
    else:
        num = lambda c: pd.to_numeric(out.get(c, np.nan), errors="coerce")
        close, stop, tp1, buy = num("종가"), num("손절가"), num("추천매도가1"), num("추천매수가")
        risk = (close - stop).clip(lower=1)
        reward = (tp1 - close).clip(lower=0)
        rr = reward / risk
        egap = (close - buy).abs() / buy.clip(lower=1) * 100
        poc = num("POC_GAP")
        tov = num("거래대금(억원)")
        risk_ok = (
            (close > stop) & (close < tp1) & (tov >= 50)
            & (egap <= 5.0) & (rr >= 1.0) & (poc.isna() | (poc <= 20.0))
        ).fillna(False)

    # 손실방어 차단 플래그는 존중한다(7월방어·회복차단·신규진입차단).
    def _blk(col):
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").fillna(0).astype(bool)
        return pd.Series(False, index=out.index)
    blocked = _blk("NEW_ENTRY_BLOCKED") | _blk("JULY_PROFIT_BLOCK_FLAG") | _blk("PROFIT_RECOVERY_BLOCK_FLAG")

    # [v37] 저점추세 결합 — 알파 통과 종목을 Low_Trend_PCT 방향으로 가르면
    # LT≥0 +1.97%/승률48% vs LT<0 -0.12%/45% (OOS n=4,230, 일별페어드 p=0.015).
    # 게이트 풀이 절반으로 정제되고 치료된 ATTACK 정의와도 일관된다.
    # 결측은 통과 처리 — 데이터 이슈가 전면 차단으로 번지지 않게.
    if "Low_Trend_PCT" in out.columns:
        lt = pd.to_numeric(out["Low_Trend_PCT"], errors="coerce")
    else:
        lt = pd.Series(np.nan, index=out.index)
    lt_ok = lt.isna() | (lt >= 0)
    out["ALPHA_LT_OK"] = lt_ok.astype(int)

    entry_ok = risk_ok & ascore.notna() & (ascore >= thr) & (~blocked) & lt_ok
    out["ALPHA_GATE_ACTIVE"] = 1
    out["ALPHA_ENTRY_OK"] = entry_ok.astype(int)

    # TOP_PICK 재정의 — 알파 게이트가 SSOT.
    out["TOP_PICK"] = entry_ok.astype(int)
    if "TOP_PICK_TYPE" not in out.columns:
        out["TOP_PICK_TYPE"] = ""
    out["TOP_PICK_TYPE"] = out["TOP_PICK_TYPE"].astype("object")
    out.loc[entry_ok, "TOP_PICK_TYPE"] = "ALPHA"
    out.loc[~entry_ok, "TOP_PICK_TYPE"] = ""

    # BUY_NOW_ELIGIBLE = 새 TOP_PICK AND BUY_NOW_PASS (컬럼 없으면 통과 취급).
    if "BUY_NOW_PASS" in out.columns:
        bnp = pd.to_numeric(out["BUY_NOW_PASS"], errors="coerce").fillna(0).astype(int) == 1
    else:
        bnp = pd.Series(True, index=out.index)
    out["BUY_NOW_ELIGIBLE"] = (entry_ok & bnp).astype(int)
    return out


def recompute_route_with_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """[v32.1] ROUTE 자체 치료 — 상태 판정을 데이터 방향으로 재계산.

    검증 통과(ALPHA_VALIDATED==1)일 때만 신규진입 상태(ATTACK/ARMED/WAIT/OVERHEAT)를
    재판정한다. 보유/청산/캐리 상태(_ROUTE_PRESERVE)는 그대로 보존.

      · OVERHEAT = 5일급등≥15% OR MFI≥85 OR RSI≥78  (실측 역신호 -2.16%p)
      · ATTACK   = 알파 ≥ 레짐문턱 AND MA20 위 AND 저점추세≥0 AND ~과열  (+3.70%p)
      · ARMED    = 알파 ≥ 문턱-15 AND (스퀴즈 OR MA20 위) AND ~과열       (+1.56%p)
      · WAIT     = 그 외

    미검증이면 ROUTE를 건드리지 않는다(기존 기술 판정 유지).
    ROUTE_REASON도 알파·구조 근거로 재작성.
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        return out
    validated = int(pd.to_numeric(out.get("ALPHA_VALIDATED", 0), errors="coerce")
                    .fillna(0).astype(int).max()) == 1
    if not validated or "ALPHA_SCORE" not in out.columns:
        return out

    num = lambda c: pd.to_numeric(out.get(c, np.nan), errors="coerce")
    ascore = num("ALPHA_SCORE")
    thr = _alpha_threshold_series(out)
    above = (num("Above_MA20") > 0).fillna(False)
    lowt = num("Low_Trend_PCT")
    ret5 = num("ret_5d_%")
    mfi = num("MFI14")
    rsi = num("RSI14")
    sq = (num("TTM_SQUEEZE") > 0).fillna(False)

    cur = out.get("ROUTE", pd.Series("", index=out.index)).astype(str).str.upper().str.strip()
    preserve = cur.isin(_ROUTE_PRESERVE)

    overheat = ((ret5 >= _ROUTE_OVERHEAT_RET5) | (mfi >= _ROUTE_OVERHEAT_MFI)
                | (rsi >= _ROUTE_OVERHEAT_RSI)).fillna(False)
    confirmed = above & (lowt.fillna(-99) >= 0)
    a_ok = ascore.notna()

    is_attack = a_ok & (ascore >= thr) & confirmed & (~overheat)
    is_armed = (~is_attack) & a_ok & (ascore >= (thr - _ROUTE_ARMED_ALPHA_GAP)) & (sq | above) & (~overheat)
    is_overheat = (~is_attack) & (~is_armed) & overheat

    new_route = pd.Series("WAIT", index=out.index, dtype="object")
    new_route[is_armed] = "ARMED"
    new_route[is_attack] = "ATTACK"
    new_route[is_overheat] = "OVERHEAT"
    # 보존 상태는 원래 값 유지.
    new_route[preserve] = out.loc[preserve, "ROUTE"].astype(str).values

    # 사유 문자열 (알파·구조 근거)
    reason = pd.Series("추세 관망", index=out.index, dtype="object")
    reason[is_armed] = "알파 상위권 + 코일(스퀴즈/추세) — 진입 임박"
    reason[is_attack] = "알파 상위 + MA20위·저점상승 확인 — 진입 우선"
    reason[is_overheat] = "과열(급등·MFI·RSI) — 실측 역신호, 관망"
    reason[preserve] = out.loc[preserve, "ROUTE_REASON"].astype(str).values if "ROUTE_REASON" in out.columns else ""

    # [v43] 위기 봉인 — 신규진입 차단 중에는 매수검토/진입대기 라벨 자체를 봉인.
    # 진단(7월 폭락, 코스피 -17%): 게이트(TOP_PICK)는 차단을 존중해 공식 추천
    # 0개인데, ROUTE 라벨은 차단을 무시하고 매수검토 10~23개를 계속 표시하는
    # 이중 신호 — 이 라벨 추종 실측 2주 -3.1%(7/23분 -7.7%).
    # 역사 실측: 차단일 ATTACK 승률 7%(비차단일 50%, 일별 t=-3.96), 같은 날
    # 유니버스 평균 대비도 열세(t=-2.37) — 절대·상대 가치 모두 없음.
    # 보존 라우트(CARRY/EXIT 등)는 건드리지 않고, 원 판정은 ROUTE_PRE_BLOCK에
    # 보관해 차단 해제 시 다음 배치에서 자동 복귀.
    def _blkflag(col):
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").fillna(0).astype(bool)
        return pd.Series(False, index=out.index)
    _blocked = (_blkflag("NEW_ENTRY_BLOCKED") | _blkflag("JULY_PROFIT_BLOCK_FLAG")
                | _blkflag("PROFIT_RECOVERY_BLOCK_FLAG")) & (~preserve)
    _sealed = _blocked & new_route.isin(["ATTACK", "ARMED"])
    out["ROUTE_PRE_BLOCK"] = new_route.where(_sealed, "")
    new_route = new_route.copy()
    new_route[_sealed] = "WAIT"
    reason[_sealed] = "🔒 신규진입 차단 중 — 매수검토 봉인 (방어 해제 시 자동 복귀)"

    out["ROUTE_PREV"] = out.get("ROUTE", "")
    out["ROUTE"] = new_route
    out["ROUTE_REASON"] = reason
    out["ROUTE_ALPHA_HEALED"] = 1
    # 상태 동기화 컬럼 (소비처 SSOT)
    _active = new_route.isin(["ATTACK", "ARMED"])
    out["IS_ACTIVE"] = _active
    out["IS_WATCH"] = new_route.eq("WAIT")
    return out
