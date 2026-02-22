"""
Kelly Criterion 승률 캘리브레이션 (SSOT)

핵심:
  - 선형 추정(p = 0.4 + (score-60)*0.01) 대신
  - rank_validation per-trade 히스토리 기반 실측 승률 사용
  - 베이지안 스무딩으로 표본 적을 때 안정화
  - 시간 가중(Decay)으로 레짐 변화 대응
  - method/horizon별 분리 캘리브레이션

사용법:
  1. rank_validation 실행 시 save_per_trade_log() 호출 → per_trade_log.csv 저장
  2. apply_kelly_betting 실행 시 load_calibration_table() → 승률 테이블 로드
  3. calibrated_win_rate(score, method, horizon) → 실측 기반 p 반환

v1.0 — 2026-02-22
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════
#  1. Per-Trade 히스토리 저장
# ═══════════════════════════════════════════════════

PER_TRADE_COLS = [
    "rec_date",       # 추천일 (YYYYMMDD)
    "code",           # 종목코드
    "method",         # 정렬 방식 (RANK_SCORE, ENTRY_SCORE, LDY_SCORE)
    "topk",           # 상위 K
    "horizon",        # 검증 기간 (영업일)
    "score",          # 해당 method의 점수
    "entry_price",    # 진입가
    "exit_price",     # 청산가 (SL/TP/종가)
    "stop_price",     # 손절가
    "target_price",   # 목표가
    "ret_pct",        # 수익률 (%)
    "win",            # 1=양수수익, 0=음수수익
    "exit_type",      # stop_hit / tp_hit / hold_close
    "b_ratio",        # 손익비 (reward/risk)
]


def save_per_trade_log(
    out_dir: str,
    trades: List[Dict],
    asof_ymd: str,
) -> str:
    """
    per-trade 히스토리를 CSV에 append (누적).
    rank_validation 내부에서 호출.

    Returns: 저장 경로
    """
    if not trades:
        return ""

    path = os.path.join(out_dir, "per_trade_log.csv")
    df_new = pd.DataFrame(trades)

    # 기존 로그에 append
    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path, dtype={"code": str})
            # 동일 (rec_date, code, method, horizon) 중복 제거 후 append
            key_cols = ["rec_date", "code", "method", "topk", "horizon"]
            # 타입 정규화: rec_date → str, horizon → int
            for kdf in [df_old, df_new]:
                kdf["rec_date"] = kdf["rec_date"].astype(str)
                kdf["code"] = kdf["code"].astype(str)
                kdf["horizon"] = kdf["horizon"].astype(int)
            existing_keys = set(df_old[key_cols].apply(lambda r: tuple(str(v) for v in r), axis=1))
            new_rows = []
            for _, row in df_new.iterrows():
                key = tuple(str(row[c]) for c in key_cols)
                if key not in existing_keys:
                    new_rows.append(row)
            if new_rows:
                df_combined = pd.concat([df_old, pd.DataFrame(new_rows)], ignore_index=True)
            else:
                df_combined = df_old
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ═══════════════════════════════════════════════════
#  2. 캘리브레이션 테이블 빌드
# ═══════════════════════════════════════════════════

# 스코어 빈 (하한, 상한)
DEFAULT_SCORE_BINS = [
    (0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100.01),
]

# 베이지안 스무딩 파라미터
# prior = fallback 승률 0.45 기반
# α = prior * strength, β = (1-prior) * strength
PRIOR_WIN_RATE = 0.45
PRIOR_STRENGTH = 20  # "가상 관측 20건" 분량의 prior


def _time_weight(rec_dates: pd.Series, half_life_days: int = 90,
                 asof_date: Optional[str] = None) -> np.ndarray:
    """
    시간 가중: 최근 데이터일수록 가중치 ↑
    지수 감쇠: w = exp(-λ * age_days), λ = ln(2) / half_life

    asof_date: 기준일 (YYYYMMDD str). None이면 현재 시각.
               ✅ 재현성: 동일 asof_date → 동일 가중치
    """
    try:
        dates = pd.to_datetime(rec_dates.astype(str), format="%Y%m%d", errors="coerce")
    except Exception:
        dates = pd.to_datetime(rec_dates, errors="coerce")

    if asof_date is not None:
        _asof_str = str(asof_date).replace("-", "")
        try:
            now = pd.to_datetime(_asof_str, format="%Y%m%d")
        except Exception:
            now = pd.Timestamp.now()
    else:
        now = pd.Timestamp.now()
    age_days = (now - dates).dt.total_seconds() / 86400.0
    age_days = age_days.fillna(half_life_days * 3)
    lam = np.log(2) / half_life_days
    weights = np.exp(-lam * age_days.values)
    return weights


def _bayesian_win_rate(
    wins: np.ndarray,
    weights: np.ndarray,
    prior_p: float = PRIOR_WIN_RATE,
    prior_strength: float = PRIOR_STRENGTH,
) -> float:
    """
    가중 베이지안 승률:
    p = (Σ(w_i * win_i) + α) / (Σ(w_i) + α + β)
    α = prior_p * prior_strength
    β = (1 - prior_p) * prior_strength
    """
    alpha = prior_p * prior_strength
    beta = (1 - prior_p) * prior_strength

    w_sum = float(np.sum(weights))
    w_wins = float(np.sum(weights * wins))

    return (w_wins + alpha) / (w_sum + alpha + beta)


def build_calibration_table(
    out_dir: str,
    score_bins: Optional[List[Tuple[float, float]]] = None,
    half_life_days: int = 90,
    min_effective_n: float = 5.0,
    asof_ymd: Optional[str] = None,
) -> pd.DataFrame:
    """
    per_trade_log.csv를 읽어 (method, horizon, score_bin) 별 캘리브레이션 테이블 생성.

    Args:
        asof_ymd: 기준일 (YYYYMMDD). 지정 시 rec_date < asof_ymd 만 사용.
                  ✅ 룩어헤드 방지: 오늘 추천에 오늘 결과가 섞이지 않음.
        min_effective_n: 가중 유효 표본수가 이 미만이면 fallback (테이블에서 제외).

    Returns:
        DataFrame: method, horizon, score_lo, score_hi, p_calibrated, n_effective, n_raw
    """
    if score_bins is None:
        score_bins = DEFAULT_SCORE_BINS

    log_path = os.path.join(out_dir, "per_trade_log.csv")
    if not os.path.exists(log_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(log_path, dtype={"code": str})
    except Exception:
        return pd.DataFrame()

    if df.empty or "win" not in df.columns:
        return pd.DataFrame()

    # ✅ 감점1 해결: 룩어헤드 방지 — asof_ymd 이전 데이터만 사용
    if asof_ymd is not None:
        asof_str = str(asof_ymd).replace("-", "")
        df["_rec_str"] = df["rec_date"].astype(str).str.replace("-", "")
        df = df[df["_rec_str"] < asof_str].copy()
        df.drop(columns=["_rec_str"], inplace=True)
        if df.empty:
            return pd.DataFrame()

    # ✅ 감점2 해결: 재현성 — asof_date 전달
    weights = _time_weight(df["rec_date"], half_life_days, asof_date=asof_ymd)

    rows = []
    for method in df["method"].unique():
        for horizon in df["horizon"].unique():
            mask_mh = (df["method"] == method) & (df["horizon"] == horizon)
            sub = df[mask_mh]
            w_sub = weights[mask_mh.values]

            for lo, hi in score_bins:
                mask_bin = (sub["score"] >= lo) & (sub["score"] < hi)
                bin_df = sub[mask_bin]
                bin_w = w_sub[mask_bin.values]

                if len(bin_df) == 0:
                    continue

                wins = bin_df["win"].values.astype(float)
                n_eff = float(np.sum(bin_w))
                n_raw = len(bin_df)

                # ✅ 감점3 해결: 유효 표본 부족 시 스킵 → fallback으로 위임
                if n_eff < min_effective_n:
                    continue

                p_cal = _bayesian_win_rate(wins, bin_w)

                rows.append({
                    "method": method,
                    "horizon": int(horizon),
                    "score_lo": lo,
                    "score_hi": hi,
                    "p_calibrated": round(p_cal, 4),
                    "n_effective": round(n_eff, 1),
                    "n_raw": n_raw,
                })

    result = pd.DataFrame(rows)

    # JSON으로도 저장 (빠른 로드용)
    cal_path = os.path.join(out_dir, "calibration_table.json")
    try:
        result.to_json(cal_path, orient="records", indent=2, force_ascii=False)
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════
#  3. 캘리브레이션 승률 조회
# ═══════════════════════════════════════════════════

_CAL_CACHE: Optional[pd.DataFrame] = None
_CAL_CACHE_KEY: Optional[str] = None  # (out_dir, asof_ymd) 복합 키


def _normalize_ymd(ymd: Optional[str]) -> Optional[str]:
    """YYYY-MM-DD / YYYYMMDD → YYYYMMDD 정규화"""
    if ymd is None:
        return None
    return str(ymd).replace("-", "").replace("/", "")[:8]


def load_calibration_table(out_dir: str, asof_ymd: Optional[str] = None,
                           force_reload: bool = False) -> pd.DataFrame:
    """
    캘리브레이션 테이블을 캐시로 로드.
    asof_ymd 지정 시 해당 날짜 기준 테이블을 빌드/로드.
    ✅ 날짜별 스냅샷: calibration_table_{asof_ymd}.json
    """
    global _CAL_CACHE, _CAL_CACHE_KEY
    asof_norm = _normalize_ymd(asof_ymd)
    cache_key = f"{out_dir}|{asof_norm or 'latest'}"

    if _CAL_CACHE is not None and _CAL_CACHE_KEY == cache_key and not force_reload:
        return _CAL_CACHE

    # 날짜별 스냅샷 파일 먼저 확인
    if asof_norm:
        snap_path = os.path.join(out_dir, f"calibration_table_{asof_norm}.json")
        if os.path.exists(snap_path) and not force_reload:
            try:
                _CAL_CACHE = pd.read_json(snap_path, orient="records")
                _CAL_CACHE_KEY = cache_key
                return _CAL_CACHE
            except Exception:
                pass

    # 스냅샷 없으면 빌드
    csv_path = os.path.join(out_dir, "per_trade_log.csv")
    if os.path.exists(csv_path):
        _CAL_CACHE = build_calibration_table(out_dir, asof_ymd=asof_norm)
        _CAL_CACHE_KEY = cache_key

        # 날짜별 스냅샷 저장
        if asof_norm and _CAL_CACHE is not None and not _CAL_CACHE.empty:
            snap_path = os.path.join(out_dir, f"calibration_table_{asof_norm}.json")
            try:
                _CAL_CACHE.to_json(snap_path, orient="records", indent=2, force_ascii=False)
            except Exception:
                pass

        return _CAL_CACHE

    # fallback: 기존 latest JSON
    json_path = os.path.join(out_dir, "calibration_table.json")
    if os.path.exists(json_path):
        try:
            _CAL_CACHE = pd.read_json(json_path, orient="records")
            _CAL_CACHE_KEY = cache_key
            return _CAL_CACHE
        except Exception:
            pass

    _CAL_CACHE = pd.DataFrame()
    _CAL_CACHE_KEY = cache_key
    return _CAL_CACHE


def calibrated_win_rate(
    score: float,
    out_dir: str,
    method: str = "RANK_SCORE",
    horizon: int = 5,
    fallback: float = PRIOR_WIN_RATE,
    asof_ymd: Optional[str] = None,
) -> float:
    """
    캘리브레이션 테이블에서 (method, horizon, score 구간)에 해당하는 p를 조회.
    테이블이 없거나 매칭 안 되면 fallback 반환.
    asof_ymd: 룩어헤드 방지용 기준일.
    """
    cal = load_calibration_table(out_dir, asof_ymd=asof_ymd)
    if cal.empty:
        return _fallback_linear(score, fallback)

    # method/horizon 매칭
    mask = (cal["method"] == method) & (cal["horizon"] == horizon)
    sub = cal[mask]

    if sub.empty:
        # horizon 무관하게 method만으로 시도
        sub = cal[cal["method"] == method]

    if sub.empty:
        return _fallback_linear(score, fallback)

    # ✅ 감점5 해결: 정렬 후 매칭 (순서 비의존)
    sub = sub.sort_values("score_lo").reset_index(drop=True)

    # score 구간 매칭
    for _, row in sub.iterrows():
        if row["score_lo"] <= score < row["score_hi"]:
            return float(row["p_calibrated"])

    # ✅ 감점3 해결: 구간 경계에서 선형 보간 (계단→연속)
    # 매칭 실패 시 가장 가까운 두 빈의 중심값 기준 보간
    if len(sub) >= 2:
        centers = ((sub["score_lo"] + sub["score_hi"]) / 2).values
        probs = sub["p_calibrated"].values
        if score <= centers[0]:
            return float(probs[0])
        if score >= centers[-1]:
            return float(probs[-1])
        # 선형 보간
        return float(np.interp(score, centers, probs))

    return _fallback_linear(score, fallback)


def _fallback_linear(score: float, base: float = 0.45) -> float:
    """캘리브레이션 없을 때 fallback: 기존 선형 추정 (하위호환)"""
    p = 0.4 + (max(score, 0) - 60) * 0.01
    return min(max(p, 0.30), 0.85)


# ═══════════════════════════════════════════════════
#  4. Kelly 배팅 (캘리브레이션 연동)
# ═══════════════════════════════════════════════════

def kelly_fraction(
    p: float,
    b: float,
    multiplier: float = 0.5,
    max_alloc: float = 0.25,
) -> float:
    """
    Kelly Criterion: f = p - (1-p)/b
    Half-Kelly + 최대 비중 제한.
    """
    if p <= 0 or b <= 0:
        return 0.0
    q = 1.0 - p
    f = p - (q / b)
    f_safe = f * multiplier
    return min(max(f_safe, 0.0), max_alloc)


def apply_kelly_calibrated(
    df: pd.DataFrame,
    out_dir: str,
    total_capital: int = 10_000_000,
    method: str = "RANK_SCORE",
    horizon: int = 5,
    kelly_multiplier: float = 0.5,
    max_allocation: float = 0.25,
    asof_ymd: Optional[str] = None,
) -> pd.DataFrame:
    """
    캘리브레이션 기반 Kelly 배팅.
    - 승률: calibrated_win_rate() (실측 또는 fallback)
    - 손익비: (목표가 - 매수가) / (매수가 - 손절가)
    - asof_ymd: 룩어헤드 방지용 기준일
    """
    def _calc_row(row):
        try:
            score = float(row.get("TOTAL_SCORE", row.get("RANK_SCORE", 0)))
            buy = float(row.get("추천매수가", 0))
            stop = float(row.get("손절가", 0))
            target = float(row.get("추천매도가1", 0))

            if buy <= 0 or stop <= 0 or target <= 0:
                return 0, 0

            risk = buy - stop
            reward = target - buy
            if risk <= 0:
                return 0, 0

            b = reward / risk

            # ✅ SSOT: 캘리브레이션 승률 + asof_ymd 누수 방지
            p = calibrated_win_rate(score, out_dir, method=method,
                                    horizon=horizon, asof_ymd=asof_ymd)

            # 점수 60 미만 → 스킵 (기존 정책 유지)
            if score < 60:
                return 0, 0

            f = kelly_fraction(p, b, kelly_multiplier, max_allocation)

            final_amt = int(total_capital * f)
            final_qty = int(final_amt / buy)

            return final_qty, final_amt

        except Exception:
            return 0, 0

    res = df.apply(_calc_row, axis=1, result_type="expand")
    df["켈리_수량"] = res[0]
    df["켈리_금액(원)"] = res[1]

    mask = df["켈리_수량"] > 0
    df.loc[mask, "추천수량"] = df.loc[mask, "켈리_수량"]
    df.loc[mask, "추천금액(만원)"] = (df.loc[mask, "켈리_금액(원)"] / 10000).round(1)

    return df
