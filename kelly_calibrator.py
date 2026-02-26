"""
Kelly Criterion 승률 캘리브레이션 (v2.0)
═══════════════════════════════════════════
[v2.0] 5건 수정:
  #1 보간법 자기모순: 구간 매칭 먼저 → 중심점 보간 먼저 (계단→연속)
  #2 미래 참조(Look-ahead): rec_date < asof → (rec_date + horizon) < asof
  #3 O(N²) apply: row별 calibrated_win_rate → pd.cut 벡터 병합
  #4 전역 _CAL_CACHE: global dict → functools.lru_cache 캡슐화
  #5 except Exception: pass → 명시적 예외 + logging
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  1. Per-Trade 히스토리 저장
# ═══════════════════════════════════════════════════

PER_TRADE_COLS = [
    "rec_date", "code", "method", "topk", "horizon",
    "score", "entry_price", "exit_price", "stop_price", "target_price",
    "ret_pct", "win", "exit_type", "b_ratio",
]


def save_per_trade_log(
    out_dir: str,
    trades: List[Dict],
    asof_ymd: str,
) -> str:
    """[v2.4] per-trade 히스토리 Append-only 저장

    v2.4 #1: 신규 행만 파일 끝에 append (O(k) I/O)
    v2.4 #2: filelock으로 멀티프로세스 CSV 충돌 방어

    Note: filelock 미설치 시 graceful fallback (락 없이 동작)
    """
    if not trades:
        return ""

    # [v2.4 #4] 디렉토리 미존재 시 자동 생성 (배포 첫날 방어)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "per_trade_log.csv")
    df_new = pd.DataFrame(trades)

    # [v3.1 #1] 절대 스키마 강제: reindex로 누락 컬럼 NaN 채움 + 순서 고정
    # Before: existing_cols 필터 → 컬럼 누락 시 열 수 불일치 → CSV 데이터 밀림
    # After:  reindex → 항상 PER_TRADE_COLS 14열 보장 + extra 컬럼 후미 배치
    extra_cols = [c for c in df_new.columns if c not in PER_TRADE_COLS]
    df_new = df_new.reindex(columns=PER_TRADE_COLS + extra_cols)

    # 컬럼 타입 정규화
    if "rec_date" in df_new.columns:
        df_new["rec_date"] = df_new["rec_date"].astype(str)
    if "code" in df_new.columns:
        df_new["code"] = df_new["code"].astype(str)
    if "horizon" in df_new.columns:
        df_new["horizon"] = pd.to_numeric(df_new["horizon"], errors="coerce").fillna(5).astype(int)

    write_header = not os.path.exists(path)

    # [v2.4 #2] 파일 락: 멀티프로세스 동시 쓰기 방어
    lock = _acquire_filelock(path)
    try:
        if lock:
            lock.acquire(timeout=10)
        df_new.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8-sig")
    except Exception as e:
        _logger.error(f"트레이드 로그 저장 실패: {e}")
    finally:
        if lock:
            try:
                lock.release()
            except Exception:
                pass

    return path


def _acquire_filelock(path: str):
    """filelock 라이브러리 존재 시 FileLock 반환, 없으면 None (graceful)"""
    try:
        from filelock import FileLock
        return FileLock(path + ".lock", timeout=10)
    except ImportError:
        return None


# ── Dedup-on-load: 파일 읽을 때 중복 제거 ──
_TRADE_KEY_COLS = ["rec_date", "code", "method", "topk", "horizon"]


def load_per_trade_log(out_dir: str) -> pd.DataFrame:
    """[v2.4 #1] 트레이드 로그 로드 + 중복 제거 (Read 시 1회)

    Append-only 파일이므로 중복 가능 → load 시 drop_duplicates
    """
    path = os.path.join(out_dir, "per_trade_log.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=PER_TRADE_COLS)

    try:
        df = pd.read_csv(path, dtype={"code": str})
    except (pd.errors.EmptyDataError, OSError) as e:
        _logger.warning(f"트레이드 로그 읽기 실패: {e}")
        return pd.DataFrame(columns=PER_TRADE_COLS)

    # 키 컬럼 정규화 + dedup
    for col in _TRADE_KEY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str) if col != "horizon" else \
                pd.to_numeric(df[col], errors="coerce").fillna(5).astype(int)

    existing_keys = [c for c in _TRADE_KEY_COLS if c in df.columns]
    if existing_keys:
        df = df.drop_duplicates(subset=existing_keys, keep="last")

    return df


# ═══════════════════════════════════════════════════
#  2. 캘리브레이션 테이블 빌드
# ═══════════════════════════════════════════════════

DEFAULT_SCORE_BINS = [
    (0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100.01),
]

PRIOR_WIN_RATE = 0.45
PRIOR_STRENGTH = 20


def _time_weight(rec_dates: pd.Series, half_life_days: int = 90,
                 asof_date: Optional[str] = None) -> np.ndarray:
    """시간 가중: 최근 데이터일수록 가중치 ↑ (지수 감쇠)"""
    try:
        dates = pd.to_datetime(rec_dates.astype(str), format="%Y%m%d", errors="coerce")
    except ValueError:
        dates = pd.to_datetime(rec_dates, errors="coerce")

    if asof_date is not None:
        _asof = str(asof_date).replace("-", "")
        try:
            now = pd.to_datetime(_asof, format="%Y%m%d")
        except ValueError:
            now = pd.Timestamp.now()
    else:
        now = pd.Timestamp.now()

    age_days = (now - dates).dt.total_seconds() / 86400.0
    age_days = age_days.fillna(half_life_days * 3)
    lam = np.log(2) / half_life_days
    return np.exp(-lam * age_days.values)


def _bayesian_win_rate(
    wins: np.ndarray,
    weights: np.ndarray,
    prior_p: float = PRIOR_WIN_RATE,
    prior_strength: float = PRIOR_STRENGTH,
) -> float:
    """가중 베이지안 승률"""
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
    """[v2.0 #2] 미래 참조 방지 — 청산 완료일 기준 필터링

    Before: rec_date < asof_ymd (추천일 기준 → 미청산 트레이드 포함 = 미래 참조)
    After:  (rec_date + horizon 영업일) < asof_ymd (청산 완료된 트레이드만)
    """
    if score_bins is None:
        score_bins = DEFAULT_SCORE_BINS

    # [v2.4 #2] load_per_trade_log 재사용 (중복 제거된 데이터로 빌드)
    df = load_per_trade_log(out_dir)
    if df.empty or "win" not in df.columns:
        return pd.DataFrame()

    # [v2.2 #1] 청산 완료일: np.busday_offset 벡터화 (for 루프 제거)
    if asof_ymd is not None:
        asof_str = str(asof_ymd).replace("-", "")
        try:
            asof_dt = pd.to_datetime(asof_str, format="%Y%m%d")
        except ValueError:
            asof_dt = pd.Timestamp.now()

        rec_dt = pd.to_datetime(df["rec_date"].astype(str), format="%Y%m%d", errors="coerce")

        # [v2.4 #1] NaT 방어: busday_offset은 NaT 입력 시 ValueError 즉사
        nat_mask = rec_dt.isna()
        if nat_mask.any():
            _logger.warning(f"rec_date 파싱 불가 {nat_mask.sum()}건 제거 (NaT 방어)")
            df = df[~nat_mask].copy()
            rec_dt = rec_dt[~nat_mask]

        horizon_days = df["horizon"].fillna(5).astype(int)

        # numpy datetime64[D]로 변환 → busday_offset 벡터 연산 (C 속도)
        rec_np = rec_dt.values.astype("datetime64[D]")
        exit_np = np.busday_offset(rec_np, horizon_days.values, roll="forward")
        exit_dt = pd.Series(pd.to_datetime(exit_np), index=df.index)

        df = df[exit_dt < asof_dt].copy()

        if df.empty:
            return pd.DataFrame()

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

                if n_eff < min_effective_n:
                    continue

                p_cal = _bayesian_win_rate(wins, bin_w)

                rows.append({
                    "method": method,
                    "horizon": int(horizon),
                    "score_lo": lo,
                    "score_hi": hi,
                    "score_center": (lo + hi) / 2,  # [v2.0 #1] 보간용 중심점
                    "p_calibrated": round(p_cal, 4),
                    "n_effective": round(n_eff, 1),
                    "n_raw": n_raw,
                })

    result = pd.DataFrame(rows)

    # JSON 저장
    cal_path = os.path.join(out_dir, "calibration_table.json")
    try:
        result.to_json(cal_path, orient="records", indent=2, force_ascii=False)
    except OSError as e:
        _logger.warning(f"캘리브레이션 JSON 저장 실패: {e}")

    return result


# ═══════════════════════════════════════════════════
#  3. 캘리브레이션 승률 조회
# ═══════════════════════════════════════════════════

def _normalize_ymd(ymd: Optional[str]) -> Optional[str]:
    if ymd is None:
        return None
    return str(ymd).replace("-", "").replace("/", "")[:8]


def _get_csv_mtime(out_dir: str) -> int:
    """CSV 파일의 mtime을 초 단위 정수로 반환 (캐시 키용)"""
    csv_path = os.path.join(out_dir, "per_trade_log.csv")
    try:
        return int(os.path.getmtime(csv_path))
    except OSError:
        return 0


@lru_cache(maxsize=32)
def _load_cal_cached(out_dir: str, asof_norm: Optional[str],
                     _mtime: int = 0) -> Optional[Tuple]:
    """[v2.5 #1] 캘리브레이션 테이블 캐시 — mtime 기반 자동 무효화

    Before: 키 = (out_dir, asof_norm) → CSV 갱신돼도 캐시 갱신 안 됨
    After:  키 = (out_dir, asof_norm, mtime) → 파일 수정 시 자동 캐시 미스
    """
    # 날짜별 스냅샷 읽기 — CSV보다 최신일 때만 유효
    if asof_norm:
        snap_path = os.path.join(out_dir, f"calibration_table_{asof_norm}.json")
        if os.path.exists(snap_path):
            try:
                snap_mtime = int(os.path.getmtime(snap_path))
                # [v3.1 #2] 좀비 스냅샷 방어: CSV가 스냅샷보다 새로우면 스냅샷 무시
                if snap_mtime >= _mtime:
                    df = pd.read_json(snap_path, orient="records")
                    if not df.empty:
                        return tuple(df.to_dict("records"))
                else:
                    _logger.info(f"스냅샷 무효화: CSV({_mtime}) > snap({snap_mtime}), 재빌드")
            except (OSError, ValueError):
                pass

    # 빌드
    csv_path = os.path.join(out_dir, "per_trade_log.csv")
    if os.path.exists(csv_path):
        cal_df = build_calibration_table(out_dir, asof_ymd=asof_norm)
        if cal_df is not None and not cal_df.empty:
            return tuple(cal_df.to_dict("records"))

    # fallback latest
    json_path = os.path.join(out_dir, "calibration_table.json")
    if os.path.exists(json_path):
        try:
            df = pd.read_json(json_path, orient="records")
            if not df.empty:
                return tuple(df.to_dict("records"))
        except (OSError, ValueError):
            pass

    return None


def load_calibration_table(out_dir: str, asof_ymd: Optional[str] = None,
                           force_reload: bool = False) -> pd.DataFrame:
    """[v2.5 #1] 캘리브레이션 테이블 로드 — mtime 기반 캐시 무효화"""
    asof_norm = _normalize_ymd(asof_ymd)

    if force_reload:
        _load_cal_cached.cache_clear()

    mtime = _get_csv_mtime(out_dir)
    records = _load_cal_cached(out_dir, asof_norm, _mtime=mtime)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(list(records))

    # 스냅샷 저장: CSV보다 오래된 스냅샷이거나 미존재 시 갱신
    if asof_norm and not df.empty:
        snap_path = os.path.join(out_dir, f"calibration_table_{asof_norm}.json")
        need_write = not os.path.exists(snap_path)
        if not need_write:
            try:
                need_write = int(os.path.getmtime(snap_path)) < mtime
            except OSError:
                need_write = True
        if need_write:
            try:
                df.to_json(snap_path, orient="records", indent=2, force_ascii=False)
            except OSError as e:
                _logger.warning(f"스냅샷 저장 실패: {e}")

    return df


def _get_interp_arrays(cal: pd.DataFrame, method: str, horizon: int
                       ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """캘리브레이션 테이블에서 (centers, probs) 보간 배열 추출 (DRY 공용)"""
    mask = (cal["method"] == method) & (cal["horizon"] == horizon)
    sub = cal[mask]
    if sub.empty:
        sub = cal[cal["method"] == method]
    if sub.empty or len(sub) < 2:
        if len(sub) == 1:
            # 빈 1개: 상수 보간용 길이 2 배열
            p = float(sub.iloc[0]["p_calibrated"])
            return np.array([0.0, 100.0]), np.array([p, p])
        return None
    sub = sub.sort_values("score_lo")
    if "score_center" in sub.columns:
        centers = sub["score_center"].values
    else:
        centers = ((sub["score_lo"] + sub["score_hi"]) / 2).values
    probs = sub["p_calibrated"].values
    return centers, probs


def calibrated_win_rate(
    score,
    out_dir: str,
    method: str = "RANK_SCORE",
    horizon: int = 5,
    fallback: float = PRIOR_WIN_RATE,
    base_score: float = 60.0,
    asof_ymd: Optional[str] = None,
):
    """[v2.4] 유니버설 승률 조회 — scalar/ndarray + base_score 전파

    base_score: fallback 수식 기준점 (모델 스케일에 맞춰 주입)
    """
    cal = load_calibration_table(out_dir, asof_ymd=asof_ymd)

    is_scalar = isinstance(score, (int, float, np.integer, np.floating))
    scores_arr = np.atleast_1d(np.asarray(score, dtype=float))

    if cal.empty:
        result = _fallback_linear(scores_arr, fallback, base_score=base_score)
    else:
        interp_data = _get_interp_arrays(cal, method, horizon)
        if interp_data is not None:
            centers, probs = interp_data
            result = np.interp(scores_arr, centers, probs)
        else:
            result = _fallback_linear(scores_arr, fallback, base_score=base_score)

    if is_scalar:
        return float(result[0])
    return result


def _fallback_linear(score, base: float = 0.45, base_score: float = 60.0):
    """[v2.5 #3] 유니버설 fallback — base 파라미터 실제 반영

    np.maximum, np.clip은 ufunc → 스칼라 입력이면 스칼라 반환.
    base_score: 기준 점수, base: 기준 점수에서의 사전 승률
    공식: p = (base - 0.05) + (max(score, 0) - base_score) * 0.01
    base=0.45 → base_score=60에서 p=0.40 (기존 호환)
    """
    p = (base - 0.05) + (np.maximum(score, 0.0) - base_score) * 0.01
    return np.clip(p, 0.30, 0.85)


# ═══════════════════════════════════════════════════
#  4. Kelly 배팅 (벡터화)
# ═══════════════════════════════════════════════════

def kelly_fraction(
    p: float,
    b: float,
    multiplier: float = 0.5,
    max_alloc: float = 0.25,
) -> float:
    """Kelly Criterion: f = p - (1-p)/b, Half-Kelly + cap"""
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
    min_score_threshold: float = 60.0,
    asof_ymd: Optional[str] = None,
) -> pd.DataFrame:
    """[v2.4] 벡터화 Kelly 배팅 — 프로덕션 안정성 강화

    v2.3: 순수 numpy 중간 연산
    v2.4 #2: min_score_threshold 파라미터화 (매직넘버 60 제거)
    v2.4 #3: pd.to_numeric(errors='coerce') 안전 캐스팅
             → "N/A", "-", "" 등 문자열 쓰레기 → NaN → 0 (ValueError 방지)
    """
    df = df.copy()

    # 점수 컬럼 결정
    score_col = "TOTAL_SCORE" if "TOTAL_SCORE" in df.columns else "RANK_SCORE"
    if score_col not in df.columns:
        df["켈리_수량"] = 0
        df["켈리_금액(원)"] = 0
        return df

    # [v2.4 #3] 안전 캐스팅: to_numeric(coerce) → 문자열 쓰레기 방어
    def _safe_values(series: pd.Series, default: float = 0.0) -> np.ndarray:
        return pd.to_numeric(series, errors="coerce").fillna(default).values.astype(float)

    scores = _safe_values(df[score_col])
    buy = _safe_values(df.get("추천매수가", pd.Series(0, index=df.index)))
    stop = _safe_values(df.get("손절가", pd.Series(0, index=df.index)))
    target = _safe_values(df.get("추천매도가1", pd.Series(0, index=df.index)))

    # ── 승률 (유니버설 함수, ndarray 반환) ──
    p = calibrated_win_rate(scores, out_dir, method=method,
                            horizon=horizon, base_score=min_score_threshold,
                            asof_ymd=asof_ymd)
    p = np.asarray(p, dtype=float)

    # ── 손익비 (순수 numpy) ──
    risk = buy - stop
    reward = target - buy
    b_ratio = np.where(risk > 0, reward / risk, 0.0)

    # ── Kelly fraction (순수 numpy) ──
    q = 1.0 - p
    f_raw = np.where(b_ratio > 0, p - (q / b_ratio), 0.0)
    f_safe = np.clip(f_raw * kelly_multiplier, 0.0, max_allocation)

    # [v2.4 #2] 유효 조건 필터 — 매직넘버 → 파라미터 주입
    valid = (scores >= min_score_threshold) & (buy > 0) & (stop > 0) & (target > 0) & (risk > 0)
    f_safe = np.where(valid, f_safe, 0.0)

    # ── 수량/금액 (순수 numpy) ──
    kelly_amt = (total_capital * f_safe).astype(int)
    kelly_qty = np.where(buy > 0, kelly_amt / buy, 0).astype(int)

    # ── 최종 대입: 여기서만 pandas 개입 ──
    df["켈리_수량"] = kelly_qty
    df["켈리_금액(원)"] = kelly_amt

    mask_pos = kelly_qty > 0
    if mask_pos.any():
        df.loc[mask_pos, "추천수량"] = kelly_qty[mask_pos]
        df.loc[mask_pos, "추천금액(만원)"] = np.round(kelly_amt[mask_pos] / 10000, 1)

    return df
