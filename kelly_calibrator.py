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
    # [v4.0] 세그먼트 캘리브레이션용 축 — recommend 컬럼명과 동일하게 둬서
    # build(로그)·lookup(recommend) 양쪽에서 같은 키로 매칭된다. (append-only)
    "MACRO_REGIME_MODE", "ACTION_TIER", "ROUTE", "TOP_PICK_TYPE",
]


# ═══════════════════════════════════════════════════
#  [v22] 5-method 확장 — winrate_table_by_{method} 분리 학습
#  ─ 참조: SwingPicker_v22_Final_Consolidated.md §2.2.5
# ═══════════════════════════════════════════════════
# 거래 1건 → method별 최대 5 row로 확장.
# 각 method는 같은 거래의 결과(win/ret)를 공유하되 score만 다른 축으로.
# 이로써 kelly_calibrator가 method별 독립 winrate_table 학습 가능.

METHOD_EXTRACTORS = [
    ("ELITE_SCORE",  lambda r: r.get("ELITE_SCORE")),
    ("RANK_SCORE",   lambda r: (r.get("DISPLAY_SCORE") 
                                or r.get("TOTAL_SCORE") 
                                or r.get("RANK_SCORE"))),
    ("AI_SCORE",     lambda r: r.get("AI_SCORE")),
    ("ROUTE_ARMED",  lambda r: r.get("ELITE_SCORE") 
                                if r.get("ROUTE") == "ARMED" else None),
    ("ROUTE_ATTACK", lambda r: r.get("ELITE_SCORE") 
                                if r.get("ROUTE") == "ATTACK" else None),
]


def _expand_trade_to_methods(trade: Dict) -> List[Dict]:
    """거래 1건 → method별 row 리스트.
    
    원본 score/method는 무시하고 METHOD_EXTRACTORS로 재생성.
    추출 결과가 None이거나 유효하지 않으면 해당 method row는 생성 안 함.
    """
    out = []
    for method_name, extractor in METHOD_EXTRACTORS:
        try:
            score = extractor(trade)
        except Exception:
            continue
        if score is None:
            continue
        try:
            score_f = float(score)
            if not np.isfinite(score_f):
                continue
        except (TypeError, ValueError):
            continue
        # 원본 trade 복제 + method/score 덮어쓰기
        expanded = dict(trade)
        expanded["method"] = method_name
        expanded["score"] = round(score_f, 2)
        out.append(expanded)
    return out


def expand_trade_to_method_rows(trade: Dict) -> List[Dict]:
    """외부 공개용 alias"""
    return _expand_trade_to_methods(trade)


def expand_trades_batch(trades: List[Dict]) -> List[Dict]:
    """여러 거래 → 전체 method별 row 리스트"""
    out = []
    for t in trades:
        out.extend(_expand_trade_to_methods(t))
    return out


def save_per_trade_log(
    out_dir: str,
    trades: List[Dict],
    asof_ymd: str,
) -> str:
    """[v22] per-trade 히스토리 저장 — 5-method 자동 확장.
    
    v22 변경:
      - 입력 trade가 ELITE_SCORE/DISPLAY_SCORE 등 다중 축 점수를 포함하면
        자동으로 method별 최대 5 row 확장 후 저장.
      - Backward-compat: method 컬럼이 이미 지정되어 있고 단일 축 점수만
        있으면 기존 동작 유지 (legacy caller 호환).

    v2.4 #1: 신규 행만 파일 끝에 append (O(k) I/O)
    v2.4 #2: filelock으로 멀티프로세스 CSV 충돌 방어
    """
    if not trades:
        return ""

    # [v22] 5-method 확장 가능 여부 판정 — 
    # ELITE_SCORE 또는 AI_SCORE 등 점수 축 컬럼이 보이면 auto-expand
    _first = trades[0]
    _auto_expand = any(
        _first.get(k) is not None
        for k in ["ELITE_SCORE", "AI_SCORE", "DISPLAY_SCORE", "TOTAL_SCORE"]
    )
    # method 컬럼이 이미 명시적으로 세팅된 legacy 호출은 그대로 보존
    _has_explicit_method = _first.get("method") in (
        "ELITE_SCORE", "RANK_SCORE", "AI_SCORE", "ROUTE_ARMED", "ROUTE_ATTACK"
    )
    if _auto_expand and not _has_explicit_method:
        trades = expand_trades_batch(trades)
        if not trades:
            return ""

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "per_trade_log.csv")
    df_new = pd.DataFrame(trades)

    unknown_cols = [c for c in df_new.columns if c not in PER_TRADE_COLS]
    if unknown_cols:
        _logger.debug(f"미지 컬럼 {unknown_cols} 삭제됨 (Strict Schema)")
    df_new = df_new.reindex(columns=PER_TRADE_COLS)

    if "rec_date" in df_new.columns:
        df_new["rec_date"] = df_new["rec_date"].astype(str)
    if "code" in df_new.columns:
        df_new["code"] = df_new["code"].astype(str)
    if "horizon" in df_new.columns:
        df_new["horizon"] = pd.to_numeric(df_new["horizon"], errors="coerce").fillna(5).astype(int)

    write_header = not os.path.exists(path)

    lock = _acquire_filelock(path)
    try:
        if lock:
            lock.acquire(timeout=10)
        # [v4.0] 기존 파일 헤더가 신규 스키마와 다르면 1회 재작성(구행 NaN 백필) →
        # 새 컬럼 append 시 정렬 어긋남 방지. 마이그레이션 후 헤더 재판정.
        _ensure_per_trade_schema(path)
        write_header = not os.path.exists(path)
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


def _ensure_per_trade_schema(path: str) -> None:
    """[v4.0] 기존 per_trade_log.csv 헤더를 현재 PER_TRADE_COLS로 1회 마이그레이션.

    구버전 파일은 신규 축 컬럼(MACRO_REGIME_MODE 등)이 없어 그대로 append하면
    컬럼 정렬이 어긋난다. 헤더가 다르면 전체를 reindex(신규컬럼 NaN)해서 재작성한다.
    멱등: 헤더가 이미 일치하면 아무것도 안 함.
    """
    if not os.path.exists(path):
        return
    try:
        old = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    except (pd.errors.EmptyDataError, OSError):
        return
    if list(old.columns) == list(PER_TRADE_COLS):
        return
    try:
        old.reindex(columns=PER_TRADE_COLS).to_csv(path, index=False, encoding="utf-8-sig")
        _logger.info("per_trade_log 스키마 마이그레이션 완료: %d→%d 컬럼",
                     len(old.columns), len(PER_TRADE_COLS))
    except OSError as e:
        _logger.warning("per_trade_log 스키마 마이그레이션 실패: %s", e)


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
        df = pd.read_csv(path, dtype={"code": str}, low_memory=False)
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
        # [v22] empirical base + softened slope (fallback 인자 무시 — base=None 전달)
        result = _fallback_linear(scores_arr, base_score=base_score,
                                  out_dir=out_dir, method=method)
    else:
        interp_data = _get_interp_arrays(cal, method, horizon)
        if interp_data is not None:
            centers, probs = interp_data
            result = np.interp(scores_arr, centers, probs)
        else:
            result = _fallback_linear(scores_arr, base_score=base_score,
                                      out_dir=out_dir, method=method)

    if is_scalar:
        return float(result[0])
    return result


# ── [v20.0] 캘리브레이션 성숙도 ──

MIN_EMPIRICAL_TRADES = 20   # 최소 20건 이상이어야 EMPIRICAL

def get_calibration_mode(out_dir: str, asof_ymd: Optional[str] = None) -> dict:
    """
    캘리브레이션 테이블 상태 진단
    
    [v22] n_trades는 method 확장 전 실거래 수 (drop_duplicates 기준).
    per_trade_log는 5-method 확장으로 1거래당 최대 5 row를 가지므로
    row 수를 그대로 쓰면 FALLBACK → MATURE 전환이 5배 빨라짐.

    Returns:
        {
            "mode": "NO_DATA" | "FALLBACK" | "LIGHT" | "MATURE",
            "n_trades": int,  # 유니크 거래 수 (method 확장 전)
            "n_rows": int,    # 실제 row 수 (method 확장 후)
            "n_bins": int,
            "table_date": str,
        }
    """
    result = {"mode": "NO_DATA", "n_trades": 0, "n_rows": 0,
              "n_bins": 0, "table_date": ""}

    # per_trade_log 건수 확인
    ptl_path = os.path.join(out_dir, "per_trade_log.csv")
    if os.path.exists(ptl_path):
        try:
            ptl = pd.read_csv(ptl_path)
            result["n_rows"] = len(ptl)
            # [v22] 유니크 거래 수 = (rec_date, code, topk, horizon) 조합
            # method 확장 전 실제 거래 단위. 일부 컬럼이 없는 레거시 로그도 관용.
            key_cols = [c for c in ["rec_date", "code", "topk", "horizon"]
                        if c in ptl.columns]
            if key_cols:
                result["n_trades"] = int(ptl.drop_duplicates(key_cols).shape[0])
            else:
                # key 컬럼 전무 — 보수적으로 row 수 사용
                result["n_trades"] = len(ptl)
        except Exception:
            pass

    # 캘리브레이션 테이블 확인
    cal = load_calibration_table(out_dir, asof_ymd=asof_ymd)
    if not cal.empty:
        result["n_bins"] = len(cal)

    # 모드 판정 — n_trades (유니크) 기준
    n = result["n_trades"]
    if n == 0:
        result["mode"] = "NO_DATA"
    elif n < MIN_EMPIRICAL_TRADES:
        result["mode"] = "FALLBACK"    # 데이터는 있지만 표본 부족
    elif n < 100:
        result["mode"] = "LIGHT"       # 참고 가능하나 신뢰구간 넓음
    else:
        result["mode"] = "MATURE"      # 통계적으로 유의미

    return result


# ═══════════════════════════════════════════════════
#  [v22] Empirical 기반 보수화 헬퍼
#  ─ 참조: SwingPicker_v22_Final_Consolidated.md §2.2.3, §2.2.4
# ═══════════════════════════════════════════════════

@lru_cache(maxsize=8)
def _get_empirical_base(out_dir: str, method: str = "RANK_SCORE",
                         target_bin_lo: float = 0.0, target_bin_hi: float = 50.0,
                         min_n: int = 100) -> Optional[float]:
    """winrate_table에서 기준 구간 (기본 0-50 bin)의 empirical p_win 조회.
    
    설계 §2.2.3: _fallback_linear의 base를 하드코딩 0.45 대신 실측 값으로.
    """
    # winrate_table (auto_backtest 생성) 우선
    for fname in [f"winrate_table_by_{method}_latest.json",
                  "winrate_table_latest.json"]:
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            table = data.get("table", data) if isinstance(data, dict) else data
            if not isinstance(table, list):
                continue
            for row in table:
                lo = row.get("score_lo")
                hi = row.get("score_hi")
                n = row.get("n_raw", 0)
                p_win = row.get("p_win")
                if (lo == target_bin_lo and hi == target_bin_hi
                    and n >= min_n and p_win is not None):
                    return float(p_win)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            continue
    return None


def _fallback_linear(score, base: Optional[float] = None,
                     base_score: float = 60.0,
                     out_dir: Optional[str] = None,
                     method: str = "RANK_SCORE"):
    """[v22] 유니버설 fallback — empirical base 동적 조회 + slope 완화
    
    설계 결정:
      - base=None이면 _get_empirical_base로 조회 (실측), 없으면 0.45 하드폴백
      - slope 0.01 → 0.008 (과격한 선형 증가 완화)
      - 상한 0.85 → 0.75 (fallback이 MATURE처럼 낙관 표시되는 과대추정 방지)
    
    공식: p = (base - 0.05) + (max(score, 0) - base_score) * 0.008
    하한/상한: [0.30, 0.75]
    """
    if base is None:
        if out_dir is not None:
            empirical = _get_empirical_base(out_dir, method=method)
            base = empirical if empirical is not None else 0.45
        else:
            base = 0.45
    p = (base - 0.05) + (np.maximum(score, 0.0) - base_score) * 0.008
    return np.clip(p, 0.30, 0.75)


def _get_winrate_table_mtime(out_dir: str, method: str = "RANK_SCORE") -> int:
    """가장 최신 winrate_table 파일의 mtime (캐시 키)"""
    mtimes = []
    for fname in [f"winrate_table_by_{method}_latest.json",
                  "winrate_table_latest.json"]:
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            try:
                mtimes.append(int(os.path.getmtime(p)))
            except OSError:
                pass
    return max(mtimes) if mtimes else 0


@lru_cache(maxsize=16)
def _load_winrate_table_impl(out_dir: str, method: str,
                               _mtime_key: int) -> Optional[pd.DataFrame]:
    """winrate_table 캐시 로드 (내부 impl). mtime_key는 cache invalidation용."""
    for fname in [f"winrate_table_by_{method}_latest.json",
                  "winrate_table_latest.json"]:
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            table = data.get("table", data) if isinstance(data, dict) else data
            if isinstance(table, list) and len(table) > 0:
                return pd.DataFrame(table)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _load_winrate_table_cached(out_dir: str,
                                 method: str = "RANK_SCORE") -> Optional[pd.DataFrame]:
    """winrate_table 캐시 로드 — mtime 기반 자동 무효화."""
    mtime_key = _get_winrate_table_mtime(out_dir, method)
    return _load_winrate_table_impl(out_dir, method, mtime_key)


# 외부에서 `_load_winrate_table_cached.cache_clear()` 호출 가능하도록 proxy 부여
_load_winrate_table_cached.cache_clear = _load_winrate_table_impl.cache_clear
_load_winrate_table_cached.cache_info = _load_winrate_table_impl.cache_info


@lru_cache(maxsize=16)
def _load_winrate_meta_impl(out_dir: str, method: str,
                              _mtime_key: int) -> Optional[Dict]:
    """[v22 v5] winrate_table meta 로드 — entry_rule 신뢰도 검증용"""
    for fname in [f"winrate_table_by_{method}_latest.json",
                  "winrate_table_latest.json"]:
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "meta" in data:
                return data["meta"]
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _load_winrate_meta_cached(out_dir: str,
                                method: str = "RANK_SCORE") -> Optional[Dict]:
    mtime_key = _get_winrate_table_mtime(out_dir, method)
    return _load_winrate_meta_impl(out_dir, method, mtime_key)


_load_winrate_meta_cached.cache_clear = _load_winrate_meta_impl.cache_clear


def _is_winrate_table_trustworthy(out_dir: str, method: str = "RANK_SCORE") -> bool:
    """[v22 v5] winrate_table의 entry_rule이 신뢰할 만한지 판정.
    
    True 조건:
      - meta 자체 없음 (옛날 테이블) → True (보수적으로 신뢰, 다른 가드 작동)
      - entry_rule_trustworthy == True (all 비중 < 5%)
      - entry_rule_all_ratio 정보 없으면 (옛날 테이블) → True
    
    False 조건:
      - meta.entry_rule_trustworthy == False
      - meta.entry_rule_all_ratio >= 0.05
    """
    meta = _load_winrate_meta_cached(out_dir, method=method)
    if not meta:
        return True   # 메타 없으면 옛날 테이블 — 보수적으로 신뢰 (다른 sufficient 가드 작동)
    
    # 명시적 trustworthy 플래그
    if "entry_rule_trustworthy" in meta:
        return bool(meta["entry_rule_trustworthy"])
    
    # all 비중으로 추정
    if "entry_rule_all_ratio" in meta:
        return float(meta["entry_rule_all_ratio"]) < 0.05
    
    return True   # 정보 없으면 신뢰


def _get_empirical_b(scores, out_dir: str, method: str = "RANK_SCORE",
                      min_n: int = 200) -> Optional[np.ndarray]:
    """점수별 empirical b_ratio 보간.
    
    설계 §2.2.4: Kelly 과대 배팅 방지.
    planned_b(선언)와 empirical_b(실측) 중 min 취함.
    
    [v22 v5] entry_rule 신뢰도 가드:
      winrate_table의 'all' fallback 비중이 5%+면 학습 데이터에 비활성 ROUTE
      종목이 섞였다는 뜻 → empirical_b 미사용 (None 반환).
      그러면 apply_kelly_calibrated가 planned_b * 0.6으로 보수화.
    """
    # [v22 v5] entry_rule 신뢰도 검증
    if not _is_winrate_table_trustworthy(out_dir, method=method):
        _logger.warning(
            f"⚠️ [v22] winrate_table_by_{method} entry_rule 신뢰도 부족 "
            f"(all fallback 비중 ≥5%) → empirical_b 미사용, planned×0.6 보수화"
        )
        return None
    
    wt = _load_winrate_table_cached(out_dir, method=method)
    if wt is None or wt.empty:
        return None
    
    if "n_raw" in wt.columns:
        valid = wt[wt["n_raw"] >= min_n].copy()
    else:
        valid = wt.copy()
    if len(valid) < 2:
        return None
    if "b_ratio" not in valid.columns or "score_lo" not in valid.columns:
        return None
    
    centers = ((valid["score_lo"] + valid["score_hi"]) / 2).values.astype(float)
    b_vals = valid["b_ratio"].values.astype(float)
    order = np.argsort(centers)
    centers = centers[order]
    b_vals = b_vals[order]
    
    scores_arr = np.atleast_1d(np.asarray(scores, dtype=float))
    return np.interp(scores_arr, centers, b_vals)


def _clip_est_win_rate_to_realized_bins(
    scores,
    win_rates,
    out_dir: str,
    method: str = "ELITE_SCORE",
    max_gap: float = 0.145,
):
    """[v22.3.10b] EST_WIN_RATE 과신 방지 캡.

    monotonicity CI는 TOP_PICK의 선언 승률이 같은 ELITE_SCORE 구간
    실현 승률보다 15%p 초과 높으면 실패시킨다. 이 함수는 실현
    winrate_table의 sufficient bin이 있을 때 표시/켈리용 EST_WIN_RATE를
    `p_win + max_gap`로 보수 캡핑한다.

    BUY_NOW_ELIGIBLE / TOP_PICK / 점수 산식은 변경하지 않는다.
    """
    wt = _load_winrate_table_cached(out_dir, method=method)
    if wt is None or wt.empty:
        return np.asarray(win_rates, dtype=float), False

    required = {"score_lo", "score_hi", "p_win"}
    if not required.issubset(set(wt.columns)):
        return np.asarray(win_rates, dtype=float), False

    scores_arr = np.asarray(scores, dtype=float)
    wr_arr = np.asarray(win_rates, dtype=float).copy()
    clipped = False

    for _, row in wt.iterrows():
        try:
            lo = float(row.get("score_lo"))
            hi = float(row.get("score_hi"))
            p_win = float(row.get("p_win"))
        except (TypeError, ValueError):
            continue

        if not np.isfinite(p_win):
            continue

        # sufficient 컬럼이 있으면 sufficient=True bin만 신뢰
        if "sufficient" in wt.columns and not bool(row.get("sufficient")):
            continue

        n_raw = row.get("n_raw", None)
        try:
            if n_raw is not None and float(n_raw) < 30:
                continue
        except (TypeError, ValueError) as e:
            _logger.debug(f"ENTRY win-rate cap n_raw parse skipped: {e}")

        mask = (scores_arr >= lo) & (scores_arr < hi)
        if not mask.any():
            continue

        cap = min(0.85, max(0.30, p_win + max_gap))
        before = wr_arr[mask].copy()
        wr_arr[mask] = np.minimum(wr_arr[mask], cap)
        if np.any(wr_arr[mask] < before - 1e-12):
            clipped = True

    return wr_arr, clipped


# ═══════════════════════════════════════════════════
#  [v22] compute_est_win_rate — SSOT 함수
# ═══════════════════════════════════════════════════

def compute_est_win_rate(df: pd.DataFrame, out_dir: str,
                          asof_ymd: Optional[str] = None,
                          horizon: int = 5) -> pd.DataFrame:
    """ELITE_SCORE 기반 EST_WIN_RATE 계산 + v22 SSOT 메타 컬럼 주입.
    
    설계 §2.2.2: 랭킹 축 = 승률 추정 축 = ELITE_SCORE.
    
    입력 전제: df에 'ELITE_SCORE' 컬럼 존재 (compute_elite_score 후 호출).
    
    주입 컬럼:
      - EST_WIN_RATE
      - EST_WIN_RATE_METHOD: "ELITE_SCORE"
      - EST_WIN_RATE_MODE: "MATURE" | "LIGHT" | "FALLBACK" | "NO_DATA"
      - EST_WIN_RATE_N
    """
    if "ELITE_SCORE" not in df.columns:
        raise ValueError("compute_est_win_rate: 'ELITE_SCORE' 컬럼 필요 "
                        "(compute_elite_score 먼저 실행)")
    
    df = df.copy()
    mode_info = get_calibration_mode(out_dir, asof_ymd=asof_ymd)
    
    scores = pd.to_numeric(df["ELITE_SCORE"], errors="coerce").fillna(0).values
    wr = calibrated_win_rate(scores, out_dir, method="ELITE_SCORE",
                             horizon=horizon, asof_ymd=asof_ymd)
    wr = np.asarray(wr, dtype=float)

    # [v22.3.10b] 실현 승률 대비 과신 방지.
    # 표시/켈리용 EST_WIN_RATE만 보수 캡핑하고,
    # BUY_NOW_ELIGIBLE / TOP_PICK / 점수 산식은 변경하지 않는다.
    wr, _wr_clipped = _clip_est_win_rate_to_realized_bins(
        scores, wr, out_dir, method="ELITE_SCORE", max_gap=0.145
    )
    
    df["EST_WIN_RATE"] = np.round(wr, 3)
    df["EST_WIN_RATE_METHOD"] = "ELITE_SCORE"
    df["EST_WIN_RATE_MODE"] = mode_info["mode"]
    df["EST_WIN_RATE_N"] = mode_info["n_trades"]
    df["EST_WIN_RATE_REALIZED_CAP"] = bool(_wr_clipped)
    
    return df


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
    method: str = "ELITE_SCORE",
    horizon: int = 5,
    kelly_multiplier: float = 0.5,
    max_allocation: float = 0.25,
    min_score_threshold: float = 60.0,
    asof_ymd: Optional[str] = None,
) -> pd.DataFrame:
    """[v22] 벡터화 Kelly 배팅 + empirical b_ratio 병용
    
    v22 변경점:
      - 점수 축 ELITE_SCORE 우선 (설계 §2.2.4: 랭킹=승률=Kelly 축 일치)
      - 기본 method="ELITE_SCORE" (기존 RANK_SCORE 호환 fallback)
      - `_get_empirical_b`로 실측 b_ratio 보간, planned_b와 `min` 취해 과대배팅 방지
      - 관측 컬럼 4종 추가: KELLY_PLANNED_B/EMPIRICAL_B/FINAL_B/FRACTION
      - empirical 미가용 시 planned_b * 0.6 보수화
    """
    df = df.copy()

    # [v22] 점수 컬럼 우선순위: ELITE_SCORE > TOTAL_SCORE > RANK_SCORE
    score_col = None
    for _sc in ["ELITE_SCORE", "TOTAL_SCORE", "RANK_SCORE"]:
        if _sc in df.columns:
            score_col = _sc
            break
    if score_col is None:
        df["켈리_수량"] = 0
        df["켈리_금액(원)"] = 0
        return df

    # [v22] method 자동 동기화 — score_col과 calibration table을 일치시킴
    # 양방향 처리:
    # - method가 기본값(RANK_SCORE)이고 df에 ELITE_SCORE가 있으면 ELITE로 승격
    # - method가 ELITE_SCORE인데 실제 score_col은 다른 축이면 score_col에 맞춤
    if method == "RANK_SCORE" and score_col == "ELITE_SCORE":
        method = "ELITE_SCORE"
    elif method == "ELITE_SCORE" and score_col != "ELITE_SCORE":
        method = score_col

    def _safe_values(series: pd.Series, default: float = 0.0) -> np.ndarray:
        return pd.to_numeric(series, errors="coerce").fillna(default).values.astype(float)

    scores = _safe_values(df[score_col])
    buy = _safe_values(df.get("추천매수가", pd.Series(0, index=df.index)))
    stop = _safe_values(df.get("손절가", pd.Series(0, index=df.index)))
    target = _safe_values(df.get("추천매도가1", pd.Series(0, index=df.index)))

    # 승률 (유니버설 함수)
    p = calibrated_win_rate(scores, out_dir, method=method,
                            horizon=horizon, base_score=min_score_threshold,
                            asof_ymd=asof_ymd)
    p = np.asarray(p, dtype=float)

    # 손익비: planned (선언) vs empirical (실측)
    risk = buy - stop
    reward = target - buy
    planned_b = np.where(risk > 0, reward / risk, 0.0)

    empirical_b = _get_empirical_b(scores, out_dir, method=method)
    if empirical_b is not None:
        # [v22] min(planned, empirical) — 보수적 선택
        b_ratio = np.minimum(planned_b, empirical_b)
    else:
        # empirical 미가용: planned의 60%만 반영 (보수화)
        b_ratio = planned_b * 0.6

    # Kelly fraction
    q = 1.0 - p
    f_raw = np.where(b_ratio > 0, p - (q / b_ratio), 0.0)
    f_safe = np.clip(f_raw * kelly_multiplier, 0.0, max_allocation)

    valid = (scores >= min_score_threshold) & (buy > 0) & (stop > 0) & (target > 0) & (risk > 0)
    f_safe = np.where(valid, f_safe, 0.0)

    # [v23.0] GUARD 반영 — GUARD_KELLY_MULT(0~1)로 분율·손익비 축소
    # guard_system이 부여한 컬럼이 있으면 차단(0)·감점(<1)을 Kelly 사이징에 직접 반영.
    if "GUARD_KELLY_MULT" in df.columns:
        _gm = pd.to_numeric(df["GUARD_KELLY_MULT"], errors="coerce").fillna(1.0).clip(0.0, 1.0).values
        f_safe = f_safe * _gm
        b_ratio = b_ratio * _gm

    # [v22] 관측 컬럼
    df["KELLY_PLANNED_B"] = np.round(planned_b, 3)
    df["KELLY_EMPIRICAL_B"] = (
        np.round(empirical_b, 3) if empirical_b is not None
        else np.full(len(df), np.nan)
    )
    df["KELLY_FINAL_B"] = np.round(b_ratio, 3)
    df["KELLY_FRACTION"] = np.round(f_safe, 4)

    kelly_amt = (total_capital * f_safe).astype(int)
    kelly_qty = np.where(buy > 0, kelly_amt / buy, 0).astype(int)

    df["켈리_수량"] = kelly_qty
    df["켈리_금액(원)"] = kelly_amt

    mask_pos = kelly_qty > 0
    if mask_pos.any():
        df.loc[mask_pos, "추천수량"] = kelly_qty[mask_pos]
        df.loc[mask_pos, "추천금액(만원)"] = np.round(kelly_amt[mask_pos] / 10000, 1)

    return df
