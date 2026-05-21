"""
services/backtest_regime.py
============================
[v3.9.19] 시장 국면별 성과 분석 — UI 비의존 로직 SSOT.

평가 로드맵 명시: Train/Test가 시간 분할 검증이라면, 국면별 성과는 조건 분할 검증.

질문 흐름:
  전체 기간에서 좋았나?        → 단일 백테스트
  프리셋별로 뭐가 나은가?      → 프리셋 비교
  주변 조합에서도 살아남나?    → 강건성 27조합
  최근 구간에서도 살아남나?    → Train/Test 분할
  하락장/냉각장에서도 버티나?  → 시장 국면별 성과 ★ 신규

설계:
- data/run_health_YYYYMMDD.json의 macro_risk (NORMAL/CAUTION/CRITICAL)를
  rec_date 기준으로 매칭
- 추천 데이터를 3개 국면별로 partition → 각각 _run_backtest
- 각 국면의 alpha + anomaly_flags + tp_saturation 첨부
- 비교 verdict (4단계):
    🟢 전천후    : 모든 국면 양호 (수익 양수 + MDD 안정 + alpha)
    🟡 국면 의존 : 활황장만 좋고 일부 국면 약함
    🔴 하락장 취약: CRITICAL/CAUTION에서 손실 또는 큰 MDD
    ⚪ 데이터 부족: 국면별 표본 부족

UI 비의존:
- nicegui import 0
- services 외 의존: components.tab_backtest._run_backtest (lazy)
                    components.backtest_verdict._calc_kospi_alpha (lazy)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

_logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# [v3.9.19] 시장 국면 임계값 — 모듈 상수
# ════════════════════════════════════════════════════════════════
# run_health 데이터 디렉토리
DEFAULT_RUN_HEALTH_DIR = "data"
RUN_HEALTH_PREFIX = "run_health_"

# 국면 분류 — macro_risk 3단계 (run_health가 SSOT)
REGIMES = ["NORMAL", "CAUTION", "CRITICAL"]

# 국면별 표본 최소
MIN_TRADES_PER_REGIME = 20

# 🟢 전천후 조건
REGIME_GREEN_POSITIVE_REGIMES = 3       # 3국면 모두 수익 양수
REGIME_GREEN_MIN_RET = 0.0              # 각 국면 수익 ≥ 0
REGIME_GREEN_MDD_MIN = -15.0            # 각 국면 MDD ≥ -15%

# [v3.9.19b 평가 1] alpha 평가 기준 — v3.9.17c / v3.9.18b 패턴 일관
# coverage가 부족하면 🟢 차단, 🟡 yellow_candidate
# 0.65 = 3국면 중 최소 2국면 alpha 필요 (2/3 = 66.67% ≥ 0.65)
REGIME_GREEN_ALPHA_COVERAGE = 0.65
REGIME_GREEN_ALPHA_MIN = 0.0             # 산출된 alpha는 모두 ≥ 0

# 🔴 하락장 취약 — CRITICAL/CAUTION에서 큰 손실
REGIME_RED_DOWNTURN_RET = -5.0          # CAUTION 또는 CRITICAL < -5% → 🔴
REGIME_RED_DOWNTURN_MDD = -20.0         # 또는 MDD < -20%


def load_macro_regime_map(
    data_dir: str = DEFAULT_RUN_HEALTH_DIR,
) -> dict:
    """run_health_YYYYMMDD.json 파일들에서 날짜별 macro_risk dict 생성.

    Returns:
        {"20260101": "NORMAL", "20260102": "CAUTION", ...}
    """
    regime_map = {}
    data_path = Path(data_dir)
    if not data_path.exists():
        _logger.debug(f"[regime] data dir 없음: {data_dir}")
        return regime_map

    for path in data_path.glob(f"{RUN_HEALTH_PREFIX}*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                health = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            _logger.debug(f"[regime] {path.name} 읽기 실패: {e}")
            continue

        # 파일명에서 날짜 추출 (run_health_20260101.json → 20260101)
        date_from_name = path.stem.replace(RUN_HEALTH_PREFIX, "")
        date_from_json = str(health.get("trade_ymd", "") or date_from_name)
        date_key = date_from_json or date_from_name

        macro_risk = health.get("macro_risk")
        if macro_risk in REGIMES:
            regime_map[date_key] = macro_risk

    return regime_map


def run_regime_split(
    all_recs: pd.DataFrame,
    preset_key: str = "balanced",
    data_dir: str = DEFAULT_RUN_HEALTH_DIR,
) -> dict:
    """[v3.9.19] 시장 국면별 성과 — NORMAL/CAUTION/CRITICAL 백테스트.

    Args:
        all_recs: 누적 추천 CSV (rec_date 필수)
        preset_key: 프리셋 이름
        data_dir: run_health JSON 디렉토리

    Returns:
        dict {
            "preset": str,
            "cfg": dict,
            "regime_map_info": {
                "n_dates_with_regime": int,
                "regime_dist": {"NORMAL": N, "CAUTION": N, "CRITICAL": N}
            },
            "regimes": {
                "NORMAL": {"n_recs": int, "result": dict 또는 error},
                "CAUTION": {...},
                "CRITICAL": {...}
            },
            "verdict": dict
        }
    """
    # lazy import — services는 components 모듈 top 의존 0
    from components.tab_backtest import PRESETS, _run_backtest  # ci-allow: layer-violation  # lazy import (top 의존 0)
    from components.backtest_verdict import _calc_kospi_alpha  # ci-allow: layer-violation  # lazy import (top 의존 0)
    from services.backtest_policy import (
        detect_anomaly_flags,
        tp_saturation_threshold,
    )

    if preset_key not in PRESETS:
        preset_key = "balanced"
    preset = PRESETS[preset_key]
    cfg = {
        "min_score": preset["min_score"],
        "top_k": preset["top_k"],
        "hold_days": preset["hold_days"],
        "target_pct": preset["target_pct"],
        "stop_pct": preset["stop_pct"],
        "cost_pct": preset["cost_pct"],
    }

    # ─── 1. 국면 맵 로드 ───
    regime_map = load_macro_regime_map(data_dir)
    if not regime_map:
        return {
            "preset": preset_key, "cfg": cfg,
            "regime_map_info": {"n_dates_with_regime": 0, "regime_dist": {}},
            "regimes": {r: {"n_recs": 0, "result": {"error": "국면 데이터 없음"}}
                        for r in REGIMES},
            "verdict": {
                "icon": "⚪",
                "level": "unknown",
                "title": "국면 데이터 없음",
                "color_class": "text-gray-400",
                "body": (
                    f"{data_dir}/run_health_*.json 파일이 없습니다. "
                    "관리자 또는 auto_collect 점검 권장."
                ),
                "reasons": ["run_health JSON 부재"],
            },
        }

    # ─── 2. 추천 데이터 검증 ───
    if all_recs.empty or "rec_date" not in all_recs.columns:
        return {
            "preset": preset_key, "cfg": cfg,
            "regime_map_info": {
                "n_dates_with_regime": len(regime_map),
                "regime_dist": _count_regime_dist(regime_map),
            },
            "regimes": {r: {"n_recs": 0, "result": {"error": "rec_date 없음"}}
                        for r in REGIMES},
            "verdict": {
                "icon": "⚪",
                "level": "unknown",
                "title": "추천 데이터 부족",
                "color_class": "text-gray-400",
                "body": "rec_date 컬럼이 없거나 데이터가 비어있음.",
                "reasons": ["추천 데이터 부재"],
            },
        }

    # ─── 3. 국면별 partition ───
    sorted_recs = all_recs.copy()
    # [v3.9.19b 평가 2] rec_date 정규화 — YYYYMMDD / YYYY-MM-DD / datetime 혼합 안전
    # 이전: astype(str)만 → "2026-05-15" 입력 시 "20260515"와 매칭 실패
    # 변경: pd.to_datetime 통해 항상 YYYYMMDD 형태로 표준화
    rec_date_dt = pd.to_datetime(
        sorted_recs["rec_date"], errors="coerce", format="mixed"
    )
    # NaT는 원본 문자열 fallback (날짜 파싱 실패 시 매칭 시도)
    rec_date_norm = rec_date_dt.dt.strftime("%Y%m%d")
    rec_date_str_raw = sorted_recs["rec_date"].astype(str)
    sorted_recs["rec_date_str"] = rec_date_norm.fillna(rec_date_str_raw)
    sorted_recs["regime"] = sorted_recs["rec_date_str"].map(regime_map)

    regimes_out = {}
    for regime in REGIMES:
        regime_recs = sorted_recs[sorted_recs["regime"] == regime].drop(
            columns=["rec_date_str", "regime"], errors="ignore"
        ).copy()
        n_recs = len(regime_recs)

        if n_recs < MIN_TRADES_PER_REGIME:
            regimes_out[regime] = {
                "n_recs": n_recs,
                "result": {
                    "error": f"표본 부족 ({n_recs}건 / 최소 {MIN_TRADES_PER_REGIME})"
                },
            }
            continue

        try:
            result = _run_backtest(
                regime_recs,
                cfg["min_score"], cfg["hold_days"],
                cfg["stop_pct"], cfg["target_pct"],
                cfg["top_k"], cfg["cost_pct"],
            )
        except Exception as e:
            _logger.warning(f"[regime] {regime} 백테스트 실패: {e}")
            result = {"error": f"백테스트 실패: {e}"}

        if "error" not in result:
            try:
                alpha, alpha_mode = _calc_kospi_alpha(result, cfg)
            except Exception as e:
                _logger.debug(f"[regime] {regime} alpha 실패: {e}")
                alpha, alpha_mode = None, None
            result["alpha"] = alpha
            result["alpha_mode"] = alpha_mode

            result["anomaly_flags"] = detect_anomaly_flags(
                total_ret=float(result.get("total_return", 0) or 0),
                sharpe_val=result.get("sharpe"),
                cagr_val=result.get("cagr"),
                trading_days=int(result.get("trading_days", 0) or 0),
            )

            sd = result.get("status_dist", {}) or {}
            n_win = int(sd.get("WIN", 0) or 0)
            n_stop = int(sd.get("STOP", 0) or 0)
            n_hold = int(sd.get("HOLD_EXIT", 0) or 0)
            n_total = n_win + n_stop + n_hold
            result["tp_saturation"] = (
                (n_win / n_total * 100) if n_total > 0 else 0.0
            )
            # STOP 비율 — 국면별 손절 빈도 (하락장 취약 판단 신호)
            result["stop_ratio"] = (
                (n_stop / n_total * 100) if n_total > 0 else 0.0
            )
            result["tp_threshold"] = tp_saturation_threshold(
                cfg["target_pct"]
            )

        regimes_out[regime] = {"n_recs": n_recs, "result": result}

    # ─── 4. verdict ───
    verdict = derive_regime_verdict(regimes_out)

    return {
        "preset": preset_key,
        "cfg": cfg,
        "regime_map_info": {
            "n_dates_with_regime": len(regime_map),
            "regime_dist": _count_regime_dist(regime_map),
        },
        "regimes": regimes_out,
        "verdict": verdict,
    }


def _count_regime_dist(regime_map: dict) -> dict:
    """{date: regime} → {regime: count}."""
    out = {r: 0 for r in REGIMES}
    for r in regime_map.values():
        if r in out:
            out[r] += 1
    return out


def derive_regime_verdict(regimes_out: dict) -> dict:
    """[v3.9.19] 국면별 성과 비교 판정 — 4단계.

    🟢 전천후    : 3국면 모두 수익 양수 + MDD 안정 + anomaly 없음
                    + alpha ≥ 0 (산출된 국면만)
    🟡 국면 의존 : NORMAL은 양호 but CAUTION/CRITICAL에서 약함 (손실은 아님)
    🔴 하락장 취약: CAUTION 또는 CRITICAL에서 -5% 이하 손실 OR MDD < -20%
    ⚪ 데이터 부족: 표본 부족으로 평가 불가 국면 ≥ 2개
    """
    # 평가 가능한 국면 카운트
    valid_regimes = {
        r: data for r, data in regimes_out.items()
        if "error" not in data["result"]
    }
    n_valid = len(valid_regimes)
    n_unknown = len(REGIMES) - n_valid

    if n_valid == 0 or n_unknown >= 2:
        # 평가 가능 국면 0~1개 — 데이터 부족
        return {
            "icon": "⚪",
            "level": "unknown",
            "title": "국면 표본 부족",
            "color_class": "text-gray-400",
            "body": (
                f"평가 가능 국면 {n_valid}개 / 3개. "
                "각 국면별 최소 거래수 확보 후 재시도."
            ),
            "reasons": [f"{n_unknown}개 국면 표본 부족"],
        }

    # ─── 🔴 하락장 취약 — CAUTION/CRITICAL에서 큰 손실 ───
    downturn_failures = []
    for r in ["CAUTION", "CRITICAL"]:
        if r not in valid_regimes:
            continue
        result = valid_regimes[r]["result"]
        ret = float(result.get("total_return", 0) or 0)
        mdd = float(result.get("mdd", 0) or 0)
        if ret < REGIME_RED_DOWNTURN_RET:
            downturn_failures.append(f"{r} 수익 {ret:+.1f}%")
        if mdd < REGIME_RED_DOWNTURN_MDD:
            downturn_failures.append(f"{r} MDD {mdd:.1f}%")

    if downturn_failures:
        return {
            "icon": "🔴",
            "level": "red",
            "title": "하락장 취약 · 실전 비권장",
            "color_class": "text-red-400",
            "body": (
                f"CAUTION/CRITICAL 국면에서 큰 손실 또는 낙폭. "
                f"{' · '.join(downturn_failures[:3])}. "
                "활황장에서만 작동하는 전략 — 실전 비권장."
            ),
            "reasons": downturn_failures[:3],
        }

    # ─── 🟢 전천후 또는 🟡 전천후 후보 ───
    all_positive = all(
        float(valid_regimes[r]["result"].get("total_return", 0) or 0)
        >= REGIME_GREEN_MIN_RET
        for r in valid_regimes
    )
    all_mdd_safe = all(
        float(valid_regimes[r]["result"].get("mdd", 0) or 0)
        >= REGIME_GREEN_MDD_MIN
        for r in valid_regimes
    )
    no_anomaly = all(
        not valid_regimes[r]["result"].get("anomaly_flags", [])
        for r in valid_regimes
    )
    # alpha — 산출된 국면만 모음
    alphas = [
        valid_regimes[r]["result"].get("alpha")
        for r in valid_regimes
    ]
    valid_alphas = [a for a in alphas if a is not None]
    alpha_coverage = (
        len(valid_alphas) / n_valid if n_valid > 0 else 0
    )
    alpha_all_nonneg = (
        len(valid_alphas) == 0
        or all(a >= REGIME_GREEN_ALPHA_MIN for a in valid_alphas)
    )

    basic_pass = (
        all_positive
        and all_mdd_safe
        and no_anomaly
        and alpha_all_nonneg
        and n_valid == 3
    )

    if basic_pass:
        # [v3.9.19b 평가 1] alpha coverage 분기 — v3.9.17c / v3.9.18b 패턴 일관
        # 시장 대비 검증 없으면 초록 확정 금지 → 🟡 전천후 후보
        if alpha_coverage >= REGIME_GREEN_ALPHA_COVERAGE:
            # 🟢 전천후 — coverage 충분 + alpha 양수
            alpha_clause = (
                f"alpha 평균 {sum(valid_alphas)/len(valid_alphas):+.2f}%p"
                f" (산출 {len(valid_alphas)}/{n_valid})"
            )
            return {
                "icon": "🟢",
                "level": "green",
                "title": "전천후 · 모든 국면 양호",
                "color_class": "text-emerald-400",
                "body": (
                    f"3국면 모두 수익 양수, MDD 안정, anomaly 없음. "
                    f"{alpha_clause}. 활황·정상·하락 모든 시장에서 살아남는 "
                    "전략 — 실전 후보."
                ),
                "reasons": ["전 국면 일관 양호"],
            }
        else:
            # 🟡 전천후 후보 · alpha 평가 보류 — coverage 부족
            return {
                "icon": "🟡",
                "level": "yellow_candidate",
                "title": "전천후 후보 · alpha 평가 보류",
                "color_class": "text-yellow-400",
                "body": (
                    f"3국면 모두 절대수익/MDD는 양호. 다만 alpha coverage "
                    f"{alpha_coverage*100:.0f}% (기준 65% 미달 — 산출 "
                    f"{len(valid_alphas)}/{n_valid}). KOSPI 대비 검증 부족 — "
                    "bench_cache 또는 kospi_daily.csv 보강 후 재평가 권장."
                ),
                "reasons": [
                    f"alpha coverage {alpha_coverage*100:.0f}% "
                    f"(기준 {REGIME_GREEN_ALPHA_COVERAGE*100:.0f}%)"
                ],
            }

    # ─── 🟡 국면 의존 (그 외) ───
    misses = []
    for r in REGIMES:
        if r not in valid_regimes:
            misses.append(f"{r} 표본 부족")
            continue
        result = valid_regimes[r]["result"]
        ret = float(result.get("total_return", 0) or 0)
        if ret < REGIME_GREEN_MIN_RET:
            misses.append(f"{r} 수익 음수 ({ret:+.1f}%)")
        elif ret < 3.0:
            misses.append(f"{r} 수익 낮음 ({ret:+.1f}%)")

    return {
        "icon": "🟡",
        "level": "yellow",
        "title": "국면 의존 · 일부 국면 약함",
        "color_class": "text-yellow-400",
        "body": (
            "일부 시장 국면에서 수익 양수지만 활황 대비 약함. "
            "macro_risk가 높은 구간 진입 시 주의 권장."
        ),
        "reasons": misses[:4] or ["일부 국면 미달"],
    }
