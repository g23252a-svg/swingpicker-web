"""
services/backtest_train_test.py
================================
[v3.9.18] Train/Test 분할 검증 — UI 비의존 로직 SSOT.

평가 v3.9.17 결정타: 과거 전체 좋아도 최근 OOS 구간에서 살아남는지 확인.
사용자 화면의 +712% / Sharpe 45 같은 비현실 결과는 거의 무조건 Train/Test로
쪼개봐야 함 — lookahead bias의 결정적 검증 도구.

설계:
- rec_date 기준 정렬 후 70/30 분할 (Train = 과거 70%, Test = 최근 30%)
- 각 구간에 _run_backtest 호출 (baseline 무수정)
- 각 구간의 alpha + anomaly_flags + tp_saturation 첨부
- 비교 verdict (4단계):
    🟢 일반화 양호 : Test도 수익 양수 + alpha 양수 + MDD 안정 + anomaly 없음
    🟡 약화        : Test 수익 양수지만 Train 대비 성능 하락
    🔴 과최적화 의심: Train 좋고 Test 손실 또는 alpha 음수
    🚨 lookahead 의심: Train/Test 모두 anomaly OR Test에서 급격 붕괴

UI 비의존:
- nicegui import 0
- services 외 의존: components.tab_backtest._run_backtest (lazy import 함수 내부)
                    components.backtest_verdict._calc_kospi_alpha (lazy)

호출처:
- components/backtest_train_test.py: UI 렌더 모듈에서 lazy import
- 향후 CLI/배치: services 모듈만 import해서 사용 가능
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

_logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# [v3.9.18] Train/Test 임계값 — 모듈 상수
# ════════════════════════════════════════════════════════════════
# 기본 분할 비율
DEFAULT_TEST_RATIO = 0.30   # Train 70% / Test 30%
MIN_RECS_PER_SPLIT = 50     # Train 또는 Test에 최소 50건 필요

# 🟢 일반화 양호 조건
TT_GREEN_TEST_RET_MIN = 0.0          # Test 수익률 ≥ 0 필수
TT_GREEN_TEST_ALPHA_MIN = 0.0        # Test alpha ≥ 0 필수 (alpha 산출 시)
TT_GREEN_TEST_MDD_MIN = -15.0        # Test MDD ≥ -15%
TT_GREEN_PERFORMANCE_RETENTION = 0.60
# Test 수익률 / Train 수익률 ≥ 60% (성능 유지율)

# 🔴 과최적화 임계
TT_RED_TEST_RET_MAX = -5.0           # Test < -5% → 🔴
TT_RED_RETENTION_MAX = 0.20          # Test/Train < 20% → 🔴 (Train만 좋음)

# 🚨 lookahead 의심
TT_LOOKAHEAD_BOTH_ANOMALY = True     # Train+Test 모두 anomaly → 🚨
TT_LOOKAHEAD_TEST_COLLAPSE = -20.0   # Test < -20% 급락 → 🚨


def run_train_test_split(
    all_recs: pd.DataFrame,
    preset_key: str = "balanced",
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> dict:
    """[v3.9.18] rec_date 기준 정렬 후 Train/Test 분할 백테스트.

    Args:
        all_recs: 누적 추천 CSV (rec_date 컬럼 필수)
        preset_key: "conservative" / "balanced" / "aggressive" / "scalping"
                    (잘못된 key → balanced fallback)
        test_ratio: Test 비율 (기본 0.30 = 최근 30%)

    Returns:
        dict {
            "preset": str,
            "cfg": dict,
            "split_info": {
                "n_total": int,
                "n_train": int,
                "n_test": int,
                "train_date_range": (start, end),
                "test_date_range": (start, end),
                "test_ratio": float,
            },
            "train_result": dict (alpha/anomaly_flags 첨부) 또는 {"error":...}
            "test_result": dict (alpha/anomaly_flags 첨부) 또는 {"error":...}
            "verdict": dict (icon/level/title/color_class/body/reasons)
        }
    """
    # lazy import — services는 components를 모듈 top에서 import 안 함
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

    # ─── 1. 데이터 검증 + 정렬 ───
    if all_recs.empty or "rec_date" not in all_recs.columns:
        return {
            "preset": preset_key, "cfg": cfg,
            "split_info": {"n_total": 0, "n_train": 0, "n_test": 0},
            "train_result": {"error": "rec_date 컬럼 없음"},
            "test_result": {"error": "rec_date 컬럼 없음"},
            "verdict": {
                "icon": "⚪",
                "level": "unknown",
                "title": "데이터 부족",
                "color_class": "text-gray-400",
                "body": "rec_date 컬럼이 없거나 데이터가 비어있음.",
                "reasons": ["데이터 검증 실패"],
            },
        }

    # [v3.9.18b] 평가 1 해결 — 고유 rec_date 기준 분할
    # 이전 v3.9.18: row 개수 기준 분할 → 같은 날짜가 Train/Test 양쪽 가능
    # 변경: 고유 날짜를 set으로 분리 → 같은 추천일이 한쪽에만 속함
    sorted_recs = all_recs.sort_values("rec_date", kind="stable").reset_index(
        drop=True
    )
    n_total = len(sorted_recs)

    # 고유 날짜 추출 (정렬된 상태로)
    unique_dates = (
        sorted_recs["rec_date"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    n_unique_dates = len(unique_dates)

    # 날짜 set 기준 70/30 분할
    date_split_idx = int(n_unique_dates * (1 - test_ratio))
    train_dates = set(unique_dates[:date_split_idx])
    test_dates = set(unique_dates[date_split_idx:])

    # row 추출 (set 기반 filter)
    rec_date_str = sorted_recs["rec_date"].astype(str)
    train_recs = sorted_recs[rec_date_str.isin(train_dates)].copy()
    test_recs = sorted_recs[rec_date_str.isin(test_dates)].copy()

    n_train = len(train_recs)
    n_test = len(test_recs)
    n_train_dates = len(train_dates)
    n_test_dates = len(test_dates)

    # 최소 50건씩 보장
    if n_train < MIN_RECS_PER_SPLIT or n_test < MIN_RECS_PER_SPLIT:
        return {
            "preset": preset_key, "cfg": cfg,
            "split_info": {
                "n_total": n_total,
                "n_train": n_train, "n_test": n_test,
                "n_unique_dates": n_unique_dates,
                "n_train_dates": n_train_dates,
                "n_test_dates": n_test_dates,
                "test_ratio": test_ratio,
            },
            "train_result": {"error": "데이터 부족"},
            "test_result": {"error": "데이터 부족"},
            "verdict": {
                "icon": "⚪",
                "level": "unknown",
                "title": "데이터 부족",
                "color_class": "text-gray-400",
                "body": (
                    f"Train({n_train}건/{n_train_dates}일) 또는 "
                    f"Test({n_test}건/{n_test_dates}일)에 최소 "
                    f"{MIN_RECS_PER_SPLIT}건 필요. recommend_*.csv 누적 후 재시도."
                ),
                "reasons": [
                    f"Train {n_train} / Test {n_test} 건"
                ],
            },
        }

    # 날짜 범위 — 정렬된 unique_dates에서 첫/마지막
    train_date_range = (
        unique_dates[0],
        unique_dates[date_split_idx - 1] if date_split_idx > 0 else "",
    )
    test_date_range = (
        unique_dates[date_split_idx] if date_split_idx < n_unique_dates else "",
        unique_dates[-1],
    )

    # ─── 2. Train/Test 각각 백테스트 ───
    def _run_segment(segment_recs: pd.DataFrame) -> dict:
        try:
            result = _run_backtest(
                segment_recs,
                cfg["min_score"], cfg["hold_days"],
                cfg["stop_pct"], cfg["target_pct"],
                cfg["top_k"], cfg["cost_pct"],
            )
        except Exception as e:
            _logger.warning(f"[train_test] 백테스트 실패: {e}")
            return {"error": f"백테스트 실패: {e}"}

        if "error" not in result:
            try:
                alpha, alpha_mode = _calc_kospi_alpha(result, cfg)
            except Exception as e:
                _logger.debug(f"[train_test] alpha 계산 실패: {e}")
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
            n_total_status = n_win + int(sd.get("STOP", 0) or 0) + int(
                sd.get("HOLD_EXIT", 0) or 0
            )
            result["tp_saturation"] = (
                (n_win / n_total_status * 100) if n_total_status > 0 else 0.0
            )
            result["tp_threshold"] = tp_saturation_threshold(
                cfg["target_pct"]
            )
        return result

    train_result = _run_segment(train_recs)
    test_result = _run_segment(test_recs)

    # ─── 3. 비교 verdict ───
    verdict = derive_train_test_verdict(train_result, test_result)

    return {
        "preset": preset_key,
        "cfg": cfg,
        "split_info": {
            "n_total": n_total,
            "n_train": n_train,
            "n_test": n_test,
            "n_unique_dates": n_unique_dates,
            "n_train_dates": n_train_dates,
            "n_test_dates": n_test_dates,
            "train_date_range": train_date_range,
            "test_date_range": test_date_range,
            "test_ratio": test_ratio,
        },
        "train_result": train_result,
        "test_result": test_result,
        "verdict": verdict,
    }


def derive_train_test_verdict(train_result: dict, test_result: dict) -> dict:
    """[v3.9.18b] Train/Test 비교 판정 — 5단계 (평가 v3.9.18 보정).

    🟢 일반화 양호       : Test 수익+alpha+MDD+retention 모두 통과 (alpha 산출)
    🟡 일반화 후보 보류   : 절대수익 양호하나 alpha 산출 안 됨 — 시장 검증 보류
    🟡 약화              : Test 수익 양수지만 일부 미달
    🔴 OOS 붕괴 / 과최적화: Test 손실 OR retention < 20% OR Test 급락
    🚨 lookahead 의심    : Train + Test 모두 anomaly (둘 다 비현실 — 구조적 의심)
    ⚪ 데이터 부족      : 한쪽 또는 양쪽 error

    [v3.9.18b 평가 보정]:
    - 평가 2: alpha None인데 🟢 일반화 양호는 과함 → 🟡 일반화 후보 보류
    - 평가 3: Test 단독 -20%는 lookahead가 아니라 OOS 붕괴/과최적화
              🚨는 양쪽 anomaly 단독 조건으로 좁힘
    """
    # 데이터 부족 케이스
    if "error" in train_result or "error" in test_result:
        train_err = train_result.get("error", "")
        test_err = test_result.get("error", "")
        return {
            "icon": "⚪",
            "level": "unknown",
            "title": "데이터 부족",
            "color_class": "text-gray-400",
            "body": (
                f"Train: {train_err or '정상'} / Test: {test_err or '정상'}"
            ),
            "reasons": ["분할 검증 불가"],
        }

    train_ret = float(train_result.get("total_return", 0) or 0)
    train_mdd = float(train_result.get("mdd", 0) or 0)
    train_anom = train_result.get("anomaly_flags", [])
    train_alpha = train_result.get("alpha")
    train_alpha_mode = train_result.get("alpha_mode")

    test_ret = float(test_result.get("total_return", 0) or 0)
    test_mdd = float(test_result.get("mdd", 0) or 0)
    test_anom = test_result.get("anomaly_flags", [])
    test_alpha = test_result.get("alpha")

    # 성능 유지율 — Train 양수일 때만 의미 있음
    if train_ret > 0:
        retention = test_ret / train_ret
    else:
        retention = None

    # ─── 🚨 lookahead 의심 — 양쪽 anomaly 단독 조건 (평가 3 보정) ───
    # 이전 v3.9.18: Test -20% 단독도 🚨로 분류 → 과함
    # v3.9.18b: 양쪽 anomaly만 🚨, Test 급락은 🔴 OOS 붕괴로 분리
    both_anomaly = bool(train_anom) and bool(test_anom)

    if both_anomaly:
        reasons = ["Train+Test 모두 anomaly"]
        if train_anom:
            reasons.append(f"Train: {train_anom[0]}")
        if test_anom:
            reasons.append(f"Test: {test_anom[0]}")
        return {
            "icon": "🚨",
            "level": "lookahead",
            "title": "lookahead 의심 · 실전 비권장",
            "color_class": "text-red-500",
            "body": (
                f"Train {train_ret:+.1f}% / Test {test_ret:+.1f}% — "
                f"양쪽 구간 모두 비현실적 수치. 사후 ret 컬럼이 미래 정보를 "
                "누설했을 가능성. OHLCV 기반 정밀 백테스트 필수."
            ),
            "reasons": reasons[:3],
        }

    # ─── 🔴 OOS 붕괴 / 과최적화 — Test 급락 포함 (평가 3 분리) ───
    test_collapse = test_ret < TT_LOOKAHEAD_TEST_COLLAPSE  # -20% 이하
    is_overfitting = test_ret < TT_RED_TEST_RET_MAX  # -5% 이하
    if retention is not None and retention < TT_RED_RETENTION_MAX:  # < 20%
        is_overfitting = True

    if test_collapse or is_overfitting:
        reasons = []
        if test_collapse:
            reasons.append(
                f"Test 급락 ({test_ret:+.1f}%) — OOS 붕괴"
            )
            title = "OOS 붕괴 · 실전 비권장"
        else:
            title = "과최적화 의심"
        if test_ret < TT_RED_TEST_RET_MAX and not test_collapse:
            reasons.append(f"Test 손실 ({test_ret:+.1f}%)")
        if retention is not None and retention < TT_RED_RETENTION_MAX:
            reasons.append(
                f"성능 유지율 {retention*100:.0f}% (기준 20%)"
            )
        if test_alpha is not None and test_alpha < 0:
            reasons.append(f"Test alpha 음수 ({test_alpha:+.2f}%p)")
        return {
            "icon": "🔴",
            "level": "red",
            "title": title,
            "color_class": "text-red-400",
            "body": (
                f"Train은 양호({train_ret:+.1f}%)했으나 Test에서 "
                f"{test_ret:+.1f}%로 붕괴. 과거 데이터에만 맞춘 전략 또는 "
                "최근 구간 구조 변화. 실전 적용 비권장."
            ),
            "reasons": reasons,
        }

    # ─── 🟢 일반화 양호 또는 🟡 일반화 후보 (평가 2 보정) ───
    basic_green = (
        test_ret >= TT_GREEN_TEST_RET_MIN
        and test_mdd >= TT_GREEN_TEST_MDD_MIN
        and not test_anom
        and (test_alpha is None or test_alpha >= TT_GREEN_TEST_ALPHA_MIN)
        and (
            retention is None
            or retention >= TT_GREEN_PERFORMANCE_RETENTION
        )
    )

    if basic_green:
        # [v3.9.18b] alpha 산출 여부로 🟢 vs 🟡 분기
        # 평가 2: alpha None인데 🟢 일반화 양호는 과함 → 🟡 일반화 후보 보류
        # v3.9.17c의 강건성 패턴과 일관 — "후보 + 평가 보류"
        retention_clause = (
            f"성능 유지율 {retention*100:.0f}%"
            if retention is not None
            else "성능 유지율 산출 불가 (Train 손실)"
        )
        if test_alpha is not None:
            # 🟢 일반화 양호 — alpha 산출 + 양수
            return {
                "icon": "🟢",
                "level": "green",
                "title": "일반화 양호",
                "color_class": "text-emerald-400",
                "body": (
                    f"Train {train_ret:+.1f}% → Test {test_ret:+.1f}%. "
                    f"{retention_clause}, Test MDD {test_mdd:.1f}%, "
                    f"Test alpha {test_alpha:+.2f}%p, anomaly 없음. "
                    "최근 구간에서도 살아남는 전략 — 실전 후보."
                ),
                "reasons": ["Test 구간 일반화 양호"],
            }
        else:
            # 🟡 일반화 후보 보류 — 절대수익 양호하나 시장 검증 없음
            return {
                "icon": "🟡",
                "level": "yellow_candidate",
                "title": "일반화 후보 · alpha 평가 보류",
                "color_class": "text-yellow-400",
                "body": (
                    f"Train {train_ret:+.1f}% → Test {test_ret:+.1f}%. "
                    f"{retention_clause}, Test MDD {test_mdd:.1f}%, "
                    "anomaly 없음 — 절대수익 기준은 통과. 다만 Test alpha "
                    "미산출(벤치 데이터 부족) — KOSPI 대비 검증 불가. "
                    "kospi_daily.csv 갱신 또는 bench_cache 보강 후 재평가 권장."
                ),
                "reasons": ["Test alpha 미산출 — 시장 검증 보류"],
            }

    # ─── 🟡 약화 (그 외) ───
    misses = []
    if test_ret < TT_GREEN_TEST_RET_MIN:
        misses.append(f"Test 수익 음수 ({test_ret:+.1f}%)")
    if test_mdd < TT_GREEN_TEST_MDD_MIN:
        misses.append(f"Test MDD 큼 ({test_mdd:.1f}%)")
    if test_anom:
        misses.append(f"Test anomaly: {test_anom[0]}")
    if test_alpha is not None and test_alpha < TT_GREEN_TEST_ALPHA_MIN:
        misses.append(f"Test alpha 음수 ({test_alpha:+.2f}%p)")
    if retention is not None and retention < TT_GREEN_PERFORMANCE_RETENTION:
        misses.append(
            f"성능 유지율 {retention*100:.0f}% (기준 60%)"
        )

    return {
        "icon": "🟡",
        "level": "yellow",
        "title": "약화 · Test 구간 일부 미달",
        "color_class": "text-yellow-400",
        "body": (
            f"Train {train_ret:+.1f}% → Test {test_ret:+.1f}% — "
            f"Test 구간에서 일부 지표 미달. 추가 검증 후 적용 권장."
        ),
        "reasons": misses or ["일부 지표 미달"],
    }
