"""
services/backtest_robustness.py
================================
[v3.9.17b] 강건성 테스트 로직 SSOT — UI 비의존.

평가 v3.9.17 4건 보정:
1. alpha 양수 비율을 🟢 조건에 반영 (절대수익만으로 강건함 판정 차단)
2. alpha coverage / positive ratio denominator 분리 → 신뢰도 낮은 alpha 차단
3. 프리셋별 강건성 (run_robustness_test가 preset key 받아서 분기)
4. UI 비의존 — components.* 의존 0, nicegui import 0

호출처:
- components/backtest_robustness.py: _render_robustness_table만 보유 (UI 렌더)
- 향후 CLI/service layer에서도 직접 사용 가능
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

_logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# [v3.9.17b] 강건성 임계값 — 모듈 상수
# ════════════════════════════════════════════════════════════════
# 🟢 강건함 조건 — 4기준 AND
ROBUST_GREEN_POSITIVE_RATIO = 0.80   # 수익 양수 비율 ≥ 80%
ROBUST_GREEN_ANOMALY_MAX = 0.20      # anomaly 비율 ≤ 20%
ROBUST_GREEN_MDD_RATIO = 0.60        # MDD ≥ -15% 비율 ≥ 60%

# [v3.9.17b 신규] alpha 기준 — 평가 지적 1 해결
ROBUST_GREEN_ALPHA_RATIO = 0.50      # alpha 양수 비율 ≥ 50% (n_success 분모)
ROBUST_GREEN_ALPHA_COVERAGE = 0.70   # alpha 산출 조합 비율 ≥ 70% (n_success 분모)
# coverage가 70% 미만이면 alpha 기준은 평가 불가 → 강건성 판정에서 alpha 무시 가능

ROBUST_YELLOW_POSITIVE_RATIO = 0.50  # 수익 양수 비율 ≥ 50% → 🟡 (이하 🔴)
ROBUST_RED_ANOMALY_RATIO = 0.50      # anomaly 비율 > 50% → 🔴 (강제)

# 27 조합 구성 — 기준 ± 5
PARAM_DELTAS = {
    "min_score": [-5, 0, +5],
    "top_k": [-5, 0, +5],
    "hold_days": [-5, 0, +5],
}


def run_robustness_test(
    all_recs: pd.DataFrame,
    base_preset_key: str = "balanced",
) -> dict:
    """[v3.9.17b] 기준 프리셋 주변 27조합 백테스트.

    Args:
        all_recs: 누적 추천 CSV
        base_preset_key: "conservative"/"balanced"/"aggressive"/"scalping"
                         (잘못된 key → balanced fallback)

    Returns:
        dict {
            "base_preset": str,
            "base_cfg": dict,
            "combos": [
                {
                    "delta_min_score": int,
                    "delta_top_k": int,
                    "delta_hold_days": int,
                    "cfg": dict,
                    "result": dict (alpha/anomaly_flags/tp_saturation 첨부)
                            또는 {"error": ...}
                }, ... 27개
            ],
            "summary": dict (positive_ret_ratio, anomaly_ratio,
                              alpha_coverage_ratio, positive_alpha_ratio, ...),
            "verdict": dict (icon/level/title/color_class/body/reasons)
        }
    """
    # lazy import — services는 components를 직접 의존하지 않도록.
    # components.tab_backtest의 _run_backtest와 PRESETS는 함수 내부에서만 사용.
    from components.tab_backtest import PRESETS, _run_backtest  # ci-allow: layer-violation  # lazy import (top 의존 0)
    from components.backtest_verdict import _calc_kospi_alpha  # ci-allow: layer-violation  # lazy import (top 의존 0)
    from services.backtest_policy import (
        detect_anomaly_flags,
        tp_saturation_threshold,
    )

    if base_preset_key not in PRESETS:
        base_preset_key = "balanced"
    base_preset = PRESETS[base_preset_key]
    base_cfg = {
        "min_score": base_preset["min_score"],
        "top_k": base_preset["top_k"],
        "hold_days": base_preset["hold_days"],
        "target_pct": base_preset["target_pct"],
        "stop_pct": base_preset["stop_pct"],
        "cost_pct": base_preset["cost_pct"],
    }

    combos = []
    for d_ms in PARAM_DELTAS["min_score"]:
        for d_tk in PARAM_DELTAS["top_k"]:
            for d_hd in PARAM_DELTAS["hold_days"]:
                cfg = dict(base_cfg)
                cfg["min_score"] = max(50, base_cfg["min_score"] + d_ms)
                cfg["top_k"] = max(1, base_cfg["top_k"] + d_tk)
                cfg["hold_days"] = max(1, base_cfg["hold_days"] + d_hd)

                try:
                    result = _run_backtest(
                        all_recs,
                        cfg["min_score"], cfg["hold_days"],
                        cfg["stop_pct"], cfg["target_pct"],
                        cfg["top_k"], cfg["cost_pct"],
                    )
                except Exception as e:
                    _logger.warning(
                        f"[robustness] ({d_ms},{d_tk},{d_hd}) 실패: {e}"
                    )
                    result = {"error": f"백테스트 실패: {e}"}

                if "error" not in result:
                    try:
                        alpha, alpha_mode = _calc_kospi_alpha(result, cfg)
                    except Exception as e:
                        _logger.debug(
                            f"[robustness] alpha 계산 실패: {e}"
                        )
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
                    result["tp_threshold"] = tp_saturation_threshold(
                        cfg["target_pct"]
                    )

                combos.append({
                    "delta_min_score": d_ms,
                    "delta_top_k": d_tk,
                    "delta_hold_days": d_hd,
                    "cfg": cfg,
                    "result": result,
                })

    summary = summarize_robustness(combos)
    verdict = derive_robustness_verdict(summary)

    return {
        "base_preset": base_preset_key,
        "base_cfg": base_cfg,
        "combos": combos,
        "summary": summary,
        "verdict": verdict,
    }


def summarize_robustness(combos: list) -> dict:
    """27조합 통계 집계.

    [v3.9.17b 보정] alpha denominator 분리:
    - alpha_coverage_ratio = (alpha 산출된 조합) / n_success
    - positive_alpha_ratio = (alpha > 0 조합) / n_success  ← coverage 낮으면 자동으로 낮음

    이전 v3.9.17은 positive_alpha_ratio = n_positive_alpha / len(alphas)였는데,
    alpha 1개만 산출되고 그게 양수면 ratio = 100%로 잘못 표시되는 문제.
    """
    n_total = len(combos)
    valid = [c for c in combos if "error" not in c["result"]]
    n_success = len(valid)

    if n_success == 0:
        return {
            "n_total": n_total, "n_success": 0,
            "n_positive_ret": 0, "n_positive_alpha": 0, "n_alpha_calculated": 0,
            "n_anomaly": 0, "n_mdd_within_15": 0,
            "positive_ret_ratio": 0.0,
            "positive_alpha_ratio": 0.0,
            "alpha_coverage_ratio": 0.0,
            "anomaly_ratio": 0.0,
            "mdd_within_15_ratio": 0.0,
            "worst_return": 0.0, "avg_return": 0.0, "best_return": 0.0,
        }

    returns = [float(c["result"].get("total_return", 0) or 0) for c in valid]
    mdds = [float(c["result"].get("mdd", 0) or 0) for c in valid]
    # alpha — None 제외하고 모음
    alphas = [
        c["result"].get("alpha")
        for c in valid
        if c["result"].get("alpha") is not None
    ]
    n_alpha_calculated = len(alphas)

    n_positive_ret = sum(1 for r in returns if r > 0)
    n_positive_alpha = sum(1 for a in alphas if a > 0)
    n_anomaly = sum(
        1 for c in valid if c["result"].get("anomaly_flags")
    )
    n_mdd_within_15 = sum(1 for m in mdds if m >= -15)

    return {
        "n_total": n_total,
        "n_success": n_success,
        "n_positive_ret": n_positive_ret,
        "n_positive_alpha": n_positive_alpha,
        "n_alpha_calculated": n_alpha_calculated,
        "n_anomaly": n_anomaly,
        "n_mdd_within_15": n_mdd_within_15,
        "positive_ret_ratio": n_positive_ret / n_success,
        # [v3.9.17b] alpha denominator를 n_success로 — coverage 낮으면 자동으로 낮음
        "positive_alpha_ratio": n_positive_alpha / n_success,
        "alpha_coverage_ratio": n_alpha_calculated / n_success,
        "anomaly_ratio": n_anomaly / n_success,
        "mdd_within_15_ratio": n_mdd_within_15 / n_success,
        "worst_return": min(returns),
        "avg_return": sum(returns) / len(returns),
        "best_return": max(returns),
    }


def derive_robustness_verdict(summary: dict) -> dict:
    """[v3.9.17b] 강건성 판정 — alpha 기준 추가.

    🟢 강건함 조건 (모두 AND):
        positive_ret_ratio   >= 80%
        anomaly_ratio        <= 20%
        mdd_within_15_ratio  >= 60%
        ★ alpha 평가 추가:
          (alpha_coverage_ratio < 70%)
            → alpha coverage 부족하면 alpha 기준 skip (절대수익 양호하면 🟢 가능)
          (alpha_coverage_ratio >= 70%)
            → positive_alpha_ratio >= 50% 필수

    🔴 과최적화 의심: positive_ret < 50% OR anomaly > 50%
    🟡 조건부: 그 외
    ⚪ 데이터 부족: n_success = 0

    평가 1: alpha 양수 비율을 🟢 조건에 반영 → 절대수익만 양수인데 시장 열위면 🟢 차단
    평가 2: coverage denominator를 n_success로 → 신뢰도 낮은 alpha "100% 양수" 차단
    """
    pr = summary["positive_ret_ratio"]
    ar = summary["anomaly_ratio"]
    mr = summary["mdd_within_15_ratio"]
    par = summary["positive_alpha_ratio"]
    acr = summary["alpha_coverage_ratio"]
    ns = summary["n_success"]

    if ns == 0:
        return {
            "icon": "⚪",
            "level": "unknown",
            "title": "데이터 부족",
            "color_class": "text-gray-400",
            "body": "27조합 모두 백테스트 실패. 데이터 부족 가능성.",
            "reasons": ["모든 조합 실패"],
        }

    # 🔴 과최적화 의심 (가장 먼저 — 절대 차단 사유)
    if pr < ROBUST_YELLOW_POSITIVE_RATIO or ar > ROBUST_RED_ANOMALY_RATIO:
        reasons = []
        if pr < ROBUST_YELLOW_POSITIVE_RATIO:
            reasons.append(f"수익 양수 비율 낮음 ({pr*100:.0f}%)")
        if ar > ROBUST_RED_ANOMALY_RATIO:
            reasons.append(f"anomaly 비율 높음 ({ar*100:.0f}%)")
        return {
            "icon": "🔴",
            "level": "red",
            "title": "과최적화 의심",
            "color_class": "text-red-400",
            "body": (
                f"27조합 중 수익 양수 {summary['n_positive_ret']}개 "
                f"({pr*100:.0f}%), anomaly {summary['n_anomaly']}개 "
                f"({ar*100:.0f}%). 기준 프리셋만 좋은 cherry-picking 또는 "
                f"lookahead bias 가능성. 실전 적용 비권장."
            ),
            "reasons": reasons,
        }

    # 🟢 강건함 — 기본 3기준 + alpha 평가
    basic_pass = (
        pr >= ROBUST_GREEN_POSITIVE_RATIO
        and ar <= ROBUST_GREEN_ANOMALY_MAX
        and mr >= ROBUST_GREEN_MDD_RATIO
    )

    # [v3.9.17b] alpha 평가:
    # - coverage 70%+: alpha 양수 비율도 50%+ 필수
    # - coverage 70% 미만: alpha 데이터 신뢰 부족 → alpha 기준 skip (절대수익만 평가)
    if acr >= ROBUST_GREEN_ALPHA_COVERAGE:
        alpha_pass = par >= ROBUST_GREEN_ALPHA_RATIO
        alpha_judgement_available = True
    else:
        alpha_pass = True  # coverage 부족 → 평가 skip (pass 처리)
        alpha_judgement_available = False

    if basic_pass and alpha_pass:
        # [v3.9.17c] alpha 평가 가능 여부에 따라 제목/색상 분기.
        # 평가 v3.9.17b 지적 1: coverage 부족인데 🟢 강건함으로 표시되면
        # 사용자가 "시장 대비 검증된 강건성"으로 오해 가능. 제목에서 명시.
        if alpha_judgement_available:
            return {
                "icon": "🟢",
                "level": "green",
                "title": "강건함",
                "color_class": "text-emerald-400",
                "body": (
                    f"27조합 중 수익 양수 {summary['n_positive_ret']}개 "
                    f"({pr*100:.0f}%), MDD -15% 이내 "
                    f"{summary['n_mdd_within_15']}개 ({mr*100:.0f}%), "
                    f"anomaly {summary['n_anomaly']}개 ({ar*100:.0f}%), "
                    f"alpha 양수 {summary['n_positive_alpha']}개 "
                    f"({par*100:.0f}%). 주변 조합도 일관되게 양호 — 실전 후보."
                ),
                "reasons": ["주변 조합 일관 양호"],
            }
        else:
            # 절충안: 절대수익 기준은 통과했으나 alpha coverage 부족
            # → 🟡 강건함 후보. 제목에서 "alpha 평가 보류" 즉시 인지.
            return {
                "icon": "🟡",
                "level": "yellow_candidate",
                "title": "강건함 후보 · alpha 평가 보류",
                "color_class": "text-yellow-400",
                "body": (
                    f"절대수익 기준은 통과: 수익 양수 "
                    f"{summary['n_positive_ret']}개 ({pr*100:.0f}%), "
                    f"MDD -15% 이내 {summary['n_mdd_within_15']}개 "
                    f"({mr*100:.0f}%), anomaly {summary['n_anomaly']}개 "
                    f"({ar*100:.0f}%). 다만 alpha coverage {acr*100:.0f}% "
                    f"(기준 70% 미달) — KOSPI 대비 검증 부족. "
                    f"kospi_daily.csv 갱신 또는 bench_cache 보강 후 재평가 권장."
                ),
                "reasons": [
                    f"alpha coverage {acr*100:.0f}% (기준 70%) — 시장 대비 평가 불가",
                ],
            }

    # 🟡 조건부 — 한두 기준 미달
    misses = []
    if pr < ROBUST_GREEN_POSITIVE_RATIO:
        misses.append(f"수익 양수 비율 {pr*100:.0f}% (기준 80%)")
    if ar > ROBUST_GREEN_ANOMALY_MAX:
        misses.append(f"anomaly 비율 {ar*100:.0f}% (기준 20%)")
    if mr < ROBUST_GREEN_MDD_RATIO:
        misses.append(f"MDD 이내 비율 {mr*100:.0f}% (기준 60%)")
    # [v3.9.17b] alpha 사유도 명시
    if alpha_judgement_available and par < ROBUST_GREEN_ALPHA_RATIO:
        misses.append(
            f"alpha 양수 비율 {par*100:.0f}% (기준 50%) — 시장 대비 열위 조합 많음"
        )
    elif not alpha_judgement_available:
        misses.append(
            f"alpha coverage {acr*100:.0f}% (기준 70% 미달 — 시장 대비 평가 불가)"
        )

    return {
        "icon": "🟡",
        "level": "yellow",
        "title": "조건부",
        "color_class": "text-yellow-400",
        "body": (
            f"27조합 중 수익 양수 {summary['n_positive_ret']}개 "
            f"({pr*100:.0f}%) — 일부 조합만 양호. "
            f"기준 프리셋 외 조합들의 안정성 추가 검증 권장."
        ),
        "reasons": misses,
    }
