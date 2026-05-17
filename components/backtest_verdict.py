"""
components/backtest_verdict.py
==============================
[v3.9.17] 전략 판정 SSOT — verdict 로직 + KOSPI 알파 계산.

이 모듈은 단일 백테스트 카드 + 4프리셋 비교표 + 27조합 강건성 테스트가
모두 같은 판정 기준을 공유하도록 보장하는 SSOT 역할.

분리 배경 (평가 v3.9.16):
- tab_backtest.py가 2,400줄을 넘어 유지보수 비용 증가
- _derive_strategy_verdict가 단일 카드/비교표 두 호출처에서 공유되는데
  같은 파일 안에 있으면 순환 의존 위험
- v3.9.17부터 강건성 테스트(27조합)도 같은 판정 사용 — 별도 모듈 필요

UI 비의존:
- nicegui import 0
- service layer (services.backtest_policy, services.benchmarks)만 사용
- tab_backtest.py를 import 하지 않음 (순환 방지)

호출처:
- components/tab_backtest.py: _render_strategy_verdict_card에서 사용
- components/backtest_preset_compare.py: 비교표 판정 컬럼
- components/backtest_robustness.py: 27조합 안의 각 조합 verdict 산출
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import pandas as pd

_logger = logging.getLogger(__name__)


def _calc_kospi_alpha(result: dict, cfg: dict):
    """[v3.9.15b] 전략 vs KOSPI 알파 계산.
    
    [v3.9.15d] 2단계 폴백:
    1. daily KOSPI CSV 있으면 → 진짜 거래일별 알파
       (각 trade의 추천일에서 hold_days 보유 KOSPI 수익률 차감)
    2. 없으면 → 간이 알파 (전략 거래당 평균 vs KOSPI 단일 시점)
    
    [v3.9.15e] critical bug fix — trades_df 실제 스키마 정합:
    - _run_backtest():446-454에서 만드는 trades_df의 실제 컬럼은
      rec_date / code / name / score / raw_ret / net_ret / status.
    - v3.9.15d에서 "date" / "net_pct"로 lookup 하던 코드는 영원히
      매칭 0건 → if len(kospi_rets) >= 10 실패 → 항상 simple fallback.
    - 즉 v3.9.15d의 "real" 경로는 코드만 있고 실제 작동 0%였음.
    - 수정: trades.columns에 "rec_date" 확인, row.get("rec_date"), row.get("net_ret").
    - 단위: net_ret도 % 단위(_run_backtest:444), KOSPI 수익률도 %라 직접 차감 OK.
    
    [v3.9.15e] 표본 부족(<10) 시 명시적 simple fallback:
    - 이전엔 if 블록 통과 못 하면 자동으로 simple로 떨어졌지만
      매칭은 됐는데 표본만 부족한 경우 vs 컬럼 자체가 없는 경우를
      구분 못 했음. 이제 두 케이스 모두 simple로 가지만 로그가 다름.
    
    Returns: (alpha_pp, mode) 튜플 — mode는 "real" 또는 "simple"
             또는 (None, None) — 데이터 없음
    """
    try:
        from services.benchmarks import (
            load_bench_cache, get_kospi_return,
            load_kospi_daily, get_kospi_return_for_date,
        )
        hold_days = int(cfg.get("hold_days", 5) or 5)

        # === 1단계: 진짜 거래일별 알파 시도 ===
        daily = load_kospi_daily()
        if daily is not None:
            trades = result.get("trades_df")
            # [v3.9.15e fix] "date" → "rec_date" (trades_df 실제 컬럼명)
            if trades is not None and not trades.empty and "rec_date" in trades.columns:
                # 각 trade의 추천일 기준 KOSPI 수익률 매칭
                kospi_rets = []
                strat_rets = []
                for _, r in trades.iterrows():
                    # [v3.9.15e fix] "date" → "rec_date"
                    date_str = str(r.get("rec_date", "")).replace("-", "")
                    if not date_str:
                        continue
                    kr = get_kospi_return_for_date(date_str, hold_days)
                    # [v3.9.15e fix] "net_pct" → "net_ret"
                    sr = r.get("net_ret")
                    try:
                        sr_val = float(sr) if sr is not None and not pd.isna(sr) else None
                    except Exception:
                        sr_val = None
                    if kr is not None and sr_val is not None:
                        kospi_rets.append(float(kr))
                        strat_rets.append(sr_val)

                n_matched = len(kospi_rets)
                if n_matched >= 10:  # 최소 표본
                    import statistics as _s
                    avg_strat = _s.mean(strat_rets)
                    avg_kospi = _s.mean(kospi_rets)
                    _logger.info(
                        f"📊 real alpha 산출: 매칭 {n_matched}건 / "
                        f"전체 {len(trades)}건 / 전략 {avg_strat:+.2f}% "
                        f"KOSPI {avg_kospi:+.2f}% → 알파 {avg_strat-avg_kospi:+.2f}%p"
                    )
                    return (avg_strat - avg_kospi, "real")
                else:
                    _logger.info(
                        f"real alpha 표본 부족 (매칭 {n_matched} < 10) → simple fallback"
                    )
            else:
                _logger.debug(
                    "trades_df 없음 또는 rec_date 컬럼 없음 → simple fallback"
                )

        # === 2단계: 간이 알파 (fallback) ===
        bench = load_bench_cache()
        kospi_ret = get_kospi_return(bench, hold_days)
        if kospi_ret is None:
            return (None, None)
        win_rate = float(result.get("win_rate", 0) or 0) / 100.0
        avg_win = float(result.get("avg_win", 0) or 0)
        avg_loss = float(result.get("avg_loss", 0) or 0)
        strat_per_trade = avg_win * win_rate + avg_loss * (1.0 - win_rate)
        return (strat_per_trade - float(kospi_ret), "simple")
    except Exception as e:
        _logger.debug(f"KOSPI 알파 계산 실패: {e}", exc_info=True)
        return (None, None)


def _derive_strategy_verdict(result: dict, cfg: dict) -> dict:
    """[v3.9.16b] 전략 판정 SSOT — 단일 카드 + 비교표 공유.

    이 함수는 _render_strategy_verdict_card()와 _render_preset_comparison_table()
    두 곳에서 같이 호출되어 판정 결과를 항상 같게 보장한다.

    추출 배경 (v3.9.16 평가):
    - v3.9.16 비교표가 단일 verdict보다 느슨한 조건으로 🟢 부여 → 가짜 🟢 위험
    - alpha is None인데 🟢, 거래 < 100인데 🟢, Sharpe < 0.8인데 🟢 등
    - v3.9.15c에서 어렵게 막은 "벤치 없음에도 초록" 회귀 가능성
    → 한 함수가 SSOT 역할, 두 호출처는 결과만 렌더

    판정 4단계 (verdict_card baseline 그대로):
      🟢 green   : 수익≥5% AND 승률≥55 AND MDD≥-15 AND 거래≥100 AND
                   alpha is not None AND alpha≥0 AND Sharpe(None or ≥0.8) AND not anomaly
      🟡 yellow  : 수익+ but 일부 미달 OR alpha None OR anomaly
      🟠 orange  : 수익+ but MDD<-20 OR 거래<50 OR Sharpe<0.5
      🔴 red     : 수익- OR MDD<-25 OR alpha<0 (시장 열위)

    Returns:
        dict {
            "icon": str,           # 🟢/🟡/🟠/🔴
            "level": str,          # green/yellow/orange/red
            "title": str,          # "실전 후보" / "관찰 후보 · 과대추정 가능성" 등
            "color_class": str,    # Tailwind text-* class
            "bg_class": str,       # 카드 배경 class (단일 카드용)
            "body": str,           # 판정 사유 본문 (단일 카드용)
            "reasons": list[str],  # 짧은 사유 (비교표용)
            "is_anomaly": bool,
            "anomaly_flags": list[str],
            "alpha": float | None,
            "alpha_mode": str | None,
            "tp_saturation": float,
            "tp_threshold": int,
            "tp_saturation_warn": bool,
        }
    """
    from services.backtest_policy import (
        detect_anomaly_flags,
        tp_saturation_threshold,
    )

    total_ret = float(result.get("total_return", 0) or 0)
    win_rate = float(result.get("win_rate", 0) or 0)
    mdd = float(result.get("mdd", 0) or 0)
    n_trades = int(result.get("total_trades", 0) or 0)
    sharpe = result.get("sharpe")
    sharpe_val = (
        float(sharpe) if sharpe is not None and not pd.isna(sharpe) else None
    )
    cagr_raw = result.get("cagr")
    cagr_val = (
        float(cagr_raw)
        if cagr_raw is not None and not pd.isna(cagr_raw)
        else None
    )
    trading_days = int(result.get("trading_days", 0) or 0)

    # alpha — result에 미리 계산되어 있으면 (비교표 경로) 그대로 사용,
    # 없으면 (단일 카드 경로) _calc_kospi_alpha 호출
    if "alpha" in result and "alpha_mode" in result:
        alpha = result["alpha"]
        alpha_mode = result["alpha_mode"]
    else:
        alpha, alpha_mode = _calc_kospi_alpha(result, cfg)

    # anomaly — result에 미리 계산되어 있으면 (비교표 경로) 그대로 사용
    if "anomaly_flags" in result:
        anomaly_flags = result["anomaly_flags"]
    else:
        anomaly_flags = detect_anomaly_flags(
            total_ret=total_ret,
            sharpe_val=sharpe_val,
            cagr_val=cagr_val,
            trading_days=trading_days,
        )
    is_anomaly = bool(anomaly_flags)

    # TP 포화율
    if "tp_saturation" in result and "tp_threshold" in result:
        tp_saturation = float(result["tp_saturation"] or 0)
        tp_threshold = int(result["tp_threshold"] or 70)
    else:
        sd = result.get("status_dist", {}) or {}
        n_win = int(sd.get("WIN", 0) or 0)
        n_stop = int(sd.get("STOP", 0) or 0)
        n_hold = int(sd.get("HOLD_EXIT", 0) or 0)
        n_total = n_win + n_stop + n_hold
        tp_saturation = (n_win / n_total * 100) if n_total > 0 else 0.0
        target_pct = float(cfg.get("target_pct", 5) or 5)
        tp_threshold = tp_saturation_threshold(target_pct)
    tp_saturation_warn = tp_saturation >= tp_threshold

    # ─── 판정 4단계 (verdict_card baseline 그대로) ───
    out = {
        "is_anomaly": is_anomaly,
        "anomaly_flags": anomaly_flags,
        "alpha": alpha,
        "alpha_mode": alpha_mode,
        "tp_saturation": tp_saturation,
        "tp_threshold": tp_threshold,
        "tp_saturation_warn": tp_saturation_warn,
    }

    if total_ret < 0 or mdd < -25:
        out["icon"] = "🔴"
        out["level"] = "red"
        out["title"] = "실전 부적합"
        out["color_class"] = "text-red-400"
        out["bg_class"] = "bg-red-900/15 border-red-500/40"
        if total_ret < 0:
            out["body"] = (
                f"누적 수익률이 {total_ret:+.2f}%로 손실입니다. "
                "현재 설정은 실전 적용에 적합하지 않습니다."
            )
            out["reasons"] = [f"손실 ({total_ret:+.2f}%)"]
        else:
            out["body"] = (
                f"최대 낙폭이 {mdd:.1f}%로 너무 큽니다. "
                "실전 적용 시 큰 손실 위험이 있습니다."
            )
            out["reasons"] = [f"MDD 과대 ({mdd:.1f}%)"]
        return out

    if alpha is not None and alpha < 0:
        out["icon"] = "🔴"
        out["level"] = "red"
        out["title"] = "실전 부적합 · 시장 열위"
        out["color_class"] = "text-red-400"
        out["bg_class"] = "bg-red-900/15 border-red-500/40"
        mode_label = "정확 알파" if alpha_mode == "real" else "간이 알파"
        out["body"] = (
            f"전략 거래당 평균이 KOSPI 동일 보유기간보다 {alpha:+.2f}%p 낮습니다 "
            f"({mode_label} 기준). 수익이 양수여도 시장보다 못하면 "
            "인덱스 매수가 더 유리합니다."
        )
        out["reasons"] = [f"시장 열위 (alpha {alpha:+.2f}%p)"]
        return out

    if mdd < -20 or n_trades < 50 or (sharpe_val is not None and sharpe_val < 0.5):
        out["icon"] = "🟠"
        out["level"] = "orange"
        out["title"] = "검증 부족"
        out["color_class"] = "text-orange-400"
        out["bg_class"] = "bg-orange-900/15 border-orange-500/40"
        reasons = []
        if mdd < -20:
            reasons.append(f"낙폭 큼 (MDD {mdd:.1f}%)")
        if n_trades < 50:
            reasons.append(f"거래 표본 부족 ({n_trades}건)")
        if sharpe_val is not None and sharpe_val < 0.5:
            reasons.append(f"Sharpe 낮음 ({sharpe_val:.2f})")
        out["body"] = (
            f"수익은 양수({total_ret:+.2f}%)지만 {' · '.join(reasons)} — "
            "주변 조합/기간 분할 검증 후 적용을 권합니다."
        )
        out["reasons"] = reasons
        return out

    # 🟢 실전 후보 조건 — verdict_card baseline 그대로 엄격하게
    if (
        total_ret >= 5
        and win_rate >= 55
        and mdd >= -15
        and n_trades >= 100
        and alpha is not None
        and alpha >= 0
        and (sharpe_val is None or sharpe_val >= 0.8)
        and not is_anomaly
    ):
        out["icon"] = "🟢"
        out["level"] = "green"
        out["title"] = "실전 후보"
        out["color_class"] = "text-emerald-400"
        out["bg_class"] = "bg-emerald-900/15 border-emerald-500/40"
        mode_label = "정확 알파" if alpha_mode == "real" else "간이 알파"
        out["body"] = (
            f"수익률 {total_ret:+.2f}%, 승률 {win_rate:.1f}%, MDD {mdd:.1f}%, "
            f"거래 {n_trades}건, {mode_label} {alpha:+.2f}%p — 모든 기준 통과. "
            "과거 데이터 기반이므로 주변 조합 강건성 검증 후 실전 적용 권장."
        )
        out["reasons"] = ["모든 기준 통과"]
        return out

    # 🟡 관찰 — anomaly 케이스 차별 표시
    out["icon"] = "🟡"
    out["level"] = "yellow"
    out["color_class"] = "text-yellow-400"
    out["bg_class"] = "bg-yellow-900/15 border-yellow-500/40"
    if is_anomaly:
        out["title"] = "관찰 후보 · 과대추정 가능성"
        out["body"] = (
            f"기본 지표는 모두 통과({total_ret:+.2f}% / 승률 {win_rate:.1f}% / "
            f"MDD {mdd:.1f}% / 거래 {n_trades}건)지만 "
            f"비현실적 수치 발견: {' · '.join(anomaly_flags)}. "
            "간이 백테스트 구조상 lookahead, 동시 보유 자금 제약 미반영, "
            "TP/SL 장중 도달 순서 미반영으로 인한 과대평가 가능성이 큽니다. "
            "실전 적용 전 OHLCV 기반 정밀 백테스트 + Train/Test 분할 검증 필수."
        )
        out["reasons"] = anomaly_flags[:2]
    else:
        out["title"] = "관찰 후보"
        misses = []
        if total_ret < 5:
            misses.append(f"수익률 작음 ({total_ret:+.2f}%)")
        if win_rate < 55:
            misses.append(f"승률 낮음 ({win_rate:.1f}%)")
        if mdd < -15:
            misses.append(f"낙폭 큼 ({mdd:.1f}%)")
        if n_trades < 100:
            misses.append(f"거래 부족 ({n_trades}건)")
        if sharpe_val is not None and sharpe_val < 0.8:
            misses.append(f"Sharpe 보통 ({sharpe_val:.2f})")
        # alpha None 명시적 표시 (v3.9.16b 보강)
        if alpha is None:
            misses.append("벤치 데이터 없음")
        miss_str = ", ".join(misses) if misses else "일부 지표 미달"
        out["body"] = (
            f"수익은 양호({total_ret:+.2f}%)하지만 {miss_str} — "
            "실전 적용 전 추가 검증이 필요합니다."
        )
        out["reasons"] = misses
    return out


