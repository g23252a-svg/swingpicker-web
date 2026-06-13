# -*- coding: utf-8 -*-
"""
combo_optimizer.py — 지표 조합 최적화 엔진 (v21.3 · v3.8.1 walk-forward)
═══════════════════════════════════════════════════
과거 추천 + 실현 수익률 데이터를 기반으로
최고 승률 지표 조합을 자동 탐색.

매일 파이프라인 종료 후 실행 → optimal_filter_latest.json 저장
→ 대시보드/TOP_PICK에서 활용

[v3.8.1 Phase 2-A · 마감] Walk-forward 검증 추가 (v3.8.0에서 도입, v3.8.1에서 완결)
  - run_combo_optimization() : 기존 grid search (전체 데이터)
                               공통 헬퍼 재활용으로 리팩토링 완료
  - run_combo_optimization_wf() : IS/OOS 분할 검증
      · 앞 절반 (IS): 조합 탐색
      · 뒤 절반 (OOS): 실전 재현성 테스트
      · Robustness = OOS_EV / IS_EV (1.0 가까울수록 견고)
      · 4단 분류: ROBUST / OVERFIT / RECOVERING / WEAK

목적:
  1. 🛡️ 콤보 조건이 진짜로 robust한지 검증
  2. 🏆 최강 조건의 overfit 여부 진단
  3. 신규 조합 후보 발굴 시 안전망
"""
import glob
import json
import logging
import os
from itertools import product
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  [v3.8.1] 공통 헬퍼 — 데이터 로드 / EV 계산
# ═══════════════════════════════════════════════════

def _load_trade_rows(data_dir: str, horizon: int) -> pd.DataFrame:
    """추천 CSV + N일 후 snapshot 매칭 → 거래 결과 DataFrame 반환.

    기존 run_combo_optimization에 있던 로직을 함수로 추출 (walk-forward와 공유).

    Returns:
        DataFrame with columns: ret, win, S, T, AI, ROUTE, SCORE, trade_date
    """
    rec_files = sorted(glob.glob(os.path.join(data_dir, "recommend_2026*.csv")))
    snap_files = sorted(glob.glob(os.path.join(data_dir, "price_snapshot_2026*.csv")))
    snap_dates = [
        os.path.basename(f).replace("price_snapshot_", "").replace(".csv", "")
        for f in snap_files
    ]

    rows = []
    matched_days = 0

    for rf in rec_files:
        rec_ymd = os.path.basename(rf).replace("recommend_", "").replace(".csv", "")
        if rec_ymd not in snap_dates:
            continue
        idx = snap_dates.index(rec_ymd)
        future_idx = idx + horizon
        if future_idx >= len(snap_dates):
            continue

        try:
            rec = pd.read_csv(rf, dtype={"종목코드": str}, encoding="utf-8-sig")
            snap = pd.read_csv(
                os.path.join(data_dir, f"price_snapshot_{snap_dates[future_idx]}.csv"),
                dtype={"종목코드": str}, encoding="utf-8-sig",
            )
        except Exception:
            continue

        rec["종목코드"] = rec["종목코드"].str.zfill(6)
        snap["종목코드"] = snap["종목코드"].str.zfill(6)
        future_close = dict(zip(snap["종목코드"], pd.to_numeric(snap["종가"], errors="coerce")))

        matched_days += 1
        for _, r in rec.iterrows():
            code = r["종목코드"]
            entry = float(pd.to_numeric(r.get("추천매수가", r.get("종가", 0)), errors="coerce") or 0)
            fc = future_close.get(code, np.nan)
            if entry <= 0 or pd.isna(fc):
                continue

            rows.append({
                "ret": (fc / entry - 1) * 100,
                "win": 1 if fc > entry else 0,
                "S": float(r.get("STRUCT_SCORE", 0) or 0),
                "T": float(r.get("TIMING_SCORE", 0) or 0),
                "AI": float(r.get("AI_SCORE", r.get("ML_SCORE", 0)) or 0),
                "ROUTE": str(r.get("ROUTE", "")),
                "SCORE": float(r.get("DISPLAY_SCORE", 0) or 0),
                "trade_date": rec_ymd,  # [v3.8.1] IS/OOS 분할용
                "code": code,           # [v24.2] DQ 패널 merge 키
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    df.attrs["matched_days"] = matched_days

    # ── [v24.2] DATA_INTEGRITY as-of 패널 merge → DQ 컬럼(1=플래그) ──
    # scripts/backtest_data_integrity.py가 생성한 패널이 있을 때만 부여.
    # 패널이 없으면 DQ=0 → dq_exclude 조합이 기존과 완전 동일(하위호환).
    df["DQ"] = 0
    panel_path = os.path.join(data_dir, "data_integrity_asof_panel.csv")
    if not df.empty and os.path.exists(panel_path):
        try:
            _pn = pd.read_csv(
                panel_path, dtype={"종목코드": str, "rec_ymd": str},
                encoding="utf-8-sig",
                usecols=["rec_ymd", "종목코드", "FLAGGED"],
            )
            _flag_keys = set(zip(
                _pn.loc[_pn["FLAGGED"].astype(bool), "rec_ymd"].astype(str),
                _pn.loc[_pn["FLAGGED"].astype(bool), "종목코드"].astype(str).str.zfill(6),
            ))
            df["DQ"] = [
                1 if (str(d), str(c)) in _flag_keys else 0
                for d, c in zip(df["trade_date"], df["code"])
            ]
            logger.info("combo_optimizer: DQ 패널 적용 — 플래그 %d/%d행",
                        int(df["DQ"].sum()), len(df))
        except Exception as e:
            logger.warning("combo_optimizer: DQ 패널 적용 실패 (DQ=0 유지): %s", e)
    return df


def _evaluate_combo(
    df: pd.DataFrame,
    s_min: int,
    t_min: int,
    ai_min: int,
    routes: List[str],
    min_samples: int = 10,
    dq_exclude: int = 0,
) -> Optional[dict]:
    """단일 조합을 데이터에 적용 → 통계 반환. 표본 부족 시 None.

    [v24.2] dq_exclude=1이면 DATA_INTEGRITY 플래그(DQ=1) 행을 제외하고 평가.
    DQ 컬럼이 없거나 dq_exclude=0이면 기존과 완전 동일.
    """
    sub = df[
        (df["S"] >= s_min)
        & (df["T"] >= t_min)
        & (df["AI"] >= ai_min)
        & (df["ROUTE"].isin(routes))
    ]
    if int(dq_exclude) == 1 and "DQ" in sub.columns:
        sub = sub[sub["DQ"] == 0]
    n = len(sub)
    if n < min_samples:
        return None

    wr = sub["win"].mean() * 100
    avg_ret = sub["ret"].mean()
    wins = sub[sub["ret"] > 0]["ret"]
    losses = sub[sub["ret"] <= 0]["ret"]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    wr_dec = wr / 100
    ev = wr_dec * avg_win - (1 - wr_dec) * avg_loss

    return {
        "n": int(n),
        "win_rate": round(wr, 1),
        "avg_ret": round(avg_ret, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "ev": round(ev, 2),
    }


def _all_combos(include_dq: bool = False) -> list:
    """탐색 조합 생성 — 항상 5-tuple (s, t, ai, routes, dq_exclude).

    include_dq=False(기본): dq 차원 [0]만 → 기존 128개 조합과 동일.
    include_dq=True: dq 차원 [0, 1] → 256개 (DATA_INTEGRITY 제외 효과 비교).
    """
    dq_dim = [0, 1] if include_dq else [0]
    return list(product(
        [60, 70, 80, 90],                                   # S_min
        [50, 60, 70, 80],                                   # T_min
        [40, 50, 60, 70],                                   # AI_min
        [["ATTACK", "ARMED"], ["ATTACK", "ARMED", "WAIT"]],  # ROUTE
        dq_dim,                                             # [v24.2] DQ 제외
    ))


# ═══════════════════════════════════════════════════
#  기존 함수 (하위호환 유지)
# ═══════════════════════════════════════════════════

def run_combo_optimization(
    data_dir: str,
    horizon: int = 3,
    min_samples: int = 10,
    top_n: int = 10,
) -> dict:
    """
    조합 최적화 실행 — 전체 데이터 기반 (walk-forward 미적용).

    [v3.8.1] 새 버전은 run_combo_optimization_wf() 사용 권장.
    이 함수는 하위호환을 위해 유지.

    Args:
        data_dir: data/ 디렉토리 경로
        horizon: 보유일수 (3 or 5)
        min_samples: 최소 샘플 수
        top_n: 상위 N개 조합 저장

    Returns:
        {"best": {...}, "top_combos": [...], "meta": {...}}
    """
    # [v3.8.1] 공통 헬퍼 사용으로 중복 제거 (legacy/wf 결과 일관성 보장)
    # 1) 데이터 로드 — _load_trade_rows() 사용
    df = _load_trade_rows(data_dir, horizon)
    if df.empty:
        logger.warning("combo_optimizer: 매칭 데이터 없음")
        return {}

    matched_days = df.attrs.get("matched_days", 0)
    total_wr = df["win"].mean() * 100

    # 2) 조합 그리드 탐색 — _all_combos() + _evaluate_combo() 사용
    results = []
    _include_dq = bool(("DQ" in df.columns) and df["DQ"].any())  # [v24.2]
    for s_min, t_min, ai_min, routes, dq in _all_combos(_include_dq):
        stat = _evaluate_combo(df, s_min, t_min, ai_min, routes, min_samples,
                               dq_exclude=dq)
        if stat is None:
            continue
        results.append({
            "S_min": int(s_min),
            "T_min": int(t_min),
            "AI_min": int(ai_min),
            "routes": routes,
            "dq_exclude": int(dq),  # [v24.2] 1=무결성 플래그 제외 조합
            **stat,  # n, win_rate, avg_ret, avg_win, avg_loss, ev
        })

    if not results:
        logger.warning("combo_optimizer: 유효 조합 없음")
        return {}

    # 3) 정렬 — 승률 우선 + EV 우선 이중 랭킹
    results.sort(key=lambda x: (-x["win_rate"], -x["avg_ret"]))
    best_wr = results[0]
    top_by_wr = results[:top_n]

    results_ev = sorted(results, key=lambda x: (-x["ev"], -x["win_rate"]))
    best_ev = results_ev[0]
    top_by_ev = results_ev[:top_n]

    output = {
        "best_wr": best_wr,
        "best_ev": best_ev,
        "best": best_wr,  # 하위호환
        "top_combos": top_by_wr,
        "top_combos_ev": top_by_ev,
        "meta": {
            "total_trades": len(df),
            "total_win_rate": round(total_wr, 1),
            "matched_days": matched_days,
            "horizon": horizon,
            "min_samples": min_samples,
        },
    }

    # 4) 저장
    try:
        from datetime import datetime
        output["generated_at"] = datetime.now().isoformat()

        out_path = os.path.join(data_dir, "optimal_filter_latest.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        logger.info(
            f"🎯 최적(승률): S≥{best_wr['S_min']} T≥{best_wr['T_min']} AI≥{best_wr['AI_min']} "
            f"| 승률 {best_wr['win_rate']}% | EV {best_wr['ev']:+.2f}"
        )
        logger.info(
            f"🎯 최적(EV): S≥{best_ev['S_min']} T≥{best_ev['T_min']} AI≥{best_ev['AI_min']} "
            f"| EV {best_ev['ev']:+.2f} | 승률 {best_ev['win_rate']}%"
        )
    except Exception as e:
        logger.warning(f"optimal_filter 저장 실패: {e}")

    return output


# ═══════════════════════════════════════════════════
#  [v3.8.1 Phase 2-A] Walk-Forward 검증 함수 (신규)
# ═══════════════════════════════════════════════════

def run_combo_optimization_wf(
    data_dir: str,
    horizon: int = 3,
    min_samples: int = 10,
    top_n: int = 10,
    oos_ratio: float = 0.5,
    robust_threshold: float = 0.7,
    verbose: bool = True,
) -> dict:
    """
    조합 최적화 Walk-Forward 검증 버전.

    방법:
      1. 전체 거래 데이터를 시간순 정렬
      2. 앞 절반 (IS, In-Sample): "공부용" — 각 조합 성능 측정
      3. 뒤 절반 (OOS, Out-Of-Sample): "시험용" — 같은 조합 재현성 검증
      4. Robustness = OOS_EV / IS_EV 계산
         · 1.0에 가까우면 robust (진짜 실력)
         · 0.7 이하면 overfit 의심 (우연)
         · 음수는 IS에선 수익 OOS에선 손실 (명백한 overfit)

    Args:
        data_dir: data/ 디렉토리 경로
        horizon: 보유일수
        min_samples: 각 구간 최소 샘플 수
        top_n: 상위 N개 저장
        oos_ratio: OOS 비율 (기본 0.5 = 앞뒤 반반)
        robust_threshold: robust로 판정할 OOS/IS 비율 최소치
        verbose: True면 상세 로그 출력

    Returns:
        {
            "robust_combos": [...],       # ✅ 실전 권장 (IS>0, OOS>0, robust)
            "overfit_combos": [...],      # 🚨 IS 좋았는데 OOS 박살
            "recovering_combos": [...],   # 🌱 IS 약했는데 OOS 개선 (v3.8.1 신규)
            "weak_combos": [...],         # ⚠️ 애매 (둘 다 약함)
            "current_combo_wf": {...},    # 🛡️ 콤보 재검증
            "strong_approximation": {...},# 🏆 최강 근사 (S≥80, T≥70, AI≥50)
            "summary": {...},
            "meta": {...}
        }
    """
    # 1) 데이터 로드
    df = _load_trade_rows(data_dir, horizon)
    if df.empty:
        logger.warning("walk-forward: 매칭 데이터 없음")
        return {}

    matched_days = df.attrs.get("matched_days", 0)

    # 2) 시간순 정렬 후 IS/OOS 분할 (일자 단위 — 같은 날 거래는 같은 split에)
    df = df.sort_values("trade_date").reset_index(drop=True)
    unique_dates = sorted(df["trade_date"].unique())
    n_dates = len(unique_dates)
    if n_dates < 4:
        logger.warning(
            f"walk-forward: 날짜 부족 ({n_dates}일). "
            f"최소 4일 필요. 기존 run_combo_optimization() 사용 권장."
        )
        return {}

    split_date_idx = int(n_dates * (1 - oos_ratio))
    is_dates = set(unique_dates[:split_date_idx])
    oos_dates = set(unique_dates[split_date_idx:])

    is_df = df[df["trade_date"].isin(is_dates)].copy()
    oos_df = df[df["trade_date"].isin(oos_dates)].copy()

    if verbose:
        logger.info(
            f"[WF] 분할 완료 — "
            f"IS: {len(is_df)}건 ({len(is_dates)}일), "
            f"OOS: {len(oos_df)}건 ({len(oos_dates)}일)"
        )

    # 3) 모든 조합에 대해 IS/OOS 동시 평가
    _include_dq = bool(("DQ" in df.columns) and df["DQ"].any())  # [v24.2]
    combos = _all_combos(_include_dq)
    results = []

    for s_min, t_min, ai_min, routes, dq in combos:
        is_stat = _evaluate_combo(is_df, s_min, t_min, ai_min, routes, min_samples,
                                  dq_exclude=dq)
        oos_stat = _evaluate_combo(oos_df, s_min, t_min, ai_min, routes,
                                   min_samples=max(3, min_samples // 2),
                                   dq_exclude=dq)

        if is_stat is None or oos_stat is None:
            continue

        # [v3.8.1] Robustness 계산 + 4단 분류 (판정 정제)
        is_ev = is_stat["ev"]
        oos_ev = oos_stat["ev"]

        # Robustness 산출
        if is_ev > 0:
            # 정상 케이스: IS 성능 대비 OOS 재현율
            robustness = round(oos_ev / is_ev, 3)
        elif is_ev == 0 and oos_ev == 0:
            robustness = 1.0
        elif is_ev <= 0 and oos_ev > 0:
            # 특이 케이스: IS는 약했는데 OOS 개선
            # robustness 해석 불가 → None 저장, 카테고리로 분류
            robustness = None
        else:
            # IS 음수 AND OOS 비양성 → 전반적 나쁨
            robustness = 0.0

        # 4단 분류 (v3.8.1 정제 — RECOVERING 신규)
        # ✅ ROBUST:     is_ev > 0 AND oos_ev > 0 AND robustness >= threshold
        # 🚨 OVERFIT:    is_ev > 0 BUT oos_ev <= 0 (명백한 과적합)
        # 🌱 RECOVERING: is_ev <= 0 BUT oos_ev > 0 (최근 개선된 조합 — 주의깊은 관찰)
        # ⚠️ WEAK:       그 외 (둘 다 약하거나 robustness 중간)
        if is_ev > 0 and oos_ev > 0 and robustness is not None and robustness >= robust_threshold:
            category = "ROBUST"
            status = "✅ ROBUST"
            is_robust = True
        elif is_ev > 0 and oos_ev <= 0:
            category = "OVERFIT"
            status = "🚨 OVERFIT"
            is_robust = False
        elif is_ev <= 0 and oos_ev > 0:
            category = "RECOVERING"
            status = "🌱 RECOVERING"
            is_robust = False  # 확신 부족 → robust 아님
        else:
            category = "WEAK"
            status = "⚠️ WEAK"
            is_robust = False

        results.append({
            "S_min": int(s_min),
            "T_min": int(t_min),
            "AI_min": int(ai_min),
            "routes": routes,
            "dq_exclude": int(dq),  # [v24.2] 1=무결성 플래그 제외 조합
            "is": is_stat,
            "oos": oos_stat,
            "robustness": robustness,
            "is_robust": is_robust,
            "category": category,   # [v3.8.1] 명시적 카테고리
            "status": status,
        })

    if not results:
        logger.warning("walk-forward: 유효 조합 없음")
        return {}

    # 4) 카테고리별 분류 (v3.8.1 — 4단)
    robust_combos = [r for r in results if r["category"] == "ROBUST"]
    overfit_combos = [r for r in results if r["category"] == "OVERFIT"]
    recovering_combos = [r for r in results if r["category"] == "RECOVERING"]
    weak_combos = [r for r in results if r["category"] == "WEAK"]

    # 정렬
    robust_combos.sort(key=lambda x: -x["oos"]["ev"])
    overfit_combos.sort(key=lambda x: -x["is"]["ev"])  # IS 좋았는데 망한 순
    recovering_combos.sort(key=lambda x: -x["oos"]["ev"])  # OOS 좋은 순

    # 5) 현재 라벨 조건 재검증
    # 🛡️ 콤보: S≥90, T≥80, AI≥60, ROUTE=[ATTACK, ARMED]
    current_combo_wf = None
    for r in results:
        if (r["S_min"] == 90 and r["T_min"] == 80 and r["AI_min"] == 60
                and r["routes"] == ["ATTACK", "ARMED"]):
            current_combo_wf = r
            break

    # 🏆 최강은 조건이 다른 축(평균/밸런스)이라 이 그리드엔 직접 없음
    # 근사: S≥80, T≥70, AI≥50 — 현재 그리드에서 "높은 기준"에 해당
    # (문서/코드 일치: 이 값이 실제 코드 기준)
    strong_approximation = None
    for r in results:
        if (r["S_min"] == 80 and r["T_min"] == 70 and r["AI_min"] == 50):
            strong_approximation = r
            break

    # 6) 결과 구조화
    output = {
        "robust_combos": robust_combos[:top_n],
        "overfit_combos": overfit_combos[:top_n],
        "recovering_combos": recovering_combos[:top_n],  # [v3.8.1] 신규
        "weak_combos": weak_combos[:5],
        "current_combo_wf": current_combo_wf,
        "strong_approximation": strong_approximation,
        "summary": {
            "n_total": len(results),
            "n_robust": len(robust_combos),
            "n_overfit": len(overfit_combos),
            "n_recovering": len(recovering_combos),  # [v3.8.1]
            "n_weak": len(weak_combos),
            "robust_pct": round(len(robust_combos) / len(results) * 100, 1),
        },
        "meta": {
            "total_trades": len(df),
            "is_trades": len(is_df),
            "oos_trades": len(oos_df),
            "matched_days": matched_days,
            "is_dates": len(is_dates),
            "oos_dates": len(oos_dates),
            "horizon": horizon,
            "min_samples": min_samples,
            "oos_ratio": oos_ratio,
            "robust_threshold": robust_threshold,
        },
    }

    # 7) 저장
    try:
        from datetime import datetime
        output["generated_at"] = datetime.now().isoformat()

        out_path = os.path.join(data_dir, "combo_walkforward_latest.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"💾 walk-forward 리포트 저장: {out_path}")

        # 핵심 결과 로그
        if current_combo_wf:
            c = current_combo_wf
            logger.info(
                f"🛡️ 콤보 WF: IS EV {c['is']['ev']:+.2f}% → "
                f"OOS EV {c['oos']['ev']:+.2f}% "
                f"(Robustness {c['robustness']}, {c['status']})"
            )
    except Exception as e:
        logger.warning(f"walk-forward 리포트 저장 실패: {e}")

    return output


def print_wf_report(result: dict) -> None:
    """Walk-forward 결과 사람이 읽기 좋게 출력 (v3.8.1 — 4단 분류 대응)."""
    if not result:
        print("❌ 결과 없음")
        return

    meta = result.get("meta", {})
    summary = result.get("summary", {})

    print("\n" + "=" * 72)
    print("🔬 Walk-Forward 검증 리포트")
    print("=" * 72)
    print(f"총 거래: {meta.get('total_trades', 0)}건")
    print(f"  IS (앞):  {meta.get('is_trades', 0)}건 ({meta.get('is_dates', 0)}일)")
    print(f"  OOS (뒤): {meta.get('oos_trades', 0)}건 ({meta.get('oos_dates', 0)}일)")
    print(f"분석 조합: {summary.get('n_total', 0)}개")
    print(f"  ✅ ROBUST:     {summary.get('n_robust', 0)}개 ({summary.get('robust_pct', 0)}%)")
    print(f"  🚨 OVERFIT:    {summary.get('n_overfit', 0)}개  (IS>0, OOS<=0)")
    print(f"  🌱 RECOVERING: {summary.get('n_recovering', 0)}개  (IS<=0, OOS>0)")
    print(f"  ⚠️ WEAK:       {summary.get('n_weak', 0)}개  (둘 다 약함)")

    # 현재 🛡️ 콤보 재검증
    print("\n" + "-" * 72)
    print("🛡️ 현재 콤보 재검증 (S≥90 AND T≥80 AND AI≥60 AND ATTACK/ARMED)")
    print("-" * 72)
    cc = result.get("current_combo_wf")
    if cc:
        print(f"IS (공부):  n={cc['is']['n']:3} 승률 {cc['is']['win_rate']:5.1f}% EV {cc['is']['ev']:+6.2f}%")
        print(f"OOS (시험): n={cc['oos']['n']:3} 승률 {cc['oos']['win_rate']:5.1f}% EV {cc['oos']['ev']:+6.2f}%")
        r_str = f"{cc['robustness']}" if cc['robustness'] is not None else "N/A"
        print(f"Robustness: {r_str}  →  {cc['status']}")
        # v3.8.1: category 기반 결론
        category = cc.get("category", "UNKNOWN")
        if category == "ROBUST":
            print("✅ 결론: 🛡️ 콤보는 실전에서도 통하는 조합 — 현재 매매 전략 유지")
        elif category == "OVERFIT":
            print("🚨 결론: 🛡️ 콤보가 과적합 의심 — 조건 재검토 고려")
        elif category == "RECOVERING":
            print("🌱 결론: 최근 개선 추세 — 추가 관찰 후 판단")
        else:
            print("⚠️ 결론: 🛡️ 콤보 약한 성능 — 주의 필요")
    else:
        print("(현재 콤보 조건이 표본 부족으로 평가 불가)")

    # 🏆 최강 근사 진단
    sa = result.get("strong_approximation")
    if sa:
        print("\n" + "-" * 72)
        print("🏆 최강 근사 진단 (S≥80 AND T≥70 AND AI≥50)")
        print("-" * 72)
        print(f"IS (공부):  n={sa['is']['n']:3} 승률 {sa['is']['win_rate']:5.1f}% EV {sa['is']['ev']:+6.2f}%")
        print(f"OOS (시험): n={sa['oos']['n']:3} 승률 {sa['oos']['win_rate']:5.1f}% EV {sa['oos']['ev']:+6.2f}%")
        r_str = f"{sa['robustness']}" if sa['robustness'] is not None else "N/A"
        print(f"Robustness: {r_str}  →  {sa['status']}")
        category = sa.get("category", "UNKNOWN")
        if category == "ROBUST":
            print("✅ 최강 조건: 표본 쌓이면 부활 자격 있음")
        elif category == "OVERFIT":
            print("🚨 최강 조건: 표본 쌓아도 의미 없음 — 기준 재설계 필요")
        elif category == "RECOVERING":
            print("🌱 최강 조건: 추세 살아남 — 계속 관찰")
        else:
            print("⚠️ 최강 조건: 전반적으로 약함")

    # Robust 표 출력 보조 함수 (robustness None 안전 처리)
    def _fmt_robust(r_val):
        if r_val is None:
            return "    N/A"
        return f"{r_val:>7.2f}"

    # Robust Top 5
    print("\n" + "-" * 72)
    print("✅ ROBUST Top 5 (OOS EV 기준)")
    print("-" * 72)
    print(f"{'S':>3} {'T':>3} {'AI':>3} {'ROUTE':<20} {'IS_EV':>8} {'OOS_EV':>8} {'Robust':>8}")
    for r in result.get("robust_combos", [])[:5]:
        routes_str = "+".join(r["routes"])[:18]
        print(f"{r['S_min']:>3} {r['T_min']:>3} {r['AI_min']:>3} {routes_str:<20} "
              f"{r['is']['ev']:>+7.2f}% {r['oos']['ev']:>+7.2f}% {_fmt_robust(r['robustness'])}")

    # Overfit Top 5
    print("\n" + "-" * 72)
    print("🚨 OVERFIT Top 5 (IS 좋음 → OOS 손실)")
    print("-" * 72)
    print(f"{'S':>3} {'T':>3} {'AI':>3} {'ROUTE':<20} {'IS_EV':>8} {'OOS_EV':>8} {'Robust':>8}")
    for r in result.get("overfit_combos", [])[:5]:
        routes_str = "+".join(r["routes"])[:18]
        print(f"{r['S_min']:>3} {r['T_min']:>3} {r['AI_min']:>3} {routes_str:<20} "
              f"{r['is']['ev']:>+7.2f}% {r['oos']['ev']:>+7.2f}% {_fmt_robust(r['robustness'])}")

    # Recovering Top 5
    recovering = result.get("recovering_combos", [])
    if recovering:
        print("\n" + "-" * 72)
        print("🌱 RECOVERING Top 5 (IS 약함 → OOS 개선 — 추가 관찰 권장)")
        print("-" * 72)
        print(f"{'S':>3} {'T':>3} {'AI':>3} {'ROUTE':<20} {'IS_EV':>8} {'OOS_EV':>8} {'Robust':>8}")
        for r in recovering[:5]:
            routes_str = "+".join(r["routes"])[:18]
            print(f"{r['S_min']:>3} {r['T_min']:>3} {r['AI_min']:>3} {routes_str:<20} "
                  f"{r['is']['ev']:>+7.2f}% {r['oos']['ev']:>+7.2f}% {_fmt_robust(r['robustness'])}")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    mode = sys.argv[2] if len(sys.argv) > 2 else "wf"  # [v3.8.1] 기본을 wf로

    if mode == "wf":
        # [v3.8.1] Walk-forward 모드
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        result = run_combo_optimization_wf(data_dir)
        print_wf_report(result)
    else:
        # 기존 모드 (하위호환)
        result = run_combo_optimization(data_dir)
        if result:
            b = result["best_wr"]
            print(f"\n🎯 최적(승률): S≥{b['S_min']} T≥{b['T_min']} AI≥{b['AI_min']}")
            print(f"   ROUTE: {'+'.join(b['routes'])}")
            print(f"   {b['n']}건 | 승률 {b['win_rate']}% | 수익 {b['avg_ret']:+.2f}% | EV {b['ev']:+.2f}")

            e = result["best_ev"]
            print(f"\n💎 최적(EV): S≥{e['S_min']} T≥{e['T_min']} AI≥{e['AI_min']}")
            print(f"   ROUTE: {'+'.join(e['routes'])}")
            print(f"   {e['n']}건 | 승률 {e['win_rate']}% | 수익 {e['avg_ret']:+.2f}% | EV {e['ev']:+.2f}")
