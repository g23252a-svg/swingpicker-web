# -*- coding: utf-8 -*-
"""Production recommendation quality guard.

The legacy pipeline exposes several useful research signals, but historically a
high score or a TOP_PICK flag alone has not implied positive realised returns.
This module creates one conservative, auditable production contract:

    PRODUCTION_BUY == 1

Anything else is either WATCH or CASH.  The guard never promotes a candidate;
it can only keep or reject an already strict (TOP_PICK + BUY_NOW_ELIGIBLE)
candidate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


POLICY_VERSION = "alpha_gate_v32"
# [v32] ROUTE는 더 이상 진입 게이트가 아니다(ATTACK 알파 -2.9%p, p=0.0004 실측).
# 검증된 알파(ALPHA_GATE_ACTIVE)가 진입 SSOT. ACTIVE_ROUTES는 레거시 폴백
# (알파 미검증일)에서만 참조되며, ROUTE 거부권 자체는 evidence_pass에서 제거됨.
ACTIVE_ROUTES = frozenset({"ATTACK"})
DANGEROUS_MACRO = frozenset({"WARNING", "CRITICAL"})
# [v27] Realised-trade audit 2026-02..07 (n=157):
#  - POC_GAP <=20 win-rate 40-48%; >20 win-rate 12-23%.  Extension chase kills EV.
#  - MARKET_BREADTH >=35 combined with the POC gate flips the daily-pick
#    strategy from -2.2%/trade (33% win) to +2.3%/trade (50% win).
# Missing columns / NaN pass the gate (legacy CSV compatibility).
POC_GAP_MAX = 20.0
BREADTH_MIN = 35.0


def _num(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _num_nan(df: pd.DataFrame, column: str) -> pd.Series:
    """Numeric column preserving NaN (missing column -> all-NaN)."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _text(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[column].fillna(default).astype(str).str.strip().str.upper()


def _flag(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return (
        df[column]
        .fillna(0)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "1.0", "true", "t", "yes", "y"})
    )


def _alpha_gate_on(df: pd.DataFrame) -> bool:
    """[v32] 알파 전면 진입 게이트가 이 배치에서 활성인지 (컬럼 없으면 False)."""
    if df is None or len(df) == 0 or "ALPHA_GATE_ACTIVE" not in df.columns:
        return False
    col = pd.to_numeric(df["ALPHA_GATE_ACTIVE"], errors="coerce").fillna(0).astype(int)
    return bool(col.max() == 1)


def _quality_score(df: pd.DataFrame) -> pd.Series:
    """Transparent 0-100 score built only from entry-quality evidence.

    It is deliberately not another alpha model.  Its job is to rank near
    misses and make every rejection explainable.
    """
    score = pd.Series(35.0, index=df.index, dtype="float64")
    display = _num(df, "DISPLAY_SCORE", 0).clip(0, 100)
    buy_now_score = _num(df, "BUY_NOW_SCORE", 0).clip(0, 100)
    grade = _text(df, "BUY_NOW_GRADE")
    route = _text(df, "ROUTE")
    pick_type = _text(df, "TOP_PICK_TYPE")
    risk = _text(df, "ENTRY_RISK_LEVEL", "UNKNOWN")
    mfi = _num(df, "MFI14", 50)
    res_near = _num(df, "RES_RATIO_NEAR", 0)
    vwap = _num(df, "VWAP_GAP", 99)
    rr = _num(df, "RR_NOW_TP1", 0)

    score += display * 0.10
    score += np.select(
        [buy_now_score >= 80, buy_now_score >= 70, buy_now_score < 60],
        [10.0, 5.0, -10.0],
        default=0.0,
    )
    score += np.select(
        [grade.eq("BUY"), grade.eq("WATCH"), grade.eq("AVOID")],
        [15.0, 5.0, -35.0],
        default=-10.0,
    )
    score += np.select(
        [res_near >= 0.12, res_near >= 0.06, res_near < 0.03],
        [15.0, 8.0, -25.0],
        default=0.0,
    )
    score += np.select(
        [(mfi >= 70) & (mfi <= 85), (mfi >= 60) & (mfi < 70), (mfi < 55) | (mfi > 90)],
        [15.0, 8.0, -15.0],
        default=0.0,
    )
    score += np.select(
        [(vwap >= 4) & (vwap <= 16), vwap <= 16, vwap > 20],
        [15.0, 8.0, -20.0],
        default=0.0,
    )
    score += np.select(
        [rr >= 1.8, rr >= 1.3, rr < 1.1],
        [12.0, 6.0, -15.0],
        default=0.0,
    )
    score += np.select(
        [route.eq("ATTACK"), route.eq("ARMED"), ~route.isin(ACTIVE_ROUTES)],
        [8.0, 0.0, -10.0],
        default=0.0,
    )
    score -= pick_type.eq("AGGRESSIVE").astype(float) * 30.0
    score -= (~risk.eq("GREEN")).astype(float) * 20.0
    return score.clip(0, 100).round(1)


def _reasons(df: pd.DataFrame, production: pd.Series) -> pd.Series:
    top = _flag(df, "TOP_PICK")
    eligible = _flag(df, "PRE_QUALITY_BUY_NOW_ELIGIBLE")
    grade = _text(df, "BUY_NOW_GRADE")
    route = _text(df, "ROUTE")
    pick_type = _text(df, "TOP_PICK_TYPE")
    risk = _text(df, "ENTRY_RISK_LEVEL", "UNKNOWN")
    macro = _text(df, "MACRO_RISK", "NORMAL")
    mfi = _num(df, "MFI14", 50)
    res_near = _num(df, "RES_RATIO_NEAR", 0)
    vwap = _num(df, "VWAP_GAP", 99)
    rr = _num(df, "RR_NOW_TP1", 0)
    display = _num(df, "DISPLAY_SCORE", 0)
    buy_now_score = _num(df, "BUY_NOW_SCORE", 0)
    abnormal = _flag(df, "ABNORMAL_HISTORY_GUARD_FLAG")
    recovery_block = _flag(df, "PROFIT_RECOVERY_BLOCK_FLAG")
    july_block = _flag(df, "JULY_PROFIT_BLOCK_FLAG")
    fresh = _flag(df, "DATA_FRESHNESS_OK", True)
    evidence = _flag(df, "QUALITY_GUARD_PASS")
    poc_gap = _num_nan(df, "POC_GAP")
    breadth = _num_nan(df, "MARKET_BREADTH")
    regime_col = _text(df, "MARKET_REGIME", "")
    # [v32] 알파 게이트 활성 시 ROUTE·모멘텀 사유는 숨기고 알파 기준으로 설명.
    alpha_gate_active = _alpha_gate_on(df)
    ascore = _num_nan(df, "ALPHA_SCORE")
    athr = _num_nan(df, "ALPHA_ENTRY_THRESHOLD")
    # [v37] 저점추세 게이트 — 컬럼 없으면(구버전 CSV) 통과 취급.
    lt_ok = _flag(df, "ALPHA_LT_OK", True)
    ltp = _num_nan(df, "LOW_TREND_PCTL")   # [v46] 당일 분위(0~100) — 사유 문구용

    result: list[str] = []
    for i in range(len(df)):
        if bool(production.iloc[i]):
            result.append("엄격 매수 기준 통과")
            continue
        row_reasons: list[str] = []
        if bool(top.iloc[i]) and bool(eligible.iloc[i]) and bool(evidence.iloc[i]):
            row_reasons.append("당일 신규진입 1종목 제한")
        if not bool(top.iloc[i]):
            if alpha_gate_active:
                _a, _t = ascore.iloc[i], athr.iloc[i]
                if pd.notna(_a) and pd.notna(_t) and _a < _t:
                    row_reasons.append(f"알파 {_a:.0f}점 (진입선 {_t:.0f}점 미달)")
                elif not bool(lt_ok.iloc[i]):
                    # [v46] 알파는 통과했지만 저점추세가 당일 하위권 — 실측 역신호.
                    #   당일 하위30% 종목 엣지 -0.57%p·승률 26.9% (t=-2.19)
                    _p = ltp.iloc[i]
                    row_reasons.append(
                        f"저점추세 당일 하위 {_p:.0f}% (알파 통과·자리 미달)"
                        if pd.notna(_p) else "저점추세 하락 (알파 통과·자리 미달)")
                else:
                    row_reasons.append("진입 조건 미달")
            else:
                row_reasons.append("TOP_PICK 아님")
        elif not bool(eligible.iloc[i]):
            row_reasons.append("즉시매수 기준 미달")
        if not alpha_gate_active and grade.iloc[i] != "BUY":
            row_reasons.append(f"진입등급 {grade.iloc[i] or '없음'}")
        if not alpha_gate_active and buy_now_score.iloc[i] < 70:
            row_reasons.append(f"즉시매수점수 {buy_now_score.iloc[i]:.0f}")
        if not alpha_gate_active and pick_type.iloc[i] == "AGGRESSIVE":
            row_reasons.append("공격형 검증 실패")
        if not alpha_gate_active and risk.iloc[i] != "GREEN":
            row_reasons.append(f"진입위험 {risk.iloc[i] or 'UNKNOWN'}")
        if not alpha_gate_active and route.iloc[i] not in ACTIVE_ROUTES:
            row_reasons.append(f"경로 {route.iloc[i] or '없음'}")
        if macro.iloc[i] in DANGEROUS_MACRO:
            row_reasons.append(f"시장위험 {macro.iloc[i]}")
        if not alpha_gate_active and (mfi.iloc[i] < 70 or mfi.iloc[i] > 88):
            row_reasons.append(f"MFI {mfi.iloc[i]:.0f}")
        if not alpha_gate_active and res_near.iloc[i] < 0.13:
            row_reasons.append("상단여력 부족")
        if not alpha_gate_active and (vwap.iloc[i] < 4 or vwap.iloc[i] > 14):
            row_reasons.append("VWAP 적정구간 아님")
        if rr.iloc[i] < 1.3:
            row_reasons.append(f"손익비 {rr.iloc[i]:.2f}")
        if not alpha_gate_active and display.iloc[i] < 70:
            row_reasons.append(f"기본점수 {display.iloc[i]:.0f}")
        if bool(abnormal.iloc[i]):
            row_reasons.append("가격이력 이상")
        if bool(recovery_block.iloc[i]) or bool(july_block.iloc[i]):
            row_reasons.append("손실방어 차단")
        if not bool(fresh.iloc[i]):
            row_reasons.append("데이터 신선도 실패")
        _poc = poc_gap.iloc[i]
        if pd.notna(_poc) and _poc > POC_GAP_MAX:
            row_reasons.append(f"POC 확장 {_poc:.0f}% (추격 위험)")
        # [v34] 알파 게이트에선 시장폭/DOWN 레짐이 하드블록이 아니다(문턱 상향으로
        # 대응) — 탈락 '사유'로 표기하면 오도. 레거시 폴백에서만 사유로 남긴다.
        _br = breadth.iloc[i]
        if not alpha_gate_active and pd.notna(_br) and _br < BREADTH_MIN:
            row_reasons.append(f"시장폭 {_br:.0f}% (내부 약세)")
        if not alpha_gate_active and regime_col.iloc[i] == "DOWN":
            row_reasons.append("하락 레짐 (신규진입 차단)")
        result.append(" · ".join(row_reasons[:4]) or "품질점수 미달")
    return pd.Series(result, index=df.index, dtype="object")


def apply_recommendation_quality_guard(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the final loss-defense contract without ever promoting a row."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    if out.empty:
        for column in (
            "PRE_QUALITY_BUY_NOW_ELIGIBLE",
            "QUALITY_GUARD_SCORE",
            "QUALITY_GUARD_PASS",
            "QUALITY_GUARD_REASON",
            "PRODUCTION_BUY",
            "ACTION_DECISION",
            "RECOMMENDED_WEIGHT_PCT",
            "QUALITY_POLICY_VERSION",
        ):
            out[column] = pd.Series(dtype="object")
        return out

    top = _flag(out, "TOP_PICK")
    # Preserve the original legacy decision when the guard is applied more
    # than once (pipeline -> CSV -> data store).  This makes the operation
    # idempotent and keeps the audit trail intact.
    # [v32] 알파 게이트 활성 시 알파가 진입 SSOT이므로, 이전 배치에서 남은
    # stale PRE_QUALITY(ROUTE 게이트 산물)를 무시하고 현재 BUY_NOW_ELIGIBLE
    # (알파 게이트가 방금 재계산)를 권위값으로 쓴다.  그렇지 않으면 재실행 시
    # ROUTE 시절 0 값이 알파 픽을 부당하게 탈락시킨다(idempotency 버그).
    if _alpha_gate_on(out):
        eligible_source = "BUY_NOW_ELIGIBLE"
    else:
        eligible_source = (
            "PRE_QUALITY_BUY_NOW_ELIGIBLE"
            if "PRE_QUALITY_BUY_NOW_ELIGIBLE" in out.columns
            else "BUY_NOW_ELIGIBLE"
        )
    eligible_before = _flag(out, eligible_source)
    out["PRE_QUALITY_BUY_NOW_ELIGIBLE"] = eligible_before.astype(int)

    score = _quality_score(out)
    grade = _text(out, "BUY_NOW_GRADE")
    route = _text(out, "ROUTE")
    pick_type = _text(out, "TOP_PICK_TYPE")
    risk = _text(out, "ENTRY_RISK_LEVEL", "UNKNOWN")
    macro = _text(out, "MACRO_RISK", "NORMAL")
    mfi = _num(out, "MFI14", 50)
    res_near = _num(out, "RES_RATIO_NEAR", 0)
    vwap = _num(out, "VWAP_GAP", 99)
    rr = _num(out, "RR_NOW_TP1", 0)
    display = _num(out, "DISPLAY_SCORE", 0)
    buy_now_score = _num(out, "BUY_NOW_SCORE", 0)
    abnormal = _flag(out, "ABNORMAL_HISTORY_GUARD_FLAG")
    recovery_block = _flag(out, "PROFIT_RECOVERY_BLOCK_FLAG")
    july_block = _flag(out, "JULY_PROFIT_BLOCK_FLAG")
    fresh = _flag(out, "DATA_FRESHNESS_OK", True)

    # [v27] extension / market-internal gates — NaN passes (legacy compat)
    poc_gap = _num_nan(out, "POC_GAP")
    breadth = _num_nan(out, "MARKET_BREADTH")
    poc_ok = poc_gap.isna() | (poc_gap <= POC_GAP_MAX)
    breadth_ok = breadth.isna() | (breadth >= BREADTH_MIN)
    # [v28] regime gate — DOWN blocks new entries; missing column passes.
    regime = _text(out, "MARKET_REGIME", "")
    regime_ok = ~regime.eq("DOWN")
    # [v28] regime sizing — NEUTRAL/UNKNOWN halve the recommended weight.
    regime_mult = pd.to_numeric(
        out.get("REGIME_SIZE_MULT", pd.Series(1.0, index=out.index)),
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.0)

    # [v32] 알파 게이트 활성 여부 — 검증된 알파가 진입 SSOT일 때는
    # ROUTE·모멘텀 형태(MFI밴드/VWAP밴드/상단여력/DISPLAY≥70/ROUTE==ATTACK)
    # 필터를 걷어낸다. 이들은 실측 역상관(ELITE IC -0.06)이거나 ATTACK 편향이라
    # 알파 픽을 부당하게 탈락시킨다. 손실방어·매크로·신선도·확장·레짐 가드는 유지.
    alpha_gate_active = _alpha_gate_on(out)

    # 손실방어·데이터·매크로 공통 가드 (양 경로 공통)
    common_guard = (
        ~macro.isin(DANGEROUS_MACRO)
        & ~abnormal
        & ~recovery_block
        & ~july_block
        & fresh
        & poc_ok
        & (rr >= 1.30)
    )

    if alpha_gate_active:
        # 알파 진입 통과분(=새 TOP_PICK)에 공통 가드만 추가.
        # [v32] breadth_ok / regime_ok 하드블록은 알파 경로에서 제외한다:
        #   · risk_off 하드블록은 apply_alpha_entry_gate가 NEW_ENTRY_BLOCKED로 이미 강제.
        #   · 내부약세(breadth<35=DOWN)는 실측상 알파 최상위 픽이 +0.84%/승률51%로 양호 —
        #     하드블록 대신 레짐 적응형 문턱(하락 상위10%)과 사이즈 축소로 대응.
        alpha_ok = _flag(out, "ALPHA_ENTRY_OK")
        evidence_pass = alpha_ok & common_guard
    else:
        # 레거시 폴백 — 알파 미검증일 때만. ROUTE 거부권은 제거(역신호),
        # 대신 기존 모멘텀형 근거로 near-miss를 걸러 보수적으로 유지.
        evidence_pass = (
            grade.eq("BUY")
            & (buy_now_score >= 70)
            & ~pick_type.eq("AGGRESSIVE")
            & risk.eq("GREEN")
            & common_guard
            & breadth_ok
            & regime_ok
            & (mfi >= 70)
            & (mfi <= 88)
            & (res_near >= 0.13)
            & (vwap >= 4)
            & (vwap <= 14)
            & (display >= 70)
            & (score >= 70)
        )
    production_candidates = top & eligible_before & evidence_pass
    # [v35] 하루 1종목 랭킹 — 알파 게이트 활성 시 검증 알파×손익비로 선정.
    # 반사실 실측(46일, 후보풀=리스크가드+알파 상위 20%, 손절 -8%캡 실현수익):
    #   품질점수 랭킹(기존): 평균 -1.29%/일 · 승률 33% · 누적 -59.5%  ← 역선택
    #   알파 1등:            평균 +1.90%/일 · 승률 43% · 누적 +87.2%
    #   알파×손익비(채택):    평균 +2.07%/일 · 승률 46% · 누적 +95.1%
    # 품질점수에 포함된 DISPLAY(모멘텀 잔재, IC -0.03)가 풀 내 최악을 골라내던
    # 구조. 단독 t=1.35(p=0.18, n=46)로 통계 확정은 아니나 검증 축과 방향 일치.
    if alpha_gate_active:
        _rank_alpha = pd.to_numeric(
            out.get("ALPHA_SCORE", pd.Series(0, index=out.index)), errors="coerce"
        ).fillna(0.0)
        _rank_rr = pd.to_numeric(
            out.get("RR_NOW_TP1", pd.Series(0, index=out.index)), errors="coerce"
        ).fillna(0.0).clip(0.0, 3.0)
        # 알파×RR 주축 + 품질점수 미세 타이브레이크
        rank_key = _rank_alpha * _rank_rr * 1_000 + score
    else:
        rank_key = score * 1_000 + display
    production = production_candidates & rank_key.where(
        production_candidates
    ).rank(method="first", ascending=False).eq(1)

    out["QUALITY_GUARD_SCORE"] = score
    out["QUALITY_GUARD_PASS"] = evidence_pass.astype(int)
    out["PRODUCTION_BUY"] = production.astype(int)
    out["QUALITY_GUARD_REASON"] = _reasons(out, production)
    out["QUALITY_POLICY_VERSION"] = POLICY_VERSION

    # Keep the legacy official flag safe for all old consumers.  The original
    # value remains available in PRE_QUALITY_BUY_NOW_ELIGIBLE for audits.
    out["BUY_NOW_ELIGIBLE"] = production.astype(int)

    watch = (~production) & (score >= 60) & ~macro.isin(DANGEROUS_MACRO)
    out["ACTION_DECISION"] = np.select(
        [production, watch], ["BUY", "WATCH"], default="CASH"
    )
    # [v28] 레짐 사이징: UP=100%, NEUTRAL/UNKNOWN=50%, DOWN=0 (진입 자체 차단됨)
    out["RECOMMENDED_WEIGHT_PCT"] = np.where(
        production,
        np.where(score >= 85, 5.0, 3.0) * regime_mult,
        0.0,
    )
    return out


def production_buy_mask(df: pd.DataFrame) -> pd.Series:
    """Read the strict contract, with a conservative legacy fallback."""
    if df is None or len(df) == 0:
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)
    if "PRODUCTION_BUY" in df.columns:
        return _flag(df, "PRODUCTION_BUY")
    return _flag(df, "TOP_PICK") & _flag(df, "BUY_NOW_ELIGIBLE")
