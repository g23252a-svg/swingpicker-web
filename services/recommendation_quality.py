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

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("recommendation_quality")


# [v54] 공식 매수 계약이 바뀌었으므로 버전을 올린다 — 알파 진입 SSOT(v32)는
# 그대로이고, 여기에 '공식 매수는 실행 가능하다'가 추가됐다. CSV에 남는
# QUALITY_POLICY_VERSION으로 어느 계약이 그 행을 만들었는지 감사할 수 있고,
# daily_briefing의 production_buy_sizable 게이트가 구 계약 산출물을 오판하지
# 않도록(SKIP) 구분하는 데도 쓰인다.
POLICY_VERSION = "alpha_gate_v57"
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


# [v54] 사이징이 0으로 강제되는 ROUTE — kelly_calibrator와 같은 집합이어야 한다.
# 실제 거부권은 두 곳에서 걸린다:
#   · kelly_calibrator.resize_kelly_with_alpha (알파 게이트 활성 경로, 6종)
#   · collector.apply_kelly_betting (레거시 경로, NEUTRAL 제외 5종)
# 후자는 전자의 부분집합이므로 상위집합 하나로 판정한다(보수적 방향).
# 값 자체는 kelly_calibrator._KELLY_INACTIVE_ROUTES를 지연 import로 읽어
# SSOT를 유지하고, import 실패 시에만 아래 리터럴로 폴백한다.
# test_v54_first_pick_contradictions.py가 두 집합의 동일성을 검사한다.
_KELLY_INACTIVE_ROUTES_FALLBACK = frozenset(
    {"WAIT", "OVERHEAT", "EXIT_WARNING", "CARRY", "BLOCKED", "NEUTRAL"})


def _inactive_routes() -> frozenset:
    try:  # 순환 import 회피 — 함수 내부에서만 참조
        from kelly_calibrator import _KELLY_INACTIVE_ROUTES

        routes = frozenset(str(r).strip().upper() for r in _KELLY_INACTIVE_ROUTES)
        if routes:
            return routes
        logger.warning(
            "kelly_calibrator._KELLY_INACTIVE_ROUTES가 비어 있음 → 폴백 리터럴 사용")
    except Exception as e:
        # 폴백이 있으므로 판정은 계속되지만, SSOT가 끊긴 사실은 남긴다.
        logger.warning(f"_KELLY_INACTIVE_ROUTES 참조 실패 ({e}) → 폴백 리터럴 사용")
    return _KELLY_INACTIVE_ROUTES_FALLBACK


def _alpha_exempt_routes() -> frozenset:
    """[v57] 알파 진입 통과 시 사이징 0 강제를 면제하는 ROUTE (SSOT: kelly_calibrator)."""
    try:  # 순환 import 회피 — 함수 내부에서만 참조
        from kelly_calibrator import _ALPHA_EXEMPT_INACTIVE_ROUTES

        return frozenset(str(r).strip().upper() for r in _ALPHA_EXEMPT_INACTIVE_ROUTES)
    except Exception as e:
        logger.warning(f"_ALPHA_EXEMPT_INACTIVE_ROUTES 참조 실패 ({e}) → 폴백 사용")
        return frozenset({"WAIT"})


def _route_sizable_mask(df: pd.DataFrame) -> pd.Series:
    """ROUTE가 신규 사이징을 허용하는가 (컬럼·값 없으면 통과 — 레거시 호환).

    [v57] 알파 진입을 통과한 관망(WAIT)은 사이징이 막히지 않는다.
    kelly_calibrator.kelly_route_zero_mask와 **같은 규칙**이어야 한다 —
    어긋나면 v54가 없앤 '공식 매수인데 0주'가 되살아난다.
    근거·측정치는 kelly_calibrator의 _ALPHA_EXEMPT_INACTIVE_ROUTES 주석에 있다.
    """
    if "ROUTE" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    route = _text(df, "ROUTE", "")
    sizable = route.eq("") | ~route.isin(_inactive_routes())
    exempt = _alpha_exempt_routes()
    if exempt and "ALPHA_ENTRY_OK" in df.columns:
        sizable = sizable | (_flag(df, "ALPHA_ENTRY_OK") & route.isin(exempt))
    return sizable


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
    # [v56] 알파 게이트가 직접 남긴 탈락 사유 (SSOT). 구 CSV엔 없으므로 폴백 유지.
    gate_reason = (df["ALPHA_ENTRY_BLOCK_REASON"].fillna("").astype(str)
                   if "ALPHA_ENTRY_BLOCK_REASON" in df.columns
                   else pd.Series("", index=df.index, dtype="object"))
    blocked_flag = (_flag(df, "NEW_ENTRY_BLOCKED")
                    | _flag(df, "JULY_PROFIT_BLOCK_FLAG")
                    | _flag(df, "PROFIT_RECOVERY_BLOCK_FLAG"))
    # [v54] 사이징 불가(ROUTE)로 탈락한 경우 '1종목 제한'이라 적으면 거짓 사유가 된다.
    sizable = _route_sizable_mask(df)

    # [v57] 관망(WAIT) 상태인데 공식 매수인 경우가 생긴다 — 화면의 ROUTE 배지는
    #   여전히 '⏸️ 관망'을 띄우므로, 설명 없이 두면 v54가 고친 모순
    #   ('매수'와 '관망'을 같은 종목에 동시 표시)으로 되읽힌다. 차이는 이렇다:
    #   v54의 모순은 **살 수 없는 것**(켈리 0주)을 매수라 부른 것이고,
    #   지금은 **살 수 있다**(사이징 유효) — 상태 라벨과 진입 판단의 출처가 다를 뿐이다.
    #   그래서 사유 줄에 출처를 적는다.
    exempt_routes = _alpha_exempt_routes()
    alpha_entry = _flag(df, "ALPHA_ENTRY_OK")

    result: list[str] = []
    for i in range(len(df)):
        if bool(production.iloc[i]):
            if (route.iloc[i] in exempt_routes) and bool(alpha_entry.iloc[i]):
                result.append(
                    f"엄격 매수 기준 통과 (상태는 {route.iloc[i]}이지만 "
                    f"진입 판단은 검증 알파가 기준 — v32)")
            else:
                result.append("엄격 매수 기준 통과")
            continue
        row_reasons: list[str] = []
        if bool(top.iloc[i]) and bool(eligible.iloc[i]) and bool(evidence.iloc[i]):
            if not bool(sizable.iloc[i]):
                row_reasons.append(
                    f"상태 {route.iloc[i] or '없음'} — 신규진입 상태 아님(0주)")
            else:
                row_reasons.append("당일 신규진입 1종목 제한")
        if not bool(top.iloc[i]):
            if alpha_gate_active:
                # [v56] 게이트가 남긴 사유를 그대로 쓴다 — 재추론하면 거짓이 된다.
                #   2026-08-06 실측: 알파 100.0점(문턱 85)인 흥구석유가 아래
                #   폴백을 타 '진입 조건 미달' → 화면에서 'AI 알파 진입 기준 미달'로
                #   번역됐다. 실제 차단자는 risk_off 전 종목 하드블록(278/278)이었다.
                _gr = str(gate_reason.iloc[i] or "").strip()
                if _gr:
                    row_reasons.append(_gr)
                else:
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
                    elif bool(blocked_flag.iloc[i]):
                        # 구 CSV(게이트 사유 컬럼 없음)에서도 최상위 차단자는 말한다.
                        row_reasons.append("시장 전체 신규진입 차단 (폭락 방어)")
                    else:
                        row_reasons.append("진입 조건 미달 (알파·자리 외 사유)")
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
    # [v54] 실행 가능성을 공식 매수의 조건으로 넣는다 — SSOT 불일치 수정.
    #
    # 결함: PRODUCTION_BUY는 여기서 ROUTE를 보지 않고 정해지는데,
    # kelly_calibrator.resize_kelly_with_alpha는 _KELLY_INACTIVE_ROUTES에 걸리면
    # 켈리 수량을 0으로 강제한다. 두 곳이 서로 모르고 각자 판정하므로
    # '공식 매수 · 엄격 매수 기준 통과 · 최대 2% 비중'으로 표시된 종목이
    # 동시에 '0주 · 추천금액 0만원'이 되는 모순이 발생했다.
    #   실측 사례 024060 흥구석유 20260805:
    #     PRODUCTION_BUY=1 · BUY_NOW_ELIGIBLE=1 · TOP_PICK=1
    #     ROUTE=WAIT → 켈리_수량 0 · 추천금액 0만원 (POSITION_PCT는 100%)
    #   사용자가 첫 실전 픽에서 바로 발견했다.
    #
    # 고치는 방법이 둘 있고, 수익 실측으로는 갈리지 않았다.
    #   (A) 켈리의 ROUTE 거부권 제거 → 픽이 실제로 매수 가능해진다
    #   (B) 사이징 불가 상태를 공식 매수라 부르지 않는다 → 픽이 줄어든다
    # 처음에는 (B)를 지지하는 실측을 얻었다고 판단했다(게이트 통과 풀 1,495행·
    # 69일에서 사이징 가능 +4.09% vs 차단 -1.13%). 그러나 분해해 보니 이 근거는
    # 이 프로젝트의 채택 기준을 통과하지 못한다:
    #   · ATTACK n=9 — v32가 실제로 검정한 축(ATTACK 알파 -2.9%p, p=0.0004)은
    #     지금 풀에서 표본이 없어 재현도 반증도 불가하다. 즉 v54 측정은
    #     v32와 모순된 것이 아니라 다른 것(대부분 ARMED)을 본 것이다.
    #   · +4%는 사실상 ARMED 효과(n=46·18일, t=+1.16, p=0.26)이고
    #     IS만 표본이 되며 OOS는 검정 불가 — 'IS/OOS 양쪽 유의' 기준 미달.
    #   · 교란이 크다: 사이징 가능 행은 RR이 낮고(1.96 vs 3.32, t=-8.63)
    #     POC 확장이 덜하다(-1.02 vs -7.90, t=+4.88). 층화하면 셀이 비어버린다.
    #   · 날짜 클러스터 부트 CI95 [-1.82, +15.78] — 0을 포함, 에피소드 6건.
    # 따라서 수익 근거는 채택 사유가 아니다(이 주석에 남겨 재사용을 막는다).
    #
    # (B)를 고른 이유는 위험 비대칭이다. (A)는 검증된 적 없는 방향으로 실제
    # 자금을 태운다 — 켈리의 ROUTE 0원 정책 자체가 v32에서 '표시 관례'로
    # 유지된 미검증 규칙이고(kelly_calibrator.resize_kelly_with_alpha 주석),
    # 그 상태(WAIT=타이밍 미도래)에 자금을 넣어도 되는지는 아무도 측정하지
    # 않았다. (B)는 새 위험을 만들지 않고, 사이징 엔진이 자금을 거부하는 것을
    # 화면이 '공식 매수'라 부르지 않게만 한다. 되돌리기도 한 줄이다.
    # v32의 결론은 보존된다: ROUTE는 여전히 근거(evidence_pass)가 아니고
    # QUALITY_GUARD_PASS에 영향을 주지 않는다 — 실행 가능성 판정에만 쓴다.
    #
    # 비용은 정직하게 적는다: 같은 풀 기준 공식 매수가 나오는 날이 69일 중
    # 18일(26%)로 줄어든다. 나머지 날은 후보가 '관망 + 0주 사유'로 표시된다
    # (v53 배선) — 살 수 없는 것을 공식 매수로 적는 것보다 정직하다.
    #
    # ── [v55 재측정 — 거부권을 지지하는 근거는 없다. 그래도 제거하지 않는다.] ──
    # v54는 느슨한 풀(알파 상위10% 고정)에서 측정했다. v55는 프로덕션 퍼널을
    # 그대로 재현해(레짐별 알파 문턱 85/90 · LT분위>30 · v51 리스크게이트 ·
    # 품질가드 RR>=1.30·POC·신선도·매크로) 다시 봤다:
    #   거부권 없음 29일 픽 · 있음 15일 픽 · 거부권이 지운 14일(에피소드 3건)
    #   공통일에는 두 계약의 픽이 사실상 같다(15일 페어드 +1.43%p, t=0.19, p=0.85)
    #   → 거부권의 가치는 '지워버리는 날'에만 있는데, 그 날들은 손실이 아니었다:
    #        지워진 14일 평균 +0.23% · 중위 -0.14% · 승률 42.9%
    #        0 대비 단일표본 t=+0.15 p=0.88  ← 손실을 막아준 게 아니다
    #   유지군의 우위(+6.20%)는 로또 2장이 전부다:
    #        상위 2건 +78.6%/+52.0% · 중위 -2.70% · sd 25.18 · 왜도 2.10
    #        최상위 1건 제거 → +1.03% · 2건 제거 → -2.89%
    #        양쪽 최상위 1건씩 제외 후 비교 +1.03% vs -1.04% (t=+0.48, p=0.64)
    #        중위 기준 Mann-Whitney p=0.63 (오히려 유지군 중위가 더 나쁨)
    #   구조적 결함도 확인됐다 — 거부권은 사실상 '상승장에 매매 금지'다:
    #        유지일 UP 20% vs 지워진일 UP 71% · 시장폭 46.1 vs 58.1 (t=-1.93, p=0.064)
    #        풀 내 사이징 가능 비율: UP 2.8% · NEUTRAL 16.5% · DOWN 10.7%
    #   다지평도 일관되지 않다(FWD20은 부호 반전 -5.30%p).
    #
    # 그런데도 제거하지 않는 이유: 제거가 더 낫다는 것도 증명되지 않았다
    # (두 계약 모두 기대값이 0과 구별되지 않는다 — 기준 t=1.65 p=0.11 /
    #  거부권 t=0.95 p=0.36). 하루 만에 반대 방향으로 뒤집을 근거가 아니고,
    # 제거는 실제 자금이 들어가는 변경이다(켈리 0주 해제). 대신 조건을 못박는다:
    #   재검증 조건 — 거부권이 지운 날 20에피소드(현재 3건) 이상 쌓이면
    #   같은 검정을 다시 하고, 그때도 '지워진 날이 손실이 아님'이 유지되면 제거한다.
    #   그 전에 이 주석의 수익 수치를 근거로 제거·유지를 주장하면 안 된다.
    route_sizable = _route_sizable_mask(out)
    production_candidates = top & eligible_before & evidence_pass & route_sizable
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


def awaiting_execution_mask(df: pd.DataFrame) -> pd.Series:
    """[v54] 근거는 통과했으나 상태(ROUTE) 때문에 사이징이 0인 후보.

    v54가 '살 수 없는 것을 공식 매수라 부르지 않기'로 바꾼 대가로, 공식 매수가
    0개인 날이 늘어난다. 그 날 화면이 그냥 '현금'만 말하면 사용자는 엔진이
    아무것도 찾지 못한 것으로 읽는다 — 실제로는 종목은 찾았고 타이밍 상태가
    아직 아닌 것이다. 그 차이를 화면에 넘기기 위한 마스크.
    """
    if df is None or len(df) == 0:
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)
    evidence = (_flag(df, "QUALITY_GUARD_PASS")
                if "QUALITY_GUARD_PASS" in df.columns
                else pd.Series(False, index=df.index, dtype=bool))
    eligible = _flag(df, "PRE_QUALITY_BUY_NOW_ELIGIBLE") if \
        "PRE_QUALITY_BUY_NOW_ELIGIBLE" in df.columns else _flag(df, "BUY_NOW_ELIGIBLE")
    return (_flag(df, "TOP_PICK") & eligible & evidence
            & ~_route_sizable_mask(df) & ~production_buy_mask(df))


def production_buy_mask(df: pd.DataFrame) -> pd.Series:
    """Read the strict contract, with a conservative legacy fallback."""
    if df is None or len(df) == 0:
        return pd.Series(False, index=getattr(df, "index", None), dtype=bool)
    if "PRODUCTION_BUY" in df.columns:
        return _flag(df, "PRODUCTION_BUY")
    return _flag(df, "TOP_PICK") & _flag(df, "BUY_NOW_ELIGIBLE")
