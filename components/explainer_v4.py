# -*- coding: utf-8 -*-
"""
components/explainer_v4.py — v4.0 회원용 '쉬운 설명' 프론트엔드 (상업용 톤)

목적:
  유료 구독자가 전문용어 없이 '오늘 뭘 해야 하는지'를 즉시 이해하도록,
  엔진의 기술 출력(ELITE_SCORE / EST_WIN_RATE / RR / Kelly / MACRO_REGIME)을
  평이한 한국어로 번역해 카드로 보여준다.

설계 원칙:
  - 전문용어 0개. 승률→"10번 중 약 N번", RR→"손실 1 감수 시 기대 수익 X", 비중→권장 비중.
  - 표현 안전: '매도/팔아라/자동매매' 단정 금지 (회사 계약 준수). '검토'·'권장'까지만.
  - 과장 금지: 승률은 과거 패턴 빈도이며 미래 보장이 아님을 항상 병기.
  - build_member_view()는 순수 함수(테스트 가능), render_*()는 Streamlit 표시 전용.

pytest tests/test_explainer_v4.py -v  (선택)
"""
from __future__ import annotations

from typing import Any, Dict, Optional


# ── 레짐 → 평이한 시장 상태 문구 ────────────────────────────────────────────
_REGIME_KO: Dict[str, Dict[str, str]] = {
    "NORMAL":                    {"tone": "🟢", "label": "평상 구간",   "msg": "시장이 무난한 흐름이에요. 평소 기준대로 봐도 좋습니다."},
    "FX_HIGH_REGIME":            {"tone": "🟡", "label": "환율 부담",   "msg": "환율이 높아 변동성이 커질 수 있어요. 평소보다 비중을 줄이는 걸 권장합니다."},
    "FX_HIGH_AND_INTERNAL_WEAK": {"tone": "🟠", "label": "방어 구간",   "msg": "환율은 높고 시장 힘은 약한 방어 구간이에요. 무리한 신규 진입보다 관망이 유리합니다."},
    "MACRO_WARNING":             {"tone": "🟠", "label": "주의 구간",   "msg": "시장 위험 신호가 있어요. 진입은 신중하게, 비중은 작게 가져가는 걸 권장합니다."},
    "MACRO_CRITICAL":            {"tone": "🔴", "label": "관망 권장",   "msg": "시장 위험이 큰 날이에요. 오늘은 신규 진입을 쉬는 걸 권장합니다."},
}

# 레짐별 권장 비중 계수 (기둥 4와 동일 — 표시용)
_REGIME_SIZE_MULT: Dict[str, float] = {
    "NORMAL": 1.0, "FX_HIGH_REGIME": 0.6, "MACRO_WARNING": 0.5,
    "FX_HIGH_AND_INTERNAL_WEAK": 0.35, "MACRO_CRITICAL": 0.0,
}


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _winrate_phrase(p: float, basis: str = "excess") -> str:
    """0.0~1.0 승률 → 평이 문구 (과거 빈도, 보장 아님).

    basis="excess"(v4 기본): 시장 평균 대비 초과 빈도 — 정직하고 과대표기 아님.
    basis="absolute": 절대 상승 빈도(상승장 인플레 위험) — 권장하지 않음.
    """
    n = max(0, min(10, round(p * 10)))
    if basis == "excess":
        return f"과거 비슷한 상황에서 시장 평균보다 나은 결과가 10번 중 약 {n}번이었어요"
    return f"과거 비슷한 상황에서 10번 중 약 {n}번 수익으로 끝났어요"


def _rr_phrase(rr: float) -> str:
    if rr <= 0:
        return "손익비 정보가 충분하지 않아요"
    return f"손실 위험 1만큼을 감수할 때, 기대 수익은 약 {rr:.1f}배 수준이에요"


def _strength_phrase(score: float) -> str:
    if score >= 85:
        return "추천 강도: 매우 강함"
    if score >= 75:
        return "추천 강도: 강함"
    if score >= 65:
        return "추천 강도: 보통"
    return "추천 강도: 약함"


# ─────────────────────────────────────────────────────────────────────────────
# 순수 함수: 엔진 row + 시장 컨텍스트 → 회원용 표시 dict
# ─────────────────────────────────────────────────────────────────────────────
def build_member_view(row: Optional[Dict[str, Any]], market_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """엔진 출력을 회원용 평이 설명 dict로 변환한다.

    row: 추천 1종목의 컬럼 dict (없으면 '오늘 공식 추천 없음' 카드)
    market_ctx: {"regime": MACRO_REGIME_MODE, "official_count": int}
    """
    regime = str(market_ctx.get("regime", "NORMAL")).upper()
    rinfo = _REGIME_KO.get(regime, _REGIME_KO["NORMAL"])
    size_mult = _REGIME_SIZE_MULT.get(regime, 1.0)
    official_count = int(market_ctx.get("official_count", 0) or 0)

    market_card = {
        "tone": rinfo["tone"],
        "title": f"오늘의 시장: {rinfo['label']}",
        "message": rinfo["msg"],
        "size_hint_pct": int(round(size_mult * 100)),
    }

    # ── 공식 추천이 없는 날 ──
    if row is None or official_count == 0:
        verdict = "오늘은 '강하게 사라'고 말할 종목이 없어요"
        sub = ("기준을 통과한 공식 추천이 0개예요. 좋은 종목이 없을 땐 "
               "억지로 사지 않고 기다리는 것도 전략입니다.")
        if regime in ("FX_HIGH_AND_INTERNAL_WEAK", "MACRO_WARNING", "MACRO_CRITICAL"):
            sub += " 특히 오늘은 시장이 방어/주의 구간이라 쉬어가기 좋은 날이에요."
        return {
            "has_pick": False,
            "market": market_card,
            "verdict": verdict,
            "verdict_sub": sub,
            "disclaimer": _DISCLAIMER,
        }

    # ── 공식 추천이 있는 날 ──
    name = str(row.get("종목명", "추천 종목"))
    price = _f(row.get("추천매수가", row.get("종가")))
    stop = _f(row.get("손절가"))
    tp1 = _f(row.get("추천매도가1"))
    score = _f(row.get("DISPLAY_SCORE", row.get("ELITE_SCORE")))
    # v4 캘리브레이션 우선, 없으면 기존
    wr = _f(row.get("EST_WIN_RATE_V4", row.get("EST_WIN_RATE")))
    rr = _f(row.get("RR_EXPECTED", row.get("RR_NOW_TP1")))
    n_val = int(_f(row.get("EST_WIN_RATE_V4_N", row.get("EST_WIN_RATE_N"))))
    # 권장 금액: 레짐 계수 반영(표시용)
    base_amt = _f(row.get("추천금액(만원)"))
    rec_amt = round(base_amt * size_mult, 1) if base_amt > 0 else None

    up_pct = round((tp1 / price - 1) * 100, 1) if price > 0 and tp1 > 0 else None
    dn_pct = round((1 - stop / price) * 100, 1) if price > 0 and stop > 0 else None

    why = []
    if score >= 75:
        why.append(_strength_phrase(score))
    if wr > 0:
        # v4 excess 컬럼이 있으면 시장 대비 문구, 아니면 절대 문구
        _basis = "excess" if ("EST_WIN_RATE_V4" in row and row.get("EST_WIN_RATE_V4") not in (None, "")) else "absolute"
        why.append(_winrate_phrase(wr, basis=_basis))
    if rr > 0:
        why.append(_rr_phrase(rr))

    plan = []
    if price > 0:
        plan.append(f"진입 검토가: 약 {int(price):,}원")
    if up_pct is not None:
        plan.append(f"1차 목표까지 여유: +{up_pct}%")
    if dn_pct is not None:
        plan.append(f"손절 기준까지: -{dn_pct}% (여기 닿으면 정리 검토)")
    if rec_amt is not None:
        plan.append(f"권장 비중(시장 상태 반영): 약 {rec_amt}만원")

    trust = (f"이 판단은 과거 {n_val:,}건의 비슷한 사례로 검증된 패턴이에요"
             if n_val >= 20 else
             "아직 검증 표본이 적어요. 참고용으로만 봐주세요")

    return {
        "has_pick": True,
        "market": market_card,
        "name": name,
        "headline": f"오늘의 후보: {name}",
        "why": why,
        "plan": plan,
        "trust": trust,
        "disclaimer": _DISCLAIMER,
    }


_DISCLAIMER = ("ⓘ 이 화면은 투자 권유가 아니라 데이터 기반 참고 정보예요. "
               "표시된 승률은 과거 패턴의 빈도이며 미래 수익을 보장하지 않습니다. "
               "최종 매매 판단과 책임은 회원 본인에게 있습니다.")


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit 렌더 (표시 전용)
# ─────────────────────────────────────────────────────────────────────────────
def render_member_card(row: Optional[Dict[str, Any]], market_ctx: Dict[str, Any]) -> None:
    import streamlit as st

    view = build_member_view(row, market_ctx)
    m = view["market"]

    # 시장 상태 배너
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:14px;background:#f5f7fb;"
        f"border:1px solid #e6e9f0;margin-bottom:14px'>"
        f"<div style='font-size:15px;font-weight:700;color:#1f2937'>{m['tone']} {m['title']}</div>"
        f"<div style='font-size:13px;color:#4b5563;margin-top:4px'>{m['message']}</div>"
        f"<div style='font-size:12px;color:#6b7280;margin-top:6px'>오늘 권장 진입 비중: "
        f"평소의 약 {m['size_hint_pct']}%</div></div>",
        unsafe_allow_html=True,
    )

    if not view["has_pick"]:
        st.markdown(
            f"<div style='padding:20px;border-radius:14px;background:#fff;"
            f"border:1px solid #e6e9f0'>"
            f"<div style='font-size:18px;font-weight:800;color:#111827'>🟠 {view['verdict']}</div>"
            f"<div style='font-size:14px;color:#4b5563;margin-top:8px;line-height:1.6'>"
            f"{view['verdict_sub']}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='font-size:20px;font-weight:800;color:#111827;margin-bottom:6px'>"
            f"🟢 {view['headline']}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**왜 이 종목인가요?**")
            for w in view["why"]:
                st.markdown(f"- {w}")
        with c2:
            st.markdown("**어떻게 접근하나요?**")
            for p in view["plan"]:
                st.markdown(f"- {p}")
        st.caption(f"🔎 {view['trust']}")

    st.caption(view["disclaimer"])


def render_from_dataframe(df, market_ctx: Dict[str, Any]) -> None:
    """recommend DataFrame에서 공식 추천 1순위를 골라 회원 카드로 렌더."""
    row = None
    if df is not None and len(df) > 0:
        col = "STRICT_OFFICIAL_BUY_ELIGIBLE" if "STRICT_OFFICIAL_BUY_ELIGIBLE" in df.columns else None
        if col is not None:
            cand = df[df[col].astype(str).isin(["1", "1.0", "True"])]
            if len(cand) > 0:
                sort_col = "DISPLAY_SCORE" if "DISPLAY_SCORE" in cand.columns else None
                if sort_col:
                    cand = cand.sort_values(sort_col, ascending=False)
                row = cand.iloc[0].to_dict()
                market_ctx = {**market_ctx, "official_count": len(cand)}
    render_member_card(row, market_ctx)


__all__ = ["build_member_view", "render_member_card", "render_from_dataframe"]
