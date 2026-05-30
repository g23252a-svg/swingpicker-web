"""
components/buy_now_badge.py
============================
[v3.9.22b] BUY_NOW 배지 표시 헬퍼.

평가 명시 '절대 지킬 룰 5개' 캡슐화:
1. TOP_PICK 정렬/선정 로직 무수정 (이 모듈은 표시 헬퍼만)
2. UI 매수 가능 표시는 BUY_NOW_ELIGIBLE만 사용
3. BUY_NOW_PASS는 화면에 직접 "매수 가능"으로 쓰지 말 것
4. TOP_PICK=0 종목은 BUY_NOW_GRADE가 BUY여도 일반 화면에서 숨김
5. AVOID도 TOP_PICK이면 숨기지 말고 "추격 금지"로 노출

핵심 API:
- get_buy_now_display(row): 표시용 dict 반환
- BUY_NOW_BADGE_LABELS: 등급별 라벨 매핑
"""
from __future__ import annotations

from typing import Any, Dict, Optional


# 배지 라벨 (절대 지킬 룰 #2~#5 적용)
BUY_NOW_BADGE_LABELS = {
    "BUY": {
        "icon": "🟢",
        "label": "공식 매수 가능",  # v22.3.8: "매수 적합" → "공식 매수 가능"
        "tone": "buy",       # CSS class
        "color": "#10b981",  # emerald
        "short": "신규 진입 가능",  # v22.3.8: "즉시 진입 가능" → "신규 진입 가능"
    },
    "WATCH": {
        "icon": "🟡",
        "label": "관찰 후보",  # v22.3.8: "관찰/눌림 대기" → "관찰 후보"
        "tone": "watch",
        "color": "#f59e0b",  # amber
        "short": "공식 매수 제외",  # v22.3.8: "눌림 대기" → "공식 매수 제외"
    },
    "AVOID": {
        "icon": "🔴",
        "label": "추격 금지",
        "tone": "avoid",
        "color": "#ef4444",  # red
        "short": "지금 매수 금지",
    },
    # TOP_PICK 아닌 행 — 화면에 표시되지 않아야 함
    "NONE": {
        "icon": "",
        "label": "",
        "tone": "none",
        "color": "#6b7280",
        "short": "",
    },
}


def _safe_int(v, default=0):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _safe_str(v, default=""):
    if v is None:
        return default
    s = str(v)
    return s if s and s.lower() != "nan" else default


def get_buy_now_display(row: Dict[str, Any]) -> Dict[str, Any]:
    """[v3.9.22b → v22.3.8 safety] 종목 1개의 BUY_NOW 표시 정보 산출.

    절대 지킬 룰 (평가 명시):
    - TOP_PICK=0 → 화면에 숨김 (visible=False)
    - TOP_PICK=1 AND GRADE=BUY AND ELIGIBLE=1 → 🟢 매수 적합 (★ 공식 매수)
    - TOP_PICK=1 AND GRADE=BUY AND ELIGIBLE=0 → 🟡 관찰 후보 (v22.3.8 추가)
    - TOP_PICK=1 AND GRADE=WATCH → 🟡 관찰/눌림 대기
    - TOP_PICK=1 AND GRADE=AVOID → 🔴 추격 금지 (숨기지 않음!)

    [v22.3.8] BUY_NOW_GRADE=BUY인데 BUY_NOW_ELIGIBLE=0인 경우:
        원래 동작: 화면에 🟢 "매수 적합" / "즉시 진입 가능"으로 표시됐음
        새 동작: display_*에서는 🟡 관찰 후보로 강등 (회원 오해 방지)
        단, "grade" 필드는 그대로 BUY 유지 (기존 22 e2e 호환)

    [v22.3.8-D1] STALE_CARRY 보유종목 가드:
        IS_STALE_CARRY=True AND DISPLAY_SCORE<30 이면 visible=False로 숨김.
        오래 끌고 온 약한 보유종목은 신규매수 후보가 아니므로 BUY_NOW 배지가
        보이면 회원 오해가 큼. raw BUY_NOW_GRADE / ELIGIBLE은 변경 없음.
        DISPLAY_SCORE 없으면 FINAL_SCORE로 fallback.

    Args:
        row: dict-like (recommend CSV row 또는 _normalize 결과)

    Returns:
        {
            "visible": bool,            # TOP_PICK이면 True (D1 가드 시 False)
            "grade": str,               # BUY/WATCH/AVOID/NONE (raw — 기존 호환)
            "eligible": bool,           # ELIGIBLE 컬럼 — 매수 가능 신호
            "official_buy": bool,       # ★ v22.3.8 신규 — 공식 매수 가능 여부
            "icon": str,                # 🟢/🟡/🔴 (raw — 기존 호환)
            "label": str,               # 매수 적합 / 관찰 / 추격 금지 (raw — 기존 호환)
            "tone": str,                # CSS class 이름 (raw — 기존 호환)
            "color": str,               # hex 색상 (raw — 기존 호환)
            "short": str,               # 한 줄 설명 (raw — 기존 호환)
            "display_icon": str,        # ★ v22.3.8 — ELIGIBLE 반영 icon
            "display_label": str,       # ★ v22.3.8 — ELIGIBLE 반영 label
            "display_short": str,       # ★ v22.3.8 — ELIGIBLE 반영 short
            "display_tone": str,        # ★ v22.3.8 — ELIGIBLE 반영 tone
            "display_color": str,       # ★ v22.3.8 — ELIGIBLE 반영 color
            "score": float,             # BUY_NOW_SCORE
            "reason": str,              # BUY_NOW_REASON (툴팁용)
            "stale_carry_guard": bool,  # ★ v22.3.8-D1 — STALE_CARRY 가드 발동 여부
        }
    """
    # TOP_PICK 우선 체크 (절대 지킬 룰 #4)
    is_top_pick = _safe_int(row.get("TOP_PICK"), 0) == 1

    grade = _safe_str(row.get("BUY_NOW_GRADE"), "")
    if grade not in ("BUY", "WATCH", "AVOID"):
        grade = "NONE"

    eligible = _safe_int(row.get("BUY_NOW_ELIGIBLE"), 0) == 1
    score = _safe_float(row.get("BUY_NOW_SCORE"), 0.0)
    reason = _safe_str(row.get("BUY_NOW_REASON"), "")

    # ★ v22.3.8-D1: STALE_CARRY 표시 가드
    # IS_STALE_CARRY=True + DISPLAY_SCORE<30 이면 BUY_NOW 배지 숨김.
    # raw 데이터(grade/eligible/score/reason)는 변경하지 않음 — 표시만 막음.
    is_stale_carry = bool(row.get("IS_STALE_CARRY", False))
    # DISPLAY_SCORE 우선, 없으면 FINAL_SCORE fallback
    _display_raw = row.get("DISPLAY_SCORE")
    if _display_raw is None or (isinstance(_display_raw, float) and _display_raw != _display_raw):
        _display_raw = row.get("FINAL_SCORE")
    display_score = _safe_float(_display_raw, default=None) if _display_raw is not None else None
    stale_carry_guard = bool(
        is_stale_carry
        and display_score is not None
        and display_score < 30
    )

    badge = BUY_NOW_BADGE_LABELS.get(grade, BUY_NOW_BADGE_LABELS["NONE"])

    # ★ v22.3.8: 공식 매수 가능 여부
    # 회원에게 "매수 가능"으로 보이려면 visible AND eligible AND grade=BUY 모두 필요.
    # 어느 하나라도 빠지면 절대 🟢 매수 적합으로 표시되면 안 됨.
    # ★ v22.3.8-D1: STALE_CARRY 가드 발동 시에도 공식 매수 아님
    official_buy = bool(
        is_top_pick and eligible and grade == "BUY" and not stale_carry_guard
    )

    # ★ v22.3.8: display_* 필드 — ELIGIBLE을 반영한 안전한 표시값
    # BUY이지만 ELIGIBLE=0이면 화면에는 "관찰 후보"로 강등하여 표시.
    if grade == "BUY" and not eligible:
        display_badge = BUY_NOW_BADGE_LABELS["WATCH"]
    else:
        display_badge = badge

    # ★ v22.3.8-D1: visible 최종 결정 — STALE_CARRY 가드 적용
    # TOP_PICK이라도 STALE_CARRY 가드 발동 시 화면 숨김.
    visible = bool(is_top_pick and not stale_carry_guard)

    return {
        # 절대 지킬 룰 #4: TOP_PICK=0이면 숨김
        # [v22.3.8-D1] STALE_CARRY 가드 시에도 숨김
        "visible": visible,
        "grade": grade,
        # 절대 지킬 룰 #2: ELIGIBLE만 매수 가능 신호 (PASS 사용 금지)
        "eligible": eligible,
        # ★ v22.3.8 신규: 공식 매수 가능 여부 (UI에서 이것만 신뢰)
        "official_buy": official_buy,
        # raw 라벨 (기존 호환 유지)
        "icon": badge["icon"],
        "label": badge["label"],
        "tone": badge["tone"],
        "color": badge["color"],
        "short": badge["short"],
        # ★ v22.3.8 신규: ELIGIBLE 반영 안전 표시값 (UI 사용 권장)
        "display_icon": display_badge["icon"],
        "display_label": display_badge["label"],
        "display_short": display_badge["short"],
        "display_tone": display_badge["tone"],
        "display_color": display_badge["color"],
        "score": score,
        "reason": reason,
        # ★ v22.3.8-D1 신규: STALE_CARRY 가드 발동 여부 (디버깅/툴팁용)
        "stale_carry_guard": stale_carry_guard,
    }


def format_buy_now_subtitle(disp: Dict[str, Any]) -> str:
    """종목 카드 보조 설명 한 줄.

    [v22.3.8] display_* 필드 사용 — ELIGIBLE 반영된 안전한 표시.
        BUY이지만 ELIGIBLE=0인 경우 자동으로 "관찰 후보"로 표시됨.

    예시:
        official_buy=True:        "🟢 BUY_NOW 80점 — 즉시 진입 가능"
        BUY but ELIGIBLE=0:       "🟡 BUY_NOW 80점 — 눌림 대기" (★ v22.3.8)
        WATCH:                    "🟡 BUY_NOW 60점 — 눌림 대기"
        AVOID:                    "🔴 BUY_NOW 0점 — 지금 매수 금지"
    """
    if not disp.get("visible") or disp.get("grade") == "NONE":
        return ""
    # ★ v22.3.8: display_* 우선 (없으면 raw로 fallback — 호환성)
    icon = disp.get("display_icon", disp.get("icon", ""))
    score = disp.get("score", 0)
    short = disp.get("display_short", disp.get("short", ""))
    prefix = "공식 신규매수" if disp.get("official_buy") else "진입조건"
    return f"{icon} {prefix} {score:.0f}점 — {short}"


def format_buy_now_tooltip(disp: Dict[str, Any]) -> str:
    """툴팁/회색 설명 — BUY_NOW_REASON 가공.

    [v22.3.8] official_buy 여부에 따라 기본 메시지 차별화.
        BUY이지만 ELIGIBLE=0이면 "공식 매수 제외" 안내 추가.
    """
    if not disp.get("visible"):
        return ""
    reason = disp.get("reason", "")

    # ★ v22.3.8: BUY인데 ELIGIBLE=0이면 회원 오해 방지 안내
    grade = disp.get("grade", "NONE")
    if grade == "BUY" and not disp.get("official_buy"):
        ineligible_note = "진입조건은 양호하나 BUY_NOW_ELIGIBLE=0 · 공식 매수 대상 아님"
        if reason:
            return f"사유: {reason} · {ineligible_note}"
        return f"사유: {ineligible_note}"

    if not reason:
        # reason 없으면 등급별 기본 메시지
        defaults = {
            "BUY": "사유: RR 양호 · 추격위험 낮음 · 데이터 정상",
            "WATCH": "사유: 일부 위험 신호 — 진입 보류 권장",
            "AVOID": "사유: 위험 신호 다수 — 추격 매수 금지",
            "NONE": "",
        }
        return defaults.get(grade, "")
    return f"사유: {reason}"

# ═══════════════════════════════════════════════════════════════════
# [v22.3.21] No-Buy / 관망 카드 모델 — FOMO-safety
#   목적: official_buy=False인 날, 프론트가 목표가/매수가를 매수 CTA처럼
#         노출해 백엔드 안전신호(TOP_PICK + BUY_NOW_ELIGIBLE)를 훼손하지 않게 한다.
#   이 함수는 '무엇을/어떻게 보여줄지'만 정하는 순수 로직(렌더 없음).
#   → UX 규칙을 테스트로 강제할 수 있다. TOP_PICK/BUY_NOW_ELIGIBLE 계약은 건드리지 않는다.
# ═══════════════════════════════════════════════════════════════════

# 공식 신규매수 ELITE 통과선(AGGRESSIVE 기준). 표시 전용 — 게이트 계약 변경 아님.
NO_BUY_ELITE_PASS = 75.0
NO_BUY_AXIS_THRESHOLDS = {"struct": 80.0, "timing": 70.0, "ai": 60.0, "balance": 50.0}


def build_no_buy_card_model(row: Dict[str, Any], is_admin: bool = False) -> Dict[str, Any]:
    """관망/No-Buy 카드의 표시 모델(dict)을 만든다. 렌더러(NiceGUI 등)는 이 결과만 소비한다.

    FOMO-safety 핵심 규칙 — official_buy=False 일 때:
      - promote_target = False        (목표 수익률을 hero로 띄우지 않음)
      - price_treatment = "footnote"  (매수가/목표가는 회색 각주로 강등 · CTA 아님)
      - headline_tone = "rest"        ('매수 금지'가 아니라 '오늘은 쉬어갑니다')
      - cta = None                    (매수 CTA 없음)
      - blockers 는 1순위/2순위로 분리

    official_buy=True 일 때만 promote_target=True / price_treatment="hero" / 매수 CTA 허용.
    """
    disp = get_buy_now_display(row)
    official_buy = bool(disp.get("official_buy"))
    name = _safe_str(row.get("종목명"), "이 종목")

    th = NO_BUY_AXIS_THRESHOLDS
    axes = [
        {"key": "struct",  "name": "구조",      "value": _safe_float(row.get("STRUCT_SCORE")), "threshold": th["struct"]},
        {"key": "timing",  "name": "타이밍",    "value": _safe_float(row.get("TIMING_SCORE")), "threshold": th["timing"]},
        {"key": "ai",      "name": "AI 신뢰도", "value": _safe_float(row.get("AI_SCORE")),     "threshold": th["ai"]},
        {"key": "balance", "name": "3축 균형",  "value": _safe_float(row.get("BALANCE_SCORE")), "threshold": th["balance"]},
    ]
    for a in axes:
        a["pass"] = a["value"] >= a["threshold"]

    elite = _safe_float(row.get("ELITE_SCORE"), _safe_float(row.get("EBS"), 0.0))
    score_gap = max(0.0, NO_BUY_ELITE_PASS - elite)

    macro_risk = _safe_str(row.get("MACRO_RISK")).upper()
    market_blocked = (macro_risk == "CRITICAL") or (_safe_int(row.get("MACRO_HARD_BLOCK_SHADOW")) == 1)

    blockers = []
    rank = 1
    if market_blocked:
        blockers.append({"rank": rank, "kind": "market", "text": "시장 위험 (오늘 전체 매수 차단)"})
        rank += 1
    if score_gap > 0:
        blockers.append({"rank": rank, "kind": "score",
                         "text": f"종합점수 {elite:.1f} / {NO_BUY_ELITE_PASS:.0f} ({score_gap:.1f} 부족)"})
        rank += 1
    if not blockers:
        blockers.append({"rank": 1, "kind": "other", "text": "공식 신규매수 기준 미충족"})

    entry = _safe_float(row.get("추천매수가"))
    target = _safe_float(row.get("추천매도가1"))
    tp1_pct = _safe_float(row.get("TP1_PCT"))
    rr = _safe_float(row.get("RR_NOW_TP1"))
    has_prices = entry > 0 and target > 0

    promote_target = official_buy
    if official_buy:
        price_treatment = "hero"
    elif has_prices:
        price_treatment = "footnote"
    else:
        price_treatment = "hidden"

    target_footnote = ""
    if (not official_buy) and has_prices:
        target_footnote = (
            f"참고용 — 조건 충족 시 잠재 목표 +{tp1_pct:.1f}% "
            f"({entry:,.0f} → {target:,.0f}), 수익:손실 {rr:.1f}:1. 오늘은 매수 대상이 아닙니다."
        )

    if official_buy:
        mode, headline_tone, cta = "official_buy", "buy", "매수 검토"
        headline = f"{name} · 신규 매수 가능"
        subtext = "공식 신규진입 기준을 통과했어요. 분할 진입과 손절 기준을 함께 확인하세요."
        watch_label = ""
        closest_note = ""
    else:
        mode = "watch_only" if disp.get("visible") else "no_buy"
        headline_tone, cta = "rest", None
        headline = "오늘은 신규 매수 쉬어갑니다"
        if market_blocked:
            subtext = ("시장 환경 위험이 높아 모든 신규 진입을 멈췄어요. 종목이 좋아 보여도 "
                       "이런 날은 무리한 매수보다 기다리는 편이 유리합니다.")
        else:
            subtext = ("오늘은 공식 신규진입 기준을 통과한 종목이 없어요. 무리한 진입보다 "
                       "다음 기회를 기다리는 것도 전략입니다.")
        watch_label = "관찰만 · 매수 아님"
        closest_note = "조건은 거의 갖췄지만, 아래 사유가 막았어요."

    model = {
        "mode": mode,
        "name": name,
        "official_buy": official_buy,
        "headline": headline,
        "headline_tone": headline_tone,      # "rest" | "buy"
        "subtext": subtext,
        "watch_label": watch_label,
        "closest_note": closest_note,
        "axes": axes,
        "blockers": blockers,
        "promote_target": promote_target,
        "price_treatment": price_treatment,  # "hero" | "footnote" | "hidden"
        "target_footnote": target_footnote,
        "cta": cta,
        "disclaimer": "투자 판단과 책임은 본인에게 있습니다",
        "admin_raw": {},
    }
    if is_admin:
        model["admin_raw"] = {
            "TOP_PICK": _safe_int(row.get("TOP_PICK")),
            "BUY_NOW_ELIGIBLE": int(bool(disp.get("eligible"))),
            "BUY_NOW_GRADE": disp.get("grade", ""),
            "ELITE_SCORE": elite,
            "STRUCT_SCORE": axes[0]["value"],
            "TIMING_SCORE": axes[1]["value"],
            "AI_SCORE": axes[2]["value"],
            "BALANCE_SCORE": axes[3]["value"],
            "ROUTE": _safe_str(row.get("ROUTE")),
            "MACRO_RISK": macro_risk,
            "ENTRY_RISK_LEVEL": _safe_str(row.get("ENTRY_RISK_LEVEL")),
            "TP1_PCT": tp1_pct,
            "RR_NOW_TP1": rr,
            "추천매수가": entry,
            "추천매도가1": target,
            "MACRO_HARD_BLOCK_SHADOW": _safe_int(row.get("MACRO_HARD_BLOCK_SHADOW")),
        }
    return model


def is_official_new_buy(row: Dict[str, Any]) -> bool:
    """[v22.3.21] 공식 신규매수 가능 = TOP_PICK==1 AND BUY_NOW_ELIGIBLE==1.

    초록 매수 CTA 라벨('오늘 신규 매수 가능' 등)은 이 함수가 True일 때만 허용한다.
    이 조건이 아니면(관찰 후보/공식 매수 미통과) 초록 CTA 대신 회색/중립 톤을 써야 한다.
    TOP_PICK/BUY_NOW_ELIGIBLE 계약 자체는 변경하지 않는다(읽기 전용 판정).
    """
    return _safe_int(row.get("TOP_PICK")) == 1 and _safe_int(row.get("BUY_NOW_ELIGIBLE")) == 1
