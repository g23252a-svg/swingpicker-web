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

    Args:
        row: dict-like (recommend CSV row 또는 _normalize 결과)

    Returns:
        {
            "visible": bool,            # TOP_PICK이면 True
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

    badge = BUY_NOW_BADGE_LABELS.get(grade, BUY_NOW_BADGE_LABELS["NONE"])

    # ★ v22.3.8: 공식 매수 가능 여부
    # 회원에게 "매수 가능"으로 보이려면 visible AND eligible AND grade=BUY 모두 필요.
    # 어느 하나라도 빠지면 절대 🟢 매수 적합으로 표시되면 안 됨.
    official_buy = bool(is_top_pick and eligible and grade == "BUY")

    # ★ v22.3.8: display_* 필드 — ELIGIBLE을 반영한 안전한 표시값
    # BUY이지만 ELIGIBLE=0이면 화면에는 "관찰 후보"로 강등하여 표시.
    if grade == "BUY" and not eligible:
        display_badge = BUY_NOW_BADGE_LABELS["WATCH"]
    else:
        display_badge = badge

    return {
        # 절대 지킬 룰 #4: TOP_PICK=0이면 숨김
        "visible": is_top_pick,
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
    return f"{icon} BUY_NOW {score:.0f}점 — {short}"


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
        ineligible_note = "BUY_NOW_ELIGIBLE=0 · 공식 매수 대상 아님"
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
