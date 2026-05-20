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
        "label": "매수 적합",
        "tone": "buy",       # CSS class
        "color": "#10b981",  # emerald
        "short": "즉시 진입 가능",
    },
    "WATCH": {
        "icon": "🟡",
        "label": "관찰/눌림 대기",
        "tone": "watch",
        "color": "#f59e0b",  # amber
        "short": "눌림 대기",
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
    """[v3.9.22b] 종목 1개의 BUY_NOW 표시 정보 산출.

    절대 지킬 룰 (평가 명시):
    - TOP_PICK=0 → 화면에 숨김 (visible=False)
    - TOP_PICK=1 AND GRADE=BUY → 🟢 매수 적합
    - TOP_PICK=1 AND GRADE=WATCH → 🟡 관찰/눌림 대기
    - TOP_PICK=1 AND GRADE=AVOID → 🔴 추격 금지 (숨기지 않음!)

    Args:
        row: dict-like (recommend CSV row 또는 _normalize 결과)

    Returns:
        {
            "visible": bool,         # TOP_PICK이면 True
            "grade": str,            # BUY/WATCH/AVOID/NONE
            "eligible": bool,        # ELIGIBLE 컬럼 — 매수 가능 신호
            "icon": str,             # 🟢/🟡/🔴
            "label": str,            # 매수 적합 / 관찰 / 추격 금지
            "tone": str,             # CSS class 이름 (buy/watch/avoid)
            "color": str,            # hex 색상
            "short": str,            # 한 줄 설명
            "score": float,          # BUY_NOW_SCORE
            "reason": str,           # BUY_NOW_REASON (툴팁용)
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

    return {
        # 절대 지킬 룰 #4: TOP_PICK=0이면 숨김
        "visible": is_top_pick,
        "grade": grade,
        # 절대 지킬 룰 #2: ELIGIBLE만 매수 가능 신호 (PASS 사용 금지)
        "eligible": eligible,
        "icon": badge["icon"],
        "label": badge["label"],
        "tone": badge["tone"],
        "color": badge["color"],
        "short": badge["short"],
        "score": score,
        "reason": reason,
    }


def format_buy_now_subtitle(disp: Dict[str, Any]) -> str:
    """종목 카드 보조 설명 한 줄.

    예시:
        "🟢 BUY_NOW 80점 — 즉시 진입 가능"
        "🟡 WATCH 60점 — 눌림 대기"
        "🔴 AVOID 0점 — RR 부족 / 추격 금지"
    """
    if not disp.get("visible") or disp.get("grade") == "NONE":
        return ""
    icon = disp["icon"]
    grade = disp["grade"]
    score = disp["score"]
    short = disp["short"]
    return f"{icon} BUY_NOW {score:.0f}점 — {short}"


def format_buy_now_tooltip(disp: Dict[str, Any]) -> str:
    """툴팁/회색 설명 — BUY_NOW_REASON 가공."""
    if not disp.get("visible"):
        return ""
    reason = disp.get("reason", "")
    if not reason:
        # reason 없으면 등급별 기본 메시지
        defaults = {
            "BUY": "사유: RR 양호 · 추격위험 낮음 · 데이터 정상",
            "WATCH": "사유: 일부 위험 신호 — 진입 보류 권장",
            "AVOID": "사유: 위험 신호 다수 — 추격 매수 금지",
            "NONE": "",
        }
        return defaults.get(disp.get("grade", "NONE"), "")
    return f"사유: {reason}"
