"""[v22 UI] 공통 용어 사전 — 시장 탭 + 종목 탭 일관성 보장

목적:
  - 모든 탭에서 사용자에게 보이는 한국형 매매 용어를 한 곳에서 관리
  - 내부 영문 코드 (ROUTE, TOP_PICK_TYPE, ELITE_LABEL 등) ↔ 화면 표시 한국어 매핑
  - 데이터 컬럼명/내부 라벨은 그대로 유지 (호환성)

사용:
  from components.ui_terms import (
      ROUTE_LABELS, route_display, route_icon,
      PICK_TYPE_LABELS, pick_type_info,
      ELITE_LABEL_DISPLAY, ELITE_LABEL_INTERNAL,
      label_to_display, label_to_internal,
      kelly_engine_label, gap_direction,
      VERDICT_LABELS,
  )
"""


# ═══════════════════════════════════════════════════
# ROUTE — 매매 단계 (진행 상태)
# ═══════════════════════════════════════════════════
# [Step AE] 외부 리뷰안 — 행동 중심 부드러운 표현 (마케팅 톤 X)
ROUTE_LABELS = {
    "ATTACK":       "매수검토",
    "ARMED":        "진입대기",
    "WAIT":         "관망",
    "NEUTRAL":      "중립",
    "CARRY":        "보유관리",
    "OVERHEAT":     "과열주의",
    "EXIT_WARNING": "매도주의",
    "BLOCKED":      "제외",
}

ROUTE_ICONS = {
    "ATTACK":       "🚀",
    "ARMED":        "🎯",
    "WAIT":         "⏸️",
    "NEUTRAL":      "👁️",
    "CARRY":        "📌",
    "OVERHEAT":     "🔥",
    "EXIT_WARNING": "⚠️",
    "BLOCKED":      "⛔",
}


def route_display(route: str) -> str:
    """ROUTE 코드 → 화면 표시명 (예: 'ATTACK' → '매수검토').

    [Step AE] 외부 리뷰안 — 행동 중심 부드러운 표현 사용.
    매핑은 ROUTE_LABELS 참조.
    """
    if not route:
        return ""
    key = str(route).strip().upper()
    return ROUTE_LABELS.get(key, key)


def route_icon(route: str) -> str:
    """ROUTE 코드 → 이모지 아이콘만 (예: 'ATTACK' → '🚀')."""
    if not route:
        return "👀"
    key = str(route).strip().upper()
    return ROUTE_ICONS.get(key, "👀")


# ═══════════════════════════════════════════════════
# TOP_PICK_TYPE — v22 추천 타입
# ═══════════════════════════════════════════════════
PICK_TYPE_LABELS = {
    "AGGRESSIVE": ("🔥", "공격형", "#EF4444"),
    "STABLE":     ("💎", "안정형", "#10B981"),
    "":           ("⭐", "추천",   "#F59E0B"),
}


def pick_type_info(pick_type: str) -> tuple:
    """TOP_PICK_TYPE → (이모지, 한국어, 색깔) 반환.
    
    사용:
        emoji, label, accent = pick_type_info(row.get('TOP_PICK_TYPE', ''))
    """
    if not pick_type:
        return PICK_TYPE_LABELS[""]
    key = str(pick_type).strip().upper()
    return PICK_TYPE_LABELS.get(key, PICK_TYPE_LABELS[""])


# ═══════════════════════════════════════════════════
# ELITE_LABEL — v3.7 라벨 체계 (내부값 유지, 화면만 한국어)
# ═══════════════════════════════════════════════════

# [Step AE] 외부 리뷰안 — 관찰/공식매수 분리 + 컬러 통일 🟣🟢🔵🟠
# 내부 ELITE_LABEL → 화면 표시
ELITE_LABEL_DISPLAY = {
    "🛡️ 콤보":   "🟣 핵심 관찰",
    "🏆 최강":   "🔵 관심관찰",
    # v22.3.8 UI safety: 내부값은 유지하되 화면에서는 매수 가능처럼 보이지 않게 표시
    "✅ 즉시진입": "🟡 관찰 후보",
    "⚠️ 추격":   "🟠 추격주의",
}

# 화면 표시 → 내부 ELITE_LABEL (필터 → 데이터 매칭용)
ELITE_LABEL_INTERNAL = {v: k for k, v in ELITE_LABEL_DISPLAY.items()}

# [Step AE] 테이블/칸반용 — 길지 않으니 동일하게
ELITE_LABEL_DISPLAY_SHORT = {
    "🛡️ 콤보":   "🟣 핵심 관찰",
    "🏆 최강":   "🔵 관심관찰",
    # 짧은 라벨도 '진입가능' 금지 — 공식 신규매수는 TOP_PICK+BUY_NOW_ELIGIBLE만
    "✅ 즉시진입": "🟡 관찰 후보",
    "⚠️ 추격":   "🟠 추격주의",
}


def label_to_display(internal: str, short: bool = False) -> str:
    """내부 ELITE_LABEL → 화면 표시.
    
    Args:
        internal: '🛡️ 콤보' / '🏆 최강' / '✅ 즉시진입' / '⚠️ 추격'
        short: True면 테이블/칸반용 짧은 버전
    """
    if not internal:
        return "—"
    src = internal.strip()
    if short:
        return ELITE_LABEL_DISPLAY_SHORT.get(src, src)
    return ELITE_LABEL_DISPLAY.get(src, src)


def label_to_internal(display: str) -> str:
    """화면 표시 → 내부 ELITE_LABEL (필터 매핑용)."""
    if not display:
        return display
    return ELITE_LABEL_INTERNAL.get(display.strip(), display)


# ═══════════════════════════════════════════════════
# 점수/지표 — 내부 컬럼명 → 한국어 표시
# ═══════════════════════════════════════════════════
SCORE_LABELS = {
    "DISPLAY_SCORE":     "점수",
    "ELITE_SCORE":       "종합 점수",
    "ELITE_RANK_SCORE":  "추천 순위",
    "STRUCT_SCORE":      "구조",
    "TIMING_SCORE":      "타이밍",
    "AI_SCORE":          "AI",
    "ML_SCORE":          "AI",          # AI_SCORE 별칭
    "BALANCE_SCORE":     "3축 균형",
    "BALANCE_CALC":      "3축 균형",
    "RR_NOW_TP1":        "수익:손실",
    "ENTRY_GAP_PCT":     "추천가 차이",
    "GAP_PCT":           "갭%",
    "EST_WIN_RATE":      "승률",
    "TP1_PCT":           "목표 수익",
}


# ═══════════════════════════════════════════════════
# Kelly engine (권장 비중 계산)
# ═══════════════════════════════════════════════════
def kelly_engine_label(engine: str) -> tuple:
    """KELLY_ENGINE 값을 (표시문, 색깔클래스, 내부엔진명) 반환.
    
    [v3.9.10] 회원에게 "v22_calibrated" 같은 내부 이름은 의미 없음.
    "권장 비중 계산: 정상/보수모드"로 단순화.
    
    [v3.9.11] 관리자/디버그용으로 3번째 반환값에 내부 엔진명 원문 추가.
    호출부에서 admin이면 tooltip(raw)을 붙여 표시 가능.
    
    사용:
        text, css_cls, raw = kelly_engine_label(row.get('KELLY_ENGINE', ''))
        if text:
            lbl = ui.label(text).classes(css_cls)
            if is_admin and raw:
                lbl.tooltip(f"engine: {raw}")
    """
    engine = (engine or "").strip()
    if not engine or engine.lower() in ("nan", "none"):
        return ("", "", "")
    if "fallback" in engine.lower():
        return ("⚠️ 권장 비중 계산: 보수모드", "text-xs text-red-300", engine)
    return ("권장 비중 계산: 정상", "text-xs text-gray-500", engine)


# ═══════════════════════════════════════════════════
# 진입갭 방향성
# ═══════════════════════════════════════════════════
def gap_direction(gap: float) -> str:
    """ENTRY_GAP_PCT → 한국어 설명.
    
    예:
        gap_direction(+0.4) → '현재가 높음'
        gap_direction(-0.3) → '현재가 낮음'
        gap_direction(0.02) → '현재가 일치'
    """
    try:
        g = float(gap)
    except (TypeError, ValueError):
        return ""
    if abs(g) < 0.05:
        return "현재가 일치"
    return "현재가 높음" if g > 0 else "현재가 낮음"


# ═══════════════════════════════════════════════════
# 시장 탭 매매 판단 verdict
# ═══════════════════════════════════════════════════
VERDICT_LABELS = {
    "OK":           "🟢 오늘 매수 OK",
    "HALF":         "🟠 절반만 매수 권장",
    "BLOCK_ENGINE": "🟠 신규 매수 자제 (엔진 제한)",
    "BLOCK_MARKET": "🔴 오늘 매수 금지 (시장 위험)",
    "OBSERVE":      "⏸️ 오늘은 지켜보세요",
    "NO_SIGNAL":    "🔴 매수 신호 없음",
}


# ═══════════════════════════════════════════════════
# max_allowed_route 허용 세트 (시장 탭 Step F2)
# ═══════════════════════════════════════════════════
ALLOWED_MAX_ROUTES = frozenset({
    "ATTACK", "ARMED",                            # 표준 ROUTE enum
    "ALL", "FULL",                                # 전체 허용 변종
    "TOP_PICK", "ATTACK_ONLY", "ALLOW_ATTACK",    # TOP_PICK 변종
})


def is_route_blocked(max_route: str) -> bool:
    """max_allowed_route가 진입 차단 상태인지.
    
    빈 문자열이면 정보 없음 = False (차단 안 함, 기본 허용).
    """
    if not max_route:
        return False
    key = str(max_route).strip().upper()
    return key not in ALLOWED_MAX_ROUTES


# ═══════════════════════════════════════════════════
# TOP_PICK 강건 파서 (1, 1.0, "True", "Y", "YES" 모두 인식)
# ═══════════════════════════════════════════════════
TRUTHY_VALUES = frozenset({"1", "1.0", "TRUE", "Y", "YES"})


def is_truthy_flag(val) -> bool:
    """TOP_PICK / IS_NOW_ENTRY 등 boolean-like 컬럼 강건 파서."""
    if val is None:
        return False
    return str(val).strip().upper() in TRUTHY_VALUES
