# -*- coding: utf-8 -*-
"""
SwingPicker — 종목 상세 v2 Full Dashboard
═══════════════════════════════════════════════════
[Step 2F COMPLETE] 정밀 차트 + 리스크/뉴스/시나리오/최종판정 기반.

구성:
  헤더 (5뱃지) + 점수 영역 (DISPLAY/3축/FINAL/ELITE/BALANCE/사유)
  ┌──────────┬──────────────────────────┬───────────┐
  │ 좌측 4패널 │ 메인 캔들차트 (ECharts)  │ 레이더 5축 │
  │ #1 가격    │ + HMA20/VWAP/SUPERTREND │           │
  │ #2 추세    │ + markLine 6개          │           │
  │ #3 모멘텀  │ + 보조차트 5개(RSI/MFI/  │           │
  │ #4 수급    │   MACD/V_POWER/거강)     │           │
  │            │ + 3분할 (수익/레벨/리스크) │           │
  │            │ + 시나리오 3카드 (A/B/C) │           │
  └──────────┴──────────────────────────┴───────────┘
  하단 4섹터 (핵심요약/분할익절/DipSniper/비교)
  최종 판정 띠 (등급별 색상 + 강점/리스크 자동)

데이터 소스: data/recommend_latest.csv + data/ohlcv_cache_*.parquet
컬럼 매핑 (실제 확인 완료):
  헤더/점수: TOP_PICK/TOP_PICK_TYPE/EBS/PASS_EBS/ACTION_PRIORITY/IS_ACTIVE
            AXIS_MEAN/AXIS_GAP/BALANCE_SCORE (CSV 원본 우선)
  사유:     SCORE_REASON_TOP1 + TOP2 + ELITE_REASON (CSV 우선, 없으면 fallback)
  가격:     종가/추천매수가/손절가/추천매도가1-3 + RR_NOW_TP1/MAX_LOSS_PCT/
            TIME_STOP_DAYS/POSITION_PCT/KELLY_FINAL_B/켈리_수량
  추세/MTF: 주봉20선_상회/주봉추세/MTF_WEEKLY_TREND/MTF_MONTHLY_TREND/
            SUPERTREND_DIR+VAL/Above_MA20/HMA20/이격도
  모멘텀:   ret_1d~120d_% + rel_20~120d_% + 벤치_60d_KOSPI/KOSDAQ
  수급:     거래대금(억원)/시가총액(억원)/외인순매수/기관순매수/메이저/개인/
            거래강도/V_POWER/Vol_Quality
  보조차트: RSI14/MFI14/MACD_Slope_PCT/V_POWER
  리스크/뉴스: NEWS_SCORE/MACRO_RISK/MARKET_BREADTH/NEWS_REASON/ROUTE_REASON/
              BB_Expanding/IS_SWING_SUPPORT/EST_WIN_RATE

UI 원칙:
  - UI는 normalize_stock_row() 결과만 사용. 컬럼명 직접 row.get() 금지.
  - 모든 텍스트 html.escape() 안전 삽입 (XSS / 화면깨짐 방지).
  - layout-critical CSS는 inline style 강제 (NiceGUI Quasar 래퍼 충돌 회피).
  - design CSS는 _inject_v2_styles()로 일괄 주입 (1회).

OHLCV 로더:
  - components.tab_stocks._get_chart_data 위임 우선 (Railway 검증된 SSOT)
  - 실패 시 자체 fallback (멀티 경로/포맷/컬럼 자동 탐지)
  - debug 정보는 DEBUG_STOCK_V2=1 또는 admin 환경에서만 노출

마지막 리뷰 점수: 97/100 — 운영 머지 가능권.
"""

from html import escape as h_escape
from typing import Optional, Dict, Any
from nicegui import ui
from shared_utils import safe_float


# ═══════════════════════════════════════════════════
# 정규화 어댑터 (UI는 이것만 사용)
# ═══════════════════════════════════════════════════

def _as_bool(v) -> bool:
    """다양한 truthy 표현을 bool로 정규화."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    try:
        if isinstance(v, (int, float)):
            return v != 0 and not (isinstance(v, float) and v != v)  # NaN 방어
    except Exception:
        pass
    return str(v).strip().lower() in ("1", "true", "yes", "y", "o")


def _safe_str(v, default="—") -> str:
    """문자열 변환 (None/nan 안전)."""
    if v is None:
        return default
    try:
        if isinstance(v, float) and v != v:  # NaN
            return default
    except Exception:
        pass
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return default
    return s


ROUTE_KR_MAP = {
    "ATTACK":   "매수검토",
    "ARMED":    "진입대기",
    "WAIT":     "관망",
    "NEUTRAL":  "중립",
    "OVERHEAT": "과열주의",
    "CARRY":    "보유유지",
}


def normalize_stock_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    recommend_latest.csv 한 행을 UI용 dict로 정규화.

    실제 CSV 컬럼명 확인 후 매핑:
      TOP_PICK / EBS / PASS_EBS / ACTION_PRIORITY / IS_ACTIVE — 직접 사용
      AXIS_MEAN / AXIS_GAP / BALANCE_SCORE — CSV 원본값 우선 (재계산 fallback만)
      ret_1d_% / ret_5d_% 등 — '_%' 접미사 포함

    Args:
        row: pandas Series 또는 dict (recommend_latest.csv 한 행)

    Returns:
        정규화된 dict — UI는 이것만 참조해야 함
    """
    # ── 기본 정보 ──
    route = _safe_str(row.get("ROUTE", "NEUTRAL"), "NEUTRAL").upper()
    top_pick = _as_bool(row.get("TOP_PICK", 0))
    top_pick_type = _safe_str(row.get("TOP_PICK_TYPE", ""), "")

    # [v3.9.22b] BUY_NOW 표시 정보 — 평가 룰 5개 캡슐화
    from components.buy_now_badge import get_buy_now_display
    buy_now = get_buy_now_display(row)

    # ── EBS (실제 컬럼: EBS + PASS_EBS) ──
    ebs = int(safe_float(row.get("EBS", 0)) or 0)
    pass_ebs = _as_bool(row.get("PASS_EBS", ebs >= 5))

    # ── ACTION_PRIORITY / IS_ACTIVE (CSV 원본) ──
    action_priority = int(safe_float(row.get("ACTION_PRIORITY", 0)) or 0)
    is_active = _as_bool(row.get("IS_ACTIVE", route in ("ARMED", "ATTACK")))

    # ── 점수 (CSV 원본값 우선, 0이면 폴백) ──
    display_score = safe_float(row.get("DISPLAY_SCORE", 0)) or 0
    final_score = safe_float(row.get("FINAL_SCORE", 0)) or 0
    elite_score = safe_float(row.get("ELITE_SCORE", 0)) or 0
    struct_s = safe_float(row.get("STRUCT_SCORE", 0)) or 0
    timing_s = safe_float(row.get("TIMING_SCORE", 0)) or 0
    ai_s = safe_float(row.get("AI_SCORE", 0)) or 0

    # AXIS_MEAN/AXIS_GAP — CSV 원본 우선, 없을 때만 재계산
    axis_mean = safe_float(row.get("AXIS_MEAN", 0))
    if not axis_mean or axis_mean == 0:
        if any([struct_s, timing_s, ai_s]):
            axis_mean = (struct_s + timing_s + ai_s) / 3
        else:
            axis_mean = 0

    axis_gap = safe_float(row.get("AXIS_GAP", 0))
    if not axis_gap or axis_gap == 0:
        non_zero = [s for s in [struct_s, timing_s, ai_s] if s > 0]
        axis_gap = (max(non_zero) - min(non_zero)) if len(non_zero) >= 2 else 0

    balance_score = safe_float(row.get("BALANCE_SCORE", 0)) or 0

    # ── [Step 2B] 가격 플랜 필드 ──
    close = safe_float(row.get("종가", 0)) or 0
    entry = safe_float(row.get("추천매수가", 0)) or 0
    stop = safe_float(row.get("손절가", 0)) or 0
    tp1 = safe_float(row.get("추천매도가1", 0)) or 0
    tp2 = safe_float(row.get("추천매도가2", 0)) or 0
    tp3 = safe_float(row.get("추천매도가3", 0)) or 0
    rr_now_tp1 = safe_float(row.get("RR_NOW_TP1", 0)) or 0
    rr_mult = safe_float(row.get("RR_MULT", 0)) or 0
    # ENTRY_GAP_PCT 우선, 없으면 GAP_PCT(대문자), gap_pct(소문자) fallback
    entry_gap_pct = safe_float(
        row.get("ENTRY_GAP_PCT", row.get("GAP_PCT", row.get("gap_pct", 0)))
    ) or 0
    stop_pct = safe_float(row.get("STOP_PCT", 0)) or 0
    max_loss_pct = safe_float(row.get("MAX_LOSS_PCT", 0)) or 0
    tp1_pct = safe_float(row.get("TP1_PCT", 0)) or 0
    tp1_prob = safe_float(row.get("TP1_PROB", 0)) or 0
    tp2_prob = safe_float(row.get("TP2_PROB", 0)) or 0
    tp3_prob = safe_float(row.get("TP3_PROB", 0)) or 0
    time_stop_days = safe_float(row.get("TIME_STOP_DAYS", row.get("TIME_STOP", 0))) or 0
    position_pct = safe_float(row.get("POSITION_PCT", 0)) or 0
    kelly_fraction = safe_float(row.get("KELLY_FRACTION", 0)) or 0
    kelly_final_b = safe_float(row.get("KELLY_FINAL_B", 0)) or 0
    kelly_planned_b = safe_float(row.get("KELLY_PLANNED_B", 0)) or 0
    kelly_empirical_b = safe_float(row.get("KELLY_EMPIRICAL_B", 0)) or 0
    # 수량/금액 (켈리_수량 우선, 추천수량 fallback)
    qty = safe_float(row.get("켈리_수량", row.get("추천수량", 0))) or 0
    # 추천금액(만원) 우선, 없으면 켈리_금액(원) / 10000 으로 계산
    amount_man = safe_float(row.get("추천금액(만원)", 0)) or 0
    if not amount_man:
        amount_won = safe_float(row.get("켈리_금액(원)", 0)) or 0
        if amount_won:
            amount_man = amount_won / 10000
    tp1_method = _safe_str(row.get("TP1_METHOD", ""), "")
    tp2_method = _safe_str(row.get("TP2_METHOD", ""), "")
    tp3_method = _safe_str(row.get("TP3_METHOD", ""), "")

    # TP1~3 상승률 자동 계산 (매수가 기준)
    tp1_gain_pct = ((tp1 / entry - 1) * 100) if entry > 0 and tp1 > 0 else 0
    tp2_gain_pct = ((tp2 / entry - 1) * 100) if entry > 0 and tp2 > 0 else 0
    tp3_gain_pct = ((tp3 / entry - 1) * 100) if entry > 0 and tp3 > 0 else 0
    # 손절 손실률 (음수)
    stop_loss_pct = ((stop / entry - 1) * 100) if entry > 0 and stop > 0 else 0

    # ── [Step 2C] 추세/MTF 필드 ──
    weekly_ma20_above = _safe_str(row.get("주봉20선_상회", ""), "")
    weekly_trend = _safe_str(row.get("주봉추세", ""), "")
    mtf_weekly = int(safe_float(row.get("MTF_WEEKLY_TREND", row.get("MTF_WEEKLY", 0))) or 0)
    mtf_monthly = int(safe_float(row.get("MTF_MONTHLY_TREND", row.get("MTF_MONTHLY", 0))) or 0)
    mtf_sufficient = _as_bool(row.get("MTF_DATA_SUFFICIENT", 0))
    supertrend_dir = int(safe_float(row.get("SUPERTREND_DIR", 0)) or 0)
    supertrend_val = safe_float(row.get("SUPERTREND_VAL", 0)) or 0
    above_ma20 = int(safe_float(row.get("Above_MA20", 0)) or 0)
    hma20 = safe_float(row.get("HMA20", 0)) or 0
    hma_trend = _safe_str(row.get("HMA_Trend", ""), "")
    hma_on = _safe_str(row.get("HMA_On", ""), "")
    igyukdo = safe_float(row.get("이격도", 0)) or 0
    vwap_gap = safe_float(row.get("VWAP_GAP", 0)) or 0

    # ── [Step 2C] 모멘텀/수익률 필드 ──
    ret_1d = safe_float(row.get("ret_1d_%", 0)) or 0
    ret_5d = safe_float(row.get("ret_5d_%", 0)) or 0
    ret_10d = safe_float(row.get("ret_10d_%", 0)) or 0
    ret_20d = safe_float(row.get("ret_20d_%", 0)) or 0
    ret_60d = safe_float(row.get("ret_60d_%", 0)) or 0
    ret_120d = safe_float(row.get("ret_120d_%", 0)) or 0
    rel_20d = safe_float(row.get("rel_20d_%", 0)) or 0
    rel_60d = safe_float(row.get("rel_60d_%", 0)) or 0
    rel_120d = safe_float(row.get("rel_120d_%", 0)) or 0
    bench_kospi = safe_float(row.get("벤치_60d_KOSPI_%", 0)) or 0
    bench_kosdaq = safe_float(row.get("벤치_60d_KOSDAQ_%", 0)) or 0

    # ── [Step 2C] 수급/유동성 필드 ──
    turnover_eok = safe_float(row.get("거래대금(억원)", 0)) or 0
    mcap_eok = safe_float(row.get("시가총액(억원)", 0)) or 0
    foreign_net = safe_float(row.get("외인순매수", 0)) or 0
    institution_net = safe_float(row.get("기관순매수", 0)) or 0
    major_net = safe_float(row.get("메이저순매수", 0)) or 0
    individual_net = safe_float(row.get("개인순매수", 0)) or 0
    trade_strength = safe_float(row.get("거래강도", 0)) or 0
    v_power = safe_float(row.get("V_POWER", 0)) or 0
    vol_quality = safe_float(row.get("Vol_Quality", 0)) or 0

    # ── [Step 2E] 기술지표 (보조 차트용) ──
    rsi14 = safe_float(row.get("RSI14", 0)) or 0
    mfi14 = safe_float(row.get("MFI14", 0)) or 0
    macd_slope_pct = safe_float(row.get("MACD_Slope_PCT", 0)) or 0
    rsi_rising = int(safe_float(row.get("RSI_Rising", 0)) or 0)
    bb_expanding = int(safe_float(row.get("BB_Expanding", 0)) or 0)
    bb_bw = safe_float(row.get("BB_BW", 0)) or 0
    ttm_squeeze = int(safe_float(row.get("TTM_SQUEEZE_CNT", row.get("TTM_SQUEEZE", 0))) or 0)
    poc_gap = safe_float(row.get("POC_GAP", 0)) or 0

    # ── [v22.3.10] ENTRY_EDGE shadow display fields ──
    entry_edge_score = safe_float(row.get("ENTRY_EDGE_SCORE", 100)) or 100
    entry_edge_level = _safe_str(row.get("ENTRY_EDGE_LEVEL", "GREEN"), "GREEN").upper()
    entry_edge_reason = _safe_str(row.get("ENTRY_EDGE_REASON", ""), "")
    entry_edge_rule = _safe_str(row.get("ENTRY_EDGE_RULE", ""), "")
    entry_edge_shadow_flag = int(safe_float(row.get("ENTRY_EDGE_SHADOW_FLAG", 0)) or 0)

    # ── [Step 2E] 레이더 5축 (TRIGGER 포함) ──
    trigger_score = safe_float(row.get("TRIGGER_SCORE", row.get("RAW_TRIGGER_SCORE", 0))) or 0

    # ── [Step 2F] 리스크 / 뉴스 / 시나리오 / 최종판정 ──
    news_score = safe_float(row.get("NEWS_SCORE", 0)) or 0
    macro_risk = _safe_str(row.get("MACRO_RISK", ""), "")
    market_breadth = safe_float(row.get("MARKET_BREADTH", 0)) or 0
    est_win_rate = safe_float(row.get("EST_WIN_RATE", 0)) or 0
    ai_comment = _safe_str(row.get("AI_COMMENT", ""), "")
    score_risk = _safe_str(row.get("SCORE_RISK", ""), "")
    stop_reason = _safe_str(row.get("STOP_REASON", ""), "")
    bb_expanding = int(safe_float(row.get("BB_Expanding", 0)) or 0)
    is_swing_support = _as_bool(row.get("IS_SWING_SUPPORT", 0))
    vwap = safe_float(row.get("VWAP", 0)) or 0
    supertrend_val_raw = safe_float(row.get("SUPERTREND_VAL", 0)) or 0

    # ── 사유 (CSV 우선, 없을 때만 fallback 자동 생성) ──
    # 우선순위:
    #   1) ELITE_REASON (가장 풍부 — "RR1.3 + 진입적정 + 3축고점 + 대기")
    #   2) SCORE_REASON_TOP1 + SCORE_REASON_TOP2 (결합)
    #   3) 자동 fallback
    elite_reason = _safe_str(row.get("ELITE_REASON", ""), "")
    reason_top1 = _safe_str(row.get("SCORE_REASON_TOP1", ""), "")
    reason_top2 = _safe_str(row.get("SCORE_REASON_TOP2", ""), "")
    news_reason = _safe_str(row.get("NEWS_REASON", ""), "")
    route_reason = _safe_str(row.get("ROUTE_REASON", ""), "")
    plan_reason = _safe_str(row.get("PLAN_REASON", ""), "")

    if elite_reason and elite_reason != "—":
        score_reason = elite_reason
    elif reason_top1 and reason_top1 != "—":
        # TOP1 + TOP2 결합 ("STRUCT 강점 + TIMING 보조")
        if reason_top2 and reason_top2 != "—":
            score_reason = f"{reason_top1} + {reason_top2}"
        else:
            score_reason = reason_top1
    else:
        # fallback 자동 생성 (CSV에 사유 컬럼이 전혀 없을 때만)
        parts = []
        if struct_s >= 90:
            parts.append("STRUCT 강점")
        elif struct_s >= 70:
            parts.append("STRUCT 양호")
        if timing_s >= 80:
            parts.append("TIMING 양호")
        if axis_gap > 0 and axis_gap < 25:
            parts.append("3축 균형")  # [v2] 3측 → 3축 오타 수정
        if route == "ARMED":
            parts.append("대기")
        elif route == "ATTACK":
            parts.append("진입검토")
        score_reason = " + ".join(parts) if parts else "—"

    return {
        # 기본
        "name": _safe_str(row.get("종목명", "?"), "?"),
        "code": _safe_str(row.get("종목코드", "?"), "?").zfill(6),
        "route": route,
        "route_kr": ROUTE_KR_MAP.get(route, "중립"),

        # status 뱃지
        "top_pick": top_pick,
        "top_pick_type": top_pick_type or ("AGGRESSIVE" if top_pick else "—"),
        # [v3.9.22b] BUY_NOW 표시 정보
        "buy_now": buy_now,
        "ebs": ebs,
        "ebs_total": 8,
        "pass_ebs": pass_ebs,
        "action_priority": action_priority,
        "is_active": is_active,

        # 점수
        "display_score": display_score,
        "final_score": final_score,
        "elite_score": elite_score,
        "struct_score": struct_s,
        "timing_score": timing_s,
        "ai_score": ai_s,
        "axis_mean": axis_mean,
        "axis_gap": axis_gap,
        "balance_score": balance_score,

        # 사유 (UI 표시용 — score_reason은 우선순위 적용 후 결과)
        "score_reason": score_reason,
        # 원본 사유들 (Step 2D-2E 패널에서 직접 표시용)
        "elite_reason": elite_reason,
        "reason_top1": reason_top1,
        "reason_top2": reason_top2,
        "news_reason": news_reason,
        "route_reason": route_reason,
        "plan_reason": plan_reason,

        # [Step 2B] 가격 플랜
        "close": close,
        "entry": entry,
        "stop": stop,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "rr_now_tp1": rr_now_tp1,
        "rr_mult": rr_mult,
        "entry_gap_pct": entry_gap_pct,
        "stop_pct": stop_pct,
        "stop_loss_pct": stop_loss_pct,
        "max_loss_pct": max_loss_pct,
        "tp1_pct": tp1_pct,
        "tp1_gain_pct": tp1_gain_pct,
        "tp2_gain_pct": tp2_gain_pct,
        "tp3_gain_pct": tp3_gain_pct,
        "tp1_prob": tp1_prob,
        "tp2_prob": tp2_prob,
        "tp3_prob": tp3_prob,
        "time_stop_days": time_stop_days,
        "position_pct": position_pct,
        "kelly_fraction": kelly_fraction,
        "kelly_final_b": kelly_final_b,
        "kelly_planned_b": kelly_planned_b,
        "kelly_empirical_b": kelly_empirical_b,
        "qty": qty,
        "amount_man": amount_man,
        "tp1_method": tp1_method,
        "tp2_method": tp2_method,
        "tp3_method": tp3_method,

        # [Step 2C] 추세/MTF
        "weekly_ma20_above": weekly_ma20_above,
        "weekly_trend": weekly_trend,
        "mtf_weekly": mtf_weekly,
        "mtf_monthly": mtf_monthly,
        "mtf_sufficient": mtf_sufficient,
        "supertrend_dir": supertrend_dir,
        "supertrend_val": supertrend_val,
        "above_ma20": above_ma20,
        "hma20": hma20,
        "hma_trend": hma_trend,
        "hma_on": hma_on,
        "igyukdo": igyukdo,
        "vwap_gap": vwap_gap,

        # [Step 2C] 모멘텀/수익률
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "ret_20d": ret_20d,
        "ret_60d": ret_60d,
        "ret_120d": ret_120d,
        "rel_20d": rel_20d,
        "rel_60d": rel_60d,
        "rel_120d": rel_120d,
        "bench_kospi": bench_kospi,
        "bench_kosdaq": bench_kosdaq,

        # [Step 2C] 수급/유동성
        "turnover_eok": turnover_eok,
        "mcap_eok": mcap_eok,
        "foreign_net": foreign_net,
        "institution_net": institution_net,
        "major_net": major_net,
        "individual_net": individual_net,
        "trade_strength": trade_strength,
        "v_power": v_power,
        "vol_quality": vol_quality,

        # [Step 2E] 기술지표 (보조 차트용)
        "rsi14": rsi14,
        "mfi14": mfi14,
        "macd_slope_pct": macd_slope_pct,
        "rsi_rising": rsi_rising,
        "bb_expanding": bb_expanding,
        "bb_bw": bb_bw,
        "ttm_squeeze": ttm_squeeze,
        "poc_gap": poc_gap,
        "trigger_score": trigger_score,
        # [v22.3.10] ENTRY_EDGE shadow — 공식 매수식 미반영 표시용
        "entry_edge_score": entry_edge_score,
        "entry_edge_level": entry_edge_level,
        "entry_edge_reason": entry_edge_reason,
        "entry_edge_rule": entry_edge_rule,
        "entry_edge_shadow_flag": entry_edge_shadow_flag,

        # [Step 2F] 리스크 / 뉴스 / 시나리오 / 최종판정
        "news_score": news_score,
        "macro_risk": macro_risk,
        "market_breadth": market_breadth,
        "est_win_rate": est_win_rate,
        "ai_comment": ai_comment,
        "score_risk": score_risk,
        "stop_reason": stop_reason,
        "bb_expanding": bb_expanding,
        "is_swing_support": is_swing_support,
        "vwap": vwap,
        "supertrend_val_raw": supertrend_val_raw,
    }


# ═══════════════════════════════════════════════════
# 디자인 토큰 (mockup HTML과 1:1 매칭)
# ═══════════════════════════════════════════════════

# 한국식 캔들 색상 (차트 Step 2C에서 사용)
KOREA_UP = "#EF4444"     # 상승 빨강
KOREA_DOWN = "#3B82F6"   # 하락 파랑


# CSS 중복 주입 방지 플래그 (context 접근 실패 시 fallback)
_V2_STYLE_INJECTED = False

# OHLCV 로더의 v1 위임 실패 사유 추적 (debug_info에서 노출)
_last_v1_error = None


def _inject_v2_styles():
    """v2 디자인 토큰 CSS 주입 (세션당 1회만, 중복 방지)."""
    # NiceGUI client 컨텍스트 단위로 1회만 주입
    # 같은 페이지에서 여러 번 render_stock_detail_v2_*가 호출되어도 안전
    try:
        from nicegui import context
        client = context.client
        # client 객체에 플래그 부착 (스레드/요청 분리됨)
        if getattr(client, "_sd_v2_style_injected", False):
            return
        client._sd_v2_style_injected = True
    except Exception:
        # context 접근 실패 시 fallback — module-level 1회만
        global _V2_STYLE_INJECTED
        if _V2_STYLE_INJECTED:
            return
        _V2_STYLE_INJECTED = True

    ui.add_head_html("""
    <style>
    .sd-v2 {
      --bg-deep: #0F1117;
      --bg-card: #1A1D26;
      --bg-card-2: #232631;
      --border: #2A2D38;
      --border-light: #353845;
      --text-white: #FFFFFF;
      --text-gray: #9CA3AF;
      --text-dim: #6B7280;
      --red: #EF4444;
      --red-dim: rgba(239, 68, 68, 0.15);
      --green: #10B981;
      --green-dim: rgba(16, 185, 129, 0.15);
      --orange: #F59E0B;
      --orange-dim: rgba(245, 158, 11, 0.15);
      --yellow: #FACC15;
      --purple: #8B5CF6;
      --purple-dim: rgba(139, 92, 246, 0.18);
      --cyan: #06B6D4;
      --cyan-dim: rgba(6, 182, 212, 0.15);
      --blue: #3B82F6;
      --pink: #EC4899;
      color: var(--text-white);
      font-size: 12px;
      line-height: 1.4;
      width: 100%;
      box-sizing: border-box;
    }
    .sd-v2 *, .sd-v2 *::before, .sd-v2 *::after { box-sizing: border-box; }

    /* ── 헤더 영역 ── */
    .sd-v2 .header {
      display: grid;
      grid-template-columns: 1fr 200px 130px 130px 110px;
      gap: 8px;
      margin-bottom: 12px;
    }
    .sd-v2 .h-title {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .sd-v2 .h-title .name {
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.5px;
    }
    .sd-v2 .h-title .name .code {
      color: var(--text-gray);
      font-size: 18px;
      font-weight: 600;
      margin-left: 4px;
    }
    .sd-v2 .h-title .meta {
      color: var(--text-dim);
      font-size: 10px;
      margin-top: 4px;
    }
    .sd-v2 .h-title .rank {
      margin-top: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .sd-v2 .h-title .rank .label {
      background: linear-gradient(135deg, #FFD700, #FFA500);
      color: #1A1D26;
      font-weight: 800;
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 4px;
    }
    .sd-v2 .h-title .rank .rank-num {
      color: var(--yellow);
      font-weight: 800;
      font-size: 14px;
    }
    .sd-v2 .h-title .rank .rank-total {
      color: var(--text-gray);
      font-size: 11px;
    }

    .sd-v2 .h-badge {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 4px;
      min-height: 96px;       /* 모든 박스 높이 동일 */
      text-align: center;
      box-sizing: border-box;
    }
    .sd-v2 .h-badge .lbl {
      color: var(--text-gray);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }
    .sd-v2 .h-badge .val {
      font-size: 14px;
      font-weight: 800;
      letter-spacing: -0.3px;
      line-height: 1.2;
    }
    .sd-v2 .h-badge .sub { color: var(--text-dim); font-size: 9px; line-height: 1.4; }
    .sd-v2 .h-badge.core {
      background: linear-gradient(135deg, var(--purple-dim), rgba(236, 72, 153, 0.1));
      border-color: var(--purple);
    }
    .sd-v2 .h-badge.core .val { color: var(--purple); }
    .sd-v2 .h-badge.toppick {
      background: var(--orange-dim);
      border-color: var(--orange);
    }
    .sd-v2 .h-badge.toppick .val { color: var(--orange); }
    .sd-v2 .h-badge.toppick.dim { background: var(--bg-card); border-color: var(--border); }
    .sd-v2 .h-badge.toppick.dim .val { color: var(--text-dim); }

    .sd-v2 .h-badge.route {
      background: var(--orange-dim);
      border-color: var(--orange);
    }
    .sd-v2 .h-badge.route .val { color: var(--orange); }
    .sd-v2 .h-badge.route.attack {
      background: var(--red-dim);
      border-color: var(--red);
    }
    .sd-v2 .h-badge.route.attack .val { color: var(--red); }
    .sd-v2 .h-badge.route.wait {
      background: var(--blue-dim);
      border-color: var(--blue);
    }
    .sd-v2 .h-badge.route.wait .val { color: var(--blue); }

    .sd-v2 .h-badge.ebs {
      background: var(--green-dim);
      border-color: var(--green);
    }
    .sd-v2 .h-badge.ebs .val { color: var(--green); font-size: 16px; }
    .sd-v2 .h-badge.ebs .pass {
      background: var(--green);
      color: #0F1117;
      font-weight: 800;
      font-size: 10px;
      padding: 2px 8px;
      border-radius: 3px;
    }
    .sd-v2 .h-badge.ebs.fail {
      background: var(--red-dim);
      border-color: var(--red);
    }
    .sd-v2 .h-badge.ebs.fail .val { color: var(--red); }
    .sd-v2 .h-badge.ebs.fail .pass {
      background: var(--red);
      color: white;
    }
    .sd-v2 .h-badge .priority { color: var(--text-dim); font-size: 9px; }
    .sd-v2 .h-badge .priority strong { color: var(--orange); }
    .sd-v2 .h-badge .active { color: var(--green); font-size: 9px; }
    .sd-v2 .h-badge .active::before { content: "● "; }
    .sd-v2 .h-badge .inactive { color: var(--text-dim); font-size: 9px; }
    .sd-v2 .h-badge .inactive::before { content: "○ "; }

    /* ── 점수 영역 ── */
    .sd-v2 .scores {
      display: grid;
      grid-template-columns: 140px 1fr 130px 130px 130px;
      gap: 8px;
      margin-bottom: 12px;
    }
    .sd-v2 .display-score {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }
    .sd-v2 .display-score .lbl {
      color: var(--text-gray);
      font-size: 10px;
      font-weight: 600;
    }
    .sd-v2 .display-score .val {
      font-size: 38px;
      font-weight: 900;
      background: linear-gradient(135deg, var(--purple), var(--pink));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1;
    }
    .sd-v2 .display-score .arc {
      width: 80px;
      height: 6px;
      background: linear-gradient(90deg, var(--red), var(--orange), var(--yellow), var(--green));
      border-radius: 3px;
      position: relative;
    }
    .sd-v2 .display-score .arc::after {
      content: "▲";
      position: absolute;
      color: var(--purple);
      font-size: 8px;
      top: 5px;
      left: 70%;
      transform: translateX(-50%);
    }

    .sd-v2 .axis-mean {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      position: relative;
    }
    .sd-v2 .axis-mean .axis-title {
      position: absolute;
      top: -10px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--bg-deep);
      padding: 0 8px;
      color: var(--text-gray);
      font-size: 10px;
      font-weight: 600;
    }
    .sd-v2 .axis-cell { text-align: center; padding-top: 4px; }
    .sd-v2 .axis-cell .axis-lbl {
      font-size: 10px;
      color: var(--text-gray);
      font-weight: 600;
      margin-bottom: 4px;
    }
    .sd-v2 .axis-cell .axis-val {
      font-size: 26px;
      font-weight: 900;
      line-height: 1;
      margin-bottom: 2px;
    }
    .sd-v2 .axis-cell .axis-sub { font-size: 10px; color: var(--text-dim); }
    .sd-v2 .axis-cell.struct .axis-val { color: var(--green); }
    .sd-v2 .axis-cell.struct .axis-sub { color: var(--green); }
    .sd-v2 .axis-cell.timing .axis-val { color: var(--blue); }
    .sd-v2 .axis-cell.ai .axis-val { color: var(--purple); }

    .sd-v2 .score-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    .sd-v2 .score-card .lbl {
      color: var(--text-gray);
      font-size: 10px;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .sd-v2 .score-card .val {
      font-size: 28px;
      font-weight: 900;
      line-height: 1;
    }
    .sd-v2 .score-card .sub {
      color: var(--text-dim);
      font-size: 10px;
      margin-top: 4px;
    }
    .sd-v2 .score-card.final .val { color: var(--orange); }
    .sd-v2 .score-card.elite .val { color: var(--yellow); }
    .sd-v2 .score-card.balance .val { color: var(--cyan); }
    .sd-v2 .score-card.balance .sub { color: var(--cyan); }

    .sd-v2 .score-reason {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 14px;
      margin-bottom: 12px;
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .sd-v2 .score-reason .lbl {
      color: var(--text-gray);
      font-size: 10px;
      font-weight: 600;
    }
    .sd-v2 .score-reason .val {
      color: var(--pink);
      font-size: 11px;
      font-weight: 600;
    }

    /* var(--blue-dim) 추가 정의 */
    .sd-v2 { --blue-dim: rgba(59, 130, 246, 0.15); }

    /* ═══════════ [Step 2B+] 메인 그리드 + 좌측 사이드 패널 ═══════════ */
    .sd-v2 .main-grid {
      display: grid;
      grid-template-columns: 240px minmax(0, 1fr) 240px;
      gap: 8px;
      margin-bottom: 12px;
      width: 100%;
    }
    .sd-v2 .center-area {
      min-width: 0;  /* 1fr 컬럼이 콘텐츠 폭에 갇히지 않도록 */
    }
    .sd-v2 .side-panel {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .sd-v2 .panel {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
    }
    .sd-v2 .panel-title {
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--text-white);
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 8px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
    }
    .sd-v2 .panel-title .num {
      background: var(--purple-dim);
      color: var(--purple);
      width: 18px;
      height: 18px;
      border-radius: 4px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 10px;
    }
    .sd-v2 .panel-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 5px 0;
      font-size: 11px;
      line-height: 1.4;
      gap: 12px;            /* 라벨과 값 사이 최소 간격 보장 */
    }
    .sd-v2 .panel-row .lbl {
      color: var(--text-gray);
      display: flex;
      align-items: center;
      gap: 4px;
      flex-shrink: 0;
    }
    .sd-v2 .panel-row .lbl::before {
      content: "✓";
      color: var(--green);
      font-size: 10px;
    }
    .sd-v2 .panel-row.note .lbl::before { content: ""; }
    .sd-v2 .panel-row .val {
      color: var(--text-white);
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }
    .sd-v2 .panel-row .val.green  { color: var(--green); }
    .sd-v2 .panel-row .val.red    { color: var(--red); }
    .sd-v2 .panel-row .val.orange { color: var(--orange); }
    .sd-v2 .panel-row .val.cyan   { color: var(--cyan); }
    .sd-v2 .panel-row .val.muted  { color: var(--text-dim); }

    .sd-v2 .panel-row.tp {
      background: rgba(16, 185, 129, 0.05);
      margin: 2px -4px;
      padding: 4px 6px;
      border-radius: 4px;
    }
    .sd-v2 .panel-row.tp .lbl { color: var(--text-white); font-weight: 600; }
    .sd-v2 .panel-row.stop .val { color: var(--red); }

    .sd-v2 .tp-prob-line {
      text-align: right;
      font-size: 9px;
      color: var(--text-dim);
      padding-right: 4px;
      margin-top: -2px;
    }

    .sd-v2 .panel-row.divider {
      border-top: 1px dashed var(--border);
      margin-top: 6px;
      padding-top: 6px;
    }
    .sd-v2 .panel-row .pct-small {
      color: var(--text-dim);
      font-size: 9px;
      margin-left: 3px;
    }
    .sd-v2 .panel-row.dot .lbl::before { content: "●"; color: var(--purple); }

    /* 수급 패널 — 아이콘 추가 */
    .sd-v2 .panel-row .ice::after { content: " 🧊"; }

    /* ═══════════════════════════════════════════════════
       모바일 대응 (≤768px) — 명시적 클래스 기반 (attribute selector X)
       inline grid도 클래스에서 display 재정의 + !important로 덮어쓰기
       ═══════════════════════════════════════════════════ */
    @media (max-width: 768px) {
      .sd-v2 { font-size: 11px !important; }

      /* 헤더: 1fr 180 140x3 → 2컬럼 격자 */
      .sd-v2 .header {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        grid-template-rows: auto !important;
        gap: 4px !important;
      }
      .sd-v2 .h-title { grid-column: 1 / -1 !important; }
      .sd-v2 .h-badge {
        min-height: 70px !important;
        padding: 6px 8px !important;
      }
      .sd-v2 .h-badge .val { font-size: 12px !important; }
      .sd-v2 .h-badge .lbl { font-size: 9px !important; }

      /* 점수 영역: 140 1fr 130x3 → 2컬럼 격자 */
      .sd-v2 .scores {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 4px !important;
      }
      .sd-v2 .scores > *:first-child,
      .sd-v2 .scores > *:nth-child(2) {
        grid-column: 1 / -1 !important;
      }

      /* 메인 그리드: 260 1fr 300 → 1컬럼 세로 적층 */
      .sd-v2.v2-main-grid {
        display: flex !important;
        flex-direction: column !important;
        grid-template-columns: 1fr !important;
        gap: 8px !important;
      }
      .sd-v2.v2-main-grid > * { width: 100% !important; }

      /* 좌측 4패널: 세로 → 2x2 격자 (모바일에서도 너무 길어지지 않게) */
      .sd-v2 .v2-left-panels {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        flex-direction: row !important;
        gap: 6px !important;
      }
      .sd-v2 .panel {
        padding: 8px !important;
        min-width: 0 !important;
      }
      .sd-v2 .panel-row {
        font-size: 10px !important;
        padding: 3px 0 !important;
        gap: 6px !important;
      }
      .sd-v2 .panel-title {
        font-size: 11px !important;
      }

      /* 우측 컬럼: 그대로 세로 유지 (레이더+AXIS_GAP+가이드) */
      .sd-v2 .v2-right-col {
        gap: 6px !important;
      }

      /* 보조차트 5개: 4 + 130 → 2x3 */
      .sd-v2 .v2-sub-charts {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 4px !important;
      }

      /* 3분할 패널 (수익률/레벨/리스크): 3컬럼 → 1컬럼 */
      .sd-v2 .v2-three-split {
        display: grid !important;
        grid-template-columns: 1fr !important;
        gap: 6px !important;
      }

      /* 시나리오 A/B/C: 3컬럼 → 1컬럼 */
      .sd-v2 .v2-scenarios {
        display: grid !important;
        grid-template-columns: 1fr !important;
        gap: 6px !important;
      }

      /* 하단 4섹터: 4컬럼 → 2x2 */
      .sd-v2 .bottom-grid {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 4px !important;
      }
      .sd-v2 .bottom-panel {
        min-height: 160px !important;
        padding: 8px !important;
      }
      .sd-v2 .bottom-panel .b-row { font-size: 9px !important; }
      .sd-v2 .bottom-panel .b-title { font-size: 11px !important; }

      /* 레이더 SVG: 모바일 높이 축소 */
      .sd-v2 svg { max-width: 100% !important; }

      /* 3분할 패널 카드 내부: 좁아도 글자가 세로로 쪼개지지 않게 */
      .sd-v2 .v2-three-split > div {
        min-width: 0 !important;
        overflow: hidden !important;
      }
      .sd-v2 .v2-three-split > div > div {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
      }
      /* 단, 값/숫자가 들어가는 child div는 nowrap 해제 (필요시 줄바꿈 허용) */
      .sd-v2 .v2-three-split > div > div[style*="display: grid"] {
        white-space: normal !important;
      }

      /* 최종 판정 띠 모바일 */
      .sd-v2 .v2-final-verdict {
        padding: 12px 14px !important;
        gap: 10px !important;
      }
    }

    /* 초소형 모바일 (≤480px) — 모든 격자 1컬럼 */
    @media (max-width: 480px) {
      .sd-v2 .header { grid-template-columns: 1fr !important; }
      .sd-v2 .scores { grid-template-columns: 1fr !important; }
      .sd-v2 .scores > * { grid-column: 1 / -1 !important; }
      .sd-v2 .v2-left-panels { grid-template-columns: 1fr !important; }
      .sd-v2 .v2-sub-charts { grid-template-columns: 1fr !important; }
      .sd-v2 .bottom-grid { grid-template-columns: 1fr !important; }
      .sd-v2 .h-badge { min-height: 50px !important; }
    }
    </style>
    """)


# ═══════════════════════════════════════════════════
# 포맷 헬퍼
# ═══════════════════════════════════════════════════

def _fmt_score(v, decimals=1):
    """점수 포맷 (소수 1자리)."""
    n = safe_float(v) or 0
    return f"{n:.{decimals}f}"


# ═══════════════════════════════════════════════════
# Step 2A: 헤더 행 (종목명 + 5개 status 뱃지)
# ═══════════════════════════════════════════════════

def render_v2_header(n: dict, rank: int = 0, total: int = 0,
                     timestamp: str = "", combo_info: Optional[dict] = None):
    """
    상단 헤더 행 렌더링.

    Args:
        n: normalize_stock_row()의 결과 dict
        rank: LDY_RANK 순위
        total: 전체 종목 수
        timestamp: 분석 시각
        combo_info: 콤보 정보 dict {color, name, n, ev, win_rate, rank}
            (CSV에 없는 정보 — 외부 데이터 또는 콤보 분석 결과)
    """
    name = h_escape(n["name"])
    code = h_escape(n["code"])
    route = n["route"]
    route_kr = n["route_kr"]
    top_pick = n["top_pick"]
    top_pick_type = h_escape(n["top_pick_type"])
    ebs = n["ebs"]
    ebs_total = n["ebs_total"]
    pass_ebs = n["pass_ebs"]
    is_active = n["is_active"]
    action_priority = n["action_priority"]

    # ROUTE 색상 클래스
    if route == "ATTACK":
        route_class = "route attack"
    elif route in ("WAIT", "NEUTRAL"):
        route_class = "route wait"
    else:  # ARMED, OVERHEAT
        route_class = "route"

    # TOP_PICK 클래스 (꺼져있으면 dim)
    top_pick_class = "toppick" if top_pick else "toppick dim"

    # EBS 클래스
    ebs_class = "ebs" if pass_ebs else "ebs fail"
    ebs_pass_text = "PASS" if pass_ebs else "WAIT"

    # IS_ACTIVE 클래스
    active_class = "active" if is_active else "inactive"
    active_text = "True" if is_active else "False"

    with ui.element("div").classes("sd-v2").style("width: 100%;"):
        with ui.element("div").classes("header").style(
            # 종목명은 넓게 (1fr), 나머지 4박스는 동일 폭 (140px) — 균일한 외관
            "display: grid; grid-template-columns: 1fr 180px 140px 140px 140px; "
            "gap: 8px; margin-bottom: 12px; width: 100%; align-items: stretch;"
        ):

            # 1. 종목명 + LDY_RANK
            ui.html(f'''
                <div class="h-title">
                    <div class="name">{name} <span class="code">({code})</span></div>
                    <div class="meta">{h_escape(timestamp) if timestamp else "—"} / SwingPicker v22</div>
                    <div class="rank">
                        <span class="label">LDY_RANK</span>
                        <span>🏆</span>
                        <span class="rank-num">{rank}위</span>
                        <span class="rank-total">/ {total}개</span>
                    </div>
                </div>
            ''')

            # 2. 핵심매수 콤보 (combo_info 있으면 풍성, 없으면 CSV fallback)
            ci = combo_info or {}
            # combo_info가 없거나 비어있을 때 CSV의 elite 정보로 fallback
            if not ci:
                # ELITE_REASON 우선 (compact descriptive), 없으면 단순 ELITE
                fallback_name = n.get("elite_reason") or "ELITE"
                # ELITE_REASON이 너무 길면 잘라서 표시 (헤더 카드 폭 한계)
                if fallback_name and len(fallback_name) > 24:
                    fallback_name = fallback_name[:22] + "…"
                ci = {
                    "name": fallback_name,
                    "color": "#8B5CF6",
                    # ELITE_SCORE / FINAL_SCORE 등으로 sub 채우기
                    "_elite_score": n.get("elite_score", 0),
                    "_final_score": n.get("final_score", 0),
                }

            combo_color = ci.get("color", "#8B5CF6")
            combo_n = ci.get("n", 0)
            combo_ev = ci.get("ev", 0)
            combo_win = ci.get("win_rate", 0)
            combo_name = h_escape(str(ci.get("name", "ELITE")))
            combo_rank = ci.get("rank", None)
            # color는 hex 코드만 허용 (CSS 주입 방어)
            import re
            if not re.match(r"^#[0-9A-Fa-f]{3}([0-9A-Fa-f]{3})?$", combo_color):
                combo_color = "#8B5CF6"

            ci_lines = []
            if combo_color:
                ci_lines.append(f'<div class="sub" style="color: {combo_color};">콤보 {combo_color}</div>')
            if combo_rank:
                ci_lines.append(f'<div class="sub" style="color: var(--text-gray);">실성능 {combo_rank}위</div>')
            if combo_n:
                ci_lines.append(f'<div class="sub">n={combo_n}, EV {combo_ev:+.2f}%</div>')
            if combo_win:
                ci_lines.append(f'<div class="sub">승률 {combo_win:.1f}%</div>')
            # combo_info fallback 모드: ELITE_SCORE / FINAL_SCORE로 sub 채움
            if ci.get("_elite_score") or ci.get("_final_score"):
                if ci.get("_elite_score"):
                    ci_lines.append(f'<div class="sub" style="color: var(--text-gray);">ELITE {ci["_elite_score"]:.1f}</div>')
                if ci.get("_final_score"):
                    ci_lines.append(f'<div class="sub">FINAL {ci["_final_score"]:.1f}</div>')
            ci_html = "\n".join(ci_lines)

            ui.html(f'''
                <div class="h-badge core">
                    <div class="lbl">🌑 핵심매수</div>
                    <div class="val">{combo_name}</div>
                    {ci_html}
                </div>
            ''')

            # 3. TOP_PICK + BUY_NOW 배지 [v3.9.22b → v22.3.8 safety]
            top_pick_display = top_pick_type if top_pick else "—"
            # 평가 절대 지킬 룰 #4: TOP_PICK=1이어야 BUY_NOW 표시
            #                  #5: AVOID도 숨기지 않고 노출
            # [v22.3.8] BUY_NOW_ELIGIBLE=0이면 BUY여도 "관찰 후보"로 강등.
            # 회원 오해 방지 — buy_now_badge가 이미 display_* 안전 필드 제공.
            _bn = n.get("buy_now", {})
            _bn_visible = _bn.get("visible", False)
            # ★ v22.3.8: display_* 우선 (ELIGIBLE 반영). 없으면 raw fallback.
            _bn_icon = _bn.get("display_icon", _bn.get("icon", ""))
            _bn_label = _bn.get("display_label", _bn.get("label", ""))
            _bn_tone = _bn.get("display_tone", _bn.get("tone", "none"))
            _bn_color = _bn.get("display_color", _bn.get("color", "#666"))
            _bn_score = _bn.get("score", 0)
            _bn_reason = h_escape(_bn.get("reason", ""))
            # 툴팁: reason or 기본
            from components.buy_now_badge import format_buy_now_tooltip
            _bn_tooltip = h_escape(format_buy_now_tooltip(_bn))

            # BUY_NOW 배지 HTML — TOP_PICK일 때만
            if _bn_visible and _bn_icon:
                _buy_now_html = (
                    f'<div class="buy-now-badge buy-now-{_bn_tone}" '
                    f'title="{_bn_tooltip}" '
                    f'style="margin-top:6px;padding:4px 8px;border-radius:6px;'
                    f'font-size:11px;font-weight:600;'
                    f'background:rgba(255,255,255,0.05);'
                    f'border:1px solid {_bn_color};'
                    f'color:{_bn_color};">'
                    f'{_bn_icon} {_bn_label} · {_bn_score:.0f}점'
                    f'</div>'
                )
            else:
                _buy_now_html = ""

            ui.html(f'''
                <div class="h-badge {top_pick_class}">
                    <div class="lbl">TOP_PICK</div>
                    <div class="val">{top_pick_display}</div>
                    {_buy_now_html}
                </div>
            ''')

            # 4. ROUTE
            ui.html(f'''
                <div class="h-badge {route_class}">
                    <div class="lbl">ROUTE</div>
                    <div class="val">{route}</div>
                    <div class="sub" style="color: var(--text-dim);">({route_kr})</div>
                </div>
            ''')

            # 5. EBS
            ui.html(f'''
                <div class="h-badge {ebs_class}">
                    <div class="lbl">EBS <span style="color: white;">{ebs}</span>/{ebs_total}</div>
                    <div class="pass">{ebs_pass_text}</div>
                    <div class="priority">ACTION_PRIORITY <strong>{action_priority}</strong></div>
                    <div class="{active_class}">IS_ACTIVE {active_text}</div>
                </div>
            ''')


# ═══════════════════════════════════════════════════
# Step 2A: 점수 영역
# ═══════════════════════════════════════════════════

def render_v2_scores(n: dict):
    """
    점수 영역 렌더링.

    CSV 원본값 사용:
      DISPLAY_SCORE / FINAL_SCORE / ELITE_SCORE / BALANCE_SCORE
      STRUCT_SCORE / TIMING_SCORE / AI_SCORE
      AXIS_MEAN / AXIS_GAP (재계산 X, CSV 우선)
      SCORE_REASON_TOP1 / ELITE_REASON (자동 생성 X)
    """
    display_score = n["display_score"]
    final_score = n["final_score"]
    elite_score = n["elite_score"]
    balance_score = n["balance_score"]
    struct_s = n["struct_score"]
    timing_s = n["timing_score"]
    ai_s = n["ai_score"]
    axis_mean = n["axis_mean"]
    axis_gap = n["axis_gap"]
    score_reason = n["score_reason"]
    entry_edge_score = n.get("entry_edge_score", 100)
    entry_edge_level = str(n.get("entry_edge_level", "GREEN") or "GREEN").upper()
    entry_edge_reason = n.get("entry_edge_reason", "") or ""
    entry_edge_shadow_flag = int(n.get("entry_edge_shadow_flag", 0) or 0)

    # 강점 표기: 가장 높은 축이 85+ 일 때만 "강점 ★"
    scores_dict = {"STRUCT": struct_s, "TIMING": timing_s, "AI": ai_s}
    max_axis = max(scores_dict, key=scores_dict.get) if any(scores_dict.values()) else None

    def _axis_tag(name, val):
        if name == max_axis and val >= 85:
            return ("강점 ★", "var(--green)")
        return ("보조", "var(--text-dim)")

    struct_tag, struct_clr = _axis_tag("STRUCT", struct_s)
    timing_tag, timing_clr = _axis_tag("TIMING", timing_s)
    ai_tag, ai_clr = _axis_tag("AI", ai_s)

    # AXIS_GAP 평가
    if axis_gap < 15:
        gap_tag = "우수"
    elif axis_gap < 25:
        gap_tag = "양호"
    else:
        gap_tag = "주의"

    with ui.element("div").classes("sd-v2").style("width: 100%;"):
        with ui.element("div").classes("scores").style(
            "display: grid; grid-template-columns: 140px 1fr 130px 130px 130px; "
            "gap: 8px; margin-bottom: 12px; width: 100%;"
        ):

            ui.html(f'''
                <div class="display-score">
                    <div class="lbl">DISPLAY_SCORE</div>
                    <div class="val">{_fmt_score(display_score)}</div>
                    <div class="arc"></div>
                </div>
            ''')

            ui.html(f'''
                <div class="axis-mean">
                    <div class="axis-title">3축 점수 (AXIS_MEAN {_fmt_score(axis_mean)})</div>
                    <div class="axis-cell struct">
                        <div class="axis-lbl">STRUCT</div>
                        <div class="axis-val">{_fmt_score(struct_s)}</div>
                        <div class="axis-sub" style="color: {struct_clr};">{struct_tag}</div>
                    </div>
                    <div class="axis-cell timing">
                        <div class="axis-lbl">TIMING</div>
                        <div class="axis-val">{_fmt_score(timing_s)}</div>
                        <div class="axis-sub" style="color: {timing_clr};">{timing_tag}</div>
                    </div>
                    <div class="axis-cell ai">
                        <div class="axis-lbl">AI</div>
                        <div class="axis-val">{_fmt_score(ai_s)}</div>
                        <div class="axis-sub" style="color: {ai_clr};">{ai_tag}</div>
                    </div>
                </div>
            ''')

            ui.html(f'''
                <div class="score-card final">
                    <div class="lbl">FINAL_SCORE</div>
                    <div class="val">{_fmt_score(final_score)}</div>
                </div>
            ''')

            ui.html(f'''
                <div class="score-card elite">
                    <div class="lbl">ELITE_SCORE</div>
                    <div class="val">{_fmt_score(elite_score)}</div>
                </div>
            ''')

            ui.html(f'''
                <div class="score-card balance">
                    <div class="lbl">BALANCE_SCORE</div>
                    <div class="val">{_fmt_score(balance_score)}</div>
                    <div class="sub">AXIS_GAP {_fmt_score(axis_gap)} ({gap_tag})</div>
                </div>
            ''')

        ui.html(f'''
            <div class="score-reason">
                <div class="lbl">점수사유:</div>
                <div class="val">{h_escape(score_reason)}</div>
            </div>
        ''')

        if entry_edge_shadow_flag or entry_edge_level == "CAUTION":
            ui.html(f'''
                <div style="margin-top: 8px; padding: 9px 11px; border-radius: 8px;
                            background: rgba(245,158,11,0.08);
                            border: 1px solid rgba(245,158,11,0.28);
                            color: var(--text-white);">
                    <div style="font-size: 11px; font-weight: 800; color: var(--orange);">
                        🧪 ENTRY_EDGE shadow · {entry_edge_score:.0f}점 · {h_escape(entry_edge_level)}
                    </div>
                    <div style="font-size: 11px; color: var(--text-gray); margin-top: 3px;">
                        {h_escape(entry_edge_reason or "B_red shadow 감점 관찰 · 공식 매수식 미반영")}
                    </div>
                    <div style="font-size: 10px; color: var(--text-dim); margin-top: 3px;">
                        표시/감점 전용입니다. BUY_NOW_ELIGIBLE 공식 신규매수 기준은 변경하지 않습니다.
                    </div>
                </div>
            ''')


# ═══════════════════════════════════════════════════
# Step 2B: 좌측 사이드 패널 #1 — 가격 플랜
# ═══════════════════════════════════════════════════

def _fmt_won(v) -> str:
    """원화 정수 포맷 (천단위 콤마). 0이면 '—'."""
    n = safe_float(v) or 0
    if n == 0:
        return "—"
    return f"{int(round(n)):,}원"


def _fmt_pct_signed(v, decimals=1) -> str:
    """부호 포함 퍼센트."""
    n = safe_float(v)
    if n is None:
        return "—"
    return f"{n:+.{decimals}f}%"


def _fmt_pct(v, decimals=1) -> str:
    """부호 없는 퍼센트."""
    n = safe_float(v)
    if n is None:
        return "—"
    return f"{n:.{decimals}f}%"


def render_v2_price_plan(n: dict):
    """
    [Step 2B] 좌측 사이드 패널 #1 — 가격 플랜.

    이미지 1번 패널 100% 재현:
        종가/매수가 (GAP)
        손절가 (-XX%)
        TP1 / 비중%
        TP2 / 비중%
        TP3 / 비중%
        ─────
        RR_NOW_TP1
        RR_MULT
        MAX_LOSS
        TIME_STOP
        POSITION
        KELLY_B
        KELLY_수량

    데이터 소스: normalize_stock_row() 결과 (CSV 원본값 우선).
    """
    close = n["close"]
    entry = n["entry"]
    stop = n["stop"]
    tp1, tp2, tp3 = n["tp1"], n["tp2"], n["tp3"]
    rr_now = n["rr_now_tp1"]
    rr_mult = n["rr_mult"]
    entry_gap = n["entry_gap_pct"]
    stop_loss = n["stop_loss_pct"]
    max_loss = n["max_loss_pct"]
    tp1_gain = n["tp1_gain_pct"]
    tp2_gain = n["tp2_gain_pct"]
    tp3_gain = n["tp3_gain_pct"]
    tp1_prob = n["tp1_prob"]
    tp2_prob = n["tp2_prob"]
    tp3_prob = n["tp3_prob"]
    time_stop = int(n["time_stop_days"]) if n["time_stop_days"] else 0
    position = n["position_pct"]
    kelly_final = n["kelly_final_b"]
    kelly_planned = n["kelly_planned_b"]
    kelly_empirical = n["kelly_empirical_b"]
    qty = int(n["qty"]) if n["qty"] else 0
    amount = n["amount_man"]

    # RR 평가
    if rr_now >= 1.5:
        rr_tag = "우수"
    elif rr_now >= 1.0:
        rr_tag = "양호"
    elif rr_now > 0:
        rr_tag = "부족"
    else:
        rr_tag = "—"

    ui.html(f'''
    <div class="sd-v2">
      <div class="panel">
        <div class="panel-title"><span class="num">1</span>가격 플랜</div>

        <div class="panel-row">
          <span class="lbl" title="전일 종가 (기준일 마감가)">종가</span>
          <span class="val">{_fmt_won(close)}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="시스템이 추천하는 분할 매수 진입가. GAP은 종가 대비 % 차이">추천매수가</span>
          <span class="val">{_fmt_won(entry)} <span class="pct-small">(GAP {_fmt_pct_signed(entry_gap)})</span></span>
        </div>
        <div class="panel-row stop">
          <span class="lbl" title="이 가격을 이탈하면 즉시 전량 손절. MAX_LOSS와 직결">손절가</span>
          <span class="val red">{_fmt_won(stop)} <span class="pct-small">({_fmt_pct_signed(stop_loss)})</span></span>
        </div>

        <div class="panel-row tp">
          <span class="lbl" title="1차 익절 목표가 (SWING_20D 기반, 도달 확률 표시)">TP1</span>
          <span class="val">{_fmt_won(tp1)} <span style="color: var(--green);">({_fmt_pct_signed(tp1_gain)})</span></span>
        </div>
        <div class="tp-prob-line">{h_escape(n["tp1_method"]) or "—"} · {int(tp1_prob)}%</div>

        <div class="panel-row tp">
          <span class="lbl" title="2차 익절 목표가 (ATR x 배수 기반)">TP2</span>
          <span class="val">{_fmt_won(tp2)} <span style="color: var(--green);">({_fmt_pct_signed(tp2_gain)})</span></span>
        </div>
        <div class="tp-prob-line">{h_escape(n["tp2_method"]) or "—"} · {int(tp2_prob)}%</div>

        <div class="panel-row tp">
          <span class="lbl" title="3차 익절 목표가 (피보나치 1.618 확장 기반)">TP3</span>
          <span class="val">{_fmt_won(tp3)} <span style="color: var(--green);">({_fmt_pct_signed(tp3_gain)})</span></span>
        </div>
        <div class="tp-prob-line">{h_escape(n["tp3_method"]) or "—"} · {int(tp3_prob)}%</div>

        <div class="panel-row divider">
          <span class="lbl" title="현재 시점 Risk:Reward 비율 (TP1까지 기대수익 ÷ 손절 시 손실). 1.0 이상이면 양호">RR_NOW_TP1</span>
          <span class="val orange">{rr_now:.2f} ({rr_tag})</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="원래 플랜의 RR 배수. RR_NOW가 이것보다 작으면 진입 매력도 떨어짐">RR_MULT</span>
          <span class="val">{rr_mult:.1f}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="포지션당 최대 손실율. 손절 발동 시 잃을 자본 비중">MAX_LOSS</span>
          <span class="val red">{_fmt_pct(max_loss)}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="이 일수 안에 TP1 미돌파 시 시간 손절 (TIME_STOP) 발동">TIME_STOP</span>
          <span class="val">{time_stop}일</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="추천 보유 포지션 비중. 100% = 풀포지션, 0% = 신규매수 부적합">POSITION</span>
          <span class="val">{_fmt_pct(position, 0)}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="켈리 공식 기반 베팅 사이즈 (B = bankroll fraction). plan은 이론값, emp는 실측 보정값">KELLY_B</span>
          <span class="val orange">{kelly_final:.1f} <span class="pct-small">(plan {kelly_planned:.3f} / emp {kelly_empirical:.1f})</span></span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="실제 매수 권장 수량 및 금액 (KELLY_B × 가용 자본 / 진입가)">KELLY_수량</span>
          <span class="val">{qty}주 <span class="pct-small">({amount:.1f}만원)</span></span>
        </div>
      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# Step 2C: 좌측 사이드 패널 #2 — 추세 / MTF
# ═══════════════════════════════════════════════════

def _trend_arrow(val_str: str, fallback_int: int = 0) -> tuple:
    """추세 표시 문자열 → (화살표, 색상클래스)."""
    s = (val_str or "").strip()
    if s in ("▲", "O", "상승", "상"):
        return ("▲", "green")
    if s in ("▼", "X", "하락", "하"):
        return ("▼", "red")
    if fallback_int > 0:
        return ("▲", "green")
    if fallback_int < 0:
        return ("▼", "red")
    return ("—", "muted")


def render_v2_trend_mtf(n: dict):
    """[Step 2C] 좌측 사이드 패널 #2 — 추세 / MTF."""
    weekly_ma = n["weekly_ma20_above"]
    weekly_tr = n["weekly_trend"]
    mtf_w = n["mtf_weekly"]
    mtf_m = n["mtf_monthly"]
    st_dir = n["supertrend_dir"]
    st_val = n["supertrend_val"]
    above_ma20 = n["above_ma20"]
    hma20 = n["hma20"]
    hma_trend = n["hma_trend"]
    hma_on = n["hma_on"]
    igyukdo = n["igyukdo"]

    # 화살표 + 색상
    wma_arrow, wma_clr = _trend_arrow(weekly_ma)
    wtr_arrow, wtr_clr = _trend_arrow(weekly_tr)
    mtf_w_text = f"상승({mtf_w})" if mtf_w > 0 else (f"하락({mtf_w})" if mtf_w < 0 else "중립")
    mtf_w_clr = "green" if mtf_w > 0 else ("red" if mtf_w < 0 else "muted")
    mtf_m_text = f"상승({mtf_m})" if mtf_m > 0 else (f"하락({mtf_m})" if mtf_m < 0 else "중립")
    mtf_m_clr = "green" if mtf_m > 0 else ("red" if mtf_m < 0 else "muted")
    st_arrow, st_clr = _trend_arrow("", st_dir)
    ma20_arrow, ma20_clr = ("▲", "green") if above_ma20 > 0 else ("▼", "red")
    ma20_text = f"상 {ma20_arrow}" if above_ma20 > 0 else f"하 {ma20_arrow}"
    hma_arrow, hma_clr = _trend_arrow(hma_trend)

    ui.html(f'''
    <div class="sd-v2">
      <div class="panel">
        <div class="panel-title"><span class="num">2</span>추세 / MTF</div>

        <div class="panel-row">
          <span class="lbl" title="주봉 종가가 주봉 20일선 위에 있는가 (장기 추세 필터)">주봉 20선 상회</span>
          <span class="val {wma_clr}">{wma_arrow}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="주봉 캔들의 방향성 (3주 연속 상승/하락 등)">주봉 추세</span>
          <span class="val {wtr_clr}">{wtr_arrow}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="Multi-Timeframe 주봉 차원의 정렬 신호 (+1=상승, 0=중립, -1=하락)">MTF 주봉</span>
          <span class="val {mtf_w_clr}">{mtf_w_text}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="Multi-Timeframe 월봉 차원의 정렬 신호">MTF 월봉</span>
          <span class="val {mtf_m_clr}">{mtf_m_text}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="SuperTrend 지표값 (ATR x multiplier 기반 추세선). ▲=매수, ▼=매도">SUPERTREND</span>
          <span class="val {st_clr}">{st_arrow} {_fmt_won(st_val).replace("원","")}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="현재가가 20일 이동평균선 위에 있는가 (단기 추세 필터)">MA20 상회</span>
          <span class="val {ma20_clr}">{ma20_text}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="Hull Moving Average 20 (HMA — 부드러운 추세선). HMA_On은 종가가 HMA 위에 있는지">HMA20</span>
          <span class="val">{_fmt_won(hma20).replace("원","")} <span class="pct-small">(HMA_On {h_escape(hma_on or "—")})</span></span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="종가/20일선 괴리율 %. 양수=상승 괴리, 너무 크면 단기 과열 신호">이격도</span>
          <span class="val">{igyukdo:.2f}</span>
        </div>
      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# Step 2C: 좌측 사이드 패널 #3 — 모멘텀 & 수익률
# ═══════════════════════════════════════════════════

def _ret_color_class(val) -> str:
    """수익률 색상 클래스."""
    v = safe_float(val) or 0
    if v > 0:
        return "green"
    if v < 0:
        return "red"
    return "muted"


def render_v2_momentum(n: dict):
    """[Step 2C] 좌측 사이드 패널 #3 — 모멘텀 & 수익률."""
    rows = [
        ("ret_1d",   n["ret_1d"]),
        ("ret_5d",   n["ret_5d"]),
        ("ret_10d",  n["ret_10d"]),
        ("ret_20d",  n["ret_20d"]),
        ("ret_60d",  n["ret_60d"]),
        ("ret_120d", n["ret_120d"]),
    ]
    rel_rows = [
        ("rel_20d",  n["rel_20d"],  "시장대비"),
        ("rel_60d",  n["rel_60d"],  ""),
        ("rel_120d", n["rel_120d"], ""),
    ]
    bench_kospi = n["bench_kospi"]
    bench_kosdaq = n["bench_kosdaq"]
    # 보통 KOSPI/KOSDAQ 동일하면 합쳐서 표시
    if abs(bench_kospi - bench_kosdaq) < 0.01:
        bench_text = f"KOSPI/KOSDAQ {_fmt_pct_signed(bench_kospi)}"
    else:
        bench_text = f"KOSPI {_fmt_pct_signed(bench_kospi)} / KOSDAQ {_fmt_pct_signed(bench_kosdaq)}"

    rows_html = ""
    for label, val in rows:
        clr = _ret_color_class(val)
        rows_html += f'''
        <div class="panel-row dot">
          <span class="lbl">{label}</span>
          <span class="val {clr}">{_fmt_pct_signed(val, 2)}</span>
        </div>'''

    rel_html = ""
    for label, val, note in rel_rows:
        clr = _ret_color_class(val)
        note_html = f' <span class="pct-small">({note})</span>' if note else ''
        rel_html += f'''
        <div class="panel-row dot">
          <span class="lbl">{label}</span>
          <span class="val {clr}">{_fmt_pct_signed(val, 2)}{note_html}</span>
        </div>'''

    ui.html(f'''
    <div class="sd-v2">
      <div class="panel">
        <div class="panel-title"><span class="num">3</span>모멘텀 & 수익률</div>
        {rows_html}
        {rel_html}
        <div class="panel-row divider note">
          <span class="lbl" style="color: var(--text-dim);">벤치 60d</span>
          <span class="val muted">{bench_text}</span>
        </div>
      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# Step 2C: 좌측 사이드 패널 #4 — 수급 / 유동성
# ═══════════════════════════════════════════════════

def _fmt_eok(v) -> str:
    """억 단위 포맷. 0이면 '—'."""
    n = safe_float(v) or 0
    if n == 0:
        return "—"
    if n >= 10000:
        return f"{n/10000:.1f}조"
    if n >= 1000:
        return f"{n:,.0f}억"  # 1000억 이상은 정수
    if n >= 1:
        return f"{n:.1f}억"   # 1~999.9억은 소수 1자리 (이미지: 173.8억)
    return f"{n:.2f}억"


def _net_color_class(val) -> str:
    """순매수 색상 — 매매 판단용:
       양수: 초록 (매수)
       음수: 빨강 (매도)
       0:    회색 (중립/데이터 없음)
    """
    v = safe_float(val) or 0
    if v > 0:
        return "green"
    if v < 0:
        return "red"
    return "muted"


def render_v2_supply(n: dict):
    """[Step 2C] 좌측 사이드 패널 #4 — 수급 / 유동성."""
    turnover = n["turnover_eok"]
    mcap = n["mcap_eok"]
    foreign = n["foreign_net"]
    institution = n["institution_net"]
    major = n["major_net"]
    individual = n["individual_net"]
    strength = n["trade_strength"]
    v_power = n["v_power"]
    vol_quality = n["vol_quality"]

    # 거래대금 아이콘 (큰 거래는 🧊)
    turnover_icon = " 🧊" if turnover >= 100 else ""

    # 거래강도 색상 (1.0 이상 주황, 미만 회색)
    strength_clr = "orange" if strength >= 1.0 else "muted"
    # V_POWER 색상
    vp_clr = _ret_color_class(v_power)

    def _fmt_net(v):
        """순매수 금액 표시. 0이면 '0', 만 단위 이상이면 정수."""
        x = safe_float(v) or 0
        if x == 0:
            return "0"
        return f"{int(round(x)):+,}"

    ui.html(f'''
    <div class="sd-v2">
      <div class="panel">
        <div class="panel-title"><span class="num">4</span>수급 / 유동성</div>

        <div class="panel-row">
          <span class="lbl" title="당일 거래대금 (가격 x 수량). 100억 이상이면 유동성 양호">거래대금</span>
          <span class="val">{_fmt_eok(turnover)}{turnover_icon}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="시가총액 (전체 발행주식수 x 현재가)">시가총액</span>
          <span class="val">{_fmt_eok(mcap)}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="외국인 당일 순매수 금액. + 매수 우위, − 매도 우위">외인 순매수</span>
          <span class="val {_net_color_class(foreign)}">{_fmt_net(foreign)}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="기관 당일 순매수 금액 (연기금/보험/투신 등 합산)">기관 순매수</span>
          <span class="val {_net_color_class(institution)}">{_fmt_net(institution)}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="메이저 = 외인 + 기관. 스마트머니 흐름 추정">메이저 순매수</span>
          <span class="val {_net_color_class(major)}">{_fmt_net(major)}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="개인 당일 순매수 금액. 보통 메이저와 반대 방향">개인 순매수</span>
          <span class="val {_net_color_class(individual)}">{_fmt_net(individual)}</span>
        </div>

        <div class="panel-row divider">
          <span class="lbl" title="당일 거래량 / 평균 거래량 비율. 1.0 = 평균, 2.0+ = 거래 폭발">거래강도</span>
          <span class="val {strength_clr}">{strength:.2f}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="Volume Power. 상승 거래량 비중 - 하락 거래량 비중. + 매수세, − 매도세">V_POWER</span>
          <span class="val {vp_clr}">{v_power:+.2f}</span>
        </div>
        <div class="panel-row">
          <span class="lbl" title="거래량 품질 점수. 높을수록 추세 동반 거래 (단순 급등락 거래와 구분)">Vol_Quality</span>
          <span class="val">{vol_quality:.2f}</span>
        </div>
      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# Step 2D: 중앙 메인 캔들차트 (ECharts + markLine)
# ═══════════════════════════════════════════════════

# Indicator SSOT — 가능하면 백엔드 함수 사용 (차트 라인 = 시스템 점수 일치)
_use_backend_indicators = False
_backend_hma = None
try:
    from indicators import calc_supertrend as _backend_supertrend, calc_vwap as _backend_vwap
    _use_backend_indicators = True
    # HMA는 백엔드에 있을 수도 없을 수도 있음 — 시도
    try:
        from indicators import calc_hma as _backend_hma_fn
        _backend_hma = _backend_hma_fn
    except ImportError:
        pass
except ImportError:
    _backend_supertrend = None
    _backend_vwap = None


def _calc_hma(values: list, period: int = 20) -> list:
    """Hull Moving Average — 부드러운 추세선."""
    import math
    n = len(values)
    if n < period:
        return [None] * n
    half = period // 2
    sqrt_p = int(math.sqrt(period))

    def _wma(vals, p):
        """가중이동평균."""
        result = []
        for i in range(len(vals)):
            if i + 1 < p:
                result.append(None)
                continue
            window = vals[i + 1 - p: i + 1]
            weights = list(range(1, p + 1))
            wsum = sum(w * v for w, v in zip(weights, window))
            result.append(wsum / sum(weights))
        return result

    wma_half = _wma(values, half)
    wma_full = _wma(values, period)
    raw = []
    for h, f in zip(wma_half, wma_full):
        if h is None or f is None:
            raw.append(None)
        else:
            raw.append(2 * h - f)
    # raw의 None 제외 부분에 sqrt_p WMA
    non_none = [v for v in raw if v is not None]
    if len(non_none) < sqrt_p:
        return [None] * n
    raw_wma = _wma(non_none, sqrt_p)
    # 결과를 원래 인덱스로 매핑
    result = [None] * n
    start = n - len(raw_wma)
    for i, v in enumerate(raw_wma):
        result[start + i] = v
    return result


def _calc_supertrend(highs, lows, closes, period=10, multiplier=3.0):
    """SUPERTREND 계산."""
    n = len(closes)
    if n < period + 1:
        return [None] * n
    # True Range
    tr = []
    for i in range(n):
        if i == 0:
            tr.append(highs[i] - lows[i])
        else:
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
    # ATR (단순평균)
    atr = [None] * n
    for i in range(period - 1, n):
        atr[i] = sum(tr[i - period + 1: i + 1]) / period

    # SUPERTREND
    st = [None] * n
    direction = [1] * n  # 1=상승, -1=하락
    for i in range(period, n):
        if atr[i] is None:
            continue
        hl2 = (highs[i] + lows[i]) / 2
        up = hl2 - multiplier * atr[i]
        dn = hl2 + multiplier * atr[i]
        prev_st = st[i-1] if st[i-1] is not None else up

        if closes[i] > prev_st:
            direction[i] = 1
            st[i] = max(up, prev_st) if direction[i-1] == 1 else up
        else:
            direction[i] = -1
            st[i] = min(dn, prev_st) if direction[i-1] == -1 else dn
    return st


def _calc_vwap(highs, lows, closes, volumes) -> list:
    """VWAP (Volume Weighted Average Price) — 누적."""
    cum_pv = 0.0
    cum_v = 0.0
    result = []
    for h, l, c, v in zip(highs, lows, closes, volumes):
        typical = (h + l + c) / 3
        cum_pv += typical * v
        cum_v += v
        if cum_v > 0:
            result.append(cum_pv / cum_v)
        else:
            result.append(None)
    return result


def render_v2_chart(n: dict, ohlcv_df=None):
    """
    [Step 2D] 중앙 메인 캔들차트 — ECharts + markLine.

    한국식 색상: 상승 빨강 (#EF4444), 하락 파랑 (#3B82F6)
    오버레이: HMA20 (노랑), VWAP (흰 점선), SUPERTREND (초록)
    markLine: TP1/TP2/TP3 (파랑), 현재가/손절 (빨강), 매수가 (노랑)

    Args:
        n: normalize_stock_row() 결과
        ohlcv_df: pandas DataFrame with [시가, 고가, 저가, 종가, 거래량] columns
                  None이면 placeholder 표시
    """
    if ohlcv_df is None or len(ohlcv_df) == 0:
        # 디버그 정보는 admin 또는 환경변수일 때만 노출
        # (일반 사용자에게 CWD/__file__/경로 노출 방지)
        import os as _os
        show_debug = _os.getenv("DEBUG_STOCK_V2", "0") == "1"
        if not show_debug:
            try:
                # services.auth가 import되어 있을 때만 admin 체크
                from services.auth import get_auth_status as _get_auth
                show_debug = _get_auth() == "admin"
            except Exception:
                show_debug = False

        if show_debug:
            debug_info = _load_ohlcv_debug_info(n["code"])
            debug_block = (
                f'<pre style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px;'
                f' font-size: 10px; color: #9CA3AF; max-width: 600px;'
                f' white-space: pre-wrap; word-break: break-all;">{h_escape(debug_info)}</pre>'
            )
        else:
            debug_block = (
                '<div style="color: var(--text-dim); font-size: 11px;">'
                '잠시 후 다시 시도해 주세요.</div>'
            )

        ui.html(f'''
            <div style="background: var(--bg-card); border: 1px solid var(--border);
                        border-radius: 8px; padding: 20px; min-height: 400px;
                        display: flex; flex-direction: column; align-items: center;
                        justify-content: center; color: #6B7280; font-size: 12px;">
                <div style="font-size: 16px; margin-bottom: 12px;">📊 OHLCV 데이터 로드 실패</div>
                {debug_block}
            </div>
        ''')
        return

    # 데이터 추출
    df = ohlcv_df.copy()
    if hasattr(df.index, 'strftime'):
        dates = df.index.strftime("%Y-%m-%d").tolist()
    else:
        dates = [str(d) for d in df.index.tolist()]

    opens = df["시가"].tolist()
    highs = df["고가"].tolist()
    lows = df["저가"].tolist()
    closes = df["종가"].tolist()
    volumes = df["거래량"].tolist()

    # 캔들 데이터: [open, close, low, high]
    candle_data = [[o, c, l, h] for o, c, l, h in zip(opens, closes, lows, highs)]

    # 거래량 색상 (양봉 빨강 / 음봉 파랑)
    vol_data = []
    for i, v in enumerate(volumes):
        color = KOREA_UP if closes[i] >= opens[i] else KOREA_DOWN
        vol_data.append({"value": v, "itemStyle": {"color": color}})

    # 지표 계산 — SSOT 우선 (백엔드 indicators.py 사용 가능 시)

    # HMA — 백엔드 시도 후 로컬 fallback
    if _backend_hma is not None:
        try:
            import pandas as _pd
            c_ser = _pd.Series(closes, index=df.index)
            hma_ser = _backend_hma(c_ser, period=20)
            hma20 = [None if (v is None or (isinstance(v, float) and v != v)) else float(v)
                     for v in hma_ser.tolist()]
        except Exception:
            hma20 = _calc_hma(closes, 20)
    else:
        hma20 = _calc_hma(closes, 20)  # 로컬 fallback

    if _use_backend_indicators and _backend_vwap is not None and _backend_supertrend is not None:
        try:
            import pandas as _pd
            h_ser = _pd.Series(highs, index=df.index)
            l_ser = _pd.Series(lows, index=df.index)
            c_ser = _pd.Series(closes, index=df.index)
            v_ser = _pd.Series(volumes, index=df.index)
            # 백엔드 VWAP은 20일 롤링 (SSOT)
            vwap_ser = _backend_vwap(h_ser, l_ser, c_ser, v_ser, window=20)
            vwap = [None if (v is None or (isinstance(v, float) and v != v)) else float(v)
                    for v in vwap_ser.tolist()]
            # 백엔드 SUPERTREND
            st_ser, _trend = _backend_supertrend(h_ser, l_ser, c_ser, period=10, multiplier=3.0)
            supertrend = [None if (v is None or (isinstance(v, float) and v != v)) else float(v)
                          for v in st_ser.tolist()]
        except Exception:
            # 백엔드 호출 실패 시 로컬 fallback
            vwap = _calc_vwap(highs, lows, closes, volumes)
            supertrend = _calc_supertrend(highs, lows, closes, period=10, multiplier=3.0)
    else:
        # indicators.py import 실패 시 로컬 계산 fallback
        vwap = _calc_vwap(highs, lows, closes, volumes)
        supertrend = _calc_supertrend(highs, lows, closes, period=10, multiplier=3.0)

    # 가격 레벨 (markLine)
    entry = n["entry"]
    stop = n["stop"]
    tp1, tp2, tp3 = n["tp1"], n["tp2"], n["tp3"]
    close = n["close"]
    hma20_last = n["hma20"]
    supertrend_val = n["supertrend_val"]

    # markLine 데이터 (가격 + 라벨)
    marklines = []
    if tp3 > 0:
        marklines.append({
            "yAxis": tp3,
            "lineStyle": {"color": "#3B82F6", "type": "dashed", "width": 1, "opacity": 0.7},
            "label": {"formatter": f"TP3 {int(tp3):,} (+{n['tp3_gain_pct']:.1f}%)",
                      "color": "#3B82F6", "fontSize": 10, "position": "insideEndTop"}
        })
    if tp2 > 0:
        marklines.append({
            "yAxis": tp2,
            "lineStyle": {"color": "#3B82F6", "type": "dashed", "width": 1, "opacity": 0.7},
            "label": {"formatter": f"TP2 {int(tp2):,} (+{n['tp2_gain_pct']:.1f}%)",
                      "color": "#3B82F6", "fontSize": 10, "position": "insideEndTop"}
        })
    if tp1 > 0:
        marklines.append({
            "yAxis": tp1,
            "lineStyle": {"color": "#EF4444", "type": "dashed", "width": 1, "opacity": 0.8},
            "label": {"formatter": f"TP1 {int(tp1):,} (+{n['tp1_gain_pct']:.1f}%)",
                      "color": "#EF4444", "fontSize": 10, "position": "insideEndTop"}
        })
    if close > 0:
        marklines.append({
            "yAxis": close,
            "lineStyle": {"color": "#EF4444", "type": "solid", "width": 1.5, "opacity": 0.9},
            "label": {"formatter": f"종가 {int(close):,}",
                      "color": "#EF4444", "fontSize": 10, "fontWeight": "bold",
                      "position": "insideEndTop"}
        })
    if entry > 0 and abs(entry - close) > 1:
        # 매수가가 종가와 다를 때만 별도 표시
        marklines.append({
            "yAxis": entry,
            "lineStyle": {"color": "#FACC15", "type": "dashed", "width": 1, "opacity": 0.7},
            "label": {"formatter": f"매수가 {int(entry):,} (GAP {n['entry_gap_pct']:+.1f}%)",
                      "color": "#FACC15", "fontSize": 10, "position": "insideEndTop"}
        })
    if stop > 0:
        marklines.append({
            "yAxis": stop,
            "lineStyle": {"color": "#EF4444", "type": "dashed", "width": 1.2, "opacity": 0.8},
            "label": {"formatter": f"손절 {int(stop):,} ({n['stop_loss_pct']:+.1f}%)",
                      "color": "#EF4444", "fontSize": 10, "position": "insideEndTop"}
        })

    # Y축 min/max 계산 — 캔들 + 모든 markLine 가격을 포함하도록 강제 확장
    # 이걸 안 하면 "scale": True가 캔들 범위만 자동 fit해서 TP2/TP3가 화면 밖
    price_values = [v for v in highs + lows if v is not None and v > 0]
    price_values += [v for v in [tp1, tp2, tp3, close, entry, stop] if v > 0]
    if price_values:
        y_max_raw = max(price_values)
        y_min_raw = min(price_values)
        # 위 5% / 아래 5% 여백 추가
        y_padding = (y_max_raw - y_min_raw) * 0.05
        y_max = y_max_raw + y_padding
        y_min = max(0, y_min_raw - y_padding)
    else:
        y_max = None
        y_min = None

    # ECharts option
    option = {
        "backgroundColor": "transparent",
        "animation": False,
        "legend": {
            "show": True,
            "top": 5,
            "left": 60,
            "textStyle": {"color": "#9CA3AF", "fontSize": 10},
            "icon": "roundRect",
            "data": ["캔들", f"HMA20 ({int(hma20_last):,})" if hma20_last else "HMA20",
                    "VWAP", f"SUPERTREND ({int(supertrend_val):,})" if supertrend_val else "SUPERTREND"],
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross"},
            "backgroundColor": "rgba(15, 17, 23, 0.95)",
            "borderColor": "#2A2D38",
            "textStyle": {"color": "#fff", "fontSize": 11},
        },
        "grid": [
            {"left": 60, "right": 100, "top": 35, "height": "62%"},
            {"left": 60, "right": 100, "top": "72%", "height": "20%"},
        ],
        "xAxis": [
            {
                "type": "category",
                "data": dates,
                "gridIndex": 0,
                "boundaryGap": True,
                "axisLine": {"lineStyle": {"color": "#2A2D38"}},
                "axisLabel": {"show": False},
                "splitLine": {"show": False},
            },
            {
                "type": "category",
                "data": dates,
                "gridIndex": 1,
                "boundaryGap": True,
                "axisLine": {"lineStyle": {"color": "#2A2D38"}},
                "axisLabel": {"color": "#6B7280", "fontSize": 9},
                "splitLine": {"show": False},
            },
        ],
        "yAxis": [
            {
                "scale": True,
                "gridIndex": 0,
                "splitNumber": 6,
                "min": y_min if y_min is not None else None,
                "max": y_max if y_max is not None else None,
                "axisLine": {"lineStyle": {"color": "#2A2D38"}},
                "axisLabel": {"color": "#6B7280", "fontSize": 9,
                              "formatter": "{value}"},
                "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.04)"}},
                "position": "right",
            },
            {
                "scale": True,
                "gridIndex": 1,
                "splitNumber": 3,
                "axisLine": {"lineStyle": {"color": "#2A2D38"}},
                "axisLabel": {
                    "color": "#6B7280",
                    "fontSize": 9,
                    # NiceGUI ECharts: ':formatter' 콜론 prefix = JS 함수로 평가됨
                    ":formatter": (
                        "function(v){"
                        "if(v>=100000000) return (v/100000000).toFixed(0)+'억';"
                        "if(v>=10000) return (v/10000).toFixed(0)+'만';"
                        "return v.toFixed(0);"
                        "}"
                    ),
                },
                "splitLine": {"show": False},
                "position": "right",
            },
        ],
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0, 1], "start": 0, "end": 100},
        ],
        "series": [
            {
                "name": "캔들",
                "type": "candlestick",
                "data": candle_data,
                "itemStyle": {
                    "color": KOREA_UP,         # 상승 (양봉) 빨강
                    "color0": KOREA_DOWN,      # 하락 (음봉) 파랑
                    "borderColor": KOREA_UP,
                    "borderColor0": KOREA_DOWN,
                },
                "markLine": {
                    "symbol": ["none", "none"],
                    "data": marklines,
                    "silent": True,
                },
            },
            {
                "name": f"HMA20 ({int(hma20_last):,})" if hma20_last else "HMA20",
                "type": "line",
                "data": hma20,
                "smooth": True,
                "showSymbol": False,
                "lineStyle": {"color": "#FACC15", "width": 1.5, "opacity": 0.9},
                "z": 5,
            },
            {
                "name": "VWAP",
                "type": "line",
                "data": vwap,
                "showSymbol": False,
                "lineStyle": {"color": "white", "width": 1, "opacity": 0.7, "type": "dashed"},
                "z": 4,
            },
            {
                "name": f"SUPERTREND ({int(supertrend_val):,})" if supertrend_val else "SUPERTREND",
                "type": "line",
                "data": supertrend,
                "showSymbol": False,
                "lineStyle": {"color": "#10B981", "width": 1.2, "opacity": 0.85},
                "z": 4,
            },
            {
                "name": "거래량",
                "type": "bar",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "data": vol_data,
            },
        ],
    }

    # 차트 헤더
    ret_1d = n["ret_1d"]
    ret_clr = "var(--red)" if ret_1d < 0 else "var(--green)" if ret_1d > 0 else "var(--text-dim)"
    ui.html(f'''
    <div class="sd-v2">
      <div class="chart-card" style="background: var(--bg-card); border: 1px solid var(--border);
                                     border-radius: 8px; padding: 8px 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center;
                    margin-bottom: 6px; font-size: 10px;">
          <div style="display: flex; gap: 8px; align-items: center;">
            <span style="color: white; font-weight: 700;">📊 일봉 차트 (한국식 캔들)</span>
            <span style="color: {ret_clr}; font-weight: 700;">
              종가 {int(close):,}원 ({n["ret_1d"]:+.2f}%)
            </span>
          </div>
          <div style="color: var(--text-dim);">거래대금 {_fmt_eok(n["turnover_eok"])}</div>
        </div>
    ''')

    # ECharts 차트 렌더링
    chart = ui.echart(option).style("width: 100%; height: 400px; background: #15171F; border-radius: 4px;")

    ui.html('</div></div>')


# ═══════════════════════════════════════════════════
# Step 2E: 보조 차트 4개 (RSI/MFI/MACD/V_POWER) + 거래강도 게이지
# ═══════════════════════════════════════════════════

def _mini_sparkline_svg(values: list, color: str = "#10B981",
                        baseline: float = None, height: int = 50) -> str:
    """미니 sparkline SVG 생성 (보조 차트용)."""
    if not values or len(values) < 2:
        return ""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    if len(vals) < 2:
        return ""
    vmin = min(vals)
    vmax = max(vals)
    rng = vmax - vmin if vmax > vmin else 1
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = i / (n - 1) * 100
        y = height - ((v - vmin) / rng * (height - 8)) - 4
        pts.append(f"{x:.1f},{y:.1f}")
    path = "M " + " L ".join(pts)
    baseline_line = ""
    if baseline is not None and vmin <= baseline <= vmax:
        by = height - ((baseline - vmin) / rng * (height - 8)) - 4
        baseline_line = (
            f'<line x1="0" y1="{by:.1f}" x2="100" y2="{by:.1f}" '
            f'stroke="rgba(255,255,255,0.1)" stroke-dasharray="2,2" stroke-width="0.5"/>'
        )
    return (
        f'<svg viewBox="0 0 100 {height}" preserveAspectRatio="none" '
        f'style="width: 100%; height: {height}px;">'
        f'{baseline_line}'
        f'<path d="{path}" stroke="{color}" stroke-width="1.5" fill="none"/>'
        f'</svg>'
    )


def _mini_histogram_svg(values: list, height: int = 50) -> str:
    """히스토그램 (MACD slope, V_POWER용)."""
    if not values or len(values) < 2:
        return ""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    if len(vals) < 2:
        return ""
    vmax_abs = max(abs(min(vals)), abs(max(vals)), 0.1)
    n = len(vals)
    bar_w = 100 / n
    bars = []
    for i, v in enumerate(vals):
        x = i * bar_w
        if v >= 0:
            h = (v / vmax_abs) * (height / 2 - 2)
            y = height / 2 - h
            color = "#10B981"
        else:
            h = (abs(v) / vmax_abs) * (height / 2 - 2)
            y = height / 2
            color = "#EF4444"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.7:.1f}" '
            f'height="{h:.1f}" fill="{color}"/>'
        )
    zero_line = (
        f'<line x1="0" y1="{height/2}" x2="100" y2="{height/2}" '
        f'stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/>'
    )
    return (
        f'<svg viewBox="0 0 100 {height}" preserveAspectRatio="none" '
        f'style="width: 100%; height: {height}px;">'
        f'{zero_line}{"".join(bars)}</svg>'
    )


def render_v2_sub_charts(n: dict, ohlcv_df=None):
    """
    [Step 2E] 보조 차트 4개 + 거래강도 게이지 (가로 5분할).

    RSI(14) / MFI(14) / MACD Slope / V_POWER + 거래강도 게이지
    """
    rsi14 = n["rsi14"]
    mfi14 = n["mfi14"]
    macd_slope = n["macd_slope_pct"]
    v_power = n["v_power"]
    strength = n["trade_strength"]

    # OHLCV에서 sparkline 시계열 추출 (간이 계산)
    rsi_series = []
    mfi_series = []
    macd_series = []
    vpow_series = []

    if ohlcv_df is not None and len(ohlcv_df) > 30:
        try:
            import pandas as pd
            closes = ohlcv_df["종가"]
            highs = ohlcv_df["고가"]
            lows = ohlcv_df["저가"]
            vols = ohlcv_df["거래량"]

            # RSI 간이 계산 (최근 30일)
            delta = closes.diff()
            up = delta.where(delta > 0, 0).rolling(14).mean()
            down = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = up / down.replace(0, 1e-10)
            rsi_full = (100 - 100 / (1 + rs))
            rsi_series = rsi_full.tail(30).tolist()

            # MACD slope 간이 (12-26 EMA)
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_hist = (macd_line - macd_line.ewm(span=9, adjust=False).mean())
            macd_series = macd_hist.tail(20).tolist()

            # MFI 간이 (close 기반 spark)
            mfi_series = closes.tail(30).pct_change().fillna(0).rolling(3).mean().fillna(0).tolist()

            # V_POWER 시계열 (거래량 기반 단순 추정)
            vol_norm = (vols - vols.rolling(20).mean()) / vols.rolling(20).std().replace(0, 1)
            vpow_series = vol_norm.tail(20).fillna(0).tolist()
        except Exception:
            pass

    # 폴백: 빈 값일 때 일관된 점선
    if not rsi_series:
        rsi_series = [rsi14] * 20
    if not mfi_series:
        mfi_series = [mfi14] * 20
    if not macd_series:
        macd_series = [macd_slope] * 10
    if not vpow_series:
        vpow_series = [v_power] * 10

    # 색상
    rsi_clr = "#F59E0B"
    mfi_clr = "#10B981"
    macd_clr = "#EF4444" if macd_slope < 0 else "#10B981"
    vpow_clr = "#EF4444" if v_power < 0 else "#10B981"

    # 태그
    rsi_tag = f"RSI_Rising {n['rsi_rising']}" if n.get("rsi_rising") else "—"
    mfi_tag = "강세 진입" if mfi14 >= 70 else ("약세" if mfi14 < 30 else "중립")
    macd_tag = "음전 시작" if macd_slope < 0 else "양전"
    vpow_tag = "매수세 약화" if v_power < 0 else "매수세 강화"

    # 게이지 SVG (반원, 거래강도)
    # 1.22 / 2.0 max 기준
    strength_pct = min(strength / 2.0, 1.0)
    # 반원: 180° (왼쪽 -90° ~ 오른쪽 +90°)
    angle = -90 + 180 * strength_pct
    import math
    angle_rad = math.radians(angle)
    needle_x = 60 + 38 * math.cos(angle_rad - math.radians(90))
    needle_y = 60 - 38 * math.sin(angle_rad + math.radians(90)) + 38

    strength_clr = "#10B981" if strength >= 1.0 else ("#F59E0B" if strength >= 0.7 else "#EF4444")

    ui.html(f'''
    <div class="sd-v2" style="margin-top: 8px; width: 100%;">
      <div class="v2-sub-charts" style="display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)) 130px; gap: 8px; width: 100%;">

        <!-- RSI -->
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 8px;">
          <div style="color: var(--text-gray); font-size: 10px; font-weight: 700;">RSI(14)</div>
          <div style="color: {rsi_clr}; font-size: 22px; font-weight: 900; line-height: 1; margin: 4px 0;">{rsi14:.1f}</div>
          {_mini_sparkline_svg(rsi_series, rsi_clr, baseline=50)}
          <div style="color: var(--text-dim); font-size: 9px; margin-top: 4px;">{rsi_tag}</div>
        </div>

        <!-- MFI -->
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 8px;">
          <div style="color: var(--text-gray); font-size: 10px; font-weight: 700;">MFI(14)</div>
          <div style="color: {mfi_clr}; font-size: 22px; font-weight: 900; line-height: 1; margin: 4px 0;">{mfi14:.1f}</div>
          {_mini_sparkline_svg(mfi_series, mfi_clr)}
          <div style="color: var(--text-dim); font-size: 9px; margin-top: 4px;">{mfi_tag}</div>
        </div>

        <!-- MACD Slope -->
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 8px;">
          <div style="color: var(--text-gray); font-size: 10px; font-weight: 700;">MACD Slope</div>
          <div style="color: {macd_clr}; font-size: 22px; font-weight: 900; line-height: 1; margin: 4px 0;">{macd_slope:+.2f}%</div>
          {_mini_histogram_svg(macd_series)}
          <div style="color: var(--text-dim); font-size: 9px; margin-top: 4px;">{macd_tag}</div>
        </div>

        <!-- V_POWER -->
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 8px;">
          <div style="color: var(--text-gray); font-size: 10px; font-weight: 700;">V_POWER</div>
          <div style="color: {vpow_clr}; font-size: 22px; font-weight: 900; line-height: 1; margin: 4px 0;">{v_power:+.2f}</div>
          {_mini_histogram_svg(vpow_series)}
          <div style="color: var(--text-dim); font-size: 9px; margin-top: 4px;">{vpow_tag}</div>
        </div>

        <!-- 거래강도 게이지 -->
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
                    padding: 8px; display: flex; flex-direction: column; align-items: center;">
          <div style="color: var(--text-gray); font-size: 10px; font-weight: 700; margin-bottom: 4px;">거래강도</div>
          <svg viewBox="0 0 120 70" style="height: 60px; width: 110px;">
            <!-- 반원 배경 -->
            <path d="M 10 60 A 50 50 0 0 1 110 60" stroke="rgba(255,255,255,0.08)" stroke-width="10" fill="none"/>
            <!-- 색 구간: 빨강(0-0.7) / 주황(0.7-1.0) / 초록(1.0+) -->
            <path d="M 10 60 A 50 50 0 0 1 45 16" stroke="#EF4444" stroke-width="10" fill="none" opacity="0.75"/>
            <path d="M 45 16 A 50 50 0 0 1 75 13" stroke="#F59E0B" stroke-width="10" fill="none" opacity="0.75"/>
            <path d="M 75 13 A 50 50 0 0 1 110 60" stroke="#10B981" stroke-width="10" fill="none" opacity="0.75"/>
            <!-- 바늘 -->
            <line x1="60" y1="60" x2="{needle_x:.1f}" y2="{needle_y:.1f}" stroke="white" stroke-width="2"/>
            <circle cx="60" cy="60" r="3" fill="white"/>
          </svg>
          <div style="color: {strength_clr}; font-size: 22px; font-weight: 900; line-height: 1; margin-top: -4px;">{strength:.2f}</div>
          <div style="color: var(--text-dim); font-size: 9px; margin-top: 2px;">평균 {strength:.1f}배</div>
        </div>

      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# Step 2F: 보조차트 아래 3분할 패널 (수익률현황 / 핵심레벨 / 리스크뉴스)
# ═══════════════════════════════════════════════════

def render_v2_returns_levels_risk(n: dict):
    """
    [Step 2F] 차트 + 보조차트 아래 3분할 패널.

    A. 수익률 현황 — ret_1d/5d/10d/20d/60d/120d + rel_20/60/120
    B. 핵심 레벨 — 현재가/매수가/HMA20/VWAP/SUPERTREND/손절/TP1/TP2/TP3 (점 색상)
    C. 리스크 / 뉴스 — NEWS_SCORE + 뉴스 본문 + ROUTE_REASON + BB_Expanding + IS_SWING_SUPPORT
    """
    # A. 수익률 현황
    returns_rows = [
        ("1일", n["ret_1d"]),
        ("5일", n["ret_5d"]),
        ("10일", n["ret_10d"]),
        ("20일", n["ret_20d"]),
        ("60일", n["ret_60d"]),
        ("120일", n["ret_120d"]),
        ("rel_20d", n["rel_20d"]),
        ("rel_60d", n["rel_60d"]),
        ("rel_120d", n["rel_120d"]),
    ]
    ret_html = ""
    for label, val in returns_rows:
        clr = "var(--green)" if val > 0 else ("var(--red)" if val < 0 else "var(--text-dim)")
        ret_html += f'''
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 3px 0; font-size: 11px;">
          <span style="color: var(--text-gray);">{label}</span>
          <span style="color: {clr}; font-weight: 700; font-variant-numeric: tabular-nums;">{val:+.2f}%</span>
        </div>'''

    # B. 핵심 레벨 (점 색상 + 가격)
    levels = [
        ("#EF4444", "현재가",     n["close"]),
        ("#FACC15", "추천매수가",  n["entry"]),
        ("#3B82F6", "HMA20",      n["hma20"]),
        ("#06B6D4", "VWAP",       n.get("vwap", 0)),
        ("#10B981", "SUPERTREND", n.get("supertrend_val_raw", n["supertrend_val"])),
        ("#EF4444", "손절",       n["stop"]),
        ("#10B981", "TP1",        n["tp1"]),
        ("#10B981", "TP2",        n["tp2"]),
        ("#10B981", "TP3",        n["tp3"]),
    ]
    level_html = ""
    for color, label, price in levels:
        if not price or price <= 0:
            continue
        level_html += f'''
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 3px 0; font-size: 11px;">
          <span style="display: flex; align-items: center; gap: 6px; color: var(--text-gray);">
            <span style="width: 8px; height: 8px; background: {color}; border-radius: 50%; display: inline-block;"></span>
            {label}
          </span>
          <span style="color: var(--text-white); font-weight: 700; font-variant-numeric: tabular-nums;">{int(price):,}</span>
        </div>'''

    # C. 리스크 / 뉴스
    news_score = n.get("news_score", 0)
    news_clr = "var(--green)" if news_score > 0 else ("var(--red)" if news_score < 0 else "var(--text-dim)")
    news_tag = "긍정" if news_score > 0 else ("악한 부정" if news_score < -0.2 else "부정" if news_score < 0 else "중립")
    macro_risk = n.get("macro_risk", "") or "—"
    market_breadth = n.get("market_breadth", 0)
    route_reason = n.get("route_reason", "") or "—"
    news_reason = n.get("news_reason", "") or "—"
    # 뉴스 본문 짧게 자르기
    if len(news_reason) > 80:
        news_reason_short = news_reason[:78] + "…"
    else:
        news_reason_short = news_reason
    # [뉴스] 접두사 제거
    news_reason_short = news_reason_short.replace("[뉴스]", "").strip()
    bb_expanding = n.get("bb_expanding", 0)
    bb_text = "확장 중" if bb_expanding else "확장 둔화"
    is_swing = n.get("is_swing_support", False)
    swing_text = "True" if is_swing else "False"
    swing_clr = "var(--green)" if is_swing else "var(--red)"

    # 3분할 패널 HTML
    ui.html(f'''
    <div class="sd-v2" style="margin-top: 8px; width: 100%;">
      <div class="v2-three-split" style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; width: 100%;">

        <!-- A. 수익률 현황 -->
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-width: 0;">
          <div style="color: var(--green); font-size: 12px; font-weight: 800; text-align: center; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border);">
            A. 수익률 현황
          </div>
          {ret_html}
        </div>

        <!-- B. 핵심 레벨 -->
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-width: 0;">
          <div style="color: var(--orange); font-size: 12px; font-weight: 800; text-align: center; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border);">
            B. 핵심 레벨
          </div>
          {level_html}
        </div>

        <!-- C. 리스크 / 뉴스 -->
        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-width: 0;">
          <div style="color: var(--red); font-size: 12px; font-weight: 800; text-align: center; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid var(--border);">
            C. 리스크 / 뉴스 ⚠️
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0; font-size: 11px;">
            <span style="color: var(--text-gray);">NEWS_SCORE</span>
            <span style="color: {news_clr}; font-weight: 700;">{news_score:+.2f} <span style="font-size: 9px; color: var(--text-dim);">({news_tag})</span></span>
          </div>
          <div style="padding: 6px 0; font-size: 10px; color: var(--text-gray); border-top: 1px dashed var(--border); margin-top: 4px;">
            <div style="color: var(--text-dim); margin-bottom: 2px;">뉴스:</div>
            <div style="line-height: 1.5;">{h_escape(news_reason_short)}</div>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0; font-size: 11px; border-top: 1px dashed var(--border); margin-top: 4px;">
            <span style="color: var(--text-gray);">ROUTE_REASON</span>
            <span style="color: var(--text-white); font-weight: 600; font-size: 10px;">{h_escape(route_reason)}</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center; padding: 3px 0; font-size: 11px;">
            <span style="color: var(--text-gray);">MACRO_RISK</span>
            <span style="color: var(--text-white); font-weight: 600;">{h_escape(macro_risk)}</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center; padding: 3px 0; font-size: 11px;">
            <span style="color: var(--text-gray);">BB_Expanding</span>
            <span style="color: var(--text-white); font-weight: 600;">{bb_expanding} <span style="font-size: 9px; color: var(--text-dim);">({bb_text})</span></span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center; padding: 3px 0; font-size: 11px;">
            <span style="color: var(--text-gray);">IS_SWING_SUPPORT</span>
            <span style="color: {swing_clr}; font-weight: 700;">{swing_text}</span>
          </div>
        </div>

      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# Step 2F: 시나리오 A / B / C (3개 카드)
# ═══════════════════════════════════════════════════

def render_v2_scenarios(n: dict):
    """
    [Step 2F] 시나리오 3개 카드 — 기본 / 보수 / 리스크.

    A. 기본 시나리오 (📈) — 진입 후 TP1→TP2→TP3 분할익절
    B. 보수 시나리오 (🔍) — HMA20 회복/안착 확인 후 대응
    C. 리스크 시나리오 (❌) — 손절 이탈 시 7일 내 미돌파면 TIME_STOP
    """
    entry = int(n["entry"]) if n["entry"] else 0
    stop = int(n["stop"]) if n["stop"] else 0
    tp1 = int(n["tp1"]) if n["tp1"] else 0
    tp2 = int(n["tp2"]) if n["tp2"] else 0
    tp3 = int(n["tp3"]) if n["tp3"] else 0
    hma20 = int(n["hma20"]) if n["hma20"] else 0
    time_stop = int(n["time_stop_days"]) if n["time_stop_days"] else 7
    route = n.get("route", "")
    is_overheat = route == "OVERHEAT"
    qty = int(n["qty"]) if n["qty"] else 0

    # 시나리오 A — 기본 (매수 가능할 때만 의미)
    if qty > 0 and not is_overheat:
        scenario_a_lines = [
            "기본 시나리오 —",
            f"{entry:,} 진입 후",
            f"{tp1:,} → {tp2:,} → {tp3:,}",
            "분할익절",
        ]
        scenario_a_clr = "#10B981"
    else:
        scenario_a_lines = [
            "기본 시나리오 —",
            f"{route or 'WAIT'} 상태",
            "신규 매수 부적합",
            "관망 권고",
        ]
        scenario_a_clr = "#6B7280"

    # 시나리오 B — 보수 (HMA20 회복 대기)
    scenario_b_lines = [
        "보수 시나리오 —",
        f"HMA20 {hma20:,}",
        "회복/안착 확인 후 대응",
    ]

    # 시나리오 C — 리스크
    scenario_c_lines = [
        "리스크 시나리오 —",
        f"{stop:,} 이탈 시 손절,",
        f"{time_stop}일 내 미돌파면 TIME_STOP",
    ]

    def _scenario_html(icon, title_color, lines, big_idx=2):
        """시나리오 카드 HTML 생성. big_idx 줄을 크게 표시."""
        html_lines = []
        for i, line in enumerate(lines):
            if i == 0:
                html_lines.append(
                    f'<div style="color: {title_color}; font-size: 13px; font-weight: 800; margin-bottom: 8px;">{h_escape(line)}</div>'
                )
            elif i == big_idx and big_idx >= 0:
                html_lines.append(
                    f'<div style="color: var(--text-white); font-size: 14px; font-weight: 700; margin: 4px 0;">{h_escape(line)}</div>'
                )
            else:
                html_lines.append(
                    f'<div style="color: var(--text-gray); font-size: 11px; margin: 3px 0;">{h_escape(line)}</div>'
                )
        return "".join(html_lines)

    ui.html(f'''
    <div class="sd-v2" style="margin-top: 8px; width: 100%;">
      <div class="v2-scenarios" style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; width: 100%;">

        <!-- 시나리오 A (기본) -->
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(16, 185, 129, 0.04));
                    border: 1px solid {scenario_a_clr}; border-radius: 8px; padding: 14px; min-width: 0;
                    min-height: 130px; display: flex; flex-direction: column; justify-content: center;">
          <div style="font-size: 16px; margin-bottom: 6px;">📈 시나리오 A <span style="color: var(--text-dim); font-size: 10px;">(기본)</span></div>
          {_scenario_html("📈", scenario_a_clr, scenario_a_lines, big_idx=2)}
        </div>

        <!-- 시나리오 B (보수) -->
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(59, 130, 246, 0.04));
                    border: 1px solid #3B82F6; border-radius: 8px; padding: 14px; min-width: 0;
                    min-height: 130px; display: flex; flex-direction: column; justify-content: center;">
          <div style="font-size: 16px; margin-bottom: 6px;">🔍 시나리오 B <span style="color: var(--text-dim); font-size: 10px;">(보수)</span></div>
          {_scenario_html("🔍", "#3B82F6", scenario_b_lines, big_idx=1)}
        </div>

        <!-- 시나리오 C (리스크) -->
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.04));
                    border: 1px solid #EF4444; border-radius: 8px; padding: 14px; min-width: 0;
                    min-height: 130px; display: flex; flex-direction: column; justify-content: center;">
          <div style="font-size: 16px; margin-bottom: 6px;">❌ 시나리오 C <span style="color: var(--text-dim); font-size: 10px;">(리스크)</span></div>
          {_scenario_html("❌", "#EF4444", scenario_c_lines, big_idx=2)}
        </div>

      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# Step 2F: 최종 판정 띠 (페이지 최하단)
# ═══════════════════════════════════════════════════

def render_v2_final_verdict(n: dict, rank: int = 0, total: int = 0):
    """
    [Step 2F] 최하단 최종 판정 띠.

    예: '👑 최종 판정 | 601개 중 1위, 실성능 검증된 콤보 등급.
         구조 최강 + 진입가 적정 + RR 1.34. 다만 뉴스 심리 리스크와 MFI 과열은 체크.'
    """
    name = h_escape(n["name"])
    rr = n["rr_now_tp1"]
    route = n.get("route", "")
    is_overheat = route == "OVERHEAT"
    struct_s = n["struct_score"]
    timing_s = n["timing_score"]
    ai_s = n["ai_score"]
    elite_score = n["elite_score"]
    top_pick = n.get("top_pick", False)
    qty = int(n["qty"]) if n["qty"] else 0
    mfi = n.get("mfi14", 0)
    news_score = n.get("news_score", 0)

    # 강점 리스트
    strengths = []
    if struct_s >= 90:
        strengths.append("구조 최강")
    elif struct_s >= 70:
        strengths.append("구조 양호")
    if 0 < abs(n["entry_gap_pct"]) <= 3:
        strengths.append("진입가 적정")
    elif n["entry_gap_pct"] == 0:
        strengths.append("진입가 적정")
    if rr >= 1.3:
        strengths.append(f"RR {rr:.2f}")
    elif rr >= 1.0:
        strengths.append(f"RR {rr:.2f} 양호")

    # 리스크 리스트
    risks = []
    if mfi >= 70:
        risks.append("MFI 과열")
    if news_score < -0.2:
        risks.append("뉴스 심리 리스크")
    if is_overheat:
        risks.append("ROUTE 과열")
    if rr < 1.0:
        risks.append("RR 부족")
    if qty == 0:
        risks.append("KELLY 0주")

    # 등급 라벨
    if top_pick and rr >= 1.3 and not is_overheat:
        verdict_label = "실성능 검증된 콤보 등급"
        verdict_color = "linear-gradient(135deg, #FFD700, #FFA500)"
        icon = "👑"
    elif is_overheat or rr < 0.5:
        verdict_label = "신규매수 부적합"
        verdict_color = "linear-gradient(135deg, #EF4444, #DC2626)"
        icon = "⚠️"
    elif rr >= 1.0:
        verdict_label = "양호한 진입 후보"
        verdict_color = "linear-gradient(135deg, #10B981, #059669)"
        icon = "✓"
    else:
        verdict_label = "관망 권고"
        verdict_color = "linear-gradient(135deg, #6B7280, #4B5563)"
        icon = "—"

    rank_text = f"{total}개 중 {rank}위" if rank and total else ""
    strengths_text = " + ".join(strengths) if strengths else "—"
    risks_text = f"다만 {', '.join(risks)}은 체크" if risks else "리스크 신호 없음"

    ui.html(f'''
    <div class="sd-v2" style="margin-top: 16px; margin-bottom: 12px; width: 100%;">
      <div class="v2-final-verdict" style="background: {verdict_color}; border-radius: 12px; padding: 18px 24px;
                  display: flex; align-items: center; gap: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
        <div style="font-size: 36px; flex-shrink: 0;">{icon}</div>
        <div style="flex: 1; min-width: 0;">
          <div style="color: rgba(0,0,0,0.85); font-size: 13px; font-weight: 700; margin-bottom: 4px;">
            최종 판정 | {h_escape(rank_text)}{', ' if rank_text else ''}{h_escape(verdict_label)}.
          </div>
          <div style="color: rgba(0,0,0,0.75); font-size: 11px; font-weight: 600; line-height: 1.5;">
            {h_escape(strengths_text)}. {h_escape(risks_text)}.
          </div>
        </div>
        <div style="font-size: 36px; flex-shrink: 0;">🚀</div>
      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# Step 2E: 우측 레이더 차트 (5축)
# ═══════════════════════════════════════════════════

def render_v2_radar(n: dict):
    """
    [Step 2E] 우측 레이더 차트 — 5축: STRUCT/TIMING/AI/BALANCE/TRIGGER.

    SVG 직접 그리기 (Plotly보다 가벼움, 정확한 위치 제어).
    """
    import math
    struct_s = n["struct_score"]
    timing_s = n["timing_score"]
    ai_s = n["ai_score"]
    balance_s = n["balance_score"]
    trigger_s = n["trigger_score"]

    # 5축, 각각 0-100 스케일
    axes = [
        ("STRUCT",  struct_s,  -90),    # 최상단
        ("TIMING",  timing_s,  -18),    # 오른쪽 위
        ("AI",      ai_s,       54),    # 오른쪽 아래
        ("BALANCE", balance_s, 126),    # 왼쪽 아래
        ("TRIGGER", trigger_s, 198),    # 왼쪽 위
    ]

    # 그리기 좌표 (캔버스 240x240, 중심 120,120, 반지름 키움 + 라벨 여백 유지)
    cx, cy = 120, 120
    max_r = 90  # 75 → 90 (펜타곤 크게)
    label_r = 110  # 라벨 위치 — 데이터 폴리곤보다 멀리

    # 펜타곤 격자 4단 (25, 50, 75, 100%)
    def _pentagon(scale: float) -> str:
        r = max_r * scale
        pts = []
        for i, (_, _, angle) in enumerate(axes):
            rad = math.radians(angle)
            x = cx + r * math.cos(rad)
            y = cy + r * math.sin(rad)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    grid_polys = ""
    for scale in [1.0, 0.75, 0.5, 0.25]:
        grid_polys += (
            f'<polygon points="{_pentagon(scale)}" fill="none" '
            f'stroke="rgba(255,255,255,0.08)" stroke-width="1"/>'
        )

    # 축 라인
    axis_lines = ""
    label_html = ""
    for label, val, angle in axes:
        rad = math.radians(angle)
        ex = cx + max_r * math.cos(rad)
        ey = cy + max_r * math.sin(rad)
        axis_lines += (
            f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="rgba(255,255,255,0.1)" stroke-width="1"/>'
        )
        # 라벨 위치 (축 끝보다 살짝 바깥 — 잘림 방지)
        lr = label_r
        lx = cx + lr * math.cos(rad)
        ly = cy + lr * math.sin(rad) + 4  # +4: baseline 보정
        # 라벨 정렬 보정 (각도에 따라)
        anchor = "middle"
        if math.cos(rad) > 0.3:
            anchor = "start"
        elif math.cos(rad) < -0.3:
            anchor = "end"
        label_html += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'fill="white" font-size="10" font-weight="700">{label} {val:.1f}</text>'
        )

    # 데이터 폴리곤
    data_pts = []
    dot_circles = ""
    for label, val, angle in axes:
        rad = math.radians(angle)
        r = max_r * (val / 100)
        x = cx + r * math.cos(rad)
        y = cy + r * math.sin(rad)
        data_pts.append(f"{x:.1f},{y:.1f}")
        dot_circles += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#8B5CF6"/>'

    data_poly = " ".join(data_pts)

    ui.html(f'''
    <div class="sd-v2">
      <div style="background: var(--bg-card); border: 1px solid var(--border);
                  border-radius: 8px; padding: 12px;">
        <div style="color: var(--purple); font-size: 12px; font-weight: 800;
                    text-align: center; margin-bottom: 8px; padding-bottom: 6px;
                    border-bottom: 1px solid var(--border);">
          3축 밸런스 레이더
        </div>
        <svg viewBox="-70 -20 380 290" style="width: 100%; height: 320px;">
          {grid_polys}
          {axis_lines}
          <polygon points="{data_poly}" fill="rgba(139, 92, 246, 0.25)"
                   stroke="#8B5CF6" stroke-width="2"/>
          {dot_circles}
          {label_html}
        </svg>
        <!-- 5축 점수 요약 -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
                    margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--border);">
          <div style="font-size: 10px; color: var(--text-gray);">STRUCT</div>
          <div style="font-size: 11px; color: var(--green); font-weight: 700; text-align: right;">{struct_s:.1f}</div>
          <div style="font-size: 10px; color: var(--text-gray);">TIMING</div>
          <div style="font-size: 11px; color: var(--orange); font-weight: 700; text-align: right;">{timing_s:.1f}</div>
          <div style="font-size: 10px; color: var(--text-gray);">AI</div>
          <div style="font-size: 11px; color: var(--purple); font-weight: 700; text-align: right;">{ai_s:.1f}</div>
          <div style="font-size: 10px; color: var(--text-gray);">BALANCE</div>
          <div style="font-size: 11px; color: var(--cyan); font-weight: 700; text-align: right;">{balance_s:.1f}</div>
          <div style="font-size: 10px; color: var(--text-gray);">TRIGGER</div>
          <div style="font-size: 11px; color: var(--yellow); font-weight: 700; text-align: right;">{trigger_s:.1f}</div>
        </div>
      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# Step 2F: 우측 추가 카드 (AXIS_GAP 큰 카드 + 투자자 가이드)
# ═══════════════════════════════════════════════════

def render_v2_right_axisgap(n: dict):
    """[Step 2F] 우측 컬럼 — AXIS_GAP 큰 카드 (비교 이미지 25.9 스타일)."""
    axis_gap = n.get("axis_gap", 0)
    # 등급 분류
    if axis_gap < 10:
        gap_label = "매우 양호"
        gap_color = "#10B981"  # green
    elif axis_gap < 20:
        gap_label = "양호"
        gap_color = "#06B6D4"  # cyan
    elif axis_gap < 30:
        gap_label = "양호"
        gap_color = "#06B6D4"
    elif axis_gap < 40:
        gap_label = "주의"
        gap_color = "#F59E0B"  # orange
    else:
        gap_label = "리스크"
        gap_color = "#EF4444"  # red

    ui.html(f'''
    <div class="sd-v2">
      <div style="background: linear-gradient(135deg, {gap_color}22, {gap_color}08);
                  border: 1px solid {gap_color}; border-radius: 8px; padding: 14px;
                  text-align: center; min-height: 140px;
                  display: flex; flex-direction: column; justify-content: center;">
        <div style="color: var(--text-gray); font-size: 11px; font-weight: 700; margin-bottom: 6px;">AXIS_GAP</div>
        <div style="color: {gap_color}; font-size: 36px; font-weight: 900; line-height: 1;">{axis_gap:.1f}</div>
        <div style="color: var(--text-gray); font-size: 11px; margin-top: 6px;">{gap_label}</div>
      </div>
    </div>
    ''')


def render_v2_right_guide(n: dict):
    """[Step 2F] 우측 컬럼 — 투자자 가이드 카드."""
    qty = int(n["qty"]) if n["qty"] else 0
    stop = int(n["stop"]) if n["stop"] else 0
    tp1 = int(n["tp1"]) if n["tp1"] else 0
    time_stop = int(n["time_stop_days"]) if n["time_stop_days"] else 7
    route = n.get("route", "")
    is_overheat = route == "OVERHEAT"
    position = n.get("position_pct", 0)

    # 케이스별 가이드
    if qty > 0 and not is_overheat:
        # 매수 가능
        guide_lines = [
            ("보유자", f"{stop:,} 손절선 관리"),
            ("신규 진입", "현재가 기준 분할 접근 가능"),
            ("1차 목표", f"TP1 {tp1:,}"),
            ("시간 기준", f"{time_stop}일"),
        ]
        title_color = "#10B981"
    else:
        # 매수 부적합 (OVERHEAT / KELLY 0)
        guide_lines = [
            ("보유자", f"{stop:,} 손절선 사수"),
            ("신규 진입", "❌ 부적합 (대기)"),
            ("재진입 조건", "RR 회복 + ROUTE 정상화"),
            ("시간 기준", f"{time_stop}일"),
        ]
        title_color = "#EF4444"

    rows_html = ""
    for label, value in guide_lines:
        rows_html += f'''
        <div style="display: flex; justify-content: space-between; align-items: flex-start; padding: 4px 0; font-size: 11px;">
          <span style="color: var(--text-gray); flex-shrink: 0;">✓ {label}</span>
          <span style="color: var(--text-white); font-weight: 600; text-align: right; padding-left: 8px;">{h_escape(value)}</span>
        </div>'''

    ui.html(f'''
    <div class="sd-v2">
      <div style="background: var(--bg-card); border: 1px solid var(--border);
                  border-radius: 8px; padding: 12px; min-height: 160px;">
        <div style="color: {title_color}; font-size: 12px; font-weight: 800;
                    text-align: center; margin-bottom: 8px; padding-bottom: 6px;
                    border-bottom: 1px solid var(--border);">
          👤 투자자 가이드
        </div>
        {rows_html}
      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# Step 2E: 하단 4섹터 (핵심요약 / 분할익절 / DipSniper / 비교)
# ═══════════════════════════════════════════════════

def render_v2_bottom_sectors(n: dict, rank: int = 0, total: int = 0,
                             compare_name: str = ""):
    """
    [Step 2E] 하단 4섹터.

    1. 핵심 요약 (rank + 점수 + 가격 + RR + KELLY + TIME_STOP)
    2. 분할 익절 플랜 (TP1-3 + 비중%)
    3. DipSniper 체크포인트 (전략 핵심 7개)
    4. 비교 종목 (선택적, compare_name 있으면)
    """
    name = h_escape(n["name"])
    code = h_escape(n["code"])
    struct_s = n["struct_score"]
    timing_s = n["timing_score"]
    ai_s = n["ai_score"]
    axis_gap = n["axis_gap"]
    balance = n["balance_score"]
    entry = n["entry"]
    stop = n["stop"]
    tp1, tp2, tp3 = n["tp1"], n["tp2"], n["tp3"]
    tp1_prob = n["tp1_prob"]
    tp2_prob = n["tp2_prob"]
    tp3_prob = n["tp3_prob"]
    tp1_gain = n["tp1_gain_pct"]
    tp2_gain = n["tp2_gain_pct"]
    tp3_gain = n["tp3_gain_pct"]
    rr = n["rr_now_tp1"]
    qty = int(n["qty"]) if n["qty"] else 0
    amount = n["amount_man"]
    position = n["position_pct"]
    time_stop = int(n["time_stop_days"]) if n["time_stop_days"] else 7
    stop_pct_text = f"{n['stop_loss_pct']:.0f}%" if n["stop_loss_pct"] else "—"

    # 핵심 요약 줄
    summary_lines = []
    if rank and total:
        summary_lines.append(f"{total}개 중 {rank}위")
    summary_lines.append(f"STRUCT {struct_s:.0f} · TIMING {timing_s:.0f} · AI {ai_s:.0f}")
    summary_lines.append(f"AXIS_GAP {axis_gap:.1f} · BALANCE {balance:.1f}")
    summary_lines.append(f"진입가={int(entry):,} / 손절 {int(stop):,} ({stop_pct_text})")
    summary_lines.append(f"RR {rr:.2f} (TP1 {tp1_gain:+.1f}% vs 손절 {stop_pct_text})")
    summary_lines.append(f"KELLY {qty}주 · {amount:.1f}만원 ({position:.0f}%)")
    summary_lines.append(f"{time_stop}일 내 미돌파 시 TIME_STOP 발동")
    summary_html = "".join(
        f'<div class="b-row"><span>{h_escape(line)}</span></div>'
        for line in summary_lines
    )

    # 분할 익절
    tp_blocks = ""
    for icon, lbl, price, gain, prob in [
        ("🎯", "TP1", tp1, tp1_gain, tp1_prob),
        ("🎯", "TP2", tp2, tp2_gain, tp2_prob),
        ("🏆", "TP3", tp3, tp3_gain, tp3_prob),
    ]:
        tp_blocks += f'''
        <div class="tp-block">
            <div class="tp-icon">{icon}</div>
            <div class="tp-info">
                <div class="tp-price">{lbl} <span style="color: var(--green);">{int(price):,}원</span>
                    <span class="gain">{gain:+.1f}%</span>
                </div>
                <div class="tp-weight">익절 비중 {int(prob)}%</div>
            </div>
        </div>
        '''

    # DipSniper 체크포인트 (KELLY 0주 / POSITION 0% / OVERHEAT 케이스 분기)
    is_overheat = n.get("route", "") == "OVERHEAT"
    is_buyable = qty > 0 and n.get("position_pct", 0) > 0 and not is_overheat

    if is_buyable:
        # 매수 가능 케이스 (와이투솔루션 등)
        dipsniper_lines = [
            f"진입가 {int(entry):,} / 손절가 {int(stop):,} 사수 (필수)",
            f"분할매수 1주씩 (KELLY {qty}주까지)",
            f"TP1 {int(tp1):,} 도달 시 자동 익절",
            f"{int(stop):,} 깨면 즉시 손절",
            f"{time_stop}일 내 미돌파 시 TIME_STOP 발동",
            "주요 뉴스 모니터링 필수",
        ]
    else:
        # 매수 부적합 케이스 (한온시스템 OVERHEAT / KELLY 0주 등)
        reasons = []
        if is_overheat:
            reasons.append("OVERHEAT 상태")
        if qty <= 0:
            reasons.append("KELLY 0주")
        if n.get("position_pct", 0) <= 0:
            reasons.append("POSITION 0%")
        reason_txt = " · ".join(reasons) if reasons else "신규매수 부적합"
        rr_now = n.get("rr_now_tp1", 0)

        dipsniper_lines = [
            f"⚠️ 신규매수 제외: {reason_txt}",
            f"RR {rr_now:.2f} (목표 1.0 이상 필요)",
            f"현재 추세 진정 대기 권장",
            f"기존 보유자: TP1 {int(tp1):,} 도달 시 분할 익절",
            f"손절 {int(stop):,} 이탈 시 전량 정리",
            "재진입은 RR 회복 + ROUTE 정상화 후",
        ]

    dipsniper_html = "".join(
        f'<div class="b-row cyan"><span>{h_escape(line)}</span></div>'
        for line in dipsniper_lines
    )

    # 비교 섹션 (compare_name 있을 때만)
    compare_html = ""
    if compare_name:
        compare_html = f'''
        <div class="bottom-panel compare" style="background: #1A1D26; border: 1px solid #2A2D38; border-radius: 8px; padding: 12px; min-height: 200px; min-width: 0;">
            <div class="b-title">{h_escape(compare_name)} 비교</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                <div style="text-align: center;">
                    <div style="color: var(--purple); font-size: 11px; font-weight: 700;
                                padding-bottom: 4px; border-bottom: 1px dashed var(--border);">
                        {name}
                    </div>
                    <div style="color: var(--text-gray); font-size: 10px; padding: 4px 0;">
                        ELITE {n["elite_score"]:.1f}
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="color: var(--orange); font-size: 11px; font-weight: 700;
                                padding-bottom: 4px; border-bottom: 1px dashed var(--border);">
                        {h_escape(compare_name)}
                    </div>
                    <div style="color: var(--text-gray); font-size: 10px; padding: 4px 0;">
                        (비교 데이터)
                    </div>
                </div>
            </div>
        </div>
        '''
    else:
        # compare_name 없으면 KELLY 요약으로 대체
        compare_html = f'''
        <div class="bottom-panel" style="background: #1A1D26; border: 1px solid #2A2D38; border-radius: 8px; padding: 12px; min-height: 200px; min-width: 0;">
            <div class="b-title">KELLY 권고</div>
            <div class="b-row"><span>플랜 B: {n["kelly_planned_b"]:.3f}</span></div>
            <div class="b-row"><span>실측 B: {n["kelly_empirical_b"]:.1f}</span></div>
            <div class="b-row"><span>최종 B: {n["kelly_final_b"]:.1f}</span></div>
            <div class="b-row"><span>수량: {qty}주</span></div>
            <div class="b-row"><span>금액: {amount:.1f}만원</span></div>
            <div class="b-row"><span>포지션: {position:.0f}%</span></div>
        </div>
        '''

    ui.html(f'''
    <style>
      .sd-v2 .bottom-grid-wrapper {{
        width: 100%;
        margin-top: 12px;
      }}
      .sd-v2 .bottom-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        width: 100%;
      }}
      .sd-v2 .bottom-panel {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
        min-height: 200px;
        overflow: hidden;
        min-width: 0;  /* 1fr 컬럼이 콘텐츠 폭에 갇히지 않도록 */
      }}
      .sd-v2 .bottom-panel .b-title {{
        color: var(--orange);
        font-size: 12px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
      }}
      .sd-v2 .bottom-panel.dipsniper .b-title {{ color: var(--cyan); }}
      .sd-v2 .bottom-panel.compare .b-title {{ color: var(--purple); }}
      .sd-v2 .bottom-panel .b-row {{
        display: flex;
        align-items: flex-start;
        gap: 6px;
        font-size: 10px;
        padding: 3px 0;
        color: var(--text-gray);
        line-height: 1.5;
      }}
      .sd-v2 .bottom-panel .b-row::before {{
        content: "✓";
        color: var(--green);
        font-size: 9px;
        flex-shrink: 0;
        margin-top: 2px;
      }}
      .sd-v2 .bottom-panel .b-row.cyan::before {{ color: var(--cyan); }}
      .sd-v2 .split-tp {{ display: flex; flex-direction: column; gap: 8px; }}
      .sd-v2 .tp-block {{
        display: grid;
        grid-template-columns: 28px 1fr;
        gap: 8px;
        align-items: center;
      }}
      .sd-v2 .tp-icon {{ font-size: 18px; text-align: center; }}
      .sd-v2 .tp-info {{ display: flex; flex-direction: column; }}
      .sd-v2 .tp-price {{ font-size: 11px; font-weight: 700; color: var(--text-white); }}
      .sd-v2 .tp-price .gain {{ color: var(--orange); margin-left: 4px; }}
      .sd-v2 .tp-weight {{
        color: var(--orange);
        font-size: 10px;
        font-weight: 600;
        margin-top: 2px;
      }}
      .sd-v2 .tp-weight::before {{ content: "→ "; }}
    </style>

    <div class="sd-v2 bottom-grid-wrapper" style="display: block; width: 100%; margin-top: 8px;">
      <div class="bottom-grid" style="display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; width: 100%;">

        <!-- 1. 핵심 요약 -->
        <div class="bottom-panel" style="background: #1A1D26; border: 1px solid #2A2D38; border-radius: 8px; padding: 12px; min-height: 200px; min-width: 0;">
          <div class="b-title">핵심 요약</div>
          {summary_html}
        </div>

        <!-- 2. 분할 익절 플랜 -->
        <div class="bottom-panel" style="background: #1A1D26; border: 1px solid #2A2D38; border-radius: 8px; padding: 12px; min-height: 200px; min-width: 0;">
          <div class="b-title">분할 익절 플랜</div>
          <div class="split-tp">
            {tp_blocks}
          </div>
        </div>

        <!-- 3. DipSniper 체크포인트 -->
        <div class="bottom-panel dipsniper" style="background: #1A1D26; border: 1px solid #2A2D38; border-radius: 8px; padding: 12px; min-height: 200px; min-width: 0;">
          <div class="b-title">DipSniper 체크포인트</div>
          {dipsniper_html}
        </div>

        <!-- 4. 비교/KELLY -->
        {compare_html}

      </div>
    </div>
    ''')


# ═══════════════════════════════════════════════════
# 메인 진입점 (Step 2A 부분)
# ═══════════════════════════════════════════════════

def _load_ohlcv_for_v2(code: str, days: int = 120):
    """
    종목코드 → 최근 N일 OHLCV DataFrame.

    [최종 전략]
      1) tab_stocks._get_chart_data 위임 (Railway에서 검증된 v1 로더 — SSOT)
      2) 실패 시: 자체 로컬 fallback (멀티 경로/포맷/컬럼 자동 탐지)

    이렇게 가는 이유:
      - v1 _get_chart_data가 같은 parquet 캐시를 잘 읽고 있음 (검증됨)
      - v2가 따로 로더를 만들면 환경 차이로 빗나갈 위험 (실제로 0개 보고)
      - SSOT 원칙: 같은 데이터, 같은 로더
    """
    # ── 1) v1 _get_chart_data 위임 (SSOT) ──
    _v1_error = None
    try:
        from components.tab_stocks import _get_chart_data
        result = _get_chart_data(str(code).zfill(6), days=days)
        if result is not None and len(result) > 0:
            return result
        _v1_error = f"v1 _get_chart_data returned {type(result).__name__} len={0 if result is None else len(result)}"
    except Exception as e:
        _v1_error = f"v1 import/call exception: {type(e).__name__}: {e}"

    # 디버그 정보 모듈 변수에 저장 (debug_info 함수에서 사용)
    global _last_v1_error
    _last_v1_error = _v1_error

    # ── 2) 자체 fallback (v1 호출 실패 시) ──
    import os, glob
    try:
        import pandas as pd
    except ImportError:
        return None

    code_norm = str(code).zfill(6)
    code_int_str = str(int(code_norm)) if code_norm.isdigit() else code_norm
    cols_needed = ["시가", "고가", "저가", "종가", "거래량"]
    code_col_candidates = ["종목코드", "code", "Code", "ticker", "Ticker", "symbol", "Symbol"]
    date_col_candidates = ["Date", "date", "날짜", "일자", "DATE"]

    here = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        os.path.join(here, "..", "data"),
        os.path.join(os.getcwd(), "data"),
        "data",
        os.path.join(here, "data"),
    ]

    files = []
    seen = set()
    for d in candidate_dirs:
        if not os.path.isdir(d):
            continue
        d_real = os.path.realpath(d)
        if d_real in seen:
            continue
        seen.add(d_real)
        files.extend(glob.glob(os.path.join(d, "ohlcv_cache_*.parquet")))
        files.extend(glob.glob(os.path.join(d, "ohlcv_cache_*.pkl")))
        files.extend(glob.glob(os.path.join(d, "ohlcv_cache_*.pickle")))
    files = sorted(set(files), reverse=True)

    if not files:
        return None

    def _try_extract(df, fp):
        if df is None or df.empty:
            return None
        idx_is_date = (df.index.name in date_col_candidates) or (
            hasattr(df.index, "dtype") and "datetime" in str(df.index.dtype)
        )
        if idx_is_date:
            df = df.reset_index()
        code_col = next((c for c in code_col_candidates if c in df.columns), None)
        if code_col is None:
            return None
        df_local = df.copy()
        df_local[code_col] = df_local[code_col].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
        sub = df_local[df_local[code_col] == code_norm].copy()
        if sub.empty:
            sub = df_local[df_local[code_col].astype(str).str.lstrip("0") == code_int_str.lstrip("0")].copy()
            if sub.empty:
                return None
        date_col = next((c for c in date_col_candidates if c in sub.columns), None)
        if date_col:
            sub = sub.sort_values(date_col).set_index(date_col)
        if all(c in sub.columns for c in cols_needed):
            return sub[cols_needed].tail(days).copy()
        eng_map = {"Open": "시가", "High": "고가", "Low": "저가", "Close": "종가", "Volume": "거래량"}
        if all(c in sub.columns for c in eng_map.keys()):
            sub_kr = sub[list(eng_map.keys())].rename(columns=eng_map)
            return sub_kr.tail(days).copy()
        return None

    for fp in files[:5]:
        try:
            if fp.endswith(".parquet"):
                df = pd.read_parquet(fp)
                result = _try_extract(df, fp)
                if result is not None and not result.empty:
                    return result
            else:
                obj = pd.read_pickle(fp)
                if isinstance(obj, dict):
                    sub = obj.get(code_norm) or obj.get(code_int_str)
                    if sub is not None and not sub.empty:
                        result = _try_extract(sub, fp)
                        if result is not None and not result.empty:
                            return result
                        if all(c in sub.columns for c in cols_needed):
                            return sub[cols_needed].tail(days).copy()
                        eng_map = {"Open": "시가", "High": "고가", "Low": "저가",
                                   "Close": "종가", "Volume": "거래량"}
                        if all(c in sub.columns for c in eng_map.keys()):
                            sub_kr = sub[list(eng_map.keys())].rename(columns=eng_map)
                            return sub_kr.tail(days).copy()
                elif hasattr(obj, "columns"):
                    result = _try_extract(obj, fp)
                    if result is not None and not result.empty:
                        return result
        except Exception:
            continue

    return None


def _load_ohlcv_debug_info(code: str) -> str:
    """차트 실패 시 진단 정보 (관리자용)."""
    import os, glob
    code_norm = str(code).zfill(6)
    here = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        ("__file__ 기준", os.path.join(here, "..", "data")),
        ("CWD 기준",      os.path.join(os.getcwd(), "data")),
        ("상대경로",       "data"),
        ("__file__ 동일", os.path.join(here, "data")),
    ]
    lines = [f"종목코드: {code_norm}", f"CWD: {os.getcwd()}", f"__file__: {here}"]
    # v1 위임 실패 사유 노출
    if _last_v1_error:
        lines.append(f"v1 위임 결과: {_last_v1_error}")
    else:
        lines.append("v1 위임: 시도 안 됨")
    for label, d in candidate_dirs:
        exists = os.path.isdir(d)
        if exists:
            n_parquet = len(glob.glob(os.path.join(d, "ohlcv_cache_*.parquet")))
            n_pkl     = len(glob.glob(os.path.join(d, "ohlcv_cache_*.pkl")))
            n_pickle  = len(glob.glob(os.path.join(d, "ohlcv_cache_*.pickle")))
            lines.append(f"  {label}: {d} ✓ parquet={n_parquet} pkl={n_pkl} pickle={n_pickle}")
        else:
            lines.append(f"  {label}: {d} ✗ (디렉토리 없음)")
    return "\n".join(lines)


def render_stock_detail_v2_partial(row: Dict[str, Any],
                                    rank: int = 0,
                                    total: int = 0,
                                    timestamp: str = "",
                                    combo_info: Optional[dict] = None,
                                    ohlcv_df=None,
                                    compare_name: str = ""):
    """
    [Step 2E COMPLETE] 종목 상세 v2 — 풀 대시보드 (placeholder 0개).

    구성:
        헤더 (종목명 + LDY_RANK + 5개 status 뱃지)
        점수 영역 (DISPLAY + 3축 + FINAL/ELITE/BALANCE + 사유)
        main-grid:
            좌측: 가격플랜 / 추세MTF / 모멘텀 / 수급 (4패널)
            중앙: 메인 캔들차트 (ECharts + markLine) + 보조 차트 4개 + 게이지
            우측: 레이더 5축 (SVG)
        하단 4섹터: 핵심요약 / 분할익절 / DipSniper체크 / KELLY권고(또는 비교)

    Args:
        row: recommend_latest.csv 한 행 (pandas Series.to_dict() 또는 dict)
        rank: LDY_RANK 순위 (별도 전달 권장, row의 LDY_RANK 사용 가능)
        total: 전체 종목 수
        timestamp: 분석 시각 (row의 기준일 사용 가능)
        combo_info: 콤보 정보 dict {color, name, n, ev, win_rate, rank}.
                    None이면 ELITE_REASON으로 fallback 자동 표시.
        ohlcv_df: 종목 OHLCV DataFrame [시가/고가/저가/종가/거래량].
                  None이면 종목코드 기준으로 parquet 캐시에서 자동 로드.
        compare_name: 하단 4섹터의 비교 종목명. 비어있으면 KELLY 요약 표시.

    Usage:
        import pandas as pd
        from components.stock_detail_v2 import render_stock_detail_v2_partial

        df = pd.read_csv("data/recommend_latest.csv", dtype={"종목코드": str})
        row = df[df["종목코드"] == "011690"].iloc[0].to_dict()

        render_stock_detail_v2_partial(
            row,
            rank=int(row.get("LDY_RANK", 0)),
            total=len(df),
            timestamp=str(row.get("기준일", "")),
        )

    Implementation status:
        ✅ Step 2A: 헤더 + 점수 영역 (CSV 정규화 어댑터)
        ✅ Step 2B: 좌측 패널 #1 — 가격 플랜
        ✅ Step 2C: 좌측 패널 #2-4 — 추세/MTF, 모멘텀, 수급
        ✅ Step 2D: 메인 캔들차트 (ECharts + HMA/VWAP/SUPERTREND + markLine)
        ✅ Step 2E: 보조 차트 4개 + 게이지 + 레이더 + 하단 4섹터

    Last review score: 95/100 (운영 머지 후보권)
    """
    # 정규화
    n = normalize_stock_row(row)

    _inject_v2_styles()

    render_v2_header(n, rank=rank, total=total,
                     timestamp=timestamp, combo_info=combo_info)
    render_v2_scores(n)

    # OHLCV 자동 로드 (외부 전달 안 했을 때)
    if ohlcv_df is None:
        ohlcv_df = _load_ohlcv_for_v2(n["code"], days=120)

    # main-grid: 좌측 4패널 + 중앙 차트 + 우측 레이더
    # inline style 강제 (NiceGUI ui.element가 CSS class만으로 grid 적용 안 되는 케이스 회피)
    # 좌측 260px (가격 플랜 표시 여유) / 중앙 minmax(0,1fr) / 우측 300px (레이더 크게)
    # v2-main-grid 클래스 추가 → 모바일 미디어 쿼리에서 명시적으로 타겟 가능
    with ui.element("div").style(
        "display: grid; grid-template-columns: 260px minmax(0, 1fr) 300px; "
        "gap: 8px; width: 100%; margin-bottom: 12px; box-sizing: border-box;"
    ).classes("sd-v2 v2-main-grid"):

        # 좌측 사이드: 패널 #1-4 (v2-left-panels 클래스)
        with ui.element("div").style(
            "display: flex; flex-direction: column; gap: 8px; min-width: 0;"
        ).classes("v2-left-panels"):
            render_v2_price_plan(n)
            render_v2_trend_mtf(n)
            render_v2_momentum(n)
            render_v2_supply(n)

        # 중앙: 메인 캔들차트 + 보조 차트 4개 + 거래강도 게이지 + 3분할 패널
        with ui.element("div").style("min-width: 0;").classes("v2-center-col"):
            render_v2_chart(n, ohlcv_df=ohlcv_df)
            render_v2_sub_charts(n, ohlcv_df=ohlcv_df)
            # [Step 2F] 수익률현황 / 핵심레벨 / 리스크뉴스 (3분할)
            render_v2_returns_levels_risk(n)
            # [Step 2F] 시나리오 A/B/C (3카드)
            render_v2_scenarios(n)

        # 우측 사이드: 레이더 차트 (5축) + AXIS_GAP 큰 카드 + 투자자 가이드
        with ui.element("div").style(
            "display: flex; flex-direction: column; gap: 8px; min-width: 0;"
        ).classes("v2-right-col"):
            render_v2_radar(n)
            render_v2_right_axisgap(n)
            render_v2_right_guide(n)

    # 하단 4섹터 (핵심 요약 / 분할 익절 / DipSniper / 비교 또는 KELLY)
    render_v2_bottom_sectors(n, rank=rank, total=total, compare_name=compare_name)

    # [Step 2F] 최종 판정 띠 (페이지 최하단)
    render_v2_final_verdict(n, rank=rank, total=total)

# ═══════════════════════════════════════════════════════════════════
# [v22.3.21] No-Buy / 관망 카드 렌더 (NiceGUI) — FOMO-safety
#   buy_now_badge.build_no_buy_card_model() 결과만 소비한다.
#   official_buy=False면 목표가/매수가를 hero로 절대 띄우지 않는다(회색 각주만).
#   관리자 raw 엔진값은 ui.expansion으로 분리(구독자 화면 비노출).
# ═══════════════════════════════════════════════════════════════════

_NB_GREEN = "#10b981"
_NB_AMBER = "#f59e0b"
_NB_DANGER_BG = "#fef2f2"
_NB_DANGER_TX = "#b91c1c"
_NB_DANGER_IC = "#ef4444"
_NB_TX = "#111827"
_NB_TX2 = "#6b7280"
_NB_TX3 = "#9ca3af"
_NB_SURF = "#f9fafb"
_NB_BORDER = "#e5e7eb"


def _nb_bar_html(a: Dict[str, Any]) -> str:
    color = _NB_GREEN if a["pass"] else _NB_AMBER
    icon = "✓" if a["pass"] else "✗"
    name = h_escape(str(a["name"]))
    val = float(a["value"]); thr = float(a["threshold"])
    fillw = max(0.0, min(100.0, val))
    markl = max(0.0, min(100.0, thr))
    return (
        '<div style="margin-bottom:10px;">'
        '<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;">'
        f'<span style="color:{_NB_TX2};">{name}</span>'
        f'<span style="color:{color};font-weight:600;">{icon} {val:.0f} '
        f'<span style="color:{_NB_TX3};font-weight:400;">· 통과선 {thr:.0f}</span></span>'
        '</div>'
        f'<div style="position:relative;height:7px;background:#f1f5f9;border-radius:4px;">'
        f'<div style="width:{fillw:.0f}%;height:100%;background:{color};border-radius:4px;"></div>'
        f'<div style="position:absolute;left:{markl:.0f}%;top:-2px;width:2px;height:11px;background:{_NB_TX3};"></div>'
        '</div></div>'
    )


def _nb_card_html(m: Dict[str, Any], pass_count=None, watch_count=None) -> str:
    """카드 모델 → HTML 문자열. official_buy=False면 가격은 각주로만 강등된다."""
    market_blocked = any(b["kind"] == "market" for b in m["blockers"])
    banner_sub = "시장 전체 위험 높음 · CRITICAL" if market_blocked else "공식 신규매수 기준 미충족"

    # 상단 배너 (rest 톤)
    html = (
        f'<div style="max-width:420px;background:#fff;border:0.5px solid {_NB_BORDER};'
        'border-radius:12px;padding:14px 16px;font-family:inherit;">'
        f'<div style="display:flex;align-items:center;gap:10px;background:{_NB_DANGER_BG};'
        'border-radius:8px;padding:11px 12px;">'
        f'<span style="font-size:20px;color:{_NB_DANGER_IC};">🛡️</span>'
        '<div>'
        f'<div style="font-weight:600;font-size:15px;color:{_NB_DANGER_TX};">{h_escape(m["headline"])}</div>'
        f'<div style="font-size:12px;color:{_NB_DANGER_TX};">{h_escape(banner_sub)}</div>'
        '</div></div>'
        f'<p style="font-size:13px;color:{_NB_TX2};line-height:1.7;margin:12px 2px;">{h_escape(m["subtext"])}</p>'
    )

    # (선택) 오늘 통과/관찰 카운트
    if pass_count is not None or watch_count is not None:
        pc = 0 if pass_count is None else int(pass_count)
        wc = 0 if watch_count is None else int(watch_count)
        html += (
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:4px;">'
            f'<div style="background:{_NB_SURF};border-radius:8px;padding:10px 12px;">'
            f'<div style="font-size:12px;color:{_NB_TX2};">오늘 매수 통과</div>'
            f'<div style="font-size:22px;font-weight:600;color:{_NB_TX};">{pc}<span style="font-size:13px;font-weight:400;color:{_NB_TX2};"> 종목</span></div></div>'
            f'<div style="background:{_NB_SURF};border-radius:8px;padding:10px 12px;">'
            f'<div style="font-size:12px;color:{_NB_TX2};">관찰 후보</div>'
            f'<div style="font-size:22px;font-weight:600;color:{_NB_TX};">{wc}<span style="font-size:13px;font-weight:400;color:{_NB_TX2};"> 종목</span></div></div>'
            '</div>'
        )

    # 가장 가까운 종목 분석
    html += f'<div style="border-top:0.5px solid {_NB_BORDER};margin-top:14px;padding-top:14px;">'
    html += '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">'
    html += (
        '<div style="display:flex;align-items:center;gap:7px;">'
        f'<span style="font-size:16px;color:{_NB_TX2};">👀</span>'
        f'<span style="font-weight:600;font-size:15px;color:{_NB_TX};">{h_escape(m["name"])}</span>'
        f'<span style="font-size:11px;color:{_NB_TX2};">가장 가까웠던 종목</span></div>'
    )
    if m["watch_label"]:
        html += (f'<span style="background:#f1f5f9;color:{_NB_TX2};font-size:11px;font-weight:600;'
                 f'padding:3px 9px;border-radius:8px;">{h_escape(m["watch_label"])}</span>')
    html += '</div>'
    if m["closest_note"]:
        html += f'<p style="font-size:13px;color:{_NB_TX2};margin:6px 2px 14px;">{h_escape(m["closest_note"])}</p>'

    for a in m["axes"]:
        html += _nb_bar_html(a)

    # 차단 사유 (1순위/2순위)
    html += f'<div style="background:{_NB_SURF};border-radius:8px;padding:10px 12px;font-size:12px;line-height:1.7;">'
    html += f'<div style="color:{_NB_TX};font-weight:600;margin-bottom:3px;">통과 못한 이유</div>'
    for b in m["blockers"]:
        rc = _NB_DANGER_IC if b["kind"] == "market" else _NB_AMBER
        html += (f'<div style="color:{_NB_TX2};"><span style="color:{rc};font-weight:600;">'
                 f'{b["rank"]}순위 ·</span> {h_escape(b["text"])}</div>')
    html += '</div>'

    # 가격 — official_buy=False면 각주로만 (FOMO-safety)
    if m["price_treatment"] == "footnote" and m["target_footnote"]:
        html += (f'<p style="font-size:11px;color:{_NB_TX3};line-height:1.6;margin:10px 2px 0;">'
                 f'ⓘ {h_escape(m["target_footnote"])}</p>')

    html += '</div>'  # close 분석 section
    html += (f'<div style="border-top:0.5px solid {_NB_BORDER};margin-top:14px;padding-top:10px;'
             f'font-size:11px;color:{_NB_TX3};">데이터 기준 오늘 · {h_escape(m["disclaimer"])}</div>')
    html += '</div>'  # close card
    return html


def render_no_buy_card(row: Dict[str, Any], is_admin: bool = False,
                       pass_count=None, watch_count=None) -> Dict[str, Any]:
    """[v22.3.21] 관망/No-Buy 카드 렌더(NiceGUI). 반환: 사용된 카드 모델(테스트/디버깅용).

    official_buy=False면 build_no_buy_card_model 규칙에 따라 목표가/매수가를 hero로
    띄우지 않고 회색 각주로만 표시한다. 관리자 raw 엔진값은 expander로 분리한다.
    """
    from components.buy_now_badge import build_no_buy_card_model
    m = build_no_buy_card_model(row, is_admin=is_admin)
    ui.html(_nb_card_html(m, pass_count=pass_count, watch_count=watch_count))

    if is_admin and m.get("admin_raw"):
        with ui.expansion("관리자 · 엔진 원시값", icon="settings").classes("w-full").style("max-width:420px"):
            rows_html = "".join(
                f'<tr><td style="color:{_NB_TX2};padding:3px 8px 3px 0;font-size:12px;">{h_escape(str(k))}</td>'
                f'<td style="text-align:right;padding:3px 0;font-size:12px;color:{_NB_TX};">{h_escape(str(v))}</td></tr>'
                for k, v in m["admin_raw"].items()
            )
            ui.html(f'<table style="width:100%;border-collapse:collapse;">{rows_html}</table>')
    return m
