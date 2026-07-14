# -*- coding: utf-8 -*-
"""
reco_evidence.py — [v28] "왜 이 종목인가" 근거 문장 생성
═══════════════════════════════════════════════════
문제의식:
  화면에 점수만 있으니 (1) 왜 오를 확률이 높은지 알 수 없어 망설이게 되고,
  (2) "점수 높음 = 좋은 종목" 착시가 생긴다. 실측 IC 검증 결과 점수는
  하락장에서 역방향(2026-07 ELITE IC -0.20)이었다.

원칙:
  1. 모든 근거는 **실측 수치**를 동반한다 ("좋음"이 아니라 "이 구간 실측 승률 48%").
  2. 실측 승률(EST_WIN_RATE)이 1순위, 점수는 보조.
  3. 리스크(반대 근거)도 반드시 같이 보여준다.
  4. ML(AI) 축이 미사용 상태면 그 사실을 숨기지 않는다.

출력 컬럼:
  RECO_EVIDENCE : ' · '로 연결된 근거 문장 (프론트 카드/상세용)
  RECO_RISK     : 리스크 요인 문장
  HONEST_PROB_PCT : 실측 승률 % (표본 충분 시), 아니면 NaN
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# v27 실측 근거 (docs/HIGH_PROB_V27.md)
_POC_BANDS = [
    # (하한, 상한, 실측 승률 문구)
    (-999.0, 0.0, "실측 표본 적음"),
    (0.0, 10.0, "실측 승률 40%"),
    (10.0, 20.0, "실측 승률 48%"),
    (20.0, 30.0, "실측 승률 20% — 추격 위험"),
    (30.0, 999.0, "실측 승률 12~23% — 추격 위험"),
]


def _num(row, key, default=None):
    try:
        v = float(row.get(key))
        if np.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _txt(row, key) -> str:
    v = row.get(key)
    return "" if v is None else str(v).strip()


def _poc_band_note(poc: float) -> str:
    for lo, hi, note in _POC_BANDS:
        if lo <= poc < hi:
            return note
    return ""


def build_evidence_row(row: dict) -> tuple:
    """한 종목의 (근거 문장, 리스크 문장, 실측승률%) 생성."""
    reasons: list = []
    risks: list = []
    honest_prob = np.nan

    # ── 1순위: 실측 승률 (자기 추천의 과거 성적) ──
    est_wr = _num(row, "EST_WIN_RATE")
    est_n = _num(row, "EST_WIN_RATE_N", 0) or 0
    est_mode = _txt(row, "EST_WIN_RATE_MODE")
    if est_wr is not None and est_wr > 0 and est_n >= 30 and est_mode == "MATURE":
        honest_prob = round(est_wr * 100, 1)
        reasons.append(
            f"이 점수 구간의 과거 실측 승률 {honest_prob:.0f}% (표본 {est_n:.0f}건)"
        )
    elif est_wr is not None and est_wr > 0:
        reasons.append(
            f"추정 승률 {est_wr*100:.0f}% (표본 {est_n:.0f}건 — 통계 미성숙, 참고만)"
        )

    # ── 시장 레짐 ──
    regime = _txt(row, "MARKET_REGIME")
    breadth = _num(row, "MARKET_BREADTH")
    if regime == "UP":
        reasons.append(
            f"상승 레짐 (시장폭 {breadth:.0f}%)" if breadth is not None else "상승 레짐"
        )
    elif regime == "NEUTRAL":
        risks.append("중립 레짐 — 사이즈 50% 권장")
    elif regime == "DOWN":
        risks.append(
            f"하락 레짐 (시장폭 {breadth:.0f}%) — 신규 진입 차단 구간"
            if breadth is not None else "하락 레짐 — 신규 진입 차단 구간"
        )

    # ── 가치영역(POC) 근접 ──
    poc = _num(row, "POC_GAP")
    if poc is not None:
        note = _poc_band_note(poc)
        if poc <= 20.0:
            reasons.append(f"거래량 밀집가 대비 {poc:+.1f}% — 가치영역 근접 ({note})")
        else:
            risks.append(f"거래량 밀집가 대비 {poc:+.1f}% 확장 ({note})")

    # ── 손익비 ──
    rr = _num(row, "RR_NOW_TP1")
    close = _num(row, "종가")
    stop = _num(row, "손절가")
    tp1 = _num(row, "추천매도가1")
    if rr is not None and rr > 0 and close and stop and tp1 and close > 0:
        stop_pct = (stop / close - 1) * 100
        tp_pct = (tp1 / close - 1) * 100
        if rr >= 1.0:
            reasons.append(
                f"손익비 {rr:.1f} (손절 {stop_pct:+.1f}% / 1차목표 {tp_pct:+.1f}%)"
            )
        else:
            risks.append(f"손익비 {rr:.2f} <1 — 이길 수 없는 구조")

    # ── 구조 근거 (구체 신호만, 점수 아님) ──
    squeeze = _num(row, "TTM_SQUEEZE_CNT", 0) or 0
    if squeeze >= 3:
        reasons.append(f"변동성 응축 {squeeze:.0f}일째 (스퀴즈 — 방향 분출 대기)")
    low_trend = _num(row, "Low_Trend_PCT")
    if low_trend is not None and low_trend > 5:
        reasons.append(f"저점 상승 추세 {low_trend:.0f}% (바닥이 높아지는 중)")

    # ── 리스크 요인 ──
    rsi = _num(row, "RSI14")
    if rsi is not None and rsi >= 70:
        risks.append(f"RSI {rsi:.0f} 과열 구간")
    gap = _num(row, "ENTRY_GAP_PCT")
    if gap is not None and gap > 3:
        risks.append(f"추천가 대비 {gap:.1f}% 이탈 — 추격 주의")
    turnover = _num(row, "거래대금(억원)")
    if turnover is not None and turnover < 50:
        risks.append(f"거래대금 {turnover:.0f}억 — 유동성 낮음")
    fresh = str(row.get("DATA_FRESHNESS_OK", "")).strip().lower()
    if fresh in ("0", "0.0", "false", "f", "no", "n"):
        risks.append("가격 데이터 stale — 표시가와 실제가 다를 수 있음")

    # ── AI(알파모델) 정직 표기 — 검증 통과 시 예측 근거, 아니면 미사용 명시 ──
    alpha_validated = str(row.get("ALPHA_VALIDATED", "")).strip().lower() in ("1", "1.0", "true")
    alpha_score = _num(row, "ALPHA_SCORE")
    alpha_prob = _num(row, "ALPHA_WIN_PROB")
    if alpha_validated and alpha_score is not None:
        _prob_txt = (
            f" — 이 구간 실측 승률 {alpha_prob*100:.0f}%" if alpha_prob is not None else ""
        )
        if alpha_score >= 70:
            reasons.append(
                f"AI 알파모델 예측 상위 {100-alpha_score:.0f}% (OOS 검증 통과 모델{_prob_txt})"
            )
        elif alpha_score < 30:
            risks.append(
                f"AI 알파모델 예측 하위 {alpha_score:.0f}%{_prob_txt} — 픽 제외 구간"
            )
    else:
        w_ai = _num(row, "W_AI")
        ai_score = _num(row, "AI_SCORE", 0) or 0
        ml_trusted = str(row.get("ML_TRUSTED", "")).strip().lower() in ("1", "true")
        if (w_ai is not None and w_ai <= 0) or (not ml_trusted and ai_score == 0):
            reasons.append("AI 예측 미사용 (검증 미통과로 가중치 0 — 룰 기반 판단만 반영)")

    return (" · ".join(reasons), " · ".join(risks), honest_prob)


def add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """recommend DataFrame 전체에 근거/리스크/실측승률 컬럼 추가."""
    if df is None or df.empty:
        if df is not None:
            for c in ("RECO_EVIDENCE", "RECO_RISK", "HONEST_PROB_PCT"):
                df[c] = pd.Series(dtype="object")
        return df
    out = df.copy()
    evidence, risk, prob = [], [], []
    for _, r in out.iterrows():
        e, k, p = build_evidence_row(r.to_dict())
        evidence.append(e)
        risk.append(k)
        prob.append(p)
    out["RECO_EVIDENCE"] = evidence
    out["RECO_RISK"] = risk
    out["HONEST_PROB_PCT"] = prob
    return out
