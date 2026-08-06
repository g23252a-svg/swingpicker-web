# -*- coding: utf-8 -*-
"""Single-decision home screen for SwingPicker.

The first screen answers one question: what, if anything, is safe enough to
buy today?  Research lanes and raw ranks remain available elsewhere, but are
never presented as equivalent to an official production decision here.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

import pandas as pd
from nicegui import ui

logger = logging.getLogger("decision_center")

from services.recommendation_quality import awaiting_execution_mask, production_buy_mask


MARKET_BREADTH_FLOOR = 35.0
DANGEROUS_MARKET_RISK = {"CRITICAL", "WARNING"}


def _number(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _humanize_reason(reason: Any) -> str:
    """Translate engine-oriented rejection text into a user-facing reason."""
    raw = str(reason or "").strip()
    if not raw:
        return "최종 매수 기준을 통과하지 못했습니다"

    parts = [part.strip() for part in re.split(r"\s*[·•]\s*", raw) if part.strip()]
    translated: list[str] = []
    for part in parts:
        if part == "TOP_PICK 아님":
            text = "최종 선별 단계 미통과"
        elif part == "즉시매수 기준 미달":
            text = "즉시 매수 조건 미충족"
        elif part.startswith("진입등급 "):
            text = f"진입 등급이 낮음 ({part.removeprefix('진입등급 ')})"
        elif part.startswith("즉시매수점수 "):
            text = f"즉시 매수 점수 부족 ({part.removeprefix('즉시매수점수 ')})"
        elif part == "공격형 검증 실패":
            text = "공격형 후보의 과거 검증 미통과"
        elif part.startswith("진입위험 "):
            text = f"진입 위험이 높음 ({part.removeprefix('진입위험 ')})"
        elif part == "경로 WAIT":
            text = "진입 타이밍 대기"
        elif part.startswith("경로 "):
            text = f"매수 경로 조건 미충족 ({part.removeprefix('경로 ')})"
        elif part.startswith("시장위험 "):
            text = f"시장 위험 단계가 높음 ({part.removeprefix('시장위험 ')})"
        elif part.startswith("MFI "):
            text = f"수급 강도가 적정 범위 밖 (MFI {part.removeprefix('MFI ')})"
        elif part == "상단여력 부족":
            text = "목표가까지 상승 여력 부족"
        elif part == "VWAP 적정구간 아님":
            text = "현재가가 적정 진입 구간 밖"
        elif part.startswith("손익비 "):
            text = f"예상 손익비 부족 ({part.removeprefix('손익비 ')})"
        elif part.startswith("기본점수 "):
            text = f"기본 품질 점수 부족 ({part.removeprefix('기본점수 ')})"
        elif part.startswith("POC 확장 "):
            text = f"단기 급등 후 추격 위험 ({part.removeprefix('POC 확장 ')})"
        elif part.startswith("시장폭 "):
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", part)
            pct = f" {match.group(1)}%" if match else ""
            text = f"상승 종목 비율{pct}로 시장 내부가 약함"
        elif part.startswith("하락 레짐"):
            text = "하락장이라 신규 진입 차단"
        elif part == "품질점수 미달":
            text = "최종 품질 점수 부족"
        elif part == "손실방어 차단":
            text = "폭락 방어(risk_off) 차단 — 시장 회복 시 자동 해제"
        elif part == "진입 조건 미달":
            text = "AI 알파 진입 기준 미달"
        else:
            # v32 알파 사유("알파 92점 (진입선 90점 미달)")는 이미 사람말 — 그대로 통과
            text = part
        if text not in translated:
            translated.append(text)
    return " · ".join(translated) if translated else "최종 매수 기준을 통과하지 못했습니다"


def _reason_items(reason: Any, limit: int = 3) -> list[str]:
    return [
        item.strip()
        for item in _humanize_reason(reason).split(" · ")
        if item.strip()
    ][:limit]


def _truthy_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    s = df[col].astype(str).str.strip().str.lower()
    return int(s.isin(["1", "1.0", "true", "t", "yes"]).sum())


def _risk_off_state(df: pd.DataFrame) -> dict[str, Any]:
    """[v34] 폭락 방어(risk_off) 잠금 상태 + 해제 진행률.

    7/16 실사용 피드백: 시장폭 88%로 '시장 환경 통과'인데 매수 0개 —
    실제 차단자(risk_off)가 화면에 없어 '왜 안 사는지' 알 수 없었다.
    df의 NEW_ENTRY_BLOCKED + STOP_OVERRIDE_REASON에서 잠금을 감지하고,
    kospi_daily로 해제선(20일선 -3%)까지의 거리를 계산해 노출한다.
    """
    out: dict[str, Any] = {"active": False, "line": "", "progress": None}
    try:
        if df is None or df.empty:
            return out
        n_blocked = _truthy_count(df, "NEW_ENTRY_BLOCKED")
        reason = df.get("STOP_OVERRIDE_REASON")
        has_ro = bool(
            reason is not None
            and reason.astype(str).str.contains("risk_off", case=False).any()
        )
        if not (has_ro and n_blocked >= max(1, int(len(df) * 0.5))):
            return out
        out["active"] = True
        out["line"] = "코스피가 하락 중인 20일선을 회복하면 자동 해제됩니다"
        from market_regime import load_kospi_daily

        k = load_kospi_daily("data")
        if k is not None and not k.empty:
            last = k.iloc[-1]
            close, ma20 = float(last["close"]), last.get("ma20")
            if pd.notna(ma20) and float(ma20) > 0:
                ma20 = float(ma20)
                unlock = ma20 * 0.97  # 해제선: 20일선 -3% 이내 (risk_off 판정식 역산)
                dev_pct = (close / ma20 - 1.0) * 100.0
                # 진행률: 폭락 최심부(-15%) → 해제선(-3%) 구간 매핑
                prog = (dev_pct - (-15.0)) / ((-3.0) - (-15.0)) * 100.0
                out["progress"] = max(0.0, min(100.0, prog))
                out["line"] = (
                    f"코스피 {close:,.0f} · 해제선 {unlock:,.0f}"
                    f" (20일선 -3% 이내) · 현재 20일선 대비 {dev_pct:+.1f}%"
                )
    except Exception as e:
        # 진행률 계산 실패는 표시 생략으로 강등 (잠금 감지 자체는 df만으로 완료)
        logger.warning(f"risk_off 해제 진행률 계산 실패 (표시 생략): {e}")
    return out


def _alpha_gate_on(df: pd.DataFrame) -> bool:
    """[v34] 이 배치가 알파 전면 게이트(v32)로 생성됐는지."""
    if df is None or df.empty or "ALPHA_GATE_ACTIVE" not in df.columns:
        return False
    return _truthy_count(df, "ALPHA_GATE_ACTIVE") > 0


def _dominant_text(df: pd.DataFrame, key: str, default: str) -> str:
    if key not in df.columns:
        return default
    values = df[key].dropna().astype(str).str.strip()
    values = values[~values.isin(["", "nan", "None"])]
    if values.empty:
        return default
    mode = values.mode()
    return str(mode.iloc[0] if not mode.empty else values.iloc[0])


def _stock_payload(row: pd.Series) -> dict[str, Any]:
    raw_reason = str(row.get("QUALITY_GUARD_REASON", ""))
    # [v34] AI 알파 — 검증된 예측 점수(당일 백분위)와 그 구간의 실측 승률
    alpha = _number(row, "ALPHA_SCORE", default=float("nan"))
    alpha_wp = _number(row, "ALPHA_WIN_PROB", default=float("nan"))
    return {
        "code": str(row.get("종목코드", "")).split(".")[0].zfill(6),
        "name": str(row.get("종목명", "-")),
        "score": _number(row, "QUALITY_GUARD_SCORE"),
        "display_score": _number(row, "DISPLAY_SCORE"),
        "alpha": alpha,
        "alpha_win_prob": alpha_wp,
        "entry": _number(row, "추천매수가"),
        "close": _number(row, "종가"),
        "stop": _number(row, "손절가"),
        "target": _number(row, "추천매도가1"),
        "target2": _number(row, "추천매도가2"),
        "rr": _number(row, "RR_NOW_TP1"),
        "weight": _number(row, "RECOMMENDED_WEIGHT_PCT"),
        "reason": _humanize_reason(raw_reason),
        "reason_items": _reason_items(raw_reason),
        "raw_reason": raw_reason,
        "route": str(row.get("ROUTE", "")),
        "decision": str(row.get("ACTION_DECISION", "CASH")),
    }


def build_decision_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {
            "status": "NO_DATA",
            "title": "추천 데이터를 확인할 수 없습니다",
            "subtitle": "새로고침 또는 수집 상태를 확인하세요.",
            "action_label": "데이터 상태를 확인하세요",
            "action_detail": "추천 데이터가 준비되기 전에는 주문하지 마세요.",
            "next_check": "데이터 수집 완료 후 다시 확인",
            "buys": [],
            "watch": [],
            "awaiting": [],
            "blockers": ["추천 데이터 없음"],
            "gates": [],
            "production_count": 0,
            "watch_count": 0,
            "ml_status": "NO_DATA",
            "market_breadth": None,
            "market_ready": False,
            "macro_risk": "UNKNOWN",
            "market_regime": "UNKNOWN",
        }

    work = df.copy()
    production = production_buy_mask(work)
    quality = pd.to_numeric(
        work.get("QUALITY_GUARD_SCORE", pd.Series(0, index=work.index)),
        errors="coerce",
    ).fillna(0)
    action = work.get(
        "ACTION_DECISION", pd.Series("CASH", index=work.index)
    ).fillna("CASH").astype(str).str.upper()
    work["_quality"] = quality

    # [v34] 알파 게이트 배치면 후보 정렬 = 검증 알파 우선 (v32 세계관 정합)
    alpha_on = _alpha_gate_on(work)
    work["_alpha"] = pd.to_numeric(
        work.get("ALPHA_SCORE", pd.Series(float("nan"), index=work.index)),
        errors="coerce",
    ).fillna(-1)
    _sort_cols = ["_alpha", "_quality"] if alpha_on else ["_quality"]

    buys_df = work[production].sort_values(_sort_cols, ascending=False).head(3)
    watch_df = work[(~production) & action.eq("WATCH")].sort_values(
        _sort_cols, ascending=False
    ).head(3)

    buys = [_stock_payload(row) for _, row in buys_df.iterrows()]
    watch = [_stock_payload(row) for _, row in watch_df.iterrows()]
    # [v54] 실행 대기(근거 통과·상태 미달) 후보 — 공식 매수 0개인 날의 실제 사정
    awaiting_df = work[awaiting_execution_mask(work)].sort_values(
        _sort_cols, ascending=False
    ).head(2)
    awaiting = [_stock_payload(row) for _, row in awaiting_df.iterrows()]
    production_count = int(production.sum())
    watch_count = int(action.eq("WATCH").sum())
    ml_values = (
        work.get("ML_STATUS", pd.Series("UNKNOWN", index=work.index))
        .fillna("UNKNOWN")
        .astype(str)
    )
    ml_status = ml_values.iloc[0] if len(ml_values) else "UNKNOWN"

    breadth_values = pd.to_numeric(
        work.get("MARKET_BREADTH", pd.Series(float("nan"), index=work.index)),
        errors="coerce",
    ).dropna()
    market_breadth = float(breadth_values.median()) if len(breadth_values) else None
    macro_risk = _dominant_text(work, "MACRO_RISK", "UNKNOWN").upper()
    market_regime = _dominant_text(work, "MARKET_REGIME", "UNKNOWN").upper()
    # [v34] 폭락 방어(risk_off) — 실제 최상위 차단자를 화면에 드러낸다.
    # 7/16 실사용: 시장폭 88% '통과' 표시인데 매수 0개 → 사용자가 이유를 알 수 없었음.
    risk_off = _risk_off_state(work)
    # [v55] 차단 상태를 코스피 원본으로 한 번 더 계산해 수치로 붙인다.
    #   _risk_off_state는 CSV의 NEW_ENTRY_BLOCKED/STOP_OVERRIDE_REASON에 의존하므로
    #   컬럼이 없거나 오래된 CSV를 보고 있으면 '왜 없는지'를 말하지 못한다.
    #   v55 실측(프로덕션 CSV): 최근 16영업일 중 15일이 전 종목 차단이었고
    #   코스피는 20일선 대비 최대 -20.1%(7/29) 이탈했다. 즉 원인은 표시가 아니라
    #   실제 폭락이었다 — 그 사실을 숫자로 보여준다.
    try:
        from services.entry_block_status import compute_entry_block_status

        risk_off["market"] = compute_entry_block_status()
    except Exception as exc:
        logger.warning("[v55] 차단 상태 계산 실패 (표시 생략): %s", exc)
        risk_off["market"] = None
    breadth_ready = market_breadth is None or market_breadth >= MARKET_BREADTH_FLOOR
    market_ready = (
        breadth_ready
        and macro_risk not in DANGEROUS_MARKET_RISK
        and market_regime != "DOWN"
        and not risk_off["active"]
    )

    nearest_reason = watch[0]["raw_reason"] if watch else ""
    blockers: list[str] = []
    if risk_off["active"]:
        blockers.append(
            # [v55] '실측 승률 17%' 단정 철회 — v55 재측정이 확인하지 못했다
            # (차단일 반사실 픽 +3.15%, Welch p=0.85 · 순열 p=0.60 · 깊은 폭락
            #  구간은 선행수익 부재로 평가 불가). 규칙 대신 조건을 적는다.
            "폭락 방어(risk_off) 작동 중 — 코스피가 하락하는 20일선 아래로 3% 이상 "
            "이탈해 전 종목 신규 진입을 보류합니다 (안전 쪽 선택 · 수익 효과는 미검증)"
        )
    if market_breadth is not None and market_breadth < MARKET_BREADTH_FLOOR:
        blockers.append(
            f"상승 종목 비율 {market_breadth:.0f}%로 기준 {MARKET_BREADTH_FLOOR:.0f}% 미만"
        )
    if market_regime == "DOWN":
        blockers.append("하락장이라 신규 진입은 AI 알파 최상위 10%만 축소 검토")
    if macro_risk in DANGEROUS_MARKET_RISK:
        blockers.append(f"시장 위험 단계가 높음 ({macro_risk})")
    # [v54] 실행 대기 후보가 있으면 그것이 오늘의 실제 1순위 차단자다.
    #   기존 목록은 '가장 가까운 관찰 후보(watch[0])의 탈락 사유'를 차단자로
    #   올렸다. 다른 종목의 사유라서, 실행 대기 후보가 있는 날에는
    #   '즉시 매수 조건 미충족 · 손익비 부족 1.16'처럼 지금 막고 있는 것과
    #   무관한 문장이 최상단에 왔다(20260805 실측). 실제 차단자를 앞에 둔다.
    if awaiting and not buys:
        _aw = awaiting[0]
        blockers.insert(0, (
            f"{_aw['name']} 상태 "
            f"{str(_aw.get('route', '') or '').strip() or '대기'} — "
            f"근거는 통과, 신규진입 상태가 아니라 사이징 0주"
        ))
    for item in _reason_items(nearest_reason, limit=6):
        if item not in blockers:
            blockers.append(item)
        if len(blockers) >= 3:
            break
    if not blockers and not buys:
        blockers = ["최종 매수 기준을 통과한 종목이 없음"]

    if buys:
        title = f"오늘 신규매수 {len(buys)}개"
        subtitle = "아래 공식 매수 카드만 주문 후보입니다. 지정가·손절가·최대 비중을 함께 지키세요."
        action_label = "공식 후보의 지정가 주문만 준비하세요"
        action_detail = "현재가 추격매수는 하지 말고, 지정가가 체결되지 않으면 그대로 보내세요."
        next_check = "체결 후 손절가와 1차 익절가를 동시에 등록"
        status = "BUY"
    else:
        title = "오늘은 신규매수하지 않습니다"
        subtitle = "공식 매수 종목이 0개입니다. 관찰 후보는 조건에 가까울 뿐 주문 대상이 아닙니다."
        action_label = "신규 주문을 넣지 말고 현금을 유지하세요"
        action_detail = "높은 점수나 관찰 표시는 매수 신호가 아닙니다. 공식 매수가 생길 때까지 기다리세요."
        # [v55] 차단 중이면 '얼마나 더 올라야 풀리는지'를 숫자로 — 막연한 대기 금지
        _m = (risk_off.get("market") or {})
        if _m.get("ok") and _m.get("risk_off"):
            next_check = (
                f"코스피가 20일선 대비 -3% 이내로 회복하면 자동 재개 "
                f"(현재 {_m['deviation_pct']:+.1f}% · {_m['unlock_gap_pct']:.1f}%p 더 필요 · "
                f"{_m['streak_days']}일 연속 차단)"
            )
        elif risk_off["active"]:
            next_check = "코스피가 해제선(20일선 -3% 이내)을 회복하면 매수 후보가 자동으로 올라옵니다"
        else:
            next_check = "다음 데이터 갱신 후 시장과 최종 후보를 다시 확인"
        # [v54] '종목을 못 찾았다'와 '찾았지만 아직 살 상태가 아니다'는 다르다.
        #   v54가 사이징 불가 상태를 공식 매수에서 제외한 대가로 0개인 날이
        #   늘어난다. 그 날 근거까지 통과한 후보가 있으면 이름과 상태를 밝힌다.
        if len(awaiting):
            _a = awaiting[0]
            _rt = str(_a.get("route", "") or "").strip() or "대기"
            subtitle = (
                f"공식 매수 0개 — {_a['name']}: 근거는 통과, 상태 {_rt}라 "
                f"사이징이 0주입니다. 종목이 없는 게 아니라 진입 타이밍이 아닙니다."
            )
            action_detail = (
                "지금 사면 엔진이 계산한 수량(0주)을 무시하는 주문이 됩니다. "
                "상태가 진입 가능으로 바뀌면 자동으로 공식 매수로 올라옵니다."
            )
            next_check = f"{_a['name']} 상태 전환 확인 (현재 {_rt})"
        status = "CASH"

    breadth_value = (
        f"상승 종목 {market_breadth:.0f}%" if market_breadth is not None else "시장폭 데이터 없음"
    )
    market_detail = f"신규 진입 기준 {MARKET_BREADTH_FLOOR:.0f}% 이상"
    if market_regime == "DOWN":
        market_detail += " · 현재 하락장"
    elif macro_risk not in {"NORMAL", "UNKNOWN", ""}:
        market_detail += f" · 위험 {macro_risk}"
    # [v34] risk_off가 최상위 차단자 — 시장폭이 좋아도 '통과'로 오도하지 않는다.
    if risk_off["active"]:
        gate1_status, gate1_value = "차단", "🔒 폭락 방어 작동 중"
        gate1_detail = "코스피가 하락 중인 20일선 아래 — 해제선 회복 시 자동 재개"
    else:
        gate1_status, gate1_value, gate1_detail = (
            "통과" if market_ready else "대기"), breadth_value, market_detail
    # [v34] 후보 검증 문구 — 알파 게이트 배치면 기준을 명시
    _thr = pd.to_numeric(
        work.get("ALPHA_ENTRY_THRESHOLD", pd.Series(float("nan"), index=work.index)),
        errors="coerce",
    ).dropna()
    if alpha_on and len(_thr):
        _top_pct = max(1, int(round(100 - float(_thr.iloc[0]))))
        gate2_detail = f"기준: AI 알파 상위 {_top_pct}% + 자리·손익비 가드"
    else:
        gate2_detail = "관찰 후보는 주문 대상이 아님"
    gates = [
        {
            "step": "1",
            "label": "시장 환경",
            "status": gate1_status,
            "value": gate1_value,
            "detail": gate1_detail,
            "tone": "green" if market_ready else "amber",
        },
        {
            "step": "2",
            "label": "후보 검증",
            "status": "통과" if production_count else ("관찰" if watch_count else "없음"),
            "value": f"공식 {production_count}개 · 관찰 {watch_count}개",
            "detail": gate2_detail,
            "tone": "green" if production_count else "amber",
        },
        {
            "step": "3",
            "label": "오늘 주문",
            "status": "가능" if production_count else "보류",
            "value": "지정가 주문" if production_count else "신규 주문 없음",
            "detail": "공식 매수 카드만 실행",
            "tone": "green" if production_count else "slate",
        },
    ]

    return {
        "status": status,
        "title": title,
        "subtitle": subtitle,
        "action_label": action_label,
        "action_detail": action_detail,
        "next_check": next_check,
        "buys": buys,
        "watch": watch,
        "awaiting": awaiting,       # [v54] 근거 통과·상태 미달로 0주인 후보
        "blockers": blockers,
        "gates": gates,
        "production_count": production_count,
        "watch_count": watch_count,
        "ml_status": ml_status,
        "market_breadth": market_breadth,
        "market_ready": market_ready,
        "macro_risk": macro_risk,
        "market_regime": market_regime,
        "risk_off": risk_off,       # [v34] 폭락 방어 잠금 상태(+해제 진행률)
        "alpha_gate": alpha_on,     # [v34] 알파 전면 게이트 배치 여부
    }


def _money(value: float) -> str:
    return f"{value:,.0f}원" if value > 0 else "—"


def _move_from_entry(entry: float, price: float) -> str:
    if entry <= 0 or price <= 0:
        return ""
    move = (price / entry - 1.0) * 100.0
    return f"{move:+.1f}% vs 진입"


def _render_buy_card(stock: dict[str, Any]) -> None:
    with ui.card().classes("sp-buy-card w-full p-5 rounded-2xl"):
        with ui.row().classes("w-full items-start justify-between gap-3"):
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.badge("공식 매수", color="#10B981").classes("font-bold")
                    ui.label(stock["name"]).classes("text-xl font-bold text-white")
                    ui.label(stock["code"]).classes("text-xs text-slate-400")
                _alpha_chip(stock)
                ui.label(stock["reason"]).classes("text-sm text-emerald-100")
            with ui.column().classes("items-end gap-0"):
                ui.label(f"품질 {stock['score']:.0f}").classes("text-xl font-bold text-emerald-300")
                ui.label(f"최대 {stock['weight']:.0f}% 비중").classes("text-xs text-slate-300")
                ui.label(f"손익비 {stock['rr']:.2f}:1").classes("text-xs text-slate-400")
        with ui.grid(columns=4).classes("sp-price-grid w-full gap-2 mt-3"):
            for label, value, detail, css in [
                ("지정 매수가", _money(stock["entry"]), "익일 지정가 기준", "text-blue-300"),
                ("손절가", _money(stock["stop"]), _move_from_entry(stock["entry"], stock["stop"]), "text-rose-300"),
                ("1차 익절", _money(stock["target"]), _move_from_entry(stock["entry"], stock["target"]), "text-emerald-300"),
                ("연장 목표", _money(stock["target2"]), _move_from_entry(stock["entry"], stock["target2"]), "text-violet-300"),
            ]:
                with ui.column().classes("sp-price-cell gap-0 rounded-xl p-3"):
                    ui.label(label).classes("text-[11px] text-slate-400")
                    ui.label(value).classes(f"text-base font-bold {css}")
                    ui.label(detail or "—").classes("text-[10px] text-slate-500")
        ui.button(
            "근거와 차트 보기",
            on_click=lambda code=stock["code"]: ui.navigate.to(f"/stock/{code}"),
        ).props("flat dense no-caps").classes("self-end text-blue-300 mt-2")


def _alpha_chip(stock: dict[str, Any]) -> None:
    """[v34] AI 알파 칩 — 검증된 예측 점수(당일 백분위) + 그 구간의 실측 승률."""
    alpha = stock.get("alpha")
    if alpha is None or not math.isfinite(alpha) or alpha < 0:
        return
    wp = stock.get("alpha_win_prob")
    wp_txt = (
        f" · 이 구간 실측 승률 {wp * 100:.0f}%"
        if wp is not None and math.isfinite(wp) and 0 < wp < 1
        else ""
    )
    with ui.row().classes("items-center gap-1 rounded-lg px-2 py-0.5").style(
        "background:rgba(59,130,246,.12); border:1px solid rgba(96,165,250,.3); width:fit-content;"
    ):
        ui.label(f"🧠 AI 알파 {alpha:.1f}점{wp_txt}").classes(
            "text-[11px] font-bold text-blue-200"
        )


def _render_watch_card(stock: dict[str, Any], rank: int) -> None:
    with ui.card().classes("sp-watch-card w-full p-4 rounded-2xl"):
        with ui.column().classes("gap-1 min-w-0"):
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.badge("매수 아님", color="#B45309").classes("font-bold")
                ui.label(f"{rank}. {stock['name']}").classes("font-bold text-slate-100")
                ui.label(stock["code"]).classes("text-xs text-slate-500")
            _alpha_chip(stock)
            ui.label(stock["reason"] or "최종 기준 미달").classes(
                "text-sm text-slate-300 leading-relaxed"
            )
        ui.button(
            "근거 보기",
            on_click=lambda code=stock["code"]: ui.navigate.to(f"/stock/{code}"),
        ).props("flat dense no-caps").classes("self-end text-xs text-slate-300 mt-1")


def render_decision_center(df: pd.DataFrame, auth: str = "free") -> None:
    del auth  # reserved for future entitlement-specific detail rows
    summary = build_decision_summary(df)
    ui.add_head_html(
        """
        <style>
          .sp-decision-shell { max-width: 1120px; margin: 0 auto; }
          .sp-decision-hero { background:#0d172b!important;
            border:1px solid rgba(148,163,184,.2); box-shadow:none!important; }
          .sp-decision-status { background:rgba(2,6,23,.38); border:1px solid rgba(148,163,184,.14); }
          .sp-next-check { background:rgba(15,23,42,.64); border:1px solid rgba(148,163,184,.12); }
          .sp-gate-card { background:#111827!important; border:1px solid rgba(148,163,184,.17); box-shadow:none!important; }
          .sp-gate-step { width:1.75rem; height:1.75rem; border-radius:9999px; display:flex; align-items:center;
            justify-content:center; background:#1e293b; color:#cbd5e1; font-size:.75rem; font-weight:800; }
          .sp-blocker-card { background:rgba(69,26,3,.18)!important; border:1px solid rgba(245,158,11,.22); box-shadow:none!important; }
          .sp-blocker-number { width:1.5rem; height:1.5rem; border-radius:9999px; display:flex; align-items:center;
            justify-content:center; background:rgba(245,158,11,.16); color:#fcd34d; font-size:.7rem; font-weight:800; flex:none; }
          .sp-buy-card { background:linear-gradient(135deg,rgba(6,78,59,.72),rgba(15,23,42,.95))!important;
            border:1px solid rgba(52,211,153,.4); box-shadow:none!important; }
          .sp-watch-card { background:#111827!important; border:1px solid rgba(100,116,139,.22); box-shadow:none!important; }
          .sp-price-cell { background:rgba(15,23,42,.55); border:1px solid rgba(148,163,184,.12); }
          .sp-rule-pill { background:rgba(15,23,42,.65); border:1px solid rgba(148,163,184,.12); }
          @media(max-width:720px){
            .sp-price-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important;}
            .sp-gate-grid{grid-template-columns:1fr!important;}
            .sp-decision-hero{padding:1.25rem!important;border-radius:1.25rem!important;}
            .sp-decision-title{font-size:1.55rem!important;line-height:1.32!important;word-break:keep-all;}
            .sp-decision-status{width:100%;align-items:flex-start!important;}
            .sp-rule-row{display:grid!important;grid-template-columns:1fr!important;}
          }
        </style>
        """
    )

    with ui.column().classes("sp-decision-shell w-full gap-4 pb-8"):
        cash = summary["status"] != "BUY"
        with ui.card().classes("sp-decision-hero w-full p-6 md:p-8 rounded-3xl"):
            with ui.row().classes("w-full items-start justify-between gap-4 flex-wrap"):
                with ui.column().classes("gap-2 max-w-3xl grow"):
                    ui.label("오늘 해야 할 일").classes(
                        "text-xs font-bold tracking-[0.18em] text-slate-400 uppercase"
                    )
                    ui.label(summary["action_label"]).classes(
                        "sp-decision-title text-2xl md:text-4xl font-bold "
                        + ("text-amber-200" if cash else "text-emerald-200")
                    )
                    ui.label(summary["action_detail"]).classes(
                        "text-sm md:text-base text-slate-300 leading-relaxed"
                    )
                with ui.column().classes("sp-decision-status items-end gap-1 rounded-2xl px-4 py-3 min-w-[150px]"):
                    ui.label("최종 결정").classes("text-[10px] font-bold tracking-wider text-slate-500")
                    ui.badge(
                        "CASH · 주문 보류" if cash else "BUY · 지정가",
                        color="#F59E0B" if cash else "#10B981",
                    ).classes("text-sm font-bold px-3 py-1")
                    ui.label(f"공식 매수 {summary['production_count']}개").classes("text-xs text-slate-400")
            with ui.row().classes("sp-next-check w-full items-start gap-3 rounded-xl p-3 mt-4"):
                ui.label("다음 확인").classes("text-xs font-bold text-sky-300 shrink-0")
                ui.label(summary["next_check"]).classes("text-xs md:text-sm text-slate-300")
            # [v34] 폭락 방어(risk_off) 잠금 스트립 — '왜 안 사는지'의 최상위 이유를
            # 해제 조건·진행률과 함께 노출 (7/16 피드백: 차단자가 화면에 없었음)
            _ro = summary.get("risk_off") or {}
            _mkt = _ro.get("market") or {}
            if _ro.get("active") or _mkt.get("risk_off"):
                with ui.column().classes("w-full gap-1 rounded-xl p-3 mt-2").style(
                    "background:rgba(180,83,9,.10); border:1px solid rgba(245,158,11,.28);"
                ):
                    # [v55] 문구 정직화. 기존 문구는 '실측 승률 17%라 자동 차단'이라고
                    # 단정했지만 v55 재측정은 그것을 확인하지 못했다 — 패널 내 차단일
                    # 6일의 반사실 픽은 오히려 +3.15%(승률 66.7%)였고 허용일보다
                    # 나빴다는 증거가 없다(Welch -0.72%p t=-0.20 p=0.85 · 순열 p=0.60).
                    # 깊은 폭락 구간(-8~-20%)은 아직 선행수익이 없어 평가 불가이고,
                    # v49 기록대로 이 축은 창 길이에 따라 부호가 뒤집힌 이력이 있다.
                    # 규칙은 그대로 두되(안전 쪽), '검증됨'이라고 말하지 않는다.
                    ui.label(
                        "🔒 폭락 방어(risk_off) 작동 중 — 코스피가 하락하는 20일선 아래로 "
                        "3% 이상 이탈한 구간에서는 전 종목 신규진입을 자동 보류합니다"
                    ).classes("text-xs md:text-sm font-bold text-amber-200")
                    if _mkt.get("ok"):
                        ui.label(_mkt.get("reason", "")).classes(
                            "text-xs text-amber-100 font-semibold")
                    if _ro.get("line"):
                        ui.label(_ro["line"]).classes("text-xs text-slate-300")
                    ui.label(
                        "이 보류 규칙이 수익을 지킨다는 것은 아직 통계로 확립되지 않았습니다 "
                        "— 안전 쪽 선택이며, 깊은 하락 구간의 성과는 표본이 쌓이면 재검증합니다"
                    ).classes("text-[10px] text-slate-500")
                    if _ro.get("progress") is not None:
                        with ui.element("div").style(
                            "width:100%; height:8px; background:rgba(148,163,184,.15);"
                            " border-radius:6px; overflow:hidden; margin-top:2px;"
                        ):
                            ui.element("div").style(
                                f"width:{_ro['progress']:.0f}%; height:100%;"
                                " background:linear-gradient(90deg,#F59E0B,#10B981);"
                            )
                        ui.label(
                            "해제 진행률 — 코스피가 20일선 -3% 이내로 회복하면 매수 후보가 자동 재개됩니다"
                        ).classes("text-[10px] text-slate-500")

        ui.label("매수 가능 여부").classes("text-lg font-bold text-white mt-1")
        with ui.grid(columns=3).classes("sp-gate-grid w-full gap-2"):
            for gate in summary["gates"]:
                status_color = {
                    "green": "#059669",
                    "amber": "#B45309",
                    "slate": "#475569",
                }[gate["tone"]]
                with ui.card().classes("sp-gate-card w-full p-4 rounded-2xl"):
                    with ui.row().classes("w-full items-center justify-between gap-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(gate["step"]).classes("sp-gate-step")
                            ui.label(gate["label"]).classes("text-sm font-bold text-slate-200")
                        ui.badge(gate["status"], color=status_color).classes("font-bold")
                    ui.label(gate["value"]).classes("text-lg font-bold text-white mt-3")
                    ui.label(gate["detail"]).classes("text-xs text-slate-500 mt-1")

        if summary["buys"]:
            ui.label("오늘의 공식 매수").classes("text-lg font-bold text-white mt-2")
            for stock in summary["buys"]:
                _render_buy_card(stock)
        else:
            with ui.card().classes("sp-blocker-card w-full p-4 md:p-5 rounded-2xl"):
                ui.label("오늘 매수가 막힌 이유").classes("font-bold text-amber-200")
                with ui.column().classes("w-full gap-2 mt-2"):
                    for index, reason in enumerate(summary["blockers"], 1):
                        with ui.row().classes("w-full items-start gap-2"):
                            ui.label(str(index)).classes("sp-blocker-number")
                            ui.label(reason).classes("text-sm text-slate-200 leading-relaxed")

        if summary["watch"]:
            with ui.row().classes("w-full items-end justify-between gap-3 flex-wrap mt-2"):
                with ui.column().classes("gap-0"):
                    ui.label("가장 가까운 관찰 후보").classes("text-lg font-bold text-white")
                    ui.label("조건에 가까운 순서이며 매수 추천이 아닙니다.").classes("text-xs text-slate-500")
                ui.badge(f"전체 관찰 {summary['watch_count']}개", color="#475569")
            for rank, stock in enumerate(summary["watch"], 1):
                _render_watch_card(stock, rank)
            if summary["watch_count"] > len(summary["watch"]):
                ui.label(
                    f"상위 {len(summary['watch'])}개만 표시했습니다. 나머지 {summary['watch_count'] - len(summary['watch'])}개는 종목 탭에서 확인하세요."
                ).classes("text-xs text-slate-500")

        with ui.row().classes("sp-rule-row w-full gap-2 mt-2"):
            for text in [
                "공식 매수 0개면 주문 없음",
                "관찰 후보·AI 점수만으로 매수 금지",
                "매수 시 지정가·손절가·비중 동시 준수",
                # [v45 정정] v37의 '당일 3~5종목 분산' 권고는 구 게이트(알파≥80,
                # 저점추세 미적용) 풀에서 측정한 값이었다. 현행 게이트(알파≥85 +
                # 저점추세≥0, 풀 27개/일)로 재측정하면 정반대 —
                #   top1 +2.86%/일 승률50% (누적 +183%)  vs  top3 -0.12%/47%
                #   페어드 t=-2.92 p=0.005 (top3가 유의하게 열위)
                # 게이트가 정제된 뒤로는 1등 픽이 최선이고, 분산은 '같은 날 쪼개기'가
                # 아니라 '종목당 비중 제한 + 날짜 분산'으로 달성한다.
                "하루 1종목 · 종목당 3~5% 비중 (당일 분산은 실측 열위)",
            ]:
                with ui.row().classes("sp-rule-pill items-center gap-2 rounded-xl px-3 py-2 grow"):
                    ui.label("✓").classes("text-emerald-400 font-bold")
                    ui.label(text).classes("text-xs text-slate-400")

        ui.label("성과를 보장하지 않습니다. 오늘 탭의 공식 결정과 가격 규칙을 함께 지켜야 검증 결과와 실제 운용의 차이를 줄일 수 있습니다.").classes(
            "text-xs text-slate-600 mt-1"
        )

