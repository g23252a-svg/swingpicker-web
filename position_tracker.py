# -*- coding: utf-8 -*-
"""
position_tracker.py — 실시간 포지션 트래킹 & 자동 알림
═══════════════════════════════════════════════════
[v14] P3 #14

핵심 원칙:
  1. 포지션 SSOT: positions.json 단일 파일에 미청산 포지션 관리
  2. 알림 중복 방지: event_key 기반 idempotency
  3. #13 calibration 연결: 청산 시 realized_pnl → per_trade_log 자동 기록
  4. 기업행위/갭/휴장 방어

사용법:
  collector.main() Step 10 이후:
    from position_tracker import track_open_positions
    track_open_positions(OUT_DIR, trade_ymd)
"""
import os
import json
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  1. 포지션 스키마 (SSOT)
# ═══════════════════════════════════════════════════

@dataclass
class Position:
    """미청산 포지션 단위"""
    code: str
    name: str
    entry_ymd: str           # 진입일
    entry_px: float          # 진입가
    qty: int = 0             # 수량
    stop_px: float = 0.0     # 현재 손절가
    stop_px_initial: float = 0.0  # 최초 손절가
    take_px1: float = 0.0    # 목표가1
    take_px2: float = 0.0    # 목표가2
    trailing_high: float = 0.0    # 트레일링 최고가
    status: str = "OPEN"     # OPEN / CLOSED_STOP / CLOSED_TP / CLOSED_MANUAL
    last_check_ymd: str = ""
    last_close_px: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl_pct: float = 0.0
    close_ymd: str = ""      # 청산일
    close_px: float = 0.0    # 청산가
    close_reason: str = ""   # 청산 사유
    alerted_events: List[str] = field(default_factory=list)  # 발송 완료 이벤트 키


# ═══════════════════════════════════════════════════
#  2. 포지션 저장소
# ═══════════════════════════════════════════════════

def _positions_path(out_dir: str) -> str:
    return os.path.join(out_dir, "positions.json")


def _history_path(out_dir: str) -> str:
    return os.path.join(out_dir, "positions_history.json")


def load_positions(out_dir: str) -> List[Position]:
    """미청산 포지션 로드"""
    path = _positions_path(out_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Position(**d) for d in data]
    except Exception as e:
        logger.warning(f"포지션 로드 실패: {e}")
        return []


def save_positions(out_dir: str, positions: List[Position]) -> None:
    """미청산 포지션 저장 (OPEN만)"""
    open_pos = [p for p in positions if p.status == "OPEN"]
    path = _positions_path(out_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in open_pos], f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"포지션 저장 실패: {e}")


def save_to_history(out_dir: str, closed: List[Position]) -> None:
    """청산 완료 포지션을 히스토리에 append"""
    if not closed:
        return
    path = _history_path(out_dir)
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.extend([asdict(p) for p in closed])
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"히스토리 저장 실패: {e}")


# ═══════════════════════════════════════════════════
#  3. 이벤트 감지 + 알림 중복 방지
# ═══════════════════════════════════════════════════

@dataclass
class TrackEvent:
    """포지션 이벤트"""
    code: str
    name: str
    event_type: str   # STOP_HIT, TP1_HIT, TP2_HIT, TRAILING_UPDATE, WARN_DRAWDOWN
    message: str
    event_key: str    # 중복 방지 키


def _make_event_key(code: str, entry_ymd: str, event_type: str, check_ymd: str) -> str:
    """이벤트 키: (종목, 진입일, 이벤트, 체크일)"""
    return f"{code}_{entry_ymd}_{event_type}_{check_ymd}"


def detect_events(
    pos: Position,
    today_close: float,
    today_high: float,
    today_low: float,
    check_ymd: str,
    corporate_action_threshold: float = 30.0,
) -> Tuple[List[TrackEvent], Position]:
    """
    포지션 이벤트 감지.
    Returns: (events, updated_position)

    안전장치:
    - 기업행위: |일일변동| > threshold → SUSPENDED 경고만
    - 갭: 종가 기준 판정 (장중 갭은 snapshot 한계로 종가 사용)
    - 중복: event_key로 이미 발송된 이벤트 스킵
    """
    events = []
    p = pos  # 참조

    if p.status != "OPEN":
        return events, p

    if today_close <= 0 or p.entry_px <= 0:
        return events, p

    # 기업행위 필터
    if p.last_close_px > 0:
        daily_change = abs(today_close / p.last_close_px - 1) * 100
        if daily_change > corporate_action_threshold:
            key = _make_event_key(p.code, p.entry_ymd, "CORPORATE_ACTION", check_ymd)
            if key not in p.alerted_events:
                events.append(TrackEvent(
                    code=p.code, name=p.name,
                    event_type="CORPORATE_ACTION",
                    message=f"⚠️ {p.name} 비정상 변동 {daily_change:.1f}% (기업행위 의심)",
                    event_key=key,
                ))
            # 기업행위 의심 시 SL/TP 판정 스킵
            p.last_close_px = today_close
            p.last_check_ymd = check_ymd
            return events, p

    # 현재 수익률
    unrealized = (today_close / p.entry_px - 1) * 100
    p.unrealized_pnl_pct = round(unrealized, 2)
    p.last_close_px = today_close
    p.last_check_ymd = check_ymd

    # 트레일링 최고가 업데이트
    if today_high > p.trailing_high:
        p.trailing_high = today_high

    # ── 이벤트 판정 ──

    # (1) 손절 히트
    if p.stop_px > 0 and today_close <= p.stop_px:
        key = _make_event_key(p.code, p.entry_ymd, "STOP_HIT", check_ymd)
        if key not in p.alerted_events:
            events.append(TrackEvent(
                code=p.code, name=p.name,
                event_type="STOP_HIT",
                message=f"🔴 {p.name} 손절 도달! 종가={today_close:,.0f} ≤ SL={p.stop_px:,.0f} ({unrealized:+.1f}%)",
                event_key=key,
            ))
        p.status = "CLOSED_STOP"
        p.close_ymd = check_ymd
        p.close_px = today_close
        p.close_reason = "STOP_HIT"
        p.realized_pnl_pct = round(unrealized, 2)
        return events, p

    # (2) 목표가1 도달
    if p.take_px1 > 0 and today_close >= p.take_px1:
        key = _make_event_key(p.code, p.entry_ymd, "TP1_HIT", check_ymd)
        if key not in p.alerted_events:
            events.append(TrackEvent(
                code=p.code, name=p.name,
                event_type="TP1_HIT",
                message=f"🟢 {p.name} TP1 도달! 종가={today_close:,.0f} ≥ TP1={p.take_px1:,.0f} ({unrealized:+.1f}%)",
                event_key=key,
            ))

    # (3) 목표가2 도달 → 청산
    if p.take_px2 > 0 and today_close >= p.take_px2:
        key = _make_event_key(p.code, p.entry_ymd, "TP2_HIT", check_ymd)
        if key not in p.alerted_events:
            events.append(TrackEvent(
                code=p.code, name=p.name,
                event_type="TP2_HIT",
                message=f"🏆 {p.name} TP2 도달! 종가={today_close:,.0f} ≥ TP2={p.take_px2:,.0f} ({unrealized:+.1f}%)",
                event_key=key,
            ))
        p.status = "CLOSED_TP"
        p.close_ymd = check_ymd
        p.close_px = today_close
        p.close_reason = "TP2_HIT"
        p.realized_pnl_pct = round(unrealized, 2)
        return events, p

    # (4) 드로다운 경고 (-5% 이하)
    if unrealized <= -5.0:
        key = _make_event_key(p.code, p.entry_ymd, "WARN_DRAWDOWN", check_ymd)
        if key not in p.alerted_events:
            events.append(TrackEvent(
                code=p.code, name=p.name,
                event_type="WARN_DRAWDOWN",
                message=f"⚠️ {p.name} 드로다운 {unrealized:+.1f}% (진입={p.entry_px:,.0f}, 현재={today_close:,.0f})",
                event_key=key,
            ))

    return events, p


# ═══════════════════════════════════════════════════
#  4. 텔레그램 발송
# ═══════════════════════════════════════════════════

def send_position_alerts(
    events: List[TrackEvent],
    tg_token: str = "",
    tg_id: str = "",
) -> List[str]:
    """
    이벤트를 텔레그램으로 발송.
    Returns: 발송 성공한 event_key 목록
    """
    token = tg_token or os.environ.get("TG_TOKEN", "")
    chat_id = tg_id or os.environ.get("TG_ID", "")

    if not token or not chat_id or not events:
        return [e.event_key for e in events]  # 토큰 없으면 "발송한 것"으로 처리 (테스트용)

    sent_keys = []
    import requests
    for ev in events:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": ev.message, "disable_web_page_preview": True}
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                sent_keys.append(ev.event_key)
            else:
                logger.warning(f"텔레그램 실패 {ev.event_key}: {resp.status_code}")
        except Exception as e:
            logger.warning(f"텔레그램 에러 {ev.event_key}: {e}")

    return sent_keys


# ═══════════════════════════════════════════════════
#  5. #13 calibration 연결
# ═══════════════════════════════════════════════════

def record_closed_to_tradelog(out_dir: str, closed: List[Position]) -> int:
    """
    청산된 포지션을 per_trade_log.csv에 기록.
    → 다음 auto_calibrate 실행 시 자동으로 승률 테이블에 반영.
    """
    if not closed:
        return 0

    records = []
    for p in closed:
        if p.entry_px <= 0:
            continue
        ret = (p.close_px / p.entry_px - 1) * 100 if p.close_px > 0 else 0
        risk = p.entry_px - p.stop_px_initial if p.stop_px_initial > 0 else p.entry_px * 0.05
        reward = p.take_px1 - p.entry_px if p.take_px1 > 0 else p.entry_px * 0.10
        b_ratio = reward / risk if risk > 0 else 1.0

        records.append({
            "rec_date": p.entry_ymd,
            "code": p.code,
            "method": "POSITION_TRACK",
            "topk": 0,
            "horizon": 0,  # 실제 보유기간은 가변
            "score": 0,
            "entry_price": p.entry_px,
            "exit_price": p.close_px,
            "stop_price": p.stop_px_initial,
            "target_price": p.take_px1,
            "ret_pct": round(ret, 4),
            "win": 1 if ret > 0 else 0,
            "exit_type": p.close_reason,
            "b_ratio": round(b_ratio, 4),
            # [v4.0] 세그먼트 축 — Position에 진입시점 값이 있으면 기록, 없으면 ""
            "MACRO_REGIME_MODE": getattr(p, "entry_regime", "") or "",
            "ACTION_TIER": getattr(p, "entry_action_tier", "") or "",
            "ROUTE": getattr(p, "entry_route", "") or "",
            "TOP_PICK_TYPE": getattr(p, "entry_top_pick_type", "") or "",
        })

    if not records:
        return 0

    try:
        from kelly_calibrator import save_per_trade_log
        path = save_per_trade_log(out_dir, records, asof_ymd=closed[0].close_ymd)
        logger.info(f"📝 청산 {len(records)}건 → per_trade_log 기록")
        return len(records)
    except Exception as e:
        logger.warning(f"per_trade_log 기록 실패: {e}")
        return 0


# ═══════════════════════════════════════════════════
#  6. 메인 루프
# ═══════════════════════════════════════════════════

def _load_today_prices(out_dir: str, ymd: str) -> Dict[str, Dict[str, float]]:
    """오늘자 price_snapshot 로드"""
    path = os.path.join(out_dir, f"price_snapshot_{ymd}.csv")
    if not os.path.exists(path):
        path = os.path.join(out_dir, "price_snapshot_latest.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype={"종목코드": str})
        df["종목코드"] = df["종목코드"].str.zfill(6)
        result = {}
        for _, row in df.iterrows():
            result[row["종목코드"]] = {
                "open": float(row.get("시가", 0) or 0),
                "high": float(row.get("고가", 0) or 0),
                "low": float(row.get("저가", 0) or 0),
                "close": float(row.get("종가", 0) or 0),
            }
        return result
    except Exception:
        return {}


def track_open_positions(
    out_dir: str,
    check_ymd: str,
    tg_token: str = "",
    tg_id: str = "",
) -> Dict:
    """
    collector.main() 끝에서 호출.
    1. 미청산 포지션 로드
    2. 오늘 종가 대조 → 이벤트 감지
    3. 알림 발송 (중복 방지)
    4. 청산 포지션 → 히스토리 + per_trade_log
    5. 포지션 저장

    Returns: {"checked": int, "events": int, "closed": int}
    """
    positions = load_positions(out_dir)
    if not positions:
        return {"checked": 0, "events": 0, "closed": 0}

    prices = _load_today_prices(out_dir, check_ymd)
    if not prices:
        logger.info("포지션 트래킹: 오늘 가격 없음")
        return {"checked": 0, "events": 0, "closed": 0}

    all_events = []
    closed_positions = []
    updated = []

    for pos in positions:
        if pos.status != "OPEN":
            continue

        px = prices.get(pos.code)
        if not px:
            updated.append(pos)
            continue

        events, pos_updated = detect_events(
            pos,
            today_close=px["close"],
            today_high=px["high"],
            today_low=px["low"],
            check_ymd=check_ymd,
        )

        # 새 이벤트 수집
        new_events = [e for e in events if e.event_key not in pos_updated.alerted_events]
        all_events.extend(new_events)

        # 발송
        if new_events:
            sent_keys = send_position_alerts(new_events, tg_token, tg_id)
            pos_updated.alerted_events.extend(sent_keys)

        if pos_updated.status != "OPEN":
            closed_positions.append(pos_updated)
        else:
            updated.append(pos_updated)

    # 청산 → 히스토리 + calibration
    if closed_positions:
        save_to_history(out_dir, closed_positions)
        record_closed_to_tradelog(out_dir, closed_positions)

    # 미청산만 저장
    save_positions(out_dir, updated)

    summary = {
        "checked": len(positions),
        "events": len(all_events),
        "closed": len(closed_positions),
    }
    logger.info(f"📍 포지션 트래킹: {summary}")
    return summary


# ═══════════════════════════════════════════════════
#  7. 추천에서 포지션 자동 등록
# ═══════════════════════════════════════════════════

def register_from_recommendations(
    out_dir: str,
    rec_df: pd.DataFrame,
    entry_ymd: str,
    top_n: int = 5,
) -> int:
    """
    오늘 추천 상위 N개를 미청산 포지션으로 등록.
    이미 등록된 종목은 스킵 (중복 방지).
    """
    existing = load_positions(out_dir)
    existing_codes = {p.code for p in existing if p.status == "OPEN"}

    # 히스토리(청산 완료)도 중복 체크 — 같은 날 청산된 종목 재등록 방지
    hist_path = _history_path(out_dir)
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            for h in hist:
                existing_codes.add(h.get("code", ""))
        except Exception:
            pass

    registered = 0
    for _, row in rec_df.head(top_n).iterrows():
        code = str(row.get("종목코드", "")).zfill(6)
        if code in existing_codes:
            continue

        pos = Position(
            code=code,
            name=str(row.get("종목명", "")),
            entry_ymd=entry_ymd,
            entry_px=float(row.get("매수가", row.get("추천매수가", 0)) or 0),
            stop_px=float(row.get("손절가", 0) or 0),
            stop_px_initial=float(row.get("손절가", 0) or 0),
            take_px1=float(row.get("TP1", row.get("추천매도가1", 0)) or 0),
            take_px2=float(row.get("TP2", row.get("추천매도가2", 0)) or 0),
            trailing_high=float(row.get("매수가", row.get("추천매수가", 0)) or 0),
        )
        if pos.entry_px > 0:
            existing.append(pos)
            existing_codes.add(code)
            registered += 1

    save_positions(out_dir, existing)
    logger.info(f"📌 포지션 등록: {registered}개 신규")
    return registered
