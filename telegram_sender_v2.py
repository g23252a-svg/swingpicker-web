# -*- coding: utf-8 -*-
"""
telegram_sender_v2.py — 📱 텔레그램 자동 발송 확장판
═══════════════════════════════════════════════════════
기존 telegram_sender.py 확장:

[신규 기능]
 1. Top 3 브리핑 + 시장 시황 종합 알림
 2. Hard Block 필터링 결과 알림 (차단된 종목 리스트)
 3. 포트폴리오 변동 알림 (급등/급락 종목)
 4. 스케줄 기반 자동 발송 (collector.py 연동)

[통합 방법]
 collector.py의 텔레그램 발송 부분을 이 모듈로 교체:
   기존: from telegram_sender import send_telegram_auto
   변경: from telegram_sender_v2 import send_telegram_enhanced
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import requests

try:
    from collector_config import DEFAULT_CONFIG
except ImportError:
    class _FakeConfig:
        tg_token = ""
        tg_id = ""
    DEFAULT_CONFIG = _FakeConfig()

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  코어 전송
# ═══════════════════════════════════════════════════

def _send_message(text: str, token: str, chat_id: str,
                  parse_mode: str = "HTML", disable_preview: bool = True) -> bool:
    """텔레그램 메시지 발송 (단일)."""
    if not token or not chat_id:
        logger.info("텔레그램 미설정 (TG_TOKEN/TG_ID)")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_preview,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.ok:
            logger.info(f"텔레그램 발송 성공 ({len(text)}자)")
            return True
        else:
            logger.warning(f"텔레그램 발송 실패: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        logger.error(f"텔레그램 발송 오류: {e}")
        return False


def _split_and_send(text: str, token: str, chat_id: str, max_len: int = 4000) -> bool:
    """4096자 제한 분할 발송."""
    if len(text) <= max_len:
        return _send_message(text, token, chat_id)

    parts = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            parts.append(current)
            current = line
        else:
            current += "\n" + line if current else line
    if current:
        parts.append(current)

    success = True
    for i, part in enumerate(parts):
        if i > 0:
            import time; time.sleep(0.5)  # Rate limit 방어
        if not _send_message(part, token, chat_id):
            success = False
    return success


# ═══════════════════════════════════════════════════
#  1. 기존 호환 함수 (send_telegram_auto 대체)
# ═══════════════════════════════════════════════════

def send_telegram_auto(
    df: pd.DataFrame,
    trade_ymd: str,
    market_summary: str = "",
    limit_count: int = 5,
    tg_token: str = "",
    tg_id: str = "",
) -> None:
    """기존 telegram_sender.py와 100% 호환 — drop-in 교체 가능."""
    token = tg_token or DEFAULT_CONFIG.tg_token
    chat_id = tg_id or DEFAULT_CONFIG.tg_id

    if not token or not chat_id:
        logger.info("텔레그램 미설정 (TG_TOKEN/TG_ID)")
        return

    try:
        top = df.head(limit_count)
        lines = [f"📊 <b>LDY Pro Trader [{trade_ymd}]</b>"]
        if market_summary:
            lines.append(market_summary)
        lines.append("")

        for i, (_, row) in enumerate(top.iterrows(), 1):
            name = row.get("종목명", row.get("name", ""))
            code = str(row.get("종목코드", "")).zfill(6)
            route = row.get("ROUTE", "")
            score = row.get("DISPLAY_SCORE", row.get("FINAL_SCORE", 0))
            buy = row.get("매수가", row.get("buy_price", 0))

            route_emoji = {"ATTACK": "🔴", "ARMED": "🟠", "WAIT": "🔵"}.get(route, "⚪")
            lines.append(f"{i}. {route_emoji}<b>{name}</b> ({code})")
            lines.append(f"   점수: {score:.0f}  |  매수가: {int(buy):,}")

            # Hard Block 표시
            hb = row.get("HARD_BLOCK", "")
            if hb:
                lines.append(f"   ⚠️ Hard Block: {hb}")

            # 손절/익절가
            stop = row.get("손절가", row.get("stop_price", 0))
            tp1 = row.get("추천매도가1", row.get("target_price_1", 0))
            if stop > 0 and tp1 > 0:
                lines.append(f"   손절: {int(stop):,} → 익절: {int(tp1):,}")
            lines.append("")

        text = "\n".join(lines)
        _split_and_send(text, token, chat_id)

    except Exception as e:
        logger.error(f"텔레그램 auto 발송 오류: {e}")


# ═══════════════════════════════════════════════════
#  2. [신규] Top 3 브리핑 + 시장 시황 종합 알림
# ═══════════════════════════════════════════════════

def send_briefing_alert(
    trade_ymd: str,
    top_stocks: List[Dict[str, Any]],
    market_temp: str = "",
    breadth: Dict[str, float] = None,
    macro_msg: str = "",
    leading_sectors: List[str] = None,
    tg_token: str = "",
    tg_id: str = "",
) -> bool:
    """Top 3 브리핑 + 시장 시황 종합 알림.

    Args:
        top_stocks: [{"name": str, "code": str, "score": float, "route": str,
                       "buy_price": int, "reason": str}, ...]
        market_temp: 시장 온도 문자열
        breadth: {"ALL": float, "KOSPI": float, "KOSDAQ": float}
        macro_msg: 매크로 필터 메시지
        leading_sectors: 주도 섹터 리스트
    """
    token = tg_token or DEFAULT_CONFIG.tg_token
    chat_id = tg_id or DEFAULT_CONFIG.tg_id

    lines = [f"🌅 <b>오늘의 브리핑 [{trade_ymd}]</b>", ""]

    # 시장 시황
    lines.append("━━━ 📡 시장 시황 ━━━")
    if market_temp:
        lines.append(f"🌡️ {market_temp}")
    if breadth:
        b_all = breadth.get("ALL", 0)
        b_emoji = "🟢" if b_all > 60 else "🟡" if b_all > 40 else "🔴"
        lines.append(f"{b_emoji} Breadth: {b_all:.0f}%")
    if macro_msg:
        lines.append(f"📊 {macro_msg}")
    if leading_sectors:
        lines.append(f"🚀 주도섹터: {', '.join(leading_sectors[:3])}")
    lines.append("")

    # Top 3 추천
    lines.append("━━━ 🏆 Today's Top 3 ━━━")
    for i, stock in enumerate(top_stocks[:3], 1):
        medal = ["🥇", "🥈", "🥉"][i - 1]
        route = stock.get("route", "")
        route_emoji = {"ATTACK": "🔴", "ARMED": "🟠", "WAIT": "🔵"}.get(route, "⚪")

        lines.append(f"{medal} <b>{stock.get('name', '')}</b> ({stock.get('code', '')})")
        lines.append(f"   {route_emoji}{route} | 점수: {stock.get('score', 0):.0f}")
        if stock.get("buy_price"):
            lines.append(f"   매수가: {int(stock['buy_price']):,}")
        if stock.get("reason"):
            lines.append(f"   💡 {stock['reason'][:60]}")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━")
    lines.append("💎 SwingPicker에서 상세 분석 보기")

    text = "\n".join(lines)
    return _split_and_send(text, token, chat_id)


# ═══════════════════════════════════════════════════
#  3. [신규] Hard Block 필터링 결과 알림
# ═══════════════════════════════════════════════════

def send_hard_block_alert(
    trade_ymd: str,
    blocked_stocks: List[Dict[str, Any]],
    total_candidates: int = 0,
    tg_token: str = "",
    tg_id: str = "",
) -> bool:
    """Hard Block에 의해 차단된 종목 알림.

    Args:
        blocked_stocks: [{"name": str, "code": str, "block_reason": str, "score": float}, ...]
        total_candidates: 전체 후보 종목 수
    """
    token = tg_token or DEFAULT_CONFIG.tg_token
    chat_id = tg_id or DEFAULT_CONFIG.tg_id

    if not blocked_stocks:
        return True

    lines = [
        f"🚫 <b>Hard Block 알림 [{trade_ymd}]</b>",
        f"전체 {total_candidates}개 후보 중 {len(blocked_stocks)}개 차단",
        "",
    ]

    for stock in blocked_stocks[:10]:  # 최대 10개
        name = stock.get("name", "")
        reason = stock.get("block_reason", "")
        score = stock.get("score", 0)
        lines.append(f"🚫 <b>{name}</b> (점수: {score:.0f})")
        lines.append(f"   사유: {reason}")

    if len(blocked_stocks) > 10:
        lines.append(f"\n... 외 {len(blocked_stocks) - 10}개 종목")

    text = "\n".join(lines)
    return _split_and_send(text, token, chat_id)


# ═══════════════════════════════════════════════════
#  4. [신규] 포트폴리오 급변 알림
# ═══════════════════════════════════════════════════

def send_portfolio_alert(
    portfolio_changes: List[Dict[str, Any]],
    threshold_pct: float = 5.0,
    tg_token: str = "",
    tg_id: str = "",
) -> bool:
    """포트폴리오 보유 종목 급등/급락 알림.

    Args:
        portfolio_changes: [{"name": str, "change_pct": float, "current_price": int,
                             "action": str}, ...]
        threshold_pct: 알림 기준 변동률(%)
    """
    token = tg_token or DEFAULT_CONFIG.tg_token
    chat_id = tg_id or DEFAULT_CONFIG.tg_id

    alerts = [c for c in portfolio_changes if abs(c.get("change_pct", 0)) >= threshold_pct]
    if not alerts:
        return True

    lines = [f"⚡ <b>포트폴리오 급변 알림</b>", ""]

    for c in sorted(alerts, key=lambda x: abs(x.get("change_pct", 0)), reverse=True):
        pct = c.get("change_pct", 0)
        emoji = "📈" if pct > 0 else "📉"
        price = c.get("current_price", 0)
        action = c.get("action", "")

        lines.append(f"{emoji} <b>{c.get('name', '')}</b>  {pct:+.1f}%")
        if price > 0:
            lines.append(f"   현재가: {price:,}원")
        if action:
            lines.append(f"   💡 {action}")
        lines.append("")

    text = "\n".join(lines)
    return _split_and_send(text, token, chat_id)


# ═══════════════════════════════════════════════════
#  5. [신규] 종합 발송 (collector.py 통합용)
# ═══════════════════════════════════════════════════

def send_telegram_enhanced(
    df: pd.DataFrame,
    trade_ymd: str,
    market_summary: str = "",
    market_temp: str = "",
    breadth: Dict[str, float] = None,
    macro_msg: str = "",
    leading_sectors: List[str] = None,
    blocked_stocks: List[Dict[str, Any]] = None,
    limit_count: int = 5,
    send_briefing: bool = True,
    send_blocks: bool = True,
    tg_token: str = "",
    tg_id: str = "",
) -> None:
    """collector.py 완료 시점에 호출하는 종합 발송 함수.

    기존 send_telegram_auto 대체 + 브리핑 + Hard Block 알림 통합.

    [collector.py 연동 예시]
    기존:
        send_telegram_auto(df_out, trade_ymd, market_summary=summary_text, limit_count=5)

    변경:
        send_telegram_enhanced(
            df_out, trade_ymd,
            market_summary=summary_text,
            market_temp=mkt_temp,
            breadth=breadth,
            macro_msg=macro_msg,
            leading_sectors=top_sectors,
            blocked_stocks=blocked_list,
            limit_count=5,
        )
    """
    token = tg_token or DEFAULT_CONFIG.tg_token
    chat_id = tg_id or DEFAULT_CONFIG.tg_id

    if not token or not chat_id:
        logger.info("텔레그램 미설정 (TG_TOKEN/TG_ID)")
        return

    # ── 1. Top 3 브리핑 알림 ──
    if send_briefing and not df.empty:
        top_stocks = []
        for _, row in df.head(3).iterrows():
            top_stocks.append({
                "name": row.get("종목명", ""),
                "code": str(row.get("종목코드", "")).zfill(6),
                "score": float(row.get("DISPLAY_SCORE", row.get("FINAL_SCORE", 0))),
                "route": str(row.get("ROUTE", "")),
                "buy_price": int(row.get("매수가", row.get("buy_price", 0)) or 0),
                "reason": str(row.get("DART_REASON", row.get("AI_REASON", "")))[:60],
            })
        send_briefing_alert(
            trade_ymd, top_stocks,
            market_temp=market_temp,
            breadth=breadth,
            macro_msg=macro_msg,
            leading_sectors=leading_sectors,
            tg_token=token, tg_id=chat_id,
        )

    # ── 2. 기존 추천 종목 리스트 ──
    if not df.empty:
        import time; time.sleep(1)
        send_telegram_auto(
            df, trade_ymd,
            market_summary=market_summary,
            limit_count=limit_count,
            tg_token=token, tg_id=chat_id,
        )

    # ── 3. Hard Block 알림 ──
    if send_blocks and blocked_stocks:
        import time; time.sleep(1)
        send_hard_block_alert(
            trade_ymd, blocked_stocks,
            total_candidates=len(df) + len(blocked_stocks),
            tg_token=token, tg_id=chat_id,
        )

    logger.info(f"📱 텔레그램 종합 발송 완료 ({trade_ymd})")
