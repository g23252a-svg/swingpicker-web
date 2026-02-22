# -*- coding: utf-8 -*-
"""
telegram_sender.py — 텔레그램 자동 발송
═══════════════════════════════════════════════════
[v14] #9 collector.py 분할 — 텔레그램 전용
"""
import os
import logging
from typing import Optional

import pandas as pd
import requests

from collector_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def send_telegram_auto(
    df: pd.DataFrame,
    trade_ymd: str,
    market_summary: str = "",
    limit_count: int = 5,
    tg_token: str = "",
    tg_id: str = "",
) -> None:
    """
    추천 종목 텔레그램 자동 발송.
    """
    token = tg_token or DEFAULT_CONFIG.tg_token
    chat_id = tg_id or DEFAULT_CONFIG.tg_id

    if not token or not chat_id:
        logger.info("텔레그램 미설정 (TG_TOKEN/TG_ID)")
        return

    try:
        # 상위 N개 종목 메시지 구성
        top = df.head(limit_count)
        lines = [f"📊 LDY Pro Trader [{trade_ymd}]"]
        if market_summary:
            lines.append(market_summary)
        lines.append("")

        for i, (_, row) in enumerate(top.iterrows(), 1):
            name = row.get("종목명", row.get("name", ""))
            code = str(row.get("종목코드", "")).zfill(6)
            route = row.get("ROUTE", "")
            score = row.get("DISPLAY_SCORE", row.get("FINAL_SCORE", 0))
            buy = row.get("매수가", row.get("buy_price", 0))
            stop = row.get("손절가", row.get("stop_price", 0))
            tp1 = row.get("TP1", row.get("목표가1", 0))

            line = f"{i}. {name}({code}) {route}"
            line += f"\n   점수:{score:.0f} | 매수:{buy:,.0f} | SL:{stop:,.0f} | TP:{tp1:,.0f}"

            # AI 코멘트
            comment = row.get("AI_COMMENT", "")
            if comment:
                line += f"\n   💡 {comment[:80]}"

            lines.append(line)

        text = "\n".join(lines)

        # 전송
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info(f"✉️ 텔레그램 발송 완료 ({limit_count}종목)")
        else:
            logger.warning(f"텔레그램 발송 실패: {resp.status_code} {resp.text[:200]}")

    except Exception as e:
        logger.warning(f"텔레그램 발송 에러: {e}")
