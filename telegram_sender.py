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
        # 최종 production 계약만 외부 발송한다. 예전처럼 df.head()를 보내면
        # 관찰 후보가 매수 추천으로 오인된다.
        if "PRODUCTION_BUY" in df.columns:
            _mask = pd.to_numeric(df["PRODUCTION_BUY"], errors="coerce").fillna(0).astype(int) == 1
        elif {"TOP_PICK", "BUY_NOW_ELIGIBLE"}.issubset(df.columns):
            _mask = (
                pd.to_numeric(df["TOP_PICK"], errors="coerce").fillna(0).astype(int).eq(1)
                & pd.to_numeric(df["BUY_NOW_ELIGIBLE"], errors="coerce").fillna(0).astype(int).eq(1)
            )
        else:
            _mask = pd.Series(False, index=df.index)
        top = df[_mask].head(limit_count)
        lines = [f"📊 SwingPicker 오늘의 결정 [{trade_ymd}]"]
        if market_summary:
            lines.append(market_summary)
        lines.append("")

        if top.empty:
            lines.extend([
                "⏸ 오늘은 신규매수하지 않습니다.",
                "최종 품질게이트 통과 종목 0개 — 현금 보유가 공식 결정입니다.",
                "관찰 후보와 높은 원점수는 매수 추천이 아닙니다.",
            ])

        for i, (_, row) in enumerate(top.iterrows(), 1):
            name = row.get("종목명", row.get("name", ""))
            code = str(row.get("종목코드", "")).zfill(6)
            route = row.get("ROUTE", "")
            score = row.get("DISPLAY_SCORE", row.get("FINAL_SCORE", 0))
            buy = row.get(
                "PRODUCTION_ENTRY_PRICE",
                row.get("추천매수가", row.get("매수가", row.get("buy_price", 0))),
            )
            stop = row.get(
                "PRODUCTION_STOP_PRICE", row.get("손절가", row.get("stop_price", 0))
            )
            tp1 = row.get(
                "PRODUCTION_TARGET_PRICE",
                row.get("추천매도가1", row.get("TP1", row.get("목표가1", 0))),
            )

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
            logger.info(f"✉️ 텔레그램 발송 완료 (공식 {len(top)}종목)")
        else:
            logger.warning(f"텔레그램 발송 실패: {resp.status_code} {resp.text[:200]}")

    except Exception as e:
        logger.warning(f"텔레그램 발송 에러: {e}")
