import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daily_briefing import select_top3
import telegram_sender_v2


def test_daily_briefing_rejects_armed_even_if_flagged_production(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "종목코드": "005930",
                "종목명": "삼성전자",
                "ROUTE": "ARMED",
                "PRODUCTION_BUY": 1,
                "DISPLAY_SCORE": 99,
                "QUALITY_GUARD_SCORE": 99,
            },
            {
                "종목코드": "000660",
                "종목명": "SK하이닉스",
                "ROUTE": "ATTACK",
                "PRODUCTION_BUY": 1,
                "DISPLAY_SCORE": 80,
                "QUALITY_GUARD_SCORE": 80,
            },
        ]
    )

    selected = select_top3(frame, str(tmp_path))

    assert selected["종목코드"].tolist() == ["000660"]


def test_telegram_v2_sends_cash_instead_of_raw_rank(monkeypatch):
    sent = []
    monkeypatch.setattr(
        telegram_sender_v2,
        "_split_and_send",
        lambda message, token, chat_id: sent.append(message) or True,
    )
    frame = pd.DataFrame(
        [
            {
                "종목코드": "005930",
                "종목명": "고득점관찰주",
                "PRODUCTION_BUY": 0,
                "DISPLAY_SCORE": 99,
                "ROUTE": "WAIT",
            }
        ]
    )

    telegram_sender_v2.send_telegram_auto(
        frame, "20260713", tg_token="test", tg_id="test"
    )

    assert len(sent) == 1
    assert "신규매수하지 않습니다" in sent[0]
    assert "고득점관찰주" not in sent[0]


def test_telegram_v2_uses_official_price_column(monkeypatch):
    sent = []
    monkeypatch.setattr(
        telegram_sender_v2,
        "_split_and_send",
        lambda message, token, chat_id: sent.append(message) or True,
    )
    frame = pd.DataFrame(
        [
            {
                "종목코드": "005930",
                "종목명": "공식종목",
                "PRODUCTION_BUY": 1,
                "DISPLAY_SCORE": 88,
                "ROUTE": "ATTACK",
                "추천매수가": 70000,
                "손절가": 66500,
                "추천매도가1": 78000,
            }
        ]
    )

    telegram_sender_v2.send_telegram_auto(
        frame, "20260713", tg_token="test", tg_id="test"
    )

    assert "공식종목" in sent[0]
    assert "70,000" in sent[0]
