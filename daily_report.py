# -*- coding: utf-8 -*-
"""
daily_report.py — 일일 성과 자동 리포트
═══════════════════════════════════════
[v14] P5 #21

핵심 원칙:
  1. #13 auto_backtest.compute_realized_returns 재사용 → 수익률 정의 일치
  2. report_key 기반 중복 방지
  3. 텔레그램 발송 (telegram_sender 또는 직접)

사용법:
  collector.main() Step 12:
    from daily_report import generate_daily_report, send_daily_report
    report = generate_daily_report(OUT_DIR, trade_ymd)
    send_daily_report(report)
"""
import os
import json
import logging
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_daily_report(
    out_dir: str,
    as_of_ymd: str,
    lookback_days: int = 5,
) -> Dict:
    """
    전일 추천 성과를 #13 auto_backtest와 동일한 가정으로 집계.

    Returns: {
        report_key, as_of_ymd, n_recommendations, n_evaluated,
        overall_winrate, avg_ret_gross_pct, avg_ret_net_pct,
        by_score_bin: [{bin, n, winrate, avg_ret}],
        top_winners, top_losers,
    }
    """
    report_key = f"daily_perf:{as_of_ymd}"

    try:
        from auto_backtest import compute_realized_returns, BacktestConfig, build_winrate_table
        config = BacktestConfig(lookback_days=lookback_days)
    except ImportError:
        logger.warning("auto_backtest 미설치 — 리포트 스킵")
        return {"report_key": report_key, "n_recommendations": 0, "error": "auto_backtest not available"}

    # #13과 동일한 realized_returns
    returns_df = compute_realized_returns(out_dir, as_of_ymd, config)

    if returns_df.empty:
        return {
            "report_key": report_key,
            "as_of_ymd": as_of_ymd,
            "n_recommendations": 0,
            "n_evaluated": 0,
            "overall_winrate": 0.0,
            "avg_ret_gross_pct": 0.0,
            "avg_ret_net_pct": 0.0,
            "by_score_bin": [],
            "top_winners": [],
            "top_losers": [],
        }

    n_total = len(returns_df)
    n_wins = int(returns_df["win"].sum())
    winrate = n_wins / n_total if n_total > 0 else 0.0
    avg_gross = float(returns_df["ret_gross_pct"].mean())
    avg_net = float(returns_df["ret_net_pct"].mean())

    # 점수대별 집계
    winrate_table = build_winrate_table(returns_df, config)
    by_bin = []
    if not winrate_table.empty:
        for _, row in winrate_table.iterrows():
            by_bin.append({
                "bin": f"{int(row['score_lo'])}~{int(row['score_hi'])}",
                "n": int(row["n_raw"]),
                "winrate": round(float(row["p_win"]) * 100, 1),
                "avg_ret": round(float(row["avg_ret_net_pct"]), 2),
            })

    # Top winners / losers
    sorted_ret = returns_df.sort_values("ret_net_pct", ascending=False)
    top_winners = []
    for _, r in sorted_ret.head(3).iterrows():
        top_winners.append({
            "code": r["code"], "ret": round(float(r["ret_net_pct"]), 2),
            "score": float(r.get("score", 0)),
        })
    top_losers = []
    for _, r in sorted_ret.tail(3).iterrows():
        top_losers.append({
            "code": r["code"], "ret": round(float(r["ret_net_pct"]), 2),
            "score": float(r.get("score", 0)),
        })

    return {
        "report_key": report_key,
        "as_of_ymd": as_of_ymd,
        "n_recommendations": n_total,
        "n_evaluated": n_total,
        "overall_winrate": round(winrate * 100, 1),
        "avg_ret_gross_pct": round(avg_gross, 4),
        "avg_ret_net_pct": round(avg_net, 4),
        "by_score_bin": by_bin,
        "top_winners": top_winners,
        "top_losers": top_losers,
    }


def _format_report_text(report: Dict) -> str:
    """리포트 dict → 텔레그램 메시지 텍스트"""
    if report.get("n_recommendations", 0) == 0:
        return f"📊 일일 성과 리포트 ({report.get('as_of_ymd', '?')})\n평가 대상 없음"

    lines = [
        f"📊 일일 성과 리포트 ({report['as_of_ymd']})",
        f"━━━━━━━━━━━━━━━━━━",
        f"평가: {report['n_evaluated']}건",
        f"승률: {report['overall_winrate']:.1f}%",
        f"평균수익(gross): {report['avg_ret_gross_pct']:+.2f}%",
        f"평균수익(net): {report['avg_ret_net_pct']:+.2f}%",
    ]

    if report.get("by_score_bin"):
        lines.append("\n[점수대별]")
        for b in report["by_score_bin"]:
            lines.append(f"  {b['bin']}: n={b['n']}, 승률={b['winrate']:.0f}%, 평균={b['avg_ret']:+.2f}%")

    if report.get("top_winners"):
        lines.append("\n🏆 Top 수익")
        for w in report["top_winners"][:3]:
            lines.append(f"  {w['code']}: {w['ret']:+.2f}%")

    if report.get("top_losers"):
        lines.append("\n💔 Top 손실")
        for l in report["top_losers"][:3]:
            lines.append(f"  {l['code']}: {l['ret']:+.2f}%")

    return "\n".join(lines)


# ── 중복 방지 저장소 ──
_SENT_REPORTS_FILE = "sent_reports.json"


def _load_sent_keys(out_dir: str) -> set:
    path = os.path.join(out_dir, _SENT_REPORTS_FILE)
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_sent_key(out_dir: str, key: str) -> None:
    keys = _load_sent_keys(out_dir)
    keys.add(key)
    path = os.path.join(out_dir, _SENT_REPORTS_FILE)
    try:
        with open(path, "w") as f:
            json.dump(list(keys), f)
    except Exception:
        pass


def send_daily_report(
    report: Dict,
    out_dir: str = "",
    tg_token: str = "",
    tg_id: str = "",
) -> bool:
    """
    리포트를 텔레그램으로 발송.
    report_key 기반 중복 방지.
    """
    if not report or report.get("n_recommendations", 0) == 0:
        return False

    key = report.get("report_key", "")
    if not key:
        return False

    # 중복 체크
    if out_dir:
        sent = _load_sent_keys(out_dir)
        if key in sent:
            logger.info(f"리포트 이미 발송됨: {key}")
            return False

    text = _format_report_text(report)

    token = tg_token or os.environ.get("TG_TOKEN", "")
    chat_id = tg_id or os.environ.get("TG_ID", "")

    sent_ok = False
    if token and chat_id:
        try:
            import requests
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
            sent_ok = resp.status_code == 200
        except Exception as e:
            logger.warning(f"리포트 텔레그램 실패: {e}")
    else:
        sent_ok = True  # 토큰 없으면 성공으로 처리 (로컬/테스트)

    if sent_ok and out_dir:
        _save_sent_key(out_dir, key)

    return sent_ok
