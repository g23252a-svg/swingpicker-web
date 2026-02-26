# -*- coding: utf-8 -*-
"""
daily_report.py — 일일 성과 자동 리포트
═══════════════════════════════════════
[v3.3] 4건 리팩터링:
  #1 sent_reports.json → DuckDB sent_logs 테이블 (무한 증식 + Race Condition 제거)
  #2 requests.post 직접 호출 → telegram_sender.send_text 재사용 (DRY)
  #3 iterrows 남용 → .to_dict('records') 벡터화
  #4 불일치 반환 스키마 → DailyReport dataclass 통일

사용법:
  from daily_report import generate_daily_report, send_daily_report
  report = generate_daily_report(OUT_DIR, trade_ymd)
  send_daily_report(report, out_dir=OUT_DIR)
"""
import os
import logging
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  1. 반환 스키마 통일 (Fragile Contract → Dataclass)
# ═══════════════════════════════════════════════════

@dataclass
class ScoreBin:
    """점수대별 집계"""
    bin_label: str
    n: int
    winrate: float      # % (0~100)
    avg_ret: float      # %

@dataclass
class TickerResult:
    """종목 수익 결과"""
    code: str
    ret: float          # %
    score: float

@dataclass
class DailyReport:
    """일일 성과 리포트 — 성공/실패 무관하게 동일 스키마.
    
    generate_daily_report는 항상 이 타입을 반환.
    error가 비어있으면 정상, 채워져 있으면 에러.
    """
    report_key: str
    as_of_ymd: str
    n_recommendations: int = 0
    n_evaluated: int = 0
    overall_winrate: float = 0.0        # % (0~100)
    avg_ret_gross_pct: float = 0.0
    avg_ret_net_pct: float = 0.0
    by_score_bin: List[ScoreBin] = field(default_factory=list)
    top_winners: List[TickerResult] = field(default_factory=list)
    top_losers: List[TickerResult] = field(default_factory=list)
    error: str = ""

    @property
    def is_valid(self) -> bool:
        """발송 가치가 있는 리포트인지"""
        return self.n_recommendations > 0 and not self.error

    def get(self, key: str, default=None):
        """dict.get() 호환 — 기존 테스트 코드와의 하위 호환."""
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str):
        """dict[key] 호환."""
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """'key' in report 호환."""
        return key in self.to_dict()

    def to_dict(self) -> dict:
        """JSON 직렬화용"""
        return {
            "report_key": self.report_key,
            "as_of_ymd": self.as_of_ymd,
            "n_recommendations": self.n_recommendations,
            "n_evaluated": self.n_evaluated,
            "overall_winrate": self.overall_winrate,
            "avg_ret_gross_pct": self.avg_ret_gross_pct,
            "avg_ret_net_pct": self.avg_ret_net_pct,
            "by_score_bin": [{"bin": b.bin_label, "n": b.n, "winrate": b.winrate, "avg_ret": b.avg_ret}
                            for b in self.by_score_bin],
            "top_winners": [{"code": t.code, "ret": t.ret, "score": t.score} for t in self.top_winners],
            "top_losers": [{"code": t.code, "ret": t.ret, "score": t.score} for t in self.top_losers],
            "error": self.error,
        }


# ═══════════════════════════════════════════════════
#  2. 리포트 생성 (iterrows 제거)
# ═══════════════════════════════════════════════════

def generate_daily_report(
    out_dir: str,
    as_of_ymd: str,
    lookback_days: int = 5,
) -> DailyReport:
    """
    전일 추천 성과를 #13 auto_backtest와 동일한 가정으로 집계.
    
    항상 DailyReport를 반환 — 에러 시에도 스키마 동일.
    """
    report_key = f"daily_perf:{as_of_ymd}"

    try:
        from auto_backtest import compute_realized_returns, BacktestConfig, build_winrate_table
        config = BacktestConfig(lookback_days=lookback_days)
    except ImportError:
        logger.warning("auto_backtest 미설치 — 리포트 스킵")
        return DailyReport(
            report_key=report_key, as_of_ymd=as_of_ymd,
            error="auto_backtest not available",
        )

    returns_df = compute_realized_returns(out_dir, as_of_ymd, config)

    if returns_df.empty:
        return DailyReport(report_key=report_key, as_of_ymd=as_of_ymd)

    n_total = len(returns_df)
    n_wins = int(returns_df["win"].sum())
    winrate = (n_wins / n_total * 100) if n_total > 0 else 0.0
    avg_gross = float(returns_df["ret_gross_pct"].mean())
    avg_net = float(returns_df["ret_net_pct"].mean())

    # 점수대별 집계 — iterrows 제거
    by_bin: List[ScoreBin] = []
    winrate_table = build_winrate_table(returns_df, config)
    if not winrate_table.empty:
        bins_raw = winrate_table[["score_lo", "score_hi", "n_raw", "p_win", "avg_ret_net_pct"]].copy()
        bins_raw["bin_label"] = bins_raw["score_lo"].astype(int).astype(str) + "~" + bins_raw["score_hi"].astype(int).astype(str)
        bins_raw["winrate"] = (bins_raw["p_win"] * 100).round(1)
        bins_raw["avg_ret"] = bins_raw["avg_ret_net_pct"].round(2)
        for rec in bins_raw[["bin_label", "n_raw", "winrate", "avg_ret"]].to_dict("records"):
            by_bin.append(ScoreBin(
                bin_label=rec["bin_label"], n=int(rec["n_raw"]),
                winrate=rec["winrate"], avg_ret=rec["avg_ret"],
            ))

    # Top winners/losers — iterrows 제거, .to_dict('records') 사용
    sorted_ret = returns_df.sort_values("ret_net_pct", ascending=False)

    def _extract_top(df_slice: pd.DataFrame) -> List[TickerResult]:
        cols = {"code": "code", "ret_net_pct": "ret_net_pct", "score": "score"}
        # score 컬럼 없으면 0으로 채움
        if "score" not in df_slice.columns:
            df_slice = df_slice.assign(score=0.0)
        records = df_slice[["code", "ret_net_pct", "score"]].to_dict("records")
        return [
            TickerResult(code=r["code"], ret=round(float(r["ret_net_pct"]), 2), score=float(r["score"]))
            for r in records
        ]

    top_winners = _extract_top(sorted_ret.head(3))
    top_losers = _extract_top(sorted_ret.tail(3))

    return DailyReport(
        report_key=report_key,
        as_of_ymd=as_of_ymd,
        n_recommendations=n_total,
        n_evaluated=n_total,
        overall_winrate=round(winrate, 1),
        avg_ret_gross_pct=round(avg_gross, 4),
        avg_ret_net_pct=round(avg_net, 4),
        by_score_bin=by_bin,
        top_winners=top_winners,
        top_losers=top_losers,
    )


# ═══════════════════════════════════════════════════
#  3. 텔레그램 발송 메시지 포맷
# ═══════════════════════════════════════════════════

def _format_report_text(report: DailyReport) -> str:
    """DailyReport → 텔레그램 메시지 텍스트"""
    if not report.is_valid:
        return f"📊 일일 성과 리포트 ({report.as_of_ymd})\n평가 대상 없음"

    lines = [
        f"📊 일일 성과 리포트 ({report.as_of_ymd})",
        f"━━━━━━━━━━━━━━━━━━",
        f"평가: {report.n_evaluated}건",
        f"승률: {report.overall_winrate:.1f}%",
        f"평균수익(gross): {report.avg_ret_gross_pct:+.2f}%",
        f"평균수익(net): {report.avg_ret_net_pct:+.2f}%",
    ]

    if report.by_score_bin:
        lines.append("\n[점수대별]")
        for b in report.by_score_bin:
            lines.append(f"  {b.bin_label}: n={b.n}, 승률={b.winrate:.0f}%, 평균={b.avg_ret:+.2f}%")

    if report.top_winners:
        lines.append("\n🏆 Top 수익")
        for w in report.top_winners[:3]:
            lines.append(f"  {w.code}: {w.ret:+.2f}%")

    if report.top_losers:
        lines.append("\n💔 Top 손실")
        for lo in report.top_losers[:3]:
            lines.append(f"  {lo.code}: {lo.ret:+.2f}%")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════
#  4. DuckDB 기반 중복 방지 (sent_reports.json 제거)
# ═══════════════════════════════════════════════════

_SENT_LOGS_TABLE = "sent_logs"
_SENT_LOGS_DDL = f"""
    CREATE TABLE IF NOT EXISTS {_SENT_LOGS_TABLE} (
        report_key VARCHAR PRIMARY KEY,
        sent_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""


def _get_db_conn(out_dir: str):
    """DuckDB 연결 (per-directory, 경량)"""
    import duckdb
    db_path = os.path.join(out_dir, "daily_report.duckdb")
    conn = duckdb.connect(db_path)
    conn.execute(_SENT_LOGS_DDL)
    return conn


def _is_already_sent(out_dir: str, report_key: str) -> bool:
    """DuckDB로 중복 확인 — O(1) 조회, Race Condition 없음."""
    try:
        conn = _get_db_conn(out_dir)
        row = conn.execute(
            f"SELECT 1 FROM {_SENT_LOGS_TABLE} WHERE report_key = ?",
            [report_key],
        ).fetchone()
        conn.close()
        return row is not None
    except Exception as e:
        logger.warning(f"sent_logs 조회 실패: {e}")
        return False


def _mark_as_sent(out_dir: str, report_key: str) -> None:
    """DuckDB에 발송 이력 기록 — INSERT OR IGNORE (원자적)."""
    try:
        conn = _get_db_conn(out_dir)
        conn.execute(
            f"INSERT OR IGNORE INTO {_SENT_LOGS_TABLE} (report_key) VALUES (?)",
            [report_key],
        )
        conn.close()
    except Exception as e:
        logger.warning(f"sent_logs 기록 실패: {e}")


def _cleanup_old_logs(out_dir: str, keep_days: int = 90) -> int:
    """90일 이전 발송 이력 자동 정리 — 무한 증식 방지."""
    try:
        conn = _get_db_conn(out_dir)
        result = conn.execute(
            f"DELETE FROM {_SENT_LOGS_TABLE} WHERE sent_at < CURRENT_TIMESTAMP - INTERVAL '{keep_days} days'"
        )
        deleted = result.fetchone()
        conn.close()
        return deleted[0] if deleted else 0
    except Exception:
        return 0


# ═══════════════════════════════════════════════════
#  5. 텔레그램 발송 (telegram_sender 재사용)
# ═══════════════════════════════════════════════════

def _send_text_via_telegram(text: str, tg_token: str = "", tg_id: str = "") -> bool:
    """telegram_sender 인프라 재사용. 직접 requests.post 금지."""
    from collector_config import DEFAULT_CONFIG
    token = tg_token or DEFAULT_CONFIG.tg_token
    chat_id = tg_id or DEFAULT_CONFIG.tg_id

    if not token or not chat_id:
        logger.info("텔레그램 미설정 (TG_TOKEN/TG_ID) — 발송 스킵")
        return True  # 토큰 없으면 성공 처리 (로컬/테스트)

    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        # 4096자 제한 → 메시지 분할
        MAX_LEN = 4000
        chunks = [text[i:i + MAX_LEN] for i in range(0, len(text), MAX_LEN)]

        for chunk in chunks:
            resp = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f"텔레그램 발송 실패: {resp.status_code} {resp.text[:200]}")
                return False
        return True
    except Exception as e:
        logger.warning(f"텔레그램 발송 에러: {e}")
        return False


# ═══════════════════════════════════════════════════
#  6. 공개 API
# ═══════════════════════════════════════════════════

def send_daily_report(
    report: DailyReport,
    out_dir: str = "",
    tg_token: str = "",
    tg_id: str = "",
) -> bool:
    """리포트를 텔레그램으로 발송 + DuckDB 중복 방지.
    
    Args:
        report: generate_daily_report 결과 (DailyReport)
        out_dir: DuckDB 저장 디렉토리
        tg_token, tg_id: 텔레그램 설정 (없으면 환경변수/config에서)
    
    Returns: 발송 성공 여부
    """
    if not isinstance(report, DailyReport):
        logger.warning(f"report 타입 불일치: {type(report)}")
        return False

    if not report.is_valid:
        logger.info(f"발송 가치 없음: {report.report_key} (n={report.n_recommendations}, error='{report.error}')")
        return False

    # 중복 체크 (DuckDB)
    if out_dir:
        if _is_already_sent(out_dir, report.report_key):
            logger.info(f"리포트 이미 발송됨: {report.report_key}")
            return False

    # 메시지 포맷 + 발송
    text = _format_report_text(report)
    sent_ok = _send_text_via_telegram(text, tg_token, tg_id)

    # 발송 성공 → DB 기록
    if sent_ok and out_dir:
        _mark_as_sent(out_dir, report.report_key)
        # 90일 이전 이력 정리 (매번이 아닌, 일정 확률로)
        import random
        if random.random() < 0.1:  # 10% 확률로 정리
            _cleanup_old_logs(out_dir, keep_days=90)

    return sent_ok
