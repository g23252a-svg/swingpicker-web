# -*- coding: utf-8 -*-
"""
daily_briefing.py — 매일 자동 브리핑 생성기
═══════════════════════════════════════════════
[Rule]
  1. collector Action 완료 후 실행
  2. ATTACK / ARMED 종목만 필터
  3. 목표가 달성(CLOSED_TP) 종목 제외
  4. DISPLAY_SCORE 상위 3종목 선정
  5. 토스/블로그/텔레그램 배포 가능한 마크다운 생성

출력:
  - data/briefing_{YYYYMMDD}.md   (일자별 아카이브)
  - data/briefing_latest.md       (최신 고정)
  - data/briefing_{YYYYMMDD}.json (구조화 데이터 — API/웹용)
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
#  1. 목표가 달성 종목 필터
# ═══════════════════════════════════════════════════

def _load_closed_tp_codes(out_dir: str) -> set:
    """positions.json에서 목표가 도달(CLOSED_TP) 종목코드 세트 반환"""
    pos_path = os.path.join(out_dir, "positions.json")
    if not os.path.exists(pos_path):
        return set()
    try:
        with open(pos_path, "r", encoding="utf-8") as f:
            positions = json.load(f)
        # 최근 30일 내 CLOSED_TP 종목만
        codes = set()
        for p in positions:
            if p.get("status") == "CLOSED_TP":
                codes.add(str(p.get("code", "")).zfill(6))
        return codes
    except Exception as e:
        logger.warning(f"positions.json 파싱 실패: {e}")
        return set()


def _load_closed_tp_from_log(out_dir: str, lookback_days: int = 14) -> set:
    """per_trade_log.csv에서 최근 N일 내 익절 종목 제외"""
    log_path = os.path.join(out_dir, "per_trade_log.csv")
    if not os.path.exists(log_path):
        return set()
    try:
        df = pd.read_csv(log_path, dtype={"code": str})
        if "exit_type" in df.columns and "exit_ymd" in df.columns:
            df["exit_ymd"] = pd.to_datetime(df["exit_ymd"], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
            recent_tp = df[(df["exit_type"] == "TP") & (df["exit_ymd"] >= cutoff)]
            return set(recent_tp["code"].astype(str).str.zfill(6))
    except Exception as e:
        logger.warning(f"per_trade_log 파싱 실패: {e}")
    return set()


# ═══════════════════════════════════════════════════
#  2. 상위 3종목 선정
# ═══════════════════════════════════════════════════

def select_top3(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    ATTACK/ARMED 종목 중 목표가 미달성 상위 3종목 선정

    Returns: DataFrame (최대 3행)
    """
    # 1) ATTACK / ARMED만
    df = df.copy()
    df["ROUTE"] = df["ROUTE"].astype(str).str.strip().str.upper()
    active = df[df["ROUTE"].isin(["ATTACK", "ARMED"])].copy()

    if active.empty:
        logger.info("📝 브리핑: ATTACK/ARMED 종목 없음")
        return pd.DataFrame()

    # 2) 목표가 달성 종목 제외
    tp_codes = _load_closed_tp_codes(out_dir) | _load_closed_tp_from_log(out_dir)
    if tp_codes:
        active["종목코드"] = active["종목코드"].astype(str).str.zfill(6)
        before = len(active)
        active = active[~active["종목코드"].isin(tp_codes)]
        excluded = before - len(active)
        if excluded > 0:
            logger.info(f"📝 브리핑: 목표가 달성 {excluded}건 제외")

    if active.empty:
        logger.info("📝 브리핑: 목표가 제외 후 ATTACK/ARMED 종목 없음")
        return pd.DataFrame()

    # 3) DISPLAY_SCORE 상위 3종목
    active["DISPLAY_SCORE"] = pd.to_numeric(active["DISPLAY_SCORE"], errors="coerce").fillna(0)
    top3 = active.nlargest(3, "DISPLAY_SCORE")

    return top3


# ═══════════════════════════════════════════════════
#  3. 마크다운 생성 (토스/블로그 배포용)
# ═══════════════════════════════════════════════════

def _safe_int(val) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _safe_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _route_emoji(route: str) -> str:
    return {"ATTACK": "🚀", "ARMED": "🔫"}.get(route, "👀")


def _route_kr(route: str) -> str:
    return {"ATTACK": "매수 돌입", "ARMED": "매수 대기"}.get(route, route)


def generate_briefing_md(top3: pd.DataFrame, trade_ymd: str, site_url: str = "https://ldyprotrader.com") -> str:
    """토스/블로그 배포용 마크다운 생성"""

    date_display = f"{trade_ymd[:4]}.{trade_ymd[4:6]}.{trade_ymd[6:]}"
    lines = []

    # 헤더
    lines.append(f"🎯 SwingPicker AI 오늘의 Top 3 ({date_display})")
    lines.append("")
    lines.append("AI가 107종목을 분석해서 뽑은 오늘의 핵심 종목입니다.")
    lines.append("")

    # 각 종목
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        code = str(row.get("종목코드", "")).zfill(6)
        name = str(row.get("종목명", code))
        route = str(row.get("ROUTE", "")).upper()
        score = _safe_float(row.get("DISPLAY_SCORE", 0))
        close = _safe_int(row.get("종가", 0))
        entry = _safe_int(row.get("추천매수가", 0))
        stop = _safe_int(row.get("손절가", 0))
        t1 = _safe_int(row.get("추천매도가1", 0))
        est_wr = _safe_float(row.get("EST_WIN_RATE", 0))

        emoji = _route_emoji(route)
        wr_pct = est_wr * 100 if est_wr <= 1 else est_wr

        # 손절/익절 퍼센트
        stop_pct = (stop / entry - 1) * 100 if entry > 0 and stop > 0 else 0
        t1_pct = (t1 / entry - 1) * 100 if entry > 0 and t1 > 0 else 0
        risk = entry - stop if entry > 0 and stop > 0 else 1
        rr = (t1 - entry) / risk if risk > 0 and t1 > 0 else 0

        lines.append(f"{'─' * 30}")
        lines.append(f"{emoji} #{rank}. {name} ({code})")
        lines.append(f"AI 점수: {score:.0f}점 | 신호: {_route_kr(route)} | 승률: {wr_pct:.0f}%")
        lines.append("")

        if entry > 0:
            lines.append(f"  현재가: {close:,}원")
            lines.append(f"  매수가: {entry:,}원")
            lines.append(f"  손절가: {stop:,}원 ({stop_pct:+.1f}%)")
            if t1 > 0:
                lines.append(f"  목표가: {t1:,}원 ({t1_pct:+.1f}%) → 손익비 {rr:.1f}:1")
            lines.append("")

        # 분석 링크
        lines.append(f"  📊 상세 분석 → {site_url}/stock/{code}")
        lines.append("")

    # 푸터
    lines.append(f"{'─' * 30}")
    lines.append("")
    lines.append(f"🔗 전체 107종목 분석: {site_url}")
    lines.append("")
    lines.append("⚠️ 본 자료는 AI 분석 참고 자료이며 투자 권유가 아닙니다.")
    lines.append("투자 판단은 본인 책임이며, 손실이 발생할 수 있습니다.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════
#  4. JSON 구조화 데이터 (웹/API용)
# ═══════════════════════════════════════════════════

def generate_briefing_json(top3: pd.DataFrame, trade_ymd: str, site_url: str = "https://ldyprotrader.com") -> dict:
    """웹 표시/API 응답용 구조화 데이터"""
    stocks = []
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        code = str(row.get("종목코드", "")).zfill(6)
        entry = _safe_int(row.get("추천매수가", 0))
        stop = _safe_int(row.get("손절가", 0))
        t1 = _safe_int(row.get("추천매도가1", 0))
        risk = entry - stop if entry > 0 and stop > 0 else 1

        stocks.append({
            "rank": rank,
            "code": code,
            "name": str(row.get("종목명", code)),
            "route": str(row.get("ROUTE", "")),
            "score": round(_safe_float(row.get("DISPLAY_SCORE", 0)), 1),
            "close": _safe_int(row.get("종가", 0)),
            "entry": entry,
            "stop": stop,
            "target1": t1,
            "target2": _safe_int(row.get("추천매도가2", 0)),
            "est_win_rate": round(_safe_float(row.get("EST_WIN_RATE", 0)), 3),
            "rr": round((t1 - entry) / risk, 1) if risk > 0 and t1 > 0 else 0,
            "sector": str(row.get("업종_대분류", "")),
            "market": str(row.get("시장", "")),
            "url": f"{site_url}/stock/{code}",
        })

    return {
        "trade_date": trade_ymd,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(stocks),
        "stocks": stocks,
    }


# ═══════════════════════════════════════════════════
#  5. 메인 실행 (collector Step 12에서 호출)
# ═══════════════════════════════════════════════════

def generate_daily_briefing(
    out_dir: str,
    trade_ymd: str,
    df: Optional[pd.DataFrame] = None,
    site_url: str = "https://ldyprotrader.com",
) -> Dict:
    """
    매일 자동 브리핑 생성 — collector 파이프라인 Step 12

    Args:
        out_dir: data/ 디렉토리
        trade_ymd: 거래 기준일 (YYYYMMDD)
        df: recommend DataFrame (None이면 CSV에서 로드)
        site_url: 사이트 URL

    Returns:
        {"count": int, "codes": list, "md_path": str, "json_path": str}
    """
    # 데이터 로드
    if df is None:
        csv_path = os.path.join(out_dir, "recommend_latest.csv")
        if not os.path.exists(csv_path):
            logger.warning("❌ recommend_latest.csv 없음 — 브리핑 스킵")
            return {"count": 0, "codes": [], "md_path": "", "json_path": ""}
        df = pd.read_csv(csv_path, dtype={"종목코드": str})

    # 상위 3종목 선정
    top3 = select_top3(df, out_dir)
    if top3.empty:
        logger.info("📝 브리핑 대상 없음 (ATTACK/ARMED 0건)")
        return {"count": 0, "codes": [], "md_path": "", "json_path": ""}

    codes = top3["종목코드"].astype(str).str.zfill(6).tolist()
    names = top3["종목명"].tolist()

    # 마크다운 생성
    md_content = generate_briefing_md(top3, trade_ymd, site_url)
    md_dated = os.path.join(out_dir, f"briefing_{trade_ymd}.md")
    md_latest = os.path.join(out_dir, "briefing_latest.md")

    with open(md_dated, "w", encoding="utf-8") as f:
        f.write(md_content)
    with open(md_latest, "w", encoding="utf-8") as f:
        f.write(md_content)

    # JSON 생성
    json_data = generate_briefing_json(top3, trade_ymd, site_url)
    json_dated = os.path.join(out_dir, f"briefing_{trade_ymd}.json")
    json_latest = os.path.join(out_dir, "briefing_latest.json")

    for p in [json_dated, json_latest]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"📝 일일 브리핑 생성: {len(top3)}종목 [{', '.join(names)}]")
    logger.info(f"   → {md_dated}")

    return {
        "count": len(top3),
        "codes": codes,
        "names": names,
        "md_path": md_dated,
        "json_path": json_dated,
    }


# ═══════════════════════════════════════════════════
#  standalone 실행
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    _dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    _ymd = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y%m%d")
    logging.basicConfig(level=logging.INFO)
    result = generate_daily_briefing(_dir, _ymd)
    if result["count"] > 0:
        print(f"\n✅ 브리핑 생성 완료: {result['count']}종목")
        print(f"   📄 {result['md_path']}")
        with open(result["md_path"], "r", encoding="utf-8") as f:
            print(f"\n{f.read()}")
    else:
        print("❌ 브리핑 대상 종목 없음")
